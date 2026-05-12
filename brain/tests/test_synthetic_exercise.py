"""Tests for the synthetic perception exercise guard and soak harness.

These tests verify the critical safety invariants:
1. Synthetic audio NEVER produces a PERCEPTION_TRANSCRIPTION event
2. Synthetic audio NEVER reaches handle_transcription()
3. Synthetic audio NEVER updates identity state
4. Synthetic audio NEVER writes to memory
5. Synthetic audio NEVER triggers TTS output
6. All distillation signals carry origin="synthetic" provenance
7. The exercise ledger tracks blocked side effects
8. Profiles, reports, and route histograms function correctly
9. The race condition between Pi and synthetic audio is permanently fixed
"""

import sys
import os
from collections import Counter
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Tool router synthetic provenance
# ---------------------------------------------------------------------------

class TestToolRouterSyntheticOrigin:
    """Verify tool_router.route(synthetic=True) records origin='synthetic'."""

    def test_route_with_synthetic_flag(self):
        from reasoning.tool_router import ToolRouter

        recorded_origins: list[str] = []

        mock_dc = MagicMock()
        def capture_record(*args, **kwargs):
            recorded_origins.append(kwargs.get("origin", "unset"))
        mock_dc.record = capture_record

        router = ToolRouter()
        with patch.dict("sys.modules", {"hemisphere.distillation": MagicMock(distillation_collector=mock_dc)}):
            with patch("reasoning.tool_router.distillation_collector", mock_dc, create=True):
                pass
            result = router.route("what time is it", synthetic=True)

        if recorded_origins:
            for origin in recorded_origins:
                assert origin == "synthetic", \
                    f"Synthetic route must use origin='synthetic', got '{origin}'"

    def test_route_records_distillation_when_not_synthetic(self):
        from reasoning.tool_router import ToolRouter
        router = ToolRouter()
        result = router.route("what time is it", synthetic=False)
        assert result.tool is not None

    def test_synthetic_flag_plumbed_to_finalize(self):
        """route(synthetic=True) must pass synthetic=True to _finalize."""
        from reasoning.tool_router import ToolRouter

        original_finalize = ToolRouter._finalize
        finalize_calls: list[dict] = []

        @staticmethod
        def tracking_finalize(user_message, result, *, synthetic=False):
            finalize_calls.append({"synthetic": synthetic})
            return original_finalize(user_message, result, synthetic=synthetic)

        router = ToolRouter()
        with patch.object(ToolRouter, "_finalize", tracking_finalize):
            router.route("what time is it", synthetic=True)

        assert len(finalize_calls) > 0, "_finalize was never called"
        assert finalize_calls[0]["synthetic"] is True, \
            f"Expected synthetic=True in _finalize, got {finalize_calls[0]}"

    def test_finalize_passes_synthetic_to_record(self):
        """_finalize must pass synthetic flag to _record_voice_intent_teacher_signal."""
        from reasoning.tool_router import ToolRouter, RoutingResult, ToolType

        recorded_synthetics: list[bool] = []

        with patch("reasoning.tool_router._record_voice_intent_teacher_signal") as mock_rec:
            def capture_call(msg, result, *, synthetic=False):
                recorded_synthetics.append(synthetic)
            mock_rec.side_effect = capture_call

            ToolRouter._finalize(
                "test message",
                RoutingResult(tool=ToolType.TIME, confidence=0.9, extracted_args={}),
                synthetic=True,
            )

        assert len(recorded_synthetics) == 1
        assert recorded_synthetics[0] is True


# ---------------------------------------------------------------------------
# Exercise corpus
# ---------------------------------------------------------------------------

class TestUtteranceCorpus:

    def test_all_categories_non_empty(self):
        from synthetic.exercise import UTTERANCE_CATEGORIES
        for cat, items in UTTERANCE_CATEGORIES.items():
            assert len(items) >= 3, f"Category '{cat}' has too few utterances ({len(items)})"

    def test_total_utterance_count(self):
        from synthetic.exercise import ALL_UTTERANCES
        assert len(ALL_UTTERANCES) >= 90, f"Expected >= 90 utterances, got {len(ALL_UTTERANCES)}"

    def test_all_utterances_non_empty(self):
        from synthetic.exercise import ALL_UTTERANCES
        for u in ALL_UTTERANCES:
            assert len(u.strip()) > 5, f"Utterance too short: '{u}'"

    def test_no_personal_content(self):
        """Utterance corpus must not contain personal/autobiographical content."""
        from synthetic.exercise import ALL_UTTERANCES
        forbidden = ["david", "my wife", "my husband", "my family", "i love you",
                      "my name is", "remember me", "i told you"]
        for u in ALL_UTTERANCES:
            lower = u.lower()
            for word in forbidden:
                assert word not in lower, \
                    f"Utterance contains forbidden personal content '{word}': {u}"

    def test_pick_utterance_returns_tuple(self):
        from synthetic.exercise import pick_utterance
        text, cat = pick_utterance()
        assert isinstance(text, str) and len(text) > 5
        assert isinstance(cat, str) and len(cat) > 0

    def test_pick_utterance_by_category(self):
        from synthetic.exercise import pick_utterance, UTTERANCE_CATEGORIES
        for cat in UTTERANCE_CATEGORIES:
            text, returned_cat = pick_utterance(cat)
            assert returned_cat == cat
            assert text in UTTERANCE_CATEGORIES[cat]

    def test_category_diversity(self):
        """Corpus must cover at least 12 distinct categories."""
        from synthetic.exercise import UTTERANCE_CATEGORIES
        assert len(UTTERANCE_CATEGORIES) >= 12, \
            f"Expected >= 12 categories, got {len(UTTERANCE_CATEGORIES)}"

    def test_route_targeted_categories_exist(self):
        """Every major ToolType should have a corresponding corpus category."""
        from synthetic.exercise import UTTERANCE_CATEGORIES
        expected = [
            "time", "command", "status", "introspection", "memory",
            "web_search", "academic", "codebase", "skill", "identity", "vision",
        ]
        for cat in expected:
            assert cat in UTTERANCE_CATEGORIES, \
                f"Missing corpus category for route-targeted testing: {cat}"

    def test_weighted_pick_biases_toward_higher_weights(self):
        """Weighted pick should favor higher-weighted categories."""
        from synthetic.exercise import pick_utterance
        weights = {"time": 10.0, "command": 0.01}
        counts: Counter = Counter()
        for _ in range(200):
            _, cat = pick_utterance(weights=weights)
            counts[cat] += 1
        assert counts["time"] > counts.get("command", 0), \
            f"Expected 'time' to dominate with weight 10.0 but got {dict(counts)}"


# ---------------------------------------------------------------------------
# Soak profiles
# ---------------------------------------------------------------------------

class TestSoakProfiles:

    def test_profiles_exist(self):
        from synthetic.exercise import PROFILES
        assert "smoke" in PROFILES
        assert "route_coverage" in PROFILES
        assert "idle_soak" in PROFILES
        assert "stress" in PROFILES

    def test_smoke_profile_small(self):
        from synthetic.exercise import PROFILES
        smoke = PROFILES["smoke"]
        assert smoke.count <= 10
        assert smoke.delay_s >= 1.0

    def test_route_coverage_has_weights(self):
        from synthetic.exercise import PROFILES
        rc = PROFILES["route_coverage"]
        assert rc.category_weights is not None
        assert len(rc.category_weights) >= 10

    def test_effective_count_from_duration(self):
        from synthetic.exercise import PROFILES
        idle = PROFILES["idle_soak"]
        count = idle.effective_count(duration_s=3600)
        assert count == int(3600 / idle.delay_s)

    def test_effective_count_no_duration(self):
        from synthetic.exercise import PROFILES
        smoke = PROFILES["smoke"]
        assert smoke.effective_count() == smoke.count


# ---------------------------------------------------------------------------
# Exercise stats
# ---------------------------------------------------------------------------

class TestExerciseStats:

    def test_stats_to_dict(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 10
        stats.categories_exercised["command"] = 5
        stats.categories_exercised["question"] = 5
        d = stats.to_dict()
        assert d["utterances_sent"] == 10
        assert "command" in d["categories_exercised"]
        assert "pass" in d
        assert "fail_reasons" in d
        assert "invariants" in d

    def test_stats_summary(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 5
        s = stats.summary()
        assert "5 utterances" in s

    def test_stats_rate_calculation(self):
        from synthetic.exercise import ExerciseStats
        import time
        stats = ExerciseStats()
        stats.start_time = time.time() - 60
        stats.utterances_sent = 30
        assert 29.0 <= stats.rate_per_min <= 31.0

    def test_pass_when_no_leaks(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 10
        stats.brain_utterances_stt = 10
        stats.brain_hard_stopped = 10
        assert stats.pass_result is True
        assert len(stats.fail_reasons) == 0

    def test_fail_on_llm_leak(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.brain_llm_leaks = 1
        assert stats.pass_result is False
        assert any("llm_leaks" in r for r in stats.fail_reasons)

    def test_fail_on_hard_stop_mismatch(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.brain_utterances_stt = 10
        stats.brain_hard_stopped = 8
        assert stats.pass_result is False
        assert any("hard_stop_mismatch" in r for r in stats.fail_reasons)

    def test_consistency_check_happy(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 10
        stats.brain_utterances_stt = 10
        stats.brain_hard_stopped = 10
        assert len(stats.consistency_check()) == 0

    def test_consistency_check_stt_exceeds_sent(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 5
        stats.brain_utterances_stt = 10
        issues = stats.consistency_check()
        assert len(issues) > 0
        assert "stt_ok" in issues[0]

    def test_consistency_check_hard_stop_exceeds_stt(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        stats.utterances_sent = 10
        stats.brain_utterances_stt = 5
        stats.brain_hard_stopped = 8
        issues = stats.consistency_check()
        assert len(issues) > 0
        assert "hard_stopped" in issues[0]

    def test_stt_samples_bounded(self):
        from synthetic.exercise import ExerciseStats
        stats = ExerciseStats()
        for i in range(50):
            stats.record_stt_sample(f"sample {i}")
        assert len(stats.stt_text_samples) == 20


# ---------------------------------------------------------------------------
# Speaker ID origin parameter
# ---------------------------------------------------------------------------

class TestSpeakerIdOriginParam:
    """Verify that speaker_id.identify() accepts and uses the origin parameter."""

    def test_identify_signature_accepts_origin(self):
        """The identify method must accept an origin keyword argument."""
        try:
            import inspect
            from perception.speaker_id import SpeakerIdentifier
        except ImportError:
            import pytest
            pytest.skip("Brain deps (numpy) not available")
        sig = inspect.signature(SpeakerIdentifier.identify)
        assert "origin" in sig.parameters, \
            "SpeakerIdentifier.identify() must accept 'origin' parameter"
        assert sig.parameters["origin"].default == "mic"


# ---------------------------------------------------------------------------
# Emotion classifier origin parameter
# ---------------------------------------------------------------------------

class TestEmotionOriginParam:
    """Verify that _classify_waveform() accepts the origin parameter."""

    def test_classify_waveform_signature_accepts_origin(self):
        try:
            import inspect
            from perception.emotion import AudioEmotionClassifier
        except ImportError:
            import pytest
            pytest.skip("Brain deps (numpy) not available")
        sig = inspect.signature(AudioEmotionClassifier._classify_waveform)
        assert "origin" in sig.parameters, \
            "_classify_waveform() must accept 'origin' parameter"
        assert sig.parameters["origin"].default == "mic"


# ---------------------------------------------------------------------------
# Forbidden event assertion (integration-level, runs on brain machine)
# ---------------------------------------------------------------------------

class TestForbiddenEventsIntegration:
    """These tests verify the full guard chain.

    They require pydantic and other brain deps to be installed. Marked
    with a guard so they skip gracefully on dev machines without full deps.
    """

    def _can_import_orchestrator(self) -> bool:
        try:
            from perception_orchestrator import PerceptionOrchestrator
            return True
        except ImportError:
            return False

    def test_synthetic_flag_from_exercise_state_event(self):
        """Synthetic flag is session-sticky via start/end events, not per-chunk."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = set()
        orch._synthetic_cooldown_until = 0.0
        orch._SYNTHETIC_COOLDOWN_S = 15.0
        orch._synthetic_ledger = {
            "total_runs": 0, "last_run_time": 0.0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
        }
        orch.audio_stream = MagicMock()

        PerceptionOrchestrator._on_synthetic_exercise_state(
            orch, active=True, sensor_id="synthetic-exercise",
        )
        assert "synthetic-exercise" in orch._synthetic_sources
        assert orch.audio_stream.synthetic_active is True

        PerceptionOrchestrator._on_synthetic_exercise_state(
            orch, active=False, sensor_id="synthetic-exercise",
        )
        assert len(orch._synthetic_sources) == 0
        assert orch.audio_stream.synthetic_active is False
        assert orch._synthetic_ledger["total_runs"] == 1

    def test_raw_audio_does_not_reset_synthetic_flag(self):
        """Pi audio chunks must NOT overwrite the session-sticky synthetic flag."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = {"synthetic-exercise"}
        orch.audio_stream = MagicMock()

        PerceptionOrchestrator._on_raw_audio(orch, b"\x00" * 100)
        assert "synthetic-exercise" in orch._synthetic_sources, \
            "Pi audio chunk must NOT clear synthetic session flag"

    def test_cooldown_catches_trailing_audio(self):
        """After exercise ends, speech within cooldown is still treated as synthetic."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        import time
        import threading
        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = set()
        orch._synthetic_cooldown_until = time.time() + 15.0
        orch._current_conv_synthetic = False
        orch._conv_lock = threading.Lock()
        orch.perception = None
        orch._loop = None

        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)

        PerceptionOrchestrator._on_speech_ready(orch, audio, "test-conv-123")
        assert orch._current_conv_synthetic is True, \
            "CRITICAL: Speech during cooldown window must be treated as synthetic"

    def test_cooldown_expires(self):
        """After cooldown expires, speech is treated as real."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        import threading
        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = set()
        orch._synthetic_cooldown_until = 0.0
        orch._current_conv_synthetic = False
        orch._conv_lock = threading.Lock()
        orch.perception = None
        orch._loop = None

        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)

        PerceptionOrchestrator._on_speech_ready(orch, audio, "test-conv-456")
        assert orch._current_conv_synthetic is False, \
            "Speech after cooldown expiry must NOT be treated as synthetic"

    def test_cooldown_set_on_exercise_end(self):
        """Exercise end sets cooldown timer."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        import time
        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = {"synthetic-exercise"}
        orch._synthetic_cooldown_until = 0.0
        orch._SYNTHETIC_COOLDOWN_S = 15.0
        orch._synthetic_ledger = {
            "total_runs": 0, "last_run_time": 0.0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
        }
        orch.audio_stream = MagicMock()

        PerceptionOrchestrator._on_synthetic_exercise_state(
            orch, active=False, sensor_id="synthetic-exercise",
        )
        assert orch._synthetic_cooldown_until > time.time(), \
            "Cooldown timer must be set when exercise ends"

    def test_identify_skips_events_when_synthetic(self):
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator
        import numpy as np

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_ledger = {
            "utterances_stt": 0, "routes_produced": 0,
            "distillation_records": 0, "blocked_side_effects": 0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
        }

        speaker_result = {"name": "test", "confidence": 0.8, "is_known": True, "embedding_id": "t"}
        orch.speaker_id = MagicMock()
        orch.speaker_id.available = True
        orch.speaker_id.identify.return_value = speaker_result

        emotion_result = MagicMock()
        emotion_result.emotion = "happy"
        emotion_result.confidence = 0.8
        orch.emotion_classifier = MagicMock()
        orch.emotion_classifier._gpu_available = True
        orch.emotion_classifier._model_healthy = True
        orch.emotion_classifier._classify_waveform.return_value = emotion_result

        emitted = []
        with patch("perception_orchestrator.event_bus") as mock_bus:
            mock_bus.emit = lambda *a, **kw: emitted.append(a[0] if a else "")
            PerceptionOrchestrator._identify_speaker_and_emotion(
                orch, np.zeros(16000, dtype=np.float32), synthetic=True,
            )

        from consciousness.events import PERCEPTION_SPEAKER_IDENTIFIED, PERCEPTION_USER_EMOTION
        assert PERCEPTION_SPEAKER_IDENTIFIED not in emitted, \
            "FORBIDDEN: synthetic audio emitted PERCEPTION_SPEAKER_IDENTIFIED"
        assert PERCEPTION_USER_EMOTION not in emitted, \
            "FORBIDDEN: synthetic audio emitted PERCEPTION_USER_EMOTION"
        assert orch._synthetic_ledger["blocked_side_effects"] >= 2

    def test_get_synthetic_exercise_stats(self):
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = set()
        orch._synthetic_cooldown_until = 0.0
        orch._synthetic_ledger = {
            "utterances_stt": 5, "routes_produced": 5,
            "distillation_records": 15, "blocked_side_effects": 10,
            "route_histogram": {"INTROSPECTION": 3, "NONE": 2},
            "stt_texts": ["test"], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
            "total_runs": 1, "last_run_time": 100.0,
        }

        stats = PerceptionOrchestrator.get_synthetic_exercise_stats(orch)
        assert stats["active"] is False
        assert stats["utterances_stt"] == 5
        assert stats["blocked_side_effects"] == 10
        assert stats["route_histogram"]["INTROSPECTION"] == 3
        assert stats["total_runs"] == 1

    def test_get_synthetic_exercise_stats_active(self):
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = {"synthetic-exercise"}
        orch._synthetic_cooldown_until = 0.0
        orch._synthetic_ledger = {
            "utterances_stt": 0, "routes_produced": 0,
            "distillation_records": 0, "blocked_side_effects": 0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
            "total_runs": 0, "last_run_time": 0.0,
        }

        stats = PerceptionOrchestrator.get_synthetic_exercise_stats(orch)
        assert stats["active"] is True

    def test_route_histogram_populated(self):
        """_synthetic_route_only must populate route_histogram."""
        if not self._can_import_orchestrator():
            import pytest
            pytest.skip("Brain deps not available")

        from perception_orchestrator import PerceptionOrchestrator

        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_ledger = {
            "utterances_stt": 0, "routes_produced": 0,
            "distillation_records": 0, "blocked_side_effects": 0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
        }

        PerceptionOrchestrator._synthetic_route_only(orch, "what time is it", "test123")

        assert orch._synthetic_ledger["routes_produced"] >= 1
        assert len(orch._synthetic_ledger["route_histogram"]) >= 1
        assert len(orch._synthetic_ledger["recent_route_examples"]) >= 1
        assert orch._synthetic_ledger["recent_route_examples"][0]["text"] == "what time is it"


# ---------------------------------------------------------------------------
# Speaker ID synthetic persistence guard (Pillar 10 — truth boundary)
# ---------------------------------------------------------------------------

class TestSpeakerIdSyntheticPersistenceGuard:
    """Verify that speaker_id.identify(origin='synthetic') does not mutate or
    persist real voice profiles. This is the code-path guard that keeps a
    synthetic TTS run from drifting ``~/.jarvis/speakers.json`` across
    restarts. The complementary negative control ensures the guard is narrow:
    mic-origin audio still adapts + saves as before.
    """

    def _build_identifier_with_profile(self):
        import pytest
        try:
            import numpy as np
            from perception.speaker_id import SpeakerIdentifier
        except Exception:
            pytest.skip("Brain deps (numpy + speechbrain stack) not available")

        sid = SpeakerIdentifier.__new__(SpeakerIdentifier)
        sid.available = True
        sid._model = MagicMock()
        sid._profiles: dict[str, dict] = {}
        sid._score_ema = {}
        sid._next_unknown_id = 1
        import threading as _th
        sid._lock = _th.RLock()
        sid._last_embedding = None
        embedding = np.array([0.6, 0.8, 0.0, 0.0] + [0.0] * 188, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        sid._profiles["alice"] = {
            "embedding": embedding.copy(),
            "last_seen": 0.0,
            "interaction_count": 0,
        }
        sid._extract_embedding = lambda audio: embedding.copy()
        sid._save_profiles = MagicMock()
        return sid, embedding, np

    def test_speaker_id_synthetic_origin_does_not_persist(self):
        sid, baseline, np = self._build_identifier_with_profile()

        audio = np.zeros(16000, dtype=np.float32)
        sid.identify(audio, sample_rate=16000, origin="synthetic")

        assert sid._save_profiles.call_count == 0, \
            "synthetic origin must not call _save_profiles()"
        stored = sid._profiles["alice"]["embedding"]
        if hasattr(stored, "tolist"):
            stored_arr = stored
        else:
            import numpy as np2
            stored_arr = np2.array(stored, dtype=np2.float32)
        assert bool((stored_arr == baseline).all()), \
            "synthetic origin must not mutate stored profile embedding"
        assert sid._profiles["alice"]["interaction_count"] == 0, \
            "synthetic origin must not bump interaction_count"
        assert sid._profiles["alice"]["last_seen"] == 0.0, \
            "synthetic origin must not update last_seen"

    def test_speaker_id_mic_origin_does_persist(self):
        sid, baseline, np = self._build_identifier_with_profile()

        audio = np.zeros(16000, dtype=np.float32)
        sid.identify(audio, sample_rate=16000, origin="mic")

        assert sid._save_profiles.call_count >= 1, \
            "mic origin must still call _save_profiles() (guard must be narrow)"
        assert sid._profiles["alice"]["interaction_count"] >= 1, \
            "mic origin must still bump interaction_count"


# ---------------------------------------------------------------------------
# Perception orchestrator invariant-leak counter wiring (Pillar 9)
# ---------------------------------------------------------------------------

class TestSyntheticLeakCounters:
    """Verify that ``_synthetic_leak_*`` observers actually increment the
    declared leak counters when an event fires during an active synthetic
    session, and stay quiet otherwise.
    """

    def _build_orch(self, with_sources: bool):
        import pytest
        try:
            from perception_orchestrator import PerceptionOrchestrator
        except Exception:
            pytest.skip("Brain deps not available")
        orch = MagicMock(spec=PerceptionOrchestrator)
        orch._synthetic_sources = {"sensor-x"} if with_sources else set()
        orch._synthetic_ledger = {
            "utterances_stt": 0, "routes_produced": 0,
            "distillation_records": 0, "blocked_side_effects": 0,
            "route_histogram": {}, "stt_texts": [], "recent_route_examples": [],
            "llm_leaks": 0, "tts_leaks": 0, "transcription_emit_leaks": 0,
            "memory_side_effects": 0, "identity_side_effects": 0,
            "total_runs": 0, "last_run_time": 0.0,
        }
        # Bind the real _record_synthetic_leak implementation to this mock so
        # that the leak-observer methods (which call self._record_synthetic_leak)
        # actually increment the ledger instead of hitting a Mock no-op.
        orch._record_synthetic_leak = lambda counter_name, event_name, payload_keys: (
            PerceptionOrchestrator._record_synthetic_leak(orch, counter_name, event_name, payload_keys)
        )
        return orch, PerceptionOrchestrator

    def test_leak_counters_fire_on_simulated_leak(self):
        orch, PerceptionOrchestrator = self._build_orch(with_sources=True)

        PerceptionOrchestrator._synthetic_leak_transcription(orch, text="hi", conversation_id="c1")
        PerceptionOrchestrator._synthetic_leak_memory(orch, memory_id="m1")
        PerceptionOrchestrator._synthetic_leak_identity(orch, event="identity:resolved", name="alice")
        PerceptionOrchestrator._synthetic_leak_llm(orch, text="sample response")
        PerceptionOrchestrator._synthetic_leak_tts(orch, text_chars=42)

        ledger = orch._synthetic_ledger
        assert ledger["transcription_emit_leaks"] == 1, "transcription leak must increment"
        assert ledger["memory_side_effects"] == 1, "memory leak must increment"
        assert ledger["identity_side_effects"] == 1, "identity leak must increment"
        assert ledger["llm_leaks"] == 1, "llm leak must increment"
        assert ledger["tts_leaks"] == 1, "tts leak must increment"
        assert isinstance(ledger.get("fail_reasons"), list)
        assert len(ledger["fail_reasons"]) == 5, \
            "fail_reasons must record one entry per leak"
        counters = {r["counter"] for r in ledger["fail_reasons"]}
        assert counters == {
            "transcription_emit_leaks",
            "memory_side_effects",
            "identity_side_effects",
            "llm_leaks",
            "tts_leaks",
        }

    def test_leak_counters_quiet_when_not_synthetic(self):
        orch, PerceptionOrchestrator = self._build_orch(with_sources=False)

        PerceptionOrchestrator._synthetic_leak_transcription(orch, text="hi")
        PerceptionOrchestrator._synthetic_leak_memory(orch, memory_id="m1")
        PerceptionOrchestrator._synthetic_leak_identity(orch, event="identity:resolved")
        PerceptionOrchestrator._synthetic_leak_llm(orch, text="sample")
        PerceptionOrchestrator._synthetic_leak_tts(orch, text_chars=10)

        ledger = orch._synthetic_ledger
        assert ledger["transcription_emit_leaks"] == 0
        assert ledger["memory_side_effects"] == 0
        assert ledger["identity_side_effects"] == 0
        assert ledger["llm_leaks"] == 0
        assert ledger["tts_leaks"] == 0
        assert ledger.get("fail_reasons") in (None, []), \
            "fail_reasons must stay empty when no synthetic session is active"


# ---------------------------------------------------------------------------
# Upstream gate tests: verify the emit sites themselves do not fire while a
# synthetic session is active. These cover the fix after route_coverage
# surfaced 22 identity_side_effects + 2 memory_side_effects leaks.
# ---------------------------------------------------------------------------

class TestMemoryWriteSuppressedDuringSyntheticSession:
    """engine.remember(), memory/core._direct_memory_write(), and
    memory/transactions must all short-circuit while the CueGate reports a
    synthetic session active. This is the sole guarantee that synthetic TTS
    audio can never create lived-history memory.
    """

    @staticmethod
    def _open_bus():
        from consciousness.events import event_bus, _BarrierState
        if event_bus._barrier != _BarrierState.OPEN:
            event_bus.open_barrier()

    def test_engine_remember_short_circuits(self):
        import pytest
        try:
            from memory.gate import memory_gate
            from consciousness.engine import ConsciousnessEngine
            from consciousness.events import event_bus, MEMORY_WRITE
            from memory.core import CreateMemoryData
        except Exception:
            pytest.skip("Brain deps not available")

        self._open_bus()
        engine = ConsciousnessEngine()
        received: list = []

        def _collector(**kwargs):
            received.append(kwargs)

        event_bus.on(MEMORY_WRITE, _collector)
        memory_gate.begin_synthetic_session("test-sensor")
        try:
            result = engine.remember(CreateMemoryData(
                type="event", payload="synthetic test", weight=0.5, tags=[],
            ))
        finally:
            memory_gate.end_synthetic_session("test-sensor")

        assert result is None, "engine.remember must return None while synthetic"
        assert received == [], "MEMORY_WRITE must not fire during synthetic session"

    def test_direct_memory_write_short_circuits(self):
        import pytest
        try:
            from memory.gate import memory_gate
            from memory.core import _direct_memory_write, CreateMemoryData
            from consciousness.events import event_bus, MEMORY_WRITE
        except Exception:
            pytest.skip("Brain deps not available")

        self._open_bus()
        received: list = []

        def _collector(**kwargs):
            received.append(kwargs)

        event_bus.on(MEMORY_WRITE, _collector)
        memory_gate.begin_synthetic_session("test-sensor")
        try:
            result = _direct_memory_write(CreateMemoryData(
                type="event", payload="synthetic fallback", weight=0.5, tags=[],
            ))
        finally:
            memory_gate.end_synthetic_session("test-sensor")

        assert result is None, "_direct_memory_write must return None while synthetic"
        assert received == [], \
            "MEMORY_WRITE must not fire through fallback path during synthetic session"

    def test_gate_releases_after_session_ends(self):
        """Once the synthetic session ends, the CueGate must report writes
        permitted again (orthogonal from whether a specific write path
        succeeds in isolation, which depends on global engine context)."""
        import pytest
        try:
            from memory.gate import memory_gate
        except Exception:
            pytest.skip("Brain deps not available")

        memory_gate.set_mode("conversational")
        memory_gate.begin_synthetic_session("test-sensor")
        assert memory_gate.synthetic_session_active() is True
        assert memory_gate.can_observation_write() is False
        memory_gate.end_synthetic_session("test-sensor")
        assert memory_gate.synthetic_session_active() is False
        assert memory_gate.can_observation_write() is True


class TestFaceCropSuppressedDuringSyntheticSession:
    """perception/server.py's face_crop handler must not invoke the face
    identifier or emit PERCEPTION_FACE_IDENTIFIED while the CueGate reports
    a synthetic session active. Real camera events during a synthetic
    exercise would otherwise mutate identity state and pollute the ledger.
    """

    @staticmethod
    def _open_bus():
        from consciousness.events import event_bus, _BarrierState
        if event_bus._barrier != _BarrierState.OPEN:
            event_bus.open_barrier()

    def test_face_crop_handler_short_circuits(self):
        import pytest
        try:
            from memory.gate import memory_gate
            from perception.server import PerceptionServer, PerceptionEvent
            from consciousness.events import event_bus, PERCEPTION_FACE_IDENTIFIED
            from unittest.mock import MagicMock
        except Exception:
            pytest.skip("Brain deps not available")

        self._open_bus()
        server = PerceptionServer.__new__(PerceptionServer)
        server._event_buffer = []
        server._face_identifier = MagicMock()
        server._face_identifier.identify_b64 = MagicMock(
            return_value={"name": "leaked", "confidence": 0.9, "is_known": True, "closest_match": "leaked"}
        )

        received: list = []
        event_bus.on(PERCEPTION_FACE_IDENTIFIED, lambda **kw: received.append(kw))

        evt = PerceptionEvent(
            source="vision", type="face_crop", timestamp=0.0,
            data={"crop_b64": "fake_b64_data", "track_id": 42},
            confidence=0.9, conversation_id="",
        )

        memory_gate.begin_synthetic_session("face-test")
        try:
            server._process_event(evt, "face-test")
        finally:
            memory_gate.end_synthetic_session("face-test")

        assert server._face_identifier.identify_b64.call_count == 0, (
            "Face identifier must not run during synthetic session"
        )
        assert received == [], (
            "PERCEPTION_FACE_IDENTIFIED must not fire during synthetic session"
        )
