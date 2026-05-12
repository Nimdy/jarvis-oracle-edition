"""Tests for Wave 3 identity hardening: tentative bridge, multi-speaker
awareness, resolution basis, and unknown identity continuity.
"""

from __future__ import annotations

import time
import pytest

from perception.identity_fusion import (
    IdentityFusion,
    IdentitySignal,
    IdentitySnapshot,
    STALE_VOICE_S,
    STALE_FACE_S,
    PERSIST_MAX_S,
    TENTATIVE_MAX_AGE_S,
    MULTI_PERSON_VOICE_THRESHOLD,
    COLD_START_BOOST_WINDOW_S,
    VOICE_SHORT_GAP_HOLD_S,
    CONVERSATION_BOUNDARY_GRACE_S,
    VOICE_DROP_GRACE_S,
)
from identity.types import CONFIDENCE_THRESHOLDS


@pytest.fixture()
def fusion():
    f = IdentityFusion()
    f._user_present = True
    return f


def _inject_voice(fusion: IdentityFusion, name: str, confidence: float,
                   is_known: bool = True, age: float = 0.0):
    fusion._voice = IdentitySignal(
        name=name if is_known else "unknown",
        confidence=confidence,
        is_known=is_known,
        timestamp=time.time() - age,
        source="voice",
    )


def _inject_face(fusion: IdentityFusion, name: str, confidence: float,
                  is_known: bool = True, age: float = 0.0):
    fusion._face = IdentitySignal(
        name=name if is_known else "unknown",
        confidence=confidence,
        is_known=is_known,
        timestamp=time.time() - age,
        source="face",
    )


# ==========================================================================
# Resolution basis
# ==========================================================================


class TestResolutionBasis:
    def test_verified_both_basis(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._resolution_basis == "voice_face_agree"

    def test_voice_only_basis(self, fusion):
        _inject_voice(fusion, "David", 0.70)
        _inject_face(fusion, "unknown", 0.1, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._resolution_basis == "voice_only"

    def test_face_only_basis(self, fusion):
        _inject_voice(fusion, "unknown", 0.1, is_known=False, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._resolution_basis == "face_only"

    def test_conflict_basis(self, fusion):
        _inject_voice(fusion, "David", 0.70)
        _inject_face(fusion, "Alice", 0.80)
        fusion._resolve()
        assert fusion._resolution_basis == "conflicted"

    def test_no_signal_basis(self, fusion):
        fusion._resolve()
        assert fusion._resolution_basis == "no_signal"

    def test_persisted_basis(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._last_known is not None

        _inject_voice(fusion, "unknown", 0.0, is_known=False, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._resolution_basis == "persisted"

    def test_basis_in_get_status(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        status = fusion._get_status_locked()
        assert status["resolution_basis"] == "voice_face_agree"


# ==========================================================================
# Voice trust state
# ==========================================================================


class TestVoiceTrustState:
    def test_verified_both_trusted(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._voice_trust_state == "trusted"

    def test_conflict_state(self, fusion):
        _inject_voice(fusion, "David", 0.70)
        _inject_face(fusion, "Alice", 0.80)
        fusion._resolve()
        assert fusion._voice_trust_state == "conflicted"

    def test_face_only_tentative(self, fusion):
        _inject_voice(fusion, "unknown", 0.1, is_known=False, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._voice_trust_state == "tentative"

    def test_no_signal_unknown(self, fusion):
        fusion._resolve()
        assert fusion._voice_trust_state == "unknown"

    def test_trust_state_in_status(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        status = fusion._get_status_locked()
        assert status["voice_trust_state"] == "trusted"
        assert "trust_reason" in status

    def test_trust_reason_populated(self, fusion):
        _inject_voice(fusion, "David", 0.70)
        _inject_face(fusion, "Alice", 0.80)
        fusion._resolve()
        assert "David" in fusion._trust_reason or "Alice" in fusion._trust_reason


# ==========================================================================
# Multi-speaker suppression
# ==========================================================================


class TestMultiSpeakerSuppression:
    def test_voice_only_suppressed_multi_person(self, fusion):
        """Voice-only resolution with low-ish confidence should be suppressed
        when multiple persons are visible."""
        fusion.set_visible_persons(2)
        _inject_voice(fusion, "David", 0.52)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._last_resolved.method == "suppressed_multi_person"
        assert fusion._last_resolved.is_known is False
        assert fusion._voice_trust_state == "degraded"

    def test_voice_only_high_conf_allowed_multi_person(self, fusion):
        """Strong voice confidence should still resolve voice_only even with
        multiple visible persons."""
        fusion.set_visible_persons(2)
        _inject_voice(fusion, "David", 0.70)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._last_resolved.method == "voice_only"
        assert fusion._last_resolved.is_known is True

    def test_verified_both_unaffected_by_multi_person(self, fusion):
        """verified_both should not be affected by multi-person presence."""
        fusion.set_visible_persons(3)
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._last_resolved.method == "verified_both"

    def test_face_only_unaffected_by_multi_person(self, fusion):
        """Face-only resolution doesn't use multi-person suppression."""
        fusion.set_visible_persons(3)
        _inject_voice(fusion, "unknown", 0.1, is_known=False, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "David", 0.85)
        fusion._resolve()
        assert fusion._last_resolved.method == "face_only"

    def test_suppression_reflected_in_status(self, fusion):
        fusion.set_visible_persons(2)
        _inject_voice(fusion, "David", 0.52)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        status = fusion._get_status_locked()
        assert status["visible_person_count"] == 2
        assert status["multi_person_suppression_active"] is True

    def test_single_person_no_suppression(self, fusion):
        fusion.set_visible_persons(1)
        _inject_voice(fusion, "David", 0.52)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._last_resolved.method == "voice_only"


# ==========================================================================
# Tentative bridge
# ==========================================================================


class TestTentativeBridge:
    def _setup_tentative(self, fusion, name="David"):
        """Prime the fusion with enough tentative evidence to reach
        tentative_match state."""
        fusion._recognition_state = "tentative_match"
        fusion._tentative_name = name
        fusion._tentative_confidence = 0.45
        fusion._presence_start = time.time()

    def test_tentative_bridge_fires_when_no_signal(self, fusion):
        """When in tentative_match with no strong signals, the tentative bridge
        should produce a tentative_bridge resolution."""
        self._setup_tentative(fusion)
        fusion._resolve()
        assert fusion._last_resolved.method == "tentative_bridge"
        assert fusion._last_resolved.name == "David"
        assert fusion._last_resolved.is_known is True
        assert fusion._threshold_assist_active is True

    def test_tentative_bridge_blocked_by_multi_person(self, fusion):
        """Multi-person scene should block tentative bridge."""
        self._setup_tentative(fusion)
        fusion.set_visible_persons(2)
        fusion._resolve()
        assert fusion._last_resolved.method == "no_signal"
        assert fusion._threshold_assist_active is False

    def test_tentative_bridge_expires_with_age(self, fusion):
        """Tentative bridge should not fire when presence started too long ago."""
        self._setup_tentative(fusion)
        fusion._presence_start = time.time() - TENTATIVE_MAX_AGE_S - COLD_START_BOOST_WINDOW_S - 10
        fusion._resolve()
        assert fusion._last_resolved.method == "no_signal"

    def test_tentative_bridge_defers_to_persistence(self, fusion):
        """Persistence window should take priority over tentative bridge."""
        fusion._last_known = IdentitySnapshot(
            name="Alice", confidence=0.80,
            method="verified_both", captured_at=time.time() - 5,
        )
        self._setup_tentative(fusion, "David")
        fusion._resolve()
        assert fusion._last_resolved.method == "persisted"
        assert fusion._last_resolved.name == "Alice"

    def test_tentative_bridge_confidence_conservative(self, fusion):
        """Bridge confidence should be conservative (80% of tentative, min 0.35)."""
        self._setup_tentative(fusion)
        fusion._tentative_confidence = 0.50
        fusion._resolve()
        assert fusion._last_resolved.confidence == pytest.approx(0.40, abs=0.01)

    def test_tentative_bridge_confidence_floor(self, fusion):
        """Bridge confidence should not drop below 0.35."""
        self._setup_tentative(fusion)
        fusion._tentative_confidence = 0.30
        fusion._resolve()
        assert fusion._last_resolved.confidence >= 0.35

    def test_tentative_bridge_shown_in_status(self, fusion):
        self._setup_tentative(fusion)
        fusion._resolve()
        status = fusion._get_status_locked()
        assert status["threshold_assist_active"] is True
        assert status["threshold_assist_name"] == "David"

    def test_tentative_bridge_not_eligible_without_tentative_state(self, fusion):
        """No bridge when recognition state is not tentative_match."""
        fusion._recognition_state = "unknown_present"
        fusion._tentative_name = "David"
        fusion._tentative_confidence = 0.45
        fusion._presence_start = time.time()
        fusion._resolve()
        assert fusion._last_resolved.method == "no_signal"


# ==========================================================================
# Unknown voice continuity
# ==========================================================================


class TestUnknownVoiceContinuity:
    def test_record_unknown_voice_on_subthreshold(self, fusion):
        """Sub-threshold voice events should be recorded for later grounding."""
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="David")
        assert len(fusion._unknown_voice_events) == 1
        evt = fusion._unknown_voice_events[0]
        assert evt["closest_match"] == "David"
        assert evt["confidence"] == 0.35

    def test_unknown_voice_not_recorded_when_known(self, fusion):
        """Known voice signals should not generate unknown voice events."""
        initial_count = len(fusion._unknown_voice_events)
        fusion._on_voice_locked("David", 0.80, True, closest_match="David")
        assert len(fusion._unknown_voice_events) == initial_count

    def test_unknown_voice_ring_buffer_limit(self, fusion):
        """Unknown voice events should be bounded."""
        for i in range(25):
            fusion._on_voice_locked(f"speaker_{i}", 0.30, False, closest_match="David")
        assert len(fusion._unknown_voice_events) == fusion._UNKNOWN_VOICE_MAX

    def test_get_unknown_voice_events_age_filter(self, fusion):
        old_evt = {
            "timestamp": time.time() - 600,
            "confidence": 0.30,
            "closest_match": "David",
            "visible_persons": 0,
            "face_name": "",
            "reason": "voice_below_threshold",
        }
        fusion._unknown_voice_events.append(old_evt)
        fusion._on_voice_locked("speaker_2", 0.35, False, closest_match="Alice")
        recent = fusion.get_unknown_voice_events(max_age_s=300.0)
        assert len(recent) == 1
        assert recent[0]["closest_match"] == "Alice"

    def test_unknown_voice_count_in_status(self, fusion):
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="David")
        status = fusion._get_status_locked()
        assert status["unknown_voice_count"] == 1

    def test_unknown_voice_records_face_context(self, fusion):
        """Unknown voice event should include current face context."""
        _inject_face(fusion, "Alice", 0.80)
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="David")
        evt = fusion._unknown_voice_events[-1]
        assert evt["face_name"] == "Alice"

    def test_unknown_voice_records_visible_persons(self, fusion):
        fusion.set_visible_persons(3)
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="")
        evt = fusion._unknown_voice_events[-1]
        assert evt["visible_persons"] == 3


# ==========================================================================
# Closest-match accumulation
# ==========================================================================


class TestClosestMatchAccumulation:
    def test_voice_accumulates_under_closest_match(self, fusion):
        """Evidence should accumulate under the real profile name, not speaker_N."""
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="David")
        assert "David" in fusion._voice_candidates

    def test_face_accumulates_under_closest_match(self, fusion):
        fusion._on_face_locked("face_1", 0.40, False, closest_match="David")
        assert "David" in fusion._face_candidates

    def test_fallback_to_event_name_without_closest(self, fusion):
        """When closest_match is empty, accumulate under the event name."""
        fusion._on_voice_locked("speaker_1", 0.35, False, closest_match="")
        assert "speaker_1" in fusion._voice_candidates


# ==========================================================================
# Voice short-gap smoothing
# ==========================================================================


class TestVoiceShortGapSmoothing:
    def test_short_gap_same_candidate_keeps_identity(self, fusion):
        fusion._on_voice_locked("David", 0.72, True, closest_match="David")
        assert fusion._last_resolved.method == "voice_only"

        fusion._on_voice_locked("speaker_1", 0.36, False, closest_match="David")

        assert fusion._last_resolved.method == "voice_only"
        assert fusion._last_resolved.name == "David"
        assert fusion._last_resolved.is_known is True
        assert fusion._voice_gap_smoothed is True
        assert fusion._get_status_locked()["voice_gap_smoothed"] is True

    def test_short_gap_smoothing_requires_matching_candidate(self, fusion):
        fusion._on_voice_locked("David", 0.72, True, closest_match="David")
        fusion._last_known = None  # isolate smoothing from persistence fallback

        fusion._on_voice_locked("speaker_1", 0.36, False, closest_match="Alice")

        assert fusion._last_resolved.method == "no_signal"
        assert fusion._last_resolved.is_known is False
        assert fusion._voice_gap_smoothed is False

    def test_short_gap_smoothing_expires_after_hold_window(self, fusion):
        fusion._on_voice_locked("David", 0.72, True, closest_match="David")
        fusion._last_known = None
        fusion._voice.timestamp = time.time() - (VOICE_SHORT_GAP_HOLD_S + 0.5)

        fusion._on_voice_locked("speaker_1", 0.36, False, closest_match="David")

        assert fusion._last_resolved.method == "no_signal"
        assert fusion._voice_gap_smoothed is False

    def test_short_gap_smoothing_blocked_for_multi_person_low_conf(self, fusion):
        fusion.set_visible_persons(2)
        fusion._on_voice_locked("David", 0.72, True, closest_match="David")
        fusion._last_known = None

        fusion._on_voice_locked(
            "speaker_1",
            MULTI_PERSON_VOICE_THRESHOLD - 0.02,
            False,
            closest_match="David",
        )

        assert fusion._last_resolved.method == "no_signal"
        assert fusion._voice_gap_smoothed is False

    def test_short_gap_smoothing_blocked_by_conflicting_fresh_face(self, fusion):
        fusion._on_voice_locked("David", 0.72, True, closest_match="David")
        fusion._on_face_locked("Alice", 0.90, True, closest_match="Alice")
        fusion._last_known = None

        fusion._on_voice_locked("speaker_1", 0.36, False, closest_match="David")

        assert fusion._last_resolved.method == "face_only"
        assert fusion._last_resolved.name == "Alice"
        assert fusion._voice_gap_smoothed is False


# ==========================================================================
# Conversation-boundary persistence
# ==========================================================================


class TestConversationBoundaryPersistence:
    def test_playback_complete_activates_boundary_grace_for_known_identity(self, fusion):
        fusion._on_voice_locked("David", 0.74, True, closest_match="David")
        fusion._on_playback_complete()
        status = fusion._get_status_locked()
        assert status["conversation_boundary_grace_active"] is True
        assert status["conversation_boundary_grace_remaining_s"] > 0
        assert status["conversation_boundary_grace_reason"] == "post_playback"

    def test_wake_word_keeps_identity_during_boundary_grace_without_face(self, fusion):
        fusion._on_voice_locked("David", 0.74, True, closest_match="David")
        fusion._on_playback_complete()

        _inject_voice(fusion, "David", 0.74, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._last_resolved.method == "persisted"

        fusion._on_wake_word_locked()
        assert fusion._last_known is not None
        assert fusion._last_known.name == "David"
        assert fusion._last_resolved.method == "persisted"

    def test_boundary_grace_expires_and_wake_word_clears_without_face(self, fusion):
        fusion._on_voice_locked("David", 0.74, True, closest_match="David")
        fusion._on_playback_complete()
        fusion._conversation_boundary_until = time.time() - 0.1

        _inject_voice(fusion, "David", 0.74, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        assert fusion._last_resolved.method == "persisted"

        fusion._on_wake_word_locked()
        assert fusion._last_known is None
        assert fusion._last_resolved.method == "no_signal"

    def test_boundary_grace_blocked_for_multi_person_scene(self, fusion):
        fusion._on_voice_locked("David", 0.74, True, closest_match="David")
        fusion._on_playback_complete()
        fusion.set_visible_persons(2)

        _inject_voice(fusion, "David", 0.74, age=STALE_VOICE_S + 1)
        _inject_face(fusion, "unknown", 0.0, is_known=False, age=STALE_FACE_S + 1)
        fusion._resolve()
        fusion._on_wake_word_locked()

        assert fusion._last_known is None
        assert fusion._last_resolved.method == "no_signal"

    def test_boundary_grace_window_constant_is_short(self):
        assert CONVERSATION_BOUNDARY_GRACE_S <= 20.0


# ==========================================================================
# Graceful degradation when voice drops
# ==========================================================================


class TestVoiceDropGracefulDegradation:
    def _prime_recent_identity(self, fusion):
        _inject_voice(fusion, "David", 0.80)
        _inject_face(fusion, "David", 0.88)
        fusion._resolve()
        assert fusion._last_resolved.is_known is True
        assert fusion._last_resolved.name == "David"

    def test_voice_drop_grace_keeps_identity_with_face_confirmation(self, fusion):
        self._prime_recent_identity(fusion)

        _inject_voice(fusion, "unknown", 0.12, is_known=False)
        _inject_face(fusion, "David", 0.86)
        fusion._resolve()

        assert fusion._last_resolved.method == "face_voice_drop_grace"
        assert fusion._last_resolved.name == "David"
        assert fusion._last_resolved.is_known is True
        assert fusion._voice_trust_state == "degraded"

    def test_voice_drop_grace_not_used_without_recent_identity(self, fusion):
        _inject_voice(fusion, "unknown", 0.12, is_known=False)
        _inject_face(fusion, "David", 0.90)
        fusion._resolve()

        assert fusion._last_resolved.method == "face_present_voice_unknown"
        assert fusion._last_resolved.is_known is False

    def test_voice_drop_grace_expires_after_window(self, fusion):
        self._prime_recent_identity(fusion)
        fusion._voice_drop_recent_ts = time.time() - (VOICE_DROP_GRACE_S + 1.0)

        _inject_voice(fusion, "unknown", 0.12, is_known=False)
        _inject_face(fusion, "David", 0.90)
        fusion._resolve()

        assert fusion._last_resolved.method == "face_present_voice_unknown"
        assert fusion._last_resolved.is_known is False

    def test_voice_drop_grace_blocked_for_multi_person(self, fusion):
        self._prime_recent_identity(fusion)
        fusion.set_visible_persons(2)

        _inject_voice(fusion, "unknown", 0.12, is_known=False)
        _inject_face(fusion, "David", 0.90)
        fusion._resolve()

        assert fusion._last_resolved.method == "face_present_voice_unknown"
        assert fusion._last_resolved.is_known is False

    def test_voice_drop_grace_requires_min_active_voice_confidence(self, fusion):
        self._prime_recent_identity(fusion)

        _inject_voice(fusion, "unknown", 0.01, is_known=False)
        _inject_face(fusion, "David", 0.90)
        fusion._resolve()

        assert fusion._last_resolved.method == "face_present_voice_unknown"
        assert fusion._last_resolved.is_known is False
