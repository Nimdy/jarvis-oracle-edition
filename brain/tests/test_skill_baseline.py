"""Tests for the SkillBaseline & Validation system.

Validates the Shadow Copy pattern: baseline → train → compare → deploy-if-better.
"""

import pytest
from skills.baseline import (
    SkillBaseline,
    SkillValidation,
    compare_metrics,
    build_validation_evidence,
    capture_speaker_id_metrics,
    capture_emotion_metrics,
    capture_cognitive_metrics,
    capture_autonomy_metrics,
    capture_generic_perceptual_metrics,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MINIMUM_IMPROVEMENT_THRESHOLD,
)


class TestSkillBaseline:
    def test_round_trip_serialization(self):
        bl = SkillBaseline(skill_id="speaker_identification", metrics={"accuracy": 0.85, "confidence": 0.72})
        d = bl.to_dict()
        restored = SkillBaseline.from_dict(d)
        assert restored.skill_id == bl.skill_id
        assert restored.metrics == bl.metrics

    def test_empty_metrics(self):
        bl = SkillBaseline(skill_id="test")
        assert bl.metrics == {}


class TestCompareMetrics:
    def test_improvement_detected(self):
        baseline = {"accuracy": 0.80, "confidence": 0.70}
        current = {"accuracy": 0.85, "confidence": 0.75}
        result = compare_metrics(baseline, current)
        assert result.passed is True
        assert "accuracy" in result.improved_metrics
        assert "confidence" in result.improved_metrics
        assert len(result.regressed_metrics) == 0

    def test_regression_detected(self):
        baseline = {"accuracy": 0.85, "confidence": 0.75}
        current = {"accuracy": 0.80, "confidence": 0.70}
        result = compare_metrics(baseline, current)
        assert result.passed is False
        assert "accuracy" in result.regressed_metrics

    def test_no_change_does_not_pass(self):
        baseline = {"accuracy": 0.80}
        current = {"accuracy": 0.80}
        result = compare_metrics(baseline, current)
        assert result.passed is False
        assert len(result.improved_metrics) == 0
        assert len(result.regressed_metrics) == 0

    def test_mixed_results_fail(self):
        baseline = {"accuracy": 0.80, "latency": 0.50}
        current = {"accuracy": 0.85, "latency": 0.60}
        result = compare_metrics(baseline, current, higher_is_better={"accuracy", "latency"})
        assert "accuracy" in result.improved_metrics
        assert result.passed is True

    def test_lower_is_better(self):
        baseline = {"false_positive_rate": 0.10}
        current = {"false_positive_rate": 0.05}
        result = compare_metrics(baseline, current, lower_is_better={"false_positive_rate"})
        assert result.passed is True
        assert "false_positive_rate" in result.improved_metrics

    def test_lower_is_better_regression(self):
        baseline = {"false_positive_rate": 0.05}
        current = {"false_positive_rate": 0.10}
        result = compare_metrics(baseline, current, lower_is_better={"false_positive_rate"})
        assert result.passed is False
        assert "false_positive_rate" in result.regressed_metrics

    def test_threshold_respected(self):
        baseline = {"accuracy": 0.800}
        current = {"accuracy": 0.805}
        result = compare_metrics(baseline, current, threshold=0.01)
        assert result.passed is False
        assert len(result.improved_metrics) == 0

    def test_just_above_threshold(self):
        baseline = {"accuracy": 0.800}
        current = {"accuracy": 0.815}
        result = compare_metrics(baseline, current, threshold=0.01)
        assert result.passed is True
        assert "accuracy" in result.improved_metrics

    def test_missing_current_metric_ignored(self):
        baseline = {"accuracy": 0.80, "missing": 0.50}
        current = {"accuracy": 0.85}
        result = compare_metrics(baseline, current)
        assert result.passed is True
        assert "missing" not in result.deltas

    def test_custom_threshold(self):
        baseline = {"accuracy": 0.80}
        current = {"accuracy": 0.81}
        result = compare_metrics(baseline, current, threshold=0.05)
        assert result.passed is False

    def test_summary_string(self):
        baseline = {"accuracy": 0.80}
        current = {"accuracy": 0.90}
        result = compare_metrics(baseline, current)
        assert "accuracy" in result.summary
        assert "improved" in result.summary

    def test_improvement_with_regression_blocks(self):
        baseline = {"accuracy": 0.80, "confidence": 0.70}
        current = {"accuracy": 0.90, "confidence": 0.55}
        result = compare_metrics(baseline, current)
        assert result.passed is False
        assert "accuracy" in result.improved_metrics
        assert "confidence" in result.regressed_metrics


class TestSkillValidationSerialization:
    def test_round_trip(self):
        v = SkillValidation(
            skill_id="test",
            baseline_metrics={"a": 1.0},
            current_metrics={"a": 2.0},
            deltas={"a": 1.0},
            improved_metrics=["a"],
            passed=True,
            summary="test",
        )
        d = v.to_dict()
        restored = SkillValidation.from_dict(d)
        assert restored.passed is True
        assert restored.improved_metrics == ["a"]


class TestBuildValidationEvidence:
    def test_produces_evidence_dict(self):
        v = SkillValidation(
            skill_id="speaker_identification",
            baseline_metrics={"accuracy": 0.80},
            current_metrics={"accuracy": 0.90},
            deltas={"accuracy": 0.10},
            improved_metrics=["accuracy"],
            passed=True,
            summary="accuracy improved",
        )

        class FakeJob:
            skill_id = "speaker_identification"
            evidence = {"required": ["test:speaker_id_accuracy_min"]}

        evidence = build_validation_evidence(v, FakeJob())
        assert evidence["result"] == "pass"
        assert any(t["name"] == "overall_improvement" and t["passed"] for t in evidence["tests"])
        assert "validation" in evidence

    def test_failure_evidence(self):
        v = SkillValidation(
            skill_id="test",
            baseline_metrics={"acc": 0.80},
            current_metrics={"acc": 0.75},
            deltas={"acc": -0.05},
            regressed_metrics=["acc"],
            passed=False,
        )

        class FakeJob:
            skill_id = "test"
            evidence = {"required": []}

        evidence = build_validation_evidence(v, FakeJob())
        assert evidence["result"] == "fail"

    def test_required_evidence_names_covered(self):
        v = SkillValidation(
            skill_id="test",
            deltas={"accuracy": 0.1},
            improved_metrics=["accuracy"],
            passed=True,
        )

        class FakeJob:
            skill_id = "test"
            evidence = {"required": ["test:custom_check", "test:another_check"]}

        evidence = build_validation_evidence(v, FakeJob())
        test_names = {t["name"] for t in evidence["tests"]}
        assert "custom_check" in test_names or "test:custom_check" in test_names
        assert "another_check" in test_names or "test:another_check" in test_names


class TestSpeakerMetrics:
    def test_with_empty_context(self):
        metrics = capture_speaker_id_metrics({})
        assert isinstance(metrics, dict)

    def test_with_mock_speaker_id(self):
        """Realistic SpeakerIdentifier shape — profiles carry enrollment_clips
        and interaction_count, never a 'confidence' field. Recognition
        confidence comes from the _score_ema dict on the identifier."""

        class MockSpeakerID:
            _profiles = {
                "alice": {"embedding": [0.1] * 192, "enrollment_clips": 5,
                          "interaction_count": 12, "last_seen": 1_700_000_000.0,
                          "registered": 1_699_900_000.0},
                "bob":   {"embedding": [0.1] * 192, "enrollment_clips": 3,
                          "interaction_count": 7, "last_seen": 1_700_000_100.0,
                          "registered": 1_699_950_000.0},
            }
            _score_ema = {"alice": 0.82, "bob": 0.74}

        metrics = capture_speaker_id_metrics({"speaker_id": MockSpeakerID()})
        assert metrics["enrolled_profiles"] == 2.0
        assert metrics["mean_enrollment_clips"] == 4.0
        assert metrics["mean_recognition_confidence"] == pytest.approx(0.78)

    def test_with_hemisphere(self):
        class MockHemiOrch:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "speaker_repr", "best_accuracy": 0.96,
                     "migration_readiness": 0.0},
                    {"focus": "emotion_depth", "best_accuracy": 0.90,
                     "migration_readiness": 0.0},
                ]}}

        metrics = capture_speaker_id_metrics({"hemisphere_orchestrator": MockHemiOrch()})
        assert metrics["speaker_repr_accuracy"] == pytest.approx(0.96)

    def test_with_distillation_stats(self):
        """Per-teacher distillation count is read from the 'total' key
        (not 'total_signals' — that aggregate only exists at the outer level)."""
        metrics = capture_speaker_id_metrics({
            "distillation_stats": {
                "teachers": {"ecapa_tdnn": {"total": 297, "quarantined": 3}},
                "total_signals": 600,  # outer aggregate — must NOT be read as per-teacher
            }
        })
        assert metrics["distillation_samples"] == 297.0


class TestEmotionMetrics:
    def test_with_empty_context(self):
        metrics = capture_emotion_metrics({})
        assert isinstance(metrics, dict)

    def test_model_healthy(self):
        class MockEmotion:
            _model_healthy = True
            _gpu_available = True

        metrics = capture_emotion_metrics({"emotion_classifier": MockEmotion()})
        assert metrics["model_healthy"] == 1.0
        assert metrics["gpu_available"] == 1.0


class TestCognitiveMetrics:
    def test_with_empty_context(self):
        metrics = capture_cognitive_metrics({})
        assert isinstance(metrics, dict)


class TestAutonomyMetrics:
    def test_with_empty_context(self):
        metrics = capture_autonomy_metrics({})
        assert isinstance(metrics, dict)


class TestGenericPerceptualMetrics:
    def test_with_hemisphere(self):
        class MockHemiOrch:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "speaker_repr", "best_accuracy": 0.6,
                     "migration_readiness": 0.0},
                    {"focus": "face_repr", "best_accuracy": 0.3,
                     "migration_readiness": 0.0},
                ]}}

        metrics = capture_generic_perceptual_metrics({"hemisphere_orchestrator": MockHemiOrch()})
        assert metrics["distilled_networks_ready"] == 1.0
        assert metrics["mean_specialist_accuracy"] > 0
        # 5 distilled focuses in the collector; only 2 present here → mean = 0.9 / 5
        assert metrics["mean_specialist_accuracy"] == pytest.approx((0.6 + 0.3) / 5)


class TestVerifierBugRegressions:
    """Regression tests for the three verifier bugs documented in
    ``autonomy/drives.py`` (fixed 2026-04-18):

      1. Hemisphere accuracy must read ``best_accuracy``, NOT
         ``migration_readiness`` (which only populates during substrate
         migration and is near-zero during standard distillation).
      2. Per-teacher distillation sample count must read
         ``teachers[t]["total"]``, NOT ``teachers[t]["total_signals"]``
         (the ``total_signals`` key only exists at the outer aggregate level).
      3. Speaker enrollment confidence must derive from the real profile
         schema (``enrollment_clips`` + ``_score_ema``), NOT a
         ``Profile.confidence`` field that never gets populated.

    Before the fix, all three reads returned 0.0 for real live data, which
    caused the Shadow Copy validation pattern to silently misreport
    "unchanged" for skills whose underlying specialists had actually
    trained substantially. See BUILD_HISTORY 2026-04-18 and
    ``drives.py::_DEFICIT_CAPABILITY_MAP`` comment.
    """

    # ── Bug 1: best_accuracy, not migration_readiness ───────────────────
    def test_speaker_reads_best_accuracy_not_migration_readiness(self):
        class HemiWithBoth:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    # Real producer shape: both fields present, only
                    # best_accuracy carries the training signal.
                    {"focus": "speaker_repr",
                     "best_accuracy": 0.92,
                     "migration_readiness": 0.0},
                ]}}

        metrics = capture_speaker_id_metrics({"hemisphere_orchestrator": HemiWithBoth()})
        assert metrics["speaker_repr_accuracy"] == pytest.approx(0.92)
        # Must NOT silently fall back to migration_readiness when
        # best_accuracy is absent — returns 0.0 honestly instead.
        class HemiOnlyMigration:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "speaker_repr", "migration_readiness": 0.9},
                ]}}
        metrics2 = capture_speaker_id_metrics({"hemisphere_orchestrator": HemiOnlyMigration()})
        assert metrics2["speaker_repr_accuracy"] == 0.0

    def test_emotion_reads_best_accuracy_not_migration_readiness(self):
        class HemiWithBoth:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "emotion_depth",
                     "best_accuracy": 0.87,
                     "migration_readiness": 0.0},
                ]}}

        metrics = capture_emotion_metrics({"hemisphere_orchestrator": HemiWithBoth()})
        assert metrics["emotion_depth_accuracy"] == pytest.approx(0.87)

    def test_generic_reads_best_accuracy_not_migration_readiness(self):
        class HemiWithBoth:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "emotion_depth",  "best_accuracy": 0.80, "migration_readiness": 0.0},
                    {"focus": "speaker_repr",   "best_accuracy": 0.90, "migration_readiness": 0.0},
                    {"focus": "face_repr",      "best_accuracy": 0.70, "migration_readiness": 0.0},
                ]}}

        metrics = capture_generic_perceptual_metrics({"hemisphere_orchestrator": HemiWithBoth()})
        # 3 of 5 distilled focuses present and all >= 0.5 → 3 ready
        assert metrics["distilled_networks_ready"] == 3.0
        # Mean over ALL 5 distilled focuses (missing ones count as 0)
        assert metrics["mean_specialist_accuracy"] == pytest.approx((0.80 + 0.90 + 0.70) / 5)
        # Old metric name must be gone
        assert "mean_migration_readiness" not in metrics

    # ── Bug 2: per-teacher ``total``, not ``total_signals`` ─────────────
    def test_speaker_reads_per_teacher_total(self):
        metrics = capture_speaker_id_metrics({
            "distillation_stats": {
                "teachers": {"ecapa_tdnn": {"total": 297, "total_signals": 0}},
            }
        })
        assert metrics["distillation_samples"] == 297.0

    def test_speaker_ignores_outer_total_signals_aggregate(self):
        # Only the outer aggregate key is set — per-teacher ``total`` is 0.
        # The collector must honestly report 0, not accidentally pick up 600.
        metrics = capture_speaker_id_metrics({
            "distillation_stats": {
                "teachers": {"ecapa_tdnn": {}},
                "total_signals": 600,
            }
        })
        assert metrics.get("distillation_samples", 0.0) == 0.0

    def test_emotion_reads_per_teacher_total(self):
        metrics = capture_emotion_metrics({
            "distillation_stats": {
                "teachers": {"wav2vec2_emotion": {"total": 412, "total_signals": 0}},
            }
        })
        assert metrics["distillation_samples"] == 412.0

    # ── Bug 3: profile.confidence is gone; use real schema ──────────────
    def test_speaker_derives_from_real_profile_schema(self):
        """Real SpeakerIdentifier._profiles entries have embedding /
        enrollment_clips / interaction_count / last_seen / registered —
        never a 'confidence' field. Collector must still produce sensible
        enrollment-quality and recognition-confidence metrics."""

        class RealShape:
            _profiles = {
                "dan": {"embedding": [0.1] * 192, "enrollment_clips": 6,
                        "interaction_count": 20, "last_seen": 1_700_000_000.0,
                        "registered": 1_699_800_000.0},
                "eve": {"embedding": [0.1] * 192, "enrollment_clips": 4,
                        "interaction_count": 14, "last_seen": 1_700_000_050.0,
                        "registered": 1_699_850_000.0},
            }
            _score_ema = {"dan": 0.88, "eve": 0.72}

        metrics = capture_speaker_id_metrics({"speaker_id": RealShape()})
        assert metrics["enrolled_profiles"] == 2.0
        assert metrics["mean_enrollment_clips"] == pytest.approx(5.0)
        assert metrics["mean_recognition_confidence"] == pytest.approx(0.80)
        # Old broken metric must be gone.
        assert "mean_enrollment_confidence" not in metrics

    def test_speaker_recognition_confidence_excludes_unseen_profiles(self):
        """Freshly enrolled profiles have no EMA entry yet (see
        speaker_id.enroll() which pops from _score_ema). They must not
        spuriously pull the mean down."""

        class Partial:
            _profiles = {
                "seen":    {"embedding": [0.1] * 192, "enrollment_clips": 3,
                            "interaction_count": 5},
                "unseen":  {"embedding": [0.1] * 192, "enrollment_clips": 3,
                            "interaction_count": 0},
            }
            _score_ema = {"seen": 0.90}  # "unseen" absent

        metrics = capture_speaker_id_metrics({"speaker_id": Partial()})
        assert metrics["mean_recognition_confidence"] == pytest.approx(0.90)

    def test_speaker_recognition_confidence_falls_back_to_zero(self):
        class NoScoreEma:
            _profiles = {
                "x": {"embedding": [0.1] * 192, "enrollment_clips": 2,
                      "interaction_count": 0},
            }
            # No _score_ema attribute at all.

        metrics = capture_speaker_id_metrics({"speaker_id": NoScoreEma()})
        assert metrics["mean_recognition_confidence"] == 0.0

    # ── End-to-end: realistic training improvement is actually detected ─
    def test_end_to_end_detects_real_training_improvement(self):
        """Pre-fix scenario: specialist trains from 0.50 → 0.92 best_accuracy
        and distillation samples grow 20 → 297. Old code read 0→0 for both
        (wrong field names) and silently declared 'unchanged'. After the
        fix, compare_metrics must see an unambiguous improvement and pass."""

        class HemiBefore:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "speaker_repr", "best_accuracy": 0.50,
                     "migration_readiness": 0.0},
                ]}}

        class HemiAfter:
            def get_state(self):
                return {"hemisphere_state": {"hemispheres": [
                    {"focus": "speaker_repr", "best_accuracy": 0.92,
                     "migration_readiness": 0.0},
                ]}}

        baseline = capture_speaker_id_metrics({
            "hemisphere_orchestrator": HemiBefore(),
            "distillation_stats": {
                "teachers": {"ecapa_tdnn": {"total": 20}},
            },
        })
        current = capture_speaker_id_metrics({
            "hemisphere_orchestrator": HemiAfter(),
            "distillation_stats": {
                "teachers": {"ecapa_tdnn": {"total": 297}},
            },
        })

        assert baseline["speaker_repr_accuracy"] == pytest.approx(0.50)
        assert current["speaker_repr_accuracy"] == pytest.approx(0.92)
        assert baseline["distillation_samples"] == 20.0
        assert current["distillation_samples"] == 297.0

        result = compare_metrics(
            baseline, current,
            higher_is_better=HIGHER_IS_BETTER["speaker"],
            lower_is_better=LOWER_IS_BETTER["speaker"],
        )
        assert result.passed is True
        assert "speaker_repr_accuracy" in result.improved_metrics
        assert "distillation_samples" in result.improved_metrics
        assert result.regressed_metrics == []


class TestMetricRegistries:
    def test_all_collectors_have_higher_lower(self):
        from skills.baseline import METRIC_COLLECTORS
        for key in METRIC_COLLECTORS:
            assert key in HIGHER_IS_BETTER, f"Missing HIGHER_IS_BETTER for {key}"
            assert key in LOWER_IS_BETTER, f"Missing LOWER_IS_BETTER for {key}"

    def test_speaker_sets_consistent(self):
        overlap = HIGHER_IS_BETTER["speaker"] & LOWER_IS_BETTER["speaker"]
        assert len(overlap) == 0, f"Metrics in both higher and lower: {overlap}"

    def test_all_sets_consistent(self):
        for key in HIGHER_IS_BETTER:
            overlap = HIGHER_IS_BETTER[key] & LOWER_IS_BETTER[key]
            assert len(overlap) == 0, f"Metrics in both higher and lower for {key}: {overlap}"


# ---------------------------------------------------------------------------
# Integration: full assess → verify flow with mock runtime objects
# ---------------------------------------------------------------------------

class _MockHemiOrch:
    """Simulates HemisphereOrchestrator with controllable specialist accuracy.

    Uses the real producer field names: ``best_accuracy`` is the distillation
    training signal, and ``migration_readiness`` is kept at 0 to prove the
    consumers no longer read that substrate-migration-only field.
    """
    def __init__(self, speaker_accuracy=0.3):
        self._speaker_accuracy = speaker_accuracy

    def get_state(self):
        return {"hemisphere_state": {"hemispheres": [
            {"focus": "speaker_repr", "best_accuracy": self._speaker_accuracy,
             "migration_readiness": 0.0},
            {"focus": "emotion_depth", "best_accuracy": 0.2,
             "migration_readiness": 0.0},
        ]}}

    def get_distillation_stats(self):
        return {"teachers": {"ecapa_tdnn": {"total": 50}}}


class _MockSpeakerID:
    """Simulates SpeakerID with controllable profiles (real profile shape)."""
    def __init__(self, profiles=None, score_ema=None):
        self._profiles = profiles or {
            "alice": {"embedding": [0.1] * 192, "enrollment_clips": 4,
                      "interaction_count": 10, "last_seen": 1_700_000_000.0,
                      "registered": 1_699_900_000.0},
            "bob":   {"embedding": [0.1] * 192, "enrollment_clips": 3,
                      "interaction_count": 6, "last_seen": 1_700_000_100.0,
                      "registered": 1_699_950_000.0},
        }
        self._score_ema = score_ema if score_ema is not None else {"alice": 0.8, "bob": 0.7}


class _MockJob:
    """Simulates a LearningJob for executor testing."""
    def __init__(self, skill_id="speaker_identification_v1"):
        self.job_id = "test_job_001"
        self.skill_id = skill_id
        self.capability_type = "perceptual"
        self.phase = "assess"
        self.status = "active"
        self.matrix_protocol = False
        self.plan = {"phases": [
            {"name": "assess", "exit_conditions": ["gate:speaker_profiles_exist"]},
            {"name": "verify", "exit_conditions": ["evidence:test:speaker_id_accuracy_min"]},
        ]}
        self.gates = {"hard": [{"id": "gate:speaker_profiles_exist", "required": True, "state": "unknown"}]}
        self.data = {"sources": [], "counters": {}}
        self.artifacts = []
        self.evidence = {
            "required": ["test:speaker_id_accuracy_min", "test:speaker_id_false_positive_max"],
            "latest": None,
            "history": [],
        }
        self.events = []
        self.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        self.executor_state = {}


class TestAssessToVerifyIntegration:
    """Integration test: assess captures baseline, verify compares against it."""

    def _build_ctx(self, speaker_accuracy=0.3):
        return {
            "speaker_id": _MockSpeakerID(),
            "hemisphere_orchestrator": _MockHemiOrch(speaker_accuracy),
            "distillation_stats": {"teachers": {"ecapa_tdnn": {"total": 50}}},
            "now_iso": "2026-03-23T12:00:00Z",
        }

    def test_assess_captures_baseline(self):
        from skills.executors.perceptual import PerceptualAssessExecutor
        job = _MockJob()
        ctx = self._build_ctx()
        executor = PerceptualAssessExecutor()

        result = executor.run(job, ctx)
        assert result.progressed is True
        assert "baseline" in job.data
        baseline = job.data["baseline"]
        assert baseline["skill_id"] == "speaker_identification_v1"
        assert "enrolled_profiles" in baseline["metrics"]
        assert baseline["metrics"]["enrolled_profiles"] == 2.0
        assert "speaker_repr_accuracy" in baseline["metrics"]
        assert baseline["metrics"]["speaker_repr_accuracy"] == 0.3

    def test_verify_with_no_improvement_fails(self):
        from skills.executors.perceptual import PerceptualAssessExecutor, PerceptualVerifyExecutor
        job = _MockJob()
        ctx = self._build_ctx(speaker_accuracy=0.3)

        PerceptualAssessExecutor().run(job, ctx)
        assert "baseline" in job.data

        job.phase = "verify"
        verify_ctx = self._build_ctx(speaker_accuracy=0.3)
        result = PerceptualVerifyExecutor().run(job, verify_ctx)

        assert result.progressed is True
        assert result.evidence is not None
        assert result.evidence["result"] == "fail"
        assert "validation" in job.data
        assert job.data["validation"]["passed"] is False

    def test_verify_with_improvement_passes(self):
        from skills.executors.perceptual import PerceptualAssessExecutor, PerceptualVerifyExecutor
        job = _MockJob()
        ctx = self._build_ctx(speaker_accuracy=0.3)

        PerceptualAssessExecutor().run(job, ctx)
        baseline_acc = job.data["baseline"]["metrics"]["speaker_repr_accuracy"]
        assert baseline_acc == 0.3

        job.phase = "verify"
        improved_ctx = self._build_ctx(speaker_accuracy=0.65)
        result = PerceptualVerifyExecutor().run(job, improved_ctx)

        assert result.progressed is True
        assert result.evidence is not None
        assert result.evidence["result"] == "pass"
        assert "validation" in job.data
        assert job.data["validation"]["passed"] is True
        assert "speaker_repr_accuracy" in job.data["validation"]["improved_metrics"]

    def test_verify_with_regression_fails(self):
        from skills.executors.perceptual import PerceptualAssessExecutor, PerceptualVerifyExecutor
        job = _MockJob()
        ctx = self._build_ctx(speaker_accuracy=0.5)

        PerceptualAssessExecutor().run(job, ctx)

        job.phase = "verify"
        regressed_ctx = self._build_ctx(speaker_accuracy=0.2)
        result = PerceptualVerifyExecutor().run(job, regressed_ctx)

        assert result.evidence["result"] == "fail"
        assert job.data["validation"]["passed"] is False
        assert "speaker_repr_accuracy" in job.data["validation"]["regressed_metrics"]

    def test_verify_falls_back_without_baseline(self):
        from skills.executors.perceptual import PerceptualVerifyExecutor
        job = _MockJob()
        job.phase = "verify"
        ctx = self._build_ctx(speaker_accuracy=0.6)

        result = PerceptualVerifyExecutor().run(job, ctx)

        assert result.progressed is True
        assert "validation" not in job.data
        assert result.evidence is not None

    def test_baseline_survives_serialization(self):
        """Baseline must survive JSON round-trip (job persistence between ticks)."""
        import json
        from skills.executors.perceptual import PerceptualAssessExecutor
        job = _MockJob()
        ctx = self._build_ctx()

        PerceptualAssessExecutor().run(job, ctx)
        serialized = json.dumps(job.data)
        restored = json.loads(serialized)
        assert restored["baseline"]["metrics"]["enrolled_profiles"] == 2.0
        assert restored["baseline"]["metrics"]["speaker_repr_accuracy"] == 0.3

    def test_evidence_includes_delta_details(self):
        from skills.executors.perceptual import PerceptualAssessExecutor, PerceptualVerifyExecutor
        job = _MockJob()
        ctx = self._build_ctx(speaker_accuracy=0.3)

        PerceptualAssessExecutor().run(job, ctx)

        job.phase = "verify"
        result = PerceptualVerifyExecutor().run(job, self._build_ctx(speaker_accuracy=0.7))

        tests = result.evidence["tests"]
        delta_tests = [t for t in tests if t["name"].startswith("delta:")]
        assert len(delta_tests) > 0
        assert any(t["name"] == "overall_improvement" for t in tests)

    def test_non_speaker_skill_uses_generic_collector(self):
        from skills.executors.perceptual import PerceptualAssessExecutor
        job = _MockJob(skill_id="vision_analysis_v1")
        ctx = self._build_ctx()

        PerceptualAssessExecutor().run(job, ctx)
        assert "baseline" in job.data
        metrics = job.data["baseline"]["metrics"]
        assert "enrolled_profiles" not in metrics
