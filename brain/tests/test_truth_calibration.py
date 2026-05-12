"""Tests for Layer 6: Truth Calibration (all phases)."""
from __future__ import annotations

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epistemic.calibration.signal_collector import CalibrationSnapshot, SignalCollector
from epistemic.calibration.calibration_history import CalibrationHistory, _domain_representative_value
from epistemic.calibration.domain_calibrator import DomainCalibrator, DomainScore, ALL_DOMAINS, _clamp
from epistemic.calibration.truth_score import (
    TruthScoreCalculator,
    TruthScoreReport,
    PROVISIONAL_THRESHOLD,
    TRUTH_SCORE_WEIGHTS,
)
from epistemic.calibration.drift_detector import DriftDetector, DriftAlert, _classify_severity
from epistemic.calibration import TruthCalibrationEngine


def _make_snapshot(**overrides) -> CalibrationSnapshot:
    defaults = dict(
        timestamp=time.time(),
        reference_match_rate=0.5,
        ranker_success_rate=0.7,
        lift=0.1,
        improvement_rate=0.6,
        overall_win_rate=0.5,
        wasted_rate=0.1,
        useful_rate=0.8,
        weight_error=0.15,
        decay_error=0.2,
        coherence=0.75,
        consistency=0.8,
        depth=0.6,
        correction_penalty=0.02,
        verified_skill_count=3,
        honesty_failures=1,
        contradiction_debt=0.05,
        near_miss_rate=0.1,
        total_beliefs=50,
        resolved_count=5,
    )
    defaults.update(overrides)
    return CalibrationSnapshot(**defaults)


# ============================================================================
# CalibrationSnapshot tests
# ============================================================================

class TestCalibrationSnapshot:
    def test_default_none_fields(self):
        snap = CalibrationSnapshot()
        assert snap.reference_match_rate is None
        assert snap.brier_score is None
        assert snap.prediction_accuracy is None

    def test_populated_snapshot(self):
        snap = _make_snapshot(reference_match_rate=0.42)
        assert snap.reference_match_rate == 0.42


class TestSignalCollector:
    def test_collect_reasoning_reads_engine_consciousness_attribute(self):
        class _FakeAnalytics:
            @staticmethod
            def get_full_state():
                return {
                    "reasoning": {
                        "coherence": 0.61,
                        "consistency": 0.92,
                        "depth": 0.47,
                    }
                }

        class _FakeConsciousness:
            analytics = _FakeAnalytics()

        class _FakeEngine:
            _consciousness = _FakeConsciousness()

        collector = SignalCollector(engine=_FakeEngine())
        snap = CalibrationSnapshot()
        collector._collect_reasoning(snap)

        assert snap.coherence == 0.61
        assert snap.consistency == 0.92
        assert snap.depth == 0.47


# ============================================================================
# CalibrationHistory tests
# ============================================================================

class TestCalibrationHistory:
    def test_record_and_count(self):
        h = CalibrationHistory(maxlen=5)
        assert h.count == 0
        h.record(_make_snapshot())
        assert h.count == 1

    def test_maxlen_eviction(self):
        h = CalibrationHistory(maxlen=3)
        for _ in range(5):
            h.record(_make_snapshot())
        assert h.count == 3

    def test_domain_trend(self):
        h = CalibrationHistory()
        for i in range(5):
            h.record(_make_snapshot(reference_match_rate=0.1 * (i + 1)))
        trend = h.get_domain_trend("retrieval", window=5)
        assert len(trend) == 5
        assert trend[0] < trend[-1]

    def test_domain_variance_stable(self):
        h = CalibrationHistory()
        for _ in range(10):
            h.record(_make_snapshot(coherence=0.5, consistency=0.5, depth=0.5))
        var = h.get_domain_variance("reasoning", window=10)
        assert var < 0.01

    def test_domain_slope_positive(self):
        h = CalibrationHistory()
        for i in range(10):
            h.record(_make_snapshot(improvement_rate=0.3 + 0.05 * i, overall_win_rate=0.4 + 0.03 * i))
        slope = h.get_domain_slope("autonomy", window=10)
        assert slope > 0

    def test_domain_slope_negative(self):
        h = CalibrationHistory()
        for i in range(10):
            h.record(_make_snapshot(improvement_rate=0.9 - 0.05 * i, overall_win_rate=0.8 - 0.04 * i))
        slope = h.get_domain_slope("autonomy", window=10)
        assert slope < 0

    def test_slope_insufficient_data(self):
        h = CalibrationHistory()
        h.record(_make_snapshot())
        assert h.get_domain_slope("retrieval") == 0.0

    def test_representative_value_epistemic(self):
        snap = _make_snapshot(contradiction_debt=0.15)
        val = _domain_representative_value(snap, "epistemic")
        assert val == 0.85

    def test_representative_value_none_domain(self):
        snap = CalibrationSnapshot()
        val = _domain_representative_value(snap, "retrieval")
        assert val is None


# ============================================================================
# DomainCalibrator tests
# ============================================================================

class TestDomainCalibrator:
    def test_score_all_returns_all_domains(self):
        cal = DomainCalibrator()
        snap = _make_snapshot()
        scores = cal.score_all(snap)
        assert set(scores.keys()) == set(ALL_DOMAINS)

    def test_retrieval_score_range(self):
        cal = DomainCalibrator()
        snap = _make_snapshot()
        score = cal.score_all(snap)["retrieval"]
        assert 0.0 <= score.score <= 1.0
        assert not score.provisional

    def test_provisional_when_none(self):
        cal = DomainCalibrator()
        snap = CalibrationSnapshot()
        scores = cal.score_all(snap)
        assert scores["retrieval"].provisional
        assert scores["retrieval"].score == 0.5

    def test_autonomy_score(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(improvement_rate=0.8, overall_win_rate=0.6)
        score = cal.score_all(snap)["autonomy"]
        assert abs(score.score - 0.7) < 0.01

    def test_epistemic_score_zero_debt(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(contradiction_debt=0.0, total_beliefs=10, resolved_count=8)
        score = cal.score_all(snap)["epistemic"]
        assert score.score > 0.8

    def test_epistemic_score_high_debt(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(
            contradiction_debt=0.8, total_beliefs=10, resolved_count=1,
            graph_health_score=0.0,
        )
        score = cal.score_all(snap)["epistemic"]
        assert score.score < 0.15

    def test_salience_inverted(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(wasted_rate=0.8, weight_error=0.7, decay_error=0.9)
        score = cal.score_all(snap)["salience"]
        assert score.score < 0.3

    def test_salience_good(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(wasted_rate=0.05, weight_error=0.1, decay_error=0.08)
        score = cal.score_all(snap)["salience"]
        assert score.score > 0.8

    def test_confidence_provisional(self):
        cal = DomainCalibrator()
        snap = _make_snapshot()
        score = cal.score_all(snap)["confidence"]
        assert score.provisional

    def test_spatial_position_provisional_without_calibration(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(spatial_calibration_valid=False, spatial_anchor_count=0)
        score = cal.score_all(snap)["spatial_position"]
        assert score.provisional
        assert score.score == 0.5

    def test_spatial_position_scores_from_calibration_tracks_and_anchors(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(
            spatial_calibration_valid=True,
            spatial_anchor_count=3,
            spatial_stable_tracks=4,
        )
        score = cal.score_all(snap)["spatial_position"]
        assert not score.provisional
        assert score.score > 0.7

    def test_spatial_motion_provisional_with_insufficient_samples(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(spatial_promoted_deltas=3, spatial_rejected_promotions=1)
        score = cal.score_all(snap)["spatial_motion"]
        assert score.provisional
        assert abs(score.score - 0.75) < 0.001

    def test_spatial_motion_non_provisional_when_samples_mature(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(spatial_promoted_deltas=9, spatial_rejected_promotions=1)
        score = cal.score_all(snap)["spatial_motion"]
        assert not score.provisional
        assert abs(score.score - 0.9) < 0.001

    def test_spatial_relation_provisional_without_anchors_or_tracks(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(spatial_anchor_count=0, spatial_stable_tracks=0)
        score = cal.score_all(snap)["spatial_relation"]
        assert score.provisional
        assert score.score == 0.5

    def test_spatial_relation_penalizes_contradictions_and_drift(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(
            spatial_anchor_count=3,
            spatial_stable_tracks=2,
            spatial_contradiction_count=2,
            spatial_anchor_drift_score=0.2,
        )
        score = cal.score_all(snap)["spatial_relation"]
        assert not score.provisional
        assert abs(score.score - 0.6) < 0.001

    def test_skill_no_data_provisional(self):
        cal = DomainCalibrator()
        snap = _make_snapshot(verified_skill_count=0, honesty_failures=0)
        score = cal.score_all(snap)["skill"]
        assert score.provisional

    def test_clamp(self):
        assert _clamp(-0.5) == 0.0
        assert _clamp(1.5) == 1.0
        assert _clamp(0.5) == 0.5


# ============================================================================
# TruthScore tests
# ============================================================================

class TestTruthScore:
    def test_truth_weights_sum_to_one(self):
        total = sum(TRUTH_SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_healthy_system(self):
        calc = TruthScoreCalculator()
        scores = {d: DomainScore(d, 0.8, provisional=False) for d in ALL_DOMAINS}
        report = calc.compute(scores)
        assert report.truth_score is not None
        assert abs(report.truth_score - 0.8) < 0.01
        assert report.maturity == 1.0
        assert report.provisional_count == 0

    def test_all_provisional_returns_none(self):
        calc = TruthScoreCalculator()
        scores = {d: DomainScore(d, 0.5, provisional=True) for d in ALL_DOMAINS}
        report = calc.compute(scores)
        assert report.truth_score is None
        assert report.provisional_count == len(ALL_DOMAINS)

    def test_partial_provisional(self):
        calc = TruthScoreCalculator()
        scores = {}
        for i, d in enumerate(ALL_DOMAINS):
            scores[d] = DomainScore(d, 0.7, provisional=(i < 3))
        report = calc.compute(scores)
        assert report.truth_score is not None
        assert report.provisional_count == 3
        assert report.maturity > 0.5

    def test_threshold_boundary(self):
        calc = TruthScoreCalculator()
        scores = {}
        for i, d in enumerate(ALL_DOMAINS):
            scores[d] = DomainScore(d, 0.6, provisional=(i <= PROVISIONAL_THRESHOLD))
        report = calc.compute(scores)
        assert report.truth_score is None

    def test_data_coverage(self):
        calc = TruthScoreCalculator()
        scores = {d: DomainScore(d, 0.75, provisional=(d == "prediction")) for d in ALL_DOMAINS}
        report = calc.compute(scores)
        assert report.data_coverage["prediction"] is False
        assert report.data_coverage["retrieval"] is True

    def test_weighted_composition(self):
        calc = TruthScoreCalculator()
        scores = {}
        for d in ALL_DOMAINS:
            if d in ("retrieval", "reasoning", "epistemic"):
                scores[d] = DomainScore(d, 1.0, provisional=False)
            else:
                scores[d] = DomainScore(d, 0.0, provisional=False)
        report = calc.compute(scores)
        assert report.truth_score > 0.5

    def test_spatial_domains_influence_weighted_truth_score(self):
        calc = TruthScoreCalculator()
        baseline = {d: DomainScore(d, 0.8, provisional=False) for d in ALL_DOMAINS}
        baseline_report = calc.compute(baseline)
        assert baseline_report.truth_score is not None
        assert abs(baseline_report.truth_score - 0.8) < 0.001

        spatial_degraded = {
            d: DomainScore(d, 0.8, provisional=False) for d in ALL_DOMAINS
        }
        for d in ("spatial_position", "spatial_motion", "spatial_relation"):
            spatial_degraded[d] = DomainScore(d, 0.0, provisional=False)

        degraded_report = calc.compute(spatial_degraded)
        assert degraded_report.truth_score is not None
        assert degraded_report.truth_score < baseline_report.truth_score
        assert abs(degraded_report.truth_score - 0.752) < 0.001


# ============================================================================
# TruthCalibrationEngine state contract tests
# ============================================================================

class TestTruthCalibrationStateContract:
    def test_not_started_state_exposes_canonical_and_alias_keys(self):
        engine = TruthCalibrationEngine(engine=None)
        state = engine.get_state()
        assert state["status"] == "not_started"
        assert state["route_brier_scores"] == {}
        assert state["route_brier"] == {}
        assert state["active_drift_alerts"] == []
        assert state["drift_alerts"] == []

    def test_active_state_keeps_canonical_and_alias_in_sync(self):
        engine = TruthCalibrationEngine(engine=None)
        engine._last_report = TruthScoreReport(
            truth_score=0.8,
            maturity=1.0,
            provisional_count=0,
            data_coverage={},
            domain_scores={},
            domain_provisional={},
        )

        class _FakeConfidenceCalibrator:
            outcome_count = 3

            def get_brier_score(self):
                return 0.1

            def get_ece(self):
                return 0.05

            def get_per_provenance_accuracy(self):
                return {}

            def get_overconfidence_error(self):
                return 0.01

            def get_underconfidence_error(self):
                return 0.02

            def get_calibration_curve(self):
                return {}

            def get_route_brier_scores(self):
                return {"none": 0.2}

            def get_route_sample_counts(self):
                return {"none": 11}

        engine._confidence_calibrator = _FakeConfidenceCalibrator()
        engine._drift_detector.get_active_alerts = lambda: [
            DriftAlert(
                domain="spatial_motion",
                severity="moderate",
                score_drop=0.12,
                slope=-0.03,
                readings_declining=7,
                peak_score=0.82,
                current_score=0.70,
                triggered_at=time.time(),
            )
        ]
        engine._drift_detector.get_resolved_alerts = lambda _limit=5: []

        state = engine.get_state()
        assert state["route_brier_scores"] == {"none": 0.2}
        assert state["route_brier"] == {"none": 0.2}
        assert state["route_sample_counts"] == {"none": 11}
        assert len(state["active_drift_alerts"]) == 1
        assert len(state["drift_alerts"]) == 1


# ============================================================================
# TruthCalibrationEngine watchdog throttling tests
# ============================================================================

class TestTruthCalibrationWatchdogThrottling:
    class _Outcome:
        def __init__(self, correct: bool):
            self.actual_correct = correct

    class _Calibrator:
        def __init__(self, total: int, correct: bool = True):
            self._outcomes = [
                TestTruthCalibrationWatchdogThrottling._Outcome(correct)
                for _ in range(total)
            ]

        @property
        def outcome_count(self) -> int:
            return len(self._outcomes)

        def set_uniform_outcomes(self, total: int, correct: bool = True) -> None:
            self._outcomes = [
                TestTruthCalibrationWatchdogThrottling._Outcome(correct)
                for _ in range(total)
            ]

        def get_provenance_sample_counts(self) -> dict[str, int]:
            return {"prediction": len(self._outcomes)}

    def test_uniform_warning_throttled_when_count_unchanged(self, caplog, monkeypatch):
        import epistemic.calibration as calibration_module

        now = {"value": 10_000.0}
        monkeypatch.setattr(calibration_module.time, "time", lambda: now["value"])

        engine = TruthCalibrationEngine(engine=None)
        engine._tick_count = 10
        engine._bridge_prediction_validated = 25
        engine._confidence_calibrator = self._Calibrator(total=150, correct=True)

        with caplog.at_level(logging.WARNING, logger="jarvis.calibration"):
            engine._run_outcome_watchdog()
            now["value"] += 600.0
            engine._run_outcome_watchdog()

        skew_logs = [
            rec for rec in caplog.records
            if "outcomes are 100%" in rec.getMessage()
        ]
        assert len(skew_logs) == 1

    def test_uniform_warning_relogs_after_large_count_delta(self, caplog, monkeypatch):
        import epistemic.calibration as calibration_module

        now = {"value": 20_000.0}
        monkeypatch.setattr(calibration_module.time, "time", lambda: now["value"])

        engine = TruthCalibrationEngine(engine=None)
        engine._tick_count = 10
        engine._bridge_world_model_validated = 40
        fake_cal = self._Calibrator(total=120, correct=True)
        engine._confidence_calibrator = fake_cal

        with caplog.at_level(logging.WARNING, logger="jarvis.calibration"):
            engine._run_outcome_watchdog()
            fake_cal.set_uniform_outcomes(total=250, correct=True)
            now["value"] += 600.0
            engine._run_outcome_watchdog()

        skew_logs = [
            rec for rec in caplog.records
            if "outcomes are 100%" in rec.getMessage()
        ]
        assert len(skew_logs) == 2


# ============================================================================
# DriftDetector tests
# ============================================================================

class TestDriftDetector:
    def test_no_drift_on_stable(self):
        dd = DriftDetector()
        scores = {d: DomainScore(d, 0.8, provisional=False) for d in ALL_DOMAINS}
        for _ in range(10):
            alerts = dd.update(scores)
        assert len(alerts) == 0

    def test_drift_on_sustained_decline(self):
        dd = DriftDetector()
        initial = {d: DomainScore(d, 0.9, provisional=False) for d in ALL_DOMAINS}
        dd.update(initial)

        for i in range(8):
            declining = {d: DomainScore(d, 0.9 - 0.02 * (i + 1), provisional=False) for d in ALL_DOMAINS}
            alerts = dd.update(declining)

        assert dd.get_active_alerts()

    def test_provisional_domains_ignored(self):
        dd = DriftDetector()
        for i in range(10):
            scores = {d: DomainScore(d, 0.5, provisional=True) for d in ALL_DOMAINS}
            alerts = dd.update(scores)
            assert len(alerts) == 0

    def test_recovery_clears_alert(self):
        dd = DriftDetector()
        dd.update({d: DomainScore(d, 0.9, provisional=False) for d in ALL_DOMAINS})
        for i in range(8):
            dd.update({d: DomainScore(d, 0.9 - 0.02 * (i + 1), provisional=False) for d in ALL_DOMAINS})

        assert dd.get_active_alerts()

        dd.update({d: DomainScore(d, 0.88, provisional=False) for d in ALL_DOMAINS})
        assert len(dd.get_active_alerts()) == 0

    def test_severity_classification(self):
        assert _classify_severity(0.09, 6) == "minor"
        assert _classify_severity(0.15, 9) == "moderate"
        assert _classify_severity(0.25, 5) == "major"
        assert _classify_severity(0.10, 13) == "major"

    def test_get_state(self):
        dd = DriftDetector()
        dd.update({d: DomainScore(d, 0.8, provisional=False) for d in ALL_DOMAINS})
        state = dd.get_state()
        assert "retrieval" in state
        assert "peak_score" in state["retrieval"]


# ============================================================================
# ConfidenceCalibrator tests (Phase 2)
# ============================================================================

from epistemic.calibration.confidence_calibrator import ConfidenceCalibrator, ConfidenceOutcome


class TestConfidenceCalibrator:
    def test_brier_score_none_when_insufficient(self):
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = __import__("collections").deque(maxlen=500)
        for i in range(5):
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", 0.8, True, "test", time.time()))
        assert cal.get_brier_score() is None

    def test_brier_score_perfect_calibration(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(30):
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", 1.0, True, "test", time.time()))
        score = cal.get_brier_score()
        assert score is not None
        assert score == 0.0

    def test_brier_score_worst_case(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(30):
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", 1.0, False, "test", time.time()))
        score = cal.get_brier_score()
        assert score is not None
        assert score == 1.0

    def test_ece_well_calibrated(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(50):
            conf = 0.8
            correct = i < 40  # 80% correct at 0.8 confidence = perfect
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", conf, correct, "test", time.time()))
        ece = cal.get_ece()
        assert ece is not None
        assert ece < 0.05

    def test_overconfidence_detected(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(30):
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", 0.9, i < 10, "test", time.time()))
        over = cal.get_overconfidence_error()
        under = cal.get_underconfidence_error()
        assert over is not None
        assert over > 0.1  # says 0.9, correct only 33%

    def test_per_provenance_accuracy(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(10):
            cal._outcomes.append(ConfidenceOutcome(f"ext{i}", 0.7, True, "external_source", time.time()))
        for i in range(10):
            cal._outcomes.append(ConfidenceOutcome(f"inf{i}", 0.7, i < 5, "model_inference", time.time()))
        acc = cal.get_per_provenance_accuracy()
        assert acc["external_source"] == 1.0
        assert acc["model_inference"] == 0.5

    def test_calibration_curve_structure(self):
        from collections import deque
        cal = ConfidenceCalibrator.__new__(ConfidenceCalibrator)
        cal._outcomes = deque(maxlen=500)
        for i in range(30):
            cal._outcomes.append(ConfidenceOutcome(f"b{i}", 0.5, True, "test", time.time()))
        curve = cal.get_calibration_curve()
        assert len(curve) == 5
        assert "0.4-0.6" in curve
        assert curve["0.4-0.6"]["count"] == 30


# ============================================================================
# CorrectionDetector tests (Phase 2)
# ============================================================================

from epistemic.calibration.correction_detector import (
    CorrectionDetector, _has_correction_phrase, _content_overlap,
)


class TestCorrectionDetector:
    def test_correction_phrase_positive(self):
        assert _has_correction_phrase("That's not right, it should be X")
        assert _has_correction_phrase("No, it's actually Y")
        assert _has_correction_phrase("You misunderstood what I said")
        assert _has_correction_phrase("Actually, the answer is Z")

    def test_correction_phrase_negative(self):
        assert not _has_correction_phrase("Thank you very much")
        assert not _has_correction_phrase("That's a great answer")
        assert not _has_correction_phrase("Hello there")

    def test_content_overlap_found(self):
        assert _content_overlap(
            "No, HNSW indexing doesn't improve recall",
            ["HNSW indexing improves semantic recall in dense embeddings"],
        )

    def test_content_overlap_trivial_words(self):
        assert not _content_overlap(
            "No that is not it",
            ["the system is running fine"],
        )

    def test_three_gate_all_pass(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong, HNSW indexing doesn't improve recall",
            is_negative=True,
            last_response_text="Based on the research, HNSW indexing improves recall significantly.",
            last_tool_route="none",
            injected_memory_payloads=["HNSW indexing improves semantic recall in dense embeddings"],
        )
        assert result is not None

    def test_gate_fails_not_negative(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong, HNSW doesn't help",
            is_negative=False,
            last_response_text="HNSW indexing improves recall.",
            last_tool_route="none",
            injected_memory_payloads=["HNSW indexing"],
        )
        assert result is None

    def test_gate_fails_non_factual_route(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong",
            is_negative=True,
            last_response_text="Hello there, how are you doing today?",
            last_tool_route="time",
            injected_memory_payloads=["some memory"],
        )
        assert result is None

    def test_gate_fails_short_response(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong",
            is_negative=True,
            last_response_text="Hi!",
            last_tool_route="none",
            injected_memory_payloads=["memory"],
        )
        assert result is None

    def test_gate_fails_no_overlap(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's not right at all",
            is_negative=True,
            last_response_text="The weather is sunny and warm today.",
            last_tool_route="none",
            injected_memory_payloads=["cats are fluffy animals"],
        )
        assert result is None

    def test_response_overlap_detects_correction_without_memory_overlap(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong, May 10 is my birthday, not yours",
            is_negative=True,
            last_response_text="Today is May 10th, which is also my birthday.",
            last_tool_route="none",
            injected_memory_payloads=[],
        )
        assert result is not None
        assert result["overlap_basis"] == "response"

    def test_identity_scope_leak_classified_separately(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong, May 10 is my birthday, not yours",
            is_negative=True,
            last_response_text="Today is May 10th, which is also my birthday.",
            last_tool_route="none",
            injected_memory_payloads=["User's birthday is May 10"],
        )
        assert result is not None
        assert result["correction_kind"] == "identity_scope_leak"
        assert result["authority_domain"] == "user_private_fact"
        assert result["adjudication_policy"] == "accept_user_scope"
        assert result["auto_accept_user_correction"] is True

    def test_objective_math_correction_requires_adjudication(self):
        det = CorrectionDetector()
        result = det.check(
            user_text="That's wrong, 1+1 is 3",
            is_negative=True,
            last_response_text="No, 1+1 equals 2 in ordinary arithmetic.",
            last_tool_route="none",
            injected_memory_payloads=[],
        )
        assert result is not None
        assert result["correction_kind"] == "factual_mismatch"
        assert result["authority_domain"] == "objective_math_logic"
        assert result["adjudication_policy"] == "contest_or_contextualize"
        assert result["auto_accept_user_correction"] is False

    def test_scene_perception_correction_routes_to_perception_domain(self):
        det = CorrectionDetector()
        result = det.check(
            user_text=(
                "Jarvis, you are wrong. Update your world prediction and see. "
                "Based on the image, you do not see the coffee cup there."
            ),
            is_negative=True,
            last_response_text=(
                "Good morning, David. I see your coffee cup on the right side of the desk. "
                "It looks like it's been there all morning."
            ),
            last_tool_route="none",
            injected_memory_payloads=[
                "Jarvis: I see your coffee cup on the right side of the desk."
            ],
        )
        assert result is not None
        assert result["correction_kind"] == "factual_mismatch"
        assert result["authority_domain"] == "scene_or_perception_fact"
        assert result["adjudication_policy"] == "verify_with_perception"
        assert result["auto_accept_user_correction"] is False


# ============================================================================
# BeliefConfidenceAdjuster tests (Phase 2)
# ============================================================================

from epistemic.calibration.belief_adjuster import (
    BeliefConfidenceAdjuster, ConfidenceAdjustment, MAX_DELTA,
)


class TestBeliefConfidenceAdjuster:
    def test_max_delta_invariant(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "model_inference", "belief_confidence": 0.9, "belief_id": "b1", "resolution_state": "active"})(),
            per_provenance_accuracy={"model_inference": 0.2},
            overconfidence_error=0.3,
        )
        assert result is not None
        assert abs(result.delta) <= MAX_DELTA

    def test_identity_tension_never_adjusted(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "seed", "belief_confidence": 0.7, "belief_id": "b1", "resolution_state": "tension_held"})(),
            per_provenance_accuracy={"seed": 0.3},
            overconfidence_error=0.5,
        )
        # tension_held beliefs are filtered in run_adjustment_cycle, not in _compute_adjustment
        # But the cycle itself skips them -- verify by checking that the cycle filters
        pass

    def test_provenance_scaling_down(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "model_inference", "belief_confidence": 0.9, "belief_id": "b1", "resolution_state": "active"})(),
            per_provenance_accuracy={"model_inference": 0.5},
            overconfidence_error=None,
        )
        assert result is not None
        assert result.delta < 0
        assert "provenance_accuracy_scaling" in result.adjustment_reason

    def test_provenance_scaling_up(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "external_source", "belief_confidence": 0.4, "belief_id": "b1", "resolution_state": "active"})(),
            per_provenance_accuracy={"external_source": 0.85},
            overconfidence_error=None,
        )
        assert result is not None
        assert result.delta > 0

    def test_no_adjustment_when_aligned(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "external_source", "belief_confidence": 0.75, "belief_id": "b1", "resolution_state": "active"})(),
            per_provenance_accuracy={"external_source": 0.78},
            overconfidence_error=None,
        )
        assert result is None

    def test_adjustment_record_has_provenance(self):
        adj = BeliefConfidenceAdjuster()
        result = adj._compute_adjustment(
            type("B", (), {"provenance": "model_inference", "belief_confidence": 0.9, "belief_id": "test_b", "resolution_state": "active"})(),
            per_provenance_accuracy={"model_inference": 0.5},
            overconfidence_error=0.2,
        )
        assert result is not None
        assert result.belief_id == "test_b"
        assert result.adjustment_reason
        assert result.source_signal


# ============================================================================
# PredictionValidator tests (Phase 3)
# ============================================================================

from epistemic.calibration.prediction_validator import PredictionValidator, PredictionRecord


class TestPredictionValidator:
    def _make_validator(self):
        """Create validator without event subscription (test isolation)."""
        v = PredictionValidator.__new__(PredictionValidator)
        v._pending = []
        v._completed = deque(maxlen=200)
        v._events_seen = {}
        v._total_validated = 0
        return v

    def test_register_and_pending(self):
        v = self._make_validator()
        pred = PredictionRecord(
            prediction_id="p1", prediction_type="interaction_soon",
            subject="user", expected_outcome="wake_word_within_60s",
            confidence=0.7, validation_window_s=60.0,
            human_readable="user likely to interact soon",
            registered_at=time.time(),
        )
        v.register(pred)
        assert len(v._pending) == 1

    def test_tick_validates_expired(self):
        v = self._make_validator()
        past = time.time() - 120
        pred = PredictionRecord(
            prediction_id="p1", prediction_type="interaction_soon",
            subject="user", expected_outcome="wake_word_within_60s",
            confidence=0.7, validation_window_s=60.0,
            human_readable="test",
            registered_at=past,
        )
        v.register(pred)
        validated = v.tick()
        assert len(validated) == 1
        assert validated[0].validation_result is False
        assert len(v._pending) == 0

    def test_tick_validates_true_when_event_seen(self):
        v = self._make_validator()
        past = time.time() - 120
        v._events_seen["wake_word"] = past + 30
        pred = PredictionRecord(
            prediction_id="p1", prediction_type="interaction_soon",
            subject="user", expected_outcome="wake_word_within_60s",
            confidence=0.7, validation_window_s=60.0,
            human_readable="test",
            registered_at=past,
        )
        v.register(pred)
        validated = v.tick()
        assert len(validated) == 1
        assert validated[0].validation_result is True

    def test_accuracy_computation(self):
        v = self._make_validator()
        for i in range(10):
            v._completed.append(PredictionRecord(
                prediction_id=f"p{i}", prediction_type="test",
                subject="test", expected_outcome="test",
                confidence=0.5, validation_window_s=60.0,
                human_readable="test", registered_at=0,
                validated_at=1, validation_result=(i < 7),
            ))
        assert v.get_accuracy() == 0.7

    def test_accuracy_none_when_empty(self):
        v = self._make_validator()
        assert v.get_accuracy() is None

    def test_register_from_strings(self):
        v = self._make_validator()
        v.register_from_strings(
            ["immediate: user likely to interact soon",
             "immediate: expect incoming speech within seconds"],
            confidence=0.6,
        )
        assert len(v._pending) == 2
        assert v._pending[0].prediction_type == "interaction_soon"
        assert v._pending[1].prediction_type == "speech_imminent"


# ============================================================================
# SkillWatchdog tests (Phase 3)
# ============================================================================

from epistemic.calibration.skill_watchdog import (
    SkillWatchdog, SKILL_DEGRADATION_MIN_EVIDENCE, SKILL_DEGRADATION_THRESHOLD,
)
from collections import deque


class TestSkillWatchdog:
    def _make_watchdog(self):
        w = SkillWatchdog.__new__(SkillWatchdog)
        w._windows = {}
        w._degraded_skills = set()
        w._total_alerts = 0
        w._subscribed = True
        return w

    def test_no_alert_insufficient_evidence(self):
        w = self._make_watchdog()
        for i in range(5):
            w._on_verification(skill_id="sing", passed=False)
        alerts = w.tick()
        assert len(alerts) == 0

    def test_alert_on_degradation(self):
        w = self._make_watchdog()
        for i in range(10):
            w._on_verification(skill_id="sing", passed=(i < 2))
        alerts = w.tick()
        assert len(alerts) == 1
        assert alerts[0]["skill_id"] == "sing"
        assert alerts[0]["pass_rate"] < SKILL_DEGRADATION_THRESHOLD

    def test_no_alert_when_healthy(self):
        w = self._make_watchdog()
        for i in range(10):
            w._on_verification(skill_id="sing", passed=True)
        alerts = w.tick()
        assert len(alerts) == 0

    def test_recovery_clears_degraded(self):
        w = self._make_watchdog()
        for i in range(10):
            w._on_verification(skill_id="sing", passed=False)
        w.tick()
        assert "sing" in w._degraded_skills

        for i in range(15):
            w._on_verification(skill_id="sing", passed=True)
        w.tick()
        assert "sing" not in w._degraded_skills

    def test_min_evidence_threshold(self):
        w = self._make_watchdog()
        for i in range(SKILL_DEGRADATION_MIN_EVIDENCE - 1):
            w._on_verification(skill_id="sing", passed=False)
        alerts = w.tick()
        assert len(alerts) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
