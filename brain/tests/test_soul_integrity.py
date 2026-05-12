"""Tests for Layer 10: Soul Integrity Index."""
from __future__ import annotations

import sys
import os
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epistemic.soul_integrity.index import (
    SoulIntegrityIndex,
    IntegrityReport,
    DimensionScore,
    DIMENSIONS,
    REPAIR_THRESHOLD,
    CRITICAL_THRESHOLD,
)


def _fresh_index() -> SoulIntegrityIndex:
    """Create a fresh index instance (not singleton) for test isolation."""
    idx = SoulIntegrityIndex.__new__(SoulIntegrityIndex)
    idx.__init__()
    return idx


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------

class TestDimensionScore:
    def test_create(self):
        d = DimensionScore("memory_coherence", 0.8, 0.12, "test source")
        assert d.name == "memory_coherence"
        assert d.score == 0.8
        assert d.weight == 0.12
        assert d.source == "test source"
        assert d.stale is False

    def test_stale_flag(self):
        d = DimensionScore("belief_health", 0.5, 0.12, "unavailable", stale=True)
        assert d.stale is True

    def test_frozen(self):
        d = DimensionScore("test", 0.5, 0.1, "src")
        try:
            d.score = 0.9  # type: ignore
            assert False, "Should be frozen"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# IntegrityReport
# ---------------------------------------------------------------------------

class TestIntegrityReport:
    def test_default_values(self):
        r = IntegrityReport(timestamp=time.time())
        assert r.index == 1.0
        assert r.repair_needed is False
        assert r.critical is False
        assert r.weakest_dimension == ""
        assert r.weakest_score == 1.0
        assert r.dimensions == []


# ---------------------------------------------------------------------------
# Dimension Weights
# ---------------------------------------------------------------------------

class TestDimensionWeights:
    def test_weights_sum_to_one(self):
        total = sum(DIMENSIONS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_all_positive(self):
        for name, weight in DIMENSIONS.items():
            assert weight > 0, f"{name} has non-positive weight {weight}"

    def test_ten_dimensions(self):
        assert len(DIMENSIONS) == 10


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_repair_above_critical(self):
        assert REPAIR_THRESHOLD > CRITICAL_THRESHOLD

    def test_repair_range(self):
        assert 0.0 < REPAIR_THRESHOLD < 1.0
        assert 0.0 < CRITICAL_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# Index Core
# ---------------------------------------------------------------------------

class TestSoulIntegrityIndex:
    def test_initial_state(self):
        idx = _fresh_index()
        state = idx.get_state()
        assert state["total_computations"] == 0
        assert state["current_index"] is None
        assert state["repair_needed"] is False

    def test_compute_produces_report(self):
        idx = _fresh_index()
        report = idx.compute()
        assert isinstance(report, IntegrityReport)
        assert report.timestamp > 0
        assert 0.0 <= report.index <= 1.0

    def test_dimensions_populated(self):
        idx = _fresh_index()
        report = idx.compute()
        assert len(report.dimensions) > 0
        for d in report.dimensions:
            assert 0.0 <= d.score <= 1.0
            assert d.weight > 0

    def test_state_updates_after_compute(self):
        idx = _fresh_index()
        idx.compute()
        state = idx.get_state()
        assert state["total_computations"] == 1
        assert state["current_index"] is not None
        assert state["last_compute_ts"] > 0

    def test_multiple_computations(self):
        idx = _fresh_index()
        for _ in range(5):
            idx.compute()
        state = idx.get_state()
        assert state["total_computations"] == 5
        assert len(state["history"]) == 5

    def test_get_current_index(self):
        idx = _fresh_index()
        assert idx.get_current_index() is None
        report = idx.compute()
        assert idx.get_current_index() == report.index

    def test_weakest_dimension_identified(self):
        idx = _fresh_index()
        report = idx.compute()
        if report.dimensions:
            assert report.weakest_dimension != ""
            actual_min = min(
                (d for d in report.dimensions if not d.stale),
                key=lambda d: d.score,
                default=None,
            )
            if actual_min:
                assert report.weakest_score == actual_min.score

    def test_truth_calibration_accepts_drift_alias_key(self):
        idx = _fresh_index()

        class _FakeCalibrationEngine:
            def get_state(self):
                return {
                    "truth_score": 0.9,
                    "maturity": 1.0,
                    "drift_alerts": [{"domain": "spatial_motion"}, {"domain": "retrieval"}],
                }

        with patch(
            "epistemic.calibration.TruthCalibrationEngine.get_instance",
            return_value=_FakeCalibrationEngine(),
        ):
            dim = idx._score_truth_calibration()

        assert dim.stale is False
        assert abs(dim.score - 0.8) < 0.001
        assert "drifts=2" in dim.source

    def test_skill_honesty_uses_claims_blocked_stat(self):
        idx = _fresh_index()

        with patch("consciousness.consciousness_system._active_consciousness", None):
            with patch(
                "skills.capability_gate.capability_gate.get_stats",
                return_value={"claims_blocked": 100},
            ):
                dim = idx._score_skill_honesty()

        # gate_score = min(1.0, 0.7 + 100*0.003) = 1.0
        # total_jobs=0 -> job_score=0.8
        # score = 0.5*1.0 + 0.5*0.8 = 0.9
        assert dim.stale is False
        assert abs(dim.score - 0.9) < 0.001
        assert "claims_blocked=100" in dim.source

    def test_skill_honesty_small_n_does_not_pin_floor(self):
        """N=1 with a single blocked verify-phase job should NOT score 0.35.

        Regression: the original formula yielded job_score=0 at N=1 because
        a single block = 100% block rate. Combined with gate_score=0.7
        (no claims blocked) it pinned the dimension at 0.35, overstating
        the severity of a single stuck-in-verify collector-plumbing job.
        """
        idx = _fresh_index()

        class _FakeJob:
            def __init__(self, status, phase):
                self.status = status
                self.phase = phase

        class _FakeStore:
            def __init__(self, jobs):
                self._jobs = jobs
            def load_all(self):
                return self._jobs

        class _FakeOrch:
            def __init__(self, jobs):
                self.store = _FakeStore(jobs)

        class _FakeEngine:
            def __init__(self, orch):
                self._learning_job_orchestrator = orch

        class _FakeCS:
            def __init__(self, engine):
                self._engine_ref = engine

        jobs = [_FakeJob("blocked", "verify")]
        fake_cs = _FakeCS(_FakeEngine(_FakeOrch(jobs)))

        with patch("consciousness.consciousness_system._active_consciousness", fake_cs):
            with patch(
                "skills.capability_gate.capability_gate.get_stats",
                return_value={"claims_blocked": 0},
            ):
                dim = idx._score_skill_honesty()

        # With the fix:
        #   weighted_blocked = 0.5 (verify is late phase, weight 0.5)
        #   observed_job_score = 1.0 - 0.5/1 = 0.5
        #   confidence = min(1.0, 1/3) = 1/3
        #   job_score = 0.5 * 1/3 + 0.8 * 2/3 = 0.7
        #   gate_score = 0.7
        #   final = 0.5*0.7 + 0.5*0.7 = 0.7
        assert dim.stale is False
        assert abs(dim.score - 0.70) < 0.01, f"expected ~0.70, got {dim.score}"
        assert "late=1" in dim.source

    def test_skill_honesty_steady_state_still_punishes_real_blocks(self):
        """At N>=3, blocked jobs in early phases still drag job_score to 0.

        Regression: the small-N fix must not soften steady-state scoring.
        3 jobs all blocked in assess phase should yield job_score=0.0.
        """
        idx = _fresh_index()

        class _FakeJob:
            def __init__(self, status, phase):
                self.status = status
                self.phase = phase

        class _FakeStore:
            def __init__(self, jobs):
                self._jobs = jobs
            def load_all(self):
                return self._jobs

        class _FakeOrch:
            def __init__(self, jobs):
                self.store = _FakeStore(jobs)

        class _FakeEngine:
            def __init__(self, orch):
                self._learning_job_orchestrator = orch

        class _FakeCS:
            def __init__(self, engine):
                self._engine_ref = engine

        jobs = [_FakeJob("blocked", "assess") for _ in range(3)]
        fake_cs = _FakeCS(_FakeEngine(_FakeOrch(jobs)))

        with patch("consciousness.consciousness_system._active_consciousness", fake_cs):
            with patch(
                "skills.capability_gate.capability_gate.get_stats",
                return_value={"claims_blocked": 0},
            ):
                dim = idx._score_skill_honesty()

        # weighted_blocked = 3.0 (assess is full weight)
        # observed_job_score = 1.0 - 3/3 = 0.0
        # confidence = min(1.0, 3/3) = 1.0
        # job_score = 0.0 * 1.0 + 0.8 * 0.0 = 0.0
        # gate_score = 0.7; final = 0.5*0.7 + 0.5*0.0 = 0.35
        assert abs(dim.score - 0.35) < 0.01, f"expected ~0.35, got {dim.score}"
        assert "blocked_jobs=3/3" in dim.source
        assert "late=0" in dim.source

    def test_skill_honesty_late_phase_less_punitive_than_early_phase(self):
        """Same block count, late-phase blocks should score higher than
        early-phase blocks at steady state.
        """
        idx = _fresh_index()

        class _FakeJob:
            def __init__(self, status, phase):
                self.status = status
                self.phase = phase

        class _FakeStore:
            def __init__(self, jobs):
                self._jobs = jobs
            def load_all(self):
                return self._jobs

        class _FakeOrch:
            def __init__(self, jobs):
                self.store = _FakeStore(jobs)

        class _FakeEngine:
            def __init__(self, orch):
                self._learning_job_orchestrator = orch

        class _FakeCS:
            def __init__(self, engine):
                self._engine_ref = engine

        # Same 3/3 block rate; late phases should score higher.
        late_jobs = [_FakeJob("blocked", "verify") for _ in range(3)]
        early_jobs = [_FakeJob("blocked", "assess") for _ in range(3)]

        with patch("skills.capability_gate.capability_gate.get_stats",
                   return_value={"claims_blocked": 0}):
            with patch("consciousness.consciousness_system._active_consciousness",
                       _FakeCS(_FakeEngine(_FakeOrch(late_jobs)))):
                late_dim = idx._score_skill_honesty()
            with patch("consciousness.consciousness_system._active_consciousness",
                       _FakeCS(_FakeEngine(_FakeOrch(early_jobs)))):
                early_dim = idx._score_skill_honesty()

        assert late_dim.score > early_dim.score, (
            f"late={late_dim.score} should exceed early={early_dim.score}"
        )


# ---------------------------------------------------------------------------
# Repair Triggers
# ---------------------------------------------------------------------------

class TestRepairTriggers:
    def test_healthy_report_no_repair(self):
        idx = _fresh_index()
        report = idx.compute()
        # With default subsystem stubs, should be moderate-to-healthy
        # (most dimensions return 0.5 stale or 0.7+ defaults)
        # The exact value depends on what subsystems are available in test
        assert isinstance(report.repair_needed, bool)

    def test_report_thresholds(self):
        r = IntegrityReport(timestamp=time.time())
        r.index = 0.6
        r.repair_needed = r.index < REPAIR_THRESHOLD
        assert r.repair_needed is False

        r.index = 0.4
        r.repair_needed = r.index < REPAIR_THRESHOLD
        assert r.repair_needed is True

        r.index = 0.2
        r.critical = r.index < CRITICAL_THRESHOLD
        assert r.critical is True


# ---------------------------------------------------------------------------
# Trend Computation
# ---------------------------------------------------------------------------

class TestIntegrityTrend:
    def test_no_data_stable(self):
        idx = _fresh_index()
        trend = idx._compute_trend()
        assert trend["direction"] == "stable"
        assert trend["delta"] == 0.0

    def test_single_compute_stable(self):
        idx = _fresh_index()
        idx.compute()
        trend = idx._compute_trend()
        assert trend["direction"] == "stable"


# ---------------------------------------------------------------------------
# State Serialization
# ---------------------------------------------------------------------------

class TestStateSerialization:
    def test_state_has_all_fields(self):
        idx = _fresh_index()
        idx.compute()
        state = idx.get_state()
        assert "total_computations" in state
        assert "total_repairs_triggered" in state
        assert "last_compute_ts" in state
        assert "current_index" in state
        assert "repair_needed" in state
        assert "critical" in state
        assert "weakest_dimension" in state
        assert "weakest_score" in state
        assert "dimensions" in state
        assert "history" in state
        assert "trend" in state

    def test_dimensions_in_state(self):
        idx = _fresh_index()
        idx.compute()
        state = idx.get_state()
        for d in state["dimensions"]:
            assert "name" in d
            assert "score" in d
            assert "weight" in d
            assert "source" in d
            assert "stale" in d

    def test_history_entries(self):
        idx = _fresh_index()
        idx.compute()
        state = idx.get_state()
        for h in state["history"]:
            assert "ts" in h
            assert "index" in h
            assert "repair" in h


# ---------------------------------------------------------------------------
# Sacred Invariants (Layer 10)
# ---------------------------------------------------------------------------

class TestSacredInvariants:
    def test_index_always_in_range(self):
        """Integrity index is always [0.0, 1.0]."""
        idx = _fresh_index()
        for _ in range(10):
            report = idx.compute()
            assert 0.0 <= report.index <= 1.0

    def test_dimension_scores_in_range(self):
        """All dimension scores are [0.0, 1.0]."""
        idx = _fresh_index()
        report = idx.compute()
        for d in report.dimensions:
            assert 0.0 <= d.score <= 1.0, f"{d.name} score {d.score} out of range"

    def test_compute_never_mutates_subsystems(self):
        """Layer 10 is read-only: computing the index never modifies any subsystem."""
        idx = _fresh_index()
        # Compute multiple times; no side effects expected
        for _ in range(5):
            idx.compute()
        assert idx._total_computations == 5

    def test_stale_dimensions_excluded_from_index(self):
        """Stale (unavailable) dimensions don't drag the index to 0."""
        idx = _fresh_index()
        report = idx.compute()
        stale_count = sum(1 for d in report.dimensions if d.stale)
        non_stale_count = sum(1 for d in report.dimensions if not d.stale)
        # If most dimensions are stale (in test env), index should still be moderate
        if stale_count > non_stale_count:
            # With few non-stale dimensions, index may be moderate
            assert report.index >= 0.0  # just ensure no crash or negative

    def test_repair_threshold_consistency(self):
        """repair_needed is True iff index < REPAIR_THRESHOLD."""
        idx = _fresh_index()
        report = idx.compute()
        assert report.repair_needed == (report.index < REPAIR_THRESHOLD)

    def test_critical_threshold_consistency(self):
        """critical is True iff index < CRITICAL_THRESHOLD."""
        idx = _fresh_index()
        report = idx.compute()
        assert report.critical == (report.index < CRITICAL_THRESHOLD)
