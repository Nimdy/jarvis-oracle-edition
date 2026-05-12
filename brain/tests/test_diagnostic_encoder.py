"""Tests for the DiagnosticEncoder hemisphere feature encoder."""

import pytest
from hemisphere.diagnostic_encoder import (
    DiagnosticEncoder,
    FEATURE_DIM,
    _DETECTOR_CATEGORIES,
)


def _full_snapshot() -> dict:
    return {
        "health": {"overall": 0.42, "checks": 15, "worst_score": 0.31},
        "reasoning": {"overall": 0.30, "coherence": 0.45, "depth": 0.25, "thought_count": 30},
        "confidence": {"current": 0.6, "volatility": 0.35, "trend": -0.2},
        "latency": {"total": 10, "slow_gt_5s": 4},
        "event_bus": {"emitted": 500, "errors": 30},
        "tick": {"p95_ms": 95.3},
    }


def _full_context() -> dict:
    return {
        "uptime_s": 7200.0,
        "quarantine_pressure": 0.15,
        "soul_integrity": 0.85,
        "mode": "conversational",
        "evolution_stage": 2,
        "consciousness_stage": 1,
        "health_trend_slope": -0.1,
        "mutations_last_hour": 3,
        "active_learning_jobs": 2,
        "improvements_today": 1,
        "last_improvement_age_s": 3600.0,
        "sandbox_pass_rate": 0.75,
        "friction_rate": 0.05,
        "correction_count": 2,
        "autonomy_level": 1,
        "target_module_lines": 250,
        "target_import_fanout": 8,
        "target_importers": 12,
        "target_symbol_count": 30,
        "target_recently_modified": True,
        "has_codebase_context": True,
        "friction_severity_high_ratio": 0.4,
        "friction_correction_ratio": 0.3,
        "friction_identity_count": 1,
        "correction_auto_accepted": 2,
        "has_friction_context": True,
    }


def _sample_opportunities() -> list:
    return [
        {
            "type": "health_degraded",
            "fingerprint": "health_degraded:engine:0.42",
            "description": "Health degraded",
            "target_module": "consciousness/engine.py",
            "priority": 4,
            "evidence": [],
            "evidence_detail": {"overall_health": 0.42, "worst_component": "engine", "worst_score": 0.31},
            "sustained_count": 3,
        },
        {
            "type": "tick_performance",
            "fingerprint": "tick_slow:95",
            "description": "Tick p95 high",
            "target_module": "consciousness/kernel.py",
            "priority": 3,
            "evidence": [],
            "evidence_detail": {"p95_ms": 95.3},
            "sustained_count": 2,
        },
    ]


class TestDiagnosticEncoderEncode:
    def test_output_dimension(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), _sample_opportunities(), _full_context())
        assert len(vec) == FEATURE_DIM

    def test_all_values_in_range(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), _sample_opportunities(), _full_context())
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_empty_snapshot(self):
        vec = DiagnosticEncoder.encode({}, [], {})
        assert len(vec) == FEATURE_DIM
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_partial_snapshot(self):
        partial = {"health": {"overall": 0.6, "checks": 5}}
        vec = DiagnosticEncoder.encode(partial, [], {"uptime_s": 1000})
        assert len(vec) == FEATURE_DIM
        assert vec[0] == pytest.approx(0.6)

    def test_health_dims_populated(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[0] == pytest.approx(0.42)
        assert vec[2] == pytest.approx(0.30)
        assert vec[4] == pytest.approx(0.35)

    def test_performance_dims_populated(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[7] == pytest.approx(4 / 10)
        assert vec[9] == pytest.approx(30 / 500)
        assert vec[10] == pytest.approx(95.3 / 200)

    def test_context_dims_populated(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[12] == pytest.approx(0.15)
        assert vec[13] == pytest.approx(0.85)

    def test_detector_pattern_dims(self):
        opps = _sample_opportunities()
        vec = DiagnosticEncoder.encode(_full_snapshot(), opps, _full_context())
        assert vec[28] == pytest.approx(2 / 6)
        assert vec[29] == pytest.approx(4 / 5)
        assert vec[30] == 1.0  # has_health_detector
        assert vec[31] == 1.0  # has_performance_detector (tick_performance)

    def test_no_opportunities_detector_pattern(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[28] == 0.0
        assert vec[29] == 0.0
        assert vec[30] == 0.0
        assert vec[31] == 0.0

    def test_confidence_trend_centered(self):
        snap = {"confidence": {"current": 0.5, "volatility": 0.0, "trend": 0.0}}
        vec = DiagnosticEncoder.encode(snap, [], {})
        assert vec[6] == pytest.approx(0.5)

    def test_confidence_trend_negative(self):
        snap = {"confidence": {"current": 0.5, "volatility": 0.0, "trend": -1.0}}
        vec = DiagnosticEncoder.encode(snap, [], {})
        assert vec[6] == pytest.approx(0.0)


class TestDiagnosticEncoderLabel:
    @pytest.mark.parametrize("det_type,expected_idx", [
        ("health_degraded", 0),
        ("reasoning_decline", 1),
        ("confidence_volatile", 2),
        ("slow_responses", 3),
        ("event_bus_errors", 4),
        ("tick_performance", 5),
    ])
    def test_label_one_hot(self, det_type, expected_idx):
        opp = {
            "type": det_type,
            "fingerprint": f"{det_type}:test",
            "target_module": "test.py",
            "evidence_detail": {},
        }
        label, meta = DiagnosticEncoder.encode_label(opp)
        assert len(label) == len(_DETECTOR_CATEGORIES)
        assert label[expected_idx] == 1.0
        assert sum(label) == 1.0

    def test_unknown_detector_all_zeros(self):
        opp = {"type": "unknown_detector", "evidence_detail": {}}
        label, meta = DiagnosticEncoder.encode_label(opp)
        assert sum(label) == 0.0

    def test_metadata_richness(self):
        opp = {
            "type": "health_degraded",
            "fingerprint": "health_degraded:engine:0.42",
            "target_module": "consciousness/engine.py",
            "sustained_count": 3,
            "evidence_detail": {"worst_component": "engine", "overall_health": 0.42},
        }
        label, meta = DiagnosticEncoder.encode_label(opp)
        assert meta["detector_type"] == "health_degraded"
        assert meta["sustained_count"] == 3
        assert meta["fingerprint"] == "health_degraded:engine:0.42"
        assert meta["top_metric"] == "engine"
        assert meta["module_hint"] == "consciousness/engine.py"

    def test_reasoning_decline_picks_worst_metric(self):
        opp = {
            "type": "reasoning_decline",
            "evidence_detail": {"depth": 0.2, "coherence": 0.4},
        }
        _, meta = DiagnosticEncoder.encode_label(opp)
        assert meta["top_metric"] == "depth"

    def test_slow_responses_top_metric(self):
        opp = {
            "type": "slow_responses",
            "evidence_detail": {"avg_slow_ms": 7500.3},
        }
        _, meta = DiagnosticEncoder.encode_label(opp)
        assert "7500" in meta["top_metric"]


class TestDiagnosticEncoderCodebaseFeatures:
    """Track 4: Codebase structural feature dimensions (32-37)."""

    def test_output_is_43_dims(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), _sample_opportunities(), _full_context())
        assert len(vec) == 43
        assert FEATURE_DIM == 43

    def test_codebase_dims_populated(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[32] == pytest.approx(250 / 500)
        assert vec[33] == pytest.approx(8 / 15)
        assert vec[34] == pytest.approx(12 / 15)
        assert vec[35] == pytest.approx(30 / 50)
        assert vec[36] == 1.0  # target_recently_modified
        assert vec[37] == 1.0  # has_codebase_context

    def test_no_codebase_context_zeros(self):
        ctx = _full_context()
        ctx["has_codebase_context"] = False
        ctx["target_module_lines"] = 0
        ctx["target_import_fanout"] = 0
        ctx["target_importers"] = 0
        ctx["target_symbol_count"] = 0
        ctx["target_recently_modified"] = False
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], ctx)
        assert vec[32] == 0.0
        assert vec[33] == 0.0
        assert vec[34] == 0.0
        assert vec[37] == 0.0

    def test_empty_context_defaults_to_zero(self):
        vec = DiagnosticEncoder.encode({}, [], {})
        for i in range(32, 38):
            assert vec[i] == 0.0

    def test_codebase_dims_clamped(self):
        ctx = _full_context()
        ctx["target_module_lines"] = 2000
        ctx["target_import_fanout"] = 100
        vec = DiagnosticEncoder.encode({}, [], ctx)
        assert vec[32] == 1.0
        assert vec[33] == 1.0


class TestDiagnosticEncoderFrictionFeatures:
    """Track 5: Friction/correction enrichment dimensions (38-42)."""

    def test_friction_dims_populated(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), [], _full_context())
        assert vec[38] == pytest.approx(0.4)
        assert vec[39] == pytest.approx(0.3)
        assert vec[40] == pytest.approx(1 / 5)
        assert vec[41] == pytest.approx(2 / 5)
        assert vec[42] == 1.0  # has_friction_context

    def test_no_friction_context_zeros(self):
        ctx = _full_context()
        ctx["has_friction_context"] = False
        ctx["friction_severity_high_ratio"] = 0.0
        ctx["friction_correction_ratio"] = 0.0
        ctx["friction_identity_count"] = 0
        ctx["correction_auto_accepted"] = 0
        vec = DiagnosticEncoder.encode({}, [], ctx)
        for i in range(38, 43):
            assert vec[i] == 0.0

    def test_friction_dims_clamped(self):
        ctx = _full_context()
        ctx["friction_identity_count"] = 50
        ctx["correction_auto_accepted"] = 50
        vec = DiagnosticEncoder.encode({}, [], ctx)
        assert vec[40] == 1.0
        assert vec[41] == 1.0

    def test_all_new_dims_in_range(self):
        vec = DiagnosticEncoder.encode(_full_snapshot(), _sample_opportunities(), _full_context())
        for i in range(32, 43):
            assert 0.0 <= vec[i] <= 1.0, f"dim {i} out of range: {vec[i]}"


class TestDiagnosticEncoderNoOpportunityLabel:
    """Negative-example label for healthy scans."""

    def test_returns_uniform_distribution(self):
        label, meta = DiagnosticEncoder.encode_no_opportunity_label()
        n = len(_DETECTOR_CATEGORIES)
        assert len(label) == n
        for v in label:
            assert v == pytest.approx(1.0 / n)

    def test_sums_to_one(self):
        label, _ = DiagnosticEncoder.encode_no_opportunity_label()
        assert sum(label) == pytest.approx(1.0)

    def test_metadata_has_no_opportunity_type(self):
        _, meta = DiagnosticEncoder.encode_no_opportunity_label()
        assert meta["detector_type"] == "no_opportunity"
        assert meta["sustained_count"] == 0
        assert meta["fingerprint"] == ""

    def test_label_compatible_with_positive_labels(self):
        """Same length as encode_label output."""
        pos_label, _ = DiagnosticEncoder.encode_label({
            "type": "health_degraded",
            "evidence_detail": {},
        })
        neg_label, _ = DiagnosticEncoder.encode_no_opportunity_label()
        assert len(neg_label) == len(pos_label)
