"""Tests for Layer 8 Active-Lite: QuarantinePressure + friction contract."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from epistemic.quarantine.pressure import (
    CATEGORY_POLICY,
    ELEVATED_THRESHOLD,
    HIGH_THRESHOLD,
    PRESSURE_NORMAL,
    QUARANTINE_SUSPECT_TAG,
    QuarantinePressure,
    PressureState,
    get_quarantine_pressure,
)
from epistemic.quarantine.scorer import (
    CATEGORY_CONTRADICTION,
    CATEGORY_IDENTITY,
    CATEGORY_MANIPULATION,
    CATEGORY_MEMORY,
    CATEGORY_CALIBRATION,
    QuarantineSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(category: str, score: float, chronic: bool = False) -> QuarantineSignal:
    return QuarantineSignal(
        score=score, category=category, reason="test",
        is_chronic=chronic, timestamp=time.time(),
    )


def _mem(**kwargs):
    defaults = dict(
        identity_needs_resolution=False,
        identity_confidence=0.8,
        provenance="observed",
        tags=(),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# PressureState invariants
# ---------------------------------------------------------------------------

class TestPressureInvariants:
    def test_zero_pressure_at_init(self):
        qp = QuarantinePressure()
        assert qp.current.composite == 0.0
        assert not qp.current.elevated
        assert not qp.current.high

    def test_pressure_always_in_0_1(self):
        qp = QuarantinePressure()
        signals = [_sig(CATEGORY_IDENTITY, 1.0)] * 5
        state = qp.update(signals, chronic_count=5)
        assert 0.0 <= state.composite <= 1.0

    def test_pressure_decays_without_signals(self):
        qp = QuarantinePressure()
        qp.update([_sig(CATEGORY_MEMORY, 0.8)], chronic_count=0)
        p1 = qp.current.composite
        qp.update([], chronic_count=0)
        p2 = qp.current.composite
        assert p2 < p1, "Pressure should decay when no signals arrive"

    def test_frozen_snapshot(self):
        p = PRESSURE_NORMAL
        with pytest.raises(AttributeError):
            p.composite = 0.5  # type: ignore[misc]

    def test_elevated_threshold(self):
        qp = QuarantinePressure()
        sigs = [_sig(c, 0.95) for c in (
            CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
            CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
        )]
        for _ in range(10):
            qp.update(sigs, 0)
        assert qp.current.elevated

    def test_high_threshold_requires_strong_signal(self):
        qp = QuarantinePressure()
        qp.update([_sig(CATEGORY_IDENTITY, 0.2)], 0)
        assert not qp.current.high

    def test_chronic_bonus_increases_pressure(self):
        qp = QuarantinePressure()
        qp.update([_sig(CATEGORY_IDENTITY, 0.5, chronic=True)], chronic_count=1)
        p_with_chronic = qp.current.by_category[CATEGORY_IDENTITY]

        qp2 = QuarantinePressure()
        qp2.update([_sig(CATEGORY_IDENTITY, 0.5, chronic=False)], chronic_count=0)
        p_without = qp2.current.by_category[CATEGORY_IDENTITY]
        assert p_with_chronic > p_without


# ---------------------------------------------------------------------------
# Category policy table
# ---------------------------------------------------------------------------

class TestCategoryPolicy:
    def test_all_categories_present(self):
        expected = {CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
                    CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION}
        assert set(CATEGORY_POLICY.keys()) == expected

    def test_identity_is_taggable(self):
        assert CATEGORY_POLICY[CATEGORY_IDENTITY].taggable is True
        assert CATEGORY_POLICY[CATEGORY_IDENTITY].belief_block is True

    def test_contradiction_not_taggable(self):
        assert CATEGORY_POLICY[CATEGORY_CONTRADICTION].taggable is False
        assert CATEGORY_POLICY[CATEGORY_CONTRADICTION].belief_block is False

    def test_memory_match_for_identity(self):
        mem_low = _mem(identity_confidence=0.3)
        mem_high = _mem(identity_confidence=0.8)
        policy = CATEGORY_POLICY[CATEGORY_IDENTITY]
        assert policy.memory_match(mem_low) is True
        assert policy.memory_match(mem_high) is False

    def test_memory_match_for_manipulation(self):
        mem_claim = _mem(provenance="user_claim")
        mem_obs = _mem(provenance="observed")
        policy = CATEGORY_POLICY[CATEGORY_MANIPULATION]
        assert policy.memory_match(mem_claim) is True
        assert policy.memory_match(mem_obs) is False


# ---------------------------------------------------------------------------
# Memory tagging
# ---------------------------------------------------------------------------

class TestMemoryTagging:
    def test_no_tag_at_normal_pressure(self):
        qp = QuarantinePressure()
        mem = _mem()
        should_tag, cats = qp.should_tag_memory(mem)
        assert not should_tag
        assert cats == []

    def test_tag_at_elevated_for_matching_category(self):
        qp = QuarantinePressure()
        sigs = [_sig(c, 0.95) for c in (
            CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
            CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
        )]
        for _ in range(10):
            qp.update(sigs, 0)
        assert qp.current.elevated
        mem = _mem(identity_confidence=0.3)
        should_tag, cats = qp.should_tag_memory(mem)
        assert should_tag
        assert CATEGORY_IDENTITY in cats

    def test_no_tag_for_non_matching_memory(self):
        """Elevated pressure via non-taggable categories → memory not tagged."""
        qp = QuarantinePressure()
        # Only contradiction + calibration have pressure (both non-taggable)
        # Plus identity (taggable but memory won't match)
        for _ in range(10):
            qp.update([
                _sig(CATEGORY_IDENTITY, 0.95),
                _sig(CATEGORY_CONTRADICTION, 0.95),
                _sig(CATEGORY_CALIBRATION, 0.95),
            ], 0)
        assert qp.current.elevated
        mem = _mem(identity_confidence=0.9, provenance="observed")
        should_tag, cats = qp.should_tag_memory(mem)
        assert not should_tag

    def test_weight_multiplier_decreases_under_pressure(self):
        qp = QuarantinePressure()
        assert qp.weight_multiplier() == 1.0
        sigs = [_sig(c, 0.95) for c in (
            CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
            CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
        )]
        for _ in range(10):
            qp.update(sigs, 0)
        assert qp.current.elevated
        assert qp.weight_multiplier() < 1.0
        assert qp.weight_multiplier() >= 0.6

    def test_memories_tagged_counter(self):
        qp = QuarantinePressure()
        sigs = [_sig(c, 0.95) for c in (
            CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
            CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
        )]
        for _ in range(10):
            qp.update(sigs, 0)
        assert qp.current.elevated
        mem = _mem()  # memory_match for CATEGORY_MEMORY is lambda m: True
        qp.should_tag_memory(mem)
        assert qp.memories_tagged >= 1


# ---------------------------------------------------------------------------
# Friction contract
# ---------------------------------------------------------------------------

class TestFrictionContract:
    def test_mutation_risk_addon_zero_at_normal(self):
        qp = QuarantinePressure()
        assert qp.mutation_risk_addon() == 0.0

    def test_mutation_risk_addon_positive_under_pressure(self):
        qp = QuarantinePressure()
        for _ in range(10):
            qp.update([_sig(c, 0.95) for c in (
                CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
                CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
            )], 0)
        assert qp.current.composite > 0.0
        assert qp.mutation_risk_addon() > 0.0

    def test_mutation_rate_factor_at_high(self):
        qp = QuarantinePressure()
        for _ in range(20):
            qp.update([_sig(CATEGORY_IDENTITY, 1.0), _sig(CATEGORY_MEMORY, 1.0)], 0)
        if qp.current.high:
            cap, cd = qp.mutation_rate_factor()
            assert cap is not None and cap < 12
            assert cd is not None and cd > 180

    def test_mutation_rate_factor_default_at_normal(self):
        qp = QuarantinePressure()
        cap, cd = qp.mutation_rate_factor()
        assert cap is None
        assert cd is None

    def test_policy_promotion_no_block_at_normal(self):
        qp = QuarantinePressure()
        f = qp.policy_promotion_friction()
        assert f["block"] is False
        assert f["allow_rollback"] is False
        assert f["win_threshold_delta"] == 0.0

    def test_policy_promotion_graduated_at_elevated(self):
        qp = QuarantinePressure()
        for _ in range(10):
            qp.update([_sig(c, 0.95) for c in (
                CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
                CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
            )], 0)
        assert qp.current.elevated
        f = qp.policy_promotion_friction()
        if not qp.current.high:
            assert f["block"] is False
        assert f["win_threshold_delta"] > 0.0

    def test_world_model_no_cap_at_normal(self):
        qp = QuarantinePressure()
        f = qp.world_model_promotion_friction()
        assert f["max_level"] is None
        assert f["accuracy_delta"] == 0.0

    def test_graph_support_gate_zero_at_normal(self):
        qp = QuarantinePressure()
        assert qp.graph_support_gate_delta() == 0.0

    def test_graph_support_gate_positive_under_pressure(self):
        qp = QuarantinePressure()
        for _ in range(10):
            qp.update([_sig(c, 0.95) for c in (
                CATEGORY_IDENTITY, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
                CATEGORY_CONTRADICTION, CATEGORY_CALIBRATION,
            )], 0)
        assert qp.current.elevated
        assert qp.graph_support_gate_delta() > 0.0


# ---------------------------------------------------------------------------
# Belief exclusion via tag
# ---------------------------------------------------------------------------

class TestBeliefExclusion:
    def test_suspect_tag_blocks_belief_extraction(self):
        from epistemic.contradiction_engine import ContradictionEngine
        engine = ContradictionEngine.__new__(ContradictionEngine)
        mem = _mem(type="factual_knowledge", weight=0.5, tags=("quarantine:suspect",))
        assert engine._is_belief_eligible(mem) is False

    def test_normal_memory_passes_belief_extraction(self):
        from epistemic.contradiction_engine import ContradictionEngine
        engine = ContradictionEngine.__new__(ContradictionEngine)
        mem = _mem(type="factual_knowledge", weight=0.5, tags=())
        assert engine._is_belief_eligible(mem) is True


# ---------------------------------------------------------------------------
# Dashboard snapshot
# ---------------------------------------------------------------------------

class TestDashboardSnapshot:
    def test_snapshot_keys(self):
        qp = QuarantinePressure()
        snap = qp.get_snapshot()
        assert "composite" in snap
        assert "band" in snap
        assert "by_category" in snap
        assert "memories_tagged" in snap
        assert "promotions_blocked" in snap

    def test_snapshot_band_values(self):
        qp = QuarantinePressure()
        assert qp.get_snapshot()["band"] == "normal"
