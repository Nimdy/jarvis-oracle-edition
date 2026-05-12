"""Tests for contradiction engine stable paradox lifecycle + burst collapse."""

import time
import pytest
from unittest.mock import patch
from epistemic.belief_record import BeliefRecord, TensionRecord, BeliefStore
from epistemic.resolution import IdentityTensionResolution, _STABLE_PARADOX_MIN_REVISITS, _STABLE_PARADOX_MIN_MATURATION
from epistemic.contradiction_engine import ContradictionEngine


class TestStableParadox:
    """Tensions with high revisit + maturation should stop being re-examined."""

    def test_stable_paradox_thresholds(self):
        assert _STABLE_PARADOX_MIN_REVISITS == 50
        assert _STABLE_PARADOX_MIN_MATURATION == 0.90

    def test_is_stable_paradox_true(self):
        resolver = IdentityTensionResolution()
        tension = TensionRecord(
            tension_id="ten_test",
            topic="identity:self",
            belief_ids=["b1", "b2"],
            conflict_key="identity_tension:self",
            created_at=time.time() - 86400,
            last_revisited=time.time(),
            revisit_count=100,
            stability=0.9,
            maturation_score=0.95,
        )
        assert resolver._is_stable_paradox(tension) is True

    def test_is_stable_paradox_false_low_revisits(self):
        resolver = IdentityTensionResolution()
        tension = TensionRecord(
            tension_id="ten_test",
            topic="identity:self",
            belief_ids=["b1", "b2"],
            conflict_key="identity_tension:self",
            created_at=time.time(),
            last_revisited=time.time(),
            revisit_count=10,
            stability=0.9,
            maturation_score=0.95,
        )
        assert resolver._is_stable_paradox(tension) is False

    def test_is_stable_paradox_false_low_maturation(self):
        resolver = IdentityTensionResolution()
        tension = TensionRecord(
            tension_id="ten_test",
            topic="identity:self",
            belief_ids=["b1", "b2"],
            conflict_key="identity_tension:self",
            created_at=time.time(),
            last_revisited=time.time(),
            revisit_count=100,
            stability=0.9,
            maturation_score=0.5,
        )
        assert resolver._is_stable_paradox(tension) is False


class TestBurstCollapse:
    """scan_corpus should collapse same-topic comparisons per cycle."""

    def test_get_stable_paradox_belief_ids(self):
        engine = ContradictionEngine()
        tension = TensionRecord(
            tension_id="ten_1",
            topic="identity:self",
            belief_ids=["b1", "b2", "b3"],
            conflict_key="identity_tension:self",
            created_at=time.time() - 86400,
            last_revisited=time.time(),
            revisit_count=200,
            stability=0.9,
            maturation_score=0.95,
        )
        engine._belief_store.add_tension(tension)
        stable = engine._get_stable_paradox_belief_ids()
        assert "b1" in stable
        assert "b2" in stable
        assert "b3" in stable

    def test_stable_beliefs_dont_block_decay(self):
        engine = ContradictionEngine()
        engine._contradiction_debt = 0.5
        engine._last_decay_time = time.time() - 7200  # 2 hours ago

        tension = TensionRecord(
            tension_id="ten_1",
            topic="identity:self",
            belief_ids=["b1"],
            conflict_key="identity_tension:self",
            created_at=time.time() - 86400,
            last_revisited=time.time(),
            revisit_count=200,
            stability=0.9,
            maturation_score=0.95,
        )
        engine._belief_store.add_tension(tension)

        b = BeliefRecord(
            belief_id="b1",
            source_memory_id="m1",
            canonical_subject="self",
            canonical_predicate="is",
            canonical_object="evolving",
            provenance="model_inference",
            extraction_confidence=0.8,
            belief_confidence=0.8,
            timestamp=time.time(),
            rendered_claim="Self is evolving",
            modality="state",
            stance="assertion",
            polarity="positive",
            claim_type="property",
            epistemic_status="active",
            scope="self",
            time_range=None,
            is_state_belief=True,
            conflict_key="identity_tension:self",
            evidence_refs=[],
            contradicts=["b2"],
            resolution_state="tension",
        )
        engine._belief_store.add(b)

        old_debt = engine._contradiction_debt
        engine.apply_passive_decay()
        assert engine._contradiction_debt < old_debt


class TestGetStateIncludesStableParadoxes:
    def test_state_has_stable_paradoxes(self):
        engine = ContradictionEngine()
        tension = TensionRecord(
            tension_id="ten_1",
            topic="identity:self",
            belief_ids=["b1"],
            conflict_key="identity_tension:self",
            created_at=time.time() - 86400,
            last_revisited=time.time(),
            revisit_count=200,
            stability=0.9,
            maturation_score=0.95,
        )
        engine._belief_store.add_tension(tension)
        state = engine.get_state()
        assert "stable_paradoxes" in state
        assert state["stable_paradoxes"] == 1
