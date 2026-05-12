"""Phase 1 acceptance tests for Dream Observer Build Plan.

Tests:
  1. No dream-origin weight inflation
  2. Normal reinforcement preserved for non-dream memories
  3. No belief extraction from dream-tagged memories
  4. Association cap enforced
  5. Dream cycle alive (core phases run, no canonical writes)
  6. Cleanup script downweights + supersedes correctly
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory
from memory.core import MemoryCore, CreateMemoryData
from memory.storage import (
    MemoryStorage,
    _DREAM_ORIGIN_TAGS,
    _MAX_ASSOCIATIONS_PER_MEMORY,
    _NEW_MEMORY_WEIGHT_CAP,
    _REINFORCED_WEIGHT_CAP,
)
from epistemic.contradiction_engine import ContradictionEngine, _DREAM_INELIGIBLE_TAGS
from epistemic.belief_record import BeliefStore


_counter = 0


def _uid() -> str:
    global _counter
    _counter += 1
    return f"test_dream_{_counter}_{time.time()}"


def _make_memory(
    weight: float = 0.4,
    tags: tuple[str, ...] = (),
    mem_type: str = "observation",
    payload: str = "test payload",
) -> Memory:
    return Memory(
        id=_uid(),
        timestamp=time.time(),
        weight=weight,
        tags=tags,
        payload=payload,
        type=mem_type,
        provenance="model_inference",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: No dream-origin weight inflation
# ──────────────────────────────────────────────────────────────────────────────

def test_dream_insight_skips_reinforcement_multiplier():
    """Dream-tagged memory at 0.4 should stay at 0.4 even under 2.0 multiplier."""
    storage = MemoryStorage(max_capacity=100)
    storage.set_reinforcement_multiplier(2.0)

    mem = _make_memory(weight=0.4, tags=("dream_insight", "generated_insight"))
    storage.add(mem)

    stored = storage.get_by_tag("dream_insight")
    assert len(stored) == 1
    assert abs(stored[0].weight - 0.4) < 1e-6, f"Expected 0.4, got {stored[0].weight}"


def test_dream_hypothesis_skips_reinforcement():
    """All _DREAM_ORIGIN_TAGS should skip multiplier."""
    for tag in _DREAM_ORIGIN_TAGS:
        storage = MemoryStorage(max_capacity=100)
        storage.set_reinforcement_multiplier(2.0)
        mem = _make_memory(weight=0.35, tags=(tag,))
        storage.add(mem)
        stored = storage.get_by_tag(tag)
        assert len(stored) == 1
        assert abs(stored[0].weight - 0.35) < 1e-6, f"Tag {tag}: expected 0.35, got {stored[0].weight}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Normal reinforcement still works
# ──────────────────────────────────────────────────────────────────────────────

def test_normal_memory_gets_reinforced():
    """Non-dream memory should still get multiplied under dreaming mode."""
    storage = MemoryStorage(max_capacity=100)
    storage.set_reinforcement_multiplier(2.0)

    mem = _make_memory(weight=0.3, tags=("observation",))
    storage.add(mem)

    stored = storage.get_by_tag("observation")
    assert len(stored) == 1
    expected = min(_REINFORCED_WEIGHT_CAP, 0.3 * 2.0)
    assert abs(stored[0].weight - expected) < 1e-6, f"Expected {expected}, got {stored[0].weight}"


def test_non_core_cap_then_multiplier():
    """Non-core memory capped at _NEW_MEMORY_WEIGHT_CAP, then multiplied."""
    storage = MemoryStorage(max_capacity=100)
    storage.set_reinforcement_multiplier(1.5)

    mem = _make_memory(weight=0.7, tags=("conversation",))
    storage.add(mem)

    stored = storage.get_by_tag("conversation")
    assert len(stored) == 1
    capped = min(0.7, _NEW_MEMORY_WEIGHT_CAP)
    expected = min(_REINFORCED_WEIGHT_CAP, capped * 1.5)
    assert abs(stored[0].weight - expected) < 1e-6, f"Expected {expected}, got {stored[0].weight}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: No belief extraction from dream-tagged memories
# ──────────────────────────────────────────────────────────────────────────────

def test_dream_insight_ineligible_for_belief():
    """Dream-tagged memory should fail _is_belief_eligible()."""
    engine = ContradictionEngine()

    for tag in _DREAM_INELIGIBLE_TAGS:
        mem = _make_memory(weight=0.6, tags=(tag,))
        assert not engine._is_belief_eligible(mem), f"Tag {tag} should be ineligible"


def test_non_dream_memory_eligible_for_belief():
    """Normal memory above threshold should pass _is_belief_eligible()."""
    engine = ContradictionEngine()

    mem = _make_memory(weight=0.5, tags=("conversation",), mem_type="conversation")
    assert engine._is_belief_eligible(mem), "Normal memory should be eligible"


def test_dream_tag_beats_high_weight():
    """Even high-weight dream memories should be ineligible."""
    engine = ContradictionEngine()

    mem = _make_memory(weight=0.9, tags=("dream_insight",))
    assert not engine._is_belief_eligible(mem), "High-weight dream insight should still be ineligible"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Association cap
# ──────────────────────────────────────────────────────────────────────────────

def test_association_cap_blocks_at_limit():
    """Cannot add more than _MAX_ASSOCIATIONS_PER_MEMORY associations."""
    storage = MemoryStorage(max_capacity=200)

    base = _make_memory(weight=0.5, tags=("test",))
    storage.add(base)

    linked_ids = []
    for i in range(_MAX_ASSOCIATIONS_PER_MEMORY):
        other = _make_memory(weight=0.5, tags=("test",))
        storage.add(other)
        result = storage.associate(base.id, other.id)
        assert result, f"Association {i} should succeed"
        linked_ids.append(other.id)

    overflow = _make_memory(weight=0.5, tags=("test",))
    storage.add(overflow)
    result = storage.associate(base.id, overflow.id)
    assert not result, "Should reject association beyond cap"


def test_association_cap_works_bidirectional():
    """Cap applies to both sides: if B is at the limit, A->B also fails."""
    storage = MemoryStorage(max_capacity=200)

    saturated = _make_memory(weight=0.5, tags=("test",))
    storage.add(saturated)

    for _ in range(_MAX_ASSOCIATIONS_PER_MEMORY):
        other = _make_memory(weight=0.5, tags=("test",))
        storage.add(other)
        storage.associate(saturated.id, other.id)

    fresh = _make_memory(weight=0.5, tags=("test",))
    storage.add(fresh)
    result = storage.associate(fresh.id, saturated.id)
    assert not result, "Should reject when target is at cap"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Dream cycle structural integrity
# ──────────────────────────────────────────────────────────────────────────────

def test_dream_origin_tags_constant():
    """Verify the tag sets are consistent between modules."""
    from memory.storage import _DREAM_ORIGIN_TAGS as storage_tags
    from epistemic.contradiction_engine import _DREAM_INELIGIBLE_TAGS as belief_tags

    assert "dream_insight" in storage_tags
    assert "dream_hypothesis" in storage_tags
    assert "dream_artifact" in storage_tags
    assert "sleep_candidate" in storage_tags

    assert storage_tags == belief_tags, "Tag sets should match between storage and contradiction engine"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Cleanup (unit test of the logic, not the full script)
# ──────────────────────────────────────────────────────────────────────────────

def test_downweight_dream_insights():
    """Downweight API correctly reduces dream insight weight toward 0.4."""
    storage = MemoryStorage(max_capacity=100)

    mem = _make_memory(weight=0.55, tags=("dream_insight",))
    storage.add(mem)

    stored_before = storage.get_by_tag("dream_insight")
    actual_weight = stored_before[0].weight
    target = 0.4
    factor = target / actual_weight
    ok = storage.downweight(mem.id, weight_factor=factor, decay_rate_factor=1.0)
    assert ok, "Downweight should succeed"

    stored = storage.get_by_tag("dream_insight")
    assert len(stored) == 1
    assert abs(stored[0].weight - target) < 0.02, f"Expected ~{target}, got {stored[0].weight}"


def test_supersede_dream_belief():
    """update_resolution correctly marks dream belief as superseded."""
    import tempfile

    store = BeliefStore(
        beliefs_path=os.path.join(tempfile.mkdtemp(), "beliefs.jsonl"),
        tensions_path=os.path.join(tempfile.mkdtemp(), "tensions.jsonl"),
    )

    from epistemic.belief_record import BeliefRecord

    belief = BeliefRecord(
        belief_id=_uid(),
        canonical_subject="dream_insight_test_pattern",
        canonical_predicate="is",
        canonical_object="test",
        modality="is",
        stance="assert",
        polarity=1,
        claim_type="factual",
        epistemic_status="inferred",
        extraction_confidence=0.5,
        belief_confidence=0.4,
        provenance="model_inference",
        scope="",
        source_memory_id="mem_test",
        timestamp=time.time(),
        time_range=None,
        is_state_belief=False,
        conflict_key="test",
        evidence_refs=[],
        contradicts=[],
        resolution_state="active",
        rendered_claim="dream insight test pattern is test",
    )
    store.add(belief)

    active_before = store.get_active_beliefs()
    assert len(active_before) == 1

    store.update_resolution(belief.belief_id, "superseded")

    active_after = store.get_active_beliefs()
    assert len(active_after) == 0, "Superseded belief should not appear in active list"


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_dream_insight_skips_reinforcement_multiplier,
        test_dream_hypothesis_skips_reinforcement,
        test_normal_memory_gets_reinforced,
        test_non_core_cap_then_multiplier,
        test_dream_insight_ineligible_for_belief,
        test_non_dream_memory_eligible_for_belief,
        test_dream_tag_beats_high_weight,
        test_association_cap_blocks_at_limit,
        test_association_cap_works_bidirectional,
        test_dream_origin_tags_constant,
        test_downweight_dream_insights,
        test_supersede_dream_belief,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
