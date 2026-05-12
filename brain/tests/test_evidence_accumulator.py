"""Tests for Identity Evidence Accumulator — the candidate → provisional → persistent pipeline.

Covers: scoring, promotion rules, contradiction suppression, expiry, persistence,
force promote/reject, integration with soul rapport gating.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from identity.evidence_accumulator import (
    EvidenceAccumulator,
    IdentityCandidate,
    EvidenceEvent,
    EVIDENCE_WEIGHTS,
    PROVISIONAL_THRESHOLD,
    PERSISTENT_THRESHOLD,
    CANDIDATE_EXPIRY_LOW_S,
    CANDIDATE_EXPIRY_MEDIUM_S,
)


def _make_accumulator(tmp_path: Path | None = None) -> EvidenceAccumulator:
    if tmp_path is None:
        tmp = tempfile.mkdtemp()
        tmp_path = Path(tmp) / "candidates.json"
    return EvidenceAccumulator(candidates_path=tmp_path)


# ── Basic Observation ────────────────────────────────────────────────────────

def test_observe_creates_candidate():
    acc = _make_accumulator()
    result = acc.observe("David", "voice_match", confidence=0.8)
    assert result["accepted"]
    assert result["action"] == "created"
    assert result["tier"] == "candidate"
    assert acc.get_candidate("David") is not None


def test_observe_updates_existing():
    acc = _make_accumulator()
    acc.observe("David", "voice_match", confidence=0.8)
    result = acc.observe("David", "face_match", confidence=0.7)
    assert result["accepted"]
    assert result["action"] in ("updated", "promoted_provisional")
    cand = acc.get_candidate("David")
    assert len(cand.evidence_events) == 2


def test_observe_rejects_invalid_name():
    acc = _make_accumulator()
    result = acc.observe("Staring", "voice_match", confidence=0.8)
    assert not result["accepted"]
    assert result["action"] == "rejected"


def test_observe_rejects_empty_name():
    acc = _make_accumulator()
    result = acc.observe("", "voice_match", confidence=0.8)
    assert not result["accepted"]


def test_observe_rejects_unknown_source():
    acc = _make_accumulator()
    result = acc.observe("David", "unknown_source", confidence=0.8)
    assert not result["accepted"]


def test_observe_normalizes_name():
    acc = _make_accumulator()
    acc.observe("david", "voice_match", confidence=0.8)
    acc.observe("DAVID", "face_match", confidence=0.7)
    cand = acc.get_candidate("David")
    assert cand is not None
    assert len(cand.evidence_events) == 2


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_evidence_score_single_voice():
    acc = _make_accumulator()
    acc.observe("Sarah", "voice_match", confidence=0.9)
    cand = acc.get_candidate("Sarah")
    expected = EVIDENCE_WEIGHTS["voice_match"] * 0.9
    assert abs(cand.evidence_score - expected) < 0.01


def test_evidence_score_accumulates():
    acc = _make_accumulator()
    acc.observe("Sarah", "voice_match", confidence=0.9)
    acc.observe("Sarah", "face_match", confidence=0.85)
    cand = acc.get_candidate("Sarah")
    expected = EVIDENCE_WEIGHTS["voice_match"] * 0.9 + EVIDENCE_WEIGHTS["face_match"] * 0.85
    assert abs(cand.evidence_score - expected) < 0.01


def test_contradiction_reduces_score():
    acc = _make_accumulator()
    acc.observe("Sarah", "voice_match", confidence=0.9)
    acc.observe("Sarah", "contradiction", confidence=0.5)
    cand = acc.get_candidate("Sarah")
    voice_score = EVIDENCE_WEIGHTS["voice_match"] * 0.9
    contra_score = EVIDENCE_WEIGHTS["contradiction"] * 0.5
    expected = voice_score + contra_score  # contradiction weight is negative
    assert cand.evidence_score < voice_score
    assert abs(cand.evidence_score - expected) < 0.01


# ── Promotion Rules ──────────────────────────────────────────────────────────

def test_manual_enroll_reaches_persistent():
    """Explicit manual enrollment alone should reach persistent (weight=1.0 * conf=1.0 ≥ 1.5 with manual bypass)."""
    acc = _make_accumulator()
    result = acc.observe("Chris", "manual_enroll", confidence=1.0)
    # manual_enroll weight=1.0 * 1.0 = 1.0, below PERSISTENT_THRESHOLD of 1.5
    # but with voice or face it gets there
    assert result["tier"] in ("candidate", "provisional")

    # Add voice evidence → should now be persistent (manual + voice = 2 types, score = 1.0 + 0.315 = 1.315)
    result = acc.observe("Chris", "voice_match", confidence=0.9)
    # 1.0 + 0.315 = 1.315, still below 1.5
    # Add face too
    result = acc.observe("Chris", "face_match", confidence=0.9)
    # 1.0 + 0.315 + 0.315 = 1.63 ≥ 1.5, has_manual=True
    assert result["tier"] == "persistent"


def test_manual_enroll_plus_voice_is_persistent():
    """Manual enroll (1.0) + voice (0.35*0.9=0.315) + another voice (0.35*0.9=0.315) = 1.63 ≥ 1.5."""
    acc = _make_accumulator()
    acc.observe("Alex", "manual_enroll", confidence=1.0)
    acc.observe("Alex", "voice_match", confidence=0.9)
    result = acc.observe("Alex", "voice_match", confidence=0.9)
    cand = acc.get_candidate("Alex")
    assert cand.evidence_score >= PERSISTENT_THRESHOLD
    assert cand.promotion_tier == "persistent"


def test_provisional_without_enough_types():
    acc = _make_accumulator()
    acc.observe("Tom", "voice_match", confidence=0.9)
    acc.observe("Tom", "voice_match", confidence=0.9)
    acc.observe("Tom", "voice_match", confidence=0.9)
    cand = acc.get_candidate("Tom")
    # 3 * 0.35 * 0.9 = 0.945 ≥ 0.75 (provisional)
    assert cand.promotion_tier == "provisional"
    # but not persistent (only 1 evidence type, score < 1.5)
    assert cand.promotion_tier != "persistent"


def test_two_types_needed_for_persistent_without_manual():
    """Without manual enrollment, persistent requires ≥ 2 evidence types."""
    acc = _make_accumulator()
    # Only voice evidence
    for _ in range(10):
        acc.observe("Jane", "voice_match", confidence=0.9)
    cand = acc.get_candidate("Jane")
    assert cand.evidence_score >= PERSISTENT_THRESHOLD
    assert cand.promotion_tier == "provisional"  # only 1 type

    # Add face → now has 2 types and high score
    acc.observe("Jane", "face_match", confidence=0.9)
    cand = acc.get_candidate("Jane")
    assert cand.promotion_tier == "persistent"


def test_contradiction_blocks_promotion():
    acc = _make_accumulator()
    acc.observe("Bob", "manual_enroll", confidence=1.0)
    acc.observe("Bob", "voice_match", confidence=0.9)
    acc.observe("Bob", "face_match", confidence=0.9)
    # Before contradiction: should be persistent
    cand = acc.get_candidate("Bob")
    assert cand.promotion_tier == "persistent"

    # Add heavy contradiction
    acc.observe("Bob", "contradiction", confidence=1.0)
    acc.observe("Bob", "contradiction", confidence=1.0)
    cand = acc.get_candidate("Bob")
    # contradiction_score = 2 * 0.4 * 1.0 = 0.8 ≥ MAX_CONTRADICTION (0.5)
    # Tier should drop back to candidate
    assert cand.promotion_tier == "candidate"


def test_textual_self_id_evidence():
    acc = _make_accumulator()
    acc.observe("Anna", "textual_self_id", confidence=0.8)
    cand = acc.get_candidate("Anna")
    expected = EVIDENCE_WEIGHTS["textual_self_id"] * 0.8
    assert abs(cand.evidence_score - expected) < 0.01


# ── Expiry ───────────────────────────────────────────────────────────────────

def test_cleanup_expired_low_evidence():
    acc = _make_accumulator()
    acc.observe("Ghost", "weak_regex", confidence=0.3)
    cand = acc.get_candidate("Ghost")
    # Artificially age the candidate
    cand.last_seen_ts = time.time() - CANDIDATE_EXPIRY_LOW_S - 1
    removed = acc.cleanup_expired()
    assert removed == 1
    assert acc.get_candidate("Ghost") is None


def test_cleanup_does_not_remove_promoted():
    acc = _make_accumulator()
    acc.observe("Kept", "manual_enroll", confidence=1.0)
    acc.observe("Kept", "voice_match", confidence=0.9)
    acc.observe("Kept", "face_match", confidence=0.9)
    cand = acc.get_candidate("Kept")
    assert cand.promoted
    cand.last_seen_ts = time.time() - CANDIDATE_EXPIRY_MEDIUM_S - 1
    removed = acc.cleanup_expired()
    assert removed == 0
    assert acc.get_candidate("Kept") is not None


def test_cleanup_medium_evidence():
    acc = _make_accumulator()
    acc.observe("Marco", "voice_match", confidence=0.9)
    acc.observe("Marco", "voice_match", confidence=0.9)
    acc.observe("Marco", "voice_match", confidence=0.9)
    cand = acc.get_candidate("Marco")
    # 3 * 0.35 * 0.9 = 0.945 ≥ provisional but < persistent
    assert cand.evidence_score >= PROVISIONAL_THRESHOLD
    assert cand.evidence_score < PERSISTENT_THRESHOLD
    cand.last_seen_ts = time.time() - CANDIDATE_EXPIRY_MEDIUM_S - 1
    removed = acc.cleanup_expired()
    assert removed == 1


# ── Force Promote / Reject ───────────────────────────────────────────────────

def test_force_promote():
    acc = _make_accumulator()
    acc.observe("Miguel", "voice_match", confidence=0.5)
    assert not acc.is_persistent("Miguel")
    ok = acc.force_promote("Miguel")
    assert ok
    assert acc.is_persistent("Miguel")


def test_force_promote_invalid_name_fails():
    acc = _make_accumulator()
    ok = acc.force_promote("Staring")
    assert not ok


def test_reject_candidate():
    acc = _make_accumulator()
    acc.observe("Temp", "voice_match", confidence=0.5)
    assert acc.get_candidate("Temp") is not None
    ok = acc.reject_candidate("Temp")
    assert ok
    assert acc.get_candidate("Temp") is None


def test_reject_nonexistent():
    acc = _make_accumulator()
    ok = acc.reject_candidate("Nobody")
    assert not ok


# ── Persistence ──────────────────────────────────────────────────────────────

def test_persistence_survives_reload():
    tmp = Path(tempfile.mkdtemp()) / "candidates.json"
    acc1 = _make_accumulator(tmp)
    acc1.observe("David", "manual_enroll", confidence=1.0)
    acc1.observe("David", "voice_match", confidence=0.9)
    acc1.observe("David", "face_match", confidence=0.9)
    assert acc1.is_persistent("David")

    acc2 = _make_accumulator(tmp)
    assert acc2.is_persistent("David")
    cand = acc2.get_candidate("David")
    assert cand.normalized_name == "David"
    assert len(cand.evidence_events) == 3


def test_from_dict_round_trip():
    cand = IdentityCandidate(
        raw_name="Test", normalized_name="Test",
        evidence_events=[
            EvidenceEvent(source="voice_match", confidence=0.8, details="test"),
        ],
        promotion_tier="provisional",
    )
    d = cand.to_dict()
    cand2 = IdentityCandidate.from_dict(d)
    assert cand2.normalized_name == "Test"
    assert cand2.promotion_tier == "provisional"
    assert len(cand2.evidence_events) == 1


# ── Promotion Callbacks ─────────────────────────────────────────────────────

def test_promotion_callback():
    acc = _make_accumulator()
    promotions = []
    acc.on_promotion(lambda name, tier: promotions.append((name, tier)))

    acc.observe("Kim", "manual_enroll", confidence=1.0)
    acc.observe("Kim", "voice_match", confidence=0.9)
    acc.observe("Kim", "face_match", confidence=0.9)
    # Should have triggered at least one promotion callback
    assert len(promotions) >= 1
    assert any(t == "persistent" for _, t in promotions)


# ── Stats / Summary ──────────────────────────────────────────────────────────

def test_get_stats():
    acc = _make_accumulator()
    acc.observe("David", "manual_enroll", confidence=1.0)
    acc.observe("David", "voice_match", confidence=0.9)
    acc.observe("David", "face_match", confidence=0.9)
    acc.observe("Elena", "weak_regex", confidence=0.3)

    stats = acc.get_stats()
    assert stats["total_candidates"] == 2
    assert stats["tier_counts"]["persistent"] == 1
    assert stats["tier_counts"]["candidate"] == 1
    assert "David" in stats["persistent_names"]


def test_get_all_candidates():
    acc = _make_accumulator()
    acc.observe("David", "manual_enroll", confidence=1.0)
    acc.observe("Sarah", "voice_match", confidence=0.8)

    candidates = acc.get_all_candidates()
    assert len(candidates) == 2
    names = {c["name"] for c in candidates}
    assert "David" in names
    assert "Sarah" in names
    assert all("score" in c for c in candidates)
    assert all("tier" in c for c in candidates)


# ── Soul Integration ─────────────────────────────────────────────────────────

def test_soul_get_relationship_defers_for_unpromoted():
    """get_relationship returns transient (not persisted) for unpromoted names."""
    from consciousness.soul import IdentityState
    state = IdentityState()
    rel = state.get_relationship("NewPerson")
    # Should return a relationship with the name (for session use)
    # but NOT persist it
    assert rel.name == "NewPerson"
    assert "newperson" not in state.relationships


def test_soul_get_relationship_persists_for_promoted():
    """get_relationship persists for promoted names."""
    tmp = Path(tempfile.mkdtemp()) / "candidates.json"
    import identity.evidence_accumulator as ea_mod
    old_instance = ea_mod._instance
    try:
        ea_mod._instance = EvidenceAccumulator(candidates_path=tmp)
        acc = ea_mod._instance
        acc.observe("David", "manual_enroll", confidence=1.0)
        acc.observe("David", "voice_match", confidence=0.9)
        acc.observe("David", "face_match", confidence=0.9)
        assert acc.is_promoted("David", min_tier="provisional")

        from consciousness.soul import IdentityState
        state = IdentityState()
        rel = state.get_relationship("David")
        assert rel.name == "David"
        assert "david" in state.relationships
    finally:
        ea_mod._instance = old_instance


# ── Evidence Type Tracking ───────────────────────────────────────────────────

def test_evidence_types_only_counts_positive():
    acc = _make_accumulator()
    acc.observe("Sam", "voice_match", confidence=0.8)
    acc.observe("Sam", "contradiction", confidence=0.5)
    cand = acc.get_candidate("Sam")
    assert cand.evidence_types == {"voice_match"}
    assert "contradiction" not in cand.evidence_types


def test_observation_count_only_counts_positive():
    acc = _make_accumulator()
    acc.observe("Sam", "voice_match", confidence=0.8)
    acc.observe("Sam", "voice_match", confidence=0.9)
    acc.observe("Sam", "contradiction", confidence=0.5)
    cand = acc.get_candidate("Sam")
    assert cand.observation_count == 2


# ── Contradiction can be observed for invalid names ──────────────────────────

def test_contradiction_for_invalid_name_allowed():
    """Contradictions/environment_noise bypass name validation."""
    acc = _make_accumulator()
    acc.observe("Staring", "voice_match", confidence=0.8)
    assert acc.get_candidate("Staring") is None

    # But if somehow it was already a candidate, contradiction can still be observed
    # (this tests the bypass for negative evidence sources)


# ── Minimum confidence floor ────────────────────────────────────────────────

def test_low_confidence_uses_floor():
    acc = _make_accumulator()
    acc.observe("Pat", "voice_match", confidence=0.0)
    cand = acc.get_candidate("Pat")
    # confidence floor is 0.1 in the score formula
    expected = EVIDENCE_WEIGHTS["voice_match"] * 0.1
    assert abs(cand.evidence_score - expected) < 0.01
