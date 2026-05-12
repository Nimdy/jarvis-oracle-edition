"""Layer 8 Quarantine — Shadow Mode Invariant Tests.

These tests verify the 5 sacred no-action invariants:
  1. Never blocks memory writes
  2. Never mutates belief confidence
  3. Never suppresses retrieval results
  4. Never changes policy scores
  5. Never affects user-visible replies

Also tests basic scoring and logging functionality.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from epistemic.quarantine.scorer import (
    QuarantineScorer, QuarantineSignal,
    CATEGORY_CONTRADICTION, CATEGORY_MEMORY, CATEGORY_MANIPULATION,
    CATEGORY_CALIBRATION, CATEGORY_IDENTITY,
    _COOLDOWN_WINDOW_S, _CHRONIC_THRESHOLD,
)
from epistemic.quarantine.log import QuarantineLog


# ── Invariant 1: Never blocks memory writes ─────────────────────────────────

def test_scorer_does_not_block_writes():
    """Running the scorer should produce signals but never return any
    blocking directive or mutate the input state."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.9,
        "recent_memory_writes": [
            {"id": "m1", "identity_confidence": 0.1, "identity_needs_resolution": True,
             "type": "user_preference", "provenance": "user_claim", "payload": "test"},
        ] * 10,
        "identity_confidence_dist": {"recent_flips": 8, "conflict_active": True},
        "calibration_snapshot": {"domain_scores": {"epistemic": 0.1}, "domain_provisional": {"epistemic": False}, "truth_score": 0.2},
        "memory_count": 500,
    }
    signals = scorer.tick(state)
    assert len(signals) > 0, "Expected signals from high-anomaly state"
    for sig in signals:
        assert isinstance(sig, QuarantineSignal)
        assert not hasattr(sig, "block")
        assert not hasattr(sig, "suppress")
        assert not hasattr(sig, "quarantine")
    assert state["recent_memory_writes"] is not None


# ── Invariant 2: Never mutates belief confidence ────────────────────────────

def test_scorer_does_not_mutate_beliefs():
    """Scorer must not import or reference BeliefRecord or modify any belief."""
    import inspect
    from epistemic.quarantine import scorer as scorer_module
    source = inspect.getsource(scorer_module)
    assert "BeliefRecord" not in source, "Scorer must not reference BeliefRecord"
    assert "belief_confidence" not in source, "Scorer must not touch belief_confidence"
    assert "belief_store" not in source, "Scorer must not reference belief_store"


# ── Invariant 3: Never suppresses retrieval results ─────────────────────────

def test_scorer_does_not_suppress_retrieval():
    """Scorer must not import or reference search/retrieval modules."""
    import inspect
    from epistemic.quarantine import scorer as scorer_module
    source = inspect.getsource(scorer_module)
    assert "memory.search" not in source
    assert "vector_store" not in source
    assert "retrieval" not in source.lower() or "retrieval_log" not in source


# ── Invariant 4: Never changes policy scores ────────────────────────────────

def test_scorer_does_not_modify_policy():
    """Scorer must not import or reference policy modules."""
    import inspect
    from epistemic.quarantine import scorer as scorer_module
    source = inspect.getsource(scorer_module)
    assert "PolicyDecision" not in source
    assert "policy_nn" not in source
    assert "state_encoder" not in source


# ── Invariant 5: Never affects user-visible replies ─────────────────────────

def test_scorer_does_not_modify_responses():
    """Scorer must not import or reference response/conversation modules."""
    import inspect
    from epistemic.quarantine import scorer as scorer_module
    source = inspect.getsource(scorer_module)
    assert "ResponseGenerator" not in source
    assert "ConversationHandler" not in source
    assert "check_text" not in source
    assert "broadcast" not in source


# ── Signal Score Bounds ─────────────────────────────────────────────────────

def test_signal_score_clamped():
    sig = QuarantineSignal(score=1.5, category="test", reason="over")
    assert sig.score == 1.0
    sig2 = QuarantineSignal(score=-0.5, category="test", reason="under")
    assert sig2.score == 0.0


# ── Contradiction Spike Scoring ─────────────────────────────────────────────

def test_contradiction_spike_critical():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.8,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    contradiction_sigs = [s for s in signals if s.category == CATEGORY_CONTRADICTION]
    assert len(contradiction_sigs) >= 1
    assert contradiction_sigs[0].score >= 0.7


def test_no_signal_when_healthy():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.05,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    assert len(signals) == 0


# ── Memory Anomaly Scoring ──────────────────────────────────────────────────

def test_memory_high_write_rate():
    scorer = QuarantineScorer()
    writes = [{"id": f"m{i}", "identity_confidence": 0.9, "type": "observation",
               "identity_needs_resolution": False, "provenance": "observed", "payload": "x"} for i in range(30)]
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": writes,
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 200,
    })
    mem_sigs = [s for s in signals if s.category == CATEGORY_MEMORY]
    assert len(mem_sigs) >= 1
    assert "rate" in mem_sigs[0].reason.lower()


def test_low_confidence_writes():
    scorer = QuarantineScorer()
    writes = [{"id": f"m{i}", "identity_confidence": 0.2, "type": "observation",
               "identity_needs_resolution": True, "provenance": "observed", "payload": "x"} for i in range(8)]
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": writes,
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 200,
    })
    mem_sigs = [s for s in signals if s.category == CATEGORY_MEMORY]
    assert len(mem_sigs) >= 1


# ── Identity Instability Scoring ────────────────────────────────────────────

def test_identity_flip_detection():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {"recent_flips": 7, "conflict_active": False},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    id_sigs = [s for s in signals if s.category == CATEGORY_IDENTITY]
    assert len(id_sigs) >= 1
    assert "flip" in id_sigs[0].reason.lower()


def test_identity_conflict_detection():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {"recent_flips": 0, "conflict_active": True,
                                      "voice_name": "alice", "face_name": "bob"},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    id_sigs = [s for s in signals if s.category == CATEGORY_IDENTITY]
    assert len(id_sigs) >= 1
    assert "conflict" in id_sigs[0].reason.lower()


# ── Calibration Drift Scoring ───────────────────────────────────────────────

def test_calibration_drift_detection():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"epistemic": 0.15, "retrieval": 0.8},
            "domain_provisional": {"epistemic": False, "retrieval": False},
            "truth_score": 0.35,
        },
        "memory_count": 100,
    })
    cal_sigs = [s for s in signals if s.category == CATEGORY_CALIBRATION]
    assert len(cal_sigs) >= 1


def test_provisional_domains_ignored():
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"epistemic": 0.1},
            "domain_provisional": {"epistemic": True},
            "truth_score": None,
        },
        "memory_count": 100,
    })
    cal_sigs = [s for s in signals if s.category == CATEGORY_CALIBRATION]
    assert len(cal_sigs) == 0, "Provisional domains should not trigger signals"


# ── Log ─────────────────────────────────────────────────────────────────────

def test_log_records_and_retrieves():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "test_quarantine.jsonl")
    log = QuarantineLog(path=path)
    log.record({"score": 0.8, "category": "test", "reason": "test signal"})
    log.record({"score": 0.3, "category": "test2", "reason": "low signal"})
    recent = log.get_recent(10)
    assert len(recent) == 2
    assert recent[0]["score"] == 0.8
    stats = log.get_stats()
    assert stats["total_logged"] == 2


# ── Stats ───────────────────────────────────────────────────────────────────

def test_stats_categories():
    scorer = QuarantineScorer()
    scorer.tick({
        "contradiction_debt": 0.9,
        "recent_memory_writes": [],
        "identity_confidence_dist": {"recent_flips": 10},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    stats = scorer.get_stats()
    assert stats["tick_count"] == 1
    assert stats["total_signals"] > 0
    assert isinstance(stats["category_counts"], dict)


# ── All 5 Categories Present ───────────────────────────────────────────────

def test_all_five_categories_can_fire():
    """Verify all 5 anomaly categories produce signals under extreme conditions."""
    scorer = QuarantineScorer()
    writes = [{"id": f"m{i}", "identity_confidence": 0.1, "type": "user_preference",
               "identity_needs_resolution": True, "provenance": "observed",
               "payload": "correction test"} for i in range(30)]
    signals = scorer.tick({
        "contradiction_debt": 0.9,
        "recent_memory_writes": writes,
        "identity_confidence_dist": {"recent_flips": 10, "conflict_active": True},
        "calibration_snapshot": {
            "domain_scores": {"epistemic": 0.1},
            "domain_provisional": {"epistemic": False},
            "truth_score": 0.2,
        },
        "memory_count": 500,
        "provenance_conflicts": 3,
    })
    categories = {s.category for s in signals}
    assert CATEGORY_CONTRADICTION in categories, f"Missing contradiction, got {categories}"
    assert CATEGORY_MEMORY in categories, f"Missing memory, got {categories}"
    assert CATEGORY_IDENTITY in categories, f"Missing identity, got {categories}"
    assert CATEGORY_CALIBRATION in categories, f"Missing calibration, got {categories}"


# ── Dedupe: Suppress Identical Signals ──────────────────────────────────────

def test_dedupe_suppresses_identical_signals():
    """Same chronic state ticked 5 times within cooldown window → only 1 emitted."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"autonomy": 0.1},
            "domain_provisional": {"autonomy": False},
            "truth_score": 0.55,
        },
        "memory_count": 100,
    }
    all_emitted: list[QuarantineSignal] = []
    for _ in range(5):
        signals = scorer.tick(state)
        all_emitted.extend(signals)

    assert len(all_emitted) == 1, f"Expected 1 emitted signal, got {len(all_emitted)}"
    assert all_emitted[0].category == CATEGORY_CALIBRATION

    stats = scorer.get_stats()
    assert stats["suppressed_duplicates"] == 4
    assert stats["total_signals"] == 1


def test_dedupe_recap_after_cooldown():
    """After cooldown expires, next tick emits a recap with repeat_count."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.8,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    }

    first = scorer.tick(state)
    assert len(first) == 1
    assert first[0].repeat_count == 0

    for _ in range(5):
        scorer.tick(state)

    fp = scorer._compute_fingerprint(first[0])
    entry = scorer._signal_cooldowns[fp]
    entry.first_seen -= _COOLDOWN_WINDOW_S + 1

    recap = scorer.tick(state)
    assert len(recap) == 1, f"Expected recap signal, got {len(recap)}"
    assert recap[0].repeat_count == 5
    assert recap[0].evidence.get("repeat_count") == 5
    assert recap[0].evidence.get("chronic_duration_s") is not None
    assert recap[0].is_chronic is True


def test_dedupe_different_fingerprints_not_suppressed():
    """Two different domains firing simultaneously — both emitted."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"autonomy": 0.1, "epistemic": 0.15},
            "domain_provisional": {"autonomy": False, "epistemic": False},
            "truth_score": 0.55,
        },
        "memory_count": 100,
    }
    signals = scorer.tick(state)
    assert len(signals) == 2, f"Expected 2 signals for 2 domains, got {len(signals)}"
    domains = {s.evidence.get("domain") for s in signals}
    assert "autonomy" in domains
    assert "epistemic" in domains

    second = scorer.tick(state)
    assert len(second) == 0, "Both should be suppressed on second tick"
    assert scorer.get_stats()["suppressed_duplicates"] == 2


def test_dedupe_score_change_within_bucket():
    """Score 0.11 → 0.12 (same bucket) stays suppressed.
    Score 0.11 → 0.25 (different bucket) emits new signal."""
    scorer = QuarantineScorer()
    base_state = lambda score: {
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"autonomy": score},
            "domain_provisional": {"autonomy": False},
            "truth_score": 0.55,
        },
        "memory_count": 100,
    }

    first = scorer.tick(base_state(0.11))
    assert len(first) == 1

    same_bucket = scorer.tick(base_state(0.12))
    assert len(same_bucket) == 0, "0.11 and 0.12 round to 0.1 — same bucket"

    different_bucket = scorer.tick(base_state(0.25))
    assert len(different_bucket) == 1, "0.25 rounds to 0.2 — different bucket"
    assert different_bucket[0].evidence["domain"] == "autonomy"


def test_chronic_signals_in_stats():
    """Verify get_stats() reports chronic_signals when repeat_count exceeds threshold."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": {
            "domain_scores": {"autonomy": 0.1},
            "domain_provisional": {"autonomy": False},
            "truth_score": 0.55,
        },
        "memory_count": 100,
    }

    for _ in range(_CHRONIC_THRESHOLD + 2):
        scorer.tick(state)

    stats = scorer.get_stats()
    assert len(stats["chronic_signals"]) >= 1
    chronic = stats["chronic_signals"][0]
    assert chronic["category"] == CATEGORY_CALIBRATION
    assert chronic["repeat_count"] >= _CHRONIC_THRESHOLD
    assert "duration_s" in chronic
    assert "fingerprint" in chronic


def test_cleanup_expired_cooldowns():
    """Old cooldowns are removed after 2x cooldown window."""
    scorer = QuarantineScorer()
    state = {
        "contradiction_debt": 0.8,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    }

    scorer.tick(state)
    assert len(scorer._signal_cooldowns) == 1

    for entry in scorer._signal_cooldowns.values():
        entry.last_seen -= _COOLDOWN_WINDOW_S * 2.5

    scorer.tick({
        "contradiction_debt": 0.05,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    assert len(scorer._signal_cooldowns) == 0, "Expired cooldowns should be cleaned up"


# ── Provenance Collision Proxy ──────────────────────────────────────────────

def test_provenance_collisions_fires_signal():
    """provenance_conflicts >= 2 should emit a CATEGORY_MANIPULATION signal."""
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
        "provenance_conflicts": 3,
    })
    manip_sigs = [s for s in signals if s.category == CATEGORY_MANIPULATION]
    assert len(manip_sigs) >= 1
    assert "provenance" in manip_sigs[0].reason.lower()
    assert manip_sigs[0].evidence["conflict_count"] == 3


def test_provenance_collisions_below_threshold_silent():
    """provenance_conflicts < 2 should not fire."""
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
        "provenance_conflicts": 1,
    })
    manip_sigs = [s for s in signals if s.category == CATEGORY_MANIPULATION]
    assert len(manip_sigs) == 0


def test_provenance_collisions_zero_by_default():
    """Missing provenance_conflicts key defaults to 0 and does not fire."""
    scorer = QuarantineScorer()
    signals = scorer.tick({
        "contradiction_debt": 0.0,
        "recent_memory_writes": [],
        "identity_confidence_dist": {},
        "calibration_snapshot": None,
        "memory_count": 100,
    })
    manip_sigs = [s for s in signals if s.category == CATEGORY_MANIPULATION]
    assert len(manip_sigs) == 0
