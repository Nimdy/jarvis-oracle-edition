"""Active-tier grounding closure (SAFE subset): an operator answer now ANCHORS the
belief — provenance -> user_claim (grounded; stops re-surfacing) + confidence by
polarity. The risky P5 parts (reward-coupling / immune quarantine) are NOT done here.
"""
from __future__ import annotations

import os
import tempfile
import types

import pytest

os.environ.setdefault("HOME", tempfile.mkdtemp())

try:
    from autonomy.grounding_queue import GroundingQueue
    from epistemic.belief_record import BeliefRecord, BeliefStore
    from epistemic.belief_graph import BeliefGraph
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("grounding closure deps unavailable", allow_module_level=True)


def _belief(bid: str, prov: str = "model_inference", conf: float = 0.4) -> BeliefRecord:
    return BeliefRecord(
        belief_id=bid, canonical_subject="silero_vad", canonical_predicate="is",
        canonical_object="endpoint_detector", modality="is", stance="assert", polarity=1,
        claim_type="factual", epistemic_status="inferred", extraction_confidence=0.6,
        belief_confidence=conf, provenance=prov, scope="", source_memory_id="mem_" + bid,
        timestamp=0.0, time_range=None, is_state_belief=False, conflict_key="fact::x::y",
        evidence_refs=[], contradicts=[], resolution_state="active", rendered_claim="x",
    )


def _wire(td: str, beliefs):
    store = BeliefStore(beliefs_path=os.path.join(td, "b.jsonl"),
                        tensions_path=os.path.join(td, "t.jsonl"))
    for b in beliefs:
        store.add(b)
    BeliefGraph._instance = types.SimpleNamespace(_belief_store=store)  # closure reaches this
    q = GroundingQueue()
    q._pending.clear()
    return q, store


def _qid(q):
    return next(iter(q._pending.values())).question_id


def teardown_function(_):
    BeliefGraph._instance = None


def test_confirm_grounds_belief_and_raises_confidence():
    with tempfile.TemporaryDirectory() as td:
        q, store = _wire(td, [_belief("b1", "model_inference", 0.4)])
        q.enqueue(belief_id="b1", question_text="Is X true?", facet="factual")
        q.answer(_qid(q), "yes, correct")
        b = store.get("b1")
        assert b.provenance == "user_claim"        # now GROUNDED (was model_inference)
        assert b.belief_confidence > 0.4           # confirm raised it


def test_refute_grounds_belief_and_tanks_confidence():
    with tempfile.TemporaryDirectory() as td:
        q, store = _wire(td, [_belief("b2", "model_inference", 0.8)])
        q.enqueue(belief_id="b2", question_text="Is X true?", facet="factual")
        q.answer(_qid(q), "no, wrong")
        b = store.get("b2")
        assert b.provenance == "user_claim"        # anchored either way (correction counts)
        assert b.belief_confidence <= 0.2          # refute tanked it (no aggressive quarantine)


def test_closure_is_safe_noop_without_belief_store():
    # No BeliefGraph wired -> closure must be a silent no-op, never breaking answer().
    BeliefGraph._instance = None
    q = GroundingQueue()
    q._pending.clear()
    q.enqueue(belief_id="bx", question_text="q", facet="factual")
    res = q.answer(_qid(q), "yes")
    assert res.get("ok") is True                   # outcome still recorded, no crash
