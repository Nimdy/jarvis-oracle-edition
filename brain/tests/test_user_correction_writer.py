"""Regression tests for the P3.2 user-correction belief-graph writer.

Covers :func:`GraphBridge.create_user_correction_link` and the public
singleton hook :meth:`BeliefGraph.on_user_correction`.

Acceptance criteria (per TODO_V2.md P3.2):

  * A simulated user correction produces a ``contradicts`` edge whose
    ``evidence_basis`` is ``user_correction``.
  * The writer is scoped to valid correction evidence (rejects missing
    beliefs, same-id self-edges, empty IDs).
  * The ``schema_emission_audit.py`` whitelist no longer needs the
    ``user_correction`` entry — the literal must now exist as a real
    writer literal in ``epistemic/belief_graph/bridge.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from epistemic.belief_record import BeliefRecord, BeliefStore
from epistemic.belief_graph.edges import (
    EdgeStore,
    VALID_EDGE_TYPES,
    VALID_EVIDENCE_BASES,
)
from epistemic.belief_graph.bridge import GraphBridge


def _belief(
    belief_id: str,
    subject: str = "user_preference",
    predicate: str = "prefers",
    obj: str = "tea",
    provenance: str = "speech_recognition",
) -> BeliefRecord:
    return BeliefRecord(
        belief_id=belief_id,
        canonical_subject=subject,
        canonical_predicate=predicate,
        canonical_object=obj,
        modality="is",
        stance="assert",
        polarity=1,
        claim_type="factual",
        epistemic_status="stabilized",
        extraction_confidence=0.8,
        belief_confidence=0.8,
        provenance=provenance,
        scope="",
        source_memory_id=f"mem_{belief_id}",
        timestamp=time.time(),
        time_range=None,
        is_state_belief=False,
        conflict_key=f"fact::{subject}::{obj}",
        evidence_refs=[],
        contradicts=[],
        resolution_state="active",
        rendered_claim=f"{subject} {predicate} {obj}",
    )


def _bridge_with_beliefs(
    tmpdir: str, beliefs: list[BeliefRecord],
) -> tuple[GraphBridge, EdgeStore, BeliefStore]:
    edge_path = os.path.join(tmpdir, "edges.jsonl")
    beliefs_path = os.path.join(tmpdir, "beliefs.jsonl")
    tensions_path = os.path.join(tmpdir, "tensions.jsonl")
    store = BeliefStore(beliefs_path=beliefs_path, tensions_path=tensions_path)
    for b in beliefs:
        store.add(b)
    edge_store = EdgeStore(edges_path=edge_path)
    bridge = GraphBridge(edge_store, store)
    return bridge, edge_store, store


class TestUserCorrectionWriter:

    def test_writes_contradicts_edge_with_user_correction_basis(self):
        corrected = _belief("b_wrong", obj="tea")
        correcting = _belief("b_right", obj="coffee", provenance="operator_correction")

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(
                td, [corrected, correcting]
            )

            ok = bridge.create_user_correction_link(
                corrected_belief_id=corrected.belief_id,
                correcting_belief_id=correcting.belief_id,
                strength=0.95,
            )
            assert ok is True

            outgoing = edge_store.get_outgoing(correcting.belief_id)
            assert len(outgoing) == 1
            edge = outgoing[0]
            assert edge.edge_type == "contradicts"
            assert edge.evidence_basis == "user_correction"
            assert edge.source_belief_id == correcting.belief_id
            assert edge.target_belief_id == corrected.belief_id
            assert 0.0 < edge.strength <= 1.0

    def test_rejects_missing_beliefs(self):
        existing = _belief("b_exists")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [existing])

            ok = bridge.create_user_correction_link(
                corrected_belief_id="b_nonexistent",
                correcting_belief_id=existing.belief_id,
            )
            assert ok is False
            assert edge_store.get_outgoing(existing.belief_id) == []

    def test_rejects_self_edge(self):
        b = _belief("b_same")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [b])

            ok = bridge.create_user_correction_link(
                corrected_belief_id=b.belief_id,
                correcting_belief_id=b.belief_id,
            )
            assert ok is False
            assert edge_store.get_outgoing(b.belief_id) == []

    def test_rejects_empty_ids(self):
        b = _belief("b_one")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [b])

            assert (
                bridge.create_user_correction_link("", b.belief_id) is False
            )
            assert (
                bridge.create_user_correction_link(b.belief_id, "") is False
            )
            assert edge_store.get_outgoing(b.belief_id) == []

    def test_strength_is_clamped_to_unit_interval(self):
        a = _belief("b_a", obj="tea")
        b = _belief("b_b", obj="coffee")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b])

            bridge.create_user_correction_link(
                corrected_belief_id=a.belief_id,
                correcting_belief_id=b.belief_id,
                strength=5.0,
            )
            edge = edge_store.get_outgoing(b.belief_id)[0]
            assert edge.strength <= 1.0

    def test_user_correction_basis_is_declared(self):
        """Guard: the literal we write must remain in the declared schema."""
        assert "user_correction" in VALID_EVIDENCE_BASES
        assert "contradicts" in VALID_EDGE_TYPES

    def test_whitelist_no_longer_contains_user_correction(self):
        """Audit should now enforce the user_correction writer."""
        from scripts.schema_emission_audit import (
            FUTURE_ONLY_EVIDENCE_BASES,
        )
        assert "user_correction" not in FUTURE_ONLY_EVIDENCE_BASES

    def test_writer_literal_present_in_bridge_source(self):
        """The audit detects writers via quoted literals; ensure ours is
        still in place as ``evidence_basis="user_correction"``."""
        bridge_src = (
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "epistemic",
                "belief_graph",
                "bridge.py",
            )
        )
        with open(bridge_src, "r", encoding="utf-8") as f:
            text = f.read()
        assert 'evidence_basis="user_correction"' in text


class TestBeliefGraphPublicHook:

    def test_on_user_correction_returns_false_when_uninitialized(self):
        from epistemic.belief_graph import BeliefGraph

        graph = BeliefGraph.__new__(BeliefGraph)
        graph._initialized = False
        graph._bridge = None
        # Stubbed _ensure_initialized that refuses to init.
        graph._ensure_initialized = lambda: False  # type: ignore[assignment]

        ok = graph.on_user_correction("a", "b")
        assert ok is False
