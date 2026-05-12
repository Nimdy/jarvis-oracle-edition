"""Regression tests for the P3.3 prerequisite-tracking belief-graph writer.

Covers :func:`GraphBridge.create_prerequisite_link` and the public hook
:meth:`BeliefGraph.on_prerequisite_detected`.

Acceptance criteria (per TODO_V2.md P3.3):

  * Writer emits ``depends_on`` edges in the correct direction
    (dependent -> prerequisite, per ``edges.py::EvidenceEdge`` docstring).
  * Writer rejects edges that would close a forward-dependency cycle.
  * Existing consumers (``propagation.py``, ``integrity.py``,
    ``topology.py``) correctly observe the emitted edges.
  * Schema audit whitelist no longer carries ``depends_on`` — the literal
    must now exist as a real writer literal in
    ``epistemic/belief_graph/bridge.py``.
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
    make_edge,
)
from epistemic.belief_graph.bridge import GraphBridge
from epistemic.belief_graph.topology import (
    get_dependents,
    get_prerequisites,
)
from epistemic.belief_graph.integrity import compute_integrity


def _belief(
    belief_id: str,
    subject: str | None = None,
    predicate: str = "requires",
    obj: str = "x",
    belief_confidence: float = 0.8,
    provenance: str = "model_inference",
) -> BeliefRecord:
    subject = subject if subject is not None else f"topic_{belief_id}"
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
        belief_confidence=belief_confidence,
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


class TestPrerequisiteWriter:

    def test_writes_depends_on_edge_in_correct_direction(self):
        dep = _belief("b_dep")
        prereq = _belief("b_prereq")

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [dep, prereq])

            ok = bridge.create_prerequisite_link(
                dependent_belief_id=dep.belief_id,
                prerequisite_belief_id=prereq.belief_id,
                strength=0.6,
            )
            assert ok is True

            outgoing = edge_store.get_outgoing(dep.belief_id)
            assert len(outgoing) == 1
            edge = outgoing[0]
            assert edge.edge_type == "depends_on"
            assert edge.evidence_basis == "causal"
            assert edge.source_belief_id == dep.belief_id
            assert edge.target_belief_id == prereq.belief_id

            assert get_prerequisites(dep.belief_id, edge_store) == [
                prereq.belief_id
            ]
            assert get_dependents(prereq.belief_id, edge_store) == [
                dep.belief_id
            ]

    def test_rejects_forward_dependency_cycle(self):
        a = _belief("b_a")
        b = _belief("b_b")
        c = _belief("b_c")

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b, c])

            assert bridge.create_prerequisite_link(a.belief_id, b.belief_id) is True
            assert bridge.create_prerequisite_link(b.belief_id, c.belief_id) is True
            # c -> a would close a cycle (a -> b -> c -> a)
            assert bridge.create_prerequisite_link(c.belief_id, a.belief_id) is False

            outgoing_c = edge_store.get_outgoing(c.belief_id)
            assert outgoing_c == []

    def test_rejects_direct_back_edge_cycle(self):
        a = _belief("b_a")
        b = _belief("b_b")

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b])

            assert bridge.create_prerequisite_link(a.belief_id, b.belief_id) is True
            # b -> a would be a direct cycle.
            assert bridge.create_prerequisite_link(b.belief_id, a.belief_id) is False
            outgoing_b = edge_store.get_outgoing(b.belief_id)
            assert outgoing_b == []

    def test_rejects_missing_beliefs(self):
        a = _belief("b_a")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a])
            assert (
                bridge.create_prerequisite_link(a.belief_id, "b_nonexistent")
                is False
            )
            assert edge_store.get_outgoing(a.belief_id) == []

    def test_rejects_self_edge_and_empty_ids(self):
        a = _belief("b_a")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a])

            assert bridge.create_prerequisite_link(a.belief_id, a.belief_id) is False
            assert bridge.create_prerequisite_link("", a.belief_id) is False
            assert bridge.create_prerequisite_link(a.belief_id, "") is False
            assert edge_store.get_outgoing(a.belief_id) == []

    def test_integrity_tracks_dangling_prerequisite(self):
        """``integrity.compute_integrity`` should observe our edge and
        count dangling when the prerequisite is not in the active set."""
        dep = _belief("b_dep")
        prereq = _belief("b_prereq")

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, belief_store = _bridge_with_beliefs(
                td, [dep, prereq]
            )
            bridge.create_prerequisite_link(dep.belief_id, prereq.belief_id)

            # Remove the prerequisite from the active set by filtering the
            # belief_store's view: compute_integrity uses the store state.
            # Simulate eviction by writing an edge whose prereq id does
            # not exist in the store.
            orphan_edge = make_edge(
                source_belief_id=dep.belief_id,
                target_belief_id="b_evicted",
                edge_type="depends_on", strength=0.5,
                provenance="model_inference", evidence_basis="causal",
            )
            edge_store.add(orphan_edge)

            metrics = compute_integrity(edge_store, belief_store)
            assert metrics["dangling_dependency_count"] >= 1

    def test_depends_on_is_in_valid_edge_types(self):
        assert "depends_on" in VALID_EDGE_TYPES

    def test_whitelist_no_longer_contains_depends_on(self):
        from scripts.schema_emission_audit import FUTURE_ONLY_EDGE_TYPES
        assert "depends_on" not in FUTURE_ONLY_EDGE_TYPES

    def test_writer_literal_present_in_bridge_source(self):
        bridge_src = os.path.join(
            os.path.dirname(__file__),
            "..",
            "epistemic",
            "belief_graph",
            "bridge.py",
        )
        with open(bridge_src, "r", encoding="utf-8") as f:
            text = f.read()
        assert 'edge_type="depends_on"' in text


class TestPropagationObservesDependsOn:

    def test_weak_prerequisite_reduces_dependent_confidence(self):
        from epistemic.belief_graph.propagation import propagate_all

        dep = _belief("b_dep", belief_confidence=0.8)
        prereq = _belief("b_prereq", belief_confidence=0.1)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, belief_store = _bridge_with_beliefs(
                td, [dep, prereq]
            )
            bridge.create_prerequisite_link(
                dep.belief_id, prereq.belief_id, strength=1.0,
            )
            views = propagate_all(edge_store, belief_store)
            assert views[dep.belief_id].effective_confidence < 0.8
