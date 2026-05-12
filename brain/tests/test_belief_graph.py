"""Tests for Layer 7: Belief Confidence Graph.

Phase 1 tests cover:
- EvidenceEdge dataclass + EdgeStore CRUD
- Bridge wiring (event-driven edge creation)
- Persistence round-trip + compaction
- Support edge gate correctness
- Edge deduplication / merge semantics
- Belief eviction cascade
- Strength decay
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from epistemic.belief_record import BeliefRecord, BeliefStore, TensionRecord
from epistemic.belief_graph.edges import (
    EvidenceEdge,
    EdgeStore,
    make_edge,
    VALID_EDGE_TYPES,
    VALID_EVIDENCE_BASES,
    EDGE_STRENGTH_MIN,
    COMPACTION_RATIO_THRESHOLD,
)
from epistemic.belief_graph.bridge import GraphBridge, SUPPORT_GATE_MIN_EXTRACTION_CONFIDENCE

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belief(
    belief_id: str = "b1",
    subject: str = "hnsw_indexing",
    predicate: str = "improves",
    obj: str = "semantic_recall",
    polarity: int = 1,
    modality: str = "is",
    stance: str = "assert",
    claim_type: str = "factual",
    provenance: str = "external_source",
    extraction_confidence: float = 0.7,
    belief_confidence: float = 0.8,
    source_memory_id: str = "mem_001",
    conflict_key: str = "fact::hnsw_indexing::semantic_recall",
    resolution_state: str = "active",
    is_state_belief: bool = False,
) -> BeliefRecord:
    return BeliefRecord(
        belief_id=belief_id,
        canonical_subject=subject,
        canonical_predicate=predicate,
        canonical_object=obj,
        modality=modality,
        stance=stance,
        polarity=polarity,
        claim_type=claim_type,
        epistemic_status="stabilized",
        extraction_confidence=extraction_confidence,
        belief_confidence=belief_confidence,
        provenance=provenance,
        scope="",
        source_memory_id=source_memory_id,
        timestamp=time.time(),
        time_range=None,
        is_state_belief=is_state_belief,
        conflict_key=conflict_key,
        evidence_refs=[],
        contradicts=[],
        resolution_state=resolution_state,
        rendered_claim=f"{subject} {predicate} {obj}",
    )


def _make_edge_store(tmpdir: str) -> EdgeStore:
    path = os.path.join(tmpdir, "test_edges.jsonl")
    store = EdgeStore(edges_path=path)
    return store


# ===========================================================================
# EdgeStore CRUD
# ===========================================================================


class TestEdgeStoreCRUD:

    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            edge = make_edge("b1", "b2", "supports", 0.8, "external_source", "shared_subject")
            assert store.add(edge) is True
            retrieved = store.get(edge.edge_id)
            assert retrieved is not None
            assert retrieved.source_belief_id == "b1"
            assert retrieved.target_belief_id == "b2"
            assert retrieved.edge_type == "supports"

    def test_add_invalid_type_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            edge = EvidenceEdge(
                edge_id="e1", source_belief_id="b1", target_belief_id="b2",
                edge_type="invalid_type", strength=0.5,
                provenance="external_source", created_at=time.time(),
                last_updated=time.time(), evidence_basis="shared_subject",
            )
            assert store.add(edge) is False

    def test_add_invalid_basis_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            edge = EvidenceEdge(
                edge_id="e1", source_belief_id="b1", target_belief_id="b2",
                edge_type="supports", strength=0.5,
                provenance="external_source", created_at=time.time(),
                last_updated=time.time(), evidence_basis="invalid_basis",
            )
            assert store.add(edge) is False

    def test_self_loop_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            edge = make_edge("b1", "b1", "supports", 0.5, "external_source", "shared_subject")
            assert store.add(edge) is False

    def test_get_outgoing(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.8, "external_source", "shared_subject")
            e2 = make_edge("b1", "b3", "contradicts", 0.6, "external_source", "resolution_outcome")
            e3 = make_edge("b2", "b3", "supports", 0.5, "external_source", "shared_subject")
            store.add(e1)
            store.add(e2)
            store.add(e3)
            outgoing = store.get_outgoing("b1")
            assert len(outgoing) == 2
            out_ids = {e.edge_id for e in outgoing}
            assert e1.edge_id in out_ids
            assert e2.edge_id in out_ids

    def test_get_incoming(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b3", "supports", 0.8, "external_source", "shared_subject")
            e2 = make_edge("b2", "b3", "depends_on", 0.6, "external_source", "causal")
            store.add(e1)
            store.add(e2)
            incoming = store.get_incoming("b3")
            assert len(incoming) == 2

    def test_get_by_type(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            store.add(make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject"))
            store.add(make_edge("b1", "b3", "contradicts", 0.6, "prov", "resolution_outcome"))
            store.add(make_edge("b2", "b3", "supports", 0.5, "prov", "shared_subject"))
            supports = store.get_by_type("supports")
            assert len(supports) == 2
            contradicts = store.get_by_type("contradicts")
            assert len(contradicts) == 1

    def test_remove(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject")
            store.add(e1)
            assert store.get(e1.edge_id) is not None
            assert store.remove(e1.edge_id) is True
            assert store.get(e1.edge_id) is None
            assert len(store.get_outgoing("b1")) == 0
            assert len(store.get_incoming("b2")) == 0

    def test_remove_nonexistent(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            assert store.remove("nonexistent") is False

    def test_remove_edges_for_belief(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            store.add(make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject"))
            store.add(make_edge("b3", "b1", "contradicts", 0.6, "prov", "resolution_outcome"))
            store.add(make_edge("b2", "b3", "supports", 0.5, "prov", "shared_subject"))
            removed = store.remove_edges_for_belief("b1")
            assert removed == 2
            assert len(store.get_outgoing("b1")) == 0
            assert len(store.get_incoming("b1")) == 0
            assert len(store.get_outgoing("b2")) == 1


# ===========================================================================
# Edge deduplication
# ===========================================================================


class TestEdgeDedup:

    def test_duplicate_merge_same_key(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            e2 = make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject")
            assert store.add(e1) is True
            assert store.add(e2) is True
            stats = store.get_stats()
            assert stats["total_edges"] == 1

    def test_duplicate_merge_takes_max_strength(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            e2 = make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject")
            store.add(e1)
            store.add(e2)
            edges = store.get_outgoing("b1")
            assert len(edges) == 1
            assert edges[0].strength == 0.8

    def test_different_type_not_merged(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            e2 = make_edge("b1", "b2", "contradicts", 0.7, "prov", "resolution_outcome")
            store.add(e1)
            store.add(e2)
            assert store.get_stats()["total_edges"] == 2

    def test_different_basis_not_merged(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e1 = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            e2 = make_edge("b1", "b2", "supports", 0.7, "prov", "extractor_link")
            store.add(e1)
            store.add(e2)
            assert store.get_stats()["total_edges"] == 2


# ===========================================================================
# Persistence round-trip
# ===========================================================================


class TestPersistence:

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edges.jsonl")
            store1 = EdgeStore(edges_path=path)
            e1 = make_edge("b1", "b2", "supports", 0.8, "external_source", "shared_subject")
            e2 = make_edge("b1", "b3", "contradicts", 0.6, "observed", "resolution_outcome")
            store1.add(e1)
            store1.add(e2)

            store2 = EdgeStore(edges_path=path)
            store2.rehydrate()
            stats = store2.get_stats()
            assert stats["total_edges"] == 2
            assert store2.get(e1.edge_id) is not None
            assert store2.get(e2.edge_id) is not None

    def test_rehydrate_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edges.jsonl")
            e1 = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            e2 = make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject")
            with open(path, "w") as f:
                f.write(json.dumps(e1.to_dict()) + "\n")
                f.write(json.dumps(e2.to_dict()) + "\n")

            store = EdgeStore(edges_path=path)
            store.rehydrate()
            assert store.get_stats()["total_edges"] == 1

    def test_edge_serialization(self):
        edge = make_edge("b1", "b2", "depends_on", 0.7, "model_inference", "causal")
        d = edge.to_dict()
        restored = EvidenceEdge.from_dict(d)
        assert restored.edge_id == edge.edge_id
        assert restored.source_belief_id == "b1"
        assert restored.target_belief_id == "b2"
        assert restored.edge_type == "depends_on"
        assert restored.strength == 0.7
        assert restored.evidence_basis == "causal"


# ===========================================================================
# Compaction
# ===========================================================================


class TestCompaction:

    def test_compaction_reduces_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edges.jsonl")
            store = EdgeStore(edges_path=path)

            for i in range(10):
                e = make_edge(f"b{i}", f"b{i+1}", "supports", 0.5, "prov", "shared_subject")
                store.add(e)
            for i in range(5):
                edges = store.get_outgoing(f"b{i}")
                if edges:
                    store.remove(edges[0].edge_id)

            store._jsonl_line_count = 100
            store.compact()

            store2 = EdgeStore(edges_path=path)
            store2.rehydrate()
            assert store2.get_stats()["total_edges"] == store.get_stats()["total_edges"]

    def test_needs_compaction(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            store._jsonl_line_count = 0
            assert store.needs_compaction() is False
            store._jsonl_line_count = 100
            assert store.needs_compaction() is True


# ===========================================================================
# Strength decay
# ===========================================================================


class TestStrengthDecay:

    def test_decay_reduces_strength(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
            store.add(e)
            store.decay_strengths(days=10.0)
            updated = store.get(e.edge_id)
            assert updated is not None
            assert updated.strength < 0.5

    def test_decay_removes_weak_edges(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e = make_edge("b1", "b2", "supports", 0.02, "prov", "shared_subject")
            store.add(e)
            store.decay_strengths(days=1000.0)
            assert store.get(e.edge_id) is None

    def test_user_correction_immune_to_decay(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            e = make_edge("b1", "b2", "supports", 0.5, "prov", "user_correction")
            store.add(e)
            store.decay_strengths(days=100.0)
            updated = store.get(e.edge_id)
            assert updated is not None
            assert updated.strength == 0.5


# ===========================================================================
# Stats
# ===========================================================================


class TestStats:

    def test_stats_counts(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            store.add(make_edge("b1", "b2", "supports", 0.8, "prov", "shared_subject"))
            store.add(make_edge("b1", "b3", "contradicts", 0.6, "prov", "resolution_outcome"))
            store.add(make_edge("b2", "b3", "depends_on", 0.5, "prov", "causal"))
            stats = store.get_stats()
            assert stats["total_edges"] == 3
            assert stats["by_type"]["supports"] == 1
            assert stats["by_type"]["contradicts"] == 1
            assert stats["by_type"]["depends_on"] == 1
            assert stats["involved_belief_count"] == 3
            assert len(stats["recent_edges"]) == 3

    def test_recent_edges_limited(self):
        with tempfile.TemporaryDirectory() as td:
            store = _make_edge_store(td)
            for i in range(20):
                store.add(make_edge(f"a{i}", f"b{i}", "supports", 0.5, "prov", "shared_subject"))
            stats = store.get_stats()
            assert len(stats["recent_edges"]) == 10


# ===========================================================================
# Bridge: Support edge gate
# ===========================================================================


class TestSupportGate:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        epath = os.path.join(self.tmpdir, "edges.jsonl")
        self.edge_store = EdgeStore(edges_path=epath)
        self.bridge = GraphBridge(self.edge_store, self.belief_store)

    def test_gate_passes_same_conflict_family(self):
        a = _belief("b1", conflict_key="fact::hnsw_indexing::recall")
        b = _belief("b2", conflict_key="fact::hnsw_indexing::precision")
        assert self.bridge._passes_support_gate(a, b) is True

    def test_gate_rejects_different_families_same_polarity(self):
        a = _belief("b1", polarity=1, modality="is", conflict_key="fact::a::b")
        b = _belief("b2", polarity=1, modality="is", conflict_key="state::x")
        assert self.bridge._passes_support_gate(a, b) is False

    def test_gate_rejects_different_families_high_confidence(self):
        a = _belief("b1", extraction_confidence=0.5, conflict_key="fact::a::b")
        b = _belief("b2", extraction_confidence=0.6, conflict_key="state::x")
        assert self.bridge._passes_support_gate(a, b) is False

    def test_gate_passes_all_conditions_met(self):
        a = _belief("b1", polarity=1, modality="is",
                    extraction_confidence=0.5, conflict_key="fact::a::b")
        b = _belief("b2", polarity=1, modality="is",
                    extraction_confidence=0.5, conflict_key="fact::x::y")
        assert self.bridge._passes_support_gate(a, b) is True

    def test_gate_fails_no_conditions_met(self):
        a = _belief("b1", polarity=1, modality="is",
                    extraction_confidence=0.3, conflict_key="fact::a::b")
        b = _belief("b2", polarity=-1, modality="should",
                    extraction_confidence=0.2, conflict_key="state::x")
        assert self.bridge._passes_support_gate(a, b) is False

    def test_predicate_compatibility_same_family(self):
        a = _belief("b1", predicate="improves")
        b = _belief("b2", predicate="boosts")
        assert self.bridge._are_predicates_compatible(a, b) is True

    def test_predicate_compatibility_opposite_family(self):
        a = _belief("b1", predicate="improves")
        b = _belief("b2", predicate="degrades")
        assert self.bridge._are_predicates_compatible(a, b) is False

    def test_predicate_incompatible_opposite_polarity(self):
        a = _belief("b1", predicate="improves", polarity=1)
        b = _belief("b2", predicate="improves", polarity=-1)
        assert self.bridge._are_predicates_compatible(a, b) is False

    def test_shared_subject_support_creates_edge(self):
        a = _belief("b1", subject="hnsw", predicate="improves", obj="recall",
                    conflict_key="fact::hnsw::recall", source_memory_id="m1")
        b = _belief("b2", subject="hnsw", predicate="boosts", obj="perf",
                    conflict_key="fact::hnsw::perf", source_memory_id="m2")
        self.belief_store.add(a)
        self.belief_store.add(b)
        self.bridge.create_shared_subject_support(a)
        stats = self.edge_store.get_stats()
        assert stats["total_edges"] >= 1

    def test_shared_subject_support_rejected_when_gate_fails(self):
        a = _belief("b1", subject="hnsw", polarity=1, modality="is",
                    extraction_confidence=0.2, conflict_key="fact::a::b",
                    source_memory_id="m1")
        b = _belief("b2", subject="hnsw", polarity=-1, modality="should",
                    extraction_confidence=0.1, conflict_key="state::x",
                    source_memory_id="m2")
        self.belief_store.add(a)
        self.belief_store.add(b)
        self.bridge.create_shared_subject_support(a)
        assert self.edge_store.get_stats()["total_edges"] == 0
        assert self.bridge._gates_rejected >= 1


# ===========================================================================
# Bridge: Extraction co-occurrence
# ===========================================================================


class TestExtractionLinks:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        epath = os.path.join(self.tmpdir, "edges.jsonl")
        self.edge_store = EdgeStore(edges_path=epath)
        self.bridge = GraphBridge(self.edge_store, self.belief_store)

    def test_same_source_creates_extractor_link(self):
        a = _belief("b1", source_memory_id="m1")
        b = _belief("b2", subject="other", source_memory_id="m1",
                    conflict_key="fact::other::x")
        self.belief_store.add(a)
        self.belief_store.add(b)
        self.bridge.create_extraction_links(b)
        edges = self.edge_store.get_outgoing("b2")
        assert len(edges) == 1
        assert edges[0].edge_type == "supports"
        assert edges[0].evidence_basis == "extractor_link"

    def test_different_source_no_link(self):
        a = _belief("b1", source_memory_id="m1")
        b = _belief("b2", source_memory_id="m2")
        self.belief_store.add(a)
        self.belief_store.add(b)
        self.bridge.create_extraction_links(b)
        assert self.edge_store.get_stats()["total_edges"] == 0


# ===========================================================================
# Bridge: Tension held -> refines edges
# ===========================================================================


class TestTensionEdges:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        epath = os.path.join(self.tmpdir, "edges.jsonl")
        self.edge_store = EdgeStore(edges_path=epath)
        self.bridge = GraphBridge(self.edge_store, self.belief_store)

    def test_tension_creates_refines_edges(self):
        a = _belief("b1", claim_type="identity")
        b = _belief("b2", claim_type="identity")
        self.belief_store.add(a)
        self.belief_store.add(b)
        tension = TensionRecord(
            tension_id="t1",
            topic="identity::continuity_vs_replication",
            belief_ids=["b1", "b2"],
            conflict_key="identity::continuity::replication",
            created_at=time.time(),
            last_revisited=time.time(),
            revisit_count=0,
            stability=0.5,
            maturation_score=0.0,
        )
        self.belief_store.add_tension(tension)
        self.bridge._on_tension_held(tension_id="t1")
        stats = self.edge_store.get_stats()
        assert stats["total_edges"] >= 2
        assert "refines" in stats["by_type"]


# ===========================================================================
# Bridge: Version link
# ===========================================================================


class TestVersionLink:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        epath = os.path.join(self.tmpdir, "edges.jsonl")
        self.edge_store = EdgeStore(edges_path=epath)
        self.bridge = GraphBridge(self.edge_store, self.belief_store)

    def test_version_link_creates_derived_from(self):
        old = _belief("b_old", resolution_state="versioned")
        new = _belief("b_new")
        self.belief_store.add(old)
        self.belief_store.add(new)
        self.bridge.create_version_link("b_old", "b_new")
        edges = self.edge_store.get_outgoing("b_new")
        assert len(edges) == 1
        assert edges[0].edge_type == "derived_from"
        assert edges[0].target_belief_id == "b_old"
        assert edges[0].evidence_basis == "belief_version"


# ===========================================================================
# make_edge factory
# ===========================================================================


class TestMakeEdge:

    def test_all_edge_types_valid(self):
        for et in VALID_EDGE_TYPES:
            edge = make_edge("b1", "b2", et, 0.5, "prov", "shared_subject")
            assert edge.edge_type == et

    def test_strength_clamped(self):
        e1 = make_edge("b1", "b2", "supports", 1.5, "prov", "shared_subject")
        assert e1.strength == 1.0
        e2 = make_edge("b1", "b2", "supports", -0.5, "prov", "shared_subject")
        assert e2.strength == 0.0

    def test_with_strength(self):
        edge = make_edge("b1", "b2", "supports", 0.5, "prov", "shared_subject")
        updated = edge.with_strength(0.9)
        assert updated.strength == 0.9
        assert updated.edge_id == edge.edge_id
        assert updated.last_updated >= edge.last_updated


# ===========================================================================
# BeliefGraph orchestrator
# ===========================================================================


class TestBeliefGraphOrchestrator:

    def test_get_state_empty(self):
        from epistemic.belief_graph import BeliefGraph
        bg = BeliefGraph()
        state = bg.get_state()
        assert state == {}

    def test_on_new_belief_noop_when_uninitialized(self):
        from epistemic.belief_graph import BeliefGraph
        bg = BeliefGraph()
        belief = _belief("b1")
        bg.on_new_belief(belief)


# ===========================================================================
# Bridge stats
# ===========================================================================


class TestBridgeStats:

    def test_initial_stats(self):
        with tempfile.TemporaryDirectory() as td:
            bpath = os.path.join(td, "beliefs.jsonl")
            tpath = os.path.join(td, "tensions.jsonl")
            bs = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
            epath = os.path.join(td, "edges.jsonl")
            es = EdgeStore(edges_path=epath)
            bridge = GraphBridge(es, bs)
            stats = bridge.get_stats()
            assert stats["edges_created"] == 0
            assert stats["gates_rejected"] == 0
            assert stats["subscribed"] is False


# ===========================================================================
# Topology queries
# ===========================================================================


class TestTopology:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.edge_store = _make_edge_store(self.tmpdir)

    def test_support_strength(self):
        from epistemic.belief_graph.topology import get_support_strength
        self.edge_store.add(make_edge("b1", "b3", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.3, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "contradicts", 0.4, "p", "resolution_outcome"))
        assert abs(get_support_strength("b3", self.edge_store) - 0.8) < 0.01

    def test_contradiction_pressure(self):
        from epistemic.belief_graph.topology import get_contradiction_pressure
        self.edge_store.add(make_edge("b1", "b2", "contradicts", 0.6, "p", "resolution_outcome"))
        self.edge_store.add(make_edge("b3", "b2", "contradicts", 0.4, "p", "resolution_outcome"))
        assert abs(get_contradiction_pressure("b2", self.edge_store) - 1.0) < 0.01

    def test_dependents_and_prerequisites(self):
        from epistemic.belief_graph.topology import get_dependents, get_prerequisites
        self.edge_store.add(make_edge("b2", "b1", "depends_on", 0.5, "p", "causal"))
        self.edge_store.add(make_edge("b3", "b1", "depends_on", 0.5, "p", "causal"))
        assert set(get_dependents("b1", self.edge_store)) == {"b2", "b3"}
        assert get_prerequisites("b2", self.edge_store) == ["b1"]

    def test_roots_and_leaves(self):
        from epistemic.belief_graph.topology import get_roots, get_leaves
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.5, "p", "shared_subject"))
        roots = get_roots(self.edge_store)
        leaves = get_leaves(self.edge_store)
        assert "b1" in roots
        assert "b3" in leaves

    def test_find_path(self):
        from epistemic.belief_graph.topology import find_path
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.5, "p", "shared_subject"))
        path = find_path("b1", "b3", self.edge_store)
        assert path == ["b1", "b2", "b3"]

    def test_find_path_no_route(self):
        from epistemic.belief_graph.topology import find_path
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        assert find_path("b2", "b1", self.edge_store) is None

    def test_find_path_self(self):
        from epistemic.belief_graph.topology import find_path
        assert find_path("b1", "b1", self.edge_store) == ["b1"]

    def test_connected_components(self):
        from epistemic.belief_graph.topology import get_connected_components
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b3", "b4", "supports", 0.5, "p", "shared_subject"))
        comps = get_connected_components(self.edge_store)
        assert len(comps) == 2

    def test_belief_centrality(self):
        from epistemic.belief_graph.topology import get_belief_centrality
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b1", "b3", "supports", 0.5, "p", "extractor_link"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.5, "p", "shared_subject"))
        centrality = get_belief_centrality(self.edge_store)
        assert centrality["b1"] > 0
        assert centrality["b3"] > 0

    def test_top_beliefs_by_centrality(self):
        from epistemic.belief_graph.topology import get_top_beliefs_by_centrality
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b1", "b3", "supports", 0.5, "p", "extractor_link"))
        self.edge_store.add(make_edge("b1", "b4", "supports", 0.5, "p", "memory_association"))
        top = get_top_beliefs_by_centrality(self.edge_store, n=2)
        assert len(top) == 2
        assert top[0][0] == "b1"


# ===========================================================================
# Integrity metrics
# ===========================================================================


class TestIntegrity:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        self.edge_store = _make_edge_store(self.tmpdir)

    def test_healthy_graph(self):
        from epistemic.belief_graph.integrity import compute_integrity
        b1 = _belief("b1")
        b2 = _belief("b2", subject="other", conflict_key="fact::other::x")
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.8, "p", "shared_subject"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["health_score"] > 0.7
        assert metrics["orphan_rate"] == 0.0
        assert metrics["cycle_count"] == 0

    def test_orphan_rate(self):
        from epistemic.belief_graph.integrity import compute_integrity
        b1 = _belief("b1")
        b2 = _belief("b2", subject="other", conflict_key="fact::other::x")
        b3 = _belief("b3", subject="orphan", conflict_key="fact::orphan::z")
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.belief_store.add(b3)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.8, "p", "shared_subject"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["orphan_count"] == 1
        assert metrics["orphan_rate"] > 0

    def test_cycle_detection(self):
        from epistemic.belief_graph.integrity import compute_integrity
        b1 = _belief("b1")
        b2 = _belief("b2", subject="x", conflict_key="f::x::y")
        b3 = _belief("b3", subject="y", conflict_key="f::y::z")
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.belief_store.add(b3)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b3", "b1", "supports", 0.5, "p", "shared_subject"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["cycle_count"] >= 1

    def test_fragmentation(self):
        from epistemic.belief_graph.integrity import compute_integrity
        for i in range(4):
            self.belief_store.add(_belief(f"b{i}", subject=f"s{i}", conflict_key=f"f::s{i}::o{i}"))
        self.edge_store.add(make_edge("b0", "b1", "supports", 0.5, "p", "shared_subject"))
        self.edge_store.add(make_edge("b2", "b3", "supports", 0.5, "p", "shared_subject"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["component_count"] == 2
        assert metrics["fragmentation"] > 0

    def test_dangling_dependency(self):
        from epistemic.belief_graph.integrity import compute_integrity
        b1 = _belief("b1")
        self.belief_store.add(b1)
        self.edge_store.add(make_edge("b1", "b_evicted", "depends_on", 0.5, "p", "causal"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["dangling_dependency_count"] == 1
        assert metrics["dangling_dependency_rate"] == 1.0

    def test_support_from_quarantined(self):
        from epistemic.belief_graph.integrity import compute_integrity
        b1 = _belief("b1", resolution_state="quarantined")
        b2 = _belief("b2")
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.5, "p", "shared_subject"))
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["support_from_quarantined_count"] == 1
        assert metrics["support_from_quarantined_rate"] == 1.0

    def test_empty_graph_healthy(self):
        from epistemic.belief_graph.integrity import compute_integrity
        metrics = compute_integrity(self.edge_store, self.belief_store)
        assert metrics["health_score"] == 1.0
        assert metrics["total_edges"] == 0


# ===========================================================================
# Propagation
# ===========================================================================


class TestPropagation:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        self.edge_store = _make_edge_store(self.tmpdir)

    def test_support_boosts_confidence(self):
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.8)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.6)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.9, "p", "shared_subject"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b2"].effective_confidence > 0.6
        assert views["b2"].structural_confidence_delta > 0

    def test_contradiction_lowers_confidence(self):
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.8)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.7)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "contradicts", 0.9, "p", "resolution_outcome"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b2"].effective_confidence < 0.7
        assert views["b2"].structural_confidence_delta < 0

    def test_orphan_keeps_base_confidence(self):
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.75)
        self.belief_store.add(b1)
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b1"].effective_confidence == 0.75
        assert views["b1"].structural_confidence_delta == 0.0

    def test_max_propagation_delta_clamped(self):
        from epistemic.belief_graph.propagation import propagate_all, MAX_PROPAGATION_DELTA
        b_target = _belief("bt", belief_confidence=0.5)
        self.belief_store.add(b_target)
        for i in range(20):
            b = _belief(f"bs{i}", subject=f"s{i}", conflict_key=f"f::s{i}::o{i}",
                       belief_confidence=1.0)
            self.belief_store.add(b)
            self.edge_store.add(make_edge(f"bs{i}", "bt", "supports", 1.0, "p", "shared_subject"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["bt"].structural_confidence_delta <= MAX_PROPAGATION_DELTA

    def test_confidence_stays_in_bounds(self):
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.95)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.98)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 1.0, "p", "shared_subject"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b2"].effective_confidence <= 1.0
        assert views["b2"].effective_confidence >= 0.0

    def test_neighborhood_propagation(self):
        from epistemic.belief_graph.propagation import propagate_neighborhood
        b1 = _belief("b1", belief_confidence=0.8)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.5)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 0.9, "p", "shared_subject"))
        view = propagate_neighborhood("b2", self.edge_store, self.belief_store)
        assert view is not None
        assert view.effective_confidence > 0.5

    def test_neighborhood_nonexistent_belief(self):
        from epistemic.belief_graph.propagation import propagate_neighborhood
        view = propagate_neighborhood("nonexistent", self.edge_store, self.belief_store)
        assert view is None


# ===========================================================================
# Sacred invariants
# ===========================================================================


class TestSacredInvariants:
    """Tests for all sacred invariants defined in the Layer 7 plan."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        bpath = os.path.join(self.tmpdir, "beliefs.jsonl")
        tpath = os.path.join(self.tmpdir, "tensions.jsonl")
        self.belief_store = BeliefStore(beliefs_path=bpath, tensions_path=tpath)
        self.edge_store = _make_edge_store(self.tmpdir)

    def test_invariant_1_identity_tension_immune(self):
        """Identity-tension beliefs are immune to propagation."""
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.8, claim_type="identity")
        b2 = _belief("b2", belief_confidence=0.6, claim_type="identity",
                     subject="identity", conflict_key="identity::continuity::x")
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        tension = TensionRecord(
            tension_id="t1", topic="identity::continuity_vs_replication",
            belief_ids=["b1", "b2"], conflict_key="identity::continuity::replication",
            created_at=time.time(), last_revisited=time.time(),
            revisit_count=0, stability=0.5, maturation_score=0.0,
        )
        self.belief_store.add_tension(tension)
        self.edge_store.add(make_edge("b1", "b2", "contradicts", 1.0, "p", "resolution_outcome"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b1"].structural_confidence_delta == 0.0
        assert views["b2"].structural_confidence_delta == 0.0
        assert views["b1"].effective_confidence == 0.8
        assert views["b2"].effective_confidence == 0.6

    def test_invariant_2_user_claim_floor(self):
        """User-claim beliefs have a floor of 0.5 effective confidence."""
        from epistemic.belief_graph.propagation import propagate_all, USER_CLAIM_FLOOR
        b1 = _belief("b1", belief_confidence=0.9)
        b_user = _belief("b_user", belief_confidence=0.3, provenance="user_claim",
                        subject="pref", conflict_key="pref::dark_mode::ui")
        self.belief_store.add(b1)
        self.belief_store.add(b_user)
        self.edge_store.add(make_edge("b1", "b_user", "contradicts", 1.0, "p", "resolution_outcome"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b_user"].effective_confidence >= USER_CLAIM_FLOOR

    def test_invariant_3_refines_excluded_from_propagation(self):
        """Refines edges do NOT contribute to effective confidence."""
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.9)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.5)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "refines", 1.0, "p", "resolution_outcome"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b2"].structural_confidence_delta == 0.0
        assert views["b2"].effective_confidence == 0.5

    def test_invariant_4_propagation_is_view_only(self):
        """Propagation NEVER mutates BeliefRecord.belief_confidence."""
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.9)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.5)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 1.0, "p", "shared_subject"))

        original_conf = self.belief_store.get("b2").belief_confidence
        propagate_all(self.edge_store, self.belief_store)
        after_conf = self.belief_store.get("b2").belief_confidence
        assert original_conf == after_conf

    def test_invariant_5_confidence_in_bounds(self):
        """Confidence always stays in [0, 1]."""
        from epistemic.belief_graph.propagation import propagate_all
        b1 = _belief("b1", belief_confidence=0.99)
        b2 = _belief("b2", subject="x", conflict_key="f::x::y", belief_confidence=0.99)
        self.belief_store.add(b1)
        self.belief_store.add(b2)
        self.edge_store.add(make_edge("b1", "b2", "supports", 1.0, "p", "shared_subject"))
        views = propagate_all(self.edge_store, self.belief_store)
        for v in views.values():
            assert 0.0 <= v.effective_confidence <= 1.0

    def test_invariant_6_orphan_keeps_base(self):
        """Orphan beliefs (no edges) keep base_confidence as effective."""
        from epistemic.belief_graph.propagation import propagate_all
        b = _belief("b1", belief_confidence=0.72)
        self.belief_store.add(b)
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["b1"].base_confidence == 0.72
        assert views["b1"].effective_confidence == 0.72
        assert views["b1"].structural_confidence_delta == 0.0

    def test_depends_on_weak_prerequisite_reduces_confidence(self):
        """When a depends_on prerequisite is very weak, dependent loses confidence."""
        from epistemic.belief_graph.propagation import propagate_all
        prereq = _belief("prereq", belief_confidence=0.1)
        dependent = _belief("dep", subject="x", conflict_key="f::x::y", belief_confidence=0.8)
        self.belief_store.add(prereq)
        self.belief_store.add(dependent)
        self.edge_store.add(make_edge("dep", "prereq", "depends_on", 1.0, "p", "causal"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["dep"].effective_confidence < 0.8
        assert views["dep"].structural_confidence_delta < 0

    def test_derived_from_provides_slight_boost(self):
        """Derived-from edges provide a small positive contribution."""
        from epistemic.belief_graph.propagation import propagate_all
        parent = _belief("parent", belief_confidence=0.9)
        child = _belief("child", subject="x", conflict_key="f::x::y", belief_confidence=0.5)
        self.belief_store.add(parent)
        self.belief_store.add(child)
        self.edge_store.add(make_edge("parent", "child", "derived_from", 0.8, "p", "belief_version"))
        views = propagate_all(self.edge_store, self.belief_store)
        assert views["child"].effective_confidence > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
