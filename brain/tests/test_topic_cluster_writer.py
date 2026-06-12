"""Regression tests for the shared_topic belief-graph writer.

Covers ``GraphBridge.fill_topic_cluster_edges`` — the de-orphaner for
research-extracted claims whose canonical_subjects AND source_memory_ids are
all unique (so ``fill_orphan_edges`` / shared_subject can't connect them).

Honesty guardrails under test:
  - links beliefs sharing >= TOPIC_MIN_SHARED_TOKENS meaningful CONTENT tokens
  - does NOT link beliefs that share only a claim-type prefix (metric/method/...)
  - does NOT link beliefs sharing a single content token (the >=2 gate)
  - forms a hub around the most-confident member (not an O(N^2) mesh)
  - emits valid edges (type "supports", basis "shared_topic"), de-orphaning both ends
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from epistemic.belief_record import BeliefRecord, BeliefStore
from epistemic.belief_graph.edges import EdgeStore, VALID_EVIDENCE_BASES
from epistemic.belief_graph.bridge import GraphBridge, TOPIC_MIN_SHARED_TOKENS


def _belief(belief_id: str, subject: str, belief_confidence: float = 0.8) -> BeliefRecord:
    # source_memory_id deliberately unique per belief (mirrors the live regime
    # that defeats shared_source linking).
    return BeliefRecord(
        belief_id=belief_id, canonical_subject=subject,
        canonical_predicate="is", canonical_object="x", modality="is",
        stance="assert", polarity=1, claim_type="factual",
        epistemic_status="stabilized", extraction_confidence=0.7,
        belief_confidence=belief_confidence, provenance="external_source", scope="",
        source_memory_id=f"mem_{belief_id}", timestamp=0.0, time_range=None,
        is_state_belief=False, conflict_key=f"fact::{subject}::x", evidence_refs=[],
        contradicts=[], resolution_state="active", rendered_claim=subject,
    )


def _bridge(td: str, beliefs: list[BeliefRecord]) -> tuple[GraphBridge, EdgeStore]:
    store = BeliefStore(beliefs_path=os.path.join(td, "beliefs.jsonl"),
                        tensions_path=os.path.join(td, "tensions.jsonl"))
    for b in beliefs:
        store.add(b)
    edge_store = EdgeStore(edges_path=os.path.join(td, "edges.jsonl"))
    return GraphBridge(edge_store, store), edge_store


def test_shared_basis_is_registered():
    assert "shared_topic" in VALID_EVIDENCE_BASES
    assert TOPIC_MIN_SHARED_TOKENS == 2


def test_links_beliefs_sharing_two_content_tokens():
    # Both subjects unique, but share "fivetier" + "routing" (2 content tokens).
    a = _belief("a", "finding_fivetier_routing_strategy_wins")
    b = _belief("b", "method_fivetier_routing_with_disambiguation")
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, [a, b])
        n = bridge.fill_topic_cluster_edges(max_per_cycle=50)
        assert n >= 2                                   # bidirectional
        # both ends gain an INCOMING edge -> de-orphaned
        assert es.get_incoming("a") and es.get_incoming("b")
        e = es.get_incoming("a")[0]
        assert e.edge_type == "supports"
        assert e.evidence_basis == "shared_topic"


def test_skips_prefix_only_overlap():
    # Share ONLY the claim-type prefix "method" (excluded) — zero content overlap.
    a = _belief("a", "method_alpha_quux_indexing")
    b = _belief("b", "method_gamma_zorp_routing")
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, [a, b])
        n = bridge.fill_topic_cluster_edges(max_per_cycle=50)
        assert n == 0
        assert not es.get_incoming("a") and not es.get_incoming("b")


def test_skips_single_shared_content_token():
    # Share exactly ONE content token ("routing") — below the >=2 gate.
    a = _belief("a", "finding_fivetier_routing_strategy")
    b = _belief("b", "problem_routing_latency_spikes")
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, [a, b])
        n = bridge.fill_topic_cluster_edges(max_per_cycle=50)
        assert n == 0


def test_second_pass_is_noop_no_budget_waste():
    # Regression: EdgeStore.add() returns True on a dedup-merge, so a naive linker
    # re-spends its budget re-merging already-linked beliefs every cycle and never
    # drains the backlog (orphan_rate plateaus). The skip-if-already-anchored guard
    # must make a second pass a true no-op.
    a = _belief("a", "finding_fivetier_routing_strategy")
    b = _belief("b", "method_fivetier_routing_disambiguation")
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, [a, b])
        n1 = bridge.fill_topic_cluster_edges(max_per_cycle=50)
        before = len(es._edges)
        assert n1 >= 2 and before >= 2
        n2 = bridge.fill_topic_cluster_edges(max_per_cycle=50)
        assert n2 == 0                 # nothing re-created -> budget preserved
        assert len(es._edges) == before  # no edge growth on re-run


def test_tiny_budget_advances_across_passes():
    # Two separate topic clusters; a budget that only covers one per pass must
    # ADVANCE to the second cluster on the next pass (not get stuck on the first).
    beliefs = [
        _belief("a1", "finding_alpha_routing_strategy"),
        _belief("a2", "method_alpha_routing_strategy"),
        _belief("b1", "finding_beta_audio_pipeline"),
        _belief("b2", "method_beta_audio_pipeline"),
    ]
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, beliefs)
        for _ in range(4):             # a few small passes
            bridge.fill_topic_cluster_edges(max_per_cycle=2)
        # every belief in both clusters ends up anchored
        assert all(es.get_incoming(b) for b in ("a1", "a2", "b1", "b2"))


def test_forms_hub_around_most_confident():
    # Three beliefs all share fivetier+routing; the highest-confidence one is the hub.
    hub = _belief("hub", "concept_fivetier_routing_strategy", belief_confidence=0.95)
    m1 = _belief("m1", "method_fivetier_routing_disambiguation", belief_confidence=0.6)
    m2 = _belief("m2", "finding_fivetier_routing_accuracy", belief_confidence=0.6)
    with tempfile.TemporaryDirectory() as td:
        bridge, es = _bridge(td, [hub, m1, m2])
        bridge.fill_topic_cluster_edges(max_per_cycle=50)
        # the members anchor onto the most-confident belief -> it accretes the
        # most incoming support (a real hub), and every member is de-orphaned.
        inc = {b: len(es.get_incoming(b)) for b in ("hub", "m1", "m2")}
        assert inc["hub"] >= 2                          # both members point at it
        assert inc["m1"] >= 1 and inc["m2"] >= 1
