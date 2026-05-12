"""Regression tests for the P1.3 belief-graph writers.

Covers two new writers added to ``epistemic.belief_graph.bridge.GraphBridge``:

  - ``create_temporal_sequence_links`` — derived_from edges with basis
    ``temporal_sequence`` between same-subject beliefs that satisfy the
    dwell + recency window.
  - ``record_causal_observation`` (and its event subscriber
    ``_on_world_model_prediction_validated``) — supports/contradicts
    edges with basis ``causal`` driven by validated world-model
    predictions.

These writers are required to exist so that the schema-emission audit
(``brain/scripts/schema_emission_audit.py``) does not flag
``temporal_sequence`` and ``causal`` as DECLARED_BUT_NEVER_EMITTED.
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
from epistemic.belief_graph.bridge import (
    GraphBridge,
    TEMPORAL_DWELL_S,
    TEMPORAL_RECENCY_S,
    TEMPORAL_MAX_EDGES_PER_NEW_BELIEF,
    CAUSAL_MIN_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belief(
    belief_id: str,
    subject: str = "user_engagement",
    predicate: str = "drops",
    obj: str = "fast",
    timestamp: float | None = None,
    extraction_confidence: float = 0.7,
    provenance: str = "speech_recognition",
    resolution_state: str = "active",
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
        extraction_confidence=extraction_confidence,
        belief_confidence=0.8,
        provenance=provenance,
        scope="",
        source_memory_id=f"mem_{belief_id}",
        timestamp=timestamp if timestamp is not None else time.time(),
        time_range=None,
        is_state_belief=False,
        conflict_key=f"fact::{subject}::{obj}",
        evidence_refs=[],
        contradicts=[],
        resolution_state=resolution_state,
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


# ---------------------------------------------------------------------------
# Temporal-sequence writer
# ---------------------------------------------------------------------------


class TestTemporalSequenceWriter:

    def test_writes_derived_from_edge_for_same_subject_with_dwell(self):
        now = time.time()
        prior = _belief("b_old", timestamp=now - (TEMPORAL_DWELL_S + 5.0))
        new = _belief("b_new", timestamp=now)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [prior, new])

            written = bridge.create_temporal_sequence_links(new)

            assert written == 1
            outgoing = edge_store.get_outgoing(new.belief_id)
            assert len(outgoing) == 1
            edge = outgoing[0]
            assert edge.edge_type == "derived_from"
            assert edge.evidence_basis == "temporal_sequence"
            assert edge.target_belief_id == prior.belief_id
            # Strength is intentionally low — temporal precedence only.
            assert 0.0 < edge.strength <= 0.5

    def test_skips_when_dwell_window_not_satisfied(self):
        # Same-batch extraction: prior and new are essentially simultaneous.
        now = time.time()
        prior = _belief("b_old", timestamp=now - 0.1)
        new = _belief("b_new", timestamp=now)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [prior, new])

            written = bridge.create_temporal_sequence_links(new)

            assert written == 0
            assert edge_store.get_outgoing(new.belief_id) == []

    def test_skips_when_outside_recency_window(self):
        # Prior belief is older than the recency window — treat as
        # unrelated long-term memory, not a sequence.
        now = time.time()
        prior = _belief("b_old", timestamp=now - (TEMPORAL_RECENCY_S + 60.0))
        new = _belief("b_new", timestamp=now)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [prior, new])

            written = bridge.create_temporal_sequence_links(new)

            assert written == 0
            assert edge_store.get_outgoing(new.belief_id) == []

    def test_skips_inactive_priors(self):
        now = time.time()
        prior = _belief(
            "b_old",
            timestamp=now - (TEMPORAL_DWELL_S + 5.0),
            resolution_state="superseded",
        )
        new = _belief("b_new", timestamp=now)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [prior, new])

            written = bridge.create_temporal_sequence_links(new)

            assert written == 0

    def test_caps_at_max_edges_per_new_belief(self):
        now = time.time()
        # Build 5 prior beliefs all in the dwell+recency window for the
        # same subject.
        priors = [
            _belief(
                f"b_{i}",
                timestamp=now - (TEMPORAL_DWELL_S + 1.0 + i * 5.0),
            )
            for i in range(5)
        ]
        new = _belief("b_new", timestamp=now)

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, priors + [new])

            written = bridge.create_temporal_sequence_links(new)

            assert written == TEMPORAL_MAX_EDGES_PER_NEW_BELIEF
            assert (
                len(edge_store.get_outgoing(new.belief_id))
                == TEMPORAL_MAX_EDGES_PER_NEW_BELIEF
            )

    def test_temporal_sequence_basis_is_in_valid_set(self):
        # Guard: the literal we write must remain in the declared schema.
        assert "temporal_sequence" in VALID_EVIDENCE_BASES
        assert "derived_from" in VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# Causal writer
# ---------------------------------------------------------------------------


class TestCausalWriter:

    def test_hit_writes_supports_causal_edge(self):
        now = time.time()
        a = _belief(
            "b_engagement",
            subject="user_engagement",
            predicate="drops",
            obj="fast",
            timestamp=now,
            extraction_confidence=0.85,
        )
        b = _belief(
            "b_conversation",
            subject="conversation_active",
            predicate="becomes",
            obj="false",
            timestamp=now,
            extraction_confidence=0.75,
        )

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b])

            written = bridge.record_causal_observation(
                label="user_engagement_drops_conversation_active",
                outcome="hit",
                confidence=0.95,
            )

            assert written == 1
            edges = list(edge_store._edges.values())
            assert len(edges) == 1
            edge = edges[0]
            assert edge.edge_type == "supports"
            assert edge.evidence_basis == "causal"
            stats = bridge.get_stats()
            assert stats["causal_edges_supports"] == 1
            assert stats["causal_edges_contradicts"] == 0

    def test_miss_writes_contradicts_causal_edge(self):
        now = time.time()
        a = _belief(
            "b_alpha",
            subject="user_engagement",
            predicate="rises",
            obj="fast",
            extraction_confidence=0.8,
            timestamp=now,
        )
        b = _belief(
            "b_beta",
            subject="conversation_active",
            predicate="stays",
            obj="true",
            extraction_confidence=0.8,
            timestamp=now,
        )

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b])

            written = bridge.record_causal_observation(
                label="user_engagement_drops_conversation_active",
                outcome="miss",
                confidence=0.9,
            )

            assert written == 1
            edges = list(edge_store._edges.values())
            assert edges[0].edge_type == "contradicts"
            assert edges[0].evidence_basis == "causal"
            assert bridge.get_stats()["causal_edges_contradicts"] == 1

    def test_low_confidence_prediction_skipped_via_event_path(self):
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [])
            # Use the event-handler path so the confidence gate fires.
            bridge._on_world_model_prediction_validated(
                prediction_label="user_engagement_drops",
                outcome="hit",
                prediction_confidence=CAUSAL_MIN_CONFIDENCE - 0.1,
            )
            stats = bridge.get_stats()
            assert stats["causal_low_confidence"] == 1
            assert stats["causal_edges_supports"] == 0
            assert len(edge_store._edges) == 0

    def test_unmapped_prediction_counted_not_emitted(self):
        # No matching beliefs -> writer counts it as unmapped.
        a = _belief("b_unrelated", subject="weather", predicate="is", obj="rainy")
        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a])
            written = bridge.record_causal_observation(
                label="user_engagement_drops",
                outcome="hit",
                confidence=0.9,
            )
            assert written == 0
            assert bridge.get_stats()["causal_unmapped"] == 1
            assert len(edge_store._edges) == 0

    def test_event_subscriber_routes_to_writer(self):
        now = time.time()
        a = _belief(
            "b_engagement",
            subject="user_engagement",
            predicate="drops",
            obj="fast",
            extraction_confidence=0.85,
            timestamp=now,
        )
        b = _belief(
            "b_conversation",
            subject="conversation_active",
            predicate="becomes",
            obj="false",
            extraction_confidence=0.75,
            timestamp=now,
        )

        with tempfile.TemporaryDirectory() as td:
            bridge, edge_store, _ = _bridge_with_beliefs(td, [a, b])
            bridge._on_world_model_prediction_validated(
                prediction_label="user_engagement_drops_conversation_active",
                outcome="hit",
                prediction_confidence=0.9,
            )
            stats = bridge.get_stats()
            assert stats["causal_edges_supports"] == 1
            assert any(
                e.evidence_basis == "causal"
                for e in edge_store._edges.values()
            )

    def test_causal_basis_is_in_valid_set(self):
        assert "causal" in VALID_EVIDENCE_BASES
        assert {"supports", "contradicts"}.issubset(VALID_EDGE_TYPES)
