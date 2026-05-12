"""Bridge module: Layer 5 events and memory associations into graph edges.

Edge budgeting: per-subject support edges capped at MAX_SUPPORT_EDGES_PER_SUBJECT
to prevent O(N^2) fanout on hot subjects. Support gate uses AND logic (all three
conditions must pass) instead of OR. Per-belief outgoing cap prevents single
beliefs from dominating the graph.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

SUPPORT_GATE_MIN_EXTRACTION_CONFIDENCE: float = 0.4

MAX_SUPPORT_EDGES_PER_SUBJECT: int = 20
MAX_SUPPORT_EDGES_PER_BELIEF: int = 8
MAX_CANDIDATES_PER_SUBJECT: int = 30

# Temporal-sequence writer (P1.3): emit a derived_from edge with basis
# ``temporal_sequence`` when a new belief lands behind one or more prior
# active beliefs about the same canonical subject. Gated on a dwell window
# (the prior belief must be at least ``TEMPORAL_DWELL_S`` seconds older
# than the new one — same-batch extractions do not count as a "sequence")
# and a recency window (older than ``TEMPORAL_RECENCY_S`` is treated as
# unrelated long-term memory, not a sequence).
TEMPORAL_DWELL_S: float = 10.0
TEMPORAL_RECENCY_S: float = 3600.0
TEMPORAL_MAX_EDGES_PER_NEW_BELIEF: int = 2

# Causal writer (P1.3): subscribed to WORLD_MODEL_PREDICTION_VALIDATED.
# We require ``prediction_confidence >= CAUSAL_MIN_CONFIDENCE`` before
# emitting any edge so low-confidence rule firings do not pollute the
# graph. We then map the label tokens back to active beliefs; if at least
# two distinct beliefs are found, we wire the most-confident pair with a
# ``supports`` edge on hit or a ``contradicts`` edge on miss, with basis
# ``causal``. Predictions that cannot be mapped are counted as unmapped
# in bridge stats so the dashboard can show honest coverage.
CAUSAL_MIN_CONFIDENCE: float = 0.7
CAUSAL_TOKEN_MIN_LEN: int = 3
CAUSAL_MAX_CANDIDATES: int = 20
CAUSAL_EDGE_STRENGTH: float = 0.55


class GraphBridge:
    """Subscribes to Layer 5 and memory events, creates edges in the EdgeStore."""

    def __init__(self, edge_store: Any, belief_store: Any) -> None:
        import threading
        from epistemic.belief_graph.edges import EdgeStore
        from epistemic.belief_record import BeliefStore

        self._edge_store: EdgeStore = edge_store
        self._belief_store: BeliefStore = belief_store
        self._lock = threading.Lock()
        self._subscribed = False
        self._edges_created: int = 0
        self._gates_rejected: int = 0
        self._budget_suppressed: int = 0
        self._last_emitted_edge_count: int = 0
        self._temporal_sequence_edges: int = 0
        self._causal_edges_supports: int = 0
        self._causal_edges_contradicts: int = 0
        self._causal_unmapped: int = 0
        self._causal_low_confidence: int = 0

    def _emit_edge_created(self, edge_type: str, basis: str) -> None:
        try:
            from consciousness.events import event_bus, BELIEF_GRAPH_EDGE_CREATED
            event_bus.emit(
                BELIEF_GRAPH_EDGE_CREATED,
                edge_type=edge_type,
                evidence_basis=basis,
                total_created=self._edges_created,
            )
        except Exception:
            pass

    def subscribe(self) -> None:
        if self._subscribed:
            return
        try:
            from consciousness.events import (
                event_bus,
                CONTRADICTION_DETECTED,
                CONTRADICTION_RESOLVED,
                CONTRADICTION_TENSION_HELD,
                MEMORY_ASSOCIATED,
                WORLD_MODEL_PREDICTION_VALIDATED,
            )
            event_bus.on(CONTRADICTION_DETECTED, self._on_contradiction_detected)
            event_bus.on(CONTRADICTION_RESOLVED, self._on_contradiction_resolved)
            event_bus.on(CONTRADICTION_TENSION_HELD, self._on_tension_held)
            event_bus.on(MEMORY_ASSOCIATED, self._on_memory_associated)
            event_bus.on(
                WORLD_MODEL_PREDICTION_VALIDATED,
                self._on_world_model_prediction_validated,
            )
            self._subscribed = True
            logger.info(
                "GraphBridge subscribed to contradiction + memory + "
                "world-model events"
            )
        except Exception:
            logger.debug("GraphBridge could not subscribe (bus not ready)")

    # -- Layer 5: Detected contradiction -> contradicts edges ----------------

    def _on_contradiction_detected(self, **kwargs: Any) -> None:
        """CONTRADICTION_DETECTED fires for confidence_adjusted / source_separated /
        policy_penalized — all produce contradicts edges."""
        try:
            conflict_type = kwargs.get("conflict_type", "")
            with self._lock:
                self._create_edges_from_recent_resolution(conflict_type, "contradicts")
        except Exception:
            logger.exception("GraphBridge._on_contradiction_detected error")

    # -- Layer 5: Resolution outcome -> derived_from -----------------------

    def _on_contradiction_resolved(self, **kwargs: Any) -> None:
        try:
            conflict_type = kwargs.get("conflict_type", "")
            action = kwargs.get("action", "")
            if action in ("belief_versioned", "temporal_versioned", "belief_superseded"):
                with self._lock:
                    self._create_edges_from_recent_resolution(conflict_type, "derived_from")
        except Exception:
            logger.exception("GraphBridge._on_contradiction_resolved error")

    def _create_edges_from_recent_resolution(self, conflict_type: str, edge_type: str) -> None:
        from epistemic.belief_graph.edges import make_edge
        active = self._belief_store.get_active_beliefs()
        now = time.time()
        for belief in active:
            if not belief.contradicts:
                continue
            if now - belief.timestamp > 120:
                continue
            for other_id in belief.contradicts:
                other = self._belief_store.get(other_id)
                if other is None:
                    continue
                strength = 0.6 if edge_type == "derived_from" else 0.7
                edge = make_edge(
                    source_belief_id=belief.belief_id,
                    target_belief_id=other_id,
                    edge_type=edge_type,
                    strength=strength,
                    provenance=belief.provenance,
                    evidence_basis="resolution_outcome",
                )
                if self._edge_store.add(edge):
                    self._edges_created += 1
                    self._emit_edge_created(edge_type, "resolution_outcome")

    # -- Layer 5: Tension held -> refines edges between members -------------

    def _on_tension_held(self, **kwargs: Any) -> None:
        try:
            tension_id = kwargs.get("tension_id")
            if not tension_id:
                return
            tension = self._belief_store.get_tension(tension_id)
            if tension is None or len(tension.belief_ids) < 2:
                return
            from epistemic.belief_graph.edges import make_edge
            ids = tension.belief_ids[:10]
            with self._lock:
                for i, a_id in enumerate(ids):
                    a = self._belief_store.get(a_id)
                    if a is None or a.resolution_state not in ("active", "tension"):
                        continue
                    for b_id in ids[i + 1:]:
                        b = self._belief_store.get(b_id)
                        if b is None or b.resolution_state not in ("active", "tension"):
                            continue
                        edge = make_edge(
                            source_belief_id=a_id, target_belief_id=b_id,
                            edge_type="refines", strength=0.4,
                            provenance=a.provenance, evidence_basis="resolution_outcome",
                        )
                        self._edge_store.add(edge)
                        edge_rev = make_edge(
                            source_belief_id=b_id, target_belief_id=a_id,
                            edge_type="refines", strength=0.4,
                            provenance=b.provenance, evidence_basis="resolution_outcome",
                        )
                        self._edge_store.add(edge_rev)
                        self._edges_created += 2
                        self._emit_edge_created("refines", "resolution_outcome")
        except Exception:
            logger.exception("GraphBridge._on_tension_held error")

    # -- Memory associations -> supports edges ------------------------------

    def _on_memory_associated(self, **kwargs: Any) -> None:
        try:
            memory_a_id = kwargs.get("memory_a_id") or kwargs.get("memory_id_a")
            memory_b_id = kwargs.get("memory_b_id") or kwargs.get("memory_id_b")
            if not memory_a_id or not memory_b_id:
                return
            with self._lock:
                beliefs_a = self._find_beliefs_by_source_memory(memory_a_id)
                beliefs_b = self._find_beliefs_by_source_memory(memory_b_id)
                if not beliefs_a or not beliefs_b:
                    return
                from epistemic.belief_graph.edges import make_edge
                for a in beliefs_a[:3]:
                    for b in beliefs_b[:3]:
                        if a.belief_id == b.belief_id:
                            continue
                        if a.resolution_state != "active" or b.resolution_state != "active":
                            continue
                        edge = make_edge(
                            source_belief_id=a.belief_id, target_belief_id=b.belief_id,
                            edge_type="supports", strength=0.4,
                            provenance=a.provenance, evidence_basis="memory_association",
                        )
                        if self._edge_store.add(edge):
                            self._edges_created += 1
                            self._emit_edge_created("supports", "memory_association")
        except Exception:
            logger.exception("GraphBridge._on_memory_associated error")

    def _find_beliefs_by_source_memory(self, memory_id: str) -> list:
        active = self._belief_store.get_active_beliefs()
        return [b for b in active if b.source_memory_id == memory_id]

    # -- Same-source extraction -> extractor_link edges ---------------------

    def create_extraction_links(self, new_belief: Any) -> None:
        from epistemic.belief_graph.edges import make_edge
        if not new_belief.source_memory_id:
            return
        siblings = self._find_beliefs_by_source_memory(new_belief.source_memory_id)
        for sibling in siblings:
            if sibling.belief_id == new_belief.belief_id:
                continue
            if sibling.resolution_state != "active":
                continue
            edge = make_edge(
                source_belief_id=new_belief.belief_id, target_belief_id=sibling.belief_id,
                edge_type="supports", strength=0.3,
                provenance=new_belief.provenance, evidence_basis="extractor_link",
            )
            if self._edge_store.add(edge):
                self._edges_created += 1
                self._emit_edge_created("supports", "extractor_link")

    # -- Gated + budgeted shared-subject support ----------------------------

    def _count_subject_support_edges(self, subject: str) -> int:
        beliefs = self._belief_store.find_by_subject(subject)
        belief_ids = {b.belief_id for b in beliefs}
        count = 0
        for bid in belief_ids:
            for edge in self._edge_store.get_outgoing(bid):
                if edge.edge_type == "supports" and edge.evidence_basis == "shared_subject":
                    if edge.target_belief_id in belief_ids:
                        count += 1
        return count

    def create_shared_subject_support(self, new_belief: Any) -> None:
        """Gated + budgeted support edges from compatible predicates on same subject.

        Budget: per-subject cap, per-belief cap, candidate list cap.
        Gate: AND logic (all conditions must pass).
        """
        from epistemic.belief_graph.edges import make_edge

        subject_edge_count = self._count_subject_support_edges(new_belief.canonical_subject)
        if subject_edge_count >= MAX_SUPPORT_EDGES_PER_SUBJECT:
            self._budget_suppressed += 1
            return

        candidates = self._belief_store.find_by_subject(new_belief.canonical_subject)
        candidates = sorted(
            candidates,
            key=lambda b: getattr(b, "extraction_confidence", 0.0),
            reverse=True,
        )[:MAX_CANDIDATES_PER_SUBJECT]

        edges_added = 0
        for candidate in candidates:
            if candidate.belief_id == new_belief.belief_id:
                continue
            if candidate.resolution_state != "active":
                continue
            if candidate.source_memory_id == new_belief.source_memory_id:
                continue
            if edges_added >= MAX_SUPPORT_EDGES_PER_BELIEF:
                self._budget_suppressed += 1
                break
            if subject_edge_count + edges_added >= MAX_SUPPORT_EDGES_PER_SUBJECT:
                self._budget_suppressed += 1
                break

            new_subj = getattr(new_belief, "identity_subject_id", "")
            cand_subj = getattr(candidate, "identity_subject_id", "")
            if new_subj and cand_subj and new_subj != cand_subj:
                new_subj_type = getattr(new_belief, "identity_subject_type", "")
                cand_subj_type = getattr(candidate, "identity_subject_type", "")
                if new_subj_type not in ("environment", "library") and cand_subj_type not in ("environment", "library"):
                    self._gates_rejected += 1
                    continue

            if not self._passes_support_gate(new_belief, candidate):
                self._gates_rejected += 1
                continue

            if self._are_predicates_compatible(new_belief, candidate):
                edge = make_edge(
                    source_belief_id=new_belief.belief_id,
                    target_belief_id=candidate.belief_id,
                    edge_type="supports", strength=0.5,
                    provenance=new_belief.provenance,
                    evidence_basis="shared_subject",
                )
                if self._edge_store.add(edge):
                    self._edges_created += 1
                    self._emit_edge_created("supports", "shared_subject")
                    edges_added += 1

    def _passes_support_gate(self, a: Any, b: Any) -> bool:
        """AND logic: all three conditions must pass."""
        a_family = a.conflict_key.split("::")[0] if "::" in a.conflict_key else a.conflict_key
        b_family = b.conflict_key.split("::")[0] if "::" in b.conflict_key else b.conflict_key
        if a_family != b_family:
            return False
        if a.polarity != b.polarity or a.modality != b.modality:
            return False
        effective_min_conf = SUPPORT_GATE_MIN_EXTRACTION_CONFIDENCE
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            effective_min_conf += get_quarantine_pressure().graph_support_gate_delta()
        except Exception:
            pass
        if (a.extraction_confidence < effective_min_conf
                or b.extraction_confidence < effective_min_conf):
            return False
        return True

    def _are_predicates_compatible(self, a: Any, b: Any) -> bool:
        if a.polarity != b.polarity:
            return False
        if a.canonical_predicate == b.canonical_predicate:
            return True
        _COMPATIBLE_FAMILIES = {
            frozenset({"improves", "helps", "boosts", "enhances", "increases"}),
            frozenset({"degrades", "hurts", "reduces", "decreases", "limits"}),
            frozenset({"is", "equals", "represents", "constitutes"}),
            frozenset({"prefers", "favors", "chooses", "wants"}),
        }
        for family in _COMPATIBLE_FAMILIES:
            if a.canonical_predicate in family and b.canonical_predicate in family:
                return True
        return False

    # -- Temporal versioning -> derived_from --------------------------------

    def create_version_link(self, old_belief_id: str, new_belief_id: str) -> None:
        from epistemic.belief_graph.edges import make_edge
        new_belief = self._belief_store.get(new_belief_id)
        if new_belief is None:
            return
        edge = make_edge(
            source_belief_id=new_belief_id, target_belief_id=old_belief_id,
            edge_type="derived_from", strength=0.6,
            provenance=new_belief.provenance, evidence_basis="belief_version",
        )
        if self._edge_store.add(edge):
            self._edges_created += 1
            self._emit_edge_created("derived_from", "belief_version")

    # -- Prerequisite tracking -> depends_on --------------------------------

    def create_prerequisite_link(
        self,
        dependent_belief_id: str,
        prerequisite_belief_id: str,
        strength: float = 0.6,
        evidence_basis: str = "causal",
    ) -> bool:
        """Emit a ``depends_on`` edge (dependent -> prerequisite).

        Semantics (see ``edges.py::EvidenceEdge``): the dependent belief
        *requires* the prerequisite. If the prerequisite weakens, the
        dependent's effective confidence drops (propagation.py); dangling
        prerequisites show up in ``integrity.py``; topology exposes the
        graph via ``get_dependents`` / ``get_prerequisites``.

        Guards:
          * both belief IDs must be non-empty and distinct;
          * both beliefs must exist in the belief store;
          * the new edge must **not** close a forward-dependency cycle
            (i.e. the prerequisite must not already transitively depend on
            the dependent).

        Returns True iff an edge was created or merged.
        """
        from epistemic.belief_graph.edges import make_edge

        if not dependent_belief_id or not prerequisite_belief_id:
            return False
        if dependent_belief_id == prerequisite_belief_id:
            return False

        dependent = self._belief_store.get(dependent_belief_id)
        prerequisite = self._belief_store.get(prerequisite_belief_id)
        if dependent is None or prerequisite is None:
            logger.debug(
                "create_prerequisite_link: belief missing "
                "(dep=%s, prereq=%s)",
                dependent_belief_id[:8] if dependent_belief_id else "",
                prerequisite_belief_id[:8] if prerequisite_belief_id else "",
            )
            return False

        if self._would_form_dependency_cycle(
            dependent_belief_id, prerequisite_belief_id
        ):
            self._gates_rejected += 1
            logger.debug(
                "create_prerequisite_link: cycle rejected (%s -> %s)",
                dependent_belief_id[:8], prerequisite_belief_id[:8],
            )
            return False

        edge = make_edge(
            source_belief_id=dependent_belief_id,
            target_belief_id=prerequisite_belief_id,
            edge_type="depends_on",
            strength=max(0.0, min(1.0, strength)),
            provenance=getattr(dependent, "provenance", "prerequisite_tracker"),
            evidence_basis=evidence_basis,
        )
        if self._edge_store.add(edge):
            self._edges_created += 1
            self._emit_edge_created("depends_on", evidence_basis)
            return True
        return False

    def _would_form_dependency_cycle(
        self, dependent_id: str, prerequisite_id: str
    ) -> bool:
        """Return True iff adding (dependent -> prerequisite) closes a cycle.

        A cycle exists when the prerequisite already transitively depends
        on the dependent through existing ``depends_on`` edges. We walk the
        prerequisite's outgoing dependency chain; if we reach the dependent
        before running out of nodes, adding the new edge would close it.
        """
        if dependent_id == prerequisite_id:
            return True
        visited: set[str] = set()
        stack: list[str] = [prerequisite_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            try:
                outgoing = self._edge_store.get_outgoing(current)
            except Exception:
                break
            for edge in outgoing:
                if edge.edge_type != "depends_on":
                    continue
                nxt = edge.target_belief_id
                if nxt == dependent_id:
                    return True
                if nxt not in visited:
                    stack.append(nxt)
        return False

    # -- User correction -> contradicts (decay-immune) ----------------------

    def create_user_correction_link(
        self,
        corrected_belief_id: str,
        correcting_belief_id: str,
        strength: float = 0.9,
    ) -> bool:
        """Emit a ``contradicts`` edge with evidence_basis=``user_correction``.

        Direction: correcting belief -> corrected belief. The decay-immunity
        gate at ``edges.py:226`` protects user-correction edges from passive
        decay, so operator corrections survive across time even when the
        surrounding context weakens.

        Returns True if the edge was created or merged with an existing one,
        False otherwise (missing beliefs, same-id guard, or store failure).
        """
        from epistemic.belief_graph.edges import make_edge

        if not corrected_belief_id or not correcting_belief_id:
            return False
        if corrected_belief_id == correcting_belief_id:
            return False

        corrected = self._belief_store.get(corrected_belief_id)
        correcting = self._belief_store.get(correcting_belief_id)
        if corrected is None or correcting is None:
            logger.debug(
                "create_user_correction_link: belief missing "
                "(corrected=%s, correcting=%s)",
                corrected_belief_id[:8] if corrected_belief_id else "",
                correcting_belief_id[:8] if correcting_belief_id else "",
            )
            return False

        edge = make_edge(
            source_belief_id=correcting_belief_id,
            target_belief_id=corrected_belief_id,
            edge_type="contradicts",
            strength=max(0.0, min(1.0, strength)),
            provenance=getattr(correcting, "provenance", "user_correction"),
            evidence_basis="user_correction",
        )
        if self._edge_store.add(edge):
            self._edges_created += 1
            self._emit_edge_created("contradicts", "user_correction")
            return True
        return False

    # -- Edge pruning for over-budget subjects ------------------------------

    def prune_subject_edges(self, subject: str) -> int:
        """Remove weakest shared-subject support edges when over budget."""
        beliefs = self._belief_store.find_by_subject(subject)
        belief_ids = {b.belief_id for b in beliefs}
        subject_edges: list[tuple[str, float]] = []
        for bid in belief_ids:
            for edge in self._edge_store.get_outgoing(bid):
                if (edge.edge_type == "supports"
                        and edge.evidence_basis == "shared_subject"
                        and edge.target_belief_id in belief_ids):
                    subject_edges.append((edge.edge_id, edge.strength))
        if len(subject_edges) <= MAX_SUPPORT_EDGES_PER_SUBJECT:
            return 0
        subject_edges.sort(key=lambda x: x[1], reverse=True)
        to_remove = subject_edges[MAX_SUPPORT_EDGES_PER_SUBJECT:]
        pruned = 0
        for edge_id, _ in to_remove:
            if self._edge_store.remove(edge_id):
                pruned += 1
        if pruned:
            logger.info("Pruned %d weak shared-subject edges for subject=%s", pruned, subject)
        return pruned

    # -- Retroactive orphan edge fill ----------------------------------------

    def fill_orphan_edges(self, max_per_cycle: int = 30) -> int:
        """Create shared-subject support edges for active beliefs that have
        zero edges in the graph.  Runs during dream cycles to reduce orphan
        rate after rehydration.

        Returns the number of new edges created.
        """
        from epistemic.belief_graph.edges import make_edge

        graph_belief_ids: set[str] = set()
        for edge in self._edge_store._edges.values():
            graph_belief_ids.add(edge.source_belief_id)
            graph_belief_ids.add(edge.target_belief_id)

        active = self._belief_store.get_active_beliefs()
        orphans = [b for b in active if b.belief_id not in graph_belief_ids]
        if not orphans:
            return 0

        created = 0
        with self._lock:
            for orphan in orphans:
                if created >= max_per_cycle:
                    break
                if not orphan.canonical_subject:
                    continue
                candidates = self._belief_store.find_by_subject(orphan.canonical_subject)
                for cand in candidates[:5]:
                    if cand.belief_id == orphan.belief_id:
                        continue
                    if cand.resolution_state != "active":
                        continue
                    if not self._are_predicates_compatible(orphan, cand):
                        continue
                    edge = make_edge(
                        source_belief_id=orphan.belief_id,
                        target_belief_id=cand.belief_id,
                        edge_type="supports", strength=0.35,
                        provenance=orphan.provenance,
                        evidence_basis="orphan_fill",
                    )
                    if self._edge_store.add(edge):
                        self._edges_created += 1
                        created += 1
                        break
        if created:
            logger.info("Orphan edge fill: created %d edges for %d orphan beliefs", created, len(orphans))
        return created

    # -- Temporal-sequence edges (basis="temporal_sequence") ----------------

    def create_temporal_sequence_links(self, new_belief: Any) -> int:
        """Emit ``derived_from`` edges with basis ``temporal_sequence``.

        For each new active belief, find prior active beliefs about the
        same canonical subject that are older than ``TEMPORAL_DWELL_S``
        (so we do not collapse a single batched extraction into a "sequence")
        and newer than ``TEMPORAL_RECENCY_S`` (so we do not link arbitrary
        long-term memory). Cap per-call at
        ``TEMPORAL_MAX_EDGES_PER_NEW_BELIEF``.

        Returns the number of edges written. This writer is intentionally
        narrow — it only encodes pure temporal precedence over the same
        subject; it does NOT make any content-relationship claim. That
        keeps ``temporal_sequence`` honest for the schema audit.
        """
        from epistemic.belief_graph.edges import make_edge

        subject = getattr(new_belief, "canonical_subject", "")
        if not subject:
            return 0
        new_ts = float(getattr(new_belief, "timestamp", 0.0) or 0.0)
        if new_ts <= 0.0:
            return 0

        candidates = list(self._belief_store.find_by_subject(subject))
        candidates.sort(
            key=lambda b: float(getattr(b, "timestamp", 0.0) or 0.0),
            reverse=True,
        )

        written = 0
        for prior in candidates:
            if written >= TEMPORAL_MAX_EDGES_PER_NEW_BELIEF:
                break
            if getattr(prior, "belief_id", None) == getattr(new_belief, "belief_id", None):
                continue
            if getattr(prior, "resolution_state", None) != "active":
                continue
            prior_ts = float(getattr(prior, "timestamp", 0.0) or 0.0)
            if prior_ts <= 0.0:
                continue
            age = new_ts - prior_ts
            if age < TEMPORAL_DWELL_S or age > TEMPORAL_RECENCY_S:
                continue

            edge = make_edge(
                source_belief_id=new_belief.belief_id,
                target_belief_id=prior.belief_id,
                edge_type="derived_from",
                strength=0.3,
                provenance=getattr(new_belief, "provenance", "unknown"),
                evidence_basis="temporal_sequence",
            )
            if self._edge_store.add(edge):
                self._edges_created += 1
                self._temporal_sequence_edges += 1
                written += 1
                self._emit_edge_created("derived_from", "temporal_sequence")
        return written

    # -- Causal edges (basis="causal") --------------------------------------

    def _on_world_model_prediction_validated(self, **kwargs: Any) -> None:
        """Subscriber for ``WORLD_MODEL_PREDICTION_VALIDATED``.

        On a validated prediction we attempt to map ``prediction_label``
        tokens to two distinct active beliefs and wire them with a single
        ``causal``-basis edge:

          * outcome ``hit``  → ``supports`` edge between the two beliefs.
          * outcome ``miss`` → ``contradicts`` edge between the two beliefs.

        Predictions below ``CAUSAL_MIN_CONFIDENCE`` are silently skipped
        (counted as ``causal_low_confidence``). Predictions whose label
        cannot be mapped to at least two active beliefs are counted as
        ``causal_unmapped`` so the dashboard can show honest coverage of
        the causal writer (i.e. how often live world-model events actually
        produce graph edges vs. how often they are scaffolding-only).
        """
        try:
            label = str(kwargs.get("prediction_label") or "")
            outcome = str(kwargs.get("outcome") or "").lower()
            confidence = float(kwargs.get("prediction_confidence") or 0.0)
            if outcome not in ("hit", "miss"):
                return
            if confidence < CAUSAL_MIN_CONFIDENCE:
                with self._lock:
                    self._causal_low_confidence += 1
                return
            if not label:
                return
            with self._lock:
                self.record_causal_observation(
                    label=label, outcome=outcome, confidence=confidence,
                )
        except Exception:
            logger.exception(
                "GraphBridge._on_world_model_prediction_validated error"
            )

    def record_causal_observation(
        self, label: str, outcome: str, confidence: float,
    ) -> int:
        """Public method used by both the live subscriber and tests.

        Returns the number of edges written (0 or 1). This method must be
        called with ``self._lock`` already held when used from the live
        event path; tests may call it directly (unlocked) since they own
        the bridge instance.
        """
        from epistemic.belief_graph.edges import make_edge

        tokens = [
            t for t in str(label).replace("-", "_").split("_")
            if len(t) >= CAUSAL_TOKEN_MIN_LEN
        ]
        if not tokens:
            self._causal_unmapped += 1
            return 0

        active = list(self._belief_store.get_active_beliefs())
        scored: list[tuple[float, Any]] = []
        for belief in active:
            text = " ".join(
                str(getattr(belief, attr, "") or "")
                for attr in (
                    "canonical_subject",
                    "canonical_predicate",
                    "canonical_object",
                    "rendered_claim",
                )
            ).lower()
            if not text:
                continue
            score = sum(1 for t in tokens if t in text)
            if score <= 0:
                continue
            scored.append(
                (score + float(getattr(belief, "extraction_confidence", 0.0)), belief)
            )

        if len(scored) < 2:
            self._causal_unmapped += 1
            return 0

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:CAUSAL_MAX_CANDIDATES]
        a = scored[0][1]
        b = scored[1][1]

        edge_type = "supports" if outcome == "hit" else "contradicts"
        edge = make_edge(
            source_belief_id=a.belief_id,
            target_belief_id=b.belief_id,
            edge_type=edge_type,
            strength=max(0.0, min(1.0, CAUSAL_EDGE_STRENGTH * confidence)),
            provenance=getattr(a, "provenance", "world_model"),
            evidence_basis="causal",
        )
        if not self._edge_store.add(edge):
            return 0

        self._edges_created += 1
        if edge_type == "supports":
            self._causal_edges_supports += 1
        else:
            self._causal_edges_contradicts += 1
        self._emit_edge_created(edge_type, "causal")
        return 1

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "edges_created": self._edges_created,
                "gates_rejected": self._gates_rejected,
                "budget_suppressed": self._budget_suppressed,
                "subscribed": self._subscribed,
                "temporal_sequence_edges": self._temporal_sequence_edges,
                "causal_edges_supports": self._causal_edges_supports,
                "causal_edges_contradicts": self._causal_edges_contradicts,
                "causal_unmapped": self._causal_unmapped,
                "causal_low_confidence": self._causal_low_confidence,
            }
