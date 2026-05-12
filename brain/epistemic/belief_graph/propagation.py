"""Confidence propagation for the Belief Confidence Graph (Layer 7).

Propagation is VIEW-ONLY: it computes effective_confidence and
structural_confidence_delta for each belief but NEVER writes back
into BeliefRecord.belief_confidence.

Single-pass weighted propagation using incoming edges.

Sacred invariants:
1. Identity-tension beliefs are immune to propagation.
2. user_claim provenance beliefs have a floor of 0.5.
3. refines edges do NOT contribute to effective confidence.
4. Propagation NEVER mutates BeliefRecord.
5. Confidence stays in [0, 1].
6. Orphan beliefs keep base_confidence as effective_confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeliefConfidenceView:
    """View-only confidence snapshot for a single belief."""
    belief_id: str
    base_confidence: float
    effective_confidence: float
    structural_confidence_delta: float


SUPPORT_WEIGHT: float = 0.12
CONTRADICT_WEIGHT: float = 0.15
DEPENDS_ON_WEIGHT: float = 0.08
DERIVED_FROM_WEIGHT: float = 0.05

USER_CLAIM_FLOOR: float = 0.5
MAX_PROPAGATION_DELTA: float = 0.3


def propagate_all(
    edge_store: Any,
    belief_store: Any,
) -> dict[str, BeliefConfidenceView]:
    """Compute effective confidence for all active beliefs.

    Returns a dict of belief_id -> BeliefConfidenceView.
    This is a pure function: no side effects, no mutations.
    """
    active_beliefs = belief_store.get_active_beliefs()
    views: dict[str, BeliefConfidenceView] = {}

    belief_map = {b.belief_id: b for b in active_beliefs}

    tension_belief_ids: set[str] = set()
    for t in belief_store.get_active_tensions():
        tension_belief_ids.update(t.belief_ids)

    for belief in active_beliefs:
        base = belief.belief_confidence

        if belief.belief_id in tension_belief_ids:
            views[belief.belief_id] = BeliefConfidenceView(
                belief_id=belief.belief_id,
                base_confidence=base,
                effective_confidence=base,
                structural_confidence_delta=0.0,
            )
            continue

        delta = _compute_structural_delta(
            belief.belief_id, edge_store, belief_map,
        )

        delta = max(-MAX_PROPAGATION_DELTA, min(MAX_PROPAGATION_DELTA, delta))
        effective = base + delta
        effective = max(0.0, min(1.0, effective))

        if belief.provenance == "user_claim" and effective < USER_CLAIM_FLOOR:
            effective = USER_CLAIM_FLOOR
            delta = effective - base

        views[belief.belief_id] = BeliefConfidenceView(
            belief_id=belief.belief_id,
            base_confidence=base,
            effective_confidence=round(effective, 4),
            structural_confidence_delta=round(delta, 4),
        )

    return views


def _compute_structural_delta(
    belief_id: str,
    edge_store: Any,
    belief_map: dict[str, Any],
) -> float:
    """Compute the structural delta from incoming edges and outgoing depends_on.

    refines edges are excluded from propagation.

    depends_on direction: dependent -> prerequisite (outgoing from the belief).
    If the prerequisite (target) is weak, the dependent (source=this belief) is
    penalized.  So we check outgoing depends_on edges.
    """
    incoming = edge_store.get_incoming(belief_id)
    outgoing = edge_store.get_outgoing(belief_id)

    if not incoming and not outgoing:
        return 0.0

    delta = 0.0

    for edge in incoming:
        if edge.edge_type == "refines":
            continue

        source = belief_map.get(edge.source_belief_id)
        if source is None:
            continue

        source_conf = source.belief_confidence

        if edge.edge_type == "supports":
            delta += SUPPORT_WEIGHT * edge.strength * source_conf
        elif edge.edge_type == "contradicts":
            delta -= CONTRADICT_WEIGHT * edge.strength * source_conf
        elif edge.edge_type == "derived_from":
            delta += DERIVED_FROM_WEIGHT * edge.strength * source_conf

    for edge in outgoing:
        if edge.edge_type != "depends_on":
            continue
        prerequisite = belief_map.get(edge.target_belief_id)
        if prerequisite is None:
            delta -= DEPENDS_ON_WEIGHT * edge.strength * 0.3
            continue
        prereq_conf = prerequisite.belief_confidence
        if prereq_conf < 0.3:
            delta -= DEPENDS_ON_WEIGHT * edge.strength * (0.3 - prereq_conf)

    return delta


def propagate_neighborhood(
    belief_id: str,
    edge_store: Any,
    belief_store: Any,
) -> BeliefConfidenceView | None:
    """Compute effective confidence for a single belief (dirty-neighborhood).

    Used for incremental updates instead of full propagation.
    """
    belief = belief_store.get(belief_id)
    if belief is None or belief.resolution_state != "active":
        return None

    tension_ids: set[str] = set()
    for t in belief_store.get_active_tensions():
        tension_ids.update(t.belief_ids)

    base = belief.belief_confidence

    if belief_id in tension_ids:
        return BeliefConfidenceView(
            belief_id=belief_id,
            base_confidence=base,
            effective_confidence=base,
            structural_confidence_delta=0.0,
        )

    active = belief_store.get_active_beliefs()
    belief_map = {b.belief_id: b for b in active}

    delta = _compute_structural_delta(belief_id, edge_store, belief_map)
    delta = max(-MAX_PROPAGATION_DELTA, min(MAX_PROPAGATION_DELTA, delta))
    effective = max(0.0, min(1.0, base + delta))

    if belief.provenance == "user_claim" and effective < USER_CLAIM_FLOOR:
        effective = USER_CLAIM_FLOOR
        delta = effective - base

    return BeliefConfidenceView(
        belief_id=belief_id,
        base_confidence=base,
        effective_confidence=round(effective, 4),
        structural_confidence_delta=round(delta, 4),
    )
