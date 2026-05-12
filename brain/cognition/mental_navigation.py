"""Mental navigation shadow — P5 Commit 8.

Pure simulation layer over :class:`cognition.spatial_scene_graph.MentalWorldSceneGraph`.
Lets the cognitive layer answer "what would the mental scene look like if
I..." questions without ever touching canonical perception, memory,
beliefs, policy, or the autonomy subsystem. Every ``simulate_*`` function:

* accepts a frozen :class:`MentalWorldSceneGraph` and returns a **new**
  frozen graph (the input is never mutated);
* records a :class:`MentalNavigationTrace` describing the hypothetical
  operation, with explicit ``applied`` / ``reason`` fields so callers
  (and the dashboard) can see *why* an imagined move didn't change
  anything;
* runs purely on the shadow graph: no HRR math, no re-encoding, no
  canonical writes, no policy influence.

Exposed ops (mirroring the P5 spatial symbol vocabulary):

* :func:`simulate_turn_left` / :func:`simulate_turn_right` — rotate
  entity regions around the camera viewpoint.
* :func:`simulate_move_forward` — approximate a forward step; bring
  "far" regions closer, push "near" entities behind.
* :func:`simulate_occlude` — hide a named entity, adding an
  ``occluded_by`` derived relation.
* :func:`simulate_return_to_last_seen` — re-surface a remembered entity
  back to its last-seen region as ``expected_in_view``.

Each op also produces a :class:`MentalNavigationTrace` whose metrics go
into the P5 status block. The :class:`MentalNavigationShadow` ring
stores recent traces, twin-gated by
:data:`HRRRuntimeConfig.enabled` *and*
:data:`HRRRuntimeConfig.spatial_scene_active`. Default off.

**Zero authority**: never writes canonical state. The module-level
``AUTHORITY_FLAGS`` mirrors the mental-world facade and is surfaced in
every status payload so the dashboard / truth-probe can verify.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from cognition.spatial_scene_graph import (
    MentalWorldEntity,
    MentalWorldRelation,
    MentalWorldSceneGraph,
)


# ---------------------------------------------------------------------------
# Authority contract — pinned false everywhere (mirror mental_world facade).
# ---------------------------------------------------------------------------

AUTHORITY_FLAGS: Dict[str, bool] = {
    "writes_memory": False,
    "writes_beliefs": False,
    "influences_policy": False,
    "influences_autonomy": False,
    "soul_integrity_influence": False,
    "llm_raw_vector_exposure": False,
    "no_raw_vectors_in_api": True,
}


# ---------------------------------------------------------------------------
# Action vocabulary (mirror library.vsa.spatial_symbols.MENTAL_NAV_SYMBOLS).
# ---------------------------------------------------------------------------

ACTION_TURN_LEFT = "turn_left"
ACTION_TURN_RIGHT = "turn_right"
ACTION_MOVE_FORWARD = "move_forward"
ACTION_OBJECT_OCCLUDED = "object_occluded"
ACTION_RETURN_TO_LAST_SEEN = "return_to_last_seen"

SUPPORTED_ACTIONS: Tuple[str, ...] = (
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_MOVE_FORWARD,
    ACTION_OBJECT_OCCLUDED,
    ACTION_RETURN_TO_LAST_SEEN,
)


# Region rotation maps — purely presentational; operate on the semantic
# region labels emitted by perception.scene_regions, never on raw pixels
# or 2.5D coords.
_TURN_LEFT_REGION_MAP: Dict[str, str] = {
    "desk_right": "desk_center",
    "desk_center": "desk_left",
    "desk_far_right": "desk_far",
    "desk_far": "desk_far_left",
    "desk_near_right": "desk_near",
    "desk_near": "desk_near_left",
}
_TURN_RIGHT_REGION_MAP: Dict[str, str] = {
    "desk_left": "desk_center",
    "desk_center": "desk_right",
    "desk_far_left": "desk_far",
    "desk_far": "desk_far_right",
    "desk_near_left": "desk_near",
    "desk_near": "desk_near_right",
}

# Move-forward brings far regions into the center field and pushes near
# regions behind the camera (state flips to ``out_of_view``).
_MOVE_FORWARD_REGION_MAP: Dict[str, str] = {
    "desk_far_left": "desk_left",
    "desk_far": "desk_center",
    "desk_far_right": "desk_right",
}
_MOVE_FORWARD_DROP_REGIONS: Tuple[str, ...] = (
    "desk_near_left",
    "desk_near",
    "desk_near_right",
)


# ---------------------------------------------------------------------------
# Trace + shadow ring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MentalNavigationTrace:
    """Read-only record of a single simulated mental-navigation step."""
    action: str
    timestamp: float
    applied: bool
    reason: Optional[str]
    before: Dict[str, Any]
    after: Dict[str, Any]
    entity_deltas: Tuple[Dict[str, Any], ...]
    target_entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "timestamp": round(self.timestamp, 3),
            "applied": bool(self.applied),
            "reason": self.reason,
            "target_entity_id": self.target_entity_id,
            "before": dict(self.before),
            "after": dict(self.after),
            "entity_deltas": [dict(d) for d in self.entity_deltas],
            **AUTHORITY_FLAGS,
        }


@dataclass
class MentalNavigationShadow:
    """Ring-buffered owner of :class:`MentalNavigationTrace` samples.

    Twin-gated: ``enabled`` is ``True`` only when both the HRR substrate
    and the spatial-scene lane are active. When disabled, every
    ``record`` call is a no-op.
    """
    capacity: int = 64
    enabled: bool = False
    _ring: Deque[MentalNavigationTrace] = field(default_factory=lambda: deque(maxlen=64))
    _total: int = 0
    _applied_count: int = 0
    _last_action: Optional[str] = None
    _last_timestamp: float = 0.0

    def __post_init__(self) -> None:
        cap = max(1, int(self.capacity))
        self.capacity = cap
        if self._ring.maxlen != cap:
            self._ring = deque(self._ring, maxlen=cap)

    def record(self, trace: MentalNavigationTrace) -> None:
        """Append a trace to the ring. No-op when the shadow is disabled."""
        if not self.enabled:
            return
        if not isinstance(trace, MentalNavigationTrace):
            raise TypeError("MentalNavigationShadow.record: trace must be MentalNavigationTrace")
        self._ring.append(trace)
        self._total += 1
        if trace.applied:
            self._applied_count += 1
        self._last_action = trace.action
        self._last_timestamp = trace.timestamp

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the most-recent ``n`` trace dicts (newest last)."""
        if not self.enabled:
            return []
        n = max(0, min(self.capacity, int(n)))
        if n == 0:
            return []
        items = list(self._ring)[-n:]
        return [t.to_dict() for t in items]

    def status(self) -> Dict[str, Any]:
        """Status block mounted under ``/api/hrr/status.mental_navigation``."""
        return {
            "status": "PRE-MATURE",
            "lane": "spatial_hrr_mental_world",
            "enabled": bool(self.enabled),
            "capacity": self.capacity,
            "total_recorded": int(self._total),
            "applied_count": int(self._applied_count),
            "last_action": self._last_action,
            "last_timestamp": round(self._last_timestamp, 3) if self._last_timestamp else 0.0,
            "supported_actions": list(SUPPORTED_ACTIONS),
            **AUTHORITY_FLAGS,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_entity(e: MentalWorldEntity, **changes: Any) -> MentalWorldEntity:
    """Return a new MentalWorldEntity with the given fields swapped."""
    return MentalWorldEntity(
        entity_id=changes.get("entity_id", e.entity_id),
        label=changes.get("label", e.label),
        state=changes.get("state", e.state),
        region=changes.get("region", e.region),
        position_room_m=changes.get("position_room_m", e.position_room_m),
        confidence=changes.get("confidence", e.confidence),
        last_seen_ts=changes.get("last_seen_ts", e.last_seen_ts),
        is_display_surface=changes.get("is_display_surface", e.is_display_surface),
    )


def _graph_summary(graph: MentalWorldSceneGraph) -> Dict[str, Any]:
    by_state: Dict[str, int] = {}
    by_region: Dict[str, int] = {}
    for e in graph.entities:
        by_state[e.state] = by_state.get(e.state, 0) + 1
        by_region[e.region] = by_region.get(e.region, 0) + 1
    return {
        "entity_count": graph.entity_count,
        "relation_count": graph.relation_count,
        "by_state": by_state,
        "by_region": by_region,
    }


def _entity_delta(
    before: MentalWorldEntity, after: MentalWorldEntity
) -> Optional[Dict[str, Any]]:
    changes: Dict[str, Any] = {}
    for field_name in ("state", "region", "confidence"):
        b = getattr(before, field_name)
        a = getattr(after, field_name)
        if b != a:
            changes[field_name] = {"before": b, "after": a}
    if not changes:
        return None
    return {"entity_id": before.entity_id, "changes": changes}


def _build_graph(
    base: MentalWorldSceneGraph,
    entities: Iterable[MentalWorldEntity],
    relations: Iterable[MentalWorldRelation],
    *,
    reason: Optional[str] = None,
) -> MentalWorldSceneGraph:
    return MentalWorldSceneGraph(
        timestamp=time.time(),
        entities=tuple(entities),
        relations=tuple(relations),
        source_scene_update_count=base.source_scene_update_count,
        source_track_count=base.source_track_count,
        source_anchor_count=base.source_anchor_count,
        source_calibration_version=base.source_calibration_version,
        reason=reason,
    )


def _validate_graph(graph: MentalWorldSceneGraph) -> None:
    if not isinstance(graph, MentalWorldSceneGraph):
        raise TypeError("simulate_*: graph must be a MentalWorldSceneGraph")


# ---------------------------------------------------------------------------
# Simulation ops
# ---------------------------------------------------------------------------


def simulate_turn_left(graph: MentalWorldSceneGraph) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    """Imagine rotating the viewpoint leftward.

    Semantic region labels shift one step to the left. Entities whose
    region has no left-neighbor in our shallow map become
    ``out_of_view`` (they left the field of view). Relations are
    carried through unchanged — this is a hypothetical, not a re-derivation.
    """
    _validate_graph(graph)
    return _rotate(graph, _TURN_LEFT_REGION_MAP, ACTION_TURN_LEFT)


def simulate_turn_right(graph: MentalWorldSceneGraph) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    """Imagine rotating the viewpoint rightward (mirror of :func:`simulate_turn_left`)."""
    _validate_graph(graph)
    return _rotate(graph, _TURN_RIGHT_REGION_MAP, ACTION_TURN_RIGHT)


def _rotate(
    graph: MentalWorldSceneGraph,
    region_map: Dict[str, str],
    action: str,
) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    new_entities: List[MentalWorldEntity] = []
    deltas: List[Dict[str, Any]] = []
    for e in graph.entities:
        if e.state != "visible":
            new_entities.append(e)
            continue
        if e.region in region_map:
            new_e = _replace_entity(e, region=region_map[e.region])
        else:
            new_e = _replace_entity(e, state="out_of_view")
        delta = _entity_delta(e, new_e)
        if delta is not None:
            deltas.append(delta)
        new_entities.append(new_e)

    new_graph = _build_graph(graph, new_entities, graph.relations)
    applied = bool(deltas)
    reason = None if applied else "no_visible_entities_to_rotate"
    trace = MentalNavigationTrace(
        action=action,
        timestamp=new_graph.timestamp,
        applied=applied,
        reason=reason,
        before=_graph_summary(graph),
        after=_graph_summary(new_graph),
        entity_deltas=tuple(deltas),
    )
    return new_graph, trace


def simulate_move_forward(graph: MentalWorldSceneGraph) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    """Imagine stepping forward.

    Entities in ``desk_far*`` regions move one step closer into the
    main field of view. Entities in ``desk_near*`` regions pass out of
    the FOV (state ``out_of_view``). All other regions are unchanged.
    """
    _validate_graph(graph)
    new_entities: List[MentalWorldEntity] = []
    deltas: List[Dict[str, Any]] = []
    for e in graph.entities:
        if e.state != "visible":
            new_entities.append(e)
            continue
        if e.region in _MOVE_FORWARD_REGION_MAP:
            new_e = _replace_entity(e, region=_MOVE_FORWARD_REGION_MAP[e.region])
        elif e.region in _MOVE_FORWARD_DROP_REGIONS:
            new_e = _replace_entity(e, state="out_of_view")
        else:
            new_e = e
        delta = _entity_delta(e, new_e)
        if delta is not None:
            deltas.append(delta)
        new_entities.append(new_e)

    new_graph = _build_graph(graph, new_entities, graph.relations)
    applied = bool(deltas)
    reason = None if applied else "no_near_or_far_entities"
    trace = MentalNavigationTrace(
        action=ACTION_MOVE_FORWARD,
        timestamp=new_graph.timestamp,
        applied=applied,
        reason=reason,
        before=_graph_summary(graph),
        after=_graph_summary(new_graph),
        entity_deltas=tuple(deltas),
    )
    return new_graph, trace


def simulate_occlude(
    graph: MentalWorldSceneGraph,
    entity_id: str,
    *,
    occluder_entity_id: Optional[str] = None,
) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    """Imagine a named entity becoming occluded.

    The target entity's state flips to ``occluded``. If
    ``occluder_entity_id`` names another entity in the graph, a derived
    ``occluded_by`` relation is added (low confidence — it is imagined,
    not observed).
    """
    _validate_graph(graph)
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("simulate_occlude: entity_id must be a non-empty string")

    ids = {e.entity_id for e in graph.entities}
    if entity_id not in ids:
        trace = MentalNavigationTrace(
            action=ACTION_OBJECT_OCCLUDED,
            timestamp=time.time(),
            applied=False,
            reason="entity_id_not_in_graph",
            before=_graph_summary(graph),
            after=_graph_summary(graph),
            entity_deltas=(),
            target_entity_id=entity_id,
        )
        return graph, trace

    new_entities: List[MentalWorldEntity] = []
    deltas: List[Dict[str, Any]] = []
    for e in graph.entities:
        if e.entity_id == entity_id and e.state != "occluded":
            new_e = _replace_entity(e, state="occluded")
            delta = _entity_delta(e, new_e)
            if delta is not None:
                deltas.append(delta)
            new_entities.append(new_e)
        else:
            new_entities.append(e)

    new_relations: List[MentalWorldRelation] = list(graph.relations)
    if occluder_entity_id and occluder_entity_id in ids and occluder_entity_id != entity_id:
        new_relations.append(
            MentalWorldRelation(
                source_entity_id=entity_id,
                target_entity_id=occluder_entity_id,
                relation_type="occluded_by",
                value_m=None,
                confidence=0.25,
            )
        )

    new_graph = _build_graph(graph, new_entities, new_relations)
    applied = bool(deltas) or len(new_relations) != len(graph.relations)
    reason = None if applied else "entity_already_occluded"
    trace = MentalNavigationTrace(
        action=ACTION_OBJECT_OCCLUDED,
        timestamp=new_graph.timestamp,
        applied=applied,
        reason=reason,
        before=_graph_summary(graph),
        after=_graph_summary(new_graph),
        entity_deltas=tuple(deltas),
        target_entity_id=entity_id,
    )
    return new_graph, trace


def simulate_return_to_last_seen(
    graph: MentalWorldSceneGraph,
    entity_id: str,
) -> Tuple[MentalWorldSceneGraph, MentalNavigationTrace]:
    """Imagine re-visiting an entity's last-seen region.

    If the target entity is currently ``missing`` or ``out_of_view``
    and has a non-empty region, its state flips to ``expected_in_view``
    and its confidence is halved (imagined, not perceived). No new
    position is invented. If the entity is already visible or has no
    remembered region, the op is no-op with a reason.
    """
    _validate_graph(graph)
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("simulate_return_to_last_seen: entity_id must be non-empty string")

    target: Optional[MentalWorldEntity] = None
    for e in graph.entities:
        if e.entity_id == entity_id:
            target = e
            break

    if target is None:
        trace = MentalNavigationTrace(
            action=ACTION_RETURN_TO_LAST_SEEN,
            timestamp=time.time(),
            applied=False,
            reason="entity_id_not_in_graph",
            before=_graph_summary(graph),
            after=_graph_summary(graph),
            entity_deltas=(),
            target_entity_id=entity_id,
        )
        return graph, trace

    if target.state not in ("missing", "out_of_view"):
        trace = MentalNavigationTrace(
            action=ACTION_RETURN_TO_LAST_SEEN,
            timestamp=time.time(),
            applied=False,
            reason="entity_not_missing_or_out_of_view",
            before=_graph_summary(graph),
            after=_graph_summary(graph),
            entity_deltas=(),
            target_entity_id=entity_id,
        )
        return graph, trace

    if not target.region:
        trace = MentalNavigationTrace(
            action=ACTION_RETURN_TO_LAST_SEEN,
            timestamp=time.time(),
            applied=False,
            reason="no_remembered_region",
            before=_graph_summary(graph),
            after=_graph_summary(graph),
            entity_deltas=(),
            target_entity_id=entity_id,
        )
        return graph, trace

    new_entities: List[MentalWorldEntity] = []
    deltas: List[Dict[str, Any]] = []
    for e in graph.entities:
        if e.entity_id == entity_id:
            new_e = _replace_entity(
                e,
                state="expected_in_view",
                confidence=max(0.0, e.confidence * 0.5),
            )
            delta = _entity_delta(e, new_e)
            if delta is not None:
                deltas.append(delta)
            new_entities.append(new_e)
        else:
            new_entities.append(e)

    new_graph = _build_graph(graph, new_entities, graph.relations)
    trace = MentalNavigationTrace(
        action=ACTION_RETURN_TO_LAST_SEEN,
        timestamp=new_graph.timestamp,
        applied=bool(deltas),
        reason=None if deltas else "state_already_expected",
        before=_graph_summary(graph),
        after=_graph_summary(new_graph),
        entity_deltas=tuple(deltas),
        target_entity_id=entity_id,
    )
    return new_graph, trace


__all__ = [
    "AUTHORITY_FLAGS",
    "ACTION_TURN_LEFT",
    "ACTION_TURN_RIGHT",
    "ACTION_MOVE_FORWARD",
    "ACTION_OBJECT_OCCLUDED",
    "ACTION_RETURN_TO_LAST_SEEN",
    "SUPPORTED_ACTIONS",
    "MentalNavigationTrace",
    "MentalNavigationShadow",
    "simulate_turn_left",
    "simulate_turn_right",
    "simulate_move_forward",
    "simulate_occlude",
    "simulate_return_to_last_seen",
]
