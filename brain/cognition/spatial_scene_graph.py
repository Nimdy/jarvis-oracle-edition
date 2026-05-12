"""Mental-world scene-graph adapter — P5 derived view, never canonical.

Pure adapter that **consumes** canonical perception outputs
(:class:`perception.scene_types.SceneSnapshot` from
:class:`perception.scene_tracker.SceneTracker` and
:class:`cognition.spatial_schema.SpatialTrack` /
:class:`cognition.spatial_schema.SpatialAnchor` from
:class:`perception.spatial.SpatialEstimator`) and emits a
:class:`MentalWorldSceneGraph` — the shadow data structure used by the P5
HRR encoder, dashboard mind-view, and Pi LCD.

Non-negotiables enforced by this module:

* **No detector path.** Accepts canonical-shaped inputs only. Never reads
  raw detections, never calls ``perception_orchestrator`` or the Pi.
* **No new geometry.** Relation math operates only on
  ``SpatialTrack.position_room_m`` / ``SpatialAnchor.position_room_m``
  coordinates that perception already produced. Threshold constants come
  from :mod:`cognition.spatial_schema`; P5 may not redefine them.
* **No canonical writes.** The returned graph is for HRR / dashboard /
  Pi consumption only. The adapter itself is a pure function.

If canonical spatial state is unavailable the adapter returns an empty
graph with ``reason="canonical_spatial_state_unavailable"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from cognition.spatial_schema import (
    CLASS_MOVE_THRESHOLDS,
    DEFAULT_MOVE_THRESHOLD,
    SpatialAnchor,
    SpatialTrack,
)
from perception.scene_types import SceneSnapshot


# ---------------------------------------------------------------------------
# Tunables (derived — never canonical; these only shape the mental view)
# ---------------------------------------------------------------------------
# Room-frame axes follow the SpatialEstimator camera_to_room projection:
#   +X = rightward from camera viewpoint (meters)
#   +Y = upward (meters)
#   +Z = forward / away from camera (meters)

# Minimum lateral delta (meters) to assert left_of/right_of between two tracks.
LATERAL_SEPARATION_MIN_M: float = 0.10

# Minimum depth delta (meters) to assert in_front_of/behind.
DEPTH_SEPARATION_MIN_M: float = 0.10

# "near" radius = max(NEAR_FLOOR_M, per-class-move-threshold * NEAR_MULTIPLIER).
# This uses CLASS_MOVE_THRESHOLDS as the sole geometry input so we don't
# invent new constants. Small objects still have a minimum grouping radius.
NEAR_MULTIPLIER: float = 3.0
NEAR_FLOOR_M: float = 0.25

# Anchor tests
ON_VERTICAL_TOLERANCE_M: float = 0.15
CENTERED_IN_FRACTION: float = 0.3


# Canonical-state unavailable reason string. Tests pin this exact value.
REASON_UNAVAILABLE: str = "canonical_spatial_state_unavailable"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MentalWorldEntity:
    """Derived mental-world projection of a :class:`SceneEntity` (+ optional track).

    Never persisted. Never exposed to policy / belief / memory / autonomy /
    LLM. Carries only what the HRR encoder and dashboard need.
    """
    entity_id: str
    label: str
    state: str  # SceneEntity.state: candidate / visible / occluded / missing / removed
    region: str  # semantic region (desk_left / desk_center / ...)
    position_room_m: Optional[tuple[float, float, float]] = None
    confidence: float = 0.0
    last_seen_ts: float = 0.0
    is_display_surface: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "state": self.state,
            "region": self.region,
            "position_room_m": list(self.position_room_m) if self.position_room_m else None,
            "confidence": round(self.confidence, 3),
            "last_seen_ts": round(self.last_seen_ts, 1),
            "is_display_surface": self.is_display_surface,
        }


@dataclass(frozen=True)
class MentalWorldRelation:
    """Directed relation between two :class:`MentalWorldEntity` instances.

    ``relation_type`` uses canonical :data:`cognition.spatial_schema.SpatialRelationType`
    strings (``left_of`` / ``right_of`` / ``in_front_of`` / ``behind`` /
    ``near`` / ``on`` / ``centered_in``) plus P5-only derived strings
    (``out_of_view`` / ``last_seen_near`` / ``occluded_by`` / ...).
    """
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    value_m: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relation_type": self.relation_type,
            "value_m": round(self.value_m, 3) if self.value_m is not None else None,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class MentalWorldSceneGraph:
    """Derived scene-graph snapshot for the HRR mental-world substrate."""
    timestamp: float
    entities: tuple[MentalWorldEntity, ...] = ()
    relations: tuple[MentalWorldRelation, ...] = ()
    source_scene_update_count: int = 0
    source_track_count: int = 0
    source_anchor_count: int = 0
    source_calibration_version: int = 0
    reason: Optional[str] = None  # set when the graph is empty-by-design

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def active_entity_count(self) -> int:
        return sum(1 for e in self.entities if e.state != "removed")

    @property
    def removed_entity_count(self) -> int:
        return sum(1 for e in self.entities if e.state == "removed")

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "entity_count": self.entity_count,
            "active_entity_count": self.active_entity_count,
            "removed_entity_count": self.removed_entity_count,
            "relation_count": self.relation_count,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "source": {
                "scene_update_count": self.source_scene_update_count,
                "track_count": self.source_track_count,
                "anchor_count": self.source_anchor_count,
                "calibration_version": self.source_calibration_version,
            },
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_near_threshold(entity: MentalWorldEntity) -> float:
    """Return the "near" radius (meters) used for this entity's class."""
    base = CLASS_MOVE_THRESHOLDS.get(entity.label, DEFAULT_MOVE_THRESHOLD)
    return max(NEAR_FLOOR_M, base * NEAR_MULTIPLIER)


def _axis_delta(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    axis: int,
) -> float:
    return float(a[axis]) - float(b[axis])


def _euclid(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _derive_pairwise_relations(
    entities: Sequence[MentalWorldEntity],
) -> list[MentalWorldRelation]:
    """Emit left/right, in_front_of/behind, near relations between positioned entities.

    Emits at most one relation per ordered pair (source=a, target=b) per
    axis so noise doesn't produce both ``a left_of b`` and ``b left_of a``.
    """
    rels: list[MentalWorldRelation] = []
    visible_entities = [e for e in entities if e.state == "visible"]
    for i, a in enumerate(visible_entities):
        if a.position_room_m is None:
            continue
        for b in visible_entities[i + 1 :]:
            if b.position_room_m is None:
                continue
            if a.is_display_surface or b.is_display_surface:
                # Screens are not physical world entities in the P5 mental view.
                continue
            dx = _axis_delta(a.position_room_m, b.position_room_m, 0)
            dz = _axis_delta(a.position_room_m, b.position_room_m, 2)

            # Lateral (x-axis) → left_of / right_of
            if abs(dx) >= LATERAL_SEPARATION_MIN_M:
                rel = "left_of" if dx < 0 else "right_of"
                rels.append(
                    MentalWorldRelation(
                        source_entity_id=a.entity_id,
                        target_entity_id=b.entity_id,
                        relation_type=rel,
                        value_m=abs(dx),
                        confidence=min(a.confidence, b.confidence),
                    )
                )

            # Depth (z-axis) → in_front_of / behind (smaller z = closer to camera)
            if abs(dz) >= DEPTH_SEPARATION_MIN_M:
                rel = "in_front_of" if dz < 0 else "behind"
                rels.append(
                    MentalWorldRelation(
                        source_entity_id=a.entity_id,
                        target_entity_id=b.entity_id,
                        relation_type=rel,
                        value_m=abs(dz),
                        confidence=min(a.confidence, b.confidence),
                    )
                )

            # Near (euclidean)
            dist = _euclid(a.position_room_m, b.position_room_m)
            threshold = max(_entity_near_threshold(a), _entity_near_threshold(b))
            if dist <= threshold:
                rels.append(
                    MentalWorldRelation(
                        source_entity_id=a.entity_id,
                        target_entity_id=b.entity_id,
                        relation_type="near",
                        value_m=dist,
                        confidence=min(a.confidence, b.confidence),
                    )
                )
    return rels


def _derive_state_relations(
    entities: Sequence[MentalWorldEntity],
) -> list[MentalWorldRelation]:
    """Emit derived relations based on :class:`SceneEntity` state.

    * ``out_of_view``: self-edge for entities whose state is ``missing`` or
      ``removed``. The relation is a property of the entity itself.
    * ``occluded_by``: emitted for entities whose state is ``occluded`` that
      sit near a visible entity in the same region.
    * ``last_seen_near``: emitted between occluded / missing entities and
      the nearest entity whose state is ``visible``, if any. Provides the
      mental world's guess at where the absent entity should be.
    """
    rels: list[MentalWorldRelation] = []
    visible_positioned = [
        e for e in entities if e.state == "visible" and e.position_room_m is not None
    ]

    for e in entities:
        if e.state in ("missing", "removed"):
            rels.append(
                MentalWorldRelation(
                    source_entity_id=e.entity_id,
                    target_entity_id=e.entity_id,
                    relation_type="out_of_view",
                    value_m=None,
                    confidence=e.confidence,
                )
            )

        if e.state in ("occluded", "missing") and e.position_room_m is not None:
            nearest = None
            nearest_dist = float("inf")
            for vp in visible_positioned:
                if vp.entity_id == e.entity_id:
                    continue
                d = _euclid(e.position_room_m, vp.position_room_m)  # type: ignore[arg-type]
                if d < nearest_dist:
                    nearest = vp
                    nearest_dist = d
            if nearest is not None and nearest_dist < float("inf"):
                rels.append(
                    MentalWorldRelation(
                        source_entity_id=e.entity_id,
                        target_entity_id=nearest.entity_id,
                        relation_type="last_seen_near",
                        value_m=nearest_dist,
                        confidence=min(e.confidence, nearest.confidence),
                    )
                )
                if e.state == "occluded" and e.region == nearest.region:
                    rels.append(
                        MentalWorldRelation(
                            source_entity_id=e.entity_id,
                            target_entity_id=nearest.entity_id,
                            relation_type="occluded_by",
                            value_m=nearest_dist,
                            confidence=min(e.confidence, nearest.confidence),
                        )
                    )

    return rels


def _derive_anchor_relations(
    entities: Sequence[MentalWorldEntity],
    anchors: Sequence[SpatialAnchor],
) -> list[MentalWorldRelation]:
    """Emit ``on`` and ``centered_in`` relations between entities and anchors.

    * ``on``: entity's y matches the anchor's top surface within
      :data:`ON_VERTICAL_TOLERANCE_M` and the entity sits inside the
      anchor's horizontal footprint.
    * ``centered_in``: entity's (x, z) is within
      :data:`CENTERED_IN_FRACTION` of the anchor half-extent of center.
    """
    rels: list[MentalWorldRelation] = []
    for e in entities:
        if e.state != "visible":
            continue
        if e.position_room_m is None or e.is_display_surface:
            continue
        ex, ey, ez = e.position_room_m
        for anc in anchors:
            ax, ay, az = anc.position_room_m
            adx, ady, adz = anc.dimensions_m

            # ``on`` test: y close to anchor top surface & inside x/z footprint.
            if ady > 0 and abs(ey - (ay + ady / 2.0)) <= ON_VERTICAL_TOLERANCE_M:
                if (
                    adx > 0 and abs(ex - ax) <= adx / 2.0
                    and adz > 0 and abs(ez - az) <= adz / 2.0
                ):
                    rels.append(
                        MentalWorldRelation(
                            source_entity_id=e.entity_id,
                            target_entity_id=anc.anchor_id,
                            relation_type="on",
                            value_m=abs(ey - (ay + ady / 2.0)),
                            confidence=min(e.confidence, anc.confidence),
                        )
                    )

            # ``centered_in`` test: inside anchor footprint, close to center.
            if adx > 0 and adz > 0:
                cx_frac = abs(ex - ax) / (adx / 2.0)
                cz_frac = abs(ez - az) / (adz / 2.0)
                if cx_frac <= CENTERED_IN_FRACTION and cz_frac <= CENTERED_IN_FRACTION:
                    rels.append(
                        MentalWorldRelation(
                            source_entity_id=e.entity_id,
                            target_entity_id=anc.anchor_id,
                            relation_type="centered_in",
                            value_m=max(cx_frac, cz_frac),
                            confidence=min(e.confidence, anc.confidence),
                        )
                    )
    return rels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def derive_scene_graph(
    scene_snapshot: Optional[SceneSnapshot],
    tracks: Optional[dict[str, SpatialTrack]] = None,
    anchors: Optional[dict[str, SpatialAnchor]] = None,
    *,
    calibration_version: int = 0,
    timestamp: Optional[float] = None,
) -> MentalWorldSceneGraph:
    """Build a :class:`MentalWorldSceneGraph` from canonical perception outputs.

    All inputs are optional; any missing scene snapshot or empty entity
    list produces an empty graph with ``reason=REASON_UNAVAILABLE``.
    Pure function: never mutates inputs, never writes anywhere.
    """
    ts = float(timestamp) if timestamp is not None else time.time()

    if scene_snapshot is None or not scene_snapshot.entities:
        return MentalWorldSceneGraph(
            timestamp=ts,
            entities=(),
            relations=(),
            source_calibration_version=int(calibration_version),
            reason=REASON_UNAVAILABLE,
        )

    track_map = dict(tracks or {})
    anchor_seq: tuple[SpatialAnchor, ...] = tuple((anchors or {}).values())

    entities: list[MentalWorldEntity] = []
    for se in scene_snapshot.entities:
        tr = track_map.get(se.entity_id)
        pos: Optional[tuple[float, float, float]] = None
        if tr is not None and tr.position_room_m:
            if any(abs(v) > 0 for v in tr.position_room_m):
                pos = (
                    float(tr.position_room_m[0]),
                    float(tr.position_room_m[1]),
                    float(tr.position_room_m[2]),
                )
        entities.append(
            MentalWorldEntity(
                entity_id=se.entity_id,
                label=se.label,
                state=se.state,
                region=se.region,
                position_room_m=pos,
                confidence=float(se.confidence),
                last_seen_ts=float(se.last_seen_ts),
                is_display_surface=bool(se.is_display_surface),
            )
        )

    relations: list[MentalWorldRelation] = []
    relations.extend(_derive_pairwise_relations(entities))
    relations.extend(_derive_state_relations(entities))
    relations.extend(_derive_anchor_relations(entities, anchor_seq))

    return MentalWorldSceneGraph(
        timestamp=ts,
        entities=tuple(entities),
        relations=tuple(relations),
        source_scene_update_count=int(scene_snapshot.update_count),
        source_track_count=len(track_map),
        source_anchor_count=len(anchor_seq),
        source_calibration_version=int(calibration_version),
        reason=None,
    )


__all__ = [
    "REASON_UNAVAILABLE",
    "LATERAL_SEPARATION_MIN_M",
    "DEPTH_SEPARATION_MIN_M",
    "NEAR_MULTIPLIER",
    "NEAR_FLOOR_M",
    "ON_VERTICAL_TOLERANCE_M",
    "CENTERED_IN_FRACTION",
    "MentalWorldEntity",
    "MentalWorldRelation",
    "MentalWorldSceneGraph",
    "derive_scene_graph",
]
