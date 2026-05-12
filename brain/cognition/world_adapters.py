"""Bridge layer: projects legacy subsystem state into canonical world-model types.

Keeps the old WorldState/CausalEngine pipeline alive while producing
canonical SensorObservation, WorldEntity, and WorldRelation objects in
parallel.  Future sensor types (GPS, LiDAR, sonar, OBD-II) add their
own ``observations_from_*`` functions here.
"""

from __future__ import annotations

import time
from typing import Any

from cognition.world_schema import (
    ArchetypeRegistry,
    CanonicalWorldState,
    SensorObservation,
    WorldEntity,
    WorldRelation,
    WorldZone,
)


# ---------------------------------------------------------------------------
# Observation adapters (sensor -> SensorObservation)
# ---------------------------------------------------------------------------

def observations_from_scene_state(
    scene_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[SensorObservation]:
    """Project SceneTracker entities and display surfaces into observations."""
    now = now or time.time()
    out: list[SensorObservation] = []

    for idx, ent in enumerate(scene_state.get("entities", [])):
        out.append(SensorObservation(
            observation_id=f"scene:entity:{idx}:{int(now)}",
            kind="vision",
            source="scene_tracker",
            ts=now,
            confidence=float(ent.get("confidence", 0.5)),
            freshness_s=0.0,
            subject_id=ent.get("entity_id") or ent.get("label"),
            region_id=ent.get("region"),
            payload=dict(ent),
            tags=("scene_entity",),
            provenance="live",
        ))

    for idx, surf in enumerate(scene_state.get("display_surfaces", [])):
        out.append(SensorObservation(
            observation_id=f"scene:display:{idx}:{int(now)}",
            kind="vision",
            source="scene_tracker",
            ts=now,
            confidence=float(surf.get("confidence", 0.5)),
            freshness_s=0.0,
            subject_id=surf.get("surface_id"),
            payload=dict(surf),
            tags=("display_surface",),
            provenance="live",
        ))

    return out


def observations_from_attention(
    attn_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[SensorObservation]:
    """Project AttentionCore state into a single observation."""
    now = now or time.time()
    if not attn_state:
        return []
    return [SensorObservation(
        observation_id=f"attention:{int(now)}",
        kind="attention",
        source="attention",
        ts=now,
        confidence=float(attn_state.get("presence_confidence", 0.0)),
        freshness_s=0.0,
        payload=dict(attn_state),
        tags=("attention_state",),
        provenance="live",
    )]


def observations_from_presence(
    pres_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[SensorObservation]:
    """Project PresenceTracker state into a single observation."""
    now = now or time.time()
    if not pres_state:
        return []
    return [SensorObservation(
        observation_id=f"presence:{int(now)}",
        kind="presence",
        source="presence",
        ts=now,
        confidence=float(pres_state.get("confidence", 0.0)),
        freshness_s=0.0,
        payload=dict(pres_state),
        tags=("presence_state",),
        provenance="live",
    )]


def observations_from_identity(
    speaker_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[SensorObservation]:
    """Project speaker/identity state into a single observation."""
    now = now or time.time()
    if not speaker_state:
        return []
    return [SensorObservation(
        observation_id=f"identity:{int(now)}",
        kind="identity",
        source="speaker_orchestrator",
        ts=now,
        confidence=float(speaker_state.get("confidence", 0.0)),
        freshness_s=0.0,
        subject_id=speaker_state.get("name"),
        payload=dict(speaker_state),
        tags=("identity_state",),
        provenance="live",
    )]


# ---------------------------------------------------------------------------
# Entity adapters (scene state -> WorldEntity)
# ---------------------------------------------------------------------------

_TOOL_LABELS = frozenset({
    "wrench", "hammer", "screwdriver", "pliers", "drill",
    "saw", "ratchet", "socket", "level", "tape_measure",
})


def entities_from_scene_state(
    scene_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[WorldEntity]:
    """Project SceneTracker entities into canonical WorldEntity objects."""
    now = now or time.time()
    out: list[WorldEntity] = []

    for idx, ent in enumerate(scene_state.get("entities", [])):
        label = str(ent.get("label", "unknown"))
        entity_id = ent.get("entity_id") or f"entity:{label}:{idx}"

        if ent.get("is_display_surface"):
            kind = "display"
        elif label == "person":
            kind = "person"
        elif label.lower() in _TOOL_LABELS:
            kind = "tool"
        else:
            kind = "object"

        out.append(WorldEntity(
            entity_id=entity_id,
            kind=kind,  # type: ignore[arg-type]
            label=label,
            ts=now,
            confidence=float(ent.get("confidence", 0.5)),
            first_seen_ts=ent.get("first_seen_ts"),
            last_seen_ts=ent.get("last_seen_ts", now),
            stable=bool(ent.get("stable_cycles", 0) >= 3),
            mobile=label == "person",
            properties=dict(ent),
            region_id=ent.get("region"),
            source_observation_ids=(f"scene:entity:{idx}:{int(now)}",),
            tags=("scene_projected",),
        ))

    for idx, surf in enumerate(scene_state.get("display_surfaces", [])):
        surface_id = surf.get("surface_id") or f"display:{idx}"
        out.append(WorldEntity(
            entity_id=surface_id,
            kind="display",
            label=surf.get("label", "display"),
            ts=now,
            confidence=float(surf.get("confidence", 0.5)),
            first_seen_ts=surf.get("first_seen_ts"),
            last_seen_ts=surf.get("last_seen_ts", now),
            stable=True,
            mobile=False,
            properties=dict(surf),
            region_id=surf.get("region"),
            source_observation_ids=(f"scene:display:{idx}:{int(now)}",),
            tags=("display_surface", "scene_projected"),
        ))

    return out


# ---------------------------------------------------------------------------
# Relation adapters (scene state -> WorldRelation)
# ---------------------------------------------------------------------------

def relations_from_scene_state(
    scene_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[WorldRelation]:
    """Derive region-based relations from scene entities."""
    now = now or time.time()
    out: list[WorldRelation] = []

    for idx, ent in enumerate(scene_state.get("entities", [])):
        label = str(ent.get("label", "unknown"))
        region = ent.get("region")
        if not region:
            continue
        entity_id = ent.get("entity_id") or f"entity:{label}:{idx}"
        out.append(WorldRelation(
            relation_id=f"rel:located_at:{entity_id}:{int(now)}",
            kind="located_at",
            subject_id=entity_id,
            object_id=f"region:{region}",
            ts=now,
            confidence=float(ent.get("confidence", 0.5)),
            source_observation_ids=(f"scene:entity:{idx}:{int(now)}",),
            tags=("region_projection",),
        ))

    return out


# ---------------------------------------------------------------------------
# Zone adapters (scene region_visibility -> WorldZone)
# ---------------------------------------------------------------------------

_REGION_TO_ZONE_KIND: dict[str, str] = {
    "desk_left": "work_area",
    "desk_center": "work_area",
    "desk_right": "work_area",
    "monitor_zone": "work_area",
    "floor_zone": "walkway",
    "shelf_zone": "storage_area",
    "door_zone": "walkway",
}


def zones_from_scene_state(
    scene_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[WorldZone]:
    """Project scene region_visibility into canonical WorldZone objects."""
    now = now or time.time()
    out: list[WorldZone] = []

    region_vis = scene_state.get("region_visibility", {})
    for region_name, visibility in region_vis.items():
        zone_kind = _REGION_TO_ZONE_KIND.get(region_name, "unknown")
        out.append(WorldZone(
            zone_id=f"zone:{region_name}",
            label=region_name.replace("_", " "),
            kind=zone_kind,  # type: ignore[arg-type]
            region_label=region_name,
            confidence=float(visibility) if isinstance(visibility, (int, float)) else 1.0,
            tags=("scene_projected",),
        ))

    return out


# ---------------------------------------------------------------------------
# Spatial observation adapter
# ---------------------------------------------------------------------------

def observations_from_spatial_state(
    spatial_state: dict[str, Any],
    *,
    now: float | None = None,
) -> list[SensorObservation]:
    """Project spatial tracks and anchors into canonical observations."""
    now = now or time.time()
    out: list[SensorObservation] = []

    for tid, track in spatial_state.get("spatial_tracks", {}).items():
        out.append(SensorObservation(
            observation_id=f"spatial:track:{tid}:{int(now)}",
            kind="spatial",
            source="spatial_estimator",
            ts=now,
            confidence=float(track.get("confidence", 0.0)),
            freshness_s=0.0,
            subject_id=tid,
            payload=dict(track),
            tags=("spatial_track", track.get("track_status", "provisional")),
            provenance="prior_based",
        ))

    for aid, anchor in spatial_state.get("spatial_anchors", {}).items():
        out.append(SensorObservation(
            observation_id=f"spatial:anchor:{aid}:{int(now)}",
            kind="spatial",
            source="spatial_estimator",
            ts=now,
            confidence=float(anchor.get("confidence", 0.0)),
            freshness_s=0.0,
            subject_id=aid,
            payload=dict(anchor),
            tags=("spatial_anchor",),
            provenance="calibrated",
        ))

    return out


# ---------------------------------------------------------------------------
# Canonical world projector
# ---------------------------------------------------------------------------

class CanonicalWorldProjector:
    """Builds a full CanonicalWorldState each tick from sensor subsystems.

    Runs in parallel with the legacy WorldState pipeline — does not replace
    or interfere with existing dashboard/benchmark consumers.
    """

    def __init__(self, registry: ArchetypeRegistry | None = None) -> None:
        if registry is None:
            from cognition.world_archetypes import default_registry
            registry = default_registry
        self._registry = registry

    def build(
        self,
        *,
        now: float,
        scene: dict[str, Any],
        attention: dict[str, Any],
        presence: dict[str, Any],
        identity: dict[str, Any],
        spatial: dict[str, Any] | None = None,
    ) -> CanonicalWorldState:
        """Build a canonical snapshot from current sensor state."""
        obs = [
            *observations_from_scene_state(scene, now=now),
            *observations_from_attention(attention, now=now),
            *observations_from_presence(presence, now=now),
            *observations_from_identity(identity, now=now),
        ]
        if spatial:
            obs.extend(observations_from_spatial_state(spatial, now=now))
        entities = entities_from_scene_state(scene, now=now)
        relations = relations_from_scene_state(scene, now=now)
        zones = zones_from_scene_state(scene, now=now)

        entity_labels = {e.label.lower() for e in entities}
        matched = self._registry.match(entity_labels)

        return CanonicalWorldState(
            timestamp=now,
            archetypes_active=tuple(matched),
            observations=tuple(obs),
            entities=tuple(entities),
            relations=tuple(relations),
            zones=tuple(zones),
        )
