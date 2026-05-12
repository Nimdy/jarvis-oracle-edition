"""Spatial fusion — merges spatial tracks with scene continuity entities.

Bridges the spatial estimation layer and the existing scene tracker by
matching spatial tracks to scene entities and producing a unified view
consumed by the canonical world projector.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from cognition.spatial_schema import (
    CONFIDENCE_THRESHOLD_STABLE,
    SpatialAnchor,
    SpatialRelationFact,
    SpatialTrack,
)

logger = logging.getLogger(__name__)


class SpatialFusion:
    """Merges spatial tracks with scene entities for world-model consumption."""

    def __init__(self) -> None:
        self._relations: list[SpatialRelationFact] = []
        self._last_fused_ts: float = 0.0

    def fuse(
        self,
        scene_state: dict[str, Any],
        tracks: dict[str, SpatialTrack],
        anchors: dict[str, SpatialAnchor],
    ) -> dict[str, Any]:
        """Produce a unified view augmenting scene state with spatial data.

        Returns a new dict (does NOT modify the input scene_state).
        """
        now = time.time()
        self._last_fused_ts = now

        augmented: list[dict[str, Any]] = []
        for ent in scene_state.get("entities", []):
            eid = ent.get("entity_id", "")
            track = tracks.get(eid)
            aug = dict(ent)
            if track and track.confidence >= CONFIDENCE_THRESHOLD_STABLE:
                aug["spatial"] = track.to_dict()
            augmented.append(aug)

        self._relations = self._derive_relations(tracks, anchors, now)

        return {
            "entities": augmented,
            "display_surfaces": scene_state.get("display_surfaces", []),
            "display_content": scene_state.get("display_content", []),
            "region_visibility": scene_state.get("region_visibility", {}),
            "spatial_tracks": {eid: t.to_dict() for eid, t in tracks.items()},
            "spatial_anchors": {aid: a.to_dict() for aid, a in anchors.items()},
            "spatial_relations": [r.to_dict() for r in self._relations],
            "fused_at": now,
        }

    def get_relations(self) -> list[SpatialRelationFact]:
        return list(self._relations)

    def _derive_relations(
        self,
        tracks: dict[str, SpatialTrack],
        anchors: dict[str, SpatialAnchor],
        now: float,
    ) -> list[SpatialRelationFact]:
        stable_tracks = {
            eid: t for eid, t in tracks.items()
            if t.track_status == "stable"
            and t.confidence >= CONFIDENCE_THRESHOLD_STABLE
        }
        if not stable_tracks or not anchors:
            return []

        relations: list[SpatialRelationFact] = []
        rel_idx = 0
        for eid, track in stable_tracks.items():
            for aid, anchor in anchors.items():
                rel_type = self._classify_relation(track, anchor)
                if rel_type:
                    dx = track.position_room_m[0] - anchor.position_room_m[0]
                    dy = track.position_room_m[1] - anchor.position_room_m[1]
                    dz = track.position_room_m[2] - anchor.position_room_m[2]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    relations.append(SpatialRelationFact(
                        relation_id=f"srel:{rel_idx}:{int(now)}",
                        source_entity_id=eid,
                        relation_type=rel_type,  # type: ignore[arg-type]
                        target_entity_id=aid,
                        value_m=dist,
                        confidence=min(track.confidence, anchor.confidence),
                        calibration_version=anchor.calibration_version,
                        timestamp=now,
                    ))
                    rel_idx += 1
        return relations

    @staticmethod
    def _classify_relation(
        track: SpatialTrack,
        anchor: SpatialAnchor,
    ) -> str | None:
        dx = track.position_room_m[0] - anchor.position_room_m[0]
        dz = track.position_room_m[2] - anchor.position_room_m[2]
        horiz_dist = math.sqrt(dx * dx + dz * dz)

        if horiz_dist < 0.3:
            return "near"
        if abs(dx) > abs(dz):
            return "left_of" if dx < 0 else "right_of"
        return "in_front_of" if dz < 0 else "behind"
