"""Layer 3B Scene Tracker — the core permanence engine.

Converts repeated Pi/VLM detections into durable scene entities with
region-aware state transitions:

    candidate -> visible -> occluded -> missing -> removed

Follows the same fusion pattern as IdentityFusion: raw signals come in,
a stateful tracker maintains continuity, stale inputs decay, and
downstream systems read stable state (not raw detections).

This module does NOT write memory or trigger curiosity — it only
maintains the world model. Memory gating is deferred to Patches 3+4.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from perception.scene_types import (
    DISPLAY_SURFACE_LABELS,
    BBox,
    DisplaySurface,
    SceneDetection,
    SceneDelta,
    SceneEntity,
    SceneSnapshot,
)
from perception.scene_regions import estimate_region_visibility, infer_region

logger = logging.getLogger(__name__)

PROMOTION_STABLE_CYCLES = 3
REMOVAL_MISSING_CYCLES = 5
MATCH_THRESHOLD = 0.55
MAX_ENTITIES = 50
REMOVED_RETENTION_CYCLES = 3

PERMANENCE_BOOST_ON_SEEN = 0.15
PERMANENCE_DECAY_OCCLUDED = 0.03
PERMANENCE_DECAY_UNCERTAIN = 0.08
PERMANENCE_DECAY_VISIBLE_EMPTY = 0.18
PERMANENCE_FLOOR_OCCLUDED = 0.35
PERMANENCE_FLOOR_UNCERTAIN = 0.10
DISPLAY_SURFACE_STALE_CYCLES = 8


class SceneTracker:
    """Maintains a persistent world model of physical scene entities."""

    def __init__(self) -> None:
        self._entities: dict[str, SceneEntity] = {}
        self._display_surfaces: dict[str, DisplaySurface] = {}
        self._display_unseen: dict[str, int] = {}
        self._last_snapshot: SceneSnapshot | None = None
        self._update_count: int = 0

    def update(
        self,
        detections: list[SceneDetection],
        frame_w: int,
        frame_h: int,
        person_bboxes: list[BBox] | None = None,
    ) -> SceneSnapshot:
        """Process a batch of detections and return updated scene snapshot."""
        now = time.time()
        self._update_count += 1
        deltas: list[SceneDelta] = []
        matched_ids: set[str] = set()

        region_vis = estimate_region_visibility(
            person_bboxes or [], frame_w, frame_h,
        )

        physical, display_interior = self._partition_detections(detections)

        for det in physical:
            region = infer_region(det.bbox, frame_w, frame_h)
            best = self._find_best_match(det, region, now)

            if best is not None:
                matched_ids.add(best.entity_id)
                delta = self._update_matched(best, det, region, now)
                if delta:
                    deltas.append(delta)
            else:
                ent = self._create_candidate(det, region, now)
                self._entities[ent.entity_id] = ent
                matched_ids.add(ent.entity_id)

        matched_surface_ids = self._update_display_surfaces(physical, now)
        self._decay_display_surfaces(matched_surface_ids)

        for ent in list(self._entities.values()):
            if ent.entity_id in matched_ids:
                continue
            delta = self._decay_unmatched(ent, region_vis, now)
            if delta:
                deltas.append(delta)

        self._evict_removed()

        snapshot = SceneSnapshot(
            timestamp=now,
            entities=list(self._entities.values()),
            deltas=deltas,
            display_surfaces=list(self._display_surfaces.values()),
            region_visibility=region_vis,
            update_count=self._update_count,
        )
        self._last_snapshot = snapshot
        return snapshot

    def get_state(self) -> dict[str, Any]:
        """Return dashboard-friendly state dict."""
        if self._last_snapshot is None:
            return {
                "entity_count": 0,
                "visible_count": 0,
                "stable_count": 0,
                "display_surfaces": [],
                "display_content": [],
                "entities": [],
                "recent_deltas": [],
                "region_visibility": {},
                "update_count": 0,
            }
        snap = self._last_snapshot
        d = snap.to_dict()
        d["recent_deltas"] = d.pop("deltas", [])[-10:]
        return d

    # --- Detection partitioning ---

    def _partition_detections(
        self, detections: list[SceneDetection],
    ) -> tuple[list[SceneDetection], list[SceneDetection]]:
        """Split detections into physical entities vs display-interior objects.

        Display surfaces themselves (monitor, tv, laptop) are physical.
        Objects whose bbox center falls INSIDE a known display surface bbox
        are classified as display-interior and excluded from physical tracking.
        """
        physical: list[SceneDetection] = []
        display_interior: list[SceneDetection] = []

        display_bboxes = [
            ds.bbox for ds in self._display_surfaces.values()
            if ds.bbox is not None
        ]

        for det in detections:
            if det.label in DISPLAY_SURFACE_LABELS:
                physical.append(det)
                continue

            if det.bbox and self._inside_any_display(det.bbox, display_bboxes):
                display_interior.append(det)
            else:
                physical.append(det)

        return physical, display_interior

    @staticmethod
    def _inside_any_display(bbox: BBox, surfaces: list[BBox]) -> bool:
        """Check if the center of bbox falls inside any display surface bbox."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for sx1, sy1, sx2, sy2 in surfaces:
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                return True
        return False

    # --- Entity matching ---

    def _find_best_match(
        self, det: SceneDetection, region: str, now: float,
    ) -> SceneEntity | None:
        best_score = 0.0
        best_ent: SceneEntity | None = None

        for ent in self._entities.values():
            if ent.state == "removed":
                continue
            score = self._match_score(ent, det, region, now)
            if score > best_score:
                best_score = score
                best_ent = ent

        if best_score >= MATCH_THRESHOLD:
            return best_ent
        return None

    @staticmethod
    def _match_score(
        entity: SceneEntity,
        det: SceneDetection,
        region: str,
        now: float,
    ) -> float:
        score = 0.0

        if entity.label == det.label:
            score += 0.45
        elif entity.label in DISPLAY_SURFACE_LABELS and det.label in DISPLAY_SURFACE_LABELS:
            score += 0.35

        if entity.region == region:
            score += 0.25

        if entity.bbox and det.bbox:
            ecx = (entity.bbox[0] + entity.bbox[2]) / 2
            ecy = (entity.bbox[1] + entity.bbox[3]) / 2
            dcx = (det.bbox[0] + det.bbox[2]) / 2
            dcy = (det.bbox[1] + det.bbox[3]) / 2
            dist = abs(ecx - dcx) + abs(ecy - dcy)
            score += max(0.0, 0.20 - min(dist / 1200.0, 0.20))

        age = now - entity.last_seen_ts
        score += max(0.0, 0.10 - min(age / 60.0, 0.10))

        return score

    # --- Entity lifecycle ---

    def _update_matched(
        self,
        ent: SceneEntity,
        det: SceneDetection,
        region: str,
        now: float,
    ) -> SceneDelta | None:
        prev_state = ent.state
        prev_region = ent.region

        ent.last_seen_ts = now
        ent.bbox = det.bbox
        ent.region = region
        ent.confidence = det.confidence
        ent.unseen_cycles = 0
        ent.stable_cycles += 1
        ent.permanence_confidence = min(
            1.0, ent.permanence_confidence + PERMANENCE_BOOST_ON_SEEN,
        )

        if prev_state == "candidate" and ent.stable_cycles >= PROMOTION_STABLE_CYCLES:
            ent.state = "visible"
            return SceneDelta(
                "entity_promoted", ent.entity_id, ent.label,
                ent.region, ent.confidence,
            )

        if prev_state == "candidate":
            pass
        else:
            ent.state = "visible"

        if prev_region != region and prev_state not in ("candidate",):
            return SceneDelta(
                "entity_moved", ent.entity_id, ent.label,
                ent.region, ent.confidence,
                {"from_region": prev_region},
            )

        return None

    @staticmethod
    def _create_candidate(
        det: SceneDetection, region: str, now: float,
    ) -> SceneEntity:
        return SceneEntity(
            entity_id=f"obj_{uuid.uuid4().hex[:10]}",
            label=det.label,
            confidence=det.confidence,
            permanence_confidence=0.4,
            bbox=det.bbox,
            region=region,
            state="candidate",
            first_seen_ts=now,
            last_seen_ts=now,
            stable_cycles=1,
            source=det.source,
            is_display_surface=det.label in DISPLAY_SURFACE_LABELS,
        )

    def _decay_unmatched(
        self,
        ent: SceneEntity,
        region_vis: dict[str, float],
        now: float,
    ) -> SceneDelta | None:
        ent.unseen_cycles += 1
        vis = region_vis.get(ent.region, 0.5)
        prev_state = ent.state

        if vis < 0.3:
            ent.permanence_confidence = max(
                PERMANENCE_FLOOR_OCCLUDED,
                ent.permanence_confidence - PERMANENCE_DECAY_OCCLUDED,
            )
            if ent.state in ("visible", "candidate"):
                ent.state = "occluded"
        elif vis >= 0.7:
            ent.permanence_confidence = max(
                0.0,
                ent.permanence_confidence - PERMANENCE_DECAY_VISIBLE_EMPTY,
            )
            if ent.state in ("visible", "occluded", "candidate"):
                ent.state = "missing"
        else:
            ent.permanence_confidence = max(
                PERMANENCE_FLOOR_UNCERTAIN,
                ent.permanence_confidence - PERMANENCE_DECAY_UNCERTAIN,
            )
            if ent.state == "visible":
                ent.state = "occluded"

        if ent.unseen_cycles >= REMOVAL_MISSING_CYCLES and vis >= 0.7:
            ent.state = "removed"
            if prev_state != "removed":
                return SceneDelta(
                    "entity_removed", ent.entity_id, ent.label,
                    ent.region, ent.permanence_confidence,
                )

        if ent.state != prev_state:
            if ent.state == "occluded":
                return SceneDelta(
                    "entity_occluded", ent.entity_id, ent.label,
                    ent.region, ent.permanence_confidence,
                )
            if ent.state == "missing":
                return SceneDelta(
                    "entity_missing", ent.entity_id, ent.label,
                    ent.region, ent.permanence_confidence,
                )

        return None

    # --- Display surface tracking ---

    def _update_display_surfaces(
        self, physical_dets: list[SceneDetection], now: float,
    ) -> set[str]:
        """Track display surfaces as physical objects with special semantics.

        Returns set of surface IDs that were matched this cycle.
        """
        matched: set[str] = set()
        for det in physical_dets:
            if det.label not in DISPLAY_SURFACE_LABELS:
                continue
            best_id = self._match_display_surface(det)
            if best_id:
                ds = self._display_surfaces[best_id]
                ds.bbox = det.bbox
                ds.confidence = det.confidence
                ds.last_seen_ts = now
                ds.stable_for_s = now - ds.first_seen_ts
                ds.kind = det.label
                matched.add(best_id)
            else:
                sid = f"disp_{uuid.uuid4().hex[:8]}"
                self._display_surfaces[sid] = DisplaySurface(
                    surface_id=sid,
                    kind=det.label,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    first_seen_ts=now,
                    last_seen_ts=now,
                )
                matched.add(sid)
        return matched

    def _match_display_surface(self, det: SceneDetection) -> str | None:
        """Match a display detection to an existing surface.

        Cross-label matching: any DISPLAY_SURFACE_LABELS member can match
        any other (YOLO often oscillates between 'tv' and 'monitor' for the
        same physical screen).
        """
        for sid, ds in self._display_surfaces.items():
            labels_compatible = (
                ds.kind == det.label
                or (ds.kind in DISPLAY_SURFACE_LABELS
                    and det.label in DISPLAY_SURFACE_LABELS)
            )
            if labels_compatible and ds.bbox and det.bbox:
                if self._inside_any_display(det.bbox, [ds.bbox]):
                    return sid
                ecx = (ds.bbox[0] + ds.bbox[2]) / 2
                dcx = (det.bbox[0] + det.bbox[2]) / 2
                ecy = (ds.bbox[1] + ds.bbox[3]) / 2
                dcy = (det.bbox[1] + det.bbox[3]) / 2
                dist = abs(ecx - dcx) + abs(ecy - dcy)
                if dist < 150:
                    return sid
        return None

    def _decay_display_surfaces(self, matched_ids: set[str]) -> None:
        """Evict display surfaces not seen for DISPLAY_SURFACE_STALE_CYCLES."""
        for sid in list(self._display_unseen.keys()):
            if sid not in self._display_surfaces:
                del self._display_unseen[sid]

        for sid in self._display_surfaces:
            if sid in matched_ids:
                self._display_unseen[sid] = 0
            else:
                self._display_unseen[sid] = self._display_unseen.get(sid, 0) + 1

        stale = [
            sid for sid, cycles in self._display_unseen.items()
            if cycles >= DISPLAY_SURFACE_STALE_CYCLES
        ]
        for sid in stale:
            del self._display_surfaces[sid]
            self._display_unseen.pop(sid, None)

    # --- Eviction ---

    def _evict_removed(self) -> None:
        """Evict removed entities after a short history window, then cap size.

        ``removed`` means canonical perception has already decided the object is
        gone. Keeping it briefly preserves the removal delta for operator/debug
        views; keeping it indefinitely makes downstream scene renders look like
        old objects are still in the room.
        """
        stale_removed = [
            eid for eid, ent in self._entities.items()
            if (
                ent.state == "removed"
                and ent.unseen_cycles
                >= REMOVAL_MISSING_CYCLES + REMOVED_RETENTION_CYCLES
            )
        ]
        for eid in stale_removed:
            self._entities.pop(eid, None)

        if len(self._entities) <= MAX_ENTITIES:
            return
        removed = [
            (eid, ent) for eid, ent in self._entities.items()
            if ent.state == "removed"
        ]
        removed.sort(key=lambda x: x[1].last_seen_ts)
        while len(self._entities) > MAX_ENTITIES and removed:
            eid, _ = removed.pop(0)
            del self._entities[eid]
