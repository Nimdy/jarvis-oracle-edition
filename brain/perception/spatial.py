"""Spatial estimator — detection-to-distance, smoothing, uncertainty.

Converts scene detections into spatial observations using known-size priors
and camera calibration.  All estimates are advisory-first; they never
directly write to memory or create beliefs.

Includes a replay harness for offline regression testing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CLASS_MOVE_THRESHOLDS,
    CONFIDENCE_THRESHOLD_TRACK,
    DEFAULT_MOVE_THRESHOLD,
    KNOWN_SIZE_PRIORS,
    SMOOTHING_ALPHA,
    STABLE_WINDOWS_REQUIRED,
    SpatialAnchor,
    SpatialObservation,
    SpatialTrack,
)
from perception.calibration import CalibrationManager
from perception.scene_types import DISPLAY_SURFACE_LABELS

logger = logging.getLogger(__name__)


class SpatialEstimator:
    """Converts scene detections into spatial observations and maintains tracks."""

    def __init__(self, calibration: CalibrationManager) -> None:
        self._cal = calibration
        self._tracks: dict[str, SpatialTrack] = {}
        self._anchors: dict[str, SpatialAnchor] = {}
        self._observation_count: int = 0

    def estimate(
        self,
        entity_id: str,
        label: str,
        bbox: tuple[int, int, int, int] | None,
        confidence: float,
        timestamp: float | None = None,
    ) -> SpatialObservation | None:
        """Estimate spatial position from a detection using known-size priors.

        Returns None if calibration is invalid or no size prior exists.
        """
        if not bbox:
            return None

        cal_state = self._cal.state
        if cal_state == "invalid":
            return None

        ts = timestamp or time.time()
        intrinsics = self._cal.intrinsics

        base_label = self._normalize_label(label)
        real_size = KNOWN_SIZE_PRIORS.get(base_label)
        if real_size is None:
            return None

        x1, y1, x2, y2 = bbox
        pixel_width = max(1, x2 - x1)
        pixel_height = max(1, y2 - y1)
        pixel_size = max(pixel_width, pixel_height)

        depth_m = (real_size * intrinsics.focal_length_px) / pixel_size
        if depth_m <= 0.01 or depth_m > 20.0:
            return None

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cam_x = (cx - intrinsics.principal_x) * depth_m / intrinsics.focal_length_px
        cam_y = (cy - intrinsics.principal_y) * depth_m / intrinsics.focal_length_px
        cam_z = depth_m
        cam_pos = (cam_x, cam_y, cam_z)

        room_pos = None
        if cal_state == "valid":
            room_pos = self._cal.camera_to_room(cam_pos)

        det_conf = min(confidence, 1.0)
        prior_conf = 0.8 if base_label in KNOWN_SIZE_PRIORS else 0.3
        cal_health = 1.0 if cal_state == "valid" else 0.5
        temporal_stab = self._temporal_stability(entity_id)
        region_cons = 0.7

        pos_confidence = (
            0.35 * det_conf
            + 0.20 * prior_conf
            + 0.20 * cal_health
            + 0.15 * temporal_stab
            + 0.10 * region_cons
        )

        uncertainty = depth_m * 0.15 * (1.0 - pos_confidence * 0.5)

        self._observation_count += 1
        return SpatialObservation(
            entity_id=entity_id,
            label=label,
            depth_m=depth_m,
            position_camera_m=cam_pos,
            position_room_m=room_pos,
            size_estimate_m=real_size,
            confidence=pos_confidence,
            uncertainty_m=uncertainty,
            calibration_version=self._cal.version,
            provenance="prior_based",
            timestamp=ts,
            bbox=bbox,
        )

    def update_track(self, obs: SpatialObservation) -> SpatialTrack:
        """Update or create a spatial track from an observation."""
        existing = self._tracks.get(obs.entity_id)

        if existing is None:
            track = SpatialTrack(
                entity_id=obs.entity_id,
                label=obs.label,
                track_status="provisional",
                position_room_m=obs.position_room_m or obs.position_camera_m,
                uncertainty_m=obs.uncertainty_m,
                confidence=obs.confidence,
                samples=1,
                stable_windows=0,
                first_seen_ts=obs.timestamp,
                last_update_ts=obs.timestamp,
                authority=AUTHORITY_LEVELS["provisional_track"],
            )
            self._tracks[obs.entity_id] = track
            return track

        new_pos = obs.position_room_m or obs.position_camera_m
        old_pos = existing.position_room_m

        move_threshold = CLASS_MOVE_THRESHOLDS.get(
            self._normalize_label(obs.label), DEFAULT_MOVE_THRESHOLD,
        )
        displacement = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(new_pos, old_pos)
        ))

        alpha = SMOOTHING_ALPHA * 0.5 if displacement < move_threshold else SMOOTHING_ALPHA
        smoothed = tuple(
            old * (1.0 - alpha) + new * alpha
            for old, new in zip(old_pos, new_pos)
        )

        new_samples = existing.samples + 1
        new_stable = existing.stable_windows
        new_status = existing.track_status

        if obs.confidence >= CONFIDENCE_THRESHOLD_TRACK:
            new_stable += 1
        else:
            new_stable = max(0, new_stable - 1)

        authority = existing.authority
        if new_stable >= STABLE_WINDOWS_REQUIRED and new_status == "provisional":
            new_status = "stable"
            authority = AUTHORITY_LEVELS["stable_track"]

        dt = max(0.01, obs.timestamp - existing.last_update_ts)
        velocity = tuple((s - o) / dt for s, o in zip(smoothed, old_pos))

        new_uncertainty = (
            existing.uncertainty_m * (1.0 - SMOOTHING_ALPHA)
            + obs.uncertainty_m * SMOOTHING_ALPHA
        )

        track = SpatialTrack(
            entity_id=obs.entity_id,
            label=obs.label,
            track_status=new_status,
            position_room_m=smoothed,
            velocity_mps=velocity,
            dimensions_m=existing.dimensions_m,
            uncertainty_m=new_uncertainty,
            confidence=obs.confidence * 0.3 + existing.confidence * 0.7,
            samples=new_samples,
            stable_windows=new_stable,
            first_seen_ts=existing.first_seen_ts,
            last_update_ts=obs.timestamp,
            anchor_id=existing.anchor_id,
            authority=authority,
        )
        self._tracks[obs.entity_id] = track
        return track

    def register_anchor(
        self,
        anchor_id: str,
        anchor_type: str,
        label: str,
        position_room_m: tuple[float, float, float],
        dimensions_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        confidence: float = 0.7,
    ) -> SpatialAnchor:
        """Register or update a stable spatial anchor."""
        now = time.time()
        existing = self._anchors.get(anchor_id)
        anchor = SpatialAnchor(
            anchor_id=anchor_id,
            anchor_type=anchor_type,  # type: ignore[arg-type]
            label=label,
            position_room_m=position_room_m,
            dimensions_m=dimensions_m,
            confidence=confidence,
            stable_since_ts=existing.stable_since_ts if existing else now,
            last_verified_ts=now,
            calibration_version=self._cal.version,
            authority=AUTHORITY_LEVELS["stable_anchor"],
        )
        self._anchors[anchor_id] = anchor
        return anchor

    def check_anchor_conflict(self, obs: SpatialObservation) -> bool:
        """Check if an observation conflicts with a stable anchor.

        Anchor authority rule: if a movable object estimate implies a
        stable anchor moved, the observation is wrong.
        Returns True if there is a conflict (observation should be rejected).
        """
        if obs.position_room_m is None:
            return False
        base_label = self._normalize_label(obs.label)
        for anchor in self._anchors.values():
            if anchor.authority <= AUTHORITY_LEVELS["provisional_track"]:
                continue
            labels_match = (
                anchor.label == base_label
                or (anchor.label in DISPLAY_SURFACE_LABELS
                    and base_label in DISPLAY_SURFACE_LABELS)
            )
            if not labels_match:
                continue
            dist = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(obs.position_room_m, anchor.position_room_m)
            ))
            threshold = CLASS_MOVE_THRESHOLDS.get(
                base_label, DEFAULT_MOVE_THRESHOLD,
            )
            if dist > threshold:
                return True
        return False

    def decay_stale_tracks(self, now: float | None = None) -> list[str]:
        """Mark tracks as stale if not updated recently."""
        now = now or time.time()
        stale_ids: list[str] = []
        for eid, track in list(self._tracks.items()):
            if track.track_status in ("provisional", "stable"):
                if now - track.last_update_ts > 30.0:
                    self._tracks[eid] = SpatialTrack(
                        entity_id=track.entity_id,
                        label=track.label,
                        track_status="stale",
                        position_room_m=track.position_room_m,
                        velocity_mps=(0.0, 0.0, 0.0),
                        dimensions_m=track.dimensions_m,
                        uncertainty_m=min(track.uncertainty_m * 1.5, 5.0),
                        confidence=track.confidence * 0.8,
                        samples=track.samples,
                        stable_windows=track.stable_windows,
                        first_seen_ts=track.first_seen_ts,
                        last_update_ts=track.last_update_ts,
                        anchor_id=track.anchor_id,
                        authority=AUTHORITY_LEVELS["provisional_track"],
                    )
                    stale_ids.append(eid)
        return stale_ids

    def reset_for_relocalization(self, *, profile_id: str = "", reason: str = "") -> None:
        """Reset tracker state after calibration/profile handoff.

        Scene-profile switches may change coordinate assumptions enough that
        old track continuity would produce false deltas; clear local track memory.
        """
        dropped_tracks = len(self._tracks)
        dropped_anchors = len(self._anchors)
        self._tracks.clear()
        self._anchors.clear()
        logger.info(
            "Spatial estimator relocalization reset: profile=%s tracks=%d anchors=%d reason=%s",
            profile_id or "unknown",
            dropped_tracks,
            dropped_anchors,
            reason or "unspecified",
        )

    def get_tracks(self) -> dict[str, SpatialTrack]:
        return dict(self._tracks)

    def get_anchors(self) -> dict[str, SpatialAnchor]:
        return dict(self._anchors)

    def get_state(self) -> dict[str, Any]:
        tracks = list(self._tracks.values())
        anchors = list(self._anchors.values())
        return {
            "total_tracks": len(tracks),
            "stable_tracks": sum(1 for t in tracks if t.track_status == "stable"),
            "provisional_tracks": sum(1 for t in tracks if t.track_status == "provisional"),
            "stale_tracks": sum(1 for t in tracks if t.track_status == "stale"),
            "total_anchors": len(anchors),
            "observations_total": self._observation_count,
            "tracks": [t.to_dict() for t in tracks],
            "anchors": [a.to_dict() for a in anchors],
            "calibration": self._cal.get_state(),
        }

    def _temporal_stability(self, entity_id: str) -> float:
        track = self._tracks.get(entity_id)
        if track is None:
            return 0.0
        return min(1.0, track.stable_windows / max(STABLE_WINDOWS_REQUIRED, 1))

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.lower().strip()
        if label in ("television",):
            return "tv"
        if label in ("screen",):
            return "monitor"
        return label


# ---------------------------------------------------------------------------
# Replay harness for offline regression testing
# ---------------------------------------------------------------------------

_REPLAY_DIR = Path.home() / ".jarvis" / "spatial" / "recordings"


class SpatialRecorder:
    """Records scene_summary dicts to JSONL for replay testing."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._active = False
        self._start_ts: float = 0.0

    def start(self) -> None:
        self._active = True
        self._start_ts = time.time()
        self._records = []

    def record(self, scene_summary: dict[str, Any]) -> None:
        if not self._active:
            return
        self._records.append({
            "t": time.time() - self._start_ts,
            "scene_summary": scene_summary,
        })

    def stop(self) -> None:
        self._active = False

    def save(self, filename: str) -> str:
        _REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = _REPLAY_DIR / filename
        with open(path, "w") as f:
            for rec in self._records:
                f.write(json.dumps(rec) + "\n")
        return str(path)

    @property
    def count(self) -> int:
        return len(self._records)


class SpatialReplayer:
    """Replays recorded scene summaries through a SpatialEstimator."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._records: list[dict[str, Any]] = []
        self._load()

    def replay(
        self, estimator: SpatialEstimator,
    ) -> list[dict[str, Any]]:
        """Replay all records through the estimator instantly.

        Returns a list of per-record results (observations, tracks).
        """
        results: list[dict[str, Any]] = []
        for rec in self._records:
            scene = rec.get("scene_summary", {})
            t = rec.get("t", 0.0)
            record_result: dict[str, Any] = {
                "t": t,
                "observations": [],
                "tracks": [],
            }
            for ent in scene.get("entities", []):
                eid = ent.get("entity_id", "")
                label = ent.get("label", "")
                bbox_raw = ent.get("bbox")
                bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None
                conf = float(ent.get("confidence", 0.5))
                obs = estimator.estimate(eid, label, bbox, conf, timestamp=t)
                if obs:
                    record_result["observations"].append(obs.to_dict())
                    track = estimator.update_track(obs)
                    record_result["tracks"].append(track.to_dict())
            results.append(record_result)
        return results

    def get_timeline(self) -> list[float]:
        return [r.get("t", 0.0) for r in self._records]

    @property
    def count(self) -> int:
        return len(self._records)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
