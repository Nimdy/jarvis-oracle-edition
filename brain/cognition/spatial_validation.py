"""Spatial validation — promotion rules and rejection tracking.

Implements the promotion chain for spatial claims:
  1. Raw observation -> track  (SpatialEstimator)
  2. Track -> validated spatial fact  (here)
  3. Track pair -> spatial delta  (here)
  4. Spatial delta -> world event  (here)
  5. World event -> memory candidate  (Ship 4, via CueGate)

Also enforces anchor authority and maintains a rejection ledger.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from typing import Any

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CLASS_MOVE_THRESHOLDS,
    CONFIDENCE_THRESHOLD_DELTA,
    DEFAULT_MOVE_THRESHOLD,
    DELTA_CONSECUTIVE_WINDOWS,
    STABLE_WINDOWS_REQUIRED,
    SpatialAnchor,
    SpatialDelta,
    SpatialTrack,
)

logger = logging.getLogger(__name__)


class RejectionLedger:
    """Tracks rejected spatial promotions with reason codes."""

    def __init__(self, maxlen: int = 200) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._counts: dict[str, int] = {}

    def record(
        self,
        entity_id: str,
        stage: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._entries.append({
            "entity_id": entity_id,
            "stage": stage,
            "reason": reason,
            "details": details or {},
            "timestamp": time.time(),
        })
        self._counts[reason] = self._counts.get(reason, 0) + 1

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        return list(self._entries)[-n:]

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def total_rejections(self) -> int:
        return sum(self._counts.values())


class SpatialValidator:
    """Validates spatial tracks and produces promoted deltas."""

    def __init__(self) -> None:
        self._previous_positions: dict[str, tuple[float, float, float]] = {}
        self._consecutive_windows: dict[str, int] = {}
        self._promoted_deltas: deque[SpatialDelta] = deque(maxlen=200)
        self._rejection_ledger = RejectionLedger()
        self._total_validated: int = 0
        self._total_promoted: int = 0

    @property
    def rejection_ledger(self) -> RejectionLedger:
        return self._rejection_ledger

    def validate_track_to_delta(
        self,
        track: SpatialTrack,
        anchors: dict[str, SpatialAnchor],
        calibration_version: int,
    ) -> SpatialDelta | None:
        """Check if a track qualifies for a promoted spatial delta."""
        eid = track.entity_id
        label_key = track.label.lower()

        if track.track_status != "stable":
            self._rejection_ledger.record(
                eid, "track_to_delta", "not_stable",
                {"status": track.track_status},
            )
            return None

        if track.stable_windows < STABLE_WINDOWS_REQUIRED:
            self._rejection_ledger.record(
                eid, "track_to_delta", "insufficient_stable_windows",
                {"windows": track.stable_windows, "required": STABLE_WINDOWS_REQUIRED},
            )
            return None

        for anchor in anchors.values():
            if anchor.authority <= track.authority:
                continue
            labels_match = (
                anchor.label.lower() == label_key
                or (anchor.label.lower() in ("monitor", "tv", "laptop")
                    and label_key in ("monitor", "tv", "laptop"))
            )
            if not labels_match:
                continue
            anchor_dist = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(track.position_room_m, anchor.position_room_m)
            ))
            threshold = CLASS_MOVE_THRESHOLDS.get(label_key, DEFAULT_MOVE_THRESHOLD)
            if anchor_dist > threshold:
                self._rejection_ledger.record(
                    eid, "track_to_delta", "anchor_authority_conflict",
                    {"anchor": anchor.anchor_id, "dist": round(anchor_dist, 3)},
                )
                return None

        prev_pos = self._previous_positions.get(eid)
        current_pos = track.position_room_m
        if prev_pos is None:
            self._previous_positions[eid] = current_pos
            self._consecutive_windows[eid] = 1
            return None

        displacement = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(current_pos, prev_pos)
        ))
        move_threshold = CLASS_MOVE_THRESHOLDS.get(label_key, DEFAULT_MOVE_THRESHOLD)

        if displacement < move_threshold:
            self._consecutive_windows[eid] = self._consecutive_windows.get(eid, 0) + 1
            self._previous_positions[eid] = current_pos
            return None

        consecutive = self._consecutive_windows.get(eid, 0)
        if consecutive < DELTA_CONSECUTIVE_WINDOWS:
            self._consecutive_windows[eid] = consecutive + 1
            return None

        dx = abs(current_pos[0] - prev_pos[0])
        dy = abs(current_pos[1] - prev_pos[1])
        dz = abs(current_pos[2] - prev_pos[2])
        if dx >= dy and dx >= dz:
            dominant = "x"
        elif dy >= dx and dy >= dz:
            dominant = "y"
        else:
            dominant = "z"

        from_conf = track.confidence
        to_conf = track.confidence
        disp_sig = min(1.0, displacement / (move_threshold * 3))
        multi_win = min(1.0, consecutive / 5)
        not_occluded = 1.0 if track.track_status == "stable" else 0.5

        delta_confidence = (
            0.30 * from_conf
            + 0.30 * to_conf
            + 0.20 * disp_sig
            + 0.10 * multi_win
            + 0.10 * not_occluded
        )

        if delta_confidence < CONFIDENCE_THRESHOLD_DELTA:
            self._rejection_ledger.record(
                eid, "track_to_delta", "low_confidence",
                {"confidence": round(delta_confidence, 3),
                 "threshold": CONFIDENCE_THRESHOLD_DELTA},
            )
            self._previous_positions[eid] = current_pos
            self._consecutive_windows[eid] = 0
            return None

        delta = SpatialDelta(
            delta_id=f"sdelta_{uuid.uuid4().hex[:10]}",
            entity_id=eid,
            label=track.label,
            delta_type="moved",
            from_position_m=prev_pos,
            to_position_m=current_pos,
            distance_m=displacement,
            dominant_axis=dominant,
            confidence=delta_confidence,
            uncertainty_m=track.uncertainty_m,
            validated=True,
            reason_codes=["stable_track", f"displacement_{displacement:.3f}m"],
            calibration_version=calibration_version,
            timestamp=time.time(),
        )
        self._promoted_deltas.append(delta)
        self._total_promoted += 1
        self._total_validated += 1
        self._previous_positions[eid] = current_pos
        self._consecutive_windows[eid] = 0
        logger.info(
            "Spatial delta: %s %s %.3fm (%s)",
            track.label, "moved", displacement, dominant,
        )
        return delta

    def check_missing_entity(
        self,
        entity_id: str,
        label: str,
        last_known_pos: tuple[float, float, float],
        calibration_version: int,
    ) -> SpatialDelta:
        delta = SpatialDelta(
            delta_id=f"sdelta_{uuid.uuid4().hex[:10]}",
            entity_id=entity_id,
            label=label,
            delta_type="missing",
            from_position_m=last_known_pos,
            confidence=0.6,
            uncertainty_m=1.0,
            reason_codes=["entity_vanished_from_stable_position"],
            calibration_version=calibration_version,
            timestamp=time.time(),
        )
        self._promoted_deltas.append(delta)
        self._total_validated += 1
        return delta

    def get_promoted_deltas(self, n: int = 20) -> list[dict[str, Any]]:
        return [d.to_dict() for d in list(self._promoted_deltas)[-n:]]

    def reset_for_relocalization(self, *, profile_id: str = "", reason: str = "") -> None:
        """Reset frame-to-frame baseline state after spatial profile handoff."""
        prev = len(self._previous_positions)
        windows = len(self._consecutive_windows)
        self._previous_positions.clear()
        self._consecutive_windows.clear()
        logger.info(
            "Spatial validator relocalization reset: profile=%s previous=%d windows=%d reason=%s",
            profile_id or "unknown",
            prev,
            windows,
            reason or "unspecified",
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "total_validated": self._total_validated,
            "total_promoted": self._total_promoted,
            "total_rejections": self._rejection_ledger.total_rejections,
            "rejection_counts": self._rejection_ledger.get_counts(),
            "recent_rejections": self._rejection_ledger.get_recent(10),
            "recent_deltas": self.get_promoted_deltas(10),
        }
