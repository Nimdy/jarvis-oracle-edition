"""Person tracking and gesture recognition via detection history.

Gesture system: Pose estimator is the primary gesture source.
Tracker falls back to primitive bbox-motion gestures if pose
data is not available for a given person.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from collections import deque

from .detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    id: int
    first_seen: float
    last_seen: float
    bbox_history: deque[tuple[int, int, int, int]] = field(default_factory=lambda: deque(maxlen=30))
    confidence_avg: float = 0.0
    gesture: str | None = None
    pose_gesture: str | None = None  # Set by pose estimator (authoritative)

    @property
    def effective_gesture(self) -> str | None:
        """Pose gesture wins over tracker motion gesture."""
        return self.pose_gesture or self.gesture

    @property
    def duration_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def center(self) -> tuple[int, int] | None:
        if not self.bbox_history:
            return None
        x1, y1, x2, y2 = self.bbox_history[-1]
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class PersonTracker:
    def __init__(self, max_absent_s: float = 5.0, iou_threshold: float = 0.3):
        self._max_absent_s = max_absent_s
        self._iou_threshold = iou_threshold
        self._tracks: dict[int, TrackedPerson] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        now = time.time()
        person_dets = [d for d in detections if d.label == "person"]
        matched_track_ids: set[int] = set()
        unmatched_dets: list[Detection] = []

        for det in person_dets:
            best_id: int | None = None
            best_iou = self._iou_threshold

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if not track.bbox_history:
                    continue
                iou = self._compute_iou(det.bbox, track.bbox_history[-1])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                track = self._tracks[best_id]
                track.last_seen = now
                track.bbox_history.append(det.bbox)
                track.confidence_avg = (track.confidence_avg * 0.9) + (det.confidence * 0.1)
                matched_track_ids.add(best_id)
            else:
                unmatched_dets.append(det)

        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = TrackedPerson(
                id=tid,
                first_seen=now,
                last_seen=now,
                bbox_history=deque([det.bbox], maxlen=30),
                confidence_avg=det.confidence,
            )
            logger.debug("New person track: id=%d", tid)

        expired = [tid for tid, t in self._tracks.items() if (now - t.last_seen) > self._max_absent_s]
        for tid in expired:
            logger.debug("Person track expired: id=%d, duration=%.1fs", tid, self._tracks[tid].duration_s)
            del self._tracks[tid]

        self._detect_gestures()

        return list(self._tracks.values())

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @property
    def active_persons(self) -> list[TrackedPerson]:
        return list(self._tracks.values())

    def set_pose_gestures(self, pose_results: list) -> None:
        """Inject pose-based gestures from PoseEstimator results.

        Each result should have .bbox and .gesture attributes.
        Matches to tracks by IOU, sets pose_gesture on the best match.
        """
        for track in self._tracks.values():
            track.pose_gesture = None

        for pose in pose_results:
            if not hasattr(pose, "gesture") or not hasattr(pose, "bbox") or not pose.bbox:
                continue
            best_tid = None
            best_iou = 0.2
            for tid, track in self._tracks.items():
                if not track.bbox_history:
                    continue
                iou = self._compute_iou(pose.bbox, track.bbox_history[-1])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                self._tracks[best_tid].pose_gesture = pose.gesture

    def _detect_gestures(self) -> None:
        """Fallback bbox-motion gestures when pose data is unavailable.

        Gated behind confidence and duration to prevent false positives
        from tracking noise. Requires oscillation (direction reversals)
        to distinguish waving from walking sideways.
        """
        for track in self._tracks.values():
            if track.pose_gesture:
                track.gesture = None
                continue

            if len(track.bbox_history) < 10:
                track.gesture = None
                continue

            if track.confidence_avg < 0.6 or track.duration_s < 1.0:
                track.gesture = None
                continue

            centers = []
            for bbox in list(track.bbox_history)[-10:]:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                centers.append((cx, cy))

            dx = abs(centers[-1][0] - centers[0][0])
            dy = abs(centers[-1][1] - centers[0][1])

            if dx > 80 and dy < 50:
                x_reversals = sum(
                    1 for i in range(2, len(centers))
                    if (centers[i][0] - centers[i - 1][0]) * (centers[i - 1][0] - centers[i - 2][0]) < 0
                )
                if x_reversals >= 2:
                    track.gesture = "wave"
                else:
                    track.gesture = None
            elif dy > 60 and dx < 50:
                y_reversals = sum(
                    1 for i in range(2, len(centers))
                    if (centers[i][1] - centers[i - 1][1]) * (centers[i - 1][1] - centers[i - 2][1]) < 0
                )
                if y_reversals >= 2:
                    track.gesture = "nod"
                else:
                    track.gesture = None
            else:
                track.gesture = None

    @staticmethod
    def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)
