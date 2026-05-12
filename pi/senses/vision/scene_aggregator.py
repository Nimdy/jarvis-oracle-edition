"""Scene Aggregator — accumulates non-person COCO detections across frames
and emits compact scene summaries to the brain at controlled intervals.

Pi runs object detection at 8-15 FPS but most objects are stable.
Instead of streaming every detection, we aggregate across frames, dedupe
by class + spatial overlap, compute a scene_change_score, and emit a
compact summary every few seconds. The brain owns the world model;
the Pi is a fast perception node.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from senses.vision.detector import Detection

SUMMARY_INTERVAL_S = 3.0
HEARTBEAT_INTERVAL_S = 15.0
CHANGE_THRESHOLD = 0.08
IOU_MERGE_THRESHOLD = 0.4
PERSON_LABEL = "person"


@dataclass
class _AggDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    hit_count: int = 1
    last_seen: float = field(default_factory=time.time)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


class SceneAggregator:
    """Accumulates non-person detections and emits periodic scene summaries."""

    def __init__(
        self,
        summary_interval: float = SUMMARY_INTERVAL_S,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_S,
        change_threshold: float = CHANGE_THRESHOLD,
    ) -> None:
        self._interval = summary_interval
        self._heartbeat = heartbeat_interval
        self._change_threshold = change_threshold

        self._current: dict[str, list[_AggDetection]] = defaultdict(list)
        self._prev_labels: set[str] = set()
        self._prev_count: int = 0

        self._last_emit_ts: float = 0.0
        self._last_heartbeat_ts: float = 0.0
        self._frame_w: int = 640
        self._frame_h: int = 480

    def set_frame_size(self, w: int, h: int) -> None:
        self._frame_w = w
        self._frame_h = h

    def feed(self, detections: list[Detection]) -> dict | None:
        """Feed a frame's worth of detections. Returns a summary dict if
        it's time to emit, otherwise None."""
        now = time.time()

        for det in detections:
            if det.label == PERSON_LABEL:
                continue
            self._merge_detection(det)

        self._prune_stale(now, max_age=self._interval * 2.5)

        elapsed = now - self._last_emit_ts
        heartbeat_due = (now - self._last_heartbeat_ts) >= self._heartbeat

        if elapsed < self._interval and not heartbeat_due:
            return None

        summary = self._build_summary(now)
        change_score = summary["scene_change_score"]

        if change_score >= self._change_threshold or heartbeat_due or self._last_emit_ts == 0.0:
            self._last_emit_ts = now
            if heartbeat_due:
                self._last_heartbeat_ts = now
            self._prev_labels = {d["label"] for d in summary["detections"]}
            self._prev_count = len(summary["detections"])
            return summary

        return None

    def _merge_detection(self, det: Detection) -> None:
        """Merge a detection into the aggregation buffer by class + spatial overlap."""
        label = det.label
        existing = self._current[label]

        for agg in existing:
            if _iou(agg.bbox, det.bbox) >= IOU_MERGE_THRESHOLD:
                agg.confidence = max(agg.confidence, det.confidence)
                agg.bbox = self._merge_bbox(agg.bbox, det.bbox)
                agg.hit_count += 1
                agg.last_seen = time.time()
                return

        existing.append(_AggDetection(
            label=label,
            confidence=det.confidence,
            bbox=det.bbox,
            hit_count=1,
            last_seen=time.time(),
        ))

    @staticmethod
    def _merge_bbox(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def _prune_stale(self, now: float, max_age: float) -> None:
        for label in list(self._current):
            self._current[label] = [
                a for a in self._current[label] if (now - a.last_seen) < max_age
            ]
            if not self._current[label]:
                del self._current[label]

    def _build_summary(self, now: float) -> dict:
        detections: list[dict] = []
        current_labels: set[str] = set()

        for label, aggs in self._current.items():
            for agg in aggs:
                detections.append({
                    "label": agg.label,
                    "confidence": round(agg.confidence, 3),
                    "bbox": list(agg.bbox),
                    "hit_count": agg.hit_count,
                })
                current_labels.add(label)

        change_score = self._compute_change(current_labels, len(detections))

        return {
            "frame_size": [self._frame_w, self._frame_h],
            "detections": detections,
            "scene_change_score": round(change_score, 3),
        }

    def _compute_change(self, current_labels: set[str], current_count: int) -> float:
        if not self._prev_labels and not current_labels:
            return 0.0
        if not self._prev_labels:
            return 1.0

        union = self._prev_labels | current_labels
        if not union:
            return 0.0
        intersection = self._prev_labels & current_labels
        jaccard_dist = 1.0 - len(intersection) / len(union)

        count_delta = abs(current_count - self._prev_count)
        count_factor = min(1.0, count_delta / max(1, self._prev_count + current_count))

        return min(1.0, jaccard_dist * 0.6 + count_factor * 0.4)
