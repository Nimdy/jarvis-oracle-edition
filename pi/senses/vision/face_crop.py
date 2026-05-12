"""Face crop extraction for identity pipeline.

Extracts aligned face regions from camera frames using person detections.
Sends crops to the brain for embedding + matching (hybrid architecture).

Current approach: head-region heuristic from person bounding boxes.
Future upgrade: replace with SCRFD HEF on Hailo for proper face detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

CROP_SIZE = 112
MIN_CROP_INTERVAL_S = 2.0
MIN_FACE_PIXELS = 30


@dataclass
class FaceCrop:
    crop: np.ndarray  # 112x112x3 uint8
    track_id: int
    confidence: float
    timestamp: float = field(default_factory=time.time)


class FaceCropExtractor:
    """Extracts face-region crops from person detections for identity matching."""

    def __init__(self, crop_interval_s: float = MIN_CROP_INTERVAL_S) -> None:
        self._crop_interval = crop_interval_s
        self._last_crop_time: dict[int, float] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and HAS_CV2

    def set_enabled(self, val: bool) -> None:
        self._enabled = val

    def extract(
        self,
        frame: np.ndarray,
        person_detections: list[Detection],
        track_ids: list[int] | None = None,
    ) -> list[FaceCrop]:
        """Extract face crops from person detections in the current frame.

        Rate-limits to one crop per track_id per crop_interval_s.
        """
        if not self._enabled or not HAS_CV2 or frame is None:
            return []

        now = time.time()
        crops: list[FaceCrop] = []

        for i, det in enumerate(person_detections[:5]):
            track_id = track_ids[i] if track_ids and i < len(track_ids) else i

            last = self._last_crop_time.get(track_id, 0)
            if now - last < self._crop_interval:
                continue

            head_box = self._estimate_head_region(det.bbox, frame.shape)
            if head_box is None:
                continue

            x1, y1, x2, y2 = head_box
            face_region = frame[y1:y2, x1:x2]
            if face_region.shape[0] < MIN_FACE_PIXELS or face_region.shape[1] < MIN_FACE_PIXELS:
                continue

            aligned = cv2.resize(face_region, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)

            crops.append(FaceCrop(
                crop=aligned,
                track_id=track_id,
                confidence=det.confidence,
            ))
            self._last_crop_time[track_id] = now

        stale = [tid for tid, t in self._last_crop_time.items() if now - t > 30.0]
        for tid in stale:
            del self._last_crop_time[tid]

        return crops

    @staticmethod
    def _estimate_head_region(
        person_bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        """Estimate head bounding box from a person detection.

        Assumes head occupies roughly the top 25% of the person bbox
        with slight horizontal centering.
        """
        x1, y1, x2, y2 = person_bbox
        person_h = y2 - y1
        person_w = x2 - x1

        if person_h < 40 or person_w < 20:
            return None

        head_h = int(person_h * 0.25)
        head_w = int(min(head_h * 0.85, person_w * 0.7))
        cx = (x1 + x2) // 2

        hx1 = max(0, cx - head_w // 2)
        hx2 = min(frame_shape[1], cx + head_w // 2)
        hy1 = max(0, y1)
        hy2 = min(frame_shape[0], y1 + head_h)

        if hx2 - hx1 < MIN_FACE_PIXELS or hy2 - hy1 < MIN_FACE_PIXELS:
            return None

        return (hx1, hy1, hx2, hy2)

    @staticmethod
    def crop_to_bytes(crop: np.ndarray) -> bytes:
        """Encode a 112x112 crop as JPEG bytes for transmission."""
        if not HAS_CV2:
            return crop.tobytes()
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
