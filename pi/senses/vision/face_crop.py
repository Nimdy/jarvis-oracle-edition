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

# Pose-keypoint face localization (preferred over the head-region heuristic).
# Uses the YOLOv8-Pose nose/eye keypoints the Pi already computes — no new model.
_NOSE, _L_EYE, _R_EYE = 0, 1, 2   # COCO-17 keypoint indices
KPT_CONF_MIN = 0.30               # min keypoint confidence to trust an eye
POSE_MATCH_MIN_IOU = 0.30         # min IOU to bind a pose to a person detection
MIN_INTEROCULAR_PX = 12           # below this the face is too small to align reliably


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
        poses: list | None = None,
    ) -> list[FaceCrop]:
        """Extract face crops from person detections in the current frame.

        When ``poses`` (YOLOv8-Pose results, already computed this frame) contain
        a confident eye pair matching the person, the crop is aligned on the eyes
        (a real face box). Otherwise it falls back to the head-region heuristic.
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

            face_box = self._face_box_from_pose(det.bbox, poses, frame.shape)
            if face_box is None:
                face_box = self._estimate_head_region(det.bbox, frame.shape)
            if face_box is None:
                continue

            x1, y1, x2, y2 = face_box
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
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _face_box_from_pose(cls, person_bbox, poses, frame_shape):
        """Eye-aligned square face box from the pose whose bbox best matches this
        person detection. Returns None (caller falls back to the heuristic) when
        no confident eye pair is available. Pure geometry; never raises.
        """
        if not poses:
            return None
        try:
            best, best_iou = None, 0.0
            for p in poses:
                pb = getattr(p, "bbox", None)
                if not pb:
                    continue
                iou = cls._bbox_iou(person_bbox, pb)
                if iou > best_iou:
                    best, best_iou = p, iou
            if best is None or best_iou < POSE_MATCH_MIN_IOU:
                return None
            kpts = getattr(best, "keypoints", None)
            if not kpts or len(kpts) <= _R_EYE:
                return None
            lx, ly, lc = kpts[_L_EYE]
            rx, ry, rc = kpts[_R_EYE]
            if lc < KPT_CONF_MIN or rc < KPT_CONF_MIN:
                return None  # need a confident eye pair to align
            d = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5  # inter-ocular distance (px)
            if d < MIN_INTEROCULAR_PX:
                return None
            eye_cx = (lx + rx) / 2.0
            eye_cy = (ly + ry) / 2.0
            half = 1.1 * d                  # square box ~2.2x inter-ocular
            cy = eye_cy + 0.5 * d           # shift center down toward nose/mouth
            H, W = frame_shape[0], frame_shape[1]
            x1 = max(0, int(eye_cx - half)); x2 = min(W, int(eye_cx + half))
            y1 = max(0, int(cy - half));     y2 = min(H, int(cy + half))
            if x2 - x1 < MIN_FACE_PIXELS or y2 - y1 < MIN_FACE_PIXELS:
                return None
            return (x1, y1, x2, y2)
        except Exception:
            return None

    @staticmethod
    def crop_to_bytes(crop: np.ndarray) -> bytes:
        """Encode a 112x112 crop as JPEG bytes for transmission."""
        if not HAS_CV2:
            return crop.tobytes()
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
