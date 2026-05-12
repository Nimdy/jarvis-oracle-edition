"""YOLOv8-Pose estimation via Hailo-10H — body language awareness.

Detects 17-point COCO keypoint skeletons and classifies high-level gestures
(waving, arms crossed, pointing, leaning) from pose geometry.

Uses the modern Hailo API (create_infer_model + configure) with multi-output
handling for the raw YOLOv8 anchor-free format (DFL bbox + conf + keypoints).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hailo_platform import VDevice, FormatType
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

_STRIDES = {80: 8, 40: 16, 20: 32}
_DFL_BINS = 16
_NMS_IOU_THRESHOLD = 0.65


@dataclass
class PoseResult:
    keypoints: list[tuple[float, float, float]]  # (x, y, confidence) per keypoint
    bbox: tuple[int, int, int, int] | None
    gesture: str
    confidence: float
    person_id: int = -1


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PoseEstimator:
    """YOLOv8-Pose on Hailo-10H for real-time skeleton tracking."""

    def __init__(
        self,
        model_path: str = "yolov8s_pose.hef",
        threshold: float = 0.4,
    ):
        self._model_path = model_path
        self._threshold = threshold
        self._vdevice = None
        self._infer_model = None
        self._configured = None
        self._input_name: str = ""
        self._output_groups: list[dict[str, str]] = []
        self._out_buffers: dict[str, np.ndarray] = {}
        self.available = False
        self._error_logged = False

        if not HAILO_AVAILABLE:
            logger.warning("hailo_platform not available — pose estimation disabled")
            return

        try:
            from pathlib import Path
            if not Path(model_path).exists():
                alt = f"/usr/share/hailo-models/{Path(model_path).name}"
                if Path(alt).exists():
                    self._model_path = alt
                else:
                    logger.warning("Pose model not found: %s", model_path)
                    return

            self.available = True
            logger.info("Pose estimator loaded: %s", self._model_path)
        except Exception as exc:
            logger.error("Failed to load pose model: %s", exc)

    def start(self, shared_vdevice=None) -> bool:
        if not self.available:
            return False
        try:
            if shared_vdevice:
                self._vdevice = shared_vdevice
            else:
                self._vdevice = VDevice()

            self._infer_model = self._vdevice.create_infer_model(self._model_path)

            for name in self._infer_model.output_names:
                self._infer_model.output(name).set_format_type(FormatType.FLOAT32)

            self._configured = self._infer_model.configure()
            self._input_name = self._infer_model.input_names[0]

            self._output_groups = self._group_outputs()
            logger.info("Pose estimator started (%d scales, %d outputs)",
                        len(self._output_groups), len(self._infer_model.output_names))
            return True
        except Exception as exc:
            logger.error("Failed to start pose estimator: %s", exc)
            return False

    def _group_outputs(self) -> list[dict[str, str]]:
        """Group output tensors by spatial scale into (bbox_dfl, conf, kpts) triples."""
        by_size: dict[int, list[tuple[str, tuple]]] = {}
        for name in self._infer_model.output_names:
            shape = self._infer_model.output(name).shape
            spatial = shape[0]  # H dimension
            by_size.setdefault(spatial, []).append((name, shape))

        groups = []
        for spatial in sorted(by_size.keys()):
            tensors = by_size[spatial]
            bbox_name = conf_name = kpts_name = ""
            for name, shape in tensors:
                channels = shape[-1]
                if channels == 64:
                    bbox_name = name
                elif channels == 1:
                    conf_name = name
                elif channels == 51:
                    kpts_name = name
            if bbox_name and conf_name and kpts_name:
                groups.append({
                    "bbox": bbox_name, "conf": conf_name, "kpts": kpts_name,
                    "spatial": spatial,
                })
        return groups

    def stop(self) -> None:
        if self._configured is not None:
            try:
                self._configured.shutdown()
            except Exception:
                pass
            self._configured = None
        self._infer_model = None

    def estimate(self, frame: np.ndarray) -> list[PoseResult]:
        """Run pose estimation on a BGR frame, return detected poses."""
        if not self.available or self._configured is None:
            return self._fallback_estimate(frame)
        try:
            input_data, lb = self._preprocess(frame)
            bindings = self._configured.create_bindings()
            bindings.input(self._input_name).set_buffer(input_data)

            if not self._out_buffers:
                for name in self._infer_model.output_names:
                    shape = self._infer_model.output(name).shape
                    self._out_buffers[name] = np.empty(shape, dtype=np.float32)

            for name, buf in self._out_buffers.items():
                bindings.output(name).set_buffer(buf)

            self._configured.run([bindings], timeout=10000)

            out_data: dict[str, np.ndarray] = {}
            for name in self._out_buffers:
                out_data[name] = bindings.output(name).get_buffer()

            return self._decode(out_data, frame.shape[:2], lb)
        except Exception as exc:
            if not self._error_logged:
                logger.warning("Pose inference error (suppressing repeats): %s", exc)
                self._error_logged = True
            return []

    def _fallback_estimate(self, frame: np.ndarray) -> list[PoseResult]:
        return []

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
        """Letterbox to 640x640, return (canvas, (scale, pad_left, pad_top))."""
        import cv2
        h, w = frame.shape[:2]
        if h == 640 and w == 640:
            return frame.astype(np.uint8), (1.0, 0, 0)
        scale = min(640 / h, 640 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        top = (640 - new_h) // 2
        left = (640 - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, (scale, left, top)

    def _decode(
        self,
        outputs: dict[str, np.ndarray],
        original_shape: tuple[int, int],
        lb: tuple[float, int, int],
    ) -> list[PoseResult]:
        """Decode raw YOLOv8-pose multi-scale outputs into PoseResults.

        Inverts the letterbox transform so coordinates are in original frame space.
        """
        oh, ow = original_shape
        scale, pad_left, pad_top = lb
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        all_boxes = []
        all_scores = []
        all_kpts = []

        for group in self._output_groups:
            spatial = group["spatial"]
            stride = _STRIDES.get(spatial, 640 // spatial)

            bbox_raw = np.array(outputs[group["bbox"]])   # [H, W, 64]
            conf_raw = np.array(outputs[group["conf"]])   # [H, W, 1]
            kpts_raw = np.array(outputs[group["kpts"]])   # [H, W, 51]

            h, w = bbox_raw.shape[:2]
            scores = _sigmoid(conf_raw.reshape(h * w))

            mask = scores > self._threshold
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            rows = indices // w
            cols = indices % w

            bbox_flat = bbox_raw.reshape(h * w, 64)[indices]
            kpts_flat = kpts_raw.reshape(h * w, 51)[indices]

            dfl = bbox_flat.reshape(-1, 4, _DFL_BINS)
            dfl_soft = _softmax(dfl)
            offsets = np.sum(dfl_soft * np.arange(_DFL_BINS), axis=-1)  # [N, 4]: l, t, r, b

            cx = (cols + 0.5) * stride
            cy = (rows + 0.5) * stride

            # Decode in 640-space then invert letterbox
            x1_640 = cx - offsets[:, 0] * stride
            y1_640 = cy - offsets[:, 1] * stride
            x2_640 = cx + offsets[:, 2] * stride
            y2_640 = cy + offsets[:, 3] * stride

            x1 = np.clip((x1_640 - pad_left) * inv_scale, 0, ow)
            y1 = np.clip((y1_640 - pad_top) * inv_scale, 0, oh)
            x2 = np.clip((x2_640 - pad_left) * inv_scale, 0, ow)
            y2 = np.clip((y2_640 - pad_top) * inv_scale, 0, oh)

            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            kp_reshaped = kpts_flat.reshape(-1, 17, 3)
            kp_x_640 = kp_reshaped[:, :, 0] * 2.0 * stride + cols[:, None] * stride
            kp_y_640 = kp_reshaped[:, :, 1] * 2.0 * stride + rows[:, None] * stride
            kp_x = np.clip((kp_x_640 - pad_left) * inv_scale, 0, ow)
            kp_y = np.clip((kp_y_640 - pad_top) * inv_scale, 0, oh)
            kp_conf = _sigmoid(kp_reshaped[:, :, 2])

            kps = np.stack([kp_x, kp_y, kp_conf], axis=-1)

            all_boxes.append(boxes)
            all_scores.append(scores[indices])
            all_kpts.append(kps)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kpts = np.concatenate(all_kpts)

        keep = self._nms(boxes, scores)

        results = []
        for idx in keep:
            b = boxes[idx]
            kp_list = [(float(kpts[idx, k, 0]), float(kpts[idx, k, 1]), float(kpts[idx, k, 2]))
                       for k in range(17)]
            gesture = classify_gesture(kp_list)
            results.append(PoseResult(
                keypoints=kp_list,
                bbox=(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                gesture=gesture,
                confidence=float(scores[idx]),
            ))
        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            remaining = order[1:]
            ious = np.array([_iou(boxes[i], boxes[j]) for j in remaining])
            order = remaining[ious < _NMS_IOU_THRESHOLD]
        return keep


def classify_gesture(keypoints: list[tuple[float, float, float]]) -> str:
    """Classify a high-level gesture from COCO keypoints.

    Returns: 'waving', 'arms_crossed', 'pointing', 'leaning_in',
             'leaning_away', 'hands_up', 'neutral'
    """
    def kp(name: str) -> tuple[float, float, float]:
        return keypoints[KP[name]]

    def visible(name: str) -> bool:
        return kp(name)[2] > 0.3

    lw = kp("left_wrist")
    rw = kp("right_wrist")
    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    le = kp("left_elbow")
    re = kp("right_elbow")
    lh = kp("left_hip")
    rh = kp("right_hip")
    nose = kp("nose")

    if not (visible("left_shoulder") and visible("right_shoulder")):
        return "neutral"

    shoulder_center_y = (ls[1] + rs[1]) / 2
    shoulder_width = abs(rs[0] - ls[0])

    if visible("left_wrist") and visible("right_wrist"):
        if lw[1] < shoulder_center_y - shoulder_width * 0.5 and rw[1] < shoulder_center_y - shoulder_width * 0.5:
            return "hands_up"

    # Waving: wrist must be clearly above the head (nose - half shoulder_width),
    # elbow must be above shoulder, AND the arm must be extended outward
    # (wrist far from shoulder horizontally). This prevents false positives
    # from resting chin on hand, scratching head, or casual arm positions.
    head_top_y = nose[1] - shoulder_width * 0.5 if visible("nose") else shoulder_center_y - shoulder_width
    if visible("left_wrist") and visible("left_elbow"):
        arm_extended = abs(lw[0] - ls[0]) > shoulder_width * 0.4
        if lw[1] < head_top_y and le[1] < ls[1] and arm_extended:
            return "waving"
    if visible("right_wrist") and visible("right_elbow"):
        arm_extended = abs(rw[0] - rs[0]) > shoulder_width * 0.4
        if rw[1] < head_top_y and re[1] < rs[1] and arm_extended:
            return "waving"

    if visible("left_wrist") and visible("right_wrist"):
        wrist_dist = abs(lw[0] - rw[0])
        if wrist_dist < shoulder_width * 0.4 and min(lw[1], rw[1]) > shoulder_center_y:
            return "arms_crossed"

    if visible("right_wrist") and visible("right_elbow"):
        arm_dx = rw[0] - re[0]
        arm_dy = abs(rw[1] - re[1])
        if arm_dx > shoulder_width * 0.5 and arm_dy < shoulder_width * 0.3:
            return "pointing"
    if visible("left_wrist") and visible("left_elbow"):
        arm_dx = le[0] - lw[0]
        arm_dy = abs(lw[1] - le[1])
        if arm_dx > shoulder_width * 0.5 and arm_dy < shoulder_width * 0.3:
            return "pointing"

    if visible("nose") and visible("left_hip") and visible("right_hip"):
        hip_center_x = (lh[0] + rh[0]) / 2
        lean = nose[0] - hip_center_x
        if lean > shoulder_width * 0.3:
            return "leaning_in"
        if lean < -shoulder_width * 0.3:
            return "leaning_away"

    return "neutral"
