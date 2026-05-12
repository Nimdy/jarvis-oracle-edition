"""CPU-based YOLOv8n scene object detector.

Runs YOLOv8n ONNX on the Pi5 CPU at low cadence (every 2-3s) to detect
non-person scene objects that the Hailo NPU misses due to quantization
degradation. Person detection stays on the Hailo for real-time 15fps
tracking; this module handles static scene objects like monitors,
keyboards, cups, chairs, etc.

Output shape of YOLOv8n ONNX: (1, 84, 8400)
  - 84 = 4 bbox coords (cx, cy, w, h) + 80 class scores
  - 8400 = number of candidate anchors across 3 scales
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    logger.warning("onnxruntime not available — scene detection disabled")


SCENE_LABELS = {
    "bicycle", "car", "motorcycle", "bus", "truck", "boat",
    "bench", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
}

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class SceneDetection:
    __slots__ = ("label", "confidence", "bbox", "timestamp")

    def __init__(self, label: str, confidence: float, bbox: tuple[int, int, int, int]):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = time.time()


class SceneDetector:
    """Low-cadence CPU YOLO for scene objects.

    Designed to run every 2-3 seconds in a background thread. ~400ms
    inference on Pi5 A76 cores = ~15% CPU at 3s cadence.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._available = False
        self._last_detections: list[SceneDetection] = []
        self._last_run_ms: float = 0.0
        self._run_count: int = 0

    def start(self) -> None:
        if not HAS_ORT:
            logger.info("SceneDetector: onnxruntime not available, staying disabled")
            return

        import os
        if not os.path.exists(self._model_path):
            logger.warning("SceneDetector: model not found at %s", self._model_path)
            return

        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._available = True

            # Warmup
            dummy = np.zeros((1, 3, self._input_size, self._input_size), dtype=np.float32)
            self._session.run(None, {self._input_name: dummy})

            logger.info(
                "SceneDetector: YOLOv8n ONNX loaded (conf=%.2f, iou=%.2f, input=%d)",
                self._conf_threshold, self._iou_threshold, self._input_size,
            )
        except Exception as exc:
            logger.error("SceneDetector: failed to load model: %s", exc)
            self._session = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_detections(self) -> list[SceneDetection]:
        return list(self._last_detections)

    @property
    def last_run_ms(self) -> float:
        return self._last_run_ms

    def detect(self, frame: np.ndarray) -> list[SceneDetection]:
        """Run scene detection on a camera frame. Returns non-person scene objects."""
        if not self._available or self._session is None:
            return []

        t0 = time.monotonic()

        blob, ratio, pad_w, pad_h = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]  # (1, 84, 8400)

        detections = self._postprocess(raw, frame.shape, ratio, pad_w, pad_h)
        self._last_detections = detections
        self._last_run_ms = (time.monotonic() - t0) * 1000
        self._run_count += 1

        if self._run_count <= 3 or self._run_count % 20 == 0:
            labels = [f"{d.label}:{d.confidence:.2f}" for d in detections[:8]]
            logger.info(
                "SceneDetector: %d objects in %.0fms%s",
                len(detections), self._last_run_ms,
                f" [{', '.join(labels)}]" if labels else "",
            )

        return detections

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize + normalize to NCHW float32."""
        h, w = frame.shape[:2]
        sz = self._input_size
        ratio = min(sz / h, sz / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        pad_h = (sz - new_h) // 2
        pad_w = (sz - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        return blob, ratio, pad_w, pad_h

    def _postprocess(
        self,
        raw: np.ndarray,
        frame_shape: tuple,
        ratio: float,
        pad_w: int,
        pad_h: int,
    ) -> list[SceneDetection]:
        """Parse (1, 84, 8400) output, apply NMS, filter to scene classes."""
        predictions = raw[0]  # (84, 8400)
        predictions = predictions.T  # (8400, 84)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]  # (8400, 80)

        max_scores = class_scores.max(axis=1)
        mask = max_scores >= self._conf_threshold
        if not mask.any():
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        class_scores = class_scores[mask]
        max_scores = max_scores[mask]
        class_ids = class_scores.argmax(axis=1)

        # Convert cx,cy,w,h -> x1,y1,x2,y2 in input-space (640x640)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS per class
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            max_scores.tolist(),
            self._conf_threshold,
            self._iou_threshold,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        h, w = frame_shape[:2]
        inv_ratio = 1.0 / ratio if ratio > 0 else 1.0
        detections: list[SceneDetection] = []

        for idx in indices:
            cls_id = int(class_ids[idx])
            if cls_id >= len(COCO_LABELS):
                continue
            label = COCO_LABELS[cls_id]

            if label == "person":
                continue
            if label not in SCENE_LABELS:
                continue

            score = float(max_scores[idx])
            bx1 = int(max(0, (boxes_xyxy[idx, 0] - pad_w) * inv_ratio))
            by1 = int(max(0, (boxes_xyxy[idx, 1] - pad_h) * inv_ratio))
            bx2 = int(min(w, (boxes_xyxy[idx, 2] - pad_w) * inv_ratio))
            by2 = int(min(h, (boxes_xyxy[idx, 3] - pad_h) * inv_ratio))

            detections.append(SceneDetection(
                label=label, confidence=score, bbox=(bx1, by1, bx2, by2),
            ))

        return detections

    def stop(self) -> None:
        self._session = None
        self._available = False

    def get_stats(self) -> dict:
        return {
            "available": self._available,
            "run_count": self._run_count,
            "last_run_ms": round(self._last_run_ms, 1),
            "last_detection_count": len(self._last_detections),
            "conf_threshold": self._conf_threshold,
            "model": self._model_path,
        }
