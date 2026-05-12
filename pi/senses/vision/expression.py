"""Facial expression analysis using Hailo-10H NPU."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

try:
    from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False

EXPRESSION_LABELS = [
    "neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"
]


@dataclass
class ExpressionResult:
    expression: str
    confidence: float
    probabilities: dict[str, float]


class ExpressionAnalyzer:
    def __init__(self, model_path: str = "facial_expression.hef", threshold: float = 0.4):
        self._model_path = model_path
        self._threshold = threshold
        self._vdevice = None
        self._owns_vdevice = False
        self._infer_model = None
        self._configured = None
        self._running = False

    def start(self, shared_vdevice=None) -> None:
        if HAS_HAILO:
            try:
                if shared_vdevice is not None:
                    self._vdevice = shared_vdevice
                    self._owns_vdevice = False
                else:
                    params = VDevice.create_params()
                    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                    self._vdevice = VDevice(params)
                    self._owns_vdevice = True
                self._infer_model = self._vdevice.create_infer_model(self._model_path)
                self._configured = self._infer_model.configure()
                logger.info("Expression model loaded: %s", self._model_path)
            except Exception as exc:
                logger.error("Failed to load expression model: %s", exc)
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._configured:
            try:
                self._configured.shutdown()
            except Exception:
                pass
        self._configured = None
        self._infer_model = None
        if self._owns_vdevice:
            self._vdevice = None

    def analyze(self, frame: np.ndarray, face_detections: list[Detection]) -> list[ExpressionResult]:
        if not self._running or not face_detections:
            return []

        results: list[ExpressionResult] = []

        for det in face_detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]

            if HAS_HAILO and self._configured:
                result = self._infer_expression(face_crop)
            else:
                result = ExpressionResult(
                    expression="neutral",
                    confidence=0.0,
                    probabilities={label: 0.0 for label in EXPRESSION_LABELS},
                )

            if result.confidence >= self._threshold:
                results.append(result)

        return results

    def _infer_expression(self, face_crop: np.ndarray) -> ExpressionResult:
        try:
            import cv2
            resized = np.zeros((48, 48, 3), dtype=np.float32)
            h, w = face_crop.shape[:2]
            if h > 0 and w > 0:
                scale = min(48 / h, 48 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                top = (48 - new_h) // 2
                left = (48 - new_w) // 2
                resized[top:top + new_h, left:left + new_w] = scaled

            input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

            bindings = self._configured.create_bindings()
            bindings.input().set_buffer(input_data)
            out_info = self._infer_model.output()
            out_size = out_info.shape[0] if len(out_info.shape) == 1 else int(np.prod(out_info.shape))
            output = np.empty(out_size, dtype=np.float32)
            bindings.output().set_buffer(output)
            self._configured.run([bindings], timeout=5000)

            probs = self._softmax(output[0])
            idx = int(np.argmax(probs))
            label = EXPRESSION_LABELS[idx] if idx < len(EXPRESSION_LABELS) else "unknown"
            prob_dict = {EXPRESSION_LABELS[i]: float(probs[i]) for i in range(min(len(probs), len(EXPRESSION_LABELS)))}

            return ExpressionResult(expression=label, confidence=float(probs[idx]), probabilities=prob_dict)

        except Exception as exc:
            logger.error("Expression inference failed: %s", exc)
            return ExpressionResult(
                expression="neutral",
                confidence=0.0,
                probabilities={label: 0.0 for label in EXPRESSION_LABELS},
            )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
