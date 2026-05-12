"""Object and person detection via Hailo-10H NPU."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False
    logger.warning("picamera2 not available -- using stub camera")

try:
    from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False
    logger.warning("hailo_platform not available -- detections will be empty")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float = field(default_factory=time.time)


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

NUM_CLASSES = 80
MAX_DETECTIONS_PER_CLASS = 100
NMS_ENTRY_SIZE = 5  # y_min, x_min, y_max, x_max, score
NMS_STRIDE = 1 + MAX_DETECTIONS_PER_CLASS * NMS_ENTRY_SIZE  # count + detections


class Detector:
    PERSON_LABELS = {"person"}

    SENSOR_FULL_W = 4656
    SENSOR_FULL_H = 3496
    FULL_FOV_RAW = (2328, 1748)

    def __init__(
        self,
        model_path: str = "yolov8s.hef",
        threshold: float = 0.5,
        scene_threshold: float = 0.30,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ):
        self._model_path = model_path
        self._threshold = threshold
        self._scene_threshold = scene_threshold
        self._camera_id = camera_id
        self._width = width
        self._height = height
        self._fps = fps
        self._camera: Picamera2 | None = None
        self._vdevice = None
        self._infer_model = None
        self._configured = None
        self._running = False
        self._last_detections: list[Detection] = []
        self._zoom_level: float = 1.0
        self._zoom_center: tuple[float, float] = (0.5, 0.5)
        self._has_autofocus: bool = False
        self._af_mode: str = "manual"

    def start(self) -> None:
        if HAS_PICAMERA:
            try:
                self._camera = Picamera2(self._camera_id)
                config = self._camera.create_preview_configuration(
                    main={"size": (self._width, self._height), "format": "RGB888"},
                    raw={"size": self.FULL_FOV_RAW},
                )
                self._camera.configure(config)
                self._camera.start()

                ctrls = self._camera.camera_controls
                self._has_autofocus = "AfMode" in ctrls
                if self._has_autofocus:
                    self._camera.set_controls({"AfMode": 2})
                    self._af_mode = "continuous"
                    logger.info("Autofocus enabled (continuous mode)")
                else:
                    logger.info("Autofocus not available (VCM disabled or not present)")

                self._apply_scaler_crop()
                logger.info(
                    "Camera started: %dx%d @ %dfps (full-FOV sensor mode %dx%d, AF=%s)",
                    self._width, self._height, self._fps,
                    self.FULL_FOV_RAW[0], self.FULL_FOV_RAW[1],
                    self._has_autofocus,
                )
            except RuntimeError as exc:
                logger.error("Camera unavailable (%s) — running in stub mode", exc)
                self._camera = None
        else:
            logger.info("Running without camera (stub mode)")

        if HAS_HAILO:
            try:
                params = VDevice.create_params()
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                self._vdevice = VDevice(params)
                self._infer_model = self._vdevice.create_infer_model(self._model_path)
                self._configured = self._infer_model.configure()
                logger.info("Hailo model loaded: %s (person=%.2f, scene=%.2f)",
                            self._model_path, self._threshold, self._scene_threshold)
            except Exception as exc:
                logger.error("Failed to load Hailo model: %s", exc)
                self._vdevice = None
                self._infer_model = None
                self._configured = None

        self._running = True

    @property
    def vdevice(self):
        """Expose VDevice for model sharing (e.g. expression analyzer)."""
        return self._vdevice

    def stop(self) -> None:
        self._running = False
        if self._configured:
            try:
                self._configured.shutdown()
            except Exception:
                pass
        self._configured = None
        self._infer_model = None
        self._vdevice = None
        if self._camera:
            self._camera.stop()
            self._camera = None

    # --- Camera control: zoom and focus ---

    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    @property
    def has_autofocus(self) -> bool:
        return self._has_autofocus

    @property
    def af_mode(self) -> str:
        return self._af_mode

    def set_zoom(self, level: float) -> None:
        """Set digital zoom (1.0 = full FOV, max 8.0x)."""
        level = max(1.0, min(level, 8.0))
        if abs(level - self._zoom_level) < 0.01:
            return
        self._zoom_level = level
        self._zoom_center = (0.5, 0.5)
        self._apply_scaler_crop()
        logger.info("Zoom set to %.1fx", level)

    def zoom_to_region(self, x1: int, y1: int, x2: int, y2: int, padding: float = 1.5) -> None:
        """Zoom to a region specified in frame coordinates with padding multiplier."""
        if not self._camera:
            return
        cx = ((x1 + x2) / 2) / self._width
        cy = ((y1 + y2) / 2) / self._height
        region_w = (x2 - x1) / self._width
        region_h = (y2 - y1) / self._height
        target_fraction = max(region_w, region_h) * padding
        target_fraction = max(0.125, min(1.0, target_fraction))
        self._zoom_level = 1.0 / target_fraction
        self._zoom_center = (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))
        self._apply_scaler_crop()
        logger.info("Zoom to region (%d,%d,%d,%d) → %.1fx center=(%.2f,%.2f)",
                     x1, y1, x2, y2, self._zoom_level, *self._zoom_center)

    def reset_zoom(self) -> None:
        """Reset to full field of view."""
        self._zoom_level = 1.0
        self._zoom_center = (0.5, 0.5)
        self._apply_scaler_crop()
        logger.info("Zoom reset to full FOV")

    def trigger_autofocus(self) -> bool:
        """Trigger a single autofocus cycle. Returns True if AF is available."""
        if not self._has_autofocus or not self._camera:
            return False
        self._camera.set_controls({"AfMode": 1, "AfTrigger": 0})
        self._af_mode = "single"
        logger.info("Autofocus triggered (single shot)")
        return True

    def set_continuous_autofocus(self) -> bool:
        """Enable continuous autofocus."""
        if not self._has_autofocus or not self._camera:
            return False
        self._camera.set_controls({"AfMode": 2})
        self._af_mode = "continuous"
        logger.info("Autofocus set to continuous")
        return True

    def set_manual_focus(self, position: float) -> bool:
        """Set manual focus (0.0 = infinity, higher = macro). Returns True if AF available."""
        if not self._has_autofocus or not self._camera:
            return False
        position = max(0.0, min(32.0, position))
        self._camera.set_controls({"AfMode": 0, "LensPosition": position})
        self._af_mode = "manual"
        logger.info("Manual focus set to %.2f", position)
        return True

    def get_camera_state(self) -> dict:
        """Return current camera control state for telemetry."""
        return {
            "zoom_level": round(self._zoom_level, 2),
            "zoom_center": [round(c, 3) for c in self._zoom_center],
            "has_autofocus": self._has_autofocus,
            "af_mode": self._af_mode,
            "resolution": [self._width, self._height],
            "sensor_mode": list(self.FULL_FOV_RAW),
        }

    def _apply_scaler_crop(self) -> None:
        if not self._camera:
            return
        fw, fh = self.SENSOR_FULL_W, self.SENSOR_FULL_H
        crop_w = int(fw / self._zoom_level)
        crop_h = int(fh / self._zoom_level)
        cx_pix = int(self._zoom_center[0] * fw)
        cy_pix = int(self._zoom_center[1] * fh)
        crop_x = max(0, min(fw - crop_w, cx_pix - crop_w // 2))
        crop_y = max(0, min(fh - crop_h, cy_pix - crop_h // 2))
        try:
            self._camera.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
        except Exception as exc:
            logger.warning("Failed to set ScalerCrop: %s", exc)

    def capture_and_detect(self) -> list[Detection]:
        if not self._running:
            return []

        frame = self._capture_frame()
        if frame is None:
            return []

        self._last_frame = frame
        detections = self._run_inference(frame)
        self._last_detections = detections
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a pre-captured frame (avoids re-capture)."""
        if not self._running:
            return []
        self._last_frame = frame
        detections = self._run_inference(frame)
        self._last_detections = detections
        return detections

    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame. Use with detect() for single-capture-per-tick."""
        return self._capture_frame()

    @property
    def last_frame(self) -> np.ndarray | None:
        return getattr(self, "_last_frame", None)

    def get_persons(self) -> list[Detection]:
        return [d for d in self._last_detections if d.label in self.PERSON_LABELS]

    def is_person_present(self) -> bool:
        return len(self.get_persons()) > 0

    @property
    def last_detections(self) -> list[Detection]:
        return list(self._last_detections)

    def _capture_frame(self) -> np.ndarray | None:
        if self._camera and HAS_PICAMERA:
            return self._camera.capture_array()
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def _run_inference(self, frame: np.ndarray) -> list[Detection]:
        if not HAS_HAILO or not self._configured:
            return self._stub_inference()

        try:
            input_data, lb = self._preprocess(frame)
            bindings = self._configured.create_bindings()
            bindings.input().set_buffer(input_data)

            out_info = self._infer_model.output()
            output_size = out_info.shape[0] if len(out_info.shape) == 1 else int(np.prod(out_info.shape))
            output_buf = np.empty(output_size, dtype=np.float32)
            bindings.output().set_buffer(output_buf)

            self._configured.run([bindings], timeout=10000)
            return self._parse_nms_output(output_buf, frame.shape, lb)
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return []

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
        """Letterbox resize: maintain aspect ratio, pad with grey (114).

        Returns (canvas, (scale, pad_left, pad_top)) so decode can invert.
        """
        h, w = frame.shape[:2]
        if h == 640 and w == 640:
            return frame.astype(np.uint8), (1.0, 0, 0)

        import cv2
        scale = min(640 / h, 640 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        top = (640 - new_h) // 2
        left = (640 - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, (scale, left, top)

    def _parse_nms_output(
        self, raw: np.ndarray, frame_shape: tuple, lb: tuple[float, int, int],
    ) -> list[Detection]:
        """Parse Hailo NMS output and map boxes from 640-space back to original frame.

        NMS coords are normalized [0,1] relative to the 640x640 input canvas.
        We invert the letterbox: pixel_640 = norm * 640, then
        x_orig = (pixel_640 - pad_left) / scale, y_orig = (pixel_640 - pad_top) / scale.
        """
        detections: list[Detection] = []
        h, w = frame_shape[:2]
        scale, pad_left, pad_top = lb
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        # Diagnostic: dump raw NMS class counts periodically
        _diag_due = not hasattr(self, "_nms_diag_ts") or (time.time() - self._nms_diag_ts > 30)
        _diag_lines: list[str] = []

        for cls_id in range(NUM_CLASSES):
            offset = cls_id * NMS_STRIDE
            if offset >= len(raw):
                break
            num_dets = int(raw[offset])

            if _diag_due and num_dets > 0:
                label_d = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
                top_score = 0.0
                for dd in range(min(num_dets, 3)):
                    base_d = offset + 1 + dd * NMS_ENTRY_SIZE
                    if base_d + NMS_ENTRY_SIZE <= len(raw):
                        top_score = max(top_score, float(raw[base_d + 4]))
                _diag_lines.append(f"  {label_d}: raw_count={num_dets} top_score={top_score:.3f}")

            if num_dets <= 0:
                continue
            num_dets = min(num_dets, MAX_DETECTIONS_PER_CLASS)

            for d in range(num_dets):
                base = offset + 1 + d * NMS_ENTRY_SIZE
                if base + NMS_ENTRY_SIZE > len(raw):
                    break
                y_min, x_min, y_max, x_max, score = raw[base:base + NMS_ENTRY_SIZE]
                label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"class_{cls_id}"
                thr = self._threshold if label in self.PERSON_LABELS else self._scene_threshold
                if score < thr:
                    continue
                x1 = int(max(0, (x_min * 640 - pad_left) * inv_scale))
                y1 = int(max(0, (y_min * 640 - pad_top) * inv_scale))
                x2 = int(min(w, (x_max * 640 - pad_left) * inv_scale))
                y2 = int(min(h, (y_max * 640 - pad_top) * inv_scale))
                detections.append(Detection(label=label, confidence=float(score), bbox=(x1, y1, x2, y2)))

        if _diag_due:
            self._nms_diag_ts = time.time()
            if _diag_lines:
                logger.info("Hailo NMS (%d classes): %s",
                            len(_diag_lines), "; ".join(_diag_lines))

        return detections

    def _stub_inference(self) -> list[Detection]:
        """Placeholder when Hailo is unavailable."""
        return []
