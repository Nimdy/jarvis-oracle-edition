"""Jarvis Senses — Pi 5 entry point (thin sensor node).

Streams raw mic audio to the laptop brain over WebSocket binary frames.
All audio processing (wake word, VAD, STT, TTS) runs on the brain.
The Pi handles vision (Hailo AI HAT+), audio capture, brain audio playback,
and the particle display UI.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque

import numpy as np

from config import SensesConfig
from transport.ws_client import TransportClient
from transport.event_schema import (
    BrainMessage, PerceptionEvent, person_detected, person_lost,
    gesture_detected, face_expression, pose_detected, sensor_status,
    face_crop_event, scene_summary, sensor_health,
)
from senses.vision.detector import Detector
from senses.vision.tracker import PersonTracker
from senses.vision.expression import ExpressionAnalyzer
from senses.vision.pose import PoseEstimator
from senses.vision.face_crop import FaceCropExtractor
from senses.vision.scene_aggregator import SceneAggregator
from senses.vision.scene_detector import SceneDetector
from senses.audio.audio_manager import AudioManager

from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

_file_handler = RotatingFileHandler(
    "/tmp/jarvis-senses.log", maxBytes=5 * 1024 * 1024, backupCount=3,
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("jarvis.senses")


class SensesService:
    def __init__(self, config: SensesConfig | None = None):
        self._config = config or SensesConfig()
        self._transport = TransportClient(
            host=self._config.transport.brain_host,
            port=self._config.transport.brain_port,
            sensor_id=self._config.transport.sensor_id,
            reconnect_interval=self._config.transport.reconnect_interval_s,
            buffer_max=self._config.transport.buffer_max_events,
        )
        self._transport.on_brain_message(self._on_brain_message)
        self._transport.on_disconnect(self._on_brain_disconnect)
        self._running = False

        # Vision
        self._detector: Detector | None = None
        self._tracker: PersonTracker | None = None
        self._expression: ExpressionAnalyzer | None = None
        self._pose: PoseEstimator | None = None
        self._face_crop = None
        self._scene_agg = SceneAggregator()
        self._scene_detector: SceneDetector | None = None
        self._last_scene_detect_ts: float = 0.0
        self._scene_detect_interval: float = 3.0
        self._was_person_present = False

        # Audio (mic capture + playback only)
        self._audio = AudioManager(
            mic_sample_rate=self._config.audio.mic_sample_rate,
            mic_name=self._config.audio.mic_name,
            speaker_name=self._config.audio.speaker_name,
        )
        self._downsample_ratio: int = 1

        # Brain audio pipe for seamless playback
        self._brain_pipe: subprocess.Popen | None = None
        self._brain_pipe_lock = threading.Lock()
        self._brain_pipe_sr: int = 0

        # UI state
        self._ui_ws_clients: list = []
        self._ui_outbox: deque = deque(maxlen=50)
        self._current_phase = "IDLE"
        self._last_consciousness_feed: dict = {}
        self._audio_playing = False
        self._playing_conversation_id: str = ""
        self._response_end_t: float = 0.0
        self._first_chunk_t: float = 0.0
        self._chunk_count: int = 0

    async def start(self) -> None:
        logger.info("Starting Jarvis Senses (thin sensor mode)")
        await self._transport.start()

        if self._config.enable_vision:
            self._init_vision()

        if self._config.enable_audio:
            self._init_audio()

        self._running = True
        logger.info("Senses service online — streaming audio to brain")

    async def stop(self) -> None:
        logger.info("Stopping Senses")
        self._running = False
        self._kill_brain_audio_pipe()
        self._audio.shutdown()

        if self._detector:
            self._detector.stop()
        if self._scene_detector:
            self._scene_detector.stop()
        if self._expression:
            self._expression.stop()
        if self._pose:
            self._pose.stop()

        await self._transport.stop()
        logger.info("Senses stopped")

    # --- Adaptive vision FPS ---

    _IDLE_FPS = 8
    _ACTIVE_FPS = 15

    def _adaptive_fps(self) -> float:
        if self._was_person_present:
            return self._ACTIVE_FPS
        return self._IDLE_FPS

    # --- Hardware health telemetry ---

    _HEALTH_INTERVAL_S = 30.0

    @staticmethod
    def _read_cpu_temp() -> float:
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=2,
            )
            return float(result.stdout.strip().replace("temp=", "").replace("'C", ""))
        except Exception:
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    return int(f.read().strip()) / 1000.0
            except Exception:
                return 0.0

    @staticmethod
    def _read_throttled() -> str:
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=2,
            )
            return result.stdout.strip().split("=")[-1]
        except Exception:
            return ""

    @staticmethod
    def _read_cpu_percent() -> float:
        try:
            with open("/proc/stat") as f:
                fields = f.readline().split()
            idle = int(fields[4])
            total = sum(int(x) for x in fields[1:])
            time.sleep(0.1)
            with open("/proc/stat") as f:
                fields2 = f.readline().split()
            idle2 = int(fields2[4])
            total2 = sum(int(x) for x in fields2[1:])
            d_idle = idle2 - idle
            d_total = total2 - total
            if d_total == 0:
                return 0.0
            return (1.0 - d_idle / d_total) * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _read_memory() -> tuple[float, float]:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info: dict[str, int] = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0) / 1024.0
            available = info.get("MemAvailable", 0) / 1024.0
            return total - available, total
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _read_uptime() -> float:
        try:
            with open("/proc/uptime") as f:
                return float(f.read().split()[0])
        except Exception:
            return 0.0

    def _send_health_report(self) -> None:
        cpu_temp = self._read_cpu_temp()
        cpu_pct = self._read_cpu_percent()
        mem_used, mem_total = self._read_memory()
        throttled = self._read_throttled()
        uptime = self._read_uptime()
        cam_fps = self._adaptive_fps() if self._detector else 0.0

        self._transport.send_event(sensor_health(
            cpu_temp_c=cpu_temp,
            cpu_percent=cpu_pct,
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            camera_fps=cam_fps,
            throttled=throttled,
            uptime_s=uptime,
            extra={
                "audio_playing": self._audio_playing,
                "phase": self._current_phase,
            },
        ))

    async def run_loop(self) -> None:
        heartbeat_interval = 30.0
        last_heartbeat = time.time()
        last_health_report = time.time()
        last_stream_check = time.time()
        last_cb_count = 0
        loop = asyncio.get_event_loop()

        while self._running:
            start = time.time()
            fps = self._adaptive_fps() if self._config.enable_vision else 1
            interval = 1.0 / fps

            if self._detector:
                await loop.run_in_executor(None, self._process_vision_frame)

            now = time.time()

            # Mic stream health check
            if now - last_stream_check >= 3.0:
                last_stream_check = now
                if self._audio._restarting:
                    last_cb_count = -1
                else:
                    cur_cb = self._audio._callback_count
                    if not self._audio.is_stream_active or (cur_cb == last_cb_count and last_cb_count > 0):
                        logger.error("Mic stream stalled (callbacks=%d) — restarting", cur_cb)
                        threading.Thread(
                            target=self._audio.restart_shared_stream,
                            daemon=True, name="mic-restart",
                        ).start()
                        last_cb_count = -1
                    else:
                        last_cb_count = cur_cb

            if now - last_health_report >= self._HEALTH_INTERVAL_S:
                last_health_report = now
                try:
                    await loop.run_in_executor(None, self._send_health_report)
                except Exception:
                    logger.debug("Health report send failed")

            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                logger.info(
                    "Heartbeat: phase=%s connected=%s muted=%s fps=%d "
                    "stream_active=%s callbacks=%d overflows=%d",
                    self._current_phase,
                    self._transport.is_connected,
                    self._audio.is_muted,
                    fps,
                    self._audio.is_stream_active,
                    self._audio._callback_count,
                    self._audio._overflow_count,
                )

                if not self._audio.is_stream_active and not self._audio._restarting:
                    logger.error("Mic stream died — restarting")
                    threading.Thread(
                        target=self._audio.restart_shared_stream,
                        daemon=True, name="mic-restart-hb",
                    ).start()

                try:
                    import sounddevice as _sd
                    d = _sd.query_devices(self._audio.mic_device)
                    mic_name = d["name"]
                except Exception:
                    mic_name = f"device={self._audio.mic_device}"

                cam_state = self._detector.get_camera_state() if self._detector else {}
                self._transport.send_event(sensor_status(
                    mic_device=mic_name,
                    mic_muted=self._audio.is_muted,
                    vision_fps=fps,
                    extra={
                        "phase": self._current_phase,
                        "callbacks": self._audio._callback_count,
                        "overflows": self._audio._overflow_count,
                        "camera": cam_state,
                    },
                ))

            elapsed = now - start
            await asyncio.sleep(max(0, interval - elapsed))

    # --- Vision pipeline (unchanged) ---

    def _init_vision(self) -> None:
        vc = self._config.vision
        self._detector = Detector(
            model_path=f"{vc.detection_model}.hef",
            threshold=vc.detection_threshold,
            camera_id=vc.camera_id,
            width=vc.width,
            height=vc.height,
            fps=vc.fps,
        )
        self._tracker = PersonTracker()
        self._detector.start()

        if vc.enable_expressions:
            self._expression = ExpressionAnalyzer(
                model_path=f"{vc.face_model}.hef",
                threshold=vc.face_threshold,
            )
            self._expression.start(shared_vdevice=self._detector.vdevice)

        if self._config.pose.enabled:
            self._pose = PoseEstimator(
                model_path=self._config.pose.model_path,
                threshold=self._config.pose.threshold,
            )
            if self._pose.available:
                self._pose.start(shared_vdevice=self._detector.vdevice)
                logger.info("Pose estimation initialized")

        self._face_crop = FaceCropExtractor(crop_interval_s=2.0)
        if self._face_crop.enabled:
            logger.info("Face crop extractor initialized (interval=2s)")

        models_dir = os.path.join(self._config.project_root, "models")
        scene_model = os.path.join(models_dir, "yolov8n.onnx")
        if os.path.exists(scene_model):
            self._scene_detector = SceneDetector(model_path=scene_model)
            self._scene_detector.start()
            if self._scene_detector.available:
                logger.info("CPU scene detector initialized (YOLOv8n ONNX)")
            else:
                self._scene_detector = None
                logger.warning("CPU scene detector failed to start")
        else:
            logger.info("CPU scene detector: model not found at %s (disabled)", scene_model)

        logger.info("Vision pipeline initialized")

    @staticmethod
    def _estimate_head_boxes(person_dets: list) -> list:
        from senses.vision.detector import Detection
        heads: list[Detection] = []
        for det in person_dets:
            x1, y1, x2, y2 = det.bbox
            person_h = y2 - y1
            if person_h < 30:
                continue
            head_h = max(int(person_h * 0.25), 20)
            head_w = x2 - x1
            pad_x = int(head_w * 0.1)
            heads.append(Detection(
                label="face",
                confidence=det.confidence,
                bbox=(max(0, x1 + pad_x), y1, x2 - pad_x, y1 + head_h),
                timestamp=det.timestamp,
            ))
        return heads

    def _process_vision_frame(self) -> None:
        assert self._detector and self._tracker
        frame = self._detector.capture_frame()
        if frame is None:
            return

        detections = self._detector.detect(frame)
        tracks = self._tracker.update(detections)
        is_present = self._tracker.active_count > 0

        if is_present and not self._was_person_present:
            best = max(tracks, key=lambda t: t.confidence_avg) if tracks else None
            bbox = best.bbox_history[-1] if best and best.bbox_history else None
            self._transport.send_event(person_detected(
                confidence=best.confidence_avg if best else 0.7, bbox=bbox,
            ))
        if not is_present and self._was_person_present:
            self._transport.send_event(person_lost(confidence=0.8))
        self._was_person_present = is_present

        h, w = frame.shape[:2]
        self._scene_agg.set_frame_size(w, h)

        now = time.time()
        if (self._scene_detector and self._scene_detector.available
                and now - self._last_scene_detect_ts >= self._scene_detect_interval):
            self._last_scene_detect_ts = now
            from senses.vision.detector import Detection as HailoDetection
            scene_dets = self._scene_detector.detect(frame)
            for sd in scene_dets:
                detections.append(HailoDetection(
                    label=sd.label, confidence=sd.confidence,
                    bbox=sd.bbox, timestamp=sd.timestamp,
                ))

        non_person = [d for d in detections if d.label != "person"]
        if non_person and (now - getattr(self, "_last_scene_diag", 0) > 30):
            labels = [f"{d.label}:{d.confidence:.2f}" for d in non_person[:8]]
            logger.info("Scene objects (%d): %s", len(non_person), ", ".join(labels))
            self._last_scene_diag = now
        summary = self._scene_agg.feed(detections)
        if summary is not None:
            logger.info("Scene summary emitted: %d objects, change=%.3f",
                        len(summary["detections"]), summary["scene_change_score"])
            self._transport.send_event(scene_summary(
                detections=summary["detections"],
                frame_size=(w, h),
                scene_change_score=summary["scene_change_score"],
            ))

        if self._expression:
            person_dets = [d for d in detections if d.label == "person"]
            head_dets = self._estimate_head_boxes(person_dets[:3])
            expressions = self._expression.analyze(frame, head_dets[:3])
            for expr in expressions:
                if expr.expression != "neutral":
                    self._transport.send_event(face_expression(expr.expression, expr.confidence))

        if self._pose and self._pose.available:
            poses = self._pose.estimate(frame)
            self._tracker.set_pose_gestures(poses)
            for p in poses:
                if p.gesture != "neutral":
                    self._transport.send_event(pose_detected(
                        keypoints=p.keypoints, gesture=p.gesture,
                        confidence=p.confidence,
                    ))

        if self._face_crop and self._face_crop.enabled and is_present:
            import base64
            person_dets = [d for d in detections if d.label == "person"]
            track_ids = [t.id for t in tracks[:len(person_dets)]] if tracks else None
            face_crops = self._face_crop.extract(frame, person_dets[:3], track_ids)
            for fc in face_crops:
                crop_bytes = FaceCropExtractor.crop_to_bytes(fc.crop)
                crop_b64 = base64.b64encode(crop_bytes).decode("ascii")
                self._transport.send_event(face_crop_event(
                    crop_b64=crop_b64, track_id=fc.track_id, confidence=fc.confidence,
                ))

        for track in tracks:
            if track.gesture and not track.pose_gesture:
                self._transport.send_event(gesture_detected(track.gesture, confidence=0.6))

    # --- Audio: thin streaming to brain ---

    _TARGET_RATE = 16000

    def _init_audio(self) -> None:
        self._audio.start_shared_stream()
        actual_rate = self._audio.mic_sample_rate

        if actual_rate == self._TARGET_RATE:
            self._resample = False
            logger.info("Audio streaming: %dHz (native, no resampling)", actual_rate)
        elif actual_rate % self._TARGET_RATE == 0:
            self._resample = False
            self._downsample_ratio = actual_rate // self._TARGET_RATE
            logger.info("Audio streaming: %dHz -> %dHz (integer ratio=%d)",
                        actual_rate, self._TARGET_RATE, self._downsample_ratio)
        else:
            self._resample = True
            self._resample_ratio = self._TARGET_RATE / actual_rate
            logger.info("Audio streaming: %dHz -> %dHz (interpolated, ratio=%.4f)",
                        actual_rate, self._TARGET_RATE, self._resample_ratio)

        src_rate = actual_rate
        tgt_rate = self._TARGET_RATE

        def _stream_to_brain(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            if src_rate == tgt_rate:
                pass
            elif src_rate % tgt_rate == 0:
                ratio = src_rate // tgt_rate
                n = len(mono) // ratio * ratio
                mono = mono[:n].reshape(-1, ratio).mean(axis=1)
            else:
                out_len = int(len(mono) * tgt_rate / src_rate)
                if out_len > 0:
                    mono = np.interp(
                        np.linspace(0, len(mono) - 1, out_len),
                        np.arange(len(mono)),
                        mono,
                    )
            pcm = (mono * 32767).astype(np.int16).tobytes()
            self._transport.send_audio(pcm)

        self._audio.set_stream_callback(_stream_to_brain)
        self._audio.set_speaking_callback(self._on_speaking_state)

    def _on_speaking_state(self, is_speaking: bool) -> None:
        """Notify brain about playback state so it can adjust wake word sensitivity."""
        pass

    # --- Brain message handling ---

    def _on_brain_message(self, msg: BrainMessage) -> None:
        _t0 = time.monotonic()
        _arrival = getattr(msg, "_ws_arrival", 0.0)
        if _arrival:
            _queue_ms = (_t0 - _arrival) * 1000
            if _queue_ms > 50:
                logger.info("[PI-LATENCY] audio_queue_wait=%.0fms type=%s", _queue_ms, msg.type)

        if msg.type == "response_chunk":
            brain_audio = msg.data.get("audio_b64", "")
            conv_tag = (msg.conversation_id or self._playing_conversation_id)[:8]
            if msg.conversation_id:
                self._playing_conversation_id = msg.conversation_id
            if brain_audio:
                self._play_brain_audio(brain_audio, _recv_t=_t0)
            else:
                logger.info("[PI-LATENCY] chunk_received (no audio) conv=%s", conv_tag)
            self._set_phase("SPEAKING")

        elif msg.type == "response_end":
            conv_id = msg.conversation_id or self._playing_conversation_id
            conv_tag = conv_id[:8] if conv_id else "?"
            self._response_end_t = time.monotonic()
            logger.info("[PI-LATENCY] response_end_received conv=%s", conv_tag)
            if msg.conversation_id:
                self._playing_conversation_id = msg.conversation_id
            threading.Thread(
                target=self._brain_audio_done_monitor, args=(conv_id,), daemon=True,
            ).start()

        elif msg.type == "response":
            logger.info("Brain response: %s", msg.text[:80])
            brain_audio = msg.data.get("audio_b64", "")
            conv_id = msg.conversation_id or self._playing_conversation_id
            if msg.conversation_id:
                self._playing_conversation_id = msg.conversation_id
            if brain_audio:
                self._play_brain_audio(brain_audio)
                self._set_phase("SPEAKING")
                threading.Thread(
                    target=self._brain_audio_done_monitor, args=(conv_id,), daemon=True,
                ).start()

        elif msg.type == "state_update":
            if msg.phase and not self._audio_playing:
                self._set_phase(msg.phase)

        elif msg.type == "command":
            action = msg.data.get("action", "")
            if action == "wake_detected":
                self._set_phase(msg.phase or "LISTENING")
                logger.info("Brain detected wake word — phase=%s", self._current_phase)
            elif action == "speaking":
                is_speaking = msg.data.get("speaking", True)
                if is_speaking:
                    self._set_phase("SPEAKING")
                else:
                    self._set_phase("IDLE")
            elif action == "speak" and msg.text:
                brain_audio = msg.data.get("audio_b64", "")
                if msg.conversation_id:
                    self._playing_conversation_id = msg.conversation_id
                if brain_audio:
                    self._play_brain_audio(brain_audio)
            elif action == "thinking":
                self._set_phase("PROCESSING")
                logger.info("Brain processing conv %s",
                            msg.conversation_id[:8] if msg.conversation_id else "?")
            elif action == "stt_failed":
                logger.info("Brain STT returned empty")
                self._set_phase("IDLE")
            elif action == "phase_update":
                new_phase = msg.data.get("phase", "LISTENING")
                if not self._audio_playing:
                    logger.info("Brain phase update → %s", new_phase)
                    self._set_phase(new_phase)
            elif action == "barge_in":
                logger.info("Brain detected barge-in — cancelling playback")
                self._kill_brain_audio_pipe()
                self._audio.cancel_playback()
                self._set_phase("LISTENING")
            elif action == "camera_control":
                self._handle_camera_control(msg.data)

        elif msg.type == "consciousness_feed":
            self._last_consciousness_feed = msg.data
            feed = {"type": "consciousness", **msg.data}
            if self._audio_playing:
                feed["phase"] = "SPEAKING"
            self._broadcast_to_ui(feed)

    # --- Camera control from brain ---

    def _handle_camera_control(self, data: dict) -> None:
        if not self._detector:
            logger.warning("Camera control received but no detector available")
            return
        ctrl = data.get("control", "")
        if ctrl == "zoom":
            level = float(data.get("level", 1.0))
            self._detector.set_zoom(level)
        elif ctrl == "zoom_to":
            region = data.get("region")
            if region and len(region) == 4:
                padding = float(data.get("padding", 1.5))
                self._detector.zoom_to_region(*region, padding=padding)
        elif ctrl == "reset":
            self._detector.reset_zoom()
        elif ctrl == "autofocus":
            self._detector.trigger_autofocus()
        elif ctrl == "continuous_af":
            self._detector.set_continuous_autofocus()
        elif ctrl == "manual_focus":
            position = float(data.get("position", 0.0))
            self._detector.set_manual_focus(position)
        else:
            logger.warning("Unknown camera control: %s", ctrl)

        state = self._detector.get_camera_state()
        self._transport.send_event(PerceptionEvent(
            source="system", type="camera_state", data=state,
        ))

    # --- Brain audio playback ---

    def _play_brain_audio(self, audio_b64: str, *, _recv_t: float = 0.0) -> None:
        import io
        import wave as _wave
        _t0 = _recv_t or time.monotonic()
        try:
            wav_data = base64.b64decode(audio_b64)
            _t_decode = time.monotonic()

            with _wave.open(io.BytesIO(wav_data), "rb") as wf:
                sr = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
            _t_parse = time.monotonic()

            opened_new = False
            with self._brain_pipe_lock:
                if self._brain_pipe is None or self._brain_pipe.poll() is not None:
                    self._brain_pipe_sr = sr
                    self._audio._mute()
                    self._brain_pipe = subprocess.Popen(
                        ["aplay", "-D", self._audio.speaker_alsa,
                         "-t", "raw", "-f", "S16_LE", "-r", str(sr), "-c", "1", "-"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    import fcntl
                    try:
                        fcntl.fcntl(self._brain_pipe.stdin.fileno(), 1031, 1048576)
                    except OSError:
                        pass
                    opened_new = True
                    self._first_chunk_t = _t0
                    self._chunk_count = 0
                pipe = self._brain_pipe

            _t_pipe_ready = time.monotonic()
            wrote_ok = False
            if pipe and pipe.stdin and pipe.poll() is None:
                try:
                    pipe.stdin.write(pcm)
                    pipe.stdin.flush()
                    wrote_ok = True
                except (BrokenPipeError, ValueError, OSError):
                    logger.debug("Brain audio pipe closed (barge-in or playback ended)")
                    return

            _t_done = time.monotonic()
            self._chunk_count += 1
            self._audio_playing = True
            self._set_phase("SPEAKING")

            conv_tag = self._playing_conversation_id[:8] if self._playing_conversation_id else "?"
            decode_ms = (_t_decode - _t0) * 1000
            parse_ms = (_t_parse - _t_decode) * 1000
            write_ms = (_t_done - _t_pipe_ready) * 1000
            total_ms = (_t_done - _t0) * 1000
            logger.info(
                "[PI-LATENCY] chunk_%d_piped=%.0fms decode=%.0fms parse=%.0fms write=%.0fms "
                "pcm=%dKB new_pipe=%s conv=%s",
                self._chunk_count, total_ms, decode_ms, parse_ms, write_ms,
                len(pcm) // 1024, opened_new, conv_tag,
            )
        except Exception as exc:
            logger.error("Brain audio pipe failed: %s", exc)

    def _is_playing_audio(self) -> bool:
        """Check if brain audio is actively playing through speakers."""
        return self._audio_playing

    def _close_brain_audio_pipe(self) -> None:
        with self._brain_pipe_lock:
            pipe = self._brain_pipe
            self._brain_pipe = None
        if pipe:
            try:
                if pipe.stdin:
                    pipe.stdin.close()
                pipe.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pipe.kill()
            except Exception:
                pass
        self._audio._unmute()

    def _kill_brain_audio_pipe(self) -> None:
        with self._brain_pipe_lock:
            pipe = self._brain_pipe
            self._brain_pipe = None
        if pipe:
            try:
                if pipe.stdin:
                    pipe.stdin.close()
                pipe.kill()
            except Exception:
                pass
        self._audio_playing = False
        self._audio._unmute()

    def _brain_audio_done_monitor(self, conv_id: str = "") -> None:
        _t0 = time.monotonic()
        self._close_brain_audio_pipe()
        _drain_ms = (time.monotonic() - _t0) * 1000
        self._audio_playing = False
        conv_tag = conv_id[:8] if conv_id else "?"
        total_chunks = self._chunk_count
        total_playback_ms = (time.monotonic() - self._first_chunk_t) * 1000 if self._first_chunk_t else 0
        logger.info(
            "[PI-LATENCY] playback_done drain=%.0fms total=%.0fms chunks=%d conv=%s",
            _drain_ms, total_playback_ms, total_chunks, conv_tag,
        )
        self._set_phase("IDLE")
        self._chunk_count = 0
        self._first_chunk_t = 0.0
        try:
            from transport.event_schema import PerceptionEvent
            self._transport.send_event(PerceptionEvent(
                source="system",
                type="playback_complete",
                conversation_id=conv_id,
            ))
        except Exception:
            logger.exception("Failed to send playback_complete")

    # --- Recovery ---

    def _force_audio_recovery(self) -> None:
        logger.warning("Audio recovery: killing playback, resetting state")
        self._kill_brain_audio_pipe()
        self._audio.cancel_playback()
        self._audio_playing = False
        self._set_phase("IDLE")

    def _on_brain_disconnect(self) -> None:
        """Called from transport when WebSocket drops. Prevents stale audio state."""
        if self._audio_playing:
            logger.warning("Brain disconnected while playing audio — running recovery")
            self._force_audio_recovery()
        else:
            logger.info("Brain disconnected (no active playback)")

    # --- UI state ---

    def _set_phase(self, phase: str) -> None:
        self._current_phase = phase
        logger.debug("Phase: %s", phase)

    def _broadcast_to_ui(self, msg: dict) -> None:
        """Send a JSON message to all connected UI WebSocket clients (thread-safe)."""
        self._ui_outbox.append(msg)

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def last_frame(self):
        if self._detector:
            return self._detector.last_frame
        return None


# --- UI Server (serves particle display + provides state via WebSocket) ---

async def start_ui_server(service: SensesService, host: str, port: int):
    from aiohttp import web
    import aiohttp

    static_dir = os.path.join(os.path.dirname(__file__), "ui", "static")
    ws_clients: list[web.WebSocketResponse] = []

    async def index_handler(request):
        return web.FileResponse(os.path.join(static_dir, "index.html"))

    service._ui_ws_clients = ws_clients

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.append(ws)
        init_phase = "SPEAKING" if service._audio_playing else service.current_phase
        await ws.send_json({"type": "state", "phase": init_phase})
        if service._last_consciousness_feed:
            await ws.send_json({"type": "consciousness", **service._last_consciousness_feed})
        try:
            async for msg in ws:
                pass
        finally:
            ws_clients.remove(ws)
        return ws

    async def mjpeg_handler(request):
        import cv2
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
            },
        )
        await resp.prepare(request)
        try:
            while True:
                frame = service.last_frame
                if frame is not None:
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    await resp.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )
                await asyncio.sleep(0.1)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def snapshot_handler(request):
        import cv2
        frame = service.last_frame
        if frame is None:
            return web.Response(status=503, text="No camera frame available")
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return web.Response(body=jpeg.tobytes(), content_type="image/jpeg")

    async def broadcast_state():
        last_phase = ""
        while True:
            phase = service.current_phase
            if service._audio_playing:
                phase = "SPEAKING"
            if phase != last_phase:
                last_phase = phase
                for ws in list(ws_clients):
                    try:
                        await ws.send_json({"type": "state", "phase": phase})
                    except Exception:
                        pass
            while service._ui_outbox:
                try:
                    msg = service._ui_outbox.popleft()
                    for ws in list(ws_clients):
                        try:
                            await ws.send_json(msg)
                        except Exception:
                            pass
                except IndexError:
                    break
            await asyncio.sleep(0.1)

    # --- Debug log page ---

    log_buffer: deque[str] = deque(maxlen=500)
    sse_clients: list[asyncio.Queue] = []

    class _SSELogHandler(logging.Handler):
        def emit(self, record):
            try:
                line = self.format(record)
                log_buffer.append(line)
                for q in list(sse_clients):
                    try:
                        q.put_nowait(line)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                pass

    _handler = _SSELogHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(_handler)

    async def debug_page_handler(request):
        html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Jarvis Debug</title>
<style>
body{background:#0a0a0a;color:#0f0;font:12px/1.4 'Fira Mono',monospace;margin:0;padding:8px}
#log{white-space:pre-wrap;word-break:break-all}
.warn{color:#ff0}.err{color:#f44}.info{color:#0f0}.debug{color:#888}
h1{color:#0ff;font-size:16px;margin:0 0 8px}
#status{color:#0ff;font-size:11px;margin-bottom:4px}
</style></head><body>
<h1>JARVIS SENSES — Live Debug Log</h1>
<div id="status">Connecting...</div>
<div id="log"></div>
<script>
const log=document.getElementById('log'),status=document.getElementById('status');
let lines=0,maxLines=800;
function classify(t){
  if(t.includes(' ERROR:'))return 'err';
  if(t.includes(' WARNING:'))return 'warn';
  if(t.includes(' DEBUG:'))return 'debug';
  return 'info';
}
function addLine(t){
  const d=document.createElement('div');
  d.className=classify(t);d.textContent=t;
  log.appendChild(d);lines++;
  if(lines>maxLines){log.removeChild(log.firstChild);lines--;}
  window.scrollTo(0,document.body.scrollHeight);
}
function connect(){
  status.textContent='Connecting...';
  const es=new EventSource('/debug/stream');
  es.onopen=()=>{status.textContent='Connected — streaming live';};
  es.onmessage=e=>{addLine(e.data);};
  es.onerror=()=>{status.textContent='Disconnected — reconnecting...';es.close();setTimeout(connect,2000);};
}
fetch('/debug/history').then(r=>r.json()).then(lines=>{lines.forEach(addLine);connect();});
</script></body></html>"""
        return web.Response(text=html, content_type="text/html")

    async def debug_history_handler(request):
        return web.json_response(list(log_buffer))

    async def debug_stream_handler(request):
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"},
        )
        await resp.prepare(request)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        sse_clients.append(q)
        try:
            while True:
                line = await q.get()
                await resp.write(f"data: {line}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            sse_clients.remove(q)
        return resp

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/video", mjpeg_handler)
    app.router.add_get("/snapshot", snapshot_handler)
    app.router.add_get("/debug", debug_page_handler)
    app.router.add_get("/debug/history", debug_history_handler)
    app.router.add_get("/debug/stream", debug_stream_handler)
    app.router.add_static("/static/", static_dir)

    # P3.14 (2026-04-25): the standalone /mind kiosk view was deleted as
    # redundant. The same mental-world signals now ride the existing
    # consciousness feed (built by perception_orchestrator._build_scene_block)
    # and modulate the live particle visualizer at pi/ui/static/particles.js.
    # Operator inspection of the spatial scene graph stays on the brain
    # dashboard at /hrr-scene; the Pi LCD shows JARVIS, not metrics.

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port, reuse_address=True)
    await site.start()
    logger.info("Particle UI: http://%s:%d", host, port)

    asyncio.create_task(broadcast_state())
    return runner


def launch_kiosk_browser(port: int) -> subprocess.Popen | None:
    env = os.environ.copy()
    uid = os.getuid()
    runtime_dir = env.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")

    wayland_sock = None
    x_display = env.get("DISPLAY")

    if not env.get("WAYLAND_DISPLAY"):
        probe = os.path.join(runtime_dir, "wayland-0")
        if os.path.exists(probe):
            wayland_sock = "wayland-0"
            env["WAYLAND_DISPLAY"] = wayland_sock
            env.setdefault("XDG_RUNTIME_DIR", runtime_dir)

    if not wayland_sock and not x_display:
        logger.info("No display detected — skipping kiosk browser")
        return None

    for browser in ("chromium", "chromium-browser"):
        if not shutil.which(browser):
            continue

        url = f"http://localhost:{port}"
        cmd = [
            browser,
            "--kiosk", "--noerrdialogs", "--disable-infobars",
            "--disable-session-crashed-bubble", "--no-first-run",
            "--disable-translate", "--disable-features=TranslateUI",
            "--check-for-update-interval=31536000", "--incognito",
            "--password-store=basic", "--enable-gpu-rasterization",
            "--ignore-gpu-blocklist", "--enable-features=VaapiVideoDecoder",
            "--disable-software-rasterizer", "--use-gl=egl",
        ]
        if wayland_sock:
            cmd.append("--ozone-platform=wayland")
        cmd.append(url)

        logger.info("Launching kiosk browser: %s → %s", browser, url)
        try:
            return subprocess.Popen(cmd, env=env,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as exc:
            logger.error("Failed to launch browser: %s", exc)
            return None

    logger.warning("No Chromium browser found — open http://localhost:%d manually", port)
    return None


# --- Entry point ---

_shutting_down = False


async def shutdown(service: SensesService) -> None:
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    logger.info("Shutdown signal received")
    await service.stop()
    for t in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        t.cancel()


async def main() -> None:
    config = SensesConfig()
    service = SensesService(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(service)))

    await service.start()

    ui_runner = None
    browser_proc = None
    if config.ui.enabled:
        ui_runner = await start_ui_server(service, config.ui.host, config.ui.port)
        browser_proc = launch_kiosk_browser(config.ui.port)

    try:
        await service.run_loop()
    except asyncio.CancelledError:
        pass
    finally:
        if browser_proc:
            browser_proc.terminate()
        if not _shutting_down:
            await service.stop()
        if ui_runner:
            await ui_runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
