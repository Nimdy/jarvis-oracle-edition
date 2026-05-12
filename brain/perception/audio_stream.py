"""Continuous audio stream processor — wake word detection + VAD + speech segmentation.

Receives raw 16kHz int16 PCM audio from Pi sensors via binary WebSocket frames.
Runs openWakeWord continuously for wake word detection. When triggered, accumulates
audio and uses Silero VAD to detect speech end, then dispatches for STT.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from enum import Enum, auto
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from openwakeword.model import Model as OWWModel
    HAS_OWW = True
except ImportError:
    HAS_OWW = False
    logger.warning("openwakeword not available — wake word detection disabled")

try:
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    logger.warning("faster_whisper.vad not available — speech end detection disabled")


class StreamState(Enum):
    IDLE = auto()
    LISTENING = auto()
    FOLLOW_UP = auto()


class AudioStreamProcessor:
    """Processes continuous 16kHz int16 PCM from the Pi sensor node.

    Runs wake word detection on every chunk. On trigger, accumulates speech
    and uses Silero VAD for endpoint detection before dispatching to STT.
    """

    def __init__(
        self,
        keyword: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        speaking_threshold_mult: float = 2.0,
        speaking_hits_required: int = 3,
        cooldown_s: float = 2.0,
        silence_duration_s: float = 1.5,
        max_record_s: float = 15.0,
        follow_up_timeout_s: float = 4.0,
        on_wake: Callable | None = None,
        on_speech_ready: Callable[[np.ndarray, str], None] | None = None,
        on_barge_in: Callable[[str], None] | None = None,
    ) -> None:
        self._keyword = keyword.lower()
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._speaking_threshold_mult = speaking_threshold_mult
        self._speaking_hits_required = speaking_hits_required
        self._cooldown_s = cooldown_s
        self._silence_duration_s = silence_duration_s
        self._max_record_s = max_record_s
        self._follow_up_timeout_s = follow_up_timeout_s

        self._on_wake = on_wake
        self._on_speech_ready = on_speech_ready
        self._on_barge_in = on_barge_in

        self._state = StreamState.IDLE
        self._state_lock = threading.Lock()

        self._speaking = False
        self._speaking_hit_count = 0
        self._speaking_first_hit_time = 0.0
        self._speaking_hit_window = 0.8
        self._wake_armed = True

        self._last_detection_time = 0.0
        self._conversation_id: str = ""
        self._prev_conversation_id: str = ""
        self._follow_up_start: float = 0.0
        self._is_follow_up_session: bool = False

        self._speech_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._speech_start_time: float = 0.0
        self._last_vad_check: float = 0.0
        self._vad_check_interval = 0.5

        self._oww_model: OWWModel | None = None
        self._oww_retry_after: float = 0.0
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=2000)
        self._running = False
        self._synthetic_active: bool = False
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        if HAS_OWW and self._oww_model is None:
            try:
                self._oww_model = OWWModel(
                    wakeword_models=[self._keyword],
                    inference_framework="onnx",
                )
                logger.info("openWakeWord model loaded: %s", self._keyword)
            except Exception as exc:
                logger.error("Failed to load openWakeWord model: %s", exc)

        self._running = True
        self._worker = threading.Thread(
            target=self._run_worker, daemon=True, name="audio-stream-worker",
        )
        self._worker.start()
        logger.info(
            "AudioStreamProcessor started (keyword=%s threshold=%.2f)",
            self._keyword, self._threshold,
        )

    def stop(self) -> None:
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
            if self._worker.is_alive():
                logger.warning("Audio stream worker didn't exit cleanly")
        self._worker = None
        logger.info("AudioStreamProcessor stopped")

    def feed(self, pcm_bytes: bytes) -> None:
        """Push raw 16kHz int16 PCM bytes from WebSocket. Thread-safe."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    @property
    def synthetic_active(self) -> bool:
        return self._synthetic_active

    @synthetic_active.setter
    def synthetic_active(self, value: bool) -> None:
        self._synthetic_active = value

    def set_speaking(self, speaking: bool) -> None:
        with self._state_lock:
            self._speaking = speaking
            if not speaking:
                self._speaking_hit_count = 0

    def set_wake_armed(self, armed: bool) -> None:
        """Arm/disarm wake word detection without disrupting the audio stream."""
        with self._state_lock:
            self._wake_armed = armed
        logger.info("Wake word detection %s", "armed" if armed else "disarmed")

    @property
    def is_speaking(self) -> bool:
        """Thread-safe read of the speaking flag."""
        with self._state_lock:
            return self._speaking

    def set_follow_up(self) -> None:
        with self._state_lock:
            self._state = StreamState.FOLLOW_UP
            self._follow_up_start = time.monotonic()
            with self._buffer_lock:
                self._speech_buffer.clear()
            self._is_follow_up_session = False
        logger.debug("Entered FOLLOW_UP mode")

    @property
    def was_follow_up(self) -> bool:
        """Whether the current listening session originated from follow-up speech."""
        return getattr(self, "_is_follow_up_session", False)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run_worker(self) -> None:
        consecutive_errors = 0

        while self._running:
            try:
                raw = self._queue.get(timeout=0.2)
            except queue.Empty:
                self._check_timeouts()
                continue
            if raw is None:
                break

            try:
                audio_i16 = np.frombuffer(raw, dtype=np.int16)
                if audio_i16.size == 0:
                    continue

                try:
                    from perception.diarization_collector import diarization_collector
                    diarization_collector.feed_audio(raw)
                except Exception:
                    pass

                with self._state_lock:
                    state = self._state

                if state == StreamState.IDLE:
                    self._process_wake_word(audio_i16)
                elif state == StreamState.LISTENING:
                    self._process_listening(audio_i16)
                elif state == StreamState.FOLLOW_UP:
                    self._process_follow_up(audio_i16)

                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    logger.error("Audio stream worker error (%d): %s", consecutive_errors, exc)
                if consecutive_errors >= 10:
                    logger.error("Audio stream worker: too many errors, exiting")
                    break

        logger.info("Audio stream worker exiting (running=%s)", self._running)

    def _check_timeouts(self) -> None:
        with self._state_lock:
            state = self._state

        now = time.monotonic()
        if state == StreamState.LISTENING:
            elapsed = now - self._speech_start_time
            if elapsed >= self._max_record_s + 5.0:
                logger.warning(
                    "LISTENING timeout (no audio chunks for %.1fs) — forcing finish",
                    elapsed,
                )
                self._finish_listening()
        elif state == StreamState.FOLLOW_UP:
            if now - self._follow_up_start > self._follow_up_timeout_s:
                with self._state_lock:
                    with self._buffer_lock:
                        self._speech_buffer.clear()
                    self._state = StreamState.IDLE
                    self._speaking = False
                    self._speaking_hit_count = 0
                self._flush_oww_model()
                logger.debug("Follow-up timeout — returning to IDLE")

    # ------------------------------------------------------------------
    # Wake word detection
    # ------------------------------------------------------------------

    _ww_diag_time: float = 0.0
    _ww_diag_count: int = 0
    _ww_diag_max_score: float = 0.0

    _OWW_RETRY_INTERVAL_S = 10.0

    def _try_load_oww_model(self) -> bool:
        """Attempt to load the openWakeWord model. Returns True on success."""
        if not HAS_OWW:
            return False
        now = time.monotonic()
        if now < self._oww_retry_after:
            return False
        try:
            self._oww_model = OWWModel(
                wakeword_models=[self._keyword],
                inference_framework="onnx",
            )
            logger.info("openWakeWord model loaded (deferred): %s", self._keyword)
            return True
        except Exception:
            self._oww_retry_after = now + self._OWW_RETRY_INTERVAL_S
            return False

    def _process_wake_word(self, audio_i16: np.ndarray) -> None:
        if not self._oww_model:
            if not self._try_load_oww_model():
                return

        if not self._wake_armed:
            return

        now = time.monotonic()
        speaking = self.is_speaking
        if not speaking and now - self._last_detection_time < self._cooldown_s:
            return

        self._oww_model.predict(audio_i16)

        best_score = 0.0
        for scores in self._oww_model.prediction_buffer.values():
            if scores:
                s = scores[-1]
                if s > best_score:
                    best_score = s

        effective_threshold = self._threshold
        if speaking:
            effective_threshold = self._threshold * self._speaking_threshold_mult

        self._ww_diag_count += 1
        if best_score > self._ww_diag_max_score:
            self._ww_diag_max_score = best_score
        if now - self._ww_diag_time >= 15.0:
            logger.info(
                "Wake listen: chunks=%d max_score=%.3f threshold=%.2f speaking=%s",
                self._ww_diag_count, self._ww_diag_max_score,
                effective_threshold, speaking,
            )
            self._ww_diag_time = now
            self._ww_diag_count = 0
            self._ww_diag_max_score = 0.0

        if best_score < effective_threshold:
            return

        if speaking:
            if not self._try_speaking_hit(now):
                return
            logger.info(
                "Wake word (barge-in): score=%.3f [%d consecutive hits]",
                best_score, self._speaking_hits_required,
            )
        else:
            logger.info("Wake word triggered: score=%.3f", best_score)

        self._last_detection_time = now
        self._flush_oww_model()

        conv_id = str(uuid.uuid4())
        with self._state_lock:
            self._prev_conversation_id = self._conversation_id
            self._conversation_id = conv_id
            self._state = StreamState.LISTENING
            with self._buffer_lock:
                self._speech_buffer.clear()
            self._speech_start_time = now
            self._last_vad_check = now
            self._is_follow_up_session = False

        if self._on_wake:
            try:
                self._on_wake()
            except Exception as exc:
                logger.error("on_wake callback error: %s", exc)

    def _try_speaking_hit(self, now: float) -> bool:
        """Track consecutive wake hits during playback. Returns True when threshold met."""
        with self._state_lock:
            if self._speaking_hit_count == 0:
                self._speaking_first_hit_time = now
                self._speaking_hit_count = 1
                return False

            if now - self._speaking_first_hit_time > self._speaking_hit_window:
                self._speaking_hit_count = 1
                self._speaking_first_hit_time = now
                return False

            self._speaking_hit_count += 1
            if self._speaking_hit_count < self._speaking_hits_required:
                return False

            self._speaking_hit_count = 0
            return True

    def _flush_oww_model(self) -> None:
        if not self._oww_model:
            return
        silence = np.zeros(32000, dtype=np.int16)
        try:
            self._oww_model.predict(silence)
            if hasattr(self._oww_model, "prediction_buffer"):
                for key in self._oww_model.prediction_buffer:
                    self._oww_model.prediction_buffer[key] = []
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Speech accumulation (LISTENING state)
    # ------------------------------------------------------------------

    def _process_listening(self, audio_i16: np.ndarray) -> None:
        with self._buffer_lock:
            self._speech_buffer.append(audio_i16)

        now = time.monotonic()
        elapsed = now - self._speech_start_time

        if elapsed >= self._max_record_s:
            logger.info("Max recording time reached (%.1fs)", elapsed)
            self._finish_listening()
            return

        # Barge-in: wake word during LISTENING means user interrupted
        if self.is_speaking and self._oww_model:
            self._oww_model.predict(audio_i16)
            best_score = 0.0
            for scores in self._oww_model.prediction_buffer.values():
                if scores:
                    s = scores[-1]
                    if s > best_score:
                        best_score = s
            if best_score >= self._threshold * self._speaking_threshold_mult:
                logger.info("Barge-in detected during LISTENING")
                self._flush_oww_model()
                conv_id = self._conversation_id
                with self._state_lock:
                    with self._buffer_lock:
                        self._speech_buffer.clear()
                    self._state = StreamState.IDLE
                if self._on_barge_in:
                    try:
                        self._on_barge_in(conv_id)
                    except Exception as exc:
                        logger.error("on_barge_in callback error: %s", exc)
                return

        if now - self._last_vad_check >= self._vad_check_interval:
            self._last_vad_check = now
            with self._buffer_lock:
                buf_snapshot = list(self._speech_buffer)
            if buf_snapshot:
                combined = np.concatenate(buf_snapshot)
                if self._check_speech_end(combined):
                    logger.info("Speech end detected (%.1fs)", elapsed)
                    self._finish_listening()

    def _finish_listening(self) -> None:
        with self._buffer_lock:
            if not self._speech_buffer:
                with self._state_lock:
                    self._state = StreamState.IDLE
                self._flush_oww_model()
                return
            combined = np.concatenate(self._speech_buffer)

        conv_id = self._conversation_id

        with self._state_lock:
            with self._buffer_lock:
                self._speech_buffer.clear()
            self._state = StreamState.IDLE

        self._flush_oww_model()
        self._dispatch_speech(combined, conv_id)

    # ------------------------------------------------------------------
    # Silero VAD speech-end detection
    # ------------------------------------------------------------------

    def _check_speech_end(self, buffer: np.ndarray) -> bool:
        if not HAS_VAD:
            return False

        total_samples = len(buffer)
        min_samples = int(self._silence_duration_s * self._sample_rate)
        if total_samples < min_samples:
            return False

        tail_samples = int((self._silence_duration_s + 0.5) * self._sample_rate)
        tail = buffer[-tail_samples:]

        audio_f32 = tail.astype(np.float32) / 32768.0

        try:
            opts = VadOptions(
                threshold=0.3,
                min_silence_duration_ms=int(self._silence_duration_s * 1000),
            )
            timestamps = get_speech_timestamps(audio_f32, opts)
        except Exception as exc:
            logger.debug("VAD check failed: %s", exc)
            return False

        if not timestamps:
            return True

        last_end = max(ts["end"] for ts in timestamps)
        silence_since_last_speech = (len(tail) - last_end) / self._sample_rate
        return silence_since_last_speech >= self._silence_duration_s

    # ------------------------------------------------------------------
    # Follow-up mode
    # ------------------------------------------------------------------

    def _process_follow_up(self, audio_i16: np.ndarray) -> None:
        now = time.monotonic()

        if now - self._follow_up_start > self._follow_up_timeout_s:
            with self._state_lock:
                with self._buffer_lock:
                    self._speech_buffer.clear()
                self._state = StreamState.IDLE
            logger.debug("Follow-up timeout — returning to IDLE")
            return

        # Also run wake word so we don't miss explicit triggers
        if self._oww_model:
            self._process_wake_word(audio_i16)
            with self._state_lock:
                if self._state == StreamState.LISTENING:
                    return

        with self._buffer_lock:
            self._speech_buffer.append(audio_i16)

            # Check for speech activity with a small window
            if len(self._speech_buffer) >= int(0.3 * self._sample_rate / max(len(audio_i16), 1)):
                buf_snapshot = list(self._speech_buffer)
            else:
                buf_snapshot = None

        if buf_snapshot is not None:
            combined = np.concatenate(buf_snapshot)
            audio_f32 = combined.astype(np.float32) / 32768.0

            if HAS_VAD:
                try:
                    opts = VadOptions(threshold=0.3, min_silence_duration_ms=300)
                    timestamps = get_speech_timestamps(audio_f32, opts)
                    if timestamps:
                        with self._state_lock:
                            prev_id = self._conversation_id or self._prev_conversation_id
                        conv_id = prev_id if prev_id else str(uuid.uuid4())
                        logger.info("Follow-up speech detected — transitioning to LISTENING (reusing conv=%s)",
                                    conv_id[:8] if conv_id else "?")
                        with self._state_lock:
                            self._conversation_id = conv_id
                            self._state = StreamState.LISTENING
                            self._speech_start_time = time.monotonic()
                            self._last_vad_check = time.monotonic()
                            self._is_follow_up_session = True
                        if self._on_wake:
                            try:
                                self._on_wake()
                            except Exception as exc:
                                logger.error("on_wake callback error: %s", exc)
                        return
                except Exception as exc:
                    logger.debug("Follow-up VAD check failed: %s", exc)

            # Keep buffer bounded — slide window
            max_follow_up_samples = int(self._follow_up_timeout_s * self._sample_rate)
            with self._buffer_lock:
                total = sum(len(c) for c in self._speech_buffer)
                while total > max_follow_up_samples and self._speech_buffer:
                    removed = self._speech_buffer.pop(0)
                    total -= len(removed)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_audio_features(audio_f32: np.ndarray, sr: int = 16000) -> dict:
        """Compute cheap acoustic features from float32 audio (~1ms with numpy)."""
        n = len(audio_f32)
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio_f32)))) / max(1, 2 * n))
        fft_mag = np.abs(np.fft.rfft(audio_f32))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mag_sum = float(np.sum(fft_mag)) + 1e-10
        spectral_centroid = float(np.sum(freqs * fft_mag) / mag_sum)
        spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / mag_sum))
        corr = np.correlate(audio_f32[: sr // 4], audio_f32[: sr // 4], mode="full")
        corr = corr[len(corr) // 2 :]
        d = np.diff(corr)
        start = int(np.argmax(d > 0))
        peak = start + int(np.argmax(corr[start:])) if start < len(corr) - 1 else 0
        pitch_hz = float(sr / peak) if peak > 0 else 0.0
        duration_s = n / sr
        return {
            "rms": round(rms, 6),
            "zcr": round(zcr, 6),
            "spectral_centroid": round(spectral_centroid, 2),
            "spectral_spread": round(spectral_spread, 2),
            "pitch_hz": round(pitch_hz, 2),
            "duration_s": round(duration_s, 3),
            "speaking_rate": 0.0,
        }

    def _dispatch_speech(self, buffer: np.ndarray, conversation_id: str) -> None:
        audio_f32 = buffer.astype(np.float32) / 32768.0

        duration = len(audio_f32) / self._sample_rate
        if duration < 0.2:
            logger.debug("Discarding very short utterance (%.2fs)", duration)
            return

        logger.info("Dispatching %.1fs speech (conv=%s)", duration, conversation_id[:8])

        try:
            features = self._extract_audio_features(audio_f32, self._sample_rate)
            from consciousness.events import event_bus, PERCEPTION_AUDIO_FEATURES
            event_bus.emit(PERCEPTION_AUDIO_FEATURES, features=features)
            from hemisphere.distillation import distillation_collector
            feature_vec = [
                features["rms"], features["zcr"], features["spectral_centroid"],
                features["spectral_spread"], features["pitch_hz"],
                features["duration_s"], features["speaking_rate"],
            ]
            dist_origin = "synthetic" if self._synthetic_active else "mic"
            dist_fidelity = 0.7 if self._synthetic_active else 1.0
            distillation_collector.record(
                "audio_features", "features", feature_vec,
                {"conversation_id": conversation_id[:8]},
                origin=dist_origin, fidelity=dist_fidelity,
            )
            recent_spk = distillation_collector.get_latest("ecapa_tdnn")
            if recent_spk is not None:
                spk_data = recent_spk.data if isinstance(recent_spk.data, list) else []
                if len(spk_data) >= 16:
                    enriched = feature_vec + spk_data[:25]
                    distillation_collector.record(
                        "audio_features_enriched", "features", enriched,
                        {"conversation_id": conversation_id[:8]},
                        origin=dist_origin, fidelity=dist_fidelity,
                    )
        except Exception as exc:
            logger.debug("Audio feature extraction failed: %s", exc)

        if self._on_speech_ready:
            try:
                self._on_speech_ready(audio_f32, conversation_id)
            except Exception as exc:
                logger.error("on_speech_ready callback error: %s", exc)
