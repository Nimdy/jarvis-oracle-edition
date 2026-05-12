"""Laptop-side emotion detection from audio.

Two modes:
  1. GPU mode (wav2vec2): Classifies emotion directly from the audio waveform
     using a fine-tuned wav2vec2 model on CUDA. More accurate, ~500MB VRAM.
  2. Heuristic mode: Falls back to rule-based classification from audio features
     (RMS, pitch, spectral centroid, speech rate) when the ML model is unavailable.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

WAV2VEC_AVAILABLE = False
_w2v_model = None
_w2v_processor = None
_w2v_device = "cpu"

try:
    import torch
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    WAV2VEC_AVAILABLE = True
except ImportError:
    pass


@dataclass
class EmotionResult:
    emotion: str  # happy, sad, angry, frustrated, neutral, excited, calm
    raw_emotion: str = ""
    confidence: float = 0.0
    method: str = "heuristic"


_EXPECTED_HEAD_PREFIXES = ("classifier.", "projector.")

_LABEL_TO_JARVIS = {
    "angry": "angry",
    "calm": "calm",
    "disgust": "frustrated",
    "fearful": "frustrated",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "excited",
}


class AudioEmotionClassifier:
    """Classify user emotion from voice characteristics or raw audio."""

    def __init__(self, device: str = "cuda", model_name: str = "") -> None:
        self._recent_results: deque[EmotionResult] = deque(maxlen=10)
        self._last_emit_time: float = 0.0
        self._min_emit_interval_s: float = 5.0
        self._lock = threading.Lock()
        self._gpu_available = False
        self._model_healthy = False
        self._health_reason: str = ""
        self._device = device
        self._emotion_map: dict[str, str] = dict(_LABEL_TO_JARVIS)

        if not model_name:
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

        if WAV2VEC_AVAILABLE:
            self._try_load_gpu_model(model_name, device)

        logger.info("Emotion trust: healthy=%s%s",
                     self._model_healthy,
                     f" reason={self._health_reason}" if self._health_reason else "")

    def _try_load_gpu_model(self, model_name: str, device: str) -> None:
        global _w2v_model, _w2v_processor, _w2v_device
        try:
            import torch

            use_device = device if torch.cuda.is_available() else "cpu"

            _w2v_processor = self._load_with_local_fallback(
                AutoFeatureExtractor, model_name,
            )
            _w2v_model = self._load_with_local_fallback(
                AutoModelForAudioClassification, model_name,
            )

            self._remap_custom_head(_w2v_model, model_name)

            healthy, reason = self._check_model_keys(_w2v_model)
            self._model_healthy = healthy
            self._health_reason = reason

            if healthy:
                smoke_ok, smoke_reason = self._smoke_test(_w2v_model, _w2v_processor)
                if not smoke_ok:
                    self._model_healthy = False
                    self._health_reason = f"smoke_test_failed: {smoke_reason}"

            self._build_emotion_map(_w2v_model)

            _w2v_model = _w2v_model.to(use_device)
            _w2v_model.eval()
            _w2v_device = use_device
            self._gpu_available = True
            logger.info("Emotion classifier ready: wav2vec2 on %s (model_healthy=%s)",
                        use_device, self._model_healthy)
        except Exception as exc:
            logger.warning("wav2vec2 emotion model failed to load: %s — using heuristics", exc)
            self._gpu_available = False
            self._model_healthy = False
            self._health_reason = f"load_failed: {exc}"

    @staticmethod
    def _load_with_local_fallback(loader_cls, model_name: str):
        """Try loading from local cache first, fall back to network."""
        from config import get_models_dir
        cache_dir = str(get_models_dir() / "huggingface")
        try:
            return loader_cls.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            logger.info("Emotion model local cache miss for %s, downloading", loader_cls.__name__)
            return loader_cls.from_pretrained(model_name, cache_dir=cache_dir)

    def _remap_custom_head(self, model, model_name: str) -> None:
        """Remap custom checkpoint head keys if the model uses a non-standard layout.

        The ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition checkpoint
        stores its classification head as classifier.dense + classifier.output,
        but Wav2Vec2ForSequenceClassification expects projector + classifier.
        When this mismatch is detected, we replace the model's head layers with
        correctly-shaped ones and copy the trained weights.  If the upstream repo
        ever fixes the keys, classifier.dense.weight will be absent and this
        function becomes a no-op.
        """
        try:
            import torch
            from huggingface_hub import try_to_load_from_cache
            from safetensors.torch import load_file as load_safetensors

            from config import get_models_dir
            hf_cache = str(get_models_dir() / "huggingface")
            cached_path = try_to_load_from_cache(model_name, "model.safetensors", cache_dir=hf_cache)
            if cached_path is None or isinstance(cached_path, str) and not cached_path:
                return

            raw_sd = load_safetensors(cached_path)

            if "classifier.dense.weight" not in raw_sd:
                return

            dense_w = raw_sd["classifier.dense.weight"]
            dense_b = raw_sd["classifier.dense.bias"]
            output_w = raw_sd["classifier.output.weight"]
            output_b = raw_sd["classifier.output.bias"]

            logger.info("Emotion head remap: detected custom checkpoint layout")
            logger.info("  classifier.dense.weight: %s", list(dense_w.shape))
            logger.info("  classifier.dense.bias:   %s", list(dense_b.shape))
            logger.info("  classifier.output.weight: %s", list(output_w.shape))
            logger.info("  classifier.output.bias:   %s", list(output_b.shape))

            hidden_size = model.config.hidden_size
            hidden_dim = dense_w.shape[0]
            num_labels = output_w.shape[0]

            if list(dense_w.shape) != [hidden_dim, hidden_size]:
                self._health_reason = (
                    f"remap_shape_mismatch: dense.weight {list(dense_w.shape)} "
                    f"expected [{hidden_dim}, {hidden_size}]"
                )
                logger.error("Emotion head remap FAILED: %s", self._health_reason)
                return
            if list(output_w.shape) != [num_labels, hidden_dim]:
                self._health_reason = (
                    f"remap_shape_mismatch: output.weight {list(output_w.shape)} "
                    f"expected [{num_labels}, {hidden_dim}]"
                )
                logger.error("Emotion head remap FAILED: %s", self._health_reason)
                return

            import torch.nn as nn
            model.projector = nn.Linear(hidden_size, hidden_dim)
            model.classifier = nn.Linear(hidden_dim, num_labels)

            model.projector.weight.data.copy_(dense_w)
            model.projector.bias.data.copy_(dense_b)
            model.classifier.weight.data.copy_(output_w)
            model.classifier.bias.data.copy_(output_b)

            logger.info(
                "Emotion head remap applied: projector %d->%d, classifier %d->%d, labels=%s",
                hidden_size, hidden_dim, hidden_dim, num_labels,
                list(model.config.id2label.values()) if hasattr(model.config, "id2label") else "?",
            )
        except Exception as exc:
            logger.warning("Emotion head remap skipped: %s", exc)

    @staticmethod
    def _smoke_test(model, processor) -> tuple[bool, str]:
        """Run a deterministic inference on a canned waveform to verify the head works."""
        try:
            import torch
            canned = np.zeros(16000, dtype=np.float32)
            inputs = processor(canned, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values).logits

            if logits.shape[1] != len(model.config.id2label):
                return False, f"logits_shape {list(logits.shape)} != num_labels {len(model.config.id2label)}"
            if not torch.all(torch.isfinite(logits)):
                return False, "logits contain NaN or Inf"
            if float(torch.std(logits)) < 1e-6:
                return False, f"degenerate logits (std={float(torch.std(logits)):.2e})"
            argmax = int(torch.argmax(logits, dim=-1).item())
            if argmax < 0 or argmax >= logits.shape[1]:
                return False, f"argmax {argmax} out of range [0, {logits.shape[1]})"

            logger.info("Emotion smoke test passed: logits_shape=%s std=%.4f argmax=%d",
                        list(logits.shape), float(torch.std(logits)), argmax)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _build_emotion_map(self, model) -> None:
        """Build emotion label mapping from runtime id2label config, not hardcoded assumptions."""
        id2label = getattr(model.config, "id2label", None)
        if not id2label:
            logger.warning("Emotion model has no id2label — using default label map")
            return

        logger.info("Emotion model id2label: %s", id2label)
        self._emotion_map = {}
        for _idx, label in id2label.items():
            raw = label.lower().strip()
            jarvis = _LABEL_TO_JARVIS.get(raw, "neutral")
            self._emotion_map[raw] = jarvis
        logger.info("Emotion map (data-driven): %s", self._emotion_map)

    @staticmethod
    def _check_model_keys(model) -> tuple[bool, str]:
        """Deterministic health check: verify classifier/projector head keys exist."""
        model_keys = set(model.state_dict().keys())
        has_head = any(
            k.startswith(prefix) for k in model_keys for prefix in _EXPECTED_HEAD_PREFIXES
        )
        if not has_head:
            missing = [p.rstrip(".") for p in _EXPECTED_HEAD_PREFIXES]
            return False, f"missing_keys: {', '.join(missing)}"

        try:
            import torch as _torch
            for name, param in model.named_parameters():
                if any(name.startswith(p) for p in _EXPECTED_HEAD_PREFIXES):
                    if param.requires_grad and _torch.all(param == 0):
                        return False, f"zeroed_head_param: {name}"
        except Exception:
            pass

        return True, ""

    def classify_audio_b64(self, audio_b64: str, sample_rate: int = 16000) -> EmotionResult | None:
        """Classify emotion from base64-encoded int16 PCM audio using the GPU model."""
        now = time.time()
        if now - self._last_emit_time < self._min_emit_interval_s:
            return None

        if not self._gpu_available:
            return None

        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
        except Exception as exc:
            logger.error("Emotion audio decode failed: %s", exc)
            return None

        return self._classify_waveform(audio_f32, sample_rate)

    @staticmethod
    def _signal_quality(audio_f32: np.ndarray, sr: int) -> float:
        """Estimate audio quality for teacher confidence weighting (0.3..1.0)."""
        n = len(audio_f32)
        if n <= 0:
            return 0.3

        rms = float(np.sqrt(np.mean(audio_f32 ** 2)) + 1e-9)
        rms_db = 20.0 * float(np.log10(rms))
        dur = n / float(sr)

        zcr = float(np.sum(np.abs(np.diff(np.sign(audio_f32)))) / max(1, 2 * n))

        fft_mag = np.abs(np.fft.rfft(audio_f32))
        freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
        mag_sum = float(np.sum(fft_mag)) + 1e-10
        centroid = float(np.sum(freqs * fft_mag) / mag_sum)

        loud = float(np.clip((rms_db + 60.0) / 35.0, 0.0, 1.0))
        d_ok = float(np.clip(dur / 1.0, 0.0, 1.0))

        if zcr < 0.005 or zcr > 0.35:
            z_ok = 0.3
        else:
            z_ok = 1.0

        c_ok = 1.0 if 150.0 <= centroid <= 3800.0 else 0.4

        q = 0.45 * loud + 0.25 * d_ok + 0.15 * z_ok + 0.15 * c_ok
        return float(np.clip(q, 0.3, 1.0))

    def _classify_waveform(self, audio_f32: np.ndarray, sample_rate: int, origin: str = "mic") -> EmotionResult | None:
        """Run wav2vec2 inference on a float32 waveform."""
        if len(audio_f32) < sample_rate * 0.5:
            return None

        with self._lock:
            try:
                import torch
                inputs = _w2v_processor(
                    audio_f32, sampling_rate=sample_rate, return_tensors="pt", padding=True,
                )
                input_values = inputs.input_values.to(_w2v_device)

                with torch.no_grad():
                    logits = _w2v_model(input_values).logits

                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()

                raw_label = _w2v_model.config.id2label.get(predicted_id, "neutral")
                emotion = self._emotion_map.get(raw_label.lower(), "neutral")

                if emotion == "neutral" and confidence < 0.7:
                    return None

                try:
                    from hemisphere.distillation import distillation_collector
                    q = self._signal_quality(audio_f32, sample_rate)
                    fidelity = float(np.clip(confidence * q, 0.0, 1.0))
                    if origin == "synthetic":
                        fidelity = min(fidelity, 0.7)
                    distillation_collector.record(
                        "wav2vec2_emotion", "logits", probs[0].cpu().tolist(),
                        {"predicted": raw_label, "confidence": round(confidence, 3), "q": round(q, 3)},
                        origin=origin, fidelity=fidelity,
                    )
                except Exception as _distill_exc:
                    logger.debug("Distillation teacher record failed: %s", _distill_exc)

                result = EmotionResult(
                    emotion=emotion,
                    raw_emotion=raw_label,
                    confidence=confidence,
                    method="wav2vec2",
                )
                self._recent_results.append(result)
                self._last_emit_time = time.time()
                logger.info("Audio emotion (wav2vec2): %s (conf=%.2f, raw=%s)",
                            emotion, confidence, raw_label)
                return result
            except Exception as exc:
                logger.debug("wav2vec2 emotion inference failed: %s", exc)
                return None

    def classify(
        self,
        rms: float = 0.0,
        spectral_centroid: float = 0.0,
        pitch_hz: float = 0.0,
        speech_rate: float = 0.0,
        duration_s: float = 0.0,
    ) -> EmotionResult | None:
        """Classify emotion from audio features using heuristics (fallback)."""
        if duration_s < 0.5:
            return None

        now = time.time()
        if now - self._last_emit_time < self._min_emit_interval_s:
            return None

        emotion = "neutral"
        confidence = 0.4

        if rms > 0.08 and pitch_hz > 250:
            emotion = "excited"
            confidence = 0.6
        elif rms > 0.06 and spectral_centroid > 3000:
            emotion = "angry"
            confidence = 0.55
        elif rms < 0.02 and pitch_hz < 150 and pitch_hz > 0:
            emotion = "sad"
            confidence = 0.5
        elif speech_rate > 4.0:
            emotion = "frustrated"
            confidence = 0.5
        elif rms > 0.04 and pitch_hz > 200 and spectral_centroid > 2000:
            emotion = "happy"
            confidence = 0.55
        elif rms < 0.03 and speech_rate < 2.0 and speech_rate > 0:
            emotion = "calm"
            confidence = 0.5

        if emotion == "neutral":
            return None

        result = EmotionResult(emotion=emotion, confidence=confidence)
        self._recent_results.append(result)
        self._last_emit_time = now
        return result

    def get_dominant_emotion(self, window: int = 5) -> str:
        """Get the most frequent non-neutral emotion in recent results."""
        if not self._recent_results:
            return "neutral"
        recent = list(self._recent_results)[-window:]
        counts: dict[str, int] = {}
        for r in recent:
            counts[r.emotion] = counts.get(r.emotion, 0) + 1
        if not counts:
            return "neutral"
        return max(counts, key=counts.get)

    def get_stats(self) -> dict:
        """Public stats for hemisphere gap detector and dashboard."""
        recent = list(self._recent_results)
        avg_conf = sum(r.confidence for r in recent) / len(recent) if recent else 0.0
        return {
            "model_healthy": self._model_healthy,
            "gpu_available": self._gpu_available,
            "avg_confidence": avg_conf,
            "recent_count": len(recent),
            "avg_inference_ms": 0.0,
        }


emotion_classifier = AudioEmotionClassifier()
