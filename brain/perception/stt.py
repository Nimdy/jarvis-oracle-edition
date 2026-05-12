"""Laptop-side speech-to-text using faster-whisper (GPU-accelerated).

Receives recorded audio clips from the Pi, transcribes with a singleton
WhisperModel, and emits PERCEPTION_TRANSCRIPTION for the brain pipeline.

Hallucination defenses (in order of strength):
    1. condition_on_previous_text=False — kills the priming chain that causes
       whisper to "snowball" boilerplate between decoder windows.
    2. compression_ratio_threshold=2.0 — stricter than default 2.4; whisper's
       hallucinations are often repetitive and have high compression ratios, so
       this rejects them internally.
    3. Per-segment no_speech_prob gate — whisper emits a self-reported silence
       probability per segment; drop any segment where it exceeds the floor.
    4. RMS-gated known-hallucination blocklist — narrow defense in depth for
       well-documented whisper training-data boilerplate (e.g. "Thanks for
       watching!"). Only fires on suspiciously low-energy audio to avoid
       over-blocking legitimate speech. Deliberately excludes ambiguous
       phrases like "Bye." or "Goodbye." that could be legitimate.
"""

from __future__ import annotations

import base64
import glob as _glob
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed — laptop STT disabled")


# Well-documented whisper hallucinations sourced from its YouTube-heavy training
# data. Kept narrow on purpose: phrases that are essentially impossible to be a
# legitimate user utterance to Jarvis. "Bye", "Goodbye", "Amen", "you", etc. are
# intentionally excluded because users could legitimately say them.
_KNOWN_HALLUCINATIONS: frozenset[str] = frozenset({
    "thanks for watching.",
    "thanks for watching!",
    "thanks for watching",
    "thank you for watching.",
    "thank you for watching!",
    "thank you for watching",
    "please subscribe.",
    "please subscribe!",
    "please like and subscribe.",
    "please like and subscribe!",
    "like and subscribe.",
    "i'll see you in the next video.",
    "see you in the next video.",
    "see you next time.",
    "subtitles by the amara.org community",
})

# RMS below this level means the audio is near-silent or room-tone, which is
# exactly the regime where whisper fabricates boilerplate. Only enforce the
# hallucination blocklist below this floor so genuine loud user utterances
# never hit the filter.
_SUSPICIOUS_RMS_THRESHOLD = 0.02

# Per-segment silence-probability floor. Segments above this are dropped
# regardless of whisper's global decision.
_PER_SEGMENT_NO_SPEECH_FLOOR = 0.6

# RMS below this skips the "VAD was probably wrong, retry without it" branch.
# The prior default (0.005) retried on effective room tone and was a direct
# contributor to hallucination leakage.
_VAD_RETRY_RMS_FLOOR = 0.02

# Whisper kwargs shared between the primary call and the no-VAD retry so both
# paths enjoy identical hallucination defenses.
_WHISPER_KWARGS = dict(
    language="en",
    beam_size=5,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.0,
    no_speech_threshold=0.6,
)


def _get_whisper_download_root() -> str:
    """Return the consolidated download root for faster-whisper models."""
    from config import get_models_dir
    return str(get_models_dir() / "faster-whisper")


def _find_cached_whisper_path(model_size: str) -> str | None:
    """Find a locally cached CTranslate2 snapshot directory for faster-whisper.

    Returns the snapshot path if a valid cache exists (has model.bin + config.json),
    otherwise None. Checks the consolidated models dir first, then the legacy HF cache.
    """
    search_dirs = [
        _get_whisper_download_root(),
        os.path.expanduser("~/.cache/huggingface/hub/"),
    ]
    for hub_dir in search_dirs:
        if not os.path.isdir(hub_dir):
            continue
        patterns = [
            os.path.join(hub_dir, f"models--*faster-whisper-{model_size}", "snapshots", "*"),
            os.path.join(hub_dir, f"models--*faster-whisper-{model_size.replace('-', '_')}", "snapshots", "*"),
        ]
        for pattern in patterns:
            for snap_dir in sorted(_glob.glob(pattern)):
                if (os.path.isfile(os.path.join(snap_dir, "model.bin"))
                        and os.path.isfile(os.path.join(snap_dir, "config.json"))):
                    return snap_dir
    return None


def _compute_rms(audio_f32: np.ndarray) -> float:
    """Root-mean-square energy of a float32 mono waveform in [-1, 1]."""
    if audio_f32 is None or len(audio_f32) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_f32 ** 2)))


def _looks_like_hallucination(text: str, rms: float) -> bool:
    """Narrow blocklist: known whisper boilerplate on suspiciously quiet audio.

    Returns True only if:
        (a) the normalized text exactly matches a well-documented whisper
            hallucination phrase, AND
        (b) the input audio RMS is below the suspicious-silence floor.

    Legitimate loud speech that happens to match a blocklist phrase is allowed
    through. Legitimate quiet speech that does NOT match a blocklist phrase is
    likewise never filtered here.
    """
    if not text:
        return False
    if rms >= _SUSPICIOUS_RMS_THRESHOLD:
        return False
    return text.strip().lower() in _KNOWN_HALLUCINATIONS


def _collect_segment_text(
    segments: "object",
    per_segment_floor: float,
) -> tuple[str, int, int, list[str]]:
    """Join whisper segments, dropping any whose no_speech_prob exceeds the floor.

    Returns (joined_text, total_segments_seen, segments_kept, drop_reasons).
    Drop reasons are only emitted for inspection/logging — callers should not
    pattern-match on them.
    """
    kept: list[str] = []
    dropped_reasons: list[str] = []
    total = 0
    for seg in segments:
        total += 1
        nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        seg_text = (getattr(seg, "text", "") or "").strip()
        if nsp > per_segment_floor:
            dropped_reasons.append(
                f"no_speech_prob={nsp:.2f} text={seg_text[:40]!r}"
            )
            continue
        if seg_text:
            kept.append(seg_text)
    return " ".join(kept).strip(), total, len(kept), dropped_reasons


class LaptopSTT:
    """Singleton STT service. One model instance, serialized GPU access."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ) -> None:
        self._lock = threading.Lock()
        self._model: WhisperModel | None = None
        self.available = False

        if not FASTER_WHISPER_AVAILABLE:
            return

        model_path = _find_cached_whisper_path(model_size) or model_size
        if model_path != model_size:
            logger.info("LaptopSTT: using local cache %s", model_path)

        dl_root = _get_whisper_download_root()
        try:
            self._model = WhisperModel(
                model_path, device=device, compute_type=compute_type,
                download_root=dl_root,
            )
            self.available = True
            logger.info("LaptopSTT ready: model=%s device=%s compute=%s",
                        model_size, device, compute_type)
        except Exception as exc:
            logger.warning("GPU load failed (%s), trying CPU fallback", exc)
            try:
                self._model = WhisperModel(
                    model_path, device="cpu", compute_type="int8",
                    download_root=dl_root,
                )
                self.available = True
                logger.info("LaptopSTT ready: model=%s device=cpu (fallback)", model_size)
            except Exception as exc2:
                logger.error("Failed to load faster-whisper model: %s", exc2)

    def transcribe_b64(
        self,
        audio_b64: str,
        sample_rate: int = 16000,
    ) -> str:
        """Decode base64 int16 PCM and transcribe. Thread-safe."""
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
        except Exception as exc:
            logger.error("Audio decode failed: %s", exc)
            return ""
        return self.transcribe(audio_f32, sample_rate)

    def transcribe(
        self,
        audio_f32: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe float32 audio. Serializes GPU access via lock.

        Applies four-layer hallucination defense (see module docstring):
          1. whisper param tightening via _WHISPER_KWARGS
          2. per-segment no_speech_prob gate
          3. RMS-gated no-VAD retry on genuinely energetic but VAD-rejected audio
          4. narrow known-hallucination blocklist on suspiciously quiet audio
        """
        if not self.available or self._model is None:
            return ""

        with self._lock:
            t0 = time.monotonic()
            dur = len(audio_f32) / sample_rate
            rms = _compute_rms(audio_f32)
            try:
                segments, info = self._model.transcribe(
                    audio_f32,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200,
                    ),
                    **_WHISPER_KWARGS,
                )
                text, total_segs, kept_segs, dropped = _collect_segment_text(
                    segments, _PER_SEGMENT_NO_SPEECH_FLOOR,
                )

                if _looks_like_hallucination(text, rms):
                    logger.info(
                        "STT: rejected known hallucination %r (rms=%.4f, dur=%.1fs)",
                        text[:60], rms, dur,
                    )
                    return ""

                if not text.strip() and rms > _VAD_RETRY_RMS_FLOOR:
                    logger.info(
                        "STT: primary returned empty but audio has energy "
                        "(rms=%.4f, dur=%.1fs) — retrying without VAD",
                        rms, dur,
                    )
                    segments2, info = self._model.transcribe(
                        audio_f32,
                        vad_filter=False,
                        **_WHISPER_KWARGS,
                    )
                    text, total_segs, kept_segs, dropped = _collect_segment_text(
                        segments2, _PER_SEGMENT_NO_SPEECH_FLOOR,
                    )
                    if _looks_like_hallucination(text, rms):
                        logger.info(
                            "STT (no-VAD retry): rejected known hallucination %r (rms=%.4f)",
                            text[:60], rms,
                        )
                        return ""
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "STT (no-VAD retry): %.1fs audio -> %r "
                        "(%.2fs, kept=%d/%d, rms=%.4f)",
                        dur, text[:80], elapsed, kept_segs, total_segs, rms,
                    )
                    return text

                elapsed = time.monotonic() - t0
                lang = getattr(info, "language", "?")
                lang_p = getattr(info, "language_probability", 0.0)
                logger.info(
                    "STT: %.1fs audio -> %r (%.2fs, kept=%d/%d, rms=%.4f, "
                    "lang=%s p=%.2f, dropped=%d)",
                    dur, text[:80], elapsed, kept_segs, total_segs, rms,
                    lang, lang_p, len(dropped),
                )
                return text
            except Exception as exc:
                logger.error("faster-whisper transcription failed: %s", exc)
                return ""
