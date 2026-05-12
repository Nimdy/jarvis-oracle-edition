"""Laptop-side TTS — Kokoro ONNX synthesis for streaming audio to Pi.

When onnxruntime-gpu is installed, Kokoro runs on CUDA for ~7x faster
synthesis (~0.25s per sentence vs ~1.8s on CPU).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from reasoning.voice_policy import OracleVoicePolicy, VoicePolicyConfig, VoicePolicyContext, VoiceProfile

logger = logging.getLogger(__name__)

KOKORO_AVAILABLE = False
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    pass


_MD_EMPHASIS = re.compile(r'\*{1,3}([^*]+?)\*{1,3}')
_MD_ORPHAN_ASTERISK = re.compile(r'\*{1,3}')
_MD_MISC = re.compile(r'[`#~_]{1,3}')
_MD_HEADER = re.compile(r'^#{1,6}\s*', re.MULTILINE)
_MD_BULLET = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
_MD_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r'^\s*>\s*', re.MULTILINE)
_MD_HR = re.compile(r'^-{3,}$', re.MULTILINE)
_MD_LINK = re.compile(r'\[([^\]]*)\]\([^)]*\)')
_EMOJI_RE = re.compile(
    r'[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000feff'
    r'\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\U00002702-\U000027b0'
    r'\U0000200d\U0000fe0f]+',
)
_MULTI_SPACE = re.compile(r'\s{2,}')

# Number normalization for spoken output: "0.78" → "78 percent"
_DECIMAL_SCORE = re.compile(r'\b0\.(\d{1,4})\b')
_PERCENT_RE = re.compile(r'\b(\d{1,3})(?:\.(\d{1,2}))?%')
_YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')
_CLOCK_TIME_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)\b')
_COMPACT_TIME_RE = re.compile(r'\b([01]\d|2[0-3])([0-5]\d)\b')
_NUMBER_COMMA_RE = re.compile(r'(?<=\d),(?=\d)')
_SLASH_RE = re.compile(r'(?<=[A-Za-z0-9])/(?=[A-Za-z0-9])')
_DOUBLE_QUOTE_RE = re.compile(r'[“”"`]')
_EDGE_APOSTROPHE_RE = re.compile(r"(?<!\w)'|'(?!\w)")
_PAREN_RE = re.compile(r'[\(\)\[\]\{\}]')
_PUNCT_RUN_RE = re.compile(r'([!?.,;:]){2,}')
_ELLIPSIS_RE = re.compile(r'\.{3,}')
_DASH_CLAUSE_RE = re.compile(r'\s[-–—]+\s')
_SPACE_BEFORE_PUNCT_RE = re.compile(r'\s+([,.;:!?])')
_MIN_TTS_CHUNK_CHARS = 60
_LEADING_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bI['’]m\b", re.IGNORECASE), "I am"),
    (re.compile(r"\bI['’]ve\b", re.IGNORECASE), "I have"),
    (re.compile(r"\bI['’]ll\b", re.IGNORECASE), "I will"),
    (re.compile(r"\bI['’]d\b", re.IGNORECASE), "I would"),
    (re.compile(r"\byou['’]re\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bwe['’]re\b", re.IGNORECASE), "we are"),
    (re.compile(r"\bthey['’]re\b", re.IGNORECASE), "they are"),
    (re.compile(r"\bit['’]s\b", re.IGNORECASE), "it is"),
    (re.compile(r"\bthat['’]s\b", re.IGNORECASE), "that is"),
    (re.compile(r"\bthere['’]s\b", re.IGNORECASE), "there is"),
    (re.compile(r"\bhere['’]s\b", re.IGNORECASE), "here is"),
    (re.compile(r"\bwhat['’]s\b", re.IGNORECASE), "what is"),
)


def _decimal_to_spoken(m: re.Match) -> str:
    """Convert 0.XX decimal scores to spoken percentages."""
    pct = float(m.group(0)) * 100
    if pct == int(pct):
        return f"{int(pct)} percent"
    return f"{pct:.1f} percent"


def _year_to_spoken(m: re.Match) -> str:
    """Speak years digit-by-digit to avoid Kokoro stumbling on commas/years."""
    return " ".join(m.group(1))


def _percent_to_spoken(m: re.Match) -> str:
    whole = m.group(1)
    frac = m.group(2)
    if frac:
        return f"{whole} point {frac} percent"
    return f"{whole} percent"


def _clock_time_to_spoken(m: re.Match) -> str:
    hour = int(m.group(1))
    minute = int(m.group(2))
    if minute == 0:
        return f"{hour} o'clock"
    if minute < 10:
        return f"{hour} oh {minute}"
    return f"{hour} {minute}"


def _compact_time_to_spoken(m: re.Match) -> str:
    hour = int(m.group(1))
    minute = int(m.group(2))
    if minute == 0:
        return f"{hour} hundred"
    if minute < 10:
        return f"{hour} oh {minute}"
    return f"{hour} {minute}"


_MAX_TTS_CHARS = 350
_CHUNK_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+')


class BrainTTS:
    """GPU-accelerated TTS on the laptop — synthesizes audio and returns base64 WAV."""

    def __init__(
        self,
        model_path: str = "",
        voices_path: str = "",
        voice: str = "af_bella",
        speed: float = 1.0,
        device: str = "cuda",
        voice_policy: OracleVoicePolicy | None = None,
    ):
        self._voice = voice
        self._speed = speed
        self._kokoro: Kokoro | None = None
        self.available = False
        self._device = device
        self._voice_policy = voice_policy or OracleVoicePolicy(
            VoicePolicyConfig(base_voice=voice, base_speed=speed)
        )

        if not model_path:
            from config import get_models_dir
            consolidated = get_models_dir() / "kokoro-v1.0.onnx"
            legacy = Path(__file__).resolve().parent.parent / "models" / "kokoro-v1.0.onnx"
            if consolidated.exists():
                model_path = str(consolidated)
            elif legacy.exists():
                model_path = str(legacy)
        if not voices_path and model_path:
            voices_path = str(Path(model_path).parent / "voices-v1.0.bin")

        if not KOKORO_AVAILABLE:
            logger.info("kokoro_onnx not installed — BrainTTS disabled")
            return
        if not model_path or not Path(model_path).exists():
            logger.info("Kokoro model not found at %s — BrainTTS disabled", model_path)
            return
        if not voices_path or not Path(voices_path).exists():
            logger.info("Kokoro voices not found at %s — BrainTTS disabled", voices_path)
            return

        try:
            if device == "cuda" and "ONNX_PROVIDER" not in os.environ:
                os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
                logger.info("Set ONNX_PROVIDER=CUDAExecutionProvider for GPU TTS")

            self._kokoro = Kokoro(model_path, voices_path)
            self.available = True

            t0 = time.monotonic()
            self._kokoro.create(text="ok", voice=voice, speed=speed)
            warmup_ms = (time.monotonic() - t0) * 1000
            logger.info("BrainTTS ready: Kokoro ONNX on %s (voice=%s, warmup=%.0fms)",
                        device, voice, warmup_ms)
        except Exception as exc:
            logger.error("Failed to load Kokoro for brain TTS: %s", exc)

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown formatting and normalize numbers for spoken output."""
        text = _MD_HR.sub('', text)
        text = _MD_HEADER.sub('', text)
        text = _MD_BULLET.sub('', text)
        text = _MD_NUMBERED.sub('', text)
        text = _MD_BLOCKQUOTE.sub('', text)
        text = _MD_LINK.sub(r'\1', text)
        text = _MD_EMPHASIS.sub(r'\1', text)
        text = _MD_ORPHAN_ASTERISK.sub('', text)
        text = _MD_MISC.sub('', text)
        text = _EMOJI_RE.sub('', text)
        text = text.replace("’", "'")
        for pattern, replacement in _LEADING_CONTRACTIONS:
            text = pattern.sub(replacement, text)
        text = text.replace("&", " and ")
        text = _NUMBER_COMMA_RE.sub('', text)
        text = _DECIMAL_SCORE.sub(_decimal_to_spoken, text)
        text = _PERCENT_RE.sub(_percent_to_spoken, text)
        text = _CLOCK_TIME_RE.sub(_clock_time_to_spoken, text)
        text = _COMPACT_TIME_RE.sub(_compact_time_to_spoken, text)
        text = _YEAR_RE.sub(_year_to_spoken, text)
        text = _SLASH_RE.sub(' or ', text)
        text = _ELLIPSIS_RE.sub('. ', text)
        text = _DASH_CLAUSE_RE.sub(', ', text)
        text = _DOUBLE_QUOTE_RE.sub('', text)
        text = _EDGE_APOSTROPHE_RE.sub('', text)
        text = _PAREN_RE.sub(' ', text)
        text = _PUNCT_RUN_RE.sub(lambda m: m.group(1), text)
        text = _MULTI_SPACE.sub(' ', text)
        text = re.sub(r'\n{2,}', '. ', text)
        text = text.replace('\n', ' ')
        text = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*;\s*', '. ', text)
        text = re.sub(r'\s*:\s*', ', ', text)
        text = re.sub(r'\s*\?\s*', '? ', text)
        text = re.sub(r'\s*!\s*', '! ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        return text.strip()

    @staticmethod
    def _split_long_text(
        text: str,
        *,
        max_chars: int = _MAX_TTS_CHARS,
        min_chunk_chars: int = _MIN_TTS_CHUNK_CHARS,
    ) -> list[str]:
        """Split text into chunks safe for Kokoro's phoneme limit."""
        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        for part in _CHUNK_SPLIT_RE.split(text):
            part = part.strip()
            if not part:
                continue
            if chunks and len(chunks[-1]) + len(part) + 1 <= max_chars:
                chunks[-1] += " " + part
            else:
                while len(part) > max_chars:
                    cut = part[:max_chars].rfind(' ')
                    if cut < 50:
                        cut = max_chars
                    chunks.append(part[:cut].strip())
                    part = part[cut:].strip()
                if part:
                    chunks.append(part)
        if not chunks:
            return [text[:max_chars]]

        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(chunk) < min_chunk_chars and len(merged[-1]) + len(chunk) + 1 <= max_chars:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    def get_voice_profile(
        self,
        *,
        tone: str = "professional",
        user_emotion: str = "neutral",
        emotion_trusted: bool = False,
        speaker_name: str = "",
        proactive: bool = False,
        style_override: str = "",
    ) -> VoiceProfile:
        return self._voice_policy.resolve(VoicePolicyContext(
            tone=tone,
            user_emotion=user_emotion,
            emotion_trusted=emotion_trusted,
            speaker_name=speaker_name,
            proactive=proactive,
            style_override=style_override,
        ))

    _LEADING_SILENCE_FIRST_S = 0.05
    _LEADING_SILENCE_CONTINUATION_S = 0.03

    def synthesize_b64(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        profile: VoiceProfile | None = None,
        leading_silence: bool = True,
    ) -> str | None:
        """Synthesize text and return base64-encoded WAV data.

        Args:
            leading_silence: If True, prepend a short onset pad (first chunk).
                             If False, no leading silence (continuation chunks).
        """
        if not self.available or not text.strip():
            return None
        try:
            selected_voice = profile.voice if profile else (voice or self._voice)
            selected_speed = float(profile.speed if profile else (speed if speed is not None else self._speed))
            selected_max_chars = profile.max_chars if profile else _MAX_TTS_CHARS
            selected_min_chunk_chars = profile.min_chunk_chars if profile else _MIN_TTS_CHUNK_CHARS
            if profile is None:
                logger.warning(
                    "BrainTTS synthesize_b64 called without VoiceProfile — using static fallback "
                    "(voice=%s speed=%.2f)",
                    selected_voice,
                    selected_speed,
                )
            text = self._clean_for_speech(text)
            if not text:
                return None
            t0 = time.monotonic()
            chunks = self._split_long_text(
                text,
                max_chars=selected_max_chars,
                min_chunk_chars=selected_min_chunk_chars,
            )
            all_samples: list[np.ndarray] = []
            sr = 24000
            for chunk in chunks:
                try:
                    chunk_samples, sr = self._kokoro.create(
                        text=chunk, voice=selected_voice, speed=selected_speed,
                    )
                    all_samples.append(chunk_samples)
                except Exception as exc:
                    logger.warning("BrainTTS chunk failed (%d chars): %s", len(chunk), exc)
                    continue
            if not all_samples:
                return None
            samples = np.concatenate(all_samples) if len(all_samples) > 1 else all_samples[0]
            synth_ms = (time.monotonic() - t0) * 1000

            pad_s = self._LEADING_SILENCE_FIRST_S if leading_silence else self._LEADING_SILENCE_CONTINUATION_S
            if pad_s > 0:
                leading_frames = int(sr * pad_s)
                samples = np.concatenate([np.zeros(leading_frames, dtype=samples.dtype), samples])

            int16 = (samples * 32768.0).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(int16.tobytes())

            audio_ms = len(samples) / sr * 1000
            profile_id = profile.profile_id if profile else "static"
            logger.info(
                "BrainTTS[%s]: %d chars → %.0fms synth, %.0fms audio (voice=%s speed=%.2f)",
                profile_id, len(text), synth_ms, audio_ms, selected_voice, selected_speed,
            )
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.error("BrainTTS synthesis failed: %s", exc)
            return None
