"""Diarization Data Collector — windowed embedding extraction for training.

Subscribes to raw audio events and extracts per-window ECAPA-TDNN embeddings
labeled against enrolled speaker profiles.  Only active when a diarization
learning job is in collect phase or later.

Output: JSONL at ~/.jarvis/diarization_training/segments.jsonl
Each line: {ts, window_start_s, window_end_s, embedding, speaker_label,
            confidence, conversation_id}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.expanduser("~/.jarvis/diarization_training")
_SEGMENTS_FILE = os.path.join(_DATA_DIR, "segments.jsonl")

WINDOW_S = 1.5
HOP_S = 0.75
SAMPLE_RATE = 16000
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_S * SAMPLE_RATE)
MIN_WINDOW_SAMPLES = int(0.5 * SAMPLE_RATE)

MAX_SEGMENTS_FILE_MB = 50
MAX_SEGMENTS_PER_SESSION = 5000


class DiarizationCollector:
    """Collects windowed speaker embeddings for diarization training."""

    def __init__(self) -> None:
        self._active = False
        self._lock = threading.Lock()
        self._buffer: bytearray = bytearray()
        self._buffer_start_ts: float = 0.0
        self._speaker_id: Any = None
        self._segment_count: int = 0
        self._session_count: int = 0
        self._errors: int = 0
        self._last_error: str = ""
        self._conversation_id: str = ""

    def activate(self, speaker_id: Any) -> None:
        with self._lock:
            self._speaker_id = speaker_id
            self._active = True
            self._session_count = 0
            os.makedirs(_DATA_DIR, exist_ok=True)
            logger.info("DiarizationCollector activated")

    def deactivate(self) -> None:
        with self._lock:
            self._active = False
            self._buffer.clear()
            logger.info("DiarizationCollector deactivated (segments this session: %d)", self._session_count)

    @property
    def is_active(self) -> bool:
        return self._active

    def set_conversation_id(self, cid: str) -> None:
        self._conversation_id = cid

    def feed_audio(self, pcm_i16: bytes) -> None:
        """Feed raw 16kHz int16 PCM audio. Extracts windows when enough data accumulates."""
        if not self._active:
            return
        if self._session_count >= MAX_SEGMENTS_PER_SESSION:
            return

        with self._lock:
            if not self._buffer:
                self._buffer_start_ts = time.time()
            self._buffer.extend(pcm_i16)

            while len(self._buffer) >= WINDOW_SAMPLES * 2:
                window_bytes = bytes(self._buffer[:WINDOW_SAMPLES * 2])
                del self._buffer[:HOP_SAMPLES * 2]

                try:
                    self._process_window(window_bytes)
                except Exception as e:
                    self._errors += 1
                    self._last_error = f"{type(e).__name__}: {e}"
                    if self._errors <= 3:
                        logger.warning("Diarization window processing error: %s", e)

    def _process_window(self, window_bytes: bytes) -> None:
        if self._speaker_id is None:
            return

        audio_i16 = np.frombuffer(window_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        if rms < 0.005:
            return

        embedding = self._speaker_id._extract_embedding(audio_f32)
        if embedding is None:
            return

        best_name = "unknown"
        best_score = 0.0
        for name, profile in self._speaker_id._profiles.items():
            stored = profile.get("embedding")
            if stored is None:
                continue
            if isinstance(stored, list):
                stored = np.array(stored, dtype=np.float32)
            score = self._speaker_id._cosine_sim(embedding, stored)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < 0.3:
            best_name = "unknown"

        now = time.time()
        window_start = now - WINDOW_S
        segment = {
            "ts": now,
            "window_start_s": round(window_start, 3),
            "window_end_s": round(now, 3),
            "embedding": [round(float(x), 6) for x in embedding[:32]],
            "embedding_full_dim": len(embedding),
            "speaker_label": best_name,
            "confidence": round(best_score, 4),
            "rms": round(rms, 6),
            "conversation_id": self._conversation_id[:16] if self._conversation_id else "",
        }

        self._write_segment(segment)
        self._segment_count += 1
        self._session_count += 1

    def _write_segment(self, segment: dict[str, Any]) -> None:
        try:
            file_size = os.path.getsize(_SEGMENTS_FILE) if os.path.exists(_SEGMENTS_FILE) else 0
            if file_size > MAX_SEGMENTS_FILE_MB * 1024 * 1024:
                self._rotate_file()

            with open(_SEGMENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(segment, separators=(",", ":")) + "\n")
        except Exception as e:
            self._errors += 1
            self._last_error = f"write: {e}"

    def _rotate_file(self) -> None:
        rotated = _SEGMENTS_FILE + f".{int(time.time())}.bak"
        try:
            os.rename(_SEGMENTS_FILE, rotated)
            logger.info("Rotated diarization segments file to %s", rotated)
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        segment_count_on_disk = 0
        try:
            if os.path.exists(_SEGMENTS_FILE):
                with open(_SEGMENTS_FILE, "r") as f:
                    segment_count_on_disk = sum(1 for _ in f)
        except Exception:
            pass

        return {
            "active": self._active,
            "total_segments": self._segment_count,
            "session_segments": self._session_count,
            "disk_segments": segment_count_on_disk,
            "errors": self._errors,
            "last_error": self._last_error,
            "data_dir": _DATA_DIR,
        }

    def load_training_data(self, max_segments: int = 5000) -> list[dict[str, Any]]:
        """Load segments from disk for training."""
        segments: list[dict[str, Any]] = []
        if not os.path.exists(_SEGMENTS_FILE):
            return segments
        try:
            with open(_SEGMENTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        segments.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(segments) >= max_segments:
                        break
        except Exception as e:
            logger.warning("Failed to load diarization training data: %s", e)
        return segments

    def get_speaker_distribution(self) -> dict[str, int]:
        """Get speaker label counts from training data."""
        dist: dict[str, int] = {}
        segments = self.load_training_data()
        for seg in segments:
            label = seg.get("speaker_label", "unknown")
            dist[label] = dist.get(label, 0) + 1
        return dist


diarization_collector = DiarizationCollector()
