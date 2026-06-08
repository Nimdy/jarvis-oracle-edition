"""Distillation Collector — Super-Synapse signal capture layer.

Teacher models generate rich cognitive signals during normal inference.
This module captures those signals with provenance tagging and fidelity
scoring, storing them in per-teacher JSONL streams for later distillation
training by Tier-1 hemisphere NNs.

Immune primitives:
  - origin: traces which sensor/model produced the signal
  - fidelity: composite trustworthiness score (teacher_confidence * signal_quality)
  - quarantine: low-fidelity samples stored separately, excluded from training
  - dedup: prevents training on near-identical consecutive samples
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path(os.path.expanduser("~/.jarvis/hemisphere_training"))
QUARANTINE_DIR = TRAINING_DATA_DIR / "quarantine"

QUARANTINE_THRESHOLD = 0.3
TEACHER_QUARANTINE_THRESHOLDS: dict[str, float] = {
    "wav2vec2_emotion": 0.08,
}
BUFFER_MAXLEN = 500
QUARANTINE_MAXLEN = 100


def _is_synthetic_origin(origin: str) -> bool:
    """True if a signal's origin marks it as synthetic-exercise data (weight-room
    P0). Synthetic exercises tag origin='synthetic'; everything else (live sensors,
    'system', 'disk', unknown) counts as lived/real — the conservative default."""
    return bool(origin) and str(origin).lower().startswith("synthetic")


# Weight-Room P1 min-N gates. A live-shadow accuracy below LIVE_SHADOW_MIN_N is
# too noisy to report as a number (1-2 samples give a meaningless 0%/100%); it
# stays None — honestly-unmeasured, never a fake 0. SUFFICIENT_N marks when the
# number is stable enough to be worth weighing. These are HONESTY floors, not
# authority floors — §24 forbids lowering them to make a panel look populated.
LIVE_SHADOW_MIN_N = 10
LIVE_SHADOW_SUFFICIENT_N = 50


def live_shadow_accuracy(
    correct: int, total: int,
    min_n: int = LIVE_SHADOW_MIN_N, sufficient_n: int = LIVE_SHADOW_SUFFICIENT_N,
) -> dict[str, Any]:
    """Weight-Room P1: the canonical lived-only shadow-accuracy shape.

    Callers pass counts already restricted to LIVED (origin != synthetic) samples
    — this helper does not see origin; it only enforces the honesty floor. Returns
    ``live_shadow_accuracy=None`` until ``min_n`` lived samples accrue (honestly
    unmeasured, NOT a fake 0.0), and flags ``sufficient_data`` at ``sufficient_n``.
    Mirrors intention_resolver._shadow_metrics_locked so every specialist reports
    the same honest shape.
    """
    total = max(0, int(total))
    correct = max(0, min(int(correct), total))
    acc = round(correct / total, 4) if total >= min_n else None
    return {
        "live_shadow_correct": correct,
        "live_shadow_total": total,
        "live_shadow_accuracy": acc,
        "sufficient_data": total >= sufficient_n,
        "min_n": min_n,
    }


DEDUP_TIME_BUCKET_S = 2.0


@dataclass
class TeacherSignal:
    teacher: str
    signal_type: str
    data: list[float] | dict
    timestamp: float
    metadata: dict
    origin: str = "system"
    fidelity: float = 1.0


def _compute_dedup_key(teacher: str, signal_type: str, data: list[float] | dict,
                       timestamp: float) -> str:
    bucket = int(timestamp / DEDUP_TIME_BUCKET_S)
    if isinstance(data, list):
        prefix = [round(v, 3) for v in data[:32]]
    else:
        prefix = str(data)[:128]
    raw = f"{teacher}:{signal_type}:{prefix}:{bucket}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class DistillationCollector:
    """Super-Synapse signal capture with provenance and immune primitives.

    Thread-safe singleton that captures teacher model outputs during normal
    inference. Signals are stored in per-teacher ring buffers (in-memory)
    and persisted to JSONL files for offline training.
    """

    _instance: DistillationCollector | None = None

    def __init__(self) -> None:
        self._buffers: dict[str, deque[TeacherSignal]] = {}
        self._quarantine: dict[str, deque[TeacherSignal]] = {}
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._synthetic_counts: dict[str, int] = {}  # weight-room P0: synthetic subset of _counts
        self._quarantine_counts: dict[str, int] = {}
        self._last_seen: dict[str, float] = {}
        self._recent_dedup_keys: deque[str] = deque(maxlen=200)
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        self._rehydrate_from_disk()

    def _rehydrate_from_disk(self) -> None:
        """Load persisted signals back into in-memory buffers on boot.

        Without this, distillation training can only use signals from the
        current session — progress accumulated across boots would be lost.
        """
        total = 0
        for jsonl_path in TRAINING_DATA_DIR.glob("distill_*.jsonl"):
            teacher = jsonl_path.stem.removeprefix("distill_")
            try:
                signals = self._parse_jsonl(jsonl_path, BUFFER_MAXLEN)
                if signals:
                    buf = self._buffers.setdefault(teacher, deque(maxlen=BUFFER_MAXLEN))
                    buf.extend(signals)
                    self._counts[teacher] = self._counts.get(teacher, 0) + len(signals)
                    syn = sum(1 for s in signals if _is_synthetic_origin(s.origin))
                    if syn:
                        self._synthetic_counts[teacher] = self._synthetic_counts.get(teacher, 0) + syn
                    if signals:
                        self._last_seen[teacher] = signals[-1].timestamp
                    total += len(signals)
            except Exception as exc:
                logger.warning("Failed to rehydrate %s: %s", jsonl_path.name, exc)

        for jsonl_path in QUARANTINE_DIR.glob("distill_*.jsonl"):
            teacher = jsonl_path.stem.removeprefix("distill_")
            try:
                signals = self._parse_jsonl(jsonl_path, QUARANTINE_MAXLEN)
                if signals:
                    buf = self._quarantine.setdefault(teacher, deque(maxlen=QUARANTINE_MAXLEN))
                    buf.extend(signals)
                    self._quarantine_counts[teacher] = self._quarantine_counts.get(teacher, 0) + len(signals)
            except Exception as exc:
                logger.warning("Failed to rehydrate quarantine %s: %s", jsonl_path.name, exc)

        if total > 0:
            logger.info("Rehydrated %d distillation signals from disk", total)

    @staticmethod
    def _parse_jsonl(path: Path, limit: int) -> list[TeacherSignal]:
        """Parse the last `limit` lines of a distillation JSONL file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []

        signals: list[TeacherSignal] = []
        for line in lines[-limit:]:
            try:
                row = json.loads(line)
                signals.append(TeacherSignal(
                    teacher=row["teacher"],
                    signal_type=row.get("type", "unknown"),
                    data=row["data"],
                    timestamp=row.get("t", 0.0),
                    metadata=row.get("meta", {}),
                    origin=row.get("origin", "disk"),
                    fidelity=row.get("fidelity", 0.5),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return signals

    @classmethod
    def get_instance(cls) -> DistillationCollector:
        if cls._instance is None:
            cls._instance = DistillationCollector()
        return cls._instance

    @classmethod
    def instance(cls) -> DistillationCollector:
        return cls.get_instance()

    def record(
        self,
        teacher: str,
        signal_type: str,
        data: list[float] | dict,
        metadata: dict | None = None,
        origin: str = "system",
        fidelity: float = 1.0,
    ) -> None:
        """Capture a teacher signal with provenance. Low-fidelity -> quarantine."""
        now = time.time()
        metadata = metadata or {}

        dedup_key = _compute_dedup_key(teacher, signal_type, data, now)
        with self._lock:
            if dedup_key in self._recent_dedup_keys:
                return
            self._recent_dedup_keys.append(dedup_key)

        signal = TeacherSignal(
            teacher=teacher,
            signal_type=signal_type,
            data=data,
            timestamp=now,
            metadata=metadata,
            origin=origin,
            fidelity=fidelity,
        )

        with self._lock:
            self._last_seen[teacher] = now

            threshold = TEACHER_QUARANTINE_THRESHOLDS.get(teacher, QUARANTINE_THRESHOLD)
            if fidelity < threshold:
                buf = self._quarantine.setdefault(teacher, deque(maxlen=QUARANTINE_MAXLEN))
                buf.append(signal)
                self._quarantine_counts[teacher] = self._quarantine_counts.get(teacher, 0) + 1
                self._write_jsonl(QUARANTINE_DIR / f"distill_{teacher}.jsonl", signal)
            else:
                buf = self._buffers.setdefault(teacher, deque(maxlen=BUFFER_MAXLEN))
                buf.append(signal)
                self._counts[teacher] = self._counts.get(teacher, 0) + 1
                if _is_synthetic_origin(origin):
                    self._synthetic_counts[teacher] = self._synthetic_counts.get(teacher, 0) + 1
                self._write_jsonl(TRAINING_DATA_DIR / f"distill_{teacher}.jsonl", signal)

    def get_training_batch(
        self,
        teacher: str,
        limit: int = 200,
        min_fidelity: float = 0.0,
        lived_only: bool = False,
    ) -> list[TeacherSignal]:
        """Return recent signals, optionally filtered by fidelity floor and/or to
        lived (non-synthetic) signals only — weight-room P0/P1: lets a caller
        compute live-shadow accuracy on real signals, excluding synthetic reps."""
        with self._lock:
            buf = self._buffers.get(teacher, deque())
            signals = list(buf)

        if lived_only:
            signals = [s for s in signals if not _is_synthetic_origin(s.origin)]
        if min_fidelity > 0:
            signals = [s for s in signals if s.fidelity >= min_fidelity]

        return signals[-limit:]

    def get_latest(self, teacher: str) -> TeacherSignal | None:
        """Return the most recent signal for a teacher, or None."""
        with self._lock:
            buf = self._buffers.get(teacher, deque())
            return buf[-1] if buf else None

    def count(self, teacher: str) -> int:
        with self._lock:
            return len(self._buffers.get(teacher, deque()))

    def get_stats(self) -> dict[str, Any]:
        """Dashboard-safe snapshot of collector state."""
        with self._lock:
            now = time.time()
            per_teacher: dict[str, dict] = {}
            for teacher in set(list(self._counts.keys()) + list(self._quarantine_counts.keys())):
                _tot = self._counts.get(teacher, 0)
                _syn = self._synthetic_counts.get(teacher, 0)
                per_teacher[teacher] = {
                    "total": _tot,
                    "quarantined": self._quarantine_counts.get(teacher, 0),
                    "buffer_size": len(self._buffers.get(teacher, deque())),
                    "synthetic": _syn,
                    "lived": max(0, _tot - _syn),
                    "last_seen_s": round(now - self._last_seen[teacher], 1) if teacher in self._last_seen else None,
                }
            _all = sum(self._counts.values())
            _all_syn = sum(self._synthetic_counts.values())
            return {
                "teachers": per_teacher,
                "total_signals": _all,
                "total_quarantined": sum(self._quarantine_counts.values()),
                "total_synthetic": _all_syn,
                "total_lived": max(0, _all - _all_syn),
            }

    _MAX_JSONL_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    @classmethod
    def _write_jsonl(cls, path: Path, signal: TeacherSignal) -> None:
        try:
            row: dict[str, Any] = {
                "t": round(signal.timestamp, 3),
                "teacher": signal.teacher,
                "type": signal.signal_type,
                "origin": signal.origin,
                "fidelity": round(signal.fidelity, 4),
            }
            if isinstance(signal.data, list):
                row["data"] = [round(v, 6) for v in signal.data]
            else:
                row["data"] = signal.data
            if signal.metadata:
                row["meta"] = signal.metadata
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            cls._maybe_rotate(path)
        except OSError:
            logger.warning("Distillation write failed", exc_info=True)

    @classmethod
    def _maybe_rotate(cls, path: Path) -> None:
        """Trim JSONL to last half when file exceeds size limit."""
        try:
            if not path.exists():
                return
            size = path.stat().st_size
            if size <= cls._MAX_JSONL_SIZE_BYTES:
                return
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2 :]
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
            logger.info("Rotated %s: %d→%d lines (was %.1fMB)",
                        path.name, len(lines), len(keep), size / 1024 / 1024)
        except OSError:
            logger.warning("Distillation rotation failed", exc_info=True)


    def review_quarantine(self) -> int:
        """Review quarantined samples and promote if corroborated.

        Called during dream/sleep mode. If a quarantined sample's teacher
        has since produced high-confidence corroborating signals (same
        teacher, fidelity >= 0.7 within 10s window), promote the sample
        from quarantine to the training buffer.

        Returns count of promoted samples.
        """
        promoted = 0
        with self._lock:
            for teacher, q_buf in list(self._quarantine.items()):
                t_buf = self._buffers.get(teacher, deque())
                if not t_buf:
                    continue

                to_promote: list[TeacherSignal] = []
                for q_sig in list(q_buf):
                    for t_sig in t_buf:
                        if (abs(t_sig.timestamp - q_sig.timestamp) < 10.0
                                and t_sig.fidelity >= 0.7):
                            to_promote.append(q_sig)
                            break

                for sig in to_promote:
                    try:
                        q_buf.remove(sig)
                    except ValueError:
                        continue
                    main_buf = self._buffers.setdefault(teacher, deque(maxlen=BUFFER_MAXLEN))
                    main_buf.append(sig)
                    self._counts[teacher] = self._counts.get(teacher, 0) + 1
                    self._write_jsonl(TRAINING_DATA_DIR / f"distill_{teacher}.jsonl", sig)
                    promoted += 1

        if promoted:
            logger.info("Quarantine review: promoted %d sample(s)", promoted)
        return promoted

    def check_consensus(self, speaker_name: str, face_name: str) -> float:
        """Check voice-face identity consensus.

        Returns 1.0 if both agree, 0.0 if conflict, 0.5 if one is unknown.
        """
        if speaker_name == "unknown" and face_name == "unknown":
            return 0.5
        if speaker_name == "unknown" or face_name == "unknown":
            return 0.5
        speaker_base = speaker_name.split("_")[0].lower() if "_" not in speaker_name else speaker_name.lower()
        face_base = face_name.split("_")[0].lower() if "_" not in face_name else face_name.lower()
        if speaker_base == face_base:
            return 1.0
        return 0.0


distillation_collector = DistillationCollector.get_instance()
