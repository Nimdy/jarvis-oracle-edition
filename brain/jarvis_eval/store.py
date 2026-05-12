"""Append-only JSONL store for eval sidecar data.

The eval sidecar is a read-only observer. This store writes ONLY to
~/.jarvis/eval/ — it never touches memory, beliefs, dreams, or policy.

All write methods are best-effort: exceptions are logged, never raised.
File rotation at MAX_JSONL_SIZE_MB with .1 suffix.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from jarvis_eval.config import (
    EVAL_DIR,
    EVENTS_FILE,
    MAX_JSONL_SIZE_MB,
    META_FILE,
    RUNS_FILE,
    SCORES_FILE,
    SCORING_VERSION,
    SCENARIO_PACK_VERSION,
    SCORECARDS_FILE,
    SNAPSHOTS_FILE,
)
from jarvis_eval.contracts import EvalEvent, EvalRun, EvalScore, EvalScorecard, EvalSnapshot

logger = logging.getLogger(__name__)


class EvalStore:
    """Append-only JSONL persistence for eval data."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._dir = base_dir or EVAL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._events_path = self._dir / EVENTS_FILE
        self._snapshots_path = self._dir / SNAPSHOTS_FILE
        self._scores_path = self._dir / SCORES_FILE
        self._scorecards_path = self._dir / SCORECARDS_FILE
        self._runs_path = self._dir / RUNS_FILE
        self._meta_path = self._dir / META_FILE

        self._total_events_written: int = 0
        self._total_snapshots_written: int = 0
        self._total_scorecards_written: int = 0
        self._dropped_event_count: int = 0
        self._rotation_counts: dict[str, int] = {
            "events": 0, "snapshots": 0, "scores": 0, "scorecards": 0,
        }
        self._last_flush_ts: float | None = None
        self._last_scorecard_ts: float | None = None
        self._created_at: float = time.time()

        self._load_meta()

    def _load_meta(self) -> None:
        """Restore counters from eval_meta.json if it exists."""
        try:
            if self._meta_path.exists():
                with open(self._meta_path) as f:
                    m = json.load(f)
                self._total_events_written = m.get("total_events_written", 0)
                self._total_snapshots_written = m.get("total_snapshots_written", 0)
                self._total_scorecards_written = m.get("total_scorecards_written", 0)
                self._dropped_event_count = m.get("dropped_event_count", 0)
                self._rotation_counts = m.get("rotation_counts", self._rotation_counts)
                self._last_flush_ts = m.get("last_flush_ts")
                self._last_scorecard_ts = m.get("last_scorecard_ts")
                self._created_at = m.get("created_at", self._created_at)
        except Exception:
            logger.warning("Failed to load eval meta, starting fresh")

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> bool:
        """Append one JSON line. Returns True on success."""
        try:
            self._maybe_rotate(path)
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            return True
        except Exception:
            logger.warning("Eval store: failed to append to %s", path.name, exc_info=True)
            return False

    def _maybe_rotate(self, path: Path) -> None:
        """Rotate file if it exceeds size threshold."""
        try:
            if path.exists() and path.stat().st_size > MAX_JSONL_SIZE_MB * 1024 * 1024:
                rotated = path.with_suffix(path.suffix + ".1")
                if rotated.exists():
                    rotated.unlink()
                path.rename(rotated)
                key = path.stem.replace("eval_", "")
                self._rotation_counts[key] = self._rotation_counts.get(key, 0) + 1
                logger.info("Eval store: rotated %s", path.name)
        except Exception:
            logger.warning("Eval store: rotation failed for %s", path.name, exc_info=True)

    def append_event(self, event: EvalEvent) -> None:
        with self._lock:
            if self._append_jsonl(self._events_path, event.to_dict()):
                self._total_events_written += 1
            else:
                self._dropped_event_count += 1

    def append_snapshot(self, snapshot: EvalSnapshot) -> None:
        with self._lock:
            if self._append_jsonl(self._snapshots_path, snapshot.to_dict()):
                self._total_snapshots_written += 1

    def append_score(self, score: EvalScore) -> None:
        with self._lock:
            self._append_jsonl(self._scores_path, score.to_dict())

    def append_scorecard(self, scorecard: EvalScorecard) -> None:
        with self._lock:
            if self._append_jsonl(self._scorecards_path, scorecard.to_dict()):
                self._total_scorecards_written += 1
                self._last_scorecard_ts = scorecard.timestamp

    def append_run(self, run: EvalRun) -> None:
        with self._lock:
            self._append_jsonl(self._runs_path, run.to_dict())

    def close_run(self, run: EvalRun) -> None:
        """Append a run-closed record with ended_at timestamp."""
        with self._lock:
            self._append_jsonl(self._runs_path, run.to_dict())

    def flush_meta(self) -> None:
        """Write eval_meta.json atomically."""
        meta = {
            "scoring_version": SCORING_VERSION,
            "scenario_pack_version": SCENARIO_PACK_VERSION,
            "created_at": self._created_at,
            "schema_version": 1,
            "rotation_counts": self._rotation_counts,
            "total_events_written": self._total_events_written,
            "total_snapshots_written": self._total_snapshots_written,
            "total_scorecards_written": self._total_scorecards_written,
            "dropped_event_count": self._dropped_event_count,
            "last_flush_ts": time.time(),
            "last_scorecard_ts": self._last_scorecard_ts,
        }
        try:
            from memory.persistence import atomic_write_json
            atomic_write_json(self._meta_path, meta)
            self._last_flush_ts = meta["last_flush_ts"]
        except Exception:
            logger.warning("Eval store: failed to write meta", exc_info=True)

    def read_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._read_tail(self._events_path, limit)

    def read_all_events(self) -> list[dict[str, Any]]:
        """Read all events — used once at startup for PVL hydration."""
        return self._read_tail(self._events_path, limit=999_999)

    def read_recent_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._read_tail(self._snapshots_path, limit)

    def read_recent_scores(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._read_tail(self._scores_path, limit)

    def read_recent_scorecards(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._read_tail(self._scorecards_path, limit)

    def _read_tail(self, path: Path, limit: int) -> list[dict[str, Any]]:
        """Read last N records from a JSONL file."""
        if not path.exists():
            return []
        try:
            records: list[dict[str, Any]] = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
            return records[-limit:]
        except Exception:
            logger.warning("Eval store: failed to read %s", path.name, exc_info=True)
            return []

    def get_file_sizes(self) -> dict[str, int]:
        """Return byte sizes for each JSONL file."""
        sizes: dict[str, int] = {}
        for name, path in [
            ("events", self._events_path),
            ("snapshots", self._snapshots_path),
            ("scores", self._scores_path),
            ("scorecards", self._scorecards_path),
            ("runs", self._runs_path),
        ]:
            try:
                sizes[name] = path.stat().st_size if path.exists() else 0
            except Exception:
                sizes[name] = 0
        return sizes

    def get_meta(self) -> dict[str, Any]:
        return {
            "scoring_version": SCORING_VERSION,
            "scenario_pack_version": SCENARIO_PACK_VERSION,
            "created_at": self._created_at,
            "schema_version": 1,
            "rotation_counts": dict(self._rotation_counts),
            "total_events_written": self._total_events_written,
            "total_snapshots_written": self._total_snapshots_written,
            "total_scorecards_written": self._total_scorecards_written,
            "dropped_event_count": self._dropped_event_count,
            "last_flush_ts": self._last_flush_ts,
            "last_scorecard_ts": self._last_scorecard_ts,
        }
