"""Library retrieval telemetry — logs chunk usage for future retriever NN training.

Two-step correlation pattern:
  1. log_retrieval_started() — emitted when context builder resolves pointer memories
  2. log_retrieval_outcome() — emitted when conversation completes/fails/barge-in

Both are joined later by conversation_id to produce (query, chunks, outcome) triples
for training a reranking NN.

Storage: append-only JSONL at ~/.jarvis/library/retrieval_log.jsonl
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
TELEMETRY_PATH = JARVIS_DIR / "library" / "retrieval_log.jsonl"
MAX_LOG_SIZE_MB = 10
_START_BUFFER_CAP = 100


class RetrievalTelemetry:
    """Append-only JSONL logger for chunk retrieval events."""

    _instance: RetrievalTelemetry | None = None

    def __init__(self, path: str | Path = "") -> None:
        self._path = Path(path) if path else TELEMETRY_PATH
        self._lock = threading.Lock()
        self._initialized = False
        self._total_starts = 0
        self._total_outcomes = 0
        self._recent_starts: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @classmethod
    def get_instance(cls) -> RetrievalTelemetry:
        if cls._instance is None:
            cls._instance = RetrievalTelemetry()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def log_retrieval_started(
        self,
        conversation_id: str,
        query: str,
        source_ids: list[str],
        chunk_ids_surfaced: list[str],
        chunk_ids_injected: list[str],
    ) -> None:
        """Log which chunks were surfaced and injected into the LLM context."""
        entry = {
            "type": "retrieval_start",
            "conversation_id": conversation_id,
            "query": query[:200],
            "source_ids": source_ids[:10],
            "chunk_ids_surfaced": chunk_ids_surfaced[:20],
            "chunk_ids_injected": chunk_ids_injected[:10],
            "timestamp": time.time(),
        }
        self._append(entry)
        self._total_starts += 1

        if conversation_id:
            self._recent_starts[conversation_id] = entry
            self._recent_starts.move_to_end(conversation_id)
            if len(self._recent_starts) > _START_BUFFER_CAP:
                self._recent_starts.popitem(last=False)

    def log_retrieval_outcome(
        self,
        conversation_id: str,
        outcome: str,
        latency_ms: float = 0.0,
        user_signal: str = "",
    ) -> None:
        """Log the conversation outcome for correlation with retrieval_start.

        Args:
            outcome: "ok", "error", "barge_in", "timeout", "cancelled"
            user_signal: optional quality signal ("follow_up", "clarification", "satisfied")
        """
        self._append({
            "type": "retrieval_outcome",
            "conversation_id": conversation_id,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 1),
            "user_signal": user_signal,
            "timestamp": time.time(),
        })
        self._total_outcomes += 1

    def get_latest_start(self, conversation_id: str) -> dict[str, Any] | None:
        """O(1) lookup of the most recent retrieval_start for a conversation."""
        return self._recent_starts.get(conversation_id)

    def get_stats(self) -> dict[str, Any]:
        recent = self.read_recent(20)
        outcomes = [e for e in recent if e.get("type") == "retrieval_outcome"]
        ok_count = sum(1 for e in outcomes if e.get("outcome") == "ok")
        return {
            "total_starts": self._total_starts,
            "total_outcomes": self._total_outcomes,
            "recent_ok": ok_count,
            "recent_total": len(outcomes),
            "recent_entries": recent[-10:],
            "log_exists": self._path.exists(),
            "log_size_kb": round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
        }

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read the most recent log entries (for dashboard / debugging)."""
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text().strip().split("\n")
            entries = []
            for line in lines[-limit:]:
                if line.strip():
                    entries.append(json.loads(line))
            return entries
        except Exception:
            return []

    def get_training_pairs(self, limit: int = 500) -> list[dict[str, Any]]:
        """Join retrieval_start + retrieval_outcome by conversation_id.

        Returns matched pairs suitable for retriever NN training:
        {query, chunk_ids_injected, outcome, latency_ms}
        """
        entries = self.read_recent(limit * 2)
        starts: dict[str, dict[str, Any]] = {}
        pairs: list[dict[str, Any]] = []

        for entry in entries:
            if entry.get("type") == "retrieval_start":
                cid = entry.get("conversation_id", "")
                if cid:
                    starts[cid] = entry
            elif entry.get("type") == "retrieval_outcome":
                cid = entry.get("conversation_id", "")
                if cid and cid in starts:
                    start = starts.pop(cid)
                    pairs.append({
                        "query": start.get("query", ""),
                        "chunk_ids_injected": start.get("chunk_ids_injected", []),
                        "source_ids": start.get("source_ids", []),
                        "outcome": entry.get("outcome", ""),
                        "latency_ms": entry.get("latency_ms", 0),
                        "user_signal": entry.get("user_signal", ""),
                    })

        return pairs[-limit:]

    def _append(self, entry: dict[str, Any]) -> None:
        if not self._initialized:
            self.init()
        with self._lock:
            try:
                if self._path.exists() and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
                    self._rotate()
                with open(self._path, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception as exc:
                logger.debug("Telemetry write failed: %s", exc)

    def _rotate(self) -> None:
        """Keep only the last half of entries when the log gets too large."""
        try:
            lines = self._path.read_text().strip().split("\n")
            keep = lines[len(lines) // 2:]
            self._path.write_text("\n".join(keep) + "\n")
            logger.info("Rotated retrieval log: %d -> %d entries", len(lines), len(keep))
        except Exception:
            pass


retrieval_telemetry = RetrievalTelemetry.get_instance()
