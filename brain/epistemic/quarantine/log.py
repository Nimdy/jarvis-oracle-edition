"""Quarantine Candidate Log — Layer 8 Shadow Mode.

Append-only JSONL persistence + in-memory ring buffer.
Rotation at 10MB. No blocking, no mutation — observation only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.expanduser("~/.jarvis/quarantine_candidates.jsonl")
_MAX_SIZE_BYTES = 10 * 1024 * 1024
_RING_SIZE = 200


class QuarantineLog:
    """Append-only JSONL log for quarantine candidate signals."""

    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._ring: deque[dict[str, Any]] = deque(maxlen=_RING_SIZE)
        self._total_logged: int = 0

    def record(self, signal_dict: dict[str, Any]) -> None:
        entry = {
            "ts": time.time(),
            **signal_dict,
        }
        self._ring.append(entry)
        self._total_logged += 1

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._maybe_rotate()
            with open(self._path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            logger.debug("Failed to write quarantine log entry", exc_info=True)

    def _maybe_rotate(self) -> None:
        try:
            if self._path.exists() and self._path.stat().st_size > _MAX_SIZE_BYTES:
                rotated = self._path.with_suffix(".jsonl.1")
                if rotated.exists():
                    rotated.unlink()
                self._path.rename(rotated)
                logger.info("Rotated quarantine log: %s -> %s", self._path, rotated)
        except Exception:
            logger.debug("Quarantine log rotation failed", exc_info=True)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._ring)[-limit:]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_logged": self._total_logged,
            "ring_size": len(self._ring),
            "path": str(self._path),
        }
