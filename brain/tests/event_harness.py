"""Event record/replay harness for integration testing.

Usage:

    # Record all events during a live session
    recorder = EventRecorder()
    recorder.start()
    # ... run system ...
    recorder.stop()
    recorder.save("session_001.jsonl")

    # Replay recorded events for deterministic testing
    replayer = EventReplayer("session_001.jsonl")
    await replayer.replay()              # real-time pacing
    await replayer.replay(speed=10.0)    # 10x fast-forward
    replayer.replay_instant()            # no delays
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from consciousness.events import event_bus

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.path.expanduser("~/.jarvis/event_logs")


class EventRecorder:
    """Attaches to EventBus.emit and logs every event to a JSONL buffer."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._active = False
        self._original_emit: Callable[..., Any] | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        if self._active:
            return
        self._start_time = time.time()
        self._records.clear()
        self._original_emit = event_bus.emit
        bus = event_bus

        def _recording_emit(event_type: str, **kwargs: Any) -> None:
            self._records.append({
                "t": round(time.time() - self._start_time, 4),
                "event": event_type,
                "data": _serialize(kwargs),
            })
            if self._original_emit:
                self._original_emit(event_type, **kwargs)

        bus.emit = _recording_emit  # type: ignore[assignment]
        self._active = True
        logger.info("EventRecorder started")

    def stop(self) -> None:
        if not self._active:
            return
        if self._original_emit:
            event_bus.emit = self._original_emit  # type: ignore[assignment]
        self._active = False
        logger.info("EventRecorder stopped — %d events captured", len(self._records))

    def save(self, filename: str, directory: str = _DEFAULT_DIR) -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record, default=str) + "\n")
        logger.info("Saved %d events to %s", len(self._records), path)
        return path

    @property
    def event_count(self) -> int:
        return len(self._records)


class EventReplayer:
    """Replays a JSONL event log through the EventBus."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        logger.info("Loaded %d events from %s", len(self._records), self._path)

    @property
    def event_count(self) -> int:
        return len(self._records)

    async def replay(self, speed: float = 1.0) -> int:
        """Replay events with real-time pacing (scaled by speed)."""
        if not self._records:
            return 0

        count = 0
        prev_t = 0.0
        for record in self._records:
            t = record.get("t", 0.0)
            delay = (t - prev_t) / speed
            if delay > 0:
                await asyncio.sleep(delay)
            prev_t = t
            event_bus.emit(record["event"], **record.get("data", {}))
            count += 1
        logger.info("Replayed %d events (speed=%.1fx)", count, speed)
        return count

    def replay_instant(self) -> int:
        """Replay all events with no delays (for unit tests)."""
        count = 0
        for record in self._records:
            event_bus.emit(record["event"], **record.get("data", {}))
            count += 1
        return count

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type is None:
            return list(self._records)
        return [r for r in self._records if r.get("event") == event_type]

    def get_timeline(self) -> list[tuple[float, str]]:
        return [(r.get("t", 0.0), r.get("event", "")) for r in self._records]


def _serialize(data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort serialization of event kwargs to JSON-safe types."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, (list, tuple)):
            result[k] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in v]
        elif isinstance(v, dict):
            result[k] = _serialize(v)
        elif hasattr(v, "to_dict"):
            result[k] = v.to_dict()
        else:
            result[k] = str(v)
    return result
