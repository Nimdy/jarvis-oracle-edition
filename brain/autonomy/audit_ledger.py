"""Durable audit subscriber for autonomy / L3 escalation events.

Phase 6.5: governance and autonomy transitions must survive restarts
as an auditable trail. The EscalationStore already persists its own
lifecycle transitions, but bus-only signals
(AUTONOMY_L3_ELIGIBLE, AUTONOMY_L3_PROMOTED,
AUTONOMY_L3_ACTIVATION_DENIED, AUTONOMY_LEVEL_CHANGED) had no durable
home. This module closes that gap: it subscribes to every autonomy /
escalation event on the bus and appends a structured record to
``~/.jarvis/autonomy_audit.jsonl``.

The ledger is a read-only observer. It never emits events and never
influences the promotion / escalation control path. Disk failures are
logged and swallowed so audit faults never block cognition.

On-disk layout: one JSON object per line, keys:
  - ``ts``       : unix seconds (float)
  - ``event``    : event-name string (e.g. ``autonomy:l3_promoted``)
  - ``payload``  : the event's keyword payload as emitted
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from consciousness.events import (
    AUTONOMY_ESCALATION_APPROVED,
    AUTONOMY_ESCALATION_EXPIRED,
    AUTONOMY_ESCALATION_PARKED,
    AUTONOMY_ESCALATION_REJECTED,
    AUTONOMY_ESCALATION_REQUESTED,
    AUTONOMY_ESCALATION_ROLLED_BACK,
    AUTONOMY_L3_ACTIVATION_DENIED,
    AUTONOMY_L3_ELIGIBLE,
    AUTONOMY_L3_PROMOTED,
    AUTONOMY_LEVEL_CHANGED,
    event_bus,
)

logger = logging.getLogger(__name__)

_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
_AUDIT_PATH = _JARVIS_DIR / "autonomy_audit.jsonl"

MAX_LOG_SIZE_MB = 10

# The exact set of events this subscriber persists. Kept deliberately
# small and explicit: adding events here is a governance decision.
AUDITED_EVENTS: tuple[str, ...] = (
    AUTONOMY_LEVEL_CHANGED,
    AUTONOMY_L3_ELIGIBLE,
    AUTONOMY_L3_PROMOTED,
    AUTONOMY_L3_ACTIVATION_DENIED,
    AUTONOMY_ESCALATION_REQUESTED,
    AUTONOMY_ESCALATION_APPROVED,
    AUTONOMY_ESCALATION_REJECTED,
    AUTONOMY_ESCALATION_ROLLED_BACK,
    AUTONOMY_ESCALATION_PARKED,
    AUTONOMY_ESCALATION_EXPIRED,
)


class AutonomyAuditLedger:
    """Subscribe to autonomy/escalation events and persist them as JSONL.

    The ledger is a read-only observer of the event bus. It writes a
    single JSON line per event to ``autonomy_audit.jsonl`` under
    ``~/.jarvis`` (overridable via ``path=`` for tests or
    ``JARVIS_HOME``). Rotation mimics the memory lifecycle log: when
    the file exceeds ``MAX_LOG_SIZE_MB`` the newest half is kept.

    Thread-safe: a single lock guards writes and rotation. Intended to
    be wired once at boot and unwired on shutdown / in tests.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _AUDIT_PATH
        self._lock = threading.Lock()
        self._cleanups: list[Callable[[], None]] = []
        self._wired: bool = False
        self._events_recorded: int = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def wired(self) -> bool:
        return self._wired

    @property
    def events_recorded(self) -> int:
        return self._events_recorded

    def wire(self, bus: Any = None) -> None:
        """Subscribe to all AUDITED_EVENTS. Idempotent."""
        if self._wired:
            return
        target = bus if bus is not None else event_bus
        for event_name in AUDITED_EVENTS:
            handler = self._make_handler(event_name)
            cleanup = target.on(event_name, handler)
            self._cleanups.append(cleanup)
        self._wired = True
        logger.info(
            "Autonomy audit ledger wired (%d events) path=%s",
            len(self._cleanups), self._path,
        )

    def unwire(self) -> None:
        for cleanup in self._cleanups:
            try:
                cleanup()
            except Exception:
                logger.debug("Audit ledger cleanup failed", exc_info=True)
        self._cleanups.clear()
        self._wired = False

    def load_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Read the newest ``limit`` audit entries from disk.

        Newest-last, oldest-first order matches file order. Corrupt
        lines are skipped.
        """
        if not self._path.exists():
            return []
        try:
            text = self._path.read_text()
        except OSError as exc:
            logger.warning("Audit ledger read failed at %s: %s", self._path, exc)
            return []
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tail = lines[-max(1, int(limit)):] if limit > 0 else lines
        out: list[dict[str, Any]] = []
        for ln in tail:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    def get_stats(self) -> dict[str, Any]:
        exists = self._path.exists()
        size_kb = 0.0
        if exists:
            try:
                size_kb = round(self._path.stat().st_size / 1024.0, 1)
            except OSError:
                size_kb = 0.0
        return {
            "wired": self._wired,
            "events_recorded_session": self._events_recorded,
            "log_exists": exists,
            "log_size_kb": size_kb,
            "path": str(self._path),
        }

    # -- internals ---------------------------------------------------------

    def _make_handler(self, event_name: str) -> Callable[..., None]:
        def handler(**payload: Any) -> None:
            self._append(event_name, payload)
        return handler

    def _append(self, event_name: str, payload: dict[str, Any]) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "event": event_name,
            "payload": _sanitize(payload),
        }
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    self._path.exists()
                    and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024
                ):
                    self._rotate_locked()
                with open(self._path, "a") as fh:
                    fh.write(json.dumps(entry, default=str) + "\n")
                self._events_recorded += 1
            except Exception as exc:
                logger.warning("Audit ledger write failed: %s", exc)

    def _rotate_locked(self) -> None:
        try:
            lines = self._path.read_text().splitlines()
            keep = lines[len(lines) // 2:]
            self._path.write_text("\n".join(keep) + ("\n" if keep else ""))
            logger.info(
                "Rotated autonomy audit ledger: %d -> %d entries",
                len(lines), len(keep),
            )
        except Exception:
            logger.debug("Audit ledger rotation failed", exc_info=True)


def _sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the payload is JSON-serialisable; coerce unknowns to str.

    We do not call ``default=str`` on the top-level json.dumps alone
    because some callers emit dataclasses or Path objects we want
    normalised to short strings rather than repr blobs.
    """
    clean: dict[str, Any] = {}
    for k, v in payload.items():
        try:
            json.dumps(v)
            clean[k] = v
        except TypeError:
            if isinstance(v, (list, tuple, set)):
                clean[k] = [str(x) for x in v]
            else:
                clean[k] = str(v)
    return clean


_ledger: AutonomyAuditLedger | None = None


def get_audit_ledger() -> AutonomyAuditLedger:
    """Module-level singleton accessor."""
    global _ledger
    if _ledger is None:
        _ledger = AutonomyAuditLedger()
    return _ledger


def reset_audit_ledger_for_tests() -> None:
    """Reset the singleton. Tests only — never call from production."""
    global _ledger
    if _ledger is not None:
        _ledger.unwire()
    _ledger = None
