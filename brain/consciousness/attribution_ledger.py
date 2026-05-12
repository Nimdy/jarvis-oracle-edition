"""Attribution Ledger — canonical epistemic truth layer.

Every significant action across all subsystems (conversation, capability gate,
autonomy, learning jobs, self-improvement, consciousness mutations, perception)
writes a structured LedgerEntry here.

Design invariants:
  - Append-only on disk: entries and outcomes are separate JSONL lines
  - In-memory: outcomes are folded onto base entries during rehydration
  - Thread-safe: all reads/writes protected by a single lock
  - Hot path: record() is O(1)
  - Causal chains: root_entry_id + parent_entry_id enable full lineage traversal
  - Evidence refs: normalized cross-references for downstream immune layers

Outcome schema (Layer 4 — Delayed Outcome Attribution):
  - Standardized outcome values: success, failure, stable, regressed, inconclusive, partial
  - Every outcome carries: confidence, latency_s, source, tier inside outcome_data
  - OutcomeScheduler handles delayed checks (Tier 3) with retry-once semantics

Future consumers: contradiction engine, trust calibration, soul integrity index.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

_LEDGER_DIR = os.path.expanduser("~/.jarvis")
_LEDGER_PATH = os.path.join(_LEDGER_DIR, "attribution_ledger.jsonl")
_MAX_FILE_MB = 20
_MAX_BUFFER = 2000
_REHYDRATE_MAX = 500

# ---------------------------------------------------------------------------
# Standardized outcome vocabulary
# ---------------------------------------------------------------------------

OutcomeStatus = Literal[
    "success", "failure", "stable", "regressed", "inconclusive", "partial",
]

OutcomeSource = Literal[
    "system_metric", "user_feedback", "verification_test",
    "health_monitor", "delta_tracker",
]

OutcomeTier = Literal["immediate", "medium", "delayed"]


OutcomeScope = Literal[
    "response_quality", "autonomy_policy", "mutation_health",
    "retrieval_selection", "skill_verification", "general",
]

BlameTarget = Literal[
    "intent_selection", "timing", "tool_choice",
    "source_quality", "memory_content", "response_generation",
    "general",
]


def build_outcome_data(
    *,
    confidence: float = 1.0,
    latency_s: float = 0.0,
    source: str = "system_metric",
    tier: str = "immediate",
    scope: str = "general",
    blame_target: str = "general",
    **extra: Any,
) -> dict[str, Any]:
    """Build a standardized outcome_data dict.

    All callers should use this so downstream consumers (contradiction engine,
    trust calibration, reward models) get consistent fields.

    scope: which subsystem the outcome evaluates (prevents cross-layer
           label pollution — e.g. autonomy_policy regressions should not
           penalize memory provenance).
    blame_target: what specifically failed (intent_selection, timing,
                  source_quality, etc.).
    """
    d: dict[str, Any] = {
        "outcome_confidence": round(min(max(confidence, 0.0), 1.0), 4),
        "latency_s": round(latency_s, 2),
        "outcome_source": source,
        "outcome_tier": tier,
        "outcome_scope": scope,
        "blame_target": blame_target,
    }
    d.update(extra)
    return d


def _short_id() -> str:
    return "led_" + uuid.uuid4().hex[:12]


@dataclass
class LedgerEntry:
    """Single canonical event record across all subsystems."""
    entry_id: str
    root_entry_id: str
    parent_entry_id: str
    timestamp: float
    subsystem: str
    event_type: str
    actor: str
    source: str
    confidence: float
    conversation_id: str
    evidence_refs: list[dict[str, str]]
    data: dict[str, Any]
    outcome: str = "pending"
    outcome_ts: float = 0.0
    outcome_data: dict[str, Any] = field(default_factory=dict)

    def to_record_dict(self) -> dict[str, Any]:
        """Serialize for JSONL persistence (entry record, no outcome fields)."""
        return {
            "type": "entry",
            "entry_id": self.entry_id,
            "root_entry_id": self.root_entry_id,
            "parent_entry_id": self.parent_entry_id,
            "ts": self.timestamp,
            "subsystem": self.subsystem,
            "event_type": self.event_type,
            "actor": self.actor,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "conversation_id": self.conversation_id,
            "evidence_refs": self.evidence_refs,
            "data": self.data,
        }

    def to_snapshot_dict(self) -> dict[str, Any]:
        """Full dict including in-memory outcome state."""
        d = self.to_record_dict()
        d["outcome"] = self.outcome
        d["outcome_ts"] = self.outcome_ts
        d["outcome_data"] = self.outcome_data
        return d

    @classmethod
    def from_record_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            entry_id=d.get("entry_id", ""),
            root_entry_id=d.get("root_entry_id", ""),
            parent_entry_id=d.get("parent_entry_id", ""),
            timestamp=d.get("ts", 0.0),
            subsystem=d.get("subsystem", ""),
            event_type=d.get("event_type", ""),
            actor=d.get("actor", "unknown"),
            source=d.get("source", ""),
            confidence=d.get("confidence", 0.0),
            conversation_id=d.get("conversation_id", ""),
            evidence_refs=d.get("evidence_refs", []),
            data=d.get("data", {}),
        )


class AttributionLedger:
    """Singleton append-only attribution ledger.

    All subsystems call record() to log structured events.
    record_outcome() appends a separate outcome line (never mutates history).
    rehydrate() folds outcomes onto base entries in memory at boot.
    """

    _instance: AttributionLedger | None = None

    @classmethod
    def get_instance(cls) -> AttributionLedger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: OrderedDict[str, LedgerEntry] = OrderedDict()
        self._total_recorded: int = 0
        self._total_outcomes: int = 0
        self._subsystem_counts: dict[str, int] = {}
        self._outcome_counts: dict[str, int] = {}
        self._scope_counts: dict[str, int] = {}
        self._rehydrated: bool = False
        self._errors: int = 0
        self._last_error: str = ""
        os.makedirs(_LEDGER_DIR, exist_ok=True)

    def record(
        self,
        subsystem: str,
        event_type: str,
        actor: str = "system",
        source: str = "",
        confidence: float = 1.0,
        conversation_id: str = "",
        evidence_refs: list[dict[str, str]] | None = None,
        data: dict[str, Any] | None = None,
        parent_entry_id: str = "",
        root_entry_id: str = "",
    ) -> str:
        """Record a new ledger entry. Returns the entry_id."""
        eid = _short_id()
        if not root_entry_id:
            root_entry_id = parent_entry_id if parent_entry_id else eid

        entry = LedgerEntry(
            entry_id=eid,
            root_entry_id=root_entry_id,
            parent_entry_id=parent_entry_id,
            timestamp=time.time(),
            subsystem=subsystem,
            event_type=event_type,
            actor=actor,
            source=source,
            confidence=confidence,
            conversation_id=conversation_id,
            evidence_refs=evidence_refs or [],
            data=data or {},
        )

        with self._lock:
            self._buffer[eid] = entry
            if len(self._buffer) > _MAX_BUFFER:
                self._buffer.popitem(last=False)
            self._total_recorded += 1
            self._subsystem_counts[subsystem] = self._subsystem_counts.get(subsystem, 0) + 1

        self._append_jsonl(entry.to_record_dict())
        self._emit_bridge(entry)
        return eid

    def record_outcome(
        self,
        entry_id: str,
        outcome: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append an outcome record linked to a base entry.

        Never mutates the original entry on disk — appends a separate line.
        In-memory, folds the outcome onto the base entry for fast queries.
        Emits OUTCOME_RESOLVED so calibration Bridge 3 receives immediate outcomes.
        """
        now = time.time()
        outcome_record = {
            "type": "outcome",
            "entry_id": entry_id,
            "outcome": outcome,
            "ts": now,
            "data": data or {},
        }

        subsystem = ""
        with self._lock:
            if entry_id in self._buffer:
                self._buffer[entry_id].outcome = outcome
                self._buffer[entry_id].outcome_ts = now
                self._buffer[entry_id].outcome_data = data or {}
                subsystem = self._buffer[entry_id].subsystem
            self._total_outcomes += 1
            self._outcome_counts[outcome] = self._outcome_counts.get(outcome, 0) + 1
            scope = (data or {}).get("outcome_scope", "general")
            self._scope_counts[scope] = self._scope_counts.get(scope, 0) + 1

        self._append_jsonl(outcome_record)

        if outcome not in ("pending",):
            try:
                from consciousness.events import event_bus, OUTCOME_RESOLVED
                _data = data or {}
                event_bus.emit(
                    OUTCOME_RESOLVED,
                    entry_id=entry_id,
                    subsystem=subsystem,
                    outcome=outcome,
                    confidence=_data.get("outcome_confidence"),
                    user_signal=_data.get("user_signal", ""),
                )
            except Exception:
                pass

    def query(
        self,
        subsystem: str = "",
        event_type: str = "",
        actor: str = "",
        since: float = 0.0,
        root_entry_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Cold-path query over in-memory buffer. Returns newest-first."""
        with self._lock:
            entries = list(self._buffer.values())

        results = []
        for e in reversed(entries):
            if subsystem and e.subsystem != subsystem:
                continue
            if event_type and e.event_type != event_type:
                continue
            if actor and e.actor != actor:
                continue
            if since and e.timestamp < since:
                continue
            if root_entry_id and e.root_entry_id != root_entry_id:
                continue
            results.append(e.to_snapshot_dict())
            if len(results) >= limit:
                break
        return results

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the n most recent entries as dicts."""
        with self._lock:
            entries = list(self._buffer.values())
        return [e.to_snapshot_dict() for e in entries[-n:]][::-1]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_recorded": self._total_recorded,
                "total_outcomes": self._total_outcomes,
                "buffer_size": len(self._buffer),
                "subsystem_counts": dict(self._subsystem_counts),
                "outcome_counts": dict(self._outcome_counts),
                "scope_counts": dict(self._scope_counts),
                "errors": self._errors,
                "last_error": self._last_error,
                "rehydrated": self._rehydrated,
            }

    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        with self._lock:
            return self._buffer.get(entry_id)

    def get_chain(self, root_entry_id: str) -> list[dict[str, Any]]:
        """Return all entries in a causal chain, ordered by timestamp."""
        with self._lock:
            entries = [
                e.to_snapshot_dict()
                for e in self._buffer.values()
                if e.root_entry_id == root_entry_id
            ]
        entries.sort(key=lambda x: x.get("ts", 0))
        return entries

    def rehydrate(self, max_entries: int = _REHYDRATE_MAX) -> int:
        """Replay JSONL into in-memory buffer at boot. Returns entries loaded."""
        if self._rehydrated:
            return 0
        self._rehydrated = True

        if not os.path.exists(_LEDGER_PATH):
            return 0

        raw_entries: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        try:
            with open(_LEDGER_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") == "outcome":
                        outcomes.append(record)
                    elif record.get("type") == "entry":
                        raw_entries.append(record)
        except Exception as e:
            logger.warning("Ledger rehydration failed: %s", e)
            return 0

        tail = raw_entries[-max_entries:]
        with self._lock:
            for rd in tail:
                entry = LedgerEntry.from_record_dict(rd)
                self._buffer[entry.entry_id] = entry
                self._subsystem_counts[entry.subsystem] = (
                    self._subsystem_counts.get(entry.subsystem, 0) + 1
                )

            for oc in outcomes:
                eid = oc.get("entry_id", "")
                if eid in self._buffer:
                    self._buffer[eid].outcome = oc.get("outcome", "")
                    self._buffer[eid].outcome_ts = oc.get("ts", 0.0)
                    self._buffer[eid].outcome_data = oc.get("data", {})
                    outcome_str = oc.get("outcome", "")
                    self._outcome_counts[outcome_str] = (
                        self._outcome_counts.get(outcome_str, 0) + 1
                    )

            self._total_recorded = len(self._buffer)
            self._total_outcomes = sum(self._outcome_counts.values())

        loaded = len(tail)
        logger.info("Ledger rehydrated: %d entries, %d outcomes", loaded, len(outcomes))
        return loaded

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            self._maybe_rotate()
            with open(_LEDGER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        except Exception as e:
            self._errors += 1
            self._last_error = f"{type(e).__name__}: {e}"
            if self._errors <= 3:
                logger.warning("Ledger write error: %s", e)

    def _maybe_rotate(self) -> None:
        try:
            if not os.path.exists(_LEDGER_PATH):
                return
            size = os.path.getsize(_LEDGER_PATH)
            if size < _MAX_FILE_MB * 1024 * 1024:
                return

            with open(_LEDGER_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            half = len(lines) // 2
            with open(_LEDGER_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[half:])
            logger.info("Ledger rotated: kept %d of %d lines", len(lines) - half, len(lines))
        except Exception as e:
            logger.warning("Ledger rotation failed: %s", e)

    def _emit_bridge(self, entry: LedgerEntry) -> None:
        """Lightweight EventBus bridge so other subsystems can subscribe."""
        try:
            from consciousness.events import event_bus, ATTRIBUTION_ENTRY_RECORDED
            event_bus.emit(
                ATTRIBUTION_ENTRY_RECORDED,
                entry_id=entry.entry_id,
                subsystem=entry.subsystem,
                event_type=entry.event_type,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OutcomeScheduler — delayed outcome checks (Tier 3)
# ---------------------------------------------------------------------------

_SCHEDULER_MAX_PENDING = 200


@dataclass
class PendingOutcome:
    """A delayed outcome check waiting to be evaluated."""
    entry_id: str
    check_at: float
    check_fn: Callable[[], tuple[str, dict[str, Any]] | None]
    subsystem: str
    description: str
    retries: int = 0
    max_retries: int = 1


class OutcomeScheduler:
    """Schedules delayed outcome checks and resolves them on tick().

    Design:
      - schedule() queues a PendingOutcome with a future check_at timestamp
      - tick() evaluates all due checks; if check_fn returns a result, stamps
        the ledger via record_outcome(); if None, retries once then marks
        inconclusive
      - Max 200 pending (FIFO eviction)
      - NOT persisted across restarts (transient by design)
      - Thread-safe via its own lock
    """

    _instance: OutcomeScheduler | None = None

    @classmethod
    def get_instance(cls) -> OutcomeScheduler:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[PendingOutcome] = deque(maxlen=_SCHEDULER_MAX_PENDING)
        self._resolved: int = 0
        self._inconclusive: int = 0
        self._errors: int = 0
        self._evicted: int = 0

    def schedule(
        self,
        entry_id: str,
        delay_s: float,
        check_fn: Callable[[], tuple[str, dict[str, Any]] | None],
        subsystem: str = "",
        description: str = "",
        max_retries: int = 1,
    ) -> None:
        """Queue a delayed outcome check."""
        po = PendingOutcome(
            entry_id=entry_id,
            check_at=time.time() + delay_s,
            check_fn=check_fn,
            subsystem=subsystem,
            description=description,
            max_retries=max_retries,
        )
        with self._lock:
            if len(self._pending) >= _SCHEDULER_MAX_PENDING:
                self._evicted += 1
            self._pending.append(po)

    def tick(self) -> int:
        """Evaluate all due checks. Returns number of outcomes resolved."""
        now = time.time()
        due: list[PendingOutcome] = []
        remaining: deque[PendingOutcome] = deque(maxlen=_SCHEDULER_MAX_PENDING)

        with self._lock:
            for po in self._pending:
                if po.check_at <= now:
                    due.append(po)
                else:
                    remaining.append(po)
            self._pending = remaining

        resolved = 0
        requeue: list[PendingOutcome] = []
        ledger = AttributionLedger.get_instance()

        for po in due:
            try:
                result = po.check_fn()
            except Exception as exc:
                self._errors += 1
                if self._errors <= 5:
                    logger.warning("OutcomeScheduler check_fn error for %s: %s",
                                   po.entry_id, exc)
                result = None

            if result is not None:
                outcome_str, outcome_data = result
                entry = ledger.get_entry(po.entry_id)
                entry_ts = entry.timestamp if entry else now
                outcome_data.setdefault("outcome_tier", "delayed")
                outcome_data.setdefault("latency_s", round(now - entry_ts, 2))
                ledger.record_outcome(po.entry_id, outcome_str, outcome_data)
                self._resolved += 1
                resolved += 1
            else:
                if po.retries < po.max_retries:
                    po.retries += 1
                    po.check_at = now + 60.0
                    requeue.append(po)
                else:
                    entry = ledger.get_entry(po.entry_id)
                    entry_ts = entry.timestamp if entry else now
                    ledger.record_outcome(po.entry_id, "inconclusive", build_outcome_data(
                        confidence=0.0,
                        latency_s=round(now - entry_ts, 2),
                        source="system_metric",
                        tier="delayed",
                        scope="general",
                        blame_target="general",
                        reason="check_returned_none",
                        description=po.description,
                    ))
                    self._inconclusive += 1
                    resolved += 1

        if requeue:
            with self._lock:
                self._pending.extend(requeue)

        return resolved

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            pending_by_subsystem: dict[str, int] = {}
            for po in self._pending:
                pending_by_subsystem[po.subsystem] = pending_by_subsystem.get(po.subsystem, 0) + 1
            return {
                "pending": len(self._pending),
                "resolved": self._resolved,
                "inconclusive": self._inconclusive,
                "errors": self._errors,
                "evicted": self._evicted,
                "pending_by_subsystem": pending_by_subsystem,
            }

attribution_ledger = AttributionLedger.get_instance()
outcome_scheduler = OutcomeScheduler.get_instance()
