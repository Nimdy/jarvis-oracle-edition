"""Identity audit system for Layer 3: Identity Boundary Engine.

Tracks identity boundary decisions, quarantine writes, and referenced-subject
exceptions for observability and debugging.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from consciousness.events import (
    IDENTITY_AMBIGUITY_DETECTED,
    IDENTITY_BOUNDARY_BLOCKED,
    IDENTITY_SCOPE_ASSIGNED,
)


@dataclass(frozen=True)
class IdentityAuditEvent:
    timestamp: float
    event_type: str
    proposed_scope: dict[str, Any] = field(default_factory=dict)
    final_scope: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    memory_id: str = ""


class IdentityAudit:
    """Ring-buffer audit trail for identity boundary decisions."""

    _instance: IdentityAudit | None = None
    _MAX_EVENTS = 100

    def __init__(self) -> None:
        self._events: deque[IdentityAuditEvent] = deque(maxlen=self._MAX_EVENTS)
        self._total_quarantined = 0
        self._total_downgraded = 0
        self._total_referenced_allows = 0
        self._total_boundary_blocks = 0
        self._total_scope_assigned = 0
        self._by_owner_type: dict[str, int] = {}
        self._by_subject_type: dict[str, int] = {}

    @classmethod
    def get_instance(cls) -> IdentityAudit:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(self, event: IdentityAuditEvent) -> None:
        self._events.append(event)

        if event.event_type == "quarantine_write":
            self._total_quarantined += 1
            ot = event.final_scope.get("owner_type", "unknown")
            st = event.final_scope.get("subject_type", "unknown")
            self._by_owner_type[ot] = self._by_owner_type.get(ot, 0) + 1
            self._by_subject_type[st] = self._by_subject_type.get(st, 0) + 1
            self._emit_event(IDENTITY_AMBIGUITY_DETECTED, event)
        elif event.event_type == "scope_downgraded":
            self._total_downgraded += 1
        elif event.event_type == "referenced_subject_allow":
            self._total_referenced_allows += 1
        elif event.event_type == "boundary_blocked":
            self._total_boundary_blocks += 1
            self._emit_event(IDENTITY_BOUNDARY_BLOCKED, event)
        elif event.event_type == "scope_assigned":
            self._total_scope_assigned += 1
            ot = event.final_scope.get("owner_type", "unknown")
            st = event.final_scope.get("subject_type", "unknown")
            self._by_owner_type[ot] = self._by_owner_type.get(ot, 0) + 1
            self._by_subject_type[st] = self._by_subject_type.get(st, 0) + 1
            self._emit_event(IDENTITY_SCOPE_ASSIGNED, event)

    @staticmethod
    def _emit_event(event_name: str, event: IdentityAuditEvent) -> None:
        try:
            from consciousness.events import event_bus
            event_bus.emit(event_name,
                           memory_id=event.memory_id,
                           reason=event.reason,
                           confidence=event.confidence)
        except Exception:
            pass

    def record_scope_assigned(
        self, memory_id: str, scope: Any, confidence: float,
    ) -> None:
        final = {}
        if scope:
            final = {
                "owner_id": getattr(scope, "owner_id", ""),
                "owner_type": getattr(scope, "owner_type", ""),
                "subject_id": getattr(scope, "subject_id", ""),
                "subject_type": getattr(scope, "subject_type", ""),
            }
        needs_res = getattr(scope, "needs_resolution", False) if scope else False

        self.record(IdentityAuditEvent(
            timestamp=time.time(),
            event_type="quarantine_write" if needs_res else "scope_assigned",
            final_scope=final,
            confidence=confidence,
            memory_id=memory_id,
        ))

    def record_boundary_block(
        self, memory_id: str, reason: str, querier_id: str = "",
    ) -> None:
        self.record(IdentityAuditEvent(
            timestamp=time.time(),
            event_type="boundary_blocked",
            reason=reason,
            memory_id=memory_id,
            evidence_refs=(querier_id,) if querier_id else (),
        ))

    def record_referenced_allow(
        self, memory_id: str, subject_id: str, querier_id: str = "",
    ) -> None:
        self.record(IdentityAuditEvent(
            timestamp=time.time(),
            event_type="referenced_subject_allow",
            reason=f"subject={subject_id}",
            memory_id=memory_id,
            evidence_refs=(querier_id,) if querier_id else (),
        ))

    def get_recent(self, count: int = 10) -> list[dict]:
        events = list(self._events)[-count:]
        return [
            {
                "ts": e.timestamp,
                "type": e.event_type,
                "reason": e.reason,
                "confidence": e.confidence,
                "memory_id": e.memory_id,
            }
            for e in events
        ]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_events": len(self._events),
            "total_quarantined": self._total_quarantined,
            "total_downgraded": self._total_downgraded,
            "total_referenced_allows": self._total_referenced_allows,
            "total_boundary_blocks": self._total_boundary_blocks,
            "total_scope_assigned": self._total_scope_assigned,
            "by_owner_type": dict(self._by_owner_type),
            "by_subject_type": dict(self._by_subject_type),
        }


identity_audit = IdentityAudit.get_instance()
