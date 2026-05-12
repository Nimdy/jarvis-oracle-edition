"""Minimal trace context helpers for release-path lineage.

This module is intentionally small and deterministic:
- derive stable trace/request IDs from conversation_id when available
- generate safe fallback IDs when conversation_id is missing
- expose event-field payloads for release-path emits
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceContext:
    """Release-path correlation context."""

    conversation_id: str
    trace_id: str
    request_id: str

    def as_event_fields(self) -> dict[str, str]:
        return {
            "conversation_id": self.conversation_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
        }


def _normalize_conversation_id(conversation_id: str) -> str:
    normalized = (conversation_id or "").strip()
    if normalized:
        return normalized
    return f"conv_{time.monotonic_ns()}"


def build_trace_context(conversation_id: str = "") -> TraceContext:
    """Create a release-path context from conversation_id.

    - trace_id is stable per conversation_id
    - request_id is unique per context construction
    """
    normalized = _normalize_conversation_id(conversation_id)
    trace_digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    request_seed = f"{normalized}:{time.monotonic_ns()}"
    request_digest = hashlib.sha1(request_seed.encode("utf-8")).hexdigest()[:12]
    return TraceContext(
        conversation_id=normalized,
        trace_id=f"trc_{trace_digest}",
        request_id=f"req_{request_digest}",
    )

