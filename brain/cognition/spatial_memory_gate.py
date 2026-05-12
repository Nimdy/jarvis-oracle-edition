"""Spatial memory gate — tightly gated episode promotion through CueGate.

This is the ONLY path from spatial intelligence into memory.  Every spatial
memory write must pass through this gate, which enforces:

  1. CueGate observation_write permission (mode-aware)
  2. Per-hour and per-day budget limits
  3. Minimum confidence threshold (CONFIDENCE_THRESHOLD_MEMORY)
  4. Human-relevance or repetition requirement
  5. Calibration version and provenance attached to every write

No raw spatial data (observations, tracks, coordinates) ever reaches memory.
Only compact, meaningful, human-relevant episodes are candidates.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from cognition.spatial_schema import (
    CONFIDENCE_THRESHOLD_MEMORY,
    SPATIAL_MEMORY_MAX_PER_DAY,
    SPATIAL_MEMORY_MAX_PER_HOUR,
    SpatialDelta,
)

logger = logging.getLogger(__name__)

_REPETITION_THRESHOLD = 2
_RELEVANCE_KEYWORDS = frozenset({
    "moved", "missing", "reappeared", "distance_changed",
})


class SpatialMemoryGate:
    """Gates spatial episode promotion into memory.

    Enforces all non-negotiable memory safety invariants from the
    Spatial Intelligence Phase 1 plan.
    """

    def __init__(self) -> None:
        self._hour_writes: deque[float] = deque()
        self._day_writes: deque[float] = deque()
        self._repetition_counts: dict[str, int] = {}
        self._total_promoted: int = 0
        self._total_blocked: int = 0
        self._block_reasons: dict[str, int] = {}

    def can_promote(
        self,
        delta: SpatialDelta,
        memory_gate: Any = None,
    ) -> tuple[bool, str]:
        """Check if a spatial delta qualifies for memory promotion.

        Returns (allowed, reason).
        """
        if memory_gate is not None:
            if not memory_gate.can_observation_write():
                self._record_block("cuegate_blocked")
                return False, "CueGate: observation writes not allowed in current mode"

        if delta.confidence < CONFIDENCE_THRESHOLD_MEMORY:
            self._record_block("low_confidence")
            return False, f"Confidence {delta.confidence:.3f} < {CONFIDENCE_THRESHOLD_MEMORY}"

        if not delta.validated:
            self._record_block("not_validated")
            return False, "Delta not validated"

        if delta.calibration_version == 0:
            self._record_block("no_calibration")
            return False, "No calibration version"

        now = time.time()
        self._prune_windows(now)

        if len(self._hour_writes) >= SPATIAL_MEMORY_MAX_PER_HOUR:
            self._record_block("hour_budget_exceeded")
            return False, f"Hourly budget exhausted ({SPATIAL_MEMORY_MAX_PER_HOUR}/hr)"

        if len(self._day_writes) >= SPATIAL_MEMORY_MAX_PER_DAY:
            self._record_block("day_budget_exceeded")
            return False, f"Daily budget exhausted ({SPATIAL_MEMORY_MAX_PER_DAY}/day)"

        is_relevant = delta.delta_type in _RELEVANCE_KEYWORDS
        entity_key = f"{delta.entity_id}:{delta.delta_type}"
        repeat_count = self._repetition_counts.get(entity_key, 0)
        is_repeated = repeat_count >= _REPETITION_THRESHOLD

        if not is_relevant and not is_repeated:
            self._repetition_counts[entity_key] = repeat_count + 1
            self._record_block("not_relevant_or_repeated")
            return False, "Not human-relevant and not yet repeated enough"

        return True, "ok"

    def record_promotion(self, delta: SpatialDelta) -> None:
        """Record that a spatial episode was promoted to memory."""
        now = time.time()
        self._hour_writes.append(now)
        self._day_writes.append(now)
        self._total_promoted += 1
        entity_key = f"{delta.entity_id}:{delta.delta_type}"
        self._repetition_counts.pop(entity_key, None)

    def build_memory_content(self, delta: SpatialDelta) -> str:
        """Build a compact, human-readable memory string from a spatial delta.

        This is what actually gets stored — not coordinates.
        """
        label = delta.label
        dtype = delta.delta_type.replace("_", " ")

        if delta.delta_type == "moved" and delta.distance_m > 0:
            dist_approx = f"~{delta.distance_m:.1f}m"
            return f"{label} {dtype} {dist_approx} ({delta.dominant_axis} axis)"

        if delta.delta_type == "missing":
            return f"{label} vanished from its usual position"

        if delta.delta_type == "reappeared":
            return f"{label} reappeared in the workspace"

        return f"{label} spatial state changed: {dtype}"

    def get_state(self) -> dict[str, Any]:
        now = time.time()
        self._prune_windows(now)
        return {
            "total_promoted": self._total_promoted,
            "total_blocked": self._total_blocked,
            "hour_writes": len(self._hour_writes),
            "day_writes": len(self._day_writes),
            "hour_budget": SPATIAL_MEMORY_MAX_PER_HOUR,
            "day_budget": SPATIAL_MEMORY_MAX_PER_DAY,
            "block_reasons": dict(self._block_reasons),
            "pending_repetitions": len(self._repetition_counts),
        }

    def _prune_windows(self, now: float) -> None:
        hour_ago = now - 3600
        day_ago = now - 86400
        while self._hour_writes and self._hour_writes[0] < hour_ago:
            self._hour_writes.popleft()
        while self._day_writes and self._day_writes[0] < day_ago:
            self._day_writes.popleft()

    def _record_block(self, reason: str) -> None:
        self._total_blocked += 1
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1
