"""CueGate — single authority for memory access policy.

Default-CLOSED gate with explicit open/close transitions and full logging.
Replaces scattered stance checks across observer, consciousness_system, and
search with a single queryable authority.

Three access classes:
  - READ: memory retrieval (search, recall). Open during waking, research,
    and dream consolidation. Closed only during hypothetical future modes.
  - OBSERVATION_WRITE: incidental writes from observer delta effects (salience
    boosts, association weight changes). Blocked during dreaming/reflective.
  - CONSOLIDATION_WRITE: intentional writes by the dream cycle (associate,
    reinforce, decay, consolidation summaries). Always allowed when the dream
    cycle is active.

The gate tracks depth (re-entrant sessions), transition history, and exposes
stats for the dashboard.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_TRANSITION_HISTORY_LIMIT = 200


class MemoryGate:
    """Single authority for all memory access policy decisions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._last_opened_at = 0.0
        self._last_closed_at = 0.0
        self._total_opens = 0
        self._transitions: list[dict[str, Any]] = []

        self._observation_writes_allowed = True
        self._consolidation_active = False
        self._current_mode = "passive"
        self._synthetic_sources: set[str] = set()

    # ------------------------------------------------------------------
    # Core gate queries
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        """Whether a read session is currently active."""
        with self._lock:
            return self._depth > 0

    def can_read(self) -> bool:
        """Whether memory retrieval is currently permitted."""
        return True

    def synthetic_session_active(self) -> bool:
        """Whether a synthetic perception exercise session is currently active.

        Truth-boundary invariant: synthetic TTS audio flowing through the real
        perception pipeline must not create lived-history artifacts (memory,
        identity records, rapport data). While a synthetic session is active,
        all memory writes are blocked at the gate.
        """
        with self._lock:
            return bool(self._synthetic_sources)

    def can_observation_write(self) -> bool:
        """Whether incidental observer writes (delta effects) are allowed.

        Blocked during dreaming and reflective modes to prevent dream state
        from contaminating canonical memory through observation side-effects.
        Also blocked during synthetic perception exercise sessions so that
        synthetic audio cannot contaminate lived-history memory.
        """
        with self._lock:
            if self._synthetic_sources:
                return False
            return self._observation_writes_allowed

    def can_consolidation_write(self) -> bool:
        """Whether dream consolidation writes are allowed.

        Always True when consolidation is active — the dream cycle's entire
        purpose is to write (associate, reinforce, consolidate). Exception:
        synthetic perception exercise sessions always block writes regardless
        of consolidation state.
        """
        with self._lock:
            if self._synthetic_sources:
                return False
            return self._consolidation_active

    # ------------------------------------------------------------------
    # Mode transitions — called by the system when mode changes
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Update the gate's internal mode policy.

        Called by perception_orchestrator or consciousness_system on mode change.
        """
        with self._lock:
            old_mode = self._current_mode
            self._current_mode = mode

            old_obs = self._observation_writes_allowed
            self._observation_writes_allowed = mode not in (
                "dreaming", "sleep", "reflective", "deep_learning",
            )

            if old_obs != self._observation_writes_allowed:
                action = "obs_writes_enabled" if self._observation_writes_allowed else "obs_writes_blocked"
                self._record_transition_unlocked(action, f"mode:{old_mode}->{mode}", "gate_policy")

    # ------------------------------------------------------------------
    # Consolidation lifecycle — opened/closed by dream cycle
    # ------------------------------------------------------------------

    def begin_consolidation(self, reason: str = "dream_cycle") -> None:
        """Mark the start of a dream consolidation phase.

        While active, all consolidation writes (associate, reinforce, decay,
        summary creation) are permitted regardless of mode.
        """
        with self._lock:
            self._consolidation_active = True
            self._record_transition_unlocked("consolidation_begin", reason, "dream_cycle")

    def end_consolidation(self, reason: str = "dream_cycle") -> None:
        """Mark the end of a dream consolidation phase."""
        with self._lock:
            self._consolidation_active = False
            self._record_transition_unlocked("consolidation_end", reason, "dream_cycle")

    @contextmanager
    def consolidation_window(self, reason: str = "dream_cycle") -> Iterator[None]:
        """RAII guard for consolidation phases."""
        self.begin_consolidation(reason)
        try:
            yield
        finally:
            self.end_consolidation(reason)

    # ------------------------------------------------------------------
    # Synthetic exercise lifecycle — called by perception_orchestrator
    # when a synthetic_exercise_start/end control message arrives.
    # ------------------------------------------------------------------

    def begin_synthetic_session(self, source: str = "synthetic") -> None:
        """Mark the start of a synthetic perception exercise session.

        While active, all observation and consolidation writes are blocked
        at the gate. Set-based to support multiple concurrent synthetic
        sources (paralleling ``PerceptionOrchestrator._synthetic_sources``).
        """
        with self._lock:
            was_empty = not self._synthetic_sources
            self._synthetic_sources.add(source or "synthetic")
            if was_empty:
                self._record_transition_unlocked(
                    "synthetic_begin", source or "synthetic", "synthetic_exercise",
                )

    def end_synthetic_session(self, source: str = "synthetic") -> None:
        """Mark the end of a synthetic perception exercise session.

        Removes the named source from the active set. Writes become permitted
        again only when the set is empty.
        """
        with self._lock:
            self._synthetic_sources.discard(source or "synthetic")
            if not self._synthetic_sources:
                self._record_transition_unlocked(
                    "synthetic_end", source or "synthetic", "synthetic_exercise",
                )

    # ------------------------------------------------------------------
    # Read sessions (observability + RAII)
    # ------------------------------------------------------------------

    def open(self, reason: str, actor: str = "") -> None:
        with self._lock:
            self._depth += 1
            self._total_opens += 1
            self._last_opened_at = time.time()
            self._record_transition_unlocked("open", reason, actor)

    def close(self, reason: str, actor: str = "") -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            self._last_closed_at = time.time()
            self._record_transition_unlocked("close", reason, actor)

    @contextmanager
    def session(self, reason: str, actor: str = "") -> Iterator[None]:
        """RAII guard for read sessions."""
        self.open(reason, actor=actor)
        try:
            yield
        finally:
            self.close(reason, actor=actor)

    # ------------------------------------------------------------------
    # Stats / dashboard
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "is_open": self._depth > 0,
                "depth": self._depth,
                "total_opens": self._total_opens,
                "last_opened_at": self._last_opened_at,
                "last_closed_at": self._last_closed_at,
                "observation_writes_allowed": self._observation_writes_allowed,
                "consolidation_active": self._consolidation_active,
                "synthetic_active": bool(self._synthetic_sources),
                "synthetic_sources": sorted(self._synthetic_sources),
                "current_mode": self._current_mode,
                "recent_transitions": list(self._transitions[-20:]),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_transition_unlocked(self, action: str, reason: str, actor: str) -> None:
        self._transitions.append({
            "ts": time.time(),
            "action": action,
            "reason": reason[:80],
            "actor": actor[:40],
            "depth": self._depth,
            "obs_writes": self._observation_writes_allowed,
            "consolidation": self._consolidation_active,
        })
        if len(self._transitions) > _TRANSITION_HISTORY_LIMIT:
            self._transitions = self._transitions[-_TRANSITION_HISTORY_LIMIT:]


memory_gate = MemoryGate()
