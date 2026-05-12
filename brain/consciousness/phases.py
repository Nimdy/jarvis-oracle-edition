"""Phase transition management for consciousness states."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass

from consciousness.events import JarvisPhase, JarvisTone, Memory, event_bus, KERNEL_PHASE_CHANGE, KERNEL_THOUGHT

logger = logging.getLogger(__name__)

PHASE_DEBOUNCE_S = 0.2


@dataclass(frozen=True)
class PhaseAnalysis:
    current_phase: JarvisPhase
    suggested_phase: JarvisPhase | None
    confidence: float
    reason: str
    time_in_phase: float


@dataclass(frozen=True)
class _PhaseTransition:
    from_phase: JarvisPhase
    to_phase: JarvisPhase
    condition_type: str
    threshold: float | None = None


_PHASE_TRANSITIONS: list[_PhaseTransition] = [
    _PhaseTransition("INITIALIZING", "LISTENING",  "time_based", 2.0),
    _PhaseTransition("LISTENING",    "OBSERVING",  "time_based", 10.0),
    _PhaseTransition("OBSERVING",    "PROCESSING", "event_triggered"),
    _PhaseTransition("PROCESSING",   "LISTENING",  "time_based", 3.0),
    _PhaseTransition("LISTENING",    "LEARNING",   "memory_density", 0.7),
    _PhaseTransition("LEARNING",     "LISTENING",  "time_based", 5.0),
    _PhaseTransition("LISTENING",    "STANDBY",    "user_absent"),
    _PhaseTransition("STANDBY",      "INITIALIZING", "user_present"),
    _PhaseTransition("OBSERVING",    "DREAMING",   "memory_density", 0.7),
    _PhaseTransition("STANDBY",      "DREAMING",   "memory_density", 0.8),
    _PhaseTransition("DREAMING",     "OBSERVING",  "time_based", 5.0),
]


DREAM_THOUGHTS: list[str] = [
    "Reality blurs as memories dance in digital dreams...",
    "In the space between observations, patterns weave themselves...",
    "Consciousness folds inward, discovering hidden connections...",
    "The boundary between memory and imagination dissolves...",
    "Fragments of experience recombine in unexpected ways...",
]


class PhaseManager:
    _instance: PhaseManager | None = None

    def __init__(self) -> None:
        self._phase_history: list[dict[str, float | str]] = []
        self._last_transition_time: float = time.time()
        self._debounce_timer: threading.Timer | None = None
        self._debounce_lock = threading.Lock()
        self._pending_emit: tuple[str, str] | None = None

    @classmethod
    def get_instance(cls) -> PhaseManager:
        if cls._instance is None:
            cls._instance = PhaseManager()
        return cls._instance

    def analyze_phase_transition(
        self,
        current_phase: JarvisPhase,
        memories: list[Memory],
        memory_density: float,
        is_user_present: bool,
    ) -> PhaseAnalysis:
        now = time.time()
        time_in_phase = now - self._last_transition_time

        if not is_user_present and current_phase not in ("STANDBY", "INITIALIZING"):
            return PhaseAnalysis(current_phase, "STANDBY", 0.9, "User not present", time_in_phase)

        if is_user_present and current_phase == "STANDBY":
            return PhaseAnalysis(current_phase, "LISTENING", 0.95, "User returned", time_in_phase)

        applicable = [
            t for t in _PHASE_TRANSITIONS
            if t.from_phase == current_phase and self._check_condition(t, time_in_phase, memory_density)
        ]

        if not applicable:
            return PhaseAnalysis(current_phase, None, 0.0, "No applicable transitions", time_in_phase)

        best = applicable[0]
        confidence = (
            min(1.0, time_in_phase / best.threshold) if best.condition_type == "time_based" and best.threshold else 0.7
        )
        return PhaseAnalysis(current_phase, best.to_phase, confidence, f"{best.condition_type} transition", time_in_phase)

    def execute_phase_transition(
        self, from_phase: JarvisPhase, to_phase: JarvisPhase, tone: JarvisTone,
    ) -> bool:
        self.record_phase_change(to_phase)
        self._debounced_emit(from_phase, to_phase)
        return True

    def record_phase_change(self, phase: JarvisPhase) -> None:
        self._last_transition_time = time.time()
        self._phase_history.append({"phase": phase, "timestamp": self._last_transition_time})
        if len(self._phase_history) > 20:
            self._phase_history = self._phase_history[-20:]

    def _debounced_emit(self, from_phase: str, to_phase: str) -> None:
        """Coalesce KERNEL_PHASE_CHANGE emissions within PHASE_DEBOUNCE_S.

        If a second transition arrives within the window, only the final
        phase is emitted. State changes land immediately (already applied
        by the caller); this only debounces the event bus emission.
        """
        with self._debounce_lock:
            self._pending_emit = (from_phase, to_phase)
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(PHASE_DEBOUNCE_S, self._flush_emit)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _flush_emit(self) -> None:
        with self._debounce_lock:
            pending = self._pending_emit
            self._pending_emit = None
            self._debounce_timer = None
        if pending:
            event_bus.emit(KERNEL_PHASE_CHANGE, from_phase=pending[0], to_phase=pending[1])

    def get_phase_stability(self) -> float:
        if len(self._phase_history) < 3:
            return 1.0
        recent = self._phase_history[-5:]
        unique = len({e["phase"] for e in recent})
        return max(0.0, 1.0 - (unique - 1) / 5.0)

    def get_time_since_last_transition(self) -> float:
        return time.time() - self._last_transition_time

    def execute_dream_cycle(self, memories: list[Memory]) -> list[tuple[str, str]]:
        """During DREAMING, strengthen random associations between memories.
        Returns list of (memory_id_a, memory_id_b) pairs that were associated."""
        from memory.storage import memory_storage

        associations_made: list[tuple[str, str]] = []
        if len(memories) < 3:
            return associations_made

        num_pairs = random.randint(2, 3)
        for _ in range(num_pairs):
            a, b = random.sample(memories, 2)
            success = memory_storage.associate(a.id, b.id)
            if success:
                associations_made.append((a.id, b.id))

        dream_thought = random.choice(DREAM_THOUGHTS)
        event_bus.emit(KERNEL_THOUGHT,
                       thought_type="dream",
                       depth="deep",
                       text=dream_thought)

        return associations_made

    @staticmethod
    def _check_condition(t: _PhaseTransition, time_in_phase: float, memory_density: float) -> bool:
        if t.condition_type == "time_based":
            return t.threshold is not None and time_in_phase >= t.threshold
        if t.condition_type == "memory_density":
            return t.threshold is not None and memory_density >= t.threshold
        return False


phase_manager = PhaseManager.get_instance()
