"""Consciousness heartbeat — budget-aware async tick loop with priority queues."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from consciousness.events import (
    JarvisPhase, JarvisTone, Memory, event_bus,
    KERNEL_TICK, KERNEL_ERROR, MEMORY_DECAY_CYCLE, TONE_SHIFT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing configuration (static scheduling intervals)
# ---------------------------------------------------------------------------

KERNEL_INTERVALS = {
    "DEFAULT_INTERVAL_S": 0.1,
    "THINKING_INTERVAL_S": 120.0,
    "TONE_CHECK_INTERVAL_S": 10.0,
    "PHASE_CHECK_INTERVAL_S": 5.0,
    "DECAY_CYCLE_INTERVAL_S": 60.0,
    "MAINTENANCE_INTERVAL_S": 300.0,
    "META_THOUGHT_INTERVAL_S": 8.0,
    "PROACTIVE_CHECK_INTERVAL_S": 10.0,
    "AUTONOMY_INTERVAL_S": 30.0,
}

# Budget defaults — sized for Python + background consciousness work.
# Most ticks are sub-1ms; budget should only flag genuine anomalies.
DEFAULT_BUDGET_MS = 50.0
IDLE_BUDGET_MS = 100.0
LOAD_BUDGET_MS = 25.0
BUDGET_WARNING_THRESHOLD_MS = 40.0
SLOW_TICK_THRESHOLD_MS = 30.0
PERFORMANCE_WINDOW_SIZE = 120


# ---------------------------------------------------------------------------
# Priority queue system
# ---------------------------------------------------------------------------

class Priority(enum.IntEnum):
    REALTIME = 0
    INTERACTIVE = 1
    BACKGROUND = 2


@dataclass
class DeferredOp:
    priority: Priority
    name: str
    callback: Callable[[], None]
    timestamp: float


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------

@dataclass
class BudgetResult:
    total_time_ms: float
    operations: list[str]
    over_budget: bool


class BudgetTracker:
    __slots__ = ("_budget_ms", "_start", "_used", "_ops")

    def __init__(self, budget_ms: float = DEFAULT_BUDGET_MS) -> None:
        self._budget_ms = budget_ms
        self._start = 0.0
        self._used = 0.0
        self._ops: list[str] = []

    @property
    def budget_ms(self) -> float:
        return self._budget_ms

    @budget_ms.setter
    def budget_ms(self, value: float) -> None:
        self._budget_ms = max(1.0, min(200.0, value))

    def start(self) -> None:
        self._start = time.perf_counter()
        self._used = 0.0
        self._ops = []

    def checkpoint(self, op_name: str) -> bool:
        elapsed = (time.perf_counter() - self._start) * 1000.0
        self._used = elapsed
        self._ops.append(op_name)
        return elapsed < self._budget_ms

    def remaining(self) -> float:
        elapsed = (time.perf_counter() - self._start) * 1000.0
        return max(0.0, self._budget_ms - elapsed)

    def finish(self) -> BudgetResult:
        total = (time.perf_counter() - self._start) * 1000.0
        return BudgetResult(
            total_time_ms=total,
            operations=list(self._ops),
            over_budget=total > self._budget_ms,
        )


# ---------------------------------------------------------------------------
# Performance metrics (O(1) rolling window)
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    tick_count: int = 0
    slow_ticks: int = 0
    budget_overruns: int = 0
    avg_tick_ms: float = 0.0
    p95_tick_ms: float = 0.0
    max_tick_ms: float = 0.0
    deferred_backlog: int = 0


class PerformanceMonitor:
    def __init__(self, window: int = PERFORMANCE_WINDOW_SIZE) -> None:
        self._tick_times: deque[float] = deque(maxlen=window)
        self._tick_count = 0
        self._slow_ticks = 0
        self._overruns = 0
        self._sorted_dirty = True
        self._sorted_cache: list[float] = []

    def record(self, tick_ms: float, over_budget: bool) -> None:
        self._tick_times.append(tick_ms)
        self._tick_count += 1
        if tick_ms > SLOW_TICK_THRESHOLD_MS:
            self._slow_ticks += 1
        if over_budget:
            self._overruns += 1
        self._sorted_dirty = True

    def get_p95(self) -> float:
        if not self._tick_times:
            return 0.0
        if self._sorted_dirty:
            self._sorted_cache = sorted(self._tick_times)
            self._sorted_dirty = False
        idx = int(len(self._sorted_cache) * 0.95)
        return self._sorted_cache[min(idx, len(self._sorted_cache) - 1)]

    def get_avg(self) -> float:
        if not self._tick_times:
            return 0.0
        return sum(self._tick_times) / len(self._tick_times)

    def get_metrics(self, deferred_backlog: int = 0) -> PerformanceMetrics:
        return PerformanceMetrics(
            tick_count=self._tick_count,
            slow_ticks=self._slow_ticks,
            budget_overruns=self._overruns,
            avg_tick_ms=self.get_avg(),
            p95_tick_ms=self.get_p95(),
            max_tick_ms=max(self._tick_times) if self._tick_times else 0.0,
            deferred_backlog=deferred_backlog,
        )


# ---------------------------------------------------------------------------
# Kernel state
# ---------------------------------------------------------------------------

@dataclass
class KernelState:
    tick: int = 0
    is_running: bool = False
    is_paused: bool = False
    last_think_time: float = 0.0
    last_tone_check: float = 0.0
    last_phase_check: float = 0.0
    last_decay_time: float = 0.0
    last_maintenance_time: float = 0.0
    last_meta_thought_time: float = 0.0
    last_proactive_check: float = 0.0
    last_autonomy_tick: float = 0.0


# ---------------------------------------------------------------------------
# Kernel callbacks protocol
# ---------------------------------------------------------------------------

class KernelCallbacks(Protocol):
    def get_kernel_state(self) -> dict[str, Any]: ...
    def update_state(self, **updates: Any) -> None: ...
    def decay_memories(self) -> int: ...
    def perform_maintenance(self) -> None: ...
    def calculate_memory_density(self) -> None: ...
    def on_thinking_cycle(self) -> None: ...
    def on_trait_modulation(self) -> None: ...
    def on_consciousness_tick(self) -> None: ...
    def on_proactive_check(self) -> None: ...
    def on_autonomy_tick(self) -> None: ...


# ---------------------------------------------------------------------------
# Budget-aware kernel loop
# ---------------------------------------------------------------------------

class KernelLoop:
    _CADENCE_MIN = 0.5
    _CADENCE_MAX = 2.0

    def __init__(self, callbacks: KernelCallbacks) -> None:
        self._callbacks = callbacks
        self._state = KernelState()
        self._task: asyncio.Task[None] | None = None
        self._budget = BudgetTracker(DEFAULT_BUDGET_MS)
        self._perf = PerformanceMonitor()
        self._deferred: list[DeferredOp] = []
        self._max_deferred = 30
        self._cadence_multiplier: float = 1.0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        if self._state.is_running:
            return False

        now = time.time()
        self._state = KernelState(
            is_running=True,
            last_think_time=now,
            last_tone_check=now,
            last_phase_check=now,
            last_decay_time=now,
            last_maintenance_time=now,
            last_meta_thought_time=now,
            last_proactive_check=now,
            last_autonomy_tick=now,
        )
        self._deferred.clear()

        self._task = asyncio.get_event_loop().create_task(self._run())
        logger.info("Budget-aware consciousness loop started (budget=%.1fms)", self._budget.budget_ms)
        return True

    def stop(self) -> None:
        if not self._state.is_running:
            return
        self._state.is_running = False
        self._state.is_paused = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._log_final_stats()
        logger.info("Consciousness loop stopped")

    def pause(self) -> bool:
        if not self._state.is_running or self._state.is_paused:
            return False
        self._state.is_paused = True
        return True

    def resume(self) -> bool:
        if not self._state.is_running or not self._state.is_paused:
            return False
        self._state.is_paused = False
        return True

    def get_state(self) -> KernelState:
        return self._state

    def get_performance(self) -> PerformanceMetrics:
        return self._perf.get_metrics(len(self._deferred))

    def set_budget(self, ms: float) -> None:
        self._budget.budget_ms = ms

    @property
    def budget_ms(self) -> float:
        return self._budget.budget_ms

    def set_cadence_multiplier(self, multiplier: float) -> None:
        """Scale tick frequency.  >1 = faster ticks, <1 = slower."""
        self._cadence_multiplier = max(
            self._CADENCE_MIN, min(self._CADENCE_MAX, multiplier),
        )
        logger.debug("Cadence multiplier set to %.2f", self._cadence_multiplier)

    @property
    def cadence_multiplier(self) -> float:
        return self._cadence_multiplier

    # -- main loop -----------------------------------------------------------

    async def _run(self) -> None:
        base_interval = KERNEL_INTERVALS["DEFAULT_INTERVAL_S"]
        try:
            while self._state.is_running:
                if not self._state.is_paused:
                    self._tick()
                interval = base_interval / self._cadence_multiplier
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def _tick(self) -> None:
        now = time.time()
        self._state.tick += 1
        tick = self._state.tick

        self._budget.start()

        try:
            self._callbacks.update_state(tick=tick)
            state = self._callbacks.get_kernel_state()
            event_bus.emit(KERNEL_TICK, tick=tick, phase=state["phase"])

            # --- REALTIME queue: always runs ---
            self._run_realtime(now, state)

            # --- Drain deferred ops that fit budget (highest priority first) ---
            self._drain_deferred()

            # --- INTERACTIVE queue: runs if budget remains ---
            if self._budget.remaining() > 1.0:
                self._run_interactive(now, tick, state)

            # --- BACKGROUND queue: runs if budget still remains ---
            if self._budget.remaining() > 1.0:
                self._run_background(now, tick, state)

        except Exception as exc:
            logger.exception("Consciousness loop error")
            event_bus.emit(KERNEL_ERROR, error=str(exc), context="kernel_loop.tick")

        result = self._budget.finish()
        self._perf.record(result.total_time_ms, result.over_budget)
        self._adapt_budget()

    # -- realtime ops (phase, tone — always run) -----------------------------

    def _run_realtime(self, now: float, state: dict[str, Any]) -> None:
        if now - self._state.last_phase_check >= KERNEL_INTERVALS["PHASE_CHECK_INTERVAL_S"]:
            self._process_phase_transitions(now, state)

        if now - self._state.last_tone_check >= KERNEL_INTERVALS["TONE_CHECK_INTERVAL_S"]:
            self._process_tone_transitions(now, state)

    # -- interactive ops (thinking, mutation application) ---------------------

    def _run_interactive(self, now: float, tick: int, state: dict[str, Any]) -> None:
        if now - self._state.last_think_time >= KERNEL_INTERVALS["THINKING_INTERVAL_S"]:
            if self._budget.checkpoint("thinking_cycle"):
                self._callbacks.on_thinking_cycle()
                self._state.last_think_time = now
            else:
                self._defer(Priority.INTERACTIVE, "thinking_cycle",
                            lambda: self._callbacks.on_thinking_cycle())

        if self._budget.remaining() > 1.0:
            if now - self._state.last_meta_thought_time >= KERNEL_INTERVALS["META_THOUGHT_INTERVAL_S"]:
                if self._budget.checkpoint("consciousness_tick"):
                    self._callbacks.on_consciousness_tick()
                    self._state.last_meta_thought_time = now
                else:
                    self._defer(Priority.INTERACTIVE, "consciousness_tick",
                                lambda: self._callbacks.on_consciousness_tick())

        if tick % 100 == 0 and self._budget.remaining() > 1.0:
            if self._budget.checkpoint("trait_modulation"):
                self._callbacks.on_trait_modulation()
            else:
                self._defer(Priority.INTERACTIVE, "trait_modulation",
                            lambda: self._callbacks.on_trait_modulation())

        if self._budget.remaining() > 1.0:
            if now - self._state.last_proactive_check >= KERNEL_INTERVALS["PROACTIVE_CHECK_INTERVAL_S"]:
                if self._budget.checkpoint("proactive_check"):
                    self._callbacks.on_proactive_check()
                    self._state.last_proactive_check = now
                else:
                    self._defer(Priority.INTERACTIVE, "proactive_check",
                                lambda: self._callbacks.on_proactive_check())

    # -- background ops (decay, maintenance, evolution, analysis) ------------

    def _run_background(self, now: float, tick: int, state: dict[str, Any]) -> None:
        if now - self._state.last_decay_time >= KERNEL_INTERVALS["DECAY_CYCLE_INTERVAL_S"]:
            if self._budget.checkpoint("memory_decay"):
                decayed = self._callbacks.decay_memories()
                if decayed > 0:
                    event_bus.emit(MEMORY_DECAY_CYCLE, memories_decayed=decayed)
                self._state.last_decay_time = now
            else:
                self._defer(Priority.BACKGROUND, "memory_decay",
                            lambda: self._callbacks.decay_memories())

        if self._budget.remaining() > 1.0:
            if now - self._state.last_maintenance_time >= KERNEL_INTERVALS["MAINTENANCE_INTERVAL_S"]:
                if self._budget.checkpoint("maintenance"):
                    self._callbacks.perform_maintenance()
                    self._callbacks.calculate_memory_density()
                    self._state.last_maintenance_time = now
                else:
                    self._defer(Priority.BACKGROUND, "maintenance",
                                lambda: (self._callbacks.perform_maintenance(),
                                         self._callbacks.calculate_memory_density()))

        if self._budget.remaining() > 1.0:
            if now - self._state.last_autonomy_tick >= KERNEL_INTERVALS["AUTONOMY_INTERVAL_S"]:
                if self._budget.checkpoint("autonomy_tick"):
                    self._callbacks.on_autonomy_tick()
                    self._state.last_autonomy_tick = now
                else:
                    self._defer(Priority.BACKGROUND, "autonomy_tick",
                                lambda: self._callbacks.on_autonomy_tick())

    # -- deferred work -------------------------------------------------------

    def _defer(self, priority: Priority, name: str, callback: Callable[[], None]) -> None:
        if len(self._deferred) >= self._max_deferred:
            self._deferred.sort(key=lambda d: d.priority)
            dropped = self._deferred.pop()
            logger.warning("Deferred queue full, dropped lowest-priority: %s", dropped.name)

        self._deferred.append(DeferredOp(
            priority=priority, name=name, callback=callback, timestamp=time.time(),
        ))

    def _drain_deferred(self) -> None:
        if not self._deferred:
            return
        self._deferred.sort(key=lambda d: (d.priority, d.timestamp))
        remaining: list[DeferredOp] = []
        for op in self._deferred:
            if self._budget.remaining() > 1.0:
                try:
                    if self._budget.checkpoint(f"deferred:{op.name}"):
                        op.callback()
                    else:
                        remaining.append(op)
                except Exception:
                    logger.exception("Deferred op %s failed", op.name)
            else:
                remaining.append(op)
        self._deferred = remaining

    # -- adaptive budget -----------------------------------------------------

    def _adapt_budget(self) -> None:
        p95 = self._perf.get_p95()
        backlog = len(self._deferred)

        if p95 > BUDGET_WARNING_THRESHOLD_MS or backlog > 15:
            self._budget.budget_ms = min(self._budget.budget_ms + 5.0, IDLE_BUDGET_MS)
        elif p95 < SLOW_TICK_THRESHOLD_MS * 0.5 and backlog == 0:
            self._budget.budget_ms = max(self._budget.budget_ms - 1.0, LOAD_BUDGET_MS)

    # -- phase/tone processing (unchanged logic) -----------------------------

    def _process_phase_transitions(self, now: float, state: dict[str, Any]) -> None:
        from consciousness.phases import phase_manager

        analysis = phase_manager.analyze_phase_transition(
            state["phase"], state["memories"], state["memory_density"], state["is_user_present"],
        )
        if analysis.suggested_phase and analysis.confidence > 0.7:
            success = phase_manager.execute_phase_transition(
                analysis.current_phase, analysis.suggested_phase, state["tone"],
            )
            if success:
                self._callbacks.update_state(phase=analysis.suggested_phase)
        self._state.last_phase_check = now

    def _process_tone_transitions(self, now: float, state: dict[str, Any]) -> None:
        from consciousness.tone import tone_engine

        recent_memories = state["memories"][-10:]
        time_since = now - self._state.last_tone_check

        analysis = tone_engine.analyze_tone_shift(
            state["tone"], recent_memories, time_since,
        )
        if analysis.suggested_tone and analysis.confidence > 0.5:
            tone_engine.record_tone_change(analysis.suggested_tone)
            self._callbacks.update_state(tone=analysis.suggested_tone)
            event_bus.emit(TONE_SHIFT, from_tone=analysis.current_tone, to_tone=analysis.suggested_tone)
        self._state.last_tone_check = now

    # -- diagnostics ---------------------------------------------------------

    def _log_final_stats(self) -> None:
        m = self._perf.get_metrics(len(self._deferred))
        if m.tick_count == 0:
            return
        overrun_pct = (m.budget_overruns / m.tick_count) * 100 if m.tick_count else 0
        logger.info(
            "Kernel stats: %d ticks, avg=%.2fms, p95=%.2fms, max=%.2fms, overruns=%d (%.1f%%)",
            m.tick_count, m.avg_tick_ms, m.p95_tick_ms, m.max_tick_ms,
            m.budget_overruns, overrun_pct,
        )
