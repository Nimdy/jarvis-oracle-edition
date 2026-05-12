"""Memory Optimizer — adaptive processing reduction under resource pressure.

Monitors Python process memory and consciousness data structure sizes.
When pressure is detected, emits events to reduce observation rate,
trim reasoning chains, and defer non-critical background operations.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from consciousness.events import (
    event_bus,
    CONSCIOUSNESS_CLEANUP_OBSERVATIONS,
    CONSCIOUSNESS_CLEANUP_OLD_CHAINS,
    CONSCIOUSNESS_CLEAR_CACHES,
    CONSCIOUSNESS_REDUCE_OBSERVATION_RATE,
)

logger = logging.getLogger(__name__)

CHECK_INTERVAL_S = 5.0
CLEANUP_COOLDOWN_S = 15.0


def _compute_thresholds() -> tuple[float, float]:
    """Derive RSS thresholds from system RAM so GPU-loaded systems don't false-alarm."""
    try:
        from hardware_profile import get_hardware_profile
        ram_gb = get_hardware_profile().cpu.ram_gb
    except Exception:
        ram_gb = 4
    optimize_mb = max(400, int(ram_gb * 1024 * 0.50))
    cleanup_mb = max(600, int(ram_gb * 1024 * 0.75))
    return optimize_mb, cleanup_mb


RSS_OPTIMIZE_MB, RSS_CLEANUP_MB = _compute_thresholds()

MAX_REASONING_CHAINS = 10
MAX_OBSERVATIONS = 50
MAX_META_THOUGHTS = 25


@dataclass(frozen=True)
class MemoryMetrics:
    rss_mb: float
    timestamp: float


@dataclass(frozen=True)
class MemoryStatus:
    current_mb: float
    trend: str  # increasing, stable, decreasing
    recommendation: str  # normal, monitor, optimize, reduce_processing


class MemoryOptimizer:
    """Monitors process memory and triggers consciousness cleanup under pressure."""

    _instance: MemoryOptimizer | None = None

    def __init__(self) -> None:
        self._metrics: deque[MemoryMetrics] = deque(maxlen=20)
        self._last_cleanup = 0.0
        logger.info(
            "MemoryOptimizer: thresholds optimize=%dMB cleanup=%dMB",
            RSS_OPTIMIZE_MB, RSS_CLEANUP_MB,
        )
        self._last_check = 0.0
        self._total_cleanups = 0
        self._reduction_active = False
        self._reduction_until = 0.0

    @classmethod
    def get_instance(cls) -> MemoryOptimizer:
        if cls._instance is None:
            cls._instance = MemoryOptimizer()
        return cls._instance

    def check(self) -> MemoryStatus | None:
        """Check process memory and take action if needed. Call periodically."""
        now = time.time()
        if now - self._last_check < CHECK_INTERVAL_S:
            return None
        self._last_check = now

        rss_mb = self._get_rss_mb()
        self._metrics.append(MemoryMetrics(rss_mb=rss_mb, timestamp=now))

        if self._reduction_active and now > self._reduction_until:
            self._reduction_active = False
            logger.info("Memory pressure reduction period ended")

        if rss_mb > RSS_CLEANUP_MB:
            logger.warning("High memory usage: %.1fMB — triggering consciousness cleanup", rss_mb)
            self._perform_cleanup(aggressive=True)
        elif rss_mb > RSS_OPTIMIZE_MB:
            logger.info("Moderate memory usage: %.1fMB — optimizing structures", rss_mb)
            self._perform_cleanup(aggressive=False)

        if len(self._metrics) >= 3:
            recent = list(self._metrics)[-3:]
            if all(recent[i].rss_mb > recent[i - 1].rss_mb for i in range(1, len(recent))):
                if rss_mb > RSS_OPTIMIZE_MB * 0.8:
                    logger.info("Memory trending up — preemptive optimization")
                    self._perform_cleanup(aggressive=False)

        return self.get_status()

    def should_reduce_processing(self) -> bool:
        """Quick check for consciousness systems to skip non-critical work."""
        if self._reduction_active:
            return True
        if not self._metrics:
            return False
        return self._metrics[-1].rss_mb > RSS_OPTIMIZE_MB * 0.9

    def get_status(self) -> MemoryStatus:
        """Current memory status with trend and recommendation."""
        if len(self._metrics) < 3:
            return MemoryStatus(
                current_mb=self._metrics[-1].rss_mb if self._metrics else 0.0,
                trend="stable",
                recommendation="normal",
            )

        recent = list(self._metrics)[-3:]
        current = recent[-1].rss_mb

        changes = [recent[i].rss_mb - recent[i - 1].rss_mb for i in range(1, len(recent))]
        avg_change = sum(changes) / len(changes)

        if avg_change > 10:
            trend = "increasing"
        elif avg_change < -10:
            trend = "decreasing"
        else:
            trend = "stable"

        if current > RSS_CLEANUP_MB:
            recommendation = "reduce_processing"
        elif current > RSS_OPTIMIZE_MB:
            recommendation = "optimize"
        elif trend == "increasing":
            recommendation = "monitor"
        else:
            recommendation = "normal"

        return MemoryStatus(current_mb=round(current, 1), trend=trend, recommendation=recommendation)

    def get_state(self) -> dict[str, Any]:
        status = self.get_status()
        return {
            "current_mb": status.current_mb,
            "trend": status.trend,
            "recommendation": status.recommendation,
            "reduction_active": self._reduction_active,
            "total_cleanups": self._total_cleanups,
            "metrics_count": len(self._metrics),
        }

    def _perform_cleanup(self, aggressive: bool) -> None:
        now = time.time()
        if now - self._last_cleanup < CLEANUP_COOLDOWN_S:
            return

        self._last_cleanup = now
        self._total_cleanups += 1

        event_bus.emit(CONSCIOUSNESS_CLEANUP_OBSERVATIONS, timestamp=now)
        event_bus.emit(CONSCIOUSNESS_CLEANUP_OLD_CHAINS, timestamp=now)

        if aggressive:
            event_bus.emit(CONSCIOUSNESS_CLEAR_CACHES, timestamp=now)
            event_bus.emit(CONSCIOUSNESS_REDUCE_OBSERVATION_RATE, timestamp=now, duration_s=60.0)
            self._reduction_active = True
            self._reduction_until = now + 60.0
            logger.info("Aggressive cleanup complete — processing reduced for 60s")
        else:
            logger.info("Optimization cleanup complete")

    @staticmethod
    def _get_rss_mb() -> float:
        """Get current process RSS in MB. Uses /proc/self/status on Linux."""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024.0
        except (ImportError, AttributeError):
            pass
        return 0.0


memory_optimizer = MemoryOptimizer.get_instance()
