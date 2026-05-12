"""Consciousness Analytics — O(1) rolling-window metrics for system health.

Tracks confidence, reasoning quality, epistemic state, and tick performance.
All computations are O(1) over fixed-size deques, never full scans.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

WINDOW_SIZE = 100
TICK_WINDOW_SIZE = 200
REFRESH_INTERVAL_S = 5.0
STALENESS_THRESHOLD_S = 120.0
BOOT_GRACE_S = 60.0


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------

@dataclass
class MetricReading:
    """Provenance-tracked metric value."""
    value: float
    source: str = "default"    # "live" or "default"
    updated_at: float = 0.0


@dataclass
class ConfidenceMetrics:
    current: float = 0.5
    avg: float = 0.5
    trend: float = 0.0     # positive = improving
    volatility: float = 0.0


@dataclass
class ReasoningQuality:
    coherence: float = 0.5       # low oscillation in tone/phase
    consistency: float = 0.5     # few contradictions
    depth: float = 0.5           # ratio of deep/profound thoughts
    overall: float = 0.5


@dataclass
class EpistemicState:
    known_count: int = 0
    uncertain_count: int = 0
    unknown_count: int = 0
    curiosity_level: float = 0.5
    openness: float = 0.5


@dataclass
class SystemHealth:
    tick_p95_ms: float = 0.0
    tick_avg_ms: float = 0.0
    deferred_backlog: int = 0
    memory_count: int = 0
    uptime_s: float = 0.0
    healthy: bool = True


@dataclass
class HealthAlert:
    severity: str  # info, warning, critical
    component: str
    message: str
    metric: float
    threshold: float
    recommendation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentHealth:
    memory_health: float = 0.5
    processing_health: float = 0.5
    personality_health: float = 0.5
    event_health: float = 0.5
    cognitive_load: float = 0.5
    overall: float = 0.5
    confidence: float = 0.0     # fraction of metrics that are strictly live
    status: str = "healthy"  # optimal, healthy, stressed, degraded, critical
    trend: str = "stable"  # improving, stable, declining, fluctuating
    alerts: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class ConsciousnessAnalytics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._confidence_window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._reasoning_window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._depth_window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._tick_times: deque[float] = deque(maxlen=TICK_WINDOW_SIZE)
        self._tone_history: deque[str] = deque(maxlen=50)
        self._phase_history: deque[str] = deque(maxlen=50)
        self._mutation_outcomes: deque[float] = deque(maxlen=30)

        self._start_time = time.time()
        self._last_refresh = 0.0

        self._memory_reading = MetricReading(0, "default", 0.0)
        self._backlog_reading = MetricReading(0, "default", 0.0)
        self._tick_reading = MetricReading(0.0, "default", 0.0)
        self._event_error_reading = MetricReading(0.0, "default", 0.0)
        self._personality_reading = MetricReading(1.0, "default", 0.0)

        self._cached_confidence = ConfidenceMetrics()
        self._cached_reasoning = ReasoningQuality()
        self._cached_epistemic = EpistemicState()
        self._cached_health = SystemHealth()

        self._health_readings: deque[float] = deque(maxlen=10)
        self._cached_component_health = ComponentHealth()
        self._health_alerts: list[HealthAlert] = []

    # -- property aliases for backward compatibility -------------------------

    @property
    def _memory_count(self) -> int:
        return int(self._memory_reading.value)

    @property
    def _deferred_backlog(self) -> int:
        return int(self._backlog_reading.value)

    @property
    def _event_error_rate(self) -> float:
        return self._event_error_reading.value

    @property
    def _personality_coherence(self) -> float:
        return self._personality_reading.value

    # -- recording methods (called by system) --------------------------------

    def record_tick(self, elapsed_ms: float) -> None:
        with self._lock:
            self._tick_times.append(elapsed_ms)
        now = time.time()
        self._tick_reading = MetricReading(elapsed_ms, "live", now)

    def record_confidence(self, value: float) -> None:
        with self._lock:
            self._confidence_window.append(max(0.0, min(1.0, value)))

    def record_thought(self, depth: str, confidence: float) -> None:
        depth_val = {"surface": 0.3, "deep": 0.7, "profound": 1.0}.get(depth, 0.3)
        with self._lock:
            self._depth_window.append(depth_val)
            self._reasoning_window.append(confidence)

    def record_tone_change(self, tone: str) -> None:
        with self._lock:
            self._tone_history.append(tone)

    def record_phase_change(self, phase: str) -> None:
        with self._lock:
            self._phase_history.append(phase)

    def record_mutation_outcome(self, improvement: float) -> None:
        """improvement > 0 means positive, < 0 means regression."""
        with self._lock:
            self._mutation_outcomes.append(improvement)

    def update_memory_count(self, count: int) -> None:
        self._memory_reading = MetricReading(count, "live", time.time())

    def update_backlog(self, count: int) -> None:
        self._backlog_reading = MetricReading(count, "live", time.time())

    def record_event_error_rate(self, rate: float) -> None:
        self._event_error_reading = MetricReading(rate, "live", time.time())

    def record_personality_coherence(self, score: float) -> None:
        self._personality_reading = MetricReading(score, "live", time.time())

    # -- provenance helpers --------------------------------------------------

    def _effective_source(self, reading: MetricReading, now: float) -> str:
        """Classify a reading's effective trust level."""
        if reading.source == "default":
            if now - self._start_time <= BOOT_GRACE_S:
                return "booting"
            return "missing"
        age = now - reading.updated_at
        if age > STALENESS_THRESHOLD_S:
            return "stale"
        return "live"

    def _all_readings(self) -> dict[str, MetricReading]:
        return {
            "memory_health": self._memory_reading,
            "processing_health": self._tick_reading,
            "personality_health": self._personality_reading,
            "event_health": self._event_error_reading,
            "cognitive_load": self._backlog_reading,
        }

    def get_liveness_faults(self) -> list[str]:
        """Return metric names that have no live data after boot grace."""
        now = time.time()
        if now - self._start_time <= BOOT_GRACE_S:
            return []
        faults = []
        for name, reading in self._all_readings().items():
            eff = self._effective_source(reading, now)
            if eff in ("missing", "stale"):
                faults.append(name)
        return faults

    # -- computed metrics (O(1) from caches) ---------------------------------

    def get_confidence(self) -> ConfidenceMetrics:
        self._maybe_refresh()
        return self._cached_confidence

    def get_reasoning_quality(self) -> ReasoningQuality:
        self._maybe_refresh()
        return self._cached_reasoning

    def get_epistemic_state(self) -> EpistemicState:
        self._maybe_refresh()
        return self._cached_epistemic

    def get_system_health(self) -> SystemHealth:
        self._maybe_refresh()
        return self._cached_health

    def get_tick_p95(self) -> float:
        with self._lock:
            return self._tick_p95_unlocked()

    def _tick_p95_unlocked(self) -> float:
        if not self._tick_times:
            return 0.0
        s = sorted(self._tick_times)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def get_mutation_success_rate(self) -> float:
        with self._lock:
            return self._mutation_success_rate_unlocked()

    def _mutation_success_rate_unlocked(self) -> float:
        if not self._mutation_outcomes:
            return 0.5
        positives = sum(1 for o in self._mutation_outcomes if o > 0)
        return positives / len(self._mutation_outcomes)

    def get_full_state(self) -> dict[str, Any]:
        self._maybe_refresh()
        return {
            "confidence": {
                "current": self._cached_confidence.current,
                "avg": self._cached_confidence.avg,
                "trend": self._cached_confidence.trend,
                "volatility": self._cached_confidence.volatility,
            },
            "reasoning": {
                "coherence": self._cached_reasoning.coherence,
                "consistency": self._cached_reasoning.consistency,
                "depth": self._cached_reasoning.depth,
                "overall": self._cached_reasoning.overall,
            },
            "epistemic": {
                "curiosity": self._cached_epistemic.curiosity_level,
                "openness": self._cached_epistemic.openness,
            },
            "health": {
                "tick_p95_ms": self._cached_health.tick_p95_ms,
                "healthy": self._cached_health.healthy,
                "memory_count": self._cached_health.memory_count,
            },
            "component_health": self.get_health_report(),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore analytics caches from persisted dict so metrics don't reset to defaults."""
        conf = data.get("confidence", {})
        if conf:
            self._cached_confidence = ConfidenceMetrics(
                current=conf.get("current", 0.5),
                avg=conf.get("avg", 0.5),
                trend=conf.get("trend", 0.0),
                volatility=conf.get("volatility", 0.0),
            )
            if self._cached_confidence.avg > 0:
                self._confidence_window.append(self._cached_confidence.avg)

        reason = data.get("reasoning", {})
        if reason:
            self._cached_reasoning = ReasoningQuality(
                coherence=reason.get("coherence", 0.5),
                consistency=reason.get("consistency", 0.5),
                depth=reason.get("depth", 0.5),
                overall=reason.get("overall", 0.5),
            )

        epist = data.get("epistemic", {})
        if epist:
            self._cached_epistemic = EpistemicState(
                curiosity_level=epist.get("curiosity", 0.5),
                openness=epist.get("openness", 0.5),
            )

    def get_health_report(self) -> dict[str, Any]:
        self._maybe_refresh()
        h = self._cached_component_health
        return {
            "components": {
                "memory_health": h.memory_health,
                "processing_health": h.processing_health,
                "personality_health": h.personality_health,
                "event_health": h.event_health,
                "cognitive_load": h.cognitive_load,
            },
            "overall": h.overall,
            "confidence": h.confidence,
            "status": h.status,
            "trend": h.trend,
            "provenance": h.provenance,
            "alerts": [
                {"severity": a.severity, "component": a.component,
                 "message": a.message, "metric": round(a.metric, 3),
                 "threshold": a.threshold}
                for a in self._health_alerts[-10:]
            ],
        }

    # -- periodic refresh (every 5s) ----------------------------------------

    def _maybe_refresh(self) -> None:
        now = time.time()
        if now - self._last_refresh < REFRESH_INTERVAL_S:
            return
        with self._lock:
            self._last_refresh = now
            self._refresh_confidence()
            self._refresh_reasoning()
            self._refresh_epistemic()
            self._refresh_health()
            self._refresh_component_health()

    def _refresh_confidence(self) -> None:
        w = self._confidence_window
        if not w:
            self._cached_confidence = ConfidenceMetrics()
            return

        current = w[-1]
        avg = sum(w) / len(w)

        half = len(w) // 2
        if half > 0:
            first_half = sum(list(w)[:half]) / half
            second_half = sum(list(w)[half:]) / (len(w) - half)
            trend = second_half - first_half
        else:
            trend = 0.0

        diffs = [abs(list(w)[i] - list(w)[i - 1]) for i in range(1, len(w))]
        volatility = sum(diffs) / len(diffs) if diffs else 0.0

        self._cached_confidence = ConfidenceMetrics(
            current=current, avg=avg, trend=trend, volatility=volatility,
        )

    def _refresh_reasoning(self) -> None:
        coherence = self._compute_coherence()
        depth = sum(self._depth_window) / len(self._depth_window) if self._depth_window else 0.5

        consistency = self._compute_epistemic_consistency()
        overall = coherence * 0.35 + consistency * 0.3 + depth * 0.35

        self._cached_reasoning = ReasoningQuality(
            coherence=coherence, consistency=consistency, depth=depth, overall=overall,
        )

    def _compute_epistemic_consistency(self) -> float:
        """Use contradiction_debt when engine is active, fallback to volatility."""
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine.get_instance()
            if engine and engine.belief_store.get_stats()["total_beliefs"] > 0:
                return max(0.0, min(1.0, 1.0 - engine.contradiction_debt))
        except Exception:
            pass
        volatility = self._cached_confidence.volatility if self._cached_confidence else 0.0
        return max(0.0, min(1.0, 1.0 - volatility * 2))

    def _refresh_epistemic(self) -> None:
        depth_vals = list(self._depth_window)
        if not depth_vals:
            self._cached_epistemic = EpistemicState()
            return

        deep_count = sum(1 for d in depth_vals if d > 0.5)
        shallow_count = sum(1 for d in depth_vals if d <= 0.3)
        mid_count = len(depth_vals) - deep_count - shallow_count

        curiosity = deep_count / len(depth_vals)
        openness = 1.0 - (self._cached_confidence.avg * 0.3)

        self._cached_epistemic = EpistemicState(
            known_count=deep_count,
            uncertain_count=mid_count,
            unknown_count=shallow_count,
            curiosity_level=curiosity,
            openness=max(0.0, min(1.0, openness)),
        )

    def _refresh_health(self) -> None:
        p95 = self._tick_p95_unlocked()
        avg = sum(self._tick_times) / len(self._tick_times) if self._tick_times else 0.0
        healthy = p95 < 50.0 and self._deferred_backlog < 15

        self._cached_health = SystemHealth(
            tick_p95_ms=p95,
            tick_avg_ms=avg,
            deferred_backlog=self._deferred_backlog,
            memory_count=self._memory_count,
            uptime_s=time.time() - self._start_time,
            healthy=healthy,
        )

    def _refresh_component_health(self) -> None:
        COMPONENT_WEIGHTS = {
            "memory_health": 0.30,
            "processing_health": 0.20,
            "personality_health": 0.20,
            "event_health": 0.15,
            "cognitive_load": 0.15,
        }

        memory_health = min(1.0, self._memory_count / 50) * 0.7 + (0.3 if self._memory_count > 10 else 0.0)
        tick_p95 = self._tick_p95_unlocked()
        if tick_p95 <= 25.0:
            processing_health = 1.0
        elif tick_p95 <= 50.0:
            processing_health = 1.0 - (tick_p95 - 25.0) / 50.0
        else:
            processing_health = max(0.0, 0.5 - (tick_p95 - 50.0) / 100.0)
        personality_health = self._personality_coherence
        event_health = 1.0 - self._event_error_rate
        cognitive_load = max(0.0, 1.0 - self._deferred_backlog / 20.0)

        raw_scores = {
            "memory_health": memory_health,
            "processing_health": processing_health,
            "personality_health": personality_health,
            "event_health": event_health,
            "cognitive_load": cognitive_load,
        }

        now = time.time()
        readings_map = self._all_readings()
        provenance: dict[str, Any] = {}
        live_count = 0
        for name, reading in readings_map.items():
            eff = self._effective_source(reading, now)
            age_s = round(now - reading.updated_at, 1) if reading.updated_at > 0 else -1.0
            provenance[name] = {"source": eff, "age_s": age_s}
            if eff == "live":
                live_count += 1

        confidence = live_count / len(readings_map) if readings_map else 0.0

        # Compute overall from non-missing inputs only (weights redistributed)
        live_weight_sum = 0.0
        weighted_score_sum = 0.0
        for name, weight in COMPONENT_WEIGHTS.items():
            eff = provenance[name]["source"]
            score = raw_scores[name]
            if eff == "missing":
                continue
            if eff == "stale":
                score = score * 0.5
            live_weight_sum += weight
            weighted_score_sum += score * weight

        overall = weighted_score_sum / live_weight_sum if live_weight_sum > 0 else 0.0

        if overall > 0.85:
            status = "optimal"
        elif overall > 0.60:
            status = "healthy"
        elif overall > 0.40:
            status = "stressed"
        elif overall > 0.25:
            status = "degraded"
        else:
            status = "critical"

        # Fail-closed: if fewer than 3 of 5 metrics are live, cannot claim healthy
        if confidence < 0.6 and status in ("optimal", "healthy"):
            status = "degraded"

        self._health_readings.append(overall)

        trend = "stable"
        readings = list(self._health_readings)
        if len(readings) >= 3:
            recent = readings[-5:]
            n = len(recent)
            x_mean = (n - 1) / 2.0
            y_mean = sum(recent) / n
            num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0.0
            variance = sum((v - y_mean) ** 2 for v in recent) / n
            if variance > 0.05:
                trend = "fluctuating"
            elif slope > 0.01:
                trend = "improving"
            elif slope < -0.01:
                trend = "declining"

        alerts: list[HealthAlert] = []
        thresholds = {
            "memory_health": 0.4,
            "processing_health": 0.5,
            "personality_health": 0.5,
            "event_health": 0.6,
            "cognitive_load": 0.4,
        }
        recommendations = {
            "memory_health": "Increase memory formation or check memory persistence",
            "processing_health": "Reduce tick workload or increase budget_ms",
            "personality_health": "Review trait consistency and coherence signals",
            "event_health": "Investigate event processing errors",
            "cognitive_load": "Clear deferred backlog or reduce background task frequency",
        }
        for comp, thresh in thresholds.items():
            eff = provenance[comp]["source"]
            if eff == "missing":
                continue
            val = raw_scores[comp]
            if eff == "stale":
                val = val * 0.5
            if val < thresh:
                severity = "critical" if val < thresh * 0.5 else "warning"
                alerts.append(HealthAlert(
                    severity=severity,
                    component=comp,
                    message=f"{comp} at {val:.2f}, below threshold {thresh}",
                    metric=val,
                    threshold=thresh,
                    recommendation=recommendations[comp],
                ))

        # Liveness fault alerts (after boot grace)
        if now - self._start_time > BOOT_GRACE_S:
            for name, reading in readings_map.items():
                eff = self._effective_source(reading, now)
                if eff == "missing":
                    alerts.append(HealthAlert(
                        severity="warning",
                        component=name,
                        message=f"No live data for {name} — data pipeline not wired",
                        metric=0.0,
                        threshold=0.0,
                        recommendation=f"Check {name} setter: never called after {BOOT_GRACE_S:.0f}s boot grace",
                    ))
                elif eff == "stale":
                    age = now - reading.updated_at
                    alerts.append(HealthAlert(
                        severity="info",
                        component=name,
                        message=f"{name} data stale ({age:.0f}s since last update)",
                        metric=reading.value,
                        threshold=STALENESS_THRESHOLD_S,
                        recommendation=f"Check {name} data source — last update {age:.0f}s ago",
                    ))

        self._health_alerts = alerts
        self._cached_component_health = ComponentHealth(
            memory_health=memory_health,
            processing_health=processing_health,
            personality_health=personality_health,
            event_health=event_health,
            cognitive_load=cognitive_load,
            overall=round(overall, 4),
            confidence=round(confidence, 2),
            status=status,
            trend=trend,
            alerts=alerts,
            provenance=provenance,
        )

    def _compute_coherence(self) -> float:
        """Low oscillation in tone/phase = high coherence."""
        tone_switches = 0
        tones = list(self._tone_history)
        for i in range(1, len(tones)):
            if tones[i] != tones[i - 1]:
                tone_switches += 1

        phase_switches = 0
        phases = list(self._phase_history)
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_switches += 1

        max_switches = max(len(tones), 1)
        tone_coherence = 1.0 - (tone_switches / max_switches)
        phase_coherence = 1.0 - (phase_switches / max(len(phases), 1))

        return (tone_coherence * 0.5 + phase_coherence * 0.5)
