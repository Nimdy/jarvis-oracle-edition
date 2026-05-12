"""Consciousness Health Monitor — formal weighted health scoring with trend prediction.

Aggregates 5 health dimensions (memory, processing, personality, events, cognitive load),
generates severity-leveled alerts, tracks trends via linear regression, and produces
actionable recommendations. O(1) hot path, snapshot-based reads.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_HISTORY = 100
CHECK_INTERVAL_S = 5.0

COMPONENT_WEIGHTS = {
    "memory": 0.30,
    "processing": 0.20,
    "personality": 0.20,
    "events": 0.15,
    "cognitive": 0.15,
}

THRESHOLDS = {
    "optimal": 0.85,
    "healthy": 0.60,
    "stressed": 0.40,
    "degraded": 0.25,
}


@dataclass(frozen=True)
class HealthAlert:
    severity: str  # info, warning, error, critical
    component: str
    message: str
    metric: float
    threshold: float
    action: str


@dataclass(frozen=True)
class HealthTrend:
    direction: str  # improving, stable, declining, fluctuating
    velocity: float  # per-hour rate of change
    confidence: float  # 0-1
    prediction: str


@dataclass
class HealthSnapshot:
    timestamp: float
    overall: float
    memory: float
    processing: float
    personality: float
    events: float
    cognitive: float
    status: str
    alert_count: int


class ConsciousnessHealthMonitor:
    """Formal 5-dimension health scoring with trend prediction and alerts."""

    _instance: ConsciousnessHealthMonitor | None = None

    def __init__(self) -> None:
        self._history: deque[HealthSnapshot] = deque(maxlen=MAX_HISTORY)
        self._start_time = time.time()
        self._last_check = 0.0
        self._total_checks = 0
        self._error_count = 0
        try:
            from memory.maintenance import MAX_MEMORIES
            self._max_memories = MAX_MEMORIES
        except ImportError:
            self._max_memories = 2000

    @classmethod
    def get_instance(cls) -> ConsciousnessHealthMonitor:
        if cls._instance is None:
            cls._instance = ConsciousnessHealthMonitor()
        return cls._instance

    def assess(
        self,
        memory_count: int = 0,
        memory_density_overall: float = 0.5,
        isolated_memory_ratio: float = 0.0,
        tick_p95_ms: float = 0.0,
        deferred_backlog: int = 0,
        personality_coherence: float = 1.0,
        trait_stability: float = 1.0,
        event_error_rate: float = 0.0,
        event_integrity_score: float = 1.0,
        circuit_breaker_trips: int = 0,
        active_reasoning_chains: int = 0,
    ) -> dict[str, Any]:
        """Run a full health assessment and return vitals."""
        now = time.time()
        self._total_checks += 1

        # --- Component scores ---
        memory_h = self._assess_memory(memory_count, memory_density_overall, isolated_memory_ratio)
        processing_h = self._assess_processing(tick_p95_ms, deferred_backlog)
        personality_h = self._assess_personality(personality_coherence, trait_stability)
        events_h = self._assess_events(event_error_rate, event_integrity_score, circuit_breaker_trips)
        cognitive_h = self._assess_cognitive(deferred_backlog, active_reasoning_chains, memory_density_overall)

        components = {
            "memory": round(memory_h, 4),
            "processing": round(processing_h, 4),
            "personality": round(personality_h, 4),
            "events": round(events_h, 4),
            "cognitive": round(1.0 - cognitive_h, 4),  # invert: lower load = higher health
        }

        overall = sum(components[k] * COMPONENT_WEIGHTS[k] for k in COMPONENT_WEIGHTS)
        overall = round(max(0.0, min(1.0, overall)), 4)

        status = self._determine_status(overall)
        alerts = self._generate_alerts(components, overall)
        recommendations = self._generate_recommendations(components, alerts)
        trend = self._compute_trend()

        snapshot = HealthSnapshot(
            timestamp=now, overall=overall,
            memory=memory_h, processing=processing_h,
            personality=personality_h, events=events_h,
            cognitive=cognitive_h, status=status,
            alert_count=len(alerts),
        )
        self._history.append(snapshot)
        self._last_check = now

        return {
            "overall": overall,
            "status": status,
            "components": components,
            "alerts": [
                {"severity": a.severity, "component": a.component,
                 "message": a.message, "action": a.action}
                for a in alerts
            ],
            "recommendations": recommendations,
            "trend": {
                "direction": trend.direction,
                "velocity": round(trend.velocity, 4),
                "confidence": round(trend.confidence, 3),
                "prediction": trend.prediction,
            },
            "vitals": {
                "uptime_s": round(now - self._start_time, 1),
                "total_checks": self._total_checks,
                "memory_utilization": round(min(1.0, memory_count / self._max_memories), 3),
                "processing_efficiency": round(processing_h, 3),
            },
            "timestamp": now,
        }

    # --- Individual component assessments ---

    def _assess_memory(self, count: int, density: float, isolated_ratio: float) -> float:
        h = 1.0
        h *= 0.2 + min(1.0, count / 50) * 0.8
        h *= 0.3 + density * 0.7
        if isolated_ratio > 0.5:
            h *= 0.7
        if density < 0.3:
            h *= 0.8
        return max(0.0, min(1.0, h))

    def _assess_processing(self, tick_p95: float, backlog: int) -> float:
        if tick_p95 <= 25.0:
            h = 1.0
        elif tick_p95 <= 50.0:
            h = 1.0 - (tick_p95 - 25.0) / 50.0
        else:
            h = max(0.0, 0.5 - (tick_p95 - 50.0) / 100.0)
        if backlog > 10:
            h *= max(0.3, 1.0 - backlog * 0.05)
        return max(0.0, min(1.0, h))

    def _assess_personality(self, coherence: float, stability: float) -> float:
        h = coherence * 0.6 + stability * 0.4
        return max(0.0, min(1.0, h))

    def _assess_events(self, error_rate: float, integrity: float, cb_trips: int) -> float:
        h = (1.0 - error_rate) * integrity
        if cb_trips > 0:
            h *= max(0.4, 1.0 - cb_trips * 0.15)
        return max(0.0, min(1.0, h))

    def _assess_cognitive(self, backlog: int, chains: int, density: float) -> float:
        load = 0.0
        load += min(1.0, backlog / 20.0) * 0.4
        load += min(1.0, chains / 10.0) * 0.3
        if density > 0.8:
            load += 0.2
        return min(1.0, load)

    # --- Status, alerts, recommendations ---

    def _determine_status(self, overall: float) -> str:
        if overall >= THRESHOLDS["optimal"]:
            return "optimal"
        if overall >= THRESHOLDS["healthy"]:
            return "healthy"
        if overall >= THRESHOLDS["stressed"]:
            return "stressed"
        if overall >= THRESHOLDS["degraded"]:
            return "degraded"
        return "critical"

    def _generate_alerts(self, components: dict[str, float], overall: float) -> list[HealthAlert]:
        alerts: list[HealthAlert] = []
        alert_rules = [
            ("memory", 0.3, "critical", "Memory system severely degraded", "Run memory maintenance and integrity check"),
            ("memory", 0.5, "warning", "Memory health declining", "Monitor fragmentation and run cleanup"),
            ("processing", 0.3, "critical", "Kernel processing critically impaired", "Activate emergency throttling"),
            ("processing", 0.5, "warning", "Processing performance degraded", "Optimize tick workload"),
            ("personality", 0.4, "error", "Personality coherence compromised", "Review trait evolution and contradictions"),
            ("events", 0.4, "error", "Event system reliability compromised", "Check circuit breakers and error rates"),
            ("cognitive", 0.3, "warning", "High cognitive load", "Defer non-critical background operations"),
        ]
        for comp, thresh, sev, msg, action in alert_rules:
            val = components.get(comp, 1.0)
            if val < thresh:
                alerts.append(HealthAlert(sev, comp, msg, round(val, 3), thresh, action))

        if overall < 0.3:
            alerts.append(HealthAlert(
                "critical", "overall",
                "Overall consciousness health critical",
                round(overall, 3), 0.3,
                "Immediate intervention — activate emergency protocols",
            ))
        return alerts

    def _generate_recommendations(self, components: dict[str, float], alerts: list[HealthAlert]) -> list[str]:
        recs: list[str] = []
        if not alerts:
            recs.append("Consciousness health optimal — maintain current operations")
            return recs

        critical = [a for a in alerts if a.severity == "critical"]
        if critical:
            recs.append("CRITICAL: Immediate attention required for consciousness stability")
            for a in critical:
                recs.append(f"  {a.action}")

        if components.get("memory", 1.0) < 0.6:
            recs.append("Consider memory optimization and defragmentation")
        if components.get("processing", 1.0) < 0.6:
            recs.append("Optimize processing load or increase tick budget")
        if components.get("personality", 1.0) < 0.6:
            recs.append("Review personality trait coherence and stability")
        if components.get("cognitive", 1.0) < 0.4:
            recs.append("Reduce cognitive load by deferring non-essential operations")

        if not recs:
            recs.append("Monitor health trends and maintain current settings")
        return recs

    # --- Trend prediction via linear regression ---

    def _compute_trend(self) -> HealthTrend:
        readings = list(self._history)
        if len(readings) < 3:
            return HealthTrend("stable", 0.0, 0.3, "Insufficient data for trend analysis")

        recent = readings[-5:]
        values = [s.overall for s in recent]
        n = len(values)

        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))

        denom = n * sum_x2 - sum_x * sum_x
        slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0
        velocity = slope * 720  # per hour (assuming ~5s checks)

        y_mean = sum_y / n
        variance = sum((v - y_mean) ** 2 for v in values) / n

        if variance > 0.05:
            direction = "fluctuating"
        elif abs(velocity) < 0.01:
            direction = "stable"
        elif velocity > 0:
            direction = "improving"
        else:
            direction = "declining"

        confidence = max(0.1, min(1.0, 1.0 - variance * 10))

        predictions = {
            "improving": "Consciousness health trending upward",
            "declining": "Health declining — intervention may be needed",
            "fluctuating": "Health unstable — investigate root causes",
            "stable": "Consciousness health stable",
        }

        return HealthTrend(direction, velocity, confidence, predictions[direction])

    # --- Public API ---

    def get_history(self) -> list[dict[str, Any]]:
        return [
            {"timestamp": s.timestamp, "overall": s.overall, "status": s.status,
             "alert_count": s.alert_count}
            for s in self._history
        ]

    def get_summary(self) -> dict[str, Any]:
        if not self._history:
            return {"status": "unknown", "overall": 0.5, "alert_count": 0}
        latest = self._history[-1]
        return {"status": latest.status, "overall": latest.overall, "alert_count": latest.alert_count}

    def get_state(self) -> dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "history_size": len(self._history),
            "uptime_s": round(time.time() - self._start_time, 1),
            "last_check": self._last_check,
            "summary": self.get_summary(),
        }


health_monitor = ConsciousnessHealthMonitor.get_instance()
