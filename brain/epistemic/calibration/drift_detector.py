"""Drift detector: hysteresis-based calibration degradation alerting.

Detects sustained calibration drift per domain with severity classification.
Drift triggers require BOTH a score drop AND sustained negative slope to prevent
alert spam from transient fluctuations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from epistemic.calibration.domain_calibrator import DomainScore, ALL_DOMAINS

DRIFT_DROP_THRESHOLD = 0.08
DRIFT_SLOPE_READINGS = 5
DRIFT_RECOVERY_MARGIN = 0.04

DriftSeverity = Literal["minor", "moderate", "major"]


@dataclass
class DriftAlert:
    domain: str
    severity: DriftSeverity
    score_drop: float
    slope: float
    readings_declining: int
    peak_score: float
    current_score: float
    triggered_at: float


@dataclass
class _DomainState:
    peak_score: float = 0.5
    consecutive_declining: int = 0
    last_score: float = 0.5
    alert_active: bool = False
    alert: DriftAlert | None = None


def _classify_severity(drop: float, readings: int) -> DriftSeverity:
    if drop > 0.20 or readings >= 12:
        return "major"
    elif drop > 0.12 or readings >= 8:
        return "moderate"
    return "minor"


class DriftDetector:
    """Detects sustained calibration drift with hysteresis and severity levels."""

    def __init__(self) -> None:
        self._states: dict[str, _DomainState] = {d: _DomainState() for d in ALL_DOMAINS}
        self._resolved_alerts: list[DriftAlert] = []

    def update(self, domain_scores: dict[str, DomainScore]) -> list[DriftAlert]:
        """Process new domain scores, return any newly triggered alerts."""
        new_alerts: list[DriftAlert] = []
        now = time.time()

        for domain in ALL_DOMAINS:
            ds = domain_scores.get(domain)
            if ds is None or ds.provisional:
                continue

            state = self._states[domain]
            score = ds.score

            if score > state.peak_score:
                state.peak_score = score

            if score < state.last_score:
                state.consecutive_declining += 1
            else:
                state.consecutive_declining = 0

            drop = state.peak_score - score

            if state.alert_active:
                if score >= state.peak_score - DRIFT_RECOVERY_MARGIN:
                    state.alert_active = False
                    if state.alert:
                        self._resolved_alerts.append(state.alert)
                    state.alert = None
                    state.consecutive_declining = 0
                elif state.alert:
                    state.alert.score_drop = round(drop, 4)
                    state.alert.current_score = round(score, 4)
                    state.alert.readings_declining = state.consecutive_declining
                    state.alert.severity = _classify_severity(drop, state.consecutive_declining)
            else:
                if drop >= DRIFT_DROP_THRESHOLD and state.consecutive_declining >= DRIFT_SLOPE_READINGS:
                    severity = _classify_severity(drop, state.consecutive_declining)
                    alert = DriftAlert(
                        domain=domain,
                        severity=severity,
                        score_drop=round(drop, 4),
                        slope=0.0,
                        readings_declining=state.consecutive_declining,
                        peak_score=round(state.peak_score, 4),
                        current_score=round(score, 4),
                        triggered_at=now,
                    )
                    state.alert_active = True
                    state.alert = alert
                    new_alerts.append(alert)

            state.last_score = score

        return new_alerts

    def get_active_alerts(self) -> list[DriftAlert]:
        return [s.alert for s in self._states.values() if s.alert_active and s.alert]

    def get_resolved_alerts(self, limit: int = 10) -> list[DriftAlert]:
        return self._resolved_alerts[-limit:]

    def get_state(self) -> dict[str, dict]:
        result = {}
        for domain, state in self._states.items():
            result[domain] = {
                "peak_score": round(state.peak_score, 4),
                "consecutive_declining": state.consecutive_declining,
                "last_score": round(state.last_score, 4),
                "alert_active": state.alert_active,
            }
        return result
