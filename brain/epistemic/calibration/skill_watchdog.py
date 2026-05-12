"""Skill watchdog: monitors verified skills for degradation.

Once a skill is verified, it should not degrade silently. The watchdog
tracks rolling pass rates per verified skill and emits alerts when
performance drops below the threshold.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jarvis.calibration.skill_watchdog")

SKILL_DEGRADATION_MIN_EVIDENCE = 8
SKILL_DEGRADATION_THRESHOLD = 0.5


@dataclass
class _SkillWindow:
    results: deque  # deque[bool]
    last_alert_time: float = 0.0


class SkillWatchdog:
    """Monitors verified skills for declining pass rates."""

    def __init__(self) -> None:
        self._windows: dict[str, _SkillWindow] = {}
        self._degraded_skills: set[str] = set()
        self._total_alerts: int = 0
        self._subscribed = False

    def _ensure_subscribed(self) -> None:
        if self._subscribed:
            return
        self._subscribed = True
        try:
            from consciousness.events import event_bus, SKILL_VERIFICATION_RECORDED
            event_bus.on(SKILL_VERIFICATION_RECORDED, self._on_verification)
        except Exception as exc:
            logger.debug("SkillWatchdog subscription failed: %s", exc)

    def _on_verification(self, skill_id: str = "", passed: bool = False, **kwargs: Any) -> None:
        if not skill_id:
            return
        if skill_id not in self._windows:
            self._windows[skill_id] = _SkillWindow(results=deque(maxlen=20))
        self._windows[skill_id].results.append(passed)

    def tick(self) -> list[dict]:
        """Check skill windows for degradation. Returns new alerts."""
        self._ensure_subscribed()
        alerts: list[dict] = []
        now = time.time()

        for skill_id, window in self._windows.items():
            if len(window.results) < SKILL_DEGRADATION_MIN_EVIDENCE:
                continue

            pass_rate = sum(1 for r in window.results if r) / len(window.results)

            if pass_rate < SKILL_DEGRADATION_THRESHOLD:
                if skill_id not in self._degraded_skills:
                    self._degraded_skills.add(skill_id)
                    self._total_alerts += 1
                    alert = {
                        "skill_id": skill_id,
                        "pass_rate": round(pass_rate, 4),
                        "evidence_count": len(window.results),
                        "timestamp": now,
                    }
                    alerts.append(alert)
                    self._emit_alert(alert)
                    window.last_alert_time = now
            else:
                self._degraded_skills.discard(skill_id)

        return alerts

    def _emit_alert(self, alert: dict) -> None:
        try:
            from consciousness.events import event_bus, SKILL_DEGRADATION_DETECTED
            event_bus.emit(SKILL_DEGRADATION_DETECTED, **alert)
        except Exception:
            pass

    def get_stats(self) -> dict:
        return {
            "monitored_skills": len(self._windows),
            "degraded_skills": list(self._degraded_skills),
            "total_alerts": self._total_alerts,
        }
