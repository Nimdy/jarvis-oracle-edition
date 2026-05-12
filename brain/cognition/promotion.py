"""Shadow-to-active promotion for the Unified World Model and Mental Simulator.

World Model — three levels:
  0 — shadow:   log only, no behavioural influence
  1 — advisory: inject world summary into LLM context
  2 — active:   feed deltas to ProactiveGovernor + AutonomyOrchestrator

Mental Simulator — two levels:
  0 — shadow:   run simulations and log traces, no behavioural influence
  1 — advisory: inject simulation summaries into conversation context

Promotion requires sustained prediction accuracy.  Auto-demotion if accuracy
drops.  State persisted to ``~/.jarvis/`` JSON files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMOTION_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "world_model_promotion.json",
)

MIN_PREDICTIONS_FOR_PROMOTION = 50
MIN_SHADOW_HOURS = 4.0
PROMOTE_ACCURACY = 0.65
DEMOTE_ACCURACY = 0.50
DEMOTE_WINDOW = 20  # consecutive outcomes below threshold → demote
TRANSITION_COOLDOWN_S = 300.0  # 5 min cooldown between promote/demote


@dataclass
class _PromotionState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active
    shadow_start_ts: float = field(default_factory=time.time)
    total_validated: int = 0
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0


class WorldModelPromotion:
    """Accuracy-gated promotion controller."""

    def __init__(self) -> None:
        self._state = _PromotionState()
        self._load()

    # -- Public API ---------------------------------------------------------

    @property
    def level(self) -> int:
        return self._state.level

    def record_outcome(self, hit: bool) -> None:
        """Record a prediction validation outcome (True=hit, False=miss)."""
        self._state.total_validated += 1
        self._state.accuracy_history.append(1.0 if hit else 0.0)
        self._check_transitions()

    def get_status(self) -> dict[str, Any]:
        hist = list(self._state.accuracy_history)
        accuracy = sum(hist) / len(hist) if hist else 0.0
        hours_in_shadow = (time.time() - self._state.shadow_start_ts) / 3600.0
        return {
            "level": self._state.level,
            "level_name": {0: "shadow", 1: "advisory", 2: "active"}.get(
                self._state.level, "unknown"),
            "total_validated": self._state.total_validated,
            "rolling_accuracy": round(accuracy, 3),
            "rolling_window_size": len(hist),
            "hours_in_shadow": round(hours_in_shadow, 1),
            "promotion_ready": self._promotion_eligible(),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }

    def save(self) -> None:
        """Persist promotion state atomically."""
        data = {
            "level": self._state.level,
            "shadow_start_ts": self._state.shadow_start_ts,
            "total_validated": self._state.total_validated,
            "accuracy_history": list(self._state.accuracy_history),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }
        try:
            path = Path(PROMOTION_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save world model promotion state", exc_info=True)

    # -- Internals ----------------------------------------------------------

    def _load(self) -> None:
        try:
            path = Path(PROMOTION_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._state.level = data.get("level", 0)
            self._state.shadow_start_ts = data.get("shadow_start_ts", time.time())
            self._state.total_validated = data.get("total_validated", 0)
            for v in data.get("accuracy_history", []):
                self._state.accuracy_history.append(float(v))
            self._state.last_promoted_at = data.get("last_promoted_at", 0.0)
            self._state.last_demoted_at = data.get("last_demoted_at", 0.0)
            logger.info(
                "World model promotion restored: level=%d, validated=%d",
                self._state.level, self._state.total_validated,
            )
        except Exception:
            logger.debug("Failed to load world model promotion state", exc_info=True)

    def _promotion_eligible(self) -> bool:
        if self._state.level >= 2:
            return False
        if self._state.total_validated < MIN_PREDICTIONS_FOR_PROMOTION:
            return False
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        if hours < MIN_SHADOW_HOURS:
            return False
        hist = list(self._state.accuracy_history)
        if len(hist) < MIN_PREDICTIONS_FOR_PROMOTION:
            return False
        accuracy = sum(hist) / len(hist)

        effective_threshold = PROMOTE_ACCURACY
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            friction = get_quarantine_pressure().world_model_promotion_friction()
            effective_threshold += friction.get("accuracy_delta", 0.0)
            max_level = friction.get("max_level")
            if max_level is not None and self._state.level >= max_level:
                return False
        except Exception:
            pass

        return accuracy >= effective_threshold

    def _check_transitions(self) -> None:
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < TRANSITION_COOLDOWN_S:
            return

        if self._promotion_eligible():
            old = self._state.level
            self._state.level = min(self._state.level + 1, 2)
            if self._state.level != old:
                self._state.last_promoted_at = now
                logger.info(
                    "World model promoted: level %d → %d", old, self._state.level,
                )
                try:
                    from consciousness.events import WORLD_MODEL_PROMOTED, event_bus
                    event_bus.emit(
                        WORLD_MODEL_PROMOTED,
                        old_level=old,
                        new_level=self._state.level,
                        total_validated=self._state.total_validated,
                        accuracy=round(
                            sum(self._state.accuracy_history)
                            / len(self._state.accuracy_history), 3
                        ) if self._state.accuracy_history else 0.0,
                    )
                except Exception:
                    pass
                self.save()
            return

        hist = list(self._state.accuracy_history)
        if len(hist) >= DEMOTE_WINDOW and self._state.level > 0:
            recent = hist[-DEMOTE_WINDOW:]
            accuracy = sum(recent) / len(recent)
            if accuracy < DEMOTE_ACCURACY:
                old = self._state.level
                self._state.level = max(self._state.level - 1, 0)
                self._state.last_demoted_at = now
                self._state.shadow_start_ts = now
                logger.warning(
                    "World model demoted: level %d → %d (accuracy %.2f < %.2f)",
                    old, self._state.level, accuracy, DEMOTE_ACCURACY,
                )
                self.save()


# ---------------------------------------------------------------------------
# Simulator promotion (Phase 3)
# ---------------------------------------------------------------------------

SIMULATOR_PROMOTION_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "simulator_promotion.json",
)

SIM_MIN_SIMULATIONS = 100
SIM_MIN_SHADOW_HOURS = 48.0
SIM_PROMOTE_ACCURACY = 0.70
SIM_DEMOTE_ACCURACY = 0.55
SIM_DEMOTE_WINDOW = 30
SIM_TRANSITION_COOLDOWN_S = 600.0  # 10 min cooldown between promote/demote


@dataclass
class _SimPromotionState:
    level: int = 0  # 0=shadow, 1=advisory
    shadow_start_ts: float = field(default_factory=time.time)
    total_validated: int = 0
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0


class SimulatorPromotion:
    """Accuracy-gated promotion for the Mental Simulator (shadow → advisory)."""

    def __init__(self) -> None:
        self._state = _SimPromotionState()
        self._load()

    @property
    def level(self) -> int:
        return self._state.level

    def record_outcome(self, hit: bool) -> None:
        """Record whether a shadow simulation's prediction matched reality."""
        self._state.total_validated += 1
        self._state.accuracy_history.append(1.0 if hit else 0.0)
        self._check_transitions()

    def get_status(self) -> dict[str, Any]:
        hist = list(self._state.accuracy_history)
        accuracy = sum(hist) / len(hist) if hist else 0.0
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        return {
            "level": self._state.level,
            "level_name": {0: "shadow", 1: "advisory"}.get(self._state.level, "unknown"),
            "total_validated": self._state.total_validated,
            "rolling_accuracy": round(accuracy, 3),
            "rolling_window_size": len(hist),
            "hours_in_shadow": round(hours, 1),
            "promotion_ready": self._promotion_eligible(),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }

    def save(self) -> None:
        data = {
            "level": self._state.level,
            "shadow_start_ts": self._state.shadow_start_ts,
            "total_validated": self._state.total_validated,
            "accuracy_history": list(self._state.accuracy_history),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }
        try:
            path = Path(SIMULATOR_PROMOTION_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save simulator promotion state", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(SIMULATOR_PROMOTION_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._state.level = data.get("level", 0)
            self._state.shadow_start_ts = data.get("shadow_start_ts", time.time())
            self._state.total_validated = data.get("total_validated", 0)
            for v in data.get("accuracy_history", []):
                self._state.accuracy_history.append(float(v))
            self._state.last_promoted_at = data.get("last_promoted_at", 0.0)
            self._state.last_demoted_at = data.get("last_demoted_at", 0.0)
            if self._state.total_validated > 0:
                logger.info(
                    "Simulator promotion restored: level=%d, validated=%d",
                    self._state.level, self._state.total_validated,
                )
        except Exception:
            logger.debug("Failed to load simulator promotion state", exc_info=True)

    def _promotion_eligible(self) -> bool:
        if self._state.level >= 1:
            return False
        if self._state.total_validated < SIM_MIN_SIMULATIONS:
            return False
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        if hours < SIM_MIN_SHADOW_HOURS:
            return False
        hist = list(self._state.accuracy_history)
        if len(hist) < SIM_MIN_SIMULATIONS:
            return False
        accuracy = sum(hist) / len(hist)
        return accuracy >= SIM_PROMOTE_ACCURACY

    def _check_transitions(self) -> None:
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < SIM_TRANSITION_COOLDOWN_S:
            return

        if self._promotion_eligible():
            old = self._state.level
            self._state.level = 1
            if self._state.level != old:
                self._state.last_promoted_at = now
                logger.info(
                    "Simulator promoted: shadow → advisory (validated=%d)",
                    self._state.total_validated,
                )
                self.save()
            return

        hist = list(self._state.accuracy_history)
        if len(hist) >= SIM_DEMOTE_WINDOW and self._state.level > 0:
            recent = hist[-SIM_DEMOTE_WINDOW:]
            accuracy = sum(recent) / len(recent)
            if accuracy < SIM_DEMOTE_ACCURACY:
                self._state.level = 0
                self._state.last_demoted_at = now
                self._state.shadow_start_ts = now
                logger.warning(
                    "Simulator demoted: advisory → shadow (accuracy %.2f < %.2f)",
                    accuracy, SIM_DEMOTE_ACCURACY,
                )
                self.save()
