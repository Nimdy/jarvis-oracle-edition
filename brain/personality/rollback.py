"""Personality rollback system — stability monitoring, snapshots, and auto-rollback."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from consciousness.events import event_bus, PERSONALITY_ROLLBACK

logger = logging.getLogger(__name__)


@dataclass
class PersonalitySnapshot:
    timestamp: float
    traits: dict[str, float]
    coherence_score: float
    stability_score: float
    rollback_risk: float  # 0 = safe, 1 = likely to trigger rollback


class PersonalityRollbackSystem:
    """Monitors personality stability and rolls back to safe states when needed."""

    SNAPSHOT_INTERVAL_S = 30.0
    MAX_SNAPSHOTS = 20
    STABILITY_THRESHOLD = 0.3
    ROLLBACK_COOLDOWN_S = 300.0  # 5 min
    EMERGENCY_DURATION_S = 600.0  # 10 min

    def __init__(self, validator: Any | None = None) -> None:
        self._snapshots: list[PersonalitySnapshot] = []
        self._last_snapshot_time: float = 0.0
        self._last_rollback_time: float = 0.0
        self._emergency_until: float = 0.0
        self._rollback_count: int = 0
        self._current_traits: dict[str, float] = {}
        self._validator = validator
        self._stability_source: str = "rollback_local"

    def set_validator(self, validator: Any) -> None:
        """Inject the trait validator for stability fallback delegation."""
        self._validator = validator

    @property
    def in_emergency(self) -> bool:
        return time.time() < self._emergency_until

    def update_traits(self, traits: dict[str, float]) -> None:
        self._current_traits = dict(traits)

    def tick(self, coherence: float = 1.0) -> dict[str, Any] | None:
        """Called periodically. Takes snapshot if needed, checks stability, triggers rollback."""
        now = time.time()

        # Emergency mode — no actions
        if now < self._emergency_until:
            return None

        if not self._current_traits:
            return None

        # Take periodic snapshot
        if now - self._last_snapshot_time >= self.SNAPSHOT_INTERVAL_S:
            stability = self._compute_stability()
            risk = max(0.0, 1.0 - stability)
            snapshot = PersonalitySnapshot(
                timestamp=now,
                traits=dict(self._current_traits),
                coherence_score=coherence,
                stability_score=stability,
                rollback_risk=risk,
            )
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.MAX_SNAPSHOTS:
                self._snapshots = self._snapshots[-self.MAX_SNAPSHOTS:]
            self._last_snapshot_time = now

            # Check if rollback needed
            if stability < self.STABILITY_THRESHOLD:
                if now - self._last_rollback_time >= self.ROLLBACK_COOLDOWN_S:
                    target = self._find_rollback_target()
                    if target:
                        benefit = self._assess_benefit(target, stability)
                        if benefit >= 0.3:
                            return self._execute_rollback(target, now)
                        else:
                            logger.info("Rollback benefit too low (%.2f), skipping", benefit)

        return None

    def _compute_stability(self) -> float:
        if len(self._snapshots) < 3:
            if self._validator is not None:
                cached = getattr(self._validator, "last_stability", None)
                if cached is not None:
                    self._stability_source = "validator_fallback"
                    return cached
            self._stability_source = "rollback_local"
            return 1.0

        self._stability_source = "rollback_local"
        recent = self._snapshots[-10:]

        # Factor 1: Trait variability across snapshots
        all_traits = set()
        for s in recent:
            all_traits.update(s.traits.keys())
        variances = []
        for trait in all_traits:
            vals = [s.traits.get(trait, 0.5) for s in recent]
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                variances.append(var)
        variability_factor = 1.0 - min(1.0, sum(variances) / max(len(variances), 1) * 10)

        # Factor 2: Memory-trait alignment (simplified — use coherence as proxy)
        if recent:
            alignment_factor = sum(s.coherence_score for s in recent) / len(recent)
        else:
            alignment_factor = 1.0

        # Factor 3: Trait strength balance
        if self._current_traits:
            max_val = max(self._current_traits.values())
            balance_factor = 1.0 if max_val <= 0.8 else max(0.3, 1.0 - (max_val - 0.8) * 3)
        else:
            balance_factor = 1.0

        # Factor 4: Contradiction penalty (simplified via coherence spread)
        coherence_spread = max(s.coherence_score for s in recent) - min(s.coherence_score for s in recent) if recent else 0.0
        contradiction_factor = max(0.1, 1.0 - coherence_spread)

        return variability_factor * alignment_factor * balance_factor * contradiction_factor

    def _find_rollback_target(self) -> PersonalitySnapshot | None:
        candidates = [
            s for s in self._snapshots
            if s.stability_score > 0.5 and s.coherence_score > 0.6
        ]
        if not candidates:
            return None
        # Prefer older, more stable states
        candidates.sort(key=lambda s: (s.stability_score, s.coherence_score), reverse=True)
        return candidates[0]

    def _assess_benefit(self, target: PersonalitySnapshot, current_stability: float) -> float:
        stability_improvement = target.stability_score - current_stability
        # Estimate disruption
        if self._current_traits and target.traits:
            total_change = sum(
                abs(target.traits.get(t, 0.5) - self._current_traits.get(t, 0.5))
                for t in set(target.traits) | set(self._current_traits)
            )
            memory_disruption = min(1.0, total_change / 2.0)
        else:
            memory_disruption = 0.0
        learning_loss = min(1.0, len(self._snapshots) * 0.02)
        return stability_improvement - memory_disruption * 0.3 - learning_loss * 0.2

    def _execute_rollback(self, target: PersonalitySnapshot, now: float) -> dict[str, Any]:
        logger.warning(
            "Personality rollback triggered — reverting to snapshot from %.0fs ago (stability %.2f → %.2f)",
            now - target.timestamp, self._compute_stability(), target.stability_score,
        )
        old_traits = dict(self._current_traits)
        self._current_traits = dict(target.traits)
        self._last_rollback_time = now
        self._rollback_count += 1

        # Enter emergency protection mode
        self._emergency_until = now + self.EMERGENCY_DURATION_S

        event_bus.emit(
            PERSONALITY_ROLLBACK,
            old_traits=old_traits,
            new_traits=target.traits,
            reason="stability_below_threshold",
            target_stability=target.stability_score,
        )

        return {
            "action": "rollback",
            "old_traits": old_traits,
            "new_traits": dict(target.traits),
            "target_stability": target.stability_score,
            "emergency_until": self._emergency_until,
        }

    def get_state(self) -> dict[str, Any]:
        stability = self._compute_stability()
        return {
            "snapshot_count": len(self._snapshots),
            "rollback_count": self._rollback_count,
            "current_stability": stability,
            "stability_source": self._stability_source,
            "in_emergency": time.time() < self._emergency_until,
            "last_rollback": self._last_rollback_time,
            "current_traits": dict(self._current_traits),
        }

    def get_rollback_target(self) -> PersonalitySnapshot | None:
        return self._find_rollback_target()


personality_rollback = PersonalityRollbackSystem()

try:
    from personality.validator import trait_validator
    personality_rollback.set_validator(trait_validator)
except Exception:
    pass
