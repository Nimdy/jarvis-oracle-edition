"""Trait evolution validator — compatibility matrix and contradiction detection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

ValidationResult = Literal["approve", "warn", "reject"]

JARVIS_TRAITS = (
    "proactive", "efficient", "detail_oriented", "empathetic",
    "technical", "humor_adaptive", "privacy_conscious",
)

# Compatibility scores: 1.0 = compatible, 0.5 = neutral, 0.0 = incompatible
COMPATIBILITY_MATRIX: dict[tuple[str, str], float] = {
    ("proactive", "efficient"): 1.0,
    ("proactive", "detail_oriented"): 0.5,
    ("proactive", "empathetic"): 0.7,
    ("proactive", "technical"): 0.5,
    ("proactive", "humor_adaptive"): 0.8,
    ("proactive", "privacy_conscious"): 0.3,
    ("efficient", "detail_oriented"): 0.4,  # tension
    ("efficient", "empathetic"): 0.5,
    ("efficient", "technical"): 0.9,
    ("efficient", "humor_adaptive"): 0.5,
    ("efficient", "privacy_conscious"): 0.7,
    ("detail_oriented", "empathetic"): 0.8,
    ("detail_oriented", "technical"): 0.9,
    ("detail_oriented", "humor_adaptive"): 0.4,  # tension
    ("detail_oriented", "privacy_conscious"): 0.7,
    ("empathetic", "technical"): 0.5,
    ("empathetic", "humor_adaptive"): 0.8,
    ("empathetic", "privacy_conscious"): 0.3,  # tension
    ("technical", "humor_adaptive"): 0.4,  # mild tension
    ("technical", "privacy_conscious"): 0.7,
    ("humor_adaptive", "privacy_conscious"): 0.5,
}


def _normalize_trait(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _get_compatibility(trait_a: str, trait_b: str) -> float:
    a = _normalize_trait(trait_a)
    b = _normalize_trait(trait_b)
    if a == b:
        return 1.0
    key = (min(a, b), max(a, b))
    return COMPATIBILITY_MATRIX.get(key, 0.5)


@dataclass
class TraitContradiction:
    trait_a: str
    trait_b: str
    contradiction_type: str  # direct_opposite, behavioral_clash, intensity_mismatch
    severity: float  # 0-1
    description: str


@dataclass
class ValidationReport:
    result: ValidationResult
    contradictions: list[TraitContradiction] = field(default_factory=list)
    coherence_score: float = 1.0
    stability_score: float = 1.0
    evolution_rate: float = 0.0
    warnings: list[str] = field(default_factory=list)
    trend: str = "stable"  # stabilizing, evolving, fragmenting, consolidating


class TraitEvolutionValidator:
    """Validates trait changes before they're applied."""

    MAX_NET_DRIFT_PER_TRAIT_HOUR = 0.4
    RATE_REJECT_THRESHOLD = 0.3  # per-trait average drift per hour
    GESTATION_RATE_REJECT_THRESHOLD = 0.8
    GESTATION_MAX_NET_DRIFT_PER_TRAIT_HOUR = 0.7
    POST_BIRTH_GRACE_S = 1800.0  # 30 min relaxed thresholds after gestation
    POST_BIRTH_RATE_REJECT_THRESHOLD = 0.6
    POST_BIRTH_MAX_NET_DRIFT_PER_TRAIT_HOUR = 0.6

    def __init__(self) -> None:
        self._baseline_snapshot: dict[str, float] | None = None
        self._baseline_time: float = 0.0
        self._snapshot_history: list[dict[str, float]] = []
        self._last_validation: float = 0.0
        self._validation_count: int = 0
        self._last_stability: float | None = None
        self._birth_time: float = 0.0

    @property
    def last_stability(self) -> float | None:
        """Most recent stability score computed during validate(), or None if never run."""
        return self._last_stability

    def mark_birth(self) -> None:
        """Called when gestation completes to start the post-birth grace window."""
        self._birth_time = time.time()
        logger.info("Trait validator: post-birth grace period started (%.0fs)", self.POST_BIRTH_GRACE_S)

    @property
    def in_post_birth_grace(self) -> bool:
        return self._birth_time > 0 and (time.time() - self._birth_time) < self.POST_BIRTH_GRACE_S

    @staticmethod
    def _is_gestation() -> bool:
        try:
            from consciousness.modes import mode_manager
            return mode_manager.mode == "gestation"
        except Exception:
            return False

    def validate(self, current_traits: dict[str, float],
                 proposed_traits: dict[str, float]) -> ValidationReport:
        """Validate proposed trait changes. Returns approval/warning/rejection."""
        now = time.time()
        self._last_validation = now
        self._validation_count += 1

        if self._baseline_snapshot is None:
            self._baseline_snapshot = dict(proposed_traits) if proposed_traits else dict(current_traits)
            self._baseline_time = now

        # Reset baseline every hour so rate limit resets naturally
        if now - self._baseline_time > 3600:
            self._baseline_snapshot = dict(current_traits)
            self._baseline_time = now

        contradictions = self._detect_contradictions(proposed_traits)
        coherence = self._compute_coherence(proposed_traits)
        stability = self._compute_stability(proposed_traits)
        self._last_stability = stability
        rate = self._compute_evolution_rate(proposed_traits)
        trend = self._compute_trend()

        self._snapshot_history.append(dict(proposed_traits))
        if len(self._snapshot_history) > 20:
            self._snapshot_history = self._snapshot_history[-20:]

        gestation = self._is_gestation()
        post_birth = self.in_post_birth_grace
        if gestation:
            rate_threshold = self.GESTATION_RATE_REJECT_THRESHOLD
        elif post_birth:
            rate_threshold = self.POST_BIRTH_RATE_REJECT_THRESHOLD
        else:
            rate_threshold = self.RATE_REJECT_THRESHOLD
        rate_limited = self._check_rate_limits(proposed_traits, gestation=gestation, post_birth=post_birth)

        warnings: list[str] = []
        result: ValidationResult = "approve"

        # REJECT conditions
        if any(c.severity > 0.5 for c in contradictions):
            result = "reject"
            warnings.append("High-severity contradiction detected")
        elif stability < 0.2:
            result = "reject"
            warnings.append(f"Stability too low ({stability:.2f})")
        elif rate > rate_threshold:
            result = "reject"
            warnings.append(f"Evolution rate too high ({rate:.2f}/hour)")
        elif rate_limited:
            result = "reject"
            warnings.append("Single trait drifted too far in one hour")
        # WARN conditions
        elif len(contradictions) > 1:
            result = "warn"
            warnings.append(f"{len(contradictions)} contradictions detected")
        elif coherence < 0.5:
            result = "warn"
            warnings.append(f"Low coherence ({coherence:.2f})")
        elif stability < 0.4:
            result = "warn"
            warnings.append(f"Low stability ({stability:.2f})")

        return ValidationReport(
            result=result,
            contradictions=contradictions,
            coherence_score=round(coherence, 4),
            stability_score=round(stability, 4),
            evolution_rate=round(rate, 4),
            warnings=warnings,
            trend=trend,
        )

    def _detect_contradictions(self, traits: dict[str, float]) -> list[TraitContradiction]:
        contradictions = []
        trait_names = list(traits.keys())
        for i, a in enumerate(trait_names):
            for b in trait_names[i+1:]:
                compat = _get_compatibility(a, b)
                val_a, val_b = traits[a], traits[b]

                # Direct opposite: low compatibility + both strong
                if compat < 0.3 and val_a > 0.6 and val_b > 0.6:
                    contradictions.append(TraitContradiction(
                        trait_a=a, trait_b=b,
                        contradiction_type="direct_opposite",
                        severity=min(1.0, (1.0 - compat) * val_a * val_b),
                        description=f"{a} and {b} are fundamentally opposed at high intensity",
                    ))
                # Behavioral clash: tension + both active
                elif compat < 0.5 and val_a > 0.5 and val_b > 0.5:
                    contradictions.append(TraitContradiction(
                        trait_a=a, trait_b=b,
                        contradiction_type="behavioral_clash",
                        severity=(1.0 - compat) * 0.6,
                        description=f"{a} and {b} create behavioral tension",
                    ))
                # Intensity mismatch: big difference in paired traits
                if compat > 0.7 and abs(val_a - val_b) > 0.5:
                    contradictions.append(TraitContradiction(
                        trait_a=a, trait_b=b,
                        contradiction_type="intensity_mismatch",
                        severity=abs(val_a - val_b) * 0.4,
                        description=f"Compatible traits {a}/{b} have mismatched intensity",
                    ))

        return contradictions

    def _compute_coherence(self, traits: dict[str, float]) -> float:
        if len(traits) < 2:
            return 1.0
        total_compat = 0.0
        count = 0
        names = list(traits.keys())
        for i, a in enumerate(names):
            for b in names[i+1:]:
                total_compat += _get_compatibility(a, b)
                count += 1
        return total_compat / count if count > 0 else 1.0

    def _compute_stability(self, current: dict[str, float]) -> float:
        if len(self._snapshot_history) < 3:
            return 1.0

        recent = self._snapshot_history[-10:]

        # Factor 1: Trait variability
        variances = []
        for trait in current:
            vals = [s.get(trait, 0.5) for s in recent]
            if len(vals) > 1:
                mean = sum(vals) / len(vals)
                var = sum((v - mean)**2 for v in vals) / len(vals)
                variances.append(var)
        avg_var = sum(variances) / max(len(variances), 1)
        variability = 1.0 - min(1.0, avg_var * 5.0)

        # Factor 2: Strength balance (mild penalty for a single dominant trait)
        max_val = max(current.values()) if current else 0.5
        balance = 1.0 if max_val <= 0.8 else max(0.5, 1.0 - (max_val - 0.8) * 1.5)

        # Factor 3: Contradiction penalty
        contradictions = self._detect_contradictions(current)
        contradiction_penalty = max(0.1, 1.0 - len(contradictions) * 0.1)

        return variability * balance * contradiction_penalty

    def _compute_evolution_rate(self, proposed: dict[str, float]) -> float:
        """Average per-trait drift from baseline — scales with trait count."""
        if self._baseline_snapshot is None:
            return 0.0
        all_traits = set(proposed) | set(self._baseline_snapshot)
        n = len(all_traits)
        if n == 0:
            return 0.0
        total_net_drift = sum(
            abs(proposed.get(t, 0.5) - self._baseline_snapshot.get(t, 0.5))
            for t in all_traits
        )
        return total_net_drift / n

    def _check_rate_limits(self, proposed: dict[str, float], *, gestation: bool = False,
                           post_birth: bool = False) -> bool:
        """Reject if any single trait drifted too far from baseline.

        When stability < 0.5 (via rollback emergency or recent thrashing),
        the drift cap is halved to dampen oscillations.
        """
        if self._baseline_snapshot is None:
            return False
        if gestation:
            cap = self.GESTATION_MAX_NET_DRIFT_PER_TRAIT_HOUR
        elif post_birth:
            cap = self.POST_BIRTH_MAX_NET_DRIFT_PER_TRAIT_HOUR
        else:
            cap = self.MAX_NET_DRIFT_PER_TRAIT_HOUR
        stability = self._compute_stability(proposed)
        if stability < 0.5:
            cap *= 0.5
        for trait, new_val in proposed.items():
            baseline_val = self._baseline_snapshot.get(trait, 0.5)
            if abs(new_val - baseline_val) > cap:
                return True
        return False

    def _compute_trend(self) -> str:
        if len(self._snapshot_history) < 3:
            return "stable"

        recent = self._snapshot_history[-5:]
        change_counts = 0
        total_change = 0.0
        for i in range(1, len(recent)):
            for trait in recent[i]:
                delta = abs(recent[i].get(trait, 0.5) - recent[i-1].get(trait, 0.5))
                if delta > 0.01:
                    change_counts += 1
                    total_change += delta

        if change_counts == 0:
            return "stabilizing"
        elif total_change > 0.5:
            return "fragmenting" if change_counts > 5 else "evolving"
        else:
            return "consolidating"

    def get_state(self) -> dict[str, Any]:
        return {
            "total_validations": self._validation_count,
            "snapshot_history_len": len(self._snapshot_history),
            "last_validation": self._last_validation,
            "has_baseline": self._baseline_snapshot is not None,
            "last_stability": round(self._last_stability, 4) if self._last_stability is not None else None,
            "in_post_birth_grace": self.in_post_birth_grace,
        }


trait_validator = TraitEvolutionValidator()
