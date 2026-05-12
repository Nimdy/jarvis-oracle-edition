"""Layer 8 Active-Lite: Quarantine Pressure + Friction Contract.

Translates shadow-mode QuarantineScorer signals into a composite
pressure metric that downstream consumers read to modulate their
existing gates.  The scorer is never modified — all active-lite
behaviour lives here and in the consumers that read PressureState.

Design invariants:
  1. Memories always write — pressure never blocks a store.
  2. Tags + weight reduction are the only per-memory mutations.
  3. Friction is proportional — elevated raises thresholds, high blocks.
  4. Pressure decays via EMA — no permanent penalty.
  5. Category-to-action policy is centralised in CATEGORY_POLICY.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from epistemic.quarantine.scorer import (
    CATEGORY_CALIBRATION,
    CATEGORY_CONTRADICTION,
    CATEGORY_IDENTITY,
    CATEGORY_MANIPULATION,
    CATEGORY_MEMORY,
    QuarantineSignal,
)

logger = logging.getLogger(__name__)

ELEVATED_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6

_EMA_ALPHA = 0.3
_EMA_DECAY = 0.85

_CATEGORY_WEIGHTS: dict[str, float] = {
    CATEGORY_IDENTITY: 0.25,
    CATEGORY_CONTRADICTION: 0.25,
    CATEGORY_MEMORY: 0.20,
    CATEGORY_CALIBRATION: 0.15,
    CATEGORY_MANIPULATION: 0.15,
}

_CHRONIC_BONUS = 0.05

QUARANTINE_SUSPECT_TAG = "quarantine:suspect"


# ---------------------------------------------------------------------------
# Category policy — single source of truth for per-category behaviour
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryPolicy:
    """Defines what active-lite can do when a given category is elevated."""

    taggable: bool = False
    belief_block: bool = False
    memory_match: Callable[[Any], bool] | None = None


def _match_identity(m: Any) -> bool:
    return bool(
        getattr(m, "identity_needs_resolution", False)
        or getattr(m, "identity_confidence", 1.0) < 0.45
    )


def _match_user_claim(m: Any) -> bool:
    return getattr(m, "provenance", "unknown") == "user_claim"


CATEGORY_POLICY: dict[str, CategoryPolicy] = {
    CATEGORY_IDENTITY: CategoryPolicy(
        taggable=True,
        belief_block=True,
        memory_match=_match_identity,
    ),
    CATEGORY_MEMORY: CategoryPolicy(
        taggable=True,
        belief_block=True,
        memory_match=lambda m: True,
    ),
    CATEGORY_MANIPULATION: CategoryPolicy(
        taggable=True,
        belief_block=True,
        memory_match=_match_user_claim,
    ),
    CATEGORY_CONTRADICTION: CategoryPolicy(
        taggable=False,
        belief_block=False,
    ),
    CATEGORY_CALIBRATION: CategoryPolicy(
        taggable=False,
        belief_block=False,
    ),
}


# ---------------------------------------------------------------------------
# Frozen snapshot exposed to consumers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PressureState:
    composite: float
    by_category: dict[str, float]
    chronic_count: int
    elevated: bool
    high: bool
    timestamp: float


_ZERO_CATEGORIES = {
    CATEGORY_IDENTITY: 0.0,
    CATEGORY_CONTRADICTION: 0.0,
    CATEGORY_MEMORY: 0.0,
    CATEGORY_CALIBRATION: 0.0,
    CATEGORY_MANIPULATION: 0.0,
}

PRESSURE_NORMAL = PressureState(
    composite=0.0,
    by_category=dict(_ZERO_CATEGORIES),
    chronic_count=0,
    elevated=False,
    high=False,
    timestamp=0.0,
)


# ---------------------------------------------------------------------------
# QuarantinePressure singleton
# ---------------------------------------------------------------------------

class QuarantinePressure:
    """Computes and caches a PressureState snapshot after each scorer tick.

    All downstream friction consumers read ``current`` — never the scorer
    directly.  This keeps the scorer pure and the policy centralised.
    """

    def __init__(self) -> None:
        self._category_ema: dict[str, float] = dict(_ZERO_CATEGORIES)
        self._current: PressureState = PRESSURE_NORMAL
        self._memories_tagged: int = 0
        self._promotions_blocked: int = 0

    @property
    def current(self) -> PressureState:
        return self._current

    @property
    def memories_tagged(self) -> int:
        return self._memories_tagged

    @property
    def promotions_blocked(self) -> int:
        return self._promotions_blocked

    def record_promotion_blocked(self) -> None:
        self._promotions_blocked += 1

    # ------------------------------------------------------------------
    # Tick — called once at the end of _run_quarantine_tick
    # ------------------------------------------------------------------

    def update(self, signals: list[QuarantineSignal], chronic_count: int) -> PressureState:
        """Recompute pressure from latest scorer signals."""
        now = time.time()

        scores_by_cat: dict[str, float] = {}
        for sig in signals:
            cat = sig.category
            prev = scores_by_cat.get(cat, 0.0)
            scores_by_cat[cat] = max(prev, sig.score)

        for cat in self._category_ema:
            if cat in scores_by_cat:
                self._category_ema[cat] = (
                    _EMA_ALPHA * scores_by_cat[cat]
                    + (1 - _EMA_ALPHA) * self._category_ema[cat]
                )
            else:
                self._category_ema[cat] *= _EMA_DECAY

        for cat in list(self._category_ema):
            chronic_for_cat = sum(
                1 for sig in signals if sig.category == cat and sig.is_chronic
            )
            self._category_ema[cat] = min(
                1.0,
                self._category_ema[cat] + chronic_for_cat * _CHRONIC_BONUS,
            )

        composite = sum(
            self._category_ema[cat] * _CATEGORY_WEIGHTS.get(cat, 0.0)
            for cat in self._category_ema
        )
        composite = max(0.0, min(1.0, composite))

        self._current = PressureState(
            composite=composite,
            by_category=dict(self._category_ema),
            chronic_count=chronic_count,
            elevated=composite > ELEVATED_THRESHOLD,
            high=composite > HIGH_THRESHOLD,
            timestamp=now,
        )
        return self._current

    # ------------------------------------------------------------------
    # Friction contract helpers — consumers call these, never raw fields
    # ------------------------------------------------------------------

    def should_tag_memory(self, memory: Any) -> tuple[bool, list[str]]:
        """Return (should_tag, [categories]) for a memory about to be written.

        Only tags when pressure is elevated AND at least one taggable
        category's ``memory_match`` predicate fires on this memory.
        """
        p = self._current
        if not p.elevated:
            return False, []

        matched_cats: list[str] = []
        for cat, policy in CATEGORY_POLICY.items():
            if not policy.taggable:
                continue
            cat_pressure = p.by_category.get(cat, 0.0)
            if cat_pressure <= 0.05:
                continue
            if policy.memory_match is not None and policy.memory_match(memory):
                matched_cats.append(cat)

        if matched_cats:
            self._memories_tagged += 1
            return True, matched_cats
        return False, []

    def weight_multiplier(self) -> float:
        """Scaling factor applied to memory weight under pressure.

        Returns 1.0 at normal, down to 0.6 at max pressure.
        """
        p = self._current
        if not p.elevated:
            return 1.0
        return max(0.6, 1.0 - p.composite * 0.4)

    def mutation_risk_addon(self) -> float:
        """Risk score added to the mutation governor's assessment."""
        return self._current.composite * 0.3

    def mutation_rate_factor(self) -> tuple[int | None, int | None]:
        """Returns (hourly_cap_override, cooldown_override) or None for default."""
        p = self._current
        if p.high:
            return 6, 360
        return None, None

    def should_reject_identity_mutation(self, identity_risk: float) -> bool:
        p = self._current
        return p.high and p.chronic_count > 0 and identity_risk > 0.3

    def policy_promotion_friction(self) -> dict[str, Any]:
        """Returns friction params for PolicyPromotion.

        Keys: win_threshold_delta, min_decisions_delta, block, allow_rollback
        """
        p = self._current
        if p.high and p.chronic_count > 0:
            self._promotions_blocked += 1
            return {
                "win_threshold_delta": p.composite * 0.15,
                "min_decisions_delta": int(p.composite * 100),
                "block": True,
                "allow_rollback": True,
            }
        if p.high:
            self._promotions_blocked += 1
            return {
                "win_threshold_delta": p.composite * 0.15,
                "min_decisions_delta": int(p.composite * 100),
                "block": True,
                "allow_rollback": False,
            }
        if p.elevated:
            return {
                "win_threshold_delta": p.composite * 0.15,
                "min_decisions_delta": int(p.composite * 100),
                "block": False,
                "allow_rollback": False,
            }
        return {
            "win_threshold_delta": 0.0,
            "min_decisions_delta": 0,
            "block": False,
            "allow_rollback": False,
        }

    def world_model_promotion_friction(self) -> dict[str, Any]:
        """Returns friction params for WorldModelPromotion.

        Keys: accuracy_delta, max_level (None = no cap)
        """
        p = self._current
        if p.high and p.chronic_count > 0:
            return {"accuracy_delta": p.composite * 0.15, "max_level": 0}
        if p.high:
            return {"accuracy_delta": p.composite * 0.15, "max_level": 1}
        if p.elevated:
            return {"accuracy_delta": p.composite * 0.15, "max_level": None}
        return {"accuracy_delta": 0.0, "max_level": None}

    def graph_support_gate_delta(self) -> float:
        """Extra extraction-confidence required for support edges."""
        p = self._current
        if p.elevated or p.high:
            return p.composite * 0.2
        return 0.0

    # ------------------------------------------------------------------
    # Dashboard snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        p = self._current
        band = "high" if p.high else ("elevated" if p.elevated else "normal")
        return {
            "composite": round(p.composite, 3),
            "band": band,
            "by_category": {k: round(v, 3) for k, v in p.by_category.items()},
            "chronic_count": p.chronic_count,
            "memories_tagged": self._memories_tagged,
            "promotions_blocked": self._promotions_blocked,
            "timestamp": p.timestamp,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: QuarantinePressure | None = None


def get_quarantine_pressure() -> QuarantinePressure:
    global _instance
    if _instance is None:
        _instance = QuarantinePressure()
    return _instance
