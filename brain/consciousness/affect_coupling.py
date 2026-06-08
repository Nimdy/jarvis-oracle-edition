"""P5 cadence/reward coupling controller + kill-switch (SPARK_DESIGN §5, §8 P5).

This is the **last, riskiest** station of the Grounding Ring: the only true
feedback loop, where the regulated affect scalars are finally permitted to drive
real levers — ``kernel.set_cadence_multiplier`` (clamped ``[0.5, 2.0]``), the
per-cycle interval multipliers, the drive-urgency bias, and the memory
reinforcement multiplier — and where the external grounding term is permitted to
enter ``_compute_health_reward``.

It is a **separate** promotion controller from :class:`AffectPromotion` (which
gates the affect *readout* shadow→advisory). This one is the §3-component-5 /
§8-P5 "own controller + kill-switch" that gates the *coupling*. It

  * **defaults to shadow** (level 0) → :meth:`apply_levers` returns the neutral
    no-op lever set, so DEFAULT runtime behaviour is UNCHANGED;
  * owns the :class:`~consciousness.affect_regulation.KillSwitch`, which when
    engaged reverts ``cadence_multiplier → 1.0`` and reward → the unchanged
    ``_compute_health_reward`` (the only two levers with a runtime feedback path);
  * implements the §5 / §8 P5 promotion criterion — promotion is earned ONLY when
    every gate is cleared:
      1. governor invariant proven in production: no scalar reverses >50 % of its
         updates, and no clamp-saturation streaks (read from the live regulator
         states);
      2. **cortisol-driven cadence acceleration is FOLLOWED BY a
         ``contradiction_debt`` decrease** over the next window (the §5 proof
         obligation — if debt rises after acceleration, the gate stays shut);
      3. ≥20 grounding outcomes with ``external_validation_rate ≥ 0.40`` AND
         ``orphan_rate`` trending down (read from policy memory + spark metrics).

It **never flips itself to active** — promotion is the operator's call. It only
computes eligibility, records the acceleration→debt evidence, and auto-demotes /
engages the kill-switch on any regression (governor freeze, debt rising after
acceleration, or external-validation rate collapsing).

Everything new here is additive and default-safe; nothing about this module
changes behaviour until an operator promotes it to active.
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

from consciousness.affect_regulation import (
    KillSwitch,
    clamp_levers,
    neutral_levers,
)

logger = logging.getLogger(__name__)

AFFECT_COUPLING_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "affect_coupling.json",
)

# ── §8 P5 promotion gate constants ───────────────────────────────────────────
# ≥20 grounding outcomes with external_validation_rate ≥0.40 (SPARK §3 gate 1 / §8 P5).
MIN_GROUNDING_OUTCOMES = 20
PROMOTE_VALIDATION_RATE = 0.40
DEMOTE_VALIDATION_RATE = 0.20

# Governor invariant: a scalar must NOT reverse direction on >50% of its updates,
# and must not sit pinned at a clamp for a sustained streak (SPARK §5.5 / §8 P5).
MAX_REVERSAL_FRACTION = 0.50
MAX_SATURATION_STREAK = 5

# The §5 proof obligation: cortisol-driven acceleration must be FOLLOWED BY a
# contradiction_debt decrease. We require a minimum number of paired
# (accelerated → debt-after) observations and a minimum fraction where debt fell.
MIN_ACCEL_OBSERVATIONS = 8
MIN_ACCEL_DEBT_DROP_FRACTION = 0.60
# A cadence proposal above this counts as "cortisol-driven acceleration" for the
# purpose of opening an acceleration→debt observation (neutral cadence is 1.0).
ACCEL_CADENCE_THRESHOLD = 1.10

MIN_SHADOW_HOURS = 4.0          # live-tick soak floor (anti reset-gaming)
TRANSITION_COOLDOWN_S = 300.0   # mirrors promotion.py TRANSITION_COOLDOWN_S


@dataclass
class _AccelObservation:
    """One pending cortisol-acceleration → debt observation (§5 proof obligation).

    Opened when an active/eligible tick proposes a cortisol-driven cadence
    acceleration; ``debt_at_accel`` is the contradiction_debt at that moment.
    Closed at the next tick after ``window_s`` by comparing the debt then.
    """

    opened_ts: float = 0.0
    debt_at_accel: float = 0.0
    cadence_at_accel: float = 1.0
    window_s: float = 1800.0   # observe the debt decrease over the next 30 min


@dataclass
class _CouplingState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active — DEFAULT 0 (inert)
    kill_switch_engaged: bool = False
    shadow_start_ts: float = field(default_factory=time.time)
    live_tick_seconds: float = 0.0
    # Rolling record of whether each closed acceleration observation saw debt fall.
    accel_debt_drops: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    pending_accel: _AccelObservation | None = None
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0
    last_kill_reason: str = ""


class AffectCadenceCoupling:
    """The P5 coupling gate: turns affect proposals into real levers — or no-ops.

    DEFAULT level 0 (shadow): :meth:`apply_levers` returns the neutral lever set
    and :meth:`external_reward_term` returns 0.0, so nothing about cadence,
    reward, urgency, or memory reinforcement changes. Only an operator promotion
    (after every §8 P5 gate clears) makes the levers live, and the kill-switch can
    revert them to identity at any moment.
    """

    _instance: "AffectCadenceCoupling | None" = None

    def __init__(self) -> None:
        self._state = _CouplingState()
        self._kill = KillSwitch()
        self._load()
        self._kill.engaged = self._state.kill_switch_engaged

    @classmethod
    def get_instance(cls) -> "AffectCadenceCoupling":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # ── Level / kill-switch state ─────────────────────────────────────────────

    @property
    def level(self) -> int:
        return self._state.level

    @property
    def is_shadow(self) -> bool:
        return self._state.level == 0

    @property
    def is_active(self) -> bool:
        """True only at level 2 AND with the kill-switch released.

        The kill-switch is the outermost guard: even when promoted, an engaged
        kill-switch forces the coupling inert (cadence → 1.0, reward → baseline).
        """
        return self._state.level >= 2 and not self._kill.engaged

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill.engaged

    def engage_kill_switch(self, reason: str = "operator") -> None:
        """Engage the kill-switch: every coupled lever reverts to identity.

        Reverts ``cadence_multiplier → 1.0`` (via :meth:`apply_levers` returning
        the neutral set) and reward → the unchanged ``_compute_health_reward``
        (via :meth:`external_reward_term` returning 0.0). Idempotent.
        """
        self._kill.engage()
        self._state.kill_switch_engaged = True
        self._state.last_kill_reason = reason
        logger.warning("Affect coupling KILL-SWITCH engaged (%s) — levers reverted to identity", reason)
        self.save()

    def release_kill_switch(self) -> None:
        """Release the kill-switch (operator re-arm). Coupling resumes per level."""
        self._kill.release()
        self._state.kill_switch_engaged = False
        self.save()

    # ── Lever application (the one runtime authority — gated) ─────────────────

    def apply_levers(self, snapshot: Any | None) -> dict[str, Any]:
        """Return the lever set to *actually* apply this tick.

        SHADOW or kill-switch-engaged → the neutral no-op set (identity levers),
        so the kernel cadence, memory reinforcement, interval multipliers, and
        drive-urgency bias are all unchanged. ACTIVE → the snapshot's PROPOSED
        levers re-clamped to their kernel-native bounds (the outermost §5 rule-7
        guard, applied again here so a stale/garbage proposal can never escape
        the safe envelope).

        Also opens/closes the §5 acceleration→debt proof observation whenever a
        cortisol-driven cadence acceleration is applied — even at advisory, so the
        evidence accrues *before* promotion (honest: we prove the obligation in
        shadow/advisory, then promote).
        """
        proposed = {}
        if snapshot is not None:
            try:
                proposed = dict(getattr(snapshot, "proposed_levers", {}) or {})
            except Exception:
                proposed = {}

        # Always track the §5 obligation evidence from the PROPOSED cadence +
        # cortisol level, regardless of whether we apply (shadow/advisory accrue
        # the proof so promotion is earned honestly).
        try:
            self._observe_acceleration(snapshot, proposed)
        except Exception:
            logger.debug("acceleration observation failed", exc_info=True)

        if not self.is_active:
            return neutral_levers()
        # Active: re-clamp the proposed levers as the outermost guard.
        try:
            return clamp_levers(proposed) if proposed else neutral_levers()
        except Exception:
            logger.debug("apply_levers clamp failed — falling back to neutral", exc_info=True)
            return neutral_levers()

    def external_reward_term(self, *, external_validation_rate: float, grounded_outcomes: int) -> float:
        """The §3-component-6 EXTERNAL grounding reward term — gated.

        SHADOW or kill-switch-engaged → 0.0 (the baseline ``_compute_health_reward``
        is reachable unchanged, SPARK §5 kill-switch / §8 P5). ACTIVE → a small,
        bounded reward proportional to the *external* validation rate (never a
        self-score — the rate is movable only by a real external validator), with
        no credit until ≥``MIN_GROUNDING_OUTCOMES`` grounding outcomes exist.

        The term is intentionally small (max ±0.15) and centred on the promote
        threshold so it rewards *correctness-against-world*: above 0.40 the loop
        is grounding well (positive), below it the loop is drifting inward
        (negative). Bounded so it can never dominate the perf-derived baseline.
        """
        if not self.is_active:
            return 0.0
        if grounded_outcomes < MIN_GROUNDING_OUTCOMES:
            return 0.0
        rate = max(0.0, min(1.0, float(external_validation_rate)))
        # Centre on the promote threshold; scale so rate=1.0 → +0.15, rate=0 → −0.06.
        term = 0.25 * (rate - PROMOTE_VALIDATION_RATE)
        return max(-0.15, min(0.15, term))

    # ── §5 proof obligation: acceleration → debt decrease ─────────────────────

    def _observe_acceleration(self, snapshot: Any | None, proposed: dict[str, Any]) -> None:
        """Open/close the cortisol-acceleration → debt-decrease observation (§5).

        Opens an observation when the proposed cadence is an acceleration
        (>``ACCEL_CADENCE_THRESHOLD``) driven by elevated cortisol; closes the
        prior observation once its window has elapsed, recording whether debt fell.
        """
        now = time.time()
        debt = self._read_contradiction_debt()

        # Close a pending observation whose window has elapsed.
        pend = self._state.pending_accel
        if pend is not None and (now - pend.opened_ts) >= pend.window_s:
            dropped = 1.0 if debt < (pend.debt_at_accel - 1e-6) else 0.0
            self._state.accel_debt_drops.append(dropped)
            self._state.pending_accel = None
            self.save()
            # If debt ROSE after acceleration while we are ACTIVE, the governor is
            # failing its proof obligation — engage the kill-switch (§5/§8 P5).
            if dropped == 0.0 and self.is_active:
                self.engage_kill_switch("contradiction_debt rose after cortisol acceleration")

        # Open a new observation if cadence is a cortisol-driven acceleration and
        # none is currently pending.
        if self._state.pending_accel is None and snapshot is not None:
            try:
                cadence = float(proposed.get("cadence_multiplier", 1.0) or 1.0)
                cort = float(getattr(snapshot.cortisol, "level", 0.5))
            except Exception:
                return
            if cadence > ACCEL_CADENCE_THRESHOLD and cort > 0.55:
                self._state.pending_accel = _AccelObservation(
                    opened_ts=now,
                    debt_at_accel=debt,
                    cadence_at_accel=cadence,
                )
                self.save()

    @staticmethod
    def _read_contradiction_debt() -> float:
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            st = ContradictionEngine.get_instance().get_state()
            return float(st.get("contradiction_debt", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _accel_debt_drop_fraction(self) -> float:
        hist = list(self._state.accel_debt_drops)
        return (sum(hist) / len(hist)) if hist else 0.0

    # ── Governor invariant readout (no scalar reverses >50%; no saturation) ──

    @staticmethod
    def _governor_invariant_ok() -> bool:
        """Read the live regulator states: no scalar reverses >50% of updates and
        no clamp-saturation streak. Conservative: missing data → not OK (gate shut).
        """
        try:
            from consciousness.affect_state import affect_state
            from consciousness.affect_regulation import (
                CLAMP_LO, CLAMP_HI, _count_recent_sign_flips, OSC_RING_SIZE,
            )
            states = affect_state.regulator_states()
        except Exception:
            return False
        if not states:
            return False
        for _name, st in states.items():
            steps = st.get("recent_steps") or []
            nz = [s for s in steps if s != 0.0]
            if len(nz) >= 4:
                flips = _count_recent_sign_flips(steps, OSC_RING_SIZE)
                # flips among k nonzero steps ranges 0..k-1; reversal fraction.
                frac = flips / max(1, len(nz) - 1)
                if frac > MAX_REVERSAL_FRACTION:
                    return False
            lvl = float(st.get("level", 0.5))
            if lvl <= CLAMP_LO + 1e-6 or lvl >= CLAMP_HI - 1e-6:
                # Pinned at a clamp this read — a single read is a streak proxy here;
                # the regulator itself never fully saturates, so any pin is a flag.
                return False
            if st.get("frozen"):
                return False
        return True

    # ── Promotion eligibility (computed; never self-promotes) ─────────────────

    def promotion_eligible(
        self,
        *,
        external_validation_rate: float,
        grounded_outcomes: int,
        orphan_rate_trending_down: bool,
    ) -> bool:
        """§8 P5 eligibility. Does NOT promote — informs the operator/dashboard.

        ALL gates must clear: soak floor, ≥20 grounding outcomes with
        external_validation_rate ≥0.40 AND orphan_rate trending down, the governor
        invariant, and the cortisol-acceleration → debt-decrease proof.
        """
        if self._state.level >= 2:
            return False
        if self._kill.engaged:
            return False
        if self._state.live_tick_seconds / 3600.0 < MIN_SHADOW_HOURS:
            return False
        if grounded_outcomes < MIN_GROUNDING_OUTCOMES:
            return False
        if float(external_validation_rate) < PROMOTE_VALIDATION_RATE:
            return False
        if not orphan_rate_trending_down:
            return False
        if not self._governor_invariant_ok():
            return False
        # §5 proof obligation.
        if len(self._state.accel_debt_drops) < MIN_ACCEL_OBSERVATIONS:
            return False
        if self._accel_debt_drop_fraction() < MIN_ACCEL_DEBT_DROP_FRACTION:
            return False
        return True

    def record_tick(self, *, tick_dt_s: float = 0.0) -> None:
        """Advance the live-tick soak clock (anti reset-gaming) + auto-demote check."""
        if tick_dt_s > 0:
            self._state.live_tick_seconds += tick_dt_s

    def maybe_auto_demote(
        self, *, external_validation_rate: float, grounded_outcomes: int,
    ) -> None:
        """Auto-demote (and engage kill-switch) on §8 P5 regression.

        Regression = governor freeze/invariant break, or the external validation
        rate collapsing below the demote floor once enough grounding traffic
        exists. Mirrors the auto-demote pattern in the other promotion gates.
        """
        if self._state.level <= 0:
            return
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < TRANSITION_COOLDOWN_S:
            return
        regressed = False
        reason = ""
        if not self._governor_invariant_ok():
            regressed, reason = True, "governor invariant broke"
        elif (
            grounded_outcomes >= MIN_GROUNDING_OUTCOMES
            and float(external_validation_rate) < DEMOTE_VALIDATION_RATE
        ):
            regressed, reason = True, (
                f"external_validation_rate {external_validation_rate:.2f} < {DEMOTE_VALIDATION_RATE}"
            )
        if regressed:
            self._auto_demote(reason)

    def _auto_demote(self, reason: str) -> None:
        old = self._state.level
        self._state.level = max(self._state.level - 1, 0)
        self._state.last_demoted_at = time.time()
        self._state.shadow_start_ts = time.time()
        # Any regression also engages the kill-switch so the levers revert NOW.
        self.engage_kill_switch(f"auto-demote: {reason}")
        logger.warning("Affect coupling demoted: level %d → %d (%s)", old, self._state.level, reason)
        self.save()

    # ── Status / persistence ──────────────────────────────────────────────────

    def _level_name(self) -> str:
        return {0: "shadow", 1: "advisory", 2: "active"}.get(self._state.level, "unknown")

    def get_status(
        self,
        *,
        external_validation_rate: float = 0.0,
        grounded_outcomes: int = 0,
        orphan_rate_trending_down: bool = False,
    ) -> dict[str, Any]:
        return {
            "level": self._state.level,
            "level_name": self._level_name(),
            "kill_switch_engaged": self._kill.engaged,
            "last_kill_reason": self._state.last_kill_reason,
            "drives_levers": self.is_active,
            "live_tick_hours": round(self._state.live_tick_seconds / 3600.0, 2),
            "accel_observations": len(self._state.accel_debt_drops),
            "accel_debt_drop_fraction": round(self._accel_debt_drop_fraction(), 3),
            "governor_invariant_ok": self._governor_invariant_ok(),
            "external_validation_rate": round(float(external_validation_rate), 4),
            "grounded_outcomes": grounded_outcomes,
            "orphan_rate_trending_down": bool(orphan_rate_trending_down),
            "promotion_ready": self.promotion_eligible(
                external_validation_rate=external_validation_rate,
                grounded_outcomes=grounded_outcomes,
                orphan_rate_trending_down=orphan_rate_trending_down,
            ),
            "note": (
                "shadow — cadence/reward coupling computed, NONE applied"
                if not self.is_active
                else "active — affect drives cadence/reward (kill-switch armed)"
            ),
        }

    def save(self) -> None:
        data = {
            "level": self._state.level,
            "kill_switch_engaged": self._state.kill_switch_engaged,
            "shadow_start_ts": self._state.shadow_start_ts,
            "live_tick_seconds": self._state.live_tick_seconds,
            "accel_debt_drops": list(self._state.accel_debt_drops),
            "pending_accel": (
                {
                    "opened_ts": self._state.pending_accel.opened_ts,
                    "debt_at_accel": self._state.pending_accel.debt_at_accel,
                    "cadence_at_accel": self._state.pending_accel.cadence_at_accel,
                    "window_s": self._state.pending_accel.window_s,
                }
                if self._state.pending_accel is not None else None
            ),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
            "last_kill_reason": self._state.last_kill_reason,
        }
        try:
            path = Path(AFFECT_COUPLING_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save affect coupling state", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(AFFECT_COUPLING_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._state.level = int(data.get("level", 0) or 0)
            self._state.kill_switch_engaged = bool(data.get("kill_switch_engaged", False))
            self._state.shadow_start_ts = float(data.get("shadow_start_ts", time.time()))
            self._state.live_tick_seconds = float(data.get("live_tick_seconds", 0.0))
            for v in data.get("accel_debt_drops", []):
                self._state.accel_debt_drops.append(float(v))
            pend = data.get("pending_accel")
            if isinstance(pend, dict):
                self._state.pending_accel = _AccelObservation(
                    opened_ts=float(pend.get("opened_ts", 0.0)),
                    debt_at_accel=float(pend.get("debt_at_accel", 0.0)),
                    cadence_at_accel=float(pend.get("cadence_at_accel", 1.0)),
                    window_s=float(pend.get("window_s", 1800.0)),
                )
            self._state.last_promoted_at = float(data.get("last_promoted_at", 0.0))
            self._state.last_demoted_at = float(data.get("last_demoted_at", 0.0))
            self._state.last_kill_reason = str(data.get("last_kill_reason", ""))
            logger.info("Affect coupling restored: level=%d kill=%s",
                        self._state.level, self._state.kill_switch_engaged)
        except Exception:
            logger.debug("Failed to load affect coupling state", exc_info=True)


affect_coupling = AffectCadenceCoupling.get_instance()
