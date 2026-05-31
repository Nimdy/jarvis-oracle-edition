"""Shadow affect readout (SPARK_DESIGN §3 component 3, §4, §7).

Three governed scalars — **dopamine / serotonin / cortisol** — computed
*strictly* as deterministic functions of counted real signals.  Affect here is
a **readout, never a feeling** (§7): each scalar carries a provenance dict
mapping its nickname to the exact ``(source_field, raw_value)`` it was derived
from, and a **cannot-lie clamp** that forces the scalar to exactly ``0.0`` when
all of its source readings are ``0`` (no backing → no signal).

Source signals (§4, by exact field name):

  - **dopamine** (reward-prediction / novelty / resolved-grounding):
      * ``DriveSignals.novelty_events``
      * ``DeltaTracker`` latest ``net_attribution`` (credited only when
        ``> MIN_MEANINGFUL_DELTA``)
      * WorldModel causal ``predictive_accuracy``
  - **serotonin** (coherence-satisfaction):
      * ``1 − contradiction_debt``
      * ``CuriosityQuestionBuffer.get_overall_satisfaction()``
      * ``1 − overconfidence_error``
  - **cortisol** (unresolved-tension; forced to 0.0 if all three read 0):
      * ``contradiction_debt``
      * ``QuarantinePressure.composite``
      * friction rate (recent gate blocks)

This module **computes and exposes** proposed lever values (cadence
multiplier, interval deltas, drive-urgency bias, memory reinforcement) per the
§4 map, but **DOES NOT apply them** — application is gated behind the affect
promotion controller (P4/P5) and is never wired here.

The governor (:mod:`consciousness.affect_regulation`) is run per-scalar so the
*regulated* level (the homeostatically-damped value) is what drives the
proposed levers; the raw reading and full provenance are retained for the
confabulation ledger and the dashboard.

Singleton: ``affect_state``.  Reads of every upstream signal are wrapped in
defensive try/except returning the neutral default, so a missing/uninitialised
subsystem degrades to baseline rather than raising on the consciousness tick.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from consciousness.affect_regulation import (
    BASELINE,
    ScalarRegulatorState,
    clamp_levers,
    neutral_levers,
    regulate,
)

logger = logging.getLogger(__name__)

# §4 map — neutral 0.5, deviation from 0.5 moves a lever.
NEUTRAL = BASELINE  # 0.5

# Dopamine novelty saturates at 10 events (mirrors drives.py:194).
_NOVELTY_SATURATION = 10
# net_attribution credit threshold (autonomy.constants.MIN_MEANINGFUL_DELTA).
_MIN_MEANINGFUL_DELTA = 0.02
# Friction rate saturation: N recent gate blocks → full friction term.
_FRICTION_SATURATION = 5
# Cortisol debt term cap (§5.6: capped /0.6 so a spike can't drive unbounded).
_DEBT_TERM_CAP = 0.6


@dataclass
class AffectReadout:
    """One scalar's computed state: raw value, regulated level, provenance.

    ``provenance`` maps the contributing signal names to ``[source_field,
    raw_value]`` pairs (the §7 honesty contract).  ``cannot_lie_clamped`` is
    True when the cannot-lie clamp forced the raw to 0.0 (all sources read 0).
    """

    nickname: str
    raw: float = NEUTRAL
    level: float = NEUTRAL
    provenance: dict[str, Any] = field(default_factory=dict)
    cannot_lie_clamped: bool = False
    all_sources_zero: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "raw": round(self.raw, 4),
            "level": round(self.level, 4),
            "provenance": self.provenance,
            "cannot_lie_clamped": self.cannot_lie_clamped,
            "all_sources_zero": self.all_sources_zero,
        }


@dataclass
class AffectSnapshot:
    """Full affect tick result: three readouts + the proposed (unapplied) levers."""

    timestamp: float
    dopamine: AffectReadout
    serotonin: AffectReadout
    cortisol: AffectReadout
    proposed_levers: dict[str, Any] = field(default_factory=dict)
    demote_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dopamine": self.dopamine.to_dict(),
            "serotonin": self.serotonin.to_dict(),
            "cortisol": self.cortisol.to_dict(),
            "proposed_levers": self.proposed_levers,
            "demote_signal": self.demote_signal,
        }


# ── Defensive signal readers ─────────────────────────────────────────────────
# Each returns (raw_value, source_field_label).  Any failure → neutral default.

def _read_novelty_events(signals: Any | None) -> tuple[float, str]:
    try:
        if signals is None:
            return 0.0, "DriveSignals.novelty_events"
        n = int(getattr(signals, "novelty_events", 0) or 0)
        return float(max(0, n)), "DriveSignals.novelty_events"
    except Exception:
        return 0.0, "DriveSignals.novelty_events"


def _read_net_attribution(delta_tracker: Any | None) -> tuple[float, str]:
    """Latest completed delta's net_attribution (credited only if meaningful)."""
    try:
        if delta_tracker is None:
            return 0.0, "DeltaTracker.net_attribution"
        recent = delta_tracker.get_recent_deltas(limit=1)
        if not recent:
            return 0.0, "DeltaTracker.net_attribution"
        na = float(recent[-1].get("net_attribution", 0.0) or 0.0)
        return na, "DeltaTracker.net_attribution"
    except Exception:
        return 0.0, "DeltaTracker.net_attribution"


def _read_world_model_accuracy(world_model: Any | None) -> tuple[float, str]:
    """WorldModel honest foresight signal: causal predictive_accuracy."""
    try:
        if world_model is None:
            return 0.0, "WorldModel.causal.predictive_accuracy"
        state = world_model.get_state()
        causal = (state or {}).get("causal", {}) or {}
        acc = float(causal.get("predictive_accuracy", 0.0) or 0.0)
        return acc, "WorldModel.causal.predictive_accuracy"
    except Exception:
        return 0.0, "WorldModel.causal.predictive_accuracy"


def _read_contradiction_debt() -> tuple[float, str]:
    try:
        from epistemic.contradiction_engine import ContradictionEngine
        debt = float(
            ContradictionEngine.get_instance().get_state().get("contradiction_debt", 0.0)
            or 0.0
        )
        return max(0.0, min(1.0, debt)), "contradiction_debt"
    except Exception:
        return 0.0, "contradiction_debt"


def _read_quarantine_pressure() -> tuple[float, str]:
    try:
        from epistemic.quarantine.pressure import get_quarantine_pressure
        comp = float(get_quarantine_pressure().current.composite or 0.0)
        return max(0.0, min(1.0, comp)), "QuarantinePressure.composite"
    except Exception:
        return 0.0, "QuarantinePressure.composite"


def _read_curiosity_satisfaction() -> tuple[float, str]:
    """Overall curiosity-buffer satisfaction in [-1,1]; mapped to [0,1]."""
    try:
        from personality.curiosity_questions import CuriosityQuestionBuffer
        sat = float(CuriosityQuestionBuffer.get_instance().get_overall_satisfaction())
        # Map [-1,1] → [0,1]; neutral 0 → 0.5.
        return max(0.0, min(1.0, (sat + 1.0) / 2.0)), "CuriosityQuestionBuffer.satisfaction"
    except Exception:
        return 0.5, "CuriosityQuestionBuffer.satisfaction"


def _read_overconfidence_error() -> tuple[float | None, str]:
    """Calibration overconfidence error in [0,1]; None when insufficient data."""
    try:
        from epistemic.calibration import TruthCalibrationEngine
        tce = TruthCalibrationEngine.get_instance()
        cc = getattr(tce, "_confidence_calibrator", None)
        if cc is None:
            return None, "ConfidenceCalibrator.overconfidence_error"
        oc = cc.get_overconfidence_error()
        if oc is None:
            return None, "ConfidenceCalibrator.overconfidence_error"
        return max(0.0, min(1.0, float(oc))), "ConfidenceCalibrator.overconfidence_error"
    except Exception:
        return None, "ConfidenceCalibrator.overconfidence_error"


def _read_friction_rate(signals: Any | None) -> tuple[float, str]:
    """Recent gate-block count, normalised to [0,1] at _FRICTION_SATURATION."""
    try:
        if signals is None:
            return 0.0, "DriveSignals.gate_blocks_recent"
        blocks = int(getattr(signals, "gate_blocks_recent", 0) or 0)
        return min(1.0, max(0, blocks) / float(_FRICTION_SATURATION)), \
            "DriveSignals.gate_blocks_recent"
    except Exception:
        return 0.0, "DriveSignals.gate_blocks_recent"


class AffectState:
    """Singleton computing the three labelled affect scalars from real signals.

    Computes + regulates + proposes levers; **never applies** them.  Persistence
    of regulator state is in-memory only here (the promotion controller owns the
    durable ledger writes); ``snapshot()`` returns the last computed
    :class:`AffectSnapshot`.
    """

    _instance: "AffectState | None" = None

    @classmethod
    def get_instance(cls) -> "AffectState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._reg = {
            "dopamine": ScalarRegulatorState(),
            "serotonin": ScalarRegulatorState(),
            "cortisol": ScalarRegulatorState(),
        }
        self._last: AffectSnapshot | None = None
        self._tick_count: int = 0

    # ── Raw scalar computation (deterministic, cannot-lie clamped) ───────────

    @staticmethod
    def _compute_dopamine(
        signals: Any | None, delta_tracker: Any | None, world_model: Any | None,
    ) -> AffectReadout:
        nov_raw, nov_field = _read_novelty_events(signals)
        na_raw, na_field = _read_net_attribution(delta_tracker)
        wm_raw, wm_field = _read_world_model_accuracy(world_model)

        nov_term = min(nov_raw, _NOVELTY_SATURATION) / float(_NOVELTY_SATURATION)
        # Grounding term: credit only a genuinely positive, meaningful delta
        # (§7 — no credit without a real positive net_attribution).
        na_term = max(0.0, na_raw) if na_raw > _MIN_MEANINGFUL_DELTA else 0.0
        na_term = min(1.0, na_term)
        wm_term = max(0.0, min(1.0, wm_raw))

        sources = [nov_term, na_term, wm_term]
        all_zero = all(s == 0.0 for s in sources)
        # Mean of the three terms, recentred so neutral baseline = 0.5.
        mean_term = sum(sources) / 3.0
        raw = NEUTRAL + (mean_term - NEUTRAL) if not all_zero else 0.0

        ro = AffectReadout(nickname="dopamine")
        ro.raw = max(0.0, min(1.0, raw))
        ro.all_sources_zero = all_zero
        # Cannot-lie clamp: dopamine's grounding term gets no credit without a
        # real positive net_attribution or world-model hit; if ALL three read 0
        # the whole scalar is forced to exactly 0.0.
        if all_zero:
            ro.raw = 0.0
            ro.cannot_lie_clamped = True
        ro.provenance = {
            "novelty": [nov_field, round(nov_raw, 4)],
            "net_attribution": [na_field, round(na_raw, 4)],
            "world_model_accuracy": [wm_field, round(wm_raw, 4)],
        }
        return ro

    @staticmethod
    def _compute_serotonin() -> AffectReadout:
        debt_raw, debt_field = _read_contradiction_debt()
        sat_raw, sat_field = _read_curiosity_satisfaction()
        oc_raw, oc_field = _read_overconfidence_error()

        coherence_term = max(0.0, min(1.0, 1.0 - debt_raw))      # 1 − contradiction_debt
        satisfaction_term = max(0.0, min(1.0, sat_raw))          # buffer satisfaction
        # 1 − overconfidence_error; when calibration has no data (None), this
        # term is omitted rather than guessed.
        calib_term = (1.0 - oc_raw) if oc_raw is not None else None

        terms = [coherence_term, satisfaction_term]
        if calib_term is not None:
            terms.append(max(0.0, min(1.0, calib_term)))

        # Serotonin's sources are "high when calm" — they are never structurally
        # all-zero (coherence is 1 with zero debt), so the cannot-lie clamp here
        # fires only on the genuinely-degenerate all-zero reading.
        all_zero = all(t == 0.0 for t in terms)
        mean_term = sum(terms) / len(terms) if terms else NEUTRAL
        raw = mean_term

        ro = AffectReadout(nickname="serotonin")
        ro.raw = max(0.0, min(1.0, raw))
        ro.all_sources_zero = all_zero
        if all_zero:
            ro.raw = 0.0
            ro.cannot_lie_clamped = True
        ro.provenance = {
            "coherence": [debt_field, round(debt_raw, 4)],
            "curiosity_satisfaction": [sat_field, round(sat_raw, 4)],
            "overconfidence_error": [
                oc_field, round(oc_raw, 4) if oc_raw is not None else None
            ],
        }
        return ro

    @staticmethod
    def _compute_cortisol(signals: Any | None) -> AffectReadout:
        debt_raw, debt_field = _read_contradiction_debt()
        pressure_raw, pressure_field = _read_quarantine_pressure()
        friction_raw, friction_field = _read_friction_rate(signals)

        # §5.6 — cap the debt term so a spike can't drive unbounded acceleration.
        debt_term = min(1.0, debt_raw / _DEBT_TERM_CAP)
        pressure_term = max(0.0, min(1.0, pressure_raw))
        friction_term = max(0.0, min(1.0, friction_raw))

        sources = [debt_term, pressure_term, friction_term]
        # CANNOT LIE (§4/§7): forced to exactly 0.0 if all three readings are 0.
        all_zero = all(s == 0.0 for s in sources)
        mean_term = sum(sources) / 3.0
        # Cortisol is an arousal term: 0 sources → neutral-floor maps to 0.0
        # under the cannot-lie clamp; otherwise lift from baseline by the mean.
        raw = NEUTRAL + mean_term * (1.0 - NEUTRAL) if not all_zero else 0.0

        ro = AffectReadout(nickname="cortisol")
        ro.raw = max(0.0, min(1.0, raw))
        ro.all_sources_zero = all_zero
        if all_zero:
            ro.raw = 0.0
            ro.cannot_lie_clamped = True
        ro.provenance = {
            "contradiction_debt": [debt_field, round(debt_raw, 4)],
            "quarantine_pressure": [pressure_field, round(pressure_raw, 4)],
            "friction_rate": [friction_field, round(friction_raw, 4)],
        }
        return ro

    # ── Proposed levers (§4 map) — computed, NEVER applied here ───────────────

    @staticmethod
    def _propose_levers(
        dopamine: float, serotonin: float, cortisol: float,
    ) -> dict[str, Any]:
        """Map regulated scalars to proposed lever values per §4.

        deviation = scalar − 0.5.  Cadence multipliers are additive around 1.0;
        urgency biases are additive deltas; interval multipliers shorten (<1)
        when arousal is high.  Everything is clamped to the kernel-native band
        by :func:`clamp_levers`.  These are PROPOSALS — the affect layer is
        shadow by default and applies none of them.
        """
        d_dev = dopamine - NEUTRAL
        s_dev = serotonin - NEUTRAL
        c_dev = cortisol - NEUTRAL

        # Cadence: dopamine +0.3·dev (speeds), serotonin −0.4·dev (slows),
        # cortisol +0.6·dev (the strongest "need more input" term).
        cadence = 1.0 + 0.3 * d_dev - 0.4 * s_dev + 0.6 * c_dev

        # Interval multipliers (>1 lengthens / rest, <1 shortens / urgency).
        # Dopamine & cortisol shorten epistemic-repair intervals; serotonin
        # lengthens them (rest) and especially lengthens curiosity (anti-nagging).
        epistemic_iv = 1.0 - 0.3 * d_dev - 0.5 * c_dev + 0.4 * s_dev
        epistemic_iv = max(0.7, epistemic_iv)  # §4 floor so it can't thrash.
        curiosity_iv = 1.0 - 0.3 * d_dev + 0.8 * s_dev

        interval_multipliers = {
            "meta_thought": max(0.7, 1.0 - 0.3 * d_dev + 0.3 * s_dev),
            "curiosity": curiosity_iv,
            "contradiction": epistemic_iv,
            "truth_calibration": epistemic_iv,
            "belief_graph": epistemic_iv,
        }

        # Drive-urgency bias (additive — never conjures a drive from zero).
        urgency_bias = {
            "curiosity": 0.20 * d_dev,                # dopamine → explore
            "truth": 0.25 * c_dev,                    # cortisol → audit
            "coherence": 0.25 * c_dev,
            "grounding": 0.25 * c_dev,
        }

        # Memory reinforcement: dopamine reinforces novel/rewarded memories;
        # serotonin deliberately ABSENT (coherence must not inflate weights, §4);
        # cortisol preserves tension/error memories (slow decay → >1).
        memory_reinforcement = 1.0 + 0.3 * d_dev + 0.3 * c_dev

        return clamp_levers({
            "cadence_multiplier": cadence,
            "interval_multipliers": interval_multipliers,
            "urgency_bias": urgency_bias,
            "memory_reinforcement": memory_reinforcement,
        })

    # ── Public tick ──────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        signals: Any | None = None,
        delta_tracker: Any | None = None,
        world_model: Any | None = None,
        now_ts: float | None = None,
    ) -> AffectSnapshot:
        """Compute one affect readout and regulate it.  Pure of side effects.

        Returns an :class:`AffectSnapshot` with raw + regulated scalars, full
        provenance, and the *proposed* (unapplied) levers.  Reads every upstream
        signal defensively; missing subsystems degrade to baseline.
        """
        now = now_ts if now_ts is not None else time.time()
        self._tick_count += 1

        dopamine = self._compute_dopamine(signals, delta_tracker, world_model)
        serotonin = self._compute_serotonin()
        cortisol = self._compute_cortisol(signals)

        demote = False
        for ro in (dopamine, serotonin, cortisol):
            st = self._reg[ro.nickname]
            dt = (now - st.last_update_ts) if st.last_update_ts > 0 else 0.0
            new_st = regulate(st, ro.raw, dt, now)
            self._reg[ro.nickname] = new_st
            ro.level = new_st.level
            if new_st.demote_signal:
                demote = True

        levers = self._propose_levers(dopamine.level, serotonin.level, cortisol.level)

        snap = AffectSnapshot(
            timestamp=now,
            dopamine=dopamine,
            serotonin=serotonin,
            cortisol=cortisol,
            proposed_levers=levers,
            demote_signal=demote,
        )
        self._last = snap
        return snap

    def snapshot(self) -> AffectSnapshot | None:
        """Last computed snapshot (or None before the first tick)."""
        return self._last

    def get_status(self) -> dict[str, Any]:
        """Dashboard-friendly status.  Shadow layer drives no real lever."""
        if self._last is None:
            return {
                "computed": False,
                "tick_count": self._tick_count,
                "proposed_levers": neutral_levers(),
                "note": "affect layer is shadow — proposals computed, none applied",
            }
        d = self._last.to_dict()
        d["computed"] = True
        d["tick_count"] = self._tick_count
        d["note"] = "affect layer is shadow — proposals computed, none applied"
        return d

    def regulator_states(self) -> dict[str, dict[str, Any]]:
        """Serialise the per-scalar governor states (for durable persistence)."""
        return {k: v.to_dict() for k, v in self._reg.items()}


affect_state = AffectState.get_instance()
