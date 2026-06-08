"""Homeostatic governor for the shadow affect layer (SPARK_DESIGN §5).

Affect→cadence is the only true feedback loop in the system, so it is
intrinsically a runaway risk.  This module is the *required* damping layer:
a per-scalar homeostatic governor that bleeds off old arousal, bounds the
absorption of new input, clamps to a never-saturating band, enforces a
refractory window after large swings, and freezes/auto-demotes on excessive
oscillation.

Everything here is **pure and deterministic**: ``regulate()`` takes the prior
state + a raw reading + elapsed wall-seconds and returns a *new* state.  No
I/O, no singletons, no clocks consulted internally (``dt`` is passed in), so
the governor is trivially unit-testable and reproducible.

Constants are the §5 first-guess values (documented as needing empirical
tuning under live load):

  - baseline 0.5, mean-reversion half-life 3600s
  - MAX_STEP ±0.15 (anti-spike), GAIN 0.5 EMA (anti-whiplash)
  - clamp band [0.05, 0.95] (never fully saturates — saturation = dead lever)
  - refractory: a single step ≥0.25 opens a 180s window during which MAX_STEP
    halves and GAIN halves
  - anti-oscillation: 6-sample ring; sign flips on 3 of last 4 updates halves
    GAIN for 3 ticks; excessive setpoint-crossing freezes the lever and signals
    auto-demotion of the affect layer to shadow
  - cortisol debt term is capped (/0.6) upstream; here the governor only adds a
    guaranteed downward pull via mean-reversion (the debt itself decays upstream)

The kill-switch hook (``KillSwitch``) reverts every governed scalar to the
neutral baseline so the levers they would drive collapse to no-ops
(cadence_multiplier → 1.0, etc.).  Lever-native clamps are applied as the
outermost guard by :func:`clamp_levers`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── §5 governor constants (first-guess; need empirical tuning) ───────────────
BASELINE: float = 0.5
MEAN_REVERSION_HALFLIFE_S: float = 3600.0
MAX_STEP: float = 0.15
GAIN: float = 0.5
CLAMP_LO: float = 0.05
CLAMP_HI: float = 0.95

REFRACTORY_SWING: float = 0.25      # a single step ≥ this opens the window
REFRACTORY_WINDOW_S: float = 180.0  # during which MAX_STEP & GAIN halve

OSC_RING_SIZE: int = 6              # 6-sample ring
OSC_FLIP_LOOKBACK: int = 4         # examine last 4 updates
OSC_FLIP_TRIGGER: int = 3          # ≥3 sign flips of 4 → critically-damp
OSC_DAMP_TICKS: int = 3            # halve GAIN for 3 ticks
OSC_FREEZE_FLIPS: int = 5          # excessive crossing within the ring → freeze+demote

# Lever-native clamps (§5 rule 7) — the outermost safety envelope.
CADENCE_LO, CADENCE_HI = 0.5, 2.0          # kernel.py:289
REINFORCEMENT_LO, REINFORCEMENT_HI = 0.5, 2.0
INTERVAL_LO, INTERVAL_HI = 0.6, 2.0
URGENCY_LO, URGENCY_HI = 0.0, 1.0          # additive-then-clamped


@dataclass
class ScalarRegulatorState:
    """Per-scalar homeostatic state.  JSON-round-trippable, backward-compatible.

    All fields default to safe neutral values so a reader tolerating their
    absence (older persisted state) rehydrates to a calm baseline.
    """

    level: float = BASELINE
    last_update_ts: float = 0.0
    # 6-sample ring of recent signed steps (most recent last).
    recent_steps: list[float] = field(default_factory=list)
    refractory_until_ts: float = 0.0
    osc_damp_ticks_left: int = 0
    frozen: bool = False             # excessive oscillation → lever frozen
    demote_signal: bool = False      # set when a freeze fires → caller demotes
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": round(self.level, 6),
            "last_update_ts": self.last_update_ts,
            "recent_steps": [round(s, 6) for s in self.recent_steps[-OSC_RING_SIZE:]],
            "refractory_until_ts": self.refractory_until_ts,
            "osc_damp_ticks_left": self.osc_damp_ticks_left,
            "frozen": self.frozen,
            "demote_signal": self.demote_signal,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScalarRegulatorState":
        if not data:
            return cls()
        st = cls()
        st.level = float(data.get("level", BASELINE))
        st.last_update_ts = float(data.get("last_update_ts", 0.0))
        steps = data.get("recent_steps") or []
        st.recent_steps = [float(s) for s in steps][-OSC_RING_SIZE:]
        st.refractory_until_ts = float(data.get("refractory_until_ts", 0.0))
        st.osc_damp_ticks_left = int(data.get("osc_damp_ticks_left", 0))
        st.frozen = bool(data.get("frozen", False))
        st.demote_signal = bool(data.get("demote_signal", False))
        st.update_count = int(data.get("update_count", 0))
        return st


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _half_life_decay(level: float, dt: float) -> float:
    """§5.1 mean reversion toward BASELINE with a 3600s half-life.

    Applied *before* absorbing input so old arousal bleeds off first.
    Negative/zero dt is a no-op (deterministic, monotonic in time).
    """
    if dt <= 0.0:
        return level
    factor = 0.5 ** (dt / MEAN_REVERSION_HALFLIFE_S)
    return BASELINE + (level - BASELINE) * factor


def _count_recent_sign_flips(steps: list[float], lookback: int) -> int:
    """Count adjacent sign changes among the last ``lookback`` nonzero steps."""
    nz = [s for s in steps if s != 0.0][-(lookback + 1):]
    flips = 0
    for a, b in zip(nz, nz[1:]):
        if (a > 0) != (b > 0):
            flips += 1
    return flips


def regulate(
    state: ScalarRegulatorState,
    raw: float,
    dt: float,
    now_ts: float,
) -> ScalarRegulatorState:
    """Apply one governed update and return a NEW state (pure; no mutation).

    Implements §5 rules 1-5 + 7-clamping for a single scalar.  ``raw`` is the
    upstream reading in [0,1]; ``dt`` is wall-seconds since the last update;
    ``now_ts`` is the wall-clock used only to open/expire the refractory window
    (passed in, never read from a clock here).

    The returned state's ``frozen``/``demote_signal`` flags tell the caller the
    lever must be frozen and the affect layer auto-demoted to shadow (§5.5).
    """
    out = ScalarRegulatorState(
        level=state.level,
        last_update_ts=now_ts,
        recent_steps=list(state.recent_steps),
        refractory_until_ts=state.refractory_until_ts,
        osc_damp_ticks_left=state.osc_damp_ticks_left,
        frozen=state.frozen,
        demote_signal=False,  # one-shot; re-evaluated below
        update_count=state.update_count + 1,
    )

    # A frozen lever stays frozen (manual reset via reset_freeze()); still bleed
    # off arousal so the recorded level decays honestly toward baseline.
    if out.frozen:
        out.level = _half_life_decay(out.level, dt)
        out.level = _clamp(out.level, CLAMP_LO, CLAMP_HI)
        return out

    # 1. Mean reversion (bleed off old arousal before absorbing input).
    level = _half_life_decay(out.level, dt)

    # Refractory + anti-oscillation modulate the effective step/gain bounds.
    in_refractory = now_ts < out.refractory_until_ts
    eff_max_step = MAX_STEP * (0.5 if in_refractory else 1.0)
    eff_gain = GAIN * (0.5 if in_refractory else 1.0)
    if out.osc_damp_ticks_left > 0:
        eff_gain *= 0.5
        out.osc_damp_ticks_left -= 1

    # 2. Bounded absorption: clamp the raw delta, then EMA toward it.
    raw_clamped = _clamp(raw, 0.0, 1.0)
    raw_delta = raw_clamped - level  # the *incoming* swing, pre-clamp
    step = _clamp(raw_delta, -eff_max_step, eff_max_step)
    applied = eff_gain * step
    level = level + applied

    # 3. Ceilings/floors (never fully saturate).
    level = _clamp(level, CLAMP_LO, CLAMP_HI)
    out.level = level

    # 4. Refractory: a *single* incoming swing ≥ REFRACTORY_SWING opens the
    #    window.  Tested on the unclamped raw delta — the absorbed step is
    #    capped at MAX_STEP (0.15) and so could never alone reach 0.25; a "big
    #    swing" is a large *incoming* demand, not a large applied change.
    if abs(raw_delta) >= REFRACTORY_SWING:
        out.refractory_until_ts = now_ts + REFRACTORY_WINDOW_S

    # 5. Anti-oscillation: record the applied step, then inspect the ring.
    out.recent_steps.append(applied)
    if len(out.recent_steps) > OSC_RING_SIZE:
        out.recent_steps = out.recent_steps[-OSC_RING_SIZE:]

    recent_flips = _count_recent_sign_flips(out.recent_steps, OSC_FLIP_LOOKBACK)
    if recent_flips >= OSC_FLIP_TRIGGER:
        out.osc_damp_ticks_left = max(out.osc_damp_ticks_left, OSC_DAMP_TICKS)

    ring_flips = _count_recent_sign_flips(out.recent_steps, OSC_RING_SIZE)
    if ring_flips >= OSC_FREEZE_FLIPS:
        out.frozen = True
        out.demote_signal = True

    return out


def reset_freeze(state: ScalarRegulatorState) -> ScalarRegulatorState:
    """Return a copy with the freeze cleared and the ring emptied (manual re-arm)."""
    out = ScalarRegulatorState.from_dict(state.to_dict())
    out.frozen = False
    out.demote_signal = False
    out.recent_steps = []
    out.osc_damp_ticks_left = 0
    return out


# ── Lever clamps (§5 rule 7) — outermost guard ───────────────────────────────

def clamp_levers(levers: dict[str, Any]) -> dict[str, Any]:
    """Clamp every proposed lever into its kernel-native safe envelope.

    Worst case the whole module is a no-op inside the pre-existing safe band.
    Unknown keys pass through untouched (additive-only / backward compatible).
    """
    out = dict(levers)
    if "cadence_multiplier" in out and out["cadence_multiplier"] is not None:
        out["cadence_multiplier"] = _clamp(
            float(out["cadence_multiplier"]), CADENCE_LO, CADENCE_HI
        )
    if "memory_reinforcement" in out and out["memory_reinforcement"] is not None:
        out["memory_reinforcement"] = _clamp(
            float(out["memory_reinforcement"]), REINFORCEMENT_LO, REINFORCEMENT_HI
        )
    # Interval *multipliers* — a dict of {interval_name: multiplier}.
    iv = out.get("interval_multipliers")
    if isinstance(iv, dict):
        out["interval_multipliers"] = {
            k: _clamp(float(v), INTERVAL_LO, INTERVAL_HI) for k, v in iv.items()
        }
    # Drive-urgency *bias* — additive deltas, clamped to a half-band so a single
    # bias can never alone push past the [0,1] urgency envelope.
    ub = out.get("urgency_bias")
    if isinstance(ub, dict):
        out["urgency_bias"] = {
            k: _clamp(float(v), -URGENCY_HI, URGENCY_HI) for k, v in ub.items()
        }
    return out


# ── Kill-switch (§5 + §10.2) ─────────────────────────────────────────────────

@dataclass
class KillSwitch:
    """Reverts cadence → 1.0 and reward → the unchanged ``_compute_health_reward``.

    A pure flag holder: when ``engaged`` is True, :func:`neutral_levers` is the
    only thing any consumer is permitted to use, and the governor's scalars are
    forced to baseline.  No I/O — the engine wires this to a real toggle.
    """

    engaged: bool = False

    def engage(self) -> None:
        self.engaged = True

    def release(self) -> None:
        self.engaged = False


def neutral_levers() -> dict[str, Any]:
    """The kill-switch / shadow no-op lever set: every lever at its identity."""
    return {
        "cadence_multiplier": 1.0,
        "memory_reinforcement": 1.0,
        "interval_multipliers": {},
        "urgency_bias": {},
    }
