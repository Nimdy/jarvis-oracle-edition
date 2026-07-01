"""Curiosity critic-test — STEP 3 of the self-sensing -> curiosity bridge. THE GATE.

STEP 1 proved the self-sensing signal is CARRIED into the drive signals; STEP 2
proved the would-attend proposer FIRES. Neither proves the target is CAUSALLY
USEFUL — that the ``curiosity_target`` sector actually points where future
learnable structure is, better than a trivial choice. That is the ONLY thing
that earns the link a real (advisory) lever in STEP 4. This module is two halves,
following the discipline that "saved weeks twice" (validate the causal signal
OFFLINE, with a negative control, before wiring it to anything):

  1. ``CuriosityCriticRecorder`` — LIVE, shadow, bounded. Every drive tick it
     appends a per-sector snapshot {skill[], lp[], act[], target, regime,
     dynamic_fraction, ts} to a fixed-size deque persisted to ONE overwritten
     file (restart-surviving). Pure evidence-gathering: authority=none, drives
     nothing, writes NO belief/memory, never grows without bound (deque maxlen +
     single file) so it cannot pollute anything. It exposes only sample COUNTS
     (never a verdict/score) so there is nothing to tune green.

  2. ``run_critic_test`` — OFFLINE, on-demand. Builds (t, t+horizon) tuples and
     asks: does the sector that was the curiosity_target at t realise MORE skill
     improvement by t+horizon than a randomly chosen *active* sector would
     (permutation/shuffle null), and does it also beat the all-sector mean? It
     returns a PASS / FAIL / INSUFFICIENT_DATA verdict with effect size and a
     permutation p-value. It is NOT wired to any live gauge. A FAIL means the
     bridge stays shadow-forever — reported honestly, never gamed.

Honest scope: the passive lidar predictor cannot "attend harder" to change its
data, so this tests the falsifiable PROXY for usefulness — is the target a
leading indicator of continued learnability (positive future skill change),
rather than noise or a spike about to revert to the mean? That is exactly the
precondition for the target being worth attending to.
"""
from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
STATE_FILE = JARVIS_DIR / "curiosity_critic_log.json"

N_SECTORS = 12
RECORD_INTERVAL_S = 55.0     # ~ the 60s drive cadence; throttle so extra callers can't over-sample
MAX_SAMPLES = 5000           # bounded: ~3+ days at 1/min, one overwritten file (memory/disk-safe)
SAVE_EVERY = 20              # persist every N new samples


def _f(v: Any) -> float | None:
    """Coerce a possibly-null per-sector value to float or None (skill can be null)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class CuriosityCriticRecorder:
    """LIVE shadow evidence recorder for the offline critic-test. Authority=none."""

    def __init__(self) -> None:
        self._samples: deque[dict[str, Any]] = deque(maxlen=MAX_SAMPLES)
        self._last_record_ts = 0.0
        self._n_recorded = 0
        self._restored = False

    def record_from_status(self, status: dict[str, Any], now: float) -> None:
        """Append ONE throttled per-sector snapshot from self_sensing.get_status().

        Pure read of ``status``; fully exception-safe (one-way tap — must never
        perturb the drive tick that feeds it)."""
        try:
            if not self._restored:
                self.restore()
            if now - self._last_record_ts < RECORD_INTERVAL_S:
                return
            ps = status.get("per_sector") or []
            if len(ps) < N_SECTORS:
                return
            skill = [_f(ps[j].get("skill")) for j in range(N_SECTORS)]
            lp = [_f(ps[j].get("lp")) for j in range(N_SECTORS)]
            act = [_f(ps[j].get("activity")) for j in range(N_SECTORS)]
            tgt = status.get("curiosity_target")
            tsec = -1
            if isinstance(tgt, dict) and tgt.get("sector") is not None:
                tsec = int(tgt["sector"])
            rec = {
                "ts": round(float(now), 1),
                "regime": str((status.get("health") or {}).get("regime", "unknown")),
                "dyn": round(float(status.get("dynamic_fraction") or 0.0), 4),
                "target": tsec,
                "skill": [None if v is None else round(v, 4) for v in skill],
                "lp": [None if v is None else round(v, 4) for v in lp],
                "act": [None if v is None else round(v, 5) for v in act],
            }
            self._samples.append(rec)
            self._n_recorded += 1
            self._last_record_ts = now
            if self._n_recorded % SAVE_EVERY == 0:
                self.save()
        except Exception:
            logger.debug("curiosity critic recorder failed", exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        if not self._restored:
            try:
                self.restore()
            except Exception:
                pass
        n = len(self._samples)
        with_target = sum(1 for s in self._samples if int(s.get("target", -1)) >= 0)
        span_h = 0.0
        if n >= 2:
            span_h = (float(self._samples[-1]["ts"]) - float(self._samples[0]["ts"])) / 3600.0
        return {
            "phase": "P0_critic_recorder_shadow",
            "authority": "zero_authority",
            "drives_levers": False,
            "samples": n,
            "samples_with_target": with_target,
            "span_hours": round(span_h, 2),
            "recorded_session": self._n_recorded,
            "max_samples": MAX_SAMPLES,
            "record_interval_s": RECORD_INTERVAL_S,
            "ready_hint": ("run scripts/curiosity_critic_test.py once samples_with_target is >= ~150 "
                           "across a real spread of desk activity; FAIL = shadow-forever (honest)"),
            "note": ("evidence gathering only — NO verdict/score here (nothing to tune green); "
                     "the critic-test is run OFFLINE, on demand, to produce a PASS/FAIL."),
        }

    def get_samples(self) -> list[dict[str, Any]]:
        if not self._restored:
            try:
                self.restore()
            except Exception:
                pass
        return list(self._samples)

    # -- persistence (bounded single file, tmp+replace) ---------------------
    def save(self) -> None:
        try:
            JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(JARVIS_DIR), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump({"samples": list(self._samples), "n_recorded": self._n_recorded}, f)
            os.replace(tmp, STATE_FILE)
        except Exception:
            logger.debug("curiosity critic save failed", exc_info=True)

    def restore(self) -> None:
        self._restored = True
        try:
            if not STATE_FILE.exists():
                return
            with open(STATE_FILE) as f:
                d = json.load(f)
            for s in d.get("samples", []):
                self._samples.append(s)
            self._n_recorded = int(d.get("n_recorded", len(self._samples)))
            if self._samples:
                self._last_record_ts = float(self._samples[-1].get("ts", 0.0))
        except Exception:
            logger.debug("curiosity critic restore failed", exc_info=True)


_recorder: CuriosityCriticRecorder | None = None


def get_curiosity_critic() -> CuriosityCriticRecorder:
    global _recorder
    if _recorder is None:
        _recorder = CuriosityCriticRecorder()
    return _recorder


# ─────────────────────────── OFFLINE critic-test ───────────────────────────

def _active_sectors(sample: dict[str, Any]) -> list[int]:
    """Sectors with meaningful recent movement at t — mirrors self_sensing's
    ``active = activity > 0.1 * max(activity)`` eligibility for a target."""
    act = sample.get("act") or []
    vals = [(j, act[j]) for j in range(len(act)) if act[j] is not None]
    if not vals:
        return []
    mx = max(v for _, v in vals)
    if mx <= 0:
        return []
    return [j for j, v in vals if v > 0.1 * mx]


def _delta(a: dict[str, Any], b: dict[str, Any], j: int) -> float | None:
    """Realised skill change in sector j from sample a to sample b (None if unmeasurable)."""
    sa = (a.get("skill") or [None] * N_SECTORS)
    sb = (b.get("skill") or [None] * N_SECTORS)
    if j >= len(sa) or j >= len(sb):
        return None
    if sa[j] is None or sb[j] is None:
        return None
    return float(sb[j]) - float(sa[j])


def run_critic_test(
    samples: list[dict[str, Any]],
    horizon: int = 5,
    min_tuples: int = 100,
    n_shuffles: int = 2000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Offline gate: does the curiosity_target predict above-baseline future skill
    improvement, surviving a shuffle negative control?

    For each sample t that had a target j* and a valid t+horizon partner, compute
    the realised skill change in j* and compare, across all such tuples, the mean
    target improvement against:
      * a permutation null (a random *active* sector per tuple, n_shuffles draws),
      * the all-sector mean improvement (the naive "everything drifts" baseline),
      * the most-active sector (the persistence-like default one might pick anyway).
    Returns PASS / FAIL / INSUFFICIENT_DATA with effect size + permutation p-value.
    """
    rng = random.Random(seed)
    n = len(samples)

    tuples: list[tuple[int, int]] = []   # (target_sector, index) with valid horizon partner
    target_deltas: list[float] = []
    allmean_deltas: list[float] = []
    mostactive_deltas: list[float] = []
    active_pools: list[list[int]] = []    # eligible active sectors per tuple (for the shuffle null)
    per_tuple_active_deltas: list[dict[int, float]] = []

    for i in range(n - horizon):
        a = samples[i]
        b = samples[i + horizon]
        j_star = int(a.get("target", -1))
        if j_star < 0:
            continue
        d_star = _delta(a, b, j_star)
        if d_star is None:
            continue
        # eligible active sectors with a measurable delta over this horizon
        pool = [j for j in _active_sectors(a) if _delta(a, b, j) is not None]
        if not pool:
            continue
        # all-sector mean (any sector with a measurable delta)
        all_d = [_delta(a, b, j) for j in range(N_SECTORS)]
        all_d = [d for d in all_d if d is not None]
        if not all_d:
            continue
        # most-active sector's delta
        act = a.get("act") or []
        ma = None
        ma_val = -1.0
        for j in range(min(N_SECTORS, len(act))):
            if act[j] is not None and act[j] > ma_val and _delta(a, b, j) is not None:
                ma_val = act[j]; ma = j
        d_ma = _delta(a, b, ma) if ma is not None else None

        tuples.append((j_star, i))
        target_deltas.append(d_star)
        allmean_deltas.append(sum(all_d) / len(all_d))
        if d_ma is not None:
            mostactive_deltas.append(d_ma)
        active_pools.append(pool)
        per_tuple_active_deltas.append({j: _delta(a, b, j) for j in pool})  # type: ignore[misc]

    m = len(tuples)
    if m < min_tuples:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "tuples": m,
            "min_tuples": min_tuples,
            "horizon": horizon,
            "samples_seen": n,
            "note": ("not enough (target, t+horizon) tuples yet — let the recorder soak across more "
                     "real desk activity, then re-run. This is honest under-sampling, not a fail."),
        }

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    obs_target = mean(target_deltas)
    obs_allmean = mean(allmean_deltas)
    obs_mostactive = mean(mostactive_deltas) if mostactive_deltas else None

    # permutation null: replace each target with a random active sector, m-tuple mean, n_shuffles draws
    ge = 0
    null_means: list[float] = []
    for _ in range(n_shuffles):
        acc = 0.0
        for k in range(m):
            pool = active_pools[k]
            j = rng.choice(pool)
            acc += per_tuple_active_deltas[k][j]
        nm = acc / m
        null_means.append(nm)
        if nm >= obs_target:
            ge += 1
    p_value = (ge + 1) / (n_shuffles + 1)
    null_mean = mean(null_means)
    effect_vs_random = obs_target - null_mean

    beats_random = p_value < 0.05
    beats_allmean = obs_target > obs_allmean
    positive = obs_target > 0.0
    verdict = "PASS" if (beats_random and beats_allmean and positive) else "FAIL"

    reasons = []
    if not positive:
        reasons.append("target's mean future skill-change is <= 0 (points at reverting/saturated sectors, not learnable ones)")
    if not beats_random:
        reasons.append("target does not beat a random active sector beyond chance (p=%.3f >= 0.05)" % p_value)
    if not beats_allmean:
        reasons.append("target does not beat the all-sector mean drift")
    if verdict == "PASS":
        reasons.append("target realises more future skill-improvement than random active sectors (p<0.05) AND beats the all-sector mean — a real leading indicator")

    return {
        "verdict": verdict,
        "tuples": m,
        "horizon": horizon,
        "samples_seen": n,
        "target_mean_future_skill_change": round(obs_target, 5),
        "random_active_null_mean": round(null_mean, 5),
        "all_sector_mean": round(obs_allmean, 5),
        "most_active_mean": (round(obs_mostactive, 5) if obs_mostactive is not None else None),
        "effect_vs_random": round(effect_vs_random, 5),
        "permutation_p_value": round(p_value, 4),
        "n_shuffles": n_shuffles,
        "reasons": reasons,
        "gate": ("PASS earns advisory-nudge eligibility (STEP 4, still kill-switched + negative-controlled); "
                 "FAIL/INSUFFICIENT = the bridge stays shadow. No metric here is tuned; this is a one-shot verdict."),
    }
