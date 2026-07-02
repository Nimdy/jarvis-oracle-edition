#!/usr/bin/env python3
"""CONFIRMATORY out-of-sample test of the PRE-REGISTERED curiosity hypothesis.

Pre-registered 2026-07-02 (see docs/CURIOSITY_CRITIC_PREREG.md), BEFORE the
held-out data existed:

  HYPOTHESIS (single, locked): an ERROR-SEEKING curiosity target — the ACTIVE
  sector with the LOWEST current skill (where the predictor is worst, i.e. most
  room to learn) — predicts positive future skill-change, beating a random
  active sector (shuffle null) AND the all-sector mean.

  This is the OPPOSITE of the original argmax-learning-progress target, which
  FAILED the causal gate (it chased spikes that revert to the mean).

  CONFIRMATORY BAR (locked): on the HELD-OUT window (ts > CUTOFF, disjoint from
  the exploratory 17h window), error-seeking must clear, at BOTH horizon 5 AND
  horizon 10: permutation p < 0.025 AND target_mean_future_skill_change > 0 AND
  target > all_sector_mean, with >= 100 held-out tuples per horizon.

  PASS  -> the signal replicated out-of-sample. First genuinely-earned autonomous
          curiosity candidate. THEN (separately, with David) consider switching
          the shadow proposer to error-seeking + a kill-switched gated advisory —
          eyes open on the predictive!=causal caveat.
  FAIL / INSUFFICIENT -> in-sample artifact. Shadow-forever, clean stop.

Usage on the brain:  python scripts/curiosity_critic_oos.py
(or --file <log> --cutoff <ts> to run locally against a pulled log)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BRAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BRAIN))

from cognition.curiosity_critic import run_critic_test, _active_sectors, STATE_FILE  # noqa: E402

# Locked at pre-registration time (max ts of the exploratory 17h window):
PREREG_CUTOFF = 1782991750.6
CONFIRM_P = 0.025          # single hypothesis, 2 horizons -> Bonferroni 0.05/2
CONFIRM_MIN_TUPLES = 100
CONFIRM_HORIZONS = (5, 10)


def _error_seeking_target(sample: dict) -> int:
    """The pre-registered target: active sector with the LOWEST current skill."""
    sk = sample.get("skill") or []
    cand = [(j, sk[j]) for j in _active_sectors(sample) if j < len(sk) and sk[j] is not None]
    return min(cand, key=lambda x: x[1])[0] if cand else -1


def main() -> int:
    ap = argparse.ArgumentParser(description="Confirmatory OOS curiosity critic-test")
    ap.add_argument("--file", default=str(STATE_FILE))
    ap.add_argument("--cutoff", type=float, default=PREREG_CUTOFF)
    ap.add_argument("--shuffles", type=int, default=3000)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"no evidence file at {path}")
        return 2
    with open(path) as f:
        d = json.load(f)
    alls = d.get("samples", []) if isinstance(d, dict) else list(d)

    held = [s for s in alls if float(s.get("ts", 0)) > args.cutoff]
    print(f"total samples: {len(alls)}  |  held-out (ts > {args.cutoff}): {len(held)}")
    if len(held) < 200:
        print("held-out window too small — let it soak longer, then re-run. (INSUFFICIENT)")
        return 2

    # apply the pre-registered error-seeking target to the held-out window only
    smp = []
    for s in held:
        ns = dict(s); ns["target"] = _error_seeking_target(s)
        smp.append(ns)
    n_t = sum(1 for x in smp if int(x.get("target", -1)) >= 0)
    print(f"held-out targeted samples: {n_t}\n")

    ok = {}
    for h in (3, *CONFIRM_HORIZONS):
        r = run_critic_test(smp, horizon=h, min_tuples=CONFIRM_MIN_TUPLES, n_shuffles=args.shuffles)
        tgt = r.get("target_mean_future_skill_change", 0.0)
        p = r.get("permutation_p_value", 1.0)
        am = r.get("all_sector_mean", 0.0)
        passed = (r.get("verdict") != "INSUFFICIENT_DATA" and p < CONFIRM_P and tgt > 0 and tgt > am
                  and r.get("tuples", 0) >= CONFIRM_MIN_TUPLES)
        if h in CONFIRM_HORIZONS:
            ok[h] = passed
        tag = "[confirmatory]" if h in CONFIRM_HORIZONS else "[info]"
        print(f"h={h:>2} {tag:<14} verdict={r.get('verdict'):<16} target={tgt:+.5f} "
              f"random={r.get('random_active_null_mean',0):+.5f} allmean={am:+.5f} "
              f"p={p:.4f} tuples={r.get('tuples')}  pass={passed}")

    confirmed = all(ok.get(h, False) for h in CONFIRM_HORIZONS)
    print("\n=== CONFIRMATORY VERDICT (pre-registered bar: p<%.3f, positive, beats all-mean, at h=%s) ==="
          % (CONFIRM_P, "&".join(map(str, CONFIRM_HORIZONS))))
    if confirmed:
        print("CONFIRMED — error-seeking replicated OUT-OF-SAMPLE. First earned autonomous-curiosity "
              "candidate. Do NOT auto-wire; bring to David for the shadow-proposer switch + gated advisory.")
    else:
        print("NOT CONFIRMED — the in-sample pass did not replicate on held-out data. "
              "The bridge stays SHADOW-FOREVER. Report honestly and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
