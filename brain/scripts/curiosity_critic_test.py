#!/usr/bin/env python3
"""Run the OFFLINE curiosity critic-test (STEP 3 gate) over the recorder's log.

Usage (on the brain, in the venv):
    python scripts/curiosity_critic_test.py                 # default file + horizon sweep
    python scripts/curiosity_critic_test.py --horizon 5     # single horizon
    python scripts/curiosity_critic_test.py --file /path/to/curiosity_critic_log.json

Reads the bounded evidence log written by cognition/curiosity_critic's live
recorder and asks whether the curiosity_target predicts above-baseline future
skill improvement, surviving a shuffle negative control. Prints a PASS / FAIL /
INSUFFICIENT_DATA verdict. A FAIL means the bridge stays shadow-forever — the
honest outcome, not something to tune around.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BRAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BRAIN))

from cognition.curiosity_critic import run_critic_test, STATE_FILE  # noqa: E402


def _load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"no evidence file at {path} — the recorder has not persisted yet")
        return []
    with open(path) as f:
        d = json.load(f)
    return d.get("samples", []) if isinstance(d, dict) else list(d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline curiosity critic-test (STEP 3 gate)")
    ap.add_argument("--file", default=str(STATE_FILE), help="evidence log path")
    ap.add_argument("--horizon", type=int, default=0, help="single horizon in samples (0 = sweep 3/5/10)")
    ap.add_argument("--min-tuples", type=int, default=100)
    ap.add_argument("--shuffles", type=int, default=2000)
    args = ap.parse_args()

    samples = _load(Path(args.file))
    print(f"loaded {len(samples)} samples from {args.file}")
    if not samples:
        return 2
    with_t = sum(1 for s in samples if int(s.get("target", -1)) >= 0)
    print(f"samples with a curiosity_target: {with_t}")

    horizons = [args.horizon] if args.horizon else [3, 5, 10]
    results = {}
    for h in horizons:
        r = run_critic_test(samples, horizon=h, min_tuples=args.min_tuples, n_shuffles=args.shuffles)
        results[h] = r
        print(f"\n── horizon {h} ──")
        print(json.dumps(r, indent=2))

    passes = [h for h, r in results.items() if r.get("verdict") == "PASS"]
    print("\n=== SUMMARY ===")
    for h, r in results.items():
        print(f"  horizon {h:>2}: {r.get('verdict')}"
              + (f" (p={r.get('permutation_p_value')}, effect={r.get('effect_vs_random')})"
                 if r.get('verdict') in ('PASS', 'FAIL') else ""))
    if any(r.get("verdict") == "INSUFFICIENT_DATA" for r in results.values()):
        print("  → soak longer, then re-run (honest under-sampling).")
    elif passes:
        print(f"  → PASS at horizon(s) {passes}. Earns STEP 4 advisory-nudge eligibility "
              "(still kill-switched + negative-controlled). Do NOT auto-wire; confirm with David.")
    else:
        print("  → FAIL at all horizons. The curiosity_target is not a leading indicator of "
              "learnability — the bridge stays SHADOW-FOREVER (honest). Report and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
