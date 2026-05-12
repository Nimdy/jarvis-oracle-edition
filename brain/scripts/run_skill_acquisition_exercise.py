#!/usr/bin/env python3
"""Run the synthetic skill-acquisition exercise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.skill_acquisition_exercise import PROFILES
from synthetic.skill_acquisition_exercise import run_skill_acquisition_exercise
from synthetic.skill_acquisition_exercise import write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic skill acquisition exercise")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-record", action="store_true", help="Do not record distillation signals")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    if args.no_record:
        from dataclasses import replace
        profile = replace(profile, record_signals=False)

    stats = run_skill_acquisition_exercise(profile=profile, count=args.count, seed=args.seed)
    path = write_report(stats)
    print(f"skill_acquisition_exercise profile={profile.name} episodes={stats.episodes} passed={stats.passed}")
    print(f"report={path}")
    if stats.invariant_failures:
        print("invariant_failures=" + ",".join(stats.invariant_failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

