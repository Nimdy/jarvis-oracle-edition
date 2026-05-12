#!/usr/bin/env python3
"""Unified synthetic training exercise runner.

Runs any combination of the new synthetic exercises (retrieval, world model,
contradiction, diagnostic, plan evaluator) without requiring a live brain
instance (except retrieval which needs memory + vector store).

Usage:
    python -m scripts.run_synthetic_exercises --exercise all --profile smoke
    python -m scripts.run_synthetic_exercises --exercise diagnostic --profile coverage
    python -m scripts.run_synthetic_exercises --exercise retrieval --profile stress
    python -m scripts.run_synthetic_exercises --exercise world_model,contradiction

All exercises respect the epistemic truth boundary: standalone instances,
capped fidelity (0.7), no event emission, no real persistence pollution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [synth] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synthetic_exercises")

REPORT_DIR = Path(os.path.expanduser("~/.jarvis/synthetic_exercise/reports"))

AVAILABLE_EXERCISES = [
    "retrieval",
    "world_model",
    "contradiction",
    "diagnostic",
    "plan_evaluator",
]


def _run_retrieval(profile_name: str, count: int | None) -> dict[str, Any]:
    from synthetic.retrieval_exercise import PROFILES, run_retrieval_exercise
    profile = PROFILES.get(profile_name, PROFILES["coverage"])
    stats = run_retrieval_exercise(profile=profile, count=count)
    print(stats.summary())
    return stats.to_dict()


def _run_world_model(profile_name: str, count: int | None) -> dict[str, Any]:
    from synthetic.world_model_exercise import PROFILES, run_world_model_exercise
    profile = PROFILES.get(profile_name, PROFILES["coverage"])
    stats = run_world_model_exercise(profile=profile, count=count)
    print(stats.summary())
    return stats.to_dict()


def _run_contradiction(profile_name: str, _count: int | None) -> dict[str, Any]:
    from synthetic.contradiction_exercise import PROFILES, run_contradiction_exercise
    profile = PROFILES.get(profile_name, PROFILES["coverage"])
    stats = run_contradiction_exercise(profile=profile)
    print(stats.summary())
    return stats.to_dict()


def _run_diagnostic(
    profile_name: str, count: int | None, record: bool,
) -> dict[str, Any]:
    from synthetic.diagnostic_exercise import PROFILES, run_diagnostic_exercise

    profile = PROFILES.get(profile_name, PROFILES["coverage"])
    if not record:
        profile.record_signals = False

    collector = None
    if profile.record_signals:
        try:
            from hemisphere.distillation import DistillationCollector
            collector = DistillationCollector()
        except Exception as exc:
            logger.warning("DistillationCollector unavailable: %s", exc)

    stats = run_diagnostic_exercise(profile=profile, count=count, collector=collector)
    print(stats.summary())
    return stats.to_dict()


def _run_plan_evaluator(
    profile_name: str, count: int | None, record: bool,
) -> dict[str, Any]:
    from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

    profile = PROFILES.get(profile_name, PROFILES["coverage"])
    if not record:
        profile.record_signals = False

    collector = None
    if profile.record_signals:
        try:
            from hemisphere.distillation import DistillationCollector
            collector = DistillationCollector()
        except Exception as exc:
            logger.warning("DistillationCollector unavailable: %s", exc)

    stats = run_plan_evaluator_exercise(profile=profile, count=count, collector=collector)
    print(stats.summary())
    return stats.to_dict()


def _save_report(results: dict[str, Any], exercises_run: list[str]) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = "_".join(exercises_run)
    filename = f"exercise_{name}_{ts}.json"
    path = REPORT_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic training exercises",
    )
    parser.add_argument(
        "--exercise", "-e",
        default="all",
        help="Comma-separated exercises or 'all' "
             f"(available: {', '.join(AVAILABLE_EXERCISES)})",
    )
    parser.add_argument(
        "--profile", "-p",
        default="coverage",
        choices=["smoke", "coverage", "stress"],
        help="Exercise profile (default: coverage)",
    )
    parser.add_argument(
        "--count", "-n",
        type=int, default=None,
        help="Override scenario/query count",
    )
    parser.add_argument(
        "--record", "-r",
        action="store_true",
        help="Record distillation signals (diagnostic + plan_evaluator only)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing JSON report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.exercise == "all":
        exercises = list(AVAILABLE_EXERCISES)
    else:
        exercises = [e.strip() for e in args.exercise.split(",")]
        for e in exercises:
            if e not in AVAILABLE_EXERCISES:
                print(f"Unknown exercise: {e}")
                print(f"Available: {', '.join(AVAILABLE_EXERCISES)}")
                sys.exit(1)

    print(f"Running exercises: {', '.join(exercises)} (profile={args.profile})")
    print("=" * 60)

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "exercises": {},
    }
    start = time.time()

    for ex_name in exercises:
        print(f"\n--- {ex_name} ---")
        try:
            if ex_name == "retrieval":
                results["exercises"][ex_name] = _run_retrieval(
                    args.profile, args.count)
            elif ex_name == "world_model":
                results["exercises"][ex_name] = _run_world_model(
                    args.profile, args.count)
            elif ex_name == "contradiction":
                results["exercises"][ex_name] = _run_contradiction(
                    args.profile, args.count)
            elif ex_name == "diagnostic":
                results["exercises"][ex_name] = _run_diagnostic(
                    args.profile, args.count, args.record)
            elif ex_name == "plan_evaluator":
                results["exercises"][ex_name] = _run_plan_evaluator(
                    args.profile, args.count, args.record)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results["exercises"][ex_name] = {"error": str(exc)}

    elapsed = time.time() - start
    results["total_elapsed_s"] = round(elapsed, 2)

    print("\n" + "=" * 60)
    print(f"Total elapsed: {elapsed:.1f}s")

    pass_count = sum(
        1 for r in results["exercises"].values()
        if isinstance(r, dict) and r.get("pass", False)
    )
    fail_count = sum(
        1 for r in results["exercises"].values()
        if isinstance(r, dict) and not r.get("pass", True)
    )
    error_count = sum(
        1 for r in results["exercises"].values()
        if isinstance(r, dict) and "error" in r
    )

    print(f"Results: {pass_count} pass, {fail_count} fail, {error_count} error")

    if not args.no_report:
        path = _save_report(results, exercises)
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
