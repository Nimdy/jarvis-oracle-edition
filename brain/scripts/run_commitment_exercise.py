#!/usr/bin/env python3
"""Synthetic commitment exercise runner.

Text-only harness for the Intention Infrastructure Stage 0 commitment
pipeline (``CommitmentExtractor`` + ``CapabilityGate.evaluate_commitment``).
No brain server required, no audio, no LLM, no network. Pure regression
harness: run it in CI, on a fresh checkout, or as a long soak.

Usage:

    python -m scripts.run_commitment_exercise --profile smoke
    python -m scripts.run_commitment_exercise --profile coverage
    python -m scripts.run_commitment_exercise --profile strict
    python -m scripts.run_commitment_exercise --profile stress --seed 7
    python -m scripts.run_commitment_exercise --count 1000 --seed 42

Reports are written as JSON to
``~/.jarvis/synthetic_exercise/commitment_reports/<profile>_<timestamp>.json``
and summaries print to stdout.

Exit codes:
    0 — profile PASS (all invariants and accuracy gates hold)
    1 — profile FAIL (some gate or invariant did not hold)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [commitment] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("commitment_exercise")

DEFAULT_REPORT_DIR = os.path.expanduser(
    "~/.jarvis/synthetic_exercise/commitment_reports"
)


def main(argv: list[str] | None = None) -> int:
    # Make the brain package importable whether we're running as
    # `python -m scripts.run_commitment_exercise` from inside brain/ or
    # directly as `python brain/scripts/run_commitment_exercise.py`.
    here = Path(__file__).resolve()
    brain_root = here.parent.parent
    if str(brain_root) not in sys.path:
        sys.path.insert(0, str(brain_root))

    from synthetic.commitment_exercise import (
        COMMITMENT_PROFILES,
        run_commitment_exercise,
    )

    parser = argparse.ArgumentParser(
        description="Synthetic commitment exercise (Stage 0 regression harness)"
    )
    parser.add_argument(
        "--profile",
        choices=sorted(COMMITMENT_PROFILES.keys()),
        default="smoke",
        help="Named exercise profile.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Override profile count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic runs.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="For long soak runs (ignored when profile delay is 0).",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Output directory for JSON report.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the JSON report file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary (report still written).",
    )
    args = parser.parse_args(argv)

    profile = COMMITMENT_PROFILES[args.profile]
    logger.info(
        "Starting commitment exercise: profile=%s count=%s seed=%s",
        profile.name, args.count or profile.count, args.seed,
    )

    stats = run_commitment_exercise(
        profile=profile,
        count=args.count,
        seed=args.seed,
        duration_s=args.duration,
    )

    if not args.quiet:
        print(stats.summary())

    if not args.no_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"{profile.name}_{timestamp}.json"
        payload = {
            "profile": profile.name,
            "stats": stats.to_dict(),
            "timestamp": timestamp,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        logger.info("Report written to %s", report_path)

    return 0 if stats.pass_result else 1


if __name__ == "__main__":
    sys.exit(main())
