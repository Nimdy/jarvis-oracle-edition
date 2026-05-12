#!/usr/bin/env python3
"""Operator seed-registration tool for the Phase E Language Kernel.

On a live brain that already has a Phase C student checkpoint on disk at
``~/.jarvis/language_corpus/phase_c/student_checkpoint.json`` but an
empty Language Kernel registry (``status == "pre_mature"`` on
``/api/language-kernel``), this script registers the current checkpoint
as the first artifact. This is the P3.4 seed-registration step.

Usage (desktop / live brain only)::

    # dry run — reports what would be registered
    python brain/scripts/seed_language_kernel.py --dry-run

    # actually seed the registry
    python brain/scripts/seed_language_kernel.py --notes "phase_c_seed_2026-04-24"

Safety:
  * This script is explicit and operator-invoked. Nothing auto-seeds.
  * It is idempotent: if the current on-disk checkpoint already matches
    the live artifact's hash, ``register_current_checkpoint`` returns
    the existing artifact without mutating the registry.
  * It NEVER touches Phase D governor levels; the ``LanguagePromotionGovernor``
    continues to enforce its own ``shadow``/``canary``/``live`` gating
    independently.

Exit codes:
  0 — seed succeeded, or no-op (idempotent re-run), or dry-run.
  1 — registry fetch failed (checkpoint missing, I/O error, etc).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve the brain/ package so the script runs from any cwd.
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

from language.kernel import get_language_kernel_registry  # noqa: E402


def _print(obj: dict) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--notes",
        default="",
        help="Optional provenance note to store with the seed artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the current kernel state without registering anything.",
    )
    args = parser.parse_args(argv)

    try:
        registry = get_language_kernel_registry()
        before = registry.get_state()
    except Exception as exc:  # noqa: BLE001
        print(f"language-kernel registry init failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("# dry-run: current Phase E language-kernel state")
        _print(before)
        if not before.get("checkpoint_exists"):
            print(
                "# NOTE: no Phase C checkpoint on disk — nothing to register.",
                file=sys.stderr,
            )
        elif before.get("status") == "registered" and before.get("matches_live_artifact"):
            print("# NOTE: registry already matches the on-disk checkpoint.")
        return 0

    if not before.get("checkpoint_exists"):
        print(
            "no Phase C checkpoint on disk; run Phase C training first "
            "(expected path: "
            f"{before.get('checkpoint_path')})",
            file=sys.stderr,
        )
        return 1

    artifact = registry.register_current_checkpoint(notes=args.notes)
    if artifact is None:
        print(
            "register_current_checkpoint returned None "
            "(see logs for reason)",
            file=sys.stderr,
        )
        return 1

    after = registry.get_state()
    print("# seed-registration complete")
    _print({
        "artifact_id": artifact.artifact_id,
        "version": artifact.version,
        "hash_prefix": artifact.hash[:12],
        "status_before": before.get("status"),
        "status_after": after.get("status"),
        "total_artifacts": after.get("total_artifacts"),
        "matches_live_artifact": after.get("matches_live_artifact"),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
