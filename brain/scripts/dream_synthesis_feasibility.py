#!/usr/bin/env python3
"""Tier-2 feasibility probe for the ``dream_synthesis`` specialist (P3.11).

The live brain on 2026-04-24 reports 839 feature signals, 0.32 accuracy,
and 1 active network for ``dream_synthesis``. That is enough to *train*
a Tier-1 shadow — but it is **not** enough, by itself, to justify
promoting the specialist to Tier-2 Matrix Protocol eligibility.

This script walks the live validator-outcome ledger
(``~/.jarvis/hemisphere_training/distill_dream_validator.jsonl``) and
reports:

  * Total labeled samples (N)
  * Class distribution across the four validator states
    (``promoted`` / ``held`` / ``discarded`` / ``quarantined``)
  * Majority-class share (imbalance warning if > 0.80)
  * Per-reason distribution (see ``REASON_CATEGORIES`` in
    ``brain/hemisphere/dream_artifact_encoder.py``)
  * A feasibility verdict:

      - ``INSUFFICIENT``: N < 200 or at least one class has zero samples;
      - ``IMBALANCED``:   majority class > 80% of samples;
      - ``FEASIBLE``:     all four classes populated + majority ≤ 80% +
                         N ≥ 200.

Usage::

    python brain/scripts/dream_synthesis_feasibility.py
    python brain/scripts/dream_synthesis_feasibility.py --path /custom.jsonl
    python brain/scripts/dream_synthesis_feasibility.py --json

This is a **read-only feasibility probe**. It does NOT train anything,
does NOT modify the specialist, and does NOT promote it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

# Resolve brain/ package so the script is runnable from any cwd.
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

from hemisphere.dream_artifact_encoder import LABEL_CLASSES  # noqa: E402


DEFAULT_LABEL_PATH = (
    Path.home() / ".jarvis" / "hemisphere_training" / "distill_dream_validator.jsonl"
)

MIN_SAMPLES_FOR_FEASIBILITY = 200
MAX_MAJORITY_CLASS_SHARE = 0.80


def _label_index_to_name(label_vec: list[float] | None) -> str:
    if not label_vec:
        return "unknown"
    try:
        idx = int(max(range(len(label_vec)), key=label_vec.__getitem__))
    except Exception:
        return "unknown"
    if 0 <= idx < len(LABEL_CLASSES):
        return LABEL_CLASSES[idx]
    return "unknown"


def analyse(label_path: Path) -> dict:
    if not label_path.exists():
        return {
            "status": "missing",
            "label_path": str(label_path),
            "verdict": "INSUFFICIENT",
            "reason": "label ledger not present",
        }

    class_counts: collections.Counter = collections.Counter()
    reason_counts: collections.Counter = collections.Counter()
    total = 0

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1

        label_vec = row.get("data") or row.get("label")
        class_name = _label_index_to_name(label_vec if isinstance(label_vec, list) else None)

        meta = row.get("metadata") or {}
        validation_state = meta.get("validation_state")
        if isinstance(validation_state, str) and validation_state in LABEL_CLASSES:
            class_name = validation_state
        class_counts[class_name] += 1

        reason = meta.get("reason_category", "uncategorized")
        if isinstance(reason, str):
            reason_counts[reason] += 1

    distribution = {cls: class_counts.get(cls, 0) for cls in LABEL_CLASSES}
    if class_counts.get("unknown", 0):
        distribution["unknown"] = class_counts["unknown"]

    missing = [cls for cls in LABEL_CLASSES if distribution.get(cls, 0) == 0]
    majority = max(distribution.get(c, 0) for c in LABEL_CLASSES) if total else 0
    majority_share = (majority / total) if total else 0.0

    if total < MIN_SAMPLES_FOR_FEASIBILITY:
        verdict = "INSUFFICIENT"
        reason = f"only {total} samples, need >= {MIN_SAMPLES_FOR_FEASIBILITY}"
    elif missing:
        verdict = "INSUFFICIENT"
        reason = f"class(es) with zero samples: {missing}"
    elif majority_share > MAX_MAJORITY_CLASS_SHARE:
        verdict = "IMBALANCED"
        reason = (
            f"majority class share {majority_share:.2%} > "
            f"{MAX_MAJORITY_CLASS_SHARE:.0%} ceiling"
        )
    else:
        verdict = "FEASIBLE"
        reason = (
            f"all four classes present; majority share "
            f"{majority_share:.2%} ≤ {MAX_MAJORITY_CLASS_SHARE:.0%}"
        )

    return {
        "status": "ok",
        "label_path": str(label_path),
        "total_samples": total,
        "class_distribution": distribution,
        "class_distribution_pct": {
            cls: (distribution.get(cls, 0) / total) if total else 0.0
            for cls in LABEL_CLASSES
        },
        "majority_class_share": majority_share,
        "missing_classes": missing,
        "reason_distribution": dict(reason_counts),
        "verdict": verdict,
        "reason": reason,
        "threshold": {
            "min_samples": MIN_SAMPLES_FOR_FEASIBILITY,
            "max_majority_class_share": MAX_MAJORITY_CLASS_SHARE,
        },
    }


def _fmt_text(result: dict) -> str:
    if result.get("status") == "missing":
        return (
            f"# dream_synthesis feasibility\n"
            f"label_path: {result['label_path']}\n"
            f"verdict: {result['verdict']} ({result['reason']})\n"
        )
    lines = [
        "# dream_synthesis Tier-2 feasibility probe",
        f"label_path: {result['label_path']}",
        f"total_samples: {result['total_samples']}",
        f"verdict: {result['verdict']}",
        f"reason: {result['reason']}",
        "",
        "class_distribution:",
    ]
    for cls in LABEL_CLASSES:
        cnt = result["class_distribution"].get(cls, 0)
        pct = result["class_distribution_pct"].get(cls, 0.0)
        lines.append(f"  - {cls}: {cnt} ({pct:.2%})")
    if "unknown" in result["class_distribution"]:
        lines.append(f"  - unknown: {result['class_distribution']['unknown']}")
    lines.append(f"majority_class_share: {result['majority_class_share']:.2%}")
    if result["missing_classes"]:
        lines.append(f"missing_classes: {result['missing_classes']}")
    if result["reason_distribution"]:
        lines.append("reason_distribution (top 10):")
        for r, c in sorted(
            result["reason_distribution"].items(), key=lambda kv: -kv[1]
        )[:10]:
            lines.append(f"  - {r}: {c}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path",
        default=str(DEFAULT_LABEL_PATH),
        help="Path to the distill_dream_validator.jsonl ledger.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    result = analyse(Path(args.path))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_fmt_text(result))

    # Exit code 0 for FEASIBLE, 2 for INSUFFICIENT/IMBALANCED/missing so
    # CI can gate on feasibility if desired.
    return 0 if result.get("verdict") == "FEASIBLE" else 2


if __name__ == "__main__":
    sys.exit(main())
