"""Threshold ranges for dashboard coloring (green/yellow/red).

Phase A baselines derived from known healthy system ranges.
These will be refined as data accumulates.
"""

from __future__ import annotations

from typing import Any

# (metric_key, green_max, yellow_max)
# green: 0..green_max, yellow: green_max..yellow_max, red: > yellow_max
# For inverted metrics (higher=worse), ranges are flipped in the adapter.

BASELINES: dict[str, dict[str, Any]] = {
    "contradiction_debt": {
        "green_max": 0.05,
        "yellow_max": 0.15,
        "inverted": True,
        "label": "Contradiction Debt",
        "unit": "",
    },
    "soul_integrity_index": {
        "green_max": 0.70,
        "yellow_max": 0.50,
        "inverted": False,
        "label": "Soul Integrity",
        "unit": "",
    },
    "quarantine_composite": {
        "green_max": 0.15,
        "yellow_max": 0.40,
        "inverted": True,
        "label": "Quarantine Pressure",
        "unit": "",
    },
    "audit_score": {
        "green_max": 0.70,
        "yellow_max": 0.45,
        "inverted": False,
        "label": "Audit Score",
        "unit": "",
    },
    "memory_avg_weight": {
        "green_max": 0.65,
        "yellow_max": 0.40,
        "inverted": False,
        "label": "Avg Memory Weight",
        "unit": "",
    },
    "dream_promotion_rate": {
        "green_max": 0.30,
        "yellow_max": 0.60,
        "inverted": True,
        "label": "Dream Promotion Rate",
        "unit": "%",
    },
    "library_substantive_ratio": {
        "green_max": 0.60,
        "yellow_max": 0.30,
        "inverted": False,
        "label": "Substantive Content Ratio",
        "unit": "%",
    },
    "language_native_usage_rate": {
        "green_max": 0.70,
        "yellow_max": 0.35,
        "inverted": False,
        "label": "Native Usage Rate",
        "unit": "%",
    },
    "language_fail_closed_rate": {
        "green_max": 0.25,
        "yellow_max": 0.50,
        "inverted": True,
        "label": "Fail-Closed Rate",
        "unit": "%",
    },
    # Phase D gate scores
    "language_provenance_fidelity": {
        "green_max": 0.90,
        "yellow_max": 0.70,
        "inverted": False,
        "label": "Provenance Fidelity",
        "unit": "%",
    },
    "language_exactness": {
        "green_max": 0.85,
        "yellow_max": 0.60,
        "inverted": False,
        "label": "Exactness",
        "unit": "%",
    },
    "language_hallucination_rate": {
        "green_max": 0.05,
        "yellow_max": 0.10,
        "inverted": True,
        "label": "Hallucination Rate",
        "unit": "%",
    },
    "language_style_quality": {
        "green_max": 0.90,
        "yellow_max": 0.70,
        "inverted": False,
        "label": "Style Quality",
        "unit": "%",
    },
}


def classify(metric_key: str, value: float | None) -> str:
    """Return 'green', 'yellow', or 'red' for a metric value."""
    if value is None:
        return "grey"
    baseline = BASELINES.get(metric_key)
    if not baseline:
        return "grey"

    if baseline.get("inverted"):
        if value <= baseline["green_max"]:
            return "green"
        elif value <= baseline["yellow_max"]:
            return "yellow"
        return "red"
    else:
        if value >= baseline["green_max"]:
            return "green"
        elif value >= baseline["yellow_max"]:
            return "yellow"
        return "red"
