"""P3.11 — Tier-2 feasibility probe regression tests for dream_synthesis.

Exercises ``brain/scripts/dream_synthesis_feasibility.py`` against
synthesised label ledgers covering each verdict branch.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.dream_synthesis_feasibility import (
    LABEL_CLASSES,
    MIN_SAMPLES_FOR_FEASIBILITY,
    analyse,
)


def _write_labels(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write ``len(rows)`` lines, each with validation_state (from tuple)
    and reason category."""
    idx = {c: i for i, c in enumerate(LABEL_CLASSES)}
    with path.open("w", encoding="utf-8") as f:
        for state, reason in rows:
            vec = [0.0] * len(LABEL_CLASSES)
            vec[idx[state]] = 1.0
            f.write(json.dumps({
                "data": vec,
                "metadata": {
                    "validation_state": state,
                    "reason_category": reason,
                },
            }) + "\n")


def test_missing_file_reports_missing_and_insufficient():
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "does_not_exist.jsonl"
        result = analyse(missing)
        assert result["status"] == "missing"
        assert result["verdict"] == "INSUFFICIENT"


def test_insufficient_when_total_below_threshold():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.jsonl"
        _write_labels(path, [
            ("promoted", "no_sources"),
            ("held", "informational_hold"),
            ("discarded", "no_sources"),
            ("quarantined", "contradicts_beliefs"),
        ])
        result = analyse(path)
        assert result["verdict"] == "INSUFFICIENT"
        assert result["total_samples"] == 4
        # All four classes present — the block is sample count, not coverage.
        assert not result["missing_classes"]


def test_insufficient_when_class_missing_even_above_threshold():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.jsonl"
        rows = [("promoted", "no_sources")] * (MIN_SAMPLES_FOR_FEASIBILITY + 50)
        _write_labels(path, rows)
        result = analyse(path)
        assert result["verdict"] == "INSUFFICIENT"
        assert "held" in result["missing_classes"]
        assert "discarded" in result["missing_classes"]
        assert "quarantined" in result["missing_classes"]


def test_imbalanced_when_majority_over_ceiling():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.jsonl"
        # 90% promoted, 10% split among the other three classes.
        rows = [("promoted", "no_sources")] * 360
        rows += [("held", "informational_hold")] * 14
        rows += [("discarded", "no_sources")] * 13
        rows += [("quarantined", "contradicts_beliefs")] * 13
        _write_labels(path, rows)
        result = analyse(path)
        assert result["verdict"] == "IMBALANCED"
        assert result["total_samples"] == 400
        assert result["majority_class_share"] > 0.80


def test_feasible_with_balanced_distribution_at_threshold():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.jsonl"
        # 250 samples, roughly balanced across the four classes.
        rows = []
        for _ in range(70):
            rows.append(("promoted", "no_sources"))
        for _ in range(60):
            rows.append(("held", "informational_hold"))
        for _ in range(60):
            rows.append(("discarded", "no_sources"))
        for _ in range(60):
            rows.append(("quarantined", "contradicts_beliefs"))
        _write_labels(path, rows)
        result = analyse(path)
        assert result["verdict"] == "FEASIBLE"
        assert result["total_samples"] == 250
        assert result["majority_class_share"] <= 0.80
        # Reason distribution populated.
        assert sum(result["reason_distribution"].values()) == 250


def test_script_main_returns_exit_2_for_non_feasible(tmp_path, capsys):
    import scripts.dream_synthesis_feasibility as mod
    rc = mod.main(["--path", str(tmp_path / "missing.jsonl"), "--json"])
    assert rc == 2
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["verdict"] == "INSUFFICIENT"
