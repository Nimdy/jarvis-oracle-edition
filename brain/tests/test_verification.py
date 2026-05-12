"""Tests for self_improve/verification.py — post-patch verification checks.

Covers:
  - compare_metrics: regression detection, improvement detection, stable state
  - PendingVerification dataclass and _to_dict serialization
  - JSONL round-trip: write_pending → read_pending → clear_pending
  - has_pending / increment_boot_count
  - Edge cases: empty baselines, missing metrics, tolerance boundaries
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import patch, MagicMock

from self_improve.verification import (
    PendingVerification,
    compare_metrics,
    read_pending,
    clear_pending,
    has_pending,
    _to_dict,
    P95_TOLERANCE,
    ERROR_TOLERANCE,
    MEMORY_GROWTH_FACTOR,
    DEFAULT_VERIFICATION_PERIOD_S,
    DEFAULT_MIN_TICKS,
    DEFAULT_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# compare_metrics — regression detection
# ---------------------------------------------------------------------------

class TestCompareMetricsRegression:
    def test_p95_regression_detected(self):
        baselines = {"tick_p95_ms": 10.0}
        current = {"tick_p95_ms": 15.0}  # 50% increase, threshold is 20%
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"
        assert any("tick_p95" in r for r in details["regressions"])

    def test_p95_within_tolerance_is_stable(self):
        baselines = {"tick_p95_ms": 10.0}
        current = {"tick_p95_ms": 11.5}  # 15% increase, within 20% tolerance
        result, details = compare_metrics(baselines, current)
        assert result == "stable"
        assert len(details["regressions"]) == 0

    def test_error_count_regression(self):
        baselines = {"error_count": 5.0}
        current = {"error_count": 8.0}  # delta=3, tolerance is 1
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"
        assert any("error_count" in r for r in details["regressions"])

    def test_error_count_within_tolerance(self):
        baselines = {"error_count": 5.0}
        current = {"error_count": 6.0}  # delta=1, exactly at tolerance
        result, details = compare_metrics(baselines, current)
        assert result == "stable"

    def test_memory_blowup_detected(self):
        baselines = {"memory_count": 100.0}
        current = {"memory_count": 250.0}  # 2.5x, threshold is 2.0x
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"
        assert any("memory" in r for r in details["regressions"])

    def test_memory_within_tolerance(self):
        baselines = {"memory_count": 100.0}
        current = {"memory_count": 180.0}  # 1.8x, within 2.0x
        result, details = compare_metrics(baselines, current)
        assert result != "regressed" or not any("memory" in r for r in details.get("regressions", []))

    def test_event_violations_spike(self):
        baselines = {"event_violations": 0.0}
        current = {"event_violations": 10.0}  # delta=10, threshold is 5
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"

    def test_event_violations_small_change_ok(self):
        baselines = {"event_violations": 2.0}
        current = {"event_violations": 5.0}  # delta=3, within 5
        result, details = compare_metrics(baselines, current)
        assert "regressed" not in [result] or not any("violation" in r for r in details.get("regressions", []))

    def test_multiple_regressions_all_reported(self):
        baselines = {"tick_p95_ms": 10.0, "error_count": 0.0, "memory_count": 50.0}
        current = {"tick_p95_ms": 20.0, "error_count": 5.0, "memory_count": 200.0}
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"
        assert len(details["regressions"]) >= 2


# ---------------------------------------------------------------------------
# compare_metrics — improvement detection
# ---------------------------------------------------------------------------

class TestCompareMetricsImprovement:
    def test_p95_improvement_detected(self):
        baselines = {"tick_p95_ms": 10.0}
        current = {"tick_p95_ms": 7.0}  # 30% decrease, threshold is 20%
        result, details = compare_metrics(baselines, current)
        assert result == "improved"
        assert "tick_p95" in details["improvements"]

    def test_target_metric_improvement(self):
        baselines = {"custom_metric": 10.0}
        current = {"custom_metric": 8.0}  # 20% decrease
        result, details = compare_metrics(baselines, current, target_metrics=["custom_metric"])
        assert result == "improved"
        assert "custom_metric" in details["improvements"]


# ---------------------------------------------------------------------------
# compare_metrics — stable state
# ---------------------------------------------------------------------------

class TestCompareMetricsStable:
    def test_all_metrics_within_tolerance(self):
        baselines = {
            "tick_p95_ms": 10.0,
            "error_count": 5.0,
            "memory_count": 100.0,
            "event_violations": 2.0,
        }
        current = {
            "tick_p95_ms": 11.0,
            "error_count": 5.0,
            "memory_count": 110.0,
            "event_violations": 3.0,
        }
        result, details = compare_metrics(baselines, current)
        assert result == "stable"
        assert len(details["regressions"]) == 0
        assert "reason" in details

    def test_empty_baselines_is_stable(self):
        result, details = compare_metrics({}, {})
        assert result == "stable"

    def test_zero_baseline_p95_skipped(self):
        baselines = {"tick_p95_ms": 0.0}
        current = {"tick_p95_ms": 50.0}
        result, details = compare_metrics(baselines, current)
        assert result == "stable"

    def test_missing_current_metric_ignored(self):
        baselines = {"tick_p95_ms": 10.0}
        current = {}
        result, details = compare_metrics(baselines, current)
        assert result == "stable"


# ---------------------------------------------------------------------------
# compare_metrics — edge cases
# ---------------------------------------------------------------------------

class TestCompareMetricsEdge:
    def test_exact_tolerance_boundary_p95(self):
        baselines = {"tick_p95_ms": 10.0}
        threshold = 10.0 * (1.0 + P95_TOLERANCE)
        current = {"tick_p95_ms": threshold}
        result, _ = compare_metrics(baselines, current)
        assert result == "stable"

    def test_just_over_tolerance_boundary_p95(self):
        baselines = {"tick_p95_ms": 10.0}
        threshold = 10.0 * (1.0 + P95_TOLERANCE) + 0.01
        current = {"tick_p95_ms": threshold}
        result, _ = compare_metrics(baselines, current)
        assert result == "regressed"

    def test_exact_error_tolerance(self):
        baselines = {"error_count": 0.0}
        current = {"error_count": float(ERROR_TOLERANCE)}
        result, _ = compare_metrics(baselines, current)
        assert result == "stable"

    def test_just_over_error_tolerance(self):
        baselines = {"error_count": 0.0}
        current = {"error_count": float(ERROR_TOLERANCE + 1)}
        result, _ = compare_metrics(baselines, current)
        assert result == "regressed"


# ---------------------------------------------------------------------------
# PendingVerification + _to_dict
# ---------------------------------------------------------------------------

class TestPendingVerification:
    def test_creation_defaults(self):
        p = PendingVerification(
            patch_id="patch_001",
            description="test patch",
            files_changed=["brain/consciousness/kernel.py"],
            snapshot_path="/tmp/snap",
            conversation_id="conv_001",
            baselines={"tick_p95_ms": 10.0},
        )
        assert p.patch_id == "patch_001"
        assert p.boot_count == 0
        assert p.max_retries == DEFAULT_MAX_RETRIES
        assert p.verification_period_s == DEFAULT_VERIFICATION_PERIOD_S
        assert p.min_ticks == DEFAULT_MIN_TICKS

    def test_to_dict_roundtrip(self):
        p = PendingVerification(
            patch_id="patch_002",
            description="optimize tick",
            files_changed=["brain/test.py"],
            snapshot_path="/tmp/snap2",
            conversation_id="conv_002",
            baselines={"tick_p95_ms": 8.0, "error_count": 0.0},
            target_metrics=["tick_p95_ms"],
            upgrade_id="upg_001",
        )
        d = _to_dict(p)
        assert d["patch_id"] == "patch_002"
        assert d["baselines"]["tick_p95_ms"] == 8.0
        assert d["target_metrics"] == ["tick_p95_ms"]
        assert d["upgrade_id"] == "upg_001"

    def test_to_dict_is_json_serializable(self):
        p = PendingVerification(
            patch_id="p1", description="d", files_changed=[], snapshot_path="s",
            conversation_id="c", baselines={},
        )
        serialized = json.dumps(_to_dict(p))
        assert json.loads(serialized)["patch_id"] == "p1"


# ---------------------------------------------------------------------------
# File I/O: write / read / clear / has_pending
# ---------------------------------------------------------------------------

class TestFileIO:
    def _make_pending_file(self) -> tuple[Path, str]:
        tmpdir = tempfile.mkdtemp(prefix="jarvis_verif_test_")
        pending_file = Path(tmpdir) / "pending_verification.json"
        return pending_file, tmpdir

    def test_write_and_read_roundtrip(self):
        pending_file, tmpdir = self._make_pending_file()
        try:
            def fake_write(path, data, indent=None):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=indent, default=str))

            mock_persistence = MagicMock()
            mock_persistence.atomic_write_json = fake_write

            with patch("self_improve.verification.PENDING_FILE", pending_file), \
                 patch.dict("sys.modules", {"memory.persistence": mock_persistence}), \
                 patch("self_improve.verification._get_git_sha", return_value="abc123"):
                from self_improve.verification import write_pending
                write_pending(
                    patch_id="patch_test",
                    description="test write",
                    files_changed=["brain/foo.py"],
                    snapshot_path="/tmp/snap",
                    conversation_id="conv_test",
                    baselines={"tick_p95_ms": 5.0},
                    upgrade_id="upg_test",
                )

                result = read_pending()
                assert result is not None
                assert result.patch_id == "patch_test"
                assert result.description == "test write"
                assert result.baselines["tick_p95_ms"] == 5.0
                assert result.upgrade_id == "upg_test"
                assert result.git_sha == "abc123"
        finally:
            shutil.rmtree(tmpdir)

    def test_read_missing_returns_none(self):
        pending_file = Path("/tmp/jarvis_test_nonexistent_pending.json")
        with patch("self_improve.verification.PENDING_FILE", pending_file):
            assert read_pending() is None

    def test_read_corrupt_clears_file(self):
        pending_file, tmpdir = self._make_pending_file()
        try:
            pending_file.write_text("not valid json {{{")
            with patch("self_improve.verification.PENDING_FILE", pending_file):
                result = read_pending()
                assert result is None
                assert not pending_file.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_clear_pending_removes_file(self):
        pending_file, tmpdir = self._make_pending_file()
        try:
            pending_file.write_text("{}")
            assert pending_file.exists()
            with patch("self_improve.verification.PENDING_FILE", pending_file):
                clear_pending()
            assert not pending_file.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_clear_pending_missing_file_ok(self):
        pending_file = Path("/tmp/jarvis_test_nonexistent_clear.json")
        with patch("self_improve.verification.PENDING_FILE", pending_file):
            clear_pending()  # should not raise

    def test_has_pending_true(self):
        pending_file, tmpdir = self._make_pending_file()
        try:
            pending_file.write_text("{}")
            with patch("self_improve.verification.PENDING_FILE", pending_file):
                assert has_pending() is True
        finally:
            shutil.rmtree(tmpdir)

    def test_has_pending_false(self):
        pending_file = Path("/tmp/jarvis_test_nonexistent_has.json")
        with patch("self_improve.verification.PENDING_FILE", pending_file):
            assert has_pending() is False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_p95_tolerance_reasonable(self):
        assert 0.05 <= P95_TOLERANCE <= 0.50

    def test_error_tolerance_reasonable(self):
        assert 0 <= ERROR_TOLERANCE <= 10

    def test_memory_growth_factor_reasonable(self):
        assert 1.5 <= MEMORY_GROWTH_FACTOR <= 5.0

    def test_defaults_reasonable(self):
        assert DEFAULT_VERIFICATION_PERIOD_S > 0
        assert DEFAULT_MIN_TICKS > 0
        assert DEFAULT_MAX_RETRIES >= 1
