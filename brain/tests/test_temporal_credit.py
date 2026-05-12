"""Tests for Phase 2: Temporal Credit + Counterfactual Baselines.

Covers:
  - MetricHistoryTracker (Welford stats, persistence, coverage)
  - DeltaTracker blended counterfactual (trend-only, blended, time-of-day)
  - DeltaResult.counterfactual_source field
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from autonomy.delta_tracker import (
    DeltaResult,
    DeltaTracker,
    MetricSnapshot,
    _LOWER_BETTER,
    _HIGHER_BETTER,
    _PendingDelta,
)
from autonomy.metric_history import (
    MIN_SAMPLES_PER_BUCKET,
    MetricHistoryTracker,
    _WelfordBucket,
    _METRIC_NAMES,
)


# ═══════════════════════════════════════════════════════════════════════
# Welford Bucket
# ═══════════════════════════════════════════════════════════════════════

class TestWelfordBucket:

    def test_empty_bucket(self):
        b = _WelfordBucket()
        assert b.count == 0
        assert b.mean == 0.0
        assert b.variance == 0.0
        assert b.std == 0.0

    def test_single_value(self):
        b = _WelfordBucket()
        b.update(5.0)
        assert b.count == 1
        assert b.mean == 5.0
        assert b.variance == 0.0

    def test_known_series(self):
        b = _WelfordBucket()
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            b.update(v)
        assert b.count == 8
        assert abs(b.mean - 5.0) < 0.001
        # Sample variance (Bessel-corrected): sum((x-mean)^2)/(n-1) = 32/7
        assert abs(b.variance - 32.0 / 7.0) < 0.001
        assert abs(b.std - (32.0 / 7.0) ** 0.5) < 0.001

    def test_roundtrip_serialization(self):
        b = _WelfordBucket()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            b.update(v)
        d = b.to_dict()
        b2 = _WelfordBucket.from_dict(d)
        assert b2.count == b.count
        assert abs(b2.mean - b.mean) < 1e-5
        assert abs(b2.m2 - b.m2) < 1e-5


# ═══════════════════════════════════════════════════════════════════════
# MetricHistoryTracker
# ═══════════════════════════════════════════════════════════════════════

class TestMetricHistoryTracker:

    def _make_tracker(self, tmp_path: Path | None = None) -> MetricHistoryTracker:
        path = tmp_path or Path(tempfile.mktemp(suffix=".json"))
        return MetricHistoryTracker(persist_path=path)

    def _sample_metrics(self, **overrides) -> dict[str, float | int]:
        base = {
            "confidence_avg": 0.7,
            "confidence_volatility": 0.1,
            "tick_p95_ms": 8.0,
            "reasoning_coherence": 0.8,
            "processing_health": 0.9,
            "memory_count": 100,
            "barge_in_count": 0,
            "error_count": 0,
        }
        base.update(overrides)
        return base

    def test_initial_state(self):
        t = self._make_tracker()
        cov = t.get_coverage()
        assert cov["hours_ready"] == 0
        assert cov["total_samples"] == 0
        assert cov["active"] is False
        assert cov["fully_calibrated"] is False

    def test_record_accumulates(self):
        t = self._make_tracker()
        for _ in range(5):
            t.record(self._sample_metrics())
        assert t._total_samples == 5

    def test_insufficient_data_returns_none(self):
        t = self._make_tracker()
        for _ in range(5):
            t.record(self._sample_metrics())
        hour = time.localtime().tm_hour
        assert t.get_hour_avg(hour, "confidence_avg") is None
        assert t.has_sufficient_data(hour) is False

    def test_sufficient_data_returns_avg(self):
        t = self._make_tracker()
        for i in range(MIN_SAMPLES_PER_BUCKET + 5):
            t.record(self._sample_metrics(confidence_avg=0.5 + i * 0.01))
        hour = time.localtime().tm_hour
        avg = t.get_hour_avg(hour, "confidence_avg")
        assert avg is not None
        assert 0.5 < avg < 1.0
        assert t.has_sufficient_data(hour) is True

    def test_get_hour_std(self):
        t = self._make_tracker()
        for _ in range(MIN_SAMPLES_PER_BUCKET + 5):
            t.record(self._sample_metrics(confidence_avg=0.7))
        hour = time.localtime().tm_hour
        std = t.get_hour_std(hour, "confidence_avg")
        assert std is not None
        assert std < 0.001

    def test_coverage_reports_ready_hours(self):
        t = self._make_tracker()
        for _ in range(MIN_SAMPLES_PER_BUCKET + 1):
            t.record(self._sample_metrics())
        cov = t.get_coverage()
        assert cov["hours_ready"] == 1
        assert cov["active"] is True
        assert cov["fully_calibrated"] is False

    def test_persistence_roundtrip(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        t1 = MetricHistoryTracker(persist_path=path)
        for _ in range(MIN_SAMPLES_PER_BUCKET + 5):
            t1.record(self._sample_metrics(confidence_avg=0.75))
        t1.save()

        assert path.exists()

        t2 = MetricHistoryTracker(persist_path=path)
        hour = time.localtime().tm_hour
        avg = t2.get_hour_avg(hour, "confidence_avg")
        assert avg is not None
        assert abs(avg - 0.75) < 0.001
        assert t2._total_samples == t1._total_samples

        path.unlink(missing_ok=True)

    def test_all_metric_names_tracked(self):
        t = self._make_tracker()
        t.record(self._sample_metrics())
        hour = time.localtime().tm_hour
        for name in _METRIC_NAMES:
            bucket = t._buckets[hour][name]
            assert bucket.count == 1

    def test_wraps_hour(self):
        t = self._make_tracker()
        assert t.get_hour_avg(25, "confidence_avg") is None
        assert t.get_hour_avg(-1, "confidence_avg") is None

    def test_get_state_serializable(self):
        t = self._make_tracker()
        t.record(self._sample_metrics())
        state = t.get_state()
        assert state["version"] == 1
        assert state["total_samples"] == 1
        serialized = json.dumps(state)
        assert len(serialized) > 0


# ═══════════════════════════════════════════════════════════════════════
# DeltaTracker — Blended Counterfactual
# ═══════════════════════════════════════════════════════════════════════

def _make_readings(n: int, base_time: float, interval: float = 5.0,
                   confidence_avg: float = 0.7, **kwargs) -> list[MetricSnapshot]:
    """Generate a series of MetricSnapshots for testing."""
    readings = []
    for i in range(n):
        readings.append(MetricSnapshot(
            timestamp=base_time + i * interval,
            confidence_avg=confidence_avg,
            confidence_volatility=kwargs.get("confidence_volatility", 0.1),
            tick_p95_ms=kwargs.get("tick_p95_ms", 8.0),
            reasoning_coherence=kwargs.get("reasoning_coherence", 0.8),
            processing_health=kwargs.get("processing_health", 0.9),
            memory_count=kwargs.get("memory_count", 100),
            barge_in_count=kwargs.get("barge_in_count", 0),
            error_count=kwargs.get("error_count", 0),
        ))
    return readings


class TestDeltaTrackerCounterfactualSource:

    def test_result_has_counterfactual_source_field(self):
        dr = DeltaResult(intent_id="x", baseline=MetricSnapshot())
        assert dr.counterfactual_source == "trend"
        d = dr.to_dict()
        assert "counterfactual_source" in d
        assert d["counterfactual_source"] == "trend"

    def test_result_blended_source(self):
        dr = DeltaResult(intent_id="x", baseline=MetricSnapshot(), counterfactual_source="blended")
        assert dr.counterfactual_source == "blended"
        assert dr.to_dict()["counterfactual_source"] == "blended"


class TestDeltaTrackerTrendOnly:
    """Verify existing trend-only behavior is preserved when no metric_history is set."""

    def test_trend_only_when_no_history(self):
        dt = DeltaTracker(metric_history=None)
        base_time = time.time() - 700
        for r in _make_readings(20, base_time):
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 50),
            completion_time=base_time + 200,
            post_check_time=base_time + 800,
        )
        result, source = dt._extrapolate_trend(pending)
        assert result is not None
        assert source == "trend"

    def test_returns_none_with_few_readings(self):
        dt = DeltaTracker(metric_history=None)
        base_time = time.time() - 100
        for r in _make_readings(3, base_time):
            dt.record_metrics(r)
        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 10),
            completion_time=base_time + 50,
            post_check_time=base_time + 100,
        )
        result, source = dt._extrapolate_trend(pending)
        assert result is None
        assert source == "trend"


class TestDeltaTrackerBlended:
    """Verify blended counterfactual with time-of-day baselines."""

    def _calibrated_history(self) -> MetricHistoryTracker:
        """Return a tracker with sufficient data for the current hour."""
        tracker = MetricHistoryTracker(persist_path=Path(tempfile.mktemp(suffix=".json")))
        metrics = {
            "confidence_avg": 0.6,
            "confidence_volatility": 0.15,
            "tick_p95_ms": 12.0,
            "reasoning_coherence": 0.7,
            "processing_health": 0.85,
            "memory_count": 80,
            "barge_in_count": 1,
            "error_count": 0,
        }
        for _ in range(MIN_SAMPLES_PER_BUCKET + 5):
            tracker.record(metrics)
        return tracker

    def test_blended_when_history_available(self):
        mh = self._calibrated_history()
        dt = DeltaTracker(metric_history=mh)
        base_time = time.time() - 700

        readings = _make_readings(20, base_time, confidence_avg=0.9)
        for r in readings:
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 50, confidence_avg=0.9),
            completion_time=base_time + 200,
            post_check_time=base_time + 800,
        )
        result, source = dt._extrapolate_trend(pending)
        assert result is not None
        assert source == "blended"

    def test_blended_uses_conservative_estimate_higher_better(self):
        """For higher-is-better metrics, blended should use min(trend, tod)."""
        mh = self._calibrated_history()
        dt = DeltaTracker(metric_history=mh)
        base_time = time.time() - 700

        readings = _make_readings(20, base_time, confidence_avg=0.9)
        for r in readings:
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 50, confidence_avg=0.9),
            completion_time=base_time + 200,
            post_check_time=base_time + 800,
        )
        result, _ = dt._extrapolate_trend(pending)

        tod_avg = mh.get_hour_avg(time.localtime().tm_hour, "confidence_avg")
        if tod_avg is not None and tod_avg < 0.9:
            assert result.confidence_avg <= 0.9 + 0.001

    def test_blended_uses_conservative_estimate_lower_better(self):
        """For lower-is-better metrics, blended should use max(trend, tod)."""
        mh = self._calibrated_history()
        dt = DeltaTracker(metric_history=mh)
        base_time = time.time() - 700

        readings = _make_readings(20, base_time, tick_p95_ms=5.0)
        for r in readings:
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 50, tick_p95_ms=5.0),
            completion_time=base_time + 200,
            post_check_time=base_time + 800,
        )
        result, _ = dt._extrapolate_trend(pending)

        tod_avg = mh.get_hour_avg(time.localtime().tm_hour, "tick_p95_ms")
        if tod_avg is not None and tod_avg > 5.0:
            assert result.tick_p95_ms >= 5.0 - 0.001

    def test_trend_only_when_history_empty(self):
        """If metric_history exists but has no data, fall back to trend."""
        mh = MetricHistoryTracker(persist_path=Path(tempfile.mktemp(suffix=".json")))
        dt = DeltaTracker(metric_history=mh)
        base_time = time.time() - 700

        for r in _make_readings(20, base_time):
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test",
            baseline=MetricSnapshot(timestamp=base_time + 50),
            completion_time=base_time + 200,
            post_check_time=base_time + 800,
        )
        result, source = dt._extrapolate_trend(pending)
        assert result is not None
        assert source == "trend"


class TestDeltaTrackerMeasure:
    """Verify that _measure() passes counterfactual_source through to DeltaResult."""

    def test_measured_result_has_source(self):
        dt = DeltaTracker(metric_history=None)
        base_time = time.time() - 1200

        for r in _make_readings(40, base_time, interval=5.0):
            dt.record_metrics(r)
        for r in _make_readings(20, base_time + 700, interval=5.0, confidence_avg=0.75):
            dt.record_metrics(r)

        pending = _PendingDelta(
            intent_id="test_measure",
            baseline=MetricSnapshot(timestamp=base_time + 100, confidence_avg=0.7),
            completion_time=base_time + 600,
            post_check_time=base_time + 700,
        )

        result = dt._measure(pending)
        assert result.status == "measured"
        assert result.counterfactual_source in ("trend", "blended", "time_of_day")

    def test_full_lifecycle_end_to_end(self):
        """start_tracking -> record -> mark_completed -> check_pending."""
        dt = DeltaTracker(metric_history=None)
        now = time.time()

        for r in _make_readings(30, now - 800, interval=5.0):
            dt.record_metrics(r)

        baseline = dt.start_tracking("lifecycle_test")
        assert baseline is not None

        for r in _make_readings(10, now - 500, interval=5.0, confidence_avg=0.75):
            dt.record_metrics(r)

        dt.mark_completed("lifecycle_test")

        for p in dt._pending:
            if p.intent_id == "lifecycle_test":
                p.post_check_time = now - 10

        for r in _make_readings(20, now - 300, interval=5.0, confidence_avg=0.78):
            dt.record_metrics(r)

        results = dt.check_pending()

        assert len(results) == 1
        dr = results[0]
        assert dr.intent_id == "lifecycle_test"
        assert dr.status == "measured"
        assert dr.counterfactual_source in ("trend", "blended", "time_of_day")
        assert "counterfactual_source" in dr.to_dict()
