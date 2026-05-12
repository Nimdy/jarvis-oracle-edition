"""Per-hour-of-day metric baselines for time-aware counterfactual estimation.

If a research job completes at 3am when the system is naturally quiet, linear
trend extrapolation gives false credit for the low-noise environment.  This
tracker accumulates Welford (online mean/variance) statistics per metric per
hour-of-day (0-23), then supplies time-of-day-adjusted expected values to the
DeltaTracker's counterfactual engine.

Requires MIN_DAYS_FOR_TOD_BASELINE (3) days of data — at least
MIN_SAMPLES_PER_BUCKET (20) samples per hour bucket — before time-of-day
baselines influence counterfactual estimates.

Persistence: JSON at ~/.jarvis/metric_hourly.json, saved every PERSIST_INTERVAL_S.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MIN_DAYS_FOR_TOD_BASELINE = 3
MIN_SAMPLES_PER_BUCKET = 20
PERSIST_INTERVAL_S = 300.0

_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
_HISTORY_PATH = _JARVIS_DIR / "metric_hourly.json"

_METRIC_NAMES = (
    "confidence_avg",
    "confidence_volatility",
    "tick_p95_ms",
    "reasoning_coherence",
    "processing_health",
    "memory_count",
    "barge_in_count",
    "error_count",
)


@dataclass
class _WelfordBucket:
    """Online mean/variance via Welford's algorithm for a single metric-hour."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": round(self.mean, 6), "m2": round(self.m2, 6)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _WelfordBucket:
        return cls(count=d.get("count", 0), mean=d.get("mean", 0.0), m2=d.get("m2", 0.0))


class MetricHistoryTracker:
    """Accumulates per-hour-of-day statistics for 8 system metrics.

    Structure: ``_buckets[hour][metric_name] = _WelfordBucket``
    where hour is 0-23 (local time).
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = persist_path or _HISTORY_PATH
        self._buckets: dict[int, dict[str, _WelfordBucket]] = {
            h: {m: _WelfordBucket() for m in _METRIC_NAMES} for h in range(24)
        }
        self._last_persist_time: float = 0.0
        self._total_samples: int = 0
        self._load()

    def record(self, metrics: dict[str, float | int]) -> None:
        """Record a metric snapshot into the current hour bucket."""
        hour = time.localtime().tm_hour
        bucket = self._buckets[hour]
        for name in _METRIC_NAMES:
            val = metrics.get(name)
            if val is not None:
                bucket[name].update(float(val))
        self._total_samples += 1
        self._maybe_persist()

    def get_hour_avg(self, hour: int, metric_name: str) -> float | None:
        """Return the mean for *metric_name* at *hour* (0-23), or None if insufficient data."""
        hour = hour % 24
        bucket = self._buckets[hour].get(metric_name)
        if bucket is None or bucket.count < MIN_SAMPLES_PER_BUCKET:
            return None
        return bucket.mean

    def get_hour_std(self, hour: int, metric_name: str) -> float | None:
        """Return the std dev for *metric_name* at *hour*, or None if insufficient data."""
        hour = hour % 24
        bucket = self._buckets[hour].get(metric_name)
        if bucket is None or bucket.count < MIN_SAMPLES_PER_BUCKET:
            return None
        return bucket.std

    def has_sufficient_data(self, hour: int) -> bool:
        """True when all metrics in *hour* have >= MIN_SAMPLES_PER_BUCKET samples."""
        hour = hour % 24
        return all(
            self._buckets[hour][m].count >= MIN_SAMPLES_PER_BUCKET
            for m in _METRIC_NAMES
        )

    def get_coverage(self) -> dict[str, Any]:
        """Return coverage stats for dashboard/introspection."""
        hours_ready = sum(1 for h in range(24) if self.has_sufficient_data(h))
        min_samples = min(
            min(self._buckets[h][m].count for m in _METRIC_NAMES)
            for h in range(24)
        )
        return {
            "hours_ready": hours_ready,
            "hours_total": 24,
            "min_samples_any_bucket": min_samples,
            "min_required": MIN_SAMPLES_PER_BUCKET,
            "total_samples": self._total_samples,
            "active": hours_ready >= 1,
            "fully_calibrated": hours_ready == 24,
        }

    def get_state(self) -> dict[str, Any]:
        """Full serializable state for persistence and dashboard."""
        state: dict[str, Any] = {"version": 1, "total_samples": self._total_samples}
        for h in range(24):
            state[str(h)] = {m: self._buckets[h][m].to_dict() for m in _METRIC_NAMES}
        return state

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                logger.warning("MetricHistoryTracker: unknown version %s, ignoring", data.get("version"))
                return
            self._total_samples = data.get("total_samples", 0)
            for h in range(24):
                hour_data = data.get(str(h), {})
                for m in _METRIC_NAMES:
                    bucket_data = hour_data.get(m)
                    if bucket_data:
                        self._buckets[h][m] = _WelfordBucket.from_dict(bucket_data)
            logger.info(
                "MetricHistoryTracker loaded: %d total samples, %d/24 hours calibrated",
                self._total_samples,
                sum(1 for h in range(24) if self.has_sufficient_data(h)),
            )
        except Exception:
            logger.warning("Failed to load metric history", exc_info=True)

    def _maybe_persist(self) -> None:
        now = time.time()
        if now - self._last_persist_time < PERSIST_INTERVAL_S:
            return
        self._last_persist_time = now
        self._persist()

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.get_state(), separators=(",", ":")), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.warning("Failed to persist metric history", exc_info=True)

    def save(self) -> None:
        """Force-persist (called on graceful shutdown)."""
        self._persist()
