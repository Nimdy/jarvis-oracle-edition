"""Autonomy Calibrator — shadow-mode statistical observer for threshold tuning.

Phase 1: collect-only. Observes delta outcomes and builds per-bucket
distributions using Welford's online algorithm. Persists state. Exposes
readiness and suggested thresholds to the dashboard. Does NOT change any
live thresholds yet.

Each bucket is keyed by (metric_name, tool_type, tag_cluster_hash) and tracks:
  - count, mean, M2 (Welford variance accumulator)
  - win count + win rate
  - suggested MIN_MEANINGFUL_DELTA (mean + 1*sigma of the noise floor)

Data quality guards:
  - Stable-only: warmup/errored/unstable outcomes are rejected
  - Dedup by intent_id: the same outcome is never recorded twice
  - Coverage metric: % of recorded outcomes that map to a recognized bucket
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from autonomy.constants import MIN_MEANINGFUL_DELTA

logger = logging.getLogger(__name__)

_JARVIS_DIR = os.path.join(Path.home(), ".jarvis")
CALIBRATION_PATH = os.path.join(_JARVIS_DIR, "calibration_state.json")

READY_SAMPLE_THRESHOLD = 30
MAX_SEEN_IDS = 2000
LOG_RATE_LIMIT_S = 60.0


@dataclass
class CalibrationBucket:
    metric: str = ""
    tool: str = ""
    cluster_hash: str = ""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    win_count: int = 0

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def sigma(self) -> float:
        return math.sqrt(self.variance) if self.variance > 0 else 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / self.count if self.count > 0 else 0.0

    @property
    def suggested_delta(self) -> float:
        return self.mean + self.sigma if self.count >= 5 else MIN_MEANINGFUL_DELTA

    def update_welford(self, value: float, is_win: bool) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if is_win:
            self.win_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "tool": self.tool,
            "cluster_hash": self.cluster_hash,
            "count": self.count,
            "mean": round(self.mean, 6),
            "m2": round(self.m2, 6),
            "variance": round(self.variance, 6),
            "sigma": round(self.sigma, 6),
            "win_count": self.win_count,
            "win_rate": round(self.win_rate, 3),
            "suggested_delta": round(self.suggested_delta, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationBucket:
        count = d.get("count", 0)
        m2 = d.get("m2")
        if m2 is None:
            variance = d.get("variance", 0.0)
            m2 = float(variance) * max(0, count - 1)
        return cls(
            metric=d.get("metric", ""),
            tool=d.get("tool", ""),
            cluster_hash=d.get("cluster_hash", ""),
            count=count,
            mean=d.get("mean", 0.0),
            m2=m2,
            win_count=d.get("win_count", 0),
        )


def _bucket_key(metric: str, tool: str, cluster_hash: str) -> str:
    return f"{metric}|{tool}|{cluster_hash}"


class AutonomyCalibrator:
    """Shadow-mode calibration observer for the autonomy subsystem."""

    def __init__(self) -> None:
        self._buckets: dict[str, CalibrationBucket] = {}
        self._seen_intent_ids: deque[str] = deque(maxlen=MAX_SEEN_IDS)
        self._seen_set: set[str] = set()
        self._total_outcomes: int = 0
        self._mapped_outcomes: int = 0
        self._rejected_warmup: int = 0
        self._rejected_unstable: int = 0
        self._rejected_dedup: int = 0
        self._last_log_time: float = 0.0
        self._load()

    def record_outcome(
        self,
        intent_id: str,
        metric: str,
        tool: str,
        cluster_hash: str,
        delta: float,
        is_win: bool,
        warmup: bool = False,
        stable: bool = True,
        errored: bool = False,
        sample_count: int = 0,
    ) -> None:
        """Record a measured delta outcome into the appropriate bucket.

        Guards:
          - warmup outcomes are rejected
          - errored outcomes are rejected
          - unstable outcomes are rejected
          - duplicate intent_ids are rejected
          - sample_count < 3 treated as unstable
        """
        if warmup:
            self._rejected_warmup += 1
            return
        if errored or not stable or sample_count < 3:
            self._rejected_unstable += 1
            return
        if intent_id in self._seen_set:
            self._rejected_dedup += 1
            return

        self._seen_intent_ids.append(intent_id)
        self._seen_set.add(intent_id)
        if len(self._seen_set) > MAX_SEEN_IDS:
            self._seen_set = set(self._seen_intent_ids)

        self._total_outcomes += 1

        if not metric and not tool:
            return

        key = _bucket_key(metric, tool, cluster_hash)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = CalibrationBucket(
                metric=metric, tool=tool, cluster_hash=cluster_hash,
            )
            self._buckets[key] = bucket

        bucket.update_welford(abs(delta), is_win)
        self._mapped_outcomes += 1

        now = time.time()
        if now - self._last_log_time >= LOG_RATE_LIMIT_S:
            self._last_log_time = now
            logger.info(
                "Calibration: recorded intent=%s bucket=%s/%s/%s delta=%.4f win=%s (count=%d)",
                intent_id[:12], metric, tool, cluster_hash[:8],
                delta, is_win, bucket.count,
            )

    def get_suggested_thresholds(self) -> dict[str, float]:
        """Compute shadow threshold suggestions (not applied in Phase 1)."""
        if not self._buckets:
            return {"min_meaningful_delta": MIN_MEANINGFUL_DELTA}

        ready_buckets = [b for b in self._buckets.values() if b.count >= READY_SAMPLE_THRESHOLD]
        if not ready_buckets:
            return {"min_meaningful_delta": MIN_MEANINGFUL_DELTA}

        weighted_sum = sum(b.suggested_delta * b.count for b in ready_buckets)
        total_count = sum(b.count for b in ready_buckets)
        suggested = weighted_sum / total_count if total_count > 0 else MIN_MEANINGFUL_DELTA

        return {
            "min_meaningful_delta": round(suggested, 4),
        }

    def get_readiness(self) -> dict[str, Any]:
        """Dashboard-ready status: readiness %, suggested thresholds, top buckets."""
        total_buckets = len(self._buckets)
        ready = sum(1 for b in self._buckets.values() if b.count >= READY_SAMPLE_THRESHOLD)
        readiness_pct = (ready / total_buckets * 100.0) if total_buckets > 0 else 0.0
        coverage_pct = (
            self._mapped_outcomes / self._total_outcomes * 100.0
            if self._total_outcomes > 0 else 0.0
        )

        suggestions = self.get_suggested_thresholds()

        top = sorted(self._buckets.values(), key=lambda b: b.count, reverse=True)[:10]

        return {
            "ready_bucket_count": ready,
            "total_buckets": total_buckets,
            "readiness_pct": round(readiness_pct, 1),
            "total_outcomes": self._total_outcomes,
            "mapped_outcomes": self._mapped_outcomes,
            "coverage_pct": round(coverage_pct, 1),
            "rejected_warmup": self._rejected_warmup,
            "rejected_unstable": self._rejected_unstable,
            "rejected_dedup": self._rejected_dedup,
            "suggested_min_delta": suggestions["min_meaningful_delta"],
            "current_min_delta": MIN_MEANINGFUL_DELTA,
            "active": False,
            "top_buckets": [b.to_dict() for b in top],
        }

    def save(self) -> None:
        try:
            os.makedirs(_JARVIS_DIR, exist_ok=True)
            data = {
                "version": 1,
                "saved_at": time.time(),
                "total_outcomes": self._total_outcomes,
                "mapped_outcomes": self._mapped_outcomes,
                "rejected_warmup": self._rejected_warmup,
                "rejected_unstable": self._rejected_unstable,
                "rejected_dedup": self._rejected_dedup,
                "buckets": {k: b.to_dict() for k, b in self._buckets.items()},
                "seen_intent_ids": list(self._seen_intent_ids),
            }
            tmp = CALIBRATION_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, CALIBRATION_PATH)
        except Exception:
            logger.warning("Failed to save calibration state", exc_info=True)

    def _load(self) -> None:
        if not os.path.isfile(CALIBRATION_PATH):
            return
        try:
            with open(CALIBRATION_PATH) as f:
                data = json.load(f)
            self._total_outcomes = data.get("total_outcomes", 0)
            self._mapped_outcomes = data.get("mapped_outcomes", 0)
            self._rejected_warmup = data.get("rejected_warmup", 0)
            self._rejected_unstable = data.get("rejected_unstable", 0)
            self._rejected_dedup = data.get("rejected_dedup", 0)
            for key, bd in data.get("buckets", {}).items():
                self._buckets[key] = CalibrationBucket.from_dict(bd)
            for iid in data.get("seen_intent_ids", []):
                self._seen_intent_ids.append(iid)
                self._seen_set.add(iid)
            logger.info(
                "Loaded calibration state: %d buckets, %d outcomes",
                len(self._buckets), self._total_outcomes,
            )
        except Exception:
            logger.warning("Failed to load calibration state", exc_info=True)
