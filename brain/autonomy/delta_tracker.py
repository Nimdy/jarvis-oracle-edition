"""Delta Tracker — before/after metric logging for autonomous job credit assignment.

Every research job and improvement attempt records:
  - baseline window (metrics averaged over the last N seconds before start)
  - post window    (metrics averaged over N seconds after completion)
  - delta          (per-metric signed improvement)
  - counterfactual (what the metric would have done with no intervention)
  - attribution    (delta minus counterfactual — true causal credit)
  - net_improvement (aggregate scalar: positive = helped)
  - stability      (did the improvement persist past the post window?)
  - sample_count   (confidence interval proxy)

The counterfactual baseline prevents false credit: if a metric was already
trending up, the job doesn't get credit for the natural improvement.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DELTA_PENDING_PATH = Path("~/.jarvis/delta_pending.json").expanduser()
DELTA_COUNTERS_PATH = Path("~/.jarvis/delta_counters.json").expanduser()

BASELINE_WINDOW_S = 600.0
POST_WINDOW_S = 600.0

_LOWER_BETTER = frozenset({
    "confidence_volatility", "tick_p95_ms", "barge_in_count", "error_count",
    "friction_rate",
})
_HIGHER_BETTER = frozenset({
    "confidence_avg", "reasoning_coherence", "processing_health", "memory_count",
    "retrieval_hit_rate", "belief_graph_coverage", "contradiction_resolution_rate",
})


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of key system metrics."""

    timestamp: float = 0.0
    confidence_avg: float = 0.0
    confidence_volatility: float = 0.0
    tick_p95_ms: float = 0.0
    reasoning_coherence: float = 0.0
    processing_health: float = 0.0
    memory_count: int = 0
    barge_in_count: int = 0
    error_count: int = 0
    retrieval_hit_rate: float = 0.0
    belief_graph_coverage: float = 0.0
    contradiction_resolution_rate: float = 0.0
    friction_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "confidence_avg": round(self.confidence_avg, 3),
            "confidence_volatility": round(self.confidence_volatility, 3),
            "tick_p95_ms": round(self.tick_p95_ms, 2),
            "reasoning_coherence": round(self.reasoning_coherence, 3),
            "processing_health": round(self.processing_health, 3),
            "memory_count": self.memory_count,
            "barge_in_count": self.barge_in_count,
            "error_count": self.error_count,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 4),
            "belief_graph_coverage": round(self.belief_graph_coverage, 4),
            "contradiction_resolution_rate": round(self.contradiction_resolution_rate, 4),
            "friction_rate": round(self.friction_rate, 4),
        }


@dataclass
class DeltaResult:
    """Before/after comparison for a single autonomy job."""

    intent_id: str
    baseline: MetricSnapshot
    post: MetricSnapshot | None = None
    counterfactual: MetricSnapshot | None = None
    deltas: dict[str, float] = field(default_factory=dict)
    counterfactual_deltas: dict[str, float] = field(default_factory=dict)
    attribution: dict[str, float] = field(default_factory=dict)
    net_improvement: float = 0.0
    net_counterfactual: float = 0.0
    net_attribution: float = 0.0
    stable: bool = False
    sample_count: int = 0
    status: str = "pending"
    counterfactual_source: str = "trend"
    source_event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "baseline": self.baseline.to_dict(),
            "post": self.post.to_dict() if self.post else None,
            "counterfactual": self.counterfactual.to_dict() if self.counterfactual else None,
            "deltas": {k: round(v, 4) for k, v in self.deltas.items()},
            "counterfactual_deltas": {k: round(v, 4) for k, v in self.counterfactual_deltas.items()},
            "attribution": {k: round(v, 4) for k, v in self.attribution.items()},
            "net_improvement": round(self.net_improvement, 4),
            "net_counterfactual": round(self.net_counterfactual, 4),
            "net_attribution": round(self.net_attribution, 4),
            "stable": self.stable,
            "sample_count": self.sample_count,
            "status": self.status,
            "counterfactual_source": self.counterfactual_source,
        }


@dataclass
class _PendingDelta:
    intent_id: str
    baseline: MetricSnapshot
    completion_time: float = 0.0
    post_check_time: float = 0.0
    source_event: str = ""


class DeltaTracker:
    """Tracks before/after metrics for every autonomy job."""

    def __init__(self, metric_history: Any | None = None) -> None:
        self._metric_ring: deque[MetricSnapshot] = deque(maxlen=120)
        self._pending: list[_PendingDelta] = []
        self._completed: deque[DeltaResult] = deque(maxlen=100)
        self._total_measured: int = 0
        self._total_improved: int = 0
        self._total_regressed: int = 0
        self._total_interrupted: int = 0
        self._metric_history = metric_history

    def record_metrics(self, snapshot: MetricSnapshot) -> None:
        """Called periodically (~5s) to build the rolling metric buffer."""
        self._metric_ring.append(snapshot)

    def start_tracking(self, intent_id: str, source_event: str = "") -> MetricSnapshot:
        """Capture a baseline before a research job starts."""
        baseline = self._avg_window(BASELINE_WINDOW_S)
        self._pending.append(_PendingDelta(
            intent_id=intent_id, baseline=baseline, source_event=source_event,
        ))
        return baseline

    def mark_completed(self, intent_id: str) -> None:
        """Schedule the post-window measurement for *intent_id*."""
        now = time.time()
        for p in self._pending:
            if p.intent_id == intent_id:
                p.completion_time = now
                p.post_check_time = now + POST_WINDOW_S
                return

    def check_pending(self) -> list[DeltaResult]:
        """Harvest any pending deltas whose post-window has elapsed."""
        now = time.time()
        ready: list[DeltaResult] = []
        remaining: list[_PendingDelta] = []

        for p in self._pending:
            if p.completion_time == 0.0:
                if now - p.baseline.timestamp > BASELINE_WINDOW_S * 3:
                    self._completed.append(DeltaResult(
                        intent_id=p.intent_id, baseline=p.baseline, status="expired",
                    ))
                else:
                    remaining.append(p)
                continue

            if now >= p.post_check_time:
                result = self._measure(p)
                self._completed.append(result)
                self._total_measured += 1
                if result.net_improvement > 0.01:
                    self._total_improved += 1
                elif result.net_improvement < -0.01:
                    self._total_regressed += 1
                ready.append(result)
            else:
                remaining.append(p)

        self._pending = remaining
        return ready

    # -- persistence -----------------------------------------------------------

    def save_pending(self) -> None:
        """Persist pending delta windows to disk (atomic write)."""
        entries = []
        for p in self._pending:
            entry: dict[str, Any] = {
                "intent_id": p.intent_id,
                "baseline": p.baseline.to_dict(),
                "completion_time": p.completion_time,
                "post_check_time": p.post_check_time,
            }
            if p.source_event:
                entry["source_event"] = p.source_event
            entries.append(entry)
        try:
            DELTA_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=DELTA_PENDING_PATH.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(entries, f, separators=(",", ":"))
                os.replace(tmp, DELTA_PENDING_PATH)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            logger.debug("DeltaTracker save_pending failed", exc_info=True)

    def save_counters(self) -> None:
        """Persist cumulative counters to disk (atomic write)."""
        data = {
            "counters": {
                "total_measured": self._total_measured,
                "total_improved": self._total_improved,
                "total_regressed": self._total_regressed,
                "total_interrupted": self._total_interrupted,
            },
        }
        try:
            DELTA_COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=DELTA_COUNTERS_PATH.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                os.replace(tmp, DELTA_COUNTERS_PATH)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            logger.debug("DeltaTracker save_counters failed", exc_info=True)

    def load_counters(self) -> None:
        """Restore cumulative counters from disk. Tolerant of missing/corrupt files."""
        if not DELTA_COUNTERS_PATH.exists():
            return
        try:
            data = json.loads(DELTA_COUNTERS_PATH.read_text())
            counters = data.get("counters", {})
            self._total_measured += counters.get("total_measured", 0)
            self._total_improved += counters.get("total_improved", 0)
            self._total_regressed += counters.get("total_regressed", 0)
            self._total_interrupted += counters.get("total_interrupted", 0)
            logger.info(
                "DeltaTracker counters restored: measured=%d, improved=%d, "
                "regressed=%d, interrupted=%d",
                self._total_measured, self._total_improved,
                self._total_regressed, self._total_interrupted,
            )
        except Exception:
            logger.warning("DeltaTracker load_counters: corrupt file, skipping")

    def load_pending(self) -> tuple[int, list[DeltaResult]]:
        """Reload pending delta windows from disk after restart.

        Returns (count_restored, list_of_interrupted_results).  Completed
        jobs whose post-window has elapsed get a terminal
        ``interrupted_by_restart`` outcome.  Jobs still within their
        post-window are restored to ``_pending`` for normal measurement.
        Never-completed jobs are discarded.
        """
        interrupted: list[DeltaResult] = []
        restored = 0

        if not DELTA_PENDING_PATH.exists():
            return 0, []

        try:
            entries = json.loads(DELTA_PENDING_PATH.read_text())
        except Exception:
            logger.warning("DeltaTracker load_pending: corrupt file, skipping")
            return 0, []

        now = time.time()
        for e in entries:
            intent_id = e.get("intent_id", "")
            completion = e.get("completion_time", 0.0)
            post_check = e.get("post_check_time", 0.0)
            baseline_dict = e.get("baseline", {})

            baseline = MetricSnapshot(
                timestamp=baseline_dict.get("timestamp", 0.0),
                confidence_avg=baseline_dict.get("confidence_avg", 0.0),
                confidence_volatility=baseline_dict.get("confidence_volatility", 0.0),
                tick_p95_ms=baseline_dict.get("tick_p95_ms", 0.0),
                reasoning_coherence=baseline_dict.get("reasoning_coherence", 0.0),
                processing_health=baseline_dict.get("processing_health", 0.0),
                memory_count=baseline_dict.get("memory_count", 0),
                barge_in_count=baseline_dict.get("barge_in_count", 0),
                error_count=baseline_dict.get("error_count", 0),
                retrieval_hit_rate=baseline_dict.get("retrieval_hit_rate", 0.0),
                belief_graph_coverage=baseline_dict.get("belief_graph_coverage", 0.0),
                contradiction_resolution_rate=baseline_dict.get("contradiction_resolution_rate", 0.0),
                friction_rate=baseline_dict.get("friction_rate", 0.0),
            )

            if completion == 0.0:
                continue

            se = e.get("source_event", "")

            if now >= post_check:
                result = DeltaResult(
                    intent_id=intent_id,
                    baseline=baseline,
                    status="interrupted_by_restart",
                )
                result.source_event = se
                interrupted.append(result)
                self._completed.append(result)
                self._total_interrupted += 1
            else:
                self._pending.append(_PendingDelta(
                    intent_id=intent_id,
                    baseline=baseline,
                    completion_time=completion,
                    post_check_time=post_check,
                    source_event=se,
                ))
                restored += 1

        try:
            DELTA_PENDING_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        if restored or interrupted:
            logger.info(
                "DeltaTracker loaded: %d restored to pending, %d interrupted_by_restart",
                restored, len(interrupted),
            )
        return restored, interrupted

    # -- introspection --------------------------------------------------------

    def get_recent_deltas(self, limit: int = 10) -> list[dict[str, Any]]:
        return [d.to_dict() for d in list(self._completed)[-limit:]]

    def get_stats(self) -> dict[str, Any]:
        now = time.time()
        return {
            "total_measured": self._total_measured,
            "total_improved": self._total_improved,
            "total_regressed": self._total_regressed,
            "total_interrupted": self._total_interrupted,
            "pending_count": len(self._pending),
            "pending_details": [
                {
                    "intent_id": p.intent_id,
                    "baseline_time": p.baseline.timestamp,
                    "completion_time": p.completion_time,
                    "post_check_time": p.post_check_time,
                    "time_until_ready_s": round(max(0, p.post_check_time - now), 1)
                        if p.post_check_time > 0 else None,
                    "status": "awaiting_completion" if p.completion_time == 0.0
                        else ("ready" if now >= p.post_check_time else "waiting_post_window"),
                }
                for p in self._pending
            ],
            "improvement_rate": (
                self._total_improved / self._total_measured
                if self._total_measured > 0 else 0.0
            ),
        }

    # -- internals ------------------------------------------------------------

    def _avg_window(self, window_s: float) -> MetricSnapshot:
        readings = list(self._metric_ring)
        if not readings:
            return MetricSnapshot(timestamp=time.time())
        now = time.time()
        window = [r for r in readings if now - r.timestamp <= window_s] or readings[-10:]
        return self._average(window)

    def _measure(self, pending: _PendingDelta) -> DeltaResult:
        post = self._post_snapshot(pending.completion_time)
        baseline = pending.baseline
        counterfactual, cf_source = self._extrapolate_trend(pending)

        deltas: dict[str, float] = {}
        cf_deltas: dict[str, float] = {}
        attribution: dict[str, float] = {}

        for name in _LOWER_BETTER | _HIGHER_BETTER:
            b = getattr(baseline, name, 0)
            p = getattr(post, name, 0)
            cf = getattr(counterfactual, name, 0) if counterfactual else b
            denom = max(abs(b), 0.001)

            if name in _LOWER_BETTER:
                deltas[name] = (b - p) / denom if b else 0.0
                cf_deltas[name] = (b - cf) / denom if b else 0.0
            else:
                deltas[name] = (p - b) / denom if b else (0.1 if p else 0.0)
                cf_deltas[name] = (cf - b) / denom if b else 0.0

            attribution[name] = deltas[name] - cf_deltas[name]

        net = sum(deltas.values()) / max(len(deltas), 1)
        net_cf = sum(cf_deltas.values()) / max(len(cf_deltas), 1)
        net_attr = sum(attribution.values()) / max(len(attribution), 1)

        readings = list(self._metric_ring)
        recent = [r for r in readings if r.timestamp > pending.completion_time + POST_WINDOW_S * 0.5]
        stable = len(recent) >= 3

        return DeltaResult(
            intent_id=pending.intent_id, baseline=baseline, post=post,
            counterfactual=counterfactual,
            deltas=deltas, counterfactual_deltas=cf_deltas, attribution=attribution,
            net_improvement=net, net_counterfactual=net_cf, net_attribution=net_attr,
            stable=stable, sample_count=len(recent), status="measured",
            counterfactual_source=cf_source,
            source_event=pending.source_event,
        )

    def _extrapolate_trend(self, pending: _PendingDelta) -> tuple[MetricSnapshot | None, str]:
        """Extrapolate pre-job metric trend to estimate what would have
        happened with no intervention (counterfactual control baseline).

        Uses linear regression over the baseline window, optionally blended
        with time-of-day historical averages from MetricHistoryTracker when
        sufficient data has been accumulated (Phase 2: Temporal Credit).

        Returns ``(snapshot, source)`` where *source* is one of
        ``"trend"``, ``"time_of_day"``, or ``"blended"``.
        """
        readings = list(self._metric_ring)
        if len(readings) < 6:
            return None, "trend"

        job_start = pending.baseline.timestamp
        pre_readings = [r for r in readings if r.timestamp <= job_start]
        if len(pre_readings) < 4:
            pre_readings = readings[:len(readings) // 2]
        if len(pre_readings) < 4:
            return None, "trend"

        target_t = pending.completion_time + POST_WINDOW_S * 0.5
        duration = target_t - job_start

        target_hour = time.localtime(target_t).tm_hour
        mh = self._metric_history
        tod_available = mh is not None and mh.has_sufficient_data(target_hour)

        result = MetricSnapshot(timestamp=target_t)
        metric_names = [f.name for f in MetricSnapshot.__dataclass_fields__.values() if f.name != "timestamp"]
        used_tod = False

        for name in metric_names:
            values = [getattr(r, name, 0) for r in pre_readings]
            n = len(values)
            if n < 2:
                setattr(result, name, getattr(pending.baseline, name, 0))
                continue

            half = n // 2
            first_half = sum(values[:half]) / half
            second_half = sum(values[half:]) / (n - half)

            first_t = sum(r.timestamp for r in pre_readings[:half]) / half
            second_t = sum(r.timestamp for r in pre_readings[half:]) / (n - half)
            dt = second_t - first_t
            if dt < 1.0:
                setattr(result, name, getattr(pending.baseline, name, 0))
                continue

            slope = (second_half - first_half) / dt
            trend_predicted = second_half + slope * duration

            if tod_available:
                tod_expected = mh.get_hour_avg(target_hour, name)
                if tod_expected is not None:
                    used_tod = True
                    if name in _LOWER_BETTER:
                        predicted = max(trend_predicted, tod_expected)
                    else:
                        predicted = min(trend_predicted, tod_expected)
                else:
                    predicted = trend_predicted
            else:
                predicted = trend_predicted

            baseline_val = getattr(pending.baseline, name, 0)
            if isinstance(baseline_val, int):
                predicted = max(0, int(round(predicted)))
            else:
                max_change = abs(baseline_val) * 0.5 + 0.1
                predicted = max(baseline_val - max_change, min(baseline_val + max_change, predicted))

            setattr(result, name, predicted)

        if tod_available and used_tod:
            source = "blended"
        elif tod_available:
            source = "time_of_day"
        else:
            source = "trend"
        return result, source

    def _post_snapshot(self, completion_time: float) -> MetricSnapshot:
        readings = list(self._metric_ring)
        post = [r for r in readings if r.timestamp > completion_time]
        if not post:
            post = readings[-5:] if readings else []
        if not post:
            return MetricSnapshot(timestamp=time.time())
        return self._average(post)

    @staticmethod
    def _average(readings: list[MetricSnapshot]) -> MetricSnapshot:
        n = len(readings)
        if n == 0:
            return MetricSnapshot(timestamp=time.time())
        dt = max(1.0, readings[-1].timestamp - readings[0].timestamp) if n > 1 else 60.0
        barge_rate = max(0, readings[-1].barge_in_count - readings[0].barge_in_count) / (dt / 3600.0)
        error_rate = max(0, readings[-1].error_count - readings[0].error_count) / (dt / 3600.0)
        return MetricSnapshot(
            timestamp=time.time(),
            confidence_avg=sum(r.confidence_avg for r in readings) / n,
            confidence_volatility=sum(r.confidence_volatility for r in readings) / n,
            tick_p95_ms=sum(r.tick_p95_ms for r in readings) / n,
            reasoning_coherence=sum(r.reasoning_coherence for r in readings) / n,
            processing_health=sum(r.processing_health for r in readings) / n,
            memory_count=readings[-1].memory_count,
            barge_in_count=int(barge_rate),
            error_count=int(error_rate),
            retrieval_hit_rate=sum(r.retrieval_hit_rate for r in readings) / n,
            belief_graph_coverage=sum(r.belief_graph_coverage for r in readings) / n,
            contradiction_resolution_rate=sum(r.contradiction_resolution_rate for r in readings) / n,
            friction_rate=sum(r.friction_rate for r in readings) / n,
        )
