"""Calibration history: rolling window of snapshots for trend analysis.

Provides per-domain trend, variance, and slope calculations over
configurable time windows.  Supports rehydration from calibration_truth.jsonl
so that cross-restart drift detection is possible.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

from epistemic.calibration.signal_collector import CalibrationSnapshot

logger = logging.getLogger(__name__)

HISTORY_MAXLEN = 200  # ~6.7 hours at 120s tick interval

_DOMAIN_FIELDS: dict[str, list[str]] = {
    "retrieval": ["reference_match_rate", "ranker_success_rate", "lift"],
    "autonomy": ["improvement_rate", "overall_win_rate"],
    "salience": ["wasted_rate", "useful_rate", "weight_error", "decay_error"],
    "reasoning": ["coherence", "consistency", "depth"],
    "epistemic": ["contradiction_debt", "near_miss_rate"],
    "skill": ["verified_skill_count", "honesty_failures"],
    "confidence": ["brier_score", "ece"],
    "prediction": ["prediction_accuracy"],
}


def _domain_representative_value(snap: CalibrationSnapshot, domain: str) -> float | None:
    """Extract a single representative score for a domain from a snapshot.

    Returns None if insufficient data exists for the domain.
    """
    if domain == "retrieval":
        vals = [v for v in [snap.reference_match_rate, snap.ranker_success_rate] if v is not None]
        return sum(vals) / len(vals) if vals else None
    elif domain == "autonomy":
        vals = [v for v in [snap.improvement_rate, snap.overall_win_rate] if v is not None]
        return sum(vals) / len(vals) if vals else None
    elif domain == "salience":
        vals = [v for v in [snap.wasted_rate, snap.weight_error, snap.decay_error] if v is not None]
        if not vals:
            return None
        return 1.0 - sum(vals) / len(vals)
    elif domain == "reasoning":
        vals = [v for v in [snap.coherence, snap.consistency, snap.depth] if v is not None]
        return sum(vals) / len(vals) if vals else None
    elif domain == "epistemic":
        return 1.0 - snap.contradiction_debt
    elif domain == "skill":
        return None  # needs domain calibrator's formula
    elif domain == "confidence":
        if snap.brier_score is not None:
            return 1.0 - snap.brier_score
        return None
    elif domain == "prediction":
        return snap.prediction_accuracy
    return None


class CalibrationHistory:
    """Rolling window of CalibrationSnapshots with trend analysis."""

    def __init__(self, maxlen: int = HISTORY_MAXLEN) -> None:
        self._snapshots: deque[CalibrationSnapshot] = deque(maxlen=maxlen)

    def record(self, snapshot: CalibrationSnapshot) -> None:
        self._snapshots.append(snapshot)

    @property
    def count(self) -> int:
        return len(self._snapshots)

    def get_domain_trend(self, domain: str, window: int = 20) -> list[float]:
        """Return recent domain scores as a list (oldest first)."""
        result: list[float] = []
        recent = list(self._snapshots)[-window:]
        for snap in recent:
            val = _domain_representative_value(snap, domain)
            if val is not None:
                result.append(val)
        return result

    def get_domain_variance(self, domain: str, window: int = 20) -> float:
        """Compute variance of domain scores over the window."""
        trend = self.get_domain_trend(domain, window)
        if len(trend) < 2:
            return 0.0
        mean = sum(trend) / len(trend)
        return sum((v - mean) ** 2 for v in trend) / len(trend)

    def get_domain_slope(self, domain: str, window: int = 20) -> float:
        """Compute linear slope of domain scores (positive = improving).

        Uses least-squares fit: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        """
        trend = self.get_domain_trend(domain, window)
        n = len(trend)
        if n < 3:
            return 0.0

        sum_x = sum(range(n))
        sum_y = sum(trend)
        sum_xy = sum(i * v for i, v in enumerate(trend))
        sum_x2 = sum(i * i for i in range(n))

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    def get_latest(self) -> CalibrationSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def get_recent(self, n: int = 10) -> list[CalibrationSnapshot]:
        return list(self._snapshots)[-n:]

    def rehydrate_from_log(self, log_path: Path, max_entries: int = 200) -> int:
        """Reload calibration snapshots from the JSONL truth log.

        Reads the most recent ``max_entries`` valid lines from
        ``calibration_truth.jsonl`` and populates the in-memory deque so
        that trend, variance, and slope analysis has continuity across
        restarts.  Malformed lines are silently skipped.

        Returns the number of snapshots loaded.
        """
        if not log_path.exists():
            return 0

        loaded = 0
        rejected = 0
        entries: list[CalibrationSnapshot] = []

        try:
            lines = log_path.read_text().splitlines()
        except Exception:
            logger.warning("CalibrationHistory rehydrate: failed to read %s", log_path)
            return 0

        for line in lines[-max_entries * 2:]:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                snap = CalibrationSnapshot(timestamp=d.get("ts", 0.0))
                snap.contradiction_debt = d.get("debt", 0.0)
                domains = d.get("domains", {})
                for field_name, val in domains.items():
                    if hasattr(snap, field_name) and val is not None:
                        try:
                            setattr(snap, field_name, float(val))
                        except (TypeError, ValueError):
                            pass
                entries.append(snap)
            except (json.JSONDecodeError, TypeError, KeyError):
                rejected += 1

        entries.sort(key=lambda s: s.timestamp)
        entries = entries[-max_entries:]

        for snap in entries:
            self._snapshots.append(snap)
            loaded += 1

        oldest_ts = entries[0].timestamp if entries else 0.0
        newest_ts = entries[-1].timestamp if entries else 0.0
        logger.info(
            "CalibrationHistory rehydrated: loaded=%d rejected=%d "
            "oldest=%.0f newest=%.0f",
            loaded, rejected, oldest_ts, newest_ts,
        )
        return loaded
