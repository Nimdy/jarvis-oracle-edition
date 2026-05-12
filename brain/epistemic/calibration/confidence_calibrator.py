"""Confidence calibrator: Brier score, ECE, over/underconfidence, per-provenance accuracy.

Tracks whether Jarvis's stated confidence matches reality. Overconfidence
(hallucination tendency) and underconfidence (timid behavior) are tracked
separately because they imply different pathologies.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("jarvis.calibration.confidence")

_JARVIS_DIR = Path(os.path.expanduser("~/.jarvis"))
_OUTCOMES_LOG = _JARVIS_DIR / "confidence_outcomes.jsonl"
_MIN_OUTCOMES = 20
_MAX_OUTCOMES = 500
_ECE_BINS = 5
_MAX_FILE_MB = 10


_MIN_ROUTE_OUTCOMES = 10


@dataclass
class ConfidenceOutcome:
    belief_id: str
    predicted_confidence: float
    actual_correct: bool
    provenance: str
    timestamp: float
    route_class: str = ""


class ConfidenceCalibrator:
    """Measures calibration between predicted confidence and actual correctness."""

    def __init__(self) -> None:
        self._outcomes: deque[ConfidenceOutcome] = deque(maxlen=_MAX_OUTCOMES)
        self._rehydrate()

    def record_outcome(self, belief_id: str, confidence: float,
                       correct: bool, provenance: str,
                       route_class: str = "") -> None:
        outcome = ConfidenceOutcome(
            belief_id=belief_id,
            predicted_confidence=max(0.0, min(1.0, confidence)),
            actual_correct=correct,
            provenance=provenance,
            timestamp=time.time(),
            route_class=route_class,
        )
        self._outcomes.append(outcome)
        self._persist(outcome)

    def get_brier_score(self) -> float | None:
        """Mean squared error between confidence and binary outcome. Lower = better."""
        if len(self._outcomes) < _MIN_OUTCOMES:
            return None
        total = 0.0
        for o in self._outcomes:
            actual = 1.0 if o.actual_correct else 0.0
            total += (o.predicted_confidence - actual) ** 2
        return round(total / len(self._outcomes), 4)

    def get_ece(self, n_bins: int = _ECE_BINS) -> float | None:
        """Expected Calibration Error: weighted average of |accuracy - confidence| per bin."""
        if len(self._outcomes) < _MIN_OUTCOMES:
            return None

        bins: list[list[ConfidenceOutcome]] = [[] for _ in range(n_bins)]
        for o in self._outcomes:
            idx = min(int(o.predicted_confidence * n_bins), n_bins - 1)
            bins[idx].append(o)

        ece = 0.0
        total = len(self._outcomes)
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(o.predicted_confidence for o in bucket) / len(bucket)
            avg_acc = sum(1.0 for o in bucket if o.actual_correct) / len(bucket)
            ece += (len(bucket) / total) * abs(avg_acc - avg_conf)

        return round(ece, 4)

    def get_overconfidence_error(self) -> float | None:
        """Mean of max(0, confidence - accuracy) per bin. High = hallucination risk."""
        return self._directional_error(over=True)

    def get_underconfidence_error(self) -> float | None:
        """Mean of max(0, accuracy - confidence) per bin. High = timid behavior."""
        return self._directional_error(over=False)

    def _directional_error(self, over: bool) -> float | None:
        if len(self._outcomes) < _MIN_OUTCOMES:
            return None

        bins: list[list[ConfidenceOutcome]] = [[] for _ in range(_ECE_BINS)]
        for o in self._outcomes:
            idx = min(int(o.predicted_confidence * _ECE_BINS), _ECE_BINS - 1)
            bins[idx].append(o)

        errors: list[float] = []
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(o.predicted_confidence for o in bucket) / len(bucket)
            avg_acc = sum(1.0 for o in bucket if o.actual_correct) / len(bucket)
            if over:
                errors.append(max(0.0, avg_conf - avg_acc))
            else:
                errors.append(max(0.0, avg_acc - avg_conf))

        if not errors:
            return None
        return round(sum(errors) / len(errors), 4)

    def get_per_provenance_accuracy(self) -> dict[str, float]:
        """Returns accuracy rate per provenance type."""
        by_prov: dict[str, list[bool]] = {}
        for o in self._outcomes:
            by_prov.setdefault(o.provenance, []).append(o.actual_correct)

        return {
            prov: round(sum(1 for c in vals if c) / len(vals), 4)
            for prov, vals in by_prov.items()
            if len(vals) >= 3
        }

    def get_calibration_curve(self) -> dict[str, dict[str, float]]:
        """Returns per-bucket {avg_confidence, avg_accuracy, count}."""
        bins: list[list[ConfidenceOutcome]] = [[] for _ in range(_ECE_BINS)]
        for o in self._outcomes:
            idx = min(int(o.predicted_confidence * _ECE_BINS), _ECE_BINS - 1)
            bins[idx].append(o)

        curve: dict[str, dict[str, float]] = {}
        for i, bucket in enumerate(bins):
            lo = round(i / _ECE_BINS, 2)
            hi = round((i + 1) / _ECE_BINS, 2)
            label = f"{lo:.1f}-{hi:.1f}"
            if not bucket:
                curve[label] = {"avg_confidence": 0.0, "avg_accuracy": 0.0, "count": 0}
            else:
                curve[label] = {
                    "avg_confidence": round(sum(o.predicted_confidence for o in bucket) / len(bucket), 4),
                    "avg_accuracy": round(sum(1.0 for o in bucket if o.actual_correct) / len(bucket), 4),
                    "count": len(bucket),
                }
        return curve

    def get_route_brier_scores(self) -> dict[str, float]:
        """Brier score per route class. Only includes routes with >= _MIN_ROUTE_OUTCOMES."""
        by_route: dict[str, list[ConfidenceOutcome]] = {}
        for o in self._outcomes:
            if o.route_class:
                by_route.setdefault(o.route_class, []).append(o)

        result: dict[str, float] = {}
        for route, outcomes in by_route.items():
            if len(outcomes) < _MIN_ROUTE_OUTCOMES:
                continue
            total = sum((o.predicted_confidence - (1.0 if o.actual_correct else 0.0)) ** 2
                        for o in outcomes)
            result[route] = round(total / len(outcomes), 4)
        return result

    def get_worst_route_brier(self) -> tuple[float | None, int]:
        """Return (worst_brier, sample_count) for the route with highest Brier score.

        Only considers routes with >= _MIN_ROUTE_OUTCOMES samples.
        Returns (None, 0) if no route qualifies.
        """
        by_route: dict[str, list[ConfidenceOutcome]] = {}
        for o in self._outcomes:
            if o.route_class:
                by_route.setdefault(o.route_class, []).append(o)

        worst_brier: float | None = None
        worst_count = 0
        for route, outcomes in by_route.items():
            if len(outcomes) < _MIN_ROUTE_OUTCOMES:
                continue
            total = sum((o.predicted_confidence - (1.0 if o.actual_correct else 0.0)) ** 2
                        for o in outcomes)
            brier = total / len(outcomes)
            if worst_brier is None or brier > worst_brier:
                worst_brier = brier
                worst_count = len(outcomes)
        return (round(worst_brier, 4) if worst_brier is not None else None, worst_count)

    def get_route_overconfidence(self, route_class: str) -> float | None:
        """Overconfidence error for a specific route. None if insufficient data."""
        route_outcomes = [o for o in self._outcomes if o.route_class == route_class]
        if len(route_outcomes) < _MIN_ROUTE_OUTCOMES:
            return None

        bins: list[list[ConfidenceOutcome]] = [[] for _ in range(_ECE_BINS)]
        for o in route_outcomes:
            idx = min(int(o.predicted_confidence * _ECE_BINS), _ECE_BINS - 1)
            bins[idx].append(o)

        errors: list[float] = []
        for bucket in bins:
            if not bucket:
                continue
            avg_conf = sum(o.predicted_confidence for o in bucket) / len(bucket)
            avg_acc = sum(1.0 for o in bucket if o.actual_correct) / len(bucket)
            errors.append(max(0.0, avg_conf - avg_acc))
        return round(sum(errors) / len(errors), 4) if errors else None

    def get_route_sample_counts(self) -> dict[str, int]:
        """Number of outcomes per route class."""
        counts: dict[str, int] = {}
        for o in self._outcomes:
            if o.route_class:
                counts[o.route_class] = counts.get(o.route_class, 0) + 1
        return counts

    def get_provenance_sample_counts(self) -> dict[str, int]:
        """Number of outcomes per provenance type."""
        counts: dict[str, int] = {}
        for o in self._outcomes:
            if o.provenance:
                counts[o.provenance] = counts.get(o.provenance, 0) + 1
        return counts

    @property
    def outcome_count(self) -> int:
        return len(self._outcomes)

    def _persist(self, outcome: ConfidenceOutcome) -> None:
        try:
            _JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            self._maybe_rotate()
            entry = {
                "bid": outcome.belief_id,
                "conf": outcome.predicted_confidence,
                "ok": outcome.actual_correct,
                "prov": outcome.provenance,
                "ts": round(outcome.timestamp, 2),
                "rc": outcome.route_class,
            }
            with open(_OUTCOMES_LOG, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception as exc:
            logger.debug("Confidence outcome persistence failed: %s", exc)

    @staticmethod
    def _maybe_rotate() -> None:
        try:
            if not _OUTCOMES_LOG.exists():
                return
            size = _OUTCOMES_LOG.stat().st_size
            if size < _MAX_FILE_MB * 1024 * 1024:
                return
            lines = _OUTCOMES_LOG.read_text().splitlines(keepends=True)
            half = len(lines) // 2
            _OUTCOMES_LOG.write_text("".join(lines[half:]))
            logger.info("confidence_outcomes.jsonl rotated: kept %d of %d lines", len(lines) - half, len(lines))
        except Exception as e:
            logger.warning("confidence_outcomes.jsonl rotation failed: %s", e)

    def _rehydrate(self) -> None:
        try:
            if not _OUTCOMES_LOG.exists():
                return
            lines = _OUTCOMES_LOG.read_text().splitlines()
            for line in lines[-_MAX_OUTCOMES:]:
                try:
                    d = json.loads(line)
                    self._outcomes.append(ConfidenceOutcome(
                        belief_id=d["bid"],
                        predicted_confidence=d["conf"],
                        actual_correct=d["ok"],
                        provenance=d["prov"],
                        timestamp=d["ts"],
                        route_class=d.get("rc", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
            if self._outcomes:
                logger.info("Rehydrated %d confidence outcomes", len(self._outcomes))
        except Exception as exc:
            logger.debug("Confidence rehydration failed: %s", exc)
