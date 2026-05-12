"""Cognitive Gap Detector -- monitors performance across cognitive dimensions
and emits CognitiveGap events when sustained underperformance is detected.

Governance policy prevents thrash from detector noise:
  1. Sustained gap requirement (N consecutive windows below threshold)
  2. Rate limit (max 1 new focus per time window)
  3. Sunset clause (NNs pruned if no impact within deadline)
  4. Per-dimension cooldown (no re-trigger unless severity worsens by Y%)
  5. Noisy label smoothing (EMA, confidence intervals)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time as _time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GAP_DETECTOR_STATE_PATH = Path("~/.jarvis/gap_detector_state.json").expanduser()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_SIZE_S = 300.0           # 5-minute rolling windows
SUSTAINED_WINDOWS = 5           # must be below threshold for N consecutive windows
MIN_TREND_SLOPE = -0.05         # alternative trigger: steep downward trend
GAP_COOLDOWN_S = 3600.0         # 60 min cooldown per dimension
SEVERITY_WORSEN_PCT = 0.25      # must worsen by 25% to re-trigger within cooldown
MAX_NEW_FOCUS_PER_WINDOW = 1    # rate limit: at most 1 new focus per 30 min
NEW_FOCUS_WINDOW_S = 1800.0     # 30-minute window for rate limiting
EMA_ALPHA = 0.3                 # smoothing factor for noisy labels
CONFIDENCE_MARGIN = 0.05        # NN must outperform by this margin


# ---------------------------------------------------------------------------
# Cognitive dimensions
# ---------------------------------------------------------------------------

COGNITIVE_DIMENSIONS: dict[str, dict[str, Any]] = {
    "response_quality": {
        "metrics": ["follow_up_rate", "sentiment_score", "barge_in_rate"],
        "threshold": 0.4,
        "weight": 1.0,
    },
    "memory_recall": {
        "metrics": ["recall_precision", "association_strength"],
        "threshold": 0.35,
        "weight": 0.8,
    },
    "mood_prediction": {
        "metrics": ["prediction_accuracy", "mood_stability"],
        "threshold": 0.4,
        "weight": 0.7,
    },
    "context_awareness": {
        "metrics": ["engagement_tracking", "context_switches"],
        "threshold": 0.3,
        "weight": 0.6,
    },
    "self_improvement": {
        "metrics": ["patch_success_rate", "lint_pass_rate"],
        "threshold": 0.3,
        "weight": 0.5,
    },
    "trait_consistency": {
        "metrics": ["conflict_rate", "stability_score"],
        "threshold": 0.5,
        "weight": 0.6,
    },
    # Perceptual dimensions (drive Tier-1 distillation training)
    "emotion_accuracy": {
        "metrics": ["emotion_confidence_avg", "emotion_consistency"],
        "threshold": 0.4,
        "weight": 0.7,
    },
    "recognition_confidence": {
        "metrics": ["speaker_id_confidence", "face_id_confidence", "identity_conflicts"],
        "threshold": 0.35,
        "weight": 0.8,
    },
    "perception_latency": {
        "metrics": ["emotion_inference_ms", "speaker_id_inference_ms"],
        "threshold": 0.5,
        "weight": 0.6,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CognitiveGap:
    dimension: str
    severity: float                     # 0-1 (higher = worse)
    evidence: list[dict[str, Any]]
    sustained_windows: int
    trend_slope: float                  # negative = declining
    suggested_focus: str
    proposed_input_features: list[str]
    proposed_output: str
    timestamp: float = field(default_factory=_time.time)


@dataclass
class DimensionState:
    """Rolling state for a single cognitive dimension."""
    scores: deque = field(default_factory=lambda: deque(maxlen=20))
    ema: float = 0.5
    consecutive_below: int = 0
    last_score: float = 0.5
    last_updated: float = 0.0


# ---------------------------------------------------------------------------
# Gap Detector
# ---------------------------------------------------------------------------


class CognitiveGapDetector:
    """Monitors cognitive dimensions and emits CognitiveGap events."""

    def __init__(self) -> None:
        self._dimensions: dict[str, DimensionState] = {
            dim: DimensionState() for dim in COGNITIVE_DIMENSIONS
        }
        self._last_triggered: dict[str, float] = {}
        self._last_severity: dict[str, float] = {}
        self._last_focus_time: float = 0.0
        self._total_gaps_emitted: int = 0
        self._gaps_history: deque[CognitiveGap] = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Feed metrics
    # ------------------------------------------------------------------

    def update_metric(self, dimension: str, metric_name: str, value: float) -> None:
        """Feed a metric value into the detector."""
        state = self._dimensions.get(dimension)
        if state is None:
            return
        # EMA smoothing for noisy labels
        state.ema = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * state.ema
        state.scores.append({"t": _time.time(), "metric": metric_name, "raw": value, "ema": state.ema})
        state.last_score = state.ema
        state.last_updated = _time.time()

    def update_dimension_score(self, dimension: str, score: float) -> None:
        """Update the aggregate score for a dimension (0-1)."""
        state = self._dimensions.get(dimension)
        if state is None:
            return
        state.ema = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * state.ema
        state.scores.append({"t": _time.time(), "aggregate": True, "raw": score, "ema": state.ema})
        state.last_score = state.ema
        state.last_updated = _time.time()

    # ------------------------------------------------------------------
    # Detect gaps (called from consciousness tick)
    # ------------------------------------------------------------------

    def detect_gaps(self) -> list[CognitiveGap]:
        """Analyze all dimensions and return any newly detected gaps.

        Enforces governance: sustained windows, rate limiting, cooldowns.
        """
        now = _time.time()
        gaps: list[CognitiveGap] = []

        for dim, config in COGNITIVE_DIMENSIONS.items():
            state = self._dimensions[dim]
            threshold = config["threshold"]

            # Skip dimensions with no data
            if not state.scores or now - state.last_updated > WINDOW_SIZE_S * 3:
                state.consecutive_below = 0
                continue

            # Check if below threshold
            if state.ema < threshold:
                state.consecutive_below += 1
            else:
                state.consecutive_below = 0
                continue

            # Compute trend slope from recent scores
            slope = self._compute_trend(state)

            # Check sustained gap or steep decline
            sustained = state.consecutive_below >= SUSTAINED_WINDOWS
            steep_decline = slope < MIN_TREND_SLOPE

            if not sustained and not steep_decline:
                continue

            severity = max(0.0, min(1.0, (threshold - state.ema) / max(threshold, 0.01)))

            # Per-dimension cooldown
            if not self._check_cooldown(dim, severity, now):
                continue

            # Rate limit
            if now - self._last_focus_time < NEW_FOCUS_WINDOW_S:
                continue

            # Emit gap
            gap = CognitiveGap(
                dimension=dim,
                severity=severity,
                evidence=list(state.scores)[-5:],
                sustained_windows=state.consecutive_below,
                trend_slope=slope,
                suggested_focus=f"{dim}_improvement",
                proposed_input_features=config["metrics"],
                proposed_output=f"{dim}_score",
            )

            gaps.append(gap)
            self._gaps_history.append(gap)
            self._total_gaps_emitted += 1
            self._last_triggered[dim] = now
            self._last_severity[dim] = severity
            self._last_focus_time = now

            logger.info(
                "Cognitive gap detected: %s (severity=%.2f, sustained=%d, slope=%.3f)",
                dim, severity, state.consecutive_below, slope,
            )

            # Only one gap per detection cycle (rate limit)
            break

        return gaps

    # ------------------------------------------------------------------
    # Governance helpers
    # ------------------------------------------------------------------

    def _check_cooldown(self, dimension: str, severity: float, now: float) -> bool:
        """Check per-dimension cooldown. Returns True if trigger is allowed."""
        last_time = self._last_triggered.get(dimension, 0.0)
        if now - last_time < GAP_COOLDOWN_S:
            last_sev = self._last_severity.get(dimension, 0.0)
            if severity < last_sev * (1.0 + SEVERITY_WORSEN_PCT):
                return False
        return True

    def _compute_trend(self, state: DimensionState) -> float:
        """Compute linear trend slope over recent scores."""
        if len(state.scores) < 3:
            return 0.0
        recent = list(state.scores)[-10:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(s.get("ema", s.get("raw", 0.5)) for s in recent) / n
        num = 0.0
        den = 0.0
        for i, s in enumerate(recent):
            y = s.get("ema", s.get("raw", 0.5))
            num += (i - x_mean) * (y - y_mean)
            den += (i - x_mean) ** 2
        return num / den if den > 0 else 0.0

    # ------------------------------------------------------------------
    # State / dashboard
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        dim_states = {}
        for dim, state in self._dimensions.items():
            config = COGNITIVE_DIMENSIONS[dim]
            dim_states[dim] = {
                "ema": round(state.ema, 4),
                "threshold": config["threshold"],
                "consecutive_below": state.consecutive_below,
                "data_points": len(state.scores),
                "last_updated": state.last_updated,
            }
        return {
            "dimensions": dim_states,
            "total_gaps_emitted": self._total_gaps_emitted,
            "recent_gaps": [
                {
                    "dimension": g.dimension,
                    "severity": round(g.severity, 3),
                    "sustained_windows": g.sustained_windows,
                    "timestamp": g.timestamp,
                }
                for g in list(self._gaps_history)[-10:]
            ],
            "cooldowns": {
                dim: round(max(0, GAP_COOLDOWN_S - (_time.time() - t)), 0)
                for dim, t in self._last_triggered.items()
            },
        }

    def get_dimension_scores(self) -> dict[str, float]:
        """Return current EMA score per dimension (for dashboard heatmap)."""
        return {dim: round(state.ema, 4) for dim, state in self._dimensions.items()}

    def save_state(self) -> None:
        """Persist dimension EMAs and consecutive_below counts to disk."""
        data: dict[str, Any] = {}
        for dim, state in self._dimensions.items():
            data[dim] = {
                "ema": state.ema,
                "consecutive_below": state.consecutive_below,
                "last_updated": state.last_updated,
                "data_points": len(state.scores),
            }
        data["_meta"] = {
            "total_gaps_emitted": self._total_gaps_emitted,
            "saved_at": _time.time(),
        }
        try:
            GAP_DETECTOR_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=GAP_DETECTOR_STATE_PATH.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                os.replace(tmp, GAP_DETECTOR_STATE_PATH)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            logger.debug("GapDetector save_state failed", exc_info=True)

    def load_state(self) -> int:
        """Restore dimension EMAs from disk. Returns count of dimensions restored."""
        if not GAP_DETECTOR_STATE_PATH.exists():
            return 0
        try:
            data = json.loads(GAP_DETECTOR_STATE_PATH.read_text())
        except Exception:
            logger.warning("GapDetector load_state: corrupt file, skipping")
            return 0

        restored = 0
        for dim, d in data.items():
            if dim == "_meta":
                self._total_gaps_emitted = d.get("total_gaps_emitted", self._total_gaps_emitted)
                continue
            state = self._dimensions.get(dim)
            if state is None:
                continue
            state.ema = d.get("ema", 0.5)
            state.consecutive_below = d.get("consecutive_below", 0)
            state.last_updated = d.get("last_updated", 0.0)
            restored += 1

        if restored:
            logger.info("GapDetector restored state for %d dimensions", restored)
        return restored
