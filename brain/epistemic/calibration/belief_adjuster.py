"""Belief confidence adjuster: feedback from calibration into Layer 5 beliefs.

Hard safety rails (all invariants, non-negotiable):
  1. Never change confidence by more than +/-0.05 per adjustment cycle
  2. Never adjust identity-tension beliefs
  3. Never mutate source memory confidence -- only BeliefStore belief_confidence
  4. Always record adjustment provenance and reason
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("jarvis.calibration.adjuster")

_JARVIS_DIR = Path(os.path.expanduser("~/.jarvis"))
_ADJUSTMENTS_LOG = _JARVIS_DIR / "confidence_adjustments.jsonl"
_MAX_FILE_MB = 10

MAX_DELTA = 0.05
OVERCONFIDENCE_THRESHOLD = 0.05


@dataclass
class ConfidenceAdjustment:
    belief_id: str
    old_confidence: float
    new_confidence: float
    delta: float
    adjustment_reason: str
    source_signal: str
    timestamp: float


class BeliefConfidenceAdjuster:
    """Adjusts belief_confidence based on calibration data. Dream/deep_learning cycles only."""

    def __init__(self) -> None:
        self._adjustments: deque[ConfidenceAdjustment] = deque(maxlen=200)
        self._total_adjustments: int = 0
        self._cycle_count: int = 0

    def run_adjustment_cycle(
        self,
        per_provenance_accuracy: dict[str, float],
        overconfidence_error: float | None,
    ) -> list[ConfidenceAdjustment]:
        """Run one adjustment cycle. Returns list of adjustments made."""
        self._cycle_count += 1
        adjustments: list[ConfidenceAdjustment] = []

        try:
            from epistemic.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine.get_instance()
            if engine is None:
                return adjustments

            from epistemic.belief_record import BeliefRecord
            store = engine._belief_store
            beliefs = store.get_active_beliefs()

            for belief in beliefs:
                if belief.resolution_state == "tension_held":
                    continue

                adj = self._compute_adjustment(belief, per_provenance_accuracy, overconfidence_error)
                if adj is not None:
                    store.update_belief_confidence(belief.belief_id, adj.new_confidence)
                    adjustments.append(adj)
                    self._persist_adjustment(adj)
                    self._emit_event(adj)

        except Exception as exc:
            logger.debug("Belief adjustment cycle failed: %s", exc)

        self._adjustments.extend(adjustments)
        self._total_adjustments += len(adjustments)

        if adjustments:
            logger.info("Adjusted %d belief confidences (cycle %d)", len(adjustments), self._cycle_count)
        return adjustments

    def _compute_adjustment(
        self,
        belief: object,
        per_provenance_accuracy: dict[str, float],
        overconfidence_error: float | None,
    ) -> ConfidenceAdjustment | None:
        prov = getattr(belief, "provenance", "unknown")
        old_conf = getattr(belief, "belief_confidence", 0.5)
        belief_id = getattr(belief, "belief_id", "")

        delta = 0.0
        reasons: list[str] = []

        if prov in per_provenance_accuracy:
            track_record = per_provenance_accuracy[prov]
            if old_conf > track_record + 0.1:
                delta -= min(MAX_DELTA, (old_conf - track_record) * 0.3)
                reasons.append("provenance_accuracy_scaling")
            elif old_conf < track_record - 0.1:
                delta += min(MAX_DELTA, (track_record - old_conf) * 0.2)
                reasons.append("provenance_accuracy_scaling")

        if overconfidence_error is not None and overconfidence_error > OVERCONFIDENCE_THRESHOLD:
            if old_conf > 0.7:
                dampening = min(MAX_DELTA, overconfidence_error * 0.1)
                delta -= dampening
                reasons.append("overconfidence_dampening")

        delta = max(-MAX_DELTA, min(MAX_DELTA, delta))

        if abs(delta) < 0.005:
            return None

        new_conf = max(0.0, min(1.0, old_conf + delta))

        return ConfidenceAdjustment(
            belief_id=belief_id,
            old_confidence=round(old_conf, 4),
            new_confidence=round(new_conf, 4),
            delta=round(delta, 4),
            adjustment_reason="+".join(reasons) if reasons else "combined",
            source_signal="per_provenance_accuracy" if per_provenance_accuracy else "overconfidence",
            timestamp=time.time(),
        )

    def _persist_adjustment(self, adj: ConfidenceAdjustment) -> None:
        try:
            _JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            self._maybe_rotate()
            entry = {
                "bid": adj.belief_id,
                "old": adj.old_confidence,
                "new": adj.new_confidence,
                "d": adj.delta,
                "reason": adj.adjustment_reason,
                "signal": adj.source_signal,
                "ts": round(adj.timestamp, 2),
            }
            with open(_ADJUSTMENTS_LOG, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception as exc:
            logger.debug("Adjustment persistence failed: %s", exc)

    @staticmethod
    def _maybe_rotate() -> None:
        try:
            if not _ADJUSTMENTS_LOG.exists():
                return
            size = _ADJUSTMENTS_LOG.stat().st_size
            if size < _MAX_FILE_MB * 1024 * 1024:
                return
            lines = _ADJUSTMENTS_LOG.read_text().splitlines(keepends=True)
            half = len(lines) // 2
            _ADJUSTMENTS_LOG.write_text("".join(lines[half:]))
            logger.info("confidence_adjustments.jsonl rotated: kept %d of %d lines", len(lines) - half, len(lines))
        except Exception as e:
            logger.warning("confidence_adjustments.jsonl rotation failed: %s", e)

    def _emit_event(self, adj: ConfidenceAdjustment) -> None:
        try:
            from consciousness.events import event_bus, CALIBRATION_CONFIDENCE_ADJUSTED
            event_bus.emit(
                CALIBRATION_CONFIDENCE_ADJUSTED,
                belief_id=adj.belief_id,
                old_confidence=adj.old_confidence,
                new_confidence=adj.new_confidence,
                delta=adj.delta,
                reason=adj.adjustment_reason,
            )
        except Exception:
            pass

    def get_stats(self) -> dict:
        return {
            "total_adjustments": self._total_adjustments,
            "cycle_count": self._cycle_count,
            "recent_adjustments": [
                {
                    "belief_id": a.belief_id,
                    "delta": a.delta,
                    "reason": a.adjustment_reason,
                }
                for a in self._adjustments[-10:]
            ],
        }
