"""Prediction validator: typed prediction tracking and validation.

Predictions are typed structures (not strings) with deterministic validation
rules. Each prediction type maps to a specific event check.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("jarvis.calibration.prediction")

_MAX_PENDING = 50
_MAX_COMPLETED = 200


@dataclass
class PredictionRecord:
    prediction_id: str
    prediction_type: str
    subject: str
    expected_outcome: str
    confidence: float
    validation_window_s: float
    human_readable: str
    registered_at: float
    validated_at: float | None = None
    validation_result: bool | None = None


_PREDICTION_WINDOWS: dict[str, float] = {
    "interaction_soon": 60.0,
    "speech_imminent": 30.0,
    "emotional_tone_persists": 120.0,
}


class PredictionValidator:
    """Registers and validates typed epistemic predictions."""

    def __init__(self) -> None:
        self._pending: list[PredictionRecord] = []
        self._completed: deque[PredictionRecord] = deque(maxlen=_MAX_COMPLETED)
        self._events_seen: dict[str, float] = {}
        self._total_validated: int = 0

        self._subscribe()

    def _subscribe(self) -> None:
        try:
            from consciousness.events import (
                event_bus, PERCEPTION_WAKE_WORD, PERCEPTION_TRANSCRIPTION,
                PERCEPTION_USER_EMOTION,
            )
            event_bus.on(PERCEPTION_WAKE_WORD, self._on_wake_word)
            event_bus.on(PERCEPTION_TRANSCRIPTION, self._on_transcription)
            event_bus.on(PERCEPTION_USER_EMOTION, self._on_emotion)
        except Exception as exc:
            logger.debug("PredictionValidator event subscription failed: %s", exc)

    def _on_wake_word(self, **kwargs: Any) -> None:
        self._events_seen["wake_word"] = time.time()

    def _on_transcription(self, **kwargs: Any) -> None:
        self._events_seen["transcription"] = time.time()

    def _on_emotion(self, **kwargs: Any) -> None:
        self._events_seen["emotion"] = time.time()

    def register(self, prediction: PredictionRecord) -> None:
        if len(self._pending) >= _MAX_PENDING:
            self._pending.pop(0)
        self._pending.append(prediction)

    def register_from_strings(self, predictions: list[str], confidence: float) -> None:
        """Convenience: convert string predictions from EpistemicEngine into typed records."""
        now = time.time()
        for pred_str in predictions:
            pred_str_lower = pred_str.lower()
            if "user likely to interact" in pred_str_lower or "interact soon" in pred_str_lower:
                self.register(PredictionRecord(
                    prediction_id=f"pred_{now:.0f}_interaction",
                    prediction_type="interaction_soon",
                    subject="user",
                    expected_outcome="wake_word_within_60s",
                    confidence=confidence,
                    validation_window_s=_PREDICTION_WINDOWS["interaction_soon"],
                    human_readable=pred_str,
                    registered_at=now,
                ))
            elif "expect incoming speech" in pred_str_lower or "speech within" in pred_str_lower:
                self.register(PredictionRecord(
                    prediction_id=f"pred_{now:.0f}_speech",
                    prediction_type="speech_imminent",
                    subject="user",
                    expected_outcome="transcription_within_30s",
                    confidence=confidence,
                    validation_window_s=_PREDICTION_WINDOWS["speech_imminent"],
                    human_readable=pred_str,
                    registered_at=now,
                ))
            elif "emotional tone" in pred_str_lower:
                self.register(PredictionRecord(
                    prediction_id=f"pred_{now:.0f}_emotion",
                    prediction_type="emotional_tone_persists",
                    subject="user",
                    expected_outcome="emotion_event_within_120s",
                    confidence=confidence,
                    validation_window_s=_PREDICTION_WINDOWS["emotional_tone_persists"],
                    human_readable=pred_str,
                    registered_at=now,
                ))

    def tick(self) -> list[PredictionRecord]:
        """Check pending predictions against observed events. Returns newly validated predictions."""
        now = time.time()
        validated: list[PredictionRecord] = []
        still_pending: list[PredictionRecord] = []

        for pred in self._pending:
            deadline = pred.registered_at + pred.validation_window_s
            if now > deadline:
                result = self._check_outcome(pred)
                pred.validated_at = now
                pred.validation_result = result
                self._completed.append(pred)
                self._total_validated += 1
                validated.append(pred)
                self._emit_event(pred)
            else:
                still_pending.append(pred)

        self._pending = still_pending
        return validated

    def _check_outcome(self, pred: PredictionRecord) -> bool:
        if pred.prediction_type == "interaction_soon":
            ts = self._events_seen.get("wake_word", 0.0)
            return ts > pred.registered_at
        elif pred.prediction_type == "speech_imminent":
            ts = self._events_seen.get("transcription", 0.0)
            return ts > pred.registered_at
        elif pred.prediction_type == "emotional_tone_persists":
            ts = self._events_seen.get("emotion", 0.0)
            return ts > pred.registered_at
        return False

    _MIN_SAMPLE_FLOOR = 10

    def get_accuracy(self) -> float | None:
        if len(self._completed) < self._MIN_SAMPLE_FLOOR:
            return None
        correct = sum(1 for p in self._completed if p.validation_result)
        return round(correct / len(self._completed), 4)

    def get_stats(self) -> dict:
        return {
            "pending": len(self._pending),
            "completed": len(self._completed),
            "total_validated": self._total_validated,
            "accuracy": self.get_accuracy(),
        }

    def _emit_event(self, pred: PredictionRecord) -> None:
        try:
            from consciousness.events import event_bus, PREDICTION_VALIDATED
            event_bus.emit(
                PREDICTION_VALIDATED,
                prediction_type=pred.prediction_type,
                result=pred.validation_result,
                confidence=pred.confidence,
            )
        except Exception:
            pass
