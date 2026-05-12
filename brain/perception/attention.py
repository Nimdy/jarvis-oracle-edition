"""AttentionCore — multi-modal perception fusion into a unified attention state.

Subscribes to raw perception events and maintains a structured
AttentionState snapshot.  Emits ATTENTION_UPDATE at a throttled rate
(2-5 Hz) plus immediate updates on significant changes.

Design: state estimator (not a router).  Composable and testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from consciousness.events import (
    event_bus,
    PERCEPTION_USER_PRESENT_STABLE,
    PERCEPTION_USER_ATTENTION,
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_USER_EMOTION,
    PERCEPTION_POSE_DETECTED,
    PERCEPTION_WAKE_WORD,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_BARGE_IN,
)

logger = logging.getLogger(__name__)

ATTENTION_UPDATE = "attention:update"
ATTENTION_SIGNIFICANT_CHANGE = "attention:significant_change"


@dataclass
class AttentionState:
    """Unified perception context, updated continuously."""

    # Presence
    person_present: bool = False
    presence_confidence: float = 0.0

    # Identity
    speaker_identity: str = "unknown"
    speaker_confidence: float = 0.0

    # Affect
    user_emotion: str = "neutral"
    emotion_confidence: float = 0.0

    # Body language
    gesture: str = "neutral"
    gesture_confidence: float = 0.0

    # Environment
    ambient_state: str = "silence"
    ambient_confidence: float = 0.0

    # Engagement (computed)
    engagement_level: float = 0.0
    focus_target: str = ""
    reasons: list[str] = field(default_factory=list)

    # Interruption
    interruption_allowed: bool = True

    # Timing
    last_interaction_time: float = 0.0
    last_update_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttentionCore:
    """Fuses multi-modal perception events into a single AttentionState."""

    THROTTLE_INTERVAL = 0.25  # max emit rate: 4 Hz
    _BASE_DECAY_RATES = {"recent": 10.0, "active": 60.0, "lingering": 300.0}

    def __init__(self) -> None:
        self._state = AttentionState()
        self._last_emit_time: float = 0.0
        self._cleanups: list[Callable[[], None]] = []
        self._dirty = False
        self._decay_rate_multiplier: float = 1.0
        self._interruption_threshold: float = 0.0

    def start(self) -> None:
        on = event_bus.on
        self._cleanups = [
            on(PERCEPTION_USER_PRESENT_STABLE, self._on_presence),
            on(PERCEPTION_SPEAKER_IDENTIFIED, self._on_speaker),
            on(PERCEPTION_USER_EMOTION, self._on_emotion),
            on(PERCEPTION_POSE_DETECTED, self._on_pose),
            on(PERCEPTION_WAKE_WORD, self._on_wake_word),
            on(PERCEPTION_TRANSCRIPTION, self._on_transcription),
            on(PERCEPTION_BARGE_IN, self._on_barge_in),
        ]
        logger.info("AttentionCore started")

    def stop(self) -> None:
        for cleanup in self._cleanups:
            cleanup()
        self._cleanups.clear()

    @property
    def state(self) -> AttentionState:
        return self._state

    def get_state(self) -> dict[str, Any]:
        return self._state.to_dict()

    # -- Event handlers ----------------------------------------------------

    def _on_presence(self, present: bool, confidence: float = 1.0, **_) -> None:
        changed = self._state.person_present != present
        self._state.person_present = present
        self._state.presence_confidence = confidence
        self._recompute_engagement()
        if changed:
            self._emit_significant("presence_change")
        else:
            self._mark_dirty()

    def _on_speaker(self, name: str, confidence: float, is_known: bool = True, **_) -> None:
        old = self._state.speaker_identity
        self._state.speaker_identity = name if is_known else "unknown"
        self._state.speaker_confidence = confidence
        if old != self._state.speaker_identity:
            self._emit_significant("speaker_change")
        else:
            self._mark_dirty()

    _EMOTION_MIN_CONFIDENCE = 0.35

    def _on_emotion(self, emotion: str, confidence: float, trust: str = "medium", **_) -> None:
        _TRUST_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.3}
        weighted_confidence = confidence * _TRUST_WEIGHTS.get(trust, 0.5)

        if trust == "low" or weighted_confidence < self._EMOTION_MIN_CONFIDENCE:
            self._state.user_emotion = "neutral"
            self._state.emotion_confidence = 0.0
            self._mark_dirty()
            return

        old = self._state.user_emotion
        self._state.user_emotion = emotion
        self._state.emotion_confidence = weighted_confidence
        self._recompute_engagement()
        if old != emotion and emotion != "neutral":
            self._emit_significant("emotion_spike")
        else:
            self._mark_dirty()

    def _on_pose(self, gesture: str = "neutral", confidence: float = 0.0, **_) -> None:
        self._state.gesture = gesture
        self._state.gesture_confidence = confidence
        self._mark_dirty()

    def _on_ambient(self, classification: str = "", confidence: float = 0.0, **_) -> None:
        self._state.ambient_state = classification
        self._state.ambient_confidence = confidence
        self._mark_dirty()

    def _on_wake_word(self, **_) -> None:
        self._state.last_interaction_time = time.time()
        self._state.focus_target = "user"
        self._recompute_engagement()
        self._emit_significant("wake_word")

    def _on_transcription(self, text: str = "", **_) -> None:
        self._state.last_interaction_time = time.time()
        self._state.focus_target = "user"
        self._recompute_engagement()
        self._mark_dirty()

    def _on_barge_in(self, **_) -> None:
        self._state.last_interaction_time = time.time()
        self._state.interruption_allowed = True
        self._emit_significant("barge_in")

    # -- Engagement computation --------------------------------------------

    def _recompute_engagement(self) -> None:
        s = self._state
        level = 0.0
        reasons: list[str] = []

        if s.person_present:
            level += 0.3
            reasons.append("present")

        recency = time.time() - s.last_interaction_time if s.last_interaction_time else 999
        decay = self._decay_rate_multiplier
        if recency < self._BASE_DECAY_RATES["recent"] * (1.0 / decay):
            level += 0.4
            reasons.append("recent_interaction")
        elif recency < self._BASE_DECAY_RATES["active"] * (1.0 / decay):
            level += 0.2
            reasons.append("recent_activity")
        elif recency < self._BASE_DECAY_RATES["lingering"] * (1.0 / decay):
            level += 0.05

        if s.user_emotion not in ("neutral", ""):
            level += 0.15
            reasons.append(f"emotion:{s.user_emotion}")

        if s.gesture not in ("neutral", "none", ""):
            level += 0.1
            reasons.append(f"gesture:{s.gesture}")

        if s.speaker_identity != "unknown":
            level += 0.05
            reasons.append("known_speaker")

        new_level = min(1.0, level)
        old_level = s.engagement_level
        s.engagement_level = new_level
        s.reasons = reasons
        s.last_update_time = time.time()
        s.interruption_allowed = new_level >= self._interruption_threshold

        if abs(new_level - old_level) > 0.05:
            event_bus.emit(PERCEPTION_USER_ATTENTION, level=new_level)

    # -- Emit logic --------------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty = True
        now = time.time()
        if now - self._last_emit_time >= self.THROTTLE_INTERVAL:
            self._recompute_engagement()
            self._emit_update()

    def _emit_update(self) -> None:
        self._last_emit_time = time.time()
        self._dirty = False
        event_bus.emit(ATTENTION_UPDATE, state=self._state.to_dict())

    def _emit_significant(self, reason: str) -> None:
        self._recompute_engagement()
        self._last_emit_time = time.time()
        self._dirty = False
        event_bus.emit(ATTENTION_SIGNIFICANT_CHANGE, reason=reason, state=self._state.to_dict())
        event_bus.emit(ATTENTION_UPDATE, state=self._state.to_dict())

    def set_decay_rate(self, multiplier: float) -> None:
        """Policy NN control: higher multiplier = engagement decays faster."""
        self._decay_rate_multiplier = max(0.2, min(5.0, multiplier))

    def set_interruption_threshold(self, threshold: float) -> None:
        """Policy NN control: engagement must exceed this to allow interruption."""
        self._interruption_threshold = max(0.0, min(1.0, threshold))

    def tick(self) -> None:
        """Call periodically (e.g. from kernel tick) to flush throttled updates."""
        if self._dirty:
            self._recompute_engagement()
            self._emit_update()
