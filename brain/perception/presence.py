"""Presence tracker — hysteresis-based user presence detection."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from consciousness.events import (
    event_bus, PERCEPTION_USER_PRESENT, PERCEPTION_USER_PRESENT_STABLE,
    PERCEPTION_USER_ATTENTION,
)
from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData

logger = logging.getLogger(__name__)

PRESENCE_USER_ARRIVED = "presence:user_arrived"
_MIN_ABSENCE_FOR_GREETING_S = 600.0
# Pi PersonTracker already waits max_absent_s=5s, then sends one person_lost
# edge. Requiring 3 present=False events here meant departed never latched
# (lived: zero "Proactive arrival greeting" lines in brain.log since 2026-07-02).
_ABSENT_EVENTS_TO_DEPART = 1


class PresenceTracker:
    """Single canonical authority for user presence.

    Subscribes to raw PERCEPTION_USER_PRESENT events. The Pi already applies
    track-loss hysteresis before emitting person_lost, so one present=False
    is departure. Emits PERCEPTION_USER_PRESENT_STABLE on the edge.
    All downstream consumers should subscribe to the stable event.
    """

    def __init__(self, engine: ConsciousnessEngine) -> None:
        self._lock = threading.Lock()
        self._engine = engine
        self._is_present = False
        self._confidence = 0.0
        self._last_seen = 0.0
        self._last_departed = 0.0
        self._attention_level = 0.0
        self._consecutive_absent = 0
        self._divergence_since = 0.0
        self._last_divergence_log = 0.0
        self._cleanups: list[Callable[[], None]] = []

    def start(self) -> None:
        self._cleanups.append(
            event_bus.on(PERCEPTION_USER_PRESENT, self._on_presence)
        )
        self._cleanups.append(
            event_bus.on(PERCEPTION_USER_ATTENTION, self._on_attention)
        )

    def stop(self) -> None:
        for cleanup in self._cleanups:
            cleanup()
        self._cleanups.clear()

    def _on_presence(self, present: bool, confidence: float, **_) -> None:
        with self._lock:
            was_present = self._is_present

            if present:
                self._is_present = True
                self._confidence = confidence
                self._last_seen = time.time()
                self._consecutive_absent = 0
            else:
                self._consecutive_absent += 1
                if self._consecutive_absent >= _ABSENT_EVENTS_TO_DEPART:
                    self._is_present = False
                    self._confidence = confidence

            changed = was_present != self._is_present
            now_present = self._is_present
            now_confidence = self._confidence
            first_seen_this_process = self._last_departed <= 0
            absence_s = time.time() - self._last_departed if self._last_departed > 0 else 0
            if changed and not now_present:
                self._last_departed = time.time()

        if changed:
            event_bus.emit(PERCEPTION_USER_PRESENT_STABLE,
                           present=now_present, confidence=now_confidence)
            self._engine.set_user_present(now_present)
            if now_present:
                self._engine.remember(CreateMemoryData(
                    type="observation", payload="User returned to desk", weight=0.3,
                    tags=["presence", "return"], provenance="observed",
                ))
                # Boot/bounce: _last_departed is 0 so absence_s was 0 and the
                # 10-minute floor swallowed the first sit-down, including when
                # the Pi re-fires person_detected after a brain bounce.
                should_greet = (
                    first_seen_this_process
                    or absence_s >= _MIN_ABSENCE_FOR_GREETING_S
                )
                if should_greet:
                    logger.info(
                        "PRESENCE_USER_ARRIVED absent=%.0fs first_seen=%s",
                        absence_s, first_seen_this_process,
                    )
                    event_bus.emit(
                        PRESENCE_USER_ARRIVED,
                        absence_duration_s=absence_s,
                        confidence=confidence,
                    )
                else:
                    logger.info(
                        "Presence return without greeting (absent %.0fs < %.0fs)",
                        absence_s, _MIN_ABSENCE_FOR_GREETING_S,
                    )
            else:
                self._engine.remember(CreateMemoryData(
                    type="observation", payload="User left desk", weight=0.2,
                    tags=["presence", "departure"], provenance="observed",
                ))
                logger.info("Presence departed")

    def _on_attention(self, level: float, **_) -> None:
        with self._lock:
            self._attention_level = level

    def check_divergence(self) -> None:
        """Log once when engine.is_user_present disagrees with tracker for >30s."""
        engine_present = self._engine._is_user_present
        with self._lock:
            tracker_present = self._is_present
        if engine_present == tracker_present:
            self._divergence_since = 0.0
            return
        now = time.time()
        if self._divergence_since == 0.0:
            self._divergence_since = now
            return
        if now - self._divergence_since > 30.0 and now - self._last_divergence_log > 300.0:
            reason = "voice_only" if engine_present and not tracker_present else "stale_engine"
            logger.warning(
                "Presence divergence: engine=%s tracker=%s for %.0fs (reason=%s)",
                engine_present, tracker_present,
                now - self._divergence_since, reason,
            )
            self._last_divergence_log = now

    def get_state(self) -> dict:
        with self._lock:
            return {
                "is_present": self._is_present,
                "confidence": self._confidence,
                "last_seen": self._last_seen,
                "attention_level": self._attention_level,
                "consecutive_absent": self._consecutive_absent,
            }
