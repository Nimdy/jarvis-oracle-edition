"""Presence arrival greeting: Pi sends edges, not 3x absent heartbeats.

Lived: zero `Proactive arrival greeting` lines in brain.log / .1 / .2
(back through 2026-07-02). PresenceTracker required 3 consecutive
`present=False` events to mark departed, but the Pi PersonTracker already
waits 5s then sends a single `person_lost`. One lost never reached 3, so
`is_present` stuck True, `_last_departed` stayed 0, and the 10-minute
greeting floor swallowed the first sit-down after boot/bounce.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import (
    PERCEPTION_USER_PRESENT,
    _BarrierState,
    event_bus,
)
from perception.presence import (
    PRESENCE_USER_ARRIVED,
    PresenceTracker,
    _MIN_ABSENCE_FOR_GREETING_S,
)


class _FakeEngine:
    def __init__(self) -> None:
        self._is_user_present = False
        self.remembered: list = []

    def set_user_present(self, present: bool) -> None:
        self._is_user_present = present

    def remember(self, data):
        self.remembered.append(data)
        return None


def _open_bus() -> None:
    if event_bus._barrier != _BarrierState.OPEN:
        event_bus.open_barrier()


def _collect_arrivals() -> tuple[list[dict], callable]:
    seen: list[dict] = []

    def _on(**kwargs) -> None:
        seen.append(kwargs)

    cleanup = event_bus.on(PRESENCE_USER_ARRIVED, _on)
    return seen, cleanup


def test_first_sit_down_after_boot_emits_arrived():
    """Bounce / process start: Pi re-fires person_detected. That is a hello."""
    _open_bus()
    engine = _FakeEngine()
    tracker = PresenceTracker(engine)
    seen, cleanup = _collect_arrivals()
    tracker.start()
    try:
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.9)
        assert engine._is_user_present is True
        assert len(seen) == 1
        assert seen[0]["absence_duration_s"] == 0
    finally:
        tracker.stop()
        cleanup()


def test_single_person_lost_marks_departed(monkeypatch):
    """Pi sends one person_lost. That is departure — the 5s track already ran."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr("perception.presence.time.time", lambda: clock["t"])
    _open_bus()
    engine = _FakeEngine()
    tracker = PresenceTracker(engine)
    tracker.start()
    try:
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.9)
        event_bus.emit(PERCEPTION_USER_PRESENT, present=False, confidence=0.8)
        assert engine._is_user_present is False
        assert tracker.get_state()["is_present"] is False
    finally:
        tracker.stop()


def test_return_after_ten_minutes_emits_welcome_back(monkeypatch):
    clock = {"t": 2_000_000.0}
    monkeypatch.setattr("perception.presence.time.time", lambda: clock["t"])
    _open_bus()
    engine = _FakeEngine()
    tracker = PresenceTracker(engine)
    seen, cleanup = _collect_arrivals()
    tracker.start()
    try:
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.9)
        seen.clear()
        event_bus.emit(PERCEPTION_USER_PRESENT, present=False, confidence=0.8)
        clock["t"] += _MIN_ABSENCE_FOR_GREETING_S + 5.0
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.91)
        assert len(seen) == 1
        assert seen[0]["absence_duration_s"] >= _MIN_ABSENCE_FOR_GREETING_S
    finally:
        tracker.stop()
        cleanup()


def test_short_absence_does_not_greet(monkeypatch):
    clock = {"t": 3_000_000.0}
    monkeypatch.setattr("perception.presence.time.time", lambda: clock["t"])
    _open_bus()
    engine = _FakeEngine()
    tracker = PresenceTracker(engine)
    seen, cleanup = _collect_arrivals()
    tracker.start()
    try:
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.9)
        seen.clear()
        event_bus.emit(PERCEPTION_USER_PRESENT, present=False, confidence=0.8)
        clock["t"] += 90.0
        event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.9)
        assert seen == []
        assert engine._is_user_present is True
    finally:
        tracker.stop()
        cleanup()


def test_on_user_arrived_consumes_calendar_greeting_slot():
    src = open(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "perception_orchestrator.py"),
        encoding="utf-8",
    ).read()
    idx = src.find("def _on_user_arrived")
    next_def = src.find("\n    def ", idx + 1)
    body = src[idx:next_def]
    assert "mark_greeting_today" in body
