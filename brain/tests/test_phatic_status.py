"""Phatic STATUS uses measured affect + personality, not mode HUD or fake feelings."""
from __future__ import annotations

from types import SimpleNamespace

from reasoning.bounded_response import (
    articulate_phatic_self_status,
    is_phatic_status_ask,
)


def _ro(*, level: float, clamped: bool = False, zero: bool = False):
    return SimpleNamespace(
        level=level,
        cannot_lie_clamped=clamped,
        all_sources_zero=zero,
    )


def test_phatic_detector_how_are_you_and_hello():
    assert is_phatic_status_ask("How are you?")
    assert is_phatic_status_ask("how are you feeling")
    assert is_phatic_status_ask("Hey Jarvis, good morning.")
    assert is_phatic_status_ask("Good morning.")
    assert not is_phatic_status_ask("how is your system health")
    assert not is_phatic_status_ask("what mode are you in")
    assert not is_phatic_status_ask("give me a status report")


def test_empty_affect_fail_closed_is_here():
    line = articulate_phatic_self_status("How are you?", affect_snapshot=None, traits=[])
    assert line == "I'm here."
    assert "mode" not in line.lower()
    assert "feel" not in line.lower()


def test_clamped_affect_does_not_invent_mood():
    snap = SimpleNamespace(
        dopamine=_ro(level=0.9, clamped=True, zero=True),
        serotonin=_ro(level=0.9, clamped=True, zero=True),
        cortisol=_ro(level=0.9, clamped=True, zero=True),
    )
    line = articulate_phatic_self_status("How are you?", affect_snapshot=snap, traits=[])
    assert line == "I'm here."


def test_cortisol_tension_is_operational():
    snap = SimpleNamespace(
        dopamine=_ro(level=0.5),
        serotonin=_ro(level=0.5),
        cortisol=_ro(level=0.7),
    )
    line = articulate_phatic_self_status("How are you feeling?", affect_snapshot=snap)
    assert line == "I'm here, under some tension."
    assert "happy" not in line.lower()
    assert "feel" not in line.lower()


def test_steady_serotonin_and_warm_trait():
    snap = SimpleNamespace(
        dopamine=_ro(level=0.4),
        serotonin=_ro(level=0.65),
        cortisol=_ro(level=0.4),
    )
    line = articulate_phatic_self_status(
        "How are you?", affect_snapshot=snap, traits=["Empathetic"],
    )
    assert line == "I'm here with you."


def test_greeting_prefix_plus_body():
    line = articulate_phatic_self_status("Good morning.", affect_snapshot=None, traits=[])
    assert line == "Good morning. I'm here."
