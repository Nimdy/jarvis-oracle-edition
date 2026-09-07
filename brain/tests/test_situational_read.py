"""P0 situational read — overexplain tell must see short asks, not only complexity=simple.

Lived 2026-09-04: 91-word census to a 4-word 'describe your architecture' sat
self_check='reply length proportionate' because complexity was moderate (the
word 'architecture'). Verbosity/person-aware then never accrued. Shadow only.
"""
from __future__ import annotations

import consciousness.situational_read as sr


def _engine(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_STATE_PATH", str(tmp_path / "sit.json"))
    return sr.SituationalReadEngine()


def test_long_reply_to_short_ask_is_overexplain_even_if_complexity_moderate(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch)
    reply = " ".join(["census"] * 91)
    read = eng.observe_turn(
        speaker="David",
        user_text="Describe your own architecture.",
        response_text=reply,
        user_emotion="happy",
        complexity="moderate",
    )
    assert read is not None
    assert "overexplain" in read.self_check
    assert read.would_have_done and "concise" in read.would_have_done


def test_short_reply_stays_proportionate(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch)
    read = eng.observe_turn(
        speaker="David",
        user_text="Describe your own architecture.",
        response_text="I am JARVIS Oracle Edition.",
        complexity="moderate",
    )
    assert read is not None
    assert read.self_check == "reply length proportionate"


def test_long_reply_to_long_complex_ask_is_not_forced_overexplain(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch)
    ask = " ".join(["please"] * 30)
    reply = " ".join(["detail"] * 91)
    read = eng.observe_turn(
        speaker="David",
        user_text=ask,
        response_text=reply,
        complexity="complex",
    )
    assert read is not None
    assert read.self_check == "reply length proportionate"
