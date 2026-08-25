"""Lived miss 2026-08-24: VISION said kitchen while the camera showed a desk.

Caption: person at a desk, three monitors, keyboard, mouse.
Spoken: "You're still in the kitchen... pot on the stove... cutting board."
Cause: cooking-dinner chat memories + conversation history rode the vision
prompt. Live frame is the only scene authority.
"""
from __future__ import annotations

from pathlib import Path

from conversation_handler import vision_reply_confabulates


_DESK_CAPTION = (
    "A person in a dark shirt sits at a desk with three monitors displaying "
    "code, video, and text. They use a keyboard and mouse."
)
_KITCHEN_LIE = (
    "I see you, David. You're still in the kitchen, focused on finishing up "
    "dinner. There's a pot on the stove, a cutting board with ingredients."
)


def test_kitchen_lie_is_ungrounded_against_desk_caption() -> None:
    assert vision_reply_confabulates(_DESK_CAPTION, _KITCHEN_LIE) is True


def test_honest_desk_report_is_grounded() -> None:
    spoken = "You're at the desk with three monitors, a keyboard, and a mouse."
    assert vision_reply_confabulates(_DESK_CAPTION, spoken) is False


def test_kitchen_in_caption_is_allowed() -> None:
    cap = "A person stands in a kitchen next to a stove."
    spoken = "You're in the kitchen by the stove."
    assert vision_reply_confabulates(cap, spoken) is False


def test_vision_skips_memory_and_history() -> None:
    src = (Path(__file__).resolve().parent.parent / "reasoning" / "response.py").read_text(
        encoding="utf-8"
    )
    assert 'tool_hint == "vision"' in src
    assert "vision live-frame is scene authority" in src
    assert "messages = [{\"role\": \"user\", \"content\": user_message}]" in src
    handler = (Path(__file__).resolve().parent.parent / "conversation_handler.py").read_text(
        encoding="utf-8"
    )
    assert "vision_reply_confabulates(scene_desc, sentence)" in handler
    assert "speaking caption" in handler
    assert "persist_response=False" in handler
    assert "_persist_spoken_turn(text, reply)" in handler
