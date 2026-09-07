"""Lived miss 2026-08-24: VISION said kitchen while the camera showed a desk.

Caption: person at a desk, three monitors, keyboard, mouse.
Spoken: "You're still in the kitchen... pot on the stove... cutting board."
Cause: cooking-dinner chat memories + conversation history rode the vision
prompt. Live frame is the only scene authority.
"""
from __future__ import annotations

from pathlib import Path

from conversation_handler import vision_reply_confabulates
from tools.vision_tool import GENERIC_SCENE_PROMPT, vqa_prompt, _snapshot_url


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
    assert "is_targeted_visual_question" in handler
    assert "vqa_prompt" in handler
    assert "VISION VQA: speaking VLM answer" in handler
    assert "(not targeted) and scene_ingest_callback" in handler
    assert "fetch_snapshot(pi_snapshot_url, fresh=True)" in handler
    fetch_at = handler.find("jpeg_bytes = await fetch_snapshot(pi_snapshot_url, fresh=True)")
    focus_at = handler.find("One moment — focusing my vision.")
    assert 0 <= fetch_at < focus_at
    assert "vision_retry_followup" in handler
    assert "vision_retry_query" in handler
    assert "follow_up_retry" in handler
    assert "engine._last_vision_query" in handler
    assert "engine._last_tool" in handler


def test_vqa_prompt_wraps_the_user_question() -> None:
    p = vqa_prompt("How many fingers am I holding up?")
    assert "How many fingers am I holding up?" in p
    assert "do not guess" in p.lower()
    assert p != GENERIC_SCENE_PROMPT
    assert vqa_prompt("") == GENERIC_SCENE_PROMPT
    retry = vqa_prompt(
        "How many fingers am I holding up?",
        correction="thumbs tucked, four fingers on each hand. Check again",
    )
    assert "How many fingers am I holding up?" in retry
    assert "four fingers" in retry
    assert "NEW camera frame" in retry


def test_fresh_snapshot_url_asks_pi_to_grab() -> None:
    assert _snapshot_url("http://192.168.1.248:8080/snapshot", False).endswith("/snapshot")
    assert _snapshot_url("http://192.168.1.248:8080/snapshot", True).endswith("grab=1")
