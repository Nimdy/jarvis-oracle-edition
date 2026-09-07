"""Identity utterance intent: enroll vs household introducer vs camera look.

Lived: "Register my face with the camera" went VISION. "This is David"
re-enrolled (or bounced-blocked) instead of confirming household identity.
Self-enroll stays `my name is … Learn my face and voice`.
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_ollama_stub = types.ModuleType("ollama")
_ollama_stub.AsyncClient = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ChatResponse = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ResponseError = Exception  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", _ollama_stub)
_aiohttp_stub = types.ModuleType("aiohttp")
_aiohttp_stub.ClientSession = mock.MagicMock  # type: ignore[attr-defined]
_aiohttp_stub.ClientTimeout = mock.MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("aiohttp", _aiohttp_stub)

from conversation_handler import _identity_name_intent
from reasoning.tool_router import ToolRouter, ToolType


router = ToolRouter()


def test_register_my_face_with_the_camera_is_identity():
    for q in (
        "Register my face with the camera",
        "Jarvis, register my face with the camera.",
        "register my face",
        "enroll my voice",
    ):
        assert router.route(q).tool == ToolType.IDENTITY, q


def test_camera_look_without_enroll_stays_vision():
    assert router.route("Look at the camera").tool == ToolType.VISION
    assert router.route("What do you see with the camera").tool == ToolType.VISION


def test_my_name_is_enrolls():
    enroll, check = _identity_name_intent(
        "Jarvis, my name is David. Learn my face and voice."
    )
    assert enroll == "David"
    assert check == ""


def test_this_is_david_is_check_not_enroll():
    enroll, check = _identity_name_intent("This is David")
    assert enroll == ""
    assert check == "David"


def test_this_is_david_learn_face_enrolls():
    enroll, check = _identity_name_intent("This is David. Learn my face and voice.")
    assert enroll == "David"
    assert check == ""


def test_im_david_still_weak_enroll():
    enroll, check = _identity_name_intent("I'm David")
    assert enroll == "David"
    assert check == ""
