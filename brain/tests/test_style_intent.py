import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reasoning.style_intent import detect_style_intent


def test_detect_solemn_style_intent():
    intent = detect_style_intent("Be more solemn when you say that.")
    assert intent is not None
    assert intent.voice_profile_id == "oracle_solemn"


def test_detect_warning_style_intent():
    intent = detect_style_intent("Say it like a warning.")
    assert intent is not None
    assert intent.voice_profile_id == "oracle_guarded"


def test_detect_soft_style_intent():
    intent = detect_style_intent("Could you say it softly?")
    assert intent is not None
    assert intent.voice_profile_id == "oracle_empathetic"
