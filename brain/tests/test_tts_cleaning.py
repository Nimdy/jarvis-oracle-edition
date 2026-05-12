import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reasoning.tts import BrainTTS


def test_clean_for_speech_expands_im_contraction():
    cleaned = BrainTTS._clean_for_speech("I'm available.")
    assert cleaned == "I am available."


def test_clean_for_speech_preserves_possessive_apostrophes_inside_words():
    cleaned = BrainTTS._clean_for_speech("David's camera is ready.")
    assert cleaned == "David's camera is ready."


def test_clean_matched_bold():
    cleaned = BrainTTS._clean_for_speech("**In summary:** you are fine.")
    assert "**" not in cleaned
    assert "In summary" in cleaned


def test_clean_orphan_double_asterisk():
    cleaned = BrainTTS._clean_for_speech("**1.")
    assert "*" not in cleaned
    assert "1." in cleaned


def test_clean_orphan_trailing_asterisk():
    cleaned = BrainTTS._clean_for_speech("What makes you Jarvis?**")
    assert "*" not in cleaned
    assert "Jarvis" in cleaned


def test_clean_orphan_single_asterisk():
    cleaned = BrainTTS._clean_for_speech("*emphasis without close")
    assert "*" not in cleaned
    assert "emphasis" in cleaned


def test_clean_numbered_list():
    cleaned = BrainTTS._clean_for_speech("1. First point\n2. Second point")
    assert cleaned.startswith("First point")
    assert "Second point" in cleaned


def test_clean_bullet_list():
    cleaned = BrainTTS._clean_for_speech("- First item\n- Second item")
    assert "-" not in cleaned
    assert "First item" in cleaned


def test_clean_header():
    cleaned = BrainTTS._clean_for_speech("## Section Title\nContent here.")
    assert "##" not in cleaned
    assert "Section Title" in cleaned


def test_clean_mixed_markdown_response():
    text = (
        "**1. What makes you Jarvis?**\n"
        "Your identity is defined by your systems.\n\n"
        "**2. How many systems?**\n"
        "- Memory search\n"
        "- Speech output\n"
        "- Speaker identification"
    )
    cleaned = BrainTTS._clean_for_speech(text)
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "-" not in cleaned or cleaned.count("-") == 0
    assert "What makes you Jarvis" in cleaned
    assert "Memory search" in cleaned
