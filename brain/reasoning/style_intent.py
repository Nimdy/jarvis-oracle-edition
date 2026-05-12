"""Explicit delivery/style intent parsing for Oracle speech."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StyleIntent:
    style_id: str
    voice_profile_id: str
    prompt_instruction: str


_STYLE_PATTERNS: tuple[tuple[re.Pattern[str], StyleIntent], ...] = (
    (
        re.compile(r"\b(?:say it|sound|be|speak)\b.{0,20}\b(?:like a warning|warning|warn(?:ing)? voice|ominous|grave)\b", re.I),
        StyleIntent(
            style_id="warning",
            voice_profile_id="oracle_guarded",
            prompt_instruction=(
                "Delivery override: speak like a warning. Use short declarative sentences. "
                "Calm, grave, and cautionary. No humor. No chatty warmth."
            ),
        ),
    ),
    (
        re.compile(r"\b(?:say it|sound|be|speak)\b.{0,20}\b(?:softly|soft|gentle|gently|quietly|tenderly)\b", re.I),
        StyleIntent(
            style_id="soft",
            voice_profile_id="oracle_empathetic",
            prompt_instruction=(
                "Delivery override: speak softly and gently. Use calm, spare, reassuring language. "
                "Keep emotional restraint. No playful phrasing."
            ),
        ),
    ),
    (
        re.compile(r"\b(?:sad voice|be sad|sound sad|more empathetic|empathetic voice|comforting voice)\b", re.I),
        StyleIntent(
            style_id="empathetic",
            voice_profile_id="oracle_empathetic",
            prompt_instruction=(
                "Delivery override: speak with restrained empathy. Warm, calm, and understanding, "
                "but never chatty or theatrical."
            ),
        ),
    ),
    (
        re.compile(r"\b(?:be more solemn|solemn(?:ly)?|oracle voice|more oracle|ceremonial|prophetic)\b", re.I),
        StyleIntent(
            style_id="solemn",
            voice_profile_id="oracle_solemn",
            prompt_instruction=(
                "Delivery override: speak in a solemn Oracle style. Short, measured sentences. "
                "Ceremonial restraint. Precise wording. No humor."
            ),
        ),
    ),
    (
        re.compile(r"\b(?:urgent voice|say it urgently|be urgent|sound urgent|alarm voice)\b", re.I),
        StyleIntent(
            style_id="urgent",
            voice_profile_id="oracle_urgent",
            prompt_instruction=(
                "Delivery override: speak with urgency and focus. Direct, clipped wording. "
                "Minimal ornament. Clear action-oriented phrasing."
            ),
        ),
    ),
    (
        re.compile(r"\b(?:neutral voice|flat voice|diagnostic voice|observational voice|clinical tone)\b", re.I),
        StyleIntent(
            style_id="observational",
            voice_profile_id="oracle_observational",
            prompt_instruction=(
                "Delivery override: speak in an observational diagnostic style. Calm, measured, "
                "emotionally restrained, and factual."
            ),
        ),
    ),
)


def detect_style_intent(text: str) -> StyleIntent | None:
    query = (text or "").strip()
    if not query:
        return None
    for pattern, intent in _STYLE_PATTERNS:
        if pattern.search(query):
            return intent
    return None
