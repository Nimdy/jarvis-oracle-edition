"""Unsolicited recap of ambient speech — fail-closed on NONE.

Lived 2026-09-15: always-listening STT of game-base / block-mode rambles
went NONE; Qwen authored markdown briefings ('let me break this down',
'What You're Asking For (Summarized)'). Companion may hear. It must not
recap unless asked. Do not concat TBS into style.
"""

from __future__ import annotations

import re

AMBIENT_RECAP_NOD = "Got it."

_ASKED_FOR_RECAP_RE = re.compile(
    r"\b(?:summarize|summarise|recap|sum up|tl;dr|tldr|"
    r"what did i (?:just )?say|"
    r"break (?:it|that|this) down|"
    r"repeat (?:that|what i said)|"
    r"notes on what i said)\b",
    re.I,
)

_RECAP_THEATER_RE = re.compile(
    r"let me break (?:this|that|it) down"
    r"|here'?s a structured summary"
    r"|structured summary of what you"
    r"|what you'?re asking for\s*\(\s*summarized\s*\)"
    r"|what you(?:'re| are) saying:"
    r"|#{2,3}\s",
    re.I,
)


def user_asked_for_recap(user_text: str) -> bool:
    return bool(user_text and _ASKED_FOR_RECAP_RE.search(user_text))


def looks_like_recap_theater(reply: str) -> bool:
    if not reply:
        return False
    if _RECAP_THEATER_RE.search(reply):
        return True
    if reply.count("\n- ") >= 4 or reply.count("\n* ") >= 4:
        return True
    return False


def should_clamp_unsolicited_recap(user_text: str, reply: str) -> bool:
    """True when the reply is a briefing of speech the user did not ask to recap."""
    if not reply or not user_text:
        return False
    if user_asked_for_recap(user_text):
        return False
    if "?" in user_text:
        return False
    return looks_like_recap_theater(reply)
