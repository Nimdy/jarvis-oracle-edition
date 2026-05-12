"""Perform tool — handles singing and other performance requests.

Routes singing requests directly to TTS output, bypassing the LLM
which tends to hedge instead of actually performing.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

_SONGS: dict[str, list[str]] = {
    "twinkle": [
        "Twinkle, twinkle, little star,",
        "How I wonder what you are.",
        "Up above the world so high,",
        "Like a diamond in the sky.",
        "Twinkle, twinkle, little star,",
        "How I wonder what you are.",
    ],
    "abc": [
        "A, B, C, D, E, F, G,",
        "H, I, J, K, L, M, N, O, P,",
        "Q, R, S, T, U, V,",
        "W, X, Y and Z.",
        "Now I know my A, B, C's,",
        "Next time won't you sing with me?",
    ],
    "row": [
        "Row, row, row your boat,",
        "Gently down the stream.",
        "Merrily, merrily, merrily, merrily,",
        "Life is but a dream.",
    ],
    "mary": [
        "Mary had a little lamb,",
        "Little lamb, little lamb.",
        "Mary had a little lamb,",
        "Its fleece was white as snow.",
        "And everywhere that Mary went,",
        "Mary went, Mary went.",
        "Everywhere that Mary went,",
        "The lamb was sure to go.",
    ],
    "happy": [
        "If you're happy and you know it, clap your hands!",
        "If you're happy and you know it, clap your hands!",
        "If you're happy and you know it, and you really want to show it,",
        "If you're happy and you know it, clap your hands!",
    ],
    "jingle": [
        "Jingle bells, jingle bells, jingle all the way!",
        "Oh what fun it is to ride in a one-horse open sleigh, hey!",
        "Jingle bells, jingle bells, jingle all the way!",
        "Oh what fun it is to ride in a one-horse open sleigh!",
    ],
}

_SONG_KEYWORDS: dict[str, list[str]] = {
    "twinkle": ["twinkle", "star", "wonder"],
    "abc": ["abc", "a b c", "alphabet"],
    "row": ["row", "boat", "stream"],
    "mary": ["mary", "lamb"],
    "happy": ["happy", "clap"],
    "jingle": ["jingle", "bells", "christmas", "sleigh"],
}


def _pick_song(user_text: str) -> tuple[str, list[str]]:
    """Pick a song based on user request, or random if no match."""
    lower = user_text.lower()
    for song_id, keywords in _SONG_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return song_id, _SONGS[song_id]
    song_id = random.choice(list(_SONGS.keys()))
    return song_id, _SONGS[song_id]


def handle_perform_request(user_text: str) -> list[str]:
    """Return a list of lyric lines to be sent to TTS one by one."""
    song_id, lyrics = _pick_song(user_text)
    logger.info("Perform: singing '%s' (%d lines)", song_id, len(lyrics))
    return lyrics
