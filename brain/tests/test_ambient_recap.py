"""Lived 2026-09-15: unsolicited markdown recaps of ambient speech."""

from reasoning.ambient_recap import (
    AMBIENT_RECAP_NOD,
    looks_like_recap_theater,
    should_clamp_unsolicited_recap,
    user_asked_for_recap,
)


_DRILL = (
    "so right now there's four drill sites sometimes there's only two "
    "sometimes there is three um you don't have to build on a drill site"
)

_DRILL_REPLY = (
    "Okay, let me break this down for clarity. You're describing a scenario "
    "involving **drill sites**. Here's a structured summary of what you're saying:\n"
    "\n### Drill Sites\n- There are **4 drill sites**\n"
)

_BLOCK_REPLY = (
    "David, I understand the frustration — you're dealing with a system that's "
    "in **block mode**. Let me break this down for you, step by step.\n"
    "### What You're Asking For (Summarized):\n"
    "- A system that's in block mode\n"
)


def test_lived_drill_sites_recap_is_clamped():
    assert should_clamp_unsolicited_recap(_DRILL, _DRILL_REPLY)
    assert looks_like_recap_theater(_DRILL_REPLY)


def test_lived_block_mode_recap_is_clamped():
    rant = (
        "this so what you were asking earlier this is what this what you were "
        "asking for it tells you like everything and this is actually in block mode"
    )
    assert should_clamp_unsolicited_recap(rant, _BLOCK_REPLY)


def test_asked_summarize_is_not_clamped():
    assert user_asked_for_recap("Jarvis, summarize what I just said")
    assert not should_clamp_unsolicited_recap(
        "Jarvis, summarize what I just said",
        _DRILL_REPLY,
    )


def test_question_is_not_clamped():
    assert not should_clamp_unsolicited_recap(
        "Who made the game Halo?",
        "Bungie developed Halo in 2001 for Xbox.",
    )


def test_companion_nod_is_not_theater():
    assert not looks_like_recap_theater("I'm here with you.")
    assert not looks_like_recap_theater(AMBIENT_RECAP_NOD)
    assert not should_clamp_unsolicited_recap("Pushing through Jarvis.", "I'm here, David.")


def test_bye_guys_without_recap_is_not_clamped():
    assert not should_clamp_unsolicited_recap(
        "Yeah, no worries. Yeah, of course. Bye, guys.",
        "You're all set, David. I'll keep the door open for you.",
    )
