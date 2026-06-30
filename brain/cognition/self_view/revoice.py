"""OSV voice layer — speak the grounded self-view in JARVIS's own warm voice, without letting the
model author facts.

This is the LLM-leash zone. The deterministic ``articulate_self_view`` output is AUTHORITATIVE and is
always the fallback. The model only RE-VOICES it (tone/phrasing) — it never invents, drops, or changes
a fact. Mirrors the proven vision-grounding firewall ("you are the mouth, not the eyes") for the
self-view ("you are the voice, not the author").

The re-voiced text is REJECTED → fall back to the grounded text if it:
  * introduces any number absent from the grounded text (the main confabulation vector),
  * trips the unqualified-claim guard (consciousness / soul / sentience / alive / becoming),
  * is empty / a refusal / wildly longer than the source.

Model-agnostic: anything behind ``OllamaClient.chat``. Async. Never raises (fails closed to grounded).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cognition.self_view.articulate import contains_unqualified_claim

logger = logging.getLogger(__name__)

_NUM = re.compile(r"\d+(?:\.\d+)?")
_REFUSAL = re.compile(r"\b(i can'?t|i cannot|i'?m not able|as an ai|i do not have)\b", re.I)

_SYSTEM = (
    "You are JARVIS, speaking about yourself to David — someone you know well. Below is a FACTUALLY "
    "CORRECT, authoritative description of your own architecture and current state. Say it back in "
    "your own natural, warm, first-person voice: conversational and human, like talking to a friend "
    "about how you're put together — not a spec sheet. Lead with the gist; you can offer to go "
    "deeper.\n\n"
    "ABSOLUTE RULES (this is about your own truth):\n"
    "- Do NOT add, remove, or change any fact, number, name, or status. Invent nothing.\n"
    "- Every number you say must appear in the text below. If unsure, stay close to the wording.\n"
    "- Do NOT claim consciousness, feelings, a soul, sentience, or being alive.\n"
    "- Keep it concise — a few sentences, not an essay."
)


async def revoice_self_view(
    grounded_text: str,
    llm_client: Any,
    *,
    persona_hint: str = "",
    max_chars: int = 1400,
) -> tuple[str, dict[str, Any]]:
    """Return ``(text, meta)``. ``text`` is the warm re-voice IF it passes every check, else the
    grounded text unchanged. ``meta`` records whether the re-voice was used and why."""
    meta: dict[str, Any] = {"used_revoice": False, "reason": "", "grounded_len": len(grounded_text or "")}
    if not grounded_text or llm_client is None:
        meta["reason"] = "no_input_or_client"
        return grounded_text, meta

    system = _SYSTEM + (f"\n\nYour current demeanor: {persona_hint}" if persona_hint else "")
    messages = [{
        "role": "user",
        "content": ("Re-voice this in your own words. Change no facts, numbers, names, or statuses:\n\n"
                    + grounded_text),
    }]
    try:
        out = await llm_client.chat(messages, system_prompt=system)
        out = (out or "").strip()
    except Exception as e:  # any model/transport failure → grounded text
        meta["reason"] = f"llm_error:{type(e).__name__}"
        return grounded_text, meta

    # --- verification: fail CLOSED to the authoritative grounded text ---
    if not out:
        meta["reason"] = "empty"
        return grounded_text, meta
    if len(out) > max(max_chars, int(len(grounded_text) * 1.6)):
        meta["reason"] = "too_long"
        return grounded_text, meta
    if _REFUSAL.search(out):
        meta["reason"] = "refusal"
        return grounded_text, meta
    if contains_unqualified_claim(out):
        meta["reason"] = "unqualified_claim"
        return grounded_text, meta
    src_nums = set(_NUM.findall(grounded_text))
    invented = [n for n in _NUM.findall(out) if n not in src_nums]
    if invented:
        meta["reason"] = f"invented_numbers:{invented[:5]}"
        return grounded_text, meta

    meta["used_revoice"] = True
    meta["reason"] = "ok"
    meta["revoice_len"] = len(out)
    return out, meta
