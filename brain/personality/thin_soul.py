"""Thin soul pass for native tool-route replies (STATUS / MEMORY / VISION).

#83 WS2: tool-route replies stay grounded and fail-closed, but must not sound
like a template printer. This is NOT native_voice, NOT revoice-live, NOT the
soul-dial ``sent_to_model`` flip, and NOT an LLM author.

What it does:
  * strip leftover markdown markers (TTS also strips; this keeps the spoken
    persist/log aligned when the native string itself had markdown)
  * revoice-style leash: if polish invents numbers or introduces an unqualified
    consciousness claim, return the grounded original
  * log the grown soul-dial *beside* the native turn (``sent_to_model=False``)
    so native STATUS/MEMORY accrue the same A/B as the LLM path

What it does not do:
  * call an LLM
  * inject think-before-speak ``would_inject``
  * rewrite facts, names, or measured numbers
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger("jarvis.thin_soul")

_NUM = re.compile(r"\d+(?:\.\d+)?")
# Same markdown markers TTS uses (reasoning/tts.py). Do NOT run number/contraction
# expansion here — that would invent digit-strings and trip the revoice leash.
_MD_EMPHASIS = re.compile(r"\*{1,3}([^*]+?)\*{1,3}")
_MD_ORPHAN_ASTERISK = re.compile(r"\*{1,3}")
_MD_MISC = re.compile(r"[`#~]{1,3}")
_MD_HEADER = re.compile(r"^#{1,6}\s*", re.M)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_MD_HR = re.compile(r"^-{3,}$", re.M)


def _strip_markdown_markers(text: str) -> str:
    """Drop markdown markers only. Do not expand numbers (that would trip the leash)."""
    t = text or ""
    t = _MD_HR.sub("", t)
    t = _MD_HEADER.sub("", t)
    t = _MD_BULLET.sub("", t)
    t = _MD_EMPHASIS.sub(r"\1", t)
    t = _MD_ORPHAN_ASTERISK.sub("", t)
    t = _MD_MISC.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _log_soul_shadow(*, route: str, grounded: str, polished: str, used: bool) -> None:
    """Same glass-box as response.py soul_dims→voice: observe only, never send."""
    try:
        from consciousness.soul import soul_service
        from reasoning.context import build_dial_trait_block

        dims = dict(getattr(soul_service.identity, "semi_stable_traits", {}) or {})
        rec: dict[str, Any] = {
            "ts": time.time(),
            "route": route,
            "dial_block": build_dial_trait_block(dims),
            "soul_dims": dims,
            "authority": "shadow_observe_only",
            "sent_to_model": False,
            "used_polish": used,
            "grounded_len": len(grounded or ""),
            "polished_len": len(polished or ""),
        }
        path = os.path.join(os.path.expanduser("~"), ".jarvis", "soul_voice_shadow.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        logger.debug("thin-soul shadow log skipped", exc_info=True)


def thin_soul_native(grounded: str, *, route: str = "") -> str:
    """Companion polish on an already-grounded native string. Fail-closed. No LLM."""
    original = grounded or ""
    if not original.strip():
        return original
    try:
        polished = _strip_markdown_markers(original)
        if not polished:
            _log_soul_shadow(route=route, grounded=original, polished=original, used=False)
            return original
        from cognition.self_view.articulate import contains_unqualified_claim

        if contains_unqualified_claim(polished) and not contains_unqualified_claim(original):
            _log_soul_shadow(route=route, grounded=original, polished=polished, used=False)
            return original
        src_nums = set(_NUM.findall(original))
        invented = [n for n in _NUM.findall(polished) if n not in src_nums]
        if invented:
            _log_soul_shadow(route=route, grounded=original, polished=polished, used=False)
            return original
        used = polished != original
        _log_soul_shadow(route=route, grounded=original, polished=polished, used=used)
        return polished
    except Exception:
        logger.debug("thin-soul native pass failed (fail-closed)", exc_info=True)
        return original
