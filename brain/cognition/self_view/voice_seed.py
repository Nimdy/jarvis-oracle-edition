"""OSV voice distillation SEED (shadow) — accumulate faithful (grounded -> warm) TEACHER pairs that a
future NATIVE voice NN will distill from, so JARVIS grows into her own voice instead of leaning on the
baseline LLM. See the native-NN-voice directive: the baseline LLM is the TEACHER, never the destination.

How it stays honest:
  * The live self-description is unchanged — the deterministic grounded ``articulate_self_view`` text is
    what's spoken. The baseline LLM NEVER speaks live here; it only generates *training targets*.
  * The teacher re-voice is run through the revoice firewall (no invented numbers / no confab / no
    refusal); ONLY verified-faithful pairs are logged. So the distillation corpus carries zero
    confabulation by construction.

MEMORY-SAFE (David's guardrail): writes ONLY to a bounded, deduped JSONL corpus — never the memory
store, never per-cycle spam, hard-capped so it cannot grow without bound. Never raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from cognition.self_view.revoice import revoice_self_view

logger = logging.getLogger(__name__)

SEED_PATH = Path(os.path.expanduser("~")) / ".jarvis" / "voice_distillation_seed.jsonl"
MAX_SEED_ENTRIES = 2000      # hard cap: stop appending past this (bounded — never grows unbounded)
_DEDUP_CAP = 500             # bounded in-process dedup window

_seen: deque[str] = deque(maxlen=_DEDUP_CAP)
_seen_set: set[str] = set()
_count_cache: int | None = None


def _key(kind: str, grounded: str) -> str:
    h = hashlib.sha1(grounded.encode("utf-8", "ignore")).hexdigest()[:16]
    return f"{kind}:{h}"


def _line_count() -> int:
    global _count_cache
    if _count_cache is not None:
        return _count_cache
    try:
        _count_cache = sum(1 for _ in SEED_PATH.open()) if SEED_PATH.exists() else 0
    except Exception:
        _count_cache = 0
    return _count_cache


async def capture_teacher_pair(grounded_text: str, kind: str, llm_client: Any,
                               *, persona_hint: str = "") -> dict[str, Any]:
    """Best-effort: generate a VERIFIED teacher re-voice of the grounded text and log the pair.

    Returns a small status dict. Never raises. The live reply is unaffected (this is shadow).
    """
    status: dict[str, Any] = {"logged": False, "reason": ""}
    try:
        if not grounded_text or llm_client is None:
            status["reason"] = "no_input_or_client"
            return status
        k = _key(kind, grounded_text)
        if k in _seen_set:
            status["reason"] = "dedup"
            return status
        if _line_count() >= MAX_SEED_ENTRIES:
            status["reason"] = "seed_full"
            return status

        voiced, meta = await revoice_self_view(grounded_text, llm_client, persona_hint=persona_hint)
        if not meta.get("used_revoice"):
            # teacher output failed the fidelity firewall — do NOT poison the corpus
            status["reason"] = f"teacher_rejected:{meta.get('reason')}"
            return status

        rec = {"ts": time.time(), "kind": kind, "grounded": grounded_text,
               "voiced": voiced, "persona": persona_hint}
        SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SEED_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        global _count_cache
        if _count_cache is not None:
            _count_cache += 1
        _seen.append(k)
        _seen_set.add(k)
        if len(_seen_set) > _DEDUP_CAP:           # keep the dedup set bounded to the deque window
            _seen_set.clear()
            _seen_set.update(_seen)

        status["logged"] = True
        status["reason"] = "ok"
        return status
    except Exception as e:
        status["reason"] = f"error:{type(e).__name__}"
        logger.debug("voice-seed capture failed", exc_info=True)
        return status


def seed_stats() -> dict[str, Any]:
    """Read-only observability for the seed corpus."""
    return {"path": str(SEED_PATH), "entries": _line_count(), "cap": MAX_SEED_ENTRIES}
