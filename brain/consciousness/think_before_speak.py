"""Think-Before-Speak — TBS-0 (SHADOW pre-speech read). docs/THINK_BEFORE_SPEAK.md

The vision (operator): "her internal thoughts are supposed to drive conversations — like humans think
before they speak." Today the companion read fires AFTER she speaks (situational_read at
conversation_handler.py:6026), so her self-knowledge arrives a full turn too late. TBS-0 adds a read of
the CURRENT user turn BEFORE generation and emits a pre-speech STANCE (lean_concise / give_space /
match_warmth / none).

TBS-0 IS PURE SHADOW (glass-box):
  * Computed BEFORE the reply is generated, but it INJECTS NOTHING into the prompt and changes nothing
    she says. ``injects_prompt`` / ``shapes_reply`` are structurally False and surfaced so it's auditable.
  * No LLM call — pure-Python over the learned theory-of-mind person-model + the incoming user signal
    (latency floor: it must be cheap, so it can run before generation without slowing the turn).
  * REUSES (does not reinvent): the theory_of_mind person-model (verbosity_pref / humor_reception /
    disposition) + the behavior_advisory stance vocabulary. ``would_inject`` is the EXACT prompt line a
    future TBS-2 would inject via the proven ``style_instruction`` seam — logged here, never injected.
  * Earns TBS-1 (advisory) / TBS-2 (active, the earned P3->P4 flip) by demonstrated accuracy against the
    post-hoc read + transcripts. Never hardcoded, never self-promoted, model-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.tbs")

_SHADOW_LOG = str(Path.home() / ".jarvis" / "pre_speech_shadow.jsonl")

# Single source of truth: the behavior_advisory person-aware floor (a learned disposition only counts
# once earned). IMPORTED, not re-hardcoded, so the two can't silently desync. Fallback if unavailable.
try:
    from consciousness.behavior_advisory import _PERSON_AWARE_DISPOSITION_FLOOR as _CONF_FLOOR
except Exception:
    _CONF_FLOOR = 0.30
_LOAD_MAX_LINES = 5000   # bound the glass-box rebuild cost on boot (counts reflect the recent window)
_VERBOSITY_CONCISE = "prefers concise replies"
_HUMOR_LANDS = "lands well"
_NEGATIVE_EMOTIONS = frozenset({"angry", "frustrated", "annoyed", "sad", "upset", "anxious", "fearful"})


@dataclass
class PreSpeechStance:
    """A pre-speech read of the CURRENT turn. Hypothesis-only; SHADOW (injected=False always in TBS-0)."""
    speaker: str
    stance: str                 # lean_concise | give_space | match_warmth | none
    confidence: float
    evidence: list = field(default_factory=list)   # [[signal, value], ...]
    would_inject: "str | None" = None               # the prompt line TBS-2 WOULD inject (logged, not injected)
    authority: str = "shadow_observe_only"
    injected: bool = False                          # structurally False in TBS-0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker, "stance": self.stance,
            "confidence": round(self.confidence, 3),
            "evidence": [list(e) for e in self.evidence],
            "would_inject": self.would_inject,
            "authority": self.authority, "injected": self.injected,
            "timestamp": self.timestamp,
        }


class PreSpeechReader:
    """TBS-0 engine. Computes + LOGS a pre-speech stance each turn (glass-box). Injects nothing."""

    _instance: "PreSpeechReader | None" = None

    def __init__(self) -> None:
        self._total = 0
        self._stance_counts: dict[str, int] = {}
        self._last: PreSpeechStance | None = None
        self._recent: deque[PreSpeechStance] = deque(maxlen=50)
        self._load()   # rebuild glass-box counters from the durable log so it survives reboots

    @classmethod
    def get_instance(cls) -> "PreSpeechReader":
        if cls._instance is None:
            cls._instance = PreSpeechReader()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def read_before_speak(self, *, speaker: str, user_text: str, user_emotion: str = "neutral",
                          person_model: "dict | None" = None) -> PreSpeechStance:
        """Read the CURRENT user turn (NO response_text) → a pre-speech stance. Pure, cheap, never raises.
        SHADOW: the returned stance is LOGGED only; the caller must NOT inject it (TBS-0)."""
        try:
            pm = person_model or {}
            verbosity = str(pm.get("verbosity_pref", "") or "")
            verb_conf = float(pm.get("verbosity_confidence", 0.0) or 0.0)
            humor = str(pm.get("humor_reception", "") or "")
            humor_conf = float(pm.get("humor_confidence", 0.0) or 0.0)
            responsiveness = str(pm.get("responsiveness", "") or "")
            negative = user_emotion in _NEGATIVE_EMOTIONS

            evidence: list = [["user_emotion", user_emotion], ["verbosity_pref", verbosity or "forming"]]
            # Priority: distress/withdrawal first, then the LEARNED concise preference, then warmth.
            if negative or responsiveness == "withdrawn":
                stance, conf = "give_space", 0.4
                would = ("Internal read: they may be frustrated or pulling back — soften, give space, "
                         "keep it brief.")
            elif verbosity == _VERBOSITY_CONCISE and verb_conf >= _CONF_FLOOR:
                stance = "lean_concise"
                conf = round(min(1.0, 0.4 + verb_conf * 0.4), 3)
                would = ("Internal read: you tend to overexplain with this person — lean concise unless "
                         "they ask for depth.")
                evidence.append(["verbosity_confidence", round(verb_conf, 3)])
            elif humor == _HUMOR_LANDS and humor_conf >= _CONF_FLOOR and not negative:
                stance = "match_warmth"
                conf = round(min(1.0, 0.4 + humor_conf * 0.4), 3)
                would = "Internal read: warmth and humor land with this person — you can lean playful and warm."
                evidence.append(["humor_confidence", round(humor_conf, 3)])
            else:
                stance, conf, would = "none", 0.3, None

            s = PreSpeechStance(speaker=speaker or "unknown", stance=stance, confidence=conf,
                                evidence=evidence, would_inject=would, timestamp=time.time())
            self._record(s)
            return s
        except Exception:
            logger.debug("read_before_speak failed (no-op)", exc_info=True)
            return PreSpeechStance(speaker=speaker or "unknown", stance="none", confidence=0.0,
                                   timestamp=time.time())

    def _record(self, s: PreSpeechStance) -> None:
        self._total += 1
        self._stance_counts[s.stance] = self._stance_counts.get(s.stance, 0) + 1
        self._last = s
        self._recent.append(s)
        try:
            Path(_SHADOW_LOG).parent.mkdir(parents=True, exist_ok=True)
            with open(_SHADOW_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(s.to_dict()) + "\n")
        except Exception:
            logger.debug("pre-speech shadow-log append failed (fail-open)", exc_info=True)

    def _load(self) -> None:
        """Rebuild the glass-box counters + recent from the durable log (fail-open) so the panel shows
        history across reboots instead of post-restart zeros."""
        try:
            if not os.path.exists(_SHADOW_LOG):
                return
            counts: dict[str, int] = {}
            total = 0
            tail: deque[PreSpeechStance] = deque(maxlen=50)
            with open(_SHADOW_LOG, encoding="utf-8") as f:
                _lines = f.readlines()
            if len(_lines) > _LOAD_MAX_LINES:        # bound boot cost; counts reflect the recent window
                _lines = _lines[-_LOAD_MAX_LINES:]
            for line in _lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    st = d.get("stance", "none")
                    counts[st] = counts.get(st, 0) + 1
                    total += 1
                    tail.append(PreSpeechStance(
                        speaker=d.get("speaker", "unknown"), stance=st,
                        confidence=float(d.get("confidence", 0.0) or 0.0),
                        evidence=d.get("evidence", []) or [], would_inject=d.get("would_inject"),
                        injected=bool(d.get("injected", False)), timestamp=d.get("timestamp", 0.0)))
            self._total = total
            self._stance_counts = counts
            self._recent = tail
            self._last = tail[-1] if tail else None
        except Exception:
            logger.debug("pre-speech load failed (fail-open)", exc_info=True)

    def get_status(self) -> dict[str, Any]:
        """GLASS BOX — the full observable state of TBS-0. ``injects_prompt`` is structurally False so the
        zero-authority guarantee is auditable from the panel."""
        return {
            "phase": "TBS-0_shadow_observe",
            "authority": "shadow_observe_only",
            "injects_prompt": False,     # <-- TBS-0 never feeds the stance into the prompt
            "shapes_reply": False,
            "total_reads": self._total,
            "stance_distribution": dict(self._stance_counts),
            "last": self._last.to_dict() if self._last else None,
            "recent": [s.to_dict() for s in list(self._recent)[-10:]],
            "earns_next_by": ("TBS-1: the pre-speech stance matches the post-hoc read + transcript review; "
                              "TBS-2: it earns prompt-injection via the style_instruction seam (the P3->P4 flip)"),
            "note": ("pre-speech stance computed BEFORE generation; LOGGED only, never injected (TBS-0). "
                     "Reuses theory_of_mind + behavior_advisory vocabulary. docs/THINK_BEFORE_SPEAK.md"),
        }


pre_speech_reader = PreSpeechReader.get_instance()
