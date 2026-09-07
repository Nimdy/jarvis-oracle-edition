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
_TBS1_PATH = str(Path.home() / ".jarvis" / "pre_speech_tbs1.json")

# Post-hoc advisory/read → TBS stance family. Soften is distress-family with give_space.
_POSTHOC_STANCE = {
    "be_concise": "lean_concise",
    "give_space": "give_space",
    "soften_tone": "give_space",
    "lean_into_warmth": "match_warmth",
}

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


_PROTECTED_LENGTH_HINTS = frozenset({"brief", "detailed"})


def earned_concise_length_hint(
    stance: PreSpeechStance | None,
    *,
    current_hint: str = "",
) -> str | None:
    """NONE LLM length-hint consume of earned ToM concise.

    When TBS stance is ``lean_concise`` (ToM ``prefers concise replies`` at
    the person-aware floor), return ``brief`` so the existing
    ``response_length_hint`` guideline fires. Does not return the TBS-2
    prompt line. Does not overwrite a this-turn brief/detailed hint.
    Unearned ToM (David 2026-09-07: confidence 0) returns None — Qwen
    stays the conversational voice until the axis earns.
    """
    if stance is None:
        return None
    if getattr(stance, "stance", "") != "lean_concise":
        return None
    cur = str(current_hint or "").strip().lower()
    if cur in _PROTECTED_LENGTH_HINTS:
        return None
    return "brief"


class PreSpeechReader:
    """TBS-0 engine. Computes + LOGS a pre-speech stance each turn (glass-box). Injects nothing."""

    _instance: "PreSpeechReader | None" = None

    def __init__(self) -> None:
        self._total = 0
        self._stance_counts: dict[str, int] = {}
        self._last: PreSpeechStance | None = None
        self._recent: deque[PreSpeechStance] = deque(maxlen=50)
        self._tbs1_scored = 0
        self._tbs1_match = 0
        self._tbs1_related = 0
        self._tbs1_mismatch = 0
        self._tbs1_abstain = 0
        self._tbs1_last: dict[str, Any] | None = None
        self._load()   # rebuild glass-box counters from the durable log so it survives reboots
        self._load_tbs1()

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
            phatic = False
            try:
                from reasoning.bounded_response import is_phatic_status_ask
                phatic = is_phatic_status_ask(user_text)
            except Exception:
                phatic = False
            # Priority: distress/withdrawal first, then the LEARNED concise preference, then warmth.
            # Lived 2026-09-07: wav2vec2 tagged "good afternoon" / hello as frustrated and
            # stamped give_space. A phatic hello is not withdrawal. Sensor noise must not
            # beat the utterance. Real distress text ("ugh") still wins.
            if phatic:
                evidence.append(["phatic_hello", True])
            if (negative or responsiveness == "withdrawn") and not phatic:
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

    @staticmethod
    def _posthoc_stance(read: Any, advisory: Any = None) -> str:
        """Map the post-hoc read/advisory onto a TBS stance family. Delay-only is none."""
        if advisory is not None:
            for sug in getattr(advisory, "suggestions", None) or []:
                adj = str((sug or {}).get("adjustment") or "")
                if adj in _POSTHOC_STANCE:
                    return _POSTHOC_STANCE[adj]
        would = str(getattr(read, "would_have_done", "") or "").lower()
        if "more concise" in would:
            return "lean_concise"
        if "giving space" in would:
            return "give_space"
        if "softening" in would:
            return "give_space"
        if "warmth" in would or "rapport" in would:
            return "match_warmth"
        return "none"

    @staticmethod
    def _agreement(pre: str, post: str) -> str:
        pre = pre or "none"
        post = post or "none"
        if pre == "none" and post == "none":
            return "abstain"
        if pre == post:
            return "match"
        # Distress family: give_space vs a mapped soften (already collapsed to give_space).
        if {pre, post} <= {"give_space", "lean_concise"} and pre != post:
            return "mismatch"
        if pre == "none" or post == "none":
            return "mismatch"
        return "mismatch"

    def score_against_post_hoc(
        self, stance: PreSpeechStance | None, read: Any, advisory: Any = None,
    ) -> dict[str, Any]:
        """TBS-1: score the pre-speech stance vs the post-hoc read. Injects nothing."""
        pre = (stance.stance if stance is not None else "none") or "none"
        post = self._posthoc_stance(read, advisory)
        verdict = self._agreement(pre, post)
        self._tbs1_scored += 1
        if verdict == "match":
            self._tbs1_match += 1
        elif verdict == "related":
            self._tbs1_related += 1
        elif verdict == "abstain":
            self._tbs1_abstain += 1
        else:
            self._tbs1_mismatch += 1
        row = {
            "pre": pre, "post": post, "verdict": verdict,
            "injected": False, "timestamp": time.time(),
        }
        self._tbs1_last = row
        self._save_tbs1()
        return row

    def _tbs1_agreement_rate(self) -> float | None:
        denom = self._tbs1_scored - self._tbs1_abstain
        if denom <= 0:
            return None
        return round((self._tbs1_match + 0.5 * self._tbs1_related) / denom, 3)

    def _save_tbs1(self) -> None:
        try:
            Path(_TBS1_PATH).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "scored": self._tbs1_scored,
                "match": self._tbs1_match,
                "related": self._tbs1_related,
                "mismatch": self._tbs1_mismatch,
                "abstain": self._tbs1_abstain,
                "last": self._tbs1_last,
            }
            with open(_TBS1_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            logger.debug("tbs1 save failed (fail-open)", exc_info=True)

    def _load_tbs1(self) -> None:
        try:
            if not os.path.exists(_TBS1_PATH):
                return
            with open(_TBS1_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self._tbs1_scored = int(data.get("scored", 0) or 0)
            self._tbs1_match = int(data.get("match", 0) or 0)
            self._tbs1_related = int(data.get("related", 0) or 0)
            self._tbs1_mismatch = int(data.get("mismatch", 0) or 0)
            self._tbs1_abstain = int(data.get("abstain", 0) or 0)
            last = data.get("last")
            self._tbs1_last = last if isinstance(last, dict) else None
        except Exception:
            logger.debug("tbs1 load failed (fail-open)", exc_info=True)

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
            "advisory_score": {
                "phase": "TBS-1_score_against_post_hoc",
                "scored": self._tbs1_scored,
                "match": self._tbs1_match,
                "related": self._tbs1_related,
                "mismatch": self._tbs1_mismatch,
                "abstain": self._tbs1_abstain,
                "agreement_rate": self._tbs1_agreement_rate(),
                "last": self._tbs1_last,
                "injects_prompt": False,
            },
            "earns_next_by": ("TBS-1: the pre-speech stance matches the post-hoc read + transcript review; "
                              "TBS-2: it earns prompt-injection via the style_instruction seam (the P3->P4 flip)"),
            "note": ("pre-speech stance computed BEFORE generation; LOGGED only, never injected (TBS-0). "
                     "TBS-1 scores that stance against the post-hoc read. Still never injected. "
                     "Reuses theory_of_mind + behavior_advisory vocabulary. docs/THINK_BEFORE_SPEAK.md"),
        }


pre_speech_reader = PreSpeechReader.get_instance()
