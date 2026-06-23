"""Companion Cognition — P0: the live situational read (LOGGED-ONLY / shadow).

JARVIS's internal read of a just-completed exchange: what it thinks is
happening, why, how confident it is, what evidence contributed, and what it
WOULD have done if it had the authority.  This phase is observation only — it
changes no behavior, writes no beliefs, asks nothing.  It exists so the read
itself can be watched and validated BEFORE any of it is allowed to steer a
conversation (the read->behavior ladder, P3+).

Honesty contract (mirrors affect_state §7 / the spark grounding ring):
  - Every read is a HYPOTHESIS, never a fact (confidence-scored).
  - Every signal that contributed is named in ``evidence`` (label -> value).
  - The salience/affect gate that WOULD trip behavior is recorded but NO-OP
    here (David's anti-chatterbox spine: "only when salience/affect trips").
    Recording it in P0 lets us validate the threshold against real
    conversation before it is ever allowed to act.
  - ``would_have_done`` is the SHADOW of a future behavior — logged, not taken.

Pure-Python, dependency-free, near-zero cost: it aggregates signals already
computed for the turn (no model call), so it is safe to run inline in the
conversation path.  Singleton: ``situational_read_engine``.

See docs/COMPANION_COGNITION_DESIGN.md (P0).
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

# Separate shadow-store persistence (NOT soul/identity — no pollution). Lets reads
# survive restarts so the P0->P1 earn-gate can actually accumulate its reps.
_STATE_DIR = os.path.expanduser("~/.jarvis")
_STATE_PATH = os.path.join(_STATE_DIR, "companion_situational_read.json")

# Salience gate threshold. In P0 it gates NOTHING — the read is logged-only.
# It is recorded so the threshold can be validated against real conversation
# before any later phase is allowed to act on it.
_SALIENCE_THRESHOLD = 0.5

# Companion P0->P1 gate (FINISH_ROADMAP #2): minimum reads before P1 has enough
# data to model from. A STRUCTURAL proxy only — the true read-validity gate is
# EARNED via transcript review, never coded.
_P0P1_MIN_TURNS = 30

# A long reply to a simple turn is the classic "am I overexplaining?" tell.
_OVEREXPLAIN_WORDS = 90
# A turn that took this long to answer is conversationally notable (the user waited).
_SLOW_TURN_MS = 8000

_NEGATIVE_EMOTIONS = frozenset(
    {"angry", "frustrated", "sad", "annoyed", "anxious", "stressed", "upset", "fear", "disgust"}
)
_POSITIVE_EMOTIONS = frozenset(
    {"happy", "excited", "amused", "grateful", "calm", "content", "playful", "joy"}
)


@dataclass
class SituationalRead:
    """JARVIS's internal read of one completed exchange. A hypothesis, logged only."""

    timestamp: float
    speaker: str
    # The read (all hypotheses, never asserted as fact):
    engagement: str              # "engaged" | "neutral" | "disengaging" | "unknown"
    user_sentiment: str          # "positive" | "neutral" | "negative"
    self_check: str              # JARVIS's read on its OWN turn (e.g. overexplaining)
    confidence: float            # 0..1 — how sure JARVIS is of this read
    evidence: list               # [[signal_label, value], ...] — what contributed
    # The behavior gate (shadow — recorded, never acted on in P0):
    salience: float              # 0..1 — how notable this moment is
    salience_tripped: bool       # would the gate have opened?
    would_have_done: "str | None"  # the action it WOULD take, if allowed (logged, not taken)
    # ── positive / warmth axis (SHADOW) — so the read is NOT corrective-only ──
    warmth_noted: bool = False              # a well-LANDED exchange (positive + engaged + follow-up)
    warm_would_have_done: "str | None" = None
    humor_attempted: bool = False           # structural tag: JARVIS's own turn looked playful (evidence only)
    authority: str = "shadow_logged_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "engagement": self.engagement,
            "user_sentiment": self.user_sentiment,
            "self_check": self.self_check,
            "confidence": round(self.confidence, 3),
            "evidence": [list(e) for e in self.evidence],
            "salience": round(self.salience, 3),
            "salience_tripped": self.salience_tripped,
            "would_have_done": self.would_have_done,
            "warmth_noted": self.warmth_noted,
            "warm_would_have_done": self.warm_would_have_done,
            "humor_attempted": self.humor_attempted,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SituationalRead":
        return cls(
            timestamp=d.get("timestamp", 0.0),
            speaker=d.get("speaker", "unknown"),
            engagement=d.get("engagement", "unknown"),
            user_sentiment=d.get("user_sentiment", "neutral"),
            self_check=d.get("self_check", ""),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            evidence=[list(e) for e in (d.get("evidence") or [])],
            salience=float(d.get("salience", 0.0) or 0.0),
            salience_tripped=bool(d.get("salience_tripped", False)),
            would_have_done=d.get("would_have_done"),
            warmth_noted=bool(d.get("warmth_noted", False)),
            warm_would_have_done=d.get("warm_would_have_done"),
            humor_attempted=bool(d.get("humor_attempted", False)),
            authority=d.get("authority", "shadow_logged_only"),
        )


def _word_count(text: str) -> int:
    return len((text or "").split())


def _looks_playful(text: str) -> bool:
    """Coarse STRUCTURAL tag: did JARVIS's OWN turn look playful? Evidence only — asserts nothing.
    Deliberately conservative (few false positives); a real banter-tone signal can replace it later."""
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in ("haha", "lol", ";)", ":)", "😄", "😉", "😂", "just kidding", "teasing"))


class SituationalReadEngine:
    """Builds + logs a SituationalRead per observed turn. Behavior-free (P0)."""

    _instance: "SituationalReadEngine | None" = None

    @classmethod
    def get_instance(cls) -> "SituationalReadEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._recent: deque = deque(maxlen=50)
        self._total = 0
        self._tripped = 0
        self._load()

    def _save(self) -> None:
        """Persist the shadow store to its OWN file (not identity). Best-effort."""
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            data = {"total": self._total, "tripped": self._tripped,
                    "recent": [r.to_dict() for r in self._recent]}
            tmp = _STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, _STATE_PATH)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            with open(_STATE_PATH) as f:
                data = json.load(f)
            self._total = int(data.get("total", 0))
            self._tripped = int(data.get("tripped", 0))
            for rd in data.get("recent", []):
                self._recent.append(SituationalRead.from_dict(rd))
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def observe_turn(
        self,
        *,
        speaker: str = "unknown",
        user_text: str = "",
        response_text: str = "",
        user_emotion: str = "neutral",
        follow_up: bool = False,
        latency_ms: int = 0,
        complexity: str = "simple",
        route: str = "",
        affect: Any | None = None,
    ) -> "SituationalRead | None":
        """Observe a completed exchange and log JARVIS's read of it. No side effects."""
        try:
            read = self._build(
                speaker=speaker or "unknown",
                user_text=user_text or "",
                response_text=response_text or "",
                user_emotion=(user_emotion or "neutral").lower(),
                follow_up=bool(follow_up),
                latency_ms=int(latency_ms or 0),
                complexity=complexity or "simple",
                route=route or "",
                affect=affect,
            )
        except Exception:
            return None
        self._recent.append(read)
        self._total += 1
        if read.salience_tripped:
            self._tripped += 1
        self._save()
        return read

    def _build(self, *, speaker, user_text, response_text, user_emotion,
               follow_up, latency_ms, complexity, route, affect) -> SituationalRead:
        evidence: list = []

        # ── user sentiment (hypothesis from the trusted emotion label) ──
        if user_emotion in _NEGATIVE_EMOTIONS:
            sentiment = "negative"
        elif user_emotion in _POSITIVE_EMOTIONS:
            sentiment = "positive"
        else:
            sentiment = "neutral"
        evidence.append(["user_emotion", user_emotion])

        # ── engagement (length of the user's turn + follow-up continuity) ──
        u_words = _word_count(user_text)
        evidence.append(["user_turn_words", u_words])
        evidence.append(["follow_up", follow_up])
        if u_words == 0:
            engagement = "unknown"
        elif follow_up and u_words >= 4:
            engagement = "engaged"
        elif u_words <= 2 and not follow_up:
            engagement = "disengaging"
        else:
            engagement = "neutral"

        # ── self-check on JARVIS's OWN turn (am I overexplaining?) ──
        r_words = _word_count(response_text)
        evidence.append(["response_words", r_words])
        evidence.append(["complexity", complexity])
        if r_words >= _OVEREXPLAIN_WORDS * 2:
            self_check = "very long reply — watch for overexplaining"
        elif r_words >= _OVEREXPLAIN_WORDS and complexity == "simple":
            self_check = "may be overexplaining (long reply to a simple turn)"
        else:
            self_check = "reply length proportionate"

        # ── affect: read cortisol (tension) / dopamine (engagement reward) ──
        cortisol = dopamine = None
        if affect is not None:
            try:
                cortisol = float(getattr(affect.cortisol, "level", 0.0))
                dopamine = float(getattr(affect.dopamine, "level", 0.0))
                evidence.append(["affect_cortisol", round(cortisol, 3)])
                evidence.append(["affect_dopamine", round(dopamine, 3)])
            except Exception:
                cortisol = dopamine = None

        # ── salience: is THIS exchange notable? (the behavior gate, shadow) ──
        # Driven by CONVERSATIONAL signals only. The affect cortisol/dopamine
        # above are JARVIS's OWN internal state (epistemic friction / reward),
        # not a read of the conversation — cortisol's floor is the 0.5 baseline,
        # so it must never trip a conversational gate. They are kept as context
        # in `evidence`; a genuinely turn-scoped affect signal can feed salience
        # in a later phase, explicitly labelled as such.
        slow_turn = bool(latency_ms and latency_ms > _SLOW_TURN_MS)
        if slow_turn:
            evidence.append(["latency_ms", latency_ms])
        overexplain = self_check != "reply length proportionate"
        salience = 0.0
        if sentiment == "negative":
            salience = max(salience, 0.7)
        if engagement == "disengaging":
            salience = max(salience, 0.6)
        if overexplain:
            salience = max(salience, 0.55)
        if slow_turn:
            salience = max(salience, 0.5)
        salience = max(0.0, min(1.0, salience))
        tripped = salience >= _SALIENCE_THRESHOLD

        # ── confidence: a trusted emotion + clearer signals raise it ──
        confidence = 0.35
        if user_emotion != "neutral":
            confidence += 0.2
        if u_words >= 4:
            confidence += 0.1
        if affect is not None:
            confidence += 0.1
        confidence = max(0.0, min(1.0, confidence))

        # ── would_have_done: the SHADOW of a future behavior (logged, NOT taken) ──
        would = None
        if tripped:
            if sentiment == "negative" and engagement == "disengaging":
                would = "would consider giving space / checking in"
            elif overexplain:
                would = "would consider being more concise / checking if this helps"
            elif sentiment == "negative":
                would = "would consider softening tone"
            elif engagement == "disengaging":
                would = "would consider pivoting or asking if this is useful"
            elif slow_turn:
                would = "would consider acknowledging the delay"

        # ── POSITIVE / warmth axis (SHADOW): a well-LANDED exchange (positive + engaged + follow-up),
        # so the read STOPS being corrective-only. It does NOT raise `salience` above — kept OFF the
        # corrective gate so it never dominates a withdraw/disengage read; behavior_advisory counts it
        # on a SEPARATE positive axis. humor_attempted is a structural tag from JARVIS's OWN turn.
        warmth_noted = bool(sentiment == "positive" and engagement == "engaged" and follow_up)
        warm_would = ("would consider mirroring the warmth / leaning into the rapport"
                      if warmth_noted else None)
        humor_attempted = _looks_playful(response_text)
        if warmth_noted:
            evidence.append(["warmth_landed", True])
        if humor_attempted:
            evidence.append(["humor_attempted", True])

        return SituationalRead(
            timestamp=time.time(),
            speaker=speaker,
            engagement=engagement,
            user_sentiment=sentiment,
            self_check=self_check,
            confidence=confidence,
            evidence=evidence,
            salience=salience,
            salience_tripped=tripped,
            would_have_done=would,
            warmth_noted=warmth_noted,
            warm_would_have_done=warm_would,
            humor_attempted=humor_attempted,
        )

    def get_recent(self, n: int = 10) -> list:
        items = list(self._recent)[-n:]
        return [r.to_dict() for r in reversed(items)]

    def _promotion_readiness(self) -> dict[str, Any]:
        """Companion P0->P1 gate (FINISH_ROADMAP #2): a STRUCTURAL readiness proxy.
        Signals only that enough reads accumulated and the salience trigger is not
        pinned off/on. The TRUE read-validity gate (reads coherent vs transcript
        reality) is EARNED via operator/transcript review — never coded."""
        total = self._total
        rate = (self._tripped / total) if total else 0.0
        enough = total >= _P0P1_MIN_TURNS
        sane_band = 0.02 <= rate <= 0.70
        blocking = []
        if not enough:
            blocking.append("need %d reads (have %d)" % (_P0P1_MIN_TURNS, total))
        if total and not sane_band:
            blocking.append("trigger-rate %.2f outside 0.02-0.70 (pinned?)" % rate)
        return {
            "gate": "P0->P1",
            "would_promote_to": "P1_theory_of_mind",
            "structural_ready": bool(enough and sane_band),
            "note": "structural proxy only; true read-validity is EARNED via transcript review",
            "observed_turns": total,
            "trigger_rate": round(rate, 3),
            "min_turns": _P0P1_MIN_TURNS,
            "blocking": blocking,
        }

    def get_status(self) -> dict[str, Any]:
        last = self._recent[-1].to_dict() if self._recent else None
        return {
            "phase": "P0_situational_read",
            "authority": "shadow_logged_only",
            "changes_behavior": False,
            "writes_beliefs": False,
            "asks": False,
            "salience_threshold": _SALIENCE_THRESHOLD,
            "observed_turns": self._total,
            "salience_tripped_count": self._tripped,
            "promotion_readiness": self._promotion_readiness(),
            "latest": last,
            "recent": self.get_recent(8),
        }


situational_read_engine = SituationalReadEngine.get_instance()
