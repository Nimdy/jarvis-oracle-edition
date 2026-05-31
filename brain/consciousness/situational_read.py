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

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

# Salience gate threshold. In P0 it gates NOTHING — the read is logged-only.
# It is recorded so the threshold can be validated against real conversation
# before any later phase is allowed to act on it.
_SALIENCE_THRESHOLD = 0.5

# Affect cortisol above this reads as conversational tension worth noting.
_CORTISOL_TENSION = 0.45
# A long reply to a simple turn is the classic "am I overexplaining?" tell.
_OVEREXPLAIN_WORDS = 90

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
            "authority": self.authority,
        }


def _word_count(text: str) -> int:
    return len((text or "").split())


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

        # ── salience: how notable is this moment? (the behavior gate, shadow) ──
        salience = 0.0
        if sentiment == "negative":
            salience = max(salience, 0.7)
        if engagement == "disengaging":
            salience = max(salience, 0.6)
        if self_check != "reply length proportionate":
            salience = max(salience, 0.55)
        if cortisol is not None and cortisol >= _CORTISOL_TENSION:
            salience = max(salience, min(1.0, 0.4 + cortisol))
        if latency_ms and latency_ms > 8000:
            salience = max(salience, 0.5)
            evidence.append(["latency_ms", latency_ms])
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
            elif self_check != "reply length proportionate":
                would = "would consider being more concise / checking if this helps"
            elif sentiment == "negative":
                would = "would consider softening tone"
            elif engagement == "disengaging":
                would = "would consider pivoting or asking if this is useful"
            else:
                would = "would consider acknowledging the shift"

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
        )

    def get_recent(self, n: int = 10) -> list:
        items = list(self._recent)[-n:]
        return [r.to_dict() for r in reversed(items)]

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
            "latest": last,
            "recent": self.get_recent(8),
        }


situational_read_engine = SituationalReadEngine.get_instance()
