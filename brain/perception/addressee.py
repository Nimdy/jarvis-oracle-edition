"""Addressee / Directedness Gate.

Determines whether a transcribed utterance is actually addressed to Jarvis
vs. overheard speech directed at another person or self-talk.

Design:
  - Runs BEFORE response generation, memory creation, capability gate, etc.
  - Fast heuristic classifier (no LLM, < 1ms)
  - Three-tier output: addressed, not_addressed, uncertain
  - Follow-up speech within an active conversation is presumed addressed
  - Explicit negation patterns ("not talking to you") are authoritative
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_TELEMETRY_SIZE = 50

# ── Explicit negation: authoritative "not for Jarvis" ──

_NEGATION_PHRASES = (
    "not talking to you",
    "wasn't talking to you",
    "i'm not talking to you",
    "i am not talking to you",
    "wasn't referring to you",
    "not referring to you",
    "i wasn't referring to you",
    "not speaking to you",
    "wasn't speaking to you",
    "disregard that",
    "disregard what i said",
    "ignore that",
    "ignore what i said",
    "don't listen to that",
    "dont listen to that",
    "do not listen to that",
    "stop listening",
    "stop listening to that",
    "stop listening now",
    "i'm talking to somebody else",
    "i'm talking to someone else",
    "i was talking to somebody else",
    "i was talking to someone else",
    "talking to someone else",
    "talking to somebody else",
    "that wasn't for you",
    "that was not for you",
    "not for you",
    "none of your business",
    "mind your own business",
    "wasn't meant for you",
    "not meant for you",
    "forget what i said",
    "forget that",
    "pretend you didn't hear that",
    "you weren't supposed to hear that",
)

_NEGATION_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in _NEGATION_PHRASES)
    + r")\b",
    re.IGNORECASE,
)

# ── Dismissive post-response patterns ──
# These indicate the user is annoyed about a previous response, NOT requesting a new one

_DISMISSIVE_PHRASES = (
    "waste of time",
    "waste of my time",
    "that was useless",
    "that was pointless",
    "that was a waste",
    "you wasted my time",
    "everything you said was",
    "i didn't ask you",
    "i didn't ask for that",
    "who asked you",
    "nobody asked you",
    "nobody asked",
    "shut up already",
    "stop talking",
    "don't respond",
    "don't reply",
    "don't answer",
)

_DISMISSIVE_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(p) for p in _DISMISSIVE_PHRASES)
    + r")\b",
    re.IGNORECASE,
)

# ── Positive addressedness signals ──

_JARVIS_NAME_RE = re.compile(r"\bjarvis\b", re.IGNORECASE)
_COMMAND_RE = re.compile(
    r"^(?:hey|ok|okay)?\s*(?:jarvis|can you|could you|would you|please|tell me|what is|what's|how do|how does|where is|when did|who is)",
    re.IGNORECASE,
)
_SECOND_PERSON_RE = re.compile(
    r"\b(?:you(?:'re|r| are| can| should| could| would| do| did| have| were| know)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AddresseeResult:
    addressed: bool  # True = addressed to Jarvis, False = not
    confidence: float  # 0.0-1.0
    reason: str  # human-readable explanation
    suppressed: bool  # True = should suppress response entirely


_MISADDRESSED_DECAY_WINDOW_S = 30.0
_MAX_CONSECUTIVE_MISADDRESSED = 3


class AddresseeGate:
    """Stateful addressee classifier with recent-context memory."""

    def __init__(self) -> None:
        self._recent_negations: list[float] = []
        self._misaddressed_count: int = 0
        self._total_checked: int = 0
        self._total_suppressed: int = 0
        self._total_follow_up: int = 0
        self._total_wake_addressed: int = 0
        self._telemetry: deque[dict[str, Any]] = deque(maxlen=_TELEMETRY_SIZE)

    def check(
        self,
        text: str,
        is_follow_up: bool = False,
        speaker_name: str = "unknown",
        had_wake_word: bool = True,
    ) -> AddresseeResult:
        """Classify whether an utterance is addressed to Jarvis.

        Args:
            text: The transcribed utterance
            is_follow_up: Whether this is within a follow-up conversation window
            speaker_name: Current identified speaker
            had_wake_word: Whether the wake word triggered this session
        """
        self._total_checked += 1
        if is_follow_up:
            self._total_follow_up += 1
        lower = text.strip().lower()
        has_name = bool(_JARVIS_NAME_RE.search(lower))

        # 1. Explicit negation is authoritative — highest priority
        if _NEGATION_RE.search(lower):
            self._record_negation()
            self._total_suppressed += 1
            result = AddresseeResult(addressed=False, confidence=0.95,
                                     reason="explicit_negation", suppressed=True)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 2. Dismissive post-response complaint — suppress and don't create new work
        if _DISMISSIVE_RE.search(lower):
            self._total_suppressed += 1
            result = AddresseeResult(addressed=False, confidence=0.85,
                                     reason="dismissive_complaint", suppressed=True)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 3. Direct Jarvis name mention — strong positive signal
        if has_name:
            result = AddresseeResult(addressed=True, confidence=0.95,
                                     reason="name_mention", suppressed=False)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 4. Follow-up within active conversation — presume addressed
        #    Wake word required for full confidence; without it, still contextual
        if is_follow_up:
            conf = 0.80 if had_wake_word else 0.70
            reason = "follow_up_conversation" if had_wake_word else "follow_up_contextual"
            result = AddresseeResult(addressed=True, confidence=conf,
                                     reason=reason, suppressed=False)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 5. Command-framed speech — likely addressed
        if _COMMAND_RE.match(lower):
            result = AddresseeResult(addressed=True, confidence=0.80,
                                     reason="command_framing", suppressed=False)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 6. Wake word session with second-person reference — probably addressed
        if had_wake_word and _SECOND_PERSON_RE.search(lower):
            self._total_wake_addressed += 1
            result = AddresseeResult(addressed=True, confidence=0.70,
                                     reason="wake_word_second_person", suppressed=False)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 7. Wake word alone — default presume addressed (standard UX)
        if had_wake_word:
            self._total_wake_addressed += 1
            result = AddresseeResult(addressed=True, confidence=0.60,
                                     reason="wake_word_default", suppressed=False)
            self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
            return result

        # 8. No wake word, no follow-up — uncertain
        result = AddresseeResult(addressed=False, confidence=0.50,
                                 reason="no_addressing_signal", suppressed=False)
        self._record_telemetry(text, result, had_wake_word, has_name, is_follow_up)
        return result

    def _record_negation(self) -> None:
        now = time.time()
        self._recent_negations.append(now)
        cutoff = now - _MISADDRESSED_DECAY_WINDOW_S
        self._recent_negations = [t for t in self._recent_negations if t > cutoff]
        self._misaddressed_count += 1

    @property
    def recent_negation_count(self) -> int:
        now = time.time()
        cutoff = now - _MISADDRESSED_DECAY_WINDOW_S
        return sum(1 for t in self._recent_negations if t > cutoff)

    def _record_telemetry(
        self,
        text: str,
        result: AddresseeResult,
        had_wake_word: bool,
        had_name_mention: bool,
        is_follow_up: bool,
    ) -> None:
        self._telemetry.append({
            "text_snippet": text[:60],
            "result": result.reason,
            "addressed": result.addressed,
            "confidence": round(result.confidence, 2),
            "would_have_been_blocked": result.suppressed,
            "was_response_generated": None,  # filled later by perception_orchestrator
            "had_wake_word": had_wake_word,
            "had_name_mention": had_name_mention,
            "is_follow_up": is_follow_up,
            "timestamp": time.time(),
        })

    def mark_response_generated(self, generated: bool) -> None:
        """Called after conversation completes to record if a response was actually produced."""
        if self._telemetry:
            self._telemetry[-1]["was_response_generated"] = generated

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_checked": self._total_checked,
            "total_suppressed": self._total_suppressed,
            "total_follow_up": self._total_follow_up,
            "total_wake_addressed": self._total_wake_addressed,
            "misaddressed_count": self._misaddressed_count,
            "recent_negations": self.recent_negation_count,
            "recent_decisions": list(self._telemetry),
        }
