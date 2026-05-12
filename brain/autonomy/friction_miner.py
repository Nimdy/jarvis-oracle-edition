"""Conversation Friction Miner — Phase 5.1a.

Extracts learning signals from user interactions where the response failed to
satisfy. Produces first-class FrictionEvent objects that feed into:
  1. metric_triggers (8th deficit dimension: friction_rate)
  2. language corpus (negative examples)
  3. autonomy scoring (friction clusters inform research priority)

Severity model:
  low      — style annoyance, verbosity preference
  medium   — routing miss, verbosity miss, too cautious/generic
  high     — grounding miss, factual mismatch
  critical — identity/scope leak
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("autonomy.friction")

_MAX_EVENTS = 500
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
_PERSISTENCE_PATH = os.path.expanduser("~/.jarvis/friction_events.jsonl")
_TEXT_TRUNCATE = 300


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class FrictionEvent:
    id: str
    timestamp: float
    conversation_id: str
    episode_id: str = ""
    route_class: str = ""
    response_class: str = ""
    friction_type: str = ""
    severity: str = "low"
    assistant_text: str = ""
    user_text: str = ""
    candidate_rewrite: str = ""
    cluster_key: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrictionEvent:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

_CORRECTION_PHRASES = [
    re.compile(r"\bthat'?s?\s+(?:not\s+(?:right|correct|true|accurate)|wrong|incorrect)\b", re.I),
    re.compile(r"\bno[,.]?\s+(?:it'?s?\s+(?:actually|not)|that'?s?\s+not|i\s+(?:said|meant))\b", re.I),
    re.compile(r"\byou\s+(?:misunderstood|got\s+(?:it|that)\s+wrong|are\s+wrong)\b", re.I),
    re.compile(r"\bi\s+(?:never\s+said|didn'?t\s+(?:say|mean))\b", re.I),
    re.compile(r"\b(?:actually|correction)[,:]?\s+", re.I),
]

_REPHRASE_SIGNALS = [
    re.compile(r"\bwhat\s+i\s+(?:meant|mean)\s+(?:is|was)\b", re.I),
    re.compile(r"\blet\s+me\s+(?:rephrase|try\s+again|say\s+(?:it|that))\b", re.I),
    re.compile(r"\bi\s+(?:said|asked|meant)\b", re.I),
]

_ANNOYANCE_PHRASES = [
    re.compile(r"\bi\s+already\s+(?:told|said|asked)\b", re.I),
    re.compile(r"\bthat'?s?\s+(?:not\s+what\s+i\s+asked|the\s+wrong)\b", re.I),
    re.compile(r"\bstop\s+(?:it|that|doing\s+that|repeating)\b", re.I),
    re.compile(r"\byou'?re?\s+(?:not\s+listening|ignoring)\b", re.I),
    re.compile(r"\bwhy\s+(?:do\s+you\s+keep|can'?t\s+you)\b", re.I),
]

_DISSATISFACTION_PHRASES = [
    re.compile(r"\bnever\s*mind\b", re.I),
    re.compile(r"\bforget\s+(?:it|about\s+it)\b", re.I),
    re.compile(r"\bjust\s+(?:stop|forget|drop\s+it)\b", re.I),
]

_TERSE_MAX_WORDS = 3
_TERSE_NEGATIVE_WORDS = frozenset({
    "no", "nope", "nah", "wrong", "stop", "bad", "ugh", "whatever",
    "sure", "ok", "fine", "meh", "huh",
})

_VERBOSITY_SIGNALS = [
    re.compile(r"\btoo\s+(?:much|long|verbose|wordy)\b", re.I),
    re.compile(r"\bget\s+to\s+the\s+point\b", re.I),
    re.compile(r"\bjust\s+(?:answer|tell\s+me)\b", re.I),
    re.compile(r"\btl;?dr\b", re.I),
]

_IDENTITY_SCOPE_SIGNALS = [
    re.compile(r"\bthat'?s?\s+(?:not\s+(?:me|mine|my)|someone\s+else)\b", re.I),
    re.compile(r"\byou'?re?\s+(?:confusing|mixing)\s+(?:me|us)\b", re.I),
    re.compile(r"\bthat'?s?\s+(?:about|for)\s+(?:a\s+different|someone|another)\b", re.I),
]

_OVERCAUTION_SIGNALS = [
    re.compile(r"\bjust\s+(?:answer|tell\s+me)\b", re.I),
    re.compile(r"\byou\s+(?:can|should)\s+(?:just|simply)\b", re.I),
    re.compile(r"\bdon'?t\s+(?:overthink|over\s*complicate|apologize)\b", re.I),
    re.compile(r"\bi\s+know\s+(?:you\s+can|that)\b", re.I),
]


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text or "") for p in patterns)


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, str] = {
    "correction": "high",
    "rephrase": "medium",
    "terse_rejection": "medium",
    "annoyance": "high",
    "overexplaining": "low",
    "wrong_route": "medium",
    "too_cautious": "medium",
    "too_generic": "medium",
    "grounding_failure": "high",
    "identity_mismatch": "critical",
    "unwanted_followup": "low",
    "dissatisfaction": "medium",
}


# ---------------------------------------------------------------------------
# FrictionMiner
# ---------------------------------------------------------------------------

class FrictionMiner:
    """Detects and clusters conversation friction events."""

    def __init__(self) -> None:
        self._events: deque[FrictionEvent] = deque(maxlen=_MAX_EVENTS)
        self._clusters: dict[str, list[float]] = {}
        self._loaded = False

    # -- Detection ----------------------------------------------------------

    def detect(
        self,
        user_text: str,
        assistant_text: str,
        route_class: str,
        response_class: str,
        correction_result: dict[str, Any] | None,
        conversation_id: str,
        episode_id: str = "",
    ) -> FrictionEvent | None:
        """Detect friction from user follow-up text after a response.

        Returns a FrictionEvent if friction is detected, None otherwise.
        """
        if not user_text or not assistant_text:
            return None

        friction_type = self._classify(user_text, assistant_text, correction_result)
        if not friction_type:
            return None

        severity = _SEVERITY_MAP.get(friction_type, "low")
        cluster_key = f"{route_class}:{friction_type}"
        confidence = self._compute_confidence(user_text, friction_type, correction_result)

        event = FrictionEvent(
            id=f"fr_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            conversation_id=conversation_id,
            episode_id=episode_id,
            route_class=route_class,
            response_class=response_class,
            friction_type=friction_type,
            severity=severity,
            assistant_text=assistant_text[:_TEXT_TRUNCATE],
            user_text=user_text[:_TEXT_TRUNCATE],
            cluster_key=cluster_key,
            confidence=confidence,
        )
        self.cluster(event)
        self._persist(event)
        return event

    def _classify(
        self,
        user_text: str,
        assistant_text: str,
        correction_result: dict[str, Any] | None,
    ) -> str:
        """Classify friction type. Returns empty string if no friction detected."""
        if correction_result:
            kind = correction_result.get("correction_kind", "")
            if kind == "identity_scope_leak":
                return "identity_mismatch"
            return "correction"

        if _matches_any(user_text, _IDENTITY_SCOPE_SIGNALS):
            return "identity_mismatch"

        if _matches_any(user_text, _CORRECTION_PHRASES):
            return "correction"

        if _matches_any(user_text, _ANNOYANCE_PHRASES):
            return "annoyance"

        if _matches_any(user_text, _DISSATISFACTION_PHRASES):
            return "dissatisfaction"

        if _matches_any(user_text, _REPHRASE_SIGNALS):
            return "rephrase"

        if _matches_any(user_text, _VERBOSITY_SIGNALS):
            return "overexplaining"

        if _matches_any(user_text, _OVERCAUTION_SIGNALS):
            return "too_cautious"

        wc = _word_count(user_text)
        if wc <= _TERSE_MAX_WORDS:
            words = set(user_text.lower().split())
            if words & _TERSE_NEGATIVE_WORDS:
                return "terse_rejection"

        return ""

    def _compute_confidence(
        self,
        user_text: str,
        friction_type: str,
        correction_result: dict[str, Any] | None,
    ) -> float:
        if correction_result:
            return 0.90
        if friction_type in ("identity_mismatch", "correction", "annoyance"):
            return 0.80
        if friction_type in ("rephrase", "dissatisfaction"):
            return 0.65
        return 0.50

    # -- Clustering ---------------------------------------------------------

    def cluster(self, event: FrictionEvent) -> None:
        """Track event in rolling cluster tracker."""
        self._events.append(event)
        key = event.cluster_key
        if key not in self._clusters:
            self._clusters[key] = []
        self._clusters[key].append(event.timestamp)
        cutoff = time.time() - 86400
        self._clusters[key] = [t for t in self._clusters[key] if t > cutoff]

    def get_active_clusters(self) -> list[dict[str, Any]]:
        """Return clusters with 2+ events in the last 24h, sorted by count."""
        now = time.time()
        cutoff = now - 86400
        active = []
        for key, timestamps in self._clusters.items():
            recent = [t for t in timestamps if t > cutoff]
            if len(recent) >= 2:
                parts = key.split(":", 1)
                active.append({
                    "cluster_key": key,
                    "route_class": parts[0] if parts else "",
                    "friction_type": parts[1] if len(parts) > 1 else "",
                    "count_24h": len(recent),
                    "last_event": max(recent),
                })
        active.sort(key=lambda c: c["count_24h"], reverse=True)
        return active

    def get_friction_rate(self, window_s: float = 3600.0) -> float:
        """Friction events per conversation in the given window."""
        now = time.time()
        cutoff = now - window_s
        events_in_window = sum(1 for e in self._events if e.timestamp > cutoff)
        convs = len({e.conversation_id for e in self._events if e.timestamp > cutoff})
        if convs == 0:
            return 0.0
        return events_in_window / convs

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent friction events for dashboard/API."""
        events = list(self._events)[-limit:]
        return [e.to_dict() for e in reversed(events)]

    def get_stats(self) -> dict[str, Any]:
        """Return summary stats for dashboard."""
        now = time.time()
        h1 = sum(1 for e in self._events if e.timestamp > now - 3600)
        h24 = sum(1 for e in self._events if e.timestamp > now - 86400)
        by_type: dict[str, int] = {}
        for e in self._events:
            by_type[e.friction_type] = by_type.get(e.friction_type, 0) + 1
        return {
            "total_events": len(self._events),
            "events_1h": h1,
            "events_24h": h24,
            "friction_rate_1h": self.get_friction_rate(3600),
            "active_clusters": len(self.get_active_clusters()),
            "by_type": by_type,
        }

    # -- Persistence --------------------------------------------------------

    def _persist(self, event: FrictionEvent) -> None:
        """Append event to JSONL file with rotation."""
        try:
            os.makedirs(os.path.dirname(_PERSISTENCE_PATH), exist_ok=True)
            if os.path.exists(_PERSISTENCE_PATH):
                sz = os.path.getsize(_PERSISTENCE_PATH)
                if sz > _MAX_FILE_BYTES:
                    rotated = _PERSISTENCE_PATH + ".1"
                    if os.path.exists(rotated):
                        os.remove(rotated)
                    os.rename(_PERSISTENCE_PATH, rotated)
            with open(_PERSISTENCE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to persist friction event: %s", exc)

    def load(self) -> None:
        """Load persisted friction events on startup."""
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(_PERSISTENCE_PATH):
            return
        try:
            with open(_PERSISTENCE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        event = FrictionEvent.from_dict(d)
                        self._events.append(event)
                        key = event.cluster_key
                        if key not in self._clusters:
                            self._clusters[key] = []
                        self._clusters[key].append(event.timestamp)
                    except Exception:
                        continue
            cutoff = time.time() - 86400
            for key in list(self._clusters):
                self._clusters[key] = [t for t in self._clusters[key] if t > cutoff]
                if not self._clusters[key]:
                    del self._clusters[key]
            logger.info("Loaded %d friction events from disk", len(self._events))
        except Exception as exc:
            logger.warning("Failed to load friction events: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: FrictionMiner | None = None


def get_friction_miner() -> FrictionMiner:
    global _instance
    if _instance is None:
        _instance = FrictionMiner()
        _instance.load()
    return _instance
