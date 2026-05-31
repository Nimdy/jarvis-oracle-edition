"""Companion Cognition — P1: theory-of-mind (SHADOW / hypotheses-only).

A per-person, rolling model of what JARVIS *infers* about someone it talks with:
how they tend to engage, what they seem to be feeling, how responsive they are —
held strictly as CONFIDENCE-SCORED HYPOTHESES, never facts. Built by accumulating
the P0 situational reads (one read per turn, see consciousness/situational_read.py)
into a per-person picture.

Honesty / no-pollution contract (the lesson from the interaction_review fix):
  - Hypotheses, never facts. Every field is confidence-scored and labelled shadow.
  - SHADOW: gates no behavior, writes no beliefs, asks nothing.
  - Kept in a SEPARATE in-memory shadow store. It does NOT mutate the persisted
    soul Relationship / IdentityState — unvalidated inferences must never pollute
    persisted identity (the same reason interaction_review is belief-ineligible).
    Cross-session persistence is a LATER step, gated on this model proving it
    forms coherent hypotheses against transcript reality (the P1-earned gate).

Pure-stdlib, near-zero cost (aggregates the read fields already computed for the
turn — no model call). Singleton: ``theory_of_mind_engine``.
See docs/COMPANION_COGNITION_DESIGN.md (P1) and docs/FINISH_ROADMAP.md (#3-4).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# A model needs a few corroborating reads before its hypotheses mean anything.
_MIN_SAMPLES_FOR_CONFIDENCE = 6
# Recent-window the rolling disposition is computed over.
_RECENT_WINDOW = 20

_ENGAGEMENT_PHRASE = {
    "engaged": "tends to be engaged",
    "neutral": "fairly neutral / steady",
    "disengaging": "tends to pull back / disengage",
    "unknown": "unclear",
}


@dataclass
class PersonModel:
    """JARVIS's rolling, hypothesis-only read of one person. Never asserted as fact."""

    name: str
    sample_count: int = 0
    # Rolling hypotheses (each paired with a confidence in [0,1]):
    disposition: str = "forming — not enough reads yet"
    disposition_confidence: float = 0.0
    current_feeling: str = "unknown"
    feeling_confidence: float = 0.0
    responsiveness: str = "unknown"          # "responsive" | "mixed" | "withdrawn"
    consistency: float = 0.0                  # how consistent recent reads are (0..1)
    last_updated: float = 0.0
    _eng_counts: dict = field(default_factory=dict)
    _sent_counts: dict = field(default_factory=dict)
    _recent: deque = field(default_factory=lambda: deque(maxlen=_RECENT_WINDOW))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "disposition": self.disposition,
            "disposition_confidence": round(self.disposition_confidence, 3),
            "current_feeling": self.current_feeling,
            "feeling_confidence": round(self.feeling_confidence, 3),
            "responsiveness": self.responsiveness,
            "consistency": round(self.consistency, 3),
            "last_updated": self.last_updated,
            "hypothesis": True,
            "authority": "shadow_logged_only",
        }


def _top(counts: dict) -> tuple[str, int, int]:
    """Return (top_key, top_count, total)."""
    total = sum(counts.values()) or 0
    if not counts:
        return ("", 0, 0)
    k = max(counts, key=lambda x: counts[x])
    return (k, counts[k], total)


class TheoryOfMindEngine:
    """Builds + holds per-person theory-of-mind from situational reads. SHADOW."""

    _instance: "TheoryOfMindEngine | None" = None

    @classmethod
    def get_instance(cls) -> "TheoryOfMindEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._people: dict[str, PersonModel] = {}
        self._observations = 0

    def observe(self, speaker: str, read: Any) -> "PersonModel | None":
        """Fold one situational read into the speaker's model. No side effects."""
        try:
            name = (speaker or "unknown").strip() or "unknown"
            if name.lower() == "unknown":
                return None  # don't model an unidentified speaker
            eng = getattr(read, "engagement", None) or "unknown"
            sent = getattr(read, "user_sentiment", None) or "neutral"
            conf = float(getattr(read, "confidence", 0.0) or 0.0)
            return self._update(name, eng, sent, conf)
        except Exception:
            return None

    def _update(self, name: str, engagement: str, sentiment: str,
                read_confidence: float) -> PersonModel:
        pm = self._people.get(name)
        if pm is None:
            pm = PersonModel(name=name)
            self._people[name] = pm
        pm.sample_count += 1
        self._observations += 1
        pm.last_updated = time.time()
        pm._recent.append((engagement, sentiment))
        # Tally over the recent window only (rolling — recent behavior weighs more).
        eng_counts: dict = {}
        sent_counts: dict = {}
        for e, s in pm._recent:
            eng_counts[e] = eng_counts.get(e, 0) + 1
            sent_counts[s] = sent_counts.get(s, 0) + 1
        pm._eng_counts = eng_counts
        pm._sent_counts = sent_counts

        # ── disposition (rolling, from engagement) ──
        top_eng, top_n, total = _top(eng_counts)
        dominance = (top_n / total) if total else 0.0
        pm.consistency = round(dominance, 3)
        ramp = min(1.0, pm.sample_count / _MIN_SAMPLES_FOR_CONFIDENCE)
        if pm.sample_count < 3:
            pm.disposition = "forming — not enough reads yet"
            pm.disposition_confidence = 0.0
        else:
            pm.disposition = _ENGAGEMENT_PHRASE.get(top_eng, "unclear")
            pm.disposition_confidence = round(dominance * ramp, 3)

        # ── current feeling (latest sentiment hypothesis) ──
        pm.current_feeling = sentiment
        pm.feeling_confidence = round(read_confidence, 3)

        # ── responsiveness (engaged vs disengaging balance) ──
        engaged = eng_counts.get("engaged", 0)
        withdrawn = eng_counts.get("disengaging", 0)
        if pm.sample_count < 3:
            pm.responsiveness = "unknown"
        elif engaged > withdrawn:
            pm.responsiveness = "responsive"
        elif withdrawn > engaged:
            pm.responsiveness = "withdrawn"
        else:
            pm.responsiveness = "mixed"
        return pm

    def get_model(self, name: str) -> "dict | None":
        pm = self._people.get((name or "").strip())
        return pm.to_dict() if pm else None

    def get_status(self) -> dict[str, Any]:
        models = sorted(self._people.values(), key=lambda p: p.sample_count, reverse=True)
        return {
            "phase": "P1_theory_of_mind",
            "authority": "shadow_logged_only",
            "changes_behavior": False,
            "writes_beliefs": False,
            "persists_to_identity": False,
            "people_tracked": len(self._people),
            "total_observations": self._observations,
            "min_samples_for_confidence": _MIN_SAMPLES_FOR_CONFIDENCE,
            "models": [m.to_dict() for m in models[:8]],
        }


theory_of_mind_engine = TheoryOfMindEngine.get_instance()
