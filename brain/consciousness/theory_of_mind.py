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

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Separate shadow-store persistence (NOT soul/identity — no pollution). Lets the
# per-person model survive restarts so it can accumulate reps and earn coherence.
_STATE_DIR = os.path.expanduser("~/.jarvis")
_TOM_STATE_PATH = os.path.join(_STATE_DIR, "companion_theory_of_mind.json")

# A model needs a few corroborating reads before its hypotheses mean anything.
_MIN_SAMPLES_FOR_CONFIDENCE = 6
# Recent-window the rolling disposition is computed over.
_RECENT_WINDOW = 20

# Companion P2 — crystallization valve. MIRRORS the real belief gates so a proposal
# is honest about why it isn't a belief yet (contradiction_engine: revisit_count>=50
# AND maturation_score>=0.90; EXTRACTION_DISCARD_THRESHOLD 0.2). P2 only PROPOSES +
# LOGS — it NEVER writes a belief (writing is a later, separately-earned step).
_CRYST_MIN_CORROBORATIONS = 50   # mirrors TensionRecord.revisit_count >= 50
_CRYST_MIN_STABILITY = 0.90      # mirrors maturation_score >= 0.90
_CRYST_MIN_CONFIDENCE = 0.2      # mirrors EXTRACTION_DISCARD_THRESHOLD
_CRYST_PROPOSE_FLOOR = 8         # don't even surface a proposal below this many reads

# Presence-read (the "be there" noticing). Relational first increment: notice when a
# person has shifted from their usual read ("seemed quieter/more pressed lately"),
# logged as a gentle would-note — NEVER spoken (shadow), a HYPOTHESIS, salience-gated
# so it flags a real shift, not noise. Complements (does NOT touch) the novel-object
# curiosity ask. Environmental "the cup moved" half is a later increment (needs the
# spatial memory-of-normal, currently PRE-MATURE).
_PRESENCE_MIN_WINDOW = 8         # need this many recent reads to judge a trend
_PRESENCE_DELTA = 0.30           # recent-vs-baseline shift to count as meaningful
_PRESENCE_ABS = 0.40             # and the recent fraction must itself be this high

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
        self._load()

    def _save(self) -> None:
        """Persist to the shadow store's OWN file (not identity). Best-effort."""
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            people = {}
            for name, pm in self._people.items():
                people[name] = {
                    "name": pm.name, "sample_count": pm.sample_count,
                    "disposition": pm.disposition,
                    "disposition_confidence": pm.disposition_confidence,
                    "current_feeling": pm.current_feeling,
                    "feeling_confidence": pm.feeling_confidence,
                    "responsiveness": pm.responsiveness,
                    "consistency": pm.consistency,
                    "last_updated": pm.last_updated,
                    "recent": [list(t) for t in pm._recent],
                }
            data = {"observations": self._observations, "people": people}
            tmp = _TOM_STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, _TOM_STATE_PATH)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            with open(_TOM_STATE_PATH) as f:
                data = json.load(f)
            for name, s in (data.get("people") or {}).items():
                pm = PersonModel(name=s.get("name", name))
                pm.sample_count = int(s.get("sample_count", 0))
                pm.disposition = s.get("disposition", pm.disposition)
                pm.disposition_confidence = float(s.get("disposition_confidence", 0.0) or 0.0)
                pm.current_feeling = s.get("current_feeling", "unknown")
                pm.feeling_confidence = float(s.get("feeling_confidence", 0.0) or 0.0)
                pm.responsiveness = s.get("responsiveness", "unknown")
                pm.consistency = float(s.get("consistency", 0.0) or 0.0)
                pm.last_updated = float(s.get("last_updated", 0.0) or 0.0)
                for item in (s.get("recent") or []):
                    try:
                        pm._recent.append((item[0], item[1]))
                    except Exception:
                        pass
                self._people[name] = pm
            self._observations = int(data.get("observations", 0))
        except FileNotFoundError:
            pass
        except Exception:
            pass

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
        self._save()
        return pm

    def get_model(self, name: str) -> "dict | None":
        pm = self._people.get((name or "").strip())
        return pm.to_dict() if pm else None

    def get_crystallization_proposals(self) -> list[dict[str, Any]]:
        """Companion P2 (SHADOW): for each STABLE person-model, PROPOSE crystallizing
        it into a relational belief — but NEVER write it. Each proposal is logged
        against the REAL belief gates (>=50 corroborations, >=0.90 stability, conf>=0.2)
        so it's honest about why it isn't a belief yet. Writing is a later earned step;
        P2 only proposes + logs. No belief-graph writes, ever, here."""
        out: list[dict[str, Any]] = []
        for pm in self._people.values():
            if pm.sample_count < _CRYST_PROPOSE_FLOOR:
                continue
            if pm.disposition.startswith("forming"):
                continue
            conf = float(pm.disposition_confidence or 0.0)
            if conf < _CRYST_MIN_CONFIDENCE:
                continue  # below the discard floor — never a belief, however often it recurs
            corrob_ok = pm.sample_count >= _CRYST_MIN_CORROBORATIONS
            stable_ok = pm.consistency >= _CRYST_MIN_STABILITY
            blocking: list[str] = []
            if not corrob_ok:
                blocking.append("corroborations %d/%d" % (pm.sample_count, _CRYST_MIN_CORROBORATIONS))
            if not stable_ok:
                blocking.append("stability %.2f/%.2f" % (pm.consistency, _CRYST_MIN_STABILITY))
            # A durable belief reflects the DOMINANT sentiment over the window, not the
            # latest turn's mood (current_feeling) — else the proposed belief flips per turn.
            dom_feeling = _top(pm._sent_counts)[0] or pm.current_feeling
            out.append({
                "person": pm.name,
                "candidate_belief": "%s — %s; tends to read as %s, %s in conversation" % (
                    pm.name, pm.disposition, dom_feeling, pm.responsiveness),
                "confidence": round(conf, 3),
                "corroborations": pm.sample_count,
                "stability": round(float(pm.consistency or 0.0), 3),
                "would_crystallize": bool(corrob_ok and stable_ok),
                "blocking": blocking,
                "writes_belief": False,
                "status": "shadow_proposed_not_written",
            })
        return out

    def get_presence_observations(self) -> list[dict[str, Any]]:
        """Presence-read (SHADOW): notice when a person has shifted from their usual
        read — log a gentle 'would note', NEVER spoken. A HYPOTHESIS; salience-gated to
        a real recent-vs-baseline shift, not noise. Complements (does NOT touch) the
        novel-object curiosity ask. The 'be there for the person' half."""
        def _frac(window, idx, val):
            return sum(1 for t in window if t[idx] == val) / len(window) if window else 0.0
        out: list[dict[str, Any]] = []
        for pm in self._people.values():
            recent = list(pm._recent)
            if len(recent) < _PRESENCE_MIN_WINDOW:
                continue
            cut = max(3, len(recent) // 3)
            newer, older = recent[-cut:], recent[:-cut]
            if not older:
                continue
            dis_new, dis_old = _frac(newer, 0, "disengaging"), _frac(older, 0, "disengaging")
            neg_new, neg_old = _frac(newer, 1, "negative"), _frac(older, 1, "negative")
            note = None
            if (dis_new - dis_old) >= _PRESENCE_DELTA and dis_new >= _PRESENCE_ABS:
                note = "%s has seemed quieter / more withdrawn than usual lately" % pm.name
            elif (neg_new - neg_old) >= _PRESENCE_DELTA and neg_new >= _PRESENCE_ABS:
                note = "%s has read as more frustrated / down than usual lately" % pm.name
            if note:
                out.append({
                    "person": pm.name,
                    "would_gently_note": note,
                    "basis": "recent reads shifted vs this person's baseline (a hypothesis, not a fact)",
                    "recent_disengaged_frac": round(dis_new, 2),
                    "recent_negative_frac": round(neg_new, 2),
                    "spoken": False,
                    "writes_belief": False,
                    "status": "shadow_logged_only",
                })
        return out

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
            "crystallization": {
                "phase": "P2_crystallization_valve",
                "writes_beliefs": False,
                "min_corroborations": _CRYST_MIN_CORROBORATIONS,
                "min_stability": _CRYST_MIN_STABILITY,
                "proposals": self.get_crystallization_proposals(),
            },
            "presence": {
                "phase": "presence_read_relational",
                "spoken": False,
                "note": "shadow — gentle would-notes when a person shifts from their usual read; complements (does not touch) the novel-object curiosity ask",
                "observations": self.get_presence_observations(),
            },
        }


theory_of_mind_engine = TheoryOfMindEngine.get_instance()
