"""Companion Cognition — P3: Read→behavior ADVISORY (SHADOW / narrate-only).

The situational read (P0) tells JARVIS what it thinks is happening; the per-person
theory-of-mind model (P1) tells it what it has *learned* about this companion. P3
joins them into a SUGGESTED conversational adjustment — soften tone / be concise /
give space / pivot / acknowledge a delay — narrated as "would have X", with the
reason and a confidence.

Honesty contract (docs/COMPANION_COGNITION_DESIGN.md P3; mirrors situational_read §P0):
  - It APPLIES NOTHING. ``applied`` is always False; ``changes_behavior`` is False.
    Every suggestion is logged for the operator to review ("would have softened /
    wrapped up / asked"), never taken. Actually adjusting is P4 (separately earned).
  - Every suggestion is a HYPOTHESIS, confidence-scored, with its contributing
    signals named in ``reason``.
  - A suggestion is ``person_aware`` (and gains confidence) ONLY when the live read
    is CORROBORATED by what JARVIS has learned about THIS person (the ToM
    disposition/responsiveness) — not from a single turn's emotion label alone.
  - P3->P4 is EARNED by demonstrated correctness against companion feedback on an
    accelerated maturation curve — never coded. The readiness here is a structural
    proxy only (enough advisories accrued, trigger not pinned).

Pure-Python, dependency-free, near-zero cost (no model call): runs inline after the
read + ToM update, last in the turn so it can never perturb it. Singleton:
``behavior_advisory_engine``.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Separate shadow-store persistence (NOT soul/identity). Lets advisories survive
# restarts so the P3->P4 earn-gate can accumulate its reps.
_STATE_DIR = os.path.expanduser("~/.jarvis")
_STATE_PATH = os.path.join(_STATE_DIR, "companion_behavior_advisory.json")

# P3->P4 structural-readiness proxy: minimum advisories before P4 could even be
# considered. A STRUCTURAL proxy only — the true correctness gate (did the
# suggestion match what actually helped?) is EARNED via companion feedback.
_P3P4_MIN_ADVISORIES = 40
# Minimum-credibility FLOOR (NOT a target, §24): the P3->P4 structural proxy must not read
# "ready" on the advisory COUNT alone. Person-aware corroboration (the learned person-model
# actually informed the suggestion) has to be exercised, else 0/N advisories that learned
# nothing about the companion would falsely light the gate. Earns up organically as the
# person-model paths trip; never tuned.
_P3P4_MIN_PERSON_AWARE_FRACTION = 0.25

# Disposition confidence floor before the learned person-model is allowed to
# corroborate (raise confidence on) a live suggestion. Below this the ToM model
# has not seen enough of this person to mean anything.
_PERSON_AWARE_DISPOSITION_FLOOR = 0.30
# Cap on the person-aware confidence boost (a learned pattern strengthens a read;
# it never manufactures certainty on its own).
_PERSON_AWARE_MAX_BOOST = 0.20

# ToM disposition strings that indicate the person tends to pull back (mirrors
# theory_of_mind._ENGAGEMENT_PHRASE["disengaging"]). Kept as substrings so a phrase
# change there degrades gracefully to "not corroborated" rather than a wrong boost.
_DISENGAGING_DISPOSITION_HINTS = ("pull back", "disengage")


@dataclass
class BehaviorAdvisory:
    """A narrate-only set of suggested conversational adjustments for one turn."""

    timestamp: float
    speaker: str
    suggestions: list = field(default_factory=list)  # [{adjustment, narration, reason, confidence, person_aware}]
    primary: "str | None" = None                     # the top suggestion's narration
    salience: float = 0.0
    applied: bool = False                            # ALWAYS False in P3 (narrate-only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "suggestions": self.suggestions,
            "primary": self.primary,
            "salience": round(self.salience, 3),
            "applied": self.applied,
            "phase": "P3_behavior_advisory",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorAdvisory":
        return cls(
            timestamp=d.get("timestamp", 0.0),
            speaker=d.get("speaker", "unknown"),
            suggestions=d.get("suggestions", []) or [],
            primary=d.get("primary"),
            salience=float(d.get("salience", 0.0) or 0.0),
            applied=bool(d.get("applied", False)),
        )


class BehaviorAdvisoryEngine:
    """Joins the read (P0) + person-model (P1) into narrate-only suggestions. SHADOW."""

    _instance: "BehaviorAdvisoryEngine | None" = None

    @classmethod
    def get_instance(cls) -> "BehaviorAdvisoryEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._recent: deque = deque(maxlen=50)
        self._total = 0           # advisories emitted (salience tripped)
        self._person_aware = 0    # of those, how many drew on the learned person-model
        self._load()

    def _save(self) -> None:
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            data = {
                "total": self._total,
                "person_aware": self._person_aware,
                "recent": [a.to_dict() for a in self._recent],
            }
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
            self._person_aware = int(data.get("person_aware", 0))
            for a in data.get("recent", []):
                self._recent.append(BehaviorAdvisory.from_dict(a))
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # -- the read->behavior join (narrate-only) -----------------------------

    def propose(self, read: Any, person_model: Any = None) -> "BehaviorAdvisory | None":
        """Build a narrate-only advisory from a situational read + person-model.

        Returns None when the read did not trip salience (nothing notable — no
        advisory, the anti-chatterbox spine). Never raises, never acts.
        """
        try:
            if read is None or not getattr(read, "salience_tripped", False):
                return None
            advisory = self._build(read, person_model)
        except Exception:
            return None
        self._recent.append(advisory)
        self._total += 1
        if any(s.get("person_aware") for s in advisory.suggestions):
            self._person_aware += 1
        self._save()
        return advisory

    def _build(self, read: Any, pm: Any) -> BehaviorAdvisory:
        sentiment = getattr(read, "user_sentiment", "neutral")
        engagement = getattr(read, "engagement", "neutral")
        self_check = getattr(read, "self_check", "")
        base_conf = float(getattr(read, "confidence", 0.0) or 0.0)
        evidence = getattr(read, "evidence", []) or []
        overexplain = self_check != "reply length proportionate" and bool(self_check)
        slow_turn = any(
            isinstance(e, (list, tuple)) and len(e) >= 1 and e[0] == "latency_ms"
            for e in evidence
        )

        # ── learned person-model signals (hypotheses, may be absent/weak) ──
        disposition = str(getattr(pm, "disposition", "") or "")
        disp_conf = float(getattr(pm, "disposition_confidence", 0.0) or 0.0)
        responsiveness = str(getattr(pm, "responsiveness", "") or "")
        person_tends_to_withdraw = (
            disp_conf >= _PERSON_AWARE_DISPOSITION_FLOOR
            and (
                any(h in disposition for h in _DISENGAGING_DISPOSITION_HINTS)
                or responsiveness == "withdrawn"
            )
        )
        # boost scales with how confident the learned disposition is, capped.
        person_boost = min(_PERSON_AWARE_MAX_BOOST, round(disp_conf * 0.3, 3))

        def _conf(base: float, corroborated: bool) -> float:
            v = base + (person_boost if corroborated else 0.0)
            return round(max(0.0, min(1.0, v)), 3)

        suggestions: list[dict[str, Any]] = []

        # give_space — the most conservative disengagement response
        if sentiment == "negative" and engagement == "disengaging":
            corr = person_tends_to_withdraw
            suggestions.append({
                "adjustment": "give_space",
                "narration": "would consider giving space / a lighter check-in",
                "reason": self._reason(
                    "negative sentiment + disengaging this turn", corr, disposition),
                "confidence": _conf(base_conf, corr),
                "person_aware": corr,
            })
        else:
            # pivot_or_check — disengaging without the negative-sentiment escalation
            if engagement == "disengaging":
                corr = person_tends_to_withdraw
                suggestions.append({
                    "adjustment": "pivot_or_check",
                    "narration": "would consider pivoting or asking if this is useful",
                    "reason": self._reason(
                        "disengaging this turn", corr, disposition),
                    "confidence": _conf(base_conf, corr),
                    "person_aware": corr,
                })
            # soften_tone — negative sentiment without the disengagement
            if sentiment == "negative" and engagement != "disengaging":
                suggestions.append({
                    "adjustment": "soften_tone",
                    "narration": "would consider softening tone",
                    "reason": "negative sentiment this turn",
                    "confidence": _conf(base_conf, False),
                    "person_aware": False,
                })

        # be_concise — the self-check tell. Now person-aware once the LEARNED brevity
        # axis is earned (this is what uncages corroboration for a steady/engaged
        # companion — engagement disposition alone could never back this one).
        if overexplain:
            verbosity_pref = str(getattr(pm, "verbosity_pref", "") or "")
            verbosity_conf = float(getattr(pm, "verbosity_confidence", 0.0) or 0.0)
            prefers_concise = (
                verbosity_pref == "prefers concise replies"
                and verbosity_conf >= _PERSON_AWARE_DISPOSITION_FLOOR
            )
            # boost scales with the brevity-axis confidence (NOT the engagement disp_conf).
            verbosity_boost = min(_PERSON_AWARE_MAX_BOOST, round(verbosity_conf * 0.3, 3))
            be_concise_conf = base_conf + (verbosity_boost if prefers_concise else 0.0)
            suggestions.append({
                "adjustment": "be_concise",
                "narration": "would consider being more concise / checking if this helps",
                "reason": self._reason(self_check, prefers_concise, verbosity_pref),
                "confidence": round(max(0.0, min(1.0, be_concise_conf)), 3),
                "person_aware": prefers_concise,
            })

        # acknowledge_delay — the user waited
        if slow_turn:
            suggestions.append({
                "adjustment": "acknowledge_delay",
                "narration": "would consider acknowledging the delay",
                "reason": "this turn was slow to answer (the user waited)",
                "confidence": _conf(base_conf, False),
                "person_aware": False,
            })

        suggestions.sort(key=lambda s: s["confidence"], reverse=True)
        primary = suggestions[0]["narration"] if suggestions else None

        return BehaviorAdvisory(
            timestamp=time.time(),
            speaker=getattr(read, "speaker", "unknown"),
            suggestions=suggestions,
            primary=primary,
            salience=float(getattr(read, "salience", 0.0) or 0.0),
            applied=False,  # P3 is narrate-only — nothing is ever applied here
        )

    @staticmethod
    def _reason(live_signal: str, corroborated: bool, disposition: str) -> str:
        if corroborated and disposition:
            return f"{live_signal}; matches a learned pattern for this person ({disposition})"
        return live_signal

    def get_recent(self, n: int = 10) -> list:
        items = list(self._recent)[-n:]
        return [a.to_dict() for a in reversed(items)]

    def _promotion_readiness(self) -> dict[str, Any]:
        """Companion P3->P4 gate: a STRUCTURAL readiness proxy only. Signals that
        enough advisories accrued and that the person-aware path is exercised. The
        TRUE gate (did the suggestion match what actually helped the companion?) is
        EARNED via feedback — never coded, never self-scored."""
        total = self._total
        enough = total >= _P3P4_MIN_ADVISORIES
        pa_frac = (self._person_aware / total) if total else 0.0
        pa_ok = pa_frac >= _P3P4_MIN_PERSON_AWARE_FRACTION
        blocking = []
        if not enough:
            blocking.append("need %d advisories (have %d)" % (_P3P4_MIN_ADVISORIES, total))
        if total and not pa_ok:
            blocking.append(
                "person-aware fraction %.2f below %.2f (%d/%d advisories corroborated by the "
                "learned person-model)" % (pa_frac, _P3P4_MIN_PERSON_AWARE_FRACTION,
                                           self._person_aware, total))
        return {
            "gate": "P3->P4",
            "would_promote_to": "P4_behavior_active",
            # BOTH the count AND person-aware corroboration must be exercised — a count-only
            # proxy would declare-ready off advisories that never learned anything actionable.
            "structural_ready": bool(enough and pa_ok),
            "note": "structural proxy only; true correctness is EARNED via companion feedback, never coded",
            "advisories_emitted": total,
            "person_aware_fraction": round(pa_frac, 3),
            "min_advisories": _P3P4_MIN_ADVISORIES,
            "min_person_aware_fraction": _P3P4_MIN_PERSON_AWARE_FRACTION,
            "blocking": blocking,
        }

    def get_status(self) -> dict[str, Any]:
        last = self._recent[-1].to_dict() if self._recent else None
        return {
            "phase": "P3_behavior_advisory",
            "authority": "advisory_shadow_narrate_only",
            "changes_behavior": False,
            "applies_suggestions": False,
            "advisories_emitted": self._total,
            "person_aware_count": self._person_aware,
            "last": last,
            "promotion_readiness": self._promotion_readiness(),
        }


behavior_advisory_engine = BehaviorAdvisoryEngine.get_instance()
