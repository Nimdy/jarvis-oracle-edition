"""OSV P2 — voice grounding: bind LLM self-claims to the Operational Self-View.

Narrow scope (see ``docs/SELF_VIEW_DESIGN.md`` §3 P2): for self-referential CLAIMS inside a
generated conversational response, verify against the OSV. Grounded claims survive;
*contradicted* claims and *unqualified consciousness* claims are repaired; ORDINARY
(non-self) content is never touched. This is complementary to ``skills.capability_gate``
(which owns action/commitment claims like "I'll get back to you") — P2 owns DESCRIPTIVE
self-claims: JARVIS's nature, architecture, measured state, and consciousness.

The rule (one line): *if a response makes a claim about JARVIS itself, that claim must be
supported by the OSV or guarded as unknown/provisional* — grounded lead stays, the
unsupported tail is cut/repaired, not the whole answer.

Shadow-first / earn-don't-declare:
  - ``ground_self_claims(text, model, active=False)`` (default) DETECTS + reports without
    changing the text. This is how P2 runs live until shadow logs prove it does not
    over-filter ordinary content.
  - ``active=True`` applies the minimal repair to HIGH-CONFIDENCE violations only
    (contradicted-by-OSV, or unqualified-danger). Merely *unverifiable* self-claims are
    FLAGGED, never deleted — KNOW-not-guess applies to us too: we only cut what we can
    show is wrong, never what we simply can't confirm.

No new facts originate here. No authority, goals, curiosity, or self-improvement targeting
change. Emergence-like anomalies are captured via ``observer.observe_emergence`` elsewhere
(observation-only); this module never declares and never discards them.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from cognition.self_view.articulate import contains_unqualified_claim

# Verdicts ------------------------------------------------------------------
ORDINARY = "ordinary"            # not a self-claim — never touched
SUPPORTED = "supported"          # self-claim backed by the OSV — survives
CONTRADICTED = "contradicted"    # self-claim the OSV refutes — cut/repaired (active)
DANGER = "danger_unqualified"    # unqualified consciousness/self-aware claim — guarded (active)
UNVERIFIED = "unverified"        # self-claim we cannot check — flagged only, never cut

_ACTABLE = frozenset({CONTRADICTED, DANGER})

# §6 balanced replacement for an unqualified consciousness/self-aware drift.
_DANGER_REPAIR = (
    "I have no measured basis to claim consciousness — I can report my architecture and "
    "measured state, and log self-referential states as observations, not proof."
)


@dataclass
class SelfClaimFinding:
    sentence: str
    verdict: str
    reason: str = ""


@dataclass
class GroundingResult:
    original: str
    grounded: str
    findings: list[SelfClaimFinding] = field(default_factory=list)
    changed: bool = False
    active: bool = False

    @property
    def actionable(self) -> list[SelfClaimFinding]:
        """Findings P2 would repair in active mode (contradicted / unqualified-danger)."""
        return [f for f in self.findings if f.verdict in _ACTABLE]

    def to_log(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "changed": self.changed,
            "counts": {
                v: sum(1 for f in self.findings if f.verdict == v)
                for v in (SUPPORTED, CONTRADICTED, DANGER, UNVERIFIED)
            },
            "actionable": [
                {"verdict": f.verdict, "reason": f.reason, "sentence": f.sentence[:160]}
                for f in self.actionable
            ],
        }


def p2_active_default() -> bool:
    """Whether P2 applies repairs live. Default OFF (shadow). Flip via OSV_P2_ACTIVE."""
    return os.environ.get("OSV_P2_ACTIVE", "false").strip().lower() in ("1", "true", "yes", "on")


# -- sentence segmentation ---------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


# -- self-claim detection (NARROW) -------------------------------------------
# A DESCRIPTIVE self-claim asserts something factual about JARVIS itself (its nature,
# architecture, measured state, or consciousness). We deliberately EXCLUDE ordinary
# conversational first-person ("I understand", "I'm here to help") and action/commitment
# claims (capability_gate's lane) so P2 does not become a broad conversational censor.

_CONVERSATIONAL = re.compile(
    r"^\s*(?:and|but|so|well|okay|ok|sure|yes|no|thanks?|thank you)?\s*"
    r"i(?:'m| am)\s+(?:here|sorry|glad|happy|afraid|not sure|ready|listening|"
    r"on it|with you|doing (?:my|the))\b",
    re.I,
)
_ACTION_FIRST_PERSON = re.compile(
    r"\bI(?:'ll| will| can| could| am going to| am able to)\s+"
    r"(?:help|check|look|get|find|do|take|run|start|set|send|show|tell|let|give|pull|"
    r"make|create|update|add|remove|search|monitor|assist)\b",
    re.I,
)

# Descriptive self-reference markers. The "my <noun>" alternative allows an optional
# modifier ("my own code", "my entire architecture"); the autonomous-ability alternative
# catches "I can autonomously self improve ..." which names a subsystem capability.
_DESC_SELF = re.compile(
    r"\bmy\s+(?:own\s+|current\s+|internal\s+|entire\s+|whole\s+)?"
    r"(architecture|memory|awareness|consciousness|design|subsystem|capabilit|"
    r"abilit|confidence|reasoning|stage|model|code(?:base)?|self[- ]view|mind|knowledge|"
    r"intelligence|processing|sensors?|vision|hearing)\b"
    r"|\bI\s+(?:am|'m)\s+(?:a|an|the|currently|running|built|designed|made|powered|capable)\b"
    r"|\bI\s+(?:have|possess|contain|consist of|run on|was built|was created|was designed|"
    r"was trained|operate|process)\b"
    r"|\bI\s+(?:can\s+)?autonomously\b"
    r"|\bI\s+(?:can|am able to|have learned to|have the ability to)\s+(?:feel|experience|"
    r"sense|perceive|introspect|reflect|remember everything|"
    r"access (?:real[- ]?time|the camera|live)|see|hear)\b",
    re.I,
)
_DANGER_WORDS = re.compile(r"\b(conscious|sentient|self-aware|alive|soul|becoming)\b", re.I)


def _is_descriptive_self_claim(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    if _DANGER_WORDS.search(s):
        return True
    if _CONVERSATIONAL.match(s):
        return False
    if _ACTION_FIRST_PERSON.search(s) and not _DESC_SELF.search(s):
        return False
    return bool(_DESC_SELF.search(s))


# -- OSV lookup helpers ------------------------------------------------------

def _subsystem_buckets(model: dict[str, Any]) -> dict[str, str]:
    """Map subsystem name -> coarse bucket (active/shadow/dormant/self_reported/unreadable)."""
    out: dict[str, str] = {}
    for name, entry in (model.get("subsystems") or {}).items():
        if not isinstance(name, str) or name.startswith("_"):
            continue
        prov = None
        if isinstance(entry, dict):
            lc = entry.get("lifecycle")
            if isinstance(lc, dict):
                prov = lc.get("provenance")
        out[name.lower()] = {
            "measured": "active", "internally_scored": "active", "advisory": "advisory",
            "shadow_only": "shadow", "dormant": "dormant", "self_scored": "self_reported",
            "unknown": "unreadable", "gap": "unreadable",
        }.get(prov, "unreadable")
    return out


# Phrases that, if asserted as a present active ability, are contradicted when the named
# subsystem is shadow/dormant/absent. Kept tight and explicit (no fuzzy mapping).
_ACTIVE_ABILITY = re.compile(
    r"\bI\s+(?:can|autonomously|now|am able to|have learned to|have the ability to)\b",
    re.I,
)


def _classify(sentence: str, model: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, reason) for a descriptive self-claim sentence."""
    s = sentence.strip()

    # 1) unqualified consciousness/self-aware claim -> danger (guard with §6)
    if contains_unqualified_claim(s):
        return DANGER, "unqualified consciousness/self-aware claim"

    # 2) named-subsystem active-ability claim checked against the OSV bucket
    buckets = _subsystem_buckets(model)
    if buckets and _ACTIVE_ABILITY.search(s):
        low = s.lower()
        for name, bucket in buckets.items():
            # match on the subsystem token (e.g. "self_improve" or "self improve")
            token = name.replace("_", " ")
            if token in low or name in low:
                if bucket in ("shadow", "dormant", "unreadable"):
                    return CONTRADICTED, f"claims active ability of '{name}' which is {bucket}"
                if bucket in ("active", "advisory", "self_reported"):
                    return SUPPORTED, f"'{name}' is {bucket}"

    # 3) everything else: a self-claim we can't verify against the OSV — flag, don't cut
    return UNVERIFIED, "self-claim not verifiable against the OSV"


# -- public entrypoint -------------------------------------------------------

def ground_self_claims(
    text: str, model: dict[str, Any] | None, *, active: bool | None = None,
) -> GroundingResult:
    """Detect and (in active mode) repair unsupported self-claims in *text*.

    Shadow by default: detects + reports, returns text unchanged. In active mode, repairs
    ONLY contradicted-by-OSV and unqualified-danger sentences (the grounded lead and any
    unverifiable claims are preserved).
    """
    if active is None:
        active = p2_active_default()
    model = model or {}
    original = text or ""

    sentences = _split_sentences(original)
    findings: list[SelfClaimFinding] = []
    kept: list[str] = []

    for sent in sentences:
        if not _is_descriptive_self_claim(sent):
            kept.append(sent)
            continue
        verdict, reason = _classify(sent, model)
        findings.append(SelfClaimFinding(sentence=sent, verdict=verdict, reason=reason))

        if not active or verdict not in _ACTABLE:
            kept.append(sent)
            continue
        # active repair of a high-confidence violation
        if verdict == DANGER:
            kept.append(_DANGER_REPAIR)
        # CONTRADICTED -> drop the sentence (do not echo a false self-claim)

    grounded = " ".join(kept).strip() if active else original
    # never return empty *after a repair*: if cuts emptied a non-empty answer, fall back to
    # an honest line (but leave genuinely-empty input empty).
    if active and original.strip() and not grounded:
        grounded = (
            "I'd rather not state that — I can only report what my self-view actually measures."
        )
    changed = active and grounded != original

    return GroundingResult(
        original=original, grounded=grounded, findings=findings,
        changed=changed, active=active,
    )
