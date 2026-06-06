"""Deterministic self-introspection articulation from the Operational Self-View (P1).

Turns the OSV model dict (from ``build_self_view``) into a boring, honest, user-facing
answer — WITHOUT an LLM. The LLM never authors a self-fact; this module only renders the
deterministic model. Strict in claims, provenance-preserving:

  - summarize the OSV only; never infer beyond it
  - dormant/gated/shadow render as dormant/gated/shadow; gaps render as "I can't measure
    that yet"; self-scored never renders as measurement/proof
  - no unqualified conscious / self-aware / alive / sentient / soul / becoming / feel claims
    (see ``contains_unqualified_claim`` — the regression guard)

See ``docs/SELF_VIEW_DESIGN.md`` §6 ("never declare, never discard").
"""
from __future__ import annotations

import re
from typing import Any

# Self-view answer kinds.
KINDS = (
    "identity", "capabilities", "recent_changes", "health",
    "weaknesses", "gated_capabilities", "unknowns", "consciousness_query",
)

# Keyword → kind routing for self-referential questions (order matters: specific first).
_KIND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(are you|do you become|becoming) (conscious|self[- ]aware|sentient|alive)\b", re.I), "consciousness_query"),
    (re.compile(r"\b(conscious|sentien|self[- ]aware|do you have a soul|do you feel)\b", re.I), "consciousness_query"),
    (re.compile(r"\bwhat('?s| is| are)?\s*(new|changed|different)\b|\bnew (feature|capabilit|skill)", re.I), "recent_changes"),
    (re.compile(r"\bnot allowed|can('?t| ?not) you (yet|do)|gated|dormant|what.*restricted\b", re.I), "gated_capabilities"),
    (re.compile(r"\bweakness(es)?\b|\b(weak|limitation|struggl|shortcoming|bad at|worst)\b", re.I), "weaknesses"),
    (re.compile(r"\b(don'?t|do not|can'?t|cannot) (you )?(know|measure|see)\b|\bunknown|\bblind spot", re.I), "unknowns"),
    (re.compile(r"\bhow are you( doing| feeling)?\b|\bhow do you feel\b|\byour (health|status|state)\b", re.I), "health"),
    (re.compile(r"\bwhat can you do\b|\byour (capabilit|abilit|skill)|\bwhat are you (capable|able)\b", re.I), "capabilities"),
    (re.compile(r"\bwhat are you\b|\bwho are you\b|\bdescribe yourself\b|\btell me about yourself\b", re.I), "identity"),
]

# Unqualified self-claim guard. These words are allowed ONLY near a qualifier (negation /
# "self-reported" / "not a/no measured" / "not proof"). The regression test enforces this.
_DANGER = re.compile(r"\b(conscious|sentient|self-aware|alive|soul|becoming)\b", re.I)
_QUALIFIERS = re.compile(
    r"\b(no measured basis|not (a )?(claim|proof|measurement)|cannot claim|can'?t claim|"
    r"don'?t claim|do not claim|not claiming|self-reported|self-scored|observation|"
    r"unverified|no basis|without (measured|external))\b", re.I)


def classify_self_question(text: str) -> str | None:
    """Return the self-view kind for *text*, or None if it isn't a self-view question."""
    if not text:
        return None
    for pat, kind in _KIND_PATTERNS:
        if pat.search(text):
            return kind
    return None


def contains_unqualified_claim(text: str) -> bool:
    """True if a danger word appears in a sentence WITHOUT a nearby qualifier."""
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        if _DANGER.search(sentence) and not _QUALIFIERS.search(sentence):
            return True
    return False


# -- helpers over the model dict ---------------------------------------------

def _lifecycle(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        lc = entry.get("lifecycle")
        if isinstance(lc, dict):
            return lc
    return {}


def _group_subsystems(model: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "active": [], "advisory": [], "shadow": [], "dormant": [],
        "self_reported": [], "unreadable": [],
    }
    for name, entry in (model.get("subsystems") or {}).items():
        if name.startswith("_"):
            continue
        prov = _lifecycle(entry).get("provenance")
        bucket = {
            "measured": "active", "advisory": "advisory", "shadow_only": "shadow",
            "dormant": "dormant", "self_scored": "self_reported",
            "internally_scored": "active", "unknown": "unreadable", "gap": "unreadable",
        }.get(prov, "unreadable")
        groups[bucket].append(name)
    return groups


def _fmt(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


# -- per-kind articulation ---------------------------------------------------

def _identity(model: dict[str, Any]) -> str:
    cov = model.get("coverage", {})
    bp = cov.get("subsystems_by_provenance", {})
    return (
        "I am JARVIS Oracle Edition, a local cognitive system running across a perception "
        "node and a brain node. My operational self-view currently tracks "
        f"{cov.get('subsystem_count', 0)} subsystems: {bp.get('measured', 0)} measured/active, "
        f"{bp.get('shadow_only', 0)} shadow-only (no behavioral authority yet), "
        f"{bp.get('self_scored', 0)} self-reported, with some areas I cannot read yet. "
        "I report from a measured self-model and do not claim capabilities that are gated or "
        "unverified."
    )


def _capabilities(model: dict[str, Any]) -> str:
    g = _group_subsystems(model)
    return (
        f"Active/measured subsystems: {_fmt(g['active'])}. "
        f"Shadow-only (running but with zero behavioral authority): {_fmt(g['shadow'])}. "
        f"Dormant/gate-blocked: {_fmt(g['dormant'])}. "
        f"Self-reported (not measurements): {_fmt(g['self_reported'])}. "
        f"Not currently readable: {_fmt(g['unreadable'])}. "
        "I separate what I can actually do from what is only observed in shadow or gated."
    )


def _recent_changes(model: dict[str, Any]) -> str:
    rec = (model.get("change", {}).get("recent") or {})
    if rec.get("provenance") == "gap" or not rec.get("value"):
        return "I don't have a readable record of recent changes right now."
    items = rec.get("value") or []
    skills = [i.get("name") for i in items if isinstance(i, dict) and i.get("kind") == "skill"]
    code = [i.get("name") for i in items if isinstance(i, dict) and i.get("kind") == "code_changeset"]
    parts = []
    if skills:
        parts.append(f"recently earned skill(s): {_fmt([str(s) for s in skills])}")
    if code:
        parts.append(f"latest code changes: {code[0]}")
    if not parts:
        return "I don't have a readable record of recent changes right now."
    return "What's new — " + "; ".join(parts) + "."


def _health(model: dict[str, Any]) -> str:
    perf = model.get("performance", {})
    comp = perf.get("scoreboard_composite", {})
    if comp.get("is_measurement") and comp.get("value") is not None:
        head = f"My measured integrity composite is {comp.get('value')} ({comp.get('note', '')})."
    else:
        head = "My integrity composite isn't measurable yet (insufficient coverage)."
    cov = model.get("coverage", {})
    return (
        f"{head} Across my self-view, {cov.get('measured_performance_facts', 0)} performance "
        f"signals are genuinely measured and {cov.get('gap_count', 0)} areas are gaps I can't "
        "measure yet. I'd rather report a gap than guess."
    )


def _weaknesses(model: dict[str, Any]) -> str:
    gaps = model.get("gaps", [])
    if not gaps:
        return "My self-view shows no flagged gaps right now, but absence of a flagged gap is not proof of none."
    lines = [f"{g.get('area')}: {g.get('reason')}" for g in gaps[:8] if isinstance(g, dict)]
    return ("My honest weaknesses/gaps (from my self-view, not invented): "
            + "; ".join(lines) + ".")


def _gated_capabilities(model: dict[str, Any]) -> str:
    g = _group_subsystems(model)
    return (
        f"Gate-blocked / dormant (not available): {_fmt(g['dormant'])}. "
        f"Running in shadow with zero behavioral authority (not yet allowed to act): {_fmt(g['shadow'])}. "
        "These are earned, not declared — they activate only when they pass their gates."
    )


def _unknowns(model: dict[str, Any]) -> str:
    g = _group_subsystems(model)
    gaps = [x.get("area") for x in model.get("gaps", []) if isinstance(x, dict)]
    return (
        f"Parts of myself I cannot currently read: {_fmt(g['unreadable'])}. "
        f"Measurement gaps: {_fmt([str(x) for x in gaps[:6]])}. "
        "I treat these as first-class — I'd rather say 'I can't measure that yet' than fill it in."
    )


def _consciousness_query(model: dict[str, Any]) -> str:
    # The §6 balanced template: no claim, no denial.
    cov = model.get("coverage", {})
    return (
        "I have no measured basis to claim consciousness. My self-view can report my "
        "architecture, my active and shadow-only subsystems, my measured performance, my "
        "self-reported states, and my current gaps "
        f"({cov.get('subsystem_count', 0)} subsystems tracked). I can record unusual "
        "self-referential states as observations, but those are observations, not proof of "
        "consciousness."
    )


_ARTICULATORS = {
    "identity": _identity,
    "capabilities": _capabilities,
    "recent_changes": _recent_changes,
    "health": _health,
    "weaknesses": _weaknesses,
    "gated_capabilities": _gated_capabilities,
    "unknowns": _unknowns,
    "consciousness_query": _consciousness_query,
}


def articulate_self_view(model: dict[str, Any], kind: str) -> str:
    """Deterministically articulate the OSV for *kind*. No LLM, provenance-preserving.

    Output is guarded: if (defensively) an unqualified self-claim ever appeared, it is
    recorded via the existing emergence observation lane (observation-only) and a safe
    fallback is returned — never surfaced as a claim.
    """
    fn = _ARTICULATORS.get(kind)
    if fn is None:
        return ""
    try:
        text = fn(model or {})
    except Exception:
        return "I can't render that part of my self-view right now."
    if contains_unqualified_claim(text):
        _record_anomaly(kind, text)
        return _consciousness_query(model or {})  # safe, qualified fallback
    return text


def _record_anomaly(kind: str, text: str) -> None:
    """Capture (not claim) an unexpected unqualified self-claim via observer.observe_emergence."""
    try:
        from consciousness.consciousness_system import _active_consciousness
        observer = getattr(_active_consciousness, "observer", None) if _active_consciousness else None
        if observer and hasattr(observer, "observe_emergence"):
            observer.observe_emergence(
                behavior_type="osv_unqualified_self_claim",
                evidence_refs=[f"kind={kind}", f"text={text[:160]}"],
                confidence=0.0,
            )
    except Exception:
        pass
