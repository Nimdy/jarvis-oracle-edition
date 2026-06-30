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
# Widened from the flight-recorder transcript (questions that should reach the OSV but
# fell through to the INTROSPECTION catch-all). Patterns require self-reference
# (you / your / yourself) so non-self questions still return None and route normally.
_KIND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # consciousness / inner-state (specific, first)
    (re.compile(r"\b(are you|do you become|becoming|you'?re)\b.{0,12}\b(conscious|self[- ]aware|sentient|alive)\b", re.I), "consciousness_query"),
    (re.compile(r"\b(conscious|sentien|self[- ]aware)\b|\bdo you have (a soul|feelings|emotions|desires|fears|hopes|consciousness|awareness)\b|\bdo you feel\b", re.I), "consciousness_query"),
    # recent changes / what's new
    (re.compile(r"\bwhat('?s| is| are| has)?\s*(new|changed|different)\b|\bnew (feature|capabilit|skill)|\bwhat.{0,20}(recently )?(changed|learned|added)\b", re.I), "recent_changes"),
    # gated / not allowed yet
    (re.compile(r"\bnot allowed\b|\baren'?t you allowed\b|\bgated\b|\bdormant\b|\brestricted\b|\bwhat can'?t you do\b|\bnot (yet )?(allowed|able) to\b", re.I), "gated_capabilities"),
    # weaknesses
    (re.compile(r"\bweakness(es)?\b|\b(your )?(limitation|struggl|shortcoming|blind spot)\b|\bwhat are you bad at\b|\byour worst\b", re.I), "weaknesses"),
    # unknowns — JARVIS not knowing ONLY (not the user's "I don't know")
    (re.compile(r"\b(you|jarvis)\b.{0,6}\b(don'?t|do not|can'?t|cannot)\b.{0,6}\b(know|measure|see|read)\b|\b(don'?t|do not|can'?t|cannot) you (know|measure|see|read)\b|\bwhat (don'?t|can'?t) you (know|measure|read)\b|\byour (unknowns|blind spots)\b", re.I), "unknowns"),
    # health / how are you
    (re.compile(r"\bhow are you( doing| feeling)?\b(?!\s*(built|structured|made|wired|designed|composed|put together))|\bhow do you feel\b|\byour (health|wellbeing)\b|\bare you (ok|okay|alright|well|healthy)\b", re.I), "health"),
    # capabilities / architecture / how you're built / how you work
    (re.compile(r"\bwhat can you do\b|\bwhat are you (capable|able)\b|\byour (capabilit|abilit|architecture|codebase|subsystem|design)\b|\b(describe|tell me about|explain|walk me through|what can you tell me about)\b.{0,25}\b(your|the)?\b.{0,6}\b(architecture|codebase|subsystem|design|how you (work|reason|think|get|produce|generate|answer))\b|\bhow (do|are) you (work|built|structured|made|put together)\b|\bhow do you (get|reach|produce|generate|come up with|arrive at) an? answer\b", re.I), "capabilities"),
    # identity (broad, near-last so specific kinds win)
    (re.compile(r"\bwhat are you\b|\bwho are you\b|\bwhat (do you think )?you are\b|\bdo you know what you are\b|\bdescribe yourself\b|\b(tell me|something)\b.{0,20}\babout yourself\b|\bwhat kind of (system|ai|thing|model|being) are you\b", re.I), "identity"),
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


# -- architecture manifest (P-A): the code-grounded full structural map -------

def _arch(model: dict[str, Any]) -> dict[str, Any]:
    a = model.get("architecture")
    return a if isinstance(a, dict) else {}


def _arch_meta(model: dict[str, Any], key: str) -> Any:
    v = (_arch(model).get("_meta") or {}).get(key)
    return v.get("value") if isinstance(v, dict) else None


def _arch_inventory(model: dict[str, Any]) -> dict[str, Any]:
    inv = _arch(model).get("inventory")
    return inv if isinstance(inv, dict) else {}


def _arch_status_counts(model: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in _arch_inventory(model).values():
        st = v.get("status") if isinstance(v, dict) else None
        val = st.get("value") if isinstance(st, dict) else None
        if val:
            counts[val] = counts.get(val, 0) + 1
    return counts


def _arch_names_by_status(model: dict[str, Any], statuses: set[str]) -> list[str]:
    out: list[str] = []
    for v in _arch_inventory(model).values():
        st = v.get("status") if isinstance(v, dict) else None
        val = st.get("value") if isinstance(st, dict) else None
        if val in statuses:
            nm = (v.get("name") or {}).get("value")
            if nm:
                out.append(str(nm))
    return out


def _arch_areas(model: dict[str, Any]) -> list[str]:
    areas: set[str] = set()
    for v in _arch_inventory(model).values():
        ar = (v.get("area") or {}).get("value") if isinstance(v, dict) else None
        if ar:
            areas.add(str(ar))
    return sorted(areas)


def _arch_summary(model: dict[str, Any]) -> str:
    """One-sentence code-grounded structural summary (counts only; never raw names — see the
    unqualified-claim guard). Empty string if the architecture section is absent."""
    n_sub = _arch_meta(model, "subsystem_count")
    if not n_sub:
        return ""
    c = _arch_status_counts(model)
    n_stack = _arch_meta(model, "integrity_layers")
    return (
        f"My code-grounded architecture covers {n_sub} subsystems across "
        f"{len(_arch_areas(model))} domains, behind a {n_stack}-layer integrity stack "
        "(L0-L12 plus L3A/L3B). By designed status: "
        f"{c.get('shipped', 0) + c.get('live', 0)} shipped/live, {c.get('shadow', 0)} shadow, "
        f"{c.get('dormant', 0)} dormant, {c.get('partial', 0)} partial, "
        f"{c.get('signal-failure', 0)} signal-failure — designed structure, code-grounded "
        "but not a live measurement."
    )


# -- per-kind articulation ---------------------------------------------------

def _live_activity_line(model: dict[str, Any]) -> str:
    """Current NN-substrate activity, honestly qualified (P-C). Counts/values only — no raw
    subsystem names (guard) and no history. Empty if no live-activity section."""
    la = model.get("live_activity")
    if not isinstance(la, dict):
        return ""

    def _v(k: str) -> Any:
        f = la.get(k)
        return f.get("value") if isinstance(f, dict) else None

    bits: list[str] = []
    ss, reg = _v("self_sensing_skill"), _v("self_sensing_regime")
    if ss is not None:
        bits.append(f"self-sensing {reg or 'active'} (dynamic skill {ss}, shadow)")
    elif reg:
        bits.append(f"self-sensing {reg} (shadow; skill still warming up)")
    if _v("hemisphere_cycles") is not None:
        bits.append(f"{_v('hemisphere_cycles')} specialist-NN evolution cycles")
    if _v("mutations_this_hour") is not None:
        bits.append(f"{_v('mutations_this_hour')} kernel mutation(s) this hour")
    if _v("world_model_version") is not None:
        bits.append(f"world-model v{_v('world_model_version')} (shadow)")
    if _v("policy"):
        bits.append(f"policy {_v('policy')}")
    if _v("transcendence_level") is not None:
        bits.append(f"transcendence {_v('transcendence_level')} (self-scored, not external evidence)")
    if not bits:
        return ""
    return " What's active in me right now: " + "; ".join(bits) + "."


def _identity(model: dict[str, Any]) -> str:
    cov = model.get("coverage", {})
    bp = cov.get("subsystems_by_provenance", {})
    parts = [
        "I am JARVIS Oracle Edition, a local cognitive system running across a perception "
        "node and a brain node."
    ]
    summ = _arch_summary(model)
    if summ:
        parts.append(summ)
    parts.append(
        f"In real time my self-view reads {cov.get('subsystem_count', 0)} subsystems "
        f"({bp.get('measured', 0)} measured/active, {bp.get('shadow_only', 0)} shadow-only, "
        f"{bp.get('self_scored', 0)} self-reported), with some areas I cannot read yet. "
        "I report from this self-model and do not claim capabilities that are gated or unverified."
    )
    return " ".join(parts)


def _capabilities(model: dict[str, Any]) -> str:
    g = _group_subsystems(model)
    summ = _arch_summary(model)
    return (
        (summ + " " if summ else "")
        + f"Active/measured subsystems (live): {_fmt(g['active'])}. "
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
        "measure yet. I'd rather report a gap than guess." + _live_activity_line(model)
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
    base = (
        f"Gate-blocked / dormant (not available): {_fmt(g['dormant'])}. "
        f"Running in shadow with zero behavioral authority (not yet allowed to act): {_fmt(g['shadow'])}. "
        "These are earned, not declared — they activate only when they pass their gates."
    )
    # Counts (not raw names) from the 98-map: names can contain words the unqualified-claim
    # guard flags (e.g. a "Consciousness Kernel" subsystem), and counts are honest + sufficient.
    c = _arch_status_counts(model)
    n_dormant = c.get("dormant", 0) + c.get("gated", 0)
    n_shadow = c.get("shadow", 0)
    n_fail = c.get("signal-failure", 0)
    total = sum(c.values())
    if total:
        base += (
            f" Across my full {total}-subsystem architecture, by design: "
            f"{n_dormant} dormant/gate-blocked, {n_shadow} shadow (zero behavioral authority), "
            f"{n_fail} signal-failure (a measured dead end, not merely gated)."
        )
    return base


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
