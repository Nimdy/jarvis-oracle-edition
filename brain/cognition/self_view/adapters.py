"""Bespoke, read-only subsystem adapters for the Operational Self-View (P0.6).

One small adapter per subsystem. Each KNOWS its subsystem's real cache shape (verified
against build_cache + the subsystem's status method) and REFUSES TO GUESS — when an
expected field is missing/malformed it degrades to unknown/gap, never a fabricated default.

Each adapter returns dict[str, Fact]: always a "lifecycle" Fact, plus subsystem-specific
signal Facts. Provenance is assigned honestly per signal:
  measured            — validated against an external/ground-truth comparator, or a real event count
  internally_scored   — computed by the system from its own outputs, not validated
  self_scored         — the system grading itself (awareness, integrity index, truth score)
  shadow_only         — observed in shadow / zero behavioral authority
  advisory            — non-authoritative signal
  dormant             — exists but gate-blocked / inactive
  unknown / gap       — present but unreadable / not yet knowable

No LLM, no writes, no behavior authority.
"""
from __future__ import annotations

from typing import Any

from cognition.self_view.provenance import Fact, Provenance, gap, unknown


def _num(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _lifecycle_from_level(level_name: Any, validated: Any = None, src: str = "") -> Fact:
    """Promotion-gated lifecycle from a level_name (shadow/advisory/active)."""
    note = f"{validated} validated" if validated is not None else ""
    if level_name == "active":
        return Fact("active", Provenance.MEASURED, note=note or "promotion: active", source=src)
    if level_name == "advisory":
        return Fact("advisory", Provenance.ADVISORY, note=note or "promotion: advisory", source=src)
    if level_name == "shadow":
        return Fact("shadow", Provenance.SHADOW_ONLY,
                    note=(note + " — not yet advisory").strip(" —"), source=src)
    return unknown("no readable promotion level", source=src)


# --- Cognition ---------------------------------------------------------------

def read_world_model(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "cognition.world_model"
    out: dict[str, Fact] = {}
    promo = blob.get("promotion") if isinstance(blob.get("promotion"), dict) else {}
    out["lifecycle"] = _lifecycle_from_level(promo.get("level_name"),
                                             promo.get("total_validated"), src)
    causal = blob.get("causal") if isinstance(blob.get("causal"), dict) else {}
    if causal.get("predictive_total"):
        out["predictive_accuracy"] = Fact(_num(causal.get("predictive_accuracy")),
                                          Provenance.MEASURED,
                                          note="validated foresight", source=src)
        out["persistence_accuracy"] = Fact(_num(causal.get("persistence_accuracy")),
                                          Provenance.INTERNALLY_SCORED,
                                          note="steady-state continuation — not foresight", source=src)
    else:
        out["predictive_accuracy"] = gap("no validated predictions yet (cold)", source=src)
    return out


def read_simulator(world_model_blob: dict[str, Any]) -> dict[str, Fact]:
    """Simulator state is nested under world_model in the cache."""
    src = "cognition.simulator"
    sp = world_model_blob.get("simulator_promotion")
    sim = world_model_blob.get("simulator")
    if not isinstance(sp, dict) and not isinstance(sim, dict):
        return {"lifecycle": unknown("simulator state not present in world_model", source=src)}
    sp = sp if isinstance(sp, dict) else {}
    sim = sim if isinstance(sim, dict) else {}
    out: dict[str, Fact] = {
        "lifecycle": _lifecycle_from_level(sp.get("level_name"), sp.get("total_validated"), src),
    }
    if sim.get("avg_confidence") is not None:
        out["avg_confidence"] = Fact(_num(sim.get("avg_confidence")), Provenance.INTERNALLY_SCORED,
                                     note="simulator's own confidence, not validated", source=src)
    return out


def read_policy(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "policy.evaluator"
    mode = blob.get("mode")
    if mode in ("shadow", "canary"):
        life = Fact(f"shadow ({mode})", Provenance.SHADOW_ONLY,
                    note="NN-vs-kernel shadow; non-authoritative until promotion", source=src)
    elif mode in ("active", "live"):
        life = Fact("active", Provenance.MEASURED, note="promoted to live control", source=src)
    elif mode is None:
        life = unknown("no policy mode", source=src)
    else:
        life = Fact(str(mode), Provenance.UNKNOWN, source=src)
    out = {"lifecycle": life}
    if blob.get("nn_win_rate") is not None:
        out["nn_win_rate"] = Fact(_num(blob.get("nn_win_rate")), Provenance.SHADOW_ONLY,
                                  note="shadow NN-vs-kernel win rate", source=src)
    if "eligible_for_control" in blob:
        out["eligible_for_control"] = Fact(bool(blob.get("eligible_for_control")),
                                           Provenance.MEASURED, source=src)
    return out


def read_hemisphere(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "hemisphere.orchestrator"
    if not blob.get("enabled", False):
        return {"lifecycle": Fact("disabled", Provenance.DORMANT,
                                  note="hemisphere training not enabled", source=src)}
    out = {"lifecycle": Fact("enabled", Provenance.MEASURED, source=src)}
    specs = blob.get("matrix_specialists")
    if isinstance(specs, (list, dict)):
        n = len(specs)
        out["specialists"] = Fact(n, Provenance.MEASURED,
                                  note="specialist NNs present", source=src)
    else:
        out["specialists"] = unknown("specialist roster shape unreadable", source=src)
    return out


# --- Self-improvement / skills ----------------------------------------------

def read_self_improve(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "self_improve.orchestrator"
    if not blob.get("active", False):
        return {"lifecycle": Fact("inactive", Provenance.DORMANT, source=src)}
    stage = blob.get("stage")
    out = {"lifecycle": Fact(f"active (stage {stage})", Provenance.MEASURED,
                             note=f"dry_run={blob.get('effective_dry_run')}; gated pipeline", source=src)}
    for f in ("total_improvements", "total_rollbacks", "total_failures"):
        if f in blob:
            out[f] = Fact(blob.get(f), Provenance.MEASURED, source=src)
    return out


def read_skills(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "skills.registry"
    bs = blob.get("by_status")
    if not isinstance(bs, dict):
        return {"lifecycle": unknown("no by_status", source=src)}
    skills = blob.get("skills") or []
    earned = sum(1 for s in skills if isinstance(s, dict) and s.get("learning_job_id"))
    return {
        "lifecycle": Fact(bs, Provenance.MEASURED, note="verified-skill counts", source=src),
        "earned_via_pipeline": Fact(earned, Provenance.MEASURED,
                                    note="acquired end-to-end (vs bootstrap)", source=src),
    }


# --- Autonomy / spark / companion -------------------------------------------

def read_autonomy(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "autonomy.orchestrator"
    if not blob.get("enabled", False):
        return {"lifecycle": Fact("disabled", Provenance.DORMANT, source=src)}
    lvl = blob.get("autonomy_level_name") or blob.get("autonomy_level")
    out = {"lifecycle": Fact(f"enabled (L{lvl})", Provenance.MEASURED, source=src)}
    if "completed_total" in blob:
        out["completed_total"] = Fact(blob.get("completed_total"), Provenance.MEASURED, source=src)
    return out


def read_grounding_ring(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "autonomy.grounding_ring"
    phase = blob.get("phase")
    authority = blob.get("authority")
    if phase is None:
        return {"lifecycle": unknown("no grounding phase", source=src)}
    # zero-authority spark is shadow by design
    prov = Provenance.SHADOW_ONLY if (authority in ("zero_authority", None)) else Provenance.ADVISORY
    return {"lifecycle": Fact(f"phase={phase}", prov,
                              note=f"authority={authority}", source=src)}


def read_companion_read(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "companion.situational_read"
    authority = blob.get("authority")
    phase = blob.get("phase")
    prov = Provenance.SHADOW_ONLY if (authority in ("shadow_logged_only", "zero_authority", None)) \
        else Provenance.ADVISORY
    out = {"lifecycle": Fact(f"phase={phase}", prov, note=f"authority={authority}", source=src)}
    if "observed_turns" in blob:
        out["observed_turns"] = Fact(blob.get("observed_turns"), Provenance.MEASURED, source=src)
    return out


# --- Epistemic ---------------------------------------------------------------

def read_belief_graph(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.belief_graph"
    if not blob.get("initialized", False):
        return {"lifecycle": Fact("uninitialized", Provenance.DORMANT, source=src)}
    out = {"lifecycle": Fact("active", Provenance.MEASURED, source=src)}
    for f in ("belief_count", "edge_count", "active_contradictions"):
        if f in blob:
            out[f] = Fact(blob.get(f), Provenance.MEASURED, source=src)
    return out


def read_truth_calibration(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.calibration"
    status = blob.get("status")
    out = {"lifecycle": Fact(str(status), Provenance.MEASURED if status else Provenance.UNKNOWN,
                             note="calibration status", source=src)}
    # truth_score / maturity are the engine's own assessment -> self_scored
    if blob.get("truth_score") is not None:
        out["truth_score"] = Fact(_num(blob.get("truth_score")), Provenance.SELF_SCORED,
                                  note="self-calibration score, not external", source=src)
    # brier / ece ARE validated against outcomes -> measured
    if blob.get("brier_score") is not None:
        out["brier_score"] = Fact(_num(blob.get("brier_score")), Provenance.MEASURED,
                                  note="calibration error vs outcomes", source=src)
    return out


def read_reflective_audit(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.reflective_audit"
    ta = blob.get("total_audits")
    if ta is None:
        return {"lifecycle": unknown("no audit counters", source=src)}
    return {
        "lifecycle": Fact("active (observability)", Provenance.MEASURED,
                          note="Layer-9 audit; observation-only", source=src),
        "total_audits": Fact(ta, Provenance.MEASURED, source=src),
        "total_findings": Fact(blob.get("total_findings"), Provenance.MEASURED, source=src),
    }


def read_contradiction(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.contradiction_engine"
    if "contradiction_debt" not in blob and "active_tensions" not in blob:
        return {"lifecycle": unknown("no contradiction counters", source=src)}
    out = {"lifecycle": Fact("active", Provenance.MEASURED, source=src)}
    for f in ("contradiction_debt", "active_tensions"):
        if f in blob:
            out[f] = Fact(blob.get(f), Provenance.MEASURED, source=src)
    return out


def read_soul_integrity(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.soul_integrity"
    ci = blob.get("current_index")
    if ci is None:
        return {"lifecycle": unknown("no soul integrity index", source=src)}
    # the index is the system's own composite self-assessment -> self_scored
    return {
        "lifecycle": Fact("critical" if blob.get("critical") else "ok",
                          Provenance.MEASURED, note="repair_needed flag", source=src),
        "current_index": Fact(_num(ci), Provenance.SELF_SCORED,
                              note="self-computed integrity index, not external", source=src),
    }


def read_quarantine(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "epistemic.quarantine"
    comp = blob.get("composite")
    if comp is None and "pressure" not in blob:
        return {"lifecycle": unknown("no quarantine pressure", source=src)}
    out = {"lifecycle": Fact("active", Provenance.MEASURED, note="anomaly monitor", source=src)}
    if comp is not None:
        out["pressure"] = Fact(_num(comp), Provenance.INTERNALLY_SCORED,
                               note="multi-signal anomaly score", source=src)
    return out


# --- Memory / consciousness self-report -------------------------------------

def read_memory(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "memory"
    # engine.get_memory_stats() reports the count as "total" (also accept legacy names)
    tm = blob.get("total", blob.get("total_memories", blob.get("memory_count")))
    if tm is None:
        return {"lifecycle": unknown("no memory totals in this section", source=src)}
    out = {
        "lifecycle": Fact("active", Provenance.MEASURED, source=src),
        "total_memories": Fact(tm, Provenance.MEASURED, source=src),
    }
    if blob.get("core_count") is not None:
        out["core_count"] = Fact(blob.get("core_count"), Provenance.MEASURED, source=src)
    return out


def read_evolution(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "consciousness.evolution"
    # stage/transcendence are self-reported -> self_scored (never measurement/proof)
    return {"lifecycle": Fact(str(blob.get("stage")), Provenance.SELF_SCORED,
                              note=f"transcendence={blob.get('transcendence_level')} — self-reported, not proof",
                              source=src)}


def read_observer(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "consciousness.observer"
    out: dict[str, Fact] = {}
    if blob.get("observation_count") is not None:
        out["lifecycle"] = Fact("active", Provenance.MEASURED, source=src)
        out["observation_count"] = Fact(blob.get("observation_count"), Provenance.MEASURED, source=src)
    else:
        out["lifecycle"] = unknown("no observation counter", source=src)
    if blob.get("awareness_level") is not None:
        out["awareness_level"] = Fact(_num(blob.get("awareness_level")), Provenance.SELF_SCORED,
                                      note="self-reported awareness, not a measurement", source=src)
    return out


def read_consciousness(blob: dict[str, Any]) -> dict[str, Fact]:
    src = "consciousness"
    return {"lifecycle": Fact({"stage": blob.get("stage"),
                               "awareness_level": blob.get("awareness_level"),
                               "transcendence_level": blob.get("transcendence_level")},
                              Provenance.SELF_SCORED,
                              note="self-reported by the system — NOT a measurement or proof",
                              source=src)}


# Registry: cache_key -> adapter. Adapters taking the world_model blob are special-cased
# in gather (simulator reads from inside world_model).
ADAPTERS = {
    "world_model": read_world_model,
    "policy": read_policy,
    "hemisphere": read_hemisphere,
    "self_improve": read_self_improve,
    "skills": read_skills,
    "autonomy": read_autonomy,
    "grounding_ring": read_grounding_ring,
    "companion_read": read_companion_read,
    "belief_graph": read_belief_graph,
    "truth_calibration": read_truth_calibration,
    "reflective_audit": read_reflective_audit,
    "contradiction": read_contradiction,
    "soul_integrity": read_soul_integrity,
    "quarantine": read_quarantine,
    "memory": read_memory,
    "evolution": read_evolution,
    "observer": read_observer,
    "consciousness": read_consciousness,
}
