"""Self-generating NN-fleet view + universe model.

The static audit (``nn_fleet_registry.json``) holds the DESIGN facts (purpose, end_state, family, the
validated as-designed classifications). This module overlays LIVE telemetry at request time — per-focus
lifecycle/accuracy/broadcast from the matrix report, lived/synthetic sample counts from the
DistillationCollector, and **consumed_now** (is the consumer actually firing on this host) — so the
fleet view is always current and **cannot drift green** as it scales toward hundreds of NNs.

``inference_consumed`` = a consumer *wire* exists (design). ``consumed_now`` = that wire is firing
here. A gated world-model inject at promotion level 0 is confirmed-live in the audit and still
``consumed_now=False``.

``build_nn_universe`` shapes the same fused data as a cosmic-web graph: every NN + the baseline LLM(s) as
NODES (clustered into family "galaxies"), connected by real data/signal-path EDGES (teacher->student
distillation, family membership). The /v2 universe map renders it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

_REGISTRY = Path(__file__).resolve().parents[1] / "nn_fleet_registry.json"
_JARVIS_HOME = Path.home() / ".jarvis"

CONSUME_NOTE = (
    "wiring_confirmed is the design audit. live_state + consumed_now are this host. "
    "inference_consumed means a consumer wire exists, not that it is firing."
)

# Best-effort NN-type tagging (for legend/colour). Most of her specialists are small feedforward
# encoders/approximators; the baseline LLM is a transformer teacher.
_TYPE_BY_NAME = {
    "memory_ranker": "feedforward", "memory_salience": "feedforward",
    "self_sensing": "feedforward-predictor", "world_model": "heuristic",
    "simulator": "heuristic", "policy_nn": "feedforward",
    "shadow_language_model": "metric-encoder", "language_phasec": "ngram",
    "audio_emotion": "cnn-frozen",
}


def _load_registry() -> dict[str, Any]:
    try:
        return json.loads(_REGISTRY.read_text())
    except Exception:
        return {"records": []}


def _live_sources(engine: Any) -> tuple[dict, dict]:
    """(matrix per-focus live state, collector per-teacher samples). Degrades to {} per source."""
    matrix: dict[str, dict] = {}
    samples: dict[str, dict] = {}
    try:
        rep = (engine.get_matrix_report() if engine else None) or {}
        for sp in rep.get("specialists", []) or []:
            if sp.get("focus"):
                matrix[sp["focus"]] = sp
    except Exception:
        pass
    try:
        from hemisphere.distillation import distillation_collector
        samples = (distillation_collector.get_stats() or {}).get("teachers", {}) or {}
    except Exception:
        pass
    return matrix, samples


def _live_state(rec: dict, sp: dict | None) -> str:
    """Auto-derived LIVE wiring state (cannot drift): the runtime truth where we have it, else the
    validated design classification."""
    if sp:
        if sp.get("in_broadcast"):
            return "live-earning"          # inference consumed in its broadcast lane
        stage = str(sp.get("lifecycle") or "")
        if stage in ("verified_probationary", "broadcast_eligible", "promoted"):
            return "earning"
        if stage in ("candidate_birth", "probationary_training"):
            return "training"
    # fall back to the validated design label
    wc = str(rec.get("wiring_confirmed") or "")
    return {
        "confirmed-live": "live-earning", "confirmed-shadow": "shadow",
        "confirmed-dormant": "dormant", "advisory-by-design": "advisory",
        "staged-by-design": "staged", "shadow-by-design": "shadow",
        "ORPHANED": "orphaned", "BROKEN": "broken",
    }.get(wc, "unknown")


def _read_promotion_json(name: str) -> dict[str, Any] | None:
    """Read a ~/.jarvis/*_promotion.json file. None if missing/unreadable. No side effects."""
    path = _JARVIS_HOME / name
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _live_world_model() -> dict[str, Any] | None:
    data = _read_promotion_json("world_model_promotion.json")
    if not data:
        return None
    level = int(data.get("level") or 0)
    return {
        "promotion_level": level,
        "promotion_level_name": {0: "shadow", 1: "advisory", 2: "active"}.get(level, "unknown"),
        "total_validated": int(data.get("total_validated") or 0),
        "synthetic_validated": int(data.get("synthetic_validated") or 0),
    }


def _live_simulator() -> dict[str, Any] | None:
    data = _read_promotion_json("simulator_promotion.json")
    if not data:
        return None
    level = int(data.get("level") or 0)
    return {
        "promotion_level": level,
        "promotion_level_name": {0: "shadow", 1: "advisory"}.get(level, "unknown"),
        "total_validated": int(data.get("total_validated") or 0),
        "synthetic_validated": int(data.get("synthetic_validated") or 0),
    }


def _live_intent_shadow() -> dict[str, Any] | None:
    """Runner state if the process has one. Fail-closed; never constructs a new live runner."""
    try:
        from reasoning import intent_shadow as _is
        runner = getattr(_is, "_runner", None)
        if runner is None:
            return None
        st = runner.get_state() or {}
        return {
            "level": st.get("level"),
            "observations_total": st.get("observations_total"),
            "nn_predictions_total": st.get("nn_predictions_total"),
            "rolling_agreement": st.get("rolling_agreement"),
            "rescues_applied": st.get("rescues_applied"),
            "primary_overrides_applied": st.get("primary_overrides_applied"),
            "ready_for_promotion": st.get("ready_for_promotion"),
            "gates": st.get("gates"),
        }
    except Exception:
        return None


def _live_weight_room() -> dict[str, Any] | None:
    try:
        from hemisphere.weight_room_gate import weight_room_gate
        st = weight_room_gate.get_status() or {}
        return {
            "enforces": bool(st.get("enforces")),
            "phase": st.get("phase"),
            "authority": st.get("authority"),
        }
    except Exception:
        return None


def _overlay_named_consume(rec: dict[str, Any]) -> None:
    """Correct live_state / consumed_now for named records whose design wire is gated.

    Does not flip any gate. Default consumed_now already equals inference_consumed.
    """
    name = rec.get("name")
    live = rec.setdefault("live", {})

    if name == "world_model":
        st = _live_world_model()
        if not st:
            rec["consumed_now"] = False
            rec["consumed_now_of"] = "llm_prompt_inject"
            rec["live_state"] = "gated"
            live["consume_note"] = "promotion file unread; mouth inject stays off until level>=1"
            return
        live.update(st)
        firing = int(st.get("promotion_level") or 0) >= 1
        rec["consumed_now"] = firing
        rec["consumed_now_of"] = "llm_prompt_inject"
        rec["live_state"] = "live-earning" if firing else "gated"
        live["consume_note"] = (
            "prompt inject on" if firing else "prompt inject OFF (promotion.level < 1)"
        )
        return

    if name == "simulator":
        st = _live_simulator()
        if st:
            live.update(st)
            firing = int(st.get("promotion_level") or 0) >= 1
            rec["consumed_now"] = firing
            rec["consumed_now_of"] = "planner_advisory"
            if not firing and rec.get("live_state") == "live-earning":
                rec["live_state"] = "gated"
        return

    if name == "intent_shadow":
        st = _live_intent_shadow()
        if st:
            live.update(st)
            level = str(st.get("level") or "shadow")
            firing = level in ("advisory", "primary")
            rec["consumed_now"] = firing
            rec["consumed_now_of"] = "router_rewrite"
            rec["live_state"] = {
                "shadow": "shadow",
                "advisory": "advisory",
                "primary": "live-earning",
            }.get(level, rec.get("live_state") or "shadow")
            preds = st.get("nn_predictions_total")
            live["consume_note"] = (
                f"NN predicting ({preds}) but pass-through at {level}"
                if not firing else f"router rewrite at {level}"
            )
        else:
            rec["consumed_now"] = False
            rec["consumed_now_of"] = "router_rewrite"
        return

    if name == "weight_room_gate":
        st = _live_weight_room()
        if st:
            live.update(st)
        else:
            live.setdefault("enforces", False)
        rec["consumed_now"] = bool(live.get("enforces"))
        rec["consumed_now_of"] = "promotion_authority"
        rec["live_state"] = "shadow" if not rec["consumed_now"] else rec.get("live_state")
        live["consume_note"] = "enforces=False (P2 would-block only)" if not rec["consumed_now"] else "enforcing"
        return


def build_fleet_view(engine: Any) -> dict[str, Any]:
    """Static design facts + live telemetry, fused. The trackable registry, self-updating."""
    reg = _load_registry()
    records = reg.get("records", [])
    matrix, samples = _live_sources(engine)
    by_state: Counter = Counter()
    for r in records:
        nm = r.get("name")
        sp = matrix.get(nm)
        r["live_state"] = _live_state(r, sp)
        live: dict[str, Any] = {}
        if sp:
            live.update(stage=sp.get("lifecycle"), accuracy=sp.get("accuracy"),
                        in_broadcast=sp.get("in_broadcast"), impact=sp.get("impact_score"))
        if nm in samples:
            live["samples"] = samples[nm]
        r["live"] = live
        r["consumed_now"] = bool(r.get("inference_consumed"))
        _overlay_named_consume(r)
        by_state[r["live_state"]] += 1
    by_consumed = Counter(bool(r.get("consumed_now")) for r in records)
    return {
        "generated_live": True,
        "total": len(records),
        "by_state": dict(by_state),
        "by_consumed_now": {"true": int(by_consumed[True]), "false": int(by_consumed[False])},
        "records": records,
        "note": CONSUME_NOTE,
    }


_UNIVERSE_SNAPSHOT = Path.home() / ".jarvis" / "nn_universe_snapshot.json"


def _apply_plasticity(nodes: list[dict], edges: list[dict]) -> dict[str, Any]:
    """Diff this universe against the last snapshot to surface GROWTH + NEURAL PLASTICITY — grounded,
    not decorative: new nodes = the brain grew a region; new edges = a connection formed; strength
    deltas = synaptic strengthening/weakening; missing edges = pruning. Snapshot is ONE overwritten
    file (bounded, never accumulates). Returns a plasticity summary; tags nodes/edges in place."""
    prev_nodes: set[str] = set()
    prev_edges: dict[str, float] = {}
    try:
        if _UNIVERSE_SNAPSHOT.exists():
            prev = json.loads(_UNIVERSE_SNAPSHOT.read_text())
            prev_nodes = set(prev.get("nodes", []))
            prev_edges = prev.get("edges", {}) or {}
    except Exception:
        pass

    new_nodes: list[str] = []
    for n in nodes:
        if n.get("is_hub"):
            continue
        if prev_nodes and n["id"] not in prev_nodes:
            n["plasticity"] = "new"          # GROWTH — a new NN/region appeared
            new_nodes.append(n["id"])

    forming = strengthening = weakening = 0
    cur_edges: dict[str, float] = {}
    for e in edges:
        eid = f"{e['source']}->{e['target']}:{e['kind']}"
        s = float(e.get("strength", 0.0))
        cur_edges[eid] = s
        if not prev_edges:
            continue
        if eid not in prev_edges:
            e["plasticity"] = "forming"      # a NEW connection between NNs
            forming += 1
        else:
            d = s - float(prev_edges.get(eid, 0.0))
            if d > 0.04:
                e["plasticity"] = "strengthening"; strengthening += 1
            elif d < -0.04:
                e["plasticity"] = "weakening"; weakening += 1
    pruned = [eid for eid in prev_edges if eid not in cur_edges] if prev_edges else []

    try:
        _UNIVERSE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        _UNIVERSE_SNAPSHOT.write_text(json.dumps(
            {"nodes": [n["id"] for n in nodes if not n.get("is_hub")], "edges": cur_edges}))
    except Exception:
        pass

    return {
        "had_baseline": bool(prev_nodes),   # False on first build (everything looks "new" otherwise)
        "new_nodes": new_nodes,
        "edges_forming": forming,
        "edges_strengthening": strengthening,
        "edges_weakening": weakening,
        "edges_pruned": len(pruned),
    }


def build_nn_universe(engine: Any) -> dict[str, Any]:
    """Cosmic-web / connectome model: NN + baseline-LLM nodes (clustered by family region), real
    signal-path edges weighted by live signal flow, with growth + plasticity diffed across builds."""
    fleet = build_fleet_view(engine)
    records = fleet["records"]

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    families: set[str] = set()
    samples_by_node: dict[str, int] = {}

    # baseline LLM — the central teacher "sun"
    baseline_model = None
    try:
        from reasoning.ollama_client import _default_model
        baseline_model = _default_model()
    except Exception:
        pass
    nodes.append({
        "id": "baseline_llm", "label": baseline_model or "baseline LLM",
        "family": "baseline", "nn_type": "transformer-baseline", "state": "teacher",
        "size": 6.0, "is_baseline": True,
        "note": "the teacher she distills from — never the destination",
    })
    families.add("baseline")

    name_set = set()
    for r in records:
        nm = r.get("name")
        if not nm:
            continue
        name_set.add(nm)
        fam = str(r.get("family") or "other")
        families.add(fam)
        live = r.get("live") or {}
        sm = live.get("samples") or {}
        samp = (sm.get("lived", 0) + sm.get("synthetic", 0)) if sm else 0
        samples_by_node[nm] = samp
        nodes.append({
            "id": nm, "label": nm, "family": fam,
            "nn_type": _TYPE_BY_NAME.get(nm, "feedforward-encoder"),
            "state": r.get("live_state", "unknown"),
            "size": 1.6 + min(2.6, samp / 80.0) + (0.8 if (live.get("in_broadcast")) else 0.0),
            "samples": samp,
            "accuracy": live.get("accuracy"),
            "purpose": (r.get("purpose") or "")[:140],
            "end_state": (r.get("end_state") or "")[:140],
        })

    # --- signal-path edges (strength = real signal flow, so plasticity is grounded) ---
    def _distill_strength(student: str) -> float:
        return round(0.18 + min(0.82, samples_by_node.get(student, 0) / 200.0), 3)

    # 1) baseline LLM teaches the native students it distills into
    for student in ("native_voice", "voice_seed_NEW", "native_reasoning", "reasoning_encoder"):
        if student in name_set:
            edges.append({"source": "baseline_llm", "target": student, "kind": "distill",
                          "strength": _distill_strength(student)})
    # 2) teacher_source -> NN where the teacher is itself a node (distillation feeds)
    for r in records:
        nm = r.get("name")
        tsrc = str(r.get("teacher_source") or "").strip()
        for cand in name_set:
            if cand != nm and cand in tsrc:
                edges.append({"source": cand, "target": nm, "kind": "distill",
                              "strength": _distill_strength(nm)})
                break
    # 3) family "galaxy"/region backbone — faint structural links that form the clusters
    fam_hubs = {f: f"hub::{f}" for f in families if f != "baseline"}
    for hid in fam_hubs.values():
        fam = hid.split("::", 1)[1]
        nodes.append({"id": hid, "label": fam, "family": fam, "nn_type": "hub",
                      "state": "hub", "size": 0.8, "is_hub": True})
    for n in nodes:
        if n.get("is_hub") or n.get("is_baseline"):
            continue
        hub = fam_hubs.get(n["family"])
        if hub:
            edges.append({"source": hub, "target": n["id"], "kind": "family", "strength": 0.12})

    plasticity = _apply_plasticity(nodes, edges)

    return {
        "generated_live": True,
        "baseline_model": baseline_model,
        "families": sorted(families),
        "by_state": fleet["by_state"],
        "plasticity": plasticity,
        "counts": {"nodes": len(nodes), "edges": len(edges), "nns": len(name_set)},
        "nodes": nodes,
        "edges": edges,
    }
