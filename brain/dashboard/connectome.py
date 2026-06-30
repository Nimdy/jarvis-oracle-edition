"""Live connectome — a passive, one-way TAP on JARVIS's actual wiring (the Super-Synapse, SyntheticSoul
§3.3), discovered 100% from the CURRENT running system. Replaces the stale static topology.json.

Disciplines (David, 2026-06-30):
  1. LIVE wiring + state — nodes/edges discovered from the running EventBus + live registries, NOT a file.
     New NNs/hemispheres auto-appear because they emit/subscribe/register.
  2. ZERO-IMPACT — a SPAN, never inline: read-only snapshots under the bus lock (copy+release), no writes,
     all scanning off the emit thread + TTL-cached. The only hot-path cost is one O(1) dict write in
     ``note_event`` (called by the existing on_any tap).
  3. Deterministic-or-UNKNOWN — the emitter side isn't on the bus, so it's CODE-DERIVED with a provenance
     tag (SINGLE/MULTI_EMITTER_CODE/RELAYED_UNKNOWN); never narrated. Subscriber side is live + authoritative.
  4. EXPECTED-vs-ACTUAL — design facts come from the CI-locked subsystem_registry (purpose/does/
     expected_idle_state…); a deviation is flagged ONLY when designed=live AND actual=quiet AND the
     expected-idle-state does not sanction the quiet (the governance 6-check) — so self-improvement never
     targets as-designed structure.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

_BRAIN = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _BRAIN / "subsystem_registry.json"
_EVENTS_PY = _BRAIN / "consciousness" / "events.py"

# ---- liveness accumulator (the on_any tap calls note_event; O(1), bounded) ------------------------
_LIVENESS: dict[str, dict[str, float]] = {}
_LIVENESS_CAP = 400


def note_event(event_type: str) -> None:
    """Hot-path safe: record that an event_type just fired. O(1). Called by the existing on_any tap."""
    e = _LIVENESS.get(event_type)
    if e is None:
        if len(_LIVENESS) >= _LIVENESS_CAP:
            return
        _LIVENESS[event_type] = {"last": time.time(), "count": 1}
    else:
        e["last"] = time.time()
        e["count"] += 1


# ---- caches (TTL; everything heavy is off the bus thread) ----------------------------------------
_CACHE: dict[str, Any] = {"emit_map": None, "emit_ts": 0.0, "reg": None, "reg_ts": 0.0}
_TTL = 300.0


def _load_registry() -> list[dict]:
    import json
    if _CACHE["reg"] is not None and time.time() - _CACHE["reg_ts"] < _TTL:
        return _CACHE["reg"]
    try:
        d = json.loads(_REGISTRY_PATH.read_text())
        recs = d.get("subsystems") or d.get("records") or []
    except Exception:
        recs = []
    _CACHE["reg"] = recs
    _CACHE["reg_ts"] = time.time()
    return recs


_CONST_RE = re.compile(r'^([A-Z][A-Z0-9_]+)\s*=\s*["\']([a-z_]+:[a-z_0-9]+)["\']', re.M)
_EMIT_RE = re.compile(r'\.emit(?:_event)?\(\s*([A-Z][A-Z0-9_]+|["\'][a-z_]+:[a-z_0-9]+["\'])')


def _build_emit_map() -> dict[str, dict]:
    """Static, code-derived emitter map: event_string -> {files:set, tag}. Off hot path, TTL-cached."""
    if _CACHE["emit_map"] is not None and time.time() - _CACHE["emit_ts"] < _TTL:
        return _CACHE["emit_map"]
    # 1) const table (NAME -> "domain:verb") from events.py
    const: dict[str, str] = {}
    try:
        for m in _CONST_RE.finditer(_EVENTS_PY.read_text()):
            const[m.group(1)] = m.group(2)
    except Exception:
        pass
    # 2) emit sites across brain/**/*.py (skip tests / caches)
    emap: dict[str, set] = {}
    dynamic = 0
    for f in _BRAIN.rglob("*.py"):
        sp = str(f)
        if "/tests/" in sp or "__pycache__" in sp or sp.endswith("_test.py"):
            continue
        try:
            txt = f.read_text()
        except Exception:
            continue
        if ".emit" not in txt:
            continue
        rel = sp.split("/brain/", 1)[-1]
        rel = "brain/" + rel if not rel.startswith("brain/") else rel
        for m in _EMIT_RE.finditer(txt):
            tok = m.group(1)
            if tok[0] in "\"'":
                ev = tok.strip("\"'")
            else:
                ev = const.get(tok)
                if ev is None:
                    dynamic += 1
                    continue
            emap.setdefault(ev, set()).add(rel)
    out: dict[str, dict] = {}
    for ev, files in emap.items():
        tag = "SINGLE_EMITTER_CODE" if len(files) == 1 else "MULTI_EMITTER_CODE"
        out[ev] = {"files": sorted(files), "tag": tag}
    out["__dynamic_count__"] = {"files": [], "tag": str(dynamic)}
    _CACHE["emit_map"] = out
    _CACHE["emit_ts"] = time.time()
    return out


def _registry_index(recs: list[dict]) -> tuple[dict, list]:
    """(name->record, [(path_prefix, record)]) for joins."""
    by_name = {}
    by_path = []
    for r in recs:
        nm = r.get("name") or r.get("id")
        if nm:
            by_name[nm] = r
            by_name[_norm(nm)] = r
        for hf in (r.get("home_files") or []):
            p = str(hf).split(":", 1)[0].strip()
            if p:
                by_path.append((p, r))
    return by_name, by_path


def _norm(s: str) -> str:
    return re.sub(r"[ _\-]", "", str(s or "").lower())


def _file_to_node(rel_file: str, by_path: list) -> str | None:
    for p, r in by_path:
        if rel_file == p or rel_file.startswith(p.rsplit(".py", 1)[0]):
            return r.get("id") or r.get("name")
    # module-name fallback (e.g. brain/cognition/self_sensing.py -> self_sensing)
    base = rel_file.rsplit("/", 1)[-1].replace(".py", "")
    return base or None


def _handler_node(h: Any, by_path: list) -> tuple[str, str]:
    """(node_id, provenance) for a subscriber handler — bound instance > module-file > UNKNOWN."""
    inst = getattr(h, "__self__", None)
    if inst is not None:
        return (type(inst).__name__, "bound_instance")
    mod = getattr(h, "__module__", "") or ""
    if mod:
        rel = "brain/" + mod.replace(".", "/") + ".py"
        nid = _file_to_node(rel, by_path)
        if nid:
            return (nid, "module")
        return (mod.split(".")[-1], "module")
    return ("UNKNOWN_HANDLER", "unknown")


def _design_facts(node_id: str, by_name: dict, by_path: list) -> dict | None:
    r = by_name.get(node_id) or by_name.get(_norm(node_id))
    if r is None:
        rel = "brain/" + node_id  # not a path; try module fallback only if it looks like one
        for p, rec in by_path:
            if _norm(p.rsplit("/", 1)[-1].replace(".py", "")) == _norm(node_id):
                r = rec
                break
    if r is None:
        return None
    return {
        "purpose": r.get("purpose"), "does": (r.get("does") or "")[:240],
        "status": r.get("status"), "authority": r.get("authority"),
        "expected_idle_state": (r.get("expected_idle_state") or "")[:240],
        "gate": r.get("gate"), "common_misread": r.get("common_misread"),
        "home_files": r.get("home_files"), "area": r.get("area"),
        "provenance": "designed-maturity (code-grounded, not live)",
    }


def _classify(node: dict) -> str:
    """EXPECTED-vs-ACTUAL. Deviation ONLY when designed=live/shipped AND the node shows NO observed
    activity AND no idle sanction. A node with LIT edges (real signal flowing through it) is
    demonstrably active and is NEVER a deviation — this is the grounded actual-state for subsystem
    nodes that carry no NN-style `state`. So self-improvement never targets working/as-designed structure."""
    df = node.get("design") or {}
    if not df:
        return "untracked"
    # ACTUAL activity, grounded: a live-earning NN state OR demonstrably-firing edges (lit).
    active = node.get("state") in ("live-earning", "earning", "training") or bool(node.get("_lit"))
    designed_live = df.get("status") in ("shipped",) and df.get("authority") in ("live", "active")
    if active or not designed_live:
        return "as-designed"
    idle = (df.get("expected_idle_state") or "").lower()
    if any(k in idle for k in ("zero", "empty", "never", "after reset", "idle", "cold",
                               "no one", "absent", "not present", "dropout", "0")):
        return "as-designed-idle"
    return "DEVIATION"


def build_connectome(engine: Any) -> dict[str, Any]:
    """Assemble the live connectome (read-only, zero-write). nodes + edges + meta."""
    recs = _load_registry()
    by_name, by_path = _registry_index(recs)
    emit_map = _build_emit_map()
    nodes: dict[str, dict] = {}

    def _add(nid: str, **kw):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": kw.get("label", nid)}
        nodes[nid].update({k: v for k, v in kw.items() if v is not None})

    # 1) live NN fleet (no disk writes)
    try:
        from dashboard.nn_fleet import build_fleet_view
        for r in build_fleet_view(engine).get("records", []):
            nm = r.get("name")
            if nm:
                _add(nm, kind="nn", family=r.get("family"), state=r.get("live_state"))
    except Exception:
        pass
    # 2) live hemisphere specialists not yet in the fleet registry (self-grow)
    try:
        rep = (engine.get_matrix_report() if engine else None) or {}
        for sp in rep.get("specialists", []) or []:
            foc = sp.get("focus")
            if foc and foc not in nodes:
                st = "live-earning" if sp.get("in_broadcast") else "training"
                _add(foc, kind="nn", family="hemisphere", state=st)
    except Exception:
        pass
    # 3) live subscribers from the EventBus (snapshot under lock, copy+release)
    listeners: dict[str, list] = {}
    try:
        from consciousness.events import event_bus
        with event_bus._lock:
            listeners = {et: list(hs) for et, hs in event_bus._listeners.items()}
    except Exception:
        pass

    edges: list[dict] = []
    now = time.time()
    for et, handlers in listeners.items():
        em = emit_map.get(et) or {}
        emitter_nodes = [n for f in em.get("files", []) if (n := _file_to_node(f, by_path))]
        prov = em.get("tag", "RELAYED_UNKNOWN" if not emitter_nodes else "SINGLE_EMITTER_CODE")
        lv = _LIVENESS.get(et)
        lit = bool(lv and now - lv["last"] < 60.0)
        for h in handlers:
            sub, sprov = _handler_node(h, by_path)
            _add(sub, kind=nodes.get(sub, {}).get("kind", "subsystem"))
            srcs = emitter_nodes or ["UNKNOWN_ORIGIN"]
            for src in srcs:
                if src != sub:
                    _add(src, kind=nodes.get(src, {}).get("kind", "subsystem"))
                    edges.append({"source": src, "target": sub, "event": et, "provenance": prov,
                                  "fired": lit, "count": (lv or {}).get("count", 0),
                                  "multi": len(srcs) > 1})
    # 4) constant-id singletons
    _add("consciousness_kernel", kind="kernel", label="Consciousness Kernel")
    try:
        from reasoning.ollama_client import _default_model
        _add("baseline_llm", kind="baseline", label=_default_model() or "baseline LLM")
    except Exception:
        _add("baseline_llm", kind="baseline", label="baseline LLM")

    # mark nodes touched by a currently-lit edge — demonstrably active, never a deviation
    for e in edges:
        if e.get("fired"):
            for end in (e.get("source"), e.get("target")):
                if end in nodes:
                    nodes[end]["_lit"] = True

    # 5) annotate design facts + deviation
    dev_counts: dict[str, int] = {}
    for nid, n in nodes.items():
        df = _design_facts(nid, by_name, by_path)
        if df:
            n["design"] = df
        n["deviation"] = _classify(n)
        dev_counts[n["deviation"]] = dev_counts.get(n["deviation"], 0) + 1

    node_list = list(nodes.values())
    return {
        "generated_live": True,
        "source": "live-tap (EventBus._listeners + matrix + fleet) + code-derived emitters + CI-locked registry facts",
        "counts": {"nodes": len(node_list), "edges": len(edges),
                   "subscribed_event_types": len(listeners),
                   "emit_map_events": len([k for k in emit_map if not k.startswith("__")]),
                   "dynamic_emit_sites": emit_map.get("__dynamic_count__", {}).get("tag")},
        "deviations": dev_counts,
        "nodes": node_list,
        "edges": edges,
    }
