"""Live source-gathering for the Operational Self-View.

Reads existing subsystem readouts into the plain ``sources`` dict the
:class:`SelfViewSynthesizer` consumes. STRICTLY READ-ONLY: every block is defensive —
on any failure or missing subsystem the key is omitted or degraded to a first-class gap,
never a fabricated default. No writes, no LLM.

P0.6: the subsystem inventory is sourced from the dashboard ``build_cache`` snapshot
(``_cache``) via small, bespoke, read-only adapters (``adapters.py``) — one per subsystem,
each refusing to guess when its fields are missing. We read the snapshot OUTPUT, never
dashboard UI formatting.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cognition.self_view.provenance import Fact, gap, unknown
from cognition.self_view.adapters import ADAPTERS, read_simulator

logger = logging.getLogger("jarvis.self_view")

# Only skills genuinely changed within this window count as "recent" — prevents
# months-old bootstrap skills from masquerading as new.
RECENT_SKILL_WINDOW_S = 30 * 86400


def subsystems_from_cache(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Build the subsystem inventory from the build_cache snapshot via bespoke adapters.

    Returns {name: {field: Fact}} — each subsystem classified by its own adapter. Curated
    subsystems present in the snapshot are read; absent ones are omitted; present-but-empty
    degrade to a gap; adapter errors degrade to unknown. Read-only.
    """
    if not isinstance(snapshot, dict) or not snapshot:
        return {"_meta": {"lifecycle": gap("dashboard snapshot unavailable", "dashboard.build_cache")}}

    inventory: dict[str, Any] = {}
    for key, adapter in ADAPTERS.items():
        if key not in snapshot:
            continue  # not in this build — omit (no fake gap spam)
        blob = snapshot.get(key)
        if not isinstance(blob, dict) or not blob:
            inventory[key] = {"lifecycle": gap("present but empty/unreadable",
                                               source=f"snapshot.{key}")}
            continue
        try:
            inventory[key] = adapter(blob)
        except Exception:
            logger.debug("self_view: adapter %s failed", key, exc_info=True)
            inventory[key] = {"lifecycle": unknown("adapter error", source=f"snapshot.{key}")}

    # simulator state is nested inside the world_model blob
    wm = snapshot.get("world_model")
    if isinstance(wm, dict) and wm:
        try:
            inventory["simulator"] = read_simulator(wm)
        except Exception:
            logger.debug("self_view: simulator adapter failed", exc_info=True)
            inventory["simulator"] = {"lifecycle": unknown("simulator adapter error")}

    return inventory


def _latest_build_history() -> dict[str, Any] | None:
    """Best-effort: the most-recent BUILD_HISTORY section = recent CODE-level changes.

    This is how the OSV sees capability additions that are code, not skills (e.g. the
    CognitivePlanner). Read-only; returns None (-> honest gap) if unreadable.
    """
    try:
        path = Path(__file__).resolve().parents[3] / "docs" / "BUILD_HISTORY.md"
        if not path.exists():
            return None
        title: str | None = None
        subs: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title is None:
                    title = line[3:].strip()
                    continue
                break  # reached the next (older) section
            if title is not None and line.startswith("### "):
                subs.append(line[4:].strip())
        if title:
            return {"name": title, "kind": "code_changeset", "items": subs[:5]}
    except Exception:
        logger.debug("self_view: BUILD_HISTORY read failed", exc_info=True)
    return None


def gather_live_sources(
    engine: Any = None,
    eval_snapshot: dict[str, Any] | None = None,
    skills_summary: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}

    # --- broad subsystem inventory from the dashboard build_cache snapshot (P0.6) ---
    sources["subsystems"] = subsystems_from_cache(snapshot)

    # --- scoreboard (#9 honest composite) from the eval snapshot ---
    if isinstance(eval_snapshot, dict):
        sb = eval_snapshot.get("scoreboard")
        if isinstance(sb, dict):
            sources["scoreboard"] = sb

    # --- skills registry (structural + recent changes) ---
    if isinstance(skills_summary, dict):
        sources["skills"] = skills_summary
        try:
            now = time.time()
            sk = [x for x in (skills_summary.get("skills") or []) if isinstance(x, dict)]
            sk.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
            recent: list[dict[str, Any]] = []
            # Only EARNED skills (acquired via the learning pipeline) count as recent —
            # bootstrap skills carry a post-reset re-registration timestamp; listing them
            # as "new" is misleading.
            for x in sk:
                when = x.get("updated_at") or 0
                if x.get("learning_job_id") and when and (now - when) <= RECENT_SKILL_WINDOW_S:
                    recent.append({"name": x.get("skill_id"), "kind": "skill",
                                   "status": x.get("status"), "when": when})
            bh = _latest_build_history()  # code-level changes the registry can't see
            if bh:
                recent.append(bh)
            if recent:
                sources["recent_changes"] = recent
        except Exception:
            logger.debug("self_view: recent-change derivation failed", exc_info=True)

    # --- world-model diagnostics: causal / simulator / promotions / planner ---
    try:
        cs = None
        if engine is not None:
            cs = getattr(engine, "_consciousness", None) or getattr(engine, "consciousness", None)
        wm = getattr(cs, "_world_model", None) if cs else None
        if wm is not None and hasattr(wm, "get_diagnostics"):
            diag = wm.get_diagnostics()
            if isinstance(diag, dict):
                for src_key, diag_key in (
                    ("causal", "causal"),
                    ("simulator", "simulator"),
                    ("simulator_promotion", "simulator_promotion"),
                    ("world_model_promotion", "promotion"),
                    ("cognitive_planner", "cognitive_planner"),
                ):
                    v = diag.get(diag_key)
                    if isinstance(v, dict):
                        sources[src_key] = v
    except Exception:
        logger.debug("self_view: world-model gather failed", exc_info=True)

    # --- counterfactual engine state ---
    try:
        from epistemic.counterfactual import get_counterfactual_engine
        cf = get_counterfactual_engine().get_state()
        if isinstance(cf, dict):
            sources["counterfactual"] = cf
    except Exception:
        logger.debug("self_view: counterfactual gather failed", exc_info=True)

    return sources
