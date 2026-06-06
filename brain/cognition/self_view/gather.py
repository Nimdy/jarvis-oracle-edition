"""Live source-gathering for the Operational Self-View (P0).

Reads existing subsystem readouts into the plain ``sources`` dict the
:class:`SelfViewSynthesizer` consumes. STRICTLY READ-ONLY: every block is defensive —
on any failure or missing subsystem the key is simply omitted, and the synthesizer
degrades it to a first-class GAP (never a fabricated default). No writes, no LLM.

Sources that are not yet cleanly extractable in P0 (policy win-rate, self-referential
belief extraction) are intentionally left out → they surface as honest gaps, not guesses.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("jarvis.self_view")


def gather_live_sources(
    engine: Any = None,
    eval_snapshot: dict[str, Any] | None = None,
    skills_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}

    # --- scoreboard (#9 honest composite) from the eval snapshot ---
    if isinstance(eval_snapshot, dict):
        sb = eval_snapshot.get("scoreboard")
        if isinstance(sb, dict):
            sources["scoreboard"] = sb

    # --- skills registry (structural + recent changes) ---
    if isinstance(skills_summary, dict):
        sources["skills"] = skills_summary
        try:
            sk = [x for x in (skills_summary.get("skills") or []) if isinstance(x, dict)]
            sk.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
            recent = [
                {"name": x.get("skill_id"), "kind": "skill",
                 "status": x.get("status"), "when": x.get("updated_at")}
                for x in sk[:3]
            ]
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
