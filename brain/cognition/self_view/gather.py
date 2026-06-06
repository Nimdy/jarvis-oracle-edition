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
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.self_view")

# Only skills genuinely changed within this window count as "recent" — prevents
# months-old bootstrap skills from masquerading as new (a P1 honesty fix).
RECENT_SKILL_WINDOW_S = 30 * 86400


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
            now = time.time()
            sk = [x for x in (skills_summary.get("skills") or []) if isinstance(x, dict)]
            sk.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
            recent: list[dict[str, Any]] = []
            # Only EARNED skills (acquired via the learning pipeline) count as recent
            # changes — bootstrap skills carry a post-reset re-registration timestamp and
            # are seeded, not new capabilities; listing them as "new" is misleading.
            for x in sk:
                when = x.get("updated_at") or 0
                if x.get("learning_job_id") and when and (now - when) <= RECENT_SKILL_WINDOW_S:
                    recent.append({"name": x.get("skill_id"), "kind": "skill",
                                   "status": x.get("status"), "when": when})
            # code-level changes the skill registry can't see (e.g. CognitivePlanner)
            bh = _latest_build_history()
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
