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

from cognition.self_view.provenance import Fact, Provenance, gap, unknown

logger = logging.getLogger("jarvis.self_view")

# ── P0.5: subsystem inventory sourced from the dashboard build_cache snapshot ──
# Curated set of snapshot keys the OSV reports as subsystems. Each is classified into a
# lifecycle Fact (provenance-honest). Keys NOT listed but present in the snapshot are still
# surfaced generically (present/unknown) so the OSV reflects the real surface; keys listed
# but ABSENT degrade to a gap. We read the snapshot's OUTPUT, never dashboard UI formatting.

# Subsystems whose headline values are SELF-REPORTED (the system grading itself) — these can
# never be measurements/proof. Honesty-critical: consciousness "awareness/transcendence".
_SELF_SCORED_KEYS = {"consciousness", "evolution"}

# Display names for the curated surface (others still surface by raw key).
_SUBSYSTEM_LABELS = {
    "consciousness": "Consciousness (self-report)",
    "evolution": "Consciousness evolution (self-report)",
    "observer": "Observer (self-monitoring)",
    "policy": "Policy NN",
    "self_improve": "Self-improvement pipeline",
    "world_model": "World model",
    "simulator": "Mental simulator",
    "mutations": "Kernel mutations",
    "language": "Language corpus",
    "kernel": "Kernel performance",
    "memory": "Memory / QSFS",
    "hemisphere": "Hemisphere specialists (NN)",
    "autonomy": "Autonomy / drives",
    "grounding_ring": "Spark / grounding ring",
    "companion_read": "Companion cognition",
    "belief_graph": "Belief graph",
    "truth_calibration": "Truth calibration",
    "reflective_audit": "Reflective audit (Layer-9)",
    "quarantine": "Quarantine pressure",
    "soul_integrity": "Soul integrity index",
    "skills": "Skills / capabilities",
    "self_view": "Operational self-view",
}


def _headline(blob: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {f: blob.get(f) for f in fields if f in blob}


def _classify_subsystem(key: str, blob: Any) -> Fact:
    """Map a snapshot subsystem blob to a provenance-honest lifecycle Fact.

    Generic lifecycle detection (shadow / advisory / active / dormant) over the common
    fields, with explicit overrides for self-reported subsystems. Unknown shapes degrade
    to 'present' (provenance unknown) — honest, not a fabricated state.
    """
    if not isinstance(blob, dict) or not blob:
        return gap("subsystem present in snapshot but empty/unreadable", source=f"snapshot.{key}")

    # Override: self-reported subsystems (awareness/transcendence/stage are self-scored)
    if key in _SELF_SCORED_KEYS:
        return Fact(_headline(blob, ("stage", "awareness_level", "transcendence_level",
                                     "reasoning_quality", "confidence_avg")),
                    Provenance.SELF_SCORED,
                    note="self-reported by the system — NOT a measurement or proof",
                    source=f"snapshot.{key}")

    level_name = blob.get("level_name")
    mode = blob.get("mode")
    active = blob.get("active")
    shadow_only = blob.get("shadow_only")

    shadow = (level_name == "shadow") or (mode in ("shadow", "canary")) or (shadow_only is True)
    advisory = (level_name == "advisory") or (mode == "advisory")
    is_active = (level_name == "active") or (mode in ("live", "active")) or (active is True)

    if shadow:
        return Fact("shadow", Provenance.SHADOW_ONLY,
                    note=_shadow_note(key, blob), source=f"snapshot.{key}")
    if advisory:
        return Fact("advisory", Provenance.ADVISORY,
                    note=f"advisory tier ({blob.get('total_validated', '?')} validated)",
                    source=f"snapshot.{key}")
    if active is False:
        return Fact("dormant", Provenance.DORMANT,
                    note=str(blob.get("reason") or "active=false"), source=f"snapshot.{key}")
    if is_active:
        return Fact("active", Provenance.MEASURED,
                    note=_active_note(key, blob), source=f"snapshot.{key}")

    # Present but no recognizable lifecycle signal — honest 'present/unknown'.
    return Fact("present", Provenance.UNKNOWN,
                note="present in snapshot; no curated lifecycle reader", source=f"snapshot.{key}")


def _shadow_note(key: str, blob: dict[str, Any]) -> str:
    if key == "policy":
        return f"nn_win_rate={blob.get('nn_win_rate')} (shadow NN-vs-kernel; advisory/non-authoritative)"
    lv = blob.get("total_validated")
    return f"shadow, not yet advisory" + (f" ({lv} validated)" if lv is not None else "")


def _active_note(key: str, blob: dict[str, Any]) -> str:
    if key == "self_improve":
        return f"active stage={blob.get('stage')} dry_run={blob.get('effective_dry_run')} (gated pipeline)"
    lv = blob.get("total_validated")
    return "active" + (f" ({lv} validated)" if lv is not None else "")


def subsystems_from_cache(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Build the subsystem inventory from the build_cache snapshot. Read-only."""
    if not isinstance(snapshot, dict) or not snapshot:
        return {"_meta": gap("dashboard snapshot unavailable", "dashboard.build_cache")}
    inventory: dict[str, Any] = {}
    # Only CURATED real subsystems — do not sweep every snapshot section (scene, sensors,
    # traits, etc. are not subsystems; labeling them "unknown subsystem" would be inflated
    # and misleading). Curated-present -> classify; curated-absent -> omit (not every build
    # carries every key). Richer per-subsystem readers are added incrementally.
    for key in _SUBSYSTEM_LABELS:
        if key in snapshot:
            inventory[key] = _classify_subsystem(key, snapshot.get(key))
    return inventory

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
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}

    # --- P0.5: broad subsystem inventory from the dashboard build_cache snapshot ---
    # This is the OSV's view of the REAL live subsystem surface (80+), classified into
    # provenance-honest lifecycle facts — not the ~7 hand-picked readouts.
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
