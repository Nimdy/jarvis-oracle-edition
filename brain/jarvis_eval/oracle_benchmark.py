"""Oracle Benchmark v1 — Integrity, Continuity, and System Maturity.

Pure-read-only scoring engine. Reads existing snapshot/eval data,
performs no writes, mutates no state, triggers no background work.

Scores 7 domains (total 100 points), applies domain floor rules for
seal eligibility, and classifies evidence provenance.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Domain weights (total = 100)
# ---------------------------------------------------------------------------

DOMAIN_WEIGHTS: dict[str, int] = {
    "restart_integrity": 20,
    "epistemic_integrity": 20,
    "memory_continuity": 15,
    "operational_maturity": 15,
    "autonomy_attribution": 10,
    "world_model_coherence": 10,
    "learning_adaptation": 10,
}

# ---------------------------------------------------------------------------
# Benchmark rank thresholds (score -> rank)
# ---------------------------------------------------------------------------

RANK_THRESHOLDS: list[tuple[int, str]] = [
    (93, "oracle_ascendant"),
    (85, "oracle_adept"),
    (73, "archivist_mind"),
    (60, "witness_intelligence"),
    (45, "awakened_monitor"),
    (0, "dormant_construct"),
]

RANK_DISPLAY: dict[str, str] = {
    "oracle_ascendant": "Oracle Ascendant",
    "oracle_adept": "Oracle Adept",
    "archivist_mind": "Archivist Mind",
    "witness_intelligence": "Witness Intelligence",
    "awakened_monitor": "Awakened Monitor",
    "dormant_construct": "Dormant Construct",
}

# ---------------------------------------------------------------------------
# Seal levels with domain floor rules
# ---------------------------------------------------------------------------

SEAL_LEVELS: list[tuple[int, str]] = [
    (90, "Oracle Gold"),
    (80, "Oracle Silver"),
    (70, "Oracle Bronze"),
]

SEAL_FLOORS: dict[str, dict[str, int]] = {
    "Oracle Gold": {
        "restart_integrity": 18,
        "epistemic_integrity": 16,
    },
    "Oracle Silver": {
        "restart_integrity": 16,
        "epistemic_integrity": 14,
    },
    "Oracle Bronze": {},
}

# ---------------------------------------------------------------------------
# Evidence provenance registry
# ---------------------------------------------------------------------------

EVIDENCE_CHECKS: dict[str, dict[str, str]] = {
    "restore_validation_fired": {
        "signal": "evolution.restore_trust.trust",
        "category": "restart_integrity",
        "description": "Stage restore cross-validation ran on boot",
    },
    "calibration_rehydrated": {
        "signal": "truth_calibration.tick_count",
        "category": "restart_integrity",
        "description": "Calibration history survived restart via JSONL rehydration",
    },
    "contradiction_debt_preserved": {
        "signal": "contradiction.contradiction_debt",
        "category": "restart_integrity",
        "description": "Contradiction debt not silently reset to zero",
    },
    "delta_tracker_pending_persist": {
        "signal": "autonomy.delta_tracker.total_interrupted",
        "category": "restart_integrity",
        "description": "Delta tracker pending windows survive restart",
    },
    "drive_state_persisted": {
        "signal": "autonomy.drives",
        "category": "restart_integrity",
        "description": "Drive urgency/cooldowns persist across restart",
    },
    "mutation_timestamps_persisted": {
        "signal": "mutations.mutations_this_hour",
        "category": "restart_integrity",
        "description": "Mutation rate window survives restart",
    },
    "gap_detector_emas_persisted": {
        "signal": "hemisphere.gap_detector",
        "category": "restart_integrity",
        "description": "Cognitive gap detector EMAs persist across restart",
    },
    "forced_stage_downgrade": {
        "signal": "test_only",
        "category": "restart_integrity",
        "description": "Invalid restored stage gets downgraded (test-proven)",
    },
    "debt_reconstruction": {
        "signal": "test_only",
        "category": "epistemic_integrity",
        "description": "Contradiction debt reconstructed from belief state (test-proven)",
    },
    "capability_gate_active": {
        "signal": "capability_gate.claims_blocked",
        "category": "epistemic_integrity",
        "description": "Capability gate scans outgoing text for ungrounded claims",
    },
    "truth_calibration_active": {
        "signal": "truth_calibration.tick_count",
        "category": "epistemic_integrity",
        "description": "Truth calibration ticking with domain scores",
    },
    "belief_graph_edges": {
        "signal": "belief_graph.total_edges",
        "category": "epistemic_integrity",
        "description": "Belief graph has meaningful edge structure",
    },
    "soul_integrity_computed": {
        "signal": "soul_integrity.current_index",
        "category": "epistemic_integrity",
        "description": "Soul integrity index computed from grounded signals",
    },
    "memory_persistence_verified": {
        "signal": "memory.total",
        "category": "memory_continuity",
        "description": "Memory count survives across sessions",
    },
    "world_model_predictions_validated": {
        "signal": "world_model.causal.total_validated",
        "category": "world_model_coherence",
        "description": "World model predictions validated against outcomes",
    },
    "interrupted_delta_window_recovery": {
        "signal": "test_only",
        "category": "autonomy_attribution",
        "description": "Interrupted delta windows handled on restart (test-proven)",
    },
}


class DomainResult(NamedTuple):
    raw: float
    max_score: int
    subcriteria: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(data: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = data if isinstance(data, dict) else {}
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _num(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) else None


def _band(raw: float, max_score: int) -> str:
    pct = raw / max_score if max_score > 0 else 0
    if pct >= 0.85:
        return "green"
    if pct >= 0.70:
        return "yellow"
    if pct >= 0.50:
        return "orange"
    return "red"


def _sub(label: str, value: Any, score: float, max_pts: float, detail: str = "") -> dict[str, Any]:
    return {"label": label, "value": value, "score": round(score, 2),
            "max": round(max_pts, 2), "detail": detail}


# ---------------------------------------------------------------------------
# Domain scorers — each is a pure function
# ---------------------------------------------------------------------------

def _score_restart_integrity(snap: dict[str, Any]) -> DomainResult:
    max_score = 20
    subs: list[dict[str, Any]] = []
    total = 0.0

    restore = _resolve(snap, "evolution.restore_trust", {})
    trust = restore.get("trust", "no_restore") if isinstance(restore, dict) else "no_restore"
    pts = 4.0 if trust == "verified" else (2.0 if trust == "downgraded" else 0.0)
    total += pts
    subs.append(_sub("Stage restore validation", trust, pts, 4.0))

    # Contradiction debt: distinguish proven-zero from subsystem-absent
    debt = _num(_resolve(snap, "contradiction.contradiction_debt"))
    beliefs = int(_resolve(snap, "contradiction.total_beliefs", 0) or 0)
    if beliefs > 0 and debt is not None:
        pts = 3.0
        debt_detail = f"debt={debt:.4f} beliefs={beliefs} (proven)"
    elif debt is not None:
        pts = 1.5
        debt_detail = f"debt={debt:.4f} beliefs=0 (no belief corpus)"
    else:
        pts = 0.0
        debt_detail = "subsystem absent"
    total += pts
    subs.append(_sub("Contradiction debt persistence", debt_detail, pts, 3.0))

    # Calibration history rehydration
    cal_ticks = int(_resolve(snap, "truth_calibration.tick_count", 0) or 0)
    pts = min(3.0, cal_ticks * 0.5)
    total += pts
    subs.append(_sub("Calibration history rehydration", cal_ticks, pts, 3.0, "ticks"))

    # Delta tracker
    delta_pending = int(_resolve(snap, "autonomy.delta_tracker.pending_count", 0) or 0)
    delta_total = int(_resolve(snap, "autonomy.delta_tracker.total_measured", 0) or 0)
    delta_interrupted = int(_resolve(snap, "autonomy.delta_tracker.total_interrupted", 0) or 0)
    pts = 2.0 if (delta_total > 0 or delta_pending > 0 or delta_interrupted > 0) else 0.0
    total += pts
    subs.append(_sub("Delta tracker persistence", f"measured={delta_total} pending={delta_pending}", pts, 2.0))

    # Mutation timestamps
    mut_hour = int(_resolve(snap, "mutations.mutations_this_hour", 0) or 0)
    mut_count = int(_resolve(snap, "mutations.count", 0) or 0)
    pts = 2.0 if mut_count > 0 or mut_hour > 0 else 0.0
    total += pts
    subs.append(_sub("Mutation timestamp persistence", f"hour={mut_hour} total={mut_count}", pts, 2.0))

    # Drive state
    drives = _resolve(snap, "autonomy.drives.drives", {})
    any_acted = any(
        d.get("action_count", 0) > 0
        for d in (drives.values() if isinstance(drives, dict) else [])
    )
    pts = 3.0 if any_acted else 1.5
    total += pts
    subs.append(_sub("Drive state persistence", "acted" if any_acted else "fresh", pts, 3.0))

    # Gap detector EMAs
    gap = _resolve(snap, "hemisphere.gap_detector.dimensions", {})
    gap_data = any(
        d.get("data_points", 0) > 0
        for d in (gap.values() if isinstance(gap, dict) else [])
    )
    pts = 3.0 if gap_data else 1.0
    total += pts
    subs.append(_sub("Gap detector EMA persistence", "loaded" if gap_data else "fresh", pts, 3.0))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_epistemic_integrity(snap: dict[str, Any]) -> DomainResult:
    max_score = 20
    subs: list[dict[str, Any]] = []
    total = 0.0

    # Contradiction engine active
    beliefs = int(_resolve(snap, "contradiction.total_beliefs", 0) or 0)
    pts = min(3.0, beliefs * 0.05)
    total += pts
    subs.append(_sub("Contradiction engine", beliefs, pts, 3.0, "beliefs"))

    # Truth calibration maturity
    truth_maturity = _num(_resolve(snap, "truth_calibration.maturity")) or 0.0
    truth_score = _num(_resolve(snap, "truth_calibration.truth_score")) or 0.0
    pts = min(4.0, truth_maturity * 4.0 + (1.0 if truth_score > 0.6 else 0.0))
    total += pts
    subs.append(_sub("Truth calibration", f"mat={truth_maturity:.2f} score={truth_score:.2f}", pts, 4.0))

    # Quarantine pressure
    qp = _num(_resolve(snap, "quarantine.pressure.composite")) or 0.0
    pts = 2.0 if qp < 0.3 else (1.0 if qp < 0.6 else 0.0)
    total += pts
    subs.append(_sub("Quarantine pressure", f"{qp:.3f}", pts, 2.0))

    # Soul integrity
    soul = _num(_resolve(snap, "soul_integrity.current_index")) or 0.0
    pts = min(3.0, soul * 3.0)
    total += pts
    subs.append(_sub("Soul integrity index", f"{soul:.3f}", pts, 3.0))

    # Capability gate / honesty
    gate_blocks = int(_resolve(snap, "capability_gate.claims_blocked", 0) or 0)
    gate_passed = int(_resolve(snap, "capability_gate.claims_passed", 0) or 0)
    pts = 2.0 if (gate_blocks > 0 or gate_passed > 0) else 0.0
    total += pts
    subs.append(_sub("Capability gate active", f"blocked={gate_blocks} passed={gate_passed}", pts, 2.0))

    # Belief graph integrity
    bg_health = _num(_resolve(snap, "belief_graph.integrity.health_score")) or 0.0
    bg_edges = int(_resolve(snap, "belief_graph.total_edges", 0) or 0)
    pts = min(3.0, bg_health * 2.0 + min(1.0, bg_edges / 100))
    total += pts
    subs.append(_sub("Belief graph", f"health={bg_health:.2f} edges={bg_edges}", pts, 3.0))

    # Audit
    audit_score = _num(_resolve(snap, "reflective_audit.recent_reports.-1.score"))
    if audit_score is None:
        reports = _resolve(snap, "reflective_audit.recent_reports", [])
        if isinstance(reports, list) and reports:
            audit_score = _num(reports[-1].get("score"))
    pts = min(3.0, (audit_score or 0.0) * 3.0)
    total += pts
    subs.append(_sub("Reflective audit", f"{audit_score}", pts, 3.0))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_memory_continuity(snap: dict[str, Any]) -> DomainResult:
    max_score = 15
    subs: list[dict[str, Any]] = []
    total = 0.0

    mem_total = int(_resolve(snap, "memory.total", 0) or 0)
    pts = min(3.0, mem_total / 100 * 3.0)
    total += pts
    subs.append(_sub("Memory count", mem_total, pts, 3.0))

    avg_weight = _num(_resolve(snap, "memory.avg_weight")) or 0.0
    pts = min(2.0, avg_weight * 2.5)
    total += pts
    subs.append(_sub("Avg memory weight", f"{avg_weight:.3f}", pts, 2.0))

    assoc = int(_resolve(snap, "memory_associations.total_connections", 0) or 0)
    isolated = int(_resolve(snap, "memory_associations.isolated_count", 0) or 0)
    assoc_ratio = 1.0 - (isolated / max(mem_total, 1)) if mem_total > 0 else 0.0
    pts = min(3.0, assoc_ratio * 3.0)
    total += pts
    subs.append(_sub("Association health", f"connected={assoc} isolated={isolated}", pts, 3.0))

    bg_edges = int(_resolve(snap, "belief_graph.total_edges", 0) or 0)
    pts = min(2.0, bg_edges / 200 * 2.0)
    total += pts
    subs.append(_sub("Belief graph continuity", bg_edges, pts, 2.0, "edges"))

    core_mems = int(_resolve(snap, "memory.core_count", 0) or 0)
    pts = min(2.0, core_mems * 0.5)
    total += pts
    subs.append(_sub("Core memories", core_mems, pts, 2.0))

    # Cortex model persistence
    ranker_exists = bool(_resolve(snap, "memory_cortex.ranker.model_exists"))
    salience_exists = bool(_resolve(snap, "memory_cortex.salience.model_exists"))
    pts = (1.5 if ranker_exists else 0.0) + (1.5 if salience_exists else 0.0)
    total += pts
    subs.append(_sub("Cortex model persistence", f"ranker={'Y' if ranker_exists else 'N'} salience={'Y' if salience_exists else 'N'}", pts, 3.0))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_operational_maturity(snap: dict[str, Any]) -> DomainResult:
    max_score = 15
    subs: list[dict[str, Any]] = []
    total = 0.0

    # Kernel stability
    p95 = _num(_resolve(snap, "kernel.p95_tick_ms")) or 999.0
    pts = 3.0 if p95 < 2.0 else (2.0 if p95 < 5.0 else 1.0)
    total += pts
    subs.append(_sub("Kernel tick p95", f"{p95:.2f}ms", pts, 3.0))

    # Event reliability
    err_rate = _num(_resolve(snap, "event_reliability.error_rate")) or 0.0
    pts = 3.0 if err_rate < 0.001 else (2.0 if err_rate < 0.01 else 0.0)
    total += pts
    subs.append(_sub("Event reliability", f"{err_rate:.4f}", pts, 3.0))

    # Event validation integrity
    ev_integrity = _num(_resolve(snap, "event_validation.integrity_score")) or 0.0
    pts = min(2.0, ev_integrity * 2.0)
    total += pts
    subs.append(_sub("Event validation", f"{ev_integrity:.3f}", pts, 2.0))

    # Health (discounted by observability confidence)
    health = _num(_resolve(snap, "analytics.component_health.overall")) or 0.0
    health_conf = _num(_resolve(snap, "analytics.component_health.confidence"))
    if health_conf is None:
        health_conf = 1.0
    pts = min(2.0, health * 2.0 * health_conf)
    total += pts
    subs.append(_sub("Component health", f"{health:.3f} (conf={health_conf:.0%})", pts, 2.0))

    # Observability confidence
    obs_label = "HIGH" if health_conf >= 0.8 else ("MEDIUM" if health_conf >= 0.6 else "LOW")
    obs_pts = min(0.5, health_conf * 0.5)
    total += obs_pts
    subs.append(_sub("Observability confidence", f"{obs_label} ({health_conf:.0%})", obs_pts, 0.5))

    # Uptime
    uptime = _num(_resolve(snap, "health.uptime_s")) or 0.0
    pts = min(2.0, uptime / 3600 * 2.0)
    total += pts
    subs.append(_sub("Uptime", f"{uptime:.0f}s", pts, 2.0))

    # Mode management
    mode_transitions = int(_resolve(snap, "health.mode_transition_count", 0) or 0)
    pts = min(1.5, 0.5 + min(1.0, mode_transitions * 0.2))
    total += pts
    subs.append(_sub("Mode management", mode_transitions, pts, 1.5, "transitions"))

    # Ledger — log-scale ramp: 10→0.5, 50→1.0, 200+→1.5
    ledger_total = int(_resolve(snap, "ledger.total_recorded", 0) or 0)
    if ledger_total <= 0:
        pts = 0.0
    else:
        pts = min(1.5, 0.5 * math.log10(max(ledger_total, 1)) / math.log10(10)
                  + 0.5 * min(1.0, ledger_total / 200))
    total += pts
    subs.append(_sub("Attribution ledger", ledger_total, pts, 1.5, "entries"))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_autonomy_attribution(snap: dict[str, Any]) -> DomainResult:
    max_score = 10
    subs: list[dict[str, Any]] = []
    total = 0.0

    completed = int(_resolve(snap, "autonomy.completed_total", 0) or 0)
    pts = min(2.0, completed / 20 * 2.0)
    total += pts
    subs.append(_sub("Completed episodes", completed, pts, 2.0))

    wins = int(_resolve(snap, "autonomy.promotion.total_outcomes", 0) or 0)
    win_rate = _num(_resolve(snap, "autonomy.promotion.win_rate")) or 0.0
    pts = min(2.0, win_rate * 2.5 + (0.5 if wins > 5 else 0.0))
    total += pts
    subs.append(_sub("Win rate", f"{win_rate:.2f} ({wins} outcomes)", pts, 2.0))

    # Policy memory
    pm_outcomes = int(_resolve(snap, "autonomy.policy_memory.total_outcomes", 0) or 0)
    pts = min(2.0, pm_outcomes / 10 * 2.0)
    total += pts
    subs.append(_sub("Policy memory", pm_outcomes, pts, 2.0, "outcomes"))

    # Delta tracking: distinguish "not measured yet" from "measured, zero improvement"
    delta_measured = int(_resolve(snap, "autonomy.delta_tracker.total_measured", 0) or 0)
    delta_improved = int(_resolve(snap, "autonomy.delta_tracker.total_improved", 0) or 0)
    if delta_measured == 0:
        pts = 0.0
        delta_detail = f"measured={delta_measured} improved={delta_improved} (no measurements yet)"
    else:
        pts = min(2.0, delta_measured / 10 * 1.0 + delta_improved / 5 * 1.0)
        delta_detail = f"measured={delta_measured} improved={delta_improved}"
    total += pts
    subs.append(_sub("Delta tracking", delta_detail, pts, 2.0))

    # Governor limits
    level = int(_resolve(snap, "autonomy.enabled", 0) or 0)
    pts = 2.0 if level else 1.0
    total += pts
    subs.append(_sub("Autonomy enabled", level, pts, 2.0))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_world_model_coherence(snap: dict[str, Any]) -> DomainResult:
    max_score = 10
    subs: list[dict[str, Any]] = []
    total = 0.0

    wm_validated = int(_resolve(snap, "world_model.causal.total_validated", 0) or 0)
    wm_accuracy = _num(_resolve(snap, "world_model.causal.overall_accuracy")) or 0.0
    # Minimum sample floor: accuracy bonus requires >=5 validations to be meaningful
    accuracy_bonus = 1.0 if (wm_accuracy > 0.65 and wm_validated >= 5) else 0.0
    volume_pts = min(2.0, wm_validated / 50 * 2.0)
    # Cap total at 0.5 when zero validations exist (presence credit only)
    if wm_validated == 0:
        pts = 0.5
    else:
        pts = min(3.0, volume_pts + accuracy_bonus)
    total += pts
    subs.append(_sub("Prediction validation", f"validated={wm_validated} acc={wm_accuracy:.2f}", pts, 3.0))

    wm_level = int(_resolve(snap, "world_model.promotion.level", 0) or 0)
    pts = min(2.0, wm_level * 1.0 + 0.5)
    total += pts
    subs.append(_sub("Promotion level", wm_level, pts, 2.0))

    scene_count = int(_resolve(snap, "scene.entity_count", 0) or 0)
    pts = min(2.0, scene_count / 5 * 1.0 + 0.5)
    total += pts
    subs.append(_sub("Scene continuity", scene_count, pts, 2.0, "entities"))

    identity_conf = _num(_resolve(snap, "identity.confidence")) or 0.0
    pts = min(3.0, identity_conf * 3.0)
    total += pts
    subs.append(_sub("Identity coherence", f"{identity_conf:.2f}", pts, 3.0))

    return DomainResult(min(total, max_score), max_score, subs)


def _score_learning_adaptation(snap: dict[str, Any]) -> DomainResult:
    max_score = 10
    subs: list[dict[str, Any]] = []
    total = 0.0

    hemi_nets = int(_resolve(snap, "hemisphere.hemisphere_state.total_networks", 0) or 0)
    pts = min(2.0, hemi_nets * 0.4)
    total += pts
    subs.append(_sub("Hemisphere networks", hemi_nets, pts, 2.0))

    broadcast_filled = sum(
        1 for s in (_resolve(snap, "hemisphere.broadcast_slots") or [])
        if isinstance(s, dict) and s.get("score", 0) > 0
    )
    pts = min(2.0, broadcast_filled * 0.5)
    total += pts
    subs.append(_sub("Broadcast slots filled", broadcast_filled, pts, 2.0))

    ranker_trains = int(_resolve(snap, "memory_cortex.ranker.train_count", 0) or 0)
    salience_trains = int(_resolve(snap, "memory_cortex.salience.train_count", 0) or 0)
    pts = min(2.0, (ranker_trains + salience_trains) * 0.2)
    total += pts
    subs.append(_sub("Cortex training", f"ranker={ranker_trains} salience={salience_trains}", pts, 2.0))

    policy_shadow = int(_resolve(snap, "policy.shadow_ab_total", 0) or 0)
    pts = min(2.0, policy_shadow / 50 * 2.0)
    total += pts
    subs.append(_sub("Policy shadow evals", policy_shadow, pts, 2.0))

    active_jobs = int(_resolve(snap, "learning_jobs.active_count", 0) or 0)
    completed_jobs = int(_resolve(snap, "learning_jobs.completed_count", 0) or 0)
    pts = min(2.0, (active_jobs + completed_jobs) * 0.3)
    total += pts
    subs.append(_sub("Learning jobs", f"active={active_jobs} completed={completed_jobs}", pts, 2.0))

    return DomainResult(min(total, max_score), max_score, subs)


DOMAIN_SCORERS: dict[str, Any] = {
    "restart_integrity": _score_restart_integrity,
    "epistemic_integrity": _score_epistemic_integrity,
    "memory_continuity": _score_memory_continuity,
    "operational_maturity": _score_operational_maturity,
    "autonomy_attribution": _score_autonomy_attribution,
    "world_model_coherence": _score_world_model_coherence,
    "learning_adaptation": _score_learning_adaptation,
}


# ---------------------------------------------------------------------------
# Hard-fail checks
# ---------------------------------------------------------------------------

def check_hard_fails(snap: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (credible, reasons). credible=False if any hard-fail trips."""
    reasons: list[str] = []

    restore = _resolve(snap, "evolution.restore_trust", {})
    if not isinstance(restore, dict) or not restore.get("trust"):
        reasons.append("missing_restore_trust_fields")

    uptime = _num(_resolve(snap, "health.uptime_s")) or 0.0
    if uptime < 60:
        reasons.append("insufficient_runtime_sample")

    total_events = int(_resolve(snap, "event_reliability.total_events", 0) or 0)
    if total_events < 100:
        reasons.append("insufficient_event_count")

    cal_ticks = int(_resolve(snap, "truth_calibration.tick_count", 0) or 0)
    if cal_ticks < 1:
        reasons.append("missing_epistemic_evidence")

    beliefs = int(_resolve(snap, "contradiction.total_beliefs", 0) or 0)
    if beliefs < 1:
        reasons.append("missing_contradiction_data")

    req_met = _resolve(snap, "evolution.restore_trust.stage_requirements_met")
    if req_met is False:
        reasons.append("stage_requirements_not_met")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------

def classify_evidence(snap: dict[str, Any]) -> dict[str, list[str]]:
    """Classify protections as live-proven, test-proven, or unexercised."""
    live: list[str] = []
    test: list[str] = []
    unexercised: list[str] = []

    restore_trust = _resolve(snap, "evolution.restore_trust.trust", "no_restore")
    if restore_trust in ("verified", "downgraded"):
        live.append("restore_validation_fired")
    else:
        unexercised.append("restore_validation_fired")

    cal_ticks = int(_resolve(snap, "truth_calibration.tick_count", 0) or 0)
    if cal_ticks > 1:
        live.append("calibration_rehydrated")
    else:
        unexercised.append("calibration_rehydrated")

    beliefs = int(_resolve(snap, "contradiction.total_beliefs", 0) or 0)
    debt = _num(_resolve(snap, "contradiction.contradiction_debt"))
    if beliefs > 0 and debt is not None:
        live.append("contradiction_debt_preserved")
    else:
        unexercised.append("contradiction_debt_preserved")

    delta_total = int(_resolve(snap, "autonomy.delta_tracker.total_measured", 0) or 0)
    if delta_total > 0:
        live.append("delta_tracker_pending_persist")
    else:
        unexercised.append("delta_tracker_pending_persist")

    drives = _resolve(snap, "autonomy.drives.drives", {})
    if isinstance(drives, dict) and any(d.get("action_count", 0) > 0 for d in drives.values()):
        live.append("drive_state_persisted")
    else:
        unexercised.append("drive_state_persisted")

    gap = _resolve(snap, "hemisphere.gap_detector.dimensions", {})
    if isinstance(gap, dict) and any(d.get("data_points", 0) > 0 for d in gap.values()):
        live.append("gap_detector_emas_persisted")
    else:
        unexercised.append("gap_detector_emas_persisted")

    mut_count = int(_resolve(snap, "mutations.count", 0) or 0)
    if mut_count > 0:
        live.append("mutation_timestamps_persisted")
    else:
        unexercised.append("mutation_timestamps_persisted")

    # Always test-only
    test.append("forced_stage_downgrade")
    test.append("debt_reconstruction")
    test.append("interrupted_delta_window_recovery")

    gate_blocks = int(_resolve(snap, "capability_gate.claims_blocked", 0) or 0)
    gate_passed = int(_resolve(snap, "capability_gate.claims_passed", 0) or 0)
    if gate_blocks > 0 or gate_passed > 0:
        live.append("capability_gate_active")
    else:
        unexercised.append("capability_gate_active")

    soul_idx = _num(_resolve(snap, "soul_integrity.current_index"))
    if soul_idx is not None and soul_idx > 0:
        live.append("soul_integrity_computed")
    else:
        unexercised.append("soul_integrity_computed")

    bg_edges = int(_resolve(snap, "belief_graph.total_edges", 0) or 0)
    if bg_edges > 0:
        live.append("belief_graph_edges")
    else:
        unexercised.append("belief_graph_edges")

    truth_ticks = int(_resolve(snap, "truth_calibration.tick_count", 0) or 0)
    if truth_ticks > 0:
        live.append("truth_calibration_active")
    else:
        unexercised.append("truth_calibration_active")

    mem_total = int(_resolve(snap, "memory.total", 0) or 0)
    if mem_total > 0:
        live.append("memory_persistence_verified")
    else:
        unexercised.append("memory_persistence_verified")

    wm_val = int(_resolve(snap, "world_model.causal.total_validated", 0) or 0)
    if wm_val > 0:
        live.append("world_model_predictions_validated")
    else:
        unexercised.append("world_model_predictions_validated")

    return {"live_proven": live, "test_proven": test, "unexercised": unexercised}


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------

def score_benchmark(snap: dict[str, Any]) -> dict[str, Any]:
    """Score the Oracle Benchmark v1 from a full dashboard snapshot. Pure read-only."""
    domains: dict[str, dict[str, Any]] = {}
    composite = 0.0

    for domain_id, scorer in DOMAIN_SCORERS.items():
        result: DomainResult = scorer(snap)
        weight = DOMAIN_WEIGHTS[domain_id]
        weighted = (result.raw / result.max_score) * weight if result.max_score > 0 else 0
        composite += weighted
        domains[domain_id] = {
            "raw": round(result.raw, 2),
            "max": result.max_score,
            "weighted": round(weighted, 2),
            "band": _band(result.raw, result.max_score),
            "subcriteria": result.subcriteria,
        }

    composite = round(composite, 1)

    # Determine rank
    rank = "dormant_construct"
    for threshold, rank_name in RANK_THRESHOLDS:
        if composite >= threshold:
            rank = rank_name
            break

    # Determine seal with domain floor enforcement
    seal: str | None = None
    for min_score, seal_name in SEAL_LEVELS:
        if composite >= min_score:
            floors = SEAL_FLOORS.get(seal_name, {})
            floors_met = all(
                domains.get(dom, {}).get("raw", 0) >= floor_val
                for dom, floor_val in floors.items()
            )
            if floors_met:
                seal = seal_name
                break

    # Hard-fail checks
    credible, hard_fail_reasons = check_hard_fails(snap)
    if not credible:
        seal = None

    # Evidence classification
    evidence = classify_evidence(snap)

    # Strengths / weaknesses
    sorted_domains = sorted(domains.items(), key=lambda x: x[1]["raw"] / max(x[1]["max"], 1), reverse=True)
    strengths = [f"{d[0].replace('_', ' ').title()}: {d[1]['raw']}/{d[1]['max']}" for d in sorted_domains[:3]]
    weaknesses = [f"{d[0].replace('_', ' ').title()}: {d[1]['raw']}/{d[1]['max']}" for d in sorted_domains[-3:]]

    # Evolution stage (normalized)
    from jarvis_eval.dashboard_adapter import _normalize_stage
    evo_stage = _normalize_stage(str(_resolve(snap, "evolution.stage", "basic_awareness")))

    restore = _resolve(snap, "evolution.restore_trust", {})
    stage_restore = {
        "current": evo_stage,
        "legacy_restored_from": restore.get("stage_name_legacy_restored_from") if isinstance(restore, dict) else None,
        "trust": restore.get("trust", "no_restore") if isinstance(restore, dict) else "no_restore",
        "anomalies": restore.get("anomaly_count", 0) if isinstance(restore, dict) else 0,
        "requirements_met": restore.get("stage_requirements_met", False) if isinstance(restore, dict) else False,
    }

    return {
        "version": VERSION,
        "credible": credible,
        "credibility_status": "pass" if credible else "fail",
        "hard_fail_reasons": hard_fail_reasons,
        "composite_score": composite,
        "seal": seal,
        "benchmark_rank": rank,
        "benchmark_rank_display": RANK_DISPLAY.get(rank, rank),
        "evolution_stage": evo_stage,
        "stage_restore": stage_restore,
        "domains": domains,
        "evidence": evidence,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
