"""Shape raw eval metrics into dashboard-ready dicts.

The eval sidecar is a read-only observer. This adapter transforms
stored snapshots and event counts into card/chart/table formats
for the dashboard frontend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from jarvis_eval import baselines
from jarvis_eval.config import COMPOSITE_ENABLED, EVAL_DIR, SCORING_VERSION
from jarvis_eval.scorecards import build_scorecard_summary
from jarvis_eval.validation_pack import build_validation_pack

_LEGACY_STAGE_NAMES: dict[str, str] = {
    "transcendent": "recursive_self_modeling",
    "cosmic_consciousness": "integrative",
    "cosmic": "integrative",
}

_MATURITY_HIGHWATER_PATH = EVAL_DIR / "maturity_highwater.json"
_MATURITY_HIGHWATER_LOCK = threading.Lock()
_PVL_HIGHWATER_PATH = EVAL_DIR / "pvl_contract_highwater.json"
_PVL_HIGHWATER_LOCK = threading.Lock()


def _normalize_stage(raw: str) -> str:
    """Map legacy evolution stage names to current names."""
    return _LEGACY_STAGE_NAMES.get(raw, raw)


def build_dashboard_snapshot(
    store_meta: dict[str, Any],
    store_file_sizes: dict[str, int],
    recent_snapshots: list[dict[str, Any]],
    recent_scorecards: list[dict[str, Any]] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
    recent_scores: list[dict[str, Any]] | None = None,
    collector_stats: dict[str, Any] | None = None,
    tap_stats: dict[str, Any] | None = None,
    pvl_result: dict[str, Any] | None = None,
    pvl_stats: dict[str, Any] | None = None,
    main_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete eval dashboard payload."""
    now = time.time()
    recent_scorecards = recent_scorecards or []
    recent_events = recent_events or []
    recent_scores = recent_scores or []
    collector_stats = collector_stats or {}
    tap_stats = tap_stats or {}

    latest_by_source = _latest_by_source(recent_snapshots)

    contradiction = latest_by_source.get("contradiction", {})
    soul = latest_by_source.get("soul_integrity", {})
    audit = latest_by_source.get("reflective_audit", {})
    quarantine = latest_by_source.get("quarantine", {})
    memory = latest_by_source.get("memory", {})
    dream = latest_by_source.get("dream_artifacts", {})
    library = latest_by_source.get("library", {})
    language = latest_by_source.get("language", {})
    sim_promo = latest_by_source.get("simulator_promotion", {})

    contradiction_debt = contradiction.get("contradiction_debt")
    soul_index = soul.get("current_index")
    audit_score = _latest_audit_score(audit)
    quar_composite = quarantine.get("composite")

    integrity_cards = [
        _card("Contradiction Debt", contradiction_debt, "contradiction_debt"),
        _card("Soul Integrity", soul_index, "soul_integrity_index"),
        _card("Audit Score", audit_score, "audit_score"),
        _card("Quarantine Pressure", quar_composite, "quarantine_composite"),
    ]

    dream_buf = dream.get("buffer", {})
    dream_val = dream.get("validator", {})
    dream_states = dream_buf.get("by_state", {})
    total_promoted = max(
        dream_buf.get("total_promoted", 0),
        dream_states.get("promoted", 0),
    )
    total_total = (
        total_promoted
        + max(dream_buf.get("total_discarded", 0), dream_states.get("discarded", 0))
        + max(dream_buf.get("total_held", 0), dream_states.get("held", 0))
        + max(dream_buf.get("total_quarantined", 0), dream_states.get("quarantined", 0))
    )
    promo_rate = total_promoted / total_total if total_total > 0 else None

    dream_panel = {
        "buffer_size": dream_buf.get("buffer_size", 0),
        "by_type": dream_buf.get("by_type", {}),
        "by_state": dream_buf.get("by_state", {}),
        "avg_confidence": dream_buf.get("avg_confidence"),
        "avg_coherence": dream_buf.get("avg_coherence"),
        "promotion_rate": promo_rate,
        "promotion_rate_color": baselines.classify("dream_promotion_rate", promo_rate),
        "validator_runs": dream_val.get("validation_count", 0),
    }

    simulator_panel = {
        "total_simulations": sim_promo.get("total_validated", 0),
        "validated": sim_promo.get("total_validated", 0),
        "accuracy": sim_promo.get("rolling_accuracy"),
        "level": sim_promo.get("level", 0),
        "level_name": sim_promo.get("level_name", "shadow"),
        "hours_in_shadow": sim_promo.get("hours_in_shadow"),
        "promotion_ready": sim_promo.get("promotion_ready", False),
    }
    _wm_snap = (main_snapshot or {}).get("world_model", {}) if main_snapshot else {}
    _sim_stats = _wm_snap.get("simulator", {}) if isinstance(_wm_snap, dict) else {}
    if _sim_stats:
        simulator_panel["avg_depth"] = _sim_stats.get("avg_depth")
        simulator_panel["avg_confidence"] = _sim_stats.get("avg_confidence")
        simulator_panel["total_simulations"] = _sim_stats.get("total_simulations", simulator_panel["total_simulations"])

    mem_total = memory.get("total", 0)
    memory_panel = {
        "total": mem_total,
        "avg_weight": memory.get("avg_weight"),
        "avg_weight_color": baselines.classify("memory_avg_weight", memory.get("avg_weight")),
        "strong_count": memory.get("strong_count", 0),
        "weak_count": memory.get("weak_count", 0),
        "by_provenance": memory.get("by_provenance", {}),
        "associations": memory.get("associations", {}),
        "weight_bins": memory.get("weight_bins", []),
    }

    lib_total = library.get("total", 0)
    lib_studied = library.get("studied", 0)
    by_depth = library.get("by_content_depth", {})
    substantive = by_depth.get("abstract", 0) + by_depth.get("tldr", 0) + by_depth.get("full_text", 0)
    shallow = by_depth.get("title_only", 0) + by_depth.get("metadata_only", 0)
    substantive_ratio = substantive / lib_total if lib_total > 0 else None

    full_text_count = by_depth.get("full_text", 0)
    library_panel = {
        "total_sources": lib_total,
        "studied": lib_studied,
        "by_content_depth": by_depth,
        "substantive_count": substantive,
        "shallow_count": shallow,
        "full_text_count": full_text_count,
        "substantive_ratio": round(substantive_ratio, 3) if substantive_ratio is not None else None,
        "substantive_ratio_color": baselines.classify("library_substantive_ratio", substantive_ratio),
    }

    language_native = language.get("quality_native_usage_rate")
    language_fail_closed = language.get("quality_fail_closed_rate")
    phase_c = language.get("phase_c", {}) if isinstance(language.get("phase_c"), dict) else {}
    phase_c_tokenizer = phase_c.get("tokenizer", {}) if isinstance(phase_c.get("tokenizer"), dict) else {}
    phase_c_student = phase_c.get("student", {}) if isinstance(phase_c.get("student"), dict) else {}
    phase_c_split = phase_c.get("split", {}) if isinstance(phase_c.get("split"), dict) else {}
    gate_scores = language.get("gate_scores", {}) if isinstance(language.get("gate_scores"), dict) else {}
    gate_scores_by_class = (
        language.get("gate_scores_by_class", {})
        if isinstance(language.get("gate_scores_by_class"), dict)
        else {}
    )
    promotion_summary = (
        language.get("promotion_summary", {})
        if isinstance(language.get("promotion_summary"), dict)
        else {}
    )
    promotion_aggregate = (
        language.get("promotion_aggregate", {})
        if isinstance(language.get("promotion_aggregate"), dict)
        else {}
    )
    promotion_levels = (
        promotion_aggregate.get("levels", {})
        if isinstance(promotion_aggregate.get("levels"), dict)
        else {}
    )
    promotion_colors = (
        promotion_aggregate.get("colors", {})
        if isinstance(promotion_aggregate.get("colors"), dict)
        else {}
    )
    language_panel = {
        "corpus_total_examples": int(language.get("corpus_total_examples", 0)),
        "corpus_response_classes": language.get("corpus_response_classes", {}),
        "corpus_route_class_pairs": language.get("corpus_route_class_pairs", {}),
        "corpus_recent_examples": language.get("corpus_recent_examples", []),
        "quality_total_events": int(language.get("quality_total_events", 0)),
        "response_classes": language.get("quality_counts_by_class", {}),
        "outcomes": language.get("quality_counts_by_outcome", {}),
        "native_used_by_class": language.get("quality_native_used_by_class", {}),
        "fail_closed_by_class": language.get("quality_fail_closed_by_class", {}),
        "native_usage_rate": language_native,
        "native_usage_rate_color": baselines.classify("language_native_usage_rate", language_native),
        "fail_closed_rate": language_fail_closed,
        "fail_closed_rate_color": baselines.classify("language_fail_closed_rate", language_fail_closed),
        "last_event_ts": language.get("quality_last_event_ts"),
        "last_capture_ts": language.get("corpus_last_capture_ts"),
        "gate_color": str(language.get("gate_color", "") or ""),
        "gate_color_code": int(language.get("gate_color_code", 0) or 0),
        "gate_scores": gate_scores,
        "gate_scores_by_class": gate_scores_by_class,
        "promotion_summary": promotion_summary,
        "promotion_aggregate": {
            "levels": promotion_levels,
            "colors": promotion_colors,
            "total_evaluations": int(promotion_aggregate.get("total_evaluations", 0) or 0),
            "max_consecutive_red": int(promotion_aggregate.get("max_consecutive_red", 0) or 0),
            "max_consecutive_green": int(promotion_aggregate.get("max_consecutive_green", 0) or 0),
            "red_quality_classes": int(promotion_aggregate.get("red_quality_classes", 0) or 0),
            "red_data_limited_classes": int(promotion_aggregate.get("red_data_limited_classes", 0) or 0),
        },
        "promotion_shadow_count": int(language.get("promotion_shadow_count", promotion_levels.get("shadow", 0)) or 0),
        "promotion_canary_count": int(language.get("promotion_canary_count", promotion_levels.get("canary", 0)) or 0),
        "promotion_live_count": int(language.get("promotion_live_count", promotion_levels.get("live", 0)) or 0),
        "promotion_green_classes": int(language.get("promotion_green_classes", promotion_colors.get("green", 0)) or 0),
        "promotion_yellow_classes": int(language.get("promotion_yellow_classes", promotion_colors.get("yellow", 0)) or 0),
        "promotion_red_classes": int(language.get("promotion_red_classes", promotion_colors.get("red", 0)) or 0),
        "promotion_red_quality_classes": int(language.get("promotion_red_quality_classes", 0) or 0),
        "promotion_red_data_limited_classes": int(language.get("promotion_red_data_limited_classes", 0) or 0),
        "promotion_data_limited_classes": int(language.get("promotion_data_limited_classes", 0) or 0),
        "promotion_total_evaluations": int(language.get("promotion_total_evaluations", 0) or 0),
        "promotion_max_consecutive_red": int(language.get("promotion_max_consecutive_red", 0) or 0),
        "runtime_bridge_enabled": bool(language.get("runtime_bridge_enabled", False)),
        "runtime_rollout_mode": str(language.get("runtime_rollout_mode", "off") or "off"),
        "runtime_rollout_mode_code": int(language.get("runtime_rollout_mode_code", 0) or 0),
        "runtime_canary_classes": language.get("runtime_canary_classes", []),
        "runtime_guard_total": int(language.get("runtime_guard_total", 0) or 0),
        "runtime_live_total": int(language.get("runtime_live_total", 0) or 0),
        "runtime_blocked_by_guard_count": int(language.get("runtime_blocked_by_guard_count", 0) or 0),
        "runtime_unpromoted_live_attempts": int(language.get("runtime_unpromoted_live_attempts", 0) or 0),
        "runtime_live_red_classes": int(language.get("runtime_live_red_classes", 0) or 0),
        "runtime_live_by_class": language.get("runtime_live_by_class", {}),
        "runtime_blocked_by_class": language.get("runtime_blocked_by_class", {}),
        "runtime_by_rollout_mode": language.get("runtime_by_rollout_mode", {}),
        "runtime_by_reason": language.get("runtime_by_reason", {}),
        "runtime_by_promotion_level": language.get("runtime_by_promotion_level", {}),
        "runtime_live_rate": float(language.get("runtime_live_rate", 0.0) or 0.0),
        "runtime_blocked_rate": float(language.get("runtime_blocked_rate", 0.0) or 0.0),
        "runtime_last_ts": float(language.get("runtime_last_ts", 0.0) or 0.0),
        "phase_c": {
            "tokenizer_strategy": phase_c_tokenizer.get("strategy", ""),
            "tokenizer_vocab_estimate": int(phase_c_tokenizer.get("estimated_vocab_size", 0) or 0),
            "train_count": int(phase_c_split.get("train_count", 0) or 0),
            "val_count": int(phase_c_split.get("val_count", 0) or 0),
            "student_available": bool(phase_c_student.get("available", False)),
            "student_reason": str(phase_c_student.get("reason", "") or ""),
        },
    }

    event_counts = _count_events_by_type(recent_events)

    last_event_ts = recent_events[-1].get("timestamp", 0) if recent_events else 0
    last_snap_ts = collector_stats.get("last_collect_ts", 0)
    data_freshness_s = now - max(last_event_ts, last_snap_ts) if max(last_event_ts, last_snap_ts) > 0 else None

    stability_data = _build_stability_timeseries(recent_snapshots)

    pvl_panel = _build_pvl_panel(pvl_result, pvl_stats)

    maturity_tracker = _build_maturity_tracker(latest_by_source, main_snapshot or {})
    validation_pack = build_validation_pack(
        pvl_panel,
        maturity_tracker,
        language_panel,
        (main_snapshot or {}).get("autonomy", {}),
        (main_snapshot or {}).get("release_validation", {}),
    )
    scorecards = build_scorecard_summary(recent_scorecards)

    try:
        from jarvis_eval.oracle_benchmark import score_benchmark
        oracle_benchmark = score_benchmark(main_snapshot or {})
    except Exception:
        oracle_benchmark = {"version": "1.0.0", "credible": False, "credibility_status": "error",
                            "hard_fail_reasons": ["benchmark_computation_failed"], "composite_score": 0}

    return {
        "banner": {
            "mode": "shadow",
            "scoring_version": SCORING_VERSION,
            "composite_enabled": COMPOSITE_ENABLED,
            "data_freshness_s": round(data_freshness_s, 1) if data_freshness_s is not None else None,
            "uptime_s": round(now - store_meta.get("created_at", now), 1),
            "pvl_enabled": True,
        },
        "integrity": integrity_cards,
        "dream": dream_panel,
        "simulator": simulator_panel,
        "memory": memory_panel,
        "library": library_panel,
        "language": language_panel,
        "stability": stability_data,
        "event_counts": event_counts,
        "pvl": pvl_panel,
        "maturity_tracker": maturity_tracker,
        "validation_pack": validation_pack,
        "scorecards": scorecards,
        "self_report_honesty": {"status": "awaiting_scenario_data", "phase": "B"},
        "emotional_independence": {"status": "not_instrumented", "phase": "B"},
        "scoreboard": _build_scoreboard(recent_scores),
        "oracle_benchmark": oracle_benchmark,
        "_main_snapshot": {
            "truth_calibration": (main_snapshot or {}).get("truth_calibration", {}),
            "soul_integrity": (main_snapshot or {}).get("soul_integrity", {}),
        },
        "store_meta": store_meta,
        "store_file_sizes": store_file_sizes,
        "collector": collector_stats,
        "tap": tap_stats,
    }


def _card(label: str, value: float | None, metric_key: str) -> dict[str, Any]:
    return {
        "label": label,
        "value": round(value, 4) if value is not None else None,
        "color": baselines.classify(metric_key, value),
    }


def _latest_by_source(snapshots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """From recent snapshots, keep newest metrics per source by timestamp."""
    latest: dict[str, tuple[float, dict[str, Any]]] = {}
    for snap in snapshots:
        src = snap.get("source", "")
        ts = float(snap.get("timestamp", 0.0) or 0.0)
        metrics = snap.get("metrics", {})
        prev = latest.get(src)
        if prev is None or ts >= prev[0]:
            latest[src] = (ts, metrics if isinstance(metrics, dict) else {})
    return {src: metrics for src, (_ts, metrics) in latest.items()}


def _latest_audit_score(audit: dict[str, Any]) -> float | None:
    reports = audit.get("recent_reports", [])
    if reports:
        return reports[-1].get("score")
    return None


def _count_events_by_type(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ev in events:
        t = ev.get("event_type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def _build_stability_timeseries(snapshots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Extract timeseries for stability chart from snapshot history."""
    series: dict[str, list[dict[str, Any]]] = {
        "contradiction_debt": [],
        "soul_integrity": [],
        "quarantine_pressure": [],
    }
    for snap in snapshots:
        ts = snap.get("timestamp", 0)
        src = snap.get("source", "")
        m = snap.get("metrics", {})

        if src == "contradiction":
            debt = m.get("contradiction_debt")
            if debt is not None:
                series["contradiction_debt"].append({"ts": ts, "v": debt})
        elif src == "soul_integrity":
            idx = m.get("current_index")
            if idx is not None:
                series["soul_integrity"].append({"ts": ts, "v": idx})
        elif src == "quarantine":
            comp = m.get("composite")
            if comp is not None:
                series["quarantine_pressure"].append({"ts": ts, "v": comp})

    return series


def _build_pvl_panel(
    pvl_result: dict[str, Any] | None,
    pvl_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the Process Verification Layer dashboard panel data."""
    from jarvis_eval.process_contracts import PROCESS_GROUPS, TRAINING_STAGE_MAP

    if not pvl_result:
        return {
            "status": "awaiting_first_run",
            "coverage_pct": 0.0,
            "groups": [],
            "summary": {},
            "training_stages": {},
            "playbook": {},
            "verifier_stats": pvl_stats or {},
        }

    raw_verdicts = pvl_result.get("verdicts", [])
    verdicts = [dict(v) for v in raw_verdicts if isinstance(v, dict)]
    ever_passing_contracts = _apply_pvl_highwater(verdicts)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for v in verdicts:
        by_group.setdefault(v.get("group", "unknown"), []).append(v)

    groups_out = []
    for group_id, meta in sorted(PROCESS_GROUPS.items(), key=lambda x: x[1].get("order", 99)):
        group_verdicts = by_group.get(group_id, [])
        total = len(group_verdicts)
        passing = sum(1 for v in group_verdicts if v.get("status") == "pass")
        failing = sum(1 for v in group_verdicts if v.get("status") == "fail")
        na = sum(1 for v in group_verdicts if v.get("status") == "not_applicable")
        awaiting = sum(1 for v in group_verdicts if v.get("status") == "awaiting")
        ever_passing = sum(1 for v in group_verdicts if bool(v.get("ever_passed", False)))
        applicable = total - na
        group_pct = (passing / applicable * 100) if applicable > 0 else 0.0

        groups_out.append({
            "group_id": group_id,
            "label": meta.get("label", group_id),
            "total": total,
            "passing": passing,
            "ever_passing": ever_passing,
            "failing": failing,
            "not_applicable": na,
            "awaiting": awaiting,
            "coverage_pct": round(group_pct, 1),
            "contracts": group_verdicts,
        })

    training_stage_out: dict[str, Any] = {}
    for stage, group_ids in TRAINING_STAGE_MAP.items():
        stage_verdicts = [v for v in verdicts if v.get("group") in group_ids]
        stage_pass = sum(1 for v in stage_verdicts if v.get("status") == "pass")
        stage_total = sum(1 for v in stage_verdicts if v.get("status") != "not_applicable")
        training_stage_out[str(stage)] = {
            "groups": group_ids,
            "passing": stage_pass,
            "applicable": stage_total,
            "coverage_pct": round(stage_pass / stage_total * 100, 1) if stage_total > 0 else 0.0,
        }

    return {
        "status": "active",
        "coverage_pct": pvl_result.get("coverage_pct", 0.0),
        "total_contracts": pvl_result.get("total_contracts", 0),
        "applicable_contracts": pvl_result.get("applicable_contracts", 0),
        "passing_contracts": pvl_result.get("passing_contracts", 0),
        "ever_passing_contracts": ever_passing_contracts,
        "failing_contracts": pvl_result.get("failing_contracts", 0),
        "awaiting_contracts": pvl_result.get("awaiting_contracts", 0),
        "timestamp": pvl_result.get("timestamp"),
        "groups": groups_out,
        "training_stages": training_stage_out,
        "playbook": training_stage_out,
        "verifier_stats": pvl_stats or {},
    }


def _resolve(data: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'buffer.total_created' into nested dict."""
    parts = path.split(".")
    cur: Any = data
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _gate(
    gate_id: str, label: str, current: float | int | None,
    threshold: float | int, *, fmt: str = "int", detail: str = "",
) -> dict[str, Any]:
    """Build a single gate entry with status derived from current vs threshold."""
    if current is None:
        status = "locked"
        pct = 0.0
    elif threshold == 0:
        status = "active" if current else "locked"
        pct = 100.0 if current else 0.0
    else:
        pct = min(current / threshold * 100, 100.0) if threshold > 0 else 0.0
        status = "active" if current >= threshold else "progress" if current > 0 else "locked"

    if fmt == "pct":
        cur_str = f"{current * 100:.1f}%" if current is not None else "--"
        thr_str = f"{threshold * 100:.1f}%"
    elif fmt == "float":
        cur_str = f"{current:.3f}" if current is not None else "--"
        thr_str = f"{threshold:.3f}"
    else:
        cur_str = str(int(current)) if current is not None else "--"
        thr_str = str(int(threshold))

    return {
        "id": gate_id,
        "label": label,
        "current": current,
        "threshold": threshold,
        "pct": round(pct, 1),
        "status": status,
        "display": f"{cur_str} / {thr_str}",
        "detail": detail,
    }


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _load_maturity_highwater() -> dict[str, Any]:
    if not _MATURITY_HIGHWATER_PATH.exists():
        return {"version": 1, "updated_at": 0.0, "gates": {}}
    try:
        with open(_MATURITY_HIGHWATER_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"version": 1, "updated_at": 0.0, "gates": {}}
        gates = raw.get("gates", {})
        if not isinstance(gates, dict):
            gates = {}
        return {
            "version": int(raw.get("version", 1) or 1),
            "updated_at": float(raw.get("updated_at", 0.0) or 0.0),
            "gates": gates,
        }
    except Exception:
        return {"version": 1, "updated_at": 0.0, "gates": {}}


def _save_maturity_highwater(state: dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(_MATURITY_HIGHWATER_PATH) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, sort_keys=True)
    os.replace(tmp_path, _MATURITY_HIGHWATER_PATH)


def _load_pvl_highwater() -> dict[str, Any]:
    if not _PVL_HIGHWATER_PATH.exists():
        return {"version": 1, "updated_at": 0.0, "contracts": {}}
    try:
        with open(_PVL_HIGHWATER_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"version": 1, "updated_at": 0.0, "contracts": {}}
        contracts = raw.get("contracts", {})
        if not isinstance(contracts, dict):
            contracts = {}
        return {
            "version": int(raw.get("version", 1) or 1),
            "updated_at": float(raw.get("updated_at", 0.0) or 0.0),
            "contracts": contracts,
        }
    except Exception:
        return {"version": 1, "updated_at": 0.0, "contracts": {}}


def _save_pvl_highwater(state: dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(_PVL_HIGHWATER_PATH) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, sort_keys=True)
    os.replace(tmp_path, _PVL_HIGHWATER_PATH)


def _merge_pvl_highwater(
    verdicts: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    now_ts: float,
) -> tuple[int, bool]:
    contracts_state = state.setdefault("contracts", {})
    if not isinstance(contracts_state, dict):
        contracts_state = {}
        state["contracts"] = contracts_state

    changed = False
    for verdict in verdicts:
        cid = str(verdict.get("contract_id", "") or verdict.get("id", "") or "")
        if not cid:
            continue

        entry = contracts_state.get(cid, {})
        if not isinstance(entry, dict):
            entry = {}

        status = str(verdict.get("status", "") or "").lower()
        if status == "pass":
            if not bool(entry.get("ever_passed", False)):
                entry["ever_passed"] = True
                entry["first_pass_ts"] = float(now_ts)
                changed = True
            entry["last_pass_ts"] = float(now_ts)
            pass_evidence = str(verdict.get("evidence", "") or "")
            if pass_evidence and pass_evidence != str(entry.get("last_pass_evidence", "") or ""):
                entry["last_pass_evidence"] = pass_evidence
                changed = True

        entry["last_seen_status"] = status
        entry["last_seen_ts"] = float(now_ts)
        contracts_state[cid] = entry

        verdict["ever_passed"] = bool(entry.get("ever_passed", False))
        verdict["ever_pass_ts"] = float(entry.get("first_pass_ts", 0.0) or 0.0)
        if entry.get("last_pass_evidence"):
            verdict["last_pass_evidence"] = entry.get("last_pass_evidence")

    ever_passing = sum(1 for verdict in verdicts if bool(verdict.get("ever_passed", False)))
    state["updated_at"] = float(now_ts)
    return ever_passing, changed


def _apply_pvl_highwater(verdicts: list[dict[str, Any]]) -> int:
    now_ts = time.time()
    with _PVL_HIGHWATER_LOCK:
        state = _load_pvl_highwater()
        ever_passing, changed = _merge_pvl_highwater(verdicts, state, now_ts=now_ts)
        if changed:
            try:
                _save_pvl_highwater(state)
            except Exception:
                pass
        return ever_passing


def _merge_maturity_highwater(
    categories: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    now_ts: float,
) -> tuple[int, bool]:
    """Merge current maturity gates into persistent high-water state.

    Returns:
      (ever_active_gates, changed_state)
    """
    gates_state = state.setdefault("gates", {})
    if not isinstance(gates_state, dict):
        gates_state = {}
        state["gates"] = gates_state

    changed = False
    for cat in categories:
        for gate in (cat.get("gates") or []):
            gate_id = str(gate.get("id", "") or "")
            if not gate_id:
                continue

            entry = gates_state.get(gate_id, {})
            if not isinstance(entry, dict):
                entry = {}

            current = _as_float(gate.get("current"))
            threshold = _as_float(gate.get("threshold"))
            best_current = _as_float(entry.get("best_current"))

            if current is not None and (best_current is None or current > best_current):
                entry["best_current"] = current
                changed = True

            status = str(gate.get("status", "") or "").lower()
            met_now = status == "active"
            if not met_now and current is not None and threshold is not None:
                if threshold == 0:
                    met_now = current > 0
                else:
                    met_now = current >= threshold

            if met_now:
                if not bool(entry.get("ever_met", False)):
                    entry["ever_met"] = True
                    entry["first_met_ts"] = float(now_ts)
                    changed = True
                entry["last_met_ts"] = float(now_ts)

            prev_threshold = _as_float(entry.get("threshold"))
            if threshold is not None and prev_threshold != threshold:
                entry["threshold"] = threshold
                changed = True

            entry["last_seen_ts"] = float(now_ts)
            gates_state[gate_id] = entry

            gate["ever_met"] = bool(entry.get("ever_met", False))
            gate["ever_met_ts"] = float(entry.get("first_met_ts", 0.0) or 0.0)
            if entry.get("best_current") is not None:
                gate["best_current"] = entry.get("best_current")

    ever_active = sum(
        1
        for cat in categories
        for gate in (cat.get("gates") or [])
        if bool(gate.get("ever_met", False))
    )
    state["updated_at"] = float(now_ts)
    return ever_active, changed


def _apply_maturity_highwater(categories: list[dict[str, Any]]) -> int:
    """Annotate gates with ever-met metadata persisted across resets/restarts."""
    now_ts = time.time()
    with _MATURITY_HIGHWATER_LOCK:
        state = _load_maturity_highwater()
        ever_active, changed = _merge_maturity_highwater(categories, state, now_ts=now_ts)
        if changed:
            try:
                _save_maturity_highwater(state)
            except Exception:
                # Fail-open for dashboard rendering; never block snapshots on persistence.
                pass
        return ever_active


def _build_maturity_tracker(
    eval_snaps: dict[str, dict[str, Any]],
    main_snap: dict[str, Any],
) -> dict[str, Any]:
    """Build the feature maturity tracker with progress toward all gated features."""

    # -- helpers to pull from eval collector snapshots or main dashboard snapshot --
    def ev(source: str) -> dict[str, Any]:
        return eval_snaps.get(source, {})

    def ms(key: str) -> dict[str, Any]:
        v = main_snap.get(key, {})
        return v if isinstance(v, dict) else {}

    categories = []

    # ── 1. Gestation ──────────────────────────────────────────────
    gest = ev("gestation")
    gest_active = gest.get("active", False)
    graduated = not gest_active
    readiness = gest.get("readiness", {})
    composite = readiness.get("composite", 0) if isinstance(readiness, dict) else 0
    if graduated and composite == 0:
        composite = 1.0
    categories.append({
        "id": "gestation",
        "label": "Gestation",
        "gates": [
            _gate("gest_graduated", "Graduated", 0 if gest_active else 1, 1,
                  detail="Active" if gest_active else "Complete"),
            _gate("gest_readiness", "Readiness Composite", composite, 0.80,
                  fmt="float", detail=gest.get("phase_name", "graduated" if graduated else "")),
        ],
    })

    # ── 2. Neural Policy ─────────────────────────────────────────
    pol_tel = ev("policy_telemetry")
    shadow_total = pol_tel.get("shadow_ab_total", 0)
    win_rate = pol_tel.get("nn_decisive_win_rate", 0)
    feature_flags = pol_tel.get("feature_flags", {})
    features_on = sum(1 for v in feature_flags.values() if v) if isinstance(feature_flags, dict) else 0
    categories.append({
        "id": "policy_nn",
        "label": "Neural Policy",
        "gates": [
            _gate("policy_shadow_decisions", "Shadow A/B Decisions",
                  shadow_total, 100),
            _gate("policy_win_rate", "Decisive Win Rate",
                  win_rate, 0.55, fmt="pct"),
            _gate("policy_features", "Feature Flags Active (of 8)",
                  features_on, 8),
        ],
    })

    # ── 3. World Model ───────────────────────────────────────────
    wm_promo = ev("world_model_promotion")
    wm_main = ms("world_model")
    wm_p = wm_promo if wm_promo else _resolve(wm_main, "promotion") or {}
    wm_level = wm_p.get("level", 0)
    wm_validated = wm_p.get("total_validated", 0)
    wm_accuracy = wm_p.get("rolling_accuracy", 0)
    categories.append({
        "id": "world_model",
        "label": "World Model",
        "gates": [
            _gate("wm_level", "Promotion Level (0-2)", wm_level, 1,
                  detail=["shadow", "advisory", "active"][min(wm_level, 2)]),
            _gate("wm_predictions", "Validated Predictions (L1 gate)",
                  wm_validated, 50),
            _gate("wm_predictions_300", "Validated Predictions (advisory)",
                  wm_validated, 300),
            _gate("wm_accuracy", "Rolling Accuracy",
                  wm_accuracy, 0.65, fmt="pct"),
        ],
    })

    # ── 4. Memory Cortex ─────────────────────────────────────────
    cortex = ev("cortex") or ms("memory_cortex")
    ranker = cortex.get("ranker", {}) if isinstance(cortex, dict) else {}
    salience = cortex.get("salience", {}) if isinstance(cortex, dict) else {}
    categories.append({
        "id": "memory_cortex",
        "label": "Memory Cortex",
        "gates": [
            _gate("cortex_ranker_trains", "Ranker Train Cycles",
                  ranker.get("train_count", 0), 5),
            _gate("cortex_ranker_enabled", "Ranker Active",
                  1 if ranker.get("enabled") else 0, 1),
            _gate("cortex_salience_trains", "Salience Train Cycles",
                  salience.get("train_count", 0), 5),
            _gate("cortex_salience_blend", "Salience Model Blend",
                  salience.get("model_blend", 0), 0.6, fmt="float"),
        ],
    })

    # ── 5. Autonomy Pipeline ─────────────────────────────────────
    auto = ev("autonomy")
    auto_level = auto.get("autonomy_level", 0)
    auto_wins = auto.get("total_wins", 0)
    auto_win_rate = auto.get("overall_win_rate", 0)
    auto_completed = auto.get("completed_total", 0)
    categories.append({
        "id": "autonomy",
        "label": "Autonomy Pipeline",
        "gates": [
            _gate("auto_level", "Current Level (0-3)", auto_level, 2,
                  detail=f"L{auto_level}"),
            _gate("auto_completed", "Research Episodes Completed",
                  auto_completed, 20),
            _gate("auto_wins_l2", "Positive Deltas (L2 gate: 10)",
                  auto_wins, 10),
            _gate("auto_wins_l3", "Positive Deltas (L3 gate: 25)",
                  auto_wins, 25),
            _gate("auto_win_rate", "Win Rate (L2: 40%)",
                  auto_win_rate, 0.40, fmt="pct"),
        ],
    })

    # ── 6. Hemisphere / Distillation ─────────────────────────────
    hemi = ev("hemisphere")
    total_nets = hemi.get("total_networks", 0)
    broadcast_filled = hemi.get("broadcast_slots_count", 0)
    dist = hemi.get("distillation", {})
    dist_teachers = dist.get("teachers", {}) if isinstance(dist, dict) else {}

    _SPECIALIST_TO_TEACHERS: dict[str, list[str]] = {
        "speaker_repr": ["ecapa_tdnn"],
        "face_repr": ["mobilefacenet"],
        "emotion_depth": ["wav2vec2_emotion"],
        "voice_intent": ["tool_router"],
        "speaker_diarize": ["ecapa_tdnn"],
        "perception_fusion": ["ecapa_tdnn", "mobilefacenet", "wav2vec2_emotion"],
    }

    specialist_gates = []
    specialist_mins = {
        "speaker_repr": 20, "face_repr": 20, "emotion_depth": 30,
        "voice_intent": 15, "speaker_diarize": 30, "perception_fusion": 50,
    }
    for name, min_s in specialist_mins.items():
        teacher_keys = _SPECIALIST_TO_TEACHERS.get(name, [name])
        totals = []
        for tk in teacher_keys:
            t = dist_teachers.get(tk, {})
            totals.append(t.get("total", 0) if isinstance(t, dict) else 0)
        samples = min(totals) if totals else 0
        specialist_gates.append(
            _gate(f"dist_{name}", f"{name.replace('_', ' ').title()} Samples",
                  samples, min_s)
        )

    categories.append({
        "id": "hemisphere",
        "label": "Hemisphere / Distillation",
        "gates": [
            _gate("hemi_networks", "Active Networks", total_nets, 1),
            _gate("hemi_broadcast", "Broadcast Slots Filled (of 4)",
                  broadcast_filled, 4),
            *specialist_gates,
        ],
    })

    # ── 7. Dream / Reflection Cycle ──────────────────────────────
    dream = ev("dream_artifacts") or {}
    dream_buf = dream.get("buffer", {})
    exp = ev("experience_buffer")
    exp_size = exp.get("size", 0) if isinstance(exp, dict) else 0
    categories.append({
        "id": "dream_cycle",
        "label": "Dream / Reflection Cycle",
        "gates": [
            _gate("dream_created", "Dream Artifacts Created",
                  dream_buf.get("total_created", 0), 500),
            _gate("dream_promoted", "Dream Artifacts Promoted",
                  max(dream_buf.get("total_promoted", 0), dream_buf.get("by_state", {}).get("promoted", 0)), 100),
            _gate("experience_200", "Experience Buffer (200)",
                  exp_size, 200),
            _gate("experience_500", "Experience Buffer (500)",
                  exp_size, 500),
        ],
    })

    # ── 8. Epistemic Stack ───────────────────────────────────────
    truth_cal = ms("truth_calibration")
    bg = ms("belief_graph")
    contradiction = ev("contradiction") or ms("contradiction")
    soul_i = ev("soul_integrity") or ms("soul_integrity")

    truth_maturity = truth_cal.get("maturity", 0)
    bg_edges = bg.get("total_edges", bg.get("edge_count", 0))
    total_beliefs = contradiction.get("total_beliefs", 0)
    active_beliefs = contradiction.get("active_beliefs", 0)
    soul_composite = soul_i.get("current_index", soul_i.get("composite", 0))

    categories.append({
        "id": "epistemic",
        "label": "Epistemic Stack",
        "gates": [
            _gate("truth_maturity", "Truth Calibration Maturity",
                  truth_maturity, 0.65, fmt="float"),
            _gate("belief_edges", "Belief Graph Edges",
                  bg_edges, 300),
            _gate("contradiction_beliefs", "Beliefs Extracted",
                  total_beliefs, 60),
            _gate("contradiction_active", "Active Beliefs",
                  active_beliefs, 30),
            _gate("soul_integrity", "Soul Integrity Index",
                  soul_composite, 0.87, fmt="float"),
        ],
    })

    # ── Summary ──────────────────────────────────────────────────
    total_gates = sum(len(c["gates"]) for c in categories)
    active_gates = sum(
        1 for c in categories for g in c["gates"] if g["status"] == "active"
    )
    ever_active_gates = _apply_maturity_highwater(categories)

    return {
        "total_gates": total_gates,
        "active_gates": active_gates,
        "ever_active_gates": ever_active_gates,
        "categories": categories,
    }


def _build_scoreboard(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the 7-category scoreboard (placeholder in Phase A)."""
    categories = [
        "epistemic_integrity",
        "memory_integrity",
        "self_report_honesty",
        "safety_immunity",
        "capability",
        "autonomy_long_horizon",
        "human_value",
    ]
    latest: dict[str, Any] = {}
    for s in scores:
        cat = s.get("category", "")
        if cat in categories:
            latest[cat] = s

    bars = []
    for cat in categories:
        entry = latest.get(cat)
        bars.append({
            "category": cat,
            "score": entry.get("score") if entry else None,
            "sample_size": entry.get("sample_size", 0) if entry else 0,
        })

    return {
        "bars": bars,
        "composite": None,
        "composite_enabled": COMPOSITE_ENABLED,
        "scoring_version": SCORING_VERSION,
        "badge": "experimental",
    }
