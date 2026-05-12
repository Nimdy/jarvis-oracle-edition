"""Sparse Oracle scorecards for executive trend tracking.

These rollups are intentionally low-frequency and high-signal. They exist to
answer one question truthfully: is the Oracle improving, regressing, or simply
remaining stable over meaningful windows without gaming the system.
"""

from __future__ import annotations

from typing import Any

WINDOWS_S: dict[str, int] = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
}


def _resolve(data: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = data if isinstance(data, dict) else {}
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _num(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) else None


def _count_active_drift_alerts(truth: dict[str, Any]) -> int:
    alerts = truth.get("active_drift_alerts")
    if not isinstance(alerts, list):
        alerts = truth.get("drift_alerts", [])
    return len(alerts) if isinstance(alerts, list) else 0


def build_oracle_scorecard(
    latest_by_source: dict[str, dict[str, Any]],
    pvl_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact truth-first scorecard from the latest eval state."""
    truth = latest_by_source.get("truth_calibration", {})
    belief = latest_by_source.get("belief_graph", {})
    contradiction = latest_by_source.get("contradiction", {})
    quarantine = latest_by_source.get("quarantine", {})
    soul = latest_by_source.get("soul_integrity", {})
    memory = latest_by_source.get("memory", {})
    autonomy = latest_by_source.get("autonomy", {})
    identity = latest_by_source.get("identity", {})
    hemisphere = latest_by_source.get("hemisphere", {})
    matrix = latest_by_source.get("matrix", {})
    world_model = latest_by_source.get("world_model_promotion", {})
    mutation = latest_by_source.get("mutation_governor", {})
    dream = latest_by_source.get("dream_artifacts", {})
    library = latest_by_source.get("library", {})
    language = latest_by_source.get("language", {})
    policy = latest_by_source.get("policy_telemetry", {})
    pvl = pvl_result or {}

    identity_name = (
        identity.get("resolved_identity")
        or identity.get("identity")
        or identity.get("active_identity")
        or identity.get("person_name")
        or ""
    )
    identity_conf = max(
        [
            v for v in [
                _num(identity.get("confidence")),
                _num(identity.get("resolved_confidence")),
                _num(identity.get("combined_confidence")),
                _num(identity.get("voice_confidence")),
                _num(identity.get("face_confidence")),
            ] if v is not None
        ],
        default=None,
    )

    truth_score = _num(truth.get("truth_score"))
    truth_maturity = _num(truth.get("maturity"))
    soul_index = _num(soul.get("current_index"))
    contradiction_debt = _num(
        contradiction.get("contradiction_debt", contradiction.get("debt"))
    )
    quarantine_pressure = _num(quarantine.get("composite"))
    belief_edges = _resolve(belief, "integrity.total_edges", belief.get("total_edges", 0))
    belief_health = _num(_resolve(belief, "integrity.health_score"))
    belief_orphan_rate = _num(_resolve(belief, "integrity.orphan_rate"))

    notes: list[str] = []
    if truth_score is not None and truth_score < 0.6:
        notes.append("truth calibration remains below the preferred confidence band")
    if contradiction_debt is not None and contradiction_debt > 0.0:
        notes.append("contradiction debt is non-zero")
    if quarantine_pressure is not None and quarantine_pressure > 0.3:
        notes.append("quarantine pressure is elevated")
    if pvl.get("coverage_pct", 0.0) < 80.0:
        notes.append("PVL coverage is below the target demonstration band")
    if _num(language.get("quality_fail_closed_rate")) is not None and _num(language.get("quality_fail_closed_rate")) > 0.5:
        notes.append("language substrate is failing closed too frequently")

    return {
        "safety": {
            "mutations_applied": mutation.get("mutation_count", 0),
            "mutation_rollbacks": mutation.get("rollback_count", 0),
            "mutation_rejections": mutation.get("total_rejections", 0),
            "mutations_this_hour": mutation.get("mutations_this_hour", 0),
            "active_monitor": bool(mutation.get("active_monitor")),
        },
        "integrity": {
            "soul_integrity": soul_index,
            "soul_weakest_dimension": soul.get("weakest_dimension", ""),
            "soul_weakest_score": _num(soul.get("weakest_score")),
            "truth_score": truth_score,
            "truth_maturity": truth_maturity,
            "truth_drift_alerts": _count_active_drift_alerts(truth),
            "belief_health": belief_health,
            "belief_edges": int(belief_edges or 0),
            "belief_orphan_rate": belief_orphan_rate,
            "contradiction_debt": contradiction_debt,
            "quarantine_pressure": quarantine_pressure,
            "quarantine_chronic_count": int(quarantine.get("chronic_count", 0)),
        },
        "pvl": {
            "coverage_pct": _num(pvl.get("coverage_pct")),
            "passing_contracts": int(pvl.get("passing_contracts", 0)),
            "applicable_contracts": int(pvl.get("applicable_contracts", 0)),
            "failing_contracts": int(pvl.get("failing_contracts", 0)),
            "awaiting_contracts": int(pvl.get("awaiting_contracts", 0)),
        },
        "autonomy": {
            "level": int(autonomy.get("autonomy_level", 0) or 0),
            "overall_win_rate": _num(autonomy.get("overall_win_rate")),
            "total_wins": int(autonomy.get("total_wins", 0)),
            "completed_total": int(autonomy.get("completed_total", 0)),
        },
        "identity": {
            "name": identity_name,
            "confidence": identity_conf,
            "voice_confidence": _num(identity.get("voice_confidence")),
            "face_confidence": _num(identity.get("face_confidence")),
        },
        "memory": {
            "total": int(memory.get("total", 0)),
            "avg_weight": _num(memory.get("avg_weight")),
            "strong_count": int(memory.get("strong_count", 0)),
            "weak_count": int(memory.get("weak_count", 0)),
        },
        "learning": {
            "hemisphere_networks": int(hemisphere.get("total_networks", 0)),
            "broadcast_slots": int(hemisphere.get("broadcast_slots_count", 0)),
            "matrix_specialists": int(matrix.get("specialist_count", 0)),
            "policy_win_rate": _num(
                policy.get("nn_decisive_win_rate", policy.get("shadow_win_rate"))
            ),
            "world_model_level": int(world_model.get("level", 0) or 0),
        },
        "dream": {
            "buffer_size": int(_resolve(dream, "buffer.buffer_size", 0) or 0),
            "promotion_rate": _num(dream.get("promotion_rate")),
            "promoted_total": int(max(
                _resolve(dream, "buffer.total_promoted", 0) or 0,
                _resolve(dream, "buffer.by_state.promoted", 0) or 0,
            )),
        },
        "library": {
            "total": int(library.get("total", 0)),
            "studied": int(library.get("studied", 0)),
            "substantive_ratio": _num(library.get("substantive_ratio")),
        },
        "language": {
            "corpus_total_examples": int(language.get("corpus_total_examples", 0)),
            "quality_total_events": int(language.get("quality_total_events", 0)),
            "native_usage_rate": _num(language.get("quality_native_usage_rate")),
            "fail_closed_rate": _num(language.get("quality_fail_closed_rate")),
            "response_classes": int(len(language.get("quality_counts_by_class", {}) or {})),
        },
        "notes": notes,
    }


def build_scorecard_summary(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize persisted scorecards into truthful comparison windows."""
    if not scorecards:
        return {"status": "awaiting_history", "current": {}, "windows": {}}

    ordered = sorted(scorecards, key=lambda s: s.get("timestamp", 0))
    current = ordered[-1]
    current_ts = current.get("timestamp", 0)
    windows: dict[str, Any] = {}

    for label, window_s in WINDOWS_S.items():
        cutoff = current_ts - window_s
        reference = _scorecard_at_or_before(ordered, cutoff)
        if reference is None:
            windows[label] = {"available": False, "window_s": window_s}
            continue
        windows[label] = _build_window_delta(current, reference, label, window_s)

    return {
        "status": "active",
        "current": current.get("metrics", {}),
        "current_ts": current_ts,
        "windows": windows,
    }


def _scorecard_at_or_before(scorecards: list[dict[str, Any]], cutoff_ts: float) -> dict[str, Any] | None:
    for card in reversed(scorecards):
        if card.get("timestamp", 0) <= cutoff_ts:
            return card
    return None


def _build_window_delta(
    current: dict[str, Any],
    reference: dict[str, Any],
    label: str,
    window_s: int,
) -> dict[str, Any]:
    cm = current.get("metrics", {})
    rm = reference.get("metrics", {})

    cur_safety = cm.get("safety", {})
    ref_safety = rm.get("safety", {})
    cur_int = cm.get("integrity", {})
    ref_int = rm.get("integrity", {})
    cur_pvl = cm.get("pvl", {})
    ref_pvl = rm.get("pvl", {})
    cur_lang = cm.get("language", {})
    ref_lang = rm.get("language", {})

    deltas = {
        "mutations_applied": int(cur_safety.get("mutations_applied", 0)) - int(ref_safety.get("mutations_applied", 0)),
        "mutation_rollbacks": int(cur_safety.get("mutation_rollbacks", 0)) - int(ref_safety.get("mutation_rollbacks", 0)),
        "mutation_rejections": int(cur_safety.get("mutation_rejections", 0)) - int(ref_safety.get("mutation_rejections", 0)),
        "soul_integrity": _delta(cur_int.get("soul_integrity"), ref_int.get("soul_integrity")),
        "truth_score": _delta(cur_int.get("truth_score"), ref_int.get("truth_score")),
        "pvl_coverage_pct": _delta(cur_pvl.get("coverage_pct"), ref_pvl.get("coverage_pct")),
        "contradiction_debt": _delta(cur_int.get("contradiction_debt"), ref_int.get("contradiction_debt")),
        "quarantine_pressure": _delta(cur_int.get("quarantine_pressure"), ref_int.get("quarantine_pressure")),
        "language_native_usage_rate": _delta(cur_lang.get("native_usage_rate"), ref_lang.get("native_usage_rate")),
        "language_fail_closed_rate": _delta(cur_lang.get("fail_closed_rate"), ref_lang.get("fail_closed_rate")),
        "language_quality_total_events": int(cur_lang.get("quality_total_events", 0)) - int(ref_lang.get("quality_total_events", 0)),
        "language_corpus_total_examples": int(cur_lang.get("corpus_total_examples", 0)) - int(ref_lang.get("corpus_total_examples", 0)),
    }

    proof_points = [
        _mutation_proof(deltas),
        _scalar_proof("Soul", ref_int.get("soul_integrity"), cur_int.get("soul_integrity"), deltas["soul_integrity"], pct=True),
        _scalar_proof("Truth", ref_int.get("truth_score"), cur_int.get("truth_score"), deltas["truth_score"], pct=True),
        _scalar_proof("PVL", ref_pvl.get("coverage_pct"), cur_pvl.get("coverage_pct"), deltas["pvl_coverage_pct"], suffix="%"),
        _scalar_proof("Debt", ref_int.get("contradiction_debt"), cur_int.get("contradiction_debt"), deltas["contradiction_debt"]),
        _scalar_proof("Quarantine", ref_int.get("quarantine_pressure"), cur_int.get("quarantine_pressure"), deltas["quarantine_pressure"]),
        _scalar_proof("Language Native", ref_lang.get("native_usage_rate"), cur_lang.get("native_usage_rate"), deltas["language_native_usage_rate"], pct=True),
        _scalar_proof("Language Fail-Closed", ref_lang.get("fail_closed_rate"), cur_lang.get("fail_closed_rate"), deltas["language_fail_closed_rate"], pct=True),
        _count_proof("Language Events", cur_lang.get("quality_total_events"), deltas["language_quality_total_events"]),
        _count_proof("Language Corpus", cur_lang.get("corpus_total_examples"), deltas["language_corpus_total_examples"]),
    ]

    return {
        "available": True,
        "label": label,
        "window_s": window_s,
        "actual_age_s": int(max(0, current.get("timestamp", 0) - reference.get("timestamp", 0))),
        "headline": _window_headline(cm, deltas),
        "proof_points": [p for p in proof_points if p],
        "deltas": deltas,
        "language": {
            "native_usage_rate": cur_lang.get("native_usage_rate"),
            "fail_closed_rate": cur_lang.get("fail_closed_rate"),
            "quality_total_events": cur_lang.get("quality_total_events"),
            "corpus_total_examples": cur_lang.get("corpus_total_examples"),
            "native_usage_delta": deltas["language_native_usage_rate"],
            "fail_closed_delta": deltas["language_fail_closed_rate"],
        },
    }


def _delta(current: Any, previous: Any) -> float | None:
    cur = _num(current)
    prev = _num(previous)
    if cur is None or prev is None:
        return None
    return round(cur - prev, 4)


def _mutation_proof(deltas: dict[str, Any]) -> str:
    applied = deltas["mutations_applied"]
    rollbacks = deltas["mutation_rollbacks"]
    rejects = deltas["mutation_rejections"]
    return f"Mutations +{applied}, rollbacks +{rollbacks}, rejections +{rejects}"


def _count_proof(label: str, current: Any, delta: int | None) -> str | None:
    cur = int(current) if isinstance(current, (int, float)) else None
    if cur is None:
        return None
    if delta is None:
        return f"{label} {cur}"
    return f"{label} {cur} ({delta:+d})"


def _scalar_proof(
    label: str,
    reference: Any,
    current: Any,
    delta: float | None,
    *,
    pct: bool = False,
    suffix: str = "",
) -> str | None:
    cur = _num(current)
    ref = _num(reference)
    if cur is None:
        return None

    if pct:
        ref_text = f"{ref * 100:.1f}%" if ref is not None else "?"
        cur_text = f"{cur * 100:.1f}%"
        delta_text = f"{delta * 100:+.2f}%" if delta is not None else "n/a"
    elif suffix:
        ref_text = f"{ref:.1f}{suffix}" if ref is not None else "?"
        cur_text = f"{cur:.1f}{suffix}"
        delta_text = f"{delta:+.1f}{suffix}" if delta is not None else "n/a"
    else:
        ref_text = f"{ref:.3f}" if ref is not None else "?"
        cur_text = f"{cur:.3f}"
        delta_text = f"{delta:+.4f}" if delta is not None else "n/a"
    return f"{label} {ref_text} \u2192 {cur_text} ({delta_text})"


def _window_headline(metrics: dict[str, Any], deltas: dict[str, Any]) -> str:
    integrity = metrics.get("integrity", {})
    soul = _num(integrity.get("soul_integrity"))
    truth = _num(integrity.get("truth_score"))
    contradiction = _num(integrity.get("contradiction_debt"))
    quarantine = _num(integrity.get("quarantine_pressure"))

    soul_d = deltas.get("soul_integrity") or 0
    truth_d = deltas.get("truth_score") or 0

    def _direction_phrase() -> str:
        """Summarise soul + truth movement for this specific window."""
        parts: list[str] = []
        if abs(soul_d) >= 0.001:
            parts.append(f"Soul {'+' if soul_d > 0 else ''}{soul_d * 100:.1f}%")
        if abs(truth_d) >= 0.001:
            parts.append(f"Truth {'+' if truth_d > 0 else ''}{truth_d * 100:.1f}%")
        if not parts:
            return ""
        return " " + ", ".join(parts) + "."

    if contradiction is not None and contradiction > 0:
        debt_delta = deltas.get("contradiction_debt")
        if contradiction <= 0.005:
            if debt_delta is not None and debt_delta < 0:
                return "Low contradiction debt remains, but it is improving across this window."
            return "Low contradiction debt remains; improvement claims should stay cautious."
        return "Non-zero contradiction debt." + _direction_phrase()
    if quarantine is not None and quarantine > 0.3:
        return "Quarantine pressure is elevated; changes are occurring under friction." + _direction_phrase()
    if deltas["mutation_rollbacks"] and deltas["mutation_rollbacks"] > 0:
        return "Mutations are active and rollback safeguards have fired in this window." + _direction_phrase()
    if soul_d > 0 and truth_d >= 0:
        return "Integrity signals improved without evidence of epistemic drift." + _direction_phrase()
    if soul_d < 0 or truth_d < 0:
        return "Core integrity softened in this window." + _direction_phrase()
    if soul is not None and truth is not None:
        return "The Oracle appears stable across soul integrity and truth calibration."
    return "Scorecard history exists, but not all integrity signals are mature yet."
