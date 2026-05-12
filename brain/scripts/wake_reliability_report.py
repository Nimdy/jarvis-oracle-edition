#!/usr/bin/env python3
"""Analyze wake-word reliability from Jarvis brain logs.

Important: this report separates *passive always-listening windows* from
*interaction attempts* so idle observation does not unfairly tank reliability.

**Operational playbook:** see ``docs/WAKE_RELIABILITY_TUNING.md`` for the
evidence-gated tuning discipline (benign vs actionable classification,
baseline/post-change pass criteria, and the before/after evidence
requirement that every wake-related parameter change must satisfy).

Usage:
  python -m scripts.wake_reliability_report
  python -m scripts.wake_reliability_report --log-path /tmp/jarvis-brain.log --window-lines 6000
  python -m scripts.wake_reliability_report --since-last-restart
  python -m scripts.wake_reliability_report --strict
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RESTART_MARKERS = (
    "Starting Jarvis Brain",
    "SYSTEM_INIT_COMPLETE",
    "Jarvis consciousness awakened",
)
_TS_PREFIX_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_WAKE_LISTEN_RE = re.compile(
    r"Wake listen: chunks=(?P<chunks>\d+)\s+max_score=(?P<score>\d+(?:\.\d+)?)\s+"
    r"threshold=(?P<threshold>\d+(?:\.\d+)?)\s+speaking=(?P<speaking>True|False)"
)
_WAKE_TRIGGER_RE = re.compile(
    r"Wake word(?:\s+triggered|\s+\(barge-in\))?:\s+score=(?P<score>\d+(?:\.\d+)?)"
)
_DISPATCH_RE = re.compile(
    r"Dispatching\s+(?P<duration>\d+(?:\.\d+)?)s speech \(conv=(?P<conv>[a-zA-Z0-9]+)\)"
)
_ROUTE_RE = re.compile(
    r"route_complete=\d+ms\s+route=(?P<route>[A-Z_]+).*?\(conv=(?P<conv>[a-zA-Z0-9]+)\)"
)
_EMPTY_STT_RE = re.compile(r"STT returned empty \(conv=(?P<conv>[a-zA-Z0-9]+)\)")
_EMPTY_TRANSCRIPT_RE = re.compile(r"STT:\s+\d+(?:\.\d+)?s audio -> ''")
_ECHO_DISCARD_RE = re.compile(r"Echo detected .*discarding:", re.IGNORECASE)


def _parse_timestamp(line: str) -> float | None:
    m = _TS_PREFIX_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f").timestamp()
    except Exception:
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    p = max(0.0, min(100.0, p))
    vals = sorted(values)
    rank = (p / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    w = rank - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def _histogram(values: list[float], *, bins: int = 10) -> list[dict[str, Any]]:
    if bins <= 0:
        bins = 10
    counts = [0] * bins
    for v in values:
        clamped = max(0.0, min(0.999999, float(v)))
        idx = int(clamped * bins)
        counts[idx] += 1
    out = []
    for i, c in enumerate(counts):
        lo = i / bins
        hi = (i + 1) / bins
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(c)})
    return out


def _find_latest_restart_index(lines: list[str]) -> int:
    idx = 0
    for i, line in enumerate(lines):
        if any(marker in line for marker in _RESTART_MARKERS):
            idx = i
    return idx


def _prepare_scope_lines(
    lines_all: list[str],
    *,
    window_lines: int,
    since_last_restart: bool,
) -> tuple[list[str], dict[str, Any]]:
    restart_idx = _find_latest_restart_index(lines_all)
    base = lines_all[restart_idx:] if since_last_restart else lines_all
    scoped = base[-window_lines:] if window_lines > 0 else base
    return scoped, {
        "since_last_restart": bool(since_last_restart),
        "restart_index": int(restart_idx),
        "total_log_lines": int(len(lines_all)),
        "base_lines": int(len(base)),
        "lines_analyzed": int(len(scoped)),
        "window_lines": int(window_lines),
    }


def _parse_wake_samples(lines: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        m = _WAKE_LISTEN_RE.search(line)
        if not m:
            continue
        score = float(m.group("score"))
        threshold = float(m.group("threshold"))
        speaking = m.group("speaking") == "True"
        samples.append(
            {
                "line_idx": i,
                "ts": _parse_timestamp(line),
                "line": line,
                "score": score,
                "threshold": threshold,
                "speaking": speaking,
                "hit": score >= threshold,
                "near_miss": (score < threshold) and (score >= threshold * 0.7),
            }
        )
    return samples


def _parse_trigger_events(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        m = _WAKE_TRIGGER_RE.search(line)
        if not m:
            continue
        out.append(
            {
                "line_idx": i,
                "ts": _parse_timestamp(line),
                "line": line,
                "score": float(m.group("score")),
            }
        )
    return out


def _parse_dispatch_events(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        m = _DISPATCH_RE.search(line)
        if not m:
            continue
        out.append(
            {
                "line_idx": i,
                "ts": _parse_timestamp(line),
                "line": line,
                "duration_s": float(m.group("duration")),
                "conv": str(m.group("conv")),
            }
        )
    return out


def _parse_route_events(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        m = _ROUTE_RE.search(line)
        if not m:
            continue
        out.append(
            {
                "line_idx": i,
                "ts": _parse_timestamp(line),
                "line": line,
                "route": str(m.group("route")),
                "conv": str(m.group("conv")),
            }
        )
    return out


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "count": 0,
            "hit_rate": 0.0,
            "near_miss_rate": 0.0,
            "avg_score": 0.0,
            "p50_score": 0.0,
            "p90_score": 0.0,
            "p95_score": 0.0,
            "max_score": 0.0,
            "avg_threshold": 0.0,
        }
    scores = [float(s["score"]) for s in samples]
    thresholds = [float(s["threshold"]) for s in samples]
    hits = sum(1 for s in samples if s["hit"])
    near = sum(1 for s in samples if s["near_miss"])
    n = len(samples)
    return {
        "count": n,
        "hit_rate": round(hits / n, 4),
        "near_miss_rate": round(near / n, 4),
        "avg_score": round(statistics.fmean(scores), 4),
        "p50_score": round(_percentile(scores, 50), 4),
        "p90_score": round(_percentile(scores, 90), 4),
        "p95_score": round(_percentile(scores, 95), 4),
        "max_score": round(max(scores), 4),
        "avg_threshold": round(statistics.fmean(thresholds), 4),
    }


def _partition_active_vs_passive(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    passive: list[dict[str, Any]] = []
    for s in samples:
        threshold = float(s["threshold"])
        passive_cutoff = max(0.05, threshold * 0.25)
        is_passive = (not s["speaking"]) and (float(s["score"]) < passive_cutoff)
        if is_passive:
            passive.append(s)
        else:
            active.append(s)
    return active, passive


def _classify_nonroute_outcome(
    lines: list[str],
    dispatches: list[dict[str, Any]],
    dispatch_index: int,
    matched_dispatch: dict[str, Any] | None,
) -> tuple[str, str]:
    """Classify why an attempt has no route event.

    Outcomes:
      - no_dispatch: trigger never reached dispatch stage
      - benign_empty_stt: dispatch happened but STT produced empty transcript
      - benign_echo_discard: dispatch happened but echo guard discarded transcript
      - downstream_route_missing: dispatch happened without known benign explanation
    """
    if matched_dispatch is None or dispatch_index < 0:
        return "no_dispatch", "dispatch_not_found"

    dispatch_line_idx = int(matched_dispatch.get("line_idx", 0))
    dispatch_conv = str(matched_dispatch.get("conv", "") or "")
    next_dispatch_idx: int | None = None
    if dispatch_index + 1 < len(dispatches):
        next_dispatch_idx = int(dispatches[dispatch_index + 1].get("line_idx", len(lines) - 1))
    # Keep the attribution window tight to avoid bleeding into a later attempt.
    end_candidate = dispatch_line_idx + 280
    if next_dispatch_idx is not None:
        end_candidate = min(end_candidate, next_dispatch_idx - 1)
    end_idx = min(len(lines) - 1, end_candidate)
    if end_idx < dispatch_line_idx:
        end_idx = min(len(lines) - 1, dispatch_line_idx + 120)
    window = lines[dispatch_line_idx : end_idx + 1]

    if dispatch_conv:
        for line in window:
            m = _EMPTY_STT_RE.search(line)
            if m and str(m.group("conv")) == dispatch_conv:
                return "benign_empty_stt", "empty_stt"

    if any(_EMPTY_TRANSCRIPT_RE.search(line) for line in window):
        return "benign_empty_stt", "empty_transcript"

    if any(_ECHO_DISCARD_RE.search(line) for line in window):
        return "benign_echo_discard", "echo_discarded"

    return "downstream_route_missing", "no_route_event"


def _match_trigger_pipeline(
    lines: list[str],
    triggers: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    *,
    dispatch_window_s: float = 20.0,
    route_window_s: float = 30.0,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    used_dispatch: set[int] = set()
    used_routes: set[int] = set()

    for trig in triggers:
        trig_ts = trig.get("ts")
        trig_idx = int(trig.get("line_idx", 0))
        matched_dispatch_idx = -1
        matched_dispatch: dict[str, Any] | None = None

        for di, d in enumerate(dispatches):
            if di in used_dispatch:
                continue
            d_ts = d.get("ts")
            if trig_ts is not None and d_ts is not None:
                if d_ts < trig_ts or d_ts > (trig_ts + dispatch_window_s):
                    continue
            else:
                d_idx = int(d.get("line_idx", 0))
                if d_idx < trig_idx or d_idx > trig_idx + 220:
                    continue
            matched_dispatch_idx = di
            matched_dispatch = d
            break

        if matched_dispatch_idx >= 0:
            used_dispatch.add(matched_dispatch_idx)

        matched_route_idx = -1
        matched_route: dict[str, Any] | None = None
        if matched_dispatch is not None:
            d_ts = matched_dispatch.get("ts")
            d_idx = int(matched_dispatch.get("line_idx", 0))
            d_conv = str(matched_dispatch.get("conv", "") or "")
            next_dispatch_line_idx: int | None = None
            if matched_dispatch_idx + 1 < len(dispatches):
                next_dispatch_line_idx = int(
                    dispatches[matched_dispatch_idx + 1].get("line_idx", 0) or 0
                )

            # Primary: same conversation id.
            for ri, r in enumerate(routes):
                if ri in used_routes:
                    continue
                if d_conv and str(r.get("conv", "") or "") != d_conv:
                    continue
                r_idx = int(r.get("line_idx", 0))
                if next_dispatch_line_idx is not None and r_idx >= next_dispatch_line_idx:
                    continue
                r_ts = r.get("ts")
                if d_ts is not None and r_ts is not None:
                    if r_ts < d_ts or r_ts > (d_ts + route_window_s):
                        continue
                else:
                    if r_idx < d_idx or r_idx > d_idx + 260:
                        continue
                matched_route_idx = ri
                matched_route = r
                break

            # Fallback: any route soon after dispatch.
            if matched_route is None:
                for ri, r in enumerate(routes):
                    if ri in used_routes:
                        continue
                    r_idx = int(r.get("line_idx", 0))
                    if next_dispatch_line_idx is not None and r_idx >= next_dispatch_line_idx:
                        continue
                    r_ts = r.get("ts")
                    if d_ts is not None and r_ts is not None:
                        if r_ts < d_ts or r_ts > (d_ts + route_window_s):
                            continue
                    else:
                        if r_idx < d_idx or r_idx > d_idx + 260:
                            continue
                    matched_route_idx = ri
                    matched_route = r
                    break

        if matched_route_idx >= 0:
            used_routes.add(matched_route_idx)

        attempt_outcome = "routed"
        outcome_reason = "route_matched"
        if matched_route is None:
            attempt_outcome, outcome_reason = _classify_nonroute_outcome(
                lines=lines,
                dispatches=dispatches,
                dispatch_index=matched_dispatch_idx,
                matched_dispatch=matched_dispatch,
            )

        attempts.append(
            {
                "trigger_line": trig.get("line", ""),
                "trigger_score": float(trig.get("score", 0.0) or 0.0),
                "dispatch_matched": matched_dispatch is not None,
                "route_matched": matched_route is not None,
                "dispatch_line": matched_dispatch.get("line", "") if matched_dispatch else "",
                "route_line": matched_route.get("line", "") if matched_route else "",
                "route": matched_route.get("route", "") if matched_route else "",
                "dispatch_conv": matched_dispatch.get("conv", "") if matched_dispatch else "",
                "attempt_outcome": attempt_outcome,
                "outcome_reason": outcome_reason,
            }
        )
    return attempts


def _summarize_attempts(triggers: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    trigger_count = len(triggers)
    dispatch_count = sum(1 for a in attempts if a.get("dispatch_matched"))
    route_count = sum(1 for a in attempts if a.get("route_matched"))
    outcome_counts = Counter(str(a.get("attempt_outcome", "unknown")) for a in attempts)
    benign_nonroute_count = (
        int(outcome_counts.get("benign_empty_stt", 0))
        + int(outcome_counts.get("benign_echo_discard", 0))
    )
    actionable_route_miss_count = int(outcome_counts.get("downstream_route_missing", 0))
    handled_count = route_count + benign_nonroute_count
    scores = [float(t.get("score", 0.0) or 0.0) for t in triggers]
    return {
        "trigger_count": trigger_count,
        "dispatch_matched": dispatch_count,
        "route_matched": route_count,
        "benign_nonroute_count": benign_nonroute_count,
        "actionable_route_miss_count": actionable_route_miss_count,
        "handled_count": handled_count,
        "trigger_to_dispatch_rate": round(dispatch_count / trigger_count, 4) if trigger_count else 0.0,
        "trigger_to_route_rate": round(route_count / trigger_count, 4) if trigger_count else 0.0,
        "trigger_to_user_handled_rate": round(handled_count / trigger_count, 4) if trigger_count else 0.0,
        "trigger_p50_score": round(_percentile(scores, 50), 4) if scores else 0.0,
        "trigger_p90_score": round(_percentile(scores, 90), 4) if scores else 0.0,
        "trigger_avg_score": round(statistics.fmean(scores), 4) if scores else 0.0,
        "attempt_outcome_counts": dict(sorted(outcome_counts.items())),
        "recent_attempts": attempts[-8:],
    }


def _assess(
    *,
    overall: dict[str, Any],
    active: dict[str, Any],
    attempt_summary: dict[str, Any],
    min_samples: int,
    min_attempts: int,
) -> dict[str, Any]:
    total_count = int(overall.get("count", 0) or 0)
    active_count = int(active.get("count", 0) or 0)
    trigger_count = int(attempt_summary.get("trigger_count", 0) or 0)
    route_rate = float(attempt_summary.get("trigger_to_route_rate", 0.0) or 0.0)
    dispatch_rate = float(attempt_summary.get("trigger_to_dispatch_rate", 0.0) or 0.0)
    handled_rate = float(attempt_summary.get("trigger_to_user_handled_rate", 0.0) or 0.0)
    benign_nonroute_count = int(attempt_summary.get("benign_nonroute_count", 0) or 0)
    actionable_route_miss_count = int(attempt_summary.get("actionable_route_miss_count", 0) or 0)

    if total_count < min_samples and trigger_count < min_attempts:
        return {
            "status": "insufficient_data",
            "message": (
                f"Need >= {min_samples} wake samples or >= {min_attempts} "
                "explicit wake attempts before tuning decisions."
            ),
        }
    if active_count == 0 and trigger_count < min_attempts:
        return {
            "status": "insufficient_interaction_samples",
            "message": (
                "Only passive always-listening windows were observed; "
                "capture explicit wake attempts before tuning."
            ),
        }

    if trigger_count >= min_attempts:
        if handled_rate >= 0.8 and dispatch_rate >= 0.8 and actionable_route_miss_count == 0:
            if route_rate < 0.8 and benign_nonroute_count > 0:
                return {
                    "status": "healthy_interaction_benign_nonroutes",
                    "message": (
                        "Wake attempts are healthy; non-routed attempts were benign "
                        "(echo/empty-STT), not downstream route failures."
                    ),
                }
            if active_count < max(10, min_samples // 8):
                return {
                    "status": "healthy_interaction_low_activity",
                    "message": (
                        "Explicit wake attempts succeed; passive always-listening windows dominate."
                    ),
                }
            return {
                "status": "healthy_interaction",
                "message": "Explicit wake attempts and downstream routing are healthy.",
            }
        if route_rate >= 0.8 and actionable_route_miss_count == 0:
            if active_count < max(10, min_samples // 8):
                return {
                    "status": "healthy_interaction_low_activity",
                    "message": (
                        "Explicit wake attempts succeed; passive always-listening windows dominate."
                    ),
                }
            return {
                "status": "healthy_interaction",
                "message": "Explicit wake attempts and downstream routing are healthy.",
            }
        if dispatch_rate >= 0.8 and route_rate >= 0.5:
            return {
                "status": "mixed_interaction",
                "message": "Wake attempts trigger reliably, but some routes are missing downstream.",
            }

    ref = active if active_count > 0 else overall
    ref_count = int(ref.get("count", 0) or 0)
    if ref_count < max(20, min_samples // 4):
        return {
            "status": "insufficient_interaction_samples",
            "message": (
                "Too few active interaction windows to judge wake quality; "
                "passive listening windows are excluded from scoring."
            ),
        }

    hit_rate = float(ref.get("hit_rate", 0.0) or 0.0)
    near_rate = float(ref.get("near_miss_rate", 0.0) or 0.0)
    p95 = float(ref.get("p95_score", 0.0) or 0.0)
    threshold = float(ref.get("avg_threshold", 0.0) or 0.0)
    if hit_rate < 0.002 and near_rate < 0.01 and (threshold <= 0.0 or p95 < 0.5 * threshold):
        return {
            "status": "very_low_signal",
            "message": (
                "Active wake windows are far below threshold; tune only with controlled A/B evidence."
            ),
        }
    if hit_rate < 0.01:
        return {
            "status": "low_signal",
            "message": "Active wake hit rate is low; gather more attempt-focused data first.",
        }
    return {
        "status": "healthy_or_active",
        "message": "Wake signal in active windows is healthy for cautious evidence-based tuning.",
    }


def build_wake_report(
    log_text: str,
    *,
    window_lines: int,
    min_samples: int,
    since_last_restart: bool = False,
    min_attempts: int = 2,
) -> dict[str, Any]:
    lines_all = log_text.splitlines()
    lines, scope = _prepare_scope_lines(
        lines_all,
        window_lines=window_lines,
        since_last_restart=since_last_restart,
    )
    samples = _parse_wake_samples(lines)
    triggers = _parse_trigger_events(lines)
    dispatches = _parse_dispatch_events(lines)
    routes = _parse_route_events(lines)
    attempts = _match_trigger_pipeline(lines, triggers, dispatches, routes)
    attempt_summary = _summarize_attempts(triggers, attempts)

    non_speaking = [s for s in samples if not s["speaking"]]
    speaking = [s for s in samples if s["speaking"]]
    active_samples, passive_samples = _partition_active_vs_passive(samples)

    overall = _summarize_samples(samples)
    active = _summarize_samples(active_samples)
    passive = _summarize_samples(passive_samples)
    non_speaking_summary = _summarize_samples(non_speaking)
    speaking_summary = _summarize_samples(speaking)
    assessment = _assess(
        overall=overall,
        active=active,
        attempt_summary=attempt_summary,
        min_samples=min_samples,
        min_attempts=min_attempts,
    )

    total = len(samples)
    active_count = len(active_samples)
    passive_count = len(passive_samples)
    passive_filter = {
        "active_windows": active_count,
        "passive_windows": passive_count,
        "active_rate": round(active_count / total, 4) if total else 0.0,
        "passive_rate": round(passive_count / total, 4) if total else 0.0,
        "rule": "passive if non-speaking and score < max(0.05, threshold*0.25)",
    }

    scores = [float(s["score"]) for s in samples]
    active_scores = [float(s["score"]) for s in active_samples]
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "scope": scope,
        "sample_count": int(len(samples)),
        "trigger_count": int(len(triggers)),
        "overall": overall,
        "active": active,
        "passive": passive,
        "passive_filter": passive_filter,
        "non_speaking": non_speaking_summary,
        "speaking": speaking_summary,
        "trigger_pipeline": attempt_summary,
        "score_histogram": _histogram(scores, bins=10),
        "active_score_histogram": _histogram(active_scores, bins=10),
        "assessment": assessment,
    }


def render_wake_markdown(report: dict[str, Any]) -> str:
    scope = report.get("scope", {})
    overall = report.get("overall", {})
    active = report.get("active", {})
    passive_filter = report.get("passive_filter", {})
    pipeline = report.get("trigger_pipeline", {})
    assessment = report.get("assessment", {})

    lines = [
        "# Wake Reliability Report",
        "",
        f"- Generated: {report.get('generated_at', '--')}",
        f"- Scope: lines={scope.get('lines_analyzed', '--')} (since_restart={scope.get('since_last_restart', False)})",
        f"- Wake samples: {report.get('sample_count', 0)}",
        f"- Triggered wake events: {report.get('trigger_count', 0)}",
        f"- Assessment: **{assessment.get('status', 'unknown')}**",
        f"- Note: {assessment.get('message', '--')}",
        "",
        "## Overall (Raw Always-Listening)",
        "",
        f"- Hit rate: {overall.get('hit_rate', 0.0):.4f}",
        f"- Near-miss rate: {overall.get('near_miss_rate', 0.0):.4f}",
        f"- Avg score / threshold: {overall.get('avg_score', 0.0):.4f} / {overall.get('avg_threshold', 0.0):.4f}",
        f"- P95 / Max score: {overall.get('p95_score', 0.0):.4f} / {overall.get('max_score', 0.0):.4f}",
        "",
        "## Passive Listening Filter",
        "",
        f"- Active windows: {passive_filter.get('active_windows', 0)} ({passive_filter.get('active_rate', 0.0):.4f})",
        f"- Passive windows: {passive_filter.get('passive_windows', 0)} ({passive_filter.get('passive_rate', 0.0):.4f})",
        f"- Active hit rate: {active.get('hit_rate', 0.0):.4f}",
        f"- Active near-miss rate: {active.get('near_miss_rate', 0.0):.4f}",
        f"- Rule: {passive_filter.get('rule', '--')}",
        "",
        "## Trigger Pipeline (Attempt-Focused)",
        "",
        f"- Trigger -> dispatch rate: {pipeline.get('trigger_to_dispatch_rate', 0.0):.4f}",
        f"- Trigger -> route rate: {pipeline.get('trigger_to_route_rate', 0.0):.4f}",
        f"- Trigger -> user-handled rate (route + benign): {pipeline.get('trigger_to_user_handled_rate', 0.0):.4f}",
        f"- Benign non-routes (echo/empty-STT): {int(pipeline.get('benign_nonroute_count', 0) or 0)}",
        f"- Actionable route misses: {int(pipeline.get('actionable_route_miss_count', 0) or 0)}",
        f"- Trigger score p50 / p90: {pipeline.get('trigger_p50_score', 0.0):.4f} / {pipeline.get('trigger_p90_score', 0.0):.4f}",
        "",
        "## Score Histogram (Raw)",
        "",
        "| Bin | Count |",
        "|---|---:|",
    ]
    for row in report.get("score_histogram", []) or []:
        lines.append(f"| {row.get('bin', '--')} | {int(row.get('count', 0) or 0)} |")
    lines.append("")
    return "\n".join(lines)


def _write_artifacts(report: dict[str, Any], output_dir: str) -> tuple[str, str]:
    out_dir = Path(os.path.expanduser(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"wake-reliability-{ts}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, sort_keys=True, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_wake_markdown(report))
    return str(json_path), str(md_path)


def _print_summary(report: dict[str, Any]) -> None:
    overall = report.get("overall", {})
    active = report.get("active", {})
    pipeline = report.get("trigger_pipeline", {})
    passive_filter = report.get("passive_filter", {})
    assessment = report.get("assessment", {})
    scope = report.get("scope", {})

    status = str(assessment.get("status", "") or "unknown")
    label = {
        "healthy_or_active": "PASS",
        "healthy_interaction": "PASS",
        "healthy_interaction_low_activity": "PASS",
        "healthy_interaction_benign_nonroutes": "PASS",
        "mixed_interaction": "WARN",
        "low_signal": "WARN",
        "very_low_signal": "WARN",
        "insufficient_data": "INFO",
        "insufficient_interaction_samples": "INFO",
    }.get(status, "INFO")
    print(f"[{label}] Wake reliability: {status}")
    print(
        "  Scope: lines="
        f"{scope.get('lines_analyzed', 0)} (since_restart={scope.get('since_last_restart', False)})"
    )
    print(f"  Wake samples: {report.get('sample_count', 0)}")
    print(f"  Triggered wake events: {report.get('trigger_count', 0)}")
    print(
        "  Passive filter: "
        f"active={passive_filter.get('active_windows', 0)} ({passive_filter.get('active_rate', 0.0):.4f}), "
        f"passive={passive_filter.get('passive_windows', 0)} ({passive_filter.get('passive_rate', 0.0):.4f})"
    )
    print(
        "  Trigger pipeline: "
        f"dispatch_rate={pipeline.get('trigger_to_dispatch_rate', 0.0):.4f}, "
        f"route_rate={pipeline.get('trigger_to_route_rate', 0.0):.4f}, "
        f"user_handled_rate={pipeline.get('trigger_to_user_handled_rate', 0.0):.4f}"
    )
    print(
        "  Trigger outcomes: "
        f"benign_nonroutes={int(pipeline.get('benign_nonroute_count', 0) or 0)}, "
        f"actionable_route_misses={int(pipeline.get('actionable_route_miss_count', 0) or 0)}"
    )
    print(
        "  Scores (raw/active): "
        f"p95={overall.get('p95_score', 0.0):.4f}/{active.get('p95_score', 0.0):.4f}, "
        f"avg_threshold={overall.get('avg_threshold', 0.0):.4f}"
    )
    print(f"  Note: {assessment.get('message', '--')}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate wake-word reliability report from logs.")
    p.add_argument("--log-path", default="/tmp/jarvis-brain.log", help="Path to Jarvis brain log file")
    p.add_argument("--window-lines", type=int, default=6000, help="How many tail lines to analyze")
    p.add_argument("--since-last-restart", action="store_true", help="Analyze only log lines after latest restart marker")
    p.add_argument("--min-samples", type=int, default=80, help="Minimum wake sample windows before strict confidence")
    p.add_argument("--min-attempts", type=int, default=2, help="Minimum explicit wake attempts for interaction assessment")
    p.add_argument("--output-dir", default="~/.jarvis/eval/wake_reports", help="Output directory for artifacts")
    p.add_argument("--no-write", action="store_true", help="Do not write JSON/Markdown artifacts")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when signal is too weak/insufficient for tuning decisions",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    log_path = Path(os.path.expanduser(args.log_path))
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return 2
    try:
        text = log_path.read_text(errors="replace")
    except Exception as exc:
        print(f"[ERROR] Failed to read log file: {exc}")
        return 2

    report = build_wake_report(
        text,
        window_lines=args.window_lines,
        min_samples=args.min_samples,
        since_last_restart=args.since_last_restart,
        min_attempts=args.min_attempts,
    )
    report["log_path"] = str(log_path)
    _print_summary(report)

    if not args.no_write:
        try:
            json_path, md_path = _write_artifacts(report, args.output_dir)
            print(f"[INFO] JSON: {json_path}")
            print(f"[INFO] MD  : {md_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write artifacts: {exc}")
            return 1

    if args.strict:
        st = str((report.get("assessment") or {}).get("status", "") or "")
        if st in {"insufficient_data", "insufficient_interaction_samples", "very_low_signal"}:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
