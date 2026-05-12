"""Shared helpers for extracting rich evidence from verify-phase history.

Used by all Register executors (procedural, perceptual, control) to carry
forward structured 5 Ws evidence into the final SkillEvidence object.
"""

from __future__ import annotations

import os
import time
from typing import Any


def find_latest_verify_evidence(job: Any) -> dict[str, Any] | None:
    """Return the most recent passing verify-phase evidence from job history."""
    best: dict[str, Any] | None = None
    best_ts: float = 0.0
    for evd in job.evidence.get("history", []):
        if evd.get("result") == "pass" or any(
            t.get("passed") for t in evd.get("tests", [])
        ):
            ts = evd.get("ts", "") or ""
            ts_f = _iso_to_float(ts) if ts else 0.0
            if ts_f >= best_ts:
                best_ts = ts_f
                best = evd
    return best


def collect_verify_details(job: Any) -> str:
    """Build a human-readable summary from verification evidence."""
    latest = find_latest_verify_evidence(job)
    if not latest:
        return f"Skill {job.skill_id} passed all required evidence checks."

    parts: list[str] = []
    for t in latest.get("tests", []):
        name = t.get("name", "unknown")
        detail = t.get("details", "")
        status = "PASS" if t.get("passed") else "FAIL"
        parts.append(f"{name}: {status}" + (f" ({detail})" if detail else ""))
    return "; ".join(parts) if parts else f"Verified via {latest.get('evidence_id', 'unknown')}"


def collect_artifact_refs(job: Any) -> list[dict[str, Any]]:
    """Collect artifact references from the job for evidence traceability."""
    refs: list[dict[str, Any]] = []
    for art in job.artifacts:
        art_type = art.get("type", "")
        path = art.get("path", "")
        ref: dict[str, Any] = {"type": art_type}
        if path:
            ref["path"] = path
            ref["exists"] = os.path.exists(path)
        if art.get("id"):
            ref["id"] = art["id"]
        refs.append(ref)
    return refs


def capture_environment(ctx: dict[str, Any]) -> dict[str, Any]:
    """Capture the verification environment from the executor context."""
    env: dict[str, Any] = {}
    if ctx.get("tts_engine"):
        tts = ctx["tts_engine"]
        env["tts_engine"] = type(tts).__name__
        if hasattr(tts, "device"):
            env["tts_device"] = str(tts.device)
    if ctx.get("hardware_tier"):
        env["hardware_tier"] = ctx["hardware_tier"]
    if ctx.get("hemisphere_orchestrator"):
        env["hemisphere_available"] = True
    if ctx.get("distillation_stats"):
        stats = ctx["distillation_stats"]
        env["distillation_teachers"] = list(stats.get("teachers", {}).keys())
    return env


def build_acceptance_criteria(job: Any) -> dict[str, Any]:
    """Build acceptance criteria dict from required evidence and test names."""
    criteria: dict[str, Any] = {}
    for req in job.evidence.get("required", []):
        criteria[req] = {"threshold": True, "comparison": "=="}
    return criteria


def build_measured_values(job: Any) -> dict[str, Any]:
    """Build measured values dict from the latest verify evidence."""
    latest = find_latest_verify_evidence(job)
    if not latest:
        return {}
    values: dict[str, Any] = {}
    for t in latest.get("tests", []):
        name = t.get("name", "unknown")
        values[name] = {
            "value": t.get("passed", False),
            "details": t.get("details", ""),
        }
    return values


def _iso_to_float(iso: str) -> float:
    """Best-effort ISO timestamp to epoch float."""
    try:
        import datetime as dt
        clean = iso.rstrip("Z")
        parsed = dt.datetime.fromisoformat(clean)
        return parsed.timestamp()
    except Exception:
        return 0.0
