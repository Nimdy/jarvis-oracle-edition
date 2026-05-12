"""Read-only skill audit packets for dashboard and scientific review."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_PREVIEW_LIMIT = 1200


def build_skill_audit_packet(
    skill_id: str,
    registry: Any,
    orchestrator: Any = None,
    acquisition_orchestrator: Any = None,
) -> dict[str, Any]:
    """Build a bounded, read-only audit packet for a skill.

    The packet summarizes canonical SkillRegistry and LearningJob state. It
    never advances phases, runs contracts, executes plugins, or writes files.
    """

    rec = registry.get(skill_id) if registry is not None else None
    skill = rec.to_dict() if rec is not None and hasattr(rec, "to_dict") else {}
    current_job_id = skill.get("learning_job_id") or ""
    jobs = _collect_related_jobs(skill_id, current_job_id, orchestrator)
    current_job = next((j for j in jobs if j.get("is_current")), None)
    if current_job is None and jobs:
        current_job = jobs[-1]

    evidence_history = _collect_evidence_history(skill, current_job)
    artifacts = _collect_artifacts(jobs)
    operational_handoff = _operational_handoff(current_job)
    acquisition_chain = _collect_acquisition_chain(current_job, acquisition_orchestrator)
    required = list(skill.get("verification_required") or _job_required(current_job) or [])
    missing = _missing_proof(required, evidence_history, artifacts)
    evidence_classes = _classify_evidence(skill, current_job, artifacts, evidence_history, acquisition_chain)

    return {
        "schema_version": 1,
        "skill_id": skill_id,
        "status": skill.get("status", "unknown"),
        "verified": skill.get("status") == "verified",
        "decision_summary": _decision_summary(skill, current_job, missing, evidence_classes, operational_handoff),
        "request_context": _request_context(current_job),
        "resolver_contract": _resolver_contract(skill, current_job, required),
        "operational_handoff": operational_handoff,
        "timeline": _timeline(jobs),
        "artifacts": artifacts,
        "acquisition_chain": acquisition_chain,
        "evidence_history": evidence_history,
        "evidence_classes": evidence_classes,
        "missing_proof": missing,
        "jobs": [
            {
                "job_id": j.get("job_id"),
                "status": j.get("status"),
                "phase": j.get("phase"),
                "created_at": j.get("created_at"),
                "updated_at": j.get("updated_at"),
                "is_current": j.get("is_current", False),
                "is_historical": j.get("is_historical", False),
            }
            for j in jobs
        ],
        "integrity_notes": [
            "Read-only audit packet built from SkillRegistry, LearningJob, and artifact JSON state.",
            "Dashboard endpoint does not execute skill code, smoke tests, plugins, or learning phases.",
            "Historical jobs are shown for traceability but do not imply current operational readiness.",
        ],
    }


def _collect_related_jobs(skill_id: str, current_job_id: str, orchestrator: Any) -> list[dict[str, Any]]:
    if orchestrator is None or not hasattr(orchestrator, "store"):
        return []

    canonical = _canonical(skill_id)
    raw_jobs = []
    try:
        raw_jobs = orchestrator.store.load_all()
    except Exception:
        raw_jobs = []

    jobs: list[dict[str, Any]] = []
    for job in raw_jobs:
        jid = getattr(job, "job_id", "")
        if _canonical(getattr(job, "skill_id", "")) != canonical:
            continue
        data = job.to_dict() if hasattr(job, "to_dict") else dict(getattr(job, "__dict__", {}))
        data["is_current"] = bool(current_job_id and jid == current_job_id)
        data["is_historical"] = not data["is_current"]
        jobs.append(data)

    jobs.sort(key=lambda j: j.get("created_at", ""))
    return jobs


def _collect_evidence_history(skill: dict[str, Any], current_job: dict[str, Any] | None) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    if current_job:
        for ev in (current_job.get("evidence") or {}).get("history", []) or []:
            evidence.append(_normalize_evidence(ev, source="current_job", is_current=True))

    latest = skill.get("verification_latest")
    if latest:
        evidence.append(_normalize_evidence(latest, source="skill_registry_latest", is_current=True))

    for ev in skill.get("verification_history", []) or []:
        evidence.append(_normalize_evidence(ev, source="skill_registry_history", is_current=False))

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ev in evidence:
        key = (str(ev.get("evidence_id", "")), str(ev.get("source", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped


def _collect_artifacts(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for job in jobs:
        for artifact in job.get("artifacts", []) or []:
            item = {
                "job_id": job.get("job_id", ""),
                "is_current_job": bool(job.get("is_current")),
                "id": artifact.get("id", ""),
                "type": artifact.get("type", ""),
                "path": artifact.get("path", ""),
                "details": artifact.get("details", {}),
            }
            item.update(_artifact_preview(item["path"]))
            artifacts.append(item)
    return artifacts


def _artifact_preview(path: str) -> dict[str, Any]:
    if not path:
        return {"exists": False, "preview": None}
    p = Path(path).expanduser()
    exists = p.exists()
    result: dict[str, Any] = {"exists": exists}
    if not exists or not p.is_file():
        result["preview"] = None
        return result
    try:
        if p.suffix.lower() == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            result["preview"] = _bounded(raw)
            result["preview_type"] = "json"
        else:
            result["preview"] = p.read_text(encoding="utf-8", errors="replace")[:_PREVIEW_LIMIT]
            result["preview_type"] = "text"
    except Exception as exc:
        result["preview_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        result["preview"] = None
    return result


def _normalize_evidence(evidence: dict[str, Any], *, source: str, is_current: bool) -> dict[str, Any]:
    tests = []
    for test in evidence.get("tests", []) or []:
        tests.append({
            "name": test.get("name", ""),
            "passed": bool(test.get("passed")),
            "details": test.get("details", ""),
            "expected": _bounded(test.get("expected")),
            "actual": _bounded(test.get("actual")),
            "artifact_refs": test.get("artifact_refs", []) or [],
        })
    return {
        "evidence_id": evidence.get("evidence_id", ""),
        "timestamp": evidence.get("ts") or evidence.get("timestamp"),
        "result": evidence.get("result", ""),
        "details": evidence.get("details", ""),
        "verification_method": evidence.get("verification_method", ""),
        "verification_scope": evidence.get("verification_scope", ""),
        "source": source,
        "is_current": is_current,
        "tests": tests,
    }


def _request_context(current_job: dict[str, Any] | None) -> dict[str, Any]:
    if not current_job:
        return {}
    requested = current_job.get("requested_by", {}) or {}
    return {
        "job_id": current_job.get("job_id", ""),
        "created_at": current_job.get("created_at", ""),
        "updated_at": current_job.get("updated_at", ""),
        "requested_by": requested,
        "user_text": requested.get("user_text", ""),
        "speaker": requested.get("speaker", ""),
        "risk_level": current_job.get("risk_level", ""),
        "matrix_protocol": bool(current_job.get("matrix_protocol")),
        "protocol_id": current_job.get("protocol_id", ""),
    }


def _resolver_contract(
    skill: dict[str, Any],
    current_job: dict[str, Any] | None,
    required: list[str],
) -> dict[str, Any]:
    plan = (current_job or {}).get("plan", {}) or {}
    return {
        "skill_id": skill.get("skill_id", ""),
        "capability_type": skill.get("capability_type", (current_job or {}).get("capability_type", "")),
        "required_evidence": required,
        "capability_contract": plan.get("capability_contract", {}) or {},
        "plan_summary": plan.get("summary", ""),
        "phases": plan.get("phases", []),
    }


def _timeline(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job.get("job_id", "")
        entries.append({
            "ts": job.get("created_at", ""),
            "job_id": job_id,
            "type": "job_created",
            "message": "Learning job created",
            "is_current_job": bool(job.get("is_current")),
        })
        for event in job.get("events", []) or []:
            entries.append({
                "ts": event.get("ts", ""),
                "job_id": job_id,
                "type": event.get("type", ""),
                "message": event.get("msg", ""),
                "is_current_job": bool(job.get("is_current")),
            })
    entries.sort(key=lambda e: (str(e.get("ts", "")), str(e.get("job_id", ""))))
    return entries[-200:]


def _classify_evidence(
    skill: dict[str, Any],
    current_job: dict[str, Any] | None,
    artifacts: list[dict[str, Any]],
    evidence_history: list[dict[str, Any]],
    acquisition_chain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    has_contract = any(a.get("type") == "contract_smoke_result" and a.get("is_current_job") for a in artifacts)
    current_tests = [
        test
        for ev in evidence_history
        if ev.get("is_current")
        for test in ev.get("tests", [])
    ]
    operational = has_contract and current_tests and all(t.get("passed") for t in current_tests)
    matrix = bool((current_job or {}).get("matrix_protocol"))
    plugin = any(a.get("type") in ("plugin_quarantined", "plugin_activation", "plugin_artifact") for a in artifacts)
    if acquisition_chain and acquisition_chain.get("plugin_name"):
        plugin = True
    lifecycle = bool(current_job)
    return {
        "lifecycle_evidence": lifecycle,
        "operational_contract_evidence": operational,
        "matrix_advisory_evidence": matrix and not operational,
        "plugin_or_acquisition_evidence": plugin,
        "self_improvement_evidence": skill.get("skill_id") == "self_improvement",
    }


def _collect_acquisition_chain(
    current_job: dict[str, Any] | None,
    acquisition_orchestrator: Any,
) -> dict[str, Any]:
    if not current_job or acquisition_orchestrator is None:
        return {}

    acquisition_id = current_job.get("parent_acquisition_id", "")
    if not acquisition_id:
        handoff = (current_job.get("data") or {}).get("operational_handoff", {})
        if isinstance(handoff, dict):
            acquisition_id = handoff.get("acquisition_id", "")
    if not acquisition_id:
        return {}

    try:
        acq_job = acquisition_orchestrator.get_job(acquisition_id)
    except Exception:
        acq_job = None
    if acq_job is None:
        return {"acquisition_id": acquisition_id, "exists": False}

    chain: dict[str, Any] = {
        "acquisition_id": acquisition_id,
        "exists": True,
        "status": getattr(acq_job, "status", ""),
        "outcome_class": getattr(acq_job, "outcome_class", ""),
        "risk_tier": getattr(acq_job, "risk_tier", ""),
        "plugin_name": getattr(acq_job, "plugin_id", ""),
        "verification_id": getattr(acq_job, "verification_id", ""),
        "code_bundle_id": getattr(acq_job, "code_bundle_id", ""),
        "lanes": {
            name: {
                "status": getattr(lane, "status", ""),
                "child_id": getattr(lane, "child_id", ""),
                "error": getattr(lane, "error", ""),
            }
            for name, lane in (getattr(acq_job, "lanes", {}) or {}).items()
        },
    }

    verification_id = chain.get("verification_id")
    if verification_id:
        try:
            verification = acquisition_orchestrator._store.load_verification(verification_id)  # noqa: SLF001
        except Exception:
            verification = None
        if verification is not None:
            chain["verification"] = {
                "overall_passed": getattr(verification, "overall_passed", False),
                "sandbox_result_ref": getattr(verification, "sandbox_result_ref", ""),
                "lane_verdicts": getattr(verification, "lane_verdicts", {}),
                "risk_assessment": _bounded(getattr(verification, "risk_assessment", {})),
            }

    plugin_name = chain.get("plugin_name")
    if plugin_name:
        try:
            from tools.plugin_registry import get_plugin_registry
            plugin_registry = get_plugin_registry()
            rec = plugin_registry.get_record(plugin_name)
        except Exception:
            rec = None
        if rec is not None:
            chain["plugin"] = {
                "name": getattr(rec, "name", plugin_name),
                "state": getattr(rec, "state", ""),
                "supervision_mode": getattr(rec, "supervision_mode", ""),
                "risk_tier": getattr(rec, "risk_tier", ""),
                "execution_mode": getattr(rec, "execution_mode", ""),
                "code_hash": getattr(rec, "code_hash", ""),
            }
    return chain


def _operational_handoff(current_job: dict[str, Any] | None) -> dict[str, Any]:
    if not current_job:
        return {}
    data = current_job.get("data") or {}
    handoff = data.get("operational_handoff", {}) if isinstance(data, dict) else {}
    if not isinstance(handoff, dict):
        return {}
    payload = handoff.get("payload") if isinstance(handoff.get("payload"), dict) else {}
    fixtures = payload.get("smoke_fixtures", []) if isinstance(payload, dict) else []
    return {
        "status": handoff.get("status", ""),
        "approval_required": bool(handoff.get("approval_required")),
        "contract_id": handoff.get("contract_id", ""),
        "required_executor_kind": handoff.get("required_executor_kind", ""),
        "acquisition_id": handoff.get("acquisition_id", ""),
        "risk_tier": handoff.get("risk_tier"),
        "outcome_class": handoff.get("outcome_class"),
        "approved_by": handoff.get("approved_by", ""),
        "approved_at": handoff.get("approved_at", ""),
        "rejected_by": handoff.get("rejected_by", ""),
        "rejected_at": handoff.get("rejected_at", ""),
        "rejection_reason": handoff.get("rejection_reason", ""),
        "last_error": handoff.get("last_error", ""),
        "last_error_type": handoff.get("last_error_type", ""),
        "updated_at": handoff.get("updated_at", ""),
        "smoke_fixture_count": len(fixtures) if isinstance(fixtures, list) else 0,
    }


def _missing_proof(
    required: list[str],
    evidence_history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    passed = {
        test.get("name")
        for ev in evidence_history
        if ev.get("is_current") and ev.get("result") == "pass"
        for test in ev.get("tests", [])
        if test.get("passed")
    }
    missing: list[dict[str, str]] = []
    for req in required:
        if req not in passed:
            missing.append({"name": req, "reason": _reason_for_missing(req, evidence_history, artifacts)})
    return missing


def _reason_for_missing(
    req: str,
    evidence_history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str:
    for ev in evidence_history:
        if not ev.get("is_current"):
            continue
        for test in ev.get("tests", []):
            if test.get("name") == req and not test.get("passed"):
                return test.get("details") or "test_failed"
    if req == "test:sandbox_execution_pass":
        has_sandbox = any(
            a.get("is_current_job") and a.get("type") in {
                "sandbox_execution_pass", "sandbox_execution_passed", "sandbox_pass", "sandbox_result",
            }
            for a in artifacts
        )
        if not has_sandbox:
            return "missing_sandbox_execution_artifact"
    return "no_current_passing_evidence"


def _decision_summary(
    skill: dict[str, Any],
    current_job: dict[str, Any] | None,
    missing: list[dict[str, str]],
    evidence_classes: dict[str, Any],
    operational_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = skill.get("status", "unknown")
    handoff_status = (operational_handoff or {}).get("status", "")
    if handoff_status == "awaiting_operator_approval":
        message = "Skill is waiting for operator approval before Jarvis may build an operational plugin/tool."
    elif handoff_status == "approval_failed":
        message = "Skill handoff approval failed before acquisition could start; see handoff error details."
    elif handoff_status == "awaiting_acquisition_proof":
        message = "Skill is waiting for the linked acquisition/plugin lane to produce sandbox-backed operational proof."
    elif handoff_status == "operator_rejected_operational_build":
        message = "Skill operational build was rejected by the operator; the skill remains unverified."
    elif status == "verified" and evidence_classes.get("operational_contract_evidence"):
        message = "Skill is operationally verified by current contract evidence."
    elif status == "verified":
        message = "Skill is marked verified, but no current operational contract evidence was found in this packet."
    elif current_job and missing:
        message = "Skill is still learning; operational verification is waiting on required proof."
    elif current_job:
        message = "Skill has an active or historical learning job; no verification decision has completed yet."
    else:
        message = "No current learning job or verification evidence is linked to this skill."
    return {
        "status": status,
        "message": message,
        "current_phase": (current_job or {}).get("phase", ""),
        "current_job_status": (current_job or {}).get("status", ""),
        "missing_count": len(missing),
    }


def _job_required(current_job: dict[str, Any] | None) -> list[str]:
    if not current_job:
        return []
    return list(((current_job.get("evidence") or {}).get("required") or []))


def _canonical(skill_id: str) -> str:
    return re.sub(r"_v\d+$", "", skill_id or "")


def _bounded(value: Any) -> Any:
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=True)
    except TypeError:
        encoded = str(value)
    if len(encoded) <= _PREVIEW_LIMIT:
        return value
    return encoded[:_PREVIEW_LIMIT] + "...[truncated]"
