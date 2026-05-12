"""Operational proof bridge between skill contracts and acquisition plugins.

This module contains glue only. It does not generate code, activate plugins, or
rewrite SkillRegistry truth. It creates/read links between a LearningJob and the
governed acquisition/plugin lane, then exposes only proven plugin callables for
contract verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

HANDOFF_ARTIFACT_ID = "operational_handoff_required"
HANDOFF_ARTIFACT_TYPE = "operational_handoff_required"
CALLABLE_ARTIFACT_ID = "operational_callable_path"
SANDBOX_ARTIFACT_ID = "sandbox_execution_pass"
WAITING_FOR_ACQUISITION = "waiting_for_acquisition_proof"
AWAITING_OPERATOR_APPROVAL = "awaiting_operator_approval"
AWAITING_ACQUISITION_PROOF = "awaiting_acquisition_proof"
HANDOFF_REJECTED = "operator_rejected_operational_build"
ACQUISITION_FAILED = "acquisition_failed"
ACQUISITION_CANCELLED = "acquisition_cancelled"


def ensure_operational_handoff(
    job: Any,
    contract: Any,
    ctx: dict[str, Any],
    artifact_dir: str,
) -> dict[str, Any] | None:
    """Create a persisted operator-approval request for missing proof.

    Returns a LearningJob artifact dict when the handoff artifact is newly
    created. Existing handoffs return ``None`` to avoid duplicate artifacts.
    """

    if not getattr(contract, "acquisition_eligible", False):
        return None

    payload = _handoff_payload(job, contract)

    if not hasattr(job, "data") or getattr(job, "data", None) is None:
        job.data = {}

    existing_handoff = job.data.get("operational_handoff", {})
    if isinstance(existing_handoff, dict):
        existing_acquisition_id = existing_handoff.get("acquisition_id", "") or _linked_acquisition_id(job)
        existing_status = existing_handoff.get("status", "")
        if existing_acquisition_id or existing_status in {AWAITING_ACQUISITION_PROOF, HANDOFF_REJECTED, "approval_failed"}:
            preserved_status = existing_status or AWAITING_ACQUISITION_PROOF
            existing_handoff.update({
                "status": preserved_status,
                "contract_id": getattr(contract, "contract_id", ""),
                "required_executor_kind": getattr(contract, "required_executor_kind", ""),
                "acquisition_id": existing_acquisition_id,
                "risk_tier": existing_handoff.get("risk_tier", payload.get("risk_tier")),
                "outcome_class": existing_handoff.get("outcome_class", payload.get("outcome_class")),
                "updated_at": _utc_iso(),
            })
            existing = _find_artifact(job, HANDOFF_ARTIFACT_TYPE)
            if existing:
                existing.setdefault("details", {}).update({
                    "contract_id": payload.get("contract_id", ""),
                    "acquisition_id": existing_acquisition_id,
                    "required_executor_kind": payload.get("required_executor_kind", ""),
                    "status": preserved_status,
                })
            return None

    job.data.setdefault("operational_handoff", {}).update({
        "status": AWAITING_OPERATOR_APPROVAL,
        "contract_id": getattr(contract, "contract_id", ""),
        "required_executor_kind": getattr(contract, "required_executor_kind", ""),
        "acquisition_id": _linked_acquisition_id(job),
        "risk_tier": payload.get("risk_tier"),
        "outcome_class": payload.get("outcome_class"),
        "payload": payload,
        "approval_required": True,
        "last_error": "",
        "last_error_type": "",
        "updated_at": _utc_iso(),
    })
    job.status = AWAITING_OPERATOR_APPROVAL
    if hasattr(job, "updated_at"):
        job.updated_at = _utc_iso()
    if hasattr(job, "failure"):
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}

    existing = _find_artifact(job, HANDOFF_ARTIFACT_TYPE)
    if existing:
        existing.setdefault("details", {}).update({
            "contract_id": payload.get("contract_id", ""),
            "acquisition_id": _linked_acquisition_id(job),
            "required_executor_kind": payload.get("required_executor_kind", ""),
            "status": AWAITING_OPERATOR_APPROVAL,
        })
        return None

    path = os.path.join(artifact_dir, "operational_handoff_required.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "id": HANDOFF_ARTIFACT_ID,
        "type": HANDOFF_ARTIFACT_TYPE,
        "path": path,
        "details": {
            "contract_id": payload.get("contract_id", ""),
            "acquisition_id": _linked_acquisition_id(job),
            "required_executor_kind": payload.get("required_executor_kind", ""),
            "status": AWAITING_OPERATOR_APPROVAL,
        },
    }


def start_operational_handoff(
    job: Any,
    contract: Any,
    acq: Any,
    *,
    approved_by: str,
    notes: str = "",
) -> tuple[bool, str]:
    """Create/link the governed acquisition job after operator approval."""

    if acq is None:
        _record_handoff_error(job, "acquisition_orchestrator_unavailable", "RuntimeError")
        return False, "acquisition_orchestrator_unavailable"

    data = getattr(job, "data", {}) or {}
    handoff = data.get("operational_handoff", {}) if isinstance(data, dict) else {}
    payload = handoff.get("payload") if isinstance(handoff, dict) else None
    if not isinstance(payload, dict):
        payload = _handoff_payload(job, contract)

    try:
        acquisition_id = _ensure_acquisition_job(job, contract, acq, payload)
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:300]}"
        _record_handoff_error(job, message, type(exc).__name__)
        logger.warning(
            "Skill operational handoff failed for job=%s skill=%s: %s",
            getattr(job, "job_id", ""),
            getattr(job, "skill_id", ""),
            message,
        )
        return False, message

    if not acquisition_id:
        _record_handoff_error(job, "acquisition_id_missing_after_creation", "RuntimeError")
        return False, "acquisition_id_missing_after_creation"

    if not hasattr(job, "data") or getattr(job, "data", None) is None:
        job.data = {}
    job.data.setdefault("operational_handoff", {}).update({
        "status": AWAITING_ACQUISITION_PROOF,
        "approval_required": False,
        "approved_by": approved_by,
        "approval_notes": notes,
        "approved_at": _utc_iso(),
        "acquisition_id": acquisition_id,
        "last_error": "",
        "last_error_type": "",
        "updated_at": _utc_iso(),
    })
    job.parent_acquisition_id = acquisition_id
    if hasattr(job, "status"):
        job.status = "active"
    if hasattr(job, "updated_at"):
        job.updated_at = _utc_iso()
    if hasattr(job, "failure"):
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
    return True, acquisition_id


def sync_acquisition_proof(job: Any, ctx: dict[str, Any]) -> None:
    """Mirror proven acquisition/plugin refs onto the learning job.

    The refs are not authority by themselves. They are pointers used by the
    contract smoke verifier to require both callable and sandbox evidence.
    """

    acq = ctx.get("acquisition_orchestrator")
    acquisition_id = _linked_acquisition_id(job)
    if acq is None or not acquisition_id:
        return

    try:
        acq_job = acq.get_job(acquisition_id)
    except Exception:
        return
    if acq_job is None:
        return

    acq_status = getattr(acq_job, "status", "") or ""
    if acq_status in {"failed", "cancelled"}:
        _record_terminal_acquisition(job, acq_job, acquisition_id, acq_status)
        return

    plugin_name = getattr(acq_job, "plugin_id", "") or ""
    verification_id = getattr(acq_job, "verification_id", "") or ""
    if not plugin_name or not verification_id:
        return

    verification = None
    try:
        verification = acq._store.load_verification(verification_id)  # noqa: SLF001 - read-only proof bridge
    except Exception:
        verification = None
    if verification is None or not getattr(verification, "overall_passed", False):
        return

    try:
        from tools.plugin_registry import get_plugin_registry
        registry = get_plugin_registry()
        rec = registry.get_record(plugin_name)
    except Exception:
        rec = None
    if rec is None or getattr(rec, "state", "") not in {"supervised", "active"}:
        return

    _append_unique_artifact(job, {
        "id": CALLABLE_ARTIFACT_ID,
        "type": CALLABLE_ARTIFACT_ID,
        "details": {
            "source": "acquisition_plugin",
            "acquisition_id": acquisition_id,
            "plugin_name": plugin_name,
            "plugin_state": getattr(rec, "state", ""),
            "verification_id": verification_id,
        },
    })
    _append_unique_artifact(job, {
        "id": SANDBOX_ARTIFACT_ID,
        "type": SANDBOX_ARTIFACT_ID,
        "details": {
            "source": "acquisition_verification",
            "acquisition_id": acquisition_id,
            "plugin_name": plugin_name,
            "verification_id": verification_id,
            "sandbox_result_ref": getattr(verification, "sandbox_result_ref", ""),
        },
    })


def _record_terminal_acquisition(job: Any, acq_job: Any, acquisition_id: str, acq_status: str) -> None:
    """Mirror terminal acquisition state onto the learning-job handoff."""
    if not hasattr(job, "data") or getattr(job, "data", None) is None:
        job.data = {}
    handoff = job.data.setdefault("operational_handoff", {})
    if not isinstance(handoff, dict):
        handoff = {}
        job.data["operational_handoff"] = handoff

    lane_name, lane_error = _terminal_lane_error(acq_job)
    status = ACQUISITION_CANCELLED if acq_status == "cancelled" else ACQUISITION_FAILED
    handoff.update({
        "status": status,
        "approval_required": False,
        "acquisition_id": acquisition_id,
        "terminal_acquisition_status": acq_status,
        "terminal_lane": lane_name,
        "terminal_error": lane_error,
        "last_error": lane_error or acq_status,
        "last_error_type": status,
        "updated_at": _utc_iso(),
    })
    if hasattr(job, "failure") and isinstance(job.failure, dict):
        job.failure = {
            "count": 0,
            "last_error": f"{status}:{lane_error or acq_status}",
            "last_failed_phase": getattr(job, "phase", ""),
        }
    if hasattr(job, "status"):
        job.status = "blocked"
    if hasattr(job, "updated_at"):
        job.updated_at = _utc_iso()
    if hasattr(job, "events"):
        last_msg = f"{status}:{acquisition_id}:{lane_name or 'unknown'}"
        if not job.events or job.events[-1].get("type") != status or job.events[-1].get("msg") != last_msg:
            job.events.append({"ts": _utc_iso(), "type": status, "msg": last_msg})


def _terminal_lane_error(acq_job: Any) -> tuple[str, str]:
    lanes = getattr(acq_job, "lanes", {}) or {}
    for name, lane in lanes.items():
        status = getattr(lane, "status", "")
        error = getattr(lane, "error", "")
        if status in {"failed", "blocked"} or error:
            return name, error
    return "", ""


def build_skill_execution_callables(acquisition_orchestrator: Any) -> dict[str, Callable[..., Any]]:
    """Return callables for skills backed by supervised/active plugins only."""

    callables: dict[str, Callable[..., Any]] = {}
    if acquisition_orchestrator is None:
        return callables

    try:
        from tools.plugin_registry import get_plugin_registry
        from tools.plugin_registry import PluginRequest
        registry = get_plugin_registry()
    except Exception:
        return callables

    try:
        acq_jobs = acquisition_orchestrator._store.list_jobs()  # noqa: SLF001 - read-only provider
    except Exception:
        acq_jobs = []

    for acq_job in acq_jobs:
        skill_id = (getattr(acq_job, "requested_by", {}) or {}).get("skill_id", "")
        plugin_name = getattr(acq_job, "plugin_id", "") or ""
        verification_id = getattr(acq_job, "verification_id", "") or ""
        if not skill_id or not plugin_name or not verification_id:
            continue
        rec = registry.get_record(plugin_name)
        if rec is None or getattr(rec, "state", "") not in {"supervised", "active"}:
            continue
        verification = None
        try:
            verification = acquisition_orchestrator._store.load_verification(verification_id)  # noqa: SLF001
        except Exception:
            verification = None
        if verification is None or not getattr(verification, "overall_passed", False):
            continue

        def _make_callable(name: str, sid: str) -> Callable[..., Any]:
            def _call(input_value: Any, fixture: Any = None) -> Any:
                context = {
                    "skill_id": sid,
                    "fixture_name": getattr(fixture, "name", ""),
                    "input_type": getattr(fixture, "input_type", ""),
                }
                request = PluginRequest(
                    request_id=f"skill_verify_{sid}_{int(time.time() * 1000)}",
                    plugin_name=name,
                    user_text=input_value if isinstance(input_value, str) else json.dumps(input_value),
                    context=context,
                    timeout_s=30.0,
                )
                response = _run_plugin_invocation(registry, request)
                if not response.success:
                    raise RuntimeError(response.error or "plugin invocation failed")
                return _normalize_plugin_result(response.result)

            return _call

        callables[skill_id] = _make_callable(plugin_name, skill_id)
    return callables


def _run_plugin_invocation(registry: Any, request: Any) -> Any:
    async def _invoke() -> Any:
        return await registry.invoke(request)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_invoke())

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_invoke())).result(timeout=(request.timeout_s or 30.0) + 5.0)
    return asyncio.run(_invoke())


def _normalize_plugin_result(result: Any) -> Any:
    if isinstance(result, dict) and "output" in result and len(result) == 1:
        output = result.get("output")
        if isinstance(output, (dict, list)):
            return output
        if isinstance(output, str):
            try:
                return json.loads(output)
            except Exception:
                return output
    return result


def _ensure_acquisition_job(job: Any, contract: Any, acq: Any, payload: dict[str, Any]) -> str:
    existing_id = _linked_acquisition_id(job)
    if existing_id:
        return existing_id

    acq_job = acq.create_skill_proof_handoff(job, contract, payload)
    acquisition_id = getattr(acq_job, "acquisition_id", "") if acq_job else ""
    if acquisition_id:
        job.parent_acquisition_id = acquisition_id
    return acquisition_id


def _handoff_payload(job: Any, contract: Any) -> dict[str, Any]:
    return {
        "skill_id": getattr(job, "skill_id", ""),
        "learning_job_id": getattr(job, "job_id", ""),
        "contract_id": getattr(contract, "contract_id", ""),
        "family": getattr(contract, "family", ""),
        "required_executor_kind": getattr(contract, "required_executor_kind", ""),
        "smoke_test_name": getattr(contract, "smoke_test_name", ""),
        "smoke_fixtures": [
            {
                "name": getattr(fixture, "name", ""),
                "input_type": getattr(fixture, "input_type", ""),
                "input": getattr(fixture, "input", None),
                "expected": getattr(fixture, "expected", {}),
            }
            for fixture in getattr(contract, "smoke_fixtures", ()) or ()
        ],
        "created_at": _utc_iso(),
        "status": AWAITING_OPERATOR_APPROVAL,
        "risk_tier": 2,
        "outcome_class": "plugin_creation",
    }


def _linked_acquisition_id(job: Any) -> str:
    direct = getattr(job, "parent_acquisition_id", "") or ""
    if direct:
        return direct
    data = getattr(job, "data", {}) or {}
    handoff = data.get("operational_handoff", {}) if isinstance(data, dict) else {}
    return handoff.get("acquisition_id", "") if isinstance(handoff, dict) else ""


def _find_artifact(job: Any, artifact_type: str) -> dict[str, Any] | None:
    for artifact in getattr(job, "artifacts", []) or []:
        if artifact.get("type") == artifact_type or artifact.get("id") == artifact_type:
            return artifact
    return None


def _append_unique_artifact(job: Any, artifact: dict[str, Any]) -> None:
    artifact_id = artifact.get("id", "")
    artifact_type = artifact.get("type", "")
    for existing in getattr(job, "artifacts", []) or []:
        if existing.get("id") == artifact_id or existing.get("type") == artifact_type:
            existing.setdefault("details", {}).update(artifact.get("details", {}))
            return
    job.artifacts.append(artifact)


def _record_handoff_error(job: Any, message: str, error_type: str) -> None:
    if not hasattr(job, "data") or getattr(job, "data", None) is None:
        job.data = {}
    job.data.setdefault("operational_handoff", {}).update({
        "status": "approval_failed",
        "last_error": message,
        "last_error_type": error_type,
        "updated_at": _utc_iso(),
    })


def _utc_iso() -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(time.time(), dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
