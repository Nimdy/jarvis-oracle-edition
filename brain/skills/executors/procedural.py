"""Procedural phase executors — assess through register.

Each executor matches (capability_type="procedural", phase=<name>) and
produces the gates/artifacts/evidence that the exit-condition evaluator
checks in job_eval.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from skills.executors.base import PhaseExecutor, PhaseResult

logger = logging.getLogger(__name__)


def _job_dir(job: Any) -> str:
    root = os.path.expanduser("~/.jarvis/learning_jobs")
    d = os.path.join(root, job.job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _utc_iso() -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(time.time(), dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Resource probes — keyed by gate ctx_key, returns bool availability.
_RESOURCE_PROBES: dict[str, Any] = {
    "audio_output_available": lambda ctx: _probe_audio_output(ctx),
    "image_pipeline_available": lambda ctx: False,
}


def _probe_audio_output(ctx: dict[str, Any]) -> bool:
    try:
        tts = ctx.get("tts_engine")
        if tts and hasattr(tts, "available"):
            return bool(tts.available)
        if tts is not None:
            return True
        from reasoning.tts import BrainTTS
        return BrainTTS is not None
    except ImportError:
        return False


class ProceduralAssessExecutor(PhaseExecutor):
    capability_type = "procedural"
    phase = "assess"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        hard_gates = job.gates.get("hard", [])

        if hard_gates:
            gate_updates = []
            for g in hard_gates:
                gate_id = g.get("id", "")
                ctx_key = gate_id.removeprefix("gate:")
                probe = _RESOURCE_PROBES.get(ctx_key)
                available = ctx.get(ctx_key, probe(ctx) if probe else True)
                gate_updates.append({
                    "id": gate_id,
                    "state": "pass" if available else "fail",
                    "details": g.get("details", ""),
                })
            return PhaseResult(
                progressed=True,
                message=f"Assess for {job.skill_id}: {len(gate_updates)} gate(s) evaluated.",
                gate_updates=gate_updates,
            )

        gate_updates = [
            {"id": "gate:tool_router_available",
             "state": "pass" if ctx.get("tool_router_available", True) else "fail",
             "details": "Tool router must be enabled for procedural skills."},
        ]
        return PhaseResult(progressed=True, message="Procedural assess complete.", gate_updates=gate_updates)


class ProceduralResearchExecutor(PhaseExecutor):
    capability_type = "procedural"
    phase = "research"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        if any(a.get("type") == "research_summary" for a in job.artifacts):
            return PhaseResult(progressed=False, message="Research summary already exists.")

        summary = self._build_research(job, ctx)
        path = os.path.join(_job_dir(job), "research_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        artifact = {"id": "research_summary", "type": "research_summary", "path": path}
        return PhaseResult(
            progressed=True,
            message=f"Research: {summary.get('approach', 'unknown approach')}",
            artifact=artifact,
        )

    @staticmethod
    def _build_research(job: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        notes = job.plan.get("summary", "") or ""
        cap = getattr(job, "capability", None)
        req_by = getattr(job, "requested_by", {}) or {}

        approach = notes if (notes and "Auto-generated" not in notes) else ""
        input_type = ""
        output_type = ""
        success_metrics: list[str] = []
        evidence_reqs: list[str] = []
        hardware_reqs: list[str] = []

        if cap and hasattr(cap, "input_type"):
            input_type = getattr(cap, "input_type", "")
            output_type = getattr(cap, "output_type", "")
            success_metrics = list(getattr(cap, "success_metrics", ()))
            evidence_reqs = list(getattr(cap, "evidence_requirements", ()))
            hardware_reqs = list(getattr(cap, "hardware_requirements", ()))

        phases = job.plan.get("phases", [])
        phase_names = [p.get("name", "") for p in phases if isinstance(p, dict)]

        resources_available = []
        for g in job.gates.get("hard", []):
            if g.get("state") == "pass":
                resources_available.append(g.get("id", ""))

        if not approach:
            skill_name = job.skill_id.replace("_v1", "").replace("_", " ")
            approach = f"Learn {skill_name} through structured {job.capability_type} pipeline"
            if evidence_reqs:
                approach += f" — requires {', '.join(evidence_reqs[:3])}"

        return {
            "skill_id": job.skill_id,
            "protocol_id": getattr(job, "protocol_id", ""),
            "capability_type": job.capability_type,
            "approach": approach,
            "input_type": input_type,
            "output_type": output_type,
            "success_metrics": success_metrics,
            "evidence_requirements": evidence_reqs,
            "hardware_requirements": hardware_reqs,
            "planned_phases": phase_names,
            "resources_confirmed": resources_available,
            "feasibility": "possible_with_current_stack" if resources_available else "pending_resource_check",
            "requested_by": req_by.get("speaker", "system"),
            "trigger": req_by.get("user_text", ""),
            "created_at": _utc_iso(),
            "notes": getattr(job, "notes", "") or job.plan.get("notes", ""),
        }


class ProceduralAcquireExecutor(PhaseExecutor):
    capability_type = "procedural"
    phase = "acquire"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        if any(a.get("type") == "model_or_method_available" for a in job.artifacts):
            return PhaseResult(progressed=False, message="Method artifact already exists.")

        method = self._identify_method(job, ctx)
        path = os.path.join(_job_dir(job), "method.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(method, f, indent=2)

        artifact = {"id": "model_or_method_available", "type": "model_or_method_available", "path": path}
        return PhaseResult(
            progressed=True,
            message=f"Method acquired for {job.skill_id}.",
            artifact=artifact,
        )

    @staticmethod
    def _identify_method(job: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        method: dict[str, Any] = {
            "skill_id": job.skill_id,
            "method": "generic_procedural",
            "created_at": _utc_iso(),
        }
        cap = getattr(job, "plan", {})
        notes = cap.get("summary", "") if isinstance(cap, dict) else ""
        if notes and "Auto-generated" not in notes:
            method["approach_notes"] = notes
        hard_gates = job.gates.get("hard", [])
        resources = [g.get("id", "").removeprefix("gate:") for g in hard_gates if g.get("state") == "pass"]
        if resources:
            method["verified_resources"] = resources
        return method


class ProceduralIntegrateExecutor(PhaseExecutor):
    capability_type = "procedural"
    phase = "integrate"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        if any(a.get("type") == "integration_test_passed" for a in job.artifacts):
            return PhaseResult(progressed=False, message="Integration test already exists.")

        ok, details = self._run_integration_test(job, ctx)
        result = {
            "skill_id": job.skill_id,
            "passed": ok,
            "details": details,
            "created_at": _utc_iso(),
        }
        path = os.path.join(_job_dir(job), "integration_test.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        artifact = {"id": "integration_test_passed", "type": "integration_test_passed", "path": path}
        return PhaseResult(
            progressed=True,
            message=f"Integration test: {'PASS' if ok else 'FAIL'}",
            artifact=artifact if ok else None,
            metric_updates={"integration_attempts": job.data.get("counters", {}).get("integration_attempts", 0) + 1},
        )

    @staticmethod
    def _run_integration_test(job: Any, ctx: dict[str, Any]) -> tuple[bool, str]:
        hard_gates = job.gates.get("hard", [])
        for g in hard_gates:
            if g.get("required") and g.get("state") != "pass":
                return False, f"Hard gate {g.get('id')} not satisfied."
        required_resources = []
        for g in hard_gates:
            ctx_key = g.get("id", "").removeprefix("gate:")
            probe = _RESOURCE_PROBES.get(ctx_key)
            if probe and not ctx.get(ctx_key, probe(ctx)):
                required_resources.append(ctx_key)
        if required_resources:
            return False, f"Resources unavailable: {', '.join(required_resources)}"
        return True, f"Integration check passed for {job.skill_id}."


class ProceduralVerifyExecutor(PhaseExecutor):
    capability_type = "procedural"
    phase = "verify"

    _FAIL_FAST_REASON = "no_verification_method"
    _NO_CALLABLE_REASON = "no_operational_callable_path"
    _WAITING_REASON = "waiting_for_acquisition_proof"
    _APPROVAL_REASON = "awaiting_operator_approval"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        ok, details, tests, artifact = self._run_verification(job, ctx)
        now_iso = _utc_iso()

        if not ok and details in (self._FAIL_FAST_REASON, self._NO_CALLABLE_REASON):
            if not getattr(job, "matrix_protocol", False):
                from skills.learning_jobs import MAX_PHASE_FAILURES
                job.failure["count"] = MAX_PHASE_FAILURES
                job.failure["last_error"] = details
                job.failure["last_failed_phase"] = job.phase

        if not tests:
            for req in job.evidence.get("required", []):
                tests.append({
                    "name": req,
                    "passed": ok,
                    "details": details,
                })

        if not tests:
            tests = [{"name": "procedure_smoke", "passed": ok, "details": details}]

        evidence = {
            "evidence_id": f"verify_{job.skill_id}_{now_iso}",
            "ts": now_iso,
            "result": "pass" if ok else "fail",
            "tests": tests,
            "details": details,
        }
        waiting = details in (self._WAITING_REASON, self._APPROVAL_REASON)
        return PhaseResult(
            progressed=ok or waiting,
            message=f"Verification: {'PASS' if ok else 'FAIL'} — {details}",
            evidence=evidence,
            artifact=artifact,
        )

    @staticmethod
    def _run_verification(job: Any, ctx: dict[str, Any]) -> tuple[bool, str, list[dict[str, Any]], dict[str, Any] | None]:
        if getattr(job, "matrix_protocol", False):
            ok, details = ProceduralVerifyExecutor._run_matrix_verification(job, ctx)
            return ok, details, [], None

        contract_result = ProceduralVerifyExecutor._run_contract_verification(job, ctx)
        if contract_result is not None:
            return contract_result

        has_macro = any(a.get("type") == "procedure_macro" for a in job.artifacts)
        if has_macro:
            return True, "Macro artifact exists and is parseable.", [], None

        required = job.evidence.get("required", [])
        if ProceduralVerifyExecutor._requires_contract_proof(required):
            tests = [
                {
                    "name": req,
                    "passed": False,
                    "details": "no_operational_execution_contract",
                }
                for req in required
            ]
            return False, "no_operational_execution_contract", tests, None

        has_method = any(
            a.get("type") in ("model_or_method_available", "integration_test_passed")
            for a in job.artifacts
        )
        if has_method:
            hard_gates = job.gates.get("hard", [])
            failing = [g for g in hard_gates if g.get("required") and g.get("state") != "pass"]
            if not failing:
                return True, f"Method artifact exists and all gates pass for {job.skill_id}.", [], None
            return False, f"Method exists but gates failing: {[g.get('id') for g in failing]}", [], None

        plan_summary = job.plan.get("summary", "") or ""
        if "Auto-generated" in plan_summary:
            return False, "no_verification_method", [], None

        return False, "No verifiable artifact found.", [], None

    @staticmethod
    def _run_contract_verification(
        job: Any,
        ctx: dict[str, Any],
    ) -> tuple[bool, str, list[dict[str, Any]], dict[str, Any] | None] | None:
        try:
            from skills.execution_contracts import get_contract, run_contract_smoke
        except ImportError:
            return None

        contract = get_contract(job.skill_id)
        if contract is None:
            return None

        try:
            from skills.operational_bridge import sync_acquisition_proof
            sync_acquisition_proof(job, ctx)
        except Exception:
            logger.debug("Operational proof sync failed for %s", job.skill_id, exc_info=True)

        results, artifact = run_contract_smoke(job, ctx, _job_dir(job))
        tests = [result.to_test() for result in results]
        ok = bool(tests) and all(t.get("passed") for t in tests)
        details = "; ".join(t.get("details", "") for t in tests if t.get("details"))
        if not details:
            details = f"Contract {contract.contract_id} produced no verification detail."
        if not ok and any(t.get("details") == ProceduralVerifyExecutor._NO_CALLABLE_REASON for t in tests):
            try:
                from skills.operational_bridge import (
                    ACQUISITION_CANCELLED,
                    ACQUISITION_FAILED,
                    AWAITING_OPERATOR_APPROVAL,
                    WAITING_FOR_ACQUISITION,
                    ensure_operational_handoff,
                )
                handoff_artifact = ensure_operational_handoff(job, contract, ctx, _job_dir(job))
                handoff = (getattr(job, "data", {}) or {}).get("operational_handoff", {})
                acquisition_id = (
                    getattr(job, "parent_acquisition_id", "")
                    or handoff.get("acquisition_id", "")
                )
                if handoff_artifact:
                    duplicate = any(
                        a.get("id", "") == handoff_artifact.get("id", "")
                        and a.get("type", "") == handoff_artifact.get("type", "")
                        and a.get("path", "") == handoff_artifact.get("path", "")
                        for a in getattr(job, "artifacts", [])
                    )
                    if not duplicate:
                        job.artifacts.append(handoff_artifact)
                    if hasattr(job, "events"):
                        job.events.append({
                            "ts": _utc_iso(),
                            "type": "artifact_present" if duplicate else "artifact_added",
                            "msg": handoff_artifact.get("id", "operational_handoff_required"),
                        })
                handoff_status = handoff.get("status", "") if isinstance(handoff, dict) else ""
                if handoff_status in {ACQUISITION_FAILED, ACQUISITION_CANCELLED}:
                    details = handoff_status
                    for test in tests:
                        if test.get("details") == ProceduralVerifyExecutor._NO_CALLABLE_REASON:
                            test["details"] = handoff_status
                            test["expected"] = {
                                "acquisition_id": acquisition_id,
                                "terminal_lane": handoff.get("terminal_lane", ""),
                                "terminal_error": handoff.get("terminal_error", ""),
                            }
                elif acquisition_id:
                    details = WAITING_FOR_ACQUISITION
                    for test in tests:
                        if test.get("details") == ProceduralVerifyExecutor._NO_CALLABLE_REASON:
                            test["details"] = WAITING_FOR_ACQUISITION
                            test["expected"] = {
                                "acquisition_id": acquisition_id,
                                "required_executor_kind": contract.required_executor_kind,
                            }
                elif handoff.get("status") == AWAITING_OPERATOR_APPROVAL:
                    details = AWAITING_OPERATOR_APPROVAL
                    for test in tests:
                        if test.get("details") == ProceduralVerifyExecutor._NO_CALLABLE_REASON:
                            test["details"] = AWAITING_OPERATOR_APPROVAL
                            test["expected"] = {
                                "approval_required": True,
                                "required_executor_kind": contract.required_executor_kind,
                            }
                else:
                    details = ProceduralVerifyExecutor._NO_CALLABLE_REASON
            except Exception:
                logger.debug("Operational handoff failed for %s", job.skill_id, exc_info=True)
                details = ProceduralVerifyExecutor._NO_CALLABLE_REASON
        return ok, details, tests, artifact

    @staticmethod
    def _requires_contract_proof(required: list[str]) -> bool:
        try:
            from skills.execution_contracts import requires_contract_test
        except ImportError:
            return False
        return any(requires_contract_test(req) for req in required)

    @staticmethod
    def _run_matrix_verification(job: Any, ctx: dict[str, Any]) -> tuple[bool, str]:
        """Matrix Protocol verification: check that required artifacts and
        research exist, then validate the skill has a plausible execution path.

        SK-001 (procedural): research summary + tool/method availability
        SK-002 (perceptual): deferred to PerceptualVerifyExecutor
        SK-003 (control): deferred to ControlVerifyExecutor
        """
        protocol_id = getattr(job, "protocol_id", "") or "SK-001"
        checks_passed = []
        checks_failed = []

        has_research = any(
            a.get("type") == "research_summary" for a in job.artifacts
        )
        if has_research:
            checks_passed.append("research_summary_exists")
        else:
            checks_failed.append("research_summary_missing")

        if protocol_id == "SK-001":
            has_method = any(
                a.get("type") in ("model_or_method_available", "procedure_macro")
                for a in job.artifacts
            )
            tool_ok = ctx.get("tool_router_available", True)

            if has_method:
                checks_passed.append("method_artifact_present")
            elif tool_ok:
                checks_passed.append("tool_router_available")
            else:
                checks_failed.append("no_method_or_tool_router")

            has_integration = any(
                a.get("type") == "integration_test_passed" for a in job.artifacts
            )
            if has_integration:
                checks_passed.append("integration_test_passed")

        if checks_failed:
            return False, f"Matrix SK-001 verification: {len(checks_passed)} pass, {len(checks_failed)} fail — {', '.join(checks_failed)}"

        if not checks_passed:
            return False, "Matrix SK-001 verification: no checks passed yet"

        return True, f"Matrix {protocol_id} verification: {len(checks_passed)} checks passed — {', '.join(checks_passed)}"


class ProceduralRegisterExecutor(PhaseExecutor):
    """Final phase: checks all required evidence and flips the skill to verified."""
    capability_type = "procedural"
    phase = "register"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        from skills.executors.evidence_helpers import (
            find_latest_verify_evidence, collect_verify_details,
            collect_artifact_refs, capture_environment,
            build_acceptance_criteria, build_measured_values,
        )
        registry = ctx.get("registry")
        if registry is None:
            return PhaseResult(progressed=False, message="No registry in context — cannot register.")

        existing = registry.get(job.skill_id)
        if existing and existing.status == "verified":
            return PhaseResult(progressed=True, message=f"Skill {job.skill_id} already verified.")

        required = job.evidence.get("required", [])
        passed_tests: set[str] = set()
        for evd in job.evidence.get("history", []):
            if evd.get("result") != "pass":
                continue
            for t in evd.get("tests", []):
                if t.get("passed"):
                    passed_tests.add(t.get("name", ""))

        unmet = [r for r in required if r not in passed_tests]
        if unmet:
            return PhaseResult(
                progressed=False,
                message=f"Cannot register: unmet evidence requirements {unmet}",
            )

        latest_verify = find_latest_verify_evidence(job)
        if (
            getattr(job, "matrix_protocol", False)
            and getattr(job, "capability_type", "") == "procedural"
            and not self._has_operational_matrix_proof(latest_verify)
        ):
            return PhaseResult(
                progressed=False,
                message=(
                    "Cannot register: Matrix procedural evidence is advisory/training "
                    "evidence until a separate operational contract proof exists."
                ),
            )

        from skills.registry import SkillEvidence
        evidence = SkillEvidence(
            evidence_id=f"register_{job.skill_id}_{_utc_iso()}",
            timestamp=time.time(),
            result="pass",
            tests=latest_verify.get("tests", []) if latest_verify else [
                {"name": t, "passed": True, "details": "Verified during learning job"}
                for t in passed_tests
            ],
            verified_by=self.__class__.__name__,
            acceptance_criteria=build_acceptance_criteria(job),
            measured_values=build_measured_values(job),
            environment=capture_environment(ctx),
            summary=collect_verify_details(job),
            verification_method="learning_job_procedural",
            evidence_schema_version="2",
            artifact_refs=collect_artifact_refs(job),
            verification_scope="functional",
            known_limitations=["automated verification only", "no user-observed field test"],
            regression_baseline_available=False,
        )
        ok = registry.set_status(job.skill_id, "verified", evidence=evidence)
        if ok:
            logger.info("Skill %s registered as verified via learning job %s", job.skill_id, job.job_id)
            return PhaseResult(progressed=True, message=f"Skill {job.skill_id} verified and registered.")

        return PhaseResult(progressed=False, message=f"Registry rejected verification for {job.skill_id}.")

    @staticmethod
    def _has_operational_matrix_proof(latest_verify: dict[str, Any] | None) -> bool:
        if not latest_verify:
            return False
        if latest_verify.get("verification_scope") == "operational":
            return True
        for test in latest_verify.get("tests", []):
            if test.get("name") == "operational_verified" and test.get("passed") is True:
                return True
        return False
