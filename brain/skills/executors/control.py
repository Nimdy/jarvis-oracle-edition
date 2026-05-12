"""Control phase executors — episodes + sim-first + safety gates.

Control skills NEVER learn without:
- Sim tests passing first
- user_present gating for real hardware
- kill_switch gating
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from skills.executors.base import PhaseExecutor, PhaseResult

logger = logging.getLogger(__name__)

_utc_iso = None


def _get_utc_iso() -> str:
    global _utc_iso
    if _utc_iso is None:
        try:
            from skills.learning_jobs import _utc_iso as _fn
            _utc_iso = _fn
        except Exception:
            import datetime as dt
            _utc_iso = lambda: dt.datetime.utcfromtimestamp(time.time()).replace(
                microsecond=0).isoformat() + "Z"
    return _utc_iso()


def _phase_exit_conditions(job: Any) -> list[str]:
    phases = getattr(job, "plan", {}).get("phases", []) or []
    current = getattr(job, "phase", "")
    for phase_entry in phases:
        if isinstance(phase_entry, dict) and phase_entry.get("name") == current:
            return list(phase_entry.get("exit_conditions", []) or [])
    return []


def _collect_metric_name(job: Any) -> str:
    for cond in _phase_exit_conditions(job):
        if not isinstance(cond, str) or not cond.startswith("metric:"):
            continue
        metric_expr = cond[len("metric:"):]
        for op in (">=", "<=", "==", ">", "<"):
            if op in metric_expr:
                return metric_expr.split(op, 1)[0].strip()
    return ""


def _required_evidence_names(job: Any) -> set[str]:
    required = getattr(job, "evidence", {}).get("required", []) or []
    names: set[str] = set()
    for req in required:
        req_str = str(req or "")
        if req_str:
            names.add(req_str)
            names.add(req_str.split(":", 1)[-1])
    return names


class ControlAssessExecutor(PhaseExecutor):
    capability_type = "control"
    phase = "assess"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        user_present = bool(ctx.get("user_present", False))
        hw_connected = bool(ctx.get("hardware_connected", False))
        kill_switch = bool(ctx.get("kill_switch_configured", False))
        sim_available = bool(ctx.get("sim_available", True))

        gate_updates = [
            {"id": "gate:user_present_required",
             "state": "pass" if user_present else "fail",
             "details": "User must be present for real-hardware control."},
            {"id": "gate:hardware_connected",
             "state": "pass" if hw_connected else "fail",
             "details": "Hardware must be connected / detectable."},
            {"id": "gate:kill_switch_configured",
             "state": "pass" if kill_switch else "fail",
             "details": "Kill switch must be configured before real runs."},
            {"id": "gate:sim_available",
             "state": "pass" if sim_available else "fail",
             "details": "A sim or dry-run environment must exist."},
        ]
        return PhaseResult(progressed=True, message="Control assess complete.", gate_updates=gate_updates)


class ControlCollectExecutor(PhaseExecutor):
    capability_type = "control"
    phase = "collect"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        sim_ok = ctx.get("sim_available", True)
        if not sim_ok:
            return PhaseResult(progressed=False, message="No sim available; cannot collect safely.")

        episodes_dir = os.path.expanduser(f"~/.jarvis/episodes/{job.skill_id}")
        os.makedirs(episodes_dir, exist_ok=True)

        metric_name = _collect_metric_name(job) or "episodes"
        cur = float(job.data.get("counters", {}).get(metric_name, 0))
        new = cur + float(ctx.get("episodes_collected_this_tick", 1))

        return PhaseResult(
            progressed=True,
            message=f"Collected episodes (sim): now {int(new)}",
            metric_updates={metric_name: new},
            artifact={"id": "artifact_episode_dir", "type": "episodes_dir", "path": episodes_dir},
        )


class ControlTrainExecutor(PhaseExecutor):
    capability_type = "control"
    phase = "train"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        ep = float(job.data.get("counters", {}).get("episodes", 0))
        if ep < 10:
            return PhaseResult(progressed=False, message="Not enough episodes to train (need >=10).")

        ckpt_dir = os.path.expanduser(f"~/.jarvis/models/control/{job.skill_id}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "policy_head_v1.pt")

        with open(ckpt_path, "w", encoding="utf-8") as f:
            f.write("placeholder checkpoint\n")

        artifact = {"id": "artifact_model_checkpoint", "type": "model_checkpoint", "path": ckpt_path}
        return PhaseResult(progressed=True, message="Control checkpoint produced.", artifact=artifact)


class ControlVerifyExecutor(PhaseExecutor):
    capability_type = "control"
    phase = "verify"

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        user_present = bool(ctx.get("user_present", False))
        sim_pass = bool(ctx.get("sim_test_pass", False))
        real_pass = bool(ctx.get("real_test_pass", False)) if user_present else False

        now_iso = ctx.get("now_iso", "")
        all_pass = sim_pass and real_pass
        required_names = _required_evidence_names(job)
        sim_test_name = "sim:test_pick_place_10runs"
        real_test_name = "real:test_pick_place_3runs_user_present"
        for name in required_names:
            if name.startswith("sim:test_"):
                sim_test_name = name if ":" in name else f"sim:{name}"
                break
        for name in required_names:
            if name.startswith("real:test_"):
                real_test_name = name if ":" in name else f"real:{name}"
                break
        evidence = {
            "evidence_id": "verification_run",
            "result": "pass" if all_pass else "fail",
            "ts": now_iso,
            "tests": [
                {"name": sim_test_name, "passed": sim_pass,
                 "details": "Sim run must pass before real."},
                {"name": real_test_name, "passed": real_pass,
                 "details": "Real test requires user present + kill switch."},
            ],
        }
        return PhaseResult(progressed=True, message="Control verification recorded.", evidence=evidence)


class ControlRegisterExecutor(PhaseExecutor):
    """Final phase: checks evidence and flips the control skill to verified."""
    capability_type = "control"
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
        from skills.registry import SkillEvidence
        evidence = SkillEvidence(
            evidence_id=f"register_{job.skill_id}_{_get_utc_iso()}",
            timestamp=time.time(),
            result="pass",
            tests=latest_verify.get("tests", []) if latest_verify else [
                {"name": t, "passed": True, "details": "Verified during control learning job"}
                for t in passed_tests
            ],
            verified_by=self.__class__.__name__,
            acceptance_criteria=build_acceptance_criteria(job),
            measured_values=build_measured_values(job),
            environment=capture_environment(ctx),
            summary=collect_verify_details(job),
            verification_method="learning_job_control_sim_and_real",
            evidence_schema_version="2",
            artifact_refs=collect_artifact_refs(job),
            verification_scope="functional",
            known_limitations=["requires user_present for real-hardware validation", "sim-first verification"],
            regression_baseline_available=False,
        )
        ok = registry.set_status(job.skill_id, "verified", evidence=evidence)
        if ok:
            logger.info("Control skill %s registered as verified via job %s", job.skill_id, job.job_id)
            return PhaseResult(progressed=True, message=f"Skill {job.skill_id} verified and registered.")

        return PhaseResult(progressed=False, message=f"Registry rejected verification for {job.skill_id}.")
