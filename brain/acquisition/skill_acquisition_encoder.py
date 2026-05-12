"""Shadow-only feature encoder for operational skill acquisition.

This encoder observes the skill-learning -> acquisition -> codegen -> plugin
proof path. It never mutates jobs, verifies skills, promotes plugins, or changes
capability truth. Its labels train a Tier-1 specialist to predict where the
operational proof path is likely to fail.
"""

from __future__ import annotations

from typing import Any

FEATURE_DIM = 40
LABEL_CLASSES = [
    "blocked",
    "planning_failed",
    "implementation_failed",
    "contract_failed",
    "verified",
]
_LABEL_INDEX = {name: i for i, name in enumerate(LABEL_CLASSES)}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class SkillAcquisitionEncoder:
    """Encode acquisition state into a bounded 40-dim vector."""

    @staticmethod
    def encode(job: Any, plan: Any = None, verification: Any = None) -> list[float]:
        vec = [0.0] * FEATURE_DIM

        requested = getattr(job, "requested_by", {}) or {}
        lanes = getattr(job, "lanes", {}) or {}
        planning_diag = getattr(job, "planning_diagnostics", {}) or {}
        prompt_diag = getattr(job, "codegen_prompt_diagnostics", {}) or {}

        # Block A: source and classification (0-7)
        vec[0] = 1.0 if requested.get("source") == "skill_operational_handoff" else 0.0
        vec[1] = 1.0 if requested.get("skill_id") else 0.0
        vec[2] = 1.0 if requested.get("contract_id") else 0.0
        vec[3] = _clamp(getattr(job, "risk_tier", 0) / 3.0)
        vec[4] = _clamp(getattr(job, "classification_confidence", 0.0))
        vec[5] = 1.0 if getattr(job, "outcome_class", "") == "plugin_creation" else 0.0
        vec[6] = _clamp(len(getattr(job, "doc_artifact_ids", []) or []) / 5.0)
        vec[7] = _clamp(len(getattr(job, "artifact_refs", []) or []) / 10.0)

        # Block B: lane progress/failures (8-17)
        lane_names = ["planning", "plan_review", "implementation", "plugin_quarantine", "verification"]
        for idx, name in enumerate(lane_names):
            lane = lanes.get(name)
            status = getattr(lane, "status", "") if lane is not None else ""
            vec[8 + idx] = 1.0 if status == "completed" else 0.0
            vec[13 + idx] = 1.0 if status == "failed" else 0.0

        # Block C: plan/prompt quality (18-27)
        if plan is not None:
            vec[18] = 1.0 if getattr(plan, "technical_approach", "") else 0.0
            vec[19] = 1.0 if getattr(plan, "implementation_sketch", "") else 0.0
            vec[20] = 1.0 if getattr(plan, "test_cases", None) else 0.0
            vec[21] = _clamp(len(getattr(plan, "test_cases", []) or []) / 5.0)
            vec[22] = _clamp(len(getattr(plan, "dependencies", []) or []) / 10.0)
        vec[23] = 1.0 if prompt_diag.get("prompt_hash") else 0.0
        vec[24] = 1.0 if prompt_diag.get("contract_id") else 0.0
        vec[25] = _clamp(len(str(prompt_diag.get("prompt_preview", ""))) / 1500.0)
        vec[26] = 0.0 if planning_diag.get("failure_reason") else 1.0
        vec[27] = _clamp(len(planning_diag.get("missing_fields", []) or []) / 3.0)

        # Block D: verification/contract outcome (28-35)
        if verification is not None:
            lane_verdicts = getattr(verification, "lane_verdicts", {}) or {}
            risk = getattr(verification, "risk_assessment", {}) or {}
            vec[28] = 1.0 if getattr(verification, "overall_passed", False) else 0.0
            vec[29] = 1.0 if lane_verdicts.get("sandbox_validation") else 0.0
            vec[30] = 1.0 if lane_verdicts.get("skill_contract_fixture") else 0.0
            vec[31] = 1.0 if risk.get("sandbox_status") == "passed" else 0.0
            vec[32] = 1.0 if risk.get("skill_contract_status") == "passed" else 0.0
            vec[33] = 1.0 if risk.get("skill_contract_status") in {"failed", "error"} else 0.0
            vec[34] = _clamp(len(risk.get("skill_contract_results", []) or []) / 5.0)
            vec[35] = 1.0 if getattr(job, "verification_id", "") else 0.0

        # Block E: final parent status (36-39)
        status = getattr(job, "status", "")
        vec[36] = 1.0 if status == "failed" else 0.0
        vec[37] = 1.0 if status == "cancelled" else 0.0
        vec[38] = 1.0 if status in {"completed", "deployed"} else 0.0
        vec[39] = _clamp((getattr(job, "completed_at", 0.0) or 0.0) > 0.0)

        return vec

    @staticmethod
    def encode_label(outcome: str) -> list[float]:
        label = [0.0] * len(LABEL_CLASSES)
        label[_LABEL_INDEX.get(outcome, 0)] = 1.0
        return label


def outcome_from_state(job: Any, verification: Any = None) -> str:
    lanes = getattr(job, "lanes", {}) or {}
    if getattr(lanes.get("planning"), "status", "") == "failed":
        return "planning_failed"
    if getattr(lanes.get("implementation"), "status", "") == "failed":
        return "implementation_failed"
    if verification is not None:
        risk = getattr(verification, "risk_assessment", {}) or {}
        if risk.get("skill_contract_status") in {"failed", "error", "missing_contract_or_fixture"}:
            return "contract_failed"
        if getattr(verification, "overall_passed", False):
            return "verified"
    if getattr(job, "status", "") in {"completed", "deployed"}:
        return "verified"
    return "blocked"

