"""AcquisitionPlanner — narrow-mandate plan synthesis.

The planner is ONLY allowed to produce:
  - Cross-lane ordering (which lanes run in what sequence)
  - Dependency edges (lane B requires artifact X from lane A)
  - Required artifact expectations (what each lane must produce)
  - Promotion criteria (what must be true to proceed to deployment)
  - Rollback coordination points (how to undo if things fail)

The planner MUST NOT decide the internal method of research, codegen,
self-improvement, or specialist training.  Those belong to the child lanes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from acquisition.job import (
    AcquisitionPlan,
    CapabilityAcquisitionJob,
    DocumentationArtifact,
    _artifact_id,
)

logger = logging.getLogger(__name__)


class AcquisitionPlanner:
    """Synthesizes execution plans for acquisition jobs."""

    def synthesize(
        self,
        job: CapabilityAcquisitionJob,
        evidence: list[dict[str, Any]] | None = None,
        doc_artifacts: list[DocumentationArtifact] | None = None,
    ) -> AcquisitionPlan:
        """Create a structured plan based on classification and evidence."""
        plan = AcquisitionPlan(
            acquisition_id=job.acquisition_id,
            objective=job.title,
            risk_level=self._risk_label(job.risk_tier),
        )

        docs = doc_artifacts or []
        plan.doc_artifact_ids = [d.artifact_id for d in docs]

        # Build lane ordering
        plan.implementation_path = self._build_implementation_path(job)
        plan.verification_path = self._build_verification_path(job)
        plan.rollback_path = self._build_rollback_path(job)
        plan.promotion_criteria = self._build_promotion_criteria(job)

        # Required artifacts based on outcome class
        plan.required_artifacts = self._determine_required_artifacts(job)
        plan.required_capabilities = self._determine_required_capabilities(job)

        return plan

    def _risk_label(self, tier: int) -> str:
        return {0: "low", 1: "medium", 2: "high", 3: "critical"}.get(tier, "unknown")

    def _build_implementation_path(self, job: CapabilityAcquisitionJob) -> list[dict[str, Any]]:
        """Ordered steps for implementation based on required lanes."""
        steps: list[dict[str, Any]] = []
        lane_order = [
            ("evidence_grounding", "Gather internal + external evidence"),
            ("doc_resolution", "Resolve documentation for critical dependencies"),
            ("planning", "Synthesize execution plan"),
            ("plan_review", "Human review of plan (if risk_tier >= 1)"),
            ("implementation", "Generate code via CodeGenService"),
            ("plugin_quarantine", "Deploy plugin to quarantine (not live)"),
            ("skill_registration", "Create LearningJob for skill tracking"),
            ("self_improve", "Apply core system modifications"),
            ("matrix_specialist", "Create Matrix specialist NN"),
            ("verification", "Run sandbox + baseline validation"),
            ("deployment", "Deploy to runtime (with supervision mode)"),
            ("plugin_activation", "Activate plugin from quarantine"),
            ("truth", "Record in attribution ledger + memory"),
        ]

        for lane_name, description in lane_order:
            if lane_name in job.required_lanes:
                steps.append({
                    "lane": lane_name,
                    "description": description,
                    "depends_on": self._get_dependencies(lane_name),
                })
        return steps

    def _get_dependencies(self, lane_name: str) -> list[str]:
        deps: dict[str, list[str]] = {
            "doc_resolution": ["evidence_grounding"],
            "planning": ["evidence_grounding", "doc_resolution"],
            "plan_review": ["planning"],
            "implementation": ["planning"],
            "plugin_quarantine": ["implementation"],
            "skill_registration": ["implementation"],
            "self_improve": ["implementation"],
            "matrix_specialist": ["evidence_grounding"],
            "verification": ["implementation"],
            "deployment": ["verification"],
            "plugin_activation": ["verification", "deployment"],
            "truth": [],
        }
        return deps.get(lane_name, [])

    def _build_verification_path(self, job: CapabilityAcquisitionJob) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        if job.outcome_class in ("plugin_creation", "core_upgrade", "skill_creation", "mixed"):
            steps.append({"type": "sandbox", "description": "AST + lint + pytest in sandbox"})
        if job.outcome_class in ("skill_creation", "mixed"):
            steps.append({"type": "baseline", "description": "Compare against SkillBaseline"})
        if job.outcome_class == "plugin_creation":
            steps.append({"type": "shadow", "description": "Shadow-mode invocations before activation"})
        return steps

    def _build_rollback_path(self, job: CapabilityAcquisitionJob) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        if job.outcome_class == "plugin_creation":
            steps.append({"type": "plugin_disable", "description": "Disable plugin, revert to prior version"})
        if job.outcome_class == "core_upgrade":
            steps.append({"type": "patch_restore", "description": "Restore pre-patch snapshot"})
        if job.outcome_class in ("skill_creation", "mixed"):
            steps.append({"type": "skill_unregister", "description": "Remove skill registration"})
        return steps

    def _build_promotion_criteria(self, job: CapabilityAcquisitionJob) -> list[str]:
        criteria: list[str] = []
        if job.outcome_class in ("plugin_creation", "core_upgrade", "skill_creation", "mixed"):
            criteria.append("All sandbox tests pass")
            criteria.append("No denied patterns detected")
        if job.risk_tier >= 1:
            criteria.append("Human deployment approval obtained")
        if job.outcome_class == "plugin_creation":
            criteria.append("Shadow mode: N successful calls without errors")
        return criteria

    def _determine_required_artifacts(self, job: CapabilityAcquisitionJob) -> list[str]:
        artifacts: list[str] = ["ResearchArtifact"]
        if job.outcome_class != "knowledge_only":
            artifacts.append("AcquisitionPlan")
        if job.outcome_class in ("plugin_creation", "core_upgrade", "skill_creation", "mixed"):
            artifacts.append("VerificationBundle")
        if job.outcome_class == "plugin_creation":
            artifacts.append("PluginArtifact")
        if job.outcome_class == "skill_creation":
            artifacts.append("SkillArtifact")
        if job.outcome_class == "core_upgrade":
            artifacts.append("UpgradeArtifact")
        artifacts.append("CapabilityClaim")
        artifacts.append("DeploymentRecord")
        return artifacts

    def _determine_required_capabilities(self, job: CapabilityAcquisitionJob) -> list[str]:
        caps: list[str] = []
        if job.outcome_class in ("plugin_creation", "core_upgrade", "skill_creation", "mixed"):
            caps.append("codegen")
            caps.append("sandbox")
        if job.outcome_class in ("plugin_creation", "mixed"):
            caps.append("plugin_registry")
        if job.outcome_class == "core_upgrade":
            caps.append("self_improve_orchestrator")
        return caps

    def get_status(self) -> dict[str, Any]:
        return {"available": True}
