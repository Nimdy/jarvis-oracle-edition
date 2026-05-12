"""Capability Acquisition Pipeline — data model and persistence.

CapabilityAcquisitionJob tracks lifecycle state only.  Plans, verification results,
and deployment records are standalone artifact objects that the job references but
does not contain.  This prevents the parent object from becoming bloated and
mutation-heavy as plans change, execution retries, and evidence accumulates.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AcquisitionStatus = Literal[
    "pending",
    "classifying",
    "planning",
    "awaiting_plan_review",
    "executing",
    "verifying",
    "awaiting_approval",
    "deployed",
    "completed",
    "failed",
    "cancelled",
]

OutcomeClass = Literal[
    "knowledge_only",
    "skill_creation",
    "plugin_creation",
    "core_upgrade",
    "specialist_nn",
    "hardware_integration",
    "mixed",
]

# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

def _acq_id() -> str:
    return f"acq_{uuid.uuid4().hex[:10]}"


def _lane_run_id(lane_name: str) -> str:
    return f"lane_{lane_name}_{uuid.uuid4().hex[:8]}"


def _artifact_id() -> str:
    return f"art_{uuid.uuid4().hex[:10]}"


def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Lane state (thin — execution semantics stay in subsystems)
# ---------------------------------------------------------------------------

@dataclass
class LaneState:
    status: str = "pending"         # pending | running | completed | failed | skipped
    lane_run_id: str = ""
    child_id: str = ""              # native ID in child subsystem
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LaneState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Parent lifecycle object
# ---------------------------------------------------------------------------

@dataclass
class CapabilityAcquisitionJob:
    acquisition_id: str = field(default_factory=_acq_id)
    title: str = ""
    status: AcquisitionStatus = "pending"
    outcome_class: OutcomeClass = "knowledge_only"

    # Intent classification
    user_intent: str = ""
    classified_at: float = 0.0
    classification_confidence: float = 0.0
    required_lanes: list[str] = field(default_factory=list)

    # Lane states
    lanes: dict[str, LaneState] = field(default_factory=dict)

    # Child artifact IDs (forward refs to existing subsystems)
    research_intent_id: str = ""
    learning_job_id: str = ""
    improvement_request_id: str = ""
    matrix_specialist_id: str = ""
    goal_id: str = ""
    plugin_id: str = ""

    # Artifact references (standalone persisted objects)
    plan_id: str = ""
    plan_review_id: str = ""
    doc_artifact_ids: list[str] = field(default_factory=list)
    code_bundle_id: str = ""
    environment_setup_id: str = ""
    verification_id: str = ""
    claim_id: str = ""
    deployment_id: str = ""

    # Truth spine
    ledger_entry_id: str = ""
    artifact_refs: list[str] = field(default_factory=list)

    # Permission model (3-concept)
    risk_tier: int = 0
    review_status: str = "not_required"
    approval_status: str = "not_required"
    supervision_mode: str = "none"
    approved_by: str = ""
    approval_timestamp: float = 0.0

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    requested_by: dict[str, Any] = field(default_factory=dict)
    planning_diagnostics: dict[str, Any] = field(default_factory=dict)
    codegen_prompt_diagnostics: dict[str, Any] = field(default_factory=dict)
    activation_diagnostics: dict[str, Any] = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────────────

    def touch(self) -> None:
        self.updated_at = time.time()

    def set_status(self, status: AcquisitionStatus) -> None:
        self.status = status
        self.touch()
        if status in ("completed", "failed", "cancelled") and not self.completed_at:
            self.completed_at = time.time()

    def init_lane(self, lane_name: str) -> LaneState:
        ls = LaneState(
            status="pending",
            lane_run_id=_lane_run_id(lane_name),
        )
        self.lanes[lane_name] = ls
        self.touch()
        return ls

    def start_lane(self, lane_name: str) -> LaneState:
        ls = self.lanes.get(lane_name)
        if ls is None:
            ls = self.init_lane(lane_name)
        ls.status = "running"
        ls.started_at = time.time()
        ls.completed_at = 0.0
        ls.error = ""
        self.touch()
        return ls

    def complete_lane(self, lane_name: str, child_id: str = "") -> None:
        ls = self.lanes.get(lane_name)
        if ls is None:
            return
        ls.status = "completed"
        ls.completed_at = time.time()
        if child_id:
            ls.child_id = child_id
        self.touch()

    def fail_lane(self, lane_name: str, error: str = "") -> None:
        ls = self.lanes.get(lane_name)
        if ls is None:
            return
        ls.status = "failed"
        ls.completed_at = time.time()
        ls.error = error
        ls.retry_count += 1
        self.touch()

    def add_artifact_ref(self, artifact_id: str) -> None:
        if artifact_id and artifact_id not in self.artifact_refs:
            self.artifact_refs.append(artifact_id)
            self.touch()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lanes"] = {k: v.to_dict() if isinstance(v, LaneState) else v
                      for k, v in self.lanes.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilityAcquisitionJob:
        lanes_raw = d.pop("lanes", {})
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        job = cls(**filtered)
        job.lanes = {k: LaneState.from_dict(v) if isinstance(v, dict) else v
                     for k, v in lanes_raw.items()}
        return job


# ---------------------------------------------------------------------------
# Artifact types — standalone persisted objects, referenced by ID
# ---------------------------------------------------------------------------

@dataclass
class ResearchArtifact:
    """Output of the evidence grounding lane."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    existing_capabilities: list[str] = field(default_factory=list)
    prior_attempts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DocumentationArtifact:
    """Versioned, source-typed documentation evidence for a capability.

    The freshness bridge — LLM training data may be stale on package APIs,
    framework changes, vendor SDKs, deprecations.
    """
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    source_type: str = ""           # official_docs | mcp_context7 | local_doc | repo_doc | web_search | uploaded_manual | pdf
    topic: str = ""
    version_scope: str = ""
    retrieved_at: float = field(default_factory=time.time)
    relevance: float = 0.0
    citations: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    freshness_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DocumentationArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AcquisitionPlan:
    """Structured execution plan — standalone, versioned, not embedded in job."""
    plan_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    objective: str = ""
    required_artifacts: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    risk_level: str = "low"
    implementation_path: list[dict[str, Any]] = field(default_factory=list)
    verification_path: list[dict[str, Any]] = field(default_factory=list)
    rollback_path: list[dict[str, Any]] = field(default_factory=list)
    promotion_criteria: list[str] = field(default_factory=list)
    version: int = 1
    doc_artifact_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # Technical design (produced by coder LLM during planning lane)
    user_story: str = ""
    technical_approach: str = ""
    implementation_sketch: str = ""
    dependencies: list[str] = field(default_factory=list)
    test_cases: list[str] = field(default_factory=list)
    risk_analysis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AcquisitionPlan:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlanReviewArtifact:
    """Human feedback on an AcquisitionPlan before implementation begins.  Immutable."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    plan_id: str = ""
    verdict: str = ""               # approved_as_is | approved_with_edits | rejected | convert_to_research_only | route_to_self_improve | route_to_plugin_upgrade | use_existing_capability
    reason_category: str = "unknown"  # VerdictReasonCategory value — recorded for future NN label richness
    operator_notes: str = ""
    suggested_changes: list[dict[str, Any]] = field(default_factory=list)
    reviewed_at: float = field(default_factory=time.time)
    reviewed_by: str = "human"
    plan_version: int = 0            # version of the plan that was reviewed — pairing key for distillation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlanReviewArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PluginCodeBundle:
    """Explicit artifact contract for implementation→quarantine handoff.

    The implementation lane produces this bundle; the quarantine lane consumes it.
    Persisted as a standalone artifact for audit trail.
    """
    bundle_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    code_files: dict[str, str] = field(default_factory=dict)
    manifest_candidate: dict[str, Any] = field(default_factory=dict)
    code_hash: str = ""
    source_plan_id: str = ""
    doc_artifact_ids: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PluginCodeBundle:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def compute_hash(self) -> str:
        import hashlib
        raw = json.dumps(self.code_files, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class EnvironmentSetupArtifact:
    """Output of the environment_setup lane — records venv creation and dep install."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    plugin_name: str = ""
    execution_mode: str = "in_process"
    venv_path: str = ""
    pinned_dependencies: list[str] = field(default_factory=list)
    installed_packages: list[str] = field(default_factory=list)
    install_log: str = ""
    import_verification_passed: bool = False
    skipped: bool = False
    skip_reason: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentSetupArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PluginArtifact:
    """Generated tool/service code + manifest."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    plugin_name: str = ""
    code_hash: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    sandbox_result: dict[str, Any] = field(default_factory=dict)
    deployment_state: str = "quarantined"   # quarantined | shadow | supervised | active | disabled
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationBundle:
    """Cross-reference index to lane-native verification artifacts.

    The bundle references and aggregates, but does NOT restate lane-native
    authority.  The sandbox owns sandbox truth.  The skill baseline owns skill
    validation truth.  This is a cross-reference index, not a replacement.
    """
    verification_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    sandbox_result_ref: str = ""
    test_result_refs: list[str] = field(default_factory=list)
    baseline_ref: str = ""
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    lane_verdicts: dict[str, bool] = field(default_factory=dict)
    overall_passed: bool = False
    verified_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerificationBundle:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SkillArtifact:
    """Reference to a registered skill produced by this acquisition."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    skill_id: str = ""
    learning_job_id: str = ""
    verification_evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpgradeArtifact:
    """Reference to a core system change produced by this acquisition."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    improvement_request_id: str = ""
    patch_plan_id: str = ""
    sandbox_passed: bool = False
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityClaim:
    """What Jarvis is now allowed to claim/use after this acquisition.  Immutable."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    claim_type: str = ""            # skill | plugin | knowledge | specialist
    claim_id: str = ""
    evidence_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentRecord:
    """Truth record of what was deployed, when, by whom.  Immutable."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    deployed_type: str = ""         # plugin | skill | upgrade | specialist
    deployed_id: str = ""
    deployed_at: float = field(default_factory=time.time)
    supervision_mode: str = "none"
    rollback_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PluginUpgradeArtifact:
    """Truth record of a plugin upgrade attempt.  Immutable."""
    artifact_id: str = field(default_factory=_artifact_id)
    acquisition_id: str = ""
    plugin_name: str = ""
    previous_version: str = ""
    candidate_version: str = ""
    triggering_reason: str = ""     # stale_dependency | failure_rate | doc_update | operator_request | scheduled
    doc_artifact_ids: list[str] = field(default_factory=list)
    benchmark_delta: dict[str, Any] = field(default_factory=dict)
    shadow_comparison: dict[str, Any] = field(default_factory=dict)
    promotion_decision: str = "pending"  # promoted | rolled_back | pending
    decided_at: float = 0.0
    rollback_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Persistence — AcquisitionStore
# ---------------------------------------------------------------------------

_JARVIS_DIR = Path.home() / ".jarvis"

_SUBDIRS = {
    "jobs": "acquisitions",
    "plans": "acquisition_plans",
    "reviews": "acquisition_reviews",
    "research": "acquisition_research",
    "docs": "acquisition_docs",
    "code_bundles": "acquisition_code_bundles",
    "environment_setups": "acquisition_environment_setups",
    "verifications": "acquisition_verifications",
    "claims": "acquisition_claims",
    "deployments": "acquisition_deployments",
    "plugin_upgrades": "plugin_upgrades",
}


class AcquisitionStore:
    """Per-entity JSON persistence for acquisition jobs and artifacts.

    Uses ``atomic_write_json`` for crash safety.  Eager save on every mutation.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or _JARVIS_DIR
        for subdir in _SUBDIRS.values():
            (self._base / subdir).mkdir(parents=True, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────

    def _path(self, category: str, obj_id: str) -> Path:
        return self._base / _SUBDIRS[category] / f"{obj_id}.json"

    def _save(self, category: str, obj_id: str, data: dict[str, Any]) -> None:
        from memory.persistence import atomic_write_json
        atomic_write_json(self._path(category, obj_id), data)

    def _load(self, category: str, obj_id: str) -> dict[str, Any] | None:
        p = self._path(category, obj_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            logger.warning("Failed to load %s/%s", category, obj_id)
            return None

    # ── job CRUD ───────────────────────────────────────────────────────

    def save_job(self, job: CapabilityAcquisitionJob) -> None:
        self._save("jobs", job.acquisition_id, job.to_dict())

    def load_job(self, acquisition_id: str) -> CapabilityAcquisitionJob | None:
        d = self._load("jobs", acquisition_id)
        if d is None:
            return None
        return CapabilityAcquisitionJob.from_dict(d)

    def list_jobs(self, status: str | None = None) -> list[CapabilityAcquisitionJob]:
        """List all persisted jobs, optionally filtered by status."""
        jobs_dir = self._base / _SUBDIRS["jobs"]
        result: list[CapabilityAcquisitionJob] = []
        if not jobs_dir.exists():
            return result
        for p in jobs_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
                if status and d.get("status") != status:
                    continue
                result.append(CapabilityAcquisitionJob.from_dict(d))
            except Exception:
                logger.warning("Failed to load job from %s", p)
        result.sort(key=lambda j: j.created_at, reverse=True)
        return result

    # ── plan CRUD ──────────────────────────────────────────────────────

    def save_plan(self, plan: AcquisitionPlan) -> None:
        self._save("plans", plan.plan_id, plan.to_dict())

    def load_plan(self, plan_id: str) -> AcquisitionPlan | None:
        d = self._load("plans", plan_id)
        return AcquisitionPlan.from_dict(d) if d else None

    # ── review CRUD ────────────────────────────────────────────────────

    def save_review(self, review: PlanReviewArtifact) -> None:
        self._save("reviews", review.artifact_id, review.to_dict())

    def load_review(self, artifact_id: str) -> PlanReviewArtifact | None:
        d = self._load("reviews", artifact_id)
        return PlanReviewArtifact.from_dict(d) if d else None

    # ── research artifact CRUD ──────────────────────────────────────────

    def save_research(self, research: ResearchArtifact) -> None:
        self._save("research", research.artifact_id, research.to_dict())

    def load_research(self, artifact_id: str) -> ResearchArtifact | None:
        d = self._load("research", artifact_id)
        return ResearchArtifact.from_dict(d) if d else None

    # ── documentation artifact CRUD ────────────────────────────────────

    def save_doc(self, doc: DocumentationArtifact) -> None:
        self._save("docs", doc.artifact_id, doc.to_dict())

    def load_doc(self, artifact_id: str) -> DocumentationArtifact | None:
        d = self._load("docs", artifact_id)
        return DocumentationArtifact.from_dict(d) if d else None

    # ── code bundle CRUD ────────────────────────────────────────────────

    def save_code_bundle(self, bundle: PluginCodeBundle) -> None:
        self._save("code_bundles", bundle.bundle_id, bundle.to_dict())

    def load_code_bundle(self, bundle_id: str) -> PluginCodeBundle | None:
        d = self._load("code_bundles", bundle_id)
        return PluginCodeBundle.from_dict(d) if d else None

    # ── environment setup CRUD ────────────────────────────────────────

    def save_environment_setup(self, artifact: EnvironmentSetupArtifact) -> None:
        self._save("environment_setups", artifact.artifact_id, artifact.to_dict())

    def load_environment_setup(self, artifact_id: str) -> EnvironmentSetupArtifact | None:
        d = self._load("environment_setups", artifact_id)
        return EnvironmentSetupArtifact.from_dict(d) if d else None

    # ── verification bundle CRUD ───────────────────────────────────────

    def save_verification(self, bundle: VerificationBundle) -> None:
        self._save("verifications", bundle.verification_id, bundle.to_dict())

    def load_verification(self, verification_id: str) -> VerificationBundle | None:
        d = self._load("verifications", verification_id)
        return VerificationBundle.from_dict(d) if d else None

    # ── claim CRUD ─────────────────────────────────────────────────────

    def save_claim(self, claim: CapabilityClaim) -> None:
        self._save("claims", claim.artifact_id, claim.to_dict())

    # ── deployment CRUD ────────────────────────────────────────────────

    def save_deployment(self, record: DeploymentRecord) -> None:
        self._save("deployments", record.artifact_id, record.to_dict())

    # ── plugin upgrade CRUD ────────────────────────────────────────────

    def save_plugin_upgrade(self, upgrade: PluginUpgradeArtifact) -> None:
        self._save("plugin_upgrades", upgrade.artifact_id, upgrade.to_dict())

    # ── summary for dashboard ──────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Quick summary for snapshot/dashboard."""
        all_jobs = self.list_jobs()
        active = [j for j in all_jobs if j.status not in ("completed", "failed", "cancelled")]
        completed = [j for j in all_jobs if j.status == "completed"]
        failed = [j for j in all_jobs if j.status == "failed"]

        recent = []
        for j in all_jobs[:10]:
            entry: dict[str, Any] = {
                "acquisition_id": j.acquisition_id,
                "title": j.title,
                "status": j.status,
                "outcome_class": j.outcome_class,
                "risk_tier": j.risk_tier,
                "lanes": {},
                "created_at": j.created_at,
                "plan_id": j.plan_id,
                "code_bundle_id": j.code_bundle_id,
                "verification_id": j.verification_id,
                "plugin_id": j.plugin_id,
                "planning_diagnostics": j.planning_diagnostics or {},
                "codegen_prompt_diagnostics": j.codegen_prompt_diagnostics or {},
            }
            for k, v in j.lanes.items():
                entry["lanes"][k] = {
                    "status": v.status,
                    "child_id": v.child_id,
                    "error": v.error or "",
                }
            if j.plan_id:
                plan = self.load_plan(j.plan_id)
                if plan:
                    entry["plan_summary"] = {
                        "objective": plan.objective,
                        "technical_approach": plan.technical_approach or "",
                        "risk_analysis": plan.risk_analysis or "",
                        "risk_level": plan.risk_level or "",
                        "dependencies": plan.dependencies or [],
                        "test_cases": plan.test_cases or [],
                        "implementation_sketch": plan.implementation_sketch or "",
                    }
            if j.code_bundle_id:
                bundle = self.load_code_bundle(j.code_bundle_id)
                if bundle:
                    entry["code_bundle_summary"] = {
                        "bundle_id": bundle.bundle_id,
                        "code_files": list((bundle.code_files or {}).keys()),
                        "code_hash": bundle.code_hash or "",
                    }
            if j.verification_id:
                vb = self.load_verification(j.verification_id)
                if vb:
                    entry["verification_summary"] = {
                        "overall_passed": vb.overall_passed,
                        "lane_verdicts": vb.lane_verdicts or {},
                    }
            recent.append(entry)

        return {
            "active_count": len(active),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "total_count": len(all_jobs),
            "recent": recent,
        }
