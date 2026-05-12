"""Plan Evaluator feature encoding, verdict labels, and shadow prediction artifacts.

Encodes acquisition plan + job metadata into a fixed 32-dim [0,1] vector for
the plan_evaluator hemisphere specialist.  Emphasis on relational quality over
raw text length.
"""

from __future__ import annotations

import enum
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 32

# ---------------------------------------------------------------------------
# Outcome-class one-hot mapping (7 classes → dims 0-6)
# ---------------------------------------------------------------------------

_OUTCOME_CLASSES = [
    "knowledge_only",
    "skill_creation",
    "plugin_creation",
    "core_upgrade",
    "specialist_nn",
    "hardware_integration",
    "mixed",
]
_OUTCOME_INDEX = {cls: i for i, cls in enumerate(_OUTCOME_CLASSES)}

# Regex for detecting measurable promotion criteria (contains numbers/thresholds)
_MEASURABLE_RE = re.compile(r"\d+|>=|<=|>|<|percent|threshold|accuracy|rate", re.IGNORECASE)

# Design sections expected from the coder LLM
_DESIGN_SECTIONS = ("user_story", "technical_approach", "implementation_sketch",
                    "dependencies", "test_cases", "risk_analysis")


# ---------------------------------------------------------------------------
# VerdictReasonCategory — recorded alongside every review verdict
# ---------------------------------------------------------------------------


class VerdictReasonCategory(str, enum.Enum):
    TECHNICAL_WEAKNESS = "technical_weakness"
    STALE_DOCS = "stale_docs"
    WRONG_LANE_CHOICE = "wrong_lane_choice"
    POLICY_SAFETY = "policy_safety"
    UNNECESSARY_DUPLICATION = "unnecessary_duplication"
    PREFERENCE_STYLE = "preference_style"
    MISSING_EVIDENCE = "missing_evidence"
    PLAN_QUALITY_OK = "plan_quality_ok"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ShadowPredictionArtifact — persisted shadow NN prediction per plan review
# ---------------------------------------------------------------------------


def _shadow_id() -> str:
    return f"shd_{uuid.uuid4().hex[:10]}"


@dataclass
class ShadowPredictionArtifact:
    """Persisted shadow prediction tied to a specific plan version.

    Never shown to the human, never gates anything.  Exists purely for
    retrospective accuracy measurement and debugging.
    """
    shadow_id: str = field(default_factory=_shadow_id)
    acquisition_id: str = ""
    plan_id: str = ""
    plan_version: int = 0
    predicted_probs: list[float] = field(default_factory=list)  # [approve, reject, revise]
    predicted_class: str = ""   # "approved" | "rejected" | "needs_revision"
    actual_verdict: str = ""    # filled after human review
    actual_class: str = ""      # filled after human review
    reason_category: str = ""   # VerdictReasonCategory value, filled after review
    correct: bool | None = None
    predicted_at: float = field(default_factory=time.time)
    reviewed_at: float = 0.0
    feature_vector: list[float] = field(default_factory=list)
    model_version: str = ""     # hemisphere registry version id of specialist used
    risk_tier: int = 0          # from the job, for per-tier accuracy analysis
    outcome_class: str = ""     # from the job, for per-class accuracy analysis

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShadowPredictionArtifact:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Verdict label encoding
# ---------------------------------------------------------------------------

_VERDICT_TO_CLASS = {
    "approved_as_is": "approved",
    "approved_with_edits": "approved",
    "rejected": "rejected",
    "cancelled": "rejected",
    "convert_to_research_only": "needs_revision",
    "route_to_self_improve": "needs_revision",
    "route_to_plugin_upgrade": "needs_revision",
    "use_existing_capability": "needs_revision",
}

_CLASS_TO_LABEL = {
    "approved": [1.0, 0.0, 0.0],
    "rejected": [0.0, 1.0, 0.0],
    "needs_revision": [0.0, 0.0, 1.0],
}


def encode_verdict(verdict: str) -> list[float]:
    """Convert a verdict string into a 3-class softmax label."""
    cls = _VERDICT_TO_CLASS.get(verdict, "needs_revision")
    return list(_CLASS_TO_LABEL[cls])


def verdict_to_class(verdict: str) -> str:
    """Map a raw verdict string to one of three canonical classes."""
    return _VERDICT_TO_CLASS.get(verdict, "needs_revision")


def label_to_class(probs: list[float]) -> str:
    """Convert 3-class probability vector to the predicted class name."""
    if len(probs) < 3:
        return "needs_revision"
    idx = max(range(3), key=lambda i: probs[i])
    return ["approved", "rejected", "needs_revision"][idx]


# ---------------------------------------------------------------------------
# PlanEvaluatorEncoder
# ---------------------------------------------------------------------------


class PlanEvaluatorEncoder:
    """Encodes a (job, plan) pair into a fixed 32-dim [0,1] feature vector.

    Block layout:
      dims  0-8:  Classification (outcome one-hot 7 + risk + confidence)
      dims  9-15: Plan structure
      dims 16-23: Relational quality
      dims 24-27: Text richness
      dims 28-31: Evidence coverage
    """

    @staticmethod
    def encode(job: Any, plan: Any) -> list[float]:
        """Produce 32-dim [0,1] feature vector from job + plan metadata."""
        vec = [0.0] * FEATURE_DIM

        # ── Classification block (dims 0-8) ───────────────────────────
        oc = getattr(job, "outcome_class", "knowledge_only")
        idx = _OUTCOME_INDEX.get(oc, 0)
        vec[idx] = 1.0
        vec[7] = _clamp(getattr(job, "risk_tier", 0) / 3.0)
        vec[8] = _clamp(getattr(job, "classification_confidence", 0.0))

        # ── Plan structure block (dims 9-15) ──────────────────────────
        impl_steps = len(getattr(plan, "implementation_path", []))
        verif_steps = len(getattr(plan, "verification_path", []))
        has_rollback = 1.0 if getattr(plan, "rollback_path", []) else 0.0
        n_caps = len(getattr(plan, "required_capabilities", []))
        n_arts = len(getattr(plan, "required_artifacts", []))
        n_promo = len(getattr(plan, "promotion_criteria", []))
        version = getattr(plan, "version", 1)

        vec[9] = _clamp(impl_steps / 20.0)
        vec[10] = _clamp(verif_steps / 10.0)
        vec[11] = has_rollback
        vec[12] = _clamp(n_caps / 10.0)
        vec[13] = _clamp(n_arts / 10.0)
        vec[14] = _clamp(n_promo / 10.0)
        vec[15] = _clamp(version / 5.0)

        # ── Relational quality block (dims 16-23) ─────────────────────
        section_flags = [
            1.0 if getattr(plan, attr, "") else 0.0
            for attr in ("user_story", "technical_approach", "implementation_sketch")
        ]
        deps_list = getattr(plan, "dependencies", []) or []
        tests_list = getattr(plan, "test_cases", []) or []
        risk_text = getattr(plan, "risk_analysis", "") or ""
        section_flags.extend([
            1.0 if deps_list else 0.0,
            1.0 if tests_list else 0.0,
            1.0 if risk_text else 0.0,
        ])
        vec[16] = 1.0 if all(f > 0 for f in section_flags) else 0.0

        caps_set = set(getattr(plan, "required_capabilities", []) or [])
        deps_set = set(d.lower() for d in deps_list)
        if caps_set:
            alignment = len(deps_set & caps_set) / len(caps_set)
        else:
            alignment = 1.0 if not deps_set else 0.0
        vec[17] = _clamp(alignment)

        doc_ids = set(getattr(plan, "doc_artifact_ids", []) or [])
        if impl_steps > 0:
            coverage = min(len(doc_ids) / max(1, impl_steps), 1.0)
        else:
            coverage = 1.0 if doc_ids else 0.0
        vec[18] = _clamp(coverage)

        verif_path = getattr(plan, "verification_path", []) or []
        verif_refs_impl = 0.0
        for v in verif_path:
            refs = str(v) if isinstance(v, dict) else ""
            if "implementation" in refs.lower() or "sandbox" in refs.lower():
                verif_refs_impl = 1.0
                break
        vec[19] = verif_refs_impl

        required_lanes = getattr(job, "required_lanes", []) or []
        has_deployment_lane = "deployment" in required_lanes
        vec[20] = 1.0 if (has_deployment_lane and has_rollback > 0) or not has_deployment_lane else 0.0

        vec[21] = _clamp(len(tests_list) / max(1, impl_steps)) if impl_steps > 0 else _clamp(len(tests_list) / 5.0)
        vec[22] = _clamp(len(deps_list) / 20.0)
        vec[23] = _clamp(len(tests_list) / 20.0)

        # ── Text richness block (dims 24-27) ──────────────────────────
        design_completeness = sum(section_flags) / len(section_flags) if section_flags else 0.0
        vec[24] = _clamp(design_completeness)

        total_chars = sum(
            len(str(getattr(plan, attr, "") or ""))
            for attr in _DESIGN_SECTIONS
        )
        vec[25] = _clamp(total_chars / 15000.0)

        promo_criteria = getattr(plan, "promotion_criteria", []) or []
        if promo_criteria:
            measurable = sum(1 for c in promo_criteria if _MEASURABLE_RE.search(str(c)))
            vec[26] = _clamp(measurable / len(promo_criteria))
        else:
            vec[26] = 0.0

        risk_specific = 0.0
        if risk_text:
            lower_risk = risk_text.lower()
            if any(w in lower_risk for w in ("mitigat", "fallback", "rollback", "if ", "when ")):
                risk_specific = 1.0
            elif len(risk_text) > 50:
                risk_specific = 0.5
        vec[27] = _clamp(risk_specific)

        # ── Evidence block (dims 28-31) ───────────────────────────────
        job_doc_ids = getattr(job, "doc_artifact_ids", []) or []
        vec[28] = _clamp(len(job_doc_ids) / 10.0)

        vec[29] = 0.0  # avg freshness — populated by caller if doc objects available

        vec[30] = 0.0  # num research sources — populated by caller if available

        req_artifacts = getattr(plan, "required_artifacts", []) or []
        if req_artifacts:
            available = len(set(job_doc_ids) | set(getattr(job, "artifact_refs", []) or []))
            vec[31] = _clamp(available / len(req_artifacts))
        else:
            vec[31] = 1.0

        return vec

    @staticmethod
    def enrich_evidence_dims(
        vec: list[float],
        doc_artifacts: list[Any] | None = None,
        research_sources: int = 0,
    ) -> list[float]:
        """Fill evidence dims (29-30) that require loaded artifacts."""
        if doc_artifacts:
            freshness_scores = [
                getattr(d, "freshness_score", 0.0) for d in doc_artifacts
                if hasattr(d, "freshness_score")
            ]
            if freshness_scores:
                vec[29] = _clamp(sum(freshness_scores) / len(freshness_scores))
        vec[30] = _clamp(research_sources / 10.0)
        return vec


def _clamp(v: float) -> float:
    """Clamp to [0, 1]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
