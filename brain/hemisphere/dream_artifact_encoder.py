"""Dream artifact feature encoding for the DREAM_SYNTHESIS Tier-1 specialist.

This is a *validator-shadow approximator* — the teacher label is "what did the
ReflectiveValidator decide," not "what is the artifact worth."  The specialist
learns validator tendencies under governance conditions, not dream importance.

This encoder has NO import path to artifact state mutation, memory writes,
or event emission.  It produces float vectors and metadata dicts only.

Dimension blocks (16-dim total, all [0,1]):
  Block A (dims  0-7):  Artifact intrinsic properties
  Block B (dims  8-12): System state at validation time
  Block C (dims 13-15): Governance pressure context
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 16

LABEL_CLASSES = ["promoted", "held", "discarded", "quarantined"]
_LABEL_INDEX = {cls: i for i, cls in enumerate(LABEL_CLASSES)}

REASON_CATEGORIES = [
    "no_sources",
    "contradicts_beliefs",
    "informational_hold",
    "low_coherence",
    "low_confidence",
    "meets_thresholds",
    "borderline_hold",
    "promotion_cap",
]
_REASON_INDEX = {cat: i for i, cat in enumerate(REASON_CATEGORIES)}

_ARTIFACT_TYPE_ORDINALS = {
    "bridge_candidate": 0.0,
    "symbolic_summary": 0.2,
    "tension_flag": 0.4,
    "consolidation_proposal": 0.6,
    "waking_question": 0.8,
    "shadow_scenario": 1.0,
}

_INFORMATIONAL_TYPES = frozenset({"tension_flag", "waking_question"})


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _classify_reason(notes: str) -> str:
    """Map validator notes to a structured reason_category."""
    if not notes:
        return "borderline_hold"
    lower = notes.lower()
    if "no source" in lower:
        return "no_sources"
    if "contradict" in lower:
        return "contradicts_beliefs"
    if "informational" in lower:
        return "informational_hold"
    if "low coherence" in lower:
        return "low_coherence"
    if "low confidence" in lower:
        return "low_confidence"
    if "meets promotion" in lower or "promotion thresholds" in lower:
        return "meets_thresholds"
    if "promotion cap" in lower:
        return "promotion_cap"
    if "borderline" in lower:
        return "borderline_hold"
    return "borderline_hold"


class DreamArtifactEncoder:
    """Encodes dream artifact + system context into a 16-dim [0,1] feature vector.

    Block layout:
      Block A (dims  0-7):  Artifact intrinsic properties
      Block B (dims  8-12): System state at validation time
      Block C (dims 13-15): Governance pressure context

    Also provides label encoding for the 4-class validation-outcome
    teacher signal with structured reason metadata.
    """

    @staticmethod
    def encode_artifact_block(artifact: dict[str, Any]) -> list[float]:
        """Block A: artifact intrinsic properties (8 dims)."""
        vec = [0.0] * 8
        atype = artifact.get("artifact_type", "")
        vec[0] = _clamp(_ARTIFACT_TYPE_ORDINALS.get(atype, 0.5))
        vec[1] = _clamp(artifact.get("confidence", 0.0))
        vec[2] = _clamp(artifact.get("cluster_coherence", 0.0))

        source_ids = artifact.get("source_memory_ids", ())
        src_count = len(source_ids) if source_ids else 0
        vec[3] = _clamp(src_count / 10.0)
        vec[4] = _clamp(len(artifact.get("content", "")) / 500.0)
        vec[5] = 1.0 if src_count > 0 else 0.0
        vec[6] = 1.0 if atype in _INFORMATIONAL_TYPES else 0.0
        vec[7] = 1.0 if atype == "consolidation_proposal" else 0.0
        return vec

    @staticmethod
    def encode_system_block(ctx: dict[str, Any]) -> list[float]:
        """Block B: system state at validation time (5 dims)."""
        vec = [0.0] * 5
        vec[0] = _clamp(ctx.get("memory_density", 0.0))
        vec[1] = _clamp(ctx.get("dream_cycle_count", 0) / 1000.0)
        vec[2] = _clamp(ctx.get("awareness", 0.0))
        vec[3] = _clamp(ctx.get("belief_count", 0) / 1000.0)
        vec[4] = _clamp(ctx.get("contradiction_debt", 0.0))
        return vec

    @staticmethod
    def encode_governance_block(ctx: dict[str, Any]) -> list[float]:
        """Block C: governance pressure context (3 dims)."""
        vec = [0.0] * 3
        vec[0] = _clamp(ctx.get("soul_integrity", 1.0))
        vec[1] = _clamp(ctx.get("quarantine_pressure", 0.0))
        vec[2] = _clamp(ctx.get("promotion_rate_session", 0.0))
        return vec

    @staticmethod
    def encode(artifact: dict[str, Any], system_context: dict[str, Any]) -> list[float]:
        """Produce 16-dim [0,1] feature vector from artifact + system context."""
        block_a = DreamArtifactEncoder.encode_artifact_block(artifact)
        block_b = DreamArtifactEncoder.encode_system_block(system_context)
        block_c = DreamArtifactEncoder.encode_governance_block(system_context)
        vec = block_a + block_b + block_c
        assert len(vec) == FEATURE_DIM
        return vec

    @staticmethod
    def encode_label(
        validation_state: str,
        artifact: dict[str, Any],
        validator_notes: str = "",
    ) -> tuple[list[float], dict[str, Any]]:
        """Encode a 4-class teacher label + structured reason metadata.

        Returns (label_vector, metadata_dict).  The metadata preserves the
        validator's reasoning for future Phase 9.1 multi-head training.
        """
        label = [0.0] * len(LABEL_CLASSES)
        idx = _LABEL_INDEX.get(validation_state)
        if idx is not None:
            label[idx] = 1.0

        reason_category = _classify_reason(validator_notes)

        metadata = {
            "artifact_id": artifact.get("artifact_id", ""),
            "artifact_type": artifact.get("artifact_type", ""),
            "reason_category": reason_category,
            "validator_notes": validator_notes,
            "coherence_at_eval": artifact.get("cluster_coherence", 0.0),
            "confidence_at_eval": artifact.get("confidence", 0.0),
        }

        return label, metadata
