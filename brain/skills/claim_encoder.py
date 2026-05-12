"""Claim handling class encoder for the CLAIM_CLASSIFIER hemisphere specialist.

Encodes a CapabilityGate claim evaluation context into a fixed 28-dim [0,1]
feature vector.  Also provides label encoding for the 8-class gate action
teacher signal.

This is a *gate action class predictor* -- the teacher label is "what action
did the gate take," not "is the claim true."  Friction/correction signals
later provide corrective supervision where the gate was too aggressive.

Dimension blocks:
  dims  0-3:  Text shape
  dims  4-7:  Pattern match
  dims  8-13: Signal flags
  dims 14-16: Context flags
  dims 17-19: Route context
  dims 20-22: Registry state
  dims 23-24: Evidence state
  dims 25-27: History
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 28

LABEL_CLASSES = (
    "conversational",
    "grounded",
    "verified",
    "preference",
    "reflective",
    "learning_acknowledged",
    "blocked",
    "rewritten",
)
NUM_CLASSES = len(LABEL_CLASSES)
_CLASS_INDEX = {name: i for i, name in enumerate(LABEL_CLASSES)}

_DECISION_TAG_TO_CLASS: dict[str, int] = {
    "conversational": 0,
    "subordinate-conversational": 0,
    "route-conversational": 0,
    "conversational-offer": 0,
    "grounded": 1,
    "verified": 2,
    "verified-context": 2,
    "verified-offer": 2,
    "preference": 3,
    "reflective": 4,
    "matrix:operational": 5,
    "matrix:limited": 5,
    "blocked": 6,
    "rewritten": 7,
}

_PATTERN_CATEGORIES = {
    "ability": 0,
    "intention": 1,
    "learning": 2,
}


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class ClaimClassifierEncoder:
    """Encodes a claim evaluation context into a 28-dim [0,1] feature vector."""

    @staticmethod
    def encode(context: dict[str, Any]) -> list[float]:
        """Produce 28-dim [0,1] feature vector from claim evaluation context.

        The context dict is populated by CapabilityGate._record_claim_signal()
        with fields from the evaluation state at decision time.
        """
        vec = [0.0] * FEATURE_DIM

        # Block 1: Text shape (dims 0-3)
        vec[0] = _clamp(context.get("token_count", 0) / 20.0)
        vec[1] = _clamp(context.get("char_len", 0) / 120.0)
        vec[2] = 1.0 if context.get("has_first_person", False) else 0.0
        vec[3] = 1.0 if context.get("has_we_pronoun", False) else 0.0

        # Block 2: Pattern match (dims 4-7)
        vec[4] = _clamp(context.get("claim_pattern_index", 0) / 15.0)
        vec[5] = 1.0 if context.get("is_readiness_frame", False) else 0.0
        cat = context.get("pattern_category", "")
        cat_idx = _PATTERN_CATEGORIES.get(cat, -1)
        vec[6] = 1.0 if cat_idx == 0 else 0.0  # ability
        vec[7] = 1.0 if cat_idx == 1 else 0.0  # intention

        # Block 3: Signal flags (dims 8-13)
        vec[8] = 1.0 if context.get("has_blocked_verb", False) else 0.0
        vec[9] = 1.0 if context.get("has_technical_signal", False) else 0.0
        vec[10] = 1.0 if context.get("has_internal_ops", False) else 0.0
        vec[11] = 1.0 if context.get("is_purely_conversational", False) else 0.0
        vec[12] = 1.0 if context.get("is_preference_alignment", False) else 0.0
        vec[13] = 1.0 if context.get("is_grounded_observation", False) else 0.0

        # Block 4: Context flags (dims 14-16)
        vec[14] = 1.0 if context.get("has_subordinate_context", False) else 0.0
        vec[15] = 1.0 if context.get("has_reflective_exclusion", False) else 0.0
        vec[16] = 1.0 if context.get("has_verified_skill_context", False) else 0.0

        # Block 5: Route context (dims 17-19)
        vec[17] = 1.0 if context.get("route_is_none", False) else 0.0
        vec[18] = 1.0 if context.get("route_is_strict", False) else 0.0
        vec[19] = 1.0 if context.get("status_mode", False) else 0.0

        # Block 6: Registry state (dims 20-22)
        vec[20] = 1.0 if context.get("registry_verified", False) else 0.0
        vec[21] = 1.0 if context.get("registry_learning", False) else 0.0
        vec[22] = 1.0 if context.get("registry_unknown", False) else 0.0

        # Block 7: Evidence state (dims 23-24)
        vec[23] = 1.0 if context.get("perception_evidence_fresh", False) else 0.0
        vec[24] = 1.0 if context.get("identity_confirmed", False) else 0.0

        # Block 8: History (dims 25-27)
        vec[25] = _clamp(context.get("family_block_count", 0) / 50.0)
        vec[26] = _clamp(context.get("session_block_count", 0) / 10.0)
        vec[27] = _clamp(context.get("time_since_last_block", 3600.0) / 3600.0)

        return vec

    @staticmethod
    def encode_label(decision_tag: str) -> tuple[list[float], dict[str, Any]]:
        """Encode an 8-class one-hot teacher label from a gate decision tag.

        Returns (label_vector, metadata_dict).  Decision tags come from
        CapabilityGate's internal annotation (e.g. "[conversational]",
        "[blocked]").  Tags are matched after stripping brackets.
        """
        tag = decision_tag.strip("[]").lower()
        label = [0.0] * NUM_CLASSES

        cls_idx = _DECISION_TAG_TO_CLASS.get(tag)
        if cls_idx is not None:
            label[cls_idx] = 1.0
        else:
            label[0] = 1.0

        metadata = {
            "decision_tag": tag,
            "class_index": cls_idx if cls_idx is not None else 0,
            "class_name": LABEL_CLASSES[cls_idx] if cls_idx is not None else "conversational",
        }
        return label, metadata

    @staticmethod
    def encode_correction_label() -> tuple[list[float], dict[str, Any]]:
        """Encode a corrective label when friction indicates the gate was too aggressive.

        Maps to class 0 (conversational) as a bootstrap approximation.
        Known simplification: some corrections belong to grounded/verified/preference.
        """
        label = [0.0] * NUM_CLASSES
        label[0] = 1.0
        metadata = {
            "decision_tag": "friction_correction",
            "class_index": 0,
            "class_name": "conversational",
        }
        return label, metadata
