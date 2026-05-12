"""IntentionDeliveryEncoder — 24-dim feature vector for Stage 2 specialist.

Encodes a resolved intention + conversation context + system governance state
into a fixed-width feature vector. Used by:
  - IntentionResolver (Stage 1): heuristic scoring enrichment
  - intention_delivery specialist (Stage 2, inert): NN training data

Design doc: docs/INTENTION_STAGE_1_DESIGN.md §2

Block A — intention intrinsic (8 features)
Block B — conversation context (8 features)
Block C — system / governance (8 features)

Every feature is computable from existing subsystems — no new state required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 24

_COMMITMENT_TYPES = ["follow_up", "deferred_action", "future_work", "task_started"]
_COMMITMENT_TYPE_INDEX = {t: i for i, t in enumerate(_COMMITMENT_TYPES)}


@dataclass
class IntentionDeliveryFeatures:
    """Raw feature inputs before encoding."""

    commitment_type: str = "generic"
    outcome_success: bool = True
    age_since_resolution_s: float = 0.0
    expected_duration_s: float = 0.0
    result_summary_len: int = 0

    turn_idx_same_speaker_since_commit: int = 0
    time_since_last_user_message_s: float = 0.0
    same_topic_keyword_overlap: float = 0.0
    speaker_present_now: bool = False
    addressee_mode_score: float = 0.0
    active_conversation: bool = False
    user_mood_valence: float = 0.0
    proactive_cooldown_remaining_norm: float = 0.0

    quarantine_pressure: float = 0.0
    contradiction_debt: float = 0.0
    soul_integrity: float = 1.0
    belief_graph_integrity: float = 1.0
    autonomy_level: int = 0
    is_interruptible_mode: bool = True
    is_reflective_mode: bool = False
    recent_user_friction_rate: float = 0.0
    fractal_recall_surfaced_recent: bool = False


def encode(features: IntentionDeliveryFeatures) -> list[float]:
    """Encode features into a 24-dim float vector in [0, 1]."""
    vec: list[float] = []

    # Block A — intention intrinsic (8 dims)
    onehot = [0.0] * 4
    idx = _COMMITMENT_TYPE_INDEX.get(features.commitment_type, -1)
    if 0 <= idx < 4:
        onehot[idx] = 1.0
    vec.extend(onehot)
    vec.append(1.0 if features.outcome_success else 0.0)
    vec.append(min(1.0, math.log1p(features.age_since_resolution_s) / math.log1p(86400)))
    expected_ratio = 0.0
    if features.expected_duration_s > 0:
        expected_ratio = min(4.0, features.age_since_resolution_s / features.expected_duration_s) / 4.0
    vec.append(expected_ratio)
    vec.append(min(1.0, features.result_summary_len / 500.0))

    # Block B — conversation context (8 dims)
    vec.append(min(1.0, features.turn_idx_same_speaker_since_commit / 20.0))
    vec.append(min(1.0, math.log1p(features.time_since_last_user_message_s) / math.log1p(3600)))
    vec.append(max(0.0, min(1.0, features.same_topic_keyword_overlap)))
    vec.append(1.0 if features.speaker_present_now else 0.0)
    vec.append(max(0.0, min(1.0, features.addressee_mode_score)))
    vec.append(1.0 if features.active_conversation else 0.0)
    vec.append(max(0.0, min(1.0, (features.user_mood_valence + 1.0) / 2.0)))
    vec.append(max(0.0, min(1.0, features.proactive_cooldown_remaining_norm)))

    # Block C — system / governance (8 dims)
    vec.append(max(0.0, min(1.0, features.quarantine_pressure)))
    vec.append(max(0.0, min(1.0, features.contradiction_debt)))
    vec.append(max(0.0, min(1.0, features.soul_integrity)))
    vec.append(max(0.0, min(1.0, features.belief_graph_integrity)))
    vec.append(min(1.0, features.autonomy_level / 3.0))
    vec.append(1.0 if features.is_interruptible_mode else 0.0)
    vec.append(1.0 if features.is_reflective_mode else 0.0)
    vec.append(max(0.0, min(1.0, features.recent_user_friction_rate / 0.5)))

    assert len(vec) == FEATURE_DIM, f"Expected {FEATURE_DIM} dims, got {len(vec)}"
    return vec


def encode_label(decision: str) -> list[float]:
    """Encode a 4-class label from a resolver decision.

    Label space: deliver_accepted, deliver_rejected,
    suppressed_was_right, suppressed_was_wrong.

    In Stage 1 (shadow-only), we approximate with the heuristic decision:
      deliver_now/deliver_on_next_turn → deliver_accepted (index 0)
      suppress → suppressed_was_right (index 2)
      defer → uniform distribution (no signal)
    """
    label = [0.0, 0.0, 0.0, 0.0]
    if decision in ("deliver_now", "deliver_on_next_turn"):
        label[0] = 1.0
    elif decision == "suppress":
        label[2] = 1.0
    else:
        label = [0.25, 0.25, 0.25, 0.25]
    return label


__all__ = [
    "IntentionDeliveryFeatures",
    "encode",
    "encode_label",
    "FEATURE_DIM",
]
