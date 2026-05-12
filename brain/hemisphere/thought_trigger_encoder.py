"""Thought trigger selection encoder for the THOUGHT_TRIGGER_SELECTOR specialist.

Shadow-only Tier-1 specialist that learns which meta-cognitive thought triggers
produce useful downstream outcomes (autonomy research success, positive policy
reward, world model prediction) given the current system state.

This encoder has NO import path to thought generation, mutation, memory writes,
or event emission.  It produces float vectors and metadata dicts only.

Dimension blocks (44-dim total, all [0,1]):
  Block A (dims  0-19): System state (from PolicyStateEncoder dims)
  Block B (dims 20-31): Trigger type one-hot (12 trigger types)
  Block C (dims 32-43): Time-since-last-fire per trigger type, normalized
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 44
NUM_TRIGGER_TYPES = 12

TRIGGER_TYPES = [
    "self_observation",
    "pattern_recognition",
    "uncertainty_acknowledgment",
    "causal_reflection",
    "consciousness_questioning",
    "memory_reflection",
    "pattern_synthesis",
    "existential_wonder",
    "emotional_awareness",
    "growth_recognition",
    "connection_discovery",
    "temporal_reflection",
]
_TRIGGER_INDEX = {name: i for i, name in enumerate(TRIGGER_TYPES)}

OUTCOME_CLASSES = [
    "research_success",
    "policy_reward",
    "world_model_prediction",
    "no_effect",
]
_OUTCOME_INDEX = {cls: i for i, cls in enumerate(OUTCOME_CLASSES)}


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def encode(
    state_vector: list[float],
    trigger_type: str,
    last_fire_times: dict[str, float],
) -> list[float]:
    """Encode a thought trigger event into a 44-dim feature vector.

    Args:
        state_vector: 20-dim system state from PolicyStateEncoder (or zeros).
        trigger_type: Which trigger fired (one of TRIGGER_TYPES).
        last_fire_times: Map of trigger_type → last-fire timestamp for all types.

    Returns:
        44-dim float vector, all values in [0, 1].
    """
    features = [0.0] * FEATURE_DIM

    # Block A: system state (dims 0-19)
    for i in range(min(20, len(state_vector))):
        features[i] = _clamp(float(state_vector[i]))

    # Block B: trigger type one-hot (dims 20-31)
    idx = _TRIGGER_INDEX.get(trigger_type)
    if idx is not None:
        features[20 + idx] = 1.0

    # Block C: time-since-last-fire (dims 32-43), normalized by 600s (10 min)
    now = time.time()
    for ttype, ti in _TRIGGER_INDEX.items():
        last = last_fire_times.get(ttype, 0.0)
        if last > 0.0:
            elapsed = now - last
            features[32 + ti] = _clamp(elapsed / 600.0)
        else:
            features[32 + ti] = 1.0  # never fired = max

    return features


def encode_label(outcome: str) -> list[float]:
    """Encode outcome as a 4-dim probability vector (one-hot).

    Args:
        outcome: One of OUTCOME_CLASSES.

    Returns:
        4-dim float vector (one-hot softmax target).
    """
    label = [0.0] * len(OUTCOME_CLASSES)
    idx = _OUTCOME_INDEX.get(outcome)
    if idx is not None:
        label[idx] = 1.0
    else:
        label[len(OUTCOME_CLASSES) - 1] = 1.0  # default: no_effect
    return label


def build_metadata(
    trigger_type: str,
    thought_id: str,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Build metadata for JSONL persistence (teacher/feature pairing).

    The ``thought_id`` is the pairing key — each thought produces one feature
    vector and (later, when outcome is known) one label vector.
    """
    meta: dict[str, Any] = {
        "thought_id": thought_id,
        "trigger_type": trigger_type,
        "timestamp": time.time(),
    }
    if outcome is not None:
        meta["outcome"] = outcome
    return meta
