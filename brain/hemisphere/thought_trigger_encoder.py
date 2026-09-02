"""Thought trigger selection encoder for the THOUGHT_TRIGGER_SELECTOR specialist.

Shadow-only. Learns which meta-cognitive thought trigger the deterministic
generator chose, given compact system state. DistillationConfig is 24→13
(SPARK §3 / types.py). The old 44-dim / 4-class outcome head was an orphan
stub that never matched the live config.

This encoder has NO import path to thought generation, mutation, memory writes,
or event emission except the collector record helper (teacher feed only).

Dimension blocks (24-dim, all [0,1]):
  Block A (dims  0-19): system state (PolicyStateEncoder dims, padded)
  Block B (dims 20-23): compact extras (tension, time-of-day, recency, spare)

Labels: one-hot over ``THOUGHT_TRIGGER_NAMES`` (13). Not outcome classes.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 24  # DistillationConfig thought_trigger_selector.input_dim

try:
    from consciousness.meta_cognitive_thoughts import THOUGHT_TRIGGER_NAMES
except Exception:  # partial test harness
    THOUGHT_TRIGGER_NAMES = (
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
        "belief_validation_curiosity",
    )

NUM_TRIGGER_TYPES = len(THOUGHT_TRIGGER_NAMES)
_TRIGGER_INDEX = {name: i for i, name in enumerate(THOUGHT_TRIGGER_NAMES)}


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def encode(
    state_vector: list[float] | None,
    trigger_type: str = "",
    last_fire_times: dict[str, float] | None = None,
) -> list[float]:
    """Encode compact system state into a 24-dim feature vector.

    ``trigger_type`` / ``last_fire_times`` are accepted for call-compat with the
    old 44-dim stub. The trigger is NOT baked into features (that leaked the
    label). Recency, if provided, occupies dim 22 as a single mean.
    """
    features = [0.0] * FEATURE_DIM
    vec = state_vector or []
    for i in range(min(20, len(vec))):
        features[i] = _clamp(float(vec[i]))

    # dim 20: spare / caller may pass tension as state_vector[20]
    if len(vec) > 20:
        features[20] = _clamp(float(vec[20]))

    # dim 21: time-of-day [0,1)
    try:
        lt = time.localtime()
        features[21] = _clamp((lt.tm_hour * 60 + lt.tm_min) / (24 * 60))
    except Exception:
        features[21] = 0.0

    # dim 22: mean recency across triggers (1.0 = never fired)
    fires = last_fire_times or {}
    if fires:
        now = time.time()
        rec = []
        for last in fires.values():
            if last and last > 0.0:
                rec.append(_clamp((now - float(last)) / 600.0))
            else:
                rec.append(1.0)
        features[22] = _clamp(sum(rec) / len(rec)) if rec else 1.0
    else:
        features[22] = 1.0

    # dim 23: whether a trigger name was supplied (context only; not the label)
    features[23] = 1.0 if trigger_type else 0.0
    return features


def encode_label(trigger_name: str) -> list[float]:
    """Encode the deterministic trigger choice as a 13-dim one-hot."""
    label = [0.0] * NUM_TRIGGER_TYPES
    idx = _TRIGGER_INDEX.get(trigger_name)
    if idx is not None:
        label[idx] = 1.0
    elif NUM_TRIGGER_TYPES:
        # unknown trigger: last slot is belief_validation_curiosity's neighbor;
        # leave zeros rather than fake a class.
        pass
    return label


def build_metadata(
    trigger_type: str,
    thought_id: str,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Pairing key for DistillationCollector (thought_id)."""
    meta: dict[str, Any] = {
        "thought_id": thought_id,
        "trigger_type": trigger_type,
        "timestamp": time.time(),
    }
    if outcome is not None:
        meta["outcome"] = outcome
    return meta


def record_teacher_pair(
    *,
    thought_id: str,
    trigger_name: str,
    state_vector: list[float] | None = None,
    grounded: bool = False,
    last_fire_times: dict[str, float] | None = None,
) -> None:
    """Shadow collector feed. Does not train, promote, or speak.

    Called from the THOUGHT_VALIDATION_OUTCOME path. thought_trigger_selector
    stays out of ``_TIER1_FOCUSES`` and Weight-Room ``blocked_by_design`` until
    Thought Maturity P3 is named. Accruing the pair is the dead-wire close.
    """
    if not thought_id:
        return
    try:
        from hemisphere.distillation import distillation_collector
    except Exception:
        return
    trigger = trigger_name or "belief_validation_curiosity"
    feats = encode(state_vector, trigger_type=trigger, last_fire_times=last_fire_times)
    label = encode_label(trigger)
    fidelity = 1.0 if grounded else 0.7
    outcome = "grounded" if grounded else "ungrounded"
    meta = build_metadata(trigger, thought_id, outcome=outcome)
    try:
        distillation_collector.record(
            teacher="thought_trigger_features",
            signal_type="thought_trigger_features",
            data=feats,
            metadata=dict(meta),
            origin="live",
            fidelity=fidelity,
        )
        distillation_collector.record(
            teacher="thought_trigger_resolver",
            signal_type="thought_trigger_resolver",
            data=label,
            metadata=dict(meta),
            origin="live",
            fidelity=fidelity,
        )
    except Exception:
        logger.debug("thought_trigger teacher pair record failed", exc_info=True)
