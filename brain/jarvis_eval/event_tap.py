"""Read-only EventBus subscriber for the eval sidecar.

The eval sidecar is a read-only observer. This module subscribes to
consciousness events and records them into the eval store. It never
emits events, writes memories, or influences any cognition path.

Handler pattern: O(1) deque append under lock. Background flush loop
writes buffered events to the JSONL store every FLUSH_INTERVAL_S.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

from consciousness.events import (
    ATTRIBUTION_ENTRY_RECORDED,
    AUTONOMY_DELTA_MEASURED,
    AUTONOMY_INTENT_BLOCKED,
    AUTONOMY_INTENT_QUEUED,
    AUTONOMY_LEVEL_CHANGED,
    AUTONOMY_RESEARCH_COMPLETED,
    AUTONOMY_RESEARCH_FAILED,
    AUTONOMY_RESEARCH_SKIPPED,
    AUTONOMY_RESEARCH_STARTED,
    BELIEF_GRAPH_EDGE_CREATED,
    BELIEF_GRAPH_INTEGRITY_CHECK,
    BELIEF_GRAPH_PROPAGATION_COMPLETE,
    CALIBRATION_CONFIDENCE_ADJUSTED,
    CALIBRATION_CORRECTION_DETECTED,
    CALIBRATION_DRIFT_DETECTED,
    CALIBRATION_UPDATED,
    CAPABILITY_CLAIM_BLOCKED,
    CAPABILITY_GAP_DETECTED,
    COMPANION_GRADUATION,
    CONSCIOUSNESS_ANALYSIS,
    CONSCIOUSNESS_CAPABILITY_UNLOCKED,
    CONSCIOUSNESS_EMERGENT_BEHAVIOR,
    CONSCIOUSNESS_EVOLUTION_EVENT,
    CONSCIOUSNESS_LEARNING_PROTOCOL,
    CONSCIOUSNESS_MUTATION_PROPOSED,
    CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
    CONTRADICTION_DETECTED,
    CONTRADICTION_RESOLVED,
    CONTRADICTION_TENSION_HELD,
    CONVERSATION_RESPONSE,
    CONVERSATION_USER_MESSAGE,
    CURIOSITY_ANSWER_PROCESSED,
    CURIOSITY_QUESTION_ASKED,
    CURIOSITY_QUESTION_GENERATED,
    FRACTAL_RECALL_SURFACED,
    GESTATION_COMPLETE,
    GESTATION_DIRECTIVE_COMPLETED,
    GESTATION_PHASE_ADVANCED,
    GESTATION_READINESS_UPDATE,
    GESTATION_STARTED,
    GOAL_ABANDONED,
    GOAL_COMPLETED,
    GOAL_CREATED,
    GOAL_PAUSED,
    GOAL_PROGRESS_UPDATE,
    GOAL_PROMOTED,
    GOAL_RESUMED,
    HEMISPHERE_ARCHITECTURE_DESIGNED,
    HEMISPHERE_DISTILLATION_STATS,
    HEMISPHERE_EVOLUTION_COMPLETE,
    HEMISPHERE_NETWORK_READY,
    HEMISPHERE_PERFORMANCE_WARNING,
    HEMISPHERE_TRAINING_PROGRESS,
    IDENTITY_BOUNDARY_BLOCKED,
    IDENTITY_SCOPE_ASSIGNED,
    IMPROVEMENT_NEEDS_APPROVAL,
    IMPROVEMENT_PROMOTED,
    IMPROVEMENT_ROLLED_BACK,
    IMPROVEMENT_STARTED,
    IMPROVEMENT_VALIDATED,
    IMPROVEMENT_DRY_RUN,
    IMPROVEMENT_SANDBOX_PASSED,
    IMPROVEMENT_SANDBOX_FAILED,
    IMPROVEMENT_POST_RESTART_VERIFIED,
    MATRIX_DEEP_LEARNING_REQUESTED,
    MATRIX_EXPANSION_TRIGGERED,
    MEMORY_ASSOCIATED,
    MEMORY_DECAY_CYCLE,
    MEMORY_TRIMMED,
    MEMORY_WRITE,
    META_THOUGHT_GENERATED,
    MUTATION_APPLIED,
    MUTATION_REJECTED,
    MUTATION_ROLLBACK,
    ONBOARDING_CHECKPOINT_MET,
    ONBOARDING_DAY_ADVANCED,
    ONBOARDING_EXERCISE_PROMPTED,
    OUTCOME_RESOLVED,
    OUTPUT_RELEASE_BLOCKED,
    OUTPUT_VALIDATION_RECORDED,
    PERCEPTION_BARGE_IN,
    PERCEPTION_FACE_IDENTIFIED,
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_WAKE_WORD,
    PERSONALITY_ROLLBACK,
    PREDICTION_VALIDATED,
    QUARANTINE_SIGNAL_EMITTED,
    QUARANTINE_TICK_COMPLETE,
    REFLECTIVE_AUDIT_COMPLETED,
    REFLECTIVE_AUDIT_FINDING,
    RETRY_EXECUTED,
    RETRY_EXHAUSTED,
    RETRY_SCHEDULED,
    SKILL_DEGRADATION_DETECTED,
    SKILL_JOB_PHASE_CHANGED,
    SKILL_LEARNING_COMPLETED,
    SKILL_LEARNING_STARTED,
    SKILL_REGISTERED,
    SKILL_STATUS_CHANGED,
    SOUL_INTEGRITY_REPAIR_NEEDED,
    SOUL_INTEGRITY_UPDATED,
    WORLD_MODEL_DELTA,
    WORLD_MODEL_PREDICTION_VALIDATED,
    WORLD_MODEL_PROMOTED,
    WORLD_MODEL_UPDATE,
    event_bus,
)
from consciousness.modes import MODE_CHANGE

from jarvis_eval.config import EVENT_BUFFER_MAXLEN, FLUSH_INTERVAL_S
from jarvis_eval.contracts import EvalEvent

logger = logging.getLogger(__name__)

# Module-local event constant (defined in perception/identity_fusion.py)
_IDENTITY_RESOLVED = "perception:identity_resolved"

_TAPPED_EVENTS: list[str] = [
    # -- Original core --
    MODE_CHANGE,
    CONVERSATION_RESPONSE,
    MEMORY_WRITE,
    CONTRADICTION_DETECTED,
    CONTRADICTION_RESOLVED,
    CONTRADICTION_TENSION_HELD,
    CALIBRATION_DRIFT_DETECTED,
    CALIBRATION_CORRECTION_DETECTED,
    CALIBRATION_CONFIDENCE_ADJUSTED,
    QUARANTINE_SIGNAL_EMITTED,
    QUARANTINE_TICK_COMPLETE,
    REFLECTIVE_AUDIT_COMPLETED,
    REFLECTIVE_AUDIT_FINDING,
    SOUL_INTEGRITY_UPDATED,
    SOUL_INTEGRITY_REPAIR_NEEDED,
    CAPABILITY_CLAIM_BLOCKED,
    CAPABILITY_GAP_DETECTED,
    MUTATION_APPLIED,
    MUTATION_REJECTED,
    MUTATION_ROLLBACK,
    AUTONOMY_RESEARCH_COMPLETED,
    CONSCIOUSNESS_ANALYSIS,
    # -- Voice pipeline --
    PERCEPTION_WAKE_WORD,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_BARGE_IN,
    CONVERSATION_USER_MESSAGE,
    ATTRIBUTION_ENTRY_RECORDED,
    OUTCOME_RESOLVED,
    OUTPUT_VALIDATION_RECORDED,
    OUTPUT_RELEASE_BLOCKED,
    # -- Identity pipeline --
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_FACE_IDENTIFIED,
    _IDENTITY_RESOLVED,
    IDENTITY_SCOPE_ASSIGNED,
    IDENTITY_BOUNDARY_BLOCKED,
    # -- Memory --
    MEMORY_ASSOCIATED,
    MEMORY_DECAY_CYCLE,
    MEMORY_TRIMMED,
    # -- Autonomy --
    AUTONOMY_INTENT_QUEUED,
    AUTONOMY_INTENT_BLOCKED,
    AUTONOMY_RESEARCH_STARTED,
    AUTONOMY_RESEARCH_FAILED,
    AUTONOMY_RESEARCH_SKIPPED,
    AUTONOMY_DELTA_MEASURED,
    AUTONOMY_LEVEL_CHANGED,
    RETRY_SCHEDULED,
    RETRY_EXECUTED,
    RETRY_EXHAUSTED,
    # -- Hemisphere / Distillation --
    HEMISPHERE_TRAINING_PROGRESS,
    HEMISPHERE_DISTILLATION_STATS,
    HEMISPHERE_NETWORK_READY,
    HEMISPHERE_ARCHITECTURE_DESIGNED,
    HEMISPHERE_EVOLUTION_COMPLETE,
    HEMISPHERE_PERFORMANCE_WARNING,
    # -- Skills --
    SKILL_REGISTERED,
    SKILL_STATUS_CHANGED,
    SKILL_LEARNING_STARTED,
    SKILL_LEARNING_COMPLETED,
    SKILL_JOB_PHASE_CHANGED,
    SKILL_DEGRADATION_DETECTED,
    # -- Gestation --
    GESTATION_STARTED,
    GESTATION_PHASE_ADVANCED,
    GESTATION_DIRECTIVE_COMPLETED,
    GESTATION_READINESS_UPDATE,
    GESTATION_COMPLETE,
    # -- Consciousness --
    META_THOUGHT_GENERATED,
    CONSCIOUSNESS_EVOLUTION_EVENT,
    CONSCIOUSNESS_MUTATION_PROPOSED,
    CONSCIOUSNESS_EMERGENT_BEHAVIOR,
    CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
    CONSCIOUSNESS_CAPABILITY_UNLOCKED,
    CONSCIOUSNESS_LEARNING_PROTOCOL,
    # -- World Model --
    WORLD_MODEL_UPDATE,
    WORLD_MODEL_PREDICTION_VALIDATED,
    WORLD_MODEL_DELTA,
    WORLD_MODEL_PROMOTED,
    # -- Calibration / Epistemic --
    CALIBRATION_UPDATED,
    PREDICTION_VALIDATED,
    BELIEF_GRAPH_EDGE_CREATED,
    BELIEF_GRAPH_PROPAGATION_COMPLETE,
    BELIEF_GRAPH_INTEGRITY_CHECK,
    # -- Curiosity Bridge --
    CURIOSITY_QUESTION_GENERATED,
    CURIOSITY_QUESTION_ASKED,
    CURIOSITY_ANSWER_PROCESSED,
    # -- Fractal Recall --
    FRACTAL_RECALL_SURFACED,
    # -- Self-Improvement --
    IMPROVEMENT_STARTED,
    IMPROVEMENT_NEEDS_APPROVAL,
    IMPROVEMENT_VALIDATED,
    IMPROVEMENT_PROMOTED,
    IMPROVEMENT_ROLLED_BACK,
    IMPROVEMENT_DRY_RUN,
    IMPROVEMENT_SANDBOX_PASSED,
    IMPROVEMENT_SANDBOX_FAILED,
    IMPROVEMENT_POST_RESTART_VERIFIED,
    # -- Goals --
    GOAL_CREATED,
    GOAL_PROMOTED,
    GOAL_COMPLETED,
    GOAL_ABANDONED,
    GOAL_PAUSED,
    GOAL_RESUMED,
    GOAL_PROGRESS_UPDATE,
    # -- Onboarding --
    ONBOARDING_DAY_ADVANCED,
    ONBOARDING_CHECKPOINT_MET,
    ONBOARDING_EXERCISE_PROMPTED,
    COMPANION_GRADUATION,
    # -- Personality --
    PERSONALITY_ROLLBACK,
    # -- Matrix Protocol --
    MATRIX_DEEP_LEARNING_REQUESTED,
    MATRIX_EXPANSION_TRIGGERED,
]


class EvalEventTap:
    """Subscribes to EventBus events and buffers them for store flush."""

    def __init__(self) -> None:
        self._buffer: deque[EvalEvent] = deque(maxlen=EVENT_BUFFER_MAXLEN)
        self._lock = threading.Lock()
        self._cleanups: list[Callable[[], None]] = []
        self._wired: bool = False
        self._run_id: str = ""
        self._current_mode: str = ""
        self._events_buffered: int = 0

    def wire(self, run_id: str = "") -> None:
        """Subscribe to all tapped events. Idempotent."""
        if self._wired:
            return
        self._run_id = run_id
        for event_name in _TAPPED_EVENTS:
            handler = self._make_handler(event_name)
            cleanup = event_bus.on(event_name, handler)
            self._cleanups.append(cleanup)
        self._wired = True
        logger.info("Eval event tap wired (%d events)", len(self._cleanups))

    def unwire(self) -> None:
        """Unsubscribe from all events."""
        for cleanup in self._cleanups:
            try:
                cleanup()
            except Exception:
                pass
        self._cleanups.clear()
        self._wired = False

    def _make_handler(self, event_name: str) -> Callable[..., None]:
        """Create an O(1) handler that appends to the buffer."""
        def _handler(**kwargs: Any) -> None:
            if event_name == MODE_CHANGE:
                self._current_mode = kwargs.get("to_mode", kwargs.get("mode", ""))

            payload = _safe_payload(kwargs)
            ev = EvalEvent(
                event_type=event_name,
                payload=payload,
                mode=self._current_mode,
                run_id=self._run_id,
            )
            with self._lock:
                self._buffer.append(ev)
                self._events_buffered += 1

        return _handler

    def drain(self) -> list[EvalEvent]:
        """Drain buffer and return events for store flushing."""
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "wired": self._wired,
                "buffer_size": len(self._buffer),
                "total_buffered": self._events_buffered,
                "tapped_event_count": len(_TAPPED_EVENTS),
                "current_mode": self._current_mode,
            }


def _safe_payload(kwargs: dict[str, Any], max_keys: int = 20) -> dict[str, Any]:
    """Extract a serializable subset of event kwargs."""
    correlation_keys = (
        "trace_id",
        "request_id",
        "upgrade_id",
        "conversation_id",
        "output_id",
        "validation_id",
        "retry_id",
        "target_event_type",
        "intent_id",
        "goal_id",
        "task_id",
        "golden_trace_id",
        "golden_command_id",
        "entry_id",
        "root_entry_id",
        "parent_entry_id",
        "source_event",
    )
    payload: dict[str, Any] = {}
    for key in correlation_keys:
        if key not in kwargs:
            continue
        try:
            payload[key] = _coerce(kwargs.get(key))
        except Exception:
            payload[key] = "<unserializable>"
        if len(payload) >= max_keys:
            return payload
    for k, v in kwargs.items():
        if k in payload:
            continue
        if len(payload) >= max_keys:
            break
        try:
            if isinstance(v, (str, int, float, bool, type(None))):
                payload[k] = v
            elif isinstance(v, dict):
                payload[k] = {str(dk): _coerce(dv) for dk, dv in list(v.items())[:10]}
            elif isinstance(v, (list, tuple)):
                payload[k] = [_coerce(x) for x in v[:10]]
            else:
                payload[k] = str(v)[:200]
        except Exception:
            payload[k] = "<unserializable>"
    return payload


def _coerce(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return str(v)[:200]
