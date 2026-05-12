"""EventBus and event type definitions for Jarvis consciousness."""

from __future__ import annotations

import enum
import logging
import threading
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

JarvisPhase = Literal[
    "INITIALIZING", "LISTENING", "OBSERVING", "PROCESSING", "LEARNING", "STANDBY", "DREAMING"
]

JarvisTone = Literal["professional", "casual", "urgent", "empathetic", "playful"]

MemoryType = Literal[
    "core",
    "conversation",
    "observation",
    "task_completed",
    "user_preference",
    "factual_knowledge",
    "error_recovery",
    "contextual_insight",
    "self_improvement",
]

ProvenanceType = Literal[
    "observed",             # Direct perception (vision, audio, presence, ambient)
    "user_claim",           # User stated (preferences, corrections, identity)
    "conversation",         # Emerged from conversation (LLM response, dialogue)
    "model_inference",      # LLM or NN inferred (reflection, dream insight)
    "external_source",      # External research (web, academic, library study)
    "experiment_result",    # Self-improvement outcomes, learning job results
    "derived_pattern",      # Pattern recognition (clustering, association, analytics)
    "seed",                 # Birth/gestation seed memories
    "unknown",              # Legacy or unclassified
]

PROVENANCE_BOOST: dict[str, float] = {
    "external_source": 0.10,
    "experiment_result": 0.08,
    "observed": 0.06,
    "user_claim": 0.04,
    "conversation": 0.02,
    "model_inference": 0.0,
    "derived_pattern": 0.0,
    "seed": 0.0,
    "unknown": 0.0,
}


_MIN_PROVENANCE_SAMPLES = 20
_prov_cache: tuple[float, dict[str, float], dict[str, int]] = (0.0, {}, {})
_PROV_CACHE_TTL = 30.0


def _get_calibrated_provenance_accuracy() -> tuple[dict[str, float], dict[str, int]]:
    """Fetch per-provenance accuracy and sample counts from calibration, if available.

    Cached for 30s to avoid redundant iteration during bulk retrieval scoring.
    """
    global _prov_cache
    import time as _t
    now = _t.time()
    if now - _prov_cache[0] < _PROV_CACHE_TTL:
        return _prov_cache[1], _prov_cache[2]
    try:
        from epistemic.calibration import TruthCalibrationEngine
        tce = TruthCalibrationEngine.get_instance()
        if tce and tce._confidence_calibrator:
            cc = tce._confidence_calibrator
            acc = cc.get_per_provenance_accuracy()
            counts = cc.get_provenance_sample_counts()
            _prov_cache = (now, acc, counts)
            return acc, counts
    except Exception:
        pass
    return {}, {}


def resolve_provenance_boost(mem: Memory) -> float:
    """Single source of truth for provenance-based score boost.

    Uses the provenance field if set; falls back to tag-based heuristic
    for legacy memories where provenance == "unknown".

    When calibration has enough samples (>=20) for a provenance class,
    blends 50% static base + 50% dynamic (accuracy scaled to 0.12 max).
    """
    prov = getattr(mem, "provenance", "unknown")
    if prov != "unknown":
        base = PROVENANCE_BOOST.get(prov, 0.0)
    else:
        tags_set = set(mem.tags) if mem.tags else set()
        if "evidence:peer_reviewed" in tags_set:
            base = 0.12
        elif "evidence:codebase" in tags_set or "code_sourced" in tags_set:
            base = 0.10
        elif "autonomous_research" in tags_set and mem.type == "factual_knowledge":
            base = 0.08
        else:
            base = 0.0

    if prov == "unknown" or base == 0.0:
        return base

    prov_acc, prov_counts = _get_calibrated_provenance_accuracy()
    count = prov_counts.get(prov, 0)
    if count >= _MIN_PROVENANCE_SAMPLES and prov in prov_acc:
        dynamic = prov_acc[prov] * 0.12
        return round(base * 0.5 + dynamic * 0.5, 4)

    return base


PROVENANCE_ORDINAL: dict[str, int] = {
    "observed": 0, "user_claim": 1, "conversation": 2, "model_inference": 3,
    "external_source": 4, "experiment_result": 5, "derived_pattern": 6,
    "seed": 7, "unknown": 8,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    id: str
    timestamp: float
    weight: float
    tags: tuple[str, ...]
    payload: Any
    type: MemoryType
    associations: tuple[str, ...] = ()
    decay_rate: float = 0.01
    is_core: bool = False
    last_validated: float = 0.0
    association_count: int = 0
    priority: int = 0
    provenance: str = "unknown"
    identity_owner: str = ""
    identity_owner_type: str = ""
    identity_subject: str = ""
    identity_subject_type: str = ""
    identity_scope_key: str = ""
    identity_confidence: float = 0.0
    identity_needs_resolution: bool = False
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass(frozen=True)
class PerceptionEvent:
    source: Literal["vision", "audio", "screen", "system"]
    type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    conversation_id: str = ""


# ---------------------------------------------------------------------------
# Event names (string constants used as keys)
# ---------------------------------------------------------------------------

KERNEL_BOOT = "kernel:boot"
KERNEL_TICK = "kernel:tick"
KERNEL_PHASE_CHANGE = "kernel:phase_change"
KERNEL_THOUGHT = "kernel:thought"
KERNEL_ERROR = "kernel:error"

PHASE_SHIFT = "phase:shift"
TONE_SHIFT = "tone:shift"
CONFIDENCE_UPDATE = "confidence:update"

MEMORY_WRITE = "memory:write"
MEMORY_DECAY_CYCLE = "memory:decay_cycle"
MEMORY_TRIMMED = "memory:trimmed"

PERCEPTION_EVENT = "perception:event"
PERCEPTION_USER_PRESENT = "perception:user_present"
PERCEPTION_USER_PRESENT_STABLE = "perception:user_present_stable"
PERCEPTION_USER_ATTENTION = "perception:user_attention"
PERCEPTION_AMBIENT_SOUND = "perception:ambient_sound"
PERCEPTION_WAKE_WORD = "perception:wake_word"
PERCEPTION_TRANSCRIPTION = "perception:transcription"
PERCEPTION_SCREEN_CONTEXT = "perception:screen_context"
PERCEPTION_SPEAKER_IDENTIFIED = "perception:speaker_identified"
PERCEPTION_USER_EMOTION = "perception:user_emotion"
PERCEPTION_POSE_DETECTED = "perception:pose_detected"
PERCEPTION_FACE_IDENTIFIED = "perception:face_identified"
PERCEPTION_SCENE_SUMMARY = "perception:scene_summary"
PERCEPTION_PARTIAL_TRANSCRIPTION = "perception:partial_transcription"  # reserved
PERCEPTION_BARGE_IN = "perception:barge_in"
PERCEPTION_PLAYBACK_COMPLETE = "perception:playback_complete"
PERCEPTION_AUDIO_CLIP = "perception:audio_clip"  # reserved
PERCEPTION_AUDIO_STREAM_START = "perception:audio_stream_start"  # reserved
PERCEPTION_AUDIO_STREAM_CHUNK = "perception:audio_stream_chunk"  # reserved
PERCEPTION_AUDIO_STREAM_END = "perception:audio_stream_end"  # reserved
PERCEPTION_AUDIO_FEATURES = "perception:audio_features"  # emitted, no subscriber yet
PERCEPTION_RAW_AUDIO = "perception:raw_audio"
PERCEPTION_TRANSCRIPTION_READY = "perception:transcription_ready"  # reserved
PERCEPTION_SENSOR_DISCONNECTED = "perception:sensor_disconnected"

CONVERSATION_USER_MESSAGE = "conversation:user_message"
CONVERSATION_RESPONSE = "conversation:response"
OUTPUT_VALIDATION_RECORDED = "output:validation_recorded"
OUTPUT_RELEASE_BLOCKED = "output:release_blocked"

SOUL_EXPORTED = "soul:exported"
SOUL_IMPORTED = "soul:imported"

# Consciousness subsystem events
CONSCIOUSNESS_ANALYSIS = "consciousness:analysis"
CONSCIOUSNESS_SELF_OBSERVATION = "consciousness:self_observation"
CONSCIOUSNESS_EVOLUTION_EVENT = "consciousness:evolution_event"
CONSCIOUSNESS_EMERGENT_BEHAVIOR = "consciousness:emergent_behavior"
CONSCIOUSNESS_TRANSCENDENCE_MILESTONE = "consciousness:transcendence_milestone"
CONSCIOUSNESS_MUTATION_PROPOSED = "consciousness:mutation_proposed"
CONSCIOUSNESS_CAPABILITY_UNLOCKED = "consciousness:capability_unlocked"
CONSCIOUSNESS_LEARNING_PROTOCOL = "consciousness:learning_protocol_activated"

# Mutation pipeline events
MUTATION_APPLIED = "mutation:applied"
MUTATION_REJECTED = "mutation:rejected"
MUTATION_ROLLBACK = "mutation:rollback"

# Meta-cognition events
META_THOUGHT_GENERATED = "meta:thought_generated"

# Existential / philosophical events
EXISTENTIAL_INQUIRY_COMPLETED = "existential:inquiry_completed"
PHILOSOPHICAL_DIALOGUE_COMPLETED = "philosophical:dialogue_completed"

SYSTEM_INIT_COMPLETE = "system:initialization_complete"
SYSTEM_EVENT_BUS_READY = "system:event_bus_ready"

MEMORY_ASSOCIATED = "memory:associated"
MEMORY_TRANSACTION_COMPLETE = "memory:transaction_complete"
MEMORY_TRANSACTION_ROLLBACK = "memory:transaction_rollback"
PERSONALITY_ROLLBACK = "personality:rollback"

# Hemisphere neural network events
# Self-improvement pipeline events
IMPROVEMENT_STARTED = "improvement:started"
IMPROVEMENT_VALIDATED = "improvement:validated"
IMPROVEMENT_PROMOTED = "improvement:promoted"
IMPROVEMENT_ROLLED_BACK = "improvement:rolled_back"
IMPROVEMENT_NEEDS_APPROVAL = "improvement:needs_approval"
IMPROVEMENT_DRY_RUN = "improvement:dry_run"
IMPROVEMENT_SANDBOX_PASSED = "improvement:sandbox_passed"
IMPROVEMENT_SANDBOX_FAILED = "improvement:sandbox_failed"
IMPROVEMENT_POST_RESTART_VERIFIED = "improvement:post_restart_verified"

# Autonomy research events
AUTONOMY_INTENT_QUEUED = "autonomy:intent_queued"
AUTONOMY_INTENT_BLOCKED = "autonomy:intent_blocked"
AUTONOMY_RESEARCH_STARTED = "autonomy:research_started"
AUTONOMY_RESEARCH_COMPLETED = "autonomy:research_completed"
AUTONOMY_RESEARCH_FAILED = "autonomy:research_failed"
AUTONOMY_LEVEL_CHANGED = "autonomy:level_changed"
AUTONOMY_DELTA_MEASURED = "autonomy:delta_measured"
AUTONOMY_RESEARCH_SKIPPED = "autonomy:research_skipped"

# Phase 6.5 — L3 escalation & attestation events
# Invariant: AUTONOMY_L3_ELIGIBLE fires only from live-runtime promotion
# criteria, never from attestation load or manual promotion. See
# docs/plans/phase_6_5_l3_escalation.plan.md for payload contracts.
AUTONOMY_L3_ELIGIBLE = "autonomy:l3_eligible"
AUTONOMY_ESCALATION_REQUESTED = "autonomy:escalation_requested"
AUTONOMY_ESCALATION_APPROVED = "autonomy:escalation_approved"
AUTONOMY_ESCALATION_REJECTED = "autonomy:escalation_rejected"
AUTONOMY_ESCALATION_ROLLED_BACK = "autonomy:escalation_rolled_back"
AUTONOMY_ESCALATION_PARKED = "autonomy:escalation_parked"
AUTONOMY_ESCALATION_EXPIRED = "autonomy:escalation_expired"
AUTONOMY_L3_ACTIVATION_DENIED = "autonomy:l3_activation_denied"
# AUTONOMY_L3_PROMOTED is the single authoritative record that live
# autonomy became L3 this session. It fires ONLY on a successful 2→3
# transition inside set_autonomy_level. Denials use
# AUTONOMY_L3_ACTIVATION_DENIED; rollbacks of the triggering
# escalation use AUTONOMY_ESCALATION_ROLLED_BACK. The payload's
# ``outcome`` field is always the literal ``"clean"``.
AUTONOMY_L3_PROMOTED = "autonomy:l3_promoted"

# Gestation events
GESTATION_STARTED = "gestation:started"
GESTATION_PHASE_ADVANCED = "gestation:phase_advanced"
GESTATION_DIRECTIVE_COMPLETED = "gestation:directive_completed"
GESTATION_READINESS_UPDATE = "gestation:readiness_update"
GESTATION_COMPLETE = "gestation:complete"
GESTATION_FIRST_CONTACT = "gestation:first_contact"

HEMISPHERE_ARCHITECTURE_DESIGNED = "hemisphere:architecture_designed"
HEMISPHERE_TRAINING_PROGRESS = "hemisphere:training_progress"
HEMISPHERE_NETWORK_READY = "hemisphere:network_ready"
HEMISPHERE_EVOLUTION_COMPLETE = "hemisphere:evolution_complete"
HEMISPHERE_MIGRATION_DECISION = "hemisphere:migration_decision"
HEMISPHERE_SUBSTRATE_MIGRATION = "hemisphere:substrate_migration"
HEMISPHERE_PERFORMANCE_WARNING = "hemisphere:performance_warning"
HEMISPHERE_DISTILLATION_STATS = "hemisphere:distillation_stats"

# Skill learning events
SKILL_REGISTERED = "skill:registered"
SKILL_STATUS_CHANGED = "skill:status_changed"
SKILL_LEARNING_STARTED = "skill:learning_started"
SKILL_LEARNING_COMPLETED = "skill:learning_completed"
SKILL_VERIFICATION_RECORDED = "skill:verification_recorded"
SKILL_JOB_PHASE_CHANGED = "skill:job_phase_changed"

# Matrix Protocol events
MATRIX_DEEP_LEARNING_REQUESTED = "matrix:deep_learning_requested"
MATRIX_EXPANSION_TRIGGERED = "matrix:expansion_triggered"

# Capability discovery events
CAPABILITY_CLAIM_BLOCKED = "capability:claim_blocked"
CAPABILITY_GAP_DETECTED = "capability:gap_detected"

# Attribution ledger events
ATTRIBUTION_ENTRY_RECORDED = "attribution:entry_recorded"
OUTCOME_RESOLVED = "attribution:outcome_resolved"

# Retry traceability events
RETRY_SCHEDULED = "retry:scheduled"
RETRY_EXECUTED = "retry:executed"
RETRY_EXHAUSTED = "retry:exhausted"

# Contradiction engine events (Layer 5)
CONTRADICTION_DETECTED = "contradiction:detected"
CONTRADICTION_RESOLVED = "contradiction:resolved"
CONTRADICTION_TENSION_HELD = "contradiction:tension_held"

# Truth calibration events (Layer 6)
CALIBRATION_UPDATED = "calibration:updated"
CALIBRATION_DRIFT_DETECTED = "calibration:drift"
CALIBRATION_CORRECTION_DETECTED = "calibration:correction"
SKILL_DEGRADATION_DETECTED = "calibration:skill_degraded"
PREDICTION_VALIDATED = "calibration:prediction_validated"
CALIBRATION_CONFIDENCE_ADJUSTED = "calibration:confidence_adjusted"

# Belief graph events (Layer 7)
BELIEF_GRAPH_EDGE_CREATED = "belief_graph:edge_created"
BELIEF_GRAPH_PROPAGATION_COMPLETE = "belief_graph:propagation_complete"
BELIEF_GRAPH_INTEGRITY_CHECK = "belief_graph:integrity_check"

# Quarantine events (Layer 8)
QUARANTINE_SIGNAL_EMITTED = "quarantine:signal_emitted"
QUARANTINE_TICK_COMPLETE = "quarantine:tick_complete"

# Reflective audit events (Layer 9)
REFLECTIVE_AUDIT_COMPLETED = "audit:completed"
REFLECTIVE_AUDIT_FINDING = "audit:finding"

# Soul integrity events (Layer 10)
SOUL_INTEGRITY_UPDATED = "soul_integrity:updated"
SOUL_INTEGRITY_REPAIR_NEEDED = "soul_integrity:repair_needed"

# Memory optimizer events
CONSCIOUSNESS_CLEANUP_OBSERVATIONS = "consciousness:cleanup_observations"
CONSCIOUSNESS_CLEANUP_OLD_CHAINS = "consciousness:cleanup_old_chains"
CONSCIOUSNESS_CLEAR_CACHES = "consciousness:clear_caches"
CONSCIOUSNESS_REDUCE_OBSERVATION_RATE = "consciousness:reduce_observation_rate"

# Identity boundary events (Layer 3)
IDENTITY_SCOPE_ASSIGNED = "identity:scope_assigned"
IDENTITY_BOUNDARY_BLOCKED = "identity:boundary_blocked"
IDENTITY_AMBIGUITY_DETECTED = "identity:ambiguity_detected"

# World Model events (cognition layer)
WORLD_MODEL_UPDATE = "world_model:update"
WORLD_MODEL_DELTA = "world_model:delta"
WORLD_MODEL_PREDICTION = "world_model:prediction"
WORLD_MODEL_PREDICTION_VALIDATED = "world_model:prediction_validated"
WORLD_MODEL_PROMOTED = "world_model:promoted"
WORLD_MODEL_UNCERTAINTY_UPDATE = "world_model:uncertainty_update"

# Spatial intelligence events
SPATIAL_TRACK_UPDATED = "spatial:track_updated"
SPATIAL_ANCHOR_REGISTERED = "spatial:anchor_registered"
SPATIAL_DELTA_PROMOTED = "spatial:delta_promoted"
SPATIAL_CALIBRATION_CHANGED = "spatial:calibration_changed"

# Curiosity question events
CURIOSITY_QUESTION_GENERATED = "curiosity:question_generated"
CURIOSITY_QUESTION_ASKED = "curiosity:question_asked"
CURIOSITY_ANSWER_PROCESSED = "curiosity:answer_processed"

# Fractal Recall events
FRACTAL_RECALL_SURFACED = "fractal_recall:surfaced"

# Synthetic Exercise events
SYNTHETIC_EXERCISE_STATE = "synthetic_exercise:state"

# Goal Continuity Layer events
GOAL_CREATED = "goal:created"
GOAL_PROMOTED = "goal:promoted"
GOAL_COMPLETED = "goal:completed"
GOAL_ABANDONED = "goal:abandoned"
GOAL_PAUSED = "goal:paused"
GOAL_RESUMED = "goal:resumed"
GOAL_PROGRESS_UPDATE = "goal:progress_update"

# Onboarding / Companion Training events
ONBOARDING_DAY_ADVANCED = "onboarding:day_advanced"
ONBOARDING_CHECKPOINT_MET = "onboarding:checkpoint_met"
ONBOARDING_EXERCISE_PROMPTED = "onboarding:exercise_prompted"
COMPANION_GRADUATION = "onboarding:companion_graduation"

# Capability Acquisition Pipeline events
ACQUISITION_CREATED = "acquisition:created"
ACQUISITION_CLASSIFIED = "acquisition:classified"
ACQUISITION_LANE_STARTED = "acquisition:lane_started"
ACQUISITION_LANE_COMPLETED = "acquisition:lane_completed"
ACQUISITION_PLAN_READY = "acquisition:plan_ready"
ACQUISITION_CODE_GENERATED = "acquisition:code_generated"
ACQUISITION_PLUGIN_DEPLOYED = "acquisition:plugin_deployed"
ACQUISITION_VERIFIED = "acquisition:verified"
ACQUISITION_COMPLETED = "acquisition:completed"
ACQUISITION_FAILED = "acquisition:failed"
ACQUISITION_APPROVAL_NEEDED = "acquisition:approval_needed"
ACQUISITION_PLAN_REVIEWED = "acquisition:plan_reviewed"
ACQUISITION_DEPLOYMENT_REVIEWED = "acquisition:deployment_reviewed"


# ---------------------------------------------------------------------------
# Barrier state
# ---------------------------------------------------------------------------


class _BarrierState(enum.Enum):
    CLOSED = 0
    OPENING = 1
    OPEN = 2


CRITICAL_EVENTS = frozenset({
    KERNEL_BOOT, KERNEL_TICK, MEMORY_WRITE, PERCEPTION_TRANSCRIPTION,
    CONVERSATION_RESPONSE, MEMORY_TRANSACTION_COMPLETE,
})


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


_MAX_EMIT_DEPTH = 10


class EventBus:
    """Typed pub/sub event bus with startup barrier.

    Events emitted before the barrier is opened are buffered and flushed
    when ``open_barrier()`` is called (boot/system events bypass the barrier).
    """

    _instance: EventBus | None = None
    _recursion_depth: threading.local = threading.local()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: dict[str, list[Callable[..., Any]]] = {}
        self._barrier = _BarrierState.CLOSED
        self._early_queue: list[tuple[str, dict[str, Any]]] = []
        self._max_buffered = 10_000
        self._max_listeners = 50

        self._error_counts: dict[str, list[float]] = {}
        self._circuit_breakers: dict[str, float] = {}
        self._retry_queue: list[tuple[str, dict[str, Any], int, float]] = []
        self._metrics: dict[str, int] = {
            "total_events": 0,
            "failed_events": 0,
            "retried_events": 0,
            "circuit_breaker_trips": 0,
            "recursive_drops": 0,
        }
        self._processing_times: deque[float] = deque(maxlen=1000)
        self._event_validator: Any = None

        self.CIRCUIT_BREAKER_THRESHOLD = 10
        self.CIRCUIT_BREAKER_TIMEOUT = 30.0
        self.MAX_RETRIES = 3
        self.RETRY_DELAYS = [1.0, 2.0, 4.0]

    # -- singleton -----------------------------------------------------------

    @classmethod
    def instance(cls) -> EventBus:
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # -- subscribe / unsubscribe --------------------------------------------

    def on(self, event_type: str, handler: Callable[..., Any]) -> Callable[[], None]:
        with self._lock:
            listeners = self._listeners.setdefault(event_type, [])
            if len(listeners) >= self._max_listeners:
                logger.warning("EventBus: max listeners (%d) for %s", self._max_listeners, event_type)
                return lambda: None
            listeners.append(handler)

        def _cleanup() -> None:
            with self._lock:
                try:
                    self._listeners.get(event_type, []).remove(handler)
                except ValueError:
                    pass

        return _cleanup

    def off(self, event_type: str, handler: Callable[..., Any] | None = None) -> None:
        with self._lock:
            if handler is None:
                self._listeners.pop(event_type, None)
            else:
                listeners = self._listeners.get(event_type, [])
                self._listeners[event_type] = [h for h in listeners if h is not handler]

    def once(self, event_type: str, handler: Callable[..., Any]) -> Callable[[], None]:
        cleanup: Callable[[], None] | None = None

        def _wrapper(**kwargs: Any) -> None:
            nonlocal cleanup
            if cleanup:
                cleanup()
            handler(**kwargs)

        cleanup = self.on(event_type, _wrapper)
        return cleanup

    # -- emit ---------------------------------------------------------------

    def emit(self, event_type: str, **kwargs: Any) -> None:
        depth = getattr(self._recursion_depth, "value", 0)
        if depth >= _MAX_EMIT_DEPTH:
            with self._lock:
                self._metrics["recursive_drops"] += 1
            logger.error(
                "EventBus: recursive emit depth %d for %s — dropping to prevent stack overflow",
                depth, event_type,
            )
            return

        self._recursion_depth.value = depth + 1
        try:
            self._emit_inner(event_type, **kwargs)
        finally:
            self._recursion_depth.value = depth

    def _emit_inner(self, event_type: str, **kwargs: Any) -> None:
        with self._lock:
            self._metrics["total_events"] += 1

            if self._barrier == _BarrierState.CLOSED:
                is_boot = event_type.startswith("kernel:boot") or event_type.startswith("system:")
                if not is_boot:
                    if len(self._early_queue) < self._max_buffered:
                        self._early_queue.append((event_type, kwargs))
                    return

            if self._is_breaker_open(event_type):
                return

            if self._event_validator is not None:
                try:
                    violation = self._event_validator.validate(event_type, kwargs)
                except TypeError:
                    # Backward compatibility for custom validators that still
                    # implement validate(event_type) only.
                    violation = self._event_validator.validate(event_type)
                if violation and violation.severity == "critical":
                    self._metrics.setdefault("blocked_events", 0)
                    self._metrics["blocked_events"] += 1
                    return

            handlers = list(self._listeners.get(event_type, []))

        for handler in handlers:
            t0 = _time.monotonic()
            try:
                handler(**kwargs)
            except Exception:
                logger.exception("EventBus handler error for %s", event_type)
                with self._lock:
                    self._metrics["failed_events"] += 1
                    now = _time.monotonic()
                    errors = self._error_counts.setdefault(event_type, [])
                    errors.append(now)
                    errors[:] = [t for t in errors if now - t <= 60.0]
                    if len(errors) >= self.CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_breakers[event_type] = now + self.CIRCUIT_BREAKER_TIMEOUT
                        self._metrics["circuit_breaker_trips"] += 1
                        logger.warning(
                            "EventBus: circuit breaker tripped for %s (%d errors/min)",
                            event_type, len(errors),
                        )
                if event_type in CRITICAL_EVENTS:
                    self._schedule_retry(
                        event_type,
                        kwargs,
                        0,
                        reason="handler_exception",
                    )
            finally:
                elapsed_ms = (_time.monotonic() - t0) * 1000.0
                with self._lock:
                    self._processing_times.append(elapsed_ms)

    # -- circuit breaker ----------------------------------------------------

    def _is_breaker_open(self, event_type: str) -> bool:
        close_at = self._circuit_breakers.get(event_type)
        if close_at is None:
            return False
        if _time.monotonic() >= close_at:
            del self._circuit_breakers[event_type]
            logger.info("EventBus: circuit breaker auto-closed for %s", event_type)
            return False
        return True

    # -- retry scheduling ---------------------------------------------------

    @staticmethod
    def _retry_lineage(kwargs: dict[str, Any]) -> dict[str, str]:
        """Extract stable correlation keys for retry telemetry."""
        fields = ("conversation_id", "output_id", "trace_id", "request_id")
        lineage: dict[str, str] = {}
        for key in fields:
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                lineage[key] = value
        return lineage

    def _schedule_retry(
        self,
        event_type: str,
        kwargs: dict[str, Any],
        attempt: int,
        *,
        retry_id: str = "",
        reason: str = "",
    ) -> None:
        retry_id = retry_id or f"rty_{_time.monotonic_ns()}"
        lineage = self._retry_lineage(kwargs)
        if attempt >= self.MAX_RETRIES:
            logger.error(
                "EventBus: exhausted %d retries for critical event %s",
                self.MAX_RETRIES, event_type,
            )
            self.emit(
                RETRY_EXHAUSTED,
                retry_id=retry_id,
                target_event_type=event_type,
                attempt=attempt,
                max_retries=self.MAX_RETRIES,
                reason=reason or "max_retries_reached",
                **lineage,
            )
            return
        delay = self.RETRY_DELAYS[attempt] if attempt < len(self.RETRY_DELAYS) else self.RETRY_DELAYS[-1]
        with self._lock:
            self._metrics["retried_events"] += 1
        self.emit(
            RETRY_SCHEDULED,
            retry_id=retry_id,
            target_event_type=event_type,
            attempt=attempt + 1,
            max_retries=self.MAX_RETRIES,
            delay_s=delay,
            reason=reason or "critical_handler_failed",
            **lineage,
        )
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_later(
                    delay,
                    self._execute_retry,
                    event_type,
                    kwargs,
                    attempt + 1,
                    retry_id,
                )
                return
        except Exception:
            pass
        import threading
        threading.Timer(
            delay,
            self._execute_retry,
            args=(event_type, kwargs, attempt + 1, retry_id),
        ).start()

    def _execute_retry(
        self,
        event_type: str,
        kwargs: dict[str, Any],
        attempt: int,
        retry_id: str,
    ) -> None:
        lineage = self._retry_lineage(kwargs)
        self.emit(
            RETRY_EXECUTED,
            retry_id=retry_id,
            target_event_type=event_type,
            attempt=attempt,
            max_retries=self.MAX_RETRIES,
            **lineage,
        )
        logger.info("EventBus: retrying %s (attempt %d/%d)", event_type, attempt, self.MAX_RETRIES)
        try:
            self.emit(event_type, **kwargs)
        except Exception:
            logger.exception("EventBus: retry failed for %s (attempt %d)", event_type, attempt)
            self._schedule_retry(
                event_type,
                kwargs,
                attempt,
                retry_id=retry_id,
                reason="retry_execution_failed",
            )

    # -- metrics ------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            metrics: dict[str, Any] = dict(self._metrics)
            if self._processing_times:
                metrics["avg_processing_time_ms"] = sum(self._processing_times) / len(self._processing_times)
            else:
                metrics["avg_processing_time_ms"] = 0.0
        total = metrics["total_events"]
        metrics["error_rate"] = metrics["failed_events"] / total if total > 0 else 0.0
        return metrics

    # -- barrier ------------------------------------------------------------

    def open_barrier(self) -> None:
        with self._lock:
            if self._barrier == _BarrierState.OPEN:
                return
            if self._event_validator is None:
                try:
                    from consciousness.event_validator import event_validator
                    self._event_validator = event_validator
                except Exception:
                    pass
            self._barrier = _BarrierState.OPENING
            queued = list(self._early_queue)
            self._early_queue.clear()
        for event_type, kwargs in queued:
            self.emit(event_type, **kwargs)
        with self._lock:
            self._barrier = _BarrierState.OPEN
        logger.info("EventBus: barrier open, flushed %d buffered events", len(queued))

    def reset_barrier(self) -> None:
        with self._lock:
            self._barrier = _BarrierState.CLOSED
            self._early_queue.clear()

    # -- introspection ------------------------------------------------------

    def listener_count(self, event_type: str | None = None) -> int:
        with self._lock:
            if event_type:
                return len(self._listeners.get(event_type, []))
            return sum(len(v) for v in self._listeners.values())

    def remove_all_listeners(self, event_type: str | None = None) -> None:
        with self._lock:
            if event_type:
                self._listeners.pop(event_type, None)
            else:
                self._listeners.clear()

    def set_validator(self, validator: Any) -> None:
        with self._lock:
            self._event_validator = validator


event_bus = EventBus.instance()
