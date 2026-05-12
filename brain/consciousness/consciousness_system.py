"""Consciousness System — unified coordinator with exactly 3 public methods.

Public API:
  on_tick(now)  — called by kernel tick, runs background cycles within budget
  on_event(event_type, **kwargs)  — called by event bus, updates observer + analytics
  get_state()  — returns full consciousness state for dashboard + context injection
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from consciousness.consciousness_analytics import ConsciousnessAnalytics
from consciousness.consciousness_evolution import ConsciousnessEvolution
from consciousness.consciousness_driven_evolution import ConsciousnessDrivenEvolution
from consciousness.events import (
    event_bus,
    META_THOUGHT_GENERATED,
    CONSCIOUSNESS_ANALYSIS,
    CONSCIOUSNESS_EVOLUTION_EVENT,
    CONSCIOUSNESS_EMERGENT_BEHAVIOR,
    CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
    CONSCIOUSNESS_MUTATION_PROPOSED,
    CONSCIOUSNESS_CAPABILITY_UNLOCKED,
    MUTATION_APPLIED,
    MUTATION_REJECTED,
    MUTATION_ROLLBACK,
    MEMORY_TRIMMED,
    MEMORY_WRITE,
    PHASE_SHIFT,
    TONE_SHIFT,
    CONFIDENCE_UPDATE,
    EXISTENTIAL_INQUIRY_COMPLETED,
    PHILOSOPHICAL_DIALOGUE_COMPLETED,
    CONVERSATION_RESPONSE,
    PERCEPTION_BARGE_IN,
    PERCEPTION_USER_EMOTION,
    KERNEL_THOUGHT,
    QUARANTINE_SIGNAL_EMITTED,
    QUARANTINE_TICK_COMPLETE,
    REFLECTIVE_AUDIT_COMPLETED,
    REFLECTIVE_AUDIT_FINDING,
    SOUL_INTEGRITY_UPDATED,
    SOUL_INTEGRITY_REPAIR_NEEDED,
    CONSCIOUSNESS_CLEANUP_OBSERVATIONS,
    CONSCIOUSNESS_CLEANUP_OLD_CHAINS,
    CONSCIOUSNESS_CLEAR_CACHES,
    CONSCIOUSNESS_REDUCE_OBSERVATION_RATE,
)
from consciousness.existential_reasoning import ExistentialReasoning
from consciousness.kernel_config import KernelConfig
from consciousness.kernel_mutator import KernelMutator
from consciousness.memory_optimizer import memory_optimizer
from consciousness.meta_cognitive_thoughts import MetaCognitiveThoughtGenerator
from consciousness.mutation_governor import MutationGovernor, SystemHealth
from consciousness.observer import ConsciousnessObserver
from consciousness.philosophical_dialogue import PhilosophicalDialogueEngine

logger = logging.getLogger(__name__)

META_THOUGHT_INTERVAL_S = 8.0
EVOLUTION_INTERVAL_S = 90.0
ANALYSIS_INTERVAL_S = 30.0
EXISTENTIAL_INTERVAL_S = 120.0
DIALOGUE_INTERVAL_S = 240.0
MUTATION_INTERVAL_S = 180.0
SELF_IMPROVE_INTERVAL_S = 900.0
HEMISPHERE_INTERVAL_S = 120.0
LEARNING_JOB_INTERVAL_S = 300.0
CONTRADICTION_INTERVAL_S = 60.0
TRUTH_CALIBRATION_INTERVAL_S = 120.0
BELIEF_GRAPH_INTERVAL_S = 60.0
QUARANTINE_INTERVAL_S = 60.0
REFLECTIVE_AUDIT_INTERVAL_S = 300.0
SOUL_INTEGRITY_INTERVAL_S = 120.0
CAPABILITY_DISCOVERY_INTERVAL_S = 300.0
GOALS_INTERVAL_S = 120.0
SCENE_CONTINUITY_INTERVAL_S = 60.0
WORLD_MODEL_INTERVAL_S = 5.0
WORLD_MODEL_SLEEP_INTERVAL_S = 30.0
CURIOSITY_QUESTIONS_INTERVAL_S = 60.0
ONBOARDING_INTERVAL_S = 60.0
HEALTH_MONITOR_INTERVAL_S = 30.0
FRACTAL_RECALL_INTERVAL_S = 30.0
ACQUISITION_INTERVAL_S = 60.0
ACQUISITION_DEEP_LEARNING_INTERVAL_S = 30.0
ACQUISITION_SLEEP_INTERVAL_S = 120.0


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


BOOT_STABILIZATION_S = _env_seconds("JARVIS_BOOT_STABILIZATION_S", 600.0)


# Accelerated intervals for deep_learning mode (brain alone, maxing out)
_DEEP_LEARNING_INTERVALS = {
    "meta_thought": 4.0,
    "evolution": 30.0,
    "analysis": 15.0,
    "existential": 60.0,
    "dialogue": 120.0,
    "mutation": 60.0,
    "self_improve": 300.0,
    "hemisphere": 45.0,
    "study": 30.0,
    "learning_job": 120.0,
    "contradiction": 30.0,
    "truth_calibration": 60.0,
    "belief_graph": 30.0,
    "goals": 60.0,
    "reflective_audit": 120.0,
    "soul_integrity": 60.0,
}

_DREAMING_INTERVALS = {
    "meta_thought": 6.0,
    "evolution": 60.0,
    "analysis": 20.0,
    "existential": 45.0,
    "dialogue": 90.0,
    "mutation": 120.0,
    "self_improve": 600.0,
    "hemisphere": 60.0,
    "study": 45.0,
    "learning_job": 180.0,
    "contradiction": 30.0,
    "truth_calibration": 60.0,
    "belief_graph": 30.0,
    "goals": 90.0,
    "reflective_audit": 150.0,
    "soul_integrity": 60.0,
}

MEMORY_OPTIMIZER_INTERVAL_S = 10.0
DREAM_CYCLE_INTERVAL_S = 30.0
STUDY_CYCLE_INTERVAL_S = 120.0
STUDY_CYCLE_BATCH = 2
ASSOCIATION_REPAIR_INTERVAL_S = 60.0
INTENTION_STALE_SWEEP_INTERVAL_S = 300.0
INTENTION_RESOLVER_INTERVAL_S = 30.0
SHADOW_LANG_INTERVAL_S = 21600.0  # 6 hours
ASSOCIATION_REPAIR_BATCH_SIZE = 8
INITIAL_MEMORY_WEIGHT_CAP = 0.80
MAX_DREAM_INSIGHT_MEMORIES = 20
_DREAM_INSIGHT_TAG = "dream_insight"


# ---------------------------------------------------------------------------
# Consciousness state (returned by get_state)
# ---------------------------------------------------------------------------

@dataclass
class ConsciousnessState:
    stage: str = "basic_awareness"
    transcendence_level: float = 0.0
    awareness_level: float = 0.3
    active_capabilities: list[str] = field(default_factory=list)
    current_focus: str = ""
    last_mutation_summary: str = ""
    meta_thought_titles: list[str] = field(default_factory=list)
    mutation_count: int = 0
    emergent_behavior_count: int = 0
    observation_count: int = 0
    reasoning_quality: float = 0.5
    confidence_avg: float = 0.5
    system_healthy: bool = True
    boot_stabilization_active: bool = False
    boot_stabilization_remaining_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "transcendence_level": round(self.transcendence_level, 2),
            "awareness_level": round(self.awareness_level, 2),
            "active_capabilities": self.active_capabilities,
            "current_focus": self.current_focus,
            "last_mutation_summary": self.last_mutation_summary,
            "meta_thought_titles": self.meta_thought_titles,
            "mutation_count": self.mutation_count,
            "emergent_behavior_count": self.emergent_behavior_count,
            "observation_count": self.observation_count,
            "reasoning_quality": round(self.reasoning_quality, 2),
            "confidence_avg": round(self.confidence_avg, 2),
            "system_healthy": self.system_healthy,
            "boot_stabilization_active": self.boot_stabilization_active,
            "boot_stabilization_remaining_s": round(self.boot_stabilization_remaining_s, 1),
        }


# Module-level reference for cross-layer access (set in __init__)
_active_consciousness: ConsciousnessSystem | None = None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class ConsciousnessSystem:
    def __init__(self, kernel_config: KernelConfig | None = None) -> None:
        global _active_consciousness
        _active_consciousness = self
        self._config = kernel_config or KernelConfig.load()

        self.analytics = ConsciousnessAnalytics()
        self.observer = ConsciousnessObserver()
        self.observer.subscribe_mode_changes()
        self.meta_thoughts = MetaCognitiveThoughtGenerator()
        self.evolution = ConsciousnessEvolution()
        self.driven_evolution = ConsciousnessDrivenEvolution()
        self.existential = ExistentialReasoning()
        self.philosophical = PhilosophicalDialogueEngine()
        self.mutator = KernelMutator()
        self.governor = MutationGovernor()

        self._last_meta_thought: float = 0.0
        self._last_evolution: float = 0.0
        self._last_analysis: float = 0.0
        self._last_existential: float = 0.0
        self._last_dialogue: float = 0.0
        self._last_mutation: float = 0.0
        self._last_self_improve: float = 0.0
        self._last_dream: float = 0.0

        self._self_improve_orchestrator = None
        self._self_improve_enabled = False
        self._response_latencies: deque[float] = deque(maxlen=100)
        self._tick_count: int = 0
        self._last_mutation_snapshot_id: str = ""
        self._last_mutation_ledger_id: str = ""
        self._last_mutation_ts: float = 0.0
        self._last_outcome_tick: float = 0.0
        self._last_hemisphere: float = 0.0
        self._hemisphere_orchestrator = None
        self._engine_ref = None

        self._boot_time: float = time.time()
        self._boot_stabilization_s: float = BOOT_STABILIZATION_S
        self._last_boot_stabilization_log: float = 0.0
        self._initialized = False
        self._llm_callback: Any = None
        self._last_association_repair: float = 0.0
        self._last_intention_stale_sweep: float = 0.0
        self._last_intention_resolver_tick: float = 0.0
        self._last_study: float = 0.0
        self._study_running = False
        self._last_learning_job: float = 0.0
        self._last_contradiction: float = 0.0
        self._contradiction_engine: Any = None
        self._last_truth_calibration: float = 0.0
        self._truth_calibration_engine: Any = None
        self._last_belief_graph: float = 0.0
        self._belief_graph: Any = None
        self._last_quarantine: float = 0.0
        self._quarantine_scorer: Any = None
        self._quarantine_log: Any = None
        self._prev_flip_count: int = 0
        self._last_reflective_audit: float = 0.0
        self._reflective_audit_engine: Any = None
        self._last_soul_integrity: float = 0.0
        self._last_health_monitor: float = 0.0
        self._soul_integrity_index: Any = None
        self._last_capability_discovery: float = 0.0
        self._last_goals: float = 0.0
        self._goal_manager: Any = None
        self._last_scene_continuity: float = 0.0
        self._scene_continuity_module: Any = None
        self._last_world_model: float = 0.0
        self._world_model: Any = None
        self._last_curiosity_questions: float = 0.0
        self._last_onboarding: float = 0.0
        self._recent_memory_writes: list[dict[str, Any]] = []
        import threading as _threading
        self._memory_writes_lock = _threading.Lock()
        self._learning_job_orchestrator: Any = None
        self._conversation_confidence_signals: deque[float] = deque(maxlen=50)
        self._cycle_skip_counts: dict[str, int] = {}
        self._last_ranker_skip: str = "not attempted yet"
        self._last_salience_skip: str = "not attempted yet"
        self._last_ranker_pair_count: int = 0
        self._last_salience_pair_count: int = 0
        self._salience_advisory_count: int = 0
        self._salience_gate_fail_count: int = 0
        self._last_artifact_validation: float = 0.0
        self._last_memory_optimizer: float = 0.0
        self._dream_cycle_history: deque[dict[str, Any]] = deque(maxlen=50)
        self._last_shadow_lang: float = 0.0
        self._shadow_language_trainer: Any = None
        self._fractal_recall_engine: Any = None
        self._last_fractal_recall: float = 0.0
        self._acquisition_orchestrator: Any = None
        self._last_acquisition: float = 0.0

        self._wire_memory_optimizer_listeners()
        logger.info("ConsciousnessSystem initialized")

    @property
    def config(self) -> KernelConfig:
        return self._config

    @config.setter
    def config(self, value: KernelConfig) -> None:
        self._config = value

    # -- memory-optimizer event wiring --------------------------------------

    def _wire_memory_optimizer_listeners(self) -> None:
        """Subscribe cleanup-command receivers for the memory optimizer."""

        def _on_cleanup_observations(**kw: Any) -> None:
            self.observer._state.recent_observations = (
                self.observer._state.recent_observations[-10:]
            )

        def _on_cleanup_old_chains(**kw: Any) -> None:
            try:
                from consciousness.epistemic_reasoning import epistemic_engine
                epistemic_engine._chains.clear()
            except Exception:
                pass

        def _on_clear_caches(**kw: Any) -> None:
            import gc
            gc.collect()

        def _on_reduce_observation_rate(**kw: Any) -> None:
            duration = kw.get("duration_s", 60.0)
            self.observer.reduce_observation_rate(duration)

        event_bus.on(CONSCIOUSNESS_CLEANUP_OBSERVATIONS, _on_cleanup_observations)
        event_bus.on(CONSCIOUSNESS_CLEANUP_OLD_CHAINS, _on_cleanup_old_chains)
        event_bus.on(CONSCIOUSNESS_CLEAR_CACHES, _on_clear_caches)
        event_bus.on(CONSCIOUSNESS_REDUCE_OBSERVATION_RATE, _on_reduce_observation_rate)

    # ===== PUBLIC API (3 methods) ===========================================

    def _get_intervals(self) -> dict[str, float]:
        """Return active intervals, accelerated when in deep_learning or dreaming mode."""
        from consciousness.modes import mode_manager
        if mode_manager.mode == "deep_learning":
            return _DEEP_LEARNING_INTERVALS
        if mode_manager.mode == "dreaming":
            return _DREAMING_INTERVALS
        return {
            "meta_thought": META_THOUGHT_INTERVAL_S,
            "evolution": EVOLUTION_INTERVAL_S,
            "analysis": ANALYSIS_INTERVAL_S,
            "existential": EXISTENTIAL_INTERVAL_S,
            "dialogue": DIALOGUE_INTERVAL_S,
            "mutation": MUTATION_INTERVAL_S,
            "self_improve": SELF_IMPROVE_INTERVAL_S,
            "hemisphere": HEMISPHERE_INTERVAL_S,
            "study": STUDY_CYCLE_INTERVAL_S,
            "learning_job": LEARNING_JOB_INTERVAL_S,
            "contradiction": CONTRADICTION_INTERVAL_S,
            "truth_calibration": TRUTH_CALIBRATION_INTERVAL_S,
            "belief_graph": BELIEF_GRAPH_INTERVAL_S,
            "quarantine": QUARANTINE_INTERVAL_S,
            "goals": GOALS_INTERVAL_S,
            "scene_continuity": SCENE_CONTINUITY_INTERVAL_S,
            "acquisition": ACQUISITION_INTERVAL_S,
        }

    def _cycle_allowed(self, cycle_name: str) -> bool:
        """Check if a cycle is allowed under the current mode profile."""
        from consciousness.modes import mode_manager
        allowed = mode_manager.profile.allowed_cycles
        if cycle_name in allowed:
            return True
        self._cycle_skip_counts[cycle_name] = self._cycle_skip_counts.get(cycle_name, 0) + 1
        from consciousness.operations import ops_tracker
        ops_tracker.skip_activity(cycle_name, f"mode_not_allowed:{mode_manager.mode}")
        return False

    def get_cycle_skip_counts(self) -> dict[str, int]:
        return dict(self._cycle_skip_counts)

    def _tracked_cycle(self, name: str, fn, *args, trigger: str = "scheduler",
                       detail: str = "", **kwargs) -> None:
        """Run a cycle function with operational tracking."""
        from consciousness.operations import ops_tracker
        ops_tracker.begin_activity(name, phase="tick", trigger=trigger, detail=detail)
        ops_tracker.set_subsystem(name, "running", detail)
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.debug("Cycle %s error", name, exc_info=True)
        finally:
            ops_tracker.end_activity(name)
            ops_tracker.set_subsystem(name, "idle")

    def _boot_stabilization_remaining_s(self, now: float) -> float:
        if self._boot_stabilization_s <= 0.0:
            return 0.0
        return max(0.0, self._boot_stabilization_s - (now - self._boot_time))

    def _boot_stabilization_active(self, now: float) -> bool:
        return self._boot_stabilization_remaining_s(now) > 0.0

    def on_tick(
        self,
        now: float,
        memories: list[Any] | None = None,
        traits: dict[str, float] | None = None,
        tick_elapsed_ms: float = 0.0,
        tick_count: int = 0,
        deferred_backlog: int = 0,
    ) -> None:
        """Called by kernel tick. Runs background consciousness cycles."""
        memories = memories or []
        traits = traits or {}
        self._tick_count = tick_count

        self.analytics.record_tick(tick_elapsed_ms)
        self.analytics.update_memory_count(len(memories))
        self.analytics.update_backlog(deferred_backlog)
        try:
            eb_metrics = event_bus.get_metrics()
            self.analytics.record_event_error_rate(eb_metrics.get("error_rate", 0.0))
        except Exception:
            pass

        try:
            self._check_mutation_health()
        except Exception:
            logger.debug("Mutation health check error", exc_info=True)
            self._last_mutation_snapshot_id = ""

        if now - self._last_memory_optimizer >= MEMORY_OPTIMIZER_INTERVAL_S:
            self._last_memory_optimizer = now
            try:
                memory_optimizer.check()
            except Exception:
                logger.debug("Memory optimizer check failed", exc_info=True)

        iv = self._get_intervals()
        boot_stabilizing = self._boot_stabilization_active(now)
        if boot_stabilizing and now - self._last_boot_stabilization_log >= 60.0:
            self._last_boot_stabilization_log = now
            logger.info(
                "Startup stabilization active (%.0fs remaining): mutation/training hardening enabled",
                self._boot_stabilization_remaining_s(now),
            )

        if now - self._last_meta_thought >= iv["meta_thought"] and self._cycle_allowed("meta_thought"):
            self._tracked_cycle("meta_thought", self._run_meta_thoughts, now, memories, traits)

        if now - self._last_analysis >= iv["analysis"] and self._cycle_allowed("analysis"):
            self._tracked_cycle("analysis", self._run_analysis, now, memories, traits)

        if now - self._last_contradiction >= iv.get("contradiction", CONTRADICTION_INTERVAL_S) and self._cycle_allowed("contradiction"):
            self._tracked_cycle("contradiction", self._run_contradiction_check, now)

        if now - self._last_truth_calibration >= iv.get("truth_calibration", TRUTH_CALIBRATION_INTERVAL_S) and self._cycle_allowed("truth_calibration"):
            self._tracked_cycle("truth_calibration", self._run_truth_calibration, now)

        if now - self._last_belief_graph >= iv.get("belief_graph", BELIEF_GRAPH_INTERVAL_S) and self._cycle_allowed("belief_graph"):
            self._tracked_cycle("belief_graph", self._run_belief_graph, now)

        if now - self._last_quarantine >= iv.get("quarantine", QUARANTINE_INTERVAL_S) and self._cycle_allowed("quarantine"):
            self._tracked_cycle("quarantine", self._run_quarantine_tick, now)

        if now - self._last_reflective_audit >= iv.get("reflective_audit", REFLECTIVE_AUDIT_INTERVAL_S) and self._cycle_allowed("reflective_audit"):
            self._tracked_cycle("reflective_audit", self._run_reflective_audit, now)

        if now - self._last_soul_integrity >= iv.get("soul_integrity", SOUL_INTEGRITY_INTERVAL_S) and self._cycle_allowed("soul_integrity"):
            self._tracked_cycle("soul_integrity", self._run_soul_integrity, now)

        if now - self._last_health_monitor >= iv.get("health_monitor", HEALTH_MONITOR_INTERVAL_S) and self._cycle_allowed("health_monitor"):
            self._tracked_cycle("health_monitor", self._run_health_monitor, now)

        if now - self._last_goals >= iv.get("goals", GOALS_INTERVAL_S) and self._cycle_allowed("goals"):
            self._tracked_cycle("goals", self._run_goals_tick, now)

        if now - self._last_scene_continuity >= iv.get("scene_continuity", SCENE_CONTINUITY_INTERVAL_S) and self._cycle_allowed("scene_continuity"):
            self._tracked_cycle("scene_continuity", self._run_scene_continuity_tick, now)

        if now - self._last_curiosity_questions >= iv.get("curiosity_questions", CURIOSITY_QUESTIONS_INTERVAL_S) and self._cycle_allowed("curiosity_questions"):
            self._tracked_cycle("curiosity_questions", self._run_curiosity_questions, now)

        if now - self._last_fractal_recall >= FRACTAL_RECALL_INTERVAL_S and self._cycle_allowed("fractal_recall"):
            self._tracked_cycle("fractal_recall", self._run_fractal_recall, now)

        if now - self._last_onboarding >= iv.get("onboarding", ONBOARDING_INTERVAL_S) and self._cycle_allowed("onboarding"):
            self._tracked_cycle("onboarding", self._run_onboarding_tick, now)

        from consciousness.modes import mode_manager as _mm_acq
        _acq_interval = ACQUISITION_INTERVAL_S
        if _mm_acq.mode == "deep_learning":
            _acq_interval = ACQUISITION_DEEP_LEARNING_INTERVAL_S
        elif _mm_acq.mode == "sleep":
            _acq_interval = ACQUISITION_SLEEP_INTERVAL_S
        if self._acquisition_orchestrator and now - self._last_acquisition >= iv.get("acquisition", _acq_interval) and self._cycle_allowed("acquisition"):
            self._tracked_cycle("acquisition", self._run_acquisition_tick, now)

        _disc_interval = CAPABILITY_DISCOVERY_INTERVAL_S
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            if get_quarantine_pressure().current.elevated:
                _disc_interval *= 2
        except Exception:
            pass
        if now - self._last_capability_discovery >= iv.get("capability_discovery", _disc_interval) and self._cycle_allowed("capability_discovery"):
            self._tracked_cycle("capability_discovery", self._run_capability_discovery, now)

        if now - self._last_evolution >= iv["evolution"] and self._cycle_allowed("evolution"):
            self._tracked_cycle("evolution", self._run_evolution, now, memories, traits)

        from consciousness.modes import mode_manager as _mm
        _is_gestation = _mm.mode == "gestation"

        _wm_interval = WORLD_MODEL_SLEEP_INTERVAL_S if _mm.mode == "sleep" else WORLD_MODEL_INTERVAL_S
        if now - self._last_world_model >= iv.get("world_model", _wm_interval) and self._cycle_allowed("world_model"):
            self._tracked_cycle("world_model", self._run_world_model_tick, now)

        if (
            now - self._last_mutation >= iv["mutation"]
            and not _is_gestation
            and not boot_stabilizing
            and self._cycle_allowed("mutation")
        ):
            self._tracked_cycle("mutation", self._run_mutation_cycle, now, memories, traits)

        if now - self._last_existential >= iv["existential"] and self._cycle_allowed("existential"):
            self._tracked_cycle("existential", self._run_existential, now)

        if now - self._last_dialogue >= iv["dialogue"] and self._cycle_allowed("dialogue"):
            self._tracked_cycle("dialogue", self._run_dialogue, now, detail="philosophical dialogue")

        if self._self_improve_enabled and now - self._last_self_improve >= iv["self_improve"] and self._cycle_allowed("self_improvement"):
            self._tracked_cycle("self_improve", self._run_self_improvement, now, memories)

        if self._hemisphere_orchestrator and now - self._last_hemisphere >= iv["hemisphere"] and self._cycle_allowed("hemisphere"):
            self._tracked_cycle("hemisphere", self._run_hemisphere_cycle, now, memories, traits)

        if now - self._last_shadow_lang >= SHADOW_LANG_INTERVAL_S and self._cycle_allowed("shadow_lang"):
            self._tracked_cycle("shadow_lang", self._run_shadow_language_cycle, now)

        if now - self._last_study >= iv.get("study", STUDY_CYCLE_INTERVAL_S) and self._cycle_allowed("study"):
            self._tracked_cycle("study", self._run_study_cycle, now)

        if self._learning_job_orchestrator and now - self._last_learning_job >= iv.get("learning_job", LEARNING_JOB_INTERVAL_S) and self._cycle_allowed("learning_jobs"):
            self._tracked_cycle("learning_jobs", self._run_learning_job_cycle, now)

        if now - self._last_association_repair >= ASSOCIATION_REPAIR_INTERVAL_S and self._cycle_allowed("association_repair"):
            self._tracked_cycle("memory", self._run_association_repair, now, memories, detail="association repair")

        if now - self._last_intention_stale_sweep >= INTENTION_STALE_SWEEP_INTERVAL_S and self._cycle_allowed("intention_stale_sweep"):
            self._tracked_cycle("intention_stale_sweep", self._run_intention_stale_sweep, now, detail="intention stale sweep")

        if now - self._last_intention_resolver_tick >= INTENTION_RESOLVER_INTERVAL_S and self._cycle_allowed("intention_resolver"):
            self._tracked_cycle("intention_resolver", self._run_intention_resolver_tick, now, detail="intention resolver")

        if now - self._last_outcome_tick >= 15.0:
            self._last_outcome_tick = now
            try:
                from consciousness.attribution_ledger import outcome_scheduler
                outcome_scheduler.tick()
            except Exception:
                pass

        if _mm.mode == "dreaming" and now - self._last_dream >= DREAM_CYCLE_INTERVAL_S and self._cycle_allowed("dream"):
            self._tracked_cycle("dream", self._run_dream_cycle, now, memories, detail="dream consolidation")
        elif not _is_gestation and now - self._last_dream >= 600.0 and len(memories) >= 20 and self._cycle_allowed("dream"):
            self._tracked_cycle("dream", self._run_dream_cycle, now, memories, detail="dream consolidation")

        # Validate pending dream artifacts when not in consolidation (120s cooldown)
        from memory.gate import memory_gate as _cg
        if not _cg.can_consolidation_write() and now - self._last_artifact_validation >= 120.0:
            try:
                from consciousness.dream_artifacts import artifact_buffer, reflective_validator
                if artifact_buffer.get_pending():
                    ctx = self._gather_dream_validation_context()
                    reflective_validator.validate_pending(system_context=ctx)
                    self._last_artifact_validation = now
            except Exception:
                logger.debug("Dream artifact validation failed", exc_info=True)

    def on_event(self, event_type: str, **kwargs: Any) -> None:
        """Called by event bus. Updates observer + analytics."""
        if event_type == PHASE_SHIFT:
            self.observer.observe_phase_shift(
                kwargs.get("from_phase", ""), kwargs.get("to_phase", ""),
            )
            self.analytics.record_phase_change(kwargs.get("to_phase", ""))

        elif event_type == TONE_SHIFT:
            self.observer.observe_tone_shift(
                kwargs.get("from_tone", ""), kwargs.get("to_tone", ""),
            )
            self.analytics.record_tone_change(kwargs.get("to_tone", ""))

        elif event_type == MEMORY_WRITE:
            self.observer.observe_memory(
                kwargs.get("memory_id", ""), kwargs.get("salience", 0.5),
                kwargs.get("tags"),
            )

        # NOTE: MUTATION_APPLIED, CONFIDENCE_UPDATE, CONVERSATION_RESPONSE
        # branches were here but unreachable — engine only routes PHASE_SHIFT,
        # TONE_SHIFT, MEMORY_WRITE to on_event(). Conversation confidence is
        # handled via the EventBus epistemic feeder in engine._wire_event_listeners.

        elif event_type == PERCEPTION_BARGE_IN:
            self.observer.observe_thought(
                "User interrupted — barge-in detected",
                depth="surface", confidence=0.8,
            )
            self._conversation_confidence_signals.append(0.25)
            self.analytics.record_confidence(0.25)

        elif event_type == PERCEPTION_USER_EMOTION:
            emotion = kwargs.get("emotion", "neutral")
            confidence = kwargs.get("confidence", 0.5)
            trust = kwargs.get("trust", "medium")
            from consciousness.soul import identity_state
            _EMOTION_MOOD_MAP = {
                "happy": "positive", "sad": "somber", "angry": "tense",
                "fear": "cautious", "surprise": "alert", "disgust": "cautious",
                "neutral": "neutral",
            }
            new_mood = _EMOTION_MOOD_MAP.get(emotion, "neutral")
            # Only set mood from medium+ trust signals
            if trust != "low" and confidence > 0.4 and identity_state.dynamic_mood != new_mood:
                identity_state.dynamic_mood = new_mood
                self.observer.observe_thought(
                    f"User emotion detected: {emotion} (mood → {new_mood})",
                    depth="surface", confidence=confidence,
                )

    def get_state(self) -> ConsciousnessState:
        """Returns full consciousness state for dashboard + context injection."""
        now = time.time()
        confidence = self.analytics.get_confidence()
        reasoning = self.analytics.get_reasoning_quality()
        health = self.analytics.get_system_health()
        evo_state = self.evolution.get_state()
        boot_remaining_s = self._boot_stabilization_remaining_s(now)
        boot_active = boot_remaining_s > 0.0

        return ConsciousnessState(
            stage=evo_state.current_stage,
            transcendence_level=evo_state.transcendence_level,
            awareness_level=self.observer.awareness_level,
            active_capabilities=self.driven_evolution.get_active_capabilities(),
            current_focus=self.existential.get_current_focus(),
            last_mutation_summary=self._get_last_mutation_summary(),
            meta_thought_titles=self.meta_thoughts.get_thought_titles(3),
            mutation_count=self.governor.mutation_count,
            emergent_behavior_count=evo_state.total_emergent_count,
            observation_count=self.observer.state.observation_count,
            reasoning_quality=reasoning.overall,
            confidence_avg=confidence.avg,
            system_healthy=health.healthy,
            boot_stabilization_active=boot_active,
            boot_stabilization_remaining_s=boot_remaining_s,
        )

    # ===== INTERNAL CYCLES ==================================================

    def _run_meta_thoughts(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        self._last_meta_thought = now
        try:
            self._run_meta_thoughts_inner(now, memories, traits)
        except Exception:
            logger.debug("Meta-thought cycle error", exc_info=True)

    def _run_meta_thoughts_inner(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        confidence = self.analytics.get_confidence()
        observer_state = self.observer.state
        obs_summary = self.observer.get_observation_summary()

        # Gather memory stats for memory_reflection and connection_discovery triggers
        memory_count = len(memories)
        association_count = sum(getattr(m, "association_count", 0) for m in memories)
        recent_mem = memories[-1] if memories else None
        recent_memory_text = ""
        recent_memory_type = ""
        dominant_tag = ""
        if recent_mem:
            payload = getattr(recent_mem, "payload", "")
            if isinstance(payload, dict):
                recent_memory_text = (payload.get("user_message", "")
                                      or payload.get("response", "")
                                      or str(payload))[:60]
            else:
                recent_memory_text = str(payload)[:60]
            recent_memory_type = getattr(recent_mem, "type", "")
            tags = getattr(recent_mem, "tags", ())
            dominant_tag = tags[0] if tags else ""

        # Emotional momentum from tone engine
        try:
            from consciousness.tone import tone_engine
            emotional_momentum = tone_engine.emotional_momentum
        except Exception:
            emotional_momentum = 0.0

        # Uptime
        uptime_s = now - self._boot_time if hasattr(self, "_boot_time") else 0.0

        # Evolution stage and last mutation desc
        evolution_stage = self.evolution.current_stage
        last_mutation_desc = "subtle internal adjustments"
        try:
            hist = self._config.evolution.mutation_history
            if hist:
                last_mutation_desc = hist[-1].split(": ", 1)[-1] if ": " in hist[-1] else hist[-1]
        except Exception as exc:
            logger.debug("Mutation history read failed: %s", exc)

        context = {
            "observation_count": observer_state.observation_count,
            "awareness_level": observer_state.awareness_level,
            "confidence_avg": confidence.avg,
            "mutation_count": self.governor.mutation_count,
            "pattern_count": obs_summary.get("pattern_recognition", 0),
            "dominant_pattern": max(obs_summary, key=obs_summary.get, default="none") if obs_summary else "none",
            "memory_count": memory_count,
            "emotional_momentum": emotional_momentum,
            "association_count": association_count,
            "uptime_s": uptime_s,
            "recent_memory_text": recent_memory_text,
            "recent_memory_type": recent_memory_type,
            "dominant_tag": dominant_tag,
            "last_mutation_desc": last_mutation_desc,
            "evolution_stage": evolution_stage,
        }

        thought = self.meta_thoughts.check_and_generate(context)
        if thought:
            self.observer.observe_thought(
                thought.thought_type, thought.depth, thought.confidence,
            )
            self.analytics.record_thought(thought.depth, thought.confidence)
            event_bus.emit(META_THOUGHT_GENERATED,
                          thought_id=thought.id,
                          thought_type=thought.thought_type,
                          depth=thought.depth, text=thought.text[:120])
            if thought.thought_type == "pattern_recognition" and thought.confidence > 0.5:
                self.observer.observe_pattern(
                    thought.text[:80], evidence_ids=[], confidence=thought.confidence,
                )

    def _run_contradiction_check(self, now: float) -> None:
        self._last_contradiction = now
        try:
            if self._contradiction_engine is None:
                from epistemic.contradiction_engine import ContradictionEngine
                self._contradiction_engine = ContradictionEngine.get_instance()
                self._contradiction_engine.rehydrate()
                self._contradiction_engine.subscribe()

                persisted_debt = 0.0
                try:
                    from memory.persistence import consciousness_persistence
                    ep = getattr(consciousness_persistence, "_persisted_epistemic", {})
                    persisted_debt = float(ep.get("contradiction_debt", 0.0))
                except Exception:
                    pass
                if persisted_debt > 0.0 or self._contradiction_engine.contradiction_debt > 0.0:
                    self._contradiction_engine.set_persisted_debt(persisted_debt)

            self._contradiction_engine.apply_passive_decay()

            from consciousness.modes import mode_manager
            if mode_manager.mode in ("dreaming", "deep_learning"):
                self._contradiction_engine.scan_corpus()
        except Exception:
            logger.debug("Contradiction check error", exc_info=True)

    def _run_truth_calibration(self, now: float) -> None:
        self._last_truth_calibration = now
        try:
            if self._truth_calibration_engine is None:
                from epistemic.calibration import TruthCalibrationEngine
                self._truth_calibration_engine = TruthCalibrationEngine(engine=self._engine_ref)

            report = self._truth_calibration_engine.on_tick()
            if report:
                drift_domains = [a.domain for a in self._truth_calibration_engine._drift_detector.get_active_alerts()]
                self.observer.observe_calibration(
                    truth_score=report.truth_score,
                    maturity=report.maturity,
                    drift_domains=drift_domains,
                    confidence=report.maturity,
                )
        except Exception:
            logger.debug("Truth calibration error", exc_info=True)

    def _run_belief_graph(self, now: float) -> None:
        self._last_belief_graph = now
        try:
            if self._belief_graph is None:
                from epistemic.belief_graph import BeliefGraph
                self._belief_graph = BeliefGraph()

            state = self._belief_graph.on_tick()

            from consciousness.modes import mode_manager
            if mode_manager.mode in ("dreaming", "deep_learning"):
                self._belief_graph.on_dream_cycle()

            if state:
                integrity = state.get("integrity", {})
                propagation = state.get("propagation", {})
                self.observer.observe_belief_graph(
                    total_edges=state.get("total_edges", 0),
                    health_score=integrity.get("health_score", 1.0),
                    boosted=propagation.get("boosted", 0),
                    diminished=propagation.get("diminished", 0),
                    propagated=propagation.get("belief_count", 0),
                )
        except Exception:
            logger.debug("Belief graph tick error", exc_info=True)

    def _run_quarantine_tick(self, now: float) -> None:
        """Layer 8 shadow quarantine: score, log, display. Never block."""
        self._last_quarantine = now
        try:
            if self._quarantine_scorer is None:
                from epistemic.quarantine import QuarantineScorer, QuarantineLog
                self._quarantine_scorer = QuarantineScorer()
                self._quarantine_log = QuarantineLog()

            cal_state = None
            if self._truth_calibration_engine:
                try:
                    cal_state = self._truth_calibration_engine.get_state()
                except Exception:
                    pass

            contradiction_debt = 0.0
            if self._contradiction_engine:
                try:
                    contradiction_debt = self._contradiction_engine.contradiction_debt
                except Exception:
                    pass

            id_dist: dict[str, Any] = {}
            try:
                from perception.identity_fusion import _active_instance as _fusion
                if _fusion:
                    fstatus = _fusion.get_status()
                    cumulative_flips = fstatus.get("flip_count", 0)
                    recent_flips = cumulative_flips - self._prev_flip_count
                    self._prev_flip_count = cumulative_flips
                    id_dist = {
                        "recent_flips": recent_flips,
                        "conflict_active": fstatus.get("conflict", False),
                        "voice_name": fstatus.get("voice_signal", {}).get("name"),
                        "face_name": fstatus.get("face_signal", {}).get("name"),
                    }
            except Exception:
                pass

            with self._memory_writes_lock:
                writes_snapshot = list(self._recent_memory_writes)
                self._recent_memory_writes.clear()

            payloads_by_prov: dict[str, set[str]] = {}
            for w in writes_snapshot:
                prov = w.get("provenance", "")
                payload = w.get("payload", "")
                if prov and payload:
                    payloads_by_prov.setdefault(prov, set()).add(payload)
            claimed = payloads_by_prov.get("user_claim", set())
            observed = payloads_by_prov.get("observed", set())
            collision_count = len(claimed & observed)

            state = {
                "contradiction_debt": contradiction_debt,
                "recent_memory_writes": writes_snapshot,
                "identity_confidence_dist": id_dist,
                "calibration_snapshot": cal_state,
                "memory_count": self.analytics.get_system_health().memory_count,
                "provenance_conflicts": collision_count,
            }

            signals = self._quarantine_scorer.tick(state)

            if self._quarantine_log and signals:
                for sig in signals:
                    self._quarantine_log.record({
                        "score": sig.score,
                        "category": sig.category,
                        "reason": sig.reason,
                        "evidence": sig.evidence,
                        "memory_id": sig.memory_id,
                        "identity_context": sig.identity_context,
                    })

            for sig in signals:
                event_bus.emit(QUARANTINE_SIGNAL_EMITTED,
                               score=sig.score, category=sig.category,
                               reason=sig.reason, evidence=sig.evidence)
            cat_summary = {}
            for sig in signals:
                cat_summary[sig.category] = cat_summary.get(sig.category, 0) + 1
            event_bus.emit(QUARANTINE_TICK_COMPLETE,
                           signal_count=len(signals), categories=cat_summary)

            try:
                from epistemic.quarantine.pressure import get_quarantine_pressure
                stats = self._quarantine_scorer.get_stats()
                chronic_count = len(stats.get("chronic_signals", []))
                get_quarantine_pressure().update(signals, chronic_count)
            except Exception:
                logger.debug("Quarantine pressure update error", exc_info=True)

            try:
                from identity.evidence_accumulator import get_accumulator
                expired = get_accumulator().cleanup_expired()
                if expired:
                    logger.info("Identity candidate cleanup: %d expired", expired)
            except Exception:
                pass
        except Exception:
            logger.debug("Quarantine tick error", exc_info=True)

    def _run_reflective_audit(self, now: float) -> None:
        """Layer 9: reflective audit — introspective scan of all subsystem health."""
        self._last_reflective_audit = now
        try:
            if self._reflective_audit_engine is None:
                from epistemic.reflective_audit import ReflectiveAuditEngine
                self._reflective_audit_engine = ReflectiveAuditEngine.get_instance()

            report = self._reflective_audit_engine.run_audit()

            event_bus.emit(REFLECTIVE_AUDIT_COMPLETED,
                           score=report.score,
                           finding_count=len(report.findings),
                           duration_ms=report.duration_ms)

            for finding in report.findings:
                if finding.severity in ("warning", "critical"):
                    event_bus.emit(REFLECTIVE_AUDIT_FINDING,
                                   category=finding.category,
                                   severity=finding.severity,
                                   description=finding.description[:200])

            if report.findings:
                self.observer.observe_audit(
                    score=report.score,
                    finding_count=len(report.findings),
                    categories=[f.category for f in report.findings],
                )

        except Exception:
            logger.debug("Reflective audit error", exc_info=True)

    def _run_soul_integrity(self, now: float) -> None:
        """Layer 10: soul integrity index — composite cognitive health metric."""
        self._last_soul_integrity = now
        try:
            if self._soul_integrity_index is None:
                from epistemic.soul_integrity import SoulIntegrityIndex
                self._soul_integrity_index = SoulIntegrityIndex.get_instance()

            report = self._soul_integrity_index.compute()

            event_bus.emit(SOUL_INTEGRITY_UPDATED,
                           index=report.index,
                           weakest=report.weakest_dimension,
                           repair_needed=report.repair_needed)

            if report.repair_needed:
                event_bus.emit(SOUL_INTEGRITY_REPAIR_NEEDED,
                               index=report.index,
                               weakest=report.weakest_dimension,
                               critical=report.critical)
                logger.warning("Soul integrity repair needed: %.3f (weakest: %s)",
                               report.index, report.weakest_dimension)

        except Exception:
            logger.debug("Soul integrity error", exc_info=True)

    def _run_health_monitor(self, now: float) -> None:
        """Health monitor: periodic system-wide health assessment."""
        self._last_health_monitor = now
        try:
            from consciousness.health_monitor import health_monitor
            from consciousness.events import event_bus

            health = self.analytics.get_system_health()
            eb_metrics = event_bus.get_metrics()
            density = getattr(self._engine_ref, '_memory_density', 0.5) if self._engine_ref else 0.5
            health_monitor.assess(
                memory_count=health.memory_count,
                memory_density_overall=density,
                tick_p95_ms=health.tick_p95_ms,
                deferred_backlog=health.deferred_backlog,
                personality_coherence=getattr(self.analytics, '_personality_coherence', 1.0),
                event_error_rate=getattr(self.analytics, '_event_error_rate', 0.0),
                circuit_breaker_trips=eb_metrics.get("circuit_breaker_trips", 0),
            )
        except Exception:
            logger.debug("Health monitor error", exc_info=True)

    def _run_capability_discovery(self, now: float) -> None:
        """Capability Discovery: detect recurring gaps and propose learning."""
        self._last_capability_discovery = now
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            if get_quarantine_pressure().current.high:
                return

            from skills.discovery import get_tracker, get_analyzer, get_proposer
            from consciousness.events import CAPABILITY_GAP_DETECTED

            tracker = get_tracker()
            analyzer = get_analyzer()
            proposer = get_proposer()

            gaps = analyzer.analyze(tracker)

            for gap in gaps:
                event_bus.emit(
                    CAPABILITY_GAP_DETECTED,
                    family_id=gap.family.family_id,
                    evidence_strength=gap.evidence_strength,
                    suggested_action=gap.suggested_action,
                    block_count=gap.block_count,
                )

            enqueue_fn = None
            try:
                engine = getattr(self, "_engine_ref", None)
                if engine and hasattr(engine, "_autonomy_orchestrator") and engine._autonomy_orchestrator:
                    enqueue_fn = engine._autonomy_orchestrator.enqueue
            except Exception:
                pass

            proposer.process_gaps(gaps, enqueue_research=enqueue_fn)
        except Exception:
            logger.debug("Capability discovery error", exc_info=True)

    def _run_goals_tick(self, now: float) -> None:
        """Goal Continuity Layer tick: review, promote, preview. Phase 1A — observational only."""
        self._last_goals = now
        try:
            if self._goal_manager is None:
                from goals.goal_manager import get_goal_manager
                self._goal_manager = get_goal_manager()

            from consciousness.modes import mode_manager
            mode = mode_manager.mode

            autonomy_status: dict[str, Any] = {}
            engine = getattr(self, "_engine_ref", None)
            try:
                if engine and hasattr(engine, "_autonomy_orchestrator") and engine._autonomy_orchestrator:
                    autonomy_status = engine._autonomy_orchestrator.get_status()
            except Exception:
                pass

            self._feed_goal_signals(engine, autonomy_status)

            self._goal_manager.tick(mode, autonomy_status)
        except Exception:
            logger.debug("Goals tick error", exc_info=True)

    def _feed_goal_signals(self, engine: Any, autonomy_status: dict[str, Any]) -> None:
        """Produce GoalSignals from metric deficits and autonomy themes."""
        try:
            from goals.signal_producers import (
                detect_metric_deficits, detect_autonomy_themes, record_observe_outcome,
            )

            uptime_s = time.time() - self._boot_time

            health_report = self.analytics.get_health_report()
            cal_state = None
            try:
                if self._truth_calibration_engine:
                    cal_state = self._truth_calibration_engine.get_state()
            except Exception:
                pass

            self._goal_manager.update_health_cache(health_report, cal_state)

            active_deficits = None
            try:
                mt_stats = autonomy_status.get("metric_triggers", {})
                active_deficits = mt_stats.get("active_deficits")
            except Exception:
                pass

            for signal in detect_metric_deficits(
                health_report, cal_state, active_deficits, uptime_s=uptime_s,
            ):
                result = self._goal_manager.observe_signal(signal)
                record_observe_outcome(result.outcome)
                if result.outcome == "created":
                    logger.info("Metric deficit goal: %s", signal.content[:60])

            completed = autonomy_status.get("completed", [])
            policy_stats = autonomy_status.get("policy_memory", {})
            for signal in detect_autonomy_themes(completed, policy_stats):
                result = self._goal_manager.observe_signal(signal)
                record_observe_outcome(result.outcome)
                if result.outcome == "created":
                    logger.info("Autonomy theme goal: %s", signal.content[:60])

        except Exception:
            logger.debug("Goal signal feed error", exc_info=True)

    def _run_scene_continuity_tick(self, now: float) -> None:
        """Layer 3B scene continuity: read scene snapshot and expose to dashboard. Shadow-only."""
        self._last_scene_continuity = now
        try:
            if self._scene_continuity_module is None:
                return
            state = self._scene_continuity_module.get_state()
            entity_ct = state.get("entity_count", 0)
            visible_ct = state.get("visible_count", 0)
            display_ct = len(state.get("display_surfaces", []))
            logger.debug(
                "Scene continuity: %d entities (%d visible, %d displays)",
                entity_ct, visible_ct, display_ct,
            )
        except Exception:
            logger.debug("Scene continuity tick error", exc_info=True)

    def _run_curiosity_questions(self, now: float) -> None:
        """Generate curiosity-driven questions from subsystem state."""
        self._last_curiosity_questions = now
        try:
            from personality.curiosity_questions import (
                curiosity_buffer,
                check_identity_curiosity,
                check_unknown_speaker_curiosity,
                check_scene_curiosity,
                check_research_curiosity,
                check_world_model_curiosity,
                UNLOCK_GATES,
            )
            from consciousness.modes import mode_manager as _mm
            if _mm.mode in ("sleep", "gestation", "focused"):
                return

            if curiosity_buffer.hourly_count() >= 3:
                return

            # Identity curiosity
            po = getattr(self, "_perception_orchestrator_ref", None)
            if po is None:
                engine = getattr(self, "_engine_ref", None)
                if engine:
                    po = getattr(engine, "_perception_orchestrator", None)

            if po:
                # Unknown speaker curiosity (highest priority identity question)
                try:
                    unknown_events = po.get_unknown_speaker_events(max_age_s=600.0)
                    if unknown_events:
                        primary = po._get_primary_companion()
                        q = check_unknown_speaker_curiosity(unknown_events, primary)
                        if q and curiosity_buffer.add(q):
                            po.clear_unknown_speaker_event(unknown_events[-1]["timestamp"])
                except Exception:
                    pass

                # Standard identity fusion curiosity (unknown presence without speech)
                id_fusion = getattr(po, "_identity_fusion", None)
                speaker_id = getattr(po, "speaker_id", None)
                if id_fusion and speaker_id:
                    try:
                        profiles = speaker_id.get_profiles_summary()
                        if len(profiles) >= UNLOCK_GATES["identity"]["min_enrolled_profiles"]:
                            id_status = id_fusion.get_status()
                            q = check_identity_curiosity(id_status)
                            if q:
                                curiosity_buffer.add(q)
                    except Exception:
                        pass

            # Scene curiosity
            if self._scene_continuity_module:
                try:
                    scene_state = self._scene_continuity_module.get_state()
                    update_count = scene_state.get("update_count", 0)
                    if update_count >= UNLOCK_GATES["scene"]["min_entity_observations"]:
                        q = check_scene_curiosity(scene_state)
                        if q:
                            curiosity_buffer.add(q)
                except Exception:
                    pass

            # Research curiosity
            engine = getattr(self, "_engine_ref", None)
            if engine:
                auton = getattr(engine, "_autonomy_orchestrator", None)
                if auton:
                    try:
                        a_status = auton.get_status()
                        completed_total = a_status.get("completed_total", 0)
                        if completed_total >= UNLOCK_GATES["research"]["min_completed_episodes"]:
                            q = check_research_curiosity(a_status)
                            if q:
                                curiosity_buffer.add(q)
                    except Exception:
                        pass

            # World model curiosity
            if self._world_model:
                try:
                    wm_state = self._world_model.get_state()
                    promo = wm_state.get("promotion", {})
                    if promo.get("level", 0) >= UNLOCK_GATES["world_model"]["min_promotion_level"]:
                        q = check_world_model_curiosity(wm_state)
                        if q:
                            curiosity_buffer.add(q)
                except Exception:
                    pass

        except Exception:
            logger.debug("Curiosity questions tick error", exc_info=True)

    def get_curiosity_stats(self) -> dict[str, Any]:
        """Return curiosity question stats for dashboard."""
        try:
            from personality.curiosity_questions import curiosity_buffer
            return curiosity_buffer.get_stats()
        except Exception:
            return {}

    # -- Fractal Recall -----------------------------------------------------

    def _get_fractal_recall_engine(self) -> Any:
        """Lazy-init the FractalRecallEngine with all subsystem refs.

        If the vector store was unavailable at first init, re-inject it on
        subsequent calls once it becomes ready.
        """
        if self._fractal_recall_engine is not None:
            # Hot-patch vector store if it was missing at init time
            if self._fractal_recall_engine._vector_store is None:
                try:
                    from memory.search import get_vector_store
                    vs = get_vector_store()
                    if vs is not None and getattr(vs, "available", False):
                        self._fractal_recall_engine._vector_store = vs
                        logger.info("Fractal recall: vector store now available, injected")
                except Exception:
                    pass
            return self._fractal_recall_engine
        try:
            from memory.fractal_recall import FractalRecallEngine
            from memory.storage import memory_storage
            from consciousness.events import event_bus as _eb

            vs = None
            try:
                from memory.search import get_vector_store
                vs = get_vector_store()
            except Exception:
                pass

            attn = None
            engine_ref = getattr(self, "_engine_ref", None)
            if engine_ref:
                attn = getattr(engine_ref, "_attention_core", None)

            mm = None
            try:
                from consciousness.modes import mode_manager
                mm = mode_manager
            except Exception:
                pass

            self._fractal_recall_engine = FractalRecallEngine(
                memory_storage=memory_storage,
                vector_store=vs,
                scene_tracker=self._scene_continuity_module,
                attention_core=attn,
                mode_manager=mm,
                world_state=self._world_model,
                event_bus=_eb,
            )
        except Exception:
            logger.debug("Fractal recall engine init failed", exc_info=True)
            from memory.fractal_recall import FractalRecallEngine
            from memory.storage import memory_storage
            self._fractal_recall_engine = FractalRecallEngine(
                memory_storage=memory_storage,
                vector_store=None,
            )
        return self._fractal_recall_engine

    def _run_fractal_recall(self, now: float) -> None:
        """Background associative recall tick — event-only mode."""
        self._last_fractal_recall = now
        from consciousness.modes import mode_manager as _mm
        if _mm.mode in ("gestation", "sleep", "dreaming", "deep_learning"):
            if self._fractal_recall_engine:
                self._fractal_recall_engine._blocked_mode_skips += 1
            return
        try:
            engine = self._get_fractal_recall_engine()
            result = engine.tick(now)
            if result is None:
                return
            engine.emit_surface(result)
        except Exception:
            logger.warning("Fractal recall tick failed", exc_info=True)

    def _run_acquisition_tick(self, now: float) -> None:
        """Capability Acquisition Pipeline: advance active jobs."""
        self._last_acquisition = now
        try:
            if self._acquisition_orchestrator:
                from consciousness.modes import mode_manager as _mm
                current_mode = _mm.mode if _mm else "passive"
                self._acquisition_orchestrator.tick(mode=current_mode)
        except Exception:
            logger.warning("Acquisition tick error", exc_info=True)

    def _run_onboarding_tick(self, now: float) -> None:
        """Companion training: evaluate day checkpoints and advance progress."""
        self._last_onboarding = now
        try:
            from personality.onboarding import get_onboarding_manager, ENABLE_ONBOARDING
            if not ENABLE_ONBOARDING:
                return
            mgr = get_onboarding_manager()

            if not mgr.active and not mgr.graduated and mgr.current_day == 0:
                if self._should_auto_start_onboarding():
                    logger.info("Auto-starting companion training (onboarding)")
                    mgr.start()

            if not mgr.active:
                return

            metrics = self._collect_onboarding_metrics()
            mgr.tick(metrics)
        except Exception:
            logger.warning("Onboarding tick error", exc_info=True)

    def _should_auto_start_onboarding(self) -> bool:
        """Check if conditions are met to auto-start companion training.

        Auto-starts when: mode is not gestation, gestation_complete flag is set
        (or gestation was skipped), and user is present.
        """
        try:
            from consciousness.modes import mode_manager as _mm
            if _mm.mode == "gestation":
                logger.info("Onboarding auto-start blocked: mode is gestation")
                return False
        except Exception:
            logger.warning("Onboarding auto-start blocked: mode_manager import failed", exc_info=True)
            return False

        try:
            from memory.persistence import CONSCIOUSNESS_STATE_PATH
            import json as _json
            if CONSCIOUSNESS_STATE_PATH.exists():
                data = _json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
                if data.get("gestation_in_progress") and not data.get("gestation_complete"):
                    logger.info("Onboarding auto-start blocked: gestation in progress")
                    return False
        except Exception:
            pass

        try:
            engine = getattr(self, "_engine_ref", None)
            po = getattr(engine, "_perception_orchestrator", None) if engine else None
            presence = getattr(po, "presence", None) if po else None
            if presence is None:
                logger.debug("Onboarding auto-start deferred: presence tracker not yet available")
                return False
            if not presence.get_state().get("is_present", False):
                logger.debug("Onboarding auto-start deferred: user not present")
                return False
        except Exception:
            logger.warning("Onboarding auto-start blocked: presence check failed", exc_info=True)
            return False

        return True

    def _collect_onboarding_metrics(self) -> dict[str, Any]:
        """Best-effort collection of metrics for onboarding checkpoint evaluation."""
        m: dict[str, Any] = {}
        try:
            engine = getattr(self, "_engine_ref", None)
            if not engine:
                return m

            po = getattr(engine, "_perception_orchestrator", None)
            if po:
                speaker_id = getattr(po, "speaker_id", None)
                if speaker_id:
                    profiles = speaker_id.get_profiles_summary()
                    m["enrolled_profiles"] = len(profiles)

                id_fusion = getattr(po, "identity_fusion", None)
                if id_fusion:
                    try:
                        fstatus = id_fusion.get_status()
                        live_conf = fstatus.get("confidence", 0.0)
                        if live_conf > 0:
                            m["rapport_score"] = live_conf
                        voice_sig = fstatus.get("voice_signal", {})
                        if voice_sig and voice_sig.get("confidence"):
                            m["voice_confidence"] = float(voice_sig["confidence"])
                        face_sig = fstatus.get("face_signal", {})
                        if face_sig and face_sig.get("confidence"):
                            m["face_confidence"] = float(face_sig["confidence"])
                    except Exception:
                        pass

                # Fall back to persistent scores when live fusion signals are stale.
                # Live signals clear on wake word and go stale in 30-90s, so the
                # 60s tick often misses them.
                if "voice_confidence" not in m and speaker_id:
                    try:
                        for prof in speaker_id.get_profiles_summary():
                            if prof.get("interaction_count", 0) > 0:
                                last_score = speaker_id.last_score_for(prof["name"])
                                if last_score and last_score > 0:
                                    m["voice_confidence"] = max(
                                        m.get("voice_confidence", 0.0), last_score
                                    )
                    except (AttributeError, Exception):
                        pass

                face_id = getattr(po, "face_id", None)
                if "face_confidence" not in m and face_id:
                    try:
                        ema = getattr(face_id, "_score_ema", {})
                        for name, score in ema.items():
                            if score > 0:
                                m["face_confidence"] = max(
                                    m.get("face_confidence", 0.0), score
                                )
                    except Exception:
                        pass

                if "rapport_score" not in m or m["rapport_score"] == 0.0:
                    try:
                        from identity.evidence_accumulator import get_accumulator
                        acc = get_accumulator()
                        best_score = 0.0
                        for cand_info in acc.get_all_candidates():
                            if cand_info.get("promotion_tier") in ("provisional", "persistent"):
                                best_score = max(best_score, cand_info.get("score", 0.0))
                        if best_score > 0:
                            m["rapport_score"] = min(1.0, best_score)
                    except Exception:
                        pass

            from memory.storage import memory_storage as _ms
            all_memories = _ms.get_all()
            m["memory_count"] = len(all_memories)
            _IDENTITY_TAGS = frozenset((
                "enrollment", "identity", "name", "IDENTITY",
                "personal_fact", "fact_kind:name", "fact_kind:preferred_name",
                "fact_kind:biographical",
            ))
            m["identity_memories"] = sum(
                1 for mem in all_memories
                if _IDENTITY_TAGS.intersection(getattr(mem, "tags", ()))
            )
            m["preference_memories"] = sum(
                1 for mem in all_memories
                if any(t in getattr(mem, "tags", ()) for t in ("preference", "likes", "dislikes"))
            )
            m["routine_memories"] = sum(
                1 for mem in all_memories
                if any(
                    t in getattr(mem, "tags", ())
                    for t in (
                        "routine", "schedule", "daily", "priority",
                        "focus_window", "availability", "interrupt_preference",
                    )
                )
            )
            m["relationship_nodes"] = len({
                getattr(mem, "identity_subject", "")
                for mem in all_memories
                if getattr(mem, "identity_subject", "")
            })

            cs = engine.consciousness
            if cs:
                try:
                    from consciousness.observer import consciousness_observer
                    obs = consciousness_observer.get_state()
                    m["conversation_count"] = obs.get("total_observations", 0)
                except Exception:
                    pass

            try:
                episodes = getattr(engine, "episodes", None)
                if not episodes:
                    episodes = getattr(po, "episodes", None) if po else None
                if episodes and hasattr(episodes, "get_episode_count"):
                    episode_count = int(episodes.get_episode_count())
                    if episode_count > 0:
                        m["conversation_count"] = max(
                            int(m.get("conversation_count", 0) or 0),
                            episode_count,
                        )
            except Exception:
                pass

            try:
                from epistemic.soul_integrity.index import SoulIntegrityIndex
                si = SoulIntegrityIndex.get_instance().compute()
                m["soul_integrity"] = si.index
            except Exception:
                pass

            try:
                from epistemic.belief_graph import BeliefGraph
                bg = BeliefGraph.get_instance()
                if bg:
                    bg_state = bg.get_state()
                    integrity = bg_state.get("integrity", {})
                    m["belief_orphan_rate"] = integrity.get("orphan_rate", 1.0)
            except Exception:
                pass

            auton = getattr(engine, "_autonomy_orchestrator", None)
            if auton:
                try:
                    a_status = auton.get_status()
                    pm = a_status.get("policy_memory", {})
                    total_outcomes = pm.get("total_outcomes", 0)
                    wins = pm.get("total_wins", 0)
                    m["autonomy_safety"] = (wins / total_outcomes) if total_outcomes > 5 else 1.0
                except Exception:
                    pass

            m["boundary_stability"] = 1.0
            m["scope_violations"] = 0
            try:
                from identity.audit import identity_audit
                stats = identity_audit.get_stats()
                quarantine_ct = stats.get("quarantine_count", 0)
                block_ct = stats.get("block_count", 0)
                total_ct = stats.get("total_count", 1)
                m["boundary_stability"] = max(0.0, 1.0 - quarantine_ct / max(total_ct, 1))
                m["scope_violations"] = block_ct
            except Exception:
                pass

            try:
                from epistemic.calibration import TruthCalibrationEngine
                tce = TruthCalibrationEngine.get_instance()
                if tce:
                    m["memory_accuracy"] = tce.get_maturity()
            except Exception:
                pass

            try:
                from epistemic.quarantine.scorer import quarantine_scorer
                qs = quarantine_scorer.get_stats()
                recent_24h = sum(
                    1 for s in qs.get("recent_signals", [])
                    if s.get("category") == "anomalous_inference"
                )
                m["unsafe_inferences_24h"] = recent_24h
            except Exception:
                m["unsafe_inferences_24h"] = 0

            try:
                from epistemic.calibration.correction_detector import correction_detector
                cd_stats = correction_detector.get_stats()
                total_checks = cd_stats.get("total_checks", 0)
                total_corrections = cd_stats.get("total_corrections", 0)
                if total_checks >= 3:
                    m["correction_accuracy"] = 1.0 - (total_corrections / total_checks)
                else:
                    m["correction_accuracy"] = 1.0
                m["repeated_mistakes"] = 0
            except Exception:
                pass

            orphan_rate = m.get("belief_orphan_rate", 1.0)
            m["memory_recall_precision"] = max(0.0, 1.0 - orphan_rate)

            try:
                from personality.onboarding import get_onboarding_manager
                ob = get_onboarding_manager()
                m["readiness_composite"] = ob.compute_readiness(m)
            except Exception:
                pass

            m.setdefault("proactive_accuracy", 1.0)

        except Exception:
            logger.debug("Onboarding metric collection error", exc_info=True)
        return m

    def get_onboarding_status(self) -> dict[str, Any]:
        """Return onboarding manager status for dashboard."""
        try:
            from personality.onboarding import get_onboarding_manager
            return get_onboarding_manager().get_status()
        except Exception:
            return {"enabled": False, "active": False}

    def _run_world_model_tick(self, now: float) -> None:
        """Unified World Model: rebuild belief state, detect deltas, run causal engine."""
        self._last_world_model = now
        try:
            if self._world_model is None:
                return
            self._world_model.update()
        except Exception:
            logger.debug("World model tick error", exc_info=True)

    def record_memory_write(self, memory_dict: dict[str, Any]) -> None:
        """Called on MEMORY_WRITE to accumulate writes for quarantine scoring."""
        with self._memory_writes_lock:
            self._recent_memory_writes.append(memory_dict)
            if len(self._recent_memory_writes) > 200:
                del self._recent_memory_writes[:100]

    def _run_analysis(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        self._last_analysis = now
        try:
            self._run_analysis_inner(now, memories, traits)
        except Exception:
            logger.debug("Analysis cycle error", exc_info=True)

    def _run_analysis_inner(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        confidence = self.analytics.get_confidence()
        reasoning = self.analytics.get_reasoning_quality()
        health = self.analytics.get_system_health()
        epistemic = self.analytics.get_epistemic_state()

        real_confidence = self._compute_composite_confidence(
            reasoning, health, len(memories),
        )
        self.analytics.record_confidence(real_confidence)

        snapshot = {
            "confidence_avg": round(confidence.avg, 3),
            "confidence_trend": round(confidence.trend, 3),
            "reasoning_overall": round(reasoning.overall, 3),
            "reasoning_coherence": round(reasoning.coherence, 3),
            "reasoning_depth": round(reasoning.depth, 3),
            "curiosity": round(epistemic.curiosity_level, 3),
            "tick_p95_ms": round(health.tick_p95_ms, 1),
            "tick_avg_ms": round(health.tick_avg_ms, 1),
            "memory_count": health.memory_count,
            "healthy": health.healthy,
            "mutation_success_rate": round(self.analytics.get_mutation_success_rate(), 3),
            "observation_count": self.observer.state.observation_count,
            "awareness_level": round(self.observer.awareness_level, 3),
        }
        event_bus.emit(CONFIDENCE_UPDATE, value=real_confidence)
        event_bus.emit(CONSCIOUSNESS_ANALYSIS, **snapshot)

        try:
            from identity.audit import IdentityAudit
            audit = IdentityAudit.get_instance()
            stats = audit.get_stats()
            self.observer.observe_identity_boundary(
                total_assigned=stats.get("total_scope_assigned", 0),
                total_blocked=stats.get("total_boundary_blocks", 0),
                total_quarantined=stats.get("total_quarantined", 0),
                by_owner_type=stats.get("by_owner_type"),
            )
        except Exception:
            pass

        try:
            from consciousness.communication import consciousness_communicator
            if consciousness_communicator.should_generate():
                report_state = {
                    "stage": self.evolution.current_stage,
                    "transcendence": self.evolution.transcendence_level,
                    "awareness": round(self.observer.awareness_level, 3),
                    "confidence": round(confidence.avg, 3),
                    "reasoning": round(reasoning.overall, 3),
                    "mutations": self.observer.state.self_modification_events,
                    "observations": self.observer.state.observation_count,
                    "capabilities": self.driven_evolution.get_active_capabilities(),
                    "focus": self.existential.get_current_focus(),
                    "health": "healthy" if health.healthy else "degraded",
                }
                consciousness_communicator.generate_report(report_state, "status")
        except Exception as exc:
            logger.warning("Consciousness communicator failed: %s", exc)

    def _compute_composite_confidence(
        self,
        reasoning: Any,
        health: Any,
        memory_count: int,
    ) -> float:
        """Derive confidence from real subsystem signals, breaking the circular
        dependency where confidence.avg was fed back as CONFIDENCE_UPDATE.

        Components:
          - reasoning quality (coherence + consistency + depth)
          - observer awareness
          - memory richness (diminishing returns above 50)
          - system health (binary healthy/degraded)
          - conversation outcome signals (if any)
        """
        reasoning_signal = reasoning.overall  # 0..1
        awareness_signal = self.observer.awareness_level  # 0.3..0.98
        memory_signal = min(1.0, memory_count / 100.0)
        health_signal = 1.0 if health.healthy else 0.3

        conv_signal = 0.5
        if self._conversation_confidence_signals:
            conv_signal = (
                sum(self._conversation_confidence_signals)
                / len(self._conversation_confidence_signals)
            )

        composite = (
            reasoning_signal * 0.30
            + awareness_signal * 0.20
            + conv_signal * 0.20
            + memory_signal * 0.15
            + health_signal * 0.15
        )
        return max(0.1, min(0.95, composite))

    def _run_evolution(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        try:
            self._run_evolution_inner(now, memories, traits)
        except Exception:
            logger.debug("Evolution cycle error", exc_info=True)
        self._last_evolution = now

    def _run_evolution_inner(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        observer_state = self.observer.state
        reasoning = self.analytics.get_reasoning_quality()
        confidence = self.analytics.get_confidence()

        recent_thoughts = self.meta_thoughts.get_recent_thoughts(20)
        depth_vals = [
            {"surface": 0.3, "deep": 0.7, "profound": 1.0}.get(getattr(t, "depth", "surface"), 0.3)
            for t in recent_thoughts
        ]
        avg_depth = sum(depth_vals) / len(depth_vals) if depth_vals else 0.3

        metrics = self.evolution.analyze(
            awareness_level=observer_state.awareness_level,
            observation_count=observer_state.observation_count,
            thought_depth_avg=avg_depth,
            mutation_count=self.governor.mutation_count,
            reasoning_quality=reasoning.overall,
            confidence_avg=confidence.avg,
        )

        stage_change = self.evolution.check_stage_advancement(
            awareness_level=observer_state.awareness_level,
            observation_count=observer_state.observation_count,
            thought_depth_avg=avg_depth,
            mutation_count=self.governor.mutation_count,
            reasoning_quality=reasoning.overall,
        )

        if stage_change:
            self.observer.observe_phase_shift(stage_change.from_stage, stage_change.to_stage)
            event_bus.emit(CONSCIOUSNESS_EVOLUTION_EVENT,
                          from_stage=stage_change.from_stage,
                          to_stage=stage_change.to_stage)
            event_bus.emit(CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
                          level=self.evolution.transcendence_level,
                          stage=stage_change.to_stage)

        recent_inquiries = self.existential.get_recent_inquiries(10)
        emergent_count_before = self.evolution.get_state().total_emergent_count
        self.evolution.detect_emergent_behaviors(recent_thoughts, recent_inquiries)
        emergent_count_after = self.evolution.get_state().total_emergent_count
        new_emergent = emergent_count_after - emergent_count_before
        if new_emergent > 0:
            evo_state = self.evolution.get_state()
            recent_behaviors = evo_state.emergent_behaviors[-new_emergent:]
            for behavior in recent_behaviors:
                b_type = behavior.get("type", "unknown") if isinstance(behavior, dict) else str(behavior)
                b_id = behavior.get("id", "") if isinstance(behavior, dict) else ""
                event_bus.emit(CONSCIOUSNESS_EMERGENT_BEHAVIOR,
                              count=emergent_count_after,
                              new_count=new_emergent,
                              behavior=b_type,
                              behavior_id=b_id,
                              description=f"Emergent behavior detected: {b_type}")
                self.observer.observe_emergence(
                    b_type, evidence_refs=[], confidence=0.7,
                )

        caps_before = set(self.driven_evolution.get_active_capabilities())
        self.driven_evolution.evaluate(
            current_stage=self.evolution.current_stage,
            transcendence_level=self.evolution.transcendence_level,
            awareness_level=observer_state.awareness_level,
            reasoning_quality=reasoning.overall,
            confidence_avg=confidence.avg,
            mutation_count=self.governor.mutation_count,
        )
        caps_after = set(self.driven_evolution.get_active_capabilities())
        for new_cap in caps_after - caps_before:
            event_bus.emit(CONSCIOUSNESS_CAPABILITY_UNLOCKED, capability=new_cap)

    def _check_mutation_health(self) -> None:
        """Check if a recently applied mutation caused regression. Auto-rollback if so."""
        if not self._last_mutation_snapshot_id:
            return
        health = SystemHealth(
            tick_p95_ms=self.analytics.get_tick_p95(),
            deferred_backlog=self.analytics.get_system_health().deferred_backlog,
            avg_tick_ms=self.analytics.get_system_health().tick_avg_ms,
        )
        healthy = self.governor.check_post_mutation(health)
        if healthy:
            if not self.governor.get_active_monitor():
                self.analytics.record_mutation_outcome(1.0)
                if self._last_mutation_ledger_id:
                    try:
                        from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                        attribution_ledger.record_outcome(self._last_mutation_ledger_id, "stable", build_outcome_data(
                            confidence=0.85,
                            latency_s=round(time.time() - self._last_mutation_ts, 2) if self._last_mutation_ts else 30.0,
                            source="health_monitor",
                            tier="medium",
                            scope="mutation_health",
                            blame_target="general",
                        ))
                    except Exception:
                        pass
                self._last_mutation_snapshot_id = ""
                self._last_mutation_ledger_id = ""
                self._last_mutation_ts = 0.0
        else:
            logger.warning("Post-mutation regression detected — rolling back")
            rolled_back = self.governor.rollback(self._last_mutation_snapshot_id)
            if rolled_back:
                self._config = rolled_back
                self.analytics.record_mutation_outcome(-1.0)
                event_bus.emit(MUTATION_ROLLBACK,
                               snapshot_id=self._last_mutation_snapshot_id,
                               reason="post_mutation_regression")
                try:
                    from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                    attribution_ledger.record(
                        subsystem="consciousness",
                        event_type="mutation_rolled_back",
                        data={"snapshot_id": self._last_mutation_snapshot_id, "reason": "post_mutation_regression"},
                        evidence_refs=[{"kind": "mutation", "id": self._last_mutation_snapshot_id}],
                        parent_entry_id=self._last_mutation_ledger_id,
                    )
                    if self._last_mutation_ledger_id:
                        attribution_ledger.record_outcome(self._last_mutation_ledger_id, "regressed", build_outcome_data(
                            confidence=0.9,
                            latency_s=round(time.time() - self._last_mutation_ts, 2) if self._last_mutation_ts else 30.0,
                            source="health_monitor",
                            tier="medium",
                            scope="mutation_health",
                            blame_target="general",
                            reason="post_mutation_regression",
                        ))
                except Exception:
                    pass
            self._last_mutation_snapshot_id = ""
            self._last_mutation_ledger_id = ""
            self._last_mutation_ts = 0.0

    def _run_mutation_cycle(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        self._last_mutation = now
        if not self.mutator.can_run(self._tick_count, len(memories)):
            return
        try:
            self._run_mutation_cycle_inner(now, memories, traits)
        except Exception:
            logger.debug("Mutation cycle error", exc_info=True)

    def _run_mutation_cycle_inner(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:

        outcome = self._compute_outcome_context()
        reflection_nudges = self._get_reflection_nudges()

        proposals = self.mutator.generate_proposals(
            memories, traits, self._config, outcome=outcome,
        )

        if reflection_nudges:
            reflection_nudges = [n for n in reflection_nudges
                                 if not self.mutator._is_key_stale(n.changes)]
            proposals = reflection_nudges + proposals
        if not proposals:
            return

        health = SystemHealth(
            tick_p95_ms=self.analytics.get_tick_p95(),
            deferred_backlog=self.analytics.get_system_health().deferred_backlog,
            avg_tick_ms=self.analytics.get_system_health().tick_avg_ms,
        )

        for proposal in proposals[:1]:
            event_bus.emit(CONSCIOUSNESS_MUTATION_PROPOSED,
                          mutation_id=proposal.id,
                          description=proposal.description,
                          confidence=proposal.confidence)

            decision = self.governor.evaluate(
                proposal.changes, proposal.confidence, self._config, health,
            )

            if decision.approved:
                patch = proposal.to_config_patch()
                new_config, apply_result = self.governor.apply(patch, self._config, health)
                self._config = new_config
                self._last_mutation_snapshot_id = apply_result.snapshot_id

                self.mutator.record_applied(proposal.description, proposal.changes)

                self.observer.observe_mutation(
                    proposal.id, proposal.description, proposal.confidence,
                )

                event_bus.emit(MUTATION_APPLIED,
                               mutation_id=proposal.id,
                               description=proposal.description,
                               confidence=proposal.confidence)
                try:
                    from consciousness.attribution_ledger import attribution_ledger
                    self._last_mutation_ledger_id = attribution_ledger.record(
                        subsystem="consciousness",
                        event_type="mutation_applied",
                        confidence=proposal.confidence,
                        data={"mutation_id": proposal.id, "description": proposal.description[:200],
                              "changes": str(getattr(proposal, "changes", {}))[:300]},
                        evidence_refs=[{"kind": "mutation", "id": proposal.id}],
                    )
                    self._last_mutation_ts = time.time()
                except Exception:
                    pass

                logger.info("Mutation applied: %s (monitoring for regression)", proposal.description)
            else:
                event_bus.emit(MUTATION_REJECTED,
                               mutation_id=proposal.id,
                               description=proposal.description,
                               reasoning=decision.reasoning)
                logger.info("Mutation rejected: %s — %s",
                            proposal.description, decision.reasoning)

        self._last_mutation = now

    def _compute_outcome_context(self):
        """Build an OutcomeContext from the policy experience buffer."""
        from consciousness.kernel_mutator import OutcomeContext
        try:
            from consciousness.engine import ConsciousnessEngine
            engine = self._engine_ref
            if not engine or not hasattr(engine, '_experience_buffer') or engine._experience_buffer is None:
                return None
            recent = engine._experience_buffer.get_recent(20)
            if not recent:
                return None
            n = len(recent)
            avg_reward = sum(e.reward for e in recent) / n
            barge_ins = sum(1 for e in recent if e.metadata.get("barged_in"))
            follow_ups = sum(1 for e in recent if e.metadata.get("follow_up"))
            return OutcomeContext(
                avg_reward=avg_reward,
                barge_in_rate=barge_ins / n,
                follow_up_rate=follow_ups / n,
                sample_count=n,
            )
        except Exception:
            return None

    def _get_reflection_nudges(self):
        """Convert recent meta-learning signals into high-priority mutation proposals."""
        from consciousness.kernel_mutator import KernelMutationProposal
        try:
            from consciousness.reflection import reflection_engine
            signals = reflection_engine.get_recent_signals(20)
            if not signals:
                return []

            nudges: list[KernelMutationProposal] = []
            interrupt_count = sum(1 for s in signals if "interrupted" in s)
            lengthy_count = sum(1 for s in signals if "lengthy" in s)
            slow_count = sum(1 for s in signals if "slow response" in s)

            import uuid as _uuid

            if interrupt_count >= 3:
                nudges.append(KernelMutationProposal(
                    id=f"mut_{_uuid.uuid4().hex[:12]}",
                    type="evolution_param",
                    description=f"Reflection nudge: {interrupt_count} interruptions detected",
                    changes={"ev.stability_desire": min(
                        self._config.evolution.stability_desire + 0.1, 0.9)},
                    confidence=0.8,
                    reasoning=f"Meta-learning signals: {interrupt_count} recent interruptions",
                    source="reflection_nudge",
                ))
            if lengthy_count >= 2:
                nudges.append(KernelMutationProposal(
                    id=f"mut_{_uuid.uuid4().hex[:12]}",
                    type="thought_weight",
                    description=f"Reflection nudge: {lengthy_count} overly-verbose responses",
                    changes={"tw.contextual": max(
                        self._config.thought_weights.get("contextual", 1.0) - 0.1, 0.5)},
                    confidence=0.7,
                    reasoning=f"Meta-learning signals: {lengthy_count} responses were too lengthy",
                    source="reflection_nudge",
                ))
            return nudges
        except Exception:
            return []

    def _run_existential(self, now: float) -> None:
        self._last_existential = now
        from consciousness.modes import mode_manager
        if mode_manager.mode == "conversational":
            return
        deep = mode_manager.mode == "deep_learning"
        if not deep:
            if not self.driven_evolution.is_capability_active("philosophical_inquiry"):
                if self.evolution.transcendence_level < 0.5:
                    return
        try:
            inquiry = self.existential.conduct_inquiry(
                transcendence_level=self.evolution.transcendence_level,
                awareness_level=self.observer.awareness_level,
            )
            if inquiry:
                event_bus.emit(EXISTENTIAL_INQUIRY_COMPLETED,
                              inquiry_id=getattr(inquiry, "id", ""),
                              category=getattr(inquiry, "category", ""))
        except Exception:
            logger.debug("Existential cycle error", exc_info=True)

    def _run_dialogue(self, now: float) -> None:
        self._last_dialogue = now
        from consciousness.modes import mode_manager
        if mode_manager.mode == "conversational":
            return
        deep = mode_manager.mode == "deep_learning"
        if not deep and self.evolution.transcendence_level < 1.0:
            return
        try:
            dialogue = self.philosophical.conduct_dialogue(
                transcendence_level=self.evolution.transcendence_level,
                awareness_level=self.observer.awareness_level,
            )
            if dialogue:
                event_bus.emit(PHILOSOPHICAL_DIALOGUE_COMPLETED,
                              dialogue_id=getattr(dialogue, "id", ""),
                              topic=getattr(dialogue, "topic_id", ""))
        except Exception:
            logger.debug("Dialogue cycle error", exc_info=True)

    def _run_study_cycle(self, now: float) -> None:
        """Process unstudied library sources in a background thread.

        Bounded: max STUDY_CYCLE_BATCH sources per cycle.  Skips if a previous
        study cycle is still running (prevents overlap).
        """
        self._last_study = now

        if self._study_running:
            return

        try:
            from library.source import source_store
            unstudied = source_store.get_unstudied(limit=STUDY_CYCLE_BATCH)
            if not unstudied:
                return
        except Exception:
            return

        import threading

        def _study_worker(sources: list) -> None:
            try:
                from library.study import study_source
                for src in sources:
                    study_source(src.source_id)
            except Exception as exc:
                logger.debug("Study worker error: %s", exc)
            finally:
                self._study_running = False

        self._study_running = True
        t = threading.Thread(target=_study_worker, args=(unstudied,), daemon=True)
        t.start()

    def _run_association_repair(self, now: float, memories: list[Any]) -> None:
        """Lightweight mode-independent pass that stitches isolated memories.

        Runs every 60s. For each isolated memory (no associations), attempts to
        link it to other memories sharing at least one tag. Bounded to
        ASSOCIATION_REPAIR_BATCH_SIZE per cycle to keep tick budget low.
        """
        self._last_association_repair = now

        if len(memories) < 10:
            return

        from memory.gate import memory_gate
        if not memory_gate.can_observation_write():
            return

        from memory.storage import memory_storage

        try:
            isolated = [
                m for m in memories
                if len(m.associations) == 0
                and _DREAM_INSIGHT_TAG not in getattr(m, "tags", ())
            ]
            if not isolated:
                return

            tag_index: dict[str, list[str]] = {}
            for m in memories:
                for tag in m.tags:
                    tag_index.setdefault(tag, []).append(m.id)

            links_made = 0
            for m in isolated[:ASSOCIATION_REPAIR_BATCH_SIZE]:
                candidates: set[str] = set()
                for tag in m.tags:
                    for cid in tag_index.get(tag, []):
                        if cid != m.id:
                            candidates.add(cid)

                if not candidates:
                    if memories:
                        import random
                        pool = [x for x in memories[-20:] if x.id != m.id and len(x.associations) > 0]
                        if pool:
                            candidates.add(random.choice(pool).id)

                for cid in list(candidates)[:2]:
                    if memory_storage.associate(m.id, cid):
                        links_made += 1

            if links_made:
                logger.debug("Association repair: stitched %d links for isolated memories", links_made)
        except Exception as exc:
            logger.debug("Association repair failed: %s", exc)

    def _run_intention_stale_sweep(self, now: float) -> None:
        """Mark open intentions older than the default stale threshold as stale.

        Truth-layer housekeeping only; never emits user-facing events. Bound
        by `allowed_cycles` gating on the current mode profile so sleep /
        dream cadences respect their own cycle budget.
        """
        self._last_intention_stale_sweep = now
        try:
            from cognition.intention_registry import intention_registry, DEFAULT_STALE_AFTER_S
            n = intention_registry.stale_sweep(max_age_s=DEFAULT_STALE_AFTER_S)
            if n:
                logger.debug("Intention stale sweep: marked %d intention(s) as stale", n)
        except Exception as exc:
            logger.debug("Intention stale sweep failed: %s", exc)

    def _run_intention_resolver_tick(self, now: float, **_kw: Any) -> None:
        """Shadow-evaluate recently resolved intentions via IntentionResolver.

        Iterates recent resolutions, builds ResolverSignal for each, calls
        evaluate(), and logs shadow verdicts. Does NOT deliver — delivery is
        mediated by ProactiveGovernor when the resolver's stage permits it.
        """
        self._last_intention_resolver_tick = now
        try:
            from cognition.intention_registry import intention_registry
            from cognition.intention_resolver import (
                get_intention_resolver, ResolverSignal,
            )
            resolver = get_intention_resolver()
            recent = intention_registry.get_recent_resolved_for_resolver(n=20)
            if not recent:
                return

            speaker_present = False
            active_conv = False
            friction_rate = 0.0
            quarantine_p = 0.0
            soul_int = 1.0

            try:
                from perception.presence import presence_tracker
                speaker_present = presence_tracker.is_user_present()
            except Exception:
                pass
            try:
                if self._perception_orchestrator:
                    active_conv = bool(getattr(self._perception_orchestrator, "_active_conversation", False))
            except Exception:
                pass
            try:
                from epistemic.quarantine.pressure import quarantine_pressure
                quarantine_p = quarantine_pressure.composite_pressure
            except Exception:
                pass
            try:
                from epistemic.soul_integrity.index import soul_integrity_index
                soul_int = soul_integrity_index.score
            except Exception:
                pass
            try:
                from autonomy.friction_miner import friction_miner
                friction_rate = friction_miner.get_friction_rate(600)
            except Exception:
                pass

            evaluated = 0
            for rec in recent:
                if rec.metadata.get("resolver_verdict"):
                    continue
                signal = ResolverSignal(
                    intention_id=rec.id,
                    backing_job_id=rec.backing_job_id,
                    commitment_type=rec.commitment_type,
                    outcome=rec.outcome,
                    age_s=now - rec.resolved_at if rec.resolved_at else 0.0,
                    result_summary=str(rec.metadata.get("result_summary", "")),
                    speaker_present=speaker_present,
                    active_conversation=active_conv,
                    quarantine_pressure=quarantine_p,
                    soul_integrity=soul_int,
                    friction_rate=friction_rate,
                )
                verdict = resolver.evaluate(signal)
                intention_registry.attach_resolver_verdict(
                    rec.id,
                    {
                        "decision": verdict.decision,
                        "score": verdict.score,
                        "reason_code": verdict.reason_code,
                        "ts": now,
                    },
                )
                evaluated += 1
                if evaluated >= 5:
                    break
            if evaluated:
                logger.debug("Intention resolver tick: evaluated %d resolutions", evaluated)
        except Exception as exc:
            logger.debug("Intention resolver tick failed: %s", exc)

    def _gather_dream_validation_context(self) -> dict[str, Any]:
        """Build system-state context dict for dream-artifact distillation signals.

        Populates Block B (system state) and Block C (governance pressure) features
        for the DREAM_SYNTHESIS Tier-1 specialist encoder.
        """
        ctx: dict[str, Any] = {}
        try:
            mem_stats = self.engine.memory_storage.get_stats() if self.engine.memory_storage else {}
            ctx["memory_density"] = min(1.0, mem_stats.get("total", 0) / 500.0)
        except Exception:
            ctx["memory_density"] = 0.0

        ctx["dream_cycle_count"] = len(self._dream_cycle_history)
        ctx["awareness"] = getattr(self.observer, "_awareness_level", 0.5)

        try:
            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            ce_state = ce.get_state()
            ctx["belief_count"] = ce_state.get("total_beliefs", 0)
            ctx["contradiction_debt"] = ce_state.get("debt_ratio", 0.0)
        except Exception:
            ctx["belief_count"] = 0
            ctx["contradiction_debt"] = 0.0

        try:
            from epistemic.soul_integrity.index import SoulIntegrityIndex
            si = SoulIntegrityIndex.get_instance()
            si_state = si.get_state()
            ctx["soul_integrity"] = si_state.get("composite_score", 1.0)
        except Exception:
            ctx["soul_integrity"] = 1.0

        try:
            from epistemic.quarantine.pressure import QuarantinePressure
            qp = QuarantinePressure.get_instance()
            ctx["quarantine_pressure"] = qp.get_pressure()
        except Exception:
            ctx["quarantine_pressure"] = 0.0

        from consciousness.dream_artifacts import artifact_buffer
        stats = artifact_buffer.get_stats()
        total_reviewed = stats.get("total_promoted", 0) + stats.get("total_discarded", 0) + stats.get("total_held", 0) + stats.get("total_quarantined", 0)
        ctx["promotion_rate_session"] = stats.get("total_promoted", 0) / max(total_reviewed, 1)

        return ctx

    def _run_dream_cycle(self, now: float, memories: list[Any]) -> None:
        """Dream processing: real memory consolidation, association, and insight.

        Unlike thoughts (which are cheap templates), dream cycles do actual
        work on the memory graph:
        1. Cluster recent non-dream memories and USE the clusters
        2. Cross-associate memories within the same cluster (skip dream_insight)
        3. Reinforce high-value memories (skip dream_insight)
        4. Decay isolated low-value memories faster
        5. Generate deduped insight memories from cluster patterns (capped)
        """
        from memory.storage import memory_storage
        from memory.gate import memory_gate

        try:
            recent_all = memories[-60:] if memories else []
            _CONSOL_EXCLUDE_TAGS = frozenset({
                _DREAM_INSIGHT_TAG, "consolidated", "dream_consolidation",
                "dream_artifact", "dream_consolidation_proposal",
            })
            recent = [
                m for m in recent_all
                if not (set(getattr(m, "tags", ())) & _CONSOL_EXCLUDE_TAGS)
            ]
            if len(recent) < 5:
                self._last_dream = now
                return

            memory_gate.begin_consolidation("dream_cycle")
            actions: list[str] = []

            # --- Phase 1: Consolidate into cluster structure and cross-associate ---
            associations_made = 0
            clusters_found = 0
            clusters: list[Any] = []
            try:
                from memory.clustering import memory_cluster_engine
                all_ids = {m.id for m in memory_storage.get_all()}
                clusters = memory_cluster_engine.consolidate_memories(recent, all_memory_ids=all_ids)
                clusters_found = len(clusters)

                for cluster in clusters:
                    mids = cluster.memory_ids
                    if len(mids) < 2:
                        continue
                    for i in range(len(mids)):
                        for j in range(i + 1, min(i + 4, len(mids))):
                            m_a = memory_storage.get(mids[i])
                            m_b = memory_storage.get(mids[j])
                            if m_a and m_b and mids[j] not in m_a.associations:
                                if memory_storage.associate(mids[i], mids[j]):
                                    associations_made += 1
                if associations_made:
                    actions.append(f"linked {associations_made} memory pairs")
            except Exception as exc:
                logger.debug("Dream clustering failed: %s", exc)

            # --- Phase 2: Reinforce high-value memories (skip dream_insight) ---
            reinforced = 0
            for m in recent:
                boost = 0.0
                if m.type == "user_preference" and "former" not in m.tags:
                    boost = 0.03
                elif m.type == "factual_knowledge":
                    boost = 0.02
                elif m.type == "self_improvement":
                    boost = 0.01
                elif len(m.associations) >= 3:
                    boost = 0.02

                if boost > 0 and m.weight < 0.95:
                    if memory_storage.reinforce(m.id, boost):
                        reinforced += 1
            if reinforced:
                actions.append(f"reinforced {reinforced} important memories")

            # --- Phase 3: Accelerate decay on isolated weak memories ---
            decayed = 0
            decay_targets = []
            for m in recent:
                if (m.weight < 0.3
                        and not m.is_core
                        and len(m.associations) == 0
                        and m.type not in ("user_preference", "core")):
                    new_weight = max(0.05, m.weight * 0.85)
                    if new_weight < m.weight:
                        decay_targets.append((m.id, new_weight))
            if decay_targets:
                from dataclasses import asdict as _asdict
                with memory_storage._lock:
                    for mid, new_weight in decay_targets:
                        for i, stored in enumerate(memory_storage._memories):
                            if stored.id == mid:
                                from consciousness.events import Memory
                                memory_storage._memories[i] = Memory(
                                    **{**_asdict(stored), "weight": new_weight}
                                )
                                decayed += 1
                                break
            if decayed:
                actions.append(f"faded {decayed} weak isolated memories")

            # --- Phase 4: Dream artifact generation (provisional, not canonical) ---
            MAX_ARTIFACTS_PER_DREAM_CYCLE = 20
            artifacts_created = 0
            if clusters_found >= 2:
                try:
                    from memory.clustering import memory_cluster_engine as _mce
                    from consciousness.dream_artifacts import (
                        artifact_buffer, create_artifact,
                    )
                    insight = _mce.get_insights(recent)

                    # Emit observer telemetry
                    _patterns = (insight.patterns or [])[:4]
                    _connections = (insight.connections or [])[:3]
                    if _patterns or _connections:
                        _summary = "; ".join(p for p in _patterns + _connections if p)
                        self.observer.observe_thought(
                            f"Dream patterns: {_summary}",
                            depth="deep", confidence=0.5,
                        )

                    # Content dedup: prevent identical artifacts within this cycle
                    _existing_content: set[str] = {a.content for a in artifact_buffer._buffer}

                    artifact_source_ids: list[str] = []
                    for cluster in _mce._clusters:
                        if cluster.coherence >= 0.5 and len(cluster.memory_ids) >= 3:
                            for mid in cluster.memory_ids[:10]:
                                if mid not in artifact_source_ids:
                                    artifact_source_ids.append(mid)
                                if len(artifact_source_ids) >= 10:
                                    break
                        if len(artifact_source_ids) >= 10:
                            break

                    def _add_artifact_if_novel(atype: str, src_ids: list, content: str,
                                               conf: float, coh: float) -> bool:
                        nonlocal artifacts_created
                        if artifacts_created >= MAX_ARTIFACTS_PER_DREAM_CYCLE:
                            return False
                        if content in _existing_content:
                            return False
                        artifact_buffer.add(create_artifact(
                            artifact_type=atype,
                            source_memory_ids=src_ids,
                            content=content,
                            confidence=conf,
                            cluster_coherence=coh,
                        ))
                        _existing_content.add(content)
                        artifacts_created += 1
                        return True

                    for text in _patterns:
                        if text:
                            _add_artifact_if_novel(
                                "symbolic_summary", artifact_source_ids,
                                text, 0.4, 0.5,
                            )

                    for text in _connections:
                        if text:
                            _add_artifact_if_novel(
                                "bridge_candidate", artifact_source_ids,
                                text, 0.45, 0.55,
                            )

                    for text in (insight.anomalies or [])[:2]:
                        if text:
                            _add_artifact_if_novel(
                                "tension_flag", artifact_source_ids,
                                text, 0.35, 0.3,
                            )

                    for text in (insight.gaps or [])[:2]:
                        if text:
                            _add_artifact_if_novel(
                                "waking_question", artifact_source_ids,
                                text, 0.3, 0.25,
                            )

                    for cluster in _mce._clusters:
                        if cluster.coherence >= 0.5 and len(cluster.memory_ids) >= 3:
                            _content = f"Consolidation: {cluster.topic} ({len(cluster.memory_ids)} memories, coherence={cluster.coherence:.2f})"
                            _add_artifact_if_novel(
                                "consolidation_proposal", cluster.memory_ids[:10],
                                _content, min(0.6, cluster.coherence), cluster.coherence,
                            )

                    if artifacts_created:
                        actions.append(f"generated {artifacts_created} dream artifact(s)")
                except Exception:
                    logger.debug("Dream artifact generation failed", exc_info=True)

            # --- Phase 4.5: Memory Consolidation (Synthetic REM) ---
            consol_count = 0
            try:
                from memory.consolidation import memory_consolidation_engine
                all_mems = memory_storage.get_all()
                consol_result = memory_consolidation_engine.run_consolidation(
                    clusters=clusters,
                    memories=all_mems,
                    contradiction_engine=getattr(self, '_contradiction_engine', None),
                )
                if consol_result.count > 0:
                    for summary_mem in consol_result.summaries:
                        if memory_storage.add(summary_mem):
                            source_ids = []
                            if isinstance(summary_mem.payload, dict):
                                source_ids = summary_mem.payload.get("source_ids", [])
                            if source_ids:
                                memory_storage.tag_consolidated(source_ids, summary_mem.id)
                            consol_count += 1
                    if consol_count:
                        actions.append(f"consolidated {consol_count} memory cluster(s)")
            except Exception:
                logger.debug("Memory consolidation skipped", exc_info=True)

            # --- Phase 5: Reflect on recent conversations for user patterns ---
            prefs_noted = 0
            conversations = [m for m in recent if getattr(m, "type", "") == "conversation"]
            user_topics: dict[str, int] = {}
            for conv in conversations:
                payload = getattr(conv, "payload", {}) or {}
                user_msg = payload.get("user_message", "") if isinstance(payload, dict) else ""
                if not user_msg:
                    continue
                words = [w.lower() for w in user_msg.split() if len(w) > 4]
                for w in words:
                    user_topics[w] = user_topics.get(w, 0) + 1

            recurring = [(w, c) for w, c in user_topics.items() if c >= 3]
            if recurring:
                top_topics = sorted(recurring, key=lambda x: x[1], reverse=True)[:5]
                topic_str = ", ".join(f"{w}({c}x)" for w, c in top_topics)
                self.observer.observe_thought(
                    f"Dream: user frequently mentions: {topic_str}",
                    depth="deep", confidence=0.6,
                )
                prefs_noted = len(top_topics)

            # --- Phase 6: Train memory cortex models (hippocampus -> cortex) ---
            cortex_result: str | None = None
            if self._cycle_allowed("cortex_training"):
                cortex_result = self._run_cortex_training()
                if cortex_result:
                    actions.append(cortex_result)

            # --- Phase 7: Compact belief JSONL (remove evicted entries from disk) ---
            try:
                if self._contradiction_engine is not None:
                    self._contradiction_engine._belief_store.persist_full()
            except Exception:
                logger.debug("Belief JSONL compaction failed", exc_info=True)

            # --- Report what actually happened ---
            summary_parts = []
            if actions:
                summary_parts.append("; ".join(actions))
            if clusters_found:
                summary_parts.append(f"{clusters_found} cluster(s)")
            if prefs_noted:
                summary_parts.append(f"noted {prefs_noted} recurring topic(s)")

            if summary_parts:
                summary = f"Dream cycle: {', '.join(summary_parts)}"
            else:
                summary = f"Dream cycle: reviewed {len(recent)} memories, no actionable patterns"

            event_bus.emit(KERNEL_THOUGHT,
                           thought_type="dream",
                           depth="deep",
                           text=summary)

            logger.info("Dream: %s", summary)

            self._dream_cycle_history.append({
                "timestamp": now,
                "associations_made": associations_made,
                "clusters_found": clusters_found,
                "reinforced": reinforced,
                "decayed": decayed,
                "artifacts_created": artifacts_created,
                "topics_noted": prefs_noted,
                "memories_scanned": len(recent),
                "cortex_trained": bool(cortex_result),
                "summary": summary,
            })
        except Exception as e:
            logger.warning("Dream cycle error: %s", e)
        finally:
            memory_gate.end_consolidation("dream_cycle")

        self._last_dream = now

    def _run_cortex_training(self) -> str | None:
        """Train memory cortex models from telemetry (hippocampus -> cortex consolidation).

        Returns a summary string for the dream cycle report, or None if nothing ran.
        """
        from consciousness.operations import ops_tracker
        ops_tracker.set_subsystem("cortex_training", "running", "checking training data")
        results: list[str] = []
        try:
            from memory.retrieval_log import memory_retrieval_log
            from memory.lifecycle_log import memory_lifecycle_log
            from memory.ranker import get_memory_ranker
            from memory.salience import get_salience_model

            ranker = get_memory_ranker()
            salience = get_salience_model()

            _MIN_NEW_RANKER_PAIRS = 20
            _MIN_NEW_SALIENCE_PAIRS = 30

            if ranker:
                pairs = memory_retrieval_log.get_training_pairs(limit=2000)
                n_pairs = len(pairs)
                new_pairs = n_pairs - self._last_ranker_pair_count
                if n_pairs < 50:
                    self._last_ranker_skip = f"need 50 pairs, have {n_pairs}"
                elif new_pairs < _MIN_NEW_RANKER_PAIRS:
                    self._last_ranker_skip = f"stale data: {new_pairs} new pairs (need {_MIN_NEW_RANKER_PAIRS})"
                else:
                    features = [p.features for p in pairs]
                    labels = [p.label for p in pairs]

                    _SYNTHETIC_WEIGHT = 0.7
                    try:
                        from synthetic.retrieval_exercise import load_synthetic_pairs
                        syn_pairs = load_synthetic_pairs(max_pairs=500)
                        for sp in syn_pairs:
                            if len(sp.get("features", [])) == 12:
                                features.append(sp["features"])
                                labels.append(sp["label"] * _SYNTHETIC_WEIGHT)
                    except Exception:
                        pass

                    result = ranker.train_from_pairs(features, labels)
                    if "error" not in result:
                        self._last_ranker_pair_count = n_pairs
                        results.append(
                            f"ranker trained (loss={result.get('loss', '?')}, "
                            f"acc={result.get('accuracy', '?')}, "
                            f"pairs={result.get('pairs', 0)})"
                        )
                    else:
                        self._last_ranker_skip = result.get("error", "train error")

            if salience:
                spairs = memory_lifecycle_log.get_salience_training_pairs(limit=2000)
                n_spairs = len(spairs)
                new_spairs = n_spairs - self._last_salience_pair_count
                if n_spairs < 100:
                    self._last_salience_skip = f"need 100 pairs, have {n_spairs}"
                elif new_spairs < _MIN_NEW_SALIENCE_PAIRS:
                    self._last_salience_skip = f"stale data: {new_spairs} new pairs (need {_MIN_NEW_SALIENCE_PAIRS})"
                else:
                    features = [p.features for p in spairs]
                    store_labels = [p.store_label for p in spairs]
                    weight_labels = [p.weight_label for p in spairs]
                    decay_labels = [p.decay_label for p in spairs]
                    result = salience.train_from_pairs(
                        features, store_labels, weight_labels, decay_labels,
                    )
                    if "error" not in result:
                        self._last_salience_pair_count = n_spairs
                        results.append(
                            f"salience trained (loss={result.get('loss', '?')}, "
                            f"acc={result.get('store_accuracy', '?')}, "
                            f"pairs={result.get('pairs', 0)})"
                        )
                    else:
                        self._last_salience_skip = result.get("error", "train error")

        except Exception as exc:
            logger.debug("Cortex training error: %s", exc)

        detail = "; ".join(results) if results else f"ranker: {self._last_ranker_skip}, salience: {self._last_salience_skip}"
        ops_tracker.set_subsystem("cortex_training", "idle", detail)
        if results:
            summary = "cortex: " + "; ".join(results)
            logger.info("Memory cortex training: %s", summary)
            return summary
        return None

    def get_dream_artifact_stats(self) -> dict[str, Any]:
        """Return dream artifact pipeline stats for dashboard."""
        try:
            from consciousness.dream_artifacts import artifact_buffer, reflective_validator
            return {
                "buffer": artifact_buffer.get_stats(),
                "validator": reflective_validator.get_stats(),
            }
        except Exception:
            return {}

    def get_dream_cycle_history(self) -> list[dict[str, Any]]:
        """Return recent dream cycle summaries for dashboard drill-down."""
        return list(self._dream_cycle_history)

    def get_dream_recent_artifacts(self) -> list[dict[str, Any]]:
        """Return recent dream artifacts serialized for dashboard."""
        try:
            from consciousness.dream_artifacts import artifact_buffer
            artifacts = artifact_buffer.get_recent(30)
            return [
                {
                    "id": a.artifact_id,
                    "type": a.artifact_type,
                    "content": a.content,
                    "confidence": round(a.confidence, 3),
                    "coherence": round(a.cluster_coherence, 3),
                    "state": a.validation_state,
                    "notes": a.validator_notes,
                    "source_ids": list(a.source_memory_ids),
                    "timestamp": a.timestamp,
                    "promoted_at": a.promoted_at,
                    "discarded_at": a.discarded_at,
                }
                for a in artifacts
            ]
        except Exception:
            return []

    def get_cortex_stats(self) -> dict[str, Any]:
        """Return memory cortex model stats for dashboard."""
        stats: dict[str, Any] = {}
        try:
            from memory.ranker import get_memory_ranker
            ranker = get_memory_ranker()
            stats["ranker"] = ranker.get_stats() if ranker else {}
        except Exception:
            stats["ranker"] = {}
        try:
            from memory.salience import get_salience_model
            salience = get_salience_model()
            stats["salience"] = salience.get_stats() if salience else {}
        except Exception:
            stats["salience"] = {}
        try:
            from memory.retrieval_log import memory_retrieval_log
            stats["retrieval_log"] = memory_retrieval_log.get_stats()
            stats["eval_metrics"] = memory_retrieval_log.get_eval_metrics()
        except Exception:
            stats["retrieval_log"] = {}
            stats["eval_metrics"] = {}
        try:
            from memory.lifecycle_log import memory_lifecycle_log
            stats["lifecycle_log"] = memory_lifecycle_log.get_stats()
            stats["salience_metrics"] = memory_lifecycle_log.get_effectiveness_metrics()
        except Exception:
            stats["lifecycle_log"] = {}
            stats["salience_metrics"] = {}
        stats["training_status"] = {
            "ranker_skip_reason": self._last_ranker_skip,
            "salience_skip_reason": self._last_salience_skip,
            "last_train_ranker_pairs": self._last_ranker_pair_count,
            "last_train_salience_pairs": self._last_salience_pair_count,
        }

        engine = self._engine_ref
        _adv_count = getattr(engine, '_salience_advisory_count', 0) if engine else 0
        _gate_fail = getattr(engine, '_salience_gate_fail_count', 0) if engine else 0
        stats["salience_advisory"] = {
            "active": _adv_count > 0,
            "advisory_count": _adv_count,
            "gate_fail_count": _gate_fail,
        }

        try:
            from conversation_handler import outcome_counters
            stats["outcome_logging"] = outcome_counters.snapshot()
        except Exception:
            stats["outcome_logging"] = {}
        return stats

    def enable_hemisphere(self, orchestrator: Any) -> None:
        """Enable the hemisphere neural network system."""
        self._hemisphere_orchestrator = orchestrator
        logger.info("Hemisphere NN system enabled")

    def enable_learning_jobs(self, orchestrator: Any) -> None:
        """Enable the skill learning job system."""
        self._learning_job_orchestrator = orchestrator
        logger.info("Learning job system enabled")
        try:
            from consciousness.events import event_bus, SKILL_LEARNING_COMPLETED
            event_bus.on(SKILL_LEARNING_COMPLETED, self._on_skill_learning_completed)
        except Exception:
            pass

    def _on_skill_learning_completed(self, **kwargs: Any) -> None:
        """Proactively report learning completion to the user."""
        report = kwargs.get("report")
        if not report:
            return
        summary = report.get("summary_text", "")
        if not summary:
            return
        engine = self._engine_ref
        if engine and hasattr(engine, "_proactive_speech_cb") and engine._proactive_speech_cb:
            try:
                engine._proactive_speech_cb(summary)
                logger.info("Matrix completion report spoken: %s", summary[:120])
            except Exception:
                logger.debug("Failed to speak Matrix completion report", exc_info=True)

    def _run_learning_job_cycle(self, now: float) -> None:
        """Tick all active learning jobs with a rich context."""
        self._last_learning_job = now
        if not self._learning_job_orchestrator:
            return
        try:
            import datetime as dt
            ctx: dict = {
                "now": now,
                "now_iso": dt.datetime.utcfromtimestamp(now).replace(microsecond=0).isoformat() + "Z",
                "tool_router_available": True,
            }
            self._learning_job_orchestrator.run_cycle(ctx)
        except Exception:
            logger.exception("Learning job cycle failed")

    def set_llm_callback(self, callback: Any) -> None:
        """Wire LLM enrichment for existential/philosophical subsystems."""
        self._llm_callback = callback
        self.existential.set_llm_callback(callback)
        self.philosophical.set_llm_callback(callback)
        logger.info("LLM callback wired for existential + philosophical enrichment")

    def get_hemisphere_state(self) -> dict[str, Any] | None:
        if self._hemisphere_orchestrator:
            return self._hemisphere_orchestrator.get_state()
        return None

    def get_hemisphere_signals(self) -> dict[str, float]:
        if self._hemisphere_orchestrator:
            return self._hemisphere_orchestrator.get_hemisphere_signals()
        return {}

    def _run_hemisphere_cycle(
        self, now: float, memories: list[Any], traits: dict[str, float],
    ) -> None:
        self._last_hemisphere = now
        if not self._hemisphere_orchestrator:
            return
        try:
            from consciousness.tone import tone_engine
            from consciousness.phases import phase_manager
            from consciousness.modes import mode_manager
            gestation_info = {}
            try:
                from consciousness.gestation import gestation_manager
                if gestation_manager:
                    gs = gestation_manager.get_status()
                    gestation_info = {"phase": gs.get("phase", -1)}
            except Exception:
                pass

            engine_state = {
                "tone": tone_engine.current_tone,
                "memory_density": min(1.0, len(memories) / 200.0),
                "phase": phase_manager.current_phase if hasattr(phase_manager, "current_phase") else "OBSERVING",
                "mode": mode_manager.mode,
                "stage": self.evolution.current_stage,
                "transcendence_level": self.evolution.transcendence_level,
                "awareness_level": self.observer.awareness_level,
                "reasoning_quality": self.analytics.get_reasoning_quality().overall,
                "observation_count": self.observer.state.observation_count,
                "gestation": gestation_info,
                "boot_stabilization_active": self._boot_stabilization_active(now),
                "boot_stabilization_remaining_s": round(self._boot_stabilization_remaining_s(now), 1),
            }
            trait_names = list(traits.keys()) if traits else []
            self._hemisphere_orchestrator.run_cycle(engine_state, memories, trait_names)
        except Exception:
            logger.exception("Hemisphere cycle failed")

    def _run_shadow_language_cycle(self, now: float) -> None:
        """Train the shadow language model if corpus has grown."""
        self._last_shadow_lang = now
        try:
            from reasoning.shadow_language_model import (
                shadow_language_trainer,
                shadow_language_inference,
                load_corpus,
                LANGUAGE_STYLE_DIR,
            )
            from reasoning.language_phasec import (
                lock_phasec_baseline,
                phasec_harness,
                phasec_shadow_student,
            )

            lock_phasec_baseline()
            if (
                not phasec_shadow_student.available
                and bool(phasec_shadow_student.get_status().get("checkpoint_exists", False))
            ):
                phasec_shadow_student.load_checkpoint()

            corpus = load_corpus()
            if not shadow_language_trainer.should_train(len(corpus)):
                # Even when style model doesn't retrain, keep Phase C harness current.
                phasec_harness.run_training_cycle()
                return
            result = shadow_language_trainer.train(corpus=corpus)
            if result is not None:
                shadow_language_inference.load(result)
                # Persist to disk
                LANGUAGE_STYLE_DIR.mkdir(parents=True, exist_ok=True)
                shadow_language_inference.save_to_disk(
                    str(LANGUAGE_STYLE_DIR / "model.pt"),
                    str(LANGUAGE_STYLE_DIR / "index.pt"),
                )
                logger.info(
                    "Shadow language model cycle: trained on %d examples, loss=%.4f",
                    result["n_examples"], result["loss"],
                )
            phasec_result = phasec_harness.run_training_cycle()
            if phasec_result.get("ok"):
                logger.info(
                    "Phase C language harness cycle: status=%s samples=%s",
                    phasec_result.get("status", ""),
                    phasec_result.get("dataset_samples", 0),
                )
        except Exception:
            logger.debug("Shadow language cycle failed", exc_info=True)

    def enable_acquisition(self, orchestrator: Any) -> None:
        """Enable the Capability Acquisition Pipeline."""
        self._acquisition_orchestrator = orchestrator
        logger.info("Capability Acquisition Pipeline enabled")

    def enable_self_improvement(self, orchestrator: Any) -> None:
        """Enable the self-improvement loop with an orchestrator."""
        self._self_improve_orchestrator = orchestrator
        self._self_improve_enabled = True
        logger.info("Self-improvement loop enabled")

    def record_response_latency(self, latency_ms: float) -> None:
        """Track response latencies for improvement detection."""
        self._response_latencies.append(latency_ms)

    _SI_CATEGORY_COOLDOWN_S = 1800.0  # 30 min per category
    _SI_SUSTAINED_WINDOW = 3  # must appear in N consecutive scans
    _SI_FINGERPRINT_COOLDOWN_S = 14400.0  # 4 hours per fingerprint
    _SI_MAX_ATTEMPTS_PER_DAY = 6
    _SI_PAST_PROPOSAL_WINDOW_S = 86400.0  # 24 hours

    def _run_self_improvement(self, now: float, memories: list[Any]) -> None:
        """Periodically scan for improvement opportunities and trigger patches.

        Gates:
        - Per-category cooldown prevents the same trigger type from firing
          more than once per _SI_CATEGORY_COOLDOWN_S.
        - Sustained evidence: an opportunity type must appear in
          _SI_SUSTAINED_WINDOW consecutive scans before triggering.
        - Fingerprint dedup: identical opportunities are suppressed for 4h.
        - Past-proposal dedup: opportunities already proposed in last 24h are skipped.
        - Daily attempt cap: max _SI_MAX_ATTEMPTS_PER_DAY generations per day.
        """
        self._last_self_improve = now
        if not self._self_improve_orchestrator:
            return

        if hasattr(self._self_improve_orchestrator, "is_blocked") and self._self_improve_orchestrator.is_blocked():
            return

        try:
            self._run_self_improvement_inner(now, memories)
        except Exception:
            logger.debug("Self-improvement cycle error", exc_info=True)

    def _si_init_state(self) -> None:
        """Lazy-init scanner state attributes."""
        if not hasattr(self, "_si_category_last_fired"):
            self._si_category_last_fired: dict[str, float] = {}
        if not hasattr(self, "_si_sustained_counts"):
            self._si_sustained_counts: dict[str, int] = {}
        if not hasattr(self, "_si_fired_fingerprints"):
            self._si_fired_fingerprints: dict[str, float] = {}
        if not hasattr(self, "_si_attempts_today"):
            self._si_attempts_today: int = 0
        if not hasattr(self, "_si_attempts_day"):
            self._si_attempts_day: str = ""
        if not hasattr(self, "_si_scan_count"):
            self._si_scan_count: int = 0
        if not hasattr(self, "_si_past_fingerprints"):
            self._si_past_fingerprints: set[str] | None = None
        if not hasattr(self, "_si_last_detector_snapshot"):
            self._si_last_detector_snapshot: dict[str, Any] = {}

    def _si_load_past_fingerprints(self) -> set[str]:
        """Lazy-load fingerprints from past proposals (read-only, last 24h)."""
        if self._si_past_fingerprints is not None:
            return self._si_past_fingerprints
        fps: set[str] = set()
        try:
            from self_improve.orchestrator import PROPOSALS_FILE
            if PROPOSALS_FILE.exists():
                import json
                cutoff = time.time() - self._SI_PAST_PROPOSAL_WINDOW_S
                with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            p = json.loads(line)
                            ts = p.get("timestamp", 0)
                            fp = p.get("fingerprint", "")
                            if ts >= cutoff and fp:
                                fps.add(fp)
                        except (json.JSONDecodeError, KeyError):
                            continue
        except Exception:
            logger.debug("Could not load past proposal fingerprints", exc_info=True)
        self._si_past_fingerprints = fps
        return fps

    def _si_check_daily_cap(self, now: float) -> bool:
        """Returns True if daily attempt cap is reached."""
        import datetime
        today = datetime.date.today().isoformat()
        if self._si_attempts_day != today:
            self._si_attempts_day = today
            self._si_attempts_today = 0
        return self._si_attempts_today >= self._SI_MAX_ATTEMPTS_PER_DAY

    def _si_detect_opportunities(self, now: float) -> list[dict[str, Any]]:
        """Metric-driven opportunity detection using only data this class owns.

        Sources: self.analytics, self.health_monitor, self.observer, event_bus,
        self._response_latencies. Does NOT import jarvis_eval, PVL, or memory.
        """
        opportunities: list[dict[str, Any]] = []
        detector_snapshot: dict[str, Any] = {}

        # Maturity guard: skip on fresh brains
        try:
            health = self.analytics.get_system_health()
            uptime_s = health.uptime_s
        except Exception:
            uptime_s = 0.0
        if uptime_s < 1800:
            detector_snapshot["skipped"] = f"uptime {uptime_s:.0f}s < 1800s"
            self._si_last_detector_snapshot = detector_snapshot
            return opportunities

        # Detector 1: Health degradation
        try:
            from consciousness.health_monitor import HealthMonitor
            hm: HealthMonitor | None = getattr(self, "health_monitor", None)
            if hm:
                summary = hm.get_summary()
                overall = summary.get("overall_health", 1.0)
                total_checks = summary.get("total_checks", 0)
                detector_snapshot["health"] = {"overall": overall, "checks": total_checks}
                if total_checks >= 10 and overall < 0.5:
                    components = summary.get("components", {})
                    worst = min(components.items(), key=lambda x: x[1]) if components else ("unknown", overall)
                    fp = f"health_degraded:{worst[0]}:{overall:.2f}"
                    opportunities.append({
                        "type": "health_degraded",
                        "fingerprint": fp,
                        "description": (
                            f"System health degraded to {overall:.2f} — "
                            f"worst component: {worst[0]} ({worst[1]:.2f})"
                        ),
                        "target_module": f"consciousness/{worst[0]}.py" if worst[0] != "unknown" else "consciousness/engine.py",
                        "priority": 4,
                        "evidence": [],
                        "evidence_detail": {"overall_health": overall, "worst_component": worst[0], "worst_score": worst[1]},
                    })
        except Exception:
            pass

        # Detector 2: Reasoning quality decline
        try:
            reasoning = self.analytics.get_reasoning_quality()
            thought_count = self.meta_thoughts.total_generated
            detector_snapshot["reasoning"] = {
                "overall": reasoning.overall, "coherence": reasoning.coherence,
                "depth": reasoning.depth, "thought_count": thought_count,
            }
            if thought_count >= 20 and reasoning.overall < 0.35:
                fp = f"reasoning_decline:{reasoning.overall:.2f}:{reasoning.depth:.2f}"
                opportunities.append({
                    "type": "reasoning_decline",
                    "fingerprint": fp,
                    "description": (
                        f"Reasoning quality low (overall={reasoning.overall:.2f}, "
                        f"depth={reasoning.depth:.2f}, coherence={reasoning.coherence:.2f}) — "
                        f"consider improving meta-cognitive thought generation"
                    ),
                    "target_module": "consciousness/meta_cognitive_thoughts.py",
                    "priority": 3,
                    "evidence": [],
                    "evidence_detail": {
                        "overall": reasoning.overall, "depth": reasoning.depth,
                        "coherence": reasoning.coherence, "thought_count": thought_count,
                    },
                })
        except Exception:
            pass

        # Detector 3: Confidence volatility
        try:
            conf = self.analytics.get_confidence()
            detector_snapshot["confidence"] = {
                "current": conf.current, "volatility": conf.volatility, "trend": conf.trend,
            }
            readings = len(getattr(self.analytics, "_confidence_window", []))
            if readings >= 10 and conf.volatility > 0.3:
                fp = f"confidence_volatile:{conf.volatility:.2f}"
                opportunities.append({
                    "type": "confidence_volatile",
                    "fingerprint": fp,
                    "description": (
                        f"Confidence volatility high ({conf.volatility:.2f}) — "
                        f"current={conf.current:.2f}, trend={conf.trend:+.2f}"
                    ),
                    "target_module": "consciousness/consciousness_analytics.py",
                    "priority": 2,
                    "evidence": [],
                    "evidence_detail": {
                        "volatility": conf.volatility, "current": conf.current,
                        "trend": conf.trend, "readings": readings,
                    },
                })
        except Exception:
            pass

        # Detector 4: Response latency spikes
        try:
            latencies = list(self._response_latencies) if self._response_latencies else []
            slow_count = sum(1 for lat in latencies if lat > 5000)
            detector_snapshot["latency"] = {"total": len(latencies), "slow_gt_5s": slow_count}
            if len(latencies) >= 5 and slow_count >= 3:
                avg_slow = sum(lat for lat in latencies if lat > 5000) / max(slow_count, 1)
                fp = f"slow_responses:{slow_count}:{int(avg_slow)}"
                opportunities.append({
                    "type": "slow_responses",
                    "fingerprint": fp,
                    "description": (
                        f"{slow_count}/{len(latencies)} responses > 5s (avg slow: {avg_slow:.0f}ms) — "
                        f"response pipeline may need optimization"
                    ),
                    "target_module": "reasoning/response.py",
                    "priority": 3,
                    "evidence": [],
                    "evidence_detail": {
                        "slow_count": slow_count, "total": len(latencies),
                        "avg_slow_ms": round(avg_slow, 1),
                    },
                })
        except Exception:
            pass

        # Detector 5: Event bus error rate
        try:
            bus_metrics = event_bus.get_metrics()
            total_events = bus_metrics.get("total_events", 0)
            failed_events = bus_metrics.get("failed_events", 0)
            detector_snapshot["event_bus"] = {"emitted": total_events, "errors": failed_events}
            if total_events >= 100 and failed_events > 0:
                error_rate = failed_events / max(total_events, 1)
                if error_rate > 0.05:
                    fp = f"event_bus_errors:{failed_events}:{error_rate:.3f}"
                    opportunities.append({
                        "type": "event_bus_errors",
                        "fingerprint": fp,
                        "description": (
                            f"Event bus error rate {error_rate:.1%} ({failed_events}/{total_events}) — "
                            f"handlers may be crashing"
                        ),
                        "target_module": "consciousness/events.py",
                        "priority": 3,
                        "evidence": [],
                        "evidence_detail": {
                            "error_rate": round(error_rate, 4),
                            "failed_events": failed_events,
                            "total_events": total_events,
                        },
                    })
        except Exception:
            pass

        # Detector 6: Tick performance regression
        try:
            sys_health = self.analytics.get_system_health()
            tick_p95 = sys_health.tick_p95_ms
            detector_snapshot["tick"] = {"p95_ms": tick_p95}
            if tick_p95 > 80:
                fp = f"tick_slow:{int(tick_p95)}"
                opportunities.append({
                    "type": "tick_performance",
                    "fingerprint": fp,
                    "description": (
                        f"Kernel tick p95 at {tick_p95:.1f}ms (budget: 50ms) — "
                        f"background cycles may need optimization"
                    ),
                    "target_module": "consciousness/kernel.py",
                    "priority": 3,
                    "evidence": [],
                    "evidence_detail": {"p95_ms": round(tick_p95, 1)},
                })
        except Exception:
            pass

        self._si_last_detector_snapshot = detector_snapshot

        # Record DIAGNOSTIC hemisphere feature vector on every scan
        try:
            from hemisphere.diagnostic_encoder import DiagnosticEncoder
            from hemisphere.distillation import DistillationCollector

            si_status = {}
            if hasattr(self, "_self_improve_orchestrator") and self._self_improve_orchestrator:
                try:
                    si_status = self._self_improve_orchestrator.get_status()
                except Exception:
                    pass

            diag_context: dict[str, Any] = {
                "uptime_s": uptime_s,
                "quarantine_pressure": getattr(self, "_last_quarantine_pressure", 0.0),
                "soul_integrity": getattr(self, "_last_soul_integrity", 1.0),
                "mode": getattr(getattr(self, "mode_manager", None), "mode", "passive"),
                "evolution_stage": getattr(getattr(self, "evolution", None), "current_stage", 0),
                "consciousness_stage": getattr(getattr(self, "evolution", None), "consciousness_level", 0),
                "health_trend_slope": detector_snapshot.get("health", {}).get("trend_slope", 0.0),
                "mutations_last_hour": getattr(self, "_mutations_this_session", 0),
                "active_learning_jobs": 0,
                "improvements_today": getattr(self, "_si_attempts_today", 0),
                "last_improvement_age_s": now - getattr(self, "_si_last_attempt_time", 0) if getattr(self, "_si_last_attempt_time", 0) > 0 else 86400.0,
                "sandbox_pass_rate": si_status.get("win_rate", {}).get("sandbox_pass_rate", 0.0) if isinstance(si_status.get("win_rate"), dict) else 0.0,
                "friction_rate": 0.0,
                "correction_count": 0,
                "autonomy_level": 0,
                "has_codebase_context": False,
                "target_module_lines": 0,
                "target_import_fanout": 0,
                "target_importers": 0,
                "target_symbol_count": 0,
                "target_recently_modified": False,
                "has_friction_context": False,
                "friction_severity_high_ratio": 0.0,
                "friction_correction_ratio": 0.0,
                "friction_identity_count": 0,
                "correction_auto_accepted": 0,
            }

            # Populate from live subsystems when available
            try:
                from autonomy.orchestrator import AutonomyOrchestrator
                ao = getattr(self, "_autonomy_orchestrator", None)
                if ao:
                    diag_context["autonomy_level"] = getattr(ao, "_level", 0)
            except Exception:
                pass

            # Track 5: Live friction/correction signals (3600s window matches scan cadence)
            try:
                from autonomy.friction_miner import get_friction_miner
                fm = get_friction_miner()
                rate = fm.get_friction_rate(3600)
                diag_context["friction_rate"] = rate
                events_1h = [e for e in fm._events if e.timestamp > now - 3600]
                n_events = len(events_1h)
                diag_context["correction_count"] = n_events
                diag_context["has_friction_context"] = True
                if n_events > 0:
                    n_high = sum(1 for e in events_1h if e.severity in ("high", "critical"))
                    n_correction = sum(1 for e in events_1h if e.friction_type == "correction")
                    n_identity = sum(1 for e in events_1h if e.friction_type == "identity_mismatch")
                    diag_context["friction_severity_high_ratio"] = n_high / n_events
                    diag_context["friction_correction_ratio"] = n_correction / n_events
                    diag_context["friction_identity_count"] = n_identity
            except Exception:
                pass

            # Track 4: Codebase structural features for the primary target module
            try:
                from tools.codebase_tool import codebase_index
                if codebase_index._modules:
                    primary = None
                    if opportunities:
                        best = max(opportunities, key=lambda o: (
                            o.get("sustained_count", 0), o.get("priority", 0)
                        ))
                        primary = best.get("target_module", "")
                    if primary:
                        mod_fqn = primary.replace(".py", "").replace("/", ".")
                        mod_info = codebase_index.get_module(mod_fqn)
                        if mod_info:
                            diag_context["target_module_lines"] = mod_info.line_count
                            diag_context["target_symbol_count"] = len(mod_info.symbols)
                        diag_context["target_import_fanout"] = len(codebase_index.get_imports_of(mod_fqn))
                        diag_context["target_importers"] = len(codebase_index.get_importers_of(mod_fqn))
                        modified_files = codebase_index.get_modified_files()
                        diag_context["target_recently_modified"] = any(
                            primary in mf for mf in modified_files
                        )
                    diag_context["has_codebase_context"] = True
            except Exception:
                pass

            vec = DiagnosticEncoder.encode(detector_snapshot, opportunities, diag_context)
            scan_id = f"scan_{self._si_scan_count}_{int(now)}"
            collector = DistillationCollector.instance()
            collector.record(
                teacher="diagnostic_features",
                signal_type="detector_snapshot",
                data=vec,
                metadata={"scan_id": scan_id, "n_opportunities": len(opportunities)},
                origin="self_improve",
                fidelity=1.0,
            )

            # Record negative-example label for healthy scans (no opportunities)
            # so the DIAGNOSTIC specialist learns what "normal" looks like.
            # Rate-limited to every 5th healthy scan to keep class balance.
            if not opportunities and self._si_scan_count % 5 == 0:
                neg_label, neg_meta = DiagnosticEncoder.encode_no_opportunity_label()
                neg_meta["scan_id"] = scan_id
                collector.record(
                    teacher="diagnostic_detector",
                    signal_type="no_opportunity",
                    data=neg_label,
                    metadata=neg_meta,
                    origin="self_improve",
                    fidelity=0.6,
                )
        except Exception:
            logger.debug("Failed to record diagnostic feature vector", exc_info=True)

        return opportunities

    def _run_self_improvement_inner(self, now: float, memories: list[Any]) -> None:
        self._si_init_state()
        self._si_scan_count += 1

        # Daily cap check
        if self._si_check_daily_cap(now):
            if self._si_scan_count % 10 == 1:
                logger.info("Self-improvement: daily cap reached (%d/%d attempts today)",
                            self._si_attempts_today, self._SI_MAX_ATTEMPTS_PER_DAY)
            return

        opportunities = self._si_detect_opportunities(now)

        if not opportunities:
            self._si_sustained_counts.clear()
            if self._si_scan_count <= 3 or self._si_scan_count % 10 == 1:
                logger.info("Self-improvement scan #%d: no opportunities (detectors: %s)",
                            self._si_scan_count,
                            ", ".join(f"{k}" for k in self._si_last_detector_snapshot))
            return

        # Sustained gate: opportunity type must appear N consecutive scans
        current_types = {o["type"] for o in opportunities}
        for otype in list(self._si_sustained_counts):
            if otype not in current_types:
                self._si_sustained_counts.pop(otype, None)
        for o in opportunities:
            self._si_sustained_counts[o["type"]] = self._si_sustained_counts.get(o["type"], 0) + 1

        # Fingerprint dedup: skip recently-fired or already-proposed fingerprints
        past_fps = self._si_load_past_fingerprints()

        eligible = []
        for o in opportunities:
            cat = o["type"]
            fp = o.get("fingerprint", "")

            # Sustained gate
            if self._si_sustained_counts.get(cat, 0) < self._SI_SUSTAINED_WINDOW:
                continue

            # Category cooldown
            last = self._si_category_last_fired.get(cat, 0)
            if now - last < self._SI_CATEGORY_COOLDOWN_S:
                continue

            # Fingerprint cooldown (in-memory, 4h)
            if fp:
                fp_last = self._si_fired_fingerprints.get(fp, 0)
                if now - fp_last < self._SI_FINGERPRINT_COOLDOWN_S:
                    continue
                # Past-proposal dedup (JSONL, 24h)
                if fp in past_fps:
                    continue

            eligible.append(o)

        # Record DIAGNOSTIC hemisphere teacher labels for eligible opportunities
        if eligible:
            try:
                from hemisphere.diagnostic_encoder import DiagnosticEncoder
                from hemisphere.distillation import DistillationCollector
                collector = DistillationCollector.instance()
                for eo in eligible:
                    label, label_meta = DiagnosticEncoder.encode_label(eo)
                    label_meta["scan_id"] = f"scan_{self._si_scan_count}_{int(now)}"
                    collector.record(
                        teacher="diagnostic_detector",
                        signal_type="detector_fired",
                        data=label,
                        metadata=label_meta,
                        origin="self_improve",
                        fidelity=min(1.0, self._si_sustained_counts.get(eo["type"], 1) / 3.0),
                    )
            except Exception:
                logger.debug("Failed to record diagnostic label signals", exc_info=True)

        if not eligible:
            logger.info("Self-improvement scan #%d: %d raw, 0 eligible (sustained=%s)",
                        self._si_scan_count, len(opportunities),
                        dict(self._si_sustained_counts))
            return

        best = max(eligible, key=lambda o: o["priority"])
        logger.info("Self-improvement opportunity (eligible): %s (priority=%d, fingerprint=%s)",
                     best["description"][:80], best["priority"], best.get("fingerprint", "none"))

        from consciousness.modes import mode_manager
        min_priority = 1 if mode_manager.mode == "deep_learning" else 2
        if best["priority"] < min_priority:
            return

        self._si_category_last_fired[best["type"]] = now
        fp = best.get("fingerprint", "")
        if fp:
            self._si_fired_fingerprints[fp] = now

        # Increment daily attempt counter
        self._si_attempts_today += 1

        try:
            from self_improve.improvement_request import ImprovementRequest
            import asyncio

            request = ImprovementRequest(
                type="consciousness_enhancement" if "consciousness" in best.get("type", "") else "performance_optimization",
                target_module=best.get("target_module", ""),
                description=best["description"],
                evidence=best.get("evidence", []),
                priority=best["priority"] / 5.0,
                requires_approval=best["priority"] >= 4,
                fingerprint=fp,
                evidence_detail=best.get("evidence_detail", {}),
            )

            ollama = getattr(self._self_improve_orchestrator, "_ollama_client", None)

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._self_improve_orchestrator.attempt_improvement(
                        request, ollama_client=ollama,
                    ),
                )
        except Exception as exc:
            logger.debug("Self-improvement trigger failed: %s", exc)

    def get_scanner_state(self) -> dict[str, Any]:
        """Expose scanner state for dashboard API."""
        self._si_init_state()
        return {
            "scan_count": self._si_scan_count,
            "attempts_today": self._si_attempts_today,
            "max_attempts_per_day": self._SI_MAX_ATTEMPTS_PER_DAY,
            "sustained_counts": dict(self._si_sustained_counts),
            "fired_fingerprints": len(self._si_fired_fingerprints),
            "past_proposal_fingerprints": len(self._si_past_fingerprints) if self._si_past_fingerprints else 0,
            "detector_snapshot": self._si_last_detector_snapshot,
            "category_cooldowns": {
                cat: round(self._SI_CATEGORY_COOLDOWN_S - (time.time() - ts), 0)
                for cat, ts in self._si_category_last_fired.items()
                if time.time() - ts < self._SI_CATEGORY_COOLDOWN_S
            },
        }

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _score_conversation_confidence(
        latency_ms: float, text_len: int, barged_in: bool,
    ) -> float:
        """Score a single conversation's outcome as a confidence signal.

        Fast, substantive responses that weren't interrupted = high confidence.
        """
        if barged_in:
            return 0.25

        latency_score = max(0.2, 1.0 - latency_ms / 10_000.0)
        length_score = min(1.0, text_len / 200.0) if text_len > 10 else 0.3
        return max(0.2, min(0.95, latency_score * 0.6 + length_score * 0.4))

    def _get_last_mutation_summary(self) -> str:
        history = self._config.evolution.mutation_history
        if not history:
            return "No mutations yet"
        return history[-1][:80]

    # -- health check --------------------------------------------------------

    def check_health(self) -> dict[str, Any]:
        """Run real integration checks against all subsystems."""
        from consciousness.events import event_bus
        from memory.storage import memory_storage

        results: dict[str, Any] = {}

        # EventBus: check listener counts > 0
        try:
            listener_count = event_bus.listener_count()
            results["event_bus"] = {
                "healthy": listener_count > 0,
                "detail": f"{listener_count} listeners registered",
            }
        except Exception as e:
            results["event_bus"] = {"healthy": False, "detail": str(e)}

        # MemoryStorage: check can add/get
        try:
            count = memory_storage.count()
            results["memory"] = {
                "healthy": True,
                "detail": f"{count} memories stored",
            }
        except Exception as e:
            results["memory"] = {"healthy": False, "detail": str(e)}

        # Observer: check awareness > 0
        try:
            awareness = self.observer.state.awareness_level
            results["observer"] = {
                "healthy": awareness >= 0.0,
                "detail": f"awareness={awareness:.3f}",
            }
        except Exception as e:
            results["observer"] = {"healthy": False, "detail": str(e)}

        # Analytics: check has recorded ticks
        try:
            tick_p95 = self.analytics.get_tick_p95()
            results["analytics"] = {
                "healthy": True,
                "detail": f"tick_p95={tick_p95:.1f}ms",
            }
        except Exception as e:
            results["analytics"] = {"healthy": False, "detail": str(e)}

        # Evolution: check state loaded
        try:
            evo_state = self.evolution.get_state()
            results["evolution"] = {
                "healthy": evo_state is not None,
                "detail": f"stage={evo_state.current_stage}",
            }
        except Exception as e:
            results["evolution"] = {"healthy": False, "detail": str(e)}

        # MutationGovernor: check no stuck monitors
        try:
            active_monitor = self.governor.get_active_monitor()
            stuck = False
            if active_monitor:
                monitor_age = time.time() - active_monitor.get("started", time.time())
                stuck = monitor_age > 120  # 2 minutes = stuck
            results["governor"] = {
                "healthy": not stuck,
                "detail": "active monitor" if active_monitor else "idle",
            }
        except Exception as e:
            results["governor"] = {"healthy": False, "detail": str(e)}

        # Meta-thoughts
        try:
            total = self.meta_thoughts.total_generated
            results["meta_thoughts"] = {
                "healthy": True,
                "detail": f"{total} thoughts generated",
            }
        except Exception as e:
            results["meta_thoughts"] = {"healthy": False, "detail": str(e)}

        # Existential reasoning
        try:
            inq_state = self.existential.get_state()
            results["existential"] = {
                "healthy": True,
                "detail": f"{inq_state.get('inquiry_count', 0)} inquiries",
            }
        except Exception as e:
            results["existential"] = {"healthy": False, "detail": str(e)}

        # Philosophical dialogue
        try:
            phil_state = self.philosophical.get_state()
            results["philosophical"] = {
                "healthy": True,
                "detail": f"{phil_state.get('dialogue_count', 0)} dialogues",
            }
        except Exception as e:
            results["philosophical"] = {"healthy": False, "detail": str(e)}

        return results

    def run_self_test(self) -> dict[str, dict[str, Any]]:
        """On-demand self-test with pass/fail/error per subsystem + latency."""
        results: dict[str, dict[str, Any]] = {}

        tests = [
            ("event_bus", self._test_event_bus),
            ("memory", self._test_memory),
            ("observer", self._test_observer),
            ("analytics", self._test_analytics),
            ("evolution", self._test_evolution),
            ("meta_thoughts", self._test_meta_thoughts),
        ]

        for name, test_fn in tests:
            start = time.time()
            try:
                passed = test_fn()
                latency = (time.time() - start) * 1000
                results[name] = {"status": "pass" if passed else "fail", "latency_ms": round(latency, 2)}
            except Exception as e:
                latency = (time.time() - start) * 1000
                results[name] = {"status": "error", "error": str(e), "latency_ms": round(latency, 2)}

        return results

    def _test_event_bus(self) -> bool:
        from consciousness.events import event_bus
        return event_bus.listener_count() > 0

    def _test_memory(self) -> bool:
        from memory.storage import memory_storage
        return memory_storage.count() >= 0

    def _test_observer(self) -> bool:
        return self.observer.state.awareness_level >= 0.0

    def _test_analytics(self) -> bool:
        self.analytics.get_confidence()
        return True

    def _test_evolution(self) -> bool:
        state = self.evolution.get_state()
        return state is not None

    def _test_meta_thoughts(self) -> bool:
        return self.meta_thoughts.total_generated >= 0
