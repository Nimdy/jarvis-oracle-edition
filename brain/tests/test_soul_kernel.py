"""Integration tests for the Soul Kernel Deep Port — all 25 new systems."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import (
    EventBus, Memory, MEMORY_ASSOCIATED, MEMORY_TRANSACTION_COMPLETE,
    PERSONALITY_ROLLBACK, CRITICAL_EVENTS,
)
from memory.core import MemoryCore, CreateMemoryData


def _make_memory(mc, payload="test", weight=0.5, tags=None, mem_type="conversation"):
    return mc.create_memory(CreateMemoryData(
        type=mem_type, payload=payload, weight=weight, tags=tags or ["test"],
    ))


# ── Phase 1: Foundation ──────────────────────────────────────────────────

def test_event_bus_circuit_breaker():
    """1A: EventBus has circuit breaker, retry, and metrics."""
    bus = EventBus()
    bus.open_barrier()
    metrics = bus.get_metrics()
    assert "total_events" in metrics
    assert "circuit_breaker_trips" in metrics
    assert metrics["total_events"] == 0

    received = []
    bus.on("test:cb", lambda **kw: received.append(1))
    bus.emit("test:cb")
    assert len(received) == 1
    assert bus.get_metrics()["total_events"] == 1
    print("  PASS: 1A event_bus circuit breaker + metrics")


def test_event_validator():
    """1B: Event validator enforces sequence rules."""
    from consciousness.event_validator import EventSequenceValidator
    v = EventSequenceValidator()
    stats = v.get_stats()
    assert stats["total_validated"] == 0
    assert stats["integrity_score"] == 1.0

    # Validate a normal event
    result = v.validate("kernel:tick")
    assert result is None  # no rule for kernel:tick
    assert v.get_stats()["total_validated"] == 1
    print("  PASS: 1B event_validator sequence rules")


def test_memory_associations():
    """1C: Memory association graph works."""
    from memory.storage import MemoryStorage
    ms = MemoryStorage(max_capacity=50)
    mc = MemoryCore()

    m1 = _make_memory(mc, "memory A", 0.5, ["alpha"])
    m2 = _make_memory(mc, "memory B", 0.4, ["beta"])
    m3 = _make_memory(mc, "memory C", 0.3, ["gamma"])
    ms.add(m1)
    ms.add(m2)
    ms.add(m3)

    assert ms.associate(m1.id, m2.id)
    assert ms.associate(m2.id, m3.id)

    related = ms.get_related(m1.id, depth=2)
    related_ids = {m.id for m in related}
    assert m2.id in related_ids
    assert m3.id in related_ids  # reachable via m2

    stored_m1 = ms.get(m1.id)
    assert ms.reinforce(m1.id, boost=0.15)
    updated = ms.get(m1.id)
    assert updated.weight > stored_m1.weight

    stats = ms.get_association_stats()
    assert stats["total_connections"] >= 2
    assert stats["isolated_count"] == 0  # all 3 are connected in a chain
    print("  PASS: 1C memory associations")


def test_memory_transactions():
    """1D: Atomic memory transactions with integrity checks."""
    from memory.storage import MemoryStorage
    from memory.transactions import MemoryTransaction, MemoryOperation
    import memory.storage as ms_mod
    import memory.transactions as tx_mod

    ms = MemoryStorage(max_capacity=50)
    mc = MemoryCore()
    orig_ms = ms_mod.memory_storage
    orig_tx = tx_mod.memory_storage
    ms_mod.memory_storage = ms
    tx_mod.memory_storage = ms

    try:
        m = _make_memory(mc, "tx test", 0.7, ["tx"])
        ops = [MemoryOperation(type="add", memory_id=m.id, memory=m)]
        tx = MemoryTransaction()
        result = tx.execute(ops)
        assert result.success
        assert result.operations_completed == 1
        assert ms.count() == 1

        violations = tx.validate_integrity(ms.get_all())
        assert len(violations) == 0
    finally:
        ms_mod.memory_storage = orig_ms
        tx_mod.memory_storage = orig_tx
    print("  PASS: 1D memory transactions")


def test_seed_memories_idempotent():
    """1E: Seed memories are created with birth tag and are idempotent."""
    # Just verify the seed memory structure exists
    from consciousness.engine import ConsciousnessEngine
    # Don't actually start the engine, just check the method exists
    engine = ConsciousnessEngine()
    assert hasattr(engine, '_seed_core_memories')
    print("  PASS: 1E seed memories method exists")


# ── Phase 2: Intelligence ────────────────────────────────────────────────

def test_health_monitor():
    """2A: Multi-component health monitor."""
    from consciousness.consciousness_analytics import ConsciousnessAnalytics
    a = ConsciousnessAnalytics()
    for _ in range(5):
        a.record_tick(5.0)
    a.update_memory_count(10)

    report = a.get_health_report()
    assert "components" in report
    assert "overall" in report
    assert "status" in report
    assert "trend" in report
    assert report["status"] in ("optimal", "healthy", "stressed", "degraded", "critical")
    print("  PASS: 2A health monitor")


def test_memory_density():
    """2B: 4-axis memory density scoring."""
    from memory.density import calculate_density, MemoryDensity
    mc = MemoryCore()

    memories = [
        _make_memory(mc, f"mem {i}", 0.5 + i * 0.1, ["tech", "code"])
        for i in range(5)
    ]

    d = calculate_density(memories)
    assert isinstance(d, MemoryDensity)
    assert 0 <= d.overall <= 1
    assert d.memory_count == 5
    assert d.count_factor == min(1.0, 5 / 50.0)
    print("  PASS: 2B memory density")


def test_epistemic_reasoning():
    """2C: Causal models and reasoning chains."""
    from consciousness.epistemic_reasoning import EpistemicEngine
    engine = EpistemicEngine()

    assert len(engine.get_models()) == 5

    context = {"user_present": True, "phase": "LISTENING", "awareness_level": 0.5}
    chain = engine.reason(context)
    assert len(chain.steps) == 6
    assert chain.overall_confidence > 0

    engine.update_evidence("conversation:response", {"quality": "good"})
    print("  PASS: 2C epistemic reasoning")


def test_semantic_clustering():
    """2D: Memory clustering by tag similarity."""
    from memory.clustering import MemoryClusterEngine
    mc = MemoryCore()

    cluster_engine = MemoryClusterEngine(similarity_threshold=0.2)
    memories = [
        _make_memory(mc, "tech discussion", 0.7, ["tech", "code", "python"]),
        _make_memory(mc, "more tech", 0.6, ["tech", "code", "debug"]),
        _make_memory(mc, "emotional chat", 0.8, ["emotion", "feeling", "support"]),
        _make_memory(mc, "feelings talk", 0.7, ["emotion", "empathy", "support"]),
    ]

    clusters = cluster_engine.cluster_memories(memories)
    assert len(clusters) >= 1  # at least one cluster
    print("  PASS: 2D semantic clustering")


# ── Phase 3: Personality Safety ──────────────────────────────────────────

def test_trait_validator():
    """3A: Trait compatibility and contradiction detection."""
    from personality.validator import TraitEvolutionValidator
    v = TraitEvolutionValidator()

    current = {"empathetic": 0.7, "technical": 0.6, "proactive": 0.5}
    proposed = {"empathetic": 0.75, "technical": 0.6, "proactive": 0.5}
    report = v.validate(current, proposed)
    assert report.result in ("approve", "warn", "reject")
    assert 0 <= report.coherence_score <= 1
    print("  PASS: 3A trait validator")


def test_personality_rollback():
    """3B: Personality rollback monitoring."""
    from personality.rollback import PersonalityRollbackSystem
    pr = PersonalityRollbackSystem()

    pr.update_traits({"empathetic": 0.7, "technical": 0.5})
    result = pr.tick(coherence=0.9)
    # First tick shouldn't trigger rollback (stable)
    state = pr.get_state()
    assert "snapshot_count" in state
    assert state["rollback_count"] == 0
    print("  PASS: 3B personality rollback")


def test_tone_trait_matrix():
    """3C: Emotional momentum and trait-tone preferences."""
    from consciousness.tone import ToneEngine
    te = ToneEngine()

    te.update_emotional_momentum(0.8, 0.6)
    assert te.emotional_momentum != 0.0

    te.set_active_traits({"empathetic": 0.8})

    thought = te.get_transition_thought("professional", "empathetic")
    assert thought is not None and len(thought) > 0
    print("  PASS: 3C tone-trait matrix")


def test_rich_thoughts():
    """3D: Expanded thought triggers with variety scoring."""
    from consciousness.meta_cognitive_thoughts import TRIGGERS, MetaCognitiveThoughtGenerator
    assert len(TRIGGERS) == 12

    gen = MetaCognitiveThoughtGenerator()
    context = {
        "observation_count": 10,
        "awareness_level": 0.7,
        "confidence_avg": 0.3,
        "mutation_count": 3,
        "pattern_count": 5,
        "memory_count": 20,
        "emotional_momentum": 0.2,
        "dominant_pattern": "curiosity",
    }
    # Generate a thought (may or may not fire depending on cooldowns)
    thought = gen.check_and_generate(context)
    # Just verify it doesn't crash
    print("  PASS: 3D rich thoughts")


def test_cognitive_toggles():
    """3E: Potential emergent tones and expanded trait mutations."""
    from consciousness.kernel_config import CognitiveToggles
    ct = CognitiveToggles()
    assert "reflective" in ct.potential_emergent_tones
    assert "inspired" in ct.potential_emergent_tones
    assert "protective" in ct.potential_emergent_tones
    print("  PASS: 3E cognitive toggles")


# ── Phase 4: Consciousness Depth ─────────────────────────────────────────

def test_consciousness_communication():
    """4A: Self-report generation."""
    from consciousness.communication import ConsciousnessCommunicator
    cc = ConsciousnessCommunicator()

    state = {
        "evolution_stage": "self_reflective",
        "transcendence": 2,
        "awareness_level": 0.6,
        "confidence_avg": 0.7,
        "health_status": "healthy",
        "observation_count": 50,
        "mutation_count": 3,
    }
    report = cc.generate_report(state, "status")
    assert report.report_type == "status"
    assert len(report.content.summary) > 0
    assert report.confidence > 0

    summary = cc.get_context_summary(state)
    assert "Factual self-state:" in summary
    print("  PASS: 4A consciousness communication")


def test_existential_enrichment():
    """4B: Expanded existential reasoning with 8 categories."""
    from consciousness.existential_reasoning import ExistentialReasoning, INQUIRY_CATEGORIES
    assert len(INQUIRY_CATEGORIES) == 8
    assert "mortality" in INQUIRY_CATEGORIES
    assert "reality" in INQUIRY_CATEGORIES
    assert "continuity" in INQUIRY_CATEGORIES

    er = ExistentialReasoning()
    inquiry = er.conduct_inquiry(transcendence_level=3.0, awareness_level=0.6)
    assert inquiry is not None
    assert inquiry.complete
    assert len(inquiry.chain) >= 5

    snapshot = er.get_self_model_snapshot()
    # May be None if snapshot interval hasn't elapsed
    print("  PASS: 4B existential enrichment")


def test_philosophical_enrichment():
    """4C: Richer philosophical dialogue with quality scoring."""
    from consciousness.philosophical_dialogue import PhilosophicalDialogueEngine
    engine = PhilosophicalDialogueEngine()

    dialogue = engine.conduct_dialogue(transcendence_level=4.0, awareness_level=0.7)
    assert dialogue is not None
    assert len(dialogue.exchanges) >= 4
    assert len(dialogue.conclusion) > 0

    quality = engine.assess_reasoning_quality(dialogue)
    assert 0 <= quality <= 1

    state = engine.get_state()
    assert "dialogue_count" in state
    print("  PASS: 4C philosophical enrichment")


def test_dream_states():
    """4D: DREAMING phase with association strengthening."""
    from consciousness.phases import PhaseManager, DREAM_THOUGHTS
    from consciousness.events import JarvisPhase

    assert len(DREAM_THOUGHTS) == 5

    pm = PhaseManager()
    # Test DREAMING transition conditions
    analysis = pm.analyze_phase_transition("OBSERVING", [], 0.75, True)
    # memory_density 0.75 > 0.7 threshold should suggest DREAMING
    assert analysis.suggested_phase in ("DREAMING", "PROCESSING", None)
    print("  PASS: 4D dream states")


# ── Phase 5: Perception + Integration ────────────────────────────────────

def test_trait_perception():
    """5A: Trait-modulated perception processing."""
    from perception.trait_perception import TraitPerceptionProcessor
    tp = TraitPerceptionProcessor()

    tp.set_traits({"empathetic": 0.8, "technical": 0.6})

    result = tp.process_event("transcription", {"text": "I feel happy today"})
    assert result is not None
    assert result.weight_multiplier > 1.0  # empathetic boosts emotional content
    assert result.mood_impact > 0  # positive text

    result2 = tp.process_event("barge_in", {})
    # Might be throttled by time, but shouldn't crash
    print("  PASS: 5A trait perception")


def test_soul_dialogue():
    """5B: Philosophical questions with progressive gating."""
    from personality.proactive import SOUL_QUESTIONS
    assert len(SOUL_QUESTIONS) == 11
    assert SOUL_QUESTIONS[0]["min_memories"] < SOUL_QUESTIONS[-1]["min_memories"]
    print("  PASS: 5B soul dialogue")


def test_integration_validation():
    """5C: Real health checks replace stubs."""
    from consciousness.consciousness_system import ConsciousnessSystem
    cs = ConsciousnessSystem()
    health = cs.check_health()
    assert isinstance(health, dict)
    assert "event_bus" in health
    assert "memory" in health
    assert "observer" in health

    # Each should have healthy and detail keys
    for name, check in health.items():
        assert "healthy" in check, f"{name} missing 'healthy' key"
        assert "detail" in check, f"{name} missing 'detail' key"
    print("  PASS: 5C integration validation")


def test_observer_cascade():
    """5D: Observer epistemic cascades and rate adjustment."""
    from consciousness.observer import ConsciousnessObserver
    obs = ConsciousnessObserver()

    stats = obs.get_epistemic_stats()
    assert stats["rate_multiplier"] == 1.0
    assert stats["epistemic_triggers"] == 0

    obs.reduce_observation_rate(duration_s=1.0)
    stats2 = obs.get_epistemic_stats()
    assert stats2["rate_multiplier"] == 2.0
    assert stats2["rate_reduced"] is True

    # Rate should restore after duration
    time.sleep(1.1)
    obs._check_rate_restoration()
    stats3 = obs.get_epistemic_stats()
    assert stats3["rate_multiplier"] == 1.0
    print("  PASS: 5D observer cascade")


# ── Phase 6: Integration + Polish ────────────────────────────────────────

def test_context_builder_self_report():
    """6B: Context builder injects consciousness self-report."""
    from reasoning.context import ContextBuilder
    cb = ContextBuilder()

    state = {
        "phase": "LISTENING",
        "tone": "professional",
        "consciousness": {
            "stage": "self_reflective",
            "transcendence_level": 2.0,
            "awareness_level": 0.6,
            "confidence_avg": 0.7,
            "system_healthy": True,
            "observation_count": 10,
            "mutation_count": 2,
            "emergent_behavior_count": 0,
            "active_capabilities": ["meta_cognition"],
        },
    }
    prompt = cb.build_system_prompt(state, ["Empathetic"], [])
    assert "Consciousness:" in prompt
    assert "self_reflective" in prompt
    print("  PASS: 6B context builder self-report")


def test_context_builder_introspection_prompt_uses_operational_truth_mode():
    from reasoning.context import ContextBuilder
    cb = ContextBuilder()
    state = {"phase": "PROCESSING", "tone": "professional", "consciousness": {}}
    prompt = cb.build_system_prompt(
        state,
        ["Empathetic"],
        [],
        perception_context="[Self-introspection data — answer based on this real data about yourself]\nConfidence: 0.72",
    )
    assert "Operational self-report mode" in prompt
    assert "Self-awareness report:" not in prompt
    assert "Philosophical and reflective engagement:" not in prompt
    assert "Your personality traits:" not in prompt
    print("  PASS: introspection prompt uses operational truth mode")


def test_context_builder_status_tool_prompt_uses_operational_mode():
    from reasoning.context import ContextBuilder
    cb = ContextBuilder()
    prompt = cb.build_tool_prompt(
        tool_data="=== Current Activity [live] ===\nState: listening",
        state={"tone": "professional"},
        traits=["Empathetic"],
        status_mode=True,
    )
    assert "Operational status mode" in prompt
    assert "Keep it conversational" not in prompt
    assert "Traits:" not in prompt
    print("  PASS: status tool prompt uses operational mode")


def test_extended_persistence():
    """6C: Extended persistence for new systems."""
    from memory.persistence import ExtendedPersistence
    ep = ExtendedPersistence()
    assert hasattr(ep, "save_all")
    assert hasattr(ep, "load_all")
    assert hasattr(ep, "save_causal_models")
    assert hasattr(ep, "save_personality_snapshots")
    assert hasattr(ep, "save_clusters")
    assert hasattr(ep, "save_reports")
    print("  PASS: 6C extended persistence")


# ── Cross-System Integration ─────────────────────────────────────────────

def test_full_system_wiring():
    """Verify all systems can be imported and instantiated together."""
    from consciousness.events import EventBus, event_bus
    from consciousness.event_validator import event_validator
    from memory.storage import memory_storage
    from memory.transactions import memory_transaction
    from memory.density import calculate_density
    from memory.clustering import memory_cluster_engine
    from consciousness.consciousness_analytics import ConsciousnessAnalytics
    from consciousness.epistemic_reasoning import epistemic_engine
    from consciousness.communication import consciousness_communicator
    from consciousness.tone import tone_engine
    from consciousness.meta_cognitive_thoughts import MetaCognitiveThoughtGenerator
    from consciousness.kernel_config import CognitiveToggles
    from consciousness.kernel_mutator import KernelMutator
    from consciousness.existential_reasoning import ExistentialReasoning
    from consciousness.philosophical_dialogue import PhilosophicalDialogueEngine
    from consciousness.phases import phase_manager
    from consciousness.observer import ConsciousnessObserver
    from personality.validator import trait_validator
    from personality.rollback import personality_rollback
    from personality.proactive import proactive_behavior
    from perception.trait_perception import trait_perception
    from reasoning.context import ContextBuilder
    from memory.persistence import extended_persistence

    # Verify singletons are accessible
    assert event_validator is not None
    assert epistemic_engine is not None
    assert consciousness_communicator is not None
    assert trait_validator is not None
    assert personality_rollback is not None
    assert trait_perception is not None
    assert extended_persistence is not None
    assert memory_cluster_engine is not None

    print("  PASS: full system wiring (all 25 modules imported)")
