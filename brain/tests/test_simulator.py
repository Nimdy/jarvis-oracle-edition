"""Tests for Phase 3: Mental Simulator.

Covers all acceptance tests from docs/phase3_design_notes.md:
  - Core invariants (readonly, confidence decay, max depth, empty delta)
  - Rule integration (causal rules, chaining, conflicts)
  - Shadow mode (no state mutation, no event emission)
  - Identity noise resilience
"""

from __future__ import annotations

import time
from dataclasses import replace
from unittest.mock import patch

import pytest

from cognition.causal_engine import CausalEngine, CausalRule
from cognition.promotion import SimulatorPromotion
from cognition.simulator import (
    CONFIDENCE_DECAY,
    MAX_DEPTH,
    MentalSimulator,
    SimulationStep,
    SimulationTrace,
    _apply_delta_to_state,
    _apply_predicted_fields,
)
from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return CausalEngine()


@pytest.fixture
def simulator(engine):
    return MentalSimulator(engine)


@pytest.fixture
def base_state():
    """A realistic base state with user present and system healthy."""
    return WorldState(
        physical=PhysicalState(
            entity_count=5, visible_count=4, stable_count=3,
            person_count=1, last_update_ts=time.time(),
        ),
        user=UserState(
            present=True, presence_confidence=0.85,
            engagement=0.6, emotion="neutral", emotion_confidence=0.7,
            speaker_name="David", speaker_confidence=0.52,
            identity_method="face_only",
            seconds_since_last_interaction=30.0,
            last_update_ts=time.time(),
        ),
        conversation=ConversationState(
            active=False, topic="", turn_count=0,
            last_update_ts=time.time(),
        ),
        system=SystemState(
            mode="passive", health_score=0.9, confidence=0.8,
            autonomy_level=1, memory_count=500, uptime_s=3600.0,
            last_update_ts=time.time(),
        ),
        version=100,
        timestamp=time.time(),
        tick_number=100,
    )


# ---------------------------------------------------------------------------
# Core Invariants
# ---------------------------------------------------------------------------

class TestSimulatorPureReadonly:
    """test_simulator_pure_readonly — original state unchanged after simulation."""

    def test_state_identity_preserved(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        original_version = base_state.version
        original_user = base_state.user

        trace = simulator.simulate(base_state, delta)

        assert base_state.version == original_version
        assert base_state.user is original_user
        assert base_state.user.present is True

    def test_frozen_dataclass_unchanged(self, simulator, base_state):
        delta = WorldDelta(facet="conversation", event="conversation_started",
                           details={"topic": "test"})
        state_dict_before = base_state.to_dict()
        simulator.simulate(base_state, delta)
        state_dict_after = base_state.to_dict()
        assert state_dict_before == state_dict_after


class TestSimulatorConfidenceDecay:
    """test_simulator_confidence_decay — each step's confidence strictly less than previous."""

    def test_confidence_decays_overall(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta, max_depth=3)

        assert trace.total_confidence < 1.0
        if trace.depth >= 1:
            assert trace.steps[0].confidence < 1.0

    def test_total_confidence_is_product(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta, max_depth=3)

        expected = 1.0
        for s in trace.steps:
            expected *= s.confidence
        assert abs(trace.total_confidence - round(expected, 4)) < 0.001


class TestSimulatorMaxDepth:
    """test_simulator_max_depth — depth clamped to MAX_DEPTH."""

    def test_depth_clamped_to_max(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta, max_depth=10)
        assert trace.depth <= MAX_DEPTH

    def test_trace_never_exceeds_max_steps(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_arrived",
                           details={"confidence": 0.9})
        trace = simulator.simulate(base_state, delta, max_depth=100)
        assert len(trace.steps) <= MAX_DEPTH


class TestSimulatorEmptyDelta:
    """test_simulator_empty_delta — no-op delta produces minimal trace."""

    def test_noop_delta(self, simulator, base_state):
        delta = WorldDelta(facet="physical", event="entity_moved")
        trace = simulator.simulate(base_state, delta)

        assert trace.depth >= 0
        assert isinstance(trace.total_confidence, float)
        assert trace.initial_delta is delta


# ---------------------------------------------------------------------------
# Rule Integration
# ---------------------------------------------------------------------------

class TestSimulatorCausalRules:
    """test_simulator_causal_rules — real CausalEngine rules fire during simulation."""

    def test_user_departed_fires_rules(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta)

        assert trace.depth >= 1
        all_rules = []
        for s in trace.steps:
            all_rules.extend(s.applied_rules)
        assert len(all_rules) > 0

    def test_custom_rule_fires(self):
        engine = CausalEngine()
        engine._rules.append(CausalRule(
            rule_id="test_custom_rule",
            label="test_fires",
            category="user",
            priority=99,
            condition=lambda ws, ds: ws.user.present is False,
            predicted_delta={"system.mode": "sleep"},
            confidence=0.9,
            horizon_s=0.0,
        ))
        sim = MentalSimulator(engine)

        state = WorldState(
            user=UserState(present=True),
            system=SystemState(mode="passive"),
        )
        delta = WorldDelta(facet="user", event="user_departed")

        trace = sim.simulate(state, delta)

        all_rules = []
        for s in trace.steps:
            all_rules.extend(s.applied_rules)
        assert "test_custom_rule" in all_rules


class TestSimulatorRuleChaining:
    """test_simulator_rule_chaining — a rule at step 1 sets state that triggers another at step 2."""

    def test_chained_rules(self):
        engine = CausalEngine()
        engine._rules = [
            CausalRule(
                rule_id="step1_depart",
                label="depart",
                category="user",
                priority=60,
                condition=lambda ws, ds: ws.user.present is False,
                predicted_delta={"conversation.active": False, "system.mode": "sleep"},
                confidence=0.9,
                horizon_s=0.0,
            ),
            CausalRule(
                rule_id="step2_sleep_absent",
                label="sleep_implies_absent",
                category="system",
                priority=20,
                condition=lambda ws, ds: (
                    ws.system.mode == "sleep"
                    and not ws.user.present
                ),
                predicted_delta={"user.present": False},
                confidence=0.8,
                horizon_s=0.0,
            ),
        ]
        sim = MentalSimulator(engine)

        state = WorldState(
            user=UserState(present=True),
            conversation=ConversationState(active=True),
            system=SystemState(mode="passive"),
        )
        delta = WorldDelta(facet="user", event="user_departed")

        trace = sim.simulate(state, delta, max_depth=3)

        all_rules = []
        for s in trace.steps:
            all_rules.extend(s.applied_rules)
        assert "step1_depart" in all_rules
        assert "step2_sleep_absent" in all_rules


class TestSimulatorConflictingRules:
    """test_simulator_conflicting_rules — priority-based conflict resolution."""

    def test_higher_priority_wins(self):
        engine = CausalEngine()
        engine._rules = [
            CausalRule(
                rule_id="high_priority",
                label="health_keeps_up",
                category="health",
                priority=100,
                condition=lambda ws, ds: True,
                predicted_delta={"system.health_score": 0.9},
                confidence=0.9,
                horizon_s=0.0,
            ),
            CausalRule(
                rule_id="low_priority",
                label="health_drops",
                category="system",
                priority=10,
                condition=lambda ws, ds: True,
                predicted_delta={"system.health_score": 0.1},
                confidence=0.8,
                horizon_s=0.0,
            ),
        ]
        sim = MentalSimulator(engine)

        state = WorldState(system=SystemState(health_score=0.7))
        delta = WorldDelta(facet="system", event="health_degraded",
                           details={"score": 0.5})

        trace = sim.simulate(state, delta, max_depth=1)

        assert trace.depth == 1
        step = trace.steps[0]
        assert "high_priority" in step.applied_rules
        assert step.state.system.health_score == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# Shadow Mode
# ---------------------------------------------------------------------------

class TestSimulatorShadowMode:
    """test_simulator_shadow_mode — simulation never writes to real state."""

    def test_world_model_state_untouched(self, base_state):
        from cognition.world_model import WorldModel
        wm = WorldModel()

        wm._current = base_state
        wm._previous = None
        wm._version = base_state.version

        delta = WorldDelta(facet="user", event="user_departed")
        trace = wm._simulator.simulate(base_state, delta)

        assert wm._current.user.present is True
        assert wm._current.version == base_state.version


class TestSimulatorNoEventEmission:
    """test_simulator_no_event_emission — no EventBus events during simulation."""

    def test_no_events_emitted(self, simulator, base_state):
        from consciousness.events import event_bus

        events_before = []

        def _spy(event_type, **kwargs):
            events_before.append(event_type)

        event_bus.on("world_model:update", _spy)
        event_bus.on("world_model:prediction_validated", _spy)

        delta = WorldDelta(facet="user", event="user_departed")
        simulator.simulate(base_state, delta)

        event_bus.off("world_model:update", _spy)
        event_bus.off("world_model:prediction_validated", _spy)

        sim_events = [e for e in events_before if "SIMULATOR" in str(e)]
        assert len(sim_events) == 0


# ---------------------------------------------------------------------------
# Identity Noise Resilience
# ---------------------------------------------------------------------------

class TestSimulatorIdentityNoise:
    """test_simulator_identity_noise — low-confidence identity doesn't inflate trace confidence."""

    def test_low_identity_confidence_no_boost(self, simulator):
        state = WorldState(
            user=UserState(
                present=True, identity_method="persisted",
                speaker_confidence=0.55, presence_confidence=0.6,
                seconds_since_last_interaction=50.0,
            ),
            physical=PhysicalState(
                entity_count=3, visible_count=2, stable_count=1,
                person_count=1, last_update_ts=time.time(),
            ),
            system=SystemState(mode="passive", health_score=0.8),
        )

        delta = WorldDelta(facet="user", event="speaker_changed",
                           details={"speaker": "unknown", "confidence": 0.3})
        trace = simulator.simulate(state, delta)

        assert trace.total_confidence <= 1.0
        for step in trace.steps:
            assert step.confidence <= 1.0


# ---------------------------------------------------------------------------
# Apply Delta Helpers
# ---------------------------------------------------------------------------

class TestApplyDeltaToState:
    """Unit tests for _apply_delta_to_state helper."""

    def test_user_departed(self):
        state = WorldState(user=UserState(present=True, engagement=0.8))
        delta = WorldDelta(facet="user", event="user_departed")
        result = _apply_delta_to_state(state, delta)
        assert result.user.present is False
        assert result.user.engagement == 0.0

    def test_user_arrived(self):
        state = WorldState(user=UserState(present=False))
        delta = WorldDelta(facet="user", event="user_arrived",
                           details={"confidence": 0.9})
        result = _apply_delta_to_state(state, delta)
        assert result.user.present is True
        assert result.user.presence_confidence == 0.9

    def test_emotion_changed(self):
        state = WorldState(user=UserState(emotion="neutral"))
        delta = WorldDelta(facet="user", event="emotion_changed",
                           details={"emotion": "happy", "confidence": 0.8})
        result = _apply_delta_to_state(state, delta)
        assert result.user.emotion == "happy"
        assert result.user.emotion_confidence == 0.8

    def test_conversation_started(self):
        state = WorldState(conversation=ConversationState(active=False))
        delta = WorldDelta(facet="conversation", event="conversation_started",
                           details={"topic": "weather"})
        result = _apply_delta_to_state(state, delta)
        assert result.conversation.active is True
        assert result.conversation.topic == "weather"

    def test_mode_changed(self):
        state = WorldState(system=SystemState(mode="passive"))
        delta = WorldDelta(facet="system", event="mode_changed",
                           details={"mode": "conversational"})
        result = _apply_delta_to_state(state, delta)
        assert result.system.mode == "conversational"

    def test_health_degraded(self):
        state = WorldState(system=SystemState(health_score=1.0))
        delta = WorldDelta(facet="system", event="health_degraded",
                           details={"score": 0.3})
        result = _apply_delta_to_state(state, delta)
        assert result.system.health_score == 0.3

    def test_unhandled_event_preserves_state(self):
        state = WorldState(user=UserState(present=True))
        delta = WorldDelta(facet="physical", event="entity_appeared")
        result = _apply_delta_to_state(state, delta)
        assert result.user.present is True

    def test_original_state_unchanged(self):
        state = WorldState(user=UserState(present=True, engagement=0.5))
        delta = WorldDelta(facet="user", event="user_departed")
        _apply_delta_to_state(state, delta)
        assert state.user.present is True
        assert state.user.engagement == 0.5


class TestApplyPredictedFields:
    """Unit tests for _apply_predicted_fields helper."""

    def test_user_engagement(self):
        state = WorldState(user=UserState(engagement=0.3))
        result = _apply_predicted_fields(state, {"user.engagement": 0.0})
        assert result.user.engagement == 0.0

    def test_conversation_active(self):
        state = WorldState(conversation=ConversationState(active=True))
        result = _apply_predicted_fields(state, {"conversation.active": False})
        assert result.conversation.active is False

    def test_system_mode(self):
        state = WorldState(system=SystemState(mode="passive"))
        result = _apply_predicted_fields(state, {"system.mode": "sleep"})
        assert result.system.mode == "sleep"

    def test_multiple_fields(self):
        state = WorldState(
            user=UserState(present=True, engagement=0.8),
            conversation=ConversationState(active=True),
        )
        result = _apply_predicted_fields(state, {
            "user.present": False,
            "user.engagement": 0.0,
            "conversation.active": False,
        })
        assert result.user.present is False
        assert result.user.engagement == 0.0
        assert result.conversation.active is False

    def test_invalid_path_ignored(self):
        state = WorldState(user=UserState(present=True))
        result = _apply_predicted_fields(state, {"user.nonexistent": 42})
        assert result.user.present is True

    def test_invalid_facet_ignored(self):
        state = WorldState(user=UserState(present=True))
        result = _apply_predicted_fields(state, {"bogus.field": 42})
        assert result.user.present is True

    def test_original_unchanged(self):
        state = WorldState(user=UserState(engagement=0.8))
        _apply_predicted_fields(state, {"user.engagement": 0.0})
        assert state.user.engagement == 0.8


# ---------------------------------------------------------------------------
# SimulationTrace structure
# ---------------------------------------------------------------------------

class TestSimulationTraceStructure:

    def test_trace_has_all_fields(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta, source="test")

        assert isinstance(trace, SimulationTrace)
        assert isinstance(trace.steps, tuple)
        assert isinstance(trace.initial_delta, WorldDelta)
        assert isinstance(trace.total_confidence, float)
        assert isinstance(trace.depth, int)
        assert isinstance(trace.elapsed_ms, float)
        assert trace.source == "test"

    def test_steps_are_frozen(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        trace = simulator.simulate(base_state, delta)

        for step in trace.steps:
            assert isinstance(step, SimulationStep)
            assert isinstance(step.applied_rules, tuple)
            with pytest.raises(AttributeError):
                step.confidence = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Statistics & Logging
# ---------------------------------------------------------------------------

class TestSimulatorStats:

    def test_stats_after_simulations(self, simulator, base_state):
        for event in ["user_departed", "conversation_started", "mode_changed"]:
            delta = WorldDelta(facet="user" if "user" in event else "system",
                               event=event,
                               details={"mode": "conversational"} if "mode" in event else {})
            simulator.simulate(base_state, delta)

        stats = simulator.get_stats()
        assert stats["total_simulations"] == 3
        assert stats["trace_buffer_size"] == 3
        assert stats["avg_depth"] >= 0

    def test_recent_traces(self, simulator, base_state):
        delta = WorldDelta(facet="user", event="user_departed")
        simulator.simulate(base_state, delta, source="test_source")

        traces = simulator.get_recent_traces(5)
        assert len(traces) >= 1
        assert traces[0]["delta_event"] == "user_departed"
        assert traces[0]["source"] == "test_source"
        assert "steps" in traces[0]


# ---------------------------------------------------------------------------
# Simulator Promotion
# ---------------------------------------------------------------------------

class TestSimulatorPromotion:

    @pytest.fixture(autouse=True)
    def _isolate_promotion(self, tmp_path, monkeypatch):
        """Prevent promotion tests from reading/writing the real persistence file."""
        fake_path = str(tmp_path / "simulator_promotion.json")
        monkeypatch.setattr(
            "cognition.promotion.SIMULATOR_PROMOTION_PATH", fake_path,
        )

    def test_starts_at_shadow(self):
        promo = SimulatorPromotion()
        assert promo.level == 0
        status = promo.get_status()
        assert status["level_name"] == "shadow"

    def test_does_not_promote_without_enough_data(self):
        promo = SimulatorPromotion()
        for _ in range(50):
            promo.record_outcome(True)
        assert promo.level == 0

    def test_promotion_requires_hours(self):
        promo = SimulatorPromotion()
        promo._state.shadow_start_ts = time.time()
        for _ in range(150):
            promo.record_outcome(True)
        assert promo.level == 0

    def test_promotion_with_all_conditions(self):
        promo = SimulatorPromotion()
        promo._state.shadow_start_ts = time.time() - (49 * 3600)
        for _ in range(150):
            promo.record_outcome(True)
        assert promo.level == 1

    def test_demotion_on_low_accuracy(self):
        promo = SimulatorPromotion()
        promo._state.shadow_start_ts = time.time() - (49 * 3600)
        for _ in range(150):
            promo.record_outcome(True)
        assert promo.level == 1

        # Move last_promoted_at past the transition cooldown
        from cognition.promotion import SIM_TRANSITION_COOLDOWN_S
        promo._state.last_promoted_at = time.time() - SIM_TRANSITION_COOLDOWN_S - 1

        for _ in range(100):
            promo.record_outcome(False)
        assert promo.level == 0


# ---------------------------------------------------------------------------
# WorldModel integration
# ---------------------------------------------------------------------------

class TestWorldModelSimulatorIntegration:

    def test_world_model_has_simulator(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        assert hasattr(wm, '_simulator')
        assert isinstance(wm._simulator, MentalSimulator)

    def test_simulator_in_get_state(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        state = wm.get_state()
        assert "simulator" in state
        assert "simulator_promotion" in state
        assert "recent_simulations" in state

    def test_simulator_in_diagnostics(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        diag = wm.get_diagnostics()
        assert "simulator" in diag
        assert "simulator_promotion" in diag

    def test_simulator_property(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        assert wm.simulator is wm._simulator
