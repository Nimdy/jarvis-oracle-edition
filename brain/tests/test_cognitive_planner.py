"""Tests for Phase 8 (#16): CognitivePlanner — multi-step path search.

The CognitivePlanner chains the MentalSimulator across action SEQUENCES and
proposes the best path. It is read-only, no-LLM, and DATA-GATED on the simulator
reaching advisory (>= 100 verified live simulations). These tests pin the
contract that matters:

  - DORMANT until the gate opens (gate-blocked by design, NOT broken).
  - When advisory, it produces ranked multi-step paths.
  - It genuinely CHAINS the simulator (step N sees step N-1's projected state).
  - It never mutates the WorldState.
  - It is deterministic and bounded.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from cognition.causal_engine import CausalEngine
from cognition.simulator import MentalSimulator
from cognition.planner import (
    BEAM_WIDTH,
    CANDIDATE_CAP,
    HORIZON,
    MAX_PLAN_OPTIONS,
    MIN_VERIFIED_SIMULATIONS,
    SIMULATOR_ADVISORY_LEVEL,
    CognitivePlanner,
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
def simulator():
    return MentalSimulator(CausalEngine())


@pytest.fixture
def base_state():
    now = time.time()
    return WorldState(
        physical=PhysicalState(
            entity_count=5, visible_count=4, stable_count=3,
            person_count=1, last_update_ts=now,
        ),
        user=UserState(
            present=True, presence_confidence=0.85,
            engagement=0.6, emotion="neutral", emotion_confidence=0.7,
            speaker_name="David", speaker_confidence=0.52,
            identity_method="face_only",
            seconds_since_last_interaction=30.0, last_update_ts=now,
        ),
        conversation=ConversationState(
            active=False, topic="", turn_count=0, last_update_ts=now,
        ),
        system=SystemState(
            mode="passive", health_score=0.9, confidence=0.8,
            autonomy_level=1, memory_count=500, uptime_s=3600.0,
            last_update_ts=now,
        ),
        version=100, timestamp=now, tick_number=100,
    )


@pytest.fixture
def deltas():
    return [
        WorldDelta(facet="user", event="user_departed", confidence=0.9),
        WorldDelta(facet="conversation", event="conversation_started",
                   details={"topic": "weather"}, confidence=0.8),
        WorldDelta(facet="user", event="emotion_changed",
                   details={"to": "happy"}, confidence=0.7),
    ]


class _RecordingSim:
    """Wraps a MentalSimulator and records every (input_state, delta) call."""

    def __init__(self, inner: MentalSimulator) -> None:
        self.inner = inner
        self.calls: list[tuple[WorldState, WorldDelta]] = []

    def simulate(self, state, delta, max_depth=3, source=""):
        self.calls.append((state, delta))
        return self.inner.simulate(state, delta, max_depth=max_depth, source=source)


def _open_gate(planner, base_state, deltas, simulator, goal_title=""):
    """Run evaluate with the data gate satisfied (advisory + 100 verified sims)."""
    return planner.evaluate(
        world_state=base_state,
        deltas=deltas,
        simulator=simulator,
        simulator_promotion_level=SIMULATOR_ADVISORY_LEVEL,
        verified_simulations=MIN_VERIFIED_SIMULATIONS,
        goal_title=goal_title,
    )


# ---------------------------------------------------------------------------
# Data gate — dormant by design
# ---------------------------------------------------------------------------

class TestDataGate:
    def test_dormant_when_simulator_shadow(self, base_state, deltas, simulator):
        """Level 0 (shadow) → dormant, regardless of verified count."""
        planner = CognitivePlanner()
        state = planner.evaluate(
            world_state=base_state, deltas=deltas, simulator=simulator,
            simulator_promotion_level=0,
            verified_simulations=MIN_VERIFIED_SIMULATIONS,
        )
        assert state["enabled"] is False
        assert state["active"] is False
        assert state["selected"] is None
        assert state["paths"] == []
        assert "simulator_not_advisory" in state["reason"]

    def test_dormant_when_under_verified_floor(self, base_state, deltas, simulator):
        """Advisory level but < 100 verified sims → still dormant (legible count)."""
        planner = CognitivePlanner()
        state = planner.evaluate(
            world_state=base_state, deltas=deltas, simulator=simulator,
            simulator_promotion_level=SIMULATOR_ADVISORY_LEVEL,
            verified_simulations=MIN_VERIFIED_SIMULATIONS - 1,
        )
        assert state["enabled"] is False
        assert state["active"] is False
        assert f"{MIN_VERIFIED_SIMULATIONS - 1}/{MIN_VERIFIED_SIMULATIONS}" in state["reason"]

    def test_no_simulation_runs_while_dormant(self, base_state, deltas, simulator):
        """A dormant planner must not burn the simulator — zero calls."""
        planner = CognitivePlanner()
        rec = _RecordingSim(simulator)
        planner.evaluate(
            world_state=base_state, deltas=deltas, simulator=rec,
            simulator_promotion_level=0, verified_simulations=0,
        )
        assert rec.calls == []

    def test_gate_floor_matches_promotion_constant(self):
        """The planner's gate must mirror cognition.promotion (no fabricated gate)."""
        from cognition.promotion import SIM_MIN_SIMULATIONS
        assert MIN_VERIFIED_SIMULATIONS == SIM_MIN_SIMULATIONS


# ---------------------------------------------------------------------------
# Active behaviour — produces ranked multi-step paths
# ---------------------------------------------------------------------------

class TestActivePlanning:
    def test_produces_paths_when_advisory(self, base_state, deltas, simulator):
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        assert state["enabled"] is True
        assert state["active"] is True
        assert state["reason"] == "ok"
        assert state["selected"] is not None
        assert len(state["paths"]) >= 1
        assert len(state["paths"]) <= MAX_PLAN_OPTIONS

    def test_selected_is_highest_utility(self, base_state, deltas, simulator):
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        utilities = [p["path_utility"] for p in state["paths"]]
        assert utilities == sorted(utilities, reverse=True)
        assert state["selected"]["path_utility"] == max(utilities)

    def test_multi_step_paths_exist(self, base_state, deltas, simulator):
        """At least one proposed path is genuinely multi-step (length >= 2)."""
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        max_len = max(p["length"] for p in state["paths"])
        assert max_len >= 2

    def test_no_deltas_means_inactive(self, base_state, simulator):
        planner = CognitivePlanner()
        state = planner.evaluate(
            world_state=base_state, deltas=[], simulator=simulator,
            simulator_promotion_level=SIMULATOR_ADVISORY_LEVEL,
            verified_simulations=MIN_VERIFIED_SIMULATIONS,
        )
        assert state["enabled"] is True
        assert state["active"] is False
        assert state["reason"] == "no_deltas"
        assert state["paths"] == []

    def test_step_ordering_is_sequential(self, base_state, deltas, simulator):
        """Each path's steps carry increasing order indices starting at 0."""
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        for path in state["paths"]:
            orders = [s["order"] for s in path["steps"]]
            assert orders == list(range(len(orders)))


# ---------------------------------------------------------------------------
# Chaining — the heart of "multi-step"
# ---------------------------------------------------------------------------

class TestChaining:
    def test_second_step_sees_projected_state(self, base_state, simulator):
        """Decisive proof of chaining: step 2 is simulated from step 1's output.

        base_state has user.present=True. A path [user_departed, ...] must
        simulate the *second* action from a state where present=False — i.e.
        from the projection of user_departed, not from the original state.
        """
        deltas = [
            WorldDelta(facet="user", event="user_departed", confidence=0.95),
            WorldDelta(facet="conversation", event="conversation_started",
                       confidence=0.9),
        ]
        rec = _RecordingSim(simulator)
        planner = CognitivePlanner()
        _open_gate(planner, base_state, deltas, rec)

        # Some conversation_started simulation must have run on a departed state.
        chained = any(
            d.event == "conversation_started" and s.user.present is False
            for (s, d) in rec.calls
        )
        assert chained, "second step was not chained from the projected state"

        # And the first-level call for the same event ran on the original state.
        from_original = any(
            d.event == "conversation_started" and s.user.present is True
            for (s, d) in rec.calls
        )
        assert from_original, "expected a depth-0 call on the original state too"


# ---------------------------------------------------------------------------
# Read-only invariant
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_world_state_not_mutated(self, base_state, deltas, simulator):
        before = replace(base_state)  # frozen snapshot copy
        planner = CognitivePlanner()
        _open_gate(planner, base_state, deltas, simulator)
        assert base_state == before
        assert base_state.user.present is True
        assert base_state.conversation.active is False

    def test_no_events_emitted(self, base_state, deltas, simulator, monkeypatch):
        """The planner must not emit on the event bus."""
        emitted = []
        try:
            from consciousness import events as ev
            monkeypatch.setattr(ev.event_bus, "emit",
                                lambda *a, **k: emitted.append((a, k)))
        except Exception:
            pytest.skip("event bus not importable in this environment")
        planner = CognitivePlanner()
        _open_gate(planner, base_state, deltas, simulator)
        assert emitted == []


# ---------------------------------------------------------------------------
# Determinism & bounds
# ---------------------------------------------------------------------------

class TestDeterminismAndBounds:
    def test_deterministic(self, base_state, deltas, simulator):
        planner_a = CognitivePlanner()
        planner_b = CognitivePlanner()
        sim_b = MentalSimulator(CausalEngine())
        state_a = _open_gate(planner_a, base_state, deltas, simulator)
        state_b = _open_gate(planner_b, base_state, deltas, sim_b)
        assert state_a["paths"] == state_b["paths"]
        assert state_a["selected"] == state_b["selected"]

    def test_candidate_cap_respected(self, base_state, simulator):
        """More deltas than CANDIDATE_CAP → only the cap's worth of events appear."""
        many = [
            WorldDelta(facet="user", event="user_departed", confidence=0.99),
            WorldDelta(facet="conversation", event="conversation_started", confidence=0.95),
            WorldDelta(facet="user", event="emotion_changed",
                       details={"to": "happy"}, confidence=0.90),
            WorldDelta(facet="conversation", event="topic_changed",
                       details={"to": "x"}, confidence=0.85),
            WorldDelta(facet="system", event="mode_changed",
                       details={"to": "active"}, confidence=0.10),
            WorldDelta(facet="system", event="health_degraded", confidence=0.05),
        ]
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, many, simulator)
        events_used = {s["event"] for p in state["paths"] for s in p["steps"]}
        assert len(events_used) <= CANDIDATE_CAP
        # The two lowest-confidence deltas should be dropped by the cap.
        assert "health_degraded" not in events_used

    def test_path_length_bounded_by_horizon(self, base_state, deltas, simulator):
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        for path in state["paths"]:
            assert path["length"] <= HORIZON

    def test_options_bounded(self, base_state, deltas, simulator):
        planner = CognitivePlanner()
        state = _open_gate(planner, base_state, deltas, simulator)
        assert len(state["paths"]) <= MAX_PLAN_OPTIONS
