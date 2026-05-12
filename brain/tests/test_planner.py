"""Tests for Phase 7 planner scaffold (shadow-only)."""

from __future__ import annotations

import time

from cognition.causal_engine import CausalEngine
from cognition.planner import MAX_PLAN_OPTIONS, WorldPlanner
from cognition.simulator import MentalSimulator
from cognition.world_state import ConversationState, SystemState, UserState, WorldDelta, WorldState


def _make_state(*, user_uncertainty: float = 0.2) -> WorldState:
    now = time.time()
    return WorldState(
        user=UserState(
            present=True,
            engagement=0.65,
            emotion="neutral",
            last_update_ts=now,
        ),
        conversation=ConversationState(active=False, last_update_ts=now),
        system=SystemState(mode="conversational", health_score=0.9, last_update_ts=now),
        version=10,
        timestamp=now,
        tick_number=10,
        uncertainty={
            "physical": 0.2,
            "user": user_uncertainty,
            "conversation": 0.2,
            "system": 0.2,
        },
    )


def _simulator() -> MentalSimulator:
    return MentalSimulator(CausalEngine())


class TestWorldPlannerGating:
    def test_disabled_before_simulator_advisory(self):
        planner = WorldPlanner()
        state = _make_state()
        deltas = [WorldDelta(facet="user", event="user_departed", confidence=0.9)]

        out = planner.evaluate(
            world_state=state,
            deltas=deltas,
            simulator=_simulator(),
            simulator_promotion_level=0,
        )

        assert out["enabled"] is False
        assert out["active"] is False
        assert out["reason"] == "simulator_not_advisory"
        assert out["options"] == []
        assert out["selected"] is None

    def test_enabled_but_inactive_without_deltas(self):
        planner = WorldPlanner()
        state = _make_state()

        out = planner.evaluate(
            world_state=state,
            deltas=[],
            simulator=_simulator(),
            simulator_promotion_level=1,
        )

        assert out["enabled"] is True
        assert out["active"] is False
        assert out["reason"] == "no_deltas"
        assert out["options"] == []


class TestWorldPlannerOptions:
    def test_generates_ranked_options(self):
        planner = WorldPlanner()
        state = _make_state()
        deltas = [
            WorldDelta(facet="user", event="user_departed", confidence=0.9),
            WorldDelta(facet="system", event="health_degraded", details={"to": 0.5}, confidence=0.8),
            WorldDelta(facet="conversation", event="conversation_started", confidence=0.7),
        ]

        out = planner.evaluate(
            world_state=state,
            deltas=deltas,
            simulator=_simulator(),
            simulator_promotion_level=1,
        )

        assert out["enabled"] is True
        assert out["active"] is True
        assert out["reason"] == "ok"
        assert 1 <= len(out["options"]) <= MAX_PLAN_OPTIONS
        assert out["selected"] == out["options"][0]

        utilities = [o["utility"] for o in out["options"]]
        assert utilities == sorted(utilities, reverse=True)
        assert all("recommendation" in o and o["recommendation"] for o in out["options"])

    def test_goal_alignment_increases_utility_for_matching_delta(self):
        planner = WorldPlanner()
        state = _make_state()
        delta = WorldDelta(
            facet="user",
            event="engagement_crossed_threshold",
            details={"value": 0.8, "direction": "up"},
            confidence=0.8,
        )

        aligned = planner.evaluate(
            world_state=state,
            deltas=[delta],
            simulator=_simulator(),
            simulator_promotion_level=1,
            goal_title="Improve engagement quality",
        )
        baseline = planner.evaluate(
            world_state=state,
            deltas=[delta],
            simulator=_simulator(),
            simulator_promotion_level=1,
            goal_title="Improve memory retrieval",
        )

        assert aligned["selected"] is not None
        assert baseline["selected"] is not None
        assert aligned["selected"]["goal_alignment"] > baseline["selected"]["goal_alignment"]
        assert aligned["selected"]["utility"] > baseline["selected"]["utility"]

    def test_higher_uncertainty_reduces_utility(self):
        planner = WorldPlanner()
        delta = WorldDelta(facet="user", event="user_departed", confidence=0.8)

        low_risk = planner.evaluate(
            world_state=_make_state(user_uncertainty=0.1),
            deltas=[delta],
            simulator=_simulator(),
            simulator_promotion_level=1,
        )
        high_risk = planner.evaluate(
            world_state=_make_state(user_uncertainty=0.9),
            deltas=[delta],
            simulator=_simulator(),
            simulator_promotion_level=1,
        )

        assert low_risk["selected"] is not None
        assert high_risk["selected"] is not None
        assert low_risk["selected"]["risk"] < high_risk["selected"]["risk"]
        assert low_risk["selected"]["utility"] > high_risk["selected"]["utility"]
