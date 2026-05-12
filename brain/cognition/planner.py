"""Shadow planner scaffold for Phase 7.

Builds deterministic planning candidates from recent world deltas and
simulator projections. It is read-only and advisory-only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cognition.simulator import MentalSimulator
from cognition.world_state import WorldDelta, WorldState

MAX_PLAN_OPTIONS = 3
SIMULATOR_ADVISORY_LEVEL = 1
GOAL_ALIGNMENT_BOOST = 1.15
RISK_PENALTY_WEIGHT = 0.35


@dataclass(frozen=True)
class PlanOption:
    """One candidate action inferred from a world delta."""

    source_event: str
    source_facet: str
    projected_depth: int
    projected_confidence: float
    risk: float
    utility: float
    goal_alignment: float
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_event": self.source_event,
            "source_facet": self.source_facet,
            "projected_depth": self.projected_depth,
            "projected_confidence": round(self.projected_confidence, 3),
            "risk": round(self.risk, 3),
            "utility": round(self.utility, 3),
            "goal_alignment": round(self.goal_alignment, 3),
            "recommendation": self.recommendation,
        }


class WorldPlanner:
    """Deterministic shadow planner fed by WorldModel + MentalSimulator."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "enabled": False,
            "active": False,
            "reason": "init",
            "generated_at": 0.0,
            "selected": None,
            "options": [],
        }

    def evaluate(
        self,
        *,
        world_state: WorldState,
        deltas: list[WorldDelta],
        simulator: MentalSimulator,
        simulator_promotion_level: int,
        goal_title: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        self._state["generated_at"] = round(now, 3)

        if simulator_promotion_level < SIMULATOR_ADVISORY_LEVEL:
            self._state.update({
                "enabled": False,
                "active": False,
                "reason": "simulator_not_advisory",
                "selected": None,
                "options": [],
            })
            return self.get_state()

        self._state["enabled"] = True

        if not deltas:
            self._state.update({
                "active": False,
                "reason": "no_deltas",
                "selected": None,
                "options": [],
            })
            return self.get_state()

        options: list[PlanOption] = []
        ranked_deltas = sorted(deltas, key=lambda d: d.confidence, reverse=True)[:MAX_PLAN_OPTIONS]
        for delta in ranked_deltas:
            trace = simulator.simulate(
                world_state,
                delta,
                max_depth=2,
                source="planner_shadow",
            )
            if trace.depth <= 0:
                continue

            risk = self._compute_risk(world_state, delta)
            alignment = self._goal_alignment_boost(goal_title, delta)
            opportunity = trace.total_confidence * delta.confidence * alignment
            utility = opportunity - (risk * RISK_PENALTY_WEIGHT)

            option = PlanOption(
                source_event=delta.event,
                source_facet=delta.facet,
                projected_depth=trace.depth,
                projected_confidence=trace.total_confidence,
                risk=risk,
                utility=utility,
                goal_alignment=alignment,
                recommendation=self._render_recommendation(delta, trace.depth, utility),
            )
            options.append(option)

        options.sort(key=lambda o: o.utility, reverse=True)
        option_dicts = [o.to_dict() for o in options[:MAX_PLAN_OPTIONS]]

        self._state.update({
            "active": bool(option_dicts),
            "reason": "ok" if option_dicts else "no_viable_options",
            "selected": option_dicts[0] if option_dicts else None,
            "options": option_dicts,
        })
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    @staticmethod
    def _compute_risk(world_state: WorldState, delta: WorldDelta) -> float:
        uncertainty = world_state.uncertainty.get(delta.facet, 0.5)
        return max(0.0, min(1.0, float(uncertainty)))

    @staticmethod
    def _goal_alignment_boost(goal_title: str, delta: WorldDelta) -> float:
        if not goal_title:
            return 1.0

        goal_tokens = {t for t in goal_title.lower().split() if len(t) > 2}
        if not goal_tokens:
            return 1.0

        signal_parts: list[str] = [delta.facet, delta.event]
        for v in delta.details.values():
            if isinstance(v, str):
                signal_parts.append(v.lower())
        signal_text = " ".join(signal_parts)
        if any(tok in signal_text for tok in goal_tokens):
            return GOAL_ALIGNMENT_BOOST
        return 1.0

    @staticmethod
    def _render_recommendation(delta: WorldDelta, depth: int, utility: float) -> str:
        tone = "prioritize" if utility >= 0.5 else "monitor"
        return (
            f"{tone} response to {delta.event} "
            f"({delta.facet}, projected {depth} step{'s' if depth != 1 else ''})"
        )
