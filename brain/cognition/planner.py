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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 (#16): CognitivePlanner — multi-step path search (read-only, data-gated)
# ─────────────────────────────────────────────────────────────────────────────
#
# Where :class:`WorldPlanner` ranks SINGLE candidate moves independently, the
# CognitivePlanner searches over multi-step action SEQUENCES. It chains the
# MentalSimulator forward across a path — the terminal projected state of one
# step becomes the input state of the next — scores the whole path, and proposes
# the best one (plus alternatives). It is:
#
#   - read-only / advisory-only: proposes paths, never executes, never emits
#     events, never mutates WorldState (the simulator already guarantees state
#     immutability and asserts it).
#   - no-LLM: deterministic beam search over REAL observed deltas — it never
#     invents actions. The action vocabulary is exactly the candidate moves the
#     world model already surfaced.
#   - DATA-GATED on the Mental Simulator reaching *advisory* — i.e. 100+ verified,
#     live-only validated simulations at >=0.70 accuracy over >=48h shadow (see
#     ``cognition.promotion.SimulatorPromotion`` / ``SIM_MIN_SIMULATIONS``). Until
#     the simulator earns that trust, the planner stays DORMANT and proposes
#     nothing. A dormant planner here is gate-blocked BY DESIGN, not broken: you
#     cannot search trustworthy multi-step paths on an unproven simulator.

HORIZON = 3                      # max actions in a planned sequence
BEAM_WIDTH = 3                   # partial paths kept per depth
CANDIDATE_CAP = 4                # candidate moves considered per step
PATH_STEP_COST = 0.02            # small per-step penalty: prefer shorter paths on ties
PATH_DEPTH_DISCOUNT = 0.9        # discount deeper (less certain) step contributions
# Mirrors cognition.promotion.SIM_MIN_SIMULATIONS — the #16 "100+ verified sims" gate.
MIN_VERIFIED_SIMULATIONS = 100


def _facet_risk(world_state: WorldState, delta: WorldDelta) -> float:
    """Facet-uncertainty risk in [0,1]. Mirrors ``WorldPlanner._compute_risk``."""
    uncertainty = world_state.uncertainty.get(delta.facet, 0.5)
    return max(0.0, min(1.0, float(uncertainty)))


def _goal_alignment(goal_title: str, delta: WorldDelta, projected: WorldState) -> float:
    """Goal-alignment multiplier in [1.0, GOAL_ALIGNMENT_BOOST].

    Mirrors ``WorldPlanner._goal_alignment_boost`` (pure token overlap, no LLM)
    but also lets the projected terminal goal title count as an alignment signal
    when a path keeps/reaches the active goal.
    """
    if not goal_title:
        return 1.0
    goal_tokens = {t for t in goal_title.lower().split() if len(t) > 2}
    if not goal_tokens:
        return 1.0
    parts: list[str] = [delta.facet, delta.event]
    for v in delta.details.values():
        if isinstance(v, str):
            parts.append(v.lower())
    proj_goal = (projected.system.active_goal_title or "").lower()
    if proj_goal:
        parts.append(proj_goal)
    signal_text = " ".join(parts)
    if any(tok in signal_text for tok in goal_tokens):
        return GOAL_ALIGNMENT_BOOST
    return 1.0


@dataclass(frozen=True)
class PlanStep:
    """One action in a planned multi-step sequence."""

    order: int
    event: str
    facet: str
    projected_confidence: float
    risk: float
    contribution: float  # this step's discounted utility contribution to the path

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "event": self.event,
            "facet": self.facet,
            "projected_confidence": round(self.projected_confidence, 3),
            "risk": round(self.risk, 3),
            "contribution": round(self.contribution, 3),
        }


@dataclass(frozen=True)
class PlanPath:
    """A scored multi-step action sequence."""

    steps: tuple[PlanStep, ...]
    path_utility: float
    terminal_goal_alignment: float
    horizon: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "length": len(self.steps),
            "path_utility": round(self.path_utility, 3),
            "terminal_goal_alignment": round(self.terminal_goal_alignment, 3),
            "horizon": self.horizon,
            "recommendation": self.recommendation,
        }


@dataclass
class _Partial:
    """Mutable beam-search frontier node (internal — never surfaced)."""

    state: WorldState
    steps: tuple[PlanStep, ...]
    used: frozenset  # candidate indices already consumed on this path
    cum_utility: float
    terminal_alignment: float


class CognitivePlanner:
    """Multi-step path planner: chains the simulator to score action sequences.

    Read-only / advisory-only and DATA-GATED on the Mental Simulator reaching
    advisory (>= ``MIN_VERIFIED_SIMULATIONS`` verified live simulations). Fed the
    same inputs as :class:`WorldPlanner` plus the simulator's verified-simulation
    count, so the gate is legible in the surfaced state.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "enabled": False,
            "active": False,
            "reason": "init",
            "generated_at": 0.0,
            "verified_simulations": 0,
            "min_verified_simulations": MIN_VERIFIED_SIMULATIONS,
            "horizon": HORIZON,
            "selected": None,
            "paths": [],
        }

    def evaluate(
        self,
        *,
        world_state: WorldState,
        deltas: list[WorldDelta],
        simulator: MentalSimulator,
        simulator_promotion_level: int,
        verified_simulations: int = 0,
        goal_title: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        self._state["generated_at"] = round(now, 3)
        self._state["verified_simulations"] = int(verified_simulations)

        # ── DATA GATE (#16): simulator advisory AND 100+ verified live sims ──
        # Belt-and-suspenders: advisory level already implies >=100 verified, but
        # we check the count explicitly so the gate matches the issue text and the
        # progress toward it is visible while dormant.
        if (simulator_promotion_level < SIMULATOR_ADVISORY_LEVEL
                or verified_simulations < MIN_VERIFIED_SIMULATIONS):
            self._state.update({
                "enabled": False,
                "active": False,
                "reason": (
                    "simulator_not_advisory "
                    f"(verified {int(verified_simulations)}/{MIN_VERIFIED_SIMULATIONS})"
                ),
                "selected": None,
                "paths": [],
            })
            return self.get_state()

        self._state["enabled"] = True

        candidates = sorted(
            deltas, key=lambda d: d.confidence, reverse=True,
        )[:CANDIDATE_CAP]
        if not candidates:
            self._state.update({
                "active": False,
                "reason": "no_deltas",
                "selected": None,
                "paths": [],
            })
            return self.get_state()

        paths = self._search(world_state, candidates, simulator, goal_title)
        path_dicts = [p.to_dict() for p in paths]
        self._state.update({
            "active": bool(path_dicts),
            "reason": "ok" if path_dicts else "no_viable_paths",
            "selected": path_dicts[0] if path_dicts else None,
            "paths": path_dicts,
        })
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    # -- Search -------------------------------------------------------------

    def _search(
        self,
        world_state: WorldState,
        candidates: list[WorldDelta],
        simulator: MentalSimulator,
        goal_title: str,
    ) -> list[PlanPath]:
        """Bounded beam search over candidate-delta sequences.

        Worst case ~ ``CANDIDATE_CAP + BEAM_WIDTH * CANDIDATE_CAP * (HORIZON-1)``
        simulator calls per tick — small, deterministic, read-only.
        """
        beam: list[_Partial] = [
            _Partial(
                state=world_state, steps=(), used=frozenset(),
                cum_utility=0.0, terminal_alignment=1.0,
            ),
        ]
        complete: list[PlanPath] = []

        for depth in range(HORIZON):
            expansions: list[_Partial] = []
            for partial in beam:
                for idx, delta in enumerate(candidates):
                    if idx in partial.used:
                        continue
                    try:
                        trace = simulator.simulate(
                            partial.state, delta, max_depth=2,
                            source="cognitive_planner",
                        )
                    except Exception:
                        continue
                    if trace.depth <= 0:
                        continue  # delta yields no projection from here — not viable

                    projected = trace.steps[-1].state if trace.steps else partial.state
                    risk = _facet_risk(partial.state, delta)
                    align = _goal_alignment(goal_title, delta, projected)
                    step_util = (
                        trace.total_confidence * delta.confidence * align
                    ) - (risk * RISK_PENALTY_WEIGHT)
                    contribution = (
                        step_util * (PATH_DEPTH_DISCOUNT ** depth)
                    ) - PATH_STEP_COST

                    step = PlanStep(
                        order=depth,
                        event=delta.event,
                        facet=delta.facet,
                        projected_confidence=trace.total_confidence,
                        risk=risk,
                        contribution=contribution,
                    )
                    expansions.append(_Partial(
                        state=projected,
                        steps=partial.steps + (step,),
                        used=partial.used | {idx},
                        cum_utility=partial.cum_utility + contribution,
                        terminal_alignment=align,
                    ))

            if not expansions:
                break

            # Every expansion is a complete candidate path (length 1..HORIZON);
            # shorter and longer paths compete directly via step-cost + discount.
            for p in expansions:
                complete.append(self._finalize(p, goal_title))

            expansions.sort(key=lambda p: p.cum_utility, reverse=True)
            beam = expansions[:BEAM_WIDTH]

        # Rank: utility desc, then prefer shorter, then higher terminal alignment.
        complete.sort(
            key=lambda pp: (pp.path_utility, -len(pp.steps), pp.terminal_goal_alignment),
            reverse=True,
        )
        seen: set[tuple[str, ...]] = set()
        ranked: list[PlanPath] = []
        for pp in complete:
            sig = tuple(s.event for s in pp.steps)
            if sig in seen:
                continue
            seen.add(sig)
            ranked.append(pp)
        return ranked[:MAX_PLAN_OPTIONS]

    @staticmethod
    def _finalize(partial: _Partial, goal_title: str) -> PlanPath:
        return PlanPath(
            steps=partial.steps,
            path_utility=partial.cum_utility,
            terminal_goal_alignment=partial.terminal_alignment,
            horizon=len(partial.steps),
            recommendation=CognitivePlanner._render(partial),
        )

    @staticmethod
    def _render(partial: _Partial) -> str:
        tone = "pursue" if partial.cum_utility >= 0.5 else "monitor"
        seq = " → ".join(s.event for s in partial.steps)
        n = len(partial.steps)
        return f"{tone} path: {seq} ({n} step{'s' if n != 1 else ''}, util {partial.cum_utility:.2f})"
