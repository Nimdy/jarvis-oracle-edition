"""Mental Simulator — Phase 3 of the World Model.

Takes a :class:`WorldState` + hypothetical :class:`WorldDelta`, projects
forward N steps using :class:`CausalEngine` rules, and returns a
:class:`SimulationTrace`.  Answers: "if X changed, what would happen next?"

Key constraints:
  - **Read-only**: never mutates real ``WorldState``.  All simulation on
    copied frozen dataclasses.
  - **Max depth 3**: prevents combinatorial explosion.
  - **Shadow mode first**: promoted to advisory after accuracy validation.
  - **Reuses existing CausalEngine**: no new rule formats or prediction types.
  - **No LLM**: purely deterministic rule evaluation.
  - **No event emission**: simulations are silent — no EventBus side-effects.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace, fields
from typing import Any

from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
)
from cognition.causal_engine import CausalEngine, CausalRule

logger = logging.getLogger(__name__)

MAX_DEPTH = 3
CONFIDENCE_DECAY = 0.85
MAX_TRACE_LOG = 200


@dataclass(frozen=True)
class SimulationStep:
    """One forward-projection step."""
    state: WorldState
    applied_rules: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class SimulationTrace:
    """Complete simulation result."""
    steps: tuple[SimulationStep, ...]
    initial_delta: WorldDelta
    total_confidence: float
    depth: int
    elapsed_ms: float = 0.0
    source: str = ""


def _apply_delta_to_state(state: WorldState, delta: WorldDelta) -> WorldState:
    """Apply a WorldDelta's details to a WorldState, producing a new frozen copy.

    Maps delta events to state field changes using the delta's ``details``
    dict.  Falls back to the original state when a delta carries no
    applicable detail keys.
    """
    phys = state.physical
    user = state.user
    conv = state.conversation
    sys = state.system

    event = delta.event
    d = delta.details

    if event == "user_arrived":
        user = replace(user, present=True,
                       presence_confidence=d.get("confidence", 0.8))
    elif event == "user_departed":
        user = replace(user, present=False, presence_confidence=0.0,
                       engagement=0.0)
    elif event == "emotion_changed":
        user = replace(user, emotion=d.get("to", d.get("emotion", user.emotion)),
                       emotion_confidence=d.get("confidence", user.emotion_confidence))
    elif event == "engagement_crossed_threshold":
        user = replace(user, engagement=d.get("value", d.get("engagement", user.engagement)))
    elif event == "speaker_changed":
        user = replace(user, speaker_name=d.get("to", d.get("speaker", "")),
                       speaker_confidence=d.get("confidence", 0.0),
                       identity_method=d.get("method", ""))
    elif event == "conversation_started":
        conv = replace(conv, active=True,
                       topic=d.get("topic", conv.topic))
    elif event == "conversation_ended":
        conv = replace(conv, active=False)
    elif event == "topic_changed":
        conv = replace(conv, topic=d.get("to", d.get("topic", conv.topic)))
    elif event == "follow_up_started":
        conv = replace(conv, follow_up_active=True)
    elif event == "mode_changed":
        sys = replace(sys, mode=d.get("to", d.get("mode", sys.mode)))
    elif event == "health_degraded":
        sys = replace(sys, health_score=d.get("to", d.get("score", 0.5)))
    elif event == "health_recovered":
        sys = replace(sys, health_score=d.get("to", d.get("score", 1.0)))
    elif event == "goal_promoted":
        sys = replace(sys, active_goal_title=d.get("title", sys.active_goal_title))
    elif event == "goal_completed":
        sys = replace(sys, active_goal_title="")
    elif event in ("entity_appeared", "entity_disappeared", "entity_moved",
                    "display_content_changed"):
        pass

    return replace(state, physical=phys, user=user,
                   conversation=conv, system=sys)


def _apply_predicted_fields(state: WorldState,
                            predicted_delta: dict[str, Any]) -> WorldState:
    """Apply dotted-path predicted values from a CausalRule to a WorldState.

    The CausalEngine's ``predicted_delta`` uses paths like
    ``"user.engagement"`` or ``"conversation.active"``.  This resolves each
    path and produces a new frozen state with the values applied.
    """
    facet_updates: dict[str, dict[str, Any]] = {}

    for path, value in predicted_delta.items():
        parts = path.split(".", 1)
        if len(parts) != 2:
            continue
        facet_name, field_name = parts
        if facet_name not in ("physical", "user", "conversation", "system"):
            continue
        facet_updates.setdefault(facet_name, {})[field_name] = value

    phys = state.physical
    user = state.user
    conv = state.conversation
    sys = state.system

    if "physical" in facet_updates:
        phys = _safe_replace(phys, facet_updates["physical"])
    if "user" in facet_updates:
        user = _safe_replace(user, facet_updates["user"])
    if "conversation" in facet_updates:
        conv = _safe_replace(conv, facet_updates["conversation"])
    if "system" in facet_updates:
        sys = _safe_replace(sys, facet_updates["system"])

    return replace(state, physical=phys, user=user,
                   conversation=conv, system=sys)


def _safe_replace(obj: Any, updates: dict[str, Any]) -> Any:
    """``dataclasses.replace`` but silently drops keys that aren't fields."""
    valid_names = {f.name for f in fields(obj)}
    filtered = {k: v for k, v in updates.items() if k in valid_names}
    if not filtered:
        return obj
    return replace(obj, **filtered)


class MentalSimulator:
    """Projects hypothetical state changes forward using causal rules.

    The simulator is read-only: it never mutates the real WorldState, never
    emits events, and never writes to any persistence layer.
    """

    def __init__(self, causal_engine: CausalEngine) -> None:
        self._causal = causal_engine
        self._trace_log: deque[SimulationTrace] = deque(maxlen=MAX_TRACE_LOG)
        self._total_simulations: int = 0
        self._total_steps: int = 0
        self._hrr_shadow = None
        try:
            from library.vsa.runtime_config import HRRRuntimeConfig
            from library.vsa.status import (
                register_simulation_shadow_reader,
                register_simulation_shadow_recent,
            )

            _hrr_cfg = HRRRuntimeConfig.from_env()
            if _hrr_cfg.enabled:
                from cognition.hrr_simulation_shadow import HRRSimulationShadow

                self._hrr_shadow = HRRSimulationShadow(_hrr_cfg)
                register_simulation_shadow_reader(self._hrr_shadow.status)
                if hasattr(self._hrr_shadow, "recent"):
                    register_simulation_shadow_recent(self._hrr_shadow.recent)
        except Exception:
            self._hrr_shadow = None

    def simulate(
        self,
        state: WorldState,
        delta: WorldDelta,
        max_depth: int = MAX_DEPTH,
        source: str = "",
    ) -> SimulationTrace:
        """Run a forward simulation from *state* after applying *delta*.

        Returns a :class:`SimulationTrace` with up to *max_depth* steps.
        Each step runs the causal engine on the projected state and applies
        any fired rule predictions to produce the next state.

        The original *state* is never modified.
        """
        t0 = time.monotonic()
        depth = min(max_depth, MAX_DEPTH)

        original_id = id(state)

        current = _apply_delta_to_state(state, delta)
        steps: list[SimulationStep] = []
        confidence = 1.0

        for step_idx in range(depth):
            fired = self._causal.infer(current, [delta] if step_idx == 0 else [])

            applied_rules: list[str] = []
            combined_predictions: dict[str, Any] = {}
            best_confidence = 0.0

            claimed: dict[str, int] = {}
            for pred in fired:
                if not pred.predicted_delta:
                    applied_rules.append(pred.rule_id)
                    continue

                skip = False
                rule_priority = self._get_rule_priority(pred.rule_id)
                for path in pred.predicted_delta:
                    if path in claimed and claimed[path] > rule_priority:
                        skip = True
                        break
                if skip:
                    continue

                for path in pred.predicted_delta:
                    claimed[path] = rule_priority

                applied_rules.append(pred.rule_id)
                combined_predictions.update(pred.predicted_delta)
                best_confidence = max(best_confidence, pred.confidence)

            step_confidence = best_confidence * CONFIDENCE_DECAY if best_confidence > 0 else CONFIDENCE_DECAY
            confidence *= step_confidence

            if combined_predictions:
                current = _apply_predicted_fields(current, combined_predictions)

            steps.append(SimulationStep(
                state=current,
                applied_rules=tuple(applied_rules),
                confidence=round(step_confidence, 4),
            ))

            if not applied_rules:
                break

        assert id(state) == original_id, "Simulator mutated the original WorldState"

        elapsed_ms = (time.monotonic() - t0) * 1000

        trace = SimulationTrace(
            steps=tuple(steps),
            initial_delta=delta,
            total_confidence=round(confidence, 4),
            depth=len(steps),
            elapsed_ms=round(elapsed_ms, 2),
            source=source,
        )

        self._trace_log.append(trace)
        self._total_simulations += 1
        self._total_steps += len(steps)

        if self._hrr_shadow is not None:
            try:
                final_state = steps[-1].state if steps else current
                self._hrr_shadow.record_trace(state, delta, final_state, trace)
            except Exception:
                pass  # observer must never affect simulator output

        return trace

    def get_recent_traces(self, n: int = 10) -> list[dict[str, Any]]:
        """Return recent simulation traces for dashboard / logging."""
        traces = list(self._trace_log)[-n:]
        result = []
        for t in reversed(traces):
            result.append({
                "delta_event": t.initial_delta.event,
                "delta_facet": t.initial_delta.facet,
                "depth": t.depth,
                "total_confidence": t.total_confidence,
                "elapsed_ms": t.elapsed_ms,
                "source": t.source,
                "steps": [
                    {
                        "applied_rules": list(s.applied_rules),
                        "confidence": s.confidence,
                    }
                    for s in t.steps
                ],
            })
        return result

    def get_stats(self) -> dict[str, Any]:
        """Aggregate simulator statistics."""
        traces = list(self._trace_log)
        avg_depth = (
            sum(t.depth for t in traces) / len(traces) if traces else 0.0
        )
        avg_confidence = (
            sum(t.total_confidence for t in traces) / len(traces) if traces else 0.0
        )
        avg_ms = (
            sum(t.elapsed_ms for t in traces) / len(traces) if traces else 0.0
        )
        return {
            "total_simulations": self._total_simulations,
            "total_steps": self._total_steps,
            "trace_buffer_size": len(traces),
            "avg_depth": round(avg_depth, 2),
            "avg_confidence": round(avg_confidence, 3),
            "avg_elapsed_ms": round(avg_ms, 2),
        }

    def _get_rule_priority(self, rule_id: str) -> int:
        """Look up rule priority from the causal engine's rule set."""
        for rule in self._causal._rules:
            if rule.rule_id == rule_id:
                return rule.priority
        return 0
