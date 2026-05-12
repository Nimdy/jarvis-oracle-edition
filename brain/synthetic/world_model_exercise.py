"""Synthetic world model prediction exercise.

Exercises the CausalEngine by constructing synthetic WorldState + WorldDelta
inputs for all 18 rules, running inference, then validating predictions
against expected outcomes.

Truth boundary:
  - Standalone CausalEngine instance — NOT the one inside WorldModel
  - NEVER calls WorldModel.update() (emits WORLD_MODEL_DELTA events)
  - NEVER emits WORLD_MODEL_PREDICTION_VALIDATED (inflates promotion counters)
  - NEVER writes to world_model_promotion.json or simulator_promotion.json
  - Stats only — no distillation signals (no WM specialist NN exists yet)
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognition.causal_engine import CausalEngine, CausalPrediction
from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
)

logger = logging.getLogger(__name__)

REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise"


# ---------------------------------------------------------------------------
# Scenario corpus — covers all 18 causal rules
# ---------------------------------------------------------------------------

def _ts() -> float:
    return time.time()


def _build_scenarios() -> list[dict[str, Any]]:
    """Return scenario dicts covering all 18 CausalEngine rules.

    Each scenario specifies:
      - name: human-readable label
      - rule_ids: which rule(s) should fire
      - before: WorldState before deltas
      - deltas: list of WorldDelta events
      - after: WorldState representing the post-delta ground truth
    """
    now = _ts()
    scenarios: list[dict[str, Any]] = []

    # 1. user_departed_engagement
    scenarios.append({
        "name": "user_departs_room",
        "rule_ids": ["user_departed_engagement"],
        "before": WorldState(
            user=UserState(present=True, engagement=0.7, last_update_ts=now),
            conversation=ConversationState(active=True),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="user", event="user_departed", timestamp=now)],
        "after": WorldState(
            user=UserState(present=False, engagement=0.0, last_update_ts=now),
            conversation=ConversationState(active=False),
            timestamp=now + 6,
        ),
    })

    # 2. user_arrived_conversation
    scenarios.append({
        "name": "user_arrives",
        "rule_ids": ["user_arrived_conversation"],
        "before": WorldState(
            user=UserState(present=False),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="user", event="user_arrived", timestamp=now)],
        "after": WorldState(
            user=UserState(present=True),
            conversation=ConversationState(active=True),
            timestamp=now + 61,
        ),
    })

    # 3. speaker_changed_greeting
    scenarios.append({
        "name": "new_speaker_detected",
        "rule_ids": ["speaker_changed_greeting"],
        "before": WorldState(
            user=UserState(present=True, speaker_name="alice"),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="user", event="speaker_changed", timestamp=now)],
        "after": WorldState(
            user=UserState(present=True, speaker_name="bob"),
            conversation=ConversationState(active=True),
            timestamp=now + 31,
        ),
    })

    # 4. conversation_started_engagement
    scenarios.append({
        "name": "conversation_begins",
        "rule_ids": ["conversation_started_engagement"],
        "before": WorldState(
            user=UserState(present=True, engagement=0.3),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="conversation", event="conversation_started", timestamp=now)],
        "after": WorldState(
            user=UserState(present=True, engagement=0.8),
            conversation=ConversationState(active=True),
            timestamp=now + 6,
        ),
    })

    # 5. conversation_ended_followup (user present)
    scenarios.append({
        "name": "conversation_ends_user_present",
        "rule_ids": ["conversation_ended_followup"],
        "before": WorldState(
            user=UserState(present=True, engagement=0.7),
            conversation=ConversationState(active=True, turn_count=5),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="conversation", event="conversation_ended", timestamp=now)],
        "after": WorldState(
            user=UserState(present=True),
            conversation=ConversationState(active=True),
            timestamp=now + 31,
        ),
    })

    # 6. health_degraded_reduce_load
    scenarios.append({
        "name": "health_degrades",
        "rule_ids": ["health_degraded_reduce_load"],
        "before": WorldState(
            system=SystemState(health_score=0.9, mode="passive"),
            timestamp=now,
        ),
        "deltas": [WorldDelta(facet="system", event="health_degraded", timestamp=now)],
        "after": WorldState(
            system=SystemState(health_score=0.5, mode="passive"),
            timestamp=now + 1,
        ),
    })

    # 7. mode_sleep_extended
    scenarios.append({
        "name": "sleep_mode_long_absence",
        "rule_ids": ["mode_sleep_extended"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=400.0),
            system=SystemState(mode="sleep"),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=False),
            system=SystemState(mode="sleep"),
            timestamp=now + 1,
        ),
    })

    # 8. quiet_desk_stays_quiet
    scenarios.append({
        "name": "quiet_desk_persists",
        "rule_ids": ["quiet_desk_stays_quiet"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=180.0),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            conversation=ConversationState(active=False),
            timestamp=now + 46,
        ),
    })

    # 9. present_user_stays
    scenarios.append({
        "name": "present_user_remains",
        "rule_ids": ["present_user_stays"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=100.0),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            timestamp=now + 46,
        ),
    })

    # 10. healthy_system_stays_healthy
    scenarios.append({
        "name": "system_stability",
        "rule_ids": ["healthy_system_stays_healthy"],
        "before": WorldState(
            system=SystemState(health_score=0.85, mode="passive"),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            system=SystemState(health_score=0.9, mode="passive"),
            timestamp=now + 46,
        ),
    })

    # 11. absent_room_stays_quiet
    scenarios.append({
        "name": "empty_room_no_activity",
        "rule_ids": ["absent_room_stays_quiet"],
        "before": WorldState(
            user=UserState(present=False),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=False),
            conversation=ConversationState(active=False),
            timestamp=now + 46,
        ),
    })

    # 12. passive_mode_persists
    scenarios.append({
        "name": "passive_mode_continues",
        "rule_ids": ["passive_mode_persists"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=120.0),
            system=SystemState(mode="passive"),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            system=SystemState(mode="passive"),
            timestamp=now + 31,
        ),
    })

    # 13. idle_user_no_conversation
    scenarios.append({
        "name": "idle_user_stays_idle",
        "rule_ids": ["idle_user_no_conversation"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=60.0),
            conversation=ConversationState(active=False),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            conversation=ConversationState(active=False),
            timestamp=now + 31,
        ),
    })

    # 14. active_conversation_persists
    scenarios.append({
        "name": "active_conversation_continues",
        "rule_ids": ["active_conversation_persists"],
        "before": WorldState(
            conversation=ConversationState(active=True, turn_count=5),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            conversation=ConversationState(active=True, turn_count=6),
            timestamp=now + 31,
        ),
    })

    # 15. stable_scene_persists
    scenarios.append({
        "name": "stable_scene_layout",
        "rule_ids": ["stable_scene_persists"],
        "before": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(
                stable_count=3, visible_count=3,
                region_visibility={"desk_left": 0.9, "monitor_zone": 0.8},
            ),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(stable_count=3, visible_count=3),
            timestamp=now + 46,
        ),
    })

    # 16. display_zone_mode_stable
    scenarios.append({
        "name": "display_implies_stability",
        "rule_ids": ["display_zone_mode_stable"],
        "before": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(display_surfaces=({"id": "monitor1"},)),
            system=SystemState(health_score=0.9, mode="passive"),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            system=SystemState(health_score=0.9, mode="passive"),
            timestamp=now + 46,
        ),
    })

    # 17. workspace_person_stays
    scenarios.append({
        "name": "person_at_workspace",
        "rule_ids": ["workspace_person_stays"],
        "before": WorldState(
            user=UserState(present=True, seconds_since_last_interaction=60.0),
            physical=PhysicalState(person_count=1, stable_count=1),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(person_count=1, stable_count=1),
            timestamp=now + 46,
        ),
    })

    # 18. multi_entity_scene_stable
    scenarios.append({
        "name": "rich_scene_stable",
        "rule_ids": ["multi_entity_scene_stable"],
        "before": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(entity_count=4, stable_count=3),
            timestamp=now,
        ),
        "deltas": [],
        "after": WorldState(
            user=UserState(present=True),
            physical=PhysicalState(entity_count=4, stable_count=3),
            timestamp=now + 46,
        ),
    })

    return scenarios


def _randomize_scenario(base: dict[str, Any]) -> dict[str, Any]:
    """Apply small random perturbations to a scenario for stress testing."""
    import copy
    s = copy.deepcopy(base)

    before: WorldState = s["before"]
    user = before.user
    sys = before.system

    health_jitter = random.uniform(-0.1, 0.1)
    new_health = max(0.0, min(1.0, sys.health_score + health_jitter))
    s["before"] = WorldState(
        physical=before.physical,
        user=UserState(
            present=user.present,
            engagement=max(0.0, min(1.0, user.engagement + random.uniform(-0.15, 0.15))),
            seconds_since_last_interaction=max(0.0, user.seconds_since_last_interaction + random.uniform(-20, 20)),
            speaker_name=user.speaker_name,
            emotion=user.emotion,
            emotion_confidence=user.emotion_confidence,
            last_update_ts=user.last_update_ts,
        ),
        conversation=before.conversation,
        system=SystemState(
            health_score=new_health,
            mode=sys.mode,
            uptime_s=sys.uptime_s + random.uniform(0, 100),
            memory_count=sys.memory_count,
        ),
        timestamp=before.timestamp,
    )
    return s


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class WorldModelExerciseProfile:
    name: str
    scenario_count: int
    randomize: bool
    description: str = ""


PROFILES: dict[str, WorldModelExerciseProfile] = {
    "smoke": WorldModelExerciseProfile(
        name="smoke", scenario_count=10, randomize=False,
        description="Quick check (10 base scenarios)",
    ),
    "coverage": WorldModelExerciseProfile(
        name="coverage", scenario_count=54, randomize=True,
        description="All 18 rules x 3 variants (54 scenarios)",
    ),
    "stress": WorldModelExerciseProfile(
        name="stress", scenario_count=200, randomize=True,
        description="Randomized high-volume (200 scenarios)",
    ),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class WorldModelExerciseStats:
    scenarios_requested: int = 0
    scenarios_run: int = 0
    predictions_generated: int = 0
    predictions_validated: int = 0
    hits: int = 0
    misses: int = 0
    rules_fired: Counter = field(default_factory=Counter)
    rules_missed: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def accuracy(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def rule_coverage(self) -> float:
        return len(self.rules_fired) / 18.0

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.predictions_generated == 0 and self.scenarios_run > 0:
            reasons.append("zero_predictions_generated")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "scenarios_requested": self.scenarios_requested,
            "scenarios_run": self.scenarios_run,
            "predictions_generated": self.predictions_generated,
            "predictions_validated": self.predictions_validated,
            "hits": self.hits,
            "misses": self.misses,
            "accuracy": round(self.accuracy, 3),
            "rule_coverage": round(self.rule_coverage, 3),
            "rules_fired": dict(self.rules_fired),
            "rules_missed": dict(self.rules_missed),
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"World Model Exercise — {self.scenarios_run} scenarios, "
            f"{self.predictions_generated} predictions, "
            f"accuracy={self.accuracy:.1%}, coverage={self.rule_coverage:.0%} "
            f"in {self.elapsed_s:.1f}s",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.rules_fired:
            lines.append(f"  Rules fired ({len(self.rules_fired)}): "
                         + ", ".join(sorted(self.rules_fired.keys())[:8]))
        if self.rules_missed:
            lines.append(f"  Rules missed: "
                         + ", ".join(sorted(self.rules_missed.keys())[:5]))
        if self.fail_reasons:
            lines.append(f"  FAIL: {', '.join(self.fail_reasons)}")
        else:
            lines.append("  PASS: all checks hold")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_world_model_exercise(
    profile: WorldModelExerciseProfile | None = None,
    count: int | None = None,
) -> WorldModelExerciseStats:
    """Run a synchronous world model prediction exercise."""
    if profile is None:
        profile = PROFILES["coverage"]

    stats = WorldModelExerciseStats(profile_name=profile.name)
    base_scenarios = _build_scenarios()
    n = count or profile.scenario_count
    stats.scenarios_requested = n

    engine = CausalEngine()

    scenarios_to_run: list[dict[str, Any]] = []
    while len(scenarios_to_run) < n:
        for sc in base_scenarios:
            if profile.randomize and len(scenarios_to_run) >= len(base_scenarios):
                scenarios_to_run.append(_randomize_scenario(sc))
            else:
                scenarios_to_run.append(sc)
            if len(scenarios_to_run) >= n:
                break

    for sc in scenarios_to_run:
        try:
            before: WorldState = sc["before"]
            deltas: list[WorldDelta] = sc["deltas"]
            after: WorldState = sc["after"]

            predictions = engine.infer(before, deltas)
            stats.predictions_generated += len(predictions)

            for pred in predictions:
                stats.rules_fired[pred.rule_id] += 1

            expected_rules: list[str] = sc.get("rule_ids", [])
            fired_ids = {p.rule_id for p in predictions}
            for rid in expected_rules:
                if rid not in fired_ids:
                    stats.rules_missed[rid] += 1

            validated = engine.validate_predictions(after)
            stats.predictions_validated += len(validated)

            for v in validated:
                if v.outcome == "hit":
                    stats.hits += 1
                elif v.outcome == "miss":
                    stats.misses += 1

            stats.scenarios_run += 1

        except Exception as exc:
            stats.errors.append(f"{sc.get('name', '?')}: {type(exc).__name__}: {exc}")

    stats.end_time = time.time()
    logger.info(
        "World model exercise: %d scenarios, accuracy=%.1f%%, coverage=%.0f%%",
        stats.scenarios_run, stats.accuracy * 100, stats.rule_coverage * 100,
    )
    return stats
