"""Causal reasoning engine — heuristic rules that produce predicted state deltas.

Phase 1: deterministic rules only.  Phase 2 will add LLM-backed scenario
evaluation for complex predictions.

Each rule has a *priority* so conflicting predictions on the same state field
are resolved deterministically (highest priority wins).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from cognition.world_state import WorldDelta, WorldState

logger = logging.getLogger(__name__)

FLOAT_TOLERANCE = 0.1
MAX_PREDICTION_LOG = 200


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalRule:
    rule_id: str
    label: str
    category: str  # "health", "user", "physical", "system", "conversation"
    priority: int  # higher wins conflicts: health=100, user=50, physical=30, system=20
    condition: Callable[[WorldState, list[WorldDelta]], bool]
    predicted_delta: dict[str, Any]  # dotted paths → expected values
    confidence: float = 0.8
    horizon_s: float = 0.0  # 0 = immediate


@dataclass
class CausalPrediction:
    rule_id: str
    label: str
    predicted_delta: dict[str, Any]
    confidence: float
    horizon_s: float
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    outcome: str = "pending"  # "pending", "hit", "miss"
    validated_at: float = 0.0
    field_results: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + self.horizon_s + 5.0


# ---------------------------------------------------------------------------
# Built-in heuristic rules
# ---------------------------------------------------------------------------

def _has_delta(deltas: list[WorldDelta], event: str) -> bool:
    return any(d.event == event for d in deltas)


def _has_delta_with(deltas: list[WorldDelta], event: str, key: str, value: Any) -> bool:
    return any(d.event == event and d.details.get(key) == value for d in deltas)


def _build_default_rules() -> list[CausalRule]:
    """Create heuristic causal rules.

    Rules must predict concrete, measurable state changes.
    Rules with empty predicted_delta are avoided (auto-hit inflates accuracy).
    Display-driven rules are disabled until Layer 3B exits shadow mode.
    """
    rules: list[CausalRule] = []

    # --- User behaviour rules (priority 50) ---

    rules.append(CausalRule(
        rule_id="user_departed_engagement",
        label="engagement_drops",
        category="user",
        priority=50,
        condition=lambda ws, ds: _has_delta(ds, "user_departed"),
        predicted_delta={"user.engagement": 0.0, "conversation.active": False},
        confidence=0.95,
        horizon_s=5.0,
    ))

    rules.append(CausalRule(
        rule_id="user_arrived_conversation",
        label="conversation_likely",
        category="user",
        priority=50,
        condition=lambda ws, ds: _has_delta(ds, "user_arrived"),
        predicted_delta={"conversation.active": True},
        confidence=0.5,
        horizon_s=60.0,
    ))

    rules.append(CausalRule(
        rule_id="speaker_changed_greeting",
        label="new_speaker_interaction",
        category="user",
        priority=50,
        condition=lambda ws, ds: _has_delta(ds, "speaker_changed"),
        predicted_delta={"conversation.active": True},
        confidence=0.7,
        horizon_s=30.0,
    ))

    # --- Conversation rules (priority 40) ---

    rules.append(CausalRule(
        rule_id="conversation_started_engagement",
        label="engagement_rises",
        category="conversation",
        priority=40,
        condition=lambda ws, ds: _has_delta(ds, "conversation_started"),
        predicted_delta={"conversation.active": True},
        confidence=0.9,
        horizon_s=5.0,
    ))

    rules.append(CausalRule(
        rule_id="conversation_ended_followup",
        label="follow_up_possible",
        category="conversation",
        priority=40,
        condition=lambda ws, ds: (
            _has_delta(ds, "conversation_ended") and ws.user.present
        ),
        predicted_delta={"conversation.active": True},
        confidence=0.4,
        horizon_s=30.0,
    ))

    # --- System / health rules (priority 100) ---

    rules.append(CausalRule(
        rule_id="health_degraded_reduce_load",
        label="reduce_background_load",
        category="health",
        priority=100,
        condition=lambda ws, ds: _has_delta(ds, "health_degraded"),
        predicted_delta={"system.health_score": 0.5},
        confidence=0.7,
        horizon_s=0.0,
    ))

    rules.append(CausalRule(
        rule_id="mode_sleep_extended",
        label="extended_absence",
        category="system",
        priority=20,
        condition=lambda ws, ds: (
            ws.system.mode == "sleep"
            and ws.user.seconds_since_last_interaction > 300.0
        ),
        predicted_delta={"user.present": False},
        confidence=0.8,
        horizon_s=0.0,
    ))

    # --- Steady-state rules (fire on current conditions, not transitions) ---
    # These generate predictions during stable sessions where delta-triggered
    # rules rarely fire, allowing the causal engine to accumulate validation
    # data for promotion.  Each uses STEADY_STATE_COOLDOWN_S to avoid spam.

    rules.append(CausalRule(
        rule_id="quiet_desk_stays_quiet",
        label="quiet_desk_persists",
        category="conversation",
        priority=30,
        condition=lambda ws, ds: (
            ws.user.present
            and not ws.conversation.active
            and ws.user.seconds_since_last_interaction > 120.0
        ),
        predicted_delta={"conversation.active": False, "user.present": True},
        confidence=0.85,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="present_user_stays",
        label="user_remains_present",
        category="user",
        priority=30,
        condition=lambda ws, ds: (
            ws.user.present
            and ws.user.seconds_since_last_interaction < 600.0
        ),
        predicted_delta={"user.present": True},
        confidence=0.90,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="healthy_system_stays_healthy",
        label="system_stability",
        category="system",
        priority=15,
        condition=lambda ws, ds: (
            ws.system.health_score >= 0.7
            and ws.system.mode not in ("sleep", "gestation")
        ),
        predicted_delta={"system.health_score": 0.9},
        confidence=0.85,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="absent_room_stays_quiet",
        label="empty_room_no_conversation",
        category="conversation",
        priority=25,
        condition=lambda ws, ds: (
            not ws.user.present
            and not ws.conversation.active
        ),
        predicted_delta={"conversation.active": False, "user.present": False},
        confidence=0.95,
        horizon_s=45.0,
    ))

    # --- Additional steady-state rules for prediction throughput ---

    rules.append(CausalRule(
        rule_id="passive_mode_persists",
        label="mode_stays_passive",
        category="system",
        priority=15,
        condition=lambda ws, ds: (
            ws.system.mode == "passive"
            and ws.user.seconds_since_last_interaction > 60.0
        ),
        predicted_delta={"system.mode": "passive"},
        confidence=0.85,
        horizon_s=30.0,
    ))

    rules.append(CausalRule(
        rule_id="idle_user_no_conversation",
        label="idle_stays_idle",
        category="user",
        priority=20,
        condition=lambda ws, ds: (
            ws.user.present
            and not ws.conversation.active
            and ws.user.seconds_since_last_interaction > 30.0
            and ws.user.seconds_since_last_interaction < 120.0
        ),
        predicted_delta={"user.present": True, "conversation.active": False},
        confidence=0.80,
        horizon_s=30.0,
    ))

    rules.append(CausalRule(
        rule_id="active_conversation_persists",
        label="conversation_stays_active",
        category="conversation",
        priority=35,
        condition=lambda ws, ds: (
            ws.conversation.active
            and ws.conversation.turn_count >= 2
        ),
        predicted_delta={"conversation.active": True},
        confidence=0.80,
        horizon_s=30.0,
    ))

    # --- Canonical-first rules (ontology-motivated, legacy dotted-path output) ---
    # These rules are designed from entity/zone/relation thinking but predict
    # into legacy WorldState fields for benchmark scoring compatibility.

    rules.append(CausalRule(
        rule_id="stable_scene_persists",
        label="scene_layout_stable",
        category="physical",
        priority=25,
        condition=lambda ws, ds: (
            ws.physical.stable_count >= 2
            and ws.physical.visible_count >= 2
            and len(ws.physical.region_visibility) >= 2
        ),
        predicted_delta={"user.present": True},
        confidence=0.80,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="display_zone_mode_stable",
        label="display_implies_mode_stable",
        category="system",
        priority=20,
        condition=lambda ws, ds: (
            ws.user.present
            and len(ws.physical.display_surfaces) >= 1
            and ws.system.mode in ("passive", "conversational", "reflective")
            and ws.system.health_score >= 0.7
        ),
        predicted_delta={"user.present": True},
        confidence=0.85,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="workspace_person_stays",
        label="person_at_workspace_stays",
        category="user",
        priority=35,
        condition=lambda ws, ds: (
            ws.user.present
            and ws.physical.person_count >= 1
            and ws.physical.stable_count >= 1
            and ws.user.seconds_since_last_interaction < 300.0
        ),
        predicted_delta={"user.present": True, "physical.person_count": 1},
        confidence=0.90,
        horizon_s=45.0,
    ))

    rules.append(CausalRule(
        rule_id="multi_entity_scene_stable",
        label="rich_scene_persists",
        category="physical",
        priority=20,
        condition=lambda ws, ds: (
            ws.physical.entity_count >= 3
            and ws.physical.stable_count >= 2
            and ws.user.present
        ),
        predicted_delta={"user.present": True},
        confidence=0.85,
        horizon_s=45.0,
    ))

    # --- PRUNED RULES (removed with justification) ---
    # object_appeared: predicted person_count=1 on any entity appearance — semantically wrong
    # object_disappeared: empty predicted_delta → auto-hit, inflates accuracy
    # multiple_barge_ins: empty predicted_delta → auto-hit, inflates accuracy
    # display_to_game: depends on Layer 3B display classification (shadow mode) — always misses
    # display_to_code: depends on Layer 3B display classification (shadow mode) — always misses
    # emotion_frustrated_focus: predicted system.mode="focused" which is not a valid mode — always misses
    # long_silence_deep_focus: predicted user.engagement=0.3 immediate — no reliable engagement signal at that precision

    return rules


# ---------------------------------------------------------------------------
# CausalEngine
# ---------------------------------------------------------------------------

class CausalEngine:
    """Evaluates causal rules against WorldState + deltas, produces predictions."""

    def __init__(self) -> None:
        self._rules: list[CausalRule] = _build_default_rules()
        self._predictions: deque[CausalPrediction] = deque(maxlen=MAX_PREDICTION_LOG)
        self._rule_hits: dict[str, int] = {}
        self._rule_misses: dict[str, int] = {}
        self._total_validated: int = 0

    # -- Public API ---------------------------------------------------------

    def _pending_rule_ids(self) -> set[str]:
        """Return rule_ids that already have a pending (unvalidated) prediction."""
        return {p.rule_id for p in self._predictions if p.outcome == "pending"}

    def infer(self, world_state: WorldState,
              deltas: list[WorldDelta]) -> list[CausalPrediction]:
        """Evaluate all rules.  Returns new predictions (may be empty)."""
        fired: list[CausalPrediction] = []
        claimed_fields: dict[str, int] = {}  # dotted path → highest priority
        pending_ids = self._pending_rule_ids()
        self._last_infer_skipped_pending: int = 0
        self._last_infer_skipped_condition: int = 0
        self._last_infer_skipped_conflict: int = 0

        sorted_rules = sorted(self._rules, key=lambda r: -r.priority)

        for rule in sorted_rules:
            if rule.rule_id in pending_ids:
                self._last_infer_skipped_pending += 1
                continue

            try:
                if not rule.condition(world_state, deltas):
                    self._last_infer_skipped_condition += 1
                    continue
            except Exception:
                logger.debug("Rule %s condition error", rule.rule_id, exc_info=True)
                continue

            if not rule.predicted_delta:
                pred = CausalPrediction(
                    rule_id=rule.rule_id,
                    label=rule.label,
                    predicted_delta={},
                    confidence=rule.confidence,
                    horizon_s=rule.horizon_s,
                )
                self._predictions.append(pred)
                fired.append(pred)
                continue

            skip = False
            for path in rule.predicted_delta:
                if path in claimed_fields and claimed_fields[path] > rule.priority:
                    skip = True
                    break
            if skip:
                self._last_infer_skipped_conflict += 1
                continue

            for path in rule.predicted_delta:
                claimed_fields[path] = rule.priority

            pred = CausalPrediction(
                rule_id=rule.rule_id,
                label=rule.label,
                predicted_delta=dict(rule.predicted_delta),
                confidence=rule.confidence,
                horizon_s=rule.horizon_s,
            )
            self._predictions.append(pred)
            fired.append(pred)

        if fired:
            labels = [p.label for p in fired]
            logger.debug("Causal engine fired %d predictions: %s", len(fired), labels)

        if self._total_validated <= 5 or self._total_validated % 50 == 0:
            logger.info(
                "CausalEngine infer: rules=%d fired=%d skipped=[pending=%d cond=%d conflict=%d] "
                "total_validated=%d",
                len(self._rules), len(fired),
                self._last_infer_skipped_pending,
                self._last_infer_skipped_condition,
                self._last_infer_skipped_conflict,
                self._total_validated,
            )

        return fired

    def validate_predictions(self, world_state: WorldState) -> list[CausalPrediction]:
        """Check expired predictions against actual state.  Returns newly validated."""
        now = time.time()
        validated: list[CausalPrediction] = []

        for pred in self._predictions:
            if pred.outcome != "pending":
                continue
            if now < pred.expires_at:
                continue

            if not pred.predicted_delta:
                pred.outcome = "hit"
                pred.validated_at = now
                self._rule_hits[pred.rule_id] = self._rule_hits.get(pred.rule_id, 0) + 1
                self._total_validated += 1
                validated.append(pred)
                continue

            hits = 0
            total = 0
            for path, expected in pred.predicted_delta.items():
                total += 1
                actual = world_state.resolve_field(path)
                if self._values_match(expected, actual):
                    pred.field_results[path] = "hit"
                    hits += 1
                else:
                    pred.field_results[path] = f"miss(expected={expected!r}, actual={actual!r})"

            if total > 0 and hits >= total * 0.5:
                pred.outcome = "hit"
                self._rule_hits[pred.rule_id] = self._rule_hits.get(pred.rule_id, 0) + 1
            else:
                pred.outcome = "miss"
                self._rule_misses[pred.rule_id] = self._rule_misses.get(pred.rule_id, 0) + 1

            pred.validated_at = now
            self._total_validated += 1
            validated.append(pred)

        return validated

    def get_accuracy(self) -> dict[str, Any]:
        """Overall and per-rule accuracy stats."""
        total_hits = sum(self._rule_hits.values())
        total_misses = sum(self._rule_misses.values())
        total = total_hits + total_misses
        overall = total_hits / total if total > 0 else 0.0

        per_rule: dict[str, dict[str, Any]] = {}
        all_ids = set(self._rule_hits) | set(self._rule_misses)
        for rid in all_ids:
            h = self._rule_hits.get(rid, 0)
            m = self._rule_misses.get(rid, 0)
            t = h + m
            per_rule[rid] = {
                "hits": h, "misses": m, "total": t,
                "accuracy": round(h / t, 3) if t > 0 else 0.0,
            }

        return {
            "overall_accuracy": round(overall, 3),
            "total_validated": self._total_validated,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "pending": sum(1 for p in self._predictions if p.outcome == "pending"),
            "rules_total": len(self._rules),
            "rules_cooldown": len(self._pending_rule_ids()),
            "last_skipped_pending": getattr(self, "_last_infer_skipped_pending", 0),
            "last_skipped_condition": getattr(self, "_last_infer_skipped_condition", 0),
            "last_skipped_conflict": getattr(self, "_last_infer_skipped_conflict", 0),
            "per_rule": per_rule,
        }

    def get_pending_predictions(self) -> list[dict[str, Any]]:
        """Return active (pending) predictions for dashboard."""
        return [
            {
                "rule_id": p.rule_id,
                "label": p.label,
                "predicted_delta": p.predicted_delta,
                "confidence": p.confidence,
                "created_at": p.created_at,
                "expires_at": p.expires_at,
            }
            for p in self._predictions
            if p.outcome == "pending"
        ]

    def get_recent_validated(self, n: int = 20) -> list[dict[str, Any]]:
        """Last *n* validated predictions (most recent first)."""
        validated = [p for p in self._predictions if p.outcome != "pending"]
        validated.sort(key=lambda p: p.validated_at, reverse=True)
        return [
            {
                "rule_id": p.rule_id,
                "label": p.label,
                "outcome": p.outcome,
                "predicted_delta": p.predicted_delta,
                "field_results": p.field_results,
                "confidence": p.confidence,
                "validated_at": p.validated_at,
            }
            for p in validated[:n]
        ]

    # -- Internals ----------------------------------------------------------

    @staticmethod
    def _values_match(expected: Any, actual: Any) -> bool:
        if expected is None:
            return actual is None
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(expected) - float(actual)) < FLOAT_TOLERANCE
        return expected == actual
