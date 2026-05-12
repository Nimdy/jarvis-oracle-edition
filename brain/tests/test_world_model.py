"""Unit tests for the Unified World Model (Phase 1).

Covers: WorldState construction, delta detection, causal rules,
prediction validation (hit/miss with tolerance), and promotion transitions.
"""

from __future__ import annotations

import time
import pytest

from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
    _compute_uncertainty,
)
from cognition.causal_engine import CausalEngine, CausalPrediction, FLOAT_TOLERANCE
from cognition.promotion import WorldModelPromotion, PROMOTE_ACCURACY, MIN_PREDICTIONS_FOR_PROMOTION
from cognition.world_model import WorldModel


# ---------------------------------------------------------------------------
# WorldState construction
# ---------------------------------------------------------------------------

class TestWorldState:
    def test_default_construction(self):
        ws = WorldState()
        assert ws.version == 0
        assert ws.physical.entity_count == 0
        assert ws.user.present is False
        assert ws.conversation.active is False
        assert ws.system.mode == "passive"
        assert "physical" in ws.uncertainty
        assert "user" in ws.uncertainty

    def test_frozen(self):
        ws = WorldState()
        with pytest.raises(AttributeError):
            ws.version = 99  # type: ignore[misc]

    def test_resolve_field(self):
        ws = WorldState(
            user=UserState(engagement=0.75, emotion="happy"),
            system=SystemState(mode="focused"),
        )
        assert ws.resolve_field("user.engagement") == 0.75
        assert ws.resolve_field("user.emotion") == "happy"
        assert ws.resolve_field("system.mode") == "focused"
        assert ws.resolve_field("nonexistent.field") is None
        assert ws.resolve_field("user.nonexistent") is None

    def test_to_dict(self):
        ws = WorldState(
            physical=PhysicalState(
                entities=({"label": "chair"},),
                entity_count=1,
            ),
            version=5,
        )
        d = ws.to_dict()
        assert d["version"] == 5
        assert isinstance(d["physical"]["entities"], list)
        assert d["physical"]["entity_count"] == 1

    def test_version_increments(self):
        ws1 = WorldState(version=1)
        ws2 = WorldState(version=2)
        assert ws2.version > ws1.version


class TestUncertainty:
    def test_zero_staleness_full_confidence(self):
        u = _compute_uncertainty("physical", 0.0, [1.0])
        assert u < 0.2

    def test_high_staleness_low_confidence(self):
        u = _compute_uncertainty("physical", 60.0, [0.1])
        assert u > 0.7

    def test_no_signals(self):
        u = _compute_uncertainty("user", 30.0, [])
        assert 0.3 < u < 0.8

    def test_clipped_to_one(self):
        u = _compute_uncertainty("physical", 9999.0, [0.0])
        assert u <= 1.0


# ---------------------------------------------------------------------------
# Delta detection
# ---------------------------------------------------------------------------

class TestDeltaDetection:
    def _make_model(self) -> WorldModel:
        return WorldModel()

    def test_user_arrived(self):
        wm = self._make_model()
        prev = WorldState(version=1, user=UserState(present=False))
        curr = WorldState(version=2, user=UserState(present=True, presence_confidence=0.9),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "user_arrived" in events

    def test_user_departed(self):
        wm = self._make_model()
        prev = WorldState(version=1, user=UserState(present=True))
        curr = WorldState(version=2, user=UserState(present=False),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "user_departed" in events

    def test_emotion_changed(self):
        wm = self._make_model()
        prev = WorldState(version=1, user=UserState(emotion="neutral"))
        curr = WorldState(version=2, user=UserState(emotion="happy", emotion_confidence=0.8),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "emotion_changed" in events

    def test_no_delta_neutral_to_neutral(self):
        wm = self._make_model()
        prev = WorldState(version=1, user=UserState(emotion="neutral"))
        curr = WorldState(version=2, user=UserState(emotion="neutral"),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        emotion_deltas = [d for d in deltas if d.event == "emotion_changed"]
        assert len(emotion_deltas) == 0

    def test_engagement_threshold_crossing(self):
        wm = self._make_model()
        prev = WorldState(version=1, user=UserState(engagement=0.2))
        curr = WorldState(version=2, user=UserState(engagement=0.55),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "engagement_crossed_threshold" in events
        thresh_delta = [d for d in deltas if d.event == "engagement_crossed_threshold"][0]
        assert thresh_delta.details["direction"] == "up"

    def test_mode_changed(self):
        wm = self._make_model()
        prev = WorldState(version=1, system=SystemState(mode="passive"))
        curr = WorldState(version=2, system=SystemState(mode="conversational"),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "mode_changed" in events

    def test_entity_appeared(self):
        wm = self._make_model()
        prev = WorldState(version=1, physical=PhysicalState(entities=()))
        curr = WorldState(version=2, physical=PhysicalState(
            entities=({"label": "cup"},),
        ), timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "entity_appeared" in events

    def test_entity_disappeared(self):
        wm = self._make_model()
        prev = WorldState(version=1, physical=PhysicalState(
            entities=({"label": "cup"}, {"label": "chair"}),
        ))
        curr = WorldState(version=2, physical=PhysicalState(
            entities=({"label": "chair"},),
        ), timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "entity_disappeared" in events
        detail = [d for d in deltas if d.event == "entity_disappeared"][0]
        assert detail.details["label"] == "cup"

    def test_conversation_started(self):
        wm = self._make_model()
        prev = WorldState(version=1, conversation=ConversationState(active=False))
        curr = WorldState(version=2, conversation=ConversationState(
            active=True, conversation_id="abc"),
            timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "conversation_started" in events

    def test_health_degraded(self):
        wm = self._make_model()
        prev = WorldState(version=1, system=SystemState(health_score=0.9))
        curr = WorldState(version=2, system=SystemState(health_score=0.5),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        events = [d.event for d in deltas]
        assert "health_degraded" in events

    def test_no_deltas_on_first_tick(self):
        """Version 0 → version 1 should not produce deltas (no previous state)."""
        wm = self._make_model()
        prev = WorldState(version=0)
        curr = WorldState(version=1, user=UserState(present=True),
                          timestamp=time.time())
        deltas = wm._detect_deltas(prev, curr)
        assert deltas == []


# ---------------------------------------------------------------------------
# Causal engine
# ---------------------------------------------------------------------------

class TestCausalEngine:
    def test_no_deltas_only_steady_state(self):
        engine = CausalEngine()
        ws = WorldState()
        preds = engine.infer(ws, [])
        for p in preds:
            assert p.rule_id in (
                "quiet_desk_stays_quiet", "present_user_stays",
                "healthy_system_stays_healthy", "absent_room_stays_quiet",
            ), f"unexpected non-steady-state rule fired: {p.rule_id}"

    def test_steady_state_cooldown(self):
        engine = CausalEngine()
        ws = WorldState()
        preds1 = engine.infer(ws, [])
        assert len(preds1) > 0
        preds2 = engine.infer(ws, [])
        assert len(preds2) == 0, "steady-state rules should not re-fire while pending"

    def test_user_departed_fires(self):
        engine = CausalEngine()
        ws = WorldState(user=UserState(present=False))
        deltas = [WorldDelta(facet="user", event="user_departed")]
        preds = engine.infer(ws, deltas)
        labels = [p.label for p in preds]
        assert "engagement_drops" in labels

    def test_user_arrived_fires(self):
        engine = CausalEngine()
        ws = WorldState(user=UserState(present=True))
        deltas = [WorldDelta(facet="user", event="user_arrived")]
        preds = engine.infer(ws, deltas)
        labels = [p.label for p in preds]
        assert "conversation_likely" in labels

    def test_priority_conflict_resolution(self):
        """When two rules predict conflicting values for the same field,
        the higher-priority rule should win."""
        engine = CausalEngine()
        ws = WorldState(
            user=UserState(present=True),
            system=SystemState(health_score=0.4),
        )
        deltas = [
            WorldDelta(facet="system", event="health_degraded"),
            WorldDelta(facet="user", event="user_arrived"),
        ]
        preds = engine.infer(ws, deltas)
        # health_degraded (priority 100) should not be overridden
        labels = [p.label for p in preds]
        assert "reduce_background_load" in labels

    def test_pruned_rules_not_present(self):
        """Rules with empty predicted_delta or broken semantics were pruned."""
        engine = CausalEngine()
        rule_ids = {r.rule_id for r in engine._rules}
        assert "object_disappeared" not in rule_ids
        assert "object_appeared" not in rule_ids
        assert "multiple_barge_ins" not in rule_ids
        assert "display_to_game" not in rule_ids
        assert "display_to_code" not in rule_ids
        assert "emotion_frustrated_focus" not in rule_ids

    def test_accuracy_initially_zero(self):
        engine = CausalEngine()
        acc = engine.get_accuracy()
        assert acc["total_validated"] == 0
        assert acc["overall_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------------

class TestPredictionValidation:
    def test_hit_exact_match(self):
        engine = CausalEngine()
        pred = CausalPrediction(
            rule_id="test",
            label="test_hit",
            predicted_delta={"user.present": True},
            confidence=0.9,
            horizon_s=0.0,
            created_at=time.time() - 10,
            expires_at=time.time() - 5,
        )
        engine._predictions.append(pred)
        ws = WorldState(user=UserState(present=True))
        validated = engine.validate_predictions(ws)
        assert len(validated) == 1
        assert validated[0].outcome == "hit"

    def test_miss(self):
        engine = CausalEngine()
        pred = CausalPrediction(
            rule_id="test",
            label="test_miss",
            predicted_delta={"user.present": True},
            confidence=0.9,
            horizon_s=0.0,
            created_at=time.time() - 10,
            expires_at=time.time() - 5,
        )
        engine._predictions.append(pred)
        ws = WorldState(user=UserState(present=False))
        validated = engine.validate_predictions(ws)
        assert len(validated) == 1
        assert validated[0].outcome == "miss"

    def test_float_tolerance(self):
        engine = CausalEngine()
        pred = CausalPrediction(
            rule_id="test",
            label="test_tolerance",
            predicted_delta={"user.engagement": 0.3},
            confidence=0.8,
            horizon_s=0.0,
            created_at=time.time() - 10,
            expires_at=time.time() - 5,
        )
        engine._predictions.append(pred)
        ws = WorldState(user=UserState(engagement=0.27))
        validated = engine.validate_predictions(ws)
        assert len(validated) == 1
        assert validated[0].outcome == "hit"

    def test_float_tolerance_miss(self):
        engine = CausalEngine()
        pred = CausalPrediction(
            rule_id="test",
            label="test_tolerance_miss",
            predicted_delta={"user.engagement": 0.3},
            confidence=0.8,
            horizon_s=0.0,
            created_at=time.time() - 10,
            expires_at=time.time() - 5,
        )
        engine._predictions.append(pred)
        ws = WorldState(user=UserState(engagement=0.8))
        validated = engine.validate_predictions(ws)
        assert len(validated) == 1
        assert validated[0].outcome == "miss"

    def test_pending_not_validated_early(self):
        engine = CausalEngine()
        pred = CausalPrediction(
            rule_id="test",
            label="test_pending",
            predicted_delta={"user.present": True},
            confidence=0.9,
            horizon_s=60.0,
        )
        engine._predictions.append(pred)
        ws = WorldState(user=UserState(present=True))
        validated = engine.validate_predictions(ws)
        assert len(validated) == 0
        assert pred.outcome == "pending"

    def test_accuracy_tracking(self):
        engine = CausalEngine()
        for hit in [True, True, False, True]:
            pred = CausalPrediction(
                rule_id="r1",
                label="test",
                predicted_delta={"user.present": hit},
                confidence=0.8,
                horizon_s=0.0,
                created_at=time.time() - 10,
                expires_at=time.time() - 5,
            )
            engine._predictions.append(pred)

        ws = WorldState(user=UserState(present=True))
        engine.validate_predictions(ws)
        acc = engine.get_accuracy()
        assert acc["total_validated"] == 4
        assert acc["per_rule"]["r1"]["hits"] == 3
        assert acc["per_rule"]["r1"]["misses"] == 1


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

class TestPromotion:
    @staticmethod
    def _fresh_promo() -> WorldModelPromotion:
        """Create a promotion instance with clean state (ignores persisted file)."""
        from cognition.promotion import _PromotionState
        promo = WorldModelPromotion.__new__(WorldModelPromotion)
        promo._state = _PromotionState()
        return promo

    def test_starts_at_shadow(self):
        promo = self._fresh_promo()
        assert promo.level == 0

    def test_no_promotion_without_enough_predictions(self):
        promo = self._fresh_promo()
        promo._state.shadow_start_ts = time.time() - 86400 * 2
        for _ in range(30):
            promo.record_outcome(True)
        assert promo.level == 0  # not enough (need 50)

    def test_promotion_with_sufficient_accuracy(self):
        promo = self._fresh_promo()
        promo._state.shadow_start_ts = time.time() - 86400 * 2
        for _ in range(60):
            promo.record_outcome(True)
        assert promo.level >= 1

    def test_demotion_on_low_accuracy(self):
        promo = self._fresh_promo()
        promo._state.level = 2
        promo._state.total_validated = 60
        for _ in range(60):
            promo._state.accuracy_history.append(1.0)
        for _ in range(25):
            promo.record_outcome(False)
        assert promo.level < 2

    def test_status_report(self):
        promo = self._fresh_promo()
        status = promo.get_status()
        assert "level" in status
        assert "level_name" in status
        assert status["level_name"] == "shadow"
        assert "rolling_accuracy" in status


# ---------------------------------------------------------------------------
# WorldModel integration
# ---------------------------------------------------------------------------

class TestWorldModelIntegration:
    def test_update_without_sources(self):
        """WorldModel should work with no subsystems wired."""
        wm = WorldModel()
        ws = wm.update()
        assert ws.version == 1
        assert ws.physical.entity_count == 0
        assert ws.user.present is False

    def test_successive_updates_increment_version(self):
        wm = WorldModel()
        ws1 = wm.update()
        ws2 = wm.update()
        assert ws2.version == ws1.version + 1

    def test_get_state_returns_dict(self):
        wm = WorldModel()
        wm.update()
        state = wm.get_state()
        assert isinstance(state, dict)
        assert "version" in state
        assert "physical" in state
        assert "user" in state
        assert "conversation" in state
        assert "system" in state
        assert "uncertainty" in state
        assert "causal" in state
        assert "promotion" in state
        assert "planner" in state
        assert "recent_deltas" in state

    def test_planner_gated_before_simulator_advisory(self, monkeypatch, tmp_path):
        monkeypatch.setattr("cognition.promotion.PROMOTION_PATH", str(tmp_path / "wm.json"))
        monkeypatch.setattr("cognition.promotion.SIMULATOR_PROMOTION_PATH", str(tmp_path / "sim.json"))
        wm = WorldModel()
        wm.update()
        planner = wm.get_state().get("planner", {})
        assert planner.get("enabled") is False
        assert planner.get("reason") == "simulator_not_advisory"

    def test_get_deltas_empty_initially(self):
        wm = WorldModel()
        wm.update()
        deltas = wm.get_deltas()
        assert isinstance(deltas, list)

    def test_context_summary(self):
        wm = WorldModel()
        wm.update()
        summary = wm.build_context_summary()
        assert "Situational Awareness" in summary
        assert "Room:" in summary
        assert "User:" in summary
        assert "Conversation:" in summary
