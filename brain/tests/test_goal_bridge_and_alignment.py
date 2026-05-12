"""Tests for the Goal-to-Execution Bridge and Autonomy Alignment.

Covers:
  - P1: GoalTask.dispatched_intent_id, GoalManager._dispatch_task(),
    derived why_not_executing, rate limiting, task status transitions
  - P2: Hard gate blocking, should_suppress(), annotate_intent(),
    classify_intent_alignment(), opportunity scorer goal impact/existential penalty,
    DriveSignals.active_user_goals curiosity dampening
  - P3: Startup sanitization, stop word additions, cleanup_blocked_jobs bug fix
  - P4: Recognition 3-state machine, evidence accumulation
  - P5: Addressee gate telemetry
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# P1: GoalTask.dispatched_intent_id
# ---------------------------------------------------------------------------

from goals.goal import GoalTask, Goal


class TestGoalTaskDispatchedIntentId:
    def test_default_empty(self):
        task = GoalTask(goal_id="g1", description="test")
        assert task.dispatched_intent_id == ""

    def test_to_dict_includes_field(self):
        task = GoalTask(goal_id="g1", description="test", dispatched_intent_id="int_123")
        d = task.to_dict()
        assert d["dispatched_intent_id"] == "int_123"

    def test_from_dict_round_trip(self):
        task = GoalTask(goal_id="g1", description="test", dispatched_intent_id="int_456")
        d = task.to_dict()
        restored = GoalTask.from_dict(d)
        assert restored.dispatched_intent_id == "int_456"

    def test_from_dict_missing_field_defaults(self):
        d = {"goal_id": "g1", "description": "test"}
        restored = GoalTask.from_dict(d)
        assert restored.dispatched_intent_id == ""


# ---------------------------------------------------------------------------
# P1: GoalManager dispatch bridge
# ---------------------------------------------------------------------------

from goals.goal_manager import GoalManager, _jaccard
from goals.goal_registry import GoalRegistry
from goals.planner import GoalPlanner
from goals.review import GoalReview
from goals.constants import DISPATCH_COOLDOWN_S

import tempfile
from pathlib import Path


def _temp_registry() -> GoalRegistry:
    """Create a GoalRegistry backed by a temp file so tests don't interfere with live data."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    return GoalRegistry(path=Path(tmp.name))


def _make_goal(**kwargs) -> Goal:
    defaults = {
        "title": "Test goal",
        "kind": "user_goal",
        "status": "active",
        "priority": 0.9,
        "explicit_user_requested": True,
        "promoted_at": time.time(),
    }
    defaults.update(kwargs)
    return Goal(**defaults)


class _FakeIntent:
    """Lightweight intent replacement for tests (avoids MagicMock attribute gotchas)."""
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "i_test")
        self.question = kwargs.get("question", "research something")
        self.source_event = kwargs.get("source_event", "drive:curiosity")
        self.source_hint = kwargs.get("source_hint", "any")
        self.goal_id = kwargs.get("goal_id", "")
        self.task_id = kwargs.get("task_id", "")
        self.tag_cluster = kwargs.get("tag_cluster", ())
        self.priority = kwargs.get("priority", 0.5)
        self.status = kwargs.get("status", "queued")
        self.scope = kwargs.get("scope", "local_only")


def _make_intent(**kwargs):
    return _FakeIntent(**kwargs)


class TestGoalManagerDispatch:
    def _make_manager(self) -> GoalManager:
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        return mgr

    def test_dispatch_needs_autonomy(self):
        mgr = self._make_manager()
        goal = _make_goal()
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        mgr._next_task_preview = GoalTask(goal_id=goal.goal_id, description="test")
        mgr._dispatch_task(time.time())
        assert mgr._dispatch_block_reason == "no_autonomy"

    def test_dispatch_needs_focus(self):
        mgr = self._make_manager()
        mgr._autonomy_orch = MagicMock()
        mgr._dispatch_task(time.time())
        assert mgr._dispatch_block_reason == "no_focus"

    def test_dispatch_blocks_on_sleep(self):
        mgr = self._make_manager()
        mgr._autonomy_orch = MagicMock()
        goal = _make_goal()
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        mgr._next_task_preview = GoalTask(goal_id=goal.goal_id, description="test")
        mgr._current_mode = "sleep"
        mgr._dispatch_task(time.time())
        assert mgr._dispatch_block_reason == "mode_gated:sleep"

    def test_dispatch_respects_cooldown(self):
        mgr = self._make_manager()
        mgr._autonomy_orch = MagicMock()
        goal = _make_goal()
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        mgr._next_task_preview = GoalTask(goal_id=goal.goal_id, description="test")
        mgr._current_mode = "focused"
        mgr._last_dispatch_ts = time.time()
        mgr._dispatch_task(time.time())
        assert mgr._dispatch_block_reason == "cooldown"

    def test_dispatch_success(self):
        mgr = self._make_manager()
        orch = MagicMock()
        orch._queue = []
        mgr._autonomy_orch = orch
        goal = _make_goal()
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        task = GoalTask(goal_id=goal.goal_id, description="research something")
        mgr._next_task_preview = task
        mgr._current_mode = "focused"
        mgr._last_dispatch_ts = 0

        mgr._dispatch_task(time.time())

        assert mgr._dispatch_block_reason == ""
        assert task.status == "running"
        assert task.dispatched_intent_id != ""
        assert goal.current_task_id == task.task_id
        assert orch.enqueue.called

    def test_dispatch_enqueue_rejected_does_not_mark_running(self):
        mgr = self._make_manager()
        orch = MagicMock()
        orch._queue = []
        orch.enqueue.return_value = False
        mgr._autonomy_orch = orch
        goal = _make_goal()
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        task = GoalTask(goal_id=goal.goal_id, description="research something")
        mgr._next_task_preview = task
        mgr._current_mode = "focused"
        mgr._last_dispatch_ts = 0

        mgr._dispatch_task(time.time())

        assert mgr._dispatch_block_reason == "enqueue_rejected"
        assert task.status == "pending"
        assert task.dispatched_intent_id == ""
        assert goal.current_task_id is None
        assert len(goal.tasks) == 0
        assert orch.enqueue.called

    def test_dispatch_idempotent_running_task(self):
        mgr = self._make_manager()
        orch = MagicMock()
        orch._queue = []
        mgr._autonomy_orch = orch
        goal = _make_goal()
        running_task = GoalTask(goal_id=goal.goal_id, description="running", status="running")
        goal.tasks.append(running_task)
        goal.current_task_id = running_task.task_id
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id
        mgr._next_task_preview = GoalTask(goal_id=goal.goal_id, description="next")
        mgr._current_mode = "focused"
        mgr._last_dispatch_ts = 0

        mgr._dispatch_task(time.time())
        assert mgr._dispatch_block_reason == "intent_already_running"

    def test_record_task_outcome_clears_current_task(self):
        mgr = self._make_manager()
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="test", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        mgr._registry.add(goal)

        mgr.record_task_outcome(
            goal.goal_id, task.task_id, "i_123",
            {"worked": True, "summary": "done"},
        )
        refreshed = mgr._registry.get(goal.goal_id)
        assert refreshed.current_task_id is None

    def test_cancel_current_task_marks_interrupted_and_clears_queue(self):
        mgr = self._make_manager()
        orch = MagicMock()
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="running", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        mgr._registry.add(goal)
        mgr._current_focus_id = goal.goal_id

        queued = _make_intent(id="ri_cancel", goal_id=goal.goal_id, task_id=task.task_id, status="queued")
        orch._queue = [queued]
        mgr._autonomy_orch = orch

        result = mgr.cancel_current_task(
            reason="cancelled in test",
            golden_context={"trace_id": "gw_test", "command_id": "GW_CANCEL_CURRENT_TASK", "golden_status": "executed"},
        )

        refreshed = mgr._registry.get(goal.goal_id)
        assert result["cancelled"] is True
        assert refreshed.current_task_id is None
        assert refreshed.tasks[0].status == "interrupted"
        assert refreshed.tasks[0].golden_trace_id == "gw_test"
        assert len(orch._queue) == 0


class TestDerivedWhyNotExecuting:
    def test_no_autonomy(self):
        mgr = GoalManager(registry=_temp_registry())
        assert mgr._derive_why_not_executing(None) == "no_autonomy_orchestrator"

    def test_no_focus(self):
        mgr = GoalManager(registry=_temp_registry())
        mgr._autonomy_orch = MagicMock()
        assert mgr._derive_why_not_executing(None) == "no_focused_goal"

    def test_executing(self):
        mgr = GoalManager(registry=_temp_registry())
        mgr._autonomy_orch = MagicMock()
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="x", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        mgr._next_task_preview = task
        assert mgr._derive_why_not_executing(goal) is None

    def test_no_pending_task(self):
        mgr = GoalManager(registry=_temp_registry())
        mgr._autonomy_orch = MagicMock()
        mgr._next_task_preview = None
        goal = _make_goal()
        assert mgr._derive_why_not_executing(goal) == "no_pending_task"


# ---------------------------------------------------------------------------
# P2: Alignment classification and suppression
# ---------------------------------------------------------------------------

class TestIntentAlignmentClassification:
    def _mgr_with_goal(self, tag_cluster=("emotion", "model", "fix")) -> GoalManager:
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(tag_cluster=tag_cluster)
        mgr._registry.add(goal)
        return mgr

    def test_linked(self):
        mgr = self._mgr_with_goal()
        intent = _make_intent(goal_id="some_goal")
        assert mgr.classify_intent_alignment(intent) == "linked"

    def test_adjacent(self):
        mgr = self._mgr_with_goal(tag_cluster=("emotion", "model", "fix"))
        intent = _make_intent(tag_cluster=("emotion", "model", "wav2vec"))
        assert mgr.classify_intent_alignment(intent) == "adjacent"

    def test_unrelated(self):
        mgr = self._mgr_with_goal(tag_cluster=("emotion", "model", "fix"))
        intent = _make_intent(tag_cluster=("consciousness", "meaning", "existence"))
        assert mgr.classify_intent_alignment(intent) == "unrelated"


class TestAnnotateIntent:
    def test_promotes_adjacent_to_linked(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(tag_cluster=("emotion", "model", "fix"))
        mgr._registry.add(goal)
        intent = _make_intent(tag_cluster=("emotion", "model", "wav2vec"), goal_id="")
        result = mgr.annotate_intent(intent)
        assert result.goal_id == goal.goal_id

    def test_no_promotion_for_unrelated(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(tag_cluster=("emotion", "model", "fix"))
        mgr._registry.add(goal)
        intent = _make_intent(tag_cluster=("consciousness", "meaning"), goal_id="")
        result = mgr.annotate_intent(intent)
        assert result.goal_id == ""


class TestShouldSuppress:
    def _mgr_with_stalled_goal(self) -> GoalManager:
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(progress=0.0, tag_cluster=("emotion", "model"))
        mgr._registry.add(goal)
        mgr._current_mode = "reflective"
        return mgr

    def test_suppresses_existential_unrelated(self):
        mgr = self._mgr_with_stalled_goal()
        intent = _make_intent(
            source_event="existential",
            tag_cluster=("consciousness", "meaning"),
        )
        assert mgr.should_suppress(intent) is True

    def test_does_not_suppress_linked(self):
        mgr = self._mgr_with_stalled_goal()
        intent = _make_intent(
            source_event="existential",
            goal_id="some_goal",
        )
        assert mgr.should_suppress(intent) is False

    def test_does_not_suppress_in_conversational_mode(self):
        mgr = self._mgr_with_stalled_goal()
        mgr._current_mode = "conversational"
        intent = _make_intent(source_event="existential", tag_cluster=("consciousness",))
        assert mgr.should_suppress(intent, mode="conversational") is False

    def test_does_not_suppress_when_no_stalled_goal(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(progress=0.5)
        mgr._registry.add(goal)
        intent = _make_intent(source_event="existential")
        assert mgr.should_suppress(intent) is False

    def test_does_not_suppress_non_existential(self):
        mgr = self._mgr_with_stalled_goal()
        intent = _make_intent(
            source_event="metric:tick_p95_ms",
            tag_cluster=("performance",),
        )
        assert mgr.should_suppress(intent) is False


class TestGetStalledUserGoal:
    def test_returns_stalled(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(progress=0.0)
        mgr._registry.add(goal)
        result = mgr.get_stalled_user_goal()
        assert result is not None
        assert result.goal_id == goal.goal_id

    def test_returns_none_when_progressing(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(progress=0.5)
        mgr._registry.add(goal)
        assert mgr.get_stalled_user_goal() is None

    def test_returns_none_when_not_user_goal(self):
        mgr = GoalManager(registry=_temp_registry())
        goal = _make_goal(progress=0.0, explicit_user_requested=False)
        mgr._registry.add(goal)
        assert mgr.get_stalled_user_goal() is None


# ---------------------------------------------------------------------------
# P2: Opportunity scorer goal impact / existential penalty
# ---------------------------------------------------------------------------

class TestOpportunityScorerGoalAlignment:
    def test_goal_linked_impact(self):
        from autonomy.opportunity_scorer import OpportunityScorer
        scorer = OpportunityScorer()
        intent = _make_intent(goal_id="g1")
        impact = scorer._compute_impact(intent)
        assert impact == 0.85

    def test_existential_penalty(self):
        from autonomy.opportunity_scorer import OpportunityScorer
        scorer = OpportunityScorer()
        intent = _make_intent(
            source_event="existential",
            tag_cluster=("consciousness", "meaning"),
            goal_id="",
        )
        impact = scorer._compute_impact(intent)
        assert impact == 0.15

    def test_metric_unchanged(self):
        from autonomy.opportunity_scorer import OpportunityScorer
        scorer = OpportunityScorer()
        intent = _make_intent(source_event="metric:tick_p95_ms")
        impact = scorer._compute_impact(intent)
        assert impact == 0.8


# ---------------------------------------------------------------------------
# P2: Drive curiosity dampening
# ---------------------------------------------------------------------------

class TestCuriosityDampening:
    def test_dampened_with_user_goals(self):
        from autonomy.drives import DriveSignals, _compute_curiosity_urgency
        signals = DriveSignals(novelty_events=5, system_health=0.9, active_user_goals=1)
        dampened = _compute_curiosity_urgency(signals)
        signals_no_goals = DriveSignals(novelty_events=5, system_health=0.9, active_user_goals=0)
        undampened = _compute_curiosity_urgency(signals_no_goals)
        assert dampened < undampened
        assert dampened == pytest.approx(undampened * 0.5, abs=0.01)


# ---------------------------------------------------------------------------
# P2: Planner scope/hints
# ---------------------------------------------------------------------------

class TestPlannerScopeHints:
    def test_research_task_external(self):
        planner = GoalPlanner()
        goal = _make_goal(tag_cluster=("emotion",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="research")
        intent = planner.create_intent_from_task(task, goal)
        assert intent is not None
        assert intent.scope == "external_ok"
        assert intent.source_hint == "any"

    def test_recall_task_memory(self):
        planner = GoalPlanner()
        goal = _make_goal(tag_cluster=("emotion",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="recall")
        intent = planner.create_intent_from_task(task, goal)
        assert intent is not None
        assert intent.scope == "local_only"
        assert intent.source_hint == "memory"

    def test_verify_task_introspection(self):
        planner = GoalPlanner()
        goal = _make_goal(tag_cluster=("emotion",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="verify")
        intent = planner.create_intent_from_task(task, goal)
        assert intent is not None
        assert intent.scope == "local_only"
        assert intent.source_hint == "introspection"

    def test_apply_task_codebase(self):
        planner = GoalPlanner()
        goal = _make_goal(tag_cluster=("emotion",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="apply")
        intent = planner.create_intent_from_task(task, goal)
        assert intent is not None
        assert intent.scope == "local_only"
        assert intent.source_hint == "codebase"

    def test_research_intent_carries_shadow_planner_metadata(self):
        planner = GoalPlanner()
        goal = _make_goal(title="Improve engagement quality", tag_cluster=("engagement",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="research")
        shadow_state = {
            "enabled": True,
            "active": True,
            "selected": {
                "source_event": "engagement_crossed_threshold",
                "source_facet": "user",
                "utility": 0.72,
                "goal_alignment": 1.05,
                "recommendation": "prioritize engagement response path",
            },
        }
        with patch.object(GoalPlanner, "_get_world_planner_state", return_value=shadow_state):
            intent = planner.create_intent_from_task(task, goal)

        assert intent is not None
        assert intent.shadow_planner_event == "engagement_crossed_threshold"
        assert intent.shadow_planner_utility == pytest.approx(0.72, abs=0.001)
        assert intent.shadow_planner_goal_alignment >= 1.05
        assert "planner_shadow" in intent.reason
        assert "applied=false" in intent.reason

    def test_non_research_task_ignores_shadow_planner_bridge(self):
        planner = GoalPlanner()
        goal = _make_goal(title="Improve engagement quality", tag_cluster=("engagement",))
        task = GoalTask(goal_id=goal.goal_id, description="test", task_type="verify")
        shadow_state = {
            "enabled": True,
            "active": True,
            "selected": {
                "source_event": "engagement_crossed_threshold",
                "utility": 0.72,
                "goal_alignment": 1.10,
                "recommendation": "prioritize engagement response path",
            },
        }
        with patch.object(GoalPlanner, "_get_world_planner_state", return_value=shadow_state):
            intent = planner.create_intent_from_task(task, goal)

        assert intent is not None
        assert intent.shadow_planner_event == ""
        assert intent.shadow_planner_utility == 0.0
        assert intent.shadow_planner_goal_alignment == 0.0


class TestPlannerShadowPolicyPreview:
    def test_preview_generated_when_shadow_fields_present(self):
        from autonomy.orchestrator import AutonomyOrchestrator

        intent = _make_intent(id="ri_shadow", goal_id="g1", task_id="t1", priority=0.73)
        intent.shadow_planner_event = "engagement_crossed_threshold"
        intent.shadow_planner_recommendation = "prioritize engagement response path"
        intent.shadow_planner_utility = 0.8
        intent.shadow_planner_goal_alignment = 1.2

        preview = AutonomyOrchestrator._build_shadow_policy_preview(intent)
        assert preview is not None
        assert preview["intent_id"] == "ri_shadow"
        assert preview["source_event"] == "engagement_crossed_threshold"
        assert preview["proposed_priority"] > preview["base_priority"]
        assert preview["applied"] is False
        assert preview["mode"] == "shadow_only"

    def test_preview_none_without_shadow_metadata(self):
        from autonomy.orchestrator import AutonomyOrchestrator

        intent = _make_intent(id="ri_plain", priority=0.5)
        preview = AutonomyOrchestrator._build_shadow_policy_preview(intent)
        assert preview is None


class TestAutonomyGoldenApplyGate:
    def test_goal_apply_enqueue_requires_golden_execution(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        from autonomy.research_intent import ResearchIntent

        orch = AutonomyOrchestrator(autonomy_level=2)
        intent = ResearchIntent(
            question="Apply fix for goal",
            source_event="goal:g1",
            source_hint="codebase",
            scope="external_ok",
            goal_id="g1",
            task_id="t1",
        )
        accepted = orch.enqueue(intent)
        assert accepted is False
        assert intent.status == "blocked"
        assert intent.blocked_reason == "golden_required:goal_apply"
        assert orch._last_enqueue_block_reason == "golden_required:goal_apply"

        intent2 = ResearchIntent(
            question="Apply fix for goal with golden",
            source_event="goal:g1",
            source_hint="codebase",
            goal_id="g1",
            task_id="t2",
            golden_status="executed",
            golden_command_id="GW_SELF_IMPROVE_EXECUTE_CONFIRM",
            golden_trace_id="gw_ok",
        )
        accepted2 = orch.enqueue(intent2)
        assert accepted2 is True

    def test_goal_apply_local_only_codebase_does_not_require_golden(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        from autonomy.research_intent import ResearchIntent

        orch = AutonomyOrchestrator(autonomy_level=2)
        intent = ResearchIntent(
            question="Apply fix for local metric goal",
            source_event="goal:g_local",
            source_hint="codebase",
            scope="local_only",
            goal_id="g_local",
            task_id="t_local",
        )
        accepted = orch.enqueue(intent)
        assert accepted is True
        assert orch._last_enqueue_block_reason == ""


# ---------------------------------------------------------------------------
# P3: Stop word additions
# ---------------------------------------------------------------------------

class TestResolverStopWords:
    def test_stance_words_filtered(self):
        from skills.resolver import _generate_skill_id
        result = _generate_skill_id("better serve you than to operate in the shadows")
        assert "better" not in result
        assert "serve" not in result
        assert "operate" not in result
        assert "shadows" not in result

    def test_emotional_words_filtered(self):
        from skills.resolver import _generate_skill_id
        result = _generate_skill_id("you are terrible and useless and a waste of time")
        assert "terrible" not in result
        assert "useless" not in result
        assert "waste" not in result


# ---------------------------------------------------------------------------
# P3: Discovery sanitization
# ---------------------------------------------------------------------------

class TestDiscoverySanitization:
    def test_is_actionable_rejects_emotional(self):
        from skills.discovery import is_actionable_capability_phrase
        assert not is_actionable_capability_phrase("better serve you than to operate in the shadows")
        assert not is_actionable_capability_phrase("you are terrible")
        assert not is_actionable_capability_phrase("pointless waste of time")

    def test_is_actionable_accepts_real_skill(self):
        from skills.discovery import is_actionable_capability_phrase
        assert is_actionable_capability_phrase("sing a song")
        assert is_actionable_capability_phrase("draw an image")
        assert is_actionable_capability_phrase("translate text")


# ---------------------------------------------------------------------------
# P4: Recognition state machine
# ---------------------------------------------------------------------------

class TestRecognitionStateMachine:
    def test_initial_state_absent(self):
        from perception.identity_fusion import IdentityFusion
        fusion = IdentityFusion()
        assert fusion._recognition_state == "absent"

    def test_presence_transitions_to_unknown_present(self):
        from perception.identity_fusion import IdentityFusion
        fusion = IdentityFusion()
        fusion._on_presence(present=True)
        assert fusion._recognition_state == "unknown_present"

    def test_departure_transitions_to_absent(self):
        from perception.identity_fusion import IdentityFusion
        fusion = IdentityFusion()
        fusion._on_presence(present=True)
        assert fusion._recognition_state == "unknown_present"
        fusion._on_presence(present=False)
        assert fusion._recognition_state == "absent"

    def test_evidence_accumulation(self):
        from perception.identity_fusion import IdentityFusion, _CandidateEvidence
        ev = _CandidateEvidence(name="Alice")
        now = time.time()
        ev.add(0.3, now)
        ev.add(0.3, now)
        ev.add(0.3, now)
        assert ev.signals == 3
        assert ev.effective_confidence(now) > 0

    def test_evidence_decays(self):
        from perception.identity_fusion import _CandidateEvidence
        ev = _CandidateEvidence(name="Alice")
        now = time.time()
        ev.add(0.5, now - 60)
        eff_old = ev.effective_confidence(now)
        ev2 = _CandidateEvidence(name="Bob")
        ev2.add(0.5, now)
        eff_new = ev2.effective_confidence(now)
        assert eff_old < eff_new

    def test_get_status_includes_recognition_state(self):
        from perception.identity_fusion import IdentityFusion
        fusion = IdentityFusion()
        fusion._on_presence(present=True)
        status = fusion.get_status()
        assert "recognition_state" in status
        assert status["recognition_state"] == "unknown_present"
        assert "cold_start_active" in status
        assert "tentative_name" in status


# ---------------------------------------------------------------------------
# P5: Addressee gate telemetry
# ---------------------------------------------------------------------------

class TestAddresseeGateTelemetry:
    def test_telemetry_recorded(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("Hey Jarvis, what time is it?", had_wake_word=True)
        stats = gate.get_stats()
        assert stats["total_checked"] == 1
        assert len(stats["recent_decisions"]) == 1
        entry = stats["recent_decisions"][0]
        assert entry["had_name_mention"] is True
        assert entry["had_wake_word"] is True
        assert entry["result"] == "name_mention"
        assert entry["addressed"] is True
        assert entry["was_response_generated"] is None  # not yet filled

    def test_mark_response_generated(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("Jarvis, help", had_wake_word=True)
        gate.mark_response_generated(True)
        stats = gate.get_stats()
        assert stats["recent_decisions"][-1]["was_response_generated"] is True

    def test_suppressed_entries_tracked(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        result = gate.check("I'm not talking to you", had_wake_word=True)
        assert result.suppressed is True
        stats = gate.get_stats()
        assert stats["total_suppressed"] == 1
        entry = stats["recent_decisions"][-1]
        assert entry["would_have_been_blocked"] is True

    def test_telemetry_ring_buffer_cap(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        for i in range(60):
            gate.check(f"test {i}", had_wake_word=True)
        stats = gate.get_stats()
        assert len(stats["recent_decisions"]) == 50  # capped at _TELEMETRY_SIZE


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self):
        assert _jaccard(("a", "b"), ("a", "b")) == 1.0

    def test_disjoint(self):
        assert _jaccard(("a", "b"), ("c", "d")) == 0.0

    def test_partial_overlap(self):
        j = _jaccard(("a", "b", "c"), ("b", "c", "d"))
        assert 0.4 < j < 0.6

    def test_both_empty(self):
        assert _jaccard((), ()) == 1.0

    def test_one_empty(self):
        assert _jaccard(("a",), ()) == 0.0


# ---------------------------------------------------------------------------
# Fix: Goal-task completion reconciliation (immediate callback)
# ---------------------------------------------------------------------------

class TestRecordTaskOutcomeIdempotency:
    """record_task_outcome must not double-count tasks_attempted on repeat calls."""

    def test_first_call_increments(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall context", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1", {"worked": True, "summary": "ok"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 1
        assert updated.current_task_id is None
        assert updated.tasks[0].status == "completed"

    def test_second_call_does_not_double_count(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall context", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "immediate"})
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "net_delta": 0.05, "stable": True, "summary": "delta refined"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks_attempted == 1  # not 2
        assert updated.tasks_succeeded == 1  # not 2

    def test_second_call_enriches_summary(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall context", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "immediate"})
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "delta refined"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks[0].result_summary == "delta refined"

    def test_failure_does_not_double_count(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall context", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": False, "summary": "failed"})
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": False, "summary": "failed delta"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 0

    def test_preserves_first_completed_at(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall context", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "first"})
        first_ts = reg.get(goal.goal_id).tasks[0].completed_at

        import time as _t
        _t.sleep(0.01)
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "second"})
        assert reg.get(goal.goal_id).tasks[0].completed_at == first_ts

    def test_deferred_does_not_flip_completed_to_failed(self):
        """Deferred delta call with worked=False must NOT override a completed status."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="recall", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        # Immediate close: success
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "found 7 memories"})
        assert reg.get(goal.goal_id).tasks[0].status == "completed"

        # Deferred delta: net_delta = 0 → worked=False, but must NOT flip status
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": False, "summary": "delta: inconclusive"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks[0].status == "completed"  # NOT failed
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 1

    def test_deferred_does_not_flip_failed_to_completed(self):
        """Deferred delta call with worked=True must NOT override a failed status."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        # Immediate close: execution failure
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"execution_ok": False, "worked": False,
                                 "summary": "exception"})
        assert reg.get(goal.goal_id).tasks[0].status == "failed"

        # Deferred delta: stable positive → worked=True, but must NOT flip status
        mgr.record_task_outcome(goal.goal_id, task.task_id, "int_1",
                                {"worked": True, "summary": "delta: improved"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks[0].status == "failed"  # NOT completed
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 0
        assert updated.tasks[0].goal_effect == "advanced"  # effect CAN upgrade


# ---------------------------------------------------------------------------
# Fix: Reboot reconciliation — orphaned running tasks
# ---------------------------------------------------------------------------

class TestRebootReconciliation:
    """reconcile_on_boot must transition running→interrupted and clear current_task_id."""

    def test_reconciles_running_task(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="stale research", status="running",
                        dispatched_intent_id="ri_old_session")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        count = mgr.reconcile_on_boot()
        assert count == 1
        updated = reg.get(goal.goal_id)
        assert updated.tasks[0].status == "interrupted"
        assert updated.tasks[0].result_summary == "interrupted:reboot"
        assert updated.tasks[0].completed_at is not None
        assert updated.current_task_id is None

    def test_does_not_touch_completed_tasks(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="done", status="completed")
        goal.tasks.append(task)
        reg.add(goal)

        count = mgr.reconcile_on_boot()
        assert count == 0
        assert reg.get(goal.goal_id).tasks[0].status == "completed"

    def test_does_not_touch_pending_tasks(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="queued", status="pending")
        goal.tasks.append(task)
        reg.add(goal)

        count = mgr.reconcile_on_boot()
        assert count == 0

    def test_reconciles_multiple_goals(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)

        g1 = _make_goal(title="Goal A")
        t1 = GoalTask(goal_id=g1.goal_id, description="r1", status="running")
        g1.tasks.append(t1)
        g1.current_task_id = t1.task_id
        reg.add(g1)

        g2 = _make_goal(title="Goal B")
        t2 = GoalTask(goal_id=g2.goal_id, description="r2", status="running")
        g2.tasks.append(t2)
        g2.current_task_id = t2.task_id
        reg.add(g2)

        count = mgr.reconcile_on_boot()
        assert count == 2
        assert reg.get(g1.goal_id).current_task_id is None
        assert reg.get(g2.goal_id).current_task_id is None

    def test_interrupted_treated_as_closed_by_record_outcome(self):
        """Subsequent record_task_outcome on an interrupted task should not re-increment counters."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="stale", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.reconcile_on_boot()

        # Simulate a delayed delta callback arriving after reconciliation
        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_old",
                                {"worked": True, "summary": "late delta"})
        updated = reg.get(goal.goal_id)
        assert updated.tasks_attempted == 0  # interrupted was already terminal
        assert updated.tasks_succeeded == 0


class TestNotifyGoalImmediate:
    """_notify_goal_immediate calls the goal callback for goal-linked intents."""

    def test_calls_callback_for_goal_linked(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
        orch._goal_callback = MagicMock()
        intent = _make_intent(goal_id="g1", task_id="t1", id="ri_test")
        orch._notify_goal_immediate(intent, execution_ok=True, worked=True, summary="found stuff")
        orch._goal_callback.assert_called_once_with(
            "g1", "t1", "ri_test",
            {"execution_ok": True, "worked": True, "net_delta": 0.0,
             "stable": False, "summary": "found stuff"},
        )

    def test_skips_non_goal_linked(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
        orch._goal_callback = MagicMock()
        intent = _make_intent(goal_id="", task_id="", id="ri_test")
        orch._notify_goal_immediate(intent, execution_ok=True, worked=True, summary="stuff")
        orch._goal_callback.assert_not_called()

    def test_skips_when_no_callback(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
        orch._goal_callback = None
        intent = _make_intent(goal_id="g1", task_id="t1", id="ri_test")
        orch._notify_goal_immediate(intent, execution_ok=True, worked=True, summary="stuff")


# ---------------------------------------------------------------------------
# Fix: Hard-gate eviction (no log spam)
# ---------------------------------------------------------------------------

class TestHardGateEviction:
    """Hard-gate-blocked intents must be evicted from the queue, not re-checked."""

    def test_hard_gate_removes_from_queue(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
        orch._queue = []
        orch._completed = []
        orch._governor = MagicMock()
        orch._delta_tracker = MagicMock()
        orch._drive_manager = None
        orch._integrator = MagicMock()
        orch._policy_memory = MagicMock()
        orch._calibrator = MagicMock()
        orch._intent_ledger_ids = {}
        orch._episodes = []
        orch._autonomy_level = 1
        orch._goal_callback = None

        fake_stalled = _make_goal()
        mgr = MagicMock()
        mgr.get_stalled_user_goal.return_value = fake_stalled
        mgr.classify_intent_alignment.return_value = "unrelated"
        orch._goal_manager = mgr

        intent = _make_intent(id="ri_block", source_event="drive:curiosity", status="queued")
        orch._queue.append(intent)

        try:
            from consciousness.events import EventBus
            orch._event_bus = EventBus.get_instance()
        except Exception:
            orch._event_bus = MagicMock()
        orch._emit_event = MagicMock()

        orch._process_next("reflective")

        assert intent not in orch._queue
        assert intent.status == "blocked"

    def test_soft_suppression_removes_from_queue(self):
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
        orch._queue = []
        orch._completed = []
        orch._governor = MagicMock()
        orch._delta_tracker = MagicMock()
        orch._drive_manager = None
        orch._integrator = MagicMock()
        orch._policy_memory = MagicMock()
        orch._calibrator = MagicMock()
        orch._intent_ledger_ids = {}
        orch._episodes = []
        orch._autonomy_level = 1
        orch._goal_callback = None

        mgr = MagicMock()
        mgr.get_stalled_user_goal.return_value = None
        mgr.should_suppress.return_value = True
        orch._goal_manager = mgr

        intent = _make_intent(id="ri_suppress", source_event="thought:existential", status="queued")
        orch._queue.append(intent)

        orch._emit_event = MagicMock()

        orch._process_next("reflective")

        assert intent not in orch._queue
        assert intent.status == "blocked"


# ---------------------------------------------------------------------------
# Preview invalidation — next_task_preview must not show terminal tasks
# ---------------------------------------------------------------------------

class TestPreviewInvalidation:
    """next_task_preview is cleared after a task reaches terminal status."""

    def test_preview_cleared_after_record_outcome(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research", task_type="research",
                        status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        # Simulate a stale preview pointing to this task
        mgr._next_task_preview = task

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"worked": True, "summary": "done"})

        assert mgr._next_task_preview is None

    def test_preview_not_cleared_for_different_task(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research", task_type="research",
                        status="running")
        other_preview = GoalTask(goal_id=goal.goal_id, description="next step",
                                 task_type="verify")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr._next_task_preview = other_preview

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"worked": True, "summary": "done"})

        assert mgr._next_task_preview is other_preview

    def test_tick_recomputes_preview_after_dispatch(self):
        """Tick runs preview computation both before and after dispatch."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        goal.status = "active"
        goal.promoted_at = time.time()
        reg.add(goal)

        # Tick should run _compute_task_preview twice (step 6 and step 8)
        mgr.tick("reflective")

        # After tick, preview should exist because there are pending template steps
        assert mgr._next_task_preview is not None


class TestInterruptedCounts:
    """Interrupted tasks should be retried in planner progression."""

    def test_planner_retries_interrupted_task_type(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal(kind="user_goal")
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="recall stuff",
            task_type="recall", status="interrupted",
        ))

        next_task = planner.plan_next_task(goal)
        # Interrupted recall should be retried, not advanced to research.
        assert next_task is not None
        assert next_task.task_type == "recall"

    def test_planner_counts_interrupted_in_prune(self):
        from goals.planner import GoalPlanner
        goal = _make_goal()
        for i in range(12):
            goal.tasks.append(GoalTask(
                goal_id=goal.goal_id, description=f"task {i}",
                task_type="research", status="interrupted",
                completed_at=time.time() - (12 - i),
            ))
        pruned = GoalPlanner.prune_tasks(goal)
        assert pruned > 0


# ---------------------------------------------------------------------------
# Stale reason recomputation — cleared on fresh outcome
# ---------------------------------------------------------------------------

class TestStaleReasonRecomputation:
    """stale_reason is cleared when last_task_outcome_at is fresh."""

    def test_record_outcome_clears_stale_reason(self):
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        goal.stale_reason = "No task outcome since 9999s ago"
        task = GoalTask(goal_id=goal.goal_id, description="test",
                        status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"worked": True, "summary": "ok"})

        updated = reg.get(goal.goal_id)
        assert updated.stale_reason == ""

    def test_review_clears_stale_when_fresh(self):
        from goals.review import GoalReview
        review = GoalReview()
        goal = _make_goal()
        goal.status = "active"
        goal.stale_reason = "No task outcome since 5000s ago"
        goal.last_task_outcome_at = time.time() - 10  # very recent

        review.review_goal(goal)
        assert goal.stale_reason == ""

    def test_review_sets_stale_when_old(self):
        from goals.review import GoalReview
        from goals.constants import STALE_WINDOW_S
        review = GoalReview()
        goal = _make_goal()
        goal.status = "active"
        goal.last_task_outcome_at = time.time() - STALE_WINDOW_S - 100

        review.review_goal(goal)
        assert goal.stale_reason != ""
        assert "No task outcome since" in goal.stale_reason


# ---------------------------------------------------------------------------
# Addressee follow-up semantics
# ---------------------------------------------------------------------------

class TestAddresseeFollowUpSemantics:
    """Follow-up turns should use follow_up_contextual, not not_addressed."""

    def test_follow_up_with_wake_word(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        result = gate.check("what about the other thing", is_follow_up=True,
                            had_wake_word=True)
        assert result.addressed is True
        assert result.reason == "follow_up_conversation"
        assert result.confidence == 0.80

    def test_follow_up_without_wake_word(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        result = gate.check("and also check memory", is_follow_up=True,
                            had_wake_word=False)
        assert result.addressed is True
        assert result.reason == "follow_up_contextual"
        assert result.confidence == 0.70

    def test_follow_up_negation_overrides(self):
        """Explicit negation still wins over follow-up context."""
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        result = gate.check("not talking to you", is_follow_up=True,
                            had_wake_word=True)
        assert result.addressed is False
        assert result.reason == "explicit_negation"

    def test_telemetry_uses_reason_as_result(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("hello jarvis", is_follow_up=False, had_wake_word=True)
        stats = gate.get_stats()
        assert len(stats["recent_decisions"]) == 1
        entry = stats["recent_decisions"][0]
        assert entry["result"] == "name_mention"
        assert entry["addressed"] is True


# ---------------------------------------------------------------------------
# Query builder — research tasks get domain-specific queries
# ---------------------------------------------------------------------------

class TestQueryBuilder:
    """Planner builds domain-specific queries for research tasks."""

    def test_research_task_expands_emotion_tags(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal()
        goal.title = "Fix emotional model"
        goal.tag_cluster = ("emotion", "model", "perception")
        task = GoalTask(goal_id=goal.goal_id, description="Research information needed for Fix emotional model",
                        task_type="research")

        query = planner._build_search_query(task, goal)
        assert query != task.description
        assert "emotion" in query.lower() or "affect" in query.lower()

    def test_recall_task_keeps_description(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal()
        goal.tag_cluster = ("emotion", "model")
        task = GoalTask(goal_id=goal.goal_id,
                        description="Recall relevant context for Fix emotional model",
                        task_type="recall")

        query = planner._build_search_query(task, goal)
        assert query == task.description

    def test_research_rotates_through_concepts(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal()
        goal.title = "Fix emotional model"
        goal.tag_cluster = ("emotion",)

        # No completed research tasks — should get concept at index 0
        task0 = GoalTask(goal_id=goal.goal_id, description="Research 1",
                         task_type="research")
        q0 = planner._build_search_query(task0, goal)

        # Add one completed research task — should rotate to index 1
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="Research 1",
            task_type="research", status="completed",
        ))
        task1 = GoalTask(goal_id=goal.goal_id, description="Research 2",
                         task_type="research")
        q1 = planner._build_search_query(task1, goal)

        assert q0 != q1

    def test_no_tags_uses_title_subject(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal()
        goal.title = "Fix speaker recognition"
        goal.tag_cluster = ()
        task = GoalTask(goal_id=goal.goal_id, description="Research info",
                        task_type="research")

        query = planner._build_search_query(task, goal)
        assert query == task.description  # no tags, falls through

    def test_intent_from_task_uses_expanded_query(self):
        from goals.planner import GoalPlanner
        planner = GoalPlanner()
        goal = _make_goal()
        goal.title = "Fix emotional model"
        goal.tag_cluster = ("emotion", "model", "perception")
        task = GoalTask(goal_id=goal.goal_id,
                        description="Research information needed for Fix emotional model",
                        task_type="research")

        intent = planner.create_intent_from_task(task, goal)
        assert intent is not None
        assert intent.question != task.description
        assert "emotion" in intent.question.lower() or "affect" in intent.question.lower()


# ---------------------------------------------------------------------------
# Execution status vs goal effect split
# ---------------------------------------------------------------------------

class TestGoalEffectField:
    """GoalTask.goal_effect is separate from status (execution outcome)."""

    def test_default_pending(self):
        task = GoalTask(goal_id="g1", description="test")
        assert task.goal_effect == "pending"

    def test_to_dict_includes_goal_effect(self):
        task = GoalTask(goal_id="g1", description="test", goal_effect="advanced")
        d = task.to_dict()
        assert d["goal_effect"] == "advanced"

    def test_from_dict_round_trip(self):
        task = GoalTask(goal_id="g1", description="test", goal_effect="inconclusive")
        d = task.to_dict()
        restored = GoalTask.from_dict(d)
        assert restored.goal_effect == "inconclusive"

    def test_from_dict_defaults_to_pending(self):
        d = {"goal_id": "g1", "description": "test"}
        task = GoalTask.from_dict(d)
        assert task.goal_effect == "pending"


class TestOutcomeSplit:
    """record_task_outcome separates execution status from goal effect."""

    def test_execution_ok_and_worked(self):
        """Task that ran successfully AND produced useful output."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research",
                        task_type="research", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": True,
                                 "summary": "found 3 papers"})

        updated = reg.get(goal.goal_id)
        t = updated.tasks[0]
        assert t.status == "completed"
        assert t.goal_effect == "advanced"
        assert updated.tasks_succeeded == 1

    def test_execution_ok_but_not_worked(self):
        """Task ran fine but didn't advance the goal (e.g., verify found nothing)."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="verify",
                        task_type="verify", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": False,
                                 "summary": "verified: not yet complete"})

        updated = reg.get(goal.goal_id)
        t = updated.tasks[0]
        assert t.status == "completed"  # execution succeeded
        assert t.goal_effect == "inconclusive"  # but didn't advance goal
        assert updated.tasks_succeeded == 0  # worked=False

    def test_execution_failed(self):
        """Task crashed — execution_ok=False."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research",
                        task_type="research", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": False, "worked": False,
                                 "summary": "timeout"})

        updated = reg.get(goal.goal_id)
        t = updated.tasks[0]
        assert t.status == "failed"  # execution failed
        assert t.goal_effect == "inconclusive"

    def test_regression_effect(self):
        """Negative net_delta → regressed."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="apply",
                        task_type="apply", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": False,
                                 "net_delta": -0.15, "summary": "metrics worsened"})

        updated = reg.get(goal.goal_id)
        assert updated.tasks[0].goal_effect == "regressed"

    def test_deferred_enrichment_upgrades_effect(self):
        """Deferred delta call can upgrade goal_effect from inconclusive to advanced."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research",
                        task_type="research", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        # Immediate: executed OK but no memories created
        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": False,
                                 "summary": "0 memories"})
        assert reg.get(goal.goal_id).tasks[0].goal_effect == "inconclusive"

        # Deferred delta: turns out it did improve metrics
        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": True,
                                 "net_delta": 0.08, "summary": "delta positive"})
        assert reg.get(goal.goal_id).tasks[0].goal_effect == "advanced"

    def test_deferred_cannot_downgrade_advanced(self):
        """Deferred delta cannot downgrade from advanced to inconclusive."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="research",
                        task_type="research", status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        # Immediate: success with memories
        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": True, "summary": "good"})
        assert reg.get(goal.goal_id).tasks[0].goal_effect == "advanced"

        # Deferred: worked=False, but advanced should stick
        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"execution_ok": True, "worked": False, "summary": "delta: none"})
        assert reg.get(goal.goal_id).tasks[0].goal_effect == "advanced"

    def test_legacy_no_execution_ok_defaults_true(self):
        """Backward compat: if execution_ok is missing, default to True."""
        reg = _temp_registry()
        mgr = GoalManager(registry=reg)
        goal = _make_goal()
        task = GoalTask(goal_id=goal.goal_id, description="old path",
                        status="running")
        goal.tasks.append(task)
        goal.current_task_id = task.task_id
        reg.add(goal)

        mgr.record_task_outcome(goal.goal_id, task.task_id, "ri_1",
                                {"worked": True, "summary": "old-style call"})
        assert reg.get(goal.goal_id).tasks[0].status == "completed"


class TestProgressFromGoalEffect:
    """Progress computation uses goal_effect weights, not just tasks_succeeded."""

    def test_all_advanced(self):
        from goals.review import GoalReview
        review = GoalReview()
        goal = _make_goal()
        goal.status = "active"
        for _ in range(3):
            goal.tasks.append(GoalTask(
                goal_id=goal.goal_id, description="t", status="completed",
                goal_effect="advanced",
            ))

        update = review.review_goal(goal)
        assert goal.progress + update.progress_delta == pytest.approx(1.0)

    def test_all_inconclusive(self):
        from goals.review import GoalReview
        review = GoalReview()
        goal = _make_goal()
        goal.status = "active"
        for _ in range(4):
            goal.tasks.append(GoalTask(
                goal_id=goal.goal_id, description="t", status="completed",
                goal_effect="inconclusive",
            ))

        update = review.review_goal(goal)
        expected = 0.25  # inconclusive weight
        assert goal.progress + update.progress_delta == pytest.approx(expected)

    def test_mixed_effects(self):
        from goals.review import GoalReview
        review = GoalReview()
        goal = _make_goal()
        goal.status = "active"
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="t", status="completed",
            goal_effect="advanced",
        ))
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="t", status="completed",
            goal_effect="inconclusive",
        ))

        update = review.review_goal(goal)
        expected = (1.0 + 0.25) / 2.0  # average of weights
        assert goal.progress + update.progress_delta == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Follow-up counter independence
# ---------------------------------------------------------------------------

class TestFollowUpCounterIndependence:
    """total_follow_up increments for all follow-up turns, not just follow-up branch hits."""

    def test_follow_up_with_name_mention_still_counted(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        result = gate.check("hey Jarvis what about that", is_follow_up=True,
                            had_wake_word=True)
        assert result.reason == "name_mention"  # name rule wins
        assert gate._total_follow_up == 1  # but follow-up counted

    def test_follow_up_with_negation_still_counted(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("not talking to you", is_follow_up=True, had_wake_word=True)
        assert gate._total_follow_up == 1

    def test_non_follow_up_not_counted(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("hello Jarvis", is_follow_up=False, had_wake_word=True)
        assert gate._total_follow_up == 0

    def test_multiple_follow_ups_accumulated(self):
        from perception.addressee import AddresseeGate
        gate = AddresseeGate()
        gate.check("hey Jarvis", is_follow_up=True, had_wake_word=True)
        gate.check("and also check the logs", is_follow_up=True, had_wake_word=False)
        gate.check("wait what time is it", is_follow_up=True, had_wake_word=True)
        assert gate._total_follow_up == 3


# ---------------------------------------------------------------------------
# Self-report routing — status/health queries must not route to NONE
# ---------------------------------------------------------------------------

class TestSelfReportRouting:
    """Queries about Jarvis's state/health must route to STATUS or INTROSPECTION."""

    def test_how_are_you_routes_to_status(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("how are you doing")
        assert result.tool == ToolType.STATUS

    def test_how_are_you_short_routes_to_status(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("how are you")
        assert result.tool == ToolType.STATUS

    def test_system_health_routes_to_status(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("what's your health")
        assert result.tool in (ToolType.STATUS, ToolType.INTROSPECTION)

    def test_are_you_okay_routes_to_status(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("are you okay")
        assert result.tool == ToolType.STATUS

    def test_accuracy_routes_to_introspection(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("what's your accuracy")
        assert result.tool == ToolType.INTROSPECTION

    def test_confidence_routes_to_introspection(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("what is your confidence")
        assert result.tool == ToolType.INTROSPECTION

    def test_unrelated_stays_none(self):
        from reasoning.tool_router import ToolRouter, ToolType
        router = ToolRouter()
        result = router.route("tell me a joke")
        assert result.tool == ToolType.NONE


# ---------------------------------------------------------------------------
# Affect gate — anthropomorphic feeling claims rewritten
# ---------------------------------------------------------------------------

class TestAffectGate:
    """Capability gate rewrites anthropomorphic affect claims."""

    def test_feeling_good_rewritten(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm feeling good today, David.")
        assert "feeling good" not in result
        assert "status" in result.lower() or "systems" in result.lower() or "operating" in result.lower()

    def test_feeling_happy_rewritten(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I feel great about our conversation.")
        assert "feel great" not in result
        assert "metric" in result.lower() or "stable" in result.lower()

    def test_doing_well_preserved(self):
        """'I'm doing well' is conversational filler, not an inner-experience claim."""
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm doing well, thanks for asking!")
        assert "doing well" in result.lower()

    def test_normal_text_untouched(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        original = "The weather seems nice today."
        result = gate.check_text(original)
        assert result == original

    def test_affect_rewrites_counter(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.check_text("I'm feeling great!")
        gate.check_text("I feel wonderful about this.")
        stats = gate.get_stats()
        assert stats["affect_rewrites"] == 2

    def test_system_language_rewritten(self):
        """Vague system self-assessment is rewritten to telemetry-grounded language."""
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        original = "My systems are running well."
        result = gate.check_text(original)
        assert "running well" not in result, f"Expected self-state rewrite, got: {result!r}"
        assert "subsystem" in result.lower() or "telemetry" in result.lower() or "normal" in result.lower()


# ---------------------------------------------------------------------------
# Legacy goal_effect backfill
# ---------------------------------------------------------------------------

class TestGoalEffectBackfill:
    """GoalRegistry backfills goal_effect on legacy tasks during load."""

    def test_completed_with_summary_gets_advanced(self):
        from goals.goal_registry import GoalRegistry
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="research",
            status="completed", goal_effect="pending",
            result_summary="found 3 papers",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 1
        assert goal.tasks[0].goal_effect == "advanced"

    def test_completed_without_summary_gets_inconclusive(self):
        from goals.goal_registry import GoalRegistry
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="verify",
            status="completed", goal_effect="pending",
            result_summary="",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 1
        assert goal.tasks[0].goal_effect == "inconclusive"

    def test_failed_gets_inconclusive(self):
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="research",
            status="failed", goal_effect="pending",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 1
        assert goal.tasks[0].goal_effect == "inconclusive"

    def test_interrupted_gets_inconclusive(self):
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="research",
            status="interrupted", goal_effect="pending",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 1
        assert goal.tasks[0].goal_effect == "inconclusive"

    def test_already_set_skipped(self):
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="research",
            status="completed", goal_effect="advanced",
            result_summary="good stuff",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 0

    def test_pending_running_not_touched(self):
        goal = _make_goal()
        goal.tasks.append(GoalTask(
            goal_id=goal.goal_id, description="next",
            status="pending", goal_effect="pending",
        ))
        reg = _temp_registry()
        reg._goals[goal.goal_id] = goal
        count = reg._backfill_goal_effects()
        assert count == 0


# ---------------------------------------------------------------------------
# Identity mention gate — names stripped when identity unconfirmed
# ---------------------------------------------------------------------------

class TestIdentityMentionGate:
    """CapabilityGate strips user names from output when identity is unconfirmed."""

    def test_name_stripped_when_unconfirmed(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(False)
        result = gate.gate_identity_mention("You're welcome, David.", "David")
        assert "David" not in result
        assert "welcome" in result.lower()

    def test_name_kept_when_confirmed(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(True)
        result = gate.gate_identity_mention("You're welcome, David.", "David")
        assert "David" in result

    def test_no_name_no_change(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(False)
        original = "You're welcome! How can I help?"
        result = gate.gate_identity_mention(original, "David")
        assert result == original

    def test_name_none_passthrough(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(False)
        original = "Hello there, David!"
        result = gate.gate_identity_mention(original, None)
        assert result == original

    def test_counter_increments(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(False)
        gate.gate_identity_mention("Hi David!", "David")
        gate.gate_identity_mention("Thanks David.", "David")
        stats = gate.get_stats()
        assert stats["identity_name_stripped"] == 2

    def test_case_insensitive(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.set_identity_confirmed(False)
        result = gate.gate_identity_mention("Hello DAVID, how are you?", "David")
        assert "david" not in result.lower()

    def test_stats_include_identity_fields(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        stats = gate.get_stats()
        assert "identity_name_stripped" in stats
        assert "identity_confirmed" in stats
        assert "learning_rewrites" in stats


# ---------------------------------------------------------------------------
# Learning-claim gate — broad claims rewritten to mechanism language
# ---------------------------------------------------------------------------

class TestLearningClaimGate:
    """CapabilityGate rewrites ungrounded learning claims."""

    def test_always_learning_rewritten(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm always learning from our conversations.")
        assert "always learning" not in result
        assert "log" in result.lower() or "memory" in result.lower() or "interaction" in result.lower()

    def test_every_conversation_teaches(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("Every conversation teaches me something new.")
        assert "teaches me" not in result
        assert "context" in result.lower() or "stored" in result.lower()

    def test_learning_from_conversations(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm learning from our conversations.")
        assert "learning from our conversations" not in result

    def test_growing_from_interactions(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm constantly growing from our interactions.")
        assert "constantly growing" not in result

    def test_getting_better_rewritten(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I'm getting better with every conversation.")
        assert "getting better" not in result

    def test_normal_text_untouched(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        original = "Let me search for that in my memory."
        assert gate.check_text(original) == original

    def test_learning_rewrites_counter(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        gate.check_text("I'm always learning.")
        gate.check_text("Every conversation teaches me.")
        stats = gate.get_stats()
        assert stats["learning_rewrites"] == 2


# ---------------------------------------------------------------------------
# Audio leading silence — TTS prepends silence
# ---------------------------------------------------------------------------

class TestTTSLeadingSilence:
    """TTS synthesize_b64 prepends leading silence for playback onset."""

    def test_leading_silence_in_source(self):
        """Verify the leading silence code exists in tts.py source."""
        import inspect
        from reasoning.tts import BrainTTS
        source = inspect.getsource(BrainTTS.synthesize_b64)
        assert "leading_frames" in source
        assert "_LEADING_SILENCE" in source


# ---------------------------------------------------------------------------
# Response canonicalization — spoken text == persisted reply
# ---------------------------------------------------------------------------

class TestResponseCanonicalization:
    """Verify that the conversation handler accumulates gated spoken text.

    Uses file reading to avoid importing conversation_handler (heavy deps).
    """

    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent / "conversation_handler.py").read_text()

    def test_broadcast_chunk_sync_returns_str(self):
        source = self._read_source()
        assert "def _broadcast_chunk_sync(text_str: str, tone_str: str, phase_str: str" in source
        assert "-> str:" in source

    def test_send_sentence_exists(self):
        """_send_sentence still exists as the gating/broadcast helper."""
        source = self._read_source()
        assert "async def _send_sentence(" in source

    def test_reply_from_full_reply_or_all_text(self):
        """After refactor, reply is set from full_reply (tool streams) or all_text (general)."""
        source = self._read_source()
        assert "reply = full_reply" in source or "reply = all_text" in source

    def test_broadcast_chunk_sync_returns_text(self):
        source = self._read_source()
        assert "return text_str" in source

    def test_user_requested_self_improve_uses_golden_gate(self):
        import re
        source = self._read_source()
        assert re.search(
            r"attempt_improvement\(\s*request,\s*ollama_client=ollama,\s*manual=is_dry_run,\s*dry_run=is_dry_run",
            source,
            re.S,
        )
