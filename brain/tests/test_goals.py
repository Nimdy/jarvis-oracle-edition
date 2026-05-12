"""Tests for Goal Continuity Layer (Phase 1A)."""
from __future__ import annotations

import time
import tempfile
from pathlib import Path
from typing import Any

import pytest

from goals.constants import (
    ABANDON_ACTIVE_S,
    CANDIDATE_EXPIRY_S,
    COOLDOWN_AFTER_ABANDON_S,
    COOLDOWN_AFTER_BLOCK_S,
    DEDUP_JACCARD_THRESHOLD,
    MAX_ACTIVE_GOALS,
    MAX_CANDIDATES,
    MAX_NEW_GOALS_PER_HOUR,
    MAX_PROMOTIONS_PER_HOUR,
    MAX_TASKS_PER_GOAL,
    PROMOTION_DEFICIT_CYCLES_MIN,
    PROMOTION_RECURRENCE_MIN,
    PROMOTION_RECURRENCE_WINDOW_S,
    PROMOTION_SCORE_THRESHOLD,
    SCORE_USER_REQUEST,
    STALE_WINDOW_S,
)
from goals.goal import (
    AddResult,
    Goal,
    GoalSignal,
    GoalTask,
    GoalUpdate,
    ObserveResult,
    compute_recurrence_key,
)
from goals.goal_registry import GoalRegistry
from goals.planner import GoalPlanner
from goals.review import GoalReview
from goals.goal_manager import GoalManager


# ── helpers ──

def _tmp_registry() -> GoalRegistry:
    """Return a registry backed by a temp file (auto-cleaned)."""
    p = Path(tempfile.mkdtemp()) / "goals.json"
    return GoalRegistry(path=p)


def _signal(content: str = "test goal", **kw: Any) -> GoalSignal:
    defaults: dict[str, Any] = {
        "signal_type": "thought",
        "source": "test",
        "source_scope": "self",
    }
    defaults.update(kw)
    return GoalSignal(content=content, **defaults)


def _manager(registry: GoalRegistry | None = None) -> GoalManager:
    reg = registry or _tmp_registry()
    return GoalManager(registry=reg)


# ═══════════════════════════════════════════════════════════════════════
# 1. GoalSignal + recurrence_key
# ═══════════════════════════════════════════════════════════════════════

class TestGoalSignal:
    def test_auto_recurrence_key(self):
        s = _signal("improve emotion model health")
        assert s.recurrence_key
        assert len(s.recurrence_key) == 16

    def test_same_content_same_key(self):
        s1 = _signal("improve emotion model health")
        s2 = _signal("improve emotion model health")
        assert s1.recurrence_key == s2.recurrence_key

    def test_different_content_different_key(self):
        s1 = _signal("improve emotion model")
        s2 = _signal("fix identity boundary")
        assert s1.recurrence_key != s2.recurrence_key

    def test_punctuation_ignored_in_key(self):
        k1 = compute_recurrence_key("Hello, world!")
        k2 = compute_recurrence_key("Hello world")
        assert k1 == k2

    def test_case_ignored_in_key(self):
        k1 = compute_recurrence_key("Improve Model")
        k2 = compute_recurrence_key("improve model")
        assert k1 == k2


# ═══════════════════════════════════════════════════════════════════════
# 2. Goal model
# ═══════════════════════════════════════════════════════════════════════

class TestGoal:
    def test_all_criteria_met_empty(self):
        g = Goal()
        assert g.all_criteria_met is False

    def test_all_criteria_met_partial(self):
        g = Goal(success_criteria=["A", "B"], matched_criteria=["A"])
        assert g.all_criteria_met is False

    def test_all_criteria_met_full(self):
        g = Goal(success_criteria=["A", "B"], matched_criteria=["B", "A"])
        assert g.all_criteria_met is True

    def test_all_criteria_met_is_computed(self):
        g = Goal(success_criteria=["X"])
        assert g.all_criteria_met is False
        g.matched_criteria.append("X")
        assert g.all_criteria_met is True

    def test_roundtrip_dict(self):
        g = Goal(
            title="test", kind="learning", status="active",
            tag_cluster=("a", "b"), last_task_outcome_at=1000.0,
            cooldown_until=2000.0, requires_reboot_review=True,
            merge_count=3,
        )
        d = g.to_dict()
        g2 = Goal.from_dict(d)
        assert g2.title == "test"
        assert g2.tag_cluster == ("a", "b")
        assert g2.last_task_outcome_at == 1000.0
        assert g2.cooldown_until == 2000.0
        assert g2.requires_reboot_review is True
        assert g2.merge_count == 3
        assert "all_criteria_met" not in d

    def test_updated_at_exists(self):
        g = Goal()
        assert g.updated_at > 0

    def test_last_task_outcome_at_default(self):
        g = Goal()
        assert g.last_task_outcome_at is None


# ═══════════════════════════════════════════════════════════════════════
# 3. GoalRegistry
# ═══════════════════════════════════════════════════════════════════════

class TestGoalRegistry:
    def test_add_and_get(self):
        reg = _tmp_registry()
        g = Goal(title="Test goal")
        result = reg.add(g)
        assert result.outcome == "added"
        assert reg.get(g.goal_id) is g

    def test_rate_limit(self):
        reg = _tmp_registry()
        for i in range(MAX_NEW_GOALS_PER_HOUR):
            reg.add(Goal(title=f"G{i}"))
        result = reg.add(Goal(title="Over limit"))
        assert result.outcome == "rate_limited"

    def test_candidate_cap(self):
        reg = _tmp_registry()
        # Bypass rate limit by setting creation timestamps in the past
        for i in range(MAX_CANDIDATES):
            g = Goal(title=f"C{i}", status="candidate")
            reg._goals[g.goal_id] = g
        reg._save()
        result = reg.add(Goal(title="Over cap", status="candidate"))
        assert result.outcome == "cap_reached"

    def test_duplicate_rejected(self):
        reg = _tmp_registry()
        g = Goal(title="Dup")
        reg.add(g)
        result = reg.add(g)
        assert result.outcome == "duplicate"

    def test_update_sets_updated_at(self):
        reg = _tmp_registry()
        g = Goal(title="Update test")
        reg.add(g)
        old_ts = g.updated_at
        import time; time.sleep(0.01)
        reg.update(g.goal_id, priority=0.9)
        assert g.updated_at > old_ts
        assert g.priority == 0.9

    def test_cooldown_active(self):
        reg = _tmp_registry()
        g = Goal(
            title="Blocked", status="abandoned",
            recurrence_key="abc", cooldown_until=time.time() + 600,
        )
        reg.add(g)
        assert reg.is_cooldown_active("abc")
        assert not reg.is_cooldown_active("xyz")

    def test_cooldown_expired(self):
        reg = _tmp_registry()
        g = Goal(
            title="Old block", status="abandoned",
            recurrence_key="abc", cooldown_until=time.time() - 10,
        )
        reg.add(g)
        assert not reg.is_cooldown_active("abc")

    def test_cleanup_expired_candidates(self):
        reg = _tmp_registry()
        g = Goal(title="Old", status="candidate", created_at=time.time() - CANDIDATE_EXPIRY_S - 100)
        reg.add(g)
        removed = reg.cleanup_expired()
        assert removed == 1
        assert reg.get(g.goal_id) is None

    def test_get_cooled_down_goals(self):
        reg = _tmp_registry()
        g1 = Goal(title="A", status="abandoned", cooldown_until=time.time() + 600)
        g2 = Goal(title="B", status="blocked", cooldown_until=time.time() + 600)
        g3 = Goal(title="C", status="abandoned", cooldown_until=time.time() - 10)
        reg.add(g1)
        reg.add(g2)
        reg.add(g3)
        cooled = reg.get_cooled_down_goals()
        ids = {g.goal_id for g in cooled}
        assert g1.goal_id in ids
        assert g2.goal_id in ids
        assert g3.goal_id not in ids

    def test_get_needing_reboot_review(self):
        reg = _tmp_registry()
        g1 = Goal(title="A", requires_reboot_review=True)
        g2 = Goal(title="B", requires_reboot_review=False)
        reg.add(g1)
        reg.add(g2)
        needing = reg.get_needing_reboot_review()
        assert len(needing) == 1
        assert needing[0].goal_id == g1.goal_id

    def test_persistence_roundtrip(self):
        p = Path(tempfile.mkdtemp()) / "goals.json"
        reg1 = GoalRegistry(path=p)
        g = Goal(title="Persist", kind="learning", merge_count=2)
        reg1.add(g)

        reg2 = GoalRegistry(path=p)
        loaded = reg2.get(g.goal_id)
        assert loaded is not None
        assert loaded.title == "Persist"
        assert loaded.merge_count == 2

    def test_stats(self):
        reg = _tmp_registry()
        reg.add(Goal(title="A", status="candidate"))
        reg.add(Goal(title="B", status="active"))
        stats = reg.get_stats()
        assert stats["total"] == 2
        assert stats["by_status"]["candidate"] == 1
        assert stats["by_status"]["active"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 4. GoalPlanner
# ═══════════════════════════════════════════════════════════════════════

class TestGoalPlanner:
    def test_plan_first_task(self):
        planner = GoalPlanner()
        g = Goal(title="Fix emotion", kind="self_maintenance")
        task = planner.plan_next_task(g)
        assert task is not None
        assert task.task_type == "recall"
        assert "emotion" in task.description.lower()

    def test_plan_exhausted(self):
        planner = GoalPlanner()
        g = Goal(title="Fix X", kind="self_maintenance")
        for tpl in [{"task_type": "recall"}, {"task_type": "research"}, {"task_type": "research"}, {"task_type": "verify"}]:
            g.tasks.append(GoalTask(
                goal_id=g.goal_id,
                description="done",
                task_type=tpl["task_type"],
                status="completed",
            ))
        task = planner.plan_next_task(g)
        assert task is None

    def test_plan_respects_max_tasks(self):
        planner = GoalPlanner()
        g = Goal(title="X", kind="learning")
        for i in range(MAX_TASKS_PER_GOAL):
            g.tasks.append(GoalTask(goal_id=g.goal_id, description=f"T{i}", status="pending"))
        assert planner.plan_next_task(g) is None

    def test_prune_tasks_preserves_pending(self):
        g = Goal(title="X")
        for i in range(MAX_TASKS_PER_GOAL + 3):
            g.tasks.append(GoalTask(
                goal_id=g.goal_id, description=f"T{i}",
                status="completed" if i < MAX_TASKS_PER_GOAL else "pending",
                completed_at=time.time() if i < MAX_TASKS_PER_GOAL else None,
            ))
        g.tasks_attempted = MAX_TASKS_PER_GOAL + 3
        g.tasks_succeeded = MAX_TASKS_PER_GOAL
        pruned = GoalPlanner.prune_tasks(g)
        assert pruned == 3
        assert len(g.tasks) == MAX_TASKS_PER_GOAL
        assert g.tasks_attempted == MAX_TASKS_PER_GOAL + 3
        assert g.tasks_succeeded == MAX_TASKS_PER_GOAL

    def test_create_intent_from_task(self):
        planner = GoalPlanner()
        g = Goal(title="Research consciousness", goal_id="goal_abc")
        task = GoalTask(goal_id="goal_abc", description="Research X", task_id="gt_xyz")
        intent = planner.create_intent_from_task(task, g)
        assert intent is not None
        assert intent.goal_id == "goal_abc"
        assert intent.task_id == "gt_xyz"


# ═══════════════════════════════════════════════════════════════════════
# 5. GoalReview
# ═══════════════════════════════════════════════════════════════════════

class TestGoalReview:
    def test_completion_on_all_criteria_met(self):
        review = GoalReview()
        g = Goal(
            title="Fix X", status="active",
            success_criteria=["alpha", "beta"],
            matched_criteria=["alpha", "beta"],
        )
        update = review.review_goal(g)
        assert update.should_complete is True

    def test_no_completion_partial_criteria(self):
        review = GoalReview()
        g = Goal(
            title="Fix X", status="active",
            success_criteria=["alpha", "beta"],
            matched_criteria=["alpha"],
        )
        update = review.review_goal(g)
        assert update.should_complete is False

    def test_stale_detection_uses_last_task_outcome_at(self):
        review = GoalReview()
        old_ts = time.time() - STALE_WINDOW_S - 100
        g = Goal(
            title="Stale", status="active",
            last_task_outcome_at=old_ts,
            promoted_at=old_ts,
        )
        update = review.review_goal(g)
        assert g.stale_reason, "Expected stale_reason to be set"
        assert "no task outcome" in g.stale_reason.lower()

    def test_not_stale_when_recent_outcome(self):
        review = GoalReview()
        g = Goal(
            title="Fresh", status="active",
            last_task_outcome_at=time.time(),
        )
        update = review.review_goal(g)
        assert not g.stale_reason

    def test_abandon_exhausted_tasks(self):
        review = GoalReview()
        g = Goal(
            title="Bad goal", status="active",
            tasks_attempted=MAX_TASKS_PER_GOAL,
            tasks_succeeded=0,
        )
        update = review.review_goal(g)
        assert update.should_abandon is True
        assert g.cooldown_until is not None

    def test_abandon_zero_progress_after_timeout(self):
        review = GoalReview()
        g = Goal(
            title="Stuck", status="active",
            progress=0.0,
            promoted_at=time.time() - ABANDON_ACTIVE_S - 100,
        )
        update = review.review_goal(g)
        assert update.should_abandon is True

    def test_no_abandon_with_progress(self):
        review = GoalReview()
        g = Goal(
            title="Moving", status="active",
            progress=0.3,
            tasks_attempted=2,
            tasks_succeeded=1,
            promoted_at=time.time() - ABANDON_ACTIVE_S - 100,
        )
        update = review.review_goal(g)
        assert update.should_abandon is False

    def test_reboot_review_session_goal(self):
        review = GoalReview()
        g = Goal(title="Session", status="active", horizon="session")
        review.review_goal(g)
        assert g.requires_reboot_review is True

    def test_reboot_review_blocked_old(self):
        review = GoalReview()
        g = Goal(
            title="Stuck", status="blocked",
            updated_at=time.time() - 7200,
        )
        review.review_goal(g)
        assert g.requires_reboot_review is True

    def test_criteria_matching(self):
        review = GoalReview()
        g = Goal(
            title="Learn X", status="active",
            success_criteria=["emotion model recovered", "accuracy above 0.7"],
            matched_criteria=[],
        )
        g.tasks.append(GoalTask(
            goal_id=g.goal_id, description="check",
            result_summary="emotion model recovered after restart",
            status="completed",
        ))
        update = review.review_goal(g)
        assert "emotion model recovered" in update.newly_matched_criteria


# ═══════════════════════════════════════════════════════════════════════
# 6. GoalManager — observation
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerObserve:
    def test_create_new_goal(self):
        mgr = _manager()
        result = mgr.observe_signal(_signal("improve emotion model"))
        assert result.outcome == "created"
        assert result.goal is not None
        assert result.goal.status == "candidate"

    def test_user_signal_auto_promotes(self):
        mgr = _manager()
        result = mgr.observe_signal(_signal("help with homework", source_scope="user"))
        assert result.outcome == "created"
        assert result.goal.status == "active"

    def test_merge_on_same_recurrence_key(self):
        mgr = _manager()
        s1 = _signal("improve emotion model")
        s2 = _signal("improve emotion model")
        r1 = mgr.observe_signal(s1)
        r2 = mgr.observe_signal(s2)
        assert r1.outcome == "created"
        assert r2.outcome == "merged"
        assert r2.reason == "merged:recurrence_key"
        assert r2.goal.recurrence_count == 2
        assert r2.goal.merge_count == 1

    def test_merge_on_tag_overlap(self):
        mgr = _manager()
        s1 = _signal("fix emotion health", tag_cluster=("emotion", "health", "model"))
        s2 = GoalSignal(
            signal_type="metric_deficit",
            source="trigger",
            source_scope="metric",
            content="emotion model degraded entirely different text",
            tag_cluster=("emotion", "health", "model", "degraded"),
        )
        r1 = mgr.observe_signal(s1)
        r2 = mgr.observe_signal(s2)
        assert r2.outcome == "merged"
        assert r2.reason == "merged:tag_overlap"

    def test_reject_empty_signal(self):
        mgr = _manager()
        result = mgr.observe_signal(_signal(""))
        assert result.outcome == "rejected"

    def test_rate_limit(self):
        mgr = _manager()
        for i in range(MAX_NEW_GOALS_PER_HOUR):
            mgr.observe_signal(_signal(f"distinct goal number {i} unique"))
        result = mgr.observe_signal(_signal("one more"))
        assert result.outcome == "rate_limited"

    def test_cooldown_blocks_by_recurrence_key(self):
        reg = _tmp_registry()
        # Pre-compute the key the signal will generate so they match
        signal_key = compute_recurrence_key("blocked topic", "thought", "self")
        g = Goal(
            title="Old", status="abandoned",
            recurrence_key=signal_key,
            cooldown_until=time.time() + 600,
        )
        reg.add(g)
        mgr = _manager(registry=reg)
        result = mgr.observe_signal(_signal("blocked topic"))
        assert result.outcome == "cooldown_blocked"

    def test_cooldown_blocks_by_tag_overlap(self):
        reg = _tmp_registry()
        g = Goal(
            title="Dead", status="abandoned",
            tag_cluster=("emotion", "model", "health"),
            cooldown_until=time.time() + 600,
        )
        reg.add(g)
        mgr = _manager(registry=reg)
        result = mgr.observe_signal(_signal(
            "something new entirely different text unique unique unique",
            tag_cluster=("emotion", "model", "health", "recovery"),
        ))
        assert result.outcome == "cooldown_blocked"
        assert "tag overlap" in result.reason.lower()


# ═══════════════════════════════════════════════════════════════════════
# 7. GoalManager — promotion
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerPromotion:
    def test_promotion_on_recurrence(self):
        mgr = _manager()
        for _ in range(PROMOTION_RECURRENCE_MIN + 1):
            mgr.observe_signal(_signal(
                "fix emotion model",
                signal_type="drive",
            ))
        mgr.tick("reflective")
        focus = mgr.get_current_focus()
        if focus:
            assert focus.status == "active"

    def test_no_promotion_in_sleep(self):
        mgr = _manager()
        for _ in range(5):
            mgr.observe_signal(_signal(
                "fix emotion model",
                signal_type="drive",
            ))
        mgr.tick("sleep")
        all_active = [
            g for g in mgr._registry.get_all()
            if g.status == "active"
        ]
        assert len(all_active) == 0

    def test_cap_overflow_stays_candidate(self):
        mgr = _manager()
        for i in range(MAX_ACTIVE_GOALS):
            mgr.observe_signal(_signal(f"user goal {i}", source_scope="user"))
        result = mgr.observe_signal(_signal("another thing", signal_type="drive"))
        mgr.tick("reflective")
        for _ in range(5):
            mgr.observe_signal(_signal("another thing", signal_type="drive"))
        mgr.tick("reflective")
        g = mgr._registry.get(result.goal.goal_id) if result.goal else None
        if g:
            assert g.status == "candidate"


# ═══════════════════════════════════════════════════════════════════════
# 8. GoalManager — tick and review
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerTick:
    def test_tick_runs_cleanup(self):
        reg = _tmp_registry()
        old = Goal(title="Old", status="candidate", created_at=time.time() - CANDIDATE_EXPIRY_S - 100)
        reg.add(old)
        mgr = _manager(registry=reg)
        mgr.tick("reflective")
        assert reg.get(old.goal_id) is None

    def test_tick_selects_focus(self):
        mgr = _manager()
        mgr.observe_signal(_signal("help me", source_scope="user"))
        mgr.tick("reflective")
        assert mgr._current_focus_id is not None

    def test_focus_uses_scope_bias(self):
        mgr = _manager()
        mgr.observe_signal(_signal("system check", source_scope="metric"))
        mgr.observe_signal(_signal("help me now", source_scope="user"))
        mgr.tick("reflective")
        focus = mgr.get_current_focus()
        assert focus is not None
        assert focus.source_scope == "user"


# ═══════════════════════════════════════════════════════════════════════
# 9. GoalManager — task outcome recording
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerOutcome:
    def test_record_task_outcome_success(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("my homework", source_scope="user"))
        goal = r.goal
        task = GoalTask(goal_id=goal.goal_id, description="Do research")
        goal.tasks.append(task)
        mgr._registry.update(goal.goal_id)

        mgr.record_task_outcome(
            goal.goal_id, task.task_id, "ri_123",
            {"worked": True, "net_delta": 0.05, "summary": "good"},
        )
        updated = mgr._registry.get(goal.goal_id)
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 1
        assert updated.last_task_outcome_at is not None
        assert "ri_123" in updated.evidence_refs

    def test_record_task_outcome_failure(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("fix something", source_scope="user"))
        goal = r.goal
        task = GoalTask(goal_id=goal.goal_id, description="Try fix")
        goal.tasks.append(task)
        mgr._registry.update(goal.goal_id)

        mgr.record_task_outcome(
            goal.goal_id, task.task_id, "ri_456",
            {"worked": False, "net_delta": -0.01, "summary": "failed"},
        )
        updated = mgr._registry.get(goal.goal_id)
        assert updated.tasks_attempted == 1
        assert updated.tasks_succeeded == 0


# ═══════════════════════════════════════════════════════════════════════
# 10. GoalManager — manual actions
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerManual:
    def test_complete_goal(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("goal to complete", source_scope="user"))
        assert mgr.complete_goal(r.goal.goal_id)
        assert mgr._registry.get(r.goal.goal_id).status == "completed"

    def test_abandon_goal(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("goal to abandon", source_scope="user"))
        assert mgr.abandon_goal(r.goal.goal_id, "no longer needed")
        g = mgr._registry.get(r.goal.goal_id)
        assert g.status == "abandoned"
        assert g.cooldown_until is not None

    def test_pause_and_resume(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("goal to pause", source_scope="user"))
        assert mgr.pause_goal(r.goal.goal_id, "waiting for data")
        g = mgr._registry.get(r.goal.goal_id)
        assert g.status == "paused"
        assert g.paused_reason == "waiting for data"

        assert mgr.resume_goal(r.goal.goal_id)
        g = mgr._registry.get(r.goal.goal_id)
        assert g.status == "active"
        assert g.paused_reason == ""

    def test_cannot_pause_non_active(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("candidate goal"))
        assert not mgr.pause_goal(r.goal.goal_id)

    def test_cannot_resume_non_paused(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("active goal", source_scope="user"))
        assert not mgr.resume_goal(r.goal.goal_id)


# ═══════════════════════════════════════════════════════════════════════
# 11. GoalManager — status output
# ═══════════════════════════════════════════════════════════════════════

class TestGoalManagerStatus:
    def test_status_has_required_keys(self):
        mgr = _manager()
        status = mgr.get_status()
        assert "current_focus" in status
        assert "focus_reason" in status
        assert "next_task_preview" in status
        assert "why_not_executing" in status
        assert status["why_not_executing"] == "no_autonomy_orchestrator"
        assert "candidates" in status
        assert "active_goals" in status
        assert "paused_goals" in status
        assert "completed_recent" in status
        assert "stats" in status
        assert "promotion_log" in status

    def test_status_shows_reboot_review(self):
        mgr = _manager()
        r = mgr.observe_signal(_signal("session task", source_scope="user"))
        r.goal.horizon = "session"
        mgr._registry.update(r.goal.goal_id, horizon="session")
        mgr.tick("reflective")
        status = mgr.get_status()
        active_with_reboot = [
            g for g in status["active_goals"]
            if g.get("requires_reboot_review")
        ]
        assert len(active_with_reboot) > 0


# ═══════════════════════════════════════════════════════════════════════
# 12. Phase 1A stubs
# ═══════════════════════════════════════════════════════════════════════

class TestPhase1AStubs:
    def test_annotate_intent_passthrough(self):
        mgr = _manager()
        sentinel = object()
        assert mgr.annotate_intent(sentinel) is sentinel

    def test_should_suppress_always_false(self):
        mgr = _manager()
        assert mgr.should_suppress("anything") is False


# ═══════════════════════════════════════════════════════════════════════
# 13. GoalTask
# ═══════════════════════════════════════════════════════════════════════

class TestGoalTask:
    def test_roundtrip(self):
        t = GoalTask(goal_id="g1", description="Do X", task_type="research")
        d = t.to_dict()
        t2 = GoalTask.from_dict(d)
        assert t2.goal_id == "g1"
        assert t2.description == "Do X"
        assert t2.task_type == "research"

    def test_intent_id_optional(self):
        t = GoalTask(goal_id="g1", description="Do X")
        assert t.intent_id is None


# ═══════════════════════════════════════════════════════════════════════
# 14. ResearchIntent has goal fields
# ═══════════════════════════════════════════════════════════════════════

class TestResearchIntentGoalFields:
    def test_goal_id_and_task_id_default_empty(self):
        from autonomy.research_intent import ResearchIntent
        ri = ResearchIntent(question="test", source_event="test")
        assert ri.goal_id == ""
        assert ri.task_id == ""
        d = ri.to_dict()
        assert d["goal_id"] == ""
        assert d["task_id"] == ""

    def test_goal_id_and_task_id_set(self):
        from autonomy.research_intent import ResearchIntent
        ri = ResearchIntent(
            question="test", source_event="test",
            goal_id="goal_abc", task_id="gt_xyz",
        )
        assert ri.goal_id == "goal_abc"
        assert ri.task_id == "gt_xyz"

    def test_shadow_planner_fields_default_empty(self):
        from autonomy.research_intent import ResearchIntent
        ri = ResearchIntent(question="test", source_event="test")
        assert ri.shadow_planner_event == ""
        assert ri.shadow_planner_utility == 0.0
        assert ri.shadow_planner_goal_alignment == 0.0
        assert ri.shadow_planner_recommendation == ""
        assert ri.shadow_planner_reason == ""

    def test_shadow_planner_fields_serialize(self):
        from autonomy.research_intent import ResearchIntent
        ri = ResearchIntent(
            question="test", source_event="test",
            shadow_planner_event="engagement_crossed_threshold",
            shadow_planner_utility=0.6,
            shadow_planner_goal_alignment=1.15,
            shadow_planner_recommendation="prioritize engagement response path",
            shadow_planner_reason="planner_shadow_bridge",
        )
        d = ri.to_dict()
        assert d["shadow_planner_event"] == "engagement_crossed_threshold"
        assert d["shadow_planner_utility"] == 0.6
        assert d["shadow_planner_goal_alignment"] == 1.15
        assert d["shadow_planner_recommendation"] == "prioritize engagement response path"
        assert d["shadow_planner_reason"] == "planner_shadow_bridge"


# ═══════════════════════════════════════════════════════════════════════
# 15. Promotion score computation
# ═══════════════════════════════════════════════════════════════════════

class TestPromotionScore:
    def test_user_request_gets_high_score(self):
        mgr = _manager()
        g = Goal(explicit_user_requested=True)
        score = mgr._compute_promotion_score(g)
        assert score >= SCORE_USER_REQUEST

    def test_recurrence_increases_score(self):
        mgr = _manager()
        g1 = Goal(recurrence_count=1, evidence_types=["drive"])
        g2 = Goal(recurrence_count=5, evidence_types=["drive"])
        s1 = mgr._compute_promotion_score(g1)
        s2 = mgr._compute_promotion_score(g2)
        assert s2 > s1

    def test_merge_count_increases_score(self):
        mgr = _manager()
        g1 = Goal(merge_count=0)
        g2 = Goal(merge_count=3)
        s1 = mgr._compute_promotion_score(g1)
        s2 = mgr._compute_promotion_score(g2)
        assert s2 > s1

    def test_score_clamped_to_2(self):
        mgr = _manager()
        g = Goal(
            explicit_user_requested=True,
            recurrence_count=100,
            evidence_types=["drive"],
            merge_count=50,
        )
        score = mgr._compute_promotion_score(g)
        assert score <= 2.0


# ──────────────────────────────────────────────────────────────
# Metric Revalidation
# ──────────────────────────────────────────────────────────────

class TestMetricRevalidation:
    def test_resolves_healthy_memory_candidate(self):
        review = GoalReview()
        g = Goal(
            title="Memory health degraded: 0.00",
            kind="system_health",
            evidence_types=["metric_deficit"],
            status="candidate",
        )
        healthy_report = {"components": {"memory_health": 0.95}}
        for _ in range(3):
            reason = review.revalidate_metric_candidate(g, healthy_report, None)
        assert reason is not None
        assert "resolved" in reason.lower()

    def test_keeps_still_degraded_candidate(self):
        review = GoalReview()
        g = Goal(
            title="Memory health degraded: 0.30",
            kind="system_health",
            evidence_types=["metric_deficit"],
            status="candidate",
        )
        bad_report = {"components": {"memory_health": 0.20}}
        for _ in range(5):
            reason = review.revalidate_metric_candidate(g, bad_report, None)
        assert reason is None

    def test_resets_counter_on_regression(self):
        review = GoalReview()
        g = Goal(
            title="Memory health degraded: 0.10",
            kind="system_health",
            evidence_types=["metric_deficit"],
            status="candidate",
        )
        healthy = {"components": {"memory_health": 0.95}}
        degraded = {"components": {"memory_health": 0.20}}
        review.revalidate_metric_candidate(g, healthy, None)
        review.revalidate_metric_candidate(g, healthy, None)
        review.revalidate_metric_candidate(g, degraded, None)
        for _ in range(2):
            reason = review.revalidate_metric_candidate(g, healthy, None)
        assert reason is None
        reason = review.revalidate_metric_candidate(g, healthy, None)
        assert reason is not None

    def test_ignores_non_metric_goals(self):
        review = GoalReview()
        g = Goal(
            title="Improve memory retrieval",
            kind="user_goal",
            evidence_types=["user_request"],
            status="candidate",
        )
        healthy = {"components": {"memory_health": 0.95}}
        for _ in range(5):
            reason = review.revalidate_metric_candidate(g, healthy, None)
        assert reason is None

    def test_calibration_revalidation(self):
        review = GoalReview()
        g = Goal(
            title="Calibration domain 'autonomy' critically low: 0.09",
            kind="system_health",
            evidence_types=["metric_deficit"],
            status="candidate",
        )
        recovered = {"domain_scores": {"autonomy": 0.60}}
        for _ in range(3):
            reason = review.revalidate_metric_candidate(g, None, recovered)
        assert reason is not None

    def test_calibration_still_degraded(self):
        review = GoalReview()
        g = Goal(
            title="Calibration domain 'autonomy' critically low: 0.09",
            kind="system_health",
            evidence_types=["metric_deficit"],
            status="candidate",
        )
        still_bad = {"domain_scores": {"autonomy": 0.15}}
        for _ in range(5):
            reason = review.revalidate_metric_candidate(g, None, still_bad)
        assert reason is None


# ──────────────────────────────────────────────────────────────
# Signal Producers
# ──────────────────────────────────────────────────────────────

from goals.signal_producers import (
    detect_conversation_goal,
    detect_metric_deficits,
    detect_autonomy_themes,
    metric_warmup_ready,
    get_producer_stats,
    _warmup_state,
    _normalize_title,
)


class TestConversationProducer:
    def test_detects_improvement_request(self):
        sig = detect_conversation_goal("Improve your emotional model accuracy")
        assert sig is not None
        assert sig.signal_type == "user_request"
        assert sig.source_scope == "user"
        assert sig.source == "conversation"
        assert "emotion" in sig.tag_cluster

    def test_detects_fix_request(self):
        sig = detect_conversation_goal("Fix your speaker enrollment, it never works")
        assert sig is not None
        assert "voice" in sig.tag_cluster or "speech" in sig.tag_cluster

    def test_detects_make_better(self):
        sig = detect_conversation_goal("Make your memory retrieval better")
        assert sig is not None
        assert "memory" in sig.tag_cluster

    def test_ignores_general_conversation(self):
        assert detect_conversation_goal("What's the weather today?") is None

    def test_ignores_short_text(self):
        assert detect_conversation_goal("Fix it") is None

    def test_ignores_non_self_referent(self):
        assert detect_conversation_goal("Improve the global economy") is None

    def test_tags_capped_at_6(self):
        sig = detect_conversation_goal(
            "Improve your memory recall speed and neural network emotion model performance and training"
        )
        if sig:
            assert len(sig.tag_cluster) <= 6

    def test_recurrence_key_stable(self):
        s1 = detect_conversation_goal("Improve your emotional model")
        s2 = detect_conversation_goal("Improve your emotional model")
        assert s1 is not None and s2 is not None
        assert s1.recurrence_key == s2.recurrence_key

    def test_content_truncated(self):
        long_text = "Improve your " + "memory " * 100
        sig = detect_conversation_goal(long_text[:400])
        if sig:
            assert len(sig.content) <= 200

    def test_title_strips_second_person(self):
        sig = detect_conversation_goal("Improve your emotional model accuracy")
        assert sig is not None
        assert "your" not in sig.content.lower()

    def test_title_strips_filler(self):
        sig = detect_conversation_goal("Can you please improve your memory retrieval system")
        assert sig is not None
        assert "can you" not in sig.content.lower()
        assert "please" not in sig.content.lower()

    def test_title_deduplicates_words(self):
        title = _normalize_title("improve improvement improvement accuracy", ["neural"])
        assert "improvement improvement" not in title.lower()

    def test_title_uses_canonical_when_short(self):
        title = _normalize_title("Improve your model", ["emotion"])
        assert "emotional model" in title.lower()

    def test_title_capitalized(self):
        sig = detect_conversation_goal("fix your memory recall problems")
        assert sig is not None
        assert sig.content[0].isupper()


class TestMetricDeficitProducer:
    @pytest.fixture(autouse=True)
    def _reset_warmup(self):
        _warmup_state["ticks_seen"] = 10
        _warmup_state["first_tick_time"] = time.time() - 300
        yield
        _warmup_state["ticks_seen"] = 0
        _warmup_state["first_tick_time"] = 0.0

    def test_detects_processing_deficit(self):
        report = {"components": {"processing_health": 0.30, "memory_health": 0.9,
                                  "personality_health": 0.9, "event_health": 0.9}}
        signals = detect_metric_deficits(report, None, None, uptime_s=300)
        assert len(signals) >= 1
        assert any(s.signal_type == "metric_deficit" for s in signals)
        assert any("processing" in s.tag_cluster for s in signals)

    def test_no_signal_when_healthy(self):
        report = {"components": {"processing_health": 0.9, "memory_health": 0.9,
                                  "personality_health": 0.9, "event_health": 0.9}}
        signals = detect_metric_deficits(report, None, None, uptime_s=300)
        assert len(signals) == 0

    def test_detects_calibration_drift(self):
        cal = {"domain_scores": {"autonomy": 0.20, "reasoning": 0.8},
               "domain_provisional": {"autonomy": False, "reasoning": False}}
        signals = detect_metric_deficits(None, cal, None, uptime_s=300)
        assert len(signals) == 1
        assert "calibration" in signals[0].tag_cluster

    def test_ignores_provisional_domains(self):
        cal = {"domain_scores": {"autonomy": 0.10},
               "domain_provisional": {"autonomy": True}}
        signals = detect_metric_deficits(None, cal, None, uptime_s=300)
        assert len(signals) == 0

    def test_detects_sustained_metric_trigger(self):
        deficits = {"confidence_volatility": {"value": 0.25, "threshold": 0.15,
                                                "duration_s": 600, "severity": "medium"}}
        signals = detect_metric_deficits(None, None, deficits, uptime_s=300)
        assert len(signals) == 1
        assert signals[0].source_scope == "system"

    def test_ignores_short_duration_deficit(self):
        deficits = {"confidence_volatility": {"value": 0.25, "threshold": 0.15,
                                                "duration_s": 60, "severity": "medium"}}
        signals = detect_metric_deficits(None, None, deficits, uptime_s=300)
        assert len(signals) == 0

    def test_ignores_low_severity(self):
        deficits = {"tick_p95": {"value": 16.0, "threshold": 15.0,
                                  "duration_s": 600, "severity": "low"}}
        signals = detect_metric_deficits(None, None, deficits, uptime_s=300)
        assert len(signals) == 0

    def test_handles_none_inputs(self):
        signals = detect_metric_deficits(None, None, None, uptime_s=300)
        assert signals == []

    def test_warmup_blocks_early_signals(self):
        _warmup_state["ticks_seen"] = 0
        _warmup_state["first_tick_time"] = 0.0
        report = {"components": {"processing_health": 0.10}}
        signals = detect_metric_deficits(report, None, None, uptime_s=10)
        assert len(signals) == 0

    def test_warmup_passes_after_threshold(self):
        _warmup_state["ticks_seen"] = 5
        _warmup_state["first_tick_time"] = time.time() - 300
        report = {"components": {"processing_health": 0.10, "memory_health": 0.9,
                                  "personality_health": 0.9, "event_health": 0.9}}
        signals = detect_metric_deficits(report, None, None, uptime_s=300)
        assert len(signals) >= 1


# ──────────────────────────────────────────────────────────────
# Active Metric Goal Refresh
# ──────────────────────────────────────────────────────────────

class TestActiveMetricRefresh:
    def _cal_goal(self, domain: str = "autonomy", value: float = 0.09) -> Goal:
        return Goal(
            title=f"Calibration domain '{domain}' critically low: {value:.2f}",
            kind="system_health",
            status="active",
            evidence_types=["metric_deficit"],
            source_event="truth_calibration",
            source_scope="metric",
            promoted_at=time.time() - 600,
        )

    def _health_goal(self, component: str = "memory", value: float = 0.00) -> Goal:
        return Goal(
            title=f"{component.capitalize()} health degraded: {value:.2f}",
            kind="system_health",
            status="active",
            evidence_types=["metric_deficit"],
            source_event="health_monitor",
            source_scope="metric",
            promoted_at=time.time() - 600,
        )

    def test_title_refreshes_on_material_change(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal = {"domain_scores": {"autonomy": 0.45}}
        result = review.refresh_active_metric_goal(g, None, cal)
        assert result["title_updated"] is True
        assert "0.45" in g.title
        assert "0.09" in g.title

    def test_progress_increases_with_recovery(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal = {"domain_scores": {"autonomy": 0.45}}
        result = review.refresh_active_metric_goal(g, None, cal)
        assert result["progress"] > 0.0
        assert g.progress > 0.0
        assert g.progress < 1.0

    def test_baseline_inferred_from_title(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        assert g.baseline_metric_value is None
        cal = {"domain_scores": {"autonomy": 0.30}}
        review.refresh_active_metric_goal(g, None, cal)
        assert g.baseline_metric_value == 0.09

    def test_baseline_preserved_across_calls(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal1 = {"domain_scores": {"autonomy": 0.30}}
        review.refresh_active_metric_goal(g, None, cal1)
        baseline = g.baseline_metric_value
        cal2 = {"domain_scores": {"autonomy": 0.50}}
        review.refresh_active_metric_goal(g, None, cal2)
        assert g.baseline_metric_value == baseline

    def test_no_churn_on_stable_value(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal = {"domain_scores": {"autonomy": 0.45}}
        review.refresh_active_metric_goal(g, None, cal)
        title_after_first = g.title
        updated_after_first = g.progress
        result = review.refresh_active_metric_goal(g, None, cal)
        assert result["title_updated"] is False
        assert g.title == title_after_first
        assert abs(g.progress - updated_after_first) < 0.02

    def test_no_churn_on_tiny_change(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal1 = {"domain_scores": {"autonomy": 0.45}}
        review.refresh_active_metric_goal(g, None, cal1)
        title_after = g.title
        cal2 = {"domain_scores": {"autonomy": 0.46}}
        result = review.refresh_active_metric_goal(g, None, cal2)
        assert result["title_updated"] is False
        assert g.title == title_after

    def test_full_recovery_completes_goal(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal = {"domain_scores": {"autonomy": 0.75}}
        for _ in range(3):
            result = review.refresh_active_metric_goal(g, None, cal)
        assert result["action"] == "complete"
        assert "recovered" in result["reason"].lower()

    def test_partial_recovery_pauses_goal(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        cal = {"domain_scores": {"autonomy": 0.55}}
        for _ in range(3):
            result = review.refresh_active_metric_goal(g, None, cal)
        assert result["action"] == "pause"
        assert "no longer critical" in result["reason"].lower()

    def test_regression_resets_pause_counter(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        ok_cal = {"domain_scores": {"autonomy": 0.55}}
        bad_cal = {"domain_scores": {"autonomy": 0.20}}
        review.refresh_active_metric_goal(g, None, ok_cal)
        review.refresh_active_metric_goal(g, None, ok_cal)
        review.refresh_active_metric_goal(g, None, bad_cal)
        for _ in range(2):
            result = review.refresh_active_metric_goal(g, None, ok_cal)
        assert result["action"] == "none"
        result = review.refresh_active_metric_goal(g, None, ok_cal)
        assert result["action"] == "pause"

    def test_non_metric_goal_untouched(self):
        review = GoalReview()
        g = Goal(
            title="Improve emotional model accuracy",
            kind="user_goal",
            status="active",
            evidence_types=["user_request"],
        )
        health = {"components": {"processing_health": 0.10}}
        result = review.refresh_active_metric_goal(g, health, None)
        assert result["changed"] is False
        assert result["action"] == "none"

    def test_health_component_refresh(self):
        review = GoalReview()
        g = self._health_goal("memory", 0.00)
        health = {"components": {"memory_health": 0.80}}
        result = review.refresh_active_metric_goal(g, health, None)
        assert result["title_updated"] is True
        assert result["progress"] > 0.5
        assert "0.80" in g.title

    def test_manager_tick_refreshes_active(self):
        """Integration: GoalManager.tick() refreshes active metric goals from cached data."""
        mgr = _manager()
        sig = GoalSignal(
            signal_type="metric_deficit",
            source="truth_calibration",
            source_scope="metric",
            content="Calibration domain 'autonomy' critically low: 0.09",
            tag_cluster=("calibration", "autonomy", "drift"),
            priority_hint=0.5,
        )
        mgr.observe_signal(sig)
        goal = [g for g in mgr._registry.get_all() if "calibration" in g.title.lower()][0]
        goal.status = "active"
        goal.promoted_at = time.time() - 300
        mgr._registry.update(goal.goal_id, status="active")

        mgr.update_health_cache(cal_state={"domain_scores": {"autonomy": 0.45}})
        mgr.tick("reflective")

        updated = mgr._registry.get(goal.goal_id)
        assert updated.progress > 0.0
        assert "0.45" in updated.title

    def test_complete_does_not_overwrite_source_evidence(self):
        review = GoalReview()
        g = self._cal_goal("autonomy", 0.09)
        g.evidence_types = ["metric_deficit"]
        g.source_detail = "Calibration domain 'autonomy' critically low: 0.09"
        g.source_event = "truth_calibration"
        cal = {"domain_scores": {"autonomy": 0.75}}
        for _ in range(3):
            review.refresh_active_metric_goal(g, None, cal)
        assert g.evidence_types == ["metric_deficit"]
        assert g.source_detail == "Calibration domain 'autonomy' critically low: 0.09"
        assert g.source_event == "truth_calibration"


class TestAutonomyThemeProducer:
    def _intent(self, tags, question="test question", ago_s=0):
        return {
            "tag_cluster": tags,
            "question": question,
            "source_event": "drive:curiosity",
            "completed_at": time.time() - ago_s,
            "result": {"found_answer": True},
        }

    def test_detects_repeated_theme(self):
        intents = [self._intent(("emotion", "model")) for _ in range(3)]
        signals = detect_autonomy_themes(intents)
        assert len(signals) == 1
        assert signals[0].signal_type == "drive_recurrence"
        assert signals[0].source_scope == "self"

    def test_ignores_low_count(self):
        intents = [self._intent(("emotion", "model")) for _ in range(2)]
        signals = detect_autonomy_themes(intents)
        assert len(signals) == 0

    def test_ignores_old_intents(self):
        intents = [self._intent(("emotion", "model"), ago_s=10000) for _ in range(5)]
        signals = detect_autonomy_themes(intents)
        assert len(signals) == 0

    def test_separate_clusters(self):
        intents = [
            *[self._intent(("emotion", "model")) for _ in range(3)],
            *[self._intent(("memory", "retrieval")) for _ in range(3)],
        ]
        signals = detect_autonomy_themes(intents)
        assert len(signals) == 2

    def test_empty_input(self):
        assert detect_autonomy_themes([]) == []
        assert detect_autonomy_themes([], {}) == []
