"""Goal Continuity Layer — GoalManager: core lifecycle orchestrator.

Phase 2: goals dispatch real tasks into the autonomy queue and gate
non-goal research when a user goal is stalled.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from goals.constants import (
    DEDUP_JACCARD_THRESHOLD,
    DISPATCH_COOLDOWN_S,
    MAX_NEW_GOALS_PER_HOUR,
    MAX_PROMOTIONS_PER_HOUR,
    PROMOTION_DEFICIT_CYCLES_MIN,
    PROMOTION_RECURRENCE_MIN,
    PROMOTION_RECURRENCE_WINDOW_S,
    PROMOTION_SCORE_THRESHOLD,
    SCORE_DEDUP_MERGE,
    SCORE_DRIVE_RECURRENCE,
    SCORE_METRIC_DEFICIT_CYCLE,
    SCORE_STALE_DECAY,
    SCORE_THOUGHT_CLUSTER,
    SCORE_USER_REQUEST,
    STALE_DECAY_INTERVAL_S,
    STALLED_PROGRESS_THRESHOLD,
)
from goals.goal import Goal, GoalSignal, GoalTask, ObserveResult
from goals.goal_registry import GoalRegistry
from goals.planner import GoalPlanner
from goals.review import GoalReview

logger = logging.getLogger(__name__)

_EXISTENTIAL_SOURCES = frozenset({
    "existential", "drive:curiosity", "drive:coherence", "drive:play",
})
_EXISTENTIAL_TAG_KEYWORDS = frozenset({
    "consciousness", "meaning", "philosophical", "reality",
    "existence", "agency", "identity",
})
_INTERACTIVE_MODES = frozenset({"conversational", "focused"})


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_PERMANENT_SUPPRESS_THRESHOLD = 3


class GoalManager:
    """Manages goal lifecycle: creation, evidence accumulation, promotion, review, status."""

    def __init__(
        self,
        registry: GoalRegistry | None = None,
        planner: GoalPlanner | None = None,
        review: GoalReview | None = None,
    ) -> None:
        self._registry = registry or GoalRegistry()
        self._planner = planner or GoalPlanner()
        self._review = review or GoalReview()
        self._current_focus_id: str | None = None
        self._next_task_preview: GoalTask | None = None
        self._promotion_log: list[dict[str, Any]] = []
        self._cached_health_report: dict[str, Any] | None = None
        self._cached_cal_state: dict[str, Any] | None = None
        self._autonomy_orch: Any = None
        self._last_dispatch_ts: float = 0.0
        self._dispatch_block_reason: str = ""
        self._current_mode: str = "passive"
        self._abandon_counts: dict[str, int] = {}
        self._rebuild_abandon_counts()

    def _rebuild_abandon_counts(self) -> None:
        """Scan historical abandoned goals to populate permanent suppress counters."""
        for g in self._registry.get_by_status("abandoned"):
            if g.recurrence_key:
                self._abandon_counts[g.recurrence_key] = (
                    self._abandon_counts.get(g.recurrence_key, 0) + 1
                )

    def set_autonomy_orchestrator(self, orch: Any) -> None:
        """Wire the autonomy orchestrator for goal task dispatch."""
        self._autonomy_orch = orch

    def reconcile_on_boot(self) -> int:
        """Reconcile orphaned running tasks after a reboot.

        On startup, any task with status="running" has no backing live intent
        since the autonomy queue is empty. Transition them to "interrupted"
        and clear current_task_id so dispatch can resume.

        Returns the number of tasks reconciled.
        """
        reconciled = 0
        for goal in self._registry.get_by_status("active"):
            dirty = False
            for task in goal.tasks:
                if task.status != "running":
                    continue
                task.status = "interrupted"
                task.result_summary = task.result_summary or "interrupted:reboot"
                task.completed_at = task.completed_at or time.time()
                dirty = True
                reconciled += 1
                logger.info(
                    "Reboot reconcile: task %s (goal %s) running→interrupted "
                    "(intent=%s)",
                    task.task_id, goal.goal_id, task.dispatched_intent_id,
                )
            if dirty:
                if goal.current_task_id:
                    goal.current_task_id = None
                goal.updated_at = time.time()
                self._registry.update(goal.goal_id)
        if reconciled:
            logger.info("Reboot reconciliation: %d orphaned task(s) closed", reconciled)
        return reconciled

    # ── Signal observation ──

    def observe_signal(self, signal: GoalSignal) -> ObserveResult:
        """Accept a GoalSignal and create/merge/reject a candidate goal."""
        now = time.time()

        if not signal.content.strip():
            return ObserveResult(outcome="rejected", reason="Empty signal content")

        if (
            signal.recurrence_key
            and signal.source_scope != "user"
            and self._abandon_counts.get(signal.recurrence_key, 0) >= _PERMANENT_SUPPRESS_THRESHOLD
        ):
            return ObserveResult(
                outcome="rejected",
                reason=f"Permanently suppressed: abandoned {self._abandon_counts[signal.recurrence_key]} times",
            )

        # Cooldown check: exact recurrence_key OR tag overlap with recent abandoned/blocked
        if self._registry.is_cooldown_active(signal.recurrence_key, now):
            return ObserveResult(outcome="cooldown_blocked", reason="Cooldown active for this signal pattern")
        if signal.tag_cluster:
            for g in self._registry.get_cooled_down_goals(now):
                if _jaccard(signal.tag_cluster, g.tag_cluster) >= DEDUP_JACCARD_THRESHOLD:
                    return ObserveResult(
                        outcome="cooldown_blocked",
                        reason=f"Tag overlap with cooled-down goal {g.goal_id}",
                    )

        # Rate limit
        if self._registry.creations_this_hour() >= MAX_NEW_GOALS_PER_HOUR:
            return ObserveResult(outcome="rate_limited", reason=f"Rate limit: {MAX_NEW_GOALS_PER_HOUR}/hr")

        # Dedup: exact recurrence_key first, then tag-cluster Jaccard second
        existing, merge_reason = self._find_matching_goal(signal)
        if existing:
            self._merge_evidence(existing, signal, now)
            return ObserveResult(outcome="merged", goal=existing, reason=merge_reason)

        # Create new candidate
        goal = Goal(
            title=signal.content[:120],
            kind=self._infer_kind(signal),
            status="candidate",
            priority=signal.priority_hint,
            tag_cluster=signal.tag_cluster,
            recurrence_count=1,
            last_observed_at=now,
            evidence_types=[signal.signal_type],
            explicit_user_requested=(signal.source_scope == "user"),
            source_event=signal.source,
            source_detail=signal.content,
            source_scope=signal.source_scope,
            recurrence_key=signal.recurrence_key,
            created_at=now,
            updated_at=now,
        )

        # Compute initial promotion score
        goal.promotion_score = self._compute_promotion_score(goal)

        # User-explicit goals promote immediately
        if signal.source_scope == "user":
            goal.status = "active"
            goal.promoted_at = now
            goal.promotion_reason = "Explicit user request"

        result = self._registry.add(goal)
        if result.outcome != "added":
            return ObserveResult(outcome="rejected", reason=result.reason)

        self._emit_event("GOAL_CREATED", goal)

        if goal.status == "active":
            self._registry.record_promotion()
            self._emit_event("GOAL_PROMOTED", goal)
            self._record_promotion(goal)

        return ObserveResult(outcome="created", goal=goal)

    # ── Tick loop ──

    def tick(self, mode: str, autonomy_status: dict[str, Any] | None = None) -> None:
        now = time.time()
        self._current_mode = mode

        # 1. Cleanup expired
        self._registry.cleanup_expired(now)

        # 2. Review active goals
        for goal in self._registry.get_by_status("active"):
            update = self._review.review_goal(goal, autonomy_status=autonomy_status)

            if update.newly_matched_criteria:
                for criterion in update.newly_matched_criteria:
                    if criterion not in goal.matched_criteria:
                        goal.matched_criteria.append(criterion)

            if update.progress_delta != 0:
                goal.progress = max(0.0, min(1.0, goal.progress + update.progress_delta))

            goal.last_reviewed_at = now

            if update.should_complete:
                self._complete_goal(goal, update.reason, now)
            elif update.should_abandon:
                self._abandon_goal(goal, goal.abandoned_reason or update.reason, now)
            elif update.should_pause:
                self._pause_goal(goal, update.reason, now)
            else:
                self._registry.update(goal.goal_id, updated_at=now, last_reviewed_at=now)

        # 3. Stale decay on candidates
        for cand in self._registry.get_candidates():
            if cand.last_observed_at and (now - cand.last_observed_at) > STALE_DECAY_INTERVAL_S:
                windows = int((now - cand.last_observed_at) / STALE_DECAY_INTERVAL_S)
                decay = windows * SCORE_STALE_DECAY
                new_score = max(0.0, cand.promotion_score + decay)
                if new_score != cand.promotion_score:
                    self._registry.update(cand.goal_id, promotion_score=new_score)

        # 3b. Revalidate metric-deficit candidates against live conditions
        self._revalidate_metric_candidates(now)

        # 3c. Refresh active metric goals against live conditions
        self._refresh_active_metric_goals(now)

        # 3d. Prune stale active goals
        self._prune_stale(now)

        # 4. Evaluate candidates for promotion
        if mode != "sleep":
            self._evaluate_promotions(now)

        # 5. Select current focus
        self._select_focus()

        # 6. Compute next task preview
        self._compute_task_preview()

        # 7. Dispatch task to autonomy (Phase 2)
        self._dispatch_task(now)

        # 8. Recompute preview after dispatch (dispatch may have consumed it)
        self._compute_task_preview()

    # ── Outcome recording ──

    def record_task_outcome(
        self,
        goal_id: str,
        task_id: str,
        intent_id: str,
        delta_result: dict[str, Any],
    ) -> None:
        """Record task completion or enrichment from deferred delta.

        Two orthogonal axes:
          - **execution status** (`status`): did the task run? completed/failed/interrupted.
            A task that ran without errors is always `completed`, even if it found
            the goal isn't done yet.
          - **goal effect** (`goal_effect`): did it advance the goal? advanced/inconclusive/regressed.
            A verify task that runs successfully but finds "not done yet" is
            completed+inconclusive, not failed.

        Lifecycle semantics:
          - The FIRST call (from _notify_goal_immediate) closes the task:
            sets terminal status, sets goal_effect, increments tasks_attempted,
            clears current_task_id, sets completed_at.
          - Subsequent calls (from deferred _process_delta_outcome) only
            ENRICH metadata (goal_effect, summary) — they never flip terminal
            execution status and never re-increment counters.
        """
        goal = self._registry.get(goal_id)
        if not goal:
            return

        now = time.time()
        execution_ok = delta_result.get("execution_ok", True)
        worked = delta_result.get("worked", False)
        summary = delta_result.get("summary", "")

        effect = "advanced" if worked else "inconclusive"
        net_delta = delta_result.get("net_delta", 0.0)
        if net_delta < -0.05:
            effect = "regressed"

        already_closed = False
        for task in goal.tasks:
            if task.task_id == task_id:
                already_closed = task.status in ("completed", "failed", "interrupted")
                if not already_closed:
                    task.status = "completed" if execution_ok else "failed"
                    task.goal_effect = effect
                    task.completed_at = now
                else:
                    # Enrichment: allow goal_effect upgrade from deferred delta
                    if effect == "advanced" and task.goal_effect != "advanced":
                        task.goal_effect = effect
                    elif effect == "regressed" and task.goal_effect == "inconclusive":
                        task.goal_effect = effect
                task.result_summary = summary or task.result_summary
                task.intent_id = intent_id or task.intent_id
                break

        if not already_closed:
            goal.tasks_attempted += 1
            if worked:
                goal.tasks_succeeded += 1
        if intent_id and intent_id not in goal.evidence_refs:
            goal.evidence_refs.append(intent_id)

        if goal.current_task_id == task_id:
            goal.current_task_id = None

        goal.last_task_outcome_at = now
        goal.updated_at = now
        goal.stale_reason = ""
        self._registry.update(goal.goal_id)
        self._emit_event("GOAL_PROGRESS_UPDATE", goal)

        GoalPlanner.prune_tasks(goal)

        # Invalidate stale preview so it doesn't advertise a terminal task
        if self._next_task_preview and self._next_task_preview.task_id == task_id:
            self._next_task_preview = None

    # ── Manual actions ──

    def complete_goal(self, goal_id: str, reason: str = "Manual completion") -> bool:
        goal = self._registry.get(goal_id)
        if not goal or goal.status in ("completed", "abandoned"):
            return False
        self._complete_goal(goal, reason, time.time())
        return True

    def abandon_goal(self, goal_id: str, reason: str = "Manual abandonment") -> bool:
        goal = self._registry.get(goal_id)
        if not goal or goal.status in ("completed", "abandoned"):
            return False
        self._abandon_goal(goal, reason, time.time())
        return True

    def pause_goal(self, goal_id: str, reason: str = "") -> bool:
        goal = self._registry.get(goal_id)
        if not goal or goal.status != "active":
            return False
        self._pause_goal(goal, reason, time.time())
        return True

    def resume_goal(self, goal_id: str) -> bool:
        goal = self._registry.get(goal_id)
        if not goal or goal.status != "paused":
            return False
        self._registry.update(
            goal.goal_id,
            status="active",
            paused_at=None,
            paused_reason="",
            updated_at=time.time(),
        )
        self._emit_event("GOAL_RESUMED", goal)
        return True

    def cancel_current_task(
        self,
        reason: str = "Cancelled by operator",
        *,
        golden_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cancel the currently running task for the focused goal.

        Returns an audit payload describing whether anything was cancelled.
        Also clears matching queued autonomy intents to avoid stale execution.
        """
        focus = self.get_current_focus()
        if not focus or focus.status != "active" or not focus.current_task_id:
            return {"cancelled": False, "reason": "no_running_goal_task"}

        now = time.time()
        task_id = focus.current_task_id
        cancelled = False

        for task in focus.tasks:
            if task.task_id != task_id:
                continue
            if task.status != "running":
                continue
            task.status = "interrupted"
            task.goal_effect = "inconclusive"
            task.completed_at = now
            task.result_summary = reason
            if golden_context:
                task.golden_trace_id = str(golden_context.get("trace_id", ""))
                task.golden_command_id = str(golden_context.get("command_id", ""))
                task.golden_status = str(golden_context.get("golden_status", "executed"))
            cancelled = True
            break

        focus.current_task_id = None
        focus.updated_at = now
        focus.last_task_outcome_at = now
        self._registry.update(focus.goal_id)
        self._emit_event("GOAL_PROGRESS_UPDATE", focus)

        cancelled_intents: list[str] = []
        if self._autonomy_orch:
            try:
                queue = getattr(self._autonomy_orch, "_queue", [])
                for intent in list(queue):
                    if getattr(intent, "goal_id", "") != focus.goal_id:
                        continue
                    if task_id and getattr(intent, "task_id", "") != task_id:
                        continue
                    intent.status = "cancelled"
                    queue.remove(intent)
                    cancelled_intents.append(getattr(intent, "id", ""))
            except Exception:
                logger.debug("Failed to clear queued intent for cancelled goal task", exc_info=True)

        self._dispatch_block_reason = "cancelled_by_operator"
        return {
            "cancelled": cancelled,
            "goal_id": focus.goal_id,
            "task_id": task_id,
            "cancelled_intent_ids": cancelled_intents,
            "reason": "cancelled" if cancelled else "task_not_running",
        }

    # ── Status ──

    def get_status(self) -> dict[str, Any]:
        focus = self._registry.get(self._current_focus_id) if self._current_focus_id else None
        active = self._registry.get_active()
        candidates = self._registry.get_candidates()

        return {
            "current_focus": focus.to_dict() if focus else None,
            "focus_reason": self._focus_reason(focus) if focus else None,
            "next_task_preview": self._next_task_preview.to_dict() if self._next_task_preview else None,
            "why_not_executing": self._derive_why_not_executing(focus),
            "dispatch_block_reason": self._dispatch_block_reason,
            "candidates": [
                {
                    "goal_id": c.goal_id,
                    "title": c.title,
                    "promotion_score": c.promotion_score,
                    "recurrence_count": c.recurrence_count,
                    "evidence_types": c.evidence_types,
                    "source_scope": c.source_scope,
                    "created_at": c.created_at,
                    "requires_reboot_review": c.requires_reboot_review,
                }
                for c in candidates[:10]
            ],
            "active_goals": [
                {
                    "goal_id": g.goal_id,
                    "title": g.title,
                    "status": g.status,
                    "priority": g.priority,
                    "progress": g.progress,
                    "matched_criteria": g.matched_criteria,
                    "blocked_reason": g.blocked_reason,
                    "stale_reason": g.stale_reason,
                    "updated_at": g.updated_at,
                    "last_task_outcome_at": g.last_task_outcome_at,
                    "requires_reboot_review": g.requires_reboot_review,
                    "baseline_metric_value": g.baseline_metric_value,
                    "last_live_value": g.last_live_value,
                    "evidence_types": g.evidence_types,
                }
                for g in active if g.status in ("active", "blocked")
            ],
            "paused_goals": [
                {
                    "goal_id": g.goal_id,
                    "title": g.title,
                    "paused_at": g.paused_at,
                    "paused_reason": g.paused_reason,
                    "requires_reboot_review": g.requires_reboot_review,
                }
                for g in active if g.status == "paused"
            ],
            "completed_recent": [
                g.to_dict() for g in sorted(
                    self._registry.get_by_status("completed"),
                    key=lambda g: g.completed_at or 0.0,
                    reverse=True,
                )[:5]
            ],
            "stats": self._registry.get_stats(),
            "promotion_log": self._promotion_log[-5:],
            "producer_health": self._get_producer_health(),
        }

    def _derive_why_not_executing(self, focus: Goal | None) -> str | None:
        """Derive the execution status — None means executing normally."""
        if not self._autonomy_orch:
            return "no_autonomy_orchestrator"
        if not focus:
            return "no_focused_goal"
        if focus.status != "active":
            return f"focus_not_active:{focus.status}"
        if not self._next_task_preview:
            return "no_pending_task"
        if focus.current_task_id:
            for t in focus.tasks:
                if t.task_id == focus.current_task_id and t.status == "running":
                    return None  # executing
            return "task_not_running"
        return self._dispatch_block_reason or None

    def get_current_focus(self) -> Goal | None:
        if self._current_focus_id:
            return self._registry.get(self._current_focus_id)
        return None

    @staticmethod
    def _get_producer_health() -> dict[str, Any]:
        try:
            from goals.signal_producers import get_producer_stats
            return get_producer_stats()
        except Exception:
            return {}

    # ── Phase 2: dispatch + alignment ──

    def _dispatch_task(self, now: float) -> None:
        """Dispatch the next goal task into the autonomy queue."""
        self._dispatch_block_reason = ""

        if not self._autonomy_orch:
            self._dispatch_block_reason = "no_autonomy"
            return

        focus = self.get_current_focus()
        if not focus or focus.status != "active":
            self._dispatch_block_reason = "no_focus"
            return

        if not self._next_task_preview:
            self._dispatch_block_reason = "no_pending_task"
            return

        if self._current_mode == "sleep":
            self._dispatch_block_reason = "mode_gated:sleep"
            return

        # Idempotency: don't dispatch if a task is already running for this goal
        if focus.current_task_id:
            for t in focus.tasks:
                if t.task_id == focus.current_task_id and t.status == "running":
                    self._dispatch_block_reason = "intent_already_running"
                    return

        # Check if a goal-linked intent is already in the autonomy queue
        try:
            queue = getattr(self._autonomy_orch, "_queue", [])
            for queued_intent in queue:
                g_id = getattr(queued_intent, "goal_id", "")
                status = getattr(queued_intent, "status", "")
                if g_id == focus.goal_id and status in ("queued", "running"):
                    self._dispatch_block_reason = "intent_in_queue"
                    return
        except Exception:
            pass

        # Rate limit
        if now - self._last_dispatch_ts < DISPATCH_COOLDOWN_S:
            self._dispatch_block_reason = "cooldown"
            return

        task = self._next_task_preview
        intent = self._planner.create_intent_from_task(task, focus)
        if not intent:
            self._dispatch_block_reason = "intent_creation_failed"
            return

        # Idempotency: check dispatched_intent_id
        if task.dispatched_intent_id:
            self._dispatch_block_reason = "task_already_dispatched"
            return

        try:
            enqueued = bool(self._autonomy_orch.enqueue(intent))
            if not enqueued:
                _enqueue_reason_raw = getattr(self._autonomy_orch, "_last_enqueue_block_reason", "")
                _enqueue_reason = _enqueue_reason_raw if isinstance(_enqueue_reason_raw, str) else ""
                self._dispatch_block_reason = _enqueue_reason or "enqueue_rejected"
                logger.info(
                    "Goal task enqueue rejected: goal=%s task=%s intent=%s reason=%s",
                    focus.goal_id, task.task_id, intent.id, self._dispatch_block_reason,
                )
                return

            task.status = "running"
            task.dispatched_intent_id = intent.id
            focus.current_task_id = task.task_id
            focus.tasks.append(task)
            focus.updated_at = now
            self._registry.update(focus.goal_id)

            self._last_dispatch_ts = now
            logger.info(
                "Goal task dispatched: goal=%s task=%s intent=%s desc='%s'",
                focus.goal_id, task.task_id, intent.id, task.description[:60],
            )
        except Exception:
            self._dispatch_block_reason = "enqueue_failed"
            logger.warning("Failed to enqueue goal task %s", task.task_id, exc_info=True)

    def get_stalled_user_goal(self) -> Goal | None:
        """Return the highest-priority active user goal with progress below threshold."""
        for goal in self._registry.get_by_status("active"):
            if goal.status != "active":
                continue
            if goal.explicit_user_requested and goal.progress < STALLED_PROGRESS_THRESHOLD:
                return goal
        return None

    def get_active_user_goal_count(self) -> int:
        """Count active goals that were explicitly user-requested."""
        return sum(
            1 for g in self._registry.get_by_status("active")
            if g.status == "active" and g.explicit_user_requested
        )

    def classify_intent_alignment(self, intent: Any) -> str:
        """Classify an intent as linked, adjacent, or unrelated to active goals."""
        goal_id = getattr(intent, "goal_id", "") or ""
        if goal_id:
            return "linked"

        tags = set(getattr(intent, "tag_cluster", ()) or ())
        for goal in self._registry.get_by_status("active"):
            if goal.status != "active" or not goal.tag_cluster:
                continue
            if _jaccard(tuple(tags), goal.tag_cluster) >= 0.3:
                return "adjacent"

        return "unrelated"

    def annotate_intent(self, intent: Any) -> Any:
        """Tag intents whose tags overlap with active goals, promoting adjacent to linked."""
        tags = set(getattr(intent, "tag_cluster", ()) or ())
        if not tags:
            return intent
        if getattr(intent, "goal_id", ""):
            return intent

        best_overlap = 0.0
        best_goal_id = ""
        for goal in self._registry.get_by_status("active"):
            if goal.status != "active" or not goal.tag_cluster:
                continue
            overlap = _jaccard(tuple(tags), goal.tag_cluster)
            if overlap >= 0.3 and overlap > best_overlap:
                best_overlap = overlap
                best_goal_id = goal.goal_id

        if best_goal_id:
            intent.goal_id = best_goal_id

        return intent

    def should_suppress(self, intent: Any, mode: str = "") -> bool:
        """Determine if a non-goal intent should be suppressed in favor of user goals."""
        mode = mode or self._current_mode

        # Interactive mode: never suppress (explicit user requests are allowed)
        if mode in _INTERACTIVE_MODES:
            return False

        stalled = self.get_stalled_user_goal()
        if not stalled:
            return False

        alignment = self.classify_intent_alignment(intent)
        if alignment in ("linked", "adjacent"):
            return False

        # Check if source is existential/philosophical
        source = getattr(intent, "source_event", "") or ""
        source_prefix = source.split(":")[0] if source else ""
        is_existential = (
            source in _EXISTENTIAL_SOURCES
            or source_prefix == "existential"
        )
        if not is_existential:
            tags = set(getattr(intent, "tag_cluster", ()) or ())
            tag_words = set()
            for t in tags:
                tag_words.update(t.lower().split(":"))
            is_existential = bool(tag_words & _EXISTENTIAL_TAG_KEYWORDS)

        if not is_existential:
            return False

        logger.info(
            "Suppressing intent (stalled user goal %s): source=%s tags=%s",
            stalled.goal_id, source, getattr(intent, "tag_cluster", ()),
        )
        return True

    # ── Internal helpers ──

    def _find_matching_goal(self, signal: GoalSignal) -> tuple[Goal | None, str]:
        """Two-pass dedup: exact recurrence_key first, then tag-cluster Jaccard."""
        all_goals = self._registry.get_candidates() + self._registry.get_active()
        # Pass 1: exact content hash
        for g in all_goals:
            if g.recurrence_key and g.recurrence_key == signal.recurrence_key:
                return g, "merged:recurrence_key"
        # Pass 2: semantic family via tag overlap
        for g in all_goals:
            if signal.tag_cluster and g.tag_cluster:
                if _jaccard(signal.tag_cluster, g.tag_cluster) >= DEDUP_JACCARD_THRESHOLD:
                    return g, "merged:tag_overlap"
        return None, "created:new_cluster"

    def _merge_evidence(self, goal: Goal, signal: GoalSignal, now: float) -> None:
        goal.recurrence_count += 1
        goal.last_observed_at = now
        goal.merge_count += 1
        if signal.signal_type not in goal.evidence_types:
            goal.evidence_types.append(signal.signal_type)
        if signal.source_scope == "user":
            goal.explicit_user_requested = True
        if signal.signal_type == "metric_deficit":
            goal.sustained_deficit_cycles += 1
            if goal.status == "paused" and goal.kind == "system_health":
                goal.status = "active"
                goal.paused_reason = ""
                goal.refresh_pause_cycles = 0
                logger.info("Resumed paused metric goal %s on re-degradation", goal.goal_id[:8])
        goal.promotion_score = self._compute_promotion_score(goal)
        goal.updated_at = now
        self._registry.update(goal.goal_id)

    def _compute_promotion_score(self, goal: Goal) -> float:
        score = 0.0
        if goal.explicit_user_requested:
            score += SCORE_USER_REQUEST

        extra_recurrences = max(0, goal.recurrence_count - 1)
        if "drive" in goal.evidence_types:
            score += extra_recurrences * SCORE_DRIVE_RECURRENCE
        thought_types = {"thought", "existential", "emergence"}
        if thought_types & set(goal.evidence_types):
            score += extra_recurrences * SCORE_THOUGHT_CLUSTER
        if "metric_deficit" in goal.evidence_types:
            score += goal.sustained_deficit_cycles * SCORE_METRIC_DEFICIT_CYCLE

        score += goal.merge_count * SCORE_DEDUP_MERGE
        return max(0.0, min(2.0, score))

    def _evaluate_promotions(self, now: float) -> None:
        if self._registry.promotions_this_hour() >= MAX_PROMOTIONS_PER_HOUR:
            return

        active_count = len([
            g for g in self._registry.get_all()
            if g.status in ("active", "paused", "blocked")
        ])

        from goals.constants import MAX_ACTIVE_GOALS
        if active_count >= MAX_ACTIVE_GOALS:
            return

        for cand in self._registry.get_candidates():
            if cand.promotion_score < PROMOTION_SCORE_THRESHOLD:
                continue

            reason = self._check_promotion_conditions(cand, now)
            if not reason:
                continue

            cand.status = "active"
            cand.promoted_at = now
            cand.promotion_reason = reason
            cand.updated_at = now
            self._registry.update(cand.goal_id)
            self._registry.record_promotion()
            self._emit_event("GOAL_PROMOTED", cand)
            self._record_promotion(cand)

            active_count += 1
            if active_count >= MAX_ACTIVE_GOALS:
                break
            if self._registry.promotions_this_hour() >= MAX_PROMOTIONS_PER_HOUR:
                break

    def _check_promotion_conditions(self, goal: Goal, now: float) -> str:
        if goal.explicit_user_requested:
            return "Explicit user request"

        if goal.recurrence_count >= PROMOTION_RECURRENCE_MIN:
            age = now - goal.created_at
            if age <= PROMOTION_RECURRENCE_WINDOW_S:
                return f"Recurrence {goal.recurrence_count}x within {age:.0f}s"

        if goal.sustained_deficit_cycles >= PROMOTION_DEFICIT_CYCLES_MIN:
            return f"Sustained deficit for {goal.sustained_deficit_cycles} cycles"

        return ""

    _SCOPE_BIAS: dict[str, float] = {
        "user": 1.0, "system": 0.6, "metric": 0.4, "self": 0.2, "derived": 0.1,
    }

    def _select_focus(self) -> None:
        active = [
            g for g in self._registry.get_active()
            if g.status == "active"
        ]
        if active:
            active.sort(
                key=lambda g: g.priority + self._SCOPE_BIAS.get(g.source_scope, 0.0),
                reverse=True,
            )
            self._current_focus_id = active[0].goal_id
        else:
            self._current_focus_id = None

    def _revalidate_metric_candidates(self, now: float) -> None:
        """Abandon metric-deficit candidates whose underlying condition resolved."""
        if not self._cached_health_report and not self._cached_cal_state:
            return

        for cand in self._registry.get_candidates():
            reason = self._review.revalidate_metric_candidate(
                cand, self._cached_health_report, self._cached_cal_state,
            )
            if reason:
                self._abandon_goal(cand, reason, now)
                logger.info("Auto-abandoned stale metric goal %s: %s", cand.goal_id, reason)

    def _refresh_active_metric_goals(self, now: float) -> None:
        """Refresh active system_health goals: update title, progress, and status from live data."""
        if not self._cached_health_report and not self._cached_cal_state:
            return

        for goal in self._registry.get_by_status("active"):
            if goal.kind != "system_health":
                continue

            result = self._review.refresh_active_metric_goal(
                goal, self._cached_health_report, self._cached_cal_state,
            )

            if result["action"] == "complete":
                self._complete_goal(goal, result["reason"], now)
                logger.info(
                    "Metric goal auto-completed %s: %s", goal.goal_id, result["reason"],
                )
            elif result["action"] == "pause":
                self._pause_goal(goal, result["reason"], now)
                logger.info(
                    "Metric goal auto-paused %s: %s", goal.goal_id, result["reason"],
                )
            elif result["changed"]:
                self._registry.update(goal.goal_id, updated_at=now)
                if result["title_updated"]:
                    logger.info(
                        "Metric goal refreshed %s: %s (progress=%.0f%%)",
                        goal.goal_id, goal.title, goal.progress * 100,
                    )

    def update_health_cache(
        self,
        health_report: dict[str, Any] | None = None,
        cal_state: dict[str, Any] | None = None,
    ) -> None:
        """Cache live health data for use during revalidation."""
        if health_report is not None:
            self._cached_health_report = health_report
        if cal_state is not None:
            self._cached_cal_state = cal_state

    def _compute_task_preview(self) -> None:
        focus = self.get_current_focus()
        if focus:
            self._next_task_preview = self._planner.plan_next_task(focus)
        else:
            self._next_task_preview = None

    def _complete_goal(self, goal: Goal, reason: str, now: float) -> None:
        self._registry.update(
            goal.goal_id,
            status="completed",
            completed_at=now,
            updated_at=now,
        )
        self._emit_event("GOAL_COMPLETED", goal)

    _STALE_GOAL_THRESHOLD_S = 7200.0  # 2 hours
    _STALE_PROGRESS_FLOOR = 0.5

    def _prune_stale(self, now: float) -> None:
        """Auto-abandon active goals that are stale >2h with progress <0.5.

        Uses last_observed_at (external signal) and goal creation time as the
        staleness anchor — NOT last_reviewed_at, which is refreshed every tick.
        """
        for goal in self._registry.get_by_status("active"):
            last_activity = max(
                goal.last_observed_at or 0.0,
                goal.created_at or 0.0,
            )
            if not last_activity:
                continue
            age = now - last_activity
            if age > self._STALE_GOAL_THRESHOLD_S and goal.progress < self._STALE_PROGRESS_FLOOR:
                reason = (
                    f"auto-pruned: stale {age / 3600:.1f}h with progress "
                    f"{goal.progress:.2f} < {self._STALE_PROGRESS_FLOOR}"
                )
                self._abandon_goal(goal, reason, now)
                logger.info("Pruned stale goal %s: %s", goal.goal_id, reason)

    def _abandon_goal(self, goal: Goal, reason: str, now: float) -> None:
        from goals.constants import COOLDOWN_AFTER_ABANDON_S
        self._registry.update(
            goal.goal_id,
            status="abandoned",
            abandoned_reason=reason,
            cooldown_until=now + COOLDOWN_AFTER_ABANDON_S,
            updated_at=now,
        )
        if goal.recurrence_key:
            self._abandon_counts[goal.recurrence_key] = (
                self._abandon_counts.get(goal.recurrence_key, 0) + 1
            )
        self._emit_event("GOAL_ABANDONED", goal)

    def _pause_goal(self, goal: Goal, reason: str, now: float) -> None:
        self._registry.update(
            goal.goal_id,
            status="paused",
            paused_at=now,
            paused_reason=reason,
            updated_at=now,
        )
        self._emit_event("GOAL_PAUSED", goal)

    def _focus_reason(self, goal: Goal) -> str:
        return f"Highest priority active goal ({goal.priority:.2f})"

    def _record_promotion(self, goal: Goal) -> None:
        entry = {
            "goal_id": goal.goal_id,
            "title": goal.title,
            "reason": goal.promotion_reason,
            "score": goal.promotion_score,
            "timestamp": time.time(),
        }
        self._promotion_log.append(entry)
        if len(self._promotion_log) > 20:
            self._promotion_log = self._promotion_log[-20:]

    @staticmethod
    def _infer_kind(signal: GoalSignal) -> str:
        if signal.source_scope == "user":
            return "user_goal"
        if signal.signal_type == "metric_deficit":
            return "system_health"
        if signal.signal_type in ("existential", "emergence"):
            return "learning"
        if signal.signal_type == "drive":
            src = signal.source.lower()
            if "mastery" in src or "truth" in src:
                return "self_maintenance"
            if "relevance" in src or "continuity" in src:
                return "relationship"
        return "learning"

    @staticmethod
    def _emit_event(event_name: str, goal: Goal) -> None:
        try:
            from consciousness import events as ev
            event_const = getattr(ev, event_name, None)
            if event_const:
                ev.event_bus.emit(event_const, goal_id=goal.goal_id, title=goal.title, status=goal.status)
        except Exception:
            pass


# ── Singleton ──

_instance: GoalManager | None = None


def get_goal_manager() -> GoalManager:
    global _instance
    if _instance is None:
        _instance = GoalManager()
    return _instance
