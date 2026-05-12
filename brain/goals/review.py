"""Goal Continuity Layer — progress review, completion, and abandonment logic."""

from __future__ import annotations

import logging
import time
from typing import Any

from goals.constants import (
    ABANDON_ACTIVE_S,
    BLOCKED_REVIEW_THRESHOLD_S,
    COOLDOWN_AFTER_ABANDON_S,
    COOLDOWN_AFTER_BLOCK_S,
    MAX_TASKS_PER_GOAL,
    REFRESH_COMPLETE_CYCLES_REQUIRED,
    REFRESH_MATERIALITY_BAND,
    REFRESH_PAUSE_CYCLES_REQUIRED,
    REFRESH_PAUSE_THRESHOLD,
    REFRESH_RESOLUTION_THRESHOLD,
    STALE_WINDOW_S,
)
from goals.goal import Goal, GoalUpdate

logger = logging.getLogger(__name__)

_REVALIDATION_CYCLES_REQUIRED = 3


class GoalReview:
    """Reviews active goals: progress, stale detection, completion, abandonment."""

    def review_goal(
        self,
        goal: Goal,
        task_outcomes: list[Any] | None = None,
        autonomy_status: dict[str, Any] | None = None,
    ) -> GoalUpdate:
        now = time.time()
        update = GoalUpdate()

        # Progress from goal_effect distribution across terminal tasks.
        # advanced=1.0, inconclusive=0.25 (execution succeeded, goal not yet),
        # regressed=0.0, pending=0.0 (skip non-terminal).
        _EFFECT_WEIGHT = {"advanced": 1.0, "inconclusive": 0.25, "regressed": 0.0}
        terminal = [t for t in goal.tasks if t.status in ("completed", "failed", "interrupted")]
        if terminal:
            weighted = sum(_EFFECT_WEIGHT.get(t.goal_effect, 0.0) for t in terminal)
            new_progress = min(1.0, weighted / len(terminal))
            update.progress_delta = new_progress - goal.progress
        elif goal.tasks_attempted > 0:
            ratio = goal.tasks_succeeded / goal.tasks_attempted
            new_progress = min(1.0, ratio)
            update.progress_delta = new_progress - goal.progress
        else:
            update.progress_delta = 0.0

        # Criteria matching against task results
        newly_matched = self._match_criteria(goal, task_outcomes)
        update.newly_matched_criteria = newly_matched

        # Strict completion: computed property all_criteria_met
        if goal.all_criteria_met:
            update.should_complete = True
            update.reason = "All success criteria met"
            return update

        # Template-exhaustion completion: all tasks dispatched with positive progress
        # and no explicit success_criteria to gate on.
        if (not goal.success_criteria
                and goal.tasks_attempted > 0
                and goal.tasks_succeeded > 0
                and goal.progress > 0.0):
            from goals.planner import GoalPlanner
            if GoalPlanner().plan_next_task(goal) is None:
                update.should_complete = True
                update.reason = f"Template exhausted with progress={goal.progress:.2f}"
                return update

        # Abandonment checks (relaxed)
        if self._should_abandon(goal, now):
            update.should_abandon = True
            return update

        # Stale detection (no auto-abandon — only sets reason)
        if self._is_stale(goal, now):
            update.reason = f"Stale: no task outcome in {STALE_WINDOW_S}s"

        # Reboot review: clear first, then re-set if conditions still apply
        goal.requires_reboot_review = False
        self._check_reboot_review(goal, now)

        return update

    def _match_criteria(
        self,
        goal: Goal,
        task_outcomes: list[Any] | None,
    ) -> list[str]:
        """Check success_criteria against task result_summaries."""
        if not goal.success_criteria or not goal.tasks:
            return []

        result_text = " ".join(
            t.result_summary.lower() for t in goal.tasks if t.result_summary
        )
        if task_outcomes:
            for outcome in task_outcomes:
                if isinstance(outcome, dict):
                    result_text += " " + str(outcome.get("summary", "")).lower()

        newly_matched: list[str] = []
        for criterion in goal.success_criteria:
            if criterion in goal.matched_criteria:
                continue
            if criterion.lower() in result_text:
                newly_matched.append(criterion)

        return newly_matched

    def _should_abandon(self, goal: Goal, now: float) -> bool:
        # Exhausted all task slots with zero success
        if goal.tasks_attempted >= MAX_TASKS_PER_GOAL and goal.tasks_succeeded == 0:
            goal.abandoned_reason = f"Exhausted {MAX_TASKS_PER_GOAL} tasks with 0 successes"
            goal.cooldown_until = now + COOLDOWN_AFTER_ABANDON_S
            return True

        # Active goal with zero progress after ABANDON_ACTIVE_S
        if goal.status == "active" and goal.progress <= 0.0:
            age = now - (goal.promoted_at or goal.created_at)
            if age > ABANDON_ACTIVE_S:
                goal.abandoned_reason = f"Active for {age:.0f}s with zero progress"
                goal.cooldown_until = now + COOLDOWN_AFTER_ABANDON_S
                return True

        # Blocked goal that has exceeded review threshold — flag, don't abandon
        if goal.status == "blocked":
            blocked_age = now - goal.updated_at
            if blocked_age > BLOCKED_REVIEW_THRESHOLD_S and not goal.stale_reason:
                goal.stale_reason = f"Blocked for {blocked_age:.0f}s, needs review"
                goal.cooldown_until = now + COOLDOWN_AFTER_BLOCK_S

        return False

    def _is_stale(self, goal: Goal, now: float) -> bool:
        if goal.status not in ("active", "blocked"):
            return False
        last_progress = goal.last_task_outcome_at or goal.promoted_at or goal.created_at
        gap = now - last_progress
        if gap > STALE_WINDOW_S:
            goal.stale_reason = f"No task outcome since {gap:.0f}s ago"
            return True
        # Clear stale_reason when fresh outcome has arrived
        if goal.stale_reason:
            goal.stale_reason = ""
        return False

    def revalidate_metric_candidate(
        self,
        goal: Goal,
        health_report: dict[str, Any] | None,
        calibration_state: dict[str, Any] | None,
    ) -> str | None:
        """Check if a metric-deficit candidate's underlying condition is still true.

        Returns an abandonment reason if the condition has been healthy for
        N consecutive review cycles, or None if still valid.
        """
        if goal.kind != "system_health":
            return None
        if "metric_deficit" not in goal.evidence_types:
            return None

        still_degraded = False
        title_lower = goal.title.lower()

        if health_report:
            components = health_report.get("components", {})
            for comp_name, threshold in [
                ("processing_health", 0.50),
                ("memory_health", 0.45),
                ("personality_health", 0.50),
                ("event_health", 0.60),
            ]:
                if comp_name.replace("_", " ").split()[0] in title_lower:
                    if components.get(comp_name, 1.0) < threshold:
                        still_degraded = True

        if calibration_state and "calibration" in title_lower:
            domain_scores = calibration_state.get("domain_scores", {})
            for domain, score in domain_scores.items():
                if domain in title_lower and score < 0.35:
                    still_degraded = True

        if "sustained metric deficit" in title_lower:
            still_degraded = True

        if still_degraded:
            goal.revalidation_pass_count = 0
            return None

        pass_count = goal.revalidation_pass_count + 1
        goal.revalidation_pass_count = pass_count

        if pass_count >= _REVALIDATION_CYCLES_REQUIRED:
            return f"Underlying condition resolved for {pass_count} consecutive cycles"

        return None

    # ── Active metric goal refresh ──

    _HEALTH_COMPONENTS: dict[str, tuple[str, float]] = {
        "processing": ("processing_health", 0.45),
        "memory": ("memory_health", 0.40),
        "personality": ("personality_health", 0.45),
        "event": ("event_health", 0.55),
    }

    def refresh_active_metric_goal(
        self,
        goal: Goal,
        health_report: dict[str, Any] | None,
        calibration_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Refresh an active system_health goal's title, progress, and status from live data.

        Returns a dict describing what changed:
          {"changed": bool, "title_updated": bool, "progress": float,
           "action": "none"|"pause"|"complete", "reason": str}
        """
        result: dict[str, Any] = {
            "changed": False, "title_updated": False,
            "progress": goal.progress, "action": "none", "reason": "",
        }

        if goal.kind != "system_health" or goal.status != "active":
            return result

        live_value = self._extract_live_value(goal, health_report, calibration_state)
        if live_value is None:
            return result

        if goal.baseline_metric_value is None:
            goal.baseline_metric_value = self._infer_baseline(goal, live_value)
            result["changed"] = True

        goal.last_live_value = live_value

        progress = self._compute_metric_progress(
            goal.baseline_metric_value, live_value, REFRESH_RESOLUTION_THRESHOLD,
        )
        if abs(progress - goal.progress) > 0.01:
            goal.progress = progress
            result["progress"] = progress
            result["changed"] = True

        new_title = self._build_refreshed_title(goal, live_value)
        if new_title and new_title != goal.title:
            old_val = self._extract_value_from_title(goal.title)
            if old_val is None or abs(live_value - old_val) >= REFRESH_MATERIALITY_BAND:
                goal.title = new_title
                result["title_updated"] = True
                result["changed"] = True

        if live_value >= REFRESH_RESOLUTION_THRESHOLD:
            goal.refresh_resolved_cycles += 1
            goal.refresh_pause_cycles = 0
            if goal.refresh_resolved_cycles >= REFRESH_COMPLETE_CYCLES_REQUIRED:
                result["action"] = "complete"
                result["reason"] = f"Metric recovered to {live_value:.2f} for {goal.refresh_resolved_cycles} cycles"
                result["changed"] = True
        elif live_value >= REFRESH_PAUSE_THRESHOLD:
            goal.refresh_pause_cycles += 1
            goal.refresh_resolved_cycles = 0
            if goal.refresh_pause_cycles >= REFRESH_PAUSE_CYCLES_REQUIRED:
                result["action"] = "pause"
                result["reason"] = f"No longer critical ({live_value:.2f}) for {goal.refresh_pause_cycles} cycles"
                result["changed"] = True
        else:
            goal.refresh_resolved_cycles = 0
            goal.refresh_pause_cycles = 0

        return result

    def _extract_live_value(
        self,
        goal: Goal,
        health_report: dict[str, Any] | None,
        calibration_state: dict[str, Any] | None,
    ) -> float | None:
        """Extract the current live metric value relevant to this goal."""
        title_lower = goal.title.lower()

        if health_report:
            components = health_report.get("components", {})
            for keyword, (comp_key, _threshold) in self._HEALTH_COMPONENTS.items():
                if keyword in title_lower:
                    val = components.get(comp_key)
                    if val is not None:
                        return float(val)

        if calibration_state and "calibration" in title_lower:
            domain_scores = calibration_state.get("domain_scores", {})
            for domain, score in domain_scores.items():
                if domain in title_lower:
                    return float(score)

        return None

    def _infer_baseline(self, goal: Goal, current_live: float) -> float:
        """Infer the original deficit value from the goal title or use current live."""
        val = self._extract_value_from_title(goal.title)
        if val is not None:
            return val
        return min(current_live, 0.3)

    @staticmethod
    def _extract_value_from_title(title: str) -> float | None:
        """Extract a numeric value like '0.09' or '0.00' from a goal title."""
        import re
        match = re.search(r"(?::\s*|=\s*|at\s+)(\d+\.\d+)", title)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _compute_metric_progress(
        baseline: float, live: float, resolution: float,
    ) -> float:
        """Progress as fraction of recovery from baseline toward resolution threshold."""
        if resolution <= baseline:
            return 1.0 if live >= resolution else 0.0
        raw = (live - baseline) / (resolution - baseline)
        return max(0.0, min(1.0, raw))

    def _build_refreshed_title(self, goal: Goal, live_value: float) -> str | None:
        """Build an updated title reflecting the current live metric value."""
        title_lower = goal.title.lower()

        for keyword, (comp_key, _) in self._HEALTH_COMPONENTS.items():
            if keyword in title_lower:
                return f"{keyword.capitalize()} health: {live_value:.2f} (was {goal.baseline_metric_value or 0:.2f})"

        if "calibration" in title_lower:
            import re
            match = re.search(r"domain\s+'(\w+)'", goal.title)
            domain = match.group(1) if match else "unknown"
            return f"Calibration domain '{domain}': {live_value:.2f} (was {goal.baseline_metric_value or 0:.2f})"

        return None

    def _check_reboot_review(self, goal: Goal, now: float) -> None:
        if goal.horizon == "session":
            goal.requires_reboot_review = True

        if goal.status == "blocked":
            blocked_age = now - goal.updated_at
            if blocked_age > 3600:
                goal.requires_reboot_review = True

        if goal.status == "active" and goal.progress <= 0.0:
            age = now - (goal.promoted_at or goal.created_at)
            if age > 14400:
                goal.requires_reboot_review = True
