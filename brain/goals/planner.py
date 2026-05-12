"""Goal Continuity Layer — template-based task decomposition.

Preview only in Phase 1A: computes the next task but does not enqueue it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from goals.constants import MAX_TASKS_PER_GOAL
from goals.goal import Goal, GoalTask

logger = logging.getLogger(__name__)

# ── Domain-specific search query expansion ──
# Maps tag_cluster roots to academic/web search concepts.
# When a goal has tags, we expand them into proper domain queries
# instead of passing the human-facing goal title to search APIs.

_TAG_SEARCH_CONCEPTS: dict[str, list[str]] = {
    "emotion": [
        "speech emotion recognition",
        "multimodal emotion classification",
        "affect modeling conversational AI",
        "user emotional state estimation",
    ],
    "memory": [
        "neural memory retrieval ranking",
        "memory salience prediction",
        "episodic memory consolidation",
    ],
    "voice": [
        "speaker recognition embedding",
        "speech representation learning",
        "speaker verification deep learning",
    ],
    "wake": [
        "keyword spotting neural network",
        "wake word detection streaming",
    ],
    "face": [
        "face recognition embedding lightweight",
        "face identification ONNX inference",
    ],
    "tts": [
        "text to speech neural synthesis",
        "expressive speech synthesis",
    ],
    "performance": [
        "real-time inference optimization",
        "Python async latency reduction",
    ],
    "personality": [
        "personality trait modeling AI",
        "conversational agent personality",
    ],
    "learning": [
        "continual learning neural network",
        "skill acquisition autonomous agent",
    ],
    "neural": [
        "neural architecture search lightweight",
        "knowledge distillation student teacher",
    ],
    "autonomy": [
        "autonomous agent goal planning",
        "intrinsic motivation reinforcement learning",
    ],
    "calibration": [
        "confidence calibration neural network",
        "Brier score expected calibration error",
    ],
    "perception": [
        "multimodal perception fusion",
        "audio visual feature integration",
    ],
    "model": [
        "model fine-tuning few-shot",
        "transfer learning domain adaptation",
    ],
    "health": [
        "system health monitoring anomaly detection",
    ],
    "processing": [
        "real-time event processing optimization",
    ],
}

_TITLE_CONCEPT_RE = re.compile(
    r"\b(?:fix|improve|upgrade|enhance|optimize|boost|stabilize|tune|calibrate|retrain)\s+",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "self_maintenance": [
        {"task_type": "recall", "desc": "Introspect on current {title} state via system metrics"},
        {"task_type": "research", "desc": "Diagnose root cause of {title} deficit"},
        {"task_type": "research", "desc": "Research approaches to fix {title}"},
        {"task_type": "verify", "desc": "Verify improvement for {title}"},
    ],
    "learning": [
        {"task_type": "recall", "desc": "Assess existing knowledge about {title}"},
        {"task_type": "research", "desc": "Research deeper into {title}"},
        {"task_type": "apply", "desc": "Integrate findings about {title} into memory"},
        {"task_type": "verify", "desc": "Verify knowledge acquisition for {title}"},
    ],
    "user_goal": [
        {"task_type": "recall", "desc": "Recall relevant context for {title}"},
        {"task_type": "research", "desc": "Research information needed for {title}"},
        {"task_type": "apply", "desc": "Synthesize findings for {title}"},
        {"task_type": "verify", "desc": "Report and verify completion of {title}"},
    ],
    "system_health": [
        {"task_type": "recall", "desc": "Measure baseline metrics for {title}"},
        {"task_type": "research", "desc": "Identify deficit contributing to {title}"},
        {"task_type": "research", "desc": "Research remediation for {title}"},
        {"task_type": "apply", "desc": "Apply fix for {title}"},
        {"task_type": "verify", "desc": "Measure improvement for {title}"},
    ],
    "relationship": [
        {"task_type": "recall", "desc": "Recall context about {title}"},
        {"task_type": "research", "desc": "Assess relationship state for {title}"},
        {"task_type": "apply", "desc": "Synthesize approach for {title}"},
    ],
}


class GoalPlanner:
    """Deterministic template-based task planner. No LLM calls."""

    def plan_next_task(self, goal: Goal, recent_outcomes: list[Any] | None = None) -> GoalTask | None:
        """Return the next task for *goal*, or None if the sequence is exhausted."""
        template = _TEMPLATES.get(goal.kind, _TEMPLATES["learning"])
        completed_types = self._completed_task_types(goal)
        total_tasks = len(goal.tasks)

        if total_tasks >= MAX_TASKS_PER_GOAL:
            return None

        for step in template:
            step_type = step["task_type"]
            step_idx = sum(1 for ct in completed_types if ct == step_type)
            needed = sum(1 for s in template if s["task_type"] == step_type)
            if step_idx < needed:
                desc = step["desc"].format(title=goal.title or "this goal")
                task = GoalTask(
                    goal_id=goal.goal_id,
                    description=desc,
                    task_type=step_type,
                )
                return task

        return None

    _TASK_SCOPE: dict[str, str] = {
        "research": "external_ok",
        "recall": "local_only",
        "verify": "local_only",
        "apply": "local_only",
    }
    _TASK_HINT: dict[str, str] = {
        "research": "any",
        "recall": "memory",
        "verify": "introspection",
        "apply": "codebase",
    }

    def create_intent_from_task(self, task: GoalTask, goal: Goal) -> Any:
        """Convert a GoalTask to a ResearchIntent for autonomy dispatch."""
        try:
            from autonomy.research_intent import ResearchIntent

            question = self._build_search_query(task, goal)
            shadow_ctx = self._build_shadow_planner_context(goal, task)

            intent = ResearchIntent(
                question=question,
                source_event=f"goal:{goal.goal_id}",
                source_hint=self._TASK_HINT.get(task.task_type, "any"),
                priority=goal.priority,
                scope=self._TASK_SCOPE.get(task.task_type, "local_only"),
                tag_cluster=goal.tag_cluster,
                goal_id=goal.goal_id,
                task_id=task.task_id,
                golden_trace_id=task.golden_trace_id,
                golden_command_id=task.golden_command_id,
                golden_status=task.golden_status,
            )
            if shadow_ctx:
                intent.shadow_planner_event = shadow_ctx.get("source_event", "")
                intent.shadow_planner_utility = float(shadow_ctx.get("utility", 0.0) or 0.0)
                intent.shadow_planner_goal_alignment = float(
                    shadow_ctx.get("goal_alignment", 0.0) or 0.0,
                )
                intent.shadow_planner_recommendation = shadow_ctx.get("recommendation", "")
                intent.shadow_planner_reason = shadow_ctx.get("reason", "")
                # Shadow-only bridge: annotate intent; do not alter actual hint/scope.
                reason_bits = [
                    "planner_shadow",
                    f"event={intent.shadow_planner_event}",
                    f"utility={intent.shadow_planner_utility:.2f}",
                    f"align={intent.shadow_planner_goal_alignment:.2f}",
                    "applied=false",
                ]
                intent.reason = f"[{' '.join(reason_bits)}]"
            return intent
        except ImportError:
            return None

    def _build_shadow_planner_context(self, goal: Goal, task: GoalTask) -> dict[str, Any]:
        """Read world planner recommendation and project a goal-alignment hint.

        This is a shadow bridge only. It never changes dispatch scope/hint.
        """
        if task.task_type != "research":
            return {}
        planner = self._get_world_planner_state()
        if not planner.get("enabled") or not planner.get("active"):
            return {}
        selected = planner.get("selected")
        if not isinstance(selected, dict) or not selected:
            return {}

        base_alignment = float(selected.get("goal_alignment", 0.0) or 0.0)
        projected_alignment = max(
            base_alignment,
            self._estimate_goal_alignment(goal, selected),
        )
        return {
            "source_event": selected.get("source_event", ""),
            "utility": float(selected.get("utility", 0.0) or 0.0),
            "goal_alignment": projected_alignment,
            "recommendation": selected.get("recommendation", ""),
            "reason": "planner_shadow_bridge",
        }

    @staticmethod
    def _get_world_planner_state() -> dict[str, Any]:
        try:
            from consciousness.consciousness_system import _active_consciousness
            cs = _active_consciousness
            wm = getattr(cs, "_world_model", None) if cs else None
            if wm is None or not hasattr(wm, "get_state"):
                return {}
            wm_state = wm.get_state()
            if not isinstance(wm_state, dict):
                return {}
            planner = wm_state.get("planner", {})
            return planner if isinstance(planner, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _estimate_goal_alignment(goal: Goal, selected_option: dict[str, Any]) -> float:
        goal_tokens = {
            t for t in _TOKEN_RE.findall((goal.title or "").lower())
            if len(t) > 2
        }
        goal_tokens.update({t.lower() for t in goal.tag_cluster if len(t) > 2})
        if not goal_tokens:
            return 1.0

        option_text = " ".join(
            [
                str(selected_option.get("source_event", "")),
                str(selected_option.get("source_facet", "")),
                str(selected_option.get("recommendation", "")),
            ],
        ).lower()
        overlap = sum(1 for t in goal_tokens if t in option_text)
        if overlap <= 0:
            return 1.0
        # Small bounded boost used only for shadow policy previewing.
        return min(1.25, 1.0 + (0.05 * overlap))

    @staticmethod
    def _build_search_query(task: GoalTask, goal: Goal) -> str:
        """Build a domain-appropriate search query from goal metadata.

        For research tasks: expand tag_cluster into proper academic/web
        search terms instead of passing human-facing prose.
        For recall/verify/apply: keep the human-facing description since
        those go to memory/introspection/codebase tools.
        """
        if task.task_type != "research":
            return task.description

        tags = goal.tag_cluster
        if not tags:
            return task.description

        # Collect domain concepts from all matching tags
        concepts: list[str] = []
        for tag in tags:
            tag_concepts = _TAG_SEARCH_CONCEPTS.get(tag, [])
            for c in tag_concepts:
                if c not in concepts:
                    concepts.append(c)

        if not concepts:
            # No known domain mapping — extract subject from title
            subject = _TITLE_CONCEPT_RE.sub("", goal.title).strip()
            if subject:
                return subject
            return task.description

        # Pick the most relevant concept (first match is highest-priority tag)
        # Rotate through concepts based on how many research tasks already ran
        research_done = sum(
            1 for t in goal.tasks
            if t.task_type == "research" and t.status in GoalPlanner._PROGRESSION_STATUSES
        )
        idx = research_done % len(concepts)
        query = concepts[idx]

        logger.info(
            "Query expansion: '%s' -> '%s' (tags=%s, rotation=%d/%d)",
            task.description[:50], query, tags, idx, len(concepts),
        )
        return query

    @staticmethod
    def prune_tasks(goal: Goal) -> int:
        """Remove oldest completed tasks if over MAX_TASKS_PER_GOAL. Returns count pruned."""
        if len(goal.tasks) <= MAX_TASKS_PER_GOAL:
            return 0
        completed = [t for t in goal.tasks if t.status in GoalPlanner._TERMINAL_STATUSES]
        completed.sort(key=lambda t: t.completed_at or t.created_at)
        to_remove = len(goal.tasks) - MAX_TASKS_PER_GOAL
        removed = 0
        for t in completed:
            if removed >= to_remove:
                break
            goal.tasks.remove(t)
            removed += 1
        return removed

    _TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
    # Interrupted tasks are terminal for storage/review, but not for sequence
    # progression. A reboot-interrupted step should be retried, not skipped.
    _PROGRESSION_STATUSES = frozenset({"completed", "failed"})

    @staticmethod
    def _completed_task_types(goal: Goal) -> list[str]:
        return [t.task_type for t in goal.tasks if t.status in GoalPlanner._PROGRESSION_STATUSES]
