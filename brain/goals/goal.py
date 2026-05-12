"""Goal Continuity Layer — data model.

Goal, GoalTask, GoalSignal, GoalUpdate, and result types.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

# ── Type aliases ──
GoalKind = Literal["user_goal", "self_maintenance", "learning", "relationship", "system_health"]
GoalStatus = Literal["candidate", "active", "paused", "blocked", "completed", "abandoned"]
GoalHorizon = Literal["now", "session", "day", "week", "persistent"]
GoalSourceScope = Literal["user", "system", "metric", "self", "derived"]
TaskType = Literal["research", "recall", "verify", "apply"]
TaskStatus = Literal["pending", "running", "completed", "failed", "interrupted"]
GoalEffect = Literal["pending", "advanced", "inconclusive", "regressed"]

ObserveOutcome = Literal["created", "merged", "rejected", "rate_limited", "cooldown_blocked"]
AddOutcome = Literal["added", "rate_limited", "cap_reached", "duplicate"]

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_WS_RE = re.compile(r"\s+")


def compute_recurrence_key(content: str, signal_type: str = "", source_scope: str = "") -> str:
    """Deterministic content hash for dedup.

    lowercase -> strip punctuation -> compress whitespace -> truncate 120 chars
    -> combine with signal_type + source_scope -> SHA1 hex[:16]
    """
    normalized = content.lower()
    normalized = _PUNCT_RE.sub("", normalized)
    normalized = _MULTI_WS_RE.sub(" ", normalized).strip()
    normalized = normalized[:120]
    raw = f"{signal_type}:{source_scope}:{normalized}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ── Result types ──

@dataclass
class ObserveResult:
    outcome: ObserveOutcome
    goal: Goal | None = None
    reason: str = ""


@dataclass
class AddResult:
    outcome: AddOutcome
    goal: Goal | None = None
    reason: str = ""


# ── GoalSignal ──

@dataclass
class GoalSignal:
    """Normalized input for the goal manager."""

    signal_type: str
    source: str
    source_scope: GoalSourceScope
    content: str
    tag_cluster: tuple[str, ...] = ()
    priority_hint: float = 0.5
    signal_id: str = field(default_factory=lambda: f"gs_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    recurrence_key: str = ""

    def __post_init__(self) -> None:
        if not self.recurrence_key:
            self.recurrence_key = compute_recurrence_key(
                self.content, self.signal_type, self.source_scope,
            )


# ── GoalTask ──

@dataclass
class GoalTask:
    goal_id: str
    description: str
    task_type: TaskType = "research"
    task_id: str = field(default_factory=lambda: f"gt_{uuid.uuid4().hex[:12]}")
    status: TaskStatus = "pending"
    goal_effect: GoalEffect = "pending"
    intent_id: str | None = None
    dispatched_intent_id: str = ""
    result_summary: str = ""
    golden_trace_id: str = ""
    golden_command_id: str = ""
    golden_status: str = "none"
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "description": self.description,
            "task_type": self.task_type,
            "status": self.status,
            "goal_effect": self.goal_effect,
            "intent_id": self.intent_id,
            "dispatched_intent_id": self.dispatched_intent_id,
            "result_summary": self.result_summary,
            "golden_trace_id": self.golden_trace_id,
            "golden_command_id": self.golden_command_id,
            "golden_status": self.golden_status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoalTask:
        return cls(
            task_id=d.get("task_id", f"gt_{uuid.uuid4().hex[:12]}"),
            goal_id=d.get("goal_id", ""),
            description=d.get("description", ""),
            task_type=d.get("task_type", "research"),
            status=d.get("status", "pending"),
            goal_effect=d.get("goal_effect", "pending"),
            intent_id=d.get("intent_id"),
            dispatched_intent_id=d.get("dispatched_intent_id", ""),
            result_summary=d.get("result_summary", ""),
            golden_trace_id=d.get("golden_trace_id", ""),
            golden_command_id=d.get("golden_command_id", ""),
            golden_status=d.get("golden_status", "none"),
            created_at=d.get("created_at", time.time()),
            completed_at=d.get("completed_at"),
        )


# ── Goal ──

@dataclass
class Goal:
    """A durable objective with evidence-based promotion and strict completion."""

    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    title: str = ""
    kind: GoalKind = "learning"
    status: GoalStatus = "candidate"
    priority: float = 0.5
    horizon: GoalHorizon = "session"

    # Completion (strict — all_criteria_met is a computed property, NOT stored)
    success_criteria: list[str] = field(default_factory=list)
    matched_criteria: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)

    # Structure
    parent_goal_id: str | None = None
    tag_cluster: tuple[str, ...] = ()

    # Evidence / promotion
    recurrence_count: int = 1
    last_observed_at: float = field(default_factory=time.time)
    promotion_score: float = 0.0
    evidence_types: list[str] = field(default_factory=list)
    explicit_user_requested: bool = False
    sustained_deficit_cycles: int = 0
    promotion_reason: str = ""
    merge_count: int = 0

    # Progress
    progress: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)
    tasks_attempted: int = 0
    tasks_succeeded: int = 0
    current_task_id: str | None = None
    tasks: list[GoalTask] = field(default_factory=list)

    # Source
    source_event: str = ""
    source_detail: str = ""
    source_scope: GoalSourceScope = "self"
    recurrence_key: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_reviewed_at: float = 0.0
    last_task_outcome_at: float | None = None
    promoted_at: float | None = None
    completed_at: float | None = None
    paused_at: float | None = None

    # Lifecycle control
    cooldown_until: float | None = None
    requires_reboot_review: bool = False

    # Metric tracking (for active system_health goals)
    baseline_metric_value: float | None = None
    last_live_value: float | None = None
    metric_domain: str = ""

    # Status reasons
    blocked_reason: str = ""
    abandoned_reason: str = ""
    stale_reason: str = ""
    paused_reason: str = ""

    # Review counters (persisted so they survive restarts)
    revalidation_pass_count: int = 0
    refresh_resolved_cycles: int = 0
    refresh_pause_cycles: int = 0

    @property
    def all_criteria_met(self) -> bool:
        if not self.success_criteria:
            return False
        return set(self.success_criteria) <= set(self.matched_criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "priority": self.priority,
            "horizon": self.horizon,
            "success_criteria": self.success_criteria,
            "matched_criteria": self.matched_criteria,
            "failure_conditions": self.failure_conditions,
            "parent_goal_id": self.parent_goal_id,
            "tag_cluster": list(self.tag_cluster),
            "recurrence_count": self.recurrence_count,
            "last_observed_at": self.last_observed_at,
            "promotion_score": self.promotion_score,
            "evidence_types": self.evidence_types,
            "explicit_user_requested": self.explicit_user_requested,
            "sustained_deficit_cycles": self.sustained_deficit_cycles,
            "promotion_reason": self.promotion_reason,
            "merge_count": self.merge_count,
            "progress": self.progress,
            "evidence_refs": self.evidence_refs[-50:],
            "tasks_attempted": self.tasks_attempted,
            "tasks_succeeded": self.tasks_succeeded,
            "current_task_id": self.current_task_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "source_event": self.source_event,
            "source_detail": self.source_detail,
            "source_scope": self.source_scope,
            "recurrence_key": self.recurrence_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_reviewed_at": self.last_reviewed_at,
            "last_task_outcome_at": self.last_task_outcome_at,
            "promoted_at": self.promoted_at,
            "completed_at": self.completed_at,
            "paused_at": self.paused_at,
            "cooldown_until": self.cooldown_until,
            "requires_reboot_review": self.requires_reboot_review,
            "baseline_metric_value": self.baseline_metric_value,
            "last_live_value": self.last_live_value,
            "metric_domain": self.metric_domain,
            "blocked_reason": self.blocked_reason,
            "abandoned_reason": self.abandoned_reason,
            "stale_reason": self.stale_reason,
            "paused_reason": self.paused_reason,
            "revalidation_pass_count": self.revalidation_pass_count,
            "refresh_resolved_cycles": self.refresh_resolved_cycles,
            "refresh_pause_cycles": self.refresh_pause_cycles,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Goal:
        tasks_raw = d.get("tasks", [])
        tasks = [GoalTask.from_dict(t) if isinstance(t, dict) else t for t in tasks_raw]
        tag = d.get("tag_cluster", ())
        return cls(
            goal_id=d.get("goal_id", f"goal_{uuid.uuid4().hex[:12]}"),
            title=d.get("title", ""),
            kind=d.get("kind", "learning"),
            status=d.get("status", "candidate"),
            priority=d.get("priority", 0.5),
            horizon=d.get("horizon", "session"),
            success_criteria=d.get("success_criteria", []),
            matched_criteria=d.get("matched_criteria", []),
            failure_conditions=d.get("failure_conditions", []),
            parent_goal_id=d.get("parent_goal_id"),
            tag_cluster=tuple(tag) if isinstance(tag, list) else tag,
            recurrence_count=d.get("recurrence_count", 1),
            last_observed_at=d.get("last_observed_at", 0.0),
            promotion_score=d.get("promotion_score", 0.0),
            evidence_types=d.get("evidence_types", []),
            explicit_user_requested=d.get("explicit_user_requested", False),
            sustained_deficit_cycles=d.get("sustained_deficit_cycles", 0),
            promotion_reason=d.get("promotion_reason", ""),
            merge_count=d.get("merge_count", 0),
            progress=d.get("progress", 0.0),
            evidence_refs=d.get("evidence_refs", []),
            tasks_attempted=d.get("tasks_attempted", 0),
            tasks_succeeded=d.get("tasks_succeeded", 0),
            current_task_id=d.get("current_task_id"),
            tasks=tasks,
            source_event=d.get("source_event", ""),
            source_detail=d.get("source_detail", ""),
            source_scope=d.get("source_scope", "self"),
            recurrence_key=d.get("recurrence_key", ""),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            last_reviewed_at=d.get("last_reviewed_at", 0.0),
            last_task_outcome_at=d.get("last_task_outcome_at"),
            promoted_at=d.get("promoted_at"),
            completed_at=d.get("completed_at"),
            paused_at=d.get("paused_at"),
            cooldown_until=d.get("cooldown_until"),
            requires_reboot_review=d.get("requires_reboot_review", False),
            baseline_metric_value=d.get("baseline_metric_value"),
            last_live_value=d.get("last_live_value"),
            metric_domain=d.get("metric_domain", ""),
            blocked_reason=d.get("blocked_reason", ""),
            abandoned_reason=d.get("abandoned_reason", ""),
            stale_reason=d.get("stale_reason", ""),
            paused_reason=d.get("paused_reason", ""),
            revalidation_pass_count=d.get("revalidation_pass_count", 0),
            refresh_resolved_cycles=d.get("refresh_resolved_cycles", 0),
            refresh_pause_cycles=d.get("refresh_pause_cycles", 0),
        )


# ── GoalUpdate (returned by review) ──

@dataclass
class GoalUpdate:
    progress_delta: float = 0.0
    should_complete: bool = False
    should_abandon: bool = False
    should_pause: bool = False
    newly_matched_criteria: list[str] = field(default_factory=list)
    reason: str = ""
