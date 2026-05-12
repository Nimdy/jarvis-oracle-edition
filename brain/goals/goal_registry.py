"""Goal Continuity Layer — persistent goal storage with cap enforcement and cooldowns."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from goals.constants import (
    MAX_ACTIVE_GOALS,
    MAX_CANDIDATES,
    MAX_COMPLETED_RETAINED,
    MAX_NEW_GOALS_PER_HOUR,
)
from goals.goal import AddResult, Goal, GoalStatus

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".jarvis" / "goals.json"
_ECHO_GOAL_MARKERS: tuple[str, ...] = (
    "i don't have that capability yet",
    "let me know if you'd like",
    "currently active",
    "learning jobs",
    "camera control and slash zoom",
    "i'm actively collecting data to improve my ability",
)


def _looks_like_echo_artifact_goal(goal: Goal) -> bool:
    """Detect malformed conversation goals created from echoed assistant speech."""
    if goal.source_scope != "user" or goal.source_event != "conversation":
        return False

    text = f"{goal.title} {goal.source_detail}".strip().lower()
    if not text:
        return False

    if "camera control and slash zoom" in text:
        return True

    has_capability_block = "i don't have that capability yet" in text
    has_learning_inventory = "learning jobs" in text and "currently active" in text
    has_offer_tail = "let me know if you'd like" in text
    has_self_training = "i'm actively collecting data to improve my ability" in text

    if has_capability_block and (has_learning_inventory or has_offer_tail):
        return True
    if has_self_training and has_learning_inventory:
        return True

    marker_count = sum(1 for marker in _ECHO_GOAL_MARKERS if marker in text)
    return marker_count >= 3


class GoalRegistry:
    """Persistent goal store with CRUD, caps, rate limits, and cooldown enforcement."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._goals: dict[str, Goal] = {}
        self._creation_timestamps: deque[float] = deque(maxlen=MAX_NEW_GOALS_PER_HOUR * 2)
        self._promotion_timestamps: deque[float] = deque(maxlen=50)
        self._load()

    # ── CRUD ──

    def add(self, goal: Goal) -> AddResult:
        now = time.time()
        self._prune_rate_window(now)

        recent_creates = sum(1 for ts in self._creation_timestamps if now - ts < 3600)
        if recent_creates >= MAX_NEW_GOALS_PER_HOUR:
            return AddResult(outcome="rate_limited", reason=f"Rate limit: {MAX_NEW_GOALS_PER_HOUR}/hr")

        if goal.status == "candidate":
            candidate_count = sum(1 for g in self._goals.values() if g.status == "candidate")
            if candidate_count >= MAX_CANDIDATES:
                return AddResult(outcome="cap_reached", reason=f"Max candidates: {MAX_CANDIDATES}")
        elif goal.status == "active":
            active_count = sum(1 for g in self._goals.values() if g.status in ("active", "paused", "blocked"))
            if active_count >= MAX_ACTIVE_GOALS:
                return AddResult(outcome="cap_reached", reason=f"Max active goals: {MAX_ACTIVE_GOALS}")

        if goal.goal_id in self._goals:
            return AddResult(outcome="duplicate", reason=f"Goal {goal.goal_id} already exists")

        self._goals[goal.goal_id] = goal
        self._creation_timestamps.append(now)
        self._save()
        return AddResult(outcome="added", goal=goal)

    def update(self, goal_id: str, **fields: Any) -> Goal | None:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        for k, v in fields.items():
            if hasattr(goal, k):
                setattr(goal, k, v)
        goal.updated_at = time.time()
        self._save()
        return goal

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def get_by_status(self, status: GoalStatus) -> list[Goal]:
        return [g for g in self._goals.values() if g.status == status]

    def get_active(self) -> list[Goal]:
        return sorted(
            [g for g in self._goals.values() if g.status in ("active", "blocked", "paused")],
            key=lambda g: g.priority,
            reverse=True,
        )

    def get_candidates(self) -> list[Goal]:
        return sorted(
            [g for g in self._goals.values() if g.status == "candidate"],
            key=lambda g: g.promotion_score,
            reverse=True,
        )

    def get_all(self) -> list[Goal]:
        return list(self._goals.values())

    def get_needing_reboot_review(self) -> list[Goal]:
        return [g for g in self._goals.values() if g.requires_reboot_review]

    def remove(self, goal_id: str) -> bool:
        if goal_id in self._goals:
            del self._goals[goal_id]
            self._save()
            return True
        return False

    # ── Rate limit helpers ──

    def record_promotion(self) -> None:
        self._promotion_timestamps.append(time.time())

    def promotions_this_hour(self) -> int:
        now = time.time()
        return sum(1 for ts in self._promotion_timestamps if now - ts < 3600)

    def creations_this_hour(self) -> int:
        now = time.time()
        return sum(1 for ts in self._creation_timestamps if now - ts < 3600)

    # ── Cooldown ──

    def is_cooldown_active(self, recurrence_key: str, now: float | None = None) -> bool:
        now = now or time.time()
        for g in self._goals.values():
            if g.status not in ("abandoned", "blocked"):
                continue
            if g.recurrence_key != recurrence_key or not recurrence_key:
                continue
            if g.cooldown_until and g.cooldown_until > now:
                return True
        return False

    def get_cooled_down_goals(self, now: float | None = None) -> list[Goal]:
        """Return abandoned/blocked goals that still have an active cooldown."""
        now = now or time.time()
        return [
            g for g in self._goals.values()
            if g.status in ("abandoned", "blocked")
            and g.cooldown_until and g.cooldown_until > now
        ]

    # ── Cleanup ──

    def cleanup_expired(self, now: float | None = None) -> int:
        from goals.constants import CANDIDATE_EXPIRY_S

        now = now or time.time()
        removed = 0
        expired_ids: list[str] = []

        for gid, g in self._goals.items():
            if g.status == "candidate" and (now - g.created_at) > CANDIDATE_EXPIRY_S:
                expired_ids.append(gid)

        for gid in expired_ids:
            del self._goals[gid]
            removed += 1

        completed = sorted(
            [g for g in self._goals.values() if g.status == "completed"],
            key=lambda g: g.completed_at or 0.0,
            reverse=True,
        )
        if len(completed) > MAX_COMPLETED_RETAINED:
            for g in completed[MAX_COMPLETED_RETAINED:]:
                del self._goals[g.goal_id]
                removed += 1

        if removed:
            self._save()
        return removed

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        kind_counts: dict[str, int] = {}
        horizon_counts: dict[str, int] = {}
        for g in self._goals.values():
            status_counts[g.status] = status_counts.get(g.status, 0) + 1
            kind_counts[g.kind] = kind_counts.get(g.kind, 0) + 1
            horizon_counts[g.horizon] = horizon_counts.get(g.horizon, 0) + 1

        cooldowns_active = sum(
            1 for g in self._goals.values()
            if g.cooldown_until and g.cooldown_until > time.time()
        )

        return {
            "total": len(self._goals),
            "by_status": status_counts,
            "by_kind": kind_counts,
            "by_horizon": horizon_counts,
            "creations_this_hour": self.creations_this_hour(),
            "promotions_this_hour": self.promotions_this_hour(),
            "cooldowns_active": cooldowns_active,
        }

    # ── Persistence ──

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for gid, gdata in data.get("goals", {}).items():
                self._goals[gid] = Goal.from_dict(gdata)
            logger.info("Loaded %d goals from %s", len(self._goals), self._path)
            sanitized = self._sanitize_echo_artifact_goals()
            if sanitized:
                logger.info("Sanitized %d malformed echoed conversation goals", sanitized)
                self._save()
            backfilled = self._backfill_goal_effects()
            if backfilled:
                logger.info("Backfilled goal_effect on %d legacy tasks", backfilled)
                self._save()
        except Exception as exc:
            logger.warning("Failed to load goals: %s", exc)

    def _sanitize_echo_artifact_goals(self) -> int:
        removed = 0
        for gid, goal in list(self._goals.items()):
            if _looks_like_echo_artifact_goal(goal):
                del self._goals[gid]
                removed += 1
        return removed

    def _backfill_goal_effects(self) -> int:
        """One-time migration: infer goal_effect for legacy tasks with effect='pending'.

        Rules:
          - completed + has result_summary → advanced
          - completed + no result_summary → inconclusive
          - failed → inconclusive (execution failed, effect unknown)
          - interrupted → inconclusive
          - pending/running → leave as pending
        """
        count = 0
        for goal in self._goals.values():
            for task in goal.tasks:
                if task.goal_effect != "pending":
                    continue
                if task.status == "completed":
                    task.goal_effect = "advanced" if task.result_summary else "inconclusive"
                    count += 1
                elif task.status in ("failed", "interrupted"):
                    task.goal_effect = "inconclusive"
                    count += 1
        return count

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"goals": {gid: g.to_dict() for gid, g in self._goals.items()}}
            from memory.persistence import atomic_write_json
            atomic_write_json(self._path, data, indent=2)
        except Exception as exc:
            logger.warning("Failed to save goals: %s", exc)

    def _prune_rate_window(self, now: float) -> None:
        cutoff = now - 3600
        while self._creation_timestamps and self._creation_timestamps[0] < cutoff:
            self._creation_timestamps.popleft()
        while self._promotion_timestamps and self._promotion_timestamps[0] < cutoff:
            self._promotion_timestamps.popleft()
