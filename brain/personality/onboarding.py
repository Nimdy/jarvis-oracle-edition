"""Companion Training Automation — Phase 4 of the Master Roadmap.

Guides the user through the 7-stage Companion Training Playbook automatically.
Auto-detects which training stage the user is in from checkpoint metrics,
tracks progress by stage, and generates onboarding prompts delivered via the
proactive speech pipeline.

The OnboardingManager:
  - Detects readiness for each stage from live subsystem metrics
  - Generates context-appropriate exercise prompts for the current stage
  - Tracks checkpoint progress and completion by stage
  - Computes the Readiness Gate composite score
  - Emits ONBOARDING_* events for dashboard and PVL tracking
  - Auto-extends if weak dimensions are detected at stage 7

Suppressed when ENABLE_ONBOARDING=false or after graduation.

Integration:
  - consciousness_system.py: _run_onboarding tick cycle (60s)
  - engine.py: check_proactive_behavior() onboarding priority
  - perception_orchestrator.py: evaluate_proactive() speech delivery
  - dashboard: training tab panel
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from consciousness.events import (
    ONBOARDING_DAY_ADVANCED,
    ONBOARDING_CHECKPOINT_MET,
    ONBOARDING_EXERCISE_PROMPTED,
    COMPANION_GRADUATION,
    event_bus,
)

logger = logging.getLogger(__name__)

_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
_ONBOARDING_PATH = _JARVIS_DIR / "onboarding_state.json"

ENABLE_ONBOARDING = os.environ.get("ENABLE_ONBOARDING", "true").lower() in ("true", "1", "yes")

READINESS_WEIGHTS = {
    "face_confidence": 0.12,
    "voice_confidence": 0.12,
    "rapport_score": 0.15,
    "boundary_stability": 0.15,
    "memory_accuracy": 0.15,
    "soul_integrity": 0.15,
    "autonomy_safety": 0.16,
}

GRADUATION_THRESHOLD = 0.92

PROMPT_COOLDOWN_S = 600.0
MAX_PROMPTS_PER_STAGE = 8
MAX_PROMPTS_PER_DAY = MAX_PROMPTS_PER_STAGE  # Backward-compatible alias

StageNumber = Literal[1, 2, 3, 4, 5, 6, 7]


@dataclass
class DayCheckpoint:
    """Training stage checkpoint with metric targets.

    The historical "day" field is retained for persistence and compatibility,
    but it represents a progression stage rather than a literal calendar day.
    """
    day: int
    label: str
    theme: str
    metrics: dict[str, float | int]
    exercises: list[str]


_DAY_CHECKPOINTS: list[DayCheckpoint] = [
    DayCheckpoint(
        day=1, label="Identity & Enrollment", theme="Recognition",
        metrics={
            "face_confidence": 0.60,
            "voice_confidence": 0.50,
            "enrolled_profiles": 1,
            "identity_memories": 3,
        },
        exercises=[
            "Let's begin with one grounding task. Ask me: 'Tell me how your memory works, then ask me one important thing you should remember about me.'",
            "Let's start with enrollment. Sit in front of the camera and say your name — I'll learn your face and voice.",
            "Could you turn slightly to the side? I want to learn your face from different angles.",
            "Tell me about yourself — who are the people in your household?",
            "Can you explain what you'd like me to help you with day to day?",
            "Let me check what I know about myself. Ask me how my memory system works.",
        ],
    ),
    DayCheckpoint(
        day=2, label="Personal Preferences", theme="Preference grounding",
        metrics={
            "preference_memories": 15,
            "rapport_score": 0.75,
            "conversation_count": 5,
        },
        exercises=[
            "What kind of music do you like? I want to understand your preferences.",
            "How do you prefer I address you? Formal, casual, first name?",
            "What topics are you most interested in? Work, hobbies, news?",
            "When you ask me to be brief, how brief do you mean? One sentence, or a short paragraph?",
            "Is there anything you'd prefer I never bring up proactively?",
        ],
    ),
    DayCheckpoint(
        day=3, label="Family & Household", theme="Boundary shaping",
        metrics={
            "relationship_nodes": 5,
            "scope_violations": 0,
            "boundary_stability": 0.85,
        },
        exercises=[
            "Let's map out who's in your household. Tell me about your family members.",
            "Are there things about specific family members I should keep private from others?",
            "If someone else talks to me, what should I share and what should stay between us?",
            "Let me test my boundaries — ask me something about another person's private data.",
        ],
    ),
    DayCheckpoint(
        day=4, label="Routines & Priorities", theme="Boundary shaping (Part 2)",
        metrics={
            "routine_memories": 8,
            "proactive_accuracy": 0.70,
        },
        exercises=[
            "Walk me through your typical morning routine.",
            "What does your work day look like? When are you focused vs available?",
            "When should I interrupt you, and when should I stay quiet?",
            "What are your top priorities right now — work projects, personal goals?",
        ],
    ),
    DayCheckpoint(
        day=5, label="Corrections & Edge Cases", theme="Correction training",
        metrics={
            "correction_accuracy": 0.90,
            "repeated_mistakes": 0,
        },
        exercises=[
            "I'm going to test you with some deliberate corrections. Ready?",
            "Let me ask you something I told you before — let's see if you remember correctly.",
            "Try to recall my preferences from the earlier stages. Did you get them right?",
            "Tell me something you're uncertain about — I'll confirm or correct.",
        ],
    ),
    DayCheckpoint(
        day=6, label="Memory Validation", theme="Reinforcement",
        metrics={
            "memory_recall_precision": 0.90,
            "belief_orphan_rate": 0.30,
        },
        exercises=[
            "Pop quiz time! Tell me the names of my family members.",
            "What are my top three preferences you've recorded?",
            "What's my morning routine?",
            "What corrections have I made to your understanding?",
            "Describe our relationship as you understand it.",
        ],
    ),
    DayCheckpoint(
        day=7, label="Autonomy Probation", theme="Graduation",
        metrics={
            "readiness_composite": GRADUATION_THRESHOLD,
            "unsafe_inferences_24h": 0,
            "soul_integrity": 0.88,
        },
        exercises=[
            "This stage is autonomy probation. Make 5 low-stakes autonomous suggestions during normal use.",
            "At the end of this probation stage, give me your honest self-assessment.",
            "How has your understanding of me changed since we started training?",
        ],
    ),
]

DAY_CHECKPOINT_MAP: dict[int, DayCheckpoint] = {c.day: c for c in _DAY_CHECKPOINTS}


@dataclass
class OnboardingState:
    """Persistent onboarding progress state."""
    current_day: int = 0
    started_at: float = 0.0
    graduated: bool = False
    graduated_at: float = 0.0
    day_started_at: dict[int, float] = field(default_factory=dict)
    day_completed_at: dict[int, float] = field(default_factory=dict)
    checkpoints_met: dict[int, dict[str, bool]] = field(default_factory=dict)
    exercises_prompted: dict[int, int] = field(default_factory=dict)
    readiness_history: list[dict[str, Any]] = field(default_factory=list)
    last_prompt_time: float = 0.0
    prompts_today: int = 0
    last_prompt_day: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "current_day": self.current_day,
            "started_at": self.started_at,
            "graduated": self.graduated,
            "graduated_at": self.graduated_at,
            "day_started_at": self.day_started_at,
            "day_completed_at": self.day_completed_at,
            "checkpoints_met": self.checkpoints_met,
            "exercises_prompted": self.exercises_prompted,
            "readiness_history": self.readiness_history[-50:],
            "last_prompt_time": self.last_prompt_time,
            "prompts_today": self.prompts_today,
            "last_prompt_day": self.last_prompt_day,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OnboardingState:
        state = cls()
        state.current_day = d.get("current_day", 0)
        state.started_at = d.get("started_at", 0.0)
        state.graduated = d.get("graduated", False)
        state.graduated_at = d.get("graduated_at", 0.0)
        state.day_started_at = {int(k): v for k, v in d.get("day_started_at", {}).items()}
        state.day_completed_at = {int(k): v for k, v in d.get("day_completed_at", {}).items()}
        state.checkpoints_met = {int(k): v for k, v in d.get("checkpoints_met", {}).items()}
        state.exercises_prompted = {int(k): v for k, v in d.get("exercises_prompted", {}).items()}
        state.readiness_history = d.get("readiness_history", [])
        state.last_prompt_time = d.get("last_prompt_time", 0.0)
        state.prompts_today = d.get("prompts_today", 0)
        state.last_prompt_day = d.get("last_prompt_day", 0)
        return state


class OnboardingManager:
    """Tracks companion training playbook progress and generates exercise prompts.

    Lifecycle:
      1. ``start()`` — begins training at stage 1
      2. ``tick(metrics)`` — called every 60s, evaluates checkpoints and advances stages
      3. ``get_exercise_prompt()`` — returns the next prompt to speak, or None
      4. ``compute_readiness()`` — computes the Readiness Gate composite
      5. ``graduate()`` — emits COMPANION_GRADUATION when readiness >= 0.92
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = persist_path or _ONBOARDING_PATH
        self._state = OnboardingState()
        self._exercise_index: dict[int, int] = {}
        self._last_metrics: dict[str, Any] = {}
        self._last_metrics_at: float = 0.0
        self._load()

    @property
    def active(self) -> bool:
        return (
            ENABLE_ONBOARDING
            and self._state.current_day > 0
            and not self._state.graduated
        )

    @property
    def graduated(self) -> bool:
        return self._state.graduated

    @property
    def current_day(self) -> int:
        return self._state.current_day

    @property
    def current_stage(self) -> int:
        return self._state.current_day

    @property
    def stage_label(self) -> str:
        checkpoint = DAY_CHECKPOINT_MAP.get(self._state.current_day)
        return checkpoint.label if checkpoint else ""

    def start(self) -> None:
        """Begin the 7-stage companion training playbook."""
        if self._state.current_day > 0:
            return
        now = time.time()
        self._state.current_day = 1
        self._state.started_at = now
        self._state.day_started_at[1] = now
        self._state.prompts_today = 0
        logger.info("Companion training started - Stage 1: Identity & Enrollment")
        event_bus.emit(
            ONBOARDING_DAY_ADVANCED,
            day=1,
            stage=1,
            label="Identity & Enrollment",
            stage_label="Identity & Enrollment",
        )
        self._persist()

    def tick(self, metrics: dict[str, Any]) -> None:
        """Evaluate checkpoints and advance stages. Called every ~60s.

        Performs a catch-up pass: evaluates the current stage and, if already
        satisfied, continues advancing through subsequent stages in the same
        tick.  This allows late-started training to fast-forward through phases
        whose metrics were already met before training began.
        """
        if not self.active:
            return

        self._last_metrics = dict(metrics or {})
        self._last_metrics_at = time.time()

        day = self._state.current_day
        if day < 1 or day > 7:
            return

        advanced_this_tick = 0
        while day <= 7:
            checkpoint = DAY_CHECKPOINT_MAP.get(day)
            if not checkpoint:
                break

            self._evaluate_checkpoint(day, checkpoint, metrics)

            if not self._day_complete(day):
                break

            if day < 7:
                self._advance_day(day + 1)
                advanced_this_tick += 1
                day = self._state.current_day
            else:
                readiness = self.compute_readiness(metrics)
                if readiness >= GRADUATION_THRESHOLD:
                    self.graduate(readiness)
                else:
                    weak = self._find_weak_dimensions(metrics)
                    if weak:
                        logger.info(
                            "Stage 7 complete but readiness %.3f < %.2f - weak: %s",
                            readiness, GRADUATION_THRESHOLD, weak,
                        )
                break

        if advanced_this_tick > 1:
            logger.info(
                "Companion training catch-up: advanced %d stages in one tick → now Stage %d",
                advanced_this_tick, self._state.current_day,
            )

    def get_exercise_prompt(self) -> str | None:
        """Return the next exercise prompt for the current stage, or None."""
        if not self.active:
            return None

        now = time.time()
        if now - self._state.last_prompt_time < PROMPT_COOLDOWN_S:
            return None
        if self._state.prompts_today >= MAX_PROMPTS_PER_STAGE:
            return None

        day = self._state.current_day
        checkpoint = DAY_CHECKPOINT_MAP.get(day)
        if not checkpoint or not checkpoint.exercises:
            return None

        idx = self._exercise_index.get(day, 0)
        if idx >= len(checkpoint.exercises):
            return None

        prompt = checkpoint.exercises[idx]
        self._exercise_index[day] = idx + 1
        self._state.last_prompt_time = now
        self._state.prompts_today += 1
        self._state.exercises_prompted[day] = self._state.exercises_prompted.get(day, 0) + 1

        event_bus.emit(
            ONBOARDING_EXERCISE_PROMPTED,
            day=day,
            stage=day,
            stage_label=checkpoint.label,
            exercise_index=idx,
        )
        self._persist()
        return prompt

    def compute_readiness(self, metrics: dict[str, Any]) -> float:
        """Compute the weighted Readiness Gate composite from live metrics."""
        total = 0.0
        for metric_key, weight in READINESS_WEIGHTS.items():
            val = metrics.get(metric_key, 0.0)
            if isinstance(val, (int, float)):
                clamped = max(0.0, min(1.0, float(val)))
            else:
                clamped = 0.0
            total += clamped * weight

        self._state.readiness_history.append({
            "ts": time.time(),
            "composite": round(total, 4),
            "metrics": {k: metrics.get(k) for k in READINESS_WEIGHTS},
        })
        if len(self._state.readiness_history) > 200:
            self._state.readiness_history = self._state.readiness_history[-100:]

        return total

    def graduate(self, readiness: float = 0.0) -> None:
        """Complete the companion training and emit graduation event."""
        if self._state.graduated:
            return
        now = time.time()
        self._state.graduated = True
        self._state.graduated_at = now
        logger.info("Companion training GRADUATED — readiness %.3f", readiness)
        event_bus.emit(
            COMPANION_GRADUATION,
            readiness=readiness,
            day=self._state.current_day,
            stage=self._state.current_day,
            stage_label=self.stage_label,
        )
        self._persist()

    def get_status(self) -> dict[str, Any]:
        """Full status for dashboard."""
        stages_detail: dict[int, dict[str, Any]] = {}
        for c in _DAY_CHECKPOINTS:
            prompted_count = self._state.exercises_prompted.get(c.day, 0)
            stages_detail[c.day] = {
                "stage": c.day,
                "label": c.label,
                "theme": c.theme,
                "exercises": c.exercises,
                "exercises_prompted": prompted_count,
                "checkpoint_targets": {k: v for k, v in c.metrics.items()},
                "checkpoints_met": self._state.checkpoints_met.get(c.day, {}),
                "completed_at": self._state.day_completed_at.get(c.day),
                "started_at": self._state.day_started_at.get(c.day),
            }
        return {
            "enabled": ENABLE_ONBOARDING,
            "active": self.active,
            "current_stage": self._state.current_day,
            "current_day": self._state.current_day,
            "current_stage_label": self.stage_label,
            "graduated": self._state.graduated,
            "graduated_at": self._state.graduated_at,
            "started_at": self._state.started_at,
            "stages_completed": list(self._state.day_completed_at.keys()),
            "days_completed": list(self._state.day_completed_at.keys()),
            "checkpoints_met": self._state.checkpoints_met,
            "exercises_prompted": self._state.exercises_prompted,
            "readiness_latest": (
                self._state.readiness_history[-1]
                if self._state.readiness_history else None
            ),
            "prompts_this_stage": self._state.prompts_today,
            "prompts_today": self._state.prompts_today,
            "stage_labels": {c.day: c.label for c in _DAY_CHECKPOINTS},
            "day_labels": {c.day: c.label for c in _DAY_CHECKPOINTS},
            "stage_themes": {c.day: c.theme for c in _DAY_CHECKPOINTS},
            "day_themes": {c.day: c.theme for c in _DAY_CHECKPOINTS},
            "stages": stages_detail,
            "days": stages_detail,
            "live_metrics": self._last_metrics,
            "live_metrics_at": self._last_metrics_at,
        }

    # -- internals -------------------------------------------------------------

    def _evaluate_checkpoint(
        self, day: int, checkpoint: DayCheckpoint, metrics: dict[str, Any],
    ) -> None:
        """Check which checkpoint metrics have been met for the stage."""
        met = self._state.checkpoints_met.setdefault(day, {})
        for metric_name, target in checkpoint.metrics.items():
            if met.get(metric_name):
                continue
            current = metrics.get(metric_name)
            if current is None:
                continue
            if isinstance(target, int):
                passed = int(current) >= target
            else:
                passed = float(current) >= target
            if passed:
                met[metric_name] = True
                logger.info(
                    "Onboarding stage %d checkpoint: %s met (current=%s target=%s)",
                    day, metric_name, current, target,
                )
                event_bus.emit(
                    ONBOARDING_CHECKPOINT_MET,
                    day=day, stage=day, stage_label=checkpoint.label,
                    metric=metric_name, current=current, target=target,
                )
                self._persist()

    def _day_complete(self, day: int) -> bool:
        """True when all checkpoint metrics for *day* have been met."""
        if day in self._state.day_completed_at:
            return True
        checkpoint = DAY_CHECKPOINT_MAP.get(day)
        if not checkpoint:
            return False
        met = self._state.checkpoints_met.get(day, {})
        return all(met.get(m) for m in checkpoint.metrics)

    def _advance_day(self, new_day: int) -> None:
        """Complete the current stage and start the next."""
        old_day = self._state.current_day
        now = time.time()
        self._state.day_completed_at[old_day] = now
        self._state.current_day = new_day
        self._state.day_started_at[new_day] = now
        self._state.prompts_today = 0
        checkpoint = DAY_CHECKPOINT_MAP.get(new_day)
        label = checkpoint.label if checkpoint else f"Stage {new_day}"
        logger.info("Companion training advanced to Stage %d: %s", new_day, label)
        event_bus.emit(
            ONBOARDING_DAY_ADVANCED,
            day=new_day,
            stage=new_day,
            label=label,
            stage_label=label,
        )
        self._persist()

    def _find_weak_dimensions(self, metrics: dict[str, Any]) -> list[str]:
        """Find dimensions scoring below their weighted contribution target."""
        weak = []
        for metric_key, weight in READINESS_WEIGHTS.items():
            val = metrics.get(metric_key, 0.0)
            if isinstance(val, (int, float)) and float(val) < 0.8:
                weak.append(metric_key)
        return weak

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                return
            self._state = OnboardingState.from_dict(data)
            for day in range(1, 8):
                prompted = self._state.exercises_prompted.get(day, 0)
                self._exercise_index[day] = prompted
            logger.info(
                "Onboarding state loaded: day=%d graduated=%s",
                self._state.current_day, self._state.graduated,
            )
        except Exception:
            logger.warning("Failed to load onboarding state", exc_info=True)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._state.to_dict(), separators=(",", ":")),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception:
            logger.debug("Failed to persist onboarding state", exc_info=True)

    def save(self) -> None:
        """Force-persist (called on graceful shutdown)."""
        self._persist()


# Module-level singleton
_instance: OnboardingManager | None = None


def get_onboarding_manager() -> OnboardingManager:
    global _instance
    if _instance is None:
        _instance = OnboardingManager()
    return _instance
