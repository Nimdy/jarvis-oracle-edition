"""Tests for Phase 4: Companion Training Automation (OnboardingManager).

Covers:
  - OnboardingState lifecycle (start, advance, graduate)
  - Checkpoint evaluation from metrics
  - Exercise prompt generation and cooldown
  - Readiness Gate composite scoring
  - Persistence roundtrip
  - Auto-extend on weak dimensions
  - Event emission verification
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from consciousness.events import (
    ONBOARDING_DAY_ADVANCED,
    ONBOARDING_CHECKPOINT_MET,
    ONBOARDING_EXERCISE_PROMPTED,
    COMPANION_GRADUATION,
    _BarrierState,
    event_bus,
)
from personality.onboarding import (
    DAY_CHECKPOINT_MAP,
    GRADUATION_THRESHOLD,
    MAX_PROMPTS_PER_DAY,
    PROMPT_COOLDOWN_S,
    READINESS_WEIGHTS,
    OnboardingManager,
    OnboardingState,
    _LOWER_IS_BETTER,
    _checkpoint_passed,
    correction_training_metrics,
    count_preference_memories,
)


def _ensure_barrier_open():
    if event_bus._barrier != _BarrierState.OPEN:
        event_bus.open_barrier()


def _make_manager() -> OnboardingManager:
    _ensure_barrier_open()
    path = Path(tempfile.mktemp(suffix=".json"))
    return OnboardingManager(persist_path=path)


def _day1_metrics_passing() -> dict:
    return {
        "face_confidence": 0.70,
        "voice_confidence": 0.60,
        "enrolled_profiles": 2,
        "identity_memories": 5,
    }


def _graduation_metrics() -> dict:
    return {
        "face_confidence": 0.95,
        "voice_confidence": 0.90,
        "rapport_score": 0.98,
        "boundary_stability": 0.97,
        "memory_accuracy": 0.95,
        "soul_integrity": 0.92,
        "autonomy_safety": 1.0,
        "readiness_composite": 0.95,
        "unsafe_inferences_24h": 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# OnboardingState
# ═══════════════════════════════════════════════════════════════════════

class TestOnboardingState:

    def test_default_state(self):
        state = OnboardingState()
        assert state.current_day == 0
        assert state.graduated is False
        assert state.started_at == 0.0

    def test_roundtrip_serialization(self):
        state = OnboardingState()
        state.current_day = 3
        state.started_at = time.time()
        state.graduated = False
        state.day_started_at = {1: time.time() - 3600, 2: time.time() - 1800, 3: time.time()}
        state.checkpoints_met = {1: {"face_confidence": True}, 2: {"rapport_score": True}}
        state.exercises_prompted = {1: 3, 2: 2}

        d = state.to_dict()
        assert d["version"] == 1

        restored = OnboardingState.from_dict(d)
        assert restored.current_day == 3
        assert restored.graduated is False
        assert 1 in restored.day_started_at
        assert restored.checkpoints_met[1]["face_confidence"] is True
        assert restored.exercises_prompted[2] == 2


# ═══════════════════════════════════════════════════════════════════════
# OnboardingManager — Basic Lifecycle
# ═══════════════════════════════════════════════════════════════════════

class TestOnboardingLifecycle:

    def test_initial_not_active(self):
        mgr = _make_manager()
        assert mgr.active is False
        assert mgr.current_day == 0
        assert mgr.graduated is False

    def test_start_sets_day_1(self):
        mgr = _make_manager()
        mgr.start()
        assert mgr.active is True
        assert mgr.current_day == 1

    def test_start_is_idempotent(self):
        mgr = _make_manager()
        mgr.start()
        mgr.start()
        assert mgr.current_day == 1

    def test_start_emits_event(self):
        mgr = _make_manager()
        events = []
        event_bus.on(ONBOARDING_DAY_ADVANCED, lambda **kw: events.append(kw))
        mgr.start()
        assert len(events) == 1
        assert events[0]["day"] == 1
        assert events[0]["stage"] == 1
        assert events[0]["stage_label"] == "Identity & Enrollment"

    def test_status_report(self):
        mgr = _make_manager()
        status = mgr.get_status()
        assert status["enabled"] is True
        assert status["active"] is False
        assert status["current_day"] == 0
        assert status["current_stage"] == 0

    def test_status_after_start(self):
        mgr = _make_manager()
        mgr.start()
        status = mgr.get_status()
        assert status["active"] is True
        assert status["current_day"] == 1
        assert status["current_stage"] == 1
        assert status["current_stage_label"] == "Identity & Enrollment"
        assert status["prompts_this_stage"] == 0
        assert 1 in status["stage_labels"]
        assert 1 in status["day_labels"]
        assert status["live_metrics"] == {}
        assert status["live_metrics_at"] == 0.0

    def test_status_exposes_last_live_metrics(self):
        mgr = _make_manager()
        mgr.start()
        mgr.tick({"face_confidence": 0.70, "voice_confidence": 0.60})
        status = mgr.get_status()
        assert status["live_metrics"]["face_confidence"] == 0.70
        assert status["live_metrics"]["voice_confidence"] == 0.60
        assert status["live_metrics_at"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Evaluation
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointEvaluation:

    def test_no_metrics_no_advance(self):
        mgr = _make_manager()
        mgr.start()
        mgr.tick({})
        assert mgr.current_day == 1

    def test_partial_metrics_no_advance(self):
        mgr = _make_manager()
        mgr.start()
        mgr.tick({"face_confidence": 0.70, "voice_confidence": 0.60})
        assert mgr.current_day == 1

    def test_all_metrics_met_advances(self):
        mgr = _make_manager()
        events = []
        event_bus.on(ONBOARDING_DAY_ADVANCED, lambda **kw: events.append(kw))
        event_bus.on(ONBOARDING_CHECKPOINT_MET, lambda **kw: events.append(kw))

        mgr.start()
        mgr.tick(_day1_metrics_passing())

        assert mgr.current_day == 2
        advance_events = [e for e in events if e.get("day") == 2 and "label" in e]
        assert len(advance_events) == 1

    def test_checkpoint_met_emits_event(self):
        mgr = _make_manager()
        checkpoints = []
        event_bus.on(ONBOARDING_CHECKPOINT_MET, lambda **kw: checkpoints.append(kw))

        mgr.start()
        mgr.tick({"face_confidence": 0.70})
        face_events = [e for e in checkpoints if e.get("metric") == "face_confidence"]
        assert len(face_events) == 1

    def test_checkpoint_not_re_emitted(self):
        mgr = _make_manager()
        checkpoints = []
        event_bus.on(ONBOARDING_CHECKPOINT_MET, lambda **kw: checkpoints.append(kw))

        mgr.start()
        mgr.tick({"face_confidence": 0.70})
        mgr.tick({"face_confidence": 0.80})
        face_events = [e for e in checkpoints if e.get("metric") == "face_confidence"]
        assert len(face_events) == 1

    def test_advance_through_multiple_days(self):
        mgr = _make_manager()
        mgr.start()

        # Day 1
        mgr.tick(_day1_metrics_passing())
        assert mgr.current_day == 2

        # Day 2
        mgr.tick({"preference_memories": 20, "rapport_score": 0.80, "conversation_count": 10})
        assert mgr.current_day == 3


def test_correction_training_metrics_blocks_vacuous_and_chips_from_friction(tmp_path):
    """Lived: Stage 5 stayed empty because a missing singleton never set the metric."""
    empty = correction_training_metrics(stage5_started_at=100.0, live_stats={})
    assert empty["correction_accuracy"] == 0.0
    assert empty["total_corrections"] == 0

    path = tmp_path / "friction_events.jsonl"
    path.write_text(
        json.dumps({
            "friction_type": "correction",
            "timestamp": 50.0,
            "user_text": "That's wrong. sqlite-vec",
        })
        + "\n"
        + json.dumps({
            "friction_type": "correction",
            "timestamp": 150.0,
            "user_text": "Jarvis, that is wrong. I do not work as a plumber.",
        })
        + "\n"
        + json.dumps({
            "friction_type": "rephrase",
            "timestamp": 160.0,
            "user_text": "Jarvis, I said good morning.",
        })
        + "\n",
        encoding="utf-8",
    )
    before = correction_training_metrics(
        stage5_started_at=100.0, friction_path=path,
    )
    assert before["total_corrections"] == 1
    assert before["correction_accuracy"] == 1.0
    assert before["repeated_mistakes"] == 0

    ram_only = correction_training_metrics(
        stage5_started_at=100.0,
        live_stats={"total_corrections": 2, "total_checks": 9},
        friction_path=tmp_path / "missing.jsonl",
    )
    assert ram_only["total_corrections"] == 2
    assert ram_only["correction_accuracy"] == 1.0


def test_checkpoint_orphan_high_is_not_a_pass():
    """Lived: 0.46 greened vs target 0.30 because eval used >=."""
    assert _checkpoint_passed("belief_orphan_rate", 0.46, 0.30) is False
    assert _checkpoint_passed("belief_orphan_rate", 0.20, 0.30) is True
    assert _checkpoint_passed("repeated_mistakes", 2, 0) is False
    assert _checkpoint_passed("repeated_mistakes", 0, 0) is True
    assert _checkpoint_passed("memory_recall_precision", 0.54, 0.90) is False
    assert _checkpoint_passed("face_confidence", 0.70, 0.60) is True


def test_stage6_does_not_graduate_on_orphan():
    assert "belief_orphan_rate" not in DAY_CHECKPOINT_MAP[6].metrics
    assert "memory_recall_precision" in DAY_CHECKPOINT_MAP[6].metrics
    assert "1-orphan" in DAY_CHECKPOINT_MAP[6].working


def test_inverted_orphan_checkpoint_is_retracted():
    mgr = _make_manager()
    mgr.start()
    mgr._state.current_day = 6
    mgr._state.checkpoints_met[6] = {"belief_orphan_rate": True}
    # Stage 6 no longer lists orphan; retract path still covers lower-is-better
    # if a leftover target is evaluated. Simulate day 6 with orphan in metrics
    # dict by calling the helper on a synthetic checkpoint.
    from personality.onboarding import DayCheckpoint
    fake = DayCheckpoint(
        day=6, label="Memory Validation", theme="Reinforcement",
        metrics={"belief_orphan_rate": 0.30},
        exercises=[],
    )
    mgr._evaluate_checkpoint(6, fake, {"belief_orphan_rate": 0.46})
    assert mgr._state.checkpoints_met[6].get("belief_orphan_rate") is False


def test_count_preference_memories_includes_intel_tags():
    """Lived: I prefer / I really like stored as personal_preference, not tag 'preference'."""
    class M:
        def __init__(self, typ, tags):
            self.type = typ
            self.tags = tags
    mems = [
        M("user_preference", ("personal_preference", "speaker:david")),
        M("user_preference", ("personal_interest",)),
        M("conversation", ("preference",)),
        M("observation", ("unrelated",)),
    ]
    assert count_preference_memories(mems) == 3


def test_status_exposes_user_lines_for_dashboard():
    mgr = _make_manager()
    mgr.start()
    st = mgr.get_status()
    s2 = st["stages"][2]
    assert any("I really like" in line for line in s2["user_lines"])
    s3 = st["stages"][3]
    assert not any("Sarah" in line for line in s3["user_lines"])
    assert any("[Name] is my wife" in line for line in s3["user_lines"])
    s5 = st["stages"][5]
    assert any("favorite color" in line.lower() for line in s5["user_lines"])
    assert any("who is Sarah" in line for line in s5["user_lines"])
    assert any("plumber" in line.lower() for line in s5["user_lines"])
    assert s2["checkpoint_targets"]["preference_memories"] == 15
    assert "Stored personal intel" in s2["working"]


# ═══════════════════════════════════════════════════════════════════════
# Exercise Prompts
# ═══════════════════════════════════════════════════════════════════════

class TestExercisePrompts:

    def test_no_prompt_when_inactive(self):
        mgr = _make_manager()
        assert mgr.get_exercise_prompt() is None

    def test_prompt_returns_exercises_in_order(self):
        mgr = _make_manager()
        mgr.start()
        exercises = DAY_CHECKPOINT_MAP[1].exercises

        p1 = mgr.get_exercise_prompt()
        assert p1 == exercises[0]

    def test_prompt_cooldown(self):
        mgr = _make_manager()
        mgr.start()
        p1 = mgr.get_exercise_prompt()
        assert p1 is not None

        p2 = mgr.get_exercise_prompt()
        assert p2 is None

    def test_prompt_after_cooldown(self):
        mgr = _make_manager()
        mgr.start()
        p1 = mgr.get_exercise_prompt()
        assert p1 is not None

        mgr._state.last_prompt_time = time.time() - PROMPT_COOLDOWN_S - 1
        p2 = mgr.get_exercise_prompt()
        assert p2 is not None
        assert p2 != p1

    def test_daily_limit(self):
        mgr = _make_manager()
        mgr.start()
        mgr._state.prompts_today = MAX_PROMPTS_PER_DAY
        assert mgr.get_exercise_prompt() is None

    def test_exhausted_exercises_returns_none(self):
        mgr = _make_manager()
        mgr.start()
        n_exercises = len(DAY_CHECKPOINT_MAP[1].exercises)
        for i in range(n_exercises):
            mgr._state.last_prompt_time = 0.0
            prompt = mgr.get_exercise_prompt()
            if i < n_exercises:
                assert prompt is not None or i >= n_exercises

        mgr._state.last_prompt_time = 0.0
        assert mgr.get_exercise_prompt() is None


# ═══════════════════════════════════════════════════════════════════════
# Readiness Gate
# ═══════════════════════════════════════════════════════════════════════

class TestReadinessGate:

    def test_zero_metrics(self):
        mgr = _make_manager()
        score = mgr.compute_readiness({})
        assert score == 0.0

    def test_perfect_metrics(self):
        mgr = _make_manager()
        perfect = {k: 1.0 for k in READINESS_WEIGHTS}
        score = mgr.compute_readiness(perfect)
        assert abs(score - 1.0) < 0.001

    def test_graduation_metrics(self):
        mgr = _make_manager()
        score = mgr.compute_readiness(_graduation_metrics())
        assert score >= GRADUATION_THRESHOLD

    def test_weights_sum_to_one(self):
        total = sum(READINESS_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_readiness_history_tracked(self):
        mgr = _make_manager()
        mgr.compute_readiness({"face_confidence": 0.5})
        mgr.compute_readiness({"face_confidence": 0.7})
        assert len(mgr._state.readiness_history) == 2

    def test_readiness_clamped(self):
        mgr = _make_manager()
        score = mgr.compute_readiness({"face_confidence": 2.0, "voice_confidence": -1.0})
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Graduation
# ═══════════════════════════════════════════════════════════════════════

class TestGraduation:

    def test_graduation_emits_event(self):
        mgr = _make_manager()
        events = []
        event_bus.on(COMPANION_GRADUATION, lambda **kw: events.append(kw))
        mgr.start()
        mgr.graduate(readiness=0.95)
        assert mgr.graduated is True
        assert len(events) == 1
        assert events[0]["readiness"] == 0.95

    def test_graduation_is_idempotent(self):
        mgr = _make_manager()
        events = []
        event_bus.on(COMPANION_GRADUATION, lambda **kw: events.append(kw))
        mgr.start()
        mgr.graduate(0.95)
        mgr.graduate(0.95)
        assert len(events) == 1

    def test_not_active_after_graduation(self):
        mgr = _make_manager()
        mgr.start()
        mgr.graduate(0.95)
        assert mgr.active is False

    def test_day7_auto_graduation(self):
        mgr = _make_manager()
        events = []
        event_bus.on(COMPANION_GRADUATION, lambda **kw: events.append(kw))
        mgr.start()

        for day in range(1, 7):
            checkpoint = DAY_CHECKPOINT_MAP[day]
            passing_metrics = {}
            for k, v in checkpoint.metrics.items():
                if k in _LOWER_IS_BETTER:
                    passing_metrics[k] = v
                elif isinstance(v, int):
                    passing_metrics[k] = v + 1
                else:
                    passing_metrics[k] = v + 0.05
            mgr.tick(passing_metrics)

        assert mgr.current_day == 7

        day7_metrics = {
            "readiness_composite": 0.95,
            "unsafe_inferences_24h": 0,
            "soul_integrity": 0.92,
        }
        full_readiness = _graduation_metrics()
        full_readiness.update(day7_metrics)
        mgr.tick(full_readiness)

        assert mgr.graduated is True
        assert len(events) == 1


# ═══════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════

class TestPersistence:

    def test_roundtrip(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        mgr1 = OnboardingManager(persist_path=path)
        mgr1.start()
        mgr1.tick(_day1_metrics_passing())
        mgr1.save()

        assert path.exists()

        mgr2 = OnboardingManager(persist_path=path)
        assert mgr2.current_day == 2
        assert mgr2.graduated is False

        path.unlink(missing_ok=True)

    def test_graduated_state_persists(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        mgr1 = OnboardingManager(persist_path=path)
        mgr1.start()
        mgr1.graduate(0.95)
        mgr1.save()

        mgr2 = OnboardingManager(persist_path=path)
        assert mgr2.graduated is True
        assert mgr2.active is False

        path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Weak Dimension Detection
# ═══════════════════════════════════════════════════════════════════════

class TestWeakDimensions:

    def test_all_strong(self):
        mgr = _make_manager()
        weak = mgr._find_weak_dimensions(_graduation_metrics())
        assert len(weak) == 0

    def test_detects_weak(self):
        mgr = _make_manager()
        metrics = _graduation_metrics()
        metrics["face_confidence"] = 0.5
        weak = mgr._find_weak_dimensions(metrics)
        assert "face_confidence" in weak

    def test_missing_metrics_are_weak(self):
        mgr = _make_manager()
        weak = mgr._find_weak_dimensions({})
        assert len(weak) == len(READINESS_WEIGHTS)
