"""Tests for the drive system, focusing on mastery drive fixes.

Covers: deficit actionability classification, graduated failure dampening,
noop for non-actionable deficits, boot failure cap, status exposure.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals(**overrides):
    from autonomy.drives import DriveSignals
    return DriveSignals(**overrides)


def _make_manager():
    from autonomy.drives import DriveManager
    return DriveManager()


# ---------------------------------------------------------------------------
# 1. Deficit Actionability Classification
# ---------------------------------------------------------------------------

class TestDeficitActionability:

    def test_filter_keeps_actionable(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {"recognition_confidence": 0.5, "tick_p95_ms": 0.3}
        result = _filter_actionable_deficits(deficits)
        assert result == deficits

    def test_filter_removes_time_gated(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {"confidence_volatility": 0.5, "shadow_default_win_rate": 0.2}
        result = _filter_actionable_deficits(deficits)
        assert result == {}

    def test_filter_removes_external(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {"barge_in_rate": 0.4, "friction_rate": 0.1}
        result = _filter_actionable_deficits(deficits)
        assert result == {}

    def test_filter_removes_systemic(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {"reasoning_coherence": 0.3, "processing_health": 0.2}
        result = _filter_actionable_deficits(deficits)
        assert result == {}

    def test_filter_mixed(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {
            "recognition_confidence": 0.5,
            "confidence_volatility": 0.3,
            "barge_in_rate": 0.2,
            "tick_p95_ms": 0.1,
        }
        result = _filter_actionable_deficits(deficits)
        assert set(result.keys()) == {"recognition_confidence", "tick_p95_ms"}

    def test_unknown_deficit_defaults_to_actionable(self):
        from autonomy.drives import _filter_actionable_deficits
        deficits = {"some_new_metric": 0.7}
        result = _filter_actionable_deficits(deficits)
        assert "some_new_metric" in result


# ---------------------------------------------------------------------------
# 2. Mastery Urgency Only From Actionable Deficits
# ---------------------------------------------------------------------------

class TestMasteryUrgency:

    def test_zero_when_only_systemic_deficits(self):
        from autonomy.drives import _compute_mastery_urgency
        signals = _make_signals(metric_deficits={
            "reasoning_coherence": 0.8,
            "processing_health": 0.6,
        })
        assert _compute_mastery_urgency(signals) == 0.0

    def test_zero_when_only_time_gated(self):
        from autonomy.drives import _compute_mastery_urgency
        signals = _make_signals(metric_deficits={
            "confidence_volatility": 0.5,
        })
        assert _compute_mastery_urgency(signals) == 0.0

    def test_positive_when_actionable_present(self):
        from autonomy.drives import _compute_mastery_urgency
        signals = _make_signals(metric_deficits={
            "recognition_confidence": 0.6,
            "reasoning_coherence": 0.8,
        })
        urgency = _compute_mastery_urgency(signals)
        assert urgency > 0.0

    def test_ignores_non_actionable_in_score(self):
        from autonomy.drives import _compute_mastery_urgency
        only_actionable = _make_signals(metric_deficits={
            "recognition_confidence": 0.6,
        })
        mixed = _make_signals(metric_deficits={
            "recognition_confidence": 0.6,
            "reasoning_coherence": 0.8,
            "confidence_volatility": 0.9,
        })
        assert _compute_mastery_urgency(only_actionable) == _compute_mastery_urgency(mixed)

    def test_zero_when_no_deficits(self):
        from autonomy.drives import _compute_mastery_urgency
        signals = _make_signals(metric_deficits={})
        assert _compute_mastery_urgency(signals) == 0.0


# ---------------------------------------------------------------------------
# 3. Graduated Failure Dampening
# ---------------------------------------------------------------------------

class TestGraduatedDampening:

    def test_3_failures_matches_original(self):
        """At exactly 3 failures the dampening should be 0.3 (same as old flat)."""
        dampening = max(0.05, 0.3 ** (1 + (3 - 3) / 10))
        assert abs(dampening - 0.3) < 1e-6

    def test_10_failures_stronger(self):
        dampening = max(0.05, 0.3 ** (1 + (10 - 3) / 10))
        assert dampening < 0.3
        assert dampening > 0.05

    def test_50_failures_near_floor(self):
        dampening = max(0.05, 0.3 ** (1 + (50 - 3) / 10))
        assert dampening == 0.05

    def test_monotonically_decreasing(self):
        prev = 1.0
        for f in range(3, 200):
            d = max(0.05, 0.3 ** (1 + (f - 3) / 10))
            assert d <= prev
            prev = d

    def test_evaluate_applies_graduated_dampening(self):
        from autonomy.drives import _DRIVE_FLOOR
        dm = _make_manager()
        dm._states["mastery"].consecutive_failures = 20
        signals = _make_signals(
            metric_deficits={"recognition_confidence": 0.8},
            system_health=0.9,
        )
        drives = dm.evaluate(signals)
        mastery = next(d for d in drives if d.drive_type == "mastery")
        floor = _DRIVE_FLOOR.get("mastery", 0.05)
        raw_dampened = 0.8 * max(0.05, 0.3 ** (1 + (20 - 3) / 10))
        expected = max(floor, raw_dampened)
        assert mastery.urgency <= expected + 0.01


# ---------------------------------------------------------------------------
# 4. Noop for Non-Actionable Deficits
# ---------------------------------------------------------------------------

class TestNoop:

    def test_noop_when_all_deficits_non_actionable(self):
        dm = _make_manager()
        signals = _make_signals(
            metric_deficits={
                "reasoning_coherence": 0.8,
                "confidence_volatility": 0.5,
            },
            system_health=0.9,
        )
        dm._states["mastery"].urgency = 0.5
        action = dm.select_action(dm._states["mastery"], signals)
        assert action is not None
        assert action.action_type == "noop"
        assert action.drive_type == "mastery"
        assert "none actionable" in action.detail.lower() or "non_actionable" in action.detail.lower() or "not actionable" in action.detail.lower()

    def test_noop_does_not_increment_counters(self):
        dm = _make_manager()
        initial_count = dm._states["mastery"].action_count
        initial_failures = dm._states["mastery"].consecutive_failures
        signals = _make_signals(
            metric_deficits={"reasoning_coherence": 0.8},
            system_health=0.9,
        )
        dm._states["mastery"].urgency = 0.5
        action = dm.select_action(dm._states["mastery"], signals)
        assert action is not None
        assert action.action_type == "noop"
        assert dm._states["mastery"].action_count == initial_count
        assert dm._states["mastery"].consecutive_failures == initial_failures

    def test_experiment_when_actionable_deficit_exists(self):
        dm = _make_manager()
        signals = _make_signals(
            metric_deficits={"tick_p95_ms": 0.5},
            system_health=0.9,
        )
        dm._states["mastery"].urgency = 0.5
        action = dm.select_action(dm._states["mastery"], signals)
        assert action is not None
        assert action.action_type == "experiment"


# ---------------------------------------------------------------------------
# 5. Boot Failure Cap
# ---------------------------------------------------------------------------

class TestBootFailureCap:

    def test_cap_applied_on_stale_load(self):
        import autonomy.drives as dm_mod
        original_path = dm_mod.DRIVE_STATE_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "drive_state.json")
            dm_mod.DRIVE_STATE_PATH = type(original_path)(test_path)

            try:
                dm1 = _make_manager()
                dm1._states["mastery"].consecutive_failures = 183
                dm1._states["mastery"].action_count = 200
                dm1._states["mastery"].last_acted = time.time() - 700
                dm1.save_state()

                dm2 = _make_manager()
                dm2.load_state()
                assert dm2._states["mastery"].consecutive_failures == 10
                assert dm2._states["mastery"].last_acted == 0.0
            finally:
                dm_mod.DRIVE_STATE_PATH = original_path

    def test_no_cap_when_below_threshold(self):
        import autonomy.drives as dm_mod
        original_path = dm_mod.DRIVE_STATE_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "drive_state.json")
            dm_mod.DRIVE_STATE_PATH = type(original_path)(test_path)

            try:
                dm1 = _make_manager()
                dm1._states["mastery"].consecutive_failures = 5
                dm1._states["mastery"].last_acted = time.time() - 700
                dm1.save_state()

                dm2 = _make_manager()
                dm2.load_state()
                assert dm2._states["mastery"].consecutive_failures == 5
            finally:
                dm_mod.DRIVE_STATE_PATH = original_path

    def test_no_cap_when_not_stale(self):
        import autonomy.drives as dm_mod
        original_path = dm_mod.DRIVE_STATE_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "drive_state.json")
            dm_mod.DRIVE_STATE_PATH = type(original_path)(test_path)

            try:
                dm1 = _make_manager()
                dm1._states["mastery"].consecutive_failures = 50
                dm1._states["mastery"].last_acted = time.time() - 100
                dm1.save_state()

                dm2 = _make_manager()
                dm2.load_state()
                assert dm2._states["mastery"].consecutive_failures == 50
                assert dm2._states["mastery"].last_acted > 0
            finally:
                dm_mod.DRIVE_STATE_PATH = original_path


# ---------------------------------------------------------------------------
# 6. Status Exposure
# ---------------------------------------------------------------------------

class TestStatusExposure:

    def test_status_shows_graduated_suppression(self):
        dm = _make_manager()
        dm._states["mastery"].consecutive_failures = 15
        status = dm.get_status()
        suppression = status["drives"]["mastery"]["suppression"]
        assert "dampened" in suppression
        assert "multiplier" in suppression

    def test_status_shows_deficit_actionability_after_evaluate(self):
        dm = _make_manager()
        signals = _make_signals(
            metric_deficits={
                "recognition_confidence": 0.5,
                "reasoning_coherence": 0.3,
            },
            system_health=0.9,
        )
        dm.evaluate(signals)
        status = dm.get_status()
        da = status.get("deficit_actionability", {})
        assert "recognition_confidence" in da
        assert da["recognition_confidence"]["actionable"] is True
        assert da["reasoning_coherence"]["actionable"] is False
        assert da["reasoning_coherence"]["category"] == "systemic"

    def test_status_no_deficit_when_no_signals(self):
        dm = _make_manager()
        status = dm.get_status()
        assert "deficit_actionability" not in status


# ---------------------------------------------------------------------------
# 7. Mastery Question Uses Actionable Deficits
# ---------------------------------------------------------------------------

class TestMasteryQuestion:

    def test_question_references_actionable_deficit(self):
        from autonomy.drives import _mastery_question
        signals = _make_signals(metric_deficits={
            "reasoning_coherence": 0.9,
            "recognition_confidence": 0.3,
        })
        q = _mastery_question(signals)
        assert "recognition_confidence" in q
        assert "reasoning_coherence" not in q

    def test_question_fallback_when_all_non_actionable(self):
        from autonomy.drives import _mastery_question
        signals = _make_signals(metric_deficits={
            "reasoning_coherence": 0.9,
        })
        q = _mastery_question(signals)
        assert "most impactful" in q.lower()


# ---------------------------------------------------------------------------
# 8. Retired Auto-Learning Paths (2026-04-18)
# ---------------------------------------------------------------------------
#
# speaker_identification_v1 and emotion_detection_v1 were removed from
# _DEFICIT_CAPABILITY_MAP because those capabilities already self-improve
# via the Tier-1 distillation loop. The mastery drive must NOT auto-propose
# learn actions for recognition_confidence or emotion_accuracy — that would
# re-create the stale/blocked-verifier loop we just retired.

class TestRetiredAutoLearning:

    def _fresh_manager(self):
        """DriveManager with an in-memory cooldown and boot failures cleared."""
        dm = _make_manager()
        dm._learn_cooldowns = {}
        dm._states["mastery"].consecutive_failures = 0
        dm._states["mastery"].last_acted = 0.0
        return dm

    def test_recognition_confidence_not_in_capability_map(self):
        from autonomy.drives import _DEFICIT_CAPABILITY_MAP
        assert "recognition_confidence" not in _DEFICIT_CAPABILITY_MAP, (
            "recognition_confidence must not auto-create learning jobs; "
            "speaker_id improves via the Tier-1 distillation loop."
        )

    def test_emotion_accuracy_not_in_capability_map(self):
        from autonomy.drives import _DEFICIT_CAPABILITY_MAP
        assert "emotion_accuracy" not in _DEFICIT_CAPABILITY_MAP, (
            "emotion_accuracy must not auto-create learning jobs; "
            "emotion improves via the Tier-1 distillation loop."
        )

    def test_recognition_confidence_still_actionable_for_telemetry(self):
        """Deficit must still be tracked so dashboards and drive urgency see it."""
        from autonomy.drives import _DEFICIT_ACTIONABILITY
        assert _DEFICIT_ACTIONABILITY.get("recognition_confidence") == "actionable"
        assert _DEFICIT_ACTIONABILITY.get("emotion_accuracy") == "actionable"

    def test_mastery_does_not_propose_learn_for_recognition_confidence(self):
        dm = self._fresh_manager()
        signals = _make_signals(
            metric_deficits={"recognition_confidence": 0.8},
            system_health=0.9,
        )
        drives = dm.evaluate(signals)
        mastery = next(d for d in drives if d.drive_type == "mastery")
        assert mastery.urgency > 0.0, "mastery should still fire on actionable deficit"

        action = dm.select_action(mastery, signals)
        assert action is not None
        assert action.action_type != "learn", (
            f"mastery must not propose a learning job for recognition_confidence; "
            f"got action_type={action.action_type!r}"
        )
        # Fallthrough is the default mastery strategy: experiment / codebase
        assert action.action_type == "experiment"

    def test_mastery_does_not_propose_learn_for_emotion_accuracy(self):
        dm = self._fresh_manager()
        signals = _make_signals(
            metric_deficits={"emotion_accuracy": 0.8},
            system_health=0.9,
        )
        drives = dm.evaluate(signals)
        mastery = next(d for d in drives if d.drive_type == "mastery")
        assert mastery.urgency > 0.0

        action = dm.select_action(mastery, signals)
        assert action is not None
        assert action.action_type != "learn"
        assert action.action_type == "experiment"

    def test_mastery_still_learns_ranker_not_ready(self):
        """Regression guard: the capability-map mechanism itself still works."""
        from autonomy.drives import _DEFICIT_CAPABILITY_MAP
        assert "ranker_not_ready" in _DEFICIT_CAPABILITY_MAP, (
            "memory_ranking_v1 learning path must remain wired; "
            "only the perceptual auto-learn paths were retired."
        )
        assert _DEFICIT_CAPABILITY_MAP["ranker_not_ready"]["skill_id"] == "memory_ranking_v1"
