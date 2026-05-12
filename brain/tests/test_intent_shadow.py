"""Regression tests for the voice-intent shadow runner (P1.4 / Ship B).

Covers:
  - SHADOW level: NN predictions are recorded but NEVER alter the regex
    result (guard against accidental rewrite at the lowest level).
  - ADVISORY level: NONE-rescue rewrites only when regex returns NONE
    AND the NN is confident in a non-general bucket; non-NONE results
    pass through untouched.
  - PRIMARY level: high-confidence NN disagreement overrides a
    low-confidence regex result; agreement and below-margin cases pass
    through.
  - Promotion gate: requires teacher-sample threshold + dwell + rolling
    agreement; refuses to promote when any gate fails.
  - Auto-rollback: when rolling agreement falls below the floor, the
    runner rolls back one level and records a rollback entry.
  - Persistence: state survives a runner restart with the same
    state_path.
  - Tool-router wiring: the runner is invoked from
    ``ToolRouter.route()`` via ``_finalize`` and can rewrite the result.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from reasoning.intent_shadow import (
    AGREEMENT_WINDOW,
    BUCKETS,
    IntentShadowLevel,
    IntentShadowRunner,
    MIN_AGREEMENT_FOR_PROMOTION,
    MIN_DWELL_OBSERVATIONS_PER_LEVEL,
    MIN_TEACHER_SAMPLES,
    NN_PRIMARY_MARGIN,
    NN_PRIMARY_MIN_CONFIDENCE,
    NN_RESCUE_MIN_CONFIDENCE,
    ROLLBACK_AGREEMENT_FLOOR,
    set_intent_shadow_runner,
)
from reasoning.tool_router import RoutingResult, ToolType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(top_bucket: str, top_prob: float = 0.9) -> list[float]:
    """Construct a one-hot-ish logit vector with ``top_prob`` mass on
    ``top_bucket`` and the remainder spread evenly across the others."""
    n = len(BUCKETS)
    rest = (1.0 - top_prob) / (n - 1)
    return [top_prob if b == top_bucket else rest for b in BUCKETS]


def _make_runner(
    tmp_path: str, *, level: IntentShadowLevel | None = None,
    inference_fn=None, teacher_samples: int = 0,
) -> IntentShadowRunner:
    state_file = os.path.join(tmp_path, "intent_shadow_state.json")
    runner = IntentShadowRunner(
        state_path=state_file,
        inference_fn=inference_fn,
        teacher_sample_provider=lambda: teacher_samples,
    )
    if level is not None:
        runner._state.level = level.value
    return runner


def _none_result() -> RoutingResult:
    return RoutingResult(tool=ToolType.NONE, confidence=0.0, extracted_args={})


def _memory_result(conf: float = 0.7) -> RoutingResult:
    return RoutingResult(tool=ToolType.MEMORY, confidence=conf, extracted_args={})


# ---------------------------------------------------------------------------
# SHADOW level
# ---------------------------------------------------------------------------


class TestShadowLevel:

    def test_shadow_never_rewrites(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
            )
            new_result, obs = runner.observe_and_rewrite("remember this", _none_result())
            assert new_result.tool == ToolType.NONE
            assert obs.rewrote is False
            assert obs.level == IntentShadowLevel.SHADOW
            assert obs.nn_top_bucket == "memory"

    def test_shadow_records_agreement_stats(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
            )
            runner.observe(
                "remember the foo",
                RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
            )
            runner.observe(
                "what time is it",
                RoutingResult(tool=ToolType.TIME, confidence=0.9, extracted_args={}),
            )
            state = runner.get_state()
            assert state["observations_total"] == 2
            assert state["nn_predictions_total"] == 2
            # NN predicts "memory" for both; first agrees (MEMORY -> memory),
            # second disagrees (TIME -> status_ops).
            assert state["agreements_total"] == 1
            assert state["disagreements_total"] == 1
            assert state["disagreements_total"] == 1


# ---------------------------------------------------------------------------
# ADVISORY level (NONE-rescue)
# ---------------------------------------------------------------------------


class TestAdvisoryLevel:

    def test_advisory_rescues_none_when_nn_confident(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("memory", 0.9),
            )
            new_result, obs = runner.observe_and_rewrite(
                "remember the chicken sandwich", _none_result()
            )
            assert new_result.tool == ToolType.MEMORY
            assert obs.rewrote is True
            assert obs.rewrite_reason == "advisory_none_rescue"
            assert new_result.extracted_args.get("intent_shadow_rescued") is True
            assert runner.get_state()["rescues_applied"] == 1

    def test_advisory_does_not_touch_non_none_results(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("status_ops", 0.95),
            )
            new_result, obs = runner.observe_and_rewrite(
                "what is the time", _memory_result()
            )
            assert new_result.tool == ToolType.MEMORY
            assert obs.rewrote is False

    def test_advisory_skips_low_confidence_nn(self):
        with tempfile.TemporaryDirectory() as td:
            below = NN_RESCUE_MIN_CONFIDENCE - 0.1
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("memory", below),
            )
            new_result, obs = runner.observe_and_rewrite(
                "kinda maybe remember", _none_result()
            )
            assert new_result.tool == ToolType.NONE
            assert obs.rewrote is False

    def test_advisory_does_not_rescue_general_chat(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("general_chat", 0.99),
            )
            new_result, obs = runner.observe_and_rewrite(
                "hey jarvis what's up", _none_result()
            )
            assert new_result.tool == ToolType.NONE
            assert obs.rewrote is False


# ---------------------------------------------------------------------------
# PRIMARY level (full takeover)
# ---------------------------------------------------------------------------


class TestPrimaryLevel:

    def test_primary_overrides_low_confidence_disagreement(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.PRIMARY,
                inference_fn=lambda msg: _make_logits(
                    "memory", NN_PRIMARY_MIN_CONFIDENCE + 0.1
                ),
            )
            regex_result = RoutingResult(
                tool=ToolType.WEB_SEARCH, confidence=0.4, extracted_args={}
            )
            new_result, obs = runner.observe_and_rewrite("ambiguous query", regex_result)
            assert new_result.tool == ToolType.MEMORY
            assert obs.rewrote is True
            assert obs.rewrite_reason == "primary_override"
            assert runner.get_state()["primary_overrides_applied"] == 1

    def test_primary_passes_through_when_nn_agrees(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.PRIMARY,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
            )
            new_result, obs = runner.observe_and_rewrite(
                "remember foo", _memory_result(conf=0.95)
            )
            assert new_result.tool == ToolType.MEMORY
            assert obs.rewrote is False
            assert obs.agreed is True

    def test_primary_respects_margin(self):
        # NN is above the absolute threshold but does NOT beat regex by
        # the required margin -> no override.
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.PRIMARY,
                inference_fn=lambda msg: _make_logits("memory", NN_PRIMARY_MIN_CONFIDENCE),
            )
            regex_result = RoutingResult(
                tool=ToolType.WEB_SEARCH,
                confidence=NN_PRIMARY_MIN_CONFIDENCE - (NN_PRIMARY_MARGIN / 2),
                extracted_args={},
            )
            new_result, _ = runner.observe_and_rewrite("ambiguous", regex_result)
            assert new_result.tool == ToolType.WEB_SEARCH


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


class TestPromotionGates:

    def test_refuses_promotion_without_teacher_samples(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES - 1,
            )
            for _ in range(MIN_DWELL_OBSERVATIONS_PER_LEVEL + 5):
                runner.observe(
                    "remember this",
                    RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
                )
            assert runner.maybe_promote() is False
            assert runner.get_state()["level"] == IntentShadowLevel.SHADOW.value

    def test_refuses_promotion_without_dwell(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            runner.observe(
                "remember",
                RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
            )
            assert runner.maybe_promote() is False

    def test_refuses_promotion_without_agreement(self):
        with tempfile.TemporaryDirectory() as td:
            # NN consistently disagrees -> 0% agreement.
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("status_ops", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            for _ in range(MIN_DWELL_OBSERVATIONS_PER_LEVEL + 5):
                runner.observe(
                    "remember",
                    RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
                )
            assert runner.maybe_promote() is False

    def test_promotes_when_all_gates_pass(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            for _ in range(MIN_DWELL_OBSERVATIONS_PER_LEVEL + 5):
                runner.observe(
                    "remember",
                    RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
                )
            assert runner.maybe_promote() is True
            assert runner.get_state()["level"] == IntentShadowLevel.ADVISORY.value


# ---------------------------------------------------------------------------
# Auto-rollback
# ---------------------------------------------------------------------------


class TestAutoRollback:

    def test_rolls_back_when_agreement_collapses(self):
        with tempfile.TemporaryDirectory() as td:
            # Pre-fill the rolling window with disagreements.
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("status_ops", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            for _ in range(AGREEMENT_WINDOW):
                runner.observe(
                    "remember",
                    RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
                )
            state = runner.get_state()
            # After enough disagreements at advisory, auto-rollback fires.
            assert state["level"] == IntentShadowLevel.SHADOW.value
            assert state["last_rollback_ts"] is not None
            assert state["rollback_history"]
            assert state["rollback_history"][-1]["from_level"] == "advisory"

    def test_shadow_does_not_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("status_ops", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            for _ in range(AGREEMENT_WINDOW):
                runner.observe(
                    "remember",
                    RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
                )
            assert runner.get_state()["level"] == IntentShadowLevel.SHADOW.value


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as td:
            r1 = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
                teacher_samples=MIN_TEACHER_SAMPLES,
            )
            r1.observe(
                "remember",
                RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
            )
            state_before = r1.get_state()

            r2 = IntentShadowRunner(
                state_path=os.path.join(td, "intent_shadow_state.json")
            )
            state_after = r2.get_state()
            assert state_after["level"] == state_before["level"]
            assert state_after["observations_total"] == state_before["observations_total"]


# ---------------------------------------------------------------------------
# Tool-router wiring
# ---------------------------------------------------------------------------


class TestToolRouterIntegration:

    def teardown_method(self, method):
        # Always clear the singleton so tests don't bleed state.
        set_intent_shadow_runner(None)

    def test_router_invokes_runner_and_can_rewrite(self):
        from reasoning.tool_router import ToolRouter

        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
            )
            set_intent_shadow_runner(runner)

            router = ToolRouter()
            # Use a phrase that hits no Tier-0/1/2 patterns -> NONE
            # fallback; ADVISORY level should rescue it to MEMORY.
            result = router.route("akjsdh1928 random asdf")
            assert result.tool == ToolType.MEMORY
            assert result.extracted_args.get("intent_shadow_rescued") is True

    def test_router_passes_through_in_shadow_mode(self):
        from reasoning.tool_router import ToolRouter

        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.SHADOW,
                inference_fn=lambda msg: _make_logits("memory", 0.95),
            )
            set_intent_shadow_runner(runner)

            router = ToolRouter()
            result = router.route("akjsdh1928 random asdf")
            assert result.tool == ToolType.NONE


# ---------------------------------------------------------------------------
# Inference availability
# ---------------------------------------------------------------------------


class TestInferenceAvailability:

    def test_no_inference_fn_does_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=None,
            )
            new_result, obs = runner.observe_and_rewrite(
                "remember", _none_result()
            )
            assert new_result.tool == ToolType.NONE
            assert obs.nn_available is False
            assert obs.agreed is None

    def test_inference_returns_wrong_shape_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=lambda msg: [0.5, 0.5],  # only 2 dims
            )
            new_result, obs = runner.observe_and_rewrite(
                "remember", _none_result()
            )
            assert obs.nn_available is False
            assert new_result.tool == ToolType.NONE

    def test_inference_raising_is_swallowed(self):
        def boom(_msg):
            raise RuntimeError("nn down")

        with tempfile.TemporaryDirectory() as td:
            runner = _make_runner(
                td,
                level=IntentShadowLevel.ADVISORY,
                inference_fn=boom,
            )
            new_result, obs = runner.observe_and_rewrite(
                "remember", _none_result()
            )
            assert obs.nn_available is False
            assert new_result.tool == ToolType.NONE
