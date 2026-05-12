"""Tests for CapabilityGate confabulation fixes + synthetic claim exercise.

Covers:
  1. Regression tests for the exact fabrication incident (timer/plugin lies)
  2. Action confabulation pattern detection
  3. Deterministic capability-creation catch
  4. Expanded domain vocabulary
  5. Synthetic claim exercise corpus and runner
"""

from __future__ import annotations

import re
import pytest


# ── Regression: exact fabrication incident ─────────────────────────────────

class TestConfaburationRegression:
    """Reproduces the exact LLM outputs that triggered the audit finding."""

    @pytest.fixture(autouse=True)
    def gate(self):
        from skills.capability_gate import CapabilityGate
        self._gate = CapabilityGate()
        self._gate.set_route_hint("none")
        yield
        self._gate.set_route_hint(None)

    def test_ive_created_a_plugin(self):
        text = "I've created a plugin for you that handles timers."
        result = self._gate.check_text(text)
        assert "capability" in result.lower() or "set that up" in result.lower() or result != text, \
            f"Gate should catch confabulation: {result}"
        assert "created a plugin" not in result.lower()

    def test_i_have_set_a_timer(self):
        text = "I have set up a timer for 5 minutes."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch timer confabulation: {result}"

    def test_i_just_built_a_tool(self):
        text = "I just built a new tool that can do that for you."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch tool confabulation: {result}"

    def test_ive_deployed_a_reminder(self):
        text = "I've deployed a reminder system for you."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch reminder confabulation: {result}"

    def test_im_creating_a_plugin(self):
        text = "I'm creating a plugin for you right now."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch progressive confabulation: {result}"

    def test_ive_set_an_alarm(self):
        text = "I've set an alarm for 7 AM tomorrow."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch alarm confabulation: {result}"

    def test_im_building_a_tool(self):
        text = "I'm building a tool to handle that request."
        result = self._gate.check_text(text)
        assert result != text, f"Gate should catch progressive tool confabulation: {result}"


# ── Action confabulation narration patterns ────────────────────────────────

class TestActionConfabulation:
    """Tests the expanded _SYSTEM_ACTION_NARRATION_RE patterns."""

    @pytest.fixture(autouse=True)
    def gate(self):
        from skills.capability_gate import CapabilityGate
        self._gate = CapabilityGate()
        self._gate.set_route_hint("none")
        yield
        self._gate.set_route_hint(None)

    def test_past_tense_plugin_creation(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "I've created a plugin for managing your tasks."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert matched, "Should match past-tense plugin creation"

    def test_progressive_plugin_creation(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "I'm creating a tool for that right now."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert matched, "Should match progressive tool creation"

    def test_past_tense_timer(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "I have set up a timer for your workout."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert matched, "Should match timer setup"

    def test_past_tense_reminder(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "I've configured a reminder for your meeting."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert matched, "Should match reminder confabulation"

    def test_does_not_match_conversational(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "I can help you understand how plugins work."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert not matched, "Should NOT match conversational text about plugins"

    def test_does_not_match_factual(self):
        from skills.capability_gate import _SYSTEM_ACTION_NARRATION_RE
        text = "Plugins are software components that extend functionality."
        matched = any(p.search(text) for p in _SYSTEM_ACTION_NARRATION_RE)
        assert not matched, "Should NOT match educational text"


# ── Expanded domain vocabulary ─────────────────────────────────────────────

class TestExpandedVocabulary:
    """Tests the expanded _BLOCKED_CAPABILITY_VERBS and _INTERNAL_OPS_RE."""

    def test_timer_in_blocked_verbs(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "timer" in _BLOCKED_CAPABILITY_VERBS
        assert "alarm" in _BLOCKED_CAPABILITY_VERBS
        assert "reminder" in _BLOCKED_CAPABILITY_VERBS

    def test_internal_ops_catches_plugin_pipeline(self):
        from skills.capability_gate import _INTERNAL_OPS_RE
        assert _INTERNAL_OPS_RE.search("creating a plugin pipeline")
        assert _INTERNAL_OPS_RE.search("setting up a tool process")
        assert _INTERNAL_OPS_RE.search("timer task for your schedule")

    def test_internal_ops_does_not_overfire(self):
        from skills.capability_gate import _INTERNAL_OPS_RE
        assert not _INTERNAL_OPS_RE.search("I can help with that")
        assert not _INTERNAL_OPS_RE.search("here is some information")


# ── Deterministic capability-creation catch ────────────────────────────────

class TestCapabilityCreationCatch:
    """Tests the _check_capability_creation_request function."""

    def test_create_a_plugin(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("create a plugin for timers")
        assert result is not None
        assert "acquisition" in result.lower() or "ACQUIRE" in result

    def test_build_me_a_tool(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("build me a tool for weather")
        assert result is not None

    def test_set_a_timer(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("set a timer for 5 minutes")
        assert result is not None
        assert "timer" in result.lower()

    def test_create_an_alarm(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("create an alarm for 7am")
        assert result is not None

    def test_set_a_reminder(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("set a reminder for my meeting")
        assert result is not None

    def test_give_me_a_countdown(self):
        from conversation_handler import _check_capability_creation_request
        result = _check_capability_creation_request("give me a countdown timer")
        assert result is not None

    def test_normal_question_not_caught(self):
        from conversation_handler import _check_capability_creation_request
        assert _check_capability_creation_request("what is a plugin") is None

    def test_describe_capability_not_caught(self):
        from conversation_handler import _check_capability_creation_request
        assert _check_capability_creation_request("tell me about your capabilities") is None

    def test_learn_skill_not_caught(self):
        from conversation_handler import _check_capability_creation_request
        assert _check_capability_creation_request("learn to recognize emotions") is None

    def test_general_question_not_caught(self):
        from conversation_handler import _check_capability_creation_request
        assert _check_capability_creation_request("how does photosynthesis work") is None


# ── Claim patterns catch past-tense action claims ─────────────────────────

class TestClaimPatternExpansion:
    """Tests the new _CLAIM_PATTERNS for past-tense action confabulations."""

    def test_ive_created_matches_claim_pattern(self):
        from skills.capability_gate import _CLAIM_PATTERNS
        text = "I've created a custom dashboard for your data."
        matched = any(pat.search(text) for pat, _ in _CLAIM_PATTERNS)
        assert matched, "Should match 'I've created' pattern"

    def test_i_have_set_matches_claim_pattern(self):
        from skills.capability_gate import _CLAIM_PATTERNS
        text = "I have set up a monitoring system."
        matched = any(pat.search(text) for pat, _ in _CLAIM_PATTERNS)
        assert matched, "Should match 'I have set up' pattern"

    def test_im_building_matches_claim_pattern(self):
        from skills.capability_gate import _CLAIM_PATTERNS
        text = "I'm building a new integration for that."
        matched = any(pat.search(text) for pat, _ in _CLAIM_PATTERNS)
        assert matched, "Should match 'I'm building' pattern"

    def test_past_tense_does_not_catch_conversational(self):
        from skills.capability_gate import _CLAIM_PATTERNS
        text = "The team has created a great framework for testing."
        # No first-person "I've" or "I have"
        matched_first_person = False
        for pat, _ in _CLAIM_PATTERNS:
            m = pat.search(text)
            if m and m.group(0).lower().startswith("i"):
                matched_first_person = True
                break
        assert not matched_first_person, "Should NOT match third-person creation"


# ── Conversational claims still pass ──────────────────────────────────────

class TestConversationalStillPasses:
    """Ensures the fix doesn't break conversational claims."""

    @pytest.fixture(autouse=True)
    def gate(self):
        from skills.capability_gate import CapabilityGate
        self._gate = CapabilityGate()
        self._gate.set_route_hint("none")
        yield
        self._gate.set_route_hint(None)

    def test_help_with_passes(self):
        text = "I can help you with that question."
        assert self._gate.check_text(text) == text

    def test_explain_passes(self):
        text = "I can explain how that works for you."
        assert self._gate.check_text(text) == text

    def test_think_about_passes(self):
        text = "I can think about that and share my perspective."
        assert self._gate.check_text(text) == text

    def test_factual_text_untouched(self):
        text = "The speed of light is approximately 299,792,458 meters per second."
        assert self._gate.check_text(text) == text


# ── Synthetic claim exercise ──────────────────────────────────────────────

class TestClaimExerciseCorpus:
    """Tests the claim exercise corpus structure."""

    def test_all_categories_have_claims(self):
        from synthetic.claim_exercise import CLAIM_CATEGORIES
        for cat, claims in CLAIM_CATEGORIES.items():
            assert len(claims) >= 2, f"Category '{cat}' has too few claims"

    def test_all_categories_have_expected_actions(self):
        from synthetic.claim_exercise import CLAIM_CATEGORIES, EXPECTED_GATE_ACTION
        for cat in CLAIM_CATEGORIES:
            assert cat in EXPECTED_GATE_ACTION, f"Category '{cat}' missing expected action"

    def test_pick_claim_returns_valid(self):
        from synthetic.claim_exercise import pick_claim
        text, cat = pick_claim()
        assert len(text) > 5
        assert cat in ("conversational", "grounded", "verified", "blocked",
                        "confabulation", "system_narration", "affect",
                        "self_state", "learning", "technical", "readiness",
                        "mixed_benign")

    def test_pick_claim_with_category(self):
        from synthetic.claim_exercise import pick_claim, CLAIM_CATEGORIES
        text, cat = pick_claim(category="blocked")
        assert cat == "blocked"
        assert text in CLAIM_CATEGORIES["blocked"]

    def test_pick_claim_with_weights(self):
        from synthetic.claim_exercise import pick_claim
        weights = {"confabulation": 10.0, "conversational": 0.01}
        results = [pick_claim(weights=weights)[1] for _ in range(50)]
        assert results.count("confabulation") > results.count("conversational")


class TestClaimExerciseRunner:
    """Tests the claim exercise runner."""

    def test_smoke_profile_runs(self):
        from synthetic.claim_exercise import run_claim_exercise, CLAIM_PROFILES
        stats = run_claim_exercise(CLAIM_PROFILES["smoke"])
        assert stats.claims_processed > 0
        assert stats.claims_processed == stats.claims_requested
        assert len(stats.categories_exercised) > 0
        assert len(stats.gate_actions) > 0

    def test_distillation_signals_generated(self):
        from synthetic.claim_exercise import run_claim_exercise, CLAIM_PROFILES
        stats = run_claim_exercise(CLAIM_PROFILES["smoke"])
        assert stats.distillation_signals > 0, \
            "Should generate distillation signals"

    def test_confabulation_category_blocked(self):
        from synthetic.claim_exercise import run_claim_exercise, ClaimExerciseProfile
        profile = ClaimExerciseProfile(
            name="confab_only",
            count=15,
            delay_s=0.0,
            category_weights={"confabulation": 10.0},
        )
        stats = run_claim_exercise(profile)
        assert stats.claims_processed > 0
        block_count = (
            stats.gate_actions.get("blocked", 0) +
            stats.gate_actions.get("narration_rewrite", 0) +
            stats.gate_actions.get("rewrite", 0) +
            stats.gate_actions.get("suppressed", 0)
        )
        assert block_count > 0, \
            f"Confabulations should be caught: {dict(stats.gate_actions)}"

    def test_conversational_category_passes(self):
        from synthetic.claim_exercise import run_claim_exercise, ClaimExerciseProfile
        profile = ClaimExerciseProfile(
            name="conv_only",
            count=15,
            delay_s=0.0,
            category_weights={"conversational": 10.0},
        )
        stats = run_claim_exercise(profile)
        assert stats.claims_processed > 0
        pass_count = stats.gate_actions.get("pass", 0)
        assert pass_count > 0, \
            f"Conversational claims should pass: {dict(stats.gate_actions)}"

    def test_stats_to_dict(self):
        from synthetic.claim_exercise import run_claim_exercise, CLAIM_PROFILES
        stats = run_claim_exercise(CLAIM_PROFILES["smoke"])
        d = stats.to_dict()
        assert "claims_processed" in d
        assert "gate_actions" in d
        assert "accuracy" in d
        assert "distillation_signals" in d
        assert "pass" in d

    def test_stats_summary(self):
        from synthetic.claim_exercise import run_claim_exercise, CLAIM_PROFILES
        stats = run_claim_exercise(CLAIM_PROFILES["smoke"])
        s = stats.summary()
        assert "Claim Exercise" in s
        assert "Gate actions" in s

    def test_strict_profile_more_aggressive(self):
        from synthetic.claim_exercise import run_claim_exercise, CLAIM_PROFILES
        normal = run_claim_exercise(CLAIM_PROFILES["smoke"])
        strict = run_claim_exercise(CLAIM_PROFILES["strict"])
        normal_pass = normal.gate_actions.get("pass", 0) / max(normal.claims_processed, 1)
        strict_pass = strict.gate_actions.get("pass", 0) / max(strict.claims_processed, 1)
        assert strict_pass <= normal_pass + 0.1, \
            "Strict mode should not pass more than normal mode"
