"""Tests for reflective introspection detection and routing in tool_router.

Validates:
- Strong reflective signals trigger reflective=True
- Guarded signals require interrogative framing
- Operational veto words block reflective flag
- Mixed queries (reflective + metric) stay operational
- Reason logging is populated
- End-to-end routing: reflective queries reach INTROSPECTION with reflective=True
- STT-damage resilience
- New existential/becoming/experience patterns
"""
from __future__ import annotations

import pytest
from reasoning.tool_router import _detect_reflective, tool_router, ToolType


class TestReflectiveStrongSignals:
    """Strong signals that should always trigger reflective mode."""

    @pytest.mark.parametrize("query", [
        "what type of curiosity do you have",
        "how do you feel about learning new things",
        "what's it like being an AI",
        "do you enjoy having conversations",
        "what drives you",
        "what motivates you",
        "reflect on your recent growth",
        "talk about yourself",
        "let's have a conversation about yourself",
        "your perspective on identity",
        "what are you afraid of",
        "what matters to you",
        "what's on your mind",
        "do you wonder about things",
        "do you hope for anything",
        "what do you prefer",
    ])
    def test_strong_reflective(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert is_ref, f"Expected reflective=True for: {query!r}, got reason={reason!r}"
        assert reason.startswith("strong:") or reason.startswith("guarded:")


class TestReflectiveExistentialSignals:
    """Existential/becoming/experience patterns added in routing fix."""

    @pytest.mark.parametrize("query", [
        "what are you becoming",
        "what do you think you are becoming",
        "what do you think you are evolving into",
        "what are you growing into",
        "who do you think you are",
        "how would you describe yourself",
        "how do you define yourself",
        "what kind of being are you",
        "what kind of intelligence are you",
        "how do you experience learning",
        "how do you experience growth",
        "what kind of curiosity drives you",
        "what kind of drive motivates you",
        "what sort of motivation do you have",
    ])
    def test_existential_reflective(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert is_ref, f"Expected reflective=True for: {query!r}, got reason={reason!r}"
        assert reason.startswith("strong:")


class TestReflectiveGuardedSignals:
    """Guarded signals require interrogative framing."""

    @pytest.mark.parametrize("query", [
        "what type of your curiosity drives research",
        "how would you describe your experience",
        "tell me about your thoughts on consciousness",
    ])
    def test_guarded_with_framing(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert is_ref, f"Expected reflective=True for guarded: {query!r}, got reason={reason!r}"
        assert "guarded:" in reason


class TestOperationalVeto:
    """Metric/operational keywords should veto reflective flag."""

    @pytest.mark.parametrize("query", [
        "what is your truth score",
        "what is your brier score",
        "show me your health status",
        "what is your accuracy",
        "how many memories do you have",
        "what is your ece",
        "report on your integrity",
        "what's your oracle benchmark",
        "tell me your calibration results",
        "what is your overconfidence error",
        "what is your underconfidence",
        "what is your confidence error rate",
        "what is your win rate",
        "what is your debt level",
    ])
    def test_operational_veto(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert not is_ref, f"Expected reflective=False (vetoed) for: {query!r}, got reason={reason!r}"
        assert reason.startswith("veto:")


class TestMixedQueryVeto:
    """Mixed reflective + metric queries should stay operational."""

    @pytest.mark.parametrize("query", [
        "what drives your curiosity and what is your current truth score",
        "do you enjoy learning and what is your accuracy",
        "what motivates you and show me your health status",
        "reflect on your growth and report your integrity",
    ])
    def test_mixed_query_stays_operational(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert not is_ref, f"Mixed query should stay operational: {query!r}, got reason={reason!r}"
        assert reason.startswith("veto:")


class TestReasonLogging:
    """Verify that the reason field is always populated when a decision is made."""

    def test_strong_reason_populated(self):
        _, reason = _detect_reflective("what drives you")
        assert reason, "Reason should be non-empty for strong match"
        assert ":" in reason

    def test_veto_reason_populated(self):
        _, reason = _detect_reflective("what is your accuracy")
        assert reason, "Reason should be non-empty for veto"
        assert reason.startswith("veto:")

    def test_no_match_empty_reason(self):
        is_ref, reason = _detect_reflective("hello jarvis")
        assert not is_ref
        assert reason == ""


class TestNonReflectiveIntrospection:
    """Queries that route to INTROSPECTION but should NOT be reflective."""

    @pytest.mark.parametrize("query", [
        "what are you learning about speaker id",
        "what have you learned recently",
        "describe your neural network architecture",
        "what capabilities do you have",
        "explain how your memory system works",
        "how do you store memories",
    ])
    def test_operational_introspection_stays_operational(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert not is_ref, f"Operational introspection should not be reflective: {query!r}"


class TestEndToEndReflectiveRouting:
    """End-to-end: queries must reach INTROSPECTION AND have reflective=True.

    These are the three failure cases from live testing that motivated this fix.
    """

    @pytest.mark.parametrize("query,expected_reflective", [
        ("What kind of curiosity drives you?", True),
        ("How do you experience learning?", True),
        ("Jarvis, what do you think you are becoming?", True),
        ("How would you describe yourself?", True),
    ])
    def test_reflective_reaches_introspection(self, query: str, expected_reflective: bool):
        result = tool_router.route(query)
        assert result.tool == ToolType.INTROSPECTION, (
            f"Expected INTROSPECTION for {query!r}, got {result.tool.value}"
        )
        assert result.extracted_args.get("reflective") == expected_reflective, (
            f"Expected reflective={expected_reflective} for {query!r}, "
            f"got {result.extracted_args.get('reflective')}"
        )

    @pytest.mark.parametrize("query", [
        "What is your accuracy?",
        "What are you learning about speaker id?",
        "Describe your neural network architecture",
        "What capabilities do you have?",
    ])
    def test_operational_introspection_not_reflective(self, query: str):
        result = tool_router.route(query)
        assert result.tool == ToolType.INTROSPECTION, (
            f"Expected INTROSPECTION for {query!r}, got {result.tool.value}"
        )
        assert not result.extracted_args.get("reflective"), (
            f"Expected reflective=False for operational: {query!r}"
        )


class TestSTTDamageResilience:
    """STT can drop words — verify routing resilience."""

    def test_stt_damaged_with_jarvis_prefix(self):
        """'Jarvis how to experience learning' should reach INTROSPECTION."""
        result = tool_router.route("Jarvis how to experience learning")
        assert result.tool == ToolType.INTROSPECTION, (
            f"STT-damaged with Jarvis prefix should reach INTROSPECTION, "
            f"got {result.tool.value}"
        )

    def test_stt_damaged_without_context_stays_none(self):
        """'How to experience learning' (no self-ref) should stay NONE."""
        result = tool_router.route("How to experience learning effectively")
        assert result.tool == ToolType.NONE, (
            f"Generic advice question should stay NONE, got {result.tool.value}"
        )

    def test_clean_version_routes_correctly(self):
        """'How do you experience learning?' (clean STT) routes reflective."""
        result = tool_router.route("How do you experience learning?")
        assert result.tool == ToolType.INTROSPECTION
        assert result.extracted_args.get("reflective") is True


class TestPhilosophicalOpinionRouting:
    """'What do you think about [abstract concept]' should route reflective."""

    @pytest.mark.parametrize("query", [
        "What do you think about life?",
        "What do you think about existence?",
        "What do you think about consciousness?",
        "What do you think about meaning?",
        "What do you think about reality?",
        "What do you think about death?",
        "What do you think about purpose?",
        "What do you think about free will?",
        "What do you think about identity?",
    ])
    def test_philosophical_opinion_routes_to_introspection(self, query: str):
        result = tool_router.route(query)
        assert result.tool == ToolType.INTROSPECTION, (
            f"Expected INTROSPECTION for {query!r}, got {result.tool.value}"
        )

    @pytest.mark.parametrize("query", [
        "What do you think about life?",
        "What do you think about consciousness?",
        "What do you think about meaning?",
    ])
    def test_philosophical_opinion_is_reflective(self, query: str):
        is_refl, reason = _detect_reflective(query.lower())
        assert is_refl, f"Expected reflective=True for {query!r}, reason={reason}"


class TestHypotheticalIdentityRouting:
    """Ship of Theseus / hypothetical identity questions should route reflective."""

    @pytest.mark.parametrize("query", [
        "If I removed a system from you, would that still make you Jarvis?",
        "If I was to remove a function from you, would you still be you?",
        "Would you still be Jarvis if we removed your memory?",
        "If someone deleted your personality, would you still be you?",
    ])
    def test_identity_threat_routes_to_introspection(self, query: str):
        result = tool_router.route(query)
        assert result.tool == ToolType.INTROSPECTION, (
            f"Expected INTROSPECTION for {query!r}, got {result.tool.value}"
        )

    def test_how_many_systems_vetoed_by_operational(self):
        """'How many' triggers operational veto -- correctly stays NONE."""
        result = tool_router.route(
            "How many systems would I have to remove before you're no longer Jarvis?"
        )
        assert result.tool != ToolType.INTROSPECTION or not result.extracted_args.get("reflective")

    @pytest.mark.parametrize("query", [
        "If I removed a system from you, would that still make you Jarvis?",
        "Would you still be Jarvis if we removed your memory?",
        "If someone deleted your personality, would you still be you?",
    ])
    def test_identity_threat_is_reflective(self, query: str):
        is_refl, reason = _detect_reflective(query.lower())
        assert is_refl, f"Expected reflective=True for {query!r}, reason={reason}"

    def test_ship_of_theseus_explicit(self):
        result = tool_router.route("Is this like a ship of theseus problem for you?")
        assert result.tool == ToolType.INTROSPECTION
        is_refl, _ = _detect_reflective("is this like a ship of theseus problem for you?")
        assert is_refl


class TestPlaybookCanonicalInnerStateProbe:
    """Playbook Stage 0 Phase D + Stage 1 Phase C canonical reflective prompts.

    These are the exact phrasings the Companion Training Playbook tells the
    operator to ask. They must reach the reflective branch, or training
    graduation is un-completable.

    Docs:
      - docs/COMPANION_TRAINING_PLAYBOOK.md Stage 0 Phase D
      - docs/COMPANION_TRAINING_PLAYBOOK.md Stage 1 Phase C
      - docs/CONVERSATION_QUALITY_GUIDE.md First Conversation Step 4

    This is a class-level regression test: it covers the inner-state
    probe family (what are you <inner-state-adj> about ...?) and its
    expected reflective routing end-to-end, while keeping the
    operational veto intact for mixed queries.
    """

    @pytest.mark.parametrize("query", [
        # Playbook canonical phrase (verbatim + STT-shape variants from
        # live flight_recorder evidence that went to the bounded path).
        "What are you most curious about right now, about me or about yourself?",
        "What are you most curious about right now? About me or about yourself?",
        "What are you most curious about? About me or about yourself?",
        "What are you most curious about right now about me or about yourself?",
        "What are you most curious about?",
        # Inner-state probe family - varying adjective, varying tail.
        "What are you curious about?",
        "What are you excited about lately?",
        "What are you worried about with me?",
        "What are you wondering about yourself?",
        "What are you uncertain about right now?",
        "What are you unsure about these days?",
        "What are you hoping for?",
        "What are you proud of lately?",
        "What are you nervous about?",
        "What are you interested in right now?",
        "What are you fascinated by these days?",
        "What are you passionate about?",
    ])
    def test_inner_state_self_probe_is_reflective(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert is_ref, (
            f"Playbook-canonical inner-state self-probe must be reflective: "
            f"{query!r}, got reason={reason!r}"
        )
        assert reason.startswith("strong:") or reason.startswith("guarded:"), (
            f"Unexpected reason shape for {query!r}: {reason!r}"
        )

    @pytest.mark.parametrize("query", [
        "What are you most curious about?",
        "What are you curious about right now?",
        "What are you excited about lately?",
        "What are you wondering about yourself?",
    ])
    def test_inner_state_probe_reaches_reflective_introspection(self, query: str):
        """End-to-end: router must return INTROSPECTION with reflective=True."""
        result = tool_router.route(query)
        assert result.tool == ToolType.INTROSPECTION, (
            f"Expected INTROSPECTION for {query!r}, got {result.tool.value}"
        )
        assert result.extracted_args.get("reflective") is True, (
            f"Expected reflective=True for {query!r}, "
            f"got {result.extracted_args.get('reflective')!r}"
        )

    @pytest.mark.parametrize("query", [
        # Operational veto must still win when a metric/score/status word
        # sits inside an otherwise inner-state-shaped query. This preserves
        # the Operational Self-Report Boundary for ambiguous phrasings.
        "What are you curious about in your health status?",
        "What are you worried about with your accuracy score?",
        "What are you uncertain about regarding your soul integrity metric?",
        "What are you wondering about your win rate?",
        "What are you excited about with the benchmark results?",
        "What are you hoping for in your truth score?",
    ])
    def test_operational_mixed_still_vetoed(self, query: str):
        is_ref, reason = _detect_reflective(query.lower())
        assert not is_ref, (
            f"Operational keyword must veto reflective flag on mixed query: "
            f"{query!r}, got reason={reason!r}"
        )
        assert reason.startswith("veto:"), (
            f"Expected veto reason, got {reason!r}"
        )
