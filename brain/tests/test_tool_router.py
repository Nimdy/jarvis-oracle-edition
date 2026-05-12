"""Tests for the tool router.

Gold probe regression suite: any user query about Jarvis's own state,
capabilities, learning, goals, systems, architecture, recent activity,
or self-description must never fall through to ToolType.NONE.

If a probe says NONE, the work is not done.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reasoning.tool_router import ToolRouter, ToolType


router = ToolRouter()


# ---------------------------------------------------------------------------
# Basic route smoke tests
# ---------------------------------------------------------------------------

def test_time_routing():
    result = router.route("What time is it?")
    assert result.tool == ToolType.TIME, f"Expected TIME, got {result.tool}"
    result2 = router.route("Tell me the current time please")
    assert result2.tool == ToolType.TIME


def test_system_routing():
    result = router.route("What is the system status?")
    assert result.tool == ToolType.SYSTEM_STATUS
    result2 = router.route("How much RAM is being used?")
    assert result2.tool == ToolType.SYSTEM_STATUS


def test_vision_routing():
    result = router.route("What do you see right now?")
    assert result.tool == ToolType.VISION
    result2 = router.route("Look around and describe the room")
    assert result2.tool == ToolType.VISION


def test_memory_routing():
    result = router.route("Do you remember what we talked about?")
    assert result.tool == ToolType.MEMORY


def test_memory_routing_personal_activity_recall():
    cases = [
        "What did I do yesterday?",
        "What did I do on Easter Sunday?",
        "Tell me what I did yesterday",
        "Tell me what I did Sunday",
        "What did I do Monday",
    ]
    for text in cases:
        result = router.route(text)
        assert result.tool == ToolType.MEMORY, f"{text!r} should route to MEMORY"


def test_recent_research_history_variants_route_to_introspection():
    cases = [
        "What was the last scientific journal you researched?",
        "What was the last peer-reviewed source you studied?",
        "What was the last peer-reviewed article you processed?",
        "What journal source did you study most recently?",
    ]
    for text in cases:
        result = router.route(text)
        assert result.tool == ToolType.INTROSPECTION, (
            f"{text!r} should route to INTROSPECTION, got {result.tool.value}"
        )


def test_completed_skill_learning_history_routes_to_introspection():
    cases = [
        "What skill did you just finish learning?",
        "What skill did you last finish learning?",
        "Jarvis, what skill did you just finish?",
        "What did you recently complete learning?",
    ]
    for text in cases:
        result = router.route(text)
        assert result.tool == ToolType.INTROSPECTION, (
            f"{text!r} should route to INTROSPECTION, got {result.tool.value}"
        )


def test_external_research_question_stays_academic_search():
    text = "What does research say about sleep and memory consolidation?"
    result = router.route(text)
    assert result.tool == ToolType.ACADEMIC_SEARCH


def test_look_up_recent_papers_routes_to_academic_search():
    text = (
        "Look up recent papers on Holographic Cognitation, HRR-VSA, "
        "see if you can find anything interesting."
    )
    result = router.route(text)
    assert result.tool == ToolType.ACADEMIC_SEARCH


def test_recent_research_preference_instruction_does_not_route_to_introspection():
    text = (
        "When I ask for your last peer research, do not include DOI unless I ask."
    )
    result = router.route(text)
    assert result.tool == ToolType.NONE, (
        f"{text!r} should route to NONE preference handling, got {result.tool.value}"
    )
    assert result.extracted_args.get("tier") == "preference_instruction"


def test_non_preference_complaint_does_not_hit_preference_instruction_tier():
    text = "Why don't you answer my question?"
    result = router.route(text)
    assert result.tool == ToolType.NONE
    assert result.extracted_args.get("tier") != "preference_instruction"


def test_identity_routing_explicit_phrases():
    assert router.route("Who am I?").tool == ToolType.IDENTITY
    assert router.route("Is this David?").tool == ToolType.IDENTITY


def test_identity_avoids_generic_am_i_false_positives():
    for text in [
        "What am I doing today?",
        "Am I ready?",
        "Am I speaking too fast?",
    ]:
        result = router.route(text)
        assert result.tool != ToolType.IDENTITY, (
            f"Generic 'am I' phrase should not route to IDENTITY: {text!r} -> {result.tool.value}"
        )


def test_skill_routing():
    for text, expected in [
        ("learn to sing", ToolType.SKILL),
        ("teach yourself robotics", ToolType.SKILL),
        ("learn how to draw", ToolType.SKILL),
        ("can you learn to sing", ToolType.SKILL),
    ]:
        result = router.route(text)
        assert result.tool == expected, f"{text!r}: expected {expected}, got {result.tool}"


def test_skill_vs_introspection():
    for text, expected in [
        ("can you learn?", ToolType.INTROSPECTION),
        ("are you learning?", ToolType.INTROSPECTION),
    ]:
        result = router.route(text)
        assert result.tool == expected, f"{text!r}: expected {expected}, got {result.tool}"


def test_skill_vs_identity():
    for text, expected in [
        ("learn my voice", ToolType.IDENTITY),
        ("remember my face", ToolType.IDENTITY),
    ]:
        result = router.route(text)
        assert result.tool == expected, f"{text!r}: expected {expected}, got {result.tool}"


def test_learning_job_routing():
    """Natural-language skill/learning-job requests must route to SKILL."""
    cases = [
        ("start a learning job to learn X", ToolType.SKILL),
        ("create a learning job for speaker identification", ToolType.SKILL),
        ("Alright Jarvis start a learning job to learn what I want", ToolType.SKILL),
        ("begin a training job", ToolType.SKILL),
        ("I need a new learning job", ToolType.SKILL),
        ("start a training job to improve emotion detection", ToolType.SKILL),
        ("learning job to learn how to recognize voices", ToolType.SKILL),
        ("Hey Jarvis start a learning job", ToolType.SKILL),
        ("okay jarvis start learning to sing", ToolType.SKILL),
        ("Jarvis, learn audio analysis.", ToolType.SKILL),
        ("Jarvis learn to code", ToolType.SKILL),
        ("Jarvis, train your speaker model", ToolType.SKILL),
    ]
    for text, expected in cases:
        result = router.route(text)
        assert result.tool == expected, (
            f"Learning job routing fail: {text!r}\n"
            f"  expected {expected.value}, got {result.tool.value}\n"
            f"  args: {result.extracted_args}"
        )


def test_learning_job_no_false_positives():
    """Phrases containing 'learning' that are NOT skill requests should not route to SKILL."""
    for text in [
        "What is machine learning?",
        "Tell me about deep learning architectures",
        "How does learning work in the brain?",
    ]:
        result = router.route(text)
        assert result.tool != ToolType.SKILL, (
            f"False positive: {text!r} routed to SKILL (should be NONE or INTROSPECTION)"
        )


def test_general_chat():
    result = router.route("Tell me a joke")
    assert result.tool == ToolType.NONE
    result2 = router.route("How does photosynthesis work?")
    assert result2.tool == ToolType.NONE


def test_explicit_web_search_command_prefix_routes_to_web_search():
    cases = [
        "web search, jarvis pi project",
        "Jarvis web search current events",
        "duckduckgo latest nvidia drivers",
    ]
    for text in cases:
        result = router.route(text)
        assert result.tool == ToolType.WEB_SEARCH, (
            f"Expected WEB_SEARCH for explicit command form: {text!r}, got {result.tool.value}"
        )
        assert result.extracted_args.get("tier") == "explicit_web_command"


def test_capability_contradiction_routes_to_introspection():
    text = (
        "You were just able to search the web right there, but previously "
        "you told me you didn't have that capability. So what is going on?"
    )
    result = router.route(text)
    assert result.tool == ToolType.INTROSPECTION
    assert result.extracted_args.get("intent") == "capability_clarification"
    assert result.extracted_args.get("domain") == "web_search"


def test_emergence_evidence_queries_route_to_introspection():
    cases = [
        "What do emergent behaviors mean on your dashboard?",
        "Are you conscious or is this digital life?",
        "What evidence do you have for internal thoughts?",
        "Does Level 7 prove sentience?",
    ]
    for text in cases:
        result = router.route(text)
        assert result.tool == ToolType.INTROSPECTION, (
            f"{text!r} should route to grounded INTROSPECTION, got {result.tool.value}"
        )
        assert result.extracted_args.get("intent") == "emergence_evidence"
        assert result.extracted_args.get("requires_grounded_answer") is True


def test_runtime_bridge_scope_does_not_change_router_contracts():
    """Phase D bridge is consumption-only; router intent boundaries stay stable."""
    introspection = router.route("What systems do you use?")
    status = router.route("How are you doing?")
    general = router.route("Write me a haiku about robots.")
    assert introspection.tool == ToolType.INTROSPECTION
    assert status.tool == ToolType.STATUS
    assert general.tool == ToolType.NONE


def test_golden_status_exact_route():
    result = router.route("Jarvis, GOLDEN COMMAND STATUS")
    assert result.tool == ToolType.STATUS
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "executed"
    assert result.golden_context is not None
    assert result.golden_context.command_id == "GW_STATUS"


def test_golden_research_web_exact_route():
    result = router.route("Jarvis, GOLDEN COMMAND RESEARCH WEB")
    assert result.tool == ToolType.WEB_SEARCH
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "executed"
    assert result.extracted_args.get("golden_command_id") == "GW_RESEARCH_WEB"
    assert result.extracted_args.get("golden_operation") == "research_web"


def test_golden_prefix_normalization():
    result = router.route("  Jarvis...   golden   command   status!!! ")
    assert result.tool == ToolType.STATUS
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "executed"


def test_golden_bare_prefix_is_not_global_default():
    result = router.route("GOLDEN COMMAND STATUS")
    assert result.extracted_args.get("tier") != "golden"


def test_golden_bare_prefix_allowed_when_explicitly_enabled():
    result = router.route("golden command status", golden_allow_bare_prefix=True)
    assert result.tool == ToolType.STATUS
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "executed"


def test_golden_internal_punctuation_normalization():
    result = router.route("Jarvis, golden command, memory, status.")
    assert result.tool == ToolType.MEMORY
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "executed"
    assert result.extracted_args.get("golden_canonical_body") == "MEMORY STATUS"


def test_golden_invalid_no_fallback():
    result = router.route("Jarvis, GOLDEN COMMAND statis")
    assert result.tool == ToolType.NONE
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "invalid"
    assert "accepted_commands" in result.extracted_args


def test_golden_near_miss_stays_normal_routing():
    result = router.route("Jarvis, do a golden command status")
    assert result.extracted_args.get("tier") != "golden"


def test_golden_execute_without_confirm_is_invalid():
    result = router.route("Jarvis, GOLDEN COMMAND SELF IMPROVE EXECUTE")
    assert result.tool == ToolType.NONE
    assert result.extracted_args.get("tier") == "golden"
    assert result.extracted_args.get("golden_status") == "invalid"


def test_golden_destructive_confirm_route():
    confirmed = router.route("Jarvis, GOLDEN COMMAND SELF IMPROVE EXECUTE CONFIRM")
    assert confirmed.tool == ToolType.SELF_IMPROVE
    assert confirmed.extracted_args.get("golden_status") == "executed"
    assert confirmed.extracted_args.get("golden_requires_confirmation") is False


# ---------------------------------------------------------------------------
# GOLD PROBE SET — locked regression contract
#
# Every probe has an exact expected route. If any of these change,
# someone broke the routing contract and must fix it before merging.
# ---------------------------------------------------------------------------

_GOLD_INTROSPECTION_PROBES: list[tuple[str, ToolType]] = [
    ("What was the last thing you studied?", ToolType.INTROSPECTION),
    ("What was the last thing you learned?", ToolType.INTROSPECTION),
    ("What have you been learning?", ToolType.INTROSPECTION),
    ("Why is your emotion detection learning job stuck?", ToolType.INTROSPECTION),
    ("What do you need from me for speaker identification?", ToolType.INTROSPECTION),
    ("Do you need anything from me to help you finish those learning jobs?", ToolType.INTROSPECTION),
    ("What systems do you use?", ToolType.INTROSPECTION),
    ("What are your systems?", ToolType.INTROSPECTION),
    ("What capabilities do you have?", ToolType.INTROSPECTION),
    ("Tell me about yourself.", ToolType.INTROSPECTION),
    ("How do you work?", ToolType.INTROSPECTION),
    ("Do you have any pending goals?", ToolType.INTROSPECTION),
    ("Tell me something you would talk about without hitting your default large language model.", ToolType.INTROSPECTION),
    ("What tools do you use?", ToolType.INTROSPECTION),
    ("What have you improved lately?", ToolType.INTROSPECTION),
    ("Tell me about your neural network.", ToolType.INTROSPECTION),
    ("What do you need me to upgrade?", ToolType.INTROSPECTION),
    ("What skills are you working on?", ToolType.INTROSPECTION),
    ("Give me a status of your Nero network.", ToolType.INTROSPECTION),
]

_GOLD_STATUS_PROBES: list[tuple[str, ToolType]] = [
    ("How are you doing?", ToolType.STATUS),
    ("How are you feeling?", ToolType.STATUS),
    ("How you doing?", ToolType.STATUS),
    ("How ya doing?", ToolType.STATUS),
    ("Hey Jarvis, how you doing?", ToolType.STATUS),
    ("Give me a status update.", ToolType.STATUS),
    ("Status report.", ToolType.STATUS),
    ("How's everything?", ToolType.STATUS),
    ("Are you okay?", ToolType.STATUS),
    ("What mode are you in?", ToolType.STATUS),
]

_GOLD_NONE_PROBES: list[str] = [
    "Help me with my homework.",
    "What is the weather in Tokyo?",
    "Tell me a joke.",
    "Can you help me write an email?",
    "What is quantum computing?",
    "Remind me to buy groceries.",
    "Can you help me debug this function?",
    "Tell me about Python decorators.",
    "How does photosynthesis work?",
]


def test_gold_introspection_probes():
    """Self-referential queries about Jarvis's state/history/capabilities
    must route to INTROSPECTION, never to NONE or STATUS."""
    for text, expected in _GOLD_INTROSPECTION_PROBES:
        result = router.route(text)
        assert result.tool == expected, (
            f"GOLD PROBE FAIL: {text!r}\n"
            f"  expected {expected.value}, got {result.tool.value}\n"
            f"  args: {result.extracted_args}"
        )


def test_gold_status_probes():
    """Operational/health/well-being queries must route to STATUS."""
    for text, expected in _GOLD_STATUS_PROBES:
        result = router.route(text)
        assert result.tool == expected, (
            f"GOLD PROBE FAIL: {text!r}\n"
            f"  expected {expected.value}, got {result.tool.value}\n"
            f"  args: {result.extracted_args}"
        )


def test_gold_none_probes():
    """External/general queries must NOT route to INTROSPECTION or STATUS."""
    for text in _GOLD_NONE_PROBES:
        result = router.route(text)
        assert result.tool not in (ToolType.INTROSPECTION, ToolType.STATUS), (
            f"GOLD PROBE FAIL (false positive): {text!r}\n"
            f"  expected NONE, got {result.tool.value}\n"
            f"  args: {result.extracted_args}"
        )


def test_self_referential_queries_never_route_to_none():
    """Superset check: every INTROSPECTION + STATUS probe must not be NONE."""
    all_self = [t for t, _ in _GOLD_INTROSPECTION_PROBES] + [t for t, _ in _GOLD_STATUS_PROBES]
    for text in all_self:
        result = router.route(text)
        assert result.tool != ToolType.NONE, (
            f"Self-query routed to NONE: {text!r} -> {result.tool.value}"
        )


# ---------------------------------------------------------------------------
# Disambiguation regression
# ---------------------------------------------------------------------------

def test_status_vs_introspection_disambiguation():
    """When a STATUS keyword matches but architecture nouns are present,
    the disambiguator must override to INTROSPECTION."""
    cases = [
        ("Give me a status of your Nero network.", ToolType.INTROSPECTION),
        ("How are your neural networks?", ToolType.INTROSPECTION),
        ("Status of your training.", ToolType.INTROSPECTION),
        ("Give me a status update.", ToolType.STATUS),
        ("How are you doing?", ToolType.STATUS),
        ("Status report.", ToolType.STATUS),
    ]
    for text, expected in cases:
        result = router.route(text)
        assert result.tool == expected, (
            f"Disambiguation fail: {text!r}\n"
            f"  expected {expected.value}, got {result.tool.value}"
        )


def test_system_status_vs_status_disambiguation():
    """Hardware words → SYSTEM_STATUS, holistic words → STATUS."""
    cases = [
        ("How much CPU is being used?", ToolType.SYSTEM_STATUS),
        ("What is the system status?", ToolType.SYSTEM_STATUS),
        ("System status report", ToolType.STATUS),
    ]
    for text, expected in cases:
        result = router.route(text)
        assert result.tool == expected, (
            f"System/Status disambiguation fail: {text!r}\n"
            f"  expected {expected.value}, got {result.tool.value}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        test_time_routing,
        test_system_routing,
        test_vision_routing,
        test_memory_routing,
        test_skill_routing,
        test_skill_vs_introspection,
        test_skill_vs_identity,
        test_general_chat,
        test_gold_introspection_probes,
        test_gold_status_probes,
        test_gold_none_probes,
        test_self_referential_queries_never_route_to_none,
        test_status_vs_introspection_disambiguation,
        test_system_status_vs_status_disambiguation,
    ]
    passed = failed = 0
    print("\n=== Tool Router Tests ===\n")
    for fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS: {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {fn.__name__}\n        {e}")
        except Exception:
            failed += 1
            print(f"  ERROR: {fn.__name__}")
            traceback.print_exc()
    print(f"\n  {passed}/{passed + failed} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        raise SystemExit(1)
    print("\n")
