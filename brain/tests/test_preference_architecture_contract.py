"""Architecture contract tests for response-preference handling.

These tests lock the intended data flow:
1) Router classifies preference instructions as preference intent (NONE route).
2) Conversation layer recognizes preference content as response_style data.
3) Strict-native recent-learning applies DOI policy from query + stored preference.
4) Capability gate allows preference acknowledgements without capability rewrites.
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Allow importing conversation_handler/response without the optional ollama dep.
_ollama_stub = types.ModuleType("ollama")
_ollama_stub.AsyncClient = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ChatResponse = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ResponseError = Exception  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", _ollama_stub)

_aiohttp_stub = types.ModuleType("aiohttp")
_aiohttp_stub.ClientSession = mock.MagicMock  # type: ignore[attr-defined]
_aiohttp_stub.ClientTimeout = mock.MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("aiohttp", _aiohttp_stub)

from conversation_handler import _build_preference_instruction_ack, _extract_personal_intel
from reasoning.tool_router import ToolRouter, ToolType
from skills.capability_gate import CapabilityGate, _normalize_punctuation
from skills.registry import SkillRegistry, _default_skills
from tools.introspection_tool import _resolve_doi_output_policy


def _fresh_registry() -> SkillRegistry:
    reg = SkillRegistry(path="/dev/null")
    reg._skills = {r.skill_id: r for r in _default_skills()}
    reg._loaded = True
    reg.save = lambda: None  # type: ignore[assignment]
    return reg


def test_router_classifies_preference_instruction_as_none_intent() -> None:
    router = ToolRouter()
    text = "When I ask for your last peer research, do not include DOI unless I ask."
    result = router.route(text)
    assert result.tool == ToolType.NONE
    assert result.extracted_args.get("tier") == "preference_instruction"


def test_router_keeps_recent_research_query_on_introspection() -> None:
    router = ToolRouter()
    text = "What was the last peer-reviewed source you studied?"
    result = router.route(text)
    assert result.tool == ToolType.INTROSPECTION


def test_conversation_preference_extractor_marks_response_style_without_writing() -> None:
    text = "When answering last peer-reviewed or research questions, do not include DOI unless I ask."
    result = _extract_personal_intel(text, speaker="david", suppress_write=True)
    assert result["stored"] == 0
    assert "response_style" in result["personal_categories"]


def test_preference_ack_is_deterministic_and_specific_for_doi() -> None:
    text = "Do not include DOI in research answers unless I ask."
    saved = _build_preference_instruction_ack(text, stored_count=1)
    exists = _build_preference_instruction_ack(text, stored_count=0)
    assert "Preference saved" in saved
    assert "already stored" in exists
    assert "DOI is omitted" in saved
    assert "DOI is omitted" in exists


def test_doi_policy_priority_explicit_query_overrides_stored_omit_preference() -> None:
    with patch("tools.introspection_tool._resolve_doi_display_preference", return_value="omit"):
        include_doi, reason = _resolve_doi_output_policy(
            "What is the DOI for the last peer-reviewed source you studied?"
        )
    assert include_doi is True
    assert reason == "query_requested"


def test_doi_policy_uses_stored_preference_when_query_is_not_explicit() -> None:
    with patch("tools.introspection_tool._resolve_doi_display_preference", return_value="omit"):
        include_doi_omit, reason_omit = _resolve_doi_output_policy(
            "What was the last peer-reviewed source you studied?"
        )
    assert include_doi_omit is False
    assert reason_omit == "preference_omit"

    with patch("tools.introspection_tool._resolve_doi_display_preference", return_value="include"):
        include_doi_inc, reason_inc = _resolve_doi_output_policy(
            "What was the last peer-reviewed source you studied?"
        )
    assert include_doi_inc is True
    assert reason_inc == "preference_include"


def test_capability_gate_allows_preference_alignment_ack_without_rewrite() -> None:
    gate = CapabilityGate(_fresh_registry())
    text = "I will keep answers concise unless you ask for details."
    out = gate.check_text(text)
    assert out == _normalize_punctuation(text)


def test_router_does_not_misclassify_non_preference_complaint() -> None:
    router = ToolRouter()
    result = router.route("Why don't you answer my question?")
    assert result.tool == ToolType.NONE
    assert result.extracted_args.get("tier") != "preference_instruction"


def test_stage4_routine_priority_phrases_are_personal_intel_not_tool_routes() -> None:
    router = ToolRouter()
    samples = [
        "Jarvis, my top priorities right now are open source release and dashboard validation.",
        "Jarvis, my morning routine is coffee, dashboard check, then coding.",
        "Jarvis, right now I am focused on finishing companion training.",
        "Jarvis, do not interrupt me while I am coding.",
    ]

    for text in samples:
        routed = router.route(text)
        extracted = _extract_personal_intel(text, speaker="david", suppress_write=True)
        assert routed.tool == ToolType.NONE
        assert "routine_priority" in extracted["personal_categories"]


def test_stage4_interrupt_preference_remains_preference_instruction_when_applicable() -> None:
    router = ToolRouter()
    text = "Jarvis, I prefer you do not interrupt me while I am coding."

    routed = router.route(text)
    extracted = _extract_personal_intel(text, speaker="david", suppress_write=True)

    assert routed.tool == ToolType.NONE
    assert routed.extracted_args.get("tier") == "preference_instruction"
    assert "personal_preference" in extracted["personal_categories"]

