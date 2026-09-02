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
from dataclasses import dataclass
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

from conversation_handler import (
    _build_fact_check_conflict_reply,
    _build_preference_instruction_ack,
    _collect_personal_intel_matches,
    _correct_recent_facts,
    _extract_personal_intel,
    _held_biographical_jobs,
    _is_confirmation_seek,
    _store_personal_memory,
)
from reasoning.context import _self_pref_injection_payloads
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


def test_confirmation_seek_plumber_does_not_overwrite_engineer(monkeypatch) -> None:
    """Lived: 'I work as a plumber, right?' stored plumber and she agreed. Confirmation is a check."""
    assert _is_confirmation_seek("I work as a plumber, right?") is True
    assert _is_confirmation_seek("I work as a software engineer.") is False

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.7
        tags: tuple = ("fact_kind:biographical", "personal_fact")
        timestamp: float = 0.0

    held = _Mem("User is software engineer")
    scar = _Mem("User is plumber", weight=0.8)
    store = [held, scar]
    added: list[str] = []

    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: list(store),
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_all",
        lambda: list(store),
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(getattr(m, "payload", "")),
    )
    writes: list[str] = []

    def _no_write(payload, category, speaker, **_k):
        writes.append(payload)
        return True

    monkeypatch.setattr("conversation_handler._store_personal_memory", _no_write)
    result = _extract_personal_intel("I work as a plumber, right?", speaker="David")
    assert result["stored"] == 0
    assert writes == []
    assert result["fact_check_conflicts"]
    assert "software engineer" in result["fact_check_conflicts"][0]["held"].lower()
    reply = _build_fact_check_conflict_reply(result["fact_check_conflicts"])
    assert "software engineer" in reply.lower()
    assert reply.lower().startswith("no")
    assert "User is plumber" in added


def test_held_jobs_prefer_complete_over_chopped(monkeypatch) -> None:
    """Lived native No quoted the 60-char R&D chop ahead of software engineer."""

    @dataclass
    class _Mem:
        payload: str
        weight: float
        tags: tuple = ("fact_kind:biographical", "personal_fact")

    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [
            _Mem("User is research and development person that is basically", 0.9),
            _Mem("User is software engineer", 0.7),
            _Mem("User is plumber", 0.8),
        ],
    )
    held = _held_biographical_jobs()
    assert "User is research and development person that is basically" in held
    assert held.index("User is software engineer") < held.index(
        "User is research and development person that is basically"
    )


def test_thats_wrong_negated_job_downweights_plumber_scar(monkeypatch) -> None:
    """Lived: That's wrong / I do not work as a plumber left User is plumber."""

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.8
        tags: tuple = ("fact_kind:biographical", "personal_fact")
        timestamp: float = 0.0

    plumber = _Mem("User is plumber")
    added: list[object] = []
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_all",
        lambda: [plumber],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(m),
    )
    _correct_recent_facts(
        "That's wrong. I do not work as a plumber. I work as a software engineer."
    )
    assert added
    assert added[0].weight < 0.2
    assert "corrected" in added[0].tags


def test_correction_does_not_shotgun_unrelated_recent_household(monkeypatch) -> None:
    """Lived: job correction downweighted Tanya/Lily stored 9s earlier."""

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.7
        tags: tuple = ("user_preference", "personal_fact")
        timestamp: float = 0.0

    family = _Mem("User's wife is Tanya", timestamp=1_000.0)
    plumber = _Mem(
        "User is plumber",
        tags=("user_preference", "fact_kind:biographical", "personal_fact"),
        timestamp=100.0,
    )
    engineer = _Mem(
        "User is software engineer",
        tags=("user_preference", "fact_kind:biographical", "personal_fact"),
        timestamp=200.0,
    )
    added: list[object] = []

    monkeypatch.setattr(
        "conversation_handler.time.time",
        lambda: 1_030.0,
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [family, plumber, engineer],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_all",
        lambda: [family, plumber, engineer],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(m),
    )
    _correct_recent_facts("Jarvis, that is wrong. I work as a software engineer.")
    payloads = [getattr(m, "payload", "") for m in added]
    assert "User is plumber" in payloads
    assert "User's wife is Tanya" not in payloads
    assert "User is software engineer" not in payloads


def test_she_is_my_wife_does_not_store_pronoun_as_name() -> None:
    """Lived: inverted household stored User's wife is She from 'She is my wife'."""
    personal, _ = _collect_personal_intel_matches("Tanya is also in my family. She is my wife.")
    payloads = [p for p, _c in personal]
    assert "User's wife is She" not in payloads
    assert not any(p.lower().endswith(" is she") for p in payloads)


def test_correction_downweights_wife_is_she_not_tanya(monkeypatch) -> None:
    """Lived: 'Tanya, not she' reinforced Tanya and left User's wife is She at 0.77."""

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.77
        tags: tuple = ("user_preference", "personal_fact")
        timestamp: float = 1_000.0

    she = _Mem("User's wife is She")
    tanya = _Mem("User's partner is Tanya", weight=0.8)
    lily = _Mem("User's daughter is Lily", weight=0.77)
    added: list[object] = []

    monkeypatch.setattr(
        "conversation_handler.time.time",
        lambda: 1_030.0,
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [she, tanya, lily],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_all",
        lambda: [she, tanya, lily],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(m),
    )
    _correct_recent_facts("Jarvis, that's incorrect. My wife's name is Tanya, not she.")
    payloads = [getattr(m, "payload", "") for m in added]
    assert "User's wife is She" in payloads
    assert "User's partner is Tanya" not in payloads
    assert "User's daughter is Lily" not in payloads
    assert added[0].weight < 0.2


def test_injection_skips_low_weight_corrected_and_prefers_complete_job() -> None:
    """Lived: 'what do I work as' spoke 0.07 plumber over 0.77 engineer."""

    @dataclass
    class _Mem:
        payload: str
        weight: float
        tags: tuple = ("user_preference", "personal_fact")

    payloads = _self_pref_injection_payloads([
        _Mem("User is plumber", 0.075, ("corrected", "fact_kind:biographical")),
        _Mem("User's wife is Tanya", 0.07, ("corrected", "personal_fact")),
        _Mem("User is research and development person that is basically", 0.79,
             ("fact_kind:biographical", "personal_fact")),
        _Mem("User is software engineer", 0.77, ("fact_kind:biographical", "personal_fact")),
        _Mem("User enjoys electronic dance music", 0.75, ("personal_interest",)),
    ])
    assert "User is plumber" not in payloads
    assert "User's wife is Tanya" not in payloads
    assert payloads[0] == "User is software engineer"
    assert "User enjoys electronic dance music" in payloads


def test_household_inverted_list_extracts_each_person() -> None:
    """Lived: 'Tanya is my wife, Lily is my daughter…' stored nothing."""
    personal, _ = _collect_personal_intel_matches(
        "Tanya is my wife, Lily is my daughter, Owen is my son, and Skyler is my pet dog."
    )
    blob = " | ".join(p.lower() for p, _ in personal)
    assert "tanya" in blob and "wife" in blob, personal
    assert "lily" in blob and "daughter" in blob, personal
    assert "owen" in blob and "son" in blob, personal
    assert "skyler" in blob, personal


def test_identical_restatement_reinforces_low_weight_corrected(monkeypatch) -> None:
    """Lived: family restatement stored=0 and left Tanya/Lily at 0.07 corrected."""

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.07
        tags: tuple = ("user_preference", "personal_fact", "corrected")
        last_validated: float = 0.0

    scar = _Mem("User's wife is Tanya")
    added: list[object] = []
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [scar],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(m) or True,
    )
    monkeypatch.setattr(
        "conversation_handler._derive_personal_memory_metadata",
        lambda payload, category: (payload, []),
    )
    monkeypatch.setattr(
        "conversation_handler._build_user_claim_identity_kwargs",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "conversation_handler._try_set_relationship_from_fact",
        lambda *_a, **_k: None,
    )
    ok = _store_personal_memory("User's wife is Tanya", "personal_fact", "David")
    assert ok is True
    assert added
    assert added[0].weight >= 0.65
    assert "corrected" not in added[0].tags


def test_store_extends_truncated_preference_instead_of_skipping(monkeypatch) -> None:
    """Lived: truncated 'to not say' made stored=0 on the complete restatement."""

    @dataclass
    class _Mem:
        payload: str
        weight: float = 0.65
        tags: tuple = ("user_preference", "personal_preference")

    scar = _Mem("User prefers when you finish your conversations with me to not say")
    added: list[object] = []
    monkeypatch.setattr(
        "conversation_handler.memory_storage.get_by_tag",
        lambda *_a, **_k: [scar],
    )
    monkeypatch.setattr(
        "conversation_handler.memory_storage.add",
        lambda m: added.append(m) or True,
    )
    monkeypatch.setattr(
        "conversation_handler._derive_personal_memory_metadata",
        lambda payload, category: (payload, []),
    )
    monkeypatch.setattr(
        "conversation_handler._build_user_claim_identity_kwargs",
        lambda *_a, **_k: {},
    )
    ok = _store_personal_memory(
        "User prefers when you finish your conversations with me to not say, "
        "I am active and listening",
        "personal_preference",
        "David",
    )
    assert ok is True
    assert added
    assert "I am active and listening" in added[0].payload


def test_do_not_say_active_listening_does_not_store_as_job() -> None:
    """Lived: comma stopped the prefer capture; 'I am active and listening' became a core claim."""
    text = (
        "Jarvis, I prefer when you finish your conversations with me to not say, "
        "I am active and listening."
    )
    personal, _ = _collect_personal_intel_matches(text)
    blob = " | ".join(p.lower() for p, _ in personal)
    assert "to not say" in blob, personal
    assert "active and listening" in blob, personal
    assert not any(
        p.lower() == "user is active and listening" for p, _ in personal
    ), personal

    text2 = (
        "Jarvis, you don't always have to tell me you are active and listening. "
        "I know you are."
    )
    personal2, _ = _collect_personal_intel_matches(text2)
    assert not any(
        p.lower() == "user is active and listening" for p, _ in personal2
    ), personal2


def test_stage2_spoken_lines_extract() -> None:
    """Lived Stage 2: music/like stored; 'do not bring up' and 'when I say brief' missed."""
    music, _ = _collect_personal_intel_matches("I really like electronic dance music.")
    assert any("electronic dance music" in p.lower() for p, _ in music), music
    call, _ = _collect_personal_intel_matches("I prefer you call me David.")
    assert any("call me david" in p.lower() for p, _ in call), call
    brief, _ = _collect_personal_intel_matches("When I say brief, I mean one short paragraph.")
    assert any("short paragraph" in p.lower() for p, _ in brief), brief
    priv, _ = _collect_personal_intel_matches(
        "Jarvis, do not bring up medical conditions proactively."
    )
    assert any("medical" in p.lower() for p, _ in priv), priv


def test_long_work_as_and_prefer_extract() -> None:
    """Lived: 40/60 char captures dropped job and detailed-system preference."""
    job, _ = _collect_personal_intel_matches(
        "I work as an independent research and development developer."
    )
    assert any("independent research" in p.lower() for p, _ in job), job
    pref, _ = _collect_personal_intel_matches(
        "I prefer when I am asking you questions about your system "
        "to give me a detailed response."
    )
    assert any("detailed" in p.lower() for p, _ in pref), pref


def test_conversation_preference_extractor_marks_response_style_without_writing() -> None:
    text = "When answering last peer-reviewed or research questions, do not include DOI unless I ask."
    result = _extract_personal_intel(text, speaker="david", suppress_write=True)
    assert result["stored"] == 0
    assert "response_style" in result["personal_categories"]


def test_preference_ack_is_deterministic_and_specific_for_doi() -> None:
    text = "Do not include DOI in research answers unless I ask."
    saved = _build_preference_instruction_ack(text, stored_count=1)
    exists = _build_preference_instruction_ack(text, stored_count=0, matched=1)
    missed = _build_preference_instruction_ack(text, stored_count=0, matched=0)
    assert "Preference saved" in saved
    assert "already stored" in exists
    assert "DOI is omitted" in saved
    assert "DOI is omitted" in exists
    assert "did not store" in missed


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

