"""Household self-fact recall consumes native MEMORY, not the LLM.

Lived 2026-08-31: 'who is in my family' / kids' names / morning routine routed
NONE and Qwen authored Emily/Mike. Native MEMORY formatter already fail-closes.
Class matchers — no operator names in the router.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from reasoning.tool_router import ToolRouter, ToolType
from tools.memory_tool import (
    _extract_about_subjects,
    _preview_matches_household_kind,
    household_recall_kind,
    is_household_self_fact_recall,
    search_memory,
)

router = ToolRouter()
_BRAIN = Path(__file__).resolve().parent.parent


def test_household_questions_route_to_memory() -> None:
    cases = [
        "Jarvis, who is in my family?",
        "Who is in my family?",
        "Tell me about my family.",
        "Jarvis, what are my kids' names?",
        "What are my kids names?",
        "What's my morning routine?",
        "Jarvis, what's my morning routine?",
        "When should you not interrupt me?",
    ]
    for text in cases:
        assert router.route(text).tool == ToolType.MEMORY, f"{text!r} -> {router.route(text).tool}"
        assert is_household_self_fact_recall(text), text


def test_household_recall_does_not_steal_general_knowledge_or_teaches() -> None:
    assert router.route("Who made the Halo game?").tool != ToolType.MEMORY
    assert not is_household_self_fact_recall("Who made the Halo game?")
    assert not is_household_self_fact_recall(
        "Jarvis, my morning routine is coffee, dashboard check, then coding."
    )
    assert not is_household_self_fact_recall("My kids are Lily and Owen. Owen is my son.")
    assert household_recall_kind("What are my hobbies?") == ""


def test_who_is_in_my_family_does_not_aboutness_cut_on_in() -> None:
    assert _extract_about_subjects("who is in my family") == set()
    assert _extract_about_subjects("Jarvis, who is in my family?") == set()


def test_household_fact_preview_keeps_taught_facts_not_recaps() -> None:
    assert _preview_matches_household_kind("[user_preference] User's wife is Tanya", "family")
    assert _preview_matches_household_kind("[user_preference] User's daughter is Lily", "kids")
    assert _preview_matches_household_kind("[user_preference] User's son is Owen", "kids")
    # Plastic: a taught dog/cousin/great-great is a pref, not a kinship regex.
    assert _preview_matches_household_kind("[user_preference] User's dog is Skylar", "family")
    assert _preview_matches_household_kind("[user_preference] User's cousin is family", "family")
    assert not _preview_matches_household_kind(
        "[conversation] Jarvis, who is in my family? | Your family includes",
        "family",
    )
    assert not _preview_matches_household_kind("[core] I think, I observe, I remember.", "family")
    assert _preview_matches_household_kind(
        "[user_preference] User daily routine: a walk with Skylar after work",
        "morning",
    )


def test_household_search_returns_facts_not_conversation_recaps(monkeypatch) -> None:
    wife = SimpleNamespace(
        id="w", type="user_preference", payload="User's wife is Tanya",
        tags=("personal_fact",), provenance="user_claim", weight=0.75,
        identity_subject="david", identity_subject_type="person",
        identity_owner_type="primary_user",
    )
    recap = SimpleNamespace(
        id="c", type="conversation",
        payload={"user_message": "Who is in my family?", "response": "Emily and Mike"},
        tags=("conversation",), provenance="conversation", weight=0.6,
        identity_subject="david", identity_subject_type="person",
        identity_owner_type="primary_user",
    )
    dog = SimpleNamespace(
        id="d", type="user_preference", payload="User's dog is Skylar",
        tags=("personal_fact",), provenance="user_claim", weight=0.76,
        identity_subject="david", identity_subject_type="person",
        identity_owner_type="primary_user",
    )

    monkeypatch.setattr(
        "tools.memory_tool._build_identity_context", lambda speaker="": None,
    )
    monkeypatch.setattr(
        "tools.memory_tool._extract_referenced_entities", lambda query: set(),
    )
    def fake_keyword(query, limit=20, **kwargs):
        q = str(query).lower()
        if "family" in q:
            return [wife, dog]
        if "wife" in q:
            return [wife]
        if "daughter" in q:
            return []
        return [recap] if "family" in q else []

    recaps = [(0.9 - i * 0.01, recap) for i in range(8)]
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: recaps,
        keyword_search=fake_keyword,
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    out = search_memory("Jarvis, who is in my family?", speaker="David")
    assert "User's wife is Tanya" in out
    assert "dog is skylar" in out.lower()
    assert "Emily" not in out


def test_household_matchers_do_not_hardcode_operator_names() -> None:
    blobs = [
        (_BRAIN / "reasoning" / "tool_router.py").read_text(encoding="utf-8"),
        (_BRAIN / "tools" / "memory_tool.py").read_text(encoding="utf-8"),
    ]
    joined = "\n".join(blobs).lower()
    # Lived leftover: VQA_NOT_FRAME already had tanya/skyler to keep VISION off
    # household questions. New household MEMORY couple must not add names.
    text = (_BRAIN / "tools" / "memory_tool.py").read_text(encoding="utf-8")
    start = text.find("_HOUSEHOLD_FAMILY_RE")
    end = text.find("# Session bookkeeping")
    assert start != -1 and end != -1 and end > start
    household_slice = text[start:end].lower()
    for name in ("tanya", "tonya", "lily", "owen", "david", "skylar", "skyler", "sarah"):
        assert name not in household_slice, name
    # Family recall is not a kinship ontology. Question matchers may say
    # "my family". Fact-preview must not require wife/son/dog/cousin.
    for role in ("wife", "husband", "cousin", "dog", "pet", "collie", "grandmother"):
        assert role not in household_slice, role
    assert "in\\s+my\\s+family" in household_slice or "who is in my family" in joined


def test_household_questions_are_self_preference_when_response_router_imports() -> None:
    try:
        from reasoning.response import route_memory_request
        from conversation_handler import _should_use_memory_search
    except ModuleNotFoundError:
        return
    text = "Who is in my family?"
    assert route_memory_request(text, set()).route_type == "self_preference"
    assert _should_use_memory_search(text) is True
