import sys
from types import SimpleNamespace

from tools.memory_tool import (
    _format_payload_preview,
    _is_system_self_memory,
    _keyword_search,
    _semantic_search,
)


def test_format_payload_preview_conversation_dict_is_human_readable() -> None:
    mem = SimpleNamespace(
        type="conversation",
        payload={
            "user_message": "What did I do yesterday?",
            "response": "You went to the mall with your kids and took Easter photos.",
        },
    )
    preview = _format_payload_preview(mem)
    assert "Jarvis recalled:" in preview
    assert "You went to the mall" in preview
    assert "User said:" not in preview
    assert "{" not in preview and "}" not in preview


def test_format_payload_preview_prefers_summary_keys_for_non_conversation_dict() -> None:
    mem = SimpleNamespace(
        type="contextual_insight",
        payload={"summary": "Interaction quality improved after using deterministic recall."},
    )
    preview = _format_payload_preview(mem)
    assert "Interaction quality improved" in preview


def test_is_system_self_memory_detects_jarvis_subject() -> None:
    mem = SimpleNamespace(
        type="observation",
        identity_subject="jarvis",
        identity_subject_type="self",
        identity_owner_type="system",
        tags=("speaker:jarvis",),
    )
    assert _is_system_self_memory(mem) is True


def test_semantic_search_filters_jarvis_self_memories_for_personal_activity(
    monkeypatch,
) -> None:
    jarvis_mem = SimpleNamespace(
        type="observation",
        payload={"summary": "Jarvis observed low confidence while idle."},
        weight=0.92,
        identity_subject="jarvis",
        identity_subject_type="self",
        identity_owner_type="system",
        tags=("speaker:jarvis",),
    )
    user_mem = SimpleNamespace(
        type="conversation",
        payload={"response": "You went to the mall with your kids on Sunday."},
        weight=0.78,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("speaker:david",),
    )

    fake_module = SimpleNamespace(
        semantic_search=lambda *args, **kwargs: [jarvis_mem, user_mem],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    results = _semantic_search(
        "What did I do Sunday?",
        limit=5,
        speaker="David",
    )

    assert len(results) == 1
    assert "[conversation]" in results[0][1]
    assert "Jarvis observed" not in results[0][1]


def test_keyword_search_normalizes_punctuation_for_temporal_query(monkeypatch) -> None:
    calls: list[str] = []
    mem = SimpleNamespace(
        id="mem_1",
        type="conversation",
        payload={"response": "You went to the mall yesterday with your kids."},
        tags=("speaker:david", "conversation"),
        weight=0.84,
    )

    def _fake_keyword_search(query: str, **kwargs):
        calls.append(query)
        if query == "yesterday":
            return [mem]
        return []

    fake_module = SimpleNamespace(keyword_search=_fake_keyword_search)
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    results = _keyword_search(
        "Tell me what I did yesterday.",
        limit=5,
        speaker="David",
    )

    assert "yesterday" in calls
    assert all("." not in token for token in calls)
    assert len(results) == 1
    assert "mall yesterday" in results[0][1].lower()
