import sys
from types import SimpleNamespace

from tools.memory_tool import (
    _format_payload_preview,
    _is_system_self_memory,
    _keyword_search,
    _semantic_search,
    search_memory,
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
        semantic_search_scored=lambda *args, **kwargs: [(0.55, jarvis_mem), (0.41, user_mem)],
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


def test_semantic_search_scores_by_similarity_not_weight(monkeypatch) -> None:
    """Regression: the recall bug keyed results by m.weight, so high-weight
    boilerplate buried the topical match. Relevance must be query similarity."""
    # High intrinsic weight, but OFF-topic (low similarity to the query).
    boilerplate = SimpleNamespace(
        type="conversation",
        payload={"response": "User's name is David."},
        weight=0.90,
        identity_subject="david", identity_subject_type="person",
        identity_owner_type="person", tags=(),
    )
    # Low intrinsic weight, but ON-topic (high similarity) — the real answer.
    topical = SimpleNamespace(
        type="conversation",
        payload={"response": "Skyler is your dog, a border collie."},
        weight=0.55,
        identity_subject="david", identity_subject_type="person",
        identity_owner_type="person", tags=(),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.62, topical), (0.24, boilerplate)],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    results = _semantic_search("what do you remember about Skylar", limit=5, speaker="David")

    # The score carried through is the SIMILARITY, not the weight.
    assert results[0][0] == 0.62 and "Skyler is your dog" in results[0][1]
    assert results[1][0] == 0.24 and "User's name is David" in results[1][1]
    # weights (0.90 / 0.55) must NOT appear as the relevance scores
    assert results[0][0] != 0.90


def test_search_memory_leads_with_topical_match_and_labels_similarity(monkeypatch) -> None:
    """End-to-end: the formatted recall string leads with the highest-similarity
    memory and labels relevance with the true similarity (not weight)."""
    boilerplate = SimpleNamespace(
        type="conversation", payload={"response": "User's name is David."},
        weight=0.90, identity_subject="david", identity_subject_type="person",
        identity_owner_type="person", tags=(),
    )
    topical = SimpleNamespace(
        type="conversation", payload={"response": "Skyler is your dog, a border collie."},
        weight=0.55, identity_subject="david", identity_subject_type="person",
        identity_owner_type="person", tags=(),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.62, topical), (0.24, boilerplate)],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    out = search_memory("what do you remember about Skylar", speaker="David")
    lines = [ln for ln in out.splitlines() if "relevance=" in ln]
    assert lines, out
    # first rendered memory is the topical one, labeled with its similarity
    assert "Skyler is your dog" in lines[0]
    assert "relevance=0.62" in lines[0]


def test_keyword_fallback_ranks_below_semantic(monkeypatch) -> None:
    """A high-weight keyword/core memory must NOT outrank a real semantic match.
    Keyword fill is re-mapped onto 0..1 strictly below the weakest semantic hit."""
    topical = SimpleNamespace(
        type="conversation", payload={"response": "Skyler is your dog, a border collie."},
        weight=0.55, identity_subject="david", identity_subject_type="person",
        identity_owner_type="person", tags=(),
    )
    # keyword path scores by memory WEIGHT (>1.0 for core) — the old bug source
    core_mem = SimpleNamespace(
        id="mem_core", type="core",
        payload={"response": "First contact about Skylar: gestation complete."},
        tags=("core",), weight=1.50,
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.40, topical)],
        keyword_search=lambda *a, **k: [core_mem],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    out = search_memory("what do you remember about Skylar", speaker="David")
    lines = [ln for ln in out.splitlines() if "relevance=" in ln]
    assert "Skyler is your dog" in lines[0]  # semantic leads
    assert "First contact" in lines[1]       # keyword fill follows
    # the keyword line is re-scored below the semantic floor (no relevance=1.50)
    assert "relevance=1.5" not in out
    assert "relevance=0.40" in lines[0]


def test_search_memory_empty_returns_no_memories_sentinel(monkeypatch) -> None:
    """No relevant memory -> the honest sentinel the route uses to avoid
    confabulating (the 'first time you heard my voice' fake-date case)."""
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    out = search_memory("when was the first time you heard my voice", speaker="David")
    assert out.lower().startswith("no memories found")


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
