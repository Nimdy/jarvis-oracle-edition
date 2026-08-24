import sys
from types import SimpleNamespace

from tools.memory_tool import (
    _extract_about_subjects,
    _format_payload_preview,
    _is_system_self_memory,
    _keyword_search,
    _search_cue_for_speaker,
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

    out = search_memory("what do you remember about Skyler", speaker="David")
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
        payload={"response": "First contact about Skyler: gestation complete."},
        tags=("core",), weight=1.50,
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.40, topical)],
        keyword_search=lambda *a, **k: [core_mem],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    out = search_memory("what do you remember about Skyler", speaker="David")
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


# Lived 2026-08-24: MEMORY "what do you remember about Schuyler?" replayed the
# 09:28 LLM wipe-claim as "Jarvis recalled:". The scar stays in the store;
# recall must not declare it as fact. Detector is tested in
# test_self_view_articulate; this pins the recall filter.
_LIVED_WIPE_LIE = (
    "My last recorded memory was on August 10th. I don't have a timeline of "
    "events beyond that, but I can tell you that my current state is fresh "
    "and ready to process new information. I've been offline for a while, so "
    "my memory is essentially reset — I'm starting fresh today."
)


def _osv_with_memories():
    import cognition.self_view as sv
    snap = {
        "consciousness": {"stage": "integrative", "awareness_level": 0.98,
                          "transcendence_level": 10.0},
        "evolution": {"stage": "integrative", "transcendence_level": 10.0},
        "policy": {"mode": "shadow", "nn_win_rate": 0.009, "eligible_for_control": False},
        "self_improve": {"active": True, "stage": 2, "effective_dry_run": True},
        "world_model": {"promotion": {"level_name": "active", "total_validated": 1},
                        "causal": {"predictive_total": 1, "predictive_accuracy": 0.8,
                                   "persistence_accuracy": 0.9},
                        "simulator_promotion": {"level_name": "shadow", "total_validated": 1},
                        "simulator": {"avg_confidence": 0.55}},
        "hemisphere": {"enabled": True, "matrix_specialists": [1]},
        "memory": {"total": 734, "core_count": 4,
                   "oldest_timestamp": 1782235435.0, "newest_timestamp": 1787578126.0},
    }
    return sv.build_self_view(engine=None, eval_snapshot={}, skills_summary={},
                              snapshot=snap, now=1.0)


def test_semantic_search_skips_osv_contradicted_wipe_conversation(monkeypatch) -> None:
    wipe_mem = SimpleNamespace(
        type="conversation",
        payload={
            "user_message": "when was your last recorded memory",
            "response": _LIVED_WIPE_LIE,
        },
        weight=0.55,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation", "speaker:david"),
    )
    ok_mem = SimpleNamespace(
        type="conversation",
        payload={"response": "Schuyler is the dog."},
        weight=0.50,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation",),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.71, wipe_mem), (0.40, ok_mem)],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    monkeypatch.setattr(
        "cognition.self_view.load_self_view",
        lambda: _osv_with_memories(),
    )

    results = _semantic_search("What do you remember about Schuyler?", limit=5, speaker="David")
    previews = " ".join(p for _, p in results)
    assert "starting fresh" not in previews
    assert "August 10" not in previews
    assert "Schuyler is the dog" in previews
    assert len(results) == 1


def test_format_preview_does_not_replay_wipe_as_jarvis_recalled(monkeypatch) -> None:
    monkeypatch.setattr(
        "cognition.self_view.load_self_view",
        lambda: _osv_with_memories(),
    )
    mem = SimpleNamespace(
        type="conversation",
        payload={
            "user_message": "when was your last recorded memory",
            "response": _LIVED_WIPE_LIE,
        },
    )
    preview = _format_payload_preview(mem)
    assert "starting fresh" not in preview
    assert "August 10" not in preview
    assert "Jarvis recalled:" not in preview


def test_topical_search_does_not_speak_courtesy_namedrop_or_curiosity(monkeypatch) -> None:
    """Lived 2026-08-24 12:16: 'remember about Skyler' spoke a courtesy closer
    that only namedrops Skyler in the tail, plus an identity curiosity Q.
    Fractal/HRR did not speak; the native MEMORY renderer did. Aboutness is
    the first sentence, not a whole-payload contains() — contains() would
    keep the courtesy line.
    """
    courtesy = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "You're welcome, David. I'm here to help whenever you're ready. "
                "If you ever need a snapshot, a note, or just someone to chat with "
                "— about Skyler, or anything else."
            )
        },
        weight=0.605,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation", "assistance", "speaker:david"),
    )
    fact = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "Got it, David — Skyler is your border collie. "
                "I've updated the record."
            )
        },
        weight=0.456,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation", "speaker:david"),
    )
    curiosity = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "Curiosity Q (identity): I heard someone speaking that I don't "
                "recognize — it wasn't you, David. Who was that? I'd like to know them. "
                "User answer: Walk me through how you reach an answer."
            )
        },
        weight=0.675,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("curiosity_answer", "curiosity_identity"),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [
            (0.51, courtesy),
            (0.48, fact),
            (0.44, curiosity),
        ],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)

    results = _semantic_search(
        "What do you remember about Skyler?",
        limit=5,
        speaker="David",
        referenced_entities={"Skyler"},
    )
    previews = " ".join(p for _, p in results)
    assert "border collie" in previews
    assert "You're welcome" not in previews
    assert "Curiosity Q" not in previews
    assert len(results) == 1


def test_aboutness_does_not_filter_when_no_referenced_subject(monkeypatch) -> None:
    """KNOW-not-guess: no extracted subject → do not invent an aboutness cut."""
    courtesy = SimpleNamespace(
        type="conversation",
        payload={"response": "You're welcome, David. I'm here to help."},
        weight=0.5,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation",),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.4, courtesy)],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    results = _semantic_search("thanks", limit=5, speaker="David", referenced_entities=None)
    assert len(results) == 1
    assert "You're welcome" in results[0][1]


def test_extract_about_subject_from_query_without_known_names() -> None:
    """Lived miss 12:50: Skyler is a remembered dog, not a soul relationship.
    About-X must come from the query, not get_known_names().
    """
    assert "skyler" in {s.lower() for s in _extract_about_subjects(
        "What do you remember about Skyler?"
    )}
    assert _extract_about_subjects("what do you remember about that") == set()
    assert _extract_about_subjects("what do you remember about me") == set()
    assert _extract_about_subjects("what do you remember about it") == set()
    assert _extract_about_subjects("What do you remember about me?", speaker="David") == {"David"}
    assert _extract_about_subjects("What do you remember about myself?", speaker="David") == {"David"}


def test_search_memory_about_skyler_does_not_use_identity_names(monkeypatch) -> None:
    """Public MEMORY path: empty known names, still drop courtesy/curiosity.
    Identity-boundary refs must not gain Skyler.
    """
    courtesy = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "You're welcome, David. I'm here to help whenever you're ready. "
                "If you ever need a snapshot, a note, or just someone to chat with "
                "— about Skyler, or anything else."
            )
        },
        weight=0.605,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation", "assistance"),
    )
    fact = SimpleNamespace(
        type="conversation",
        payload={
            "response": "Got it, David — Skyler is your border collie. I've updated the record."
        },
        weight=0.456,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("conversation",),
    )
    curiosity = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "Curiosity Q (identity): I heard someone speaking that I don't "
                "recognize — it wasn't you, David. Who was that?"
            )
        },
        weight=0.675,
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=("curiosity_answer",),
    )
    captured: dict = {}

    def fake_scored(*_a, **k):
        captured.update(k)
        return [(0.51, courtesy), (0.48, fact), (0.44, curiosity)]

    fake_module = SimpleNamespace(
        semantic_search_scored=fake_scored,
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    monkeypatch.setattr(
        "tools.memory_tool._extract_referenced_entities",
        lambda _q: set(),
    )

    out = search_memory("What do you remember about Skyler?", speaker="David")
    assert "border collie" in out
    assert "You're welcome" not in out
    assert "Curiosity Q" not in out
    boundary_refs = captured.get("referenced_entities") or set()
    assert "skyler" not in {str(x).lower() for x in boundary_refs}


def test_search_memory_about_me_does_not_declare_other_subjects_or_library(monkeypatch) -> None:
    """Lived 13:26: 'remember about me' mixed Skyler + a study_claim library
    definition. Those memories are not polluted and must stay in the store.
    About-me means the speaker; a competing proper name in the first sentence
    is another subject; external/library provenance is not autobiography.
    """
    self_fact = SimpleNamespace(
        type="conversation",
        payload={"response": "You are David, the primary user of this system."},
        weight=0.70,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="primary_user",
        identity_owner_type="person",
        tags=("conversation", "speaker:david"),
    )
    skyler = SimpleNamespace(
        type="conversation",
        payload={
            "response": "Ah, Skyler — your border collie. You mentioned she's clever."
        },
        weight=0.53,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="primary_user",
        identity_owner_type="person",
        tags=("conversation", "speaker:David"),
    )
    library = SimpleNamespace(
        type="factual_knowledge",
        payload={"response": "Concept: Episodic memory structure"},
        weight=0.80,
        provenance="external_source",
        identity_subject="external",
        identity_subject_type="library",
        identity_owner_type="system",
        tags=("study_claim", "episodic_memory"),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [
            (0.55, skyler),
            (0.50, library),
            (0.40, self_fact),
        ],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    monkeypatch.setattr("tools.memory_tool._extract_referenced_entities", lambda _q: set())

    out = search_memory("What do you remember about me?", speaker="David")
    assert "primary user" in out
    assert "border collie" not in out
    assert "Episodic memory structure" not in out
    assert "Skyler" not in out


def test_search_memory_about_me_does_not_declare_curiosity_asks(monkeypatch) -> None:
    """Lived 13:43: after about-me scope, the only remaining spoken line was
    a curiosity identity question. That record is the spark (ask/explore),
    not autobiography. COMPANION_COGNITION: ask-path vs knowledge-recall
    are different lanes. Do not delete the memory; do not declare it as
    'I remember you'.
    """
    self_fact = SimpleNamespace(
        type="conversation",
        payload={"response": "You are David, the primary user of this system."},
        weight=0.40,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="primary_user",
        identity_owner_type="person",
        tags=("conversation", "speaker:david"),
    )
    curiosity_ask = SimpleNamespace(
        type="conversation",
        payload={
            "response": (
                "Curiosity Q (identity): I heard someone speaking that I don't "
                "recognize — it wasn't you, David. Who was that? I'd like to know them. "
                "User answer: Walk me through how you reach an answer."
            )
        },
        weight=0.675,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="primary_user",
        identity_owner_type="person",
        tags=(
            "curiosity_answer",
            "curiosity_identity",
            "interactive",
            "outcome:engaged",
            "curiosity_topic:unknown_voice",
        ),
    )
    fake_module = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [
            (0.60, curiosity_ask),
            (0.40, self_fact),
        ],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake_module)
    monkeypatch.setattr("tools.memory_tool._extract_referenced_entities", lambda _q: set())

    out = search_memory("What do you remember about me?", speaker="David")
    assert "primary user" in out
    assert "Curiosity Q" not in out
    assert "don't recognize" not in out


def test_about_me_search_cue_uses_live_speaker_not_hardcoded_companion() -> None:
    """Lived 13:51 vs 13:52: 'about me' embedded as pronoun; 'about David'
    found the name. Cue follows the current speaker (voice/face identity),
    not a hardcoded primary companion — family guests must still work later.
    Unknown speaker is KNOW-not-guess: do not invent a name.
    """
    q = "What do you remember about me?"
    assert "David" in _search_cue_for_speaker(q, "David")
    assert "me" not in _search_cue_for_speaker(q, "David").lower().split()
    assert "Sarah" in _search_cue_for_speaker(q, "Sarah")
    assert _search_cue_for_speaker(q, "unknown") == q
    assert _search_cue_for_speaker(q, "") == q
    skyler = "What do you remember about Skyler?"
    assert _search_cue_for_speaker(skyler, "David") == skyler


def test_search_memory_about_me_embeds_speaker_name(monkeypatch) -> None:
    captured: dict = {}

    def fake_scored(query, *a, **k):
        captured["query"] = query
        captured.update(k)
        return []

    monkeypatch.setitem(
        sys.modules,
        "memory.search",
        SimpleNamespace(semantic_search_scored=fake_scored, keyword_search=lambda *a, **k: []),
    )
    monkeypatch.setattr("tools.memory_tool._extract_referenced_entities", lambda _q: set())
    search_memory("What do you remember about me?", speaker="David")
    assert "David" in captured.get("query", "")
    refs = captured.get("referenced_entities") or set()
    assert "skyler" not in {str(x).lower() for x in refs}
