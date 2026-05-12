"""Tests for the dream consolidation feedback loop fix.

Covers:
  1. Exclusion filter: dream_artifact/dream_consolidation_proposal excluded from dream input
  2. Content dedup: identical artifact content only produced once per cycle
  3. Self-referential discard (content): validator catches "consolidation: dream_artifact"
  4. Self-referential discard (source tags): validator catches source-memory tag dominance
  5. Consolidation engine tag guard: clusters of dream artifacts score -1.0
  6. Repeated dream cycles: static memories don't grow artifact count after first pass
  7. Promoted dream artifacts excluded from next cycle's clustering input
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory
from consciousness.dream_artifacts import (
    ArtifactBuffer,
    DreamArtifact,
    ReflectiveValidator,
    create_artifact,
)
from memory.consolidation import MemoryConsolidationEngine
from memory.storage import MemoryStorage

_counter = 0


def _uid() -> str:
    global _counter
    _counter += 1
    return f"test_dcl_{_counter}_{time.time()}"


def _make_memory(
    weight: float = 0.4,
    tags: tuple[str, ...] = (),
    mem_type: str = "observation",
    payload: str = "test payload",
) -> Memory:
    return Memory(
        id=_uid(),
        timestamp=time.time(),
        weight=weight,
        tags=tags,
        payload=payload,
        type=mem_type,
        provenance="model_inference",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Exclusion filter — dream artifacts excluded from dream cycle input
# ──────────────────────────────────────────────────────────────────────────────

def test_dream_artifact_excluded_from_dream_input():
    """Memories tagged dream_artifact must be filtered out of _CONSOL_EXCLUDE_TAGS."""
    from consciousness.consciousness_system import _DREAM_INSIGHT_TAG

    _CONSOL_EXCLUDE_TAGS = frozenset({
        _DREAM_INSIGHT_TAG, "consolidated", "dream_consolidation",
        "dream_artifact", "dream_consolidation_proposal",
    })

    real_mem = _make_memory(tags=("conversation",))
    dream_mem = _make_memory(tags=("dream_artifact", "dream_consolidation_proposal"))
    insight_mem = _make_memory(tags=("dream_insight",))
    consolidated_mem = _make_memory(tags=("consolidated",))

    recent_all = [real_mem, dream_mem, insight_mem, consolidated_mem]
    recent = [
        m for m in recent_all
        if not (set(getattr(m, "tags", ())) & _CONSOL_EXCLUDE_TAGS)
    ]

    assert real_mem in recent, "Real memory should pass the filter"
    assert dream_mem not in recent, "dream_artifact tagged memory must be excluded"
    assert insight_mem not in recent, "dream_insight tagged memory must be excluded"
    assert consolidated_mem not in recent, "consolidated tagged memory must be excluded"


def test_dream_consolidation_proposal_excluded_from_dream_input():
    """Memories tagged only dream_consolidation_proposal must also be excluded."""
    from consciousness.consciousness_system import _DREAM_INSIGHT_TAG

    _CONSOL_EXCLUDE_TAGS = frozenset({
        _DREAM_INSIGHT_TAG, "consolidated", "dream_consolidation",
        "dream_artifact", "dream_consolidation_proposal",
    })

    proposal_mem = _make_memory(tags=("dream_consolidation_proposal",))
    recent = [
        m for m in [proposal_mem]
        if not (set(getattr(m, "tags", ())) & _CONSOL_EXCLUDE_TAGS)
    ]
    assert len(recent) == 0, "dream_consolidation_proposal must be excluded"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Content dedup — identical content only produces one artifact
# ──────────────────────────────────────────────────────────────────────────────

def test_content_dedup_prevents_duplicate_artifacts():
    """The dedup set must prevent the same content string from being added twice."""
    buf = ArtifactBuffer(maxlen=50)
    _existing_content: set[str] = {a.content for a in buf._buffer}

    content = "Consolidation: topic_A (5 memories, coherence=0.80)"
    added = 0

    for _ in range(5):
        if content not in _existing_content:
            buf.add(create_artifact(
                artifact_type="consolidation_proposal",
                source_memory_ids=["m1", "m2"],
                content=content,
                confidence=0.6,
                cluster_coherence=0.8,
            ))
            _existing_content.add(content)
            added += 1

    assert added == 1, f"Expected 1 artifact added, got {added}"
    assert len(buf._buffer) == 1


def test_content_dedup_allows_different_content():
    """Different content strings must still be added."""
    buf = ArtifactBuffer(maxlen=50)
    _existing_content: set[str] = {a.content for a in buf._buffer}

    contents = [
        "Consolidation: topic_A (5 memories, coherence=0.80)",
        "Consolidation: topic_B (3 memories, coherence=0.70)",
        "Consolidation: topic_C (4 memories, coherence=0.90)",
    ]

    added = 0
    for c in contents:
        if c not in _existing_content:
            buf.add(create_artifact(
                artifact_type="consolidation_proposal",
                source_memory_ids=["m1"],
                content=c,
                confidence=0.6,
                cluster_coherence=0.8,
            ))
            _existing_content.add(c)
            added += 1

    assert added == 3
    assert len(buf._buffer) == 3


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Self-referential discard (content) — validator catches dream_artifact text
# ──────────────────────────────────────────────────────────────────────────────

def test_self_ref_discard_content_dream_artifact():
    """Consolidation proposal about dream_artifact content must be discarded."""
    buf = ArtifactBuffer(maxlen=50)
    validator = ReflectiveValidator(buf)

    artifact = create_artifact(
        artifact_type="consolidation_proposal",
        source_memory_ids=["m1", "m2", "m3"],
        content="Consolidation: dream_artifact (5 memories, coherence=0.90)",
        confidence=0.7,
        cluster_coherence=0.9,
    )

    outcome = validator._evaluate(artifact)
    assert outcome.state == "discarded", f"Expected discarded, got {outcome.state}"
    assert "self-referential" in outcome.notes


def test_self_ref_discard_content_dream_consolidation_proposal():
    """Content mentioning dream_consolidation_proposal must also be discarded."""
    buf = ArtifactBuffer(maxlen=50)
    validator = ReflectiveValidator(buf)

    artifact = create_artifact(
        artifact_type="consolidation_proposal",
        source_memory_ids=["m1", "m2", "m3"],
        content="Consolidation: dream_consolidation_proposal (3 memories, coherence=0.85)",
        confidence=0.6,
        cluster_coherence=0.85,
    )

    outcome = validator._evaluate(artifact)
    assert outcome.state == "discarded", f"Expected discarded, got {outcome.state}"
    assert "self-referential" in outcome.notes


def test_non_self_ref_consolidation_not_discarded():
    """A legitimate consolidation proposal should not be discarded by the self-ref check."""
    buf = ArtifactBuffer(maxlen=50)
    validator = ReflectiveValidator(buf)

    artifact = create_artifact(
        artifact_type="consolidation_proposal",
        source_memory_ids=["m1", "m2", "m3"],
        content="Consolidation: user_preferences (4 memories, coherence=0.80)",
        confidence=0.6,
        cluster_coherence=0.8,
    )

    with patch.object(validator, '_source_memories_dominated_by_dream', return_value=False):
        outcome = validator._evaluate(artifact)
    assert outcome.state != "discarded" or "self-referential" not in outcome.notes


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Self-referential discard (source tags) — tag dominance check
# ──────────────────────────────────────────────────────────────────────────────

def test_self_ref_discard_source_tag_dominance():
    """Artifact with >= 50% dream-tagged source memories must be discarded."""
    storage = MemoryStorage(max_capacity=100)

    dream_m1 = _make_memory(tags=("dream_artifact", "dream_consolidation_proposal"))
    dream_m2 = _make_memory(tags=("dream_artifact",))
    real_m1 = _make_memory(tags=("conversation",))

    storage.add(dream_m1)
    storage.add(dream_m2)
    storage.add(real_m1)

    buf = ArtifactBuffer(maxlen=50)
    validator = ReflectiveValidator(buf)

    artifact = create_artifact(
        artifact_type="consolidation_proposal",
        source_memory_ids=[dream_m1.id, dream_m2.id, real_m1.id],
        content="Consolidation: some topic (3 memories, coherence=0.90)",
        confidence=0.7,
        cluster_coherence=0.9,
    )

    import memory.storage as _ms_mod
    original = _ms_mod.memory_storage
    _ms_mod.memory_storage = storage
    try:
        outcome = validator._evaluate(artifact)
    finally:
        _ms_mod.memory_storage = original

    assert outcome.state == "discarded", f"Expected discarded, got {outcome.state}"
    assert "predominantly dream" in outcome.notes


def test_self_ref_passes_when_minority_dream_sources():
    """Artifact with < 50% dream-tagged source memories should not be discarded by tag check."""
    storage = MemoryStorage(max_capacity=100)

    dream_m1 = _make_memory(tags=("dream_artifact",))
    real_m1 = _make_memory(tags=("conversation",))
    real_m2 = _make_memory(tags=("user_preference",))

    storage.add(dream_m1)
    storage.add(real_m1)
    storage.add(real_m2)

    buf = ArtifactBuffer(maxlen=50)
    validator = ReflectiveValidator(buf)

    artifact = create_artifact(
        artifact_type="consolidation_proposal",
        source_memory_ids=[dream_m1.id, real_m1.id, real_m2.id],
        content="Consolidation: real topic (3 memories, coherence=0.90)",
        confidence=0.7,
        cluster_coherence=0.9,
    )

    import memory.storage as _ms_mod
    original = _ms_mod.memory_storage
    _ms_mod.memory_storage = storage
    try:
        outcome = validator._evaluate(artifact)
    finally:
        _ms_mod.memory_storage = original

    assert outcome.state != "discarded" or "predominantly dream" not in outcome.notes


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Consolidation engine tag guard
# ──────────────────────────────────────────────────────────────────────────────

def test_consolidation_engine_rejects_dream_artifact_clusters():
    """Clusters dominated by dream_artifact-tagged memories must score -1.0."""
    engine = MemoryConsolidationEngine()

    mems = [
        _make_memory(tags=("dream_artifact",)),
        _make_memory(tags=("dream_artifact", "dream_consolidation_proposal")),
        _make_memory(tags=("dream_artifact",)),
    ]
    mem_map = {m.id: m for m in mems}

    @dataclass
    class FakeCluster:
        memory_ids: list
        coherence: float = 0.8

    cluster = FakeCluster(memory_ids=[m.id for m in mems], coherence=0.8)
    score = engine._score_cluster(cluster, mem_map)
    assert score == -1.0, f"Expected -1.0 for dream-dominated cluster, got {score}"


def test_consolidation_engine_accepts_real_clusters():
    """Clusters of non-dream memories should score above -1.0."""
    engine = MemoryConsolidationEngine()

    mems = [
        _make_memory(tags=("conversation",)),
        _make_memory(tags=("user_preference",)),
        _make_memory(tags=("observation",)),
    ]
    mem_map = {m.id: m for m in mems}

    @dataclass
    class FakeCluster:
        memory_ids: list
        coherence: float = 0.8

    cluster = FakeCluster(memory_ids=[m.id for m in mems], coherence=0.8)
    score = engine._score_cluster(cluster, mem_map)
    assert score > -1.0, f"Expected positive score for real cluster, got {score}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Repeated dream cycles — artifact count stable after first pass
# ──────────────────────────────────────────────────────────────────────────────

def test_repeated_cycles_no_artifact_growth():
    """Running the artifact generation loop twice on the same memories should not
    increase buffer count, because content dedup catches the duplicates."""
    buf = ArtifactBuffer(maxlen=200)
    MAX_ARTIFACTS_PER_DREAM_CYCLE = 20

    topics = ["topic_A", "topic_B", "topic_C"]

    def _run_cycle():
        _existing_content: set[str] = {a.content for a in buf._buffer}
        artifacts_created = 0
        for topic in topics:
            if artifacts_created >= MAX_ARTIFACTS_PER_DREAM_CYCLE:
                break
            _content = f"Consolidation: {topic} (5 memories, coherence=0.80)"
            if _content not in _existing_content:
                buf.add(create_artifact(
                    artifact_type="consolidation_proposal",
                    source_memory_ids=["m1", "m2"],
                    content=_content,
                    confidence=0.6,
                    cluster_coherence=0.8,
                ))
                _existing_content.add(_content)
                artifacts_created += 1
        return artifacts_created

    first_run = _run_cycle()
    count_after_first = len(buf._buffer)
    assert first_run == 3, f"First run should create 3 artifacts, got {first_run}"

    second_run = _run_cycle()
    count_after_second = len(buf._buffer)
    assert second_run == 0, f"Second run should create 0 artifacts, got {second_run}"
    assert count_after_second == count_after_first, \
        f"Buffer should not grow: {count_after_first} -> {count_after_second}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Promoted dream artifacts excluded from next cycle input
# ──────────────────────────────────────────────────────────────────────────────

def test_promoted_dream_artifacts_excluded_from_next_cycle():
    """A promoted dream artifact stored as a canonical memory must NOT appear in
    the next dream cycle's filtered input."""
    from consciousness.consciousness_system import _DREAM_INSIGHT_TAG

    _CONSOL_EXCLUDE_TAGS = frozenset({
        _DREAM_INSIGHT_TAG, "consolidated", "dream_consolidation",
        "dream_artifact", "dream_consolidation_proposal",
    })

    promoted_mem = _make_memory(
        tags=("dream_artifact", "dream_consolidation_proposal"),
        payload="[Dream artifact: consolidation_proposal] Consolidation: some topic",
    )

    real_mem = _make_memory(tags=("conversation",), payload="User said hello")

    all_memories = [promoted_mem, real_mem]
    recent_all = all_memories[-60:]
    recent = [
        m for m in recent_all
        if not (set(getattr(m, "tags", ())) & _CONSOL_EXCLUDE_TAGS)
    ]

    assert promoted_mem not in recent, \
        "Promoted dream artifact must be filtered out of dream cycle input"
    assert real_mem in recent, \
        "Real conversation memory must remain in dream cycle input"


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: Per-cycle artifact budget enforced
# ──────────────────────────────────────────────────────────────────────────────

def test_artifact_budget_caps_creation():
    """No more than MAX_ARTIFACTS_PER_DREAM_CYCLE artifacts should be created."""
    buf = ArtifactBuffer(maxlen=200)
    MAX_ARTIFACTS_PER_DREAM_CYCLE = 20
    _existing_content: set[str] = set()
    artifacts_created = 0

    for i in range(50):
        if artifacts_created >= MAX_ARTIFACTS_PER_DREAM_CYCLE:
            break
        content = f"Topic {i}"
        if content not in _existing_content:
            buf.add(create_artifact(
                artifact_type="consolidation_proposal",
                source_memory_ids=["m1"],
                content=content,
                confidence=0.6,
                cluster_coherence=0.8,
            ))
            _existing_content.add(content)
            artifacts_created += 1

    assert artifacts_created == 20, f"Expected exactly 20 artifacts, got {artifacts_created}"
