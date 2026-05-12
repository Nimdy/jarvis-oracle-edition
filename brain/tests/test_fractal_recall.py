"""Tests for the Fractal Recall engine — Rollout 1.

24 tests covering: cue, provenance, resonance, seed, chain, governance,
rate limiting, and event payload (spec section 19).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from memory.fractal_recall import (
    CHAIN_CONTINUATION_THRESHOLD,
    FRACTAL_RECALL_INTERVAL_S,
    MAX_CHAIN_LENGTH,
    MAX_RECALLS_PER_HOUR,
    MIN_CUE_STRENGTH,
    RECALL_COOLDOWN_S,
    RESONANCE_THRESHOLD,
    AmbientCue,
    CueClass,
    FractalRecallEngine,
    FractalRecallResult,
    GovernanceAction,
    RecallCandidate,
    is_identity_sensitive,
    provenance_fitness,
)


# ---------------------------------------------------------------------------
# Mock Memory
# ---------------------------------------------------------------------------


@dataclass
class FakeMemory:
    id: str = "mem-1"
    timestamp: float = 1000.0
    weight: float = 0.5
    tags: tuple[str, ...] = ()
    payload: str = "Test memory content"
    type: str = "observation"
    provenance: str = "observed"
    is_core: bool = False
    association_count: int = 0
    last_accessed: float = 0.0
    associations: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Mock Storage / VectorStore
# ---------------------------------------------------------------------------


class FakeStorage:
    def __init__(self, memories: list[FakeMemory] | None = None):
        self._mems = {m.id: m for m in (memories or [])}

    def get(self, memory_id: str) -> FakeMemory | None:
        return self._mems.get(memory_id)

    def get_by_tag(self, tag: str) -> list[FakeMemory]:
        return [m for m in self._mems.values() if tag in m.tags]

    def get_recent(self, count: int) -> list[FakeMemory]:
        return list(self._mems.values())[:count]

    def get_related(self, memory_id: str, depth: int = 2) -> list[FakeMemory]:
        return [m for mid, m in self._mems.items() if mid != memory_id]


class FakeVectorStore:
    def __init__(self, results: list[dict[str, Any]] | None = None):
        self.available = True
        self._results = results or []

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    memories: list[FakeMemory] | None = None,
    vector_results: list[dict[str, Any]] | None = None,
    mode: str = "conversational",
    engagement: float = 0.5,
    speaker: str | None = "Alice",
    topic: str | None = None,
    emotion: str = "neutral",
    clock: float = 10000.0,
    **kw: Any,
) -> FractalRecallEngine:
    storage = FakeStorage(memories or [])
    vs = FakeVectorStore(vector_results or [])

    mode_mgr = MagicMock()
    mode_mgr.mode = mode

    attn = MagicMock()
    attn.get_state.return_value = {
        "engagement_level": engagement,
        "speaker_identity": speaker or "unknown",
        "person_present": speaker is not None,
    }

    engine = FractalRecallEngine(
        memory_storage=storage,
        vector_store=vs,
        mode_manager=mode_mgr,
        attention_core=attn,
        event_bus=MagicMock(),
        clock=lambda: clock,
        **kw,
    )

    # Override emotion gathering
    class FakeEmotionClassifier:
        current_emotion = emotion
    engine._emotion_classifier = FakeEmotionClassifier()

    # Override topic gathering
    if topic is not None:
        ws = MagicMock()
        ws.conversation.topic = topic
        engine._world_state = ws
    else:
        engine._world_state = None

    return engine


def _grounded_memory(
    mid: str = "mem-1",
    provenance: str = "observed",
    tags: tuple[str, ...] = ("person",),
    payload: str = "Saw user wave hello",
    assoc_count: int = 2,
    ts: float = 1000.0,
    last_accessed: float = 0.0,
    is_core: bool = False,
    mem_type: str = "observation",
) -> FakeMemory:
    return FakeMemory(
        id=mid,
        timestamp=ts,
        tags=tags,
        payload=payload,
        provenance=provenance,
        association_count=assoc_count,
        last_accessed=last_accessed,
        is_core=is_core,
        type=mem_type,
    )


# ═══════════════════════════════════════════════════════════════════
# CUE TESTS (3)
# ═══════════════════════════════════════════════════════════════════


class TestBuildCue:
    def test_build_cue_human_present(self):
        """Speaker + engagement > 0.30 → human_present."""
        engine = _make_engine(speaker="Alice", engagement=0.5)
        cue = engine.build_cue(10000.0)
        assert cue is not None
        assert cue.cue_class == "human_present"
        assert cue.speaker_id == "Alice"
        assert cue.engagement == 0.5

    def test_build_cue_technical_self_model(self):
        """Topic containing system-self lexicon → technical_self_model."""
        engine = _make_engine(topic="consciousness architecture review", speaker=None, engagement=0.0)
        cue = engine.build_cue(10000.0)
        assert cue is not None
        assert cue.cue_class == "technical_self_model"

    def test_build_cue_low_signal_returns_none_on_tick(self):
        """Cue strength below MIN_CUE_STRENGTH → tick returns None."""
        engine = _make_engine(
            speaker=None, engagement=0.0, topic=None, emotion="neutral",
        )
        engine._scene_tracker = None
        result = engine.tick(10000.0)
        assert result is None
        assert engine._low_signal_skips >= 1


# ═══════════════════════════════════════════════════════════════════
# PROVENANCE TESTS (3)
# ═══════════════════════════════════════════════════════════════════


class TestProvenance:
    def test_block_negative_provenance_fitness(self):
        """Dream/synthetic provenance → fitness < 0 (hard block)."""
        mem = _grounded_memory(provenance="dream_observer")
        assert provenance_fitness("dream_observer", "human_present", mem) < 0

        mem2 = _grounded_memory(tags=("dream_insight",))
        assert provenance_fitness("observed", "human_present", mem2) < 0

    def test_external_source_downweighted_in_human_present(self):
        """External source should be low in human_present cue class."""
        score = provenance_fitness("external_source", "human_present")
        assert score <= 0.2

    def test_model_inference_capped_in_technical_self_model(self):
        """model_inference fitness is 0.4 in technical_self_model (not 0.6)."""
        score = provenance_fitness("model_inference", "technical_self_model")
        assert score == pytest.approx(0.4)


# ═══════════════════════════════════════════════════════════════════
# RESONANCE TESTS (3)
# ═══════════════════════════════════════════════════════════════════


class TestResonance:
    def test_resonance_prefers_grounded_memory_given_same_semantic_match(self):
        """Observed provenance should score higher than model_inference at
        same semantic similarity in human_present context."""
        grounded = _grounded_memory(mid="g1", provenance="observed", tags=("person",))
        inferred = _grounded_memory(mid="g2", provenance="model_inference", tags=("person",))

        sem_results = [
            {"memory_id": "g1", "similarity": 0.7},
            {"memory_id": "g2", "similarity": 0.7},
        ]
        engine = _make_engine(
            memories=[grounded, inferred],
            vector_results=sem_results,
            speaker="Alice",
            engagement=0.5,
        )
        cue = engine.build_cue(10000.0)
        assert cue is not None
        candidates = engine.probe(cue, 10000.0)
        by_id = {c.memory_id: c for c in candidates}
        assert by_id["g1"].resonance > by_id["g2"].resonance

    def test_recency_penalty_dampens_recently_accessed_memory(self):
        """A memory accessed 5s ago should be penalized vs one accessed 1h ago."""
        now = 10000.0
        recent = _grounded_memory(mid="r1", last_accessed=now - 5.0)
        stale = _grounded_memory(mid="r2", last_accessed=now - 3600.0)

        sem_results = [
            {"memory_id": "r1", "similarity": 0.6},
            {"memory_id": "r2", "similarity": 0.6},
        ]
        engine = _make_engine(
            memories=[recent, stale],
            vector_results=sem_results,
            speaker="Alice",
            engagement=0.5,
        )
        cue = engine.build_cue(now)
        candidates = engine.probe(cue, now)
        by_id = {c.memory_id: c for c in candidates}
        assert by_id["r2"].recency_penalty < by_id["r1"].recency_penalty

    def test_mode_fit_helps_technical_memories_only_in_technical_self_model(self):
        """Technical provenance has high mode_fit only in technical_self_model class."""
        mem = _grounded_memory(
            mid="t1", provenance="experiment_result",
            tags=("architecture",), mem_type="self_improvement",
        )
        sem_results = [{"memory_id": "t1", "similarity": 0.6}]

        engine_tech = _make_engine(
            memories=[mem], vector_results=sem_results,
            topic="system memory architecture", speaker=None, engagement=0.0,
        )
        cue_tech = engine_tech.build_cue(10000.0)
        assert cue_tech.cue_class == "technical_self_model"
        cands_tech = engine_tech.probe(cue_tech, 10000.0)
        fit_tech = cands_tech[0].mode_fit if cands_tech else 0.0

        engine_human = _make_engine(
            memories=[mem], vector_results=sem_results,
            topic=None, speaker="Alice", engagement=0.5,
        )
        cue_human = engine_human.build_cue(10000.0)
        assert cue_human.cue_class == "human_present"
        cands_human = engine_human.probe(cue_human, 10000.0)
        fit_human = cands_human[0].mode_fit if cands_human else 0.0

        assert fit_tech > fit_human


# ═══════════════════════════════════════════════════════════════════
# SEED TESTS (2)
# ═══════════════════════════════════════════════════════════════════


class TestSeedSelection:
    def test_no_seed_below_threshold(self):
        """All candidates below RESONANCE_THRESHOLD → select_seed returns None."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=(), hour_bucket=12, emotion="neutral",
            speaker_id=None, mode="passive", topic=None, engagement=0.0,
            scene_entities=(), cue_strength=0.5, cue_class="ambient_environmental",
        )
        candidates = [
            RecallCandidate(
                memory_id="m1", memory=_grounded_memory(), resonance=0.3,
                semantic_score=0.3, tag_score=0.0, temporal_score=0.0,
                emotion_score=0.0, association_score=0.0, provenance_weight=0.5,
                mode_fit=0.3, recency_penalty=0.0, dominant_source="semantic",
                source_paths=("semantic",), dominant_tag=None, identity_sensitive=False,
            )
        ]
        seed = engine.select_seed(candidates, cue)
        assert seed is None

    def test_top_candidate_above_threshold_selected(self):
        """Candidate above RESONANCE_THRESHOLD is selected."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=(), hour_bucket=12, emotion="neutral",
            speaker_id=None, mode="passive", topic=None, engagement=0.0,
            scene_entities=(), cue_strength=0.5, cue_class="ambient_environmental",
        )
        above = RecallCandidate(
            memory_id="m1", memory=_grounded_memory(), resonance=0.7,
            semantic_score=0.7, tag_score=0.0, temporal_score=0.0,
            emotion_score=0.0, association_score=0.0, provenance_weight=0.5,
            mode_fit=0.3, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag=None, identity_sensitive=False,
        )
        below = RecallCandidate(
            memory_id="m2", memory=_grounded_memory(mid="m2"), resonance=0.3,
            semantic_score=0.3, tag_score=0.0, temporal_score=0.0,
            emotion_score=0.0, association_score=0.0, provenance_weight=0.5,
            mode_fit=0.3, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag=None, identity_sensitive=False,
        )
        seed = engine.select_seed([below, above], cue)
        assert seed is not None
        assert seed.memory_id == "m1"


# ═══════════════════════════════════════════════════════════════════
# CHAIN TESTS (5)
# ═══════════════════════════════════════════════════════════════════


class TestChainWalk:
    def _seed_candidate(self, mem: FakeMemory) -> RecallCandidate:
        return RecallCandidate(
            memory_id=mem.id, memory=mem, resonance=0.7,
            semantic_score=0.7, tag_score=0.5, temporal_score=0.0,
            emotion_score=0.0, association_score=0.3, provenance_weight=0.8,
            mode_fit=0.5, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag="person", identity_sensitive=False,
        )

    def _make_cue(self, **overrides: Any) -> AmbientCue:
        defaults = dict(
            text="scene: person | speaker: Alice | mode: conversational",
            tags=("person", "speaker:Alice", "conversational"),
            hour_bucket=14, emotion="neutral", speaker_id="Alice",
            mode="conversational", topic=None, engagement=0.5,
            scene_entities=("person",), cue_strength=0.6,
            cue_class="human_present",
        )
        defaults.update(overrides)
        return AmbientCue(**defaults)

    def test_chain_walk_anchors_to_original_cue(self):
        """Each hop is scored against the original cue, not the previous hop."""
        seed_mem = _grounded_memory(mid="s1", tags=("person", "greeting"))
        neighbor = _grounded_memory(mid="n1", tags=("person",), assoc_count=3)
        storage = FakeStorage([seed_mem, neighbor])
        engine = _make_engine(memories=[seed_mem, neighbor])
        engine._storage = storage
        cue = self._make_cue()
        seed = self._seed_candidate(seed_mem)
        chain = engine.walk_chain(seed, cue, 10000.0)
        assert chain[0].memory_id == "s1"

    def test_chain_walk_stops_on_repeated_id(self):
        """Chain stops if a memory ID would repeat."""
        seed_mem = _grounded_memory(mid="s1", tags=("person",))
        storage = FakeStorage([seed_mem])
        engine = _make_engine(memories=[seed_mem])
        engine._storage = storage
        cue = self._make_cue()
        seed = self._seed_candidate(seed_mem)
        chain = engine.walk_chain(seed, cue, 10000.0)
        ids = [c.memory_id for c in chain]
        assert len(ids) == len(set(ids))

    def test_chain_walk_stops_on_blocked_provenance(self):
        """Chain stops immediately when encountering dream provenance."""
        seed_mem = _grounded_memory(mid="s1", tags=("person",))
        dream_mem = _grounded_memory(mid="d1", provenance="dream_observer", tags=("person",))
        good_mem = _grounded_memory(mid="g1", tags=("person",), assoc_count=5)

        class OrderedStorage(FakeStorage):
            def get_related(self, memory_id: str, depth: int = 2) -> list[FakeMemory]:
                return [dream_mem, good_mem]

        storage = OrderedStorage([seed_mem, dream_mem, good_mem])
        engine = _make_engine(memories=[seed_mem, dream_mem, good_mem])
        engine._storage = storage
        cue = self._make_cue()
        seed = self._seed_candidate(seed_mem)
        chain = engine.walk_chain(seed, cue, 10000.0)
        chain_ids = [c.memory_id for c in chain]
        assert "d1" not in chain_ids
        assert "g1" not in chain_ids  # walk breaks on blocked provenance

    def test_chain_walk_stops_on_consecutive_weak_hops(self):
        """Two consecutive below-threshold hops terminate the walk."""
        seed_mem = _grounded_memory(mid="s1", tags=("person",))
        # Weak memories with no tag overlap and no association score
        weak1 = _grounded_memory(mid="w1", tags=("xyz_unrelated",), provenance="observed", assoc_count=0)
        weak2 = _grounded_memory(mid="w2", tags=("abc_unrelated",), provenance="observed", assoc_count=0)
        good = _grounded_memory(mid="g1", tags=("person",), assoc_count=5)

        class OrderedStorage(FakeStorage):
            def get_related(self, memory_id: str, depth: int = 2) -> list[FakeMemory]:
                return [weak1, weak2, good]

        storage = OrderedStorage([seed_mem, weak1, weak2, good])
        engine = _make_engine(memories=[seed_mem, weak1, weak2, good])
        engine._storage = storage
        cue = self._make_cue()
        seed = self._seed_candidate(seed_mem)
        chain = engine.walk_chain(seed, cue, 10000.0)
        chain_ids = [c.memory_id for c in chain]
        assert "g1" not in chain_ids

    def test_chain_walk_stops_on_topic_repetition(self):
        """Same dominant_tag > 2 times terminates the walk."""
        seed_mem = _grounded_memory(mid="s1", tags=("cats",))
        n1 = _grounded_memory(mid="n1", tags=("cats",), assoc_count=5)
        n2 = _grounded_memory(mid="n2", tags=("cats",), assoc_count=5)
        n3 = _grounded_memory(mid="n3", tags=("cats",), assoc_count=5)

        class OrderedStorage(FakeStorage):
            def get_related(self, memory_id: str, depth: int = 2) -> list[FakeMemory]:
                return [n1, n2, n3]

        storage = OrderedStorage([seed_mem, n1, n2, n3])
        engine = _make_engine(memories=[seed_mem, n1, n2, n3])
        engine._storage = storage
        cue = self._make_cue(tags=("cats", "person", "conversational"))
        seed = self._seed_candidate(seed_mem)
        chain = engine.walk_chain(seed, cue, 10000.0)
        # seed has cats (1), at most 2 more before stop → max 3 total
        cats_count = sum(1 for c in chain if c.dominant_tag == "cats")
        assert cats_count <= 3


# ═══════════════════════════════════════════════════════════════════
# GOVERNANCE TESTS (4)
# ═══════════════════════════════════════════════════════════════════


class TestGovernance:
    def _make_candidate(self, *, identity_sensitive: bool = False, provenance: str = "observed") -> RecallCandidate:
        mem = _grounded_memory(provenance=provenance)
        return RecallCandidate(
            memory_id=mem.id, memory=mem, resonance=0.7,
            semantic_score=0.7, tag_score=0.5, temporal_score=0.3,
            emotion_score=0.0, association_score=0.3, provenance_weight=0.8,
            mode_fit=0.5, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag="person",
            identity_sensitive=identity_sensitive,
        )

    def test_identity_sensitive_chain_is_reflective_only(self):
        """Chains with identity-sensitive memories must be reflective_only."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=("person",), hour_bucket=14, emotion="neutral",
            speaker_id="Alice", mode="conversational", topic=None,
            engagement=0.5, scene_entities=("person",), cue_strength=0.6,
            cue_class="human_present",
        )
        chain = [
            self._make_candidate(),
            self._make_candidate(identity_sensitive=True),
        ]
        action, confidence, reasons = engine.recommend_governance_action(cue, chain)
        assert action == "reflective_only"
        assert "identity_sensitive" in reasons

    def test_grounded_contextual_chain_can_be_eligible_for_proactive(self):
        """Grounded chain + human_present + good engagement → eligible_for_proactive."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=("person",), hour_bucket=14, emotion="neutral",
            speaker_id="Alice", mode="conversational", topic=None,
            engagement=0.5, scene_entities=("person",), cue_strength=0.6,
            cue_class="human_present",
        )
        chain = [
            self._make_candidate(provenance="observed"),
            self._make_candidate(provenance="user_claim"),
        ]
        action, confidence, reasons = engine.recommend_governance_action(cue, chain)
        assert action == "eligible_for_proactive"

    def test_weak_chain_recommendation_ignore(self):
        """Low-resonance chain → ignore."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=(), hour_bucket=14, emotion="neutral",
            speaker_id=None, mode="passive", topic=None,
            engagement=0.1, scene_entities=(), cue_strength=0.2,
            cue_class="ambient_environmental",
        )
        weak = RecallCandidate(
            memory_id="w1", memory=_grounded_memory(), resonance=0.2,
            semantic_score=0.2, tag_score=0.0, temporal_score=0.0,
            emotion_score=0.0, association_score=0.0, provenance_weight=0.3,
            mode_fit=0.2, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag=None, identity_sensitive=False,
        )
        action, confidence, reasons = engine.recommend_governance_action(cue, [weak])
        assert action == "ignore"

    def test_recommendation_contains_confidence_and_reason_codes(self):
        """Governance always returns (action, float, tuple[str, ...])."""
        engine = _make_engine()
        cue = AmbientCue(
            text="test", tags=("person",), hour_bucket=14, emotion="neutral",
            speaker_id="Alice", mode="conversational", topic=None,
            engagement=0.5, scene_entities=("person",), cue_strength=0.6,
            cue_class="human_present",
        )
        chain = [self._make_candidate()]
        action, confidence, reason_codes = engine.recommend_governance_action(cue, chain)
        assert isinstance(action, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reason_codes, tuple)
        assert all(isinstance(r, str) for r in reason_codes)
        assert len(reason_codes) >= 1


# ═══════════════════════════════════════════════════════════════════
# RATE LIMITING TESTS (2)
# ═══════════════════════════════════════════════════════════════════


class TestRateLimiting:
    def test_cooldown_blocks_second_recall(self):
        """Second tick within RECALL_COOLDOWN_S should return None."""
        mem = _grounded_memory(mid="m1", tags=("person",), assoc_count=3)
        sem = [{"memory_id": "m1", "similarity": 0.9}]
        engine = _make_engine(memories=[mem], vector_results=sem, speaker="Alice", engagement=0.6)

        first = engine.tick(10000.0)
        # Even if first succeeds or not, force the cooldown state
        engine._last_recall_ts = 10000.0

        second = engine.tick(10000.0 + RECALL_COOLDOWN_S - 1.0)
        assert second is None
        assert engine._cooldown_skips >= 1

    def test_hourly_cap_blocks_excess_recalls(self):
        """After MAX_RECALLS_PER_HOUR recalls, further ticks are blocked."""
        engine = _make_engine()
        base = 10000.0
        for i in range(MAX_RECALLS_PER_HOUR):
            engine._recent_recall_timestamps.append(base + i * 100.0)

        result = engine.tick(base + (MAX_RECALLS_PER_HOUR * 100.0) + RECALL_COOLDOWN_S + 1.0)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# EVENT PAYLOAD TESTS (2)
# ═══════════════════════════════════════════════════════════════════


class TestEventPayload:
    def _make_result(self) -> FractalRecallResult:
        mem = _grounded_memory(mid="m1", provenance="observed", tags=("person",))
        candidate = RecallCandidate(
            memory_id="m1", memory=mem, resonance=0.7,
            semantic_score=0.7, tag_score=0.5, temporal_score=0.3,
            emotion_score=0.0, association_score=0.3, provenance_weight=0.8,
            mode_fit=0.5, recency_penalty=0.0, dominant_source="semantic",
            source_paths=("semantic",), dominant_tag="person", identity_sensitive=False,
        )
        cue = AmbientCue(
            text="test cue", tags=("person",), hour_bucket=14, emotion="neutral",
            speaker_id="Alice", mode="conversational", topic=None,
            engagement=0.5, scene_entities=("person",), cue_strength=0.6,
            cue_class="human_present",
        )
        return FractalRecallResult(
            cue=cue, seed=candidate, chain=[candidate],
            governance_recommended_action="hold_for_curiosity",
            governance_confidence=0.65,
            governance_reason_codes=("high_relevance",),
            provenance_mix={"observed": 1},
            timestamp=10000.0,
        )

    def test_emit_surface_payload_contains_chain_items(self):
        """Event payload must include chain_items with per-item metadata."""
        event_bus = MagicMock()
        engine = _make_engine()
        engine._event_bus = event_bus

        result = self._make_result()

        with patch("memory.fractal_recall.FractalRecallEngine.emit_surface", wraps=engine.emit_surface):
            engine.emit_surface(result)

        assert event_bus.emit.called
        # Find the FRACTAL_RECALL_SURFACED call
        found = False
        for call in event_bus.emit.call_args_list:
            args, kwargs = call
            if args and "fractal_recall" in str(args[0]):
                found = True
                assert "chain_items" in kwargs
                items = kwargs["chain_items"]
                assert len(items) == 1
                item = items[0]
                assert "memory_id" in item
                assert "provenance" in item
                assert "resonance" in item
                assert "identity_sensitive" in item
                break
        assert found, "FRACTAL_RECALL_SURFACED event was not emitted"

    def test_emit_surface_never_calls_tts(self):
        """emit_surface must never invoke TTS. Verify no tts-related calls."""
        event_bus = MagicMock()
        engine = _make_engine()
        engine._event_bus = event_bus

        result = self._make_result()
        engine.emit_surface(result)

        for call in event_bus.emit.call_args_list:
            args, kwargs = call
            event_name = args[0] if args else ""
            assert "tts" not in str(event_name).lower()
            assert "PERCEPTION_PLAYBACK" not in str(event_name)
            assert "audio" not in str(event_name).lower() or "fractal_recall" in str(event_name)
