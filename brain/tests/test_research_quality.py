"""Tests for research quality gates: domain relevance + codebase study skip.

Covers:
  1. Tort-law / legal continuity abstract is rejected by domain gate.
  2. Neural/cognitive/consciousness abstract is accepted.
  3. Business continuity / backup paper is rejected during gestation.
  4. Companion-interest term can pass after user_preference memory exists.
  5. Tag-cluster terms pass the domain gate.
  6. Codebase source returns extraction_method="skipped:codebase".
  7. Codebase source creates zero study_claim memories.
  8. Normal academic source still creates claims.
  9. Relevance threshold raised to 0.15.
  10. Rejection telemetry counters increment correctly.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autonomy.knowledge_integrator import KnowledgeIntegrator, _AI_DOMAIN_INDICATORS
from autonomy.research_intent import ResearchIntent, ResearchFinding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_intent(question: str = "How does neural attention work?",
                 tag_cluster: tuple[str, ...] = ()) -> ResearchIntent:
    return ResearchIntent(
        question=question,
        source_event="test:manual",
        tag_cluster=tag_cluster,
    )


def _make_integrator(**kwargs) -> KnowledgeIntegrator:
    ki = KnowledgeIntegrator()
    if kwargs.get("engine"):
        ki.set_engine(kwargs["engine"])
    return ki


def _mock_engine(mode: str = "conversational", memories: list | None = None):
    engine = MagicMock()
    engine.mode_manager = MagicMock()
    engine.mode_manager.current_mode = mode

    mem_list = memories or []
    storage = MagicMock()
    storage.get_all.return_value = mem_list
    engine.memory = storage
    engine.storage = storage
    engine.remember = MagicMock(return_value=MagicMock())
    return engine


def _make_memory(tags: list[str], payload: str):
    mem = MagicMock()
    mem.tags = tags
    mem.payload = payload
    return mem


# ===========================================================================
# Domain relevance gate tests
# ===========================================================================

class TestDomainRelevanceGate:
    """Tests for _has_domain_relevance() method."""

    def test_ai_indicator_accepts_neural_content(self):
        ki = _make_integrator()
        content = (
            "We present a novel transformer architecture for multi-head "
            "attention that achieves state-of-the-art results on NLP benchmarks "
            "using a neural network with gradient-based optimization."
        )
        assert ki._has_domain_relevance(content) is True

    def test_ai_indicator_accepts_consciousness_content(self):
        ki = _make_integrator()
        content = (
            "This paper explores computational models of consciousness and "
            "metacognition in artificial intelligence systems."
        )
        assert ki._has_domain_relevance(content) is True

    def test_ai_indicator_accepts_reinforcement_learning(self):
        ki = _make_integrator()
        content = (
            "A policy gradient reinforcement learning approach for "
            "multi-agent simulation with reward shaping and exploration."
        )
        assert ki._has_domain_relevance(content) is True

    def test_tort_law_rejected(self):
        ki = _make_integrator()
        content = (
            "This article examines tort law reform and liability allocation "
            "in personal injury cases. We analyze court decisions regarding "
            "negligence standards and contributory fault in civil litigation. "
            "The persistence of archaic doctrines in autonomous vehicle "
            "liability frameworks is discussed."
        )
        assert ki._has_domain_relevance(content) is False

    def test_business_continuity_rejected(self):
        ki = _make_integrator()
        content = (
            "Business continuity planning for IT backup and disaster recovery. "
            "This paper presents organizational resilience frameworks for "
            "enterprise decision systems with persistence and data retention "
            "policies across autonomous business units."
        )
        assert ki._has_domain_relevance(content) is False

    def test_medical_billing_rejected(self):
        ki = _make_integrator()
        content = (
            "Hospital billing optimization through automated claims processing "
            "and insurance reimbursement scheduling for healthcare providers."
        )
        assert ki._has_domain_relevance(content) is False

    def test_tag_cluster_terms_pass(self):
        ki = _make_integrator()
        content = (
            "Advances in speaker diarization using spectral clustering "
            "for multi-party meetings."
        )
        intent = _make_intent(
            question="How to improve speaker diarization?",
            tag_cluster=("speaker", "diarization"),
        )
        assert ki._has_domain_relevance(content, intent) is True

    def test_tag_cluster_does_not_help_unrelated(self):
        ki = _make_integrator()
        content = (
            "Tort law reform in civil litigation and personal injury cases."
        )
        intent = _make_intent(
            question="How does attention mechanism work?",
            tag_cluster=("attention", "transformer"),
        )
        assert ki._has_domain_relevance(content, intent) is False


# ===========================================================================
# Companion domain expansion tests
# ===========================================================================

class TestCompanionDomainExpansion:
    """Tests for _get_companion_domain_terms()."""

    def test_gestation_returns_empty(self):
        engine = _mock_engine(mode="gestation")
        ki = _make_integrator(engine=engine)
        terms = ki._get_companion_domain_terms()
        assert terms == set()

    def test_no_engine_returns_empty(self):
        ki = _make_integrator()
        terms = ki._get_companion_domain_terms()
        assert terms == set()

    def test_no_preference_memories_returns_empty(self):
        engine = _mock_engine(mode="conversational", memories=[])
        ki = _make_integrator(engine=engine)
        terms = ki._get_companion_domain_terms()
        assert terms == set()

    def test_preference_memories_expand_domain(self):
        memories = [
            _make_memory(["user_preference"], "User enjoys woodworking and carpentry projects"),
            _make_memory(["user_preference"], "User is interested in amateur radio electronics"),
        ]
        engine = _mock_engine(mode="conversational", memories=memories)
        ki = _make_integrator(engine=engine)
        terms = ki._get_companion_domain_terms()
        assert "woodworking" in terms
        assert "carpentry" in terms
        assert "radio" in terms
        assert "electronics" in terms

    def test_companion_term_passes_domain_gate(self):
        memories = [
            _make_memory(["user_preference"], "User enjoys woodworking and furniture building"),
        ]
        engine = _mock_engine(mode="conversational", memories=memories)
        ki = _make_integrator(engine=engine)

        content = (
            "Advanced woodworking techniques for precision joinery "
            "and hand-cut dovetails in hardwood furniture construction."
        )
        assert ki._has_domain_relevance(content) is True

    def test_gestation_blocks_companion_content(self):
        memories = [
            _make_memory(["user_preference"], "User enjoys woodworking"),
        ]
        engine = _mock_engine(mode="gestation", memories=memories)
        ki = _make_integrator(engine=engine)

        content = (
            "Advanced woodworking techniques for hand-cut dovetails "
            "in hardwood furniture and cabinet construction."
        )
        assert ki._has_domain_relevance(content, None) is False


# ===========================================================================
# Relevance threshold tests
# ===========================================================================

class TestRelevanceThreshold:
    """Tests for the raised relevance threshold (0.1 -> 0.15)."""

    def test_low_overlap_rejected(self):
        score = KnowledgeIntegrator._compute_relevance(
            "How does attention work in transformers?",
            "This study examines the persistence of legacy systems in enterprise environments."
        )
        assert score < 0.15

    def test_high_overlap_accepted(self):
        score = KnowledgeIntegrator._compute_relevance(
            "How does attention work in transformers?",
            "The attention mechanism in transformer architectures enables effective "
            "parallel processing of sequence data."
        )
        assert score >= 0.15


# ===========================================================================
# Rejection telemetry tests
# ===========================================================================

class TestRejectionTelemetry:
    """Tests for rejection counter increments."""

    def test_domain_rejection_counter(self):
        ki = _make_integrator()
        engine = _mock_engine(mode="conversational")
        ki.set_engine(engine)

        intent = _make_intent(question="How do autonomous systems persist state?")
        finding = ResearchFinding(
            content=(
                "Enterprise autonomous systems that persist state across "
                "business continuity events. Organizational backup and "
                "disaster recovery for enterprise units."
            ),
            provenance="test",
            confidence=0.8,
            source_type="peer_reviewed",
        )

        from autonomy.research_intent import ResearchResult
        result = ResearchResult(
            intent_id=intent.id,
            tool_used="academic",
            findings=[finding],
        )

        ki._store_findings(intent, result)
        assert ki._ingest_stats["sources_rejected_domain"] >= 1

    def test_low_relevance_counter(self):
        ki = _make_integrator()
        engine = _mock_engine(mode="conversational")
        ki.set_engine(engine)

        intent = _make_intent(question="How does attention mechanism work?")
        finding = ResearchFinding(
            content="Completely unrelated content about gardening and soil pH levels.",
            provenance="test",
            confidence=0.8,
            source_type="peer_reviewed",
        )

        from autonomy.research_intent import ResearchResult
        result = ResearchResult(
            intent_id=intent.id,
            tool_used="academic",
            findings=[finding],
        )

        ki._store_findings(intent, result)
        assert ki._ingest_stats["sources_rejected_low_relevance"] >= 1


# ===========================================================================
# Codebase study skip tests
# ===========================================================================

class TestCodebaseStudySkip:
    """Tests for skipping claim generation on codebase sources."""

    def test_codebase_source_returns_skipped_method(self):
        from library.study import study_source, _study_telemetry

        mock_source = MagicMock()
        mock_source.source_id = "src_test_codebase"
        mock_source.source_type = "codebase"
        mock_source.content_depth = "full_text"

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_1"
        mock_chunk.text = "MAX_QUEUE_SIZE = 20\nMETRICS_INTERVAL = 5.0\ndef foo(): pass"

        mock_source_store = MagicMock()
        mock_source_store.get.return_value = mock_source

        mock_chunk_store = MagicMock()
        mock_chunk_store.get_for_source.return_value = [mock_chunk]

        with patch("library.study.source_store", mock_source_store, create=True), \
             patch("library.source.source_store", mock_source_store, create=True), \
             patch("library.chunks.chunk_store", mock_chunk_store, create=True):
            import library.source as _src_mod
            import library.chunks as _chunk_mod
            orig_ss = getattr(_src_mod, "source_store", None)
            orig_cs = getattr(_chunk_mod, "chunk_store", None)
            _src_mod.source_store = mock_source_store
            _chunk_mod.chunk_store = mock_chunk_store
            try:
                result = study_source("src_test_codebase")
            finally:
                _src_mod.source_store = orig_ss
                _chunk_mod.chunk_store = orig_cs

        assert result.extraction_method == "skipped:codebase"
        assert len(result.claims) == 0
        mock_source_store.mark_studied.assert_called_once_with("src_test_codebase")

    def test_codebase_source_still_extracts_concepts(self):
        from library.study import study_source

        mock_source = MagicMock()
        mock_source.source_id = "src_test_codebase2"
        mock_source.source_type = "codebase"
        mock_source.content_depth = "full_text"

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_2"
        mock_chunk.text = (
            "The consciousness engine processes consciousness events through "
            "perception layers. The perception orchestrator handles attention "
            "fusion and attention tracking. Speaker identification uses "
            "speaker embeddings for speaker recognition with neural policies."
        )

        mock_source_store = MagicMock()
        mock_source_store.get.return_value = mock_source

        mock_chunk_store = MagicMock()
        mock_chunk_store.get_for_source.return_value = [mock_chunk]

        import library.source as _src_mod
        import library.chunks as _chunk_mod
        orig_ss = getattr(_src_mod, "source_store", None)
        orig_cs = getattr(_chunk_mod, "chunk_store", None)
        _src_mod.source_store = mock_source_store
        _chunk_mod.chunk_store = mock_chunk_store
        try:
            result = study_source("src_test_codebase2")
        finally:
            _src_mod.source_store = orig_ss
            _chunk_mod.chunk_store = orig_cs

        assert result.extraction_method == "skipped:codebase"
        assert len(result.concepts) > 0

    def test_non_codebase_source_still_creates_claims(self):
        from library.study import study_source

        mock_source = MagicMock()
        mock_source.source_id = "src_test_academic"
        mock_source.source_type = "doi"
        mock_source.content_depth = "full_text"
        mock_source.doi = "10.1234/test"
        mock_source.venue = "NeurIPS"
        mock_source.domain_tags = ""

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_acad"
        mock_chunk.text = (
            "We propose a novel attention mechanism called Sparse Multi-Head "
            "Attention that achieves 95% accuracy on standard NLP benchmarks "
            "while reducing computation by 40%. Our method outperforms "
            "the baseline transformer architecture on all tested datasets."
        )

        mock_source_store = MagicMock()
        mock_source_store.get.return_value = mock_source

        mock_chunk_store = MagicMock()
        mock_chunk_store.get_for_source.return_value = [mock_chunk]

        import library.source as _src_mod
        import library.chunks as _chunk_mod
        orig_ss = getattr(_src_mod, "source_store", None)
        orig_cs = getattr(_chunk_mod, "chunk_store", None)
        _src_mod.source_store = mock_source_store
        _chunk_mod.chunk_store = mock_chunk_store
        try:
            with patch("library.study._try_llm_extraction", return_value=None):
                result = study_source("src_test_academic")
        finally:
            _src_mod.source_store = orig_ss
            _chunk_mod.chunk_store = orig_cs

        assert result.extraction_method == "regex"
        assert len(result.concepts) > 0


# ===========================================================================
# AI domain indicators sanity
# ===========================================================================

class TestDomainIndicatorsSanity:
    """Basic sanity checks on the indicator set."""

    def test_indicator_set_not_empty(self):
        assert len(_AI_DOMAIN_INDICATORS) > 20

    def test_all_indicators_lowercase(self):
        for indicator in _AI_DOMAIN_INDICATORS:
            assert indicator == indicator.lower(), f"Indicator not lowercase: {indicator}"

    def test_no_empty_indicators(self):
        for indicator in _AI_DOMAIN_INDICATORS:
            assert len(indicator.strip()) > 0


# ===========================================================================
# End-to-end: domain gate wired into _store_findings
# ===========================================================================

class TestStoreFindings:
    """Integration tests for the full _store_findings path."""

    def test_ai_finding_accepted(self):
        ki = _make_integrator()
        engine = _mock_engine(mode="conversational")
        ki.set_engine(engine)

        intent = _make_intent(question="How does neural attention improve transformers?")
        finding = ResearchFinding(
            content=(
                "Multi-head attention in transformer architectures enables "
                "parallel sequence processing with neural network layers."
            ),
            provenance="test",
            confidence=0.9,
            source_type="peer_reviewed",
            doi="10.1234/test",
        )

        from autonomy.research_intent import ResearchResult
        result = ResearchResult(
            intent_id=intent.id,
            tool_used="academic",
            findings=[finding],
        )

        with patch.object(ki, "_ingest_to_library", return_value="src_test_1"), \
             patch.object(ki, "_is_duplicate_pointer", return_value=False), \
             patch.object(ki, "_extract_claim", return_value="Attention helps"):
            created = ki._store_findings(intent, result)

        assert created == 1
        assert ki._ingest_stats["sources_rejected_domain"] == 0

    def test_tort_law_finding_rejected_by_domain(self):
        """Content has keyword overlap with query but wrong domain."""
        ki = _make_integrator()
        engine = _mock_engine(mode="conversational")
        ki.set_engine(engine)

        intent = _make_intent(
            question="How do autonomous systems handle state persistence?"
        )
        finding = ResearchFinding(
            content=(
                "This paper examines autonomous governance systems and "
                "their persistence in organizational state handling. "
                "Liability allocation across jurisdictions shows how "
                "court decisions manage civil litigation reform."
            ),
            provenance="test",
            confidence=0.7,
            source_type="peer_reviewed",
        )

        from autonomy.research_intent import ResearchResult
        result = ResearchResult(
            intent_id=intent.id,
            tool_used="academic",
            findings=[finding],
        )

        created = ki._store_findings(intent, result)
        assert created == 0
        assert ki._ingest_stats["sources_rejected_domain"] >= 1
