"""Tests for the upgraded study pipeline: LLM extraction + document fetching."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestStudyLLMExtraction:
    """Tests for LLM-based structured knowledge extraction."""

    def test_set_llm_callback_enables_llm(self):
        from library.study import set_llm_callback, _llm_callback
        original = _llm_callback

        mock_llm = AsyncMock(return_value="test")
        set_llm_callback(mock_llm)

        from library import study
        assert study._llm_callback is mock_llm

        set_llm_callback(original)

    def test_set_llm_callback_none_disables(self):
        from library.study import set_llm_callback
        from library import study

        set_llm_callback(None)
        assert study._llm_callback is None

    def test_parse_llm_claims_valid_json(self):
        from library.study import _parse_llm_claims

        mock_chunks = [MagicMock(chunk_id="chunk_1"), MagicMock(chunk_id="chunk_2")]
        raw = json.dumps({
            "problem": "Training deep networks is computationally expensive",
            "approaches_tried": ["Pruning", "Distillation", "Quantization"],
            "what_failed": ["Naive pruning loses accuracy on complex tasks"],
            "what_worked": "Structured pruning with knowledge distillation retains 95% accuracy",
            "key_metrics": ["95% accuracy retention", "3x speedup", "60% memory reduction"],
            "conclusion": "Combined approach of structured pruning and distillation is optimal",
            "limitations": ["Only tested on vision models", "Requires teacher model"],
            "novel_concepts": ["Structured Knowledge Pruning (SKP)"],
        })

        claims = _parse_llm_claims(raw, mock_chunks)
        assert claims is not None
        assert len(claims) > 0

        types = {c.claim_type for c in claims}
        assert "problem" in types
        assert "result" in types
        assert "conclusion" in types
        assert "metric" in types
        assert "negative_result" in types
        assert "method" in types
        assert "definition" in types
        assert "limitation" in types

    def test_parse_llm_claims_with_code_fence(self):
        from library.study import _parse_llm_claims

        mock_chunks = [MagicMock(chunk_id="chunk_1")]
        raw = """Here is the extracted data:
```json
{
    "problem": "Memory is expensive",
    "what_worked": "New compression approach works well",
    "conclusion": "Use compression",
    "approaches_tried": [],
    "what_failed": [],
    "key_metrics": [],
    "limitations": [],
    "novel_concepts": []
}
```"""
        claims = _parse_llm_claims(raw, mock_chunks)
        assert claims is not None
        assert any(c.claim_type == "problem" for c in claims)

    def test_parse_llm_claims_invalid_json(self):
        from library.study import _parse_llm_claims
        mock_chunks = [MagicMock(chunk_id="chunk_1")]

        assert _parse_llm_claims("not json at all", mock_chunks) is None
        assert _parse_llm_claims("", mock_chunks) is None
        assert _parse_llm_claims("{invalid", mock_chunks) is None

    def test_parse_llm_claims_empty_fields(self):
        from library.study import _parse_llm_claims

        mock_chunks = [MagicMock(chunk_id="chunk_1")]
        raw = json.dumps({
            "problem": "",
            "approaches_tried": [],
            "what_failed": [],
            "what_worked": "",
            "key_metrics": [],
            "conclusion": "",
            "limitations": [],
            "novel_concepts": [],
        })

        claims = _parse_llm_claims(raw, mock_chunks)
        assert claims is None

    def test_claim_chunk_ids_from_chunks(self):
        from library.study import _parse_llm_claims

        mock_chunks = [
            MagicMock(chunk_id="c1"),
            MagicMock(chunk_id="c2"),
            MagicMock(chunk_id="c3"),
            MagicMock(chunk_id="c4"),
        ]
        raw = json.dumps({
            "problem": "Solving the alignment problem in large language models",
            "what_worked": "RLHF with human feedback works",
            "conclusion": "RLHF is effective",
            "approaches_tried": [],
            "what_failed": [],
            "key_metrics": [],
            "limitations": [],
            "novel_concepts": [],
        })

        claims = _parse_llm_claims(raw, mock_chunks)
        assert claims is not None
        for claim in claims:
            assert len(claim.chunk_ids) <= 3

    def test_try_llm_extraction_no_callback(self):
        from library.study import _try_llm_extraction, set_llm_callback

        set_llm_callback(None)
        result = _try_llm_extraction("some text", [], MagicMock())
        assert result is None

    def test_try_llm_extraction_short_content(self):
        from library.study import _try_llm_extraction, set_llm_callback

        set_llm_callback(AsyncMock(return_value=""))
        result = _try_llm_extraction("too short", [], MagicMock())
        assert result is None
        set_llm_callback(None)

    def test_study_result_has_extraction_method(self):
        from library.study import StudyResult
        r = StudyResult(source_id="test")
        assert r.extraction_method == "regex"

    def test_ensure_list(self):
        from library.study import _ensure_list
        assert _ensure_list(["a", "b"]) == ["a", "b"]
        assert _ensure_list("single") == ["single"]
        assert _ensure_list("") == []
        assert _ensure_list(None) == []
        assert _ensure_list(42) == []


class TestDocumentFetching:
    """Tests for _try_fetch_full_text in knowledge_integrator."""

    def test_no_url_returns_empty(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Some abstract text",
            provenance="academic: 10.1234/test",
            open_access_pdf_url="",
        )

        result, depth = KnowledgeIntegrator._try_fetch_full_text(finding)
        assert result == ""
        assert depth == ""

    def test_pdf_url_without_pdftotext(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Some abstract",
            provenance="test",
            open_access_pdf_url="https://example.com/paper.pdf",
        )

        with patch("config.BrainConfig") as mock_cfg:
            mock_cfg.return_value.research.fetch_full_text = True
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.headers.get.return_value = "application/pdf"
                mock_resp.read.return_value = b"fake pdf content"
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp

                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = FileNotFoundError("pdftotext not found")

                    result, depth = KnowledgeIntegrator._try_fetch_full_text(finding)
                    assert result == ""

    def test_html_url_fetch_success(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Some abstract",
            provenance="test",
            doi_url="https://example.com/paper",
        )

        long_text = (
            "Abstract: We propose a novel approach to neural architecture search. "
            "Our methodology involves evaluation of baseline models on a standard dataset. "
            "The results show significant improvement over state-of-the-art methods. "
            "In this experiment we present an empirical framework for quantitative analysis. "
        ) * 12

        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.read.return_value = long_text.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("config.BrainConfig") as mock_cfg:
            mock_cfg.return_value.research.fetch_full_text = True
            with patch("urllib.request.urlopen", return_value=mock_resp):
                result, depth = KnowledgeIntegrator._try_fetch_full_text(finding)
                assert depth == "full_text"
                assert len(result) > 0

    def test_html_url_fetch_too_short(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Some abstract",
            provenance="test",
            doi_url="https://example.com/paper",
        )

        with patch("config.BrainConfig") as mock_cfg:
            mock_cfg.return_value.research.fetch_full_text = True
            with patch("library.ingest._fetch_url", return_value=("short", "")):
                result, depth = KnowledgeIntegrator._try_fetch_full_text(finding)
                assert result == ""
                assert depth == ""

    def test_fetch_disabled_by_config(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Some abstract",
            provenance="test",
            open_access_pdf_url="https://example.com/paper.pdf",
        )

        with patch("config.BrainConfig") as mock_cfg:
            mock_cfg.return_value.research.fetch_full_text = False
            result, depth = KnowledgeIntegrator._try_fetch_full_text(finding)
            assert result == ""


class TestResearchFindingPDFUrl:
    """Tests for open_access_pdf_url threading through the pipeline."""

    def test_research_finding_has_pdf_url_field(self):
        from autonomy.research_intent import ResearchFinding
        f = ResearchFinding(
            content="test",
            provenance="test",
            open_access_pdf_url="https://example.com/paper.pdf",
        )
        assert f.open_access_pdf_url == "https://example.com/paper.pdf"

    def test_research_finding_pdf_url_defaults_empty(self):
        from autonomy.research_intent import ResearchFinding
        f = ResearchFinding(content="test", provenance="test")
        assert f.open_access_pdf_url == ""


class TestResearchConfig:
    """Tests for new config flags."""

    def test_fetch_full_text_default_true(self):
        from config import ResearchConfig
        cfg = ResearchConfig()
        assert cfg.fetch_full_text is True

    def test_llm_study_default_true(self):
        from config import ResearchConfig
        cfg = ResearchConfig()
        assert cfg.llm_study is True

    def test_fetch_open_access_default_true(self):
        from config import ResearchConfig
        cfg = ResearchConfig()
        assert cfg.fetch_open_access is True


class TestIngestUsesFullText:
    """Tests that _ingest_to_library upgrades content when full text is fetched."""

    def test_ingest_prefers_full_text(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki = KnowledgeIntegrator.__new__(KnowledgeIntegrator)
        finding = ResearchFinding(
            content="Short abstract text about neural networks.",
            provenance="academic: 10.1234/test",
            confidence=0.8,
            doi="10.1234/test",
            open_access_pdf_url="https://example.com/paper.pdf",
        )

        full_text = "This is a much longer paper. " * 100

        with patch.object(
            KnowledgeIntegrator, "_try_fetch_full_text",
            return_value=(full_text, "full_text"),
        ):
            with patch("library.source.source_store") as mock_ss:
                mock_ss.exists.return_value = False
                mock_ss.add.return_value = True

                with patch("library.chunks.chunk_text", return_value=[]):
                    with patch("library.chunks.chunk_store"):
                        with patch("library.index.library_index"):
                            result = ki._ingest_to_library(finding, "academic")

            if mock_ss.add.called:
                source_arg = mock_ss.add.call_args[0][0]
                assert source_arg.content_depth == "full_text"
                assert source_arg.license_flags == "open_access_full_text"
                assert len(source_arg.content_text) > 100


class TestLLMExtractionIntegration:
    """Integration-style tests for the full study flow with LLM."""

    def test_study_uses_llm_when_available(self):
        from library.study import study_source, set_llm_callback

        llm_response = json.dumps({
            "problem": "Training is expensive",
            "what_worked": "Distillation reduces cost by 10x",
            "conclusion": "Use distillation for deployment",
            "approaches_tried": ["Pruning", "Quantization"],
            "what_failed": ["Naive quantization loses accuracy"],
            "key_metrics": ["10x cost reduction", "95% accuracy"],
            "limitations": ["Only tested on BERT"],
            "novel_concepts": ["Progressive Distillation"],
        })

        mock_llm = AsyncMock(return_value=llm_response)
        set_llm_callback(mock_llm)

        mock_source = MagicMock()
        mock_source.content_depth = "abstract"
        mock_source.source_id = "test_src_001"
        mock_source.doi = "10.1234/test"
        mock_source.venue = "NeurIPS"

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_001"
        mock_chunk.text = "Training deep neural networks requires significant " \
                          "computational resources. We propose Progressive " \
                          "Distillation, a method that reduces training cost " \
                          "by 10x while retaining 95% accuracy. " * 5

        with patch("library.source.source_store") as mock_ss:
            mock_ss.get.return_value = mock_source
            with patch("library.chunks.chunk_store") as mock_cs:
                mock_cs.get_for_source.return_value = [mock_chunk]
                with patch("library.study._create_claim_memories"):
                    result = study_source("test_src_001")

        assert result.extraction_method == "llm"
        assert len(result.claims) > 0
        assert any(c.claim_type == "problem" for c in result.claims)
        assert any(c.claim_type == "result" for c in result.claims)

        set_llm_callback(None)

    def test_study_falls_back_to_regex_on_llm_failure(self):
        from library.study import study_source, set_llm_callback

        mock_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        set_llm_callback(mock_llm)

        mock_source = MagicMock()
        mock_source.content_depth = "abstract"
        mock_source.source_id = "test_src_002"
        mock_source.doi = "10.1234/test2"

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk_002"
        mock_chunk.text = "Our method achieves state-of-the-art performance " \
                          "on all benchmarks. We propose a novel approach " \
                          "that outperforms existing methods by 15%."

        with patch("library.source.source_store") as mock_ss:
            mock_ss.get.return_value = mock_source
            with patch("library.chunks.chunk_store") as mock_cs:
                mock_cs.get_for_source.return_value = [mock_chunk]
                with patch("library.study._create_claim_memories"):
                    result = study_source("test_src_002")

        assert result.extraction_method == "regex"

        set_llm_callback(None)
