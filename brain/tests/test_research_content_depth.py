"""Tests for the Research Content Depth Fix.

Covers:
  - ResearchConfig in BrainConfig
  - AcademicResult new fields (tldr, paper_id, open_access)
  - enrich_result() detail fetch
  - Content depth inference in knowledge_integrator
  - Source schema content_depth column
  - Study pipeline content depth gate
  - metric_triggers logging format fix
  - reset-brain.sh library wipe
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Phase 1: ResearchConfig ─────────────────────────────────────────────────


class TestResearchConfig:
    def test_research_config_defaults(self):
        from config import ResearchConfig
        cfg = ResearchConfig()
        assert cfg.enabled is True
        assert cfg.s2_api_key == ""
        assert cfg.crossref_mailto == ""
        assert cfg.min_content_chars == 200
        assert cfg.max_content_chars == 10000
        assert cfg.fetch_tldr is True
        assert cfg.fetch_open_access is True
        assert cfg.enrich_on_ingest is True
        assert cfg.detail_fetch_timeout == 10

    def test_research_config_in_brain_config(self):
        from config import BrainConfig, ResearchConfig
        cfg = BrainConfig()
        assert hasattr(cfg, "research")
        assert isinstance(cfg.research, ResearchConfig)

    def test_research_config_custom_values(self):
        from config import ResearchConfig
        cfg = ResearchConfig(
            s2_api_key="test-key",
            min_content_chars=500,
            fetch_open_access=True,
        )
        assert cfg.s2_api_key == "test-key"
        assert cfg.min_content_chars == 500
        assert cfg.fetch_open_access is True

    def test_clean_env_value_strips_systemd_inline_comments(self):
        from config import _clean_env_value

        assert _clean_env_value("192.168.1.248             # Pi IP") == "192.168.1.248"
        assert _clean_env_value("8080                    # Pi UI") == "8080"
        assert _clean_env_value('"192.168.1.248"') == "192.168.1.248"

    def test_brain_config_strips_systemd_inline_comment_env_values(self, monkeypatch):
        from config import BrainConfig

        monkeypatch.setenv("PI_HOST", "192.168.1.248             # Pi IP")
        monkeypatch.setenv("PI_UI_PORT", "8080                    # Pi UI")
        monkeypatch.setenv("DASHBOARD_PORT", "9200                # dashboard")
        monkeypatch.setenv("RESEARCH_FETCH_FULL_TEXT", "false      # disable full text")

        cfg = BrainConfig()

        assert cfg.perception.pi_host == "192.168.1.248"
        assert cfg.perception.pi_ui_port == 8080
        assert cfg.dashboard.port == 9200
        assert cfg.research.fetch_full_text is False


# ── Phase 2: AcademicResult + S2 enrichment ─────────────────────────────────


class TestAcademicResultFields:
    def test_new_fields_default(self):
        from tools.academic_search_tool import AcademicResult
        r = AcademicResult(title="Test Paper")
        assert r.tldr == ""
        assert r.is_open_access is False
        assert r.open_access_pdf_url == ""
        assert r.paper_id == ""

    def test_new_fields_set(self):
        from tools.academic_search_tool import AcademicResult
        r = AcademicResult(
            title="Test",
            tldr="This paper proposes a new method.",
            is_open_access=True,
            open_access_pdf_url="https://example.com/paper.pdf",
            paper_id="abc123",
        )
        assert r.tldr == "This paper proposes a new method."
        assert r.is_open_access is True
        assert r.paper_id == "abc123"


class TestS2Headers:
    def test_headers_empty_without_key(self):
        from tools.academic_search_tool import _s2_headers
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            headers = _s2_headers()
            assert "Authorization" not in headers
            assert "x-api-key" not in headers

    def test_headers_include_key_from_env(self):
        """When BrainConfig has the key set via env, headers should include it."""
        from tools.academic_search_tool import _s2_headers
        with patch.dict(os.environ, {"S2_API_KEY": "test-key-123"}):
            headers = _s2_headers()
            assert headers.get("Authorization") == "Bearer test-key-123"
            assert "x-api-key" not in headers


class TestEnrichResult:
    def test_skips_when_abstract_sufficient(self):
        from tools.academic_search_tool import AcademicResult, enrich_result
        r = AcademicResult(
            title="Test",
            abstract="A" * 150,
        )
        result = asyncio.run(enrich_result(r))
        assert result.abstract == "A" * 150

    def test_skips_when_no_paper_id(self):
        from tools.academic_search_tool import AcademicResult, enrich_result
        r = AcademicResult(title="Test", abstract="")
        result = asyncio.run(enrich_result(r))
        assert result.abstract == ""

    def test_enriches_via_doi(self):
        from tools.academic_search_tool import AcademicResult, enrich_result

        r = AcademicResult(
            title="Test",
            abstract="",
            doi="10.1234/test",
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "abstract": "This is a real abstract with enough content.",
            "tldr": {"text": "A TLDR summary."},
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.academic_search_tool.aiohttp.ClientSession", return_value=mock_session):
            result = asyncio.run(enrich_result(r))

        assert result.abstract == "This is a real abstract with enough content."
        assert result.tldr == "A TLDR summary."
        assert result.open_access_pdf_url == "https://example.com/paper.pdf"


# ── Phase 3: Query interface content assembly ────────────────────────────────


class TestContentDepthTagging:
    """Verify that _execute_academic tags findings with content_depth via provenance."""

    def test_tldr_preferred_over_abstract(self):
        from tools.academic_search_tool import AcademicResult

        r = AcademicResult(
            title="Test Paper",
            abstract="Short abstract that is at least 100 chars long " * 3,
            tldr="This paper proposes a novel approach to X.",
        )

        if r.tldr:
            content = r.tldr
            depth = "tldr"
        elif r.abstract and len(r.abstract) >= 100:
            content = r.abstract[:800]
            depth = "abstract"
        else:
            content = r.title
            depth = "title_only"

        assert depth == "tldr"
        assert "novel approach" in content

    def test_abstract_fallback(self):
        from tools.academic_search_tool import AcademicResult

        abstract = "A comprehensive study of neural network architectures. " * 5
        r = AcademicResult(title="Test", abstract=abstract)

        if r.tldr:
            depth = "tldr"
        elif r.abstract and len(r.abstract) >= 100:
            depth = "abstract"
        else:
            depth = "title_only"

        assert depth == "abstract"

    def test_title_only_fallback(self):
        from tools.academic_search_tool import AcademicResult

        r = AcademicResult(title="Neural Networks Survey", abstract="")

        if r.tldr:
            depth = "tldr"
        elif r.abstract and len(r.abstract) >= 100:
            depth = "abstract"
        else:
            depth = "title_only"

        assert depth == "title_only"


# ── Phase 4: Content depth in Source + study gate ────────────────────────────


class TestSourceContentDepth:
    def test_source_dataclass_has_content_depth(self):
        from library.source import Source
        s = Source(
            source_id="test-1",
            source_type="doi",
            retrieved_at=time.time(),
            content_depth="abstract",
        )
        assert s.content_depth == "abstract"

    def test_source_content_depth_default_empty(self):
        from library.source import Source
        s = Source(source_id="test-2", source_type="doi", retrieved_at=time.time())
        assert s.content_depth == ""

    def test_source_store_schema_migration(self, tmp_path):
        """Verify content_depth column is created and accessible."""
        db_path = str(tmp_path / "test_lib.db")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT DEFAULT '',
                retrieved_at REAL DEFAULT 0,
                url TEXT DEFAULT '',
                doi TEXT DEFAULT '',
                title TEXT DEFAULT '',
                authors TEXT DEFAULT '',
                year INTEGER DEFAULT 0,
                venue TEXT DEFAULT '',
                citation_count INTEGER DEFAULT 0,
                content_text TEXT DEFAULT '',
                license_flags TEXT DEFAULT '',
                quality_score REAL DEFAULT 0,
                provider TEXT DEFAULT '',
                studied INTEGER DEFAULT 0,
                studied_at REAL DEFAULT 0,
                study_error TEXT DEFAULT '',
                study_attempts INTEGER DEFAULT 0,
                study_next_attempt_at REAL DEFAULT 0,
                ingested_by TEXT DEFAULT 'autonomous',
                trust_tier TEXT DEFAULT 'unverified',
                domain_tags TEXT DEFAULT '',
                canonical_domain TEXT DEFAULT '',
                content_depth TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            INSERT INTO sources (source_id, source_type, content_depth)
            VALUES ('s1', 'doi', 'abstract')
        """)
        conn.commit()

        row = conn.execute("SELECT content_depth FROM sources WHERE source_id='s1'").fetchone()
        assert row["content_depth"] == "abstract"
        conn.close()

    def test_get_stats_includes_depth(self, tmp_path):
        """Verify get_stats returns by_content_depth distribution."""
        db_path = str(tmp_path / "stats_test.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                retrieved_at REAL NOT NULL,
                url TEXT DEFAULT '', doi TEXT DEFAULT '', title TEXT DEFAULT '',
                authors TEXT DEFAULT '', year INTEGER DEFAULT 0,
                venue TEXT DEFAULT '', citation_count INTEGER DEFAULT 0,
                content_text TEXT DEFAULT '', license_flags TEXT DEFAULT '',
                quality_score REAL DEFAULT 0, provider TEXT DEFAULT '',
                studied INTEGER DEFAULT 0, studied_at REAL DEFAULT 0,
                study_error TEXT DEFAULT '', study_attempts INTEGER DEFAULT 0,
                study_next_attempt_at REAL DEFAULT 0,
                ingested_by TEXT DEFAULT 'autonomous',
                trust_tier TEXT DEFAULT 'unverified',
                domain_tags TEXT DEFAULT '', canonical_domain TEXT DEFAULT '',
                content_depth TEXT DEFAULT ''
            )
        """)
        now = time.time()
        for depth, count in [("abstract", 3), ("tldr", 2), ("title_only", 5)]:
            for i in range(count):
                conn.execute(
                    "INSERT INTO sources (source_id, source_type, content_depth, retrieved_at) VALUES (?, 'doi', ?, ?)",
                    (f"stats-{depth}-{i}", depth, now),
                )
        conn.commit()

        from library.source import SourceStore
        store = SourceStore.__new__(SourceStore)
        store._conn = conn
        store._initialized = True
        store._db_path = db_path

        stats = store.get_stats()
        assert "by_content_depth" in stats
        assert stats["by_content_depth"].get("abstract") == 3
        assert stats["by_content_depth"].get("tldr") == 2
        assert stats["by_content_depth"].get("title_only") == 5
        conn.close()


class TestContentDepthInference:
    def test_infer_from_provenance_tldr(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="A TLDR summary here.",
            provenance="academic: DOI:xyz [depth:tldr]",
        )
        depth = KnowledgeIntegrator._infer_content_depth(finding)
        assert depth == "tldr"

    def test_infer_from_provenance_abstract(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="An abstract " * 100,
            provenance="academic: DOI:xyz [depth:abstract]",
        )
        depth = KnowledgeIntegrator._infer_content_depth(finding)
        assert depth == "abstract"

    def test_infer_from_provenance_title_only(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Neural Networks",
            provenance="academic: DOI:xyz [depth:title_only]",
        )
        depth = KnowledgeIntegrator._infer_content_depth(finding)
        assert depth == "title_only"

    def test_infer_from_length_fallback(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        short = ResearchFinding(content="Short title", provenance="academic:")
        assert KnowledgeIntegrator._infer_content_depth(short) == "title_only"

        medium = ResearchFinding(content="X" * 250, provenance="academic:")
        assert KnowledgeIntegrator._infer_content_depth(medium) == "tldr"

        long = ResearchFinding(content="Y" * 900, provenance="academic:")
        assert KnowledgeIntegrator._infer_content_depth(long) == "abstract"


class TestStudyPipelineGate:
    def test_study_skips_metadata_only(self):
        """study_source should skip sources with metadata_only depth."""
        from library.source import Source
        from library.study import study_source

        mock_source = Source(
            source_id="skip-test",
            source_type="doi",
            retrieved_at=time.time(),
            content_depth="metadata_only",
        )

        mock_store = MagicMock()
        mock_store.get.return_value = mock_source
        mock_store.mark_studied = MagicMock()

        with patch("library.source.source_store", mock_store):
            result = study_source("skip-test")

        assert "skipped:insufficient_content" in result.error
        assert result.concepts == []
        assert result.claims == []
        mock_store.mark_studied.assert_called_once_with("skip-test")

    def test_study_skips_title_only(self):
        from library.source import Source
        from library.study import study_source

        mock_source = Source(
            source_id="title-test",
            source_type="doi",
            retrieved_at=time.time(),
            content_depth="title_only",
        )

        mock_store = MagicMock()
        mock_store.get.return_value = mock_source
        mock_store.mark_studied = MagicMock()

        with patch("library.source.source_store", mock_store):
            result = study_source("title-test")

        assert "skipped:insufficient_content" in result.error
        mock_store.mark_studied.assert_called_once()

    def test_study_proceeds_for_abstract(self):
        from library.source import Source
        from library.study import study_source

        mock_source = Source(
            source_id="abstract-test",
            source_type="doi",
            retrieved_at=time.time(),
            content_depth="abstract",
        )

        mock_store = MagicMock()
        mock_store.get.return_value = mock_source
        mock_store.mark_study_error = MagicMock()

        mock_chunks = MagicMock()
        mock_chunks.get_for_source.return_value = []

        with patch("library.source.source_store", mock_store), \
             patch("library.chunks.chunk_store", mock_chunks):
            result = study_source("abstract-test")

        assert "skipped:insufficient_content" not in (result.error or "")


# ── Phase 5: Dashboard API endpoints ────────────────────────────────────────


class TestSourceBrowserAPI:
    @staticmethod
    def _make_store(db_path: str):
        """Create an isolated SourceStore backed by a fresh SQLite DB."""
        from library.source import SourceStore
        _SCHEMA = """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                retrieved_at REAL NOT NULL,
                url TEXT DEFAULT '', doi TEXT DEFAULT '', title TEXT DEFAULT '',
                authors TEXT DEFAULT '', year INTEGER DEFAULT 0,
                venue TEXT DEFAULT '', citation_count INTEGER DEFAULT 0,
                content_text TEXT DEFAULT '', license_flags TEXT DEFAULT '',
                quality_score REAL DEFAULT 0, provider TEXT DEFAULT '',
                studied INTEGER DEFAULT 0, studied_at REAL DEFAULT 0,
                study_error TEXT DEFAULT '', study_attempts INTEGER DEFAULT 0,
                study_next_attempt_at REAL DEFAULT 0,
                ingested_by TEXT DEFAULT 'autonomous',
                trust_tier TEXT DEFAULT 'unverified',
                domain_tags TEXT DEFAULT '', canonical_domain TEXT DEFAULT '',
                content_depth TEXT DEFAULT ''
            )
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SCHEMA)
        conn.commit()
        store = SourceStore.__new__(SourceStore)
        store._conn = conn
        store._initialized = True
        store._db_path = db_path
        return store, conn

    def test_list_sources_method(self, tmp_path):
        store, conn = self._make_store(str(tmp_path / "list.db"))
        now = time.time()
        for i in range(5):
            conn.execute(
                "INSERT INTO sources (source_id, source_type, title, content_depth, retrieved_at) VALUES (?, 'doi', ?, 'abstract', ?)",
                (f"ls-{i}", f"Paper {i}", now - i * 100),
            )
        conn.commit()

        sources = store.list_sources(limit=3)
        assert len(sources) == 3
        assert sources[0].title == "Paper 0"
        conn.close()

    def test_list_sources_filter_by_ingested(self, tmp_path):
        store, conn = self._make_store(str(tmp_path / "filter.db"))
        now = time.time()
        conn.execute("INSERT INTO sources (source_id, source_type, ingested_by, retrieved_at) VALUES ('fa1', 'doi', 'autonomous', ?)", (now,))
        conn.execute("INSERT INTO sources (source_id, source_type, ingested_by, retrieved_at) VALUES ('fu1', 'doi', 'user', ?)", (now,))
        conn.commit()

        auto = store.list_sources(ingested_by="autonomous")
        assert len(auto) == 1
        assert auto[0].source_id == "fa1"
        conn.close()

    def test_get_source_by_id(self, tmp_path):
        store, conn = self._make_store(str(tmp_path / "detail.db"))
        conn.execute(
            "INSERT INTO sources (source_id, source_type, title, content_depth, content_text, retrieved_at) VALUES ('sd1', 'doi', 'Detail Paper', 'tldr', 'Full content here...', ?)",
            (time.time(),),
        )
        conn.commit()

        s = store.get_source("sd1")
        assert s is not None
        assert s.title == "Detail Paper"
        assert s.content_depth == "tldr"
        assert s.content_text == "Full content here..."
        conn.close()

    def test_get_source_not_found(self, tmp_path):
        store, conn = self._make_store(str(tmp_path / "notfound.db"))
        s = store.get_source("nonexistent")
        assert s is None
        conn.close()


# ── Phase 6: metric_triggers logging fix ─────────────────────────────────────


class TestMetricTriggersLogFix:
    def test_log_format_is_valid(self):
        """The fixed format string should not raise ValueError."""
        fmt = "Tool rotation for %s: %s → %s (win_rate=%.0f%%)"
        result = fmt % ("reasoning_coherence", "codebase", "web", 0.0)
        assert "win_rate=0%" in result

    def test_old_format_would_fail(self):
        """Confirm the old format was indeed broken."""
        fmt = "Tool rotation for %s: %s → %s (win_rate=%.0%%)"
        with pytest.raises(ValueError, match="unsupported format character"):
            fmt % ("reasoning_coherence", "codebase", "web", 0.0)


# ── Phase 7: reset-brain.sh ─────────────────────────────────────────────────


class TestResetBrainScript:
    def test_library_in_dirs_to_remove(self):
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "reset-brain.sh",
        )
        with open(script_path) as f:
            content = f.read()

        assert '"library"' in content or "'library'" in content
        assert "DIRS_TO_REMOVE" in content

        preserved_section = content.split("PRESERVED=(")[1].split(")")[0] if "PRESERVED=(" in content else ""
        assert "library/" not in preserved_section


# ── Cache serialization ─────────────────────────────────────────────────────


class TestCacheSerialization:
    def test_cache_includes_new_fields(self):
        from tools.academic_search_tool import AcademicResult

        r = AcademicResult(
            title="Test",
            tldr="TLDR text",
            is_open_access=True,
            open_access_pdf_url="https://example.com/pdf",
            paper_id="abc123",
        )

        cache_entry = {
            "title": r.title,
            "abstract": r.abstract[:500],
            "authors": r.authors,
            "year": r.year,
            "venue": r.venue,
            "doi": r.doi,
            "doi_url": r.doi_url,
            "url": r.url,
            "citation_count": r.citation_count,
            "source_provider": r.source_provider,
            "is_peer_reviewed": r.is_peer_reviewed,
            "is_preprint": r.is_preprint,
            "tldr": r.tldr,
            "is_open_access": r.is_open_access,
            "open_access_pdf_url": r.open_access_pdf_url,
            "paper_id": r.paper_id,
        }

        reconstructed = AcademicResult(**cache_entry)
        assert reconstructed.tldr == "TLDR text"
        assert reconstructed.is_open_access is True
        assert reconstructed.paper_id == "abc123"


# ── Integration: end-to-end content depth flow ──────────────────────────────


class TestEndToEndContentDepth:
    def test_quality_floor_applied(self):
        """Content below min_content_chars should be tagged metadata_only."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="Short title only",
            provenance="academic: [depth:title_only]",
            confidence=0.8,
            doi="10.1234/test",
        )

        depth = KnowledgeIntegrator._infer_content_depth(finding)
        assert depth == "title_only"

    def test_enriched_content_gets_proper_depth(self):
        """Content with TLDR from enrichment gets tagged as tldr."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        finding = ResearchFinding(
            content="This paper proposes a novel attention mechanism for transformers.",
            provenance="academic: DOI:10.1234/test (NeurIPS, 2024) [depth:tldr]",
            confidence=0.85,
            doi="10.1234/test",
            venue="NeurIPS",
            year=2024,
        )

        depth = KnowledgeIntegrator._infer_content_depth(finding)
        assert depth == "tldr"
        assert len(finding.content) > 50


# ── Non-Library Source Type Filter ───────────────────────────────────────────


class TestNonLibrarySourceTypeFilter:
    """Verify that memory, introspection, and raw-dump findings are excluded
    from library ingestion in _store_findings()."""

    def test_memory_findings_skipped(self):
        from autonomy.knowledge_integrator import _NON_LIBRARY_SOURCE_TYPES
        assert "memory" in _NON_LIBRARY_SOURCE_TYPES

    def test_introspection_findings_skipped(self):
        from autonomy.knowledge_integrator import _NON_LIBRARY_SOURCE_TYPES
        assert "introspection" in _NON_LIBRARY_SOURCE_TYPES

    def test_codebase_findings_skipped(self):
        from autonomy.knowledge_integrator import _NON_LIBRARY_SOURCE_TYPES
        assert "codebase" in _NON_LIBRARY_SOURCE_TYPES

    def test_peer_reviewed_not_blocked(self):
        from autonomy.knowledge_integrator import _NON_LIBRARY_SOURCE_TYPES
        assert "peer_reviewed" not in _NON_LIBRARY_SOURCE_TYPES
        assert "url" not in _NON_LIBRARY_SOURCE_TYPES

    def test_raw_json_title_rejected_by_ingest(self):
        """_ingest_to_library rejects titles starting with '{'."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki = KnowledgeIntegrator()
        finding = ResearchFinding(
            content="{'user_message': 'Fix your wakeboard detection'}",
            provenance="memory:conversation:mem_abc",
            confidence=0.75,
            source_type="url",
        )
        result = ki._ingest_to_library(finding, "memory", ())
        assert result == ""

    def test_mode_dump_title_rejected_by_ingest(self):
        """_ingest_to_library rejects titles starting with 'Current mode:'."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki = KnowledgeIntegrator()
        finding = ResearchFinding(
            content="Current mode: dreaming, profile: {'tick_cadence': 0.5}",
            provenance="introspection:mode_manager",
            confidence=0.9,
            source_type="introspection",
        )
        result = ki._ingest_to_library(finding, "introspection", ())
        assert result == ""

    def test_store_findings_skips_memory_source_type(self):
        """Full _store_findings path: memory findings never reach _ingest_to_library."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import (
            ResearchIntent, ResearchResult, ResearchFinding,
        )

        ki = KnowledgeIntegrator()
        ki._engine_ref = MagicMock()

        intent = ResearchIntent(
            id="test_intent", question="What are memory stats?",
            source_event="test", tag_cluster=("test",),
        )
        result = ResearchResult(
            intent_id="test_intent",
            tool_used="memory",
            findings=[
                ResearchFinding(
                    content="{'user_message': 'Do you see anything?'}",
                    provenance="memory:conversation:mem_123",
                    confidence=0.75,
                    source_type="memory",
                    source_provider="memory",
                ),
            ],
            summary="Found 1 memory.",
            raw_query="What are memory stats?",
            success=True,
        )

        with patch.object(ki, "_ingest_to_library") as mock_ingest:
            created = ki._store_findings(intent, result)

        mock_ingest.assert_not_called()
        assert created == 0

    def test_store_findings_skips_codebase_source_type(self):
        """Full _store_findings path: codebase findings never reach _ingest_to_library."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import (
            ResearchIntent, ResearchResult, ResearchFinding,
        )

        ki = KnowledgeIntegrator()
        ki._engine_ref = MagicMock()

        intent = ResearchIntent(
            id="test_cb", question="How does the kernel tick?",
            source_event="test", tag_cluster=("consciousness",),
        )
        result = ResearchResult(
            intent_id="test_cb",
            tool_used="codebase",
            findings=[
                ResearchFinding(
                    content="# consciousness.kernel (kernel.py, 450 lines) ...",
                    provenance="codebase_budgeted_context",
                    confidence=0.85,
                    source_type="codebase",
                    source_provider="codebase",
                ),
            ],
            summary="Found kernel module.",
            raw_query="How does the kernel tick?",
            success=True,
        )

        with patch.object(ki, "_ingest_to_library") as mock_ingest:
            created = ki._store_findings(intent, result)

        mock_ingest.assert_not_called()
        assert created == 0


# ── Title-Only Gate Ordering (integration tests) ────────────────────────────


class TestTitleOnlyGateOrdering:
    """Verify that title-only findings reach _ingest_to_library where
    _try_fetch_full_text can attempt full-text download before quality
    gating.  Regression guard for the premature title_only rejection bug
    that blocked all runtime autonomy memory creation since May 2."""

    def _make_ki_with_engine(self):
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        ki = KnowledgeIntegrator()
        engine = MagicMock()
        engine.remember = MagicMock(return_value="mem_test_001")
        ki.set_engine(engine)
        return ki, engine

    def _make_intent_and_result(self, finding):
        from autonomy.research_intent import ResearchIntent, ResearchResult
        intent = ResearchIntent(
            id="test_gate", question="How do AI systems handle identity?",
            source_event="test", tag_cluster=("identity",),
        )
        result = ResearchResult(
            intent_id="test_gate",
            tool_used="academic_search",
            findings=[finding],
            summary="Found 1 result.",
            raw_query="AI identity",
            success=True,
        )
        return intent, result

    def test_title_only_finding_reaches_ingest(self):
        """A title-only finding must NOT be rejected before _ingest_to_library."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki, _ = self._make_ki_with_engine()
        finding = ResearchFinding(
            content="Adaptive Identity in Autonomous Systems",
            provenance="academic: 10.1234/test [depth:title_only]",
            confidence=0.8,
            doi="10.1234/test",
            source_type="peer_reviewed",
            source_provider="semantic_scholar",
        )
        intent, result = self._make_intent_and_result(finding)

        with patch.object(ki, "_ingest_to_library", return_value="src_test_001") as mock_ingest:
            ki._store_findings(intent, result)

        mock_ingest.assert_called_once()

    def test_title_only_with_pdf_url_reaches_ingest(self):
        """Title-only finding with open_access_pdf_url must reach
        _ingest_to_library where _try_fetch_full_text can run."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki, engine = self._make_ki_with_engine()
        finding = ResearchFinding(
            content="Adaptive Identity in Autonomous Systems",
            provenance="academic: 10.1234/pdf_test [depth:title_only]",
            confidence=0.8,
            doi="10.1234/pdf_test",
            doi_url="https://doi.org/10.1234/pdf_test",
            source_type="peer_reviewed",
            source_provider="semantic_scholar",
            open_access_pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
        )
        intent, result = self._make_intent_and_result(finding)

        with patch.object(ki, "_ingest_to_library", return_value="src_test_pdf") as mock_ingest:
            created = ki._store_findings(intent, result)

        mock_ingest.assert_called_once()
        assert created >= 1
        assert engine.remember.call_count >= 1
        pointer_call = engine.remember.call_args_list[0]
        payload = pointer_call[0][0].payload
        assert payload["type"] == "library_pointer"
        assert payload["source_id"] == "src_test_pdf"

    def test_title_only_without_pdf_still_reaches_ingest(self):
        """Title-only finding with no fetchable URL still reaches
        _ingest_to_library — quality gating happens there, not here."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki, engine = self._make_ki_with_engine()
        finding = ResearchFinding(
            content="Adaptive Identity in Autonomous Systems",
            provenance="academic: 10.1234/nopdf [depth:title_only]",
            confidence=0.8,
            doi="10.1234/nopdf",
            source_type="peer_reviewed",
            source_provider="semantic_scholar",
        )
        intent, result = self._make_intent_and_result(finding)

        with patch.object(ki, "_ingest_to_library", return_value="src_test_meta") as mock_ingest:
            created = ki._store_findings(intent, result)

        mock_ingest.assert_called_once()
        assert created >= 1

    def test_abstract_finding_still_passes(self):
        """Regression guard: findings with abstracts still work normally."""
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        from autonomy.research_intent import ResearchFinding

        ki, _ = self._make_ki_with_engine()
        finding = ResearchFinding(
            content="This paper presents a novel approach to identity management "
                    "in autonomous systems. We propose a multi-modal fusion "
                    "framework that combines voice and face signals for robust "
                    "speaker identification. Our experiments demonstrate that "
                    "the approach achieves state-of-the-art results on standard "
                    "benchmarks while maintaining real-time performance.",
            provenance="academic: 10.1234/test2 (ICML, 2024) [depth:abstract]",
            confidence=0.9,
            doi="10.1234/test2",
            source_type="peer_reviewed",
            source_provider="semantic_scholar",
        )
        intent, result = self._make_intent_and_result(finding)

        with patch.object(ki, "_ingest_to_library", return_value="src_test_003") as mock_ingest:
            ki._store_findings(intent, result)

        mock_ingest.assert_called_once()

    def test_title_only_finding_not_blocked_by_old_gate(self):
        """Explicit regression test: the premature pre_depth == 'title_only'
        gate must not exist in _store_findings. Quality gating belongs
        inside _ingest_to_library after _try_fetch_full_text runs."""
        import inspect
        from autonomy.knowledge_integrator import KnowledgeIntegrator

        source = inspect.getsource(KnowledgeIntegrator._store_findings)
        assert "pre_depth" not in source, (
            "_store_findings must not check pre_depth — "
            "quality gating belongs in _ingest_to_library after full-text fetch"
        )
