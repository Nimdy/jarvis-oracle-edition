"""Tests for consolidation pipeline quality gates.

Verifies the three fixes for the recursive hollow consolidation bug:
1. Dream cycle excludes already-consolidated memories from clustering input
2. _build_summary strips meta-headers and rejects hollow content
3. _score_cluster rejects clusters dominated by already-consolidated memories
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory
from memory.consolidation import MemoryConsolidationEngine
from memory.clustering import MemoryCluster


def _mem(
    mid: str,
    payload: str | dict,
    tags: tuple[str, ...] = (),
    weight: float = 0.55,
    access_count: int = 0,
) -> Memory:
    return Memory(
        id=mid,
        timestamp=time.time(),
        weight=weight,
        tags=tags,
        payload=payload,
        type="contextual_insight",
        associations=(),
        decay_rate=0.02,
        priority=600,
        provenance="consolidation" if "consolidated" in tags else "observed",
        access_count=access_count,
    )


def _cluster(mids: list[str], coherence: float = 0.8, topic: str = "test") -> MemoryCluster:
    return MemoryCluster(
        id="clust_test",
        cluster_type="experiential",
        memory_ids=mids,
        coherence=coherence,
        topic=topic,
    )


# ── Strip Meta-Headers ──

class TestStripMetaHeaders:
    def test_strips_consolidated_from_header(self):
        engine = MemoryConsolidationEngine()
        text = "[Consolidated from 10 memories] Actual content here"
        assert engine._strip_meta_headers(text) == "Actual content here"

    def test_strips_dream_artifact_header(self):
        engine = MemoryConsolidationEngine()
        text = "[Dream artifact: consolidation_proposal] Some real text"
        assert engine._strip_meta_headers(text) == "Some real text"

    def test_strips_consolidation_stats(self):
        engine = MemoryConsolidationEngine()
        text = "Consolidation: consolidated (5 memories, coherence=0.87) remaining"
        assert engine._strip_meta_headers(text) == "remaining"

    def test_strips_plus_more(self):
        engine = MemoryConsolidationEngine()
        text = "Some text [+3 more] trailing"
        assert engine._strip_meta_headers(text) == "Some text trailing"

    def test_strips_nested_meta_chains(self):
        engine = MemoryConsolidationEngine()
        text = (
            "[Consolidated from 10 memories] "
            "[Dream artifact: consolidation_proposal] "
            "Consolidation: consolidated (5 memories, coherence=0.87) | "
            "[Dream artifact: consolidation_proposal] "
            "Consolidation: dream_artifact (3 memories, coherence=1.00)"
        )
        result = engine._strip_meta_headers(text)
        assert len(result) < 20, f"Should be nearly empty after stripping, got: {result!r}"

    def test_preserves_real_content(self):
        engine = MemoryConsolidationEngine()
        text = "Scene observation: Person sitting at desk with multiple monitors"
        assert engine._strip_meta_headers(text) == text

    def test_extracts_content_from_mixed(self):
        engine = MemoryConsolidationEngine()
        text = "[Consolidated from 5 memories] Scene observation: Person at desk | Had a conversation about coffee"
        result = engine._strip_meta_headers(text)
        assert "Scene observation" in result
        assert "coffee" in result


# ── Build Summary Quality Gate ──

class TestBuildSummaryQuality:
    def test_rejects_cluster_of_hollow_consolidations(self):
        engine = MemoryConsolidationEngine()
        hollow_payload = {
            "text": (
                "[Consolidated from 10 memories] "
                "[Dream artifact: consolidation_proposal] "
                "Consolidation: consolidated (5 memories, coherence=0.87) | "
                "[Dream artifact: consolidation_proposal] "
                "Consolidation: consolidated (8 memories, coherence=0.94)"
            ),
            "summary": (
                "[Dream artifact: consolidation_proposal] "
                "Consolidation: consolidated (5 memories, coherence=0.87) | "
                "[Dream artifact: consolidation_proposal] "
                "Consolidation: consolidated (8 memories, coherence=0.94)"
            ),
        }
        mems = [
            _mem(f"m{i}", hollow_payload, tags=("consolidated", "dream_consolidation"))
            for i in range(5)
        ]
        cluster = _cluster([m.id for m in mems])
        result = engine._build_summary(cluster, mems)
        assert result is None, "Should reject cluster of hollow consolidations"

    def test_accepts_cluster_with_real_content(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem("m1", "Scene observation: Person sitting at desk with monitors"),
            _mem("m2", "Had a conversation about morning coffee routine"),
            _mem("m3", "User mentioned they enjoy coding leading-edge technology"),
        ]
        cluster = _cluster([m.id for m in mems])
        result = engine._build_summary(cluster, mems)
        assert result is not None
        assert "Scene observation" in result.payload["text"]

    def test_filters_hollow_texts_keeps_real_from_mixed_cluster(self):
        engine = MemoryConsolidationEngine()
        hollow = {
            "text": "[Consolidated from 5 memories] Consolidation: consolidated (3 memories, coherence=0.90)",
        }
        mems = [
            _mem("m1", hollow, tags=("consolidated", "dream_consolidation")),
            _mem("m2", "Real observation about room layout"),
            _mem("m3", "User asked about system status and got detailed response"),
            _mem("m4", hollow, tags=("consolidated", "dream_consolidation")),
        ]
        cluster = _cluster([m.id for m in mems])
        result = engine._build_summary(cluster, mems)
        assert result is not None
        assert "[Consolidated from" not in result.payload["summary"]
        assert "room layout" in result.payload["text"] or "system status" in result.payload["text"]


# ── Score Cluster Rejects Consolidated-Heavy ──

class TestScoreClusterGuard:
    def test_rejects_all_consolidated_cluster(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem(f"m{i}", "some text", tags=("consolidated", "dream_consolidation"))
            for i in range(5)
        ]
        mems_map = {m.id: m for m in mems}
        cluster = _cluster([m.id for m in mems])
        score = engine._score_cluster(cluster, mems_map)
        assert score == -1.0

    def test_rejects_majority_consolidated_cluster(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem("m1", "real content", tags=()),
            _mem("m2", "some text", tags=("consolidated", "dream_consolidation")),
            _mem("m3", "some text", tags=("consolidated", "dream_consolidation")),
            _mem("m4", "some text", tags=("consolidated", "dream_consolidation")),
        ]
        mems_map = {m.id: m for m in mems}
        cluster = _cluster([m.id for m in mems])
        score = engine._score_cluster(cluster, mems_map)
        assert score == -1.0

    def test_accepts_cluster_without_consolidated(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem(f"m{i}", "real observation content", tags=("scene", "vision"), access_count=5)
            for i in range(4)
        ]
        mems_map = {m.id: m for m in mems}
        cluster = _cluster([m.id for m in mems], coherence=0.8)
        score = engine._score_cluster(cluster, mems_map)
        assert score > 0

    def test_accepts_cluster_with_minority_consolidated(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem("m1", "real content", tags=("scene",), access_count=3),
            _mem("m2", "real content 2", tags=("scene",), access_count=3),
            _mem("m3", "real content 3", tags=("scene",), access_count=3),
            _mem("m4", "consolidated text", tags=("consolidated",), access_count=1),
        ]
        mems_map = {m.id: m for m in mems}
        cluster = _cluster([m.id for m in mems], coherence=0.8)
        score = engine._score_cluster(cluster, mems_map)
        assert score > 0, "Minority consolidated should still allow consolidation"


# ── Full Pipeline ──

class TestFullPipelineQuality:
    def test_run_consolidation_skips_hollow_clusters(self):
        engine = MemoryConsolidationEngine()
        hollow = {
            "text": (
                "[Consolidated from 10 memories] "
                "[Dream artifact: consolidation_proposal] "
                "Consolidation: consolidated (5 memories, coherence=0.87)"
            ),
        }
        mems = [
            _mem(f"m{i}", hollow, tags=("consolidated", "dream_consolidation"))
            for i in range(6)
        ]
        cluster = _cluster([m.id for m in mems], coherence=0.9)
        result = engine.run_consolidation([cluster], mems)
        assert result.count == 0, "Should not produce summaries from hollow clusters"

    def test_run_consolidation_produces_real_summaries(self):
        engine = MemoryConsolidationEngine()
        mems = [
            _mem("m1", "User mentioned coffee cup on right side of desk", tags=("scene",), access_count=5),
            _mem("m2", "Scene observation: person at desk with monitors", tags=("scene",), access_count=3),
            _mem("m3", "User said they drink coffee every morning", tags=("conversation",), access_count=4),
            _mem("m4", "Observed keyboard with blue backlighting on desk", tags=("scene",), access_count=2),
        ]
        cluster = _cluster([m.id for m in mems], coherence=0.75)
        result = engine.run_consolidation([cluster], mems)
        assert result.count == 1
        summary = result.summaries[0]
        assert "coffee" in summary.payload["text"].lower() or "desk" in summary.payload["text"].lower()

    def test_exact_duplicate_consolidation_excluded(self):
        """Consolidated memories from cycle N should not re-consolidate in cycle N+1."""
        engine = MemoryConsolidationEngine()
        cycle1_output = _mem(
            "consol_abc123",
            {
                "text": "[Consolidated from 3 memories] Coffee routine discussed | Desk setup observed | Morning greeting",
                "summary": "Coffee routine discussed | Desk setup observed | Morning greeting",
                "source_ids": ["m1", "m2", "m3"],
            },
            tags=("consolidated", "dream_consolidation"),
        )
        more_consolidated = [
            _mem(
                f"consol_{i}",
                {
                    "text": f"[Consolidated from {i+2} memories] some consolidated content {i}",
                    "summary": f"some consolidated content {i}",
                },
                tags=("consolidated", "dream_consolidation"),
            )
            for i in range(4)
        ]
        all_mems = [cycle1_output] + more_consolidated
        cluster = _cluster([m.id for m in all_mems], coherence=0.9)
        mems_map = {m.id: m for m in all_mems}
        score = engine._score_cluster(cluster, mems_map)
        assert score == -1.0, "Cluster of all consolidated memories should be rejected"
