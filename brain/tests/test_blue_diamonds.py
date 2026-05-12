"""Tests for Blue Diamonds archive — graduation, dedup, reload, stats, quality gates."""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class MockSource:
    source_id: str = "test_src_001"
    source_type: str = "peer_reviewed"
    doi: str = "10.1234/test"
    url: str = "https://example.com/paper"
    title: str = "Test Paper on Deep Learning"
    authors: str = "Smith, Jones"
    year: int = 2024
    venue: str = "NeurIPS"
    citation_count: int = 42
    content_text: str = (
        "We present a novel deep learning architecture for neural network "
        "training with limited data. The transformer-based model achieves "
        "state-of-the-art results on reinforcement learning benchmarks "
        "using attention mechanisms and episodic memory retrieval."
    )
    content_depth: str = "abstract"
    quality_score: float = 0.65
    domain_tags: str = "deep_learning,neural_networks,peer_reviewed"
    canonical_domain: str = "example.com"
    provider: str = "semantic_scholar"
    license_flags: str = "open_access"
    ingested_by: str = "autonomous"
    trust_tier: str = "verified"


@dataclass
class MockChunk:
    chunk_id: str = "chunk_001"
    source_id: str = "test_src_001"
    text: str = "This is chunk text about deep learning methods."
    chunk_type: str = "abstract"
    concepts: list = None
    offset: int = 0

    def __post_init__(self):
        if self.concepts is None:
            self.concepts = ["deep_learning", "neural_networks"]


class TestBlueDiamondsArchive(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        from library.blue_diamonds import BlueDiamondsArchive
        self.archive = BlueDiamondsArchive(db_path=self.tmp_dir)
        self.archive.init()

    def tearDown(self):
        self.archive.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_graduate_source_success(self):
        source = MockSource()
        chunks = [MockChunk(), MockChunk(chunk_id="chunk_002", offset=1)]
        result = self.archive.graduate(
            source, chunks,
            concepts=["deep_learning", "neural_networks"],
            claims=["DNNs outperform traditional methods"],
        )
        self.assertTrue(result)
        self.assertTrue(self.archive.is_archived("test_src_001"))

    def test_graduate_dedup(self):
        source = MockSource()
        chunks = [MockChunk()]
        self.assertTrue(self.archive.graduate(source, chunks))
        self.assertFalse(self.archive.graduate(source, chunks))

    def test_has_content(self):
        self.assertFalse(self.archive.has_content())
        self.archive.graduate(MockSource(), [MockChunk()])
        self.assertTrue(self.archive.has_content())

    def test_reload_all_returns_data(self):
        source = MockSource()
        chunks = [
            MockChunk(chunk_id="c1", text="First chunk text."),
            MockChunk(chunk_id="c2", text="Second chunk text.", offset=1),
        ]
        self.archive.graduate(source, chunks,
                              concepts=["deep_learning"],
                              claims=["Claim one"])

        entries = self.archive.reload_all()
        self.assertEqual(len(entries), 1)

        src_dict, chunk_dicts = entries[0]
        self.assertEqual(src_dict["source_id"], "test_src_001")
        self.assertEqual(src_dict["title"], "Test Paper on Deep Learning")
        self.assertEqual(src_dict["content_depth"], "abstract")
        self.assertEqual(len(chunk_dicts), 2)
        self.assertEqual(chunk_dicts[0]["text"], "First chunk text.")
        self.assertEqual(chunk_dicts[1]["text"], "Second chunk text.")

    def test_stats_accuracy(self):
        self.archive.graduate(
            MockSource(content_depth="abstract"),
            [MockChunk()],
        )
        self.archive.graduate(
            MockSource(
                source_id="src_002", content_depth="full_text",
                content_text=(
                    "A comprehensive survey of episodic memory systems in "
                    "cognitive architectures reveals that retrieval augmented "
                    "generation combined with semantic search provides the "
                    "best performance for knowledge-intensive tasks."
                ),
            ),
            [MockChunk(chunk_id="c2", source_id="src_002"),
             MockChunk(chunk_id="c3", source_id="src_002")],
        )

        stats = self.archive.get_stats()
        self.assertEqual(stats["total_diamonds"], 2)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["by_depth"]["abstract"], 1)
        self.assertEqual(stats["by_depth"]["full_text"], 1)
        self.assertIsNotNone(stats["last_graduated_at"])
        self.assertEqual(stats["archive_path"], str(self.tmp_dir))

    def test_log_reload(self):
        self.archive.log_reload(5, 20, trigger="gestation")
        stats = self.archive.get_stats()
        self.assertEqual(len(stats["reload_history"]), 1)
        self.assertEqual(stats["reload_history"][0]["diamonds_loaded"], 5)
        self.assertEqual(stats["reload_history"][0]["chunks_loaded"], 20)
        self.assertEqual(stats["reload_history"][0]["trigger"], "gestation")

    def test_audit_trail_written(self):
        self.archive.graduate(MockSource(), [MockChunk()])
        self.archive.log_rejection("bad_source", "quality:0.1")

        audit_path = self.tmp_dir / "audit.jsonl"
        self.assertTrue(audit_path.exists())
        lines = audit_path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)

        grad_entry = json.loads(lines[0])
        self.assertEqual(grad_entry["event"], "graduated")
        self.assertEqual(grad_entry["diamond_id"], "test_src_001")

        reject_entry = json.loads(lines[1])
        self.assertEqual(reject_entry["event"], "rejected")
        self.assertEqual(reject_entry["source_id"], "bad_source")

    def test_multiple_sources_ordered_by_graduation_time(self):
        import time
        s1 = MockSource(source_id="s1", title="Paper One")
        s2 = MockSource(
            source_id="s2", title="Paper Two",
            content_text=(
                "A comprehensive survey of episodic memory systems in "
                "cognitive architectures reveals that retrieval augmented "
                "generation combined with semantic search provides the "
                "best performance for knowledge-intensive tasks."
            ),
        )
        self.archive.graduate(s1, [MockChunk(chunk_id="c1", source_id="s1")])
        time.sleep(0.01)
        self.archive.graduate(s2, [MockChunk(chunk_id="c2", source_id="s2")])

        entries = self.archive.reload_all()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][0]["source_id"], "s1")
        self.assertEqual(entries[1][0]["source_id"], "s2")


class TestGraduationGateLogic(unittest.TestCase):

    def test_title_only_not_eligible(self):
        from library.blue_diamonds import GRADUATION_ELIGIBLE_DEPTHS
        self.assertNotIn("title_only", GRADUATION_ELIGIBLE_DEPTHS)
        self.assertNotIn("metadata_only", GRADUATION_ELIGIBLE_DEPTHS)
        self.assertNotIn("tldr", GRADUATION_ELIGIBLE_DEPTHS)

    def test_abstract_is_eligible(self):
        from library.blue_diamonds import GRADUATION_ELIGIBLE_DEPTHS
        self.assertIn("abstract", GRADUATION_ELIGIBLE_DEPTHS)
        self.assertIn("full_text", GRADUATION_ELIGIBLE_DEPTHS)

    def test_quality_threshold(self):
        from library.blue_diamonds import GRADUATION_MIN_QUALITY
        self.assertGreaterEqual(GRADUATION_MIN_QUALITY, 0.5)
        self.assertLessEqual(GRADUATION_MIN_QUALITY, 0.7)

    def test_unverified_quality_threshold_higher(self):
        from library.blue_diamonds import (
            GRADUATION_MIN_QUALITY, GRADUATION_MIN_QUALITY_UNVERIFIED,
        )
        self.assertGreater(GRADUATION_MIN_QUALITY_UNVERIFIED, GRADUATION_MIN_QUALITY)
        self.assertGreaterEqual(GRADUATION_MIN_QUALITY_UNVERIFIED, 0.65)


class TestQualityGates(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        from library.blue_diamonds import BlueDiamondsArchive
        self.archive = BlueDiamondsArchive(db_path=self.tmp_dir)
        self.archive.init()

    def tearDown(self):
        self.archive.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_rejects_non_english(self):
        source = MockSource(
            source_id="non_eng",
            content_text=(
                "Permasalahan kesehatan mental di Indonesia memerlukan "
                "sistem dukungan psikologis awal yang mudah diakses, aman, "
                "dan bertanggung jawab. Perkembangan Large Language Model "
                "memungkinkan pengembangan chatbot kesehatan mental yang "
                "mampu menghasilkan respons empatik dan kontekstual."
            ),
        )
        self.assertFalse(self.archive.graduate(source, [MockChunk()]))
        self.assertFalse(self.archive.is_archived("non_eng"))

    def test_rejects_irrelevant_domain(self):
        source = MockSource(
            source_id="irrelevant",
            domain_tags="supply_chain,logistics,peer_reviewed",
            content_text=(
                "Modern supply chains operate under persistent volatility "
                "where disruptions driven by geopolitical instability and "
                "logistics bottlenecks routinely challenge operational "
                "continuity in enterprise resource planning systems."
            ),
        )
        self.assertFalse(self.archive.graduate(source, [MockChunk()]))
        self.assertFalse(self.archive.is_archived("irrelevant"))

    def test_accepts_relevant_by_tags(self):
        source = MockSource(
            source_id="tag_relevant",
            domain_tags="reinforcement_learning,policy_gradient,peer_reviewed",
        )
        self.assertTrue(self.archive.graduate(source, [MockChunk(source_id="tag_relevant")]))

    def test_accepts_relevant_by_content(self):
        source = MockSource(
            source_id="content_relevant",
            domain_tags="proceedings_of_foo,peer_reviewed",
            content_text=(
                "This paper proposes a novel neural network architecture "
                "for deep learning in reinforcement learning environments. "
                "We demonstrate that attention mechanisms improve episodic "
                "memory retrieval in cognitive architectures."
            ),
        )
        self.assertTrue(self.archive.graduate(
            source, [MockChunk(source_id="content_relevant")],
        ))

    def test_rejects_duplicate_content(self):
        s1 = MockSource(source_id="orig")
        s2 = MockSource(
            source_id="dupe",
            doi="10.9999/different",
            content_text=s1.content_text,
        )
        self.assertTrue(self.archive.graduate(s1, [MockChunk(source_id="orig")]))
        self.assertFalse(self.archive.graduate(s2, [MockChunk(source_id="dupe")]))

    def test_rejects_low_quality_unverified(self):
        source = MockSource(
            source_id="low_unverified",
            source_type="unverified",
            quality_score=0.55,
        )
        self.assertFalse(self.archive.graduate(source, [MockChunk()]))

    def test_accepts_high_quality_unverified(self):
        source = MockSource(
            source_id="high_unverified",
            source_type="unverified",
            quality_score=0.75,
        )
        self.assertTrue(self.archive.graduate(
            source, [MockChunk(source_id="high_unverified")],
        ))

    def test_purge_diamond(self):
        source = MockSource()
        self.archive.graduate(source, [MockChunk()])
        self.assertTrue(self.archive.is_archived("test_src_001"))
        self.assertTrue(self.archive.purge_diamond("test_src_001", "test_cleanup"))
        self.assertFalse(self.archive.is_archived("test_src_001"))

    def test_purge_nonexistent(self):
        self.assertFalse(self.archive.purge_diamond("doesnt_exist"))


class TestLanguageDetection(unittest.TestCase):

    def test_english_detected(self):
        from library.blue_diamonds import _is_english
        text = (
            "We present a novel approach to deep learning that uses "
            "attention mechanisms for improved neural network training "
            "with limited data in reinforcement learning environments."
        )
        self.assertTrue(_is_english(text))

    def test_non_english_detected(self):
        from library.blue_diamonds import _is_english
        text = (
            "Permasalahan kesehatan mental di Indonesia memerlukan "
            "sistem dukungan psikologis awal yang mudah diakses aman "
            "dan bertanggung jawab perkembangan besar model bahasa "
            "memungkinkan pengembangan chatbot kesehatan mental yang "
            "mampu menghasilkan respons empatik dan kontekstual."
        )
        self.assertFalse(_is_english(text))

    def test_short_text_passes(self):
        from library.blue_diamonds import _is_english
        self.assertTrue(_is_english("short text"))


class TestRelevanceDetection(unittest.TestCase):

    def test_relevant_by_tags(self):
        from library.blue_diamonds import _is_relevant_to_jarvis
        self.assertTrue(_is_relevant_to_jarvis(
            "neural_network,training,peer_reviewed", "",
        ))

    def test_relevant_by_content(self):
        from library.blue_diamonds import _is_relevant_to_jarvis
        self.assertTrue(_is_relevant_to_jarvis(
            "proceedings_of_foo,peer_reviewed",
            "This paper demonstrates deep learning with neural network architectures",
        ))

    def test_irrelevant_rejected(self):
        from library.blue_diamonds import _is_relevant_to_jarvis
        self.assertFalse(_is_relevant_to_jarvis(
            "supply_chain,logistics,peer_reviewed",
            "Supply chain disruptions affect enterprise resource planning systems",
        ))

    def test_venue_tags_ignored_for_relevance(self):
        from library.blue_diamonds import _is_relevant_to_jarvis
        self.assertFalse(_is_relevant_to_jarvis(
            "journal_of_banking_and_financial_dynamics,peer_reviewed",
            "Real-time payment processing has become critical infrastructure",
        ))


class TestSingleton(unittest.TestCase):

    def test_get_instance_returns_same_object(self):
        from library.blue_diamonds import BlueDiamondsArchive
        BlueDiamondsArchive._instance = None
        a = BlueDiamondsArchive.get_instance()
        b = BlueDiamondsArchive.get_instance()
        self.assertIs(a, b)
        BlueDiamondsArchive._instance = None


if __name__ == "__main__":
    unittest.main()
