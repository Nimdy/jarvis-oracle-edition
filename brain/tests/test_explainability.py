"""Tests for Phase 6.4 Explainability Layer."""

import unittest
from unittest.mock import patch, MagicMock

from reasoning.explainability import (
    build_provenance_trace,
    build_evidence_chain,
    cite_sources,
    compact_trace,
    _extract_citations,
    _build_summary,
    _narrate_step,
)


class TestBuildProvenanceTrace(unittest.TestCase):
    """Tests for build_provenance_trace()."""

    def test_none_seed_returns_unavailable(self):
        result = build_provenance_trace(None)
        self.assertFalse(result["available"])
        self.assertIn("No provenance", result["summary"])

    def test_empty_seed_returns_unavailable(self):
        result = build_provenance_trace({})
        self.assertFalse(result["available"])

    def test_grounded_tool_data_trace(self):
        seed = {
            "route": "status",
            "response_class": "self_status",
            "provenance_verdict": "grounded_tool_data",
            "confidence": 0.95,
            "native_used": True,
            "safety_flags": ["status_mode"],
            "meaning_frame": {"facts": ["CPU: 12%", "Memory: 4.2GB"], "fact_count": 2},
            "grounding_payload": {"autonomy": {"mode": "sleep"}},
        }
        result = build_provenance_trace(seed, conversation_id="conv-1")
        self.assertTrue(result["available"])
        self.assertEqual(result["provenance_verdict"], "grounded_tool_data")
        self.assertEqual(result["source_description"], "Live system data (status tool)")
        self.assertEqual(result["response_class"], "self_status")
        self.assertEqual(result["response_class_label"], "System Status Report")
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertTrue(result["native_used"])
        self.assertEqual(result["fact_count"], 2)
        self.assertEqual(result["conversation_id"], "conv-1")
        self.assertIn("bounded articulation", result["summary"])
        self.assertIn("95%", result["summary"])

    def test_memory_recall_trace(self):
        seed = {
            "route": "memory",
            "response_class": "memory_recall",
            "provenance_verdict": "grounded_memory_context",
            "confidence": 0.85,
            "native_used": False,
            "safety_flags": [],
            "meaning_frame": {"facts": [], "fact_count": 0},
            "grounding_payload": {"memory_context": "User prefers dark mode."},
        }
        result = build_provenance_trace(seed)
        self.assertTrue(result["available"])
        self.assertEqual(result["source_description"], "Verified memory records")
        self.assertIn("LLM generation", result["summary"])
        self.assertTrue(len(result["citations"]) > 0)
        self.assertEqual(result["citations"][0]["type"], "memory")

    def test_reflective_introspection_trace(self):
        seed = {
            "route": "introspection",
            "response_class": "self_introspection",
            "provenance_verdict": "reflective_introspection",
            "confidence": 0.80,
            "native_used": False,
            "safety_flags": ["capability_gate_active"],
            "meaning_frame": {"facts": ["a", "b", "c"], "fact_count": 3},
            "grounding_payload": {},
        }
        result = build_provenance_trace(seed)
        self.assertTrue(result["available"])
        self.assertEqual(result["response_class_label"], "Self-Reflective Analysis")
        self.assertEqual(result["fact_count"], 3)
        self.assertIn("capability_gate_active", result["safety_flags"])

    def test_unknown_verdict_gets_fallback_description(self):
        seed = {
            "provenance_verdict": "some_future_verdict",
            "confidence": 0.5,
            "native_used": False,
            "meaning_frame": {},
            "grounding_payload": None,
        }
        result = build_provenance_trace(seed)
        self.assertTrue(result["available"])
        self.assertEqual(result["source_description"], "Unclassified source")

    def test_grounded_prefix_gets_generic_description(self):
        seed = {
            "provenance_verdict": "grounded_new_type",
            "confidence": 0.7,
            "native_used": False,
            "meaning_frame": {},
        }
        result = build_provenance_trace(seed)
        self.assertEqual(result["source_description"], "Grounded data source")

    def test_negative_prefix_gets_training_description(self):
        seed = {
            "provenance_verdict": "negative:capability_gate_rewrite",
            "confidence": 0.0,
            "native_used": False,
            "meaning_frame": {},
        }
        result = build_provenance_trace(seed)
        self.assertEqual(result["source_description"], "Negative example (training data)")

    def test_ledger_entry_id_included(self):
        seed = {"provenance_verdict": "grounded_tool_data", "confidence": 0.9}
        result = build_provenance_trace(seed, ledger_entry_id="led_abc123")
        self.assertEqual(result["ledger_entry_id"], "led_abc123")


class TestExtractCitations(unittest.TestCase):
    """Tests for _extract_citations()."""

    def test_none_grounding_returns_empty(self):
        self.assertEqual(_extract_citations(None, "self_status"), [])

    def test_memory_recall_extracts_context(self):
        grounding = {"memory_context": "User likes jazz music and plays guitar."}
        cites = _extract_citations(grounding, "memory_recall")
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["type"], "memory")
        self.assertIn("jazz", cites[0]["detail"])

    def test_self_status_extracts_subsystem_data(self):
        grounding = {
            "autonomy": {"mode": "sleep", "uptime": 135},
            "health": {"confidence": 0.85},
        }
        cites = _extract_citations(grounding, "self_status")
        self.assertEqual(len(cites), 2)
        labels = [c["label"] for c in cites]
        self.assertIn("Live autonomy data", labels)
        self.assertIn("Live health data", labels)

    def test_identity_answer_extracts_speaker(self):
        grounding = {"name": "Aaron", "confidence": 0.92}
        cites = _extract_citations(grounding, "identity_answer")
        self.assertEqual(len(cites), 1)
        self.assertIn("Aaron", cites[0]["detail"])

    def test_string_grounding_creates_text_citation(self):
        cites = _extract_citations("Some raw context string", "general")
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["type"], "text")

    def test_long_detail_is_truncated(self):
        grounding = {"memory_context": "x" * 500}
        cites = _extract_citations(grounding, "memory_recall")
        self.assertTrue(cites[0]["detail"].endswith("..."))
        self.assertTrue(len(cites[0]["detail"]) <= 203)

    def test_learning_job_extracts_jobs(self):
        grounding = {
            "jobs": [
                {"topic": "Neural networks", "status": "completed"},
                {"topic": "Graph theory", "status": "in_progress"},
            ]
        }
        cites = _extract_citations(grounding, "recent_learning")
        self.assertEqual(len(cites), 2)
        self.assertEqual(cites[0]["label"], "Neural networks")

    def test_capability_status_extracts_skills(self):
        grounding = {
            "skills": [
                {"name": "web_search", "status": "verified"},
                {"name": "code_edit", "status": "learning"},
            ]
        }
        cites = _extract_citations(grounding, "capability_status")
        self.assertEqual(len(cites), 2)
        self.assertEqual(cites[0]["type"], "registry")


class TestCompactTrace(unittest.TestCase):
    """Tests for compact_trace() — hot path metadata."""

    def test_none_seed_returns_fallback(self):
        result = compact_trace(None)
        self.assertEqual(result["provenance"], "fallback_unclassified")
        self.assertEqual(result["source"], "fallback:missing_language_seed")
        self.assertAlmostEqual(result["confidence"], 0.0)
        self.assertFalse(result["native"])
        self.assertEqual(result["response_class"], "unknown")
        self.assertTrue(result["fallback"])

    def test_returns_minimal_fields(self):
        seed = {
            "provenance_verdict": "grounded_tool_data",
            "confidence": 0.95,
            "native_used": True,
            "response_class": "self_status",
        }
        result = compact_trace(seed)
        self.assertEqual(result["provenance"], "grounded_tool_data")
        self.assertEqual(result["source"], "Live system data (status tool)")
        self.assertAlmostEqual(result["confidence"], 0.95)
        self.assertTrue(result["native"])
        self.assertEqual(result["response_class"], "self_status")

    def test_unknown_verdict(self):
        seed = {"provenance_verdict": "unknown", "confidence": 0.0}
        result = compact_trace(seed)
        self.assertEqual(result["provenance"], "unknown")
        self.assertEqual(result["source"], "")


class TestCiteSources(unittest.TestCase):
    """Tests for cite_sources()."""

    def test_no_seed_no_memories_returns_empty(self):
        self.assertEqual(cite_sources(None), [])

    def test_seed_citations_extracted(self):
        seed = {
            "response_class": "memory_recall",
            "grounding_payload": {"memory_context": "Likes jazz."},
        }
        cites = cite_sources(seed)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["type"], "memory")

    def test_memory_results_cited(self):
        memories = [
            {"id": "mem_001", "payload": "Prefers dark mode", "provenance": "user_claim", "weight": 0.8},
            {"id": "mem_002", "payload": "Born in 1990", "provenance": "conversation", "weight": 0.5},
        ]
        cites = cite_sources(None, memory_results=memories)
        self.assertEqual(len(cites), 2)
        self.assertEqual(cites[0]["provenance"], "user_claim")
        self.assertEqual(cites[1]["provenance"], "conversation")

    def test_combined_seed_and_memories(self):
        seed = {
            "response_class": "memory_recall",
            "grounding_payload": {"memory_context": "Context here"},
        }
        memories = [{"id": "mem_x", "payload": "Data", "provenance": "observed", "weight": 0.6}]
        cites = cite_sources(seed, memory_results=memories)
        self.assertEqual(len(cites), 2)


class TestBuildEvidenceChain(unittest.TestCase):
    """Tests for build_evidence_chain()."""

    def test_no_ledger_returns_unavailable(self):
        with patch.dict("sys.modules", {"consciousness.attribution_ledger": None}):
            # Force import failure
            with patch("reasoning.explainability.build_evidence_chain") as mock:
                mock.return_value = {"available": False, "reason": "test"}
                result = mock("")
                self.assertFalse(result["available"])

    @patch("reasoning.explainability.build_evidence_chain.__module__", "reasoning.explainability")
    def test_with_mock_ledger(self):
        mock_ledger = MagicMock()
        mock_ledger.query.return_value = [
            {
                "entry_id": "led_root",
                "root_entry_id": "led_root",
                "conversation_id": "conv-1",
                "subsystem": "conversation",
                "event_type": "user_message",
                "ts": 1000.0,
                "confidence": 1.0,
                "outcome": "success",
                "data": {"text": "How are you?"},
            },
        ]
        mock_ledger.get_chain.return_value = [
            {
                "entry_id": "led_root",
                "root_entry_id": "led_root",
                "subsystem": "conversation",
                "event_type": "user_message",
                "ts": 1000.0,
                "confidence": 1.0,
                "outcome": "success",
                "data": {"text": "How are you?"},
            },
            {
                "entry_id": "led_resp",
                "root_entry_id": "led_root",
                "subsystem": "conversation",
                "event_type": "response_complete",
                "ts": 1001.5,
                "confidence": 0.95,
                "outcome": "success",
                "data": {"tool": "introspection", "reply_len": 120, "latency_ms": 1500},
            },
        ]

        with patch("consciousness.attribution_ledger.attribution_ledger", mock_ledger):
            result = build_evidence_chain("conv-1")
            self.assertTrue(result["available"])
            self.assertEqual(result["step_count"], 2)
            self.assertEqual(result["root_entry_id"], "led_root")
            self.assertIn("User sent a message", result["steps"][0]["narrative"])
            self.assertIn("generated a response", result["steps"][1]["narrative"])
            self.assertIn("1.", result["narrative"])
            self.assertIn("2.", result["narrative"])


class TestNarrateStep(unittest.TestCase):
    """Tests for _narrate_step()."""

    def test_user_message_narration(self):
        entry = {
            "subsystem": "conversation",
            "event_type": "user_message",
            "outcome": "success",
            "data": {"text": "What time is it?"},
        }
        narration = _narrate_step(entry)
        self.assertIn("User sent a message", narration)
        self.assertIn("What time is it?", narration)
        self.assertIn("[success]", narration)

    def test_response_complete_narration(self):
        entry = {
            "subsystem": "conversation",
            "event_type": "response_complete",
            "outcome": "pending",
            "data": {"tool": "time"},
        }
        narration = _narrate_step(entry)
        self.assertIn("generated a response", narration)
        self.assertIn("tool: time", narration)

    def test_unknown_event_gets_generic_narration(self):
        entry = {
            "subsystem": "custom_module",
            "event_type": "special_event",
            "outcome": "pending",
            "data": {},
        }
        narration = _narrate_step(entry)
        self.assertIn("custom_module", narration)
        self.assertIn("special_event", narration)

    def test_pending_outcome_not_appended(self):
        entry = {
            "subsystem": "conversation",
            "event_type": "user_message",
            "outcome": "pending",
            "data": {},
        }
        narration = _narrate_step(entry)
        self.assertNotIn("[pending]", narration)


class TestBuildSummary(unittest.TestCase):
    """Tests for _build_summary()."""

    def test_bounded_path_summary(self):
        summary = _build_summary("Live system data", "System Status Report", 0.95, True, 5)
        self.assertIn("bounded articulation", summary)
        self.assertIn("95%", summary)
        self.assertIn("5 verified facts", summary)
        self.assertIn("Live system data", summary)

    def test_llm_path_summary(self):
        summary = _build_summary("Verified memory records", "Memory Retrieval", 0.85, False, 0)
        self.assertIn("LLM generation", summary)
        self.assertIn("85%", summary)
        self.assertNotIn("verified facts", summary)


if __name__ == "__main__":
    unittest.main()
