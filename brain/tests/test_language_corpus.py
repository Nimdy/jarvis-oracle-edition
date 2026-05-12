import json
import tempfile
import unittest
from pathlib import Path

from reasoning.language_corpus import LanguageCorpusStore


class LanguageCorpusStoreTests(unittest.TestCase):
    def test_append_example_writes_expected_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            store = LanguageCorpusStore(path)

            example_id = store.append_example(
                conversation_id="conv-123",
                query="What did you learn last?",
                route="introspection",
                response_class="recent_learning",
                meaning_frame={"lead": "The latest verified learning record is below."},
                grounding_payload={"kind": "source", "title": "World models"},
                teacher_answer="I most recently studied World models.",
                final_answer="I most recently studied World models.",
                provenance_verdict="strict_provenance_grounded",
                user_feedback="positive",
                confidence=0.98,
                safety_flags=["fail_closed_when_missing"],
            )

            self.assertTrue(example_id.startswith("lang_"))
            self.assertTrue(path.exists())

            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

            payload = json.loads(lines[0])
            self.assertEqual(payload["conversation_id"], "conv-123")
            self.assertEqual(payload["route"], "introspection")
            self.assertEqual(payload["response_class"], "recent_learning")
            self.assertEqual(payload["meaning_frame"]["lead"], "The latest verified learning record is below.")
            self.assertEqual(payload["grounding_payload"]["title"], "World models")
            self.assertEqual(payload["provenance_verdict"], "strict_provenance_grounded")
            self.assertEqual(payload["user_feedback"], "positive")
            self.assertEqual(payload["safety_flags"], ["fail_closed_when_missing"])
            self.assertEqual(payload["schema_version"], 1)

    def test_confidence_is_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            store = LanguageCorpusStore(path)

            store.append_example(
                conversation_id="conv-456",
                query="Status?",
                route="status",
                response_class="self_status",
                meaning_frame={"lead": "Here is my current measured status."},
                grounding_payload="ok",
                teacher_answer="",
                final_answer="Status is healthy.",
                provenance_verdict="grounded_tool_data",
                confidence=3.0,
            )

            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["confidence"], 1.0)

    def test_get_stats_reports_counts_and_recent_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            store = LanguageCorpusStore(path)

            store.append_example(
                conversation_id="conv-a",
                query="Who is this?",
                route="identity",
                response_class="identity_answer",
                meaning_frame={"lead": "The current speaker is identified as David."},
                grounding_payload={"kind": "current_voice", "name": "David"},
                teacher_answer="",
                final_answer="The current speaker is identified as David.",
                provenance_verdict="grounded_identity_status",
                user_feedback="positive",
                confidence=0.97,
                safety_flags=["local_biometrics_only"],
            )
            store.append_example(
                conversation_id="conv-b",
                query="Explain your architecture.",
                route="codebase",
                response_class="system_explanation",
                meaning_frame={"lead": "System explanation:"},
                grounding_payload={"title": "System explanation", "body": "Memory routes through storage."},
                teacher_answer="",
                final_answer="System explanation: Memory routes through storage.",
                provenance_verdict="grounded_codebase_answer",
                confidence=0.9,
                safety_flags=["grounded_codebase_answer"],
            )

            stats = store.get_stats()
            self.assertEqual(stats["total_examples"], 2)
            self.assertEqual(stats["counts_by_route"]["identity"], 1)
            self.assertEqual(stats["counts_by_route"]["codebase"], 1)
            self.assertEqual(stats["counts_by_response_class"]["system_explanation"], 1)
            self.assertEqual(stats["counts_by_route_class"]["identity|identity_answer"], 1)
            self.assertEqual(stats["counts_by_route_class"]["codebase|system_explanation"], 1)
            self.assertEqual(stats["counts_by_provenance"]["grounded_identity_status"], 1)
            self.assertEqual(stats["counts_by_feedback"]["positive"], 1)
            self.assertEqual(stats["counts_by_safety_flag"]["local_biometrics_only"], 1)
            self.assertEqual(stats["recent_example_count"], 2)
            self.assertEqual(stats["recent_examples"][-1]["response_class"], "system_explanation")

    def test_rehydrate_includes_rotated_examples_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            rotated = path.with_suffix(path.suffix + ".1")
            rotated.write_text(
                json.dumps({
                    "example_id": "lang_old",
                    "conversation_id": "conv-old",
                    "query": "Old query",
                    "route": "introspection",
                    "response_class": "recent_learning",
                    "meaning_frame": {"lead": "Older lead"},
                    "grounding_payload": {},
                    "teacher_answer": "",
                    "final_answer": "",
                    "provenance_verdict": "strict_provenance_grounded",
                    "user_feedback": "",
                    "confidence": 1.0,
                    "timestamp": 1.0,
                    "safety_flags": [],
                    "schema_version": 1,
                }) + "\n",
                encoding="utf-8",
            )
            path.write_text(
                json.dumps({
                    "example_id": "lang_new",
                    "conversation_id": "conv-new",
                    "query": "New query",
                    "route": "status",
                    "response_class": "self_status",
                    "meaning_frame": {"lead": "Newer lead"},
                    "grounding_payload": {},
                    "teacher_answer": "",
                    "final_answer": "",
                    "provenance_verdict": "grounded_tool_data",
                    "user_feedback": "",
                    "confidence": 1.0,
                    "timestamp": 2.0,
                    "safety_flags": [],
                    "schema_version": 1,
                }) + "\n",
                encoding="utf-8",
            )

            store = LanguageCorpusStore(path)
            stats = store.get_stats()

            self.assertEqual(stats["total_examples"], 2)
            self.assertEqual(stats["retained_file_count"], 2)
            self.assertEqual(stats["counts_by_response_class"]["recent_learning"], 1)
            self.assertEqual(stats["counts_by_response_class"]["self_status"], 1)


if __name__ == "__main__":
    unittest.main()
