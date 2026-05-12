"""Regression tests for curiosity question answer retention and suppression."""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory


class _StubStorage:
    def __init__(self, memories: list[Memory]) -> None:
        self._memories = list(memories)

    def get_by_tag(self, tag: str) -> list[Memory]:
        return [m for m in self._memories if tag in m.tags]


class CuriosityQuestionRegressionTests(unittest.TestCase):
    def test_unknown_voice_legacy_answer_blocks_repeat_question(self) -> None:
        from personality.curiosity_questions import (
            check_unknown_speaker_curiosity,
            has_existing_answer,
        )

        mem = Memory(
            id="mem_curiosity_unknown_voice",
            timestamp=time.time(),
            weight=0.55,
            tags=("curiosity_answer", "curiosity_identity", "interactive", "outcome:engaged"),
            payload=(
                "Curiosity Q (identity): I heard someone speaking that I don't recognize "
                "- it wasn't you, David. Who was that? I'd like to know them.\n"
                "User answer: That was my wife Tanya."
            ),
            type="conversation",
            provenance="conversation",
        )
        fake_storage_module = types.ModuleType("memory.storage")
        fake_storage_module.memory_storage = _StubStorage([mem])

        with patch.dict(sys.modules, {"memory.storage": fake_storage_module}):
            self.assertTrue(has_existing_answer("identity", "unknown_voice"))
            question = check_unknown_speaker_curiosity(
                [{"timestamp": 1234.0, "had_known_user": True, "primary_user": "David"}],
                primary_user="David",
            )
            self.assertIsNone(question)

    def test_classify_curiosity_outcome_treats_repeat_correction_as_dismissed(self) -> None:
        from personality.curiosity_questions import classify_curiosity_outcome

        outcome = classify_curiosity_outcome(
            "Jarvis, I told you already that was my wife Tanya."
        )
        self.assertEqual(outcome, "dismissed")

    def test_infer_curiosity_topic_tags_marks_unknown_voice_answers(self) -> None:
        from personality.curiosity_questions import infer_curiosity_topic_tags

        tags = infer_curiosity_topic_tags(
            "identity",
            question=(
                "I heard someone speaking that I don't recognize - it wasn't you, "
                "David. Who was that? I'd like to know them."
            ),
            evidence="unknown_voice_event at 123, companion=David",
        )
        self.assertIn("curiosity_topic:unknown_voice", tags)


if __name__ == "__main__":
    unittest.main()
