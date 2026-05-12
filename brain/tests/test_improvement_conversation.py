"""Tests for self_improve/conversation.py — multi-turn persistence and state machine.

Covers:
  - ConversationTurn serialization and content truncation
  - ImprovementConversation state transitions, turn accumulation, message building
  - JSONL round-trip: save → load from disk → verify integrity
  - Edge cases: empty conversations, resume after reload, metadata handling
  - System prompt existence and basic invariants
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import patch

from self_improve.conversation import (
    ConversationTurn,
    ImprovementConversation,
    _save_conversation,
    _load_recent_conversations,
    THINKER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    CONVERSATIONS_DIR,
    MAX_ITERATIONS,
)


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------

class TestConversationTurn:
    def test_basic_creation(self):
        turn = ConversationTurn(role="think", content="analyze the code")
        assert turn.role == "think"
        assert turn.content == "analyze the code"
        assert turn.timestamp > 0
        assert turn.metadata == {}

    def test_to_dict_roundtrip(self):
        turn = ConversationTurn(
            role="code", content="def fix(): pass",
            timestamp=1000.0, metadata={"attempt": 1},
        )
        d = turn.to_dict()
        assert d["role"] == "code"
        assert d["content"] == "def fix(): pass"
        assert d["timestamp"] == 1000.0
        assert d["metadata"] == {"attempt": 1}

    def test_content_truncated_at_2000(self):
        long_content = "x" * 3000
        turn = ConversationTurn(role="think", content=long_content)
        d = turn.to_dict()
        assert len(d["content"]) == 2000

    def test_all_roles_accepted(self):
        for role in ("think", "code", "validate", "review", "system"):
            turn = ConversationTurn(role=role, content="test")
            assert turn.role == role

    def test_serializable_as_json(self):
        turn = ConversationTurn(
            role="validate", content="all tests pass",
            metadata={"errors": [], "passed": True},
        )
        serialized = json.dumps(turn.to_dict())
        restored = json.loads(serialized)
        assert restored["role"] == "validate"
        assert restored["metadata"]["passed"] is True


# ---------------------------------------------------------------------------
# ImprovementConversation — state + turn management
# ---------------------------------------------------------------------------

class TestImprovementConversation:
    def test_creation_defaults(self):
        conv = ImprovementConversation(
            id="conv_001", request_description="fix memory leak",
        )
        assert conv.id == "conv_001"
        assert conv.status == "started"
        assert conv.iteration == 0
        assert conv.turns == []
        assert conv.target_files == []
        assert conv.started_at > 0
        assert conv.completed_at == 0.0

    def test_add_turn(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("think", "analyzing the issue", attempt=1)
        assert len(conv.turns) == 1
        assert conv.turns[0].role == "think"
        assert conv.turns[0].content == "analyzing the issue"
        assert conv.turns[0].metadata["attempt"] == 1

    def test_turn_accumulation(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("system", "you are a code generator")
        conv.add_turn("think", "plan: change line 42")
        conv.add_turn("code", "def fix(): return True")
        conv.add_turn("validate", "syntax ok, tests pass")
        assert len(conv.turns) == 4
        assert [t.role for t in conv.turns] == ["system", "think", "code", "validate"]

    def test_status_transitions(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        assert conv.status == "started"

        conv.status = "thinking"
        assert conv.status == "thinking"

        conv.status = "coding"
        assert conv.status == "coding"

        conv.status = "validating"
        assert conv.status == "validating"

        conv.status = "reviewing"
        assert conv.status == "reviewing"

        conv.status = "completed"
        conv.completed_at = time.time()
        assert conv.status == "completed"
        assert conv.completed_at > 0

    def test_failure_status(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.status = "failed"
        assert conv.status == "failed"

    def test_iteration_tracking(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        assert conv.iteration == 0
        conv.iteration += 1
        assert conv.iteration == 1
        conv.iteration += 1
        assert conv.iteration == 2

    def test_to_dict(self):
        conv = ImprovementConversation(
            id="conv_002", request_description="optimize kernel tick",
            target_files=["brain/consciousness/kernel.py"],
        )
        conv.add_turn("think", "analysis")
        conv.add_turn("code", "patch")
        conv.iteration = 1
        conv.status = "completed"

        d = conv.to_dict()
        assert d["id"] == "conv_002"
        assert d["request"] == "optimize kernel tick"
        assert d["target_files"] == ["brain/consciousness/kernel.py"]
        assert d["turn_count"] == 2
        assert d["iteration"] == 1
        assert d["status"] == "completed"

    def test_to_dict_truncates_request(self):
        conv = ImprovementConversation(
            id="c1", request_description="x" * 300,
        )
        d = conv.to_dict()
        assert len(d["request"]) == 200


# ---------------------------------------------------------------------------
# get_messages_for_coder — Ollama chat message building
# ---------------------------------------------------------------------------

class TestGetMessagesForCoder:
    def test_empty_turns(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        assert conv.get_messages_for_coder() == []

    def test_think_becomes_user(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("think", "please fix line 42")
        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "please fix line 42"

    def test_code_becomes_assistant(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("code", "def fix(): pass")
        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_validate_becomes_user(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("validate", "tests failed: ...")
        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_system_becomes_system(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("system", "you are a code generator")
        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_full_think_code_validate_cycle(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("system", "you are a coder")
        conv.add_turn("think", "plan: change X")
        conv.add_turn("code", '{"files": [...]}')
        conv.add_turn("validate", "syntax error on line 5")
        conv.add_turn("think", "revised plan: change Y")
        conv.add_turn("code", '{"files": [...]}')

        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 6
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user", "user", "assistant"
        ]

    def test_review_role_skipped(self):
        conv = ImprovementConversation(id="c1", request_description="fix")
        conv.add_turn("review", "approved")
        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 0


# ---------------------------------------------------------------------------
# JSONL persistence — save + load round-trip
# ---------------------------------------------------------------------------

class TestJSONLPersistence:
    def _make_tmpdir(self):
        return tempfile.mkdtemp(prefix="jarvis_conv_test_")

    def test_save_creates_file(self):
        tmpdir = self._make_tmpdir()
        try:
            conv = ImprovementConversation(id="test_save", request_description="test")
            conv.add_turn("think", "hello")
            conv.add_turn("code", "world")

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "test_save.jsonl"
            assert saved.exists()
            lines = saved.read_text().strip().splitlines()
            assert len(lines) == 2
        finally:
            shutil.rmtree(tmpdir)

    def test_save_content_is_valid_jsonl(self):
        tmpdir = self._make_tmpdir()
        try:
            conv = ImprovementConversation(id="test_jsonl", request_description="test")
            conv.add_turn("system", "prompt")
            conv.add_turn("think", "analysis")
            conv.add_turn("code", '{"files": []}')

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "test_jsonl.jsonl"
            lines = saved.read_text().strip().splitlines()
            for line in lines:
                parsed = json.loads(line)
                assert "role" in parsed
                assert "content" in parsed
                assert "timestamp" in parsed
        finally:
            shutil.rmtree(tmpdir)

    def test_save_preserves_turn_order(self):
        tmpdir = self._make_tmpdir()
        try:
            conv = ImprovementConversation(id="test_order", request_description="test")
            roles = ["system", "think", "code", "validate", "think", "code"]
            for role in roles:
                conv.add_turn(role, f"content for {role}")

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "test_order.jsonl"
            lines = saved.read_text().strip().splitlines()
            restored_roles = [json.loads(line)["role"] for line in lines]
            assert restored_roles == roles
        finally:
            shutil.rmtree(tmpdir)

    def test_load_recent_empty_dir(self):
        tmpdir = self._make_tmpdir()
        try:
            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                result = _load_recent_conversations()
            assert result == []
        finally:
            shutil.rmtree(tmpdir)

    def test_load_recent_returns_summaries(self):
        tmpdir = self._make_tmpdir()
        try:
            for cid in ("conv_a", "conv_b"):
                conv = ImprovementConversation(id=cid, request_description="test")
                conv.add_turn("think", f"analysis for {cid}")
                conv.add_turn("code", f"patch for {cid}")
                with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                    _save_conversation(conv)
                time.sleep(0.01)

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                summaries = _load_recent_conversations(limit=5)

            assert len(summaries) == 2
            for s in summaries:
                assert "id" in s
                assert "turns" in s
                assert s["turns"] == 2
                assert s["first_role"] == "think"
                assert s["last_role"] == "code"
        finally:
            shutil.rmtree(tmpdir)

    def test_load_recent_respects_limit(self):
        tmpdir = self._make_tmpdir()
        try:
            for i in range(5):
                conv = ImprovementConversation(id=f"conv_{i}", request_description="test")
                conv.add_turn("think", "x")
                with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                    _save_conversation(conv)
                time.sleep(0.01)

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                summaries = _load_recent_conversations(limit=2)

            assert len(summaries) == 2
        finally:
            shutil.rmtree(tmpdir)

    def test_load_recent_nonexistent_dir(self):
        nonexistent = Path("/tmp/jarvis_test_conv_nonexistent_dir_xyz")
        if nonexistent.exists():
            shutil.rmtree(nonexistent)
        with patch("self_improve.conversation.CONVERSATIONS_DIR", nonexistent):
            result = _load_recent_conversations()
        assert result == []

    def test_save_overwrites_on_resave(self):
        """Re-saving appends no duplicate — it overwrites the file."""
        tmpdir = self._make_tmpdir()
        try:
            conv = ImprovementConversation(id="test_resave", request_description="test")
            conv.add_turn("think", "first")

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            conv.add_turn("code", "second")
            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "test_resave.jsonl"
            lines = saved.read_text().strip().splitlines()
            assert len(lines) == 2  # not 3 (overwrite, not append)
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# State machine simulation — think/code/validate loop
# ---------------------------------------------------------------------------

class TestStateMachineSimulation:
    """Simulates the orchestrator's usage pattern of ImprovementConversation."""

    def test_happy_path_single_iteration(self):
        conv = ImprovementConversation(
            id="sim_001", request_description="fix memory leak in storage.py",
            target_files=["brain/memory/storage.py"],
        )

        # Phase: thinking
        conv.status = "thinking"
        conv.add_turn("system", THINKER_SYSTEM_PROMPT[:100])
        conv.add_turn("think", "Plan: add gc.collect() after bulk eviction")

        # Phase: coding
        conv.status = "coding"
        conv.add_turn("code", '{"files": [{"path": "brain/memory/storage.py", "edits": []}]}')
        conv.iteration += 1

        # Phase: validating
        conv.status = "validating"
        conv.add_turn("validate", "AST ok, lint ok, 45 tests passed")

        # Phase: reviewing
        conv.status = "reviewing"
        conv.add_turn("review", '{"approved": true, "reasoning": "clean fix"}')

        # Complete
        conv.status = "completed"
        conv.completed_at = time.time()

        assert conv.status == "completed"
        assert conv.iteration == 1
        assert len(conv.turns) == 5
        assert conv.completed_at > conv.started_at

    def test_failed_validation_retry_loop(self):
        conv = ImprovementConversation(
            id="sim_002", request_description="optimize kernel tick",
        )

        for attempt in range(MAX_ITERATIONS):
            conv.status = "thinking"
            conv.add_turn("think", f"attempt {attempt + 1}: plan adjustment")

            conv.status = "coding"
            conv.add_turn("code", f'{{"files": [], "attempt": {attempt + 1}}}')
            conv.iteration += 1

            conv.status = "validating"
            if attempt < MAX_ITERATIONS - 1:
                conv.add_turn("validate", f"FAILED: syntax error on attempt {attempt + 1}")
            else:
                conv.add_turn("validate", "all tests passed")

        conv.status = "completed"

        assert conv.iteration == MAX_ITERATIONS
        assert conv.status == "completed"
        assert len(conv.turns) == MAX_ITERATIONS * 3

    def test_exhausted_iterations_fails(self):
        conv = ImprovementConversation(
            id="sim_003", request_description="fix broken import",
        )

        for attempt in range(MAX_ITERATIONS):
            conv.add_turn("think", f"attempt {attempt + 1}")
            conv.add_turn("code", "bad code")
            conv.add_turn("validate", "FAILED")
            conv.iteration += 1

        conv.status = "failed"
        assert conv.status == "failed"
        assert conv.iteration == MAX_ITERATIONS

    def test_messages_after_retry_include_full_history(self):
        """Coder should see all prior attempts when generating the next fix."""
        conv = ImprovementConversation(id="c1", request_description="fix")

        conv.add_turn("system", "you are a coder")
        conv.add_turn("think", "plan v1")
        conv.add_turn("code", "patch v1")
        conv.add_turn("validate", "FAILED: syntax error")
        conv.add_turn("think", "revised plan v2")

        msgs = conv.get_messages_for_coder()
        assert len(msgs) == 5
        assert msgs[-1]["role"] == "user"
        assert "revised plan" in msgs[-1]["content"]


# ---------------------------------------------------------------------------
# System prompts — existence and basic invariants
# ---------------------------------------------------------------------------

class TestSystemPrompts:
    def test_thinker_prompt_exists(self):
        assert len(THINKER_SYSTEM_PROMPT) > 100
        assert "analyze" in THINKER_SYSTEM_PROMPT.lower()

    def test_planner_prompt_exists(self):
        assert len(PLANNER_SYSTEM_PROMPT) > 100
        assert "technical" in PLANNER_SYSTEM_PROMPT.lower()

    def test_coder_prompt_exists(self):
        assert len(CODER_SYSTEM_PROMPT) > 100
        assert "json" in CODER_SYSTEM_PROMPT.lower()

    def test_reviewer_prompt_exists(self):
        assert len(REVIEWER_SYSTEM_PROMPT) > 50
        assert "review" in REVIEWER_SYSTEM_PROMPT.lower()

    def test_coder_prompt_forbids_dangerous_imports(self):
        for forbidden in ("subprocess", "eval", "exec"):
            assert forbidden in CODER_SYSTEM_PROMPT

    def test_max_iterations_is_reasonable(self):
        assert 1 <= MAX_ITERATIONS <= 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_conversation_to_dict(self):
        conv = ImprovementConversation(id="empty", request_description="nothing")
        d = conv.to_dict()
        assert d["turn_count"] == 0
        assert d["status"] == "started"

    def test_metadata_preserved_through_serialization(self):
        tmpdir = tempfile.mkdtemp(prefix="jarvis_conv_edge_")
        try:
            conv = ImprovementConversation(id="meta", request_description="test")
            conv.add_turn("think", "analysis", error_count=3, files=["a.py", "b.py"])

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "meta.jsonl"
            parsed = json.loads(saved.read_text().strip())
            assert parsed["metadata"]["error_count"] == 3
            assert parsed["metadata"]["files"] == ["a.py", "b.py"]
        finally:
            shutil.rmtree(tmpdir)

    def test_unicode_content_handled(self):
        tmpdir = tempfile.mkdtemp(prefix="jarvis_conv_unicode_")
        try:
            conv = ImprovementConversation(id="unicode", request_description="test")
            conv.add_turn("think", "analyze 日本語 content with émojis 🎉")

            with patch("self_improve.conversation.CONVERSATIONS_DIR", Path(tmpdir)):
                _save_conversation(conv)

            saved = Path(tmpdir) / "unicode.jsonl"
            parsed = json.loads(saved.read_text(encoding="utf-8").strip())
            assert "日本語" in parsed["content"]
        finally:
            shutil.rmtree(tmpdir)

    def test_save_handles_os_error_gracefully(self):
        conv = ImprovementConversation(id="fail", request_description="test")
        conv.add_turn("think", "x")
        with patch("self_improve.conversation.CONVERSATIONS_DIR", Path("/nonexistent/path/xyz")):
            _save_conversation(conv)  # should not raise
