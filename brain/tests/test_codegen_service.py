"""Tests for codegen/service.py — evidence sufficiency gate and status.

CodeGenService is the shared API consumed by both self-improvement and the
capability acquisition pipeline. The evidence sufficiency gate is the first
defense: it prevents code generation from running when evidence is too weak
for the given risk tier.

Covers:
  - _check_evidence_sufficiency() — tier 0 always passes, tier 1+ needs evidence
  - get_status() shape
  - coder_available property
  - generate() delegation (mocked CoderServer)
  - set_coder_server() wiring
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, AsyncMock
import asyncio

from codegen.service import CodeGenService


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Evidence sufficiency gate
# ---------------------------------------------------------------------------

class TestEvidenceSufficiency:
    def test_tier_0_always_passes(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(0, [])
        assert result["sufficient"] is True

    def test_tier_0_passes_even_with_no_evidence(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(0, [])
        assert result["sufficient"] is True
        assert result["tier"] == 0

    def test_tier_1_no_evidence_fails(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(1, [])
        assert result["sufficient"] is False
        assert "at least 1" in result["reason"]

    def test_tier_1_with_evidence_passes(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(1, ["art_abc123"])
        assert result["sufficient"] is True
        assert result["evidence_count"] == 1

    def test_tier_2_no_evidence_fails(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(2, [])
        assert result["sufficient"] is False

    def test_tier_2_with_evidence_passes(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(2, ["a", "b"])
        assert result["sufficient"] is True
        assert result["evidence_count"] == 2

    def test_tier_3_no_evidence_fails(self):
        svc = CodeGenService()
        result = svc._check_evidence_sufficiency(3, [])
        assert result["sufficient"] is False


# ---------------------------------------------------------------------------
# coder_available property
# ---------------------------------------------------------------------------

class TestCoderAvailable:
    def test_no_server_not_available(self):
        svc = CodeGenService()
        assert svc.coder_available is False

    def test_server_available(self):
        svc = CodeGenService()
        mock_server = MagicMock()
        mock_server.is_available.return_value = True
        svc.set_coder_server(mock_server)
        assert svc.coder_available is True

    def test_server_not_available(self):
        svc = CodeGenService()
        mock_server = MagicMock()
        mock_server.is_available.return_value = False
        svc.set_coder_server(mock_server)
        assert svc.coder_available is False


# ---------------------------------------------------------------------------
# generate() delegation
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_no_server_returns_none(self):
        svc = CodeGenService()
        result = _run(svc.generate([{"role": "user", "content": "test"}]))
        assert result is None

    def test_server_not_available_returns_none(self):
        svc = CodeGenService()
        mock_server = MagicMock()
        mock_server.is_available.return_value = False
        svc.set_coder_server(mock_server)
        result = _run(svc.generate([{"role": "user", "content": "test"}]))
        assert result is None

    def test_successful_generation_increments_count(self):
        svc = CodeGenService()
        mock_server = MagicMock()
        mock_server.is_available.return_value = True
        mock_server.generate = AsyncMock(return_value='{"files": []}')
        svc.set_coder_server(mock_server)

        result = _run(svc.generate([{"role": "user", "content": "test"}]))
        assert result == '{"files": []}'
        assert svc._total_generations == 1


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_shape_no_server(self):
        svc = CodeGenService()
        status = svc.get_status()
        assert "coder" in status
        assert "total_generations" in status
        assert "total_validations" in status
        assert "total_failures" in status
        assert status["coder"]["available"] is False

    def test_status_with_server(self):
        svc = CodeGenService()
        mock_server = MagicMock()
        mock_server.get_status.return_value = {"available": True, "port": 8081}
        svc.set_coder_server(mock_server)
        status = svc.get_status()
        assert status["coder"]["available"] is True

    def test_counters_initialized_zero(self):
        svc = CodeGenService()
        status = svc.get_status()
        assert status["total_generations"] == 0
        assert status["total_validations"] == 0
        assert status["total_failures"] == 0


# ---------------------------------------------------------------------------
# _parse_raw_output — static JSON extraction (Fix 1)
# ---------------------------------------------------------------------------

class TestParseRawOutput:
    """Validate the static parser that replaced the broken _parse_coder_response."""

    def test_valid_json_with_files(self):
        raw = '{"files": [{"path": "test.py", "content": "print(1)"}], "description": "test patch"}'
        patch = CodeGenService._parse_raw_output(raw, plan_id="plan_123")
        assert patch is not None
        assert len(patch.files) == 1
        assert patch.files[0].path == "test.py"
        assert patch.files[0].new_content == "print(1)"
        assert patch.plan_id == "plan_123"
        assert patch.provider == "codegen_service"
        assert patch.description == "test patch"

    def test_json_embedded_in_text(self):
        raw = 'Here is the code:\n```json\n{"files": [{"path": "a.py", "content": "x=1"}]}\n```'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is not None
        assert len(patch.files) == 1

    def test_no_json_returns_none(self):
        raw = "This is just plain text with no JSON"
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is None

    def test_empty_files_returns_none(self):
        raw = '{"files": []}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is None

    def test_missing_files_key_returns_none(self):
        raw = '{"description": "no files key"}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is None

    def test_file_without_path_skipped(self):
        raw = '{"files": [{"content": "x=1"}, {"path": "b.py", "content": "y=2"}]}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is not None
        assert len(patch.files) == 1
        assert patch.files[0].path == "b.py"

    def test_file_with_edits_and_no_source_skipped(self):
        raw = '{"files": [{"path": "nonexistent.py", "edits": [{"search": "x", "replace": "y"}]}]}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is None

    def test_confidence_extracted(self):
        raw = '{"files": [{"path": "a.py", "content": "x=1"}], "confidence": 0.9}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is not None
        assert patch.confidence == 0.9

    def test_default_confidence(self):
        raw = '{"files": [{"path": "a.py", "content": "x=1"}]}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is not None
        assert patch.confidence == 0.5

    def test_invalid_json_returns_none(self):
        raw = '{"files": [{"path": "a.py", "content": INVALID}]}'
        patch = CodeGenService._parse_raw_output(raw)
        assert patch is None

    def test_extract_json_nested_braces(self):
        """_extract_json handles nested braces correctly."""
        text = 'prefix {"key": {"nested": true}} suffix'
        result = CodeGenService._extract_json(text)
        assert result == '{"key": {"nested": true}}'

    def test_extract_json_no_object(self):
        result = CodeGenService._extract_json("no json here")
        assert result is None
