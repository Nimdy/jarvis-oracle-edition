"""Tests for self_improve/provider.py — JSON parsing, edit application, schema validation.

Covers:
  - _extract_json: find JSON in LLM output with noise/markdown fences
  - _apply_edits: search-and-replace logic, missing search, empty edits
  - _parse_response: schema validation, required keys, file handling
  - Cloud gate: environment variable gating for cloud providers
  - get_status: shape and content
  - JSON retry prompt existence
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock

from self_improve.provider import (
    PatchProvider,
    MAX_PARSE_RETRIES,
    JSON_REPAIR_PROMPT,
    _self_improve_cloud_plugins_enabled,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        text = '{"files": [], "description": "test"}'
        result = PatchProvider._extract_json(text)
        assert result is not None
        assert json.loads(result)["description"] == "test"

    def test_json_with_leading_text(self):
        text = 'Here is the patch:\n{"files": [{"path": "brain/x.py"}], "description": "fix"}'
        result = PatchProvider._extract_json(text)
        assert result is not None
        assert "files" in json.loads(result)

    def test_json_with_markdown_fences(self):
        text = '```json\n{"files": [], "description": "test"}\n```'
        result = PatchProvider._extract_json(text)
        assert result is not None

    def test_no_json(self):
        result = PatchProvider._extract_json("no json here at all")
        assert result is None

    def test_nested_json(self):
        text = '{"files": [{"path": "x.py", "edits": [{"search": "{}", "replace": "{}"}]}]}'
        result = PatchProvider._extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed["files"]) == 1

    def test_empty_string(self):
        assert PatchProvider._extract_json("") is None

    def test_unclosed_brace(self):
        assert PatchProvider._extract_json('{"files": [') is None


# ---------------------------------------------------------------------------
# _apply_edits
# ---------------------------------------------------------------------------

class TestApplyEdits:
    def test_single_edit_success(self):
        original = "x = 1\ny = 2\n"
        edits = [{"search": "x = 1", "replace": "x = 42"}]
        result = PatchProvider._apply_edits(original, edits)
        assert result == "x = 42\ny = 2\n"

    def test_multiple_edits(self):
        original = "a = 1\nb = 2\nc = 3\n"
        edits = [
            {"search": "a = 1", "replace": "a = 10"},
            {"search": "c = 3", "replace": "c = 30"},
        ]
        result = PatchProvider._apply_edits(original, edits)
        assert "a = 10" in result
        assert "c = 30" in result
        assert "b = 2" in result

    def test_edit_search_not_found_returns_none(self):
        original = "x = 1\n"
        edits = [{"search": "y = 2", "replace": "y = 3"}]
        result = PatchProvider._apply_edits(original, edits)
        assert result is None

    def test_empty_search_skipped(self):
        original = "x = 1\n"
        edits = [{"search": "", "replace": "ignored"}]
        result = PatchProvider._apply_edits(original, edits)
        assert result == "x = 1\n"

    def test_empty_edits_list(self):
        original = "x = 1\n"
        result = PatchProvider._apply_edits(original, [])
        assert result == "x = 1\n"

    def test_replaces_only_first_occurrence(self):
        original = "x = 1\nx = 1\n"
        edits = [{"search": "x = 1", "replace": "x = 2"}]
        result = PatchProvider._apply_edits(original, edits)
        assert result == "x = 2\nx = 1\n"

    def test_multiline_search(self):
        original = "def foo():\n    return 1\n"
        edits = [{"search": "def foo():\n    return 1", "replace": "def foo():\n    return 42"}]
        result = PatchProvider._apply_edits(original, edits)
        assert "return 42" in result


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def _make_provider(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": ""}):
            return PatchProvider()

    def test_valid_response_with_edits(self):
        provider = self._make_provider()
        response = json.dumps({
            "files": [{"path": "brain/test.py", "edits": [
                {"search": "x = 1", "replace": "x = 2"}
            ]}],
            "description": "fix",
            "confidence": 0.8,
        })
        with patch.object(PatchProvider, "_read_original_file", return_value="x = 1\ny = 2\n"):
            patch_obj = provider._parse_response(response, "test", "plan_001")
        assert patch_obj is not None
        assert len(patch_obj.files) == 1
        assert patch_obj.files[0].path == "brain/test.py"
        assert "x = 2" in patch_obj.files[0].new_content

    def test_valid_response_with_content(self):
        provider = self._make_provider()
        response = json.dumps({
            "files": [{"path": "brain/test.py", "content": "x = 42\n"}],
            "description": "replace",
            "confidence": 0.9,
        })
        patch_obj = provider._parse_response(response, "test", "plan_002")
        assert patch_obj is not None
        assert patch_obj.files[0].new_content == "x = 42\n"

    def test_missing_required_key_returns_none(self):
        provider = self._make_provider()
        response = json.dumps({"description": "no files key"})
        patch_obj = provider._parse_response(response, "test", "plan_003")
        assert patch_obj is None

    def test_empty_files_returns_none(self):
        provider = self._make_provider()
        response = json.dumps({"files": []})
        patch_obj = provider._parse_response(response, "test", "plan_004")
        assert patch_obj is None

    def test_no_json_returns_none(self):
        provider = self._make_provider()
        patch_obj = provider._parse_response("just plain text", "test", "plan_005")
        assert patch_obj is None

    def test_file_without_path_skipped(self):
        provider = self._make_provider()
        response = json.dumps({
            "files": [
                {"path": "", "content": "x = 1"},
                {"path": "brain/good.py", "content": "y = 2"},
            ],
            "description": "test",
        })
        patch_obj = provider._parse_response(response, "test", "plan_006")
        assert patch_obj is not None
        assert len(patch_obj.files) == 1
        assert patch_obj.files[0].path == "brain/good.py"


# ---------------------------------------------------------------------------
# Cloud gate
# ---------------------------------------------------------------------------

class TestCloudGate:
    def test_cloud_disabled_by_default(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": ""}):
            assert _self_improve_cloud_plugins_enabled() is False

    def test_cloud_enabled_with_true(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": "true"}):
            assert _self_improve_cloud_plugins_enabled() is True

    def test_cloud_enabled_with_1(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": "1"}):
            assert _self_improve_cloud_plugins_enabled() is True

    def test_cloud_disabled_with_random_value(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": "maybe"}):
            assert _self_improve_cloud_plugins_enabled() is False

    def test_provider_no_claude_without_env(self):
        with patch.dict(os.environ, {
            "SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": "",
            "ANTHROPIC_API_KEY": "sk-test",
        }, clear=False):
            p = PatchProvider()
            assert p._claude_available is False

    def test_provider_claude_with_env(self):
        with patch.dict(os.environ, {
            "SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": "true",
            "ANTHROPIC_API_KEY": "sk-test",
        }, clear=False):
            p = PatchProvider()
            assert p._claude_available is True


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_shape(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": ""}):
            p = PatchProvider()
        status = p.get_status()
        assert "claude_available" in status
        assert "openai_available" in status
        assert "local_available" in status
        assert "coder" in status
        assert "cloud_plugins_enabled" in status

    def test_status_with_coder_server(self):
        with patch.dict(os.environ, {"SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS": ""}):
            p = PatchProvider()
        mock_coder = MagicMock()
        mock_coder.get_status.return_value = {"available": True, "running": False}
        p.set_coder_server(mock_coder)
        status = p.get_status()
        assert status["coder"]["available"] is True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestProviderConstants:
    def test_max_parse_retries(self):
        assert MAX_PARSE_RETRIES >= 1

    def test_json_repair_prompt_has_schema(self):
        assert "files" in JSON_REPAIR_PROMPT
        assert "edits" in JSON_REPAIR_PROMPT
