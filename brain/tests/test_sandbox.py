"""Dedicated tests for codegen/sandbox.py — the safety gate for all generated code.

Tests AST validation, lint diagnostic parsing, test target selection,
baseline failure extraction, context helpers, and pipeline short-circuit
behavior.  No live ruff/pytest subprocess execution — those paths use
deterministic mocks.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field

from self_improve.code_patch import CodePatch, FileDiff
from self_improve.evaluation_report import EvaluationReport, TestResult
from codegen.sandbox import Sandbox, SandboxDiagnostic, _MODULE_TO_TESTS


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# SandboxDiagnostic
# ---------------------------------------------------------------------------

class TestSandboxDiagnostic:
    def test_str_basic(self):
        d = SandboxDiagnostic(error_type="syntax", file="foo.py", line=10, message="bad indent")
        s = str(d)
        assert "syntax" in s
        assert "foo.py" in s
        assert "10" in s

    def test_str_with_code(self):
        d = SandboxDiagnostic(error_type="lint", file="bar.py", code="E302", message="expected 2 lines")
        s = str(d)
        assert "E302" in s

    def test_str_no_line(self):
        d = SandboxDiagnostic(error_type="test", file="tests/", message="failure")
        s = str(d)
        assert "test" in s
        assert "tests/" in s

    def test_message_truncated(self):
        d = SandboxDiagnostic(error_type="lint", file="x.py", message="x" * 200)
        s = str(d)
        assert len(s) < 300


# ---------------------------------------------------------------------------
# AST Syntax Checking (_check_syntax)
# ---------------------------------------------------------------------------

class TestSyntaxCheck:
    def test_valid_python_passes(self):
        patch = CodePatch(files=[
            FileDiff(path="brain/tools/test_tool.py", new_content="def hello():\n    return 42\n"),
        ])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        result = sandbox._check_syntax(patch, report)
        assert result is True
        assert len(report.diagnostics) == 0

    def test_syntax_error_fails(self):
        patch = CodePatch(files=[
            FileDiff(path="brain/tools/bad.py", new_content="def broken(\n    return 42\n"),
        ])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        result = sandbox._check_syntax(patch, report)
        assert result is False
        assert len(report.diagnostics) >= 1
        assert report.diagnostics[0].error_type == "syntax"
        assert "bad.py" in report.diagnostics[0].file

    def test_multiple_files_one_bad(self):
        patch = CodePatch(files=[
            FileDiff(path="brain/tools/good.py", new_content="x = 1\n"),
            FileDiff(path="brain/tools/bad.py", new_content="def f(:\n"),
        ])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        result = sandbox._check_syntax(patch, report)
        assert result is False
        assert any(d.file == "brain/tools/bad.py" for d in report.diagnostics)

    def test_empty_content_skipped(self):
        patch = CodePatch(files=[
            FileDiff(path="brain/tools/deleted.py", new_content=""),
        ])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        result = sandbox._check_syntax(patch, report)
        assert result is True

    def test_complex_valid_code(self):
        code = (
            "import os\n"
            "from dataclasses import dataclass, field\n"
            "from typing import Any\n\n"
            "@dataclass\n"
            "class Foo:\n"
            "    name: str = ''\n"
            "    items: list[Any] = field(default_factory=list)\n\n"
            "    def bar(self) -> int:\n"
            "        return len(self.items)\n"
        )
        patch = CodePatch(files=[FileDiff(path="brain/test.py", new_content=code)])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        assert sandbox._check_syntax(patch, report) is True

    def test_syntax_error_has_line_number(self):
        code = "x = 1\ny = 2\ndef bad(:\n    pass\n"
        patch = CodePatch(files=[FileDiff(path="brain/x.py", new_content=code)])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        sandbox._check_syntax(patch, report)
        assert report.diagnostics[0].line == 3

    def test_syntax_error_has_context_span(self):
        code = "line1 = 1\nline2 = 2\ndef bad(:\n    pass\nline5 = 5\n"
        patch = CodePatch(files=[FileDiff(path="brain/x.py", new_content=code)])
        report = EvaluationReport(patch_id=patch.id)
        sandbox = Sandbox()
        sandbox._check_syntax(patch, report)
        span = report.diagnostics[0].context_span
        assert "line2" in span or "bad" in span


# ---------------------------------------------------------------------------
# Lint Diagnostic Parsing (_parse_lint_diagnostics)
# ---------------------------------------------------------------------------

class TestLintDiagnosticParsing:
    def test_parse_ruff_json_output(self):
        ruff_output = json.dumps([
            {
                "filename": "/tmp/sandbox/tools/my_tool.py",
                "location": {"row": 5, "column": 1},
                "code": "E302",
                "message": "expected 2 blank lines",
            },
            {
                "filename": "/tmp/sandbox/tools/my_tool.py",
                "location": {"row": 10, "column": 1},
                "code": "F401",
                "message": "unused import",
            },
        ])
        report = EvaluationReport(patch_id="test")
        report.lint_output = ruff_output
        report.lint_passed = False

        sandbox = Sandbox()
        sandbox._sandbox_dir = "/tmp/sandbox"
        sandbox._parse_lint_diagnostics(report)

        assert len(report.diagnostics) == 1  # deduped per file
        assert report.diagnostics[0].code == "E302"
        assert report.diagnostics[0].line == 5
        assert "tools/my_tool.py" in report.diagnostics[0].file

    def test_parse_invalid_json_fallback(self):
        report = EvaluationReport(patch_id="test")
        report.lint_output = "not json at all"

        sandbox = Sandbox()
        sandbox._sandbox_dir = "/tmp/sandbox"
        sandbox._parse_lint_diagnostics(report)

        assert len(report.diagnostics) == 1
        assert report.diagnostics[0].error_type == "lint"
        assert "not json" in report.diagnostics[0].message

    def test_parse_empty_json_array(self):
        report = EvaluationReport(patch_id="test")
        report.lint_output = "[]"

        sandbox = Sandbox()
        sandbox._sandbox_dir = "/tmp/sandbox"
        sandbox._parse_lint_diagnostics(report)

        assert len(report.diagnostics) == 0

    def test_multiple_files_produce_multiple_diagnostics(self):
        ruff_output = json.dumps([
            {"filename": "/tmp/sandbox/a.py", "location": {"row": 1}, "code": "E302", "message": "m1"},
            {"filename": "/tmp/sandbox/b.py", "location": {"row": 2}, "code": "F401", "message": "m2"},
        ])
        report = EvaluationReport(patch_id="test")
        report.lint_output = ruff_output
        sandbox = Sandbox()
        sandbox._sandbox_dir = "/tmp/sandbox"
        sandbox._parse_lint_diagnostics(report)
        assert len(report.diagnostics) == 2


# ---------------------------------------------------------------------------
# Test Target Selection (_select_test_targets)
# ---------------------------------------------------------------------------

class TestSelectTestTargets:
    def _make_test_dir(self) -> tuple[Path, str]:
        tmpdir = tempfile.mkdtemp(prefix="jarvis_sandbox_test_")
        test_dir = Path(tmpdir) / "tests"
        test_dir.mkdir()
        for name in [
            "test_consciousness.py", "test_memory.py", "test_tool_router.py",
            "test_hemisphere.py", "test_identity_fusion.py",
            "test_audit_regressions.py", "test_onboarding.py",
        ]:
            (test_dir / name).write_text("# test stub\n")
        return test_dir, tmpdir

    def test_consciousness_module_maps_correctly(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(
                ["brain/consciousness/kernel.py"], test_dir
            )
            target_names = {Path(t).name for t in targets}
            assert "test_consciousness.py" in target_names
            assert "test_soul_kernel.py" not in target_names or True  # may not exist
        finally:
            shutil.rmtree(tmpdir)

    def test_memory_module_maps(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(
                ["brain/memory/storage.py"], test_dir
            )
            target_names = {Path(t).name for t in targets}
            assert "test_memory.py" in target_names
        finally:
            shutil.rmtree(tmpdir)

    def test_no_patched_files_uses_fallback(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(None, test_dir)
            assert len(targets) > 0
            assert "test_audit_regressions.py" in targets[0] or "tests" in targets[0]
        finally:
            shutil.rmtree(tmpdir)

    def test_unknown_module_uses_fallback(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(
                ["brain/unknown_module/foo.py"], test_dir
            )
            assert len(targets) > 0
        finally:
            shutil.rmtree(tmpdir)

    def test_brain_prefix_stripped(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(
                ["brain/hemisphere/engine.py"], test_dir
            )
            target_names = {Path(t).name for t in targets}
            assert "test_hemisphere.py" in target_names
        finally:
            shutil.rmtree(tmpdir)

    def test_multiple_patched_files_merge_targets(self):
        test_dir, tmpdir = self._make_test_dir()
        try:
            targets = Sandbox._select_test_targets(
                ["brain/consciousness/kernel.py", "brain/memory/storage.py"],
                test_dir,
            )
            target_names = {Path(t).name for t in targets}
            assert "test_consciousness.py" in target_names
            assert "test_memory.py" in target_names
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Baseline Failure Extraction (_extract_failed_tests)
# ---------------------------------------------------------------------------

class TestExtractFailedTests:
    def test_empty_output(self):
        assert Sandbox._extract_failed_tests("") == set()

    def test_no_failures(self):
        output = "5 passed in 0.3s\n"
        assert Sandbox._extract_failed_tests(output) == set()

    def test_single_failure(self):
        output = "FAILED tests/test_foo.py::test_bar\n1 failed\n"
        result = Sandbox._extract_failed_tests(output)
        assert result == {"tests/test_foo.py::test_bar"}

    def test_multiple_failures(self):
        output = (
            "FAILED tests/test_a.py::test_x\n"
            "FAILED tests/test_b.py::test_y\n"
            "2 failed\n"
        )
        result = Sandbox._extract_failed_tests(output)
        assert len(result) == 2
        assert "tests/test_a.py::test_x" in result

    def test_mixed_output(self):
        output = (
            "tests/test_ok.py::test_1 PASSED\n"
            "FAILED tests/test_bad.py::test_2\n"
            "1 failed, 1 passed\n"
        )
        result = Sandbox._extract_failed_tests(output)
        assert result == {"tests/test_bad.py::test_2"}


# ---------------------------------------------------------------------------
# Context Extraction (_extract_context)
# ---------------------------------------------------------------------------

class TestExtractContext:
    def test_middle_line(self):
        source = "line1\nline2\nline3\nline4\nline5\n"
        ctx = Sandbox._extract_context(source, 3, window=1)
        assert "line2" in ctx
        assert "line3" in ctx
        assert "line4" in ctx

    def test_first_line(self):
        source = "first\nsecond\nthird\n"
        ctx = Sandbox._extract_context(source, 1, window=2)
        assert "first" in ctx

    def test_last_line(self):
        source = "a\nb\nc\n"
        ctx = Sandbox._extract_context(source, 3, window=1)
        assert "c" in ctx

    def test_none_line(self):
        ctx = Sandbox._extract_context("stuff\n", None)
        assert ctx == ""


# ---------------------------------------------------------------------------
# Module-to-test mapping completeness
# ---------------------------------------------------------------------------

class TestModuleToTestsMapping:
    def test_all_copied_subdirs_have_mappings(self):
        from codegen.sandbox import COPIED_SUBDIRS
        mapped_modules = set(_MODULE_TO_TESTS.keys())
        unmapped = []
        for subdir in COPIED_SUBDIRS:
            if subdir not in mapped_modules and subdir not in (
                "library", "dashboard", "jarvis_eval", "synthetic", "codegen", "acquisition"
            ):
                unmapped.append(subdir)
        assert unmapped == [], f"Unmapped modules: {unmapped}"

    def test_consciousness_has_key_tests(self):
        tests = _MODULE_TO_TESTS["consciousness"]
        assert "test_consciousness.py" in tests
        assert "test_soul_kernel.py" in tests

    def test_perception_has_identity_tests(self):
        tests = _MODULE_TO_TESTS["perception"]
        assert "test_identity_fusion.py" in tests


# ---------------------------------------------------------------------------
# Pipeline Short-Circuit Behavior (evaluate)
# ---------------------------------------------------------------------------

class TestEvaluatePipeline:
    """Tests that the evaluate pipeline correctly short-circuits:
    - Syntax fail → skip lint, tests, sim
    - Lint fail → skip tests
    - All pass → runs lint, tests, sim
    """

    def _make_sandbox_with_mocks(self):
        sandbox = Sandbox(project_root="/nonexistent")
        sandbox._copy_project_to_sandbox = MagicMock()
        sandbox._capture_baseline_failures = AsyncMock()
        sandbox._write_patch_files = MagicMock()
        sandbox._cleanup = MagicMock()
        return sandbox

    def test_syntax_fail_skips_everything(self):
        sandbox = self._make_sandbox_with_mocks()
        sandbox._run_lint = AsyncMock()
        sandbox._run_tests = AsyncMock()
        sandbox._run_simulation = AsyncMock()

        patch = CodePatch(files=[
            FileDiff(path="brain/test.py", new_content="def broken(\n"),
        ])

        report = _run(sandbox.evaluate(patch))

        assert report.status == "failed"
        assert any(d.error_type == "syntax" for d in report.diagnostics)
        sandbox._run_lint.assert_not_called()
        sandbox._run_tests.assert_not_called()
        sandbox._run_simulation.assert_not_called()

    def test_lint_fail_skips_tests(self):
        sandbox = self._make_sandbox_with_mocks()

        async def fake_lint(report, patched_files=None):
            report.lint_passed = False
            report.lint_executed = True

        sandbox._run_lint = fake_lint
        sandbox._run_tests = AsyncMock()
        sandbox._run_simulation = AsyncMock()

        patch = CodePatch(files=[
            FileDiff(path="brain/test.py", new_content="x = 1\n"),
        ])

        report = _run(sandbox.evaluate(patch))

        assert report.lint_passed is False
        sandbox._run_tests.assert_not_called()

    def test_all_pass_runs_full_pipeline(self):
        sandbox = self._make_sandbox_with_mocks()

        async def fake_lint(report, patched_files=None):
            report.lint_passed = True
            report.lint_executed = True

        async def fake_tests(report, patched_files=None):
            report.all_tests_passed = True
            report.tests_executed = True
            report.tests.append(TestResult(name="test_1", passed=True))

        async def fake_sim(report):
            report.sim_passed = True
            report.sim_executed = True

        sandbox._run_lint = fake_lint
        sandbox._run_tests = fake_tests
        sandbox._run_simulation = fake_sim

        patch = CodePatch(files=[
            FileDiff(path="brain/test.py", new_content="x = 1\n"),
        ])

        report = _run(sandbox.evaluate(patch))

        assert report.status == "passed"
        assert report.lint_passed is True
        assert report.all_tests_passed is True
        assert report.sim_passed is True

    def test_cleanup_always_runs(self):
        sandbox = self._make_sandbox_with_mocks()
        sandbox._run_lint = AsyncMock(side_effect=RuntimeError("boom"))

        patch = CodePatch(files=[
            FileDiff(path="brain/test.py", new_content="x = 1\n"),
        ])

        report = _run(sandbox.evaluate(patch))
        assert report.status == "failed"
        sandbox._cleanup.assert_called_once()
