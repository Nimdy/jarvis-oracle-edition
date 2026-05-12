"""Tests for the self-improvement pipeline.

Covers: sandbox evaluation, patch plan validation, denied pattern detection,
capability escalation, provider JSON parsing, targeted test selection,
and end-to-end dry-run flow (with mocked Ollama).

Run: python -m pytest tests/test_self_improve.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from self_improve.code_patch import CodePatch, FileDiff
from self_improve.evaluation_report import EvaluationReport
from self_improve.improvement_request import ImprovementRequest
from self_improve.patch_plan import (
    ALLOWED_PATHS,
    DENIED_PATTERNS,
    PatchPlan,
    check_ast_forbidden_calls,
    check_denied_patterns,
    detect_capability_escalation,
)
from self_improve.sandbox import Sandbox, COPIED_SUBDIRS, _MODULE_TO_TESTS


BRAIN_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Sandbox parity
# ---------------------------------------------------------------------------


class TestSandboxParity:
    """Verify sandbox copies all importable modules."""

    def test_copied_subdirs_covers_allowed_paths(self):
        """Every directory in ALLOWED_PATHS must be covered by COPIED_SUBDIRS.

        A nested path like ``brain/tools/plugins/`` is covered when its root
        package (``tools``) is in COPIED_SUBDIRS, because package copy is
        recursive.
        """
        for allowed in ALLOWED_PATHS:
            module = allowed.replace("brain/", "").rstrip("/")
            root = module.split("/", 1)[0]
            assert root in COPIED_SUBDIRS, (
                f"ALLOWED_PATHS includes 'brain/{module}/' but COPIED_SUBDIRS is missing root package '{root}'"
            )

    def test_copied_subdirs_covers_real_packages(self):
        """Every actual Python package directory in brain/ should be in COPIED_SUBDIRS."""
        for entry in BRAIN_DIR.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith((".", "_", "tests", "scripts", "config")):
                continue
            init = entry / "__init__.py"
            has_py = any(entry.glob("*.py"))
            if init.exists() or has_py:
                assert entry.name in COPIED_SUBDIRS, (
                    f"Package '{entry.name}' exists on disk but is missing from COPIED_SUBDIRS"
                )


# ---------------------------------------------------------------------------
# Patch Plan validation
# ---------------------------------------------------------------------------


class TestPatchPlanScope:
    """Test scope validation for patch plans."""

    def test_allowed_path_passes(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/engine.py"])
        assert plan.validate_scope() == []

    def test_denied_path_fails(self):
        plan = PatchPlan(files_to_modify=["brain/main.py"])
        violations = plan.validate_scope()
        assert len(violations) == 1
        assert "outside allowed scope" in violations[0]

    def test_multiple_files_mixed(self):
        plan = PatchPlan(
            files_to_modify=["brain/memory/core.py", "brain/main.py"],
        )
        violations = plan.validate_scope()
        assert len(violations) == 1

    def test_new_file_in_allowed_path(self):
        plan = PatchPlan(files_to_create=["brain/tools/new_helper.py"])
        assert plan.validate_scope() == []

    def test_dangerous_file_detection(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/mutation_governor.py"])
        assert plan.check_dangerous() is True

    def test_safe_file_not_dangerous(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/tone.py"])
        assert plan.check_dangerous() is False

    def test_diff_budget_violation(self):
        plan = PatchPlan(
            files_to_modify=["a.py", "b.py", "c.py", "d.py"],
        )
        violations = plan.validate_diff_budget()
        assert any("Too many files" in v for v in violations)


# ---------------------------------------------------------------------------
# Denied patterns
# ---------------------------------------------------------------------------


class TestDeniedPatterns:
    """Regex-based content scanning."""

    def test_subprocess_denied(self):
        violations = check_denied_patterns("import subprocess\nsubprocess.run(['ls'])")
        assert any("subprocess" in v for v in violations)

    def test_eval_denied(self):
        violations = check_denied_patterns("result = eval(user_input)")
        assert any("eval" in v for v in violations)

    def test_safe_code_passes(self):
        violations = check_denied_patterns(
            "import logging\nlogger = logging.getLogger(__name__)\nlogger.info('ok')"
        )
        assert violations == []

    def test_os_system_denied(self):
        violations = check_denied_patterns("os.system('rm -rf /')")
        assert any("os.system" in v for v in violations)

    def test_password_denied(self):
        violations = check_denied_patterns("password = 'hunter2'")
        assert any("password" in v for v in violations)


# ---------------------------------------------------------------------------
# AST forbidden calls
# ---------------------------------------------------------------------------


class TestASTForbiddenCalls:
    """AST-level function call detection."""

    def test_subprocess_run_forbidden(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        violations = check_ast_forbidden_calls(code)
        assert any("subprocess.run" in v for v in violations)

    def test_bare_eval_forbidden(self):
        code = "x = eval('1+1')"
        violations = check_ast_forbidden_calls(code)
        assert any("eval" in v for v in violations)

    def test_safe_code_no_violations(self):
        code = "import json\ndata = json.loads('{}')"
        violations = check_ast_forbidden_calls(code)
        assert violations == []

    def test_syntax_error_returns_empty(self):
        violations = check_ast_forbidden_calls("def broken(")
        assert violations == []


# ---------------------------------------------------------------------------
# Capability escalation
# ---------------------------------------------------------------------------


class TestCapabilityEscalation:
    """Detect new network/subprocess/security boundary introductions."""

    def test_new_network_import(self):
        original = "import json"
        new = "import json\nimport requests"
        escalations = detect_capability_escalation(original, new)
        assert any("network" in e.lower() for e in escalations)

    def test_no_escalation_when_import_exists(self):
        original = "import requests\nx = 1"
        new = "import requests\nx = 2"
        escalations = detect_capability_escalation(original, new)
        assert escalations == []

    def test_new_subprocess_import(self):
        original = "import os"
        new = "import os\nimport subprocess"
        escalations = detect_capability_escalation(original, new)
        assert any("subprocess" in e.lower() for e in escalations)

    def test_security_boundary_modification(self):
        original = "x = 1"
        new = "x = 1\nALLOWED_PATHS = ['brain/everything/']"
        escalations = detect_capability_escalation(original, new)
        assert any("security" in e.lower() for e in escalations)


# ---------------------------------------------------------------------------
# CodePatch validation
# ---------------------------------------------------------------------------


class TestCodePatch:
    """Test the CodePatch validation pipeline."""

    def test_valid_patch_no_violations(self):
        patch = CodePatch(files=[
            FileDiff(
                path="brain/consciousness/tone.py",
                new_content="import logging\nlogger = logging.getLogger(__name__)\n",
            ),
        ])
        assert patch.validate() == []
        assert patch.validate_syntax() == []

    def test_syntax_error_detected(self):
        patch = CodePatch(files=[
            FileDiff(
                path="brain/consciousness/tone.py",
                new_content="def broken(\n",
            ),
        ])
        errors = patch.validate_syntax()
        assert len(errors) >= 1

    def test_denied_pattern_in_patch(self):
        patch = CodePatch(files=[
            FileDiff(
                path="brain/tools/new_tool.py",
                new_content="import subprocess\nsubprocess.run(['ls'])\n",
            ),
        ])
        violations = patch.validate()
        assert len(violations) >= 1

    def test_diff_budget_enforced(self):
        files = [
            FileDiff(path=f"brain/consciousness/file{i}.py", new_content=f"x = {i}\n")
            for i in range(5)
        ]
        patch = CodePatch(files=files)
        violations = patch.validate_diff_budget()
        assert any("Too many files" in v for v in violations)


# ---------------------------------------------------------------------------
# Cloud provider gating (keys alone must not enable self-improve cloud paths)
# ---------------------------------------------------------------------------


class TestPatchProviderCloudGating:
    def test_keys_ignored_without_allow_flag(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS", raising=False)
        from self_improve.provider import PatchProvider

        p = PatchProvider()
        assert p._claude_available is False
        assert p._openai_available is False
        st = p.get_status()
        assert st["claude_available"] is False
        assert st["openai_available"] is False
        assert st["cloud_plugins_enabled"] is False

    def test_keys_honored_when_allow_flag_true(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS", "true")
        from self_improve.provider import PatchProvider

        p = PatchProvider()
        assert p._claude_available is True
        assert p._openai_available is True
        assert p.get_status()["cloud_plugins_enabled"] is True


# ---------------------------------------------------------------------------
# Provider JSON parsing
# ---------------------------------------------------------------------------


class TestProviderParsing:
    """Test patch provider response parsing."""

    def test_parse_valid_json(self):
        from self_improve.provider import PatchProvider
        provider = PatchProvider()
        response = json.dumps({
            "files": [{"path": "brain/consciousness/tone.py", "content": "x = 1\n"}],
            "description": "test patch",
            "confidence": 0.8,
        })
        patch = provider._parse_response(response, "test", "plan_001")
        assert patch is not None
        assert len(patch.files) == 1
        assert patch.confidence == 0.8

    def test_parse_json_with_markdown_fences(self):
        from self_improve.provider import PatchProvider
        provider = PatchProvider()
        response = '```json\n{"files": [{"path": "brain/tools/x.py", "content": "y = 2\\n"}], "description": "test"}\n```'
        patch = provider._parse_response(response, "test", "plan_002")
        assert patch is not None
        assert len(patch.files) == 1

    def test_parse_no_json_returns_none(self):
        from self_improve.provider import PatchProvider
        provider = PatchProvider()
        patch = provider._parse_response("no json here at all", "test", "plan_003")
        assert patch is None

    def test_parse_missing_files_key(self):
        from self_improve.provider import PatchProvider
        provider = PatchProvider()
        response = json.dumps({"description": "no files"})
        patch = provider._parse_response(response, "test", "plan_004")
        assert patch is None

    def test_parse_empty_files_returns_none(self):
        from self_improve.provider import PatchProvider
        provider = PatchProvider()
        response = json.dumps({"files": [], "description": "empty"})
        patch = provider._parse_response(response, "test", "plan_005")
        assert patch is None


# ---------------------------------------------------------------------------
# Targeted test selection
# ---------------------------------------------------------------------------


class TestTargetedTestSelection:
    """Verify the sandbox maps patched files to correct test targets."""

    def test_consciousness_module_maps_to_tests(self):
        test_dir = BRAIN_DIR / "tests"
        targets = Sandbox._select_test_targets(
            ["brain/consciousness/engine.py"], test_dir,
        )
        assert any("test_consciousness" in t for t in targets)

    def test_memory_module_maps_to_tests(self):
        test_dir = BRAIN_DIR / "tests"
        targets = Sandbox._select_test_targets(
            ["brain/memory/core.py"], test_dir,
        )
        assert any("test_memory" in t for t in targets)

    def test_unknown_module_falls_back_to_regression(self):
        test_dir = BRAIN_DIR / "tests"
        targets = Sandbox._select_test_targets(
            ["brain/config.py"], test_dir,
        )
        assert any("audit_regressions" in t for t in targets)

    def test_no_files_falls_back_to_regression(self):
        test_dir = BRAIN_DIR / "tests"
        targets = Sandbox._select_test_targets(None, test_dir)
        assert any("audit_regressions" in t for t in targets)

    def test_multiple_modules_union(self):
        test_dir = BRAIN_DIR / "tests"
        targets = Sandbox._select_test_targets(
            ["brain/consciousness/engine.py", "brain/epistemic/calibration/__init__.py"],
            test_dir,
        )
        assert any("test_consciousness" in t for t in targets)
        assert any("test_contradiction_engine" in t or "test_truth_calibration" in t for t in targets)

    def test_all_mapped_modules_have_existing_test_files(self):
        """Every test file referenced in _MODULE_TO_TESTS must actually exist."""
        test_dir = BRAIN_DIR / "tests"
        for module, test_files in _MODULE_TO_TESTS.items():
            for tf in test_files:
                assert (test_dir / tf).exists(), (
                    f"_MODULE_TO_TESTS['{module}'] references '{tf}' which does not exist"
                )


# ---------------------------------------------------------------------------
# Plan formatting
# ---------------------------------------------------------------------------


class TestPlanFormatting:
    """Ensure Jarvis-style prompt context includes tests and validation intent."""

    def test_format_plan_includes_curiosity_loop_and_targeted_tests(self):
        from self_improve.orchestrator import SelfImprovementOrchestrator

        orch = SelfImprovementOrchestrator(engine=None, dry_run_mode=True)
        request = ImprovementRequest(
            type="self_improve",
            target_module="brain/self_improve/orchestrator.py",
            description="Trace and improve the self-improvement loop",
            evidence=["operator_report", "dry_run_failure"],
            priority=0.5,
        )
        plan = PatchPlan(
            files_to_modify=["brain/self_improve/orchestrator.py"],
            constraints=["Only modify files within allowed scope"],
            test_plan=["Run existing unit tests", "Run kernel simulation for 20 ticks"],
            write_category="self_improve",
        )

        text = orch._format_plan_for_llm(request, plan)

        assert "Jarvis is driving this investigation" in text
        assert "curiosity-led troubleshooting loop" in text
        assert "Targeted Test Contracts" in text
        assert "tests/test_self_improve_sprint1.py" in text
        assert "tests/test_audit_regressions.py" in text
        assert "behavioral contracts" in text
        assert "Master trace reference: docs/SELF_IMPROVEMENT_TRACE_MASTER.md" in text


# ---------------------------------------------------------------------------
# Sandbox evaluation (unit-level with trivial patches)
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a fresh event loop, safe across test suites."""
    return asyncio.run(coro)


class TestSandboxEvaluation:
    """Run sandbox.evaluate() with controlled patches."""

    def test_syntax_error_patch_fails(self):
        sandbox = Sandbox(project_root=str(BRAIN_DIR))
        p = CodePatch(files=[
            FileDiff(
                path="brain/consciousness/tone.py",
                new_content="def broken syntax here(\n",
            ),
        ])
        report = _run_async(sandbox.evaluate(p))
        assert report.status == "failed"
        assert not report.overall_passed
        assert any(d.error_type == "syntax" for d in report.diagnostics)

    def test_valid_trivial_patch_passes_syntax(self):
        """A trivially valid Python file should pass syntax check."""
        sandbox = Sandbox(project_root=str(BRAIN_DIR))
        p = CodePatch(files=[
            FileDiff(
                path="brain/consciousness/tone.py",
                new_content=(
                    '"""Tone module."""\n\n'
                    "from __future__ import annotations\n\n"
                    "TONES = ['professional', 'casual']\n"
                ),
            ),
        ])
        report = _run_async(sandbox.evaluate(p))
        assert report.status in ("passed", "failed")
        assert not any(d.error_type == "syntax" for d in report.diagnostics)


# ---------------------------------------------------------------------------
# Silent stub detection
# ---------------------------------------------------------------------------


class TestSilentStubDetection:
    """EvaluationReport.has_silent_stubs() catches faked passes."""

    def test_lint_stub(self):
        r = EvaluationReport()
        r.lint_passed = True
        r.lint_executed = False
        assert r.has_silent_stubs()

    def test_test_stub(self):
        r = EvaluationReport()
        r.lint_passed = True
        r.lint_executed = True
        r.all_tests_passed = True
        r.tests_executed = False
        assert r.has_silent_stubs()

    def test_sim_stub(self):
        r = EvaluationReport()
        r.lint_passed = True
        r.lint_executed = True
        r.all_tests_passed = True
        r.tests_executed = True
        r.sim_passed = True
        r.sim_executed = False
        assert r.has_silent_stubs()

    def test_no_stubs_when_all_executed(self):
        r = EvaluationReport()
        r.lint_passed = True
        r.lint_executed = True
        r.all_tests_passed = True
        r.tests_executed = True
        r.sim_passed = True
        r.sim_executed = True
        assert not r.has_silent_stubs()


# ---------------------------------------------------------------------------
# Orchestrator dry-run (mocked Ollama)
# ---------------------------------------------------------------------------


class TestOrchestratorDryRun:
    """Test the orchestrator dry-run path with mocked LLM."""

    def test_dry_run_with_mocked_ollama(self):
        """Dry-run produces a record with status='dry_run' when sandbox passes."""
        from self_improve.orchestrator import SelfImprovementOrchestrator

        orch = SelfImprovementOrchestrator(engine=None, dry_run_mode=True)
        orch._auto_frozen = False
        orch._last_attempt_time = 0

        valid_response = json.dumps({
            "files": [{
                "path": "brain/consciousness/tone.py",
                "content": (
                    '"""Tone module."""\n\n'
                    "from __future__ import annotations\n\n"
                    "TONES = ['professional', 'casual', 'urgent']\n"
                ),
            }],
            "description": "Add urgent tone",
            "confidence": 0.7,
        })

        mock_ollama = AsyncMock()
        mock_ollama.chat = AsyncMock(return_value=valid_response)

        request = ImprovementRequest(
            type="consciousness_enhancement",
            target_module="consciousness/tone.py",
            description="Add urgent tone option",
            evidence=["manual_test"],
            priority=0.3,
        )

        passing_report = EvaluationReport(
            patch_id="test",
            status="passed",
            lint_passed=True, lint_executed=True,
            all_tests_passed=True, tests_executed=True,
            sim_passed=True, sim_executed=True,
            overall_passed=True,
            recommendation="promote",
        )

        with (
            patch("self_improve.orchestrator._save_conversation"),
            patch("self_improve.patch_plan.PatchPlan.validate_write_boundaries", return_value=[]),
            patch("self_improve.verification.has_pending", return_value=False),
            patch.object(orch, "_get_code_context", return_value="# tone.py source"),
            patch.object(orch, "_gather_relevant_research", return_value=""),
            patch.object(orch._sandbox, "evaluate", AsyncMock(return_value=passing_report)),
            patch.object(orch._provider, "_claude_available", False),
        ):
            record = _run_async(orch.attempt_improvement(
                request, ollama_client=mock_ollama, dry_run=True, manual=True,
            ))

        assert record.status == "dry_run"
        assert record.iterations >= 1
        assert record.patch is not None
        assert len(record.patch.files) == 1

    def test_dry_run_refused_when_paused(self):
        from self_improve.orchestrator import SelfImprovementOrchestrator

        orch = SelfImprovementOrchestrator(engine=None, dry_run_mode=True)
        orch._paused = True

        request = ImprovementRequest(description="test")
        record = _run_async(orch.attempt_improvement(request, dry_run=True, manual=True))
        assert record.status == "failed"

    def test_dry_run_refused_when_auto_frozen(self):
        from self_improve.orchestrator import SelfImprovementOrchestrator

        orch = SelfImprovementOrchestrator(engine=None, dry_run_mode=True)
        orch._auto_frozen = True

        request = ImprovementRequest(description="test")
        record = _run_async(orch.attempt_improvement(
            request, dry_run=True, manual=False,
        ))
        assert record.status == "failed"


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


class TestVerification:
    """Test the verification module helpers."""

    def test_compare_metrics_stable(self):
        from self_improve.verification import compare_metrics

        baselines = {"tick_p95_ms": 10.0, "error_count": 0.0, "memory_count": 100.0}
        current = {"tick_p95_ms": 10.5, "error_count": 0.0, "memory_count": 105.0}
        result, details = compare_metrics(baselines, current)
        assert result == "stable"

    def test_compare_metrics_regression(self):
        from self_improve.verification import compare_metrics

        baselines = {"tick_p95_ms": 10.0, "error_count": 0.0}
        current = {"tick_p95_ms": 15.0, "error_count": 5.0}
        result, details = compare_metrics(baselines, current)
        assert result == "regressed"

    def test_compare_metrics_improvement(self):
        from self_improve.verification import compare_metrics

        baselines = {"tick_p95_ms": 10.0, "error_count": 0.0}
        current = {"tick_p95_ms": 5.0, "error_count": 0.0}
        result, details = compare_metrics(baselines, current)
        assert result == "improved"

    def test_write_and_read_pending(self, tmp_path):
        from self_improve.verification import (
            PENDING_FILE, write_pending, read_pending, clear_pending,
        )
        fake_pending = tmp_path / "pending_verification.json"
        with patch("self_improve.verification.PENDING_FILE", fake_pending):
            write_pending(
                patch_id="test-001",
                description="test",
                files_changed=["brain/consciousness/tone.py"],
                snapshot_path="/tmp/snap",
                conversation_id="conv_001",
                baselines={"tick_p95_ms": 10.0},
                upgrade_id="upg_pending_test",
            )
            pv = read_pending()
            assert pv is not None
            assert pv.patch_id == "test-001"
            assert pv.upgrade_id == "upg_pending_test"
            assert pv.baselines["tick_p95_ms"] == 10.0

            clear_pending()
            assert not fake_pending.exists()
