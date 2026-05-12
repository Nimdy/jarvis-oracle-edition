"""Tests for self_improve/evaluation_report.py — promotion/rollback logic.

The critical safety invariant here is has_silent_stubs(): it detects when
a validation phase reported passed=True but executed=False, meaning the
result was faked. compute_overall() must reject all silent stubs.

Covers:
  - compute_overall() all paths: all-pass, lint fail, test fail, sim fail
  - has_silent_stubs() detection for all three phases
  - get_first_diagnostics() extraction
  - TestResult dataclass
  - to_dict() serialization
  - Recommendation logic: promote vs rollback vs manual_review
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from self_improve.evaluation_report import EvaluationReport, TestResult


# ---------------------------------------------------------------------------
# has_silent_stubs() — the critical safety invariant
# ---------------------------------------------------------------------------

class TestSilentStubs:
    def test_lint_passed_but_not_executed(self):
        report = EvaluationReport(lint_passed=True, lint_executed=False)
        assert report.has_silent_stubs() is True

    def test_tests_passed_but_not_executed(self):
        report = EvaluationReport(all_tests_passed=True, tests_executed=False)
        assert report.has_silent_stubs() is True

    def test_sim_passed_but_not_executed(self):
        report = EvaluationReport(sim_passed=True, sim_executed=False)
        assert report.has_silent_stubs() is True

    def test_all_executed_no_stubs(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            all_tests_passed=True, tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        assert report.has_silent_stubs() is False

    def test_failed_but_not_executed_is_not_stub(self):
        report = EvaluationReport(
            lint_passed=False, lint_executed=False,
            all_tests_passed=False, tests_executed=False,
            sim_passed=False, sim_executed=False,
        )
        assert report.has_silent_stubs() is False

    def test_all_three_stubbed(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=False,
            all_tests_passed=True, tests_executed=False,
            sim_passed=True, sim_executed=False,
        )
        assert report.has_silent_stubs() is True


# ---------------------------------------------------------------------------
# compute_overall() — promotion/rollback decision
# ---------------------------------------------------------------------------

class TestComputeOverall:
    def test_all_pass_promotes(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[TestResult(name="test_a", passed=True)],
            tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.overall_passed is True
        assert report.recommendation == "promote"

    def test_lint_fail_rollback(self):
        report = EvaluationReport(
            lint_passed=False, lint_executed=True,
            all_tests_passed=True, tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.overall_passed is False
        assert report.recommendation == "rollback"

    def test_test_fail_rollback(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[TestResult(name="test_a", passed=False)],
            tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.overall_passed is False
        assert report.all_tests_passed is False
        assert report.recommendation == "rollback"

    def test_sim_fail_manual_review(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[TestResult(name="test_a", passed=True)],
            tests_executed=True,
            sim_passed=False, sim_executed=True,
        )
        report.compute_overall()
        assert report.overall_passed is False
        assert report.recommendation == "manual_review"

    def test_silent_stubs_block_promotion(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=False,
            all_tests_passed=True, tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.overall_passed is False
        assert report.recommendation == "manual_review"
        assert "silent stubs" in report.risk_assessment.lower()

    def test_empty_tests_not_executed_not_stub(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[], tests_executed=False,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.has_silent_stubs() is False
        assert report.all_tests_passed is False
        assert report.overall_passed is False

    def test_empty_tests_executed_passes(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[], tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.all_tests_passed is True
        assert report.overall_passed is True

    def test_mixed_test_results(self):
        report = EvaluationReport(
            lint_passed=True, lint_executed=True,
            tests=[
                TestResult(name="t1", passed=True),
                TestResult(name="t2", passed=False),
                TestResult(name="t3", passed=True),
            ],
            tests_executed=True,
            sim_passed=True, sim_executed=True,
        )
        report.compute_overall()
        assert report.all_tests_passed is False
        assert report.overall_passed is False


# ---------------------------------------------------------------------------
# get_first_diagnostics()
# ---------------------------------------------------------------------------

class TestGetFirstDiagnostics:
    def test_empty_diagnostics(self):
        report = EvaluationReport()
        assert report.get_first_diagnostics() == []

    def test_extracts_attrs(self):
        class FakeDiag:
            error_type = "lint"
            file = "brain/test.py"
            line = 10
            code = "E501"
            message = "line too long"
            context_span = "x = 1"

        report = EvaluationReport(diagnostics=[FakeDiag()])
        result = report.get_first_diagnostics()
        assert len(result) == 1
        assert result[0]["error_type"] == "lint"
        assert result[0]["file"] == "brain/test.py"
        assert result[0]["line"] == 10

    def test_respects_limit(self):
        class FakeDiag:
            error_type = "lint"
            file = ""
            line = 0
            code = ""
            message = ""
            context_span = ""

        report = EvaluationReport(diagnostics=[FakeDiag() for _ in range(10)])
        assert len(report.get_first_diagnostics(limit=3)) == 3

    def test_non_object_diagnostics_skipped(self):
        report = EvaluationReport(diagnostics=["just a string", 42])
        result = report.get_first_diagnostics()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

class TestTestResult:
    def test_creation(self):
        r = TestResult(name="test_foo", passed=True, output="ok", duration_s=0.5)
        assert r.name == "test_foo"
        assert r.passed is True
        assert r.duration_s == 0.5

    def test_defaults(self):
        r = TestResult(name="test_bar", passed=False)
        assert r.output == ""
        assert r.duration_s == 0.0


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_shape(self):
        report = EvaluationReport(patch_id="patch_abc")
        d = report.to_dict()
        assert d["patch_id"] == "patch_abc"
        assert "lint_passed" in d
        assert "tests_passed" in d
        assert "sim_passed" in d
        assert "overall_passed" in d
        assert "recommendation" in d
        assert "has_silent_stubs" in d

    def test_json_serializable(self):
        import json
        report = EvaluationReport(patch_id="patch_xyz")
        json.dumps(report.to_dict())

    def test_diagnostics_count(self):
        report = EvaluationReport(diagnostics=[1, 2, 3])
        d = report.to_dict()
        assert d["diagnostics_count"] == 3
