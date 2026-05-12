"""Evaluation Report -- test results, diagnostics, performance comparison, and
risk assessment for a patch.

Includes structured diagnostics (SandboxDiagnostic) that can be fed back to
the coding LLM for iterative fixing, and execution flags to detect silent stubs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from self_improve.sandbox import SandboxDiagnostic


@dataclass
class TestResult:
    name: str
    passed: bool
    output: str = ""
    duration_s: float = 0.0


@dataclass
class EvaluationReport:
    patch_id: str = ""
    timestamp: float = field(default_factory=time.time)
    status: Literal["pending", "testing", "passed", "failed", "promoted", "rolled_back"] = "pending"

    # Lint
    lint_passed: bool = False
    lint_output: str = ""
    lint_executed: bool = False

    # Tests
    tests: list[TestResult] = field(default_factory=list)
    all_tests_passed: bool = False
    tests_executed: bool = False

    # Simulation
    sim_ticks: int = 0
    sim_p95_before: float = 0.0
    sim_p95_after: float = 0.0
    sim_passed: bool = False
    sim_executed: bool = False

    # Shadow evaluation (future)
    shadow_decisions: int = 0
    shadow_improvement: float = 0.0
    shadow_passed: bool = False

    # Overall
    overall_passed: bool = False
    risk_assessment: str = ""
    recommendation: Literal["promote", "rollback", "manual_review"] = "manual_review"

    # Structured diagnostics for LLM feedback loop
    diagnostics: list[Any] = field(default_factory=list)

    def compute_overall(self) -> None:
        if self.tests:
            self.all_tests_passed = all(t.passed for t in self.tests)
        elif self.tests_executed:
            self.all_tests_passed = True
        # If tests list is empty AND tests weren't executed (e.g. skipped
        # because lint failed), leave all_tests_passed at its default (False)
        # so has_silent_stubs() doesn't false-positive.

        if self.has_silent_stubs():
            self.overall_passed = False
            self.recommendation = "manual_review"
            self.risk_assessment = "silent stubs detected"
            return

        self.overall_passed = (
            self.lint_passed
            and self.all_tests_passed
            and self.sim_passed
        )
        if self.overall_passed:
            self.recommendation = "promote"
        elif not self.lint_passed or not self.all_tests_passed:
            self.recommendation = "rollback"
        else:
            self.recommendation = "manual_review"

    def has_silent_stubs(self) -> bool:
        """Detect if any validation phase was faked (returned constant without running)."""
        if self.lint_passed and not self.lint_executed:
            return True
        if self.all_tests_passed and not self.tests_executed:
            return True
        if self.sim_passed and not self.sim_executed:
            return True
        return False

    def get_first_diagnostics(self, limit: int = 3) -> list[dict[str, Any]]:
        """Return first N diagnostics as dicts for LLM feedback."""
        results: list[dict[str, Any]] = []
        for diag in self.diagnostics[:limit]:
            if hasattr(diag, "__dict__"):
                results.append({
                    "error_type": getattr(diag, "error_type", "unknown"),
                    "file": getattr(diag, "file", ""),
                    "line": getattr(diag, "line", None),
                    "code": getattr(diag, "code", None),
                    "message": getattr(diag, "message", ""),
                    "context_span": getattr(diag, "context_span", ""),
                })
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "status": self.status,
            "lint_passed": self.lint_passed,
            "lint_executed": self.lint_executed,
            "tests_passed": self.all_tests_passed,
            "tests_executed": self.tests_executed,
            "test_count": len(self.tests),
            "sim_passed": self.sim_passed,
            "sim_executed": self.sim_executed,
            "sim_p95_after": self.sim_p95_after,
            "overall_passed": self.overall_passed,
            "recommendation": self.recommendation,
            "risk": self.risk_assessment,
            "diagnostics_count": len(self.diagnostics),
            "has_silent_stubs": self.has_silent_stubs(),
        }
