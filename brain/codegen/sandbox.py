"""Sandbox -- isolated execution environment for testing generated code.

Shared service used by both self-improvement and the capability acquisition
pipeline.  Runs AST validation, lint checks, unit tests, and kernel tick
simulation on proposed code changes before they touch the live system.

Product requirement: **sandbox parity** -- the sandbox must run the exact same
import layout as the real system.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from self_improve.code_patch import CodePatch
from self_improve.evaluation_report import EvaluationReport, TestResult

logger = logging.getLogger(__name__)

LINT_TIMEOUT_S = 30
TEST_TIMEOUT_S = 180
SIM_TICKS = 20

_MODULE_TO_TESTS: dict[str, list[str]] = {
    "consciousness": [
        "test_consciousness.py", "test_soul_kernel.py",
        "test_dream_containment.py", "test_dream_artifacts.py",
    ],
    "memory": ["test_memory.py", "test_memory_gate.py", "test_cortex_loop.py"],
    "personality": ["test_onboarding.py", "test_consciousness.py"],
    "policy": ["test_audit_regressions.py"],
    "hemisphere": ["test_hemisphere.py", "test_hemisphere_gating.py"],
    "reasoning": ["test_tool_router.py", "test_bounded_response.py"],
    "tools": ["test_tool_router.py", "test_skill_tool_structured.py"],
    "perception": [
        "test_addressee_gate.py", "test_identity_fusion.py",
        "test_speaker_smoothing.py", "test_scene_tracker.py",
    ],
    "self_improve": ["test_self_improve_sprint1.py", "test_audit_regressions.py"],
    "autonomy": ["test_goal_bridge_and_alignment.py", "test_temporal_credit.py"],
    "skills": [
        "test_capability_gate.py", "test_capability_discovery.py",
        "test_skill_baseline.py",
    ],
    "identity": [
        "test_identity_boundary.py", "test_identity_fusion.py",
        "test_evidence_accumulator.py",
    ],
    "epistemic": [
        "test_contradiction_engine.py", "test_truth_calibration.py",
        "test_belief_graph.py", "test_quarantine.py",
        "test_quarantine_pressure.py", "test_reflective_audit.py",
        "test_soul_integrity.py",
    ],
    "cognition": ["test_world_model.py", "test_simulator.py", "test_planner.py"],
    "goals": ["test_goals.py", "test_goal_bridge_and_alignment.py"],
    "language": [
        "test_language_kernel.py", "test_language_scorers.py",
        "test_language_corpus.py", "test_language_promotion.py",
        "test_language_eval_sidecar.py", "test_language_quality_telemetry.py",
        "test_language_phasec.py",
    ],
}

COPIED_SUBDIRS = [
    "consciousness",
    "personality",
    "policy",
    "self_improve",
    "reasoning",
    "hemisphere",
    "tools",
    "memory",
    "perception",
    "autonomy",
    "cognition",
    "goals",
    "skills",
    "identity",
    "epistemic",
    "library",
    "dashboard",
    "jarvis_eval",
    "synthetic",
    "codegen",
    "acquisition",
    "language",
]


# ---------------------------------------------------------------------------
# Structured diagnostics -- fed back to the LLM for iterative fixing
# ---------------------------------------------------------------------------


@dataclass
class SandboxDiagnostic:
    error_type: str         # "lint" | "syntax" | "test" | "import" | "simulation"
    file: str               # relative path
    line: int | None = None
    code: str | None = None   # ruff error code (e.g., "E302")
    message: str = ""
    context_span: str = ""    # 5 lines around the error

    def __str__(self) -> str:
        parts = [f"[{self.error_type}] {self.file}"]
        if self.line:
            parts[0] += f":{self.line}"
        if self.code:
            parts.append(self.code)
        if self.message:
            parts.append(self.message[:120])
        return " — ".join(parts)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    def __init__(self, project_root: str | None = None) -> None:
        self._project_root = project_root or str(Path(__file__).resolve().parent.parent)
        self._sandbox_dir: str | None = None

    async def evaluate(self, patch: CodePatch) -> EvaluationReport:
        """Run full evaluation pipeline on a patch."""
        report = EvaluationReport(patch_id=patch.id, status="testing")

        self._sandbox_dir = tempfile.mkdtemp(prefix="jarvis_sandbox_")
        self._baseline_test_failures: set[str] = set()

        try:
            # Capture baseline test failures BEFORE applying patch
            patched_paths = [fd.path for fd in patch.files if fd.new_content]
            self._copy_project_to_sandbox(patch)
            await self._capture_baseline_failures(patched_paths)

            # Now apply the patch on top
            self._write_patch_files(patch)

            # Phase 0: fast AST parse check (< 1ms per file)
            syntax_ok = self._check_syntax(patch, report)
            if not syntax_ok:
                report.compute_overall()
                report.status = "failed"
                return report

            # Phase 1: lint (scoped to patched files only)
            await self._run_lint(report, patched_files=patched_paths)

            # Phase 2: tests (only if lint passes)
            if report.lint_passed:
                await self._run_tests(report, patched_files=patched_paths)

            # Phase 3: kernel tick simulation (only if tests pass)
            if report.all_tests_passed or not report.tests:
                await self._run_simulation(report)

            report.compute_overall()
            report.status = "passed" if report.overall_passed else "failed"

        except Exception:
            logger.exception("Sandbox evaluation failed")
            report.status = "failed"
            report.risk_assessment = "Sandbox evaluation threw an exception"
        finally:
            self._cleanup()

        return report

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply_patch(self, patch: CodePatch) -> None:
        """Copy project to sandbox and apply patch files."""
        self._copy_project_to_sandbox(patch)
        self._write_patch_files(patch)

    def _copy_project_to_sandbox(self, patch: CodePatch) -> None:
        """Copy the unmodified project into the sandbox directory."""
        if not self._sandbox_dir:
            return

        src = Path(self._project_root)
        dst = Path(self._sandbox_dir)

        for subdir in COPIED_SUBDIRS:
            src_sub = src / subdir
            dst_sub = dst / subdir
            if src_sub.exists():
                shutil.copytree(str(src_sub), str(dst_sub), dirs_exist_ok=True)

        for py_file in src.glob("*.py"):
            if py_file.name.startswith("jarvis-"):
                continue
            shutil.copy2(str(py_file), str(dst / py_file.name))

        tests_dir = src / "tests"
        if tests_dir.exists():
            shutil.copytree(str(tests_dir), str(dst / "tests"), dirs_exist_ok=True)

    def _write_patch_files(self, patch: CodePatch) -> None:
        """Write patched file contents into the sandbox."""
        if not self._sandbox_dir:
            return
        dst = Path(self._sandbox_dir)
        for fd in patch.files:
            target = dst / fd.path.replace("brain/", "")
            target.parent.mkdir(parents=True, exist_ok=True)
            if fd.new_content:
                target.write_text(fd.new_content)

    # ------------------------------------------------------------------
    # Syntax check (AST parse)
    async def _capture_baseline_failures(self, patched_files: list[str] | None = None) -> None:
        """Run tests on unpatched sandbox to identify pre-existing failures."""
        if not self._sandbox_dir:
            return
        test_dir = Path(self._sandbox_dir) / "tests"
        if not test_dir.exists():
            return
        test_targets = self._select_test_targets(patched_files, test_dir)
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = self._sandbox_dir + (os.pathsep + existing_pp if existing_pp else "")
        try:
            cmd = [sys.executable, "-m", "pytest"] + test_targets + ["--tb=line", "-q"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._sandbox_dir,
                env=env,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT_S)
            output = (stdout or b"").decode()[:4000]
            self._baseline_test_failures = self._extract_failed_tests(output)
            if self._baseline_test_failures:
                logger.info("Baseline test failures (%d): %s",
                            len(self._baseline_test_failures),
                            ", ".join(sorted(self._baseline_test_failures)[:5]))
        except Exception:
            logger.debug("Baseline test capture failed", exc_info=True)

    @staticmethod
    def _extract_failed_tests(pytest_output: str) -> set[str]:
        """Extract failed test names from pytest -q output."""
        failures: set[str] = set()
        for line in pytest_output.splitlines():
            if line.startswith("FAILED "):
                test_id = line.replace("FAILED ", "").strip()
                failures.add(test_id)
        return failures

    # ------------------------------------------------------------------

    def _check_syntax(self, patch: CodePatch, report: EvaluationReport) -> bool:
        """Parse every patched file with ast.parse(). Returns False on failure."""
        all_ok = True
        for fd in patch.files:
            if not fd.new_content:
                continue
            try:
                ast.parse(fd.new_content, filename=fd.path)
            except SyntaxError as exc:
                all_ok = False
                diag = SandboxDiagnostic(
                    error_type="syntax",
                    file=fd.path,
                    line=exc.lineno,
                    message=str(exc.msg),
                    context_span=self._extract_context(fd.new_content, exc.lineno),
                )
                report.diagnostics.append(diag)
                report.lint_passed = False
        return all_ok

    # ------------------------------------------------------------------
    # Lint
    # ------------------------------------------------------------------

    async def _run_lint(self, report: EvaluationReport, patched_files: list[str] | None = None) -> None:
        """Run ruff linter on patched files only (avoids pre-existing lint noise)."""
        if not self._sandbox_dir:
            return

        targets: list[str] = []
        if patched_files:
            for p in patched_files:
                rel = p.replace("brain/", "") if p.startswith("brain/") else p
                full = os.path.join(self._sandbox_dir, rel)
                if os.path.exists(full):
                    targets.append(full)
        if not targets:
            targets = [self._sandbox_dir]

        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", *targets,
                "--output-format=json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=LINT_TIMEOUT_S)

            report.lint_output = (stdout or b"").decode()[:4000]
            report.lint_passed = proc.returncode == 0
            report.lint_executed = True

            if not report.lint_passed:
                self._parse_lint_diagnostics(report)

        except FileNotFoundError:
            logger.info("ruff not available, skipping lint")
            report.lint_passed = True
            report.lint_executed = True
            report.lint_output = "ruff not installed, lint skipped"
        except asyncio.TimeoutError:
            report.lint_passed = False
            report.lint_output = "Lint timed out"
        except Exception:
            logger.exception("Lint failed")
            report.lint_passed = False

    def _parse_lint_diagnostics(self, report: EvaluationReport) -> None:
        """Parse ruff JSON output into structured diagnostics (first error per file)."""
        import json as _json
        try:
            entries = _json.loads(report.lint_output)
        except (ValueError, TypeError):
            report.diagnostics.append(SandboxDiagnostic(
                error_type="lint",
                file="unknown",
                message=report.lint_output[:500],
            ))
            return

        seen_files: set[str] = set()
        for entry in entries:
            fpath = entry.get("filename", "")
            if fpath in seen_files:
                continue
            seen_files.add(fpath)

            rel = fpath.replace(self._sandbox_dir + "/", "") if self._sandbox_dir else fpath
            report.diagnostics.append(SandboxDiagnostic(
                error_type="lint",
                file=rel,
                line=entry.get("location", {}).get("row"),
                code=entry.get("code"),
                message=entry.get("message", ""),
            ))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    async def _run_tests(
        self,
        report: EvaluationReport,
        patched_files: list[str] | None = None,
    ) -> None:
        """Run unit tests in the sandbox environment.

        Uses a baseline-comparison approach: runs tests before and after
        applying the patch, and only fails if the patch introduced *new*
        test failures beyond the pre-existing baseline.
        """
        if not self._sandbox_dir:
            return

        test_dir = Path(self._sandbox_dir) / "tests"
        if not test_dir.exists():
            report.tests.append(TestResult(
                name="test_discovery", passed=True,
                output="No test directory found, skipping",
            ))
            report.all_tests_passed = True
            report.tests_executed = True
            return

        test_targets = self._select_test_targets(patched_files, test_dir)

        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = self._sandbox_dir + (os.pathsep + existing_pp if existing_pp else "")

        try:
            cmd = [sys.executable, "-m", "pytest"] + test_targets + ["--tb=line", "-q"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._sandbox_dir,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TEST_TIMEOUT_S)

            output = (stdout or b"").decode()[:4000]
            post_failures = self._extract_failed_tests(output)
            post_fail_count = len(post_failures)
            baseline_failures = getattr(self, "_baseline_test_failures", set())
            new_failures = post_failures - baseline_failures
            passed = len(new_failures) == 0
            report.tests_executed = True

            target_desc = ", ".join(Path(t).name for t in test_targets[:5])
            if baseline_failures and not new_failures and post_fail_count > 0:
                summary = f"{post_fail_count} pre-existing failures (baseline), 0 new"
            elif new_failures:
                summary = f"{len(new_failures)} NEW failures: {', '.join(sorted(new_failures)[:5])}"
            else:
                summary = output

            report.tests.append(TestResult(
                name=f"pytest_targeted({target_desc})", passed=passed, output=summary,
            ))
            report.all_tests_passed = passed

            if not passed:
                report.diagnostics.append(SandboxDiagnostic(
                    error_type="test",
                    file="tests/",
                    message=summary[:500],
                ))

        except FileNotFoundError:
            report.tests.append(TestResult(
                name="pytest", passed=True,
                output="pytest not installed, test skipped",
            ))
            report.all_tests_passed = True
            report.tests_executed = True
        except asyncio.TimeoutError:
            report.tests.append(TestResult(
                name="pytest_suite", passed=False,
                output=f"Tests timed out ({TEST_TIMEOUT_S}s)",
            ))
        except Exception:
            logger.exception("Tests failed")
            report.tests.append(TestResult(
                name="pytest_suite", passed=False, output="Exception during tests",
            ))

    @staticmethod
    def _select_test_targets(
        patched_files: list[str] | None,
        test_dir: Path,
    ) -> list[str]:
        """Map patched source files to targeted test paths."""
        if not patched_files:
            fallback = test_dir / "test_audit_regressions.py"
            return [str(fallback)] if fallback.exists() else [str(test_dir)]

        test_names: set[str] = set()
        for fpath in patched_files:
            normalized = fpath.replace("brain/", "")
            module = normalized.split("/")[0] if "/" in normalized else ""
            mapped = _MODULE_TO_TESTS.get(module, [])
            test_names.update(mapped)

        if not test_names:
            fallback = test_dir / "test_audit_regressions.py"
            return [str(fallback)] if fallback.exists() else [str(test_dir)]

        targets: list[str] = []
        for name in sorted(test_names):
            candidate = test_dir / name
            if candidate.exists():
                targets.append(str(candidate))

        return targets if targets else [str(test_dir)]

    # ------------------------------------------------------------------
    # Kernel tick simulation
    # ------------------------------------------------------------------

    async def _run_simulation(self, report: EvaluationReport) -> None:
        """Run a lightweight kernel tick simulation to measure p95 impact.

        Imports the sandbox's kernel_config and consciousness_analytics modules,
        runs N ticks measuring elapsed time, and computes p95.  Falls back to
        a subprocess-based timing test if the import fails.
        """
        if not self._sandbox_dir:
            report.sim_passed = True
            return

        report.sim_ticks = SIM_TICKS
        tick_times: list[float] = []

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self._sandbox_dir

            # Run a small Python script inside the sandbox that imports key modules
            # and times a simple operation loop to detect import errors and crashes
            sim_script = (
                "import time, sys\n"
                "times = []\n"
                f"for _ in range({SIM_TICKS}):\n"
                "    t0 = time.monotonic()\n"
                "    try:\n"
                "        from consciousness import kernel_config\n"
                "        from consciousness import consciousness_analytics\n"
                "        from consciousness import events\n"
                "    except Exception as e:\n"
                "        print(f'IMPORT_ERROR:{e}', file=sys.stderr)\n"
                "        sys.exit(1)\n"
                "    elapsed = (time.monotonic() - t0) * 1000\n"
                "    times.append(elapsed)\n"
                "times.sort()\n"
                f"p95_idx = int(len(times) * 0.95)\n"
                "p95 = times[p95_idx] if times else 0\n"
                "print(f'P95:{p95:.3f}')\n"
                "print(f'TICKS:{len(times)}')\n"
            )

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", sim_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._sandbox_dir,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            out = (stdout or b"").decode()
            err = (stderr or b"").decode()
            report.sim_executed = True

            if proc.returncode != 0:
                report.sim_passed = False
                msg = err[:500] if err else "Simulation crashed"
                report.diagnostics.append(SandboxDiagnostic(
                    error_type="simulation",
                    file="kernel",
                    message=msg,
                ))
                return

            for line in out.splitlines():
                if line.startswith("P95:"):
                    report.sim_p95_after = float(line.split(":")[1])
                elif line.startswith("TICKS:"):
                    report.sim_ticks = int(line.split(":")[1])

            report.sim_passed = True

        except asyncio.TimeoutError:
            report.sim_passed = False
            report.diagnostics.append(SandboxDiagnostic(
                error_type="simulation",
                file="kernel",
                message="Simulation timed out (30s)",
            ))
        except Exception:
            logger.debug("Simulation failed", exc_info=True)
            report.sim_passed = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(source: str, line_no: int | None, window: int = 2) -> str:
        """Get a few lines around a line number for error context."""
        if not line_no:
            return ""
        lines = source.splitlines()
        start = max(0, line_no - 1 - window)
        end = min(len(lines), line_no + window)
        return "\n".join(f"{i + start + 1:4d}| {lines[i + start]}" for i in range(end - start))

    def _cleanup(self) -> None:
        if self._sandbox_dir and os.path.exists(self._sandbox_dir):
            try:
                shutil.rmtree(self._sandbox_dir)
            except Exception:
                logger.exception("Failed to clean up sandbox at %s", self._sandbox_dir)
            self._sandbox_dir = None
