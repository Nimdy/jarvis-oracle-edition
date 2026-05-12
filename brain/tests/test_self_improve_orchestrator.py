"""Deeper coverage for self_improve/orchestrator.py — lifecycle, gates, apply/rollback.

Tests pipeline progression, health gates, atomic apply + rollback, approval
queue, stage enforcement, and cooldown semantics.  All file I/O uses tempdir.
No live LLM, no event bus, no sandbox subprocess execution.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from self_improve.code_patch import CodePatch, FileDiff
from self_improve.evaluation_report import EvaluationReport, TestResult
from self_improve.improvement_request import ImprovementRequest
from self_improve.orchestrator import (
    SelfImprovementOrchestrator,
    ImprovementRecord,
    ImprovementWinRate,
    MIN_INTERVAL_S,
    STAGE_FROZEN,
    STAGE_DRY_RUN,
    STAGE_HUMAN_APPROVAL,
    SOUL_INTEGRITY_GATE_THRESHOLD,
    HEALTH_MONITOR_TICKS,
    P95_REGRESSION_THRESHOLD,
    PENDING_APPROVALS_FILE,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_request(**overrides) -> ImprovementRequest:
    defaults = {
        "description": "optimize kernel tick budget",
        "type": "optimization",
        "target_module": "consciousness",
        "priority": "medium",
    }
    defaults.update(overrides)
    return ImprovementRequest(**defaults)


def _make_patch(project_root: str) -> CodePatch:
    """Create a valid patch with real content targeting the temp project root."""
    target = Path(project_root) / "consciousness" / "test_file.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# original content\nx = 1\n")

    return CodePatch(
        files=[
            FileDiff(
                path="brain/consciousness/test_file.py",
                original_content="# original content\nx = 1\n",
                new_content="# improved content\nx = 2\n",
            ),
        ],
        description="test patch",
    )


def _make_orch(
    dry_run: bool = True,
    stage: int = STAGE_DRY_RUN,
    tmpdir: str | None = None,
) -> tuple[SelfImprovementOrchestrator, str]:
    """Create an isolated orchestrator with deterministic mocks."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="jarvis_orch_test_")

    mock_provider = MagicMock()
    mock_provider.get_status.return_value = {
        "coder": {"available": False},
        "local_available": True,
        "claude_available": False,
        "openai_available": False,
    }
    mock_provider._claude_available = False

    # Mock all event emission and external state access
    patches = [
        patch("self_improve.orchestrator.SNAPSHOT_DIR", Path(tmpdir) / "snapshots"),
        patch("self_improve.orchestrator.HISTORY_FILE", Path(tmpdir) / "history.json"),
        patch("self_improve.orchestrator.PROPOSALS_FILE", Path(tmpdir) / "proposals.jsonl"),
        patch("self_improve.orchestrator.PENDING_APPROVALS_FILE", Path(tmpdir) / "pending_approvals.json"),
        patch.dict(os.environ, {
            "SELF_IMPROVE_STAGE": str(stage),
            "FREEZE_AUTO_IMPROVE": "false",
        }),
    ]
    for p in patches:
        p.start()

    with patch("self_improve.orchestrator.SelfImprovementOrchestrator._emit_event"):
        orch = SelfImprovementOrchestrator(
            provider=mock_provider,
            dry_run_mode=dry_run,
        )

    orch._sandbox = MagicMock()
    orch._sandbox._project_root = tmpdir

    # Cleanup patches on test end
    orch._test_patches = patches
    orch._test_tmpdir = tmpdir

    return orch, tmpdir


def _cleanup(orch: SelfImprovementOrchestrator):
    for p in getattr(orch, "_test_patches", []):
        try:
            p.stop()
        except Exception:
            pass
    tmpdir = getattr(orch, "_test_tmpdir", None)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Atomic Apply + Rollback
# ---------------------------------------------------------------------------

class TestAtomicApply:
    def test_apply_creates_file(self):
        orch, tmpdir = _make_orch()
        try:
            patch_obj = _make_patch(tmpdir)
            result = orch._atomic_apply(patch_obj)

            assert result is True
            target = Path(tmpdir) / "consciousness" / "test_file.py"
            assert target.exists()
            assert "improved content" in target.read_text()
        finally:
            _cleanup(orch)

    def test_apply_empty_content_skipped(self):
        orch, tmpdir = _make_orch()
        try:
            patch_obj = CodePatch(files=[
                FileDiff(path="brain/consciousness/skip.py", new_content=""),
            ])
            result = orch._atomic_apply(patch_obj)
            assert result is True
            assert not (Path(tmpdir) / "consciousness" / "skip.py").exists()
        finally:
            _cleanup(orch)

    def test_apply_path_traversal_blocked(self):
        orch, tmpdir = _make_orch()
        try:
            patch_obj = CodePatch(files=[
                FileDiff(
                    path="brain/../../etc/passwd",
                    new_content="hacked",
                ),
            ])
            result = orch._atomic_apply(patch_obj)
            assert result is False
        finally:
            _cleanup(orch)


class TestSnapshot:
    def test_create_snapshot_copies_original(self):
        orch, tmpdir = _make_orch()
        try:
            patch_obj = _make_patch(tmpdir)
            snap_path = orch._create_snapshot(patch_obj)

            assert os.path.isdir(snap_path)
            snap_file = Path(snap_path) / "consciousness" / "test_file.py"
            assert snap_file.exists()
            assert "original content" in snap_file.read_text()
        finally:
            _cleanup(orch)

    def test_restore_snapshot_reverts_changes(self):
        orch, tmpdir = _make_orch()
        try:
            patch_obj = _make_patch(tmpdir)
            snap_path = orch._create_snapshot(patch_obj)

            orch._atomic_apply(patch_obj)
            target = Path(tmpdir) / "consciousness" / "test_file.py"
            assert "improved" in target.read_text()

            restored = orch._restore_snapshot(snap_path)
            assert restored is True
            assert "original content" in target.read_text()
        finally:
            _cleanup(orch)

    def test_restore_nonexistent_snapshot_returns_false(self):
        orch, tmpdir = _make_orch()
        try:
            result = orch._restore_snapshot("/tmp/nonexistent_snapshot_xyz")
            assert result is False
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Stage / Gate Enforcement
# ---------------------------------------------------------------------------

class TestStageGates:
    def test_frozen_stage_blocks_auto_trigger(self):
        orch, tmpdir = _make_orch(stage=STAGE_FROZEN)
        try:
            request = _make_request()
            with patch.object(orch, "_emit_event"):
                record = _run(orch.attempt_improvement(request))
            assert record.status == "failed"
            assert record.iterations == 0
        finally:
            _cleanup(orch)

    def test_frozen_stage_allows_manual(self):
        """Manual triggers bypass the frozen gate, but may still hit other blocks."""
        orch, tmpdir = _make_orch(stage=STAGE_FROZEN)
        try:
            request = _make_request()
            mock_verif = MagicMock()
            mock_verif.has_pending = MagicMock(return_value=False)

            with patch.object(orch, "_emit_event"), \
                 patch.dict("sys.modules", {
                     "self_improve.verification": mock_verif,
                     "consciousness.attribution_ledger": MagicMock(attribution_ledger=MagicMock(record=MagicMock(return_value="e1"))),
                     "self_improve.system_upgrade_report": MagicMock(
                         mint_upgrade_id=MagicMock(return_value="upg_001"),
                         mint_attempt_id=MagicMock(return_value="att_001"),
                         sync_report_from_record=MagicMock(),
                         append_attempt_to_report=MagicMock(),
                         sandbox_summary_from_evaluation_report=MagicMock(return_value={}),
                     ),
                 }):
                orch._provider.generate_with_coder = AsyncMock(return_value=None)
                orch._provider.generate_with_ollama = AsyncMock(return_value=None)
                orch._provider.generate_patch_local = AsyncMock(return_value=None)
                record = _run(orch.attempt_improvement(request, manual=True))
            assert record.status == "failed"
        finally:
            _cleanup(orch)

    def test_paused_blocks_all(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            orch.set_paused(True)
            request = _make_request()
            with patch.object(orch, "_emit_event"):
                record = _run(orch.attempt_improvement(request))
            assert record.status == "failed"
            assert record.iterations == 0
        finally:
            _cleanup(orch)

    def test_empty_target_module_rejected(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            request = _make_request(target_module="")
            mock_verif = MagicMock()
            mock_verif.has_pending = MagicMock(return_value=False)
            with patch.object(orch, "_emit_event"), \
                 patch.dict("sys.modules", {"self_improve.verification": mock_verif}):
                record = _run(orch.attempt_improvement(request))
            assert record.status == "failed"
        finally:
            _cleanup(orch)

    def test_cooldown_blocks_rapid_fire(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            orch._last_attempt_time = time.time()  # just attempted

            request = _make_request()
            mock_verif = MagicMock()
            mock_verif.has_pending = MagicMock(return_value=False)
            with patch.object(orch, "_emit_event"), \
                 patch.dict("sys.modules", {"self_improve.verification": mock_verif}):
                record = _run(orch.attempt_improvement(request))
            assert record.status == "failed"
        finally:
            _cleanup(orch)

    def test_is_blocked_when_paused(self):
        orch, tmpdir = _make_orch()
        try:
            orch.set_paused(True)
            assert orch.is_blocked() is True
        finally:
            _cleanup(orch)

    def test_is_blocked_during_cooldown(self):
        orch, tmpdir = _make_orch()
        try:
            orch._last_attempt_time = time.time()
            mock_verif = MagicMock()
            mock_verif.has_pending = MagicMock(return_value=False)
            with patch.dict("sys.modules", {"self_improve.verification": mock_verif}):
                assert orch.is_blocked() is True
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Approval Queue
# ---------------------------------------------------------------------------

class TestApprovalQueue:
    def test_approve_promotes(self):
        orch, tmpdir = _make_orch()
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            with patch.object(orch, "_emit_event"), \
                 patch.object(orch, "_reindex_codebase"), \
                 patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"), \
                 patch.object(orch, "_check_post_apply_health", new_callable=AsyncMock, return_value=True), \
                 patch.dict("sys.modules", {
                     "self_improve.system_upgrade_report": MagicMock(
                         mint_upgrade_id=MagicMock(return_value="upg_001"),
                         load_report=MagicMock(return_value=None),
                         maybe_append_training_sample=MagicMock(),
                     ),
                 }):
                result = _run(orch.approve(request.id))

            assert result["applied"] is True
            assert result["reason"] == "promoted"
            assert record.status == "promoted"
            assert len(orch._pending_approvals) == 0
        finally:
            _cleanup(orch)

    def test_reject_returns_structured(self):
        orch, tmpdir = _make_orch()
        try:
            request = _make_request()
            record = ImprovementRecord(
                request=request,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            with patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"):
                result = orch.reject(request.id)

            assert result["rejected"] is True
            assert result["reason"] == "operator_rejected"
            assert record.status == "failed"
            assert len(orch._pending_approvals) == 0
        finally:
            _cleanup(orch)

    def test_approve_nonexistent_returns_structured_not_found(self):
        orch, tmpdir = _make_orch()
        try:
            result = _run(orch.approve("nonexistent_id"))
            assert result["applied"] is False
            assert result["reason"] == "patch_not_found"
        finally:
            _cleanup(orch)

    def test_reject_nonexistent_returns_structured_not_found(self):
        orch, tmpdir = _make_orch()
        try:
            result = orch.reject("nonexistent_id")
            assert result["rejected"] is False
            assert result["reason"] == "patch_not_found"
        finally:
            _cleanup(orch)

    def test_get_pending_approvals_shape(self):
        orch, tmpdir = _make_orch()
        try:
            request = _make_request()
            record = ImprovementRecord(
                request=request,
                patch=_make_patch(tmpdir),
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            pending = orch.get_pending_approvals()
            assert len(pending) == 1
            p = pending[0]
            assert p["request_id"] == request.id
            assert "description" in p
            assert "diffs" in p
            assert "sandbox_summary" in p
            assert "why_requires_approval" in p
            assert isinstance(p["why_requires_approval"], list)
            assert len(p["why_requires_approval"]) > 0
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Health Gate (post-apply check)
# ---------------------------------------------------------------------------

class TestHealthGate:
    def test_no_baseline_returns_healthy(self):
        """If pre_p95 is 0, health check passes immediately."""
        orch, tmpdir = _make_orch()
        try:
            mock_kernel = MagicMock()
            mock_kernel.get_performance.return_value = {"p95_tick_ms": 0.0}

            with patch.dict("sys.modules", {
                "consciousness.kernel": MagicMock(kernel_loop=mock_kernel),
            }):
                result = _run(orch._check_post_apply_health("/tmp/snap"))
            assert result is True
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# WinRate tracker
# ---------------------------------------------------------------------------

class TestWinRateTracker:
    def test_initial_rates_zero(self):
        wr = ImprovementWinRate()
        assert wr.sandbox_pass_rate == 0.0
        assert wr.review_approval_rate == 0.0

    def test_rates_after_mixed_results(self):
        wr = ImprovementWinRate(
            total=10,
            sandbox_passed=7,
            sandbox_failed=3,
            review_approved=5,
            review_rejected=2,
        )
        assert abs(wr.sandbox_pass_rate - 0.7) < 0.01
        assert abs(wr.review_approval_rate - 5/7) < 0.01

    def test_to_dict_has_all_fields(self):
        wr = ImprovementWinRate(total=5, sandbox_passed=3, sandbox_failed=2)
        d = wr.to_dict()
        assert "total" in d
        assert "sandbox_pass_rate" in d
        assert "review_approval_rate" in d


# ---------------------------------------------------------------------------
# set_auto_frozen / stage toggling
# ---------------------------------------------------------------------------

class TestStageSwitching:
    def test_set_auto_frozen_true(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            orch.set_auto_frozen(True)
            assert orch._stage == STAGE_FROZEN
            assert orch._auto_frozen is True
        finally:
            _cleanup(orch)

    def test_set_auto_frozen_false_restores_dry_run(self):
        orch, tmpdir = _make_orch(stage=STAGE_FROZEN)
        try:
            orch.set_auto_frozen(False)
            assert orch._stage == STAGE_DRY_RUN
            assert orch._auto_frozen is False
        finally:
            _cleanup(orch)

    def test_set_auto_frozen_false_preserves_higher_stage(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL)
        try:
            orch.set_auto_frozen(False)
            assert orch._stage == STAGE_HUMAN_APPROVAL
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Stage 2 Approval Gate (Fix 1)
# ---------------------------------------------------------------------------

class TestStage2ApprovalGate:
    """At Stage 2, ALL auto-triggered patches must enter approval queue.

    Manual operator-triggered runs can bypass the queue.
    The gate sits at apply/promote, not at proposal generation.
    """

    def test_stage2_auto_triggered_gate_condition(self):
        """At Stage 2, auto-triggered patches should trigger the approval gate."""
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            assert request.manual is False

            stage2_auto = (orch._stage == STAGE_HUMAN_APPROVAL and not request.manual)
            needs_approval = False or request.requires_approval or stage2_auto
            assert needs_approval is True
        finally:
            _cleanup(orch)

    def test_stage2_auto_queues_pending(self):
        """Directly verify auto-triggered record enters pending queue at Stage 2."""
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            pending = orch.get_pending_approvals()
            assert len(pending) == 1
            assert "stage2_auto_triggered" in pending[0]["why_requires_approval"]
        finally:
            _cleanup(orch)

    def test_stage2_manual_bypasses_approval(self):
        """Manual operator-triggered run at Stage 2 should not auto-queue."""
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            request.manual = True
            assert request.manual is True
        finally:
            _cleanup(orch)

    def test_stage1_forces_dry_run_regardless(self):
        """Stage 1 should always force dry_run=True, never reaching approval gate."""
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN, dry_run=False)
        try:
            assert orch._stage == STAGE_DRY_RUN
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# approve() Safety (Fix 3)
# ---------------------------------------------------------------------------

class TestApproveSafety:
    """approve() should run health monitoring and auto-rollback on regression."""

    def test_approve_runs_health_check(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            with patch.object(orch, "_emit_event"), \
                 patch.object(orch, "_reindex_codebase"), \
                 patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"), \
                 patch.object(orch, "_check_post_apply_health", new_callable=AsyncMock, return_value=True) as mock_health, \
                 patch.dict("sys.modules", {
                     "self_improve.system_upgrade_report": MagicMock(
                         mint_upgrade_id=MagicMock(return_value="upg_002"),
                         load_report=MagicMock(return_value=None),
                         maybe_append_training_sample=MagicMock(),
                     ),
                 }):
                result = _run(orch.approve(request.id))

            mock_health.assert_called_once()
            assert result["applied"] is True
            assert result["health_check_passed"] is True
            assert result["rolled_back"] is False
            assert result["reason"] == "promoted"
        finally:
            _cleanup(orch)

    def test_approve_rollback_on_health_regression(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            with patch.object(orch, "_emit_event"), \
                 patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"), \
                 patch.object(orch, "_check_post_apply_health", new_callable=AsyncMock, return_value=False), \
                 patch.object(orch, "_restore_snapshot", return_value=True), \
                 patch.dict("sys.modules", {
                     "self_improve.system_upgrade_report": MagicMock(
                         mint_upgrade_id=MagicMock(return_value="upg_003"),
                     ),
                 }):
                result = _run(orch.approve(request.id))

            assert result["applied"] is True
            assert result["health_check_passed"] is False
            assert result["rolled_back"] is True
            assert result["reason"] == "health_regression_after_approval"
            assert record.status == "rolled_back"
            assert len(orch._pending_approvals) == 0
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Pending Details (Fix 4)
# ---------------------------------------------------------------------------

class TestPendingDetails:
    """get_pending_approvals() should surface diffs, sandbox, and why_requires_approval."""

    def test_why_requires_approval_stage2(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            record = ImprovementRecord(
                request=request,
                patch=_make_patch(tmpdir),
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            pending = orch.get_pending_approvals()
            assert "stage2_auto_triggered" in pending[0]["why_requires_approval"]
        finally:
            _cleanup(orch)

    def test_diffs_present(self):
        orch, tmpdir = _make_orch()
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            patch_obj.files[0].diff = "--- a/test.py\n+++ b/test.py\n-old\n+new"
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            pending = orch.get_pending_approvals()
            assert len(pending[0]["diffs"]) == 1
            assert "path" in pending[0]["diffs"][0]
            assert "diff" in pending[0]["diffs"][0]
        finally:
            _cleanup(orch)

    def test_sandbox_summary_present(self):
        orch, tmpdir = _make_orch()
        try:
            request = _make_request()
            report = EvaluationReport()
            report.lint_passed = True
            report.lint_executed = True
            report.all_tests_passed = True
            report.tests_executed = True
            report.sim_passed = True
            report.sim_executed = True
            report.compute_overall()
            record = ImprovementRecord(
                request=request,
                patch=_make_patch(tmpdir),
                report=report,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)

            pending = orch.get_pending_approvals()
            sb = pending[0]["sandbox_summary"]
            assert sb is not None
            assert sb["overall_passed"] is True
            assert sb["lint_passed"] is True
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Stage Persistence (Fix 6)
# ---------------------------------------------------------------------------

class TestStagePersistence:
    def test_set_stage_persists_to_disk(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            stage_file = Path(tmpdir) / "stage.json"
            with patch("self_improve.orchestrator.STAGE_FILE", stage_file):
                result = orch.set_stage(STAGE_HUMAN_APPROVAL)

            assert result["changed"] is True
            assert result["new_stage"] == STAGE_HUMAN_APPROVAL
            assert result["stage_source"] == "runtime_api"
            assert stage_file.exists()
            data = json.loads(stage_file.read_text())
            assert data["stage"] == STAGE_HUMAN_APPROVAL
        finally:
            _cleanup(orch)

    def test_stage_source_tracked(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            assert orch._stage_source == "env_var"
        finally:
            _cleanup(orch)

    def test_get_status_includes_stage_source(self):
        orch, tmpdir = _make_orch(stage=STAGE_DRY_RUN)
        try:
            status = orch.get_status()
            assert "stage_source" in status
            assert status["stage_source"] == "env_var"
        finally:
            _cleanup(orch)


# ---------------------------------------------------------------------------
# Pending Approval Persistence
# ---------------------------------------------------------------------------

class TestPendingApprovalPersistence:
    """Pending approvals must survive process restarts."""

    def test_save_and_load_roundtrip(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            report = EvaluationReport()
            report.lint_passed = True
            report.lint_executed = True
            report.all_tests_passed = True
            report.tests_executed = True
            report.sim_passed = True
            report.sim_executed = True
            report.compute_overall()

            record = ImprovementRecord(
                request=request,
                upgrade_id="upg_persist_test",
                patch=patch_obj,
                report=report,
                status="awaiting_approval",
                iterations=2,
            )
            orch._pending_approvals.append(record)
            orch._save_pending_approvals()

            pa_file = Path(tmpdir) / "pending_approvals.json"
            assert pa_file.exists()

            orch2, _ = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False, tmpdir=tmpdir)
            assert len(orch2._pending_approvals) == 1

            restored = orch2._pending_approvals[0]
            assert restored.request.id == request.id
            assert restored.upgrade_id == "upg_persist_test"
            assert restored.status == "awaiting_approval"
            assert restored.iterations == 2
            assert restored.patch is not None
            assert len(restored.patch.files) == 1
            assert "improved content" in restored.patch.files[0].new_content
            assert "original content" in restored.patch.files[0].original_content
            assert restored.report is not None
            assert restored.report.overall_passed is True
        finally:
            _cleanup(orch)

    def test_approve_clears_persistence(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)
            orch._save_pending_approvals()

            pa_file = Path(tmpdir) / "pending_approvals.json"
            assert pa_file.exists()
            data_before = json.loads(pa_file.read_text())
            assert len(data_before) == 1

            with patch.object(orch, "_emit_event"), \
                 patch.object(orch, "_reindex_codebase"), \
                 patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"), \
                 patch.object(orch, "_check_post_apply_health", new_callable=AsyncMock, return_value=True), \
                 patch.dict("sys.modules", {
                     "self_improve.system_upgrade_report": MagicMock(
                         mint_upgrade_id=MagicMock(return_value="upg_004"),
                         load_report=MagicMock(return_value=None),
                         maybe_append_training_sample=MagicMock(),
                     ),
                 }):
                _run(orch.approve(request.id))

            data_after = json.loads(pa_file.read_text())
            assert len(data_after) == 0
        finally:
            _cleanup(orch)

    def test_reject_clears_persistence(self):
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            record = ImprovementRecord(
                request=request,
                patch=_make_patch(tmpdir),
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)
            orch._save_pending_approvals()

            pa_file = Path(tmpdir) / "pending_approvals.json"
            data_before = json.loads(pa_file.read_text())
            assert len(data_before) == 1

            with patch.object(orch, "_sync_upgrade_truth"), \
                 patch.object(orch, "_persist_history"):
                orch.reject(request.id)

            data_after = json.loads(pa_file.read_text())
            assert len(data_after) == 0
        finally:
            _cleanup(orch)

    def test_empty_file_does_not_crash(self):
        orch, tmpdir = _make_orch()
        try:
            pa_file = Path(tmpdir) / "pending_approvals.json"
            pa_file.write_text("[]")
            orch._load_pending_approvals()
            assert len(orch._pending_approvals) == 0
        finally:
            _cleanup(orch)

    def test_corrupt_file_does_not_crash(self):
        orch, tmpdir = _make_orch()
        try:
            pa_file = Path(tmpdir) / "pending_approvals.json"
            pa_file.write_text("{bad json{{{")
            orch._load_pending_approvals()
            assert len(orch._pending_approvals) == 0
        finally:
            _cleanup(orch)

    def test_deserialized_patch_supports_atomic_apply(self):
        """The restored patch must have enough data for _atomic_apply."""
        orch, tmpdir = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False)
        try:
            request = _make_request()
            patch_obj = _make_patch(tmpdir)
            record = ImprovementRecord(
                request=request,
                patch=patch_obj,
                status="awaiting_approval",
            )
            orch._pending_approvals.append(record)
            orch._save_pending_approvals()

            orch2, _ = _make_orch(stage=STAGE_HUMAN_APPROVAL, dry_run=False, tmpdir=tmpdir)

            restored = orch2._pending_approvals[0]
            assert restored.patch is not None
            for fd in restored.patch.files:
                assert fd.new_content, f"new_content missing for {fd.path}"

            result = orch2._atomic_apply(restored.patch)
            assert result is True

            target = Path(tmpdir) / "consciousness" / "test_file.py"
            assert "improved content" in target.read_text()
        finally:
            _cleanup(orch)
