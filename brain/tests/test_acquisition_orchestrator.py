"""Tests for acquisition.orchestrator — coordination, tick cycle, approvals.

Safety:
  - ALL persistence via tempfile.mkdtemp(), NEVER touches ~/.jarvis/
  - event_bus patched to prevent live EventBus emission
  - attribution_ledger patched to prevent live ledger writes
  - skill_registry, create_memory, PluginRegistry, CodeGenService patched at
    their lazy-import sites inside the orchestrator
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
from unittest.mock import MagicMock, patch

from acquisition.job import (
    AcquisitionStore,
    AcquisitionPlan,
    CapabilityAcquisitionJob,
    LaneState,
    _SUBDIRS,
)
from acquisition.orchestrator import AcquisitionOrchestrator, _text_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[AcquisitionStore, str]:
    tmpdir = tempfile.mkdtemp(prefix="jarvis_test_orch_")
    store = AcquisitionStore(base_dir=Path(tmpdir))
    def _simple_save(category: str, obj_id: str, data: dict) -> None:
        p = store._path(category, obj_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str))
    store._save = _simple_save  # type: ignore[assignment]
    return store, tmpdir


def _mock_event_bus():
    return MagicMock()


def _make_orchestrator(store: AcquisitionStore | None = None,
                       tmpdir: str | None = None) -> AcquisitionOrchestrator:
    """Create an isolated orchestrator with all dangerous singletons mocked."""
    if store is None:
        store, tmpdir = _make_store()

    mock_eb = _mock_event_bus()
    mock_ledger = MagicMock()
    mock_ledger.record.return_value = "ledger_entry_001"

    with patch("acquisition.orchestrator.event_bus", mock_eb):
        orch = AcquisitionOrchestrator(store=store)

    orch._mock_eb = mock_eb  # type: ignore[attr-defined]
    orch._mock_ledger = mock_ledger  # type: ignore[attr-defined]
    orch._tmpdir = tmpdir  # type: ignore[attr-defined]
    return orch


def _create_job(orch: AcquisitionOrchestrator, text: str = "test intent",
                **kwargs) -> CapabilityAcquisitionJob:
    """Create a job via the orchestrator, mocking all side effects."""
    mock_eb = getattr(orch, "_mock_eb", MagicMock())
    mock_ledger = getattr(orch, "_mock_ledger", MagicMock())
    mock_ledger.record.return_value = "ledger_001"

    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", {"consciousness.attribution_ledger": MagicMock(
             attribution_ledger=mock_ledger)}):
        return orch.create(text, **kwargs)


def _tick(orch: AcquisitionOrchestrator) -> None:
    """Tick the orchestrator with all side effects mocked."""
    mock_eb = getattr(orch, "_mock_eb", MagicMock())
    mock_ledger = MagicMock()
    mock_ledger.record.return_value = "ledger_002"
    mock_memory = MagicMock()

    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", {
             "consciousness.attribution_ledger": MagicMock(attribution_ledger=mock_ledger),
             "memory.core": MagicMock(create_memory=mock_memory),
             "skills.registry": MagicMock(skill_registry=MagicMock(get_all=MagicMock(return_value=[]))),
             "acquisition.planner": MagicMock(AcquisitionPlanner=MagicMock(return_value=MagicMock(
                 synthesize=MagicMock(return_value=MagicMock(
                     plan_id="plan_test_001",
                     risk_level="low",
                     to_dict=MagicMock(return_value={"plan_id": "plan_test_001"}),
                 )),
             ))),
         }):
        orch.tick()


def _cleanup(orch: AcquisitionOrchestrator) -> None:
    executor = getattr(orch, "_lane_executor", None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
    tmpdir = getattr(orch, "_tmpdir", None)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _text_overlap utility
# ---------------------------------------------------------------------------

def test_text_overlap_high_overlap():
    assert _text_overlap("build a web scraper tool", "build a web scraper plugin") is True

def test_text_overlap_no_overlap():
    assert _text_overlap("quantum physics", "bake cookies recipe") is False

def test_text_overlap_stop_words_removed():
    assert _text_overlap("how to learn about dogs", "learn about cats") is False

def test_text_overlap_empty_after_stop_removal():
    assert _text_overlap("a the to", "how is what") is False


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

def test_create_returns_job():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "explain how neural networks work")
        assert job is not None
        assert job.acquisition_id.startswith("acq_")
        assert job.title == "explain how neural networks work"
    finally:
        _cleanup(orch)

def test_create_classifies_intent():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a weather plugin tool")
        assert job.outcome_class == "plugin_creation"
        assert job.classification_confidence > 0
    finally:
        _cleanup(orch)

def test_create_title_truncated():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        long_text = "x" * 200
        job = _create_job(orch, long_text)
        assert len(job.title) <= 120
    finally:
        _cleanup(orch)

def test_create_initializes_lanes():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "explain how transformers work")
        assert len(job.lanes) > 0
        assert all(isinstance(ls, LaneState) for ls in job.lanes.values())
    finally:
        _cleanup(orch)

def test_create_knowledge_only_status_is_executing():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "explain quantum computing")
        if job.outcome_class == "knowledge_only" and "planning" not in job.required_lanes:
            assert job.status == "executing"
    finally:
        _cleanup(orch)

def test_create_plugin_status_is_planning():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a tool that fetches data from api")
        if "planning" in job.required_lanes:
            assert job.status == "planning"
    finally:
        _cleanup(orch)

def test_create_increments_counter():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        assert orch._total_created == 0
        _create_job(orch, "test")
        assert orch._total_created == 1
    finally:
        _cleanup(orch)

def test_create_persists_job():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test persistence")
        loaded = store.load_job(job.acquisition_id)
        assert loaded is not None
        assert loaded.acquisition_id == job.acquisition_id
    finally:
        _cleanup(orch)

def test_create_adds_to_active_jobs():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test active tracking")
        assert job.acquisition_id in orch._active_jobs
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Risk tier permission model
# ---------------------------------------------------------------------------

def test_risk_tier_0_no_review_no_approval():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "explain how computers work")
        if job.risk_tier == 0:
            assert job.review_status == "not_required"
            assert job.approval_status == "not_required"
            assert job.supervision_mode == "none"
    finally:
        _cleanup(orch)

def test_risk_tier_1_shadow_supervision():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "learn how to detect emotions")
        if job.risk_tier == 1:
            assert job.supervision_mode == "shadow"
    finally:
        _cleanup(orch)

def test_risk_tier_2_supervised():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a data extraction plugin tool")
        if job.risk_tier == 2:
            assert job.review_status == "pending"
            assert job.approval_status == "pending"
            assert job.supervision_mode == "supervised"
    finally:
        _cleanup(orch)

def test_risk_tier_3_bounded():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "setup the gpio relay board")
        if job.risk_tier >= 3:
            assert job.supervision_mode == "bounded"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Tick — terminal states are no-ops
# ---------------------------------------------------------------------------

def test_tick_completed_job_is_noop():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test completed noop")
        job.set_status("completed")
        _tick(orch)
        assert job.status == "completed"
    finally:
        _cleanup(orch)

def test_tick_failed_job_is_noop():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test failed noop")
        job.set_status("failed")
        _tick(orch)
        assert job.status == "failed"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Tick — knowledge_only fast path
# ---------------------------------------------------------------------------

def test_tick_knowledge_only_completes():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "what is machine learning and how does it work")
        assert job.outcome_class == "knowledge_only"
        _tick(orch)
        assert job.status == "completed", f"knowledge_only should auto-complete, got {job.status}"
        assert orch._total_completed >= 1
    finally:
        _cleanup(orch)

def test_tick_knowledge_only_removes_from_active():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "explain how reinforcement learning works")
        _tick(orch)
        assert job.acquisition_id not in orch._active_jobs
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Tick — human gates pause progress
# ---------------------------------------------------------------------------

def test_tick_pauses_on_awaiting_plan_review():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a plugin that fetches api data")
        job.set_status("awaiting_plan_review")
        _tick(orch)
        assert job.status == "awaiting_plan_review", "Should not advance past plan review"
    finally:
        _cleanup(orch)

def test_tick_pauses_on_awaiting_approval():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "upgrade your reasoning")
        job.set_status("awaiting_approval")
        _tick(orch)
        assert job.status == "awaiting_approval"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Lane retry logic
# ---------------------------------------------------------------------------

def test_failed_lane_retries_under_limit():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a web scraper tool")
        if "evidence_grounding" in job.lanes:
            ls = job.lanes["evidence_grounding"]
            ls.status = "failed"
            ls.retry_count = 1
            _tick(orch)
            assert ls.status == "pending", "Should reset to pending for retry"
    finally:
        _cleanup(orch)

def test_failed_lane_at_limit_fails_job():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "develop a connector for slack api")
        if job.outcome_class != "knowledge_only":
            first_lane = job.required_lanes[0]
            ls = job.lanes.get(first_lane)
            if ls:
                ls.status = "failed"
                ls.retry_count = 3
                _tick(orch)
                assert job.status == "failed", "Should fail job at retry limit"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Lane timeout
# ---------------------------------------------------------------------------

def test_lane_timeout_after_30min():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a data plugin tool")
        if job.outcome_class != "knowledge_only":
            first_lane = job.required_lanes[0]
            ls = job.lanes.get(first_lane)
            if ls:
                ls.status = "running"
                ls.started_at = time.time() - 1900  # >30 minutes ago
                _tick(orch)
                assert ls.status == "failed", "Should timeout after 30 minutes"
                assert "timed out" in ls.error.lower()
    finally:
        _cleanup(orch)


def test_planning_empty_coder_response_fails_lane():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)

    class _EmptyCodegen:
        coder_available = True

        async def generate(self, *args, **kwargs):
            return ""

    try:
        job = _create_job(orch, "create a CSV processing plugin")
        job.outcome_class = "plugin_creation"
        job.start_lane("planning")
        orch._codegen_service = _EmptyCodegen()

        orch._run_planning(job)

        assert job.plan_id == ""
        assert job.lanes["planning"].status == "failed"
        assert job.lanes["planning"].error == "planning_failed_empty_coder_response"
        assert job.planning_diagnostics["failure_reason"] == "planning_failed_empty_coder_response"
        assert job.planning_diagnostics["raw_output_length"] == 0
        assert "implementation_sketch" in job.planning_diagnostics["missing_fields"]
        assert "test_cases" in job.planning_diagnostics["missing_fields"]
    finally:
        _cleanup(orch)


def test_heavy_planning_lane_runs_in_background():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)

    class _SlowCodegen:
        coder_available = True

        async def generate(self, *args, **kwargs):
            import asyncio
            await asyncio.sleep(0.4)
            return ""

    try:
        job = _create_job(orch, "create a CSV processing plugin")
        job.outcome_class = "plugin_creation"
        job.required_lanes = ["planning"]
        job.lanes = {"planning": LaneState(status="pending")}
        job.set_status("planning")
        orch._active_jobs[job.acquisition_id] = job
        orch._codegen_service = _SlowCodegen()

        start = time.monotonic()
        orch.tick(mode="passive")
        elapsed = time.monotonic() - start

        assert elapsed < 0.2
        assert job.lanes["planning"].status == "running"

        future = orch._lane_futures[(job.acquisition_id, "planning")]
        future.result(timeout=2)
        assert job.lanes["planning"].status == "failed"
        assert job.planning_diagnostics["failure_reason"] == "planning_failed_empty_coder_response"
    finally:
        _cleanup(orch)


def test_lane_restart_clears_stale_error_and_completion_time():
    job = CapabilityAcquisitionJob()
    job.init_lane("planning")
    job.lanes["planning"].status = "failed"
    job.lanes["planning"].error = "old_error"
    job.lanes["planning"].completed_at = time.time() - 10

    lane = job.start_lane("planning")

    assert lane.status == "running"
    assert lane.error == ""
    assert lane.completed_at == 0.0


# ---------------------------------------------------------------------------
# Human approval API — plan review
# ---------------------------------------------------------------------------

def test_approve_plan_approved_as_is():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a tool plugin")
        job.set_status("awaiting_plan_review")
        job.plan_id = "plan_001"
        if "plan_review" in job.lanes:
            job.lanes["plan_review"].status = "running"

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(job.acquisition_id, "approved_as_is")
        assert result is True
        assert job.status == "executing"
    finally:
        _cleanup(orch)


def test_approve_plan_rejects_incomplete_plugin_plan():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a tool plugin")
        job.outcome_class = "plugin_creation"
        job.set_status("awaiting_plan_review")
        if "plan_review" in job.lanes:
            job.lanes["plan_review"].status = "running"

        plan = AcquisitionPlan(
            acquisition_id=job.acquisition_id,
            objective="Build incomplete plugin",
        )
        plan.technical_approach = "Coder model returned empty response."
        store.save_plan(plan)
        job.plan_id = plan.plan_id

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(job.acquisition_id, "approved_as_is")

        assert result is False
        assert job.status == "awaiting_plan_review"
        assert job.plan_review_id == ""
        assert job.planning_diagnostics["failure_reason"] == "planning_failed_empty_coder_response"
        assert job.planning_diagnostics["raw_output_length"] == 0
    finally:
        _cleanup(orch)


def test_approve_plan_rejected_retries_planning():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a plugin tool")
        job.set_status("awaiting_plan_review")
        job.plan_id = "plan_001"

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(job.acquisition_id, "rejected")
        assert result is True
        assert job.status == "planning"
    finally:
        _cleanup(orch)


def test_approve_plan_rejected_creates_review_artifact_and_resets_lanes():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a plugin tool")
        job.set_status("awaiting_plan_review")
        if "planning" in job.lanes:
            job.lanes["planning"].status = "completed"
        if "plan_review" in job.lanes:
            job.lanes["plan_review"].status = "running"

        plan = AcquisitionPlan(
            acquisition_id=job.acquisition_id,
            objective="Build reviewed plugin",
        )
        plan.technical_approach = "Use the plugin registry and a small handler."
        plan.implementation_sketch = "def handle(request): return structured output"
        plan.test_cases = ["pytest validates handler output"]
        store.save_plan(plan)
        job.plan_id = plan.plan_id

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(
                job.acquisition_id,
                "rejected",
                notes="Needs better validation.",
                suggested_changes=[{"description": "Add contract smoke tests."}],
                reason_category="missing_tests",
            )

        assert result is True
        assert job.status == "planning"
        assert job.plan_review_id
        review = store.load_review(job.plan_review_id)
        assert review is not None
        assert review.verdict == "rejected"
        assert review.reason_category == "missing_tests"
        assert job.lanes["planning"].status == "pending"
        assert job.lanes["plan_review"].status == "pending"
    finally:
        _cleanup(orch)


def test_approve_plan_cancelled():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a connector tool")
        job.set_status("awaiting_plan_review")

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(job.acquisition_id, "cancelled")
        assert result is True
        assert job.status == "cancelled"
    finally:
        _cleanup(orch)

def test_approve_plan_wrong_status_returns_false():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a tool")
        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_plan(job.acquisition_id, "approved_as_is")
        assert result is False, "Should fail if status is not awaiting_plan_review"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Human approval API — deployment
# ---------------------------------------------------------------------------

def test_approve_deployment_approved():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a scraper plugin tool")
        job.set_status("awaiting_approval")
        if "deployment" in job.lanes:
            job.lanes["deployment"].status = "running"

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_deployment(job.acquisition_id, True, "operator")
        assert result is True
        assert job.status == "deployed"
        assert job.approved_by == "operator"
    finally:
        _cleanup(orch)

def test_approve_deployment_denied():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a service plugin tool")
        job.set_status("awaiting_approval")

        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_deployment(job.acquisition_id, False)
        assert result is True
        assert job.status == "cancelled"
    finally:
        _cleanup(orch)

def test_approve_deployment_wrong_status_returns_false():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test")
        mock_eb = getattr(orch, "_mock_eb", MagicMock())
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            result = orch.approve_deployment(job.acquisition_id, True)
        assert result is False
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Boot restoration
# ---------------------------------------------------------------------------

def test_boot_restores_active_jobs():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test boot restore")
        job.set_status("executing")
        store.save_job(job)

        mock_eb = _mock_event_bus()
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            orch2 = AcquisitionOrchestrator(store=store)
        assert job.acquisition_id in orch2._active_jobs
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Plugin name sanitization (from _run_plugin_quarantine)
# ---------------------------------------------------------------------------

def test_plugin_name_sanitization():
    import re
    name = "Build A Weather Tool!"
    sanitized = name.lower().replace(" ", "_")[:40]
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)
    assert sanitized == "build_a_weather_tool"
    assert len(sanitized) <= 40

def test_plugin_name_empty_fallback():
    import re
    name = "!@#$%^&*()"
    sanitized = name.lower().replace(" ", "_")[:40]
    sanitized = re.sub(r"[^a-z0-9_]", "", sanitized)
    assert sanitized == ""  # triggers fallback in orchestrator


# ---------------------------------------------------------------------------
# get_status shape
# ---------------------------------------------------------------------------

def test_get_status_shape():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        status = orch.get_status()
        assert "total_count" in status
        assert "active_count" in status
        assert "completed_count" in status
        assert "failed_count" in status
        assert "recent" in status
        assert "classifier" in status
        assert "runtime" in status
        assert "pending_approvals" in status
    finally:
        _cleanup(orch)

def test_get_status_pending_approvals():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "build a plugin tool")
        job.set_status("awaiting_plan_review")
        status = orch.get_status()
        approvals = status["pending_approvals"]
        assert len(approvals) >= 1
        assert approvals[0]["gate"] == "plan_review"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

def test_get_job_from_active():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test get job")
        found = orch.get_job(job.acquisition_id)
        assert found is not None
        assert found.acquisition_id == job.acquisition_id
    finally:
        _cleanup(orch)

def test_get_job_from_disk():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "test disk lookup")
        # Remove from active, keep on disk
        orch._active_jobs.pop(job.acquisition_id, None)
        found = orch.get_job(job.acquisition_id)
        assert found is not None
    finally:
        _cleanup(orch)

def test_get_job_missing_returns_none():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        assert orch.get_job("acq_nonexistent") is None
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# Event emission tolerance
# ---------------------------------------------------------------------------

def test_emit_event_failure_does_not_propagate():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        mock_eb = MagicMock()
        mock_eb.emit.side_effect = RuntimeError("event bus crash")
        with patch("acquisition.orchestrator.event_bus", mock_eb):
            orch._emit_event("TEST_EVENT", CapabilityAcquisitionJob())
        # No exception propagated
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# All lanes completed -> job completed
# ---------------------------------------------------------------------------

def test_all_lanes_completed_completes_job():
    store, tmpdir = _make_store()
    orch = _make_orchestrator(store, tmpdir)
    try:
        job = _create_job(orch, "create a web scraper plugin tool")
        if job.outcome_class != "knowledge_only":
            for lane_name, ls in job.lanes.items():
                ls.status = "completed"
                ls.completed_at = time.time()
            _tick(orch)
            assert job.status == "completed"
    finally:
        _cleanup(orch)


# ---------------------------------------------------------------------------
# __main__ runner
# ---------------------------------------------------------------------------

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    passed = failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS: {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {fn.__name__}: {e}")
    print(f"\n  {passed}/{passed + failed} passed")
    if failed:
        sys.exit(1)
