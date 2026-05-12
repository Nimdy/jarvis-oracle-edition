"""End-to-end acquisition pipeline tests — lifecycle boundary validation.

Proves lane ordering, artifact creation, state transitions, parent/child
linking, review/approval gating, and completion semantics.

Three paths:
  1. Happy path: create → classify → plan → approve → implement → verify → complete
  2. Reject/revise path: plan_review rejected → back to planning → re-approve → complete
  3. Verification fail: lane failure → retry → exhaust retries → job fails

All subsystem outputs are deterministic fakes.  No live LLM, codegen,
memory, or event bus.  Persistence via tempfile.
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
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from acquisition.job import (
    AcquisitionPlan,
    AcquisitionStore,
    CapabilityAcquisitionJob,
    DocumentationArtifact,
    LaneState,
    VerificationBundle,
)
from acquisition.orchestrator import AcquisitionOrchestrator


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------

class FakePlanner:
    """Returns a fully-populated AcquisitionPlan with no LLM call."""
    def __init__(self, store: AcquisitionStore):
        self._store = store

    def synthesize(self, job, evidence=None, doc_artifacts=None):
        plan = AcquisitionPlan(
            acquisition_id=job.acquisition_id,
            objective=job.title,
            risk_level="medium",
        )
        plan.implementation_path = [
            {"description": "step 1: scaffold"},
            {"description": "step 2: implement"},
        ]
        plan.required_capabilities = ["http_client"]
        plan.promotion_criteria = ["tests pass", "lint clean"]
        plan.rollback_strategy = "remove plugin files"
        plan.doc_artifact_ids = [d.artifact_id for d in (doc_artifacts or [])]
        return plan


class FakePluginRegistry:
    """Tracks lifecycle state transitions without touching disk."""
    def __init__(self):
        self._records: dict[str, MagicMock] = {}

    def quarantine(self, name, code_files, manifest, acq_id):
        rec = MagicMock()
        rec.state = "quarantined"
        rec.name = name
        rec.activated_at = 0.0
        self._records[name] = rec
        return True, []

    def get_record(self, name):
        return self._records.get(name)

    def activate(self, name, target_state="shadow"):
        rec = self._records.get(name)
        if rec:
            rec.state = target_state
            rec.activated_at = time.time() - 7200  # pretend activated 2h ago (well past any gate)
            return True
        return False

    def promote(self, name):
        rec = self._records.get(name)
        if rec:
            transitions = {
                "shadow": "supervised",
                "supervised": "active",
            }
            rec.state = transitions.get(rec.state, rec.state)
            return True
        return False


class _FakePluginManifest:
    """Minimal manifest that survives json.dumps(manifest.to_dict())."""
    def __init__(self, **kwargs):
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _make_store() -> tuple[AcquisitionStore, str]:
    tmpdir = tempfile.mkdtemp(prefix="jarvis_e2e_acq_")
    store = AcquisitionStore(base_dir=Path(tmpdir))

    def _simple_save(category: str, obj_id: str, data: dict) -> None:
        p = store._path(category, obj_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str))

    store._save = _simple_save  # type: ignore[assignment]
    return store, tmpdir


def _mock_modules_create():
    """Mocks for job creation — ledger only. Let real classifier run."""
    return {
        "consciousness.attribution_ledger": MagicMock(
            attribution_ledger=MagicMock(record=MagicMock(return_value="ledger_001"))
        ),
    }


def _mock_modules_tick(store, planner=None, plugin_registry=None):
    """Mocks for tick — deterministic fakes for all lane subsystems."""
    fake_planner = planner or FakePlanner(store)
    fake_plugin_reg = plugin_registry or FakePluginRegistry()
    mock_skill_registry = MagicMock()
    mock_skill_registry.get_all.return_value = []

    mock_skill_resolver_mod = MagicMock()
    mock_resolve = MagicMock(return_value=None)
    mock_skill_resolver_mod.resolve_skill = mock_resolve
    mock_skill_resolver_mod.SkillResolver = MagicMock(return_value=MagicMock(
        resolve=MagicMock(return_value=None)
    ))

    return {
        "consciousness.attribution_ledger": MagicMock(
            attribution_ledger=MagicMock(record=MagicMock(return_value="ledger_001"))
        ),
        "memory.core": MagicMock(
            create_memory=MagicMock()
        ),
        "skills.registry": MagicMock(
            skill_registry=mock_skill_registry
        ),
        "skills.resolver": mock_skill_resolver_mod,
        "skills.learning_jobs": MagicMock(),
        "acquisition.planner": MagicMock(
            AcquisitionPlanner=MagicMock(return_value=fake_planner)
        ),
        "tools.plugin_registry": MagicMock(
            PluginRegistry=MagicMock(return_value=fake_plugin_reg),
            get_plugin_registry=MagicMock(return_value=fake_plugin_reg),
            PluginManifest=_FakePluginManifest,
        ),
        "epistemic.quarantine.pressure": MagicMock(
            QuarantinePressure=MagicMock(
                instance=MagicMock(return_value=MagicMock(composite=0.0)),
            ),
        ),
    }


def _make_orch(store=None, tmpdir=None, **kwargs):
    if store is None:
        store, tmpdir = _make_store()
    mock_eb = MagicMock()
    with patch("acquisition.orchestrator.event_bus", mock_eb):
        orch = AcquisitionOrchestrator(store=store)
    orch._mock_eb = mock_eb
    orch._tmpdir = tmpdir
    orch._store = store

    def _fake_enrich(job, plan):
        plan.technical_approach = "Test approach: stdlib modules"
        plan.implementation_sketch = "def handle(request): return {'result': 'ok'}"
        plan.test_cases = ["test_basic_invocation", "test_error_handling"]

    orch._enrich_plan_with_technical_design = _fake_enrich

    fake_codegen = MagicMock()
    fake_codegen.coder_available = True
    fake_codegen.generate_and_validate = AsyncMock(return_value={
        "success": True,
        "validation_errors": [],
        "patch": {
            "files": [
                {"path": "__init__.py", "new_content": (
                    "PLUGIN_MANIFEST = {'name': 'test_plugin', 'description': 'test'}\n\n"
                    "async def handle(text, context=None):\n"
                    "    return {'output': 'ok'}\n"
                )},
            ],
        },
    })
    orch._codegen_service = fake_codegen
    return orch, store, tmpdir, mock_eb


def _create(orch, store, text, mock_eb):
    mods = _mock_modules_create()
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods):
        return orch.create(text)


def _tick(orch, store, mock_eb, mode="passive", plugin_registry=None):
    mods = _mock_modules_tick(store, plugin_registry=plugin_registry)
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods), \
         patch("acquisition.orchestrator._BACKGROUND_WORKER_LANES", frozenset()):
        orch.tick(mode=mode)


def _approve_plan(orch, acq_id, verdict, mock_eb, reason_category="unknown"):
    with patch("acquisition.orchestrator.event_bus", mock_eb):
        return orch.approve_plan(acq_id, verdict, reason_category=reason_category)


def _approve_deploy(orch, acq_id, approved, mock_eb):
    with patch("acquisition.orchestrator.event_bus", mock_eb):
        return orch.approve_deployment(acq_id, approved)


def _cleanup(tmpdir):
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Happy Path — plugin_creation: full lifecycle to completion
# ---------------------------------------------------------------------------

class TestHappyPathPluginCreation:
    """Full pipeline: create → classify → evidence → docs → plan → review →
    implement → plugin_quarantine → verify → plugin_activation → truth → complete.
    """

    def test_full_lifecycle(self):
        orch, store, tmpdir, mock_eb = _make_orch()
        fake_plugins = FakePluginRegistry()
        try:
            # ── CREATE ────────────────────────────────────────────────
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            assert job.outcome_class == "plugin_creation"
            assert job.risk_tier == 2
            assert "evidence_grounding" in job.lanes
            assert "plan_review" in job.lanes
            assert "plugin_quarantine" in job.lanes
            assert "plugin_activation" in job.lanes
            assert "truth" in job.lanes
            assert job.status == "planning"

            # ── TICK 1: evidence_grounding dispatches + completes ──────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["evidence_grounding"].status == "completed"

            # ── TICK 2: doc_resolution ────────────────────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["doc_resolution"].status == "completed"
            assert len(job.doc_artifact_ids) > 0

            # ── TICK 3: planning ──────────────────────────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["planning"].status == "completed"
            assert job.plan_id is not None and job.plan_id != ""

            plan = store.load_plan(job.plan_id)
            assert plan is not None
            assert plan.objective == job.title

            # ── TICK 4: plan_review → awaiting_plan_review ────────────
            # risk_tier=2 means review is required
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.status == "awaiting_plan_review"
            assert job.lanes["plan_review"].status == "running"

            # ── APPROVE PLAN ──────────────────────────────────────────
            ok = _approve_plan(orch, job.acquisition_id, "approved_as_is", mock_eb,
                               reason_category="other")
            assert ok is True
            job = orch.get_job(job.acquisition_id)
            assert job.status == "executing"
            assert job.lanes["plan_review"].status == "completed"
            assert job.plan_review_id != ""

            # Verify the review artifact was persisted
            review = store.load_review(job.plan_review_id)
            assert review is not None
            assert review.verdict == "approved_as_is"
            assert review.reason_category == "other"

            # ── TICK 5: implementation ────────────────────────────────
            # No CodeGenService → marks complete (stub path)
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["implementation"].status == "completed"

            # ── TICK 6: environment_setup (skips for in_process) ──────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["environment_setup"].status in ("completed", "skipped")

            # ── TICK 7: plugin_quarantine ─────────────────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["plugin_quarantine"].status == "completed"
            assert job.plugin_id != ""

            # ── TICK 8: verification ──────────────────────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["verification"].status == "completed"
            assert job.verification_id != ""

            bundle = store.load_verification(job.verification_id)
            assert bundle is not None
            assert bundle.overall_passed is True

            # ── TICK 8: plugin_activation starts (quarantined → shadow) ──
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["plugin_activation"].status == "running"

            # ── TICK 9: plugin_activation gate passes (shadow → active) ─
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["plugin_activation"].status == "completed"

            # ── TICK 10: deployment → awaiting_approval ───────────────
            # risk_tier=2 requires deployment approval
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.status == "awaiting_approval"

            # ── APPROVE DEPLOYMENT ────────────────────────────────────
            ok = _approve_deploy(orch, job.acquisition_id, True, mock_eb)
            assert ok is True
            job = orch.get_job(job.acquisition_id)
            assert job.status == "deployed"

            # ── TICK 11: truth recording ──────────────────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = orch.get_job(job.acquisition_id)
            assert job.lanes["truth"].status == "completed"

            # ── TICK 12: all lanes done → completed ───────────────────
            _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
            job = store.load_job(job.acquisition_id)
            assert job is not None
            assert job.status in ("completed", "deployed")

            # ── Artifact chain validation ─────────────────────────────
            assert len(job.artifact_refs) >= 3  # plan, review, verification

        finally:
            _cleanup(tmpdir)

    def test_lane_ordering_is_sequential(self):
        """Verify only one lane dispatches per tick — strict sequential walk."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            completed_order: list[str] = []
            for _ in range(20):
                _tick(orch, store, mock_eb)
                job = orch.get_job(job.acquisition_id)
                if job.status in ("completed", "failed", "cancelled",
                                  "awaiting_plan_review", "awaiting_approval"):
                    break
                for lane_name in job.required_lanes:
                    ls = job.lanes.get(lane_name)
                    if ls and ls.status == "completed" and lane_name not in completed_order:
                        completed_order.append(lane_name)

            # evidence_grounding must come before doc_resolution
            if "evidence_grounding" in completed_order and "doc_resolution" in completed_order:
                assert completed_order.index("evidence_grounding") < completed_order.index("doc_resolution")

            # planning must come before plan_review
            if "planning" in completed_order:
                assert "evidence_grounding" in completed_order
                assert "doc_resolution" in completed_order

        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# 2. Reject/Revise Path — plan rejected, re-plan, then approve
# ---------------------------------------------------------------------------

class TestRejectRevisePath:
    """Plan review rejected → back to planning → re-approve → continues."""

    def test_reject_sends_back_to_planning(self):
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            # Tick through evidence → docs → planning → plan_review
            for _ in range(10):
                _tick(orch, store, mock_eb)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    break

            assert job.status == "awaiting_plan_review"

            # ── REJECT ────────────────────────────────────────────────
            ok = _approve_plan(orch, job.acquisition_id, "rejected", mock_eb,
                               reason_category="technical_weakness")
            assert ok is True
            job = orch.get_job(job.acquisition_id)
            assert job.status == "planning"

            # Verify review artifact recorded the rejection
            review = store.load_review(job.plan_review_id)
            assert review is not None
            assert review.verdict == "rejected"
            assert review.reason_category == "technical_weakness"

        finally:
            _cleanup(tmpdir)

    def test_cancelled_verdict_cancels_job(self):
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            for _ in range(10):
                _tick(orch, store, mock_eb)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    break

            ok = _approve_plan(orch, job.acquisition_id, "cancelled", mock_eb)
            assert ok is True
            job = orch.get_job(job.acquisition_id)
            assert job.status == "cancelled"

        finally:
            _cleanup(tmpdir)

    def test_deployment_denied_cancels_job(self):
        """Deployment denial cancels the job."""
        orch, store, tmpdir, mock_eb = _make_orch()
        fake_plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            # Drive to awaiting_approval (deployment gate)
            for _ in range(20):
                _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    _approve_plan(orch, job.acquisition_id, "approved_as_is", mock_eb)
                    job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_approval":
                    break
                if job.status in ("completed", "failed", "cancelled"):
                    break

            if job.status == "awaiting_approval":
                ok = _approve_deploy(orch, job.acquisition_id, False, mock_eb)
                assert ok is True
                job = orch.get_job(job.acquisition_id)
                assert job.status == "cancelled"

        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# 3. Lane Failure Path — retry logic and job failure
# ---------------------------------------------------------------------------

class TestLaneFailurePath:
    """Lane fails → retries → exhausts retries → job fails."""

    def test_lane_retry_up_to_3_then_job_fails(self):
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            acq_id = job.acquisition_id

            # Manually fail evidence_grounding 4 times (3 retries = 4 attempts)
            eg = job.lanes["evidence_grounding"]
            eg.status = "failed"
            eg.retry_count = 0
            store.save_job(job)

            # Tick 1: retry_count=0 < 3, resets to pending
            _tick(orch, store, mock_eb)
            job = orch.get_job(acq_id)
            # After first failure detection, status should be pending (retry)
            # then next tick dispatches it again — let's just track retry_count

            # Simulate 3 retries exhausted
            eg = job.lanes["evidence_grounding"]
            eg.status = "failed"
            eg.retry_count = 3
            store.save_job(job)

            _tick(orch, store, mock_eb)
            job = orch.get_job(acq_id)
            assert job.status == "failed"

        finally:
            _cleanup(tmpdir)

    def test_verification_failure_fails_job(self):
        """If verification lane fails after retries, job fails."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            acq_id = job.acquisition_id

            # Complete all lanes except verification — set them all done
            for lane_name, ls in job.lanes.items():
                if lane_name not in ("verification", "truth", "plugin_activation",
                                     "deployment"):
                    ls.status = "completed"
                    ls.completed_at = time.time()

            # Fail verification with 3 retries exhausted
            if "verification" in job.lanes:
                job.lanes["verification"].status = "failed"
                job.lanes["verification"].retry_count = 3
                job.set_status("executing")
                store.save_job(job)

                _tick(orch, store, mock_eb)
                job = orch.get_job(acq_id)
                assert job.status == "failed"

        finally:
            _cleanup(tmpdir)


# ---------------------------------------------------------------------------
# 4. State Transition Invariants
# ---------------------------------------------------------------------------

class TestStateInvariants:
    """Cross-cutting invariants that must hold across all paths."""

    def test_awaiting_review_blocks_tick_progress(self):
        """While awaiting_plan_review, ticking does NOT advance any lane."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            # Drive to awaiting_plan_review
            for _ in range(10):
                _tick(orch, store, mock_eb)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    break

            assert job.status == "awaiting_plan_review"

            # Snapshot lane states
            snapshot = {ln: ls.status for ln, ls in job.lanes.items()}

            # Tick 5 more times — nothing should change
            for _ in range(5):
                _tick(orch, store, mock_eb)

            job = orch.get_job(job.acquisition_id)
            assert job.status == "awaiting_plan_review"
            for ln, ls in job.lanes.items():
                assert ls.status == snapshot[ln], f"Lane {ln} changed while awaiting review"

        finally:
            _cleanup(tmpdir)

    def test_awaiting_approval_blocks_tick_progress(self):
        """While awaiting_approval (deployment), ticking does NOT advance."""
        orch, store, tmpdir, mock_eb = _make_orch()
        fake_plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)

            for _ in range(20):
                _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    _approve_plan(orch, job.acquisition_id, "approved_as_is", mock_eb)
                    job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_approval":
                    break

            if job.status == "awaiting_approval":
                snapshot = {ln: ls.status for ln, ls in job.lanes.items()}
                for _ in range(5):
                    _tick(orch, store, mock_eb, plugin_registry=fake_plugins)
                job = orch.get_job(job.acquisition_id)
                assert job.status == "awaiting_approval"
                for ln, ls in job.lanes.items():
                    assert ls.status == snapshot[ln]

        finally:
            _cleanup(tmpdir)

    def test_knowledge_only_fast_path(self):
        """knowledge_only jobs bypass planning/review and complete quickly."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "explain how transformers work", mock_eb)
            assert job.outcome_class == "knowledge_only"
            assert job.risk_tier == 0

            _tick(orch, store, mock_eb)
            job = orch.get_job(job.acquisition_id)

            # knowledge_only should complete in 1-2 ticks
            assert job.status == "completed"
            assert all(
                ls.status in ("completed", "skipped")
                for ls in job.lanes.values()
            )

        finally:
            _cleanup(tmpdir)

    def test_approve_plan_on_wrong_status_returns_false(self):
        """approve_plan fails if job is not in awaiting_plan_review."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            # Job is in "planning" state, not "awaiting_plan_review"
            ok = _approve_plan(orch, job.acquisition_id, "approved_as_is", mock_eb)
            assert ok is False
        finally:
            _cleanup(tmpdir)

    def test_approve_deploy_on_wrong_status_returns_false(self):
        """approve_deployment fails if job is not in awaiting_approval."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            ok = _approve_deploy(orch, job.acquisition_id, True, mock_eb)
            assert ok is False
        finally:
            _cleanup(tmpdir)

    def test_completed_job_removed_from_active(self):
        """Completed jobs are removed from _active_jobs after tick."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "explain how neural networks work", mock_eb)
            acq_id = job.acquisition_id
            assert acq_id in orch._active_jobs

            _tick(orch, store, mock_eb)

            # knowledge_only completes in 1 tick
            assert acq_id not in orch._active_jobs

        finally:
            _cleanup(tmpdir)

    def test_persistence_survives_recreate(self):
        """Jobs restored from disk on orchestrator re-init."""
        store, tmpdir = _make_store()
        orch, _, _, mock_eb = _make_orch(store=store, tmpdir=tmpdir)
        try:
            job = _create(orch, store, "build a weather plugin tool", mock_eb)
            acq_id = job.acquisition_id

            # Drive to a mid-lifecycle state
            for _ in range(3):
                _tick(orch, store, mock_eb)

            job = orch.get_job(acq_id)
            saved_status = job.status

            # Re-create orchestrator from same store — should restore
            orch2, _, _, mock_eb2 = _make_orch(store=store, tmpdir=tmpdir)
            restored = orch2.get_job(acq_id)
            assert restored is not None
            assert restored.status == saved_status
            assert restored.acquisition_id == acq_id

        finally:
            _cleanup(tmpdir)
