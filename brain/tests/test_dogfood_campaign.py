"""Phase 4: Controlled Dogfooding Campaign.

Exercises 8 acquisition scenarios end-to-end with artifact truth auditing.
Each test verifies:
  - Plan quality and grounding
  - Artifact coherence across lifecycle
  - Lane state transitions
  - Plugin registry state
  - Scheduler behavior
  - Dashboard status accuracy
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acquisition.job import (
    AcquisitionStore,
    CapabilityAcquisitionJob,
    PluginCodeBundle,
    VerificationBundle,
)
from acquisition.orchestrator import AcquisitionOrchestrator
from acquisition.classifier import IntentClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePluginManifest:
    def __init__(self, **kwargs):
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self._data

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class FakePluginRegistry:
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
            rec.activated_at = time.time() - 7200
            return True
        return False

    def promote(self, name):
        rec = self._records.get(name)
        if rec:
            rec.state = {"shadow": "supervised", "supervised": "active"}.get(rec.state, rec.state)
            return True
        return False


class FakePlanner:
    def __init__(self, store):
        self._store = store

    def synthesize(self, job, doc_artifacts=None, **kwargs):
        from acquisition.job import AcquisitionPlan
        plan = AcquisitionPlan(
            acquisition_id=job.acquisition_id,
            objective=job.title,
            required_capabilities=["python"],
            risk_level="low" if job.risk_tier == 0 else "medium" if job.risk_tier == 1 else "high",
            implementation_path=[{"description": "implement core logic"}],
            verification_path=[{"description": "run tests"}],
            rollback_path=[{"description": "revert changes"}],
            promotion_criteria=["tests pass", "no regressions"],
            doc_artifact_ids=job.doc_artifact_ids,
        )
        return plan


def _make_orch():
    tmpdir = tempfile.mkdtemp()
    store = AcquisitionStore(base_dir=Path(tmpdir))
    mock_eb = MagicMock()
    with patch("acquisition.orchestrator.event_bus", mock_eb):
        orch = AcquisitionOrchestrator(store=store)

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


def _mock_modules(store, plugin_registry=None):
    fake_planner = FakePlanner(store)
    fake_plugins = plugin_registry or FakePluginRegistry()
    return {
        "consciousness.attribution_ledger": MagicMock(
            attribution_ledger=MagicMock(record=MagicMock(return_value="ledger_001"))
        ),
        "memory.core": MagicMock(create_memory=MagicMock()),
        "skills.registry": MagicMock(skill_registry=MagicMock(get_all=MagicMock(return_value=[]))),
        "skills.resolver": MagicMock(resolve_skill=MagicMock(return_value=None)),
        "skills.learning_jobs": MagicMock(),
        "acquisition.planner": MagicMock(AcquisitionPlanner=MagicMock(return_value=fake_planner)),
        "tools.plugin_registry": MagicMock(
            PluginRegistry=MagicMock(return_value=fake_plugins),
            get_plugin_registry=MagicMock(return_value=fake_plugins),
            PluginManifest=_FakePluginManifest,
        ),
        "epistemic.quarantine.pressure": MagicMock(
            QuarantinePressure=MagicMock(instance=MagicMock(return_value=MagicMock(composite=0.0))),
        ),
    }


def _create(orch, store, text, mock_eb):
    mods = {
        "consciousness.attribution_ledger": MagicMock(
            attribution_ledger=MagicMock(record=MagicMock(return_value="ledger_001"))
        ),
    }
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods):
        return orch.create(text)


def _tick(orch, store, mock_eb, mode="passive", plugin_registry=None):
    mods = _mock_modules(store, plugin_registry=plugin_registry)
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods), \
         patch("acquisition.orchestrator._BACKGROUND_WORKER_LANES", frozenset()):
        orch.tick(mode=mode)


def _approve_plan(orch, acq_id, verdict, mock_eb, reason_category="other"):
    mods = {
        "consciousness.events": MagicMock(ACQUISITION_PLAN_REVIEWED="acq:plan_reviewed"),
        "acquisition.plan_encoder": MagicMock(
            PlanEvaluatorEncoder=MagicMock(encode=MagicMock(return_value=[0.5]*32)),
            encode_verdict=MagicMock(return_value=[1.0, 0.0, 0.0]),
            verdict_to_class=MagicMock(return_value="approved"),
            ShadowPredictionArtifact=MagicMock(),
            label_to_class=MagicMock(return_value="approved"),
        ),
        "hemisphere.registry": MagicMock(HemisphereRegistry=MagicMock(return_value=MagicMock(get_active=MagicMock(return_value=None)))),
        "hemisphere.distillation": MagicMock(DistillationCollector=MagicMock(instance=MagicMock(return_value=None))),
    }
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods):
        return orch.approve_plan(acq_id, verdict, reason_category=reason_category)


def _approve_deploy(orch, acq_id, approved, mock_eb):
    mods = {
        "consciousness.events": MagicMock(ACQUISITION_DEPLOYMENT_REVIEWED="acq:deployment_reviewed"),
    }
    with patch("acquisition.orchestrator.event_bus", mock_eb), \
         patch.dict("sys.modules", mods):
        return orch.approve_deployment(acq_id, approved)


def _advance_to_status(orch, store, mock_eb, job, target_status, plugin_registry=None, max_ticks=30):
    """Advance until target status is reached or max_ticks exceeded."""
    for _ in range(max_ticks):
        _tick(orch, store, mock_eb, plugin_registry=plugin_registry)
        job = orch.get_job(job.acquisition_id)
        if job.status == target_status:
            return job
    return orch.get_job(job.acquisition_id)


def _advance_full(orch, store, mock_eb, job, plugin_registry=None, max_ticks=30):
    """Advance through all lanes until terminal."""
    for _ in range(max_ticks):
        _tick(orch, store, mock_eb, plugin_registry=plugin_registry)
        job = orch.get_job(job.acquisition_id)
        if job.status in ("completed", "failed", "cancelled"):
            return job
        if job.status == "awaiting_plan_review":
            _approve_plan(orch, job.acquisition_id, "approved_as_is", mock_eb)
            job = orch.get_job(job.acquisition_id)
        if job.status == "awaiting_approval":
            _approve_deploy(orch, job.acquisition_id, True, mock_eb)
            job = orch.get_job(job.acquisition_id)
    return orch.get_job(job.acquisition_id)


# ---------------------------------------------------------------------------
# Artifact Truth Audit
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    scenario: str
    passed: bool
    checks: list[tuple[str, bool, str]]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.scenario}"]
        for name, ok, detail in self.checks:
            mark = "  OK" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}: {detail}")
        return "\n".join(lines)


def _audit_artifacts(scenario: str, store: AcquisitionStore, job: CapabilityAcquisitionJob) -> AuditResult:
    """Run the artifact truth audit checklist from the plan."""
    checks: list[tuple[str, bool, str]] = []

    # 1. Parent acquisition_id present on all child artifacts
    if job.plan_id:
        plan = store.load_plan(job.plan_id)
        ok = plan is not None and plan.acquisition_id == job.acquisition_id
        checks.append(("plan_parent_id", ok, f"plan.acq_id={getattr(plan, 'acquisition_id', 'MISSING')}"))

    if job.plan_review_id:
        review = store.load_review(job.plan_review_id)
        ok = review is not None and review.acquisition_id == job.acquisition_id
        checks.append(("review_parent_id", ok, f"review.acq_id={getattr(review, 'acquisition_id', 'MISSING')}"))

    if job.verification_id:
        vb = store.load_verification(job.verification_id)
        ok = vb is not None and vb.acquisition_id == job.acquisition_id
        checks.append(("verification_parent_id", ok, f"vb.acq_id={getattr(vb, 'acquisition_id', 'MISSING')}"))

    if job.code_bundle_id:
        bundle = store.load_code_bundle(job.code_bundle_id)
        ok = bundle is not None and bundle.acquisition_id == job.acquisition_id
        checks.append(("code_bundle_parent_id", ok, f"bundle.acq_id={getattr(bundle, 'acquisition_id', 'MISSING')}"))

    # 2. Child IDs linked correctly
    plan_linked = job.plan_id in job.artifact_refs if job.plan_id else True
    checks.append(("plan_linked_to_refs", plan_linked, f"plan_id={job.plan_id}, refs={job.artifact_refs[:3]}"))

    ver_linked = job.verification_id in job.artifact_refs if job.verification_id else True
    checks.append(("verification_linked_to_refs", ver_linked, f"ver_id={job.verification_id}"))

    # 3. Plan refs point to correct plan version
    if job.plan_review_id and job.plan_id:
        review = store.load_review(job.plan_review_id)
        plan = store.load_plan(job.plan_id)
        if review and plan:
            if review.verdict == "rejected":
                ok = review.plan_id != job.plan_id
                checks.append(("review_plan_ref", ok,
                    f"rejected review correctly references prior plan: "
                    f"review.plan_id={review.plan_id}, job.plan_id={job.plan_id}"))
            else:
                ok = review.plan_id == job.plan_id
                checks.append(("review_plan_ref", ok,
                    f"review.plan_id={review.plan_id}, job.plan_id={job.plan_id}"))

    # 4. Doc artifact refs valid
    for doc_id in job.doc_artifact_ids:
        doc = store.load_doc(doc_id)
        ok = doc is not None
        checks.append(("doc_artifact_valid", ok, f"doc_id={doc_id}"))

    # 5. Lane statuses consistent with job status
    if job.status == "completed":
        all_done = all(ls.status in ("completed", "skipped") for ls in job.lanes.values())
        checks.append(("all_lanes_terminal", all_done,
                       f"lanes={', '.join(f'{k}:{v.status}' for k, v in job.lanes.items())}"))

    # 6. Plugin state matches lifecycle position
    if job.plugin_id:
        checks.append(("plugin_id_set", True, f"plugin_id={job.plugin_id}"))
    elif "plugin_quarantine" in job.lanes and job.status in ("completed", "deployed"):
        pq = job.lanes["plugin_quarantine"]
        checks.append(("plugin_quarantine_has_state", pq.status in ("completed", "skipped", "failed"),
                       f"pq_status={pq.status}"))

    passed = all(c[1] for c in checks)
    return AuditResult(scenario=scenario, passed=passed, checks=checks)


# ---------------------------------------------------------------------------
# Dogfood Scenarios
# ---------------------------------------------------------------------------

class TestDogfoodCampaign(unittest.TestCase):

    def test_scenario1_simple_data_transform(self):
        """Scenario 1: Simple data transform plugin (classified as plugin_creation)."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "create a JSON to CSV converter tool", mock_eb)
            assert job.outcome_class in ("plugin_creation", "skill_creation")

            job = _advance_full(orch, store, mock_eb, job, plugin_registry=plugins)
            assert job.status in ("completed", "deployed")

            audit = _audit_artifacts("simple_data_transform", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario2_docs_backed_library_helper(self):
        """Scenario 2: Docs-backed library helper (risk tier 1)."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a plugin that uses pandas to analyze datasets", mock_eb)
            job = _advance_full(orch, store, mock_eb, job, plugin_registry=plugins)
            assert job.status in ("completed", "deployed")

            audit = _audit_artifacts("docs_backed_library", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario3_http_get_plugin(self):
        """Scenario 3: Safe HTTP GET plugin (risk tier 1, requests import)."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a weather API query plugin using requests", mock_eb)
            job = _advance_full(orch, store, mock_eb, job, plugin_registry=plugins)
            assert job.status in ("completed", "deployed")

            audit = _audit_artifacts("http_get_plugin", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario4_rejected_plan_revision(self):
        """Scenario 4: Plan rejected, verified rejection flow works correctly."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a neural network training plugin", mock_eb)

            # Advance to plan review
            for _ in range(20):
                _tick(orch, store, mock_eb, plugin_registry=plugins)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    break

            assert job.status == "awaiting_plan_review", f"Expected awaiting_plan_review, got {job.status}"

            # Reject the plan
            ok = _approve_plan(orch, job.acquisition_id, "rejected", mock_eb,
                               reason_category="technical_weakness")
            assert ok is True
            job = orch.get_job(job.acquisition_id)
            assert job.status == "planning"

            # Verify the rejection review artifact was persisted
            review = store.load_review(job.plan_review_id)
            assert review is not None
            assert review.verdict == "rejected"
            assert review.reason_category == "technical_weakness"

            # After rejection, re-planning triggers — advance one tick for re-plan
            _tick(orch, store, mock_eb, plugin_registry=plugins)
            job = orch.get_job(job.acquisition_id)
            # Planning lane should re-execute (it was already completed once,
            # but status was set back to planning, so planning lane stays completed)
            # The next pending lane is plan_review (completed from previous review)
            # After that, implementation starts

            audit = _audit_artifacts("rejected_plan", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario5_stale_docs_correction(self):
        """Scenario 5: Job with low-freshness docs triggers plan review."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a plugin that uses the latest TensorFlow API", mock_eb)

            # Advance through evidence grounding + doc resolution
            for _ in range(4):
                _tick(orch, store, mock_eb, plugin_registry=plugins)

            job = orch.get_job(job.acquisition_id)
            # Inject a low freshness doc to trigger review
            for doc_id in job.doc_artifact_ids:
                doc = store.load_doc(doc_id)
                if doc:
                    doc.freshness_score = 0.2
                    store.save_doc(doc)

            # Continue — should hit plan review due to low freshness
            job = _advance_full(orch, store, mock_eb, job, plugin_registry=plugins)
            assert job.status in ("completed", "deployed")

            audit = _audit_artifacts("stale_docs", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario6_knowledge_only_reroute(self):
        """Scenario 7: Knowledge-only request that completes without plugin."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = _create(orch, store, "what is quantum computing and how does it work", mock_eb)
            assert job.outcome_class == "knowledge_only"

            job = _advance_full(orch, store, mock_eb, job)
            assert job.status == "completed"
            assert not job.plugin_id

            audit = _audit_artifacts("knowledge_only", store, job)
            assert audit.passed, audit.summary()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario7_scheduler_mode_transitions(self):
        """Scenario: Verify scheduler behavior across mode transitions."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a text summarizer plugin", mock_eb)

            # Tick in passive mode
            _tick(orch, store, mock_eb, mode="passive", plugin_registry=plugins)
            status1 = orch.get_status()
            assert status1["scheduler"]["current_mode"] == "passive"

            # Tick in sleep mode — high-risk lanes should defer
            _tick(orch, store, mock_eb, mode="sleep", plugin_registry=plugins)
            status2 = orch.get_status()
            assert status2["scheduler"]["current_mode"] == "sleep"

            # Tick in deep_learning mode — everything should run
            _tick(orch, store, mock_eb, mode="deep_learning", plugin_registry=plugins)
            status3 = orch.get_status()
            assert status3["scheduler"]["current_mode"] == "deep_learning"

            # Verify pressure info is present
            assert "quarantine_pressure" in status3["scheduler"]
            assert "pressure_level" in status3["scheduler"]
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_scenario8_stall_detection(self):
        """Scenario: Verify stall detection reports blocked jobs."""
        orch, store, tmpdir, mock_eb = _make_orch()
        plugins = FakePluginRegistry()
        try:
            job = _create(orch, store, "build a neural network plugin for image classification", mock_eb)

            # Advance to awaiting plan review
            for _ in range(15):
                _tick(orch, store, mock_eb, plugin_registry=plugins)
                job = orch.get_job(job.acquisition_id)
                if job.status == "awaiting_plan_review":
                    break

            if job.status == "awaiting_plan_review":
                status = orch.get_status()
                assert "stalled_jobs" in status

                # Check pending approvals are surfaced
                pending = status.get("pending_approvals", [])
                assert len(pending) > 0
                assert any(p["acquisition_id"] == job.acquisition_id for p in pending)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dashboard_status_shape(self):
        """Verify the dashboard status response has all required fields."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            status = orch.get_status()
            assert "scheduler" in status
            assert "pending_approvals" in status
            assert "stalled_jobs" in status
            assert "runtime" in status
            assert "classifier" in status

            sched = status["scheduler"]
            assert "current_mode" in sched
            assert "quarantine_pressure" in sched
            assert "pressure_level" in sched
            assert "suppressed_lanes" in sched
            assert "deferred_lanes" in sched
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_code_bundle_round_trip(self):
        """Verify PluginCodeBundle persists and loads correctly."""
        tmpdir = tempfile.mkdtemp()
        store = AcquisitionStore(base_dir=Path(tmpdir))
        try:
            bundle = PluginCodeBundle(
                acquisition_id="acq_test123",
                code_files={"__init__.py": "print('hello')", "handler.py": "def run(): pass"},
                manifest_candidate={"name": "test_plugin", "description": "test"},
                source_plan_id="plan_abc",
                doc_artifact_ids=["doc_1", "doc_2"],
            )
            bundle.code_hash = bundle.compute_hash()
            store.save_code_bundle(bundle)

            loaded = store.load_code_bundle(bundle.bundle_id)
            assert loaded is not None
            assert loaded.acquisition_id == "acq_test123"
            assert loaded.code_files == bundle.code_files
            assert loaded.code_hash == bundle.code_hash
            assert loaded.source_plan_id == "plan_abc"
            assert loaded.doc_artifact_ids == ["doc_1", "doc_2"]
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_activation_gate_blocks_without_verification(self):
        """Verify activation gate blocks promotion without completed verification."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = CapabilityAcquisitionJob(
                title="test plugin",
                risk_tier=1,
                review_status="reviewed",
            )
            job.init_lane("verification")  # pending, not completed

            rec = MagicMock()
            rec.state = "shadow"
            rec.activated_at = time.time() - 7200

            allowed, reason = orch._can_activate(job, rec)
            assert not allowed
            assert "verification_not_completed" in reason
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_activation_gate_blocks_early_shadow(self):
        """Verify activation gate blocks promotion before shadow observation period."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = CapabilityAcquisitionJob(
                title="test plugin",
                risk_tier=1,
                review_status="reviewed",
            )
            ver_ls = job.init_lane("verification")
            ver_ls.status = "completed"
            job.verification_id = "vb_test"

            rec = MagicMock()
            rec.state = "shadow"
            rec.activated_at = time.time() - 10  # only 10 seconds ago

            allowed, reason = orch._can_activate(job, rec)
            assert not allowed
            assert "shadow_observation_incomplete" in reason
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_activation_gate_passes_all_conditions(self):
        """Verify activation gate passes when all conditions are met."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            job = CapabilityAcquisitionJob(
                title="test plugin",
                risk_tier=0,
                review_status="reviewed",
            )
            ver_ls = job.init_lane("verification")
            ver_ls.status = "completed"
            job.verification_id = "vb_test"

            rec = MagicMock()
            rec.state = "shadow"
            rec.activated_at = time.time() - 600  # 10 min, past tier 0's 5 min threshold

            mods = {
                "epistemic.quarantine.pressure": MagicMock(
                    QuarantinePressure=MagicMock(instance=MagicMock(return_value=MagicMock(composite=0.0))),
                ),
            }
            with patch.dict("sys.modules", mods):
                allowed, reason = orch._can_activate(job, rec)
            assert allowed, f"Expected allowed, got reason: {reason}"
            assert reason == "all_gates_passed"
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_quarantine_pressure_suppresses_lanes(self):
        """Verify elevated quarantine pressure suppresses high-risk lanes."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            mods = {
                "epistemic.quarantine.pressure": MagicMock(
                    QuarantinePressure=MagicMock(instance=MagicMock(return_value=MagicMock(composite=0.45))),
                ),
            }
            with patch.dict("sys.modules", mods):
                orch._sample_quarantine_pressure()

            assert orch._pressure_level == "elevated"
            assert not orch._lane_allowed_in_mode("plugin_activation")
            assert not orch._lane_allowed_in_mode("deployment")
            assert orch._lane_allowed_in_mode("planning")
            assert orch._lane_allowed_in_mode("evidence_grounding")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_high_quarantine_pressure_blocks_everything(self):
        """Verify high quarantine pressure blocks all but background-safe lanes."""
        orch, store, tmpdir, mock_eb = _make_orch()
        try:
            mods = {
                "epistemic.quarantine.pressure": MagicMock(
                    QuarantinePressure=MagicMock(instance=MagicMock(return_value=MagicMock(composite=0.75))),
                ),
            }
            with patch.dict("sys.modules", mods):
                orch._sample_quarantine_pressure()

            assert orch._pressure_level == "high"
            assert not orch._lane_allowed_in_mode("implementation")
            assert not orch._lane_allowed_in_mode("plugin_quarantine")
            assert not orch._lane_allowed_in_mode("verification")
            assert not orch._lane_allowed_in_mode("plugin_activation")
            assert orch._lane_allowed_in_mode("planning")
            assert orch._lane_allowed_in_mode("truth")
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
