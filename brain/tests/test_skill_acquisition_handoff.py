from __future__ import annotations

from types import SimpleNamespace


def _data_job(tmp_path):
    return SimpleNamespace(
        job_id="job-proof-bridge",
        skill_id="data_processing_v1",
        phase="verify",
        matrix_protocol=False,
        artifacts=[],
        plan={"summary": "Needs operational contract proof."},
        gates={"hard": []},
        evidence={
            "required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
            "history": [],
            "latest": None,
        },
        failure={"count": 0, "last_error": None, "last_failed_phase": None},
        data={},
        events=[],
    )


class _FakeAcquisitionOrchestrator:
    def __init__(self, acq_job=None, verification=None):
        self.created = []
        self._job = acq_job
        self._store = SimpleNamespace(
            list_jobs=lambda: [self._job] if self._job is not None else [],
            load_verification=lambda _vid: verification,
        )

    def create_skill_proof_handoff(self, learning_job, contract, handoff):
        job = SimpleNamespace(
            acquisition_id="acq-proof-1",
            requested_by={
                "source": "skill_operational_handoff",
                "skill_id": learning_job.skill_id,
                "learning_job_id": learning_job.job_id,
                "contract_id": contract.contract_id,
            },
            plugin_id="",
            verification_id="",
            status="planning",
        )
        self.created.append((learning_job, contract, handoff))
        self._job = job
        self._store.list_jobs = lambda: [job]
        return job

    def get_job(self, acquisition_id):
        if self._job and self._job.acquisition_id == acquisition_id:
            return self._job
        return None


def _patch_plugin_registry(monkeypatch, state="supervised", result=None):
    import tools.plugin_registry as plugin_registry_mod

    rec = SimpleNamespace(
        name="data_transform_plugin",
        state=state,
        supervision_mode=state,
        risk_tier=1,
        execution_mode="in_process",
        code_hash="abc123",
    )

    class _FakeRegistry:
        def get_record(self, name):
            return rec if name == "data_transform_plugin" else None

        async def invoke(self, request):
            return SimpleNamespace(
                success=True,
                result=result or {
                    "output": {
                        "row_count": 2,
                        "columns": ["item", "quantity", "price"],
                        "numeric_sums": {"quantity": 6, "price": 8},
                        "computed_metrics": {"quantity_x_price": 26},
                    }
                },
                error="",
            )

    fake_registry = _FakeRegistry()
    monkeypatch.setattr(plugin_registry_mod, "get_plugin_registry", lambda: fake_registry)
    return fake_registry


def test_missing_callable_creates_operator_approval_request(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = _data_job(tmp_path)

    result = ProceduralVerifyExecutor().run(job, {})

    assert result.progressed is True
    assert result.evidence["result"] == "fail"
    assert "awaiting_operator_approval" in result.message
    assert getattr(job, "status") == "awaiting_operator_approval"
    assert job.data["operational_handoff"]["status"] == "awaiting_operator_approval"
    assert job.data["operational_handoff"]["approval_required"] is True
    assert not job.data["operational_handoff"]["acquisition_id"]
    assert any(a["type"] == "operational_handoff_required" for a in job.artifacts)


def test_learning_job_artifacts_upsert_same_current_file(tmp_path, monkeypatch):
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    orch = LearningJobOrchestrator(store)
    job = LearningJob(
        job_id="job-artifact-dedupe",
        skill_id="data_processing_v1",
        capability_type="procedural",
    )
    artifact = {
        "id": "contract_smoke_result",
        "type": "contract_smoke_result",
        "path": str(tmp_path / "learning_jobs" / "job-artifact-dedupe" / "smoke_result.json"),
        "details": {"passed": False},
    }

    orch.add_artifact(job, artifact)
    orch.add_artifact(job, {**artifact, "details": {"passed": True}})

    assert len(job.artifacts) == 1
    assert job.artifacts[0]["details"]["passed"] is True
    assert [e["type"] for e in job.events] == ["artifact_added", "artifact_updated"]


def test_normal_skill_completion_uses_skill_learning_report(tmp_path, monkeypatch):
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    orch = LearningJobOrchestrator(store)
    job = LearningJob(
        job_id="job-skill-report",
        skill_id="data_processing_v1",
        capability_type="procedural",
        matrix_protocol=False,
        evidence={
            "required": ["test:data_processing_smoke"],
            "latest": {"tests": [{"name": "data_processing_smoke", "passed": True}]},
            "history": [],
        },
    )

    orch.complete_job(job)

    completed_event = next(e for e in job.events if e["type"] == "job_completed")
    assert completed_event["msg"].startswith("Skill learning report for Data Processing")
    assert "Matrix Protocol" not in completed_event["msg"]
    assert any(a["type"] == "skill_learning_report" for a in job.artifacts)
    assert (tmp_path / ".jarvis" / "learning_jobs" / "job-skill-report" / "skill_learning_report.json").exists()


def test_matrix_skill_completion_keeps_matrix_report(tmp_path, monkeypatch):
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    orch = LearningJobOrchestrator(store)
    job = LearningJob(
        job_id="job-matrix-report",
        skill_id="scene_understanding_v1",
        capability_type="perceptual",
        matrix_protocol=True,
        protocol_id="SK-002",
        evidence={
            "required": ["test:scene_smoke"],
            "latest": {"tests": [{"name": "scene_smoke", "passed": True}]},
            "history": [],
        },
    )

    orch.complete_job(job)

    completed_event = next(e for e in job.events if e["type"] == "job_completed")
    assert completed_event["msg"].startswith("Matrix Protocol learning report for Scene Understanding")
    assert any(a["type"] == "matrix_report" for a in job.artifacts)
    assert (tmp_path / ".jarvis" / "learning_jobs" / "job-matrix-report" / "matrix_report.json").exists()


def test_quarantined_plugin_is_not_exposed_as_skill_callable(monkeypatch):
    from skills.operational_bridge import build_skill_execution_callables

    acq_job = SimpleNamespace(
        requested_by={"skill_id": "data_processing_v1"},
        plugin_id="data_transform_plugin",
        verification_id="ver-proof-1",
    )
    acq = _FakeAcquisitionOrchestrator(
        acq_job=acq_job,
        verification=SimpleNamespace(overall_passed=True),
    )
    _patch_plugin_registry(monkeypatch, state="quarantined")

    callables = build_skill_execution_callables(acq)

    assert "data_processing_v1" not in callables


def test_operator_approval_creates_real_acquisition_and_links(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionStore
    from acquisition.orchestrator import AcquisitionOrchestrator
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-real-approval",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]

    result = ProceduralVerifyExecutor().run(job, {})
    assert result.progressed is True
    assert job.status == "awaiting_operator_approval"

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)
    acq = AcquisitionOrchestrator(AcquisitionStore(tmp_path / "acquisition"))

    approval = learning.approve_operational_handoff(
        job.job_id,
        acq,
        approved_by="test",
        notes="approved for real integration test",
    )

    assert approval["ok"] is True
    acquisition_id = approval["acquisition_id"]
    assert acquisition_id
    linked = store.load(job.job_id)
    assert linked is not None
    assert linked.status == "active"
    assert linked.parent_acquisition_id == acquisition_id
    assert linked.data["operational_handoff"]["status"] == "awaiting_acquisition_proof"
    acq_job = acq.get_job(acquisition_id)
    assert acq_job is not None
    assert acq_job.outcome_class == "plugin_creation"
    assert acq_job.risk_tier == 2
    assert acq_job.requested_by["source"] == "skill_operational_handoff"


def test_post_approval_verification_preserves_linked_handoff(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionStore
    from acquisition.orchestrator import AcquisitionOrchestrator
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-preserve-linked",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    ProceduralVerifyExecutor().run(job, {})

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)
    acq = AcquisitionOrchestrator(AcquisitionStore(tmp_path / "acquisition"))

    approval = learning.approve_operational_handoff(
        job.job_id,
        acq,
        approved_by="test",
        notes="approved for proof build",
    )
    assert approval["ok"] is True

    linked = store.load(job.job_id)
    assert linked is not None
    result = ProceduralVerifyExecutor().run(linked, {"acquisition_orchestrator": acq})

    assert result.progressed is True
    assert "waiting_for_acquisition_proof" in result.message
    assert linked.data["operational_handoff"]["status"] == "awaiting_acquisition_proof"
    assert linked.data["operational_handoff"]["acquisition_id"] == approval["acquisition_id"]
    assert linked.status == "active"


def test_failed_acquisition_closes_handoff_instead_of_waiting(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    failed_acq_job = SimpleNamespace(
        acquisition_id="acq-failed-1",
        requested_by={"skill_id": "data_processing_v1"},
        plugin_id="",
        verification_id="",
        status="failed",
        lanes={
            "planning": SimpleNamespace(
                status="failed",
                error="planning_failed_empty_coder_response",
            ),
        },
    )
    acq = _FakeAcquisitionOrchestrator(acq_job=failed_acq_job)
    job = _data_job(tmp_path)
    job.parent_acquisition_id = "acq-failed-1"
    job.status = "active"
    job.data["operational_handoff"] = {
        "status": "awaiting_acquisition_proof",
        "acquisition_id": "acq-failed-1",
    }

    result = ProceduralVerifyExecutor().run(job, {"acquisition_orchestrator": acq})

    assert result.progressed is False
    assert result.evidence["result"] == "fail"
    assert "acquisition_failed" in result.message
    assert job.status == "blocked"
    handoff = job.data["operational_handoff"]
    assert handoff["status"] == "acquisition_failed"
    assert handoff["terminal_lane"] == "planning"
    assert handoff["terminal_error"] == "planning_failed_empty_coder_response"
    assert job.failure["last_error"] == "acquisition_failed:planning_failed_empty_coder_response"


def test_learning_cycle_syncs_terminal_acquisition_before_cooldown(tmp_path, monkeypatch):
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    failed_acq_job = SimpleNamespace(
        acquisition_id="acq-failed-before-cooldown",
        status="failed",
        lanes={
            "plugin_activation": SimpleNamespace(
                status="failed",
                error="verification_failed",
            ),
        },
    )
    acq = _FakeAcquisitionOrchestrator(acq_job=failed_acq_job)
    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    job = LearningJob(
        job_id="job-terminal-sync",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
        status="active",
    )
    job.parent_acquisition_id = failed_acq_job.acquisition_id
    job.executor_state["_last_tick_ts"] = 9999999999.0
    job.data["operational_handoff"] = {
        "status": "awaiting_acquisition_proof",
        "acquisition_id": failed_acq_job.acquisition_id,
    }
    store.save(job)

    class _Registry:
        def __init__(self):
            self.status = None

        def set_status(self, skill_id, status):
            self.status = (skill_id, status)

    registry = _Registry()
    learning = LearningJobOrchestrator(store, registry=registry)
    learning.set_context_provider("acquisition_orchestrator", acq)

    learning.run_cycle()

    saved = store.load(job.job_id)
    assert saved is not None
    assert saved.status == "blocked"
    assert saved.data["operational_handoff"]["status"] == "acquisition_failed"
    assert saved.data["operational_handoff"]["terminal_lane"] == "plugin_activation"
    assert saved.failure["last_error"] == "acquisition_failed:verification_failed"
    assert registry.status == ("data_processing_v1", "blocked")
    assert learning.get_status()["active_count"] == 0


def test_retry_operational_handoff_creates_new_acquisition(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionStore
    from acquisition.orchestrator import AcquisitionOrchestrator
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-retry-handoff",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
        status="blocked",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    job.parent_acquisition_id = "acq-old-failed"
    job.data["operational_handoff"] = {
        "status": "acquisition_failed",
        "acquisition_id": "acq-old-failed",
        "terminal_lane": "planning",
        "terminal_error": "planning_failed_empty_coder_response",
    }

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)
    acq = AcquisitionOrchestrator(AcquisitionStore(tmp_path / "acquisition"))

    retry = learning.retry_operational_handoff(
        job.job_id,
        acq,
        approved_by="test",
        notes="try again",
    )

    assert retry["ok"] is True
    assert retry["acquisition_id"]
    assert retry["acquisition_id"] != "acq-old-failed"
    saved = store.load(job.job_id)
    assert saved is not None
    assert saved.status == "active"
    assert saved.parent_acquisition_id == retry["acquisition_id"]
    handoff = saved.data["operational_handoff"]
    assert handoff["status"] == "awaiting_acquisition_proof"
    assert "acq-old-failed" in handoff["previous_acquisition_ids"]


def test_dispatcher_preserves_terminal_acquisition_failure_reason(tmp_path, monkeypatch):
    from skills.executors.base import PhaseExecutor, PhaseResult
    from skills.executors.dispatcher import ExecutorDispatcher
    from skills.learning_jobs import LearningJob, LearningJobStore

    class _TerminalFailureExecutor(PhaseExecutor):
        phase = "verify"

        def run(self, job, ctx):
            job.status = "blocked"
            job.failure = {
                "count": 0,
                "last_error": "acquisition_failed:planning_failed_empty_coder_response",
                "last_failed_phase": "verify",
            }
            job.data["operational_handoff"] = {
                "status": "acquisition_failed",
                "terminal_error": "planning_failed_empty_coder_response",
            }
            return PhaseResult(
                progressed=False,
                message="Verification: FAIL — acquisition_failed",
            )

    class _JobOrch:
        def __init__(self, store):
            self.store = store

        def set_gate(self, *_args, **_kwargs):
            raise AssertionError("not used")

        def add_artifact(self, *_args, **_kwargs):
            raise AssertionError("not used")

        def record_evidence(self, *_args, **_kwargs):
            raise AssertionError("not used")

    monkeypatch.setenv("HOME", str(tmp_path))
    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    job = LearningJob(
        job_id="job-dispatch-terminal",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
        status="active",
    )
    job.data["operational_handoff"] = {}

    ExecutorDispatcher([_TerminalFailureExecutor()]).tick_one_job(
        job,
        {},
        _JobOrch(store),
        object(),
    )

    assert job.failure["last_error"] == "acquisition_failed:planning_failed_empty_coder_response"


def test_operator_approval_waiting_state_restores_on_boot(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-waiting-restore",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    ProceduralVerifyExecutor().run(job, {})

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)

    restored = LearningJobOrchestrator(store)

    active_ids = {j.job_id for j in restored.get_active_jobs()}
    assert job.job_id in active_ids
    assert store.load(job.job_id).status == "awaiting_operator_approval"


def test_operator_rejection_records_durable_unverified_reason(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-reject-approval",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    ProceduralVerifyExecutor().run(job, {})

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)

    rejection = learning.reject_operational_handoff(
        job.job_id,
        rejected_by="test",
        reason="not needed for this validation run",
    )

    assert rejection["ok"] is True
    saved = store.load(job.job_id)
    assert saved is not None
    assert saved.status == "blocked"
    assert saved.failure["last_error"] == "operator_rejected_operational_build"
    assert saved.data["operational_handoff"]["status"] == "operator_rejected_operational_build"
    assert saved.data["operational_handoff"]["rejection_reason"] == "not needed for this validation run"


def test_operator_rejection_blocked_after_acquisition_link(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionStore
    from acquisition.orchestrator import AcquisitionOrchestrator
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-reject-linked",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    ProceduralVerifyExecutor().run(job, {})

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)
    acq = AcquisitionOrchestrator(AcquisitionStore(tmp_path / "acquisition"))
    approval = learning.approve_operational_handoff(job.job_id, acq, approved_by="test")
    assert approval["ok"] is True

    rejection = learning.reject_operational_handoff(
        job.job_id,
        rejected_by="test",
        reason="operator clicked the wrong rejection path",
    )

    assert rejection["ok"] is False
    assert rejection["reason"] == "handoff_already_linked_to_acquisition"
    saved = store.load(job.job_id)
    assert saved is not None
    assert saved.status == "active"
    assert saved.data["operational_handoff"]["status"] == "awaiting_acquisition_proof"


def test_acquisition_creation_failure_is_persisted(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore

    class _BrokenAcquisition:
        def create_skill_proof_handoff(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job-failed-approval",
        skill_id="data_processing_v1",
        capability_type="procedural",
        phase="verify",
    )
    job.evidence["required"] = ["test:data_processing_smoke", "test:sandbox_execution_pass"]
    ProceduralVerifyExecutor().run(job, {})

    store = LearningJobStore(str(tmp_path / "learning_jobs"))
    store.save(job)
    learning = LearningJobOrchestrator(store)

    approval = learning.approve_operational_handoff(
        job.job_id,
        _BrokenAcquisition(),
        approved_by="test",
        notes="should fail visibly",
    )

    assert approval["ok"] is False
    saved = store.load(job.job_id)
    assert saved is not None
    assert saved.data["operational_handoff"]["status"] == "approval_failed"
    assert saved.data["operational_handoff"]["last_error_type"] == "RuntimeError"


def test_active_plugin_plus_sandbox_proof_satisfies_contract(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.operational_bridge import build_skill_execution_callables

    monkeypatch.setenv("HOME", str(tmp_path))
    acq_job = SimpleNamespace(
        acquisition_id="acq-proof-2",
        requested_by={"skill_id": "data_processing_v1"},
        plugin_id="data_transform_plugin",
        verification_id="ver-proof-2",
    )
    acq = _FakeAcquisitionOrchestrator(
        acq_job=acq_job,
        verification=SimpleNamespace(
            overall_passed=True,
            sandbox_result_ref="sandbox://proof",
        ),
    )
    _patch_plugin_registry(monkeypatch, state="active")
    callables = build_skill_execution_callables(acq)

    job = _data_job(tmp_path)
    job.parent_acquisition_id = "acq-proof-2"

    result = ProceduralVerifyExecutor().run(
        job,
        {
            "acquisition_orchestrator": acq,
            "skill_execution_callables": callables,
        },
    )

    assert result.progressed is True
    assert result.evidence["result"] == "pass"
    tests = {t["name"]: t for t in result.evidence["tests"]}
    assert tests["test:data_processing_smoke"]["passed"] is True
    assert tests["test:sandbox_execution_pass"]["passed"] is True
    assert any(a["type"] == "operational_callable_path" for a in job.artifacts)
    assert any(a["type"] == "sandbox_execution_pass" for a in job.artifacts)
