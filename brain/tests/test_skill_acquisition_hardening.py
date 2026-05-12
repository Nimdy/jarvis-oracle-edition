from __future__ import annotations

from types import SimpleNamespace


def test_acquisition_codegen_prompt_contains_skill_contract_fixture(tmp_path):
    from acquisition.job import AcquisitionPlan, AcquisitionStore, CapabilityAcquisitionJob, DocumentationArtifact
    from acquisition.orchestrator import AcquisitionOrchestrator

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_prompt",
        title="Build operational proof plugin for data_processing_v1",
        user_intent="Process CSV data and summarize numeric totals.",
        outcome_class="plugin_creation",
        risk_tier=1,
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "data_processing_v1",
            "learning_job_id": "job_prompt",
            "contract_id": "data_transform_v1",
        },
    )
    doc = DocumentationArtifact(
        acquisition_id=job.acquisition_id,
        source_type="repo_doc",
        topic="csv processing",
        relevance=0.9,
        citations=[{"path": "brain/skills/execution_contracts.py"}],
    )
    store.save_doc(doc)
    job.doc_artifact_ids = [doc.artifact_id]
    plan = AcquisitionPlan(
        acquisition_id=job.acquisition_id,
        objective="Build CSV totals plugin",
        technical_approach="Use csv.DictReader and numeric coercion.",
        implementation_sketch="def run(args): return {...}",
        dependencies=["csv"],
        test_cases=["csv_basic_totals"],
    )

    messages, _system, evidence = orch._build_acquisition_codegen_packet(job, plan)
    prompt = messages[0]["content"]

    assert "Jarvis Skill Acquisition Engineering Mode" in prompt
    assert "csv_basic_totals" in prompt
    assert "numeric_sums" in prompt
    assert "quantity_x_price" in prompt
    assert job.codegen_prompt_diagnostics["contract_id"] == "data_transform_v1"
    assert evidence[0]["source_type"] == "repo_doc"


def test_plugin_quarantine_refuses_missing_code_bundle(tmp_path):
    from acquisition.job import AcquisitionStore, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(acquisition_id="acq_no_bundle", title="Build missing bundle", risk_tier=1)
    job.init_lane("plugin_quarantine")

    orch._run_plugin_quarantine(job)

    assert job.lanes["plugin_quarantine"].status == "failed"
    assert "missing_code_bundle" in job.lanes["plugin_quarantine"].error


def test_skill_plugin_bundle_adds_run_adapter_for_handle_only_output(tmp_path):
    from acquisition.job import AcquisitionPlan, AcquisitionStore, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator
    from self_improve.code_patch import CodePatch, FileDiff

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_handle_only",
        title="Build operational proof plugin for data_processing_v1",
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "data_processing_v1",
            "contract_id": "data_transform_v1",
        },
    )
    plan = AcquisitionPlan(
        acquisition_id=job.acquisition_id,
        objective="Build CSV totals plugin",
    )
    patch = CodePatch(
        plan_id=plan.plan_id,
        files=[
            FileDiff(
                path="brain/tools/plugins/_gen/handler.py",
                new_content=(
                    "def handle(request):\n"
                    "    return {'row_count': 2, 'columns': ['item', 'quantity', 'price'], "
                    "'numeric_sums': {'quantity': 6, 'price': 8}, "
                    "'computed_metrics': {'quantity_x_price': 26}}\n"
                ),
            ),
            FileDiff(
                path="brain/tools/plugins/_gen/__init__.py",
                new_content=(
                    "PLUGIN_MANIFEST = {}\n\n"
                    "async def handle(text, context):\n"
                    "    from .handler import run\n"
                    "    return {'output': run({'request': text})}\n"
                ),
            ),
        ],
    )

    bundle = orch._build_code_bundle(job, plan, {"patch": patch})

    assert bundle is not None
    assert "def run(args):" in bundle.code_files["handler.py"]
    assert "return handle(payload)" in bundle.code_files["handler.py"]


def test_skill_plugin_bundle_uses_acquisition_specific_name_and_text_alias(tmp_path):
    from acquisition.job import AcquisitionPlan, AcquisitionStore, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator
    from self_improve.code_patch import CodePatch, FileDiff

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_abcdef1234",
        title="Build operational proof plugin for data_processing_v1",
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "data_processing_v1",
            "contract_id": "data_transform_v1",
        },
    )
    plan = AcquisitionPlan(acquisition_id=job.acquisition_id)
    patch = CodePatch(
        plan_id=plan.plan_id,
        files=[
            FileDiff(
                path="brain/tools/plugins/_gen/handler.py",
                new_content="def run(args):\n    return {'text': args.get('text'), 'input': args.get('input')}\n",
            ),
        ],
    )

    bundle = orch._build_code_bundle(job, plan, {"patch": patch})

    assert bundle is not None
    assert bundle.manifest_candidate["name"].endswith("_ef1234")
    assert '"text": text' in bundle.code_files["__init__.py"]
    assert '"input": text' in bundle.code_files["__init__.py"]
    assert '"request": text' in bundle.code_files["__init__.py"]


def test_skill_contract_verifier_passes_text_input_alias(tmp_path):
    from acquisition.job import AcquisitionStore, CapabilityAcquisitionJob, PluginCodeBundle, VerificationBundle
    from acquisition.orchestrator import AcquisitionOrchestrator

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_contract_text",
        requested_by={"source": "skill_operational_handoff", "skill_id": "data_processing_v1"},
    )
    code_bundle = PluginCodeBundle(
        acquisition_id=job.acquisition_id,
        code_files={
            "handler.py": (
                "def run(args):\n"
                "    text = args.get('text', '')\n"
                "    return {\n"
                "        'row_count': 2 if text else 0,\n"
                "        'columns': ['item', 'quantity', 'price'] if text else [],\n"
                "        'numeric_sums': {'quantity': 6, 'price': 8} if text else {},\n"
                "        'computed_metrics': {'quantity_x_price': 26} if text else {},\n"
                "    }\n"
            )
        },
    )
    verification = VerificationBundle(acquisition_id=job.acquisition_id)

    assert orch._run_skill_contract_on_bundle(job, verification, code_bundle) is True
    assert verification.risk_assessment["skill_contract_status"] == "passed"


def test_plugin_activation_fails_terminally_when_skill_contract_fails(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionStore, CapabilityAcquisitionJob, VerificationBundle
    from acquisition.orchestrator import AcquisitionOrchestrator
    import tools.plugin_registry as plugin_registry

    class FakeRegistry:
        def __init__(self):
            self.activate_called = False

        def get_record(self, _plugin_name):
            return SimpleNamespace(state="quarantined", activated_at=0.0)

        def activate(self, *_args, **_kwargs):
            self.activate_called = True
            return True

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    registry = FakeRegistry()
    monkeypatch.setattr(plugin_registry, "get_plugin_registry", lambda: registry)

    job = CapabilityAcquisitionJob(
        acquisition_id="acq_contract_failed",
        title="Build operational proof plugin",
        status="executing",
        outcome_class="plugin_creation",
        plugin_id="csv_totals_plugin",
        requested_by={"source": "skill_operational_handoff", "skill_id": "data_processing_v1"},
    )
    job.init_lane("verification")
    job.complete_lane("verification")
    job.init_lane("plugin_activation")
    job.start_lane("plugin_activation")

    bundle = VerificationBundle(
        acquisition_id=job.acquisition_id,
        lane_verdicts={"skill_contract_fixture": False},
        overall_passed=False,
    )
    store.save_verification(bundle)
    job.verification_id = bundle.verification_id

    orch._run_plugin_activation(job)

    assert registry.activate_called is False
    assert job.status == "failed"
    assert job.lanes["plugin_activation"].status == "failed"
    assert job.lanes["plugin_activation"].error == "verification_failed"


def test_skill_acquisition_encoder_is_pure_and_bounded():
    from acquisition.skill_acquisition_encoder import SkillAcquisitionEncoder

    job = SimpleNamespace(
        acquisition_id="acq_test",
        status="failed",
        outcome_class="plugin_creation",
        risk_tier=1,
        classification_confidence=1.0,
        doc_artifact_ids=["doc1"],
        artifact_refs=["plan1"],
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "data_processing_v1",
            "contract_id": "data_transform_v1",
        },
        lanes={"planning": SimpleNamespace(status="failed")},
        planning_diagnostics={"failure_reason": "bad", "missing_fields": ["test_cases"]},
        codegen_prompt_diagnostics={},
        verification_id="",
        completed_at=0.0,
    )

    vec = SkillAcquisitionEncoder.encode(job)

    assert len(vec) == 40
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert SkillAcquisitionEncoder.encode_label("verified") == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_synthetic_skill_acquisition_smoke_records_no_signals_by_default():
    from synthetic.skill_acquisition_exercise import PROFILES, run_skill_acquisition_exercise

    stats = run_skill_acquisition_exercise(profile=PROFILES["smoke"], seed=1)

    assert stats.passed is True
    assert stats.episodes == PROFILES["smoke"].episode_count
    assert stats.features_recorded == 0
    assert stats.labels_recorded == 0
    assert stats.invariant_failures == []


def test_codegen_rejects_none_found_structured_evidence_for_risky_work():
    from codegen.service import CodeGenService

    svc = CodeGenService()
    result = svc._check_evidence_sufficiency(
        1,
        [{"source_type": "none_found", "citations": [], "relevance": 0.0}],
    )

    assert result["sufficient"] is False
    assert "meaningful" in result["reason"]


def test_shadow_only_specialists_are_excluded_from_live_broadcast_slots():
    from hemisphere.orchestrator import _SHADOW_ONLY_TIER1_FOCUSES
    from hemisphere.types import HemisphereFocus

    assert HemisphereFocus.SKILL_ACQUISITION in _SHADOW_ONLY_TIER1_FOCUSES
    assert HemisphereFocus.PLAN_EVALUATOR in _SHADOW_ONLY_TIER1_FOCUSES
    assert HemisphereFocus.CLAIM_CLASSIFIER in _SHADOW_ONLY_TIER1_FOCUSES

