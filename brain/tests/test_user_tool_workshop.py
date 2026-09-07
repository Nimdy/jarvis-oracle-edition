"""Operator off-catalog learn-X uses the plugin workshop, not the stub verify-and-die path."""
from __future__ import annotations

import json
from types import SimpleNamespace

from skills.learning_jobs import LearningJob, LearningJobOrchestrator, LearningJobStore
from skills.registry import SkillRecord, SkillRegistry


def test_resolve_skill_stays_generic_for_auto_gate():
    from skills.resolver import is_generic_fallback_resolution, resolve_skill

    resolution = resolve_skill("roll a 20-sided dice")
    assert resolution is not None
    assert is_generic_fallback_resolution(resolution)


def test_promote_user_tool_is_acquisition_eligible():
    from skills.resolver import (
        is_generic_fallback_resolution,
        promote_user_tool_resolution,
        resolve_skill,
    )

    raw = resolve_skill("roll a 20-sided dice")
    promoted = promote_user_tool_resolution(raw, "roll a 20-sided dice")
    assert is_generic_fallback_resolution(raw)
    assert not is_generic_fallback_resolution(promoted)
    assert promoted.capability is not None
    assert promoted.capability.acquisition_eligible is True
    phases = [p["name"] for p in promoted.default_phases]
    assert phases == ["assess", "research", "integrate", "verify", "register"]


def test_user_tool_verify_waits_for_approval(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor
    from skills.resolver import promote_user_tool_resolution, resolve_skill

    monkeypatch.setenv("HOME", str(tmp_path))
    raw = resolve_skill("roll a 20-sided dice")
    promoted = promote_user_tool_resolution(raw, "roll a 20-sided dice")
    job = SimpleNamespace(
        job_id="job-d20-workshop",
        skill_id=promoted.skill_id,
        phase="verify",
        status="active",
        matrix_protocol=False,
        artifacts=[],
        events=[],
        data={},
        plan={
            "summary": promoted.notes,
            "phases": promoted.default_phases,
            "capability_contract": {
                "input_type": "operator_request",
                "output_type": "structured_tool_result",
                "execution_contract_id": promoted.capability.execution_contract_id,
                "required_executor_kind": "plugin",
                "acquisition_eligible": True,
                "smoke_fixtures": [{
                    "name": "operator_request_smoke",
                    "input_type": "operator_request",
                    "input": {"request": "roll a 20-sided dice"},
                    "expected": {"ok": True, "honors_request": True},
                }],
            },
            "operator_trigger": "roll a 20-sided dice",
        },
        gates={"hard": []},
        evidence={
            "required": list(promoted.required_evidence),
            "history": [],
            "latest": None,
        },
        failure={"count": 0, "last_error": None, "last_failed_phase": None},
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
    )

    result = ProceduralVerifyExecutor().run(job, {})

    assert result.progressed is True
    assert "awaiting_operator_approval" in result.message
    assert job.status == "awaiting_operator_approval"
    assert job.failure["count"] == 0
    assert "no_verification_method" not in (job.failure.get("last_error") or "")


def test_research_packet_has_required_and_expected(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralResearchExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = SimpleNamespace(
        job_id="job-d20-research",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        protocol_id="",
        notes="",
        artifacts=[],
        plan={
            "summary": "Auto-generated from: \"roll a 20-sided dice\". Operator-requested tool.",
            "phases": [{"name": "research", "exit_conditions": ["artifact:research_summary"]}],
            "capability_contract": {
                "acquisition_eligible": True,
                "input_type": "operator_request",
                "output_type": "structured_tool_result",
                "success_metrics": ["procedure_smoke_passed"],
                "evidence_requirements": ["active_plugin_or_tool_path"],
            },
        },
        gates={"hard": []},
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
    )
    result = ProceduralResearchExecutor().run(job, {})
    assert result.progressed is True
    assert result.artifact is not None
    import json
    blob = json.loads(open(result.artifact["path"], encoding="utf-8").read())
    assert blob["required"]["operator_approval"] is True
    assert blob["expected"]["must_not_claim_until_verified"] is True
    assert "roll a 20-sided dice" in blob["expected"]["honors_operator_request"]
    assert blob["test_cases"]
    assert blob["plugin_structure"]
    assert "secrets.randbelow" in blob["implementation_sketch"]
    assert "1..20" in blob["expected"]["rolls"]
    assert not str(blob["technical_approach"]).startswith("Governed plugin:")
    assert any("LLM must not pick" in n or "must not pick the number" in n for n in blob["design_notes"])
    assert "d20" in blob.get("accepted_utterances", [])


def test_recover_blocked_user_job_applies_workshop(tmp_path):
    store = LearningJobStore(root=str(tmp_path / "jobs"))
    registry = SkillRegistry(path=str(tmp_path / "registry.json"))
    registry._loaded = True
    orch = LearningJobOrchestrator(store=store, registry=registry)
    job = LearningJob(
        job_id="job_d20_blocked",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        status="blocked",
        phase="verify",
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
        plan={"summary": 'Auto-generated from: "roll a 20-sided dice"'},
        evidence={"required": ["test:procedure_smoke"], "latest": None, "history": []},
        failure={
            "count": 11,
            "last_error": "Verification: FAIL — no_verification_method",
            "last_failed_phase": "verify",
        },
    )
    store.save(job)
    registry.register(SkillRecord(
        skill_id=job.skill_id,
        name="Roll 20sided Dice",
        status="blocked",
        capability_type="procedural",
        learning_job_id=job.job_id,
    ))

    assert orch.recover_blocked_job(job.job_id) is True
    kept = store.load(job.job_id)
    assert kept is not None
    assert kept.status == "active"
    assert kept.phase == "assess"
    assert kept.plan["capability_contract"]["acquisition_eligible"] is True
    assert "test:sandbox_execution_pass" in kept.evidence["required"]


def test_generic_workshop_research_is_thin_and_rebuilds(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralResearchExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job_dir = tmp_path / ".jarvis" / "learning_jobs" / "job-thin"
    job_dir.mkdir(parents=True)
    path = job_dir / "research_summary.json"
    path.write_text(json.dumps({
        "required": {"operator_approval": True},
        "expected": {"must_not_claim_until_verified": True},
        "technical_approach": "Governed plugin: operator approves, CodeGen writes brain/tools/plugins/<skill>/",
        "test_cases": [{"name": "operator_request_smoke"}],
    }))
    job = SimpleNamespace(
        job_id="job-thin",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        protocol_id="",
        notes="",
        artifacts=[{"id": "research_summary", "type": "research_summary", "path": str(path)}],
        plan={
            "capability_contract": {"acquisition_eligible": True},
            "operator_trigger": "roll a 20-sided dice",
            "phases": [],
            "summary": 'Auto-generated from: "roll a 20-sided dice"',
        },
        gates={"hard": []},
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
    )
    ex = ProceduralResearchExecutor()
    assert ex._research_is_thin(job, job.artifacts[0]) is True
    result = ex.run(job, {})
    assert result.progressed is True
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert "secrets.randbelow" in blob["implementation_sketch"]
    assert blob["plugin_structure"]


def test_awaiting_approval_refreshes_thin_research(tmp_path, monkeypatch):
    store = LearningJobStore(root=str(tmp_path / "jobs"))
    registry = SkillRegistry(path=str(tmp_path / "registry.json"))
    registry._loaded = True
    orch = LearningJobOrchestrator(store=store, registry=registry)
    monkeypatch.setenv("HOME", str(tmp_path))
    job = LearningJob(
        job_id="job_refresh",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        status="awaiting_operator_approval",
        phase="verify",
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
        plan={
            "capability_contract": {"acquisition_eligible": True},
            "operator_trigger": "roll a 20-sided dice",
        },
        artifacts=[{
            "id": "research_summary",
            "type": "research_summary",
            "path": str(tmp_path / ".jarvis" / "learning_jobs" / "job_refresh" / "research_summary.json"),
        }],
    )
    art_path = tmp_path / ".jarvis" / "learning_jobs" / "job_refresh" / "research_summary.json"
    art_path.parent.mkdir(parents=True)
    art_path.write_text(json.dumps({
        "required": {"operator_approval": True},
        "expected": {"honors_operator_request": "roll a 20-sided dice"},
        "technical_approach": "Governed plugin: operator approves, CodeGen writes brain/tools/plugins/<skill>/",
    }))
    store.save(job)
    orch._active_jobs[job.job_id] = job
    orch._tick_job(job, {})
    blob = json.loads(art_path.read_text(encoding="utf-8"))
    assert "secrets.randbelow" in blob["implementation_sketch"]
    assert job.status == "awaiting_operator_approval"


def test_default_skills_cannot_be_removed(tmp_path):
    from skills.registry import SkillRecord, SkillRegistry, get_default_skill_ids

    sid = sorted(get_default_skill_ids())[0]
    registry = SkillRegistry(path=str(tmp_path / "registry.json"))
    registry._loaded = True
    registry.register(SkillRecord(skill_id=sid, name="Baseline"))
    assert registry.remove(sid) is False
    assert registry.get(sid) is not None
    registry.register(SkillRecord(skill_id="roll_20sided_dice_v1", name="Dice"))
    assert registry.remove("roll_20sided_dice_v1") is True
    assert registry.get("roll_20sided_dice_v1") is None


def test_research_packet_accepts_d20_and_spaced_die(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralResearchExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = SimpleNamespace(
        job_id="job-d20-aliases",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        protocol_id="",
        notes="",
        artifacts=[],
        plan={
            "capability_contract": {"acquisition_eligible": True},
            "operator_trigger": "roll a 20-sided dice",
            "phases": [],
            "summary": 'Auto-generated from: "roll a 20-sided dice"',
        },
        gates={"hard": []},
        requested_by={"source": "user", "user_text": "roll a 20-sided dice", "speaker": "David"},
    )
    blob = json.loads(open(ProceduralResearchExecutor().run(job, {}).artifact["path"], encoding="utf-8").read())
    assert "d20" in blob["accepted_utterances"]
    assert "roll a 20 sided die" in blob["accepted_utterances"]
    assert "def handle" in blob["implementation_sketch"]
    assert "secrets.randbelow" in blob["implementation_sketch"]


def test_planning_applies_learning_research_contract(tmp_path, monkeypatch):
    from acquisition.job import AcquisitionPlan, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator

    monkeypatch.setenv("HOME", str(tmp_path))
    research_path = tmp_path / ".jarvis" / "learning_jobs" / "job_research" / "research_summary.json"
    research_path.parent.mkdir(parents=True)
    research_path.write_text(json.dumps({
        "approach": "Local 20-sided die plugin.",
        "technical_approach": "Implement brain/tools/plugins/roll_20sided_dice_v1/roll.py with secrets.randbelow.",
        "implementation_sketch": "import secrets\ndef roll(sides=20, n=1):\n    return {'ok': True}\n",
        "plugin_structure": ["brain/tools/plugins/roll_20sided_dice_v1/roll.py"],
        "test_cases": [{"name": "range_and_type", "expected": {"min": 1, "max": 20}}],
        "accepted_utterances": ["d20", "roll a 20 sided die"],
    }))
    orch = AcquisitionOrchestrator.__new__(AcquisitionOrchestrator)
    job = CapabilityAcquisitionJob(title="t", user_intent="plugin")
    job.learning_job_id = "job_research"
    plan = AcquisitionPlan(
        acquisition_id=job.acquisition_id,
        objective="competing",
        technical_approach="keyword matching with random.randint",
        implementation_sketch="def handle(request): return random.randint(1,20)",
        risk_level="high",
    )
    assert orch._apply_learning_research_contract(job, plan) is True
    assert "import secrets" in plan.implementation_sketch
    assert "random.randint" not in plan.implementation_sketch
    assert "roll_20sided_dice_v1/roll.py" in plan.technical_approach
    assert plan.risk_level == "low"
    assert any("d20" in t for t in plan.test_cases)


def test_workshop_contract_counts_as_t2_codegen_evidence(tmp_path):
    """Off-catalog learn-X has no library docs. The workshop contract is the evidence."""
    from acquisition.job import (
        AcquisitionPlan,
        AcquisitionStore,
        CapabilityAcquisitionJob,
        DocumentationArtifact,
    )
    from acquisition.orchestrator import AcquisitionOrchestrator
    from codegen.service import CodeGenService

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_d20_ev",
        title="Build operational proof plugin for roll_20sided_dice_v1",
        user_intent="roll a 20-sided dice",
        outcome_class="plugin_creation",
        risk_tier=2,
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "roll_20sided_dice_v1",
            "learning_job_id": "job_377e",
            "contract_id": "roll_20sided_dice_v1_plugin",
            "required_executor_kind": "plugin",
            "operator_trigger": "roll a 20-sided dice",
            "smoke_fixtures": [{
                "name": "operator_request_smoke",
                "input_type": "operator_request",
                "input": {"request": "roll a 20-sided dice"},
                "expected": {"ok": True, "honors_request": True},
            }],
        },
    )
    job.learning_job_id = "job_377e"
    doc = DocumentationArtifact(
        acquisition_id=job.acquisition_id,
        source_type="none_found",
        topic="Build operational proof plugin for roll_20sided_dice_v1",
        relevance=0.0,
        citations=[],
    )
    store.save_doc(doc)
    job.doc_artifact_ids = [doc.artifact_id]
    plan = AcquisitionPlan(
        acquisition_id=job.acquisition_id,
        objective="d20 plugin",
        technical_approach="secrets.randbelow",
        implementation_sketch="import secrets\ndef roll(sides=20): return {'ok': True}",
        risk_level="low",
    )

    messages, _system, evidence = orch._build_acquisition_codegen_packet(job, plan)
    assert job.codegen_prompt_diagnostics["contract_id"] == "roll_20sided_dice_v1_plugin"
    assert any(row.get("source_type") == "skill_contract" for row in evidence)
    check = CodeGenService()._check_evidence_sufficiency(2, evidence)
    assert check["sufficient"] is True
    prompt = messages[0]["content"]
    assert "operator_request_smoke" in prompt
    assert "roll_20sided_dice_v1_plugin" in prompt


def test_workshop_contract_loads_from_learning_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store = LearningJobStore(root=str(tmp_path / ".jarvis" / "learning_jobs"))
    lj = LearningJob(
        job_id="job_from_disk",
        skill_id="roll_20sided_dice_v1",
        capability_type="procedural",
        plan={
            "capability_contract": {
                "acquisition_eligible": True,
                "execution_contract_id": "roll_20sided_dice_v1_plugin",
                "required_executor_kind": "plugin",
                "smoke_fixtures": [{
                    "name": "operator_request_smoke",
                    "input_type": "operator_request",
                    "input": {"request": "roll a 20-sided dice"},
                    "expected": {"ok": True, "honors_request": True},
                }],
            },
            "operator_trigger": "roll a 20-sided dice",
        },
    )
    store.save(lj)

    from acquisition.job import AcquisitionStore, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator

    orch = AcquisitionOrchestrator(AcquisitionStore(tmp_path / "acq"))
    job = CapabilityAcquisitionJob(
        requested_by={
            "skill_id": "roll_20sided_dice_v1",
            "learning_job_id": "job_from_disk",
        },
    )
    job.learning_job_id = "job_from_disk"
    contract = orch._resolve_execution_contract(job)
    assert contract is not None
    assert contract.contract_id == "roll_20sided_dice_v1_plugin"
    assert contract.smoke_fixtures[0].name == "operator_request_smoke"


def test_workshop_roll_module_promotes_to_handler_and_passes_contract(tmp_path):
    """Research-shaped roll.py + run() must satisfy skill_contract_fixture."""
    from acquisition.job import (
        AcquisitionPlan,
        AcquisitionStore,
        CapabilityAcquisitionJob,
        VerificationBundle,
    )
    from acquisition.orchestrator import AcquisitionOrchestrator
    from self_improve.code_patch import CodePatch, FileDiff

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_roll_promote",
        title="Build operational proof plugin for roll_20sided_dice_v1",
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "roll_20sided_dice_v1",
            "contract_id": "roll_20sided_dice_v1_plugin",
            "required_executor_kind": "plugin",
            "operator_trigger": "roll a 20-sided dice",
            "smoke_fixtures": [{
                "name": "operator_request_smoke",
                "input_type": "operator_request",
                "input": {"request": "roll a 20-sided dice"},
                "expected": {"ok": True, "honors_request": True},
            }],
        },
    )
    plan = AcquisitionPlan(acquisition_id=job.acquisition_id, objective="d20")
    patch = CodePatch(
        plan_id=plan.plan_id,
        files=[
            FileDiff(
                path="brain/tools/plugins/roll_20sided_dice_v1/roll.py",
                new_content=(
                    "def roll(sides=20, n=1):\n"
                    "    return {'ok': True, 'rolls': [7], 'sides': sides, 'n': n}\n"
                    "def run(args):\n"
                    "    req = args.get('text') or args.get('input') or args.get('request') or ''\n"
                    "    if not isinstance(req, str):\n"
                    "        return {'ok': False, 'error': 'Input must be a string'}\n"
                    "    out = roll()\n"
                    "    out['honors_request'] = bool(req)\n"
                    "    return out\n"
                ),
            ),
            FileDiff(
                path="brain/tools/plugins/roll_20sided_dice_v1/__init__.py",
                new_content="from .roll import roll\n__all__ = ['roll']\n",
            ),
        ],
    )
    bundle = orch._build_code_bundle(job, plan, {"patch": patch})
    assert bundle is not None
    assert "def run(" in bundle.code_files["handler.py"]
    assert "async def handle(" in bundle.code_files["__init__.py"]
    verification = VerificationBundle(acquisition_id=job.acquisition_id)
    assert orch._run_skill_contract_on_bundle(job, verification, bundle) is True
    assert verification.risk_assessment["skill_contract_status"] == "passed"


def test_fixture_user_text_unwraps_operator_request_dict():
    from types import SimpleNamespace
    from acquisition.orchestrator import AcquisitionOrchestrator

    fixture = SimpleNamespace(input={"request": "roll a 20-sided dice"})
    assert AcquisitionOrchestrator._fixture_user_text(fixture) == "roll a 20-sided dice"
    assert AcquisitionOrchestrator._fixture_user_text(SimpleNamespace(input="csv,text")) == "csv,text"


def test_plugin_spoken_reply_says_dice_result():
    from conversation_handler import _plugin_spoken_reply

    nested = {
        "output": {
            "ok": True,
            "rolls": [6],
            "sides": 20,
            "n": 1,
            "honors_request": True,
        }
    }
    assert _plugin_spoken_reply(nested) == "I rolled the Dice, you got 6"
    assert _plugin_spoken_reply({"rolls": [6, 12]}) == "I rolled the Dice, you got 6, 12"
    assert _plugin_spoken_reply({"output": "I rolled the Dice, you got 3"}) == "I rolled the Dice, you got 3"


def test_improvement_job_still_promotes_roll_module_to_handler(tmp_path):
    from acquisition.job import AcquisitionPlan, AcquisitionStore, CapabilityAcquisitionJob
    from acquisition.orchestrator import AcquisitionOrchestrator
    from self_improve.code_patch import CodePatch, FileDiff

    store = AcquisitionStore(tmp_path)
    orch = AcquisitionOrchestrator(store)
    job = CapabilityAcquisitionJob(
        acquisition_id="acq_improve1",
        title="[improve gen-1] d20",
        requested_by={
            "source": "operator_improvement",
            "skill_id": "roll_20sided_dice_v1",
            "contract_id": "roll_20sided_dice_v1_plugin",
        },
    )
    plan = AcquisitionPlan(acquisition_id=job.acquisition_id)
    patch = CodePatch(
        plan_id=plan.plan_id,
        files=[
            FileDiff(
                path="brain/tools/plugins/x/roll.py",
                new_content="def run(args):\n    return {'ok': True, 'rolls': [1]}\n",
            ),
            FileDiff(
                path="brain/tools/plugins/x/__init__.py",
                new_content="from .roll import roll\n",
            ),
        ],
    )
    bundle = orch._build_code_bundle(job, plan, {"patch": patch})
    assert bundle is not None
    assert "def run(" in bundle.code_files["handler.py"]
    assert bundle.manifest_candidate["name"].endswith("improve1"[-6:])
