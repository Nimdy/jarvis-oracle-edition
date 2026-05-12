from __future__ import annotations

from types import SimpleNamespace


def _data_job(tmp_path, artifacts=None):
    return SimpleNamespace(
        job_id="job-contract-1",
        skill_id="data_processing_v1",
        phase="verify",
        matrix_protocol=False,
        artifacts=list(artifacts or []),
        plan={"summary": "Acquisition-eligible: may produce a plugin for data transformation."},
        gates={"hard": []},
        evidence={
            "required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
            "history": [],
            "latest": None,
        },
        failure={"count": 0, "last_error": None, "last_failed_phase": None},
    )


def _csv_processor(csv_text: str):
    import csv
    from io import StringIO

    rows = list(csv.DictReader(StringIO(csv_text)))
    columns = list(rows[0].keys()) if rows else []
    quantity_sum = sum(int(row["quantity"]) for row in rows)
    price_sum = sum(int(row["price"]) for row in rows)
    return {
        "row_count": len(rows),
        "columns": columns,
        "numeric_sums": {"quantity": quantity_sum, "price": price_sum},
        "computed_metrics": {
            "quantity_x_price": sum(int(row["quantity"]) * int(row["price"]) for row in rows),
        },
    }


def test_contract_lookup_for_data_processing():
    from skills.execution_contracts import get_contract

    contract = get_contract("data_processing_v1")

    assert contract is not None
    assert contract.family == "data_transform"
    assert contract.smoke_test_name == "test:data_processing_smoke"
    assert contract.requires_sandbox is True
    assert contract.acquisition_eligible is True


def test_data_contract_requires_production_callable_path(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = _data_job(tmp_path, artifacts=[
        {"id": "integration_test_passed", "type": "integration_test_passed"},
    ])

    result = ProceduralVerifyExecutor().run(job, {})

    assert result.progressed is True
    assert "awaiting_operator_approval" in result.message
    assert job.status == "awaiting_operator_approval"
    assert job.failure["count"] == 0
    assert job.data["operational_handoff"]["status"] == "awaiting_operator_approval"
    tests = {t["name"]: t for t in result.evidence["tests"]}
    assert tests["test:data_processing_smoke"]["passed"] is False
    assert tests["test:sandbox_execution_pass"]["passed"] is False
    assert tests["test:sandbox_execution_pass"]["details"] == "missing_sandbox_execution_artifact"


def test_data_contract_passes_only_with_callable_and_sandbox_artifact(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = _data_job(tmp_path, artifacts=[
        {"id": "sandbox_execution_pass", "type": "sandbox_execution_pass", "path": str(tmp_path / "sandbox.json")},
    ])

    result = ProceduralVerifyExecutor().run(
        job,
        {"skill_execution_callables": {"data_processing_v1": _csv_processor}},
    )

    assert result.progressed is True
    tests = {t["name"]: t for t in result.evidence["tests"]}
    assert tests["test:data_processing_smoke"]["passed"] is True
    assert tests["test:sandbox_execution_pass"]["passed"] is True
    assert result.artifact is not None
    assert result.artifact["type"] == "contract_smoke_result"


def test_integration_artifact_cannot_satisfy_named_contract_evidence(tmp_path, monkeypatch):
    from skills.executors.procedural import ProceduralVerifyExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    job = _data_job(tmp_path, artifacts=[
        {"id": "model_or_method_available", "type": "model_or_method_available"},
        {"id": "integration_test_passed", "type": "integration_test_passed"},
    ])

    result = ProceduralVerifyExecutor().run(job, {})

    assert result.progressed is True
    assert "awaiting_operator_approval" in result.message
    assert result.evidence["result"] == "fail"
    assert not any(t["passed"] for t in result.evidence["tests"])


def test_resolver_marks_acquisition_eligible_contract_metadata():
    from skills.resolver import resolve_skill

    resolution = resolve_skill("learn to process csv data")

    assert resolution is not None
    assert resolution.skill_id == "data_processing_v1"
    assert resolution.capability is not None
    assert resolution.capability.acquisition_eligible is True
    assert resolution.capability.execution_contract_id == "data_transform_v1"
    assert resolution.capability.required_executor_kind == "callable_or_plugin"


def test_matrix_procedural_register_requires_operational_proof():
    from skills.executors.procedural import ProceduralRegisterExecutor

    registry = SimpleNamespace(
        get=lambda skill_id: SimpleNamespace(status="learning"),
        set_status=lambda *args, **kwargs: True,
    )
    job = SimpleNamespace(
        job_id="job-matrix-1",
        skill_id="data_processing_v1",
        capability_type="procedural",
        matrix_protocol=True,
        artifacts=[],
        gates={"hard": []},
        evidence={
            "required": ["test:procedure_smoke"],
            "history": [{
                "evidence_id": "verify_matrix",
                "ts": "2026-01-01T00:00:00Z",
                "result": "pass",
                "tests": [{"name": "test:procedure_smoke", "passed": True}],
            }],
        },
    )

    result = ProceduralRegisterExecutor().run(job, {"registry": registry})

    assert result.progressed is False
    assert "Matrix procedural evidence is advisory" in result.message
