from __future__ import annotations

import json
from types import SimpleNamespace


class _FakeRecord:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


class _FakeRegistry:
    def __init__(self, record):
        self.record = record

    def get(self, skill_id):
        if self.record.payload.get("skill_id") == skill_id:
            return self.record
        return None


class _FakeJob:
    def __init__(self, payload):
        self.__dict__.update(payload)

    def to_dict(self):
        return dict(self.__dict__)


def _fake_orchestrator(jobs):
    return SimpleNamespace(store=SimpleNamespace(load_all=lambda: jobs))


def test_skill_audit_packet_shows_current_missing_operational_proof(tmp_path):
    from skills.audit_trail import build_skill_audit_packet

    smoke_path = tmp_path / "smoke_result.json"
    smoke_path.write_text(json.dumps({
        "contract_id": "data_transform_v1",
        "results": [
            {
                "name": "test:data_processing_smoke",
                "passed": False,
                "details": "no_operational_callable_path",
                "expected": {"required_executor_kind": "callable_or_plugin"},
                "actual": None,
            },
            {
                "name": "test:sandbox_execution_pass",
                "passed": False,
                "details": "missing_sandbox_execution_artifact",
                "expected": {"sandbox_execution_pass": True},
                "actual": {"sandbox_execution_pass": False},
            },
        ],
    }))

    current_job = _FakeJob({
        "job_id": "job_current",
        "skill_id": "data_processing_v1",
        "status": "active",
        "phase": "verify",
        "created_at": "2026-05-04T19:27:29Z",
        "updated_at": "2026-05-04T19:35:00Z",
        "requested_by": {
            "source": "user",
            "speaker": "David",
            "user_text": "Learn data processing as a procedural skill",
        },
        "risk_level": "low",
        "matrix_protocol": False,
        "protocol_id": "",
        "plan": {
            "summary": "Acquisition-eligible: may produce a plugin for data transformation.",
            "capability_contract": {
                "execution_contract_id": "data_transform_v1",
                "required_executor_kind": "callable_or_plugin",
                "acquisition_eligible": True,
            },
            "phases": [{"name": "verify", "exit_conditions": ["evidence:test:data_processing_smoke"]}],
        },
        "artifacts": [{"id": "contract_smoke_result", "type": "contract_smoke_result", "path": str(smoke_path)}],
        "evidence": {
            "required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
            "history": [{
                "evidence_id": "verify_current",
                "ts": "2026-05-04T19:35:00Z",
                "result": "fail",
                "tests": [
                    {
                        "name": "test:data_processing_smoke",
                        "passed": False,
                        "details": "no_operational_callable_path",
                        "expected": {"required_executor_kind": "callable_or_plugin"},
                        "actual": None,
                    },
                    {
                        "name": "test:sandbox_execution_pass",
                        "passed": False,
                        "details": "missing_sandbox_execution_artifact",
                    },
                ],
            }],
            "latest": None,
        },
        "events": [
            {"ts": "2026-05-04T19:27:29Z", "type": "job_created", "msg": "Created learning job."},
            {"ts": "2026-05-04T19:35:00Z", "type": "evidence_recorded", "msg": "verify_current"},
        ],
    })
    record = _FakeRecord({
        "skill_id": "data_processing_v1",
        "name": "Data Processing",
        "status": "learning",
        "capability_type": "procedural",
        "learning_job_id": "job_current",
        "verification_required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
        "verification_latest": None,
        "verification_history": [],
    })

    packet = build_skill_audit_packet(
        "data_processing_v1",
        _FakeRegistry(record),
        _fake_orchestrator([current_job]),
    )

    assert packet["status"] == "learning"
    assert packet["verified"] is False
    assert packet["request_context"]["speaker"] == "David"
    assert packet["resolver_contract"]["capability_contract"]["execution_contract_id"] == "data_transform_v1"
    assert packet["evidence_classes"]["lifecycle_evidence"] is True
    assert packet["evidence_classes"]["operational_contract_evidence"] is False
    assert {m["reason"] for m in packet["missing_proof"]} == {
        "no_operational_callable_path",
        "missing_sandbox_execution_artifact",
    }
    assert packet["artifacts"][0]["exists"] is True
    assert packet["artifacts"][0]["preview"]["contract_id"] == "data_transform_v1"
    assert any(e["type"] == "evidence_recorded" for e in packet["timeline"])


def test_skill_audit_packet_labels_old_false_positive_as_historical(tmp_path):
    from skills.audit_trail import build_skill_audit_packet

    old_job = _FakeJob({
        "job_id": "job_old",
        "skill_id": "data_processing_v1",
        "status": "completed",
        "phase": "register",
        "created_at": "2026-05-04T17:43:27Z",
        "updated_at": "2026-05-04T18:04:45Z",
        "requested_by": {"source": "user"},
        "risk_level": "low",
        "matrix_protocol": False,
        "protocol_id": "",
        "plan": {},
        "artifacts": [{"id": "integration_test_passed", "type": "integration_test_passed"}],
        "evidence": {
            "required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
            "history": [{
                "evidence_id": "verify_old",
                "ts": "2026-05-04T17:59:40Z",
                "result": "pass",
                "tests": [
                    {
                        "name": "test:data_processing_smoke",
                        "passed": True,
                        "details": "Method artifact exists and all gates pass for data_processing_v1.",
                    },
                    {
                        "name": "test:sandbox_execution_pass",
                        "passed": True,
                        "details": "Method artifact exists and all gates pass for data_processing_v1.",
                    },
                ],
            }],
        },
        "events": [{"ts": "2026-05-04T18:04:45Z", "type": "job_completed", "msg": "Old false-positive job"}],
    })
    current_job = _FakeJob({
        "job_id": "job_current",
        "skill_id": "data_processing_v1",
        "status": "active",
        "phase": "assess",
        "created_at": "2026-05-04T19:27:29Z",
        "updated_at": "2026-05-04T19:27:29Z",
        "requested_by": {"source": "user"},
        "risk_level": "low",
        "matrix_protocol": False,
        "protocol_id": "",
        "plan": {},
        "artifacts": [],
        "evidence": {
            "required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
            "history": [],
        },
        "events": [{"ts": "2026-05-04T19:27:29Z", "type": "job_created", "msg": "Created learning job."}],
    })
    record = _FakeRecord({
        "skill_id": "data_processing_v1",
        "name": "Data Processing",
        "status": "learning",
        "capability_type": "procedural",
        "learning_job_id": "job_current",
        "verification_required": ["test:data_processing_smoke", "test:sandbox_execution_pass"],
        "verification_latest": None,
        "verification_history": [],
    })

    packet = build_skill_audit_packet(
        "data_processing_v1",
        _FakeRegistry(record),
        _fake_orchestrator([old_job, current_job]),
    )

    jobs = {j["job_id"]: j for j in packet["jobs"]}
    assert jobs["job_old"]["is_historical"] is True
    assert jobs["job_current"]["is_current"] is True
    assert packet["verified"] is False
    assert packet["evidence_classes"]["operational_contract_evidence"] is False
    assert "historical jobs" in " ".join(packet["integrity_notes"]).lower()
