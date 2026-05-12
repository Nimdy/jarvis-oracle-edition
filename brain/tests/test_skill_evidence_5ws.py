"""Tests for the Skill Evidence 5 Ws enrichment.

Covers:
- SkillEvidence new field serialization roundtrip
- Backward compatibility with v1 evidence (missing fields)
- Register executor evidence preservation
- create_job rejection of non-actionable phrases
- Dashboard snapshot evidence_summary shape
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSkillEvidenceFields:
    """SkillEvidence enrichment: new 5 Ws fields."""

    def test_new_fields_present(self):
        from skills.registry import SkillEvidence
        ev = SkillEvidence(
            evidence_id="test_ev",
            timestamp=time.time(),
            result="pass",
        )
        assert ev.verified_by == ""
        assert ev.acceptance_criteria == {}
        assert ev.measured_values == {}
        assert ev.environment == {}
        assert ev.summary == ""
        assert ev.verification_method == ""
        assert ev.evidence_schema_version == "2"
        assert ev.artifact_refs == []
        assert ev.verification_scope == "smoke"
        assert ev.known_limitations == []
        assert ev.regression_baseline_available is False

    def test_roundtrip_to_dict_from_dict(self):
        from skills.registry import SkillEvidence
        ev = SkillEvidence(
            evidence_id="rt_test",
            timestamp=123456.0,
            result="pass",
            tests=[{"name": "a", "passed": True}],
            details="some detail",
            verified_by="ProceduralRegisterExecutor",
            acceptance_criteria={"test_a": {"threshold": True, "comparison": "=="}},
            measured_values={"test_a": {"value": True}},
            environment={"model": "kokoro_gpu"},
            summary="All tests passed.",
            verification_method="learning_job_procedural",
            evidence_schema_version="2",
            artifact_refs=[{"type": "research_summary", "path": "/tmp/a.json"}],
            verification_scope="functional",
            known_limitations=["no baseline"],
            regression_baseline_available=False,
        )
        d = ev.to_dict()
        restored = SkillEvidence.from_dict(d)
        assert restored.verified_by == "ProceduralRegisterExecutor"
        assert restored.verification_scope == "functional"
        assert restored.known_limitations == ["no baseline"]
        assert restored.acceptance_criteria == ev.acceptance_criteria
        assert restored.measured_values == ev.measured_values
        assert restored.environment == {"model": "kokoro_gpu"}
        assert restored.summary == "All tests passed."
        assert restored.artifact_refs == ev.artifact_refs

    def test_backward_compat_v1_evidence(self):
        """Old evidence without 5 Ws fields should deserialize with defaults."""
        from skills.registry import SkillEvidence
        v1_dict = {
            "evidence_id": "old_ev",
            "timestamp": 100.0,
            "result": "pass",
            "tests": [],
            "details": "Legacy evidence",
        }
        ev = SkillEvidence.from_dict(v1_dict)
        assert ev.evidence_id == "old_ev"
        assert ev.verified_by == ""
        assert ev.evidence_schema_version == "1"  # missing → defaults to "1"
        assert ev.verification_scope == "smoke"
        assert ev.known_limitations == []
        assert ev.regression_baseline_available is False


class TestCreateJobRejection:
    """create_job should reject non-actionable skill IDs."""

    def test_rejects_non_actionable_phrase(self):
        from skills.learning_jobs import LearningJobOrchestrator, LearningJobStore
        store = LearningJobStore.__new__(LearningJobStore)
        store.root = "/tmp/test_jarvis_jobs"
        orch = LearningJobOrchestrator.__new__(LearningJobOrchestrator)
        orch.store = store
        orch._registry = None
        orch._last_tick = 0

        result = orch.create_job(
            skill_id="better serve you than to operate in the shadows",
            capability_type="procedural",
            requested_by={"source": "test"},
        )
        assert result is None

    def test_accepts_actionable_phrase(self):
        from skills.learning_jobs import LearningJobOrchestrator, LearningJobStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = LearningJobStore(root=tmp)
            orch = LearningJobOrchestrator(store=store, registry=None)

            result = orch.create_job(
                skill_id="singing_v1",
                capability_type="procedural",
                requested_by={"source": "test"},
            )
            assert result is not None
            assert result.skill_id == "singing_v1"

    def test_accepts_camera_control(self):
        from skills.learning_jobs import LearningJobOrchestrator, LearningJobStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = LearningJobStore(root=tmp)
            orch = LearningJobOrchestrator(store=store, registry=None)

            result = orch.create_job(
                skill_id="camera_control",
                capability_type="control",
                requested_by={"source": "test"},
            )
            assert result is not None
            assert result.skill_id == "camera_control"


class TestVerifyExecutorFix:
    """ProceduralVerifyExecutor fail-fast for auto-generated skills."""

    def test_auto_generated_skill_blocks_immediately(self):
        from skills.executors.procedural import ProceduralVerifyExecutor
        from skills.learning_jobs import MAX_PHASE_FAILURES
        executor = ProceduralVerifyExecutor()
        job = MagicMock()
        job.skill_id = "test_skill"
        job.artifacts = [{"type": "research_summary"}]
        job.plan = {"summary": "Auto-generated procedural skill"}
        job.evidence = {"required": []}
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        job.phase = "verify"
        job.matrix_protocol = False
        ctx = {}

        result = executor.run(job, ctx)
        assert result.progressed is False
        assert "no_verification_method" in result.message
        assert job.failure["count"] == MAX_PHASE_FAILURES

    def test_matrix_protocol_auto_generated_does_not_block(self):
        from skills.executors.procedural import ProceduralVerifyExecutor
        executor = ProceduralVerifyExecutor()
        job = MagicMock()
        job.skill_id = "test_skill"
        job.artifacts = [{"type": "research_summary"}]
        job.plan = {"summary": "Auto-generated procedural skill"}
        job.evidence = {"required": ["test:procedure_smoke"]}
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        job.phase = "verify"
        job.matrix_protocol = True
        job.protocol_id = "SK-001"
        ctx = {"tool_router_available": True}

        result = executor.run(job, ctx)
        assert result.progressed is True
        assert "Matrix" in result.message

    def test_non_auto_generated_skill_fails_normally(self):
        from skills.executors.procedural import ProceduralVerifyExecutor
        executor = ProceduralVerifyExecutor()
        job = MagicMock()
        job.skill_id = "test_skill"
        job.artifacts = [{"type": "research_summary"}]
        job.plan = {"summary": "User-created skill"}
        job.evidence = {"required": []}
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        job.phase = "verify"
        job.matrix_protocol = False
        ctx = {}

        result = executor.run(job, ctx)
        assert result.progressed is False
        assert job.failure["count"] == 0


class TestRegisterEvidencePreservation:
    """Register executors should produce evidence with rich 5 Ws fields."""

    def _make_job(self):
        job = MagicMock()
        job.skill_id = "test_skill"
        job.job_id = "job_001"
        job.artifacts = [
            {"id": "research_summary", "type": "research_summary", "path": "/tmp/r.json"},
        ]
        job.evidence = {
            "required": [],
            "history": [{
                "evidence_id": "verify_test",
                "ts": "2025-01-01T00:00:00Z",
                "result": "pass",
                "tests": [{"name": "procedure_smoke", "passed": True, "details": "Smoke test OK"}],
            }],
        }
        return job

    def test_procedural_register_has_5ws(self):
        from skills.executors.procedural import ProceduralRegisterExecutor
        from skills.registry import SkillEvidence

        registry = MagicMock()
        registry.get.return_value = MagicMock(status="learning")
        registry.set_status.return_value = True

        executor = ProceduralRegisterExecutor()
        job = self._make_job()
        ctx = {"registry": registry}
        result = executor.run(job, ctx)

        assert result.progressed is True
        call_args = registry.set_status.call_args
        ev = call_args[1]["evidence"]
        assert isinstance(ev, SkillEvidence)
        assert ev.verified_by == "ProceduralRegisterExecutor"
        assert ev.verification_method == "learning_job_procedural"
        assert ev.evidence_schema_version == "2"
        assert ev.verification_scope == "functional"
        assert len(ev.known_limitations) > 0
        assert len(ev.artifact_refs) > 0
        assert ev.summary != ""

    def test_perceptual_register_has_5ws(self):
        from skills.executors.perceptual import PerceptualRegisterExecutor
        from skills.registry import SkillEvidence

        registry = MagicMock()
        registry.get.return_value = MagicMock(status="learning")
        registry.set_status.return_value = True

        executor = PerceptualRegisterExecutor()
        job = self._make_job()
        ctx = {"registry": registry}
        result = executor.run(job, ctx)

        assert result.progressed is True
        ev = registry.set_status.call_args[1]["evidence"]
        assert isinstance(ev, SkillEvidence)
        assert ev.verified_by == "PerceptualRegisterExecutor"
        assert "distillation" in ev.verification_method
        assert ev.evidence_schema_version == "2"

    def test_control_register_has_5ws(self):
        from skills.executors.control import ControlRegisterExecutor
        from skills.registry import SkillEvidence

        registry = MagicMock()
        registry.get.return_value = MagicMock(status="learning")
        registry.set_status.return_value = True

        executor = ControlRegisterExecutor()
        job = self._make_job()
        ctx = {"registry": registry}
        result = executor.run(job, ctx)

        assert result.progressed is True
        ev = registry.set_status.call_args[1]["evidence"]
        assert isinstance(ev, SkillEvidence)
        assert ev.verified_by == "ControlRegisterExecutor"
        assert "control" in ev.verification_method
        assert ev.evidence_schema_version == "2"


class TestSnapshotEnrichment:
    """Dashboard snapshot should include evidence_summary instead of bare boolean."""

    def test_snapshot_has_evidence_summary(self):
        from skills.registry import SkillRegistry, SkillEvidence, SkillRecord

        reg = SkillRegistry.__new__(SkillRegistry)
        reg._skills = {}
        reg._loaded = True

        ev = SkillEvidence(
            evidence_id="test",
            timestamp=time.time(),
            result="pass",
            verified_by="TestExecutor",
            verification_scope="functional",
            summary="All tests passed.",
            known_limitations=["smoke only"],
            regression_baseline_available=False,
            evidence_schema_version="2",
        )
        rec = SkillRecord(
            skill_id="test_skill",
            name="Test Skill",
            status="verified",
            verification_latest=ev,
            verification_history=[ev],
        )
        reg._skills["test_skill"] = rec

        snap = reg.get_status_snapshot()
        skill_snap = snap["skills"][0]
        assert skill_snap["has_evidence"] is True
        assert skill_snap["evidence_summary"] is not None

        es = skill_snap["evidence_summary"]
        assert es["summary"] == "All tests passed."
        assert es["verification_scope"] == "functional"
        assert es["verified_by"] == "TestExecutor"
        assert es["result"] == "pass"
        assert es["known_limitations"] == ["smoke only"]
        assert es["regression_baseline_available"] is False
        assert es["evidence_schema_version"] == "2"

    def test_snapshot_no_evidence(self):
        from skills.registry import SkillRegistry, SkillRecord

        reg = SkillRegistry.__new__(SkillRegistry)
        reg._skills = {}
        reg._loaded = True

        rec = SkillRecord(
            skill_id="unverified",
            name="Unverified Skill",
            status="learning",
        )
        reg._skills["unverified"] = rec

        snap = reg.get_status_snapshot()
        skill_snap = snap["skills"][0]
        assert skill_snap["has_evidence"] is False
        assert skill_snap["evidence_summary"] is None


class TestEvidenceHelpers:
    """Tests for evidence_helpers.py shared utilities."""

    def test_find_latest_verify_evidence(self):
        from skills.executors.evidence_helpers import find_latest_verify_evidence

        job = MagicMock()
        job.evidence = {
            "history": [
                {"evidence_id": "old", "ts": "2024-01-01T00:00:00Z", "result": "pass", "tests": [{"passed": True}]},
                {"evidence_id": "new", "ts": "2025-06-01T00:00:00Z", "result": "pass", "tests": [{"passed": True}]},
            ]
        }
        latest = find_latest_verify_evidence(job)
        assert latest is not None
        assert latest["evidence_id"] == "new"

    def test_find_latest_returns_none_on_empty(self):
        from skills.executors.evidence_helpers import find_latest_verify_evidence

        job = MagicMock()
        job.evidence = {"history": []}
        assert find_latest_verify_evidence(job) is None

    def test_collect_artifact_refs(self):
        from skills.executors.evidence_helpers import collect_artifact_refs

        job = MagicMock()
        job.artifacts = [
            {"id": "a1", "type": "research_summary", "path": "/nonexistent/path"},
            {"id": "a2", "type": "method"},
        ]
        refs = collect_artifact_refs(job)
        assert len(refs) == 2
        assert refs[0]["type"] == "research_summary"
        assert refs[0]["exists"] is False
        assert refs[1]["type"] == "method"

    def test_build_acceptance_criteria(self):
        from skills.executors.evidence_helpers import build_acceptance_criteria

        job = MagicMock()
        job.evidence = {"required": ["test:smoke", "test:integration"]}
        criteria = build_acceptance_criteria(job)
        assert "test:smoke" in criteria
        assert "test:integration" in criteria
        assert criteria["test:smoke"]["threshold"] is True
