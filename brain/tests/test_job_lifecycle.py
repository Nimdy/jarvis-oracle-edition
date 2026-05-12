"""Tests for learning job deletion, supersession, and dedup safety."""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from skills.learning_jobs import (
    LearningJob,
    LearningJobOrchestrator,
    LearningJobStore,
)
from skills.registry import SkillRecord, SkillRegistry


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture()
def store(tmp_dir):
    return LearningJobStore(root=tmp_dir)


@pytest.fixture()
def registry(tmp_dir):
    path = os.path.join(tmp_dir, "skill_registry.json")
    reg = SkillRegistry(path=path)
    reg._loaded = True
    return reg


@pytest.fixture()
def orch(store, registry):
    return LearningJobOrchestrator(store=store, registry=registry)


def _make_job(
    job_id: str,
    skill_id: str = "speaker_identification_v1",
    status: str = "active",
    phase: str = "assess",
) -> LearningJob:
    return LearningJob(
        job_id=job_id,
        skill_id=skill_id,
        capability_type="perceptual",
        status=status,
        phase=phase,
    )


# -----------------------------------------------------------------------
# Delete basics
# -----------------------------------------------------------------------

class TestDeleteJob:
    def test_delete_blocked_job(self, orch, store):
        job = _make_job("job_a", status="blocked", phase="verify")
        store.save(job)
        orch._active_jobs["job_a"] = job

        assert orch.delete_job("job_a") is True
        assert "job_a" not in orch._active_jobs
        assert store.load("job_a") is None

    def test_delete_completed_job(self, orch, store):
        job = _make_job("job_b", status="completed", phase="register")
        store.save(job)

        assert orch.delete_job("job_b") is True
        assert store.load("job_b") is None

    def test_refuse_active_job_without_force(self, orch, store):
        job = _make_job("job_c", status="active", phase="train")
        store.save(job)
        orch._active_jobs["job_c"] = job

        assert orch.delete_job("job_c") is False
        assert "job_c" in orch._active_jobs
        assert store.load("job_c") is not None

    def test_force_delete_active_job(self, orch, store):
        job = _make_job("job_d", status="active", phase="collect")
        store.save(job)
        orch._active_jobs["job_d"] = job

        assert orch.delete_job("job_d", remove_skill=True) is True
        assert "job_d" not in orch._active_jobs

    def test_delete_nonexistent_returns_false(self, orch):
        assert orch.delete_job("job_phantom") is False


# -----------------------------------------------------------------------
# Skill reconciliation after delete
# -----------------------------------------------------------------------

def _register_skill(registry, skill_id, name="Test", cap_type="perceptual", **kwargs):
    rec = SkillRecord(skill_id=skill_id, name=name, capability_type=cap_type, **kwargs)
    registry.register(rec)
    return rec


class TestSkillReconciliation:
    def test_repoints_skill_to_surviving_completed(self, orch, store, registry):
        rec = _register_skill(registry, "speaker_identification_v1", "Speaker ID")
        rec.learning_job_id = "job_blocked"
        rec.status = "blocked"
        registry.save()

        blocked = _make_job("job_blocked", status="blocked", phase="verify")
        completed = _make_job("job_completed", status="completed", phase="register")
        store.save(blocked)
        store.save(completed)
        orch._active_jobs["job_blocked"] = blocked

        orch.delete_job("job_blocked")

        rec_after = registry.get("speaker_identification_v1")
        assert rec_after.learning_job_id == "job_completed"
        assert rec_after.status == "verified"

    def test_clears_link_when_no_surviving_job(self, orch, store, registry):
        _register_skill(registry, "test_skill_v1", "Test Skill")
        rec = registry.get("test_skill_v1")
        rec.learning_job_id = "job_only"
        registry.save()

        job = _make_job("job_only", skill_id="test_skill_v1", status="blocked")
        store.save(job)
        orch._active_jobs["job_only"] = job

        orch.delete_job("job_only")

        rec_after = registry.get("test_skill_v1")
        assert rec_after.learning_job_id is None

    def test_no_reconcile_when_skill_points_elsewhere(self, orch, store, registry):
        _register_skill(registry, "other_v1", "Other")
        rec = registry.get("other_v1")
        rec.learning_job_id = "job_other"
        registry.save()

        job = _make_job("job_delete_me", skill_id="other_v1", status="blocked")
        store.save(job)
        orch._active_jobs["job_delete_me"] = job

        orch.delete_job("job_delete_me")

        rec_after = registry.get("other_v1")
        assert rec_after.learning_job_id == "job_other"


# -----------------------------------------------------------------------
# Supersession: completing a job cancels stale siblings
# -----------------------------------------------------------------------

class TestSupersession:
    def test_completing_cancels_blocked_sibling(self, orch, store):
        blocked = _make_job("job_old", status="blocked", phase="verify")
        store.save(blocked)
        orch._active_jobs["job_old"] = blocked

        completing = _make_job("job_new", status="active", phase="register")
        completing.status = "completed"
        store.save(completing)

        orch._supersede_stale_siblings(completing)

        assert "job_old" not in orch._active_jobs
        assert store.load("job_old") is None

    def test_does_not_cancel_different_skill(self, orch, store):
        other = _make_job("job_other", skill_id="emotion_detection_v1", status="blocked")
        store.save(other)
        orch._active_jobs["job_other"] = other

        completing = _make_job("job_new")
        completing.status = "completed"
        store.save(completing)

        orch._supersede_stale_siblings(completing)

        assert "job_other" in orch._active_jobs

    def test_does_not_cancel_active_sibling(self, orch, store):
        active = _make_job("job_active", status="active", phase="train")
        store.save(active)
        orch._active_jobs["job_active"] = active

        completing = _make_job("job_new")
        completing.status = "completed"
        store.save(completing)

        orch._supersede_stale_siblings(completing)

        assert "job_active" in orch._active_jobs


# -----------------------------------------------------------------------
# Dedup: create_job blocks duplicates including blocked
# -----------------------------------------------------------------------

class TestCreateJobDedup:
    def test_blocks_when_existing_active(self, orch, store):
        existing = _make_job("job_existing", status="active")
        store.save(existing)
        orch._active_jobs["job_existing"] = existing

        result = orch.create_job(
            "speaker_identification_v1", "perceptual",
            requested_by={"speaker": "system"},
        )
        assert result is None

    def test_blocks_when_existing_blocked(self, orch, store):
        existing = _make_job("job_existing", status="blocked")
        store.save(existing)
        orch._active_jobs["job_existing"] = existing

        result = orch.create_job(
            "speaker_identification_v1", "perceptual",
            requested_by={"speaker": "system"},
        )
        assert result is None

    def test_allows_when_existing_completed(self, orch, store):
        existing = _make_job("job_done", status="completed", phase="register")
        store.save(existing)

        result = orch.create_job(
            "speaker_identification_v1", "perceptual",
            requested_by={"speaker": "system"},
        )
        assert result is not None


class TestRetiredV1Migration:
    """Regression: stale _v1 perceptual skills must be purged on load (2026-04-18)."""

    def _write_registry(self, path, skills):
        import json
        with open(path, "w") as f:
            json.dump({"schema_version": 1, "skills": skills}, f)

    def _make_skill_dict(self, skill_id, status="blocked", learning_job_id=None):
        return {
            "skill_id": skill_id,
            "name": skill_id.replace("_", " ").title(),
            "status": status,
            "capability_type": "perceptual",
            "created_at": time.time(),
            "updated_at": time.time(),
            "learning_job_id": learning_job_id,
        }

    def test_blocked_speaker_v1_purged_on_load(self, tmp_dir):
        path = os.path.join(tmp_dir, "skill_registry.json")
        self._write_registry(path, [
            self._make_skill_dict("speaker_identification", status="verified"),
            self._make_skill_dict(
                "speaker_identification_v1",
                status="blocked",
                learning_job_id="job_stale_123",
            ),
        ])
        reg = SkillRegistry(path=path)
        reg.load()
        assert reg.get("speaker_identification_v1") is None
        assert reg.get("speaker_identification") is not None
        assert reg.get("speaker_identification").status == "verified"

    def test_blocked_emotion_v1_purged_on_load(self, tmp_dir):
        path = os.path.join(tmp_dir, "skill_registry.json")
        self._write_registry(path, [
            self._make_skill_dict("emotion_detection", status="verified"),
            self._make_skill_dict("emotion_detection_v1", status="blocked"),
        ])
        reg = SkillRegistry(path=path)
        reg.load()
        assert reg.get("emotion_detection_v1") is None
        assert reg.get("emotion_detection") is not None

    def test_migration_is_noop_on_clean_registry(self, tmp_dir):
        path = os.path.join(tmp_dir, "skill_registry.json")
        self._write_registry(path, [
            self._make_skill_dict("speaker_identification", status="verified"),
            self._make_skill_dict("emotion_detection", status="verified"),
        ])
        reg = SkillRegistry(path=path)
        reg.load()
        assert reg.get("speaker_identification") is not None
        assert reg.get("emotion_detection") is not None
        assert reg.get("speaker_identification_v1") is None

    def test_migration_persists_across_reload(self, tmp_dir):
        """After purge + save, reloading must not re-introduce the entry."""
        path = os.path.join(tmp_dir, "skill_registry.json")
        self._write_registry(path, [
            self._make_skill_dict("speaker_identification", status="verified"),
            self._make_skill_dict("speaker_identification_v1", status="blocked"),
        ])
        reg1 = SkillRegistry(path=path)
        reg1.load()
        assert reg1.get("speaker_identification_v1") is None

        reg2 = SkillRegistry(path=path)
        reg2.load()
        assert reg2.get("speaker_identification_v1") is None

    def test_migration_archives_purged_skills(self, tmp_dir, monkeypatch):
        """Purged skills must go through the archive helper so there's an audit trail."""
        path = os.path.join(tmp_dir, "skill_registry.json")
        self._write_registry(path, [
            self._make_skill_dict("speaker_identification_v1", status="blocked"),
        ])

        captured: list[list[dict]] = []
        monkeypatch.setattr(
            SkillRegistry,
            "_archive_purged_skills",
            staticmethod(lambda purged: captured.append(list(purged))),
        )

        reg = SkillRegistry(path=path)
        reg.load()

        assert len(captured) == 1, "Expected the archive helper to be called exactly once"
        assert any(s.get("skill_id") == "speaker_identification_v1" for s in captured[0])


class TestSkillToolRegistrationRollback:
    """Regression (2026-04-18): if ``create_job`` rejects a request (for any reason —
    non-actionable skill_id, dedup, etc.), the skill_tool must NOT leave a ghost
    ``SkillRecord`` on disk. Pre-existing records are left alone; only records
    this request speculatively created are rolled back.

    Root cause this test guards against:
      ``handle_skill_request_structured`` used to call ``_skill_registry.register()``
      before ``_learning_job_orch.create_job()``. If ``create_job`` returned
      ``None`` (e.g. the downstream actionability guard in
      ``skills/discovery.py::is_actionable_capability_phrase`` rejected the
      phrase), the registration was never rolled back and the user saw a
      phantom ``status="unknown"`` entry in the dashboard until the next
      brain restart cleared it via ``_sanitize_non_actionable_skills``.
    """

    def _setup_tool(self, tmp_dir):
        """Wire skill_tool with a real registry + a mock orchestrator."""
        from tools import skill_tool
        from skills.registry import SkillRegistry

        reg_path = os.path.join(tmp_dir, "skill_registry.json")
        reg = SkillRegistry(path=reg_path)
        reg._loaded = True

        class _MockOrch:
            def __init__(self):
                self.create_result = None
                self.create_calls = []

            def create_job(self, **kwargs):
                self.create_calls.append(kwargs)
                return self.create_result

            def get_active_jobs(self):
                return []

            store = None

        orch = _MockOrch()
        skill_tool.set_registry(reg)
        skill_tool.set_orchestrator(orch)
        return skill_tool, reg, orch

    def test_registration_rolled_back_when_create_job_returns_none(self, tmp_dir):
        """Non-actionable skill_id: create_job returns None → no ghost record."""
        skill_tool, reg, orch = self._setup_tool(tmp_dir)
        orch.create_result = None

        result = skill_tool.handle_skill_request_structured(
            "Jarvis, learn audio analysis.", speaker="user",
        )

        assert result["outcome"] == "job_creation_failed"
        assert result["status"] == "blocked"
        assert reg.get(result["skill_id"]) is None, (
            f"Registry should not retain a ghost record after create_job failure, "
            f"but found: {reg.get(result['skill_id'])}"
        )

    def test_preexisting_record_preserved_on_create_job_failure(self, tmp_dir):
        """If a SkillRecord already existed, a later dedup failure must NOT remove it."""
        skill_tool, reg, orch = self._setup_tool(tmp_dir)
        orch.create_result = None

        # Resolve what the tool would produce for this request, then pre-seed.
        from skills.resolver import resolve_skill
        resolution = resolve_skill("Jarvis, learn speaker diarization.")
        assert resolution is not None

        seeded = SkillRecord(
            skill_id=resolution.skill_id,
            name=resolution.name,
            status="blocked",
            capability_type=resolution.capability_type,
        )
        reg.register(seeded)

        result = skill_tool.handle_skill_request_structured(
            "Jarvis, learn speaker diarization.", speaker="user",
        )

        assert result["outcome"] == "job_creation_failed"
        still = reg.get(resolution.skill_id)
        assert still is not None, (
            "Pre-existing SkillRecord must not be removed by rollback — "
            "rollback only applies to records created by this request."
        )
        assert still.status == "blocked"

    def test_successful_create_job_leaves_record_in_place(self, tmp_dir):
        """Success path: record is registered AND stays registered."""
        skill_tool, reg, orch = self._setup_tool(tmp_dir)

        class _FakeJob:
            job_id = "job_ok_0001"
            skill_id = "speaker_diarization_v1"
            phase = "assess"
            status = "active"
            protocol_id = ""
            created_at = ""
            events: list = []
            matrix_protocol = False
            matrix_target = ""
            verification_profile = ""
            claimability_status = ""

        orch.create_result = _FakeJob()

        result = skill_tool.handle_skill_request_structured(
            "Jarvis, learn speaker diarization.", speaker="user",
        )

        assert result["outcome"] == "job_started"
        assert reg.get(result["skill_id"]) is not None, (
            "Successful job creation must leave the SkillRecord in the registry."
        )
