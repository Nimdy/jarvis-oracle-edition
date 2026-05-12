"""Tests for acquisition.job — dataclasses, state transitions, persistence.

Safety: ALL persistence uses tempfile.mkdtemp(), NEVER touches ~/.jarvis/.
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
from unittest.mock import patch

from acquisition.job import (
    AcquisitionPlan,
    AcquisitionStore,
    CapabilityAcquisitionJob,
    DocumentationArtifact,
    LaneState,
    PlanReviewArtifact,
    ResearchArtifact,
    VerificationBundle,
    _SUBDIRS,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_job(**overrides) -> CapabilityAcquisitionJob:
    defaults = dict(
        acquisition_id="acq_test_0001",
        title="test intent",
        status="pending",
        outcome_class="knowledge_only",
        user_intent="test intent",
        required_lanes=["evidence_grounding", "truth"],
        risk_tier=0,
    )
    defaults.update(overrides)
    return CapabilityAcquisitionJob(**defaults)


def _make_store() -> tuple[AcquisitionStore, str]:
    tmpdir = tempfile.mkdtemp(prefix="jarvis_test_acqstore_")
    store = AcquisitionStore(base_dir=Path(tmpdir))
    # Replace _save to avoid the lazy import of memory.persistence.atomic_write_json
    def _simple_save(category: str, obj_id: str, data: dict) -> None:
        p = store._path(category, obj_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str))
    store._save = _simple_save  # type: ignore[assignment]
    return store, tmpdir


# ---------------------------------------------------------------------------
# LaneState tests
# ---------------------------------------------------------------------------

def test_lane_state_defaults():
    ls = LaneState()
    assert ls.status == "pending"
    assert ls.retry_count == 0
    assert ls.error == ""

def test_lane_state_round_trip():
    ls = LaneState(status="running", lane_run_id="lane_test_abc", child_id="ch_1",
                   started_at=100.0, retry_count=2)
    d = ls.to_dict()
    ls2 = LaneState.from_dict(d)
    assert ls2.status == ls.status
    assert ls2.lane_run_id == ls.lane_run_id
    assert ls2.retry_count == 2

def test_lane_state_from_dict_filters_unknown():
    d = {"status": "completed", "bogus_key": 42, "retry_count": 1}
    ls = LaneState.from_dict(d)
    assert ls.status == "completed"
    assert ls.retry_count == 1
    assert not hasattr(ls, "bogus_key") or getattr(ls, "bogus_key", None) is None


# ---------------------------------------------------------------------------
# CapabilityAcquisitionJob — status transitions
# ---------------------------------------------------------------------------

def test_set_status_terminal_sets_completed_at():
    job = _make_job()
    assert job.completed_at == 0.0
    job.set_status("completed")
    assert job.completed_at > 0.0, "completed_at should be set on terminal status"

def test_set_status_terminal_only_once():
    job = _make_job()
    job.set_status("completed")
    first_ts = job.completed_at
    time.sleep(0.01)
    job.set_status("failed")
    assert job.completed_at == first_ts, "completed_at must not be overwritten"

def test_set_status_nonterminal_no_completed_at():
    job = _make_job()
    job.set_status("executing")
    assert job.completed_at == 0.0

def test_set_status_updates_updated_at():
    job = _make_job()
    before = job.updated_at
    time.sleep(0.01)
    job.set_status("planning")
    assert job.updated_at > before


# ---------------------------------------------------------------------------
# CapabilityAcquisitionJob — lane operations
# ---------------------------------------------------------------------------

def test_init_lane():
    job = _make_job()
    ls = job.init_lane("evidence_grounding")
    assert ls.status == "pending"
    assert "evidence_grounding" in job.lanes
    assert ls.lane_run_id.startswith("lane_evidence_grounding_")

def test_start_lane_sets_running():
    job = _make_job()
    job.init_lane("truth")
    ls = job.start_lane("truth")
    assert ls.status == "running"
    assert ls.started_at > 0.0

def test_start_lane_auto_inits():
    job = _make_job()
    ls = job.start_lane("new_lane")
    assert "new_lane" in job.lanes
    assert ls.status == "running"

def test_complete_lane():
    job = _make_job()
    job.init_lane("evidence_grounding")
    job.start_lane("evidence_grounding")
    job.complete_lane("evidence_grounding", child_id="child_123")
    ls = job.lanes["evidence_grounding"]
    assert ls.status == "completed"
    assert ls.child_id == "child_123"
    assert ls.completed_at > 0.0

def test_complete_lane_missing_is_noop():
    job = _make_job()
    job.complete_lane("nonexistent_lane")
    assert "nonexistent_lane" not in job.lanes

def test_fail_lane_increments_retry():
    job = _make_job()
    job.init_lane("planning")
    job.start_lane("planning")
    job.fail_lane("planning", "some error")
    ls = job.lanes["planning"]
    assert ls.status == "failed"
    assert ls.retry_count == 1
    assert ls.error == "some error"
    job.fail_lane("planning", "again")
    assert ls.retry_count == 2

def test_fail_lane_missing_is_noop():
    job = _make_job()
    job.fail_lane("nonexistent")
    assert "nonexistent" not in job.lanes


# ---------------------------------------------------------------------------
# CapabilityAcquisitionJob — artifact refs
# ---------------------------------------------------------------------------

def test_add_artifact_ref_dedup():
    job = _make_job()
    job.add_artifact_ref("art_001")
    job.add_artifact_ref("art_002")
    job.add_artifact_ref("art_001")
    assert job.artifact_refs == ["art_001", "art_002"], "Duplicate refs should be rejected"

def test_add_artifact_ref_empty_ignored():
    job = _make_job()
    job.add_artifact_ref("")
    assert job.artifact_refs == []


# ---------------------------------------------------------------------------
# CapabilityAcquisitionJob — serialization round-trip
# ---------------------------------------------------------------------------

def test_job_to_dict_from_dict_round_trip():
    job = _make_job(outcome_class="plugin_creation", risk_tier=2)
    job.init_lane("evidence_grounding")
    job.start_lane("evidence_grounding")
    job.complete_lane("evidence_grounding", child_id="eg_1")
    job.init_lane("implementation")
    job.add_artifact_ref("art_x")

    d = job.to_dict()
    job2 = CapabilityAcquisitionJob.from_dict(d)

    assert job2.acquisition_id == job.acquisition_id
    assert job2.outcome_class == "plugin_creation"
    assert job2.risk_tier == 2
    assert "evidence_grounding" in job2.lanes
    assert isinstance(job2.lanes["evidence_grounding"], LaneState)
    assert job2.lanes["evidence_grounding"].status == "completed"
    assert job2.lanes["evidence_grounding"].child_id == "eg_1"
    assert "implementation" in job2.lanes
    assert job2.artifact_refs == ["art_x"]

def test_job_from_dict_filters_unknown_keys():
    d = _make_job().to_dict()
    d["unknown_future_field"] = 999
    job = CapabilityAcquisitionJob.from_dict(d)
    assert not hasattr(job, "unknown_future_field") or getattr(job, "unknown_future_field", None) is None


# ---------------------------------------------------------------------------
# Artifact round-trips
# ---------------------------------------------------------------------------

def test_research_artifact_to_dict():
    r = ResearchArtifact(acquisition_id="acq_1", sources=[{"s": 1}])
    d = r.to_dict()
    assert d["acquisition_id"] == "acq_1"
    assert d["sources"] == [{"s": 1}]

def test_documentation_artifact_round_trip():
    doc = DocumentationArtifact(acquisition_id="acq_1", source_type="mcp", topic="react")
    d = doc.to_dict()
    doc2 = DocumentationArtifact.from_dict(d)
    assert doc2.source_type == "mcp"
    assert doc2.topic == "react"

def test_plan_round_trip():
    plan = AcquisitionPlan(acquisition_id="acq_1", objective="build thing", risk_level="high")
    d = plan.to_dict()
    plan2 = AcquisitionPlan.from_dict(d)
    assert plan2.objective == "build thing"
    assert plan2.plan_id == plan.plan_id

def test_review_round_trip():
    review = PlanReviewArtifact(acquisition_id="acq_1", verdict="approved_as_is",
                                operator_notes="looks good")
    d = review.to_dict()
    review2 = PlanReviewArtifact.from_dict(d)
    assert review2.verdict == "approved_as_is"

def test_verification_bundle_round_trip():
    bundle = VerificationBundle(acquisition_id="acq_1", overall_passed=True,
                                lane_verdicts={"eg": True, "impl": False})
    d = bundle.to_dict()
    bundle2 = VerificationBundle.from_dict(d)
    assert bundle2.overall_passed is True
    assert bundle2.lane_verdicts == {"eg": True, "impl": False}


# ---------------------------------------------------------------------------
# AcquisitionStore — CRUD
# ---------------------------------------------------------------------------

def test_store_creates_subdirs():
    tmpdir = tempfile.mkdtemp(prefix="jarvis_test_store_dirs_")
    try:
        AcquisitionStore(base_dir=Path(tmpdir))
        for subdir in _SUBDIRS.values():
            assert (Path(tmpdir) / subdir).is_dir(), f"Missing subdir: {subdir}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_save_load_job():
    store, tmpdir = _make_store()
    try:
        job = _make_job()
        store.save_job(job)
        loaded = store.load_job(job.acquisition_id)
        assert loaded is not None
        assert loaded.acquisition_id == job.acquisition_id
        assert loaded.title == job.title
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_load_missing_returns_none():
    store, tmpdir = _make_store()
    try:
        assert store.load_job("acq_nonexistent") is None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_list_jobs_all():
    store, tmpdir = _make_store()
    try:
        j1 = _make_job(acquisition_id="acq_a", created_at=100.0, status="completed")
        j2 = _make_job(acquisition_id="acq_b", created_at=200.0, status="pending")
        store.save_job(j1)
        store.save_job(j2)
        jobs = store.list_jobs()
        assert len(jobs) == 2
        assert jobs[0].acquisition_id == "acq_b", "Should be sorted newest first"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_list_jobs_with_status_filter():
    store, tmpdir = _make_store()
    try:
        j1 = _make_job(acquisition_id="acq_c", status="completed")
        j2 = _make_job(acquisition_id="acq_d", status="pending")
        store.save_job(j1)
        store.save_job(j2)
        pending = store.list_jobs(status="pending")
        assert len(pending) == 1
        assert pending[0].acquisition_id == "acq_d"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_list_jobs_skips_corrupt():
    store, tmpdir = _make_store()
    try:
        jobs_dir = Path(tmpdir) / _SUBDIRS["jobs"]
        (jobs_dir / "acq_corrupt.json").write_text("NOT VALID JSON {{{")
        j = _make_job(acquisition_id="acq_good")
        store.save_job(j)
        jobs = store.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].acquisition_id == "acq_good"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_save_load_plan():
    store, tmpdir = _make_store()
    try:
        plan = AcquisitionPlan(plan_id="plan_001", acquisition_id="acq_1", objective="obj")
        store.save_plan(plan)
        loaded = store.load_plan("plan_001")
        assert loaded is not None
        assert loaded.objective == "obj"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_save_load_review():
    store, tmpdir = _make_store()
    try:
        review = PlanReviewArtifact(artifact_id="rev_001", verdict="rejected")
        store.save_review(review)
        loaded = store.load_review("rev_001")
        assert loaded is not None
        assert loaded.verdict == "rejected"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_store_get_summary():
    store, tmpdir = _make_store()
    try:
        j1 = _make_job(acquisition_id="acq_s1", status="completed")
        j2 = _make_job(acquisition_id="acq_s2", status="failed")
        j3 = _make_job(acquisition_id="acq_s3", status="executing")
        store.save_job(j1)
        store.save_job(j2)
        store.save_job(j3)
        summary = store.get_summary()
        assert summary["total_count"] == 3
        assert summary["completed_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["active_count"] == 1
        assert len(summary["recent"]) == 3
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
