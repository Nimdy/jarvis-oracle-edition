"""Tests for acquisition.planner — plan synthesis, lane ordering, dependencies.

Safety: No filesystem interaction, no live state access.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acquisition.job import (
    CapabilityAcquisitionJob,
    DocumentationArtifact,
)
from acquisition.planner import AcquisitionPlanner


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_job_for_planner(
    outcome_class: str = "plugin_creation",
    required_lanes: list[str] | None = None,
    risk_tier: int = 2,
    title: str = "test capability",
) -> CapabilityAcquisitionJob:
    if required_lanes is None:
        from acquisition.classifier import _LANE_MAP
        required_lanes = list(_LANE_MAP.get(outcome_class, []))
    return CapabilityAcquisitionJob(
        acquisition_id="acq_planner_test",
        title=title,
        outcome_class=outcome_class,
        required_lanes=required_lanes,
        risk_tier=risk_tier,
    )


# ---------------------------------------------------------------------------
# Risk label mapping
# ---------------------------------------------------------------------------

def test_risk_label_low():
    p = AcquisitionPlanner()
    assert p._risk_label(0) == "low"

def test_risk_label_medium():
    p = AcquisitionPlanner()
    assert p._risk_label(1) == "medium"

def test_risk_label_high():
    p = AcquisitionPlanner()
    assert p._risk_label(2) == "high"

def test_risk_label_critical():
    p = AcquisitionPlanner()
    assert p._risk_label(3) == "critical"

def test_risk_label_unknown():
    p = AcquisitionPlanner()
    assert p._risk_label(99) == "unknown"


# ---------------------------------------------------------------------------
# Implementation path ordering
# ---------------------------------------------------------------------------

def test_implementation_path_contains_only_required_lanes():
    job = _make_job_for_planner(
        outcome_class="plugin_creation",
        required_lanes=["evidence_grounding", "planning", "implementation", "truth"],
    )
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    lane_names = [step["lane"] for step in plan.implementation_path]
    assert set(lane_names) == {"evidence_grounding", "planning", "implementation", "truth"}

def test_implementation_path_respects_lane_order():
    job = _make_job_for_planner(
        outcome_class="plugin_creation",
        required_lanes=["truth", "implementation", "evidence_grounding", "planning"],
    )
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    lane_names = [step["lane"] for step in plan.implementation_path]
    assert lane_names.index("evidence_grounding") < lane_names.index("planning")
    assert lane_names.index("planning") < lane_names.index("implementation")
    assert lane_names.index("implementation") < lane_names.index("truth")

def test_implementation_path_excludes_unrequired_lanes():
    job = _make_job_for_planner(
        outcome_class="knowledge_only",
        required_lanes=["evidence_grounding", "truth"],
    )
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    lane_names = [step["lane"] for step in plan.implementation_path]
    assert "implementation" not in lane_names
    assert "plugin_quarantine" not in lane_names


# ---------------------------------------------------------------------------
# Dependency correctness
# ---------------------------------------------------------------------------

def test_doc_resolution_depends_on_evidence_grounding():
    job = _make_job_for_planner(required_lanes=["evidence_grounding", "doc_resolution", "truth"])
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    for step in plan.implementation_path:
        if step["lane"] == "doc_resolution":
            assert "evidence_grounding" in step["depends_on"]

def test_implementation_depends_on_planning():
    job = _make_job_for_planner(required_lanes=["evidence_grounding", "planning", "implementation", "truth"])
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    for step in plan.implementation_path:
        if step["lane"] == "implementation":
            assert "planning" in step["depends_on"]

def test_truth_has_no_dependencies():
    job = _make_job_for_planner(required_lanes=["evidence_grounding", "truth"])
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    for step in plan.implementation_path:
        if step["lane"] == "truth":
            assert step["depends_on"] == []


# ---------------------------------------------------------------------------
# Verification path by outcome class
# ---------------------------------------------------------------------------

def test_verification_path_knowledge_only():
    job = _make_job_for_planner(outcome_class="knowledge_only")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert plan.verification_path == [], "knowledge_only needs no verification"

def test_verification_path_plugin_creation():
    job = _make_job_for_planner(outcome_class="plugin_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    types = [s["type"] for s in plan.verification_path]
    assert "sandbox" in types
    assert "shadow" in types

def test_verification_path_skill_creation():
    job = _make_job_for_planner(outcome_class="skill_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    types = [s["type"] for s in plan.verification_path]
    assert "sandbox" in types
    assert "baseline" in types


# ---------------------------------------------------------------------------
# Rollback path by outcome class
# ---------------------------------------------------------------------------

def test_rollback_path_knowledge_only():
    job = _make_job_for_planner(outcome_class="knowledge_only")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert plan.rollback_path == []

def test_rollback_path_plugin_creation():
    job = _make_job_for_planner(outcome_class="plugin_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    types = [s["type"] for s in plan.rollback_path]
    assert "plugin_disable" in types

def test_rollback_path_core_upgrade():
    job = _make_job_for_planner(outcome_class="core_upgrade")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    types = [s["type"] for s in plan.rollback_path]
    assert "patch_restore" in types


# ---------------------------------------------------------------------------
# Promotion criteria
# ---------------------------------------------------------------------------

def test_promotion_criteria_requires_human_approval_for_risky():
    job = _make_job_for_planner(outcome_class="plugin_creation", risk_tier=2)
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert any("approval" in c.lower() for c in plan.promotion_criteria)

def test_promotion_criteria_no_human_approval_for_safe():
    job = _make_job_for_planner(outcome_class="knowledge_only", risk_tier=0)
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert not any("approval" in c.lower() for c in plan.promotion_criteria)

def test_promotion_criteria_plugin_shadow():
    job = _make_job_for_planner(outcome_class="plugin_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert any("shadow" in c.lower() for c in plan.promotion_criteria)


# ---------------------------------------------------------------------------
# Required artifacts
# ---------------------------------------------------------------------------

def test_required_artifacts_knowledge_only():
    job = _make_job_for_planner(outcome_class="knowledge_only")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert "ResearchArtifact" in plan.required_artifacts
    assert "AcquisitionPlan" not in plan.required_artifacts

def test_required_artifacts_plugin_creation():
    job = _make_job_for_planner(outcome_class="plugin_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert "PluginArtifact" in plan.required_artifacts
    assert "VerificationBundle" in plan.required_artifacts

def test_required_artifacts_skill_creation():
    job = _make_job_for_planner(outcome_class="skill_creation")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert "SkillArtifact" in plan.required_artifacts

def test_required_artifacts_core_upgrade():
    job = _make_job_for_planner(outcome_class="core_upgrade")
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert "UpgradeArtifact" in plan.required_artifacts


# ---------------------------------------------------------------------------
# Doc artifact ID propagation
# ---------------------------------------------------------------------------

def test_doc_artifact_ids_propagated():
    job = _make_job_for_planner(outcome_class="plugin_creation")
    doc1 = DocumentationArtifact(artifact_id="doc_001")
    doc2 = DocumentationArtifact(artifact_id="doc_002")
    p = AcquisitionPlanner()
    plan = p.synthesize(job, doc_artifacts=[doc1, doc2])
    assert plan.doc_artifact_ids == ["doc_001", "doc_002"]


# ---------------------------------------------------------------------------
# Plan has independent ID
# ---------------------------------------------------------------------------

def test_plan_id_differs_from_acquisition_id():
    job = _make_job_for_planner()
    p = AcquisitionPlanner()
    plan = p.synthesize(job)
    assert plan.plan_id != job.acquisition_id
    assert plan.plan_id.startswith("art_")
    assert plan.acquisition_id == job.acquisition_id


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

def test_planner_get_status():
    p = AcquisitionPlanner()
    status = p.get_status()
    assert status == {"available": True}


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
