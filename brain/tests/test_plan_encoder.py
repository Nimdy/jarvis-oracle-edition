"""Tests for acquisition.plan_encoder — feature encoding, verdict labels, shadow artifacts.

Safety: No persistence, no network, no event bus. Pure dataclass + math tests.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acquisition.plan_encoder import (
    FEATURE_DIM,
    PlanEvaluatorEncoder,
    ShadowPredictionArtifact,
    VerdictReasonCategory,
    encode_verdict,
    label_to_class,
    verdict_to_class,
)
from acquisition.job import AcquisitionPlan, CapabilityAcquisitionJob


# ---------------------------------------------------------------------------
# Feature vector basics
# ---------------------------------------------------------------------------


class TestFeatureVectorBasics:
    def test_output_dim(self):
        job = CapabilityAcquisitionJob()
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert len(vec) == FEATURE_DIM == 32

    def test_all_values_clamped_01(self):
        job = CapabilityAcquisitionJob()
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_empty_plan_produces_valid_vector(self):
        job = CapabilityAcquisitionJob()
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_full_plan_produces_valid_vector(self):
        job = CapabilityAcquisitionJob(
            outcome_class="plugin_creation",
            risk_tier=2,
            classification_confidence=0.85,
            required_lanes=["evidence_grounding", "planning", "implementation", "deployment"],
            doc_artifact_ids=["doc_1", "doc_2"],
            artifact_refs=["art_1", "art_2", "art_3"],
        )
        plan = AcquisitionPlan(
            objective="Build a joke tool",
            required_capabilities=["codegen", "sandbox", "plugin_registry"],
            required_artifacts=["art_1", "art_2"],
            risk_level="medium",
            implementation_path=[
                {"lane": "implementation", "description": "code gen"},
                {"lane": "sandbox", "description": "test"},
            ],
            verification_path=[
                {"type": "sandbox", "description": "run tests on implementation"},
            ],
            rollback_path=[{"type": "revert", "description": "revert plugin"}],
            promotion_criteria=["accuracy >= 90%", "latency < 500ms"],
            version=2,
            doc_artifact_ids=["doc_1"],
            user_story="As a user, I want random jokes so I can laugh.",
            technical_approach="Use a joke API with fallback to local corpus.",
            implementation_sketch="class JokeTool:\n    def invoke(self):\n        ...",
            dependencies=["requests", "codegen"],
            test_cases=["test_joke_returns_string", "test_fallback"],
            risk_analysis="Low risk. Fallback to local corpus if API is down.",
        )
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert len(vec) == 32
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"


# ---------------------------------------------------------------------------
# Classification block (dims 0-8)
# ---------------------------------------------------------------------------


class TestClassificationBlock:
    def test_outcome_class_one_hot(self):
        classes = [
            "knowledge_only", "skill_creation", "plugin_creation",
            "core_upgrade", "specialist_nn", "hardware_integration", "mixed",
        ]
        for idx, cls in enumerate(classes):
            job = CapabilityAcquisitionJob(outcome_class=cls)
            plan = AcquisitionPlan()
            vec = PlanEvaluatorEncoder.encode(job, plan)
            assert vec[idx] == 1.0, f"{cls} should set dim {idx}=1.0"
            for j in range(7):
                if j != idx:
                    assert vec[j] == 0.0, f"{cls} should NOT set dim {j}"

    def test_risk_tier_normalized(self):
        for tier in range(4):
            job = CapabilityAcquisitionJob(risk_tier=tier)
            plan = AcquisitionPlan()
            vec = PlanEvaluatorEncoder.encode(job, plan)
            expected = tier / 3.0
            assert abs(vec[7] - expected) < 1e-6

    def test_confidence(self):
        job = CapabilityAcquisitionJob(classification_confidence=0.73)
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert abs(vec[8] - 0.73) < 1e-6


# ---------------------------------------------------------------------------
# Relational quality block (dims 16-23)
# ---------------------------------------------------------------------------


class TestRelationalQuality:
    def test_has_all_sections_true(self):
        plan = AcquisitionPlan(
            user_story="story",
            technical_approach="approach",
            implementation_sketch="sketch",
            dependencies=["dep"],
            test_cases=["test"],
            risk_analysis="risk",
        )
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[16] == 1.0

    def test_has_all_sections_false_missing_one(self):
        plan = AcquisitionPlan(
            user_story="story",
            technical_approach="approach",
            implementation_sketch="",
            dependencies=["dep"],
            test_cases=["test"],
            risk_analysis="risk",
        )
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[16] == 0.0

    def test_rollback_exists_for_deployment(self):
        job = CapabilityAcquisitionJob(required_lanes=["deployment"])
        plan = AcquisitionPlan(rollback_path=[{"type": "revert"}])
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[20] == 1.0

    def test_rollback_missing_for_deployment(self):
        job = CapabilityAcquisitionJob(required_lanes=["deployment"])
        plan = AcquisitionPlan(rollback_path=[])
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[20] == 0.0

    def test_no_deployment_rollback_ok(self):
        job = CapabilityAcquisitionJob(required_lanes=["planning", "implementation"])
        plan = AcquisitionPlan(rollback_path=[])
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[20] == 1.0


# ---------------------------------------------------------------------------
# Text richness block (dims 24-27)
# ---------------------------------------------------------------------------


class TestTextRichness:
    def test_measurable_promotion_criteria(self):
        plan = AcquisitionPlan(
            promotion_criteria=[
                "accuracy >= 90%",
                "good performance",
                "latency < 500ms",
                "generally works",
            ],
        )
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert abs(vec[26] - 0.5) < 1e-6

    def test_risk_analysis_specificity_high(self):
        plan = AcquisitionPlan(
            risk_analysis="Medium risk. Mitigation: use fallback endpoint if primary fails.",
        )
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[27] == 1.0

    def test_risk_analysis_specificity_low(self):
        plan = AcquisitionPlan(risk_analysis="Low risk.")
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[27] == 0.0


# ---------------------------------------------------------------------------
# Verdict encoding
# ---------------------------------------------------------------------------


class TestVerdictEncoding:
    def test_approved_as_is(self):
        assert encode_verdict("approved_as_is") == [1.0, 0.0, 0.0]

    def test_approved_with_edits(self):
        assert encode_verdict("approved_with_edits") == [1.0, 0.0, 0.0]

    def test_rejected(self):
        assert encode_verdict("rejected") == [0.0, 1.0, 0.0]

    def test_cancelled(self):
        assert encode_verdict("cancelled") == [0.0, 1.0, 0.0]

    def test_convert_to_research_only(self):
        assert encode_verdict("convert_to_research_only") == [0.0, 0.0, 1.0]

    def test_route_to_self_improve(self):
        assert encode_verdict("route_to_self_improve") == [0.0, 0.0, 1.0]

    def test_use_existing_capability(self):
        assert encode_verdict("use_existing_capability") == [0.0, 0.0, 1.0]

    def test_unknown_verdict_defaults_to_revision(self):
        assert encode_verdict("something_unexpected") == [0.0, 0.0, 1.0]

    def test_all_7_verdicts_covered(self):
        verdicts = [
            "approved_as_is", "approved_with_edits", "rejected", "cancelled",
            "convert_to_research_only", "route_to_self_improve",
            "route_to_plugin_upgrade", "use_existing_capability",
        ]
        for v in verdicts:
            label = encode_verdict(v)
            assert len(label) == 3
            assert abs(sum(label) - 1.0) < 1e-6, f"Label for {v} doesn't sum to 1.0"


# ---------------------------------------------------------------------------
# verdict_to_class and label_to_class
# ---------------------------------------------------------------------------


class TestClassMappings:
    def test_verdict_to_class_approved(self):
        assert verdict_to_class("approved_as_is") == "approved"
        assert verdict_to_class("approved_with_edits") == "approved"

    def test_verdict_to_class_rejected(self):
        assert verdict_to_class("rejected") == "rejected"
        assert verdict_to_class("cancelled") == "rejected"

    def test_verdict_to_class_revision(self):
        assert verdict_to_class("convert_to_research_only") == "needs_revision"
        assert verdict_to_class("route_to_self_improve") == "needs_revision"

    def test_verdict_to_class_unknown(self):
        assert verdict_to_class("???") == "needs_revision"

    def test_label_to_class_approve(self):
        assert label_to_class([0.8, 0.1, 0.1]) == "approved"

    def test_label_to_class_reject(self):
        assert label_to_class([0.1, 0.7, 0.2]) == "rejected"

    def test_label_to_class_revision(self):
        assert label_to_class([0.1, 0.2, 0.7]) == "needs_revision"

    def test_label_to_class_short_vector(self):
        assert label_to_class([0.5]) == "needs_revision"

    def test_label_to_class_empty(self):
        assert label_to_class([]) == "needs_revision"


# ---------------------------------------------------------------------------
# VerdictReasonCategory enum
# ---------------------------------------------------------------------------


class TestVerdictReasonCategory:
    def test_all_categories_are_strings(self):
        for cat in VerdictReasonCategory:
            assert isinstance(cat.value, str)

    def test_expected_categories(self):
        expected = {
            "technical_weakness", "stale_docs", "wrong_lane_choice",
            "policy_safety", "unnecessary_duplication", "preference_style",
            "missing_evidence", "plan_quality_ok", "unknown",
        }
        actual = {c.value for c in VerdictReasonCategory}
        assert actual == expected


# ---------------------------------------------------------------------------
# ShadowPredictionArtifact
# ---------------------------------------------------------------------------


class TestShadowPredictionArtifact:
    def test_creation(self):
        a = ShadowPredictionArtifact(
            acquisition_id="acq_123",
            plan_id="art_456",
            plan_version=2,
            predicted_probs=[0.7, 0.2, 0.1],
            predicted_class="approved",
        )
        assert a.acquisition_id == "acq_123"
        assert a.plan_version == 2
        assert a.predicted_class == "approved"
        assert a.correct is None
        assert a.shadow_id.startswith("shd_")

    def test_provenance_fields_default_empty(self):
        a = ShadowPredictionArtifact(acquisition_id="acq_p")
        assert a.reason_category == ""
        assert a.model_version == ""
        assert a.risk_tier == 0
        assert a.outcome_class == ""

    def test_provenance_fields_roundtrip(self):
        a = ShadowPredictionArtifact(
            acquisition_id="acq_prov",
            plan_id="art_prov",
            plan_version=3,
            predicted_probs=[0.4, 0.4, 0.2],
            predicted_class="approved",
            reason_category="technical_weakness",
            model_version="hemi_plan_evaluator_v7",
            risk_tier=4,
            outcome_class="new_capability",
        )
        d = a.to_dict()
        b = ShadowPredictionArtifact.from_dict(d)
        assert b.reason_category == "technical_weakness"
        assert b.model_version == "hemi_plan_evaluator_v7"
        assert b.risk_tier == 4
        assert b.outcome_class == "new_capability"

    def test_to_dict_roundtrip(self):
        a = ShadowPredictionArtifact(
            acquisition_id="acq_x",
            plan_id="art_y",
            plan_version=1,
            predicted_probs=[0.3, 0.5, 0.2],
            predicted_class="rejected",
            actual_verdict="rejected",
            actual_class="rejected",
            correct=True,
        )
        d = a.to_dict()
        b = ShadowPredictionArtifact.from_dict(d)
        assert b.acquisition_id == a.acquisition_id
        assert b.plan_version == a.plan_version
        assert b.correct is True
        assert b.predicted_probs == a.predicted_probs

    def test_from_dict_ignores_extra_fields(self):
        d = {
            "shadow_id": "shd_test",
            "acquisition_id": "acq_1",
            "plan_id": "art_1",
            "plan_version": 1,
            "predicted_probs": [0.5, 0.3, 0.2],
            "predicted_class": "approved",
            "extra_field": "should be ignored",
        }
        a = ShadowPredictionArtifact.from_dict(d)
        assert a.shadow_id == "shd_test"
        assert not hasattr(a, "extra_field")


# ---------------------------------------------------------------------------
# Evidence enrichment
# ---------------------------------------------------------------------------


class TestEvidenceEnrichment:
    def test_enrich_freshness(self):
        job = CapabilityAcquisitionJob()
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[29] == 0.0

        class FakeDoc:
            freshness_score = 0.8

        vec = PlanEvaluatorEncoder.enrich_evidence_dims(vec, doc_artifacts=[FakeDoc(), FakeDoc()])
        assert abs(vec[29] - 0.8) < 1e-6

    def test_enrich_research_sources(self):
        job = CapabilityAcquisitionJob()
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        vec = PlanEvaluatorEncoder.enrich_evidence_dims(vec, research_sources=5)
        assert abs(vec[30] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_extreme_risk_tier_clamped(self):
        job = CapabilityAcquisitionJob(risk_tier=100)
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[7] == 1.0

    def test_unknown_outcome_class(self):
        job = CapabilityAcquisitionJob(outcome_class="nonexistent")
        plan = AcquisitionPlan()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[0] == 1.0  # falls back to index 0

    def test_many_deps_clamped(self):
        plan = AcquisitionPlan(dependencies=["d"] * 50)
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[22] == 1.0  # 50/20 clamped to 1.0

    def test_plan_version_high_clamped(self):
        plan = AcquisitionPlan(version=20)
        job = CapabilityAcquisitionJob()
        vec = PlanEvaluatorEncoder.encode(job, plan)
        assert vec[15] == 1.0
