"""Tests for Phase D language eval gate scorers."""

import unittest

from jarvis_eval.language_scorers import (
    score_sample_count,
    score_provenance_fidelity,
    score_exactness,
    score_hallucination_rate,
    score_fail_closed_correctness,
    score_native_usage_rate,
    score_style_quality,
    compute_gate_scores,
    classify_gate,
    classify_gate_reason,
    _is_grounded_verdict,
    MIN_SAMPLES_FOR_EVAL,
    BOUNDED_RESPONSE_CLASSES,
)


class TestIsGroundedVerdict(unittest.TestCase):
    def test_grounded_prefixes_recognized(self):
        for prefix in ("bounded_", "grounded_", "native_", "deterministic_",
                        "strict_", "registry_", "reflective_"):
            self.assertTrue(_is_grounded_verdict(f"{prefix}something"),
                            f"Expected {prefix}something to be grounded")

    def test_unknown_is_not_grounded(self):
        self.assertFalse(_is_grounded_verdict("unknown"))

    def test_empty_is_not_grounded(self):
        self.assertFalse(_is_grounded_verdict(""))

    def test_negative_prefix_is_not_grounded(self):
        self.assertFalse(_is_grounded_verdict("negative:capability_gate_rewrite"))

    def test_random_string_is_not_grounded(self):
        self.assertFalse(_is_grounded_verdict("some_random_verdict"))

    def test_legacy_native_verdicts_are_grounded(self):
        self.assertTrue(_is_grounded_verdict("introspection_capability_status_native"))
        self.assertTrue(_is_grounded_verdict("none_route_capability_status_native"))
        self.assertTrue(_is_grounded_verdict("none_route_memory_recall_native"))


class TestScoreSampleCount(unittest.TestCase):
    def test_at_threshold_returns_one(self):
        stats = {"total_examples": MIN_SAMPLES_FOR_EVAL}
        self.assertAlmostEqual(score_sample_count(stats), 1.0)

    def test_above_threshold_returns_one(self):
        stats = {"total_examples": 100}
        self.assertAlmostEqual(score_sample_count(stats), 1.0)

    def test_zero_returns_zero(self):
        stats = {"total_examples": 0}
        self.assertAlmostEqual(score_sample_count(stats), 0.0)

    def test_half_threshold_returns_half(self):
        stats = {"total_examples": MIN_SAMPLES_FOR_EVAL // 2}
        self.assertAlmostEqual(score_sample_count(stats), 0.5, places=1)

    def test_per_class_count(self):
        stats = {"counts_by_response_class": {"self_status": 20}}
        score = score_sample_count(stats, "self_status")
        self.assertAlmostEqual(score, 20 / MIN_SAMPLES_FOR_EVAL, places=3)

    def test_missing_class_returns_zero(self):
        stats = {"counts_by_response_class": {"self_status": 20}}
        self.assertAlmostEqual(score_sample_count(stats, "identity_answer"), 0.0)


class TestScoreProvenanceFidelity(unittest.TestCase):
    def test_all_grounded_returns_one(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "provenance_verdict": "grounded_tool_data"},
                {"response_class": "self_status", "provenance_verdict": "bounded_introspection"},
            ],
        }
        self.assertAlmostEqual(score_provenance_fidelity(stats), 1.0)

    def test_negative_examples_excluded(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "provenance_verdict": "grounded_tool_data"},
                {"response_class": "negative_example", "provenance_verdict": "negative:rewrite"},
            ],
        }
        # Only the non-negative example counts
        self.assertAlmostEqual(score_provenance_fidelity(stats), 1.0)

    def test_ungrounded_drags_score(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "provenance_verdict": "grounded_tool_data"},
                {"response_class": "self_status", "provenance_verdict": "unknown"},
            ],
        }
        self.assertAlmostEqual(score_provenance_fidelity(stats), 0.5)

    def test_empty_recent_falls_back_to_corpus(self):
        stats = {
            "recent_examples": [],
            "total_examples": 10,
            "counts_by_provenance": {
                "grounded_tool_data": 8,
                "unknown": 2,
            },
        }
        self.assertAlmostEqual(score_provenance_fidelity(stats), 0.8)


class TestScoreExactness(unittest.TestCase):
    def test_all_native_returns_one(self):
        corpus = {}
        telemetry = {
            "native_used_by_class": {"self_status": 10},
            "counts_by_response_class": {"self_status": 10},
        }
        self.assertAlmostEqual(score_exactness(corpus, telemetry, "self_status"), 1.0)

    def test_no_native_returns_zero(self):
        corpus = {}
        telemetry = {
            "native_used_by_class": {"self_status": 0},
            "counts_by_response_class": {"self_status": 10},
        }
        self.assertAlmostEqual(score_exactness(corpus, telemetry, "self_status"), 0.0)

    def test_zero_total_returns_zero(self):
        self.assertAlmostEqual(score_exactness({}, {"total_events": 0}), 0.0)


class TestScoreHallucinationRate(unittest.TestCase):
    def test_all_grounded_returns_one(self):
        stats = {
            "total_examples": 10,
            "counts_by_response_class": {},
            "counts_by_provenance": {"grounded_tool_data": 10},
        }
        self.assertAlmostEqual(score_hallucination_rate(stats), 1.0)

    def test_negative_examples_excluded_from_denominator(self):
        stats = {
            "total_examples": 20,
            "counts_by_response_class": {"negative_example": 10},
            "counts_by_provenance": {
                "grounded_tool_data": 10,
                "negative:rewrite": 10,
            },
        }
        # Non-neg total = 10, grounded = 10 → 10/10 = 1.0
        self.assertAlmostEqual(score_hallucination_rate(stats), 1.0)

    def test_zero_denominator_returns_one(self):
        stats = {
            "total_examples": 5,
            "counts_by_response_class": {"negative_example": 5},
            "counts_by_provenance": {},
        }
        self.assertAlmostEqual(score_hallucination_rate(stats), 1.0)


class TestScoreFailClosedCorrectness(unittest.TestCase):
    def test_all_handled_returns_one(self):
        telemetry = {
            "native_used_by_class": {"self_status": 8},
            "fail_closed_by_class": {"self_status": 2},
            "counts_by_response_class": {"self_status": 10},
        }
        self.assertAlmostEqual(score_fail_closed_correctness(telemetry, "self_status"), 1.0)

    def test_some_unhandled(self):
        telemetry = {
            "native_used_by_class": {"self_status": 5},
            "fail_closed_by_class": {"self_status": 2},
            "counts_by_response_class": {"self_status": 10},
        }
        self.assertAlmostEqual(score_fail_closed_correctness(telemetry, "self_status"), 0.7)


class TestScoreNativeUsageRate(unittest.TestCase):
    def test_precomputed_rate(self):
        telemetry = {"native_usage_rate": 0.75}
        self.assertAlmostEqual(score_native_usage_rate(telemetry), 0.75)

    def test_per_class(self):
        telemetry = {
            "native_used_by_class": {"self_status": 7},
            "counts_by_response_class": {"self_status": 10},
        }
        self.assertAlmostEqual(score_native_usage_rate(telemetry, "self_status"), 0.7)


class TestScoreStyleQuality(unittest.TestCase):
    def test_all_good_returns_one(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "lead": "Status ok", "confidence": 0.9, "safety_flags": []},
                {"response_class": "self_status", "lead": "All good", "confidence": 0.8, "safety_flags": []},
            ],
        }
        self.assertAlmostEqual(score_style_quality(stats), 1.0)

    def test_negative_examples_excluded(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "lead": "Status ok", "confidence": 0.9, "safety_flags": []},
                {"response_class": "negative_example", "lead": False, "confidence": 0.0, "safety_flags": []},
            ],
        }
        self.assertAlmostEqual(score_style_quality(stats), 1.0)

    def test_missing_lead_drags_score(self):
        stats = {
            "recent_examples": [
                {"response_class": "self_status", "lead": "", "confidence": 0.9, "safety_flags": []},
                {"response_class": "self_status", "lead": "OK", "confidence": 0.9, "safety_flags": []},
            ],
        }
        self.assertAlmostEqual(score_style_quality(stats), 0.5)


class TestComputeGateScores(unittest.TestCase):
    def test_returns_all_seven_dimensions(self):
        corpus = {"total_examples": 50, "recent_examples": [], "counts_by_provenance": {}, "counts_by_response_class": {}}
        telemetry = {"total_events": 0, "native_used_by_class": {}, "fail_closed_by_class": {}, "counts_by_response_class": {}}
        scores = compute_gate_scores(corpus, telemetry)
        self.assertEqual(len(scores), 7)
        expected_keys = {
            "sample_count", "provenance_fidelity", "exactness",
            "hallucination_rate", "fail_closed_correctness",
            "native_usage_rate", "style_quality",
        }
        self.assertEqual(set(scores.keys()), expected_keys)


class TestClassifyGate(unittest.TestCase):
    def test_all_green_scores(self):
        scores = {
            "sample_count": 1.0,
            "provenance_fidelity": 0.95,
            "exactness": 0.90,
            "hallucination_rate": 0.96,
            "fail_closed_correctness": 0.95,
            "native_usage_rate": 0.80,
            "style_quality": 0.95,
        }
        self.assertEqual(classify_gate(scores), "green")

    def test_low_sample_count_is_red(self):
        scores = {
            "sample_count": 0.3,
            "provenance_fidelity": 0.95,
            "exactness": 0.90,
            "hallucination_rate": 0.96,
            "fail_closed_correctness": 0.95,
            "native_usage_rate": 0.80,
            "style_quality": 0.95,
        }
        self.assertEqual(classify_gate(scores), "red")

    def test_high_hallucination_is_red(self):
        scores = {
            "sample_count": 1.0,
            "provenance_fidelity": 0.95,
            "exactness": 0.90,
            "hallucination_rate": 0.85,  # Below 1.0 - 0.10 = 0.90
            "fail_closed_correctness": 0.95,
            "native_usage_rate": 0.80,
            "style_quality": 0.95,
        }
        self.assertEqual(classify_gate(scores), "red")

    def test_moderate_scores_are_yellow(self):
        scores = {
            "sample_count": 1.0,
            "provenance_fidelity": 0.85,
            "exactness": 0.75,
            "hallucination_rate": 0.96,
            "fail_closed_correctness": 0.85,
            "native_usage_rate": 0.50,
            "style_quality": 0.80,
        }
        self.assertEqual(classify_gate(scores), "yellow")

    def test_reason_insufficient_samples(self):
        scores = {
            "sample_count": 0.2,
            "provenance_fidelity": 1.0,
            "exactness": 1.0,
            "hallucination_rate": 1.0,
            "fail_closed_correctness": 1.0,
            "native_usage_rate": 1.0,
            "style_quality": 1.0,
        }
        self.assertEqual(classify_gate_reason(scores), "insufficient_samples")

    def test_reason_quality_risk(self):
        scores = {
            "sample_count": 1.0,
            "provenance_fidelity": 1.0,
            "exactness": 1.0,
            "hallucination_rate": 0.80,
            "fail_closed_correctness": 1.0,
            "native_usage_rate": 1.0,
            "style_quality": 1.0,
        }
        self.assertEqual(classify_gate_reason(scores), "hallucination_ceiling")


class TestBoundedResponseClasses(unittest.TestCase):
    def test_expected_classes_present(self):
        self.assertIn("self_status", BOUNDED_RESPONSE_CLASSES)
        self.assertIn("self_introspection", BOUNDED_RESPONSE_CLASSES)
        self.assertIn("memory_recall", BOUNDED_RESPONSE_CLASSES)
        self.assertEqual(len(BOUNDED_RESPONSE_CLASSES), 7)


if __name__ == "__main__":
    unittest.main()
