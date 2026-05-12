import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jarvis_eval import language_promotion as lp
from jarvis_eval.language_scorers import BOUNDED_RESPONSE_CLASSES
from reasoning.language_runtime_bridge import RuntimeLanguagePolicy, decide_runtime_consumption


class LanguagePromotionGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = lp.PROMOTION_STATE_PATH
        self._orig_shadow_to_canary = lp.SHADOW_TO_CANARY_THRESHOLD
        self._orig_canary_to_live = lp.CANARY_TO_LIVE_THRESHOLD
        self._orig_rollback = lp.ROLLBACK_THRESHOLD
        self._orig_shadow_dwell = lp.MIN_SHADOW_DWELL_S
        self._orig_canary_dwell = lp.MIN_CANARY_DWELL_S
        lp.PROMOTION_STATE_PATH = Path(self._tmp.name) / "language_promotion.json"
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        lp.PROMOTION_STATE_PATH = self._orig_path
        lp.SHADOW_TO_CANARY_THRESHOLD = self._orig_shadow_to_canary
        lp.CANARY_TO_LIVE_THRESHOLD = self._orig_canary_to_live
        lp.ROLLBACK_THRESHOLD = self._orig_rollback
        lp.MIN_SHADOW_DWELL_S = self._orig_shadow_dwell
        lp.MIN_CANARY_DWELL_S = self._orig_canary_dwell
        self._tmp.cleanup()

    def _build_stats(self, bump: int = 0):
        per_class_count = 35 + bump
        class_counts = {rc: per_class_count for rc in BOUNDED_RESPONSE_CLASSES}
        total_examples = sum(class_counts.values())
        recent_examples = [
            {
                "response_class": rc,
                "lead": "ok",
                "confidence": 0.95,
                "safety_flags": [],
                "provenance_verdict": "bounded_native",
            }
            for rc in BOUNDED_RESPONSE_CLASSES
        ]
        corpus = {
            "total_examples": total_examples,
            "counts_by_response_class": class_counts,
            "counts_by_provenance": {"grounded_tool_data": total_examples},
            "recent_examples": recent_examples,
            "last_capture_ts": float(100 + bump),
        }
        telemetry = {
            "total_events": total_examples,
            "counts_by_response_class": class_counts,
            "native_used_by_class": dict(class_counts),
            "fail_closed_by_class": {rc: 0 for rc in BOUNDED_RESPONSE_CLASSES},
            "native_usage_rate": 1.0,
            "last_event_ts": float(200 + bump),
        }
        return corpus, telemetry

    def test_evaluate_skips_when_signature_unchanged(self) -> None:
        gov = lp.LanguagePromotionGovernor()
        corpus, telemetry = self._build_stats()
        first = gov.evaluate(corpus, telemetry)
        second = gov.evaluate(corpus, telemetry)
        for rc in BOUNDED_RESPONSE_CLASSES:
            self.assertEqual(first[rc]["total_evaluations"], 1)
            self.assertEqual(second[rc]["total_evaluations"], 1)

    def test_evaluate_runs_when_signature_changes(self) -> None:
        gov = lp.LanguagePromotionGovernor()
        corpus, telemetry = self._build_stats()
        first = gov.evaluate(corpus, telemetry)
        corpus2, telemetry2 = self._build_stats(bump=1)
        second = gov.evaluate(corpus2, telemetry2)
        for rc in BOUNDED_RESPONSE_CLASSES:
            self.assertEqual(first[rc]["total_evaluations"], 1)
            self.assertEqual(second[rc]["total_evaluations"], 2)

    def test_promotion_transition_records_reason_and_metadata(self) -> None:
        lp.SHADOW_TO_CANARY_THRESHOLD = 1
        lp.MIN_SHADOW_DWELL_S = 0.0

        gov = lp.LanguagePromotionGovernor()
        corpus, telemetry = self._build_stats()
        result = gov.evaluate(corpus, telemetry)

        for rc in BOUNDED_RESPONSE_CLASSES:
            row = result[rc]
            self.assertEqual(row["level"], "canary")
            self.assertTrue(row["level_changed"])
            self.assertEqual(row["last_transition_to"], "canary")
            self.assertEqual(row["last_transition_from"], "shadow")
            self.assertTrue(row["last_transition_reason"])
            self.assertGreater(row["promotion_history_len"], 0)
            self.assertGreaterEqual(row["dwell_s"], 0.0)

    def test_rollback_records_reason_and_persists(self) -> None:
        lp.SHADOW_TO_CANARY_THRESHOLD = 1
        lp.MIN_SHADOW_DWELL_S = 0.0
        lp.ROLLBACK_THRESHOLD = 1

        gov = lp.LanguagePromotionGovernor()
        corpus_good, telemetry_good = self._build_stats()
        gov.evaluate(corpus_good, telemetry_good)

        corpus_red = {
            "total_examples": 0,
            "counts_by_response_class": {},
            "counts_by_provenance": {},
            "recent_examples": [],
            "last_capture_ts": 999.0,
        }
        telemetry_red = {
            "total_events": 0,
            "counts_by_response_class": {},
            "native_used_by_class": {},
            "fail_closed_by_class": {},
            "native_usage_rate": 0.0,
            "last_event_ts": 1000.0,
        }
        result = gov.evaluate(corpus_red, telemetry_red)

        for rc in BOUNDED_RESPONSE_CLASSES:
            row = result[rc]
            self.assertEqual(row["level"], "shadow")
            self.assertTrue(row["last_rollback_reason"])
            self.assertEqual(row["last_transition_to"], "shadow")

        gov._save()
        gov2 = lp.LanguagePromotionGovernor()
        gov2._load()
        summary = gov2.get_summary()
        for rc in BOUNDED_RESPONSE_CLASSES:
            self.assertEqual(summary[rc]["level"], "shadow")
            self.assertTrue(summary[rc]["last_transition_reason"])
            self.assertTrue(summary[rc]["last_rollback_reason"])


class RuntimeBridgePolicyTests(unittest.TestCase):
    def test_bridge_off_allows_native_candidate(self) -> None:
        decision = decide_runtime_consumption(
            "self_introspection",
            native_candidate=True,
            strict_native=False,
            policy=RuntimeLanguagePolicy(enabled=False, rollout_mode="off", canary_classes=frozenset()),
        )
        self.assertTrue(decision["native_allowed"])
        self.assertEqual(decision["reason"], "bridge_disabled")

    def test_canary_rollout_blocks_non_canary_class(self) -> None:
        decision = decide_runtime_consumption(
            "memory_recall",
            native_candidate=True,
            strict_native=False,
            policy=RuntimeLanguagePolicy(
                enabled=True,
                rollout_mode="canary",
                canary_classes=frozenset({"self_introspection"}),
            ),
        )
        self.assertFalse(decision["native_allowed"])
        self.assertTrue(decision["blocked_by_guard"])
        self.assertEqual(decision["reason"], "class_not_in_canary_rollout")

    def test_full_rollout_requires_promoted_level(self) -> None:
        with patch("reasoning.language_runtime_bridge.get_promotion_level", return_value="canary"), patch(
            "reasoning.language_runtime_bridge._get_promotion_summary_row",
            return_value={"color": "green"},
        ):
            decision = decide_runtime_consumption(
                "self_introspection",
                native_candidate=True,
                strict_native=False,
                policy=RuntimeLanguagePolicy(
                    enabled=True,
                    rollout_mode="full",
                    canary_classes=frozenset(),
                ),
            )
        self.assertTrue(decision["native_allowed"])
        self.assertTrue(decision["runtime_live"])
        self.assertEqual(decision["reason"], "promoted_level_canary")

    def test_strict_native_is_never_blocked(self) -> None:
        decision = decide_runtime_consumption(
            "capability_status",
            native_candidate=True,
            strict_native=True,
            policy=RuntimeLanguagePolicy(
                enabled=True,
                rollout_mode="full",
                canary_classes=frozenset(),
            ),
        )
        self.assertTrue(decision["native_allowed"])
        self.assertFalse(decision["blocked_by_guard"])
        self.assertEqual(decision["reason"], "strict_native_invariant")


if __name__ == "__main__":
    unittest.main()

