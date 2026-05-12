import tempfile
import unittest
from pathlib import Path

from reasoning.language_telemetry import LanguageQualityTelemetry


class LanguageQualityTelemetryTests(unittest.TestCase):
    def test_quality_stats_aggregate_native_and_fail_closed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_events.jsonl"
            telemetry = LanguageQualityTelemetry(path)

            telemetry.record_event(
                conversation_id="conv-1",
                route="introspection",
                response_class="recent_learning",
                provenance_verdict="strict_provenance_grounded",
                outcome="ok",
                user_feedback="positive",
                confidence=0.98,
                native_used=True,
                fail_closed=False,
                safety_flags=["fail_closed_when_missing"],
                query="What did you learn?",
                reply="The latest research record I can verify is below.",
            )
            telemetry.record_event(
                conversation_id="conv-2",
                route="introspection",
                response_class="recent_learning",
                provenance_verdict="strict_provenance_grounded",
                outcome="ok",
                user_feedback="negative",
                confidence=0.98,
                native_used=True,
                fail_closed=True,
                safety_flags=["fail_closed_when_missing"],
                query="What did you learn?",
                reply="I don't have a verified recent learning record yet.",
            )
            telemetry.record_event(
                conversation_id="conv-3",
                route="status",
                response_class="self_status",
                provenance_verdict="grounded_tool_data",
                outcome="barge_in",
                user_feedback="follow_up",
                confidence=0.95,
                native_used=False,
                fail_closed=False,
                safety_flags=["status_mode"],
                query="Status?",
                reply="Here is my current measured status.",
            )

            stats = telemetry.get_stats()
            self.assertEqual(stats["total_events"], 3)
            self.assertEqual(stats["counts_by_response_class"]["recent_learning"], 2)
            self.assertEqual(stats["counts_by_outcome"]["ok"], 2)
            self.assertEqual(stats["counts_by_feedback"]["negative"], 1)
            self.assertEqual(stats["native_used_by_class"]["recent_learning"], 2)
            self.assertEqual(stats["fail_closed_by_class"]["recent_learning"], 1)
            self.assertAlmostEqual(stats["native_usage_rate"], 2 / 3, places=5)
            self.assertAlmostEqual(stats["fail_closed_rate"], 1 / 3, places=5)
            self.assertEqual(stats["outcomes_by_class"]["self_status"]["barge_in"], 1)
            self.assertEqual(len(stats["recent_events"]), 3)

    def test_rehydrate_includes_rotated_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_events.jsonl"
            rotated = path.with_suffix(path.suffix + ".1")
            rotated.write_text(
                '{"timestamp": 1, "conversation_id": "conv-1", "route": "introspection", '
                '"response_class": "recent_learning", "provenance_verdict": "strict", '
                '"outcome": "ok", "user_feedback": "", "confidence": 1.0, '
                '"native_used": true, "fail_closed": false}\n',
                encoding="utf-8",
            )
            path.write_text(
                '{"timestamp": 2, "conversation_id": "conv-2", "route": "status", '
                '"response_class": "self_status", "provenance_verdict": "grounded", '
                '"outcome": "ok", "user_feedback": "", "confidence": 1.0, '
                '"native_used": false, "fail_closed": false}\n',
                encoding="utf-8",
            )

            telemetry = LanguageQualityTelemetry(path)
            stats = telemetry.get_stats()

            self.assertEqual(stats["total_events"], 2)
            self.assertEqual(stats["retained_file_count"], 2)
            self.assertEqual(stats["counts_by_response_class"]["recent_learning"], 1)
            self.assertEqual(stats["counts_by_response_class"]["self_status"], 1)

    def test_shadow_comparisons_do_not_pollute_quality_event_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_events.jsonl"
            telemetry = LanguageQualityTelemetry(path)

            telemetry.record_event(
                conversation_id="conv-main",
                route="introspection",
                response_class="self_introspection",
                provenance_verdict="bounded_introspection",
                outcome="ok",
                user_feedback="",
                confidence=0.92,
                native_used=True,
                fail_closed=False,
                query="How are you changing?",
                reply="I track measured changes across subsystems.",
            )
            telemetry.record_shadow_comparison(
                conversation_id="conv-main",
                response_class="self_introspection",
                query="How are you changing?",
                bounded_reply="I track measured changes across subsystems.",
                llm_reply="I feel more aware lately.",
                bounded_confidence=0.92,
                chosen="bounded",
                reason="phasec_adapter_student",
                model_family="phasec_adapter",
            )

            stats = telemetry.get_stats()
            self.assertEqual(stats["total_events"], 1)
            self.assertEqual(stats["counts_by_outcome"]["ok"], 1)
            self.assertEqual(stats["shadow_comparisons"]["total"], 1)
            self.assertEqual(stats["shadow_comparisons"]["by_model_family"]["phasec_adapter"], 1)
            self.assertEqual(stats["shadow_comparisons"]["by_choice"]["bounded"], 1)

            # Rehydrate path must keep the same isolation behavior.
            telemetry2 = LanguageQualityTelemetry(path)
            stats2 = telemetry2.get_stats()
            self.assertEqual(stats2["total_events"], 1)
            self.assertEqual(stats2["shadow_comparisons"]["total"], 1)

    def test_ambiguous_intent_probes_are_isolated_from_quality_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_events.jsonl"
            telemetry = LanguageQualityTelemetry(path)

            telemetry.record_event(
                conversation_id="conv-main",
                route="none",
                response_class="self_introspection",
                provenance_verdict="bounded_introspection",
                outcome="ok",
                user_feedback="",
                confidence=0.88,
                native_used=False,
                fail_closed=False,
                query="What did you read last?",
                reply="I can share what I most recently researched.",
            )
            telemetry.record_ambiguous_intent_probe(
                conversation_id="conv-main",
                query="What did you read last?",
                selected_route="NONE",
                candidate_intent="recent_research_or_processing",
                candidate_confidence=0.68,
                trigger="self_read_phrase",
                outcome="ok",
                user_feedback="correction",
            )

            stats = telemetry.get_stats()
            self.assertEqual(stats["total_events"], 1)
            self.assertEqual(stats["ambiguous_intent"]["total"], 1)
            self.assertEqual(stats["ambiguous_intent"]["by_selected_route"]["NONE"], 1)
            self.assertEqual(stats["ambiguous_intent"]["by_candidate_intent"]["recent_research_or_processing"], 1)
            self.assertEqual(stats["ambiguous_intent"]["by_feedback"]["correction"], 1)

            telemetry2 = LanguageQualityTelemetry(path)
            stats2 = telemetry2.get_stats()
            self.assertEqual(stats2["total_events"], 1)
            self.assertEqual(stats2["ambiguous_intent"]["total"], 1)

    def test_runtime_guard_metrics_track_blocked_and_live_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_events.jsonl"
            telemetry = LanguageQualityTelemetry(path)

            telemetry.record_event(
                conversation_id="conv-live",
                route="introspection",
                response_class="self_introspection",
                provenance_verdict="bounded_introspection",
                outcome="ok",
                user_feedback="",
                confidence=0.91,
                native_used=True,
                fail_closed=False,
                query="How are your systems doing?",
                reply="I can report measured subsystem state.",
                runtime_policy={
                    "bridge_enabled": True,
                    "rollout_mode": "full",
                    "promotion_level": "canary",
                    "gate_color": "green",
                    "native_candidate": True,
                    "native_allowed": True,
                    "strict_native": False,
                    "blocked_by_guard": False,
                    "runtime_live": True,
                    "unpromoted_live_attempt": False,
                    "reason": "promoted_level_canary",
                },
            )
            telemetry.record_event(
                conversation_id="conv-block",
                route="introspection",
                response_class="self_introspection",
                provenance_verdict="runtime_guard_forced_llm",
                outcome="ok",
                user_feedback="",
                confidence=0.65,
                native_used=False,
                fail_closed=False,
                query="How are your systems doing?",
                reply="I can provide a grounded summary.",
                runtime_policy={
                    "bridge_enabled": True,
                    "rollout_mode": "full",
                    "promotion_level": "shadow",
                    "gate_color": "red",
                    "native_candidate": True,
                    "native_allowed": False,
                    "strict_native": False,
                    "blocked_by_guard": True,
                    "runtime_live": False,
                    "unpromoted_live_attempt": True,
                    "reason": "unpromoted_level_shadow",
                },
            )

            stats = telemetry.get_stats()
            runtime = stats["runtime_guard"]
            self.assertEqual(runtime["total"], 2)
            self.assertEqual(runtime["live_total"], 1)
            self.assertEqual(runtime["blocked_by_guard_count"], 1)
            self.assertEqual(runtime["unpromoted_live_attempts"], 1)
            self.assertEqual(runtime["live_red_classes"], 0)
            self.assertEqual(runtime["live_by_class"]["self_introspection"], 1)
            self.assertEqual(runtime["blocked_by_class"]["self_introspection"], 1)
            self.assertEqual(runtime["by_rollout_mode"]["full"], 2)
            self.assertEqual(runtime["by_promotion_level"]["shadow"], 1)
            self.assertEqual(runtime["by_reason"]["unpromoted_level_shadow"], 1)


if __name__ == "__main__":
    unittest.main()
