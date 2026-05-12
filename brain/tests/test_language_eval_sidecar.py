import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class LanguageEvalSidecarTests(unittest.TestCase):
    def test_collector_reads_language_snapshot(self) -> None:
        from jarvis_eval.collector import EvalCollector

        fake_corpus = types.SimpleNamespace(get_stats=lambda: {
            "total_examples": 12,
            "counts_by_response_class": {"recent_learning": 5, "identity_answer": 7},
            "counts_by_route": {"introspection": 5, "identity": 7},
            "last_capture_ts": 123.0,
        })
        fake_quality = types.SimpleNamespace(get_stats=lambda: {
            "total_events": 20,
            "native_usage_rate": 0.8,
            "fail_closed_rate": 0.1,
            "counts_by_response_class": {"recent_learning": 10},
            "counts_by_outcome": {"ok": 18, "barge_in": 2},
            "native_used_by_class": {"recent_learning": 10},
            "fail_closed_by_class": {"recent_learning": 1},
            "last_event_ts": 456.0,
            "runtime_guard": {
                "total": 4,
                "live_total": 2,
                "blocked_by_guard_count": 1,
                "unpromoted_live_attempts": 0,
                "live_red_classes": 0,
                "live_by_class": {"self_introspection": 2},
                "blocked_by_class": {"self_introspection": 1},
                "by_rollout_mode": {"off": 4},
                "by_reason": {"bridge_disabled": 4},
                "by_promotion_level": {"shadow": 4},
                "live_rate": 0.5,
                "blocked_rate": 0.25,
                "last_ts": 455.0,
            },
        })
        fake_corpus_mod = types.ModuleType("reasoning.language_corpus")
        fake_corpus_mod.language_corpus = fake_corpus
        fake_telemetry_mod = types.ModuleType("reasoning.language_telemetry")
        fake_telemetry_mod.language_quality_telemetry = fake_quality

        collector = EvalCollector(engine=types.SimpleNamespace())
        with patch.dict("sys.modules", {
            "reasoning.language_corpus": fake_corpus_mod,
            "reasoning.language_telemetry": fake_telemetry_mod,
        }):
            snap = collector._read_language()

        self.assertEqual(snap["corpus_total_examples"], 12)
        self.assertEqual(snap["quality_total_events"], 20)
        self.assertAlmostEqual(snap["quality_native_usage_rate"], 0.8)
        self.assertEqual(snap["quality_counts_by_outcome"]["ok"], 18)
        self.assertIn("gate_scores_by_class", snap)
        self.assertIn("promotion_summary", snap)
        self.assertIn("promotion_aggregate", snap)
        self.assertIn("gate_color_code", snap)
        self.assertIn("runtime_guard_total", snap)
        self.assertIn("runtime_unpromoted_live_attempts", snap)
        self.assertIn("runtime_live_red_classes", snap)
        self.assertIn("runtime_rollout_mode", snap)

    def test_scorecard_includes_language_metrics_and_window_deltas(self) -> None:
        from jarvis_eval.scorecards import build_oracle_scorecard, build_scorecard_summary

        current = build_oracle_scorecard({
            "truth_calibration": {"truth_score": 0.8, "maturity": 0.7},
            "belief_graph": {"integrity": {"health_score": 0.9, "orphan_rate": 0.05, "total_edges": 20}},
            "contradiction": {"contradiction_debt": 0.0},
            "quarantine": {"composite": 0.1, "chronic_count": 0},
            "soul_integrity": {"current_index": 0.9, "weakest_dimension": "none", "weakest_score": 0.9},
            "memory": {"total": 30, "avg_weight": 0.6, "strong_count": 10, "weak_count": 1},
            "autonomy": {"autonomy_level": 1, "overall_win_rate": 0.6, "total_wins": 3, "completed_total": 5},
            "identity": {},
            "hemisphere": {"total_networks": 2, "broadcast_slots_count": 1},
            "matrix": {"specialist_count": 0},
            "world_model_promotion": {"level": 1},
            "mutation_governor": {"mutation_count": 1, "rollback_count": 0, "total_rejections": 0, "mutations_this_hour": 1, "active_monitor": False},
            "dream_artifacts": {"buffer": {"buffer_size": 0, "total_promoted": 0, "by_state": {}}, "promotion_rate": 0.0},
            "library": {"total": 5, "studied": 3, "substantive_ratio": 0.6},
            "policy_telemetry": {},
            "language": {
                "corpus_total_examples": 40,
                "quality_total_events": 25,
                "quality_native_usage_rate": 0.8,
                "quality_fail_closed_rate": 0.1,
                "quality_counts_by_class": {"recent_learning": 10, "identity_answer": 15},
            },
        })
        reference = build_oracle_scorecard({
            "truth_calibration": {"truth_score": 0.75, "maturity": 0.65},
            "belief_graph": {"integrity": {"health_score": 0.88, "orphan_rate": 0.06, "total_edges": 18}},
            "contradiction": {"contradiction_debt": 0.0},
            "quarantine": {"composite": 0.12, "chronic_count": 0},
            "soul_integrity": {"current_index": 0.88, "weakest_dimension": "none", "weakest_score": 0.88},
            "memory": {"total": 28, "avg_weight": 0.58, "strong_count": 9, "weak_count": 1},
            "autonomy": {"autonomy_level": 1, "overall_win_rate": 0.55, "total_wins": 2, "completed_total": 4},
            "identity": {},
            "hemisphere": {"total_networks": 2, "broadcast_slots_count": 1},
            "matrix": {"specialist_count": 0},
            "world_model_promotion": {"level": 1},
            "mutation_governor": {"mutation_count": 0, "rollback_count": 0, "total_rejections": 0, "mutations_this_hour": 0, "active_monitor": False},
            "dream_artifacts": {"buffer": {"buffer_size": 0, "total_promoted": 0, "by_state": {}}, "promotion_rate": 0.0},
            "library": {"total": 4, "studied": 2, "substantive_ratio": 0.5},
            "policy_telemetry": {},
            "language": {
                "corpus_total_examples": 20,
                "quality_total_events": 10,
                "quality_native_usage_rate": 0.5,
                "quality_fail_closed_rate": 0.2,
                "quality_counts_by_class": {"recent_learning": 10},
            },
        })

        summary = build_scorecard_summary([
            {"timestamp": 1000.0, "metrics": reference},
            {"timestamp": 1000.0 + 3600.0, "metrics": current},
        ])

        current_lang = summary["current"]["language"]
        self.assertEqual(current_lang["corpus_total_examples"], 40)
        self.assertEqual(current_lang["quality_total_events"], 25)
        self.assertAlmostEqual(current_lang["native_usage_rate"], 0.8)

        win = summary["windows"]["15m"]
        self.assertTrue(win["available"])

        win_1h = summary["windows"]["1h"]
        self.assertTrue(win_1h["available"])
        self.assertAlmostEqual(win_1h["language"]["native_usage_delta"], 0.3)
        self.assertAlmostEqual(win_1h["language"]["fail_closed_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
