import json
import tempfile
import unittest
from pathlib import Path

from reasoning import language_phasec as lp


class _PathPatch:
    def __init__(self, base_dir: Path):
        self.base = base_dir
        self.orig: dict[str, object] = {}

    def apply(self) -> None:
        mapping = {
            "JARVIS_DIR": self.base,
            "LANGUAGE_CORPUS_DIR": self.base / "language_corpus",
            "PHASEC_DIR": self.base / "language_corpus" / "phase_c",
            "BASELINE_LOCK_PATH": self.base / "language_corpus" / "phase_c" / "baseline_lock.json",
            "TOKENIZER_STRATEGY_PATH": self.base / "language_corpus" / "phase_c" / "tokenizer_strategy.json",
            "DATASET_PATH": self.base / "language_corpus" / "phase_c" / "dataset.jsonl",
            "DATASET_MANIFEST_PATH": self.base / "language_corpus" / "phase_c" / "dataset_manifest.json",
            "SPLIT_MANIFEST_PATH": self.base / "language_corpus" / "phase_c" / "split_manifest.json",
            "CHECKPOINT_PATH": self.base / "language_corpus" / "phase_c" / "student_checkpoint.json",
            "TRAIN_RUNS_PATH": self.base / "language_corpus" / "phase_c" / "train_runs.jsonl",
            "CORPUS_PATH": self.base / "language_corpus" / "examples.jsonl",
            "CORPUS_ROTATED_PATH": self.base / "language_corpus" / "examples.jsonl.1",
        }
        for k, v in mapping.items():
            self.orig[k] = getattr(lp, k)
            setattr(lp, k, v)
        lp.phasec_shadow_student = lp.PhaseCAdapterStudent()
        lp.phasec_harness = lp.PhaseCHarness()

    def restore(self) -> None:
        for k, v in self.orig.items():
            setattr(lp, k, v)
        lp.phasec_shadow_student = lp.PhaseCAdapterStudent()
        lp.phasec_harness = lp.PhaseCHarness()


class LanguagePhaseCTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = _PathPatch(Path(self._tmp.name))
        self._patch.apply()
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        self._patch.restore()
        self._tmp.cleanup()

    def test_baseline_lock_and_tokenizer_strategy_persist(self) -> None:
        baseline = lp.lock_phasec_baseline()
        self.assertEqual(baseline["baseline_version"], "phasec_v1")
        self.assertTrue(lp.BASELINE_LOCK_PATH.exists())

        records = [
            {
                "query": "How are you today?",
                "final_answer": "All systems operational with healthy metrics.",
                "meaning_frame": {"lead": "System is healthy"},
            },
            {
                "query": "What did you learn?",
                "final_answer": "I recently studied grounded introspection patterns.",
                "meaning_frame": {"lead": "Recent learning verified"},
            },
        ]
        strategy = lp.evaluate_tokenizer_strategy(records)
        self.assertTrue(lp.TOKENIZER_STRATEGY_PATH.exists())
        self.assertIn(strategy.get("strategy"), {"bpe", "sentencepiece"})
        if not strategy.get("candidates", {}).get("sentencepiece", {}).get("available", False):
            self.assertEqual(strategy.get("strategy"), "bpe")

    def test_dataset_contract_and_deterministic_split(self) -> None:
        records = [
            {
                "conversation_id": "c1",
                "query": "status",
                "route": "status",
                "response_class": "self_status",
                "provenance_verdict": "grounded_tool_data",
                "confidence": 0.9,
                "final_answer": "Measured status is stable and healthy.",
                "meaning_frame": {"lead": "Measured status"},
                "timestamp": 1.0,
            },
            {
                "conversation_id": "c2",
                "query": "bad",
                "route": "status",
                "response_class": "negative_example",
                "provenance_verdict": "negative:rewrite",
                "confidence": 0.0,
                "final_answer": "bad",
                "meaning_frame": {},
                "timestamp": 2.0,
            },
            {
                "conversation_id": "c3",
                "query": "too low confidence",
                "route": "status",
                "response_class": "self_status",
                "provenance_verdict": "grounded_tool_data",
                "confidence": 0.0,
                "final_answer": "ignored",
                "meaning_frame": {},
                "timestamp": 3.0,
            },
        ]
        manifest = lp.build_dataset(records)
        self.assertEqual(manifest["total_samples"], 1)
        self.assertEqual(manifest["skipped_negative"], 1)
        self.assertEqual(manifest["skipped_low_confidence"], 1)

        split1 = lp.build_split_manifest()
        split2 = lp.build_split_manifest()
        self.assertEqual(split1["split_hash"], split2["split_hash"])
        self.assertEqual(split1["train_count"] + split1["val_count"], 1)

        rows = [json.loads(x) for x in lp.DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(rows[0]["objective"], "next_token_grounded_pair")
        self.assertIn(rows[0]["split"], {"train", "val"})

    def test_student_train_generate_and_checkpoint_roundtrip(self) -> None:
        records = []
        for i in range(60):
            records.append({
                "conversation_id": f"conv-{i}",
                "query": f"status query {i}",
                "route": "status",
                "response_class": "self_status",
                "provenance_verdict": "grounded_tool_data",
                "confidence": 0.95,
                "final_answer": f"System status variant {i} remains healthy and stable.",
                "meaning_frame": {"lead": "Status healthy"},
                "timestamp": float(i + 1),
            })

        manifest = lp.build_dataset(records)
        lp.build_split_manifest()
        rows = [json.loads(x) for x in lp.DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]

        student = lp.PhaseCAdapterStudent()
        result = student.train(
            rows,
            tokenizer_strategy="bpe",
            dataset_hash=str(manifest.get("dataset_hash", "")),
            max_epochs=1,
        )
        self.assertTrue(result.get("trained", False))
        reply = student.generate_shadow(
            query="status now",
            response_class="self_status",
            prompt="CLASS:self_status\nQUERY:status now\nANSWER:",
        )
        self.assertTrue(isinstance(reply, str) and len(reply) > 0)

        self.assertTrue(student.save_checkpoint())
        restored = lp.PhaseCAdapterStudent()
        self.assertTrue(restored.load_checkpoint())
        self.assertTrue(restored.available)

    def test_harness_cycle_runs_and_surfaces_status(self) -> None:
        lp.CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(lp.CORPUS_PATH, "w", encoding="utf-8") as f:
            for i in range(45):
                rec = {
                    "example_id": f"lang_{i}",
                    "conversation_id": f"conv_{i}",
                    "query": f"what is status {i}",
                    "route": "status",
                    "response_class": "self_status",
                    "meaning_frame": {"lead": "Status lead"},
                    "grounding_payload": {"src": "test"},
                    "teacher_answer": "",
                    "final_answer": f"status answer {i} remains healthy and stable",
                    "provenance_verdict": "grounded_tool_data",
                    "user_feedback": "",
                    "confidence": 0.9,
                    "timestamp": float(i),
                }
                f.write(json.dumps(rec) + "\n")

        result = lp.phasec_harness.run_training_cycle()
        self.assertTrue(result.get("ok", False))
        status = lp.phasec_harness.get_status()
        self.assertIn("tokenizer", status)
        self.assertIn("dataset", status)
        self.assertIn("split", status)
        self.assertIn("student", status)
        self.assertIn("reset_aware_context", status)


if __name__ == "__main__":
    unittest.main()

