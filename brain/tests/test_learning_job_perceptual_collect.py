import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills.executors.base import PhaseExecutor, PhaseResult
from skills.executors.dispatcher import ExecutorDispatcher
from skills.executors.perceptual import (
    PerceptualAssessExecutor,
    PerceptualCollectExecutor,
    PerceptualVerifyExecutor,
)


class PerceptualCollectExecutorTests(unittest.TestCase):
    def test_collect_emotion_samples_reads_teacher_signal_count(self) -> None:
        result = PerceptualCollectExecutor._collect_emotion_samples(
            object(),
            {
                "distillation_stats": {
                    "teachers": {
                        "wav2vec2_emotion": {"total": 37},
                    }
                }
            },
        )
        self.assertEqual(result.metric_updates["emotion_samples"], 37.0)

    def test_collect_speaker_samples_counts_single_embedding_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            speakers_path = os.path.join(tmp, "speakers.json")
            with open(speakers_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "David": {"embedding": [0.1, 0.2, 0.3]},
                        "Sarah": {"embedding": [0.4, 0.5, 0.6]},
                    },
                    f,
                )

            real_expanduser = os.path.expanduser

            def fake_expanduser(path: str) -> str:
                if path == "~/.jarvis/speakers.json":
                    return speakers_path
                return real_expanduser(path)

            with patch("os.path.expanduser", side_effect=fake_expanduser):
                result = PerceptualCollectExecutor._collect_speaker_samples(
                    object(),
                    {"distillation_stats": {"teachers": {}}},
                )

        self.assertEqual(result.metric_updates["speaker_samples"], 2.0)

    def test_collect_speaker_samples_prefers_live_teacher_signal_total(self) -> None:
        result = PerceptualCollectExecutor._collect_speaker_samples(
            object(),
            {
                "distillation_stats": {
                    "teachers": {
                        "ecapa_tdnn": {"total": 24},
                    }
                }
            },
        )
        self.assertEqual(result.metric_updates["speaker_samples"], 24.0)

    def test_collect_run_uses_metric_contract_not_skill_id(self) -> None:
        executor = PerceptualCollectExecutor()
        job = SimpleNamespace(
            skill_id="custom_collect_v1",
            phase="collect",
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
            ]},
        )
        result = executor.run(
            job,
            {
                "distillation_stats": {
                    "teachers": {
                        "wav2vec2_emotion": {"total": 19},
                    }
                }
            },
        )
        self.assertEqual(result.metric_updates["emotion_samples"], 19.0)

    def test_assess_run_uses_gate_contract_not_skill_id(self) -> None:
        assess = PerceptualAssessExecutor()
        job = SimpleNamespace(
            skill_id="custom_assess_v1",
            gates={"hard": [{"id": "gate:emotion_model_available"}]},
            evidence={"required": []},
            data={},
        )
        result = assess.run(
            job,
            {
                "emotion_classifier": SimpleNamespace(
                    _model_healthy=True,
                    _gpu_available=False,
                    _health_reason="",
                )
            },
        )
        self.assertEqual(result.gate_updates[0]["id"], "gate:emotion_model_available")
        self.assertEqual(result.gate_updates[0]["state"], "pass")

    def test_dispatcher_preserves_higher_existing_counter(self) -> None:
        class _FakeExecutor(PhaseExecutor):
            capability_type = "perceptual"
            phase = "collect"

            def run(self, job, ctx):
                return PhaseResult(progressed=True, metric_updates={"emotion_samples": 10.0})

        job = SimpleNamespace(
            capability_type="perceptual",
            phase="collect",
            status="active",
            job_id="job-1",
            data={"counters": {"emotion_samples": 12.0}},
            evidence={"latest": None},
            failure={},
        )
        fake_orch = SimpleNamespace(store=SimpleNamespace(save=lambda job: None))
        dispatcher = ExecutorDispatcher([_FakeExecutor()])

        with patch("skills.job_runner.try_auto_advance", return_value=False):
            dispatcher.tick_one_job(job, {}, fake_orch, registry=object())

        self.assertEqual(job.data["counters"]["emotion_samples"], 12.0)

    def test_verify_emotion_uses_classifier_health_flags(self) -> None:
        result = PerceptualVerifyExecutor._verify_emotion(
            object(),
            {
                "emotion_classifier": SimpleNamespace(
                    _model_healthy=True,
                    _gpu_available=False,
                    _health_reason="",
                )
            },
        )
        self.assertTrue(all(test["passed"] for test in result))

    def test_matrix_verify_uses_required_evidence_contract_not_skill_id(self) -> None:
        executor = PerceptualVerifyExecutor()
        job = SimpleNamespace(
            skill_id="custom_verify_v1",
            matrix_protocol=True,
            evidence={"required": ["test:emotion_accuracy_min", "test:emotion_confusion_matrix_ok"]},
            data={},
        )
        result = executor._run_matrix_verification(
            job,
            {
                "emotion_classifier": SimpleNamespace(
                    _model_healthy=True,
                    _gpu_available=False,
                    _health_reason="",
                )
            },
        )
        self.assertTrue(result.progressed)
        self.assertEqual(result.evidence["result"], "pass")
        self.assertTrue(any(t["name"] == "test:emotion_accuracy_min" for t in result.evidence["tests"]))


if __name__ == "__main__":
    unittest.main()
