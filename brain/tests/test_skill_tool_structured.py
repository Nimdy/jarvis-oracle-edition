import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class SkillToolStructuredTests(unittest.TestCase):
    def test_protocol_layer_parses_labeled_collect_submission(self) -> None:
        from skills.verification_protocols import parse_collect_submission

        parsed = parse_collect_submission(
            {"parser": "labeled_text"},
            "happy: I feel great today.",
        )

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["label"], "happy")
        self.assertEqual(parsed["text"], "I feel great today.")

    def test_protocol_layer_builds_collect_artifact_from_schema(self) -> None:
        from skills.verification_protocols import build_collect_artifact

        artifact = build_collect_artifact(
            {
                "session_id": "sess-proto",
                "artifact_type": "protocol_collect_sample",
                "artifact_schema": {"label_field": "class_label", "text_field": "utterance"},
            },
            speaker="David",
            emotion="calm",
            conversation_id="conv-proto",
            metric_name="teacher_samples",
            captured_index=0,
            parsed_sample={"label": "calm", "text": "this is a sample"},
        )

        self.assertEqual(artifact["type"], "protocol_collect_sample")
        self.assertEqual(artifact["details"]["class_label"], "calm")
        self.assertEqual(artifact["details"]["utterance"], "this is a sample")

    def test_returns_already_verified_structured_result(self) -> None:
        from tools import skill_tool

        fake_registry = SimpleNamespace(
            get=lambda skill_id: SimpleNamespace(
                skill_id=skill_id,
                name="Singing",
                status="verified",
                capability_type="procedural",
            )
        )
        fake_orch = SimpleNamespace(get_active_jobs=lambda: [])
        fake_resolver = types.ModuleType("skills.resolver")
        fake_resolver.is_generic_fallback_resolution = lambda resolution: False
        fake_resolver.resolve_skill = lambda text: SimpleNamespace(
            skill_id="singing_v1",
            name="Singing",
            capability_type="procedural",
            required_evidence=[],
            notes="",
            risk_level="low",
            default_phases=["assess"],
            hard_gates=[],
        )

        with patch.object(skill_tool, "_skill_registry", fake_registry), \
             patch.object(skill_tool, "_learning_job_orch", fake_orch), \
             patch.dict("sys.modules", {"skills.resolver": fake_resolver}):
            result = skill_tool.handle_skill_request_structured("learn to sing")

        self.assertEqual(result["outcome"], "already_verified")
        self.assertEqual(result["skill_id"], "singing_v1")
        self.assertEqual(result["status"], "verified")
        self.assertIn("verified skill", result["message"])

    def test_returns_job_started_structured_result(self) -> None:
        from tools import skill_tool

        fake_registry = SimpleNamespace(
            get=lambda skill_id: None,
            register=lambda record: None,
        )
        fake_job = SimpleNamespace(
            job_id="job-42",
            status="learning",
            phase="assess",
            created_at=1.0,
            events=[],
            protocol_id="",
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [],
            create_job=lambda **kwargs: fake_job,
            store=SimpleNamespace(save=lambda job: None),
        )
        fake_resolver = types.ModuleType("skills.resolver")
        fake_resolver.is_generic_fallback_resolution = lambda resolution: False
        fake_resolver.resolve_skill = lambda text: SimpleNamespace(
            skill_id="face_tracking_v1",
            name="Face tracking",
            capability_type="perceptual",
            required_evidence=["verify_face_tracking"],
            notes="Track faces robustly.",
            risk_level="medium",
            default_phases=["assess", "verify"],
            hard_gates=[],
        )
        fake_registry_module = types.ModuleType("skills.registry")

        class _SkillRecord:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_registry_module.SkillRecord = _SkillRecord

        with patch.object(skill_tool, "_skill_registry", fake_registry), \
             patch.object(skill_tool, "_learning_job_orch", fake_orch), \
             patch.dict("sys.modules", {
                 "skills.resolver": fake_resolver,
                 "skills.registry": fake_registry_module,
             }):
            result = skill_tool.handle_skill_request_structured("learn face tracking")

        self.assertEqual(result["outcome"], "job_started")
        self.assertEqual(result["skill_id"], "face_tracking_v1")
        self.assertEqual(result["job_id"], "job-42")
        self.assertEqual(result["phase"], "assess")
        self.assertEqual(result["risk_level"], "medium")

    def test_guided_collect_starts_for_active_collect_job(self) -> None:
        from tools import skill_tool

        fake_job = SimpleNamespace(
            job_id="job-emotion-1",
            skill_id="emotion_detection_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            executor_state={},
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"emotion_samples": 12}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
            ]},
        )
        fake_registry = SimpleNamespace(
            get=lambda skill_id: SimpleNamespace(
                skill_id=skill_id,
                name="Emotion Detection",
                status="learning",
                capability_type="perceptual",
            )
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            store=SimpleNamespace(save=lambda job: None),
        )
        fake_resolver = types.ModuleType("skills.resolver")
        fake_resolver.is_generic_fallback_resolution = lambda resolution: False
        fake_resolver.resolve_skill = lambda text: SimpleNamespace(
            skill_id="emotion_detection_v1",
            name="Emotion Detection",
            capability_type="perceptual",
            required_evidence=[],
            notes="",
            risk_level="low",
            default_phases=["assess", "collect"],
            hard_gates=[],
        )

        with patch.object(skill_tool, "_skill_registry", fake_registry), \
             patch.object(skill_tool, "_learning_job_orch", fake_orch), \
             patch.dict("sys.modules", {"skills.resolver": fake_resolver}):
            result = skill_tool.handle_skill_request_structured(
                "help train emotion detection",
                speaker="David",
            )

        self.assertEqual(result["outcome"], "guided_collect_started")
        self.assertEqual(result["skill_id"], "emotion_detection_v1")
        self.assertIn("Training mode", result["prompt"])
        self.assertTrue(fake_job.executor_state["guided_collect"]["active"])
        self.assertEqual(fake_job.executor_state["guided_collect"]["metric_name"], "emotion_samples")

    def test_guided_collect_turn_records_sample_and_advances(self) -> None:
        from tools import skill_tool

        artifacts: list[dict] = []
        fake_job = SimpleNamespace(
            job_id="job-emotion-1",
            skill_id="emotion_detection_v1",
            status="active",
            phase="collect",
            executor_state={
                "guided_collect": {
                    "active": True,
                    "session_id": "sess-1",
                    "speaker": "David",
                    "mode": "open_labeled",
                    "metric_name": "emotion_samples",
                    "prompt": "Training mode: provide labeled samples.",
                }
            },
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"emotion_samples": 12}},
        )

        def _add_artifact(job, artifact):
            artifacts.append(artifact)

        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            add_artifact=_add_artifact,
            store=SimpleNamespace(save=lambda job: None),
        )

        with patch.object(skill_tool, "_learning_job_orch", fake_orch):
            result = skill_tool.consume_guided_collect_turn(
                speaker="David",
                user_text="happy: I feel great today.",
                emotion="happy",
                conversation_id="conv-1",
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["outcome"], "guided_collect_continue")
        self.assertEqual(fake_job.data["counters"]["emotion_samples"], 13.0)
        self.assertEqual(artifacts[0]["details"]["label"], "happy")

    def test_guided_collect_uses_declarative_plan_config(self) -> None:
        from tools import skill_tool

        fake_job = SimpleNamespace(
            job_id="job-custom-1",
            skill_id="custom_metric_v1",
            status="active",
            phase="collect",
            executor_state={},
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"teacher_samples": 4}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:teacher_samples>=10"]},
                ],
                "guided_collect": {
                    "mode": "open_labeled",
                    "metric_name": "teacher_samples",
                    "prompt_template": "Declarative prompt for {skill_name} on {metric_label}.{remaining_hint}",
                },
            },
        )

        result = skill_tool._start_guided_collect_for_job(fake_job, speaker="David")

        self.assertEqual(result["outcome"], "guided_collect_started")
        self.assertIn("Declarative prompt", result["prompt"])
        self.assertIn("teacher samples", result["prompt"])

    def test_guided_collect_uses_protocol_owned_matrix_config(self) -> None:
        from tools import skill_tool

        fake_job = SimpleNamespace(
            job_id="job-matrix-emotion-1",
            skill_id="emotion_detection_v1",
            status="active",
            phase="collect",
            matrix_protocol=True,
            protocol_id="SK-002",
            executor_state={},
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"emotion_samples": 12}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
                ],
            },
        )

        result = skill_tool._start_guided_collect_for_job(fake_job, speaker="David")

        self.assertEqual(result["outcome"], "guided_collect_started")
        self.assertIn("happy:", result["prompt"])
        self.assertEqual(fake_job.executor_state["guided_collect"]["parser"], "labeled_text")
        self.assertEqual(fake_job.executor_state["guided_collect"]["artifact_type"], "guided_collect_sample")

    def test_guided_collect_not_available_for_autonomous_collect_job(self) -> None:
        from tools import skill_tool

        fake_job = SimpleNamespace(
            job_id="job-auto-collect-1",
            skill_id="perception_distilled_v1",
            status="active",
            phase="collect",
            executor_state={},
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"teacher_samples": 4}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:teacher_samples>=10"]},
                ],
            },
        )

        result = skill_tool._start_guided_collect_for_job(fake_job, speaker="David")

        self.assertEqual(result["outcome"], "guided_collect_not_available")
        self.assertIn("collecting evidence autonomously", result["message"])

    def test_guided_collect_artifact_shape_can_come_from_session_schema(self) -> None:
        from tools import skill_tool

        artifacts: list[dict] = []
        fake_job = SimpleNamespace(
            job_id="job-custom-schema-1",
            skill_id="custom_metric_v1",
            status="active",
            phase="collect",
            executor_state={
                "guided_collect": {
                    "active": True,
                    "session_id": "sess-schema",
                    "speaker": "David",
                    "parser": "labeled_text",
                    "artifact_type": "protocol_collect_sample",
                    "artifact_schema": {"label_field": "class_label", "text_field": "utterance"},
                    "metric_name": "teacher_samples",
                    "prompt": "Use label: example",
                }
            },
            updated_at="2026-03-20T10:00:00Z",
            events=[],
            data={"counters": {"teacher_samples": 4}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:teacher_samples>=10"]},
            ]},
        )

        def _add_artifact(job, artifact):
            artifacts.append(artifact)

        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            add_artifact=_add_artifact,
            store=SimpleNamespace(save=lambda job: None),
        )

        with patch.object(skill_tool, "_learning_job_orch", fake_orch):
            result = skill_tool.consume_guided_collect_turn(
                speaker="David",
                user_text="calm: this is a sample",
                emotion="calm",
                conversation_id="conv-schema",
            )

        self.assertEqual(result["outcome"], "guided_collect_continue")
        self.assertEqual(artifacts[0]["type"], "protocol_collect_sample")
        self.assertEqual(artifacts[0]["details"]["class_label"], "calm")
        self.assertEqual(artifacts[0]["details"]["utterance"], "this is a sample")


if __name__ == "__main__":
    unittest.main()
