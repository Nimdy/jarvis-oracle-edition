"""Regression tests for strict recent learning/research introspection answers."""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class IntrospectionRecentLearningTests(unittest.TestCase):
    @staticmethod
    def _load_introspection_tool():
        fake_engine_module = types.ModuleType("consciousness.engine")
        fake_engine_module.ConsciousnessEngine = object
        with patch.dict(sys.modules, {"consciousness.engine": fake_engine_module}):
            sys.modules.pop("tools.introspection_tool", None)
            return importlib.import_module("tools.introspection_tool")

    def test_scholarly_query_returns_verified_peer_reviewed_source_without_doi_by_default(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_source = SimpleNamespace(
            studied=True,
            studied_at=now - 30,
            retrieved_at=now - 60,
            title="Integrated world modeling theory expanded",
            venue="Frontiers in Computational Neuroscience",
            year=2022,
            doi="10.3389/fncom.2022.642397",
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=40: [fake_source])
        fake_library.classify_effective_source_type = lambda src: "peer_reviewed"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last scientific journal you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("Integrated world modeling theory expanded", answer)
        self.assertNotIn("10.3389/fncom.2022.642397", answer)
        self.assertNotIn("machine learning ethics", answer.lower())

    def test_scholarly_query_includes_doi_when_explicitly_requested(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_source = SimpleNamespace(
            studied=True,
            studied_at=now - 30,
            retrieved_at=now - 60,
            title="Integrated world modeling theory expanded",
            venue="Frontiers in Computational Neuroscience",
            year=2022,
            doi="10.3389/fncom.2022.642397",
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=40: [fake_source])
        fake_library.classify_effective_source_type = lambda src: "peer_reviewed"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What is the DOI for the last peer-reviewed journal you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("Integrated world modeling theory expanded", answer)
        self.assertIn("10.3389/fncom.2022.642397", answer)

    def test_scholarly_query_respects_stored_doi_include_preference(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_source = SimpleNamespace(
            studied=True,
            studied_at=now - 30,
            retrieved_at=now - 60,
            title="Integrated world modeling theory expanded",
            venue="Frontiers in Computational Neuroscience",
            year=2022,
            doi="10.3389/fncom.2022.642397",
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=40: [fake_source])
        fake_library.classify_effective_source_type = lambda src: "peer_reviewed"

        fake_pref = SimpleNamespace(
            payload="User response format preference: include DOI by default",
            tags=("user_preference", "response_style"),
            timestamp=now - 5,
        )
        fake_memory = types.ModuleType("memory.storage")
        fake_memory.memory_storage = SimpleNamespace(
            get_by_tag=lambda tag: [fake_pref] if tag == "response_style" else [],
        )

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library, "memory.storage": fake_memory}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last scientific journal you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("Integrated world modeling theory expanded", answer)
        self.assertIn("10.3389/fncom.2022.642397", answer)

    def test_explicit_doi_query_overrides_stored_omit_preference(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_source = SimpleNamespace(
            studied=True,
            studied_at=now - 30,
            retrieved_at=now - 60,
            title="Integrated world modeling theory expanded",
            venue="Frontiers in Computational Neuroscience",
            year=2022,
            doi="10.3389/fncom.2022.642397",
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=40: [fake_source])
        fake_library.classify_effective_source_type = lambda src: "peer_reviewed"

        fake_pref = SimpleNamespace(
            payload="User response format preference: omit DOI unless I ask",
            tags=("user_preference", "response_style"),
            timestamp=now - 5,
        )
        fake_memory = types.ModuleType("memory.storage")
        fake_memory.memory_storage = SimpleNamespace(
            get_by_tag=lambda tag: [fake_pref] if tag == "response_style" else [],
        )

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library, "memory.storage": fake_memory}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What is the DOI for the last peer-reviewed journal you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("10.3389/fncom.2022.642397", answer)

    def test_article_query_uses_scholarly_lane_and_ignores_internal_memory_source(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        internal_recent = SimpleNamespace(
            studied=True,
            studied_at=now - 10,
            retrieved_at=now - 20,
            title="User corrected a Jarvis response",
            source_type="url",
            provider="memory",
            doi="",
        )
        scholarly_older = SimpleNamespace(
            studied=True,
            studied_at=now - 120,
            retrieved_at=now - 180,
            title="Human values are crucial to human decision-making",
            venue="AAAI Conference on Artificial Intelligence",
            year=2023,
            doi="10.1609/aaai.v38i18.29970",
            source_type="peer_reviewed",
            provider="semantic_scholar",
        )

        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(
            get_recent=lambda limit=40: [internal_recent, scholarly_older]
        )
        fake_library.classify_effective_source_type = (
            lambda src: "internal_signal"
            if getattr(src, "provider", "") in {"memory", "introspection"}
            else ("peer_reviewed" if getattr(src, "doi", "") else "web")
        )

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last article you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("Human values are crucial to human decision-making", answer)
        self.assertNotIn("User corrected a Jarvis response", answer)

    def test_recent_research_query_ignores_internal_memory_sources(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        internal_recent = SimpleNamespace(
            studied=True,
            studied_at=now - 5,
            retrieved_at=now - 10,
            title="User corrected a Jarvis response",
            source_type="url",
            provider="memory",
        )
        external_older = SimpleNamespace(
            studied=True,
            studied_at=now - 90,
            retrieved_at=now - 120,
            title="Reliable external source",
            source_type="url",
            provider="semantic_scholar",
        )

        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(
            get_recent=lambda limit=20: [internal_recent, external_older]
        )
        fake_library.classify_effective_source_type = (
            lambda src: "internal_signal"
            if getattr(src, "provider", "") in {"memory", "introspection"}
            else "web"
        )

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last thing you researched?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("Reliable external source", answer)
        self.assertNotIn("User corrected a Jarvis response", answer)

    def test_general_learning_query_prefers_latest_autonomy_research_record(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_orch = SimpleNamespace(
            get_status=lambda: {
                "recent_learnings": [{
                    "timestamp": now - 10,
                    "question": "How to improve philosophical in AI systems?",
                    "summary": "Found 5 scholarly results with 2 content-rich sources.",
                    "tool": "academic_search",
                }]
            }
        )
        older_source = SimpleNamespace(
            studied=True,
            studied_at=now - 120,
            retrieved_at=now - 180,
            title="Older studied source",
            source_type="codebase",
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=20: [older_source])
        fake_library.classify_effective_source_type = lambda src: "codebase"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=fake_orch), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last thing you learned?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("How to improve philosophical in AI systems?", answer)
        self.assertIn("academic_search", answer)

    def test_research_query_variant_what_research_did_you_do_recently_matches(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_orch = SimpleNamespace(
            get_status=lambda: {
                "recent_learnings": [{
                    "timestamp": now - 10,
                    "question": "What concrete research plan reduces routing misses?",
                    "summary": "Compared strict-native route guards against grounding misses.",
                    "tool": "academic_search",
                }]
            }
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=20: [])
        fake_library.classify_effective_source_type = lambda src: "unknown"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=fake_orch), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What research did you do recently?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("What concrete research plan reduces routing misses?", answer)
        self.assertIn("academic_search", answer)

    def test_recent_learning_query_fails_closed_without_records(self) -> None:
        it = self._load_introspection_tool()

        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=20: [])
        fake_library.classify_effective_source_type = lambda src: "unknown"
        fake_friction = types.ModuleType("autonomy.friction_miner")
        fake_friction._PERSISTENCE_PATH = f"/tmp/jarvis_test_no_friction_{time.time_ns()}.jsonl"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=None), \
             patch.dict(sys.modules, {"library.source": fake_library, "autonomy.friction_miner": fake_friction}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last thing you learned?",
            )

        self.assertEqual(answer, "I don't have a verified recent learning record yet.")

    def test_conversational_learning_uses_memory_not_completed_learning_job(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_memory_record = SimpleNamespace(
            type="conversation",
            provenance="conversation",
            tags=("conversation",),
            payload="User corrected that recent-learning answers must separate conversations from skill jobs.",
            timestamp=now - 30,
        )
        fake_memory = types.ModuleType("memory.storage")
        fake_memory.memory_storage = SimpleNamespace(get_all=lambda: [fake_memory_record])

        completed_job = SimpleNamespace(
            skill_id="data_processing_v1",
            status="completed",
            phase="register",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 5)),
        )
        fake_job_orch = SimpleNamespace(
            store=SimpleNamespace(load_all=lambda: [completed_job]),
        )
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=20: [])
        fake_library.classify_effective_source_type = lambda src: "unknown"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_job_orch), \
             patch.dict(sys.modules, {"library.source": fake_library, "memory.storage": fake_memory}):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What have you learned from recent conversations?",
            )

        self.assertIsNotNone(answer)
        self.assertIn("recent-learning answers must separate conversations", answer)
        self.assertNotIn("data_processing_v1", answer)
        self.assertNotIn("learning-job", answer)

    def test_general_recent_learning_does_not_fallback_to_completed_learning_job(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        completed_job = SimpleNamespace(
            skill_id="data_processing_v1",
            status="completed",
            phase="register",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 5)),
        )
        fake_job_orch = SimpleNamespace(
            store=SimpleNamespace(load_all=lambda: [completed_job]),
        )
        fake_memory = types.ModuleType("memory.storage")
        fake_memory.memory_storage = SimpleNamespace(get_all=lambda: [])
        fake_library = types.ModuleType("library.source")
        fake_library.source_store = SimpleNamespace(get_recent=lambda limit=20: [])
        fake_library.classify_effective_source_type = lambda src: "unknown"
        fake_friction = types.ModuleType("autonomy.friction_miner")
        fake_friction._PERSISTENCE_PATH = f"/tmp/jarvis_test_no_friction_{time.time_ns()}.jsonl"

        with patch.object(it, "_get_autonomy_orchestrator_instance", return_value=None), \
             patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_job_orch), \
             patch.dict(sys.modules, {
                 "library.source": fake_library,
                 "memory.storage": fake_memory,
                 "autonomy.friction_miner": fake_friction,
             }):
            answer = it.get_grounded_recent_learning_answer(
                engine=SimpleNamespace(),
                query="What was the last thing you learned?",
            )

        self.assertEqual(answer, "I don't have a verified recent learning record yet.")

    def test_learning_job_status_query_reports_collect_blocker_and_user_input(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_job = SimpleNamespace(
            job_id="job-emotion-1",
            skill_id="emotion_detection_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 90)),
            matrix_protocol=True,
            protocol_id="SK-002",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"emotion_samples": 12}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
            ]},
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            _registry=object(),
        )

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", return_value=(False, ["metric:emotion_samples>=30"])):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="Why is your emotion detection learning job stuck in collect?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_status")
        self.assertEqual(record["skill_id"], "emotion_detection_v1")
        self.assertEqual(record["phase"], "collect")
        self.assertEqual(record["metric_name"], "emotion_samples")
        self.assertEqual(record["current_metric"], 12.0)
        self.assertEqual(record["target_metric"], 30.0)
        self.assertTrue(record["user_input_needed"])
        self.assertTrue(record["suggested_user_inputs"])

    def test_learning_job_status_query_matches_need_from_me_wording(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_job = SimpleNamespace(
            job_id="job-speaker-1",
            skill_id="speaker_identification_v1",
            capability_type="perceptual",
            status="active",
            phase="train",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 45)),
            matrix_protocol=False,
            protocol_id="",
            claimability_status="unverified",
            artifacts=[{"id": "a1"}],
            evidence={"history": []},
            events=[],
            data={"counters": {"speaker_samples": 24}},
            plan={"phases": [
                {"name": "train", "exit_conditions": ["artifact:train_tick"]},
            ]},
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            _registry=object(),
        )

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", return_value=(False, ["artifact:train_tick"])):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="What do you need from me for speaker identification?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_status")
        self.assertEqual(record["skill_id"], "speaker_identification_v1")
        self.assertEqual(record["phase"], "train")

    def test_learning_job_help_summary_handles_plural_generic_query(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        emotion_job = SimpleNamespace(
            job_id="job-emotion-1",
            skill_id="emotion_detection_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 30)),
            matrix_protocol=True,
            protocol_id="SK-002",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"emotion_samples": 12}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
            ]},
        )
        speaker_job = SimpleNamespace(
            job_id="job-speaker-1",
            skill_id="speaker_identification_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 20)),
            matrix_protocol=False,
            protocol_id="",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"speaker_samples": 8}},
            plan={"phases": [
                {"name": "collect", "exit_conditions": ["metric:speaker_samples>=20"]},
            ]},
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [emotion_job, speaker_job],
            _registry=object(),
        )

        def _fake_check_exit(job, registry, exit_conditions):
            if job.skill_id == "emotion_detection_v1":
                return False, ["metric:emotion_samples>=30"]
            return False, ["metric:speaker_samples>=20"]

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", side_effect=_fake_check_exit):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="Do you need anything from me to help you finish those learning jobs?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_help_summary")
        self.assertEqual(record["active_job_count"], 2)
        self.assertTrue(record["user_input_needed"])
        self.assertEqual(len(record["jobs"]), 2)
        self.assertTrue(record["suggested_user_inputs"])

    def test_learning_job_status_uses_declarative_guided_collect_hints(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_job = SimpleNamespace(
            job_id="job-custom-1",
            skill_id="teacher_alignment_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 10)),
            matrix_protocol=False,
            protocol_id="",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"teacher_samples": 6}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:teacher_samples>=12"]},
                ],
                "guided_collect": {
                    "user_input_hints": [
                        "Declarative hint for {skill_name} using {metric_label}.{remaining_hint}",
                    ],
                },
            },
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            _registry=object(),
        )

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", return_value=(False, ["metric:teacher_samples>=12"])):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="What do you need from me for teacher alignment?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_status")
        self.assertTrue(record["suggested_user_inputs"])
        self.assertIn("Declarative hint", record["suggested_user_inputs"][0])

    def test_learning_job_status_can_use_protocol_owned_collect_hints(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_job = SimpleNamespace(
            job_id="job-matrix-emotion-1",
            skill_id="emotion_detection_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 10)),
            matrix_protocol=True,
            protocol_id="SK-002",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"emotion_samples": 12}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
                ],
            },
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            _registry=object(),
        )

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", return_value=(False, ["metric:emotion_samples>=30"])):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="What do you need from me for emotion detection?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_status")
        self.assertTrue(record["suggested_user_inputs"])
        self.assertIn("Self-labeled examples help most", record["suggested_user_inputs"][1])

    def test_learning_job_status_does_not_request_user_input_for_autonomous_collect(self) -> None:
        it = self._load_introspection_tool()

        now = time.time()
        fake_job = SimpleNamespace(
            job_id="job-auto-collect-1",
            skill_id="perception_distilled_v1",
            capability_type="perceptual",
            status="active",
            phase="collect",
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 10)),
            matrix_protocol=False,
            protocol_id="",
            claimability_status="unverified",
            artifacts=[],
            evidence={"history": []},
            events=[],
            data={"counters": {"teacher_samples": 6}},
            plan={
                "phases": [
                    {"name": "collect", "exit_conditions": ["metric:teacher_samples>=12"]},
                ],
            },
        )
        fake_orch = SimpleNamespace(
            get_active_jobs=lambda: [fake_job],
            _registry=object(),
        )

        with patch.object(it, "_get_learning_job_orchestrator_instance", return_value=fake_orch), \
             patch("skills.job_eval.check_exit_conditions", return_value=(False, ["metric:teacher_samples>=12"])):
            record = it.get_grounded_learning_job_status_record(
                engine=SimpleNamespace(),
                query="What do you need from me for perceptual distillation?",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["kind"], "learning_job_status")
        self.assertFalse(record["user_input_needed"])
        self.assertEqual(record["suggested_user_inputs"], [])


if __name__ == "__main__":
    unittest.main()
