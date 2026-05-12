import time
import unittest

from reasoning.bounded_response import (
    MAX_ARTICULATE_CHARS,
    MAX_ARTICULATE_FACTS,
    MAX_ARTICULATE_SENTENCES,
    MeaningFrame,
    articulate_meaning_frame,
    build_meaning_frame,
)


class BoundedResponseTests(unittest.TestCase):
    def test_self_status_frame_articulates_operational_sections(self) -> None:
        frame = build_meaning_frame(
            response_class="self_status",
            grounding_payload=(
                "=== Current Activity [live] ===\n"
                "State: Processing request\n"
                "Status: active\n"
                "Detail: tool: STATUS\n"
                "Pipeline: Wake[done] -> Listen[done]\n\n"
                "=== Background Operations [none active] ===\n"
                "All background subsystems idle\n\n"
                "=== Operating Mode [live] ===\n"
                "Mode: passive\n"
                "Dwell: 0s in current mode\n"
            ),
        )

        reply = articulate_meaning_frame(frame)
        self.assertIn("handling a request", reply)
        self.assertIn("passive mode", reply)
        self.assertNotIn("I am operational", reply)
        self.assertNotIn("I am processing", reply)

    def test_recent_research_frame_includes_verified_facts(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "scholarly_source",
                "timestamp": time.time() - 120,
                "title": "Integrated world modeling theory expanded",
                "venue": "Frontiers in Computational Neuroscience",
                "year": 2022,
                "doi": "10.3389/fncom.2022.642397",
                "include_doi": True,
            },
        )

        self.assertEqual(frame.response_class, "recent_research")
        self.assertIn("peer-reviewed source", frame.lead)
        self.assertTrue(any("Integrated world modeling theory expanded" in fact for fact in frame.facts))
        self.assertTrue(any("DOI: 10.3389/fncom.2022.642397" == fact for fact in frame.facts))

        reply = articulate_meaning_frame(frame)
        self.assertIn("Integrated world modeling theory expanded", reply)
        self.assertIn("10.3389/fncom.2022.642397", reply)

    def test_recent_research_frame_omits_doi_by_default(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "scholarly_source",
                "timestamp": time.time() - 90,
                "title": "Grounded introspection with strict routing",
                "venue": "Journal of Reliable AI",
                "year": 2025,
                "doi": "10.1000/example",
            },
        )

        self.assertFalse(any(fact.startswith("DOI:") for fact in frame.facts))
        reply = articulate_meaning_frame(frame)
        self.assertNotIn("10.1000/example", reply)

    def test_missing_recent_learning_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_learning",
            grounding_payload={"kind": "missing_learning", "timestamp": 0.0},
        )

        self.assertEqual(frame.missing_reason, "missing_learning")
        self.assertEqual(
            articulate_meaning_frame(frame),
            "I don't have a verified recent learning record yet.",
        )

    def test_memory_recall_frame_uses_mode_specific_lead(self) -> None:
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={
                "mode": "search",
                "memory_context": "Memory recall:\nDavid likes low latency.\nHe asked about calibration yesterday.",
            },
        )

        self.assertEqual(frame.metadata.get("mode"), "search")
        self.assertIn("memory details", frame.lead)
        reply = articulate_meaning_frame(frame)
        self.assertIn("David likes low latency.", reply)

    def test_identity_check_match_frame_confirms_identity(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_match",
                "check_name": "David",
                "matched_modalities": ["voice (69%)", "face (91%)"],
            },
        )

        self.assertIn("Yes, this is David.", frame.lead)
        reply = articulate_meaning_frame(frame)
        self.assertIn("Confirmed by voice (69%), face (91%)", reply)

    def test_capability_status_job_started_frame_keeps_registry_facts(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "job_started",
                "message": "I started a learning job for 'Face tracking'.",
                "skill_name": "Face tracking",
                "status": "learning",
                "capability_type": "perceptual",
                "job_id": "job-123",
                "phase": "assess",
                "risk_level": "medium",
                "protocol_id": "SK-002",
            },
        )

        self.assertIn("I started a learning job", frame.lead)
        self.assertTrue(any(fact == "Job: job-123" for fact in frame.facts))
        self.assertTrue(any(fact == "Protocol: SK-002" for fact in frame.facts))

    def test_system_explanation_frame_uses_grounded_lines(self) -> None:
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "System explanation",
                "query": "Explain your architecture",
                "body": "Memory routes through storage.\nRetrieval uses a closed-loop telemetry log.\nBounded responses now render exact classes.",
            },
        )

        self.assertEqual(frame.lead, "System explanation:")
        self.assertEqual(frame.metadata.get("title"), "System explanation")
        reply = articulate_meaning_frame(frame)
        self.assertIn("Memory routes through storage.", reply)
        self.assertIn("Bounded responses now render exact classes.", reply)

    def test_runtime_bridge_policy_is_outside_bounded_articulation(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=(
                "=== Consciousness Metrics ===\n"
                "Stage: integrative\n"
                "Confidence: 0.80\n"
            ),
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("integrative", reply.lower())
        self.assertNotIn("runtime rollout", reply.lower())

    def test_self_introspection_memory_probe_prefers_memory_architecture_facts(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=(
                "=== Consciousness Metrics ===\n"
                "Stage: self_reflective\n"
                "Reasoning quality: 0.66\n\n"
                "=== Analytics ===\n"
                "Confidence: 0.81\n\n"
                "=== Memory ===\n"
                "Total memories: 417 (2 core)\n"
                "Average weight: 0.42\n\n"
                "=== Architecture & Storage ===\n"
                "Memory storage: in-memory dict + JSON persistence (~/.jarvis/memories.json)\n"
                "Memory search: sqlite-vec semantic embeddings + keyword matching\n"
                "Memory retrieval: hybrid search (semantic + keyword), scored by ranker NN\n"
            ),
            preferred_categories=["memory", "architecture"],
        )

        reply = articulate_meaning_frame(frame)

        self.assertIn("417", reply)
        self.assertIn("sqlite-vec", reply)
        self.assertNotIn("self_reflective stage", reply)
        self.assertNotIn("Reasoning quality", reply)

    def test_learning_job_status_frame_surfaces_blocker_and_user_input(self) -> None:
        frame = build_meaning_frame(
            response_class="learning_job_status",
            grounding_payload={
                "kind": "learning_job_status",
                "skill_id": "emotion_detection_v1",
                "status": "active",
                "phase": "collect",
                "artifact_count": 0,
                "evidence_count": 0,
                "phase_age_s": 90.0,
                "matrix_protocol": True,
                "protocol_id": "SK-002",
                "claimability_status": "unverified",
                "blocker_summary": "It is waiting on emotion_samples: 12/30 collected.",
                "metric_name": "emotion_samples",
                "current_metric": 12.0,
                "target_metric": 30.0,
                "suggested_user_inputs": [
                    "A short labeled calibration round would help most. This collect phase is waiting on emotion samples.",
                ],
            },
        )

        self.assertIn("emotion_detection_v1 is currently in collect phase.", frame.lead)
        self.assertTrue(any("Blocker: It is waiting on emotion_samples: 12/30 collected." == fact for fact in frame.facts))
        self.assertTrue(any("User input:" in fact for fact in frame.facts))
        reply = articulate_meaning_frame(frame)
        self.assertIn("Progress: emotion_samples 12/30", reply)

    def test_learning_job_help_summary_frame_surfaces_multiple_jobs(self) -> None:
        frame = build_meaning_frame(
            response_class="learning_job_status",
            grounding_payload={
                "kind": "learning_job_help_summary",
                "active_job_count": 2,
                "jobs": [
                    {
                        "skill_id": "emotion_detection_v1",
                        "phase": "collect",
                        "status": "active",
                        "blocker_summary": "It is waiting on emotion_samples: 12/30 collected.",
                    },
                    {
                        "skill_id": "speaker_identification_v1",
                        "phase": "collect",
                        "status": "active",
                        "blocker_summary": "It is waiting on speaker_samples: 8/20 collected.",
                    },
                ],
                "suggested_user_inputs": [
                    "A short labeled calibration round would help most. This collect phase is waiting on emotion samples.",
                ],
            },
        )

        self.assertIn("I can verify 2 active learning jobs right now.", frame.lead)
        self.assertTrue(any("emotion_detection_v1: phase=collect, status=active" == fact for fact in frame.facts))
        self.assertTrue(any("speaker_identification_v1 blocker:" in fact for fact in frame.facts))
        self.assertTrue(any("User input:" in fact for fact in frame.facts))


class SelfIntrospectionTests(unittest.TestCase):
    """Phase B.1: self_introspection frame builder + articulator."""

    RICH_PAYLOAD = (
        "=== Consciousness Metrics ===\n"
        "Stage: integrative\n"
        "Stage progression score: 6.20\n"
        "Observer awareness score: 0.98\n"
        "Reasoning quality: 0.85\n"
        "Confidence: 0.81\n"
        "System health flag: True\n\n"
        "=== Analytics ===\n"
        "Confidence: avg=0.81, trend=stable\n"
        "Reasoning: overall=0.85\n"
        "System health: healthy\n"
        "  tick_p95=2.3ms\n\n"
        "=== Memory ===\n"
        "Total memories: 489 (12 core)\n"
        "Average weight: 0.450\n"
        "Top memory themes: research(45), conversation(38), observation(21)\n\n"
        "=== Evolution Metrics ===\n"
        "Current stage: integrative\n"
        "Stage progression score: 6.20\n"
        "Emergent behavior count: 14\n\n"
        "=== Self-Modifications (Mutations) ===\n"
        "Total mutations applied: 87\n"
        "Last mutation: adjusted tick cadence\n"
    )

    def test_frame_parses_sections_and_ranks_facts(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )

        self.assertEqual(frame.response_class, "self_introspection")
        self.assertGreater(frame.fact_count, 0)
        self.assertGreater(frame.section_count, 0)
        self.assertGreater(frame.frame_confidence, 0.0)
        self.assertEqual(frame.parse_warnings, [])
        self.assertLessEqual(len(frame.facts), MAX_ARTICULATE_FACTS)
        self.assertIn("bounded_introspection", frame.safety_flags)

    def test_frame_ranked_facts_prioritize_current_state(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )

        first_fact = frame.facts[0]
        self.assertTrue(
            "stage" in first_fact.lower()
            or "confidence" in first_fact.lower()
            or "awareness" in first_fact.lower()
            or "reasoning" in first_fact.lower(),
            f"First fact should be current_state category, got: {first_fact}",
        )

    def test_articulation_produces_spoken_sentences(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )
        reply = articulate_meaning_frame(frame)

        self.assertIn("current measured state", reply.lower())
        sentence_count = reply.count(". ") + (1 if reply.endswith(".") else 0)
        self.assertLessEqual(sentence_count, MAX_ARTICULATE_SENTENCES)
        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)

    def test_articulation_contains_no_raw_kv(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )
        reply = articulate_meaning_frame(frame)

        self.assertNotIn("===", reply)
        self.assertNotIn("Stage progression score:", reply)

    def test_empty_payload_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload="",
        )

        self.assertEqual(frame.missing_reason, "no_introspection_facts_parsed")
        self.assertIn("fail_closed_when_missing", frame.safety_flags)
        self.assertEqual(frame.frame_confidence, 0.0)

        reply = articulate_meaning_frame(frame)
        self.assertIn("don't have detailed introspection data", reply)

    def test_malformed_payload_produces_warnings(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload="just some random text without sections",
        )

        self.assertIn("no_sections_parsed", frame.parse_warnings)
        self.assertFalse(frame.is_structurally_healthy)

    def test_output_caps_enforced(self) -> None:
        many_sections = ""
        for i in range(20):
            many_sections += f"=== Section {i} ===\nFact_{i}_A: value_{i}a\nFact_{i}_B: value_{i}b\n\n"

        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=many_sections,
        )
        reply = articulate_meaning_frame(frame)

        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)
        self.assertLessEqual(len(frame.facts), MAX_ARTICULATE_FACTS)

    def test_is_structurally_healthy_for_rich_data(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )
        self.assertTrue(frame.is_structurally_healthy)

    def test_fact_to_sentence_transforms(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=(
                "=== Consciousness Metrics ===\n"
                "Stage: integrative\n"
                "Confidence: 0.81\n"
            ),
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("integrative", reply)
        self.assertIn("0.81", reply)
        self.assertNotIn("Stage:", reply)

    def test_anti_confab_no_hedging(self) -> None:
        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=self.RICH_PAYLOAD,
        )
        reply = articulate_meaning_frame(frame)

        for phrase in ("I think", "I feel", "I believe", "this means", "perhaps"):
            self.assertNotIn(phrase, reply.lower())

    def test_meaning_frame_guardrail_fields_exist(self) -> None:
        frame = MeaningFrame(
            response_class="test",
            lead="test",
            frame_confidence=0.8,
            fact_count=5,
            section_count=3,
            parse_warnings=["minor_warning"],
        )
        d = frame.to_dict()
        self.assertEqual(d["frame_confidence"], 0.8)
        self.assertEqual(d["fact_count"], 5)
        self.assertEqual(d["section_count"], 3)
        self.assertEqual(d["parse_warnings"], ["minor_warning"])
        self.assertTrue(frame.is_structurally_healthy)

    def test_meaning_frame_unhealthy_with_critical_warning(self) -> None:
        frame = MeaningFrame(
            response_class="test",
            lead="test",
            frame_confidence=0.8,
            parse_warnings=["critical: section parse failed"],
        )
        self.assertFalse(frame.is_structurally_healthy)

    def test_meaning_frame_unhealthy_with_low_confidence(self) -> None:
        frame = MeaningFrame(
            response_class="test",
            lead="test",
            frame_confidence=0.1,
        )
        self.assertFalse(frame.is_structurally_healthy)


class RecentLearningArticulatorTests(unittest.TestCase):
    """Phase B.2: _articulate_recent_learning for 7 kind subtypes."""

    def test_scholarly_source_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "scholarly_source",
                "timestamp": time.time() - 300,
                "title": "Neural correlates of consciousness",
                "venue": "Nature Neuroscience",
                "year": 2024,
                "doi": "10.1038/s41593-024-1234",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("Neural correlates of consciousness", reply)
        self.assertIn("Nature Neuroscience", reply)
        self.assertNotIn("Title:", reply)
        self.assertNotIn("Venue:", reply)
        self.assertTrue(reply.endswith("."))

    def test_source_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_learning",
            grounding_payload={
                "kind": "source",
                "timestamp": time.time() - 60,
                "title": "Attention mechanism overview",
                "source_type": "document",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("Attention mechanism overview", reply)
        self.assertIn("document", reply)
        self.assertNotIn("Title:", reply)

    def test_autonomy_research_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "autonomy_research",
                "timestamp": time.time() - 600,
                "question": "How does episodic memory consolidation work?",
                "tool": "academic",
                "summary": "Found 3 relevant papers on hippocampal replay.",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("episodic memory consolidation", reply)
        self.assertIn("academic", reply)
        self.assertNotIn("Question:", reply)
        self.assertNotIn("Tool:", reply)

    def test_learning_job_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_learning",
            grounding_payload={
                "kind": "learning_job",
                "timestamp": time.time() - 120,
                "skill_id": "emotion_detection_v1",
                "phase": "collect",
                "status": "active",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("emotion detection v1", reply)
        self.assertIn("active", reply)
        self.assertNotIn("Skill:", reply)

    def test_missing_scholarly_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={"kind": "missing_scholarly"},
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("peer-reviewed", reply)
        self.assertEqual(frame.missing_reason, "missing_scholarly")

    def test_missing_research_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={"kind": "missing_research"},
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("research record", reply)

    def test_missing_learning_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_learning",
            grounding_payload={"kind": "missing_learning"},
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("learning record", reply)

    def test_output_caps_enforced(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "autonomy_research",
                "timestamp": time.time(),
                "question": "x" * 200,
                "tool": "academic",
                "summary": "y" * 400,
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)

    def test_no_hedging_filler(self) -> None:
        frame = build_meaning_frame(
            response_class="recent_research",
            grounding_payload={
                "kind": "scholarly_source",
                "timestamp": time.time(),
                "title": "Test paper",
                "venue": "Test venue",
                "year": 2024,
            },
        )
        reply = articulate_meaning_frame(frame)
        for phrase in ("I think", "I feel", "I believe", "perhaps"):
            self.assertNotIn(phrase, reply.lower())


class IdentityAnswerArticulatorTests(unittest.TestCase):
    """Phase B.2: _articulate_identity_answer for 7 kind subtypes."""

    def test_match_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_match",
                "check_name": "David",
                "matched_modalities": ["voice (69%)", "face (91%)"],
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("David", reply)
        self.assertIn("Confirmed by", reply)
        self.assertNotIn("Match:", reply)

    def test_mismatch_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_mismatch",
                "check_name": "Alice",
                "actual_name": "David",
                "actual_confidence": 0.87,
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("Alice", reply)
        self.assertIn("David", reply)
        self.assertNotIn("Asked about:", reply)

    def test_enrolled_but_not_match(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_enrolled_but_not_match",
                "check_name": "Bob",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("Bob", reply)
        self.assertIn("does not match", reply)

    def test_unknown_profile(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_unknown_profile",
                "check_name": "Charlie",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("Charlie", reply)
        self.assertIn("profile", reply)

    def test_current_voice_with_confidence(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "current_voice",
                "name": "David",
                "confidence": 0.85,
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("David", reply)
        self.assertIn("speaker", reply.lower())
        self.assertNotIn("Speaker:", reply)

    def test_current_face_with_confidence(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "current_face",
                "name": "David",
                "confidence": 0.92,
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("David", reply)
        self.assertIn("face", reply.lower())
        self.assertNotIn("Face:", reply)

    def test_unknown_identity_with_enrolled_profiles(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "unknown_identity",
                "enrolled_voices": ["David", "Alice"],
                "enrolled_faces": ["David"],
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("don't recognize", reply)
        self.assertIn("David", reply)

    def test_no_hedging_filler(self) -> None:
        frame = build_meaning_frame(
            response_class="identity_answer",
            grounding_payload={
                "kind": "identity_check_match",
                "check_name": "David",
                "matched_modalities": ["voice"],
            },
        )
        reply = articulate_meaning_frame(frame)
        for phrase in ("I think", "I feel", "I believe", "perhaps"):
            self.assertNotIn(phrase, reply.lower())


class MemoryRecallArticulatorTests(unittest.TestCase):
    """Phase B.3: _articulate_memory_recall."""

    def test_search_mode_counts_entries(self) -> None:
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={
                "mode": "search",
                "memory_context": (
                    "David likes low latency.\n"
                    "He asked about calibration yesterday.\n"
                    "David prefers direct answers."
                ),
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("3 relevant memory entries", reply)
        self.assertIn("David likes low latency.", reply)
        self.assertNotIn("Memory recall:", reply)

    def test_summary_mode_natural_intro(self) -> None:
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={
                "mode": "summary",
                "memory_context": "The user enjoys hiking and programming.",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("from memory", reply)
        self.assertIn("hiking", reply)

    def test_empty_memory_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={"mode": "search", "memory_context": ""},
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("couldn't verify", reply)

    def test_output_caps_enforced(self) -> None:
        long_ctx = "\n".join(f"Memory fact number {i} with some detail." for i in range(50))
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={"mode": "search", "memory_context": long_ctx},
        )
        reply = articulate_meaning_frame(frame)
        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)

    def test_no_hedging_filler(self) -> None:
        frame = build_meaning_frame(
            response_class="memory_recall",
            grounding_payload={
                "mode": "search",
                "memory_context": "User prefers concise answers.\nUser works in AI.",
            },
        )
        reply = articulate_meaning_frame(frame)
        for phrase in ("I think", "I feel", "I believe", "perhaps"):
            self.assertNotIn(phrase, reply.lower())


class PreLLMBoundedGateTests(unittest.TestCase):
    """Phase B.3: narrow pre-LLM bounded introspection gate logic.

    Tests the gate conditions in isolation (does not test full conversation handler).
    """

    def _make_frame_and_meta(
        self,
        total_facts: int = 20,
        confidence: float = 0.8,
        warnings: list[str] | None = None,
        topics: list[str] | None = None,
    ) -> tuple[MeaningFrame, dict]:
        sections_text = ""
        for i in range(total_facts // 2):
            sections_text += f"=== Section {i} ===\nFact_{i}: value_{i}\n\n"

        frame = build_meaning_frame(
            response_class="self_introspection",
            grounding_payload=sections_text,
        )
        if warnings:
            frame.parse_warnings = warnings
        frame.frame_confidence = confidence

        meta = {
            "total_facts": total_facts,
            "matched_topics": topics or ["consciousness", "health"],
        }
        return frame, meta

    def _gate_passes(self, frame: MeaningFrame, meta: dict) -> bool:
        return (
            meta.get("total_facts", 0) >= 15
            and not frame.missing_reason
            and frame.is_structurally_healthy
            and frame.frame_confidence >= 0.6
            and not frame.parse_warnings
            and any(
                t in meta.get("matched_topics", [])
                for t in ("consciousness", "identity", "memory", "health", "learning")
            )
        )

    def test_gate_passes_for_rich_self_state_data(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.8, topics=["consciousness"]
        )
        self.assertTrue(self._gate_passes(frame, meta))

    def test_gate_rejects_low_fact_count(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=5, confidence=0.8, topics=["consciousness"]
        )
        meta["total_facts"] = 5
        self.assertFalse(self._gate_passes(frame, meta))

    def test_gate_rejects_low_confidence(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.3, topics=["consciousness"]
        )
        self.assertFalse(self._gate_passes(frame, meta))

    def test_gate_rejects_parse_warnings(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.8, warnings=["some_warning"], topics=["consciousness"]
        )
        self.assertFalse(self._gate_passes(frame, meta))

    def test_gate_rejects_non_self_state_topics(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.8, topics=["architecture", "epistemic"]
        )
        self.assertFalse(self._gate_passes(frame, meta))

    def test_gate_passes_for_identity_topic(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.8, topics=["identity"]
        )
        self.assertTrue(self._gate_passes(frame, meta))

    def test_gate_passes_for_memory_topic(self) -> None:
        frame, meta = self._make_frame_and_meta(
            total_facts=20, confidence=0.8, topics=["memory"]
        )
        self.assertTrue(self._gate_passes(frame, meta))


class CapabilityStatusArticulatorTests(unittest.TestCase):
    """Phase B.4: _articulate_capability_status."""

    def test_unverified_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "perform_unverified",
                "skill_name": "singing",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("singing", reply)
        self.assertIn("verified", reply)
        self.assertNotIn("Status:", reply)

    def test_verified_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "perform_verified",
                "skill_name": "face tracking",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("face tracking", reply)
        self.assertIn("verified", reply)

    def test_job_started_includes_phase(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "job_started",
                "message": "I started a learning job for emotion detection.",
                "skill_name": "emotion detection",
                "phase": "assess",
                "capability_type": "perceptual",
                "job_id": "job-456",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("learning job", reply)
        self.assertIn("assess", reply)
        self.assertNotIn("Job:", reply)

    def test_system_uninitialized(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "system_uninitialized",
                "message": "System is still starting up.",
                "skill_name": "",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("not fully initialized", reply)

    def test_guided_collect_started(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "guided_collect_started",
                "message": "Starting guided collection for speaker ID.",
                "skill_name": "speaker identification",
                "status": "active",
                "phase": "collect",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("guided collection", reply.lower())

    def test_no_hedging_filler(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "perform_unverified",
                "skill_name": "singing",
            },
        )
        reply = articulate_meaning_frame(frame)
        for phrase in ("I think", "I feel", "I believe", "perhaps"):
            self.assertNotIn(phrase, reply.lower())

    def test_output_caps_enforced(self) -> None:
        frame = build_meaning_frame(
            response_class="capability_status",
            grounding_payload={
                "kind": "job_started",
                "message": "x" * 200,
                "skill_name": "y" * 100,
                "status": "active",
                "capability_type": "perceptual",
                "job_id": "job-long",
                "phase": "assess",
                "risk_level": "high",
                "protocol_id": "SK-999",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)


class SystemExplanationArticulatorTests(unittest.TestCase):
    """Phase B.4: _articulate_system_explanation."""

    def test_codebase_answer_spoken_naturally(self) -> None:
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "Memory architecture",
                "query": "how does memory work",
                "body": "Memory routes through storage.\nRetrieval uses semantic search.\nResults ranked by cortex model.",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("memory architecture", reply.lower())
        self.assertIn("Memory routes through storage.", reply)
        self.assertNotIn("System explanation:", reply)

    def test_generic_title_gets_default_intro(self) -> None:
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "System explanation",
                "query": "tell me about X",
                "body": "X does Y.\nZ is related.",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("codebase", reply.lower())

    def test_missing_body_fails_closed(self) -> None:
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "Something",
                "query": "what",
                "body": "",
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertIn("don't have", reply)

    def test_output_caps_enforced(self) -> None:
        long_body = "\n".join(f"Line {i} with some detail about the system." for i in range(50))
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "Big explanation",
                "query": "everything",
                "body": long_body,
            },
        )
        reply = articulate_meaning_frame(frame)
        self.assertLessEqual(len(reply), MAX_ARTICULATE_CHARS + 50)

    def test_no_hedging_filler(self) -> None:
        frame = build_meaning_frame(
            response_class="system_explanation",
            grounding_payload={
                "title": "Test",
                "query": "test",
                "body": "The system handles requests via routing.",
            },
        )
        reply = articulate_meaning_frame(frame)
        for phrase in ("I think", "I feel", "I believe", "perhaps"):
            self.assertNotIn(phrase, reply.lower())


if __name__ == "__main__":
    unittest.main()
