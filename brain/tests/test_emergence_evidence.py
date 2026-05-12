"""Emergence evidence truth-surface regressions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard.snapshot import _build_emergence_evidence_snapshot
from reasoning.bounded_response import articulate_meaning_frame, build_meaning_frame


def test_emergence_evidence_ladder_shape_and_level7_boundary():
    payload = _build_emergence_evidence_snapshot(
        {
            "consciousness": {
                "stage": "integrative",
                "awareness_level": 0.72,
                "confidence_avg": 0.81,
                "reasoning_quality": 0.74,
                "emergent_behavior_count": 144,
            },
            "thoughts": {
                "total_generated": 12,
                "recent": [
                    {"type": "pattern_recognition", "text": "I noticed a recurring pattern."}
                ],
            },
            "observer": {"observation_count": 42},
            "evolution": {
                "state": {
                    "stage_history": ["basic_awareness", "self_reflective"],
                    "emergent_behaviors": [{"type": "thought_cluster"}],
                },
                "restore_trust": {"trust": "verified"},
            },
            "mutations": {"count": 2, "history": ["tuned kernel budget"]},
            "world_model": {"validated_count": 3},
        }
    )

    assert payload["summary"]["stance"] == "operational_emergence_evidence_not_sentience_proof"
    assert len(payload["levels"]) == 8
    level7 = payload["levels"][7]
    assert level7["level"] == 7
    assert level7["status"] == "not_claimed"
    assert level7["evidence_count"] == 0


def test_emergence_evidence_bounded_answer_rejects_sentience_claim():
    payload = _build_emergence_evidence_snapshot(
        {
            "consciousness": {"stage": "integrative", "emergent_behavior_count": 3},
            "thoughts": {"total_generated": 1, "recent": []},
        }
    )
    frame = build_meaning_frame(
        response_class="emergence_evidence",
        grounding_payload=payload,
    )
    answer = articulate_meaning_frame(frame)

    assert "not proof of sentience" in answer
    assert "Real substrate evidence, not roleplay" in answer
    assert "Level 7 is not claimed" in answer


def test_l3_counts_grounded_curiosity_questions_not_only_autonomy():
    payload = _build_emergence_evidence_snapshot(
        {
            "consciousness": {"stage": "integrative"},
            "thoughts": {"total_generated": 0, "recent": []},
            "curiosity": {
                "total_generated": 2,
                "total_asked": 1,
                "recent_questions": [
                    {
                        "source": "identity",
                        "question": "I heard a voice I don't recognize. Who was that?",
                        "evidence": "unknown_voice_event at 123, companion=David",
                        "asked": True,
                    },
                    {
                        "source": "scene",
                        "question": "I see something that looks like a cup on the desk. Is that new?",
                        "evidence": "entity=cup, region=desk, stable_cycles=8",
                        "asked": False,
                    },
                ],
            },
            "autonomy": {"completed": []},
        }
    )

    level3 = payload["levels"][3]
    assert level3["level"] == 3
    assert level3["status"] == "supported"
    assert level3["evidence_count"] == 2
    assert "personality/curiosity_questions.py" in level3["source_paths"]
    assert any("voice I don't recognize" in item for item in level3["representative_examples"])
