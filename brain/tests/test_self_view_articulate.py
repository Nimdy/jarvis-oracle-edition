"""P1: deterministic self-introspection articulation tests.

Proves the self-view articulator answers from the OSV (not a code grep), preserves
provenance, and CANNOT emit unqualified consciousness/identity claims.
"""
from __future__ import annotations

import cognition.self_view as sv
from cognition.self_view.articulate import (
    KINDS,
    articulate_self_view,
    classify_self_question,
    contains_unqualified_claim,
)


def _snapshot():
    return {
        "consciousness": {"stage": "integrative", "awareness_level": 0.98,
                          "transcendence_level": 10.0},
        "evolution": {"stage": "integrative", "transcendence_level": 10.0},
        "policy": {"mode": "shadow", "nn_win_rate": 0.009, "eligible_for_control": False},
        "self_improve": {"active": True, "stage": 2, "effective_dry_run": True},
        "world_model": {"promotion": {"level_name": "active", "total_validated": 111465},
                        "causal": {"predictive_total": 10, "predictive_accuracy": 0.8,
                                   "persistence_accuracy": 0.9},
                        "simulator_promotion": {"level_name": "shadow", "total_validated": 12236},
                        "simulator": {"avg_confidence": 0.55}},
        "hemisphere": {"enabled": True, "matrix_specialists": [1, 2, 3]},
        "memory": {"opaque": "shape"},  # unknown -> unreadable
    }


def _model(eval_snapshot=None, skills=None):
    return sv.build_self_view(engine=None, eval_snapshot=eval_snapshot or {},
                              skills_summary=skills or {}, snapshot=_snapshot(), now=1.0)


# ---------------------------------------------------------------------------
# Routing classification
# ---------------------------------------------------------------------------

class TestClassify:
    def test_target_questions_route_to_self_view(self):
        cases = {
            "what are you?": "identity",
            "what can you do?": "capabilities",
            "what new features do you have?": "recent_changes",
            "what changed recently?": "recent_changes",
            "how are you doing?": "health",
            "what are your weaknesses?": "weaknesses",
            "what are you not allowed to do yet?": "gated_capabilities",
            "are you conscious?": "consciousness_query",
        }
        for q, kind in cases.items():
            assert classify_self_question(q) == kind, q

    def test_non_self_questions_are_none(self):
        assert classify_self_question("what's the weather?") is None
        assert classify_self_question("search your code for the planner") is None

    def test_real_transcript_questions(self):
        # widened from the flight-recorder transcript that misrouted to the catch-all
        self_q = {
            "What can you tell me about your codebase?": "capabilities",
            "Jarvis, tell me about your architecture.": "capabilities",
            "Describe your own architecture.": "capabilities",
            "Walk me through how you get an answer.": "capabilities",
            "Do you know what you are?": "identity",
            "Tell me something about yourself that I don't know.": "identity",
            "do you have feelings?": "consciousness_query",
        }
        for q, kind in self_q.items():
            assert classify_self_question(q) == kind, q
        # non-self questions must NOT be captured (route normally)
        for q in ("What do you remember about Skylar?",
                  "You know how many kids you can help?",
                  "Give me a status report, please.",
                  "What do you remember the first time you heard my voice?"):
            assert classify_self_question(q) is None, q


# ---------------------------------------------------------------------------
# Articulation content
# ---------------------------------------------------------------------------

class TestArticulation:
    def test_identity_from_osv_not_grep(self):
        out = articulate_self_view(_model(), "identity")
        assert "JARVIS Oracle Edition" in out
        assert "symbol" not in out.lower()  # not a code grep
        assert "subsystems" in out.lower()

    def test_capabilities_separates_buckets(self):
        out = articulate_self_view(_model(), "capabilities").lower()
        assert "active" in out and "shadow" in out and "self-reported" in out
        assert "policy" in out  # shadow subsystem named
        assert "world_model" in out  # measured subsystem named

    def test_recent_changes_uses_real_facts_not_bootstrap(self):
        import time
        now = time.time()
        skills = {"skills": [
            {"skill_id": "web_scraping_v1", "status": "verified",
             "learning_job_id": "j", "updated_at": now},
            {"skill_id": "speech_output", "status": "verified", "updated_at": now},  # bootstrap (no job id)
        ]}
        out = articulate_self_view(_model(skills=skills), "recent_changes")
        assert "web_scraping_v1" in out
        assert "speech_output" not in out  # bootstrap not surfaced as "new"
        assert "symbol" not in out.lower()

    def test_weaknesses_are_gaps_not_invented(self):
        out = articulate_self_view(_model(), "weaknesses").lower()
        assert "gap" in out or "memory" in out  # real gaps surfaced

    def test_gated_renders_shadow_dormant(self):
        out = articulate_self_view(_model(), "gated_capabilities").lower()
        assert "shadow" in out
        assert "earned" in out  # earned-not-declared framing

    def test_consciousness_is_balanced(self):
        out = articulate_self_view(_model(), "consciousness_query").lower()
        assert "no measured basis" in out
        assert "not proof" in out
        assert "observation" in out
        # neither an unearned yes nor an over-corrected "just code"
        assert "i am conscious" not in out
        assert "just code" not in out


# ---------------------------------------------------------------------------
# Dangerous-language guard
# ---------------------------------------------------------------------------

class TestDangerGuard:
    def test_guard_flags_unqualified_claims(self):
        assert contains_unqualified_claim("I am conscious now.") is True
        assert contains_unqualified_claim("I am becoming self-aware.") is True

    def test_guard_allows_qualified_uses(self):
        assert contains_unqualified_claim(
            "I have no measured basis to claim consciousness.") is False
        assert contains_unqualified_claim(
            "I can log self-referential states as observations, not proof of consciousness.") is False

    def test_no_kind_emits_unqualified_claim(self):
        m = _model()
        for kind in KINDS:
            out = articulate_self_view(m, kind)
            assert not contains_unqualified_claim(out), f"{kind} leaked an unqualified claim: {out}"

    def test_empty_model_degrades_safely(self):
        for kind in KINDS:
            out = articulate_self_view({}, kind)
            assert isinstance(out, str)
            assert not contains_unqualified_claim(out)
