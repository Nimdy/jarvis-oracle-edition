"""P1: deterministic self-introspection articulation tests.

Proves the self-view articulator answers from the OSV (not a code grep), preserves
provenance, and CANNOT emit unqualified consciousness/identity claims.
"""
from __future__ import annotations

import cognition.self_view as sv
from cognition.self_view.articulate import (
    KINDS,
    articulate_self_view,
    asserts_memory_wipe,
    classify_register,
    classify_self_question,
    contains_unqualified_claim,
    contradicts_measured_continuity,
    register_from_preference_payload,
    resolve_stored_register,
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


def _model(eval_snapshot=None, skills=None, snapshot=None):
    return sv.build_self_view(engine=None, eval_snapshot=eval_snapshot or {},
                              skills_summary=skills or {}, snapshot=snapshot or _snapshot(), now=1.0)


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
            "Walk me through how you get an answer.": "answer_path",
            "Walk me through how you reach an answer.": "answer_path",
            "walk me through how you reach an answer": "answer_path",
            "how do you reach an answer?": "answer_path",
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

    def test_continuity_questions_route_to_self_view(self):
        """Lived miss: last-memory-after-power-off must not fall through to the LLM."""
        live = (
            "what was the last thing you remember because you've been powered off "
            "for a long time when was your last um recorded memory because today is august 24th"
        )
        assert classify_self_question(live) == "continuity"
        assert classify_self_question("when was your last recorded memory?") == "continuity"
        assert classify_self_question("have you been offline — is your memory reset?") == "continuity"
        # MEMORY recall of *content* must still not be stolen
        assert classify_self_question("What do you remember about Skylar?") is None
        assert classify_self_question("what do you remember about the meeting") is None
        assert classify_self_question("What do you know about Skyler from before?") is None


# ---------------------------------------------------------------------------
# Articulation content
# ---------------------------------------------------------------------------

class TestArticulation:
    def test_identity_from_osv_not_grep(self):
        out = articulate_self_view(_model(), "identity", register="tech")
        assert "JARVIS Oracle Edition" in out
        assert "symbol" not in out.lower()  # not a code grep
        assert "subsystems" in out.lower()

    def test_identity_exec_is_brief(self):
        out = articulate_self_view(_model(), "identity")
        low = out.lower()
        assert "JARVIS Oracle Edition" in out
        assert "98" not in out
        assert "l0-l12" not in low
        assert "ask if you want" in low

    def test_capabilities_separates_buckets(self):
        out = articulate_self_view(_model(), "capabilities", register="tech").lower()
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
        out = articulate_self_view(_model(), "gated_capabilities", register="tech").lower()
        assert "shadow" in out
        assert "earned" in out  # earned-not-declared framing

    def test_answer_path_is_measured_not_theater(self):
        """Lived 14:24: walk-through classified capabilities and recited the
        architecture inventory. This kind must describe the turn path from the
        OSV/architecture map — regex router live, voice-intent shadow — and
        must not invent understanding/feeling/percent-confidence theater.
        """
        out = articulate_self_view(_model(), "answer_path")
        low = out.lower()
        assert "router" in low or "routing" in low
        assert "shadow" in low
        assert "llm" in low or "language model" in low
        assert "understand" not in low
        assert "86 percent" not in low and "86%" not in low
        assert "i parse" not in low
        assert "pattern recognition" not in low
        assert contains_unqualified_claim(out) is False
        # Lived 14:45/14:47: kind was correct, mouth was a designed-status dump.
        # Keep this speakable. Do not put log headers or inventory tokens on TTS.
        assert "speech in:" not in low
        assert "designed-status" not in low
        assert out.count(".") <= 6

    def test_what_can_you_do_is_exec_brief(self):
        assert classify_self_question("What can you do?") == "capabilities"
        assert classify_register("What can you do?") == "exec"
        out = articulate_self_view(_model(), "capabilities").lower()
        assert "shadow" in out or "gated" in out
        assert "world_model" not in out
        assert "policy" not in out
        assert "98" not in out
        assert "designed-status" not in out
        assert "ask if you want the numbers" in out

    def test_what_can_you_do_in_detail_is_tech_inventory(self):
        q = "What can you do in detail?"
        assert classify_self_question(q) == "capabilities"
        assert classify_register(q) == "tech"
        out = articulate_self_view(_model(), "capabilities", register="tech").lower()
        assert "active" in out and "shadow" in out
        assert "world_model" in out

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


# ---------------------------------------------------------------------------
# Continuity (process restart vs wipe) — lived miss 2026-08-24
# ---------------------------------------------------------------------------

class TestContinuityArticulation:
    def _mem_model(self, total=734, oldest=1782235435.0, newest=1787578126.0):
        snap = _snapshot()
        snap["memory"] = {
            "total": total,
            "core_count": 4,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
        }
        return _model(snapshot=snap)

    def test_does_not_claim_wipe_when_memories_exist(self):
        out = articulate_self_view(self._mem_model(), "continuity")
        low = out.lower()
        assert "734" in out
        assert "reset" not in low
        assert "starting fresh" not in low
        assert "wipe" in low  # "not a wipe" / process restart
        assert contains_unqualified_claim(out) is False

    def test_reports_measured_span_not_invented_date(self):
        import datetime
        oldest, newest = 1782235435.0, 1787578126.0
        out = articulate_self_view(self._mem_model(oldest=oldest, newest=newest), "continuity")
        def _day(ts: float) -> str:
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        assert _day(oldest) in out
        assert _day(newest) in out

    def test_unknown_memory_does_not_invent_a_wipe(self):
        out = articulate_self_view(_model(), "continuity").lower()
        assert "reset" not in out
        assert "starting fresh" not in out
        assert "can't measure" in out or "cannot measure" in out or "not readable" in out

    # Lived 2026-08-24 dual-write: LLM draft stored as conversation memory while
    # fail-closed speech used the measured dump. These pin the detector the
    # write-path and MEMORY recall both consult. Never a wipe of the scar —
    # a contradicted blank-slate claim must not be treated as autobiography.
    _LIVED_LIE = (
        "My last recorded memory was on August 10th. I don't have a timeline of "
        "events beyond that, but I can tell you that my current state is fresh "
        "and ready to process new information. I've been offline for a while, so "
        "my memory is essentially reset — I'm starting fresh today."
    )

    def test_lived_llm_lie_asserts_a_memory_wipe(self):
        assert asserts_memory_wipe(self._LIVED_LIE) is True

    def test_continuity_answer_does_not_assert_a_wipe(self):
        out = articulate_self_view(self._mem_model(), "continuity")
        assert asserts_memory_wipe(out) is False

    def test_lived_lie_contradicts_measured_store(self):
        assert contradicts_measured_continuity(self._LIVED_LIE, self._mem_model()) is True

    def test_cannot_prove_contradiction_when_store_unreadable(self):
        # KNOW-not-guess: no measured count → do not treat as contradicted.
        assert contradicts_measured_continuity(self._LIVED_LIE, _model()) is False

    def test_ordinary_recall_is_not_a_wipe_claim(self):
        assert asserts_memory_wipe("You went to the mall with your kids.") is False
        assert contradicts_measured_continuity(
            "You went to the mall with your kids.", self._mem_model()
        ) is False


class TestRegister:
    """Exec / tech / ops mouth. Full OSV always; register is this-turn speaker."""

    def test_turn_overrides_and_stored_default(self):
        assert classify_register("What can you do?") == "exec"
        assert classify_register("What can you do in detail?") == "tech"
        assert classify_register("give me the numbers") == "tech"
        assert classify_register("Walk me through how you reach an answer.") == "exec"
        assert classify_register(
            "Walk me through how you reach an answer, with the numbers."
        ) == "tech"
        assert classify_register("what's running") == "ops"
        assert classify_register("what can you do, keep it high level", stored="tech") == "exec"
        assert classify_register("what can you do", stored="tech") == "tech"
        assert classify_register("what can you do", stored="nope") == "exec"

    def test_preference_payload_map(self):
        assert register_from_preference_payload("User prefers concise responses") == "exec"
        assert register_from_preference_payload("User prefers detailed responses") == "tech"
        assert register_from_preference_payload("User prefers exec briefing") == "exec"
        assert register_from_preference_payload("User prefers tech briefing") == "tech"
        assert register_from_preference_payload("User prefers ops briefing") == "ops"
        assert register_from_preference_payload("User enjoys pizza") is None

    def test_stored_register_is_this_turn_speaker_not_another_person(self, monkeypatch):
        class Rel:
            def __init__(self, prefs):
                self.preferences = prefs

        class DummySoul:
            identity = type("I", (), {
                "relationships": {
                    "david": Rel({"briefing_register": "tech"}),
                }
            })()

        class DummyMem:
            def get_by_tag(self, tag):
                return []

        import consciousness.soul as soul_mod
        import memory.storage as mem_mod
        monkeypatch.setattr(soul_mod, "soul_service", DummySoul())
        monkeypatch.setattr(mem_mod, "memory_storage", DummyMem())

        assert resolve_stored_register("unknown") is None
        assert resolve_stored_register("") is None
        assert resolve_stored_register("david") == "tech"
        assert resolve_stored_register("sarah") is None

    def test_answer_path_tech_uses_numbers_not_speech_in(self):
        out = articulate_self_view(_model(), "answer_path", register="tech").lower()
        assert "designed-status" in out
        assert "speech in:" not in out
        assert "understand" not in out

    def test_capabilities_ops_is_counts_not_names(self):
        out = articulate_self_view(_model(), "capabilities", register="ops").lower()
        assert "live and measured" in out
        assert "world_model" not in out
        assert "ask if you want the names" in out
