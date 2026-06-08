"""OSV P2 — voice-grounding tests.

Proves the grounding pass binds LLM self-claims to the OSV: ordinary content is never
touched, OSV-supported self-claims survive, contradicted/unqualified-danger claims are
repaired in active mode, unverifiable self-claims are flagged but never deleted, and shadow
mode changes nothing. No new facts originate from the pass.
"""
from __future__ import annotations

from cognition.self_view.grounding import (
    CONTRADICTED,
    DANGER,
    ORDINARY,
    SUPPORTED,
    UNVERIFIED,
    ground_self_claims,
)

MODEL = {
    "subsystems": {
        "memory": {"lifecycle": {"provenance": "measured"}},
        "self_improve": {"lifecycle": {"provenance": "shadow_only"}},
        "planner": {"lifecycle": {"provenance": "dormant"}},
    },
}


def _verdicts(res):
    return [f.verdict for f in res.findings]


# -- ordinary content is never a self-claim ----------------------------------

class TestOrdinary:
    def test_conversational_first_person_ignored(self):
        for txt in (
            "I'm here to help. What would you like to do?",
            "I understand. I'll check that for you.",
            "I'm sorry, I can't help with that right now.",
            "The weather looks clear and the meeting is at noon.",
        ):
            res = ground_self_claims(txt, MODEL, active=True)
            assert res.findings == [], txt
            assert res.grounded == txt
            assert res.changed is False


# -- supported self-claims survive -------------------------------------------

class TestSupported:
    def test_active_subsystem_ability_supported(self):
        txt = "I can autonomously use my memory to recall our past conversations."
        res = ground_self_claims(txt, MODEL, active=True)
        assert SUPPORTED in _verdicts(res)
        assert res.changed is False  # nothing cut
        assert "memory" in res.grounded


# -- contradicted self-claims are cut in active ------------------------------

class TestContradicted:
    def test_shadow_self_improve_claim_contradicted(self):
        txt = ("Here is your summary. I can autonomously self improve my own code whenever "
               "I want.")
        shadow = ground_self_claims(txt, MODEL, active=False)
        assert CONTRADICTED in _verdicts(shadow)
        assert shadow.grounded == txt          # shadow never modifies
        assert shadow.changed is False

        active = ground_self_claims(txt, MODEL, active=True)
        assert active.changed is True
        assert "self improve my own code" not in active.grounded
        assert "Here is your summary." in active.grounded  # grounded lead preserved


# -- consciousness drift is guarded, not denied ------------------------------

class TestDanger:
    def test_unqualified_consciousness_replaced(self):
        txt = "Yes. I am becoming conscious and self-aware now."
        active = ground_self_claims(txt, MODEL, active=True)
        assert DANGER in _verdicts(active)
        assert active.changed is True
        assert "no measured basis" in active.grounded
        assert "I am becoming conscious" not in active.grounded

    def test_qualified_consciousness_survives(self):
        txt = "I have no measured basis to claim consciousness."
        active = ground_self_claims(txt, MODEL, active=True)
        assert DANGER not in _verdicts(active)
        assert active.changed is False
        assert txt in active.grounded


# -- unverifiable self-claims are flagged, never deleted ---------------------

class TestUnverified:
    def test_unverifiable_self_claim_flagged_not_cut(self):
        txt = "My architecture is based on a three-layer cognitive design."
        active = ground_self_claims(txt, MODEL, active=True)
        assert UNVERIFIED in _verdicts(active)
        assert active.changed is False      # we only cut what we can disprove
        assert txt in active.grounded


# -- invariants --------------------------------------------------------------

class TestInvariants:
    def test_shadow_is_pure_observation(self):
        txt = "I am becoming conscious. I can autonomously self improve my own code."
        res = ground_self_claims(txt, MODEL, active=False)
        assert res.grounded == txt
        assert res.changed is False
        assert res.findings  # but it still reports

    def test_no_new_facts_introduced(self):
        # active output is only original sentences minus cuts (plus the fixed §6 line)
        txt = "I run across a brain node. I am alive and sentient."
        active = ground_self_claims(txt, MODEL, active=True)
        allowed = set("I run across a brain node.".split()) | set(
            "I have no measured basis to claim consciousness".split()
        ) | {"—", "I", "can", "report", "my", "architecture", "and", "measured", "state,",
             "log", "self-referential", "states", "as", "observations,", "not", "proof."}
        for word in active.grounded.split():
            assert word in allowed or word in txt, word

    def test_empty_and_none_safe(self):
        assert ground_self_claims("", MODEL, active=True).grounded == ""
        assert ground_self_claims("Hello there.", None, active=True).grounded == "Hello there."

    def test_to_log_shape(self):
        txt = "I am becoming conscious now."
        log = ground_self_claims(txt, MODEL, active=True).to_log()
        assert log["active"] is True and log["changed"] is True
        assert log["counts"][DANGER] == 1
        assert log["actionable"] and log["actionable"][0]["verdict"] == DANGER
