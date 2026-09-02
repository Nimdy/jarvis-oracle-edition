"""Operator-pull grounding queue — the Step-3 activation.

The shadow grounding drive now populates the async PULL queue (SPARK §6, the
"biggest efficiency win"): operator answers at /v2/grounding feed the promotion
gates (external-only) so the ring earns its way to advisory honestly. NO TTS-push,
NO belief mutation (P5) — verified here at the queue mechanism level.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# isolate the queue file from real ~/.jarvis state
os.environ.setdefault("HOME", tempfile.mkdtemp())

try:
    from autonomy.grounding_queue import GroundingQueue, route_channel
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("grounding_queue import unavailable", allow_module_level=True)


def _q():
    g = GroundingQueue()
    g._pending.clear()
    return g


def test_channel_routing():
    assert route_channel("factual") == "web"
    assert route_channel("identity") == "operator"
    assert route_channel("scene") == "pi_senses"
    # web facet degrades to local_only when web is exhausted (honest starvation)
    assert route_channel("factual", web_exhausted=True) == "local_only"


def test_enqueue_populates_and_dedups_by_belief():
    q = _q()
    q.enqueue(belief_id="b1", question_text="q1", facet="factual",
              channel=route_channel("factual"), grounding_tension=0.7,
              asked_synchronously=False)
    q.enqueue(belief_id="b2", question_text="q2", facet="identity",
              channel=route_channel("identity"), grounding_tension=0.9,
              asked_synchronously=False)
    assert q.pending_count() == 2
    # same belief refreshes, never duplicates
    q.enqueue(belief_id="b1", question_text="q1-refreshed", facet="factual",
              channel="web", grounding_tension=0.8)
    assert q.pending_count() == 2


def test_pull_only_never_marked_synchronous():
    q = _q()
    q.enqueue(belief_id="b1", question_text="q", facet="factual",
              channel="web", asked_synchronously=False)
    rec = next(iter(q._pending.values()))
    assert getattr(rec, "asked_synchronously", False) is False


def test_operator_answer_records_external_validation():
    q = _q()
    q.enqueue(belief_id="b1", question_text="Is X true?", facet="factual",
              channel="web", grounding_tension=0.7)
    qid = next(iter(q._pending.values())).question_id
    res = q.answer(qid, "Yes, that's correct.")
    assert res["ok"] is True
    assert res["record"]["external_validation"] == "confirmed"
    assert res["record"]["grounded"] is True
    assert q._total_answered == 1 and q._total_confirmed == 1


def test_being_corrected_still_counts_as_grounded():
    # SPARK §7: a "no, you're wrong" is refuted but STILL grounded=True
    q = _q()
    q.enqueue(belief_id="b1", question_text="Is X true?", facet="factual", channel="web")
    qid = next(iter(q._pending.values())).question_id
    res = q.answer(qid, "No, that's wrong.")
    assert res["record"]["external_validation"] == "refuted"
    assert res["record"]["grounded"] is True  # moved from inferred -> externally-anchored
    assert q.is_settled("b1") is True
    assert q.is_settled("missing") is False
    # Lived: set_wake_armed would-have every 120s after the queue item existed.
    again = q.enqueue(belief_id="b1", question_text="Is X true?", facet="factual", channel="web")
    assert again is not None and again.answered is True
    assert q.pending_count() == 0


def test_belief_less_would_have_still_lands_on_pull_queue():
    """Lived 2026-08-24: shadow skipped enqueue when belief_id was empty."""
    q = _q()
    rec = q.enqueue(
        belief_id="",
        question_text="Which currently-held belief is inferred and unsupported?",
        facet="factual",
        channel="web",
        asked_synchronously=False,
    )
    assert rec is not None
    assert rec.belief_id.startswith("q:")
    assert rec.asked_synchronously is False
    assert q.pending_count() == 1
    rec2 = q.enqueue(
        belief_id="",
        question_text="Which currently-held belief is inferred and unsupported?",
        facet="factual",
        channel="web",
    )
    assert rec2.belief_id == rec.belief_id
    assert q.pending_count() == 1


def test_answered_belief_does_not_resurface():
    q = _q()
    q.enqueue(belief_id="b1", question_text="Is X true?", facet="factual", channel="web")
    qid = next(iter(q._pending.values())).question_id
    q.answer(qid, "yes")
    assert q.pending_count() == 0
    again = q.enqueue(belief_id="b1", question_text="Is X true again?", facet="factual", channel="web")
    assert again.answered is True
    assert q.pending_count() == 0


def test_shadow_handler_queues_without_belief_tag_and_never_tts(monkeypatch):
    from types import SimpleNamespace
    from autonomy.orchestrator import AutonomyOrchestrator
    from autonomy.drives import DriveAction
    from autonomy.grounding_queue import GroundingQueue

    q = _q()
    monkeypatch.setattr(GroundingQueue, "get_instance", classmethod(lambda cls: q))

    orch = object.__new__(AutonomyOrchestrator)
    orch._last_grounding_report = SimpleNamespace(top_tensions=[])
    gate = SimpleNamespace(
        is_active=lambda: False,
        is_advisory=lambda: False,
        note_shadow_selection=lambda **k: None,
    )

    class FakePromo:
        @staticmethod
        def get_instance():
            return gate

    monkeypatch.setattr("autonomy.drives.GroundingDrivePromotion", FakePromo)

    action = DriveAction(
        drive_type="grounding",
        action_type="research",
        urgency=0.55,
        question=(
            "Which currently-held belief is inferred and structurally "
            "unsupported, and what external evidence would ground it?"
        ),
        tool_hint="web",
        scope="external_ok",
        tags=("drive:grounding", "action:research", "grounding:facet:factual"),
    )
    handled = AutonomyOrchestrator._handle_grounding_action_shadow(orch, action)
    assert handled is True
    assert q.pending_count() == 1
    rec = next(iter(q._pending.values()))
    assert rec.asked_synchronously is False
    assert "unsupported" in rec.question_text.lower()
