"""Operator-proxy TAP contracts. These pin the TAP, not a live sit.

A sit is POST /api/operator/tap against the running brain (or Pi voice).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_BRAIN = Path(__file__).resolve().parent.parent


def test_operator_proxy_is_a_provenance_class():
    from consciousness.events import PROVENANCE_BOOST, PROVENANCE_ORDINAL
    assert "operator_proxy" in PROVENANCE_BOOST
    assert "operator_proxy" in PROVENANCE_ORDINAL
    assert PROVENANCE_BOOST["operator_proxy"] == PROVENANCE_BOOST["user_claim"]


def test_voice_intent_teacher_skipped_on_operator_proxy_turn():
    from consciousness.events import operator_proxy_turn
    from reasoning.tool_router import RoutingResult, ToolType
    from reasoning.tool_router import _record_voice_intent_teacher_signal

    tok = operator_proxy_turn.set(True)
    try:
        with patch("hemisphere.distillation.distillation_collector") as coll:
            _record_voice_intent_teacher_signal(
                "What do you remember about me?",
                RoutingResult(tool=ToolType.MEMORY, confidence=0.9, extracted_args={}),
            )
            coll.record.assert_not_called()
    finally:
        operator_proxy_turn.reset(tok)


def test_retired_chat_is_410_and_tap_exists():
    raw = (_BRAIN / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "status_code=410" in raw
    assert "/api/operator/tap" in raw
    assert "inject_operator_turn_async" in raw
    assert "_response_gen.respond(message)" not in raw


def test_tap_follow_up_is_not_default_new_sit():
    raw = (_BRAIN / "perception_orchestrator.py").read_text(encoding="utf-8")
    assert "follow_up=use_follow_up" in raw
    assert "client_new_sit" in raw
    assert "ear_follow_up_window" in raw
    assert "expects_follow_up" in raw
    # TAP skip-wake is the inject, not follow_up=True on every sit.
    assert "TAP skip-wake is this inject" in raw


def test_tap_seam_is_handle_transcription():
    raw = (_BRAIN / "perception_orchestrator.py").read_text(encoding="utf-8")
    assert "OPERATOR-PROXY TAP" in raw
    assert "inject_operator_turn_async" in raw
    assert "handle_transcription(" in raw
    assert 'method": "operator_proxy"' in raw or "operator_proxy" in raw


def test_enrolled_this_turn_speaker_beats_unknown_face_crop():
    """JARVIS already knows David (enrolled). Unknown crop is not him.

    Lived TAP: fusion face_11 known=False stole L3 → guest → LLM cousins.
    Pi voice never did that — this-turn speaker was the enrolled identity.
    Same law for TAP and ear. No TAP login flag. No hardcoded name.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from identity.resolver import IdentityResolver

    r = IdentityResolver()
    soul = MagicMock()
    soul.relationships = {"david": MagicMock()}
    r.set_soul(soul)
    fusion = MagicMock()
    fusion.current = SimpleNamespace(
        name="face_11",
        confidence=0.54,
        is_known=False,
        method="live_crop",
    )
    r.set_fusion(fusion)
    ctx = r.resolve_for_memory(provenance="conversation", speaker="David")
    assert ctx.identity_id == "david"
    assert ctx.identity_type == "primary_user"
    assert ctx.resolved_by == "speaker_tag"


def test_tap_status_endpoint_exists():
    raw = (_BRAIN / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "/api/operator/tap/status" in raw
    assert "follow_up" in raw
