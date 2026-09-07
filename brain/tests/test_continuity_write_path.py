"""Write-path integrity for P1 continuity.

Lived 2026-08-24: respond_stream persisted the LLM wipe-draft via
_finalize_response *before* INTROSPECTION fail-closed to the measured dump.
The spoken reply was honest; the store learned the lie. These tests pin:

1. _finalize_response will not remember an OSV-contradicted wipe claim
2. the introspection LLM arm does not persist the discarded draft
3. ordinary conversation still persists
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

# ResponseGenerator imports ollama; tests here do not talk to a model.
sys.modules.setdefault(
    "ollama",
    SimpleNamespace(AsyncClient=object, ChatResponse=object, ResponseError=Exception),
)

import cognition.self_view as sv
from memory.core import CreateMemoryData
from reasoning.response import ResponseGenerator

_LIVED_LIE = (
    "My last recorded memory was on August 10th. I don't have a timeline of "
    "events beyond that, but I can tell you that my current state is fresh "
    "and ready to process new information. I've been offline for a while, so "
    "my memory is essentially reset — I'm starting fresh today."
)

_SRC = Path(__file__).resolve().parent.parent / "conversation_handler.py"


def _osv_with_memories():
    snap = {
        "consciousness": {"stage": "integrative", "awareness_level": 0.98,
                          "transcendence_level": 10.0},
        "evolution": {"stage": "integrative", "transcendence_level": 10.0},
        "policy": {"mode": "shadow", "nn_win_rate": 0.009, "eligible_for_control": False},
        "self_improve": {"active": True, "stage": 2, "effective_dry_run": True},
        "world_model": {"promotion": {"level_name": "active", "total_validated": 1},
                        "causal": {"predictive_total": 1, "predictive_accuracy": 0.8,
                                   "persistence_accuracy": 0.9},
                        "simulator_promotion": {"level_name": "shadow", "total_validated": 1},
                        "simulator": {"avg_confidence": 0.55}},
        "hemisphere": {"enabled": True, "matrix_specialists": [1]},
        "memory": {"total": 734, "core_count": 4,
                   "oldest_timestamp": 1782235435.0, "newest_timestamp": 1787578126.0},
    }
    return sv.build_self_view(engine=None, eval_snapshot={}, skills_summary={},
                              snapshot=snap, now=1.0)


def _generator(remembered: list, history: list) -> ResponseGenerator:
    engine = SimpleNamespace(
        remember=lambda data: remembered.append(data),
        set_phase=lambda _p: None,
    )
    gen = object.__new__(ResponseGenerator)
    gen._engine = engine
    gen._last_injected_memories = []
    return gen


def test_finalize_does_not_remember_lived_wipe_lie(monkeypatch) -> None:
    remembered: list = []
    history: list = []
    gen = _generator(remembered, history)
    monkeypatch.setattr("cognition.self_view.load_self_view", _osv_with_memories)
    monkeypatch.setattr(
        "reasoning.response.context_builder",
        SimpleNamespace(
            add_assistant_message=lambda *a, **k: history.append(a[0] if a else ""),
            save=lambda: None,
        ),
    )
    gen._finalize_response(
        "when was your last recorded memory",
        _LIVED_LIE,
        start=time.time(),
        persist_response=True,
    )
    assert remembered == []
    assert history == []


def test_finalize_still_remembers_ordinary_conversation(monkeypatch) -> None:
    remembered: list = []
    history: list = []
    gen = _generator(remembered, history)
    monkeypatch.setattr("cognition.self_view.load_self_view", _osv_with_memories)
    monkeypatch.setattr(
        "reasoning.response.context_builder",
        SimpleNamespace(
            add_assistant_message=lambda *a, **k: history.append(a[0] if a else ""),
            save=lambda: None,
        ),
    )
    spoken = "Schuyler is the dog, a border collie."
    gen._finalize_response(
        "What do you remember about Schuyler?",
        spoken,
        start=time.time(),
        persist_response=True,
    )
    assert len(remembered) == 1
    data = remembered[0]
    assert isinstance(data, CreateMemoryData)
    assert data.type == "conversation"
    assert data.payload["response"] == spoken
    assert history == [spoken]


def test_introspection_llm_arm_does_not_persist_discarded_draft() -> None:
    """The 09:28 dual-write: persist happened inside respond_stream before
    fail-closed replaced the spoken text. Draft persist must be off."""
    src = _SRC.read_text(encoding="utf-8")
    idx = src.find('tool_hint="introspection"')
    assert idx != -1
    window = src[idx - 500: idx + 80]
    assert "persist_response=False" in window, window[-200:]


def test_p1_path_persists_the_spoken_self_view_reply() -> None:
    """P1 must write the measured reply, not silence, so the store can
    learn the truth next to the scar (which recall will skip)."""
    src = _SRC.read_text(encoding="utf-8")
    # The self-view kind arm is the P1 override.
    marker = "Routing override: self-view introspection"
    # persist helper must be invoked after articulate_self_view
    start = src.find("elif routing.extracted_args and routing.extracted_args.get(\"self_view_kind\"):")
    assert start != -1
    end = src.find("elif routing.tool == ToolType.MEMORY:", start)
    body = src[start:end]
    assert "articulate_self_view" in body
    assert "_persist_spoken_turn" in body
