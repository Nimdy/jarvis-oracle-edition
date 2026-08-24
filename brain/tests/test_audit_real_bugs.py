"""Pins the 2026-08-24 ground-truth REAL BUGS. Behavior, not theater.

Lived miss: pytest against the live brain HOME overwrote plugin_registry.
These tests must stay isolated (tmp paths / source pins / in-memory objects).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognition.self_view.articulate import KINDS
from consciousness.kernel import PerformanceMetrics
from identity.resolver import IdentityResolver
from memory.core import CreateMemoryData
from tools.memory_tool import (
    _leads_with_referenced_subject,
    _matches_aboutness,
    search_memory,
)
from tools.plugin_registry import PluginManifest, PluginRegistry


_HANDLER = Path(__file__).resolve().parent.parent / "conversation_handler.py"
_CS = Path(__file__).resolve().parent.parent / "consciousness" / "consciousness_system.py"


def _handler_src() -> str:
    return _HANDLER.read_text(encoding="utf-8")


def test_p1_kinds_are_ten() -> None:
    assert "continuity" in KINDS
    assert "answer_path" in KINDS
    assert len(KINDS) == 10


def test_session_start_and_enrollment_use_create_memory_data() -> None:
    src = _handler_src()
    start = src.find("first_this_session")
    assert start != -1
    window = src[start:start + 1800]
    assert "CreateMemoryData(" in window
    assert 'memory_type="interaction"' not in window
    enroll = src.find("identity_enrollment")
    assert enroll != -1
    ewin = src[enroll - 800:enroll + 400]
    assert "CreateMemoryData(" in ewin
    assert 'memory_type="milestone"' not in ewin


def test_domain_recall_does_not_steal_memory_about_x() -> None:
    src = _handler_src()
    marker = "capability-domain recall"
    idx = src.find(marker)
    assert idx != -1
    window = src[idx - 1600:idx + 250]
    assert "about_subjects" in window
    assert "ToolType.MEMORY" in window
    assert "_about_x" in window


def test_native_memory_persists_spoken_turn() -> None:
    src = _handler_src()
    start = src.find("elif routing.tool == ToolType.MEMORY:")
    end = src.find("elif routing.tool == ToolType.INTROSPECTION:", start)
    body = src[start:end]
    assert "_persist_spoken_turn" in body
    assert "deterministic_grounded_recall" in body


def test_meta_thoughts_inner_passes_grounding_tension() -> None:
    src = _CS.read_text(encoding="utf-8")
    start = src.find("def _run_meta_thoughts_inner(")
    end = src.find("\n    def _run_contradiction_check(", start)
    body = src[start:end]
    assert '"grounding_tension"' in body
    assert '"grounding_target_id"' in body


def test_aboutness_skylar_matches_skyler_store() -> None:
    mem = SimpleNamespace(
        type="conversation",
        payload={"response": "Skyler is your dog, a border collie."},
        weight=0.55,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=(),
    )
    assert _leads_with_referenced_subject(mem, {"Skylar"})
    assert _matches_aboutness(mem, {"Skylar"}, speaker="David")


def test_about_me_drops_courtesy_lead() -> None:
    mem = SimpleNamespace(
        type="conversation",
        payload={"response": "You're welcome, David."},
        weight=0.40,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=(),
    )
    assert _matches_aboutness(mem, {"David"}, speaker="David") is False


def test_search_memory_about_skylar_returns_skyler(monkeypatch) -> None:
    topical = SimpleNamespace(
        type="conversation",
        payload={"response": "Skyler is your dog, a border collie."},
        weight=0.55,
        provenance="conversation",
        identity_subject="david",
        identity_subject_type="person",
        identity_owner_type="person",
        tags=(),
    )
    fake = SimpleNamespace(
        semantic_search_scored=lambda *a, **k: [(0.62, topical)],
        keyword_search=lambda *a, **k: [],
    )
    monkeypatch.setitem(sys.modules, "memory.search", fake)
    monkeypatch.setattr("tools.memory_tool._extract_referenced_entities", lambda _q: set())
    out = search_memory("What do you remember about Skylar?", speaker="David")
    assert "border collie" in out


def test_who_is_extracts_about_subject() -> None:
    from tools.memory_tool import _extract_about_subjects
    assert "Skyler" in _extract_about_subjects("Who is Skyler?")
    assert "Skylar" in _extract_about_subjects("Who's Skylar?")


def test_resolver_same_speaker_keeps_fusion_known() -> None:
    """Lived 17:40: David + persisted David must stay primary_user, not guest."""
    r = IdentityResolver()
    r._known_names = set()
    fusion = MagicMock()
    fusion.current = SimpleNamespace(
        name="David",
        confidence=0.9,
        is_known=True,
        method="persisted",
    )
    r.set_fusion(fusion)
    ctx = r.resolve_for_memory(speaker="David")
    assert ctx.identity_id == "david"
    assert ctx.identity_type == "primary_user"
    assert ctx.resolved_by.startswith("fusion:")


def test_resolver_this_turn_speaker_beats_persisted_face() -> None:
    r = IdentityResolver()
    r._known_names = {"david"}
    fusion = MagicMock()
    fusion.current = SimpleNamespace(
        name="David",
        confidence=0.9,
        is_known=True,
        method="persisted",
    )
    r.set_fusion(fusion)
    ctx = r.resolve_for_memory(speaker="Guest")
    assert ctx.identity_id == "guest"
    assert ctx.identity_type == "guest"
    assert ctx.resolved_by == "speaker_tag"


def test_resolver_fusion_this_turn_voice_still_used_when_no_speaker() -> None:
    r = IdentityResolver()
    r._known_names = {"david"}
    fusion = MagicMock()
    fusion.current = SimpleNamespace(
        name="David",
        confidence=0.9,
        is_known=True,
        method="voice_only",
    )
    r.set_fusion(fusion)
    ctx = r.resolve_for_memory()
    assert ctx.identity_id == "david"
    assert "fusion" in ctx.resolved_by


def test_health_gate_reads_performance_metrics_dataclass() -> None:
    from self_improve.orchestrator import SelfImprovementOrchestrator, _perf_field

    metrics = PerformanceMetrics(p95_tick_ms=5.0, last_tick_ms=5.0, tick_count=200)
    assert _perf_field(metrics, "p95_tick_ms") == 5.0
    assert _perf_field(metrics, "last_tick_ms") == 5.0
    assert _perf_field({"p95_tick_ms": 7.0}, "p95_tick_ms") == 7.0

    o = SelfImprovementOrchestrator.__new__(SelfImprovementOrchestrator)
    o._engine = MagicMock()
    o._engine._kernel.get_performance = lambda: metrics
    import asyncio
    from unittest.mock import AsyncMock, patch

    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        assert asyncio.run(o._check_post_apply_health("/tmp/x")) is True


def test_approve_prepares_restart_verify_without_none_conv() -> None:
    src = Path(__file__).resolve().parent.parent / "self_improve" / "orchestrator.py"
    text = src.read_text(encoding="utf-8")
    start = text.find("async def approve(")
    end = text.find("\n    async def ", start + 1)
    body = text[start:end if end != -1 else start + 2500]
    assert "conv=None" not in body
    assert "_prepare_restart_verify" in body


def test_plugin_allowlist_accepts_os_path_and_urllib_parse() -> None:
    reg = PluginRegistry.__new__(PluginRegistry)
    manifest = PluginManifest(allowed_imports=[])
    src = "from os.path import join\nfrom urllib.parse import quote\n"
    errors = PluginRegistry._check_imports(reg, src, "plugin.py", manifest)
    assert errors == []


def test_finalize_persists_gated_not_raw_capability_claim(monkeypatch) -> None:
    from reasoning.response import ResponseGenerator
    from memory.core import CreateMemoryData as CMD

    remembered: list = []
    history: list = []
    engine = SimpleNamespace(
        remember=lambda data: remembered.append(data),
        set_phase=lambda _p: None,
    )
    gen = object.__new__(ResponseGenerator)
    gen._engine = engine
    gen._last_injected_memories = []
    monkeypatch.setattr(
        "cognition.self_view.load_self_view",
        lambda: {"memory": {"total": 10, "oldest_timestamp": 1, "newest_timestamp": 2}},
    )
    monkeypatch.setattr(
        "reasoning.response.context_builder",
        SimpleNamespace(
            add_assistant_message=lambda *a, **k: history.append(a[0] if a else ""),
            save=lambda: None,
        ),
    )

    class _Gate:
        def check_text(self, text):
            if "I can sing" in text:
                return "I don't have that capability yet."
            return text

        def evaluate_commitment(self, text, *_a, **_k):
            return text, False

    monkeypatch.setattr("skills.capability_gate.capability_gate", _Gate())
    gen._finalize_response(
        "sing a song",
        "I can sing you a lullaby.",
        start=time.time(),
        persist_response=True,
    )
    assert len(remembered) == 1
    assert isinstance(remembered[0], CMD)
    assert "I can sing" not in remembered[0].payload["response"]
    assert "don't have that capability" in remembered[0].payload["response"]
