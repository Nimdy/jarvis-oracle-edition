"""Contracts for explainability fallback provenance coverage."""

from __future__ import annotations

from pathlib import Path

from reasoning.explainability import compact_trace


def test_compact_trace_returns_fallback_when_seed_missing() -> None:
    trace = compact_trace(None)
    assert trace["provenance"] == "fallback_unclassified"
    assert trace["fallback"] is True
    assert trace["response_class"] == "unknown"


def test_conversation_handler_persists_fallback_provenance_to_ledger_data() -> None:
    src = Path("conversation_handler.py").read_text(encoding="utf-8")
    assert '"provenance": "fallback_unclassified"' in src
    assert '"provenance": _provenance_meta' in src


def test_snapshot_explainability_synthesizes_fallback_trace() -> None:
    src = Path("dashboard/snapshot.py").read_text(encoding="utf-8")
    assert '"provenance": "fallback_from_ledger"' in src
    assert '"fallback:snapshot_response_complete"' in src
