"""Contracts for P2 trace explorer and reconstructability surfaces."""

from __future__ import annotations

from pathlib import Path

from dashboard.snapshot import (
    _build_reconstructability_metadata,
    _build_trace_explorer_snapshot,
)


def test_trace_explorer_snapshot_groups_roots_runs_and_tools() -> None:
    entries = [
        {
            "entry_id": "led_a1",
            "root_entry_id": "led_a1",
            "parent_entry_id": "",
            "ts": 100.0,
            "subsystem": "autonomy",
            "event_type": "research_started",
            "conversation_id": "",
            "outcome": "pending",
            "data": {
                "intent_id": "intent_1",
                "goal_id": "goal_1",
                "task_id": "task_1",
                "tool_hint": "web_search",
                "trace_id": "trc_a",
                "request_id": "req_a",
            },
        },
        {
            "entry_id": "led_a2",
            "root_entry_id": "led_a1",
            "parent_entry_id": "led_a1",
            "ts": 101.5,
            "subsystem": "autonomy",
            "event_type": "query_executed",
            "conversation_id": "",
            "outcome": "success",
            "data": {
                "intent_id": "intent_1",
                "goal_id": "goal_1",
                "task_id": "task_1",
                "tool": "academic_search",
                "trace_id": "trc_a",
                "request_id": "req_a2",
            },
        },
        {
            "entry_id": "led_c1",
            "root_entry_id": "led_c1",
            "parent_entry_id": "",
            "ts": 200.0,
            "subsystem": "conversation",
            "event_type": "response_complete",
            "conversation_id": "conv_1",
            "outcome": "success",
            "data": {
                "tool": "status",
                "output_id": "out_1",
                "trace_id": "trc_c",
                "request_id": "req_c",
            },
        },
    ]

    tx = _build_trace_explorer_snapshot(entries, max_roots=10, max_agent_runs=10, max_tool_lineage=10)
    assert tx["entry_count"] == 3
    assert len(tx["root_chains"]) == 2
    assert len(tx["agent_runs"]) == 1
    assert len(tx["tool_lineage"]) >= 2

    run = tx["agent_runs"][0]
    assert run["intent_id"] == "intent_1"
    assert run["event_count"] == 2
    assert "academic_search" in run["tools"]

    roots = {r["root_entry_id"]: r for r in tx["root_chains"]}
    assert roots["led_a1"]["entry_count"] == 2
    assert "autonomy" in roots["led_a1"]["subsystems"]


def test_reconstructability_metadata_has_core_surface_contracts() -> None:
    meta = _build_reconstructability_metadata()
    assert "trace_explorer" in meta
    assert "operations_panel" in meta
    assert meta["trace_explorer"]["reconstructability"] == "reconstructable"
    assert meta["operations_panel"]["reconstructability"] == "non_reconstructable"


def test_dashboard_api_and_ui_wiring_for_trace_explorer_exists() -> None:
    app_src = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert '@app.get("/api/trace/explorer")' in app_src
    assert '@app.get("/api/trace/explorer/chain/{root_id}")' in app_src
    assert '@app.get("/api/reconstructability")' in app_src

    renderer_src = Path("dashboard/static/renderers.js").read_text(encoding="utf-8")
    assert "window.openTraceExplorer" in renderer_src
    assert "_renderTraceExplorerPanel" in renderer_src
    assert "_renderReconstructabilityPanel" in renderer_src

    interactive_src = Path("dashboard/static/interactives.js").read_text(encoding="utf-8")
    assert "window.openTraceExplorer = function()" in interactive_src
    assert "window.openTraceChain = function(rootId)" in interactive_src
