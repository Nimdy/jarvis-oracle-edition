"""AST regressions for spatial handoff/verify guardrails."""

from __future__ import annotations

import ast
from pathlib import Path


_ORCHESTRATOR = Path(__file__).resolve().parents[1] / "perception_orchestrator.py"


def _class_def() -> ast.ClassDef:
    tree = ast.parse(_ORCHESTRATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PerceptionOrchestrator":
            return node
    raise AssertionError("PerceptionOrchestrator class not found")


def _method(name: str) -> ast.FunctionDef:
    cls = _class_def()
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {name} not found")


def _walk_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def test_relocalization_match_includes_active_profile():
    fn = _method("_maybe_relocalize_spatial")
    calls = _walk_calls(fn)
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "match_profile":
            for kw in call.keywords:
                if kw.arg == "include_active":
                    assert isinstance(kw.value, ast.Constant)
                    assert kw.value.value is True
                    return
    raise AssertionError("match_profile call with include_active=True not found")


def test_process_spatial_invokes_post_handoff_verify_hook():
    fn = _method("_process_spatial")
    calls = _walk_calls(fn)
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_maybe_verify_spatial_handoff":
            return
    raise AssertionError("_process_spatial does not call _maybe_verify_spatial_handoff")


def test_post_handoff_verify_calls_calibration_verify_true():
    fn = _method("_maybe_verify_spatial_handoff")
    calls = _walk_calls(fn)
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "verify":
            for kw in call.keywords:
                if kw.arg == "anchor_consistency_ok":
                    assert isinstance(kw.value, ast.Constant)
                    assert kw.value.value is True
                    return
    raise AssertionError("verify(anchor_consistency_ok=True) not found in post-handoff hook")
