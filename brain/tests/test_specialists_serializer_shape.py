"""Contract: ``self_improve.specialists`` serializer shape is stable.

The ``_build_self_improve_cache`` helper used to silently degrade the
``specialists`` sub-payload into a bare ``{}`` when ``_build_si_specialists``
raised. That violated the consumer contract (the dashboard, validation
pack, and truth probe all index into ``specialists.specialists`` as a
list and ``specialists.distillation`` as a dict) and hid real failures.

These regressions lock the post-fix shape in both the happy path and the
forced-exception path.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from dashboard import snapshot as snapshot_module
from dashboard.snapshot import _build_self_improve_cache


class _StubOrchestrator:
    def get_status(self) -> dict:
        return {"phase": "idle", "tick": 0}


class _StubConsciousness:
    def __init__(self, *, raise_on_scanner: bool = False) -> None:
        self._self_improve_orchestrator = _StubOrchestrator()
        self._raise_on_scanner = raise_on_scanner

    def get_scanner_state(self) -> dict:
        if self._raise_on_scanner:
            raise RuntimeError("scanner offline")
        return {"status": "idle"}


class _StubEngine:
    def __init__(self, *, raise_on_scanner: bool = False) -> None:
        self._consciousness = _StubConsciousness(
            raise_on_scanner=raise_on_scanner
        )

    def get_hemisphere_state(self) -> dict:
        return {
            "hemisphere_state": {"hemispheres": []},
            "distillation": {
                "teachers": {},
                "total_signals": 0,
                "total_quarantined": 0,
            },
            "tier1_gating": {"failure_counts": {}, "disabled_for_session": []},
        }


def _assert_specialists_shape(payload: dict) -> None:
    """Hard shape contract that both paths must preserve."""
    assert isinstance(payload, dict), (
        f"specialists must be a dict; got {type(payload).__name__}"
    )
    assert "specialists" in payload, "missing 'specialists' key"
    assert "distillation" in payload, "missing 'distillation' key"
    assert isinstance(payload["specialists"], list), (
        "specialists.specialists must be a list; got "
        f"{type(payload['specialists']).__name__}"
    )
    assert isinstance(payload["distillation"], dict), (
        "specialists.distillation must be a dict; got "
        f"{type(payload['distillation']).__name__}"
    )


def test_specialists_shape_on_happy_path():
    engine = _StubEngine()
    cache = _build_self_improve_cache(engine)

    assert cache["active"] is True
    assert "specialists" in cache
    _assert_specialists_shape(cache["specialists"])
    assert cache["specialists"]["specialists"] == []
    assert cache["specialists"]["distillation"] == {
        "total_signals": 0,
        "total_quarantined": 0,
    }
    assert "_error" not in cache["specialists"]


def test_specialists_shape_preserved_on_exception(monkeypatch):
    def _boom(_engine):
        raise RuntimeError("synthetic serializer failure")

    monkeypatch.setattr(snapshot_module, "_build_si_specialists", _boom)

    engine = _StubEngine()
    cache = _build_self_improve_cache(engine)

    assert cache["active"] is True
    _assert_specialists_shape(cache["specialists"])
    assert cache["specialists"]["_error"] == "RuntimeError"


def test_specialists_shape_never_collapses_to_raw_empty_dict(monkeypatch):
    def _boom(_engine):
        raise ValueError("another synthetic failure")

    monkeypatch.setattr(snapshot_module, "_build_si_specialists", _boom)

    engine = _StubEngine()
    cache = _build_self_improve_cache(engine)

    spec = cache["specialists"]
    assert spec != {}, (
        "specialists must never be a bare {}; consumers rely on the list/dict "
        "contract"
    )
    _assert_specialists_shape(spec)


def test_specialists_list_is_populated_for_configured_focuses():
    """Happy path with real hemisphere rows still produces well-shaped output."""

    class _EngineWithHemispheres(_StubEngine):
        def get_hemisphere_state(self):
            return {
                "hemisphere_state": {
                    "hemispheres": [
                        {
                            "focus": "diagnostic",
                            "status": "training",
                            "best_accuracy": 0.41,
                            "best_training_accuracy": 0.55,
                            "best_validation_accuracy": 0.38,
                            "total_attempts": 3,
                            "network_count": 2,
                            "migration_readiness": 0.1,
                        }
                    ]
                },
                "distillation": {
                    "teachers": {},
                    "total_signals": 5,
                    "total_quarantined": 0,
                },
                "tier1_gating": {
                    "failure_counts": {},
                    "disabled_for_session": [],
                },
            }

    cache = _build_self_improve_cache(_EngineWithHemispheres())
    _assert_specialists_shape(cache["specialists"])
    specialists = cache["specialists"]["specialists"]
    assert len(specialists) == 1
    assert specialists[0]["focus"] == "diagnostic"
    assert specialists[0]["maturity"] == "bootstrap"
