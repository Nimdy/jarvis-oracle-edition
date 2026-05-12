"""Regression tests for boot stabilization visibility and persistence merge behavior."""

from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from consciousness.consciousness_system import ConsciousnessState
from memory.persistence import ConsciousnessPersistence


class _StateObj:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class _FakeConsciousnessSystem:
    def __init__(self) -> None:
        self.evolution = SimpleNamespace(
            get_state=lambda: _StateObj({"current_stage": "basic_awareness", "transcendence_level": 0.1})
        )
        self.observer = SimpleNamespace(
            get_state=lambda: _StateObj({"awareness_level": 0.3, "observation_count": 1})
        )
        self.driven_evolution = SimpleNamespace(get_state=lambda: {"active_capabilities": []})
        self.config = SimpleNamespace(to_dict=lambda: {"thought_weights": {}})
        self.governor = SimpleNamespace(
            mutation_count=1,
            rollback_count=0,
            last_mutation_time=0.0,
            _mutation_timestamps=[],
        )
        self.analytics = SimpleNamespace(get_full_state=lambda: {"confidence": {"avg": 0.5}})


def test_consciousness_state_exposes_boot_stabilization_fields():
    state = ConsciousnessState(
        boot_stabilization_active=True,
        boot_stabilization_remaining_s=123.456,
    )
    payload = state.to_dict()
    assert payload["boot_stabilization_active"] is True
    assert payload["boot_stabilization_remaining_s"] == 123.5


def test_autosave_merge_preserves_gestation_keys_without_info_log_noise(tmp_path, caplog, monkeypatch):
    cp = ConsciousnessPersistence()
    cp._path = tmp_path / "consciousness_state.json"
    cp._path.write_text(
        json.dumps(
            {
                "gestation_in_progress": True,
                "gestation_complete": False,
                "gestation_completed_at": 12345.0,
            }
        )
    )

    # Keep this test focused on the merge path behavior.
    monkeypatch.setattr("memory.persistence.extended_persistence.save_all", lambda: {})

    fake_system = _FakeConsciousnessSystem()
    fake_engine = SimpleNamespace(_restore_complete=True)

    with caplog.at_level(logging.INFO):
        assert cp.save_from_system(fake_system, engine=fake_engine)

        # Simulate an external write dropping gestation keys; sticky merge cache should keep them.
        cp._path.write_text(json.dumps({"_provenance": {"schema_version": 2}}))
        assert cp.save_from_system(fake_system, engine=fake_engine)

    final_state = json.loads(cp._path.read_text())
    assert final_state["gestation_in_progress"] is True
    assert final_state["gestation_complete"] is False
    assert final_state["gestation_completed_at"] == 12345.0

    info_messages = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    assert all("Loaded consciousness state from" not in msg for msg in info_messages)
