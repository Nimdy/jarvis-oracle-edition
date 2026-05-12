"""Gestation dashboard status regressions."""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Gestation imports config.py (Pydantic). Skip this focused regression when the
# lightweight local test env does not include full runtime dependencies.
pytest.importorskip("pydantic")

from consciousness import gestation as gestation_mod
from consciousness.gestation import GestationManager


def test_get_status_falls_back_to_birth_certificate_after_graduation(
    tmp_path,
    monkeypatch,
):
    cert = tmp_path / "gestation_summary.json"
    cert.write_text(
        json.dumps(
            {
                "instance_id": "jarvis_test",
                "gestation_started": 1000.0,
                "gestation_completed": 2000.0,
                "duration_s": 1000.0,
                "readiness_at_birth": {
                    "overall": 0.895,
                    "self_knowledge": 1.0,
                    "personality_emergence": 0.0,
                    "policy_experience": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gestation_mod, "BIRTH_CERTIFICATE_PATH", cert)

    cfg = SimpleNamespace(
        min_duration_s=7200.0,
        min_measured_deltas=10,
        readiness_threshold=0.8,
        readiness_threshold_waiting=0.6,
        person_sustained_s=30.0,
        min_memories_for_ready=50,
    )
    mgr = GestationManager(cfg)
    # Simulate post-birth runtime growth for fields that are expectedly low at birth.
    monkeypatch.setattr(mgr, "_get_experience_count", lambda: 25)
    monkeypatch.setattr(mgr, "_get_trait_deviation", lambda: 0.12)
    monkeypatch.setattr(mgr, "_get_delta_measured", lambda: 6)

    status = mgr.get_status()

    assert status["active"] is False
    assert status["graduated"] is True
    assert status["readiness_source"] == "birth_certificate"
    assert status["readiness"]["overall"] == 0.895
    assert status["readiness"]["recommendation"] == "graduated"
    assert status["birth_snapshot"]["instance_id"] == "jarvis_test"
    assert status["post_birth_progress"]["policy_experience"] == 0.5
    assert status["post_birth_progress"]["personality_emergence"] == 0.4
