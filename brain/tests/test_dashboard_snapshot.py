"""Dashboard snapshot helper regressions."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.snapshot import _build_post_gestation_snapshot


def test_post_gestation_snapshot_returns_base_without_certificate(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    snap = _build_post_gestation_snapshot(engine=None, partial_snapshot={})
    assert snap == {"active": False}


def test_post_gestation_snapshot_uses_birth_certificate_and_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    jarvis_dir = tmp_path / ".jarvis"
    jarvis_dir.mkdir(parents=True, exist_ok=True)
    (jarvis_dir / "gestation_summary.json").write_text(
        json.dumps(
            {
                "instance_id": "jarvis_test",
                "gestation_started": 100.0,
                "gestation_completed": 8200.0,
                "duration_s": 8100.0,
                "readiness_at_birth": {
                    "overall": 0.895,
                    "self_knowledge": 1.0,
                    "policy_experience": 0.0,
                    "personality_emergence": 0.0,
                },
                "directives_completed": 31,
                "research_jobs_completed": 17,
            }
        ),
        encoding="utf-8",
    )

    class _Engine:
        _experience_buffer = [0] * 25

    partial_snapshot = {
        "personality": {
            "rollback": {
                "current_traits": {"Technical": 0.8, "Empathetic": 0.2},
            }
        },
        "autonomy": {"delta_tracker": {"total_measured": 6}},
    }
    snap = _build_post_gestation_snapshot(engine=_Engine(), partial_snapshot=partial_snapshot)

    assert snap["active"] is False
    assert snap["graduated"] is True
    assert snap["readiness_source"] == "birth_certificate"
    assert snap["readiness"]["overall"] == 0.895
    assert snap["birth_snapshot"]["instance_id"] == "jarvis_test"
    assert snap["post_birth_progress"]["policy_experience"] == 0.5
    assert snap["post_birth_progress"]["personality_emergence"] == 1.0
    assert snap["post_birth_progress"]["loop_integrity"] == 0.6
