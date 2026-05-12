"""Spatial 1.5 relocalization tests for profile matching/handoff state."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from perception.calibration import CalibrationManager, RoomTransform
from perception import calibration as cal_mod


def test_scene_profile_id_is_stable(monkeypatch, tmp_path):
    monkeypatch.setattr(cal_mod, "_CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(cal_mod, "_CALIBRATION_FILE", tmp_path / "calibration.json")

    cal = CalibrationManager()
    sig = {"labels": ["monitor", "keyboard"], "display_kinds": ["monitor"], "regions": ["desk_center"], "entity_count": 2}
    pid1 = cal.suggest_profile_id(sig)
    pid2 = cal.suggest_profile_id(sig)
    assert pid1 == pid2
    assert pid1.startswith("scene_")


def test_profile_match_and_handoff_activation(monkeypatch, tmp_path):
    monkeypatch.setattr(cal_mod, "_CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(cal_mod, "_CALIBRATION_FILE", tmp_path / "calibration.json")

    cal = CalibrationManager()
    cal.setup_pi_camera_defaults(frame_width=640, frame_height=480, camera_position_m=(0.0, 1.2, 0.0))
    sig_a = {"labels": ["monitor", "keyboard"], "display_kinds": ["monitor"], "regions": ["desk_center"], "entity_count": 2}
    sig_b = {"labels": ["tv", "chair"], "display_kinds": ["tv"], "regions": ["desk_left"], "entity_count": 2}
    cal.upsert_profile("workspace_a", scene_signature=sig_a, persist=True)
    cal.upsert_profile(
        "workspace_b",
        scene_signature=sig_b,
        transform=RoomTransform(camera_position_m=(1.0, 0.0, 0.0)),
        persist=True,
    )

    match = cal.match_profile(sig_b, min_score=0.2, include_active=True)
    assert match is not None
    assert match[0] == "workspace_b"
    assert match[1] > 0.7

    assert cal.activate_profile("workspace_b", stale_reason="profile_handoff_local_only", persist=True)
    state = cal.get_state()
    assert state["active_profile_id"] == "workspace_b"
    assert state["state"] == "stale"
    assert state["reason"] == "profile_handoff_local_only"


def test_profile_match_returns_none_for_unrelated_signature(monkeypatch, tmp_path):
    monkeypatch.setattr(cal_mod, "_CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(cal_mod, "_CALIBRATION_FILE", tmp_path / "calibration.json")

    cal = CalibrationManager()
    sig_a = {"labels": ["monitor", "keyboard"], "display_kinds": ["monitor"], "regions": ["desk_center"], "entity_count": 2}
    sig_c = {"labels": ["fridge"], "display_kinds": [], "regions": ["kitchen"], "entity_count": 1}
    cal.upsert_profile("workspace_a", scene_signature=sig_a, persist=False)
    assert cal.match_profile(sig_c, min_score=0.9, include_active=True) is None
