"""Spatial estimation — calibration math, prior-based distance, smoothing, uncertainty."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CLASS_MOVE_THRESHOLDS,
    CONFIDENCE_THRESHOLD_TRACK,
    STABLE_WINDOWS_REQUIRED,
    SpatialObservation,
)
from perception.calibration import CalibrationManager, CameraIntrinsics, RoomTransform
from perception.spatial import SpatialEstimator, SpatialRecorder, SpatialReplayer


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


def _make_calibration(
    focal: float = 800.0,
    cam_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CalibrationManager:
    """Create an in-memory calibration manager with valid state."""
    cal = CalibrationManager.__new__(CalibrationManager)
    cal._intrinsics = CameraIntrinsics(focal_length_px=focal)
    cal._transform = RoomTransform(camera_position_m=cam_pos)
    cal._version = 1
    cal._state = "valid"
    cal._last_verified_ts = time.time()
    cal._created_ts = time.time()
    cal._anchor_consistency_ok = True
    return cal


def test_calibration_starts_invalid():
    cal = CalibrationManager.__new__(CalibrationManager)
    cal._intrinsics = CameraIntrinsics()
    cal._transform = RoomTransform()
    cal._version = 0
    cal._state = "invalid"
    cal._last_verified_ts = 0.0
    cal._created_ts = 0.0
    cal._anchor_consistency_ok = False
    assert cal.state == "invalid"
    assert not cal.is_usable()


def test_calibration_becomes_valid():
    cal = _make_calibration()
    assert cal.state == "valid"
    assert cal.is_usable()
    assert not cal.is_advisory_only()


def test_calibration_becomes_stale():
    cal = _make_calibration()
    cal._last_verified_ts = time.time() - 100000
    cal._refresh_state()
    assert cal.state == "stale"
    assert cal.is_advisory_only()
    assert not cal.is_usable()


def test_calibration_very_old_stays_stale_for_local_advisory():
    cal = _make_calibration()
    cal._last_verified_ts = time.time() - 500000
    cal._refresh_state()
    assert cal.state == "stale"
    assert cal.is_advisory_only()
    assert not cal.is_usable()
    state = cal.get_state()
    assert state["reason"] == "verification_expired_local_only"


def test_calibration_becomes_invalid_on_bad_consistency():
    cal = _make_calibration()
    cal._anchor_consistency_ok = False
    cal._refresh_state()
    assert cal.state == "invalid"


def test_calibration_get_state():
    cal = _make_calibration()
    state = cal.get_state()
    assert state["state"] == "valid"
    assert state["reason"] == "verified_recently"
    assert state["version"] == 1
    assert "intrinsics" in state
    assert "transform" in state


def test_calibration_profiles_roundtrip_and_activation(monkeypatch, tmp_path):
    from perception import calibration as cal_mod

    monkeypatch.setattr(cal_mod, "_CALIBRATION_DIR", tmp_path)
    monkeypatch.setattr(cal_mod, "_CALIBRATION_FILE", tmp_path / "calibration.json")

    cal = cal_mod.CalibrationManager()
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
    matched = cal.match_profile(sig_b, min_score=0.2, include_active=True)
    assert matched is not None
    assert matched[0] == "workspace_b"
    assert cal.activate_profile("workspace_b", stale_reason="profile_handoff_local_only", persist=True)

    cal2 = cal_mod.CalibrationManager()
    assert cal2.profile_count >= 2
    assert cal2.active_profile_id == "workspace_b"


def test_camera_to_room_transform():
    cal = _make_calibration(cam_pos=(1.0, 2.0, 3.0))
    room = cal.camera_to_room((0.5, 0.3, 1.0))
    assert abs(room[0] - 1.5) < 0.001
    assert abs(room[1] - 2.3) < 0.001
    assert abs(room[2] - 4.0) < 0.001


# ---------------------------------------------------------------------------
# Estimation tests
# ---------------------------------------------------------------------------


def test_estimate_returns_none_without_calibration():
    cal = CalibrationManager.__new__(CalibrationManager)
    cal._intrinsics = CameraIntrinsics()
    cal._transform = RoomTransform()
    cal._version = 0
    cal._state = "invalid"
    cal._last_verified_ts = 0.0
    cal._created_ts = 0.0
    cal._anchor_consistency_ok = False
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_1", "cup", (400, 500, 450, 600), 0.9)
    assert obs is None


def test_estimate_returns_none_without_bbox():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_1", "cup", None, 0.9)
    assert obs is None


def test_estimate_returns_none_for_unknown_class():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_1", "alien_artifact", (400, 500, 450, 600), 0.9)
    assert obs is None


def test_estimate_cup_produces_observation():
    cal = _make_calibration(focal=800.0)
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_cup", "cup", (400, 500, 460, 580), 0.9)
    assert obs is not None
    assert obs.entity_id == "obj_cup"
    assert obs.label == "cup"
    assert obs.depth_m > 0
    assert obs.confidence > 0
    assert obs.position_camera_m[2] > 0
    assert obs.position_room_m is not None


def test_estimate_depth_inversely_proportional_to_pixel_size():
    cal = _make_calibration(focal=800.0)
    est = SpatialEstimator(cal)
    obs_large = est.estimate("a", "cup", (400, 400, 500, 500), 0.9)
    obs_small = est.estimate("b", "cup", (400, 400, 440, 440), 0.9)
    assert obs_large is not None and obs_small is not None
    assert obs_small.depth_m > obs_large.depth_m


def test_estimate_confidence_has_calibration_component():
    cal_valid = _make_calibration()
    est_valid = SpatialEstimator(cal_valid)
    obs_valid = est_valid.estimate("a", "cup", (400, 500, 460, 580), 0.9)

    cal_stale = _make_calibration()
    cal_stale._last_verified_ts = time.time() - 100000
    cal_stale._refresh_state()
    est_stale = SpatialEstimator(cal_stale)
    obs_stale = est_stale.estimate("a", "cup", (400, 500, 460, 580), 0.9)

    assert obs_valid is not None and obs_stale is not None
    assert obs_valid.confidence > obs_stale.confidence


# ---------------------------------------------------------------------------
# Track tests
# ---------------------------------------------------------------------------


def test_first_observation_creates_provisional_track():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_cup", "cup", (400, 500, 460, 580), 0.9, timestamp=1.0)
    assert obs is not None
    track = est.update_track(obs)
    assert track.entity_id == "obj_cup"
    assert track.track_status == "provisional"
    assert track.samples == 1


def test_track_promotes_to_stable_after_enough_windows():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    for i in range(STABLE_WINDOWS_REQUIRED + 2):
        obs = est.estimate("obj_cup", "cup", (400, 500, 460, 580), 0.9, timestamp=float(i))
        assert obs is not None
        track = est.update_track(obs)
    assert track.track_status == "stable"
    assert track.authority == AUTHORITY_LEVELS["stable_track"]


def test_track_smoothing_reduces_jitter():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs1 = est.estimate("a", "cup", (400, 500, 460, 580), 0.9, timestamp=1.0)
    est.update_track(obs1)
    obs2 = est.estimate("a", "cup", (402, 502, 462, 582), 0.9, timestamp=2.0)
    track = est.update_track(obs2)
    pos_diff = sum(
        abs(a - b) for a, b in zip(track.position_room_m, obs2.position_room_m)
    )
    assert pos_diff < sum(abs(v) for v in obs2.position_room_m)


def test_decay_stale_tracks():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs = est.estimate("a", "cup", (400, 500, 460, 580), 0.9, timestamp=1.0)
    est.update_track(obs)
    stale = est.decay_stale_tracks(now=100.0)
    assert "a" in stale
    assert est.get_tracks()["a"].track_status == "stale"


def test_reset_for_relocalization_clears_tracks_and_anchors():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    obs = est.estimate("obj_cup", "cup", (400, 500, 460, 580), 0.9, timestamp=1.0)
    est.update_track(obs)
    est.register_anchor("anchor_desk", "desk_plane", "desk", (0.0, -0.5, 1.0))
    assert len(est.get_tracks()) == 1
    assert len(est.get_anchors()) == 1
    est.reset_for_relocalization(profile_id="scene_a", reason="matched_profile")
    assert est.get_tracks() == {}
    assert est.get_anchors() == {}


# ---------------------------------------------------------------------------
# Anchor tests
# ---------------------------------------------------------------------------


def test_register_anchor():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    anchor = est.register_anchor(
        "anchor_desk", "desk_plane", "desk", (0.0, -0.5, 1.0),
    )
    assert anchor.anchor_id == "anchor_desk"
    assert anchor.authority == AUTHORITY_LEVELS["stable_anchor"]
    assert "anchor_desk" in est.get_anchors()


def test_anchor_conflict_detected():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    est.register_anchor(
        "anchor_mon", "monitor_center", "monitor", (0.0, 0.0, 1.0),
    )
    obs = SpatialObservation(
        entity_id="obj_mon",
        label="monitor",
        depth_m=2.0,
        position_camera_m=(0.0, 0.0, 2.0),
        position_room_m=(0.0, 0.0, 2.0),
        confidence=0.8,
        calibration_version=1,
    )
    assert est.check_anchor_conflict(obs) is True


def test_no_anchor_conflict_when_close():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    est.register_anchor(
        "anchor_mon", "monitor_center", "monitor", (0.0, 0.0, 1.0),
    )
    obs = SpatialObservation(
        entity_id="obj_mon",
        label="monitor",
        depth_m=1.0,
        position_camera_m=(0.0, 0.0, 1.0),
        position_room_m=(0.05, 0.02, 1.03),
        confidence=0.8,
        calibration_version=1,
    )
    assert est.check_anchor_conflict(obs) is False


# ---------------------------------------------------------------------------
# Replay harness tests
# ---------------------------------------------------------------------------


def test_replay_harness_record_and_replay():
    recorder = SpatialRecorder()
    recorder.start()
    recorder.record({"entities": [{"entity_id": "a", "label": "cup",
                                    "bbox": [400, 500, 460, 580],
                                    "confidence": 0.9}]})
    recorder.record({"entities": [{"entity_id": "a", "label": "cup",
                                    "bbox": [402, 502, 462, 582],
                                    "confidence": 0.88}]})
    recorder.stop()
    assert recorder.count == 2

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    try:
        saved = recorder.save(os.path.basename(path))
        recorder2 = SpatialRecorder()
        recorder2.start()
        recorder2.record({"entities": []})
        recorder2.stop()
        saved_path = recorder2.save("test_replay.jsonl")

        cal = _make_calibration()
        est = SpatialEstimator(cal)
        replayer = SpatialReplayer(saved)
        results = replayer.replay(est)
        assert len(results) == 2
        assert len(results[0]["observations"]) >= 1
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Get state
# ---------------------------------------------------------------------------


def test_get_state_structure():
    cal = _make_calibration()
    est = SpatialEstimator(cal)
    state = est.get_state()
    assert "total_tracks" in state
    assert "stable_tracks" in state
    assert "total_anchors" in state
    assert "calibration" in state
    assert state["calibration"]["state"] == "valid"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
