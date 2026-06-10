"""Tests for the LiDAR → room-frame extrinsic (brain/cognition/lidar_calibration.py).

Zero live-brain deps. The contract: identity is a pass-through (inert), the
rotation is rigid (distance-preserving), translation/mount-height apply, the
room view is additive (lidar-frame fields preserved), and the module imports
nothing that could write beliefs/memory/policy.
"""
from __future__ import annotations

import math

import pytest

from cognition.lidar_calibration import LidarExtrinsic, load_extrinsic


def test_identity_is_passthrough_inert():
    ex = LidarExtrinsic.identity()
    assert ex.is_identity
    for x, z in [(0.0, 0.0), (1.0, 2.0), (-3.4, 0.7)]:
        assert ex.transform_xz(x, z) == (x, 0.0, z)     # room == lidar, y at 0 mount


def test_default_extrinsic_is_identity():
    # no env configured → inert by default
    assert LidarExtrinsic().is_identity
    assert load_extrinsic().is_identity


def test_yaw_is_rigid_distance_preserving():
    ex = LidarExtrinsic(yaw_rad=0.7)
    for x, z in [(1.0, 0.0), (0.0, 1.0), (2.0, -3.0)]:
        xr, yr, zr = ex.transform_xz(x, z)
        assert math.isclose(math.hypot(xr, zr), math.hypot(x, z), rel_tol=1e-9)  # norm preserved
        assert yr == 0.0


def test_yaw_90_maps_forward_to_right():
    ex = LidarExtrinsic(yaw_rad=math.pi / 2)
    # convention: x'=x cosθ + z sinθ, z'=-x sinθ + z cosθ → forward(+z) goes to +x
    xr, _, zr = ex.transform_xz(0.0, 1.0)
    assert math.isclose(xr, 1.0, abs_tol=1e-9) and math.isclose(zr, 0.0, abs_tol=1e-9)
    xr, _, zr = ex.transform_xz(1.0, 0.0)                # right(+x) goes to -z (behind)
    assert math.isclose(xr, 0.0, abs_tol=1e-9) and math.isclose(zr, -1.0, abs_tol=1e-9)


def test_translation_and_mount_height():
    ex = LidarExtrinsic(yaw_rad=0.0, tx_m=0.5, ty_m=1.2, tz_m=-0.3)
    assert ex.transform_xz(0.0, 0.0) == (0.5, 1.2, -0.3)
    assert ex.transform_xz(1.0, 2.0) == (1.5, 1.2, 1.7)
    assert not ex.is_identity


def test_transform_room_is_additive_and_telemetry_only():
    room = {
        "kind": "lidar_room", "frame": "lidar_sensor",
        "walls": [{"start_m": [0.0, 1.0], "end_m": [1.0, 1.0], "confidence": 0.9}],
        "points_m": [[0.0, 1.0], [1.0, 1.0]],
        "dimensions_m": [1.0, 0.0],
    }
    ex = LidarExtrinsic(ty_m=1.5)
    out = ex.transform_room(room)
    # original lidar-frame fields preserved (additive)
    assert out["walls"][0]["start_m"] == [0.0, 1.0]
    # room-frame geometry added at mount height
    assert out["walls"][0]["start_room_m"] == [0.0, 1.5, 1.0]
    assert out["points_room_m"][0] == [0.0, 1.5, 1.0]
    assert out["frame"] == "room"
    assert out["extrinsic"]["is_identity"] is False
    # a frame change is NEVER a belief
    assert out["authority"] == "spatial_telemetry_only"
    assert out["writes_beliefs"] is False


def test_transform_room_identity_preserves_xz():
    room = {"walls": [], "points_m": [[2.0, 3.0], [-1.0, 0.5]]}
    out = LidarExtrinsic.identity().transform_room(room)
    assert out["points_room_m"] == [[2.0, 0.0, 3.0], [-1.0, 0.0, 0.5]]
    assert out["extrinsic"]["is_identity"] is True


def test_deterministic():
    ex = LidarExtrinsic(yaw_rad=0.3, tx_m=0.1, ty_m=1.0, tz_m=0.2)
    room = {"walls": [{"start_m": [0.1, 0.2], "end_m": [0.3, 0.4]}], "points_m": [[1.0, 1.0]]}
    assert ex.transform_room(room) == ex.transform_room(dict(room))


def test_no_forbidden_imports():
    try:
        from jarvis_eval.validation_pack import _scan_hrr_forbidden_imports, _HRR_MODULE_ROOTS
    except Exception:
        pytest.skip("validation_pack unavailable")
    assert any("lidar_calibration" in r for r in _HRR_MODULE_ROOTS), \
        "register lidar_calibration in _HRR_MODULE_ROOTS"
    result = _scan_hrr_forbidden_imports()
    is_clean = result[0] if isinstance(result, tuple) else (
        result.get("is_clean", result.get("clean", True)) if isinstance(result, dict) else bool(result))
    assert is_clean


# ---- per-instance mount config (load_extrinsic) -----------------------------
def test_load_extrinsic_from_config_file(tmp_path, monkeypatch):
    import json
    from cognition import lidar_calibration as lc
    cfg = tmp_path / "lidar_extrinsic.json"
    cfg.write_text(json.dumps({"ty_m": 1.092, "yaw_rad": 0.0, "tx_m": 0.0, "tz_m": 0.0}))
    monkeypatch.setattr(lc, "_EXTRINSIC_FILE", str(cfg))
    ex = lc.load_extrinsic()
    assert abs(ex.ty_m - 1.092) < 1e-9 and not ex.is_identity
    # the configured mount height lands in the room Y (scan-plane height)
    assert abs(ex.transform_xz(0.0, 0.0)[1] - 1.092) < 1e-9


def test_load_extrinsic_missing_file_is_identity(tmp_path, monkeypatch):
    from cognition import lidar_calibration as lc
    monkeypatch.setattr(lc, "_EXTRINSIC_FILE", str(tmp_path / "nope.json"))
    for k in ("JARVIS_LIDAR_YAW_RAD", "JARVIS_LIDAR_TX_M", "JARVIS_LIDAR_MOUNT_HEIGHT_M", "JARVIS_LIDAR_TZ_M"):
        monkeypatch.delenv(k, raising=False)
    assert lc.load_extrinsic().is_identity     # product default: no mount => no-op
