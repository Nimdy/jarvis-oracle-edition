"""Tests for lidar↔camera object fusion (brain/cognition/lidar_fusion.py).

The contract: camera supplies the LABEL (a hypothesis), lidar supplies the metric
RANGE (truth); a fused object is telemetry, never a belief; the yaw self-estimator
refuses to declare a yaw it hasn't earned (needs angular spread).
"""
from __future__ import annotations

import math

import pytest

from cognition.lidar_fusion import (
    camera_bearing, lidar_range_at, fuse, FusedObject, YawEstimator, TWO_PI,
)


def _profile(n=360, **bearing_deg_to_range):
    """360-bin profile, None everywhere except the given bearing(deg)->range_m."""
    prof = [None] * n
    for deg, r in bearing_deg_to_range.items():
        prof[int(deg) % n] = r
    return prof


def test_camera_bearing_centre_is_zero_right_is_positive():
    assert abs(camera_bearing((950, 0, 970, 100), principal_x=960.0, focal_px=800.0)) < 1e-9
    # bbox centre to the RIGHT of the optical axis → +bearing (room +X)
    assert camera_bearing((1200, 0, 1300, 100), 960.0, 800.0) > 0
    assert camera_bearing((100, 0, 200, 100), 960.0, 800.0) < 0


def test_lidar_range_at_window_picks_nearest():
    prof = _profile(**{"90": 3.0, "91": 1.2})        # two returns near bearing 90°
    bin_w = TWO_PI / 360
    r = lidar_range_at(prof, math.radians(90.5), bin_w, window_bins=2)
    assert r == 1.2                                   # nearest in the window
    assert lidar_range_at(prof, math.radians(200), bin_w) is None   # honest no-measurement


def test_fuse_attaches_lidar_range_at_camera_bearing():
    # a 'chair' dead-ahead (bbox centred) → room bearing 0 → lidar profile at bearing 0
    prof = _profile(**{"0": 2.5})
    ents = [{"entity_id": "e1", "label": "chair", "confidence": 0.8,
             "bbox": (955, 100, 965, 300)}]      # centred on principal_x=960
    out = fuse(ents, prof, yaw_rad=0.0, focal_px=800.0, principal_x=960.0, mount_height_m=1.092)
    assert len(out) == 1
    f = out[0]
    assert f.label == "chair" and abs(f.label_confidence - 0.8) < 1e-9
    assert f.has_lidar_range and abs(f.range_m - 2.5) < 1e-6
    # dead-ahead → room position ≈ (0, 1.092, 2.5)
    x, y, z = f.position_room_m
    assert abs(x) < 1e-3 and abs(y - 1.092) < 1e-9 and abs(z - 2.5) < 1e-3


def test_fuse_yaw_maps_camera_to_lidar_profile():
    # camera dead-ahead, but yaw=+10° means the lidar sees the object at bearing -10° (=350°)
    prof = _profile(**{"350": 1.3})
    ents = [{"entity_id": "p", "label": "person", "confidence": 0.9, "bbox": (955, 0, 965, 400)}]
    out = fuse(ents, prof, yaw_rad=math.radians(10), principal_x=960.0, focal_px=800.0)
    assert out[0].has_lidar_range and abs(out[0].range_m - 1.3) < 1e-6


def test_fuse_camera_only_when_no_lidar_return():
    # camera sees a label but the lidar has nothing at that bearing → honest camera-only
    out = fuse([{"entity_id": "x", "label": "mug", "confidence": 0.6, "bbox": (955, 0, 965, 50)}],
               _profile(), yaw_rad=0.0, principal_x=960.0)
    assert len(out) == 1
    assert out[0].label == "mug" and out[0].range_m is None
    assert out[0].has_lidar_range is False and out[0].position_room_m is None


def test_fuse_is_telemetry_only_and_label_is_hypothesis():
    out = fuse([{"entity_id": "e", "label": "tv", "confidence": 0.7, "bbox": (955, 0, 965, 50)}],
               _profile(**{"0": 2.0}), principal_x=960.0)
    d = out[0].to_dict()
    assert d["authority"] == "spatial_telemetry_only" and d["writes_beliefs"] is False
    assert d["label_provenance"] == "camera_hypothesis"          # never asserted as fact
    assert d["range_provenance"] == "lidar_metric"


def test_fuse_skips_entities_without_bbox():
    out = fuse([{"entity_id": "n", "label": "noise", "confidence": 0.5}], _profile(**{"0": 2.0}))
    assert out == []


def test_yaw_estimator_recovers_offset_with_spread():
    est = YawEstimator(min_pairs=8, min_spread_rad=math.radians(12))
    true_yaw = math.radians(10)
    # sweep camera bearings ±30°; lidar bearing = room − yaw
    for deg in range(-30, 31, 5):
        rb = math.radians(deg)
        est.observe(rb, (rb - true_yaw) % TWO_PI)
    got = est.estimate()
    assert got is not None and abs(got - true_yaw) < math.radians(0.5)


def test_yaw_estimator_refuses_without_spread():
    est = YawEstimator(min_pairs=8, min_spread_rad=math.radians(12))
    for _ in range(20):
        est.observe(math.radians(2.0), math.radians(-8.0))    # all the same point
    assert est.estimate() is None                              # won't declare an unearned yaw


def test_no_forbidden_imports():
    try:
        from jarvis_eval.validation_pack import _scan_hrr_forbidden_imports, _HRR_MODULE_ROOTS
    except Exception:
        pytest.skip("validation_pack unavailable")
    assert any("lidar_fusion" in r for r in _HRR_MODULE_ROOTS)
    result = _scan_hrr_forbidden_imports()
    is_clean = result[0] if isinstance(result, tuple) else (
        result.get("is_clean", result.get("clean", True)) if isinstance(result, dict) else bool(result))
    assert is_clean
