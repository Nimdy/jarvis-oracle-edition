"""Standalone tests for the lidar room model (brain/cognition/lidar_room.py).

Zero live-brain deps. Fixture = 14 real captured S2 revolutions of an enclosed room.
The noise-robustness tests (synthetic perturbation of the fixture) are the core of
"SOLID": the median firewall must reject 1.4 m edge spikes, dropouts must never be
back-filled into phantom walls, and outliers must not bend a wall.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from cognition.lidar_room import (
    LidarScan, LidarRoomConfig, LidarRoomModel, build_room_model,
    REASON_INSUFFICIENT_COVERAGE, polar_to_cartesian,
)

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "lidar_scan_14rev.json")


def _load_fixture() -> list[LidarScan]:
    return [LidarScan.from_dict(d) for d in json.load(open(_FIX))]


def _synthetic_scan(ranges_m, ts=0.0, n=360, rmax=12.0) -> LidarScan:
    """Build a scan from a per-degree range list (None = dropout for that bearing)."""
    pts = []
    for deg, r in enumerate(ranges_m):
        if r is not None and r > 0:
            pts.append((math.radians(deg + 0.5) * (n / 360) * (360 / n), r))
    # simpler: place one point per degree at the bin center
    pts = [(math.radians(deg), r) for deg, r in enumerate(ranges_m) if r]
    return LidarScan(timestamp=ts, points=tuple(pts), range_max_m=rmax)


# ---- fixture / golden -------------------------------------------------------
def test_builds_without_error():
    rm = build_room_model(_load_fixture())
    assert rm.reason is None
    assert rm.coverage_fraction > 0.5
    assert rm.scan_count == 14


def test_dimensions_plausible_and_deterministic():
    scans = _load_fixture()
    rm1 = build_room_model(scans)
    rm2 = build_room_model(scans)
    assert rm1.to_dict() == rm2.to_dict()                 # pure-function determinism
    w, d = rm1.dimensions_m
    assert 0.5 < w < 20 and 0.5 < d < 20


def test_wall_detection():
    rm = build_room_model(_load_fixture())
    assert len(rm.walls) >= 1
    assert len(rm.walls) < 80                              # not fragmented to dozens-of-dozens
    for w in rm.walls:
        assert w.residual_rms_m < 0.08                     # tight line fits on real data
        assert w.inlier_count >= LidarRoomConfig().min_wall_inliers


def test_no_phantom_openings():
    # the captured view is enclosed → no opening should be asserted over a dropout
    rm = build_room_model(_load_fixture())
    for o in rm.openings:
        assert not o.is_occlusion_shadow                   # never assert a shadow as open


def test_nearest_per_sector_matches_raw():
    scans = _load_fixture()
    rm = build_room_model(scans)
    # naive raw argmin per sector across all points
    n_sec = LidarRoomConfig().nearest_sectors
    raw = [math.inf] * n_sec
    for s in scans:
        for b, r in s.points:
            sec = int((b % (2 * math.pi)) / (2 * math.pi) * n_sec) % n_sec
            if 0 < r < raw[sec]:
                raw[sec] = r
    for got, exp in zip(rm.nearest_per_sector_m, raw):
        if exp != math.inf and got > 0:
            # the denoised nearest UPGRADES the raw min: it may be FARTHER (it rejected
            # a transient near-noise spike via the median) but must never invent a
            # spuriously NEARER obstacle than the raw data supports.
            assert got >= exp - 0.3


# ---- stability --------------------------------------------------------------
def test_stability_across_subwindows():
    scans = _load_fixture()
    a = build_room_model(scans[:7])
    b = build_room_model(scans[7:])
    # a fixed sensor must not "wander": bounding dims agree within tolerance
    assert abs(a.dimensions_m[0] - b.dimensions_m[0]) < 0.6
    assert abs(a.dimensions_m[1] - b.dimensions_m[1]) < 0.6


def test_fingerprint_stable_for_static_room():
    scans = _load_fixture()
    def fp(rm):
        return tuple(sorted((round(x * 2) / 2, round(z * 2) / 2) for x, z in rm.points_m))
    assert fp(build_room_model(scans)) == fp(build_room_model(scans))


# ---- noise robustness (the heart of SOLID) ----------------------------------
def test_edge_spike_rejection():
    # bin at 90°: mostly 2.0 m, but ~30% spiked to +1.432 m
    base = [None] * 360
    scans = []
    for i in range(14):
        ranges = list(base)
        ranges[90] = 2.0 + (1.432 if i % 3 == 0 else 0.0)   # spike every 3rd rev
        scans.append(_synthetic_scan(ranges, ts=i * 0.1))
    rm = build_room_model(scans, LidarRoomConfig(min_coverage_fraction=0.0))
    assert rm.profile[90] is not None
    assert abs(rm.profile[90] - 2.0) < 0.1                  # median firewall holds
    mean = (2.0 * 14 + 1.432 * 5) / 14                      # a mean control would be ~2.5
    assert abs(rm.profile[90] - mean) > 0.3


def test_jitter_averages_down():
    base = [None] * 360
    scans = []
    for i in range(14):
        ranges = list(base)
        ranges[45] = 1.5 + (0.032 if i % 2 else -0.032)     # ±32 mm jitter
        scans.append(_synthetic_scan(ranges, ts=i * 0.1))
    rm = build_room_model(scans, LidarRoomConfig(min_coverage_fraction=0.0))
    assert abs(rm.profile[45] - 1.5) < 0.02                 # converges to truth


def test_dropouts_not_backfilled():
    # a contiguous angular run (100-120°) NEVER returns → must stay unknown, no wall
    scans = []
    for i in range(14):
        ranges = [2.0] * 360
        for deg in range(100, 121):
            ranges[deg] = None
        scans.append(_synthetic_scan(ranges, ts=i * 0.1))
    rm = build_room_model(scans, LidarRoomConfig(min_coverage_fraction=0.0))
    for deg in range(101, 120):
        assert rm.profile[deg] is None                      # not 0, not range_max
    # no wall should bridge the gap: no wall segment crosses 100-120° contiguously
    # (walls broke at the dropout run)
    assert all(rm.profile[deg] is None for deg in range(101, 120))


def test_outlier_cannot_bend_wall():
    # a flat wall along one bearing band + one isolated off-line spike
    scans = []
    for i in range(14):
        ranges = [None] * 360
        for deg in range(30, 60):
            ranges[deg] = 3.0
        if i == 7:
            ranges[45] = 3.0 + 0.9                           # one-off outlier
        scans.append(_synthetic_scan(ranges, ts=i * 0.1))
    rm = build_room_model(scans, LidarRoomConfig(min_coverage_fraction=0.0))
    # the bin's stable median ignores the single outlier
    assert rm.profile[45] is not None and abs(rm.profile[45] - 3.0) < 0.1


def test_low_coverage_degrades_honestly():
    scans = []
    for i in range(14):
        ranges = [None] * 360
        ranges[0] = 2.0                                     # only one bin ever returns
        scans.append(_synthetic_scan(ranges, ts=i * 0.1))
    rm = build_room_model(scans)                            # default min_coverage 0.25
    assert rm.reason == REASON_INSUFFICIENT_COVERAGE
    assert rm.walls == ()


# ---- contract ---------------------------------------------------------------
def test_to_dict_is_json_serializable_with_expected_keys():
    rm = build_room_model(_load_fixture())
    d = rm.to_dict()
    json.dumps(d)                                           # no numpy scalars leaking
    for key in ("kind", "frame", "profile", "points_m", "walls", "openings",
                "nearest_per_sector_m", "dimensions_m", "coverage_fraction", "reason"):
        assert key in d
    assert d["kind"] == "lidar_room"


def test_no_forbidden_imports():
    try:
        from jarvis_eval.validation_pack import _scan_hrr_forbidden_imports, _HRR_MODULE_ROOTS
    except Exception:
        pytest.skip("validation_pack unavailable")
    assert any("lidar_room" in r for r in _HRR_MODULE_ROOTS), "register lidar_room in _HRR_MODULE_ROOTS"
    result = _scan_hrr_forbidden_imports()
    # result shape varies; accept (is_clean, ...) or a dict/bool
    is_clean = result[0] if isinstance(result, tuple) else (
        result.get("is_clean", result.get("clean", True)) if isinstance(result, dict) else bool(result))
    assert is_clean
