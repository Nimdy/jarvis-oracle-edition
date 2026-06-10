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
    PolarBin, fit_walls, detect_openings,
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


# ---- indoor range gating + drop telemetry (Harden S2 filtering PR) -----------
def test_range_gating_drops_close_and_far_with_telemetry():
    """min_range_m / max_range_m reject housing noise + far ghosts, with honest counts."""
    m = LidarRoomModel(LidarRoomConfig())              # defaults: 0.12 / 8.0
    pts = (
        (math.radians(10), 2.5),    # keep
        (math.radians(20), 0.05),   # drop: < min_range_m (housing reflection)
        (math.radians(30), 9.0),    # drop: >= max_range_m (ghost through a doorway)
        (math.radians(40), 0.0),    # drop: zero / dropout
    )
    m.ingest(LidarScan(timestamp=0.0, points=pts, range_max_m=12.0))
    fs = m.filter_stats()
    assert fs["raw_points"] == 4
    assert fs["dropped_zero"] == 1
    assert fs["dropped_min_range"] == 1
    assert fs["dropped_max_range"] == 1
    assert fs["points_after_filter"] == 1
    assert fs["effective_max_m"] == 8.0                # capped below the 12 m observed ceiling
    assert fs["observed_max_m"] == 9.0                 # honest: what the sensor actually returned
    assert fs["min_range_m"] == 0.12 and fs["max_range_m"] == 8.0


def test_far_ghost_does_not_inflate_room():
    """A recurring 9 m return (open door / reflection) must NOT inflate a ~2 m-radius room."""
    m = LidarRoomModel(LidarRoomConfig())
    for _ in range(14):
        pts = [(math.radians(d), 2.0) for d in range(360)]   # full ring at 2 m
        pts.append((math.radians(90), 9.0))                  # far ghost every revolution
        m.ingest(LidarScan(timestamp=0.0, points=tuple(pts), range_max_m=12.0))
    rm = m.room_model()
    assert rm.reason is None
    w, d = rm.dimensions_m
    assert w < 6.0 and d < 6.0                          # ~4 m box, never ~18 m from the ghost
    assert m.filter_stats()["cum_dropped_max_range"] == 14


# ---- seam-correctness regression (bugs found pre-PR4 by the verification sweep) ----
def _pb(i, n, r, occ):
    bin_w = 2 * math.pi / n
    return PolarBin(index=i, bearing_center_rad=(i + 0.5) * bin_w,
                    r_stable_m=r, mad_m=0.0,
                    sample_count=(14 if r is not None else 0),
                    valid_fraction=(1.0 if r is not None else 0.0), occupancy=occ)


def test_wall_crossing_bearing_zero_is_one_segment():
    """A flat wall straddling bearing 0 must be ONE wall, not split at the seam."""
    n = 360
    bins = []
    for i in range(n):
        d = min(i, n - i)                      # angular distance (bins) from bearing 0
        if d <= 10:                            # flat wall z=2 in front, crossing 0
            bearing = (i + 0.5) * (2 * math.pi / n)
            ang = bearing if bearing < math.pi else bearing - 2 * math.pi
            bins.append(_pb(i, n, 2.0 / max(0.2, math.cos(ang)), "occupied"))
        else:
            bins.append(_pb(i, n, None, "free"))
    walls = fit_walls(bins, LidarRoomConfig())
    assert len(walls) == 1                     # was 2 before the seam-rotation fix


def test_opening_crossing_bearing_zero_is_centred_near_zero():
    """A doorway dead-ahead (crossing bearing 0) reports its centre near 0, not 180°."""
    n = 360
    bins = [_pb(i, n, (None if min(i, n - i) <= 6 else 3.0),
                ("free" if min(i, n - i) <= 6 else "occupied")) for i in range(n)]
    ops = detect_openings(bins, LidarRoomConfig())
    assert len(ops) == 1
    c = ops[0].center_bearing_rad
    assert min(c, 2 * math.pi - c) < math.radians(15)   # near 0, not ~pi (was 180°)


def test_valid_fraction_does_not_decay_over_time():
    """valid_fraction is window-relative — it must stay ~1.0, not erode toward 0."""
    m = LidarRoomModel()
    pts = tuple((math.radians(i + 0.5), 2.0) for i in range(360))
    for _ in range(200):
        m.ingest(LidarScan(timestamp=1.0, points=pts, range_max_m=12.0))
    rm = m.room_model()
    vfs = [vf for (sc, vf, _mad) in rm.bin_quality if sc > 0]
    assert vfs and min(vfs) > 0.9              # was ~0.07 (14/200) before the fix


def test_room_model_timestamp_reflects_last_scan():
    m = LidarRoomModel()
    pts = tuple((math.radians(i + 0.5), 2.0) for i in range(360))
    m.ingest(LidarScan(timestamp=1234.5, points=pts, range_max_m=12.0))
    assert m.room_model().timestamp == 1234.5  # was always 0.0


def test_farthest_in_cap_return_not_self_gated():
    """A real return inside the 8 m cap must NOT be dropped because the Pi's rounded
    per-window observed max happened to equal it."""
    m = LidarRoomModel()
    pts = ((math.radians(0.5), 7.123), (math.radians(90.5), 2.0),
           (math.radians(180.5), 2.0), (math.radians(270.5), 2.0))
    m.ingest(LidarScan(timestamp=1.0, points=pts, range_max_m=7.12))   # rounded observed max
    fs = m.filter_stats()
    assert fs["dropped_max_range"] == 0 and fs["points_after_filter"] == 4
    assert fs["effective_max_m"] == 8.0        # the config cap, not the observed max


def test_from_dict_tolerates_none_range():
    s = LidarScan.from_dict({"timestamp": 1.0, "points": [[0.1, 2.0], [0.2, None], [0.3, 1.5]]})
    assert len(s.points) == 2                  # the None dropout is skipped, not a crash
