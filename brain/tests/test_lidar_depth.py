"""Tests for the Tier-3 dense-3D anchoring core (brain/cognition/lidar_depth.py).

The contract — the flagship's whole claim to TRUTH: recover the metric scale/shift from
the lidar row, REFUSE when the evidence is thin (never fabricate a scale), and keep holes
as holes (never interpolate). Tested with synthetic depth — zero frames, zero models.
"""
from __future__ import annotations

import math

import pytest

from cognition.lidar_depth import (
    AnchorResult, anchor_depth_affine, anchor_best_yaw, depth_to_points,
    lidar_plane_row, TWO_PI,
)

F = 470.0      # focal (640x480 IMX, per the live calibration)
PX, PY = 320.0, 240.0


def _synthetic_row_and_profile(s_true, t_true, n_bins=360):
    """Build a disparity row + lidar profile that EXACTLY encode scale=s_true, shift=t_true:
    inv_metric = s*disp + t ⇒ Z = 1/inv ⇒ lidar range = Z/cos(theta) at that bearing."""
    row = [None] * 640
    prof = [None] * n_bins
    bin_w = TWO_PI / n_bins
    for x in range(20, 620, 13):                 # spaced so each lands in a distinct bin
        disp = 0.4 + 0.002 * x                    # varies 0.44..1.64 → real spread
        theta = math.atan2(x - PX, F)
        z = 1.0 / (s_true * disp + t_true)        # metric Z-depth
        rng = z / math.cos(theta)                 # radial range the lidar would measure
        row[x] = disp
        prof[int((theta % TWO_PI) / bin_w) % n_bins] = rng
    return row, prof


def test_anchor_recovers_known_scale_and_shift():
    row, prof = _synthetic_row_and_profile(s_true=2.0, t_true=0.3)
    a = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX, min_inliers=10)
    assert a.valid and a.reason == "ok"
    assert abs(a.scale - 2.0) < 1e-3 and abs(a.shift - 0.3) < 1e-3
    assert a.rms < 1e-4                            # near-exact fit on clean synthetic data


def test_anchor_refuses_without_disparity_spread():
    # every lidar-backed pixel has the SAME disparity → can't solve an affine → refuse
    row = [None] * 640
    prof = [None] * 360
    bin_w = TWO_PI / 360
    for x in range(20, 620, 13):
        theta = math.atan2(x - PX, F)
        row[x] = 1.0                              # constant disparity (degenerate)
        prof[int((theta % TWO_PI) / bin_w)] = 2.0 / math.cos(theta)
    a = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX, min_inliers=10)
    assert not a.valid and a.reason == "degenerate_no_disp_spread"


def test_anchor_refuses_with_too_few_lidar_pixels():
    row = [0.5, 0.7, 0.9] + [None] * 637          # only 3 disparities, no lidar backing
    a = anchor_depth_affine(row, [None] * 360, yaw_rad=0.0, focal_px=F, principal_x=PX)
    assert not a.valid and a.inlier_count < 20


def test_anchor_is_telemetry_only():
    d = AnchorResult(1.0, 0.0, 50, True, 0.001, "ok").to_dict()
    assert d["authority"] == "spatial_telemetry_only" and d["writes_beliefs"] is False


def test_depth_to_points_lifts_metric_geometry_and_keeps_holes():
    # 4x4 maps; anchor scale=1, shift=0 ⇒ Z = 1/disp. yaw=0 ⇒ room == lidar frame.
    cam_h = 1.219
    anchor = AnchorResult(scale=1.0, shift=0.0, inlier_count=99, valid=True, rms=0.0, reason="ok")
    disp = [[0.5 for _ in range(4)] for _ in range(4)]   # disp 0.5 → Z = 2.0 m everywhere
    disp[0][0] = None                                     # an explicit hole
    disp[1][1] = -5.0                                     # sky/invalid (inv ≤ 0) → hole
    rgb = [[(10, 20, 30) for _ in range(4)] for _ in range(4)]
    pts = depth_to_points(disp, rgb, anchor, focal_px=F, principal_x=1.5, principal_y=1.5,
                          camera_height_m=cam_h, yaw_rad=0.0, stride=1, max_points=999)
    # holes are absent (4x4=16 minus the 2 holes = 14 points), never back-filled
    assert len(pts) == 14
    # a pixel at the principal point (x=1.5,y=1.5 → but integer pixels) — check geometry of (2,2)
    by_pixel = {}
    for (xx, yy, zz, r, g, b) in pts:
        by_pixel[round(xx, 3), round(yy, 3), round(zz, 3)] = (r, g, b)
    # pixel (x=2,y=2): Z=2, x_room=(2-1.5)*2/470, y_room=cam_h-(2-1.5)*2/470, z=2
    xr = (2 - 1.5) * 2.0 / F
    assert any(abs(zz - 2.0) < 1e-6 and abs(xx - round(xr, 3)) < 1e-3 for (xx, yy, zz) in
               [(p[0], p[1], p[2]) for p in pts])
    assert all(rgb_v == (10, 20, 30) for rgb_v in by_pixel.values())   # color carried through


def test_depth_to_points_refuses_on_invalid_anchor():
    bad = AnchorResult(0.0, 0.0, 0, False, math.inf, "too_few_lidar_pixels")
    pts = depth_to_points([[0.5] * 4] * 4, [[(0, 0, 0)] * 4] * 4, bad,
                          focal_px=F, principal_x=1.5, principal_y=1.5, camera_height_m=1.2)
    assert pts == []                                      # no fabricated geometry


def test_lidar_plane_row_is_near_centre_for_coaxial_mount():
    # camera 1.219 m, lidar 1.254 m (3.5 cm above) → anchor row a few px above centre
    row = lidar_plane_row(camera_height_m=1.219, mount_height_m=1.254, focal_px=F,
                          principal_y=240.0, ref_distance_m=2.5)
    assert 230 <= row <= 240 and row < 240                # above centre, but close


def test_no_forbidden_imports():
    try:
        from jarvis_eval.validation_pack import _scan_hrr_forbidden_imports, _HRR_MODULE_ROOTS
    except Exception:
        pytest.skip("validation_pack unavailable")
    assert any("lidar_depth" in r for r in _HRR_MODULE_ROOTS)
    result = _scan_hrr_forbidden_imports()
    is_clean = result[0] if isinstance(result, tuple) else (
        result.get("is_clean", result.get("clean", True)) if isinstance(result, dict) else bool(result))
    assert is_clean


def _row_profile(disp_fn, s_true, t_true, yaw_true=0.0, n_bins=360):
    """Encode inv_metric = s_true*disp + t_true, with the lidar ranges stored as if the
    true camera→lidar yaw is yaw_true (so the clean fit is only found at that yaw)."""
    row = [None] * 640
    prof = [None] * n_bins
    bw = TWO_PI / n_bins
    for x in range(20, 620, 13):
        disp = disp_fn(x)
        th = math.atan2(x - PX, F)
        if math.cos(th) <= 0.1:
            continue
        inv = s_true * disp + t_true
        if inv <= 0:
            continue
        row[x] = disp
        prof[int(((th - yaw_true) % TWO_PI) / bw) % n_bins] = (1.0 / inv) / math.cos(th)
    return row, prof


def test_anchor_rejects_non_physical_negative_scale():
    # disparity INVERSELY related to inverse-depth (s_true<0) → non-physical → refuse
    row, prof = _row_profile(lambda x: 0.4 + 0.002 * x, s_true=-1.0, t_true=3.0)
    a = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX, min_inliers=10)
    assert not a.valid and a.reason == "non_physical_scale" and a.scale < 0


def test_anchor_rejects_weak_correlation():
    # a real positive trend buried in big oscillation decorrelated from disp → |corr| < 0.4
    row = [None] * 640
    prof = [None] * 360
    bw = TWO_PI / 360
    for x in range(20, 620, 13):
        disp = 0.4 + 0.002 * x
        th = math.atan2(x - PX, F)
        if math.cos(th) <= 0.1:
            continue
        inv = 0.3 * disp + 1.5 + 0.8 * math.sin(x * 1.7)   # trend + decorrelated noise
        row[x] = disp
        prof[int((th % TWO_PI) / bw) % 360] = (1.0 / inv) / math.cos(th)
    a = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX,
                            min_inliers=10, min_corr=0.4)
    assert not a.valid and a.reason == "weak_correlation" and abs(a.corr) < 0.4


def test_anchor_best_yaw_finds_physical_fit_off_base():
    row, prof = _row_profile(lambda x: 0.4 + 0.002 * x, s_true=2.0, t_true=0.3,
                             yaw_true=math.radians(12))
    base = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX, min_inliers=10)
    best, yaw = anchor_best_yaw(row, prof, base_yaw_rad=0.0, focal_px=F, principal_x=PX,
                                search_deg=30, step_deg=2, min_inliers=10)
    assert best.valid and best.scale > 0 and best.corr > 0.9
    assert abs(math.degrees(yaw) - 12) <= 2          # search recovered the true yaw


def test_anchor_corr_in_dict():
    row, prof = _synthetic_row_and_profile(s_true=2.0, t_true=0.3)
    a = anchor_depth_affine(row, prof, yaw_rad=0.0, focal_px=F, principal_x=PX, min_inliers=10)
    assert a.to_dict()["corr"] >= 0.9                # clean synthetic ⇒ near-perfect correlation
