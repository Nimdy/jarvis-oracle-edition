"""Tier-3 dense-3D core: anchor a relative monocular depth map to METRIC truth using
the lidar's 360° ring (shadow / telemetry-only).

This is the flagship's actual IP. A monocular depth net (Depth-Anything) is fast but its
output is *relative* — scale- and shift-ambiguous disparity, not meters. The LiDAR, by
contrast, measures EXACT metric range, but only along one horizontal slice at its mount
height. So we use the lidar's slice as ground truth to solve the affine (scale, shift)
that turns the whole relative map into metric depth (the LiDARTouch / scale-shift
alignment). The depth net guesses the *shape*; the lidar supplies the *scale*. The result
is a dense colored cloud whose geometry is anchored to something actually measured —
NOT fabricated interpolation passed off as truth.

Honesty, like everywhere in JARVIS: this module refuses to invent a scale it can't justify
(too few inliers / no disparity spread → ``valid=False``), and holes (sky / invalid depth)
stay holes — never back-filled. Pure stdlib, no live-brain deps; registered in the HRR
forbidden-import scan (``writes_beliefs=false``, ``authority=spatial_telemetry_only``).

Conventions match ``lidar_fusion``: room X=right, Z=forward, bearing 0 = +Z (camera optical
axis); ``room_bearing = lidar_bearing + yaw``. Depth-Anything output is treated as
*disparity* (inverse depth), so metric ``Z = 1 / (scale·disp + shift)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class AnchorResult:
    """The affine (scale, shift) mapping disparity → inverse-metric-depth, plus the
    evidence that earned it. ``valid=False`` ⇒ the solve was refused (don't use it)."""
    scale: float
    shift: float
    inlier_count: int
    valid: bool
    rms: float                      # rms residual in inverse-depth units (lower = tighter)
    reason: str                     # why it's invalid, when it is

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": round(self.scale, 6), "shift": round(self.shift, 6),
            "inlier_count": self.inlier_count, "valid": self.valid,
            "rms": round(self.rms, 5), "reason": self.reason,
            "authority": "spatial_telemetry_only", "writes_beliefs": False,
        }


def lidar_plane_row(camera_height_m: float, mount_height_m: float, focal_px: float,
                    principal_y: float, ref_distance_m: float = 2.5) -> int:
    """The image row where the lidar's horizontal scan plane lands (the anchor row).

    The plane sits ``mount_height_m − camera_height_m`` above the optical centre, so for a
    target at ``ref_distance_m`` it projects ``focal·Δh/d`` pixels above the principal row
    (image y grows downward). For a near-co-axial mount this is ~the centre row.
    """
    dy = focal_px * (mount_height_m - camera_height_m) / max(0.1, ref_distance_m)
    return int(round(principal_y - dy))


def _ols(xs: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Ordinary least-squares y ≈ s·x + t. None if x has no spread (unsolvable)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vxx = sum((x - mx) ** 2 for x in xs)
    if vxx <= 1e-12:
        return None
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vxx
    return s, my - s * mx


def anchor_depth_affine(disparity_row: Sequence[Optional[float]],
                        lidar_profile: Sequence[Optional[float]], *,
                        yaw_rad: float, focal_px: float, principal_x: float,
                        min_inliers: int = 20, mad_k: float = 3.0,
                        min_disp_spread: float = 1e-3) -> AnchorResult:
    """Solve the affine (scale, shift) so ``scale·disp + shift ≈ 1/Z_metric`` along the
    lidar scan row — the row's pixels paired with the lidar's measured range at the same
    bearing. Robust (one MAD-filtered re-solve), and it REFUSES rather than fabricate when
    the evidence is thin (too few lidar-backed pixels, or no disparity spread).
    """
    n = len(lidar_profile)
    if n == 0:
        return AnchorResult(0.0, 0.0, 0, False, math.inf, "no_lidar_profile")
    bin_w = TWO_PI / n
    disp: list[float] = []
    inv_metric: list[float] = []     # target: 1 / Z_metric (inverse Z-depth)
    for x, d in enumerate(disparity_row):
        if d is None or not math.isfinite(d) or d <= 0.0:
            continue                                       # invalid / sky disparity
        theta = math.atan2(x - principal_x, focal_px)      # room bearing of this column
        cos_t = math.cos(theta)
        if cos_t <= 0.1:                                   # too oblique (near ±90°) — skip
            continue
        rng = lidar_profile[int(((theta - yaw_rad) % TWO_PI) / bin_w) % n]
        if rng is None or rng <= 0.0:
            continue                                       # lidar saw nothing this bearing
        z = rng * cos_t                                    # radial range → pinhole Z-depth
        disp.append(float(d))
        inv_metric.append(1.0 / z)

    if len(disp) < min_inliers:
        return AnchorResult(0.0, 0.0, len(disp), False, math.inf, "too_few_lidar_pixels")
    if (max(disp) - min(disp)) < min_disp_spread:
        return AnchorResult(0.0, 0.0, len(disp), False, math.inf, "degenerate_no_disp_spread")

    fit = _ols(disp, inv_metric)
    if fit is None:
        return AnchorResult(0.0, 0.0, len(disp), False, math.inf, "unsolvable")
    s, t = fit
    # one robust re-solve: drop pixels whose residual exceeds mad_k · MAD
    resid = [s * d + t - y for d, y in zip(disp, inv_metric)]
    med = sorted(resid)[len(resid) // 2]
    mad = sorted(abs(r - med) for r in resid)[len(resid) // 2] or 1e-9
    keep = [(d, y) for d, y, r in zip(disp, inv_metric, resid) if abs(r - med) <= mad_k * mad]
    if len(keep) >= min_inliers:
        fit2 = _ols([d for d, _ in keep], [y for _, y in keep])
        if fit2 is not None:
            s, t = fit2
            disp = [d for d, _ in keep]
            inv_metric = [y for _, y in keep]

    rms = math.sqrt(sum((s * d + t - y) ** 2 for d, y in zip(disp, inv_metric)) / len(disp))
    return AnchorResult(s, t, len(disp), True, rms, "ok")


def depth_to_points(disparity_map: Sequence[Sequence[Optional[float]]],
                    rgb_frame: Sequence[Sequence[Sequence[int]]], anchor: AnchorResult, *,
                    focal_px: float, principal_x: float, principal_y: float,
                    camera_height_m: float, yaw_rad: float = 0.0,
                    stride: int = 4, max_points: int = 12000,
                    max_depth_m: float = 12.0, min_inv: float = 1e-3
                    ) -> list[tuple[float, float, float, int, int, int]]:
    """Lift an anchored disparity map to dense colored points in the LIDAR frame (so they
    overlay the lidar walls). Each kept pixel → (x, y, z, r, g, b). Sky / invalid depth
    becomes a HOLE (skipped) — never interpolated. Decimated by ``stride``, capped at
    ``max_points``. Returns [] if the anchor is invalid (no fabricated geometry).
    """
    if not anchor.valid:
        return []
    s, t = anchor.scale, anchor.shift
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)          # room → lidar rotation (by −yaw)
    h = len(disparity_map)
    out: list[tuple[float, float, float, int, int, int]] = []
    for y in range(0, h, stride):
        row = disparity_map[y]
        crow = rgb_frame[y] if y < len(rgb_frame) else None
        for x in range(0, len(row), stride):
            d = row[x]
            if d is None or not math.isfinite(d) or d <= 0.0:
                continue
            inv = s * d + t
            if inv <= min_inv:                             # ≤0 ⇒ sky / behind camera → HOLE
                continue
            z = 1.0 / inv
            if z <= 0.0 or z > max_depth_m:
                continue
            x_room = (x - principal_x) * z / focal_px       # right
            y_room = camera_height_m - (y - principal_y) * z / focal_px   # up (image y down)
            # rotate the horizontal (x_room, z) by −yaw into the lidar frame
            x_l = x_room * cy - z * sy
            z_l = x_room * sy + z * cy
            r, g, b = (0, 0, 0)
            if crow is not None and x < len(crow):
                px = crow[x]
                r, g, b = int(px[0]), int(px[1]), int(px[2])
            out.append((round(x_l, 3), round(y_room, 3), round(z_l, 3), r, g, b))
            if len(out) >= max_points:
                return out
    return out
