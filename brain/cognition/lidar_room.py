"""Lidar room model — a SOLID, standalone 2D reconstruction from a FIXED RPLIDAR S2.

The S2 never moves, so it produces exactly one r(theta) function of the room. That
makes a polar occupancy histogram (fixed absolute angular bins) the natural sensor
model: temporal accumulation is legal with no SLAM/ICP/pose-graph. We denoise each
bin with a robust temporal MEDIAN + MAD (the 32 mm jitter averages down; the up-to-
1.4 m edge spikes become statistically invisible), classify each bearing ternary
(occupied / free / unknown, with occlusion-shadow flagging — honest openings), and
extract walls (split-and-merge with a MAD-driven inlier band), openings, dimensions,
free space, and a coarse nearest-per-sector safety vector.

PURE + STANDALONE: stdlib only, frozen dataclasses, no live-brain deps (this module is
registered in jarvis_eval.validation_pack._HRR_MODULE_ROOTS, which forbids policy /
belief / memory / autonomy / identity imports). It is built + tested in isolation
against captured scan fixtures; integration (anchors → scene graph, /mind viz) is a
separate, additive seam that never disrupts live cognition.
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Optional, Sequence


def _env_float(name: str, default: float) -> float:
    """Config that PERSISTS via env (not a hardcoded magic number) — override at the
    launch env without a code change; the literal here is the documented default."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)

TWO_PI = 2.0 * math.pi

REASON_INSUFFICIENT_COVERAGE = "insufficient_coverage"


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class LidarRoomConfig:
    n_bins: int = 360                       # angular resolution (up to 720 @ 0.5°)
    ring_len_K: int = 14                    # per-bin temporal window (= fixture revs)
    min_samples_per_bin: int = 3            # below this, a bin is "unknown"
    wall_inlier_floor_m: float = 0.05       # min RANSAC inlier band
    wall_mad_mult: float = 1.5              # inlier band = max(floor, mult * local MAD)
    min_wall_inliers: int = 8               # a segment shorter than this is dropped
    door_width_band_m: tuple[float, float] = (0.6, 1.2)
    nearest_sectors: int = 12
    range_max_m: float = 12.0               # S2 sensor ceiling / fallback + shadow ratio
    # --- indoor range gating (the S2 truth-layer noise fix; env-overridable) ---
    min_range_m: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_MIN_RANGE_M", 0.12))
    max_range_m: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_MAX_RANGE_M", 8.0))
    min_coverage_fraction: float = 0.25     # below → empty-by-design RoomModel
    discontinuity_m: float = 0.30           # range jump that breaks a wall run
    occlusion_shadow_ratio: float = 0.5     # gap behind a bin < ratio*neighbour → shadow


# ------------------------------------------------------------------------ datamodels
@dataclass(frozen=True)
class LidarScan:
    """One raw revolution. range_m == 0 / None means a dropout for that sample."""
    timestamp: float
    points: tuple[tuple[float, float], ...]   # (bearing_rad, range_m)
    range_max_m: float = 12.0
    quality: str = "good"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LidarScan":
        pts = tuple((float(b), float(r)) for b, r in d.get("points", ()))
        return LidarScan(timestamp=float(d.get("timestamp", 0.0)), points=pts,
                         range_max_m=float(d.get("range_max_m", 12.0)),
                         quality=str(d.get("quality", "good")))


@dataclass(frozen=True)
class PolarBin:
    index: int
    bearing_center_rad: float
    r_stable_m: Optional[float]
    mad_m: float
    sample_count: int
    valid_fraction: float
    occupancy: str                            # "occupied" | "free" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bearing_center_rad": round(self.bearing_center_rad, 5),
            "r_stable_m": None if self.r_stable_m is None else round(self.r_stable_m, 3),
            "mad_m": round(self.mad_m, 4),
            "sample_count": self.sample_count,
            "valid_fraction": round(self.valid_fraction, 3),
            "occupancy": self.occupancy,
        }


@dataclass(frozen=True)
class WallSegment:
    start_m: tuple[float, float]
    end_m: tuple[float, float]
    normal: tuple[float, float]
    length_m: float
    inlier_count: int
    residual_rms_m: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_m": [round(self.start_m[0], 3), round(self.start_m[1], 3)],
            "end_m": [round(self.end_m[0], 3), round(self.end_m[1], 3)],
            "normal": [round(self.normal[0], 3), round(self.normal[1], 3)],
            "length_m": round(self.length_m, 3),
            "inlier_count": self.inlier_count,
            "residual_rms_m": round(self.residual_rms_m, 4),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class Opening:
    center_bearing_rad: float
    width_m: float
    angular_width_rad: float
    kind: str                                 # door | wide_gap | window | passage
    flanked_by_walls: bool
    is_occlusion_shadow: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_bearing_rad": round(self.center_bearing_rad, 5),
            "width_m": round(self.width_m, 3),
            "angular_width_rad": round(self.angular_width_rad, 4),
            "kind": self.kind,
            "flanked_by_walls": self.flanked_by_walls,
            "is_occlusion_shadow": self.is_occlusion_shadow,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class RoomModel:
    timestamp: float
    profile: tuple[Optional[float], ...]      # r_stable(theta), length n_bins
    points_m: tuple[tuple[float, float], ...]
    walls: tuple[WallSegment, ...]
    openings: tuple[Opening, ...]
    nearest_per_sector_m: tuple[float, ...]
    dimensions_m: tuple[float, float]
    free_space_area_m2: float
    inscribed_radius_m: float
    bin_quality: tuple[tuple[int, float, float], ...]   # (sample_count, valid_fraction, mad_m)
    coverage_fraction: float
    scan_count: int
    n_bins: int
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lidar_room",
            "timestamp": round(self.timestamp, 3),
            "frame": "lidar_sensor",          # planar sensor frame; reconciled by the extrinsic later
            "reason": self.reason,
            "n_bins": self.n_bins,
            "scan_count": self.scan_count,
            "coverage_fraction": round(self.coverage_fraction, 3),
            "profile": [None if v is None else round(v, 3) for v in self.profile],
            "points_m": [[round(x, 3), round(z, 3)] for x, z in self.points_m],
            "walls": [w.to_dict() for w in self.walls],
            "openings": [o.to_dict() for o in self.openings],
            "nearest_per_sector_m": [round(v, 3) for v in self.nearest_per_sector_m],
            "dimensions_m": [round(self.dimensions_m[0], 3), round(self.dimensions_m[1], 3)],
            "free_space_area_m2": round(self.free_space_area_m2, 3),
            "inscribed_radius_m": round(self.inscribed_radius_m, 3),
        }


# ----------------------------------------------------------------------- pure helpers
def polar_to_cartesian(bearing_rad: float, range_m: float) -> tuple[float, float]:
    """Room X-Z convention: X = right (sin), Z = forward (cos)."""
    return (range_m * math.sin(bearing_rad), range_m * math.cos(bearing_rad))


def _mad(values: Sequence[float], med: float) -> float:
    if not values:
        return 0.0
    return median([abs(v - med) for v in values])


def denoise_bins(rings: Sequence[Sequence[float]], scan_count: int,
                 config: LidarRoomConfig) -> tuple[PolarBin, ...]:
    """Per-bin temporal median (r_stable) + MAD + ternary occupancy.

    A bin's ring holds the per-revolution NEAREST valid range (or nothing for a
    dropout that revolution). Median ignores up to ~half the ring being outliers, so
    the 1.4 m edge spikes can't drag a stable wall metres off. Occlusion-shadow
    flagging is applied in a second pass once neighbours are known.
    """
    n = config.n_bins
    bin_w = TWO_PI / n
    prelim: list[PolarBin] = []
    for i in range(n):
        ring = [v for v in rings[i] if v is not None and v > 0.0]
        sc = len(ring)
        vf = (sc / scan_count) if scan_count > 0 else 0.0
        if sc >= config.min_samples_per_bin:
            r = float(median(ring))
            occ = "occupied"
        elif scan_count > 0 and sc == 0:
            r = None
            occ = "free"          # consistently no return in range → genuinely open
        else:
            r = None
            occ = "unknown"
        prelim.append(PolarBin(
            index=i, bearing_center_rad=(i + 0.5) * bin_w,
            r_stable_m=r, mad_m=_mad(ring, r) if r is not None else 0.0,
            sample_count=sc, valid_fraction=vf, occupancy=occ,
        ))

    # occlusion-shadow pass: a "free" bin sitting angularly behind a much-nearer
    # neighbour wall is NOT honestly open — demote it to "unknown" (never assert open).
    out: list[PolarBin] = []
    for i, b in enumerate(prelim):
        occ = b.occupancy
        if occ == "free":
            ln = prelim[(i - 1) % n].r_stable_m
            rn = prelim[(i + 1) % n].r_stable_m
            near = min([v for v in (ln, rn) if v is not None], default=None)
            if near is not None and near < config.range_max_m * config.occlusion_shadow_ratio:
                occ = "unknown"
        out.append(PolarBin(index=b.index, bearing_center_rad=b.bearing_center_rad,
                            r_stable_m=b.r_stable_m, mad_m=b.mad_m,
                            sample_count=b.sample_count, valid_fraction=b.valid_fraction,
                            occupancy=occ))
    return tuple(out)


def _perp_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, az = a
    bx, bz = b
    px, pz = p
    dx, dz = bx - ax, bz - az
    L = math.hypot(dx, dz)
    if L < 1e-9:
        return math.hypot(px - ax, pz - az)
    return abs((px - ax) * dz - (pz - az) * dx) / L


def _fit_line(pts: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], float]:
    """PCA line fit → (unit direction, residual_rms) about the centroid."""
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cz = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - cx) ** 2 for p in pts)
    szz = sum((p[1] - cz) ** 2 for p in pts)
    sxz = sum((p[0] - cx) * (p[1] - cz) for p in pts)
    # principal direction via the 2x2 covariance eigenvector
    theta = 0.5 * math.atan2(2 * sxz, sxx - szz)
    d = (math.cos(theta), math.sin(theta))
    res = math.sqrt(sum(_perp_dist(p, (cx, cz), (cx + d[0], cz + d[1])) ** 2 for p in pts) / n)
    return d, res


def _split_and_merge(pts: list[tuple[float, float]], mads: list[float],
                     config: LidarRoomConfig) -> list[list[int]]:
    """Recursive split-and-merge over an ORDERED point run → index groups (segments)."""
    groups: list[list[int]] = []

    def split(lo: int, hi: int) -> None:
        if hi - lo + 1 < 2:
            return
        a, b = pts[lo], pts[hi]
        worst, wi = -1.0, -1
        for k in range(lo + 1, hi):
            dd = _perp_dist(pts[k], a, b)
            if dd > worst:
                worst, wi = dd, k
        local_mad = median(mads[lo:hi + 1]) if hi > lo else 0.0
        band = max(config.wall_inlier_floor_m, config.wall_mad_mult * local_mad)
        if worst > band and wi > lo:
            split(lo, wi)
            split(wi, hi)
        else:
            groups.append(list(range(lo, hi + 1)))

    split(0, len(pts) - 1)
    return groups


def fit_walls(bins: Sequence[PolarBin], config: LidarRoomConfig) -> tuple[WallSegment, ...]:
    """Extract wall segments from the stable profile. Breaks runs at dropouts and at
    range discontinuities so a doorway never gets bridged by a phantom wall."""
    n = len(bins)
    # contiguous runs of occupied bins with no big range jump
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in range(n):
        b = bins[i]
        if b.r_stable_m is None or b.occupancy != "occupied":
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
            continue
        if cur:
            prev = bins[cur[-1]]
            if prev.r_stable_m is not None and abs(b.r_stable_m - prev.r_stable_m) > config.discontinuity_m:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
        cur.append(i)
    if len(cur) >= 2:
        runs.append(cur)

    walls: list[WallSegment] = []
    for run in runs:
        pts = [polar_to_cartesian(bins[i].bearing_center_rad, bins[i].r_stable_m) for i in run]
        mads = [bins[i].mad_m for i in run]
        for grp in _split_and_merge(pts, mads, config):
            if len(grp) < config.min_wall_inliers:
                continue
            seg_pts = [pts[k] for k in grp]
            d, res = _fit_line(seg_pts)
            start, end = seg_pts[0], seg_pts[-1]
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length < 1e-3:
                continue
            normal = (-d[1], d[0])
            conf = max(0.0, min(1.0, 1.0 - res / max(config.wall_inlier_floor_m, 1e-3) * 0.25))
            walls.append(WallSegment(
                start_m=start, end_m=end, normal=normal, length_m=length,
                inlier_count=len(grp), residual_rms_m=res, confidence=conf,
            ))
    return tuple(walls)


def detect_openings(bins: Sequence[PolarBin], config: LidarRoomConfig) -> tuple[Opening, ...]:
    """Openings = contiguous unknown/free runs flanked by occupied bins, that are NOT
    occlusion shadows. A run dominated by occlusion-shadow bins is not asserted open."""
    n = len(bins)
    bin_w = TWO_PI / n
    openings: list[Opening] = []
    i = 0
    visited = [False] * n
    for start in range(n):
        if visited[start] or bins[start].occupancy == "occupied":
            continue
        # walk the contiguous non-occupied run (wrap-safe, bounded)
        run = []
        k = start
        steps = 0
        while bins[k % n].occupancy != "occupied" and steps < n:
            run.append(k % n)
            visited[k % n] = True
            k += 1
            steps += 1
        if not run:
            continue
        left = bins[(run[0] - 1) % n]
        right = bins[(run[-1] + 1) % n]
        flanked = left.occupancy == "occupied" and right.occupancy == "occupied"
        if not flanked or left.r_stable_m is None or right.r_stable_m is None:
            continue
        # a run that is mostly genuine "free" (not shadow/dropout) is a real opening
        free_n = sum(1 for idx in run if bins[idx].occupancy == "free")
        is_shadow = free_n < max(1, len(run) // 2)
        ang = len(run) * bin_w
        depth = 0.5 * (left.r_stable_m + right.r_stable_m)
        width = 2.0 * depth * math.sin(ang / 2.0)
        center = ((run[0] + run[-1]) / 2.0 + 0.5) * bin_w
        lo, hi = config.door_width_band_m
        kind = "door" if lo <= width <= hi else ("passage" if width > hi else "window")
        conf = 0.0 if is_shadow else max(0.0, min(1.0, 1.0 - abs(left.r_stable_m - right.r_stable_m)))
        if not is_shadow:
            openings.append(Opening(
                center_bearing_rad=center % TWO_PI, width_m=width, angular_width_rad=ang,
                kind=kind, flanked_by_walls=flanked, is_occlusion_shadow=False, confidence=conf,
            ))
    return tuple(openings)


def nearest_per_sector(bins: Sequence[PolarBin], n_sectors: int) -> tuple[float, ...]:
    n = len(bins)
    per = n / n_sectors
    out = []
    for s in range(n_sectors):
        vals = [bins[i].r_stable_m for i in range(n)
                if int(i // per) == s and bins[i].r_stable_m is not None]
        out.append(min(vals) if vals else 0.0)
    return tuple(out)


def free_space(points_m: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Star-shaped polygon enclosed by the profile → (area_m2, inscribed_radius_m)."""
    if len(points_m) < 3:
        return (0.0, 0.0)
    area = 0.0
    for i in range(len(points_m)):
        x1, z1 = points_m[i]
        x2, z2 = points_m[(i + 1) % len(points_m)]
        area += x1 * z2 - x2 * z1
    area = abs(area) / 2.0
    inscribed = min(math.hypot(x, z) for x, z in points_m)
    return (area, inscribed)


# ----------------------------------------------------------------------- accumulator
class LidarRoomModel:
    """Stateful accumulator: ingest scans into fixed bins, derive a frozen RoomModel."""

    def __init__(self, config: LidarRoomConfig = LidarRoomConfig()) -> None:
        self._config = config
        self._rings: list[deque] = [deque(maxlen=config.ring_len_K) for _ in range(config.n_bins)]
        self._scan_count = 0
        self._fstats: dict[str, Any] = self._zero_fstats()

    def _zero_fstats(self) -> dict[str, Any]:
        return {
            "raw_points": 0, "dropped_zero": 0, "dropped_min_range": 0,
            "dropped_max_range": 0, "points_after_filter": 0,
            "observed_min_m": None, "observed_max_m": None,
            "effective_max_m": self._config.max_range_m,
            "cum_raw_points": 0, "cum_dropped_zero": 0, "cum_dropped_min_range": 0,
            "cum_dropped_max_range": 0, "cum_points_after_filter": 0,
        }

    def ingest(self, scan: LidarScan) -> None:
        try:
            cfg = self._config
            n = cfg.n_bins
            bin_w = TWO_PI / n
            nearest: list[Optional[float]] = [None] * n
            # hard indoor ceiling: a far return (open door/window/specular reflection) is
            # ghost geometry that inflates the room — cap it. Also drop too-close housing noise.
            eff_max = min(scan.range_max_m or cfg.range_max_m, cfg.max_range_m)
            rmin = cfg.min_range_m
            raw = d_zero = d_min = d_max = kept = 0
            o_min: Optional[float] = None
            o_max: Optional[float] = None
            for bearing, rng in scan.points:
                raw += 1
                if rng is None or rng <= 0.0:
                    d_zero += 1
                    continue
                if o_min is None or rng < o_min:
                    o_min = rng
                if o_max is None or rng > o_max:
                    o_max = rng
                if rng < rmin:                       # too-close housing reflection
                    d_min += 1
                    continue
                if rng >= eff_max:                   # ghost beyond the indoor cap
                    d_max += 1
                    continue
                kept += 1
                b = int((bearing % TWO_PI) / bin_w) % n
                if nearest[b] is None or rng < nearest[b]:
                    nearest[b] = rng
            for i in range(n):
                # carry the per-rev nearest (or a dropout None) — NEVER back-fill 0/range_max
                self._rings[i].append(nearest[i])
            self._scan_count += 1
            fs = self._fstats
            fs.update(raw_points=raw, dropped_zero=d_zero, dropped_min_range=d_min,
                      dropped_max_range=d_max, points_after_filter=kept,
                      observed_min_m=o_min, observed_max_m=o_max, effective_max_m=eff_max)
            fs["cum_raw_points"] += raw
            fs["cum_dropped_zero"] += d_zero
            fs["cum_dropped_min_range"] += d_min
            fs["cum_dropped_max_range"] += d_max
            fs["cum_points_after_filter"] += kept
        except Exception:
            pass  # a bad scan must never corrupt accumulated state

    def filter_stats(self) -> dict[str, Any]:
        """Drop-telemetry to PROVE the gating works (raw→after-filter, ghost rejection,
        observed-vs-effective range). Honest counters, never a belief."""
        fs = dict(self._fstats)
        fs["min_range_m"] = self._config.min_range_m
        fs["max_range_m"] = self._config.max_range_m
        fs["scan_count"] = self._scan_count
        return fs

    def reset(self) -> None:
        for r in self._rings:
            r.clear()
        self._scan_count = 0
        self._fstats = self._zero_fstats()

    @property
    def coverage_fraction(self) -> float:
        if self._scan_count == 0:
            return 0.0
        occupied = sum(1 for r in self._rings if any(v is not None and v > 0 for v in r))
        return occupied / self._config.n_bins

    def room_model(self, *, timestamp: Optional[float] = None) -> RoomModel:
        cfg = self._config
        ts = timestamp if timestamp is not None else 0.0
        rings = [list(r) for r in self._rings]
        bins = denoise_bins(rings, self._scan_count, cfg)
        coverage = self.coverage_fraction
        n_bins = cfg.n_bins
        if self._scan_count == 0 or coverage < cfg.min_coverage_fraction:
            return RoomModel(
                timestamp=ts, profile=tuple(b.r_stable_m for b in bins),
                points_m=(), walls=(), openings=(), nearest_per_sector_m=(),
                dimensions_m=(0.0, 0.0), free_space_area_m2=0.0, inscribed_radius_m=0.0,
                bin_quality=tuple((b.sample_count, b.valid_fraction, b.mad_m) for b in bins),
                coverage_fraction=coverage, scan_count=self._scan_count, n_bins=n_bins,
                reason=REASON_INSUFFICIENT_COVERAGE,
            )
        points = tuple(polar_to_cartesian(b.bearing_center_rad, b.r_stable_m)
                       for b in bins if b.r_stable_m is not None)
        walls = fit_walls(bins, cfg)
        openings = detect_openings(bins, cfg)
        sectors = nearest_per_sector(bins, cfg.nearest_sectors)
        area, inscribed = free_space(points)
        xs = [p[0] for p in points] or [0.0]
        zs = [p[1] for p in points] or [0.0]
        dims = (max(xs) - min(xs), max(zs) - min(zs))
        return RoomModel(
            timestamp=ts, profile=tuple(b.r_stable_m for b in bins), points_m=points,
            walls=walls, openings=openings, nearest_per_sector_m=sectors,
            dimensions_m=dims, free_space_area_m2=area, inscribed_radius_m=inscribed,
            bin_quality=tuple((b.sample_count, b.valid_fraction, b.mad_m) for b in bins),
            coverage_fraction=coverage, scan_count=self._scan_count, n_bins=n_bins, reason=None,
        )

    def features(self) -> dict[str, Any]:
        return self.room_model().to_dict()


def build_room_model(scans: Sequence[LidarScan], config: LidarRoomConfig = LidarRoomConfig(),
                     *, timestamp: Optional[float] = None) -> RoomModel:
    """Pure one-shot: feed scans through a fresh accumulator → frozen RoomModel."""
    acc = LidarRoomModel(config)
    for s in scans:
        acc.ingest(s)
    ts = timestamp if timestamp is not None else (scans[-1].timestamp if scans else 0.0)
    return acc.room_model(timestamp=ts)
