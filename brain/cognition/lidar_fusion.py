"""LiDAR ↔ camera object fusion (Tier-2, shadow / telemetry-only).

The lidar and camera are co-located, so a camera detection at image-bearing θ_cam and
a lidar return at the matching bearing are the SAME physical object. The split is
honest and load-bearing:

  • camera  → the LABEL ("chair") — a *hypothesis* about WHAT it is (low authority)
  • lidar   → the exact metric RANGE / position — geometry TRUTH the monocular camera
              can't scale on its own

So a fused object is "camera's guess at what + lidar's measurement of where". The label
is NEVER asserted as fact here: a ``FusedObject`` is telemetry, never a belief, and the
caller must keep it shadow until a verification step (camera re-confirm + grounding)
earns it. Pure stdlib, no live-brain deps (registered in the HRR forbidden-import scan).

Frame: room X = right (range·sin bearing), Z = forward (range·cos), bearing 0 = +Z
(the camera's optical axis). The extrinsic yaw maps between the two:
``room_bearing = lidar_bearing + yaw``  ⇒  to look a camera (room) bearing up in the
lidar profile, use ``lidar_bearing = room_bearing − yaw``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

TWO_PI = 2.0 * math.pi


def camera_bearing(bbox: Sequence[float], principal_x: float, focal_px: float) -> float:
    """Room/camera bearing (rad) of a detection's horizontal centre off the optical axis.

    bbox = (x1, y1, x2, y2). +bearing = object to the camera's right (room +X);
    0 = dead-ahead (room +Z). Pinhole: atan2(centre_x − principal_x, focal).
    """
    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    return math.atan2(cx - principal_x, focal_px)


def lidar_range_at(profile: Sequence[Optional[float]], bearing_rad: float,
                   bin_w: float, window_bins: int = 2) -> Optional[float]:
    """Nearest stable lidar range at ``bearing_rad`` (± a few bins for slop), or None.

    Returns the NEAREST return in the window (an object is the closest thing along that
    ray); None if the lidar saw nothing there — an honest "no measurement", not 0.
    """
    n = len(profile)
    if n == 0:
        return None
    b0 = int((bearing_rad % TWO_PI) / bin_w) % n
    cand = [profile[(b0 + off) % n] for off in range(-window_bins, window_bins + 1)]
    cand = [v for v in cand if v is not None and v > 0.0]
    return min(cand) if cand else None


@dataclass(frozen=True)
class FusedObject:
    """A camera label + a lidar measurement of where it is. TELEMETRY, never a belief."""
    label: str
    label_confidence: float           # CAMERA — a hypothesis, not a fact
    bearing_rad: float                # ROOM/camera-frame bearing
    lidar_bearing_rad: float          # LIDAR-frame bearing (= room − yaw; where the profile was sampled)
    range_m: Optional[float]          # LIDAR metric range (None = lidar saw nothing there)
    position_room_m: Optional[tuple[float, float, float]]    # (x, y=mount height, z), room frame
    position_lidar_m: Optional[tuple[float, float, float]]   # same, LIDAR frame — for the radar overlay
    source_entity_id: str
    has_lidar_range: bool             # False ⇒ camera-only (no metric confirmation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_confidence": round(self.label_confidence, 3),
            "label_provenance": "camera_hypothesis",     # honest: a guess until verified
            "bearing_rad": round(self.bearing_rad, 5),
            "lidar_bearing_rad": round(self.lidar_bearing_rad, 5),
            "range_m": None if self.range_m is None else round(self.range_m, 3),
            "range_provenance": "lidar_metric" if self.has_lidar_range else "none",
            "position_room_m": (None if self.position_room_m is None
                                else [round(v, 3) for v in self.position_room_m]),
            "position_lidar_m": (None if self.position_lidar_m is None
                                 else [round(v, 3) for v in self.position_lidar_m]),
            "source_entity_id": self.source_entity_id,
            "has_lidar_range": self.has_lidar_range,
            "authority": "spatial_telemetry_only",
            "writes_beliefs": False,
        }


def fuse(camera_entities: Sequence[dict], lidar_profile: Sequence[Optional[float]], *,
         yaw_rad: float = 0.0, focal_px: float = 800.0, principal_x: float = 960.0,
         mount_height_m: float = 0.0, window_bins: int = 2) -> list[FusedObject]:
    """Fuse camera detections (need a ``bbox``) with the lidar range profile.

    Each entity: {entity_id, label, confidence, bbox=(x1,y1,x2,y2)}. The object's room
    POSITION uses the camera bearing (room frame) + the lidar RANGE; the profile is
    sampled at ``bearing − yaw`` (camera→lidar). Pure: returns a fresh list, mutates nothing.
    """
    n = len(lidar_profile)
    if n == 0:
        return []
    bin_w = TWO_PI / n
    out: list[FusedObject] = []
    for e in camera_entities:
        bbox = e.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        b_room = camera_bearing(bbox, principal_x, focal_px)          # camera = room frame
        b_lidar = (b_room - yaw_rad) % TWO_PI                          # where to look in the profile
        rng = lidar_range_at(lidar_profile, b_lidar, bin_w, window_bins)
        pos_room = pos_lidar = None
        if rng is not None:
            pos_room = (round(rng * math.sin(b_room), 3), round(mount_height_m, 3),
                        round(rng * math.cos(b_room), 3))             # room frame (camera bearing)
            pos_lidar = (round(rng * math.sin(b_lidar), 3), round(mount_height_m, 3),
                         round(rng * math.cos(b_lidar), 3))           # lidar frame (sits on the return)
        out.append(FusedObject(
            label=str(e.get("label", "?")),
            label_confidence=float(e.get("confidence", 0.0) or 0.0),
            bearing_rad=b_room, lidar_bearing_rad=b_lidar, range_m=rng,
            position_room_m=pos_room, position_lidar_m=pos_lidar,
            source_entity_id=str(e.get("entity_id", "")),
            has_lidar_range=rng is not None,
        ))
    return out


class YawEstimator:
    """Self-calibrate the lidar↔camera yaw from co-observed moving objects.

    A person walking sweeps both sensors; the yaw is the constant offset that best aligns
    "where the camera sees them" (room bearing) with "where the lidar sees the matching
    close/moving return" (lidar bearing). ``yaw = room_bearing − lidar_bearing``. Returns
    None until there is enough ANGULAR SPREAD in the observations — a single point can't
    solve an offset, so it never declares a yaw it hasn't earned.
    """

    def __init__(self, max_pairs: int = 256, min_pairs: int = 8,
                 min_spread_rad: float = math.radians(12)) -> None:
        self._pairs: deque[tuple[float, float]] = deque(maxlen=max_pairs)
        self._min_pairs = min_pairs
        self._min_spread = min_spread_rad

    def observe(self, room_bearing_rad: float, lidar_bearing_rad: float) -> None:
        self._pairs.append((room_bearing_rad, lidar_bearing_rad))

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % TWO_PI - math.pi

    def estimate(self) -> Optional[float]:
        if len(self._pairs) < self._min_pairs:
            return None
        room = [r for r, _ in self._pairs]
        if (max(room) - min(room)) < self._min_spread:
            return None                                  # not a solve, just a point — refuse
        diffs = sorted(self._wrap(r - l) for r, l in self._pairs)
        return diffs[len(diffs) // 2]                    # median offset (robust to mismatches)

    def confidence(self) -> dict[str, Any]:
        return {"pairs": len(self._pairs), "ready": self.estimate() is not None}

    def diagnostics(self) -> dict[str, Any]:
        """Visibility into WHY estimate() is/ isn't ready — the camera-bearing spread and
        the raw median offset (even when spread is still below the gate)."""
        if len(self._pairs) < 2:
            return {"spread_deg": 0.0, "raw_median_deg": None, "need_spread_deg": round(math.degrees(self._min_spread), 1)}
        room = [r for r, _ in self._pairs]
        diffs = sorted(self._wrap(r - l) for r, l in self._pairs)
        return {
            "spread_deg": round(math.degrees(max(room) - min(room)), 1),
            "raw_median_deg": round(math.degrees(diffs[len(diffs) // 2]), 1),
            "need_spread_deg": round(math.degrees(self._min_spread), 1),
        }


class YawCalibrator:
    """Self-calibrate yaw from a MOVING object (a walking person), without needing a
    pre-known correspondence. Frame-to-frame, the lidar bin whose range CHANGED the most
    is the moving thing; pair its bearing with the camera's person bearing and feed the
    estimator. ADVISORY ONLY — it *suggests* a yaw; it never auto-overwrites the config.
    """

    def __init__(self, min_change_m: float = 0.25) -> None:
        self._prev: Optional[list[Optional[float]]] = None
        self._est = YawEstimator()
        self._min_change = min_change_m

    def update(self, person_camera_bearing_rad: float,
               profile: Sequence[Optional[float]], bin_w: Optional[float] = None) -> None:
        n = len(profile)
        if n == 0:
            return
        bw = bin_w or (TWO_PI / n)
        if self._prev is not None and len(self._prev) == n:
            # the bin that got CLOSER (a − b > 0) is where the person now IS (the bin they
            # LEFT got farther — same magnitude but the old position). Track the arrival.
            best_i, best_d = -1, self._min_change
            for i in range(n):
                a, b = self._prev[i], profile[i]
                if a is not None and b is not None and (a - b) > best_d:
                    best_d, best_i = a - b, i
            if best_i >= 0:                          # a real approach → a usable pair
                self._est.observe(person_camera_bearing_rad, (best_i + 0.5) * bw)
        self._prev = list(profile)

    def state(self) -> dict[str, Any]:
        sy = self._est.estimate()
        return {
            "suggested_yaw_rad": None if sy is None else round(sy, 5),
            "suggested_yaw_deg": None if sy is None else round(math.degrees(sy), 1),
            "applies_automatically": False,          # advisory — operator/self-improve approves
            **self._est.confidence(),
            **self._est.diagnostics(),
        }


# per-sensor calibrators (mutable module state — pure logic, writes no belief/memory/policy)
_calibrators: dict[str, YawCalibrator] = {}


def feed_calibration(sensor_id: str, person_camera_bearing_rad: float,
                     profile: Sequence[Optional[float]], bin_w: Optional[float] = None) -> None:
    cal = _calibrators.get(sensor_id)
    if cal is None:
        cal = _calibrators[sensor_id] = YawCalibrator()
    cal.update(person_camera_bearing_rad, profile, bin_w)


def calibration_state(sensor_id: str) -> dict[str, Any]:
    cal = _calibrators.get(sensor_id)
    return cal.state() if cal else {"suggested_yaw_rad": None, "pairs": 0, "ready": False,
                                     "applies_automatically": False}
