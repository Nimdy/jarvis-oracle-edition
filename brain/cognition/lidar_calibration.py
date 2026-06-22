"""LiDAR → room-frame extrinsic (PR3 — inert by default).

The S2 ``RoomModel`` geometry is in the **lidar_sensor frame**: a single horizontal
slice where X = right (range·sin bearing), Z = forward (range·cos), sampled at the
sensor's mount height. Camera entities + spatial anchors live in the **room frame**
(X = right, Y = up, Z = forward — a LEFT-handed / Unity-style frame, NOT OpenGL
right-handed; a camera-fusion frame must match this, not assume Z = backward). To ever
attach a camera label to a lidar wall, both must be expressed in the SAME frame. This module is that rigid
transform — and nothing more.

INERT BY DEFAULT. The identity extrinsic (all zeros) is a pass-through: room frame
== lidar frame, so importing or calling this changes no geometry until a real
extrinsic is configured. Pure stdlib, no live-brain deps, telemetry-only
(registered in the HRR forbidden-import scan). The fusion code that actually USES
this transform — synthesising anchors, attaching labels — is a separate, gated
step (PR4); this file does not wire anything into the live path.

A level (horizontal) S2 needs only a yaw about room +Y plus a translation; tilt
(roll/pitch) is a Phase-2 concern and intentionally not modelled here.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any


def _env_float(name: str, default: float) -> float:
    """Config that PERSISTS via env (not a hardcoded magic number); the literal is
    the documented default. Identity unless explicitly configured."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class LidarExtrinsic:
    """Rigid ``lidar_sensor`` → ``room`` transform for a level 2D lidar.

    yaw_rad : rotation of the lidar's bearing-zero about room +Y (vertical/up).
    tx/ty/tz_m : the lidar origin's position in the room frame, metres
                 (ty = the height of the horizontal scan plane).

    Identity (all zero) == pass-through == inert. Convention (yaw θ):
        x_room = x·cosθ + z·sinθ
        z_room = −x·sinθ + z·cosθ
    i.e. a positive yaw rotates the lidar frame clockwise viewed from above.
    """
    yaw_rad: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_YAW_RAD", 0.0))
    tx_m: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_TX_M", 0.0))
    ty_m: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_MOUNT_HEIGHT_M", 0.0))
    tz_m: float = field(default_factory=lambda: _env_float("JARVIS_LIDAR_TZ_M", 0.0))
    # Camera optical-centre height in the room frame (metres). Not part of the rigid
    # lidar→room transform, but a per-rig mount parameter stored alongside it — the Tier-3
    # depth anchor reads it off the extrinsic (lidar_plane_row uses ty_m − camera_height_m).
    # Default preserves the prior hard-coded fallback.
    camera_height_m: float = field(default_factory=lambda: _env_float("JARVIS_CAMERA_HEIGHT_M", 1.219))

    @classmethod
    def identity(cls) -> "LidarExtrinsic":
        return cls(0.0, 0.0, 0.0, 0.0)

    @property
    def is_identity(self) -> bool:
        return self.yaw_rad == 0.0 and self.tx_m == 0.0 and self.ty_m == 0.0 and self.tz_m == 0.0

    # -- core transform ----------------------------------------------------
    def transform_xz(self, x_l: float, z_l: float) -> tuple[float, float, float]:
        """A 2D lidar point (x = right, z = forward) → 3D room point (x, y, z).

        The scan plane is horizontal, so the room Y is the constant mount height.
        """
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        x_r = c * x_l + s * z_l
        z_r = -s * x_l + c * z_l
        return (x_r + self.tx_m, self.ty_m, z_r + self.tz_m)

    def transform_wall(self, wall: dict[str, Any]) -> dict[str, Any]:
        """Add room-frame 3D endpoints to a ``RoomModel`` wall dict (additive, pure).

        Keeps the original lidar-frame ``start_m``/``end_m`` untouched; adds
        ``start_room_m``/``end_room_m`` so the provenance of each is explicit.
        """
        sx = wall.get("start_m") or (0.0, 0.0)
        ex = wall.get("end_m") or (0.0, 0.0)
        s = self.transform_xz(float(sx[0]), float(sx[1]))
        e = self.transform_xz(float(ex[0]), float(ex[1]))
        out = dict(wall)
        out["start_room_m"] = [round(v, 3) for v in s]
        out["end_room_m"] = [round(v, 3) for v in e]
        return out

    def transform_room(self, room: dict[str, Any]) -> dict[str, Any]:
        """Add room-frame geometry to a ``RoomModel.to_dict()`` (telemetry-only view).

        Additive + pure. CRUCIAL: ``frame`` is left UNCHANGED — only the explicit
        ``*_room_m`` fields are room-frame; ``points_m``/``profile``/``openings`` bearings/
        ``nearest_per_sector_m``/walls' ``start_m``/``end_m`` all remain LIDAR-frame. A
        consumer must read the room-frame fields explicitly; keying on ``frame`` to decide
        the geometry's frame would rotate-corrupt the un-transformed fields by yaw.
        """
        out = dict(room)
        out["room_frame_available"] = True            # NOT 'frame' — the dict stays mixed-frame
        out["extrinsic"] = {
            "yaw_rad": round(self.yaw_rad, 5),
            "translation_m": [round(self.tx_m, 3), round(self.ty_m, 3), round(self.tz_m, 3)],
            "is_identity": self.is_identity,
        }
        out["walls"] = [self.transform_wall(w) for w in room.get("walls", [])]
        out["points_room_m"] = [
            [round(v, 3) for v in self.transform_xz(float(p[0]), float(p[1]))]
            for p in room.get("points_m", [])
        ]
        out["authority"] = "spatial_telemetry_only"
        out["writes_beliefs"] = False
        return out


_EXTRINSIC_FILE = os.path.join(os.path.expanduser("~"), ".jarvis", "lidar_extrinsic.json")


def load_extrinsic() -> LidarExtrinsic:
    """The configured lidar→room extrinsic. Resolution (first wins, else identity):

    1. ``~/.jarvis/lidar_extrinsic.json`` — per-instance mount config (persists across
       restarts; this is where a specific rig's measured mount lives).
    2. ``JARVIS_LIDAR_*`` env vars.
    3. Identity — no mount configured (the product default for users with no lidar
       or an un-calibrated one; the geometry simply isn't re-framed).
    """
    import json
    try:
        with open(_EXTRINSIC_FILE) as f:
            c = json.load(f)
        if not isinstance(c, dict):
            return LidarExtrinsic()       # malformed (e.g. a JSON list) → identity, never crash
        return LidarExtrinsic(
            yaw_rad=float(c.get("yaw_rad", 0.0)),
            tx_m=float(c.get("tx_m", 0.0)),
            ty_m=float(c.get("ty_m", 0.0)),
            tz_m=float(c.get("tz_m", 0.0)),
            camera_height_m=float(c.get("camera_height_m", 1.219)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return LidarExtrinsic()   # env-overridable, else identity
