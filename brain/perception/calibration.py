"""Camera calibration and room-frame transform for spatial intelligence.

Manages intrinsic camera parameters, calibration state (valid/stale/invalid),
and coordinate transforms from camera frame to room frame.  Calibration
files are stored in ~/.jarvis/spatial/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognition.spatial_schema import (
    CalibrationState,
    CALIBRATION_VALID_DURATION_S,
    CALIBRATION_STALE_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

_CALIBRATION_DIR = Path.home() / ".jarvis" / "spatial"
_CALIBRATION_FILE = _CALIBRATION_DIR / "calibration.json"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for spatial estimation."""
    focal_length_px: float = 800.0
    principal_x: float = 960.0
    principal_y: float = 540.0
    frame_width: int = 1920
    frame_height: int = 1080

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_length_px": self.focal_length_px,
            "principal_x": self.principal_x,
            "principal_y": self.principal_y,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CameraIntrinsics:
        return CameraIntrinsics(
            focal_length_px=d.get("focal_length_px", 800.0),
            principal_x=d.get("principal_x", 960.0),
            principal_y=d.get("principal_y", 540.0),
            frame_width=d.get("frame_width", 1920),
            frame_height=d.get("frame_height", 1080),
        )


@dataclass
class RoomTransform:
    """Transform from camera coordinates to room-frame coordinates.

    Phase 1 (fixed camera, single room): simple offset, no rotation.
    """
    camera_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_rotation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def camera_to_room(
        self, cam_pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Transform camera-frame position to room-frame position."""
        return (
            cam_pos[0] + self.camera_position_m[0],
            cam_pos[1] + self.camera_position_m[1],
            cam_pos[2] + self.camera_position_m[2],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_position_m": list(self.camera_position_m),
            "camera_rotation_rpy": list(self.camera_rotation_rpy),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RoomTransform:
        return RoomTransform(
            camera_position_m=tuple(d.get("camera_position_m", [0, 0, 0])),
            camera_rotation_rpy=tuple(d.get("camera_rotation_rpy", [0, 0, 0])),
        )


@dataclass
class CalibrationProfile:
    """Reusable calibration profile for scene relocalization handoff."""

    profile_id: str
    intrinsics: CameraIntrinsics
    transform: RoomTransform
    scene_signature: dict[str, Any] = field(default_factory=dict)
    last_verified_ts: float = 0.0
    created_ts: float = 0.0
    updated_ts: float = 0.0
    anchor_consistency_ok: bool = True
    use_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "intrinsics": self.intrinsics.to_dict(),
            "transform": self.transform.to_dict(),
            "scene_signature": dict(self.scene_signature or {}),
            "last_verified_ts": float(self.last_verified_ts or 0.0),
            "created_ts": float(self.created_ts or 0.0),
            "updated_ts": float(self.updated_ts or 0.0),
            "anchor_consistency_ok": bool(self.anchor_consistency_ok),
            "use_count": int(self.use_count or 0),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CalibrationProfile:
        now = time.time()
        created = float(d.get("created_ts", 0.0) or 0.0)
        updated = float(d.get("updated_ts", 0.0) or 0.0)
        if created <= 0.0:
            created = now
        if updated <= 0.0:
            updated = created
        return CalibrationProfile(
            profile_id=str(d.get("profile_id", "") or "default"),
            intrinsics=CameraIntrinsics.from_dict(d.get("intrinsics", {}) or {}),
            transform=RoomTransform.from_dict(d.get("transform", {}) or {}),
            scene_signature=dict(d.get("scene_signature", {}) or {}),
            last_verified_ts=float(d.get("last_verified_ts", 0.0) or 0.0),
            created_ts=created,
            updated_ts=updated,
            anchor_consistency_ok=bool(d.get("anchor_consistency_ok", True)),
            use_count=int(d.get("use_count", 0) or 0),
        )


class CalibrationManager:
    """Manages camera calibration state and persistence.

    Three degradation states gate what the spatial pipeline can do:
      valid   -> full spatial pipeline active
      stale   -> diagnostics + shadow/advisory only
      invalid -> hard trust break only (uncalibrated/inconsistent anchors)
    """

    def __init__(self) -> None:
        self._intrinsics = CameraIntrinsics()
        self._transform = RoomTransform()
        self._version: int = 0
        self._state: CalibrationState = "invalid"
        self._state_reason: str = "uncalibrated"
        self._last_verified_ts: float = 0.0
        self._created_ts: float = 0.0
        self._anchor_consistency_ok: bool = False
        self._handoff_pending_verify: bool = False
        self._profiles: dict[str, CalibrationProfile] = {}
        self._active_profile_id: str = "default"
        self._load()
        self._ensure_profile_state()

    @property
    def state(self) -> CalibrationState:
        self._refresh_state()
        return self._state

    @property
    def version(self) -> int:
        return self._version

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def transform(self) -> RoomTransform:
        return self._transform

    def is_usable(self) -> bool:
        return self.state == "valid"

    def is_advisory_only(self) -> bool:
        return self.state == "stale"

    @property
    def active_profile_id(self) -> str:
        self._ensure_profile_state()
        return self._active_profile_id

    @property
    def profile_count(self) -> int:
        self._ensure_profile_state()
        return len(self._profiles)

    def suggest_profile_id(self, scene_signature: dict[str, Any]) -> str:
        """Build a stable profile id from a scene signature."""
        raw = json.dumps(scene_signature or {}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"scene_{digest}"

    def upsert_profile(
        self,
        profile_id: str,
        *,
        scene_signature: dict[str, Any] | None = None,
        intrinsics: CameraIntrinsics | None = None,
        transform: RoomTransform | None = None,
        anchor_consistency_ok: bool | None = None,
        last_verified_ts: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Create or update a calibration profile."""
        self._ensure_profile_state()
        pid = str(profile_id or "").strip() or "default"
        now = time.time()
        profile = self._profiles.get(pid)
        if profile is None:
            profile = CalibrationProfile(
                profile_id=pid,
                intrinsics=intrinsics or self._intrinsics,
                transform=transform or self._transform,
                created_ts=now,
                updated_ts=now,
                last_verified_ts=float(last_verified_ts or self._last_verified_ts or now),
                anchor_consistency_ok=(
                    bool(anchor_consistency_ok)
                    if anchor_consistency_ok is not None
                    else bool(self._anchor_consistency_ok)
                ),
            )
            self._profiles[pid] = profile
        else:
            if intrinsics is not None:
                profile.intrinsics = intrinsics
            if transform is not None:
                profile.transform = transform
            if anchor_consistency_ok is not None:
                profile.anchor_consistency_ok = bool(anchor_consistency_ok)
            if last_verified_ts is not None:
                profile.last_verified_ts = float(last_verified_ts)
            profile.updated_ts = now
        if scene_signature:
            profile.scene_signature = dict(scene_signature)
            profile.updated_ts = now
        if persist:
            self._save()
        return profile.to_dict()

    def match_profile(
        self,
        scene_signature: dict[str, Any] | None,
        *,
        min_score: float = 0.55,
        include_active: bool = False,
    ) -> tuple[str, float] | None:
        """Return best matching profile_id and score for a scene signature."""
        self._ensure_profile_state()
        sig = scene_signature or {}
        if not sig:
            return None

        best_id = ""
        best_score = 0.0
        for pid, profile in self._profiles.items():
            if not include_active and pid == self._active_profile_id:
                continue
            score = self._score_scene_signature(sig, profile.scene_signature)
            if score > best_score:
                best_score = score
                best_id = pid
        if not best_id or best_score < min_score:
            return None
        return best_id, best_score

    def activate_profile(
        self,
        profile_id: str,
        *,
        stale_reason: str = "profile_handoff_local_only",
        persist: bool = True,
    ) -> bool:
        """Activate an existing profile and switch runtime calibration to it."""
        self._ensure_profile_state()
        pid = str(profile_id or "").strip()
        if not pid or pid not in self._profiles:
            return False
        profile = self._profiles[pid]
        now = time.time()

        self._active_profile_id = pid
        self._intrinsics = profile.intrinsics
        self._transform = profile.transform
        self._last_verified_ts = float(profile.last_verified_ts or self._last_verified_ts or now)
        self._anchor_consistency_ok = bool(profile.anchor_consistency_ok)
        if self._created_ts == 0.0:
            self._created_ts = now
        self._version += 1
        self._handoff_pending_verify = True
        self._refresh_state()

        # Activating a profile after a scene jump is advisory until re-verified.
        if self._state == "valid":
            self._state = "stale"
        self._state_reason = stale_reason

        profile.use_count += 1
        profile.updated_ts = now
        if persist:
            self._save()
        return True

    def update_intrinsics(
        self,
        focal_length_px: float,
        principal_x: float | None = None,
        principal_y: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> None:
        self._ensure_profile_state()
        self._intrinsics = CameraIntrinsics(
            focal_length_px=focal_length_px,
            principal_x=principal_x if principal_x is not None else self._intrinsics.principal_x,
            principal_y=principal_y if principal_y is not None else self._intrinsics.principal_y,
            frame_width=frame_width if frame_width is not None else self._intrinsics.frame_width,
            frame_height=frame_height if frame_height is not None else self._intrinsics.frame_height,
        )
        self._bump_version()

    def update_transform(
        self,
        camera_position_m: tuple[float, float, float],
        camera_rotation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._ensure_profile_state()
        self._transform = RoomTransform(
            camera_position_m=camera_position_m,
            camera_rotation_rpy=camera_rotation_rpy,
        )
        self._bump_version()

    def verify(self, anchor_consistency_ok: bool = True) -> None:
        """Mark calibration as freshly verified."""
        self._ensure_profile_state()
        self._last_verified_ts = time.time()
        self._anchor_consistency_ok = anchor_consistency_ok
        self._handoff_pending_verify = False
        self._refresh_state()
        self._sync_active_profile_from_runtime()
        self._save()

    def setup_pi_camera_defaults(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        camera_position_m: tuple[float, float, float] = (0.0, 1.2, 0.0),
    ) -> bool:
        """Apply known Pi Camera Module 3 (IMX708) defaults if uncalibrated.

        Returns True if defaults were applied, False if calibration already set.

        IMX708: 4.74mm focal length, 1.4µm pixel pitch.
        At native 4608x2592 → f_px ≈ 3386.
        Raw mode 2328x1748, output resized to 640x480.
        Effective focal length at 640px wide = 3386 * (640/4608) ≈ 470px.
        """
        self._ensure_profile_state()
        if self._version > 0:
            return False

        scale = frame_width / 4608.0
        focal_px = 3386.0 * scale

        self._intrinsics = CameraIntrinsics(
            focal_length_px=round(focal_px, 1),
            principal_x=frame_width / 2.0,
            principal_y=frame_height / 2.0,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self._transform = RoomTransform(
            camera_position_m=camera_position_m,
            camera_rotation_rpy=(0.0, 0.0, 0.0),
        )
        self._bump_version()
        logger.info(
            "Applied Pi Camera Module 3 defaults: f=%.1fpx, frame=%dx%d, pos=%s",
            focal_px, frame_width, frame_height, camera_position_m,
        )
        return True

    def invalidate(self, reason: str = "") -> None:
        """Force calibration to invalid state."""
        self._ensure_profile_state()
        self._anchor_consistency_ok = False
        self._handoff_pending_verify = False
        self._refresh_state()
        self._sync_active_profile_from_runtime()
        if reason:
            logger.warning("Calibration invalidated: %s", reason)

    def camera_to_room(
        self, cam_pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return self._transform.camera_to_room(cam_pos)

    def get_state(self) -> dict[str, Any]:
        self._ensure_profile_state()
        self._refresh_state()
        now = time.time()
        age = now - self._last_verified_ts if self._last_verified_ts > 0 else None
        profile_rows: list[dict[str, Any]] = []
        for profile in self._profiles.values():
            profile_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "labels": list((profile.scene_signature or {}).get("labels", [])),
                    "entity_count": int((profile.scene_signature or {}).get("entity_count", 0) or 0),
                    "display_count": int((profile.scene_signature or {}).get("display_count", 0) or 0),
                    "last_verified_ts": round(float(profile.last_verified_ts or 0.0), 1),
                    "updated_ts": round(float(profile.updated_ts or 0.0), 1),
                    "use_count": int(profile.use_count or 0),
                }
            )
        profile_rows.sort(key=lambda p: p.get("updated_ts", 0.0), reverse=True)
        return {
            "state": self._state,
            "reason": self._state_reason,
            "version": self._version,
            "intrinsics": self._intrinsics.to_dict(),
            "transform": self._transform.to_dict(),
            "active_profile_id": self._active_profile_id,
            "profile_count": len(self._profiles),
            "profiles": profile_rows[:8],
            "last_verified_ts": round(self._last_verified_ts, 1),
            "created_ts": round(self._created_ts, 1),
            "anchor_consistency_ok": self._anchor_consistency_ok,
            "handoff_pending_verify": bool(self._handoff_pending_verify),
            "age_s": round(age, 1) if age is not None else None,
        }

    # --- internals ---

    def _refresh_state(self) -> None:
        if self._version == 0 or self._last_verified_ts == 0.0:
            self._state = "invalid"
            self._state_reason = "uncalibrated"
            return
        if not self._anchor_consistency_ok:
            self._state = "invalid"
            self._state_reason = "anchor_inconsistency"
            return
        if bool(getattr(self, "_handoff_pending_verify", False)):
            self._state = "stale"
            self._state_reason = "profile_handoff_local_only"
            return
        age = time.time() - self._last_verified_ts
        if age <= CALIBRATION_VALID_DURATION_S:
            self._state = "valid"
            self._state_reason = "verified_recently"
        elif age <= CALIBRATION_STALE_TIMEOUT_S:
            self._state = "stale"
            self._state_reason = "verification_aged"
        else:
            # Keep long-expired calibration in advisory mode so mobile scenes
            # continue to produce camera-relative spatial continuity.
            self._state = "stale"
            self._state_reason = "verification_expired_local_only"

    def _bump_version(self) -> None:
        self._ensure_profile_state()
        self._version += 1
        now = time.time()
        self._last_verified_ts = now
        if self._created_ts == 0.0:
            self._created_ts = now
        self._anchor_consistency_ok = True
        self._handoff_pending_verify = False
        self._refresh_state()
        self._sync_active_profile_from_runtime()
        self._save()

    def _save(self) -> None:
        self._ensure_profile_state()
        try:
            _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
            self._sync_active_profile_from_runtime()
            data = {
                "version": self._version,
                "intrinsics": self._intrinsics.to_dict(),
                "transform": self._transform.to_dict(),
                "last_verified_ts": self._last_verified_ts,
                "created_ts": self._created_ts,
                "anchor_consistency_ok": self._anchor_consistency_ok,
                "handoff_pending_verify": self._handoff_pending_verify,
                "active_profile_id": self._active_profile_id,
                "profiles": {
                    pid: prof.to_dict()
                    for pid, prof in self._profiles.items()
                },
            }
            tmp = _CALIBRATION_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp), str(_CALIBRATION_FILE))
        except Exception as e:
            logger.warning("Failed to save calibration: %s", e)

    def _load(self) -> None:
        try:
            if _CALIBRATION_FILE.exists():
                with open(_CALIBRATION_FILE) as f:
                    data = json.load(f)
                self._version = data.get("version", 0)
                self._intrinsics = CameraIntrinsics.from_dict(
                    data.get("intrinsics", {}),
                )
                self._transform = RoomTransform.from_dict(
                    data.get("transform", {}),
                )
                self._last_verified_ts = data.get("last_verified_ts", 0.0)
                self._created_ts = data.get("created_ts", 0.0)
                self._anchor_consistency_ok = data.get(
                    "anchor_consistency_ok", False,
                )
                self._handoff_pending_verify = bool(
                    data.get("handoff_pending_verify", False),
                )
                self._active_profile_id = str(data.get("active_profile_id", "") or "default")
                raw_profiles = data.get("profiles", {})
                if isinstance(raw_profiles, dict):
                    for pid, row in raw_profiles.items():
                        if not isinstance(row, dict):
                            continue
                        loaded = CalibrationProfile.from_dict(row)
                        loaded.profile_id = str(loaded.profile_id or pid or "default")
                        self._profiles[loaded.profile_id] = loaded
                # Backward compatibility: old file format had no profile map.
                if not self._profiles:
                    now = time.time()
                    self._profiles["default"] = CalibrationProfile(
                        profile_id="default",
                        intrinsics=self._intrinsics,
                        transform=self._transform,
                        last_verified_ts=float(self._last_verified_ts or 0.0),
                        created_ts=float(self._created_ts or now),
                        updated_ts=float(self._created_ts or now),
                        anchor_consistency_ok=bool(self._anchor_consistency_ok),
                        scene_signature={},
                        use_count=0,
                    )
                if self._active_profile_id not in self._profiles:
                    self._active_profile_id = "default"
                active = self._profiles.get(self._active_profile_id)
                if active is not None:
                    # Use active profile values as runtime source of truth.
                    self._intrinsics = active.intrinsics
                    self._transform = active.transform
                    if active.last_verified_ts > 0.0:
                        self._last_verified_ts = active.last_verified_ts
                    self._anchor_consistency_ok = bool(active.anchor_consistency_ok)
                self._refresh_state()
                logger.info(
                    "Loaded calibration v%d (state=%s)",
                    self._version, self._state,
                )
        except Exception as e:
            logger.warning("Failed to load calibration: %s", e)

    def _ensure_profile_state(self) -> None:
        if not isinstance(getattr(self, "_profiles", None), dict):
            self._profiles = {}
        if not isinstance(getattr(self, "_active_profile_id", None), str):
            self._active_profile_id = "default"
        if not isinstance(getattr(self, "_handoff_pending_verify", None), bool):
            self._handoff_pending_verify = False

        if not self._active_profile_id:
            self._active_profile_id = "default"

        if self._active_profile_id not in self._profiles:
            now = time.time()
            self._profiles[self._active_profile_id] = CalibrationProfile(
                profile_id=self._active_profile_id,
                intrinsics=self._intrinsics,
                transform=self._transform,
                scene_signature={},
                last_verified_ts=float(getattr(self, "_last_verified_ts", 0.0) or 0.0),
                created_ts=float(getattr(self, "_created_ts", 0.0) or now),
                updated_ts=now,
                anchor_consistency_ok=bool(getattr(self, "_anchor_consistency_ok", True)),
                use_count=0,
            )

    def _sync_active_profile_from_runtime(self) -> None:
        self._ensure_profile_state()
        now = time.time()
        profile = self._profiles.get(self._active_profile_id)
        if profile is None:
            return
        profile.intrinsics = self._intrinsics
        profile.transform = self._transform
        profile.last_verified_ts = float(self._last_verified_ts or 0.0)
        profile.anchor_consistency_ok = bool(self._anchor_consistency_ok)
        if profile.created_ts <= 0.0:
            profile.created_ts = float(self._created_ts or now)
        profile.updated_ts = now

    @staticmethod
    def _score_scene_signature(a: dict[str, Any], b: dict[str, Any]) -> float:
        if not a or not b:
            return 0.0

        def _set(key: str) -> tuple[set[str], set[str]]:
            av = a.get(key, [])
            bv = b.get(key, [])
            if not isinstance(av, list):
                av = []
            if not isinstance(bv, list):
                bv = []
            return set(str(x) for x in av), set(str(x) for x in bv)

        labels_a, labels_b = _set("labels")
        displays_a, displays_b = _set("display_kinds")
        regions_a, regions_b = _set("regions")

        def _jaccard(sa: set[str], sb: set[str]) -> float:
            union = sa | sb
            if not union:
                return 0.0
            return len(sa & sb) / len(union)

        label_score = _jaccard(labels_a, labels_b)
        display_score = _jaccard(displays_a, displays_b)
        region_score = _jaccard(regions_a, regions_b)

        cnt_a = int(a.get("entity_count", 0) or 0)
        cnt_b = int(b.get("entity_count", 0) or 0)
        denom = max(1, cnt_a, cnt_b)
        count_score = max(0.0, 1.0 - (abs(cnt_a - cnt_b) / denom))

        return (
            0.55 * label_score
            + 0.20 * display_score
            + 0.15 * region_score
            + 0.10 * count_score
        )
