"""Spatial Intelligence Phase 1 — data contracts and configuration.

Defines the bounded spatial observation types, anchor/track state machines,
class-specific thresholds, and authority ordering.  These types augment the
existing scene continuity system without replacing SceneEntity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Typed states
# ---------------------------------------------------------------------------

TrackStatus = Literal["provisional", "stable", "occluded", "stale"]
DeltaType = Literal["moved", "stabilized", "distance_changed", "missing", "reappeared"]
AnchorType = Literal[
    "desk_plane", "monitor_left", "monitor_center", "monitor_right",
    "chair_zone", "seat_origin", "custom",
]
CalibrationState = Literal["valid", "stale", "invalid"]

SpatialRelationType = Literal[
    "left_of", "right_of", "in_front_of", "behind",
    "near", "on", "centered_in",
]

# ---------------------------------------------------------------------------
# Authority ordering (higher value = more authoritative)
# ---------------------------------------------------------------------------

AUTHORITY_LEVELS: dict[str, int] = {
    "calibration": 50,
    "stable_anchor": 40,
    "stable_track": 30,
    "provisional_track": 20,
    "raw_observation": 10,
}

# ---------------------------------------------------------------------------
# Per-class jitter thresholds (meters)
# ---------------------------------------------------------------------------

CLASS_MOVE_THRESHOLDS: dict[str, float] = {
    "monitor": 0.15,
    "tv": 0.15,
    "laptop": 0.10,
    "keyboard": 0.08,
    "mouse": 0.05,
    "cup": 0.05,
    "bottle": 0.05,
    "chair": 0.12,
    "person": 0.20,
    "desk": 0.30,
}

DEFAULT_MOVE_THRESHOLD: float = 0.10

# ---------------------------------------------------------------------------
# Known-size priors (meters) for distance estimation
# ---------------------------------------------------------------------------

KNOWN_SIZE_PRIORS: dict[str, float] = {
    "monitor": 0.60,
    "tv": 0.80,
    "laptop": 0.35,
    "keyboard": 0.45,
    "mouse": 0.06,
    "cup": 0.08,
    "bottle": 0.07,
    "chair": 0.50,
    "person": 0.45,
}

# ---------------------------------------------------------------------------
# Promotion thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD_TRACK: float = 0.45
CONFIDENCE_THRESHOLD_STABLE: float = 0.60
CONFIDENCE_THRESHOLD_ANCHOR: float = 0.70
CONFIDENCE_THRESHOLD_DELTA: float = 0.80
CONFIDENCE_THRESHOLD_MEMORY: float = 0.88

STABLE_WINDOWS_REQUIRED: int = 3
DELTA_CONSECUTIVE_WINDOWS: int = 2

# ---------------------------------------------------------------------------
# Calibration degradation timing
# ---------------------------------------------------------------------------

CALIBRATION_VALID_DURATION_S: float = 86400.0
CALIBRATION_STALE_TIMEOUT_S: float = 172800.0

# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------

SPATIAL_MEMORY_MAX_PER_HOUR: int = 5
SPATIAL_MEMORY_MAX_PER_DAY: int = 20

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

SMOOTHING_ALPHA: float = 0.3

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SpatialObservation:
    """Per-frame spatial estimate for a scene entity."""
    entity_id: str
    label: str
    depth_m: float
    position_camera_m: tuple[float, float, float]
    position_room_m: tuple[float, float, float] | None = None
    size_estimate_m: float = 0.0
    confidence: float = 0.0
    uncertainty_m: float = 1.0
    calibration_version: int = 0
    provenance: str = "prior_based"
    timestamp: float = 0.0
    bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "depth_m": round(self.depth_m, 3),
            "position_camera_m": tuple(round(v, 3) for v in self.position_camera_m),
            "position_room_m": (
                tuple(round(v, 3) for v in self.position_room_m)
                if self.position_room_m else None
            ),
            "size_estimate_m": round(self.size_estimate_m, 3),
            "confidence": round(self.confidence, 3),
            "uncertainty_m": round(self.uncertainty_m, 3),
            "calibration_version": self.calibration_version,
            "provenance": self.provenance,
            "timestamp": round(self.timestamp, 1),
        }


@dataclass
class SpatialAnchor:
    """Stable room reference point (desk, monitor, chair, etc.)."""
    anchor_id: str
    anchor_type: AnchorType
    label: str
    position_room_m: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    confidence: float = 0.0
    stable_since_ts: float = 0.0
    last_verified_ts: float = 0.0
    calibration_version: int = 0
    authority: int = AUTHORITY_LEVELS["stable_anchor"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "label": self.label,
            "position_room_m": tuple(round(v, 3) for v in self.position_room_m),
            "dimensions_m": tuple(round(v, 3) for v in self.dimensions_m),
            "confidence": round(self.confidence, 3),
            "stable_since_ts": round(self.stable_since_ts, 1),
            "last_verified_ts": round(self.last_verified_ts, 1),
            "calibration_version": self.calibration_version,
            "authority": self.authority,
        }


@dataclass
class SpatialTrack:
    """Smoothed per-entity spatial state with uncertainty."""
    entity_id: str
    label: str
    track_status: TrackStatus = "provisional"
    position_room_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uncertainty_m: float = 1.0
    confidence: float = 0.0
    samples: int = 0
    stable_windows: int = 0
    first_seen_ts: float = 0.0
    last_update_ts: float = 0.0
    anchor_id: str | None = None
    authority: int = AUTHORITY_LEVELS["provisional_track"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "track_status": self.track_status,
            "position_room_m": tuple(round(v, 3) for v in self.position_room_m),
            "velocity_mps": tuple(round(v, 3) for v in self.velocity_mps),
            "dimensions_m": tuple(round(v, 3) for v in self.dimensions_m),
            "uncertainty_m": round(self.uncertainty_m, 3),
            "confidence": round(self.confidence, 3),
            "samples": self.samples,
            "stable_windows": self.stable_windows,
            "authority": self.authority,
        }


@dataclass
class SpatialDelta:
    """Promotion-facing spatial change event."""
    delta_id: str
    entity_id: str
    label: str
    delta_type: DeltaType
    from_position_m: tuple[float, float, float] | None = None
    to_position_m: tuple[float, float, float] | None = None
    distance_m: float = 0.0
    dominant_axis: str = ""
    confidence: float = 0.0
    uncertainty_m: float = 1.0
    validated: bool = False
    reason_codes: list[str] = field(default_factory=list)
    calibration_version: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "entity_id": self.entity_id,
            "label": self.label,
            "delta_type": self.delta_type,
            "from_position_m": (
                tuple(round(v, 3) for v in self.from_position_m)
                if self.from_position_m else None
            ),
            "to_position_m": (
                tuple(round(v, 3) for v in self.to_position_m)
                if self.to_position_m else None
            ),
            "distance_m": round(self.distance_m, 3),
            "dominant_axis": self.dominant_axis,
            "confidence": round(self.confidence, 3),
            "uncertainty_m": round(self.uncertainty_m, 3),
            "validated": self.validated,
            "reason_codes": self.reason_codes,
            "calibration_version": self.calibration_version,
        }


@dataclass
class SpatialRelationFact:
    """Spatial relation between two entities."""
    relation_id: str
    source_entity_id: str
    relation_type: SpatialRelationType
    target_entity_id: str
    value_m: float | None = None
    confidence: float = 0.0
    calibration_version: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_entity_id": self.source_entity_id,
            "relation_type": self.relation_type,
            "target_entity_id": self.target_entity_id,
            "value_m": round(self.value_m, 3) if self.value_m is not None else None,
            "confidence": round(self.confidence, 3),
            "calibration_version": self.calibration_version,
        }
