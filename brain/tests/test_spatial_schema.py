"""Spatial schema — dataclass and configuration tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CLASS_MOVE_THRESHOLDS,
    CONFIDENCE_THRESHOLD_ANCHOR,
    CONFIDENCE_THRESHOLD_DELTA,
    CONFIDENCE_THRESHOLD_MEMORY,
    CONFIDENCE_THRESHOLD_STABLE,
    CONFIDENCE_THRESHOLD_TRACK,
    DEFAULT_MOVE_THRESHOLD,
    KNOWN_SIZE_PRIORS,
    SPATIAL_MEMORY_MAX_PER_DAY,
    SPATIAL_MEMORY_MAX_PER_HOUR,
    SpatialAnchor,
    SpatialDelta,
    SpatialObservation,
    SpatialRelationFact,
    SpatialTrack,
)


# -- Authority ordering --


def test_authority_ordering():
    """Calibration > stable_anchor > stable_track > provisional_track > raw_observation."""
    assert AUTHORITY_LEVELS["calibration"] > AUTHORITY_LEVELS["stable_anchor"]
    assert AUTHORITY_LEVELS["stable_anchor"] > AUTHORITY_LEVELS["stable_track"]
    assert AUTHORITY_LEVELS["stable_track"] > AUTHORITY_LEVELS["provisional_track"]
    assert AUTHORITY_LEVELS["provisional_track"] > AUTHORITY_LEVELS["raw_observation"]


# -- Class-specific thresholds --


def test_class_move_thresholds_populated():
    assert len(CLASS_MOVE_THRESHOLDS) >= 8
    for label, threshold in CLASS_MOVE_THRESHOLDS.items():
        assert threshold > 0, f"{label} threshold must be positive"


def test_monitor_threshold_higher_than_cup():
    assert CLASS_MOVE_THRESHOLDS["monitor"] > CLASS_MOVE_THRESHOLDS["cup"]


def test_desk_threshold_is_highest():
    for label, threshold in CLASS_MOVE_THRESHOLDS.items():
        if label != "desk":
            assert CLASS_MOVE_THRESHOLDS["desk"] >= threshold, (
                f"desk threshold should be >= {label}"
            )


def test_default_move_threshold_positive():
    assert DEFAULT_MOVE_THRESHOLD > 0


# -- Known-size priors --


def test_known_size_priors_populated():
    assert len(KNOWN_SIZE_PRIORS) >= 6
    for label, size in KNOWN_SIZE_PRIORS.items():
        assert size > 0, f"{label} size must be positive"


# -- Promotion threshold ordering --


def test_promotion_thresholds_increasing():
    assert CONFIDENCE_THRESHOLD_TRACK < CONFIDENCE_THRESHOLD_STABLE
    assert CONFIDENCE_THRESHOLD_STABLE < CONFIDENCE_THRESHOLD_ANCHOR
    assert CONFIDENCE_THRESHOLD_ANCHOR < CONFIDENCE_THRESHOLD_DELTA
    assert CONFIDENCE_THRESHOLD_DELTA < CONFIDENCE_THRESHOLD_MEMORY


# -- Memory budget --


def test_memory_budget_reasonable():
    assert SPATIAL_MEMORY_MAX_PER_HOUR >= 1
    assert SPATIAL_MEMORY_MAX_PER_DAY >= SPATIAL_MEMORY_MAX_PER_HOUR


# -- Dataclass construction and serialization --


def test_spatial_observation_to_dict():
    obs = SpatialObservation(
        entity_id="obj_abc",
        label="cup",
        depth_m=1.5,
        position_camera_m=(0.1, -0.2, 1.5),
        position_room_m=(1.1, -0.2, 1.5),
        confidence=0.75,
    )
    d = obs.to_dict()
    assert d["entity_id"] == "obj_abc"
    assert d["depth_m"] == 1.5
    assert d["position_room_m"] is not None
    assert len(d["position_camera_m"]) == 3


def test_spatial_observation_no_room_pos():
    obs = SpatialObservation(
        entity_id="obj_abc",
        label="cup",
        depth_m=1.5,
        position_camera_m=(0.1, -0.2, 1.5),
    )
    d = obs.to_dict()
    assert d["position_room_m"] is None


def test_spatial_anchor_to_dict():
    anchor = SpatialAnchor(
        anchor_id="anchor_desk",
        anchor_type="desk_plane",
        label="desk",
        position_room_m=(0.0, -0.5, 1.0),
        confidence=0.85,
    )
    d = anchor.to_dict()
    assert d["anchor_id"] == "anchor_desk"
    assert d["authority"] == AUTHORITY_LEVELS["stable_anchor"]


def test_spatial_track_to_dict():
    track = SpatialTrack(
        entity_id="obj_cup",
        label="cup",
        track_status="stable",
        position_room_m=(0.3, -0.1, 1.2),
        confidence=0.70,
        samples=5,
        stable_windows=3,
    )
    d = track.to_dict()
    assert d["track_status"] == "stable"
    assert d["samples"] == 5


def test_spatial_delta_to_dict():
    delta = SpatialDelta(
        delta_id="sdelta_123",
        entity_id="obj_cup",
        label="cup",
        delta_type="moved",
        from_position_m=(0.0, 0.0, 1.0),
        to_position_m=(0.5, 0.0, 1.0),
        distance_m=0.5,
        dominant_axis="x",
        confidence=0.85,
        validated=True,
        reason_codes=["stable_track"],
    )
    d = delta.to_dict()
    assert d["delta_type"] == "moved"
    assert d["distance_m"] == 0.5
    assert d["validated"] is True


def test_spatial_relation_to_dict():
    rel = SpatialRelationFact(
        relation_id="srel:0:1234",
        source_entity_id="obj_cup",
        relation_type="left_of",
        target_entity_id="anchor_desk",
        value_m=0.3,
        confidence=0.7,
    )
    d = rel.to_dict()
    assert d["relation_type"] == "left_of"
    assert d["value_m"] == 0.3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
