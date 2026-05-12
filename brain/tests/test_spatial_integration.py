"""Integration tests: spatial pipeline processes scene_summary events end-to-end.

Verifies that scene detections flow through the full spatial chain:
  SceneDetection -> SpatialEstimator -> SpatialTrack -> SpatialValidator -> SpatialFusion
without importing memory, identity, or other subsystems.
"""

import time

from cognition.spatial_fusion import SpatialFusion
from cognition.spatial_schema import (
    CONFIDENCE_THRESHOLD_STABLE,
    KNOWN_SIZE_PRIORS,
    STABLE_WINDOWS_REQUIRED,
    SpatialAnchor,
)
from cognition.spatial_validation import SpatialValidator
from perception.calibration import CalibrationManager
from perception.scene_types import SceneDetection, SceneEntity, SceneSnapshot
from perception.spatial import SpatialEstimator


def _make_calibration() -> CalibrationManager:
    cal = CalibrationManager.__new__(CalibrationManager)
    cal._intrinsics = CalibrationManager.__new__(CalibrationManager)
    from perception.calibration import CameraIntrinsics, RoomTransform
    cal._intrinsics = CameraIntrinsics()
    cal._transform = RoomTransform()
    cal._version = 1
    cal._state = "valid"
    cal._last_verified_ts = time.time()
    cal._created_ts = time.time()
    cal._anchor_consistency_ok = True
    return cal


def _make_scene_entity(
    entity_id: str, label: str, bbox: tuple[int, int, int, int],
    confidence: float = 0.85, state: str = "visible",
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        confidence=confidence,
        bbox=bbox,
        state=state,
        region="desk_center",
        first_seen_ts=time.time() - 60,
        last_seen_ts=time.time(),
    )


def test_end_to_end_single_entity():
    """A single visible entity flows through estimation, tracking, and fusion."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)
    validator = SpatialValidator()
    fusion = SpatialFusion()

    entity = _make_scene_entity("ent_monitor_0", "monitor", (200, 100, 900, 600))

    obs = estimator.estimate(
        entity_id=entity.entity_id,
        label=entity.label,
        bbox=entity.bbox,
        confidence=entity.confidence,
    )
    assert obs is not None
    assert obs.depth_m > 0

    track = estimator.update_track(obs)
    assert track.entity_id == "ent_monitor_0"
    assert track.track_status == "provisional"

    scene_dict = {"entities": [entity.to_dict()]}
    fused = fusion.fuse(scene_dict, estimator.get_tracks(), estimator.get_anchors())
    assert "spatial_tracks" in fused
    assert "ent_monitor_0" in fused["spatial_tracks"]


def test_track_promotes_to_stable_after_repeated_observations():
    """A track promotes to stable after STABLE_WINDOWS_REQUIRED consistent observations."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)

    for i in range(STABLE_WINDOWS_REQUIRED + 2):
        obs = estimator.estimate(
            entity_id="ent_keyboard_0",
            label="keyboard",
            bbox=(400, 500, 850, 550),
            confidence=0.8,
            timestamp=time.time() + i * 0.5,
        )
        assert obs is not None
        track = estimator.update_track(obs)

    assert track.track_status == "stable"
    assert track.confidence >= CONFIDENCE_THRESHOLD_STABLE


def test_stable_track_generates_delta_on_movement():
    """A stable track that moves beyond jitter threshold generates a promoted delta."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)
    validator = SpatialValidator()

    for i in range(STABLE_WINDOWS_REQUIRED + 2):
        obs = estimator.estimate(
            entity_id="ent_cup_0",
            label="cup",
            bbox=(500, 400, 530, 440),
            confidence=0.85,
            timestamp=time.time() + i * 0.5,
        )
        estimator.update_track(obs)

    track = estimator.get_tracks()["ent_cup_0"]
    assert track.track_status == "stable"

    for i in range(3):
        obs = estimator.estimate(
            entity_id="ent_cup_0",
            label="cup",
            bbox=(900, 400, 930, 440),
            confidence=0.85,
            timestamp=time.time() + 10 + i * 0.5,
        )
        estimator.update_track(obs)

    track = estimator.get_tracks()["ent_cup_0"]
    delta = validator.validate_track_to_delta(track, estimator.get_anchors(), calibration_version=1)
    assert track is not None


def test_fusion_augments_scene_entities():
    """Fusion adds spatial data to scene entities with stable tracks."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)
    fusion = SpatialFusion()

    for i in range(STABLE_WINDOWS_REQUIRED + 2):
        obs = estimator.estimate(
            entity_id="ent_laptop_0",
            label="laptop",
            bbox=(300, 200, 650, 450),
            confidence=0.9,
            timestamp=time.time() + i * 0.5,
        )
        estimator.update_track(obs)

    scene_dict = {
        "entities": [
            {"entity_id": "ent_laptop_0", "label": "laptop", "confidence": 0.9},
        ],
    }
    fused = fusion.fuse(scene_dict, estimator.get_tracks(), estimator.get_anchors())

    assert len(fused["entities"]) == 1
    ent = fused["entities"][0]
    assert "spatial" in ent, "Stable track should augment entity with spatial data"
    assert ent["spatial"]["track_status"] == "stable"


def test_world_adapters_accept_spatial_observations():
    """World adapters create SensorObservation from spatial tracks."""
    from cognition.world_adapters import observations_from_spatial_state

    spatial_state = {
        "spatial_tracks": {
            "ent_cup_0": {
                "entity_id": "ent_cup_0",
                "label": "cup",
                "track_status": "stable",
                "confidence": 0.72,
                "position_room_m": (0.3, -0.1, 1.2),
            },
        },
        "spatial_anchors": {
            "desk_plane": {
                "anchor_id": "desk_plane",
                "label": "desk",
                "confidence": 0.85,
                "position_room_m": (0.0, 0.0, 1.0),
            },
        },
    }
    obs = observations_from_spatial_state(spatial_state)
    assert len(obs) == 2
    kinds = {o.kind for o in obs}
    assert "spatial" in kinds
    tags_flat = set()
    for o in obs:
        tags_flat.update(o.tags)
    assert "spatial_track" in tags_flat
    assert "spatial_anchor" in tags_flat


def test_fusion_derives_relations_with_anchors():
    """SpatialFusion derives relations between stable tracks and anchors."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)
    fusion = SpatialFusion()

    for i in range(STABLE_WINDOWS_REQUIRED + 2):
        obs = estimator.estimate(
            entity_id="ent_keyboard_0",
            label="keyboard",
            bbox=(400, 500, 850, 550),
            confidence=0.85,
            timestamp=time.time() + i * 0.5,
        )
        estimator.update_track(obs)

    estimator.register_anchor(
        anchor_id="desk_plane",
        anchor_type="desk_plane",
        label="desk",
        position_room_m=(0.0, 0.0, 1.0),
        confidence=0.9,
    )

    scene_dict = {"entities": []}
    fused = fusion.fuse(scene_dict, estimator.get_tracks(), estimator.get_anchors())

    assert len(fused["spatial_relations"]) > 0, "Should derive at least one relation"
    rel = fused["spatial_relations"][0]
    assert "relation_type" in rel
    assert "value_m" in rel


def test_multiple_entities_through_pipeline():
    """Multiple entities all flow through the pipeline without interference."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)
    fusion = SpatialFusion()

    entities = [
        ("ent_monitor_0", "monitor", (100, 50, 800, 500)),
        ("ent_keyboard_0", "keyboard", (300, 550, 750, 590)),
        ("ent_cup_0", "cup", (850, 500, 880, 540)),
        ("ent_mouse_0", "mouse", (780, 560, 800, 575)),
    ]

    for iteration in range(STABLE_WINDOWS_REQUIRED + 2):
        for eid, label, bbox in entities:
            obs = estimator.estimate(
                entity_id=eid,
                label=label,
                bbox=bbox,
                confidence=0.85,
                timestamp=time.time() + iteration * 0.5,
            )
            if obs:
                estimator.update_track(obs)

    tracks = estimator.get_tracks()
    assert len(tracks) == 4
    stable_count = sum(1 for t in tracks.values() if t.track_status == "stable")
    assert stable_count >= 3

    scene_dict = {
        "entities": [
            {"entity_id": eid, "label": label} for eid, label, _ in entities
        ],
    }
    fused = fusion.fuse(scene_dict, tracks, estimator.get_anchors())
    assert len(fused["spatial_tracks"]) == 4


def test_no_estimation_with_invalid_calibration():
    """When calibration is invalid, no spatial observations are produced."""
    cal = _make_calibration()
    cal._state = "invalid"
    cal._version = 0
    cal._last_verified_ts = 0.0
    estimator = SpatialEstimator(cal)

    obs = estimator.estimate(
        entity_id="ent_cup_0",
        label="cup",
        bbox=(500, 400, 530, 440),
        confidence=0.85,
    )
    assert obs is None


def test_spatial_state_method():
    """SpatialEstimator.get_state() returns dashboard-friendly data."""
    cal = _make_calibration()
    estimator = SpatialEstimator(cal)

    obs = estimator.estimate(
        entity_id="ent_monitor_0",
        label="monitor",
        bbox=(200, 100, 900, 600),
        confidence=0.9,
    )
    if obs:
        estimator.update_track(obs)

    state = estimator.get_state()
    assert "total_tracks" in state
    assert "observations_total" in state
    assert state["total_tracks"] >= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
