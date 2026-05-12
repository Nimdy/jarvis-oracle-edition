"""Spatial fusion — scene entity + spatial track merge tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CONFIDENCE_THRESHOLD_STABLE,
    SpatialAnchor,
    SpatialTrack,
)
from cognition.spatial_fusion import SpatialFusion


def _scene_state(entities: list[dict] | None = None) -> dict:
    return {
        "entities": entities or [],
        "display_surfaces": [],
        "display_content": [],
        "region_visibility": {"desk_center": 1.0},
    }


def _make_track(
    eid: str = "obj_cup",
    label: str = "cup",
    status: str = "stable",
    pos: tuple[float, float, float] = (0.3, -0.1, 1.2),
    confidence: float = 0.75,
) -> SpatialTrack:
    return SpatialTrack(
        entity_id=eid,
        label=label,
        track_status=status,
        position_room_m=pos,
        confidence=confidence,
        samples=5,
        stable_windows=4,
        authority=AUTHORITY_LEVELS["stable_track"],
    )


def _make_anchor(
    aid: str = "anchor_desk",
    label: str = "desk",
    pos: tuple[float, float, float] = (0.0, -0.5, 1.0),
) -> SpatialAnchor:
    return SpatialAnchor(
        anchor_id=aid,
        anchor_type="desk_plane",
        label=label,
        position_room_m=pos,
        confidence=0.85,
        calibration_version=1,
        authority=AUTHORITY_LEVELS["stable_anchor"],
    )


# -- Basic fusion --


def test_fuse_empty():
    fusion = SpatialFusion()
    result = fusion.fuse(_scene_state(), {}, {})
    assert result["entities"] == []
    assert result["spatial_tracks"] == {}
    assert result["spatial_anchors"] == {}


def test_fuse_augments_entities_with_spatial():
    fusion = SpatialFusion()
    scene = _scene_state([{"entity_id": "obj_cup", "label": "cup"}])
    tracks = {"obj_cup": _make_track(confidence=0.75)}
    result = fusion.fuse(scene, tracks, {})
    aug_ent = result["entities"][0]
    assert "spatial" in aug_ent
    assert aug_ent["spatial"]["track_status"] == "stable"


def test_fuse_does_not_augment_low_confidence():
    fusion = SpatialFusion()
    scene = _scene_state([{"entity_id": "obj_cup", "label": "cup"}])
    tracks = {"obj_cup": _make_track(confidence=0.3)}
    result = fusion.fuse(scene, tracks, {})
    aug_ent = result["entities"][0]
    assert "spatial" not in aug_ent


def test_fuse_does_not_modify_input():
    fusion = SpatialFusion()
    original = [{"entity_id": "obj_cup", "label": "cup"}]
    scene = _scene_state(original)
    tracks = {"obj_cup": _make_track()}
    fusion.fuse(scene, tracks, {})
    assert "spatial" not in original[0]


def test_fuse_includes_spatial_tracks_and_anchors():
    fusion = SpatialFusion()
    tracks = {"obj_cup": _make_track()}
    anchors = {"anchor_desk": _make_anchor()}
    result = fusion.fuse(_scene_state(), tracks, anchors)
    assert "obj_cup" in result["spatial_tracks"]
    assert "anchor_desk" in result["spatial_anchors"]


# -- Relation derivation --


def test_derive_relations_from_stable_tracks():
    fusion = SpatialFusion()
    tracks = {"obj_cup": _make_track(pos=(-0.5, -0.1, 1.2))}
    anchors = {"anchor_desk": _make_anchor(pos=(0.0, -0.5, 1.0))}
    result = fusion.fuse(_scene_state(), tracks, anchors)
    assert len(result["spatial_relations"]) >= 1
    rel = result["spatial_relations"][0]
    assert rel["source_entity_id"] == "obj_cup"
    assert rel["target_entity_id"] == "anchor_desk"
    assert rel["relation_type"] in (
        "left_of", "right_of", "in_front_of", "behind", "near",
    )


def test_no_relations_from_provisional_tracks():
    fusion = SpatialFusion()
    tracks = {"obj_cup": _make_track(status="provisional")}
    anchors = {"anchor_desk": _make_anchor()}
    result = fusion.fuse(_scene_state(), tracks, anchors)
    assert len(result["spatial_relations"]) == 0


def test_no_relations_without_anchors():
    fusion = SpatialFusion()
    tracks = {"obj_cup": _make_track()}
    result = fusion.fuse(_scene_state(), tracks, {})
    assert len(result["spatial_relations"]) == 0


def test_relation_near_when_close():
    fusion = SpatialFusion()
    tracks = {"obj_cup": _make_track(pos=(0.05, -0.45, 1.05))}
    anchors = {"anchor_desk": _make_anchor(pos=(0.0, -0.5, 1.0))}
    result = fusion.fuse(_scene_state(), tracks, anchors)
    assert len(result["spatial_relations"]) >= 1
    assert result["spatial_relations"][0]["relation_type"] == "near"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
