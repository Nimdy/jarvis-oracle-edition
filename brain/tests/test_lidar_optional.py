"""JARVIS must run IDENTICALLY without lidar — it's a bonus sense, never a crutch.

A user with no lidar (or a different lidar brand) must get the same camera-only
cognition. This pins the contract so the camera↔lidar fusion (PR4) can never
sneak in a hard dependency: no lidar input ⇒ the scene graph, world model, and
dashboards behave exactly as if lidar had never existed.

(Lidar is also brand-agnostic: the brain consumes a generic `scan_2d` event with
`points_polar = [[bearing_rad, range_m], …]`. Every S2-specific detail lives in
pi/senses/lidar/lidar_sensor.py, so a different lidar is a drop-in Pi adapter —
the brain doesn't change.)
"""
from __future__ import annotations

import pytest

try:
    from perception.scene_types import SceneEntity, SceneSnapshot
    from cognition.spatial_scene_graph import derive_scene_graph
    _SG = True
except Exception:  # pragma: no cover - heavy deps absent
    _SG = False


def _entity(eid: str, label: str, *, state: str = "visible", region: str = "desk_center",
            conf: float = 0.85, ts: float = 100.0) -> "SceneEntity":
    return SceneEntity(
        entity_id=eid, label=label, confidence=conf, permanence_confidence=conf,
        bbox=(100, 100, 200, 200), region=region, state=state,  # type: ignore[arg-type]
        first_seen_ts=ts - 10.0, last_seen_ts=ts,
        unseen_cycles=0 if state in ("visible", "candidate") else 2,
        stable_cycles=4 if state == "visible" else 0, is_display_surface=False)


def _snapshot(entities):
    return SceneSnapshot(timestamp=100.0, entities=entities, deltas=[],
                         display_surfaces=[], display_content=[], region_visibility={})


@pytest.mark.skipif(not _SG, reason="scene graph deps unavailable")
def test_scene_graph_identical_with_or_without_lidar():
    """No lidar (anchors=None) must equal empty-lidar (anchors={}) AND still build a
    real camera-only scene — proving the fusion seam never REQUIRES lidar anchors."""
    snap = _snapshot([_entity("e1", "cup"), _entity("e2", "keyboard", region="desk_left")])
    g_none = derive_scene_graph(snap, tracks={}, anchors=None)    # no lidar at all
    g_empty = derive_scene_graph(snap, tracks={}, anchors={})     # lidar present, no anchors
    assert g_none.to_dict() == g_empty.to_dict()                  # byte-identical
    assert len(g_none.entities) == 2                              # the camera scene stands alone
    assert g_none.reason is None


@pytest.mark.skipif(not _SG, reason="scene graph deps unavailable")
def test_scene_graph_no_camera_no_lidar_is_safe_empty():
    """No camera entities AND no lidar ⇒ honest empty graph, never a crash."""
    assert derive_scene_graph(None, tracks={}, anchors=None).entities == ()
    assert derive_scene_graph(_snapshot([]), tracks={}, anchors=None).entities == ()


def test_pi5_devices_lidar_absent_is_graceful():
    """With no lidar in the cache, the body view shows it ABSENT — never errors."""
    from dashboard.pi5_devices import derive_pi5_devices
    devs = derive_pi5_devices({})                                 # empty cache = no lidar
    lidar = [d for d in devs if d.get("kind") == "lidar"]
    assert lidar and lidar[0]["present"] is False and lidar[0]["status"] == "absent"


def test_room_model_and_extrinsic_inert_without_data():
    """No scans ⇒ room model is empty-by-design; no mount ⇒ the extrinsic is identity."""
    from cognition.lidar_room import LidarRoomModel, REASON_INSUFFICIENT_COVERAGE
    from cognition.lidar_calibration import LidarExtrinsic
    rm = LidarRoomModel().room_model()
    assert rm.reason == REASON_INSUFFICIENT_COVERAGE and rm.walls == ()
    assert LidarExtrinsic().is_identity                            # no mount configured ⇒ no-op


def test_surface_accessor_empty_without_sensors():
    """The brain's room-model accessor returns {} when no lidar ever connected."""
    try:
        from perception.server import PerceptionServer
    except Exception:
        pytest.skip("perception.server import unavailable")
    from types import SimpleNamespace
    assert PerceptionServer.get_lidar_room_model(SimpleNamespace(_lidar_rooms={})) == {}
