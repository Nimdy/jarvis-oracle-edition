"""Tests for the canonical scene-graph adapter (brain/cognition/spatial_scene_graph.py).

Inputs are constructed as canonical-shaped objects directly — the adapter
must never depend on a Pi detector path or on perception_orchestrator
internals.
"""

from __future__ import annotations

import os
import sys
from typing import get_args

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    SpatialAnchor,
    SpatialRelationType,
    SpatialTrack,
)
from cognition.spatial_scene_graph import (
    LATERAL_SEPARATION_MIN_M,
    MentalWorldEntity,
    MentalWorldRelation,
    MentalWorldSceneGraph,
    REASON_UNAVAILABLE,
    derive_scene_graph,
)
from perception.scene_types import SceneEntity, SceneSnapshot


# ---------------------------------------------------------------------------
# Fixture builders (canonical shapes only)
# ---------------------------------------------------------------------------


def _entity(
    entity_id: str,
    label: str,
    *,
    state: str = "visible",
    region: str = "desk_center",
    confidence: float = 0.85,
    last_seen_ts: float = 100.0,
    is_display: bool = False,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        confidence=confidence,
        permanence_confidence=confidence,
        bbox=(100, 100, 200, 200),
        region=region,
        state=state,  # type: ignore[arg-type]
        first_seen_ts=last_seen_ts - 10.0,
        last_seen_ts=last_seen_ts,
        unseen_cycles=0 if state in ("visible", "candidate") else 2,
        stable_cycles=4 if state == "visible" else 0,
        is_display_surface=is_display,
    )


def _track(
    entity_id: str,
    label: str,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> SpatialTrack:
    return SpatialTrack(
        entity_id=entity_id,
        label=label,
        track_status="stable",
        position_room_m=pos,
        velocity_mps=(0.0, 0.0, 0.0),
        dimensions_m=(0.1, 0.1, 0.1),
        uncertainty_m=0.05,
        confidence=0.85,
        samples=10,
        stable_windows=5,
        first_seen_ts=90.0,
        last_update_ts=100.0,
        anchor_id=None,
        authority=AUTHORITY_LEVELS["stable_track"],
    )


def _anchor(
    anchor_id: str,
    label: str,
    *,
    pos: tuple[float, float, float] = (0.0, 0.4, 0.6),
    dims: tuple[float, float, float] = (1.4, 0.04, 0.6),
) -> SpatialAnchor:
    return SpatialAnchor(
        anchor_id=anchor_id,
        anchor_type="desk_plane",
        label=label,
        position_room_m=pos,
        orientation_rpy=(0.0, 0.0, 0.0),
        dimensions_m=dims,
        confidence=0.9,
        stable_since_ts=80.0,
        last_verified_ts=100.0,
        calibration_version=1,
        authority=AUTHORITY_LEVELS["stable_anchor"],
    )


def _snapshot(
    entities: list[SceneEntity],
    *,
    update_count: int = 7,
    timestamp: float = 100.0,
) -> SceneSnapshot:
    return SceneSnapshot(
        timestamp=timestamp,
        entities=entities,
        deltas=[],
        display_surfaces=[],
        display_content=[],
        region_visibility={},
        update_count=update_count,
    )


# ---------------------------------------------------------------------------
# Empty-state contracts
# ---------------------------------------------------------------------------


def test_none_snapshot_returns_unavailable_reason():
    g = derive_scene_graph(None, tracks={}, anchors={})
    assert isinstance(g, MentalWorldSceneGraph)
    assert g.entity_count == 0
    assert g.relation_count == 0
    assert g.reason == REASON_UNAVAILABLE
    assert g.reason == "canonical_spatial_state_unavailable"


def test_empty_snapshot_returns_unavailable_reason():
    g = derive_scene_graph(_snapshot([]))
    assert g.reason == REASON_UNAVAILABLE
    assert g.entity_count == 0


def test_missing_tracks_and_anchors_still_produces_entities():
    snap = _snapshot([_entity("cup_0", "cup")])
    g = derive_scene_graph(snap)
    assert g.reason is None
    assert g.entity_count == 1
    assert g.entities[0].entity_id == "cup_0"
    # No track → no position_room_m available.
    assert g.entities[0].position_room_m is None


# ---------------------------------------------------------------------------
# Pairwise spatial relations
# ---------------------------------------------------------------------------


def test_left_right_derivation_two_entities():
    """A is at x=-0.5 (left), B is at x=+0.5 (right) → a left_of b."""
    snap = _snapshot([_entity("a", "cup"), _entity("b", "cup")])
    tracks = {
        "a": _track("a", "cup", pos=(-0.5, 0.7, 1.0)),
        "b": _track("b", "cup", pos=(+0.5, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    pairs = {(r.source_entity_id, r.target_entity_id, r.relation_type) for r in g.relations}
    assert ("a", "b", "left_of") in pairs
    # Should not also contain the reverse (one ordered emission per pair).
    assert ("b", "a", "left_of") not in pairs
    assert ("b", "a", "right_of") not in pairs


def test_below_lateral_threshold_no_left_right():
    """Tiny x delta (under LATERAL_SEPARATION_MIN_M) emits no left/right."""
    # |dx| = 2*delta must be < LATERAL_SEPARATION_MIN_M
    delta = LATERAL_SEPARATION_MIN_M / 4.0
    snap = _snapshot([_entity("a", "cup"), _entity("b", "cup")])
    tracks = {
        "a": _track("a", "cup", pos=(-delta, 0.7, 1.0)),
        "b": _track("b", "cup", pos=(+delta, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    rel_types = {r.relation_type for r in g.relations}
    assert "left_of" not in rel_types
    assert "right_of" not in rel_types


def test_in_front_of_and_behind_derivation():
    """A is closer to camera (z=0.5), B is further (z=1.5) → a in_front_of b."""
    snap = _snapshot([_entity("a", "cup"), _entity("b", "cup")])
    tracks = {
        "a": _track("a", "cup", pos=(0.0, 0.7, 0.5)),
        "b": _track("b", "cup", pos=(0.0, 0.7, 1.5)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    types = {(r.source_entity_id, r.target_entity_id, r.relation_type) for r in g.relations}
    assert ("a", "b", "in_front_of") in types
    assert ("a", "b", "behind") not in types


def test_near_relation_for_adjacent_objects():
    """Two cups within near radius should produce a `near` relation."""
    snap = _snapshot([_entity("c1", "cup"), _entity("c2", "cup")])
    tracks = {
        "c1": _track("c1", "cup", pos=(0.0, 0.7, 1.0)),
        "c2": _track("c2", "cup", pos=(0.05, 0.7, 1.05)),  # ~7 cm away
    }
    g = derive_scene_graph(snap, tracks=tracks)
    near_rels = [r for r in g.relations if r.relation_type == "near"]
    assert len(near_rels) == 1
    assert near_rels[0].value_m is not None
    assert near_rels[0].value_m < 0.10


def test_distant_objects_emit_no_near_relation():
    snap = _snapshot([_entity("c1", "cup"), _entity("c2", "cup")])
    tracks = {
        "c1": _track("c1", "cup", pos=(-1.0, 0.7, 1.0)),
        "c2": _track("c2", "cup", pos=(+1.0, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    rel_types = {r.relation_type for r in g.relations}
    assert "near" not in rel_types


def test_display_surface_skipped_in_pairwise():
    """Display surfaces never appear as pairwise sources/targets."""
    snap = _snapshot(
        [
            _entity("monitor_0", "monitor", is_display=True, region="monitor_zone"),
            _entity("cup_0", "cup", region="desk_center"),
        ]
    )
    tracks = {
        "monitor_0": _track("monitor_0", "monitor", pos=(0.0, 0.5, 0.6)),
        "cup_0": _track("cup_0", "cup", pos=(0.5, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    for r in g.relations:
        assert r.source_entity_id != "monitor_0", r
        assert r.target_entity_id != "monitor_0", r


# ---------------------------------------------------------------------------
# Derived state relations (out_of_view / occluded_by / last_seen_near)
# ---------------------------------------------------------------------------


def test_missing_entity_emits_out_of_view_self_edge():
    snap = _snapshot([_entity("c1", "cup", state="missing")])
    g = derive_scene_graph(snap)
    rels = [r for r in g.relations if r.relation_type == "out_of_view"]
    assert len(rels) == 1
    assert rels[0].source_entity_id == "c1"
    assert rels[0].target_entity_id == "c1"


def test_occluded_entity_emits_last_seen_near_to_visible_neighbor():
    """An occluded cup near a visible cup yields a last_seen_near edge."""
    snap = _snapshot(
        [
            _entity("hidden", "cup", state="occluded", region="desk_center"),
            _entity("seen", "cup", state="visible", region="desk_center"),
        ]
    )
    tracks = {
        "hidden": _track("hidden", "cup", pos=(0.0, 0.7, 1.0)),
        "seen": _track("seen", "cup", pos=(0.05, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    types = {(r.source_entity_id, r.target_entity_id, r.relation_type) for r in g.relations}
    assert ("hidden", "seen", "last_seen_near") in types
    assert ("hidden", "seen", "occluded_by") in types  # same region, occluded


def test_missing_entity_without_position_no_last_seen_near():
    snap = _snapshot([_entity("ghost", "cup", state="missing")])
    g = derive_scene_graph(snap)
    last_seen = [r for r in g.relations if r.relation_type == "last_seen_near"]
    assert last_seen == []


def test_removed_entity_is_history_not_active_spatial_relation():
    """Removed entities remain visible as history but do not create live edges."""
    snap = _snapshot(
        [
            _entity("active", "cup", state="visible"),
            _entity("gone", "cup", state="removed"),
        ]
    )
    tracks = {
        "active": _track("active", "cup", pos=(0.0, 0.7, 1.0)),
        "gone": _track("gone", "cup", pos=(0.5, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)

    assert g.entity_count == 2
    assert g.active_entity_count == 1
    assert g.removed_entity_count == 1
    assert any(e.entity_id == "gone" and e.state == "removed" for e in g.entities)

    live_edges = [
        r for r in g.relations
        if r.source_entity_id == "gone" or r.target_entity_id == "gone"
    ]
    assert [(r.source_entity_id, r.target_entity_id, r.relation_type) for r in live_edges] == [
        ("gone", "gone", "out_of_view")
    ]


# ---------------------------------------------------------------------------
# Anchor relations (centered_in / on)
# ---------------------------------------------------------------------------


def test_centered_in_anchor_relation():
    """A cup placed near the desk center yields a centered_in relation."""
    snap = _snapshot([_entity("cup_0", "cup")])
    tracks = {"cup_0": _track("cup_0", "cup", pos=(0.05, 0.45, 0.6))}
    anchors = {"desk_main": _anchor("desk_main", "desk")}
    g = derive_scene_graph(snap, tracks=tracks, anchors=anchors)
    types = [
        (r.source_entity_id, r.target_entity_id, r.relation_type)
        for r in g.relations
    ]
    assert ("cup_0", "desk_main", "centered_in") in types


def test_on_anchor_relation():
    """A cup sitting on the desk top surface yields an `on` relation."""
    snap = _snapshot([_entity("cup_0", "cup")])
    tracks = {"cup_0": _track("cup_0", "cup", pos=(0.0, 0.42, 0.6))}
    anchors = {"desk_main": _anchor("desk_main", "desk")}
    g = derive_scene_graph(snap, tracks=tracks, anchors=anchors)
    types = {
        (r.source_entity_id, r.target_entity_id, r.relation_type)
        for r in g.relations
    }
    assert ("cup_0", "desk_main", "on") in types


# ---------------------------------------------------------------------------
# Vocabulary invariants
# ---------------------------------------------------------------------------


def test_relation_types_within_canonical_or_derived_set():
    """Every emitted relation_type must be canonical or in the derived set."""
    canonical = set(get_args(SpatialRelationType))
    derived = {
        "facing", "moving_toward", "moving_away",
        "occluded_by", "last_seen_near",
        "expected_in_view", "out_of_view",
    }
    allowed = canonical | derived

    # Build a busy fixture that exercises many code paths.
    snap = _snapshot(
        [
            _entity("a", "cup", region="desk_left"),
            _entity("b", "cup", region="desk_right"),
            _entity("c", "cup", state="occluded", region="desk_left"),
            _entity("d", "cup", state="missing"),
        ]
    )
    tracks = {
        "a": _track("a", "cup", pos=(-0.3, 0.7, 1.0)),
        "b": _track("b", "cup", pos=(+0.3, 0.7, 1.5)),
        "c": _track("c", "cup", pos=(-0.31, 0.7, 1.0)),
        "d": _track("d", "cup", pos=(0.0, 0.7, 1.0)),
    }
    anchors = {"desk_main": _anchor("desk_main", "desk")}
    g = derive_scene_graph(snap, tracks=tracks, anchors=anchors)

    for r in g.relations:
        assert r.relation_type in allowed, f"unknown relation_type: {r.relation_type}"


# ---------------------------------------------------------------------------
# Adapter must never reach into private orchestrator state
# ---------------------------------------------------------------------------


def test_adapter_signature_takes_canonical_inputs_only():
    """The public function must accept canonical objects, not orchestrator handles.

    This is a structural test: derive_scene_graph's parameters are the
    canonical SceneSnapshot / SpatialTrack / SpatialAnchor types, not a
    perception_orchestrator instance.
    """
    import inspect
    sig = inspect.signature(derive_scene_graph)
    param_names = list(sig.parameters)
    assert param_names[0] == "scene_snapshot"
    assert "tracks" in param_names
    assert "anchors" in param_names
    # Parameter must not be named anything resembling perception orchestrator.
    for forbidden in ("perception_orchestrator", "perc_orch", "po"):
        assert forbidden not in param_names, f"adapter signature leaks {forbidden}"


def test_adapter_module_does_not_import_perception_orchestrator():
    """No adapter file should import the orchestrator (would create a cycle)."""
    import ast
    import cognition.spatial_scene_graph as mod

    with open(mod.__file__) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                assert "perception_orchestrator" not in n.name, n.name
        elif isinstance(node, ast.ImportFrom):
            assert "perception_orchestrator" not in (node.module or ""), node.module
            for n in node.names:
                assert "perception_orchestrator" not in n.name, n.name
            # Private members of perception are forbidden via from-imports.
            assert "_scene_tracker" not in (node.module or "")
            assert "_last_scene_snapshot" not in (node.module or "")


# ---------------------------------------------------------------------------
# Side-effect free
# ---------------------------------------------------------------------------


def test_derivation_does_not_mutate_inputs():
    snap = _snapshot([_entity("a", "cup"), _entity("b", "cup")])
    tracks = {
        "a": _track("a", "cup", pos=(-0.5, 0.7, 1.0)),
        "b": _track("b", "cup", pos=(+0.5, 0.7, 1.0)),
    }
    anchors = {"desk_main": _anchor("desk_main", "desk")}

    snap_repr_before = repr(snap)
    tracks_repr_before = repr(tracks)
    anchors_repr_before = repr(anchors)

    derive_scene_graph(snap, tracks=tracks, anchors=anchors)

    assert repr(snap) == snap_repr_before
    assert repr(tracks) == tracks_repr_before
    assert repr(anchors) == anchors_repr_before


def test_returned_graph_is_immutable():
    snap = _snapshot([_entity("a", "cup")])
    g = derive_scene_graph(snap)
    with pytest.raises(Exception):
        g.entities = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        g.entities[0].entity_id = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# to_dict shape
# ---------------------------------------------------------------------------


def test_to_dict_shape_has_no_raw_vector_keys():
    """The serialized form must never contain raw HRR vectors."""
    snap = _snapshot([_entity("a", "cup"), _entity("b", "cup")])
    tracks = {
        "a": _track("a", "cup", pos=(-0.5, 0.7, 1.0)),
        "b": _track("b", "cup", pos=(+0.5, 0.7, 1.0)),
    }
    g = derive_scene_graph(snap, tracks=tracks)
    payload = g.to_dict()
    flat = repr(payload)
    for forbidden in ("vector", "embedding", "raw_vector", "ndarray"):
        assert forbidden not in flat, f"to_dict leaked {forbidden}: {flat}"
    assert isinstance(payload["entity_count"], int)
    assert isinstance(payload["active_entity_count"], int)
    assert isinstance(payload["removed_entity_count"], int)
    assert isinstance(payload["relation_count"], int)
    assert payload["source"]["scene_update_count"] == snap.update_count
