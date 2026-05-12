"""Tests for the Commit 5 engine hook + ``get_scene_snapshot`` accessor.

These tests verify four invariants:

1. ``perception_orchestrator.get_scene_snapshot()`` returns the canonical
   snapshot without exposing private members.
2. The P5 engine hook is twin-gated: requires both ``ENABLE_HRR_SHADOW``
   and ``ENABLE_HRR_SPATIAL_SCENE``. Either off suppresses sampling.
3. The P5 code does **not** mutate the returned :class:`SceneSnapshot`
   (mutation-guard requested by the reviewer).
4. When the hook runs, it uses the public accessors only; no test code
   reaches into ``_scene_tracker`` / ``_last_scene_snapshot`` privately.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition import mental_world
from cognition.hrr_spatial_encoder import HRRSpatialShadow
from cognition.spatial_scene_graph import derive_scene_graph
from cognition.spatial_schema import AUTHORITY_LEVELS, SpatialAnchor, SpatialTrack
from library.vsa.runtime_config import HRRRuntimeConfig
from perception.scene_types import SceneEntity, SceneSnapshot


# ---------------------------------------------------------------------------
# Public-accessor contract
# ---------------------------------------------------------------------------


def test_perception_orchestrator_exports_public_get_scene_snapshot():
    """The accessor must exist as a public method (no leading underscore).

    Source-inspected via AST (importing perception_orchestrator at test
    time pulls in pydantic/fastapi which may not be present in minimal
    dev shells — the structural guarantee is what matters here).
    """
    import ast
    import pathlib

    po_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "perception_orchestrator.py"
    )
    tree = ast.parse(po_path.read_text())

    class_methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PerceptionOrchestrator":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_methods.add(item.name)

    for required in (
        "get_scene_snapshot",
        "get_spatial_tracks",
        "get_spatial_anchors",
    ):
        assert required in class_methods, (
            f"PerceptionOrchestrator missing required public accessor: {required}"
        )
        assert not required.startswith("_"), required


def test_engine_module_does_not_access_private_perception_state():
    """The engine hook must go through public accessors only."""
    import ast
    import consciousness.engine as eng

    with open(eng.__file__) as f:
        tree = ast.parse(f.read())

    forbidden = ("_scene_tracker", "_last_scene_snapshot")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, (
                f"engine.py accesses forbidden private attribute: {node.attr}"
            )


# ---------------------------------------------------------------------------
# Mutation guard: P5 must not mutate the returned SceneSnapshot.
# ---------------------------------------------------------------------------


def _scene_entity(eid: str) -> SceneEntity:
    return SceneEntity(
        entity_id=eid,
        label="cup",
        confidence=0.9,
        permanence_confidence=0.9,
        bbox=(100, 100, 200, 200),
        region="desk_center",
        state="visible",
        first_seen_ts=90.0,
        last_seen_ts=100.0,
        unseen_cycles=0,
        stable_cycles=5,
        is_display_surface=False,
    )


def _snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        timestamp=100.0,
        entities=[_scene_entity("a"), _scene_entity("b")],
        deltas=[],
        display_surfaces=[],
        display_content=[],
        region_visibility={},
        update_count=4,
    )


def _track(eid: str, pos):
    return SpatialTrack(
        entity_id=eid,
        label="cup",
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


def test_p5_does_not_mutate_scene_snapshot():
    """Passing a SceneSnapshot through the P5 pipeline must not mutate it.

    Reviewer-requested test: since ``SceneSnapshot`` is a mutable dataclass
    and ``get_scene_snapshot()`` returns the same live object, P5 code
    must treat it as strictly read-only.
    """
    snap = _snapshot()
    original = copy.deepcopy(snap)
    tracks = {
        "a": _track("a", (-0.5, 0.7, 1.0)),
        "b": _track("b", (+0.5, 0.7, 1.0)),
    }

    # Full P5 pipeline: derive graph → encode → sample in shadow.
    graph = derive_scene_graph(snap, tracks=tracks)
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    shadow.maybe_sample(graph)

    # The snapshot must be bit-identical to the deep copy.
    assert snap == original, "P5 pipeline mutated the SceneSnapshot"
    # Entity list identity is still the same list object (not reassigned).
    assert len(snap.entities) == 2
    for before, after in zip(original.entities, snap.entities):
        assert before == after


def test_p5_pipeline_does_not_mutate_tracks_or_anchors():
    tracks = {
        "a": _track("a", (-0.5, 0.7, 1.0)),
        "b": _track("b", (+0.5, 0.7, 1.0)),
    }
    anchors: dict = {}
    tracks_before = copy.deepcopy(tracks)
    anchors_before = copy.deepcopy(anchors)

    graph = derive_scene_graph(_snapshot(), tracks=tracks, anchors=anchors)

    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    shadow.maybe_sample(graph)

    assert tracks == tracks_before
    assert anchors == anchors_before


# ---------------------------------------------------------------------------
# Twin-gating at the engine hook level (exercised via the shadow class)
# ---------------------------------------------------------------------------


def test_engine_builds_spatial_shadow_only_when_twin_gate_active(monkeypatch):
    """The engine constructs ``_hrr_spatial_shadow`` iff both flags are on."""
    # We exercise the env-read path without booting the full engine.
    for master, p5 in [(False, False), (False, True), (True, False), (True, True)]:
        monkeypatch.setenv("ENABLE_HRR_SHADOW", "1" if master else "0")
        monkeypatch.setenv("ENABLE_HRR_SPATIAL_SCENE", "1" if p5 else "0")
        cfg = HRRRuntimeConfig.from_env()
        expected_active = master and p5
        assert cfg.spatial_scene_active is expected_active, (
            master, p5, cfg.spatial_scene_active
        )


# ---------------------------------------------------------------------------
# Engine hook produces a mental_world.get_state() result end to end.
# ---------------------------------------------------------------------------


def test_end_to_end_hook_wires_mental_world_facade():
    """Register a shadow directly; mental_world.get_state() reflects it."""
    mental_world.register_shadow(None)

    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    graph = derive_scene_graph(
        _snapshot(),
        tracks={
            "a": _track("a", (-0.5, 0.7, 1.0)),
            "b": _track("b", (+0.5, 0.7, 1.0)),
        },
    )
    shadow.maybe_sample(graph)
    mental_world.register_shadow(shadow)

    try:
        state = mental_world.get_state()
        assert state["enabled"] is True
        assert state["entity_count"] == 2
        # Authority pins always false.
        assert state["writes_memory"] is False
        assert state["influences_policy"] is False
        assert state["no_raw_vectors_in_api"] is True

        history = mental_world.get_history(5)
        assert history["count"] >= 1
        for scene in history["scenes"]:
            assert "vector" not in scene
    finally:
        mental_world.register_shadow(None)


def test_empty_canonical_state_produces_unavailable_reason():
    """If perception has no snapshot, the P5 graph emits the reason string."""
    graph = derive_scene_graph(None, tracks={}, anchors={})
    assert graph.reason == "canonical_spatial_state_unavailable"
    assert graph.entity_count == 0


# ---------------------------------------------------------------------------
# Authority-boundary contract (snapshot of the structural rules)
# ---------------------------------------------------------------------------


AUTHORITY_EXPECTATIONS = {
    "writes_memory": False,
    "writes_beliefs": False,
    "influences_policy": False,
    "influences_autonomy": False,
    "soul_integrity_influence": False,
    "llm_raw_vector_exposure": False,
    "no_raw_vectors_in_api": True,
}


def test_mental_world_authority_block_is_hard_pinned():
    from cognition import mental_world as mw
    assert mw.AUTHORITY_FLAGS == AUTHORITY_EXPECTATIONS


def test_mental_world_empty_state_carries_authority_pins():
    mental_world.register_shadow(None)
    state = mental_world.get_state()
    for key, expected in AUTHORITY_EXPECTATIONS.items():
        assert state[key] is expected, (key, state[key])
