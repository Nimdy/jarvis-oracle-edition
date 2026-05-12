"""Tests for the P5 Commit 8 mental navigation shadow.

Covers:

* Purity: ``simulate_*`` never mutates its input graph.
* Correctness: each op flips the right states / regions.
* Trace fidelity: ``applied`` / ``reason`` / ``entity_deltas`` match the
  observed change.
* Ring: :class:`MentalNavigationShadow` is twin-gated and respects its
  capacity; disabled shadow is a no-op.
* Authority pins: AUTHORITY_FLAGS all False and ``no_raw_vectors_in_api``
  True in every emitted payload.
* Architecture: module does not import policy / belief / memory /
  autonomy / canonical-writer paths or raw HRR vector internals.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRAIN_ROOT = os.path.join(REPO_ROOT, "brain")
for p in (BRAIN_ROOT, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from cognition.mental_navigation import (  # noqa: E402
    ACTION_MOVE_FORWARD,
    ACTION_OBJECT_OCCLUDED,
    ACTION_RETURN_TO_LAST_SEEN,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    AUTHORITY_FLAGS,
    MentalNavigationShadow,
    MentalNavigationTrace,
    SUPPORTED_ACTIONS,
    simulate_move_forward,
    simulate_occlude,
    simulate_return_to_last_seen,
    simulate_turn_left,
    simulate_turn_right,
)
from cognition.spatial_scene_graph import (  # noqa: E402
    MentalWorldEntity,
    MentalWorldRelation,
    MentalWorldSceneGraph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entity(entity_id: str, *, region: str = "desk_center", state: str = "visible",
            label: str = "cup", confidence: float = 0.8) -> MentalWorldEntity:
    return MentalWorldEntity(
        entity_id=entity_id,
        label=label,
        state=state,
        region=region,
        position_room_m=None,
        confidence=confidence,
        last_seen_ts=0.0,
        is_display_surface=False,
    )


def _graph(entities, relations=()) -> MentalWorldSceneGraph:
    return MentalWorldSceneGraph(
        timestamp=1000.0,
        entities=tuple(entities),
        relations=tuple(relations),
        source_scene_update_count=1,
        source_track_count=len(entities),
        source_anchor_count=0,
        source_calibration_version=1,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Purity — no input mutation
# ---------------------------------------------------------------------------


def test_simulate_turn_left_does_not_mutate_input():
    g = _graph([_entity("a", region="desk_right"), _entity("b", region="desk_center")])
    before_snapshot = (
        tuple((e.entity_id, e.state, e.region) for e in g.entities),
        tuple(g.relations),
    )
    simulate_turn_left(g)
    after_snapshot = (
        tuple((e.entity_id, e.state, e.region) for e in g.entities),
        tuple(g.relations),
    )
    assert before_snapshot == after_snapshot


def test_graph_is_frozen_and_new_instance_returned():
    g = _graph([_entity("a", region="desk_right")])
    new_g, _ = simulate_turn_left(g)
    assert new_g is not g
    # Frozen dataclass — mutation should raise.
    with pytest.raises(Exception):
        new_g.entities[0].region = "somewhere"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Correctness — turn / move_forward
# ---------------------------------------------------------------------------


def test_turn_left_shifts_desk_right_to_desk_center():
    g = _graph([_entity("a", region="desk_right")])
    new_g, trace = simulate_turn_left(g)
    assert new_g.entities[0].region == "desk_center"
    assert trace.applied is True
    assert trace.action == ACTION_TURN_LEFT
    assert trace.reason is None
    assert len(trace.entity_deltas) == 1


def test_turn_left_drops_leftmost_out_of_view():
    g = _graph([_entity("a", region="desk_left")])
    new_g, trace = simulate_turn_left(g)
    assert new_g.entities[0].state == "out_of_view"
    assert trace.applied is True


def test_turn_right_mirror():
    g = _graph([_entity("a", region="desk_left")])
    new_g, trace = simulate_turn_right(g)
    assert new_g.entities[0].region == "desk_center"
    assert trace.action == ACTION_TURN_RIGHT


def test_turn_left_no_visible_entities_noop_with_reason():
    g = _graph([_entity("a", region="desk_right", state="occluded")])
    new_g, trace = simulate_turn_left(g)
    assert trace.applied is False
    assert trace.reason == "no_visible_entities_to_rotate"
    # Occluded entity passes through untouched.
    assert new_g.entities[0].state == "occluded"


def test_move_forward_brings_far_closer_and_drops_near():
    g = _graph([
        _entity("far",  region="desk_far"),
        _entity("near", region="desk_near"),
        _entity("mid",  region="desk_center"),
    ])
    new_g, trace = simulate_move_forward(g)
    regions = {e.entity_id: (e.region, e.state) for e in new_g.entities}
    assert regions["far"] == ("desk_center", "visible")
    assert regions["near"][1] == "out_of_view"
    assert regions["mid"] == ("desk_center", "visible")
    assert trace.action == ACTION_MOVE_FORWARD
    assert trace.applied is True


# ---------------------------------------------------------------------------
# Correctness — occlude / return_to_last_seen
# ---------------------------------------------------------------------------


def test_simulate_occlude_flips_state_and_optionally_adds_relation():
    g = _graph([_entity("a"), _entity("b")])
    new_g, trace = simulate_occlude(g, "a", occluder_entity_id="b")
    by_id = {e.entity_id: e for e in new_g.entities}
    assert by_id["a"].state == "occluded"
    assert by_id["b"].state == "visible"
    assert len(new_g.relations) == 1
    rel = new_g.relations[0]
    assert rel.relation_type == "occluded_by"
    assert rel.source_entity_id == "a"
    assert rel.target_entity_id == "b"
    assert 0 < rel.confidence < 1
    assert trace.applied is True
    assert trace.target_entity_id == "a"


def test_simulate_occlude_unknown_entity_is_noop():
    g = _graph([_entity("a")])
    new_g, trace = simulate_occlude(g, "zzz")
    assert new_g is g  # short-circuit returns same graph
    assert trace.applied is False
    assert trace.reason == "entity_id_not_in_graph"


def test_simulate_occlude_already_occluded_is_noop():
    g = _graph([_entity("a", state="occluded")])
    new_g, trace = simulate_occlude(g, "a")
    assert trace.applied is False
    assert trace.reason == "entity_already_occluded"
    assert new_g.entities[0].state == "occluded"


def test_return_to_last_seen_flips_missing_to_expected():
    g = _graph([_entity("a", state="missing", region="desk_left", confidence=0.9)])
    new_g, trace = simulate_return_to_last_seen(g, "a")
    e = new_g.entities[0]
    assert e.state == "expected_in_view"
    assert e.region == "desk_left"
    assert e.confidence == pytest.approx(0.45, abs=1e-6)
    assert trace.applied is True


def test_return_to_last_seen_visible_is_noop_with_reason():
    g = _graph([_entity("a", state="visible", region="desk_left")])
    new_g, trace = simulate_return_to_last_seen(g, "a")
    assert trace.applied is False
    assert trace.reason == "entity_not_missing_or_out_of_view"
    assert new_g.entities[0].state == "visible"


def test_return_to_last_seen_no_region_is_noop():
    g = _graph([_entity("a", state="missing", region="")])
    _, trace = simulate_return_to_last_seen(g, "a")
    assert trace.applied is False
    assert trace.reason == "no_remembered_region"


def test_return_to_last_seen_unknown_id_is_noop():
    g = _graph([_entity("a", state="missing", region="desk_left")])
    _, trace = simulate_return_to_last_seen(g, "zzz")
    assert trace.applied is False
    assert trace.reason == "entity_id_not_in_graph"


# ---------------------------------------------------------------------------
# Trace envelope — authority pins + no raw vectors
# ---------------------------------------------------------------------------


def test_trace_dict_contains_authority_pins():
    g = _graph([_entity("a", region="desk_right")])
    _, trace = simulate_turn_left(g)
    d = trace.to_dict()
    for k, v in AUTHORITY_FLAGS.items():
        assert d[k] == v
    # No raw vectors anywhere (authority flag names like
    # ``llm_raw_vector_exposure`` are allowed — we only forbid
    # *content* keys that expose the vector itself).
    forbidden_keys = {"vector", "raw_vector", "composite_vector", "ndarray"}
    for key in _walk_keys(d):
        assert key.lower() not in forbidden_keys


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


# ---------------------------------------------------------------------------
# MentalNavigationShadow ring
# ---------------------------------------------------------------------------


def test_shadow_disabled_is_noop():
    shadow = MentalNavigationShadow(capacity=8, enabled=False)
    trace = MentalNavigationTrace(
        action=ACTION_TURN_LEFT,
        timestamp=0.0,
        applied=True,
        reason=None,
        before={},
        after={},
        entity_deltas=(),
    )
    shadow.record(trace)
    shadow.record(trace)
    assert shadow.recent(10) == []
    status = shadow.status()
    assert status["enabled"] is False
    assert status["total_recorded"] == 0


def test_shadow_enabled_records_and_respects_capacity():
    shadow = MentalNavigationShadow(capacity=3, enabled=True)
    for i in range(5):
        shadow.record(MentalNavigationTrace(
            action=ACTION_TURN_LEFT,
            timestamp=float(i),
            applied=True,
            reason=None,
            before={}, after={},
            entity_deltas=(),
        ))
    assert len(shadow.recent(10)) == 3
    status = shadow.status()
    assert status["total_recorded"] == 5
    assert status["applied_count"] == 5
    assert status["last_timestamp"] == 4.0
    # Newest last.
    ts = [item["timestamp"] for item in shadow.recent(10)]
    assert ts == sorted(ts)
    assert ts[-1] == 4.0


def test_shadow_status_pins_authority_flags():
    shadow = MentalNavigationShadow(capacity=4, enabled=True)
    status = shadow.status()
    for k, v in AUTHORITY_FLAGS.items():
        assert status[k] == v
    assert status["status"] == "PRE-MATURE"
    assert status["lane"] == "spatial_hrr_mental_world"
    assert set(status["supported_actions"]) == set(SUPPORTED_ACTIONS)


def test_shadow_recent_is_clamped():
    shadow = MentalNavigationShadow(capacity=3, enabled=True)
    shadow.record(MentalNavigationTrace(
        action=ACTION_MOVE_FORWARD, timestamp=1.0,
        applied=True, reason=None, before={}, after={},
        entity_deltas=(),
    ))
    assert len(shadow.recent(100)) == 1  # clamped to capacity
    assert shadow.recent(0) == []


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


def test_module_has_no_forbidden_imports():
    """Mental navigation must never import policy / belief / memory /
    autonomy / identity writers, HRR vector internals, or the
    perception orchestrator."""
    src = os.path.join(BRAIN_ROOT, "cognition", "mental_navigation.py")
    with open(src, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    forbidden_prefixes = (
        "policy",
        "beliefs",
        "belief_engine",
        "memory",
        "autonomy",
        "identity",
        "perception_orchestrator",
    )
    forbidden_exact = {"library.vsa.hrr", "numpy", "np"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(
                mod == pref or mod.startswith(pref + ".") for pref in forbidden_prefixes
            ), f"mental_navigation imports forbidden module {mod!r}"
            assert mod not in forbidden_exact, (
                f"mental_navigation imports forbidden module {mod!r}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                assert not any(
                    name == pref or name.startswith(pref + ".") for pref in forbidden_prefixes
                ), f"mental_navigation imports forbidden module {name!r}"
                assert name not in forbidden_exact, (
                    f"mental_navigation imports forbidden module {name!r}"
                )
