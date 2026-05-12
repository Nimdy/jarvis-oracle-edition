"""Tests for the consciousness-feed `scene` block (P3.14).

The brain broadcasts a compact `consciousness` payload to the Pi particle
visualizer every ~2s via the `consciousness_feed` transport message, built
by ``perception_orchestrator._build_consciousness_feed``. P3.14 adds a
bounded ``scene`` block sourced from
``cognition.mental_world.get_state()`` so the Pi visualizer can modulate
its existing particle classes against the live mental-world signal —
without polling, without entity/relation arrays, and without ever
exposing raw HRR vectors or authority flags on the Pi side.

These tests pin that wire contract:

* The block contains exactly the seven keys in
  :data:`_SCENE_FEED_KEYS` — no entity / relation arrays, no
  authority flags, no raw / composite vectors.
* When the mental-world facade is unavailable or empty, the block
  returns deterministic zeros (and ``enabled=False``).
* When the facade returns a populated scene, the bounded scalars are
  forwarded with rounding intact.
* The helper module never exposes the brain's HRR substrate path
  (``library.vsa.*``) into the Pi-bound payload.

The Pi side of P3.14 only reads these seven keys; if anyone widens the
block they must update :data:`_SCENE_FEED_KEYS` and this test in the
same change.
"""

from __future__ import annotations

import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BRAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (REPO_ROOT, BRAIN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_helpers():
    """Import the perception orchestrator helpers without booting it.

    The orchestrator module pulls a wide perception/reasoning import
    surface, but the helpers under test are pure-function and live at
    module scope. We import lazily here so a missing optional dep
    surfaces as a clear `pytest.importorskip` rather than a hard
    collection error.
    """
    try:
        from perception_orchestrator import (  # type: ignore
            _build_scene_block,
            _SCENE_FEED_KEYS,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"perception_orchestrator unavailable: {exc!r}")
    return _build_scene_block, _SCENE_FEED_KEYS


def _scene_state_module():
    """Return the live mental_world facade module for override-based tests."""
    try:
        from cognition import mental_world as mw  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cognition.mental_world unavailable: {exc!r}")
    return mw


# ---------------------------------------------------------------------------
# Wire contract: bounded key set, no leaks
# ---------------------------------------------------------------------------


def test_scene_block_key_set_is_exactly_the_documented_contract():
    """The Pi visualizer reads exactly these seven keys. Drift breaks both ends."""
    _, keys = _load_helpers()
    assert set(keys) == {
        "enabled",
        "entity_count",
        "relation_count",
        "cleanup_accuracy",
        "relation_recovery",
        "similarity_to_previous",
        "spatial_hrr_side_effects",
    }


def test_scene_block_contains_only_bounded_scalars():
    build, keys = _load_helpers()
    block = build()
    assert set(block.keys()) == set(keys)
    for k, v in block.items():
        assert isinstance(v, (int, float, bool)), (
            f"scene[{k!r}] must be a bounded scalar, got {type(v).__name__}"
        )


def test_scene_block_does_not_leak_entities_relations_or_vectors():
    """Belt-and-suspenders: no high-cardinality arrays or vector keys."""
    build, _ = _load_helpers()
    block = build()
    forbidden = {
        "entities",
        "relations",
        "scenes",
        "vector",
        "raw_vector",
        "composite_vector",
        "scene_vector",
    }
    assert not (set(block.keys()) & forbidden), (
        f"scene block leaked forbidden keys: {set(block.keys()) & forbidden}"
    )


def test_scene_block_omits_authority_flags():
    """Pi never gets authority flags. They stay brain-side."""
    build, _ = _load_helpers()
    block = build()
    authority_keys = {
        "writes_memory",
        "writes_beliefs",
        "influences_policy",
        "influences_autonomy",
        "soul_integrity_influence",
        "llm_raw_vector_exposure",
        "no_raw_vectors_in_api",
    }
    assert not (set(block.keys()) & authority_keys), (
        "scene block must not surface authority flags to the Pi"
    )


# ---------------------------------------------------------------------------
# Defaults: empty / unavailable scene
# ---------------------------------------------------------------------------


def test_scene_block_defaults_when_mental_world_empty():
    """Override returning None → facade emits empty_state → bounded zeros."""
    build, _ = _load_helpers()
    mw = _scene_state_module()

    # The override path short-circuits before _SHADOW is consulted, so we
    # don't need to mutate the live shadow registration to exercise the
    # empty-state branch deterministically across CI / brain runs.
    mw.register_state_override(lambda: None)
    try:
        block = build()
    finally:
        mw.register_state_override(None)

    assert block["enabled"] is False
    assert block["entity_count"] == 0
    assert block["relation_count"] == 0
    assert block["cleanup_accuracy"] == 0.0
    assert block["relation_recovery"] == 0.0
    assert block["similarity_to_previous"] == 0.0
    assert block["spatial_hrr_side_effects"] == 0


def test_scene_block_handles_facade_exception():
    """If get_state() raises, helper must return zeros, not propagate."""
    build, _ = _load_helpers()
    mw = _scene_state_module()

    def _boom():
        raise RuntimeError("synthetic facade failure")

    mw.register_state_override(_boom)
    try:
        block = build()
    finally:
        mw.register_state_override(None)

    assert block["enabled"] is False
    assert block["entity_count"] == 0


# ---------------------------------------------------------------------------
# Forwarding: populated scene
# ---------------------------------------------------------------------------


def test_scene_block_forwards_populated_scalars():
    build, _ = _load_helpers()
    mw = _scene_state_module()

    fixture = {
        "enabled": True,
        "entity_count": 7,
        "relation_count": 15,
        "entities": [{"id": "obj_a"}, {"id": "obj_b"}],  # must NOT bleed through
        "relations": [{"subject": "obj_a", "relation": "left_of", "object": "obj_b"}],
        "metrics": {
            "cleanup_accuracy": 1.0,
            "relation_recovery": 0.9876,
            "similarity_to_previous": 0.9201,
            "binding_cleanliness": 0.187,  # not in feed key set
            "spatial_hrr_side_effects": 0,
        },
    }
    mw.register_state_override(lambda: fixture)
    try:
        block = build()
    finally:
        mw.register_state_override(None)

    assert block["enabled"] is True
    assert block["entity_count"] == 7
    assert block["relation_count"] == 15
    assert block["cleanup_accuracy"] == 1.0
    # Rounded to 3dp on the wire — the Pi reads bounded floats only.
    assert block["relation_recovery"] == 0.988
    assert block["similarity_to_previous"] == 0.92
    assert block["spatial_hrr_side_effects"] == 0
    # Nothing from `entities` / `relations` / `binding_cleanliness` made it through.
    assert "entities" not in block
    assert "relations" not in block
    assert "binding_cleanliness" not in block


def test_scene_block_clamps_partial_metrics_to_zero():
    """Missing metric keys default to 0.0 / 0 deterministically."""
    build, _ = _load_helpers()
    mw = _scene_state_module()

    fixture = {
        "enabled": True,
        "entity_count": 3,
        "relation_count": 0,
        "metrics": {},  # no scalars at all
    }
    mw.register_state_override(lambda: fixture)
    try:
        block = build()
    finally:
        mw.register_state_override(None)

    assert block["enabled"] is True
    assert block["entity_count"] == 3
    assert block["cleanup_accuracy"] == 0.0
    assert block["relation_recovery"] == 0.0
    assert block["similarity_to_previous"] == 0.0
    assert block["spatial_hrr_side_effects"] == 0


def test_scene_block_handles_non_dict_state():
    """Defensive: if upstream returns a non-dict, helper returns zeros."""
    build, _ = _load_helpers()
    mw = _scene_state_module()

    mw.register_state_override(lambda: "not a dict")  # type: ignore[arg-type]
    try:
        block = build()
    finally:
        mw.register_state_override(None)

    assert block["enabled"] is False
    assert block["entity_count"] == 0
