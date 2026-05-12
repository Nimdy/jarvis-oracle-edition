"""Tests for the HRR spatial encoder (brain/cognition/hrr_spatial_encoder.py).

These tests verify:

* The encoder is pure (no mutation of inputs, no filesystem writes).
* Entity-state cleanup accuracy is perfect on small scenes at dim=1024.
* Relation recovery succeeds for the relation vocabulary used in the scene.
* ``spatial_hrr_side_effects`` stays at 0.
* The spatial shadow ring honors the P4 + P5 twin gate.
* No raw vectors leak into status / recent / scene payloads.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.hrr_spatial_encoder import HRRSpatialShadow, encode_scene_graph
from cognition.spatial_scene_graph import (
    MentalWorldEntity,
    MentalWorldRelation,
    MentalWorldSceneGraph,
    derive_scene_graph,
)
from cognition.spatial_schema import AUTHORITY_LEVELS, SpatialAnchor, SpatialTrack
from library.vsa.hrr import HRRConfig
from library.vsa.runtime_config import HRRRuntimeConfig
from library.vsa.symbols import SymbolDictionary
from perception.scene_types import SceneEntity, SceneSnapshot


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _cfg(dim: int = 1024, seed: int = 0) -> HRRConfig:
    return HRRConfig(dim=dim, seed=seed)


def _entity(
    entity_id: str,
    label: str,
    *,
    state: str = "visible",
    region: str = "desk_center",
    pos: tuple[float, float, float] | None = (0.0, 0.7, 1.0),
) -> MentalWorldEntity:
    return MentalWorldEntity(
        entity_id=entity_id,
        label=label,
        state=state,
        region=region,
        position_room_m=pos,
        confidence=0.85,
        last_seen_ts=100.0,
        is_display_surface=False,
    )


def _graph(
    entities: list[MentalWorldEntity],
    relations: list[MentalWorldRelation] | None = None,
) -> MentalWorldSceneGraph:
    return MentalWorldSceneGraph(
        timestamp=100.0,
        entities=tuple(entities),
        relations=tuple(relations or ()),
        source_scene_update_count=3,
        source_track_count=len(entities),
        source_anchor_count=0,
        source_calibration_version=1,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Encoder — pure-function contract
# ---------------------------------------------------------------------------


def test_encoder_on_empty_graph_returns_zero_vector():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    g = MentalWorldSceneGraph(
        timestamp=0.0, reason="canonical_spatial_state_unavailable"
    )
    out = encode_scene_graph(g, cfg, symbols)
    assert out["entities_encoded"] == 0
    assert out["relations_encoded"] == 0
    assert out["binding_cleanliness"] is None
    assert out["cleanup_accuracy"] is None
    assert out["relation_recovery"] is None
    assert out["cleanup_failures"] == 0
    assert out["side_effects"] == 0
    assert out["reason"] == "canonical_spatial_state_unavailable"
    v = out["vector"]
    assert isinstance(v, np.ndarray)
    assert v.shape == (cfg.dim,)
    assert float(np.linalg.norm(v)) == 0.0


def test_encoder_side_effects_zero_and_no_input_mutation():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    g = _graph([_entity("cup_0", "cup"), _entity("cup_1", "cup", state="occluded")])
    g_repr_before = repr(g)
    out = encode_scene_graph(g, cfg, symbols)
    assert out["side_effects"] == 0
    assert repr(g) == g_repr_before


def test_encoder_rejects_non_graph_input():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    with pytest.raises(TypeError):
        encode_scene_graph({"entities": []}, cfg, symbols)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cleanup quality at dim=1024
# ---------------------------------------------------------------------------


def test_entity_state_cleanup_accuracy_is_perfect_on_small_scene():
    """With a handful of entities the cleanup top-1 should always be correct."""
    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    ents = [
        _entity("cup_0", "cup", state="visible"),
        _entity("cup_1", "cup", state="occluded"),
        _entity("cup_2", "cup", state="missing"),
        _entity("monitor_0", "monitor", state="visible"),
    ]
    out = encode_scene_graph(_graph(ents), cfg, symbols)
    assert out["cleanup_accuracy"] == 1.0
    assert out["cleanup_failures"] == 0
    assert out["binding_cleanliness"] is not None
    assert out["binding_cleanliness"] >= 0.3  # at least meaningful separation


def test_relation_recovery_hits_expected_entity_set():
    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    ents = [_entity("a", "cup"), _entity("b", "cup"), _entity("c", "cup")]
    rels = [
        MentalWorldRelation(
            source_entity_id="a",
            target_entity_id="b",
            relation_type="left_of",
            value_m=0.5,
            confidence=0.9,
        ),
        MentalWorldRelation(
            source_entity_id="b",
            target_entity_id="c",
            relation_type="near",
            value_m=0.1,
            confidence=0.8,
        ),
    ]
    out = encode_scene_graph(_graph(ents, rels), cfg, symbols)
    assert out["entities_encoded"] == 3
    assert out["relations_encoded"] == 2
    assert out["relation_recovery"] is not None
    assert out["relation_recovery"] >= 0.5  # at least one relation's probe lands


def test_similarity_to_previous_is_clamped_to_unit_interval():
    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    g = _graph([_entity("a", "cup")])
    out1 = encode_scene_graph(g, cfg, symbols)
    out2 = encode_scene_graph(g, cfg, symbols, prev_vector=out1["vector"])
    assert out2["similarity_to_previous"] is not None
    assert -1.0 <= out2["similarity_to_previous"] <= 1.0


# ---------------------------------------------------------------------------
# Integration with derive_scene_graph
# ---------------------------------------------------------------------------


def _scene_entity(
    entity_id: str,
    label: str = "cup",
    state: str = "visible",
    region: str = "desk_center",
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        confidence=0.85,
        permanence_confidence=0.85,
        bbox=(100, 100, 200, 200),
        region=region,
        state=state,  # type: ignore[arg-type]
        first_seen_ts=90.0,
        last_seen_ts=100.0,
        unseen_cycles=0,
        stable_cycles=5,
        is_display_surface=False,
    )


def _track(
    entity_id: str,
    label: str,
    pos: tuple[float, float, float],
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


def test_end_to_end_canonical_snapshot_to_encoder():
    snap = SceneSnapshot(
        timestamp=100.0,
        entities=[
            _scene_entity("a", state="visible"),
            _scene_entity("b", state="visible"),
        ],
        deltas=[],
        display_surfaces=[],
        display_content=[],
        region_visibility={},
        update_count=11,
    )
    tracks = {
        "a": _track("a", "cup", (-0.5, 0.7, 1.0)),
        "b": _track("b", "cup", (+0.5, 0.7, 1.0)),
    }
    graph = derive_scene_graph(snap, tracks=tracks)
    assert graph.relation_count > 0

    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    out = encode_scene_graph(graph, cfg, symbols)
    assert out["entities_encoded"] == 2
    assert out["relations_encoded"] >= 1
    assert out["cleanup_accuracy"] == 1.0
    assert out["cleanup_failures"] == 0
    assert out["side_effects"] == 0


# ---------------------------------------------------------------------------
# Ring-buffered shadow (twin-gate)
# ---------------------------------------------------------------------------


def test_shadow_respects_p4_master_gate_off():
    """If ENABLE_HRR_SHADOW is off, sampling never runs even with P5 on."""
    runtime = HRRRuntimeConfig(enabled=False, spatial_scene_enabled=True)
    shadow = HRRSpatialShadow(runtime)
    assert shadow.enabled is False
    g = _graph([_entity("a", "cup")])
    assert shadow.maybe_sample(g) is None
    assert shadow.status()["samples_total"] == 0


def test_shadow_respects_p5_twin_gate_off():
    """If ENABLE_HRR_SPATIAL_SCENE is off, sampling never runs even with P4 on."""
    runtime = HRRRuntimeConfig(enabled=True, spatial_scene_enabled=False)
    shadow = HRRSpatialShadow(runtime)
    assert shadow.enabled is False
    g = _graph([_entity("a", "cup")])
    assert shadow.maybe_sample(g) is None


def test_shadow_samples_when_both_flags_enabled():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,  # sample every tick for the test
    )
    shadow = HRRSpatialShadow(runtime)
    assert shadow.enabled is True
    g = _graph([_entity("a", "cup"), _entity("b", "cup", state="occluded")])
    metrics = shadow.maybe_sample(g)
    assert metrics is not None
    assert metrics["entities_encoded"] == 2
    assert metrics["spatial_hrr_side_effects"] == 0
    assert shadow.status()["samples_total"] == 1


def test_shadow_honors_sample_every_ticks():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=3,
    )
    shadow = HRRSpatialShadow(runtime)
    g = _graph([_entity("a", "cup")])
    out1 = shadow.maybe_sample(g)
    out2 = shadow.maybe_sample(g)
    out3 = shadow.maybe_sample(g)
    out4 = shadow.maybe_sample(g)
    assert out1 is None
    assert out2 is None
    assert out3 is not None
    assert out4 is None
    assert shadow.status()["samples_total"] == 1


def test_shadow_ring_capacity_is_bounded():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    g = _graph([_entity("a", "cup")])
    for _ in range(shadow.RING_CAPACITY + 50):
        shadow.maybe_sample(g)
    status = shadow.status()
    assert status["samples_retained"] == shadow.RING_CAPACITY
    assert status["samples_total"] == shadow.RING_CAPACITY + 50


# ---------------------------------------------------------------------------
# No raw-vector leakage surface
# ---------------------------------------------------------------------------


def test_shadow_status_contains_no_vectors():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    shadow.maybe_sample(_graph([_entity("a", "cup")]))
    status = shadow.status()
    flat = repr(status)
    for bad in ("ndarray", "array(", "vector"):
        assert bad not in flat, f"status leaked {bad!r}: {status}"


def test_shadow_recent_contains_no_vectors():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    shadow.maybe_sample(_graph([_entity("a", "cup")]))
    recent = shadow.recent(20)
    flat = repr(recent)
    for bad in ("ndarray", "array("):
        assert bad not in flat, f"recent leaked {bad!r}"


def test_shadow_scene_payload_strips_vectors_but_keeps_metrics():
    runtime = HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        spatial_scene_sample_every_ticks=1,
    )
    shadow = HRRSpatialShadow(runtime)
    shadow.maybe_sample(_graph([_entity("a", "cup"), _entity("b", "cup")]))
    payload = shadow.latest_scene_payload()
    assert payload is not None
    assert "entities" in payload
    assert "metrics" in payload
    assert "entities_encoded" in payload["metrics"]
    assert "vector" not in repr(payload)
    assert "ndarray" not in repr(payload)


# ---------------------------------------------------------------------------
# Twin-gate property test
# ---------------------------------------------------------------------------


def test_runtime_spatial_scene_active_property():
    assert HRRRuntimeConfig(enabled=False, spatial_scene_enabled=False).spatial_scene_active is False
    assert HRRRuntimeConfig(enabled=True, spatial_scene_enabled=False).spatial_scene_active is False
    assert HRRRuntimeConfig(enabled=False, spatial_scene_enabled=True).spatial_scene_active is False
    assert HRRRuntimeConfig(enabled=True, spatial_scene_enabled=True).spatial_scene_active is True
