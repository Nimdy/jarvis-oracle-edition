"""Tests for the HRR shadow world encoder (brain/cognition/hrr_world_encoder.py).

Pure-function tests: same input → same output, delta proportional to change,
no mutation of the input WorldState, no forbidden writer imports, flag-gated
ring buffer caps at its configured maximum.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from cognition.hrr_world_encoder import HRRWorldShadow, encode_world_state
from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldState,
)
from library.vsa.hrr import HRRConfig, similarity
from library.vsa.runtime_config import HRRRuntimeConfig
from library.vsa.symbols import SymbolDictionary


def _mk_ws(**overrides):
    """Build a WorldState with sensible non-default values we can perturb."""
    defaults = WorldState(
        physical=PhysicalState(person_count=1),
        user=UserState(
            present=True,
            engagement=0.72,
            emotion="calm",
            speaker_name="user",
        ),
        conversation=ConversationState(
            active=True,
            topic="weather",
            turn_count=2,
        ),
        system=SystemState(mode="active", health_score=0.92, active_goal_kind="none"),
        version=1,
        tick_number=1,
    )
    if not overrides:
        return defaults
    # WorldState is frozen, so swap fields via dataclasses.replace on its facets.
    kwargs = dict(
        physical=defaults.physical,
        user=defaults.user,
        conversation=defaults.conversation,
        system=defaults.system,
    )
    for key, val in overrides.items():
        kwargs[key] = val
    return replace(defaults, **kwargs)


def _cfg_and_dict(dim=512, seed=0):
    cfg = HRRConfig(dim=dim, seed=seed)
    d = SymbolDictionary(cfg)
    return cfg, d


# ---------------------------------------------------------------------------
# Pure encoder
# ---------------------------------------------------------------------------


def test_encode_world_state_deterministic():
    cfg, d = _cfg_and_dict()
    ws = _mk_ws()
    r1 = encode_world_state(ws, cfg, d)
    r2 = encode_world_state(ws, cfg, d)
    np.testing.assert_allclose(r1["vector"], r2["vector"])
    assert r1["facts_encoded"] == r2["facts_encoded"]
    assert r1["cleanup_accuracy"] == r2["cleanup_accuracy"]


def test_encode_world_state_cleanup_perfect_on_own_vocabulary():
    """Since we build the cleanup memory from the same snapshot we encoded,
    cleanup accuracy must be 1.0.
    """
    cfg, d = _cfg_and_dict()
    ws = _mk_ws()
    result = encode_world_state(ws, cfg, d)
    assert result["cleanup_accuracy"] == 1.0
    assert result["facts_encoded"] >= 8
    assert result["side_effects"] == 0


def test_encode_world_state_changes_with_input():
    cfg, d = _cfg_and_dict()
    ws_a = _mk_ws()
    ws_b = _mk_ws(
        user=UserState(present=False, engagement=0.1, emotion="sad", speaker_name="unknown")
    )
    r_a = encode_world_state(ws_a, cfg, d)
    r_b = encode_world_state(ws_b, cfg, d)
    # Different facts → different composites.
    assert not np.allclose(r_a["vector"], r_b["vector"])
    # But similar enough that cosine is nonzero (shared system/conversation state).
    sim = similarity(r_a["vector"], r_b["vector"])
    assert 0.0 < sim < 1.0


def test_encode_world_state_similarity_to_previous():
    cfg, d = _cfg_and_dict()
    ws_a = _mk_ws()
    ws_b = _mk_ws(
        user=UserState(present=True, engagement=0.75, emotion="calm", speaker_name="user")
    )
    r_a = encode_world_state(ws_a, cfg, d)
    r_b = encode_world_state(ws_b, cfg, d, prev_vector=r_a["vector"])
    assert r_b["similarity_to_previous"] is not None
    assert 0.5 <= r_b["similarity_to_previous"] <= 1.0


def test_encode_world_state_does_not_mutate_input():
    """WorldState is frozen, but also confirm the encoder isn't reaching into
    mutable facet attributes.
    """
    cfg, d = _cfg_and_dict()
    ws = _mk_ws()
    before_user = ws.user
    before_system = ws.system
    encode_world_state(ws, cfg, d)
    assert ws.user is before_user
    assert ws.system is before_system


def test_encode_empty_world_state_returns_zero_vector():
    cfg = HRRConfig(dim=128, seed=0)
    d = SymbolDictionary(cfg)

    class _EmptyWS:
        user = None
        conversation = None
        system = None
        physical = None
        uncertainty = None

    result = encode_world_state(_EmptyWS(), cfg, d)
    assert result["facts_encoded"] == 0
    assert result["cleanup_accuracy"] is None
    assert np.all(result["vector"] == 0.0)


# ---------------------------------------------------------------------------
# HRRWorldShadow ring-buffer owner
# ---------------------------------------------------------------------------


def test_shadow_never_samples_when_disabled():
    runtime = HRRRuntimeConfig(enabled=False, dim=128, sample_every_ticks=1)
    shadow = HRRWorldShadow(runtime)
    ws = _mk_ws()
    for _ in range(200):
        out = shadow.maybe_sample(ws)
        assert out is None
    s = shadow.status()
    assert s["enabled"] is False
    assert s["samples_total"] == 0
    assert s["samples_retained"] == 0


def test_shadow_respects_sample_every_ticks():
    runtime = HRRRuntimeConfig(enabled=True, dim=128, sample_every_ticks=10)
    shadow = HRRWorldShadow(runtime)
    ws = _mk_ws()
    fires = 0
    for _ in range(100):
        if shadow.maybe_sample(ws) is not None:
            fires += 1
    assert fires == 10
    s = shadow.status()
    assert s["samples_total"] == 10
    assert s["samples_retained"] == 10


def test_shadow_ring_buffer_caps_at_capacity():
    runtime = HRRRuntimeConfig(enabled=True, dim=128, sample_every_ticks=1)
    shadow = HRRWorldShadow(runtime)
    ws = _mk_ws()
    total = HRRWorldShadow.RING_CAPACITY + 50
    for _ in range(total):
        shadow.maybe_sample(ws)
    s = shadow.status()
    assert s["samples_total"] == total
    assert s["samples_retained"] == HRRWorldShadow.RING_CAPACITY
    assert s["ring_capacity"] == HRRWorldShadow.RING_CAPACITY


def test_shadow_status_surfaces_metrics_after_sample():
    runtime = HRRRuntimeConfig(enabled=True, dim=256, sample_every_ticks=1)
    shadow = HRRWorldShadow(runtime)
    ws = _mk_ws()
    shadow.maybe_sample(ws)
    s = shadow.status()
    assert s["enabled"] is True
    assert s["samples_total"] == 1
    assert s["samples_retained"] == 1
    assert s["cleanup_accuracy"] is not None


def test_shadow_none_world_state_is_noop():
    runtime = HRRRuntimeConfig(enabled=True, dim=128, sample_every_ticks=1)
    shadow = HRRWorldShadow(runtime)
    assert shadow.maybe_sample(None) is None
    assert shadow.status()["samples_total"] == 0


# ---------------------------------------------------------------------------
# Import-graph guard: encoder must not import forbidden writers
# ---------------------------------------------------------------------------


def test_encoder_source_has_no_forbidden_imports():
    import cognition.hrr_world_encoder as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from policy.state_encoder",
        "from policy.policy_nn",
        "from epistemic.belief_graph.bridge",
        "from memory.persistence",
        "from memory.storage",
        "from memory.canonical",
        "from autonomy",
        "from identity.kernel",
    )
    for token in forbidden:
        assert token not in src, f"encoder must not import {token!r}"
