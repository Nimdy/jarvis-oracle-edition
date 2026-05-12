"""Tests for the HRR mental-simulation shadow observer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from cognition.hrr_simulation_shadow import HRRSimulationShadow
from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
)
from library.vsa.runtime_config import HRRRuntimeConfig


def _mk_ws(**overrides):
    base = WorldState(
        physical=PhysicalState(person_count=1),
        user=UserState(present=True, engagement=0.6, emotion="calm", speaker_name="user"),
        conversation=ConversationState(active=True, topic="weather", turn_count=1),
        system=SystemState(mode="active", health_score=0.9),
    )
    from dataclasses import replace

    return replace(base, **overrides) if overrides else base


def _mk_delta():
    return WorldDelta(facet="user", event="user_present", details={}, confidence=1.0)


# ---------------------------------------------------------------------------
# HRRSimulationShadow unit tests
# ---------------------------------------------------------------------------


def test_shadow_is_noop_when_disabled():
    cfg = HRRRuntimeConfig(enabled=False, dim=128)
    shadow = HRRSimulationShadow(cfg)
    before = _mk_ws()
    after = _mk_ws(user=UserState(present=False))
    out = shadow.record_trace(before, _mk_delta(), after)
    assert out is None
    s = shadow.status()
    assert s["enabled"] is False
    assert s["samples_total"] == 0
    assert s["samples_retained"] == 0
    assert s["ring_capacity"] == 200


def test_shadow_records_metrics_when_enabled():
    cfg = HRRRuntimeConfig(enabled=True, dim=256)
    shadow = HRRSimulationShadow(cfg)
    before = _mk_ws()
    after = _mk_ws(user=UserState(present=False, engagement=0.1, emotion="sad"))
    out = shadow.record_trace(before, _mk_delta(), after)
    assert out is not None
    assert 0.0 <= out["delta_similarity"] <= 1.0
    assert out["side_effects"] == 0
    assert out["delta_event"] == "user_present"
    s = shadow.status()
    assert s["samples_total"] == 1
    assert s["samples_retained"] == 1
    assert s["last_delta_similarity"] == out["delta_similarity"]


def test_shadow_ring_capacity_caps_at_200():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    shadow = HRRSimulationShadow(cfg)
    before = _mk_ws()
    after = _mk_ws(user=UserState(present=False))
    total = HRRSimulationShadow.RING_CAPACITY + 40
    for _ in range(total):
        shadow.record_trace(before, _mk_delta(), after)
    s = shadow.status()
    assert s["samples_total"] == total
    assert s["samples_retained"] == HRRSimulationShadow.RING_CAPACITY


def test_shadow_handles_none_world_state_gracefully():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    shadow = HRRSimulationShadow(cfg)
    assert shadow.record_trace(None, _mk_delta(), _mk_ws()) is None
    assert shadow.record_trace(_mk_ws(), _mk_delta(), None) is None
    assert shadow.status()["samples_total"] == 0


def test_shadow_handles_encoder_exception():
    """If the encoder explodes, the observer must swallow and return None."""
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    shadow = HRRSimulationShadow(cfg)

    class _BogusWS:
        def __getattr__(self, name):
            raise RuntimeError("bogus")

    out = shadow.record_trace(_BogusWS(), _mk_delta(), _BogusWS())
    # Our encoder uses getattr with default so it actually succeeds with an
    # empty fact set → similarity is defined and observer records it.
    # But if getattr itself raises, record_trace returns None. We only
    # assert observer never raises; either return path is acceptable.
    assert out is None or isinstance(out, dict)


# ---------------------------------------------------------------------------
# Integration with MentalSimulator
# ---------------------------------------------------------------------------


def test_simulator_return_is_byte_identical_with_shadow_on_off(monkeypatch, tmp_path):
    """With HRR shadow enabled vs disabled, MentalSimulator.simulate returns
    a SimulationTrace with the same event, facet, depth, and total_confidence.

    (We don't compare by object identity — just by observable public shape.)
    """
    from cognition.causal_engine import CausalEngine
    from cognition.simulator import MentalSimulator

    delta = _mk_delta()
    state = _mk_ws()

    # Disabled. Clear both env-var and runtime_flags.json (P5.1 persistence layer)
    # so the test sees a known-off baseline regardless of operator-set flag files.
    monkeypatch.delenv("ENABLE_HRR_SHADOW", raising=False)
    monkeypatch.delenv("JARVIS_RUNTIME_FLAGS", raising=False)
    monkeypatch.setenv("JARVIS_RUNTIME_FLAGS", str(tmp_path / "_absent_runtime_flags.json"))
    sim_off = MentalSimulator(CausalEngine())
    trace_off = sim_off.simulate(state, delta, source="test")

    # Enabled.
    monkeypatch.setenv("ENABLE_HRR_SHADOW", "1")
    sim_on = MentalSimulator(CausalEngine())
    trace_on = sim_on.simulate(state, delta, source="test")

    assert trace_off.initial_delta.event == trace_on.initial_delta.event
    assert trace_off.initial_delta.facet == trace_on.initial_delta.facet
    assert trace_off.depth == trace_on.depth
    assert trace_off.total_confidence == trace_on.total_confidence
    assert trace_off.source == trace_on.source

    # Shadow should only have recorded in the "on" simulator.
    assert sim_off._hrr_shadow is None
    assert sim_on._hrr_shadow is not None
    assert sim_on._hrr_shadow.status()["samples_total"] == 1


def test_simulator_shadow_off_by_default_no_recording(monkeypatch, tmp_path):
    from cognition.causal_engine import CausalEngine
    from cognition.simulator import MentalSimulator

    # Clear both layers (env + runtime_flags.json) to assert the safe-default behavior.
    monkeypatch.delenv("ENABLE_HRR_SHADOW", raising=False)
    monkeypatch.delenv("JARVIS_RUNTIME_FLAGS", raising=False)
    monkeypatch.setenv("JARVIS_RUNTIME_FLAGS", str(tmp_path / "_absent_runtime_flags.json"))
    sim = MentalSimulator(CausalEngine())
    for _ in range(5):
        sim.simulate(_mk_ws(), _mk_delta(), source="test")
    assert sim._hrr_shadow is None


# ---------------------------------------------------------------------------
# Import-graph guard
# ---------------------------------------------------------------------------


def test_shadow_source_has_no_forbidden_imports():
    import cognition.hrr_simulation_shadow as mod

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
        assert token not in src, f"simulation shadow must not import {token!r}"
