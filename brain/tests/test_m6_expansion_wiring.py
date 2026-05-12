"""Regression tests for the two P3.5 Matrix-unblock fixes.

1. `ShadowPolicyRunner.set_hemisphere_signals()` forwards to the shadow
   encoder so M6 A/B evaluation sees slots 4-5 instead of zero-padding.
2. `_check_specialist_promotions` stamps `specialist_verification_ts` at
   the `PROBATIONARY_TRAINING -> VERIFIED_PROBATIONARY` transition, which
   unblocks `_check_expansion_trigger()`.

These tests anchor two real bugs surfaced by the 2026-04-24 TODO_V2 trace
validation. Without them the Matrix / Tier-2 promotion ladder is dead-ended.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Fix 1: Shadow encoder receives hemisphere signals
# ---------------------------------------------------------------------------


def test_shadow_runner_exposes_set_hemisphere_signals():
    """Confirm the method exists on ShadowPolicyRunner (reachable from engine)."""
    from policy.shadow_runner import ShadowPolicyRunner

    runner = ShadowPolicyRunner()
    assert hasattr(runner, "set_hemisphere_signals")
    assert callable(runner.set_hemisphere_signals)


def test_shadow_runner_set_hemisphere_signals_noop_when_inactive():
    """Calling set_hemisphere_signals before start_shadow must not raise."""
    from policy.shadow_runner import ShadowPolicyRunner

    runner = ShadowPolicyRunner()
    # No exception; signals silently dropped.
    runner.set_hemisphere_signals({"slot_4": 0.7, "slot_5": 0.9})


def test_shadow_encoder_reads_slot_4_and_slot_5_directly():
    """Baseline: the shadow encoder itself consumes slot_4 / slot_5 correctly.

    Separate from the runner wiring: proves the encoder contract the engine
    now relies on.
    """
    from policy.state_encoder import ShadowStateEncoder, SHADOW_STATE_DIM

    assert SHADOW_STATE_DIM == 22

    enc = ShadowStateEncoder()
    enc.set_hemisphere_signals({
        "slot_0": 0.1, "slot_1": 0.2, "slot_2": 0.3, "slot_3": 0.4,
        "slot_4": 0.7, "slot_5": 0.9,
    })
    state = {
        "consciousness": {
            "stage": "self_reflective",
            "transcendence_level": 3.0,
        },
        "memory_density": 0.5,
    }
    vec = enc.encode(state)
    assert len(vec) == SHADOW_STATE_DIM
    # Dims 20-21 are the M6 expanded slots.
    assert vec[20] == pytest.approx(0.7)
    assert vec[21] == pytest.approx(0.9)


def test_shadow_runner_forwards_signals_when_active():
    """When shadow is active, set_hemisphere_signals must reach the encoder."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not available; shadow runner cannot start")

    from policy.shadow_runner import ShadowPolicyRunner

    runner = ShadowPolicyRunner()
    started = runner.start_shadow(arch="mlp2")
    if not started:
        pytest.skip("Shadow runner did not start (torch init failure)")

    try:
        runner.set_hemisphere_signals({
            "slot_0": 0.0, "slot_1": 0.0, "slot_2": 0.0, "slot_3": 0.0,
            "slot_4": 0.42, "slot_5": 0.84,
        })
        enc = runner._shadow_encoder
        assert enc is not None
        assert enc._hemisphere_signals.get("slot_4") == pytest.approx(0.42)
        assert enc._hemisphere_signals.get("slot_5") == pytest.approx(0.84)

        state = {
            "consciousness": {
                "stage": "self_reflective",
                "transcendence_level": 3.0,
            },
            "memory_density": 0.5,
        }
        vec = enc.encode(state)
        assert len(vec) == 22
        assert vec[20] == pytest.approx(0.42)
        assert vec[21] == pytest.approx(0.84)
    finally:
        runner.mark_rolled_back()


def test_engine_tick_feeds_shadow_runner_when_active(monkeypatch):
    """Source-level guard: engine `_on_tick` path pipes signals into the
    shadow runner when the promotion pipeline exposes an active runner.

    Checking via inspection keeps the test hermetic (no event loop needed).
    """
    import inspect
    from consciousness import engine as engine_mod

    src = inspect.getsource(engine_mod.ConsciousnessEngine)
    # The wiring pattern must match exactly to keep the signal path alive.
    assert "shadow_runner.set_hemisphere_signals" in src, (
        "engine._on_tick must forward hemisphere signals into the shadow "
        "runner; otherwise M6 A/B sees dims 20-21 as zero forever."
    )
    # The engine looks up the promotion pipeline defensively via getattr
    # so downstream callers can monkey-patch it in tests. Accept either
    # the attribute access pattern or the getattr-lookup pattern.
    assert (
        "self._promotion_pipeline" in src
        or "\"_promotion_pipeline\"" in src
    )


# ---------------------------------------------------------------------------
# Fix 2: specialist_verification_ts is stamped at VERIFIED_PROBATIONARY
# ---------------------------------------------------------------------------


def _make_probationary_network(
    focus_value: str = "positive_memory",
    accuracy: float = 0.8,
    name_suffix: str = "a",
):
    """Construct a minimal NetworkArchitecture in PROBATIONARY_TRAINING."""
    from hemisphere.types import (
        NetworkArchitecture,
        NetworkTopology,
        LayerDefinition,
        PerformanceMetrics,
        TrainingProgress,
        SpecialistLifecycleStage,
        HemisphereFocus,
    )

    layer = LayerDefinition(
        id="h1", layer_type="hidden", node_count=8, activation="relu",
    )
    topology = NetworkTopology(
        input_size=8, layers=(layer,), output_size=4,
        total_parameters=100, activation_functions=("relu",),
    )
    perf = PerformanceMetrics(accuracy=accuracy)
    progress = TrainingProgress(current_epoch=1)
    arch = NetworkArchitecture(
        id=f"spec_{focus_value}_{name_suffix}",
        name=f"matrix_{focus_value}_{name_suffix}",
        focus=HemisphereFocus(focus_value),
        topology=topology,
        performance=perf,
        training_progress=progress,
    )
    arch.specialist_lifecycle = SpecialistLifecycleStage.PROBATIONARY_TRAINING
    return arch


def test_verification_ts_stamped_at_verified_probationary_transition():
    """PROBATIONARY_TRAINING -> VERIFIED_PROBATIONARY must stamp the ts."""
    from hemisphere.orchestrator import HemisphereOrchestrator
    from hemisphere.types import SpecialistLifecycleStage

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    net = _make_probationary_network(accuracy=0.8)
    assert net.specialist_verification_ts == 0.0

    # Inject into the orchestrator networks map.
    with orch._networks_lock:
        orch._networks[net.id] = net

    before = time.time()
    orch._check_specialist_promotions()
    after = time.time()

    assert net.specialist_lifecycle == SpecialistLifecycleStage.VERIFIED_PROBATIONARY
    assert net.specialist_verification_ts > 0.0
    assert before <= net.specialist_verification_ts <= after


def test_verification_ts_not_stamped_when_accuracy_below_floor():
    """Gate is accuracy > 0.5; below floor must stay in PROBATIONARY_TRAINING."""
    from hemisphere.orchestrator import HemisphereOrchestrator
    from hemisphere.types import SpecialistLifecycleStage

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    net = _make_probationary_network(accuracy=0.3)

    with orch._networks_lock:
        orch._networks[net.id] = net

    orch._check_specialist_promotions()

    assert net.specialist_lifecycle == SpecialistLifecycleStage.PROBATIONARY_TRAINING
    assert net.specialist_verification_ts == 0.0


def test_verification_ts_written_only_once_across_repeated_ticks():
    """Subsequent ticks on an already-verified specialist must NOT re-stamp.

    Once the specialist has crossed into VERIFIED_PROBATIONARY, the
    verification timestamp anchors the stability window for M6. Re-stamping
    on every tick would reset the window and permanently block expansion.
    """
    from hemisphere.orchestrator import HemisphereOrchestrator
    from hemisphere.types import SpecialistLifecycleStage

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    net = _make_probationary_network(accuracy=0.9)
    with orch._networks_lock:
        orch._networks[net.id] = net

    orch._check_specialist_promotions()
    first_ts = net.specialist_verification_ts
    assert first_ts > 0.0
    assert net.specialist_lifecycle == SpecialistLifecycleStage.VERIFIED_PROBATIONARY

    # Simulate additional ticks.
    time.sleep(0.01)
    orch._check_specialist_promotions()
    orch._check_specialist_promotions()

    assert net.specialist_verification_ts == first_ts, (
        "verification_ts must be write-once per VERIFIED_PROBATIONARY entry; "
        "re-stamping on subsequent ticks resets the M6 stability window."
    )


# ---------------------------------------------------------------------------
# Fix 2 downstream: _check_expansion_trigger() can actually succeed
# ---------------------------------------------------------------------------


def test_check_expansion_trigger_fires_with_two_promoted_and_old_verification():
    """Synthesize the three M6 conditions and confirm the trigger fires.

    Before the P3.5 fix, `specialist_verification_ts` was never written, so
    the `earliest_verification <= 0` guard at `orchestrator.py:1632` always
    short-circuited the trigger. This test proves the path is live now.
    """
    from hemisphere.orchestrator import (
        HemisphereOrchestrator,
        EXPANSION_MIN_PROMOTED,
        EXPANSION_MIN_IMPACT,
        EXPANSION_STABILITY_DAYS,
    )
    from hemisphere.types import SpecialistLifecycleStage

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    assert EXPANSION_MIN_PROMOTED == 2
    assert EXPANSION_MIN_IMPACT == 0.05
    assert EXPANSION_STABILITY_DAYS == 7

    # Two promoted specialists, both verified > 7 days ago, mean impact > 0.05.
    old_ts = time.time() - (EXPANSION_STABILITY_DAYS + 1) * 86400.0
    net_a = _make_probationary_network(
        focus_value="positive_memory", accuracy=0.9, name_suffix="prom_a",
    )
    net_a.specialist_lifecycle = SpecialistLifecycleStage.PROMOTED
    net_a.specialist_impact_score = 0.10
    net_a.specialist_verification_ts = old_ts

    net_b = _make_probationary_network(
        focus_value="negative_memory", accuracy=0.85, name_suffix="prom_b",
    )
    net_b.specialist_lifecycle = SpecialistLifecycleStage.PROMOTED
    net_b.specialist_impact_score = 0.12
    net_b.specialist_verification_ts = old_ts + 3600.0

    with orch._networks_lock:
        orch._networks[net_a.id] = net_a
        orch._networks[net_b.id] = net_b

    assert orch._expansion_triggered is False

    orch._check_expansion_trigger()

    assert orch._expansion_triggered is True, (
        "With 2 PROMOTED, mean impact > 0.05, and verification older than "
        "7 days, _check_expansion_trigger() must fire. If this assertion "
        "fails, P3.5 Task 3 has regressed."
    )


def test_check_expansion_trigger_blocked_when_verification_too_recent():
    """Stability-window guard: fresh verifications must block expansion."""
    from hemisphere.orchestrator import (
        HemisphereOrchestrator,
        EXPANSION_STABILITY_DAYS,
    )
    from hemisphere.types import SpecialistLifecycleStage

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    # Two promoted specialists verified a few seconds ago.
    recent = time.time() - 60.0
    net_a = _make_probationary_network(
        focus_value="positive_memory", accuracy=0.9, name_suffix="fresh_a",
    )
    net_a.specialist_lifecycle = SpecialistLifecycleStage.PROMOTED
    net_a.specialist_impact_score = 0.2
    net_a.specialist_verification_ts = recent

    net_b = _make_probationary_network(
        focus_value="negative_memory", accuracy=0.9, name_suffix="fresh_b",
    )
    net_b.specialist_lifecycle = SpecialistLifecycleStage.PROMOTED
    net_b.specialist_impact_score = 0.2
    net_b.specialist_verification_ts = recent

    with orch._networks_lock:
        orch._networks[net_a.id] = net_a
        orch._networks[net_b.id] = net_b

    orch._check_expansion_trigger()

    assert orch._expansion_triggered is False, (
        "Fresh verification must not pass the %d-day stability window."
        % EXPANSION_STABILITY_DAYS
    )
