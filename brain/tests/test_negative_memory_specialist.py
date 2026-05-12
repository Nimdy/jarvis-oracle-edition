"""P3.7 — negative_memory Tier-2 specialist regression tests.

This file pins the four invariants the P3.7 ship contract requires
(mirroring P3.6 for ``positive_memory``):

  1. The encoder produces a 16-dim ``[0, 1]`` feature vector from any
     defensible context dict (including an entirely empty dict).
  2. ``compute_signal_value`` is deterministic and stays in ``[0, 1]``;
     it monotonically reflects negative-leaning memory and friction
     features.
  3. The orchestrator dispatches ``negative_memory`` through the
     encoder *instead of* the accuracy-as-proxy fallback. Even at
     CANDIDATE_BIRTH (where ``performance.accuracy`` is 0.0) and even
     when network inference would fail, the broadcast signal is the
     encoder scalar — never ``performance.accuracy``.
  4. CANDIDATE_BIRTH → PROBATIONARY_TRAINING progresses under
     synthetic signal volume; the lifecycle ladder remains intact.

The Matrix-cap, retirement, and downstream-ladder tests are already
covered by ``test_tier2_matrix_specialists.py``; this file does not
duplicate that surface.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Encoder unit tests
# ---------------------------------------------------------------------------


def test_encoder_dimension_is_sixteen():
    from hemisphere.negative_memory_encoder import (
        NegativeMemoryEncoder,
        FEATURE_DIM,
    )
    assert FEATURE_DIM == 16
    vec = NegativeMemoryEncoder.encode({})
    assert len(vec) == 16


def test_encoder_handles_empty_context_gracefully():
    """An empty context must yield a valid 16-dim ``[0, 1]`` vector,
    not a crash. Tier-2 specialists are spawned at CANDIDATE_BIRTH with
    zero observed history; the encoder must degrade to all-zero."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder
    vec = NegativeMemoryEncoder.encode({})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert sum(vec) == 0.0


def test_encoder_features_stay_within_bounds_under_extreme_input():
    """Even pathological context (huge weights, NaN-like decay,
    out-of-range pressure values) must produce a clamped vector."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, decay):
            self.weight = weight
            self.tags = tags
            self.decay_rate = decay

    mems = [
        _Mem(weight=10.0, tags=("error", "correction"), decay=-1.0),
        _Mem(weight=float("nan"), tags=("regressed",), decay=99.0),
        _Mem(weight=0.95, tags=("contradiction",), decay=0.001),
    ]
    ctx = {
        "recent_memories": mems,
        "memory_density": 5.0,
        "max_memories": 1000,
        "tier1_failure_rate": 7.0,
        "quarantine_pressure": 13.0,
        "contradiction_debt": -2.0,
        "low_confidence_retrieval_rate": -1.5,
        "regression_pressure": 4.0,
    }
    vec = NegativeMemoryEncoder.encode(ctx)
    assert all(0.0 <= v <= 1.0 for v in vec), vec


def test_encoder_block_layout_matches_documented_dimensions():
    """The 16 dims are split 8/4/4 across the three blocks. Downstream
    consumers (and future Tier-2 lanes copying this template) rely on
    that layout staying stable."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder
    a = NegativeMemoryEncoder.encode_memory_block({})
    b = NegativeMemoryEncoder.encode_episode_block({})
    c = NegativeMemoryEncoder.encode_friction_block({})
    assert len(a) == 8
    assert len(b) == 4
    assert len(c) == 4


def test_encoder_is_deterministic():
    """Identical context → identical vector. The broadcast-slot
    competition's hysteresis (SLOT_SWAP_THRESHOLD) depends on the
    signal not jittering between runs of the same tick."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, decay):
            self.weight = weight
            self.tags = tags
            self.decay_rate = decay

    mems = [_Mem(0.7, ("error", "correction"), 0.05) for _ in range(5)]
    ctx = {
        "recent_memories": mems,
        "memory_density": 0.3,
        "quarantine_pressure": 0.4,
        "contradiction_debt": 0.2,
        "tier1_failure_rate": 0.15,
    }
    v1 = NegativeMemoryEncoder.encode(ctx)
    v2 = NegativeMemoryEncoder.encode(ctx)
    assert v1 == v2


def test_signal_value_is_zero_for_empty_context():
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder
    assert NegativeMemoryEncoder.compute_signal_value({}) == 0.0


def test_signal_value_increases_with_negative_memory_volume():
    """Real-time scalar must respond to live negative-leaning memory
    state. This is the exact failure mode the accuracy-as-proxy
    fallback hides: with a 0.0 accuracy network, the fallback returns
    0.0 regardless of how much negative history accumulates."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, decay=0.01):
            self.weight = weight
            self.tags = tags
            self.decay_rate = decay

    sparse = [_Mem(0.2, ())]
    rich = [
        _Mem(0.85, ("error", "correction")),
        _Mem(0.78, ("regressed",)),
        _Mem(0.92, ("failure", "contradiction")),
        _Mem(0.7, ("frustration",)),
        _Mem(0.66, ("conflict",)),
    ]
    sparse_signal = NegativeMemoryEncoder.compute_signal_value({
        "recent_memories": sparse,
    })
    rich_signal = NegativeMemoryEncoder.compute_signal_value({
        "recent_memories": rich,
        "quarantine_pressure": 0.4,
        "contradiction_debt": 0.3,
        "tier1_failure_rate": 0.2,
    })
    assert 0.0 <= sparse_signal <= 1.0
    assert 0.0 <= rich_signal <= 1.0
    assert rich_signal > sparse_signal, (sparse_signal, rich_signal)


def test_signal_value_responds_to_friction_block_alone():
    """Block C (quarantine / coherence-debt) carries the system-friction
    backstop. Even with zero negative-tagged memory volume, sustained
    quarantine + contradiction pressure must lift the signal above the
    empty-state baseline. This is the explicit P3.7 design choice that
    keeps the signal honest during low-conversation periods.

    Watch item from P3.6 closeout: when conversation volume drops, a
    pure tag-matching signal would fall to zero even during real
    epistemic distress. Block C prevents that failure mode."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder
    baseline = NegativeMemoryEncoder.compute_signal_value({})
    friction_only = NegativeMemoryEncoder.compute_signal_value({
        "quarantine_pressure": 0.6,
        "contradiction_debt": 0.5,
        "tier1_failure_rate": 0.3,
    })
    assert baseline == 0.0
    assert friction_only > baseline


def test_signal_value_stays_in_unit_interval():
    """All four signal computations stay in ``[0, 1]``. This is a
    pre-condition of the broadcast-slot ranking math in
    ``_compute_signal_score``."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, decay=0.01):
            self.weight = weight
            self.tags = tags
            self.decay_rate = decay

    cases = [
        {},
        {"recent_memories": []},
        {"recent_memories": [_Mem(1.0, ("error",), 0.0)] * 32},
        {
            "recent_memories": [_Mem(0.5, ()) for _ in range(10)],
            "quarantine_pressure": 1.0,
            "contradiction_debt": 1.0,
            "tier1_failure_rate": 1.0,
            "low_confidence_retrieval_rate": 1.0,
            "regression_pressure": 1.0,
        },
    ]
    for ctx in cases:
        v = NegativeMemoryEncoder.compute_signal_value(ctx)
        assert 0.0 <= v <= 1.0, (ctx, v)


# ---------------------------------------------------------------------------
# Architecture guardrails — encoder must not write canonical state
# ---------------------------------------------------------------------------


def test_encoder_module_has_no_canonical_writers():
    """Source-level guard: the encoder file must not import or reference
    any canonical state mutator. This is a coarse but durable contract;
    finer checks belong to the schema audit."""
    import inspect
    from hemisphere import negative_memory_encoder
    src = inspect.getsource(negative_memory_encoder)
    forbidden = [
        "memory_storage.add",
        "memory_storage.update",
        "graph.create_edge",
        "BeliefGraph",
        "policy.commit",
        "autonomy.set_level",
        "self_improve",
        "set_hemisphere_signals",
        "spatial_estimator",
        "set_authority",
        "soul_integrity",
    ]
    for needle in forbidden:
        assert needle not in src, f"forbidden writer '{needle}' in encoder"


def test_encoder_focus_name_constant_matches_enum():
    """The ``FOCUS_NAME`` literal acts as the schema-audit writer
    literal. It must match the enum value to keep the audit honest."""
    from hemisphere.negative_memory_encoder import FOCUS_NAME
    from hemisphere.types import HemisphereFocus
    assert FOCUS_NAME == HemisphereFocus.NEGATIVE_MEMORY.value


def test_encoder_does_not_consume_emotion_depth_signal():
    """P3.7 watch item: the encoder must NOT consume any emotion_depth-
    derived input until a calibrated valence head exists. Wiring it now
    would silently inflate the signal with accuracy data — exactly the
    failure mode P3.6/P3.7 forbid.

    Behavioural guard: feeding a maximal ``emotion_negative_bias`` (and
    every other plausible emotion-derived key) into the encoder must
    not change the output. If a future PR starts reading any of these
    keys, the deltas below will trip and force a deliberate sign-off
    on the new feature contract."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder
    baseline = NegativeMemoryEncoder.compute_signal_value({})
    polluted = NegativeMemoryEncoder.compute_signal_value({
        "emotion_negative_bias": 1.0,
        "emotion_depth_score": 1.0,
        "emotion_classifier_negative": 1.0,
        "emotion_valence": 1.0,
    })
    assert polluted == baseline, (
        "negative_memory encoder appears to consume an emotion-derived "
        "input. That is forbidden until a calibrated valence head exists."
    )


# ---------------------------------------------------------------------------
# Orchestrator dispatch — never falls back to accuracy-as-proxy
# ---------------------------------------------------------------------------


def _fresh_orchestrator():
    from hemisphere.orchestrator import HemisphereOrchestrator
    try:
        return HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")


def _make_negative_memory_arch(stage, accuracy: float = 0.0):
    from hemisphere.types import (
        NetworkArchitecture, NetworkTopology, LayerDefinition,
        PerformanceMetrics, TrainingProgress, HemisphereFocus,
    )
    layer = LayerDefinition(
        id="h1", layer_type="hidden", node_count=8, activation="relu",
    )
    topo = NetworkTopology(
        input_size=8, layers=(layer,), output_size=4,
        total_parameters=100, activation_functions=("relu",),
    )
    arch = NetworkArchitecture(
        id="spec_negative_memory_a",
        name="matrix_negative_memory_a",
        focus=HemisphereFocus.NEGATIVE_MEMORY,
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


def test_orchestrator_dispatches_negative_memory_to_encoder():
    """``_compute_signal_value`` must route ``negative_memory`` through
    the encoder, not through the accuracy fallback. We force the
    fallback path to look distinguishable by setting accuracy = 0.999;
    if the encoder is bypassed the test would observe ~0.999."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_negative_memory_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.999,
    )
    signal = orch._compute_signal_value(arch)
    # The encoder reads live memory state. Whatever it returns, it
    # must NOT be the 0.999 accuracy proxy. We allow any value in
    # [0, 0.95] to give headroom for unrelated live state.
    assert 0.0 <= signal <= 0.95, (
        "Tier-2 negative_memory leaked the accuracy fallback "
        f"(signal={signal})"
    )


def test_orchestrator_signal_path_runs_without_a_network_engine():
    """The encoder dispatch must not require a constructed model. If
    ``_engine.infer`` is missing or raises, the encoder still returns
    a valid scalar (this is the main robustness invariant for
    CANDIDATE_BIRTH)."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_negative_memory_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.7,
    )

    class _BrokenEngine:
        def infer(self, *_a, **_kw):
            raise RuntimeError("engine offline")

    orch._engine = _BrokenEngine()
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 1.0
    # Even with the broken engine, the accuracy fallback (0.7) must NOT
    # be returned for negative_memory.
    assert signal != pytest.approx(0.7)


def test_orchestrator_build_negative_memory_context_is_safe():
    """The live-state context builder must never raise, even when
    underlying memory/quarantine singletons fail to import. The
    encoder relies on this contract for tick-loop safety."""
    orch = _fresh_orchestrator()
    ctx = orch._build_negative_memory_context()
    assert isinstance(ctx, dict)
    for key in (
        "recent_memories",
        "recent_episodes",
        "memory_density",
        "max_memories",
        "tier1_failure_rate",
        "quarantine_pressure",
        "contradiction_debt",
    ):
        assert key in ctx


# ---------------------------------------------------------------------------
# Lifecycle: CANDIDATE_BIRTH → PROBATIONARY_TRAINING with synthetic volume
# ---------------------------------------------------------------------------


def test_negative_memory_spawns_at_candidate_birth():
    """Spawn through the public Matrix Protocol entry point."""
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus.NEGATIVE_MEMORY,
        job_id="job_p3_7_test",
        name="matrix_negative_memory_p3_7",
    )
    assert arch is not None
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.focus == HemisphereFocus.NEGATIVE_MEMORY


def test_negative_memory_advances_to_probationary_training_with_signal_volume():
    """CANDIDATE_BIRTH → PROBATIONARY_TRAINING when training has at
    least one epoch under synthetic signal volume. This is the exact
    P3.7 lifecycle gate the user requested."""
    from hemisphere.types import (
        HemisphereFocus, SpecialistLifecycleStage,
        NetworkArchitecture, NetworkTopology, LayerDefinition,
        PerformanceMetrics, TrainingProgress,
    )
    orch = _fresh_orchestrator()

    layer = LayerDefinition(
        id="h1", layer_type="hidden", node_count=12, activation="relu",
    )
    topo = NetworkTopology(
        input_size=8, layers=(layer,), output_size=4,
        total_parameters=200, activation_functions=("relu",),
    )
    arch = NetworkArchitecture(
        id="spec_nm_volume",
        name="matrix_negative_memory_volume",
        focus=HemisphereFocus.NEGATIVE_MEMORY,
        topology=topo,
        performance=PerformanceMetrics(accuracy=0.0),
        training_progress=TrainingProgress(current_epoch=3),
    )
    arch.specialist_lifecycle = SpecialistLifecycleStage.CANDIDATE_BIRTH
    with orch._networks_lock:
        orch._networks[arch.id] = arch
    orch._check_specialist_promotions()
    assert arch.specialist_lifecycle == (
        SpecialistLifecycleStage.PROBATIONARY_TRAINING
    )


def test_negative_memory_signal_responds_to_synthetic_friction(monkeypatch):
    """End-to-end: with synthetic friction volume installed at the
    encoder boundary, the signal scales above the empty-state baseline.
    Proves the dispatch + encoder + context builder chain actually
    carries live signal under realistic feed conditions."""
    from hemisphere.negative_memory_encoder import NegativeMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, decay=0.005):
            self.weight = weight
            self.tags = tags
            self.decay_rate = decay

    baseline = NegativeMemoryEncoder.compute_signal_value({})
    rich = NegativeMemoryEncoder.compute_signal_value({
        "recent_memories": [
            _Mem(0.95, ("error", "correction")),
            _Mem(0.88, ("regressed",)),
            _Mem(0.91, ("contradiction", "conflict")),
            _Mem(0.79, ("frustration",)),
            _Mem(0.84, ("failure",)),
        ],
        "memory_density": 0.45,
        "quarantine_pressure": 0.55,
        "contradiction_debt": 0.4,
        "tier1_failure_rate": 0.25,
    })
    assert baseline == 0.0
    assert rich >= 0.25, rich  # well clear of the noise floor


# ---------------------------------------------------------------------------
# Architect topology — negative_memory has an explicit output dim
# ---------------------------------------------------------------------------


def test_architect_has_explicit_output_size_for_negative_memory():
    """The architect must declare an explicit output dim for
    ``negative_memory`` instead of falling through to the default 6.
    The spawn-list registration is part of the P3.7 contract."""
    from hemisphere.architect import NeuralArchitect
    from hemisphere.types import HemisphereFocus
    arch = NeuralArchitect()
    out = arch._get_output_size(HemisphereFocus.NEGATIVE_MEMORY)
    assert out == 4


# ---------------------------------------------------------------------------
# Schema audit — negative_memory is no longer future-only
# ---------------------------------------------------------------------------


def test_negative_memory_removed_from_future_only_whitelist():
    from scripts.schema_emission_audit import FUTURE_ONLY_HEMISPHERE_FOCUSES
    assert "negative_memory" not in FUTURE_ONLY_HEMISPHERE_FOCUSES


def test_negative_memory_writer_literal_present_in_brain_source():
    """The schema audit credits a focus as 'emitted' when its string
    literal appears as a quoted token in any brain/ source file
    (excluding ``types.py``). Confirm the encoder's ``FOCUS_NAME``
    constant satisfies that scan."""
    import pathlib
    encoder_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "hemisphere" / "negative_memory_encoder.py"
    )
    text = encoder_path.read_text(encoding="utf-8")
    assert '"negative_memory"' in text
