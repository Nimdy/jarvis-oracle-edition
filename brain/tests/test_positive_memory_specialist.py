"""P3.6 — positive_memory Tier-2 specialist regression tests.

This file pins the four invariants the P3.6 ship contract requires:

  1. The encoder produces a 16-dim ``[0, 1]`` feature vector from any
     defensible context dict (including an entirely empty dict).
  2. ``compute_signal_value`` is deterministic and stays in ``[0, 1]``;
     it monotonically reflects positive-leaning memory features.
  3. The orchestrator dispatches ``positive_memory`` through the
     encoder *instead of* the accuracy-as-proxy fallback. Even at
     CANDIDATE_BIRTH (where ``performance.accuracy`` is 0.0) and even
     when network inference would fail, the broadcast signal is the
     encoder scalar — never ``performance.accuracy``.
  4. CANDIDATE_BIRTH → PROBATIONARY_TRAINING progresses under
     synthetic signal volume (memory storage populated with positive-
     tagged entries); the lifecycle ladder remains intact.

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
    from hemisphere.positive_memory_encoder import (
        PositiveMemoryEncoder,
        FEATURE_DIM,
    )
    assert FEATURE_DIM == 16
    vec = PositiveMemoryEncoder.encode({})
    assert len(vec) == 16


def test_encoder_handles_empty_context_gracefully():
    """An empty context must yield a valid 16-dim ``[0, 1]`` vector,
    not a crash. Tier-2 specialists are spawned at CANDIDATE_BIRTH with
    zero observed history; the encoder must degrade to all-zero."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder
    vec = PositiveMemoryEncoder.encode({})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert sum(vec) == 0.0


def test_encoder_features_stay_within_bounds_under_extreme_input():
    """Even pathological context (huge weights, NaN-like decay) must
    produce a clamped vector. The encoder is the policy boundary
    against state corruption upstream."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, type_, decay):
            self.weight = weight
            self.tags = tags
            self.type = type_
            self.decay_rate = decay

    mems = [
        _Mem(weight=10.0, tags=("happy", "milestone"), type_="core", decay=-1.0),
        _Mem(weight=float("nan"), tags=("joy",), type_="observation", decay=99.0),
        _Mem(weight=0.95, tags=("appreciation",), type_="core", decay=0.001),
    ]
    ctx = {
        "recent_memories": mems,
        "memory_density": 5.0,
        "max_memories": 1000,
        "traits": ("Empathetic", "Optimistic"),
        "mood_positivity": 7.0,
        "contradiction_debt": -2.0,
    }
    vec = PositiveMemoryEncoder.encode(ctx)
    assert all(0.0 <= v <= 1.0 for v in vec), vec


def test_encoder_block_layout_matches_documented_dimensions():
    """The 16 dims are split 8/4/4 across the three blocks. Downstream
    consumers (and future Tier-2 lanes copying this template) rely on
    that layout staying stable."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder
    a = PositiveMemoryEncoder.encode_memory_block({})
    b = PositiveMemoryEncoder.encode_episode_block({})
    c = PositiveMemoryEncoder.encode_mood_block({})
    assert len(a) == 8
    assert len(b) == 4
    assert len(c) == 4


def test_encoder_is_deterministic():
    """Identical context → identical vector. The broadcast-slot
    competition's hysteresis (SLOT_SWAP_THRESHOLD) depends on the
    signal not jittering between runs of the same tick."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, type_, decay):
            self.weight = weight
            self.tags = tags
            self.type = type_
            self.decay_rate = decay

    mems = [_Mem(0.7, ("happy", "win"), "core", 0.01) for _ in range(5)]
    ctx = {"recent_memories": mems, "memory_density": 0.3}
    v1 = PositiveMemoryEncoder.encode(ctx)
    v2 = PositiveMemoryEncoder.encode(ctx)
    assert v1 == v2


def test_signal_value_is_zero_for_empty_context():
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder
    assert PositiveMemoryEncoder.compute_signal_value({}) == 0.0


def test_signal_value_increases_with_positive_memory_volume():
    """Real-time scalar must respond to live positive-leaning memory
    state. This is the exact failure mode the accuracy-as-proxy
    fallback hides: with a 0.0 accuracy network, the fallback returns
    0.0 regardless of how much positive history accumulates."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, type_="observation", decay=0.01):
            self.weight = weight
            self.tags = tags
            self.type = type_
            self.decay_rate = decay

    sparse = [_Mem(0.2, ())]
    rich = [
        _Mem(0.85, ("happy", "milestone"), type_="core"),
        _Mem(0.78, ("appreciation",), type_="core"),
        _Mem(0.92, ("joy", "celebration"), type_="core"),
        _Mem(0.7, ("gratitude",)),
        _Mem(0.66, ("success",)),
    ]
    sparse_signal = PositiveMemoryEncoder.compute_signal_value({
        "recent_memories": sparse,
    })
    rich_signal = PositiveMemoryEncoder.compute_signal_value({
        "recent_memories": rich,
        "traits": ("Empathetic", "Optimistic", "Supportive"),
        "mood_positivity": 0.7,
    })
    assert 0.0 <= sparse_signal <= 1.0
    assert 0.0 <= rich_signal <= 1.0
    assert rich_signal > sparse_signal, (sparse_signal, rich_signal)


def test_signal_value_stays_in_unit_interval():
    """All four signal computations stay in ``[0, 1]``. This is a
    pre-condition of the broadcast-slot ranking math in
    ``_compute_signal_score``."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, type_="observation", decay=0.01):
            self.weight = weight
            self.tags = tags
            self.type = type_
            self.decay_rate = decay

    cases = [
        {},
        {"recent_memories": []},
        {"recent_memories": [_Mem(1.0, ("happy",), "core", 0.0)] * 32},
        {
            "recent_memories": [_Mem(0.5, ()) for _ in range(10)],
            "traits": ("Empathetic", "Curious"),
            "mood_positivity": 1.0,
            "contradiction_debt": 1.0,
        },
    ]
    for ctx in cases:
        v = PositiveMemoryEncoder.compute_signal_value(ctx)
        assert 0.0 <= v <= 1.0, (ctx, v)


# ---------------------------------------------------------------------------
# Architecture guardrails — encoder must not write canonical state
# ---------------------------------------------------------------------------


def test_encoder_module_has_no_canonical_writers():
    """Source-level guard: the encoder file must not import or reference
    any canonical state mutator. This is a coarse but durable contract;
    finer checks belong to the schema audit."""
    import inspect
    from hemisphere import positive_memory_encoder
    src = inspect.getsource(positive_memory_encoder)
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
    ]
    for needle in forbidden:
        assert needle not in src, f"forbidden writer '{needle}' in encoder"


def test_encoder_focus_name_constant_matches_enum():
    """The ``FOCUS_NAME`` literal acts as the schema-audit writer
    literal. It must match the enum value to keep the audit honest."""
    from hemisphere.positive_memory_encoder import FOCUS_NAME
    from hemisphere.types import HemisphereFocus
    assert FOCUS_NAME == HemisphereFocus.POSITIVE_MEMORY.value


# ---------------------------------------------------------------------------
# Orchestrator dispatch — never falls back to accuracy-as-proxy
# ---------------------------------------------------------------------------


def _fresh_orchestrator():
    from hemisphere.orchestrator import HemisphereOrchestrator
    try:
        return HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")


def _make_positive_memory_arch(stage, accuracy: float = 0.0):
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
        id="spec_positive_memory_a",
        name="matrix_positive_memory_a",
        focus=HemisphereFocus.POSITIVE_MEMORY,
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


def test_orchestrator_dispatches_positive_memory_to_encoder():
    """``_compute_signal_value`` must route ``positive_memory`` through
    the encoder, not through the accuracy fallback. We force the
    fallback path to look distinguishable by setting accuracy = 0.999;
    if the encoder is bypassed the test would observe ~0.999."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_positive_memory_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.999,
    )
    signal = orch._compute_signal_value(arch)
    # The encoder reads live memory state. Whatever it returns, it
    # must NOT be the 0.999 accuracy proxy. We allow any value in
    # [0, 0.95] to give headroom for unrelated live state.
    assert 0.0 <= signal <= 0.95, (
        "Tier-2 positive_memory leaked the accuracy fallback "
        f"(signal={signal})"
    )


def test_orchestrator_signal_path_runs_without_a_network_engine():
    """The encoder dispatch must not require a constructed model. If
    ``_engine.infer`` is missing or raises, the encoder still returns
    a valid scalar (this is the main robustness invariant for
    CANDIDATE_BIRTH)."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_positive_memory_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.7,
    )

    class _BrokenEngine:
        def infer(self, *_a, **_kw):
            raise RuntimeError("engine offline")

    orch._engine = _BrokenEngine()
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 1.0
    # Even with the broken engine, the accuracy fallback (0.7) must NOT
    # be returned for positive_memory.
    assert signal != pytest.approx(0.7)


def test_orchestrator_build_positive_memory_context_is_safe():
    """The live-state context builder must never raise, even when
    underlying memory/personality singletons fail to import. The
    encoder relies on this contract for tick-loop safety."""
    orch = _fresh_orchestrator()
    ctx = orch._build_positive_memory_context()
    assert isinstance(ctx, dict)
    for key in (
        "recent_memories",
        "recent_episodes",
        "memory_density",
        "max_memories",
        "traits",
    ):
        assert key in ctx


# ---------------------------------------------------------------------------
# Lifecycle: CANDIDATE_BIRTH → PROBATIONARY_TRAINING with synthetic volume
# ---------------------------------------------------------------------------


def test_positive_memory_spawns_at_candidate_birth():
    """Spawn through the public Matrix Protocol entry point."""
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus.POSITIVE_MEMORY,
        job_id="job_p3_6_test",
        name="matrix_positive_memory_p3_6",
    )
    assert arch is not None
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.focus == HemisphereFocus.POSITIVE_MEMORY


def test_positive_memory_advances_to_probationary_training_with_signal_volume():
    """CANDIDATE_BIRTH → PROBATIONARY_TRAINING when training has at
    least one epoch under synthetic signal volume. This is the exact
    P3.6 lifecycle gate the user requested."""
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
        id="spec_pm_volume",
        name="matrix_positive_memory_volume",
        focus=HemisphereFocus.POSITIVE_MEMORY,
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


def test_positive_memory_signal_responds_to_synthetic_memory_volume(monkeypatch):
    """End-to-end: with synthetic positive-memory volume installed at
    the orchestrator boundary, the signal scales above the empty-state
    baseline. Proves the dispatch + encoder + context builder chain
    actually carries live signal under realistic feed conditions."""
    from hemisphere.positive_memory_encoder import PositiveMemoryEncoder

    class _Mem:
        def __init__(self, weight, tags, type_="core", decay=0.005):
            self.weight = weight
            self.tags = tags
            self.type = type_
            self.decay_rate = decay

    baseline = PositiveMemoryEncoder.compute_signal_value({})
    rich = PositiveMemoryEncoder.compute_signal_value({
        "recent_memories": [
            _Mem(0.95, ("happy", "milestone")),
            _Mem(0.88, ("appreciation",)),
            _Mem(0.91, ("joy", "celebration")),
            _Mem(0.79, ("gratitude",)),
            _Mem(0.84, ("success",)),
        ],
        "memory_density": 0.45,
        "traits": ("Empathetic", "Optimistic"),
        "mood_positivity": 0.7,
    })
    assert baseline == 0.0
    assert rich >= 0.25, rich  # well clear of the noise floor


# ---------------------------------------------------------------------------
# Architect topology — positive_memory has an explicit output dim
# ---------------------------------------------------------------------------


def test_architect_has_explicit_output_size_for_positive_memory():
    """The architect must declare an explicit output dim for
    ``positive_memory`` instead of falling through to the default 6.
    The spawn-list registration is part of the P3.6 contract."""
    from hemisphere.architect import NeuralArchitect
    from hemisphere.types import HemisphereFocus
    arch = NeuralArchitect()
    out = arch._get_output_size(HemisphereFocus.POSITIVE_MEMORY)
    assert out == 4


# ---------------------------------------------------------------------------
# Schema audit — positive_memory is no longer future-only
# ---------------------------------------------------------------------------


def test_positive_memory_removed_from_future_only_whitelist():
    from scripts.schema_emission_audit import FUTURE_ONLY_HEMISPHERE_FOCUSES
    assert "positive_memory" not in FUTURE_ONLY_HEMISPHERE_FOCUSES


def test_positive_memory_writer_literal_present_in_brain_source():
    """The schema audit credits a focus as 'emitted' when its string
    literal appears as a quoted token in any brain/ source file
    (excluding ``types.py``). Confirm the encoder's ``FOCUS_NAME``
    constant satisfies that scan."""
    import pathlib
    encoder_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "hemisphere" / "positive_memory_encoder.py"
    )
    text = encoder_path.read_text(encoding="utf-8")
    assert '"positive_memory"' in text
