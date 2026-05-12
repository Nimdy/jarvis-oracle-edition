"""P3.9 — temporal_pattern Tier-2 specialist regression tests.

Pins the five invariants the P3.9 ship contract requires (mirrors
P3.6 / P3.7 / P3.8 plus the temporal-specific privacy guard):

  1. The encoder produces a 16-dim ``[0, 1]`` feature vector from any
     defensible context dict (including an entirely empty dict).
  2. ``compute_signal_value`` is deterministic and stays in ``[0, 1]``;
     it monotonically reflects activity recency / density / mode
     stability.
  3. The orchestrator dispatches ``temporal_pattern`` through the
     encoder *instead of* the accuracy-as-proxy fallback. Even at
     CANDIDATE_BIRTH (where ``performance.accuracy`` is 0.0) and even
     when network inference would fail, the broadcast signal is the
     encoder scalar — never ``performance.accuracy``.
  4. CANDIDATE_BIRTH → PROBATIONARY_TRAINING progresses under
     synthetic signal volume; the lifecycle ladder remains intact.
  5. **No schedule claims.** The encoder consumes only counts, time
     deltas, and mode scalars; it never reads
     ``hour_of_day`` / ``weekday`` / ``schedule`` / ``calendar``-shaped
     keys, and feeding such keys into the context must NOT change
     output. This is the speaker-equivalent privacy fence for the
     temporal lane.
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
    from hemisphere.temporal_pattern_encoder import (
        TemporalPatternEncoder,
        FEATURE_DIM,
    )
    assert FEATURE_DIM == 16
    vec = TemporalPatternEncoder.encode({})
    assert len(vec) == 16


def test_encoder_handles_empty_context_gracefully():
    """An empty context must yield a valid 16-dim ``[0, 1]`` vector,
    not a crash. Tier-2 specialists are spawned at CANDIDATE_BIRTH
    with zero observed history; the encoder must degrade to all-zero."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    vec = TemporalPatternEncoder.encode({})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert sum(vec) == 0.0


def test_encoder_features_stay_within_bounds_under_extreme_input():
    """Even pathological context (NaN deltas, huge counts, negative
    durations) must produce a clamped vector."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    ctx = {
        "seconds_since_last_activity": -1.0,
        "count_last_10min": 10**9,
        "count_last_1hour": 10**9,
        "count_last_24hour": 10**9,
        "idle_seconds": float("nan"),
        "mode": "passive",
        "mode_duration_s": float("inf"),
        "mode_history_len": 10**6,
        "mode_transitions_last_hour": 10**6,
        "recent_activity_timestamps": [0.0, -1.0, float("nan"), 1.0e15],
        "medium_activity_timestamps": (1.0, 2.0),
        "activity_count_last_30min": -10,
        "activity_count_prior_30min": -10,
        "rhythm_familiarity": 5.0,
    }
    vec = TemporalPatternEncoder.encode(ctx)
    assert all(0.0 <= v <= 1.0 for v in vec), vec


def test_encoder_block_layout_matches_documented_dimensions():
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    a = TemporalPatternEncoder.encode_recency_block({})
    b = TemporalPatternEncoder.encode_mode_block({})
    c = TemporalPatternEncoder.encode_cadence_block({})
    assert len(a) == 8
    assert len(b) == 4
    assert len(c) == 4


def test_encoder_is_deterministic():
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    ctx = {
        "seconds_since_last_activity": 5.0,
        "count_last_10min": 3,
        "count_last_1hour": 12,
        "count_last_24hour": 80,
        "idle_seconds": 5.0,
        "mode": "reflective",
        "mode_duration_s": 180.0,
        "mode_history_len": 4,
        "mode_transitions_last_hour": 2,
        "recent_activity_timestamps": (1000.0, 1010.0, 1020.0, 1030.0),
        "activity_count_last_30min": 8,
        "activity_count_prior_30min": 6,
    }
    v1 = TemporalPatternEncoder.encode(ctx)
    v2 = TemporalPatternEncoder.encode(ctx)
    assert v1 == v2


def test_signal_value_is_zero_for_empty_context():
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    assert TemporalPatternEncoder.compute_signal_value({}) == 0.0


def test_signal_value_increases_with_activity_density():
    """Real-time scalar must respond to live cadence state."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder

    sparse = {
        "seconds_since_last_activity": 2400.0,  # 40 min ago
        "count_last_10min": 0,
        "count_last_1hour": 1,
        "count_last_24hour": 5,
        "idle_seconds": 2400.0,
        "mode": "passive",
        "mode_duration_s": 30.0,
        "mode_history_len": 1,
        "mode_transitions_last_hour": 0,
    }
    rich = {
        "seconds_since_last_activity": 5.0,
        "count_last_10min": 8,
        "count_last_1hour": 30,
        "count_last_24hour": 150,
        "idle_seconds": 5.0,
        "mode": "engaged",
        "mode_duration_s": 600.0,
        "mode_history_len": 2,
        "mode_transitions_last_hour": 1,
        "recent_activity_timestamps": tuple(
            1000.0 + 30.0 * i for i in range(20)
        ),
        "activity_count_last_30min": 15,
        "activity_count_prior_30min": 12,
    }
    sparse_signal = TemporalPatternEncoder.compute_signal_value(sparse)
    rich_signal = TemporalPatternEncoder.compute_signal_value(rich)
    assert 0.0 <= sparse_signal <= 1.0
    assert 0.0 <= rich_signal <= 1.0
    assert rich_signal > sparse_signal, (sparse_signal, rich_signal)


def test_signal_value_drops_with_mode_thrashing_and_idle():
    """High mode-transition rate and long idle should reduce the
    signal: temporal context is unstable, broadcast slot competition
    should NOT score this state highly."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder

    base = {
        "seconds_since_last_activity": 5.0,
        "count_last_10min": 5,
        "count_last_1hour": 20,
        "count_last_24hour": 100,
        "idle_seconds": 5.0,
        "mode": "engaged",
        "mode_duration_s": 600.0,
        "mode_history_len": 2,
        "mode_transitions_last_hour": 1,
    }
    thrash = dict(base)
    thrash.update({
        "mode_duration_s": 5.0,
        "mode_transitions_last_hour": 25,
        "mode_history_len": 50,
        "idle_seconds": 1900.0,
    })
    base_signal = TemporalPatternEncoder.compute_signal_value(base)
    thrash_signal = TemporalPatternEncoder.compute_signal_value(thrash)
    assert thrash_signal < base_signal, (base_signal, thrash_signal)


def test_signal_value_stays_in_unit_interval():
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    cases = [
        {},
        {
            "seconds_since_last_activity": 0.0,
            "count_last_10min": 1000,
            "count_last_1hour": 1000,
            "count_last_24hour": 1000,
            "idle_seconds": 0.0,
            "mode": "engaged",
            "mode_duration_s": 1.0e6,
            "mode_history_len": 0,
            "mode_transitions_last_hour": 0,
            "rhythm_familiarity": 1.0,
        },
        {
            "seconds_since_last_activity": 1.0e9,
            "idle_seconds": 1.0e9,
            "mode_history_len": 1000,
            "mode_transitions_last_hour": 1000,
        },
    ]
    for ctx in cases:
        v = TemporalPatternEncoder.compute_signal_value(ctx)
        assert 0.0 <= v <= 1.0, (ctx, v)


def test_signal_value_zero_when_only_mode_label_present():
    """A bare ``mode`` label without recency / dwell / transitions is
    still a low-information state. The mode_known feature itself
    contributes some signal, but the overall scalar must remain low —
    never above the rich/active baseline. Specifically, a context
    with ONLY ``mode`` set must score below a context with rich
    activity data."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    only_mode = TemporalPatternEncoder.compute_signal_value({"mode": "passive"})
    rich = TemporalPatternEncoder.compute_signal_value({
        "seconds_since_last_activity": 5.0,
        "count_last_10min": 5,
        "count_last_1hour": 20,
        "count_last_24hour": 100,
        "idle_seconds": 5.0,
        "mode": "engaged",
        "mode_duration_s": 600.0,
        "mode_history_len": 2,
        "mode_transitions_last_hour": 1,
    })
    assert only_mode < rich
    assert 0.0 <= only_mode <= 0.20


# ---------------------------------------------------------------------------
# Architecture guardrails — encoder must not write canonical state
# ---------------------------------------------------------------------------


def test_encoder_module_has_no_canonical_writers():
    """Source-level guard: the encoder file must not import or
    reference any canonical state mutator."""
    import inspect
    from hemisphere import temporal_pattern_encoder
    src = inspect.getsource(temporal_pattern_encoder)
    forbidden = [
        "memory_storage.add",
        "memory_storage.update",
        "memory_storage.remove",
        "graph.create_edge",
        "BeliefGraph",
        "policy.commit",
        "autonomy.set_level",
        "self_improve",
        "set_hemisphere_signals",
        "spatial_estimator",
        "set_authority",
        "soul_integrity",
        "set_relationships",
        "set_schedule",
        "save_schedule",
    ]
    for needle in forbidden:
        assert needle not in src, f"forbidden writer '{needle}' in encoder"


def test_encoder_focus_name_constant_matches_enum():
    from hemisphere.temporal_pattern_encoder import FOCUS_NAME
    from hemisphere.types import HemisphereFocus
    assert FOCUS_NAME == HemisphereFocus.TEMPORAL_PATTERN.value


# ---------------------------------------------------------------------------
# Privacy guardrail — no schedule / calendar / weekday claims
# ---------------------------------------------------------------------------


def test_encoder_does_not_consume_schedule_claim_keys():
    """P3.9 contract: the encoder must NOT consume any schedule /
    calendar / weekday / hour-of-day field. Feeding such keys into
    the context must NOT change the output. This is the load-bearing
    privacy guard for the temporal_pattern lane — without it, the
    specialist could drift into 'David is here at 9 AM'-shaped
    inference, which violates the truth boundary the user explicitly
    flagged."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    baseline = TemporalPatternEncoder.compute_signal_value({})
    polluted = TemporalPatternEncoder.compute_signal_value({
        "hour_of_day": 9,
        "weekday": "monday",
        "day_of_week": 1,
        "schedule": {"david": "09:00-18:00"},
        "calendar": [{"start": 0, "end": 100, "title": "meeting"}],
        "user_schedule": {"morning": True},
        "presence_schedule": {"david": [9, 17]},
        "wake_word_timestamp": 1700000000.0,
        "stt_timestamp": 1700000001.0,
        "tts_timestamp": 1700000002.0,
        # The keys below ARE consumed (they're part of the legitimate
        # contract), so we deliberately do NOT include them in the
        # pollution set: count_last_10min, count_last_1hour, etc.
    })
    assert polluted == baseline, (
        "temporal_pattern encoder appears to consume a schedule / "
        "calendar / hour-of-day / weekday key. That is forbidden by "
        "the no-schedule-claim contract — the encoder may only "
        "consume bounded counts, time deltas, and mode scalars."
    )


def test_encoder_module_does_not_make_schedule_claims():
    """Static guard: the encoder source must not name any schedule /
    calendar / hour-of-day / weekday field as a context consumer.
    Adding such a field in a future PR forces a deliberate sign-off
    on the privacy contract."""
    import inspect
    from hemisphere import temporal_pattern_encoder
    src = inspect.getsource(temporal_pattern_encoder)
    forbidden_consumer_patterns = [
        '_safe_attr(ctx, "hour_of_day"',
        '_safe_attr(ctx, "weekday"',
        '_safe_attr(ctx, "day_of_week"',
        '_safe_attr(ctx, "schedule"',
        '_safe_attr(ctx, "calendar"',
        '_safe_attr(ctx, "user_schedule"',
        '_safe_attr(ctx, "presence_schedule"',
        'ctx.get("hour_of_day"',
        'ctx.get("weekday"',
        'ctx.get("day_of_week"',
        'ctx.get("schedule"',
        'ctx.get("calendar"',
    ]
    for needle in forbidden_consumer_patterns:
        assert needle not in src, (
            f"temporal_pattern encoder appears to read '{needle}' — "
            "schedule / calendar / weekday claims are not allowed by "
            "the privacy contract."
        )


def test_orchestrator_context_does_not_carry_schedule_claims():
    """The orchestrator-level context builder must not surface any
    schedule / calendar / weekday / hour-of-day field. Second fence
    around the no-claim contract."""
    orch = _fresh_orchestrator()
    ctx = orch._build_temporal_pattern_context()
    forbidden = {
        "hour_of_day", "weekday", "day_of_week",
        "schedule", "calendar", "user_schedule",
        "presence_schedule",
    }
    for key in forbidden:
        assert key not in ctx, (
            f"_build_temporal_pattern_context surfaced '{key}' — "
            "schedule / calendar fields must not enter the encoder "
            "context."
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


def _make_temporal_pattern_arch(stage, accuracy: float = 0.0):
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
        id="spec_temporal_pattern_a",
        name="matrix_temporal_pattern_a",
        focus=HemisphereFocus.TEMPORAL_PATTERN,
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


def test_orchestrator_dispatches_temporal_pattern_to_encoder():
    """``_compute_signal_value`` must route ``temporal_pattern``
    through the encoder, not through the accuracy fallback. Setting
    accuracy = 0.999 makes the fallback path observable; if the
    encoder is bypassed the test would observe ~0.999."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_temporal_pattern_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.999,
    )
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 0.95, (
        "Tier-2 temporal_pattern leaked the accuracy fallback "
        f"(signal={signal})"
    )


def test_orchestrator_signal_path_runs_without_a_network_engine():
    """The encoder dispatch must not require a constructed model."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_temporal_pattern_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.7,
    )

    class _BrokenEngine:
        def infer(self, *_a, **_kw):
            raise RuntimeError("engine offline")

    orch._engine = _BrokenEngine()
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 1.0
    assert signal != pytest.approx(0.7)


def test_orchestrator_build_temporal_pattern_context_is_safe():
    """The live-state context builder must never raise, even when
    underlying memory / mode-manager singletons fail."""
    orch = _fresh_orchestrator()
    ctx = orch._build_temporal_pattern_context()
    assert isinstance(ctx, dict)
    for key in (
        "seconds_since_last_activity",
        "count_last_10min",
        "count_last_1hour",
        "count_last_24hour",
        "idle_seconds",
        "mode",
        "mode_duration_s",
        "mode_history_len",
        "mode_transitions_last_hour",
        "recent_activity_timestamps",
        "medium_activity_timestamps",
        "activity_count_last_30min",
        "activity_count_prior_30min",
    ):
        assert key in ctx


# ---------------------------------------------------------------------------
# Lifecycle: CANDIDATE_BIRTH → PROBATIONARY_TRAINING with synthetic volume
# ---------------------------------------------------------------------------


def test_temporal_pattern_spawns_at_candidate_birth():
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus.TEMPORAL_PATTERN,
        job_id="job_p3_9_test",
        name="matrix_temporal_pattern_p3_9",
    )
    assert arch is not None
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.focus == HemisphereFocus.TEMPORAL_PATTERN


def test_temporal_pattern_advances_to_probationary_training_with_signal_volume():
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
        id="spec_tp_volume",
        name="matrix_temporal_pattern_volume",
        focus=HemisphereFocus.TEMPORAL_PATTERN,
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


def test_temporal_pattern_signal_responds_to_synthetic_cadence():
    """End-to-end: with synthetic activity-cadence state installed at
    the encoder boundary, the signal scales above the empty-state
    baseline."""
    from hemisphere.temporal_pattern_encoder import TemporalPatternEncoder
    baseline = TemporalPatternEncoder.compute_signal_value({})
    rich = TemporalPatternEncoder.compute_signal_value({
        "seconds_since_last_activity": 5.0,
        "count_last_10min": 6,
        "count_last_1hour": 25,
        "count_last_24hour": 120,
        "idle_seconds": 5.0,
        "mode": "engaged",
        "mode_duration_s": 600.0,
        "mode_history_len": 2,
        "mode_transitions_last_hour": 1,
        "recent_activity_timestamps": tuple(
            1000.0 + 30.0 * i for i in range(20)
        ),
        "activity_count_last_30min": 15,
        "activity_count_prior_30min": 12,
    })
    assert baseline == 0.0
    assert rich >= 0.40, rich


# ---------------------------------------------------------------------------
# Architect topology — temporal_pattern has an explicit output dim
# ---------------------------------------------------------------------------


def test_architect_has_explicit_output_size_for_temporal_pattern():
    from hemisphere.architect import NeuralArchitect
    from hemisphere.types import HemisphereFocus
    arch = NeuralArchitect()
    out = arch._get_output_size(HemisphereFocus.TEMPORAL_PATTERN)
    assert out == 4


# ---------------------------------------------------------------------------
# Schema audit — temporal_pattern is no longer future-only
# ---------------------------------------------------------------------------


def test_temporal_pattern_removed_from_future_only_whitelist():
    from scripts.schema_emission_audit import FUTURE_ONLY_HEMISPHERE_FOCUSES
    assert "temporal_pattern" not in FUTURE_ONLY_HEMISPHERE_FOCUSES


def test_temporal_pattern_writer_literal_present_in_brain_source():
    """The schema audit credits a focus as 'emitted' when its string
    literal appears as a quoted token in any brain/ source file."""
    import pathlib
    encoder_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "hemisphere" / "temporal_pattern_encoder.py"
    )
    text = encoder_path.read_text(encoding="utf-8")
    assert '"temporal_pattern"' in text
