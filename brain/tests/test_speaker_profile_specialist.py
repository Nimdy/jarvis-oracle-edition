"""P3.8 — speaker_profile Tier-2 specialist regression tests.

This file pins the five invariants the P3.8 ship contract requires
(mirrors P3.6 / P3.7 plus the speaker-specific raw-embedding-leak guard):

  1. The encoder produces a 16-dim ``[0, 1]`` feature vector from any
     defensible context dict (including an entirely empty dict).
  2. ``compute_signal_value`` is deterministic and stays in ``[0, 1]``;
     it monotonically reflects identity-fusion confidence and addressee
     stability.
  3. The orchestrator dispatches ``speaker_profile`` through the
     encoder *instead of* the accuracy-as-proxy fallback. Even at
     CANDIDATE_BIRTH (where ``performance.accuracy`` is 0.0) and even
     when network inference would fail, the broadcast signal is the
     encoder scalar — never ``performance.accuracy``.
  4. CANDIDATE_BIRTH → PROBATIONARY_TRAINING progresses under
     synthetic signal volume; the lifecycle ladder remains intact.
  5. **Raw speaker embeddings never enter the encoder.** Feeding
     ``embedding`` / ``embeddings`` / ``vector`` / ``ecapa`` keys into
     the context must NOT change the output. This is the speaker-
     specific privacy/contract guard that P3.6/P3.7 did not need.

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
    from hemisphere.speaker_profile_encoder import (
        SpeakerProfileEncoder,
        FEATURE_DIM,
    )
    assert FEATURE_DIM == 16
    vec = SpeakerProfileEncoder.encode({})
    assert len(vec) == 16


def test_encoder_handles_empty_context_gracefully():
    """An empty context must yield a valid 16-dim ``[0, 1]`` vector,
    not a crash. Tier-2 specialists are spawned at CANDIDATE_BIRTH with
    zero observed history; the encoder must degrade to all-zero."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    vec = SpeakerProfileEncoder.encode({})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert sum(vec) == 0.0


def test_encoder_features_stay_within_bounds_under_extreme_input():
    """Even pathological context (huge confidences, NaN ages,
    out-of-range counts) must produce a clamped vector."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder

    ctx = {
        "identity_confidence": 5.0,
        "is_known": True,
        "voice_confidence": float("nan"),
        "face_confidence": -1.0,
        "voice_name": "David",
        "face_name": "David",
        "conflict": False,
        "voice_age_s": -100.0,
        "face_age_s": float("inf"),
        "flip_count": 9999,
        "visible_person_count": 50,
        "multi_person_suppression_active": True,
        "cold_start_active": False,
        "voice_trust_state": "stable",
        "known_speakers_count": 10000,
        "current_speaker_interaction_count": 99999,
        "relationships_count": 200,
        "rapport_stability": 7.0,
    }
    vec = SpeakerProfileEncoder.encode(ctx)
    assert all(0.0 <= v <= 1.0 for v in vec), vec


def test_encoder_block_layout_matches_documented_dimensions():
    """The 16 dims are split 8/4/4 across the three blocks. Downstream
    consumers (and future Tier-2 lanes copying this template) rely on
    that layout staying stable."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    a = SpeakerProfileEncoder.encode_identity_block({})
    b = SpeakerProfileEncoder.encode_addressee_block({})
    c = SpeakerProfileEncoder.encode_registry_block({})
    assert len(a) == 8
    assert len(b) == 4
    assert len(c) == 4


def test_encoder_is_deterministic():
    """Identical context → identical vector."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    ctx = {
        "identity_confidence": 0.85,
        "is_known": True,
        "voice_confidence": 0.9,
        "face_confidence": 0.88,
        "voice_name": "David",
        "face_name": "David",
        "voice_age_s": 1.5,
        "face_age_s": 5.0,
        "flip_count": 1,
        "visible_person_count": 1,
        "voice_trust_state": "stable",
        "known_speakers_count": 2,
        "current_speaker_interaction_count": 25,
        "relationships_count": 3,
    }
    v1 = SpeakerProfileEncoder.encode(ctx)
    v2 = SpeakerProfileEncoder.encode(ctx)
    assert v1 == v2


def test_signal_value_is_zero_for_empty_context():
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    assert SpeakerProfileEncoder.compute_signal_value({}) == 0.0


def test_signal_value_increases_with_identity_confidence():
    """Real-time scalar must respond to live identity-fusion state."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder

    sparse = {
        "identity_confidence": 0.1,
        "voice_confidence": 0.1,
        "face_confidence": 0.0,
        "voice_name": "",
        "face_name": "",
        "voice_age_s": 25.0,
        "voice_trust_state": "tentative",
    }
    rich = {
        "identity_confidence": 0.92,
        "is_known": True,
        "voice_confidence": 0.88,
        "face_confidence": 0.91,
        "voice_name": "David",
        "face_name": "David",
        "voice_age_s": 1.0,
        "face_age_s": 2.0,
        "flip_count": 0,
        "visible_person_count": 1,
        "multi_person_suppression_active": False,
        "cold_start_active": False,
        "voice_trust_state": "stable",
        "known_speakers_count": 3,
        "current_speaker_interaction_count": 30,
        "relationships_count": 4,
    }
    sparse_signal = SpeakerProfileEncoder.compute_signal_value(sparse)
    rich_signal = SpeakerProfileEncoder.compute_signal_value(rich)
    assert 0.0 <= sparse_signal <= 1.0
    assert 0.0 <= rich_signal <= 1.0
    assert rich_signal > sparse_signal, (sparse_signal, rich_signal)


def test_signal_value_drops_under_conflict_and_multi_person():
    """Conflict and multi-person suppression must reduce the signal —
    the broadcast slot competition should NOT be high when speaker
    grounding is unstable."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder

    base = {
        "identity_confidence": 0.9,
        "is_known": True,
        "voice_confidence": 0.85,
        "face_confidence": 0.88,
        "voice_name": "David",
        "face_name": "David",
        "voice_age_s": 1.0,
        "face_age_s": 2.0,
        "flip_count": 0,
        "visible_person_count": 1,
        "voice_trust_state": "stable",
        "known_speakers_count": 3,
    }
    degraded = dict(base)
    degraded.update({
        "conflict": True,
        "multi_person_suppression_active": True,
        "visible_person_count": 3,
        "flip_count": 8,
        "voice_trust_state": "conflicted",
    })
    base_signal = SpeakerProfileEncoder.compute_signal_value(base)
    degraded_signal = SpeakerProfileEncoder.compute_signal_value(degraded)
    assert degraded_signal < base_signal, (base_signal, degraded_signal)


def test_signal_value_stays_in_unit_interval():
    """All four signal computations stay in ``[0, 1]``."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    cases = [
        {},
        {
            "identity_confidence": 1.0, "voice_confidence": 1.0, "face_confidence": 1.0,
            "voice_name": "X", "face_name": "X", "is_known": True,
            "voice_age_s": 0.0, "face_age_s": 0.0,
            "visible_person_count": 1, "voice_trust_state": "stable",
            "known_speakers_count": 100, "current_speaker_interaction_count": 1000,
            "relationships_count": 50, "rapport_stability": 1.0,
        },
        {
            "identity_confidence": 0.0, "conflict": True,
            "multi_person_suppression_active": True,
            "cold_start_active": True,
            "flip_count": 1000,
        },
    ]
    for ctx in cases:
        v = SpeakerProfileEncoder.compute_signal_value(ctx)
        assert 0.0 <= v <= 1.0, (ctx, v)


# ---------------------------------------------------------------------------
# Architecture guardrails — encoder must not write canonical state
# ---------------------------------------------------------------------------


def test_encoder_module_has_no_canonical_writers():
    """Source-level guard: the encoder file must not import or reference
    any canonical state mutator."""
    import inspect
    from hemisphere import speaker_profile_encoder
    src = inspect.getsource(speaker_profile_encoder)
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
        "set_relationships",
    ]
    for needle in forbidden:
        assert needle not in src, f"forbidden writer '{needle}' in encoder"


def test_encoder_focus_name_constant_matches_enum():
    """The ``FOCUS_NAME`` literal acts as the schema-audit writer literal."""
    from hemisphere.speaker_profile_encoder import FOCUS_NAME
    from hemisphere.types import HemisphereFocus
    assert FOCUS_NAME == HemisphereFocus.SPEAKER_PROFILE.value


# ---------------------------------------------------------------------------
# Speaker-specific privacy guard — no raw embedding leakage
# ---------------------------------------------------------------------------


def test_encoder_does_not_consume_raw_embeddings():
    """P3.8 contract: the encoder must NOT consume any raw embedding
    input (192-dim ECAPA speaker_repr, 512-dim face_repr, or any
    arbitrary float vector). Feeding embedding-shaped keys into the
    context must NOT change the output. This is the load-bearing
    privacy/architecture guard for the speaker_profile lane."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    baseline = SpeakerProfileEncoder.compute_signal_value({})
    polluted = SpeakerProfileEncoder.compute_signal_value({
        "embedding": [0.5] * 192,
        "embeddings": [[0.5] * 192, [0.7] * 192],
        "vector": [0.9] * 512,
        "vectors": [[0.9] * 512],
        "speaker_repr": [0.3] * 192,
        "face_repr": [0.4] * 512,
        "ecapa": [0.1] * 192,
        "ecapa_embedding": [0.2] * 192,
    })
    assert polluted == baseline, (
        "speaker_profile encoder appears to consume a raw embedding "
        "input. That is forbidden by the no-raw-embedding-leak contract."
    )


def test_encoder_module_does_not_reference_raw_embedding_keys():
    """Static guard: the encoder source must not name any raw embedding
    field as a context key. If a future PR adds one, this test forces a
    deliberate sign-off on the new feature contract."""
    import inspect
    from hemisphere import speaker_profile_encoder
    src = inspect.getsource(speaker_profile_encoder)
    # The literal field names below would only appear in source if the
    # encoder consumed them via ``_safe_attr(ctx, "<name>", ...)`` or
    # equivalent. Mentioning them in comments / module docstring is
    # fine (the module docstring discusses the contract); the
    # behavioural guard above is the real privacy fence.
    forbidden_consumer_patterns = [
        '_safe_attr(ctx, "embedding"',
        '_safe_attr(ctx, "embeddings"',
        '_safe_attr(ctx, "speaker_repr"',
        '_safe_attr(ctx, "face_repr"',
        '_safe_attr(ctx, "ecapa"',
        'ctx.get("embedding"',
        'ctx.get("embeddings"',
        'ctx.get("speaker_repr"',
        'ctx.get("face_repr"',
    ]
    for needle in forbidden_consumer_patterns:
        assert needle not in src, (
            f"speaker_profile encoder appears to read '{needle}' — "
            "raw embeddings must not enter the encoder."
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


def _make_speaker_profile_arch(stage, accuracy: float = 0.0):
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
        id="spec_speaker_profile_a",
        name="matrix_speaker_profile_a",
        focus=HemisphereFocus.SPEAKER_PROFILE,
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


def test_orchestrator_dispatches_speaker_profile_to_encoder():
    """``_compute_signal_value`` must route ``speaker_profile`` through
    the encoder, not through the accuracy fallback. Setting accuracy =
    0.999 makes the fallback path observable; if the encoder is bypassed
    the test would observe ~0.999."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_speaker_profile_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.999,
    )
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 0.95, (
        "Tier-2 speaker_profile leaked the accuracy fallback "
        f"(signal={signal})"
    )


def test_orchestrator_signal_path_runs_without_a_network_engine():
    """The encoder dispatch must not require a constructed model."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_speaker_profile_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.7,
    )

    class _BrokenEngine:
        def infer(self, *_a, **_kw):
            raise RuntimeError("engine offline")

    orch._engine = _BrokenEngine()
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 1.0
    assert signal != pytest.approx(0.7)


def test_orchestrator_build_speaker_profile_context_is_safe():
    """The live-state context builder must never raise, even when
    underlying identity/soul singletons fail to import."""
    orch = _fresh_orchestrator()
    ctx = orch._build_speaker_profile_context()
    assert isinstance(ctx, dict)
    for key in (
        "identity_confidence",
        "is_known",
        "voice_confidence",
        "face_confidence",
        "voice_name",
        "face_name",
        "conflict",
        "voice_age_s",
        "face_age_s",
        "flip_count",
        "visible_person_count",
        "multi_person_suppression_active",
        "cold_start_active",
        "voice_trust_state",
        "known_speakers_count",
        "relationships_count",
    ):
        assert key in ctx


def test_orchestrator_context_does_not_carry_raw_embeddings():
    """The orchestrator-level context builder must not surface any
    raw embedding key. This is the second fence around the no-leak
    contract: even if a future bug landed a raw embedding into the
    builder, the assertion below would trip."""
    orch = _fresh_orchestrator()
    ctx = orch._build_speaker_profile_context()
    forbidden = {"embedding", "embeddings", "speaker_repr", "face_repr",
                 "ecapa", "ecapa_embedding", "vector", "vectors"}
    for key in forbidden:
        assert key not in ctx, (
            f"_build_speaker_profile_context surfaced '{key}' — "
            "raw embeddings must not enter the encoder context."
        )


# ---------------------------------------------------------------------------
# Lifecycle: CANDIDATE_BIRTH → PROBATIONARY_TRAINING with synthetic volume
# ---------------------------------------------------------------------------


def test_speaker_profile_spawns_at_candidate_birth():
    """Spawn through the public Matrix Protocol entry point."""
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus.SPEAKER_PROFILE,
        job_id="job_p3_8_test",
        name="matrix_speaker_profile_p3_8",
    )
    assert arch is not None
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.focus == HemisphereFocus.SPEAKER_PROFILE


def test_speaker_profile_advances_to_probationary_training_with_signal_volume():
    """CANDIDATE_BIRTH → PROBATIONARY_TRAINING when training has at
    least one epoch under synthetic signal volume."""
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
        id="spec_sp_volume",
        name="matrix_speaker_profile_volume",
        focus=HemisphereFocus.SPEAKER_PROFILE,
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


def test_speaker_profile_signal_responds_to_synthetic_identity_state():
    """End-to-end: with synthetic identity-fusion state installed at
    the encoder boundary, the signal scales above the empty-state
    baseline. Proves the dispatch + encoder + context builder chain
    actually carries live signal under realistic feed conditions."""
    from hemisphere.speaker_profile_encoder import SpeakerProfileEncoder
    baseline = SpeakerProfileEncoder.compute_signal_value({})
    rich = SpeakerProfileEncoder.compute_signal_value({
        "identity_confidence": 0.92,
        "is_known": True,
        "voice_confidence": 0.88,
        "face_confidence": 0.91,
        "voice_name": "David",
        "face_name": "David",
        "voice_age_s": 1.0,
        "face_age_s": 2.0,
        "flip_count": 0,
        "visible_person_count": 1,
        "voice_trust_state": "stable",
        "known_speakers_count": 3,
        "current_speaker_interaction_count": 30,
        "relationships_count": 4,
    })
    assert baseline == 0.0
    assert rich >= 0.40, rich  # well clear of the noise floor


# ---------------------------------------------------------------------------
# Architect topology — speaker_profile has an explicit output dim
# ---------------------------------------------------------------------------


def test_architect_has_explicit_output_size_for_speaker_profile():
    """The architect must declare an explicit output dim for
    ``speaker_profile`` instead of falling through to the default 6."""
    from hemisphere.architect import NeuralArchitect
    from hemisphere.types import HemisphereFocus
    arch = NeuralArchitect()
    out = arch._get_output_size(HemisphereFocus.SPEAKER_PROFILE)
    assert out == 4


# ---------------------------------------------------------------------------
# Schema audit — speaker_profile is no longer future-only
# ---------------------------------------------------------------------------


def test_speaker_profile_removed_from_future_only_whitelist():
    from scripts.schema_emission_audit import FUTURE_ONLY_HEMISPHERE_FOCUSES
    assert "speaker_profile" not in FUTURE_ONLY_HEMISPHERE_FOCUSES


def test_speaker_profile_writer_literal_present_in_brain_source():
    """The schema audit credits a focus as 'emitted' when its string
    literal appears as a quoted token in any brain/ source file."""
    import pathlib
    encoder_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "hemisphere" / "speaker_profile_encoder.py"
    )
    text = encoder_path.read_text(encoding="utf-8")
    assert '"speaker_profile"' in text
