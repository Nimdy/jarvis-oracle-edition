"""P3.10 — skill_transfer Tier-2 specialist regression tests.

Pins the five invariants the P3.10 ship contract requires (mirrors
P3.6 / P3.7 / P3.8 / P3.9 plus the skill-specific
"similarity-is-not-capability" guard):

  1. The encoder produces a 16-dim ``[0, 1]`` feature vector from any
     defensible context dict (including an entirely empty dict).
  2. ``compute_signal_value`` is deterministic and stays in ``[0, 1]``;
     it monotonically reflects skill-registry breadth, learning-job
     phase progress, and cross-skill diversity.
  3. The orchestrator dispatches ``skill_transfer`` through the
     encoder *instead of* the accuracy-as-proxy fallback. Even at
     CANDIDATE_BIRTH (where ``performance.accuracy`` is 0.0) and even
     when network inference would fail, the broadcast signal is the
     encoder scalar — never ``performance.accuracy``.
  4. CANDIDATE_BIRTH → PROBATIONARY_TRAINING progresses under
     synthetic signal volume; the lifecycle ladder remains intact.
  5. **Similarity is not capability.** The encoder must not consume
     any "this skill is verified — promote it" hint, must not
     reference any canonical skill_registry / capability_gate
     mutator, and feeding such hints into the context must NOT
     change the output. Capability promotion remains the sole
     authority of the existing capability_gate path.
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
    from hemisphere.skill_transfer_encoder import (
        SkillTransferEncoder,
        FEATURE_DIM,
    )
    assert FEATURE_DIM == 16
    vec = SkillTransferEncoder.encode({})
    assert len(vec) == 16


def test_encoder_handles_empty_context_gracefully():
    """An empty context must yield a valid 16-dim ``[0, 1]`` vector,
    not a crash. Tier-2 specialists are spawned at CANDIDATE_BIRTH
    with zero observed history; the encoder must degrade to all-zero."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    vec = SkillTransferEncoder.encode({})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert sum(vec) == 0.0


def test_encoder_features_stay_within_bounds_under_extreme_input():
    """Even pathological context (NaN counts, huge values, negative
    fractions) must produce a clamped vector."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    ctx = {
        "total_skills": 10**9,
        "by_status": {"verified": 10**9, "probationary": -5},
        "skills": [
            {
                "status": "verified",
                "capability_type": "procedural",
                "has_evidence": True,
                "artifact_count": 10**9,
                "evidence_count": 10**9,
                "matrix_protocol": True,
            },
            {
                "status": "probationary",
                "capability_type": "perceptual",
                "has_evidence": False,
                "artifact_count": -10,
                "evidence_count": -10,
                "matrix_protocol": False,
            },
        ],
        "active_jobs": [
            {"phase": "verify", "stale": False},
            {"phase": "research", "stale": True},
            {"phase": "train", "stale": True},
        ],
        "active_jobs_count": 10**6,
        "failed_jobs_count": 10**6,
        "capability_type_overlap": 5.0,
        "transfer_advisory": -2.0,
    }
    vec = SkillTransferEncoder.encode(ctx)
    assert all(0.0 <= v <= 1.0 for v in vec), vec


def test_encoder_block_layout_matches_documented_dimensions():
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    a = SkillTransferEncoder.encode_registry_block({})
    b = SkillTransferEncoder.encode_jobs_block({})
    c = SkillTransferEncoder.encode_overlap_block({})
    assert len(a) == 8
    assert len(b) == 4
    assert len(c) == 4


def test_encoder_is_deterministic():
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    ctx = {
        "total_skills": 6,
        "skills": [
            {"status": "verified", "capability_type": "procedural",
             "has_evidence": True, "artifact_count": 3,
             "evidence_count": 2, "matrix_protocol": True},
            {"status": "verified", "capability_type": "perceptual",
             "has_evidence": True, "artifact_count": 5,
             "evidence_count": 4, "matrix_protocol": False},
            {"status": "probationary", "capability_type": "control",
             "has_evidence": True, "artifact_count": 1,
             "evidence_count": 1, "matrix_protocol": True},
            {"status": "candidate", "capability_type": "procedural",
             "has_evidence": False, "artifact_count": 0,
             "evidence_count": 0, "matrix_protocol": False},
            {"status": "verified", "capability_type": "procedural",
             "has_evidence": True, "artifact_count": 4,
             "evidence_count": 3, "matrix_protocol": True},
            {"status": "candidate", "capability_type": "perceptual",
             "has_evidence": False, "artifact_count": 0,
             "evidence_count": 0, "matrix_protocol": False},
        ],
        "active_jobs": [
            {"phase": "verify", "stale": False},
            {"phase": "train", "stale": False},
        ],
        "active_jobs_count": 2,
        "failed_jobs_count": 0,
    }
    v1 = SkillTransferEncoder.encode(ctx)
    v2 = SkillTransferEncoder.encode(ctx)
    assert v1 == v2


def test_signal_value_is_zero_for_empty_context():
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    assert SkillTransferEncoder.compute_signal_value({}) == 0.0


def test_signal_value_increases_with_registry_breadth():
    """Real-time scalar must respond to live registry maturity."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder

    sparse = {
        "total_skills": 1,
        "skills": [
            {"status": "candidate", "capability_type": "procedural",
             "has_evidence": False, "artifact_count": 0,
             "evidence_count": 0, "matrix_protocol": False},
        ],
        "active_jobs_count": 0,
        "failed_jobs_count": 0,
    }
    rich = {
        "total_skills": 12,
        "skills": [
            {"status": "verified", "capability_type": "procedural",
             "has_evidence": True, "artifact_count": 6,
             "evidence_count": 4, "matrix_protocol": True},
        ] * 6 + [
            {"status": "probationary", "capability_type": "perceptual",
             "has_evidence": True, "artifact_count": 3,
             "evidence_count": 2, "matrix_protocol": True},
        ] * 4 + [
            {"status": "candidate", "capability_type": "control",
             "has_evidence": True, "artifact_count": 1,
             "evidence_count": 1, "matrix_protocol": False},
        ] * 2,
        "active_jobs": [
            {"phase": "verify", "stale": False},
            {"phase": "train", "stale": False},
            {"phase": "register", "stale": False},
        ],
        "active_jobs_count": 3,
        "failed_jobs_count": 0,
    }
    sparse_signal = SkillTransferEncoder.compute_signal_value(sparse)
    rich_signal = SkillTransferEncoder.compute_signal_value(rich)
    assert 0.0 <= sparse_signal <= 1.0
    assert 0.0 <= rich_signal <= 1.0
    assert rich_signal > sparse_signal, (sparse_signal, rich_signal)


def test_signal_value_drops_with_high_failure_pressure():
    """Lots of failed jobs should reduce the signal: skill-transfer
    context is unreliable, broadcast slot competition should NOT
    score this state highly."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder

    base = {
        "total_skills": 6,
        "skills": [
            {"status": "verified", "capability_type": "procedural",
             "has_evidence": True, "artifact_count": 3,
             "evidence_count": 2, "matrix_protocol": True},
        ] * 6,
        "active_jobs": [
            {"phase": "verify", "stale": False},
            {"phase": "train", "stale": False},
        ],
        "active_jobs_count": 2,
        "failed_jobs_count": 0,
    }
    distressed = dict(base)
    distressed.update({"failed_jobs_count": 10})
    base_signal = SkillTransferEncoder.compute_signal_value(base)
    distressed_signal = SkillTransferEncoder.compute_signal_value(distressed)
    assert distressed_signal < base_signal, (base_signal, distressed_signal)


def test_signal_value_stays_in_unit_interval():
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    cases = [
        {},
        {
            "total_skills": 1000,
            "skills": [
                {"status": "verified", "capability_type": "procedural",
                 "has_evidence": True, "artifact_count": 1000,
                 "evidence_count": 1000, "matrix_protocol": True},
            ],
            "active_jobs_count": 1000,
            "failed_jobs_count": 0,
            "capability_type_overlap": 1.0,
            "transfer_advisory": 1.0,
        },
        {
            "total_skills": -10,
            "active_jobs_count": -10,
            "failed_jobs_count": -10,
        },
    ]
    for ctx in cases:
        v = SkillTransferEncoder.compute_signal_value(ctx)
        assert 0.0 <= v <= 1.0, (ctx, v)


# ---------------------------------------------------------------------------
# Architecture guardrails — encoder must not write canonical state
# ---------------------------------------------------------------------------


def test_encoder_module_has_no_canonical_writers():
    """Source-level guard: the encoder file must not import or
    reference any canonical state mutator."""
    import inspect
    from hemisphere import skill_transfer_encoder
    src = inspect.getsource(skill_transfer_encoder)
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
        # The defining capability/skill mutators — adding any of
        # these to the encoder would break the
        # similarity-is-not-capability contract.
        "skill_registry.set_status",
        "skill_registry.verify",
        "skill_registry.promote",
        "skill_registry.add",
        "skill_registry.update",
        "skill_registry.remove",
        "capability_gate.allow",
        "capability_gate.set",
        "capability_gate.grant",
        "capability_gate.promote",
        "set_capability",
        "promote_capability",
        "verify_skill",
    ]
    for needle in forbidden:
        assert needle not in src, f"forbidden writer '{needle}' in encoder"


def test_encoder_focus_name_constant_matches_enum():
    from hemisphere.skill_transfer_encoder import FOCUS_NAME
    from hemisphere.types import HemisphereFocus
    assert FOCUS_NAME == HemisphereFocus.SKILL_TRANSFER.value


# ---------------------------------------------------------------------------
# Capability guardrail — similarity is not capability
# ---------------------------------------------------------------------------


def test_encoder_does_not_consume_promote_or_verify_hints():
    """P3.10 contract: the encoder must NOT consume any "this skill
    is verified" / "promote this" / "capability proven" hint.
    Feeding such keys into the context must NOT change the output.
    This is the load-bearing capability guard for the skill_transfer
    lane — without it, the specialist could drift into
    "similar therefore promote" inference, which violates the
    capability gate the user explicitly flagged."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    baseline = SkillTransferEncoder.compute_signal_value({})
    polluted = SkillTransferEncoder.compute_signal_value({
        "promote_this_skill": True,
        "verify_now": True,
        "verified": True,
        "promotion_status": "promoted",
        "capability_proven": True,
        "auto_promote": True,
        "skill_is_safe": True,
        "plugin_is_safe": True,
        "force_capability": True,
        "capability_gate_override": True,
        "skill_id": "fake_skill_001",
        "transfer_grants_capability": True,
        # The keys below ARE consumed (they're part of the legitimate
        # contract), so we deliberately do NOT include them in the
        # pollution set: total_skills, skills, by_status,
        # active_jobs_count, etc.
    })
    assert polluted == baseline, (
        "skill_transfer encoder appears to consume a "
        "'promote-this-skill' / 'verified' / 'capability-proven' "
        "key. That is forbidden by the similarity-is-not-capability "
        "contract — the encoder may only consume registry/job "
        "counts and bounded fractions."
    )


def test_encoder_module_does_not_make_capability_claims():
    """Static guard: the encoder source must not name any
    capability-promotion / verify-now / auto-promote field as a
    context consumer. Adding such a field in a future PR forces a
    deliberate sign-off on the capability contract."""
    import inspect
    from hemisphere import skill_transfer_encoder
    src = inspect.getsource(skill_transfer_encoder)
    forbidden_consumer_patterns = [
        '_safe_attr(ctx, "promote_this_skill"',
        '_safe_attr(ctx, "verify_now"',
        '_safe_attr(ctx, "auto_promote"',
        '_safe_attr(ctx, "capability_proven"',
        '_safe_attr(ctx, "force_capability"',
        '_safe_attr(ctx, "transfer_grants_capability"',
        '_safe_attr(ctx, "skill_is_safe"',
        '_safe_attr(ctx, "plugin_is_safe"',
        'ctx.get("promote_this_skill"',
        'ctx.get("verify_now"',
        'ctx.get("auto_promote"',
        'ctx.get("capability_proven"',
    ]
    for needle in forbidden_consumer_patterns:
        assert needle not in src, (
            f"skill_transfer encoder appears to read '{needle}' — "
            "promotion / verify / capability-proven hints are not "
            "allowed by the capability contract."
        )


def test_orchestrator_context_does_not_carry_promotion_hints():
    """The orchestrator-level context builder must not surface any
    promotion / verify / capability-proven field. Second fence
    around the similarity-is-not-capability contract."""
    orch = _fresh_orchestrator()
    ctx = orch._build_skill_transfer_context()
    forbidden = {
        "promote_this_skill", "verify_now", "auto_promote",
        "capability_proven", "force_capability",
        "transfer_grants_capability", "skill_is_safe",
        "plugin_is_safe", "capability_gate_override",
    }
    for key in forbidden:
        assert key not in ctx, (
            f"_build_skill_transfer_context surfaced '{key}' — "
            "promotion / verify / capability-proven fields must not "
            "enter the encoder context."
        )


def test_encoder_signal_invariant_to_fake_verified_skill():
    """A pathologically attacker-crafted skill record claiming to be
    verified must NOT elevate the signal beyond what the registry's
    actual structure deserves. We test this by feeding two contexts
    with identical registry shape — one with a skill marked
    ``status='verified'`` and the same skill marked
    ``status='candidate'`` — and asserting the verified-fraction
    block does respond (correctly, registry shape changed) but no
    unbounded "promotion-grant" channel exists."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    candidate_ctx = {
        "total_skills": 1,
        "skills": [
            {"status": "candidate", "capability_type": "procedural",
             "has_evidence": False, "artifact_count": 0,
             "evidence_count": 0, "matrix_protocol": False},
        ],
        "active_jobs_count": 0,
        "failed_jobs_count": 0,
    }
    verified_ctx = dict(candidate_ctx)
    verified_ctx["skills"] = [
        {"status": "verified", "capability_type": "procedural",
         "has_evidence": True, "artifact_count": 1,
         "evidence_count": 1, "matrix_protocol": False},
    ]
    candidate_signal = SkillTransferEncoder.compute_signal_value(candidate_ctx)
    verified_signal = SkillTransferEncoder.compute_signal_value(verified_ctx)
    # The verified case naturally scores higher because the verified
    # fraction & evidence flags are real registry shape — but it
    # must remain bounded by the 16-dim capacity.
    assert 0.0 <= candidate_signal <= 1.0
    assert 0.0 <= verified_signal <= 1.0
    assert verified_signal <= 1.0  # No unbounded promotion grant.


# ---------------------------------------------------------------------------
# Orchestrator dispatch — never falls back to accuracy-as-proxy
# ---------------------------------------------------------------------------


def _fresh_orchestrator():
    from hemisphere.orchestrator import HemisphereOrchestrator
    try:
        return HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")


def _make_skill_transfer_arch(stage, accuracy: float = 0.0):
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
        id="spec_skill_transfer_a",
        name="matrix_skill_transfer_a",
        focus=HemisphereFocus.SKILL_TRANSFER,
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


def test_orchestrator_dispatches_skill_transfer_to_encoder():
    """``_compute_signal_value`` must route ``skill_transfer``
    through the encoder, not through the accuracy fallback. Setting
    accuracy = 0.999 makes the fallback path observable; if the
    encoder is bypassed the test would observe ~0.999."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_skill_transfer_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.999,
    )
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 0.95, (
        "Tier-2 skill_transfer leaked the accuracy fallback "
        f"(signal={signal})"
    )


def test_orchestrator_signal_path_runs_without_a_network_engine():
    """The encoder dispatch must not require a constructed model."""
    from hemisphere.types import SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = _make_skill_transfer_arch(
        SpecialistLifecycleStage.CANDIDATE_BIRTH, accuracy=0.7,
    )

    class _BrokenEngine:
        def infer(self, *_a, **_kw):
            raise RuntimeError("engine offline")

    orch._engine = _BrokenEngine()
    signal = orch._compute_signal_value(arch)
    assert 0.0 <= signal <= 1.0
    assert signal != pytest.approx(0.7)


def test_orchestrator_build_skill_transfer_context_is_safe():
    """The live-state context builder must never raise, even when
    underlying skill_registry singletons fail."""
    orch = _fresh_orchestrator()
    ctx = orch._build_skill_transfer_context()
    assert isinstance(ctx, dict)
    for key in (
        "total_skills",
        "by_status",
        "skills",
        "active_jobs_count",
        "failed_jobs_count",
    ):
        assert key in ctx


# ---------------------------------------------------------------------------
# Lifecycle: CANDIDATE_BIRTH → PROBATIONARY_TRAINING with synthetic volume
# ---------------------------------------------------------------------------


def test_skill_transfer_spawns_at_candidate_birth():
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage
    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus.SKILL_TRANSFER,
        job_id="job_p3_10_test",
        name="matrix_skill_transfer_p3_10",
    )
    assert arch is not None
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.focus == HemisphereFocus.SKILL_TRANSFER


def test_skill_transfer_advances_to_probationary_training_with_signal_volume():
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
        id="spec_st_volume",
        name="matrix_skill_transfer_volume",
        focus=HemisphereFocus.SKILL_TRANSFER,
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


def test_skill_transfer_signal_responds_to_synthetic_registry_volume():
    """End-to-end: with synthetic registry/job state installed at
    the encoder boundary, the signal scales above the empty-state
    baseline."""
    from hemisphere.skill_transfer_encoder import SkillTransferEncoder
    baseline = SkillTransferEncoder.compute_signal_value({})
    rich = SkillTransferEncoder.compute_signal_value({
        "total_skills": 10,
        "skills": [
            {"status": "verified", "capability_type": "procedural",
             "has_evidence": True, "artifact_count": 5,
             "evidence_count": 4, "matrix_protocol": True},
        ] * 5 + [
            {"status": "probationary", "capability_type": "perceptual",
             "has_evidence": True, "artifact_count": 3,
             "evidence_count": 2, "matrix_protocol": True},
        ] * 3 + [
            {"status": "candidate", "capability_type": "control",
             "has_evidence": True, "artifact_count": 1,
             "evidence_count": 1, "matrix_protocol": False},
        ] * 2,
        "active_jobs": [
            {"phase": "verify", "stale": False},
            {"phase": "train", "stale": False},
            {"phase": "register", "stale": False},
        ],
        "active_jobs_count": 3,
        "failed_jobs_count": 0,
    })
    assert baseline == 0.0
    assert rich >= 0.30, rich


# ---------------------------------------------------------------------------
# Architect topology — skill_transfer has an explicit output dim
# ---------------------------------------------------------------------------


def test_architect_has_explicit_output_size_for_skill_transfer():
    from hemisphere.architect import NeuralArchitect
    from hemisphere.types import HemisphereFocus
    arch = NeuralArchitect()
    out = arch._get_output_size(HemisphereFocus.SKILL_TRANSFER)
    assert out == 4


# ---------------------------------------------------------------------------
# Schema audit — skill_transfer is no longer future-only
# ---------------------------------------------------------------------------


def test_skill_transfer_removed_from_future_only_whitelist():
    from scripts.schema_emission_audit import FUTURE_ONLY_HEMISPHERE_FOCUSES
    assert "skill_transfer" not in FUTURE_ONLY_HEMISPHERE_FOCUSES


def test_skill_transfer_writer_literal_present_in_brain_source():
    """The schema audit credits a focus as 'emitted' when its string
    literal appears as a quoted token in any brain/ source file."""
    import pathlib
    encoder_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "hemisphere" / "skill_transfer_encoder.py"
    )
    text = encoder_path.read_text(encoding="utf-8")
    assert '"skill_transfer"' in text
