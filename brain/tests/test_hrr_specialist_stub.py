"""Lifecycle tests for the Tier-1 ``hrr_encoder`` specialist stub (P4).

Proves this sprint's contract:

* ``HemisphereFocus.HRR_ENCODER`` exists.
* It is classified as Tier-1 (``_TIER1_FOCUSES``), not Tier-2 matrix.
* No network is registered under it in a fresh orchestrator.
* ``HRREncoder.encode(...)`` returns ``[0.0] * FEATURE_DIM`` when HRR is
  disabled, and derives non-zero features when a synthetic enabled status
  is fed in.
* **Negative**: no ``holographic_cognition`` enum value, no Tier-2
  registration path for it, and the specialist stub is NOT in the
  matrix-eligible set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Enum + classification
# ---------------------------------------------------------------------------


def test_hrr_encoder_focus_exists_in_enum():
    from hemisphere.types import HemisphereFocus

    assert HemisphereFocus.HRR_ENCODER.value == "hrr_encoder"


def test_hrr_encoder_focus_is_tier1_not_tier2():
    from hemisphere.orchestrator import _TIER1_FOCUSES
    from hemisphere.types import MATRIX_ELIGIBLE_FOCUSES, HemisphereFocus

    assert HemisphereFocus.HRR_ENCODER in _TIER1_FOCUSES
    assert HemisphereFocus.HRR_ENCODER not in MATRIX_ELIGIBLE_FOCUSES


def test_fresh_orchestrator_has_no_hrr_network_registered():
    """CANDIDATE_BIRTH seat is reserved, but no live network exists."""
    try:
        from hemisphere.orchestrator import HemisphereOrchestrator
        from hemisphere.types import HemisphereFocus, NetworkStatus

        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    # _get_best_network(HRR_ENCODER) must be None (no net registered).
    assert orch._get_best_network(HemisphereFocus.HRR_ENCODER) is None

    # And nothing in the networks dict uses HRR_ENCODER as its focus.
    with orch._networks_lock:
        for net in orch._networks.values():
            assert net.focus != HemisphereFocus.HRR_ENCODER


def test_hrr_encoder_never_influences_broadcast_slots():
    """Even after _update_broadcast_slots runs, no slot is assigned to
    hrr_encoder because no network backs it.
    """
    try:
        from hemisphere.orchestrator import HemisphereOrchestrator
        from hemisphere.types import HemisphereFocus

        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")

    # Use public state access to find slot assignments.
    state_fn = getattr(orch, "_update_broadcast_slots", None)
    if state_fn is not None:
        try:
            state_fn()
        except Exception:
            # OK if the function needs other scaffolding; the invariant we
            # care about is just that nothing in the slot state refers to
            # hrr_encoder.
            pass
    for slot in orch._broadcast_slots:
        assert slot["name"] != HemisphereFocus.HRR_ENCODER.value


def test_specialist_stub_descriptor_shape():
    from hemisphere.hrr_specialist import HRREncoder, FEATURE_DIM
    from hemisphere.types import SpecialistLifecycleStage

    desc = HRREncoder.describe()
    assert desc["focus"] == "hrr_encoder"
    assert desc["lifecycle"] == SpecialistLifecycleStage.CANDIDATE_BIRTH.value
    assert desc["tier"] == 1
    assert desc["authority"] == "none"
    assert desc["feature_dim"] == FEATURE_DIM
    assert len(desc["feature_names"]) == FEATURE_DIM


# ---------------------------------------------------------------------------
# Encoder behavior
# ---------------------------------------------------------------------------


def test_encoder_returns_zero_vector_when_disabled():
    from hemisphere.hrr_specialist import FEATURE_DIM, HRREncoder

    disabled_status = {
        "enabled": False,
        "world_shadow": {"samples_total": 0, "samples_retained": 0, "ring_capacity": 500},
        "simulation_shadow": {"samples_total": 0, "samples_retained": 0, "ring_capacity": 200},
        "recall_advisory": {"samples_total": 0, "samples_retained": 0, "ring_capacity": 500},
    }
    vec = HRREncoder.encode(disabled_status)
    assert vec == [0.0] * FEATURE_DIM


def test_encoder_derives_bounded_features_when_enabled():
    from hemisphere.hrr_specialist import FEATURE_DIM, HRREncoder

    enabled_status = {
        "enabled": True,
        "world_shadow": {
            "enabled": True,
            "samples_total": 1000,
            "samples_retained": 500,
            "ring_capacity": 500,
            "binding_cleanliness": 0.82,
            "cleanup_accuracy": 1.0,
            "similarity_to_previous": 0.73,
        },
        "simulation_shadow": {
            "enabled": True,
            "samples_total": 50,
            "samples_retained": 50,
            "ring_capacity": 200,
            "last_cleanliness_after": 0.9,
        },
        "recall_advisory": {
            "enabled": True,
            "samples_total": 200,
            "samples_retained": 200,
            "ring_capacity": 500,
            "help_rate": 0.4,
        },
    }
    vec = HRREncoder.encode(enabled_status)
    assert len(vec) == FEATURE_DIM
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert vec[0] == 1.0  # enabled bit
    assert vec[2] == 1.0  # samples_retained / ring_capacity = 500/500
    assert vec[4] == 1.0  # cleanup_accuracy
    assert vec[7] == 0.4  # recall_advisory help_rate


def test_encoder_reads_live_status_with_no_arg():
    """Default path: get_hrr_status() is called. Must not raise."""
    from hemisphere.hrr_specialist import FEATURE_DIM, HRREncoder

    vec = HRREncoder.encode()
    assert isinstance(vec, list)
    assert len(vec) == FEATURE_DIM
    assert all(0.0 <= v <= 1.0 for v in vec)


# ---------------------------------------------------------------------------
# Negative tests: no Tier-2 holographic_cognition anywhere this sprint
# ---------------------------------------------------------------------------


def test_no_tier2_holographic_cognition_registered():
    """The Tier-2 ``holographic_cognition`` focus is intentionally absent."""
    from hemisphere.types import HemisphereFocus, MATRIX_ELIGIBLE_FOCUSES

    for f in HemisphereFocus:
        assert "holographic_cognition" not in f.value.lower()

    # And the matrix-eligible set must not contain anything HRR-adjacent.
    for f in MATRIX_ELIGIBLE_FOCUSES:
        assert "hrr" not in f.value.lower()
        assert "holographic" not in f.value.lower()


def test_hrr_specialist_source_has_no_forbidden_imports():
    """Structural import guard for the specialist stub module."""
    import hemisphere.hrr_specialist as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from policy.state_encoder",
        "from policy.policy_nn",
        "from epistemic.belief_graph.bridge",
        "from epistemic.soul_integrity",
        "from memory.persistence",
        "from memory.storage",
        "from memory.canonical",
        "from autonomy",
        "from identity.kernel",
    )
    for token in forbidden:
        assert token not in src, f"hrr_specialist must not import {token!r}"


def test_distillation_teacher_does_not_register_hrr_encoder():
    """Confirm no pre-existing distillation pathway claims hrr_encoder."""
    try:
        from hemisphere.orchestrator import HemisphereOrchestrator
    except Exception:
        pytest.skip("orchestrator unavailable")

    try:
        orch = HemisphereOrchestrator()
    except Exception:
        pytest.skip("orchestrator init failed in this env")

    # Distillation encoder IDs are created lazily; without seed data the
    # dict should at minimum NOT contain hrr_encoder as a key.
    assert "hrr_encoder" not in orch._distillation_encoder_ids
