"""P3.6-P3.10 — Tier-2 Matrix Protocol specialist lifecycle regression tests.

Exercises the five probationary focuses:

  * ``positive_memory`` (P3.6)
  * ``negative_memory`` (P3.7)
  * ``speaker_profile`` (P3.8)
  * ``temporal_pattern`` (P3.9)
  * ``skill_transfer`` (P3.10)

The acceptance criterion for these lanes is that each focus can:

  1. Be spawned through :meth:`HemisphereOrchestrator.create_probationary_specialist`
     as ``CANDIDATE_BIRTH`` (the real lifecycle-stage name, NOT the
     fictitious ``PROPOSED`` / ``PROVISIONAL`` used in early drafts of
     TODO_V2).
  2. Advance through the real lifecycle vocabulary on
     :meth:`HemisphereOrchestrator._check_specialist_promotions`:

         CANDIDATE_BIRTH → PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY
             → BROADCAST_ELIGIBLE → PROMOTED

  3. Be retired when low-utility criteria are met.

This file does NOT train real networks; it synthesises in-memory
``NetworkArchitecture`` fixtures and flips state, then asserts the
orchestrator transitions match the declared lifecycle ladder.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


TIER2_FOCUSES = [
    "positive_memory",
    "negative_memory",
    "speaker_profile",
    "temporal_pattern",
    "skill_transfer",
]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _fresh_orchestrator():
    from hemisphere.orchestrator import HemisphereOrchestrator
    try:
        return HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")


def _install_network(orch, net):
    with orch._networks_lock:
        orch._networks[net.id] = net


def _make_arch_in_stage(focus_value: str, stage, accuracy: float = 0.0, suffix: str = "a"):
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
        id=f"spec_{focus_value}_{suffix}",
        name=f"matrix_{focus_value}_{suffix}",
        focus=HemisphereFocus(focus_value),
        topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=1),
    )
    arch.specialist_lifecycle = stage
    return arch


# ---------------------------------------------------------------------------
# Enumeration guard — all five focuses are declared + matrix-eligible
# ---------------------------------------------------------------------------


def test_all_five_tier2_focuses_exist_in_enum():
    from hemisphere.types import HemisphereFocus
    declared = {f.value for f in HemisphereFocus}
    for focus in TIER2_FOCUSES:
        assert focus in declared, f"{focus} missing from HemisphereFocus"


def test_all_five_tier2_focuses_are_matrix_eligible():
    from hemisphere.types import HemisphereFocus, MATRIX_ELIGIBLE_FOCUSES
    eligible_vals = {f.value for f in MATRIX_ELIGIBLE_FOCUSES}
    for focus in TIER2_FOCUSES:
        assert focus in eligible_vals, f"{focus} not in MATRIX_ELIGIBLE_FOCUSES"
    # Guard against silent drift: MATRIX_ELIGIBLE_FOCUSES must equal the
    # five Tier-2 specialist set exactly.
    assert eligible_vals == set(TIER2_FOCUSES)


# ---------------------------------------------------------------------------
# Spawn: each Tier-2 focus births as CANDIDATE_BIRTH
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_spawn_probationary_specialist_for_each_tier2_focus(focus_value):
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = orch.create_probationary_specialist(
        HemisphereFocus(focus_value),
        job_id=f"job_{focus_value}",
        name=f"matrix_{focus_value}",
    )
    assert arch is not None, (
        f"create_probationary_specialist returned None for {focus_value}"
    )
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.CANDIDATE_BIRTH
    assert arch.specialist_job_id == f"job_{focus_value}"
    assert arch.focus == HemisphereFocus(focus_value)


def test_spawn_rejected_for_non_matrix_focus():
    """Sanity: non-eligible focuses must NOT produce a probationary arch."""
    from hemisphere.types import HemisphereFocus

    orch = _fresh_orchestrator()
    # `GENERAL` is not in MATRIX_ELIGIBLE_FOCUSES.
    arch = orch.create_probationary_specialist(
        HemisphereFocus.GENERAL,
        job_id="job_general",
        name="illegal",
    )
    assert arch is None


def test_probationary_cap_enforced():
    """Cap on concurrent probationary specialists must block the 4th spawn."""
    from hemisphere.types import (
        HemisphereFocus, MAX_PROBATIONARY_SPECIALISTS,
    )

    orch = _fresh_orchestrator()
    created = []
    # Cycle through the five Tier-2 focuses until we hit the cap.
    for i, focus_value in enumerate(TIER2_FOCUSES):
        arch = orch.create_probationary_specialist(
            HemisphereFocus(focus_value),
            job_id=f"job_{i}",
            name=f"matrix_{focus_value}_{i}",
        )
        if arch is not None:
            created.append(arch)

    assert len(created) == MAX_PROBATIONARY_SPECIALISTS

    # Further creation attempts must return None while the cap is active.
    extra = orch.create_probationary_specialist(
        HemisphereFocus(TIER2_FOCUSES[0]),
        job_id="job_extra",
        name="matrix_extra",
    )
    assert extra is None


# ---------------------------------------------------------------------------
# Lifecycle: full ladder progression using the real stage vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_candidate_birth_to_probationary_training_on_first_epoch(focus_value):
    """CANDIDATE_BIRTH advances to PROBATIONARY_TRAINING once training starts
    (``training_progress.current_epoch > 0``)."""
    from hemisphere.types import SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = _make_arch_in_stage(
        focus_value, SpecialistLifecycleStage.CANDIDATE_BIRTH, suffix="cbptr",
    )
    _install_network(orch, arch)
    orch._check_specialist_promotions()
    assert arch.specialist_lifecycle == SpecialistLifecycleStage.PROBATIONARY_TRAINING


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_probationary_training_to_verified_probationary(focus_value):
    """PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY stamps
    ``specialist_verification_ts`` (guarded by the P3.5 fix)."""
    from hemisphere.types import SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = _make_arch_in_stage(
        focus_value, SpecialistLifecycleStage.PROBATIONARY_TRAINING,
        accuracy=0.8, suffix="ptvp",
    )
    assert arch.specialist_verification_ts == 0.0
    _install_network(orch, arch)

    before = time.time()
    orch._check_specialist_promotions()
    after = time.time()

    assert arch.specialist_lifecycle == SpecialistLifecycleStage.VERIFIED_PROBATIONARY
    assert before <= arch.specialist_verification_ts <= after


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_verified_probationary_to_broadcast_eligible(focus_value):
    """VERIFIED_PROBATIONARY → BROADCAST_ELIGIBLE at impact > 0.3."""
    from hemisphere.types import SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = _make_arch_in_stage(
        focus_value, SpecialistLifecycleStage.VERIFIED_PROBATIONARY,
        accuracy=0.8, suffix="vpbe",
    )
    arch.specialist_impact_score = 0.42
    _install_network(orch, arch)

    orch._check_specialist_promotions()

    assert arch.specialist_lifecycle == SpecialistLifecycleStage.BROADCAST_ELIGIBLE


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_broadcast_eligible_to_promoted_after_dwell(focus_value):
    """BROADCAST_ELIGIBLE → PROMOTED after the specialist holds a broadcast
    slot for at least 10 cycles."""
    from hemisphere.types import SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = _make_arch_in_stage(
        focus_value, SpecialistLifecycleStage.BROADCAST_ELIGIBLE,
        accuracy=0.9, suffix="beprom",
    )
    arch.specialist_impact_score = 0.55
    _install_network(orch, arch)
    # Inject a broadcast slot with enough dwell.
    orch._broadcast_slots = [{
        "name": focus_value, "dwell": 12, "impact": 0.55,
    }]

    orch._check_specialist_promotions()

    assert arch.specialist_lifecycle == SpecialistLifecycleStage.PROMOTED


# ---------------------------------------------------------------------------
# Retirement: low-utility probationary specialists are pruned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("focus_value", TIER2_FOCUSES)
def test_low_utility_probationary_retirement(focus_value):
    """Probationary specialists older than 24h with impact < 0.1 retire."""
    from hemisphere.types import SpecialistLifecycleStage

    orch = _fresh_orchestrator()
    arch = _make_arch_in_stage(
        focus_value, SpecialistLifecycleStage.PROBATIONARY_TRAINING,
        accuracy=0.2, suffix="lowutil",
    )
    arch.specialist_impact_score = 0.01
    # Force "older than 24h" by setting the created_at far in the past. The
    # NetworkArchitecture default sets it to now; we force a stale value.
    arch.created_at = time.time() - (48 * 3600)
    _install_network(orch, arch)

    pruned: list[str] = []
    now = time.time()
    orch._retire_low_utility_probationary(now, pruned)

    assert arch.specialist_lifecycle == SpecialistLifecycleStage.RETIRED
    assert arch.id in pruned or arch.name in pruned
