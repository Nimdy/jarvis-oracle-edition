"""Matrix Protocol observability — `HemisphereOrchestrator.matrix_report()`.

The read-only Tier-2 lifecycle view that makes the Matrix Protocol verifiable:
per-specialist stage + the next gate to clear (have/need/met), and `not_born`
for eligible focuses with no specialist yet (absence made visible, not silent).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _orch():
    from hemisphere.orchestrator import HemisphereOrchestrator
    try:
        return HemisphereOrchestrator()
    except Exception:
        pytest.skip("Cannot instantiate orchestrator in this environment")


def _arch(focus_value, stage, *, accuracy=0.0, impact=0.0, epoch=1, suffix="a"):
    from hemisphere.types import (
        NetworkArchitecture, NetworkTopology, LayerDefinition,
        PerformanceMetrics, TrainingProgress, HemisphereFocus,
    )
    layer = LayerDefinition(id="h1", layer_type="hidden", node_count=8, activation="relu")
    topo = NetworkTopology(
        input_size=8, layers=(layer,), output_size=4,
        total_parameters=100, activation_functions=("relu",),
    )
    arch = NetworkArchitecture(
        id=f"spec_{focus_value}_{suffix}", name=f"matrix_{focus_value}_{suffix}",
        focus=HemisphereFocus(focus_value), topology=topo,
        performance=PerformanceMetrics(accuracy=accuracy),
        training_progress=TrainingProgress(current_epoch=epoch),
    )
    arch.specialist_lifecycle = stage
    arch.specialist_impact_score = impact
    return arch


def _install(orch, net):
    with orch._networks_lock:
        orch._networks[net.id] = net


def test_empty_report_lists_all_eligible_as_not_born():
    orch = _orch()
    r = orch.matrix_report()
    assert set(r["not_born"]) == set(r["eligible_focuses"])
    assert len(r["eligible_focuses"]) == 5
    assert not r["specialists"]
    assert r["promoted_count"] == 0
    assert r["expansion_min_promoted"] >= 1


def test_specialist_appears_with_stage_and_next_gate():
    from hemisphere.types import SpecialistLifecycleStage as S
    orch = _orch()
    _install(orch, _arch("speaker_profile", S.PROBATIONARY_TRAINING, accuracy=0.42))
    r = orch.matrix_report()
    assert "speaker_profile" not in r["not_born"]
    spec = next(s for s in r["specialists"] if s["focus"] == "speaker_profile")
    assert spec["lifecycle"] == "probationary_training"
    assert spec["next_gate"]["gate"] == "accuracy>0.5"
    assert spec["next_gate"]["met"] is False  # 0.42 < 0.5


def test_gate_met_flips_true_above_threshold():
    from hemisphere.types import SpecialistLifecycleStage as S
    orch = _orch()
    _install(orch, _arch("temporal_pattern", S.PROBATIONARY_TRAINING, accuracy=0.6))
    spec = next(s for s in orch.matrix_report()["specialists"]
                if s["focus"] == "temporal_pattern")
    assert spec["next_gate"]["met"] is True


def test_verified_probationary_gate_is_impact():
    from hemisphere.types import SpecialistLifecycleStage as S
    orch = _orch()
    _install(orch, _arch("positive_memory", S.VERIFIED_PROBATIONARY, impact=0.1))
    spec = next(s for s in orch.matrix_report()["specialists"]
                if s["focus"] == "positive_memory")
    assert spec["next_gate"]["gate"] == "impact>0.3"
    assert spec["next_gate"]["met"] is False


def test_candidate_birth_gate_is_training_started():
    from hemisphere.types import SpecialistLifecycleStage as S
    orch = _orch()
    _install(orch, _arch("negative_memory", S.CANDIDATE_BIRTH, epoch=0))
    spec = next(s for s in orch.matrix_report()["specialists"]
                if s["focus"] == "negative_memory")
    assert spec["next_gate"]["gate"] == "training_started"
    assert spec["next_gate"]["met"] is False  # epoch 0 → not started


def test_by_stage_and_promoted_count():
    from hemisphere.types import SpecialistLifecycleStage as S
    orch = _orch()
    _install(orch, _arch("positive_memory", S.PROMOTED, suffix="p"))
    _install(orch, _arch("skill_transfer", S.CANDIDATE_BIRTH, epoch=0, suffix="c"))
    r = orch.matrix_report()
    assert r["by_stage"].get("promoted") == 1
    assert r["promoted_count"] == 1
    assert "positive_memory" not in r["not_born"]
    assert "skill_transfer" not in r["not_born"]


# ---------------------------------------------------------------------------
# Stub-based coverage — exercises matrix_report() WITHOUT a full orchestrator
# (NetworkArchitecture is a pure dataclass; only HemisphereOrchestrator.__init__
# needs torch/device, so we call the unbound method against a minimal `self`).
# This runs in any environment, unlike the orchestrator-backed tests above.
# ---------------------------------------------------------------------------

def _stub(nets, slots=None):
    import threading
    from types import SimpleNamespace
    return SimpleNamespace(
        _networks={n.id: n for n in nets},
        _networks_lock=threading.Lock(),
        _broadcast_slots=slots or [],
    )


def test_matrix_report_logic_via_stub():
    from hemisphere.orchestrator import HemisphereOrchestrator
    from hemisphere.types import SpecialistLifecycleStage as S
    nets = [
        _arch("speaker_profile", S.PROBATIONARY_TRAINING, accuracy=0.42),
        _arch("positive_memory", S.PROMOTED, suffix="p"),
        _arch("temporal_pattern", S.BROADCAST_ELIGIBLE, impact=0.5, suffix="b"),
    ]
    slots = [{"name": "temporal_pattern", "dwell": 4}]
    r = HemisphereOrchestrator.matrix_report(_stub(nets, slots))

    assert r["promoted_count"] == 1
    assert set(r["not_born"]) == {"negative_memory", "skill_transfer"}

    sp = next(s for s in r["specialists"] if s["focus"] == "speaker_profile")
    assert sp["next_gate"] == {"gate": "accuracy>0.5", "have": 0.42, "met": False}

    tp = next(s for s in r["specialists"] if s["focus"] == "temporal_pattern")
    # broadcast_eligible: in a slot but dwell 4 < 10 -> gate not met
    assert tp["next_gate"]["gate"] == "broadcast dwell>=10"
    assert tp["next_gate"]["in_broadcast"] is True
    assert tp["next_gate"]["have"] == 4
    assert tp["next_gate"]["met"] is False
    assert tp["broadcast_dwell"] == 4


def test_stub_promoted_dwell_gate_met():
    from hemisphere.orchestrator import HemisphereOrchestrator
    from hemisphere.types import SpecialistLifecycleStage as S
    nets = [_arch("skill_transfer", S.BROADCAST_ELIGIBLE, impact=0.6)]
    slots = [{"name": "skill_transfer", "dwell": 12}]
    r = HemisphereOrchestrator.matrix_report(_stub(nets, slots))
    st = r["specialists"][0]
    assert st["next_gate"]["met"] is True  # in slot + dwell 12 >= 10
