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

def _stub(nets, slots=None, buffers=None):
    import threading
    from types import SimpleNamespace
    return SimpleNamespace(
        _networks={n.id: n for n in nets},
        _networks_lock=threading.Lock(),
        _broadcast_slots=slots or [],
        _matrix_signal_buffers=buffers or {},
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


# ── Phase M: autonomous birth ──

def test_matrix_label_bucketing():
    from hemisphere.orchestrator import HemisphereOrchestrator as H
    assert H._matrix_label(0.0) == 0
    assert H._matrix_label(0.24) == 0
    assert H._matrix_label(0.25) == 1
    assert H._matrix_label(0.5) == 2
    assert H._matrix_label(0.74) == 2
    assert H._matrix_label(0.75) == 3
    assert H._matrix_label(1.0) == 3


def test_autonomous_birth_gate_via_stub():
    import collections, threading
    from types import SimpleNamespace
    from hemisphere.orchestrator import HemisphereOrchestrator as H, MATRIX_BIRTH_MIN_SAMPLES
    from hemisphere.types import HemisphereFocus

    births = []
    sample = ([0.0] * 4, 0)
    stub = SimpleNamespace(
        _networks={}, _networks_lock=threading.Lock(),
        _matrix_signal_buffers={
            "speaker_profile": collections.deque([sample] * MATRIX_BIRTH_MIN_SAMPLES),
            "positive_memory": collections.deque([sample] * (MATRIX_BIRTH_MIN_SAMPLES - 1)),
        },
        count_probationary_specialists=lambda: len(births),
        create_probationary_specialist=(
            lambda focus, job_id="": (births.append(focus) or SimpleNamespace(focus=focus))
        ),
    )
    H._check_matrix_births(stub)
    # at-threshold focus births; under-threshold + never-observed focuses do not
    assert HemisphereFocus.SPEAKER_PROFILE in births
    assert HemisphereFocus.POSITIVE_MEMORY not in births
    assert HemisphereFocus.NEGATIVE_MEMORY not in births  # no buffer at all


def test_birth_skipped_when_specialist_exists_or_capped():
    import collections, threading
    from types import SimpleNamespace
    from hemisphere.orchestrator import HemisphereOrchestrator as H, MATRIX_BIRTH_MIN_SAMPLES
    from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage

    # cap reached -> no births at all
    births = []
    capped = SimpleNamespace(
        _networks={}, _networks_lock=threading.Lock(),
        _matrix_signal_buffers={"speaker_profile": collections.deque([([0.0], 0)] * MATRIX_BIRTH_MIN_SAMPLES)},
        count_probationary_specialists=lambda: 3,  # == MAX_PROBATIONARY_SPECIALISTS
        create_probationary_specialist=lambda focus, job_id="": births.append(focus),
    )
    H._check_matrix_births(capped)
    assert births == []

    # focus already has a live specialist -> not re-born
    born2 = []
    existing_net = SimpleNamespace(focus=HemisphereFocus.SPEAKER_PROFILE,
                                   specialist_lifecycle=SpecialistLifecycleStage.PROBATIONARY_TRAINING)
    has_existing = SimpleNamespace(
        _networks={"n": existing_net}, _networks_lock=threading.Lock(),
        _matrix_signal_buffers={"speaker_profile": collections.deque([([0.0], 0)] * MATRIX_BIRTH_MIN_SAMPLES)},
        count_probationary_specialists=lambda: 0,
        create_probationary_specialist=lambda focus, job_id="": born2.append(focus),
    )
    H._check_matrix_births(has_existing)
    assert HemisphereFocus.SPEAKER_PROFILE not in born2


def test_observe_accumulates_samples_via_stub():
    from types import SimpleNamespace
    from hemisphere.orchestrator import HemisphereOrchestrator as H

    class FakeEnc:
        @staticmethod
        def encode(ctx):
            return [0.1, 0.2, 0.3, 0.4]
        @staticmethod
        def compute_signal_value(ctx):
            return 0.8  # -> label 3

    stub = SimpleNamespace(
        _matrix_signal_buffers={},
        _matrix_label=H._matrix_label,
        _matrix_encoder_for=lambda focus: ({"ctx": 1}, FakeEnc),
    )
    H._observe_matrix_signals(stub)
    # all 5 eligible focuses accumulated exactly one (features, label) sample
    assert len(stub._matrix_signal_buffers) == 5
    feats, label = list(stub._matrix_signal_buffers["speaker_profile"])[0]
    assert feats == [0.1, 0.2, 0.3, 0.4] and label == 3
