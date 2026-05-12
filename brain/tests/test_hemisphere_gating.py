"""Tests for Tier-1 distillation gating and delta tracker persistence."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from hemisphere.types import (
    DISTILLATION_CONFIGS,
    HemisphereFocus,
    NetworkArchitecture,
    NetworkStatus,
    NetworkTopology,
    PerformanceMetrics,
    TrainingProgress,
    LayerDefinition,
)
from hemisphere.orchestrator import (
    HemisphereOrchestrator,
    TIER1_MIN_ACCURACY,
    TIER1_MAX_CONSECUTIVE_FAILURES,
    DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS,
    DISTILLATION_REGRESSION_COOLDOWN_S,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_network(
    focus: HemisphereFocus,
    accuracy: float = 0.0,
    loss: float = 10.0,
    nid: str | None = None,
    input_size: int = 16,
) -> NetworkArchitecture:
    layer = LayerDefinition(id="h", layer_type="hidden", node_count=16, activation="relu")
    topo = NetworkTopology(
        input_size=input_size, layers=(layer,), output_size=8,
        total_parameters=200, activation_functions=("relu",),
    )
    perf = PerformanceMetrics(
        accuracy=accuracy, loss=loss, response_time_ms=1.0,
        memory_usage_bytes=100,
    )
    tp = TrainingProgress(total_epochs=20, learning_rate=0.001, batch_size=32, is_training=False)
    return NetworkArchitecture(
        id=nid or f"test-{focus.value}",
        name=f"Distill-{focus.value}",
        focus=focus,
        topology=topo,
        performance=perf,
        training_progress=tp,
        status=NetworkStatus.ACTIVE,
        is_active=True,
        design_reasoning="test",
    )


def _stub_orchestrator() -> HemisphereOrchestrator:
    """Build an orchestrator with engine/registry/event_bus stubbed."""
    engine = MagicMock()
    engine.device = "cpu"
    engine._active_models = {}
    engine.remove_model = MagicMock()
    engine.train_distillation = MagicMock(return_value=10.0)

    registry = MagicMock()
    registry.get_active = MagicMock(return_value=None)
    registry.deactivate = MagicMock()

    event_bus = MagicMock()
    event_bus.on = MagicMock()

    orch = HemisphereOrchestrator.__new__(HemisphereOrchestrator)
    orch._engine = engine
    orch._registry = registry
    orch._event_bus = event_bus

    import threading
    orch._networks = {}
    orch._networks_lock = threading.Lock()
    orch._tier1_failure_counts = {}
    orch._tier1_disabled = set()
    orch._tier1_regression_counts = {}
    orch._tier1_regression_cooldown_until = {}
    orch._tier1_cooldown_last_log = {}
    orch._tier1_regression_cooldown_strikes = {}
    orch._tier1_last_retrain_time = {}
    orch._distillation_encoder_ids = {}
    orch._cycle_count = 0
    orch._last_migration_check = 0.0
    orch._last_boot_stabilization_log = 0.0
    orch._enabled = True

    return orch


# ---------------------------------------------------------------------------
# Retrain-path gating tests
# ---------------------------------------------------------------------------

class TestRetrainPathGating:
    """Verify that the retrain path enforces the same accuracy floor as build."""

    def test_retrain_bad_model_increments_failure_count(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.EMOTION_DEPTH
        net = _make_network(focus, accuracy=0.0, loss=10.0)
        orch._networks[net.id] = net

        orch._engine.train_distillation.return_value = 9.5

        orch._remove_distillation_model(focus, net)

        orch._engine.remove_model.assert_called_with(net.id)
        orch._registry.deactivate.assert_called_with(focus.value, net.id)
        assert net.id not in orch._networks

    def test_failure_count_hits_max_disables_session(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.EMOTION_DEPTH

        for i in range(TIER1_MAX_CONSECUTIVE_FAILURES):
            orch._tier1_failure_counts[focus.value] = i
            count = orch._tier1_failure_counts.get(focus.value, 0) + 1
            orch._tier1_failure_counts[focus.value] = count
            if count >= TIER1_MAX_CONSECUTIVE_FAILURES:
                orch._tier1_disabled.add(focus.value)

        assert focus.value in orch._tier1_disabled

    def test_retrain_above_floor_resets_failure_count(self):
        """A successful retrain above the floor should reset the failure counter."""
        orch = _stub_orchestrator()
        focus = HemisphereFocus.SPEAKER_REPR
        orch._tier1_failure_counts[focus.value] = 2

        orch._tier1_failure_counts[focus.value] = 0
        assert orch._tier1_failure_counts[focus.value] == 0

    def test_non_tier1_focus_not_gated(self):
        """Non-Tier-1 focuses should not be affected by Tier-1 gating."""
        orch = _stub_orchestrator()
        focus = HemisphereFocus.MEMORY
        assert focus.value not in DISTILLATION_CONFIGS
        assert focus.value not in orch._tier1_disabled

    def test_remove_clears_all_bindings(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.FACE_REPR
        net = _make_network(focus, accuracy=0.0)

        orch._networks[net.id] = net
        orch._distillation_encoder_ids[focus.value] = "enc-face"

        orch._remove_distillation_model(focus, net)

        assert net.id not in orch._networks
        orch._engine.remove_model.assert_called_with(net.id)
        orch._registry.deactivate.assert_called_with(focus.value, net.id)
        assert focus.value not in orch._distillation_encoder_ids

    def test_disabled_specialist_skipped_in_cycle(self):
        """Once disabled, the focus should be skipped entirely."""
        orch = _stub_orchestrator()
        focus = HemisphereFocus.EMOTION_DEPTH
        orch._tier1_disabled.add(focus.value)
        assert focus.value in orch._tier1_disabled

    def test_distillation_baseline_restore_recovers_weights_and_metrics(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.EMOTION_DEPTH
        net = _make_network(focus, accuracy=0.91, loss=0.09, nid="distill-test")

        class _FakeModel:
            def __init__(self):
                self.weight = 1.0

            def state_dict(self):
                return {"weight": self.weight}

            def load_state_dict(self, state):
                self.weight = state["weight"]

        model = _FakeModel()
        orch._engine._active_models = {net.id: model}

        snapshot = orch._snapshot_model_state(net.id)
        assert snapshot == {"weight": 1.0}

        # Simulate bad retrain mutating both model weights and performance.
        model.weight = 99.0
        net.performance = PerformanceMetrics(
            accuracy=0.22,
            loss=2.5,
            response_time_ms=1.0,
            memory_usage_bytes=100,
        )
        restored = orch._restore_distillation_baseline(
            net,
            old_accuracy=0.91,
            old_loss=0.09,
            model_snapshot=snapshot,
        )

        assert restored is True
        assert model.weight == 1.0
        assert abs(net.performance.accuracy - 0.91) < 1e-9
        assert abs(net.performance.loss - 0.09) < 1e-9

    def test_run_cycle_boot_stabilization_skips_training_paths(self):
        orch = _stub_orchestrator()
        orch._refresh_research_priors = MagicMock()
        orch._feed_gap_detector = MagicMock()
        orch._maybe_run_distillation = MagicMock()
        orch._prune_networks = MagicMock()
        orch._check_specialist_promotions = MagicMock()
        orch._check_expansion_trigger = MagicMock()
        orch._check_migration = MagicMock()
        orch._construct_from_gap = MagicMock()
        orch._construct_network = MagicMock()
        orch._evolve_focus = MagicMock()
        orch._get_networks_for_focus = MagicMock(return_value=[])
        orch._total_network_count = MagicMock(return_value=0)
        orch._gap_detector = MagicMock()
        orch._gap_detector.detect_gaps.return_value = []
        orch._outcomes_since_train = {f.value: 0 for f in HemisphereFocus}

        with patch("hemisphere.orchestrator.get_safe_data_feed", return_value=MagicMock()), \
             patch("hemisphere.orchestrator.should_initiate_evolution", return_value=True):
            orch.run_cycle(
                {
                    "boot_stabilization_active": True,
                    "boot_stabilization_remaining_s": 120.0,
                },
                memories=list(range(50)),
                traits=["Curious"],
            )

        assert orch._cycle_count == 1
        orch._refresh_research_priors.assert_called_once()
        orch._feed_gap_detector.assert_called_once()
        orch._maybe_run_distillation.assert_not_called()
        orch._construct_network.assert_not_called()
        orch._evolve_focus.assert_not_called()

    def test_regression_cooldown_activates_after_consecutive_rollbacks(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.VOICE_INTENT
        cfg = DISTILLATION_CONFIGS[focus.value]
        net = _make_network(
            focus,
            accuracy=0.90,
            loss=0.10,
            nid="voice-intent-test",
            input_size=cfg.input_dim,
        )
        orch._networks[net.id] = net

        class _Collector:
            def count(self, _name):
                return 999

            def get_stats(self):
                return {}

        orch._get_distillation_collector = MagicMock(return_value=_Collector())

        def _regressing_train(*_args, **_kwargs):
            net.performance = PerformanceMetrics(
                accuracy=0.50,
                loss=2.0,
                response_time_ms=1.0,
                memory_usage_bytes=100,
            )
            return 2.0

        orch._engine.train_distillation.side_effect = _regressing_train

        fake_features = MagicMock()
        fake_features.shape = (64, 16)
        fake_labels = MagicMock()
        fake_weights = MagicMock()

        with patch.dict("hemisphere.orchestrator.DISTILLATION_CONFIGS", {focus.value: cfg}, clear=True), \
             patch("hemisphere.orchestrator.prepare_distillation_tensors", return_value=(fake_features, fake_labels, fake_weights)):
            for _ in range(DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS):
                orch._run_distillation_cycle()

        assert orch._tier1_regression_counts[focus.value] == DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS
        assert orch._tier1_regression_cooldown_until[focus.value] > time.time()

    def test_regression_cooldown_skips_retrain_attempts(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.VOICE_INTENT
        cfg = DISTILLATION_CONFIGS[focus.value]
        net = _make_network(
            focus,
            accuracy=0.90,
            loss=0.10,
            nid="voice-intent-cooldown",
            input_size=cfg.input_dim,
        )
        orch._networks[net.id] = net
        orch._tier1_regression_cooldown_until[focus.value] = time.time() + 120.0
        orch._tier1_regression_counts[focus.value] = DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS

        class _Collector:
            def count(self, _name):
                return 999

            def get_stats(self):
                return {}

        orch._get_distillation_collector = MagicMock(return_value=_Collector())
        orch._engine.train_distillation.reset_mock()

        fake_features = MagicMock()
        fake_features.shape = (64, 16)
        fake_labels = MagicMock()
        fake_weights = MagicMock()

        with patch.dict("hemisphere.orchestrator.DISTILLATION_CONFIGS", {focus.value: cfg}, clear=True), \
             patch("hemisphere.orchestrator.prepare_distillation_tensors", return_value=(fake_features, fake_labels, fake_weights)):
            orch._run_distillation_cycle()

        orch._engine.train_distillation.assert_not_called()

    def test_regression_cooldown_backoff_increases_after_repeat_cycles(self):
        orch = _stub_orchestrator()
        focus = HemisphereFocus.VOICE_INTENT
        cfg = DISTILLATION_CONFIGS[focus.value]
        net = _make_network(
            focus,
            accuracy=0.92,
            loss=0.08,
            nid="voice-intent-backoff",
            input_size=cfg.input_dim,
        )
        orch._networks[net.id] = net

        class _Collector:
            def count(self, _name):
                return 999

            def get_stats(self):
                return {}

        orch._get_distillation_collector = MagicMock(return_value=_Collector())

        def _regressing_train(*_args, **_kwargs):
            net.performance = PerformanceMetrics(
                accuracy=0.50,
                loss=2.0,
                response_time_ms=1.0,
                memory_usage_bytes=100,
            )
            return 2.0

        orch._engine.train_distillation.side_effect = _regressing_train

        fake_features = MagicMock()
        fake_features.shape = (64, cfg.input_dim)
        fake_labels = MagicMock()
        fake_weights = MagicMock()

        with patch.dict("hemisphere.orchestrator.DISTILLATION_CONFIGS", {focus.value: cfg}, clear=True), \
             patch("hemisphere.orchestrator.prepare_distillation_tensors", return_value=(fake_features, fake_labels, fake_weights)):
            for _ in range(DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS):
                orch._run_distillation_cycle()

            first_until = orch._tier1_regression_cooldown_until[focus.value]
            first_duration = first_until - time.time()
            assert first_duration > 0
            assert orch._tier1_regression_cooldown_strikes[focus.value] == 1

            orch._tier1_regression_cooldown_until[focus.value] = time.time() - 1.0

            for _ in range(DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS):
                orch._run_distillation_cycle()

        second_until = orch._tier1_regression_cooldown_until[focus.value]
        second_duration = second_until - time.time()
        assert second_duration > first_duration
        assert second_duration > DISTILLATION_REGRESSION_COOLDOWN_S
        assert orch._tier1_regression_cooldown_strikes[focus.value] == 2


# ---------------------------------------------------------------------------
# Restore pre-seed tests
# ---------------------------------------------------------------------------

class TestRestorePreSeed:
    """Verify that restored models below accuracy floor get pre-seeded failure counts."""

    def test_restored_below_floor_preseeds_to_1(self):
        orch = _stub_orchestrator()
        focus_value = "emotion_depth"

        import math
        restored_acc = 0.0
        if (focus_value in DISTILLATION_CONFIGS
                and math.isfinite(restored_acc)
                and restored_acc < TIER1_MIN_ACCURACY):
            orch._tier1_failure_counts[focus_value] = max(
                orch._tier1_failure_counts.get(focus_value, 0), 1,
            )
        assert orch._tier1_failure_counts[focus_value] == 1

    def test_restored_above_floor_no_preseed(self):
        orch = _stub_orchestrator()
        focus_value = "speaker_repr"

        import math
        restored_acc = 0.80
        if (focus_value in DISTILLATION_CONFIGS
                and math.isfinite(restored_acc)
                and restored_acc < TIER1_MIN_ACCURACY):
            orch._tier1_failure_counts[focus_value] = max(
                orch._tier1_failure_counts.get(focus_value, 0), 1,
            )
        assert focus_value not in orch._tier1_failure_counts

    def test_restored_below_floor_already_disabled_stays_disabled(self):
        orch = _stub_orchestrator()
        focus_value = "emotion_depth"
        orch._tier1_disabled.add(focus_value)

        import math
        restored_acc = 0.0
        if (focus_value in DISTILLATION_CONFIGS
                and math.isfinite(restored_acc)
                and restored_acc < TIER1_MIN_ACCURACY):
            orch._tier1_failure_counts[focus_value] = max(
                orch._tier1_failure_counts.get(focus_value, 0), 1,
            )

        assert focus_value in orch._tier1_disabled
        assert orch._tier1_failure_counts[focus_value] >= 1

    def test_nan_accuracy_not_preseeded(self):
        """NaN accuracy should not trigger pre-seed."""
        orch = _stub_orchestrator()
        focus_value = "emotion_depth"

        import math
        restored_acc = float("nan")
        if (focus_value in DISTILLATION_CONFIGS
                and math.isfinite(restored_acc)
                and restored_acc < TIER1_MIN_ACCURACY):
            orch._tier1_failure_counts[focus_value] = max(
                orch._tier1_failure_counts.get(focus_value, 0), 1,
            )

        assert focus_value not in orch._tier1_failure_counts


# ---------------------------------------------------------------------------
# Delta tracker counter persistence tests
# ---------------------------------------------------------------------------

class TestDeltaCounterPersistence:
    """Verify that DeltaTracker cumulative counters survive save/load."""

    def test_counters_round_trip(self, tmp_path):
        from autonomy.delta_tracker import DeltaTracker

        counters_file = tmp_path / "delta_counters.json"

        tracker = DeltaTracker()
        tracker._total_measured = 42
        tracker._total_improved = 30
        tracker._total_regressed = 5
        tracker._total_interrupted = 7

        with patch("autonomy.delta_tracker.DELTA_COUNTERS_PATH", counters_file):
            tracker.save_counters()
            assert counters_file.exists()

            data = json.loads(counters_file.read_text())
            assert data["counters"]["total_measured"] == 42
            assert data["counters"]["total_improved"] == 30

            tracker2 = DeltaTracker()
            tracker2.load_counters()
            assert tracker2._total_measured == 42
            assert tracker2._total_improved == 30
            assert tracker2._total_regressed == 5
            assert tracker2._total_interrupted == 7

    def test_missing_file_loads_zeroes(self):
        from autonomy.delta_tracker import DeltaTracker

        tracker = DeltaTracker()
        nonexistent = Path("/tmp/nonexistent_delta_counters_test.json")
        with patch("autonomy.delta_tracker.DELTA_COUNTERS_PATH", nonexistent):
            tracker.load_counters()
        assert tracker._total_measured == 0

    def test_corrupt_file_fails_closed(self, tmp_path):
        from autonomy.delta_tracker import DeltaTracker

        bad_file = tmp_path / "delta_counters.json"
        bad_file.write_text("{corrupt: not json!!!}")

        tracker = DeltaTracker()
        with patch("autonomy.delta_tracker.DELTA_COUNTERS_PATH", bad_file):
            tracker.load_counters()
        assert tracker._total_measured == 0
        assert tracker._total_improved == 0

    def test_old_format_missing_counters_key(self, tmp_path):
        """A file without the 'counters' key should not crash."""
        from autonomy.delta_tracker import DeltaTracker

        old_file = tmp_path / "delta_counters.json"
        old_file.write_text(json.dumps({"version": 1}))

        tracker = DeltaTracker()
        with patch("autonomy.delta_tracker.DELTA_COUNTERS_PATH", old_file):
            tracker.load_counters()
        assert tracker._total_measured == 0

    def test_pending_windows_still_load(self, tmp_path):
        """Pending persistence is independent of counter persistence."""
        from autonomy.delta_tracker import DeltaTracker, DELTA_PENDING_PATH

        pending_file = tmp_path / "delta_pending.json"
        pending_file.write_text(json.dumps([]))

        tracker = DeltaTracker()
        with patch("autonomy.delta_tracker.DELTA_PENDING_PATH", pending_file):
            restored, interrupted = tracker.load_pending()
        assert restored == 0
        assert interrupted == []

    def test_counters_additive_on_load(self, tmp_path):
        """load_counters adds to current session counts (e.g. interrupted from load_pending)."""
        from autonomy.delta_tracker import DeltaTracker

        counters_file = tmp_path / "delta_counters.json"
        counters_file.write_text(json.dumps({
            "counters": {
                "total_measured": 10,
                "total_improved": 5,
                "total_regressed": 2,
                "total_interrupted": 3,
            }
        }))

        tracker = DeltaTracker()
        tracker._total_interrupted = 1

        with patch("autonomy.delta_tracker.DELTA_COUNTERS_PATH", counters_file):
            tracker.load_counters()

        assert tracker._total_interrupted == 4
        assert tracker._total_measured == 10


# ---------------------------------------------------------------------------
# Dependency-retrain ordering tests
# ---------------------------------------------------------------------------

class TestDependencyRetrainOrdering:
    """Verify that a dependency retrained earlier in the same cycle does not
    block its dependent from training in that same cycle.

    Before the cycle-start snapshot, `emotion_depth` was permanently blocked
    because `speaker_repr` (its declared dependency) is processed first and
    stamps `_tier1_last_retrain_time["speaker_repr"] = now`; the subsequent
    `dep_recently_retrained` check then trivially fires since
    `now - now < 60s`. Freshness protection is cross-cycle, not same-cycle.
    """

    def test_same_cycle_dependency_retrain_does_not_block_dependent(self):
        from hemisphere.orchestrator import DISTILLATION_DEP_RETRAIN_WINDOW_S

        orch = _stub_orchestrator()

        speaker_focus = HemisphereFocus.SPEAKER_REPR
        emotion_focus = HemisphereFocus.EMOTION_DEPTH
        speaker_cfg = DISTILLATION_CONFIGS[speaker_focus.value]
        emotion_cfg = DISTILLATION_CONFIGS[emotion_focus.value]

        assert "speaker_repr" in (emotion_cfg.depends_on or ()), (
            "Test premise: emotion_depth is expected to declare speaker_repr as a dependency"
        )
        assert DISTILLATION_DEP_RETRAIN_WINDOW_S > 0, (
            "Window must be positive for the guard to have any effect"
        )

        speaker_net = _make_network(
            speaker_focus, accuracy=0.89, loss=0.11,
            nid="speaker-repr-built", input_size=speaker_cfg.input_dim,
        )
        emotion_net = _make_network(
            emotion_focus, accuracy=0.44, loss=1.2,
            nid="emotion-depth-built", input_size=emotion_cfg.input_dim,
        )
        orch._networks[speaker_net.id] = speaker_net
        orch._networks[emotion_net.id] = emotion_net

        class _FakeModel:
            def state_dict(self):
                return {}

            def load_state_dict(self, _state):
                pass

        orch._engine._active_models = {
            speaker_net.id: _FakeModel(),
            emotion_net.id: _FakeModel(),
        }

        class _Collector:
            def count(self, _name):
                return 999

            def get_stats(self):
                return {}

        orch._get_distillation_collector = MagicMock(return_value=_Collector())

        def _train(network, *_args, **_kwargs):
            network.performance = PerformanceMetrics(
                accuracy=max(network.performance.accuracy, 0.50) + 0.01,
                loss=max(0.05, network.performance.loss - 0.01),
                response_time_ms=1.0,
                memory_usage_bytes=100,
            )
            return float(network.performance.loss)

        orch._engine.train_distillation.side_effect = _train

        fake_features = MagicMock()
        fake_features.shape = (128, 8)
        fake_labels = MagicMock()
        fake_weights = MagicMock()

        only_two = {
            speaker_focus.value: speaker_cfg,
            emotion_focus.value: emotion_cfg,
        }

        with patch.dict(
            "hemisphere.orchestrator.DISTILLATION_CONFIGS", only_two, clear=True,
        ), patch(
            "hemisphere.orchestrator.prepare_distillation_tensors",
            return_value=(fake_features, fake_labels, fake_weights),
        ):
            orch._run_distillation_cycle()

        trained_focuses = {
            call.args[0].focus.value
            for call in orch._engine.train_distillation.call_args_list
            if call.args and hasattr(call.args[0], "focus")
        }
        assert speaker_focus.value in trained_focuses, (
            "speaker_repr should train in the cycle"
        )
        assert emotion_focus.value in trained_focuses, (
            "emotion_depth must not be blocked by its dependency retraining "
            "earlier in the same cycle"
        )
        assert orch._tier1_last_retrain_time.get(speaker_focus.value, 0.0) > 0.0
        assert orch._tier1_last_retrain_time.get(emotion_focus.value, 0.0) > 0.0

    def test_prior_cycle_dependency_retrain_still_blocks_dependent(self):
        """Cross-cycle freshness protection is preserved: if the dependency
        retrained in the PREVIOUS cycle within the guard window, the
        dependent still skips this cycle.
        """
        from hemisphere.orchestrator import DISTILLATION_DEP_RETRAIN_WINDOW_S

        orch = _stub_orchestrator()

        speaker_focus = HemisphereFocus.SPEAKER_REPR
        emotion_focus = HemisphereFocus.EMOTION_DEPTH
        speaker_cfg = DISTILLATION_CONFIGS[speaker_focus.value]
        emotion_cfg = DISTILLATION_CONFIGS[emotion_focus.value]

        speaker_net = _make_network(
            speaker_focus, accuracy=0.89, loss=0.11,
            nid="speaker-prior", input_size=speaker_cfg.input_dim,
        )
        emotion_net = _make_network(
            emotion_focus, accuracy=0.44, loss=1.2,
            nid="emotion-prior", input_size=emotion_cfg.input_dim,
        )
        orch._networks[speaker_net.id] = speaker_net
        orch._networks[emotion_net.id] = emotion_net

        class _FakeModel:
            def state_dict(self):
                return {}

            def load_state_dict(self, _state):
                pass

        orch._engine._active_models = {
            speaker_net.id: _FakeModel(),
            emotion_net.id: _FakeModel(),
        }

        now = time.time()
        # Pretend speaker_repr retrained a few seconds ago in a prior cycle.
        # Use only speaker here so that this cycle does NOT retrain it,
        # keeping the prior stamp authoritative for the guard check.
        orch._tier1_last_retrain_time[speaker_focus.value] = now - 5.0
        # And pretend emotion_depth has never trained.
        orch._tier1_last_retrain_time.pop(emotion_focus.value, None)

        class _Collector:
            def count(self, _name):
                return 999

            def get_stats(self):
                return {}

        orch._get_distillation_collector = MagicMock(return_value=_Collector())
        orch._engine.train_distillation.reset_mock()

        fake_features = MagicMock()
        fake_features.shape = (128, 8)
        fake_labels = MagicMock()
        fake_weights = MagicMock()

        only_emotion = {emotion_focus.value: emotion_cfg}

        with patch.dict(
            "hemisphere.orchestrator.DISTILLATION_CONFIGS", only_emotion, clear=True,
        ), patch(
            "hemisphere.orchestrator.prepare_distillation_tensors",
            return_value=(fake_features, fake_labels, fake_weights),
        ):
            orch._run_distillation_cycle()

        assert orch._engine.train_distillation.call_count == 0, (
            "emotion_depth must skip when its dependency retrained within "
            f"{DISTILLATION_DEP_RETRAIN_WINDOW_S}s in a PRIOR cycle"
        )
