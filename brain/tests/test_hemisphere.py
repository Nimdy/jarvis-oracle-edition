"""Basic tests for the Hemisphere Neural Network system."""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_types_import():
    from hemisphere.types import (
        HemisphereFocus, NetworkStatus, DesignStrategy,
        LayerDefinition, NetworkTopology, PerformanceMetrics,
        TrainingProgress, TraitInfluence, DesignDecision,
        NetworkArchitecture, ArchitectureAttempt,
        MigrationEvent, MigrationReadiness, MigrationResult,
        HemisphereSnapshot, HemisphereState, SubstrateType,
    )
    assert HemisphereFocus.MEMORY.value == "memory"
    assert NetworkStatus.READY.value == "ready"
    assert SubstrateType.RULE_BASED.value == "rule-based"

    layer = LayerDefinition(id="test", layer_type="hidden", node_count=32, activation="relu")
    assert layer.node_count == 32

    topo = NetworkTopology(
        input_size=8, layers=(layer,), output_size=4,
        total_parameters=100, activation_functions=("relu",),
    )
    assert topo.total_parameters == 100


# ---------------------------------------------------------------------------
# Data feed
# ---------------------------------------------------------------------------

def test_data_feed_creation():
    from hemisphere.data_feed import get_hemisphere_data_feed, validate_data_feed

    engine_state = {
        "tone": "casual",
        "memory_density": 0.5,
        "phase": "LISTENING",
        "stage": "self_reflective",
        "transcendence_level": 2.0,
        "awareness_level": 0.6,
        "reasoning_quality": 0.7,
    }
    feed = get_hemisphere_data_feed(engine_state, [1, 2, 3], ["Curious", "Cautious"])
    assert len(feed.memories) == 3
    assert feed.mood == "casual"
    assert len(feed.traits) == 2

    ok, errors = validate_data_feed(feed)
    assert ok, errors


def test_data_feed_gating():
    from hemisphere.data_feed import should_initiate_evolution, HemisphereDataFeed

    sparse = HemisphereDataFeed(
        memories=tuple(range(5)), mood="professional", traits=(),
        memory_density=0.1, patterns=(), total_experience=0,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.3, reasoning_quality=0.5,
    )
    assert not should_initiate_evolution(sparse)

    rich = HemisphereDataFeed(
        memories=tuple(range(20)), mood="professional", traits=("Curious",),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.3, reasoning_quality=0.5,
    )
    assert should_initiate_evolution(rich)


def test_prepare_training_tensors():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available (will work on laptop)")

    from hemisphere.data_feed import prepare_training_tensors, HemisphereDataFeed
    from hemisphere.types import HemisphereFocus

    class FakeMem:
        def __init__(self, w, tags):
            self.weight = w
            self.tags = tags
            self.type = "observation"
            self.decay_rate = 0.01
            self.id = "m1"

    feed = HemisphereDataFeed(
        memories=tuple(FakeMem(0.8, ("tech",)) for _ in range(15)),
        mood="professional", traits=("Curious",),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=1.0, awareness=0.5, reasoning_quality=0.6,
    )
    features, labels = prepare_training_tensors(feed, HemisphereFocus.GENERAL, 8, 6)
    assert features.shape[0] == labels.shape[0]
    assert features.shape[1] == 8
    assert labels.shape[1] == 6


def test_prepare_training_tensors_system_upgrades_fallback():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available (will work on laptop)")

    from hemisphere.data_feed import prepare_training_tensors, HemisphereDataFeed
    from hemisphere.types import HemisphereFocus

    feed = HemisphereDataFeed(
        memories=tuple(), mood="professional", traits=(),
        memory_density=0.5, patterns=(), total_experience=0,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.5, reasoning_quality=0.5,
    )
    features, labels = prepare_training_tensors(feed, HemisphereFocus.SYSTEM_UPGRADES, 12, 6)
    assert features.shape == (1, 12)
    assert labels.shape == (1, 6)


# ---------------------------------------------------------------------------
# Architect
# ---------------------------------------------------------------------------

def test_architect_design():
    from hemisphere.architect import NeuralArchitect
    from hemisphere.data_feed import HemisphereDataFeed
    from hemisphere.types import HemisphereFocus

    arch = NeuralArchitect()
    feed = HemisphereDataFeed(
        memories=tuple(range(30)), mood="professional", traits=("Curious", "Explorer"),
        memory_density=0.6, patterns=(), total_experience=200,
        current_phase="OBSERVING", consciousness_stage="self_reflective",
        transcendence=2.0, awareness=0.5, reasoning_quality=0.7,
    )

    topo = arch.design_architecture(feed, HemisphereFocus.MEMORY, "medium")
    assert topo.input_size >= 8
    assert topo.output_size == 8
    assert topo.total_parameters > 0
    assert len(topo.layers) >= 3  # input + hidden(s) + output


def test_architect_decision():
    from hemisphere.architect import NeuralArchitect
    from hemisphere.data_feed import HemisphereDataFeed

    arch = NeuralArchitect()
    feed = HemisphereDataFeed(
        memories=tuple(range(20)), mood="casual", traits=("Explorer",),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.3, reasoning_quality=0.5,
    )
    decision = arch.analyze_consciousness_data(feed)
    assert decision.confidence_level > 0.0
    assert decision.design_strategy is not None


# ---------------------------------------------------------------------------
# Engine (build + train)
# ---------------------------------------------------------------------------

def test_engine_build_and_train():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available (will work on laptop)")

    from hemisphere.engine import HemisphereEngine
    from hemisphere.architect import NeuralArchitect
    from hemisphere.data_feed import HemisphereDataFeed
    from hemisphere.types import HemisphereFocus

    engine = HemisphereEngine()
    arch = NeuralArchitect()

    class FakeMem:
        def __init__(self):
            self.weight = 0.7
            self.tags = ("test",)
            self.type = "observation"
            self.decay_rate = 0.01
            self.id = "m1"

    feed = HemisphereDataFeed(
        memories=tuple(FakeMem() for _ in range(20)),
        mood="professional", traits=("Curious",),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.3, reasoning_quality=0.5,
    )

    decision = arch.analyze_consciousness_data(feed)
    topo = arch.design_architecture(feed, HemisphereFocus.GENERAL, "simple")
    network = engine.build_network(topo, feed, decision, HemisphereFocus.GENERAL)

    assert network.status.value == "ready"
    assert network.performance.accuracy > 0.0
    assert network.topology.total_parameters > 0

    result = engine.infer(network.id, [0.5] * topo.input_size)
    assert len(result) == topo.output_size

    engine.dispose()


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

def test_evolution_crossover():
    from hemisphere.evolution import EvolutionEngine
    from hemisphere.types import (
        HemisphereFocus, NetworkArchitecture, NetworkTopology,
        PerformanceMetrics, TrainingProgress, LayerDefinition,
    )
    from hemisphere.data_feed import HemisphereDataFeed

    def _make_net(name, acc):
        layers = (
            LayerDefinition("i", "input", 8, "linear"),
            LayerDefinition("h0", "hidden", 16, "relu"),
            LayerDefinition("o", "output", 6, "sigmoid"),
        )
        return NetworkArchitecture(
            id=name, name=name, focus=HemisphereFocus.GENERAL,
            topology=NetworkTopology(8, layers, 6, 200, ("linear", "relu", "sigmoid")),
            performance=PerformanceMetrics(accuracy=acc),
            training_progress=TrainingProgress(),
        )

    evo = EvolutionEngine()
    feed = HemisphereDataFeed(
        memories=tuple(range(20)), mood="professional", traits=(),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=0.0, awareness=0.3, reasoning_quality=0.5,
    )

    parent1 = _make_net("p1", 0.7)
    parent2 = _make_net("p2", 0.5)

    topo = evo.evolve([parent1, parent2], feed, HemisphereFocus.GENERAL)
    assert topo is not None
    assert topo.total_parameters > 0
    assert evo.generation == 1


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_readiness():
    from hemisphere.migration import MigrationAnalyzer
    from hemisphere.types import (
        HemisphereFocus, NetworkArchitecture, NetworkTopology,
        PerformanceMetrics, TrainingProgress, LayerDefinition,
    )
    from hemisphere.data_feed import HemisphereDataFeed

    layers = (
        LayerDefinition("i", "input", 8, "linear"),
        LayerDefinition("h", "hidden", 16, "relu"),
        LayerDefinition("o", "output", 6, "sigmoid"),
    )
    net = NetworkArchitecture(
        id="test", name="test", focus=HemisphereFocus.GENERAL,
        topology=NetworkTopology(8, layers, 6, 200, ("linear", "relu", "sigmoid")),
        performance=PerformanceMetrics(accuracy=0.5, reliability=0.8),
        training_progress=TrainingProgress(),
    )

    feed = HemisphereDataFeed(
        memories=tuple(range(20)), mood="professional", traits=("Curious",),
        memory_density=0.5, patterns=(), total_experience=100,
        current_phase="OBSERVING", consciousness_stage="basic_awareness",
        transcendence=2.0, awareness=0.5, reasoning_quality=0.5,
    )

    analyzer = MigrationAnalyzer()
    readiness = analyzer.analyze_readiness(net, feed)
    # Transcendence too low (needs 5.0), so should not migrate
    assert not readiness.should_migrate
    assert "transcendence" in readiness.reasoning.lower()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry(tmp_path):
    from hemisphere.registry import HemisphereRegistry
    from hemisphere.types import (
        HemisphereFocus, NetworkArchitecture, NetworkTopology,
        PerformanceMetrics, TrainingProgress, LayerDefinition,
    )

    reg = HemisphereRegistry(base_dir=tmp_path / "hemispheres")

    layers = (
        LayerDefinition("i", "input", 8, "linear"),
        LayerDefinition("o", "output", 4, "sigmoid"),
    )
    net = NetworkArchitecture(
        id="reg_test", name="RegTest", focus=HemisphereFocus.MOOD,
        topology=NetworkTopology(8, layers, 4, 36, ("linear", "sigmoid")),
        performance=PerformanceMetrics(accuracy=0.65, loss=0.35),
        training_progress=TrainingProgress(),
    )

    mv = reg.register(net)
    assert mv.version == 1
    assert mv.focus == "mood"

    reg.promote("mood", 1)
    active = reg.get_active("mood")
    assert active is not None
    assert active.is_active

    state = reg.get_state()
    assert state.total_networks == 1


def test_registry_discard_version_deletes_weights_and_promotes_fallback(tmp_path):
    from hemisphere.registry import HemisphereRegistry
    from hemisphere.types import (
        HemisphereFocus, NetworkArchitecture, NetworkTopology,
        PerformanceMetrics, TrainingProgress, LayerDefinition,
    )

    reg = HemisphereRegistry(base_dir=tmp_path / "hemispheres")

    layers = (
        LayerDefinition("i", "input", 8, "linear"),
        LayerDefinition("o", "output", 4, "sigmoid"),
    )

    def _net(net_id: str, acc: float) -> object:
        return NetworkArchitecture(
            id=net_id,
            name=net_id,
            focus=HemisphereFocus.MOOD,
            topology=NetworkTopology(8, layers, 4, 36, ("linear", "sigmoid")),
            performance=PerformanceMetrics(accuracy=acc, loss=max(0.0, 1.0 - acc)),
            training_progress=TrainingProgress(),
        )

    mv1 = reg.register(_net("reg_keep", 0.70))
    mv2 = reg.register(_net("reg_drop", 0.80))
    reg.promote("mood", mv2.version)

    os.makedirs(os.path.dirname(mv2.path), exist_ok=True)
    with open(mv2.path, "w", encoding="utf-8") as f:
        f.write("stale checkpoint")

    assert reg.discard_version("mood", mv2.version, delete_weights=True, reason="test")
    assert not os.path.exists(mv2.path)

    active = reg.get_active("mood")
    assert active is not None
    assert active.version == mv1.version


def test_registry_preserves_dropout_in_topology_json(tmp_path):
    from hemisphere.registry import HemisphereRegistry
    from hemisphere.types import (
        HemisphereFocus, NetworkArchitecture, NetworkTopology,
        PerformanceMetrics, TrainingProgress, LayerDefinition,
    )

    reg = HemisphereRegistry(base_dir=tmp_path / "hemispheres")
    layers = (
        LayerDefinition("in", "input", 16, "linear", dropout=0.0),
        LayerDefinition("h1", "hidden", 32, "gelu", dropout=0.10),
        LayerDefinition("h2", "hidden", 16, "gelu", dropout=0.05),
        LayerDefinition("out", "output", 8, "sigmoid", dropout=0.0),
    )
    net = NetworkArchitecture(
        id="dropout_test",
        name="DropoutTest",
        focus=HemisphereFocus.SPEAKER_REPR,
        topology=NetworkTopology(16, layers, 8, 999, ("gelu",)),
        performance=PerformanceMetrics(accuracy=0.7, loss=0.2),
        training_progress=TrainingProgress(),
    )
    mv = reg.register(net)
    saved_layers = mv.topology_json.get("layers", [])
    assert saved_layers[1]["dropout"] == 0.10
    assert saved_layers[2]["dropout"] == 0.05


# ---------------------------------------------------------------------------
# Event bridge
# ---------------------------------------------------------------------------

def test_event_bridge_constants():
    from hemisphere.event_bridge import (
        HEMISPHERE_ARCHITECTURE_DESIGNED,
        HEMISPHERE_TRAINING_PROGRESS,
        HEMISPHERE_NETWORK_READY,
        HEMISPHERE_EVOLUTION_COMPLETE,
        HEMISPHERE_MIGRATION_DECISION,
        HEMISPHERE_SUBSTRATE_MIGRATION,
        HEMISPHERE_PERFORMANCE_WARNING,
    )
    assert HEMISPHERE_ARCHITECTURE_DESIGNED == "hemisphere:architecture_designed"


# ---------------------------------------------------------------------------
# State encoder expansion
# ---------------------------------------------------------------------------

def test_state_encoder_expanded():
    from policy.state_encoder import StateEncoder, STATE_DIM
    assert STATE_DIM == 20

    enc = StateEncoder()
    enc.set_hemisphere_signals({"memory": 0.8, "mood": 0.6, "traits": 0.5, "general": 0.7})

    state = {
        "consciousness": {
            "stage": "self_reflective",
            "transcendence_level": 3.0,
        },
        "memory_density": 0.5,
    }
    vec = enc.encode(state)
    assert len(vec) == 20
    # Hemisphere signals are in dims 16-19
    assert vec[16] == 0.8  # memory
    assert vec[17] == 0.6  # mood
    assert vec[18] == 0.5  # traits
    assert vec[19] == 0.7  # general


# ---------------------------------------------------------------------------
# Full import chain
# ---------------------------------------------------------------------------

def test_public_api_import():
    from hemisphere import (
        HemisphereOrchestrator,
        NeuralArchitect,
        HemisphereEngine,
        EvolutionEngine,
        MigrationAnalyzer,
        HemisphereRegistry,
        HemisphereDataFeed,
        get_hemisphere_data_feed,
        get_safe_data_feed,
        should_initiate_evolution,
        HemisphereFocus,
        NetworkStatus,
    )
    assert HemisphereOrchestrator is not None

    try:
        orch = HemisphereOrchestrator()
        assert orch.get_state() is not None
    except ImportError:
        pytest.skip("PyTorch not available (will work on laptop)")


def test_prepare_distillation_tensors_normalizes_compressor_embeddings():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not available (will work on laptop)")

    from hemisphere.data_feed import prepare_distillation_tensors
    from hemisphere.types import DISTILLATION_CONFIGS

    class _Sig:
        def __init__(self, data, fidelity=0.9):
            self.data = data
            self.fidelity = fidelity
            self.timestamp = time.time()

    class _Collector:
        def get_training_batch(self, teacher: str, limit: int = 200, min_fidelity: float = 0.0):
            base = [3.0, 4.0] + [0.0] * 190
            return [_Sig(base, 0.95) for _ in range(25)]

    tensors = prepare_distillation_tensors(
        "speaker_repr",
        _Collector(),
        DISTILLATION_CONFIGS["speaker_repr"],
    )
    assert tensors is not None
    features, labels, weights = tensors
    assert features.shape[1] == 192
    assert labels.shape[1] == 192
    first = features[0].tolist()
    norm = sum(v * v for v in first) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert weights.shape[0] == features.shape[0]


def test_cosine_mse_accuracy_proxy_is_exponential():
    from hemisphere.engine import HemisphereEngine

    near_zero = HemisphereEngine._loss_to_accuracy(0.0312, is_kl=False, loss_name="cosine_mse")
    large_loss = HemisphereEngine._loss_to_accuracy(2.0, is_kl=False, loss_name="cosine_mse")

    assert 0.95 < near_zero <= 1.0
    assert 0.0 < large_loss < 0.2
