"""Hemisphere Neural Network System -- public API.

The brain designs, builds, trains, evolves, and (eventually) migrates to
specialized neural networks organized by cognitive focus area.
"""

from hemisphere.types import (
    DistillationConfig,
    DISTILLATION_CONFIGS,
    HemisphereFocus,
    NetworkStatus,
    DesignStrategy,
    NetworkArchitecture,
    NetworkTopology,
    PerformanceMetrics,
    TrainingProgress,
    HemisphereState,
    HemisphereSnapshot,
    MigrationEvent,
    MigrationReadiness,
    MigrationResult,
)
from hemisphere.orchestrator import HemisphereOrchestrator
from hemisphere.architect import NeuralArchitect
from hemisphere.engine import HemisphereEngine
from hemisphere.evolution import EvolutionEngine
from hemisphere.migration import MigrationAnalyzer
from hemisphere.registry import HemisphereRegistry
from hemisphere.data_feed import (
    HemisphereDataFeed,
    get_hemisphere_data_feed,
    get_safe_data_feed,
    should_initiate_evolution,
)
from hemisphere.distillation import distillation_collector, DistillationCollector

__all__ = [
    "DistillationCollector",
    "DistillationConfig",
    "DISTILLATION_CONFIGS",
    "HemisphereFocus",
    "NetworkStatus",
    "DesignStrategy",
    "NetworkArchitecture",
    "NetworkTopology",
    "PerformanceMetrics",
    "TrainingProgress",
    "HemisphereState",
    "HemisphereSnapshot",
    "MigrationEvent",
    "MigrationReadiness",
    "MigrationResult",
    "HemisphereOrchestrator",
    "NeuralArchitect",
    "HemisphereEngine",
    "EvolutionEngine",
    "MigrationAnalyzer",
    "HemisphereRegistry",
    "HemisphereDataFeed",
    "get_hemisphere_data_feed",
    "get_safe_data_feed",
    "should_initiate_evolution",
    "distillation_collector",
]
