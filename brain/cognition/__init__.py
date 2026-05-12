"""Cognition layer — world model, causal reasoning, simulation, and (future) planning."""

from cognition.world_state import (
    PhysicalState,
    UserState,
    ConversationState,
    SystemState,
    WorldState,
    WorldDelta,
)
from cognition.world_model import WorldModel
from cognition.causal_engine import CausalEngine
from cognition.promotion import WorldModelPromotion, SimulatorPromotion
from cognition.simulator import MentalSimulator, SimulationStep, SimulationTrace
from cognition.planner import WorldPlanner, PlanOption
from cognition.world_schema import (
    SensorObservation,
    WorldEntity,
    WorldRelation,
    WorldNorm,
    WorldPrediction,
    WorldOutcome,
    WorldAdvisory,
    WorldEpisode,
    PredictionTarget,
    WorldPosition,
    WorldZone,
    ArchetypePack,
    ArchetypeRegistry,
    CanonicalWorldState,
)
from cognition.world_adapters import CanonicalWorldProjector
from cognition.intention_registry import (
    IntentionRecord,
    IntentionRegistry,
    intention_registry,
)
from cognition.commitment_extractor import (
    CommitmentMatch,
    extract_commitments,
)

__all__ = [
    "PhysicalState",
    "UserState",
    "ConversationState",
    "SystemState",
    "WorldState",
    "WorldDelta",
    "WorldModel",
    "CausalEngine",
    "WorldModelPromotion",
    "SimulatorPromotion",
    "MentalSimulator",
    "SimulationStep",
    "SimulationTrace",
    "WorldPlanner",
    "PlanOption",
    # Canonical substrate
    "SensorObservation",
    "WorldEntity",
    "WorldRelation",
    "WorldNorm",
    "WorldPrediction",
    "WorldOutcome",
    "WorldAdvisory",
    "WorldEpisode",
    "PredictionTarget",
    # Ontology expansion
    "WorldPosition",
    "WorldZone",
    "ArchetypePack",
    "ArchetypeRegistry",
    "CanonicalWorldState",
    "CanonicalWorldProjector",
    # Intention infrastructure (Stage 0: truth only, no delivery)
    "IntentionRecord",
    "IntentionRegistry",
    "intention_registry",
    "CommitmentMatch",
    "extract_commitments",
]
