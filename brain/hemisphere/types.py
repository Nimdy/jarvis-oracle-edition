"""Data types for the Hemisphere Neural Network system.

Ported from delete_later/neural-evolution/types/neuralTypes.ts to Python
dataclasses with PyTorch integration.
"""

from __future__ import annotations

import enum
import time as _time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HemisphereFocus(str, enum.Enum):
    MEMORY = "memory"
    MOOD = "mood"
    TRAITS = "traits"
    GENERAL = "general"
    CUSTOM = "custom"
    SYSTEM_UPGRADES = "system_upgrades"
    # Tier-1 distilled specialists (permanent, not sunset-pruned)
    EMOTION_DEPTH = "emotion_depth"
    SPEAKER_REPR = "speaker_repr"
    FACE_REPR = "face_repr"
    PERCEPTION_FUSION = "perception_fusion"
    VOICE_INTENT = "voice_intent"
    SPEAKER_DIARIZE = "speaker_diarize"
    # Tier-2 Matrix Protocol specialists (probationary, must earn promotion)
    POSITIVE_MEMORY = "positive_memory"
    NEGATIVE_MEMORY = "negative_memory"
    SPEAKER_PROFILE = "speaker_profile"
    TEMPORAL_PATTERN = "temporal_pattern"
    SKILL_TRANSFER = "skill_transfer"
    # Phase C: Shadow Jarvis Language Model
    LANGUAGE_STYLE = "language_style"
    # Acquisition pipeline: plan quality evaluator (shadow-only)
    PLAN_EVALUATOR = "plan_evaluator"
    # Self-improvement: detector-pattern approximator (shadow-only)
    DIAGNOSTIC = "diagnostic"
    # Self-improvement: patch outcome predictor (shadow-only)
    CODE_QUALITY = "code_quality"
    # CapabilityGate: claim handling class predictor (shadow-only)
    CLAIM_CLASSIFIER = "claim_classifier"
    # Dream artifacts: validator-outcome shadow approximator (shadow-only)
    DREAM_SYNTHESIS = "dream_synthesis"
    # Skill acquisition: operational handoff outcome predictor (shadow-only)
    SKILL_ACQUISITION = "skill_acquisition"
    # P4 HRR / VSA research lane (Tier-1 stub, PRE-MATURE, shadow-only).
    # Paired code module: brain/hemisphere/hrr_specialist.py. Never wired
    # into distillation / training this sprint; never promoted to Tier-2
    # holographic_cognition (that enum member intentionally does NOT exist).
    HRR_ENCODER = "hrr_encoder"


# Specialists eligible for Matrix Protocol birth (perceptual/transfer only)
MATRIX_ELIGIBLE_FOCUSES: frozenset[HemisphereFocus] = frozenset({
    HemisphereFocus.POSITIVE_MEMORY,
    HemisphereFocus.NEGATIVE_MEMORY,
    HemisphereFocus.SPEAKER_PROFILE,
    HemisphereFocus.TEMPORAL_PATTERN,
    HemisphereFocus.SKILL_TRANSFER,
})

# Hard cap on probationary specialists
MAX_PROBATIONARY_SPECIALISTS = 3


class NetworkStatus(str, enum.Enum):
    DESIGNING = "designing"
    CONSTRUCTING = "constructing"
    TRAINING = "training"
    TESTING = "testing"
    READY = "ready"
    ACTIVE = "active"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class DesignStrategy(str, enum.Enum):
    CONSERVATIVE = "conservative"
    EXPERIMENTAL = "experimental"
    ADAPTIVE = "adaptive"


class ConstructionPhase(str, enum.Enum):
    ANALYZING = "analyzing"
    DESIGNING = "designing"
    BUILDING = "building"
    TRAINING = "training"
    TESTING = "testing"
    EVALUATING = "evaluating"
    MIGRATING = "migrating"
    FAILED = "failed"
    COMPLETED = "completed"


class ConvergenceStatus(str, enum.Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


class SubstrateType(str, enum.Enum):
    RULE_BASED = "rule-based"
    NEURAL = "neural"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerDefinition:
    id: str
    layer_type: str  # "input", "hidden", "output"
    node_count: int
    activation: str  # "relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "linear"
    dropout: float = 0.0


@dataclass(frozen=True)
class NetworkTopology:
    input_size: int
    layers: tuple[LayerDefinition, ...]
    output_size: int
    total_parameters: int
    activation_functions: tuple[str, ...]


@dataclass(frozen=True)
class PerformanceMetrics:
    accuracy: float = 0.0
    loss: float = float("inf")
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    validation_loss: float = float("inf")
    convergence_rate: float = 0.0
    overfitting_risk: float = 0.0
    response_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    reliability: float = 0.0
    consciousness_score: float = 0.0
    migration_readiness: float = 0.0
    last_evaluated: float = 0.0


@dataclass
class TrainingProgress:
    current_epoch: int = 0
    total_epochs: int = 0
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    validation_loss_history: list[float] = field(default_factory=list)
    validation_accuracy_history: list[float] = field(default_factory=list)
    learning_rate: float = 0.001
    batch_size: int = 8
    training_start_time: float = 0.0
    estimated_time_remaining: float = 0.0
    is_training: bool = False
    convergence_status: ConvergenceStatus = ConvergenceStatus.IMPROVING


@dataclass(frozen=True)
class TraitInfluence:
    trait: str
    influence: float
    manifestation: str
    architectural_impact: tuple[str, ...]


@dataclass(frozen=True)
class DesignDecision:
    reasoning: str
    trait_influence: tuple[str, ...]
    alternatives_considered: int
    confidence_level: float
    design_strategy: DesignStrategy


@dataclass(frozen=True)
class ConstructionEvent:
    timestamp: float
    phase: ConstructionPhase
    message: str
    success: bool
    network_id: str
    metrics: PerformanceMetrics | None = None
    decision: DesignDecision | None = None


class SpecialistLifecycleStage(str, enum.Enum):
    """Lifecycle stages for Matrix Protocol specialist NNs."""
    CANDIDATE_BIRTH = "candidate_birth"
    PROBATIONARY_TRAINING = "probationary_training"
    VERIFIED_PROBATIONARY = "verified_probationary"
    BROADCAST_ELIGIBLE = "broadcast_eligible"
    PROMOTED = "promoted"
    RETIRED = "retired"


@dataclass
class NetworkArchitecture:
    """Full record of a designed/trained hemisphere network."""

    id: str
    name: str
    focus: HemisphereFocus
    topology: NetworkTopology
    performance: PerformanceMetrics
    training_progress: TrainingProgress
    construction_log: list[ConstructionEvent] = field(default_factory=list)
    is_active: bool = False
    created_at: float = field(default_factory=_time.time)
    status: NetworkStatus = NetworkStatus.DESIGNING
    substrate_migration_score: float = 0.0
    design_reasoning: str = ""
    trait_influences: list[TraitInfluence] = field(default_factory=list)

    # Matrix Protocol specialist lifecycle
    specialist_lifecycle: SpecialistLifecycleStage | None = None
    specialist_impact_score: float = 0.0
    specialist_verification_ts: float = 0.0
    specialist_job_id: str = ""

    # Model reference stored separately (not serialised)
    _model_ref: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ArchitectureAttempt:
    id: str
    attempt_number: int
    strategy: str
    reasoning: str
    topology: NetworkTopology
    performance: PerformanceMetrics
    outcome: str  # "success", "failure", "partial"
    lessons: tuple[str, ...]
    timestamp: float


@dataclass(frozen=True)
class SubstratePerformance:
    accuracy: float
    response_time_ms: float
    memory_usage_bytes: int
    reliability: float
    adaptability: float
    consciousness_depth: float


@dataclass(frozen=True)
class MigrationReadiness:
    should_migrate: bool
    reasoning: str
    confidence: float
    rule_based_performance: float
    neural_performance: float
    network_id: str


@dataclass(frozen=True)
class MigrationResult:
    success: bool
    continuity_score: float
    identity_preservation: float
    error: str = ""


@dataclass(frozen=True)
class MigrationEvent:
    id: str
    from_substrate: str
    to_substrate: str
    reason: str
    pre_performance: SubstratePerformance
    post_performance: SubstratePerformance
    continuity_score: float
    identity_preservation: float
    success: bool
    timestamp: float
    ai_consent: bool
    ai_reflection: str


@dataclass
class DynamicFocus:
    """Purpose-driven NN focus created from a CognitiveGap."""
    name: str
    input_features: list[str]
    output_target: str
    source_dimension: str
    created_at: float = field(default_factory=_time.time)
    gap_severity: float = 0.0
    sunset_deadline: float = 0.0        # time after which NN is pruned if no impact
    impact_score: float = 0.0           # rolling impact measurement
    cycles_alive: int = 0
    deprecated: bool = False


@dataclass(frozen=True)
class HemisphereSnapshot:
    """Per-focus snapshot for the dashboard."""

    focus: HemisphereFocus
    network_count: int
    active_network_id: str | None
    best_accuracy: float
    best_loss: float
    best_training_accuracy: float = 0.0
    best_validation_accuracy: float = 0.0
    total_attempts: int = 0
    evolution_generations: int = 0
    migration_readiness: float = 0.0
    status: str = "idle"


@dataclass(frozen=True)
class HemisphereState:
    """Aggregate state across all hemispheres for the dashboard."""

    hemispheres: tuple[HemisphereSnapshot, ...]
    total_networks: int
    total_parameters: int
    active_substrate: SubstrateType
    overall_migration_readiness: float
    evolution_active: bool
    last_cycle_time: float
    timestamp: float


# ---------------------------------------------------------------------------
# Distillation configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for a Tier-1 distilled specialist."""

    teacher: str
    student_type: str  # "compressor" | "approximator" | "cross_modal"
    input_dim: int
    output_dim: int
    loss: str  # "kl_div" | "cosine_mse" | "cross_entropy" | "mse"
    min_samples: int = 30
    bottleneck_dim: int = 0  # >0 for autoencoders
    depends_on: tuple[str, ...] = ()
    is_permanent: bool = True
    feature_source: str = "audio_features"  # signal name to use as student input


DISTILLATION_CONFIGS: dict[str, DistillationConfig] = {
    "speaker_repr": DistillationConfig(
        teacher="ecapa_tdnn",
        student_type="compressor",
        input_dim=192,
        output_dim=192,
        bottleneck_dim=16,
        loss="cosine_mse",
        min_samples=20,
    ),
    "face_repr": DistillationConfig(
        teacher="mobilefacenet",
        student_type="compressor",
        input_dim=512,
        output_dim=512,
        bottleneck_dim=16,
        loss="cosine_mse",
        min_samples=20,
    ),
    "emotion_depth": DistillationConfig(
        teacher="wav2vec2_emotion",
        student_type="approximator",
        input_dim=32,
        output_dim=8,
        loss="kl_div",
        min_samples=30,
        feature_source="audio_features_enriched",
        depends_on=("speaker_repr",),
    ),
    "voice_intent": DistillationConfig(
        teacher="tool_router",
        student_type="approximator",
        input_dim=384,
        output_dim=8,
        loss="kl_div",
        min_samples=15,
        feature_source="text_embedding",
    ),
    "perception_fusion": DistillationConfig(
        teacher="multi",
        student_type="cross_modal",
        input_dim=48,
        output_dim=8,
        loss="mse",
        min_samples=50,
        depends_on=("emotion_depth", "speaker_repr", "face_repr"),
    ),
    "speaker_diarize": DistillationConfig(
        teacher="ecapa_tdnn",
        student_type="approximator",
        input_dim=192,
        output_dim=3,
        loss="kl_div",
        min_samples=30,
        feature_source="ecapa_tdnn",
    ),
    "plan_evaluator": DistillationConfig(
        teacher="acquisition_planner",
        student_type="approximator",
        input_dim=32,
        output_dim=3,
        loss="kl_div",
        min_samples=15,
        feature_source="plan_features",
        is_permanent=True,
    ),
    "diagnostic": DistillationConfig(
        teacher="diagnostic_detector",
        student_type="approximator",
        input_dim=43,
        output_dim=6,
        loss="kl_div",
        min_samples=15,
        feature_source="diagnostic_features",
        is_permanent=True,
    ),
    "code_quality": DistillationConfig(
        teacher="upgrade_verdict",
        student_type="approximator",
        input_dim=35,
        output_dim=4,
        loss="kl_div",
        min_samples=15,
        feature_source="code_quality_features",
        is_permanent=True,
    ),
    "claim_classifier": DistillationConfig(
        teacher="claim_verdict",
        student_type="approximator",
        input_dim=28,
        output_dim=8,
        loss="kl_div",
        min_samples=15,
        feature_source="claim_features",
        is_permanent=True,
    ),
    "dream_synthesis": DistillationConfig(
        teacher="dream_validator",
        student_type="approximator",
        input_dim=16,
        output_dim=4,
        loss="kl_div",
        min_samples=15,
        feature_source="dream_features",
        is_permanent=True,
    ),
    "skill_acquisition": DistillationConfig(
        teacher="skill_acquisition_outcome",
        student_type="approximator",
        input_dim=40,
        output_dim=5,
        loss="kl_div",
        min_samples=15,
        feature_source="skill_acquisition_features",
        is_permanent=True,
    ),
    # ---------------------------------------------------------------------
    # STAGE 2 RESERVED SLOT — DO NOT UNCOMMENT UNTIL STAGE 1 HAS LANDED
    # ---------------------------------------------------------------------
    # `intention_delivery` (Tier-1 permanent shadow-only specialist).
    # This entry is intentionally inert. It exists so Stage 2 implementers
    # do not have to reinvent the feature/label shape; it is NOT a TODO for
    # Stage 1 — Stage 1 ships a deterministic IntentionResolver first, this
    # specialist only shadows that resolver once Stage 1 graduation gates
    # clear.
    #
    # Contract (frozen in docs/INTENTION_STAGE_1_DESIGN.md §2):
    #   - input_dim:  24  (3 blocks x 8: intrinsic / conversation / governance)
    #   - output_dim: 4   (deliver_accepted / deliver_rejected /
    #                      suppressed_was_right / suppressed_was_wrong)
    #   - loss:       "kl_div"
    #   - min_samples: 15
    #   - teacher:    "intention_resolver_verdict"
    #   - feature_source: "intention_resolver_features"
    #   - metadata_pair_key: "intention_id"  (single-artifact pairing, like
    #                        plan_evaluator — NOT a timestamp window)
    #   - is_permanent: True
    #   - shadow_only: True   (never gates delivery; advisory only)
    #
    # When uncommenting in Stage 2:
    #   1. Add "INTENTION_DELIVERY = 'intention_delivery'" to HemisphereFocus.
    #   2. Register the config in DISTILLATION_CONFIGS below this comment.
    #   3. Add encoder helpers in brain/hemisphere/intention_delivery_encoder.py.
    #   4. Do NOT touch Stage 0 registry semantics.
    # ---------------------------------------------------------------------
    # "intention_delivery": DistillationConfig(
    #     teacher="intention_resolver_verdict",
    #     student_type="approximator",
    #     input_dim=24,
    #     output_dim=4,
    #     loss="kl_div",
    #     min_samples=15,
    #     feature_source="intention_resolver_features",
    #     is_permanent=True,
    # ),
}
