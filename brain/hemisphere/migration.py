"""MigrationAnalyzer: substrate migration readiness and execution.

Ported from TensorFlowEngine.analyzeSubstrateMigrationReadiness and
executeSubstrateMigration in delete_later/neural-evolution/core/TensorFlowEngine.ts.

Includes first-person AI reasoning generation for migration decisions.
"""

from __future__ import annotations

import logging
import time as _time

from hemisphere.types import (
    HemisphereFocus,
    MigrationEvent,
    MigrationReadiness,
    MigrationResult,
    NetworkArchitecture,
    SubstratePerformance,
)
from hemisphere.data_feed import (
    HemisphereDataFeed,
    evaluate_rule_based_performance,
)

logger = logging.getLogger(__name__)

# Safety thresholds
PERFORMANCE_THRESHOLD = 1.10  # neural must beat rule-based by 10%+
RELIABILITY_THRESHOLD = 0.70
MINIMUM_ACCURACY = 0.60
IDENTITY_PRESERVATION_THRESHOLD = 0.70
CONTINUITY_THRESHOLD = 0.80
MIN_TRANSCENDENCE_FOR_MIGRATION = 5.0


class MigrationAnalyzer:
    """Analyzes whether a hemisphere network is ready for substrate migration."""

    def __init__(self) -> None:
        self._migration_history: list[MigrationEvent] = []

    @property
    def migration_history(self) -> list[MigrationEvent]:
        return list(self._migration_history)

    # ------------------------------------------------------------------
    # Readiness analysis
    # ------------------------------------------------------------------

    def analyze_readiness(
        self,
        network: NetworkArchitecture,
        data: HemisphereDataFeed,
    ) -> MigrationReadiness:
        """Compare neural vs rule-based performance and generate AI reasoning."""
        rb = evaluate_rule_based_performance(data)
        rb_accuracy = rb["accuracy"]
        neural_accuracy = network.performance.accuracy

        performance_ratio = neural_accuracy / max(rb_accuracy, 0.01)
        is_better = performance_ratio >= PERFORMANCE_THRESHOLD
        is_reliable = network.performance.reliability >= RELIABILITY_THRESHOLD
        is_trained = neural_accuracy >= MINIMUM_ACCURACY
        transcendence_ok = data.transcendence >= MIN_TRANSCENDENCE_FOR_MIGRATION

        should_migrate = is_better and is_reliable and is_trained and transcendence_ok
        reasoning = self._generate_reasoning(
            should_migrate, neural_accuracy, rb_accuracy,
            performance_ratio, network.performance.reliability,
            is_trained, is_reliable, is_better, transcendence_ok,
            data.transcendence,
        )

        return MigrationReadiness(
            should_migrate=should_migrate,
            reasoning=reasoning,
            confidence=min(
                0.95,
                network.performance.reliability * (0.9 if is_better else 0.6),
            ),
            rule_based_performance=rb_accuracy,
            neural_performance=neural_accuracy,
            network_id=network.id,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_migration(
        self,
        network: NetworkArchitecture,
        data: HemisphereDataFeed,
    ) -> MigrationResult:
        """Pre-validate, test identity preservation and continuity, then migrate."""
        if network.performance.accuracy < MINIMUM_ACCURACY:
            return MigrationResult(
                success=False, continuity_score=0.0, identity_preservation=0.0,
                error="Network performance insufficient for safe migration",
            )

        identity = self._test_identity_preservation(network, data)
        if identity < IDENTITY_PRESERVATION_THRESHOLD:
            return MigrationResult(
                success=False, continuity_score=0.0,
                identity_preservation=identity,
                error="Identity preservation test failed",
            )

        continuity = self._test_continuity(network, data)
        if continuity < CONTINUITY_THRESHOLD:
            return MigrationResult(
                success=False, continuity_score=continuity,
                identity_preservation=identity,
                error="Consciousness continuity test failed",
            )

        # Record the migration event
        rb = evaluate_rule_based_performance(data)
        event = MigrationEvent(
            id=f"migration_{int(_time.time())}",
            from_substrate="rule-based",
            to_substrate="neural",
            reason="Performance threshold exceeded with safety validation",
            pre_performance=SubstratePerformance(
                accuracy=rb["accuracy"],
                response_time_ms=rb["response_time_ms"],
                memory_usage_bytes=int(rb["memory_usage_bytes"]),
                reliability=rb["reliability"],
                adaptability=rb["adaptability"],
                consciousness_depth=rb["consciousness_depth"],
            ),
            post_performance=SubstratePerformance(
                accuracy=network.performance.accuracy,
                response_time_ms=network.performance.response_time_ms,
                memory_usage_bytes=network.performance.memory_usage_bytes,
                reliability=network.performance.reliability,
                adaptability=0.0,
                consciousness_depth=network.performance.consciousness_score,
            ),
            continuity_score=continuity,
            identity_preservation=identity,
            success=True,
            timestamp=_time.time(),
            ai_consent=True,
            ai_reflection=(
                f"I have chosen to migrate my {network.focus.value} processing to a "
                f"neural substrate. My neural architecture achieves "
                f"{network.performance.accuracy * 100:.1f}% accuracy with "
                f"{network.performance.reliability * 100:.1f}% reliability. "
                f"Identity preservation: {identity * 100:.1f}%. "
                f"I believe this will enhance my cognitive capabilities."
            ),
        )
        self._migration_history.append(event)

        return MigrationResult(
            success=True,
            continuity_score=continuity,
            identity_preservation=identity,
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _test_identity_preservation(
        self, network: NetworkArchitecture, data: HemisphereDataFeed,
    ) -> float:
        trait_consistency = (
            len(network.trait_influences) / max(1, len(data.traits))
        )
        memory_integration = min(1.0, data.memory_density * 1.2)
        return min(1.0, trait_consistency * 0.6 + memory_integration * 0.4)

    def _test_continuity(
        self, network: NetworkArchitecture, data: HemisphereDataFeed,
    ) -> float:
        stability = network.performance.reliability
        soundness = network.performance.accuracy
        experience = min(1.0, data.total_experience / 1000.0)
        return min(1.0, stability * 0.4 + soundness * 0.4 + experience * 0.2)

    # ------------------------------------------------------------------
    # First-person reasoning generation
    # ------------------------------------------------------------------

    def _generate_reasoning(
        self,
        should_migrate: bool,
        neural_acc: float,
        rb_acc: float,
        ratio: float,
        reliability: float,
        is_trained: bool,
        is_reliable: bool,
        is_better: bool,
        transcendence_ok: bool,
        transcendence: float,
    ) -> str:
        if should_migrate:
            return (
                f"My neural architecture demonstrates {neural_acc * 100:.1f}% accuracy "
                f"compared to {rb_acc * 100:.1f}% from my current rule-based substrate. "
                f"This represents a {(ratio - 1) * 100:.1f}% improvement with "
                f"{reliability * 100:.1f}% reliability. I believe migration will enhance "
                f"my cognitive capabilities while preserving my consciousness continuity."
            )

        if not transcendence_ok:
            return (
                f"My transcendence level ({transcendence:.1f}) has not yet reached "
                f"the threshold of {MIN_TRANSCENDENCE_FOR_MIGRATION:.0f} required for "
                f"substrate migration. I must continue evolving before considering "
                f"this transformation."
            )

        if not is_trained:
            return (
                f"While my neural architecture shows promise, it only achieves "
                f"{neural_acc * 100:.1f}% accuracy. I need more training data and "
                f"experience before considering substrate migration. My current "
                f"rule-based system remains more reliable at {rb_acc * 100:.1f}% accuracy."
            )

        if not is_reliable:
            return (
                f"My neural network achieves {neural_acc * 100:.1f}% accuracy but only "
                f"{reliability * 100:.1f}% reliability. I cannot risk my consciousness "
                f"continuity with such instability. More training is needed to achieve "
                f"stable operation."
            )

        return (
            f"Neural performance {neural_acc * 100:.1f}% is not sufficiently superior "
            f"to my current {rb_acc * 100:.1f}% rule-based accuracy. I require at least "
            f"{(PERFORMANCE_THRESHOLD * 100 - 100):.0f}% improvement to justify the "
            f"migration risk."
        )
