"""EvolutionEngine: crossover and mutation of hemisphere architectures.

Ported from NeuralArchitect.designEvolutionaryArchitecture in
delete_later/neural-evolution/core/NeuralArchitect.ts.

Mutation space (informed by established neuroevolution research):
- Node count per layer: ±20% (original)
- Activation function: mutate across {relu, gelu, silu, tanh}
- Layer depth: add or remove a hidden layer
"""

from __future__ import annotations

import logging
import random
import time as _time
import uuid as _uuid

from hemisphere.types import (
    ArchitectureAttempt,
    HemisphereFocus,
    LayerDefinition,
    NetworkArchitecture,
    NetworkTopology,
    PerformanceMetrics,
)
from hemisphere.data_feed import HemisphereDataFeed

logger = logging.getLogger(__name__)

PARENT_ACCURACY_THRESHOLD = 0.30
PARENT_ACCURACY_THRESHOLD_KL = 0.10
DEFAULT_MUTATION_RATE = 0.10
MAX_MUTATION_FACTOR = 0.20

MUTABLE_ACTIVATIONS = ("relu", "gelu", "silu", "tanh")
ACTIVATION_MUTATION_RATE = 0.08
LAYER_DEPTH_MUTATION_RATE = 0.05
MAX_HIDDEN_LAYERS = 4
MIN_HIDDEN_LAYERS = 1


class EvolutionEngine:
    """Evolves hemisphere architectures through crossover and mutation."""

    def __init__(self) -> None:
        self._generation = 0
        self._attempts: list[ArchitectureAttempt] = []

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def attempts(self) -> list[ArchitectureAttempt]:
        return list(self._attempts)

    def evolve(
        self,
        parents: list[NetworkArchitecture],
        data: HemisphereDataFeed,
        focus: HemisphereFocus,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
    ) -> NetworkTopology | None:
        """Select best parents, crossover hidden layers, apply mutations.

        Returns None if no viable parents exist.
        """
        threshold = PARENT_ACCURACY_THRESHOLD_KL if focus in (HemisphereFocus.MOOD, HemisphereFocus.TRAITS) else PARENT_ACCURACY_THRESHOLD
        viable = sorted(
            [p for p in parents if p.performance.accuracy > threshold],
            key=lambda p: p.performance.accuracy,
            reverse=True,
        )
        if not viable:
            logger.info(
                "No viable parents (accuracy > %.0f%%), skipping evolution for %s",
                threshold * 100, focus.value,
            )
            return None

        parent1 = viable[0]
        parent2 = viable[1] if len(viable) > 1 else parent1

        topology = self._crossover(parent1, parent2, focus, mutation_rate)
        self._generation += 1

        attempt = ArchitectureAttempt(
            id=f"evo_{_uuid.uuid4().hex[:8]}",
            attempt_number=len(self._attempts) + 1,
            strategy="evolutionary_crossover",
            reasoning=(
                f"Generation {self._generation}: hybrid of "
                f"{parent1.name} ({parent1.performance.accuracy:.1%}) + "
                f"{parent2.name} ({parent2.performance.accuracy:.1%})"
            ),
            topology=topology,
            performance=PerformanceMetrics(),
            outcome="pending",
            lessons=(),
            timestamp=_time.time(),
        )
        self._attempts.append(attempt)
        return topology

    def record_outcome(
        self,
        attempt_id: str,
        performance: PerformanceMetrics,
        lessons: tuple[str, ...],
    ) -> None:
        """Update an attempt with its actual outcome after training."""
        for i, a in enumerate(self._attempts):
            if a.id == attempt_id:
                outcome = (
                    "success" if performance.accuracy > 0.6
                    else "partial" if performance.accuracy > 0.3
                    else "failure"
                )
                self._attempts[i] = ArchitectureAttempt(
                    id=a.id,
                    attempt_number=a.attempt_number,
                    strategy=a.strategy,
                    reasoning=a.reasoning,
                    topology=a.topology,
                    performance=performance,
                    outcome=outcome,
                    lessons=lessons,
                    timestamp=a.timestamp,
                )
                return

    def _crossover(
        self,
        parent1: NetworkArchitecture,
        parent2: NetworkArchitecture,
        focus: HemisphereFocus,
        mutation_rate: float,
    ) -> NetworkTopology:
        p1_hidden = [la for la in parent1.topology.layers if la.layer_type == "hidden"]
        p2_hidden = [la for la in parent2.topology.layers if la.layer_type == "hidden"]

        max_layers = max(len(p1_hidden), len(p2_hidden))
        hybrid_hidden: list[LayerDefinition] = []

        for i in range(max_layers):
            use_p1 = random.random() < 0.5
            source = None
            if use_p1 and i < len(p1_hidden):
                source = p1_hidden[i]
            elif not use_p1 and i < len(p2_hidden):
                source = p2_hidden[i]
            if source is None:
                source = p1_hidden[i] if i < len(p1_hidden) else (
                    p2_hidden[i] if i < len(p2_hidden) else None
                )
            if source is None:
                continue

            node_count = source.node_count
            if random.random() < mutation_rate:
                factor = 1.0 + (random.random() - 0.5) * 2.0 * MAX_MUTATION_FACTOR
                node_count = max(2, int(node_count * factor))

            activation = source.activation
            if random.random() < ACTIVATION_MUTATION_RATE:
                candidates = [a for a in MUTABLE_ACTIVATIONS if a != activation]
                if candidates:
                    activation = random.choice(candidates)
                    logger.debug("Activation mutation: %s → %s (layer %d)",
                                 source.activation, activation, i)

            dropout = getattr(source, "dropout", 0.0)

            hybrid_hidden.append(LayerDefinition(
                id=f"evolved_hidden_{i}",
                layer_type="hidden",
                node_count=node_count,
                activation=activation,
                dropout=dropout,
            ))

        hybrid_hidden = self._mutate_depth(hybrid_hidden)

        input_size = max(parent1.topology.input_size, parent2.topology.input_size)
        output_size = parent1.topology.output_size

        parent_out = [la for la in parent1.topology.layers if la.layer_type == "output"]
        output_activation = parent_out[0].activation if parent_out else "sigmoid"

        all_layers: list[LayerDefinition] = [
            LayerDefinition(
                id="evolved_input",
                layer_type="input",
                node_count=input_size,
                activation="linear",
            ),
            *hybrid_hidden,
            LayerDefinition(
                id="evolved_output",
                layer_type="output",
                node_count=output_size,
                activation=output_activation,
            ),
        ]

        total_params = 0
        for j in range(len(all_layers) - 1):
            total_params += all_layers[j].node_count * all_layers[j + 1].node_count
            total_params += all_layers[j + 1].node_count

        return NetworkTopology(
            input_size=input_size,
            layers=tuple(all_layers),
            output_size=output_size,
            total_parameters=total_params,
            activation_functions=tuple(la.activation for la in all_layers),
        )

    @staticmethod
    def _mutate_depth(hidden: list[LayerDefinition]) -> list[LayerDefinition]:
        """Possibly add or remove a hidden layer."""
        if random.random() >= LAYER_DEPTH_MUTATION_RATE:
            return hidden

        if len(hidden) < MAX_HIDDEN_LAYERS and random.random() < 0.5:
            ref = hidden[-1] if hidden else LayerDefinition(
                id="new", layer_type="hidden", node_count=16, activation="relu",
            )
            new_size = max(4, ref.node_count // 2)
            new_layer = LayerDefinition(
                id=f"evolved_hidden_{len(hidden)}",
                layer_type="hidden",
                node_count=new_size,
                activation=ref.activation,
                dropout=getattr(ref, "dropout", 0.0),
            )
            hidden.append(new_layer)
            logger.debug("Depth mutation: added layer (%d nodes)", new_size)

        elif len(hidden) > MIN_HIDDEN_LAYERS:
            removed = hidden.pop()
            logger.debug("Depth mutation: removed layer (%d nodes)", removed.node_count)

        return hidden
