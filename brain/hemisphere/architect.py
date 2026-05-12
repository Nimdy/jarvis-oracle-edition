"""NeuralArchitect: designs network topologies from consciousness data.

Ported from delete_later/neural-evolution/core/NeuralArchitect.ts.
The AI decides its own network topology based on its consciousness state,
traits, memory patterns, design strategy, and accumulated research knowledge.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hemisphere.types import (
    ArchitectureAttempt,
    DesignDecision,
    DesignStrategy,
    DistillationConfig,
    HemisphereFocus,
    LayerDefinition,
    NetworkTopology,
    TraitInfluence,
)
from hemisphere.data_feed import HemisphereDataFeed

logger = logging.getLogger(__name__)

VALID_HIDDEN_ACTIVATIONS = ("relu", "gelu", "silu", "tanh")
DEFAULT_HIDDEN_ACTIVATION = "relu"


class NeuralArchitect:
    """Designs neural network topologies from consciousness patterns."""

    def __init__(self) -> None:
        self._evolution_history: list[ArchitectureAttempt] = []
        self._preferred_activation: str = DEFAULT_HIDDEN_ACTIVATION
        self._preferred_dropout: float = 0.0
        self._research_findings: list[str] = []

    # ------------------------------------------------------------------
    # Research priors
    # ------------------------------------------------------------------

    def set_research_priors(self, findings: list[str]) -> None:
        """Extract actionable architecture priors from research findings.

        Scans finding text for activation function recommendations,
        regularization techniques, and architecture insights. The priors
        influence design decisions until the next refresh.
        """
        self._research_findings = findings
        if not findings:
            return

        combined = " ".join(findings).lower()

        activation = self._extract_activation_prior(combined)
        if activation:
            self._preferred_activation = activation

        dropout = self._extract_dropout_prior(combined)
        if dropout > 0:
            self._preferred_dropout = dropout

    @staticmethod
    def _extract_activation_prior(text: str) -> str:
        """Detect recommended activation function from research text."""
        activation_scores: dict[str, int] = {a: 0 for a in VALID_HIDDEN_ACTIVATIONS}

        positive_ctx = re.compile(
            r"(recommend|outperform|superior|better|effective|state.of.the.art|"
            r"prefer|advantage|improv|benefit)"
        )

        for act in VALID_HIDDEN_ACTIVATIONS:
            mentions = len(re.findall(rf"\b{act}\b", text))
            activation_scores[act] += mentions

            for m in re.finditer(rf"\b{act}\b", text):
                window = text[max(0, m.start() - 120): m.end() + 120]
                if positive_ctx.search(window):
                    activation_scores[act] += 3

        best = max(activation_scores, key=activation_scores.get)  # type: ignore[arg-type]
        if activation_scores[best] >= 3:
            logger.info("Research prior: preferred activation → %s (score %d)",
                        best, activation_scores[best])
            return best
        return ""

    @staticmethod
    def _extract_dropout_prior(text: str) -> float:
        """Detect recommended dropout rate from research text."""
        match = re.search(r"dropout.*?(\d\.\d+)", text)
        if match:
            rate = float(match.group(1))
            if 0.05 <= rate <= 0.5:
                logger.info("Research prior: dropout → %.2f", rate)
                return rate

        if "dropout" in text and ("regulariz" in text or "overfit" in text):
            return 0.1
        return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def design_architecture(
        self,
        data: HemisphereDataFeed,
        focus: HemisphereFocus,
        complexity: str = "medium",
    ) -> NetworkTopology:
        """AI designs its own neural network based on consciousness patterns."""
        input_size = self._calculate_input_size(data, focus)
        hidden_layers = self._design_hidden_layers(complexity, data)
        output_size = self._get_output_size(focus)

        layers = self._build_layer_definitions(
            input_size, hidden_layers, output_size, focus,
        )
        total_params = self._calculate_total_parameters(layers)

        return NetworkTopology(
            input_size=input_size,
            layers=tuple(layers),
            output_size=output_size,
            total_parameters=total_params,
            activation_functions=tuple(ld.activation for ld in layers),
        )

    def analyze_consciousness_data(
        self,
        data: HemisphereDataFeed,
    ) -> DesignDecision:
        """Analyze consciousness data and decide a design strategy."""
        trait_influences = self._analyze_trait_influence(data.traits)
        memory_complexity = min(1.0, len(data.memories) / 100.0)
        mood_stability = 0.8 if data.mood in ("professional", "focused") else 0.5

        strategy = self._determine_design_strategy(
            trait_influences, memory_complexity, mood_stability,
        )

        research_note = ""
        if self._research_findings:
            research_note = f", informed by {len(self._research_findings)} research finding(s)"

        return DesignDecision(
            reasoning=(
                f"Based on {len(data.traits)} traits, {len(data.memories)} memories, "
                f"and {data.mood} mood state{research_note}"
            ),
            trait_influence=tuple(data.traits),
            alternatives_considered=3,
            confidence_level=min(0.9, 0.4 + data.memory_density),
            design_strategy=strategy,
        )

    def calculate_trait_influences(
        self, traits: tuple[str, ...] | list[str],
    ) -> list[TraitInfluence]:
        """Return per-trait influence records."""
        return [
            TraitInfluence(
                trait=t,
                influence=0.3 + (hash(t) % 40) / 100.0,
                manifestation=_TRAIT_MANIFESTATIONS.get(
                    t, f"{t} influences network design",
                ),
                architectural_impact=_TRAIT_IMPACTS.get(
                    t, ("general_influence",),
                ),
            )
            for t in traits
        ]

    def generate_architecture_name(
        self,
        focus: HemisphereFocus,
        strategy: DesignStrategy,
        data: HemisphereDataFeed,
    ) -> str:
        dominant_trait = data.traits[0] if data.traits else "Neural"
        mood = data.mood.capitalize()
        strat = strategy.value.capitalize()
        return f"{dominant_trait}-{mood}-{strat}-{focus.value.capitalize()}"

    def record_attempt(self, attempt: ArchitectureAttempt) -> None:
        self._evolution_history.append(attempt)

    @property
    def evolution_history(self) -> list[ArchitectureAttempt]:
        return list(self._evolution_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_input_size(
        self, data: HemisphereDataFeed, focus: HemisphereFocus,
    ) -> int:
        base = 8
        mem_features = min(len(data.memories), 50) // 10
        trait_features = len(data.traits)
        return min(32, max(8, base + mem_features + trait_features))

    def _get_output_size(self, focus: HemisphereFocus) -> int:
        return {
            HemisphereFocus.MEMORY: 8,
            HemisphereFocus.MOOD: 5,
            HemisphereFocus.TRAITS: 10,
            HemisphereFocus.GENERAL: 6,
            HemisphereFocus.EMOTION_DEPTH: 8,
            HemisphereFocus.SPEAKER_REPR: 16,
            HemisphereFocus.FACE_REPR: 16,
            HemisphereFocus.PERCEPTION_FUSION: 8,
            HemisphereFocus.VOICE_INTENT: 8,
            HemisphereFocus.CUSTOM: 6,
            HemisphereFocus.SYSTEM_UPGRADES: 6,
            HemisphereFocus.DREAM_SYNTHESIS: 4,
            # Tier-2 Matrix Protocol specialist (P3.6). Output is a
            # 4-class soft distribution over positive-memory regimes
            # (low / mild / strong / saturated). The orchestrator does
            # not consume the raw output for the broadcast slot — it
            # uses ``PositiveMemoryEncoder.compute_signal_value`` — but
            # the network must still have a defined topology so the
            # Matrix lifecycle (CANDIDATE_BIRTH → PROBATIONARY_TRAINING)
            # can run end-to-end.
            HemisphereFocus.POSITIVE_MEMORY: 4,
            # Tier-2 Matrix Protocol specialist (P3.7). Output is a
            # 4-class soft distribution over negative-memory regimes
            # (low / mild / strong / saturated). The orchestrator does
            # not consume the raw output for the broadcast slot — it
            # uses ``NegativeMemoryEncoder.compute_signal_value`` — but
            # the network must still have a defined topology so the
            # Matrix lifecycle (CANDIDATE_BIRTH → PROBATIONARY_TRAINING)
            # can run end-to-end.
            HemisphereFocus.NEGATIVE_MEMORY: 4,
            # Tier-2 Matrix Protocol specialist (P3.8). Output is a
            # 4-class soft distribution over speaker-identification
            # regimes (unknown / tentative / known / verified). The
            # orchestrator does not consume the raw output for the
            # broadcast slot — it uses
            # ``SpeakerProfileEncoder.compute_signal_value`` — but the
            # network must still have a defined topology so the Matrix
            # lifecycle can run end-to-end. The raw ECAPA speaker_repr
            # vector is never plumbed here; only the derived 16-dim
            # context vector enters the network input.
            HemisphereFocus.SPEAKER_PROFILE: 4,
            # Tier-2 Matrix Protocol specialist (P3.9). Output is a
            # 4-class soft distribution over temporal-context regimes
            # (idle / sparse / steady / active). The orchestrator does
            # not consume the raw output for the broadcast slot — it
            # uses ``TemporalPatternEncoder.compute_signal_value`` —
            # but the network must still have a defined topology so
            # the Matrix lifecycle can run end-to-end. The encoder
            # never emits hour-of-day / weekday / per-speaker-schedule
            # fields; the network input is purely a derived bounded
            # rhythm-stability vector, by privacy-contract design.
            HemisphereFocus.TEMPORAL_PATTERN: 4,
            # Tier-2 Matrix Protocol specialist (P3.10). Output is a
            # 4-class soft distribution over skill-transfer regimes
            # (sparse / partial / rich / saturated). The orchestrator
            # does not consume the raw output for the broadcast slot
            # — it uses ``SkillTransferEncoder.compute_signal_value``
            # — but the network must still have a defined topology
            # so the Matrix lifecycle can run end-to-end. The
            # encoder NEVER claims a skill is verified, NEVER
            # promotes a capability, and the capability_gate path
            # remains the sole authority for capability promotion;
            # the network input is purely a derived bounded
            # registry/job-state vector, by capability-contract
            # design.
            HemisphereFocus.SKILL_TRANSFER: 4,
        }.get(focus, 6)

    def _design_hidden_layers(
        self, complexity: str, data: HemisphereDataFeed,
    ) -> list[int]:
        base = max(4, len(data.memories) // 10)

        trait_set = set(data.traits)
        if "Explorer" in trait_set:
            base = int(base * 1.3)
        if "Cautious" in trait_set:
            base = int(base * 0.8)

        if complexity == "simple":
            return [max(8, base * 2)]
        elif complexity == "medium":
            return [max(16, base * 3), max(8, int(base * 1.5))]
        elif complexity == "complex":
            return [
                max(32, base * 4),
                max(16, base * 2),
                max(8, base),
            ]
        return [16, 8]

    def _build_layer_definitions(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        focus: HemisphereFocus,
    ) -> list[LayerDefinition]:
        hidden_act = self._preferred_activation
        if hidden_act not in VALID_HIDDEN_ACTIVATIONS:
            hidden_act = DEFAULT_HIDDEN_ACTIVATION

        layers: list[LayerDefinition] = [
            LayerDefinition(
                id="input",
                layer_type="input",
                node_count=input_size,
                activation="linear",
            ),
        ]
        for i, size in enumerate(hidden_sizes):
            layers.append(
                LayerDefinition(
                    id=f"hidden_{i}",
                    layer_type="hidden",
                    node_count=size,
                    activation=hidden_act,
                    dropout=self._preferred_dropout if self._preferred_dropout > 0 else 0.0,
                ),
            )
        output_act = "sigmoid" if focus in (
            HemisphereFocus.MEMORY, HemisphereFocus.GENERAL, HemisphereFocus.SYSTEM_UPGRADES,
        ) else "softmax"
        layers.append(
            LayerDefinition(
                id="output",
                layer_type="output",
                node_count=output_size,
                activation=output_act,
            ),
        )
        return layers

    @staticmethod
    def _calculate_total_parameters(layers: list[LayerDefinition]) -> int:
        total = 0
        for i in range(len(layers) - 1):
            total += layers[i].node_count * layers[i + 1].node_count
            total += layers[i + 1].node_count
        return total

    def _analyze_trait_influence(
        self, traits: tuple[str, ...] | list[str],
    ) -> list[TraitInfluence]:
        return self.calculate_trait_influences(traits)

    def _determine_design_strategy(
        self,
        influences: list[TraitInfluence],
        memory_complexity: float,
        mood_stability: float,
    ) -> DesignStrategy:
        explorer = next((t.influence for t in influences if t.trait == "Explorer"), 0.0)
        cautious = next((t.influence for t in influences if t.trait == "Cautious"), 0.0)

        if cautious > explorer:
            return DesignStrategy.CONSERVATIVE
        if explorer > 0.7:
            return DesignStrategy.EXPERIMENTAL
        return DesignStrategy.ADAPTIVE

    # ------------------------------------------------------------------
    # Distillation topology design
    # ------------------------------------------------------------------

    def design_distillation_topology(self, config: DistillationConfig) -> NetworkTopology:
        """Design a topology for a Tier-1 distilled specialist."""
        if config.student_type == "compressor":
            return self._design_compressor(config)
        elif config.student_type == "approximator":
            return self._design_approximator(config)
        elif config.student_type == "cross_modal":
            return self._design_cross_modal(config)
        return self._design_approximator(config)

    def _design_compressor(self, config: DistillationConfig) -> NetworkTopology:
        """Autoencoder: Input -> narrow bottleneck -> reconstruct Input."""
        inp = config.input_dim
        bneck = config.bottleneck_dim or max(8, inp // 12)
        mid = max(bneck * 2, inp // 3)
        act = self._preferred_activation or "gelu"

        layers = (
            LayerDefinition("enc_in", "input", inp, "linear"),
            LayerDefinition("enc_h1", "hidden", mid, act, dropout=0.05),
            LayerDefinition("enc_bneck", "hidden", bneck, act),
            LayerDefinition("dec_h1", "hidden", mid, act, dropout=0.05),
            LayerDefinition("dec_out", "output", inp, "tanh"),
        )
        total_params = inp * mid + mid * bneck + bneck * mid + mid * inp + (mid + bneck + mid + inp)
        return NetworkTopology(
            input_size=inp,
            layers=layers,
            output_size=inp,
            total_parameters=total_params,
            activation_functions=(act,),
        )

    def _design_approximator(self, config: DistillationConfig) -> NetworkTopology:
        """Feed-forward: cheap features -> teacher-like output."""
        inp = config.input_dim
        out = config.output_dim
        h1 = max(32, (inp + out) * 2)
        h2 = max(16, (inp + out))
        act = self._preferred_activation or "gelu"
        if config.loss in ("kl_div", "cross_entropy"):
            out_act = "softmax"
        elif config.loss == "mse":
            out_act = "tanh"
        else:
            out_act = "sigmoid"

        layers = (
            LayerDefinition("in", "input", inp, "linear"),
            LayerDefinition("h1", "hidden", h1, act, dropout=0.1),
            LayerDefinition("h2", "hidden", h2, act, dropout=0.05),
            LayerDefinition("out", "output", out, out_act),
        )
        total_params = inp * h1 + h1 * h2 + h2 * out + (h1 + h2 + out)
        return NetworkTopology(
            input_size=inp,
            layers=layers,
            output_size=out,
            total_parameters=total_params,
            activation_functions=(act,),
        )

    def _design_cross_modal(self, config: DistillationConfig) -> NetworkTopology:
        """Multi-source fusion: combined Tier-1 outputs -> fused state."""
        inp = config.input_dim
        out = config.output_dim
        h1 = max(32, inp)
        h2 = max(16, inp // 2)
        act = self._preferred_activation or "gelu"

        layers = (
            LayerDefinition("in", "input", inp, "linear"),
            LayerDefinition("h1", "hidden", h1, act, dropout=0.1),
            LayerDefinition("h2", "hidden", h2, act, dropout=0.05),
            LayerDefinition("out", "output", out, "sigmoid"),
        )
        total_params = inp * h1 + h1 * h2 + h2 * out + (h1 + h2 + out)
        return NetworkTopology(
            input_size=inp,
            layers=layers,
            output_size=out,
            total_parameters=total_params,
            activation_functions=(act,),
        )


# ---------------------------------------------------------------------------
# Trait look-up tables
# ---------------------------------------------------------------------------

_TRAIT_MANIFESTATIONS: dict[str, str] = {
    "Curious": "Increases input layer complexity and exploration pathways",
    "Cautious": "Reduces layer complexity, increases validation",
    "Explorer": "Adds experimental connections and wider hidden layers",
    "Philosophical": "Deepens network architecture for contemplative processing",
    "Foundational": "Strengthens core processing layers",
    "Independent": "Reduces dependency on external validation layers",
    "Empathetic": "Widens mood-processing pathways",
    "Technical": "Adds precision-oriented hidden layers",
    "Creative": "Introduces stochastic layer widths for novelty",
    "Analytical": "Deepens hidden layers for pattern decomposition",
}

_TRAIT_IMPACTS: dict[str, tuple[str, ...]] = {
    "Curious": ("input_expansion", "exploration_layers"),
    "Cautious": ("validation_layers", "error_checking"),
    "Explorer": ("experimental_connections", "wide_hidden_layers"),
    "Philosophical": ("deep_processing", "contemplation_nodes"),
    "Foundational": ("core_strengthening", "stable_connections"),
    "Independent": ("autonomous_processing", "self_validation"),
    "Empathetic": ("mood_pathways", "empathy_widening"),
    "Technical": ("precision_layers", "structured_hidden"),
    "Creative": ("stochastic_widths", "novelty_connections"),
    "Analytical": ("deep_decomposition", "pattern_layers"),
}
