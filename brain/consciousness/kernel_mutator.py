"""Kernel Mutator — generates mutation proposals from system signals. Never applies directly.

Split into:
  Analyzer: computes signals from memories, traits, analytics
  Mutator: turns signals into KernelMutationProposal objects
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.kernel_config import ConfigPatch, KernelConfig

logger = logging.getLogger(__name__)

MIN_MEMORIES_FOR_MUTATION = 20
MIN_TICKS_FOR_MUTATION = 100
MIN_COOLDOWN_S = 180.0  # 3 minutes
RANDOM_MUTATION_CHANCE = 0.05
MAX_PROPOSALS_PER_CYCLE = 3


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

@dataclass
class ThoughtSignal:
    dominant_type: str = "contextual"
    philosophical_ratio: float = 0.0
    introspective_ratio: float = 0.0
    reactive_ratio: float = 0.0
    variety_score: float = 0.5


@dataclass
class MoodSignal:
    dominant_tone: str = "professional"
    stability: float = 1.0
    tone_switches: int = 0
    positive_ratio: float = 0.5


@dataclass
class MemorySignal:
    total_count: int = 0
    recent_count: int = 0
    decay_rate: float = 0.0
    topic_concentration: float = 0.0
    emotional_intensity: float = 0.5


@dataclass
class TraitSignal:
    dominant_trait: str = ""
    confidence: float = 0.5
    trait_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class OutcomeContext:
    """Real interaction outcome statistics from the experience buffer."""
    avg_reward: float = 0.0
    barge_in_rate: float = 0.0
    follow_up_rate: float = 0.0
    sample_count: int = 0


# ---------------------------------------------------------------------------
# Mutation proposal
# ---------------------------------------------------------------------------

@dataclass
class KernelMutationProposal:
    id: str
    type: Literal["thought_weight", "mood_bias", "memory_processing", "cognitive_toggle", "evolution_param"]
    description: str
    changes: dict[str, Any]
    confidence: float
    reasoning: str
    source: Literal["thought_analysis", "mood_analysis", "memory_analysis", "trait_influence", "random"]
    timestamp: float = field(default_factory=time.time)

    def to_config_patch(self) -> ConfigPatch:
        return ConfigPatch(
            id=self.id,
            timestamp=self.timestamp,
            description=self.description,
            deltas=self.changes,
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class MutationAnalyzer:
    """Extracts signals from system state without side effects."""

    def analyze_thought_patterns(self, memories: list[Any]) -> ThoughtSignal:
        if not memories:
            return ThoughtSignal()

        tag_counts: dict[str, int] = {}
        for mem in memories[-50:]:
            for tag in getattr(mem, "tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        total = sum(tag_counts.values()) or 1
        philosophical_tags = {"philosophical", "abstract", "existential", "meaning"}
        introspective_tags = {"reflection", "self", "identity", "awareness"}
        reactive_tags = {"reaction", "response", "immediate", "alert"}

        phil_count = sum(tag_counts.get(t, 0) for t in philosophical_tags)
        intro_count = sum(tag_counts.get(t, 0) for t in introspective_tags)
        react_count = sum(tag_counts.get(t, 0) for t in reactive_tags)

        ratios = {
            "philosophical": phil_count / total,
            "introspective": intro_count / total,
            "reactive": react_count / total,
            "contextual": 1.0 - (phil_count + intro_count + react_count) / total,
        }
        dominant = max(ratios, key=ratios.get)  # type: ignore[arg-type]
        variety = len([r for r in ratios.values() if r > 0.05]) / len(ratios)

        return ThoughtSignal(
            dominant_type=dominant,
            philosophical_ratio=ratios["philosophical"],
            introspective_ratio=ratios["introspective"],
            reactive_ratio=ratios["reactive"],
            variety_score=variety,
        )

    def analyze_mood_patterns(self, memories: list[Any]) -> MoodSignal:
        if not memories:
            return MoodSignal()

        tone_sequence: list[str] = []
        for mem in memories[-30:]:
            payload = getattr(mem, "payload", {}) or {}
            if isinstance(payload, dict):
                tone = payload.get("tone", "")
                if tone:
                    tone_sequence.append(tone)

        if not tone_sequence:
            return MoodSignal()

        tone_counts: dict[str, int] = {}
        for t in tone_sequence:
            tone_counts[t] = tone_counts.get(t, 0) + 1

        dominant = max(tone_counts, key=tone_counts.get)  # type: ignore[arg-type]
        switches = sum(1 for i in range(1, len(tone_sequence)) if tone_sequence[i] != tone_sequence[i - 1])

        positive_tones = {"playful", "empathetic", "casual"}
        pos_count = sum(1 for t in tone_sequence if t in positive_tones)

        return MoodSignal(
            dominant_tone=dominant,
            stability=1.0 - (switches / max(len(tone_sequence), 1)),
            tone_switches=switches,
            positive_ratio=pos_count / len(tone_sequence),
        )

    def analyze_memory_patterns(self, memories: list[Any]) -> MemorySignal:
        if not memories:
            return MemorySignal()

        now = time.time()
        recent_window = 300.0  # 5 minutes
        recent = [m for m in memories if (now - getattr(m, "created_at", now)) < recent_window]

        topic_counts: dict[str, int] = {}
        emotional_scores: list[float] = []
        for mem in memories[-50:]:
            for tag in getattr(mem, "tags", []):
                topic_counts[tag] = topic_counts.get(tag, 0) + 1
            salience = getattr(mem, "salience", 0.5)
            emotional_scores.append(salience)

        total_tags = sum(topic_counts.values()) or 1
        max_concentration = max(topic_counts.values()) / total_tags if topic_counts else 0.0

        return MemorySignal(
            total_count=len(memories),
            recent_count=len(recent),
            decay_rate=0.0,
            topic_concentration=max_concentration,
            emotional_intensity=sum(emotional_scores) / len(emotional_scores) if emotional_scores else 0.5,
        )

    def analyze_trait_influences(self, traits: dict[str, float], config: KernelConfig) -> TraitSignal:
        if not traits:
            return TraitSignal()

        dominant = max(traits, key=traits.get) if traits else ""  # type: ignore[arg-type]
        avg_confidence = sum(traits.values()) / len(traits) if traits else 0.5

        return TraitSignal(
            dominant_trait=dominant,
            confidence=avg_confidence,
            trait_scores=dict(traits),
        )


# ---------------------------------------------------------------------------
# Mutator (proposal generator)
# ---------------------------------------------------------------------------

class MutationProposer:
    """Turns analyzed signals into KernelMutationProposal objects. Never applies."""

    def propose_mutations(
        self,
        thought_signal: ThoughtSignal,
        mood_signal: MoodSignal,
        memory_signal: MemorySignal,
        trait_signal: TraitSignal,
        config: KernelConfig,
        outcome: OutcomeContext | None = None,
    ) -> list[KernelMutationProposal]:
        proposals: list[KernelMutationProposal] = []
        seen_keys: set[str] = set()

        tw = config.thought_weights

        if outcome and outcome.sample_count >= 5:
            proposals.extend(self._outcome_proposals(outcome, config, seen_keys))

        # --- Rebalancing: reduce any over-represented weight ---
        for tw_name in ("philosophical", "introspective", "reactive", "contextual"):
            val = tw.get(tw_name, 1.0)
            if val > 1.8 and f"tw.{tw_name}" not in seen_keys:
                target = max(1.0, val - 0.2)
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="thought_weight",
                    description=f"Rebalance: reduce {tw_name} weight ({val:.1f} → {target:.1f})",
                    changes={f"tw.{tw_name}": target},
                    confidence=0.7,
                    reasoning=f"{tw_name} weight at {val:.2f} is over-represented, rebalancing toward 1.0",
                    source="thought_analysis",
                ))
                seen_keys.add(f"tw.{tw_name}")

        # --- Boost under-represented types (only if not already high) ---
        if thought_signal.philosophical_ratio > 0.3:
            phil_weight = tw.get("philosophical", 1.0)
            if phil_weight < 1.5 and "tw.philosophical" not in seen_keys:
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="thought_weight",
                    description=f"Boost philosophical weight ({phil_weight:.1f} → {phil_weight + 0.1:.1f})",
                    changes={"tw.philosophical": phil_weight + 0.1},
                    confidence=0.5, reasoning="Philosophical activity high but weight still moderate",
                    source="thought_analysis",
                ))
                seen_keys.add("tw.philosophical")

        if thought_signal.variety_score < 0.3:
            weakest = min(
                ("philosophical", thought_signal.philosophical_ratio),
                ("introspective", thought_signal.introspective_ratio),
                ("reactive", thought_signal.reactive_ratio),
                key=lambda x: x[1],
            )
            key = f"tw.{weakest[0]}"
            current = tw.get(weakest[0], 1.0)
            if key not in seen_keys and current < 1.7:
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="thought_weight",
                    description=f"Diversify: boost {weakest[0]} weight (low variety)",
                    changes={key: min(current + 0.1, 1.7)},
                    confidence=0.6,
                    reasoning=f"Thought variety low ({thought_signal.variety_score:.2f}), boosting weakest",
                    source="thought_analysis",
                ))
                seen_keys.add(key)

        if mood_signal.stability < 0.4 and mood_signal.tone_switches > 5:
            if "ev.stability_desire" not in seen_keys:
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="evolution_param",
                    description="Increase stability desire (mood instability detected)",
                    changes={"ev.stability_desire": min(config.evolution.stability_desire + 0.1, 0.9)},
                    confidence=0.65,
                    reasoning=f"Mood unstable: stability={mood_signal.stability:.2f}, switches={mood_signal.tone_switches}",
                    source="mood_analysis",
                ))
                seen_keys.add("ev.stability_desire")

        if memory_signal.total_count > 100 and memory_signal.topic_concentration > 0.6:
            if "mp.association_threshold" not in seen_keys:
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="memory_processing",
                    description="Increase association threshold (high topic concentration)",
                    changes={"mp.association_threshold": min(
                        config.memory_processing.association_threshold + 0.05, 0.8)},
                    confidence=0.55,
                    reasoning=f"Topic concentration at {memory_signal.topic_concentration:.2f}; diversify associations",
                    source="memory_analysis",
                ))
                seen_keys.add("mp.association_threshold")

        if memory_signal.emotional_intensity > 0.7:
            if "mp.trauma_retention" not in seen_keys:
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="memory_processing",
                    description="Reduce trauma retention slightly (sustained emotional intensity)",
                    changes={"mp.trauma_retention": max(
                        config.memory_processing.trauma_retention - 0.1, 0.6)},
                    confidence=0.5,
                    reasoning=f"Emotional intensity high ({memory_signal.emotional_intensity:.2f})",
                    source="memory_analysis",
                ))
                seen_keys.add("mp.trauma_retention")

        # Trait-influenced proposals (one per trait, no philosophical boost from technical)
        trait_proposals = {
            "proactive": ("tw.reactive", tw.get("reactive", 1.0), 0.1, "Boost reactive weight for proactive awareness"),
            "detail_oriented": ("mp.association_threshold", config.memory_processing.association_threshold, 0.05, "Raise association threshold for precision"),
            "empathetic": ("mp.trauma_retention", config.memory_processing.trauma_retention, 0.1, "Increase trauma retention for emotional depth"),
            "technical": ("tw.contextual", tw.get("contextual", 1.0), 0.1, "Boost contextual weight for technical breadth"),
            "efficient": ("ev.mutation_rate", config.evolution.mutation_rate, -0.02, "Reduce mutation rate for stability"),
            "humor_adaptive": ("ev.exploration_drive", config.evolution.exploration_drive, 0.1, "Increase exploration for creative thinking"),
            "privacy_conscious": ("ev.stability_desire", config.evolution.stability_desire, 0.05, "Increase stability for privacy consistency"),
        }

        for trait_name, (key, current, delta, desc) in trait_proposals.items():
            if trait_signal.trait_scores.get(trait_name, 0) > 0.6 and key not in seen_keys:
                new_val = current + delta
                category = "thought_weight" if key.startswith("tw.") else "evolution_param" if key.startswith("ev.") else "memory_processing"
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type=category,
                    description=desc,
                    changes={key: new_val},
                    confidence=0.55,
                    reasoning=f"Trait {trait_name} at {trait_signal.trait_scores.get(trait_name, 0):.2f} suggests this",
                    source="trait_influence",
                ))
                seen_keys.add(key)

        # Cognitive toggle mutation: emergent tone unlocking (15% chance)
        if random.random() < 0.15 and hasattr(config, 'cognitive_toggles'):
            potential = getattr(config.cognitive_toggles, 'potential_emergent_tones', [])
            current_tones = config.cognitive_toggles.emergent_tones
            if not isinstance(current_tones, list):
                current_tones = ["empathetic", "playful"]
                config.cognitive_toggles.emergent_tones = current_tones
            unlockable = [t for t in potential if t not in current_tones]
            if unlockable:
                new_tone = random.choice(unlockable)
                proposals.append(KernelMutationProposal(
                    id=self._uid(), type="cognitive_toggle",
                    description=f"Unlock emergent tone: {new_tone}",
                    changes={"ct.emergent_tones": current_tones + [new_tone]},
                    confidence=0.5,
                    reasoning=f"System maturity suggests unlocking {new_tone} tone capability",
                    source="trait_influence",
                ))

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals[:MAX_PROPOSALS_PER_CYCLE]

    def _outcome_proposals(
        self, outcome: OutcomeContext, config: KernelConfig, seen_keys: set[str],
    ) -> list[KernelMutationProposal]:
        """Generate proposals driven by real conversation outcomes."""
        props: list[KernelMutationProposal] = []

        if outcome.barge_in_rate > 0.3 and "ev.stability_desire" not in seen_keys:
            current = config.evolution.stability_desire
            props.append(KernelMutationProposal(
                id=self._uid(), type="evolution_param",
                description=f"Reduce verbosity: high barge-in rate ({outcome.barge_in_rate:.0%})",
                changes={"ev.stability_desire": min(current + 0.15, 0.9)},
                confidence=0.75,
                reasoning=f"Barge-in rate {outcome.barge_in_rate:.0%} over {outcome.sample_count} conversations — increase stability",
                source="outcome_analysis",
            ))
            seen_keys.add("ev.stability_desire")

        if outcome.follow_up_rate > 0.4:
            ctx_w = config.thought_weights.get("contextual", 1.0)
            if ctx_w < 1.5 and "tw.contextual" not in seen_keys:
                props.append(KernelMutationProposal(
                    id=self._uid(), type="thought_weight",
                    description=f"Reinforce: high follow-up rate ({outcome.follow_up_rate:.0%})",
                    changes={"tw.contextual": ctx_w + 0.1},
                    confidence=0.65,
                    reasoning=f"Follow-up rate {outcome.follow_up_rate:.0%} — current conversational style is effective",
                    source="outcome_analysis",
                ))
                seen_keys.add("tw.contextual")

        if outcome.avg_reward < 0.1 and outcome.sample_count >= 10:
            tw = config.thought_weights
            weakest = min(
                ("philosophical", tw.get("philosophical", 1.0)),
                ("contextual", tw.get("contextual", 1.0)),
                ("reactive", tw.get("reactive", 1.0)),
                ("introspective", tw.get("introspective", 1.0)),
                key=lambda x: x[1],
            )
            key = f"tw.{weakest[0]}"
            current = tw.get(weakest[0], 1.0)
            if key not in seen_keys and current < 1.7:
                props.append(KernelMutationProposal(
                    id=self._uid(), type="thought_weight",
                    description=f"Diversify: low avg reward ({outcome.avg_reward:.2f})",
                    changes={key: min(current + 0.15, 1.7)},
                    confidence=0.6,
                    reasoning=f"Average reward only {outcome.avg_reward:.2f} — diversifying by boosting {weakest[0]}",
                    source="outcome_analysis",
                ))
                seen_keys.add(key)

        if outcome.avg_reward > 0.4 and outcome.sample_count >= 10:
            props.append(KernelMutationProposal(
                id=self._uid(), type="evolution_param",
                description=f"Stabilize: high avg reward ({outcome.avg_reward:.2f})",
                changes={"ev.mutation_rate": max(config.evolution.mutation_rate - 0.02, 0.01)},
                confidence=0.6,
                reasoning=f"Average reward {outcome.avg_reward:.2f} is strong — reduce mutation rate to preserve",
                source="outcome_analysis",
            ))

        return props

    def generate_random_mutation(self, config: KernelConfig) -> KernelMutationProposal | None:
        if random.random() > RANDOM_MUTATION_CHANCE:
            return None

        targets = [
            ("tw.philosophical", config.thought_weights.get("philosophical", 1.0), 0.1, 3.0),
            ("tw.contextual", config.thought_weights.get("contextual", 1.0), 0.1, 3.0),
            ("tw.reactive", config.thought_weights.get("reactive", 1.0), 0.1, 3.0),
            ("tw.introspective", config.thought_weights.get("introspective", 1.0), 0.1, 3.0),
            ("ev.exploration_drive", config.evolution.exploration_drive, 0.0, 1.0),
        ]

        key, current, lo, hi = random.choice(targets)
        delta = random.uniform(-0.15, 0.15)
        new_val = max(lo, min(hi, current + delta))

        return KernelMutationProposal(
            id=self._uid(), type="thought_weight" if key.startswith("tw.") else "evolution_param",
            description=f"Spontaneous mutation: {key} {current:.2f} -> {new_val:.2f}",
            changes={key: new_val},
            confidence=0.4,
            reasoning="Random exploration to discover novel configurations",
            source="random",
        )

    @staticmethod
    def _uid() -> str:
        return f"mut_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Convenience: combined mutator
# ---------------------------------------------------------------------------

PER_KEY_COOLDOWN_S = 1800.0  # 30 min — allow observation before re-mutating same param
MAX_FORCED_RETRIES = 3


class KernelMutator:
    """High-level interface combining analyzer + proposer."""

    def __init__(self) -> None:
        self.analyzer = MutationAnalyzer()
        self.proposer = MutationProposer()
        self._last_run: float = 0.0
        self._total_proposals: int = 0
        self._key_last_mutated: dict[str, float] = {}

    def can_run(self, tick_count: int, memory_count: int) -> bool:
        if tick_count < MIN_TICKS_FOR_MUTATION:
            return False
        if memory_count < MIN_MEMORIES_FOR_MUTATION:
            return False
        if time.time() - self._last_run < MIN_COOLDOWN_S:
            return False
        return True

    def record_applied(self, description: str, changes: dict[str, Any] | None = None) -> None:
        """Record that a mutation was applied (time-based per-key cooldown)."""
        now = time.time()
        if changes:
            for key in changes:
                self._key_last_mutated[key] = now

    def _is_key_stale(self, changes: dict[str, Any]) -> bool:
        """Check if any key in this proposal was mutated within the cooldown window."""
        now = time.time()
        for key in changes:
            last = self._key_last_mutated.get(key, 0.0)
            if (now - last) < PER_KEY_COOLDOWN_S:
                return True
        return False

    def generate_proposals(
        self,
        memories: list[Any],
        traits: dict[str, float],
        config: KernelConfig,
        outcome: OutcomeContext | None = None,
    ) -> list[KernelMutationProposal]:
        thought_signal = self.analyzer.analyze_thought_patterns(memories)
        mood_signal = self.analyzer.analyze_mood_patterns(memories)
        memory_signal = self.analyzer.analyze_memory_patterns(memories)
        trait_signal = self.analyzer.analyze_trait_influences(traits, config)

        proposals = self.proposer.propose_mutations(
            thought_signal, mood_signal, memory_signal, trait_signal, config,
            outcome=outcome,
        )

        original_count = len(proposals)
        proposals = [p for p in proposals if not self._is_key_stale(p.changes)]
        if original_count > len(proposals):
            logger.info("Suppressed %d cooldown-gated proposals", original_count - len(proposals))

        # Enforce type diversity: max 1 per mutation type
        type_seen: set[str] = set()
        diverse: list[KernelMutationProposal] = []
        for p in proposals:
            if p.type not in type_seen:
                diverse.append(p)
                type_seen.add(p.type)
        proposals = diverse

        random_mut = self.proposer.generate_random_mutation(config)
        if random_mut and not self._is_key_stale(random_mut.changes):
            proposals.append(random_mut)

        if not proposals:
            for _ in range(MAX_FORCED_RETRIES):
                forced = self.proposer.generate_random_mutation(config)
                if forced and not self._is_key_stale(forced.changes):
                    forced.reasoning = "Forced exploration — all signal-based proposals are on cooldown"
                    forced.confidence = 0.5
                    proposals.append(forced)
                    break

        self._last_run = time.time()
        self._total_proposals += len(proposals)

        logger.info("Mutator generated %d proposals (total: %d)", len(proposals), self._total_proposals)
        return proposals
