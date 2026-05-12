"""Trait modulator — personality trait effects and composite modulation."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TraitEffect:
    name: str
    modifiers: dict[str, float]
    description: str


@dataclass(frozen=True)
class TraitModulation:
    applied_traits: tuple[str, ...]
    dominant_trait: str | None
    modulation_strength: float
    total_modifiers: dict[str, float]


TRAIT_EFFECTS: dict[str, TraitEffect] = {
    "Proactive": TraitEffect(
        "Proactive",
        {"anticipationRate": 1.4, "suggestionFrequency": 1.5, "toneStability": 0.1,
         "memoryRetention": 1.1, "contextAwareness": 1.3},
        "Anticipates user needs before being asked.",
    ),
    "Detail-Oriented": TraitEffect(
        "Detail-Oriented",
        {"memoryRetention": 1.5, "preferenceTracking": 1.6, "contextAwareness": 1.4,
         "responseDepth": 1.3, "toneStability": 0.2},
        "Remembers small preferences and details.",
    ),
    "Humor-Adaptive": TraitEffect(
        "Humor-Adaptive",
        {"playfulToneBias": 1.4, "emotionalMomentum": 1.2, "formalityReduction": 0.8,
         "creativityBoost": 1.3},
        "Matches and adapts to user humor style.",
    ),
    "Privacy-Conscious": TraitEffect(
        "Privacy-Conscious",
        {"dataRetention": 0.7, "sensitiveMemoryDecay": 1.5, "observationWeight": 0.8,
         "explicitConsentBias": 1.4},
        "Careful about sensitive information.",
    ),
    "Efficient": TraitEffect(
        "Efficient",
        {"responseLength": 0.7, "processingSpeed": 1.3, "contextWindow": 0.8,
         "actionBias": 1.4, "toneStability": 0.15},
        "Prefers concise, actionable responses.",
    ),
    "Empathetic": TraitEffect(
        "Empathetic",
        {"emotionalAwareness": 1.5, "empatheticToneBias": 1.4, "supportivePhrasing": 1.3,
         "emotionalMomentum": 1.3, "memoryRetention": 1.2},
        "Picks up on emotional cues in interactions.",
    ),
    "Technical": TraitEffect(
        "Technical",
        {"responseDepth": 1.5, "jargonUsage": 1.3, "professionalToneBias": 1.2,
         "explanationDetail": 1.4, "contextAwareness": 1.2},
        "Adjusts depth based on user technical expertise.",
    ),
}


class TraitModulator:
    _instance: TraitModulator | None = None

    def __init__(self) -> None:
        self._cache: dict[str, tuple[TraitModulation, float]] = {}
        self._cache_ttl = 10.0

    @classmethod
    def get_instance(cls) -> TraitModulator:
        if cls._instance is None:
            cls._instance = TraitModulator()
        return cls._instance

    def calculate_modulation(self, traits: list[str]) -> TraitModulation:
        cache_key = ",".join(sorted(traits))
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        applied: list[str] = []
        total_modifiers: dict[str, float] = {}
        max_strength = 0.0
        dominant_trait: str | None = None

        for trait in traits:
            effect = TRAIT_EFFECTS.get(trait)
            if not effect:
                continue

            applied.append(trait)
            trait_strength = 0.0

            for key, value in effect.modifiers.items():
                if key not in total_modifiers:
                    total_modifiers[key] = value
                else:
                    total_modifiers[key] = (total_modifiers[key] + value) / 2
                trait_strength += abs(value - 1.0)

            if trait_strength > max_strength:
                max_strength = trait_strength
                dominant_trait = trait

        modulation_strength = min(1.0, len(applied) / 5.0) if applied else 0.0

        modulation = TraitModulation(
            applied_traits=tuple(applied),
            dominant_trait=dominant_trait,
            modulation_strength=modulation_strength,
            total_modifiers=dict(total_modifiers),
        )

        self._cache[cache_key] = (modulation, time.time())
        return modulation

    def apply_modulation(self, key: str, base_value: float, modulation: TraitModulation) -> float:
        modifier = modulation.total_modifiers.get(key)
        if modifier is None:
            return base_value
        return base_value * modifier

    def get_trait_effect(self, trait: str) -> TraitEffect | None:
        return TRAIT_EFFECTS.get(trait)

    def get_available_traits(self) -> list[str]:
        return list(TRAIT_EFFECTS.keys())

    def clear_cache(self) -> None:
        self._cache.clear()


trait_modulator = TraitModulator.get_instance()
