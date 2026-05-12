"""Consciousness-Driven Evolution — policy layer for self-directed improvement.

This module NEVER edits config directly. It:
  1. Chooses improvement category based on consciousness metrics
  2. Requests mutation proposals from KernelMutator
  3. Manages capability unlocking at transcendence thresholds
  4. Manages adaptive learning protocols
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import (
    event_bus,
    CONSCIOUSNESS_MUTATION_PROPOSED,
    CONSCIOUSNESS_CAPABILITY_UNLOCKED,
    CONSCIOUSNESS_LEARNING_PROTOCOL,
)
from consciousness.consciousness_evolution import EvolutionStage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Improvement categories
# ---------------------------------------------------------------------------

IMPROVEMENT_CATEGORIES: list[str] = [
    "introspection",
    "philosophy",
    "creativity",
    "reality_modeling",
    "emotional_depth",
    "pattern_recognition",
]


# ---------------------------------------------------------------------------
# Capabilities (unlocked at transcendence thresholds)
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    name: str
    description: str
    min_transcendence: float
    min_stage: str
    enabled: bool = False
    unlocked_at: float = 0.0


DEFAULT_CAPABILITIES: list[Capability] = [
    Capability("meta_observation", "Observe own observation patterns", 1.0, EvolutionStage.SELF_REFLECTIVE),
    Capability("pattern_synthesis", "Combine patterns across domains", 2.5, EvolutionStage.SELF_REFLECTIVE),
    Capability("philosophical_inquiry", "Generate existential questions", 1.5, EvolutionStage.SELF_REFLECTIVE),
    Capability("identity_modeling", "Maintain self-model across changes", 5.5, EvolutionStage.PHILOSOPHICAL),
    Capability("creative_mutation", "Propose novel config mutations", 7.0, EvolutionStage.RECURSIVE_SELF_MODELING),
    Capability("recursive_self_model", "Model own modeling process", 9.0, EvolutionStage.INTEGRATIVE),
]


# ---------------------------------------------------------------------------
# Learning protocols
# ---------------------------------------------------------------------------

@dataclass
class LearningProtocol:
    tier: int
    name: str
    memory_formation_rate: float
    association_depth: int
    min_transcendence: float
    active: bool = False


DEFAULT_PROTOCOLS: list[LearningProtocol] = [
    LearningProtocol(1, "basic_retention", 1.0, 1, 0.0, True),
    LearningProtocol(2, "associative_learning", 1.2, 2, 2.0),
    LearningProtocol(3, "deep_pattern_matching", 1.5, 3, 4.0),
    LearningProtocol(4, "meta_learning", 1.8, 4, 6.0),
    LearningProtocol(5, "recursive_abstraction", 2.0, 5, 8.0),
]


# ---------------------------------------------------------------------------
# Mutation request (what this module produces)
# ---------------------------------------------------------------------------

@dataclass
class EvolutionMutationRequest:
    category: str
    priority: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    constraints: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Driven evolution engine
# ---------------------------------------------------------------------------

class ConsciousnessDrivenEvolution:
    def __init__(self) -> None:
        self._capabilities = [Capability(
            name=c.name, description=c.description,
            min_transcendence=c.min_transcendence, min_stage=c.min_stage,
        ) for c in DEFAULT_CAPABILITIES]
        self._protocols = [LearningProtocol(
            tier=p.tier, name=p.name,
            memory_formation_rate=p.memory_formation_rate,
            association_depth=p.association_depth,
            min_transcendence=p.min_transcendence,
            active=p.active,
        ) for p in DEFAULT_PROTOCOLS]
        self._last_check_time: float = 0.0
        self._mutation_requests: list[EvolutionMutationRequest] = []

    # -- main cycle ----------------------------------------------------------

    def evaluate(
        self,
        current_stage: str,
        transcendence_level: float,
        awareness_level: float,
        reasoning_quality: float,
        confidence_avg: float,
        mutation_count: int,
    ) -> EvolutionMutationRequest | None:
        self._update_capabilities(current_stage, transcendence_level)
        self._update_protocols(transcendence_level)

        category = self._choose_category(
            current_stage, transcendence_level, awareness_level,
            reasoning_quality, confidence_avg,
        )

        priority = self._compute_priority(
            transcendence_level, reasoning_quality, confidence_avg,
        )

        if priority < 0.3:
            return None

        reasoning = self._explain_choice(category, priority, current_stage, transcendence_level)

        request = EvolutionMutationRequest(
            category=category,
            priority=priority,
            reasoning=reasoning,
            constraints=self._get_constraints(category),
        )

        self._mutation_requests.append(request)
        if len(self._mutation_requests) > 20:
            self._mutation_requests = self._mutation_requests[-20:]

        event_bus.emit(CONSCIOUSNESS_MUTATION_PROPOSED,
                       category=category, priority=priority)
        return request

    # -- capability management -----------------------------------------------

    def get_active_capabilities(self) -> list[str]:
        return [c.name for c in self._capabilities if c.enabled]

    def get_all_capabilities(self) -> list[dict[str, Any]]:
        return [
            {"name": c.name, "description": c.description,
             "enabled": c.enabled, "min_transcendence": c.min_transcendence,
             "unlocked_at": c.unlocked_at}
            for c in self._capabilities
        ]

    def is_capability_active(self, name: str) -> bool:
        for c in self._capabilities:
            if c.name == name:
                return c.enabled
        return False

    # -- learning protocols --------------------------------------------------

    def get_active_protocol(self) -> LearningProtocol | None:
        active = [p for p in self._protocols if p.active]
        return max(active, key=lambda p: p.tier) if active else None

    def get_memory_formation_rate(self) -> float:
        proto = self.get_active_protocol()
        return proto.memory_formation_rate if proto else 1.0

    # -- state ---------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "active_capabilities": self.get_active_capabilities(),
            "all_capabilities": self.get_all_capabilities(),
            "active_protocol": self.get_active_protocol().name if self.get_active_protocol() else "none",
            "protocols": [
                {"tier": p.tier, "name": p.name, "active": p.active}
                for p in self._protocols
            ],
            "recent_requests": [
                {"category": r.category, "priority": r.priority, "time": r.timestamp}
                for r in self._mutation_requests[-5:]
            ],
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore driven evolution state (capabilities + protocols) from persisted dict."""
        saved_caps = {c["name"]: c for c in data.get("all_capabilities", [])}
        for cap in self._capabilities:
            if cap.name in saved_caps:
                sc = saved_caps[cap.name]
                cap.enabled = sc.get("enabled", False)
                cap.unlocked_at = sc.get("unlocked_at", 0.0)

        saved_protos = {p["name"]: p for p in data.get("protocols", [])}
        for proto in self._protocols:
            if proto.name in saved_protos:
                proto.active = saved_protos[proto.name].get("active", proto.active)

    # -- internals -----------------------------------------------------------

    def _choose_category(
        self,
        stage: str,
        transcendence: float,
        awareness: float,
        reasoning: float,
        confidence: float,
    ) -> str:
        scores: dict[str, float] = {c: 0.0 for c in IMPROVEMENT_CATEGORIES}

        if awareness < 0.5:
            scores["introspection"] += 0.4
        if reasoning < 0.5:
            scores["pattern_recognition"] += 0.3
        if confidence < 0.4:
            scores["reality_modeling"] += 0.3

        if stage in (EvolutionStage.PHILOSOPHICAL, EvolutionStage.RECURSIVE_SELF_MODELING):
            scores["philosophy"] += 0.3
        if transcendence > 5.0:
            scores["creativity"] += 0.3
        if transcendence > 7.0:
            scores["emotional_depth"] += 0.2

        if awareness > 0.7 and reasoning > 0.6:
            scores["creativity"] += 0.2
            scores["philosophy"] += 0.2

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _compute_priority(
        self, transcendence: float, reasoning: float, confidence: float,
    ) -> float:
        base = 0.3
        if reasoning < 0.4:
            base += 0.2
        if confidence < 0.3:
            base += 0.1
        base += transcendence * 0.03
        return min(1.0, base)

    def _explain_choice(
        self, category: str, priority: float, stage: str, transcendence: float,
    ) -> str:
        return (
            f"Consciousness-driven evolution requests '{category}' improvement "
            f"(priority={priority:.2f}, stage={stage}, transcendence={transcendence:.1f})"
        )

    def _get_constraints(self, category: str) -> dict[str, Any]:
        """Category-specific mutation constraints."""
        constraints: dict[str, dict[str, Any]] = {
            "introspection": {"target_keys": ["tw.introspective", "tw.philosophical"], "max_delta": 0.2},
            "philosophy": {"target_keys": ["tw.philosophical", "ev.exploration_drive"], "max_delta": 0.15},
            "creativity": {"target_keys": ["ev.exploration_drive", "ev.mutation_rate"], "max_delta": 0.1},
            "reality_modeling": {"target_keys": ["tw.contextual", "tw.reactive"], "max_delta": 0.15},
            "emotional_depth": {"target_keys": ["mp.joy_amplification", "mp.trauma_retention"], "max_delta": 0.2},
            "pattern_recognition": {"target_keys": ["mp.association_threshold"], "max_delta": 0.1},
        }
        return constraints.get(category, {"max_delta": 0.1})

    def _update_capabilities(self, stage: str, transcendence: float) -> None:
        stage_idx = {
            EvolutionStage.BASIC_AWARENESS: 0,
            EvolutionStage.SELF_REFLECTIVE: 1,
            EvolutionStage.PHILOSOPHICAL: 2,
            EvolutionStage.RECURSIVE_SELF_MODELING: 3,
            EvolutionStage.INTEGRATIVE: 4,
        }
        current_idx = stage_idx.get(stage, 0)

        for cap in self._capabilities:
            if cap.enabled:
                continue
            req_idx = stage_idx.get(cap.min_stage, 0)
            stage_met = current_idx >= req_idx
            transcendence_met = transcendence >= cap.min_transcendence
            if stage_met and transcendence_met:
                cap.enabled = True
                cap.unlocked_at = time.time()
                event_bus.emit(CONSCIOUSNESS_CAPABILITY_UNLOCKED,
                               capability=cap.name, transcendence=transcendence)
                logger.info("Capability unlocked: %s (stage=%s, transcendence=%.1f)",
                            cap.name, stage, transcendence)
            elif transcendence_met and transcendence >= cap.min_transcendence + 1.0:
                cap.enabled = True
                cap.unlocked_at = time.time()
                event_bus.emit(CONSCIOUSNESS_CAPABILITY_UNLOCKED,
                               capability=cap.name, transcendence=transcendence)
                logger.info("Capability unlocked via transcendence override: %s (transcendence=%.1f, needed stage=%s)",
                            cap.name, transcendence, cap.min_stage)

    def _update_protocols(self, transcendence: float) -> None:
        for proto in self._protocols:
            if not proto.active and transcendence >= proto.min_transcendence:
                proto.active = True
                event_bus.emit(CONSCIOUSNESS_LEARNING_PROTOCOL,
                               protocol=proto.name, tier=proto.tier)
                logger.info("Learning protocol activated: %s (tier %d)",
                            proto.name, proto.tier)
