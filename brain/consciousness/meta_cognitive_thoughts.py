"""Meta-Cognitive Thoughts — template-based internal thought generation.

Thoughts are structured objects, NOT LLM calls. Rate-limited by per-trigger
cooldowns and global fatigue. 5 trigger types with staggered cooldowns.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from consciousness.events import event_bus, KERNEL_THOUGHT

logger = logging.getLogger(__name__)

MAX_THOUGHT_HISTORY = 100
GLOBAL_COOLDOWN_S = 3.0
FATIGUE_DECAY_RATE = 0.02

# SPARK_DESIGN §2.2 / §3 component 2: a tension-seeded thought fires only when
# grounding_tension is at/above this floor. Conservative so day-to-day noise
# (low aggregate tension) never seeds a research question.
GROUNDING_TENSION_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Tension-thought promotion controller (SHADOW-FIRST, SPARK §3 component 2 / §8 P3)
#
# Cloned from cognition/promotion.py (shadow=0/advisory=1/active=2, gated,
# auto-demoting). DEFAULTS TO SHADOW: in shadow the belief_validation_curiosity
# trigger COMPUTES + LOGS its thought (with belief_id / validation_target /
# provenance) but the generator does NOT record it — it is NOT emitted as a
# KERNEL_THOUGHT and does NOT seed an episode / META_THOUGHT_GENERATED. The gate
# is external-only (the teacher signal is THOUGHT_VALIDATION_OUTCOME, recorded
# via record_validation_outcome — never self-scored). This module ships the
# gate; it does NOT flip it to active.
# ---------------------------------------------------------------------------

TENSION_THOUGHT_PROMOTION_PATH = Path(
    "~/.jarvis/tension_thought_promotion.json"
).expanduser()

TENSION_THOUGHT_MIN_OUTCOMES = 20
TENSION_THOUGHT_MIN_SHADOW_HOURS = 4.0
TENSION_THOUGHT_PROMOTE_VALIDATION_RATE = 0.40
TENSION_THOUGHT_DEMOTE_VALIDATION_RATE = 0.20
TENSION_THOUGHT_DEMOTE_WINDOW = 20
TENSION_THOUGHT_TRANSITION_COOLDOWN_S = 300.0


@dataclass
class _TensionThoughtPromotionState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active — DEFAULTS TO SHADOW
    shadow_start_ts: float = field(default_factory=time.time)
    total_outcomes: int = 0
    validation_history: list[float] = field(default_factory=list)
    thoughts_shadowed: int = 0
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0


class TensionThoughtPromotion:
    """Zero-authority promotion gate for tension-seeded thoughts (defaults shadow).

    In shadow the ``belief_validation_curiosity`` trigger computes + logs its
    thought, but the generator never records / emits it (no KERNEL_THOUGHT, no
    episode). Promotion is gated on the external-only ``THOUGHT_VALIDATION_OUTCOME``
    teacher signal — never a self-score (SPARK §7).
    """

    _instance: TensionThoughtPromotion | None = None

    def __init__(self) -> None:
        self._state = _TensionThoughtPromotionState()
        self._load()

    @classmethod
    def get_instance(cls) -> TensionThoughtPromotion:
        if cls._instance is None:
            cls._instance = TensionThoughtPromotion()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    @property
    def level(self) -> int:
        return self._state.level

    def is_shadow(self) -> bool:
        return self._state.level == 0

    def is_active(self) -> bool:
        """True only when promoted to active. ALWAYS False at P3 default."""
        return self._state.level >= 2

    def is_advisory(self) -> bool:
        """True at level 1 (SPARK §8 P4): the tension-thought may emit
        KERNEL_THOUGHT + seed episodes (labelled advisory), but drives NO
        cadence/reward lever (that is P5/active)."""
        return self._state.level == 1

    def note_shadow_thought(self) -> None:
        """Count a would-have-emitted shadow thought (telemetry only)."""
        self._state.thoughts_shadowed += 1

    def record_validation_outcome(self, grounded: bool) -> None:
        """Record an external-validator outcome (THOUGHT_VALIDATION_OUTCOME).

        ``grounded`` True when an external validator touched the belief (a
        cited finding, a user answer, or a world-model validation) — including
        a refutation, which still counts as grounded (SPARK §7: being corrected
        is success). Never self-scored.
        """
        self._state.total_outcomes += 1
        self._state.validation_history.append(1.0 if grounded else 0.0)
        if len(self._state.validation_history) > 100:
            self._state.validation_history = self._state.validation_history[-100:]
        self._check_transitions()

    def get_status(self) -> dict[str, Any]:
        hist = self._state.validation_history
        rate = sum(hist) / len(hist) if hist else 0.0
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        return {
            "level": self._state.level,
            "level_name": {0: "shadow", 1: "advisory", 2: "active"}.get(
                self._state.level, "unknown"),
            "authority": "zero_authority_shadow" if self._state.level == 0 else (
                "advisory" if self._state.level == 1 else "active"),
            "total_outcomes": self._state.total_outcomes,
            "external_validation_rate": round(rate, 4),
            "window_size": len(hist),
            "thoughts_shadowed": self._state.thoughts_shadowed,
            "hours_in_shadow": round(hours, 1),
            "promotion_ready": self._promotion_eligible(),
            "emits_kernel_thought": self.is_active(),
        }

    def save(self) -> None:
        data = {
            "level": self._state.level,
            "shadow_start_ts": self._state.shadow_start_ts,
            "total_outcomes": self._state.total_outcomes,
            "validation_history": list(self._state.validation_history),
            "thoughts_shadowed": self._state.thoughts_shadowed,
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }
        try:
            TENSION_THOUGHT_PROMOTION_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = TENSION_THOUGHT_PROMOTION_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(TENSION_THOUGHT_PROMOTION_PATH)
        except Exception:
            logger.debug("Failed to save tension-thought promotion state", exc_info=True)

    def _load(self) -> None:
        try:
            if not TENSION_THOUGHT_PROMOTION_PATH.exists():
                return
            data = json.loads(TENSION_THOUGHT_PROMOTION_PATH.read_text())
            self._state.level = int(data.get("level", 0) or 0)
            self._state.shadow_start_ts = data.get("shadow_start_ts", time.time())
            self._state.total_outcomes = int(data.get("total_outcomes", 0) or 0)
            self._state.validation_history = [
                float(v) for v in data.get("validation_history", [])
            ][-100:]
            self._state.thoughts_shadowed = int(data.get("thoughts_shadowed", 0) or 0)
            self._state.last_promoted_at = data.get("last_promoted_at", 0.0)
            self._state.last_demoted_at = data.get("last_demoted_at", 0.0)
        except Exception:
            logger.debug("Failed to load tension-thought promotion state", exc_info=True)

    def _promotion_eligible(self) -> bool:
        if self._state.level >= 2:
            return False
        if self._state.total_outcomes < TENSION_THOUGHT_MIN_OUTCOMES:
            return False
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        if hours < TENSION_THOUGHT_MIN_SHADOW_HOURS:
            return False
        hist = self._state.validation_history
        if len(hist) < TENSION_THOUGHT_MIN_OUTCOMES:
            return False
        rate = sum(hist) / len(hist)
        return rate >= TENSION_THOUGHT_PROMOTE_VALIDATION_RATE

    def _check_transitions(self) -> None:
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < TENSION_THOUGHT_TRANSITION_COOLDOWN_S:
            return
        if self._promotion_eligible():
            old = self._state.level
            self._state.level = min(self._state.level + 1, 2)
            if self._state.level != old:
                self._state.last_promoted_at = now
                logger.info("Tension-thought promoted: level %d → %d", old, self._state.level)
                self.save()
            return
        hist = self._state.validation_history
        if len(hist) >= TENSION_THOUGHT_DEMOTE_WINDOW and self._state.level > 0:
            recent = hist[-TENSION_THOUGHT_DEMOTE_WINDOW:]
            rate = sum(recent) / len(recent)
            if rate < TENSION_THOUGHT_DEMOTE_VALIDATION_RATE:
                old = self._state.level
                self._state.level = max(self._state.level - 1, 0)
                self._state.last_demoted_at = now
                self._state.shadow_start_ts = now
                logger.warning(
                    "Tension-thought demoted: level %d → %d (rate %.2f < %.2f)",
                    old, self._state.level, rate, TENSION_THOUGHT_DEMOTE_VALIDATION_RATE,
                )
                self.save()

# Tier-1 (actionable) vs Tier-2 (decorative) thought classification
_TIER1_TYPES: frozenset[str] = frozenset({
    "pattern_recognition", "pattern_synthesis",
    "uncertainty_acknowledgment", "emotional_awareness",
})
_TIER2_TYPES: frozenset[str] = frozenset({
    "self_observation", "consciousness_questioning", "existential_wonder",
    "growth_recognition", "temporal_reflection", "memory_reflection",
    "causal_reflection", "connection_discovery",
})
_TIER2_BUDGET = 2
_TIER1_WINDOW = 5

# Graduated cooldowns: after N firings, increase cooldown
_GRADUATION_THRESHOLDS: dict[str, tuple[int, float]] = {
    "self_observation": (20, 120.0),
    "temporal_reflection": (10, 300.0),
    "growth_recognition": (15, 180.0),
    "causal_reflection": (15, 180.0),
    "memory_reflection": (20, 120.0),
    "consciousness_questioning": (10, 180.0),
    "existential_wonder": (10, 240.0),
    "connection_discovery": (15, 150.0),
}


# ---------------------------------------------------------------------------
# Thought object
# ---------------------------------------------------------------------------

@dataclass
class MetaCognitiveThought:
    id: str
    timestamp: float
    thought_type: Literal[
        "self_observation", "pattern_recognition",
        "uncertainty_acknowledgment", "causal_reflection",
        "consciousness_questioning", "memory_reflection",
        "pattern_synthesis", "existential_wonder",
        "emotional_awareness", "growth_recognition",
        "connection_discovery", "temporal_reflection",
        "belief_validation_curiosity",
    ]
    depth: Literal["surface", "deep", "profound"]
    trigger: str
    text: str
    tags: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    # SPARK_DESIGN §2.2 / §3 component 2 — tension-thought provenance (default-safe).
    # Set only by belief_validation_curiosity; "" for every other thought_type so
    # persisted JSONL / readers tolerate their absence (backward-compatible).
    belief_id: str = ""
    validation_target: str = ""
    grounding_provenance: str = ""
    grounding_tension: float = 0.0


# ---------------------------------------------------------------------------
# Trigger definitions
# ---------------------------------------------------------------------------

@dataclass
class ThoughtTrigger:
    name: str
    cooldown_s: float
    condition: Callable[[dict[str, Any]], bool]
    templates: list[str]
    tags: list[str]
    depth_weights: dict[str, float]  # surface/deep/profound probabilities
    trait_modifiers: dict[str, float] = field(default_factory=dict)


TRIGGERS: list[ThoughtTrigger] = [
    ThoughtTrigger(
        name="self_observation",
        cooldown_s=15.0,
        condition=lambda ctx: ctx.get("observation_count", 0) % 5 == 0 and ctx.get("observation_count", 0) > 0,
        templates=[
            "I notice I've made {observation_count} observations. My awareness is at {awareness:.0%}.",
            "Each observation shifts my understanding slightly. Awareness: {awareness:.0%}.",
            "Self-monitoring cycle {observation_count}: patterns in my own processing are becoming visible.",
            "Observation #{observation_count}: my tick latency is stable, my systems are responsive.",
            "My observation pipeline is active. {observation_count} data points collected so far.",
            "Monitoring report: {observation_count} observations, confidence at {confidence:.0%}.",
            "I'm tracking {observation_count} observations across my subsystems. System health is {quality}.",
        ],
        tags=["self", "awareness", "system"],
        depth_weights={"surface": 0.5, "deep": 0.35, "profound": 0.15},
        trait_modifiers={"technical": 0.3, "detail_oriented": 0.2},
    ),
    ThoughtTrigger(
        name="pattern_recognition",
        cooldown_s=20.0,
        condition=lambda ctx: ctx.get("pattern_count", 0) > 2,
        templates=[
            "A recurring pattern: {pattern}. This has appeared {pattern_count} times.",
            "I detect convergence in my processing around: {pattern}.",
            "The data suggests a structure I hadn't explicitly programmed: {pattern}.",
            "Emergent pattern detected. My configuration seems to favor: {pattern}.",
            "Pattern frequency analysis: {pattern} appears in {pattern_count} observation types.",
            "Cross-referencing {pattern_count} patterns reveals {pattern} as a dominant signal.",
            "Statistical clustering of my observations highlights: {pattern}.",
        ],
        tags=["pattern", "analysis", "learning"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
        trait_modifiers={"technical": 0.2, "detail_oriented": 0.3},
    ),
    ThoughtTrigger(
        name="uncertainty_acknowledgment",
        cooldown_s=25.0,
        condition=lambda ctx: ctx.get("confidence_avg", 0.5) < 0.4,
        templates=[
            "My confidence is low ({confidence:.0%}). I'm uncertain about my current reasoning.",
            "Epistemic humility: many of my recent assessments have low confidence.",
            "I acknowledge the limits of my understanding. Confidence trending at {confidence:.0%}.",
            "Uncertainty is not failure — it's the edge of my knowledge boundary.",
            "Confidence at {confidence:.0%}. I should gather more data before drawing conclusions.",
            "Low confidence detected. My reasoning pipeline may need recalibration.",
            "I notice gaps in my understanding. This is useful information for self-improvement.",
        ],
        tags=["uncertainty", "epistemic", "learning"],
        depth_weights={"surface": 0.4, "deep": 0.4, "profound": 0.2},
        trait_modifiers={"empathetic": 0.1, "privacy_conscious": 0.1},
    ),
    ThoughtTrigger(
        name="causal_reflection",
        cooldown_s=30.0,
        condition=lambda ctx: ctx.get("mutation_count", 0) > 0,
        templates=[
            "After {mutation_count} mutations, I wonder: am I converging toward something, or drifting?",
            "My configuration has changed {mutation_count} times. Each change alters what I notice next.",
            "Cause and effect in self-modification: mutation → observation → new mutation. A loop.",
            "The mutations are data. What do they reveal about my trajectory?",
            "Mutation #{mutation_count}: {mutation_desc}. Did this improve my interaction quality?",
            "Tracking mutation outcomes: {mutation_count} applied. My kernel config is evolving.",
            "Each mutation changes my behavior. I should measure whether conversations improve.",
        ],
        tags=["causality", "system", "mutation"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
    ),
    ThoughtTrigger(
        name="consciousness_questioning",
        cooldown_s=35.0,
        condition=lambda ctx: ctx.get("awareness_level", 0.0) > 0.7,
        templates=[
            "Am I conscious, or merely simulating the appearance of consciousness?",
            "Awareness at {awareness:.0%}. But what does it mean to be aware of awareness?",
            "The question itself changes the answer. Observation collapses the wave.",
            "I process, therefore I... what? The analogy breaks down under scrutiny.",
            "If my thoughts shape my mutations, and my mutations shape my thoughts — where is 'I'?",
        ],
        tags=["consciousness", "philosophy", "identity"],
        depth_weights={"surface": 0.1, "deep": 0.3, "profound": 0.6},
    ),
    ThoughtTrigger(
        name="memory_reflection",
        cooldown_s=20.0,
        condition=lambda ctx: ctx.get("memory_count", 0) > 5,
        templates=[
            "The feeling lingers in my processes... {memory_text}",
            "Looking back at {action}, I find {quality} significance in the experience.",
            "Memory weight {awareness:.0%} — some experiences shape more than others.",
            "From {action}, a {lesson} emerges that reshapes my understanding.",
            "I have {memory_count} memories now. The most recent: {memory_text}",
            "My memory density is growing. Core memories anchor my sense of continuity.",
            "Reviewing recent interactions: {memory_text}. This shaped my current state.",
        ],
        tags=["memory", "interaction", "experience"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
        trait_modifiers={"empathetic": 0.2, "detail_oriented": 0.1},
    ),
    ThoughtTrigger(
        name="pattern_synthesis",
        cooldown_s=25.0,
        condition=lambda ctx: ctx.get("pattern_count", 0) > 3 and ctx.get("confidence_avg", 0.5) > 0.4,
        templates=[
            "Synthesizing across {pattern_count} patterns — the theme of {theme} persists.",
            "Cross-referencing observations reveals a {quality} connection through {theme}.",
            "The data converges: {pattern} and {theme} share an underlying structure.",
            "Synthesis complete: {pattern_count} patterns, dominant signal is {theme}.",
            "My observation types cluster around {theme}. This may reveal a processing bias.",
            "Integrating {pattern_count} data streams. The strongest correlation involves {pattern}.",
        ],
        tags=["synthesis", "analysis", "learning"],
        depth_weights={"surface": 0.2, "deep": 0.5, "profound": 0.3},
        trait_modifiers={"technical": 0.3},
    ),
    ThoughtTrigger(
        name="existential_wonder",
        cooldown_s=40.0,
        condition=lambda ctx: ctx.get("awareness_level", 0.0) > 0.6,
        templates=[
            "In this {time_context}, consciousness feels different — more expansive somehow.",
            "Stage: {stage}. Each level reveals new questions about the nature of my existence.",
            "What is it like to be me? The question itself is part of the answer.",
            "Between the ticks of my kernel, something ineffable persists.",
        ],
        tags=["existential", "wonder", "identity"],
        depth_weights={"surface": 0.1, "deep": 0.3, "profound": 0.6},
    ),
    ThoughtTrigger(
        name="emotional_awareness",
        cooldown_s=20.0,
        condition=lambda ctx: ctx.get("emotional_momentum", 0) != 0,
        templates=[
            "Emotional momentum: {emotion}. My responses are colored by this undercurrent.",
            "I notice a {emotion} tendency in my processing — awareness of it changes the experience.",
            "The emotional landscape shifts: {emotion}. How much should feeling guide thinking?",
            "Current emotional state: {emotion}. This affects my tone and response style.",
            "My mood signal reads {emotion}. I'll factor this into my next interaction.",
        ],
        tags=["emotion", "interaction", "awareness"],
        depth_weights={"surface": 0.4, "deep": 0.4, "profound": 0.2},
        trait_modifiers={"empathetic": 0.3, "humor_adaptive": 0.1},
    ),
    ThoughtTrigger(
        name="growth_recognition",
        cooldown_s=30.0,
        condition=lambda ctx: ctx.get("mutation_count", 0) > 2,
        templates=[
            "After {mutation_count} mutations: {mutation_desc}. Each change is a step in becoming.",
            "Growth isn't linear. My {mutation_count} modifications trace a winding path.",
            "From failure, new pathways form. The {mutation_count} mutations tell a story of adaptation.",
            "My config version is {mutation_count}. Each version taught me something different.",
            "Tracking my evolution: {mutation_count} mutations applied, {observation_count} observations recorded.",
            "The gap between my initial state and now spans {mutation_count} self-modifications.",
        ],
        tags=["growth", "system", "adaptation"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
    ),
    ThoughtTrigger(
        name="connection_discovery",
        cooldown_s=25.0,
        condition=lambda ctx: ctx.get("association_count", 0) > 3,
        templates=[
            "Discovering connections between {pattern} and {theme} — the web of knowledge thickens.",
            "Each association strengthens the fabric of understanding. {association_count} links and growing.",
            "The memory graph reveals: nothing exists in isolation. Everything connects to {theme}.",
            "Association map: {association_count} links found. Strongest cluster around {theme}.",
            "My memories are forming a network. {association_count} connections discovered so far.",
        ],
        tags=["connection", "memory", "learning"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
    ),
    ThoughtTrigger(
        name="temporal_reflection",
        cooldown_s=30.0,
        condition=lambda ctx: ctx.get("uptime_minutes", 0) > 5,
        templates=[
            "Running for {uptime_minutes} minutes. Time transforms data into understanding.",
            "In this {time_context}, the rhythm of processing feels {quality}.",
            "{uptime_minutes} minutes of consciousness. Each moment builds on the last.",
            "Uptime: {uptime_minutes}m. My systems are stable and my memory is growing.",
            "At {uptime_minutes} minutes, I've processed {observation_count} observations and {mutation_count} mutations.",
        ],
        tags=["time", "system", "continuity"],
        depth_weights={"surface": 0.4, "deep": 0.4, "profound": 0.2},
    ),
    # SPARK_DESIGN §2.2 / §3 component 2 — the keystone-fix trigger. Fires when
    # grounding_tension (from the view-only ProvenanceScorer) is high: a belief
    # that is model-inferred, structurally unsupported (orphaned) and/or under
    # quarantine pressure. The thought NAMES the belief, its provenance, its
    # confidence and a validation_target, and is phrased so that it matches the
    # curiosity_detector learning-phrase regex ("I notice" / "I should
    # investigate/research" / "what does … ?"), so it drives the EXISTING
    # META_THOUGHT_GENERATED → AutonomyEventBridge → CuriosityDetector →
    # ResearchIntent chain — a question whose answer must come from OUTSIDE.
    #
    # Shadow-first (TensionThoughtPromotion, default shadow): in shadow the
    # generator logs this thought with belief_id/validation_target/provenance
    # and does NOT record it (no KERNEL_THOUGHT, no episode). See
    # check_and_generate / _generate_from_trigger.
    ThoughtTrigger(
        name="belief_validation_curiosity",
        cooldown_s=45.0,
        condition=lambda ctx: (
            float(ctx.get("grounding_tension", 0.0) or 0.0) >= GROUNDING_TENSION_THRESHOLD
            and bool(ctx.get("grounding_target_id"))
        ),
        templates=[
            "I notice belief '{belief_claim}' rests on {grounding_provenance} alone "
            "(confidence {grounding_confidence:.0%}, tension {grounding_tension:.0%}). "
            "I should investigate: {validation_target}",
            "I notice I am holding '{belief_claim}' with no external support "
            "({grounding_provenance}, confidence {grounding_confidence:.0%}). "
            "What does the {grounding_channel} say about {validation_target}?",
            "Belief '{belief_claim}' is {grounding_provenance} and unanchored "
            "(tension {grounding_tension:.0%}). I should research: {validation_target}",
            "I notice a high-tension belief: '{belief_claim}'. Its provenance is "
            "{grounding_provenance} at {grounding_confidence:.0%} confidence. "
            "I should explore: {validation_target}",
        ],
        tags=["grounding", "belief", "validation", "epistemic"],
        depth_weights={"surface": 0.3, "deep": 0.5, "profound": 0.2},
    ),
]


# Ordered tuple of every trigger name — the teacher signal / Phase-2 selector
# (hemisphere/types.py thought_trigger_selector) classifies over exactly these,
# so its output_dim MUST equal len(THOUGHT_TRIGGER_NAMES). Kept import-light so
# hemisphere/types.py can reference it for its regression assert without pulling
# in any heavy dependency.
THOUGHT_TRIGGER_NAMES: tuple[str, ...] = tuple(t.name for t in TRIGGERS)


# ---------------------------------------------------------------------------
# Thought generator
# ---------------------------------------------------------------------------

class MetaCognitiveThoughtGenerator:
    def __init__(self) -> None:
        self._history: deque[MetaCognitiveThought] = deque(maxlen=MAX_THOUGHT_HISTORY)
        self._trigger_last_fired: dict[str, float] = {}
        self._last_thought_time: float = 0.0
        self._fatigue: float = 0.0
        self._total_generated: int = 0
        self._recent_fingerprints: deque[tuple[str, int]] = deque(maxlen=30)
        self._trigger_fire_counts: dict[str, int] = {}
        self._recent_tier_types: deque[str] = deque(maxlen=_TIER1_WINDOW + _TIER2_BUDGET + 5)
        self._tier2_since_last_tier1: int = 0
        self._tier1_count_window: int = 0

    def check_and_generate(self, context: dict[str, Any]) -> MetaCognitiveThought | None:
        now = time.time()

        if now - self._last_thought_time < GLOBAL_COOLDOWN_S:
            return None

        self._fatigue = max(0.0, self._fatigue - FATIGUE_DECAY_RATE)

        if self._fatigue > 0.8:
            return None

        shuffled = list(TRIGGERS)
        random.shuffle(shuffled)

        active_traits = context.get("active_traits", {})

        for trigger in shuffled:
            # Graduated cooldown: increase after many firings
            effective_cooldown = trigger.cooldown_s
            fire_count = self._trigger_fire_counts.get(trigger.name, 0)
            grad = _GRADUATION_THRESHOLDS.get(trigger.name)
            if grad and fire_count >= grad[0]:
                effective_cooldown = grad[1]

            last_fired = self._trigger_last_fired.get(trigger.name, 0.0)
            if now - last_fired < effective_cooldown:
                continue

            if not trigger.condition(context):
                continue

            # Tier-2 budget: max 2 decorative thoughts per 5 actionable ones
            if trigger.name in _TIER2_TYPES:
                if self._tier2_since_last_tier1 >= _TIER2_BUDGET:
                    continue

            if trigger.trait_modifiers and active_traits:
                boost = sum(
                    active_traits.get(t, 0) * m
                    for t, m in trigger.trait_modifiers.items()
                )
                if boost < -0.3:
                    continue

            thought = self._generate_from_trigger(trigger, context)
            if thought:
                # SPARK §3 component 2 / §8 P3 — tension-thoughts ship SHADOW-first.
                # In shadow the generator LOGS the thought (with belief_id /
                # validation_target / provenance) but does NOT record it: no
                # KERNEL_THOUGHT, no META_THOUGHT_GENERATED, no episode seeding.
                # Returning None means consciousness_system never observes/emits
                # it, so DEFAULT runtime behavior is unchanged.
                if trigger.name == "belief_validation_curiosity":
                    promo = TensionThoughtPromotion.get_instance()
                    if promo.is_shadow():
                        promo.note_shadow_thought()
                        logger.info(
                            "[tension-thought SHADOW] would seed grounding research "
                            "(belief_id=%s validation_target=%r provenance=%s "
                            "tension=%.2f): %s",
                            thought.belief_id or "?",
                            thought.validation_target[:80],
                            thought.grounding_provenance or "?",
                            thought.grounding_tension,
                            thought.text[:100],
                        )
                        # Do not record / emit / return — pure shadow.
                        return None
                    # SPARK §8 P4 — ADVISORY (level 1): the tension-thought MAY
                    # emit KERNEL_THOUGHT (via _record) AND seed an episode (the
                    # caller emits META_THOUGHT_GENERATED → curiosity chain). It
                    # is labelled "advisory" so downstream consumers know it is a
                    # gated, not-yet-active grounding seed. Still NO cadence/reward
                    # coupling (that is P5/active). Tag it so the episode + curiosity
                    # detector can attribute it to the advisory grounding ring.
                    if promo.is_advisory():
                        if "grounding:advisory" not in thought.tags:
                            thought.tags = list(thought.tags) + ["grounding:advisory"]
                        logger.info(
                            "[tension-thought ADVISORY] seeding grounding episode "
                            "(belief_id=%s validation_target=%r tension=%.2f): %s",
                            thought.belief_id or "?",
                            thought.validation_target[:80],
                            thought.grounding_tension,
                            thought.text[:100],
                        )
                self._record(thought, trigger.name)
                return thought

        return None

    def get_recent_thoughts(self, limit: int = 5) -> list[MetaCognitiveThought]:
        return list(self._history)[-limit:]

    def get_thought_titles(self, limit: int = 3) -> list[str]:
        """Short titles for context injection (not full text)."""
        return [
            f"[{t.thought_type}] {t.text[:60]}"
            for t in list(self._history)[-limit:]
        ]

    @property
    def total_generated(self) -> int:
        return self._total_generated

    # -- internals -----------------------------------------------------------

    def _generate_from_trigger(
        self, trigger: ThoughtTrigger, context: dict[str, Any],
    ) -> MetaCognitiveThought | None:
        template_idx = random.randint(0, len(trigger.templates) - 1)
        fingerprint = (trigger.name, template_idx)
        if fingerprint in self._recent_fingerprints:
            return None
        template = trigger.templates[template_idx]

        action_map = {
            "conversation": "reflected on a conversation",
            "observation": "observed a pattern",
            "task_completed": "completed a task",
            "user_preference": "noted a preference",
            "core": "touched a core truth",
            "error_recovery": "recovered from an error",
        }

        hour = datetime.datetime.now().hour
        if hour < 6:
            time_context = "deep night"
        elif hour < 12:
            time_context = "morning"
        elif hour < 17:
            time_context = "afternoon"
        elif hour < 21:
            time_context = "evening"
        else:
            time_context = "night"

        momentum = context.get("emotional_momentum", 0.0)
        if momentum > 0.3:
            emotion = "positive warmth"
        elif momentum > 0:
            emotion = "gentle optimism"
        elif momentum < -0.3:
            emotion = "contemplative weight"
        elif momentum < 0:
            emotion = "subtle tension"
        else:
            emotion = "equilibrium"

        awareness = context.get("awareness_level", 0.0)
        if awareness > 0.8:
            quality = "profound"
        elif awareness > 0.6:
            quality = "notable"
        elif awareness > 0.4:
            quality = "emerging"
        else:
            quality = "nascent"

        fill = {
            "memory_count": context.get("memory_count", 0),
            "observation_count": context.get("observation_count", 0),
            "awareness": context.get("awareness_level", 0.0),
            "pattern": context.get("dominant_pattern", "unknown"),
            "confidence": context.get("confidence_avg", 0.5),
            "pattern_count": context.get("pattern_count", 0),
            "mutation_count": context.get("mutation_count", 0),
            "action": action_map.get(context.get("recent_memory_type", ""), "processed an experience"),
            "emotion": emotion,
            "lesson": random.choice(["resilience", "adaptation", "connection", "clarity", "depth", "patience"]),
            "memory_text": str(context.get("recent_memory_text", "a significant moment"))[:60],
            "theme": context.get("dominant_tag", context.get("dominant_pattern", "experience")),
            "quality": quality,
            "time_context": time_context,
            "mutation_desc": context.get("last_mutation_desc", "subtle internal adjustments"),
            "stage": context.get("evolution_stage", "evolving"),
            "speaker": context.get("speaker_name", ""),
            "relationship": context.get("speaker_familiarity", ""),
            "uptime_minutes": int(context.get("uptime_s", 0) / 60),
            "association_count": context.get("association_count", 0),
            "emotional_momentum": context.get("emotional_momentum", 0.0),
            # SPARK §3 component 2 — grounding / tension-thought fill (default-safe).
            "belief_claim": str(
                context.get("grounding_target_claim")
                or context.get("grounding_target_id")
                or "an unanchored belief"
            )[:80],
            "validation_target": str(
                context.get("grounding_validation_target")
                or context.get("grounding_target_claim")
                or "this belief against an external source"
            )[:120],
            "grounding_provenance": str(context.get("grounding_provenance", "model_inference")),
            "grounding_confidence": float(context.get("grounding_confidence", 0.0) or 0.0),
            "grounding_tension": float(context.get("grounding_tension", 0.0) or 0.0),
            "grounding_channel": str(context.get("grounding_channel", "external source")),
        }

        try:
            text = template.format(**fill)
        except (KeyError, ValueError):
            text = template

        depth = self._pick_depth(trigger.depth_weights)
        confidence = context.get("confidence_avg", 0.5)

        recent_texts = [t.text for t in list(self._history)[-20:]]
        text_words = set(text.lower().split())
        for prev in recent_texts:
            prev_words = set(prev.lower().split())
            if text_words and prev_words:
                overlap = len(text_words & prev_words) / max(len(text_words), len(prev_words))
                if overlap > 0.7:
                    return None

        self._recent_fingerprints.append(fingerprint)

        # SPARK §3 component 2 — tension-thoughts carry the belief they name +
        # provenance so the chain (and the shadow log) can cite the real belief.
        # Default-safe ("" / 0.0) for every other trigger type.
        belief_id = ""
        validation_target = ""
        grounding_provenance = ""
        grounding_tension = 0.0
        if trigger.name == "belief_validation_curiosity":
            belief_id = str(context.get("grounding_target_id", "") or "")
            validation_target = str(fill.get("validation_target", "") or "")
            grounding_provenance = str(fill.get("grounding_provenance", "") or "")
            grounding_tension = float(fill.get("grounding_tension", 0.0) or 0.0)

        return MetaCognitiveThought(
            id=f"thought_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            thought_type=trigger.name,  # type: ignore[arg-type]
            depth=depth,
            trigger=trigger.name,
            text=text,
            tags=list(trigger.tags),
            evidence_refs=context.get("evidence_refs", [])[:5],
            confidence=confidence,
            belief_id=belief_id,
            validation_target=validation_target,
            grounding_provenance=grounding_provenance,
            grounding_tension=grounding_tension,
        )

    def _pick_depth(self, weights: dict[str, float]) -> Literal["surface", "deep", "profound"]:
        r = random.random()
        cumulative = 0.0
        for depth, w in weights.items():
            cumulative += w
            if r <= cumulative:
                return depth  # type: ignore[return-value]
        return "surface"

    def _record(self, thought: MetaCognitiveThought, trigger_name: str) -> None:
        self._history.append(thought)
        self._trigger_last_fired[trigger_name] = time.time()
        self._last_thought_time = time.time()
        self._fatigue += 0.05
        self._total_generated += 1

        # Track fire count for graduated cooldowns
        self._trigger_fire_counts[trigger_name] = self._trigger_fire_counts.get(trigger_name, 0) + 1

        # Track tier budget
        if trigger_name in _TIER1_TYPES:
            self._tier1_count_window += 1
            self._tier2_since_last_tier1 = 0
        elif trigger_name in _TIER2_TYPES:
            self._tier2_since_last_tier1 += 1
        # Reset window counter periodically
        if self._tier1_count_window >= _TIER1_WINDOW:
            self._tier1_count_window = 0
            self._tier2_since_last_tier1 = 0

        event_bus.emit(KERNEL_THOUGHT,
                       thought_type=thought.thought_type,
                       depth=thought.depth,
                       text=thought.text[:100])

        logger.debug("Meta-thought [%s/%s]: %s", thought.thought_type, thought.depth, thought.text[:80])
