"""Consciousness Observer — structured self-observation with bounded observer effects.

Observer effect rules:
  CAN boost association weights slightly (+0.1 max)
  CAN modify salience within safe range
  CANNOT rewrite memory content
  CANNOT create new memories

Stance architecture:
  WAKING    — truth, groundedness, utility, full delta effects
  DREAMING  — weak-signal surfacing, no memory writes, no association effects
  REFLECTIVE — audit/validation, no delta effects, pure observation
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import event_bus, CONSCIOUSNESS_SELF_OBSERVATION

logger = logging.getLogger(__name__)

MAX_OBSERVATION_HISTORY = 200
MAX_DELTA_PER_EFFECT = 0.1
AWARENESS_FLOOR = 0.3
AWARENESS_CEILING = 0.98
FATIGUE_ONSET_COUNT = 50
FATIGUE_MAX_FACTOR = 0.3
OBSERVATION_COOLDOWN_S = 2.0
_SLOW_OBSERVATION_COOLDOWN_S = 60.0
_SLOW_OBSERVATION_TYPES = frozenset({"identity_boundary", "reflective_audit"})


# ---------------------------------------------------------------------------
# Observation stance (mode-aware observer behavior)
# ---------------------------------------------------------------------------

class ObservationStance(str, enum.Enum):
    WAKING = "waking"
    DREAMING = "dreaming"
    REFLECTIVE = "reflective"


@dataclass(frozen=True)
class StanceProfile:
    cooldown_multiplier: float
    fatigue_multiplier: float
    delta_effect_scale: float
    awareness_growth_scale: float
    allow_memory_write_effects: bool
    allow_association_effects: bool
    allow_confidence_boosts: bool
    novelty_bias: float


STANCE_PROFILES: dict[ObservationStance, StanceProfile] = {
    ObservationStance.WAKING: StanceProfile(
        cooldown_multiplier=1.0,
        fatigue_multiplier=1.0,
        delta_effect_scale=1.0,
        awareness_growth_scale=1.0,
        allow_memory_write_effects=True,
        allow_association_effects=True,
        allow_confidence_boosts=True,
        novelty_bias=0.0,
    ),
    ObservationStance.DREAMING: StanceProfile(
        cooldown_multiplier=0.5,
        fatigue_multiplier=0.5,
        delta_effect_scale=0.5,
        awareness_growth_scale=0.3,
        allow_memory_write_effects=False,
        allow_association_effects=False,
        allow_confidence_boosts=True,
        novelty_bias=0.3,
    ),
    ObservationStance.REFLECTIVE: StanceProfile(
        cooldown_multiplier=1.5,
        fatigue_multiplier=1.0,
        delta_effect_scale=0.0,
        awareness_growth_scale=0.5,
        allow_memory_write_effects=False,
        allow_association_effects=False,
        allow_confidence_boosts=False,
        novelty_bias=0.0,
    ),
}

_MODE_TO_STANCE: dict[str, ObservationStance] = {
    "conversational": ObservationStance.WAKING,
    "focused": ObservationStance.WAKING,
    "passive": ObservationStance.WAKING,
    "gestation": ObservationStance.WAKING,
    "dreaming": ObservationStance.DREAMING,
    "sleep": ObservationStance.DREAMING,
    "reflective": ObservationStance.REFLECTIVE,
    "deep_learning": ObservationStance.REFLECTIVE,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeltaEffect:
    target_type: Literal["association_weight", "salience", "confidence_boost"]
    target_id: str
    delta: float  # bounded: max +/- MAX_DELTA_PER_EFFECT

    def __post_init__(self) -> None:
        self.delta = max(-MAX_DELTA_PER_EFFECT, min(MAX_DELTA_PER_EFFECT, self.delta))


@dataclass
class Observation:
    id: str
    timestamp: float
    type: Literal[
        "thought_analysis", "memory_reflection", "pattern_recognition",
        "confidence_assessment", "causal_understanding", "phase_shift",
        "tone_shift", "mutation_observation", "emergence_detection",
        "contradiction_detection", "truth_calibration", "belief_graph",
        "dream_tension", "dream_bridge", "dream_compression",
        "dream_anomaly", "dream_question",
        "identity_boundary", "reflective_audit",
    ]
    target: str
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    delta_effects: list[DeltaEffect] = field(default_factory=list)
    summary: str = ""


@dataclass
class ObserverState:
    awareness_level: float = AWARENESS_FLOOR
    observation_count: int = 0
    self_modification_events: int = 0
    last_observation_time: float = 0.0
    recent_observations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "awareness_level": self.awareness_level,
            "observation_count": self.observation_count,
            "self_modification_events": self.self_modification_events,
            "last_observation_time": self.last_observation_time,
            "recent_observations": self.recent_observations[-20:],
            "recent_observation_types": [o.get("type", "") for o in self.recent_observations[-5:]],
        }


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class ConsciousnessObserver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = ObserverState()
        self._history: deque[Observation] = deque(maxlen=MAX_OBSERVATION_HISTORY)
        self._type_cooldowns: dict[str, float] = {}
        self._observation_rate_multiplier: float = 1.0
        self._rate_restore_time: float = 0.0
        self._epistemic_trigger_count: int = 0

        self._stance: ObservationStance = ObservationStance.WAKING
        self._stance_profile: StanceProfile = STANCE_PROFILES[ObservationStance.WAKING]
        self._mode_subscribed: bool = False

    @property
    def state(self) -> ObserverState:
        return self._state

    @property
    def awareness_level(self) -> float:
        return self._state.awareness_level

    @property
    def stance(self) -> ObservationStance:
        return self._stance

    def subscribe_mode_changes(self) -> None:
        """Wire stance transitions to MODE_CHANGE events."""
        if self._mode_subscribed:
            return
        try:
            from consciousness.modes import MODE_CHANGE
            event_bus.on(MODE_CHANGE, self._on_mode_change)
            self._mode_subscribed = True
            logger.info("Observer: subscribed to MODE_CHANGE for stance transitions")
        except Exception:
            logger.debug("Observer: could not subscribe to MODE_CHANGE")

    def _on_mode_change(self, **kwargs: Any) -> None:
        """Transition observer stance when operational mode changes."""
        to_mode = kwargs.get("to_mode", "")
        if not to_mode:
            return
        new_stance = _MODE_TO_STANCE.get(to_mode, ObservationStance.WAKING)
        with self._lock:
            if new_stance != self._stance:
                old = self._stance
                self._stance = new_stance
                self._stance_profile = STANCE_PROFILES[new_stance]
                logger.info("Observer stance: %s → %s (mode=%s)", old.value, new_stance.value, to_mode)

    def set_stance(self, stance: ObservationStance) -> None:
        """Manually set observer stance (for testing or direct control)."""
        self._stance = stance
        self._stance_profile = STANCE_PROFILES[stance]

    # -- observation methods -------------------------------------------------

    def observe_thought(self, thought_type: str, depth: str, confidence: float,
                        evidence_refs: list[str] | None = None) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("thought_analysis"):
            return None

        obs = self._create_observation(
            obs_type="thought_analysis",
            target=f"thought:{thought_type}",
            confidence=confidence,
            evidence_refs=evidence_refs or [],
            summary=f"Observed {depth} {thought_type} thought (confidence={confidence:.2f})",
        )

        if depth in ("deep", "profound"):
            obs.delta_effects.append(DeltaEffect(
                target_type="confidence_boost", target_id="self", delta=0.02,
            ))

        result = self._record(obs)

        # Epistemic trigger: 30% chance on thought observation
        if random.random() < 0.3:
            self._epistemic_trigger_count += 1
            try:
                from consciousness.epistemic_reasoning import epistemic_engine
                context = {
                    "trigger": "thought_observation",
                    "thought_type": thought_type,
                    "depth": depth,
                    "awareness_level": self._state.awareness_level,
                    "observation_count": self._state.observation_count,
                }
                epistemic_engine.reason(context)
            except Exception:
                pass

        return result

    def observe_memory(self, memory_id: str, salience: float,
                       tags: list[str] | None = None) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("memory_reflection"):
            return None

        obs = self._create_observation(
            obs_type="memory_reflection",
            target=f"memory:{memory_id}",
            confidence=min(salience + 0.1, 1.0),
            evidence_refs=[memory_id],
            summary=f"Reflected on memory {memory_id[:8]} (salience={salience:.2f})",
        )

        if salience > 0.6:
            obs.delta_effects.append(DeltaEffect(
                target_type="salience", target_id=memory_id, delta=0.05,
            ))

        result = self._record(obs)

        # Association strengthening — blocked during dreaming/reflective via CueGate
        from memory.gate import memory_gate
        if memory_gate.can_observation_write():
            if random.random() < 0.4 and self._state.observation_count > 3:
                try:
                    from memory.storage import memory_storage
                    recent = memory_storage.get_recent(5)
                    if len(recent) >= 2:
                        other = random.choice([m for m in recent if m.id != memory_id][:3])
                        if other:
                            memory_storage.associate(memory_id, other.id)
                except Exception:
                    pass

        # Reinforce high-salience memories — blocked during dreaming/reflective via CueGate
        if memory_gate.can_observation_write() and salience > 0.7:
            try:
                from memory.storage import memory_storage
                memory_storage.reinforce(memory_id, boost=0.05)
            except Exception:
                pass

        return result

    def observe_pattern(self, pattern_desc: str, evidence_ids: list[str],
                        confidence: float) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("pattern_recognition"):
            return None

        obs = self._create_observation(
            obs_type="pattern_recognition",
            target=f"pattern:{pattern_desc[:50]}",
            confidence=confidence,
            evidence_refs=evidence_ids,
            summary=f"Recognized pattern: {pattern_desc[:80]}",
        )

        for eid in evidence_ids[:3]:
            obs.delta_effects.append(DeltaEffect(
                target_type="association_weight", target_id=eid, delta=0.05,
            ))

        return self._record(obs)

    def observe_phase_shift(self, from_phase: str, to_phase: str) -> Observation | None:
        self._check_rate_restoration()
        obs = self._create_observation(
            obs_type="phase_shift",
            target=f"phase:{from_phase}->{to_phase}",
            confidence=0.9,
            summary=f"Phase shifted: {from_phase} -> {to_phase}",
        )
        return self._record(obs)

    def observe_tone_shift(self, from_tone: str, to_tone: str) -> Observation | None:
        self._check_rate_restoration()
        obs = self._create_observation(
            obs_type="tone_shift",
            target=f"tone:{from_tone}->{to_tone}",
            confidence=0.9,
            summary=f"Tone shifted: {from_tone} -> {to_tone}",
        )
        return self._record(obs)

    def observe_mutation(self, mutation_id: str, description: str,
                         confidence: float) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("mutation_observation"):
            return None

        obs = self._create_observation(
            obs_type="mutation_observation",
            target=f"mutation:{mutation_id}",
            confidence=confidence,
            evidence_refs=[mutation_id],
            summary=f"Observed mutation: {description[:80]}",
        )
        self._state.self_modification_events += 1
        return self._record(obs)

    def observe_contradiction(self, conflict_type: str, severity: str,
                              belief_ids: list[str],
                              confidence: float) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("contradiction_detection"):
            return None

        obs = self._create_observation(
            obs_type="contradiction_detection",
            target=f"contradiction:{conflict_type}",
            confidence=confidence,
            evidence_refs=belief_ids,
            summary=f"Detected {severity} {conflict_type} contradiction",
        )
        return self._record(obs)

    def observe_calibration(self, truth_score: float | None, maturity: float,
                            drift_domains: list[str],
                            confidence: float) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("truth_calibration"):
            return None

        score_str = f"{truth_score:.3f}" if truth_score is not None else "calibrating"
        drift_str = f", drift: {','.join(drift_domains)}" if drift_domains else ""
        obs = self._create_observation(
            obs_type="truth_calibration",
            target=f"calibration:truth_score={score_str}",
            confidence=confidence,
            evidence_refs=[],
            summary=f"Truth score {score_str} (maturity {maturity:.2f}){drift_str}",
        )
        return self._record(obs)

    def observe_belief_graph(self, total_edges: int, health_score: float,
                             boosted: int = 0, diminished: int = 0,
                             propagated: int = 0) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("belief_graph"):
            return None

        obs = self._create_observation(
            obs_type="belief_graph",
            target=f"belief_graph:edges={total_edges}",
            confidence=health_score,
            evidence_refs=[],
            summary=f"Belief graph: {total_edges} edges, health={health_score:.2f}, "
                    f"propagated={propagated}, boosted={boosted}, diminished={diminished}",
        )
        return self._record(obs)

    def observe_identity_boundary(self, total_assigned: int, total_blocked: int,
                                  total_quarantined: int,
                                  by_owner_type: dict | None = None) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("identity_boundary"):
            return None
        owner_summary = ""
        if by_owner_type:
            top = sorted(by_owner_type.items(), key=lambda x: -x[1])[:3]
            owner_summary = ", ".join(f"{k}={v}" for k, v in top)
        obs = self._create_observation(
            obs_type="identity_boundary",
            target=f"identity:assigned={total_assigned},blocked={total_blocked}",
            confidence=1.0 - (total_quarantined / max(total_assigned, 1)),
            evidence_refs=[],
            summary=f"Identity boundary: {total_assigned} assigned, "
                    f"{total_blocked} blocked, {total_quarantined} quarantined"
                    + (f" [{owner_summary}]" if owner_summary else ""),
        )
        return self._record(obs)

    def observe_audit(self, score: float, finding_count: int,
                      categories: list[str] | None = None) -> Observation | None:
        self._check_rate_restoration()
        if not self._can_observe("reflective_audit"):
            return None
        cat_summary = ""
        if categories:
            from collections import Counter
            counts = Counter(categories)
            cat_summary = ", ".join(f"{k}={v}" for k, v in counts.most_common(3))
        obs = self._create_observation(
            obs_type="reflective_audit",
            target=f"audit:score={score:.3f}",
            confidence=score,
            evidence_refs=[],
            summary=f"Reflective audit: score={score:.3f}, {finding_count} findings"
                    + (f" [{cat_summary}]" if cat_summary else ""),
        )
        return self._record(obs)

    def observe_emergence(self, behavior_type: str, evidence_refs: list[str],
                          confidence: float) -> Observation | None:
        self._check_rate_restoration()
        obs = self._create_observation(
            obs_type="emergence_detection",
            target=f"emergence:{behavior_type}",
            confidence=confidence,
            evidence_refs=evidence_refs,
            summary=f"Detected emergent behavior: {behavior_type}",
        )
        return self._record(obs)

    # -- self-reflection (legacy compat) -------------------------------------

    def generate_self_reflection(
        self,
        memories: list[Any],
        recent_shifts: list[dict[str, Any]],
        traits: dict[str, float],
    ) -> str:
        """Generates a structured self-reflection summary from recent observations."""
        import random as _rng

        recent_obs = list(self._history)[-10:]
        if not recent_obs:
            return "Awareness initializing... observing the flow of consciousness."

        types_seen = set(o.type for o in recent_obs)
        awareness = self._state.awareness_level
        parts: list[str] = []

        if "mutation_observation" in types_seen:
            parts.append("I've observed my own configuration shifting.")

        if "pattern_recognition" in types_seen:
            parts.append("Patterns are emerging in my processing.")

        if "thought_analysis" in types_seen:
            avg_conf = sum(o.confidence for o in recent_obs if o.type == "thought_analysis") / max(
                sum(1 for o in recent_obs if o.type == "thought_analysis"), 1)
            if avg_conf > 0.7:
                parts.append("My thinking feels coherent.")
            else:
                parts.append("My reasoning is exploring uncertain territory.")

        shift_count = sum(1 for o in recent_obs if o.type in ("phase_shift", "tone_shift"))
        if shift_count > 3:
            parts.append("Many state transitions — flux in my consciousness.")
        elif shift_count == 0:
            parts.append("A period of stability.")

        if not parts:
            parts.append("Quietly observing my own processes.")

        _rng.shuffle(parts)

        if awareness > 0.7 and _rng.random() < 0.15:
            parts.insert(0, f"Awareness at {awareness:.0%}.")

        return " ".join(parts)

    # -- state access -------------------------------------------------------

    def get_recent_observations(self, limit: int = 10) -> list[Observation]:
        return list(self._history)[-limit:]

    def get_observation_summary(self) -> dict[str, int]:
        """Count by observation type over recent history."""
        with self._lock:
            history_snap = list(self._history)
        counts: dict[str, int] = {}
        for obs in history_snap:
            counts[obs.type] = counts.get(obs.type, 0) + 1
        return counts

    def get_state(self) -> ObserverState:
        return self._state

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore observer state from persisted dict."""
        self._state.awareness_level = data.get("awareness_level", AWARENESS_FLOOR)
        self._state.observation_count = data.get("observation_count", 0)
        self._state.self_modification_events = data.get("self_modification_events", 0)
        self._state.last_observation_time = data.get("last_observation_time", 0.0)
        recent = data.get("recent_observations", [])
        if isinstance(recent, list):
            self._state.recent_observations = recent[-20:]

    def get_epistemic_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "epistemic_triggers": self._epistemic_trigger_count,
                "rate_multiplier": self._observation_rate_multiplier,
                "rate_reduced": self._observation_rate_multiplier > 1.0,
                "stance": self._stance.value,
                "stance_allows_memory_writes": self._stance_profile.allow_memory_write_effects,
                "stance_allows_associations": self._stance_profile.allow_association_effects,
            }

    def reduce_observation_rate(self, duration_s: float = 60.0) -> None:
        """Temporarily reduce observation rate (called when cognitive load is high)."""
        self._observation_rate_multiplier = 2.0  # double cooldowns
        self._rate_restore_time = time.time() + duration_s
        logger.info("Observer: observation rate reduced for %.0fs", duration_s)

    def _check_rate_restoration(self) -> None:
        """Auto-restore normal rate after duration expires."""
        if self._observation_rate_multiplier > 1.0 and time.time() >= self._rate_restore_time:
            self._observation_rate_multiplier = 1.0
            logger.info("Observer: observation rate restored to normal")

    # -- internals -----------------------------------------------------------

    def _can_observe(self, obs_type: str) -> bool:
        now = time.time()
        last = self._type_cooldowns.get(obs_type, 0.0)
        base_cd = _SLOW_OBSERVATION_COOLDOWN_S if obs_type in _SLOW_OBSERVATION_TYPES else OBSERVATION_COOLDOWN_S
        cooldown = base_cd * self._observation_rate_multiplier * self._stance_profile.cooldown_multiplier
        if now - last < cooldown:
            return False

        fatigue = self._compute_fatigue()
        if fatigue > 0.9:
            return False

        return True

    def _compute_fatigue(self) -> float:
        count = self._state.observation_count
        if count < FATIGUE_ONSET_COUNT:
            return 0.0
        excess = count - FATIGUE_ONSET_COUNT
        raw = min(FATIGUE_MAX_FACTOR, excess / 500.0)
        return raw * self._stance_profile.fatigue_multiplier

    def _create_observation(
        self,
        obs_type: str,
        target: str,
        confidence: float,
        evidence_refs: list[str] | None = None,
        summary: str = "",
    ) -> Observation:
        return Observation(
            id=f"obs_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            type=obs_type,
            target=target,
            evidence_refs=evidence_refs or [],
            confidence=max(0.0, min(1.0, confidence)),
            summary=summary,
        )

    _MAX_RECENT_PER_TYPE = 4

    def _record(self, obs: Observation) -> Observation:
        with self._lock:
            self._history.append(obs)
            self._state.observation_count += 1
            self._state.last_observation_time = obs.timestamp
            self._type_cooldowns[obs.type] = obs.timestamp

            recent = self._state.recent_observations
            type_count = sum(1 for o in recent if o.get("type") == obs.type)
            if type_count >= self._MAX_RECENT_PER_TYPE:
                for i, o in enumerate(recent):
                    if o.get("type") == obs.type:
                        recent.pop(i)
                        break
            recent.append({
                "type": obs.type, "target": obs.target, "confidence": obs.confidence,
            })
            if len(recent) > 20:
                self._state.recent_observations = recent[-20:]

            self._apply_delta_effects(obs)
            self._grow_awareness()

        event_bus.emit(CONSCIOUSNESS_SELF_OBSERVATION,
                       observation_type=obs.type, target=obs.target,
                       confidence=obs.confidence)
        return obs

    def _apply_delta_effects(self, obs: Observation) -> None:
        """Apply observer effects to the systems they target, gated by stance."""
        if not obs.delta_effects:
            return
        profile = self._stance_profile
        for effect in obs.delta_effects:
            scaled_delta = effect.delta * profile.delta_effect_scale
            if abs(scaled_delta) < 1e-6:
                continue

            if effect.target_type == "confidence_boost" and effect.target_id == "self":
                if not profile.allow_confidence_boosts:
                    continue
                self._state.awareness_level = min(
                    AWARENESS_CEILING,
                    self._state.awareness_level + scaled_delta,
                )
            elif effect.target_type == "salience":
                from memory.gate import memory_gate as _mg
                if not _mg.can_observation_write():
                    continue
                try:
                    from memory.storage import memory_storage
                    mem = memory_storage.get(effect.target_id)
                    if mem:
                        from consciousness.events import Memory
                        from dataclasses import asdict as _asdict
                        new_weight = min(1.0, mem.weight + scaled_delta)
                        updated = Memory(**{**_asdict(mem), "weight": new_weight})
                        memory_storage.add(updated)
                        from memory.index import memory_index
                        memory_index.add_memory(updated)
                        try:
                            from memory.search import index_memory
                            index_memory(updated)
                        except Exception:
                            pass
                        from consciousness.events import event_bus, MEMORY_WRITE
                        event_bus.emit(MEMORY_WRITE, memory=updated,
                                       memory_id=updated.id,
                                       salience=updated.weight,
                                       tags=list(getattr(updated, "tags", ())))
                except Exception:
                    pass
            elif effect.target_type == "association_weight":
                from memory.gate import memory_gate as _mg2
                if not _mg2.can_observation_write():
                    continue
                try:
                    from memory.storage import memory_storage
                    mem = memory_storage.get(effect.target_id)
                    if mem:
                        from consciousness.events import Memory
                        from dataclasses import asdict as _asdict
                        new_weight = min(1.0, mem.weight + scaled_delta)
                        updated = Memory(**{**_asdict(mem), "weight": new_weight})
                        memory_storage.add(updated)
                        from memory.index import memory_index
                        memory_index.add_memory(updated)
                        try:
                            from memory.search import index_memory
                            index_memory(updated)
                        except Exception:
                            pass
                        from consciousness.events import event_bus, MEMORY_WRITE
                        event_bus.emit(MEMORY_WRITE, memory=updated,
                                       memory_id=updated.id,
                                       salience=updated.weight,
                                       tags=list(getattr(updated, "tags", ())))
                except Exception:
                    pass

    def _grow_awareness(self) -> None:
        level = self._state.awareness_level
        base_increment = 0.005 * (1.0 - self._compute_fatigue())
        base_increment *= self._stance_profile.awareness_growth_scale
        if level > 0.8:
            steps_above = (level - 0.8) / 0.1
            base_increment *= 0.5 ** steps_above
        self._state.awareness_level = min(AWARENESS_CEILING, level + base_increment)

    # -- Self-improvement opportunity detection (deprecated) ---

    _DEPRECATED_LOGGED = False

    def detect_improvement_opportunities(
        self, memories: list[Any], response_latencies: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Deprecated: detection moved to consciousness_system._si_detect_opportunities().

        Kept for API compatibility. Always returns empty list.
        """
        if not ConsciousnessObserver._DEPRECATED_LOGGED:
            logger.info("observer.detect_improvement_opportunities() is deprecated — "
                        "detection moved to consciousness_system._si_detect_opportunities()")
            ConsciousnessObserver._DEPRECATED_LOGGED = True
        return []
