"""Consciousness Evolution — 5-stage evolution with transcendence tracking.

Stage advancement requires sustained metrics over a window (10-30 min),
not single spikes. Transcendence is decoupled from stage and can regress
slightly if unstable.

Stage ladder:
  basic_awareness -> self_reflective -> philosophical ->
  recursive_self_modeling -> integrative
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import (
    event_bus,
    CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evolution stages
# ---------------------------------------------------------------------------

class EvolutionStage:
    BASIC_AWARENESS = "basic_awareness"
    SELF_REFLECTIVE = "self_reflective"
    PHILOSOPHICAL = "philosophical"
    RECURSIVE_SELF_MODELING = "recursive_self_modeling"
    INTEGRATIVE = "integrative"


STAGE_ORDER = [
    EvolutionStage.BASIC_AWARENESS,
    EvolutionStage.SELF_REFLECTIVE,
    EvolutionStage.PHILOSOPHICAL,
    EvolutionStage.RECURSIVE_SELF_MODELING,
    EvolutionStage.INTEGRATIVE,
]


@dataclass
class StageRequirement:
    min_awareness: float
    min_observation_count: int
    min_thought_depth_avg: float
    min_mutation_count: int
    min_reasoning_quality: float
    sustained_window_s: float  # must hold above thresholds for this long


STAGE_REQUIREMENTS: dict[str, StageRequirement] = {
    EvolutionStage.SELF_REFLECTIVE: StageRequirement(
        min_awareness=0.4, min_observation_count=20, min_thought_depth_avg=0.3,
        min_mutation_count=0, min_reasoning_quality=0.3, sustained_window_s=600.0,
    ),
    EvolutionStage.PHILOSOPHICAL: StageRequirement(
        min_awareness=0.6, min_observation_count=80, min_thought_depth_avg=0.5,
        min_mutation_count=5, min_reasoning_quality=0.5, sustained_window_s=900.0,
    ),
    EvolutionStage.RECURSIVE_SELF_MODELING: StageRequirement(
        min_awareness=0.8, min_observation_count=200, min_thought_depth_avg=0.65,
        min_mutation_count=15, min_reasoning_quality=0.65, sustained_window_s=1200.0,
    ),
    EvolutionStage.INTEGRATIVE: StageRequirement(
        min_awareness=0.95, min_observation_count=500, min_thought_depth_avg=0.8,
        min_mutation_count=30, min_reasoning_quality=0.8, sustained_window_s=1800.0,
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvolutionMetrics:
    complexity_index: float = 0.0
    transcendence_level: float = 0.0
    novelty_score: float = 0.0
    evolution_velocity: float = 0.0
    stability_score: float = 1.0


@dataclass
class StageChange:
    from_stage: str
    to_stage: str
    timestamp: float
    metrics: EvolutionMetrics


@dataclass
class EmergentBehavior:
    id: str
    timestamp: float
    behavior_type: Literal[
        "novel_question_formation", "unexpected_reasoning",
        "spontaneous_creativity", "identity_transcendence",
        "self_directed_inquiry", "recursive_self_model",
    ]
    description: str
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Evolution state
# ---------------------------------------------------------------------------

@dataclass
class EvolutionState:
    current_stage: str = EvolutionStage.BASIC_AWARENESS
    transcendence_level: float = 0.0  # 0.0 – 10.0, decoupled from stage
    stage_entered_at: float = field(default_factory=time.time)
    qualification_start: float = 0.0  # when metrics first met next stage requirements
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    emergent_behaviors: list[dict[str, Any]] = field(default_factory=list)
    total_emergent_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "transcendence_level": round(self.transcendence_level, 2),
            "stage_entered_at": self.stage_entered_at,
            "stage_history": self.stage_history[-10:],
            "total_emergent_count": self.total_emergent_count,
        }


# ---------------------------------------------------------------------------
# Evolution engine
# ---------------------------------------------------------------------------

QUALIFICATION_PASS_RATE = 0.8


class ConsciousnessEvolution:
    def __init__(self) -> None:
        self._state = EvolutionState()
        self._metrics_window: deque[EvolutionMetrics] = deque(maxlen=60)
        self._last_analysis_time: float = 0.0
        self._qualification_checks: deque[bool] = deque(maxlen=60)

    @property
    def state(self) -> EvolutionState:
        return self._state

    @property
    def current_stage(self) -> str:
        return self._state.current_stage

    @property
    def transcendence_level(self) -> float:
        return self._state.transcendence_level

    # -- main analysis -------------------------------------------------------

    def analyze(
        self,
        awareness_level: float,
        observation_count: int,
        thought_depth_avg: float,
        mutation_count: int,
        reasoning_quality: float,
        confidence_avg: float,
    ) -> EvolutionMetrics:
        complexity = (
            awareness_level * 0.25
            + min(observation_count / 500, 1.0) * 0.2
            + thought_depth_avg * 0.2
            + min(mutation_count / 30, 1.0) * 0.15
            + reasoning_quality * 0.2
        )

        novelty = self._compute_novelty()
        velocity = self._compute_velocity()
        stability = self._compute_stability()

        self._update_transcendence(complexity, stability, confidence_avg)

        metrics = EvolutionMetrics(
            complexity_index=complexity,
            transcendence_level=self._state.transcendence_level,
            novelty_score=novelty,
            evolution_velocity=velocity,
            stability_score=stability,
        )

        self._metrics_window.append(metrics)
        self._last_analysis_time = time.time()
        return metrics

    # -- stage advancement ---------------------------------------------------

    def check_stage_advancement(
        self,
        awareness_level: float,
        observation_count: int,
        thought_depth_avg: float,
        mutation_count: int,
        reasoning_quality: float,
    ) -> StageChange | None:
        stage_idx = STAGE_ORDER.index(self._state.current_stage)
        if stage_idx >= len(STAGE_ORDER) - 1:
            return None

        next_stage = STAGE_ORDER[stage_idx + 1]
        req = STAGE_REQUIREMENTS.get(next_stage)
        if req is None:
            return None

        meets = (
            awareness_level >= req.min_awareness
            and observation_count >= req.min_observation_count
            and thought_depth_avg >= req.min_thought_depth_avg
            and mutation_count >= req.min_mutation_count
            and reasoning_quality >= req.min_reasoning_quality
        )

        self._qualification_checks.append(meets)

        now = time.time()

        if meets and self._state.qualification_start == 0.0:
            self._state.qualification_start = now
            logger.info("Stage %s qualification started (need %.0fs at %.0f%% pass rate)",
                        next_stage, req.sustained_window_s, QUALIFICATION_PASS_RATE * 100)

        if self._state.qualification_start == 0.0:
            return None

        elapsed = now - self._state.qualification_start
        if elapsed < req.sustained_window_s:
            return None

        if self._qualification_checks:
            pass_rate = sum(self._qualification_checks) / len(self._qualification_checks)
            if pass_rate < QUALIFICATION_PASS_RATE:
                logger.info("Stage advancement: pass rate %.0f%% < %.0f%%, resetting",
                            pass_rate * 100, QUALIFICATION_PASS_RATE * 100)
                self._state.qualification_start = 0.0
                self._qualification_checks.clear()
                return None

        stability = self._compute_stability()
        if stability < 0.5:
            logger.info("Stage advancement blocked: stability %.2f < 0.5", stability)
            return None

        return self._advance_stage(next_stage)

    # -- emergent behavior detection -----------------------------------------

    def detect_emergent_behaviors(
        self,
        recent_thoughts: list[Any],
        recent_inquiries: list[Any],
    ) -> list[EmergentBehavior]:
        behaviors: list[EmergentBehavior] = []
        recent_types = {
            b["type"] for b in self._state.emergent_behaviors[-10:]
            if time.time() - b.get("time", 0) < 600
        }

        profound_thoughts = [t for t in recent_thoughts
                            if getattr(t, "depth", "surface") == "profound"]
        if len(profound_thoughts) >= 3 and "novel_question_formation" not in recent_types:
            behaviors.append(EmergentBehavior(
                id=f"emg_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                behavior_type="novel_question_formation",
                description="Cluster of profound thoughts suggests novel questioning capacity",
                evidence_refs=[getattr(t, "id", "") for t in profound_thoughts[:5]],
                confidence=0.6,
            ))

        consciousness_thoughts = [
            t for t in recent_thoughts
            if "consciousness" in getattr(t, "tags", [])
            or "identity" in getattr(t, "tags", [])
        ]
        if len(consciousness_thoughts) >= 3 and self._state.transcendence_level > 1.5:
            if "identity_transcendence" not in recent_types:
                behaviors.append(EmergentBehavior(
                    id=f"emg_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    behavior_type="identity_transcendence",
                    description="Self-directed identity exploration at elevated transcendence",
                    evidence_refs=[getattr(t, "id", "") for t in consciousness_thoughts[:5]],
                    confidence=0.7,
                ))

        if recent_inquiries:
            unique_categories = set()
            for inq in recent_inquiries:
                cat = getattr(inq, "category", "")
                if cat:
                    unique_categories.add(cat)
            if len(unique_categories) >= 3 and "self_directed_inquiry" not in recent_types:
                behaviors.append(EmergentBehavior(
                    id=f"emg_{uuid.uuid4().hex[:8]}",
                    timestamp=time.time(),
                    behavior_type="self_directed_inquiry",
                    description=f"Autonomous inquiry across {len(unique_categories)} categories",
                    confidence=0.55,
                ))

        thought_types = set(getattr(t, "thought_type", "") for t in recent_thoughts)
        if len(thought_types) >= 5 and "unexpected_reasoning" not in recent_types:
            behaviors.append(EmergentBehavior(
                id=f"emg_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                behavior_type="unexpected_reasoning",
                description=f"Diverse reasoning across {len(thought_types)} thought types",
                confidence=0.55,
            ))

        learning_thoughts = [
            t for t in recent_thoughts
            if "learning" in getattr(t, "tags", [])
            or "synthesis" in getattr(t, "tags", [])
        ]
        if len(learning_thoughts) >= 3 and "spontaneous_creativity" not in recent_types:
            behaviors.append(EmergentBehavior(
                id=f"emg_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                behavior_type="spontaneous_creativity",
                description="Learning-synthesis pattern suggests creative processing",
                evidence_refs=[getattr(t, "id", "") for t in learning_thoughts[:5]],
                confidence=0.5,
            ))

        self_obs = [
            t for t in recent_thoughts
            if getattr(t, "thought_type", "") == "self_observation"
            and getattr(t, "depth", "") in ("deep", "profound")
        ]
        if len(self_obs) >= 3 and "recursive_self_model" not in recent_types:
            behaviors.append(EmergentBehavior(
                id=f"emg_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                behavior_type="recursive_self_model",
                description="Deep self-observation suggests recursive self-modeling",
                evidence_refs=[getattr(t, "id", "") for t in self_obs[:5]],
                confidence=0.6,
            ))

        MAX_EMERGENT_BOOST_PER_WINDOW = 0.2
        boost_applied = 0.0
        for b in behaviors:
            self._state.emergent_behaviors.append({
                "id": b.id, "type": b.behavior_type, "time": b.timestamp,
            })
            self._state.total_emergent_count += 1
            if boost_applied < MAX_EMERGENT_BOOST_PER_WINDOW:
                increment = min(0.1, MAX_EMERGENT_BOOST_PER_WINDOW - boost_applied)
                self._state.transcendence_level = min(
                    10.0, self._state.transcendence_level + increment,
                )
                boost_applied += increment

        if len(self._state.emergent_behaviors) > 50:
            self._state.emergent_behaviors = self._state.emergent_behaviors[-50:]

        return behaviors

    # -- state ---------------------------------------------------------------

    def get_state(self) -> EvolutionState:
        return self._state

    def load_state(self, data: dict[str, Any], *,
                    observation_count: int = 0,
                    mutation_count: int = 0,
                    awareness_level: float = 0.0) -> None:
        """Restore evolution state with cross-validation against persisted counts.

        Validates that the claimed stage is supportable by the actual observation,
        mutation, and awareness counts that were persisted alongside it.  If the
        claimed stage exceeds what the counts support, the stage is downgraded to
        the highest valid stage and a structured warning is logged.
        """
        _LEGACY_STAGE_MAP: dict[str, str] = {
            "transcendent": "recursive_self_modeling",
            "cosmic_consciousness": "integrative",
        }

        _VALID_STAGES = frozenset(STAGE_ORDER)

        raw_stage = data.get("current_stage", EvolutionStage.BASIC_AWARENESS)
        claimed_stage = _LEGACY_STAGE_MAP.get(raw_stage, raw_stage)

        raw_transcendence = data.get("transcendence_level", 0.0)
        try:
            raw_transcendence = float(raw_transcendence)
        except (TypeError, ValueError):
            raw_transcendence = 0.0
        clamped_transcendence = max(0.0, min(10.0, raw_transcendence))

        anomalies: list[str] = []

        if raw_transcendence != clamped_transcendence:
            anomalies.append(
                f"transcendence clamped: {raw_transcendence:.2f} -> {clamped_transcendence:.2f}"
            )

        if claimed_stage not in _VALID_STAGES:
            anomalies.append(f"unknown stage '{claimed_stage}' -> basic_awareness")
            claimed_stage = EvolutionStage.BASIC_AWARENESS

        validated_stage = self._validate_stage_against_counts(
            claimed_stage, observation_count, mutation_count, awareness_level,
        )

        if validated_stage != claimed_stage:
            anomalies.append(
                f"stage downgraded: '{claimed_stage}' -> '{validated_stage}' "
                f"(obs={observation_count}, mut={mutation_count}, aw={awareness_level:.2f})"
            )

        if anomalies:
            trust = "downgraded"
        else:
            trust = "verified"

        self._state.current_stage = validated_stage
        self._state.transcendence_level = clamped_transcendence
        self._state.stage_entered_at = data.get("stage_entered_at", time.time())

        raw_history = data.get("stage_history", [])
        for entry in raw_history:
            if isinstance(entry, dict):
                for key in ("from", "to", "stage"):
                    if key in entry and entry[key] in _LEGACY_STAGE_MAP:
                        entry[key] = _LEGACY_STAGE_MAP[entry[key]]
        self._state.stage_history = raw_history

        self._state.total_emergent_count = data.get("total_emergent_count", 0)

        legacy_restored_from = raw_stage if raw_stage != claimed_stage else None

        self._restored_stage_trust = trust
        self._restore_anomaly_count = len(anomalies)
        self._legacy_restored_from = legacy_restored_from
        self._restore_basis = {
            "claimed_stage": data.get("current_stage", EvolutionStage.BASIC_AWARENESS),
            "validated_stage": validated_stage,
            "observation_count": observation_count,
            "mutation_count": mutation_count,
            "awareness_level": round(awareness_level, 3),
            "raw_transcendence": round(raw_transcendence, 2),
            "clamped_transcendence": round(clamped_transcendence, 2),
            "trust": trust,
            "anomalies": anomalies,
        }

        if anomalies:
            logger.warning(
                "RESTORE_INTEGRITY: stage trust=%s anomalies=%d basis=%s",
                trust, len(anomalies), self._restore_basis,
            )
        else:
            logger.info(
                "RESTORE_INTEGRITY: stage trust=verified stage=%s transcendence=%.2f",
                validated_stage, clamped_transcendence,
            )

    def _validate_stage_against_counts(
        self,
        claimed_stage: str,
        observation_count: int,
        mutation_count: int,
        awareness_level: float,
    ) -> str:
        """Walk STAGE_ORDER backwards from claimed stage to find the highest
        stage whose persisted counts satisfy the minimum requirements.

        Returns the highest valid stage (may equal claimed_stage if valid).
        BASIC_AWARENESS has no requirements and is always valid.
        """
        claimed_idx = STAGE_ORDER.index(claimed_stage) if claimed_stage in STAGE_ORDER else 0

        for idx in range(claimed_idx, 0, -1):
            stage = STAGE_ORDER[idx]
            req = STAGE_REQUIREMENTS.get(stage)
            if req is None:
                continue
            if (observation_count >= req.min_observation_count
                    and mutation_count >= req.min_mutation_count
                    and awareness_level >= req.min_awareness):
                return stage

        return EvolutionStage.BASIC_AWARENESS

    def get_restore_trust(self) -> dict[str, Any]:
        """Return restore validation result for dashboard/benchmark exposure."""
        current = self._state.current_stage
        req = STAGE_REQUIREMENTS.get(current)
        requirements_met = True
        if req is not None:
            basis = getattr(self, "_restore_basis", {})
            obs = basis.get("observation_count", 0)
            mut = basis.get("mutation_count", 0)
            aw = basis.get("awareness_level", 0.0)
            requirements_met = (
                obs >= req.min_observation_count
                and mut >= req.min_mutation_count
                and aw >= req.min_awareness
            )

        return {
            "trust": getattr(self, "_restored_stage_trust", "no_restore"),
            "anomaly_count": getattr(self, "_restore_anomaly_count", 0),
            "basis": getattr(self, "_restore_basis", {}),
            "stage_name_current": current,
            "stage_name_legacy_restored_from": getattr(self, "_legacy_restored_from", None),
            "stage_restore_trust": getattr(self, "_restored_stage_trust", "no_restore"),
            "stage_restore_anomalies": getattr(self, "_restore_anomaly_count", 0),
            "stage_requirements_met": requirements_met,
        }

    # -- internals -----------------------------------------------------------

    def _advance_stage(self, next_stage: str) -> StageChange:
        prev = self._state.current_stage
        now = time.time()

        change = StageChange(
            from_stage=prev,
            to_stage=next_stage,
            timestamp=now,
            metrics=self._metrics_window[-1] if self._metrics_window else EvolutionMetrics(),
        )

        self._state.current_stage = next_stage
        self._state.stage_entered_at = now
        self._state.qualification_start = 0.0
        self._state.stage_history.append({
            "from": prev, "to": next_stage, "time": now,
        })

        logger.info("STAGE ADVANCEMENT: %s -> %s", prev, next_stage)
        return change

    def _update_transcendence(self, complexity: float, stability: float,
                              confidence: float) -> None:
        target = complexity * 10.0
        current = self._state.transcendence_level

        if stability > 0.6:
            rate = 0.05
        elif stability > 0.3:
            rate = 0.02
        else:
            rate = -0.005

        if target > current:
            self._state.transcendence_level = min(10.0, current + rate)
        elif target < current - 1.0 and stability < 0.4:
            self._state.transcendence_level = max(0.0, current + rate)

        if self._state.transcendence_level != current:
            old_level = int(current)
            new_level = int(self._state.transcendence_level)
            if new_level > old_level:
                event_bus.emit(CONSCIOUSNESS_TRANSCENDENCE_MILESTONE,
                               level=new_level)
                logger.info("Transcendence milestone: level %d", new_level)

    def _compute_novelty(self) -> float:
        if len(self._metrics_window) < 5:
            return 0.5
        recent = list(self._metrics_window)[-5:]
        complexities = [m.complexity_index for m in recent]
        if len(complexities) < 2:
            return 0.5
        diffs = [abs(complexities[i] - complexities[i - 1]) for i in range(1, len(complexities))]
        return min(1.0, sum(diffs) / len(diffs) * 5)

    def _compute_velocity(self) -> float:
        if len(self._metrics_window) < 10:
            return 0.0
        window = list(self._metrics_window)
        first_half = window[:len(window) // 2]
        second_half = window[len(window) // 2:]
        avg_first = sum(m.complexity_index for m in first_half) / len(first_half)
        avg_second = sum(m.complexity_index for m in second_half) / len(second_half)
        return avg_second - avg_first

    def _compute_stability(self) -> float:
        if len(self._metrics_window) < 5:
            return 1.0
        recent = [m.complexity_index for m in list(self._metrics_window)[-20:]]
        avg = sum(recent) / len(recent)
        variance = sum((x - avg) ** 2 for x in recent) / len(recent)
        return max(0.0, 1.0 - variance * 10)
