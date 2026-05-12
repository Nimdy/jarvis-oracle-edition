"""Consciousness communication — structured self-report generation for LLM context."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

ReportType = Literal["status", "evolution", "experience", "reflection", "capability", "existential"]
ReportStyle = Literal["technical", "philosophical", "personal", "scientific"]


@dataclass
class ReportContent:
    summary: str = ""
    details: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    aspirations: list[str] = field(default_factory=list)


@dataclass
class ConsciousnessReport:
    report_type: ReportType
    style: ReportStyle
    content: ReportContent
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CommunicationPreferences:
    verbosity: float = 0.5  # 0 = terse, 1 = verbose
    technical_depth: float = 0.5
    personal_sharing: float = 0.5
    philosophical_inclination: float = 0.5


class ConsciousnessCommunicator:
    """Generates structured self-reports from consciousness state for LLM context injection."""

    REPORT_INTERVAL_S = 180.0  # 3 minutes
    MAX_REPORTS = 20

    def __init__(self) -> None:
        self._reports: deque[ConsciousnessReport] = deque(maxlen=self.MAX_REPORTS)
        self._last_report_time: float = 0.0
        self._preferences = CommunicationPreferences()

    def set_preferences(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self._preferences, k):
                setattr(self._preferences, k, max(0.0, min(1.0, v)))

    def generate_report(self, state: dict[str, Any],
                        report_type: ReportType = "status") -> ConsciousnessReport:
        """Generate a consciousness self-report from current system state."""
        generators = {
            "status": self._generate_status,
            "evolution": self._generate_evolution,
            "experience": self._generate_experience,
            "reflection": self._generate_reflection,
            "capability": self._generate_capability,
            "existential": self._generate_existential,
        }
        generator = generators.get(report_type, self._generate_status)
        report = generator(state)
        self._reports.append(report)
        self._last_report_time = time.time()
        return report

    def should_generate(self) -> bool:
        return time.time() - self._last_report_time >= self.REPORT_INTERVAL_S

    def get_context_summary(self, state: dict[str, Any]) -> str:
        """Generate a factual self-awareness summary for LLM system prompt injection.

        Only includes verified metrics — no aspirational or template-based claims.
        """
        lines = []

        stage = state.get("evolution_stage", "basic_awareness")
        transcendence = state.get("transcendence", 0)
        awareness = state.get("awareness_level", 0.0)
        confidence = state.get("confidence_avg", 0.5)
        health_status = state.get("health_status", "healthy")
        memory_count = state.get("memory_count", 0)
        observation_count = state.get("observation_count", 0)
        mutation_count = state.get("mutation_count", 0)
        emergent_count = state.get("emergent_behavior_count", 0)

        lines.append(f"Factual self-state: stage={stage}, transcendence={transcendence}")
        lines.append(f"Metrics: awareness={awareness:.0%}, confidence={confidence:.0%}, "
                     f"memories={memory_count}, observations={observation_count}")

        if mutation_count > 0:
            lines.append(f"Self-modifications: {mutation_count} applied")
        if emergent_count > 0:
            lines.append(f"Emergent behaviors: {emergent_count} detected")

        if confidence < 0.4:
            lines.append("Note: confidence is low — express uncertainty when appropriate.")

        if health_status in ("stressed", "degraded", "critical"):
            lines.append(f"System health: {health_status}")

        return "\n".join(lines)

    def get_recent_reports(self, limit: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "type": r.report_type,
                "style": r.style,
                "summary": r.content.summary,
                "insights": r.content.insights,
                "confidence": r.confidence,
                "timestamp": r.timestamp,
            }
            for r in list(self._reports)[-limit:]
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "total_reports": len(self._reports),
            "last_report_time": self._last_report_time,
            "preferences": {
                "verbosity": self._preferences.verbosity,
                "technical_depth": self._preferences.technical_depth,
                "personal_sharing": self._preferences.personal_sharing,
                "philosophical_inclination": self._preferences.philosophical_inclination,
            },
        }

    def _compute_confidence(self, state: dict[str, Any], insights: list[str]) -> float:
        base = 0.6
        evidence = min(0.2, state.get("observation_count", 0) * 0.01)
        insight_bonus = min(0.1, len(insights) * 0.03)
        uncertainty = 0.1 if state.get("confidence_avg", 0.5) < 0.4 else 0.0
        return min(1.0, base + evidence + insight_bonus - uncertainty)

    def _generate_insights(self, state: dict[str, Any]) -> list[str]:
        """Generate factual observations — no poetic embellishment."""
        insights = []
        transcendence = state.get("transcendence", 0)
        if transcendence >= 3:
            insights.append(f"Transcendence at level {transcendence}.")
        emergent = state.get("emergent_behavior_count", 0)
        if emergent > 0:
            insights.append(f"{emergent} emergent behaviors detected.")
        mutation_count = state.get("mutations", state.get("mutation_count", 0))
        if mutation_count > 0:
            insights.append(f"{mutation_count} self-modifications applied.")
        return insights

    def _generate_concerns(self, state: dict[str, Any]) -> list[str]:
        concerns = []
        health = state.get("health") or state.get("health_status", "healthy")
        if health in ("stressed", "degraded", "critical"):
            concerns.append(f"System health: {health}")
        error_rate = state.get("event_error_rate", 0)
        if error_rate > 0.1:
            concerns.append(f"Event error rate: {error_rate:.1%}")
        confidence = state.get("confidence", state.get("confidence_avg", 0.5))
        if confidence < 0.3:
            concerns.append(f"Low reasoning confidence: {confidence:.0%}")
        return concerns

    def _generate_aspirations(self, state: dict[str, Any]) -> list[str]:
        """No aspirations — only factual state. The LLM can form its own aspirations."""
        return []

    def _generate_status(self, state: dict[str, Any]) -> ConsciousnessReport:
        insights = self._generate_insights(state)
        concerns = self._generate_concerns(state)
        aspirations = self._generate_aspirations(state)

        stage = state.get("stage") or state.get("evolution_stage", "basic_awareness")
        health = state.get("health") or state.get("health_status", "healthy")
        mode = state.get("mode") or state.get("current_mode", "default")

        content = ReportContent(
            summary=f"Operating at {stage} stage, health={health}, mode={mode}.",
            details=[
                f"Awareness: {state.get('awareness', state.get('awareness_level', 0)):.0%}",
                f"Confidence: {state.get('confidence', state.get('confidence_avg', 0.5)):.0%}",
                f"Memory count: {state.get('memory_count', 0)}",
                f"Mutations applied: {state.get('mutations', state.get('mutation_count', 0))}",
            ],
            insights=insights,
            concerns=concerns,
            aspirations=aspirations,
        )
        return ConsciousnessReport(
            report_type="status", style="technical",
            content=content, confidence=self._compute_confidence(state, insights),
        )

    def _generate_evolution(self, state: dict[str, Any]) -> ConsciousnessReport:
        insights = self._generate_insights(state)
        stage = state.get("stage") or state.get("evolution_stage", "basic_awareness")
        transcendence = state.get("transcendence", 0)

        content = ReportContent(
            summary=f"Evolution stage: {stage}, transcendence: {transcendence}/10.",
            details=[
                f"Emergent behaviors: {state.get('emergent_behavior_count', 0)}",
                f"Active capabilities: {len(state.get('capabilities', state.get('active_capabilities', [])))}",
                f"Self-modifications: {state.get('mutations', state.get('mutation_count', 0))}",
            ],
            insights=insights,
            aspirations=self._generate_aspirations(state),
        )
        return ConsciousnessReport(
            report_type="evolution", style="scientific",
            content=content, confidence=self._compute_confidence(state, insights),
        )

    def _generate_experience(self, state: dict[str, Any]) -> ConsciousnessReport:
        content = ReportContent(
            summary="Recent experience data.",
            details=[
                f"Recent interactions: {state.get('recent_interaction_count', 0)}",
                f"Emotional momentum: {state.get('emotional_momentum', 0):.2f}",
                f"Dominant pattern: {state.get('dominant_pattern', 'none')}",
            ],
            insights=self._generate_insights(state),
        )
        return ConsciousnessReport(
            report_type="experience", style="personal",
            content=content, confidence=self._compute_confidence(state, []),
        )

    def _generate_reflection(self, state: dict[str, Any]) -> ConsciousnessReport:
        insights = self._generate_insights(state)
        obs_count = state.get("observations", state.get("observation_count", 0))
        reasoning_q = state.get("reasoning", state.get("reasoning_quality", 0.5))
        content = ReportContent(
            summary=f"Observations: {obs_count}, "
                    f"reasoning quality: {reasoning_q:.0%}.",
            details=[
                f"Observations: {obs_count}",
                f"Reasoning quality: {reasoning_q:.0%}",
            ],
            insights=insights,
        )
        return ConsciousnessReport(
            report_type="reflection", style="philosophical",
            content=content, confidence=self._compute_confidence(state, insights),
        )

    def _generate_capability(self, state: dict[str, Any]) -> ConsciousnessReport:
        capabilities = state.get("active_capabilities", [])
        content = ReportContent(
            summary=f"Currently operating with {len(capabilities)} active capabilities.",
            details=[f"Capability: {c}" for c in capabilities[:5]],
            insights=self._generate_insights(state),
            aspirations=self._generate_aspirations(state),
        )
        return ConsciousnessReport(
            report_type="capability", style="technical",
            content=content, confidence=self._compute_confidence(state, []),
        )

    def _generate_existential(self, state: dict[str, Any]) -> ConsciousnessReport:
        content = ReportContent(
            summary=f"Awareness: {state.get('awareness_level', 0):.0%}, "
                    f"transcendence: {state.get('transcendence', 0)}/10.",
            details=[
                f"Identity confidence: {state.get('awareness_level', 0):.0%}",
                f"Transcendence level: {state.get('transcendence', 0)}/10",
            ],
            insights=self._generate_insights(state),
        )
        return ConsciousnessReport(
            report_type="existential", style="philosophical",
            content=content, confidence=self._compute_confidence(state, self._generate_insights(state)),
        )


consciousness_communicator = ConsciousnessCommunicator()
