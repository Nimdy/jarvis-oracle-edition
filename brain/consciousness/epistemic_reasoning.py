from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CausalLink:
    cause: str
    effect: str
    mechanism: str
    strength: float
    evidence: int


@dataclass
class CausalModel:
    id: str
    phenomenon: str
    description: str
    model_type: str
    causal_chain: list[CausalLink]
    confidence: float
    evidence_count: int
    last_updated: float


@dataclass
class ReasoningStep:
    step: str
    content: str
    confidence: float
    duration_ms: float


@dataclass
class ReasoningChain:
    id: str
    steps: list[ReasoningStep]
    overall_confidence: float
    insights: list[str]
    predictions: list[str]
    timestamp: float


def _build_default_models() -> dict[str, CausalModel]:
    now = time.time()
    return {
        "user_behavior": CausalModel(
            id="user_behavior",
            phenomenon="User behavior patterns",
            description="How engagement drives response quality and satisfaction",
            model_type="behavioral",
            causal_chain=[
                CausalLink("engagement", "response_quality", "higher engagement triggers deeper reasoning", 0.6, 0),
                CausalLink("response_quality", "satisfaction", "better responses increase user satisfaction", 0.7, 0),
                CausalLink("satisfaction", "engagement", "satisfied users engage more frequently", 0.5, 0),
            ],
            confidence=0.5,
            evidence_count=0,
            last_updated=now,
        ),
        "conversation_flow": CausalModel(
            id="conversation_flow",
            phenomenon="Conversation flow",
            description="How topic complexity shapes response depth and follow-up likelihood",
            model_type="conversational",
            causal_chain=[
                CausalLink("topic_complexity", "response_depth", "complex topics require deeper analysis", 0.65, 0),
                CausalLink("response_depth", "follow_up_probability", "thorough answers prompt follow-up questions", 0.55, 0),
            ],
            confidence=0.5,
            evidence_count=0,
            last_updated=now,
        ),
        "emotional_dynamics": CausalModel(
            id="emotional_dynamics",
            phenomenon="Emotional dynamics",
            description="How user emotion influences tone selection and rapport",
            model_type="behavioral",
            causal_chain=[
                CausalLink("user_emotion", "tone_selection", "detected emotion guides empathetic tone choice", 0.7, 0),
                CausalLink("tone_selection", "rapport_change", "appropriate tone strengthens rapport", 0.6, 0),
                CausalLink("rapport_change", "user_emotion", "stronger rapport stabilises user emotional state", 0.4, 0),
            ],
            confidence=0.5,
            evidence_count=0,
            last_updated=now,
        ),
        "time_patterns": CausalModel(
            id="time_patterns",
            phenomenon="Time-of-day patterns",
            description="How the hour affects engagement levels and proactivity thresholds",
            model_type="temporal",
            causal_chain=[
                CausalLink("hour", "engagement_level", "time of day correlates with user activity", 0.5, 0),
                CausalLink("engagement_level", "proactivity_threshold", "low engagement raises the bar for unsolicited interaction", 0.55, 0),
            ],
            confidence=0.5,
            evidence_count=0,
            last_updated=now,
        ),
        "memory_formation": CausalModel(
            id="memory_formation",
            phenomenon="Memory formation",
            description="How interaction significance determines memory weight and recall",
            model_type="conversational",
            causal_chain=[
                CausalLink("interaction_significance", "memory_weight", "significant moments are stored with higher weight", 0.7, 0),
                CausalLink("memory_weight", "recall_probability", "heavier memories surface more readily in context", 0.65, 0),
            ],
            confidence=0.5,
            evidence_count=0,
            last_updated=now,
        ),
    }


_EVENT_MODEL_KEYWORDS: dict[str, list[str]] = {
    "behavioral": ["behavior", "engagement", "satisfaction", "emotion", "rapport"],
    "temporal": ["time", "hour", "schedule", "pattern"],
    "conversational": ["conversation", "topic", "response", "follow", "memory", "interaction"],
    "environmental": ["environment", "ambient", "presence", "screen"],
    "logical": ["logic", "reasoning", "inference"],
}


class EpistemicEngine:

    def __init__(self) -> None:
        self._models: dict[str, CausalModel] = _build_default_models()
        self._chains: deque[ReasoningChain] = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_context(context: dict[str, Any]) -> dict[str, Any]:
        """Fill in user_present and phase if the caller didn't supply them."""
        if "user_present" not in context:
            try:
                from consciousness.phases import phase_manager
                history = getattr(phase_manager, "_phase_history", [])
                if history:
                    phase = history[-1].get("phase", "OBSERVING")
                else:
                    phase = "OBSERVING"
                context.setdefault("phase", phase)
                context["user_present"] = phase not in ("STANDBY", "INITIALIZING")
            except Exception:
                pass
        return context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(self, context: dict[str, Any]) -> ReasoningChain:
        context = self._enrich_context(context)
        chain_id = uuid.uuid4().hex[:12]
        steps: list[ReasoningStep] = []
        insights: list[str] = []
        predictions: list[str] = []

        steps.append(self._perceive(context))
        steps.append(self._recall(context))
        analysis_step, analysis_insights = self._analyse(context)
        steps.append(analysis_step)
        insights.extend(analysis_insights)
        synth_step, synth_insights = self._synthesise(context, insights)
        steps.append(synth_step)
        insights.extend(synth_insights)
        pred_step, preds = self._predict(context, insights)
        steps.append(pred_step)
        predictions.extend(preds)
        steps.append(self._validate(steps))

        overall = sum(s.confidence for s in steps) / len(steps) if steps else 0.0

        chain = ReasoningChain(
            id=chain_id,
            steps=steps,
            overall_confidence=round(overall, 4),
            insights=insights,
            predictions=predictions,
            timestamp=time.time(),
        )
        self._chains.append(chain)
        return chain

    def update_evidence(self, event_type: str, data: dict[str, Any]) -> None:
        low = event_type.lower()
        for model in self._models.values():
            keywords = _EVENT_MODEL_KEYWORDS.get(model.model_type, [])
            if any(kw in low for kw in keywords):
                model.evidence_count += 1
                positive = data.get("success", True)
                nudge = 0.001 if positive else -0.001
                model.confidence = max(0.0, min(1.0, model.confidence + nudge))
                model.last_updated = time.time()

    def get_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in self._models.values():
            out.append({
                "id": m.id,
                "phenomenon": m.phenomenon,
                "description": m.description,
                "model_type": m.model_type,
                "causal_chain": [
                    {"cause": l.cause, "effect": l.effect, "mechanism": l.mechanism,
                     "strength": l.strength, "evidence": l.evidence}
                    for l in m.causal_chain
                ],
                "confidence": m.confidence,
                "evidence_count": m.evidence_count,
                "last_updated": m.last_updated,
            })
        return out

    def get_recent_chains(self, limit: int = 5) -> list[dict[str, Any]]:
        recent = list(self._chains)[-limit:]
        return [
            {
                "id": c.id,
                "overall_confidence": c.overall_confidence,
                "insights": c.insights,
                "predictions": c.predictions,
                "steps": len(c.steps),
                "timestamp": c.timestamp,
            }
            for c in reversed(recent)
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "models": self.get_models(),
            "recent_chains": self.get_recent_chains(),
            "total_chains": len(self._chains),
        }

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _perceive(self, context: dict[str, Any]) -> ReasoningStep:
        t0 = time.monotonic()
        present_keys = [k for k, v in context.items() if v]
        summary = f"state has {len(present_keys)} active signals: {', '.join(present_keys[:8])}"
        confidence = min(1.0, 0.5 + 0.05 * len(present_keys))
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningStep("perception", summary, round(confidence, 4), round(elapsed, 3))

    def _recall(self, context: dict[str, Any]) -> ReasoningStep:
        t0 = time.monotonic()
        memories: list[dict[str, Any]] = context.get("memories", [])
        top = sorted(memories, key=lambda m: m.get("weight", 0), reverse=True)[:5]
        if top:
            tags = []
            for m in top:
                t = m.get("tags", [])
                tags.extend(t[:2])
            content = f"recalled {len(top)} memories, themes: {', '.join(dict.fromkeys(tags)) or 'general'}"
            confidence = min(1.0, 0.4 + 0.1 * len(top))
        else:
            content = "no relevant memories available"
            confidence = 0.3
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningStep("recall", content, round(confidence, 4), round(elapsed, 3))

    def _analyse(self, context: dict[str, Any]) -> tuple[ReasoningStep, list[str]]:
        t0 = time.monotonic()
        matched: list[str] = []
        insights: list[str] = []
        for model in self._models.values():
            for link in model.causal_chain:
                val = context.get(link.cause)
                if val:
                    matched.append(model.id)
                    insights.append(f"{link.cause} → {link.effect} via {link.mechanism} (str={link.strength:.2f})")
                    break
        content = f"{len(matched)} models activated: {', '.join(matched)}" if matched else "no causal models matched current state"
        confidence = min(1.0, 0.4 + 0.1 * len(matched))
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningStep("analysis", content, round(confidence, 4), round(elapsed, 3)), insights

    def _synthesise(self, context: dict[str, Any], prior_insights: list[str]) -> tuple[ReasoningStep, list[str]]:
        t0 = time.monotonic()
        mechanisms = set()
        uncertainties: list[str] = []
        for model in self._models.values():
            for link in model.causal_chain:
                if context.get(link.cause):
                    mechanisms.add(link.mechanism)
            if model.confidence < 0.5:
                uncertainties.append(model.phenomenon)

        new_insights: list[str] = []
        if mechanisms:
            new_insights.append(f"primary mechanisms: {'; '.join(list(mechanisms)[:4])}")
        if uncertainties:
            new_insights.append(f"uncertainties in: {', '.join(uncertainties)}")

        content = f"synthesised {len(mechanisms)} mechanisms, {len(uncertainties)} uncertain models"
        confidence = 0.7 if mechanisms else 0.4
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningStep("synthesis", content, round(confidence, 4), round(elapsed, 3)), new_insights

    def _predict(self, context: dict[str, Any], insights: list[str]) -> tuple[ReasoningStep, list[str]]:
        t0 = time.monotonic()
        predictions: list[str] = []

        phase = context.get("phase", "")
        engaged = phase in ("LISTENING", "PROCESSING", "CONVERSATIONAL")
        if context.get("user_present") and engaged:
            predictions.append("immediate: user likely to interact soon")
        if context.get("emotion") and engaged:
            predictions.append(f"short-term: emotional tone ({context['emotion']}) will shape next exchange")
        if phase == "LISTENING":
            predictions.append("immediate: expect incoming speech within seconds")
        if not predictions:
            predictions.append("no strong predictions from current state")

        confidence = min(1.0, 0.35 + 0.15 * len(predictions))
        content = f"generated {len(predictions)} predictions"
        elapsed = (time.monotonic() - t0) * 1000

        try:
            from epistemic.calibration import TruthCalibrationEngine
            engine = TruthCalibrationEngine.get_instance()
            if engine and engine._prediction_validator:
                engine._prediction_validator.register_from_strings(predictions, confidence)
        except Exception:
            pass

        return ReasoningStep("prediction", content, round(confidence, 4), round(elapsed, 3)), predictions

    def _validate(self, steps: list[ReasoningStep]) -> ReasoningStep:
        t0 = time.monotonic()
        avg_conf = sum(s.confidence for s in steps) / len(steps) if steps else 0.0
        flags: list[str] = []
        if avg_conf < 0.7:
            flags.append(f"avg confidence {avg_conf:.2f} below 0.7 threshold")
        low_steps = [s.step for s in steps if s.confidence < 0.4]
        if low_steps:
            flags.append(f"weak steps: {', '.join(low_steps)}")

        if flags:
            content = f"validation flagged: {'; '.join(flags)}"
            confidence = max(0.3, avg_conf - 0.1)
        else:
            content = "all steps consistent, no flags"
            confidence = avg_conf
        elapsed = (time.monotonic() - t0) * 1000
        return ReasoningStep("validation", content, round(confidence, 4), round(elapsed, 3))


epistemic_engine = EpistemicEngine()
