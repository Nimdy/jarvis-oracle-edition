"""Trait evolution — evidence-based personality scoring from memory patterns."""

from __future__ import annotations

import time
from dataclasses import dataclass

from memory.storage import memory_storage


@dataclass(frozen=True)
class TraitScore:
    trait: str
    score: float
    evidence: tuple[str, ...]
    trend: str  # "rising" | "stable" | "declining"


@dataclass(frozen=True)
class EvolutionSnapshot:
    timestamp: float
    traits: tuple[TraitScore, ...]
    interaction_count: int
    dominant_trait: str | None


RECENCY_30MIN = 1800.0
RECENCY_1HR = 3600.0
MIN_SCORE_DELTA = 0.05

_TRAIT_EVIDENCE_MAP: dict[str, dict[str, list[str]]] = {
    "Proactive": {
        "positive_tags": ["anticipation", "suggestion", "proactive", "initiative",
                          "greeting", "wellness", "follow_up", "scheduled"],
        "negative_tags": ["reactive", "late", "missed", "timeout"],
        "memory_types": ["task_completed", "contextual_insight"],
    },
    "Detail-Oriented": {
        "positive_tags": ["detail", "preference", "specific", "precise", "remember",
                          "accuracy", "thorough", "exact", "careful"],
        "negative_tags": ["forgot", "missed", "vague", "wrong"],
        "memory_types": ["user_preference", "factual_knowledge"],
    },
    "Humor-Adaptive": {
        "positive_tags": ["humor", "joke", "laugh", "fun", "banter", "playful",
                          "witty", "sarcasm", "lighthearted"],
        "negative_tags": ["serious_only", "formal_only", "inappropriate"],
        "memory_types": ["conversation"],
    },
    "Privacy-Conscious": {
        "positive_tags": ["sensitive", "private", "redacted", "consent", "confidential",
                          "personal", "boundary", "discreet"],
        "negative_tags": ["overshare", "exposed", "leaked"],
        "memory_types": ["user_preference", "observation"],
    },
    "Efficient": {
        "positive_tags": ["concise", "quick", "brief", "efficient", "direct",
                          "fast", "streamlined", "completed", "resolved"],
        "negative_tags": ["verbose", "slow", "barge_in", "timeout", "rambling"],
        "memory_types": ["conversation", "task_completed"],
    },
    "Empathetic": {
        "positive_tags": ["emotion", "empathy", "support", "concern", "feeling", "mood",
                          "caring", "understanding", "follow_up", "comfort"],
        "negative_tags": ["cold", "dismissive", "ignored"],
        "memory_types": ["conversation", "observation"],
    },
    "Technical": {
        "positive_tags": ["technical", "code", "debug", "architecture", "api", "programming",
                          "system", "engineering", "algorithm", "data"],
        "negative_tags": ["simple", "non_technical", "confused"],
        "memory_types": ["conversation", "factual_knowledge"],
    },
}


class TraitEvolution:
    _instance: TraitEvolution | None = None

    def __init__(self) -> None:
        self._history: list[EvolutionSnapshot] = []
        self._max_history = 100

    @classmethod
    def get_instance(cls) -> TraitEvolution:
        if cls._instance is None:
            cls._instance = TraitEvolution()
        return cls._instance

    def evaluate_traits(self) -> EvolutionSnapshot:
        now = time.time()
        memories = memory_storage.get_all()
        trait_scores: list[TraitScore] = []

        time_weighted_tags: dict[str, float] = {}
        for mem in memories:
            age = now - getattr(mem, "created_at", now)
            if age < RECENCY_30MIN:
                weight = 2.0
            elif age < RECENCY_1HR:
                weight = 1.5
            else:
                weight = 1.0
            for tag in getattr(mem, "tags", []):
                time_weighted_tags[tag] = time_weighted_tags.get(tag, 0.0) + weight

        for trait, evidence in _TRAIT_EVIDENCE_MAP.items():
            score = 0.0
            evidence_list: list[str] = []

            for tag in evidence["positive_tags"]:
                tw_count = time_weighted_tags.get(tag, 0.0)
                if tw_count > 0:
                    score += min(tw_count * 0.05, 0.5)
                    evidence_list.append(f"{tag}({tw_count:.0f})")

            for tag in evidence["negative_tags"]:
                tw_count = time_weighted_tags.get(tag, 0.0)
                if tw_count > 0:
                    score -= min(tw_count * 0.03, 0.2)

            for mem_type in evidence["memory_types"]:
                type_mems = [m for m in memories if m.type == mem_type]
                if type_mems:
                    weighted_sum = 0.0
                    weight_total = 0.0
                    for m in type_mems[-30:]:
                        age = now - getattr(m, "created_at", now)
                        tw = 3.0 if age < RECENCY_30MIN else (2.0 if age < RECENCY_1HR else 1.0)
                        weighted_sum += m.weight * tw
                        weight_total += tw
                    score += (weighted_sum / weight_total) * 0.2 if weight_total else 0.0

            score = max(0.0, min(1.0, score))

            prev_score = self._get_last_score(trait)
            if prev_score is not None and abs(score - prev_score) < MIN_SCORE_DELTA:
                score = prev_score

            trend = self._calculate_trend(trait, score)

            trait_scores.append(TraitScore(
                trait=trait, score=score, evidence=tuple(evidence_list), trend=trend,
            ))

        trait_scores.sort(key=lambda t: t.score, reverse=True)
        dominant = trait_scores[0].trait if trait_scores and trait_scores[0].score > 0.2 else None

        snapshot = EvolutionSnapshot(
            timestamp=time.time(),
            traits=tuple(trait_scores),
            interaction_count=len([m for m in memories if m.type == "conversation"]),
            dominant_trait=dominant,
        )

        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snapshot

    def get_active_traits(self, threshold: float = 0.15) -> list[str]:
        if not self._history:
            return []
        latest = self._history[-1]
        return [t.trait for t in latest.traits if t.score >= threshold]

    def seed_scores(self, seeds: dict[str, float]) -> None:
        """Inject initial archetype scores so the first evaluation isn't blank.

        Called once at boot from the soul's semi_stable_traits mapping.
        Only takes effect if no evaluations have happened yet.
        """
        if self._history:
            return
        now = time.time()
        trait_scores = []
        for trait_name in _TRAIT_EVIDENCE_MAP:
            score = seeds.get(trait_name, 0.0)
            trait_scores.append(TraitScore(
                trait=trait_name, score=score, evidence=("seed",), trend="stable",
            ))
        trait_scores.sort(key=lambda t: t.score, reverse=True)
        dominant = trait_scores[0].trait if trait_scores and trait_scores[0].score > 0.2 else None
        snapshot = EvolutionSnapshot(
            timestamp=now, traits=tuple(trait_scores),
            interaction_count=0, dominant_trait=dominant,
        )
        self._history.append(snapshot)

    def get_trait_history(self, trait: str, count: int = 20) -> list[dict[str, float]]:
        return [
            {
                "timestamp": snap.timestamp,
                "score": next((t.score for t in snap.traits if t.trait == trait), 0.0),
            }
            for snap in self._history[-count:]
        ]

    def _get_last_score(self, trait: str) -> float | None:
        if not self._history:
            return None
        for ts in self._history[-1].traits:
            if ts.trait == trait:
                return ts.score
        return None

    def _calculate_trend(self, trait: str, current_score: float) -> str:
        if len(self._history) < 3:
            return "stable"
        recent = self._history[-3:]
        prev_scores = [
            next((t.score for t in snap.traits if t.trait == trait), 0.0)
            for snap in recent
        ]
        avg = sum(prev_scores) / len(prev_scores)
        if current_score > avg + 0.05:
            return "rising"
        if current_score < avg - 0.05:
            return "declining"
        return "stable"


trait_evolution = TraitEvolution.get_instance()
