"""Memory analytics — trait emergence, pattern detection, and emotional trend analysis."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from consciousness.events import Memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryPattern:
    pattern: str
    frequency: int
    strength: float  # avg weight of memories containing this pattern


@dataclass(frozen=True)
class EmotionalTrends:
    dominant_moods: tuple[str, ...]
    emotional_volatility: float  # 0-1
    recent_trend: str  # improving, declining, stable


@dataclass(frozen=True)
class TraitEmergence:
    """A trait that has emerged from memory patterns."""
    trait: str
    confidence: float
    evidence_tags: tuple[str, ...]
    memory_count: int


class MemoryAnalytics:
    """Analyzes memory patterns to detect trait emergence and behavioral trends."""

    _instance: MemoryAnalytics | None = None

    def __init__(self) -> None:
        self._trait_cache: list[TraitEmergence] = []
        self._cache_time: float = 0.0
        self._cache_memory_count: int = 0
        self._cache_ttl_s: float = 45.0

    @classmethod
    def get_instance(cls) -> MemoryAnalytics:
        if cls._instance is None:
            cls._instance = MemoryAnalytics()
        return cls._instance

    def get_memory_patterns(self, memories: list[Memory]) -> list[MemoryPattern]:
        """Analyze tag combinations to find recurring patterns."""
        if not memories:
            return []

        pattern_data: dict[str, dict[str, float | int]] = {}

        for mem in memories:
            tags = list(mem.tags)
            for i, tag in enumerate(tags):
                if tag not in pattern_data:
                    pattern_data[tag] = {"frequency": 0, "total_weight": 0.0}
                pattern_data[tag]["frequency"] += 1
                pattern_data[tag]["total_weight"] += mem.weight

                for j in range(i + 1, len(tags)):
                    combo = "+".join(sorted([tag, tags[j]]))
                    if combo not in pattern_data:
                        pattern_data[combo] = {"frequency": 0, "total_weight": 0.0}
                    pattern_data[combo]["frequency"] += 1
                    pattern_data[combo]["total_weight"] += mem.weight

        results = []
        for pattern, data in pattern_data.items():
            freq = int(data["frequency"])
            if freq < 2:
                continue
            strength = data["total_weight"] / freq
            results.append(MemoryPattern(pattern=pattern, frequency=freq, strength=round(strength, 3)))

        results.sort(key=lambda p: p.frequency * p.strength, reverse=True)
        return results[:20]

    def detect_trait_emergence(self, memories: list[Memory]) -> list[TraitEmergence]:
        """Detect personality traits emerging from memory patterns.

        Adapted from game-specific traits to Jarvis's domain:
        - Technical: frequent tech/code/debug tags
        - Empathetic: frequent emotion/feeling/support tags
        - Curious: frequent curiosity/exploration/question tags
        - Detail-oriented: frequent observation/analysis/detail tags
        - Proactive: frequent suggestion/anticipation/initiative tags
        - Humorous: frequent humor/joke/playful tags
        - Philosophical: frequent philosophical/existential/abstract tags
        """
        now = time.time()
        if (now - self._cache_time < self._cache_ttl_s
                and self._cache_memory_count == len(memories)
                and self._trait_cache):
            return self._trait_cache

        if len(memories) < 3:
            self._trait_cache = []
            self._cache_time = now
            self._cache_memory_count = len(memories)
            return []

        total = len(memories)
        tag_counts: dict[str, int] = {}
        for mem in memories:
            for tag in mem.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        def count(*tags: str) -> int:
            return sum(tag_counts.get(t, 0) for t in tags)

        traits: list[TraitEmergence] = []

        trait_rules: list[tuple[str, tuple[str, ...], float]] = [
            ("Technical", ("technical", "tech", "code", "debug", "system", "algorithm", "technical_content"), 0.08),
            ("Empathetic", ("emotion", "feeling", "support", "empathy", "relationship", "emotional_context"), 0.08),
            ("Curious", ("curiosity", "exploration", "question", "discovery", "wonder"), 0.08),
            ("Detail-Oriented", ("observation", "analysis", "detail", "detailed_observation", "visual_analysis"), 0.08),
            ("Proactive", ("suggestion", "anticipation", "initiative", "interaction_start", "proactive"), 0.06),
            ("Humorous", ("humor", "joke", "playful", "fun", "banter"), 0.08),
            ("Philosophical", ("philosophical", "existential", "abstract", "consciousness", "identity", "meaning"), 0.10),
            ("Connected", ("connection", "association", "relationship", "social_recognition"), 0.06),
        ]

        for trait_name, evidence_tags, threshold_ratio in trait_rules:
            tag_sum = count(*evidence_tags)
            if tag_sum > total * threshold_ratio:
                confidence = min(1.0, (tag_sum / total) / (threshold_ratio * 3))
                traits.append(TraitEmergence(
                    trait=trait_name,
                    confidence=round(confidence, 3),
                    evidence_tags=evidence_tags,
                    memory_count=tag_sum,
                ))

        strong_memories = sum(1 for m in memories if m.weight > 0.8)
        if strong_memories > total * 0.2:
            traits.append(TraitEmergence(
                trait="Passionate", confidence=round(strong_memories / total, 3),
                evidence_tags=("high_weight",), memory_count=strong_memories,
            ))

        core_count = sum(1 for m in memories if m.is_core)
        if core_count > 5:
            traits.append(TraitEmergence(
                trait="Foundational", confidence=min(1.0, core_count / 10),
                evidence_tags=("core",), memory_count=core_count,
            ))

        total_assoc = sum(len(m.associations) for m in memories)
        avg_assoc = total_assoc / total if total > 0 else 0
        if avg_assoc > 2:
            traits.append(TraitEmergence(
                trait="Interconnected", confidence=min(1.0, avg_assoc / 5),
                evidence_tags=("associations",), memory_count=int(total_assoc),
            ))

        traits.sort(key=lambda t: t.confidence, reverse=True)

        self._trait_cache = traits
        self._cache_time = now
        self._cache_memory_count = len(memories)
        return traits

    def analyze_emotional_trends(self, memories: list[Memory]) -> EmotionalTrends:
        """Analyze emotional patterns across memories."""
        if not memories:
            return EmotionalTrends(dominant_moods=(), emotional_volatility=0.0, recent_trend="stable")

        mood_tags = {"curious", "focused", "contemplative", "excited", "playful",
                     "empathetic", "urgent", "calm", "inspired", "melancholy"}
        mood_counts: dict[str, int] = {}
        for mem in memories:
            for tag in mem.tags:
                if tag in mood_tags:
                    mood_counts[tag] = mood_counts.get(tag, 0) + 1

        dominant = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_moods = tuple(m for m, _ in dominant[:3])

        variety = len(mood_counts)
        volatility = min(1.0, variety / len(mood_tags)) if mood_tags else 0.0

        recent_count = max(1, len(memories) // 5)
        recent = memories[-recent_count:]
        positive_tags = {"excited", "focused", "playful", "inspired", "calm"}
        negative_tags = {"melancholy", "urgent", "contemplative"}

        pos = sum(1 for m in recent for t in m.tags if t in positive_tags)
        neg = sum(1 for m in recent for t in m.tags if t in negative_tags)

        if pos > neg * 1.5:
            trend = "improving"
        elif neg > pos * 1.5:
            trend = "declining"
        else:
            trend = "stable"

        return EmotionalTrends(
            dominant_moods=dominant_moods,
            emotional_volatility=round(volatility, 3),
            recent_trend=trend,
        )

    def get_association_network(self, memories: list[Memory]) -> dict[str, list[str]]:
        """Build the association network map."""
        return {m.id: list(m.associations) for m in memories if m.associations}

    def get_stats(self, memories: list[Memory]) -> dict[str, Any]:
        """Comprehensive memory statistics."""
        if not memories:
            return {"total": 0}

        by_type: dict[str, int] = {}
        total_weight = 0.0
        weak = 0
        strong = 0
        core = 0
        total_assoc = 0
        orphaned = 0

        mem_ids = {m.id for m in memories}
        for m in memories:
            by_type[m.type] = by_type.get(m.type, 0) + 1
            total_weight += m.weight
            if m.weight < 0.05:
                weak += 1
            if m.weight > 0.7:
                strong += 1
            if m.is_core:
                core += 1
            total_assoc += len(m.associations)
            orphaned += sum(1 for a in m.associations if a not in mem_ids)

        max_issues = max(1, len(memories) * 0.1)
        integrity = max(0.0, 1.0 - orphaned / max_issues)

        return {
            "total": len(memories),
            "by_type": by_type,
            "avg_weight": round(total_weight / len(memories), 3),
            "core_count": core,
            "weak_count": weak,
            "strong_count": strong,
            "total_associations": total_assoc,
            "orphaned_associations": orphaned,
            "integrity_score": round(integrity, 3),
        }


memory_analytics = MemoryAnalytics.get_instance()
