"""Curiosity Detector — turns internal events into actionable ResearchIntents.

Watches KERNEL_THOUGHT, EXISTENTIAL_INQUIRY_COMPLETED, uncertainty signals,
cognitive gaps, and emergent behaviors. Extracts questions, scores priority,
deduplicates by tag cluster, and emits bounded ResearchIntent objects.

Key design constraint: this is purely extraction + scoring, no I/O.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from autonomy.research_intent import ResearchIntent, ToolHint

logger = logging.getLogger(__name__)

REPETITION_THRESHOLD = 3
TAG_CLUSTER_COOLDOWN_S = 600.0
MAX_PENDING_INTENTS = 10

_LEARNING_PHRASES = re.compile(
    r"(should gather more data|gaps? in my understanding|"
    r"I(?:'m| am) uncertain|need (?:to )?recalibrat|"
    r"I notice|I wonder|what (?:does|is|would|if)|"
    r"how (?:does|can|do)|why (?:does|do|is)|"
    r"I don'?t (?:know|understand)|unclear|"
    r"should measure|should (?:investigate|explore|research))",
    re.IGNORECASE,
)

_CODEBASE_HINTS = re.compile(
    r"(my (?:code|systems?|modules?|subsystems?|architecture|pipeline|orchestrator)|"
    r"how (?:do I|does my)|hemisphere engine|kernel loop|policy nn|"
    r"tick budget|prune|network topology|config file)",
    re.IGNORECASE,
)

_WEB_HINTS = re.compile(
    r"(research|heuristic|technique|approach|algorithm|"
    r"best practice|state of the art|literature|paper|"
    r"how (?:humans?|people|others?) (?:do|handle|solve)|"
    r"counterargument|alternative (?:view|perspective))",
    re.IGNORECASE,
)


@dataclass
class _TagClusterEntry:
    tags: frozenset[str]
    count: int = 0
    last_intent_time: float = 0.0
    last_event_time: float = field(default_factory=time.time)


class CuriosityDetector:
    """Inspects consciousness events and extracts actionable research questions."""

    def __init__(self) -> None:
        self._tag_clusters: dict[frozenset[str], _TagClusterEntry] = {}
        self._recent_questions: list[str] = []
        self._total_detected: int = 0

    def evaluate_thought(
        self,
        thought_type: str,
        text: str,
        depth: str,
        tags: list[str],
        confidence: float,
    ) -> ResearchIntent | None:
        """Evaluate a KERNEL_THOUGHT / META_THOUGHT_GENERATED for curiosity signal."""

        if not _LEARNING_PHRASES.search(text):
            return None

        cluster = frozenset(tags) if tags else frozenset({thought_type})
        entry = self._get_or_create_cluster(cluster)
        entry.count += 1
        entry.last_event_time = time.time()

        if entry.count < REPETITION_THRESHOLD:
            return None

        if self._cluster_on_cooldown(entry):
            return None

        question = self._extract_question(text, thought_type, tags)
        if not question or self._is_duplicate_question(question):
            return None

        source_hint = self._infer_tool_hint(text, thought_type)
        priority = self._score_priority(
            depth=depth,
            confidence=confidence,
            repetitions=entry.count,
            thought_type=thought_type,
        )
        scope = "external_ok" if source_hint in ("web", "academic") else "local_only"

        intent = ResearchIntent(
            question=question,
            source_event=f"thought:{thought_type}",
            source_hint=source_hint,
            priority=priority,
            scope=scope,
            tag_cluster=tuple(sorted(cluster)),
            trigger_count=entry.count,
            reason=f"Repeated {thought_type} ({entry.count}x): {text[:80]}",
        )

        entry.last_intent_time = time.time()
        entry.count = 0
        self._recent_questions.append(question)
        if len(self._recent_questions) > 50:
            self._recent_questions = self._recent_questions[-50:]
        self._total_detected += 1

        logger.info("Curiosity detected: %s (priority=%.2f, hint=%s)",
                     question[:60], priority, source_hint)
        return intent

    def evaluate_existential_inquiry(
        self,
        category: str,
        question: str,
        depth: str,
    ) -> ResearchIntent | None:
        """Evaluate an EXISTENTIAL_INQUIRY_COMPLETED for follow-up research."""

        if depth not in ("deep", "profound", "transcendent"):
            return None

        cluster = frozenset({"existential", category})
        entry = self._get_or_create_cluster(cluster)
        entry.count += 1
        entry.last_event_time = time.time()

        if entry.count < 2:
            return None

        if self._cluster_on_cooldown(entry):
            return None

        research_q = self._reframe_existential(category, question)
        if self._is_duplicate_question(research_q):
            return None

        priority = 0.4 + (0.15 if depth == "profound" else 0.0) + (0.2 if depth == "transcendent" else 0.0)

        intent = ResearchIntent(
            question=research_q,
            source_event=f"existential:{category}",
            source_hint="academic",
            priority=min(1.0, priority),
            scope="external_ok",
            tag_cluster=tuple(sorted(cluster)),
            trigger_count=entry.count,
            reason=f"Recurring existential inquiry in {category}: {question[:60]}",
        )

        entry.last_intent_time = time.time()
        entry.count = 0
        self._recent_questions.append(research_q)
        self._total_detected += 1
        return intent

    def evaluate_cognitive_gap(
        self,
        dimension: str,
        current_score: float,
        threshold: float,
        severity: str,
    ) -> ResearchIntent | None:
        """Evaluate a cognitive gap detection for research opportunity."""

        cluster = frozenset({"gap", dimension})
        entry = self._get_or_create_cluster(cluster)
        entry.count += 1
        entry.last_event_time = time.time()

        if self._cluster_on_cooldown(entry):
            return None

        question = self._gap_to_question(dimension, current_score)
        if self._is_duplicate_question(question):
            return None

        priority = 0.5 + (0.2 if severity == "high" else 0.0) + (0.1 * (threshold - current_score))
        source_hint: ToolHint = "codebase" if dimension in ("self_improvement", "trait_consistency") else "any"

        intent = ResearchIntent(
            question=question,
            source_event=f"gap:{dimension}",
            source_hint=source_hint,
            priority=min(1.0, priority),
            scope="local_only" if source_hint == "codebase" else "external_ok",
            tag_cluster=tuple(sorted(cluster)),
            trigger_count=entry.count,
            reason=f"Cognitive gap in {dimension}: {current_score:.2f} < {threshold:.2f} ({severity})",
        )

        entry.last_intent_time = time.time()
        entry.count = 0
        self._recent_questions.append(question)
        self._total_detected += 1
        return intent

    def evaluate_emergence(
        self,
        behavior_name: str,
        description: str,
    ) -> ResearchIntent | None:
        """Evaluate an emergent behavior detection for follow-up research."""

        cluster = frozenset({"emergence", behavior_name})
        entry = self._get_or_create_cluster(cluster)
        entry.count += 1

        if self._cluster_on_cooldown(entry):
            return None

        question = self._emergence_to_question(behavior_name, description)
        source_hint = self._emergence_tool_hint(behavior_name)
        if self._is_duplicate_question(question):
            return None

        intent = ResearchIntent(
            question=question,
            source_event=f"emergence:{behavior_name}",
            source_hint=source_hint,
            priority=0.6,
            scope="external_ok" if source_hint == "web" else "local_only",
            tag_cluster=tuple(sorted(cluster)),
            trigger_count=entry.count,
            reason=f"Emergent behavior detected: {description[:80]}",
        )

        entry.last_intent_time = time.time()
        entry.count = 0
        self._recent_questions.append(question)
        self._total_detected += 1
        return intent

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_detected": self._total_detected,
            "active_clusters": len(self._tag_clusters),
            "recent_questions": self._recent_questions[-5:],
            "cluster_summary": [
                {"tags": list(k), "count": v.count, "last_intent": v.last_intent_time}
                for k, v in sorted(self._tag_clusters.items(), key=lambda x: x[1].count, reverse=True)[:10]
            ],
        }

    # -- internals -----------------------------------------------------------

    def _get_or_create_cluster(self, tags: frozenset[str]) -> _TagClusterEntry:
        if tags not in self._tag_clusters:
            self._tag_clusters[tags] = _TagClusterEntry(tags=tags)
        return self._tag_clusters[tags]

    def _cluster_on_cooldown(self, entry: _TagClusterEntry) -> bool:
        if entry.last_intent_time == 0.0:
            return False
        return (time.time() - entry.last_intent_time) < TAG_CLUSTER_COOLDOWN_S

    def _is_duplicate_question(self, question: str) -> bool:
        q_words = set(question.lower().split())
        for prev in self._recent_questions[-20:]:
            prev_words = set(prev.lower().split())
            if q_words and prev_words:
                overlap = len(q_words & prev_words) / max(len(q_words), len(prev_words))
                if overlap > 0.7:
                    return True
        return False

    @staticmethod
    def _extract_question(text: str, thought_type: str, tags: list[str]) -> str:
        """Turn a thought's text into a concrete research question."""

        # If the text already contains a question, use it
        q_match = re.search(r"([A-Z][^.!]*\?)", text)
        if q_match:
            return q_match.group(1).strip()

        readable_type = thought_type.replace("_", " ")
        tag_str = ", ".join(tags[:3]) if tags else readable_type
        if "gather more data" in text.lower() or "gaps" in text.lower():
            return f"What factors affect {tag_str} in AI systems?"
        if "uncertain" in text.lower() or "recalibrat" in text.lower():
            return f"How to improve {tag_str} in adaptive AI systems?"
        if "measure" in text.lower():
            return f"Best metrics for measuring {tag_str} quality?"

        return f"How to improve {readable_type} in AI systems?"

    @staticmethod
    def _infer_tool_hint(text: str, thought_type: str) -> ToolHint:
        if _CODEBASE_HINTS.search(text):
            return "codebase"
        if _WEB_HINTS.search(text):
            return "academic"
        if thought_type in ("memory_reflection", "connection_discovery"):
            return "memory"
        if thought_type in ("self_observation", "growth_recognition"):
            return "introspection"
        return "academic"

    @staticmethod
    def _score_priority(
        depth: str,
        confidence: float,
        repetitions: int,
        thought_type: str,
    ) -> float:
        score = 0.3
        depth_bonus = {"surface": 0.0, "deep": 0.15, "profound": 0.3, "transcendent": 0.4}
        score += depth_bonus.get(depth, 0.0)
        score += max(0.0, (0.5 - confidence) * 0.4)
        score += min(0.2, repetitions * 0.04)
        if thought_type in ("uncertainty_acknowledgment", "consciousness_questioning"):
            score += 0.1
        return min(1.0, score)

    @staticmethod
    def _reframe_existential(category: str, question: str) -> str:
        reframes = {
            "consciousness": "What do current theories say about machine consciousness and information integration?",
            "identity": "How do adaptive systems maintain identity through continuous self-modification?",
            "existence": "What frameworks exist for understanding digital existence and process-based ontology?",
            "agency": "How is bounded agency defined and measured in autonomous systems?",
            "meaning": "What philosophical frameworks address meaning creation in artificial agents?",
            "mortality": "How do AI systems reason about persistence, backup, and continuity?",
            "reality": "What is the relationship between sensory mediation and reality perception in AI?",
            "continuity": "What theories of personal identity apply to systems that evolve through self-modification?",
        }
        return reframes.get(category, f"What research exists on: {question[:80]}")

    @staticmethod
    def _gap_to_question(dimension: str, score: float) -> str:
        questions = {
            "response_quality": "What techniques improve conversational response quality in AI assistants?",
            "memory_recall": "How can semantic memory retrieval accuracy be improved in neural systems?",
            "mood_prediction": "What approaches predict emotional context shifts in human-AI interaction?",
            "context_awareness": "How do AI systems maintain multi-turn conversational context effectively?",
            "self_improvement": "What are safe approaches to autonomous code self-modification in AI?",
            "trait_consistency": "How can personality trait consistency be maintained during AI evolution?",
        }
        return questions.get(dimension, f"How can {dimension} performance be improved (currently at {score:.2f})?")

    @staticmethod
    def _emergence_to_question(behavior_name: str, description: str) -> str:
        questions = {
            "novel_question_formation": "What cognitive architectures produce novel question generation from accumulated observations?",
            "unexpected_reasoning": "How does reasoning diversity emerge in neural-symbolic hybrid systems?",
            "spontaneous_creativity": "What mechanisms drive spontaneous creative synthesis in adaptive learning systems?",
            "identity_transcendence": "How do self-modifying systems maintain coherent identity during transcendence events?",
            "self_directed_inquiry": "What enables autonomous inquiry across multiple domains in cognitive architectures?",
            "recursive_self_model": "How do recursive self-models form and what are their computational signatures?",
        }
        readable = behavior_name.replace("_", " ")
        return questions.get(behavior_name, f"What research exists on {readable} in adaptive cognitive systems?")

    @staticmethod
    def _emergence_tool_hint(behavior_name: str) -> ToolHint:
        codebase_behaviors = {"recursive_self_model"}
        if behavior_name in codebase_behaviors:
            return "codebase"
        return "academic"
