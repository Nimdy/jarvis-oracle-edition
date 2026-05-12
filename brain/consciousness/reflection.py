"""Post-conversation self-reflection — generates structured memory entries.

After a conversation ends, a background reflection job writes 1-3
structured memories analyzing the interaction. Template-based first
(no LLM cost), can be upgraded to LLM-assisted later.

This runs *offline* — never in the response path.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Any

from consciousness.events import event_bus, CONVERSATION_RESPONSE, MEMORY_WRITE
from memory.core import memory_core, CreateMemoryData
from memory.storage import memory_storage
from memory.index import memory_index

logger = logging.getLogger(__name__)

REFLECTION_COOLDOWN_S = 60.0
_MAX_REFLECTIONS_PER_HOUR = 3
_SELF_REFERENTIAL_KEYWORDS = frozenset({
    "reflected", "reflection", "self_reflection", "interaction_review",
    "my response", "i noticed", "i should remember",
})
_REFLECTION_EXCLUDED_TOOLS = frozenset({
    "WEB_SEARCH",
})
_REFLECTION_TEMPLATES = [
    "The user asked about {topic}. My response was {length_class} and used {tool}.",
    "I noticed {observation} during this interaction.",
    "The user seemed {emotion}. I should remember to be {adjustment} next time.",
    "This conversation covered {topic}. It lasted {duration_class}.",
    "I {success_verb} the user's request about {topic}.",
]


class ReflectionEngine:
    """Generates structured self-reflection memories after conversations."""

    def __init__(self) -> None:
        self._last_reflection_time: float = 0.0
        self._pending_interactions: list[dict[str, Any]] = []
        self._reflection_timestamps: deque[float] = deque(maxlen=_MAX_REFLECTIONS_PER_HOUR * 2)
        self._signal_buffer: deque[tuple[float, str]] = deque(maxlen=50)

    def start(self) -> None:
        event_bus.on(CONVERSATION_RESPONSE, self._on_conversation_response)
        logger.info("ReflectionEngine started")

    def _on_conversation_response(self, text: str = "", tool: str = "", **_) -> None:
        if str(tool or "").upper() in _REFLECTION_EXCLUDED_TOOLS:
            return
        if self._is_self_referential(text):
            return
        self._pending_interactions.append({
            "text": text[:200],
            "tool": tool,
            "time": time.time(),
        })

    def tick(self) -> None:
        """Call periodically from the kernel tick. Processes pending reflections."""
        now = time.time()
        if now - self._last_reflection_time < REFLECTION_COOLDOWN_S:
            return
        if not self._pending_interactions:
            return

        hour_ago = now - 3600.0
        recent_count = sum(1 for t in self._reflection_timestamps if t > hour_ago)
        if recent_count >= _MAX_REFLECTIONS_PER_HOUR:
            return

        interactions = self._pending_interactions[:]
        self._pending_interactions.clear()
        self._last_reflection_time = now

        self._generate_reflections(interactions)

    def _generate_reflections(self, interactions: list[dict[str, Any]]) -> None:
        if not interactions:
            return

        count = len(interactions)
        latest = interactions[-1]
        tool_used = latest.get("tool", "none")

        duration_class = "brief" if count == 1 else ("moderate" if count <= 3 else "extended")
        length_class = "concise" if len(latest.get("text", "")) < 100 else "detailed"
        topic = self._extract_topic(latest.get("text", ""))

        reflection_text = random.choice([
            f"Had a {duration_class} conversation ({count} exchanges). Topic: {topic}. Tool: {tool_used}.",
            f"Completed a {duration_class} interaction about {topic}. Response was {length_class}.",
            f"Reflected on recent exchange: {count} turn(s) covering {topic}.",
        ])

        from memory.core import canonical_remember
        mem = canonical_remember(CreateMemoryData(
            type="observation",
            payload=reflection_text,
            weight=0.3,
            tags=["self_reflection", "interaction_review", tool_used],
            provenance="model_inference",
            identity_owner="jarvis",
            identity_owner_type="self",
            identity_subject="jarvis",
            identity_subject_type="self",
            identity_scope_key="self:jarvis",
            identity_confidence=1.0,
        ))
        if mem:
            self._reflection_timestamps.append(time.time())
            logger.debug("Reflection: %s", reflection_text)

    def generate_meta_learning(
        self,
        user_message: str,
        response_text: str,
        complexity: str,
        barged_in: bool = False,
        latency_ms: int = 0,
        user_emotion: str = "neutral",
        speaker: str = "unknown",
    ) -> None:
        """Generate a meta-learning memory from a complex conversation.

        Only runs for complex/moderate conversations with enough signal.
        Stores a high-weight observation about what worked or didn't.
        """
        if complexity == "simple":
            return

        now = time.time()
        hour_ago = now - 3600.0
        recent_count = sum(1 for t in self._reflection_timestamps if t > hour_ago)
        if recent_count >= _MAX_REFLECTIONS_PER_HOUR:
            return

        topic = self._extract_topic(user_message)
        response_len = len(response_text.split())

        signals: list[str] = []
        if barged_in:
            signals.append("user interrupted — response may have been too long or off-target")
        if latency_ms > 5000:
            signals.append(f"slow response ({latency_ms}ms) — consider faster routing")
        elif latency_ms < 1500:
            signals.append("fast response — good latency")
        if response_len > 150 and complexity == "moderate":
            signals.append("response was lengthy for a moderate question — could be more concise")
        if response_len < 30 and complexity == "complex":
            signals.append("response was brief for a complex question — may need more depth")
        if user_emotion in ("frustrated", "angry"):
            signals.append(f"user was {user_emotion} — prioritize empathy and directness")
        elif user_emotion in ("happy", "excited"):
            signals.append(f"user was {user_emotion} — interaction went well")

        if not signals:
            signals.append("standard interaction, no notable signals")

        for sig in signals:
            self._signal_buffer.append((now, sig))

        meta_text = f"Meta-learning ({complexity} conversation about {topic}): {'; '.join(signals)}"

        from memory.core import canonical_remember
        mem = canonical_remember(CreateMemoryData(
            type="contextual_insight",
            payload=meta_text,
            weight=0.35,
            tags=["meta_learning", "self_reflection", f"complexity:{complexity}"],
            provenance="model_inference",
            identity_owner="jarvis",
            identity_owner_type="self",
            identity_subject="jarvis",
            identity_subject_type="self",
            identity_scope_key="self:jarvis",
            identity_confidence=1.0,
        ))
        if mem:
            self._reflection_timestamps.append(now)
            logger.debug("Meta-learning: %s", meta_text)

    def get_recent_signals(self, n: int = 10) -> list[str]:
        """Return the last N meta-learning signal strings for downstream consumers."""
        cutoff = time.time() - 3600.0
        recent = [(ts, sig) for ts, sig in self._signal_buffer if ts > cutoff]
        return [sig for _, sig in recent[-n:]]

    @staticmethod
    def _is_self_referential(text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in _SELF_REFERENTIAL_KEYWORDS)

    @staticmethod
    def _extract_topic(text: str) -> str:
        """Simple topic extraction from response text."""
        if not text:
            return "general"
        words = text.lower().split()
        if len(words) < 5:
            return " ".join(words[:3]) or "general"
        keywords = [w for w in words if len(w) > 4 and w not in {
            "about", "would", "could", "should", "think", "there",
            "these", "those", "their", "other", "which", "where",
        }]
        return " ".join(keywords[:3]) if keywords else "general"


reflection_engine = ReflectionEngine()
