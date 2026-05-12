"""Research Governor — rate limits, mode gating, topic boundaries, budget caps.

This is the autonomy equivalent of MutationGovernor: it prevents Jarvis from
becoming a runaway background web crawler while still allowing bounded,
purposeful autonomous research.

Safety-first policy:
  - Research runs only in non-conversational modes (passive, sleep, dreaming,
    reflective, deep_learning)
  - External web searches blocked by default unless intent explicitly allows
  - Hourly and daily rate limits
  - Per-topic cooldowns
  - Total token/time budget caps
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)

MAX_RESEARCH_PER_HOUR = 8
MAX_RESEARCH_PER_DAY = 30
MAX_WEB_SEARCHES_PER_HOUR = 3
MAX_ACADEMIC_SEARCHES_PER_HOUR = 10
MAX_CONCURRENT_RESEARCH = 1
TOPIC_COOLDOWN_S = 600.0
CLUSTER_OVERLAP_THRESHOLD = 0.5
CLUSTER_COOLDOWN_S = 900.0

ALLOWED_RESEARCH_MODES = frozenset({
    "gestation", "passive", "dreaming", "reflective", "deep_learning",
})

_GESTATION_LIMITS = {
    "hourly": 50,
    "daily": 300,
    "academic_hourly": 50,
    "web_hourly": 15,
    "topic_cooldown_s": 300.0,
    "cluster_cooldown_s": 450.0,
}

BLOCKED_MODES = frozenset({
    "conversational", "focused",
})


@dataclass
class GovernorDecision:
    allowed: bool
    reason: str
    intent_id: str


@dataclass
class _TopicRecord:
    topic_key: str
    tag_set: frozenset[str] = field(default_factory=frozenset)
    last_researched: float = 0.0
    total_count: int = 0


class ResearchGovernor:
    """Gates autonomous research with rate limits, mode checks, and budgets."""

    def __init__(self) -> None:
        self._hourly_log: deque[float] = deque(maxlen=500)
        self._daily_log: deque[float] = deque(maxlen=5000)
        self._web_hourly_log: deque[float] = deque(maxlen=500)
        self._academic_hourly_log: deque[float] = deque(maxlen=500)
        self._topics: dict[str, _TopicRecord] = {}
        self._running_count: int = 0
        self._total_allowed: int = 0
        self._total_blocked: int = 0
        self._block_reasons: dict[str, int] = {}
        self._disabled: bool = False

    def evaluate(self, intent: ResearchIntent, current_mode: str) -> GovernorDecision:
        """Decide whether a research intent is allowed right now."""

        if self._disabled:
            return self._block(intent, "governor_disabled")

        if current_mode in BLOCKED_MODES:
            return self._block(intent, f"mode_blocked:{current_mode}")

        if current_mode not in ALLOWED_RESEARCH_MODES:
            return self._block(intent, f"mode_not_allowed:{current_mode}")

        if self._running_count >= MAX_CONCURRENT_RESEARCH:
            return self._block(intent, "max_concurrent_reached")

        now = time.time()
        hour_ago = now - 3600.0
        day_ago = now - 86400.0

        is_gestation = current_mode == "gestation"
        max_hourly = _GESTATION_LIMITS["hourly"] if is_gestation else MAX_RESEARCH_PER_HOUR
        max_daily = _GESTATION_LIMITS["daily"] if is_gestation else MAX_RESEARCH_PER_DAY
        max_academic = _GESTATION_LIMITS["academic_hourly"] if is_gestation else MAX_ACADEMIC_SEARCHES_PER_HOUR
        max_web = _GESTATION_LIMITS["web_hourly"] if is_gestation else MAX_WEB_SEARCHES_PER_HOUR

        hourly_count = sum(1 for t in self._hourly_log if t > hour_ago)
        if hourly_count >= max_hourly:
            return self._block(intent, f"hourly_limit:{hourly_count}/{max_hourly}")

        daily_count = sum(1 for t in self._daily_log if t > day_ago)
        if daily_count >= max_daily:
            return self._block(intent, f"daily_limit:{daily_count}/{max_daily}")

        if intent.source_hint == "academic":
            academic_hourly = sum(1 for t in self._academic_hourly_log if t > hour_ago)
            if academic_hourly >= max_academic:
                return self._block(intent, f"academic_hourly_limit:{academic_hourly}/{max_academic}")
        elif intent.source_hint == "web" or intent.scope == "external_ok":
            web_hourly = sum(1 for t in self._web_hourly_log if t > hour_ago)
            if web_hourly >= max_web:
                return self._block(intent, f"web_hourly_limit:{web_hourly}/{max_web}")

        effective_topic_cd = _GESTATION_LIMITS["topic_cooldown_s"] if is_gestation else TOPIC_COOLDOWN_S
        topic_key = self._topic_key(intent)
        topic = self._topics.get(topic_key)
        if topic and (now - topic.last_researched) < effective_topic_cd:
            remaining = effective_topic_cd - (now - topic.last_researched)
            return self._block(intent, f"topic_cooldown:{topic_key}:{remaining:.0f}s remaining")

        effective_cluster_cd = _GESTATION_LIMITS["cluster_cooldown_s"] if is_gestation else CLUSTER_COOLDOWN_S
        similar = self._find_similar_cluster(intent, now, cluster_cooldown=effective_cluster_cd)
        if similar:
            return self._block(intent, f"cluster_overlap:{similar}")

        return self._allow(intent)

    def record_start(self, intent: ResearchIntent) -> None:
        """Record that a research job has started executing."""
        self._running_count += 1

    def record_complete(self, intent: ResearchIntent) -> None:
        """Record completion and update rate limit counters."""
        self._running_count = max(0, self._running_count - 1)

        now = time.time()
        self._hourly_log.append(now)
        self._daily_log.append(now)

        if intent.source_hint == "academic":
            self._academic_hourly_log.append(now)
        elif intent.source_hint == "web" or intent.scope == "external_ok":
            self._web_hourly_log.append(now)

        topic_key = self._topic_key(intent)
        if topic_key not in self._topics:
            self._topics[topic_key] = _TopicRecord(
                topic_key=topic_key,
                tag_set=frozenset(intent.tag_cluster),
            )
        record = self._topics[topic_key]
        record.last_researched = now
        record.total_count += 1
        if intent.tag_cluster:
            record.tag_set = frozenset(intent.tag_cluster)

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled
        logger.info("Research governor %s", "disabled" if disabled else "enabled")

    def get_stats(self, current_mode: str = "") -> dict[str, Any]:
        now = time.time()
        hour_ago = now - 3600.0
        day_ago = now - 86400.0
        is_gestation = current_mode == "gestation"
        return {
            "total_allowed": self._total_allowed,
            "total_blocked": self._total_blocked,
            "running_count": self._running_count,
            "hourly_used": sum(1 for t in self._hourly_log if t > hour_ago),
            "hourly_limit": _GESTATION_LIMITS["hourly"] if is_gestation else MAX_RESEARCH_PER_HOUR,
            "daily_used": sum(1 for t in self._daily_log if t > day_ago),
            "daily_limit": _GESTATION_LIMITS["daily"] if is_gestation else MAX_RESEARCH_PER_DAY,
            "web_hourly_used": sum(1 for t in self._web_hourly_log if t > hour_ago),
            "web_hourly_limit": _GESTATION_LIMITS["web_hourly"] if is_gestation else MAX_WEB_SEARCHES_PER_HOUR,
            "academic_hourly_used": sum(1 for t in self._academic_hourly_log if t > hour_ago),
            "academic_hourly_limit": _GESTATION_LIMITS["academic_hourly"] if is_gestation else MAX_ACADEMIC_SEARCHES_PER_HOUR,
            "disabled": self._disabled,
            "block_reasons": dict(self._block_reasons),
            "active_topics": len(self._topics),
            "gestation_mode": is_gestation,
        }

    # -- internals -----------------------------------------------------------

    def _allow(self, intent: ResearchIntent) -> GovernorDecision:
        self._total_allowed += 1
        return GovernorDecision(allowed=True, reason="approved", intent_id=intent.id)

    def _block(self, intent: ResearchIntent, reason: str) -> GovernorDecision:
        self._total_blocked += 1
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1
        intent.status = "blocked"
        intent.blocked_reason = reason
        logger.debug("Research blocked: %s — %s", intent.question[:40], reason)
        return GovernorDecision(allowed=False, reason=reason, intent_id=intent.id)

    def _find_similar_cluster(
        self, intent: ResearchIntent, now: float,
        cluster_cooldown: float = CLUSTER_COOLDOWN_S,
    ) -> str:
        """Check if any recently-researched topic cluster overlaps significantly.

        Uses Jaccard similarity on full stored tag sets (not the truncated
        topic_key). This prevents 5 different intents that all touch
        "voice pacing" in a row.
        """
        intent_tags = set(intent.tag_cluster)
        if len(intent_tags) < 2:
            return ""

        for record in self._topics.values():
            if (now - record.last_researched) > cluster_cooldown:
                continue
            record_tags = set(record.tag_set) if record.tag_set else set()
            if not record_tags:
                continue
            intersection = intent_tags & record_tags
            union = intent_tags | record_tags
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= CLUSTER_OVERLAP_THRESHOLD:
                remaining = cluster_cooldown - (now - record.last_researched)
                return f"{record.topic_key}|jaccard={jaccard:.2f}|{remaining:.0f}s"

        return ""

    @staticmethod
    def _topic_key(intent: ResearchIntent) -> str:
        if intent.tag_cluster:
            return "|".join(sorted(intent.tag_cluster)[:3])
        words = intent.question.lower().split()[:4]
        return "|".join(sorted(words))
