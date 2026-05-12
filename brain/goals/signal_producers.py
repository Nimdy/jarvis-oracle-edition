"""Goal signal producers — bridge between live cognition and the Goal Continuity Layer.

Each producer converts a specific runtime signal source into GoalSignal objects.
All producers respect Phase 1A constraints: observational only, no execution.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from goals.goal import GoalSignal

logger = logging.getLogger(__name__)

# ── Producer telemetry ──

_producer_stats: dict[str, int] = {
    "conversation_created": 0,
    "conversation_ignored": 0,
    "metric_created": 0,
    "metric_warmup_skipped": 0,
    "metric_ignored": 0,
    "autonomy_created": 0,
    "autonomy_ignored": 0,
    "merges": 0,
    "rejections": 0,
}


def get_producer_stats() -> dict[str, int]:
    return dict(_producer_stats)


def record_observe_outcome(outcome: str) -> None:
    """Called after observe_signal to track merges and rejections."""
    if outcome == "merged":
        _producer_stats["merges"] += 1
    elif outcome in ("rejected", "rate_limited", "cooldown_blocked"):
        _producer_stats["rejections"] += 1


# ── Metric warmup gate ──

_WARMUP_MIN_UPTIME_S = 180.0
_WARMUP_MIN_TICKS = 2

_warmup_state: dict[str, Any] = {
    "ticks_seen": 0,
    "first_tick_time": 0.0,
}


def metric_warmup_ready(uptime_s: float) -> bool:
    """Returns True once the metric producer has warmed up enough to trust readings.

    Gates on: engine uptime > 180s AND at least 2 goal ticks have occurred.
    This prevents transient boot-time zeros from creating false deficit goals.
    """
    if _warmup_state["first_tick_time"] == 0.0:
        _warmup_state["first_tick_time"] = time.time()
    _warmup_state["ticks_seen"] += 1

    if uptime_s < _WARMUP_MIN_UPTIME_S:
        return False
    if _warmup_state["ticks_seen"] < _WARMUP_MIN_TICKS:
        return False
    return True


# ── Conversation producer ──

_IMPROVEMENT_RE = re.compile(
    r"\b(?:improve|fix|upgrade|enhance|make\b.{0,40}\bbetter|optimize|"
    r"work\s+on|reduce|increase|speed\s+up|tune|calibrate|retrain|"
    r"boost|sharpen|stabilize)\b",
    re.IGNORECASE,
)

_NEGATION_COMPLAINT_RE = re.compile(
    r"^(?:no[,.]?\s+|nah|nope|not really|i(?:'m| am) (?:just|not)|"
    r"forget it|never\s?mind|stop|leave me|don'?t bother|"
    r"that'?s not what|you'?re not)\b",
    re.IGNORECASE,
)

_SELF_REFERENT_RE = re.compile(
    r"\b(?:your|you|jarvis|the\s+system|the\s+brain|yourself)\b",
    re.IGNORECASE,
)

_ECHO_CAPABILITY_MARKERS: tuple[str, ...] = (
    "i don't have that capability yet",
    "let me know if you'd like",
    "currently active",
    "learning jobs",
    "camera control and slash zoom",
    "i'm actively collecting data to improve my ability",
)

_TAG_HINTS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"\b(?:emotion|emotional|feeling|mood)\b", re.I),
     ("emotion", "model", "perception")),
    (re.compile(r"\b(?:memory|memories|recall|remember)\b", re.I),
     ("memory", "retrieval", "recall")),
    (re.compile(r"\b(?:voice|speaker|speech|stt|transcription)\b", re.I),
     ("voice", "speech", "stt")),
    (re.compile(r"\b(?:wake\s*word|wake|trigger)\b", re.I),
     ("wake", "word", "detection")),
    (re.compile(r"\b(?:face|facial|recognition|identity)\b", re.I),
     ("face", "recognition", "identity")),
    (re.compile(r"\b(?:tts|voice\s*quality|speak(?:ing)?|pronunciation)\b", re.I),
     ("tts", "voice", "quality")),
    (re.compile(r"\b(?:speed|fast|slow|latency|performance|tick)\b", re.I),
     ("performance", "latency", "optimization")),
    (re.compile(r"\b(?:personality|trait|tone)\b", re.I),
     ("personality", "traits", "tone")),
    (re.compile(r"\b(?:learn|skill|capability|training)\b", re.I),
     ("learning", "skill", "capability")),
    (re.compile(r"\b(?:neural|network|hemisphere|nn|model)\b", re.I),
     ("neural", "hemisphere", "model")),
    (re.compile(r"\b(?:autonomy|research|curiosity|drive)\b", re.I),
     ("autonomy", "research", "drives")),
]

_TAG_TO_CANONICAL: dict[str, str] = {
    "emotion": "emotional model",
    "memory": "memory retrieval",
    "voice": "voice recognition",
    "wake": "wake word detection",
    "face": "face recognition",
    "tts": "text-to-speech",
    "performance": "system performance",
    "personality": "personality system",
    "learning": "skill learning",
    "neural": "neural networks",
    "autonomy": "autonomy system",
}

_STRIP_SELF_RE = re.compile(
    r"\b(?:your|you|jarvis(?:'s)?|the\s+system(?:'s)?|the\s+brain(?:'s)?|yourself)\b",
    re.IGNORECASE,
)
_STRIP_FILLER_RE = re.compile(
    r"\b(?:please|can\s+you|could\s+you|i\s+want\s+you\s+to|you\s+(?:should|need\s+to|have\s+to))\b",
    re.IGNORECASE,
)
_MULTI_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[.\s]+$")


def _normalize_title(raw: str, tags: list[str]) -> str:
    """Normalize user phrasing into a clean canonical goal title.

    Strips second-person scaffolding, filler verbs, and trailing punctuation.
    Uses tag-derived canonical subsystem names when a strong tag match exists.
    """
    title = raw.strip()[:200]

    title = _STRIP_SELF_RE.sub("", title)
    title = _STRIP_FILLER_RE.sub("", title)
    title = _MULTI_WS.sub(" ", title).strip()
    title = _TRAILING_PUNCT.sub("", title).strip()

    if not title:
        if tags:
            canonical = _TAG_TO_CANONICAL.get(tags[0], tags[0])
            return f"improve {canonical}"
        return "general improvement request"

    words = title.lower().split()
    deduped: list[str] = []
    for w in words:
        if not deduped or w != deduped[-1]:
            deduped.append(w)
    title = " ".join(deduped)

    if tags and len(title.split()) <= 3:
        canonical = _TAG_TO_CANONICAL.get(tags[0])
        if canonical and canonical.lower() not in title.lower():
            improvement_verbs = {"improve", "fix", "upgrade", "enhance", "optimize",
                                 "boost", "sharpen", "stabilize", "tune", "calibrate", "retrain"}
            first_word = title.split()[0].lower() if title else ""
            if first_word in improvement_verbs:
                title = f"{first_word} {canonical}"
            else:
                title = f"improve {canonical}: {title}"

    if title and title[0].islower():
        title = title[0].upper() + title[1:]

    return title[:120]


def _looks_like_echo_artifact(text: str) -> bool:
    """Detect assistant self-echo/capability inventory transcripts misheard as user goals."""
    lower = text.strip().lower()
    if not lower:
        return False

    if "camera control and slash zoom" in lower:
        return True

    has_capability_block = "i don't have that capability yet" in lower
    has_learning_inventory = "learning jobs" in lower and "currently active" in lower
    has_offer_tail = "let me know if you'd like" in lower
    has_self_training = "i'm actively collecting data to improve my ability" in lower

    if has_capability_block and (has_learning_inventory or has_offer_tail):
        return True
    if has_self_training and has_learning_inventory:
        return True

    marker_count = sum(1 for marker in _ECHO_CAPABILITY_MARKERS if marker in lower)
    return marker_count >= 3


def detect_conversation_goal(text: str) -> GoalSignal | None:
    """Detect whether a user utterance contains an improvement request for Jarvis.

    Returns a GoalSignal if the text expresses a request to improve/fix/enhance
    some aspect of the system. Returns None for general conversation.
    """
    if len(text) < 10 or len(text) > 500:
        _producer_stats["conversation_ignored"] += 1
        return None

    if _NEGATION_COMPLAINT_RE.search(text.strip()):
        _producer_stats["conversation_ignored"] += 1
        return None

    if _looks_like_echo_artifact(text):
        _producer_stats["conversation_ignored"] += 1
        return None

    if not _IMPROVEMENT_RE.search(text):
        _producer_stats["conversation_ignored"] += 1
        return None

    if not _SELF_REFERENT_RE.search(text):
        _producer_stats["conversation_ignored"] += 1
        return None

    tags: list[str] = []
    for pattern, tag_set in _TAG_HINTS:
        if pattern.search(text):
            tags.extend(tag_set)

    if not tags:
        tags = ["general", "improvement"]

    tags = list(dict.fromkeys(tags))[:6]
    title = _normalize_title(text, tags)

    _producer_stats["conversation_created"] += 1
    return GoalSignal(
        signal_type="user_request",
        source="conversation",
        source_scope="user",
        content=title,
        tag_cluster=tuple(tags),
        priority_hint=0.7,
    )


# ── Metric-deficit producer ──

_DEFICIT_CONFIGS: list[dict[str, Any]] = [
    {"component": "processing_health", "threshold": 0.45, "tags": ("performance", "processing", "health"),
     "template": "Processing health degraded: {value:.2f}"},
    {"component": "memory_health", "threshold": 0.40, "tags": ("memory", "health", "integrity"),
     "template": "Memory health degraded: {value:.2f}"},
    {"component": "personality_health", "threshold": 0.45, "tags": ("personality", "stability", "health"),
     "template": "Personality stability degraded: {value:.2f}"},
    {"component": "event_health", "threshold": 0.55, "tags": ("events", "reliability", "health"),
     "template": "Event system health degraded: {value:.2f}"},
]


def detect_metric_deficits(
    health_report: dict[str, Any] | None,
    calibration_state: dict[str, Any] | None,
    active_deficits: dict[str, Any] | None,
    *,
    uptime_s: float = 0.0,
) -> list[GoalSignal]:
    """Convert persistent health deficits and calibration drift into GoalSignals.

    Called from the goals tick (every 120s). Only emits signals for deficits
    that are genuinely below threshold — the GoalManager handles dedup/merge.
    Gated behind warmup to avoid false positives from boot-time transients.
    """
    if not metric_warmup_ready(uptime_s):
        _producer_stats["metric_warmup_skipped"] += 1
        return []

    signals: list[GoalSignal] = []

    if health_report:
        components = health_report.get("components", {})
        for cfg in _DEFICIT_CONFIGS:
            value = components.get(cfg["component"], 1.0)
            if value < cfg["threshold"]:
                signals.append(GoalSignal(
                    signal_type="metric_deficit",
                    source="health_monitor",
                    source_scope="metric",
                    content=cfg["template"].format(value=value),
                    tag_cluster=cfg["tags"],
                    priority_hint=0.4 + 0.3 * (1.0 - value),
                ))

    if calibration_state:
        domain_scores = calibration_state.get("domain_scores", {})
        provisional = calibration_state.get("domain_provisional", {})
        for domain, score in domain_scores.items():
            if provisional.get(domain):
                continue
            if score < 0.30:
                signals.append(GoalSignal(
                    signal_type="metric_deficit",
                    source="truth_calibration",
                    source_scope="metric",
                    content=f"Calibration domain '{domain}' critically low: {score:.2f}",
                    tag_cluster=("calibration", domain, "drift"),
                    priority_hint=0.5,
                ))

    if active_deficits:
        for metric_name, info in active_deficits.items():
            if not isinstance(info, dict):
                continue
            duration = info.get("duration_s", 0)
            if duration < 300:
                continue
            severity = info.get("severity", "low")
            if severity == "low":
                continue
            signals.append(GoalSignal(
                signal_type="metric_deficit",
                source="metric_triggers",
                source_scope="system",
                content=f"Sustained metric deficit: {metric_name} ({severity}, {duration:.0f}s)",
                tag_cluster=("metric", metric_name.replace("_", " ").split()[0], severity),
                priority_hint=0.5 if severity == "medium" else 0.6,
            ))

    if signals:
        _producer_stats["metric_created"] += len(signals)
    else:
        _producer_stats["metric_ignored"] += 1

    return signals


# ── Autonomy/drive producer ──

_MIN_COMPLETIONS_FOR_THEME = 3
_THEME_WINDOW_S = 7200.0  # 2 hours


def detect_autonomy_themes(
    completed_intents: list[dict[str, Any]],
    policy_stats: dict[str, Any] | None = None,
) -> list[GoalSignal]:
    """Detect repeated research themes from completed autonomy intents.

    Scans the last N completed intents for recurring tag clusters. When a theme
    appears >= 3 times within 2 hours, emits a GoalSignal. The GoalManager
    handles dedup so the same theme merges rather than creating duplicates.
    """
    if not completed_intents:
        return []

    now = time.time()
    cutoff = now - _THEME_WINDOW_S

    tag_groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for intent in completed_intents:
        tags = intent.get("tag_cluster", ())
        if not tags:
            continue
        ts = intent.get("completed_at", intent.get("created_at", 0))
        if ts < cutoff:
            continue
        key = frozenset(tags)
        tag_groups.setdefault(key, []).append(intent)

    signals: list[GoalSignal] = []
    for tag_set, intents in tag_groups.items():
        if len(intents) < _MIN_COMPLETIONS_FOR_THEME:
            continue

        questions = [i.get("question", "")[:80] for i in intents[:3]]
        representative = questions[0] if questions else "repeated research theme"

        signals.append(GoalSignal(
            signal_type="drive_recurrence",
            source="autonomy",
            source_scope="self",
            content=f"Repeated research theme ({len(intents)}x): {representative}",
            tag_cluster=tuple(sorted(tag_set))[:6],
            priority_hint=min(0.6, 0.3 + 0.05 * len(intents)),
        ))

    if signals:
        _producer_stats["autonomy_created"] += len(signals)
    else:
        _producer_stats["autonomy_ignored"] += 1

    return signals
