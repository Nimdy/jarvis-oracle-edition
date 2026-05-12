"""Curiosity-to-Conversation Bridge — Phase 1 of the Master Roadmap.

Bridges internal curiosity (autonomy drives, identity fusion, scene tracker,
world model) into grounded conversational questions directed at the user.

Every question must cite a specific observation — no template-only generation.

Question lifecycle:
  1. Subsystem event/state triggers candidate generation
  2. Candidate enters CuriosityQuestionBuffer (ring buffer, dedup, cooldown)
  3. Memory check: skip if a curiosity_answer memory already covers this topic
  4. Consciousness tick picks the highest-priority non-expired candidate
  5. ProactiveGovernor gates delivery (rate limit, mode, emotion)
  6. Question is spoken to the user via TTS
  7. User's response is routed back to the originating subsystem
  8. Outcome tracked: engagement/dismissal/annoyance adjusts future cooldowns

Adaptive learning:
  - Positive engagement: standard cooldown, policy NN sees positive reward
  - Dismissal/annoyance: cooldown multiplied 4x for that category,
    policy NN sees negative reward, learns to time questions better
  - Per-category satisfaction score decays toward neutral over hours

Integration points:
  - consciousness_system.py: _run_curiosity_questions tick cycle
  - proactive.py: ProactiveGovernor gating (curiosity > wellness > soul priority)
  - conversation_handler.py: response routing + outcome classification
  - engine.py: proactive speech callback
  - policy/state_encoder.py: curiosity_satisfaction signal in state vector
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import (
    CURIOSITY_QUESTION_GENERATED,
    CURIOSITY_QUESTION_ASKED,
    event_bus,
)

logger = logging.getLogger(__name__)

QuestionSource = Literal["identity", "scene", "research", "world_model", "fractal_recall"]

QUESTION_BUFFER_MAXLEN = 20
QUESTION_EXPIRY_S = 1800.0     # 30 minutes
COOLDOWN_PER_KEY_S = 3600.0    # 1 hour per dedup key
MAX_QUESTIONS_PER_HOUR = 3

DISMISSAL_COOLDOWN_MULTIPLIER = 4.0
ANNOYANCE_COOLDOWN_MULTIPLIER = 8.0
SATISFACTION_DECAY_HALFLIFE_S = 7200.0  # 2 hours

_DISMISSAL_PATTERNS = re.compile(
    r"already\s+(covered|asked|told|answered)|"
    r"we\s+(already|just)\s+(talked|discussed|covered)|"
    r"(stop|quit|enough)\s+(asking|with)\s+(that|this|the\s+question)|"
    r"i\s+(already|just)\s+(said|told|answered)|"
    r"i\s+(said|told|answered)(?:\s+you)?\s+already|"
    r"why\s+(are\s+you|do\s+you\s+keep)\s+ask",
    re.IGNORECASE,
)

_ANNOYANCE_PATTERNS = re.compile(
    r"(shut\s+up|stop\s+it|leave\s+me\s+alone|not\s+now|go\s+away|"
    r"annoying|irritating|don.t\s+(care|ask)|"
    r"i\s+don.t\s+want\s+to\s+(talk|answer))",
    re.IGNORECASE,
)

# Maturity gates — each category requires accumulated data before unlocking
UNLOCK_GATES: dict[QuestionSource, dict[str, Any]] = {
    "identity": {"min_enrolled_profiles": 1},
    "scene": {"min_entity_observations": 50},
    "research": {"min_completed_episodes": 20},
    "world_model": {"min_promotion_level": 1},
    "fractal_recall": {"min_memories": 20},
}


@dataclass
class CuriosityOutcome:
    source: QuestionSource
    question: str
    answer: str
    outcome: str  # "engaged", "dismissed", "annoyed", "ignored"
    timestamp: float = field(default_factory=time.time)


@dataclass
class CuriosityQuestion:
    source: QuestionSource
    question_text: str
    evidence: str
    priority: float
    cooldown_key: str
    created_at: float = field(default_factory=time.time)
    asked: bool = False
    asked_at: float = 0.0


def classify_curiosity_outcome(answer_text: str) -> str:
    """Classify user response to a curiosity question."""
    if _ANNOYANCE_PATTERNS.search(answer_text):
        return "annoyed"
    if _DISMISSAL_PATTERNS.search(answer_text):
        return "dismissed"
    if len(answer_text.strip()) < 3:
        return "ignored"
    return "engaged"


def _normalize_topic_fragment(text: str) -> str:
    """Normalize a topic fragment into a stable lowercase token."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return re.sub(r"_+", "_", cleaned)


def infer_curiosity_topic_tags(
    source: str,
    question: str = "",
    evidence: str = "",
) -> list[str]:
    """Infer stable topic tags so future asks can be suppressed reliably."""
    tags: list[str] = []
    question_lower = (question or "").lower()
    evidence_lower = (evidence or "").lower()

    if source == "identity":
        if (
            "unknown_voice_event" in evidence_lower
            or ("who was that" in question_lower and "don't recognize" in question_lower)
            or ("who was that" in question_lower and "voice" in question_lower)
        ):
            tags.append("curiosity_topic:unknown_voice")

    elif source == "scene":
        match = re.search(r"entity=([^,]+)", evidence, re.IGNORECASE)
        if match:
            label = _normalize_topic_fragment(match.group(1))
            if label:
                tags.append(f"curiosity_topic:scene:{label}")

    elif source == "world_model":
        match = re.search(r"delta_type=([^,]+)", evidence, re.IGNORECASE)
        if match:
            delta_type = _normalize_topic_fragment(match.group(1))
            if delta_type:
                tags.append(f"curiosity_topic:world_model:{delta_type}")

    return tags


def _matches_curiosity_topic(
    source: str,
    topic_key: str,
    tags: set[str],
    payload_lower: str,
) -> bool:
    """Check whether a stored curiosity answer already covers a topic."""
    normalized = _normalize_topic_fragment(topic_key)
    if not normalized:
        return False

    if source == "identity" and normalized == "unknown_voice":
        if "curiosity_topic:unknown_voice" in tags:
            return True
        return (
            "curiosity q (identity):" in payload_lower
            and "who was that" in payload_lower
            and ("don't recognize" in payload_lower or "voice i don't recognize" in payload_lower)
        )

    if source == "scene":
        if f"curiosity_topic:scene:{normalized}" in tags:
            return True
        return "curiosity q (scene):" in payload_lower and normalized in payload_lower

    if source == "world_model":
        if f"curiosity_topic:world_model:{normalized}" in tags:
            return True
        return "curiosity q (world_model):" in payload_lower and normalized in payload_lower

    return normalized in payload_lower


def has_existing_answer(source: str, topic_key: str) -> bool:
    """Check if memory already has a curiosity_answer for this topic.

    Searches memory by keyword and checks for the curiosity_answer tag
    combined with the source-specific tag (e.g. curiosity_scene).
    """
    try:
        from memory.storage import memory_storage

        source_tag = f"curiosity_{source}"
        results = memory_storage.get_by_tag("curiosity_answer")
        for mem in results:
            tags = getattr(mem, "tags", None) or []
            if "curiosity_answer" not in tags or source_tag not in tags:
                continue
            payload = getattr(mem, "payload", "")
            payload_lower = payload.lower() if isinstance(payload, str) else str(payload).lower()
            if _matches_curiosity_topic(source, topic_key, set(tags), payload_lower):
                return True
    except Exception:
        pass
    return False


class CuriosityQuestionBuffer:
    """Ring buffer of candidate curiosity questions with dedup, expiry,
    memory-aware suppression, and adaptive per-category cooldowns."""

    _instance: CuriosityQuestionBuffer | None = None

    def __init__(self) -> None:
        self._buffer: deque[CuriosityQuestion] = deque(maxlen=QUESTION_BUFFER_MAXLEN)
        self._cooldowns: dict[str, float] = {}
        self._cooldown_multipliers: dict[str, float] = {}
        self._asked_times: deque[float] = deque(maxlen=MAX_QUESTIONS_PER_HOUR * 2)
        self._total_generated: int = 0
        self._total_asked: int = 0
        self._total_expired: int = 0
        self._total_suppressed: int = 0
        self._total_memory_blocked: int = 0
        self._pending_answer_source: QuestionSource | None = None
        self._pending_answer_evidence: str = ""
        self._pending_question_text: str = ""
        self._category_satisfaction: dict[str, float] = {}
        self._category_satisfaction_ts: dict[str, float] = {}
        self._outcomes: deque[CuriosityOutcome] = deque(maxlen=50)

    @classmethod
    def get_instance(cls) -> CuriosityQuestionBuffer:
        if cls._instance is None:
            cls._instance = CuriosityQuestionBuffer()
        return cls._instance

    def add(self, question: CuriosityQuestion) -> bool:
        """Add a candidate question. Returns False if suppressed by dedup/cooldown/memory."""
        now = time.time()

        effective_cooldown = COOLDOWN_PER_KEY_S * self._cooldown_multipliers.get(
            question.cooldown_key, 1.0
        )
        last_cooldown = self._cooldowns.get(question.cooldown_key, 0.0)
        if now - last_cooldown < effective_cooldown:
            self._total_suppressed += 1
            return False

        for existing in self._buffer:
            if existing.cooldown_key == question.cooldown_key and not existing.asked:
                self._total_suppressed += 1
                return False

        self._buffer.append(question)
        self._total_generated += 1

        event_bus.emit(
            CURIOSITY_QUESTION_GENERATED,
            source=question.source,
            question=question.question_text,
            evidence=question.evidence,
            priority=question.priority,
        )
        return True

    def get_best_candidate(self) -> CuriosityQuestion | None:
        """Return the highest-priority non-expired, non-asked question."""
        now = time.time()
        best: CuriosityQuestion | None = None
        expired_indices: list[int] = []

        for i, q in enumerate(self._buffer):
            if q.asked:
                continue
            if now - q.created_at > QUESTION_EXPIRY_S:
                expired_indices.append(i)
                continue
            if best is None or q.priority > best.priority:
                best = q

        self._total_expired += len(expired_indices)
        return best

    def mark_asked(self, question: CuriosityQuestion) -> None:
        """Mark a question as asked and record the cooldown."""
        now = time.time()
        question.asked = True
        question.asked_at = now
        self._cooldowns[question.cooldown_key] = now
        self._asked_times.append(now)
        self._total_asked += 1
        self._pending_answer_source = question.source
        self._pending_answer_evidence = question.evidence
        self._pending_question_text = question.question_text

        event_bus.emit(
            CURIOSITY_QUESTION_ASKED,
            source=question.source,
            question=question.question_text,
            evidence=question.evidence,
        )

    def record_outcome(self, source: str, question: str, answer: str, outcome: str) -> None:
        """Record curiosity interaction outcome and adapt cooldowns."""
        self._outcomes.append(CuriosityOutcome(
            source=source,  # type: ignore[arg-type]
            question=question,
            answer=answer,
            outcome=outcome,
        ))

        cat_key = f"category:{source}"
        now = time.time()

        if outcome == "engaged":
            self._category_satisfaction[cat_key] = min(
                1.0, self._get_satisfaction(cat_key) + 0.2
            )
        elif outcome == "dismissed":
            self._category_satisfaction[cat_key] = max(
                -1.0, self._get_satisfaction(cat_key) - 0.4
            )
            for key, ts in list(self._cooldowns.items()):
                if key.startswith(hashlib.md5(source.encode()).hexdigest()[:4]):
                    self._cooldown_multipliers[key] = DISMISSAL_COOLDOWN_MULTIPLIER
        elif outcome == "annoyed":
            self._category_satisfaction[cat_key] = max(
                -1.0, self._get_satisfaction(cat_key) - 0.7
            )
            for key in list(self._cooldowns.keys()):
                self._cooldown_multipliers[key] = max(
                    self._cooldown_multipliers.get(key, 1.0),
                    ANNOYANCE_COOLDOWN_MULTIPLIER,
                )

        self._category_satisfaction_ts[cat_key] = now
        logger.info(
            "Curiosity outcome: source=%s outcome=%s satisfaction=%.2f",
            source, outcome, self._get_satisfaction(cat_key),
        )

    def _get_satisfaction(self, cat_key: str) -> float:
        """Get category satisfaction with time decay toward 0 (neutral)."""
        raw = self._category_satisfaction.get(cat_key, 0.0)
        if raw == 0.0:
            return 0.0
        ts = self._category_satisfaction_ts.get(cat_key, 0.0)
        elapsed = time.time() - ts
        decay = 0.5 ** (elapsed / SATISFACTION_DECAY_HALFLIFE_S)
        return raw * decay

    def get_category_satisfaction(self, source: str) -> float:
        """Public accessor for policy state encoder."""
        return self._get_satisfaction(f"category:{source}")

    def get_overall_satisfaction(self) -> float:
        """Weighted average satisfaction across all categories."""
        values = [self._get_satisfaction(k) for k in self._category_satisfaction]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def get_pending_answer_context(self) -> dict[str, Any] | None:
        """If we asked a curiosity question, return its context for response routing."""
        if self._pending_answer_source is None:
            return None
        return {
            "source": self._pending_answer_source,
            "evidence": self._pending_answer_evidence,
            "question": self._pending_question_text,
        }

    def clear_pending_answer(self) -> None:
        self._pending_answer_source = None
        self._pending_answer_evidence = ""
        self._pending_question_text = ""

    def hourly_count(self) -> int:
        """How many questions asked in the last hour."""
        now = time.time()
        cutoff = now - 3600.0
        return sum(1 for t in self._asked_times if t > cutoff)

    def get_stats(self) -> dict[str, Any]:
        now = time.time()
        active_questions = [
            q for q in self._buffer if now - q.created_at <= QUESTION_EXPIRY_S
        ]
        pending = sum(1 for q in active_questions if not q.asked)
        return {
            "buffer_size": len(self._buffer),
            "pending_candidates": pending,
            "total_generated": self._total_generated,
            "total_asked": self._total_asked,
            "total_expired": self._total_expired,
            "total_suppressed": self._total_suppressed,
            "total_memory_blocked": self._total_memory_blocked,
            "hourly_asked": self.hourly_count(),
            "has_pending_answer": self._pending_answer_source is not None,
            "pending_source": self._pending_answer_source,
            "overall_satisfaction": round(self.get_overall_satisfaction(), 2),
            "recent_outcomes": [o.outcome for o in list(self._outcomes)[-5:]],
            "recent_questions": [
                {
                    "source": q.source,
                    "question": q.question_text,
                    "evidence": q.evidence,
                    "priority": round(q.priority, 3),
                    "asked": q.asked,
                    "created_at": q.created_at,
                    "asked_at": q.asked_at,
                }
                for q in list(active_questions)[-5:]
            ],
        }


def _dedup_key(source: str, *parts: str) -> str:
    """Build a stable dedup key from source + identifying parts."""
    raw = f"{source}:{'|'.join(parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Category-specific question generators ──────────────────────────

def check_identity_curiosity(identity_status: dict[str, Any]) -> CuriosityQuestion | None:
    """Generate a question when identity fusion sees an unknown person directly."""
    recognition_state = identity_status.get("recognition_state", "absent")
    is_known = identity_status.get("is_known", False)
    user_present = identity_status.get("user_present", False)

    if recognition_state == "absent" and not user_present:
        return None
    if is_known:
        return None
    if recognition_state not in ("unknown_present", "tentative_match"):
        return None

    key = _dedup_key("identity", "unknown_visitor")
    return CuriosityQuestion(
        source="identity",
        question_text="I noticed someone I don't recognize. Would you like to tell me who that is so I can remember them?",
        evidence=f"recognition_state={recognition_state}, is_known={is_known}",
        priority=0.8,
        cooldown_key=key,
    )


def check_unknown_speaker_curiosity(
    unknown_events: list[dict[str, Any]],
    primary_user: str = "",
) -> CuriosityQuestion | None:
    """Generate a genuine curiosity question when an unknown voice was heard
    while a known user is present. This is the highest-priority identity
    question because Jarvis has an active information gap: someone spoke
    and Jarvis doesn't know who they are.

    Skips if a curiosity_answer for an unknown voice already exists in memory
    (user already told us who it was -- we just need enrollment, not another ask).
    """
    if not unknown_events:
        return None

    if has_existing_answer("identity", "unknown_voice"):
        return None

    event = unknown_events[-1]
    had_known_user = event.get("had_known_user", False)
    companion = event.get("primary_user", "") or primary_user
    ts = event.get("timestamp", 0.0)

    key = _dedup_key("identity", "unknown_voice", str(int(ts / 300)))

    if had_known_user and companion:
        question = (
            f"I heard someone speaking that I don't recognize — "
            f"it wasn't you, {companion}. Who was that? "
            f"I'd like to know them."
        )
    else:
        question = (
            "I heard a voice I don't recognize. "
            "Who was that? I'd like to remember them for next time."
        )

    return CuriosityQuestion(
        source="identity",
        question_text=question,
        evidence=f"unknown_voice_event at {ts:.0f}, companion={companion}",
        priority=0.9,
        cooldown_key=key,
    )


def check_scene_curiosity(scene_state: dict[str, Any]) -> CuriosityQuestion | None:
    """Generate a question about novel objects in the scene.
    Skips objects that already have a curiosity_answer in memory."""
    entities = scene_state.get("entities", [])
    if not entities:
        return None

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        state = entity.get("state", "")
        stable_cycles = entity.get("stable_cycles", 0)
        label = entity.get("label", "unknown")

        if state == "visible" and 5 <= stable_cycles <= 20:
            region = entity.get("region", "nearby")
            if has_existing_answer("scene", label):
                continue
            key = _dedup_key("scene", label, region)
            return CuriosityQuestion(
                source="scene",
                question_text=f"I see something that looks like a {label} {region}. Is that new, or has it been there?",
                evidence=f"entity={label}, region={region}, stable_cycles={stable_cycles}",
                priority=0.6,
                cooldown_key=key,
            )
    return None


def check_research_curiosity(autonomy_status: dict[str, Any]) -> CuriosityQuestion | None:
    """Generate a question when research was inconclusive."""
    completed = autonomy_status.get("completed", [])
    if not completed:
        return None

    for entry in reversed(completed):
        if not isinstance(entry, dict):
            continue
        worked = entry.get("worked", True)
        question = entry.get("question", "")
        if not worked and question:
            key = _dedup_key("research", question[:60])
            return CuriosityQuestion(
                source="research",
                question_text=f"I was trying to learn about something but couldn't find a clear answer. Do you know about: {question[:80]}?",
                evidence=f"inconclusive_research: {question[:80]}",
                priority=0.5,
                cooldown_key=key,
            )
    return None


def check_world_model_curiosity(world_model_state: dict[str, Any]) -> CuriosityQuestion | None:
    """Generate a question about unexplained world model deltas."""
    deltas = world_model_state.get("recent_deltas", [])
    if not deltas:
        return None

    for delta in reversed(deltas):
        if not isinstance(delta, dict):
            continue
        delta_type = delta.get("type", "")
        description = delta.get("description", "")
        if delta_type in ("user_departed", "user_arrived"):
            continue
        if description:
            key = _dedup_key("world_model", delta_type, description[:40])
            return CuriosityQuestion(
                source="world_model",
                question_text=f"I noticed something changed: {description[:100]}. Was that intentional?",
                evidence=f"delta_type={delta_type}, description={description[:80]}",
                priority=0.4,
                cooldown_key=key,
            )
    return None


def ingest_fractal_recall(result: Any) -> bool:
    """Intake adapter: convert a FractalRecallResult into a curiosity question.

    Only processes results whose governance recommends ``hold_for_curiosity``.
    Returns True if the question was accepted by the buffer.
    """
    action = getattr(result, "governance_recommended_action", "ignore")
    if action != "hold_for_curiosity":
        return False

    chain = getattr(result, "chain", []) or []
    cue = getattr(result, "cue", None)
    if not chain or not cue:
        return False

    # Derive question text from cue + chain payload
    seed = chain[0]
    seed_payload = getattr(getattr(seed, "memory", None), "payload", "")
    if isinstance(seed_payload, str):
        snippet = seed_payload[:100]
    else:
        snippet = str(seed_payload)[:100]

    topic = getattr(cue, "topic", "") or ""
    cue_class = getattr(cue, "cue_class", "")

    question_text = f"I recalled something that seems relevant: \"{snippet}\""
    if topic:
        question_text += f" — it connects to {topic}."
    else:
        question_text += "."

    evidence_parts = [
        f"cue_class={cue_class}",
        f"chain_len={len(chain)}",
        f"seed_resonance={getattr(seed, 'resonance', 0.0):.3f}",
    ]
    reason_codes = getattr(result, "governance_reason_codes", ())
    if reason_codes:
        evidence_parts.append(f"reasons={','.join(reason_codes)}")
    evidence = ", ".join(evidence_parts)

    # Priority based on resonance + emotional salience
    avg_resonance = sum(getattr(c, "resonance", 0.0) for c in chain) / max(len(chain), 1)
    emotional_bump = 0.1 if (getattr(cue, "emotion", "neutral") != "neutral") else 0.0
    priority = min(0.9, 0.3 + avg_resonance * 0.5 + emotional_bump)

    key = _dedup_key("fractal_recall", getattr(seed, "memory_id", ""), str(int(getattr(result, "timestamp", 0.0) / 300)))

    q = CuriosityQuestion(
        source="fractal_recall",
        question_text=question_text,
        evidence=evidence,
        priority=priority,
        cooldown_key=key,
    )
    return curiosity_buffer.add(q)


# ── Module-level singleton ─────────────────────────────────────────

curiosity_buffer = CuriosityQuestionBuffer.get_instance()
