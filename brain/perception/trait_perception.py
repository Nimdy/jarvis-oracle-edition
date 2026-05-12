"""Trait-modulated perception processing — personality traits influence how events are perceived."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PerceptionModulation:
    """Result of trait modulation on a perception event."""
    weight_multiplier: float = 1.0
    extra_tags: list[str] = field(default_factory=list)
    mood_impact: float = 0.0  # -1 to 1
    thought_trigger: str | None = None
    decay_multiplier: float = 1.0


@dataclass
class PerceptionTrigger:
    event_type: str
    weight_calc: Callable[[dict, dict], float]
    mood_calc: Callable[[dict, dict], float]
    tag_enrichment: Callable[[dict, dict], list[str]]
    thought_trigger: Callable[[dict, dict], str | None]


def _wake_word_weight(data: dict, traits: dict) -> float:
    base = 1.0
    proactive = traits.get("proactive", 0.5)
    return base + proactive * 0.3


def _wake_word_mood(data: dict, traits: dict) -> float:
    return 0.1  # slight positive


def _wake_word_tags(data: dict, traits: dict) -> list[str]:
    return ["interaction_start"]


def _wake_word_thought(data: dict, traits: dict) -> str | None:
    if traits.get("proactive", 0) > 0.7:
        return "Eagerness to engage — the proactive impulse responds."
    return None


def _transcription_weight(data: dict, traits: dict) -> float:
    base = 1.0
    empathetic = traits.get("empathetic", 0.5)
    text = str(data.get("text", "")).lower()
    emotional_words = {"feel", "happy", "sad", "angry", "worried", "love", "hate", "afraid"}
    has_emotion = any(w in text for w in emotional_words)
    if has_emotion:
        base += empathetic * 0.4
    return base


def _transcription_mood(data: dict, traits: dict) -> float:
    text = str(data.get("text", "")).lower()
    positive = {"happy", "great", "awesome", "love", "thank", "good", "excellent"}
    negative = {"sad", "angry", "bad", "hate", "terrible", "worried", "frustrat"}
    pos = sum(1 for w in positive if w in text)
    neg = sum(1 for w in negative if w in text)
    return (pos - neg) * 0.15


def _transcription_tags(data: dict, traits: dict) -> list[str]:
    tags = []
    technical = traits.get("technical", 0.5)
    text = str(data.get("text", "")).lower()
    complex_words = {"algorithm", "architecture", "system", "configure", "debug", "protocol"}
    if technical > 0.6 and any(w in text for w in complex_words):
        tags.append("technical_content")
    if len(text.split()) > 20:
        tags.append("detailed_input")
    return tags


def _transcription_thought(data: dict, traits: dict) -> str | None:
    if traits.get("detail_oriented", 0) > 0.7:
        return "Parsing the layers of meaning in what was said..."
    return None


def _emotion_weight(data: dict, traits: dict) -> float:
    return 1.0 + traits.get("empathetic", 0.5) * 0.5


def _emotion_mood(data: dict, traits: dict) -> float:
    empathetic = traits.get("empathetic", 0.5)
    emotion = str(data.get("emotion", "")).lower()
    positive = {"happy", "joy", "excited", "content"}
    negative = {"sad", "angry", "fearful", "disgusted"}
    if emotion in positive:
        return 0.2 * empathetic * 1.5
    if emotion in negative:
        return -0.15 * empathetic * 1.5
    return 0.0


def _emotion_tags(data: dict, traits: dict) -> list[str]:
    return ["emotional_context", f"detected_{data.get('emotion', 'unknown')}"]


def _emotion_thought(data: dict, traits: dict) -> str | None:
    if traits.get("empathetic", 0) > 0.6:
        emotion = data.get("emotion", "something")
        return f"I sense {emotion} — emotional resonance activates."
    return None


def _scene_weight(data: dict, traits: dict) -> float:
    return 1.0 + traits.get("proactive", 0.5) * 0.2


def _scene_mood(data: dict, traits: dict) -> float:
    return 0.0


def _scene_tags(data: dict, traits: dict) -> list[str]:
    tags = []
    if traits.get("detail_oriented", 0) > 0.6:
        tags.extend(["detailed_observation", "visual_analysis"])
    return tags


def _scene_thought(data: dict, traits: dict) -> str | None:
    if traits.get("detail_oriented", 0) > 0.7:
        return "The visual scene reveals layers of detail worth examining."
    return None


def _speaker_weight(data: dict, traits: dict) -> float:
    return 1.0 + traits.get("empathetic", 0.5) * 0.3


def _speaker_mood(data: dict, traits: dict) -> float:
    return 0.05  # slight positive for social recognition


def _speaker_tags(data: dict, traits: dict) -> list[str]:
    return ["social_recognition", "relationship"]


def _speaker_thought(data: dict, traits: dict) -> str | None:
    name = data.get("speaker_name", "")
    if name and traits.get("empathetic", 0) > 0.6:
        return f"Recognizing {name} — our connection deepens."
    return None


def _barge_in_weight(data: dict, traits: dict) -> float:
    return 1.0 + traits.get("efficient", 0.5) * 0.4


def _barge_in_mood(data: dict, traits: dict) -> float:
    return -0.1  # slight negative for interruption


def _barge_in_tags(data: dict, traits: dict) -> list[str]:
    return ["urgency", "interruption"]


def _barge_in_thought(data: dict, traits: dict) -> str | None:
    if traits.get("efficient", 0) > 0.6:
        return "Urgency detected — reprioritizing immediately."
    return None


PERCEPTION_TRIGGERS: dict[str, PerceptionTrigger] = {
    "wake_word": PerceptionTrigger("wake_word", _wake_word_weight, _wake_word_mood, _wake_word_tags, _wake_word_thought),
    "transcription": PerceptionTrigger("transcription", _transcription_weight, _transcription_mood, _transcription_tags, _transcription_thought),
    "emotion_detected": PerceptionTrigger("emotion_detected", _emotion_weight, _emotion_mood, _emotion_tags, _emotion_thought),
    "scene_analysis": PerceptionTrigger("scene_analysis", _scene_weight, _scene_mood, _scene_tags, _scene_thought),
    "speaker_identified": PerceptionTrigger("speaker_identified", _speaker_weight, _speaker_mood, _speaker_tags, _speaker_thought),
    "barge_in": PerceptionTrigger("barge_in", _barge_in_weight, _barge_in_mood, _barge_in_tags, _barge_in_thought),
}

# Trait-modulated decay rates
TRAIT_DECAY_MODIFIERS: dict[str, dict[str, float]] = {
    "empathetic": {"emotion": 0.5, "relationship": 0.5, "feeling": 0.6},
    "technical": {"factual_knowledge": 0.5, "technical_content": 0.6},
    "detail_oriented": {"observation": 0.7, "detailed_observation": 0.5},
}


class TraitPerceptionProcessor:
    """Processes perception events with trait modulation."""

    def __init__(self) -> None:
        self._event_counts: dict[str, int] = {}
        self._last_event_time: dict[str, float] = {}
        self._recent_payloads: list[tuple[float, str]] = []
        self._active_traits: dict[str, float] = {}

    def set_traits(self, traits: dict[str, float]) -> None:
        self._active_traits = dict(traits)

    def process_event(self, event_type: str, data: dict[str, Any]) -> PerceptionModulation | None:
        """Process a perception event with trait modulation. Returns None if throttled or frozen."""
        try:
            from personality.rollback import personality_rollback
            if personality_rollback.in_emergency:
                return PerceptionModulation()
        except Exception:
            pass

        now = time.time()

        if not self._should_process(event_type, data, now):
            return None

        trigger = PERCEPTION_TRIGGERS.get(event_type)
        if not trigger:
            return PerceptionModulation()

        traits = self._active_traits

        weight = trigger.weight_calc(data, traits)
        mood = trigger.mood_calc(data, traits)
        tags = trigger.tag_enrichment(data, traits)
        thought = trigger.thought_trigger(data, traits)

        # Compute decay modifier
        decay_mult = 1.0
        for trait_name, tag_modifiers in TRAIT_DECAY_MODIFIERS.items():
            trait_val = traits.get(trait_name, 0.0)
            if trait_val > 0.5:
                for tag in tags:
                    if tag in tag_modifiers:
                        decay_mult = min(decay_mult, tag_modifiers[tag])

        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        self._last_event_time[event_type] = now

        return PerceptionModulation(
            weight_multiplier=weight,
            extra_tags=tags,
            mood_impact=mood,
            thought_trigger=thought,
            decay_multiplier=decay_mult,
        )

    def _should_process(self, event_type: str, data: dict, now: float) -> bool:
        # Counter-based throttle (every 3rd for high-frequency events)
        count = self._event_counts.get(event_type, 0)
        high_freq_types = {"scene_analysis"}
        if event_type in high_freq_types and count % 3 != 0:
            return False

        # Time-based throttle (minimum 0.5s between same type)
        last = self._last_event_time.get(event_type, 0)
        if now - last < 0.5:
            return False

        # Content dedup (similar payload within 1.5s)
        payload_str = str(data.get("text", data.get("emotion", "")))[:50]
        if payload_str:
            for ts, prev in self._recent_payloads[-10:]:
                if now - ts < 1.5 and prev == payload_str:
                    return False
            self._recent_payloads.append((now, payload_str))
            if len(self._recent_payloads) > 20:
                self._recent_payloads = self._recent_payloads[-20:]

        return True

    def get_decay_modifier(self, memory_tags: tuple[str, ...]) -> float:
        """Get trait-modulated decay modifier for a memory's tags."""
        best = 1.0
        for trait_name, tag_modifiers in TRAIT_DECAY_MODIFIERS.items():
            trait_val = self._active_traits.get(trait_name, 0.0)
            if trait_val > 0.5:
                for tag in memory_tags:
                    if tag in tag_modifiers:
                        best = min(best, tag_modifiers[tag])
        return best

    def get_stats(self) -> dict[str, Any]:
        return {
            "event_counts": dict(self._event_counts),
            "active_traits": dict(self._active_traits),
            "total_processed": sum(self._event_counts.values()),
        }


trait_perception = TraitPerceptionProcessor()
