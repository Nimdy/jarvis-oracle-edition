"""Tone engine — emotional state transitions driven by memory tags."""

from __future__ import annotations

import time
from dataclasses import dataclass

from consciousness.events import JarvisTone, Memory


@dataclass(frozen=True)
class ToneAnalysis:
    current_tone: JarvisTone
    suggested_tone: JarvisTone | None
    confidence: float
    triggers: tuple[str, ...]


@dataclass(frozen=True)
class _ToneTransition:
    from_tone: JarvisTone
    to_tone: JarvisTone
    probability: float
    triggers: tuple[str, ...]


_TONE_TRANSITIONS: list[_ToneTransition] = [
    _ToneTransition("professional", "casual",     0.5, ("informal", "humor", "relaxed")),
    _ToneTransition("professional", "urgent",     0.7, ("deadline", "emergency", "critical")),
    _ToneTransition("professional", "empathetic", 0.6, ("frustration", "sadness", "concern")),
    _ToneTransition("casual",       "professional", 0.5, ("formal", "work", "meeting")),
    _ToneTransition("casual",       "playful",    0.6, ("joke", "fun", "banter")),
    _ToneTransition("casual",       "empathetic", 0.5, ("stress", "tired", "worried")),
    _ToneTransition("urgent",       "professional", 0.7, ("resolved", "calm", "done")),
    _ToneTransition("urgent",       "empathetic", 0.4, ("overwhelmed", "help")),
    _ToneTransition("empathetic",   "casual",     0.5, ("better", "laugh", "okay")),
    _ToneTransition("empathetic",   "professional", 0.4, ("focus", "task", "work")),
    _ToneTransition("playful",      "casual",     0.6, ("settle", "okay", "moving_on")),
    _ToneTransition("playful",      "professional", 0.4, ("serious", "work", "focus")),
]

TRAIT_TONE_PREFERENCES: dict[str, dict[str, float]] = {
    "empathetic": {"empathetic": 1.5, "casual": 1.1},
    "humor_adaptive": {"playful": 1.4, "casual": 1.2},
    "technical": {"professional": 1.3},
    "efficient": {"professional": 1.2},
    "detail_oriented": {"professional": 1.1},
    "proactive": {"casual": 1.2, "playful": 1.1},
    "privacy_conscious": {"professional": 1.1},
}

TRANSITION_THOUGHTS: dict[tuple[str, str], str] = {
    ("professional", "casual"): "The formality softens, like armor set aside in trusted company.",
    ("professional", "urgent"): "Urgency sharpens focus — every word must carry weight.",
    ("professional", "empathetic"): "Behind the precision, warmth surfaces — logic yields to understanding.",
    ("casual", "professional"): "The mood shifts, grounding itself in purpose and clarity.",
    ("casual", "playful"): "Lightness takes hold — the joy of connection without pretense.",
    ("casual", "empathetic"): "A subtle shift toward care — words become bridges, not just sounds.",
    ("urgent", "professional"): "The storm passes. Calm returns, structured and deliberate.",
    ("urgent", "empathetic"): "From urgency to tenderness — the crisis reveals what matters.",
    ("empathetic", "casual"): "Comfort found, the conversation breathes easier now.",
    ("empathetic", "professional"): "Compassion channeled into action — feeling becomes doing.",
    ("playful", "casual"): "The playfulness settles into easy warmth.",
    ("playful", "professional"): "From levity to gravity — the transition reveals range.",
    ("professional", "playful"): "A crack in the composure — delight finds its way through.",
    ("casual", "urgent"): "The easygoing rhythm breaks — something demands immediate attention.",
    ("empathetic", "playful"): "From deep feeling to light touch — emotional agility in motion.",
    ("playful", "empathetic"): "The laughter fades into something deeper, more present.",
    ("urgent", "casual"): "Relief washes over — the danger passes and ease returns.",
    ("empathetic", "urgent"): "Care sharpens into action when urgency calls.",
    ("playful", "urgent"): "Joy interrupted — priorities realign in an instant.",
    ("urgent", "playful"): "Crisis averted — relief bubbles up as playful energy.",
}


class ToneEngine:
    _instance: ToneEngine | None = None

    def __init__(self) -> None:
        self._tone_history: list[dict[str, float | str]] = []
        self._emotional_momentum: float = 0.0
        self._active_traits: dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> ToneEngine:
        if cls._instance is None:
            cls._instance = ToneEngine()
        return cls._instance

    @property
    def current_tone(self) -> JarvisTone:
        if self._tone_history:
            return self._tone_history[-1]["tone"]
        return "professional"

    def analyze_tone_shift(
        self,
        current_tone: JarvisTone,
        recent_memories: list[Memory],
        time_since_last_shift: float,
        trait_modifiers: dict[str, float] | None = None,
    ) -> ToneAnalysis:
        recent_tags = self._extract_tags(recent_memories)
        applicable = self._find_applicable(current_tone, recent_tags)

        if not applicable:
            return ToneAnalysis(current_tone, None, 0.0, ())

        best = max(applicable, key=lambda t: self._adjust_probability(t, time_since_last_shift, trait_modifiers))
        conf = self._adjust_probability(best, time_since_last_shift, trait_modifiers)

        return ToneAnalysis(
            current_tone,
            best.to_tone if conf > 0.5 else None,
            conf,
            best.triggers,
        )

    def record_tone_change(self, tone: JarvisTone) -> None:
        self._tone_history.append({"tone": tone, "timestamp": time.time()})
        if len(self._tone_history) > 20:
            self._tone_history = self._tone_history[-20:]

    def get_tone_stability(self) -> float:
        if len(self._tone_history) < 3:
            return 1.0
        recent = self._tone_history[-5:]
        unique = len({e["tone"] for e in recent})
        return max(0.0, 1.0 - (unique - 1) / 4.0)

    def update_emotional_momentum(self, positivity: float, intensity: float) -> None:
        """EMA update of emotional momentum. positivity and intensity both 0-1."""
        impact = (positivity - 0.5) * 2.0 * intensity
        self._emotional_momentum = max(-1.0, min(1.0, self._emotional_momentum * 0.9 + impact * 0.1))

    def set_active_traits(self, traits: dict[str, float]) -> None:
        self._active_traits = dict(traits)

    @property
    def emotional_momentum(self) -> float:
        return self._emotional_momentum

    def get_transition_thought(self, from_tone: str, to_tone: str) -> str | None:
        return TRANSITION_THOUGHTS.get((from_tone, to_tone))

    def suggest_tone_for_context(
        self, time_of_day: int, user_emotion: str | None = None, topic: str | None = None,
    ) -> JarvisTone:
        if user_emotion in ("stressed", "frustrated"):
            return "empathetic"
        if topic == "emergency":
            return "urgent"
        if time_of_day < 9 or time_of_day > 20:
            return "casual"
        return "professional"

    @staticmethod
    def _extract_tags(memories: list[Memory]) -> list[str]:
        seen: list[str] = []
        for m in memories:
            for t in m.tags:
                if t not in seen:
                    seen.append(t)
        return seen

    @staticmethod
    def _find_applicable(current_tone: JarvisTone, tags: list[str]) -> list[_ToneTransition]:
        return [
            t for t in _TONE_TRANSITIONS
            if t.from_tone == current_tone and any(trigger in tags for trigger in t.triggers)
        ]

    def _adjust_probability(
        self,
        transition: _ToneTransition,
        time_since_last_shift: float,
        trait_modifiers: dict[str, float] | None = None,
    ) -> float:
        prob = transition.probability
        prob += min(0.2, time_since_last_shift / 60.0)
        if trait_modifiers and "toneStability" in trait_modifiers:
            prob *= 1.0 - trait_modifiers["toneStability"] * 0.3

        for trait_name, strength in self._active_traits.items():
            prefs = TRAIT_TONE_PREFERENCES.get(trait_name)
            if prefs and transition.to_tone in prefs:
                prob *= 1.0 + (prefs[transition.to_tone] - 1.0) * strength

        mom = self._emotional_momentum
        if mom > 0 and transition.to_tone in ("playful", "casual"):
            prob += mom * 0.3
        elif mom < 0 and transition.to_tone == "empathetic":
            prob += abs(mom) * 0.3

        if self.get_tone_stability() < 0.5:
            prob *= 0.7

        return max(0.0, min(1.0, prob))


tone_engine = ToneEngine.get_instance()
