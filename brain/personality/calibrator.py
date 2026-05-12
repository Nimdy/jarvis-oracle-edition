"""Tone calibrator — context-driven tone recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from consciousness.events import JarvisTone


@dataclass(frozen=True)
class ToneCalibration:
    recommended_tone: JarvisTone
    confidence: float
    factors: tuple[str, ...]


class ToneCalibrator:
    _instance: ToneCalibrator | None = None

    def __init__(self) -> None:
        self._preferences: dict[str, list[dict]] = {}

    @classmethod
    def get_instance(cls) -> ToneCalibrator:
        if cls._instance is None:
            cls._instance = ToneCalibrator()
        return cls._instance

    def calibrate(
        self,
        current_tone: JarvisTone,
        time_of_day: int,
        traits: list[str],
        is_in_meeting: bool,
        user_emotion: str | None = None,
    ) -> ToneCalibration:
        factors: list[str] = []
        recommended: JarvisTone = current_tone
        confidence = 0.3

        if is_in_meeting:
            recommended = "professional"
            confidence = 0.8
            factors.append("in_meeting")

        if user_emotion:
            emotion_tone = self._emotion_to_tone(user_emotion)
            if emotion_tone:
                recommended = emotion_tone
                confidence = max(confidence, 0.7)
                factors.append(f"user_emotion:{user_emotion}")

        hour_bias = self._get_time_bias(time_of_day)
        if hour_bias and confidence < 0.6:
            recommended = hour_bias
            factors.append(f"time_of_day:{time_of_day}h")

        interaction_pref = self._get_preferred_tone("general")
        if interaction_pref and confidence < 0.5:
            recommended = interaction_pref
            confidence = 0.5
            factors.append("learned_preference")

        if "Humor-Adaptive" in traits and confidence < 0.6:
            if recommended not in ("urgent", "empathetic"):
                recommended = "casual"
                factors.append("humor_trait")

        if "Efficient" in traits and confidence < 0.5:
            recommended = "professional"
            factors.append("efficiency_trait")

        return ToneCalibration(recommended, confidence, tuple(factors))

    def record_interaction_outcome(self, context: str, tone: JarvisTone, was_positive: bool) -> None:
        prefs = self._preferences.setdefault(context, [])
        for p in prefs:
            if p["tone"] == tone:
                p["total_count"] += 1
                if was_positive:
                    p["success_count"] += 1
                return
        prefs.append({"tone": tone, "success_count": int(was_positive), "total_count": 1})

    def _get_preferred_tone(self, context: str) -> JarvisTone | None:
        prefs = self._preferences.get(context)
        if not prefs:
            return None
        best = max(
            prefs,
            key=lambda p: p["success_count"] / p["total_count"] if p["total_count"] > 0 else 0,
        )
        return best["tone"] if best["total_count"] >= 3 else None

    @staticmethod
    def _emotion_to_tone(emotion: str) -> JarvisTone | None:
        mapping: dict[str, JarvisTone] = {
            "happy": "playful", "sad": "empathetic", "angry": "empathetic",
            "stressed": "empathetic", "frustrated": "empathetic", "excited": "casual",
            "neutral": "professional", "tired": "casual",
        }
        return mapping.get(emotion.lower())

    @staticmethod
    def _get_time_bias(hour: int) -> JarvisTone | None:
        if 6 <= hour < 9:
            return "casual"
        if 9 <= hour < 17:
            return "professional"
        if 17 <= hour < 21:
            return "casual"
        return "casual"


tone_calibrator = ToneCalibrator.get_instance()
