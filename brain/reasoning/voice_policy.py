"""Oracle voice policy — map tone/emotion context to TTS delivery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoicePolicyConfig:
    base_voice: str
    base_speed: float = 1.0
    solemn_voice: str = ""
    empathetic_voice: str = ""
    urgent_voice: str = ""
    observational_voice: str = ""
    guarded_voice: str = ""


@dataclass(frozen=True)
class VoicePolicyContext:
    tone: str = "professional"
    user_emotion: str = "neutral"
    emotion_trusted: bool = False
    speaker_name: str = ""
    proactive: bool = False
    style_override: str = ""


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    voice: str
    speed: float
    max_chars: int
    min_chunk_chars: int


def _clamp_speed(value: float, low: float = 0.78, high: float = 1.05) -> float:
    return max(low, min(high, value))


class OracleVoicePolicy:
    """Resolve a stable Oracle persona into contextual delivery variants."""

    def __init__(self, cfg: VoicePolicyConfig) -> None:
        self._cfg = cfg

    def resolve(self, ctx: VoicePolicyContext) -> VoiceProfile:
        tone = (ctx.tone or "professional").lower()
        emotion = (ctx.user_emotion or "neutral").lower()
        trusted = bool(ctx.emotion_trusted)
        base_speed = float(self._cfg.base_speed or 1.0)
        style_override = (ctx.style_override or "").lower()

        if style_override == "oracle_urgent":
            return VoiceProfile(
                profile_id="oracle_urgent",
                voice=self._cfg.urgent_voice or self._cfg.base_voice,
                speed=_clamp_speed(max(base_speed, 0.92), 0.84, 1.08),
                max_chars=260,
                min_chunk_chars=36,
            )
        if style_override == "oracle_guarded":
            return VoiceProfile(
                profile_id="oracle_guarded",
                voice=self._cfg.guarded_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.87)),
                max_chars=230,
                min_chunk_chars=36,
            )
        if style_override == "oracle_empathetic":
            return VoiceProfile(
                profile_id="oracle_empathetic",
                voice=self._cfg.empathetic_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.84)),
                max_chars=210,
                min_chunk_chars=34,
            )
        if style_override == "oracle_observational":
            return VoiceProfile(
                profile_id="oracle_observational",
                voice=self._cfg.observational_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.9), 0.82, 1.0),
                max_chars=240,
                min_chunk_chars=38,
            )
        if style_override == "oracle_solemn":
            return VoiceProfile(
                profile_id="oracle_solemn",
                voice=self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.88)),
                max_chars=220,
                min_chunk_chars=40,
            )

        if tone == "urgent":
            return VoiceProfile(
                profile_id="oracle_urgent",
                voice=self._cfg.urgent_voice or self._cfg.base_voice,
                speed=_clamp_speed(max(base_speed, 0.92), 0.84, 1.08),
                max_chars=260,
                min_chunk_chars=36,
            )

        if tone == "empathetic" or (trusted and emotion in {"sad", "worried", "stressed", "tired", "overwhelmed"}):
            return VoiceProfile(
                profile_id="oracle_empathetic",
                voice=self._cfg.empathetic_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.84)),
                max_chars=210,
                min_chunk_chars=34,
            )

        if trusted and emotion in {"angry", "frustrated", "disgust", "fearful"}:
            return VoiceProfile(
                profile_id="oracle_guarded",
                voice=self._cfg.guarded_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.87)),
                max_chars=230,
                min_chunk_chars=36,
            )

        if tone in {"casual", "playful"} or ctx.proactive:
            return VoiceProfile(
                profile_id="oracle_observational",
                voice=self._cfg.observational_voice or self._cfg.solemn_voice or self._cfg.base_voice,
                speed=_clamp_speed(min(base_speed, 0.9), 0.82, 1.0),
                max_chars=240,
                min_chunk_chars=38,
            )

        return VoiceProfile(
            profile_id="oracle_solemn",
            voice=self._cfg.solemn_voice or self._cfg.base_voice,
            speed=_clamp_speed(min(base_speed, 0.88)),
            max_chars=220,
            min_chunk_chars=40,
        )
