import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reasoning.voice_policy import OracleVoicePolicy, VoicePolicyConfig, VoicePolicyContext


def _policy() -> OracleVoicePolicy:
    return OracleVoicePolicy(VoicePolicyConfig(
        base_voice="af_bella",
        base_speed=1.0,
        solemn_voice="af_sarah",
        empathetic_voice="af_bella",
        urgent_voice="af_nicole",
        observational_voice="af_bella",
        guarded_voice="af_sarah",
    ))


def test_professional_defaults_to_solemn_oracle_profile():
    profile = _policy().resolve(VoicePolicyContext(tone="professional"))
    assert profile.profile_id == "oracle_solemn"
    assert profile.voice == "af_sarah"


def test_empathetic_tone_selects_empathetic_profile():
    profile = _policy().resolve(VoicePolicyContext(
        tone="empathetic",
        user_emotion="sad",
        emotion_trusted=True,
    ))
    assert profile.profile_id == "oracle_empathetic"
    assert profile.speed < 0.9


def test_urgent_tone_selects_urgent_profile():
    profile = _policy().resolve(VoicePolicyContext(tone="urgent"))
    assert profile.profile_id == "oracle_urgent"
    assert profile.voice == "af_nicole"
    assert profile.speed >= 0.92


def test_style_override_forces_solemn_profile():
    profile = _policy().resolve(VoicePolicyContext(
        tone="playful",
        style_override="oracle_solemn",
    ))
    assert profile.profile_id == "oracle_solemn"
    assert profile.voice == "af_sarah"
