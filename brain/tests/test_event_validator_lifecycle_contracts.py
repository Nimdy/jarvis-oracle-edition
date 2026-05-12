"""Contract tests for response lifecycle and retry validation rules."""

from __future__ import annotations

from consciousness.event_validator import EventSequenceValidator
from consciousness.events import (
    CONVERSATION_RESPONSE,
    CONVERSATION_USER_MESSAGE,
    PERCEPTION_PLAYBACK_COMPLETE,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_WAKE_WORD,
    RETRY_EXECUTED,
    RETRY_EXHAUSTED,
    RETRY_SCHEDULED,
)


def test_user_message_requires_recent_transcription() -> None:
    validator = EventSequenceValidator()
    validator.validate(CONVERSATION_USER_MESSAGE, {"conversation_id": "conv_missing"})

    recent = validator.get_recent_violations(limit=5)
    assert any(v["rule_id"] == "user_message_follows_transcription" for v in recent)


def test_lifecycle_transcription_response_playback_passes() -> None:
    validator = EventSequenceValidator()
    payload = {"conversation_id": "conv_ok"}

    validator.record_event(PERCEPTION_WAKE_WORD)
    validator.validate(PERCEPTION_TRANSCRIPTION, payload)
    validator.validate(CONVERSATION_USER_MESSAGE, payload)
    validator.validate(CONVERSATION_RESPONSE, {**payload, "output_id": "out_ok"})
    validator.validate(PERCEPTION_PLAYBACK_COMPLETE, payload)

    assert validator.get_stats()["total_violations"] == 0


def test_known_fallback_response_reason_is_allowed_without_input() -> None:
    validator = EventSequenceValidator()
    validator.validate(
        CONVERSATION_RESPONSE,
        {
            "conversation_id": "conv_fallback",
            "output_id": "out_fallback",
            "release_reason": "handle_transcription_timeout",
        },
    )
    assert validator.get_stats()["total_violations"] == 0


def test_retry_terminal_event_without_schedule_is_flagged() -> None:
    validator = EventSequenceValidator()
    validator.validate(
        RETRY_EXECUTED,
        {
            "retry_id": "rty_missing",
            "target_event_type": "conversation:response",
            "attempt": 1,
        },
    )

    recent = validator.get_recent_violations(limit=8)
    assert any(v["rule_id"] == "retry_terminal_requires_schedule" for v in recent)


def test_retry_schedule_execute_exhaust_chain_passes() -> None:
    validator = EventSequenceValidator()
    retry_payload = {
        "retry_id": "rty_ok",
        "target_event_type": "conversation:response",
        "attempt": 1,
    }

    validator.validate(RETRY_SCHEDULED, retry_payload)
    validator.validate(RETRY_EXECUTED, retry_payload)
    validator.validate(RETRY_EXHAUSTED, retry_payload)

    assert validator.get_stats()["total_violations"] == 0
