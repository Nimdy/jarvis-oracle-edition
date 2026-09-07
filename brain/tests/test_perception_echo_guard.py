import re
from pathlib import Path


_SRC_PATH = Path(__file__).resolve().parent.parent / "perception_orchestrator.py"


def _source() -> str:
    return _SRC_PATH.read_text(encoding="utf-8")


def _echo_helpers():
    src = _source()
    start = src.find("_ECHO_SENTENCE_SPLIT_RE")
    end = src.find("\nclass PerceptionOrchestrator")
    assert start != -1 and end != -1 and end > start
    ns: dict = {"re": re}
    exec(src[start:end], ns)
    return ns


def _function_body(src: str, fn_name: str) -> str:
    marker = f"def {fn_name}("
    start = src.find(marker)
    assert start != -1, f"Missing function {fn_name}"
    next_fn = src.find("\n    def ", start + 1)
    if next_fn == -1:
        next_fn = len(src)
    return src[start:next_fn]


def test_on_conversation_response_resets_playback_marker() -> None:
    src = _source()
    body = _function_body(src, "_on_conversation_response")
    assert "_playback_complete_time = 0.0" in body


def test_speak_proactive_resets_playback_marker() -> None:
    src = _source()
    body = _function_body(src, "_speak_proactive")
    assert "_playback_complete_time = 0.0" in body


def test_echo_expiry_prefers_fresh_response_timestamp() -> None:
    src = _source()
    body = _function_body(src, "_is_echo_expired")
    assert "_last_response_set_time > self._playback_complete_time" in body
    assert "return (now - self._last_response_set_time) > self._ECHO_STALE_S" in body


def test_echo_spans_include_last_sentence() -> None:
    spans = _echo_helpers()["echo_reference_spans"](
        "Here's what I remember about that. David is software engineer."
    )
    assert spans[0].startswith("here's what i remember")
    assert spans[-1] == "david is software engineer"


def test_tts_tail_engineers_matches_last_sentence() -> None:
    """Lived: follow-up STT 'we're engineers.' was TTS of 'software engineer'."""
    reply = "Here's what I remember about that. David is software engineer."
    heard = "we're engineers."
    sim = _echo_helpers()["echo_best_similarity"](heard, reply)
    assert sim >= 0.50, sim


def test_unknown_follow_up_is_identity_floor() -> None:
    src = _source()
    body = _function_body(src, "_is_speaker_echo")
    assert "was_follow_up" in body
    assert "unknown_follow_up" in body
    assert "echo_best_similarity" in body
    assert "echo_reference_spans" in _source()


def test_unknown_speaker_thought_no_longer_asks_who() -> None:
    src = _source()
    assert "I wonder who this is." not in src
    assert "Unrecognized speaker signal detected" in src


def test_terminal_response_helper_includes_lineage_fields() -> None:
    src = _source()
    body = _function_body(src, "_emit_conversation_response")
    assert "event_bus.emit(CONVERSATION_RESPONSE" in body
    assert "conversation_id" in body
    assert "trace_id" in body
    assert "request_id" in body
    assert "output_id" in body
    assert "validation_id" in body
    assert "validation_passed" in body
    assert "release_status" in body


def test_perception_terminal_emits_use_lineage_helper() -> None:
    src = _source()
    assert "event_bus.emit(CONVERSATION_RESPONSE, text=text)" not in src
    assert "event_bus.emit(CONVERSATION_RESPONSE, text=dismiss_msg[\"text\"])" not in src
    assert "event_bus.emit(CONVERSATION_RESPONSE, text=\"\")" not in src
