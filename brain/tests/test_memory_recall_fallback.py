from pathlib import Path


_SRC_PATH = Path(__file__).resolve().parent.parent / "conversation_handler.py"


def _source() -> str:
    return _SRC_PATH.read_text(encoding="utf-8")


def _function_body(src: str, fn_name: str) -> str:
    marker = f"def {fn_name}("
    start = src.find(marker)
    assert start != -1, f"Missing function {fn_name}"
    next_fn = src.find("\ndef ", start + 1)
    next_async_fn = src.find("\nasync def ", start + 1)
    candidates = [x for x in (next_fn, next_async_fn) if x != -1]
    end = min(candidates) if candidates else len(src)
    return src[start:end]


def test_none_route_has_personal_activity_memory_fallback() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "_is_personal_activity_recall_query(text)" in body
    assert "\"provenance_verdict\": \"none_route_memory_recall_native\"" in body
    assert "\"none_route_memory_recall_fallback\"" in body
    assert "NONE route: personal activity recall native fallback applied" in body


def test_memory_route_uses_search_for_personal_activity_queries() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "_should_use_memory_search(text, extracted_args=routing.extracted_args)" in body
    assert "deterministic_personal_activity_recall" in body
    assert "_is_personal_activity_recall_query(text)" in body
    assert "_format_personal_activity_memory_reply(memory_ctx)" in body
    assert "_format_personal_activity_memory_reply(_memory_ctx)" in body


def test_personal_activity_regex_covers_tell_me_what_i_did() -> None:
    src = _source()
    assert "tell\\s+me\\s+what\\s+(?:i|we)\\s+did" in src
    assert "(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)" in src


def test_personal_activity_formatter_helpers_exist() -> None:
    src = _source()
    assert "def _normalize_memory_preview(" in src
    assert "def _to_speakable_memory_sentence(" in src
    assert "def _format_personal_activity_memory_reply(" in src


def test_personal_activity_formatter_uses_conversational_memory_voice() -> None:
    src = _source()
    body = _function_body(src, "_format_personal_activity_memory_reply")
    assert "Here's what I remember from that time." in body
    assert "Most relevant" not in body
    assert "_memory_priority(" in body
    assert "_to_speakable_memory_sentence(" in body


def test_none_route_fragment_noise_guard_present() -> None:
    src = _source()
    assert "_LIKELY_STT_FRAGMENT_RE" in src
    assert "def _is_likely_fragment_noise(" in src
    body = _function_body(src, "handle_transcription")
    assert "if _is_likely_fragment_noise(text, follow_up):" in body
    assert "fragment-noise clarification reply applied" in body


def test_negative_identity_fact_guard_present() -> None:
    src = _source()
    guard_body = _function_body(src, "_is_unstable_personal_fact")
    collect_body = _function_body(src, "_collect_personal_intel_matches")
    assert "category != \"personal_fact\"" in guard_body
    assert "startswith(\"user is not \")" in guard_body
    assert "if _is_unstable_personal_fact(payload, category):" in collect_body

