import ast
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


def test_capability_status_query_detector_includes_verified_phrase() -> None:
    src = _source()
    body = _function_body(src, "_is_capability_status_query")
    assert "_CAPABILITY_STATUS_QUERY_RE" in src
    assert "verified right now" in body
    assert "currently|actually|right now|at the moment" in body
    assert "what (?:are|'re) your current (?:skills|abilities)" in src


def test_recent_research_query_detector_exists_and_covers_research_terms() -> None:
    src = _source()
    body = _function_body(src, "_is_recent_research_query")
    assert "_RECENT_RESEARCH_QUERY_RE" in src
    assert "research" in body


def test_none_route_has_native_capability_fallback() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "if _is_capability_status_query(text):" in body
    assert "response_class=\"capability_status\"" in body
    assert "\"provenance_verdict\": \"none_route_capability_status_native\"" in body
    assert "\"none_route_capability_fallback\"" in body
    assert "NONE route: capability_status native fallback applied" in body


def test_none_route_default_general_chat_has_classified_provenance() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "\"response_class\": \"general_conversation\"" in body
    assert "\"provenance_verdict\": \"none_route_general_conversation\"" in body
    assert "\"llm_articulation_only\"" in body
    assert "\"commitment_gate\"" in body


def test_introspection_route_has_native_capability_fallback() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "elif _is_capability_status_query(introspection_query):" in body
    assert "_build_none_route_capability_payload()" in body
    assert "\"provenance_verdict\": \"introspection_capability_status_native\"" in body
    assert "Introspection capability-status native answer used" in body


def test_build_none_route_capability_payload_calls_use_zero_arguments() -> None:
    src = _source()
    tree = ast.parse(src)

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "_build_none_route_capability_payload":
            calls.append(node)

    assert calls, "Expected at least one call to _build_none_route_capability_payload"
    for call in calls:
        assert len(call.args) == 0, "Call should not pass positional args"
        assert len(call.keywords) == 0, "Call should not pass keyword args"


def test_introspection_recent_learning_branch_relabels_research_queries() -> None:
    src = _source()
    body = _function_body(src, "handle_transcription")
    assert "elif strict_recent_learning_reply:" in body
    assert "_is_recent_research_query(introspection_query)" in body
    assert "\"missing_research\"" in body
    assert "\"missing_scholarly\"" in body
    assert "_response_class = \"recent_research\"" in body


def test_recent_learning_records_include_completed_learning_jobs() -> None:
    src = (Path(__file__).resolve().parent.parent / "tools" / "introspection_tool.py").read_text(encoding="utf-8")
    body = _function_body(src, "get_grounded_recent_learning_record")
    answer_body = _function_body(src, "_build_recent_learning_answer")
    assert "from skills.learning_jobs import LearningJobStore" in body
    assert "status not in {\"completed\", \"verified\"}" in body
    assert "\"kind\": \"learning_job\"" in body
    assert "skill_learning_report" in body
    assert "skill" in src and "finish" in src and "_RECENT_ACTIVITY_QUERY_RE" in src
    assert "latest completed skill-learning record" in answer_body
