from pathlib import Path


_SRC_PATH = Path(__file__).resolve().parent.parent / "conversation_handler.py"


def _source() -> str:
    return _SRC_PATH.read_text(encoding="utf-8")


def test_conversation_response_emit_has_lineage_fields() -> None:
    src = _source()
    marker = "event_bus.emit(\n        CONVERSATION_RESPONSE,"
    start = src.find(marker)
    assert start != -1, "Missing CONVERSATION_RESPONSE emission block"
    block = src[start:start + 700]
    assert "conversation_id=_trace_ctx.conversation_id" in block
    assert "trace_id=_trace_ctx.trace_id" in block
    assert "request_id=_trace_ctx.request_id" in block
    assert "output_id=_output_id" in block
    assert "validation_id=_validation_decision.validation_id" in block
    assert "validation_passed=_validation_decision.passed" in block
    assert "release_status=_validation_decision.effective_release_status" in block


def test_conversation_user_emit_shares_trace_context_fields() -> None:
    src = _source()
    marker = "event_bus.emit(\n        CONVERSATION_USER_MESSAGE,"
    start = src.find(marker)
    assert start != -1, "Missing CONVERSATION_USER_MESSAGE emission block"
    block = src[start:start + 700]
    assert "conversation_id=_trace_ctx.conversation_id" in block
    assert "trace_id=_trace_ctx.trace_id" in block
    assert "request_id=_trace_ctx.request_id" in block


def test_output_id_is_persisted_in_response_ledger_data() -> None:
    src = _source()
    assert "\"output_id\": _output_id" in src
    assert "\"trace_id\": _trace_ctx.trace_id" in src
    assert "\"request_id\": _trace_ctx.request_id" in src
    assert "\"validation_id\": _validation_decision.validation_id" in src
    assert "\"validation_passed\": _validation_decision.passed" in src


def test_episode_and_flight_recorder_include_ledger_linkage_fields() -> None:
    src = _source()
    assert "episodes.add_user_turn(" in src
    assert "conversation_entry_id=_conv_ledger_id" in src
    assert "episodes.add_assistant_turn(" in src
    assert "response_entry_id=_response_ledger_id" in src
    assert "\"conversation_entry_id\": _conv_ledger_id" in src
    assert "\"response_entry_id\": _response_ledger_id" in src
    assert "\"root_entry_id\": _conv_ledger_id or _response_ledger_id" in src
