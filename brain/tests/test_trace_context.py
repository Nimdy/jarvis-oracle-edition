from consciousness.trace_context import build_trace_context


def test_build_trace_context_is_stable_for_same_conversation_id() -> None:
    first = build_trace_context("conv_abc123")
    second = build_trace_context("conv_abc123")
    assert first.conversation_id == "conv_abc123"
    assert second.conversation_id == "conv_abc123"
    assert first.trace_id == second.trace_id
    assert first.request_id.startswith("req_")
    assert second.request_id.startswith("req_")


def test_build_trace_context_generates_non_empty_fallback_context() -> None:
    ctx = build_trace_context("")
    assert ctx.conversation_id.startswith("conv_")
    assert ctx.trace_id.startswith("trc_")
    assert ctx.request_id.startswith("req_")


def test_trace_context_event_fields_match_context_values() -> None:
    ctx = build_trace_context("conv_xyz")
    fields = ctx.as_event_fields()
    assert fields["conversation_id"] == ctx.conversation_id
    assert fields["trace_id"] == ctx.trace_id
    assert fields["request_id"] == ctx.request_id
