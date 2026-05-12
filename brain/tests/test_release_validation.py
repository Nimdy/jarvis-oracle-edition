from consciousness.events import OUTPUT_RELEASE_BLOCKED, OUTPUT_VALIDATION_RECORDED
from consciousness.release_validation import OutputReleaseValidator
from jarvis_eval.event_tap import _TAPPED_EVENTS


def _new_validator(tmp_path) -> OutputReleaseValidator:
    return OutputReleaseValidator(record_path=str(tmp_path / "release_validation.jsonl"))


def test_released_output_validation_passes_with_required_fields(tmp_path) -> None:
    validator = _new_validator(tmp_path)
    decision = validator.validate_output(
        text="hello world",
        conversation_id="conv_1",
        trace_id="trc_1",
        request_id="req_1",
        output_id="out_1",
        release_status="released",
        release_reason="",
        source="test",
    )
    assert decision.passed
    assert decision.effective_release_status == "released"
    assert decision.violations == ()


def test_released_output_validation_blocks_on_missing_trace_fields(tmp_path) -> None:
    validator = _new_validator(tmp_path)
    decision = validator.validate_output(
        text="hello world",
        conversation_id="conv_2",
        trace_id="",
        request_id="req_2",
        output_id="out_2",
        release_status="released",
        release_reason="",
        source="test",
    )
    assert not decision.passed
    assert decision.effective_release_status == "blocked"
    assert "missing_trace_id" in decision.violations


def test_release_observer_detects_released_without_validation(tmp_path) -> None:
    validator = _new_validator(tmp_path)
    validator._on_conversation_response(
        release_status="released",
        conversation_id="conv_3",
        trace_id="trc_3",
        request_id="req_3",
        output_id="out_3",
        validation_id="",
        validation_passed=False,
    )
    stats = validator.get_stats()
    assert stats["released_total"] == 1
    assert stats["released_without_validation"] == 1
    assert stats["released_validated"] == 0


def test_release_observer_counts_validated_releases(tmp_path) -> None:
    validator = _new_validator(tmp_path)
    decision = validator.validate_output(
        text="hello world",
        conversation_id="conv_4",
        trace_id="trc_4",
        request_id="req_4",
        output_id="out_4",
        release_status="released",
        release_reason="",
        source="test",
    )
    validator._on_conversation_response(
        release_status="released",
        conversation_id="conv_4",
        trace_id="trc_4",
        request_id="req_4",
        output_id="out_4",
        validation_id=decision.validation_id,
        validation_passed=True,
    )
    stats = validator.get_stats()
    assert stats["released_total"] == 1
    assert stats["released_validated"] == 1
    assert stats["released_without_validation"] == 0


def test_eval_event_tap_includes_output_validation_events() -> None:
    assert OUTPUT_VALIDATION_RECORDED in _TAPPED_EVENTS
    assert OUTPUT_RELEASE_BLOCKED in _TAPPED_EVENTS
