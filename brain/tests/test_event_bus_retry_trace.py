from unittest.mock import MagicMock, patch

from consciousness.events import (
    ATTRIBUTION_ENTRY_RECORDED,
    EventBus,
    OUTCOME_RESOLVED,
    RETRY_EXECUTED,
    RETRY_EXHAUSTED,
    RETRY_SCHEDULED,
)
from jarvis_eval.event_tap import _TAPPED_EVENTS


def _new_open_bus() -> EventBus:
    bus = EventBus()
    bus.open_barrier()
    return bus


def test_retry_scheduled_emits_lineage_event() -> None:
    bus = _new_open_bus()
    captured: list[dict] = []
    bus.on(RETRY_SCHEDULED, lambda **kwargs: captured.append(kwargs))

    with patch("asyncio.get_event_loop", side_effect=RuntimeError), patch("threading.Timer") as timer_cls:
        timer = MagicMock()
        timer_cls.return_value = timer
        bus._schedule_retry(
            "conversation:response",
            {"conversation_id": "conv_1", "output_id": "out_1"},
            0,
            retry_id="rty_test",
            reason="unit_test",
        )

    assert len(captured) == 1
    event = captured[0]
    assert event["retry_id"] == "rty_test"
    assert event["target_event_type"] == "conversation:response"
    assert event["conversation_id"] == "conv_1"
    assert event["output_id"] == "out_1"
    assert event["attempt"] == 1
    assert event["reason"] == "unit_test"
    timer_cls.assert_called_once()
    timer.start.assert_called_once()


def test_retry_exhausted_emits_terminal_retry_event() -> None:
    bus = _new_open_bus()
    captured: list[dict] = []
    bus.on(RETRY_EXHAUSTED, lambda **kwargs: captured.append(kwargs))

    bus._schedule_retry(
        "conversation:response",
        {"conversation_id": "conv_2"},
        bus.MAX_RETRIES,
        retry_id="rty_done",
    )

    assert len(captured) == 1
    event = captured[0]
    assert event["retry_id"] == "rty_done"
    assert event["target_event_type"] == "conversation:response"
    assert event["attempt"] == bus.MAX_RETRIES


def test_retry_executed_event_emitted_before_retry_emit() -> None:
    bus = _new_open_bus()
    executed: list[dict] = []
    payloads: list[dict] = []
    bus.on(RETRY_EXECUTED, lambda **kwargs: executed.append(kwargs))
    bus.on("test:event", lambda **kwargs: payloads.append(kwargs))

    bus._execute_retry(
        "test:event",
        {"conversation_id": "conv_3", "output_id": "out_3"},
        1,
        "rty_exec",
    )

    assert len(executed) == 1
    assert executed[0]["retry_id"] == "rty_exec"
    assert executed[0]["conversation_id"] == "conv_3"
    assert len(payloads) == 1
    assert payloads[0]["conversation_id"] == "conv_3"


def test_eval_event_tap_includes_retry_events() -> None:
    assert RETRY_SCHEDULED in _TAPPED_EVENTS
    assert RETRY_EXECUTED in _TAPPED_EVENTS
    assert RETRY_EXHAUSTED in _TAPPED_EVENTS


def test_eval_event_tap_includes_attribution_lifecycle_events() -> None:
    assert ATTRIBUTION_ENTRY_RECORDED in _TAPPED_EVENTS
    assert OUTCOME_RESOLVED in _TAPPED_EVENTS
