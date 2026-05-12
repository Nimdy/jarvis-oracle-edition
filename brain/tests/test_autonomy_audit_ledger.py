"""Phase 6.5 — durable audit subscriber for autonomy/L3 events.

Scope:
- The audit ledger subscribes to every autonomy/escalation event on the
  bus and appends one JSON line per event to a durable file.
- Disk failures must never raise into the bus handler.
- Rotation caps the file at ~MAX_LOG_SIZE_MB.
- ``load_recent`` returns the expected tail.
- Idempotent wire; unwire stops further appends.
- Event taxonomy: AUDITED_EVENTS covers every exported autonomy event.

Tests construct a local ``EventBus()`` so they do not touch the module
singleton or its CLOSED boot-barrier. Each test gets a fresh bus +
fresh ledger writing under ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_BRAIN = _HERE.parent
if str(_BRAIN) not in sys.path:
    sys.path.insert(0, str(_BRAIN))

from autonomy.audit_ledger import (  # noqa: E402
    AUDITED_EVENTS,
    AutonomyAuditLedger,
    MAX_LOG_SIZE_MB,
    get_audit_ledger,
    reset_audit_ledger_for_tests,
)
from consciousness.events import (  # noqa: E402
    AUTONOMY_ESCALATION_APPROVED,
    AUTONOMY_ESCALATION_EXPIRED,
    AUTONOMY_ESCALATION_PARKED,
    AUTONOMY_ESCALATION_REJECTED,
    AUTONOMY_ESCALATION_REQUESTED,
    AUTONOMY_ESCALATION_ROLLED_BACK,
    AUTONOMY_L3_ACTIVATION_DENIED,
    AUTONOMY_L3_ELIGIBLE,
    AUTONOMY_L3_PROMOTED,
    AUTONOMY_LEVEL_CHANGED,
    EventBus,
)


@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus with the boot barrier open.

    The production singleton starts CLOSED and buffers non-boot events;
    tests need the barrier OPEN so handlers run synchronously.
    """
    b = EventBus()
    b.open_barrier()
    return b


@pytest.fixture()
def ledger(tmp_path: Path, bus: EventBus) -> AutonomyAuditLedger:
    L = AutonomyAuditLedger(path=tmp_path / "autonomy_audit.jsonl")
    yield L
    L.unwire()


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_audit_ledger_for_tests()
    yield
    reset_audit_ledger_for_tests()


# --------------------------------------------------------------------------
# Subscription behaviour
# --------------------------------------------------------------------------


def test_audited_events_covers_all_autonomy_events() -> None:
    """Every autonomy/escalation event we export must be in AUDITED_EVENTS.

    Regression guard: if a new event constant is added but not added to
    AUDITED_EVENTS, the audit trail silently drops it. That is a
    governance bug, so this test asserts the coverage contract.
    """
    expected = {
        AUTONOMY_LEVEL_CHANGED,
        AUTONOMY_L3_ELIGIBLE,
        AUTONOMY_L3_PROMOTED,
        AUTONOMY_L3_ACTIVATION_DENIED,
        AUTONOMY_ESCALATION_REQUESTED,
        AUTONOMY_ESCALATION_APPROVED,
        AUTONOMY_ESCALATION_REJECTED,
        AUTONOMY_ESCALATION_ROLLED_BACK,
        AUTONOMY_ESCALATION_PARKED,
        AUTONOMY_ESCALATION_EXPIRED,
    }
    assert set(AUDITED_EVENTS) == expected


def test_wire_is_idempotent(bus: EventBus, ledger: AutonomyAuditLedger) -> None:
    ledger.wire(bus=bus)
    first_count = len(ledger._cleanups)  # type: ignore[attr-defined]
    ledger.wire(bus=bus)
    assert len(ledger._cleanups) == first_count  # type: ignore[attr-defined]
    assert ledger.wired is True


def test_unwire_stops_recording(bus: EventBus, ledger: AutonomyAuditLedger) -> None:
    ledger.wire(bus=bus)
    bus.emit(AUTONOMY_L3_ELIGIBLE, reason="test-before-unwire")
    ledger.unwire()
    bus.emit(AUTONOMY_L3_ELIGIBLE, reason="test-after-unwire")

    entries = ledger.load_recent(limit=10)
    reasons = [e["payload"].get("reason") for e in entries]
    assert "test-before-unwire" in reasons
    assert "test-after-unwire" not in reasons


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------


def test_every_audited_event_reaches_disk(
    bus: EventBus, ledger: AutonomyAuditLedger
) -> None:
    ledger.wire(bus=bus)
    bus.emit(AUTONOMY_LEVEL_CHANGED, old_level=1, new_level=2)
    bus.emit(AUTONOMY_L3_ELIGIBLE, reason="wins earned", wins=30, win_rate=0.55)
    bus.emit(
        AUTONOMY_L3_ACTIVATION_DENIED,
        reason="missing_evidence_path", caller_id="test",
    )
    bus.emit(
        AUTONOMY_L3_PROMOTED,
        outcome="clean", prior_level=2, evidence_path="runtime:live",
    )
    bus.emit(
        AUTONOMY_ESCALATION_REQUESTED, escalation_id="esc-1", metric="m",
    )
    bus.emit(AUTONOMY_ESCALATION_APPROVED, escalation_id="esc-1", metric="m")
    bus.emit(AUTONOMY_ESCALATION_REJECTED, escalation_id="esc-2", metric="m")
    bus.emit(AUTONOMY_ESCALATION_ROLLED_BACK, escalation_id="esc-1", metric="m")
    bus.emit(AUTONOMY_ESCALATION_PARKED, escalation_id="esc-3", metric="m")
    bus.emit(AUTONOMY_ESCALATION_EXPIRED, escalation_id="esc-4", metric="m")

    entries = ledger.load_recent(limit=100)
    recorded = [e["event"] for e in entries]
    for event_name in AUDITED_EVENTS:
        assert event_name in recorded, f"missing {event_name}"
    assert ledger.events_recorded == len(AUDITED_EVENTS)


def test_non_audited_events_are_ignored(
    bus: EventBus, ledger: AutonomyAuditLedger
) -> None:
    ledger.wire(bus=bus)
    bus.emit("some:unrelated_event", data="ignore me")
    assert ledger.load_recent(limit=10) == []
    assert ledger.events_recorded == 0


def test_load_recent_honors_limit_and_order(
    bus: EventBus, ledger: AutonomyAuditLedger
) -> None:
    ledger.wire(bus=bus)
    for i in range(5):
        bus.emit(AUTONOMY_L3_ELIGIBLE, reason=f"r-{i}")
    entries = ledger.load_recent(limit=3)
    assert [e["payload"]["reason"] for e in entries] == ["r-2", "r-3", "r-4"]


def test_payload_sanitizer_coerces_non_json_types(
    bus: EventBus, ledger: AutonomyAuditLedger
) -> None:
    """Non-JSON-safe payload values must coerce to strings, not crash."""
    ledger.wire(bus=bus)
    bus.emit(
        AUTONOMY_L3_ELIGIBLE,
        reason="path-test",
        evidence_path=Path("/tmp/evidence.json"),
    )
    entries = ledger.load_recent(limit=10)
    assert entries
    payload = entries[-1]["payload"]
    assert payload["reason"] == "path-test"
    assert isinstance(payload["evidence_path"], str)
    assert "/tmp/evidence.json" in payload["evidence_path"]


# --------------------------------------------------------------------------
# Fault tolerance
# --------------------------------------------------------------------------


def test_disk_write_failure_does_not_crash_handler(
    tmp_path: Path, bus: EventBus, monkeypatch
) -> None:
    """If disk is unwritable the bus handler must still return cleanly."""
    bad_path = tmp_path / "nonexistent_dir" / "audit.jsonl"
    L = AutonomyAuditLedger(path=bad_path)
    L.wire(bus=bus)

    real_open = open

    def selective_boom(path, *a, **kw):
        if str(path).endswith("audit.jsonl"):
            raise OSError("disk full")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", selective_boom)
    bus.emit(AUTONOMY_L3_ELIGIBLE, reason="should-not-raise")


def test_corrupt_line_is_skipped_by_load_recent(
    ledger: AutonomyAuditLedger,
) -> None:
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    ledger.path.write_text(
        json.dumps({"ts": 1.0, "event": "autonomy:l3_eligible", "payload": {"reason": "ok"}}) + "\n"
        + "{bad json}\n"
        + json.dumps({"ts": 2.0, "event": "autonomy:l3_promoted", "payload": {"outcome": "clean"}}) + "\n"
    )
    entries = ledger.load_recent(limit=10)
    assert len(entries) == 2
    assert entries[0]["event"] == "autonomy:l3_eligible"
    assert entries[1]["event"] == "autonomy:l3_promoted"


# --------------------------------------------------------------------------
# Rotation
# --------------------------------------------------------------------------


def test_rotation_truncates_when_file_exceeds_cap(
    tmp_path: Path, bus: EventBus,
) -> None:
    """Once the file crosses MAX_LOG_SIZE_MB, rotation keeps the newest
    half and new entries continue appending."""
    L = AutonomyAuditLedger(path=tmp_path / "autonomy_audit.jsonl")
    L.path.parent.mkdir(parents=True, exist_ok=True)
    seed_line = json.dumps(
        {"ts": 0.0, "event": "autonomy:l3_eligible", "payload": {"n": "seed"}}
    ) + "\n"
    needed = (MAX_LOG_SIZE_MB * 1024 * 1024) // len(seed_line.encode()) + 10
    with open(L.path, "w") as fh:
        for _ in range(needed):
            fh.write(seed_line)
    size_before = L.path.stat().st_size
    assert size_before > MAX_LOG_SIZE_MB * 1024 * 1024

    L.wire(bus=bus)
    bus.emit(AUTONOMY_L3_ELIGIBLE, reason="post-rotation marker")

    size_after = L.path.stat().st_size
    assert size_after < size_before
    tail = L.load_recent(limit=5)
    assert any(e["payload"].get("reason") == "post-rotation marker" for e in tail)


# --------------------------------------------------------------------------
# Singleton accessor
# --------------------------------------------------------------------------


def test_singleton_returns_same_instance() -> None:
    a = get_audit_ledger()
    b = get_audit_ledger()
    assert a is b


def test_reset_singleton_yields_fresh_instance() -> None:
    a = get_audit_ledger()
    reset_audit_ledger_for_tests()
    b = get_audit_ledger()
    assert a is not b
