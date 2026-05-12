"""Tests for the Attribution Ledger — epistemic foundation layer.

Tests cover:
  - LedgerEntry creation, serialization, deserialization
  - Append-only JSONL persistence (entries and outcomes are separate lines)
  - Rehydration with outcome folding
  - Causal chain traversal (root_entry_id / parent_entry_id)
  - Evidence refs structure
  - Stats and query
  - Capability gate integration (blocked claim -> ledger entry)
  - File rotation
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.attribution_ledger import (
    AttributionLedger, LedgerEntry, _short_id, _LEDGER_PATH,
)


def _temp_ledger() -> tuple[AttributionLedger, str]:
    """Create a fresh ledger with a temp file path."""
    import consciousness.attribution_ledger as mod
    tmpdir = tempfile.mkdtemp()
    tmppath = os.path.join(tmpdir, "test_ledger.jsonl")
    old_path = mod._LEDGER_PATH
    mod._LEDGER_PATH = tmppath

    ledger = AttributionLedger()
    ledger._rehydrated = False

    mod._LEDGER_PATH = old_path
    return ledger, tmppath


def test_entry_creation_and_serialization():
    entry = LedgerEntry(
        entry_id="led_test123",
        root_entry_id="led_test123",
        parent_entry_id="",
        timestamp=time.time(),
        subsystem="test",
        event_type="test_event",
        actor="tester",
        source="unit_test",
        confidence=0.95,
        conversation_id="conv_abc",
        evidence_refs=[{"kind": "test", "id": "t1"}],
        data={"key": "value"},
    )
    d = entry.to_record_dict()
    assert d["type"] == "entry"
    assert d["entry_id"] == "led_test123"
    assert d["root_entry_id"] == "led_test123"
    assert d["evidence_refs"] == [{"kind": "test", "id": "t1"}]
    assert d["data"] == {"key": "value"}

    restored = LedgerEntry.from_record_dict(d)
    assert restored.entry_id == entry.entry_id
    assert restored.subsystem == entry.subsystem
    assert restored.evidence_refs == entry.evidence_refs
    print("  PASS: entry creation and serialization")


def test_record_and_stats():
    ledger = AttributionLedger()
    ledger._buffer.clear()
    ledger._total_recorded = 0
    ledger._subsystem_counts.clear()

    eid = ledger.record(
        subsystem="test_sub",
        event_type="test_event",
        actor="tester",
        data={"hello": "world"},
        evidence_refs=[{"kind": "test", "id": "x"}],
    )
    assert eid.startswith("led_")

    stats = ledger.get_stats()
    assert stats["total_recorded"] >= 1
    assert "test_sub" in stats["subsystem_counts"]

    entry = ledger.get_entry(eid)
    assert entry is not None
    assert entry.subsystem == "test_sub"
    assert entry.evidence_refs == [{"kind": "test", "id": "x"}]
    print("  PASS: record and stats")


def test_root_entry_id_auto_set():
    ledger = AttributionLedger()
    ledger._buffer.clear()

    eid1 = ledger.record(subsystem="a", event_type="root")
    entry1 = ledger.get_entry(eid1)
    assert entry1.root_entry_id == eid1

    eid2 = ledger.record(subsystem="a", event_type="child", parent_entry_id=eid1)
    entry2 = ledger.get_entry(eid2)
    assert entry2.root_entry_id == eid1
    assert entry2.parent_entry_id == eid1

    eid3 = ledger.record(subsystem="a", event_type="grandchild",
                          parent_entry_id=eid2, root_entry_id=eid1)
    entry3 = ledger.get_entry(eid3)
    assert entry3.root_entry_id == eid1
    assert entry3.parent_entry_id == eid2
    print("  PASS: root_entry_id auto-set")


def test_append_only_outcomes():
    """Outcomes are appended as separate JSONL lines, never mutating the original."""
    import consciousness.attribution_ledger as mod
    tmpdir = tempfile.mkdtemp()
    tmppath = os.path.join(tmpdir, "test_outcomes.jsonl")
    original_path = mod._LEDGER_PATH
    mod._LEDGER_PATH = tmppath

    try:
        ledger = AttributionLedger()
        ledger._buffer.clear()
        ledger._total_recorded = 0
        ledger._total_outcomes = 0

        eid = ledger.record(subsystem="test", event_type="initial")

        ledger.record_outcome(eid, "success", {"metric": 0.95})

        entry = ledger.get_entry(eid)
        assert entry.outcome == "success"
        assert entry.outcome_data == {"metric": 0.95}

        with open(tmppath, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["type"] == "entry"
        assert lines[1]["type"] == "outcome"
        assert lines[1]["entry_id"] == eid
        assert lines[1]["outcome"] == "success"

        stats = ledger.get_stats()
        assert stats["total_outcomes"] >= 1
    finally:
        mod._LEDGER_PATH = original_path
    print("  PASS: append-only outcomes")


def test_rehydration_with_outcome_folding():
    import consciousness.attribution_ledger as mod
    tmpdir = tempfile.mkdtemp()
    tmppath = os.path.join(tmpdir, "test_rehydrate.jsonl")
    original_path = mod._LEDGER_PATH
    mod._LEDGER_PATH = tmppath

    try:
        ledger1 = AttributionLedger()
        ledger1._buffer.clear()
        ledger1._total_recorded = 0

        eid = ledger1.record(subsystem="test", event_type="rehydrate_test",
                              data={"val": 42})
        ledger1.record_outcome(eid, "success", {"score": 0.9})

        ledger2 = AttributionLedger()
        ledger2._rehydrated = False
        ledger2._buffer.clear()
        loaded = ledger2.rehydrate()
        assert loaded >= 1

        entry = ledger2.get_entry(eid)
        assert entry is not None
        assert entry.subsystem == "test"
        assert entry.outcome == "success"
        assert entry.outcome_data == {"score": 0.9}
    finally:
        mod._LEDGER_PATH = original_path
    print("  PASS: rehydration with outcome folding")


def test_causal_chain():
    ledger = AttributionLedger()
    ledger._buffer.clear()

    root = ledger.record(subsystem="conv", event_type="user_message")
    child1 = ledger.record(subsystem="gate", event_type="claim_blocked",
                            parent_entry_id=root, root_entry_id=root)
    child2 = ledger.record(subsystem="learning_jobs", event_type="job_created",
                            parent_entry_id=child1, root_entry_id=root)

    chain = ledger.get_chain(root)
    assert len(chain) == 3
    ids = [e["entry_id"] for e in chain]
    assert root in ids
    assert child1 in ids
    assert child2 in ids
    assert chain[0]["ts"] <= chain[1]["ts"] <= chain[2]["ts"]
    print("  PASS: causal chain traversal")


def test_query():
    ledger = AttributionLedger()
    ledger._buffer.clear()

    ledger.record(subsystem="gate", event_type="block", actor="David")
    ledger.record(subsystem="conv", event_type="message", actor="David")
    ledger.record(subsystem="gate", event_type="block", actor="system")

    gate_blocks = ledger.query(subsystem="gate")
    assert len(gate_blocks) == 2

    david_events = ledger.query(actor="David")
    assert len(david_events) == 2

    gate_david = ledger.query(subsystem="gate", actor="David")
    assert len(gate_david) == 1
    print("  PASS: query filtering")


def test_get_recent():
    ledger = AttributionLedger()
    ledger._buffer.clear()

    for i in range(5):
        ledger.record(subsystem="test", event_type=f"evt_{i}")

    recent = ledger.get_recent(3)
    assert len(recent) == 3
    assert recent[0]["event_type"] == "evt_4"
    assert recent[2]["event_type"] == "evt_2"
    print("  PASS: get_recent")


def test_evidence_refs_structure():
    ledger = AttributionLedger()
    ledger._buffer.clear()

    eid = ledger.record(
        subsystem="learning_jobs",
        event_type="job_created",
        evidence_refs=[
            {"kind": "skill", "id": "speaker_diarization_v1"},
            {"kind": "job", "id": "job_abc123"},
            {"kind": "conversation", "id": "conv_xyz"},
        ],
    )
    entry = ledger.get_entry(eid)
    assert len(entry.evidence_refs) == 3
    kinds = {r["kind"] for r in entry.evidence_refs}
    assert kinds == {"skill", "job", "conversation"}
    print("  PASS: evidence_refs structure")


def test_capability_gate_records_block():
    from skills.capability_gate import CapabilityGate
    from skills.registry import SkillRegistry, _default_skills
    from consciousness.attribution_ledger import attribution_ledger as singleton_ledger

    reg = SkillRegistry(path="/dev/null")
    reg._skills = {r.skill_id: r for r in _default_skills()}
    reg._loaded = True
    reg.save = lambda: None  # type: ignore[assignment]

    old_total = singleton_ledger._total_recorded

    gate = CapabilityGate(reg)
    gate.check_text("I can synthesize new audio samples.")

    assert singleton_ledger._total_recorded > old_total, "Gate block should have recorded a ledger entry"
    recent = singleton_ledger.get_recent(5)
    gate_entries = [e for e in recent if e["subsystem"] == "capability_gate"]
    assert len(gate_entries) > 0
    assert gate_entries[0]["event_type"] == "claim_blocked"
    print("  PASS: capability gate records block to ledger")


if __name__ == "__main__":
    print("\n=== Attribution Ledger Tests ===\n")
    test_entry_creation_and_serialization()
    test_record_and_stats()
    test_root_entry_id_auto_set()
    test_append_only_outcomes()
    test_rehydration_with_outcome_folding()
    test_causal_chain()
    test_query()
    test_get_recent()
    test_evidence_refs_structure()
    test_capability_gate_records_block()
    print("\n  All tests passed!\n")
