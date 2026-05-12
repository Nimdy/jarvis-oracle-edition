"""Tests for Layer 4: Delayed Outcome Attribution.

Covers:
  - OutcomeScheduler: schedule, tick, resolve, retry, inconclusive, eviction
  - build_outcome_data helper
  - Standardized outcome constants
  - LedgerEntry outcome stamping round-trip
  - LearningJob ledger_entry_id persistence
"""

import os
import sys
import time
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from consciousness.attribution_ledger import (
    AttributionLedger,
    OutcomeScheduler,
    PendingOutcome,
    LedgerEntry,
    build_outcome_data,
    _short_id,
)
from consciousness.events import OUTCOME_RESOLVED, ATTRIBUTION_ENTRY_RECORDED


# ---------------------------------------------------------------------------
# build_outcome_data
# ---------------------------------------------------------------------------


class TestBuildOutcomeData:
    def test_default_fields(self):
        d = build_outcome_data()
        assert d["outcome_confidence"] == 1.0
        assert d["latency_s"] == 0.0
        assert d["outcome_source"] == "system_metric"
        assert d["outcome_tier"] == "immediate"

    def test_custom_fields(self):
        d = build_outcome_data(
            confidence=0.7, latency_s=12.5,
            source="user_feedback", tier="delayed",
            custom_key="hello",
        )
        assert d["outcome_confidence"] == 0.7
        assert d["latency_s"] == 12.5
        assert d["outcome_source"] == "user_feedback"
        assert d["outcome_tier"] == "delayed"
        assert d["custom_key"] == "hello"

    def test_confidence_clamped(self):
        d = build_outcome_data(confidence=1.5)
        assert d["outcome_confidence"] == 1.0
        d2 = build_outcome_data(confidence=-0.3)
        assert d2["outcome_confidence"] == 0.0

    def test_default_scope_and_blame(self):
        d = build_outcome_data()
        assert d["outcome_scope"] == "general"
        assert d["blame_target"] == "general"

    def test_custom_scope_and_blame(self):
        d = build_outcome_data(
            scope="autonomy_policy",
            blame_target="intent_selection",
        )
        assert d["outcome_scope"] == "autonomy_policy"
        assert d["blame_target"] == "intent_selection"

    def test_response_quality_scope(self):
        d = build_outcome_data(
            scope="response_quality",
            blame_target="response_generation",
            user_signal="positive",
        )
        assert d["outcome_scope"] == "response_quality"
        assert d["blame_target"] == "response_generation"
        assert d["user_signal"] == "positive"


# ---------------------------------------------------------------------------
# OutcomeScheduler
# ---------------------------------------------------------------------------


def _fresh_scheduler():
    """Create a fresh scheduler instance (not the singleton)."""
    s = OutcomeScheduler.__new__(OutcomeScheduler)
    s.__init__()
    return s


class TestOutcomeScheduler:
    def test_schedule_and_tick_resolves(self):
        sched = _fresh_scheduler()
        sched.schedule(
            entry_id="led_test123",
            delay_s=0,
            check_fn=lambda: ("success", {"metric": "ok"}),
            subsystem="test",
            description="test check",
        )
        assert sched.get_stats()["pending"] == 1
        resolved = sched.tick()
        assert resolved == 1
        stats = sched.get_stats()
        assert stats["pending"] == 0
        assert stats["resolved"] == 1

    def test_tick_defers_future_checks(self):
        sched = _fresh_scheduler()
        sched.schedule(
            entry_id="led_future",
            delay_s=9999,
            check_fn=lambda: ("success", {}),
            subsystem="test",
            description="far future",
        )
        resolved = sched.tick()
        assert resolved == 0
        assert sched.get_stats()["pending"] == 1

    def test_retry_then_inconclusive(self):
        sched = _fresh_scheduler()
        call_count = [0]

        def _check():
            call_count[0] += 1
            return None

        sched.schedule(
            entry_id="led_retry",
            delay_s=0,
            check_fn=_check,
            subsystem="test",
            description="will fail",
            max_retries=1,
        )
        resolved = sched.tick()
        assert resolved == 0
        assert call_count[0] == 1
        assert sched.get_stats()["pending"] == 1

        for po in sched._pending:
            po.check_at = 0

        resolved2 = sched.tick()
        assert resolved2 == 1
        assert call_count[0] == 2
        assert sched.get_stats()["inconclusive"] == 1

    def test_eviction_on_overflow(self):
        sched = _fresh_scheduler()
        for i in range(201):
            sched.schedule(
                entry_id=f"led_{i}",
                delay_s=9999,
                check_fn=lambda: ("success", {}),
                subsystem="test",
                description=f"item {i}",
            )
        stats = sched.get_stats()
        assert stats["pending"] <= 200
        assert stats["evicted"] >= 1

    def test_check_fn_exception_handled(self):
        sched = _fresh_scheduler()
        sched.schedule(
            entry_id="led_err",
            delay_s=0,
            check_fn=lambda: 1 / 0,
            subsystem="test",
            description="will raise",
        )
        resolved = sched.tick()
        assert resolved == 0
        assert sched.get_stats()["errors"] == 1
        assert sched.get_stats()["pending"] == 1

    def test_stats_by_subsystem(self):
        sched = _fresh_scheduler()
        sched.schedule("a", 9999, lambda: None, "autonomy", "a")
        sched.schedule("b", 9999, lambda: None, "autonomy", "b")
        sched.schedule("c", 9999, lambda: None, "self_improve", "c")
        stats = sched.get_stats()
        assert stats["pending_by_subsystem"]["autonomy"] == 2
        assert stats["pending_by_subsystem"]["self_improve"] == 1


# ---------------------------------------------------------------------------
# AttributionLedger record_outcome round-trip
# ---------------------------------------------------------------------------


def _fresh_ledger():
    """Create a fresh ledger instance with temp storage."""
    l = AttributionLedger.__new__(AttributionLedger)
    l.__init__()
    return l


class TestLedgerOutcome:
    def test_record_then_outcome(self):
        ledger = _fresh_ledger()
        eid = ledger.record(
            subsystem="test", event_type="test_action",
            data={"msg": "hello"},
        )
        assert eid.startswith("led_")
        entry = ledger.get_entry(eid)
        assert entry is not None
        assert entry.outcome == "pending"

        ledger.record_outcome(eid, "success", build_outcome_data(
            confidence=0.9, latency_s=1.5,
            source="user_feedback", tier="immediate",
        ))

        entry = ledger.get_entry(eid)
        assert entry.outcome == "success"
        assert entry.outcome_data["outcome_confidence"] == 0.9
        assert entry.outcome_data["latency_s"] == 1.5
        assert entry.outcome_data["outcome_source"] == "user_feedback"
        assert entry.outcome_data["outcome_tier"] == "immediate"

    def test_outcome_counts(self):
        ledger = _fresh_ledger()
        eid1 = ledger.record(subsystem="test", event_type="a")
        eid2 = ledger.record(subsystem="test", event_type="b")
        ledger.record_outcome(eid1, "success")
        ledger.record_outcome(eid2, "failure")

        stats = ledger.get_stats()
        assert stats["total_outcomes"] == 2
        assert stats["outcome_counts"]["success"] == 1
        assert stats["outcome_counts"]["failure"] == 1

    def test_chain_includes_outcomes(self):
        ledger = _fresh_ledger()
        root_eid = ledger.record(subsystem="test", event_type="root")
        child_eid = ledger.record(
            subsystem="test", event_type="child",
            parent_entry_id=root_eid,
        )
        ledger.record_outcome(root_eid, "stable", {"note": "good"})

        chain = ledger.get_chain(root_eid)
        assert len(chain) == 2
        root_entry = next(e for e in chain if e["entry_id"] == root_eid)
        assert root_entry["outcome"] == "stable"

    def test_snapshot_dict_includes_outcome(self):
        ledger = _fresh_ledger()
        eid = ledger.record(subsystem="test", event_type="snap")
        ledger.record_outcome(eid, "partial", {"x": 1})

        entry = ledger.get_entry(eid)
        snap = entry.to_snapshot_dict()
        assert snap["outcome"] == "partial"
        assert snap["outcome_data"]["x"] == 1
        assert snap["outcome_ts"] > 0


# ---------------------------------------------------------------------------
# LearningJob ledger_entry_id field
# ---------------------------------------------------------------------------


class TestLearningJobLedgerField:
    def test_field_exists_and_persists(self):
        from skills.learning_jobs import LearningJob
        job = LearningJob(
            job_id="test_job",
            skill_id="test_skill",
            capability_type="procedural",
        )
        assert job.ledger_entry_id == ""

        job.ledger_entry_id = "led_abc123def456"
        d = job.to_dict()
        assert d["ledger_entry_id"] == "led_abc123def456"

        loaded = LearningJob.from_dict(d)
        assert loaded.ledger_entry_id == "led_abc123def456"

    def test_from_dict_without_field(self):
        """Legacy jobs without ledger_entry_id should load cleanly."""
        from skills.learning_jobs import LearningJob
        raw = {
            "job_id": "old_job",
            "skill_id": "old_skill",
            "capability_type": "procedural",
            "status": "active",
            "phase": "assess",
        }
        job = LearningJob.from_dict(raw)
        assert job.ledger_entry_id == ""


# ---------------------------------------------------------------------------
# Event constants
# ---------------------------------------------------------------------------


class TestEventConstants:
    def test_outcome_resolved_exists(self):
        assert OUTCOME_RESOLVED == "attribution:outcome_resolved"

    def test_entry_recorded_exists(self):
        assert ATTRIBUTION_ENTRY_RECORDED == "attribution:entry_recorded"


# ---------------------------------------------------------------------------
# Provenance tie-breaker gating by outcome_scope
# ---------------------------------------------------------------------------


class TestProvenanceTiebreakerGating:
    """Verify that autonomy_policy outcomes don't penalize memory provenance."""

    def test_response_quality_applies_provenance_bonus(self):
        from memory.retrieval_log import (
            MemoryRetrievalLog,
            CandidateRecord,
        )
        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log.__init__()

        eid = log.log_retrieval(
            query_text="test",
            candidates=[
                CandidateRecord(
                    memory_id="m1", similarity=0.9, recency_score=0.5,
                    weight=0.8, memory_type="conversation", tag_count=2,
                    association_count=0, priority=0, provenance_boost=0.12,
                    speaker_match=False, heuristic_score=0.7, selected=True,
                    injected=True,
                ),
            ],
            selected_memory_ids=["m1"],
            conversation_id="conv_test",
        )
        log.mark_injected(eid, ["m1"])
        log.log_outcome("conv_test", "ok", outcome_scope="response_quality")

        pairs = log.get_training_pairs()
        assert len(pairs) == 1
        # injected + not-referenced + ok = 0.8, provenance bonus +0.05 = 0.85
        assert pairs[0].label >= 0.85

    def test_autonomy_policy_skips_provenance_penalty(self):
        from memory.retrieval_log import (
            MemoryRetrievalLog,
            CandidateRecord,
        )
        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log.__init__()

        eid = log.log_retrieval(
            query_text="test",
            candidates=[
                CandidateRecord(
                    memory_id="m2", similarity=0.9, recency_score=0.5,
                    weight=0.8, memory_type="conversation", tag_count=2,
                    association_count=0, priority=0, provenance_boost=0.01,
                    speaker_match=False, heuristic_score=0.7, selected=True,
                    injected=True,
                ),
            ],
            selected_memory_ids=["m2"],
            conversation_id="conv_test2",
        )
        log.mark_injected(eid, ["m2"])
        log.log_outcome("conv_test2", "error", outcome_scope="autonomy_policy")

        pairs = log.get_training_pairs()
        assert len(pairs) == 1
        # injected + error = 0.3, autonomy_policy scope => provenance penalty skipped
        assert pairs[0].label == 0.3


# ---------------------------------------------------------------------------
# Memory storage provenance breakdown
# ---------------------------------------------------------------------------


class TestMemoryProvenance:
    def test_stats_include_provenance_breakdown(self):
        from memory.storage import MemoryStorage
        from consciousness.events import Memory

        storage = MemoryStorage()
        storage.add(Memory(
            id="m1", timestamp=time.time(), weight=0.5,
            tags=("test",), payload="hello", type="conversation",
            provenance="observed",
        ))
        storage.add(Memory(
            id="m2", timestamp=time.time(), weight=0.6,
            tags=("test",), payload="world", type="factual_knowledge",
            provenance="external_source",
        ))
        storage.add(Memory(
            id="m3", timestamp=time.time(), weight=0.7,
            tags=("test",), payload="old", type="conversation",
            provenance="observed",
        ))

        stats = storage.get_stats()
        assert "by_provenance" in stats
        assert stats["by_provenance"]["observed"] == 2
        assert stats["by_provenance"]["external_source"] == 1

    def test_recent_with_provenance(self):
        from memory.storage import MemoryStorage
        from consciousness.events import Memory

        storage = MemoryStorage()
        storage.add(Memory(
            id="m1", timestamp=time.time(), weight=0.5,
            tags=("test",), payload="hello world", type="conversation",
            provenance="user_claim",
        ))

        recent = storage.get_recent_with_provenance(5)
        assert len(recent) == 1
        assert recent[0]["provenance"] == "user_claim"
        assert recent[0]["type"] == "conversation"
        assert "payload_preview" in recent[0]
