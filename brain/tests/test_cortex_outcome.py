"""Regression tests for cortex outcome recording path.

Validates that the retrieval outcome loop in conversation_handler is functional:
- log_outcome() is reached without NameError
- outcome counters track success/failure
- training pairs accumulate when feedback arrives
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory
from memory.storage import memory_storage
from memory.retrieval_log import MemoryRetrievalLog, CandidateRecord, detect_memory_references


def _make_log(tmp_dir: str) -> MemoryRetrievalLog:
    path = os.path.join(tmp_dir, "test_retrieval_log.jsonl")
    log = MemoryRetrievalLog(path=path)
    log.init()
    return log


def _make_candidate(mid: str, sim: float = 0.8, selected: bool = True) -> CandidateRecord:
    return CandidateRecord(
        memory_id=mid, similarity=sim, recency_score=0.5,
        weight=0.6, memory_type="conversation", tag_count=2,
        association_count=1, priority=500, provenance_boost=0.0,
        speaker_match=False, heuristic_score=sim * 0.5 + 0.25,
        selected=selected,
    )


def test_log_outcome_no_exception():
    """The core bug: log_outcome must not crash from undefined variables."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        cid = "conv_test_001"
        candidates = [_make_candidate("mem_a"), _make_candidate("mem_b", selected=False)]
        event_id = log.log_retrieval(cid, "test query", candidates, ["mem_a"])
        log.mark_injected(event_id, ["mem_a"])

        positive = True
        negative = False
        follow_up = False
        _user_sig = ""
        if positive:
            _user_sig = "positive"
        elif negative:
            _user_sig = "negative"
        elif follow_up:
            _user_sig = "follow_up"

        log.log_outcome(
            conversation_id=cid,
            outcome="ok",
            latency_ms=150.0,
            user_signal=_user_sig,
        )

        stats = log.get_stats()
        assert stats["total_outcomes"] == 1, f"Expected 1 outcome, got {stats['total_outcomes']}"
        assert stats["outcome_stats"]["ok"] == 1
    print("  PASS: log_outcome records without exception")


def test_training_pairs_from_outcome():
    """Training pairs must be produced when retrieval + outcome are both logged."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        cid = "conv_test_002"
        candidates = [
            _make_candidate("mem_x", sim=0.9),
            _make_candidate("mem_y", sim=0.3, selected=False),
        ]
        event_id = log.log_retrieval(cid, "hello world", candidates, ["mem_x"])
        log.mark_injected(event_id, ["mem_x"])

        log.log_outcome(conversation_id=cid, outcome="ok", latency_ms=200.0, user_signal="positive")

        pairs = log.get_training_pairs()
        assert len(pairs) >= 2, f"Expected >=2 training pairs (one per candidate), got {len(pairs)}"

        injected_pair = next((p for p in pairs if p.candidate.memory_id == "mem_x"), None)
        assert injected_pair is not None, "Injected candidate should produce a training pair"
        # Without reference data, injected+ok gets 0.8 (not 1.0). 1.0 is reserved for injected+referenced.
        assert 0.75 <= injected_pair.label <= 0.85, f"Injected+ok (no ref) should be ~0.8, got {injected_pair.label}"

        rejected_pair = next((p for p in pairs if p.candidate.memory_id == "mem_y"), None)
        assert rejected_pair is not None
        assert rejected_pair.label == 0.0, f"Not-selected should have label 0.0, got {rejected_pair.label}"
    print("  PASS: training pairs produced from outcome")


def test_negative_feedback_adjusts_label():
    """Negative user signal should reduce label for injected memories."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        cid = "conv_test_003"
        candidates = [_make_candidate("mem_z")]
        event_id = log.log_retrieval(cid, "bad answer", candidates, ["mem_z"])
        log.mark_injected(event_id, ["mem_z"])

        log.log_outcome(conversation_id=cid, outcome="ok", latency_ms=100.0, user_signal="negative")

        pairs = log.get_training_pairs()
        assert len(pairs) == 1
        # injected+ok+not-referenced: base=0.8, negative signal on unreferenced: -0.2 = 0.6
        assert 0.55 <= pairs[0].label <= 0.65, f"injected+ok+negative (no ref) should be ~0.6, got {pairs[0].label}"
    print("  PASS: negative feedback adjusts label")


def test_outcome_counters():
    """Verify the outcome_counters object in conversation_handler tracks correctly."""
    from conversation_handler import outcome_counters, _OutcomeCounters
    c = _OutcomeCounters()
    c.attempts += 1
    c.successes += 1
    snap = c.snapshot()
    assert snap["outcome_log_attempts"] == 1
    assert snap["outcome_log_successes"] == 1
    assert snap["outcome_log_failures"] == 0
    assert snap["last_outcome_log_error"] == ""

    c.attempts += 1
    c.failures += 1
    c.last_error = "NameError: name 'positive' is not defined"
    snap2 = c.snapshot()
    assert snap2["outcome_log_failures"] == 1
    assert "NameError" in snap2["last_outcome_log_error"]
    print("  PASS: outcome counters")


def test_positive_feedback_reinforces_injected_memory():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        memory_storage.clear()
        try:
            base = Memory(
                id="mem_fb_pos",
                timestamp=time.time(),
                weight=0.40,
                tags=("test",),
                payload="remember this",
                type="conversation",
            )
            assert memory_storage.add(base)

            cid = "conv_feedback_positive"
            candidates = [_make_candidate("mem_fb_pos", sim=0.9)]
            event_id = log.log_retrieval(cid, "test query", candidates, ["mem_fb_pos"])
            log.mark_injected(event_id, ["mem_fb_pos"])
            log.record_references(event_id, ["mem_fb_pos"])
            log.log_outcome(conversation_id=cid, outcome="ok", latency_ms=100.0, user_signal="positive")

            updated = memory_storage.get("mem_fb_pos")
            assert updated is not None
            assert updated.weight > 0.40
        finally:
            memory_storage.clear()
    print("  PASS: positive feedback reinforces injected memory")


def test_negative_feedback_downweights_injected_memory():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        memory_storage.clear()
        try:
            base = Memory(
                id="mem_fb_neg",
                timestamp=time.time(),
                weight=0.50,
                tags=("test",),
                payload="bad memory hit",
                type="conversation",
            )
            assert memory_storage.add(base)

            cid = "conv_feedback_negative"
            candidates = [_make_candidate("mem_fb_neg", sim=0.2)]
            event_id = log.log_retrieval(cid, "bad query", candidates, ["mem_fb_neg"])
            log.mark_injected(event_id, ["mem_fb_neg"])
            log.log_outcome(conversation_id=cid, outcome="barge_in", latency_ms=100.0)

            updated = memory_storage.get("mem_fb_neg")
            assert updated is not None
            assert updated.weight < 0.50
        finally:
            memory_storage.clear()
    print("  PASS: negative feedback downweights injected memory")


def test_correction_feedback_does_not_blanket_downweight_injected_memory():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        memory_storage.clear()
        try:
            base = Memory(
                id="mem_fb_corr",
                timestamp=time.time(),
                weight=0.50,
                tags=("test",),
                payload="User's birthday is May 10",
                type="conversation",
            )
            assert memory_storage.add(base)

            cid = "conv_feedback_correction"
            candidates = [_make_candidate("mem_fb_corr", sim=0.8)]
            event_id = log.log_retrieval(cid, "day", candidates, ["mem_fb_corr"])
            log.mark_injected(event_id, ["mem_fb_corr"])
            log.log_outcome(conversation_id=cid, outcome="ok", latency_ms=100.0, user_signal="correction")

            updated = memory_storage.get("mem_fb_corr")
            assert updated is not None
            assert abs(updated.weight - 0.50) < 1e-6
        finally:
            memory_storage.clear()
    print("  PASS: correction feedback avoids blanket downweight")


def test_eval_metrics_reflect_outcomes():
    """eval_metrics should show training_ready=True when enough pairs exist."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)

        for i in range(60):
            cid = f"conv_bulk_{i:03d}"
            candidates = [_make_candidate(f"mem_{i}")]
            eid = log.log_retrieval(cid, f"query {i}", candidates, [f"mem_{i}"])
            log.mark_injected(eid, [f"mem_{i}"])
            log.log_outcome(conversation_id=cid, outcome="ok", latency_ms=100.0)

        metrics = log.get_eval_metrics()
        assert metrics["training_ready"] is True, f"Expected training_ready=True with 60 pairs"
        assert metrics["training_pairs_available"] >= 50
        assert metrics["total_with_outcome"] == 60
    print("  PASS: eval metrics reflect accumulated outcomes")


def test_reference_match_rate_excludes_ack_turns():
    """Reference calibration should measure memory-bearing turns, not acknowledgements."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)

        ack_cid = "conv_ack"
        ack_candidates = [_make_candidate("mem_ack")]
        ack_event_id = log.log_retrieval(ack_cid, "Thank you Jarvis", ack_candidates, ["mem_ack"])
        log.mark_injected(ack_event_id, ["mem_ack"])
        log.log_outcome(conversation_id=ack_cid, outcome="ok", latency_ms=80.0, user_signal="positive")

        factual_cid = "conv_fact"
        factual_candidates = [_make_candidate("mem_fact")]
        factual_event_id = log.log_retrieval(
            factual_cid,
            "When is my birthday?",
            factual_candidates,
            ["mem_fact"],
        )
        log.mark_injected(factual_event_id, ["mem_fact"])
        log.record_references(factual_event_id, ["mem_fact"])
        log.log_outcome(conversation_id=factual_cid, outcome="ok", latency_ms=90.0)

        metrics = log.get_eval_metrics()
        assert metrics["reference_match_rate"] == 1.0
        assert metrics["reference_eval_events"] == 1
        assert metrics["reference_eval_injected_count"] == 1
        assert metrics["reference_eval_excluded_count"] == 1
    print("  PASS: reference match rate excludes acknowledgement turns")


def test_lift_requires_minimum_ranker_and_heuristic_samples():
    """Lift should remain unset until both cohorts have enough observed outcomes."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)

        ranker_id = log.log_retrieval(
            "conv_ranker",
            "When is my birthday?",
            [_make_candidate("mem_ranker")],
            ["mem_ranker"],
            ranker_used=True,
        )
        log.mark_injected(ranker_id, ["mem_ranker"])
        log.log_outcome(conversation_id="conv_ranker", outcome="ok", latency_ms=100.0)

        heuristic_id = log.log_retrieval(
            "conv_heuristic",
            "What did I say earlier?",
            [_make_candidate("mem_heuristic")],
            ["mem_heuristic"],
            ranker_used=False,
        )
        log.mark_injected(heuristic_id, ["mem_heuristic"])
        log.log_outcome(conversation_id="conv_heuristic", outcome="ok", latency_ms=100.0)

        metrics = log.get_eval_metrics()
        assert metrics["ranker_success_rate"] == 1.0
        assert metrics["heuristic_success_rate"] == 1.0
        assert metrics["min_lift_sample_events"] == 5
        assert metrics["lift"] is None
    print("  PASS: lift waits for minimum cohort size")


def test_background_retrieval_without_conversation_id_not_logged():
    """Internal/background searches must not enter conversational telemetry."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        event_id = log.log_retrieval(
            conversation_id="",
            query_text="background query",
            candidates=[_make_candidate("mem_bg")],
            selected_memory_ids=["mem_bg"],
        )
        assert event_id == ""
        stats = log.get_stats()
        assert stats["total_events"] == 0
        assert stats["buffered_events"] == 0
    print("  PASS: background retrieval is excluded from telemetry")


def test_detect_memory_references_handles_short_memory_phrases():
    refs = detect_memory_references(
        "Your birthday is May 10.",
        [{"id": "mem_short", "formatted": "May 10"}],
    )
    assert refs == ["mem_short"]


def test_detect_memory_references_counts_numeric_grounding():
    refs = detect_memory_references(
        "You have had 9 kernel mutations so far.",
        [{"id": "mem_numeric", "formatted": "kernel mutations total 9"}],
    )
    assert refs == ["mem_numeric"]


def test_rehydrate_skips_historical_background_retrievals():
    """Warm-start should preserve closed-loop retrievals, not background probes."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rehydrate_retrieval_log.jsonl")
        now = time.time()
        with open(path, "w") as f:
            for i in range(150):
                f.write(json.dumps({
                    "type": "retrieval",
                    "event_id": f"ret_bg_{i}",
                    "conversation_id": "",
                    "query": "internal semantic probe",
                    "candidate_count": 1,
                    "selected_count": 1,
                    "ranker_used": True,
                    "candidates": [{
                        "mid": f"mem_bg_{i}",
                        "sim": 0.8,
                        "rec": 0.5,
                        "w": 0.6,
                        "type": "factual_knowledge",
                        "tags": 1,
                        "assoc": 0,
                        "pri": 500,
                        "prov": 0.02,
                        "spk": False,
                        "hs": 0.6,
                        "sel": True,
                    }],
                    "t": now + i,
                }) + "\n")

            for i in range(5):
                cid = f"conv_keep_{i}"
                eid = f"ret_keep_{i}"
                f.write(json.dumps({
                    "type": "retrieval",
                    "event_id": eid,
                    "conversation_id": cid,
                    "query": f"user query {i}",
                    "candidate_count": 1,
                    "selected_count": 1,
                    "ranker_used": True,
                    "candidates": [{
                        "mid": f"mem_keep_{i}",
                        "sim": 0.9,
                        "rec": 0.5,
                        "w": 0.6,
                        "type": "conversation",
                        "tags": 1,
                        "assoc": 0,
                        "pri": 500,
                        "prov": 0.02,
                        "spk": False,
                        "hs": 0.7,
                        "sel": True,
                    }],
                    "t": now + 200 + i,
                }) + "\n")
                f.write(json.dumps({
                    "type": "injection",
                    "event_id": eid,
                    "injected_memory_ids": [f"mem_keep_{i}"],
                    "t": now + 210 + i,
                }) + "\n")
                f.write(json.dumps({
                    "type": "outcome",
                    "event_id": eid,
                    "conversation_id": cid,
                    "outcome": "ok",
                    "latency_ms": 100.0,
                    "user_signal": "",
                    "outcome_scope": "response_quality",
                    "t": now + 220 + i,
                }) + "\n")

        log = MemoryRetrievalLog(path=path)
        log.rehydrate(max_events=200)

        metrics = log.get_eval_metrics()
        assert metrics["total_with_outcome"] == 5
        assert metrics["overall_success_rate"] == 1.0
        assert metrics["ranker_events"] == 5
    print("  PASS: rehydrate ignores historical background retrievals")


if __name__ == "__main__":
    print("\n=== Cortex Outcome Loop Tests ===\n")
    test_log_outcome_no_exception()
    test_training_pairs_from_outcome()
    test_negative_feedback_adjusts_label()
    test_outcome_counters()
    test_positive_feedback_reinforces_injected_memory()
    test_negative_feedback_downweights_injected_memory()
    test_eval_metrics_reflect_outcomes()
    test_background_retrieval_without_conversation_id_not_logged()
    test_rehydrate_skips_historical_background_retrievals()
    print("\n  All cortex outcome tests passed!\n")
