"""Tests for IntentionResolver (Stage 1) + IntentionDeliveryEncoder.

Validates:
  - Shadow-only start: no delivery until operator promotes
  - Heuristic logic: deliver_now, suppress, defer conditions
  - Controlled reason-code vocabulary
  - JSONL verdict logging
  - No Stage 0 mutation: resolver never changes registry outcomes
  - Feature encoder: 24-dim shape and bounds
  - Duplicate delivery suppression
  - Promotion ladder and rollback
  - Registry additions: get_recent_resolved_for_resolver, attach_resolver_verdict
"""

import json
import os
import tempfile
import time

import pytest

from cognition.intention_resolver import (
    IntentionResolver,
    ResolverSignal,
    ResolverVerdict,
    REASON_CODES,
    STAGE_ORDER,
    VERDICTS_PATH,
)
from cognition.intention_registry import IntentionRegistry, IntentionRecord
from hemisphere.intention_delivery_encoder import (
    IntentionDeliveryFeatures,
    encode,
    encode_label,
    FEATURE_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(**overrides):
    defaults = dict(
        intention_id="int_test001",
        backing_job_id="job_test001",
        commitment_type="follow_up",
        outcome="resolved",
        age_s=60.0,
        result_summary="Research completed successfully.",
        speaker_present=True,
        active_conversation=True,
        topic_overlap=0.5,
        quarantine_pressure=0.0,
        soul_integrity=0.9,
        proactive_cooldown_remaining=0.0,
        friction_rate=0.0,
        same_speaker_present=True,
    )
    defaults.update(overrides)
    return ResolverSignal(**defaults)


def _fresh_resolver():
    return IntentionResolver()


def _fresh_registry():
    reg = IntentionRegistry()
    reg._loaded = True
    return reg


# ---------------------------------------------------------------------------
# Core: shadow-only, heuristics, vocabulary
# ---------------------------------------------------------------------------

class TestResolverShadowOnly:
    def test_starts_shadow_only(self):
        r = _fresh_resolver()
        assert r.get_stage() == "shadow_only"
        assert not r.can_deliver()

    def test_cannot_deliver_in_shadow(self):
        r = _fresh_resolver()
        for stage in ("shadow_only", "shadow_advisory"):
            r.set_stage(stage)
            assert not r.can_deliver()

    def test_can_deliver_after_promotion(self):
        r = _fresh_resolver()
        r.set_stage("advisory_canary")
        assert r.can_deliver()
        r.set_stage("advisory")
        assert r.can_deliver()
        r.set_stage("active")
        assert r.can_deliver()


class TestHeuristicLogic:
    def test_deliver_now_fresh_present(self):
        r = _fresh_resolver()
        signal = _make_signal(age_s=60, speaker_present=True, active_conversation=True, topic_overlap=0.5)
        verdict = r.evaluate(signal)
        assert verdict.decision == "deliver_now"
        assert verdict.reason_code == "fresh_actionable_result"
        assert verdict.score > 0.3

    def test_suppress_stale(self):
        r = _fresh_resolver()
        signal = _make_signal(age_s=8000)
        verdict = r.evaluate(signal)
        assert verdict.decision == "suppress"
        assert verdict.reason_code == "stale_low_relevance"

    def test_defer_governance_blocked(self):
        r = _fresh_resolver()
        signal = _make_signal(quarantine_pressure=0.8)
        verdict = r.evaluate(signal)
        assert verdict.decision == "defer"
        assert verdict.reason_code == "governance_blocked"

    def test_defer_low_soul_integrity(self):
        r = _fresh_resolver()
        signal = _make_signal(soul_integrity=0.3)
        verdict = r.evaluate(signal)
        assert verdict.decision == "defer"
        assert verdict.reason_code == "governance_blocked"

    def test_defer_cooldown(self):
        r = _fresh_resolver()
        signal = _make_signal(proactive_cooldown_remaining=10.0)
        verdict = r.evaluate(signal)
        assert verdict.decision == "defer"
        assert verdict.reason_code == "cooldown_defer"

    def test_speaker_gone_queues_next_turn(self):
        r = _fresh_resolver()
        signal = _make_signal(age_s=60, speaker_present=False)
        verdict = r.evaluate(signal)
        assert verdict.decision == "deliver_on_next_turn"
        assert verdict.reason_code == "fresh_speaker_gone"

    def test_failed_result_informational(self):
        r = _fresh_resolver()
        signal = _make_signal(outcome="failed", friction_rate=0.0)
        verdict = r.evaluate(signal)
        assert verdict.decision == "deliver_on_next_turn"
        assert verdict.reason_code == "failed_result_informational"

    def test_failed_result_noisy_suppressed(self):
        r = _fresh_resolver()
        signal = _make_signal(outcome="failed", friction_rate=0.2)
        verdict = r.evaluate(signal)
        assert verdict.decision == "suppress"
        assert verdict.reason_code == "failed_result_noisy"

    def test_conversation_inactive_queues_next_turn(self):
        r = _fresh_resolver()
        signal = _make_signal(
            age_s=600,
            speaker_present=True,
            active_conversation=False,
            topic_overlap=0.0,
            same_speaker_present=False,
        )
        verdict = r.evaluate(signal)
        assert verdict.decision == "deliver_on_next_turn"
        assert verdict.reason_code == "conversation_inactive_wait"


class TestDuplicateSuppression:
    def test_same_intention_not_delivered_twice(self):
        r = _fresh_resolver()
        signal = _make_signal(intention_id="int_dup01")
        v1 = r.evaluate(signal)
        assert v1.decision == "deliver_now"
        v2 = r.evaluate(signal)
        assert v2.decision == "suppress"
        assert v2.reason_code == "duplicate_of_earlier_delivery"


class TestReasonCodeVocabulary:
    def test_all_verdicts_use_known_codes(self):
        r = _fresh_resolver()
        signals = [
            _make_signal(),
            _make_signal(age_s=8000),
            _make_signal(quarantine_pressure=0.8),
            _make_signal(proactive_cooldown_remaining=10),
            _make_signal(speaker_present=False),
            _make_signal(outcome="failed"),
            _make_signal(outcome="failed", friction_rate=0.2),
        ]
        for s in signals:
            v = r.evaluate(s)
            assert v.reason_code in REASON_CODES, f"Unknown code: {v.reason_code}"

    def test_invalid_reason_code_raises(self):
        with pytest.raises(ValueError, match="Unknown reason_code"):
            ResolverVerdict(
                intention_id="x",
                decision="suppress",
                score=0.0,
                reason_code="invalid_code_not_in_vocab",
            )


# ---------------------------------------------------------------------------
# Verdict logging
# ---------------------------------------------------------------------------

class TestVerdictLogging:
    def test_verdict_logged_to_jsonl(self, tmp_path, monkeypatch):
        log_path = tmp_path / "verdicts.jsonl"
        monkeypatch.setattr("cognition.intention_resolver.VERDICTS_PATH", log_path)
        r = _fresh_resolver()
        r.evaluate(_make_signal())
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "resolver_verdict"
        assert entry["stage"] == "shadow_only"
        assert "signal" in entry
        assert "verdict" in entry


# ---------------------------------------------------------------------------
# No Stage 0 mutation
# ---------------------------------------------------------------------------

class TestNoStageMutation:
    def test_resolver_never_changes_registry_outcomes(self):
        reg = _fresh_registry()
        iid = reg.register(
            utterance="test",
            commitment_phrase="I'll look into that",
            commitment_type="follow_up",
            backing_job_id="job_001",
            backing_job_kind="research",
        )
        assert iid is not None
        reg.resolve(backing_job_id="job_001", outcome="resolved", reason="done")
        rec_before = reg.get_by_id(iid)
        assert rec_before is not None
        outcome_before = rec_before.outcome

        r = _fresh_resolver()
        signal = _make_signal(intention_id=iid)
        r.evaluate(signal)

        rec_after = reg.get_by_id(iid)
        assert rec_after.outcome == outcome_before


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------

class TestFeatureEncoder:
    def test_encode_24dim(self):
        features = IntentionDeliveryFeatures()
        vec = encode(features)
        assert len(vec) == FEATURE_DIM
        assert all(0.0 <= v <= 1.0 for v in vec), f"Out of bounds: {vec}"

    def test_encode_with_real_values(self):
        features = IntentionDeliveryFeatures(
            commitment_type="follow_up",
            outcome_success=True,
            age_since_resolution_s=120.0,
            expected_duration_s=300.0,
            result_summary_len=250,
            turn_idx_same_speaker_since_commit=5,
            time_since_last_user_message_s=30.0,
            same_topic_keyword_overlap=0.6,
            speaker_present_now=True,
            addressee_mode_score=0.8,
            active_conversation=True,
            user_mood_valence=0.3,
            proactive_cooldown_remaining_norm=0.0,
            quarantine_pressure=0.1,
            contradiction_debt=0.05,
            soul_integrity=0.85,
            belief_graph_integrity=0.9,
            autonomy_level=2,
            is_interruptible_mode=True,
            is_reflective_mode=False,
            recent_user_friction_rate=0.05,
            fractal_recall_surfaced_recent=False,
        )
        vec = encode(features)
        assert len(vec) == FEATURE_DIM
        assert all(0.0 <= v <= 1.0 for v in vec)
        assert vec[0] == 1.0  # follow_up one-hot
        assert vec[1] == 0.0
        assert vec[4] == 1.0  # outcome_success

    def test_commitment_type_onehot(self):
        types = ["follow_up", "deferred_action", "future_work", "task_started"]
        for i, ct in enumerate(types):
            features = IntentionDeliveryFeatures(commitment_type=ct)
            vec = encode(features)
            for j in range(4):
                expected = 1.0 if j == i else 0.0
                assert vec[j] == expected, f"type={ct}, dim={j}: expected {expected}, got {vec[j]}"

    def test_unknown_commitment_type_zeros(self):
        features = IntentionDeliveryFeatures(commitment_type="unknown")
        vec = encode(features)
        assert vec[0:4] == [0.0, 0.0, 0.0, 0.0]

    def test_encode_label_deliver(self):
        label = encode_label("deliver_now")
        assert label == [1.0, 0.0, 0.0, 0.0]
        label2 = encode_label("deliver_on_next_turn")
        assert label2 == [1.0, 0.0, 0.0, 0.0]

    def test_encode_label_suppress(self):
        label = encode_label("suppress")
        assert label == [0.0, 0.0, 1.0, 0.0]

    def test_encode_label_defer_uniform(self):
        label = encode_label("defer")
        assert label == [0.25, 0.25, 0.25, 0.25]


# ---------------------------------------------------------------------------
# Promotion ladder and rollback
# ---------------------------------------------------------------------------

class TestPromotionLadder:
    def test_stage_order(self):
        assert STAGE_ORDER == [
            "shadow_only", "shadow_advisory",
            "advisory_canary", "advisory", "active",
        ]

    def test_set_stage(self):
        r = _fresh_resolver()
        assert r.set_stage("advisory_canary")
        assert r.get_stage() == "advisory_canary"

    def test_set_invalid_stage(self):
        r = _fresh_resolver()
        assert not r.set_stage("nonexistent")
        assert r.get_stage() == "shadow_only"

    def test_rollback_demotes_one_rung(self):
        r = _fresh_resolver()
        r.set_stage("advisory")
        new = r.rollback()
        assert new == "advisory_canary"
        new = r.rollback()
        assert new == "shadow_advisory"

    def test_rollback_at_bottom_stays(self):
        r = _fresh_resolver()
        new = r.rollback()
        assert new == "shadow_only"


# ---------------------------------------------------------------------------
# Registry additions
# ---------------------------------------------------------------------------

class TestRegistryAdditions:
    def test_get_recent_resolved_for_resolver(self):
        reg = _fresh_registry()
        for i in range(3):
            reg.register(
                utterance=f"u{i}",
                commitment_phrase=f"phrase{i}",
                commitment_type="follow_up",
                backing_job_id=f"job_{i}",
                backing_job_kind="research",
            )
            reg.resolve(backing_job_id=f"job_{i}", outcome="resolved", reason="ok")

        results = reg.get_recent_resolved_for_resolver(n=10)
        assert len(results) == 3
        assert results[0].backing_job_id == "job_2"

    def test_attach_resolver_verdict(self):
        reg = _fresh_registry()
        iid = reg.register(
            utterance="test",
            commitment_phrase="I'll check",
            commitment_type="follow_up",
            backing_job_id="job_att",
            backing_job_kind="research",
        )
        reg.resolve(backing_job_id="job_att", outcome="resolved", reason="done")

        payload = {"decision": "suppress", "score": 0.1, "reason_code": "stale_low_relevance"}
        assert reg.attach_resolver_verdict(iid, payload)
        rec = reg.get_by_id(iid)
        assert rec.metadata["resolver_verdict"] == payload

    def test_attach_verdict_write_once(self):
        reg = _fresh_registry()
        iid = reg.register(
            utterance="test",
            commitment_phrase="I'll check",
            commitment_type="follow_up",
            backing_job_id="job_wo",
            backing_job_kind="research",
        )
        reg.resolve(backing_job_id="job_wo", outcome="resolved", reason="done")

        assert reg.attach_resolver_verdict(iid, {"first": True})
        assert not reg.attach_resolver_verdict(iid, {"second": True})
        rec = reg.get_by_id(iid)
        assert rec.metadata["resolver_verdict"] == {"first": True}

    def test_attach_verdict_missing_id_returns_false(self):
        reg = _fresh_registry()
        assert not reg.attach_resolver_verdict("nonexistent", {"x": 1})


# ---------------------------------------------------------------------------
# Shadow metrics
# ---------------------------------------------------------------------------

class TestShadowMetrics:
    def test_initial_metrics(self):
        r = _fresh_resolver()
        m = r.get_shadow_metrics()
        assert m["shadow_total"] == 0
        assert m["shadow_accuracy"] == 0.0
        assert not m["sufficient_data"]

    def test_record_outcomes(self):
        r = _fresh_resolver()
        for _ in range(30):
            r.record_shadow_outcome(True)
        for _ in range(20):
            r.record_shadow_outcome(False)
        m = r.get_shadow_metrics()
        assert m["shadow_total"] == 50
        assert m["sufficient_data"]
        assert abs(m["shadow_accuracy"] - 0.6) < 0.01


# ---------------------------------------------------------------------------
# Status endpoint shape
# ---------------------------------------------------------------------------

class TestStatusShape:
    def test_status_has_required_keys(self):
        r = _fresh_resolver()
        s = r.get_status()
        assert "stage" in s
        assert "total_evaluated" in s
        assert "verdict_counts" in s
        assert "reason_counts" in s
        assert "recent_verdicts" in s
        assert "shadow_metrics" in s
        assert "uptime_s" in s
        assert s["stage"] == "shadow_only"
        assert s["total_evaluated"] == 0
