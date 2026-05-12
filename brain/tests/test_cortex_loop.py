"""Tests for the closed cortex training loop.

Covers:
  - Stale-data training guard
  - Three-case reference labels (injected+referenced, injected+not-referenced, not-injected+referenced)
  - Provenance-weighted label modifiers (tie-breaker only, max ±0.05)
  - Validated prediction wiring (reinforcement/eviction → record_validated_prediction)
  - Salience advisory gate (4-gate rollback-safe check)
  - build_creation_features() shared helper
  - Dashboard observability fields
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# build_creation_features
# ---------------------------------------------------------------------------

class TestBuildCreationFeatures:
    def test_produces_11_dim_vector(self):
        from memory.lifecycle_log import build_creation_features
        features = build_creation_features(
            source="conversation",
            initial_weight=0.7,
            user_present=True,
            mode="conversational",
            memory_count=100,
            speaker_known=True,
            conversation_active=True,
            memory_type="conversation",
            payload_length=200,
            provenance="user_claim",
        )
        assert len(features) == 11
        assert all(isinstance(f, float) for f in features)

    def test_unknown_source_maps_to_max(self):
        from memory.lifecycle_log import build_creation_features
        features = build_creation_features(
            source="never_seen_source",
            initial_weight=0.5,
            user_present=False,
            mode="passive",
            memory_count=0,
            speaker_known=False,
            conversation_active=False,
            memory_type="conversation",
            payload_length=100,
            provenance="unknown",
        )
        assert features[0] == 8 / 8.0  # unknown source -> ordinal 8

    def test_provenance_mapped_correctly(self):
        from memory.lifecycle_log import build_creation_features
        for prov, expected_ord in [("observed", 0), ("user_claim", 1), ("external_source", 4), ("unknown", 8)]:
            features = build_creation_features(
                source="conversation", initial_weight=0.5, user_present=False,
                mode="passive", memory_count=0, speaker_known=False,
                conversation_active=False, memory_type="conversation",
                payload_length=100, provenance=prov,
            )
            assert features[10] == expected_ord / 8.0, f"provenance={prov}"


# ---------------------------------------------------------------------------
# Stale-data training guard
# ---------------------------------------------------------------------------

class TestStaleDataGuard:
    def test_skips_retrain_when_no_new_data(self):
        from consciousness.consciousness_system import ConsciousnessSystem
        cs = ConsciousnessSystem()
        cs._last_ranker_pair_count = 100
        cs._last_salience_pair_count = 200

        mock_ranker = MagicMock()
        mock_ranker.is_ready.return_value = True

        mock_salience = MagicMock()
        mock_salience.is_ready.return_value = True

        # Simulate having 105 pairs (only 5 new, below threshold of 20)
        mock_pairs = [MagicMock(features=[0.0]*12, label=1.0) for _ in range(105)]
        mock_spairs = [MagicMock(features=[0.0]*11, store_label=0.8, weight_label=0.5, decay_label=0.3) for _ in range(210)]

        with patch("memory.retrieval_log.memory_retrieval_log") as mock_ret_log, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_life_log, \
             patch("memory.ranker.get_memory_ranker", return_value=mock_ranker), \
             patch("memory.salience.get_salience_model", return_value=mock_salience), \
             patch("consciousness.operations.ops_tracker"):
            mock_ret_log.get_training_pairs.return_value = mock_pairs
            mock_life_log.get_salience_training_pairs.return_value = mock_spairs

            result = cs._run_cortex_training()

        mock_ranker.train_from_pairs.assert_not_called()
        mock_salience.train_from_pairs.assert_not_called()
        assert "stale data" in cs._last_ranker_skip
        assert "stale data" in cs._last_salience_skip

    def test_trains_when_enough_new_data(self):
        from consciousness.consciousness_system import ConsciousnessSystem
        cs = ConsciousnessSystem()
        cs._last_ranker_pair_count = 50
        cs._last_salience_pair_count = 100

        mock_ranker = MagicMock()
        mock_ranker.train_from_pairs.return_value = {"loss": 0.1, "accuracy": 0.9, "pairs": 80}

        mock_pairs = [MagicMock(features=[0.0]*12, label=1.0) for _ in range(80)]

        with patch("memory.retrieval_log.memory_retrieval_log") as mock_ret_log, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_life_log, \
             patch("memory.ranker.get_memory_ranker", return_value=mock_ranker), \
             patch("memory.salience.get_salience_model", return_value=None), \
             patch("consciousness.operations.ops_tracker"):
            mock_ret_log.get_training_pairs.return_value = mock_pairs
            mock_life_log.get_salience_training_pairs.return_value = []

            result = cs._run_cortex_training()

        mock_ranker.train_from_pairs.assert_called_once()
        assert cs._last_ranker_pair_count == 80


# ---------------------------------------------------------------------------
# Three-case reference labels
# ---------------------------------------------------------------------------

class TestReferenceLabels:
    def _make_log_with_data(self, injected_ids, referenced_ids, outcome="ok", user_signal=""):
        from memory.retrieval_log import (
            MemoryRetrievalLog, CandidateRecord, RetrievalEvent, RetrievalOutcome,
        )
        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log._lock = __import__("threading").Lock()
        log._recent_references = {}
        log._reference_without_injection_count = 0

        eid = "evt_test"
        candidates = []
        for mid in ["mem_a", "mem_b", "mem_c"]:
            candidates.append(CandidateRecord(
                memory_id=mid, similarity=0.8, recency_score=0.5, weight=0.6,
                memory_type="conversation", tag_count=2, association_count=1,
                priority=500, provenance_boost=0.02, speaker_match=False,
                heuristic_score=0.5, selected=mid in injected_ids or mid == "mem_c",
                injected=mid in injected_ids,
            ))

        event = RetrievalEvent(
            event_id=eid, conversation_id="conv_1", query_text="test",
            candidates=candidates,
            selected_memory_ids=[c.memory_id for c in candidates if c.selected],
            injected_memory_ids=injected_ids,
            timestamp=time.time(), ranker_used=False,
        )

        from collections import OrderedDict
        log._recent_events = OrderedDict({eid: event})
        log._recent_outcomes = OrderedDict({eid: RetrievalOutcome(
            event_id=eid, conversation_id="conv_1", outcome=outcome,
            latency_ms=100.0, user_signal=user_signal, timestamp=time.time(),
        )})
        if referenced_ids:
            log._recent_references[eid] = set(referenced_ids)

        return log

    def test_injected_and_referenced_gets_1_0(self):
        log = self._make_log_with_data(
            injected_ids=["mem_a"],
            referenced_ids=["mem_a"],
        )
        pairs = log.get_training_pairs()
        label_a = next(p.label for p in pairs if p.candidate.memory_id == "mem_a")
        assert label_a >= 1.0  # 1.0 + possible provenance boost, clamped

    def test_injected_not_referenced_gets_0_8(self):
        log = self._make_log_with_data(
            injected_ids=["mem_a"],
            referenced_ids=[],
        )
        pairs = log.get_training_pairs()
        label_a = next(p.label for p in pairs if p.candidate.memory_id == "mem_a")
        assert 0.75 <= label_a <= 0.85  # 0.8 ± provenance modifier

    def test_negative_signal_protected_for_referenced(self):
        log = self._make_log_with_data(
            injected_ids=["mem_a"],
            referenced_ids=["mem_a"],
            user_signal="negative",
        )
        pairs = log.get_training_pairs()
        label_a = next(p.label for p in pairs if p.candidate.memory_id == "mem_a")
        assert label_a >= 0.8  # protected floor

    def test_negative_signal_reduces_unreferenced(self):
        log = self._make_log_with_data(
            injected_ids=["mem_a"],
            referenced_ids=[],
            user_signal="negative",
        )
        pairs = log.get_training_pairs()
        label_a = next(p.label for p in pairs if p.candidate.memory_id == "mem_a")
        assert label_a < 0.7  # 0.8 - 0.2 = 0.6

    def test_not_injected_but_referenced_no_credit(self):
        log = self._make_log_with_data(
            injected_ids=["mem_a"],
            referenced_ids=["mem_b"],  # mem_b was NOT injected
        )
        pairs = log.get_training_pairs()
        label_b = next(p.label for p in pairs if p.candidate.memory_id == "mem_b")
        assert label_b <= 0.55  # selected but not injected: 0.5 + possible modifiers
        assert log._reference_without_injection_count >= 1

    def test_reference_loaded_in_rehydrate(self):
        import json
        import tempfile
        from memory.retrieval_log import MemoryRetrievalLog

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            eid = "ret_test123"
            f.write(json.dumps({
                "type": "retrieval", "event_id": eid, "conversation_id": "c1",
                "query": "q", "candidates": [{
                    "mid": "m1", "sim": 0.9, "rec": 0.5, "w": 0.7,
                    "type": "conversation", "tags": 1, "assoc": 0,
                    "pri": 500, "prov": 0.02, "spk": False, "hs": 0.5, "sel": True,
                }], "t": time.time(),
            }) + "\n")
            f.write(json.dumps({
                "type": "reference", "event_id": eid,
                "referenced_memory_ids": ["m1"],
                "t": time.time(),
            }) + "\n")
            f.flush()
            path = f.name

        log = MemoryRetrievalLog(path=path)
        log.rehydrate(max_events=100)
        assert eid in log._recent_references
        assert "m1" in log._recent_references[eid]


# ---------------------------------------------------------------------------
# Provenance label modifiers
# ---------------------------------------------------------------------------

class TestProvenanceLabels:
    def _make_candidate_log(self, provenance_boost, outcome="ok"):
        from memory.retrieval_log import (
            MemoryRetrievalLog, CandidateRecord, RetrievalEvent, RetrievalOutcome,
        )
        from collections import OrderedDict
        import threading

        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log._lock = threading.Lock()
        log._recent_references = {}
        log._reference_without_injection_count = 0

        eid = "evt_prov"
        candidate = CandidateRecord(
            memory_id="mem_x", similarity=0.8, recency_score=0.5, weight=0.6,
            memory_type="factual_knowledge", tag_count=2, association_count=1,
            priority=500, provenance_boost=provenance_boost, speaker_match=False,
            heuristic_score=0.5, selected=True, injected=True,
        )
        event = RetrievalEvent(
            event_id=eid, conversation_id="conv_1", query_text="test",
            candidates=[candidate],
            selected_memory_ids=["mem_x"],
            injected_memory_ids=["mem_x"],
            timestamp=time.time(), ranker_used=False,
        )
        log._recent_events = OrderedDict({eid: event})
        log._recent_outcomes = OrderedDict({eid: RetrievalOutcome(
            event_id=eid, conversation_id="conv_1", outcome=outcome,
            latency_ms=100.0, user_signal="", timestamp=time.time(),
        )})
        return log

    def test_high_provenance_positive_boost(self):
        log = self._make_candidate_log(provenance_boost=0.10, outcome="ok")
        pairs = log.get_training_pairs()
        label = pairs[0].label
        # base 0.8 (injected + not referenced) + 0.05 provenance = 0.85
        assert label > 0.8

    def test_low_provenance_negative_penalty(self):
        log = self._make_candidate_log(provenance_boost=0.02, outcome="error")
        pairs = log.get_training_pairs()
        label = pairs[0].label
        # base 0.3 (injected + error) - 0.05 provenance = 0.25
        assert label < 0.3

    def test_labels_clamped_to_01(self):
        log = self._make_candidate_log(provenance_boost=0.12, outcome="ok")
        log._recent_references["evt_prov"] = {"mem_x"}
        pairs = log.get_training_pairs()
        label = pairs[0].label
        # 1.0 (referenced+injected) + 0.05 = 1.05, should clamp to 1.0
        assert label == 1.0


# ---------------------------------------------------------------------------
# Validated prediction wiring
# ---------------------------------------------------------------------------

class TestValidatedPredictions:
    def test_reinforcement_calls_record_validated(self):
        from memory.lifecycle_log import MemoryLifecycleLog, CreationRecord
        import threading

        log = MemoryLifecycleLog.__new__(MemoryLifecycleLog)
        log._lock = threading.Lock()
        log._initialized = True
        log._total_events = 0
        log._path = __import__("pathlib").Path("/dev/null")

        from collections import OrderedDict
        record = CreationRecord(
            memory_id="m1", memory_type="conversation",
            initial_weight=0.5, initial_decay_rate=0.001,
            tag_count=1, payload_length=100, source="conversation",
            user_present=True, speaker_known=False, conversation_active=True,
            mode="conversational", memory_count_at_creation=50,
            created_at=time.time() - 7200, peak_weight=0.5,
            salience_advised=True,
        )
        log._creations = OrderedDict({"m1": record})

        with patch("memory.salience.get_salience_model") as mock_get:
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            log.log_reinforced("m1", boost=0.1, new_weight=0.6)

        mock_model.record_validated_prediction.assert_called_once()

    def test_eviction_calls_record_validated(self):
        from memory.lifecycle_log import MemoryLifecycleLog, CreationRecord
        import threading

        log = MemoryLifecycleLog.__new__(MemoryLifecycleLog)
        log._lock = threading.Lock()
        log._initialized = True
        log._total_events = 0
        log._path = __import__("pathlib").Path("/dev/null")

        from collections import OrderedDict
        record = CreationRecord(
            memory_id="m2", memory_type="conversation",
            initial_weight=0.3, initial_decay_rate=0.01,
            tag_count=0, payload_length=50, source="conversation",
            user_present=False, speaker_known=False, conversation_active=False,
            mode="passive", memory_count_at_creation=100,
            created_at=time.time() - 7200, peak_weight=0.3,
            salience_advised=True,
        )
        log._creations = OrderedDict({"m2": record})

        with patch("memory.salience.get_salience_model") as mock_get:
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            log.log_evicted("m2", final_weight=0.05)

        mock_model.record_validated_prediction.assert_called_once()

    def test_no_call_when_not_salience_advised(self):
        from memory.lifecycle_log import MemoryLifecycleLog, CreationRecord
        import threading

        log = MemoryLifecycleLog.__new__(MemoryLifecycleLog)
        log._lock = threading.Lock()
        log._initialized = True
        log._total_events = 0
        log._path = __import__("pathlib").Path("/dev/null")

        from collections import OrderedDict
        record = CreationRecord(
            memory_id="m3", memory_type="conversation",
            initial_weight=0.5, initial_decay_rate=0.001,
            tag_count=1, payload_length=100, source="conversation",
            user_present=True, speaker_known=False, conversation_active=True,
            mode="conversational", memory_count_at_creation=50,
            created_at=time.time() - 7200, peak_weight=0.5,
            salience_advised=False,
        )
        log._creations = OrderedDict({"m3": record})

        with patch("memory.salience.get_salience_model") as mock_get:
            mock_model = MagicMock()
            mock_get.return_value = mock_model
            log.log_reinforced("m3", boost=0.1, new_weight=0.6)

        mock_model.record_validated_prediction.assert_not_called()


# ---------------------------------------------------------------------------
# Salience advisory gate
# ---------------------------------------------------------------------------

class TestSalienceAdvisoryGate:
    def _make_engine(self):
        from consciousness.engine import ConsciousnessEngine
        engine = ConsciousnessEngine.__new__(ConsciousnessEngine)
        engine._phase = "PROCESSING"
        engine._is_user_present = True
        engine._current_speaker = "David"
        engine._current_mode = "conversational"
        engine._salience_advisory_count = 0
        engine._salience_gate_fail_count = 0
        return engine

    def _make_memory(self):
        from consciousness.events import Memory
        return Memory(
            id="test_mem",
            type="conversation",
            payload="test payload content here",
            weight=0.7,
            decay_rate=0.005,
            tags=("test",),
            timestamp=time.time(),
            priority=500,
        )

    def _make_data(self):
        from memory.core import CreateMemoryData
        return CreateMemoryData(
            type="conversation",
            payload="test payload content here",
            weight=0.7,
            tags=["test"],
            provenance="user_claim",
        )

    def test_gate_fails_when_model_not_ready(self):
        engine = self._make_engine()
        memory = self._make_memory()
        data = self._make_data()

        with patch("memory.salience.get_salience_model") as mock_get:
            mock_model = MagicMock()
            mock_model.is_ready.return_value = False
            mock_get.return_value = mock_model

            result = engine._apply_salience_advisory(memory, data)

        assert result is False
        assert engine._salience_gate_fail_count == 1

    def test_gate_fails_when_insufficient_validations(self):
        engine = self._make_engine()
        memory = self._make_memory()
        data = self._make_data()

        with patch("memory.salience.get_salience_model") as mock_get:
            mock_model = MagicMock()
            mock_model.is_ready.return_value = True
            mock_model._validated_predictions = 10  # below 30
            mock_get.return_value = mock_model

            result = engine._apply_salience_advisory(memory, data)

        assert result is False
        assert engine._salience_gate_fail_count == 1

    def test_gate_fails_when_high_weight_error(self):
        engine = self._make_engine()
        memory = self._make_memory()
        data = self._make_data()

        with patch("memory.salience.get_salience_model") as mock_get, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_ll:
            mock_model = MagicMock()
            mock_model.is_ready.return_value = True
            mock_model._validated_predictions = 50
            mock_get.return_value = mock_model
            mock_ll.get_effectiveness_metrics.return_value = {"weight_error": 0.5}

            result = engine._apply_salience_advisory(memory, data)

        assert result is False
        assert engine._salience_gate_fail_count == 1

    def test_gate_fails_on_large_delta(self):
        engine = self._make_engine()
        memory = self._make_memory()
        data = self._make_data()

        with patch("memory.salience.get_salience_model") as mock_get, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_ll, \
             patch("memory.lifecycle_log.build_creation_features", return_value=[0.0]*11):
            mock_model = MagicMock()
            mock_model.is_ready.return_value = True
            mock_model._validated_predictions = 50
            mock_model.advise_weight.return_value = (0.2, 0.005)  # delta=0.5, too big
            mock_get.return_value = mock_model
            mock_ll.get_effectiveness_metrics.return_value = {"weight_error": 0.1}

            result = engine._apply_salience_advisory(memory, data)

        assert result is False
        assert engine._salience_gate_fail_count == 1

    def test_gate_passes_all_conditions(self):
        engine = self._make_engine()
        memory = self._make_memory()
        data = self._make_data()

        with patch("memory.salience.get_salience_model") as mock_get, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_ll, \
             patch("memory.lifecycle_log.build_creation_features", return_value=[0.0]*11):
            mock_model = MagicMock()
            mock_model.is_ready.return_value = True
            mock_model._validated_predictions = 50
            mock_model.advise_weight.return_value = (0.65, 0.004)  # within delta cap
            mock_get.return_value = mock_model
            mock_ll.get_effectiveness_metrics.return_value = {"weight_error": 0.15}

            result = engine._apply_salience_advisory(memory, data)

        assert result is True
        assert memory.weight == 0.65
        assert memory.decay_rate == 0.004
        assert engine._salience_advisory_count == 1
        assert engine._salience_gate_fail_count == 0


# ---------------------------------------------------------------------------
# Dashboard observability
# ---------------------------------------------------------------------------

class TestDashboardObservability:
    def test_cortex_stats_include_training_freshness(self):
        from consciousness.consciousness_system import ConsciousnessSystem
        cs = ConsciousnessSystem()
        cs._last_ranker_pair_count = 75
        cs._last_salience_pair_count = 150

        mock_engine = MagicMock()
        mock_engine._salience_advisory_count = 5
        mock_engine._salience_gate_fail_count = 12
        cs._engine_ref = mock_engine

        with patch("memory.ranker.get_memory_ranker", return_value=None), \
             patch("memory.salience.get_salience_model", return_value=None), \
             patch("memory.retrieval_log.memory_retrieval_log") as mock_rl, \
             patch("memory.lifecycle_log.memory_lifecycle_log") as mock_ll:
            mock_rl.get_stats.return_value = {}
            mock_rl.get_eval_metrics.return_value = {}
            mock_ll.get_stats.return_value = {}
            mock_ll.get_effectiveness_metrics.return_value = {}

            stats = cs.get_cortex_stats()

        assert stats["training_status"]["last_train_ranker_pairs"] == 75
        assert stats["training_status"]["last_train_salience_pairs"] == 150
        assert stats["salience_advisory"]["active"] is True
        assert stats["salience_advisory"]["advisory_count"] == 5
        assert stats["salience_advisory"]["gate_fail_count"] == 12

    def test_eval_metrics_include_reference_fields(self):
        from memory.retrieval_log import (
            MemoryRetrievalLog, CandidateRecord, RetrievalEvent, RetrievalOutcome,
        )
        from collections import OrderedDict
        import threading

        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log._lock = threading.Lock()
        log._recent_references = {}
        log._reference_without_injection_count = 3

        eid = "evt_eval"
        candidate = CandidateRecord(
            memory_id="m1", similarity=0.8, recency_score=0.5, weight=0.6,
            memory_type="conversation", tag_count=2, association_count=1,
            priority=500, provenance_boost=0.06, speaker_match=False,
            heuristic_score=0.5, selected=True, injected=True,
        )
        event = RetrievalEvent(
            event_id=eid, conversation_id="c1", query_text="test",
            candidates=[candidate],
            selected_memory_ids=["m1"],
            injected_memory_ids=["m1"],
            timestamp=time.time(), ranker_used=False,
        )
        log._recent_events = OrderedDict({eid: event})
        log._recent_outcomes = OrderedDict({eid: RetrievalOutcome(
            event_id=eid, conversation_id="c1", outcome="ok",
            latency_ms=100.0, timestamp=time.time(),
        )})
        log._recent_references[eid] = {"m1"}

        metrics = log.get_eval_metrics()
        assert "reference_match_rate" in metrics
        assert metrics["reference_match_rate"] == 1.0
        assert metrics["reference_without_injection_count"] == 3
        assert "provenance_weighted_success_rate" in metrics
        assert metrics["provenance_weighted_success_rate"] is not None


# ---------------------------------------------------------------------------
# record_references method
# ---------------------------------------------------------------------------

class TestRecordReferences:
    def test_record_references_populates_dict(self):
        from memory.retrieval_log import MemoryRetrievalLog
        from collections import OrderedDict
        import threading

        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log._lock = threading.Lock()
        log._recent_references = OrderedDict()

        log.record_references("evt_1", ["m1", "m2"])
        assert "evt_1" in log._recent_references
        assert log._recent_references["evt_1"] == {"m1", "m2"}

        log.record_references("evt_1", ["m3"])
        assert log._recent_references["evt_1"] == {"m1", "m2", "m3"}

    def test_record_references_noop_on_empty(self):
        from memory.retrieval_log import MemoryRetrievalLog
        import threading

        log = MemoryRetrievalLog.__new__(MemoryRetrievalLog)
        log._lock = threading.Lock()
        log._recent_references = {}

        log.record_references("", ["m1"])
        log.record_references("evt_1", [])
        assert len(log._recent_references) == 0
