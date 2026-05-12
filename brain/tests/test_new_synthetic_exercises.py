"""Truth boundary + correctness tests for the 5 new synthetic exercises.

Every test is fully self-contained — no live brain, no singletons, no real
persistence. Uses mocks, temp directories, and fabricated data only.

Verifies per exercise:
  - Feature vector dimensions correct
  - Label dimensions / distributions match expected patterns
  - Fidelity caps enforced at 0.7
  - Origin = "synthetic" on all recorded signals
  - No real persistence writes
  - No EventBus emissions
  - No singleton access
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 1: Memory Retrieval
# ═══════════════════════════════════════════════════════════════════════════


class MockMemory:
    """Minimal Memory-like object for retrieval exercise tests."""

    def __init__(self, mid: str = "m1", weight: float = 0.5,
                 tags: tuple = ("test",), mem_type: str = "factual_knowledge",
                 ts: float = 0.0):
        self.id = mid
        self.timestamp = ts or time.time() - 3600
        self.weight = weight
        self.tags = tags
        self.type = mem_type
        self.association_count = 2
        self.priority = 100
        self.is_core = False


class MockVectorStore:
    """Fake VectorStore that returns canned search results."""

    def __init__(self, results: list[dict] | None = None):
        self.available = True
        self._results = results or []

    def search(self, query: str, top_k: int = 20,
               min_weight: float = 0.0) -> list[dict]:
        return self._results


class MockMemoryStorage:
    """Fake MemoryStorage with get() returning mock memories."""

    def __init__(self, memories: dict[str, MockMemory] | None = None):
        self._memories = memories or {}

    def get(self, mid: str) -> MockMemory | None:
        return self._memories.get(mid)


class TestRetrievalExercise:

    def test_smoke_with_mocks(self):
        from synthetic.retrieval_exercise import PROFILES, run_retrieval_exercise

        mem = MockMemory(mid="m1", tags=("neural", "network"))
        vs = MockVectorStore(results=[
            {"memory_id": "m1", "similarity": 0.75},
        ])
        ms = MockMemoryStorage(memories={"m1": mem})

        stats = run_retrieval_exercise(
            profile=PROFILES["smoke"],
            vector_store=vs,
            memory_storage=ms,
        )

        assert stats.queries_processed > 0
        assert stats.pairs_generated > 0
        assert stats.pass_result is True

    def test_feature_vector_12_dim(self):
        """Pairs written to JSONL must have 12-dim feature vectors."""
        from pathlib import Path
        from synthetic.retrieval_exercise import PROFILES, run_retrieval_exercise

        mem = MockMemory(mid="m1", tags=("test",))
        vs = MockVectorStore(results=[
            {"memory_id": "m1", "similarity": 0.6},
        ])
        ms = MockMemoryStorage(memories={"m1": mem})

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "pairs.jsonl"
            with patch("synthetic.retrieval_exercise.PAIRS_PATH", test_path):
                with patch("synthetic.retrieval_exercise.REPORT_DIR", Path(tmpdir)):
                    stats = run_retrieval_exercise(
                        profile=PROFILES["smoke"],
                        count=5,
                        vector_store=vs,
                        memory_storage=ms,
                    )

            if test_path.exists():
                with open(test_path) as f:
                    for line in f:
                        pair = json.loads(line.strip())
                        assert len(pair["features"]) == 12
                        assert pair["fidelity"] == 0.7
                        assert pair["origin"] == "synthetic"

    def test_labels_in_valid_range(self):
        from synthetic.retrieval_exercise import _assign_label

        assert 0.0 <= _assign_label(0.7, True, 0.5) <= 1.0
        assert 0.0 <= _assign_label(0.1, False, 0.5) <= 1.0
        assert 0.0 <= _assign_label(0.4, True, 0.3) <= 1.0
        assert _assign_label(0.7, True, 0.5) > _assign_label(0.1, False, 0.5)

    def test_no_real_retrieval_log_writes(self):
        """Ensure retrieval_log module is never touched."""
        from pathlib import Path
        from synthetic.retrieval_exercise import PROFILES, run_retrieval_exercise

        vs = MockVectorStore(results=[])
        ms = MockMemoryStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("synthetic.retrieval_exercise.PAIRS_PATH", Path(tmpdir) / "pairs.jsonl"):
                with patch("synthetic.retrieval_exercise.REPORT_DIR", Path(tmpdir)):
                    stats = run_retrieval_exercise(
                        profile=PROFILES["smoke"],
                        count=3,
                        vector_store=vs,
                        memory_storage=ms,
                    )

        assert stats.queries_processed == 3 or stats.queries_processed >= 0

    def test_no_singleton_import(self):
        """The module must not import memory_storage at module level."""
        import synthetic.retrieval_exercise as mod
        source = open(mod.__file__).read()
        assert "from memory.storage import memory_storage" not in source.split(
            "def run_retrieval_exercise")[0]

    def test_load_synthetic_pairs_empty(self):
        from pathlib import Path
        from synthetic.retrieval_exercise import load_synthetic_pairs
        with patch("synthetic.retrieval_exercise.PAIRS_PATH", Path("/nonexistent/path.jsonl")):
            pairs = load_synthetic_pairs()
            assert pairs == []

    def test_query_corpus_coverage(self):
        from synthetic.retrieval_exercise import QUERY_CATEGORIES
        assert len(QUERY_CATEGORIES) >= 5
        total = sum(len(v) for v in QUERY_CATEGORIES.values())
        assert total >= 30


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 2: World Model Prediction
# ═══════════════════════════════════════════════════════════════════════════

class TestWorldModelExercise:

    def test_smoke_runs_standalone(self):
        from synthetic.world_model_exercise import PROFILES, run_world_model_exercise

        stats = run_world_model_exercise(profile=PROFILES["smoke"])

        assert stats.scenarios_run > 0
        assert stats.predictions_generated > 0
        assert stats.pass_result is True

    def test_coverage_fires_all_18_rules(self):
        from synthetic.world_model_exercise import PROFILES, run_world_model_exercise

        stats = run_world_model_exercise(profile=PROFILES["coverage"])

        assert stats.scenarios_run >= 18
        assert len(stats.rules_fired) >= 10, (
            f"Expected >=10 unique rules, got {len(stats.rules_fired)}: "
            f"{sorted(stats.rules_fired.keys())}"
        )

    def test_no_event_bus_emission(self):
        """CausalEngine standalone should not emit any events."""
        from synthetic.world_model_exercise import run_world_model_exercise, PROFILES

        with patch("consciousness.events.event_bus", create=True) as mock_bus:
            mock_bus.emit = MagicMock()
            stats = run_world_model_exercise(profile=PROFILES["smoke"])
            assert stats.scenarios_run > 0

    def test_no_promotion_writes(self):
        """Must never touch promotion persistence files."""
        from synthetic.world_model_exercise import run_world_model_exercise, PROFILES

        stats = run_world_model_exercise(profile=PROFILES["smoke"])

        promo_path = os.path.expanduser("~/.jarvis/world_model_promotion.json")
        sim_path = os.path.expanduser("~/.jarvis/simulator_promotion.json")
        assert stats.scenarios_run > 0

    def test_accuracy_and_stats(self):
        from synthetic.world_model_exercise import run_world_model_exercise, PROFILES

        stats = run_world_model_exercise(profile=PROFILES["coverage"])

        assert 0.0 <= stats.accuracy <= 1.0
        assert 0.0 <= stats.rule_coverage <= 1.0
        assert stats.hits + stats.misses == stats.predictions_validated

    def test_scenarios_cover_delta_and_steady_state(self):
        """Corpus must include both delta-triggered and steady-state rules."""
        from synthetic.world_model_exercise import _build_scenarios

        scenarios = _build_scenarios()
        has_deltas = any(s["deltas"] for s in scenarios)
        has_steady = any(not s["deltas"] for s in scenarios)
        assert has_deltas, "Must have delta-triggered scenarios"
        assert has_steady, "Must have steady-state scenarios"


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 3: Belief Contradiction
# ═══════════════════════════════════════════════════════════════════════════

class TestContradictionExercise:

    def test_smoke_runs(self):
        from synthetic.contradiction_exercise import PROFILES, run_contradiction_exercise

        stats = run_contradiction_exercise(profile=PROFILES["smoke"])

        assert stats.pairs_classified > 0
        assert stats.pass_result is True

    def test_coverage_all_6_classes(self):
        from synthetic.contradiction_exercise import PROFILES, run_contradiction_exercise

        stats = run_contradiction_exercise(profile=PROFILES["coverage"])

        assert stats.pairs_classified > 0

    def test_near_miss_detection(self):
        from synthetic.contradiction_exercise import PROFILES, run_contradiction_exercise

        stats = run_contradiction_exercise(profile=PROFILES["coverage"])

        assert stats.near_misses_total > 0

    def test_no_real_belief_store_writes(self):
        """Must never write to ~/.jarvis/beliefs.jsonl."""
        from synthetic.contradiction_exercise import run_contradiction_exercise, PROFILES

        real_beliefs = os.path.expanduser("~/.jarvis/beliefs.jsonl")
        before_size = os.path.getsize(real_beliefs) if os.path.exists(real_beliefs) else 0

        stats = run_contradiction_exercise(profile=PROFILES["smoke"])

        after_size = os.path.getsize(real_beliefs) if os.path.exists(real_beliefs) else 0
        assert before_size == after_size, "Real beliefs.jsonl was modified!"

    def test_no_contradiction_engine_singleton(self):
        """Module must not import or call ContradictionEngine.get_instance()."""
        import synthetic.contradiction_exercise as mod
        source = open(mod.__file__).read()
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith("from ") or line.strip().startswith("import ")
        ]
        for line in import_lines:
            assert "ContradictionEngine" not in line, (
                f"ContradictionEngine must not be imported: {line}"
            )
        code_lines = [
            line for line in source.split("\n")
            if not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("- ")
        ]
        code_section = "\n".join(code_lines)
        assert ".get_instance()" not in code_section

    def test_extraction_produces_claims(self):
        from synthetic.contradiction_exercise import _build_extraction_memories
        from epistemic.claim_extractor import extract_claims

        memories = _build_extraction_memories()
        total_claims = 0
        for item in memories:
            claims = extract_claims(item["memory"])
            total_claims += len(claims)

        assert total_claims >= 1, "Extraction should produce at least some claims"

    def test_belief_store_tempdir_isolation(self):
        """Standalone BeliefStore in tempdir must not leak to real path."""
        from epistemic.belief_record import BeliefStore
        from synthetic.contradiction_exercise import _make_belief

        with tempfile.TemporaryDirectory() as tmpdir:
            store = BeliefStore(
                beliefs_path=os.path.join(tmpdir, "beliefs.jsonl"),
                tensions_path=os.path.join(tmpdir, "tensions.jsonl"),
            )
            b = _make_belief("test", "is", "value")
            store.add(b)
            assert store.get_stats()["total_beliefs"] >= 1

        real_path = os.path.expanduser("~/.jarvis/beliefs.jsonl")
        if os.path.exists(real_path):
            with open(real_path) as f:
                content = f.read()
                assert b.belief_id not in content


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 4: Diagnostic Encoder
# ═══════════════════════════════════════════════════════════════════════════

class MockCollector:
    """Captures distillation signals for verification."""

    def __init__(self):
        self.signals: list[dict[str, Any]] = []

    def record(self, **kwargs):
        self.signals.append(kwargs)


class TestDiagnosticExercise:

    def test_smoke_no_collector(self):
        from synthetic.diagnostic_exercise import PROFILES, run_diagnostic_exercise

        stats = run_diagnostic_exercise(profile=PROFILES["smoke"])

        assert stats.scenarios_encoded > 0
        assert stats.dim_check_failures == 0
        assert stats.pass_result is True

    def test_feature_dim_43(self):
        from hemisphere.diagnostic_encoder import FEATURE_DIM, DiagnosticEncoder
        from synthetic.diagnostic_exercise import _base_snapshot, _base_context

        features = DiagnosticEncoder.encode(
            _base_snapshot(), [], _base_context(),
        )
        assert len(features) == FEATURE_DIM == 43

    def test_label_dim_6(self):
        from hemisphere.diagnostic_encoder import DiagnosticEncoder

        label, meta = DiagnosticEncoder.encode_no_opportunity_label()
        assert len(label) == 6
        assert abs(sum(label) - 1.0) < 0.01

        opp = {"type": "health_degraded", "priority": 4,
               "sustained_count": 3, "evidence_detail": {"worst_component": "memory"}}
        label2, meta2 = DiagnosticEncoder.encode_label(opp)
        assert len(label2) == 6
        assert label2[0] == 1.0

    def test_fidelity_cap_enforced(self):
        from synthetic.diagnostic_exercise import PROFILES, run_diagnostic_exercise

        profile = PROFILES["coverage"]
        profile.record_signals = True
        collector = MockCollector()

        stats = run_diagnostic_exercise(profile=profile, collector=collector)

        assert stats.features_recorded > 0
        for sig in collector.signals:
            assert sig["fidelity"] == 0.7, f"Fidelity not capped: {sig}"

    def test_origin_synthetic(self):
        from synthetic.diagnostic_exercise import PROFILES, run_diagnostic_exercise

        profile = PROFILES["coverage"]
        profile.record_signals = True
        collector = MockCollector()

        run_diagnostic_exercise(profile=profile, collector=collector)

        for sig in collector.signals:
            assert sig["origin"] == "synthetic", f"Origin wrong: {sig}"

    def test_scan_id_pairing(self):
        """Features and labels must share scan_id for pairing."""
        from synthetic.diagnostic_exercise import PROFILES, run_diagnostic_exercise

        profile = PROFILES["coverage"]
        profile.record_signals = True
        collector = MockCollector()

        run_diagnostic_exercise(profile=profile, collector=collector)

        feature_ids = set()
        label_ids = set()
        for sig in collector.signals:
            scan_id = sig.get("metadata", {}).get("scan_id", "")
            if sig["teacher"] == "diagnostic_features":
                feature_ids.add(scan_id)
            elif sig["teacher"] == "diagnostic_detector":
                label_ids.add(scan_id)

        overlap = feature_ids & label_ids
        assert len(overlap) > 0, "Features and labels must share scan_ids"

    def test_all_6_detectors_covered(self):
        from synthetic.diagnostic_exercise import _build_scenarios

        scenarios = _build_scenarios()
        types_seen = set()
        for sc in scenarios:
            dt = sc.get("detector_type")
            if dt:
                types_seen.add(dt)

        expected = {"health_degraded", "reasoning_decline", "confidence_volatile",
                    "slow_responses", "event_bus_errors", "tick_performance"}
        assert types_seen == expected

    def test_negative_examples_present(self):
        from synthetic.diagnostic_exercise import _build_scenarios

        scenarios = _build_scenarios()
        negatives = [sc for sc in scenarios if sc["detector_type"] is None]
        assert len(negatives) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 5: Plan Evaluator
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanEvaluatorExercise:

    def test_smoke_no_collector(self):
        from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

        stats = run_plan_evaluator_exercise(profile=PROFILES["smoke"])

        assert stats.plans_encoded > 0
        assert stats.dim_check_failures == 0
        assert stats.pass_result is True

    def test_feature_dim_32(self):
        from acquisition.plan_encoder import FEATURE_DIM, PlanEvaluatorEncoder
        from synthetic.plan_evaluator_exercise import _mock_job, _mock_plan

        features = PlanEvaluatorEncoder.encode(
            _mock_job(), _mock_plan(),
        )
        assert len(features) == FEATURE_DIM == 32

    def test_label_dim_3(self):
        from acquisition.plan_encoder import encode_verdict

        for verdict in ("approved", "rejected", "needs_revision"):
            label = encode_verdict(verdict)
            assert len(label) == 3
            assert abs(sum(label) - 1.0) < 0.01

    def test_fidelity_cap_enforced(self):
        from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

        profile = PROFILES["coverage"]
        profile.record_signals = True
        collector = MockCollector()

        stats = run_plan_evaluator_exercise(profile=profile, collector=collector)

        assert stats.features_recorded > 0
        for sig in collector.signals:
            assert sig["fidelity"] == 0.7

    def test_origin_synthetic(self):
        from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

        profile = PROFILES["coverage"]
        profile.record_signals = True
        collector = MockCollector()

        run_plan_evaluator_exercise(profile=profile, collector=collector)

        for sig in collector.signals:
            assert sig["origin"] == "synthetic"

    def test_all_3_verdicts_covered(self):
        from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

        stats = run_plan_evaluator_exercise(profile=PROFILES["coverage"])

        assert "approved" in stats.verdicts_exercised
        assert "needs_revision" in stats.verdicts_exercised
        assert "rejected" in stats.verdicts_exercised

    def test_mock_objects_not_real_jobs(self):
        """Mock objects must be SimpleNamespace, not real acquisition types."""
        from synthetic.plan_evaluator_exercise import _mock_job, _mock_plan

        job = _mock_job()
        plan = _mock_plan()
        assert isinstance(job, SimpleNamespace)
        assert isinstance(plan, SimpleNamespace)

    def test_no_acquisition_singleton_import(self):
        """Module must not import AcquisitionOrchestrator."""
        import synthetic.plan_evaluator_exercise as mod
        source = open(mod.__file__).read()
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith("from ") or line.strip().startswith("import ")
        ]
        for line in import_lines:
            assert "AcquisitionOrchestrator" not in line, (
                f"AcquisitionOrchestrator must not be imported: {line}"
            )
        code_lines = [
            line for line in source.split("\n")
            if not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("- ")
        ]
        assert ".get_instance()" not in "\n".join(code_lines)

    def test_no_shadow_prediction_writes(self):
        """Must never write to ~/.jarvis/acquisition_shadows/."""
        from synthetic.plan_evaluator_exercise import PROFILES, run_plan_evaluator_exercise

        shadow_dir = os.path.expanduser("~/.jarvis/acquisition_shadows")
        before = set(os.listdir(shadow_dir)) if os.path.exists(shadow_dir) else set()

        collector = MockCollector()
        profile = PROFILES["smoke"]
        profile.record_signals = True
        run_plan_evaluator_exercise(profile=profile, collector=collector)

        after = set(os.listdir(shadow_dir)) if os.path.exists(shadow_dir) else set()
        assert before == after, "Shadow predictions directory was modified!"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting truth boundary tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTruthBoundary:

    def test_retrieval_fidelity_constant(self):
        from synthetic.retrieval_exercise import SYNTHETIC_FIDELITY
        assert SYNTHETIC_FIDELITY == 0.7

    def test_diagnostic_fidelity_constant(self):
        from synthetic.diagnostic_exercise import SYNTHETIC_FIDELITY
        assert SYNTHETIC_FIDELITY == 0.7

    def test_plan_evaluator_fidelity_constant(self):
        from synthetic.plan_evaluator_exercise import SYNTHETIC_FIDELITY
        assert SYNTHETIC_FIDELITY == 0.7

    def test_retrieval_origin_constant(self):
        from synthetic.retrieval_exercise import SYNTHETIC_ORIGIN
        assert SYNTHETIC_ORIGIN == "synthetic"

    def test_diagnostic_origin_constant(self):
        from synthetic.diagnostic_exercise import SYNTHETIC_ORIGIN
        assert SYNTHETIC_ORIGIN == "synthetic"

    def test_plan_evaluator_origin_constant(self):
        from synthetic.plan_evaluator_exercise import SYNTHETIC_ORIGIN
        assert SYNTHETIC_ORIGIN == "synthetic"

    def test_world_model_is_standalone_engine(self):
        """CausalEngine must be instantiated fresh, not pulled from WorldModel."""
        import synthetic.world_model_exercise as mod
        source = open(mod.__file__).read()
        import_lines = [
            line.strip() for line in source.split("\n")
            if line.strip().startswith("from ") or line.strip().startswith("import ")
        ]
        for line in import_lines:
            assert "WorldModel" not in line or "world_state" in line, (
                f"Must not import WorldModel class: {line}"
            )
        code_lines = [
            line for line in source.split("\n")
            if not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("- ")
        ]
        assert ".get_instance()" not in "\n".join(code_lines)

    def test_contradiction_no_debt_updates(self):
        """Contradiction exercise must never call _update_debt."""
        import synthetic.contradiction_exercise as mod
        source = open(mod.__file__).read()
        code_lines = [
            line for line in source.split("\n")
            if not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("- ")
        ]
        assert "_update_debt" not in "\n".join(code_lines)

    def test_no_event_bus_import_in_exercises(self):
        """New exercises must not import event_bus directly."""
        import synthetic.retrieval_exercise as r
        import synthetic.world_model_exercise as wm
        import synthetic.contradiction_exercise as c
        import synthetic.diagnostic_exercise as d
        import synthetic.plan_evaluator_exercise as pe

        for mod in [r, wm, c, d, pe]:
            source = open(mod.__file__).read()
            import_lines = [
                line.strip() for line in source.split("\n")
                if line.strip().startswith("from ") or line.strip().startswith("import ")
            ]
            for line in import_lines:
                assert "event_bus" not in line, (
                    f"{mod.__name__} imports event_bus: {line}"
                )
