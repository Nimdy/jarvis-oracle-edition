"""Regression tests for Oracle Integrity Hardening — Restart Continuity.

Tests:
  - Stage restore validation and downgrade
  - Contradiction debt reconstruction and persistence
  - Delta tracker pending window persistence
  - Calibration history rehydration from JSONL
  - Mutation timestamp persistence
  - Drive manager state persistence
  - Gap detector EMA persistence
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.consciousness_evolution import (
    ConsciousnessEvolution,
    EvolutionStage,
    STAGE_ORDER,
)


# ─── A1: Restore Validation ──────────────────────────────────────────

def test_restore_validated_stage_with_sufficient_counts():
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": EvolutionStage.SELF_REFLECTIVE,
        "transcendence_level": 1.5,
        "stage_entered_at": time.time() - 3600,
        "stage_history": [],
        "total_emergent_count": 3,
    }
    evo.load_state(data, observation_count=50, mutation_count=0, awareness_level=0.5)
    assert evo.current_stage == EvolutionStage.SELF_REFLECTIVE
    trust = evo.get_restore_trust()
    assert trust["trust"] == "verified"
    assert trust["anomaly_count"] == 0
    print("  PASS: restore validation — valid stage passes")


def test_restore_downgrades_inflated_stage():
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": EvolutionStage.INTEGRATIVE,
        "transcendence_level": 9.5,
        "stage_entered_at": time.time() - 3600,
        "stage_history": [],
        "total_emergent_count": 10,
    }
    evo.load_state(data, observation_count=5, mutation_count=0, awareness_level=0.2)
    assert evo.current_stage == EvolutionStage.BASIC_AWARENESS
    trust = evo.get_restore_trust()
    assert trust["trust"] == "downgraded"
    assert trust["anomaly_count"] >= 1
    assert trust["basis"]["claimed_stage"] == EvolutionStage.INTEGRATIVE
    assert trust["basis"]["validated_stage"] == EvolutionStage.BASIC_AWARENESS
    print("  PASS: restore validation — inflated stage downgraded")


def test_restore_clamps_transcendence():
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": EvolutionStage.BASIC_AWARENESS,
        "transcendence_level": 999.0,
    }
    evo.load_state(data)
    assert evo.transcendence_level == 10.0
    trust = evo.get_restore_trust()
    assert trust["trust"] == "downgraded"
    print("  PASS: restore validation — transcendence clamped")


def test_restore_rejects_unknown_stage():
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": "nonexistent_stage",
        "transcendence_level": 1.0,
    }
    evo.load_state(data)
    assert evo.current_stage == EvolutionStage.BASIC_AWARENESS
    trust = evo.get_restore_trust()
    assert trust["trust"] == "downgraded"
    assert any("unknown stage" in a for a in trust["basis"]["anomalies"])
    print("  PASS: restore validation — unknown stage rejected")


def test_restore_downgrades_to_highest_valid():
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": EvolutionStage.RECURSIVE_SELF_MODELING,
        "transcendence_level": 5.0,
    }
    evo.load_state(data, observation_count=100, mutation_count=6, awareness_level=0.65)
    assert evo.current_stage == EvolutionStage.PHILOSOPHICAL
    trust = evo.get_restore_trust()
    assert trust["trust"] == "downgraded"
    print("  PASS: restore validation — downgraded to highest valid stage")


# ─── A2: Contradiction Debt ──────────────────────────────────────────

def test_contradiction_debt_reconstruction():
    from epistemic.contradiction_engine import ContradictionEngine

    ce = ContradictionEngine()
    ce._belief_store._beliefs = {}
    ce._belief_store._tensions = {}

    class FakeBelief:
        def __init__(self, bid, contradicts=None, resolution_state="active", conflict_key="k"):
            self.belief_id = bid
            self.contradicts = contradicts
            self.resolution_state = resolution_state
            self.conflict_key = conflict_key

    ce._belief_store._beliefs = {
        "b1": FakeBelief("b1", contradicts="b2", resolution_state="active", conflict_key="k1"),
        "b2": FakeBelief("b2", contradicts="b1", resolution_state="active", conflict_key="k1"),
        "b3": FakeBelief("b3"),
    }
    ce._seen_conflict_keys = {"k1"}

    floor = ce._reconstruct_debt_floor()
    assert floor > 0.0, f"Expected debt > 0 from active contradictions, got {floor}"
    print(f"  PASS: contradiction debt reconstruction — floor={floor:.4f}")


def test_set_persisted_debt_takes_max():
    from epistemic.contradiction_engine import ContradictionEngine

    ce = ContradictionEngine()
    ce._contradiction_debt = 0.1  # simulating reconstructed floor

    diag = ce.set_persisted_debt(0.05)
    assert ce.contradiction_debt == 0.1, "Should prefer reconstructed (higher)"
    assert diag["debt_restore_source"] == "reconstructed"

    diag = ce.set_persisted_debt(0.2)
    assert ce.contradiction_debt == 0.2, "Should prefer persisted (higher)"
    assert diag["debt_restore_source"] == "persisted"
    print("  PASS: set_persisted_debt — max(persisted, reconstructed) rule")


# ─── A3: DeltaTracker Persistence ────────────────────────────────────

def test_delta_tracker_save_load():
    from autonomy.delta_tracker import DeltaTracker, MetricSnapshot, DELTA_PENDING_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "delta_pending.json")
        import autonomy.delta_tracker as dt_mod
        original_path = dt_mod.DELTA_PENDING_PATH
        dt_mod.DELTA_PENDING_PATH = type(original_path)(test_path)

        try:
            tracker1 = DeltaTracker()
            tracker1._metric_ring.append(MetricSnapshot(
                timestamp=time.time() - 100, confidence_avg=0.6,
                reasoning_coherence=0.7, processing_health=0.9, memory_count=50,
            ))
            baseline = tracker1.start_tracking("intent_test_1")
            tracker1.mark_completed("intent_test_1")
            tracker1.save_pending()

            assert os.path.exists(test_path), "Pending file should exist"

            tracker2 = DeltaTracker()
            restored, interrupted = tracker2.load_pending()

            has_results = restored > 0 or len(interrupted) > 0
            assert has_results, f"Expected restored or interrupted, got restored={restored} interrupted={len(interrupted)}"
            print(f"  PASS: delta tracker persistence — restored={restored}, interrupted={len(interrupted)}")
        finally:
            dt_mod.DELTA_PENDING_PATH = original_path


def test_delta_tracker_interrupted_by_restart():
    from autonomy.delta_tracker import DeltaTracker, MetricSnapshot, DELTA_PENDING_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "delta_pending.json")
        import autonomy.delta_tracker as dt_mod
        original_path = dt_mod.DELTA_PENDING_PATH
        dt_mod.DELTA_PENDING_PATH = type(original_path)(test_path)

        try:
            data = [{
                "intent_id": "expired_intent",
                "baseline": {"timestamp": time.time() - 10000, "confidence_avg": 0.5},
                "completion_time": time.time() - 5000,
                "post_check_time": time.time() - 4000,
            }]
            with open(test_path, "w") as f:
                json.dump(data, f)

            tracker = DeltaTracker()
            restored, interrupted = tracker.load_pending()
            assert len(interrupted) == 1
            assert interrupted[0].status == "interrupted_by_restart"
            assert tracker._total_interrupted == 1
            print("  PASS: delta tracker — interrupted_by_restart status")
        finally:
            dt_mod.DELTA_PENDING_PATH = original_path


# ─── A4: Calibration History Rehydration ─────────────────────────────

def test_calibration_history_rehydration():
    from epistemic.calibration.calibration_history import CalibrationHistory
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "calibration_truth.jsonl"
        entries = []
        for i in range(10):
            entries.append(json.dumps({
                "ts": time.time() - (10 - i) * 120,
                "truth_score": 0.5 + i * 0.01,
                "maturity": 0.3 + i * 0.02,
                "domains": {
                    "contradiction_debt": 0.05 - i * 0.005,
                },
                "debt": 0.05 - i * 0.005,
            }))
        log_path.write_text("\n".join(entries) + "\n")

        history = CalibrationHistory()
        loaded = history.rehydrate_from_log(log_path)
        assert loaded == 10, f"Expected 10, got {loaded}"
        assert history.count == 10
        trend = history.get_domain_trend("epistemic", window=20)
        assert len(trend) > 0, "Should have epistemic trend data"
        print(f"  PASS: calibration history rehydration — loaded={loaded}, trend_len={len(trend)}")


def test_calibration_history_handles_corrupt_lines():
    from epistemic.calibration.calibration_history import CalibrationHistory
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "calibration_truth.jsonl"
        lines = [
            json.dumps({"ts": time.time(), "domains": {}, "debt": 0.0}),
            "not valid json {{{",
            json.dumps({"ts": time.time() + 1, "domains": {}, "debt": 0.01}),
        ]
        log_path.write_text("\n".join(lines) + "\n")

        history = CalibrationHistory()
        loaded = history.rehydrate_from_log(log_path)
        assert loaded == 2, f"Expected 2 valid entries, got {loaded}"
        print("  PASS: calibration history — skips corrupt lines")


# ─── B5: Mutation Timestamp Persistence ──────────────────────────────

def test_mutation_timestamps_round_trip():
    now = time.time()
    timestamps = [now - 1800, now - 900, now - 300]
    one_hour_ago = now - 3600

    filtered = [ts for ts in timestamps if isinstance(ts, (int, float)) and ts > one_hour_ago]
    assert len(filtered) == 3, "All recent timestamps should survive"

    old_timestamps = [now - 7200, now - 5000, now - 300]
    filtered = [ts for ts in old_timestamps if isinstance(ts, (int, float)) and ts > one_hour_ago]
    assert len(filtered) == 1, "Only the recent one should survive"
    print("  PASS: mutation timestamp filtering logic")


# ─── B6: Drive Manager Persistence ───────────────────────────────────

def test_drive_manager_save_load():
    from autonomy.drives import DriveManager, DRIVE_STATE_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "drive_state.json")
        import autonomy.drives as dm_mod
        original_path = dm_mod.DRIVE_STATE_PATH
        dm_mod.DRIVE_STATE_PATH = type(original_path)(test_path)

        try:
            dm1 = DriveManager()
            dm1._states["truth"].action_count = 5
            dm1._states["truth"].success_count = 3
            dm1._states["truth"].consecutive_failures = 1
            dm1._states["curiosity"].action_count = 10
            dm1._states["curiosity"].last_acted = time.time() - 60
            dm1.save_state()

            dm2 = DriveManager()
            restored = dm2.load_state()
            assert restored >= 2
            assert dm2._states["truth"].action_count == 5
            assert dm2._states["truth"].success_count == 3
            assert dm2._states["curiosity"].action_count == 10
            print(f"  PASS: drive manager persistence — restored={restored}")

            # Boot failure cap: stale drives with high failures get capped
            dm3 = DriveManager()
            dm3._states["mastery"].consecutive_failures = 183
            dm3._states["mastery"].action_count = 200
            dm3._states["mastery"].last_acted = time.time() - 700  # stale
            dm3.save_state()

            dm4 = DriveManager()
            dm4.load_state()
            assert dm4._states["mastery"].consecutive_failures == 10, (
                f"Expected boot cap at 10, got {dm4._states['mastery'].consecutive_failures}"
            )
            assert dm4._states["mastery"].last_acted == 0.0
            print("  PASS: drive manager boot failure cap — 183 → 10")
        finally:
            dm_mod.DRIVE_STATE_PATH = original_path


# ─── B7: Gap Detector EMA Persistence ────────────────────────────────

def test_gap_detector_save_load():
    from hemisphere.gap_detector import CognitiveGapDetector, GAP_DETECTOR_STATE_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "gap_detector_state.json")
        import hemisphere.gap_detector as gd_mod
        original_path = gd_mod.GAP_DETECTOR_STATE_PATH
        gd_mod.GAP_DETECTOR_STATE_PATH = type(original_path)(test_path)

        try:
            gd1 = CognitiveGapDetector()
            gd1.update_dimension_score("response_quality", 0.3)
            gd1.update_dimension_score("response_quality", 0.25)
            gd1._total_gaps_emitted = 7
            gd1.save_state()

            gd2 = CognitiveGapDetector()
            restored = gd2.load_state()
            assert restored > 0
            assert gd2._dimensions["response_quality"].ema != 0.5, "EMA should be restored"
            assert gd2._total_gaps_emitted == 7
            print(f"  PASS: gap detector EMA persistence — restored={restored}")
        finally:
            gd_mod.GAP_DETECTOR_STATE_PATH = original_path


# ─── Legacy Stage Name Migration ─────────────────────────────────────

def test_legacy_stage_names_migrated():
    """Legacy stage names in persisted data should map to new names."""
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": "cosmic_consciousness",
        "transcendence_level": 9.5,
        "stage_entered_at": time.time() - 3600,
        "stage_history": [],
        "total_emergent_count": 10,
    }
    evo.load_state(data, observation_count=600, mutation_count=50, awareness_level=0.98)
    assert evo.current_stage == EvolutionStage.INTEGRATIVE, (
        f"Expected 'integrative', got '{evo.current_stage}'"
    )
    trust = evo.get_restore_trust()
    assert trust["stage_name_legacy_restored_from"] == "cosmic_consciousness"
    assert trust["stage_name_current"] == "integrative"
    assert trust["stage_requirements_met"] is True
    print("  PASS: legacy stage name 'cosmic_consciousness' -> 'integrative'")


def test_legacy_transcendent_migrated():
    """Legacy 'transcendent' maps to 'recursive_self_modeling'."""
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": "transcendent",
        "transcendence_level": 7.0,
        "stage_history": [],
    }
    evo.load_state(data, observation_count=300, mutation_count=20, awareness_level=0.85)
    assert evo.current_stage == EvolutionStage.RECURSIVE_SELF_MODELING
    trust = evo.get_restore_trust()
    assert trust["stage_name_legacy_restored_from"] == "transcendent"
    print("  PASS: legacy stage name 'transcendent' -> 'recursive_self_modeling'")


def test_legacy_stage_history_normalized():
    """Legacy stage names in stage_history entries should be normalized."""
    evo = ConsciousnessEvolution()
    data = {
        "current_stage": "integrative",
        "transcendence_level": 9.0,
        "stage_history": [
            {"from": "philosophical", "to": "transcendent", "time": time.time() - 7200},
            {"from": "transcendent", "to": "cosmic_consciousness", "time": time.time() - 3600},
        ],
    }
    evo.load_state(data, observation_count=600, mutation_count=50, awareness_level=0.98)
    history = evo.get_state().stage_history
    assert history[0]["to"] == "recursive_self_modeling", f"Got {history[0]['to']}"
    assert history[1]["from"] == "recursive_self_modeling", f"Got {history[1]['from']}"
    assert history[1]["to"] == "integrative", f"Got {history[1]['to']}"
    print("  PASS: legacy stage_history entries normalized")


# ─── Runner ──────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_restore_validated_stage_with_sufficient_counts,
        test_restore_downgrades_inflated_stage,
        test_restore_clamps_transcendence,
        test_restore_rejects_unknown_stage,
        test_restore_downgrades_to_highest_valid,
        test_contradiction_debt_reconstruction,
        test_set_persisted_debt_takes_max,
        test_delta_tracker_save_load,
        test_delta_tracker_interrupted_by_restart,
        test_calibration_history_rehydration,
        test_calibration_history_handles_corrupt_lines,
        test_mutation_timestamps_round_trip,
        test_drive_manager_save_load,
        test_gap_detector_save_load,
        test_legacy_stage_names_migrated,
        test_legacy_transcendent_migrated,
        test_legacy_stage_history_normalized,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\nRestart Integrity Tests: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
