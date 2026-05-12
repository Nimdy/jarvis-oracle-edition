"""Tests for the Process Verification Layer (PVL).

Covers:
  - Contract definitions: completeness, uniqueness, group membership
  - ProcessVerifier: event-based, snapshot-based, compound checks
  - Mode gating: applicability based on seen modes
  - Coverage computation: pass/fail/awaiting/not_applicable math
  - Session-once semantics for one-time events
  - Dashboard adapter PVL panel shape
"""

from __future__ import annotations

import pytest
from jarvis_eval.process_contracts import (
    ALL_CONTRACTS,
    PROCESS_GROUPS,
    PLAYBOOK_DAY_MAP,
    ProcessContract,
    get_contracts_by_group,
    get_contracts_for_playbook_day,
)
from jarvis_eval.process_verifier import (
    ProcessVerifier,
    ProcessVerdict,
    VerificationResult,
    _resolve_dotted_key,
)


# ── Contract Definitions ────────────────────────────────────────────


class TestContractDefinitions:

    def test_contract_count_minimum(self):
        assert len(ALL_CONTRACTS) >= 50, f"Expected >=50 contracts, got {len(ALL_CONTRACTS)}"

    def test_unique_contract_ids(self):
        ids = [c.contract_id for c in ALL_CONTRACTS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_groups_have_metadata(self):
        groups_used = {c.group for c in ALL_CONTRACTS}
        for g in groups_used:
            assert g in PROCESS_GROUPS, f"Group '{g}' missing from PROCESS_GROUPS"

    def test_group_orders_unique(self):
        orders = [v["order"] for v in PROCESS_GROUPS.values()]
        assert len(orders) == len(set(orders)), "Duplicate group orders"

    def test_every_contract_has_label(self):
        for c in ALL_CONTRACTS:
            assert c.label, f"Contract {c.contract_id} missing label"

    def test_event_contracts_have_event_type(self):
        for c in ALL_CONTRACTS:
            if c.method == "event":
                assert c.event_type, f"Event contract {c.contract_id} missing event_type"

    def test_snapshot_contracts_have_source_and_key(self):
        for c in ALL_CONTRACTS:
            if c.method == "snapshot":
                assert c.snapshot_source, f"Snapshot contract {c.contract_id} missing source"
                assert c.snapshot_key, f"Snapshot contract {c.contract_id} missing key"

    def test_compound_contracts_have_both(self):
        for c in ALL_CONTRACTS:
            if c.method == "compound":
                assert c.event_type, f"Compound contract {c.contract_id} missing event_type"
                assert c.snapshot_source, f"Compound contract {c.contract_id} missing source"

    def test_all_groups_populated(self):
        by_group = get_contracts_by_group()
        assert len(by_group) == 23, f"Expected 23 groups, got {len(by_group)}: {sorted(by_group.keys())}"
        assert "intention_truth" in by_group, "intention_truth group (Stage 0) must be registered"

    def test_playbook_day_map_covers_7_days(self):
        assert set(PLAYBOOK_DAY_MAP.keys()) == {1, 2, 3, 4, 5, 6, 7}

    def test_playbook_groups_exist(self):
        for day, groups in PLAYBOOK_DAY_MAP.items():
            for g in groups:
                assert g in PROCESS_GROUPS, f"Day {day} references unknown group '{g}'"

    def test_get_contracts_for_playbook_day(self):
        day1 = get_contracts_for_playbook_day(1)
        assert len(day1) > 0
        groups_in_day1 = {c.group for c in day1}
        assert "voice_pipeline" in groups_in_day1
        assert "identity_pipeline" in groups_in_day1

    def test_frozen_contracts(self):
        c = ALL_CONTRACTS[0]
        with pytest.raises(AttributeError):
            c.label = "changed"


# ── Resolve Dotted Key ──────────────────────────────────────────────


class TestResolveDottedKey:

    def test_simple_key(self):
        assert _resolve_dotted_key({"a": 5}, "a") == 5

    def test_dotted_key(self):
        assert _resolve_dotted_key({"a": {"b": 10}}, "a.b") == 10

    def test_deep_key(self):
        assert _resolve_dotted_key({"a": {"b": {"c": 42}}}, "a.b.c") == 42

    def test_missing_key(self):
        assert _resolve_dotted_key({"a": 1}, "b") is None

    def test_missing_nested(self):
        assert _resolve_dotted_key({"a": {"b": 1}}, "a.c") is None

    def test_non_dict_intermediate(self):
        assert _resolve_dotted_key({"a": 5}, "a.b") is None


# ── ProcessVerifier: Event-Based Contracts ──────────────────────────


class TestVerifierEventChecks:

    def _make_verifier(self) -> ProcessVerifier:
        return ProcessVerifier()

    def test_pass_on_event_seen(self):
        v = self._make_verifier()
        events = [{"event_type": "memory:write", "mode": "gestation"}]
        snapshots: dict = {}
        result = v.verify(events, snapshots, "gestation")
        memory_written = next(
            (vd for vd in result.verdicts if vd.contract_id == "memory_written"), None
        )
        assert memory_written is not None
        assert memory_written.status == "pass"

    def test_fail_when_event_not_seen(self):
        v = self._make_verifier()
        events = [{"event_type": "some_other:event", "mode": "conversational"}]
        result = v.verify(events, {}, "conversational")
        wake = next(
            (vd for vd in result.verdicts if vd.contract_id == "wake_word_detected"), None
        )
        assert wake is not None
        assert wake.status == "fail"

    def test_awaiting_when_no_events(self):
        v = self._make_verifier()
        result = v.verify([], {}, "conversational")
        for vd in result.verdicts:
            if vd.status not in ("not_applicable", "awaiting"):
                pytest.fail(f"Expected awaiting or n/a, got {vd.status} for {vd.contract_id}")

    def test_rare_event_contract_can_remain_awaiting_after_other_events(self):
        v = self._make_verifier()
        result = v.verify(
            [{"event_type": "memory:write", "mode": "passive"}],
            {},
            "passive",
        )
        contradiction = next(
            (vd for vd in result.verdicts if vd.contract_id == "contradiction_scanned"), None
        )
        assert contradiction is not None
        assert contradiction.status == "awaiting"

    def test_session_once_persists(self):
        v = self._make_verifier()
        v.verify(
            [{"event_type": "gestation:started", "mode": "gestation"}],
            {}, "gestation",
        )
        result2 = v.verify([], {}, "gestation")
        gest = next(
            (vd for vd in result2.verdicts if vd.contract_id == "gestation_started"), None
        )
        assert gest is not None
        assert gest.status == "pass", "session_once contract should persist"


# ── ProcessVerifier: Snapshot-Based Contracts ───────────────────────


class TestVerifierSnapshotChecks:

    def test_pass_on_snapshot_above_threshold(self):
        v = ProcessVerifier()
        snapshots = {"memory": {"total": 5}}
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots, "conversational",
        )
        mem_count = next(
            (vd for vd in result.verdicts if vd.contract_id == "memory_count_growing"), None
        )
        assert mem_count is not None
        assert mem_count.status == "pass"

    def test_fail_on_snapshot_zero(self):
        v = ProcessVerifier()
        snapshots = {"memory": {"total": 0}}
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots, "conversational",
        )
        mem_count = next(
            (vd for vd in result.verdicts if vd.contract_id == "memory_count_growing"), None
        )
        assert mem_count is not None
        assert mem_count.status == "fail"

    def test_pass_on_snapshot_exactly_at_threshold(self):
        """Value == min_val should pass (>= semantics, not strict >)."""
        v = ProcessVerifier()
        snapshots = {"memory": {"total": 1}}
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots, "conversational",
        )
        mem_count = next(
            (vd for vd in result.verdicts if vd.contract_id == "memory_count_growing"), None
        )
        assert mem_count is not None
        assert mem_count.status == "pass"

    def test_awaiting_when_source_missing(self):
        v = ProcessVerifier()
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            {}, "conversational",
        )
        mem_count = next(
            (vd for vd in result.verdicts if vd.contract_id == "memory_count_growing"), None
        )
        assert mem_count is not None
        assert mem_count.status == "awaiting"

    def test_dotted_key_in_snapshot(self):
        v = ProcessVerifier()
        snapshots = {"study_telemetry": {"cumulative_studied": 3}}
        result = v.verify(
            [{"event_type": "dummy", "mode": "gestation"}],
            snapshots, "gestation",
        )
        llm = next(
            (vd for vd in result.verdicts if vd.contract_id == "llm_extraction_used"), None
        )
        assert llm is not None
        assert llm.status == "pass"

    def test_skill_learning_contracts_pass_from_persisted_snapshots(self):
        v = ProcessVerifier()
        snapshots = {
            "skills": {"total": 14},
            "learning_jobs": {
                "total_count": 1,
                "completed_count": 1,
                "phase_transition_count": 5,
            },
        }
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots,
            "conversational",
        )
        expected = (
            "skill_registered",
            "learning_job_started",
            "job_phase_advanced",
            "skill_learning_completed",
        )
        for cid in expected:
            verdict = next((vd for vd in result.verdicts if vd.contract_id == cid), None)
            assert verdict is not None
            assert verdict.status == "pass"

    def test_evolution_analyzed_passes_from_consciousness_snapshot(self):
        v = ProcessVerifier()
        snapshots = {
            "consciousness": {"emergent_behavior_count": 3, "stage": "integrative"},
        }
        result = v.verify(
            [{"event_type": "dummy", "mode": "reflective"}],
            snapshots,
            "reflective",
        )
        evolution = next(
            (vd for vd in result.verdicts if vd.contract_id == "evolution_analyzed"), None
        )
        assert evolution is not None
        assert evolution.status == "pass"

    def test_language_promotion_contracts_pass_from_persisted_snapshot_metrics(self):
        v = ProcessVerifier()
        snapshots = {
            "language": {
                "total_examples": 140,
                "native_usage_rate": 0.82,
                "fail_closed_rate": 0.10,
                "provenance_coverage": 0.93,
                "gate_color_code": 2,
                "promotion_total_evaluations": 35,
                "promotion_red_classes": 0,
                "promotion_red_quality_classes": 0,
                "runtime_unpromoted_live_attempts": 0,
                "runtime_live_red_classes": 0,
            },
        }
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots,
            "conversational",
        )
        expected = (
            "lang_gate_not_red",
            "lang_promotion_evals_recorded",
            "lang_red_pressure_controlled",
            "lang_runtime_unpromoted_live_zero",
            "lang_runtime_live_red_zero",
        )
        for cid in expected:
            verdict = next((vd for vd in result.verdicts if vd.contract_id == cid), None)
            assert verdict is not None
            assert verdict.status == "pass"

    def test_matrix_jobs_observed_passes_with_completed_job(self):
        v = ProcessVerifier()
        snapshots = {
            "matrix": {
                "active_matrix_jobs": 0,
                "completed_matrix_jobs": 1,
                "matrix_jobs_observed": 1,
            },
        }
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots,
            "conversational",
        )
        matrix_jobs = next(
            (vd for vd in result.verdicts if vd.contract_id == "matrix_active_jobs"),
            None,
        )
        assert matrix_jobs is not None
        assert matrix_jobs.status == "pass"

    def test_matrix_specialists_not_applicable_without_deep_learning_mode(self):
        v = ProcessVerifier()
        snapshots = {"matrix": {"specialist_count": 0}}
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots,
            "conversational",
        )
        specialists = next(
            (vd for vd in result.verdicts if vd.contract_id == "matrix_specialists_exist"),
            None,
        )
        assert specialists is not None
        assert specialists.status == "not_applicable"

    def test_matrix_dl_requested_can_await_when_not_seen(self):
        v = ProcessVerifier()
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            {},
            "conversational",
        )
        matrix_dl = next(
            (vd for vd in result.verdicts if vd.contract_id == "matrix_dl_requested"),
            None,
        )
        assert matrix_dl is not None
        assert matrix_dl.status == "awaiting"

    def test_snapshot_max_failure_reports_above_max_evidence(self):
        v = ProcessVerifier()
        snapshots = {
            "language": {
                "promotion_red_quality_classes": 5,
            },
        }
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            snapshots,
            "conversational",
        )
        red_pressure = next(
            (vd for vd in result.verdicts if vd.contract_id == "lang_red_pressure_controlled"),
            None,
        )
        assert red_pressure is not None
        assert red_pressure.status == "fail"
        assert "above max" in red_pressure.evidence


# ── ProcessVerifier: Mode Gating ────────────────────────────────────


class TestVerifierModeGating:

    def test_not_applicable_when_mode_not_seen(self):
        v = ProcessVerifier()
        result = v.verify(
            [{"event_type": "perception:wake_word", "mode": "gestation"}],
            {}, "gestation",
        )
        wake = next(
            (vd for vd in result.verdicts if vd.contract_id == "wake_word_detected"), None
        )
        assert wake is not None
        assert wake.status == "not_applicable", (
            "Voice contracts require post-gestation modes"
        )

    def test_applicable_after_mode_seen(self):
        v = ProcessVerifier()
        v.verify(
            [{"event_type": "mode:change", "mode": "conversational"}],
            {}, "conversational",
        )
        result = v.verify(
            [{"event_type": "perception:wake_word", "mode": "conversational"}],
            {}, "conversational",
        )
        wake = next(
            (vd for vd in result.verdicts if vd.contract_id == "wake_word_detected"), None
        )
        assert wake is not None
        assert wake.status == "pass"

    def test_gestation_contracts_not_applicable_post_gestation(self):
        v = ProcessVerifier()
        result = v.verify(
            [{"event_type": "dummy", "mode": "conversational"}],
            {}, "conversational",
        )
        gest = next(
            (vd for vd in result.verdicts if vd.contract_id == "gestation_started"), None
        )
        assert gest is not None
        assert gest.status == "not_applicable"

    def test_gestation_contracts_applicable_during_gestation(self):
        v = ProcessVerifier()
        result = v.verify(
            [{"event_type": "gestation:started", "mode": "gestation"}],
            {}, "gestation",
        )
        gest = next(
            (vd for vd in result.verdicts if vd.contract_id == "gestation_started"), None
        )
        assert gest is not None
        assert gest.status == "pass"


# ── ProcessVerifier: Coverage Computation ───────────────────────────


class TestVerifierCoverage:

    def test_zero_coverage_on_empty(self):
        v = ProcessVerifier()
        result = v.verify([], {}, "")
        assert result.coverage_pct == 0.0

    def test_coverage_increases_with_events(self):
        v = ProcessVerifier()
        many_events = [
            {"event_type": "memory:write", "mode": "gestation"},
            {"event_type": "memory:associated", "mode": "gestation"},
            {"event_type": "contradiction:detected", "mode": "gestation"},
            {"event_type": "quarantine:tick_complete", "mode": "gestation"},
            {"event_type": "audit:completed", "mode": "gestation"},
            {"event_type": "soul_integrity:updated", "mode": "gestation"},
            {"event_type": "meta:thought_generated", "mode": "gestation"},
            {"event_type": "consciousness:analysis", "mode": "gestation"},
            {"event_type": "mode:change", "mode": "gestation"},
            {"event_type": "gestation:started", "mode": "gestation"},
            {"event_type": "gestation:phase_advanced", "mode": "gestation"},
        ]
        snapshots = {
            "memory": {"total": 10},
            "library": {"total_sources": 5, "studied_count": 3},
            "study_telemetry": {"llm_extractions": 2, "total_claims": 15},
        }
        result = v.verify(many_events, snapshots, "gestation")
        assert result.coverage_pct > 0
        assert result.passing_contracts > 5
        assert result.applicable_contracts > 0

    def test_result_to_dict_shape(self):
        v = ProcessVerifier()
        result = v.verify([], {}, "gestation")
        d = result.to_dict()
        assert "verdicts" in d
        assert "coverage_pct" in d
        assert "total_contracts" in d
        assert "applicable_contracts" in d
        assert isinstance(d["verdicts"], list)

    def test_consecutive_tracking(self):
        v = ProcessVerifier()
        events = [{"event_type": "memory:write", "mode": "gestation"}]
        v.verify(events, {}, "gestation")
        result = v.verify(events, {}, "gestation")
        mem = next(vd for vd in result.verdicts if vd.contract_id == "memory_written")
        assert mem.consecutive_passes == 2

    def test_event_stays_seen_across_windows(self):
        """Once an event type is seen, it stays seen even if absent from later windows."""
        v = ProcessVerifier()
        events = [{"event_type": "memory:write", "mode": "conversational"}]
        v.verify(events, {}, "conversational")
        result = v.verify(
            [{"event_type": "other", "mode": "conversational"}],
            {}, "conversational",
        )
        mem = next(vd for vd in result.verdicts if vd.contract_id == "memory_written")
        assert mem.status == "pass"
        assert mem.consecutive_passes == 2

    def test_hydrate_from_history_seeds_seen_types(self):
        """hydrate_from_history() populates _seen_event_types before first verify()."""
        v = ProcessVerifier()
        history = [
            {"event_type": "perception:wake_word", "mode": "conversational"},
            {"event_type": "perception:transcription", "mode": "conversational"},
            {"event_type": "conversation:user_message", "mode": "conversational"},
            {"event_type": "conversation:response", "mode": "conversational"},
        ]
        v.hydrate_from_history(history)
        assert v._hydrated
        assert "perception:wake_word" in v._seen_event_types
        assert "conversational" in v._seen_modes

        result = v.verify(
            [{"event_type": "other", "mode": "reflective"}],
            {}, "reflective",
        )
        ww = next(vd for vd in result.verdicts if vd.contract_id == "wake_word_detected")
        assert ww.status == "pass"


# ── ProcessVerdict ──────────────────────────────────────────────────


class TestProcessVerdict:

    def test_to_dict(self):
        v = ProcessVerdict(
            contract_id="test", group="test_group", label="Test",
            status="pass", evidence="event seen",
        )
        d = v.to_dict()
        assert d["contract_id"] == "test"
        assert d["status"] == "pass"
        assert d["evidence"] == "event seen"


# ── VerificationResult ──────────────────────────────────────────────


class TestVerificationResult:

    def test_empty_result(self):
        r = VerificationResult()
        d = r.to_dict()
        assert d["total_contracts"] == 0
        assert d["coverage_pct"] == 0.0

    def test_result_roundtrip(self):
        r = VerificationResult(
            total_contracts=10, applicable_contracts=8,
            passing_contracts=6, failing_contracts=2,
            coverage_pct=75.0,
        )
        d = r.to_dict()
        assert d["coverage_pct"] == 75.0
        assert d["passing_contracts"] == 6


# ── Dashboard Adapter PVL Panel ─────────────────────────────────────


class TestDashboardAdapterPVL:

    def test_pvl_panel_awaiting(self):
        from jarvis_eval.dashboard_adapter import build_dashboard_snapshot

        result = build_dashboard_snapshot(
            store_meta={"created_at": 0},
            store_file_sizes={},
            recent_snapshots=[],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
            pvl_result=None,
            pvl_stats={},
        )
        assert "pvl" in result
        assert result["pvl"]["status"] == "awaiting_first_run"

    def test_pvl_panel_active(self):
        from jarvis_eval.dashboard_adapter import build_dashboard_snapshot

        pvl_result = {
            "verdicts": [
                {"contract_id": "memory_written", "group": "memory_pipeline",
                 "label": "Memory Written", "status": "pass", "evidence": "event seen"},
            ],
            "total_contracts": 55,
            "applicable_contracts": 40,
            "passing_contracts": 30,
            "failing_contracts": 5,
            "awaiting_contracts": 5,
            "coverage_pct": 75.0,
            "timestamp": 1000.0,
        }

        result = build_dashboard_snapshot(
            store_meta={"created_at": 0},
            store_file_sizes={},
            recent_snapshots=[],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
            pvl_result=pvl_result,
            pvl_stats={"run_count": 5},
        )

        pvl = result["pvl"]
        assert pvl["status"] == "active"
        assert pvl["coverage_pct"] == 75.0
        assert pvl["passing_contracts"] == 30
        assert isinstance(pvl["groups"], list)
        assert isinstance(pvl["playbook"], dict)
        assert "1" in pvl["playbook"]
        assert "verifier_stats" in pvl

    def test_pvl_banner_flag(self):
        from jarvis_eval.dashboard_adapter import build_dashboard_snapshot

        result = build_dashboard_snapshot(
            store_meta={"created_at": 0},
            store_file_sizes={},
            recent_snapshots=[],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
        )
        assert result["banner"]["pvl_enabled"] is True

    def test_language_panel_passes_gate_and_promotion_diagnostics(self):
        from jarvis_eval.dashboard_adapter import build_dashboard_snapshot

        language_metrics = {
            "corpus_total_examples": 140,
            "quality_total_events": 80,
            "quality_native_usage_rate": 0.8,
            "quality_fail_closed_rate": 0.15,
            "quality_counts_by_class": {"self_status": 20},
            "quality_counts_by_outcome": {"ok": 80},
            "quality_native_used_by_class": {"self_status": 20},
            "quality_fail_closed_by_class": {"self_status": 1},
            "quality_last_event_ts": 100.0,
            "corpus_last_capture_ts": 90.0,
            "gate_color": "yellow",
            "gate_color_code": 1,
            "gate_scores": {"sample_count": 1.0},
            "gate_scores_by_class": {"self_status": {"color": "yellow", "scores": {"sample_count": 1.0}}},
            "promotion_summary": {
                "self_status": {
                    "level": "canary",
                    "color": "yellow",
                    "consecutive_green": 1,
                    "consecutive_red": 0,
                    "total_evaluations": 8,
                    "dwell_s": 12.0,
                    "promotion_history_len": 1,
                }
            },
            "promotion_aggregate": {
                "levels": {"shadow": 6, "canary": 1, "live": 0},
                "colors": {"green": 1, "yellow": 6, "red": 0},
                "total_evaluations": 56,
                "max_consecutive_red": 0,
                "max_consecutive_green": 2,
            },
            "promotion_shadow_count": 6,
            "promotion_canary_count": 1,
            "promotion_live_count": 0,
            "promotion_green_classes": 1,
            "promotion_yellow_classes": 6,
            "promotion_red_classes": 0,
            "promotion_total_evaluations": 56,
            "promotion_max_consecutive_red": 0,
            "phase_c": {
                "tokenizer": {"strategy": "bpe", "estimated_vocab_size": 4096},
                "split": {"train_count": 100, "val_count": 20},
                "student": {"available": True, "reason": ""},
            },
        }

        result = build_dashboard_snapshot(
            store_meta={"created_at": 0},
            store_file_sizes={},
            recent_snapshots=[
                {"source": "language", "metrics": language_metrics, "timestamp": 123.0},
            ],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
            pvl_result=None,
            pvl_stats={},
        )

        language_panel = result["language"]
        assert language_panel["gate_color"] == "yellow"
        assert language_panel["gate_color_code"] == 1
        assert language_panel["promotion_canary_count"] == 1
        assert language_panel["promotion_total_evaluations"] == 56
        assert "self_status" in language_panel["promotion_summary"]
        assert "self_status" in language_panel["gate_scores_by_class"]


# ── Study Telemetry ─────────────────────────────────────────────────


class TestStudyTelemetry:

    def test_get_study_telemetry_returns_copy(self):
        from library.study import get_study_telemetry
        t1 = get_study_telemetry()
        t2 = get_study_telemetry()
        assert t1 is not t2
        assert isinstance(t1, dict)
        assert "llm_extractions" in t1
        assert "regex_fallbacks" in t1
        assert "total_claims" in t1
        assert "total_concepts" in t1
        assert "sources_studied" in t1

    def test_telemetry_keys_are_numeric(self):
        from library.study import get_study_telemetry
        t = get_study_telemetry()
        for k, v in t.items():
            assert isinstance(v, (int, float)), f"Key {k} has non-numeric value {v}"


# ── System upgrades PVL ─────────────────────────────────────────────


class TestSystemUpgradesPvl:

    def test_system_upgrades_group_registered(self):
        from jarvis_eval.process_contracts import PROCESS_GROUPS, get_contracts_by_group
        assert "system_upgrades" in PROCESS_GROUPS
        assert len(get_contracts_by_group()["system_upgrades"]) >= 8

    def test_sandbox_pass_compound_contract_passes(self):
        from jarvis_eval.process_contracts import ALL_CONTRACTS
        from jarvis_eval.process_verifier import ProcessVerifier

        compound = next(
            c for c in ALL_CONTRACTS if c.contract_id == "si_structured_report_with_sandbox_pass"
        )
        v = ProcessVerifier()
        v.hydrate_from_history([{"event_type": "improvement:sandbox_passed", "mode": "passive"}])
        recent = [{"event_type": "improvement:sandbox_passed", "mode": "passive"}]
        snaps = {"system_upgrades": {"upgrade_reports_total": 2.0}}
        status = v._evaluate_contract(compound, {"improvement:sandbox_passed"}, snaps)
        assert status == "pass"
