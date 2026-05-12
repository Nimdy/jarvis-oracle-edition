"""Tests for structured system_upgrades reports and PVL snapshot helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve.system_upgrade_report import (
    SystemUpgradeReport,
    can_write_bounded_memory,
    get_pvl_snapshot,
    is_complete_for_training,
    load_report,
    mint_attempt_id,
    mint_upgrade_id,
    report_path,
    save_report,
    sandbox_summary_from_evaluation_report,
    verdict_from_parts,
)


def test_mint_ids_unique():
    assert mint_upgrade_id().startswith("upg_")
    assert mint_attempt_id().startswith("att_")
    assert mint_upgrade_id() != mint_upgrade_id()


def test_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "self_improve.system_upgrade_report.REPORTS_DIR",
        tmp_path / "reports",
    )
    uid = "upg_test123456"
    rep = SystemUpgradeReport(
        upgrade_id=uid,
        request_id="imp_1",
        description_short="test",
        target_module="consciousness",
        orchestrator_status="promoted",
        verdict="verified_stable",
        sandbox_summary={
            "overall_passed": True,
            "sim_executed": True,
            "sim_p95_before": 10.0,
            "sim_p95_after": 9.0,
        },
    )
    save_report(rep)
    path = report_path(uid)
    assert path.exists()
    loaded = load_report(uid)
    assert loaded is not None
    assert loaded.request_id == "imp_1"
    assert loaded.verdict == "verified_stable"


def test_can_write_bounded_memory_requires_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "self_improve.system_upgrade_report.REPORTS_DIR",
        tmp_path / "reports",
    )
    rep = SystemUpgradeReport(
        upgrade_id="upg_x",
        request_id="imp_1",
        description_short="d",
        target_module="t",
    )
    assert not can_write_bounded_memory(rep)
    save_report(rep)
    assert can_write_bounded_memory(rep)


def test_is_complete_for_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "self_improve.system_upgrade_report.REPORTS_DIR",
        tmp_path / "reports",
    )
    rep = SystemUpgradeReport(
        upgrade_id="upg_tr",
        request_id="imp_1",
        description_short="d",
        target_module="t",
        orchestrator_status="promoted",
        verdict="verified_stable",
        sandbox_summary={
            "overall_passed": True,
            "sim_executed": True,
        },
    )
    save_report(rep)
    assert is_complete_for_training(rep)
    rep2 = SystemUpgradeReport(
        upgrade_id="upg_tr2",
        request_id="imp_2",
        description_short="d",
        target_module="t",
        verdict="none",
        sandbox_summary={"overall_passed": True, "sim_executed": True},
    )
    save_report(rep2)
    assert not is_complete_for_training(rep2)


def test_verdict_from_parts():
    assert verdict_from_parts("verifying") == "pending_verification"
    assert verdict_from_parts("rolled_back") == "rolled_back"
    assert verdict_from_parts("promoted", "improved") == "verified_improved"
    assert verdict_from_parts("promoted", "stable") == "verified_stable"


def test_sandbox_summary_from_evaluation_report_none():
    assert sandbox_summary_from_evaluation_report(None) == {}


def test_get_pvl_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "self_improve.system_upgrade_report.REPORTS_DIR",
        tmp_path / "r",
    )
    snap = get_pvl_snapshot()
    assert "upgrade_reports_total" in snap
    assert "truth_lane_ready" in snap


def test_report_json_serializable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "self_improve.system_upgrade_report.REPORTS_DIR",
        tmp_path / "reports",
    )
    rep = SystemUpgradeReport(
        upgrade_id="upg_ser",
        request_id="imp_1",
        description_short="x",
        target_module="m",
        attempts=[{"attempt_id": "att_1", "iteration": 1}],
    )
    save_report(rep)
    data = json.loads(report_path("upg_ser").read_text(encoding="utf-8"))
    assert data["attempts"][0]["attempt_id"] == "att_1"
