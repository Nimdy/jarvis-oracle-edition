"""Tests for :mod:`scripts.dashboard_truth_probe`.

The probe is the machine-verifiable truth surface the dashboard rebuild
(P1.7) depends on. These tests feed it hand-crafted snapshots + tmp_path
evidence fixtures to lock the three classes of dashboard lies it
catches:

  - Serializer shape violations.
  - Empty-where-data-exists.
  - Attestation boundary inconsistencies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from scripts.dashboard_truth_probe import (
    SEVERITY_FAIL,
    SEVERITY_INFO,
    SEVERITY_WARN,
    run_probe,
)


def _finding_codes(report: dict, severity: str | None = None) -> set[str]:
    return {
        f["code"]
        for f in report.get("findings", [])
        if severity is None or f.get("severity") == severity
    }


def _mk_evidence_root(
    tmp_path: Path,
    *,
    distill_files: dict[str, str] | None = None,
    attestation_records: list[dict] | None = None,
) -> Path:
    jarvis = tmp_path / ".jarvis"
    jarvis.mkdir(parents=True, exist_ok=True)
    if distill_files is not None:
        hemi = jarvis / "hemisphere_training"
        hemi.mkdir(parents=True, exist_ok=True)
        for name, body in distill_files.items():
            (hemi / name).write_text(body, encoding="utf-8")
    if attestation_records is not None:
        eval_dir = jarvis / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "ever_proven_attestation.json").write_text(
            json.dumps(attestation_records), encoding="utf-8"
        )
    return tmp_path


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_clean_snapshot_is_ok(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {"specialists": [], "distillation": {}},
        },
        "autonomy": {"attestation": {"prior_attested_ok": False}},
    }
    evidence_root = _mk_evidence_root(tmp_path)

    report = run_probe(snapshot, evidence_root=evidence_root)

    assert report["status"] == "ok", report
    assert report["severity_counts"][SEVERITY_FAIL] == 0
    assert report["severity_counts"][SEVERITY_WARN] == 0


# --------------------------------------------------------------------------
# Shape violations
# --------------------------------------------------------------------------


def test_flags_raw_empty_dict_specialists(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {},
        },
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    codes = _finding_codes(report, severity=SEVERITY_FAIL)
    assert "shape.specialists.list_wrong" in codes
    assert "shape.specialists.distillation_wrong" in codes
    assert report["status"] == "fail"


def test_flags_specialists_as_non_dict(tmp_path):
    snapshot = {
        "self_improve": {"active": True, "specialists": "not-a-dict"},
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    assert "shape.specialists.not_dict" in _finding_codes(report)
    assert report["status"] == "fail"


def test_flags_degraded_fallback_from_serializer(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {
                "specialists": [],
                "distillation": {},
                "_error": "RuntimeError",
            },
        },
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    codes_warn = _finding_codes(report, severity=SEVERITY_WARN)
    assert "shape.specialists.degraded" in codes_warn


def test_missing_self_improve_is_info_only(tmp_path):
    snapshot = {}
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    assert report["status"] == "ok"
    assert "shape.self_improve.missing" in _finding_codes(report, SEVERITY_INFO)


# --------------------------------------------------------------------------
# Empty-where-data-exists
# --------------------------------------------------------------------------


def test_flags_empty_specialists_with_on_disk_data(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {"specialists": [], "distillation": {}},
        },
    }
    evidence_root = _mk_evidence_root(
        tmp_path,
        distill_files={
            "distill_diagnostic.jsonl": '{"x": 1}\n{"x": 2}\n',
            "distill_code_quality.jsonl": '{"y": 1}\n',
        },
    )

    report = run_probe(snapshot, evidence_root=evidence_root)

    codes = _finding_codes(report, severity=SEVERITY_WARN)
    assert "empty.specialists.data_exists" in codes
    finding = next(
        f
        for f in report["findings"]
        if f["code"] == "empty.specialists.data_exists"
    )
    populated = set(finding["evidence"]["populated_files"])
    assert "distill_diagnostic.jsonl" in populated
    assert "distill_code_quality.jsonl" in populated


def test_empty_specialists_without_on_disk_data_is_not_flagged(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {"specialists": [], "distillation": {}},
        },
    }
    evidence_root = _mk_evidence_root(
        tmp_path,
        distill_files={"distill_empty.jsonl": ""},
    )

    report = run_probe(snapshot, evidence_root=evidence_root)

    assert "empty.specialists.data_exists" not in _finding_codes(report)


# --------------------------------------------------------------------------
# L3 three-axis shape
# --------------------------------------------------------------------------


def test_flags_l3_missing_required_fields(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {"specialists": [], "distillation": {}},
            "l3_escalation": {"current_ok": False},
        },
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    fails = _finding_codes(report, severity=SEVERITY_FAIL)
    assert "shape.l3.missing_fields" in fails
    finding = next(
        f for f in report["findings"] if f["code"] == "shape.l3.missing_fields"
    )
    assert set(finding["evidence"]["missing"]) == {
        "prior_attested_ok",
        "request_ok",
        "activation_ok",
    }


def test_full_l3_three_axis_passes(tmp_path):
    snapshot = {
        "self_improve": {
            "active": True,
            "specialists": {"specialists": [], "distillation": {}},
            "l3_escalation": {
                "current_ok": False,
                "prior_attested_ok": False,
                "request_ok": False,
                "activation_ok": False,
            },
        },
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    assert "shape.l3.missing_fields" not in _finding_codes(report)


# --------------------------------------------------------------------------
# Attestation boundary
# --------------------------------------------------------------------------


def test_flags_claim_without_ledger(tmp_path):
    snapshot = {
        "autonomy": {"attestation": {"prior_attested_ok": True}},
    }
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    assert "attestation.claim_without_ledger" in _finding_codes(
        report, severity=SEVERITY_FAIL
    )


def test_flags_ledger_without_claim(tmp_path):
    snapshot = {
        "autonomy": {"attestation": {"prior_attested_ok": False}},
    }
    evidence_root = _mk_evidence_root(
        tmp_path,
        attestation_records=[{"capability_id": "autonomy.l3"}],
    )
    report = run_probe(snapshot, evidence_root=evidence_root)

    assert "attestation.ledger_without_claim" in _finding_codes(
        report, severity=SEVERITY_WARN
    )


def test_consistent_claim_with_ledger_is_ok(tmp_path):
    snapshot = {
        "autonomy": {"attestation": {"prior_attested_ok": True}},
    }
    evidence_root = _mk_evidence_root(
        tmp_path,
        attestation_records=[{"capability_id": "autonomy.l3"}],
    )
    report = run_probe(snapshot, evidence_root=evidence_root)

    codes = _finding_codes(report)
    assert "attestation.claim_without_ledger" not in codes
    assert "attestation.ledger_without_claim" not in codes


# --------------------------------------------------------------------------
# Orphaned field typing
# --------------------------------------------------------------------------


def test_flags_autonomy_not_dict(tmp_path):
    snapshot = {"autonomy": ["unexpected", "list"]}
    report = run_probe(snapshot, evidence_root=_mk_evidence_root(tmp_path))

    assert "shape.autonomy.not_dict" in _finding_codes(
        report, severity=SEVERITY_FAIL
    )
