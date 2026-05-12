"""Tests for the Phase 6.5 attestation seeder CLI.

Covers the contract-critical parts of ``scripts.seed_ever_proven_from_report``:

- Fail-closed behavior when the deterministic parser cannot extract
  the mandatory measurement set.
- Operator --measured flag supplementation with correct
  measured_source classification (parsed / operator_supplied / mixed).
- Reason minimum-length rule.
- End-to-end happy path writes a valid record the ledger accepts back.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from autonomy.attestation import AttestationLedger, CAPABILITY_REQUIRED_FIELDS
from scripts.seed_ever_proven_from_report import (
    SeedError,
    _classify_measured_source,
    _parse_measured_flags,
    build_record,
    main,
)


FULL_REPORT = (
    "# Phase 9 complete\n"
    "Oracle composite: 95.1\n"
    "Autonomy domain score: 10.0/10\n"
    "autonomy_level_reached: 3\n"
    "Win rate: 79%\n"
    "Total outcomes: 208\n"
)

PARTIAL_REPORT = (
    "# Phase 9 partial\n"
    "Oracle composite: 95.1\n"
    "Win rate: 79%\n"
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestFailClosed:
    def test_missing_required_fields_raises(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(PARTIAL_REPORT)
        with pytest.raises(SeedError, match="missing"):
            build_record(
                report_path=report,
                capability="autonomy.l3",
                reason="Partial report, deterministic parse is not sufficient",
                accepted_by="operator:test",
                measured_flags=[],
                window_start="",
                window_end="",
            )

    def test_missing_required_fields_can_be_supplied_by_flags(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(PARTIAL_REPORT)
        rec = build_record(
            report_path=report,
            capability="autonomy.l3",
            reason="Partial report, filling missing fields from operator flags",
            accepted_by="operator:test",
            measured_flags=[
                "autonomy_domain_score=10.0/10",
                "autonomy_level_reached=3",
                "total_outcomes=208",
            ],
            window_start="",
            window_end="",
        )
        assert rec.measured_source == "mixed"
        for k in CAPABILITY_REQUIRED_FIELDS["autonomy.l3"]:
            assert k in rec.measured_values

    def test_unknown_capability_fails(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        with pytest.raises(SeedError, match="Unknown capability"):
            build_record(
                report_path=report,
                capability="made.up",
                reason="Should never succeed — unknown capability id here",
                accepted_by="operator:test",
                measured_flags=[],
                window_start="",
                window_end="",
            )

    def test_missing_report_file_fails(self, tmp_dir):
        with pytest.raises(SeedError, match="not found"):
            build_record(
                report_path=tmp_dir / "nope.md",
                capability="autonomy.l3",
                reason="Report file is missing from the seed command inputs",
                accepted_by="operator:test",
                measured_flags=[],
                window_start="",
                window_end="",
            )


class TestMeasuredSourceClassification:
    def test_parsed_only(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        rec = build_record(
            report_path=report,
            capability="autonomy.l3",
            reason="All required fields came from deterministic parsing only",
            accepted_by="operator:test",
            measured_flags=[],
            window_start="",
            window_end="",
        )
        assert rec.measured_source == "parsed"

    def test_operator_supplied_only(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text("no measurements here at all\n")
        rec = build_record(
            report_path=report,
            capability="autonomy.l3",
            reason="Parser found nothing; operator supplied every required field",
            accepted_by="operator:test",
            measured_flags=[
                "oracle_composite=95.1",
                "autonomy_domain_score=10.0/10",
                "autonomy_level_reached=3",
                "win_rate=0.79",
                "total_outcomes=208",
            ],
            window_start="",
            window_end="",
        )
        assert rec.measured_source == "operator_supplied"

    def test_mixed(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(PARTIAL_REPORT)  # parses oracle + win_rate
        rec = build_record(
            report_path=report,
            capability="autonomy.l3",
            reason="Mix of parsed oracle/win_rate and operator-supplied rest",
            accepted_by="operator:test",
            measured_flags=[
                "autonomy_domain_score=10.0/10",
                "autonomy_level_reached=3",
                "total_outcomes=208",
            ],
            window_start="",
            window_end="",
        )
        assert rec.measured_source == "mixed"

    def test_operator_flags_override_parsed_values(self, tmp_dir):
        """Operator override wins the value, and the record is tagged
        ``mixed`` — any operator touch on a required field is surfaced
        to auditors, not hidden behind a ``parsed`` label. This is the
        more conservative audit posture required by the plan.
        """
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        rec = build_record(
            report_path=report,
            capability="autonomy.l3",
            reason="Operator overrides oracle_composite value to correct typo",
            accepted_by="operator:test",
            measured_flags=["oracle_composite=95.2"],
            window_start="",
            window_end="",
        )
        assert rec.measured_values["oracle_composite"] == 95.2
        assert rec.measured_source == "mixed"


class TestReasonValidation:
    def test_short_reason_rejected(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        with pytest.raises(SeedError, match="20 characters"):
            build_record(
                report_path=report,
                capability="autonomy.l3",
                reason="too short",
                accepted_by="operator:test",
                measured_flags=[],
                window_start="",
                window_end="",
            )

    def test_empty_accepted_by_rejected(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        with pytest.raises(SeedError, match="accepted-by"):
            build_record(
                report_path=report,
                capability="autonomy.l3",
                reason="Reason is fine and of sufficient length for validation",
                accepted_by="",
                measured_flags=[],
                window_start="",
                window_end="",
            )


class TestMeasuredFlagParsing:
    def test_key_value_with_equals(self):
        got = _parse_measured_flags(["a=1", "b=2.5", "c=10.0/10"])
        assert got == {"a": 1, "b": 2.5, "c": "10.0/10"}

    def test_missing_equals_rejected(self):
        with pytest.raises(SeedError, match="key=value"):
            _parse_measured_flags(["bad-entry"])

    def test_empty_key_rejected(self):
        with pytest.raises(SeedError, match="non-empty"):
            _parse_measured_flags(["=value"])


class TestClassificationHelper:
    def test_all_parsed(self):
        parsed = {"a": 1, "b": 2}
        assert _classify_measured_source(parsed, {}, {"a", "b"}) == "parsed"

    def test_all_flagged(self):
        flagged = {"a": 1, "b": 2}
        assert _classify_measured_source({}, flagged, {"a", "b"}) == "operator_supplied"

    def test_mixed(self):
        assert _classify_measured_source({"a": 1}, {"b": 2}, {"a", "b"}) == "mixed"


class TestEndToEndCliHappyPath:
    def test_dry_run_does_not_write(self, tmp_dir, capsys):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        ledger_path = tmp_dir / "ledger.json"
        rc = main([
            "--report", str(report),
            "--capability", "autonomy.l3",
            "--reason", "End-to-end dry-run must not mutate the ledger file",
            "--accepted-by", "operator:pytest",
            "--ledger-path", str(ledger_path),
            "--dry-run",
        ])
        assert rc == 0
        assert not ledger_path.exists()

    def test_real_run_writes_a_loadable_record(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(FULL_REPORT)
        ledger_path = tmp_dir / "ledger.json"
        rc = main([
            "--report", str(report),
            "--capability", "autonomy.l3",
            "--reason", "End-to-end happy path writes a valid ledger record",
            "--accepted-by", "operator:pytest",
            "--ledger-path", str(ledger_path),
            "--yes",
        ])
        assert rc == 0
        ledger = AttestationLedger(path=ledger_path)
        records = ledger.load()
        assert len(records) == 1
        assert records[0].capability_id == "autonomy.l3"
        assert records[0].measured_source == "parsed"
        assert records[0].artifact_status == "hash_verified"
        assert records[0].attestation_strength == "verified"

    def test_missing_required_fields_exits_nonzero(self, tmp_dir):
        report = tmp_dir / "r.md"
        report.write_text(PARTIAL_REPORT)
        ledger_path = tmp_dir / "ledger.json"
        rc = main([
            "--report", str(report),
            "--capability", "autonomy.l3",
            "--reason", "CLI fail-closed on partial input with no flags",
            "--accepted-by", "operator:pytest",
            "--ledger-path", str(ledger_path),
            "--yes",
        ])
        assert rc == 2
        assert not ledger_path.exists()
