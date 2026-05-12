"""Tests for the Phase 6.5 attestation ledger.

Covers:
- Round-trip save and load
- Schema validation (capability_id, schema_version, required fields,
  measured_source enum, report_hash prefix)
- ``artifact_status`` computation: hash_verified / hash_mismatch /
  missing / hash_unverifiable
- ``attestation_strength`` surface: verified / archived_missing /
  None for rejected records
- Duplicate protection and ``force`` override
- Corrupt file fail-closed behavior
- Never-mutate-current invariant: adding a record must not touch
  autonomy_state.json, maturity_highwater.json, or
  pvl_contract_highwater.json
- Deterministic parser: valid report parses all five mandatory fields;
  empty input returns empty dict (seed tool handles the fail-closed
  decision)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from autonomy.attestation import (
    ARTIFACT_HASH_MISMATCH,
    ARTIFACT_HASH_UNVERIFIABLE,
    ARTIFACT_HASH_VERIFIED,
    ARTIFACT_MISSING,
    AttestationLedger,
    AttestationLedgerError,
    AttestationRecord,
    CAPABILITY_PARSERS,
    CAPABILITY_REQUIRED_FIELDS,
    STRENGTH_ARCHIVED_MISSING,
    STRENGTH_VERIFIED,
    sha256_of_path,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_artifact(path: Path, body: bytes = b"report body") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _make_l3_record(artifact: Path, *, report_hash: str | None = None) -> AttestationRecord:
    if report_hash is None:
        report_hash = _write_artifact(artifact)
    return AttestationRecord(
        capability_id="autonomy.l3",
        evidence_source=str(artifact),
        evidence_window_start="2026-03-21T00:00:00Z",
        evidence_window_end="2026-04-17T07:00:00Z",
        measured_values={
            "oracle_composite": 95.1,
            "autonomy_domain_score": "10.0/10",
            "autonomy_level_reached": 3,
            "win_rate": 0.79,
            "total_outcomes": 208,
        },
        acceptance_reason="test attestation record with adequate descriptive reason",
        accepted_by="operator:test",
        accepted_at="2026-04-22T00:00:00Z",
        report_hash=report_hash,
        artifact_refs=[str(artifact)],
    )


class TestRoundTrip:
    def test_load_empty_when_file_missing(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        assert ledger.load() == []

    def test_add_then_load_preserves_record(self, tmp_dir):
        artifact = tmp_dir / "report.md"
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        rec = _make_l3_record(artifact)
        ledger.add(rec)

        records = ledger.load()
        assert len(records) == 1
        r = records[0]
        assert r.capability_id == "autonomy.l3"
        assert r.measured_values["oracle_composite"] == 95.1
        assert r.artifact_status == ARTIFACT_HASH_VERIFIED
        assert r.attestation_strength == STRENGTH_VERIFIED

    def test_prior_attested_ok_true_for_verified(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        ledger.add(_make_l3_record(tmp_dir / "report.md"))
        assert ledger.prior_attested_ok("autonomy.l3") is True

    def test_prior_attested_ok_false_when_no_record(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        assert ledger.prior_attested_ok("autonomy.l3") is False


class TestArtifactStatus:
    def test_hash_verified(self, tmp_dir):
        artifact = tmp_dir / "r.md"
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        ledger.add(_make_l3_record(artifact))
        [rec] = ledger.load()
        assert rec.artifact_status == ARTIFACT_HASH_VERIFIED
        assert rec.attestation_strength == STRENGTH_VERIFIED

    def test_hash_mismatch_not_attested(self, tmp_dir):
        artifact = tmp_dir / "r.md"
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        rec = _make_l3_record(artifact)
        ledger.add(rec)
        artifact.write_bytes(b"TAMPERED")

        [loaded] = ledger.load()
        assert loaded.artifact_status == ARTIFACT_HASH_MISMATCH
        assert loaded.attestation_strength is None
        assert ledger.prior_attested_ok("autonomy.l3") is False

    def test_missing_artifact_still_counts_as_archived(self, tmp_dir):
        artifact = tmp_dir / "r.md"
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        ledger.add(_make_l3_record(artifact))
        artifact.unlink()

        [loaded] = ledger.load()
        assert loaded.artifact_status == ARTIFACT_MISSING
        assert loaded.attestation_strength == STRENGTH_ARCHIVED_MISSING
        assert ledger.prior_attested_ok("autonomy.l3") is True

    def test_hash_unverifiable_not_attested(self, tmp_dir, monkeypatch):
        artifact = tmp_dir / "r.md"
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        ledger.add(_make_l3_record(artifact))

        def _raise(_path, *, chunk=1 << 16):
            raise OSError("permission denied")

        monkeypatch.setattr("autonomy.attestation._sha256_of_file", _raise)

        [loaded] = ledger.load()
        assert loaded.artifact_status == ARTIFACT_HASH_UNVERIFIABLE
        assert loaded.attestation_strength is None
        assert ledger.prior_attested_ok("autonomy.l3") is False


class TestSchemaValidation:
    def test_unknown_capability_rejected_on_write(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        bad = _make_l3_record(tmp_dir / "r.md")
        bad.capability_id = "made.up"
        with pytest.raises(AttestationLedgerError, match="Unknown capability"):
            ledger.add(bad)

    def test_missing_required_measurement_rejected(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        bad = _make_l3_record(tmp_dir / "r.md")
        bad.measured_values.pop("oracle_composite")
        with pytest.raises(AttestationLedgerError, match="missing required"):
            ledger.add(bad)

    def test_bad_schema_version_rejected_on_write(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        bad = _make_l3_record(tmp_dir / "r.md")
        bad.schema_version = 99
        with pytest.raises(AttestationLedgerError, match="schema_version"):
            ledger.add(bad)

    def test_non_sha256_report_hash_rejected(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        artifact = tmp_dir / "r.md"
        artifact.write_bytes(b"x")
        bad = _make_l3_record(artifact, report_hash="md5:deadbeef")
        with pytest.raises(AttestationLedgerError, match="sha256"):
            ledger.add(bad)

    def test_bad_measured_source_rejected(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        bad = _make_l3_record(tmp_dir / "r.md")
        bad.measured_source = "guessed"
        with pytest.raises(AttestationLedgerError, match="measured_source"):
            ledger.add(bad)

    def test_unknown_capability_skipped_on_read(self, tmp_dir):
        path = tmp_dir / "ledger.json"
        path.write_text(json.dumps([
            {"capability_id": "future.thing", "schema_version": 1},
        ]))
        ledger = AttestationLedger(path=path)
        assert ledger.load() == []

    def test_bad_schema_version_skipped_on_read(self, tmp_dir):
        artifact = tmp_dir / "r.md"
        h = _write_artifact(artifact)
        path = tmp_dir / "ledger.json"
        path.write_text(json.dumps([
            {
                "capability_id": "autonomy.l3",
                "schema_version": 99,
                "report_hash": h,
                "artifact_refs": [str(artifact)],
                "measured_values": {},
            },
        ]))
        ledger = AttestationLedger(path=path)
        assert ledger.load() == []


class TestDuplicateProtection:
    def test_duplicate_rejected_without_force(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        rec = _make_l3_record(tmp_dir / "r.md")
        ledger.add(rec)
        with pytest.raises(AttestationLedgerError, match="Duplicate"):
            ledger.add(rec)

    def test_force_allows_duplicate(self, tmp_dir):
        ledger = AttestationLedger(path=tmp_dir / "ledger.json")
        rec = _make_l3_record(tmp_dir / "r.md")
        ledger.add(rec)
        ledger.add(rec, force=True)
        assert len(ledger.load()) == 2


class TestCorruptFile:
    def test_corrupt_json_returns_empty(self, tmp_dir):
        path = tmp_dir / "ledger.json"
        path.write_text("{ not valid json !!")
        ledger = AttestationLedger(path=path)
        assert ledger.load() == []

    def test_non_list_root_returns_empty(self, tmp_dir):
        path = tmp_dir / "ledger.json"
        path.write_text(json.dumps({"capability_id": "autonomy.l3"}))
        ledger = AttestationLedger(path=path)
        assert ledger.load() == []

    def test_non_dict_entry_skipped(self, tmp_dir):
        path = tmp_dir / "ledger.json"
        path.write_text(json.dumps(["not a dict", 42]))
        ledger = AttestationLedger(path=path)
        assert ledger.load() == []


class TestNeverMutateCurrentInvariant:
    """The ledger must write to exactly one file — its own. It must never
    touch autonomy_state.json, maturity_highwater.json, or
    pvl_contract_highwater.json.
    """

    def test_adding_record_only_touches_ledger_file(self, tmp_dir):
        ledger_path = tmp_dir / "eval" / "ever_proven_attestation.json"
        sentinel_paths = [
            tmp_dir / "autonomy_state.json",
            tmp_dir / "eval" / "maturity_highwater.json",
            tmp_dir / "eval" / "pvl_contract_highwater.json",
        ]
        for p in sentinel_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"should_not_change": True}))

        ledger = AttestationLedger(path=ledger_path)
        ledger.add(_make_l3_record(tmp_dir / "r.md"))

        for p in sentinel_paths:
            assert json.loads(p.read_text()) == {"should_not_change": True}, (
                f"ledger write mutated sentinel file {p}"
            )
        assert ledger_path.exists()


class TestDeterministicParser:
    """The deterministic parser is the contract between the seed tool
    and the loader's mandatory-field validation.
    """

    def test_extracts_all_five_mandatory_fields(self):
        text = (
            "# Phase 9 report\n"
            "Oracle composite: 95.1\n"
            "Autonomy domain score: 10.0/10\n"
            "L3 autonomy reached (manual-only restore)\n"
            "Win rate: 79%\n"
            "Total outcomes: 208\n"
        )
        parser = CAPABILITY_PARSERS["autonomy.l3"]
        got = parser(text)

        required = CAPABILITY_REQUIRED_FIELDS["autonomy.l3"]
        assert set(got) >= required, (
            f"parser must extract all mandatory fields; missing "
            f"{required - set(got)}"
        )
        assert got["oracle_composite"] == 95.1
        assert got["autonomy_domain_score"] == "10.0/10"
        assert got["autonomy_level_reached"] == 3
        assert got["win_rate"] == 0.79
        assert got["total_outcomes"] == 208

    def test_accepts_alternate_field_syntax(self):
        text = (
            "oracle_composite=88.2 autonomy_domain_score: 8.5/10 "
            "autonomy_level_reached: 2 win_rate: 0.65 total_outcomes=120"
        )
        got = CAPABILITY_PARSERS["autonomy.l3"](text)
        assert got["oracle_composite"] == 88.2
        assert got["autonomy_level_reached"] == 2
        assert got["win_rate"] == 0.65
        assert got["total_outcomes"] == 120

    def test_empty_input_returns_empty_dict(self):
        got = CAPABILITY_PARSERS["autonomy.l3"]("")
        assert got == {}

    def test_partial_input_returns_partial_dict(self):
        text = "Oracle composite: 90.0 and nothing else measurable"
        got = CAPABILITY_PARSERS["autonomy.l3"](text)
        assert got == {"oracle_composite": 90.0}

    def test_win_rate_percent_normalized(self):
        assert CAPABILITY_PARSERS["autonomy.l3"](
            "Win rate: 79%"
        )["win_rate"] == 0.79
        assert CAPABILITY_PARSERS["autonomy.l3"](
            "win rate: 0.5"
        )["win_rate"] == 0.5


def test_sha256_of_path_matches_hashlib(tmp_dir):
    p = tmp_dir / "x"
    p.write_bytes(b"hello")
    assert sha256_of_path(p) == hashlib.sha256(b"hello").hexdigest()
