"""Phase 6.5 attestation ledger — hash-attested operator-seeded records.

Problem this module solves
--------------------------
After a brain reset, live policy-memory counters rebuild from zero. The
previous brain had already demonstrated capabilities (e.g. L3 autonomy
with a 95.1 Oracle composite), but the new brain cannot re-earn that
evidence overnight. The attestation ledger gives operators a way to
seed "this capability has been demonstrated before" so manual L3
promotion becomes *requestable*, without contaminating current-runtime
health fields.

Design invariants (enforced by tests)
-------------------------------------
- Attestation NEVER mutates ``maturity_highwater.json``,
  ``pvl_contract_highwater.json``, ``autonomy_state.json``, or any
  ``ever_*`` field. The ledger lives in its own file only.
- Attestation can satisfy ``prior_attested_ok`` for a capability, which
  can satisfy ``request_ok`` for *manual* L3 promotion. It CANNOT
  generate escalation requests on its own (those require live
  ``autonomy_level >= 3``). See plan doc: Two escalation generation
  modes.
- Records are tamper-evident by sha256 of the seed-time artifact. The
  word "signed" is intentionally avoided because this is not a PKI
  signature; future phases may add that.
- Loader computes ``artifact_status`` per record at read time and
  exposes it as a machine-readable field so dashboards and validation
  tooling can render warnings without re-implementing the check.

The canonical design is in
``docs/plans/phase_6_5_l3_escalation.plan.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
_ATTESTATION_DIR = _JARVIS_DIR / "eval"
_ATTESTATION_PATH = _ATTESTATION_DIR / "ever_proven_attestation.json"


# --------------------------------------------------------------------------
# Artifact status enum-like constants
# --------------------------------------------------------------------------

ARTIFACT_HASH_VERIFIED = "hash_verified"
ARTIFACT_HASH_MISMATCH = "hash_mismatch"
ARTIFACT_MISSING = "missing"
ARTIFACT_HASH_UNVERIFIABLE = "hash_unverifiable"

ARTIFACT_STATUSES = frozenset({
    ARTIFACT_HASH_VERIFIED,
    ARTIFACT_HASH_MISMATCH,
    ARTIFACT_MISSING,
    ARTIFACT_HASH_UNVERIFIABLE,
})

STRENGTH_VERIFIED = "verified"
STRENGTH_ARCHIVED_MISSING = "archived_missing"


# --------------------------------------------------------------------------
# Capability registry — mandatory fields and deterministic parsers
# --------------------------------------------------------------------------

CAPABILITY_REQUIRED_FIELDS: dict[str, set[str]] = {
    # These five fields are exactly what check_promotion_eligibility()
    # consults. An L3 attestation missing any of them is incomplete and
    # MUST NOT be silently accepted.
    "autonomy.l3": {
        "oracle_composite",
        "autonomy_domain_score",
        "autonomy_level_reached",
        "win_rate",
        "total_outcomes",
    },
}


def _parse_autonomy_l3_from_markdown(text: str) -> dict[str, Any]:
    """Deterministically extract autonomy.l3 measurements from a report.

    Matches the format of ``docs/pre_reset_report_phase9_complete.md``
    and similar post-phase proof reports. Returns a dict containing any
    of the mandatory fields it could extract; callers (the seed tool)
    are responsible for validating completeness against
    ``CAPABILITY_REQUIRED_FIELDS``.

    The parser is intentionally narrow: it looks for labelled numeric
    values and does not attempt to guess from prose. If the report
    format changes in a future phase, this parser should grow a new
    branch rather than become fuzzy.
    """
    extracted: dict[str, Any] = {}

    # Oracle composite (e.g. "Oracle composite: 95.1" or "oracle_composite=95.1")
    m = re.search(
        r"oracle[\s_]?composite[^\d\-]*([0-9]+(?:\.[0-9]+)?)",
        text, re.IGNORECASE,
    )
    if m:
        extracted["oracle_composite"] = float(m.group(1))

    # Autonomy domain score (e.g. "Autonomy domain score: 10.0/10")
    m = re.search(
        r"autonomy[\s_]?domain[\s_]?score[^\d\-]*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*([0-9]+(?:\.[0-9]+)?))?",
        text, re.IGNORECASE,
    )
    if m:
        if m.group(2):
            extracted["autonomy_domain_score"] = f"{m.group(1)}/{m.group(2)}"
        else:
            extracted["autonomy_domain_score"] = float(m.group(1))

    # Autonomy level reached (e.g. "L3 autonomy reached" or "autonomy_level_reached: 3")
    m = re.search(
        r"autonomy[\s_]?level[\s_]?reached[^\d\-]*([0-3])",
        text, re.IGNORECASE,
    )
    if m:
        extracted["autonomy_level_reached"] = int(m.group(1))
    else:
        m = re.search(r"\bL([0-3])\s+autonomy\s+reached\b", text, re.IGNORECASE)
        if m:
            extracted["autonomy_level_reached"] = int(m.group(1))

    # Win rate (e.g. "win rate: 0.79" or "Win rate: 79%")
    m = re.search(
        r"win[\s_]?rate[^\d\-]*([0-9]+(?:\.[0-9]+)?)\s*(%?)",
        text, re.IGNORECASE,
    )
    if m:
        val = float(m.group(1))
        if m.group(2) == "%" or val > 1.0:
            val = val / 100.0
        extracted["win_rate"] = round(val, 4)

    # Total outcomes (e.g. "Total outcomes: 208" or "total_outcomes=208")
    m = re.search(
        r"total[\s_]?outcomes[^\d\-]*([0-9]+)",
        text, re.IGNORECASE,
    )
    if m:
        extracted["total_outcomes"] = int(m.group(1))

    return extracted


CAPABILITY_PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "autonomy.l3": _parse_autonomy_l3_from_markdown,
}


# --------------------------------------------------------------------------
# Record and ledger
# --------------------------------------------------------------------------


@dataclass
class AttestationRecord:
    """A single operator-seeded, hash-attested capability record.

    ``artifact_status`` and ``attestation_strength`` are computed at
    load time, not stored in the JSON file — the file only carries the
    tamper-evidence inputs (``report_hash`` + ``artifact_refs``).
    """

    capability_id: str
    evidence_source: str
    evidence_window_start: str
    evidence_window_end: str
    measured_values: dict[str, Any]
    acceptance_reason: str
    accepted_by: str
    accepted_at: str
    report_hash: str
    artifact_refs: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    measured_source: str = "parsed"  # parsed | operator_supplied | mixed
    # Derived fields (populated by the loader, not persisted):
    artifact_status: str = ARTIFACT_HASH_UNVERIFIABLE
    attestation_strength: str | None = None

    def to_persisted_dict(self) -> dict[str, Any]:
        """Dict form for on-disk JSON — excludes derived fields."""
        d = asdict(self)
        d.pop("artifact_status", None)
        d.pop("attestation_strength", None)
        return d

    def to_api_dict(self) -> dict[str, Any]:
        """Dict form for dashboard/API — includes derived fields."""
        return asdict(self)


class AttestationLedgerError(Exception):
    """Raised for unrecoverable ledger errors (schema, parse, corruption)."""


class AttestationLedger:
    """Hash-attested operator-seeded capability ledger.

    Use :meth:`load` to read the current ledger (derived fields are
    recomputed each load) and :meth:`add` to append a new record. The
    ledger is stored at ``~/.jarvis/eval/ever_proven_attestation.json``
    as a JSON array of records.

    The ledger never touches ``maturity_highwater.json``, autonomy
    state, or ``ever_*`` fields — that invariant is enforced
    structurally (we only know our own path) and verified by tests.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else _ATTESTATION_PATH

    @property
    def path(self) -> Path:
        return self._path

    # -- load / save ---------------------------------------------------------

    def load(self) -> list[AttestationRecord]:
        """Load and validate all records; return derived-field-populated list.

        Records with unknown ``capability_id`` or mismatched
        ``schema_version`` are dropped with a logged warning (they are
        never auto-migrated). Records with ``artifact_status ==
        hash_mismatch`` or ``hash_unverifiable`` are still returned so
        consumers can render warnings, but
        :meth:`prior_attested_ok` will not return True for them.
        """
        if not self._path.exists():
            return []

        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Attestation ledger corrupt or unreadable at %s: %s",
                self._path, exc,
            )
            return []

        if not isinstance(raw, list):
            logger.error(
                "Attestation ledger root is not a list (got %s); treating as empty",
                type(raw).__name__,
            )
            return []

        records: list[AttestationRecord] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                logger.warning("Attestation ledger entry %d is not a dict, skipping", i)
                continue
            cap = entry.get("capability_id")
            if cap not in CAPABILITY_REQUIRED_FIELDS:
                logger.warning(
                    "Attestation ledger entry %d has unknown capability_id=%r, skipping",
                    i, cap,
                )
                continue
            sv = entry.get("schema_version")
            if sv != SCHEMA_VERSION:
                logger.warning(
                    "Attestation ledger entry %d has schema_version=%r (expected %d), skipping",
                    i, sv, SCHEMA_VERSION,
                )
                continue
            try:
                rec = AttestationRecord(
                    capability_id=entry["capability_id"],
                    evidence_source=entry.get("evidence_source", ""),
                    evidence_window_start=entry.get("evidence_window_start", ""),
                    evidence_window_end=entry.get("evidence_window_end", ""),
                    measured_values=dict(entry.get("measured_values", {})),
                    acceptance_reason=entry.get("acceptance_reason", ""),
                    accepted_by=entry.get("accepted_by", ""),
                    accepted_at=entry.get("accepted_at", ""),
                    report_hash=entry.get("report_hash", ""),
                    artifact_refs=list(entry.get("artifact_refs", [])),
                    schema_version=sv,
                    measured_source=entry.get("measured_source", "parsed"),
                )
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Attestation ledger entry %d malformed: %s — skipping", i, exc,
                )
                continue
            self._populate_derived(rec)
            records.append(rec)

        return records

    def prior_attested_records(
        self, capability_id: str | None = None
    ) -> list[AttestationRecord]:
        """Return records that count as ``prior_attested_ok``.

        Records with ``artifact_status`` in ``{hash_mismatch,
        hash_unverifiable}`` are filtered out. ``missing`` still
        counts (archived artifacts may legitimately be moved) but
        surfaces with ``attestation_strength == archived_missing``.
        """
        out: list[AttestationRecord] = []
        for rec in self.load():
            if capability_id is not None and rec.capability_id != capability_id:
                continue
            if rec.attestation_strength is None:
                continue
            out.append(rec)
        return out

    def prior_attested_ok(self, capability_id: str) -> bool:
        """Is there at least one accepted record for this capability?"""
        return len(self.prior_attested_records(capability_id)) > 0

    def add(self, rec: AttestationRecord, *, force: bool = False) -> None:
        """Append a new record atomically.

        Refuses duplicates (same ``capability_id`` + ``report_hash``)
        unless ``force=True``. Schema validation is performed; malformed
        records raise :class:`AttestationLedgerError`.
        """
        self._validate_record_for_write(rec)
        existing = self.load()
        if not force:
            for e in existing:
                if (
                    e.capability_id == rec.capability_id
                    and e.report_hash == rec.report_hash
                ):
                    raise AttestationLedgerError(
                        f"Duplicate attestation for {rec.capability_id} "
                        f"with same report_hash={rec.report_hash} "
                        "(use force=True to override)"
                    )
        existing.append(rec)
        self._atomic_write([e.to_persisted_dict() for e in existing])

    # -- internals -----------------------------------------------------------

    def _validate_record_for_write(self, rec: AttestationRecord) -> None:
        if rec.capability_id not in CAPABILITY_REQUIRED_FIELDS:
            raise AttestationLedgerError(
                f"Unknown capability_id={rec.capability_id!r}; register in "
                "CAPABILITY_REQUIRED_FIELDS before adding records"
            )
        if rec.schema_version != SCHEMA_VERSION:
            raise AttestationLedgerError(
                f"schema_version={rec.schema_version} (expected {SCHEMA_VERSION})"
            )
        required = CAPABILITY_REQUIRED_FIELDS[rec.capability_id]
        missing = required - set(rec.measured_values.keys())
        if missing:
            raise AttestationLedgerError(
                f"Attestation for {rec.capability_id} is missing required "
                f"measured_values fields: {sorted(missing)}"
            )
        if not rec.report_hash or not rec.report_hash.startswith("sha256:"):
            raise AttestationLedgerError(
                "report_hash must be a 'sha256:<hex>' string"
            )
        if rec.measured_source not in {"parsed", "operator_supplied", "mixed"}:
            raise AttestationLedgerError(
                f"measured_source must be parsed|operator_supplied|mixed "
                f"(got {rec.measured_source!r})"
            )

    def _populate_derived(self, rec: AttestationRecord) -> None:
        rec.artifact_status = self._compute_artifact_status(rec)
        if rec.artifact_status == ARTIFACT_HASH_VERIFIED:
            rec.attestation_strength = STRENGTH_VERIFIED
        elif rec.artifact_status == ARTIFACT_MISSING:
            rec.attestation_strength = STRENGTH_ARCHIVED_MISSING
        else:
            rec.attestation_strength = None

    @staticmethod
    def _compute_artifact_status(rec: AttestationRecord) -> str:
        """Return the current status of the attested artifact on disk."""
        if not rec.artifact_refs:
            return ARTIFACT_MISSING
        # We only attempt to verify the first artifact reference. Multi-artifact
        # attestations must register all their hashes in a future schema.
        ref = rec.artifact_refs[0]
        path = Path(ref)
        if not path.is_absolute():
            # Resolve relative to the repository root (two levels above
            # this file: brain/autonomy/attestation.py -> repo root).
            repo_root = Path(__file__).resolve().parent.parent.parent
            path = repo_root / ref
        if not path.exists():
            return ARTIFACT_MISSING
        try:
            actual = _sha256_of_file(path)
        except OSError as exc:
            logger.warning("Cannot hash %s: %s", path, exc)
            return ARTIFACT_HASH_UNVERIFIABLE
        expected = rec.report_hash.removeprefix("sha256:").strip().lower()
        if actual.lower() == expected:
            return ARTIFACT_HASH_VERIFIED
        return ARTIFACT_HASH_MISMATCH

    def _atomic_write(self, payload: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".attestation_", suffix=".json", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def _sha256_of_file(path: Path, *, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_path(path: Path | str) -> str:
    """Public helper for the seed tool."""
    return _sha256_of_file(Path(path))
