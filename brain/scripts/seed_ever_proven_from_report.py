#!/usr/bin/env python3
"""Phase 6.5 — seed a hash-attested operator-accepted capability record.

Reads a post-phase proof report (e.g.
``docs/pre_reset_report_phase9_complete.md``), extracts the mandatory
measurement set for the target capability using a deterministic parser,
hashes the source artifact, and appends an attestation record to
``~/.jarvis/eval/ever_proven_attestation.json``.

Contract (from docs/plans/phase_6_5_l3_escalation.plan.md):
- Fail-closed by default: if the parser cannot extract the complete
  mandatory measurement set for the target capability, the tool exits
  non-zero and writes nothing.
- Operators may supply missing or overriding fields with repeated
  ``--measured key=value`` flags; the record is then tagged
  ``measured_source: "operator_supplied"`` (all from flags) or
  ``"mixed"`` (some parsed, some flagged).
- ``--dry-run`` prints the record without writing.
- Refuses to add a duplicate (same ``capability_id`` + ``report_hash``)
  unless ``--force``.

Usage::

    python -m scripts.seed_ever_proven_from_report \\
        --report docs/pre_reset_report_phase9_complete.md \\
        --capability autonomy.l3 \\
        --reason "Pre-reset Phase 9 brain reached Oracle 95.1, L3, 79% win rate on 208 outcomes" \\
        --accepted-by "operator:$(hostname)"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonomy.attestation import (
    CAPABILITY_PARSERS,
    CAPABILITY_REQUIRED_FIELDS,
    SCHEMA_VERSION,
    AttestationLedger,
    AttestationLedgerError,
    AttestationRecord,
    sha256_of_path,
)


class SeedError(Exception):
    """Non-zero-exit error class for the CLI."""


def _coerce_measured_value(key: str, raw: str) -> Any:
    """Coerce ``--measured key=value`` inputs to the appropriate type.

    Integer-looking values become int; float-looking values become
    float; everything else stays a string. This preserves the
    ``measured_values`` shape the loader expects.
    """
    # Try int first, then float, then string.
    s = raw.strip()
    try:
        if "." not in s and "e" not in s.lower() and "/" not in s:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_measured_flags(flags: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in flags:
        if "=" not in raw:
            raise SeedError(
                f"--measured expects key=value (got {raw!r})"
            )
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise SeedError(f"--measured key must be non-empty (got {raw!r})")
        out[k] = _coerce_measured_value(k, v)
    return out


def _classify_measured_source(
    parsed: dict[str, Any],
    flagged: dict[str, Any],
    required: set[str],
) -> str:
    """Return 'parsed' | 'operator_supplied' | 'mixed' per the plan contract.

    - All required fields came from the parser and no operator flags:
      ``parsed``.
    - All required fields came from operator flags (parser extracted
      nothing that the final record relies on): ``operator_supplied``.
    - Required fields came from both sources: ``mixed``.
    """
    parsed_keys = set(parsed) & required
    flagged_keys = set(flagged) & required
    if parsed_keys and not flagged_keys:
        return "parsed"
    if flagged_keys and not parsed_keys:
        return "operator_supplied"
    return "mixed"


def _confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        reply = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return reply in {"y", "yes"}


def build_record(
    *,
    report_path: Path,
    capability: str,
    reason: str,
    accepted_by: str,
    measured_flags: list[str],
    window_start: str,
    window_end: str,
) -> AttestationRecord:
    """Core logic — also callable from tests.

    Raises :class:`SeedError` on any failure that should exit non-zero.
    """
    if capability not in CAPABILITY_REQUIRED_FIELDS:
        raise SeedError(
            f"Unknown capability {capability!r}. Registered: "
            f"{sorted(CAPABILITY_REQUIRED_FIELDS)}"
        )
    if capability not in CAPABILITY_PARSERS:
        raise SeedError(
            f"No deterministic parser registered for capability {capability!r}"
        )

    if not report_path.exists():
        raise SeedError(f"Report file not found: {report_path}")

    if not reason or len(reason.strip()) < 20:
        raise SeedError(
            "--reason is required and must be at least 20 characters "
            "(a real explanation of why this attestation is being accepted)"
        )
    if not accepted_by.strip():
        raise SeedError("--accepted-by is required")

    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SeedError(f"Cannot read {report_path}: {exc}") from exc

    parser = CAPABILITY_PARSERS[capability]
    parsed = parser(text)
    flagged = _parse_measured_flags(measured_flags)

    measured: dict[str, Any] = {}
    measured.update(parsed)
    measured.update(flagged)  # operator flags override parsed values

    required = CAPABILITY_REQUIRED_FIELDS[capability]
    missing = required - set(measured.keys())
    if missing:
        raise SeedError(
            f"Deterministic parser could not extract all required fields for "
            f"{capability}: missing {sorted(missing)}. Supply them explicitly "
            "with --measured key=value flags, or fix the report format. "
            "(Fail-closed by design — see plan doc.)"
        )

    measured_source = _classify_measured_source(parsed, flagged, required)
    report_hash = "sha256:" + sha256_of_path(report_path)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return AttestationRecord(
        capability_id=capability,
        evidence_source=str(report_path),
        evidence_window_start=window_start or "",
        evidence_window_end=window_end or now,
        measured_values=measured,
        acceptance_reason=reason.strip(),
        accepted_by=accepted_by.strip(),
        accepted_at=now,
        report_hash=report_hash,
        artifact_refs=[str(report_path)],
        schema_version=SCHEMA_VERSION,
        measured_source=measured_source,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="seed_ever_proven_from_report",
        description=(
            "Seed a hash-attested operator-accepted capability attestation "
            "from a post-phase proof report."
        ),
    )
    p.add_argument(
        "--report",
        required=True,
        help="Path to the markdown report that contains the measured values",
    )
    p.add_argument(
        "--capability",
        required=True,
        help="Capability id, e.g. 'autonomy.l3'",
    )
    p.add_argument(
        "--reason",
        required=True,
        help=(
            "Human explanation for accepting this attestation "
            "(minimum 20 characters)"
        ),
    )
    p.add_argument(
        "--accepted-by",
        required=True,
        help="Identity accepting this attestation, e.g. 'operator:hostname'",
    )
    p.add_argument(
        "--window-start",
        default="",
        help="ISO8601 start of the evidence window (optional)",
    )
    p.add_argument(
        "--window-end",
        default="",
        help="ISO8601 end of the evidence window (defaults to now)",
    )
    p.add_argument(
        "--measured",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override or supply a measured value. Repeat for multiple "
            "fields. Any required field supplied via this flag tags the "
            "record's measured_source as 'operator_supplied' or 'mixed'."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the record that would be written and exit without writing",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow writing a duplicate (capability_id + report_hash match)",
    )
    p.add_argument(
        "--ledger-path",
        default=None,
        help="Override the ledger path (defaults to ~/.jarvis/eval/ever_proven_attestation.json)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        record = build_record(
            report_path=Path(args.report),
            capability=args.capability,
            reason=args.reason,
            accepted_by=args.accepted_by,
            measured_flags=args.measured,
            window_start=args.window_start,
            window_end=args.window_end,
        )
    except SeedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    display = record.to_persisted_dict()
    print("Attestation record to write:")
    print(json.dumps(display, indent=2, sort_keys=True))
    print(f"measured_source: {record.measured_source}")

    if args.dry_run:
        print("[dry-run] No write performed.")
        return 0

    if not _confirm(
        "Write this record to the attestation ledger?", assume_yes=args.yes
    ):
        print("Aborted by operator.")
        return 1

    ledger = AttestationLedger(
        path=Path(args.ledger_path) if args.ledger_path else None
    )
    try:
        ledger.add(record, force=args.force)
    except AttestationLedgerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(f"Wrote attestation to {ledger.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
