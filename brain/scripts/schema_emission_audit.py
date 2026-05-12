#!/usr/bin/env python3
"""Schema emission audit (P1.3).

Walks four declared-enum surfaces and verifies that each declared value is
either:

  (a) **Emitted** — used as a string literal in at least one writer file
      under ``brain/`` (e.g. ``edge_type="supports"``), AND/OR appears at
      least once in the corresponding on-disk evidence file, OR
  (b) **Whitelisted as future-only** — explicitly listed in
      ``FUTURE_ONLY_*`` below as scaffolding that does not yet have a
      live writer. Every future-only entry MUST carry a one-line comment
      explaining why it is parked.

Anything else is a schema violation:

  - **DECLARED_BUT_NEVER_EMITTED** — declared in the enum, no writer
    literal anywhere in ``brain/``, no rows on disk, and not future-only.
    This is the "vaporware schema slot" failure mode.
  - **EMITTED_BUT_NOT_DECLARED** — present in the on-disk evidence file
    but missing from the declared enum. This is the "schema drifted"
    failure mode.

Surfaces audited
----------------

1. ``epistemic.belief_graph.edges.VALID_EDGE_TYPES``
2. ``epistemic.belief_graph.edges.VALID_EVIDENCE_BASES``
3. ``hemisphere.types.HemisphereFocus``
4. Distillation teacher keys declared in
   ``jarvis_eval.dashboard_adapter._SPECIALIST_TO_TEACHERS`` (auto-extracted).

This script is intentionally:

  - read-only (never writes)
  - deterministic (uses pure source-grep + on-disk JSONL counts)
  - importable (``run_audit()`` returns a structured report so tests can
    consume it without invoking the CLI)

Usage
-----

    python -m scripts.schema_emission_audit
    python -m scripts.schema_emission_audit --json
    python -m scripts.schema_emission_audit --strict   # nonzero exit on any
                                                       # violation
    python -m scripts.schema_emission_audit --include-future-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Future-only whitelists.
#
# Each entry MUST carry a one-line justification. The audit treats these as
# "declared scaffolding without a live writer yet" instead of failures.
# When a writer lands, REMOVE the entry from the whitelist so the audit can
# enforce that the writer keeps emitting.
# ---------------------------------------------------------------------------

FUTURE_ONLY_EVIDENCE_BASES: dict[str, str] = {
    # Operator-correction writer landed in ``GraphBridge
    # .create_user_correction_link`` (P3.2). Remaining evidence bases are
    # all wired to live writers, so the whitelist is intentionally empty.
}

FUTURE_ONLY_EDGE_TYPES: dict[str, str] = {
    # Prerequisite-tracking writer landed in ``GraphBridge
    # .create_prerequisite_link`` (P3.3). All declared edge types now
    # have live writers, so the whitelist is intentionally empty.
}

FUTURE_ONLY_HEMISPHERE_FOCUSES: dict[str, str] = {
    # ``custom`` is an operator-only escape hatch; never expected to be
    # spawned by the architect on its own.
    "custom": "operator-only escape hatch, not auto-spawned",
    # The remaining Tier-2 Matrix Protocol specialists are declared in
    # the enum but are probationary — they only spawn after their
    # respective promotion gates are satisfied. Until those gates fire
    # in live runtime they have no writer literal in source; that is
    # correct behavior, not a schema violation.
    #
    # ``positive_memory`` was removed from this whitelist on 2026-04-25
    # when ``brain/hemisphere/positive_memory_encoder.py`` shipped its
    # writer literal (P3.6). The encoder is shadow-only / CANDIDATE_BIRTH
    # and does NOT promote to live broadcast slots without the Matrix
    # lifecycle clearing.
    #
    # ``negative_memory`` was removed from this whitelist on 2026-04-25
    # when ``brain/hemisphere/negative_memory_encoder.py`` shipped its
    # writer literal (P3.7). Same shadow-only / CANDIDATE_BIRTH contract
    # as positive_memory.
    #
    # ``speaker_profile`` was removed from this whitelist on 2026-04-25
    # when ``brain/hemisphere/speaker_profile_encoder.py`` shipped its
    # writer literal (P3.8). Same shadow-only / CANDIDATE_BIRTH contract.
    # The encoder enforces a strict no-raw-embedding-leak boundary —
    # only ``IdentityFusion.get_status()``-derived scalars and counts
    # cross the input.
    #
    # ``temporal_pattern`` was removed from this whitelist on 2026-04-25
    # when ``brain/hemisphere/temporal_pattern_encoder.py`` shipped its
    # writer literal (P3.9). Same shadow-only / CANDIDATE_BIRTH contract.
    # The encoder enforces a strict no-schedule-claim boundary — only
    # bounded recency / cadence / mode-stability scalars cross the
    # input; never hour-of-day, weekday, or per-speaker schedule fact.
    #
    # ``skill_transfer`` was removed from this whitelist on 2026-04-25
    # when ``brain/hemisphere/skill_transfer_encoder.py`` shipped its
    # writer literal (P3.10). Same shadow-only / CANDIDATE_BIRTH
    # contract. The encoder enforces a strict
    # similarity-is-not-capability boundary — it never promotes a
    # capability, never marks a skill as verified, and never consumes
    # a "promote-this-skill" hint. Capability promotion remains the
    # sole authority of the existing capability_gate path.
    # HRR / VSA encoder (P4) is a Tier-1 stub with NO distillation teacher
    # and no network backing it. It is wired into ``_TIER1_FOCUSES`` by the
    # hemisphere orchestrator via the enum member (not a string literal),
    # so the writer-literal check intentionally returns no match until the
    # specialist is referenced by name (e.g. a future scheduler rule).
    # This is an expected PRE-MATURE state under P4.5 governance, not a
    # schema violation. Remove once HRR earns Tier-2 promotion or a writer
    # literal is introduced.
    "hrr_encoder": "Tier-1 PRE-MATURE stub, dormant under ENABLE_HRR_SHADOW",
    "thought_trigger_selector": "Tier-1 shadow-only stub, no teacher signal wired yet (Phase 2 thought maturity roadmap)",
}

FUTURE_ONLY_TEACHER_KEYS: dict[str, str] = {
    # No future-only teachers right now. Every teacher key in the
    # specialist-to-teacher map is wired to a live distillation buffer.
}

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_FAIL = "fail"


@dataclass
class SurfaceFinding:
    surface: str
    value: str
    classification: str  # "emitted" | "future_only" | "declared_not_emitted"
                         # | "emitted_not_declared"
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "value": self.value,
            "classification": self.classification,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class SurfaceReport:
    surface: str
    declared: list[str]
    emitted: list[str]
    future_only: list[str]
    findings: list[SurfaceFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "declared_count": len(self.declared),
            "emitted_count": len(self.emitted),
            "future_only_count": len(self.future_only),
            "violations": [
                f.to_dict() for f in self.findings
                if f.severity == SEVERITY_FAIL
            ],
            "all_findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class AuditReport:
    surfaces: list[SurfaceReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surfaces": [s.to_dict() for s in self.surfaces],
            "violation_count": sum(
                1
                for s in self.surfaces
                for f in s.findings
                if f.severity == SEVERITY_FAIL
            ),
        }


# ---------------------------------------------------------------------------
# Source / evidence scanning helpers
# ---------------------------------------------------------------------------


def _brain_root() -> Path:
    """Return the ``brain/`` directory regardless of how the script is run."""
    here = Path(__file__).resolve()
    # brain/scripts/schema_emission_audit.py -> brain/
    return here.parent.parent


_SELF_PATH = Path(__file__).resolve()


def _iter_python_sources(root: Path) -> Iterable[Path]:
    """Iterate ``.py`` files under ``root`` (excluding tests, venvs, caches).

    Also skips the audit script itself so that the FUTURE_ONLY_*
    whitelists (which contain the declared values as dictionary keys) do
    not falsely credit themselves as writer literals.
    """
    skip_parts = {
        "__pycache__", ".venv", "venv", ".git", "tests", "test",
        "node_modules", "build", "dist",
    }
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & skip_parts:
            continue
        if path.name.startswith("test_"):
            continue
        if path.resolve() == _SELF_PATH:
            continue
        yield path


def _grep_literals(root: Path, pattern: re.Pattern[str]) -> set[str]:
    """Return the set of capture-group-1 values matched in any source file."""
    found: set[str] = set()
    for path in _iter_python_sources(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in pattern.finditer(text):
            found.add(m.group(1))
    return found


def _count_jsonl_field(path: Path, field_name: str) -> dict[str, int]:
    """Count occurrences of ``record[field_name]`` values in a JSONL file."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = rec.get(field_name)
                if isinstance(v, str):
                    counts[v] = counts.get(v, 0) + 1
    except Exception:
        return counts
    return counts


# ---------------------------------------------------------------------------
# Surface 1 + 2: belief-graph edge schema
# ---------------------------------------------------------------------------


_EDGE_TYPE_LITERAL = re.compile(r'edge_type\s*=\s*"([^"]+)"')
_EVIDENCE_BASIS_LITERAL = re.compile(r'evidence_basis\s*=\s*"([^"]+)"')


_WRITER_CALL_HINT = re.compile(
    r"\b(make_edge|EvidenceEdge|_create_edges_from_recent_resolution)\b"
)


def _scan_quoted_values_in(
    root: Path,
    candidates: set[str],
    exclude: set[Path] | None = None,
) -> set[str]:
    """Return the subset of ``candidates`` that appear as quoted literals
    in files that also reference a known edge-writer call site.

    Restricting the scan to files that mention ``make_edge``,
    ``EvidenceEdge``, or the bridge's positional helper avoids crediting
    pure readers (propagation, integrity, topology) as writers, which
    would silently mask DECLARED_BUT_NEVER_EMITTED violations for edge
    types that are only consumed.
    """
    if not candidates:
        return set()
    excluded = {p.resolve() for p in (exclude or set())}
    patterns = {v: re.compile(rf'["\']({re.escape(v)})["\']') for v in candidates}
    found: set[str] = set()
    for path in _iter_python_sources(root):
        if path.resolve() in excluded:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not _WRITER_CALL_HINT.search(text):
            continue
        for v, pat in patterns.items():
            if v in found:
                continue
            if pat.search(text):
                found.add(v)
        if len(found) == len(candidates):
            break
    return found


def _audit_belief_graph_schema(
    brain_root: Path,
    evidence_root: Path,
) -> tuple[SurfaceReport, SurfaceReport]:
    from epistemic.belief_graph.edges import (
        VALID_EDGE_TYPES,
        VALID_EVIDENCE_BASES,
    )

    edge_writers = _grep_literals(brain_root, _EDGE_TYPE_LITERAL)
    basis_writers = _grep_literals(brain_root, _EVIDENCE_BASIS_LITERAL)

    # Fallback: edge_type values are sometimes passed positionally through
    # ``_create_edges_from_recent_resolution(conflict_type, "contradicts")``
    # instead of as a keyword argument. Scan the belief-graph subtree for
    # quoted occurrences so positional-only writers are still credited.
    # The declaration site (``edges.py``) is excluded so the
    # ``VALID_EDGE_TYPES`` / ``VALID_EVIDENCE_BASES`` set literals do not
    # credit themselves.
    bg_root = brain_root / "epistemic" / "belief_graph"
    declaration_site = (bg_root / "edges.py").resolve()
    if bg_root.exists():
        edge_writers |= _scan_quoted_values_in(
            bg_root, set(VALID_EDGE_TYPES), exclude={declaration_site}
        )
        basis_writers |= _scan_quoted_values_in(
            bg_root, set(VALID_EVIDENCE_BASES), exclude={declaration_site}
        )

    edge_runtime = _count_jsonl_field(
        evidence_root / "belief_edges.jsonl", "edge_type"
    )
    basis_runtime = _count_jsonl_field(
        evidence_root / "belief_edges.jsonl", "evidence_basis"
    )

    edges_report = _build_surface_report(
        surface="VALID_EDGE_TYPES",
        declared=set(VALID_EDGE_TYPES),
        writers=edge_writers,
        runtime_counts=edge_runtime,
        future_only=FUTURE_ONLY_EDGE_TYPES,
    )
    bases_report = _build_surface_report(
        surface="VALID_EVIDENCE_BASES",
        declared=set(VALID_EVIDENCE_BASES),
        writers=basis_writers,
        runtime_counts=basis_runtime,
        future_only=FUTURE_ONLY_EVIDENCE_BASES,
    )
    return edges_report, bases_report


# ---------------------------------------------------------------------------
# Surface 3: HemisphereFocus enum
# ---------------------------------------------------------------------------


def _audit_hemisphere_focus(brain_root: Path) -> SurfaceReport:
    from hemisphere.types import HemisphereFocus

    declared = {member.value for member in HemisphereFocus}

    # A focus is "emitted" if its string value appears as a literal in any
    # non-test source file (excluding ``brain/hemisphere/types.py`` itself,
    # since that is the declaration site, not a writer).
    types_path = (brain_root / "hemisphere" / "types.py").resolve()

    emitted: set[str] = set()
    for value in declared:
        # Look for the value as a quoted literal.
        pattern = re.compile(rf'"({re.escape(value)})"')
        for path in _iter_python_sources(brain_root):
            if path.resolve() == types_path:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if pattern.search(text):
                emitted.add(value)
                break

    return _build_surface_report(
        surface="HemisphereFocus",
        declared=declared,
        writers=emitted,
        runtime_counts={},
        future_only=FUTURE_ONLY_HEMISPHERE_FOCUSES,
    )


# ---------------------------------------------------------------------------
# Surface 4: distillation teacher keys
# ---------------------------------------------------------------------------


# Mirror of ``_SPECIALIST_TO_TEACHERS`` defined inside
# ``jarvis_eval.dashboard_adapter`` (it is a function-local). Kept in sync
# by hand because the audit must remain importable without initializing
# the full dashboard adapter call site.
SPECIALIST_TO_TEACHERS: dict[str, list[str]] = {
    "speaker_repr": ["ecapa_tdnn"],
    "face_repr": ["mobilefacenet"],
    "emotion_depth": ["wav2vec2_emotion"],
    "voice_intent": ["tool_router"],
    "speaker_diarize": ["ecapa_tdnn"],
    "perception_fusion": ["ecapa_tdnn", "mobilefacenet", "wav2vec2_emotion"],
}


def _audit_teacher_keys(brain_root: Path, evidence_root: Path) -> SurfaceReport:
    declared: set[str] = set()
    for teachers in SPECIALIST_TO_TEACHERS.values():
        for tk in teachers:
            declared.add(tk)

    # A teacher key is considered emitted if it appears as a string literal
    # in any non-test source under ``brain/``. This is a best-effort
    # static check; the canonical runtime check is via on-disk
    # distillation buffers below.
    emitted: set[str] = set()
    for value in declared:
        pattern = re.compile(rf'"({re.escape(value)})"')
        for path in _iter_python_sources(brain_root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if pattern.search(text):
                emitted.add(value)
                break

    # Runtime check: any ``distill_<teacher>.jsonl`` file with at least one
    # row counts the teacher as emitted.
    runtime_counts: dict[str, int] = {}
    for value in declared:
        candidate = evidence_root / f"distill_{value}.jsonl"
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    runtime_counts[value] = sum(1 for line in f if line.strip())
            except Exception:
                pass

    return _build_surface_report(
        surface="distillation.teacher_keys",
        declared=declared,
        writers=emitted,
        runtime_counts=runtime_counts,
        future_only=FUTURE_ONLY_TEACHER_KEYS,
    )


# ---------------------------------------------------------------------------
# Generic surface report builder
# ---------------------------------------------------------------------------


def _build_surface_report(
    surface: str,
    declared: set[str],
    writers: set[str],
    runtime_counts: dict[str, int],
    future_only: dict[str, str],
) -> SurfaceReport:
    findings: list[SurfaceFinding] = []
    emitted_set: set[str] = set()

    for value in sorted(declared):
        has_writer = value in writers
        runtime_n = runtime_counts.get(value, 0)
        if has_writer or runtime_n > 0:
            emitted_set.add(value)
            findings.append(SurfaceFinding(
                surface=surface, value=value,
                classification="emitted",
                severity=SEVERITY_INFO,
                detail=(
                    f"writer_literal={'yes' if has_writer else 'no'} "
                    f"runtime_rows={runtime_n}"
                ),
            ))
        elif value in future_only:
            findings.append(SurfaceFinding(
                surface=surface, value=value,
                classification="future_only",
                severity=SEVERITY_WARN,
                detail=f"whitelisted: {future_only[value]}",
            ))
        else:
            findings.append(SurfaceFinding(
                surface=surface, value=value,
                classification="declared_not_emitted",
                severity=SEVERITY_FAIL,
                detail=(
                    "declared in enum but no writer literal in brain/ and "
                    "no runtime rows on disk"
                ),
            ))

    # Reverse direction: anything in runtime_counts but not declared.
    for value, n in sorted(runtime_counts.items()):
        if value in declared:
            continue
        findings.append(SurfaceFinding(
            surface=surface, value=value,
            classification="emitted_not_declared",
            severity=SEVERITY_FAIL,
            detail=f"on-disk rows={n} but value missing from declared enum",
        ))

    return SurfaceReport(
        surface=surface,
        declared=sorted(declared),
        emitted=sorted(emitted_set),
        future_only=sorted(future_only.keys()),
        findings=findings,
    )


# ---------------------------------------------------------------------------
# Top-level audit entry point
# ---------------------------------------------------------------------------


def run_audit(evidence_root: Path | None = None) -> AuditReport:
    """Run the full audit and return a structured report."""
    brain_root = _brain_root()
    if evidence_root is None:
        evidence_root = Path(os.path.expanduser("~/.jarvis"))
    edges_report, bases_report = _audit_belief_graph_schema(
        brain_root, evidence_root
    )
    hemisphere_report = _audit_hemisphere_focus(brain_root)
    teacher_report = _audit_teacher_keys(brain_root, evidence_root)
    return AuditReport(
        surfaces=[edges_report, bases_report, hemisphere_report, teacher_report]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_human(report: AuditReport, include_future_only: bool) -> str:
    lines: list[str] = []
    lines.append("Schema Emission Audit")
    lines.append("=" * 60)
    total_violations = 0
    for surface in report.surfaces:
        lines.append("")
        lines.append(
            f"{surface.surface}: declared={len(surface.declared)} "
            f"emitted={len(surface.emitted)} "
            f"future_only={len(surface.future_only)}"
        )
        for f in surface.findings:
            if f.severity == SEVERITY_INFO and not include_future_only:
                continue
            if f.classification == "future_only" and not include_future_only:
                marker = "[future-only]"
            elif f.severity == SEVERITY_FAIL:
                marker = "[VIOLATION]"
                total_violations += 1
            else:
                marker = "[ok]"
            lines.append(f"  {marker} {f.value}: {f.detail}")
    lines.append("")
    lines.append(f"Total violations: {report.to_dict()['violation_count']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit declared enum surfaces against actual writers."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero if any DECLARED_BUT_NEVER_EMITTED or "
            "EMITTED_BUT_NOT_DECLARED violations are present."
        ),
    )
    parser.add_argument(
        "--include-future-only",
        action="store_true",
        help="Include future-only and emitted entries in human output.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Override the evidence root (default: ~/.jarvis).",
    )
    args = parser.parse_args(argv)

    report = run_audit(evidence_root=args.evidence_root)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(_format_human(report, include_future_only=args.include_future_only))

    if args.strict and report.to_dict()["violation_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
