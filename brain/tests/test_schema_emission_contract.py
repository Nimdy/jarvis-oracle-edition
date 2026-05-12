"""Regression tests for ``brain/scripts/schema_emission_audit.py``.

Two contracts are enforced here:

  1. **Live brain contract** — running the audit against the real
     ``brain/`` source tree (with no on-disk evidence override) MUST find
     no DECLARED_BUT_NEVER_EMITTED or EMITTED_BUT_NOT_DECLARED violations
     for the four declared surfaces. Future-only entries are tolerated
     because they carry an explicit whitelist comment.
  2. **Detector contract** — when we synthetically inject a fake
     declared-but-never-emitted value into the underlying enum, the audit
     MUST surface it as a violation. This proves the audit is not silent.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts import schema_emission_audit


# ---------------------------------------------------------------------------
# Live contract
# ---------------------------------------------------------------------------


class TestLiveSchemaEmissionContract:

    def test_no_violations_on_live_brain_source(self):
        # Use an empty evidence root so the audit relies entirely on
        # source-grep. This protects against a flaky test that depends on
        # ~/.jarvis/ contents.
        with tempfile.TemporaryDirectory() as td:
            report = schema_emission_audit.run_audit(evidence_root=Path(td))

        violations = [
            f
            for s in report.surfaces
            for f in s.findings
            if f.severity == schema_emission_audit.SEVERITY_FAIL
        ]
        assert violations == [], (
            "Schema audit reported violations on live source tree:\n"
            + "\n".join(
                f"  {f.surface}::{f.value} [{f.classification}] {f.detail}"
                for f in violations
            )
        )

    def test_belief_graph_surfaces_present(self):
        with tempfile.TemporaryDirectory() as td:
            report = schema_emission_audit.run_audit(evidence_root=Path(td))
        surfaces = {s.surface for s in report.surfaces}
        assert "VALID_EDGE_TYPES" in surfaces
        assert "VALID_EVIDENCE_BASES" in surfaces
        assert "HemisphereFocus" in surfaces
        assert "distillation.teacher_keys" in surfaces

    def test_temporal_sequence_and_causal_are_emitted(self):
        # P1.3 acceptance: the writers must drive these into the
        # "emitted" set in the source-grep pass.
        with tempfile.TemporaryDirectory() as td:
            report = schema_emission_audit.run_audit(evidence_root=Path(td))

        for surface in report.surfaces:
            if surface.surface != "VALID_EVIDENCE_BASES":
                continue
            assert "temporal_sequence" in surface.emitted, (
                f"temporal_sequence not detected as emitted; "
                f"emitted={surface.emitted}"
            )
            assert "causal" in surface.emitted, (
                f"causal not detected as emitted; emitted={surface.emitted}"
            )


# ---------------------------------------------------------------------------
# Detector contract
# ---------------------------------------------------------------------------


class TestDetectorContract:

    def test_synthetic_declared_not_emitted_is_caught(self):
        # Inject a fake value into VALID_EVIDENCE_BASES and confirm the
        # audit flags it as DECLARED_BUT_NEVER_EMITTED. We never call
        # ``add()`` with it and we never write it to evidence, so it is
        # truly orphaned in the schema.
        from epistemic.belief_graph import edges as edges_mod

        original = edges_mod.VALID_EVIDENCE_BASES
        synthetic = frozenset(
            list(original) + ["bogus_basis_for_audit_test"]
        )
        edges_mod.VALID_EVIDENCE_BASES = synthetic
        try:
            with tempfile.TemporaryDirectory() as td:
                report = schema_emission_audit.run_audit(
                    evidence_root=Path(td)
                )
            violations = [
                f
                for s in report.surfaces
                for f in s.findings
                if (
                    f.severity == schema_emission_audit.SEVERITY_FAIL
                    and f.value == "bogus_basis_for_audit_test"
                )
            ]
            assert len(violations) == 1
            assert violations[0].classification == "declared_not_emitted"
        finally:
            edges_mod.VALID_EVIDENCE_BASES = original

    def test_synthetic_emitted_not_declared_is_caught(self):
        # Drop ``shared_subject`` from the declared set so on-disk evidence
        # appears emitted-but-not-declared. We synthesize a one-line
        # belief_edges.jsonl with that basis to exercise the runtime path.
        from epistemic.belief_graph import edges as edges_mod

        original = edges_mod.VALID_EVIDENCE_BASES
        shrunk = frozenset(v for v in original if v != "shared_subject")
        edges_mod.VALID_EVIDENCE_BASES = shrunk
        try:
            with tempfile.TemporaryDirectory() as td:
                # Write a synthetic on-disk row to trigger the
                # "emitted_not_declared" path.
                ev_root = Path(td)
                jsonl = ev_root / "belief_edges.jsonl"
                jsonl.write_text(
                    '{"edge_type": "supports", "evidence_basis": '
                    '"shared_subject"}\n',
                    encoding="utf-8",
                )
                report = schema_emission_audit.run_audit(evidence_root=ev_root)

            violations = [
                f
                for s in report.surfaces
                for f in s.findings
                if (
                    f.severity == schema_emission_audit.SEVERITY_FAIL
                    and f.classification == "emitted_not_declared"
                    and f.value == "shared_subject"
                )
            ]
            assert len(violations) == 1
        finally:
            edges_mod.VALID_EVIDENCE_BASES = original

    def test_strict_mode_exits_nonzero_on_violations(self):
        from epistemic.belief_graph import edges as edges_mod

        original = edges_mod.VALID_EVIDENCE_BASES
        edges_mod.VALID_EVIDENCE_BASES = frozenset(
            list(original) + ["another_bogus_basis"]
        )
        try:
            with tempfile.TemporaryDirectory() as td:
                rc = schema_emission_audit.main(
                    ["--json", "--strict", "--evidence-root", td]
                )
            assert rc == 1
        finally:
            edges_mod.VALID_EVIDENCE_BASES = original

    def test_strict_mode_exits_zero_on_clean_audit(self):
        with tempfile.TemporaryDirectory() as td:
            rc = schema_emission_audit.main(
                ["--json", "--strict", "--evidence-root", td]
            )
        assert rc == 0
