# Validation reports — historical evidence (not bundled in this snapshot)

`docs/BUILD_HISTORY.md` links to validation reports under this directory (e.g.
`skill_acquisition_hardening-2026-05-05.md`, `tla_phase_65-2026-04-25.md`,
`p3_5_m6_expansion_wiring-2026-04-25.md`, …).

**Those report files were generated during full development and live in the complete
dev history, not in this published snapshot.** They are *historical* evidence of work
that was done — not a claim about current runtime state. The published repo is a
squashed snapshot; the full history (with these reports) is the source of record.

For **current** validation, the runtime writes fresh reports to
`~/.jarvis/eval/validation_reports/` via:

```
PYTHONPATH=$(pwd) python -m scripts.run_validation_pack --output-dir ~/.jarvis/eval/validation_reports
```

— i.e. current verdicts are produced live and timestamped, never read from these
historical files. (Fidelity #12: don't reference artifacts as if present when they
are historical; "missing" here means "in the full history," stated plainly.)
