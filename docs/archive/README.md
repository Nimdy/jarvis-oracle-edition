# Archive — superseded planning docs

These files are **historical and no longer active.** They are preserved for the
record (and for JARVIS's self-knowledge of its own past planning), not as a plan to
execute. **The single source of truth is now GitHub:**

- **Forward roadmap** → [GitHub Issues](https://github.com/Nimdy/jarvis-oracle-edition/issues) + [project board](https://github.com/users/Nimdy/projects/2)
- **Shipped changelog** → [GitHub Releases](https://github.com/Nimdy/jarvis-oracle-edition/releases) + [`docs/BUILD_HISTORY.md`](../BUILD_HISTORY.md)
- **Design/doctrine** → the `docs/*_DESIGN.md` / `*_GUIDE.md` files (still authoritative)

## Contents

| File | What it was | Status |
| --- | --- | --- |
| `TODO_2026-04-24.md` | "Runtime Continuation Plan (No Reset)" — the April active planning file | Superseded by `TODO_V2` then by GitHub Issues |
| `TODO_V2_2026-04-27.md` | "Post-Release Engineering Plan" — the successor lane catalog (P3.0–P3.16) | **Fully reconciled & superseded** (see below) |

## Reconciliation of `TODO_V2` against reality (validated 2026-06-09)

The April lane catalog was checked lane-by-lane against the shipped state:

- **P3.0–P3.5, P3.14, P3.16** — SHIPPED in April; in `BUILD_HISTORY.md` / releases.
- **P3.6–P3.10** (the five Tier-2 Matrix specialists) — shipped April at `CANDIDATE_BIRTH`
  only; have since **advanced to PROMOTED + weight-persistent** (June, v1.2.0). Tracked
  under the Matrix v2 EPIC (#27 / #32).
- **P3.12 long-horizon attention, P3.13 IntentionResolver Stage 1** — open/deferred in
  April; now tracked as **GitHub #18**.
- **P3.11 `dream_synthesis` promotion** — DEFERRED on data (the `quarantined` class had 0
  samples); still data-blocked, revisit when those samples exist. Not a separate issue.
- **P3.15** — observation-only, opens only if the pattern recurs ≥3× in a 7-day window.

Nothing actionable was lost in archiving: done items are in the changelog, open items
are GitHub issues, and the per-lane evidence lives in [`docs/validation_reports/`](../validation_reports/).
