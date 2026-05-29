# Phase 6.5 formal verification

This directory contains a TLA+ specification of the Phase 6.5 L3-governance
invariants and the TLC model-checker configuration used to prove them over
a bounded state space.

## Files

- `phase_65.tla` — the specification (single capability, one escalation
  lifecycle, bounded audit log).
- `phase_65.cfg` — TLC configuration (constants + invariants).
- `../validation_reports/tla_phase_65-YYYY-MM-DD.md` — evidence artifact
  from the most recent TLC run.

## What it proves

The six Phase 6.5 invariants, stated as TLA+ safety properties:

1. **`CurrentOkIsLiveSourced`** — `current_ok` is never assigned from the
   attestation ledger; it is re-derived from a live probe on every tick and
   on every restart. Structurally enforced by `RestartProcess` choosing a
   fresh boolean non-deterministically.
2. **`NoAutoPromotion`** — no audit-log entry ever records a
   `(prior=2, new=3)` transition whose action is anything other than
   `Apply`. `Apply` requires `phase = "approved"`, which requires a prior
   `Approve` action.
3. **`EvidenceClassSeparation`** — `current_ok`, `prior_attested_ok`, and
   `activation_ok` live in three distinct variables; no action copies
   between them.
4. **`AuditLogAppendOnly`** — `audit_log` only grows via `Append(...)` or
   shrinks via `RotateLedger`'s tail-slice; no action mutates an existing
   entry.
5. **`AttestationImmutability`** — once `SeedAttestation` runs, the
   attestation record is `frozen = TRUE` and no further action modifies
   it.
6. **`RequestOkDerivationInv`** — `request_ok = current_ok ∨
   prior_attested_ok` at every state.

## Reproducing

Install a TLA+ toolchain (TLC):

```
# Option 1: TLA+ Toolbox (GUI, bundles TLC)
# https://lamport.azurewebsites.net/tla/toolbox.html
#
# Option 2: tla2tools.jar CLI (what this artifact uses)
curl -LO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

Run TLC against the spec:

```
cd docs/formal
java -jar tla2tools.jar -config phase_65.cfg phase_65.tla
```

A successful run ends with something like:

```
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
    calculated (optimistic):  val = 1.2E-11
    based on the actual fingerprints:  val = 2.0E-12
```

Save stdout + stderr into
`docs/validation_reports/tla_phase_65-YYYY-MM-DD.md` along with the
constants used and a state-space summary (states discovered / distinct
states / diameter).

## Bounded state space

- `level ∈ {1, 2, 3}`
- `attestation_strength ∈ {none, archived_missing, verified}` — matches
  `brain/autonomy/attestation.py:71-72` (`STRENGTH_VERIFIED`,
  `STRENGTH_ARCHIVED_MISSING`) and the Python assertion in
  `tests/test_l3_snapshot_caches.py::test_archived_missing_strength_distinguished_from_verified`.
- `approval horizon ≤ MAX_APPROVALS = 4`
- `audit_log length ≤ MAX_AUDIT_LEN = 8` before `RotateLedger` trims.
- `clock ≤ MAX_AUDIT_LEN * 4 = 32` enforced by the TLC `CONSTRAINT
  StateBound` in `phase_65.cfg`. This prunes exploration past the bound
  without flagging a TypeOK violation.

Larger bounds are future work; the current bounds exhaust every reachable
state that matters for the six safety properties.

## Relationship to Python tests

The Python tests prove these invariants by example:

| Invariant | Python anchor |
|---|---|
| `CurrentOkIsLiveSourced` | `brain/tests/test_l3_snapshot_caches.py` (`current_ok` not cached across restart) |
| `NoAutoPromotion` | `brain/tests/test_l3_promotion_invariant.py` |
| `EvidenceClassSeparation` | `brain/tests/test_l3_escalation.py` + dashboard snapshot assertions |
| `AuditLogAppendOnly` | `brain/tests/test_autonomy_audit_ledger.py` |
| `AttestationImmutability` | `brain/tests/test_l3_escalation.py` (frozen flag) |
| `RequestOkDerivationInv` | `brain/dashboard/snapshot.py:1918-1948` + per-field assertions |

The TLA+ spec proves them by exhaustive model checking over the bounded
state space. This is a credibility anchor, not a replacement for the Python
tests; ship both.

## Guardrails

- The spec mirrors the code. If the spec disagrees with the code, the spec
  is wrong for this release; file a follow-up audit before touching the code.
- Do not add new variables or actions without updating the invariants —
  especially Invariant 3 (EvidenceClassSeparation).
- Bump `MAX_AUDIT_LEN` before `MAX_APPROVALS` if the state space feels too
  small; the audit log is the dominant fan-out.
