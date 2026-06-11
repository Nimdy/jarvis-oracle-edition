# Matrix Protocol — End-to-End Verification (Tier-2 specialist lifecycle)

**Date:** 2026-06-11 · **Method:** signal-provenance trace + gate audit on the live brain
(`192.168.1.222:9200`) and source (`brain/hemisphere/orchestrator.py`). The skill-learning
analog: there we caught *fake scrapers*; here we ask whether the Matrix Protocol promotes
specialists on **real lived signal** or a synthetic/constant leak.

Companion: `MATRIX_PROTOCOL_GUIDE.md` (the honesty boundary — neural intuition, NOT skill upload).

## Verdict: REAL + SOLID on integrity. Two honest weaknesses (richness, persistence).

The lifecycle promotes genuine lived-signal specialists through honest gates. No synthetic leak.
The limits are *signal richness* (idle brain = thin signal) and *durability* (promotions don't
survive a reboot) — engineering gaps, not integrity failures.

## What the live brain showed (pre-reboot)
`GET /api/matrix`: **2 specialists PROMOTED** and sustained —
- `temporal_pattern`: promoted, acc 0.998, impact 0.88, broadcast_dwell 366 (gate=10), in_broadcast.
- `speaker_profile`: promoted, acc 1.000, impact 0.88, broadcast_dwell 366, in_broadcast.
- `promoted_count: 2` == `expansion_min_promoted: 2` (the self-improvement expansion threshold, met).
- The other 3 focuses `not_born` (distinct_labels < 2 → birth-variation gate holds).

So the autonomous lane drove the full ladder — birth → train → acc>0.5 → impact>0.3 →
broadcast → dwell≥10 → **PROMOTED** — and held it (dwell 366, not a flicker).

## Provenance: LIVED, not synthetic (the core integrity question)
The matrix signal buffer (`_matrix_signal_buffers`) is fed **only** by
`_observe_matrix_signals()` → `_matrix_encoder_for(focus)` → a per-focus `_build_*_context()`
that reads **live singletons**:
- `_build_speaker_profile_context` → `IdentityFusion.get_status` (identity/voice/face confidence,
  visible-person count, flip count), `IdentityResolver.get_known_names`, `soul_service` relationship
  **count**. Explicit *no-embedding-leak* contract (ECAPA/face vectors never cross), enforced by a
  regression test that feeds embedding-shaped keys and asserts the output is unchanged.
- `_build_temporal_pattern_context` → `memory_storage` write **timestamps** as an activity proxy
  (counts/time-deltas only, never content), mode-manager scalars. Privacy contract: the encoder is
  statically scanned for `hour_of_day`/`weekday`/`schedule`/`calendar` keys (none allowed).

**The synthetic exercisers** (`synthetic/claim_exercise.py`, `skill_acquisition_exercise.py`,
`diagnostic_exercise.py`, …) feed the **Tier-1 distillation/SI bus** (`/api/self-improve/specialists`)
— they never reach `_matrix_signal_buffers`. So Tier-2 matrix training is lived-only.

## Gates: honest (all fire on real data)
`orchestrator.py::_check_matrix_births` + `_check_specialist_promotions`:
1. **Birth-variation gate** — only birth a focus whose accumulated signal shows ≥2 regimes
   (`MATRIX_TRAIN_MIN_DISTINCT_LABELS`); a constant signal is untrainable and would hog the cap.
2. **acc > 0.5** (real NN trains its own encoder on the buffered (features,label) set).
3. **impact > 0.3** (`_matrix_focus_signal`).
4. **broadcast dwell ≥ 10** in the *sub-conscious matrix-vs-matrix lane* (`matrix_broadcast_slots`) —
   a separate lane by design, because matrix specialists structurally can't out-score the ~1.0
   Tier-1 nets in the main slots. Promotion is real *within that lane*; authority stays **advisory**.

## Persistence — DONE (weights survive reboot; authority re-earns). NOT a gap.
The post-reboot `acc=0.0, epoch=0, probationary_training` *looked* like "wiped" — tracing
`_restore_persisted_specialists` (`orchestrator.py:245`) proved the opposite. By firewall design,
matrix Tier-2 specialists get their trained **weights restored** from the registry on boot, but ALL
live standing resets (lifecycle→PROBATIONARY_TRAINING, accuracy/impact/verification→0) so they
**re-walk every gate on live reps** — a stale persisted accuracy must never skip a gate. The weights
are warm (fast rebuild, NOT retrain-from-zero); only *authority* re-earns. **Confirmed on disk:**
`~/.jarvis/hemispheres/temporal_pattern/` + `speaker_profile/` hold the checkpoints (both in
`registry_state.json` foci). The save side (`orchestrator.py:1086`) documents that this was the
original gap and is now CLOSED: *"Registering here makes the weights survive restart; P1 reloads them
at PROBATIONARY so the specialist rebuilds fast and re-earns authority (firewall intact)."*
> Process note: the prior memory said "Tier-2 SKIPPED → P1" — STALE. Verified against current code +
> disk before asserting. (gate-blocked ≠ broken; an EMPTY/reset-after-restart is expected, not a bug.)

## The one honest limitation — signal richness (idle brain)
Most context fields sit at defaults when no one is interacting. The ≥2-regime variation that enables
birth+training comes from sparse lived events (a person appearing, memory-write cadence shifting).
Hence only 2 of 5 focuses birth, and 0.998/1.0 accuracy = an *easily separable* task, not deep
mastery. The 3 `not_born` focuses (positive/negative_memory, skill_transfer) honestly wait for real
lived events with ≥2 regimes. This matches the GUIDE's "not kung-fu" boundary. **This is EARNED on
real interaction over time — it cannot be coded.**

## Recommended next
Integrity verified + persistence confirmed → the Tier-2 machinery is **SOLID**. What remains is not
engineering:
1. **Richer lived signal** for the `not_born` focuses — EARNED on real interaction (real positive *and*
   negative memory events, speaker variation → ≥2 regimes → birth). Can't be coded.
2. **Witnessed deliberate walk** — optionally observe the warm-weight re-climb to PROMOTED (fast, since
   weights are restored), capturing each gate transition as the documented end-to-end trace.
3. **Matrix v2 — Capability Domains** (the north-star) — isolated, deletable per-domain
   sub-consciousness. The big frontier once v1 is banked. See `docs/MATRIX_V2_CAPABILITY_DOMAINS.md`.

## Bottom line
The Matrix Protocol Tier-2 lifecycle is **verified real AND durable** — lived signal, honest gates, no
synthetic leak, advisory-only as designed, and the trained weights survive a reboot (authority
re-earns per the firewall). It is **not** a fake, a hand-flipped state, and it does **not** forget on
restart. The single honest limit is signal *richness* in an idle brain — earned through real lived
interaction, not code. The machinery is SOLID.
