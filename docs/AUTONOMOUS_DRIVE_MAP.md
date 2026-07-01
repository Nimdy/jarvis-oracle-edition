# Autonomous Drive — ground-truth map (2026-07-01)

> Deep-dive of JARVIS's autonomous-growth stack (weight room + synthetic training + spark + self-sensing +
> autonomy/drive), agents grounded in the design + honesty docs first (no theater / no gaming / no bypassing
> firewalls or maturity gates / no polluting). This is the honest state + the single keystone + the
> architecture-respecting build. It heavily impacts JARVIS — every step here is earn-don't-declare, shadow-first.

## Honest current state — what actually self-drives
- **Autonomy L1 inward research (FIRES + EARNS):** at default level 1, `on_tick` autonomously fires
  MetricTrigger-deficit + DriveManager research/recall/audit/experiment, executes **read-only** queries,
  integrates findings via the conflict/upgrade path, credit-assigned by DeltaTracker→PolicyMemory. Earns lived
  deltas toward L1→L2. Cannot APPLY changes (L2+ is earned/dormant).
- **Self-sensing predictor (EARNS a signal, authority=none):** the ONE offline-PASS, non-operator,
  self-supervised causal growth signal she genuinely earns from her own senses. Runs every 2s. STARVED at a
  quiet desk = correct (event-bandwidth-limited), not a bug.
- **Dream-cycle synthetic amplification of memory_ranker:** the only self-fired synthetic training — and it
  depends on a human-seeded file; degrades to lived-only if absent.
- **Spark / Grounding Ring (FIRES its questions, OPERATOR-PULL close):** autonomously computes belief-tension
  and formulates external-validation questions every tick, but only EARNS when David answers `/v2/grounding`
  (external-validator-only gate). On answer it correctly mutates the belief (lived-triggered only). The
  autonomous close (P5b web-fire) is default-OFF and earns its own separate gate (0.70 accuracy) — may honestly
  never clear.
- **Shadow/gated by design (NOT bugs):** weight_room_gate (`enforces=False`, no promotion transition calls it;
  nothing birthed to gate); L2 safe-apply / L3 self-mod (earned; L3 auto forbidden by invariant); all 11
  synthetic gyms (operator-pull).

## THE KEYSTONE GAP
JARVIS's one **validated, non-operator, causal growth signal** (self-sensing `learning_progress` /
`curiosity_target`) is **computed-then-DISCARDED** — zero consumers. Meanwhile the drive that DOES fire
autonomously (curiosity) is fueled only by evolution-engine `novelty_events`, not by this signal
(`DriveSignals` has no self-sensing field). **The proven fuel and the firing engine are structurally
disconnected.** That is the single break between "produces a real growth signal" and "that signal makes her go
attend/explore/learn on her own." It is literally the strategy doc's recommendation #3 and self-sensing's own
stated "Phase 2 (next)". Everything else (weight-room enforcement, synthetic scheduling, L2/L3, P5b) is either
downstream of having a real self-fired signal, or is a loop the evidence says can't causally turn (policy NN,
synthetic gyms).

## Highest-leverage next step — the shadow self-sensing → curiosity bridge
Two thin, additive, **shadow**, firewalled parts:
1. **`DriveSignals` gains a read-only field** carrying self-sensing `learning_progress` + `curiosity_target`,
   populated at the orchestrator's signal-assembly point (where `novelty_events` is set) from
   `self_sensing.get_status()`. Ends the computed-then-discarded waste; changes nothing about urgency yet.
2. **A shadow attention/exploration PROPOSER** that reads `curiosity_target` and logs "she WOULD attend/explore
   sector X because that's where she's learning most" — would-have telemetry, `authority=none`, influences
   nothing (mirrors weight_room's would-block + grounding's selections_shadowed). Logs proposals AND no-target
   ticks.

Then the earn ladder: soak → **offline critic-test** (does attending the LP-target sector actually reduce
next-frame prediction error beyond persistence, surviving shuffle/negative controls?) → only after an
**externally-attributed causal win** does it earn advisory-nudge authority (kill-switch, fail-closed-to-shadow,
never a rising self-score). If the critic-test FAILS → shadow-forever is the honest success.

## What it MUST NOT do (guardrails — a change breaking any of these is WRONG for JARVIS)
- MUST NOT flip self_sensing authority (keep the 3 hardcoded `False`); no rising self-score earns authority.
- MUST NOT skip the critic-test; no wiring to real urgency on faith.
- MUST NOT weaken integrity controls to read EARNING (persistence baseline, motion gate, DYNAMIC_THRESH,
  STARVED/FAILED regimes stay) — a quiet desk STAYS STARVED (signage, never §24 tuning).
- MUST NOT write canonical memory/belief or earn weight-room authority (lived-before-synthetic; asymmetric gate
  trains but only lived externally-attributed reps earn).
- MUST NOT touch the grounding ladder to force the spark to fire (no flipping GroundingDrivePromotion level, no
  lowering GROUNDING_MIN_OUTCOMES/rate, no reusing the grounding gate; `_ground_belief` stays operator-only).
- MUST NOT raise autonomy_level, auto-promote L1→L2 (needs 10 lived deltas @win_rate≥0.4, warmup/synthetic
  excluded), or ever auto-reach L3 (invariant).
- MUST NOT manufacture a target under starvation (`curiosity_target=None` on a static scene is the
  non-fabrication firewall) or log silently.

## Sequencing
- **STEP 0 (verify, no code):** read the REMOTE brain's live `/api/grounding-ring` + `/api/self-sensing` —
  confirm self-sensing is at least sometimes EARNING/marginal (not permanently STARVED) so the bridge has fuel;
  the 8-day-old grounding level=2 claim is unverifiable from the snapshot.
- **STEP 1:** DriveSignals shadow field (additive, reversible).
- **STEP 2:** shadow would-attend proposer + read-only /v2 surfacing.
- **STEP 3:** soak + offline critic-test (shuffle/negative controls). FAIL → shadow-forever (honest).
- **STEP 4 (only if STEP 3 passes with an externally-attributed win):** advisory-nudge real curiosity urgency,
  kill-switch, negative-control mandatory.
- **STEP 5 (separate, optional):** earn P5b autonomous web-fire on its own gate (operator's flip) — don't
  conflate with the senses bridge.
- **STEP 6:** persist the shadow ledger + boot-rehydrate (survives reboots).

## Risks (all honest-dormancy, not tune-to-green)
Event-bandwidth ceiling (quiet desk → slow/dormant = honest); critic-test may fail (→ shadow-forever);
beautiful-telemetry trap (bind STEP 4 to a real causal win); firewall scope-creep; stale-snapshot decisions
(do STEP 0); conflating the keystone with the spark's operator-close (forbidden to force-fire grounding).
