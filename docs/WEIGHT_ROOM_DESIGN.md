# The Weight-Room Discipline — Lived-Gated Promotion for Distilled Specialists

> Status: **DESIGN — not yet built.** Companion to `SPARK_DESIGN.md`. Authored 2026-05-31 from a
> 6-agent grounding + adversarial workflow (`wf_ab13fbad-89c`), every claim tied to file:line.
> Nothing here is implemented; this document exists to be read and approved before any code.

## 1. Thesis

The synthetic-training subsystem — the **weight room** — is built and good: 8+ isolated exercises
(`synthetic/*.py`), profiles (smoke/coverage/strict/stress), all "offshore" (fresh instances, no
LLM/TTS/memory/identity leaks, every signal tagged `origin="synthetic"` at capped `fidelity=0.7`),
feeding the `DistillationCollector` → Tier-1 student NNs. As a *training harness* it meets the bar.

What is missing is the **discipline a real gym enforces: you cannot earn a promotion by faking the
reps.** Today a synthetic-fed specialist can climb the promotion ladder on accuracy computed from
*mixed* synthetic+live data, with no requirement to have proven itself on real outcomes. The intended
rule — `AWAKENING_PROTOCOL.md:116`: *"lived baseline first, synthetic amplification second, promotion
third — never in reverse"* — exists **only as documentation**. No code enforces it.

This design makes that rule a code gate. The keystone reframe (forced by the adversarial pass): the
gate is **asymmetric** — synthetic training and shadow inference are *never* blocked; only the path to
**authority** (lifecycle promotion, broadcast-slot eligibility, `is_active`) is gated on a *lived*
baseline and *live-shadow* accuracy. You can lift in the gym all day; you just can't start in the game
until you've shown you can do it for real.

## 2. The precursor problem (why this can't start with the gate)

**You cannot gate on live-vs-synthetic until you can measure it — and today you can't.** The adversarial
"circularity" pass found that the signal the gate needs does not exist in the promotion path:

- `TeacherSignal` carries an `origin` field (`hemisphere/distillation.py:42-50`), but
  `DistillationCollector.get_stats()` (`distillation.py:201-215`) breaks down only by *teacher*, never
  by *origin* — the live/synthetic split lives in the JSONL and is invisible to any promotion decision.
- World-Model / Mental-Simulator promotion records outcomes as a bare `deque[float]` of hit/miss
  (`promotion.py`, `accuracy_history`), with **no origin metadata** — outcomes cannot even be
  relabelled live-vs-synthetic after the fact.
- `skill_acquisition_specialist.live_shadow_accuracy` is a **hardcoded `0.0`** and `promotion_eligible`
  a **hardcoded `False`** (`dashboard/snapshot.py:2596,2599`) — the "synthetic-vs-live separation" that
  looked like the right pattern is actually an unfilled stub.

### 2a. A real pre-existing leak (independent of this design — arguably fix first)
`world_model.py` calls `validate_predictions() → promotion.record_outcome()` every ~5s
*unconditionally* (`world_model.py:200-202`). `memory/gate.py:begin_synthetic_session()` blocks
`can_observation_write()` / `can_consolidation_write()` but **not** `record_outcome()`. So during a
synthetic session, **synthetic predictions already feed WorldModel and Simulator promotion accuracy**
(`SimulatorPromotion`, `promotion.py:227`, has the identical hole). This is a live self-deception path
today. Closing it is **P0** below and stands on its own merits even if the rest is deferred.

## 3. The five components

**(1) Origin-aware telemetry (the precursor).** Make live-vs-synthetic observable *everywhere a
promotion reads*. Add `origin_is_synthetic: bool` to `TeacherSignal` (default `True` — fail safe:
unmarked = synthetic), set `False` only on genuinely lived captures. Add `lived_sample_count` /
`synthetic_sample_count` per teacher and an origin breakdown to `get_stats()`. Change
`promotion.py:accuracy_history` from `deque[float]` to `deque[dict]` `{ts, outcome, origin}` with a
`get_accuracy(origin_filter="live")` method. Guard `record_outcome()` with
`memory_gate.synthetic_session_active()` (closes §2a). **Zero authority — instrumentation only.**

**(2) Live-shadow accuracy (fill the stub, generalise it).** Compute `live_shadow_accuracy` for every
specialist = accuracy over `origin=live` signals only, min-N gated. The real lived ground truth already
exists for at least skill_acquisition (`acquisition/orchestrator.py:2593-2625` `outcome_from_state()` —
job status + verification). Fill the `snapshot.py:2599` stub and surface both `synthetic_accuracy` and
`live_shadow_accuracy` on the dashboard (the lab distillation panel) so the gap is visible before it
gates anything.

**(3) Per-specialist lived-baseline registry (NOT one-size-fits-all).** The adversarial "false-block"
pass is decisive here: a uniform high floor would **permanently stall** rare-event specialists. So a
registry maps each specialist → `{lived_baseline_signal, min_lived_outcomes, mode}`:
- `language` classes → `LanguagePromotionGovernor.get_level()` (`language_promotion.py:154-160`) — the
  one mature signal that exists.
- `skill_acquisition`, `diagnostic`, `claim_classifier` → **rare-event**: low floors (3–20, *not* 100)
  or a **hybrid** regime (`synthetic_coverage ≥ 50 AND live_outcomes ≥ 3`).
- `thought_trigger_selector` (`types.py:402-420`) → **blocked-by-design**: it has *no live signal
  source* until `THOUGHT_VALIDATION_OUTCOME` emission exists (a separate build). It is *correct* to be
  shadow-only now; the registry records "blocked until feedback loop exists," not a floor it can never
  meet.
- specialists with no origin signal today (`capability_gate`/`claim_verdict` records nothing to the
  collector; `retrieval_log` has no origin field; contradiction is origin-blind) → flagged as
  **not-yet-gatable**; component (1) must instrument them first or they are explicitly exempt+logged.

**(4) The asymmetric authority gate.** Training and shadow inference are *never* blocked. Three
authority surfaces are gated, all on `(lived_baseline_met AND live_shadow_accuracy ≥ floor)`:
- **Lifecycle promotion** — insert at `orchestrator.py:779-783`, the `VERIFIED_PROBATIONARY →
  BROADCAST_ELIGIBLE` transition (cleanest point; leaves `PROBATIONARY_TRAINING` and verification
  untouched).
- **Broadcast-slot eligibility** — `get_hemisphere_signals()` / `_update_broadcast_slots()`
  (`orchestrator.py:1310-1370`). Critical: slot occupancy is *independent* of promotion today, so a
  shadow synthetic-only specialist can occupy slots 1–3 and influence the StateEncoder **before** any
  lived validation. Gating promotion without gating slots leaves the hole open.
- **`is_active`** — `registry.promote()` (`registry.py:100-112`). Also sync `is_active` with
  `_tier1_disabled` (a pre-existing desync: a specialist can be `is_active=True` yet disabled).

**(5) Governance spine.** Shadow-first, exactly like `cognition/promotion.py`: the gate first computes
**"would-block" decisions and logs them** with zero authority; it earns enforcement authority
gate-by-gate. **Fail-closed-to-shadow**: any error in the gate *blocks the promotion* (denies
authority) but never crashes training. Kill-switch. Grandfathering is explicit (next section).

## 4. Grandfathering & blast radius

Six always-on Tier-1 specialists are already promoted and have *no* `lifecycle_stage`
(`emotion_depth, speaker_repr, face_repr, perception_fusion, voice_intent, speaker_diarize`). Retro-
validating them is impossible (no historical origin record) and would regress working perception. So:
add `is_exempt_from_sequencing_gate: bool` to `NetworkArchitecture` (default `False`); at boot-restore,
set it `True` for a hardcoded exemption list; **log every exemption** (no silent grandfathering — the
honesty rule applies to the gate's own carve-outs). New specialists are *not* exempt. A new specialist
may inherit the lived signal of a promoted dependency without that dependency re-qualifying.

## 5. Honesty guardrails

- **Lived-only accuracy cannot be faked**: `origin` is set at capture from the real call path, never
  relabelled downstream; synthetic is the *default* (unmarked = synthetic).
- **Being-blocked is surfaced, not hidden**: every "would-block"/"blocked" decision is logged with its
  reason and shown on the dashboard.
- **Exemptions are explicit and logged** (§4).
- **The gate measures behaviour, not self-score**: live-shadow accuracy comes from real outcomes
  (`outcome_from_state`, language promotion, validated predictions), never from the specialist grading
  itself.

## 6. Phased rollout (smallest-safe-first; each gated on the prior)

- **P0 · Origin telemetry + close the leak (zero behaviour change).** Component (1). Add origin to
  `TeacherSignal`/`accuracy_history`, per-origin `get_stats()`, and guard `record_outcome()` with the
  synthetic-session check (§2a). *Advance when:* origin breakdown visible in snapshot; the
  synthetic→promotion leak is closed (a synthetic session no longer moves WorldModel/Simulator accuracy).
- **P1 · Live-shadow accuracy computed + surfaced (shadow).** Component (2). Fill the stub for all
  specialists; render `synthetic_accuracy` vs `live_shadow_accuracy` on the lab panel. *Advance when:*
  every specialist reports a real (or honestly-null) live-shadow number.
- **P2 · Lived-baseline registry + would-block logging (shadow).** Components (3)+(5). The gate computes
  decisions and logs "would have blocked promotion of X (lived baseline not met / live-shadow 0.0 <
  floor)" but enforces nothing. *Advance when:* the would-block log matches expectation and stalls no
  legitimate specialist (the false-block check, live).
- **P3 · Enforce promotion gate (advisory).** Component (4), lifecycle-promotion surface only. New
  promotions require the gate; existing specialists grandfathered. Training/shadow untouched.
- **P4 · Gate broadcast-slot eligibility.** Close the StateEncoder-influence hole — shadow specialists
  excluded from broadcast scoring until lived-validated.
- **P5 · Staleness re-validation.** `last_lived_sample_timestamp` + a periodic check: a specialist
  promoted long ago on sparse lived data that hasn't seen fresh lived signal in N days is flagged (and
  optionally re-shadowed). The "aged-out muscle" case.

## 7. Success metrics

- Every promoted/broadcasting specialist has `live_shadow_accuracy ≥ floor` AND
  `lived_outcome_count ≥ baseline` (or an explicit, logged exemption).
- **Zero synthetic-only promotions** after P3.
- The WorldModel/Simulator synthetic→promotion leak is closed (negative control: a synthetic session
  produces **no** change in promotion accuracy).
- No regression in the six grandfathered Tier-1 specialists (perception keeps working).
- Negative control on false-blocking: across a soak, no legitimate specialist is permanently stalled by
  the gate (rare-event specialists still reach authority via their low/hybrid floors, or are honestly
  marked blocked-by-design).

## 8. Open questions (decide before P3)

1. The per-specialist lived floors — exact values for the rare-event set (skill_acquisition,
   diagnostic, claim_classifier). Suggested starting points: hybrid `(synthetic ≥ 50 AND live ≥ 3)`.
2. `thought_trigger_selector` and `claim_classifier`: build their live feedback loops
   (`THOUGHT_VALIDATION_OUTCOME`; `capability_gate → DistillationCollector`) now, or leave them
   blocked-by-design and out of scope?
3. For any specialist that is *currently* promoted on synthetic-heavy data with no lived record: block
   only *new* promotions (grandfather it), or actively *re-shadow* it until it earns a live baseline?
4. Staleness window for P5 (30 days / 100 lived samples?), and demote-vs-flag on staleness.
5. Floor metric: live-shadow accuracy is a loss-derived proxy (`engine.py:269-282`), not true
   classification accuracy — is the proxy trustworthy enough to gate on, or does a specialist need a
   held-out live test set?

## 9. What this is NOT

Not a rewrite of the weight room (it's good). Not a block on synthetic training or shadow inference
(asymmetric by design). Not a new scoring system — it reuses existing lived signals
(`outcome_from_state`, language promotion, validated predictions). It is the missing *discipline layer*:
the rule that authority is earned on real reps, enforced in code, shadow-first, and reversible — the
same standard the language gates already hold for the voice, extended to the specialist NNs.
