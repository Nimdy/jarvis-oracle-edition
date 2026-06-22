# P5 — Autonomous Research Firing (DESIGN, shadow-first / earn-don't-declare)

> Status: **DESIGN / PRE-MATURE**. Extends the SPARK / grounding-ring (docs/SPARK_DESIGN.md). This is
> the earned step from **operator-pull research questions → JARVIS researching on her own**. Nothing
> here grants authority on day one; every capability earns its way up a shadow ladder. Held inside the
> existing autonomy governance (dedup gate, governor mode-gate, rate limits, operator kill switch).

## The north-star insight (David)
*"If JARVIS fully understands her codebase, she should be able to research improvements."* Today she
already **generates** the right outward-facing questions — the grounding drive is earned and active —
but they sit in an **operator-pull queue** (`/v2/grounding`) for David to answer. **P5 = she fires the
research herself** (the academic / Semantic-Scholar tool), validates beliefs against external truth,
and — when her own codebase research is exhausted — escalates outward for improvement approaches.
Earned, never declared.

## Current state (P4 — verified)
- `autonomy/grounding_queue.py`: `phase = P4_advisory_async_queue`, **`auto_fires = False`**, OPERATOR-GATED.
  Scoring/enqueue are view-only; the single mutating action is an **operator answer**.
- `GroundingDrivePromotion` (`autonomy/drives.py`) is **level 2 / active**, generating belief-tension
  questions (7,856 shadow selections, 29 answered). It **persists** correctly.
- The `academic_search` tool (Semantic Scholar) **works** (no auth/429 errors). Metric-deficit triggers
  are hardcoded **inward** (`source=codebase / local_only`, `metric_triggers.py:74-104`) — they never
  reach the journals. The only academic-bound lane (existential/curiosity) is ~100% **dedup-skipped**
  by `check_prior_knowledge` (already-studied topics).
- `SparkPromotion` is telemetry-only and now **persists** its earned reps (bug fixed: df7c8bb + f6a0ca0).

So the machinery exists and is earned; what's missing is the **P5 active loop** the code already names.

## The two P5 capabilities
1. **Autonomous belief-grounding.** For a belief-tension question the grounding drive raises, JARVIS
   auto-fires `academic_search`, fetches sources, validates the belief against external truth, and
   integrates the finding — instead of queuing it for the operator.
2. **Codebase→improvement escalation** (David's insight). When a *sustained* metric deficit's internal
   research is **exhausted** (≥N codebase attempts fail to close the gap), escalate to **academic**
   research for external improvement approaches. This wires the existing `mastery` drive
   (`tools=(codebase, introspection)`, `escalation: academic`, `drives.py`) to actually fire outward —
   the literal realization of "understands her code → researches how to improve it."

## The maturity ladder — how P5 is EARNED (shadow-first, zero-authority until the last step)
Each phase earns the next **only** by demonstrated accuracy, mirroring the spark/drive promotion gates.

| Phase | What runs | Authority | Earns the next by |
|---|---|---|---|
| **P5a · shadow-propose** | Generate the auto-fire **candidates** (questions she *would* fire) + log them beside the operator-pull queue | none | candidate set matches what the operator actually answers (precision/recall vs the answered set) |
| **P5b · shadow-execute** | Auto-fire `academic_search` **in shadow** — fetch sources, compute the validation she *would* conclude, but **do NOT mutate the belief**; log the shadow conclusion | none | shadow conclusions match the operator's answers (predictive accuracy ≥ threshold over ≥N reps) — *the critical earning gate: prove autonomous research is as good as operator-validated* |
| **P5c · advisory** | Auto-fired findings become **advisory** — flagged for operator review/approve before any belief moves | none (operator-gated) | operator approval rate ≥ threshold |
| **P5d · active** | **Rate-limited** autonomous belief-grounding — validated findings nudge belief confidence from external sources, no operator in the loop | earned, revocable | continuous accuracy monitor stays above threshold (auto-demote on drop) |

The promotion gate is a **NEW, SEPARATE counter — `AutonomousResearchPromotion`, starting at level 0 /
shadow.** It reuses the spark/drive promotion *machinery* (≥N outcomes, ≥H hours, ≥rate, persisted across
restarts) but **MUST NOT reuse the already-earned `GroundingDrivePromotion` gate.** That gate is already at
**active / level 2**, earned on 29 *operator-pull* answers — a *different* capability than "autonomous
research is as accurate as the operator." Gating P5d's autonomous belief-mutation under the existing
`is_active()` would be a **stealth authority grant** (an earn-don't-declare violation). The new gate earns
**specifically on autonomous-vs-operator accuracy** (P5b matched pairs), so the ladder can't be skipped and
survives reboots.

## Governance (the discipline)
- **Zero-authority until P5d.** Belief-mutation authority is the *last* step, earned by demonstrated accuracy.
- **Existing gates HONORED**, not bypassed: `check_prior_knowledge` dedup (never re-research settled
  topics), the **governor mode-gate** (research only in gestation/passive/dreaming/reflective/deep_learning),
  and hard **rate limits** on outward firing.
- **Operator kill switch** + an **auto-demotion** trigger (accuracy drop → back to advisory).
- **Honest telemetry**: `/v2/grounding` shows the ladder phase + the live shadow accuracy, PRE-MATURE-pinned,
  so the operator watches it *earn* (no green-tuning, §24).
- **No new perception/geometry**: consumes the existing grounding drive + `academic_search` + the belief
  field via public accessors. `hrr_side_effects` / authority surfaces unchanged until P5d.

## What this is NOT (anti-AI-theater)
- **Not a flipped gate.** Each phase is gated on *demonstrated* accuracy; auto-fire is unlocked only after
  P5b proves the shadow conclusions match the operator's (earn-don't-declare).
- **Real, measurable value**, not impressive machinery: the **orphan_rate** (~48% ungrounded beliefs) falls
  as she autonomously grounds them; `external_validation_rate` rises from **0**; sustained metric deficits
  close via external improvement research. Every claim is a number on the panel, reported from live data.
- **Rate-limited + kill-switched + auto-demoting** — bounded autonomy, not a runaway.

## Fit with the roadmap
Closes the **curiosity → exploration → self-improvement growth loop** the FINISH campaign points at. Builds
on the already-earned spark/grounding-drive, the working `academic_search`, and the drives map
(`grounding`/`mastery`/`curiosity` → `academic` / `external_ok`). It is the bridge from "JARVIS queues
questions for David" to "JARVIS researches journals on her own" — the thing that actually advances the soul.

## Validation outcome (2026-06-22) — verdict: BUILD-WITH-FIXES
A 4-lens validation (architecture/signal-trace, governance, earnability/anti-theater, regression) against the
**live** brain. The foundation is **real and confirmed live**: `GroundingDrivePromotion` is at `level=2/active`,
29 external-only operator answers, `selections_shadowed=7856`, persisted — and **P5a shadow-propose is already
wired** (`note_shadow_selection`). The surrounding gates (dedup, governor mode-gate, academic 10/hr rate-limit,
single-sited operator-only belief mutation) are real and honored. The P5 *active* loop is **entirely unbuilt**
(the design admits this), so the must-fixes below ARE the build plan, ordered:

1. **FIRST / BLOCKING — separate gate.** Do NOT reuse `GroundingDrivePromotion` for P5 authority (it's already
   active, earned on a *different* capability → stealth grant). Introduce `AutonomousResearchPromotion` at
   level 0/shadow, earned on autonomous-vs-operator accuracy. *(Pinned above.)*
2. **P5b shadow-execute (the critical earning link).** When a grounding `DriveAction` targets a **factual**-facet
   belief, auto-fire `academic_search` **in shadow** (governor + dedup HONORED), compute the would-be conclusion
   (confidence + polarity via `knowledge_integrator.integrate`'s factual/contextual classifier), **mutate nothing**,
   log to a durable shadow-outcome log separate from `grounding_outcomes.jsonl`.
3. **P5b→P5c earning score against INDEPENDENT operator answers only.** On `grounding_queue.answer()`, if the
   answered `belief_id` has a prior shadow conclusion, record match/mismatch into the NEW gate. No self-scoring.
4. **Throughput / coverage honesty (write it into the design).** Matched-pair accuracy accrues **slowly**
   (operator answers ~weekly) and **only for the `factual` facet** (identity/self/scene route to operator/Pi, never
   academic). Set N/threshold accordingly; the panel must not imply faster earning than the data allows.
5. **Per-gate kill-switch + force-demote** (the global `governor.set_disabled` is too coarse).
6. **P5d active = a NEW belief-mutation closure** (NOT the operator `_ground_belief`), gated on the NEW
   `is_active()`, rate-limited (≤3/hr), with **smaller** nudges than operator answers, auto-demote on accuracy drop.
7. **Mastery→academic escalation executor** in `metric_triggers.py`: track failed codebase attempts per metric
   cluster; only after ≥N failures with low win-rate, emit an academic-routed intent (still dedup/governor-gated).
8. **Telemetry**: `/v2/grounding` must distinguish *gate level* from *autonomous belief mutations actually applied*
   (currently 0) and show the live PRE-MATURE shadow accuracy.
9. **Findings safety**: only operator answers + P5d-active research may mutate `factual_knowledge`; P5a/b/c research
   feeds `contextual_insight` only, advisory-flagged.

**Bottom line (validation):** directionally sound, genuinely earnable (P5b is *not* chicken-and-egg — the
operator-answered set is real ground truth, no authority needed to score), real measurable value (orphan_rate
falls, `external_validation_rate` rises from 0), non-breaking to the live P4 loop — **provided the separate gate
(#1) lands first.** The recommended first build is **P5b in shadow** (the earning link), behind the new
zero-authority gate, with nothing mutated — exactly the project's earn-don't-declare discipline.
