# Long-Horizon Attention — Pinned Use Case (P3.12)

**Date:** 2026-04-24
**Lane:** TODO_V2 §3 P3.12 (long-horizon attention feasibility)
**Status:** Use-case pinning only — **no code**.
**Rationale:** Per operator instruction and TODO_V2 §4 sequencing, P3.12
must have a pinned use case *before* any implementation. This document
is that artifact. It is the single input to the go/no-go decision.

---

## 1. Problem statement (one sentence)

> Jarvis must be able to hold a single, operator-attested task across
> multi-day sessions — including reboots, sleep/wake, and intervening
> unrelated conversation — and surface it *on its own initiative* at the
> correct moment without the operator re-reminding it.

This is a **capability**, not a feature. It is observable to the
operator as a single behaviour: "yesterday I asked it about X; today it
brought X up at the right moment, correctly, without being prompted."

## 2. The one pinned use case

**Name:** `medication_refill_followup`
**Scope:** single operator (owner), single task class, single surface.
**Horizon:** 1–30 days (a medication refill window).
**Surface:** proactive speech, through the existing `ProactiveEngine`.

### 2.1 Narrative

Day 0, evening:

> Operator: "My lisinopril runs out around the 12th of next month."

Day 1–(k-2): no mention. Normal conversations continue. The system
sleeps, wakes, restarts, talks about everything else.

Day (k-1) — one day before the refill window closes:

> Jarvis (unprompted): "Your lisinopril should run out in about a day.
> Want me to check the refill?"

If the operator says "yes", the task is complete. If the operator says
"no, I already did it", the task is closed with a learned "user handled
it themselves" signal. Either way, the task does not re-fire.

### 2.2 Why this use case (not a generic one)

- **Specific enough to measure.** There is one target date, one target
  surface, one correct behaviour. No "remind me about stuff" catch-alls.
- **Honest memory pattern.** Medication refills are real long-horizon
  tasks the owner does experience. The system is not inventing a use
  case to justify the machinery.
- **Bounded blast radius.** A false positive ("I reminded you when I
  shouldn't have") is mildly annoying, not dangerous. A false negative
  ("I forgot") is the exact failure mode the capability must guard
  against — so metrics align with the capability itself.
- **Cross-restart and cross-day.** Forces the capability to survive
  sleep, reboots, supervisor restarts, and language-kernel rollbacks.
  This is the hard problem; anything softer is not really "long-horizon".
- **Operator-attested.** The horizon is set by the operator in natural
  language, not by a special command. This is the correct shape for the
  production capability.

## 3. Success metric

One composite metric, stamped at the lane's post-implementation review.

**`long_horizon_recall_rate`** = (correct proactive surfaces) / (tasks that should have fired).

Each denominator entry is an operator-attested "this should have fired
at roughly time T". The numerator counts surfaces that fire within a
soft window of T (default ±12h) and pass truthfulness (did not hallucinate
the context).

### 3.1 Target

- **Go:** `long_horizon_recall_rate ≥ 0.80` across ≥ 5 real tasks.
- **No-go:** below 0.80, or any **truthfulness** violation (hallucinated
  context in the surface).

Accuracy-as-proxy is forbidden (same guardrail as Tier-2 specialists).
A 0.99 recall rate with a single fabricated detail is a **no-go**.

### 3.2 False-positive cost

Tracked but not in the go/no-go gate:

**`long_horizon_noise_rate`** = (surfaces that fired without an
operator-attested task) / (total surfaces).

Soft target: ≤ 0.10. Noise above that requires a separate fix (the
operator does not want the system developing a "helpful nag" failure
mode).

## 4. Degradation guard — the single deliberate failure mode

The pinned degradation guard is the one rule the capability must enforce
even when everything else goes wrong:

> **Never fire a proactive reminder unless the originating operator
> statement is provably in memory and attested as the source.**

In code-visible terms:

- The proactive surface must carry an attributed origin
  (`memory_id` + quoted operator clause).
- If the origin is missing, the surface is suppressed.
- If the operator says *"I never said that"*, the system must:
  1. Suppress the follow-up permanently for that task.
  2. Emit a `user_correction` edge into the belief graph (P3.2 writer
     already in place as of 2026-04-24).
  3. Decrement the long-horizon attention's trust score.

This guard is what makes long-horizon attention safe to ship. Without
it, the capability collapses into the "hallucinating assistant" failure
mode we have been avoiding since the start of the project.

## 5. Non-goals (what this lane is NOT)

- **Not** a generic reminder system. No UI for setting reminders.
- **Not** a calendar surface. If the operator wants a calendar, they
  have a calendar.
- **Not** a to-do list. Tasks are extracted from natural conversation,
  not entered as discrete items.
- **Not** a multi-party capability. Single operator only for the first
  lane. Multi-user is a separate future problem.
- **Not** a learner without the operator. The system does not learn
  long-horizon patterns from its own inferences — only from
  operator-attested tasks.

## 6. Preconditions before any code lands

These are the non-negotiable preconditions. Implementation may only
start when all are true:

1. **P3.2 `user_correction` writer live.** (Shipped 2026-04-24.)
2. **Phase 6.5 three-axis invariant** still clean on the live brain.
   Long-horizon surfaces must respect the autonomy level.
3. **Attribution ledger bounded.** Long-horizon surfaces must be
   queryable against the attribution ledger.
4. **Belief-graph `user_correction` + `depends_on` writers live.**
   (Both shipped 2026-04-24 — P3.2 and P3.3.)
5. **Language kernel artifact registered.** (Operator action, staged
   2026-04-24 — P3.4 seed script shipped.) Without a pinned language
   kernel identity, a rollback would drift the voice of the proactive
   surface.
6. **ProactiveEngine has a dedicated long-horizon channel.** Must not
   piggy-back on the existing short-horizon proactive path to avoid
   polluting its noise budget.

## 7. Minimum viable interface (reference only, not binding)

This is intentionally sketchy; the implementation lane will sharpen it.

- **Capture:** a detector in the correction / intent layer recognises
  "operator-attested future task" utterances and writes a
  `long_horizon_task` record with `{memory_id, operator_clause, target_ts, soft_window}`.
- **Hold:** the record persists through restarts via the existing
  `~/.jarvis` path, same as other memory surfaces.
- **Surface:** a ticker (one tick per minute is enough) checks records
  against the current time, applies the soft window, and — if within
  window, not already fired, and origin attribution is intact — emits a
  proactive speech event.
- **Close:** the operator's response closes the record ("yes", "no —
  done", "no — wrong"). A "wrong" response triggers the P3.2 writer.

## 8. Open questions the implementation lane must answer

- How does the capture detector distinguish "I need to refill my
  lisinopril around the 12th" from "wouldn't it be funny if people had
  to refill their souls monthly"? The current correction / intent
  detectors don't cover this.
- How is the soft window (`±12h`) determined per-task class? For a
  medication refill, 12h may be generous; for a birthday, it should be
  narrower.
- Does the surface respect the current autonomy level? (Yes, it must.)
  At L1 it must be suppressed; at L2 it may speak; at L3 it may act
  (e.g., draft a pharmacy message).
- What is the right retirement policy? After the task fires and is
  acknowledged, is the record deleted, archived, or kept indefinitely
  for pattern learning?

These are the four questions the next agent working on P3.12 must
answer before cutting code. This document pins the *what*; the
implementation lane answers the *how*.

## 9. What this document replaces

Before this pass, P3.12 had only the TODO_V2 line "needs a pinned use
case before any code". That line remains correct; this document is its
evidence artifact. The status board for P3.12 can now legitimately
read:

> `use_case_pinned: medication_refill_followup` (see
> `docs/LONG_HORIZON_ATTENTION_USE_CASE.md`), `status: ready_to_scope`.

It is **not** `ready_to_implement` — the six preconditions in §6 must
all be verified on the live brain first, and the four open questions in
§8 must be answered by the implementation lane's design doc before any
PR lands.
