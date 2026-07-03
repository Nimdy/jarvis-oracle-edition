# Directed-learning A/B — design for review (2026-07-03)

> **Status: DESIGN ONLY — not built.** This is the causal follow-on to the earned (out-of-sample
> confirmed) error-seeking curiosity signal. It needs David's sign-off before any code, and it will
> itself be pre-registered (bar locked before results) and earn-gated. See
> `docs/CURIOSITY_CRITIC_PREREG.md`, `cognition/curiosity_critic.py`, `docs/AUTONOMOUS_DRIVE_MAP.md`.

## The question this answers (and why it's the right one)
We confirmed **predictively** that the active sector where the world-model is *worst* (lowest skill)
reliably improves next — error-seeking curiosity. We did **not** show that *acting on* that signal
*causes* the improvement; it's consistent with mean-reversion, and the passive lidar loop has **no
attention actuator** (looking "harder" at a sector changes no data). So a live "advisory to attend"
would be theater.

But the passive loop *does* have one real actuator: **what the predictor chooses to learn from.**
Directing the world-model's *own learning* toward its weakest sectors ("hard-example mining" /
active learning) is something the system genuinely *does* that changes the model — and its effect is
measurable against controls. **This A/B is a real causal test**, and it resolves the predictive-vs-
causal question: if directing learning at the weak sectors *causally* beats not doing so, it is by
definition **not** mean-reversion.

## Design — 3 parallel shadow predictors, one data stream
All three see the *same* incoming lidar frames, share the *same* rolling buffer / features / ridge
λ / motion gate. They differ **only** in the per-sector loss weighting at refit. Each maintains its
**own** weight matrix and its own next-frame skill EMA (predict *before* the frame enters training →
out-of-sample by construction, exactly like the existing skill metric).

- **Arm U — uniform (control):** the current refit, every sector weighted equally. This *is* today's
  live predictor.
- **Arm E — error-seeking (treatment):** weighted ridge; the loss on each sector `j` is scaled by
  `w_j = 1 + λ · (1 − skill_norm_j)`, where `skill_norm_j` is the sector's skill min-max-normalized
  into [0,1] over active sectors (worst sector → weight `1+λ`, best → weight `1`). Weights clipped to
  `[1, 1+λ]`; inactive/NaN sectors get weight 1. `λ` **pre-registered** (proposal: λ=3).
- **Arm R — random (negative control):** the *same* weighting magnitude applied to a **random**
  subset of active sectors (re-drawn each refit), not the weak ones. Isolates "target the weak
  sectors *specifically*" from "reweight the loss *at all*."

Weighted multi-output ridge stays closed-form (just a per-output loss weight in the normal
equations), so this is ~3× a cheap refit — negligible cost.

## Metric & pre-registered success bar (to be locked BEFORE reading results)
- **Primary metric:** each arm's **overall dynamic skill** (`skill_vs_persistence_dynamic`) over a
  rolling window — *overall*, not just the weak sectors, so we measure the true **net** effect.
- **Bar (proposal, to finalize at pre-registration):** on a **held-out** window, paired per-frame,
  Arm E's mean dynamic skill exceeds **both** Arm U **and** Arm R by a margin `δ` (proposal
  δ ≥ 0.02 absolute) with a bootstrap 95% CI on `(E − U)` and `(E − R)` excluding 0, over
  ≥ N observations (proposal N ≥ 5000 dynamic frames). Run once, no retry-until-green.

## Honest failure modes designed in (any of these → FAIL, reported plainly)
1. **Robbing Peter to pay Paul.** Upweighting weak sectors may improve them while *hurting* the
   well-predicted ones → net skill flat or down. The overall-skill metric catches this; a per-sector
   improvement that tanks the whole is a FAIL, not a win.
2. **"Reweighting itself helps."** If Arm R also beats U, the specific "target the weak sectors"
   claim fails — only E‑beats‑both counts.
3. **Instability.** Extreme/NaN per-sector skill could blow up weights; the clip + kept ridge
   regularization guard this. If an arm destabilizes, that arm FAILs.
4. **Too small to matter.** The predictive edge was modest (+0.014). The causal edge may be smaller
   still, or zero. A null result is a real, publishable-to-David outcome — the signal then stays a
   passive uncertainty readout.

## Decision rule
- **E beats U and R, sustained, out-of-sample confirmed** → directing learning at the weak sectors
  *causally* improves the world-model. This is the first genuinely **self-directed learning** loop
  (the model chooses where to learn and measurably gets better). It earns eligibility for a **gated
  flip**: make Arm E's weighting the live predictor's refit — **David approves, never auto**.
- **Otherwise** → the error-seeking signal stays a passive "where I'm weakest" readout (still useful
  to *other* systems, e.g. camera-gaze priority if/when an actuator exists). No directed-learning
  lever. Clean stop.

## Build outline (only if approved)
1. `cognition/directed_learning_ab.py` — holds the 3 weight matrices + 3 skill EMAs; refit in
   parallel each `REFIT_EVERY` with the 3 weighting schemes; **authority=none** (the live predictor
   stays Arm U). Read-only `ab` block on `/api/self-sensing`; a bounded recorder for offline stats
   (same one-file, restart-surviving pattern as `curiosity_critic`).
2. `docs/DIRECTED_LEARNING_PREREG.md` — lock λ, δ, N, and the held-out cutoff *before* results.
3. Soak → offline compare (bootstrap CI + the Arm R negative control) → out-of-sample confirm →
   bring the verdict to David. Gated flip is a separate, explicit decision.

## Guardrails (unchanged)
Shadow-first; the live predictor is untouched until a flip is explicitly approved. No self-sensing
authority flip, no §24 tuning, no memory/belief write, no grounding-ladder touch, no autonomy-level
change. The A/B *earns* the right to propose a flip; it never takes one.

## Open questions for David
- OK to run this as the causal test, or point the earned "where I'm weakest" signal at a different
  actuator (e.g. camera-gaze / depth-compute priority) instead?
- λ = 3 and the δ ≥ 0.02 / N ≥ 5000 bar reasonable, or tune the pre-registration first?
