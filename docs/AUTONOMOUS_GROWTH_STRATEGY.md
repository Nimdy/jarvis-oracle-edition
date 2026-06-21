# Autonomous Growth Strategy — crossing from shadow to earned

> Origin: an 8-scientist AGI/ASI panel (2026-06-20) + a decisive offline critic
> diagnostic run against the live 838-experience policy buffer. This is the
> durable record of *how* JARVIS's stuck learning loops can genuinely earn out
> of shadow without gaming gates (§24), confabulating, or recklessly granting
> live authority — and what the evidence says is actually possible.

## Core diagnosis (7 of 8 scientists, code-confirmed)
The policy loop is **not gate-stuck — it is *signal*-stuck on a non-causal estimand.**
- `evaluator.py:score_retrospective` sets `nn_reward = kernel_reward = actual_reward`
  (the NN's proposal is never executed → no counterfactual). "Wins" came from a
  hand-tuned `DEVIATION_BONUS` correlating divergence with reward that happened to
  move — **counterfeit credit, a confabulation inside the training loop.**
- **The imitation paradox:** the NN is trained to *clone* the kernel
  (`train_imitation`) and then asked to *beat* it. The cloning optimum **is** the
  kernel, so a well-trained model (val-loss 0.062) produces near-noop deviations
  (live data: kernel weights all 1.0 vs NN deltas ~0.01) → ties forever.
- **Part A** (shipped, `ee0caa7`) unflattened the *variance* of this metric (no more
  96% health-tie flooding) but it still contains **zero causal content**. So
  **Part B (interleaved execution) is NOT the right next step** — it would gamble
  live turns to manufacture a signal we must first compute offline.

## The reframe that makes it tractable
Every "gated, working-as-designed" loop must be classified — because the three
demand opposite responses, and **§24 (don't lower gates) only protects the first:**
- **Competence-limited** → soak / patience is correct.
- **Measurement-limited** → the policy loop. Signal is non-causal. *Fix the
  instrument, never the gate.*
- **Starvation-limited** → the spark/research loop. Its tension is a static
  *deficit* signal that zeroes the instant any external source touches a belief
  (`provenance_scorer.py:355`), fed by templated lookup-dict questions. *Starved,
  not gated — unparking P0 won't help.*

## The decisive test + its VERDICT (2026-06-20)
The panel's "one thing": build an offline critic `A(s,a)` on the existing lived
tuples and ask — **does it rank held-out real outcomes better than the kernel-mean
baseline?** PASS → a real counterfactual gradient with zero live risk. FAIL → the
reward/state carry no decision-quality info and *the reward itself* must be
redesigned (no gate/soak/interleave can fix that).

**Result (838 real interaction-outcome tuples, GBM + RandomForest, 5-fold × 10 seeds,
shuffle control):**
- real Spearman ≈ **+0.06 ± 0.02**, shuffle ≈ **0.00 ± 0.02**, separation ≈ **+0.06**.
- R² ≈ **0 / slightly negative** — the model does **not** beat predicting the mean.
- Action adds ~+0.03 Spearman over state-only — i.e. the policy's knobs (`budget_ms`,
  `thought_weights_delta`) have a *real but negligible* effect on the conversation
  outcome they're scored against.

**Verdict: effectively FAIL.** There is a tiny, statistically-real signal, but it is
**far too weak to build a trustworthy critic or drive promotion** (need ~0.3+; have
~0.06). The 348 trained versions (loss 0.062) were competence on a *synthetic
distribution that does not predict reality.* **The policy NN, as framed
(tune cognition knobs → improve conversation outcomes), cannot meaningfully grow on
this reward/state** — because the action barely moves the reward, and what does move
it (conversation content) isn't in the 20-dim state.

## Self-sensing verdict: PASS (2026-06-21) — the pivot is evidence-proven
Same discipline, applied to the senses. Collected a dense 11.2-min LIDAR window (400
samples @1.5s, the 12-sector nearest-range vector) and ran the predict-beyond-persistence
test (predict the next frame's delta from position+velocity; held-out, multi-seed, shuffle
control; static vs dynamic split at >2cm).
- 23% of transitions were genuinely dynamic (>2cm movement); 77% static (mm-noise).
- A learned model beats the persistence baseline ("nothing changed") by **+26% overall and
  +45.6% on the dynamic transitions** (Ridge; GBM +17%/+43%). The **shuffle control scores
  -6.3%** (no skill) — so the predictive skill is REAL, not overfitting. Naive momentum is
  WORSE than persistence (-136%): the dynamics are non-trivial but learnable.

**Verdict: PASS.** Unlike the policy loop (signal absent, Spearman ~0.06), the senses carry
a real, learnable, NON-OPERATOR signal beyond persistence. **Autonomous growth is physically
possible here.** Honest caveats: (1) the signal is SPARSE — it lives in the ~23% of moments
the world changes (this is exactly why the world-model is 98% persistence; the world IS mostly
static). (2) +26%/+46% is a FLOOR — simple model, coarse 1.5s cadence, one motion window.
(3) Growth here is paced by EVENT-bandwidth (how often the sensed world changes) — modest at a
quiet desk, richer in an active environment. The recommended build (below, #3) is therefore
CONFIRMED: the self-sensing learning-progress loop is the real autonomous-growth engine.

## Revised recommendations (given the FAIL)
1. **STOP investing in the policy NN as-is** — no critic, no Part B, no more training
   churn. Chasing a ~0.06 effect with live-turn risk is not worth it.
2. **If the policy loop is to live at all, redesign its reward/state** — score it on
   outcomes its knobs actually influence (e.g. latency/cost for `budget_ms`; reasoning
   quality for `thought_weights`), and add state features that capture what drives
   those — *before* any further mechanism work. Speculative; validate with the same
   critic test before trusting it.
3. **PIVOT to the loop where growth can actually happen: grounding / curiosity.** Here
   the action → outcome link is *directly causal* (research X / ask → belief grounded,
   measurable) and lowest blast-radius (read/compute, not live conversation):
   - Ship the **async operator Grounding Queue as a fan-out evidence event** (one
     answer anchors a belief + records a calibrator outcome + feeds the reasoning
     encoder + posts a real A/B win where it touched a live decision).
   - **Re-fuel the spark with learning-progress, not throughput** + a self-supervised
     *falsifiable-prediction* channel (predict the next Pi-sensed scene / own next
     behavior, check vs reality) so it earns on never-absent validators (time + senses)
     without waiting on the operator. Replace templated questions with ones about the
     real world.
4. **Durable evaluation state (cheap hygiene, helps the loops that *can* earn):**
   persist the per-decision A/B ledger + spark phase clock + specialist dwell, rehydrate
   on boot, with a restart-integrity test. (The A/B window is an in-RAM `deque` today —
   verified live to reset to 0 on every reboot. Lower priority for the policy loop now
   that it's FAIL, but correct for grounding/spark/dwell.)
5. **(Strategic) The growth allocator** — a metacognitive controller that classifies
   each loop (competence/measurement/starvation), routes success through one shared
   external-grounding currency, and spends one scarce live-authority budget, one canary
   at a time. Turns five stuck modules into one growing self.

## Integrity guardrails (non-negotiable)
Never change a threshold · earn the right to a synthetic signal by proving it predicts
reality (synthetic-firewalled) · no counterfactual credit (delete `DEVIATION_BONUS` +
the misattributed-kernel training target — itself an integrity fix) · any live trial
touches only reversible low-blast dims · **authority grows only on externally-attributed
causal wins, never a rising self-score** · **negative controls mandatory** (if grounding
throughput does NOT drop when the operator is away, it's fabricating ground-truth →
FREEZE) · durability before authority · one canary at a time on the single GPU. **Volume
is the enemy; watch compounding (chain length, repeated correct prediction), never count.**

## The honest caveat
This cannot manufacture ground-truth bandwidth that physically does not exist (one
household, a mostly-absent companion, a narrow sensor cone). The honest outcome may be
that **a few loops genuinely cross and compound while the rest stay dormant — possibly
forever — and that is success, not failure. Counting honest dormancy as failure is
itself the integrity violation.** The deepest risk is years of beautiful provenance-
tagged telemetry from loops that are causally incapable of turning, called "safety." The
critic test is the antidote — it forced a causal verdict here, and the verdict was
uncomfortable. Acting on it (not retreating to "gated by design") is the discipline.

## Open choices for the operator
- **Policy loop:** redesign its reward/state (speculative) vs. retire it as a growth
  loop and keep it shadow-only-forever (honest dormancy).
- **How hard to push:** honest dormancy as the system's most valuable alignment property
  (safety/epistemics view) vs. an organism denied its play period fails to develop
  (developmental/red-team view). The grounding-loop pivot is the lowest-regret way to
  push without the policy loop's risks.
