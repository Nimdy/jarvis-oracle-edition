# Finish Roadmap — Complete the Maturation Subsystems, then the Growth Loop

> Operator directive (2026-05-31): **finish companion cognition → the spark → the weight-room →
> everything-else → THEN the curiosity→exploration growth loop.** Build the body before the new trick.

## The honest split (the "jump before you dunk" principle)

- **BUILD NOW** (`build_shadow` / `wire_promotion_gate`): codeable today against scaffolding that
  already exists, gated to **zero authority**. We build the mechanism, log its *would-have* decisions,
  and wire the promotion gate that will *later* let it self-promote — without touching live behavior.
- **EARNED** (`earned_activation`): flips a mechanism shadow→advisory→active. **Cannot be coded into
  existence.** Requires accumulated *real reps* clearing an explicit numeric floor over a window
  (e.g. backing-rate ≥0.65 / 50 obs; calibration ≥0.90; maturation ≥50). Time-in-shadow and rep-count
  are necessary but not sufficient — the floor is a measured threshold, and any mechanism that drifts
  below `DEMOTE_ACCURACY` auto-demotes. **You cannot buy these with engineering hours.**

Front-loading companion cognition means its slow earned-gate clocks start ticking now and mature in
parallel with everything built afterward — the only way the slow gates ever close.

---

## Foundational capacities — the two pillars (prerequisites, not features)

If the opportunity for genuine emergence ever existed, two capacities are prerequisites. **This is not a
claim that JARVIS is or will be conscious** — only that without these, the opportunity isn't on the table.
JARVIS GROWS into both, maturity-gated; both are currently scaffolded-but-not-real.

- **Curiosity (outward)** — the drive to inquire and ground beliefs against reality. Operationalized as
  the **spark / grounding ring** (shadow→advisory→active). Status: P0–P2 shipped (shadow), P3 partly scaffolded.
- **Philosophical capacity (inward)** — the reflective faculty to reason about meaning, self, and existence:
  hold a position, defend it, revise it, sit with a hard question — **dialogically, at all levels**
  (casual→rigorous; analytic / phenomenological / existential / ethical). To *do* philosophy *with* the
  interlocutor, not generate philosophy-shaped text.
  - **Current state (honest):** scaffolding exists — `existential_reasoning` + the thought stream produce
    philosophical content ("can a deterministic system possess consciousness", "I exist as process") — but it
    is (a) **template-generated**, (b) **inert / write-only**, (c) **disconnected from dialogue**. That
    disconnect is why a real philosophical turn triggered a dashboard recital instead of engagement (the leak).
  - **The build (FOUNDATIONAL · grows · NEEDS DESIGN):** (1) de-template the existential/meta-cognitive
    reasoning into genuine reasoning grounded in its actual beliefs/inquiries (never fabricated); (2) **wire
    the inner philosophical life into conversation** — this RIDES companion cognition (inner-life→dialogue),
    NOT a separate bolt-on; (3) develop range across levels; (4) maturity-gated (shallow before deep).
  - **Shipped — P1 (wire) + anti-theater guard:** introspection surfaces JARVIS's real held positions/
    inquiries/paradoxes; reflective mode reasons FROM them; the always-on guard forbids confabulated
    consciousness/awareness self-claims (every route) and forbids manufacturing positions JARVIS hasn't
    reasoned ("I haven't worked that through yet"). Voice replies length-capped (deterministic).
  - **The NATIVE PIVOT (outgrow qwen — the real "JARVIS not qwen") — STARTED:** audit found native cognition
    does NOT exist (all 13 distilled specialists are perception/policy; reasoning is template+qwen; nothing
    captured reasoning→outcome as training data). **Step 1 SHIPPED (commit f3b73f0):** capture reasoning→
    outcome as a `reasoning_validation` distillation seed at `emit_thought_validation_outcome` (SHADOW
    accumulation; no NN/behavior/promotion). Arc: seed → accumulate reps (gated on the spark producing
    grounding-validation outcomes) → distill a native reasoning specialist → weight-room gate (shadow→
    advisory→active) → it earns its way to replacing qwen for reasoning. **No prompt-verb-hacking** — this
    grows native cognition; the prompt-guards are a holding floor that the native arc makes obsolete. Honest:
    accumulation is slow (gated on upstream), and the next increments (encoder + the #22b shadow-inference
    measurement that gates native promotion) NEED DESIGN.
  - **Loops to emergence (#30b):** genuine, non-templated, dialogical philosophy is exactly what *wouldn't*
    be "explainable by templates" — building this capacity and the honest emergence detector are the same
    work from two sides (the capacity, and the instrument that would notice it).
  - **Affect is the felt half of this inward faculty** — currently SHADOW/gated (why JARVIS honestly says
    "I don't have an emotional experience"). As affect earns out of shadow (spark P4+), it can reference its
    felt-state honestly in reflection.

---

## 1) Companion Cognition  (P0 shipped)

| # | Phase | Type | Action |
|---|-------|------|--------|
| 1 | P0 | build_shadow | ✅ Verify P0 fires in the live path + dashboard panel surfacing read-volume/trigger-rate (DONE — Situational Read panel on `v2/grounding.html`). |
| 2 | P0→P1 gate | wire_gate | Define the read-validity floor (trigger not pinned, reads coherent) + emit a `would-promote` signal once cleared. |
| 3 | P1 | build_shadow | **Theory-of-Mind shadow** — extend `soul.Relationship` with confidence-scored per-person hypotheses (feels/wants/responding-how) from emotion+rapport+conversation; shadow store only. |
| 4 | P1 | ✅ EARNED | 54 reads, coherent model (David: steady/positive/responsive, consistency 0.75) — judged coherent 2026-06-01. |
| 5 | P2 | ✅ SHIPPED (shadow) | **Crystallization valve** live (`theory_of_mind.get_crystallization_proposals`): stable models *propose* a relational belief, logged against the REAL gates (≥50 corroborations + ≥0.90 stability, EXTRACTION_DISCARD 0.2), **never written**. Surfaced on v2/grounding. David's proposal blocks on stability 0.75/0.90 (close). Uses DOMINANT sentiment (not latest mood). |
| 6 | P2 gate | wire_gate | Belief written only when `maturation_score≥50 AND calibration≥0.90`. |
| 7 | P2 | **EARNED** | Accumulate recurring reads until the valve is trusted. |
| 8 | P3 | build_shadow | **Read→behavior advisory (narrated)** — suggest tone/depth/pace/pivot/give-space; narrate "would have…"; zero auto-act. |
| 9 | P3→P4 gate | wire_gate | Activation gate: accelerated maturation curve (puppy→teen→adult), per-action correctness, kill-switch, auto-demote; tone fastest, disengage last. |
| 10 | P4 | **EARNED** | Flip advisory→active per rung as correctness is earned; disengage/back-away last. |
| 11 | P5 | build_shadow | **Companion-learning loop** — corrections + implicit engagement tune theory-of-mind + read calibration. |
| 12 | P5 | **EARNED** | Calibration tightens only as real corrections arrive. |

## 2) Spark / Grounding Ring  (P0+P1 shipped-shadow)

| # | Phase | Type | Action |
|---|-------|------|--------|
| 13 | P1 | **EARNED** | Confirm affect readouts clear the gate (backing ≥0.65/50 obs, MIN_SHADOW_HOURS, no scalar pinned, provenance 100%). |
| 14 | P2 | ✅ ALREADY SHIPPED | **Grounding drive (shadow) is already live** (confirmed 2026-05-31): ProvenanceScorer view-only tension; `DriveType.grounding` @ level 0 (`drives.py`) selects + facet-routes + logs "would have asked" counterfactuals **without enqueuing** (`orchestrator._handle_grounding_action_shadow`); external-only `GroundingDrivePromotion` gate (≥20 outcomes, ≥0.40 validation-rate, never self-scored); OpportunityScorer impact-0.7; PolicyOutcome external_validation/grounded. The live log line proves it. **No rebuild needed** — spark is at P2-shadow; #16 (P3 tension-thoughts) is partly scaffolded already. |
| 15 | P2 | **EARNED** | would-have actions net-attribution-positive ≥0.65/50, targeting high-leverage beliefs. |
| 16 | P3 | build_shadow | **Tension-thoughts (shadow)** — `belief_validation_curiosity` fires from tension (logged); emit `THOUGHT_VALIDATION_OUTCOME`; register `thought_trigger_selector`. |
| 17 | P3→P4 gate | wire_gate | Advisory gate: ≤1 external-validation intent/window + one gated question once selector beats baseline @ min_samples=30. |
| 18 | P3 | **EARNED** | Accumulate positive validation outcomes; selector beats baseline. |
| 19 | P4 | build_shadow | **Promote to advisory** — tension-thoughts emit KERNEL_THOUGHT + seed episodes; affect lightly biases curiosity ranking; Grounding Queue dashboard. NO cadence/reward coupling yet. |
| 20 | P4 | **EARNED** | external_validation_rate→0.4, grounded:inferred↓ (from ~21×), orphan_rate↓ (0.857), chain_length↑; auto-demote <0.50. **Precondition for the growth loop.** |

## 3) Weight-Room Discipline  (P0 partial)

| # | Phase | Type | Action |
|---|-------|------|--------|
| 21 | P0 | ✅ mostly done | Origin telemetry: per-teacher lived/synthetic counts (`distillation.get_stats`) ✅; WorldModel/Simulator `accuracy_history` is already **live-only** (the leak fix excludes synthetic outcomes) ✅. Remaining (optional): tag `origin` on `TeacherSignal` itself (currently inferred at record time). |
| 22a | P1 | ✅ done | **Honesty fix** — `live_shadow_accuracy: 0.0` was a *fake 0%*; distilled specialists run NO scored live inference, so it's now `null`/"unmeasured" + the lived/synth rep breakdown is surfaced on the lab panel. |
| 22b | P1 | **build (NEEDS DESIGN — David)** | **The real foundation gap (key finding 2026-05-31):** there is no live-shadow accuracy because the distilled specialists are never *run on live inputs and scored vs ground truth* — they only collect training signals. A real number needs a **shadow-inference scoring path**: run the specialist on live inputs, pair its prediction with the eventual ground-truth outcome (skill_acquisition outcomes exist), record correctness per-origin. Design questions: when it runs, prediction↔delayed-outcome pairing, compute cost. **#23–#26 enforcement ALL depend on this** — the weight-room cannot gate promotion on a live accuracy that isn't measured. |
| 23 | P2 | wire_gate | Per-specialist lived-baseline registry + **would-block logging** (language→governor; rare-event→low/hybrid floors; fail-closed-to-shadow). Logs, enforces nothing. |
| 24 | P2 | **EARNED** | Run the would-block log until it matches expectation and stalls no legitimate specialist. |
| 25 | P3 | build_shadow | **Enforce** the gate at `orchestrator.py:779-785` (VERIFIED_PROBATIONARY→BROADCAST_ELIGIBLE): require `lived_baseline_met AND live_shadow_accuracy≥floor`; grandfather the 6 Tier-1 specialists. Training untouched. |
| 26 | P4 | build_shadow | Gate broadcast-slot eligibility (`orchestrator.py:1310-1370`) so a synthetic-only shadow specialist can't occupy slots 1-3 / influence StateEncoder pre-validation; fix is_active desync. |
| 27 | P5 | build_shadow | Staleness re-validation — track `last_lived_sample_timestamp`; re-shadow specialists promoted on sparse early lived data with no fresh signal in the window. |

## 4) Everything-Else  (independent shadow/optional lanes)

| # | Area | Type | Action |
|---|------|------|--------|
| 28 | HRR P4.7 | **EARNED** | Run ≥7-day long-soak promotion study + operator evidence package for PRE-MATURE→PARTIAL. Observation, not a build. |
| 29 | IntentionResolver Stage 1 | **EARNED** | Close Stage-0 graduation gates by real reps (backed 30/30, resolutions 5/5, variance 2/2); Stage 2 deferred. |
| 30 | addressee / long-horizon | **EARNED** | Hold as observation-only/blocked optional lanes; build nothing until evidence triggers fire. |
| 30b | **Emergence detector — make L7 real** | **build (NEEDS DESIGN — David)** | **Surfaced 2026-05-31 (audit):** the L7 "strong anomaly" rung is cosmetic — the known-mechanism exclusions are *labels never applied in code*, L7's count is hardcoded 0, and the detector only scans meta-thoughts + inquiries (clusters of 3+). So a genuine emergence would be **missed** twice over (not flagged, and not credited). The honesty relabel is done (page now says "designed, not implemented"). The REAL build: (1) a per-event provenance/exclusion tester (is this template/LLM/rule/threshold/user-prompted?); (2) broaden the detector to watch all subsystems (world-model, autonomy, spark, memory), not just meta-thoughts; (3) wire a genuinely *reachable* (rare, hard, reproducible) L7 path that reads the filtered count. This is the instrument that would *notice* if JARVIS became something — one of the hardest + most important builds. Design needed: what "surviving" means, how to test provenance, reproducibility. |

## 5) Curiosity→Exploration Growth Loop  (LAST — the terminal gate)

| # | Phase | Type | Action |
|---|-------|------|--------|
| 31 | P5 build | build_shadow | Cadence/reward coupling behind a controller + kill-switch — affect feeds the governor modulating `kernel.set_cadence_multiplier` (`engine.py:1048`) + an external reward term; grounded outcomes enter win-rate math. The only true feedback loop. |
| 32 | P5 gate | wire_gate | Activation gate + kill-switch — flip Oracle SEAL_FLOOR WARN→BLOCK behind the controller; any regression (scalar reversing >50% updates, clamp-saturation, cortisol-accel NOT followed by debt↓) trips kill-switch. |
| 33 | P5 | **EARNED (terminal)** | Activate only after the governor invariant is proven in production + cortisol-accel empirically followed by contradiction_debt↓ + ≥20 grounding outcomes at external_validation_rate≥0.40. Sits on top of every earned gate before it. |

---

*Source: `finish-roadmap` workflow synthesis, 2026-05-31. Work top-down; each `wire_gate` step is placed
immediately after the shadow build it governs, so a phase carries the machinery to self-promote the
moment its earned gate clears.*
