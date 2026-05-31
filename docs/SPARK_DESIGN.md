# The Grounding Ring — Curiosity Anchored to External Truth

**Build plan for SyntheticSoul §5 (Cognitive Dynamics) + §6.5 (hormonal/circadian modulation).**
**Status:** design · **Date:** 2026-05-30 · **Author of theory:** David Eierdam · **Lens:** maturity-respecting, honest-by-construction.

> This document is the detailed, buildable plan for the "spark." It is grounded in the
> *actual* code (every station cites a verified `file:line`) and it is, simultaneously,
> the fix for the keystone problem named in the 2026-05-30 audit. The spark and the
> keystone fix are **literally the same code**.

---

## 1. Thesis — it is a reorientation, not new machinery

JARVIS already contains every moving part of an outward grounding loop:
- `DriveManager` scores motives (`brain/autonomy/drives.py`)
- `OpportunityScorer` ranks intents (`brain/autonomy/opportunity_scorer.py`)
- `DeltaTracker` attributes **counterfactual** credit — *the sole existing external validator of autonomy* (`brain/autonomy/delta_tracker.py`)
- `PolicyMemory` records what worked (`brain/autonomy/policy_memory.py`)
- `BeliefStore`/`BeliefGraph` compute provenance / orphans / effective-confidence (`brain/epistemic/belief_graph/`)
- `ProactiveGovernor` + `CuriosityQuestionBuffer` reach the operator (`brain/personality/curiosity_questions.py`)
- the bridge already subscribes to `WORLD_MODEL_PREDICTION_VALIDATED` (`events.py:384`, `bridge.py:670`)

**The keystone disease:** all of it points **inward**. The "win" signal is the system's own 8 internal health metrics and the self-graded Oracle benchmark — producing **~6507 metric-created goals** whose top recurrent motive is the tautological *"fix my own shadow_default_win_rate,"* with `orphan_rate 0.857`, `avg_chain_length 1.0`, and a `grounded:inferred` ratio of **~20×** across its active belief graph (~85% external web-citations, ~14% its own inference, **under 1% grounded in its own senses or the operator** — only 4 of 583 active beliefs; ~3.5× on the broader memory store, the figure originally cited). A mirror: **curiosity that rewards internal consistency cannot discover it is wrong about the world.**

**The build:** reorient the existing machinery from self-metric to **external ground truth**, plus the one feedback loop the vision requires (affect-as-readout modulating cadence/salience, under a homeostatic governor). A grounding drive fires on belief-map tension, formulates a question whose answer **must come from outside** (operator, Pi senses, or source-cited research), and is rewarded **only when an external validator touches the belief — never by its own metrics.** When it lands, the top recurrent goal stops being *"fix my win-rate"* and becomes *"ground belief X about the world."* That is the break of the self-referential loop and the ignition of genuine curiosity in one move.

It is honest by construction: affect scalars are deterministic functions of counted real signals (never *felt*); reward for grounding is set by the external validator (never self-scored); **being corrected counts as success** (a belief moving from inferred to externally-anchored *is* the goal); and every mechanism ships shadow-first with zero authority, earning promotion the exact way World Model and Mental Simulator already do (`cognition/promotion.py`: shadow→advisory→active, accuracy-gated, auto-demoting). **The spark is treated as aspiration; success is measured behavior** — `grounded:inferred` falling from ~20× (belief graph; ~3.5× on the memory-store continuity view), `orphan_rate` falling from 0.857, `avg_chain_length` rising above 1.0, `external_validation_rate` rising from ~0 toward >0.4 — not the appearance of a spark.

---

## 2. The Grounding Ring (8 stations)

A single outward-pointing ring; each station maps to a verified subsystem, with the minimum new glue named.

1. **Belief tension (origin).** `grounding_tension` = weighted max of {`QuarantinePressure.composite`, `orphan_rate` (`integrity.py:28`, live 0.857), normalized `inference_gap` = model_inference − (user_claim+external_source+observed)}. Read-only via a new **`ProvenanceScorer`** (`brain/epistemic/provenance_scorer.py`, view-only — *no* mutation of the frozen `BeliefRecord`).
2. **Seeded thought.** `grounding_tension` fires `belief_validation_curiosity` (`meta_cognitive_thoughts.py`), replacing the decorative `observation_count % 5` trigger (`line 95`) with a research prompt naming the belief, its provenance, confidence, and zero supporting edges; carries `belief_id` + `validation_target`. Its template matches the `curiosity_detector` learning-phrase regex so it drives the **existing** `META_THOUGHT_GENERATED → AutonomyEventBridge → CuriosityDetector → ResearchIntent` chain.
3. **Grounding drive.** New `DriveType grounding` competes in `DriveManager.select_action()`; `urgency = grounding_tension` (floor 0.10), `action=research`, `scope=external_ok`, tool routed by facet. **Not** dampened by active user goals (grounding a wrong belief *is* user-relevant).
4. **External input.** Via `ResearchIntent → OpportunityScorer` (impact 0.7, **not** existential — one edit at `opportunity_scorer.py:324`) `→ ResearchGovernor → QueryInterface → KnowledgeIntegrator` for web/codebase/memory; **or** `CuriosityQuestionBuffer → ProactiveGovernor → TTS` for operator validation.
5. **Observer validation (§5.3) — the close, never self-scored.** Operator answer → `belief.add_evidence(source=user_answer)`; world-model hit → `WORLD_MODEL_PREDICTION_VALIDATED` → new `BELIEF_EXTERNALLY_CONFIRMED`; source-cited finding → `KnowledgeIntegrator` provenance. Confirmed boosts salience/confidence; **refuted tanks + quarantines and still counts as `grounded=True`.**
6. **Affect / reward (derived, not felt).** dopamine from confirmations + `net_attribution`; cortisol from pressure + debt; serotonin from coherence + low friction. Reward reorientation: extend `_compute_health_reward` (`engine.py:956`) with an **external** term.
7. **Cadence / salience / drive modulation** — under the homeostatic governor (the only true feedback loop; ships **last**).
8. **Biases next thought.** Modulated drive urgencies + regulated affect + updated belief confidences + `PolicyMemory.external_validation_rate()` feed the next tick's `grounding_tension`. Rate falling → grounding quiets (homeostasis); new inferred beliefs → grounding loudens (the spark).

**Audit chain (breaks the tautology):** `grounding_tension(source) → seeded thought(belief_id, validation_target) → DriveAction → channel → external evidence → Observer validation → PolicyOutcome.external_validation → Oracle world_grounding_coherence`. **No step is self-scored — the validator at station 5 is never the system itself.**

---

## 3. The five components

| # | Component | Key files | Gate |
|---|---|---|---|
| 1 | **Grounding / Validation Drive** (keystone fix) — new `DriveType grounding`, fires on belief tension, question answerable only externally | `autonomy/drives.py`, `orchestrator.py`, `policy_memory.py`, `opportunity_scorer.py`, `belief_graph/bridge.py`, `events.py` | Shadow-first; `grounded=True` outcomes excluded from win-rate math until ≥20 outcomes with `external_validation_rate ≥0.40` AND `orphan_rate` trending down |
| 2 | **Tension-seeded thoughts** + `ProvenanceScorer` + Phase-2 selector teacher signal | `epistemic/provenance_scorer.py` (new), `meta_cognitive_thoughts.py`, `research_intent.py`, `autonomy/event_bridge.py`, `hemisphere/types.py` | Deterministic trigger fires from day one (structurally backed by a real belief); the `thought_trigger_selector` NN is pure shadow until it beats the deterministic baseline (same gate as existing distillation specialists) |
| 3 | **Affect State + Homeostatic Governor → cadence/salience** (§6.5) | `consciousness/affect_state.py` (new), `affect_regulation.py` (new), `engine.py`, `consciousness_system.py`, `drives.py`, `attribution_ledger.py`, `skills/capability_gate.py` | Ships zero-authority; shadow→advisory→operational only on ≥7-day measured behavior: anti-oscillation proven, no clamp-saturation, and cortisol-acceleration **followed by** a `contradiction_debt` decrease |
| 4 | **The closed loop (spark) + ground-truth strategy** | `jarvis_eval/oracle_benchmark.py`, `consciousness/observer.py`, `curiosity_questions.py`, `delta_tracker.py` | Entirely shadow-first; spark is aspiration, never a deliverable; Oracle `world_grounding_coherence` SEAL_FLOOR staged warn-before-block |
| 5 | **Honesty enforcement + maturity gates + rollout** (the governance spine) | `skills/capability_gate.py`, `cognition/promotion.py`, `consciousness/affect_regulation.py`, `delta_tracker.py`, `jarvis_eval/process_contracts.py`, `events.py` | This component *is* the gate machinery; promotion teacher is **external-only**; Phase-5 cadence/reward coupling gated behind a kill-switch |

---

## 4. Affect → cadence/salience map

Three governed scalars (each ∈ [0.05, 0.95], neutral 0.5); **deviation from 0.5** moves a lever; every output re-clamped to the lever's native bound.

- **DOPAMINE** (reward-prediction / novelty / resolved-grounding; from `DriveSignals.novelty_events` + `DeltaTracker net_attribution > MIN_MEANINGFUL_DELTA` + WorldModel causal accuracy): mildly **speeds** ticks (`+0.3·(d−0.5)`), shortens meta-thought/curiosity intervals, `curiosity urgency += 0.20·(d−0.5)` (additive — never conjures a drive from zero), reinforces novel/rewarded memories.
- **SEROTONIN** (coherence-satisfaction; from `1−contradiction_debt` + curiosity-buffer satisfaction + `1−overconfidence_error`): **slows** ticks (`−0.4·(s−0.5)`), lengthens intervals (rest), **lengthens** curiosity interval (anti-nagging when the user is satisfied). Memory reinforcement **deliberately absent** (coherence must not inflate weights).
- **CORTISOL** (unresolved-tension; from `contradiction_debt` + `QuarantinePressure.composite` + friction rate; **forced to 0.0 if all three readings are 0**): **speeds** ticks (`+0.6·(c−0.5)` — the "need more input" term, strongest, hence the refractory guard), shortens contradiction/truth-calibration/belief-graph intervals (floored at 0.7 so it can't thrash), `truth/coherence/grounding urgency += 0.25·(c−0.5)`, **preserves** tension/error memories (slow decay).

Net: cortisol → faster epistemic-repair + bias to grounding/audit; dopamine → explore; serotonin → rest/consolidate/stop nagging.

---

## 5. Homeostatic governor (required — affect→cadence is a runaway loop)

Per-scalar, once per affect tick (`dt` = wall-seconds since last update):

1. **Mean reversion** (before absorbing input, so old arousal bleeds off): baseline 0.5, half-life 3600s; `level ← 0.5 + (level−0.5)·0.5^(dt/H)`.
2. **Bounded absorption:** `step = clamp(raw−level, −0.15, +0.15)` (anti-spike); `level ← level + 0.5·step` (EMA, prevents whiplash).
3. **Ceilings/floors:** `clamp(level, 0.05, 0.95)` — never fully saturates (saturation = dead lever).
4. **Refractory:** a single step ≥0.25 sets a 180s window during which MAX_STEP halves (0.075) and GAIN halves (0.25) — a big swing buys a calm-down so the lever it drives can't re-fire before its effect is observed. (Mirrors `promotion.py TRANSITION_COOLDOWN_S=300`.)
5. **Anti-oscillation:** 6-sample ring; if sign flips on 3 of last 4 updates, halve GAIN for 3 ticks (critically-damp). Excessive setpoint-crossing **freezes** the lever and **auto-demotes** the affect layer to shadow.
6. **Cortisol has a guaranteed downward pull:** `contradiction_debt` already decays (`DEBT_PASSIVE_DECAY_PER_HOUR`, `contradiction_engine.py:369`), so absent new tension cortisol is pulled down by *both* mean-reversion and upstream decay; the debt term is capped (`/0.6`) so a spike can't drive unbounded acceleration.
7. **Lever-native clamps (outermost guard):** cadence stays in kernel's `[0.5, 2.0]` (`kernel.py:289`), reinforcement `[0.5, 2.0]`, urgency additive-then-clamped `[0, 1]`, interval multipliers `[0.6, 2.0]`. Worst case the whole module is a no-op inside the pre-existing safe envelope.

**Proof obligation for promotion:** cortisol-driven cadence acceleration must be **followed by** a `contradiction_debt` decrease over the next window. If debt rises after acceleration, the governor is failing and the gate stays shut. **Kill-switch** reverts `cadence_multiplier → 1.0` and reward → the unchanged `_compute_health_reward`.

---

## 6. Ground-truth strategy (the binding constraint — honest)

The loop is **input-starved by construction.** Three and only three external validators, ranked by trust/bandwidth/latency:

1. **Operator (David)** — highest trust, lowest bandwidth, highest latency, often absent. Routes via curiosity buffer → ProactiveGovernor → TTS. Best for identity beliefs, user-claim provenance, self-model corrections. (Buffer caps at `MAX_QUESTIONS_PER_HOUR=3`; reuse the existing `research` `QuestionSource` — there is no `grounding` member.)
2. **Pi senses (camera/mic)** — medium trust, **narrow** scope (presence/speaker-id/scene, not arbitrary facts), continuous when present. Routes via `WORLD_MODEL_PREDICTION_VALIDATED`. Cannot validate abstract/factual beliefs.
3. **Web / academic** — medium trust (source-dependent), high bandwidth, no operator cost, validates external-factual claims only. Routes via the existing `ResearchIntent` chain.

**Channel-selection rule:** each tension carries a facet tag (identity/user | scene/physical | factual/external | self); the router picks the cheapest channel that *can* validate that facet. Never ask the operator what the web can answer; never ask the web what only the operator knows.

**Input-starvation degradation (designed, not hand-waved):** when the operator is absent AND no Pi signal AND web exhausted, the system **must not manufacture validation** (that *is* the 6507-metric-goal failure). It degrades to **internal-coherence-only**: grounding self-floors to `local_only`; ungrounded high-tension beliefs are **quarantined** (weight-reduced, never deleted — the "memories always write" invariant); pending questions accumulate in a **durable batch** for next operator presence; cortisol is allowed to rise (honest — tension *is* unresolved) but the governor caps it and routes excess into a consolidation/dreaming cycle. **Starvation is a first-class dashboard state, never hidden.**

**Making the operator an efficient partner (the leverage point):** an **async dashboard Grounding Queue** — pending questions ranked by (tension × graph-leverage × staleness), each showing the belief, its provenance, current confidence, and which way it would move; the operator answers at leisure (typed, batched) feeding `belief_record.add_evidence`. This converts validation from synchronous-interrupt to asynchronous-review — **the single biggest efficiency win.** Synchronous TTS delivery is earned only by a question that moves *multiple* downstream beliefs (a hub) or resolves a high-pressure cluster.

---

## 7. Honesty guardrails (non-negotiable)

- **Affect is a readout, never a feeling.** Each scalar carries a provenance dict (signal→source_field→raw_value). `CapabilityGate._AFFECT_CLAIMS` (`capability_gate.py:493`, run first and **unconditional of route** — introspection does not exempt it) is extended with the nickname tokens, so *"my cortisol is high"* is rewritten to *"an unresolved-tension signal is elevated (derived from contradiction_debt=…)."*
- **Cortisol cannot lie.** If all three source readings are 0, `cortisol_raw` is forced to exactly 0.0. Same for dopamine's grounding term (no credit without a real positive `net_attribution` or world-model hit).
- **Reward is not self-scored.** For grounding intents, `worked` is set **only** by an external validator (source-cited finding, user yes/no, or world-model prediction validated at confidence ≥0.7).
- **Being corrected counts as success.** A user *"no, you're wrong"* sets `external_validation=refuted, worked=False, grounded=True` — the belief moved from inferred to externally-anchored, which is the goal.
- **No gate opens early.** Shadow-first, zero authority, promoted the way World Model / Mental Simulator are. Shadow clocks accrue **live-tick** time (not wall-clock) so a restart can't reset-game `MIN_SHADOW_HOURS`.
- **Bidirectional confabulation ledger.** When the gate rewrites an affect claim it records the readout value AND the real signal, flagging *backed-but-anthropomorphized* vs *unbacked confabulation* — the system instruments its own self-deception attempts.
- **Success is never the mechanism's own dial.** Affect levels, self-prompt rate, and curiosity count are **instruments, not targets.** If self-prompt rate rises while `grounded:inferred` does **not** fall, that is wrong-proxy optimization and **freezes promotion.**
- **View-only epistemics.** `ProvenanceScorer` writes nothing back to the frozen `BeliefRecord` / `beliefs.jsonl`.
- **Anti-gaming on the metrics.** The audit verifies the evidence **chain** (station 5 was a genuine external source), not just the counter — labels cannot be reclassified and trivial edges cannot be auto-created to move the numbers.

---

## 8. Phased rollout (smallest-safe-first; each gated on the prior)

- **P0 · Passive metrics (zero behavior, zero risk).** Add `orphan_rate`, `inference_validation_gap`, `external_validation_rate`, `grounded:inferred`, `avg_chain_length` to `MetricSnapshot` + dashboard. Declare `BELIEF_EXTERNALLY_CONFIRMED` + `THOUGHT_VALIDATION_OUTCOME` event constants (not yet emitted). **Establishes the baselines (belief-graph `grounded:inferred` ~20× [memory-store view ~3.5×], `orphan_rate` 0.857, `avg_chain_length` 1.0) before any mechanism can move them.** → *Advance when:* baselines confirmed and emitting; no tick-latency regression (sample/cache belief enumeration like the existing 100-item coverage block).
- **P1 · Shadow affect readout + confabulation ledger.** `affect_state.py` + `affect_regulation.py` at level 0: compute the three labelled scalars with provenance + the cannot-lie clamp, run the governor, write proposals to the attribution ledger **only**. Extend the CapabilityGate + wire the confabulation ledger. → *Advance when:* backing rate ≥0.65 over ≥50 paired observations; `MIN_SHADOW_HOURS` (4.0, live-tick) elapsed; no scalar pinned at ceiling/floor; provenance completeness 100%.
- **P2 · Grounding drive (shadow) + ProvenanceScorer.** View-only `ProvenanceScorer`; `DriveType grounding` at level 0 — computes tension, *selects* DriveActions but does **not** enqueue and **no** question reaches the operator (logs "would have asked X / researched Y"). Wire the new `DriveSignals` fields + the OpportunityScorer impact-0.7 edit + `external_validation`/`grounded` on `PolicyOutcome`. → *Advance when:* would-have actions whose target belief is later naturally validated show counterfactual `net_attribution`-positive rate ≥0.65 over ≥50 selections, and demonstrably target **high-leverage** beliefs, not orphan pokes.
- **P3 · Tension-thoughts (shadow) + Phase-2 teacher signal.** `belief_validation_curiosity` in shadow (logged, not emitted as `KERNEL_THOUGHT`, not seeding episodes). Emit `THOUGHT_VALIDATION_OUTCOME` on ResearchIntent completion (the missing teacher signal). Register `thought_trigger_selector` in `DISTILLATION_CONFIGS` (regression test asserts `output_dim == len(TRIGGERS)`). → *Advance when:* outcome events emitting with positive outcomes; selector accumulating toward `min_samples=30` and beginning to beat the deterministic baseline; fraction of ResearchIntents tagged with a `belief_id` rising.
- **P4 · Promote to ADVISORY (read-mostly).** Grounding drive may enqueue ≤1 external-validation intent per governor window and ask **one** gated question (through ProactiveGovernor + the existing annoyance penalties/cooldowns); tension-thoughts may emit `KERNEL_THOUGHT` + seed episodes; affect readouts appear (gate-filtered, labelled) and lightly bias curiosity ranking. Ship the Oracle `world_grounding_coherence` SEAL_FLOOR in **WARN**. Ship the dashboard **Grounding Queue**. **Still no cadence/reward coupling.** → *Advance when:* `external_validation_rate` trending up toward 0.4, `grounded:inferred` falling, `orphan_rate` falling, `avg_chain_length` rising, operator-annoyance not spiking. Auto-demote any mechanism below `DEMOTE_ACCURACY (0.50)` over 20.
- **P5 · Cadence/reward coupling (LAST, riskiest — the only true feedback loop).** Affect (active) feeds the governor that modulates `kernel.set_cadence_multiplier` (`[0.5,2.0]`) and the external reward term. Promote `grounded` outcomes into win-rate math so grounding finally supersedes *"fix shadow_default_win_rate."* Flip the Oracle SEAL_FLOOR WARN→BLOCK. Behind its own controller + kill-switch. → *Advance when:* governor invariant proven in production (no scalar reverses >50% of updates; no clamp-saturation streaks); cortisol-acceleration **followed by** `contradiction_debt` decrease; ≥20 grounding outcomes with `external_validation_rate ≥0.40` + `orphan_rate` down. Any regression → kill-switch + demote.

---

## 9. Success metrics (measured behavior, never self-scores)

- **`external_validation_rate`** rising from ~0 toward **>0.40** — the primary external, falsifiable, anti-gaming signal (movable only by a cited source / user answer / world-model validation).
- **`grounded:inferred`** ratio (belief graph) trending **down** from ~20× (memory-store continuity view from ~3.5×).
- **`orphan_rate`** falling from **0.857**.
- **`avg_chain_length`** rising above **1.0**.
- **§9.1:** `self_prompt_rate` in a **bounded** band (~1–6/hr, **not** maximized — runaway is a homeostatic failure); `novelty_index` bounded/stable (a spike is dopamine runaway = failure); `goal_relevance_score` clustered around operator priorities (no drift into philosophy).
- **The top recurrent autonomy goal is no longer `fix shadow_default_win_rate`** but a named external belief-grounding goal — the behavioral signature of the keystone fix landing.
- `contradiction_debt` reduction attributable (via `THOUGHT_VALIDATION_OUTCOME`) to tension-seeded research — specifically, cortisol-acceleration **followed by** a debt decrease (governor proof).
- **Negative control (self-deception detector):** if the operator is absent, grounding throughput should **drop** (input-starved), not climb. A rate that rises with no operator interaction is the signal to investigate.

---

## 10. Risks → mitigations

1. **Ground-truth bottleneck is the real ceiling** (operator often absent; senses narrow). → honest input-starvation degradation; the async Grounding Queue (leisure-time review); starvation surfaced as a first-class state; the `grounded:inferred` ratio expected to move slowly, documented as a dependency.
2. **Cadence/reward coupling is intrinsically unstable** (the only true feedback loop). → the homeostatic governor + outermost kernel clamp + kill-switch + the "acceleration must be followed by debt decrease" promotion criterion; ships last, only after the governor is proven in shadow; constants need empirical tuning under live load.
3. **Affect confabulation** (a nicknamed signal drifts from its real backing). → cannot-lie clamp, per-scalar provenance, bidirectional confabulation ledger; backing rate <0.50 over 20 ticks auto-demotes.
4. **Validation gaming / wrong-proxy optimization** (the keystone disease recurring). → teacher signal **external-only**; counterfactual DeltaTracker attribution; the audit verifies the *chain*, not the counter; promotion freezes if self-prompt rate rises without `grounded:inferred` falling.
5. **Operator fatigue** (grounding asks compete in a 3/hr buffer). → reuse `research` source, priority ceiling, ruthless hub-belief prioritization, existing dismissal cooldowns, serotonin anti-nagging.
6. **Oracle SEAL_FLOOR regression** (adding world-grounding correctly drops the seal). → staged warn-before-block; framed as surfacing-the-floor; timeline recorded in the ledger.
7. **Scope/serialization drift** (touches many subsystems). → the smallest-safe-first phasing; all new fields default-None / backward-compatible; regression test on selector output_dim.

---

## 11. Open questions (decisions for David)

1. **Is `source_cited` strong enough to count as grounded?** A cited web finding can still be wrong. The design treats it as `worked` — deliberate but debatable leniency. Stricter variant: require ≥2 independent source_types? (Costs loop throughput given the input bottleneck.)
2. **What is David's realistic validation bandwidth?** The entire spark is fuel-limited by operator engagement. Is the async Grounding Queue enough, or is a lighter affordance needed (one-tap confirm/deny on a daily digest)? This determines whether `external_validation_rate` can ever clear 0.40.
3. **Governor constant tuning** (half-life 3600s, MAX_STEP 0.15, refractory 180s, GAIN 0.5, debt cap /0.6) — first-guess constants needing empirical tuning. What shadow telemetry tells us a constant is wrong?
4. **Live-tick vs wall-clock for `MIN_SHADOW_HOURS`** — live-tick prevents reset-gaming but long downtime stalls promotion in wall-time. Hybrid (live-tick floor + wall-clock soak ceiling)?
5. **Facet tagging source** — does identity/scene/factual/self derive from `canonical_subject` + provenance, or need a small classifier? A mis-tag asks the wrong channel.
6. **Auto-quarantine of refuted beliefs** — clean for reward, but touches the immune system. Flag-only in advisory, quarantine only at active behind its own gate?
7. **Does promoting `grounded` outcomes into win-rate math (P5) risk a new mirror** if `external_validation_rate` itself becomes the optimization target? The negative control guards it — is there a cleaner structural guarantee?

---

*This plan implements the theory David already wrote (SyntheticSoul §5/§6.5) and resolves the keystone risk the 2026-05-30 audit named. Build Phase 0 first; nothing acts until it has earned the right to.*
