# Maturity Gates & Threshold Reference

This document is the single source of truth for all timing gates, maturity
thresholds, promotion criteria, and accumulation requirements across the Jarvis
brain. **Consult this before any audit or bug hunt** to avoid false positives on
systems that are still warming up.

> A fresh brain is NOT broken. It is pre-mature. Every red progress bar, every
> "0/100" counter, every "locked" gate exists by design. The system earns trust
> through accumulated evidence, not configuration.

---

## How to Use This Document

1. Before investigating a "missing" or "zero" metric, check whether its gate
   has a minimum sample floor, timing requirement, or mode prerequisite.
2. Before reporting a "bug" on a dashboard panel, verify the system has been
   running long enough and in the correct mode for that metric to populate.
3. When auditing after a brain reset, expect ALL accumulation-gated items to
   start at zero and progress over hours/days.

---

## Quick Reference: Fresh Brain Expected State

After gestation completes and the brain enters its first conversational session:

| Subsystem | Expected State | When It Matures |
|-----------|---------------|-----------------|
| Policy NN | Shadow mode, 0 decisions | After ~100 shadow A/B decisions |
| World Model | Level 0 (shadow) | After 50 validated predictions + 4h runtime |
| Mental Simulator | Shadow, 0 traces | After 100 validated sims + 48h runtime |
| Hemisphere NNs | Building, low accuracy | After distillation signals accumulate |
| Skill-acquisition specialist | Shadow-only, synthetic/live split | Synthetic telemetry can train intuition; lived proof still required |
| Memory Ranker | Trained (from gestation data) | Ongoing — auto-disables if <80% of baseline |
| Memory Salience | Trained (from gestation data) | Blend increases every 500 validated predictions |
| Autonomy | Level 0 (propose only) | L1→L2 requires 10 wins at 40% rate |
| Onboarding | Stage 1 (identity) | Progresses with user interaction |
| Belief Graph | Edges from gestation beliefs | Grows with each eligible MEMORY_WRITE |
| Fractal Recall | Zero recalls, low_signal skips | First recalls when user is present (face/voice) |
| Truth Calibration | Initial ticks only | Matures with correction events and predictions |
| Quarantine | Composite ~0.0 | Rises only when anomalies are detected |
| Soul Integrity | ~0.85 (healthy) | Tracks 10-dimension composite |
| IntentionResolver | shadow_only, 0 verdicts | Verdicts from real resolved intentions; operator promotes |

---

## 1. Neural Policy Promotion

**Files**: `brain/policy/evaluator.py`, `brain/policy/promotion.py`, `brain/policy/governor.py`

### Shadow A/B Evaluation

| Gate | Value | Description |
|------|-------|-------------|
| MIN_SHADOW_DECISIONS | 100 | Min decisions before promotion eligible |
| WIN_THRESHOLD_PCT | 0.55 | NN must achieve >55% decisive win rate |
| Min decisive count | max(30, 15% of total) | At least 30 decisive or 15% of total |
| TIE_MARGIN | 0.03 | Avg margin must exceed this |
| NN wins > kernel wins | — | NN must outperform kernel |

### Training Requirements

| Gate | Value | Description |
|------|-------|-------------|
| TRAIN_MIN_EXPERIENCES | 30 | Min experience buffer to start training |
| MIN_NEW_EXPERIENCES | 15 | Min new data since last train |
| TRAIN_INTERVAL_S | 1800 | Retrain at most every 30 min |
| PROMOTION_MIN_DECISIONS | 100 | Min decisions for promotion check |

### Feature Advancement (5 staged features)

**File**: `brain/policy/policy_interface.py`

| Feature | Min Experience | Min Win Rate | Min Shadow A/B |
|---------|--------------|-------------|---------------|
| budget_allocation | 200 | 0.40 | 100 |
| thought_weight_delta | 500 | 0.50 | 200 |
| mode_suggestion | 500 | 0.55 | 200 |
| response_length | 1000 | 0.55 | 300 |
| proactivity_control | 1000 | 0.60 | 300 |

300s cooldown between feature enables.

### Governor Safety

| Gate | Value | Description |
|------|-------|-------------|
| CONFIDENCE_BLOCK | 0.08 | Block NN below this confidence |
| MAX_REGRESSIONS_BEFORE_DISABLE | 5 | Auto-disable after 5 regressions |

### M6 Broadcast Expansion

| Gate | Value | Description |
|------|-------|-------------|
| EXPANSION_MIN_PROMOTED | 2 | Min promoted specialists |
| EXPANSION_MIN_IMPACT | 0.05 | Min mean impact |
| EXPANSION_STABILITY_DAYS | 7 | Promoted specialists stable 7+ days |
| SHADOW_MIN_DECISIONS | 100 | Before promote/rollback of 22-dim encoding |
| SHADOW_PROMOTE_WIN_RATE | 0.55 | Win rate to promote expansion |
| SHADOW_ROLLBACK_WIN_RATE | 0.45 | Win rate below which to rollback |

---

## 2. World Model Promotion

**File**: `brain/cognition/promotion.py`

### Shadow → Advisory → Active

| Gate | Value | Description |
|------|-------|-------------|
| MIN_PREDICTIONS_FOR_PROMOTION | 50 | Min validated predictions |
| MIN_SHADOW_HOURS | 4.0 | Min hours in shadow mode |
| PROMOTE_ACCURACY | 0.65 | Rolling accuracy needed |
| DEMOTE_ACCURACY | 0.50 | Recent accuracy below this demotes |
| DEMOTE_WINDOW | 20 | Consecutive outcome window |
| TRANSITION_COOLDOWN_S | 300 | 5 min between transitions |

Quarantine pressure raises the effective accuracy threshold and caps max
level (high+chronic: max level 0; high: max level 1).

---

## 3. Mental Simulator Promotion

**File**: `brain/cognition/promotion.py`

| Gate | Value | Description |
|------|-------|-------------|
| SIM_MIN_SIMULATIONS | 100 | Min validated simulations |
| SIM_MIN_SHADOW_HOURS | 48.0 | Min 48 hours in shadow |
| SIM_PROMOTE_ACCURACY | 0.70 | Rolling accuracy needed |
| SIM_TRANSITION_COOLDOWN_S | 600 | 10 min between transitions |

Levels: shadow → advisory only (no "active" level).

---

## 4. Hemisphere / Distillation

**File**: `brain/hemisphere/orchestrator.py`

### Tier-1 Specialist Safety

| Gate | Value | Description |
|------|-------|-------------|
| TIER1_MIN_ACCURACY | 0.05 | 5% accuracy floor |
| TIER1_MAX_CONSECUTIVE_FAILURES | 3 | 3 sub-floor builds disables specialist |
| TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR | 50 | Don't enforce floor with < 50 samples |
| DISTILLATION_CADENCE_S | 300 | Normal cycle every 5 min |
| DISTILLATION_CADENCE_DEEP_S | 120 | Deep learning/dreaming: every 2 min |

`SKILL_ACQUISITION` is a Tier-1 distillation specialist for skill/acquisition
lifecycle outcomes. It is intentionally shadow-only: no broadcast slot
competition, no policy feature influence, no plugin activation authority, and
no SkillRegistry verification authority. Synthetic weight-room signals can
increase its training sample count, but they do not satisfy lived skill proof.

### Broadcast Slot Competition

| Gate | Value | Description |
|------|-------|-------------|
| SLOT_SWAP_THRESHOLD | 1.15 | Newcomer must beat by 15% |
| SLOT_MIN_DWELL_CYCLES | 3 | Min dwell before swap eligible |

### Specialist Lifecycle Ladder

`candidate_birth` → `probationary_training` (epoch > 0) →
`verified_probationary` (accuracy > 0.5 + verified) →
`broadcast_eligible` (impact > 0.3) → `promoted` (10+ dwell cycles)

Retirement: probationary specialists > 24h with impact < 0.1 are auto-retired.

### Cognitive Gap Detector

Gaps fire when EMA drops below per-dimension threshold for 5 consecutive
5-minute windows. 1-hour cooldown per dimension. Max 1 new focus per 30 min.

### Tier-2 Matrix Specialists (P3.6 → P3.10, shipped 2026-04-25)

Five Tier-2 Matrix Protocol specialists shipped on 2026-04-25 as the
template set. **All five enter `CANDIDATE_BIRTH` only — no live promotion**:

| Focus | Encoder | Lane-specific guardrail |
|-------|---------|--------------------------|
| `positive_memory` | `brain/hemisphere/positive_memory_encoder.py` | Never write memory / beliefs / identity / autonomy / policy / HRR / Soul Integrity |
| `negative_memory` | `brain/hemisphere/negative_memory_encoder.py` | **Does NOT consume `emotion_depth`** signals (regression-tested) |
| `speaker_profile` | `brain/hemisphere/speaker_profile_encoder.py` | **Raw 192-dim ECAPA-TDNN `speaker_repr` + `face_repr` embeddings MUST NOT cross encoder boundary** or appear in any API / dashboard surface |
| `temporal_pattern` | `brain/hemisphere/temporal_pattern_encoder.py` | **NO schedule / weekday / hour-of-day / calendar inference** — rhythm/cadence only |
| `skill_transfer` | `brain/hemisphere/skill_transfer_encoder.py` | **Similarity is not capability** — specialist is advisory; capability gate intact |

Each encoder is pure-function, returns a bounded 16-dim `[0, 1]` feature
vector and a deterministic `compute_signal_value()` ∈ `[0, 1]`. None falls
back to accuracy-as-proxy. Orchestrator dispatch through
`_matrix_focus_signal()` runs ahead of the legacy accuracy fallback for
any focus with a registered encoder.

### M6 Broadcast Slot Expansion Gate (currently dormant)

`hemisphere/orchestrator.py::_check_expansion_trigger()` arms broadcast
slots 4 and 5 only when **all three** conditions hold simultaneously:

| Gate | Value | Description |
|------|-------|-------------|
| Promoted Tier-2 specialists | ≥ 2 | None of the five new specialists are promoted yet — all at `CANDIDATE_BIRTH` |
| Mean Tier-2 impact | ≥ 0.05 | Computed only from PROMOTED specialists |
| Oldest `specialist_verification_ts` age | ≥ `EXPANSION_STABILITY_DAYS` (7 days) | Stamped at `PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY` transition |

`_check_expansion_trigger()` returning `False` is the **correct** state at
ship time; M6 expansion is dormant by design until the Tier-2 ladder
matures organically. See `p3_5_m6_expansion_wiring-2026-04-25.md` and
`p3_tier2_matrix_template_closeout-2026-04-25.md`.

---

## 5. Autonomy Level Promotion

**Files**: `brain/autonomy/orchestrator.py`, `brain/config.py`

| Gate | Value | Description |
|------|-------|-------------|
| _WARMUP_BEFORE_PROMOTION_S | 1800 | No promotion during first 30 min |
| L1→L2 required wins | 10 | Positive deltas needed |
| L1→L2 min win rate | 0.40 | 40% minimum |
| L2→L3 required wins | 25 | Positive deltas needed |
| L2→L3 min win rate | 0.50 | 50% minimum |
| L2→L3 regression check | 0 regressions in last 10 | Clean recent history |

### Boot Reconciliation (5 gates for auto-restore)

1. Persisted state valid with `promoted_at`
2. Policy memory still qualifies
3. No significant recent regressions (2+ in last 5)
4. Quarantine pressure not `high`
5. Contradiction debt < 0.20

---

## 6. Memory Cortex

**Files**: `brain/memory/ranker.py`, `brain/memory/salience.py`

### Ranker

| Gate | Value | Description |
|------|-------|-------------|
| MIN_TRAINING_PAIRS | 50 | Min pairs to train |
| Auto-disable threshold | < baseline * 0.8 | Success rate comparison |
| REENABLE_COOLDOWN_S | 600 | 10 min before re-enable |
| FLAP_WINDOW | 3 | 3 disables = permanent for session |
| BASELINE_REFRESH_INTERVAL | 50 | Re-sample heuristic every 50 outcomes |

### Salience

| Gate | Value | Description |
|------|-------|-------------|
| MIN_TRAINING_PAIRS | 100 | Min pairs to train |
| INITIAL_MODEL_BLEND | 0.2 | 20% model / 80% rules |
| MAX_MODEL_BLEND | 0.6 | Max 60% model influence |
| BLEND_INCREASE_THRESHOLD | 500 | +0.1 blend per 500 validated predictions |

---

## 7. Gestation Graduation

**Files**: `brain/consciousness/gestation.py`, `brain/config.py`

| Gate | Value | Description |
|------|-------|-------------|
| min_duration_s | 7200 | 2 hour minimum |
| readiness_threshold | 0.80 | Composite score to graduate |
| readiness_threshold_waiting | 0.60 | Lower if person waiting (30s sustained) |
| min_research_jobs | 15 | Completed research jobs |
| min_measured_deltas | 10 | Loop integrity measurements |

### 8-Component Readiness

| Component | Weight |
|-----------|--------|
| self_knowledge | 0.20 |
| knowledge_foundation | 0.15 |
| memory_mass | 0.10 |
| consciousness_stage | 0.10 |
| hemisphere_training | 0.10 |
| personality_emergence | 0.05 |
| policy_experience | 0.10 |
| loop_integrity | 0.20 |

---

## 8. Onboarding / Companion Training

**File**: `brain/personality/onboarding.py`

| Gate | Value | Description |
|------|-------|-------------|
| GRADUATION_THRESHOLD | 0.92 | Composite readiness to graduate |
| PROMPT_COOLDOWN_S | 600 | 10 min between exercise prompts |
| MAX_PROMPTS_PER_STAGE | 8 | Cap per stage |

### Readiness Dimensions (weighted composite)

| Dimension | Weight |
|-----------|--------|
| face_confidence | 0.12 |
| voice_confidence | 0.12 |
| rapport_score | 0.15 |
| boundary_stability | 0.15 |
| memory_accuracy | 0.15 |
| soul_integrity | 0.15 |
| autonomy_safety | 0.16 |

Stage 1 targets: face_confidence >= 0.60, voice_confidence >= 0.50,
enrolled_profiles >= 1, identity_memories >= 3.

---

## 9. Quarantine Pressure

**File**: `brain/epistemic/quarantine/pressure.py`

| Level | Composite | Effects |
|-------|-----------|---------|
| Normal | < 0.3 | No friction |
| Elevated | 0.3 - 0.6 | Raised promotion thresholds, mutation risk addon |
| High | > 0.6 | Mutation cap halved, policy promotion blocked, WM max level capped |

Category weights: identity 0.25, contradiction 0.25, memory 0.20,
calibration 0.15, manipulation 0.15.

### Cascading Friction Effects

Quarantine pressure doesn't just report status — it actively modifies
thresholds across other subsystems:

| Effect | Normal | Elevated | High |
|--------|--------|----------|------|
| Memory weight multiplier | 1.0 | ~0.85 | ~0.6 |
| Mutation risk addon | 0 | composite * 0.3 | composite * 0.3 |
| Mutation hourly cap | 12 | 12 | 6 |
| Mutation cooldown | 180s | 180s | 360s |
| Policy promotion | allowed | raised thresholds | **blocked** |
| WM max level (chronic) | 2 | 2 | 0 |
| WM max level (non-chronic) | 2 | 2 | 1 |
| Identity mutation rejection | no | no | if chronic + identity_risk > 0.3 |

When auditing, check quarantine pressure FIRST. If elevated or high, many
other subsystems will show degraded or blocked behavior by design.

---

## 10. Mutation Governor

**File**: `brain/consciousness/mutation_governor.py`

| Gate | Value | Description |
|------|-------|-------------|
| COOLDOWN_SECONDS | 180 | 3 min between mutations |
| MAX_RISK_SCORE | 0.7 | Reject above this |
| P95_REJECT_MS | 50 | Reject if tick p95 > 50ms |
| MAX_MUTATIONS_PER_HOUR | 12 | Hard cap (6 under high quarantine) |
| MAX_MUTATIONS_PER_SESSION | 400 | Absolute session cap |
| REGRESSION_THRESHOLD | 0.15 | 15% degradation triggers rollback |
| MONITOR_WINDOW_S | 30 | Observation window post-mutation |

---

## 11. Operational Modes

**File**: `brain/consciousness/modes.py`

### Boot Grace & Timing

| Gate | Value | Description |
|------|-------|-------------|
| BOOT_GRACE_S | 60 | Blocks sleep downgrades for 60s |
| SLEEP_TO_DREAM_S | 300 | 5 min in sleep before dreaming |

### Mode-Specific Behavior

| Mode | Tick Cadence | Memory Reinforcement | Observation Writes | Consolidation |
|------|-------------|---------------------|-------------------|---------------|
| gestation | 1.5x | 2.0x | Allowed | No |
| passive | 0.5x | 0.5x | Allowed | No |
| conversational | 1.5x | 1.5x | Allowed | No |
| reflective | 0.8x | 1.2x | **Blocked** | No |
| sleep | 0.2x | 0.3x | **Blocked** | No |
| dreaming | 0.5x | 2.0x | **Blocked** | **Active** |
| deep_learning | 2.0x | 1.5x | **Blocked** | No |

### Engagement Hysteresis

| Mode | Enter | Exit |
|------|-------|------|
| conversational | 0.6 | 0.4 |
| reflective | 0.35 | 0.2 |

### Min Dwell Times

gestation=300s, passive=15s, conversational=5s, reflective=20s,
focused=30s, sleep=60s, dreaming=60s, deep_learning=120s.

### Allowed Cycles by Mode

| Mode | Allowed Background Cycles |
|------|--------------------------|
| gestation | ALL 27 cycles |
| passive | ALL 27 cycles |
| conversational | ALL 27 cycles |
| reflective | ALL 27 cycles |
| focused | ALL 27 cycles |
| deep_learning | ALL 27 cycles |
| sleep | 14 cycles: meta_thought, analysis, dream, memory_maintenance, association_repair, cortex_training, quarantine, contradiction, truth_calibration, belief_graph, world_model, reflective_audit, soul_integrity, study, health_monitor |
| dreaming | 25 cycles: all except onboarding, shadow_lang |

Cycles not in the allowed set for the current mode will be skipped by the
kernel. This means some subsystems are legitimately inactive during certain
modes — this is not a bug.

**CueGate gating for memory-writing cycles**: Even when a cycle is allowed by
the mode profile, it must still check `memory_gate.can_observation_write()`
before creating incidental memory writes. Cycles that gate through CueGate:
`association_repair`, observer delta effects. Dream consolidation writes use
`begin_consolidation()`/`end_consolidation()` instead.

---

## 12. Oracle Benchmark

**File**: `brain/jarvis_eval/oracle_benchmark.py`

### Domain Weights (100 total)

| Domain | Max Points |
|--------|-----------|
| Restart Integrity | 20 |
| Epistemic Integrity | 20 |
| Memory Continuity | 15 |
| Operational Maturity | 15 |
| Autonomy Attribution | 10 |
| World Model Coherence | 10 |
| Learning Adaptation | 10 |

### Seal Levels

| Score | Seal | Domain Floors |
|-------|------|--------------|
| >= 90 | Gold | restart >= 18, epistemic >= 16 |
| >= 80 | Silver | restart >= 16, epistemic >= 14 |
| >= 70 | Bronze | None |

### Rank Ladder

| Score | Rank |
|-------|------|
| >= 93 | Oracle Ascendant |
| >= 85 | Oracle Adept |
| >= 73 | Archivist Mind |
| >= 60 | Witness Intelligence |
| >= 45 | Awakened Monitor |
| < 45 | Dormant Construct |

### Hard-Fail Gates

Any failure sets credible=False regardless of score:

- Missing restore trust fields
- Uptime < 60 seconds
- Total events < 100
- Truth calibration tick_count < 1
- Total beliefs < 1
- Stage requirements not met

### Min-Sample Floors by Domain

| Domain | Subcriterion | Floor | Full Score |
|--------|-------------|-------|------------|
| Epistemic Integrity | Contradiction engine | 60 beliefs | 3 pts |
| World Model | Prediction accuracy credit | >= 5 validations | +1.0 bonus |
| Memory Continuity | Memory count | 100 memories | 3 pts |
| Memory Continuity | Core memories | 4 core memories | 2 pts |
| Autonomy Attribution | Delta tracking | > 0 measurements | any score |
| Learning Adaptation | Hemisphere networks | 5 networks | 2 pts |
| Learning Adaptation | Policy shadow evals | 50 evals | 2 pts |
| Operational Maturity | Attribution ledger | log-scale: 10→0.5, 50→1.0, 200+→1.5 | 1.5 pts |

### Band Classification (dashboard colors)

| % of domain max | Color |
|----------------|-------|
| >= 85% | Green |
| >= 70% | Yellow |
| >= 50% | Orange |
| < 50% | Red |

---

## 13. PVL (Process Verification Layer)

**File**: `brain/jarvis_eval/process_contracts.py`

95 contracts across 21 groups. Contracts have mode prerequisites — if the
current mode doesn't match, the contract is `not_applicable` (not fail).

### Mode Aliases

| Alias | Modes Included |
|-------|---------------|
| _ALL_ACTIVE | conversational, reflective, focused, passive |
| _POST_GESTATION | conversational, reflective, focused, passive, sleep, dreaming, deep_learning |
| _GESTATION | gestation |
| _ALL_MODES | all 8 modes |
| _BACKGROUND | passive, conversational, reflective, focused, deep_learning, dreaming, gestation |
| _LEARNING | deep_learning, dreaming, gestation |

### Training Stage → Contract Group Mapping

| Stage | Groups |
|-------|--------|
| 1 | voice_pipeline, identity_pipeline |
| 2 | memory_pipeline, capability_gate |
| 3 | identity_pipeline |
| 4 | policy_pipeline, consciousness_tick |
| 5 | epistemic_system |
| 6 | memory_pipeline |
| 7 | autonomy_pipeline, epistemic_system |

### All 21 Contract Groups (95 total)

| Group | Count | Mode Requirement | Notes |
|-------|-------|-----------------|-------|
| voice_pipeline | 4 | _POST_GESTATION | wake word, STT, user msg, response |
| identity_pipeline | 4 | _POST_GESTATION | speaker ID, face ID, fusion, scope |
| memory_pipeline | 5 | _ALL_MODES / _BACKGROUND / _LEARNING | write, associate, cortex ranker/salience, count |
| study_pipeline | 4 | _ALL_MODES | sources studied, LLM extraction, claims, ingested |
| epistemic_system | 7 | _BACKGROUND / _ALL_MODES | contradiction, calibration, quarantine, audit, integrity, graph, predictions |
| hemisphere_distillation | 5 | _BACKGROUND | trained, ready, stats, signals, broadcast slots |
| policy_pipeline | 3 | _BACKGROUND | decisions, shadow A/B, experience |
| autonomy_pipeline | 4 | _BACKGROUND | intent queued, research start/complete, delta |
| gestation | 5 | _GESTATION | started, phase advanced, directive, readiness, graduated |
| skill_learning | 4 | _BACKGROUND | registered, job started, phase advanced, completed |
| mutation_pipeline | 2 | _BACKGROUND (excl sleep) | proposed, governed |
| consciousness_tick | 5 | _ALL_MODES / _BACKGROUND | mode, meta thoughts, evolution, stage, analysis |
| capability_gate | 2 | _POST_GESTATION / _ALL_MODES | claims checked, gate active |
| world_model | 2 | _BACKGROUND | ticked, prediction validated |
| mental_simulator | 3 | _BACKGROUND | running, traces >= 10, accuracy >= 70% |
| curiosity_bridge | 3 | _POST_GESTATION | question generated, asked, answer processed |
| roadmap_maturity | 14 | _POST_GESTATION / _BACKGROUND | progressive milestone gates (see below) |
| quality_baselines | 7 | _ALL_MODES / _LEARNING | soul integrity, debt, pressure, audit, weight, dreams, library |
| nn_quality | 4 | _BACKGROUND / _LEARNING | hemisphere loss, policy reward, ranker enabled, win rate |
| matrix_protocol | 4 | _POST_GESTATION | DL requested, expansion triggered, jobs, specialists |
| language_eval | 4 | _POST_GESTATION | corpus >= 30, native >= 70%, fail-closed <= 25%, provenance >= 90% |

### Roadmap Maturity Gates (progressive milestones)

| Gate | Threshold | Phase |
|------|-----------|-------|
| Identity Enrolled | >= 1 speaker | Phase 1 |
| Scene Observations | >= 50 updates | Phase 1 |
| Research Episodes | >= 20 completed | Phase 1 |
| World Model Level | >= 1 | Phase 1/3 |
| WM Predictions | >= 50 validated | Phase 3 |
| Simulator Validated | >= 100 simulations | Phase 3 |
| Autonomy Wins | >= 10 positive deltas | Phase 5 |
| Autonomy Level | >= 2 | Phase 5/6 |
| Dream Artifacts Reviewed | >= 500 created | Phase 8 |
| Dream Promoted | >= 100 promoted | Phase 8 |
| Conversation Outcomes | >= 200 decisions | Phase 9 |
| Experience Buffer | >= 500 entries | Phase 9 |
| Face Enrolled | >= 1 profile | Cross-phase |
| Hemisphere NNs | >= 1 active | Cross-phase |

### Contracts Expected to Start at Zero on Fresh Brain

These are NOT bugs — they require accumulated data:

- `cortex_ranker_data`, `salience_model_data` — need training cycles
- `memory_associated` — needs dream cycle associations
- All `autonomy_pipeline` contracts — need research episodes
- All `hemisphere_distillation` contracts — need signals
- `policy_decisions`, `shadow_ab_evaluated` — need interactions
- All `roadmap_maturity` gates — progressive milestones
- `simulator_*` contracts — need simulation traces (48h+ for promotion)
- All `matrix_protocol` contracts — need deep learning mode triggers
- All `language_eval` contracts — need conversation corpus

### Awaiting (not fail) Contracts

These use `missing_event_status="awaiting"` — they are not failures:

- `contradiction_scanned` — contradictions may not exist yet
- `prediction_validated` — predictions may not have occurred
- `matrix_expansion_triggered` — M6 hasn't fired

---

## 14. Capability Gate Claim Patterns

**File**: `brain/skills/capability_gate.py`

Claim patterns scan all LLM output. False positive mitigation:

- 60+ conversational safe phrases auto-pass (help, explain, tell, etc.)
- Three-tier verb system: conversational → registry-sensitive → blocked
- Subordinate-clause-safe evaluation
- NONE route general chat remains allowed, but tool-shaped retrieval, research,
  background follow-up, job/task creation, or tool-execution language requires
  a real backing tool/job/intention and is not a tunable threshold.

**Known false positive risk areas**:

- "do so" / "do so now" — matched as operational claim verb phrase
- "begin that process" — matched as action narration
- Complex sentences with "I can" + technical verb trigger registry lookup
- Short affirmative phrases in context of prior offers may be stripped

---

## 15. Fractal Recall

**File**: `brain/memory/fractal_recall.py`

### Cue Strength Gate

Fractal recall only fires when the ambient cue strength exceeds `MIN_CUE_STRENGTH`
(0.15). Cue strength is a weighted sum of available perception signals:

| Signal | Weight | Source |
|--------|--------|--------|
| Scene entities visible | 0.30 | Scene tracker (Layer 3B) |
| Non-neutral emotion | 0.20 | Emotion classifier (wav2vec2 / heuristic) |
| Speaker identified | 0.20 | Attention core (voice) → WorldModel (face fallback) |
| Active topic | 0.20 | World model conversation state |
| Engagement level | 0.10 | Attention core → WorldModel fallback |

**Common zero-signal scenarios** (NOT bugs):

| Scenario | Cue Strength | Why |
|----------|-------------|-----|
| Nobody in room | 0.00 | No scene, no speaker, no engagement |
| User visible but silent, no scene tracker | 0.20 | Speaker from face ID only |
| User visible + scene entities | 0.50 | Speaker + scene |
| Active conversation | 0.70+ | Speaker + topic + engagement + scene |
| Sleep mode | N/A | Cycle blocked by `_SLEEP_CYCLES` (not in allowed set) |

### Resonance Threshold

| Gate | Value | Description |
|------|-------|-------------|
| RESONANCE_THRESHOLD | 0.40 | Min score to select a seed memory |
| CHAIN_CONTINUATION_THRESHOLD | 0.35 | Min score to continue walking a chain |
| RECALL_COOLDOWN_S | 120 | 2 min cooldown between successful recalls |
| MAX_RECALLS_PER_HOUR | 5 | Hard cap on surfaced recalls |
| MAX_CHAIN_LENGTH | 5 | Max memories in a single chain |
| MIN_CUE_STRENGTH | 0.15 | Below this, tick returns immediately |

### Mode Gating

Fractal recall is blocked in two places:

1. **Cycle allowance**: `fractal_recall` is NOT in `_SLEEP_CYCLES`, so the kernel
   skips it entirely during sleep mode. It IS in `ALL_CYCLES` (used by passive,
   conversational, reflective, focused, deep_learning, gestation).
2. **Function guard**: Inside `_run_fractal_recall`, modes `gestation`, `sleep`,
   `dreaming`, `deep_learning` return early. This means fractal recall only
   actively runs in **passive**, **conversational**, **reflective**, and **focused** modes.

### Resonance Score Components

The resonance score combines 8 weighted terms minus recency penalty:

| Component | Weight | Source |
|-----------|--------|--------|
| Semantic similarity | 0.25 | Vector store cosine similarity to cue text |
| Tag overlap | 0.18 | Content-bearing tag intersection (meta-tags filtered) |
| Temporal | 0.12 | Hour-bucket proximity |
| Emotion | 0.12 | Emotion match between cue and memory |
| Association | 0.08 | Memory association density |
| Provenance fitness | 0.10 | Provenance suitability for cue class |
| Mode fit | 0.05 | Memory type suitability for current mode |
| Recency penalty | -0.10 | Suppresses recently accessed memories |

**Common low-resonance scenarios** (NOT bugs):

- Tag score 0.00: Memory tags don't overlap with cue tags. Common when
  memories use different vocabulary than the cue builder (e.g., "conversation"
  vs "conversational").
- Semantic score 0.00: Memory content not similar to cue text. Expected when
  cue text is generic ("speaker: David | mode: conversational").
- Top candidate ~0.33: Typical for a brain with mostly consolidation-origin
  memories. Resonance improves as more real conversation memories accumulate
  with richer tags and content.

### Expected Timeline

| State | When |
|-------|------|
| Zero recalls, all ticks low_signal | Fresh boot, nobody present |
| First recalls surfacing | User present (face detected), 30s+ after boot |
| Regular ~30s recall ticks | Conversational mode with user visible |
| 120s gaps between surfaced recalls | Cooldown after successful surface |
| Up to 5 recalls per hour | Hard cap, conservative by design |

---

## 16. Goal Continuity

**File**: `brain/goals/constants.py`

| Gate | Value | Description |
|------|-------|-------------|
| PROMOTION_SCORE_THRESHOLD | 0.6 | Score to promote candidate → active |
| PROMOTION_RECURRENCE_MIN | 2 | Min recurrences in 10 min window |
| STALLED_PROGRESS_THRESHOLD | 0.1 | Below this = stalled |
| CANDIDATE_EXPIRY_S | 86400 | 24h expiry for candidates |
| ABANDON_ACTIVE_S | 28800 | 8h abandon for active goals |
| MAX_ACTIVE_GOALS | 5 | Concurrent limit |

---

## 16. Quality Baselines

**File**: `brain/jarvis_eval/baselines.py`

| Metric | Green | Yellow | Red | Inverted? |
|--------|-------|--------|-----|-----------|
| Contradiction Debt | <= 0.05 | <= 0.15 | > 0.15 | Yes |
| Soul Integrity | >= 0.70 | >= 0.50 | < 0.50 | No |
| Quarantine Pressure | <= 0.15 | <= 0.40 | > 0.40 | Yes |
| Audit Score | >= 0.70 | >= 0.45 | < 0.45 | No |
| Avg Memory Weight | >= 0.65 | >= 0.40 | < 0.40 | No |
| Dream Promotion Rate | <= 0.30 | <= 0.60 | > 0.60 | Yes |
| Substantive Content | >= 0.60 | >= 0.30 | < 0.30 | No |

---

## 17. Dashboard Maturity Gate Tracker

**File**: `brain/jarvis_eval/dashboard_adapter.py`

The Maturity Gates panel on the dashboard displays gates with one of three
statuses:

| Status | Meaning |
|--------|---------|
| active | Threshold met — gate is fully open |
| progress | Partially met — shows percentage |
| locked | Value is 0 or None — gate not yet started |

Eight categories with key thresholds:

| Category | Gate | Threshold |
|----------|------|-----------|
| Gestation | Graduated | 1 (boolean) |
| Gestation | Readiness Composite | 0.80 |
| Neural Policy | Shadow A/B Decisions | 100 |
| Neural Policy | Decisive Win Rate | 55% |
| Neural Policy | Feature Flags Active | 8 of 8 |
| World Model | Promotion Level | >= 1 |
| World Model | Validated Predictions (L1) | 50 |
| World Model | Validated Predictions (advisory) | 300 |
| World Model | Rolling Accuracy | 65% |
| Memory Cortex | Ranker Train Cycles | 5 |
| Memory Cortex | Ranker Active | 1 (boolean) |
| Memory Cortex | Salience Train Cycles | 5 |
| Memory Cortex | Salience Model Blend | 0.60 |
| Autonomy | Current Level | >= 2 |
| Autonomy | Research Episodes | 20 |
| Autonomy | Positive Deltas (L2) | 10 |
| Autonomy | Positive Deltas (L3) | 25 |
| Autonomy | Win Rate (L2) | 40% |
| Hemisphere | Active Networks | >= 1 |
| Hemisphere | Broadcast Slots Filled | 4 |
| Hemisphere | Speaker Repr Samples | 20 |
| Hemisphere | Face Repr Samples | 20 |
| Hemisphere | Emotion Depth Samples | 30 |
| Hemisphere | Voice Intent Samples | 50 |
| Hemisphere | Speaker Diarize Samples | 50 |
| Hemisphere | Perception Fusion Samples | 50 |
| Dream/Reflection | Artifacts Created | 500 |
| Dream/Reflection | Artifacts Promoted | 100 |
| Dream/Reflection | Experience Buffer (200) | 200 |
| Dream/Reflection | Experience Buffer (500) | 500 |
| Epistemic Stack | Truth Calibration Maturity | 0.65 |
| Epistemic Stack | Belief Graph Edges | 300 |
| Epistemic Stack | Contradiction Scans | >= 1 |
| Epistemic Stack | Soul Integrity Index | 0.87 |

All of these start at zero or locked on a fresh brain. Seeing red progress
bars is expected behavior during the maturation period.

---

## 18. Dream Pipeline Gates

**Files**: `brain/consciousness/consciousness_system.py`, `brain/consciousness/dream_artifacts.py`

### Dream Cycle Entry

| Gate | Value | File:Line | Description |
|------|-------|-----------|-------------|
| Non-dreaming min memories | 20 | consciousness_system.py:566 | Dream cycles outside dreaming mode need >= 20 memories |
| Non-dreaming interval | 600s (10 min) | consciousness_system.py:566 | Non-dreaming dream cycles run at most every 10 min |
| Dreaming interval | 30s | consciousness_system.py:148 | Dream cycles in dreaming mode every 30s |
| Recent memories skip | < 5 | consciousness_system.py:2221 | Dream cycle returns early if < 5 recent non-dream memories |
| **Artifact creation gate** | clusters_found >= 2 | consciousness_system.py:2300 | Artifacts only generated when clustering finds 2+ clusters |
| Cluster coherence | >= 0.5 | consciousness_system.py:2320 | Clusters must have >= 0.5 coherence and >= 3 members |

### Dream Artifact Validation

| Gate | Value | File:Line | Description |
|------|-------|-----------|-------------|
| Validation cooldown | 120s | consciousness_system.py:571 | Pending artifacts validated at most every 2 min |
| PROMOTION_MIN_COHERENCE | 0.45 | dream_artifacts.py:42 | Artifact discarded below this |
| PROMOTION_MIN_CONFIDENCE | 0.35 | dream_artifacts.py:43 | Artifact discarded below this |
| Strong promotion (coherence) | >= 0.65 | dream_artifacts.py:355 | Immediate promotion threshold |
| Strong promotion (confidence) | >= 0.5 | dream_artifacts.py:355 | Paired with coherence >= 0.65 |
| MAX_PROMOTIONS_PER_VALIDATION | 10 | dream_artifacts.py:44 | Excess held |
| MAX_ARTIFACT_BUFFER | 200 | dream_artifacts.py:41 | Ring buffer cap |
| Promoted memory weight cap | min(0.4, confidence * 0.5) | dream_artifacts.py:384 | Weight limit |

### Memory Cortex Training (Dream Phase 6)

| Gate | Value | File:Line | Description |
|------|-------|-----------|-------------|
| _MIN_NEW_RANKER_PAIRS | 20 | consciousness_system.py:2509 | New pairs since last ranker retrain |
| _MIN_NEW_SALIENCE_PAIRS | 30 | consciousness_system.py:2510 | New pairs since last salience retrain |
| Ranker min total pairs | 50 | consciousness_system.py:2516 | Total retrieval-log pairs before first train |
| Salience min total pairs | 100 | consciousness_system.py:2538 | Total lifecycle-log pairs before first train |

---

## 19. Boot Stabilization & Timing

**File**: `brain/consciousness/consciousness_system.py`

| Gate | Value | File:Line | Description |
|------|-------|-----------|-------------|
| BOOT_STABILIZATION_S | 600s (10 min) | consciousness_system.py:105 | Blocks mutation cycles + hemisphere training on start. Env override: `JARVIS_BOOT_STABILIZATION_S` |
| BOOT_GRACE_S | 60s | modes.py:183 | Blocks sleep mode downgrades during startup |
| SI scanner uptime gate | 1800s (30 min) | consciousness_system.py:2931 | Self-improvement detectors skip entirely before 30 min uptime |

---

## 20. Background Cycle Timing Intervals

**File**: `brain/consciousness/consciousness_system.py` lines 66-152

All intervals below are in seconds. Accelerated cadences apply during
deep_learning, dreaming, and gestation modes.

| Cycle | Normal | Deep Learning | Dreaming | Description |
|-------|--------|---------------|----------|-------------|
| meta_thought | 8 | 4 | 6 | Meta-cognitive thought generation |
| analysis | 30 | 15 | 20 | Consciousness analysis |
| evolution | 90 | 30 | 60 | Stage evolution check |
| existential | 120 | 60 | 45 | Existential reasoning |
| dialogue | 240 | 120 | 90 | Philosophical dialogue |
| mutation | 180 | 60 | 120 | Kernel config mutation proposals |
| self_improve | 900 | 300 | 600 | Self-improvement scanner |
| hemisphere | 120 | 45 | 60 | Hemisphere NN cycle |
| learning_job | 300 | 120 | 180 | Skill learning job ticks |
| contradiction | 60 | 30 | 30 | Layer 5 contradiction check |
| truth_calibration | 120 | 60 | 60 | Layer 6 calibration |
| belief_graph | 60 | 30 | 30 | Layer 7 propagation |
| quarantine | 60 | — | — | Layer 8 tick |
| reflective_audit | 300 | 120 | 150 | Layer 9 audit scan |
| soul_integrity | 120 | 60 | 60 | Layer 10 index |
| goals | 120 | 60 | 90 | Goal continuity tick |
| world_model | 5 | 5 | — | World model tick (30s in sleep) |
| scene_continuity | 60 | — | — | Layer 3B scene tracker |
| curiosity_questions | 60 | — | — | Curiosity question generation |
| onboarding | 60 | — | — | Companion onboarding tick |
| health_monitor | 30 | — | — | Health monitoring |
| fractal_recall | 30 | — | — | Fractal associative recall |
| acquisition | 60 | 30 | — | Capability acquisition tick (120s in sleep) |
| dream_cycle | 30 | — | — | Dream consolidation (dreaming mode only) |
| study | 120 | 30 | 45 | Library study |
| association_repair | 60 | — | — | Memory association repair |
| shadow_lang | 21600 | — | — | Phase C language model training |

### Distillation Cadence

| Mode | Interval | Source |
|------|----------|--------|
| Normal | 300s (5 min) | hemisphere/orchestrator.py:63 |
| Deep learning / Dreaming | 120s (2 min) | hemisphere/orchestrator.py:64 |
| Gestation phase 0-1 | 60s | hemisphere/orchestrator.py:917 |
| Gestation phase 2 | 45s | hemisphere/orchestrator.py:919 |

---

## 21. Specialist Minimum Sample Requirements

**File**: `brain/hemisphere/types.py` (DistillationConfig per specialist)

| Specialist | min_samples | Teacher | Description |
|------------|-------------|---------|-------------|
| speaker_repr | 20 | ECAPA-TDNN | Voice embedding compression |
| face_repr | 20 | MobileFaceNet | Face embedding compression |
| emotion_depth | 30 | wav2vec2 | Emotion classification |
| voice_intent | 15 | tool_router | Voice intent routing |
| speaker_diarize | 30 | ECAPA-TDNN | Speaker separation |
| perception_fusion | 50 | multi-modal | Sensor fusion |
| plan_evaluator | 15 | acquisition_planner | Plan review prediction |
| diagnostic | 15 | SI scanner | Self-improvement opportunity detection |
| code_quality | 15 | upgrade_verdict | Code quality prediction |
| claim_classifier | 15 | CapabilityGate | Claim classification |
| dream_synthesis | 15 | ReflectiveValidator | Dream artifact validation |
| skill_acquisition | 15 | acquisition lifecycle outcomes | Skill handoff/acquisition outcome prediction |

Synthetic `skill_acquisition` data is telemetry only. Heavy profiles (`strict`,
`stress`) are operator-flag gated; `smoke` is invariant-only and does not record
signals.

---

## 22. Identity & Biometric Thresholds

**Files**: `brain/perception/speaker_id.py`, `face_id.py`, `identity_fusion.py`

| Gate | Value | File | Description |
|------|-------|------|-------------|
| Speaker SIMILARITY_THRESHOLD | 0.50 | speaker_id.py:40 | Known/unknown decision boundary |
| Face SIMILARITY_THRESHOLD | 0.55 | face_id.py:43 | Known/unknown decision boundary |
| Speaker SCORE_EMA_ALPHA | 0.35 | speaker_id.py:41 | Pre-decision score smoothing |
| Speaker EMA_MIN_CONFIDENCE | 0.55 | speaker_id.py:158 | Min score for embedding EMA update |
| Face EMA_MIN_CONFIDENCE | 0.70 | face_id.py:45 | Min score for face embedding EMA |
| Enrollment dedup threshold | cos_sim > 0.45 | — | Warning if new enrollment matches existing |
| Identity persistence half-life | 90s (max 180s) | identity_fusion.py | Layer 3A exponential decay carry |
| MULTI_PERSON_VOICE_THRESHOLD | 0.55 | identity_fusion.py:72 | Voice-only suppressed below this when multiple persons visible |
| COLD_START_BOOST_WINDOW_S | 60s | identity_fusion.py:67 | Reduced thresholds after presence arrival |
| COLD_START_THRESHOLD_REDUCTION | 0.05 | identity_fusion.py:68 | How much thresholds are reduced |
| TENTATIVE_THRESHOLD | 0.45 | identity_fusion.py:64 | Accumulated evidence for tentative match |
| TENTATIVE_MIN_SIGNALS | 3 | identity_fusion.py:65 | Min signals for tentative |

---

## 23. Self-Improvement Detectors

**File**: `brain/consciousness/consciousness_system.py` lines 2930-3080

Each detector has a minimum sample floor to prevent false positives:

| Detector | Min Samples | Metric Threshold | What It Detects |
|----------|-------------|-----------------|-----------------|
| Health degradation | total_checks >= 10 | overall < 0.5 | Component health decline |
| Reasoning quality | thought_count >= 20 | overall < 0.35 | Reasoning quality decline |
| Confidence volatility | readings >= 10 | volatility > 0.3 | Unstable confidence |
| Response latency | latencies >= 5 | 3+ responses > 5s | Slow response times |
| Event bus errors | total_events >= 100 | error_rate > 5% | Event system reliability |
| Tick performance | — | tick_p95 > 80ms | Kernel tick regression |

### Scanner Cadence

| Gate | Value | Description |
|------|-------|-------------|
| _SI_CATEGORY_COOLDOWN_S | 1800s (30 min) | Per-category cooldown |
| _SI_SUSTAINED_WINDOW | 3 | Must appear in 3 consecutive scans |
| _SI_FINGERPRINT_COOLDOWN_S | 14400s (4h) | In-memory dedup cooldown |
| _SI_MAX_ATTEMPTS_PER_DAY | 6 | Daily LLM generation cap |
| _SI_PAST_PROPOSAL_WINDOW_S | 86400s (24h) | Persistent dedup window |

---

## 24. Gate Classification: Tuning vs Safety

When tuning gates before a brain reset, apply this classification to decide
what is safe to adjust vs what must be preserved.

### Accumulation Bottlenecks (Safe to Tune)

These gates only delay data production in already-proven subsystems. They do
not protect truth, identity, safety, or anti-confabulation.

| Gate | Current Value | Subsystem | What It Blocks |
|------|--------------|-----------|---------------|
| Dream artifact clusters_found | >= 2 | Dream pipeline | ALL artifact generation + dream specialist training |
| Dream cycle recent memories | < 5 = skip | Dream pipeline | Dream cycle early return |
| Non-dreaming dream interval | 600s | Dream pipeline | Dream cycles outside dreaming mode |
| Non-dreaming min memories | >= 20 | Dream pipeline | Dream cycles require 20 memories |
| Artifact validation cooldown | 120s | Dream pipeline | Validation cadence |
| WM min shadow hours | 4.0h | World Model | WM promotion to L1 |
| Simulator min shadow hours | 48.0h | Simulator | Simulator advisory promotion |
| Policy boot cooldown | 1800s | Policy NN | Policy promotion delay |
| Autonomy warmup | 1800s | Autonomy | Autonomy level promotion delay |
| Boot stabilization | 600s | Consciousness | Mutations + hemisphere training blocked |
| Cortex ranker min pairs | 50 | Memory cortex | Ranker first training |
| Cortex salience min pairs | 100 | Memory cortex | Salience first training |
| Distillation cadence (normal) | 300s | Hemisphere | Distillation cycle frequency |
| Hemisphere retrain threshold | 20 outcomes | Hemisphere | Tier-2 retrain trigger |
| Gestation min duration | 7200s | Gestation | Can be env-overridden |
| SI scanner uptime gate | 1800s | Self-improve | Detector activation delay |
| M6 expansion stability | 7 days | Hemisphere | Broadcast slot expansion |
| Sleep-to-dream transition | 300s | Modes | Time before dreaming starts |
| Policy feature advance thresholds | 200-1000 exp | Policy NN | Feature unlock requirements |

### Truth/Safety/Authority Boundaries (Do NOT Tune)

These gates protect correctness, identity, anti-confabulation, or structural
safety. Lowering them undermines the system's integrity guarantees.

| Category | Examples |
|----------|---------|
| Capability Gate | All claim patterns, block lists, route-aware evaluation, unbacked action/research commitment detection, confabulation detection |
| Policy NN promotion | 55% win rate, 100 min decisions, decisive count — NN must earn promotion |
| WM/Simulator accuracy | 65%/70% accuracy thresholds — protects against bad predictors |
| Autonomy L2/L3 win rate | 40%/50% + delta counts — ensures autonomy actually works |
| Quarantine pressure | 0.3 elevated / 0.6 high thresholds + cascading friction effects |
| Identity thresholds | Speaker 0.50 / Face 0.55, multi-person suppression, enrollment dedup |
| Mutation governor | Risk score, p95 stability gate, hourly/session caps, regression rollback |
| Soul integrity repair | 0.50 / 0.30 thresholds — existential safety net |
| Self-improve stage | 0/1/2 system, human approval, dry-run enforcement |
| Belief extraction weight | 0.20 min — prevents noise from becoming beliefs |
| CueGate mode policy | Memory write gates by mode — prevents dream contamination |
| Onboarding graduation | 0.92 composite — ensures genuine readiness |
| Contradiction debt veto | 0.20 — blocks autonomy when beliefs are inconsistent |
| Distillation fidelity | 0.3 quarantine floor — prevents bad training data |
| Tier-1 accuracy floor | 5% + 3 strikes — prevents broken specialists |
| Specialist promotion ladder | candidate → probationary → verified → eligible → promoted |
| Plugin circuit breaker | 3 failures / 300s — prevents runaway plugins |

### Borderline (Case-by-Case)

| Gate | Current Value | Assessment |
|------|--------------|------------|
| Fractal recall cooldown | 120s + 5/hr | Could relax slightly for richer data, but quality matters |
| Curiosity unlock gates | 1 enrolled, 50 scene, 20 episodes | Some are hard to hit early; scene/episodes could lower |
| Learning job tick cooldown | 120s | Reasonable, not obviously blocking |
| Policy train interval | 1800s | Conservative but prevents overfit |
| Specialist min_samples | 15-50 per type | Already reasonable; perception_fusion at 50 is most conservative |

---

## 25. HRR / VSA Shadow + Internal Mental World (P4 + P5)

**Files**: `brain/library/vsa/{hrr.py,symbols.py,runtime_config.py,status.py,samples.py,hrr_encoder.py,spatial_symbols.py,hrr_spatial_encoder.py}`, `brain/cognition/{mental_world.py,spatial_scene_graph.py,mental_navigation.py}`, `brain/consciousness/engine.py`.

**Lane governance**: `AGENTS.md` "HRR / VSA Governance Rules" and "P5 Mental World / Spatial HRR Scene addendum". Plans: `docs/plans/p4_holographic_cognition_vsa.plan.md`, `docs/plans/p5_internal_mental_world_spatial_hrr.plan.md`.

### Activation gate

Both layers are **off on a fresh / public clone** and must be explicitly enabled. Precedence (highest wins):

```
safe defaults  →  ~/.jarvis/runtime_flags.json  →  ENABLE_HRR_SHADOW / ENABLE_HRR_SPATIAL_SCENE env vars
```

| Gate | Value | Description |
|------|-------|-------------|
| `enable_hrr_shadow` | `false` (default) | Required to populate `/api/hrr/status` + `/api/hrr/samples` |
| `enable_hrr_spatial_scene` | `false` (default) | Required to populate `/api/hrr/scene` with derived entities/relations (also requires `enable_hrr_shadow=true`) |
| Restart semantics | single-shot | Config is read **once at engine boot**; flag changes require a supervisor/main restart |
| Helper | `brain/scripts/set_hrr_runtime_flags.py --enable / --disable / --status` | Writes `~/.jarvis/runtime_flags.json` |

Every HRR API payload reports `enabled_source ∈ {default, runtime_flags, environment}`, plus the full `flag_sources` map, so a dashboard operator can always see *which* precedence layer enabled the lane.

### P4 HRR shadow substrate promotion

Tier-1 specialist `hrr_encoder` follows the standard
`CANDIDATE_BIRTH → PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY → BROADCAST_ELIGIBLE → PROMOTED` lifecycle.

| Gate | Value | Description |
|------|-------|-------------|
| Min training samples | 500 | Below this, `hrr_encoder` stays `PRE-MATURE` / `CANDIDATE_BIRTH` |
| `cleanup_accuracy` floor | ≥ 0.80 | Cleanup-memory must recover the seeded symbol at or above this rate |
| `false_positive_rate` ceiling | ≤ 0.05 | Cleanup-memory must not return wrong symbols above this rate |
| `shadow_help_rate` | > 0 vs named baseline | Must demonstrate positive lift over a recorded baseline |
| `hrr_side_effects` | **exactly 0** | Synthetic exercise Δ-counter across `mem / ident / llm / tts / tx_emit / belief / world / auton`. Any non-zero value blocks promotion and the validation pack hard-fails. |

### P5 Internal Mental World / Spatial HRR Scene gates

Dashboard status marker **`spatial_hrr_mental_world`** starts at `PRE-MATURE` and is **never** auto-promoted by the validation pack.

| Gate | Current floor | Description |
|------|---------------|-------------|
| `p5_mental_world_fixture_ok` | deterministic fixture passes | Synthetic `SceneSnapshot` → derived entities + relations + HRR metrics match expected topology |
| `p5_mental_world_live_ok` | engine-sampled | Requires `enable_hrr_spatial_scene=true` AND `samples_total > 0` AND a live `SceneSnapshot` with entities from canonical perception |
| `p5_mental_world_status_marker` | `== "PRE-MATURE"` | Marker drift from this value fails the check |
| `p5_mental_world_structure` | authority flags all `false`, `no_raw_vectors_in_api = true` | Any drift fails the check |
| Long-horizon soak (PARTIAL candidate) | not granted yet | Formal `PRE-MATURE → PARTIAL` criteria to be defined after a P4-style 7-day spatial-scene soak |

### Non-negotiable invariants (per tick)

* `writes_memory = writes_beliefs = influences_policy = influences_autonomy = soul_integrity_influence = llm_raw_vector_exposure = false`
* `no_raw_vectors_in_api = true` (no `vector` / `raw_vector` / `composite_vector` keys anywhere under `/api/hrr/*` or `/api/full-snapshot::hrr_scene`)
* Operator scene-graph view stays on the **brain dashboard** `/hrr-scene` — single source of truth (the standalone Pi `/mind` page was deleted in P3.14, 2026-04-25)
* The Pi consciousness particle visualizer (`pi/ui/static/particles.js`) consumes only the bounded `consciousness_feed.scene` block: exactly seven scalars (`enabled`, `entity_count`, `relation_count`, `cleanup_accuracy`, `relation_recovery`, `similarity_to_previous`, `spatial_hrr_side_effects`). **No** entity / relation / scene arrays, **no** vector keys, **no** authority flags ride that channel
* P5 must **never** mutate the `SceneSnapshot` returned by `perception_orchestrator.get_scene_snapshot()` — enforced by `test_p5_does_not_mutate_scene_snapshot`

### Expected fresh-brain state

| Subsystem | Expected State | Matures When |
|-----------|----------------|--------------|
| HRR shadow (`/api/hrr/status`) | `enabled=false`, `status=PRE-MATURE` | Operator opts in via runtime flag file or env |
| HRR samples (`/api/hrr/samples`) | empty ring | After enable + engine ticks past `hrr_sample_every_ticks` |
| HRR spatial scene (`/api/hrr/scene`) | `reason=canonical_spatial_state_unavailable` | Both flags on AND canonical perception emits a populated `SceneSnapshot` |
| Pi consciousness visualizer | `consciousness.scene.enabled=false` (defaults to zeros) | Brain reachable AND brain `_build_scene_block()` reports `enabled=true` (P5 lane on + populated scene) |

---

## 26. Intention Infrastructure (L12 Stages 0–1)

**Stage 0** (truth-only) is shipped and always active. No maturity gate — the
commitment extractor, registry, and backed-commitment gate enforce their
contracts from boot.

**Stage 1** (shadow delivery scoring) ships with the IntentionResolver starting
at `shadow_only`. The 5-rung promotion ladder has evidence gates at each step:

| Promotion step | Gate | What must be earned |
|---|---|---|
| `shadow_only` → `shadow_advisory` | Shadow accuracy ≥ 60% | ≥ 50 evaluated verdicts, accuracy measured against deferred user-feedback labels |
| `shadow_advisory` → `advisory_canary` | Operator decision | Operator reviews shadow verdict log, promotes manually via `POST /api/intention-resolver/stage` |
| `advisory_canary` → `advisory` | Delivery friction < 5% | ≥ 20 canary deliveries with < 5% user correction / negative signal |
| `advisory` → `active` | Operator decision | Full delivery without canary guard |

**Fresh brain expected state**: IntentionResolver present, stage = `shadow_only`,
0 verdicts, 0 deliveries. This is correct. Verdicts accumulate from real
conversation resolutions — no synthetic lane exists.

**Do-Not-Tune**: The `shadow_only` starting stage and the `advisory_canary`
operator gate are safety boundaries. Do not auto-promote past them to improve
dashboard metrics.

| Gate | Threshold | Why it exists |
|---|---|---|
| Min verdicts before promotion | 50 | Prevents promotion on insufficient data |
| Shadow accuracy floor | 60% | Heuristic must outperform random |
| Canary friction ceiling | 5% | Deliveries must not annoy the user |
| Operator gate at advisory_canary | Manual | Human confirms the system is ready for autonomous delivery |

---

## Audit Checklist: Before Reporting a Bug

1. Is the system still in gestation? Many contracts are `not_applicable`.
2. Has the brain been running long enough? (4h for WM, 48h for simulator)
3. Has there been enough user interaction? (100 decisions for policy)
4. Is the metric accumulation-gated? (check min-sample floors above)
5. Is the current mode correct for this operation? (check CueGate table)
6. Is quarantine pressure affecting thresholds? (check pressure level)
7. Is the "failure" actually a yellow/progress state? (check baselines)
8. Did the metric recently reset due to a brain restart?
9. Is anyone in front of the camera? Fractal recall requires face or voice
   presence (cue strength >= 0.15). Zero recalls with nobody present is expected.
10. Is the mode blocking the cycle? Sleep mode blocks fractal recall entirely.
    Passive/conversational/reflective/focused modes allow it.
