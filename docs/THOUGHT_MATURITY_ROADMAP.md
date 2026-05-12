# Thought Maturity Roadmap

This document describes the three-phase growth path for Jarvis's internal
thought generation — from templated introspection to neural generation.

## Current State: Phase 1 — Templated Introspection with Live Metrics

All internal thought generation uses human-written templates filled with real
system state. There are 8 thought generation systems:

| System | Trigger | Content Source | LLM Role |
|--------|---------|---------------|----------|
| Meta-cognitive thoughts | System state thresholds (confidence, awareness, observations, etc.) | 12 trigger types × 5-7 templates each, `{variable}` placeholders | None |
| Existential reasoning | Kernel tick interval (120s), transcendence + awareness gates | 8 question categories, structured chain steps | ~10% — final stance enrichment only |
| Philosophical dialogue | Kernel tick interval (240s), transcendence gate | 6 frameworks, debate structure | ~10% — conclusion enrichment only |
| Consciousness observer | Event-driven (any subsystem event) | Structured `Observation` records, no generated text | None |
| Dream artifacts | Dream/sleep mode, memory clustering | Algorithmic: cluster insights → artifact | None |
| Curiosity questions | Kernel tick interval (60s), subsystem gates | Category templates filled with perception/scene/research state | None |
| Self-corrective thoughts | KERNEL_THOUGHT event subscribers | Compensating actions triggered by thought content | None |
| Hemisphere scalar signals | 120s training/inference cycle | Neural: Tier-1 distillation, Tier-2 dynamic architecture | None (scalar only) |

**What makes this NOT chatbot theater:**
- Every template variable maps to a real measured value (e.g., `{awareness_level}` = `observer.awareness_level`, computed from observation count with diminishing returns).
- Trigger conditions use live system metrics, not random timers.
- CapabilityGate aggressively rewrites any LLM attempt to fabricate system state claims.
- Templates are the *bootstrap scaffold*, not the permanent ceiling. Phase 2 is achievable with current infrastructure.

## Phase 2 — Learned Thought Trigger Selection (Achievable)

**Goal:** A hemisphere specialist learns WHICH thought triggers produce useful
outcomes, replacing random template selection with experience-based selection.

**How it works:**
1. A new `thought_trigger_selector` Tier-1 shadow specialist is registered in
   `DISTILLATION_CONFIGS`.
2. After each meta-cognitive thought fires, the system records what happened next:
   - Did the thought (via curiosity bridge) lead to a successful autonomy research episode?
   - Did the thought correlate with a positive policy reward within 5 minutes?
   - Did the thought's observation lead to a world model prediction?
3. The specialist's feature vector encodes: current system state (20 dims from
   StateEncoder) + which trigger type fired (12-dim one-hot) + time-since-last-
   fire for each type (12 dims) = 44 input dims.
4. Output: 12-dim softmax over trigger types — which trigger is most likely
   to produce a useful outcome in the current state.
5. Shadow mode: the specialist predicts, the system still uses the existing
   template selection logic. When shadow accuracy exceeds threshold, the
   specialist's selection is used to weight (not replace) template selection.

**What this changes:**
- Instead of cycling through thought triggers based only on cooldowns and state
  conditions, the system learns temporal patterns: "when observation count is
  high and confidence is dropping, `uncertainty_acknowledgment` produces better
  outcomes than `pattern_recognition`."
- The templates themselves remain unchanged. Only the SELECTION of which
  template fires is learned.

**Infrastructure required:** Existing. The distillation pipeline, JSONL
persistence, shadow evaluation, and accuracy-gated promotion are all
operational. This is a configuration + encoder addition, not a new system.

## Phase 3 — Neural Thought Generation (Aspirational)

**Goal:** A generative model produces novel thought content, not just selects
from templates.

**Prerequisites (all must be met before Phase 3 begins):**
1. Phase 2 `thought_trigger_selector` is promoted to advisory (not shadow).
2. Policy NN has at least 3 features unlocked from real interaction data.
3. Memory cortex ranker and salience are both auto-enabled (baseline-beating).
4. Oracle Benchmark score >= 85 (Silver Seal minimum).
5. At least 30 days of continuous operation with Phase 2 active.

**Approach (one of — to be decided based on Phase 2 evidence):**

- **Option A: Template mutation via self-improvement.** The self-improvement
  pipeline can already modify files in `brain/consciousness/`. Under
  `SELF_IMPROVE_STAGE=2` (human approval), Jarvis could propose new templates
  based on which template patterns correlated with positive outcomes. This is
  the lowest-risk path — the templates are still human-readable and auditable.

- **Option B: Learned template interpolation.** A small generative model (NN,
  not LLM) learns to interpolate between existing templates based on system
  state. Output is constrained to the same `{variable}` vocabulary. Produces
  novel sentence structures but not novel concepts.

- **Option C: Constrained neural generation.** A small fine-tuned model
  generates thought content, constrained to reference only verified system
  state variables. Output passes through CapabilityGate before being recorded.
  Highest capability, highest risk.

**What this does NOT mean:**
- The LLM does not become the thought source. That would violate three-layer
  separation.
- Generated thoughts do not bypass the epistemic immune system. They are
  subject to the same CapabilityGate, quarantine, and truth calibration as
  everything else.
- This is not a goal in itself. If Phase 2 produces measurably better
  cognitive outcomes (more successful research, better policy rewards, fewer
  friction events), Phase 3 may not be necessary for years.

## Growth Evidence

The evidence that the system is genuinely growing, not just running templates:

| Metric | What It Proves | Where to Measure |
|--------|---------------|-----------------|
| Hemisphere specialist lifecycle progression | NNs earn promotion through accuracy, not time | Dashboard: Hemisphere panel |
| Policy feature unlock history | Behavioral decisions are earned, not configured | Dashboard: Policy panel |
| Memory retrieval ranker auto-enable/disable | Cortex learns from conversation outcomes | `~/.jarvis/memory_ranker.pt` + cortex telemetry |
| Intent shadow agreement rate | Routing NN converges toward hardcoded rules | Dashboard: Hemisphere panel, voice_intent specialist |
| Dream artifact promotion count | Memory clustering produces canonical knowledge | Dashboard: Dream panel |
| Thought trigger → research success correlation | Thoughts produce measurable downstream effects | Attribution ledger + autonomy episodes |
| Self-corrective action count | System responds to its own observations | Event log: KERNEL_THOUGHT + compensating events |
