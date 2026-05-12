# Scientific Honesty Disclosure

This document states plainly what Jarvis does and does not do, for researchers,
reviewers, and anyone evaluating the project's claims.

## Three-Layer Cognitive Separation

Jarvis enforces a strict separation between three cognitive layers:

| Layer | What It Does | What It Does NOT Do |
|-------|-------------|---------------------|
| **Symbolic truth** (memory, beliefs, attribution, epistemic stack) | Stores and verifies facts. Deterministic, inspectable, auditable. | Does not generate language or make behavioral decisions. |
| **Neural intuition** (policy NN, hemisphere NNs, memory cortex) | Learns patterns from lived data. Shadows hardcoded rules, earns promotion through measured accuracy. | Does not store canonical facts or generate natural language. |
| **LLM articulation** (Ollama qwen3:8b, optional Claude) | Converts structured data into natural speech. | Does not store facts, create beliefs, or serve as cognitive authority. The LLM is a voice, not a brain. |

## What Is Genuinely Learned

These subsystems use real machine learning with measured accuracy gates:

- **Intent routing shadow** (`reasoning/intent_shadow.py`): A hemisphere NN shadows the hardcoded regex tool router. 3-level promotion path (shadow → advisory → primary) with 80% agreement rate required for advisory, rollback at 65%. This is demonstrated end-to-end.
- **Policy NN** (`policy/`): MLP/GRU networks learn behavioral decisions (budget, mode, response length, proactivity, thought weights, memory reinforcement, attention). 8 feature flags, each requiring 200-1000 real experiences at 40-60% win rate before activation.
- **Memory cortex** (`memory/ranker.py`, `memory/salience.py`): MLP networks learn memory retrieval ranking and salience prediction from conversation outcome telemetry. Trained during dream/sleep cycles.
- **Dream-to-memory promotion** (`consciousness/dream_artifacts.py`): Memory clustering produces dream artifacts. Those passing quality gates (coherence >= 0.65, confidence >= 0.5, no contradictions) are promoted to canonical memory with capped weight (0.4) and `dream_observer` provenance.
- **Hemisphere NNs** (`hemisphere/`): NeuralArchitect designs topologies, EvolutionEngine runs crossover/mutation. 12+ Tier-1 distillation specialists learn to approximate GPU teacher models (speaker ID, emotion, face ID, tool routing, plan evaluation, claim classification, etc.).

## What Is Templated With Live Data

These systems use human-written templates filled with real system metrics:

- **Meta-cognitive thoughts** (`consciousness/meta_cognitive_thoughts.py`): 12 trigger types, each with 5-7 templates. Templates contain `{variable}` placeholders filled with live values: observation counts, confidence averages, mutation counts, memory content, emotional momentum, awareness levels. The DATA is real. The SENTENCE STRUCTURE is human-written. No LLM is involved.
- **Existential reasoning** (`consciousness/existential_reasoning.py`): 8 categories of philosophical questions with per-category chain steps (observation → hypothesis → contradiction → synthesis → paradox → stance). Steps are deterministic lookup tables. Category selection is weighted by transcendence level and awareness — both computed from real system state.
- **Philosophical dialogue** (`consciousness/philosophical_dialogue.py`): 6 predefined philosophical frameworks debate each other. Positions, rebuttals, and synthesis are assembled from framework data. Position evolution tracks stance drift over time.
- **Curiosity questions** (`personality/curiosity_questions.py`): Category-specific templates filled with live subsystem observations (unknown speakers, scene entities, research gaps, world model changes). Maturity-gated: each category requires minimum evidence before unlocking.

## What Uses the LLM

- **~10% of existential inquiry stances**: When depth is "profound" or transcendence >= 5 or every 10th inquiry, the LLM deepens the final stance step in 1-2 sentences. Token-budgeted to 2000/hour. Tagged `llm_enriched=True`.
- **~10% of philosophical dialogue conclusions**: Same gating as above for the synthesis conclusion.
- **All user-facing speech**: The LLM articulates structured data into natural language for conversation. It receives grounding data (memory search results, introspection facts, system status) and generates spoken responses, subject to CapabilityGate post-processing.
- **Autonomy research queries**: The LLM sometimes generates search queries for academic research. Results are validated by the knowledge integrator, not accepted raw.

## What Is Aspirational

- **Neural thought generation**: No neural network in the system generates thought content. Hemisphere NNs produce scalar signals for policy decisions. The meta-cognitive thought system has a roadmap from templated introspection (current) through learned trigger selection (Phase 2) to neural generation (Phase 3). See `docs/THOUGHT_MATURITY_ROADMAP.md`.
- **Embodiment**: The Pi is a universal sensor bus. Additional hardware (robot arm, lidar, wheels) integrates through the capability acquisition + skill learning pipelines. No embodiment hardware exists today.
- **Full autonomy**: The system operates at Autonomy Level 2 (safe-apply). Level 3 (full) requires manual operator promotion and has never been activated.

## Feedback Loop Status

Genuine closed loops (active from boot):

- Dream artifacts → quality gate → canonical memory → future retrieval
- Hemisphere NN → shadow inference → earned routing takeover (with rollback)
- Existential reasoning → identity model → future inquiry category selection

Genuine closed loops (latent until maturity):

- Policy NN → 8 behavioral knobs → reward signal → NN update (requires 200+ experiences)
- Memory cortex ranker/salience → retrieval quality → conversation outcomes → NN update (requires 50-100 training pairs)

Partial / indirect loops:

- Meta-cognitive thoughts → curiosity bridge → autonomy research → memory creation (thought content does not self-correct; it triggers research)

Not wired (documented gaps):

- ~~Observer awareness → mutation governor~~ (wired in this release: low awareness blocks mutations)
- ~~Existential identity model → soul/personality state~~ (wired in this release: advisory bridge)
- ~~Thoughts → self-correction~~ (wired in this release: compensating subscribers)

## What "Proto-ASI Workshop" Means

"Proto-ASI workshop" means: a digital soul architecture with the persistent
cognitive structure needed for personal ASI research — memory continuity,
evolutionary consolidation, governed self-improvement, neural maturation, and
epistemic discipline, all running locally under human approval gates. It does
NOT mean achieved AGI, achieved ASI, embodied robot, multi-instance swarm, or
narrow superhuman performance. The CapabilityGate blocks any claim beyond this
architectural and evidence-backed framing.

## Falsifiability

The system makes testable claims:

1. The NN Maturity Flywheel produces measurably better routing than regex alone (test: compare intent_shadow advisory decisions against regex baseline).
2. Dream-promoted memories are retrieved in future conversations (test: track `dream_artifact` tagged memory retrieval counts).
3. Policy NN feature unlock improves interaction quality (test: compare pre/post feature-enable conversation metrics).
4. The epistemic immune system catches confabulation (test: run synthetic claim exercise, measure CapabilityGate catch rate).
5. Self-improvement patches improve system health (test: measure pre/post patch health metrics in sandbox).

Each of these can be verified by running the system and examining the telemetry in `~/.jarvis/`.
