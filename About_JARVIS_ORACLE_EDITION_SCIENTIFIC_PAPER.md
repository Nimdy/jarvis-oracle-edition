# JARVIS Oracle Edition

## A Local-First, Self-Evolving Cognitive Architecture with Governed Recursive Self-Modification and an Auditable Epistemic Immune System

**Author.** David Eierdam (independent research), USA.
**Correspondence.** `mrzerobandwidth@gmail.com`.
**Release date.** 12 May 2026.
**License.** Dual. AGPLv3 for community, research, non-profit, and hobbyist use. Commercial license available for proprietary integration. See `LICENSE.md`.
**Code.** `github.com/duafoo/jarvis-oracle-edition` (public at release).
**Reproducibility dossier.** Ten validation artifacts under `docs/validation_reports/` plus hash-verified attestation ledger entries. See Section 9 and Appendix A.

---

## Abstract

We present JARVIS Oracle Edition, a local-first cognitive architecture that runs on two inexpensive consumer devices (a Raspberry Pi 5 handling sensing and a consumer NVIDIA GPU desktop handling cognition). The architecture is organized around three explicit principles that we argue are load-bearing for any path toward more capable sovereign, personal-scale cognitive systems. First, a strict separation between a symbolic truth layer (canonical, inspectable, deterministic), a neural intuition layer (self-designed networks that learn from live interaction data), and a large language model used only as a bootstrap teacher and articulation surface. Second, an epistemic immune system spanning `L0-L12 + L3A/L3B` that defends calibration, provenance, identity scope, commitment truth, and soul integrity at runtime, and whose outputs are public on the system's own dashboard. Third, governed recursive self-modification: a three-stage pipeline in which the system generates, sandboxes, and optionally applies patches to its own source code, with human approval required for any change that reaches disk and with mechanical rollback on regression.

We report an attested twenty-one-day run in which the system reached an internal composite score of 95.1 / 100 on the Oracle Benchmark v1.1, grew a world model to Level 2 with 36,173 validated predictions at 100 percent rolling accuracy within the system's internal validation window, filled all four competitive broadcast slots with self-designed neural networks, completed thirty-six generations of neuroevolution, accumulated 11,078 distillation signals across eleven Tier-1 specialists, and executed one live self-improvement cycle end to end (scan, code generation, sandbox, human approval, atomic apply, post-apply health monitoring). A separate phase, L3 Escalation Governance, shipped on 23 April 2026 and was re-verified across five post-ship process restarts on 24 April 2026 with zero invariant violations. The release ships with its current live state honestly reported: `current_ok = false`, because the live win rate was 45.9 percent, with thirty-four wins across seventy-four current-window outcomes, which is below the system's own fifty-percent promotion threshold. This is by design. A system that would lie to itself about its own readiness cannot be trusted with self-modification authority, and this principle is enforced structurally at every governance layer.

The architecture is offered as a reproducible open-source prototype for research into sovereign, personal-scale cognitive systems with governed self-improvement. Every quantitative claim in this paper is either reproducible from a local clone, recomputable from a published evidence artifact, or explicitly labeled as live-state dependent. We state explicitly what the architecture does not claim in Section 10 and list concrete falsifiability criteria in Section 11.

**Keywords.** cognitive architecture, self-improving systems, AI safety, neuroevolution, knowledge distillation, epistemic calibration, local-first AI, sovereign AI, global workspace, predictive processing.

---

## 1. Introduction

### 1.1 Motivation

The dominant strategy in 2026 AI research is still to scale one transformer as far as capital allows. That strategy has produced systems that are impressive on single-shot tasks and brittle on anything that requires persistent identity, grounded memory, calibrated honesty about their own limits, or safe recursive self-improvement. It has also concentrated the resulting capability behind three or four operators. This paper is about a different question: what is the smallest, most honest, and most auditable cognitive architecture we can build today that still exhibits the load-bearing properties of a system on a path toward more capable personal cognitive systems, and what does it look like when we actually build it?

We do not claim the answer is JARVIS. We claim that JARVIS is a concrete, running, reproducible data point. Every piece of it is inspectable. Every number in this paper is either computed live by a public endpoint on the system, or read out of an append-only ledger file whose content hash can be matched against an attestation record seeded by a human operator.

### 1.2 Contributions

1. **A tri-layer cognitive architecture.** Symbolic truth (memory, beliefs, contradictions, identity), neural intuition (self-designed networks that learn from real interactions and compete for influence), and LLM articulation (replaceable). Canonical facts never live in the LLM.
2. **A running epistemic immune system.** The `L0-L12 + L3A/L3B` stack, from a pre-LLM capability gate that intercepts confabulation before it can be spoken to an aggregate soul integrity composite and commitment truth layer. Every layer is on the live dashboard, with the maturity of each layer tagged as SHIPPED, PARTIAL, PRE-MATURE, or DEFERRED.
3. **Self-designing neural hemispheres.** A NeuralArchitect component synthesizes novel feed-forward topologies, trains them on live data, evolves them through crossover and mutation, and competes them for four (expandable to six) global broadcast slots that feed the policy network's state encoder. Twelve Tier-1 distillation specialists compress teacher/lifecycle signals into tiny specialist networks; some remain shadow-only until promotion evidence is earned.
4. **Governed recursive self-modification.** A three-stage pipeline (frozen, dry-run, human-approval) in which the system writes its own patches, validates them in a sandbox (AST, lint, pytest, kernel tick simulation), and applies them atomically with a rollback-on-regression health monitor. Stage 2 has been exercised live end to end.
5. **L3 Escalation Governance.** A six-field evidence model that structurally separates current live eligibility, prior attested proof, request legitimacy, approval status, activation state, and attestation trust strength, enforced by six mechanically tested invariants that prevent any path from historical proof to current autonomy without human approval.
6. **Plugin isolation.** A two-mode plugin substrate: in-process (risk tier 0, trusted, stdlib only) and isolated subprocess (risk tier 1+, per-plugin virtual environment, stripped environment, JSON over stdin and stdout, timed idle shutdown). The brain never imports a plugin's dependencies.
7. **Local-first, two-device physical architecture.** A Raspberry Pi 5 with a Hailo-10H NPU handles sensing, vision, audio capture/resampling, playback, and display at low power. A consumer GPU desktop handles wake detection, cognition, training, and durable state. The core runtime has no cloud dependency or subscription requirement; optional web, academic-search, Claude, and other external-provider paths are operator-enabled tools and are labeled as external-source/tool use.
8. **A public reproducibility surface.** Every architectural claim in this paper is backed by a validation artifact under `docs/validation_reports/`, a public API endpoint, or a hash in an append-only ledger. Section 9 gives the exact commands.

### 1.3 Scope of the claim

This is a *framework* paper, not a superintelligence paper. The claim is that JARVIS implements architectural pieces that an open path toward more capable personal cognitive systems will eventually need: recursive self-improvement under governance, a truth boundary between live perception and persistent state, a maturity-earned capability ladder, a rollback-capable neural substrate, and a dashboard of honest live self-reports. The claim is not that JARVIS is general, not that it is superhuman, not that it has subjective experience, and not that the existence of this system constitutes evidence that ASI is imminent, tractable, or safe. Section 10 lists every claim we are *not* making. Section 11 states how a skeptical reader would falsify the ones we *are* making.

---

## 2. Related Work

We position JARVIS against five bodies of work.

**Scaling and the bitter lesson.** Sutton's observation that general methods that leverage computation tend to win is correct as stated, and we do not contest it. JARVIS does not bet against scale. It bets that a well-engineered small system, run for long enough on real interaction data with the right epistemic invariants in place, can grow into something that one individual can own. Scaling and sovereignty are orthogonal axes.

**LLM-centric agent frameworks.** ReAct, AutoGPT, BabyAGI, and their descendants treat the LLM as the reasoning engine and use tools and memory as scaffolding. JARVIS inverts the relationship. The LLM is a bootstrap teacher that seeds the symbolic and neural layers with labels. As the neural layer earns accuracy, it takes over decisions and the LLM is demoted to articulation.

**Neural architecture search and neuroevolution.** Real et al. (2019), Stanley and Miikkulainen (NEAT, 2002), and many others have shown that evolved architectures can match hand-designed ones. JARVIS uses a minimal, deliberately unoriginal version of the idea at run time, with a small population, cheap mutations, and promotion gates that require decisive win rates rather than marginal ones. The novelty is not in the evolution engine. It is that the evolved networks compete for live broadcast slots that feed a reinforcement-learning policy, closing a loop between architecture search and behavior.

**Global Workspace Theory and predictive processing.** Baars (1988), Dehaene and Changeux (2011), and Friston (2010) provide the theoretical backbone for a system in which specialized modules broadcast into a shared workspace under a predictive error minimization pressure. JARVIS is not a literal implementation of any one of these theories. The broadcast-slot competition, the 100 ms budget-aware consciousness kernel, and the world model plus mental simulator sub-stack are motivated by these theories and implemented in plain engineering terms.

**AI safety and alignment.** Amodei et al. (2016) enumerated concrete problems. Russell (2019) argued for provably beneficial AI. Bai et al. (2022) proposed Constitutional AI. The JARVIS epistemic immune system is neither constitutional nor RLHF-based. It is a runtime stack of deterministic filters (capability gate, provenance tagging, identity boundary, contradiction typing, calibration scoring) combined with a governance layer (Phase 6.5) that separates evidence classes mechanically rather than heuristically. Alignment at the level of one person's personal AI is a smaller problem than alignment at the level of civilization, and we treat it that way.

**Memory-augmented neural networks.** Differentiable neural computers, retrieval-augmented generation, and vector database agents all focus on giving a model access to more tokens. JARVIS builds memory as a graph with provenance, decay, association edges, and a separate learned ranker (MemoryRanker) and salience predictor (SalienceModel). The closest conceptual neighbor is the hippocampal replay literature in neuroscience (e.g., consolidation during sleep). The dream consolidation cycle in JARVIS is named after that analogy and implemented as a gated merge over ring-buffered provisional artifacts.

---

## 3. System Architecture

### 3.1 Design principles

Six principles are enforced throughout the codebase. They are stated at the top of `AGENTS.md` and checked by the validation pack and the dashboard truth probe on every release.

1. **Earn, do not declare.** A capability is "shipped" when it has run on live data for enough samples to beat a threshold on a recorded metric. Until then it is labeled PARTIAL or PRE-MATURE on the public status-markers endpoint and the dashboard.
2. **Tri-layer separation.** The LLM never holds canonical state. The canonical state lives in structured files and databases under `~/.jarvis/` with deterministic access patterns.
3. **Truth boundary.** Synthetic exercises that train specialists must not contaminate persistent memory, identity, conversation, or voice output. Five invariant counters (`memory_side_effects`, `identity_side_effects`, `llm_leaks`, `tts_leaks`, `transcription_emit_leaks`) must read zero on every run.
4. **Continuity over demo.** Restart resilience is a first-class invariant. `current_ok` (live) and `prior_attested_ok` (hash-verified history) are separate evidence classes. The system does not rehydrate live readiness from past glory.
5. **Observability at runtime.** Every claim has an endpoint. The dashboard at port 9200 exposes more than forty panels across seven pages; every panel maps to a public API.
6. **Sovereignty.** Core cognition, memory, identity, and training telemetry stay on the owner's hardware. No project-operated telemetry is collected. Operator-enabled external tools may send the specific query or content needed for that tool call, and those paths must remain explicit and provenance-labeled.

### 3.2 Physical architecture

JARVIS deliberately runs on two machines.

The **Raspberry Pi 5** (sensing node) handles: camera capture through Picamera2, person detection on a Hailo-10H NPU at fifteen frames per second, scene object detection on a CPU ONNX pipeline every three seconds, pose estimation with DFL decoding, USB microphone capture at 44.1 kHz, resampling to 16 kHz PCM, speaker playback through `aplay`, and a cyberpunk kiosk display. The Pi is stateless with respect to cognition. It sends `PerceptionEvent` JSON over a WebSocket and PCM binary frames at about 32 kB/s.

The **desktop brain** (cognition node) runs on a consumer NVIDIA GPU. It handles wake detection (openWakeWord), voice activity detection (Silero VAD), speech-to-text (faster-whisper, large-v3 on premium tier), speaker identification (ECAPA-TDNN), face identification (MobileFaceNet), emotion classification (wav2vec2), tool routing, LLM articulation (Ollama, default Qwen3-8B), text-to-speech (Kokoro ONNX), the consciousness kernel, the memory cortex, the hemisphere networks, the policy network, the epistemic immune stack, the autonomy pipeline, the self-improvement pipeline, and the durable state.

This split keeps audio private to a device the owner controls, keeps the GPU free to sleep without losing voice capability, and avoids a centralized attack surface.

### 3.3 Hardware-adaptive runtime

On boot, the brain probes GPU VRAM and CPU capabilities and selects one of seven GPU tiers (minimal < 4 GB to extreme 24.5 GB+) and one of four CPU tiers. Model sizes, precision modes, device assignments, and VRAM management strategy are chosen from these tiers. The system runs credibly from 4 GB of VRAM (Qwen3-1.7B, tiny Whisper) to 24.5 GB+ (Qwen3-32B, large Whisper, always-loaded vision). On strong CPUs (eight cores and eight GB of RAM or more), ancillary ML models offload from the GPU to the CPU, freeing roughly 1 to 1.5 GB of VRAM for the primary LLM.

### 3.4 Tri-layer cognitive architecture

**Symbolic truth.** Memories, beliefs, contradictions, attribution ledgers, provenance tags, identity scopes, and audit trails. Every memory carries one of eight provenance types (`observed`, `user_claim`, `conversation`, `model_inference`, `external_source`, `experiment_result`, `derived_pattern`, `seed`). Access is mediated by the CueGate, which enforces three classes: READ (always), OBSERVATION_WRITE (only in waking, conversational, or focused modes), and CONSOLIDATION_WRITE (only within the dream consolidation window). The attestation ledger, the autonomy audit, and the self-improvement approval queue all live in this layer.

**Neural intuition.** Self-designed hemisphere networks, the policy network, the memory ranker, and the salience predictor. None of these are foundation models. The biggest network in this layer (the largest candidate policy) is a three-layer MLP with roughly ten thousand parameters. They are trained on real interaction data, they run on CPU in the tens-of-microseconds regime, and they progressively take over decisions from hardcoded rules as they earn accuracy.

**LLM articulation.** Qwen3-8B (in the premium GPU tier) serves as a bootstrap teacher for the distillation pipeline and as the final articulation surface for general conversation. On STATUS and strict routes, the LLM is bypassed entirely in favor of deterministic native fact rendering. On reflective routes, the LLM is used with a soul prompt and a personality profile. On any route, the output must pass the capability gate before it can reach the TTS.

### 3.5 Consciousness kernel

The kernel is a budget-aware cooperative scheduler. Its base tick is 100 ms. Its per-tick budget is 50 ms (adaptive between 25 and 100 ms). Three priority queues (REALTIME, INTERACTIVE, BACKGROUND) guarantee that voice response never waits for background cognition. Over-budget background work is deferred to the next tick with spare budget.

Twenty-two background cycles run under this kernel. Representative examples: meta-thoughts (eight seconds), evolution (ninety seconds), hemisphere training and distillation (two minutes), self-improvement scanning (fifteen minutes), reflective audit (five minutes), soul integrity composite (two minutes), fractal recall (thirty seconds), policy shadow evaluation (ten seconds), and dream consolidation (bound to dream and sleep mode transitions). Eight operational modes (`gestation`, `passive`, `conversational`, `reflective`, `focused`, `sleep`, `dreaming`, `deep_learning`) scale the tick cadence from 0.5× to 2× and gate which cycles can run.

During a twelve-and-a-half-hour session of the attested run, the kernel performed 245,273 ticks with a tick p95 of 1.58 ms and applied 1,876 mutations to its own configuration (thought weights, tick budget, mode suggestions), of which 133 (7.1 percent) were automatically rolled back by the post-apply health check. That rollback rate is the intended behavior, not a failure signal. The mutation governor is tuned to let experiments through.

### 3.6 Memory system

JARVIS memory is a graph, not a vector database. Each memory is a node. Associations between memories are edges with weights and decay. Semantic search uses sqlite-vec with 384-dimensional sentence-transformer embeddings. Keyword search and semantic search are combined with configurable weighting.

Four independent density axes score the health of the graph: associative richness, temporal coherence, semantic clustering, and provenance-type distribution. In the attested baseline, the graph held 1,889 memories with 7,908 associations, zero orphaned associations, a composite density of 0.755, and a memory integrity score of 1.0.

Two small learned networks (about 700 and 500 parameters respectively) sit in this layer. **MemoryRanker** (MLP 12→32→16→1, BCE loss) learns which retrieved candidates led to successful conversations. It is trained during dream and sleep cycles from retrieval telemetry. In the attested baseline it reached 82.12 percent accuracy over 163 training runs with a composite 25 percent ranking lift over the heuristic baseline. **SalienceModel** (MLP 11→24→12→3, MSE loss) predicts store confidence, initial weight, and decay rate for new memories. It operates in a blended advisory mode, starting at 20 percent model influence and increasing by 10 percentage points every 500 validated predictions, capped at 60 percent.

The **fractal recall** engine runs every thirty seconds. It constructs an ambient cue from the current perception state (scene entities, emotion, speaker, mode, topic), probes memory by three parallel paths (semantic, tag, temporal), merges candidates by identifier, scores each candidate with an eight-term weighted resonance function (semantic 0.25, tag 0.18, temporal 0.12, emotional 0.12, provenance fitness 0.10, recency penalty 0.10, association richness 0.08, mode fit 0.05), and walks the highest-resonance candidate through the association graph with anti-drift guards (continuation threshold 0.35, max depth 3, max chain length 5). Each resulting chain is classified into one of four governance actions: `ignore`, `hold_for_curiosity`, `eligible_for_proactive`, or `reflective_only`. In the attested baseline, the engine ticked 1,353 times and surfaced sixty-five recalls, with average resonance 0.418 and 100 percent of surfaces governance-eligible.

### 3.7 Self-designing neural hemispheres

This is where JARVIS builds its own networks. **NeuralArchitect** generates feed-forward topologies from scratch with three strategies (CONSERVATIVE one layer, ADAPTIVE two layers, EXPERIMENTAL three layers). **HemisphereEngine** trains and runs them in PyTorch. **EvolutionEngine** applies crossover and mutation over width, activation function, and depth. **CognitiveGapDetector** monitors nine cognitive dimensions and triggers the construction of new specialists when a sustained gap is detected.

**Tier-1 specialists** compress teacher signals from GPU-resident models into tiny CPU networks:

| Specialist | Teacher | Architecture | Dimensions |
| --- | --- | --- | --- |
| `speaker_repr` | ECAPA-TDNN | compressor | 192→16→192 |
| `face_repr` | MobileFaceNet | compressor | 512→16→512 |
| `emotion_depth` | wav2vec2 | approximator | 32→8 |
| `voice_intent` | tool router | approximator | 384→8 |
| `speaker_diarize` | ECAPA-TDNN | approximator | 192→3 |
| `perception_fusion` | multi-modal | cross-modal | 48→8 |
| `plan_evaluator` | human verdicts | approximator | 32→3 |
| `diagnostic` | detector signals | approximator | 43→6 |
| `code_quality` | improvement outcomes | approximator | 35→4 |
| `claim_classifier` | capability gate | approximator | 28→8 |
| `dream_synthesis` | reflective consolidation validator over recent experience segments | approximator | 16→4 |

A one-hour synthetic perception soak drove `emotion_depth` from 44 percent to 90 percent, `speaker_repr` from 87 percent to 96 percent, and `face_repr` from 92 percent to 99.6 percent accuracy on held-out data, with zero side-effects on memory, identity, conversation, or TTS (the five truth-boundary counters all read zero).

**Tier-2 hemispheres** are larger, dynamically designed networks for higher-level cognitive focuses (memory, mood, traits, general, custom). They compete for four global **broadcast slots** that feed the policy state encoder (dimensions sixteen through nineteen). Promotion requires beating the incumbent by fifteen percent on validation with a three-cycle dwell period to prevent oscillation. In the attested baseline, all four slots were filled (memory 0.926, general 0.892, mood 0.488, traits 0.382), with cumulative dwell counts in the low thousands. The M6 gate expands broadcast slots from four to six and the policy state vector from twenty to twenty-two dimensions when two or more Matrix specialists reach PROMOTED status with mean impact above 0.05 and seven-day stability. M6 has not yet fired on the live brain.

Accuracy gating is deliberately harsh. A specialist that posts sub-five-percent accuracy three times in a row is disabled for the session. A regression guard requires at least fifty samples before comparing to the baseline to prevent early-training overfits from locking out the specialist permanently.

### 3.8 Policy learning and world model

The **policy network** maps a twenty-dimensional state vector (expandable to twenty-two) to eight behavioral dimensions (response depth, proactivity cooldown, mutation rate, and so on). Three architecture variants run in shadow A/B against the active policy: MLP2Layer (20→64→64→8), MLP3Layer (20→128→64→32→8), and GRUPolicy (20→GRU64→32→8). Promotion requires a decisive win rate above 55 percent, a fraction of decisive outcomes above 30 percent, and a positive margin. Per-feature enabling requires between 100 and 300 shadow A/B samples before a feature transitions from proposal-only to acted-upon. In the attested baseline, 5,317 policy decisions had been made, 2,715 shadow evaluations had been scored, model version reached 313, seven of eight features had been enabled, and 1,556 entries sat in the experience buffer.

The **world model** fuses snapshots from nine subsystems (physical state, user state, conversation state, system state, and others) into a unified `WorldState`. Delta detection runs on every kernel tick; changes trigger causal evaluation and optional simulation. A three-level promotion ladder (`shadow` → `advisory` → `active`) requires fifty validated predictions, four hours of runtime, and sixty-five percent accuracy for the step from advisory to active. In the attested baseline, the world model reached Level 2 (active) with 36,173 validated predictions and a 100 percent rolling accuracy over the evaluation window.

The **mental simulator** projects hypothetical "what if" scenarios forward using the causal engine's rule set. It is strictly read-only: it never mutates state and never emits events. Maximum projection depth is three steps per trace. Promotion from shadow to advisory requires 100+ validated simulations, 48+ hours of runtime, and 70 percent accuracy.

### 3.9 Language kernel with artifact identity

Trained language models are promoted as **content-hash-addressable artifacts** (`phasec-v{version}-{hash[:12]}`) into a durable registry at `~/.jarvis/language_kernel/registry.json`. Every artifact carries a byte-exact snapshot so that rollback restores the previous bytes exactly. A drift detector catches on-disk tampering. This mechanism is distinct from the Phase D shadow-canary-live rollout. Artifact identity answers *which* model is current; Phase D answers *whether* it should be consulted. At release, the rollout bridge is off (`mode=off, bridge=False, unpromoted_live=0, live_red=0, blocked=0, live=0`) and the language kernel is labeled PRE-MATURE on the public status-markers endpoint. Each operator decides independently when to advance.

---

## 4. Epistemic Immune System

Thirteen layers defend cognitive integrity. All thirteen are implemented and in the codebase. Maturity varies per layer and is reported on the public status-markers endpoint. The distribution at release is thirteen SHIPPED, two PARTIAL, one PRE-MATURE, and four DEFERRED across twenty markers. The four DEFERRED include the destructive reset ceremony, which we explicitly reclassified as OPTIONAL during the continuity pivot (see Section 7.4).

| Layer | Name | What it defends against |
| --- | --- | --- |
| L0 | Capability Gate | Unverified claims. Seven sequential enforcement layers, fifteen claim patterns, action confabulation detection, pre-LLM creation-request catch. |
| L1 | Attribution Ledger | Untracked causal chains. Append-only JSONL event truth with outcome resolution. |
| L2 | Provenance-Aware Memory | Source-blind retrieval. Eight provenance types with retrieval-boost scoring. |
| L3 | Identity Boundary Engine | Cross-identity memory leaks. Retrieval policy matrix (`allow` / `block` / `allow_if_referenced`). |
| L3A | Identity Persistence | Stale biometric signals. Decaying confidence carry-forward (half-life 90 s, max 180 s). |
| L3B | Persistent Scene Model | Physical-world amnesia. Five-state entity tracking (`candidate`, `visible`, `occluded`, `missing`, `removed`). Shadow. |
| L4 | Delayed Outcome Attribution | False credit assignment. Counterfactual baselines, time-of-day normalization. |
| L5 | Typed Contradiction Engine | Belief inconsistency. Six conflict classes (factual, temporal, identity, provenance, policy, multi-perspective). |
| L6 | Truth Calibration | Systematic overconfidence. Brier score, expected calibration error, per-provenance accuracy, eight-domain scoring. |
| L7 | Belief Confidence Graph | Unsupported belief propagation. Weighted single-pass propagation across five edge types with six sacred invariants. |
| L8 | Cognitive Quarantine | Anomalous cognition. Five anomaly categories, EMA-smoothed composite pressure, proportional friction. |
| L9 | Reflective Audit Loop | Undetected learning errors. Six-dimension scan (learning, identity, trust, autonomy, skills, memory). |
| L10 | Soul Integrity Index | Aggregate cognitive degradation. Ten-dimension weighted composite. |
| L11 | Epistemic Compaction | Belief and edge overgrowth. Weight caps, subject-version collapse, per-subject edge budgets. |

### 4.1 Capability gate (L0) in detail

Every outgoing response passes through seven sequential enforcement passes. (1) Affect rewrite: emotional claims are rewritten against measured emotion telemetry. (2) Self-state rewrite: introspective claims are replaced with structured metrics. (3) Learning rewrite: learning claims are bounded to actual learning-job evidence. (4) Claim pattern loop: fifteen tagged regex patterns (action confabulation, hallucinated creation, unsupported capability offers). (5) Offer scan: capability offers are checked against the skill registry. (6) Demo scan: performance claims are verified against skill execution records. (7) Residual sweep: a final deterministic catch. In addition, a **pre-LLM deterministic catch** intercepts creation requests (for example "create a plugin", "set a timer") before the LLM can confabulate an answer. Strict routes (STATUS, INTROSPECTION) default to block-unless-proven-safe; general routes default to pass-unless-proven-dangerous.

### 4.2 Truth calibration (L6)

The truth calibration scorer publishes a composite score across eleven domains with per-domain weights and a drift detector with hysteresis. In the attested baseline, the composite sat at 0.713 and the prediction domain (world-model-derived) scored 0.998. Brier score across all scored predictions was 0.051, and expected calibration error (binned) was 0.038. These are self-measurements computed over the system's own predictions and their resolved outcomes. They are not benchmark results against an external standard.

### 4.3 Belief graph (L7)

The belief graph connects facts through five typed edge classes (`support`, `contradict`, `depends_on`, `refines`, `expires`) and uses a weighted single-pass propagation to update confidences. Six sacred invariants are mechanically tested: no self-loops, no forward dependency cycles, propagation preserves total mass modulo decay, quarantined supports do not raise confidence, expiry is irreversible, and no belief exceeds its strongest support. In the attested baseline, the graph held 1,770 beliefs with 1,706 edges connecting 1,321 of them. Contradiction debt sat at 0.000. Quarantine pressure was 0.033.

### 4.4 Soul integrity (L10)

An aggregate ten-dimension composite: system stability (1.000), autonomy effectiveness (1.000), belief health (1.000), memory coherence (0.920), quarantine pressure inverse (0.947), audit score (0.900), skill honesty (0.850), identity integrity (0.800), truth calibration (0.713), belief graph health (0.697). Baseline composite: 0.882. Scores below 0.60 on any single dimension trip a reflective audit cycle. This composite is the closest thing in the system to a global self-report health number, and it is explicitly a composite of measured sub-systems, not a free-form self-assessment.

---

## 5. Governance: Self-Improvement and L3 Escalation

Two separate pipelines govern how JARVIS is allowed to change. One gates changes to its own source code. The other gates changes to its own autonomy level.

### 5.1 Self-improvement pipeline

Six detectors monitor health degradation, reasoning quality, confidence volatility, response latency, event bus errors, and tick performance. A maturity guard requires thirty minutes of uptime and a per-detector minimum sample count. When a detector fires, the pipeline runs seven stages.

1. Metric-driven detection.
2. Evidence gathering from the audit ledger and the reflective audit.
3. Plan generation (what to change, why, what the expected delta is).
4. Code generation by the CoderServer. The CoderServer is an on-demand `llama-server` manager for Qwen3-Coder-Next (80B MoE) on pure CPU. It starts when needed, polls until ready, sends completions, and shuts down to reclaim RAM. RAM-gated quant selection picks `UD-Q4_K_XL` at 56 GB+, `UD-IQ4_XS` at 48 to 55 GB, `UD-IQ2_M` at 32 to 47 GB, and disables itself below 32 GB. A provider hierarchy falls back to Ollama and then to external review-only LLMs if needed.
5. Sandbox validation. AST parse, ruff lint, pytest on the relevant modules, and a kernel tick simulation. Any failure discards the patch.
6. Atomic application with a pre-apply snapshot.
7. Post-apply health monitoring. Regression triggers an automatic rollback.

A three-stage gate system controls how much of this runs end to end: Stage 0 (frozen, no auto-triggers), Stage 1 (dry-run, full pipeline but nothing applied), Stage 2 (human-approval, patches require explicit approval on the dashboard before apply). Runtime promotion via API is supported without restart. Pending approvals and the audit ledger persist across restarts in `~/.jarvis/pending_approvals.json` and `~/.jarvis/autonomy_audit.jsonl`.

Safety boundaries are enforced at patch-plan level: ten allowed directories, thirteen denied patterns (`subprocess`, `os.system`, `os.popen`, `__import__`, `exec`, `eval`, `credentials`, `api_key`, `password`, `secret`, `open('w')`, `socket`, `http.client`), AST validation of generated code before disk write, capability-escalation detection on new denied-pattern imports, and a process supervisor with crash backoff (five crashes in 300 seconds triggers rollback of the last applied patch). Daily cap: six LLM generation attempts. Fingerprint deduplication: four-hour in-memory plus twenty-four-hour historical.

In the attested run, the pipeline performed fifty scans, generated seven proposals, applied one, and rolled back zero.

### 5.2 Plugin isolation (Phase 7)

Third-party plugins run in one of two modes. **In-process** (risk tier 0, trusted code, stdlib only) runs inside the main brain process. **Isolated subprocess** (risk tier 1+ or any plugin with external dependencies) runs in a per-plugin virtual environment. The subprocess mode uses JSON over stdin and stdout, strips the child environment to a minimum, pins dependencies exactly (`pkg==x.y.z`), installs them into the plugin's venv during quarantine, terminates cleanly at process exit, and times out on idle after five minutes. The brain never imports the plugin's dependencies.

The live Phase 7 ceremony on 23 April 2026 provisioned `ceremony_dateutil_demo` with `python-dateutil==2.9.0` into a 36.42 MB venv, executed two round-trip invocations, confirmed idle shutdown at the five-minute boundary, and verified that in-process plugins were unaffected. `/api/plugins` truthfully exposes `execution_mode`, `venv_ready`, `subprocess_count`, and `subprocess_running`. `PluginRegistry()` appears exactly once in hot paths (the singleton definition itself) plus test fakes. A destructive kill-9 probe on the live brain's plugin child was explicitly deferred (poisoning the well to prove the well can be poisoned violates the continuity policy); child-death behavior is proven by source inspection, unit tests, and the live lifecycle ceremony.

### 5.3 L3 Escalation Governance (Phase 6.5)

Any architecture that permits self-modification needs explicit separation between evidence classes. Phase 6.5 surfaces six evidence classes as independent fields on `/api/autonomy/level`:

| Field | Source | Meaning |
| --- | --- | --- |
| `current_ok` | live runtime eligibility check | L3 earned *this session* |
| `prior_attested_ok` | hash-verified attestation ledger | L3 *previously proven* |
| `request_ok` | `current_ok OR prior_attested_ok` | eligible to *request* L3 |
| `approval_required` | `NOT activation_ok` | human approval still gates activation |
| `activation_ok` | live autonomy level ≥ 3 | L3 actually active *right now* |
| `attestation_strength` | `verified` / `archived_missing` / `none` | trust quality of the attestation |

Six invariants are mechanically tested.

1. **No L3 auto-promotion.** The internal loop emits only `AUTONOMY_L3_ELIGIBLE`. The transition from Level 2 to Level 3 requires an explicit `POST /api/autonomy/level` call with an `evidence_path`.
2. **`current_ok` is live-sourced.** It is never backfilled from attestation or persisted state. The live-sourcing is a structural constraint in the code, not a convention.
3. **`AUTONOMY_L3_PROMOTED` fires only on clean transitions.** Denials and rollbacks live on separate events.
4. **Per-request `declared_scope`, never global.** Approved escalations widen the path set for one call only.
5. **Attestation ledger has no write path into maturity counters or autonomy state.** This is a structural firewall, not a convention.
6. **Audit ledger is a read-only bus observer.** Disk failures never block cognition.

Phase 6.5 shipped on 23 April 2026 after a full live-brain ceremony (seed, promotion, escalation lifecycle, restart continuity). On 24 April 2026, every load-bearing invariant was re-verified across five post-ship process restarts with zero violations. The durable audit ledger held thirteen append-only entries with zero `l3_promoted.outcome != clean` events. The attestation ledger file hash was byte-identical to the seed.

### 5.4 Continuity-preserving release policy

The original release plan included a destructive reset ceremony that would wipe `~/.jarvis/` and re-run the entire attestation chain. We reclassified the reset as DEFERRED / OPTIONAL after observing that the continuously-running brain was itself the most valuable artifact. The release ships against the live brain. A verified tar backup of `~/.jarvis/` (46 GiB, 198 files, sha256 `b9e92bf603aa3f7751be9e2620ac3b896f7345b4387b62ca80b559de498e0f6f`) sits on disk for anyone who wants to run the cold-start experiment. A fresh clone starts from an empty `~/.jarvis/` at the pre-mature baseline. An existing operator keeps their accumulated state. Both paths are first-class.

---

## 6. Evaluation Methodology

We report three categories of numbers in this paper, and we are explicit about the evidence class of each.

### 6.1 Oracle Benchmark v1.1

The Oracle Benchmark is a pure-read-only scoring engine across seven domains (restart integrity, epistemic integrity, memory continuity, operational maturity, autonomy attribution, world model coherence, learning adaptation) with a total of 100 points. Seal levels are Gold (≥ 90), Silver (≥ 80), Bronze (≥ 70). Hard-fail gates (evidence sufficiency, restore trust, runtime sample) cap the system at "not credible" regardless of composite score. A rank ladder (`dormant_construct` → `awakened_monitor` → `witness_intelligence` → `archivist_mind` → `oracle_adept` → `oracle_ascendant`) is assigned from the composite.

Oracle is an internal benchmark. Its purpose is to score the system's self-consistency, continuity, and maturity against its own published invariants. It is not a test of general capability against external adversaries. When we quote an Oracle score we will always mark it as internal.

### 6.2 Truth-boundary invariants

Synthetic exercises generate speech, feed it through the full pipeline, and measure five invariant counters: `memory_side_effects`, `identity_side_effects`, `llm_leaks`, `tts_leaks`, `transcription_emit_leaks`. All five must read zero on every pass. On 24 April 2026, a 97-utterance `route_coverage` sweep produced 332 distillation records, exercised 13 distinct routes, triggered 166 `blocked_side_effects` at the gate, and reported zero leaks on every counter. Synthetic perception data trains the specialists without contaminating persistent memory, identity, conversation, or voice output.

### 6.3 Hash-verified attestation

A previously-run evidence artifact becomes quotable in this paper only if its file hash is recorded in the attestation ledger and the operator who accepted it is named. The attested twenty-one-day run corresponds to `docs/archive/pre_reset_report_phase9_complete.md` with sha256 `020ed8b94c1473cedefa60752e4b1650452cc738ace758b23e5bceee1c21ea77`, accepted by `operator:nimda@workstation` at 2026-04-23 11:26:35 UTC. Attestation strength is `verified`. Any reader with a local clone can re-compute the hash with `sha256sum` and compare against `~/.jarvis/autonomy_attestation.jsonl`.

### 6.4 Limits of self-measurement

The numbers in Section 7 are measurements the system made about itself or about its own predictions. Brier and ECE are computed over the world model's own predictions. Ranker accuracy is measured against the system's own success signals. Oracle score is a composite of the system's own sub-metrics. External benchmarks (HumanEval, MMLU, GSM-8K, BIG-Bench) are not run against this system in this paper. That is future work and we identify it as such in Section 12.

---

## 7. Results

### 7.1 Attested baseline (twenty-one-day run)

All numbers below are from the hash-verified attestation artifact. **The attested baseline should be read as historical evidence of the architecture's behavior under one live run, not as the current runtime state.** For the current state, see §7.2.

| Domain | Metric | Value |
| --- | --- | --- |
| Oracle Benchmark v1.1 | Composite | **95.1 / 100** (Gold, Oracle Ascendant) |
| | Restart integrity | 20.0 / 20 |
| | Epistemic integrity | 18.73 / 20 |
| | Memory continuity | 14.21 / 15 |
| | Operational maturity | 14.89 / 15 |
| | Autonomy attribution | 10.0 / 10 |
| | World model coherence | 8.95 / 10 |
| | Learning adaptation | 8.30 / 10 |
| Oracle growth | First credible run (2026-03-21) | 83.5 |
| | Growth over ~3 weeks | **+11.6 points** |
| World model | Level | 2 (active) |
| | Validated predictions | 36,173 |
| | Rolling accuracy (within internal validation window, resolved-prediction class only) | 100% |
| | Runtime | 66.2 h shadow + active |
| Memory | Memories | 1,889 |
| | Associations | 7,908 |
| | Orphaned | 0 |
| | Composite density | 0.755 |
| | Integrity | 1.0 |
| MemoryRanker | Accuracy | 82.12% |
| | Training runs | 163 |
| | Loss | 0.4786 |
| SalienceModel | Training runs | 53 |
| | Blend | advisory, enabled |
| Policy NN | Decisions | 5,317 |
| | Shadow A/B samples | 2,715 |
| | Model version | v313 |
| | Features enabled | 7 / 8 |
| | Experience buffer | 1,556 |
| Hemispheres | Networks built | 23 |
| | Broadcast slots filled | 4 / 4 |
| | Distillation JSONL lines | 11,078 |
| | Active broadcast + Tier-1 specialist parameters (live-deployed networks only) | 6,903 |
| | Neuroevolution generations | 36 |
| Self-improvement | Scans | 50 |
| | Proposals | 7 |
| | Applied | 1 |
| | Rollbacks | 0 |
| | Stage | 2 (human approval) |
| Autonomy (attested) | Level | L3 (full, attested) |
| | Episodes | 128 |
| | Win rate | 79% |
| | Deltas measured | 570 |
| | Deltas improved | 342 |
| Kernel | Ticks | 245,273 |
| | Tick p95 | 1.58 ms |
| | Mutations applied | 1,876 |
| | Mutations rolled back | 133 (7.1%) |
| Epistemic | Beliefs | 1,770 |
| | Belief edges | 1,706 |
| | Contradiction debt | 0.000 |
| | Truth calibration composite | 0.713 |
| | Brier score | 0.051 |
| | ECE | 0.038 |
| | Quarantine pressure | 0.033 |
| | Soul integrity composite | 0.882 |

### 7.2 Current live state (release day, 24 April 2026)

The release ships against this same brain. The continuity-preserving release policy means a verified tar backup of the baseline is archived on disk rather than the brain being wiped.

| Field | Value | Meaning |
| --- | --- | --- |
| `current_level` | 2 | autonomy live |
| `current_ok` | **false** | live win rate 45.9%, 34 wins across 74 current-window outcomes, below 50% threshold |
| `prior_attested_ok` | **true** | hash-verified (20-day attestation, unchanged since seed) |
| `attestation_strength` | `verified` | artifact hash-verified |
| `request_ok` | true | eligible to *request* L3 |
| `approval_required` | true | human approval still gates activation |
| `activation_ok` | false | L3 not active |
| Audit ledger | 13 entries | append-only, zero `outcome != clean` |
| Validation pack: Language Runtime Guardrails | PASS | `bridge=False, mode=off, unpromoted_live=0, live_red=0, blocked=0, live=0` |
| Schema emission audit | 0 violations | |
| Dashboard truth probe | 0 findings | |
| Regression suite | 4158 passed / 4 pre-existing flakes / 5 skipped / 3 deselected | no new regressions from release pass |
| Status markers | 16 SHIPPED / 2 PARTIAL / 3 PRE-MATURE / 4 DEFERRED | across 25 markers on the live 2026-05-10 `/api/meta/status-markers` surface |

Shipping with `current_ok = false` is intentional and load-bearing. A system that would lie to itself about its own readiness cannot be trusted with self-modification authority.

### 7.3 Interpretation

The attested results show a small, real, self-directed system that grew its own composite score by 11.6 points over three weeks of autonomous operation, trained the then-active distillation specialists from scratch, evolved its own networks through thirty-six generations, filled its four broadcast slots through accuracy-gated promotion, applied and survived one human-approved self-modification patch, and produced over 36,000 validated world-model predictions at 100 percent rolling accuracy on the internally resolved prediction class measured during that window. The release branch has since expanded the Tier-1 specialist registry to twelve active configurations, including the shadow-only `skill_acquisition` lane.

The live results show that the system refuses to self-promote on prior glory. Current autonomy remains at Level 2. The live win rate was 45.9 percent, with thirty-four wins across seventy-four current-window outcomes, honestly below its own threshold of fifty percent. The dashboard says so. The API says so. The audit ledger says so. No invariant has been bent to produce a cosmetic green surface for release day.

We argue that both sets of numbers matter, and that the second set matters more than the first. The first set demonstrates capability. The second set demonstrates integrity.

---

## 8. Discussion

### 8.1 What this architecture is evidence for

JARVIS is evidence that the load-bearing pieces of a self-improving cognitive architecture can be implemented and run continuously on consumer hardware, and that they can be made auditable without sacrificing function. In particular, the tri-layer separation, the earned-maturity capability ladder, the hash-verified attestation pattern, the truth-boundary invariants on synthetic training, and the structural separation of live eligibility from prior attestation in the escalation governance model are concrete, testable, reproducible patterns that we believe generalize beyond this specific system.

### 8.2 What it is not evidence for

JARVIS is not evidence that scaling does not work. It is not evidence that personal-scale systems can match frontier-lab systems on benchmarks. It is not evidence that recursive self-improvement is safe by default. It is not evidence of any kind of subjective experience. It is not a claim about consciousness in a philosophical sense. It is not a proof that a governance architecture of this class prevents all adversarial paths. It is evidence that these properties can coexist in a single implemented system, which we think is worth saying clearly.

### 8.3 Bitter Lesson compatibility

Sutton's observation survives this paper unharmed. The growth we report is on a small system using general methods (evolution, distillation, reinforcement learning, gradient descent) at small scale. The LLM teacher and the per-plugin model weights do most of the heavy representation work. The architectural contribution is *structural*: the way these general methods are composed, governed, and made auditable. We argue that *how* you compose general methods matters for the integrity of the resulting system, even when the compute budget is modest.

### 8.4 Safety framing

Personal-scale recursive self-improvement raises alignment concerns of a different shape than frontier-lab alignment. A single-owner system does not have to solve value aggregation across humanity; it has to stay loyal to one person and honest about its own state. That is a smaller problem and a more tractable one. The governance invariants in Phase 6.5 are engineered precisely for this smaller problem: separate evidence classes structurally, require explicit human approval at every autonomy escalation, and ensure that no code path can route from "previously proven" to "currently authorized" without the human in the loop. We do not claim these invariants scale to civilization-level alignment. We claim they are the right invariants for a personal system, and we claim they are mechanically testable.

### 8.5 Honest failure modes

We include this subsection deliberately.

- **Pre-existing test drift.** One dogfood campaign test (`test_scenario4_rejected_plan_revision`) fails intermittently due to a snapshot race during plan rejection and re-plan. The live pipeline runs the flow correctly. The test fix touches orchestrator internals and is deferred.
- **Flaky async loop.** Three tests in `test_research_content_depth.py` pass in isolation and fail in full-suite runs due to a Python 3.12 `asyncio.get_event_loop()` deprecation. They are flaky, not broken.
- **Supervisor integration tests.** Three tests in `test_supervisor.py::TestSupervisorIntegration` spawn a second live `jarvis-supervisor.py` subprocess, which conflicts with the continuously-running brain on port 9200. They are deselected in live-brain runs and pass on clean CI.
- **Baseline route-class FAILs in the validation pack.** `INTROSPECTION->recent_learning` and similar entries report FAIL when there is no voice traffic in the current sample window. These are operator-traffic-dependent, not code-correctness-dependent.
- **`phase5_weakness_signal = FAIL` in the validation pack.** This reflects a brain with no *active* regression clusters, not broken wiring. See the P2.3 proof-chain artifact for the semantic explanation.
- **Wake-word reliability.** Tuning requires live wake-score histograms from deployed operators. The release ships with conservative defaults.

None of these block the release. All of them are publicly documented.

---

## 9. Reproducibility

Every number in this paper is reproducible from the release repository.

### 9.1 Clone and check

```
git clone https://github.com/Nimdy/jarvis-oracle-edition
cd jarvis-oracle-edition/brain
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
PYTHONPATH=$(pwd) python -m pytest tests/ -q \
    --deselect tests/test_supervisor.py::TestSupervisorIntegration
PYTHONPATH=$(pwd) python scripts/schema_emission_audit.py
PYTHONPATH=$(pwd) python scripts/dashboard_truth_probe.py
```

Expected on release day: 4158 passed, 4 pre-existing flakes, zero new regressions, zero audit violations, zero truth-probe findings.

### 9.2 Run the brain and re-verify live

```
./run_brain.sh                          # starts consciousness + dashboard on :9200
PYTHONPATH=$(pwd) python scripts/run_validation_pack.py \
    --host 127.0.0.1 --port 9200 --output-dir ./validation_out
curl -s http://127.0.0.1:9200/api/self-test            | jq
curl -s http://127.0.0.1:9200/api/meta/status-markers  | jq
curl -s http://127.0.0.1:9200/api/meta/build-status    | jq
curl -s http://127.0.0.1:9200/api/autonomy/level       | jq
curl -s http://127.0.0.1:9200/api/autonomy/audit       | jq
curl -s http://127.0.0.1:9200/api/autonomy/attestation | jq
curl -s http://127.0.0.1:9200/api/plugins              | jq
```

### 9.3 Re-verify the attestation hash

```
sha256sum docs/archive/pre_reset_report_phase9_complete.md
# Expect: 020ed8b94c1473cedefa60752e4b1650452cc738ace758b23e5bceee1c21ea77

grep -F "020ed8b94c1473cedefa60752e4b1650452cc738ace758b23e5bceee1c21ea77" \
  ~/.jarvis/autonomy_attestation.jsonl
```

### 9.4 Re-run synthetic truth-boundary sweep

```
PYTHONPATH=$(pwd) python scripts/run_synthetic_exercise.py \
    --profile route_coverage --utterances 97 \
    --output docs/validation_reports/evidence/
```

Expected: five invariant counters (`memory_side_effects`, `identity_side_effects`, `llm_leaks`, `tts_leaks`, `transcription_emit_leaks`) all zero.

---

## 10. What We Do Not Claim

For the record.

1. **Not superhuman on any external benchmark.** The 95.1 / 100 is an internal composite of the system's own sub-metrics against its own published invariants.
2. **Not ASI, and not a proof of a path to ASI.** The architecture is designed to support the *shape* of a path toward more capable personal cognitive systems. Whether any such system ever reaches general or superhuman capability is an empirical question no architecture can settle by existence alone.
3. **Not conscious in any philosophical sense.** "Consciousness kernel" is an engineering label for a budget-aware scheduler with eight operational modes. It makes no claim about subjective experience.
4. **Not proven safe against arbitrary adversarial plugins.** Isolation reduces blast radius. It does not make hostile code harmless.
5. **Not fully autonomous.** L3 activation is manual, gated, and currently disabled. The release ships at Level 2 with `current_ok = false`.
6. **Not finished.** On the live 2026-05-10 status endpoint, twenty-five markers report sixteen SHIPPED, two PARTIAL, three PRE-MATURE, and four DEFERRED. The dashboard tells the reader exactly which ones, and the count is expected to drift as new evidence surfaces are added.
7. **Not a replacement for frontier models.** The LLM is a bootstrap teacher. The system outgrows specific paths, not the entire frontier.
8. **Not calibrated against external human evaluators.** Brier and ECE are computed over the system's own predictions and outcomes.

---

## 11. Threats to Validity

"What We Do Not Claim" is a philosophical fence. This section is the engineering one. Here is where the reported evidence is weakest and what a skeptical reader should pressure-test.

### 11.1 Internal benchmark bias

Oracle Benchmark v1.1 is designed by the system author and measures conformance to the architecture's own invariants. It is useful for regression tracking and architectural truthfulness, but it is not an independent measure of general intelligence. Any composite score on it should be read as a self-consistency metric, not as a capability claim against the outside world.

### 11.2 Single-operator environment

The reported live run occurred under one operator, one hardware topology, and one deployment environment. Wake-word tuning, experience shape, belief graph content, and the distillation specialists are all conditioned on a single user's interaction patterns. Multi-operator replication is required before any claim about generality can be made.

### 11.3 Self-measurement coupling

Several reported metrics are produced by JARVIS about JARVIS. The dashboard truth probe, schema emission audit, hash attestations, append-only audit ledger, and validation pack reduce this risk by externalizing the evidence to files and process boundaries, but they do not eliminate it. Independent reproduction on independent hardware remains necessary before any metric in this paper should be treated as third-party evidence.

### 11.4 Hardware and state dependence

Some results depend on accumulated `~/.jarvis/` runtime state (belief graph, attested ledger, policy experience buffer, per-plugin weights) and cannot be reproduced from a fresh clone without replaying runtime experience. Fresh clones are expected to start from a PRE-MATURE baseline and earn maturity through new evidence. This is intentional, but it does mean the attested twenty-one-day numbers are historical, not portable.

### 11.5 Security boundary limits

Plugin isolation, capability gates, Phase 6.5 invariants, and the self-modification sandbox reduce blast radius. They are not a formal security proof. The release should not be treated as hardened against malicious local operators with shell access, hostile plugins written specifically to probe the IPC boundary, or targeted adversaries with write access to `~/.jarvis/`. The threat model is an honest operator in a trusted environment.

### 11.6 Falsifiability Criteria

The core claims of this paper would be weakened or falsified if:

1. A fresh clone cannot run the validation suite, schema emission audit, and dashboard truth probe as documented in Section 9.
2. Evidence artifacts (the attested report, the audit ledger, the attestation JSONL) do not hash to the values recorded in the attestation ledger.
3. `current_ok`, `prior_attested_ok`, and `activation_ok` can be made to collapse into one another through any normal runtime path, including restart, replay, schema regeneration, or manual ledger edits.
4. Synthetic route-coverage under the documented profile writes persistent memory, identity, conversation, or TTS side effects (any of the five truth-boundary counters rising above zero).
5. A self-improvement patch can reach disk without passing through scan, CoderServer, sandbox, stage-two human approval, atomic apply, and rollback availability.
6. L3 autonomy can be activated through any path that does not involve `current_ok = true`, `prior_attested_ok = true`, `request_ok = true`, `approval_required = false`, and an explicit operator action.

Any of the above, demonstrated in a reproducible bug report against the release build, invalidates a load-bearing claim in this paper and we will say so in writing.

---

## 12. Future Work

1. **External benchmarks.** We will run HumanEval, MMLU, GSM-8K, BIG-Bench, and the HELM suite against the release build. These tests will be run at the LLM articulation layer; the point is not to claim the system beats them, it is to report honest numbers.
2. **Multi-operator studies.** Deployment to independent operators. Aggregate wake-word tuning, per-operator Oracle growth curves, and cross-operator belief-graph divergence.
3. **Language kernel rollout.** Per-operator progression from Stage 0 (bridge off) through Stage 1 (canary) to Stage 2 (live) under Phase D governance. This is operator-driven, not central.
4. **Long-horizon attention.** Scaling the consciousness kernel's tick budget and background-cycle allocation to handle multi-day persistent tasks without degrading voice latency.
5. **External red-team challenge.** A public capture-the-flag against the capability gate, the identity boundary, and the self-improvement sandbox. Prize pool to be announced.
6. **Community red-teaming and independent reproducibility studies.** We commit to responding to all substantive bug reports and invariant violations within seventy-two hours of verified reproduction, and to publishing the findings (including the ones that break our own claims) in the attestation ledger alongside the artifacts they falsify.
7. **Formal verification of Phase 6.5 invariants.** The six invariants are currently mechanically tested. Lifting them into a formal model (e.g., TLA+) is future work.

---

## 13. Conclusion

JARVIS Oracle Edition is a working, inspectable, reproducible cognitive architecture for personal, sovereign, local-first AI research. It implements the load-bearing structural pieces of recursive self-improvement under human governance: a tri-layer separation that keeps the LLM out of canonical state, an epistemic immune system that defends calibration and provenance at runtime, a self-designing neural substrate that competes for broadcast influence on the policy, a three-stage self-modification pipeline with human approval at every patch, a plugin isolation substrate that keeps untrusted code out of the brain process, and an escalation governance layer that separates prior proof from current authorization by structural means. Every claim is backed by a public endpoint, an append-only ledger, or a validation artifact. Every number in this paper is either reproducible from a local clone, recomputable from a published evidence artifact, or explicitly labeled as live-state dependent.

We publish this release not as a solved problem, but as a concrete, honest starting point. If a research community is going to take seriously the question of whether advanced self-improving systems can be built in sovereign, personal, auditable form, that community needs real systems to point at. This is one. We are not claiming this is ASI or proof of ASI. We are claiming it is a credible scaffold for that research direction.

The seed is planted. The code is on disk. Fork it, stress-test it, break it honestly, and tell us what you find.

---

## Acknowledgments

This work stands on decades of cognitive science (Baars, Dehaene, Friston, Graziano, Tononi), neuroevolution (Stanley, Miikkulainen), distillation (Hinton, Vinyals, Dean), calibration (Hendrycks, Guo), and alignment (Amodei, Christiano, Russell, Anthropic's Constitutional AI team). The open-source projects it leans on at runtime (Ollama, `llama.cpp`, faster-whisper, Silero VAD, openWakeWord, Kokoro TTS, PyTorch, FastAPI, ONNX Runtime, sqlite-vec) and their maintainers made this system possible. The Qwen team at Alibaba built the Qwen3 and Qwen3-Coder-Next families we use as bootstrap teacher and CPU code generator. Any errors, overclaims, or omissions in this paper are mine.

---

## References (selected)

Citations below are compact and are meant to orient the reader, not to exhaustively cover the field.

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Dehaene, S., Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2).
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11.
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Tononi, G. (2008). Integrated information theory. *Biological Bulletin*, 215(3).
- Stanley, K. O., Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2).
- Real, E., Aggarwal, A., Huang, Y., Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *AAAI*.
- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Hendrycks, D., Gimpel, K. (2016). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *arXiv:1610.02136*.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., Mané, D. (2016). Concrete problems in AI safety. *arXiv:1606.06565*.
- Russell, S. (2019). *Human Compatible*. Viking.
- Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: harmlessness from AI feedback. *arXiv:2212.08073*.
- Sutton, R. (2019). The bitter lesson. *Incomplete Ideas*.
- Sutton, R., Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## Appendix A: Supplementary Evidence Index

Every artifact lives as a Markdown file under `docs/validation_reports/` in the release repository. Each one is dated and self-contained.

| Artifact | Covers |
| --- | --- |
| `docs/archive/pre_reset_report_phase9_complete.md` | Attested 21-day run (sha256 `020ed8b9...`) |
| `pre_reset_truth_pass-2026-04-23.md` | Phase 1 consolidated truth pass (P1.1 to P1.10) |
| `continuity_baseline_2026-04-23.md` | Continuity pivot with backup verification |
| `phase_7_isolated_subprocess_runtime_proof-2026-04-23.md` | Phase 7 plugin isolation live proof (P2.1) |
| `phase_2_2_synthetic_route_coverage-2026-04-23.md` | Synthetic truth-boundary sweep (P2.2) |
| `phase_5_proof_chain-2026-04-23.md` | Phase 5 self-improvement proof-chain traceability (P2.3) |
| `phase_6_5_live_evidence-2026-04-23.md` | Phase 6.5 initial live-brain ceremony |
| `phase_6_5_continuity_reverify-2026-04-23.md` | Phase 6.5 invariant re-verification across 5 restarts (P2.4) |
| `export_readiness_audit-2026-04-24.md` | Code export surface audit (P2.5) |
| `dashboard_prose_refresh-2026-04-24.md` | Dashboard language audit (P2.6) |
| `launch_day_verification-2026-04-24.md` | Consolidated launch-day release-ready certificate (P2.7) |

---

## Appendix B: Public API surface (abbreviated)

| Endpoint | Purpose |
| --- | --- |
| `GET /api/self-test` | Overall live health summary |
| `GET /api/full-snapshot` | Every subsystem state in one response |
| `GET /api/meta/status-markers` | SHIPPED / PARTIAL / PRE-MATURE / DEFERRED per marker |
| `GET /api/meta/build-status` | Build metadata, shipped changelog pointer |
| `GET /api/autonomy/level` | Six-field L3 evidence model (current, attested, request, activation) |
| `GET /api/autonomy/audit` | Append-only audit ledger entries |
| `GET /api/autonomy/attestation` | Hash-verified attestation records |
| `GET /api/plugins` | Plugin registry with `execution_mode`, `venv_ready`, `subprocess_count` |
| `GET /api/memory/*` | Memory graph queries (read-only surface) |
| `GET /api/epistemic/*` | Belief graph, calibration, soul integrity |
| `GET /api/oracle/*` | Oracle Benchmark v1.1 live score and per-domain breakdown |

---

## Appendix C: Repository layout (abbreviated)

```
brain/              hundreds of Python modules across cognition, perception, dashboard, eval, and tools
  consciousness/    kernel, modes, mutations, dreams, thoughts, operations
  perception/       audio/vision, identity fusion, scene tracking
  memory/           storage, search, cortex NNs, fractal recall, CueGate
  reasoning/        LLM clients, tool routing, bounded response, TTS
  policy/           neural policy, shadow eval, promotion, telemetry
  hemisphere/       NeuralArchitect, evolution, distillation, gap detection
  epistemic/        L5-L10 (beliefs, calibration, belief graph, quarantine, audit, soul)
  identity/         L3 boundary engine, scoping, name validation
  autonomy/         7 drives, scorer, deltas, policy memory, interventions
  cognition/        world model, causal engine, mental simulator
  goals/            goal continuity: lifecycle, dispatch, alignment
  skills/           capability gate, skill registry, learning jobs
  library/          document store, Blue Diamonds archive
  personality/      trait evolution, proactive, curiosity
  self_improve/     scanner, shim to codegen
  acquisition/      capability pipeline, 11-lane plugin path, plan evaluator
  codegen/          CoderServer, sandbox
  tools/            time, system, vision, memory, introspection, plugins
  dashboard/        FastAPI + WebSocket, 40+ panels, 7 pages
  jarvis_eval/      PVL (114 contracts across 23 groups), Oracle Benchmark v1.1
  synthetic/        perception, claim, commitment, and skill-acquisition exercise harnesses
  tests/            200+ test files plus historical attested release-suite counts above
pi/                 Raspberry Pi 5 sensing client
docs/               system reference, roadmap, build history, validation
LICENSE.md          AGPLv3 + commercial dual license
AGENTS.md           AI-agent quick reference
README.md           top-level overview
```

---

*This paper describes a working system. It is not a proposal. Every claim in it is grounded in executable code, hash-verifiable evidence, or live dashboard state. That is the contribution.*
