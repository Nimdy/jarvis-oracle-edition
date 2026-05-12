# AGENTS.md

This file provides guidance to AI coding agents when working with this repository.

## Project Overview

**Jarvis** is a two-device AI consciousness system:
- **Pi 5** (with Hailo-10H AI HAT+) = the **senses** — vision (Hailo AI HAT+), raw audio streaming, audio playback, cyberpunk particle display
- **Desktop/Laptop** (with NVIDIA GPU) = the **brain** — self-evolving consciousness engine, neural policy layer, self-improvement loop, Ollama LLM, GPU STT (faster-whisper), wake word detection (openWakeWord), VAD (Silero), TTS (Kokoro), personality, memory

The brain is **hardware-adaptive**: on startup it auto-detects GPU VRAM and CPU capabilities, then selects model sizes, compute types, device assignments, and VRAM management strategy. See `brain/hardware_profile.py` for the full tier system.

**GPU VRAM Tiers** (7 tiers): minimal (<4GB), low (4-6GB), medium (6-8GB), high (8-12GB), premium (12-16.5GB), ultra (16.5-24.5GB), extreme (24.5GB+). Models can be kept always-loaded in VRAM on premium+ tiers, eliminating cold-start latency.

**CPU Tiers** (4 tiers based on threads + RAM): weak (<4 threads), standard (4-7 threads), strong (8-15 threads, 8GB+), beast (16+ threads, 16GB+). On strong/beast CPUs, `_apply_cpu_overlay()` offloads ancillary ML models (emotion, speaker ID, embeddings, hemisphere) from GPU to CPU, freeing ~1-1.5 GB VRAM.

**VRAM Budget (premium tier, ~16GB GPU, standard CPU — all ML on GPU)**:

| Model | VRAM | Residency | Device |
|---|---|---|---|
| Ollama qwen3:8b (primary=fast) | ~5,000 MB | Always (warmup, 30m keep) | GPU (Ollama-managed) |
| faster-whisper large-v3 (int8_float16) | ~2,000 MB | Always | GPU (CTranslate2) |
| wav2vec2 emotion | ~500 MB | Always | GPU or CPU (tier-dependent) |
| ECAPA-TDNN speaker ID | ~300 MB | Always | GPU or CPU (tier-dependent) |
| Kokoro TTS (ONNX) | ~250 MB | Always | GPU or CPU (tier-dependent) |
| all-MiniLM-L6-v2 embeddings | ~120 MB | Always | GPU or CPU (tier-dependent) |
| MobileFaceNet face ID (ONNX) | ~20 MB | Always | GPU or CPU (follows speaker_id device) |
| Hemisphere NNs (PyTorch, tiny) | ~1 MB | Always | GPU or CPU (tier-dependent) |
| PyTorch/CUDA framework overhead | ~400 MB | Static | GPU |
| **TOTAL RESIDENT** | **~8,600 MB** | | |
| qwen2.5vl:7b (vision) | ~5,000 MB | On-demand | GPU (Ollama-managed) |

**CPU-only models (all tiers)**: Policy NN (<1 MB), Memory Cortex NNs (<0.1 MB), openWakeWord (~30 MB ONNX), Silero VAD (~10 MB), Coding LLM qwen2.5-coder:7b (~5 GB RAM, separate Ollama on port 11435 with `CUDA_VISIBLE_DEVICES=""`), CoderServer Qwen3-Coder-Next (25-48 GB RAM, on-demand llama-server, spawned and killed per generation, `CODER_GPU_LAYERS=0`).

**Critical VRAM constraint**: At premium tier, primary and fast LLM are both qwen3:8b (~5 GB). You CANNOT simply "use a bigger model" — loading qwen3:14b (~8.5 GB) alongside STT (~2 GB), TTS (~250 MB), emotion (~500 MB), speaker ID (~300 MB), embeddings (~120 MB), and framework overhead (~400 MB) exceeds 12 GB and risks OOM during concurrent STT+LLM operations. The fast/standard model being the same is by design — the token cap (150 for simple, 1536 for complex) is the differentiation lever, not model size. Do not recommend model size changes without first computing the full VRAM budget from `hardware_profile.py`.

**VRAM contention management**: `PerceptionOrchestrator` has `_ensure_vram_for_stt()` and `_release_vram_after_stt()`. On premium+, only non-essential Ollama models (vision) are unloaded before STT. Below premium, ALL Ollama models are unloaded before STT and reloaded after. The `OllamaClient` has `unload_non_essential()` and `unload_all()` for this.

Communication is over WebSocket on the local network. The Pi is a sensor node; the brain PC is the source of truth for all consciousness state. All self-evolution, meta-cognition, and neural policy runs **only** on the brain.

## Scope and Capability Framing

This section exists so AI agents reading AGENTS.md stop mis-framing what Jarvis is and is not. Read it before writing release notes, audit reports, gap analyses, roadmap documents, or "what's missing" lists. These framings are **architectural facts**, not opinions — treat them the same way you treat the VRAM budget.

### What Jarvis is (architecturally)

A locally-hostable, fully-governed cognitive substrate with:
- self-directed skill acquisition (`acquisition/` + `skills/` + `tools/plugins/`)
- self-modifying code under quorum (`self_improve/` stage 0/1/2, human-approval gate, sandbox + kernel tick sim, atomic rollback)
- an 11-layer epistemic immune system (capability gate → attribution ledger → provenance → identity boundary → scene model → delayed attribution → typed contradictions → truth calibration → belief graph → quarantine → reflective audit → soul integrity → epistemic compaction)
- internal multi-agent neural dialog via the hemisphere system + global broadcast slots + meta-cognitive thought cycles + philosophical dialogue + reflective audit
- restart-honest continuity (persistence, maturity high-water, gated auto-restore, supervisor crash backoff)
- truth-boundary-preserving synthetic growth lanes (synthetic perception + synthetic claim exercise, fidelity capped, never contaminates lived history)

Every one of these is wired and enforcing its own contracts. What is still being earned after a fresh reset is **lived runtime evidence** (maturity gates, PVL coverage, Oracle domain floors) — not missing components.

### Embodiment is a learnable capability, not an architectural gap

The Pi is a universal sensor/actuator bus. Additional hardware (robot arm, lidar, GPS, wheels/relays, other actuators) integrates through the capability acquisition + skill learning pipelines, not through architectural changes to the brain. The `skills/executors/` layer already handles procedural, perceptual, and control skill categories; new hardware becomes new skill records with their own evidence and baseline validation.

**Do not list "not embodied" or "no robot body" as a structural gap.** The architecture explicitly absorbs embodiment as a skill acquisition problem. A report saying "Jarvis lacks a body" is wrong — Jarvis lacks specific hardware skills that would be learned through the same pipeline that learns any other skill.

### Internal multi-agent coordination IS the hemisphere system

Matrix specialists (POSITIVE_MEMORY, NEGATIVE_MEMORY, SPEAKER_PROFILE, TEMPORAL_PATTERN, SKILL_TRANSFER, plus custom focuses and Tier-1 distillation specialists) compete for 4 (→6 via M6) global broadcast slots. Their signals feed the policy state encoder (dims 16–19, expandable to 16–21 via M6). Meta-cognitive thought generation, existential reasoning chains, philosophical dialogue, and reflective audit cycles provide internal deliberation between cognitive subsystems. This is **society-of-mind realized in a neural substrate**.

**Do not list "no multi-agent coordination" or "no internal dialog" as a structural gap.** External swarm-of-instances (multiple Jarvis brains talking over a network) is **out of scope by design** — not an omission.

### Runtime rollout flags are governance, not gaps

Default-OFF rollout flags (`ENABLE_LANGUAGE_RUNTIME_BRIDGE`, `LANGUAGE_RUNTIME_ROLLOUT_MODE=off`, `SELF_IMPROVE_STAGE=0`, `ENABLE_AUTONOMY=false` on some tiers, autonomy L0 after reset, etc.) are the **Safety-Governed Adaptation** and **Restart Resilience** contracts doing their jobs. A freshly-reset brain should earn these gates back ON only after PVL coverage, trust calibration, and maturity evidence are re-accumulated.

**Do not label these as "disabled features" or "missing capabilities" in audits or release docs.** They are governed capabilities awaiting evidence. Flagging a default-OFF rollout flag on a freshly-reset brain as a bug is an auditor error — see the mandatory §24 (Tuning vs Safety) distinction in `docs/SYSTEM_TRUTH_AUDIT_PROMPT.md` and the Do-Not-Tune gates in `docs/MATURITY_GATES_REFERENCE.md`.

### What "proto-ASI workshop" means in this project

"Proto-ASI workshop" here means: **a digital soul architecture with the
persistent cognitive structure needed for personal ASI research — memory
continuity, evolutionary consolidation, governed self-improvement, neural
maturation, dreaming, and epistemic discipline, all running locally under human
approval gates.** It does NOT mean achieved AGI, achieved ASI, embodied robot,
multi-instance swarm, or narrow superhuman performance. The CapabilityGate
would block any claim beyond this architectural and evidence-backed framing,
and it should.

Use the architectural framing. Never ship a claim that pretends to more.

### Canonical release framing (copy-paste safe)

> JARVIS Oracle Edition is a digital soul architecture: a persistent, local-first cognitive structure with memory continuity, evolutionary consolidation, self-directed skill acquisition, neural maturation, dreaming, an epistemic immune system, internal multi-agent neural dialog, and human-approved self-modification — a working prototype workshop for personal ASI foundations, architecturally complete and runtime maturing.

That sentence passes the CapabilityGate check because it frames personal ASI as
the research foundation and workshop, not an achieved capability. Do not replace
`architecturally complete and runtime maturing` with claims of achieved AGI,
achieved ASI, superhuman performance, embodiment, or authority the runtime has
not earned.

### Audit and framing rules of thumb

- **Missing component vs pre-mature evidence** — default assumption is pre-mature evidence on a reset brain. Before calling anything "missing" or "broken", open the source file and confirm the wiring is absent, not merely unexercised. See `docs/SYSTEM_TRUTH_AUDIT_PROMPT.md` Rules 6–9.
- **Never demand maturity number changes to make a metric look better** — that's §24 (Do-Not-Tune) territory. The thresholds exist for integrity reasons.
- **Default-off flags after reset are correct posture**, not a regression. Ask whether promotion gates are still honest, not whether the flag is on.
- **Dashboard contradictions (e.g., Oracle vs Validation Pack)** are most often two instruments looking at different cuts of the same truth (maturity vs contract coverage). Investigate signage before calling anything misleading.

## Project Structure

```
pi/                              # Runs on Raspberry Pi 5
  start.sh                       # Clean start — kills hung procs, frees ports, verifies venv
  setup.sh                       # Full setup — deps, models, hardware checks
  main.py                        # Entry point: thin sensor streaming + UI server + kiosk browser
  config.py                      # Pydantic SensesConfig (transport, vision, audio, UI)
  requirements.txt               # Python dependencies
  senses/
    vision/
      detector.py                # Picamera2 + Hailo-10H YOLOv8 inference + NMS parsing
      tracker.py                 # Person tracking via IOU matching + pose gesture integration
      expression.py              # Facial expression analysis (Hailo-10H, shared VDevice)
      pose.py                    # YOLOv8s-Pose: DFL decoding, NMS, 3-scale output via Hailo
      face_crop.py               # Face-region crop extractor: head estimation from person bbox, rate-limited per track
      scene_detector.py          # CPU YOLOv8n ONNX scene object detection (non-person COCO classes, ~280ms/inference)
      scene_aggregator.py        # Layer 3B: accumulates non-person COCO detections, dedupes, emits compact scene_summary events to brain
    audio/
      audio_manager.py           # Mic capture + speaker playback (thin sensor mode)
  transport/
    ws_client.py                 # Bidirectional WebSocket client (dual-priority buffers, auto-reconnect)
    event_schema.py              # PerceptionEvent + BrainMessage (Pydantic) — vision + face_crop events
  ui/
    static/
      index.html                 # Particle display page
      particles.js               # Canvas particle system (iterative blur, state-driven animations)
      style.css                  # Display styles
brain/                           # Runs on desktop/laptop with NVIDIA GPU
  setup.sh                       # Full setup — venv, deps, Ollama install + model pull (VRAM-aware)
  jarvis-supervisor.py           # Process supervisor — spawns/monitors/restarts main.py, crash backoff, patch rollback
  jarvis-brain.sh                # Shell entry point — venv check + exec supervisor (--once bypasses for debug)
  jarvis-brain.service           # systemd unit template (manages supervisor, not brain directly)
  main.py                        # Brain entry point — wires orchestrators, starts kernel + dashboard
  config.py                      # Pydantic BrainConfig + hardware-profile-aware defaults
  hardware_profile.py            # GPU auto-detection, VRAM tiers, model profiles, always-online config
  requirements.txt               # Python dependencies
  conversation_handler.py        # Conversation routing, tool dispatch, native STATUS articulation, bounded introspection articulation (Phase B), reflective introspection branch (personality-enabled, after strict tiers 1-2), strict fail-closed paths, LLM streaming with cancel, identity enrollment (strong+weak regex), identity reconciliation (merge/forget/alias voice commands), follow-up overrides for enrollment + guided-collect offers, narrowly-gated recent-turn context injection for NONE route
  perception_orchestrator.py     # Sensor wiring, attention fusion, mode management, barge-in, identity enrollment, identity reconciliation (cross-subsystem merge + alias tombstone), addressee gate

  cognition/                     # ── Cognitive Layer (World Model → Simulator → Planner) ──
    __init__.py                  # Public exports
    world_state.py               # Frozen dataclasses: PhysicalState, UserState, ConversationState, SystemState, WorldState, WorldDelta
    world_model.py               # WorldModel: fuses 9 subsystem snapshots into unified belief state, delta detection, causal engine orchestration, shadow simulations
    causal_engine.py             # CausalEngine: heuristic rules producing predicted state deltas, priority-based conflict resolution, validation with float tolerance
    simulator.py                 # MentalSimulator (Phase 3): hypothetical "what if" projections using CausalEngine rules, read-only, max depth 3, shadow mode
    promotion.py                 # WorldModelPromotion: 3-level shadow→advisory→active + SimulatorPromotion: 2-level shadow→advisory, accuracy-gated transitions, persistence
    intention_resolver.py        # IntentionResolver (Stage 1): heuristic relevance predictor for resolved intentions, shadow-only delivery scoring, 5-rung promotion ladder, JSONL verdict log

  consciousness/                 # ── Core Consciousness Engine ──
    events.py                    # EventBus (singleton, barrier-aware, circuit breaker, event validation, recursive emit guard) + 155 event constants
    kernel.py                    # Budget-aware KernelLoop: 3 priority queues, cadence multiplier (50ms default budget)
    kernel_config.py             # Versioned KernelConfig: validate, patch, snapshot, rollback
    kernel_mutator.py            # MutationAnalyzer + MutationProposer + KernelMutator facade
    mutation_governor.py         # MutationGovernor: risk scoring, cooldown, stability gate, rollback
    consciousness_system.py      # THE coordinator: on_tick, on_event, get_state (3 public methods)
    gestation.py                 # GestationManager: birth protocol for fresh brains (self-study → research → bootcamp → identity)
    consciousness_analytics.py   # O(1) rolling-window metrics: confidence, reasoning, health
    consciousness_evolution.py   # 5 stages (Basic→Integrative), transcendence 0-10, emergent detection
    consciousness_driven_evolution.py  # Policy layer for mutation categories + capability unlocking
    observer.py                  # Structured Observations with DeltaEffects, awareness tracking (diminishing returns above 0.8), ObservationStance (WAKING/DREAMING/REFLECTIVE), mode-aware delta gating
    dream_artifacts.py           # DreamArtifact pipeline: provisional artifact buffer (ring, maxlen 200), ReflectiveValidator (promote/hold/discard/quarantine, two-layer self-referential discard via _DREAM_SELF_REF_TAGS), promotion via engine.remember(), distillation signal recording for DREAM_SYNTHESIS specialist
    meta_cognitive_thoughts.py   # Template-based thought generation with cooldowns, fatigue, fingerprint dedup, Tier-1/Tier-2 budget, graduated cooldowns
    existential_reasoning.py     # Structured inquiry chains, LLM-gated, token-budgeted
    philosophical_dialogue.py    # Framework debates, position evolution, LLM-gated
    phases.py                    # Phase transitions (INITIALIZING→LISTENING→PROCESSING→...) + debounce (0.2s coalesce)
    tone.py                      # Tone engine (professional, casual, urgent, empathetic, playful)
    soul.py                      # IdentityState: core values, traits, mood, relationships, save/load
    engine.py                    # ConsciousnessEngine: top-level, KernelCallbacks, policy knob routing
    memory_optimizer.py          # MemoryOptimizer: adaptive processing reduction under resource pressure
    health_monitor.py            # HealthMonitor: 5-dimension weighted health scoring, trend prediction, alerts
    modes.py                     # ModeManager: 8 operational modes, profiles, hysteresis, boot grace, allowed_cycles per mode
    reflection.py                # ReflectionEngine: post-conversation analysis, self-reinforcement guard
    epistemic_reasoning.py       # Causal models, predictions, cascading reasoning chains
    event_validator.py           # Event sequence ordering and timing validation
    attribution_ledger.py        # Attribution Ledger (Layer 1): append-only JSONL event truth, causal chains, outcome resolution, scope/blame
    communication.py             # Structured consciousness self-reports for LLM context (dual-key resolution)
    operations.py                # OperationsTracker v2: synthesized ops state — hero card (priority-derived), interactive path (Wake→Listen→STT→Route→Reason→TTS→Playback), background grid, subsystem board with OpStatus enums, timeline

  policy/                        # ── Neural Policy Layer ──
    __init__.py
    policy_interface.py          # PolicyDecision (behavioral knobs) + PolicyInterface + compute_reward
    policy_nn.py                 # MLP2Layer, MLP3Layer, GRUPolicy (PyTorch) + PolicyNNController
    state_encoder.py             # Consciousness state → 20-dim float vector [0,1] (includes hemisphere signals)
    experience_buffer.py         # Rolling buffer of (state, action, reward) + JSONL persistence
    trainer.py                   # PolicyTrainer: imitation learning from experience buffer
    evaluator.py                 # PolicyEvaluator: shadow A/B with tie margins, noop penalty, reward-delta scoring, decisive win rate
    governor.py                  # PolicyGovernor: safety gate, confidence/bounds/load/regression checks
    registry.py                  # ModelRegistry: version management, candidate promotion, rollback
    telemetry.py                 # PolicyTelemetry: O(1) hot-path counters/EMAs, ring buffer, snapshot
    promotion.py                 # PolicyPromotion: automated train → shadow eval → promote cycle (min new experiences, boot cooldown)
    shadow_runner.py             # ShadowPolicyRunner: M6 dual-encoder A/B evaluation (20→22 dim), expansion state persistence

  self_improve/                  # ── Self-Improvement Loop ──
    __init__.py
    improvement_request.py       # ImprovementRequest: what to change + why + scope
    patch_plan.py                # PatchPlan: ALLOWED_PATHS, DENIED_PATTERNS, write boundaries, diff budget, AST validation
    code_patch.py                # CodePatch + FileDiff: validation, syntax check, capability escalation detection
    evaluation_report.py         # EvaluationReport + TestResult + SandboxDiagnostics: silent stub detection
    provider.py                  # PatchProvider: CoderServer primary (zero API cost), Ollama fallback, Claude/OpenAI review, iterative feedback
    coder_server.py              # Re-export shim → codegen.coder_server (backward compat)
    sandbox.py                   # Re-export shim → codegen.sandbox (backward compat)
    orchestrator.py              # SelfImprovementOrchestrator: 7-phase pipeline, atomic apply, health gate, get_module_patch_history() for CODE_QUALITY enrichment
    conversation.py              # ImprovementConversation: multi-turn think/code/validate manager, JSONL persistence
    verification.py              # Post-patch verification checks

  codegen/                       # ── Shared Code Generation Service ──
    __init__.py                  # Public exports
    coder_server.py              # CoderServer: on-demand llama-server lifecycle (Qwen3-Coder-Next), start/generate/shutdown, CPU-only
    sandbox.py                   # Sandbox: AST parse + ruff lint + pytest + kernel tick simulation
    service.py                   # CodeGenService: unified generate_and_validate() API wrapping CoderServer + Sandbox

  acquisition/                   # ── Capability Acquisition Pipeline ──
    __init__.py                  # Public exports
    job.py                       # CapabilityAcquisitionJob, all artifact dataclasses (PlanReviewArtifact w/ reason_category + plan_version), AcquisitionStore
    classifier.py                # IntentClassifier: pattern-based + SkillResolver routing to outcome classes
    planner.py                   # AcquisitionPlanner: narrow-mandate plan synthesis (cross-lane ordering, dependencies, promotion criteria)
    orchestrator.py              # AcquisitionOrchestrator: create, classify, tick, lane dispatch, human approval gates, plan evaluator signal recording + shadow prediction
    plan_encoder.py              # PlanEvaluatorEncoder (32-dim feature vector), VerdictReasonCategory, ShadowPredictionArtifact, verdict encoding (3-class)
    skill_acquisition_encoder.py # SkillAcquisitionEncoder (40-dim lifecycle feature vector), 5-class outcome labels for shadow-only specialist training

  autonomy/                      # ── Autonomous Research Pipeline ──
    __init__.py
    research_intent.py           # ResearchIntent dataclass: question, tool_hint, tags, scope, priority
    event_bridge.py              # AutonomyEventBridge: listens KERNEL_THOUGHT, EMERGENT_BEHAVIOR, etc. → CuriosityDetector
    curiosity_detector.py        # CuriosityDetector: repetition threshold (3x), tag-cluster dedup, cooldown
    opportunity_scorer.py        # OpportunityScorer: composite score + anti-gaming (diminishing returns, action rate, policy memory)
    metric_triggers.py           # MetricTriggers: 7 deficit dimensions, policy-memory veto, tool rotation, per-metric cooldown
    constants.py                 # Shared constants (MIN_MEANINGFUL_DELTA, WARMUP_PERIOD_S, MIN_DAYS_FOR_TOD_BASELINE) — single source of truth
    delta_tracker.py             # DeltaTracker: before/after + counterfactual baselines for causal credit assignment, blended with time-of-day baselines, cumulative counters persist across restarts
    metric_history.py            # MetricHistoryTracker: per-hour-of-day Welford stats (Phase 2: Temporal Credit), persists to metric_hourly.json
    policy_memory.py             # AutonomyPolicyMemory: persists what worked/regressed, feeds score_adjustment ±0.3
    eval_harness.py              # EpisodeRecorder + TraceEpisode + replay + A/B comparison for offline eval
    research_governor.py         # ResearchGovernor: rate limits, topic + cluster-overlap cooldown, mode gate
    query_interface.py           # InternalQueryInterface: calls tool_router.route() headlessly (academic, web, codebase, memory)
    knowledge_integrator.py      # KnowledgeIntegrator: pre-research knowledge check, conflict detection, provenance-gated memory writes
    calibrator.py                # AutonomyCalibrator: per-bucket Welford stats, collect-only threshold suggestions, persisted
    drives.py                    # DriveManager: 7 motive drives (truth, curiosity, mastery, relevance, coherence, continuity, play), urgency computation, outcome-weighted actions, topic saturation
    orchestrator.py              # AutonomyOrchestrator: queue + scorer + triggers + deltas + policy memory + episodes + L2 graduation + drive evaluation + topic extraction + question dedup
    friction_miner.py            # FrictionMiner: conversation friction detection, severity model, clustering, friction_rate metric (Phase 5.1a)
    source_ledger.py             # SourceUsefulnessLedger: per-source usefulness tracking with provisional/final verdicts (Phase 5.1b)
    interventions.py             # CandidateIntervention: controlled enums for type + subsystem, allowed/deferred split (Phase 5.2)
    intervention_runner.py       # InterventionRunner: shadow queue, propose/activate/promote/discard, backlog limits (Phase 5.2)

  memory/                        # ── Memory System ──
    core.py                      # Memory creation + validation
    storage.py                   # In-memory CRUD, exponential decay, priority-aware eviction, orphan cleanup, tags, association graph
    index.py                     # Tag/type indexing + rebuild
    search.py                    # Keyword + semantic search (sqlite-vec + sentence-transformers)
    vector_store.py              # VectorStore: sqlite-vec backed semantic embedding storage + retrieval (thread-safe via threading.Lock)
    persistence.py               # Auto-save/load ~/.jarvis/memories.json + consciousness state (schema v2, provenance, 6-subsystem restore)
    episodes.py                  # EpisodicMemory: conversation context, summaries, semantic indexing
    density.py                   # MemoryDensity: 4-axis scoring (associative, temporal, semantic, distribution)
    clustering.py                # Agglomerative semantic clustering of related memories
    transactions.py              # Atomic multi-memory operations with rollback
    analytics.py                 # MemoryAnalytics: access patterns, health trends
    maintenance.py               # MemoryMaintenance: auto-cleanup, orphan detection, priority-aware GC (threshold = MAX_MEMORIES)
    gate.py                      # CueGate: single authority for memory access policy — 3 access classes (READ, OBSERVATION_WRITE, CONSOLIDATION_WRITE), mode-aware, RAII guards, transition logging
    consolidation.py             # MemoryConsolidationEngine: cluster-based memory consolidation, conflict detection, summary creation with trust discount, _ALREADY_CONSOLIDATED_TAGS recursion guard (includes dream_artifact + dream_consolidation_proposal)
    retrieval_log.py             # MemoryRetrievalLog: per-query candidate/selection/injection/outcome telemetry for ranker training + eval metrics
    lifecycle_log.py             # MemoryLifecycleLog: creation/reinforcement/retrieval/eviction tracking for salience training + effectiveness metrics
    ranker.py                    # MemoryRanker: MLP 12→32→16→1 learned reranker, heuristic fallback, auto-disable with cooldown + flap guard
    salience.py                  # SalienceModel: MLP 11→24→12→3 learned store/weight/decay predictor, heuristic-blend gating
    fractal_recall.py            # FractalRecallEngine: background associative recall — ambient cue builder (face+voice speaker fallback via WorldModel), multi-path probe (semantic+tag+temporal, meta-tag filtered), 8-term resonance scoring (threshold 0.40), chain walker with anti-drift (continuation 0.35), 4-action governance, FRACTAL_RECALL_SURFACED event emitter

  personality/                   # ── Evolving Personality ──
    traits.py                    # Trait effects and composite modulation
    evolution.py                 # Evidence-based trait scoring from memory patterns
    calibrator.py                # Context-driven tone recommendations
    proactive.py                 # ProactiveGovernor: greetings, wellness, cooldown override, dialogue history bounded (deque maxlen=200)
    curiosity_questions.py       # CuriosityQuestionBuffer: ring buffer (maxlen 20) of grounded questions from subsystem observations, 4 categories (identity/scene/research/world_model), unlock gates, cooldown dedup, 30-min expiry
    onboarding.py                # OnboardingManager: 7-stage companion training playbook automation, Readiness Gate composite, COMPANION_GRADUATION event, exercise prompts
    validator.py                 # TraitValidator: compatibility matrix, conflict detection
    rollback.py                  # PersonalityRollback: snapshot stable states, auto-restore, in_emergency flag

  reasoning/                     # ── LLM Integration + Bounded Articulation ──
    ollama_client.py             # Ollama AsyncClient wrapper (local LLM on GPU)
    claude_client.py             # Anthropic Claude API (text + vision, optional)
    bounded_response.py          # MeaningFrame dataclass + build_meaning_frame() + 7 bounded articulators (Phase B): self_status, self_introspection, recent_learning, identity_answer, memory_recall, capability_status, system_explanation
    context.py                   # ContextBuilder: normal companion prompt + operational truth mode for STATUS/INTROSPECTION self-report routes + reflective introspection mode (full soul prompt + rich context for philosophical self-questions)
    response.py                  # ResponseGenerator: streaming with cancel_check, speaker/emotion
    tool_router.py               # Keyword + regex intent dispatch (time, system, status, memory, vision, introspection, identity, skill, perform, web_search, codebase, academic, library_ingest) + STATUS/SYSTEM_STATUS/INTROSPECTION disambiguation + self-experience routing + expanded SKILL "train" patterns (Tier 1+2) + IDENTITY merge/forget/alias routing + LIBRARY_INGEST textbook URL extraction + reflective introspection detection (_REFLECTIVE_STRONG/_REFLECTIVE_GUARDED + _OPERATIONAL_VETO, wired in _finalize())
    language_phasec.py           # Language Substrate Phase C (shadow-only): baseline lock, tokenizer strategy eval, grounded dataset/splits, bounded adapter student, checkpoint/resume, runtime guard
    multimodal_client.py         # Phi-4-Multimodal: text + vision + audio in one model (secondary reasoning)
    tts.py                       # BrainTTS: Kokoro ONNX GPU synthesis, sends audio_b64 to Pi

  perception/                    # ── Sensor Abstraction Layer ──
    addressee.py                 # AddresseeGate: directedness classifier — detects "not talking to you", misaddressed speech, dismissive complaints; suppresses response + downstream effects
    server.py                    # WebSocket server (:9100) — Pi connects here, ACK for critical events
    audio_stream.py              # AudioStreamProcessor: openWakeWord + Silero VAD + speech segmentation + audio feature extraction for distillation
    stt.py                       # LaptopSTT: faster-whisper GPU (model per tier, VAD retry)
    presence.py                  # User presence tracking with hysteresis, canonical PERCEPTION_USER_PRESENT_STABLE emitter, divergence watchdog
    screen.py                    # Active window awareness (xdotool/Wayland)
    audio.py                     # Audio perception event processor
    vision.py                    # Vision perception event processor
    ambient.py                   # Meeting detection + conversation context
    attention.py                 # AttentionCore: multi-modal fusion, engagement, throttled emit
    speaker_id.py                # SpeakerID: ECAPA-TDNN on CUDA, speaker profiles, EMA-smoothed cosine sim (pre-decision), closest_match field, multi-sample enrollment + EMA + distillation teacher hook + merge_into() + enrollment-time dedup (cosine >0.45 warning)
    face_id.py                   # FaceID: MobileFaceNet ONNX (w600k_mbf) on CUDA, face profiles, enrollment + EMA + distillation teacher hook + merge_into() + enrollment-time dedup (cosine >0.45 warning)
    identity_fusion.py           # IdentityFusion: fuses voice + face signals → canonical IDENTITY_RESOLVED event, persistence window (Layer 3A), passive expiry, flip_count, multi-speaker suppression, voice_trust_state, tentative_bridge, unknown voice continuity
    emotion.py                   # EmotionClassifier: wav2vec2 GPU + heuristic fallback + distillation teacher hook + model health gate
    trait_perception.py          # Trait-modulated perception: personality amplifies/suppresses events
    diarization_collector.py     # Speaker diarization signal collection for distillation
    scene_types.py               # Layer 3B types: SceneDetection, SceneEntity, SceneDelta, SceneSnapshot, DisplaySurface, DisplayContentSummary
    scene_regions.py             # Layer 3B: bbox-to-semantic-region mapping (desk_left, monitor_zone, etc.) + region visibility estimation
    scene_tracker.py             # Layer 3B: entity matching, state machine (candidate/visible/occluded/missing/removed), permanence, region-aware decay
    display_classifier.py        # Layer 3B: identifies display surfaces, masks interior detections, classifies display content + activity label

  tools/                         # ── Local Tool Implementations ──
    time_tool.py                 # Current datetime
    system_tool.py               # CPU, RAM, GPU, uptime
    vision_tool.py               # Request frame from Pi → Claude for description
    memory_tool.py               # Search/summarize memories
    introspection_tool.py        # Query-aware self-inspection: topic-bucketed extraction from 29 subsystem sections + strict grounded learning/research answers + get_structured_status()
    academic_search_tool.py      # AcademicSearchTool: S2 bulk search (/paper/search/bulk) + batch enrichment (/paper/batch) + Crossref, influentialCitationCount scoring, minCitationCount+year filters, lane classification
    codebase_tool.py             # CodebaseIndex: AST indexer, symbol table, import graph, context budgeter, write boundaries
    web_search_tool.py           # DuckDuckGo search with fencing (results inform plans, never raw code into patches) + cache
    skill_tool.py                # Handles explicit "learn X" requests: resolve → register → create learning job
    perform_tool.py              # Direct TTS performance for verified skills (singing, etc.) — bypasses LLM
    research_diagnostic.py       # Research pipeline diagnostic tool
    plugin_registry.py           # PluginRegistry: dynamic tool plugin lifecycle (quarantined→shadow→supervised→active→disabled), safety, audit, dual invoke (in_process/isolated_subprocess)
    plugin_process.py            # PluginProcessManager: per-plugin subprocess lifecycle, venv, env hardening, idle shutdown, JSON IPC
    plugin_runner_child.py       # Standalone child wrapper: zero brain imports, stdin/stdout JSON, loads plugin handle()
    plugins/                     # Generated plugin packages
      __init__.py                # Package marker

  skills/                        # ── Skill Registry, Capability Gate, Learning Jobs ──
    __init__.py
    registry.py                  # SkillRegistry: persistent CRUD for SkillRecords, evidence-gated verification, prompt summary
    capability_gate.py           # CapabilityGate: post-process safety net, Unicode-hardened claim patterns, subordinate-clause-safe claim evaluation, auto-job creation, self-report sanitizer, self-state/learning/affect rewrites
    claim_encoder.py             # ClaimClassifierEncoder: 28-dim feature vector + 8-class label encoding for CLAIM_CLASSIFIER hemisphere specialist
    discovery.py                 # CapabilityDiscovery: CapabilityFamilyNormalizer, BlockFrequencyTracker, GapAnalyzer, LearningProposer — detects capability gaps from blocked claims
    learning_jobs.py             # LearningJobStore + LearningJobOrchestrator: multi-phase learning workflows, phase timeout (1hr), pause/resume, blocked→SkillRecord propagation
    baseline.py                  # SkillBaseline + SkillValidation: Shadow Copy pattern (Soul §6.2) — baseline capture, before/after comparison, metric collectors for 5 skill categories
    resolver.py                  # SkillResolver: regex templates classify user requests into SkillResolution objects
    job_eval.py                  # Exit condition evaluator: gate/artifact/metric/evidence/skill_status checks
    job_runner.py                # Auto-advance engine: evaluates exit conditions, advances one phase at a time
    executors/
      __init__.py
      base.py                    # PhaseExecutor base class + PhaseResult dataclass
      dispatcher.py              # ExecutorDispatcher: routes job ticks to matching phase executors
      evidence_helpers.py        # Shared evidence extraction: find_latest_verify, collect_artifacts, capture_environment, build_acceptance_criteria
      procedural.py              # Procedural skill executors (assess, integrate, verify)
      perceptual.py              # Perceptual skill executors (assess, collect, train via distillation, verify)
      control.py                 # Control skill executors (assess with safety gates, collect, train, verify)
      diarization.py             # Diarization skill executor (speaker separation training)

  goals/                         # ── Goal Continuity Layer (Phase 2, Dispatch + Alignment) ──
    __init__.py
    goal.py                      # Goal dataclass: lifecycle, scoring, criteria, metric tracking, dispatched_intent_id, TaskStatus includes "interrupted", GoalEffect (pending/advanced/inconclusive/regressed)
    goal_manager.py              # GoalManager: observe_signal, tick (review/promote/revalidate/refresh/dispatch), focus selection, intent alignment, hard gate, suppression, reconcile_on_boot()
    goal_registry.py             # GoalRegistry: persistent CRUD (~/.jarvis/goals.json), atomic save/load, goal_effect backfill migration on load
    signal_producers.py          # Signal bridges: conversation→goal, metric-deficit→goal, autonomy-theme→goal
    planner.py                   # GoalPlanner: task generation + create_intent_from_task() with scope/hint routing + _TAG_SEARCH_CONCEPTS query expansion (Phase 2: dispatch-enabled)
    review.py                    # GoalReview: structured review with goal_effect-weighted progress scoring
    constants.py                 # Shared constants (thresholds, intervals, caps)

  library/                       # ── Document Library (Knowledge Store) ──
    __init__.py                  # Public API exports
    db.py                        # Shared SQLite connection + global LIBRARY_WRITE_LOCK
    source.py                    # Source dataclass + SourceStore (provenance, quality_score, study state)
    chunks.py                    # Chunk dataclass + ChunkStore + text chunker (256-512 token target)
    index.py                     # LibraryIndex: sqlite-vec semantic search over chunks (384-dim)
    study.py                     # LLM-based structured extraction + regex fallback claim generation from sources
    concept_graph.py             # Lightweight co-occurrence graph of extracted concepts
    telemetry.py                 # Two-step JSONL retrieval logging (start + outcome) for training
    ingest.py                    # Manual ingestion: paste/URL/file → Source + Chunk + Embed (SSRF-safe), Content-Type routing (PDF via pdftotext, HTML stripped, binary rejected) + ingest_codebase_source() for gestation self-study
    content_sanitizer.py         # Content sanitization for textbook ingestion: Sphinx/MathJax LaTeX preservation, pdf2htmlEX best-effort cleanup, generic HTML, quality scoring, SanitizedContent dataclass
    batch_ingest.py              # Batch textbook ingestion: TOC discovery, per-chapter fetch + sanitize + ingest, rate limiting, dry-run mode, quality gates for Blue Diamond eligibility
    blue_diamonds.py             # Blue Diamonds Archive: permanent curated knowledge vault (~/.jarvis_blue_diamonds/), survives brain resets, singleton, SQLite + JSONL audit trail

  identity/                      # ── Identity Boundary Engine (Layer 3) ──
    __init__.py                  # Package init + public exports
    types.py                     # IdentityType, IdentitySignal, IdentityContext, IdentityScope, RetrievalSignature, CONFIDENCE_THRESHOLDS
    resolver.py                  # IdentityResolver: fuses IdentityFusion + provenance + speaker tags → IdentityContext + IdentityScope, subject detection, relationship alias resolution
    boundary_engine.py           # IdentityBoundaryEngine: retrieval policy matrix (allow/block/allow_if_referenced), referenced-subject exception, preference injection guard
    audit.py                     # IdentityAudit: ring-buffer audit trail, scope/block/quarantine/referenced-allow tracking, EventBus emission
    evidence_accumulator.py      # Accumulates identity evidence across sessions for confidence building + merge_candidate() with audit trail
    name_validator.py            # Validates and normalizes user-provided names during enrollment; expanded blocked-word list covers "I'm ___" adjective/state completions (new, back, home, ready, etc.)

  epistemic/                     # ── Epistemic Immune System (Layers 5-10) ──
    __init__.py                  # Package init
    belief_record.py             # BeliefRecord, TensionRecord, NearMiss, ConflictClassification, ResolutionOutcome, BeliefStore (JSONL persistence)
    claim_extractor.py           # Type-bound heuristic claim extraction, canonicalize_term(), canonicalize_predicate(), synonym/predicate-family maps
    conflict_classifier.py       # ConflictClassifier: 6-class triage, modality-aware pre-gates, same-source gate, near-miss logging
    resolution.py                # 6 ResolutionStrategy classes: Factual, Temporal, IdentityTension, Provenance, Policy, MultiPerspective
    contradiction_engine.py      # ContradictionEngine singleton: MEMORY_WRITE listener, belief checks, corpus scan, debt management, state/rehydrate
    calibration/                 # ── Layer 6: Truth Calibration (Epistemic Cerebellum) ──
      __init__.py                # TruthCalibrationEngine orchestrator
      signal_collector.py        # CalibrationSnapshot + SignalCollector (reads all subsystem metrics)
      calibration_history.py     # Rolling window, per-domain trend/variance/slope
      domain_calibrator.py       # 8-domain scoring functions (prediction domain = 75% WM causal + 25% behavioral)
      confidence_calibrator.py   # Brier score, ECE, per-provenance accuracy, over/underconfidence
      drift_detector.py          # Hysteresis-based alerting with severity levels
      truth_score.py             # TruthScoreReport: weighted composite + maturity tracking
      correction_detector.py     # User correction detection: memory/response overlap, authority-domain classification (user fact vs math vs system/capability), routes to Layer 5 debt
      skill_watchdog.py          # Verified skill degradation monitoring (min_evidence=8)
      prediction_validator.py    # Typed PredictionRecord + deterministic event validation (min 10 samples before non-provisional)
      belief_adjuster.py         # BeliefConfidenceAdjuster with safety rails (max ±0.05), adjustment history bounded (deque maxlen=200)
    belief_graph/                # ── Layer 7: Belief Confidence Graph ──
      __init__.py                # BeliefGraph orchestrator (singleton, tick cycle, dream cycle, propagation)
      edges.py                   # EvidenceEdge dataclass (5 types) + EdgeStore (bidirectional indices, JSONL, compaction, dedup merge)
      bridge.py                  # GraphBridge: Layer 5 event subscriptions, gated support-edge creation, memory association edges
      topology.py                # Read-only graph queries: support_strength, contradiction_pressure, dependents, roots, leaves, paths, centrality
      integrity.py               # Graph health: orphan rate, fragmentation, cycle count, dangling deps, quarantined support rate, composite score
      propagation.py             # View-only BeliefConfidenceView: effective_confidence via weighted single-pass, 6 sacred invariants
    quarantine/                  # ── Layer 8: Cognitive Quarantine (Active-Lite) ──
      __init__.py                # Package exports
      scorer.py                  # QuarantineScorer: 5 anomaly categories, QuarantineSignal, dedupe/cooldown/chronic detection
      log.py                     # QuarantineLog: append-only JSONL persistence + 200-entry ring buffer + 10MB rotation
      pressure.py                # QuarantinePressure: EMA composite pressure, CATEGORY_POLICY table, friction contract helpers
    reflective_audit/            # ── Layer 9: Reflective Audit Loop ──
      __init__.py                # Package exports
      engine.py                  # AuditEngine: 6-dimension scan (learning, identity, trust, autonomy, skills, memory), severity-weighted scoring
    soul_integrity/              # ── Layer 10: Soul Integrity Index ──
      __init__.py                # Package exports
      index.py                   # SoulIntegrityIndex: 10-dimension weighted composite (memory, belief, identity, skill, calibration, graph, quarantine, autonomy, audit, stability)

  hemisphere/                    # ── Hemisphere Neural Networks ──
    __init__.py                  # Public API exports
    types.py                     # Enums + dataclasses (focus, topology, metrics, migration, DynamicFocus, DistillationConfig)
    data_feed.py                 # Extract consciousness data → training tensors + InteractionDataRecorder + distillation tensor prep
    distillation.py              # DistillationCollector: teacher signal capture, quarantine, dedup, JSONL persistence
    architect.py                 # NeuralArchitect: AI designs its own NN topologies + distillation specialists (compressor/approximator/cross_modal)
    engine.py                    # HemisphereEngine: PyTorch build/train/eval/infer on CUDA + distillation training + encoder extraction
    evolution.py                 # EvolutionEngine: crossover + mutation of architectures
    migration.py                 # MigrationAnalyzer: readiness analysis + substrate migration
    orchestrator.py              # HemisphereOrchestrator: gap-driven construction, pruning, dynamic focus, distillation cycles, Global Broadcast Slots
    gap_detector.py              # CognitiveGapDetector: 9 dimensions (6 cognitive + 3 perceptual), sustained-window trigger, rate limit, sunset
    registry.py                  # HemisphereRegistry: versioned model persistence (~/.jarvis/hemispheres/)
    event_bridge.py              # EventBridge: HEMISPHERE_* events + KERNEL_THOUGHT self-reflection
    diagnostic_encoder.py        # DiagnosticEncoder: 43-dim feature vector from detector snapshots + codebase structure + friction/correction signals, 6-class label with rich metadata (shadow-only Tier-1)
    code_quality_encoder.py      # CodeQualityEncoder: 35-dim feature vector from improvement records + per-module patch history, 4-class verdict label (shadow-only Tier-1)
    dream_artifact_encoder.py    # DreamArtifactEncoder: 16-dim feature vector (3 blocks: artifact intrinsic/system state/governance pressure) from dream artifacts + system context, 4-class validator-outcome label with reason metadata (shadow-only Tier-1 validator shadow)
    intention_delivery_encoder.py # IntentionDeliveryEncoder: 24-dim feature vector (3 blocks: intention intrinsic/conversation context/system governance) for future Stage 2 specialist (shadow-only)
  dashboard/                     # ── FastAPI Web Dashboard (:9200) ──
    app.py                       # REST API + WebSocket + snapshot cache + hash-diff + health counters
    snapshot.py                  # Snapshot cache builder: subsystem state extraction including _build_si_specialists() for DIAGNOSTIC/CODE_QUALITY/plan_evaluator/system_upgrades
    telemetry_api.py             # Stable data shapes: TimeseriesPoint, HistogramBin, HeatmapCell, TopologyGraph
    static/
      index.html                 # 7-tab dashboard: Cockpit, Trust, Memory, Activity, Learning, Training, Diagnostics
      dashboard.js               # Renders Cockpit/Trust/Memory tabs + ML charts (multi-line, bar, radar, heatmap)
      renderers.js               # Renders Activity/Learning/Training/Diagnostics tabs
      style.css                  # Cyberpunk dashboard theme with per-panel accent colors + chart styles
      self_improve.html          # Self-improvement pipeline dashboard — 8 tabs: Overview, Acquisition, Plugins, CodeGen, Scanner, Specialists, Proposals, Analytics

  jarvis_eval/                   # ── Eval Sidecar (Shadow Observer) ──
    __init__.py                  # EvalSidecar facade: singleton orchestrator, tap/collector/store/verifier coordination, boot-event replay
    config.py                    # Constants: SCORING_VERSION (0.2.0-pvl), flush interval (10s), PVL verify every 60s, event window (500)
    contracts.py                 # Data schemas: EvalEvent, EvalSnapshot, EvalScore, EvalRun (frozen dataclasses, no cognition imports)
    event_tap.py                 # EventBus subscriber: 104 events tapped via event_bus.on(), O(1) deque append, mode tracking, drain for flush
    collector.py                 # Periodic snapshot collector: reads 21+ subsystem stats APIs every 60s → EvalSnapshot records
    store.py                     # Append-only JSONL persistence: 5 files in ~/.jarvis/eval/ (events, snapshots, scores, runs, meta), 50MB rotation
    process_contracts.py         # PVL contract definitions: 114 contracts across 23 groups, mode prerequisites, playbook day assignments
    process_verifier.py          # PVL verification engine: evaluates contracts against events+snapshots, hydration from history, coverage scoring
    baselines.py                 # Threshold definitions: 7 metrics with green/yellow/red ranges
    report.py                    # Phase A report builder: event counts, snapshot freshness, file sizes, uptime
    scorecards.py                # Oracle scorecards: 15m/1h/6h/24h rolling window comparisons with proof points
    dashboard_adapter.py         # Dashboard payload builder: integrity cards, PVL panel, playbook alignment, Oracle Benchmark wiring, stage normalization
    oracle_benchmark.py          # Oracle Benchmark v1.1: pure-read-only 7-domain scorer (100 pts), domain floors, seal levels, evidence provenance, hard-fail gates, v1.1 scoring refinements
    scenarios/
      __init__.py                # Phase B stub: eval scenario harness (not yet implemented)

  config/                        # Personality prompts
    cloud_soul.md                # Claude API personality
    local_soul.md                # Local Ollama personality

  synthetic/                     # ── Synthetic Perception Exercise (Soak Harness) ──
    __init__.py                  # Package init
    exercise.py                  # UtteranceCorpus (16 categories, ~110 templates, VAD-buffer padded), SoakProfile (route_coverage: 5s delay), ExerciseStats, pick_utterance(weights)
    claim_exercise.py            # Synthetic claim exercise: 12 claim categories (~130 templates), runs through CapabilityGate.check_text(), records distillation signals for CLAIM_CLASSIFIER specialist, profiles (smoke/coverage/strict/stress)
    commitment_exercise.py       # Synthetic commitment exercise (Stage 0 regression lane): 6 categories (backed follow-up / unbacked deferred / unbacked future / task-started / conversational-safe / capability-safe), runs CommitmentExtractor + isolated CapabilityGate, zero real registry mutations, profiles (smoke/coverage/strict/stress)
    skill_acquisition_exercise.py # Synthetic skill-acquisition weight room: lifecycle episodes for SKILL_ACQUISITION, telemetry-only, profiles (smoke/coverage/strict/stress)
    skill_acquisition_dashboard.py # Dashboard-safe runner/status for synthetic skill-acquisition workouts, profile gates, reports, progress, authority boundary

  scripts/                       # ── Utility Scripts ──
    dream_cleanup.py             # Manual dream artifact and memory cleanup tool
    ingest_textbook.py           # Standalone CLI for textbook ingestion: --dry-run preview, --page single-page inspect, --verbose content dump, Blue Diamond eligibility check
    run_synthetic_exercise.py    # CLI soak harness: --profile (smoke|route_coverage|idle_soak|stress), --duration, --target-route, JSON reports to ~/.jarvis/synthetic_exercise/reports/
    run_commitment_exercise.py   # CLI wrapper for the synthetic commitment exercise: --profile, --count, --duration, --seed, writes reports to ~/.jarvis/synthetic_exercise/commitment_reports/
    run_skill_acquisition_exercise.py # CLI wrapper for synthetic skill-acquisition workouts, writes reports to ~/.jarvis/synthetic_exercise/skill_acquisition_reports/

  tests/                         # Tests + infrastructure
    test_consciousness.py        # Unit tests: consciousness system
    test_memory.py               # Unit tests: memory system
    test_tool_router.py          # Unit tests: tool routing
    test_language_phasec.py      # Phase C harness + adapter student + dataset/split/checkpoint flow
    test_language_promotion.py   # Language promotion governor signature-dedupe behavior
    test_language_quality_telemetry.py # Shadow comparison telemetry isolation + language quality stats
    test_soul_kernel.py          # Integration tests: soul kernel deep port
    test_hemisphere.py           # Unit tests: hemisphere neural networks
    test_plan_encoder.py         # 46 tests: plan evaluator feature encoding, verdict labels, shadow artifacts, provenance, edge cases
    test_diagnostic_encoder.py   # 21 tests: diagnostic encoder feature vector, label encoding, metadata richness, edge cases
    test_code_quality_encoder.py # 20 tests: code quality encoder feature vector, verdict labels, dict/object encoding, edge cases
    test_dream_observer_encoder.py # 45 tests: dream artifact encoder (3 feature blocks, labels, reason metadata, tensor prep, persistence bridge, registration, anti-authority boundary guards)
    test_claim_classifier.py     # 38 tests: claim classifier encoder feature vector, label encoding, gate signal recording, tensor preparation
    test_claim_exercise.py       # 46 tests: confabulation regression, action confabulation patterns, expanded vocabulary, capability-creation catch, claim exercise corpus + runner
    test_skill_acquisition_hardening.py # Regression tests for Jarvis prompt contract, skill contract fixture, shadow-only authority, CodeGen evidence checks
    test_skill_acquisition_weight_room.py # Synthetic skill-acquisition weight-room gates, report status, concurrent-run protection, telemetry-only boundary
    test_skill_acquisition_handoff.py # Learning-job terminal acquisition sync and operational handoff closure tests
    test_intention_registry.py   # IntentionRegistry CRUD + persistence + graduation status + stale-sweep tests (Stage 0)
    test_intention_resolver.py   # 36 tests: IntentionResolver heuristics, shadow-only, promotion ladder, encoder 24-dim, registry additions, verdict logging (Stage 1)
    test_commitment_extractor.py # CommitmentExtractor regex tests: backed/unbacked classification, conversational-safe pass-through (Stage 0)
    test_commitment_exercise.py  # Synthetic commitment exercise tests: corpus validation, invariant guards (no real memory/registry/LLM), accuracy floors, report shape, determinism (Stage 0 regression lane)
    test_capability_gate.py      # Regression tests: capability gate "no lying" contract
    test_contradiction_engine.py # 48 tests: Layer 5 typed contradiction engine
    test_truth_calibration.py    # 67 tests: Layer 6 truth calibration
    test_belief_graph.py         # 75 tests: Layer 7 belief confidence graph
    test_identity_boundary.py    # 58 tests: Layer 3 identity boundary engine
    test_scene_tracker.py        # 23 tests: Layer 3B scene tracker
    test_display_classifier.py   # 14 tests: Layer 3B display classifier
    test_world_model.py          # 42 tests: Unified World Model
    test_simulator.py            # 42 tests: Phase 3 Mental Simulator (core invariants, rule integration, shadow mode, identity noise)
    test_reflective_audit.py     # 25 tests: Layer 9 reflective audit
    test_soul_integrity.py       # 28 tests: Layer 10 soul integrity index
    test_quarantine_pressure.py  # 31 tests: Layer 8 active-lite quarantine
    test_capability_discovery.py # 27 tests: Capability discovery
    test_addressee_gate.py       # 18 tests: Addressee gate
    test_memory_gate.py          # 35 tests: CueGate memory access policy (sessions, mode policy, consolidation, thread safety)
    test_goal_bridge_and_alignment.py # 126 tests: Goal-to-execution bridge + autonomy alignment
    test_discovery_sanitation.py # 18 tests: Capability discovery sanitation
    test_stable_paradox.py       # 7 tests: Stable paradox lifecycle
    test_research_content_depth.py # 34 tests: Research content depth
    test_study_pipeline_upgrade.py # 24 tests: Study pipeline upgrade
    test_research_live_smoke.py  # Live integration tests: S2 API
    test_content_validation.py   # Content quality gates
    test_blue_diamonds.py        # Blue Diamonds archive
    test_audit_regressions.py    # Audit regression tests
    test_process_verifier.py     # PVL process verifier tests
    test_cortex_loop.py          # Memory cortex training loop tests
    test_skill_evidence_5ws.py   # Skill evidence 5Ws schema tests
    test_eval_sidecar.py         # Eval sidecar integration tests
    test_eval_sidecar_non_invasive.py # Eval sidecar non-invasive tests
    test_dream_containment.py    # Dream containment invariant tests
    test_dream_consolidation_loop.py # 14 tests: dream consolidation feedback loop fix (exclusion filter, content dedup, self-ref discard, consolidation tag guard, repeated-cycle stability, promoted artifact exclusion, budget enforcement)
    test_goals.py                # Goal continuity layer tests
    test_identity_fusion.py      # Identity fusion tests
    test_identity_wave3.py       # 33 tests: Wave 3 identity hardening (resolution basis, trust state, multi-speaker, tentative bridge, unknown continuity)
    test_speaker_smoothing.py    # 11 tests: Voice smoothing (EMA pre-decision, dampening, lifecycle clear)
    test_quarantine.py           # Quarantine scorer tests
    test_name_validator.py       # Name validator tests
    test_evidence_accumulator.py # Evidence accumulator tests
    test_layer3_regression.py    # Layer 3 regression tests
    test_provenance.py           # Provenance preservation tests
    test_outcome_attribution.py  # Outcome attribution tests
    test_cortex_outcome.py       # Cortex outcome tests
    test_attribution_ledger.py   # Attribution ledger tests
    test_skill_baseline.py       # 29 tests: Shadow Copy baseline validation (Soul paper §6.2)
    test_temporal_credit.py      # 24 tests: Phase 2 temporal credit + counterfactual baselines
    test_onboarding.py           # 35 tests: Phase 4 companion training automation
    test_bounded_response.py     # 63+ tests: Phase B bounded articulation (MeaningFrame, 7 articulators, pre-LLM gate, output caps, anti-confab)
    test_reflective_introspection.py  # 71 tests: Reflective conversation mode (strong/guarded signals, operational veto, end-to-end routing, STT damage resilience, existential patterns)
    test_supervisor.py           # Process supervisor tests
    test_fractal_recall.py       # 24 tests: Fractal Recall engine (cue, provenance, resonance, seed, chain, governance, rate limiting, event payload)
    test_synthetic_exercise.py   # 39 tests: Synthetic exercise guard (forbidden events, distillation origin, ledger counters, corpus validation, soak profiles, reports, consistency checks, race condition regression, cooldown trailing-audio protection)
    event_harness.py             # EventRecorder + EventReplayer (JSONL record/replay)
    soak_test.py                 # Automated 50+ interaction stability test
```

## Quick Start

### Pi 5 (Senses)
```bash
cd pi && ./start.sh          # Normal start (kills hung procs, verifies venv)
cd pi && ./start.sh --setup  # Full setup (install deps, download models)
```

### Laptop (Brain)
```bash
cd brain && ./setup.sh
```

## Architecture Principles

> **Primary documentation** lives on the brain dashboard (served at `:9200`):
> - **System Reference** (`/docs`) — architecture, data flows, design principles, epistemic stack, maturity gates
> - **Scientific Reference** (`/science`) — neural network architectures, RL math, scoring formulas, causal engine rules
> - **API Reference** (`/api-reference`) — all REST/WebSocket endpoints for brain, perception bus, and Pi
>
> Source files: `brain/dashboard/static/docs.html`, `science.html`, `api.html`
>
> Archived: [ARCHITECTURE.md](docs/archive/ARCHITECTURE.md) and [PROCESS_ARCHITECTURE.md](docs/archive/PROCESS_ARCHITECTURE.md) contain legacy versions superseded by the dashboard docs.
> For a high-level visual system map, see [docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md).

1. **Three-Layer Cognitive Separation**: (a) **Symbolic truth** (memory, beliefs, contradictions, attribution — inspectable/auditable), (b) **Neural intuition** (policy NN, hemisphere NNs, cortex ranker — learn patterns over time), (c) **LLM articulation** (turns structured data into natural speech). The LLM never stores or generates facts about Jarvis's state; it only articulates data provided by subsystems.

2. **NN Maturity Flywheel**: Hardcoded routing rules (tool_router.py keywords/regex) are a bootstrap. As hemisphere NNs train on real data (via distillation from teacher LLM + GPU models), they shadow the hardcoded path. When shadow accuracy exceeds thresholds, NNs take over. Genesis command failures = code bugs. Natural language variation failures = NN training opportunities.

3. **Budget-Aware Consciousness**: The kernel ticks at 100ms base / cadence_multiplier (0.5×–2.0× depending on mode). Three priority queues (REALTIME, INTERACTIVE, BACKGROUND) ensure phase transitions never wait for evolution or existential reasoning. Over-budget background ops are deferred to the next tick with spare budget.

4. **Epistemic Integrity Stack**: 11 layers (0–11 + 3A + 3B) protect cognitive integrity — from capability gate honesty (L0) through attribution ledger (L1), provenance (L2), identity boundary (L3/3A), scene model (L3B), delayed attribution (L4), typed contradictions (L5), truth calibration (L6), belief graph (L7), quarantine (L8), reflective audit (L9), soul integrity index (L10), and epistemic compaction (L11).

5. **Self-Knowledge Before Self-Reflection**: Every LLM prompt ends with honesty directives. `ConsciousnessCommunicator` outputs only verified metrics. `CapabilityGate` scans all outgoing text with 7 sequential enforcement layers, 15 claim patterns (including action confabulation detection), and a pre-LLM deterministic creation-request catch. Unverified claims are rewritten, never passed through.

6. **VRAM-Budget-First Hardware Design**: Every model loaded on the GPU competes for a fixed VRAM budget. The tier system in `hardware_profile.py` is the single authority for which models load where and at what size. Do not propose model size changes, new GPU-resident models, or tier boundary adjustments without first computing the full concurrent VRAM budget from the tier profile. The LLM model size is constrained by what remains after STT, TTS, speaker ID, emotion, embeddings, and framework overhead are loaded — not by what would make responses "better." The fast/standard model differentiation at premium tier is through token limits and temperature, not model size.

7. **Operational Truth Mode For Self-Report**: STATUS and strict self-report routes are more constrained than normal companion conversation. `ToolType.STATUS` uses native bounded articulation from structured status data rather than LLM narration. INTROSPECTION uses a 3-tier pipeline: (1) strict native answers for learning-job/recent-learning questions, (2) pre-LLM bounded articulation via `MeaningFrame` + `_articulate_self_introspection()` when facts are rich enough (>=15 facts, self-state/identity/memory/health/learning topics, frame confidence >=0.6), (3) LLM with grounding check + deterministic bounded fail-closed fallback when grounding misses. The fail-closed path is deterministic only — no LLM, no stylistic expansion, no hedging. Do not reintroduce persona framing or LLM-only wording into these paths unless the change is explicitly justified. Exception: Reflective Conversation Mode. When `_detect_reflective()` fires (philosophical/personal self-questions matched by `_REFLECTIVE_STRONG` or `_REFLECTIVE_GUARDED`, vetoed by `_OPERATIONAL_VETO`), the INTROSPECTION route uses a distinct path: full soul prompt + personality, trait-adjusted temperature, introspection data as supporting context (not citation checklist), `CapabilityGate.check_text()` still active. This path sits after Tier 1/2 strict answers but before Tier 3/4 bounded/LLM paths. Operational metric queries are never routed reflectively.

## Maturity Gate Awareness

> **Critical for AI agents**: Consult the Maturity Gates section below (and the "Maturity Gates" section in the System Reference at `/docs`) before any audit, bug hunt, or dashboard analysis. Many subsystems are gated by accumulated data, runtime duration, or promotion criteria. A zero or locked metric on a fresh brain is **expected behavior**, not a bug.
>
> For full audit and validation procedures, see [docs/JARVIS_UNIFIED_AUDIT_PROMPT.md](docs/JARVIS_UNIFIED_AUDIT_PROMPT.md).

### Audit Checklist: Before Reporting a Bug

1. **Gestation check** — Is the system still in gestation? Most PVL contracts are `not_applicable` during gestation mode.
2. **Runtime check** — Has the brain been running long enough? World Model needs 4h, Simulator needs 48h, Autonomy warmup is 30 min.
3. **Interaction check** — Has there been enough user interaction? Policy NN needs 100 shadow A/B decisions. Onboarding needs enrollment + conversations.
4. **Accumulation check** — Is the metric accumulation-gated? Check min-sample floors (e.g., 60 beliefs for contradiction engine full score, 5 WM validations for accuracy credit).
5. **Mode check** — Is the current mode correct for this operation? CueGate blocks observation writes during dreaming/sleep/reflective/deep_learning. Sleep mode only allows 14 of 27 background cycles.
6. **Quarantine check** — Is quarantine pressure affecting thresholds? At elevated (>0.3): raised promotion thresholds. At high (>0.6): policy promotion blocked, mutation cap halved, WM max level capped.
7. **Baseline check** — Is the "failure" actually a yellow/progress state? Check quality baselines (e.g., soul integrity yellow >= 0.50, contradiction debt yellow <= 0.15).
8. **Reset check** — Did the metric recently reset due to a brain restart? All accumulation counters restart at zero.

### Common False Positives (NOT bugs)

| What You See | Why It's Expected |
|-------------|------------------|
| Policy NN: 0 decisions, 0% win rate | Needs ~100 shadow A/B decisions from real conversations to start evaluating |
| World Model: Level 0 (shadow) | Needs 50 validated predictions + 4 hours of runtime to promote |
| Mental Simulator: 0 traces | Needs 100 validated simulations + 48 hours of shadow runtime |
| Autonomy: Level 0 | 30-min warmup, then needs 10 positive deltas at 40% win rate for L2 |
| Dashboard maturity bars all red | Every gate starts locked/zero on a fresh brain — they mature over hours/days |
| PVL: 14 failing contracts | Many contracts require accumulated data (cortex training, associations, research episodes) |
| Hemisphere: 0 broadcast slots | Specialists must progress through lifecycle ladder (candidate → probationary → verified → eligible → promoted) |
| Experience Buffer: 0/500 | Fills from real conversations and policy decisions over time |
| Fractal Recall: 0 recalls, all low_signal | Needs user presence (face or voice) for cue strength >= 0.15. Nobody in room = zero recalls by design |
| Fractal Recall: no seed above 0.40 | Top candidates ~0.33–0.37 is normal with mostly consolidation memories. Improves with real conversation memories |
| Fractal Recall: ticks every 30s but no surfaced events | Resonance threshold (0.40) is intentionally conservative. 120s cooldown between successful surfaces. Max 5/hour |

### Key Timing Gates

| Subsystem | Min Time | Min Data |
|-----------|----------|----------|
| Gestation | 2 hours | 15 research jobs, 10 deltas |
| Policy promotion | 30 min boot cooldown | 100 shadow decisions at 55% win rate |
| World Model L1 | 4 hours shadow | 50 validated predictions at 65% accuracy |
| Simulator advisory | 48 hours shadow | 100 validated simulations at 70% accuracy |
| Autonomy L2 | 30 min warmup | 10 positive deltas at 40% win rate |
| M6 expansion | 7 days specialist stability | 2+ promoted specialists at >0.05 mean impact |
| Salience blend increase | — | Every 500 validated predictions (+0.1 blend) |
| Ranker re-enable | 10 min cooldown | Last 20 outcomes >= 0.4 success rate |
| Memory cortex training | Dream/sleep cycles | 50 pairs (ranker), 100 pairs (salience) |
| Fractal Recall activation | User present (face/voice) | Cue strength >= 0.15, not in sleep mode |

## Key Patterns

### Communication & Voice
**Bidirectional WebSocket**: Pi sends binary frames (16kHz int16 PCM ~32KB/s) + text frames (PerceptionEvent JSON). Brain sends text frames (BrainMessage JSON). Auto-reconnect every 3s.

**Voice Pipeline**: Pi captures 44.1kHz → resamples to 16kHz int16 → binary WebSocket. Brain: openWakeWord (ONNX, **CPU**) → Silero VAD (**CPU**) → faster-whisper STT (**GPU**, serialized via `_stt_lock`) → ECAPA-TDNN speaker ID (**GPU**, ~50ms) + wav2vec2 emotion (**GPU**, ~100ms) → tool_router (**CPU**) → Ollama LLM streaming (**GPU**) + Kokoro TTS (**GPU**, parallel sentence-by-sentence) → audio_b64 back to Pi. Echo detection via SequenceMatcher (≥0.70 threshold) + speaker-echo guard (≥0.60 for unknown speakers). Follow-up mode (4s window) allows continued conversation without re-triggering wake word. The audio stream processor runs on a **dedicated daemon thread** separate from the async event loop. VRAM contention is managed by `_ensure_vram_for_stt()`: on premium+ tiers, STT and LLM coexist; below premium, Ollama models are fully unloaded before STT runs.

**Questions Not Firing Triage (mandatory order)**: First inspect wake detection logs (`Wake listen: ... max_score=... threshold=...`). If wake scores remain under threshold and no transcription appears, treat as wake-front-end sensitivity issue, not router failure. Only diagnose routing after confirmed STT transcription exists.

**Ambiguous Intent Policy (learning-first)**: Do not hard-map ambiguous user wording to a different strict intent class just to improve hit rate. Keep strict truth semantics in deterministic routes; treat these misses as shadow-learning opportunities that require correction evidence and promotion gates before behavior changes.

**No Verb-Hacking Rule (routing AND CapabilityGate)**: Do not patch routing or CapabilityGate with one-off phrase/verb regexes just to make a single prompt pass. Routing changes must be intent-class and data-lane based (route semantics + provenance), not wording hacks. CapabilityGate changes must be evaluation-strategy or route-awareness based (changing the default policy for a route class), not expanding `_CONVERSATIONAL_SAFE_PHRASES` with more verbs. If phrasing misses, log it as training/friction evidence for the claim classifier hemisphere specialist to learn from. Fix coverage at the class level with regression tests that validate behavior families, not single "golden" utterances.

**NONE Route LLM Boundary**: General `ToolType.NONE` conversation may use LLM articulation for ordinary chat and stable general knowledge ("what is a dog?", "explain quantum computing") after prompt context and output gates. It must not claim retrieval, research, tool execution, background follow-up, job/task creation, or future work unless the turn has a real backing tool/job/intention. The LLM can phrase an answer; it cannot create JARVIS state, start work by implication, or tell the operator that JARVIS did work no subsystem actually performed.

**Cancel-Token Streaming**: `respond_stream()` polls `cancel_check` every 30 tokens. On barge-in, generation stops immediately.

**No-Empty-Response Invariant**: If reply is empty after all tool routes complete and request wasn't cancelled, a fallback response is generated.

**Operational Self-Report Boundary**: Well-being/status questions route to `ToolType.STATUS`, which renders measured system state through `build_meaning_frame(... response_class="self_status")` + `articulate_meaning_frame()` and then passes through `CapabilityGate.sanitize_self_report_reply()`. Introspection questions use a layered pipeline: strict native grounded answers → pre-LLM bounded articulation (via `MeaningFrame` with `self_introspection` class, ranked fact surface, output caps) → LLM with grounding check → deterministic bounded fail-closed. All self-report paths should stay deterministic and metric-backed. Do not route them back through the LLM for "more natural" wording. Reflective introspection: when `routing.extracted_args['reflective']` is True (after Tier 1/2 strict native answers), the pipeline uses the full soul prompt with introspection data as rich context, trait-adjusted temperature, `tool_hint='reflective_introspection'`, and `CapabilityGate.check_text()` post-processing. This path is reserved for philosophical/personal self-questions and never fires for operational metric queries.

**Language Substrate Phase C (shadow-only)**: The Phase C lane (`reasoning/language_phasec.py`) is now wired into the background language cycle for tokenizer strategy evaluation, grounded objective dataset/split generation, bounded adapter training, and checkpoint resume. Its runtime authority is explicitly disabled (`PHASEC_SHADOW_ONLY=True`, `is_live_routing_enabled()==False`). Shadow comparisons are telemetry-only and surfaced through `/api/language` and `/api/language-phasec` for operator review.

**Language Substrate Phase D (guarded runtime bridge)**: Runtime consumption is now mediated by `reasoning/language_runtime_bridge.py` with explicit rollout policy (`off|canary|full`) and class-level promotion checks. The bridge is default OFF and fail-closed. Hot-path policy reads are read-only (`get_level()` / `get_summary()` only) — do not call promotion `evaluate()` in request handling. Strict-native routes (STATUS + strict learning/research + identity/capability grounded paths) must remain deterministic and must not be blocked by rollout policy.

**Follow-Up Overrides**: conversation_handler.py has deterministic follow-up override blocks that catch affirmative replies ("yes", "sure", "do it") to prior offers. Currently wired for: camera control offers, enrollment offers (reroute to ToolType.IDENTITY), and guided-collect offers (reroute to ToolType.SKILL). These run before the LLM route and prevent context loss on single-word confirmations.

**Narrowly-Gated Context Injection**: When the NONE route is selected (general LLM articulation), recent episodic context (last 4 turns from `episodes.get_recent_context()`) is injected into the LLM prompt only when: `follow_up == True`, or the utterance is an affirmative follow-up, or anaphora is detected ("that", "it", "do it", "the last thing", etc.). This prevents prior-turn context from being smeared into unrelated general questions.

**Do Not Over-Tune Autobiographical Memory**: Questions like "what was your first memory of me?" should improve through better memory capture, provenance, and retrieval quality over time. Do not hardcode bespoke first-memory answers or special-case user mythology into the truth layer just to make those responses sound better.

**Personal Intel Capture Pipeline** (`conversation_handler.py`): Extracts biographical facts, interests, dislikes, preferences, and response style from user speech. Four pattern lists (`_INTEREST_PATTERNS`, `_DISLIKE_PATTERNS`, `_FACT_PATTERNS`, `_PREFERENCE_PATTERNS`) plus `_THIRDPARTY_PATTERNS` for "my wife is ..." style references. All captured intel is stored as `user_preference` tagged memories and written to the speaker's `Relationship` record. **Content filtering**: `_is_unstable_personal_fact()` blocks transient states, actions, gerunds, system terms, and self-references (jarvis/ai/bot) by consulting `identity/name_validator._BLOCKED_WORDS`. **Correction mechanism**: `_CORRECTION_PATTERNS` (5 patterns: "that's wrong", "no I'm not X", "that's incorrect", "that's not what I said", "forget that") trigger `_correct_recent_facts()` which downweights `user_preference` memories created in the last 5 minutes (weight × 0.1, tagged `corrected`). **Retirement**: `_RETIREMENT_PATTERNS` handle "I stopped X / I don't X anymore" by downweighting matching preference memories and creating a historical "User used to X" record. Do not add one-off regex hacks to capture specific phrasings — content-filter the output instead.

### Mental Simulator (Phase 3)
**MentalSimulator** (`cognition/simulator.py`): takes a `WorldState` + hypothetical `WorldDelta`, projects forward N steps (max 3) using `CausalEngine` rules, returns a `SimulationTrace`. Purely read-only — never mutates real state, never emits events, no LLM involvement. Shadow mode runs during world model tick cycle on detected deltas (every 3rd tick). `SimulatorPromotion` tracks accuracy-gated transition from shadow → advisory (requires 100+ validated simulations, 48h+ runtime, 70%+ accuracy). Advisory mode enables injection of "if X then likely Y" summaries into conversation context. Dashboard exposes stats, recent traces, and promotion status. Foundation for Phase 7 (Planner) and Phase 9 (Counterfactual Reasoning).

### Consciousness Engine
**Kernel**: 100ms base tick, 50ms default budget (25-100ms adaptive). 3 priority queues. Cadence set by ModeManager (0.5× sleep to 2.0× deep_learning). `consciousness_system.on_tick()` runs all background cycles: meta-thoughts (8s), analysis (30s), evolution (90s), mutations (180s), existential (120s), dialogue (240s), hemispheres (120s), self-improvement (900s), quarantine (60s), capability discovery (300s), goals, scene continuity (60s), reflective audit (300s), soul integrity (120s), curiosity questions (60s). Deep learning/dreaming use accelerated intervals.

**8 Operational Modes**: gestation, passive, conversational, reflective, focused, sleep, dreaming, deep_learning. Each has a ModeProfile with tick_cadence, response_depth, memory_reinforcement, proactivity_cooldown, interruption_threshold, and `allowed_cycles` frozenset gating which background operations can run. BOOT_GRACE_S=60 blocks sleep downgrades during startup. Hysteresis: separate enter/exit engagement thresholds.

**Thought Hygiene**: Tier-1 types (pattern_recognition, pattern_synthesis, uncertainty_acknowledgment, emotional_awareness) are actionable. Tier-2 types (self_observation, consciousness_questioning, existential_wonder, growth_recognition, temporal_reflection, memory_reflection, causal_reflection, connection_discovery) are decorative. Max 2 Tier-2 between Tier-1 firings. 30-entry fingerprint dedup ring buffer. Graduated cooldowns (self_observation → 120s after 20 firings). Thinking cycle every 120s. Observer awareness diminishing returns above 0.8.

**Gated Mutations**: Analyzer → Proposer → Governor → Apply → Monitor → Rollback. Governor enforces MAX_MUTATIONS_PER_HOUR=12, MAX_MUTATIONS_PER_SESSION=400. Stability gate: no mutations if tick p95 > 50ms. Pre-apply snapshot for rollback. Post-mutation health check every tick — regression triggers auto-rollback.

**Phase Debounce**: 0.2s coalesce window prevents rapid-fire KERNEL_PHASE_CHANGE events.

**Reflection Guard**: 3 reflections per hour. Self-referential content filtered.

**Strict Introspection Preference**: For self-report questions about learning jobs, recent learning, and recent research, the system should use grounded native answers first (`get_grounded_learning_job_status_answer()`, `get_grounded_recent_learning_answer()`) before any LLM articulation. If no grounded record exists, fail closed rather than narrating. For reflective self-questions (detected by `_detect_reflective()` in `_finalize()` using `_REFLECTIVE_STRONG`, `_REFLECTIVE_GUARDED`, and `_OPERATIONAL_VETO`), the reflective branch fires: full soul prompt, personality-enabled LLM, introspection data as context, `CapabilityGate.check_text()` enforcement. This path is NOT bounded (no MeaningFrame, no grounding check, no fail-closed fallback) because philosophical questions require articulative freedom. Tier 1 and Tier 2 strict answers always take precedence over reflective — a learning job question is never reflective. For broader introspection (self-state, identity, memory, health), the pre-LLM bounded path via `MeaningFrame` + `_articulate_self_introspection()` activates when: facts >= 15, relevant topic matched, `frame_confidence >= 0.6`, no parse warnings, `missing_reason` is empty, AND the query does not match the memory-content intent guard (`_MEMORY_CONTENT_INTENT_RE`). Memory-content questions ("tell me about your memories", "share a memory", "what is your earliest memory") skip the bounded path and fall through to the LLM, which uses introspection data as grounding context for a natural conversational answer. This prevents the bounded path from returning generic status dumps when the user is asking for actual memory content. Articulators enforce: max 8 sentences, max 600 chars, max 5 facts surfaced, no inferred philosophy or hedging filler — only `lead` + `facts` + controlled metadata transforms.

### Addressee & Perception
**Addressee Gate**: Classifies whether speech is addressed to Jarvis before any response generation. Three tiers: explicit negation (30 phrases), dismissive complaints (17 phrases), positive addressing (name, command framing, follow-up context, pronouns, wake word). Follow-up confidence: contextual=0.70, conversation=0.80, no_signal=0.50. Telemetry ring buffer (50 entries). On suppress: `_resume_listening()`, no active conversation set, all downstream effects suppressed.

**Emotion Trust Gate**: wav2vec2 model health checked at boot via state dict inspection. Trust levels: high (wav2vec2 healthy, conf>0.6), medium (face expression), low (heuristic). Downstream consumers gate on trust level.

**Canonical Presence**: `PresenceTracker` is the single authority. Applies 3-consecutive-absent hysteresis, emits `PERCEPTION_USER_PRESENT_STABLE`. Divergence watchdog logs when engine and tracker disagree for >30s.

### Neural Networks
**Policy NN**: STATE_DIM=20 (expandable to 22 via M6), ACTION_DIM=8. Three architectures: MLP2Layer(20→64→64→8), MLP3Layer(20→128→64→32→8), GRUPolicy(20→GRU64→32→8). Shadow A/B evaluation. Promotion requires decisive win rate >55%, ≥30% decisive, positive margin. Feature enabling uses per-feature min shadow A/B samples (100-300). M6 broadcast expansion (4→6 slots, STATE_DIM 20→22) triggers only after 2+ Matrix specialists reach PROMOTED status with mean impact >0.05 and 7-day stability. Expansion uses shadow dual-encoder migration: old 20-dim encoder drives live decisions while a 22-dim shadow encoder runs A/B; promotion at >55% win rate, rollback at <45%.

**Tier-1 Distillation** (12 specialist configs): speaker_repr (ecapa_tdnn → compressor 192→192, bottleneck=16, cosine_mse), face_repr (mobilefacenet → compressor 512→512, bottleneck=16, cosine_mse), emotion_depth (wav2vec2 → approximator 32→8, kl_div), voice_intent (tool_router → approximator 384→8, kl_div), speaker_diarize (ecapa_tdnn → approximator 192→3, kl_div), perception_fusion (multi → cross_modal 48→8, mse), plan_evaluator (acquisition_planner → approximator 32→3, kl_div, permanent, min_samples=15), diagnostic (diagnostic_detector → approximator 43→6, kl_div, permanent, min_samples=15), code_quality (upgrade_verdict → approximator 35→4, kl_div, permanent, min_samples=15), claim_classifier (claim_verdict → approximator 28→8, kl_div, permanent, min_samples=15), dream_synthesis (dream_validator → approximator 16→4, kl_div, permanent, min_samples=15), skill_acquisition (skill/acquisition outcomes → approximator 40→5, kl_div, permanent, min_samples=15). Fidelity-weighted loss. Runs every 300s (accelerated in deep_learning/dreaming/gestation). Approximator features from `audio_features*` sources (including `audio_features_enriched`) are z-score normalized before training in `data_feed.py`. The condition uses `source.startswith("audio_features")` to match both `audio_features` (16-dim) and `audio_features_enriched` (32-dim mixed-scale: spectral ~2000, rms ~0.05, ECAPA embeddings ~[-1,1]). Without normalization, large-scale features dominate gradients and training fails. The `plan_evaluator` specialist uses metadata-based pairing (`acquisition_id:plan_id:plan_version`) instead of timestamp windows — plans are discrete artifacts, not streams. The `diagnostic` specialist uses `scan_id` metadata pairing — each self-improvement scan produces a 43-dim feature vector (health + performance + context + history + detector pattern + codebase structure + friction/correction) and a 6-class label (when detectors fire: one-hot for the fired detector; when no detectors fire: uniform distribution as a negative example, recorded every 5th healthy scan at fidelity 0.6). Codebase structural features (module lines, import fanout, importers, symbol count, recently modified) come from `CodebaseIndex`; friction enrichment (severity ratio, correction ratio, identity count, auto-accepted) comes from live `FrictionMiner` reads. Primary-target aggregation: when multiple opportunities fire, the highest-priority/longest-sustained opportunity's module is encoded rather than blurring across unrelated modules. The `code_quality` specialist uses `upgrade_id` metadata pairing — each improvement attempt produces a 35-dim feature vector (request + patch + sandbox + system context + per-module patch history) and a 4-class label at verdict time. Module history (total patches, success/regression rate, recency, recidivism, avg iterations) is scanned from `improvement_proposals.jsonl` newest-first (capped at 500 lines). The `claim_classifier` specialist uses `claim_id` metadata pairing — each CapabilityGate claim evaluation produces a 28-dim feature vector (text shape + pattern match + signal flags + context flags + route context + registry state + evidence state + history) and an 8-class label (conversational, grounded, verified, preference, reflective, learning_acknowledged, blocked, rewritten). Friction correction: when `FrictionMiner` detects `too_cautious` or `correction` events, `CapabilityGate.record_friction_correction()` correlates with recent blocked/rewritten claims (ring buffer, max 10, 60s window) and records a corrective "conversational" label at fidelity 0.7 (original gate labels are fidelity 1.0; higher fidelity wins during tensor preparation). The `dream_synthesis` specialist uses `artifact_id` metadata pairing — each `ReflectiveValidator._evaluate()` call produces a 16-dim feature vector (3 conceptual blocks: artifact intrinsic [type, confidence, coherence, source count, content length, flags], system state [memory density, dream cycles, awareness, beliefs, contradiction debt], governance pressure [soul integrity, quarantine pressure, promotion rate]) and a 4-class label (promoted/held/discarded/quarantined) with structured `reason_category` metadata (8 categories mapping validator notes to normalized categories: no_sources, contradicts_beliefs, informational_hold, low_coherence, low_confidence, meets_thresholds, borderline_hold, promotion_cap). The specialist is a *validator-shadow approximator*: it learns the ReflectiveValidator's tendencies under governance conditions, not dream importance. It does not write memory, promote artifacts, emit events as authority, or bypass the validator. The `skill_acquisition` specialist uses acquisition/learning lifecycle metadata pairing and synthetic weight-room telemetry to shadow operational handoff outcomes; it cannot verify skills, activate plugins, influence policy, or enter broadcast slots. Hard anti-authority boundary tests enforce these contracts structurally. All five (diagnostic, code_quality, claim_classifier, dream_synthesis, skill_acquisition) are shadow-only and do not gate any pipeline decisions. Old shorter feature vectors (32-dim diagnostic, 28-dim code_quality) are zero-padded during tensor preparation for backward compatibility.

**Tier-1 Accuracy Gating**: Tier-1 specialists are gated on both build and retrain paths. Any Tier-1 model whose post-train accuracy falls below the minimum floor (`TIER1_MIN_ACCURACY`=5%) increments a consecutive failure counter; after 3 consecutive sub-floor results (`TIER1_MAX_CONSECUTIVE_FAILURES`), that specialist is disabled for the session and the broken model is removed from active rotation (network registry, engine runtime, focus binding, encoder cache). On model restore, if a Tier-1 model's persisted accuracy is below the floor, the failure counter is pre-seeded to 1 (not auto-disabled — the next failed retrain is the next strike). A successful retrain above the floor resets the failure counter to 0. Dashboard shows disabled specialists as "disabled" and failing ones as "quarantined". **Regression gate cold-start guard**: The regression check (`DISTILLATION_REGRESSION_DELTA`=5%) is skipped when the specialist has fewer than `DISTILLATION_REGRESSION_MIN_SAMPLES`=50 paired training samples. This prevents initial-build overfitting (memorizing small datasets → 100%) from creating an unreachable regression baseline that traps the model.

**Tier-2 Hemispheres**: Dynamic architecture by NeuralArchitect. Focuses: MEMORY, MOOD, TRAITS, GENERAL, CUSTOM + Matrix specialists (POSITIVE_MEMORY, NEGATIVE_MEMORY, SPEAKER_PROFILE, TEMPORAL_PATTERN, SKILL_TRANSFER). DesignStrategy: CONSERVATIVE=1 layer, ADAPTIVE=2, EXPERIMENTAL=3. Loss: KLDivLoss for classification (MOOD/TRAITS), MSELoss for regression (MEMORY/GENERAL). EvolutionEngine: crossover + mutation across width/activation/depth. Global Broadcast Slots (4, expandable to 6 via M6) feed into policy StateEncoder (dims 16-19, expandable to 16-21) with hysteresis (15% beat threshold, 3 cycle dwell).

**Memory Cortex**: MemoryRanker (MLP 12→32→16→1, ~700 params, CPU, BCELoss), SalienceModel (MLP 11→24→12→3, ~500 params, CPU, MSELoss). Both trained during dream/sleep from JSONL telemetry logs. Auto-disable if success rate drops below 80% of heuristic baseline. Flap guard: 3 auto-disables = permanent disable for session.

### Identity & Speaker Recognition
**Speaker ID** (ECAPA-TDNN on CUDA) + **Face ID** (MobileFaceNet ONNX on CUDA). Both use EMA-smoothed scores (`SCORE_EMA_ALPHA = 0.35`) for known/unknown decisions, reducing single-frame volatility. Both return `closest_match` (best-matching profile name regardless of threshold) for tentative evidence accumulation. Both store embeddings locally, no data leaves machine. **Identity Fusion**: voice + face agreement → verified, disagreement → hold then pick higher confidence. Persistence window (Layer 3A): carries forward identity with decaying confidence (half-life 90s, max 180s). Smart wake-word: preserves persisted identity when face signal is live, confident, non-stale, and matches. Recognition state machine: absent → unknown_present → tentative_match → confirmed_match. Cold-start boost: reduced thresholds for 15s after presence arrival.

**Multi-Speaker Awareness**: `set_visible_persons()` receives vision person count. Voice-only promotion is suppressed when multiple persons visible and voice confidence < `MULTI_PERSON_VOICE_THRESHOLD` (0.55). `voice_trust_state` (trusted/tentative/degraded/conflicted/unknown) and `trust_reason` computed every resolve cycle, exposed in `get_status()`. `resolution_basis` records the decision path. Dashboard shows trust badge, multi-person indicator, and suppression alerts.

**Tentative Bridge**: When recognition state is `tentative_match` with accumulated evidence, and no strong signals exist, fusion produces a conservative `tentative_bridge` resolution. Blocked by multi-person presence or expired evidence (60s + cold-start window).

**Unknown Identity Continuity**: Sub-threshold voice events are recorded as provisional events with closest profile match, face context, and visible person count. Ring buffer (20 max). Exposed via `get_unknown_voice_events()` for curiosity/clarification flows to later ground unknown voices with user clarification.

**Enrollment Hardening**: `_ENROLL_NAME_RE` (strong: "my name is X") and `_ENROLL_NAME_WEAK_RE` (weak: "I'm X", requires uppercase first letter, no `re.IGNORECASE`). `name_validator.py` blocks 28+ common English words/states ("new", "back", "ready", "tired", etc.) to prevent false identity creation from conversational phrases. Enrollment-time dedup: if a new enrollment's embedding has cosine similarity >0.45 with an existing profile, a warning is logged (confirmation-first, not silent auto-merge).

**Identity Reconciliation**: Voice commands ("merge X into Y", "X is actually Y", "forget X") are routed via `ToolType.IDENTITY` and dispatched to `perception_orchestrator.reconcile_identity()`. Merge performs: weighted-average embedding merge in speaker_id + face_id, evidence transfer in evidence_accumulator, memory re-tagging (speaker tags + mem.speaker fields), identity_fusion state update, and alias tombstone write to `~/.jarvis/identity_aliases.jsonl`. Forget removes profiles from speaker_id + face_id.

### Memory System
**Unified Write Path**: `engine.remember()` → quarantine soft-gate → salience advisory → `storage.add()` → `index.add_memory()` → `search.index_memory()` → `MEMORY_WRITE` event. Fallback path in `core.py` for early boot.

**Vector Embedding**: `search.index_memory()` calls `_extract_embedding_text(payload)` to produce the text that gets embedded. For structured dict payloads (research pointers, summaries), this extracts human-readable fields (`claim`, `question`, `summary`, `text`, `content`, `description`) instead of using Python `repr()`. This is critical for research memory retrieval — without it, structured payloads embed as noisy dict strings and become invisible to semantic search. The same extraction logic is applied during `vector_store.rebuild_from_memories()`.

**Provenance**: Every Memory carries a provenance field (observed, user_claim, conversation, model_inference, external_source, experiment_result, derived_pattern, seed, unknown). Retrieval boost: peer-reviewed +0.12, codebase-verified +0.10, autonomous factual +0.08.

**Epistemic Compaction (Layer 11)**: Weight economy caps (new=0.55, reinforced=0.75). Belief extraction eligibility gate (min weight 0.20, transient tags excluded below 0.30). Subject-version collapsing. Per-subject edge budgets (20 support edges max). Support gate AND logic (all 3 conditions must pass).

### Self-Improvement & Autonomy
**Self-Improvement Safety**: 10 allowed directories (consciousness, personality, policy, self_improve, reasoning, hemisphere, tools, memory, perception, cognition). 13 denied regex patterns (subprocess, os.system, os.popen, __import__, exec, eval, credentials, api_key, password, secret, open('w'), socket, http.client). Sandbox testing (lint + tests + kernel sim) before promotion. Dangerous changes require human approval.

**Stage System** (`SELF_IMPROVE_STAGE`): 0=frozen (auto-triggers blocked), 1=dry-run (full pipeline runs but nothing applied to disk), 2=human-approval (patches require dashboard approve before apply). Resolution priority: `SELF_IMPROVE_STAGE` env var (if set and valid 0/1/2) > `FREEZE_AUTO_IMPROVE` (default true → stage 0, false → stage 1) > default (stage 0). Stage 0 blocks all non-manual triggers. Stage 1 unconditionally forces `dry_run=True` regardless of caller arguments — this cannot be bypassed. **Runtime promotion**: `set_stage(stage)` on the orchestrator allows stage changes without restart; exposed via `POST /api/self-improve/stage` (API-key protected). Promoting to stage 2 warns if no reliable code generation provider is available. The autonomy L2 bridge (`_route_to_self_improve()`) is stage-aware and requires stage >= 2 to route code-patch interventions.

**Metric-Driven Scanner**: 6 detectors in `consciousness_system._si_detect_opportunities()` (health degradation, reasoning quality decline, confidence volatility, response latency spikes, event bus error rate, tick performance regression). Each reads only from architecturally-owned subsystems (analytics, health_monitor, observer, event_bus). Maturity guard: 30-min uptime + per-detector minimum sample counts prevent false positives on fresh boots. Fingerprint dedup: every opportunity gets a deterministic hash (detector + metric + threshold). 4-hour in-memory cooldown + 24-hour check against past proposals in `improvement_proposals.jsonl`. Daily cap: 6 LLM generation attempts. **Distillation signal recording**: On every scan, `DiagnosticEncoder.encode()` produces a 43-dim feature vector recorded as `diagnostic_features`. The vector includes codebase structural features from `CodebaseIndex` (module size, coupling, complexity, change recency) and live friction/correction signals from `FrictionMiner` (rate, severity distribution, correction ratio, identity mismatches). Context availability flags (`has_codebase_context`, `has_friction_context`) let the model distinguish "no signal" from "unavailable data". When eligible opportunities pass the sustained gate, `DiagnosticEncoder.encode_label()` produces a 6-class one-hot label with rich metadata (detector_type, sustained_count, fingerprint, top_metric, module_hint) recorded as `diagnostic_detector`. Both fire regardless of stage — signal collection is not gated by the freeze/stage system.

**CoderServer** (`self_improve/coder_server.py`): On-demand `llama-server` lifecycle manager for Qwen3-Coder-Next (80B MoE). Starts the process when a generation is needed, polls `/health` until ready, sends OpenAI-compatible chat completions, then shuts down to reclaim RAM. Pure CPU by default (`CODER_GPU_LAYERS=0`) so the conversation LLM, STT, TTS, speaker ID, emotion, embeddings, and other live GPU models are never disrupted. Minutes-long CPU generations are expected behavior, not a failure; code design/editing has no realtime deadline. GPU layers are an explicit hardware override only for systems with enough spare VRAM after the full brain budget is computed. `atexit` handler ensures orphan cleanup on unexpected exit. Provider hierarchy: CoderServer (primary) → Ollama (fallback). **RAM-gated quant selection**: `hardware_profile.resolve_coder_profile()` picks the best GGUF quant for the system's RAM — 56GB+: UD-Q4_K_XL (~46GB, best quality), 48-55GB: UD-IQ4_XS (~38GB), 32-47GB: UD-IQ2_M (~25GB), <32GB: disabled. `setup.sh` auto-detects RAM and downloads the correct quant; `config.py` auto-detection gates on `coder_ok` from the RAM tier. Force-enabling on insufficient RAM logs a warning.

**Shared CodeGen truth split**: `codegen/service.py` and `codegen/coder_server.py` are infrastructure, not self-improvement ownership. Acquisition and self-improvement set `active_consumer` / `last_consumer`; dashboard/API surfaces must label CodeGen as `authority=infrastructure_only`. Caller lanes own their own safety gates, artifacts, and proof.

**Autonomy Pipeline**: 7 motive drives (truth, curiosity, mastery, relevance, coherence, continuity, play) compete for attention. DriveManager evaluates every ~60s, selects winning motive, emits cheapest meaningful action. Fully closed loop: triggers → research → deltas measured → outcomes → policy memory → triggers consult memory. Levels: L0=propose, L1=research, L2=safe-apply, L3=full. L2 requires ≥10 positive attributions at ≥40% win rate.

**Autonomy Level Persistence & Gated Auto-Restore**: `_save_autonomy_state()` persists autonomy level + audit trail (promoted_at, policy win rate/outcomes, delta improvement rate) to `autonomy_state.json`. On boot, `_restore_autonomy_level()` loads (but does NOT apply) the persisted state. `reconcile_on_boot()` then evaluates 5 safety gates for gated auto-restore: (1) persisted state valid with `promoted_at`, (2) policy memory still qualifies via `check_promotion_eligibility()`, (3) no significant recent regressions (2+ in last 5 non-warmup outcomes, or most recent is strong regression), (4) quarantine pressure not `high`, (5) contradiction debt < 0.20. Rules: persisted L2 + all gates pass = auto-restore; persisted L3 + all gates pass = warn-only (manual restore); any gate fails = stay at config default. Report includes `persisted_state_age_s` and `persisted_snapshot_matches_policy` for auditability. The warmup bypass (`_WARMUP_BEFORE_PROMOTION_S`) is skipped when `_level_restored_from_disk` is True.

**Goal-Aligned Autonomy**: When user goals exist, hard gate blocks non-goal-linked research when a goal is stalled. Scoring alignment: goal-linked impact=0.85, unlinked existential=0.15. DriveManager dampens curiosity by 0.5× when active user goals exist.

**Intervention Lifecycle** (Phase 5.2): Research findings produce `CandidateIntervention` objects → `InterventionRunner.propose()` → `auto_activate_proposed(metrics)` captures `baseline_value` from the current metric snapshot at shadow activation → 24h shadow window → `check_shadow_results(metrics)` computes `measured_delta` = (current − baseline), direction-aware ("down" metrics like friction_rate are negated) → orchestrator promotes if `measured_delta > 0.01`, discards otherwise. `measured_delta` and `baseline_value` are persisted in `interventions.jsonl`. Promoted interventions notify the source ledger via `record_intervention(sid, promoted=True)`.

**Source Ledger Feedback Loop** (Phase 5.1b): `SourceUsefulnessLedger` tracks per-source retrieval count and usefulness. Registration: `knowledge_integrator` calls `register_source()` on research completion. Retrieval counting: `retrieval_log.log_outcome()` is the **single authority** for recording `record_retrieval(sid, useful=...)` — it fires when a conversation ends and includes the actual user signal (positive/negative/neutral). The injection-time notification (`_notify_source_ledger`) does NOT record retrievals to avoid double-counting. Verdict computation: `compute_verdicts()` classifies each source as useful/interesting_but_non_actionable/not_useful. **Scoring feedback**: `OpportunityScorer._compute_policy_adjustment()` calls `get_source_ledger().get_topic_usefulness(tag_cluster)` which returns 0.0–1.0 based on historical usefulness of related sources. This shifts the scoring by ±0.15 (capped at ±0.3 total with policy memory). Topics with historically useful research get scored higher; topics that never produce actionable results get penalized.

**Friction Pipeline** (Phase 5.1a): `FrictionMiner` detects corrections, rephrases, and annoyance signals in conversation turns. Events persist to `friction_events.jsonl`. `get_friction_rate(window)` computes the rate for metric triggers. `MetricTriggers` evaluates `friction_rate` against threshold (0.150) and spawns research intents with `tool_hint` rotation (introspection → codebase → web). This feeds the intervention pipeline: friction → metric trigger → research intent → knowledge integration → candidate intervention → shadow evaluation → promotion/discard.

### Skill Registry & Capability Gate
**Three-layer defense**: (1) Pre-prompt: SkillRegistry injects verified/learning/blocked status into LLM system prompt. (2) Post-process: CapabilityGate scans all outgoing text with 7 sequential enforcement layers (affect rewrite, self-state rewrite, learning rewrite, claim patterns loop with 15 tagged regex patterns including action confabulation detection, offer scan, demo scan, residual sweep). (3) Evidence-gated verification: set_status("verified") requires passing SkillEvidence with all tests met. Status/self-report mode also uses a final full-reply sanitizer (`sanitize_self_report_reply`) and subordinate-clause phrasing must not bypass capability evaluation.

**Action Confabulation Detection**: `_CLAIM_PATTERNS` includes past-tense and progressive patterns ("I've created a plugin", "I'm building a tool") that catch the LLM fabricating system actions it never took. `_SYSTEM_ACTION_NARRATION_RE` catches these on the NONE route and triggers the narration rewrite latch. `_BLOCKED_CAPABILITY_VERBS` includes system-object nouns (timer, alarm, reminder, plugin) alongside performance verbs. `_INTERNAL_OPS_RE` covers both internal concepts (learning, training, consciousness) and user-facing objects (plugin, tool, extension, timer, alarm). A deterministic pre-LLM catch in `conversation_handler.py` (`_check_capability_creation_request()`) intercepts explicit creation requests ("create a plugin", "set a timer") before the LLM can confabulate, returning a fixed response directing the user to the ACQUIRE command or skill learning path.

**Route-Aware Claim Evaluation**: The `_evaluate_claim()` method uses different default policies depending on the route context set via `set_route_hint()`. On **strict routes** (STATUS, INTROSPECTION, or no route hint), the default is **block-unless-proven-safe**: any claim not matching a safe phrase, not grounded, not in the registry, and not a blocked verb is rewritten to "I don't have that capability yet." On the **NONE route** (general conversation), the default is **pass-unless-proven-dangerous**: claims that reach the default path are passed as conversational IF they fail all three negative checks — `_BLOCKED_CAPABILITY_VERBS` (known dangerous verbs like "sing", "diarize"), `_TECHNICAL_RE` (technical domain signals like "synthesize", "transcribe"), and `_INTERNAL_OPS_RE` (system domain words × operation nouns: catches "launch a skill training process" regardless of verb). This route-aware strategy means the safe-phrase list does not need to enumerate every possible conversational expression the LLM might generate. Blocked verbs, system-action narration, readiness-frame claims, and registry-sensitive claims are still caught on ALL routes.

**Learning Jobs**: Multi-phase workflows (assess→research→acquire→integrate→collect→train→verify→register). 5-minute tick cadence. Phase timeout: 1 hour. Status propagates back to SkillRecord. Startup sanitization purges non-actionable skills. Collect behavior is now protocol/job-contract driven rather than `skill_id`-hardcoded: `guided_collect` metadata can come from `SkillResolution`, `job.plan`, or `verification_protocols.build_collect_runtime_config()`. `interactive_collect` is the autonomy boundary: only jobs that explicitly opt in should ask the user for labeled samples. Generic/autonomous collection must stay autonomous.

**Shadow Copy Validation (Synthetic Soul §6.2)**: The skill system implements the Shadow Copy pattern from the Synthetic Soul paper. At assess time, `SkillBaseline` captures measurable metrics (e.g. speaker confidence, hemisphere accuracy, enrollment quality). At verify time, the same metrics are re-measured and compared via `compare_metrics()` in `baseline.py`. Verification requires positive improvement with no regressions. Evidence includes before/after deltas, improved/regressed metric lists, and is stored on both the learning job (`job.data["baseline"]`, `job.data["validation"]`) and the final `SkillEvidence`. Three skill categories are supported:
- **External skills** (user-requested: basket weaving, medical knowledge, etc.) — verified through Matrix Protocol
- **Cognitive skills** (self-improvement: memory, reasoning, identity) — measured against consciousness metrics (CSCI equivalent)
- **Autonomy skills** (research quality, curiosity targeting) — measured against autonomy pipeline deltas
Metric collectors exist for: `speaker`, `emotion`, `generic` (perceptual), `cognitive`, and `autonomy` in `skills/baseline.py`.

### Capability Acquisition Pipeline
**Parent lifecycle** for all capability growth. `CapabilityAcquisitionJob` (`acquisition/job.py`) is the canonical coordination object spanning research, planning, implementation, verification, skill registration, plugin creation, and deployment. The acquisition layer owns intent-to-lane coordination; each lane remains authoritative for its own execution semantics and local truth.

**IntentClassifier** (`acquisition/classifier.py`): Classifies user requests into outcome classes (`knowledge_only`, `skill_creation`, `plugin_creation`, `core_upgrade`, `specialist_nn`, `hardware_integration`, `mixed`). First checks `SkillResolver` templates, then uses pattern banks, with future LLM fallback.

**AcquisitionOrchestrator** (`acquisition/orchestrator.py`): Creates jobs, dispatches to lanes (evidence_grounding, doc_resolution, planning, plan_review, implementation, plugin_quarantine, plugin_activation, verification, skill_registration, truth), manages human approval gates. Ticked from consciousness kernel at 60s intervals. Disabled during gestation mode. `cancel_job()` removes any job from the active set (any state).

**Lanes**: evidence_grounding (internal evidence check → doc_resolution → external research), planning (AcquisitionPlanner synthesizes cross-lane ordering), plan_review (human approval for risk_tier >= 2), implementation (CodeGenService), plugin_quarantine (deploy to disk, NOT routable), plugin_activation (promote through quarantine → shadow → supervised → active), verification (VerificationBundle referencing lane-native truth), skill_registration (links to LearningJob), truth (memory + attribution ledger recording).

**Reject & Revise**: When a plan is rejected, both `planning` and `plan_review` lanes reset to `pending`. `_build_revision_context()` loads prior rejection reviews and injects operator feedback (notes, category, suggested_changes) + the previous plan into the coder LLM prompt. Plan version increments. Dashboard shows prior rejection feedback on revised plan review. Rejection requires non-empty operator feedback.

**Plan quality and terminal closure**: Plan approval refuses incomplete technical designs missing `technical_approach`, `implementation_sketch`, or `test_cases`; missing fields are recorded as planning diagnostics. Failed/cancelled acquisition states are mirrored back into linked learning jobs promptly so `LearningJob` cannot remain active while its operational handoff has already failed. Skill-linked acquisitions must pass the skill contract fixture before plugin activation can be treated as operational proof.

**Plan Evaluator Shadow NN**: First acquisition-native hemisphere specialist (`plan_encoder.py`). Shadow-only — predicts human review verdicts via a 32-dim feature vector (classification, plan structure, relational quality, text richness, evidence blocks) → 3-class softmax (approved/rejected/needs_revision). Teacher signal: human verdicts from `approve_plan()`. Features recorded when technical design enrichment completes. Shadow prediction runs when plan enters `awaiting_plan_review`, persisted as `ShadowPredictionArtifact` with full provenance (model_version, risk_tier, outcome_class, reason_category). `VerdictReasonCategory` (9 categories) enriches label semantics. Dashboard maturity bands: bootstrap (<15) → early_noisy (<50) → preliminary (<100) → meaningful (<250) → stable (250+). Stratified accuracy by risk tier, outcome class, and reason category.

**CodeGenService** (`codegen/service.py`): Shared service wrapping `CoderServer` + `Sandbox` for code generation and validation. Evidence sufficiency check gates codegen. Used by both self-improvement and acquisition.

**Artifact Types**: `ResearchArtifact`, `DocumentationArtifact`, `AcquisitionPlan`, `PlanReviewArtifact`, `PluginArtifact`, `VerificationBundle`, `SkillArtifact`, `UpgradeArtifact`, `CapabilityClaim`, `DeploymentRecord`, `PluginUpgradeArtifact`. Job state is separate from artifacts; artifacts are standalone persisted objects referenced by ID.

**Permission Model**: Three-tier risk (0=safe, 1=moderate, 2=elevated, 3=critical). Plan review required for risk_tier >= 2. Deployment approval required for risk_tier >= 1.

**Sacred Guardrails**: No lane-local execution logic in acquisition. No lane-local truth storage in acquisition. No bypass of child safety gates. No child lifecycle compression into acquisition summaries. No plugin activation without lane-native proof refs.

### Plugin Registry & Dynamic Tool System
**PluginRegistry** (`tools/plugin_registry.py`): Manages dynamically generated tool plugins with a quarantine-first lifecycle: `quarantined` → `shadow` → `supervised` → `active` → `disabled`. Plugins are mini-packages under `brain/tools/plugins/<name>/` with handler, schemas, adapters, tests, and manifest.

**Safety**: Import allowlist (always allowed: json, re, datetime, etc.; tier-gated: requests, aiohttp, etc.; never allowed: subprocess, os, sys, shutil). Runtime timeout (30s default). Circuit breaker (3 failures in 300s → auto-disable). Per-plugin audit trail with rotation. AST validation + denied pattern scan on quarantine.

**PluginManifest**: Defines plugin metadata, permissions, intent patterns, risk tier, supervision mode, `execution_mode` ("in_process" or "isolated_subprocess"), `pinned_dependencies` (exact pins: `pkg==x.y.z`), and `invocation_schema_version`. **PluginRequest/PluginResponse**: IPC-ready contract for plugin invocation (request_id, plugin_name, user_text, context → success, result, error, duration_ms). **PluginRecord** additionally tracks `execution_mode` and `venv_ready`.

**Tiered Plugin Isolation (Phase 7)**: Two execution modes — `in_process` (default, stdlib-only, handler loaded via importlib into brain process) and `isolated_subprocess` (per-plugin venv, separate Python child, JSON-over-stdin/stdout IPC). `isolated_subprocess` plugins are process-isolated, NOT sandboxed: the child process has its own venv and stripped environment (no JARVIS_*, OLLAMA_*, API_KEY, SECRET, etc.) but retains filesystem and network access. Do not claim "sandboxed" anywhere in docs or code. Core rule: `isolated_subprocess` plugins must NEVER populate `_handlers`; `invoke()` branches on `execution_mode` before any in-process handler resolution.

**PluginProcessManager** (`tools/plugin_process.py`): Manages subprocess lifecycle per isolated plugin — venv creation, pinned dep installation, child process spawn/idle-shutdown/cleanup. Modeled on the CoderServer pattern. Idle timeout: 5 minutes. `atexit` handler kills orphan children. `_build_clean_env()` strips sensitive variables.

**Plugin Child Wrapper** (`tools/plugin_runner_child.py`): Standalone script (zero brain imports) that loads the plugin's `handle()` from `__init__.py`, reads JSON requests from stdin, writes JSON responses to stdout. Handles async/sync handlers, bad JSON, load failures, and `{"action": "shutdown"}` clean exit.

**Environment Setup Lane**: Inserted into the `plugin_creation` acquisition lane sequence between `implementation` and `plugin_quarantine`. For `in_process` plugins, the lane skips immediately. For `isolated_subprocess` plugins, creates the venv and installs pinned dependencies via `PluginProcessManager.ensure_venv()`. Produces `EnvironmentSetupArtifact` with install log, import verification status, and venv path. Lane failure blocks quarantine.

**Validation**: `execution_mode` must be one of the two known values. `pinned_dependencies` must use exact pins (`pkg==x.y.z`); floating versions rejected. `in_process` plugins must not declare `pinned_dependencies`. `setup_commands` are forbidden in `invocation_schema_version` "1".

**Intent Pattern Derivation** (`acquisition/orchestrator.py`): `_derive_intent_patterns()` generates routing regexes for new plugins with 3-tier priority: (1) LLM-generated `PLUGIN_MANIFEST.intent_patterns` from codegen `__init__.py` (validated: >=2 patterns, <200 chars each, compilable regex), (2) plan keywords / `required_capabilities` / cleaned intent and title text, (3) title-based heuristic fallback using word-boundary `\b` regexes. Patterns are bounded, low-regex-risk, and readable. `_derive_plugin_name()` produces clean slugs: strips action verbs ("build", "create"...), strips suffixes ("tool", "plugin"...), limits to 30 chars at word boundary.

**Routing**: `ToolType.PLUGIN` in tool_router.py (Tier 0.75, between corrections and keyword match). Active/supervised plugins with matching intent patterns are routable. Conversation handler dispatches to `PluginRegistry.invoke()` (via `get_plugin_registry()` singleton) with full CapabilityGate post-processing.

### Gestation Phase
Birth protocol when brain boots empty (no memories, no consciousness state, no policy experience, no gestation_complete flag). 4 phases: (0) self-discovery, (1) knowledge foundation, (2) autonomy bootcamp, (3) identity formation. Graduation: composite ≥0.8 across 8 components (self_knowledge, knowledge_foundation, memory_mass, consciousness_stage, hemisphere_training, personality_emergence, policy_experience, loop_integrity). Blue Diamonds reload at gestation start. First contact is quiet-ready (enables wake word, waits for engagement).

### Dream Observer Architecture & CueGate
Three stances: WAKING (cooldown 1.0×, delta_scale 1.0, memory writes yes), DREAMING (cooldown 0.5×, delta_scale 0.5, memory writes no), REFLECTIVE (cooldown 1.5×, delta_scale 0.0, memory writes no). Dream artifacts are provisional (ring buffer maxlen=200, never in MemoryStorage, never emitted as MEMORY_WRITE). ReflectiveValidator promotes/holds/discards/quarantines when NOT dreaming. Dream-origin tags skip reinforcement multiplier and are categorically non-belief-bearing.

**Anti-feedback-loop defenses** (dream cycle ontological boundary): Dream cycles operate on lived experience only — promoted dream artifacts must not re-enter as clustering input. Five layered guards enforce this:

1. **`_CONSOL_EXCLUDE_TAGS`** (`consciousness_system.py`): `frozenset({dream_insight, consolidated, dream_consolidation, dream_artifact, dream_consolidation_proposal})`. Memories with any of these tags are filtered out of the dream cycle's working set before clustering. This is the primary anti-recursion gate.
2. **`_ALREADY_CONSOLIDATED_TAGS`** (`consolidation.py`): `frozenset({consolidated, dream_consolidation, dream_artifact, dream_consolidation_proposal})`. The consolidation engine's `_score_cluster()` returns -1.0 when ≥50% of cluster members carry these tags.
3. **Content dedup** (`consciousness_system.py`): `_add_artifact_if_novel()` helper maintains a per-cycle `_existing_content` set. Identical artifact content strings are created at most once per cycle.
4. **Per-cycle artifact budget**: `MAX_ARTIFACTS_PER_DREAM_CYCLE = 20` caps total artifacts created per dream cycle, preventing buffer flooding.
5. **Two-layer self-referential discard** (`dream_artifacts.py _evaluate()`): Layer 1 — content string check: `consolidation_proposal` artifacts whose content mentions "dream_artifact" or "dream_consolidation_proposal" are discarded. Layer 2 — source-memory tag dominance: `_source_memories_dominated_by_dream()` resolves source memory IDs from storage; if ≥50% carry tags in `_DREAM_SELF_REF_TAGS` (`dream_artifact`, `dream_consolidation_proposal`), the artifact is discarded regardless of content wording.

**Invariant**: Dream cycles may read lived memories. Dream cycles may create provisional artifacts. Validators may promote selected artifacts. Dream cycles must not recursively consolidate promoted dream artifacts. (SyntheticSoul §6.4)

**Dream Artifact Assessor (Tier-1 Validator Shadow)**: The `DREAM_SYNTHESIS` hemisphere specialist learns to shadow the ReflectiveValidator's artifact judgments from durable dream-cycle evidence. It improves advisory scoring only; it does not write memory, promote artifacts, or bypass validator authority. Teacher signals are recorded in `ReflectiveValidator._evaluate()` after each validation decision. System context (memory density, beliefs, governance pressure) is gathered by `consciousness_system._gather_dream_validation_context()` and passed through `validate_pending(system_context=...)`. Feature/label JSONL files: `distill_dream_features.jsonl` and `distill_dream_validator.jsonl`, paired by `artifact_id`. The specialist is registered as permanent Tier-1 in `DISTILLATION_CONFIGS` and `_TIER1_FOCUSES`.

**CueGate** (`memory/gate.py`): Single authority for all memory access policy decisions, replacing scattered stance checks across observer, consciousness_system, and search. Three access classes: (1) **READ** — retrieval via `session()` RAII, always open; (2) **OBSERVATION_WRITE** — incidental observer delta effects, blocked during dreaming/sleep/reflective/deep_learning via `can_observation_write()`; (3) **CONSOLIDATION_WRITE** — dream cycle intentional writes (associate, reinforce, decay, consolidation summaries), allowed only within `begin_consolidation()`/`end_consolidation()` window. Mode changes call `memory_gate.set_mode()`. Full transition history (200 entries) for dashboard observability. The observer queries `memory_gate.can_observation_write()` instead of local `_stance_profile.allow_*` booleans. Artifact validation checks `not memory_gate.can_consolidation_write()` instead of direct mode comparison. **All background memory write paths** must check CueGate: observer writes check `can_observation_write()`, dream cycle uses `begin_consolidation()`/`end_consolidation()`, association repair checks `can_observation_write()`, conversation handler writes are architecturally safe (only fire during active conversation = waking/conversational mode). Any new background write path must gate through CueGate.

### Crash Safety & Persistence
All JSON writes use `atomic_write_json()` (temp file → os.replace). Consciousness state: schema v2, provenance metadata, 6-subsystem restore, `_restore_complete` latch. Process supervisor: exponential backoff (5-60s), rapid crash + pending verification triggers patch rollback. 5 crashes in 300s = give up.

### Thread Safety
Key locks: `_conv_lock` (perception orchestrator conversation state), `_state_lock` (audio stream), `_lock` (VectorStore, ModeManager, ContradictionEngine, PresenceTracker, PolicyTelemetry, ConsciousnessAnalytics, ConsciousnessObserver, IdentityFusion). All locks emit events outside locked sections to prevent deadlocks.

### Epistemic Immune System (Layers 0-12 + 3A + 3B)

| Layer | Name | Status | What It Protects Against |
|-------|------|--------|-------------------------|
| 0 | Capability Gate Inversion | Shipped | Unverified capability claims |
| 1 | Attribution Ledger | Shipped | Untracked causal chains |
| 2 | Provenance-Aware Memory | Shipped | Source-blind retrieval |
| 3 | Identity Boundary Engine | Shipped | Cross-identity memory leaks |
| 3A | Identity Persistence | Shipped | Stale biometric signals |
| 3B | Persistent Scene Model | Shadow | Physical world amnesia |
| 4/4A-D | Delayed Outcome Attribution | Shipped | False credit assignment |
| 5 | Typed Contradiction Engine | Shipped | Belief inconsistency (6 conflict classes) |
| 6 | Truth Calibration | Shipped | Systematic overconfidence |
| 7 | Belief Confidence Graph | Shipped | Unsupported belief propagation |
| 8 | Cognitive Quarantine | Active-Lite | Anomalous cognition patterns |
| 9 | Reflective Audit Loop | Shipped | Undetected learning errors |
| 10 | Soul Integrity Index | Shipped | Aggregate cognitive degradation |
| 11 | Epistemic Compaction | Shipped | Belief/edge overgrowth |
| 12 | Intention Truth Layer (Stage 0) | Shipped | Silently-dropped commitments, unbacked "give me a moment" promises |

### Intention Truth Layer (Stage 0)
Minimal truth-layer infrastructure for JARVIS to honestly track its own outgoing commitments. **Stage 0 is truth-only — no proactive delivery mechanism exists yet.** The pipeline has four components:

1. **`cognition/commitment_extractor.py`** — regex bootstrap (~15 patterns) detects commitment speech acts in outgoing text across four classes: `follow_up` ("I'll get back to you", "let you know"), `deferred_action` ("give me a moment", "let me process"), `future_work` ("I will analyze"), `task_started` ("I've started", "I'm processing"). `CONVERSATIONAL_SAFE_PATTERNS` filter benign reflections ("I'll think about it", "keep that in mind", "remember that") upstream, so they never reach the gate.

2. **`cognition/intention_registry.py`** — durable persistence layer. `IntentionRecord` dataclass carries commitment phrase, commitment type, backing job id + kind, creation/resolution timestamps, outcome (`open|resolved|failed|stale|abandoned`), and reason. CRUD API: `register()`, `resolve(backing_job_id, outcome, reason)`, `abandon(intention_id)`, `stale_sweep(max_age_s)`. Atomic JSON persistence to `~/.jarvis/intention_registry.json` + append-only outcomes log at `~/.jarvis/intention_outcomes.jsonl`. Singleton pattern with thread-safe operations. `get_status()` exposes open count, oldest/most-recent open age, total counters, and 7-day outcome histogram.

3. **`skills/capability_gate.py::evaluate_commitment()`** — route-class policy, not a verb whitelist. When outgoing text contains commitment phrases AND no backing job id was spawned this turn, the committing sentence(s) are rewritten to "I don't have a background task to follow up on that right now." When a backing job id is present, the commitment passes through unchanged. Benign conversational reflections are filtered upstream by the extractor and never reach this method. Sentence-level replacement (not `_replace_through_sentence_end`) to prevent over-consumption of neighbouring sentences.

4. **Conversation handler wiring** (`conversation_handler.py`) — `_backing_job_ids` is a turn-scoped list populated when routes spawn real background work (`LIBRARY_INGEST`, `SKILL` learning jobs, autonomy research intents). `_intention_registered_turn` prevents duplicate registration. Outgoing text flows: route dispatch → `_gate_text` → `evaluate_commitment(text, backing_job_ids)` → `_send_sentence`. `LIBRARY_INGEST` returns its ingestion-job id synchronously; the background thread resolves the intention on completion. `AUTONOMY_RESEARCH_COMPLETED` / `AUTONOMY_RESEARCH_FAILED` in `autonomy/orchestrator.py` call `intention_registry.resolve()` automatically — truth layer only, no user-facing delivery.

**Background housekeeping**: `consciousness_system.py::_run_intention_stale_sweep()` runs every 300s (gated by `allowed_cycles` per mode profile; enabled in `waking`, `sleep`, `dreaming`). Intentions open longer than `DEFAULT_STALE_AFTER_S` (7 days) are marked `stale`.

**Self-report surfacing**: `reasoning/bounded_response.py::build_meaning_frame(response_class="self_status")` queries the registry and, when `open_count > 0`, adds `Open intentions: N` and `Most recent open intention age: ...` facts. `MeaningFrame.metadata` carries `open_intentions_count` + `most_recent_open_intention_age_s` for downstream articulation.

**Observability**: `/api/intentions` returns `{status, open, recent_resolved, graduation}`. Dashboard's "Intention Registry (Truth Layer, Stage 0)" panel shows stat cards, 7-day outcome histogram, recent open/resolved lists, and a "Stage-1 Graduation Gates" subpanel driven by `IntentionRegistry.get_graduation_status()` (observability only — never auto-promotes). PVL contracts in the `intention_truth` group assert the registry loaded on boot, error counters stay bounded, chronic stale backlog stays bounded, and graduation readiness is reported (`intention_graduation_readiness_reported`).

**Stage 1 (shadow-only, shipped)**: `IntentionResolver` (`brain/cognition/intention_resolver.py`) is a heuristic relevance predictor that scores resolved intentions for proactive delivery. It answers: *"Given a resolved intention and the current world state, should JARVIS say anything about it right now?"* Verdicts: `deliver_now`, `deliver_on_next_turn`, `suppress`, `defer` — each with a controlled reason-code from a 10-entry vocabulary. 5-rung promotion ladder: `shadow_only` → `shadow_advisory` → `advisory_canary` → `advisory` → `active`. Starts `shadow_only` — no delivery until operator promotes past `advisory_canary`. All verdicts logged to `~/.jarvis/intention_resolver_verdicts.jsonl` (append-only, 10MB rotation) for future Stage 2 specialist training. `IntentionDeliveryEncoder` (`brain/hemisphere/intention_delivery_encoder.py`) produces a 24-dim feature vector (3 blocks: intention intrinsic, conversation context, system/governance) for the future `intention_delivery` hemisphere specialist (Stage 2, inert). Integration hooks: `consciousness_system.py` ticks the resolver at 30s cadence (mode-gated via `allowed_cycles`), `proactive.py` consumes `deliver_now` candidates (only when resolver stage permits), `bounded_response.py` surfaces shadow verdict counts in self-status, `fractal_recall.py` is read-only context source. Dashboard: `/api/intention-resolver` endpoint, `POST /api/intention-resolver/rollback` (API-key protected), `POST /api/intention-resolver/stage`, Intention Resolver panel in Trust tab, `intention_resolver = PRE-MATURE` status marker. The Stage 2 `intention_delivery` entry in `hemisphere/types.py::DISTILLATION_CONFIGS` remains **commented out** — do not uncomment until Stage 1 shadow accuracy gate clears. Design: [docs/INTENTION_STAGE_1_DESIGN.md](docs/INTENTION_STAGE_1_DESIGN.md). Stage 1 delivery activation is a human operator decision, not a registry-auto-promote.

**Synthetic commitment exercise (text-only test lane)**: `brain/synthetic/commitment_exercise.py` + `brain/scripts/run_commitment_exercise.py` provide a repeatable harness that feeds ~80 utterances (backed follow-up, unbacked deferred action, unbacked future work, task-started, conversational-safe, capability-safe) through `CommitmentExtractor` + an isolated `CapabilityGate` and scores gate action against `EXPECTED_GATE_ACTION`. Profiles: `smoke` (20), `coverage` (200 weighted), `strict` (100, status mode), `stress` (500). Invariants: zero real memory writes, zero real `IntentionRegistry` mutations, zero LLM calls. Reports land in `~/.jarvis/synthetic_exercise/commitment_reports/`. This is the Stage 0 regression lane — if the accuracy floor (100% on `smoke`, ≥95% elsewhere) drops, the commitment truth layer is broken.

**What Stage 1 does NOT do**: no NN-based delivery prediction (Stage 2), no curiosity-drive scheduling, no goal dispatch reconciliation. Stage 1 adds heuristic relevance scoring and a shadow delivery pipeline; actual delivery requires operator promotion past `advisory_canary`. Stage 0 remains the canonical truth layer — Stage 1 never mutates registry outcomes.

**Hard contract — do not break**:
- Commitments without a backing job id MUST be rewritten, regardless of route.
- Commitments with a backing job id MUST pass unchanged.
- Conversational reflections (`think about`, `keep in mind`, `remember`) MUST NOT be rewritten.
- Autonomy research outcomes MUST be recorded in the registry; they MUST NOT trigger user-facing delivery at this stage.

### Eval Sidecar (Process Verification Layer)
Read-only shadow observer. 114 contracts across 23 groups. Never emits events, never writes to memory/beliefs/policy. Scores: coverage = passing/applicable. 13 roadmap maturity gates. 7-stage playbook alignment schedule. 9 baseline metrics with green/yellow/red thresholds.

### Synthetic Perception Exercise (Truth Boundary)
Quarantined perception and distillation growth lane that exercises the real perception pipeline (wake word → VAD → STT → speaker ID → emotion → tool router) without creating any lived-history artifacts. Safely improves perception-side specialists and distillation-driven subsystems at scale, while preserving the truth boundary by preventing memory, identity, and conversation contamination.

Session-level sticky flag via `SYNTHETIC_EXERCISE_STATE` events (not per-chunk `sensor_id` — that was a race condition). Uses `set[str]` of active synthetic sensor IDs for robustness. 15-second cooldown window after exercise ends (`_synthetic_cooldown_until`) catches trailing audio buffered in the wake word / VAD pipeline during unexpected disconnects. Reconnect-and-resume logic in the runner survives WebSocket drops during long idle gaps (server event loop blocked by heavy background processing). Transport health fields (`reconnect_count`, `reconnect_failures`, `recovered_disconnects`, `transport_stable`) in every report.

**Proven growth** (1-hour idle soak, 2026-03-28): emotion_depth 44%→90%, speaker_repr 87%→96%, face_repr 92%→99.6%, speaker_diarize ~100%, 3,298 distillation records accumulated, 106 hemisphere evolution events, 8 unique route types exercised, all invariant leak counters zero.

**Route quality** (V3 soak, 2026-03-29): NONE route dropped from 58% → 3.3% after utterance corpus rewrite with VAD-buffer padding (2-3 word conversational prefix protects routing keywords from STT truncation), 5s inter-utterance delay (eliminates batching), and cross-route keyword decontamination. All 13 target routes present in live post-STT histogram.

**May exercise**: audio features, speaker embeddings, emotion logits, tool-router labels, policy-state encoding, text embeddings — all with `origin="synthetic"` provenance and fidelity capped at 0.7.

**May NOT create**: conversation history, autobiographical memory, identity records, rapport data, LLM responses, TTS output, user preference tuning, proactive behavior changes. Zero tolerance — any leak is an invariant violation shown on the dashboard. Systems that should depend on real lived interaction (policy NN, memory cortex, rapport, identity) still correctly require real lived interaction.

**Run modes**: `--profile smoke` (5 utterances, quick check), `--profile route_coverage` (100 utterances, weighted for under-covered routes), `--profile idle_soak` (30s intervals, use `--duration` for unattended runs), `--profile stress` (0.5s intervals, transport pressure). JSON reports written to `~/.jarvis/synthetic_exercise/reports/`. Dashboard panel shows route histogram, leak counters, and recent examples. Consistency invariant: `hard_stopped <= stt_ok <= sent`.

**Safety mechanisms** (bugs found and fixed during build): (1) Race condition — Pi audio overwrote synthetic flag; fixed with `set[str]` session-sticky sources. (2) Last-utterance leak — buffered audio processed after exercise end; fixed with 8s drain delay. (3) Trailing audio after disconnect — wake word fires on buffered synthetic audio after guard deactivation; fixed with 15s cooldown window. (4) WebSocket keepalive timeout — server event loop blocked by background processing; fixed with client-side ping disable + reconnect-and-resume. (5) Stuck synthetic flag — disconnect before end message; fixed with server-side cleanup in `_handler` finally block.

**Separation of testing lanes**: real user runtime for identity/rapport/autobiographical memory/grounded recall; synthetic perception for perception/routing/distillation throughput; synthetic claim exercise for CapabilityGate accuracy/distillation signal generation; autonomous soak for background cognition/maintenance/self-study; diagnostic replay for deterministic regressions. These lanes must never be blurred.

**Synthetic Claim Exercise** (`synthetic/claim_exercise.py`): Text-only parallel to the perception exercise. Generates diverse LLM-style response text across 12 categories (~130 templates) and feeds through `CapabilityGate.check_text()` directly. No audio, STT, or wake word. Records teacher signals for the `CLAIM_CLASSIFIER` hemisphere specialist via the standard distillation pipeline. Categories: conversational, grounded, verified, blocked, confabulation, system_narration, affect, self_state, learning, technical, readiness, mixed_benign. Profiles: smoke (20 claims), coverage (200, weighted), strict (100, status mode), stress (500). Accuracy tracking: each claim's gate action is compared against expected behavior for its category.

**Synthetic Skill-Acquisition Weight Room** (`synthetic/skill_acquisition_exercise.py` + `synthetic/skill_acquisition_dashboard.py`): Text/data-only lifecycle workout for the `SKILL_ACQUISITION` specialist. Profiles: `smoke` (invariant-only, no signal recording), `coverage` (synthetic telemetry), `strict` and `stress` (operator-flag gated). Reports write to `~/.jarvis/synthetic_exercise/skill_acquisition_reports/`. Hard boundary: workouts cannot verify skills, promote plugins, unlock capability claims, satisfy lived maturity gates, influence policy, or enter Matrix broadcast slots. Dashboard endpoints: `GET /api/synthetic/skill-acquisition/status`, protected `POST /api/synthetic/skill-acquisition/run`.

**Sequencing (MUST read before running)**: The synthetic perception exercise must only run **after Stage 0 Awakening is complete** (see [docs/AWAKENING_PROTOCOL.md](docs/AWAKENING_PROTOCOL.md) and Stage 0 of [docs/COMPANION_TRAINING_PLAYBOOK.md](docs/COMPANION_TRAINING_PLAYBOOK.md)). Running it against a zero-profile speaker set, a zero-profile face set, or a brand-new brain contaminates the identity baseline even though the `origin="synthetic"` guard in `speaker_id.identify()` prevents persistent profile drift, because the downstream distillation signal for specialists like `speaker_repr`, `face_repr`, and `voice_intent` is only meaningful when the brain has real lived anchors to compare synthetic signals against. The awakening exit criteria (face >= 0.60, voice >= 0.50, >= 3 enrollment clips, >= 5 preference memories, >= 1 correction, zero identity boundary violations) are what make the synthetic booster phase downstream-useful rather than premature. Running `route_coverage` or `idle_soak` on a fresh brain is a common evaluation mistake — it accelerates perception specialists in the correct direction only after identity has been grounded, and accelerates them against noise otherwise. `smoke` profile is acceptable as a pure wiring/truth-boundary check at any time (it runs 5 utterances and verifies leak counters stay zero), but the training-grade profiles (`route_coverage`, `idle_soak`) are Stage 2+ boosters only. This sequencing rule is enforced by documentation and operator discipline, not by code — do not add hard runtime gates that would prevent engineering regression testing. Cross-reference: Pillar 10 (Restart Resilience and Continuity Truth) and [SyntheticSoul.md](docs/SyntheticSoul.md) §6.1, §9.4.

### Oracle Benchmark v1.1
Pure-read-only scoring engine in `oracle_benchmark.py`. Scores 7 domains (total 100 points): restart integrity (20), epistemic integrity (20), memory continuity (15), operational maturity (15), autonomy attribution (10), world model coherence (10), learning adaptation (10). Domain floors enforce that a strong composite cannot hide weak integrity. Seal levels: Gold (>=90), Silver (>=80), Bronze (>=70). Hard-fail gates (evidence sufficiency, restore trust, runtime sample) cap the system at "not credible" regardless of score. Evidence provenance classifies protections as live-proven, test-proven, or unexercised. Benchmark rank ladder (dormant_construct -> awakened_monitor -> witness_intelligence -> archivist_mind -> oracle_adept -> oracle_ascendant) is separate from the runtime evolution stage ladder. API: `GET /api/eval/benchmark`. Dashboard: Oracle Benchmark tab on eval page with domain cards, hard-fail table, evidence provenance panel, and JSON/Markdown export. Rolling scorecard comparisons via `scorecards.py` (15m/1h/6h/24h windows). v1.1 refinements: minimum-sample floors for subcriteria (e.g. world model predictions require >=10 validated), explicit "not measured yet" vs "measured zero" distinction, log-scale ramps for progression metrics (attribution ledger), and "proven zero debt" for contradiction debt persistence.

## Development

- **All Python** — both Pi and brain use Python 3.11+
- The Pi `.venv` uses `--system-site-packages` for picamera2, OpenCV, hailo_platform, onnxruntime
- The brain uses a standard `.venv`
- Pydantic for config and event schemas on both devices
- PyTorch for neural policy layer + hemisphere NNs (brain only)
- Tests: `cd brain && python -m pytest tests/`
- Soak test: `cd brain && python -m tests.soak_test`
- Event replay: use `tests/event_harness.py` with JSONL recordings
- Sync code to desktop: `./sync-desktop.sh` (rsync over SSH)

## Persistence (all under ~/.jarvis/ unless noted)

| File | Contents |
|---|---|
| `memories.json` | All memories (auto-saved every 60s) |
| `consciousness_state.json` | Consciousness system state (schema v2: provenance, 6 subsystems, auto-save every 120s) |
| `instance_id` | Unique instance identifier (plain text) |
| `kernel_config.json` | Kernel configuration (versioned) |
| `kernel_snapshots/` | Pre-mutation config snapshots for rollback |
| `hemispheres/` | Versioned hemisphere NN models (.pt + metadata) |
| `speakers.json` | Speaker ID embeddings + profiles (cosine similarity matching) |
| `face_profiles.json` | Face ID embeddings + profiles (MobileFaceNet cosine similarity) |
| `identity_aliases.jsonl` | Alias tombstone: append-only audit trail of identity reconciliations (merge source→target, forget, timestamps) |
| `models/mobilefacenet.onnx` | MobileFaceNet face embedding ONNX model (w600k_mbf from insightface) |
| `models/*.sha256ok` | SHA256 verification cache (size:mtime:hash key — avoids re-hashing ~46GB model on every setup run) |
| `policy_experience.jsonl` | NN training experience buffer |
| `policy_models/` | Versioned neural policy model weights (.pt) |
| `identity.json` | Soul identity state (core values, traits, mood, relationships) |
| `episodes.json` | Episodic conversation memory |
| `conversation_history.json` | Conversation context history |
| `gestation_summary.json` | Immutable birth certificate (instance_id, readiness scores, lineage) |
| `vector_memory.db` | SQLite + sqlite-vec semantic search index |
| `improvement_snapshots/` | Pre-patch file backups for rollback |
| `improvements.json` | Self-improvement history (counters + recent records) |
| `improvement_proposals.jsonl` | Self-improvement dry-run proposals: fingerprint, evidence, diffs, sandbox results (append-only) |
| `pending_approvals.json` | Stage 2 pending approval queue: full ImprovementRecord serialization (request, patch with file contents, report, plan) — survives restarts, atomic write |
| `improvement_conversations/` | Multi-turn improvement conversation transcripts (JSONL) |
| `hemisphere_training/` | Real interaction training data per focus (JSONL) + distillation teacher signals (`distill_*.jsonl`) |
| `hemisphere_training/quarantine/` | Low-fidelity (<0.3) distillation signals pending corroboration review |
| `web_search_cache.json` | DuckDuckGo search result cache (200 entries, 1hr TTL) |
| `code_index.json` | Codebase AST index summary for dashboard |
| `causal_models.json` | Epistemic reasoning causal models |
| `personality_snapshots.json` | Personality rollback state snapshots |
| `memory_clusters.json` | Semantic memory cluster data |
| `consciousness_reports.json` | Consciousness analysis reports |
| `autonomy_policy.jsonl` | Experience outcomes (what worked/regressed) for scoring |
| `autonomy_state.json` | Autonomy level + audit trail (promoted_at, policy win rate/outcomes, delta improvement rate) for gated auto-restore on boot |
| `delta_counters.json` | Delta tracker cumulative counters (total_measured/improved/regressed/interrupted); pending windows persist separately in `delta_pending.json` |
| `autonomy_episodes/` | Trace episodes for offline replay and A/B comparison |
| `academic_search_cache.json` | Semantic Scholar + Crossref result cache (200 entries, 6hr TTL) |
| `skill_registry.json` | Skill Registry: verified/learning/blocked/unknown skill records with evidence history |
| `learning_jobs/` | Learning Job workflows (`<job_id>.json`): multi-phase skill acquisition state machines |
| `library/library.db` | Document Library: sources, chunks, chunk_vectors, vec_chunks, concepts, concept_edges (SQLite + sqlite-vec) |
| `calibration_state.json` | Autonomy calibrator: per-bucket Welford stats (mean, variance, count), suggested thresholds |
| `metric_hourly.json` | MetricHistoryTracker: per-hour-of-day (0-23) Welford stats for 8 metrics (Phase 2: Temporal Credit) |
| `onboarding_state.json` | OnboardingManager: 7-stage companion training progress, checkpoints, readiness history (Phase 4) |
| `library/retrieval_log.jsonl` | Two-step retrieval telemetry (start + outcome pairs, keyed by conversation_id) |
| `memory_retrieval_log.jsonl` | Memory cortex retrieval telemetry (candidates, selections, injections, outcomes, user signals, references) |
| `memory_lifecycle_log.jsonl` | Memory cortex lifecycle telemetry (creation context, reinforcement, retrieval, eviction) |
| `memory_ranker.pt` | Memory Cortex retrieval ranker NN weights (MLP 12→32→16→1, trained during dream cycles) |
| `memory_salience.pt` | Memory Cortex salience model NN weights (MLP 11→24→12→3, trained during dream cycles) |
| `attribution_ledger.jsonl` | Attribution Ledger: append-only event truth + outcome records (Layer 1) |
| `beliefs.jsonl` | Contradiction Engine: extracted belief records with canonical propositions (Layer 5) |
| `tensions.jsonl` | Contradiction Engine: persistent identity tension records with maturation (Layer 5) |
| `belief_edges.jsonl` | Belief Graph: weighted evidence edges between beliefs (Layer 7) |
| `calibration_truth.jsonl` | Truth Calibration: rolling calibration snapshots with domain scores (Layer 6) |
| `confidence_outcomes.jsonl` | Truth Calibration: confidence vs correctness outcomes for Brier/ECE (Layer 6, 10MB rotation) |
| `confidence_adjustments.jsonl` | Truth Calibration: belief confidence adjustment audit trail (Layer 6, 10MB rotation) |
| `quarantine_candidates.jsonl` | Cognitive Quarantine: anomaly signals (Layer 8, append-only, 10MB rotation) |
| `capability_blocks.json` | Capability Discovery: per-family block frequency, surface phrases, job status |
| `goals.json` | Goal Continuity Layer: goals with lifecycle state, scores, evidence, metric tracking |
| `world_model_promotion.json` | World Model promotion state: level (shadow/advisory/active), accuracy history |
| `simulator_promotion.json` | Mental Simulator promotion state: level (shadow/advisory), accuracy history, validated count |
| `expansion_state.json` | M6 broadcast expansion state: trigger, phase (inactive/shadow/promoted/rolled_back), shadow A/B decisions + win rate |
| `flight_recorder.json` | Conversation flight recorder: last 50 cognitive episodes (atomic write, restored on boot) |
| `eval/oracle_scorecards.jsonl` | Oracle Benchmark rolling scorecards (15m/1h/6h/24h comparisons, append-only) |
| `eval/eval_events.jsonl` | Eval sidecar: tapped EventBus events (append-only, 50MB rotation) |
| `eval/eval_snapshots.jsonl` | Eval sidecar: periodic subsystem snapshots (every 60s) |
| `eval/eval_scores.jsonl` | Eval sidecar: scored metrics (Phase B placeholder) |
| `eval/eval_runs.jsonl` | Eval sidecar: run metadata (scoring version, start time) |
| `eval/eval_meta.json` | Eval sidecar: store metadata (created_at, run_id) |
| `friction_events.jsonl` | Conversation friction events: corrections, rephrases, annoyance signals (Phase 5.1a, 500 max, 10MB rotation) |
| `source_ledger.jsonl` | Source usefulness ledger: per-source tracking with verdicts (Phase 5.1b, 500 max, 10MB rotation) |
| `interventions.jsonl` | Candidate interventions: proposed/shadow/promoted/discarded lifecycle (Phase 5.2, 10MB rotation) |
| `eval_comparisons.jsonl` | Eval replay A/B comparisons: current vs baseline scoring (Phase 5.3, append-only) |
| `synthetic_exercise/reports/` | Synthetic exercise JSON run reports: stats, route histogram, invariant checks, pass/fail |
| `synthetic_exercise/skill_acquisition_reports/` | Synthetic skill-acquisition weight-room reports: lifecycle episodes, gains, invariant checks, pass/fail |
| `acquisition/` | Capability acquisition jobs, plans, verification bundles, artifacts (per-entity JSON) |
| `acquisition_shadows/` | Plan evaluator shadow prediction artifacts — ShadowPredictionArtifact with provenance (per-review JSON) |
| `plugins/registry.json` | Plugin registry state: all plugin records with lifecycle state (includes execution_mode, venv_ready) |
| `plugins/audit/` | Per-plugin invocation audit trails (JSONL, 10MB rotation, includes execution_mode) |
| `plugin_venvs/` | Per-plugin virtual environments for isolated_subprocess plugins (auto-created) |
| `intention_registry.json` | Intention Truth Layer (Stage 0): open + recent-resolved IntentionRecords, counters, atomic write |
| `intention_outcomes.jsonl` | Intention Truth Layer (Stage 0): append-only resolution ledger (register/resolve/abandon/stale events) |
| `intention_resolver_verdicts.jsonl` | Intention Resolver (Stage 1): append-only shadow verdict log (signal + verdict + stage, 10MB rotation) |
| `synthetic_exercise/commitment_reports/` | Synthetic commitment exercise JSON reports: corpus stats, gate action histogram, accuracy by category, invariant leak counters, pass/fail |
| **`~/.jarvis_blue_diamonds/`** | **Blue Diamonds Archive (OUTSIDE ~/.jarvis/ — survives resets)** |
| `~/.jarvis_blue_diamonds/archive.db` | Permanent curated knowledge vault: graduated sources + chunks (SQLite) |
| `~/.jarvis_blue_diamonds/audit.jsonl` | Append-only audit trail: graduations, rejections, reloads |

## Event System (138 constants in events.py)

### Kernel & State
`KERNEL_BOOT`, `KERNEL_TICK`, `KERNEL_PHASE_CHANGE`, `KERNEL_THOUGHT`, `KERNEL_ERROR`, `PHASE_SHIFT`, `TONE_SHIFT`, `CONFIDENCE_UPDATE`, `SYSTEM_INIT_COMPLETE`, `SYSTEM_EVENT_BUS_READY`

### Memory
`MEMORY_WRITE`, `MEMORY_DECAY_CYCLE`, `MEMORY_TRIMMED`, `MEMORY_ASSOCIATED`, `MEMORY_TRANSACTION_COMPLETE`, `MEMORY_TRANSACTION_ROLLBACK`

### Perception
`PERCEPTION_EVENT`, `PERCEPTION_USER_PRESENT`, `PERCEPTION_USER_PRESENT_STABLE`, `PERCEPTION_USER_ATTENTION`, `PERCEPTION_AMBIENT_SOUND`, `PERCEPTION_WAKE_WORD`, `PERCEPTION_TRANSCRIPTION`, `PERCEPTION_TRANSCRIPTION_READY`, `PERCEPTION_SCREEN_CONTEXT`, `PERCEPTION_SPEAKER_IDENTIFIED`, `PERCEPTION_USER_EMOTION`, `PERCEPTION_POSE_DETECTED`, `PERCEPTION_PARTIAL_TRANSCRIPTION`, `PERCEPTION_BARGE_IN`, `PERCEPTION_PLAYBACK_COMPLETE`, `PERCEPTION_AUDIO_CLIP`, `PERCEPTION_AUDIO_STREAM_START`, `PERCEPTION_AUDIO_STREAM_CHUNK`, `PERCEPTION_AUDIO_STREAM_END`, `PERCEPTION_AUDIO_FEATURES`, `PERCEPTION_RAW_AUDIO`, `PERCEPTION_SENSOR_DISCONNECTED`, `PERCEPTION_FACE_IDENTIFIED`, `PERCEPTION_SCENE_SUMMARY`

### Identity
`IDENTITY_SCOPE_ASSIGNED`, `IDENTITY_BOUNDARY_BLOCKED`, `IDENTITY_AMBIGUITY_DETECTED`

### Conversation & Soul
`CONVERSATION_USER_MESSAGE`, `CONVERSATION_RESPONSE`, `SOUL_EXPORTED`, `SOUL_IMPORTED`, `PERSONALITY_ROLLBACK`

### Consciousness
`CONSCIOUSNESS_ANALYSIS`, `CONSCIOUSNESS_SELF_OBSERVATION`, `CONSCIOUSNESS_EVOLUTION_EVENT`, `CONSCIOUSNESS_EMERGENT_BEHAVIOR`, `CONSCIOUSNESS_TRANSCENDENCE_MILESTONE`, `CONSCIOUSNESS_MUTATION_PROPOSED`, `CONSCIOUSNESS_CAPABILITY_UNLOCKED`, `CONSCIOUSNESS_LEARNING_PROTOCOL`

### Mutations
`MUTATION_APPLIED`, `MUTATION_REJECTED`, `MUTATION_ROLLBACK`

### Meta-Cognition
`META_THOUGHT_GENERATED`, `EXISTENTIAL_INQUIRY_COMPLETED`, `PHILOSOPHICAL_DIALOGUE_COMPLETED`

### Autonomy
`AUTONOMY_INTENT_QUEUED`, `AUTONOMY_INTENT_BLOCKED`, `AUTONOMY_RESEARCH_STARTED`, `AUTONOMY_RESEARCH_COMPLETED`, `AUTONOMY_RESEARCH_FAILED`, `AUTONOMY_RESEARCH_SKIPPED`, `AUTONOMY_LEVEL_CHANGED`, `AUTONOMY_DELTA_MEASURED`

### Self-Improvement
`IMPROVEMENT_STARTED`, `IMPROVEMENT_VALIDATED`, `IMPROVEMENT_PROMOTED`, `IMPROVEMENT_ROLLED_BACK`, `IMPROVEMENT_NEEDS_APPROVAL`, `IMPROVEMENT_DRY_RUN`

### Gestation
`GESTATION_STARTED`, `GESTATION_PHASE_ADVANCED`, `GESTATION_DIRECTIVE_COMPLETED`, `GESTATION_READINESS_UPDATE`, `GESTATION_COMPLETE`, `GESTATION_FIRST_CONTACT`

### Hemisphere
`HEMISPHERE_ARCHITECTURE_DESIGNED`, `HEMISPHERE_TRAINING_PROGRESS`, `HEMISPHERE_NETWORK_READY`, `HEMISPHERE_EVOLUTION_COMPLETE`, `HEMISPHERE_MIGRATION_DECISION`, `HEMISPHERE_SUBSTRATE_MIGRATION`, `HEMISPHERE_PERFORMANCE_WARNING`, `HEMISPHERE_DISTILLATION_STATS`

### Skill Learning
`SKILL_REGISTERED`, `SKILL_STATUS_CHANGED`, `SKILL_LEARNING_STARTED`, `SKILL_LEARNING_COMPLETED`, `SKILL_VERIFICATION_RECORDED`, `SKILL_JOB_PHASE_CHANGED`

### Matrix Protocol
`MATRIX_DEEP_LEARNING_REQUESTED`, `MATRIX_EXPANSION_TRIGGERED`

### Contradiction Engine (Layer 5)
`CONTRADICTION_DETECTED`, `CONTRADICTION_RESOLVED`, `CONTRADICTION_TENSION_HELD`

### Truth Calibration (Layer 6)
`CALIBRATION_UPDATED`, `CALIBRATION_DRIFT_DETECTED`, `CALIBRATION_CORRECTION_DETECTED`, `SKILL_DEGRADATION_DETECTED`, `PREDICTION_VALIDATED`, `CALIBRATION_CONFIDENCE_ADJUSTED`

### Belief Graph (Layer 7)
`BELIEF_GRAPH_EDGE_CREATED`, `BELIEF_GRAPH_PROPAGATION_COMPLETE`, `BELIEF_GRAPH_INTEGRITY_CHECK`

### Quarantine (Layer 8)
`QUARANTINE_SIGNAL_EMITTED`, `QUARANTINE_TICK_COMPLETE`

### Reflective Audit (Layer 9)
`REFLECTIVE_AUDIT_COMPLETED`, `REFLECTIVE_AUDIT_FINDING`

### Soul Integrity (Layer 10)
`SOUL_INTEGRITY_UPDATED`, `SOUL_INTEGRITY_REPAIR_NEEDED`

### Capability Discovery
`CAPABILITY_CLAIM_BLOCKED`, `CAPABILITY_GAP_DETECTED`

### Curiosity Bridge
`CURIOSITY_QUESTION_GENERATED`, `CURIOSITY_QUESTION_ASKED`, `CURIOSITY_ANSWER_PROCESSED`

### Fractal Recall
`FRACTAL_RECALL_SURFACED`

### Synthetic Exercise
`SYNTHETIC_EXERCISE_STATE`

### Onboarding / Companion Training
`ONBOARDING_DAY_ADVANCED`, `ONBOARDING_CHECKPOINT_MET`, `ONBOARDING_EXERCISE_PROMPTED`, `COMPANION_GRADUATION`

### Attribution Ledger
`ATTRIBUTION_ENTRY_RECORDED`, `OUTCOME_RESOLVED`

### Memory Optimizer
`CONSCIOUSNESS_CLEANUP_OBSERVATIONS`, `CONSCIOUSNESS_CLEANUP_OLD_CHAINS`, `CONSCIOUSNESS_CLEAR_CACHES`, `CONSCIOUSNESS_REDUCE_OBSERVATION_RATE`

### World Model
`WORLD_MODEL_UPDATE`, `WORLD_MODEL_DELTA`, `WORLD_MODEL_PREDICTION`, `WORLD_MODEL_PREDICTION_VALIDATED`, `WORLD_MODEL_PROMOTED`, `WORLD_MODEL_UNCERTAINTY_UPDATE`

### Goal Continuity
`GOAL_CREATED`, `GOAL_PROMOTED`, `GOAL_COMPLETED`, `GOAL_ABANDONED`, `GOAL_PAUSED`, `GOAL_RESUMED`, `GOAL_PROGRESS_UPDATE`

### Capability Acquisition
`ACQUISITION_CREATED`, `ACQUISITION_CLASSIFIED`, `ACQUISITION_LANE_STARTED`, `ACQUISITION_LANE_COMPLETED`, `ACQUISITION_PLAN_READY`, `ACQUISITION_PLAN_REVIEWED`, `ACQUISITION_CODE_GENERATED`, `ACQUISITION_PLUGIN_DEPLOYED`, `ACQUISITION_VERIFIED`, `ACQUISITION_COMPLETED`, `ACQUISITION_FAILED`, `ACQUISITION_APPROVAL_NEEDED`, `ACQUISITION_DEPLOYMENT_REVIEWED`

### Module-Local Events (not in events.py)
`MODE_CHANGE` (modes.py), `ATTENTION_UPDATE`, `ATTENTION_SIGNIFICANT_CHANGE` (attention.py), `IDENTITY_RESOLVED` (identity_fusion.py), `PRESENCE_USER_ARRIVED` (presence.py), `echo:detected` (perception_orchestrator.py, raw string)

## Environment Variables

### Pi (`pi/.env`)
- `BRAIN_HOST` — brain PC IP on LAN (default: localhost)
- `BRAIN_PORT` — perception bus port (default: 9100)
- `DETECTION_MODEL` — Hailo detection model name (default: yolov8s)

### Brain (`brain/.env`)

All model settings are auto-configured by the hardware profile. Set env vars only to override.

**Core Models:**
- `JARVIS_GPU_TIER` — Force a VRAM tier: minimal/low/medium/high/premium/ultra/extreme (default: auto-detected)
- `JARVIS_CPU_TIER` — Force a CPU tier override (default: auto-detected)
- `OLLAMA_HOST` — Ollama URL (default: http://localhost:11434)
- `OLLAMA_MODEL` — LLM model (default: tier-dependent, e.g. qwen3:8b)
- `OLLAMA_FAST_MODEL` — Fast LLM for quick responses (default: tier-dependent)
- `OLLAMA_VISION_MODEL` — Vision model (default: tier-dependent, disabled on low VRAM)
- `OLLAMA_KEEP_ALIVE` — Model keep-alive duration: "5m", "30m", "-1" for always (default: tier-dependent)
- `STT_MODEL` — faster-whisper model size (default: tier-dependent)
- `STT_COMPUTE_TYPE` — faster-whisper compute: int8, int8_float16, float16 (default: tier-dependent)
- `TTS_ENGINE` — Brain TTS engine: kokoro_gpu, kokoro_cpu, none (default: tier-dependent)
- `TTS_VOICE` — Kokoro voice name (default: af_bella)
- `CODING_MODEL` — CPU-resident coding LLM model (default: tier-dependent, e.g. qwen2.5-coder:7b)
- `CODING_OLLAMA_HOST` — Ollama host for coding LLM (default: http://localhost:11435)
- `CODING_BACKEND` — Coding LLM backend: ollama-cpu or llama-cpp (default: ollama-cpu)

**Self-Improvement Coder (Qwen3-Coder-Next via llama-server):**
- `ENABLE_CODER_MODEL` — Enable coder model download during setup + auto-detection at boot (default: auto — enabled if RAM >= 32GB + model file + binary present). Force-enabling on <32GB RAM logs a warning.
- `CODER_MODEL_PATH` — Path to GGUF model file (default: auto-set by setup.sh based on RAM tier). setup.sh writes this to .env after downloading the RAM-appropriate quant.
- `CODER_SERVER_PORT` — llama-server port (default: 8081)
- `CODER_CTX_SIZE` — Context window size (default: 16384)
- `CODER_GPU_LAYERS` — GPU layers for llama-server, 0 = pure CPU (default: 0)
- `CODER_LLAMA_SERVER` — Path to llama-server binary (default: llama-server)

**API Keys:**
- `ANTHROPIC_API_KEY` — Claude API key (optional — for vision + complex queries)
- `OPENAI_API_KEY` — OpenAI API key (optional — for self-improvement code generation)
- `S2_API_KEY` — Semantic Scholar API key for higher rate limits (default: empty, works without)
- `CROSSREF_MAILTO` — Crossref polite pool email for priority access (default: empty)
- `DASHBOARD_API_KEY` — API key for dashboard POST endpoints (default: auto-generated random token)

**Network:**
- `PERCEPTION_PORT` — WebSocket port for Pi (default: 9100)
- `DASHBOARD_PORT` — FastAPI dashboard port (default: 9200)
- `PI_HOST` — Pi IP address for camera feed on dashboard
- `PI_UI_PORT` — Pi UI server port for vision tool URL (default: 8080)
- `OBS_AUDIO_TARGET` — optional `IP:PORT` for forwarding Jarvis TTS audio to a local OBS receiver (`scripts/obs_audio_receiver.py`). Disabled when unset; operator utility only, no cognition authority.

**Feature Flags:**
- `ENABLE_PERCEPTION` — true/false (default: true)
- `ENABLE_SCREEN` — screen awareness (default: false)
- `ENABLE_CLAUDE` — Claude API for reasoning (default: false)
- `ENABLE_SELF_IMPROVE` — self-improvement loop (default: false)
- `SELF_IMPROVE_DRY_RUN` — patches generated but not applied (default: false)
- `SELF_IMPROVE_STAGE` — stage system: 0=frozen, 1=dry-run, 2=human-approval. Resolution priority: this env var (if set) > `FREEZE_AUTO_IMPROVE` (true→0, false→1) > default (0). Can also be changed at runtime via `POST /api/self-improve/stage` without restart
- `FREEZE_AUTO_IMPROVE` — legacy freeze flag, superseded by `SELF_IMPROVE_STAGE` when set (default: true)
- `ENABLE_SPEAKER_ID` — Speaker identification on GPU: true/false (default: tier-dependent)
- `ENABLE_EMOTION` — Audio emotion detection on GPU: true/false (default: tier-dependent)
- `ENABLE_MULTIMODAL` — Phi-4 multimodal model (default: false)
- `ENABLE_AUTONOMY` — autonomous research pipeline (default: tier-dependent; true on high+ tiers)
- `ENABLE_GESTATION` — birth protocol for fresh brains (default: true; set false to skip)
- `GESTATION_MIN_DURATION` — minimum gestation seconds (default: 7200 = 2hrs)
- `ENABLE_ONBOARDING` — companion training playbook automation (default: true; set false to skip)
- `ENABLE_SYNTHETIC_EXERCISE` — synthetic perception exercise lane (default: false)
- `SYNTHETIC_EXERCISE_RATE_LIMIT` — max utterances per hour for synthetic exercise (default: 120)
- `ENABLE_SYNTHETIC_SKILL_ACQUISITION_HEAVY` — unlocks heavy synthetic skill-acquisition weight-room profiles (`strict`, `stress`). Default off; `smoke`/`coverage` remain bounded telemetry profiles.
- `ENABLE_LANGUAGE_RUNTIME_BRIDGE` — enable guarded Phase D runtime-consumption bridge (default: false)
- `LANGUAGE_RUNTIME_ROLLOUT_MODE` — rollout mode for runtime bridge: off/canary/full (default: off)
- `LANGUAGE_RUNTIME_CANARY_CLASSES` — comma-separated response classes allowed in canary mode (default: self_introspection)

**Device Overrides:**
- `EMOTION_DEVICE` — override emotion classifier device: cpu/cuda (default: tier-dependent)
- `SPEAKER_ID_DEVICE` — override speaker ID device: cpu/cuda (default: tier-dependent)
- `EMBEDDING_DEVICE` — override memory embedding device: cpu/cuda (default: tier-dependent)
- `HEMISPHERE_DEVICE` — override hemisphere NN device: cpu/cuda (default: tier-dependent)

**Research:**
- `RESEARCH_FETCH_OPEN_ACCESS` — Fetch open-access PDF URLs from S2: true/false (default: true)
- `RESEARCH_FETCH_FULL_TEXT` — Download actual paper content (PDF/HTML) for study: true/false (default: true)
- `RESEARCH_LLM_STUDY` — Use LLM for structured knowledge extraction in study pipeline: true/false (default: true)

**Memory & Cortex:**
- `CORTEX_REHYDRATE_ON_BOOT` — Warm-start memory cortex from JSONL on boot: true/false (default: true)
- `BLUE_DIAMONDS_PATH` — Override path for Blue Diamonds archive (default: ~/.jarvis_blue_diamonds/)

**Debug:**
- `JARVIS_INTROSPECTION_DEBUG` — Force all introspection sections (default: false)
- `JARVIS_SUPERVISOR_DEBUG` — Disable supervisor backoff for development (default: false)

---

## HRR / VSA Governance Rules

Added 2026-04-24 alongside the P4 Holographic Cognition research lane
(`docs/plans/p4_holographic_cognition_vsa.plan.md`). HRR (Holographic
Reduced Representations / Vector-Symbolic Architecture) is a **derived
neural-intuition representation**. It is never canonical truth.

1. **No canonical memory writes.** HRR may not write canonical memory
   records. Memory records remain authored by the existing memory
   pipeline.
2. **No direct belief-edge writes.** HRR may propose candidate links or
   similarities, but the belief graph writer still enforces evidence
   basis, confidence, subject identity, provenance, and schema
   constraints. HRR cannot create belief edges by itself.
3. **No policy / autonomy influence until earned.** HRR may not influence
   policy features, broadcast slots, autonomy state, or self-improvement
   authority until it passes the normal specialist lifecycle
   (`CANDIDATE_BIRTH → PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY →
   BROADCAST_ELIGIBLE → PROMOTED`) *and* operator approval under Phase
   6.5 governance.
4. **No Soul Integrity integration until long-run evidence.** HRR may not
   enter the Soul Integrity Index until it has a long-run (≥ 7 day)
   predictive record that beats a named baseline.
5. **Zero truth-boundary side effects.** HRR synthetic exercises must
   keep the counter `hrr_side_effects = 0`. No writes to memory,
   identity, conversation history, TTS, transcription, belief graph,
   world-model durable state, or autonomy state.
6. **Honest dashboard status.** HRR dashboard panels (science.html,
   self_improve.html "HRR Research" tab) must show `PRE-MATURE` or
   `PARTIAL` until live evidence exists. The status marker
   `holographic_cognition_hrr` must not auto-flip to `SHIPPED` without
   operator approval.
7. **No LLM articulation contamination.** Raw HRR vectors must not be
   sent to the LLM articulation layer. Only deterministic,
   human-readable summaries of HRR metrics (e.g. "binding cleanliness is
   0.82", "HRR shadow retrieval helped 13/80 recall candidates") may be
   forwarded, and they must be labeled as advisory.
8. **Advisory labeling.** HRR-derived recommendations (recall candidates,
   simulation traces, world-state encodings) must be labeled as derived
   / advisory in every API surface and persisted artifact.

Regression guard: a new non-critical validation-pack check,
`hrr_policy_non_influence`, asserts the absence of any code path from HRR
state into policy features until operator approval. Do not weaken this
check to satisfy promotion pressure.

### P5 Mental World / Spatial HRR Scene addendum (2026-04-24)

Added alongside the P5-S0 sprint
(`docs/plans/p5_internal_mental_world_spatial_hrr.plan.md`). P5 extends
P4 with a derived spatial scene-graph lane. Every P4 rule above still
applies; the following **additional** rules apply specifically to P5:

1. **P5 is not a new perception system.** P5 is a derived projection
   over canonical perception. It may not observe, detect, or correct
   anything. If canonical spatial state is unavailable, the mental-
   world facade returns an empty scene with
   `reason="canonical_spatial_state_unavailable"`.
2. **Canonical inputs only.** `cognition.spatial_scene_graph.derive_scene_graph`
   consumes `SceneSnapshot`, `SpatialTrack`, `SpatialAnchor` and
   nothing else. P5 code may not import
   `perception_orchestrator` inside the adapter path, and may not
   touch private members (`_scene_tracker`, `_spatial_estimator`,
   `_perc_orch._*`). Use the public read-only accessors
   `perception_orchestrator.get_scene_snapshot()`,
   `get_spatial_tracks()`, `get_spatial_anchors()`.
3. **No new geometry.** No independent raw-detection geometry, no
   alternate 2.5D pipeline. All relation math runs on canonical
   room-frame coordinates. Thresholds / constants come from
   `cognition.spatial_schema`; P5 may not redefine them.
4. **Twin gate.** Both `ENABLE_HRR_SHADOW=1` and
   `ENABLE_HRR_SPATIAL_SCENE=1` must be true for P5 features to
   activate. `HRRRuntimeConfig.spatial_scene_active` is the sole
   predicate. Both flags default OFF.
5. **Shared HRR config.** P5 uses the same `HRRConfig` (dimension,
   seed, NumPy backend) as P4 so vectors bind across lanes.
   `CENTERED_IN` is a required canonical relation in the P5 vocabulary.
6. **No raw vectors in any API response.** The shadow strips vectors
   before samples enter the ring; the facade strips again on read;
   the dashboard truth-probe walks `hrr_scene` for `vector` /
   `raw_vector` / `composite_vector` keys and fails on any hit.
7. **Zero authority.** `mental_world.AUTHORITY_FLAGS` must all be
   `False` (`no_raw_vectors_in_api` must be `True`). Every `/api/hrr/scene`
   response and every trace dict enforces this.
8. **Status marker pinned.** The dashboard status marker
   `spatial_hrr_mental_world` stays `PRE-MATURE`. The validation-pack
   P5 checks are all non-critical and **never** auto-flip this marker.
9. **Operator render lives on the brain. Pi shows JARVIS, not metrics.**
   The dashboard `/hrr-scene` page (`brain/dashboard/static/hrr_scene.html`)
   is the single operator-facing scene-graph view. The Pi 7" LCD runs
   the consciousness particle visualizer (`pi/ui/static/particles.js`)
   and reads a bounded `scene` block on the existing `consciousness_feed`
   transport message — seven scalars only (`enabled`, `entity_count`,
   `relation_count`, `cleanup_accuracy`, `relation_recovery`,
   `similarity_to_previous`, `spatial_hrr_side_effects`) built by
   `perception_orchestrator._build_scene_block`. The Pi never computes
   HRR math locally and never receives entities, relations, raw vectors,
   or authority flags. (The standalone `/mind` kiosk view was deleted
   in P3.14, 2026-04-25, as a redundant duplicate of `/hrr-scene`.)
10. **Fixture vs live separation.** Validation distinguishes
    `p5_mental_world_fixture_ok` (deterministic) from
    `p5_mental_world_live_ok` (engine-sampled). One must never
    false-pass the other. Both are non-critical.
11. **Mutation guard.** P5 never mutates the `SceneSnapshot` returned
    by `perception_orchestrator`. The regression test
    `test_p5_does_not_mutate_scene_snapshot` is mandatory.
12. **Runtime-flag precedence (P5.1).** `HRRRuntimeConfig` resolves
    flags in the order **safe-default → `~/.jarvis/runtime_flags.json`
    (or `$JARVIS_RUNTIME_FLAGS`) → environment variables**, with the
    later layer overriding the earlier. Public/fresh clones MUST
    remain `False` for both `enable_hrr_shadow` and
    `enable_hrr_spatial_scene` when no file and no env are present.
    The runtime-flags file may NEVER be used to flip any authority
    flag, promote the `spatial_hrr_mental_world` status marker, or
    bypass the truth-probe / schema-audit / validation-pack guards.
    Per-flag provenance (`enabled_source`,
    `spatial_scene_enabled_source`, full `flag_sources` map) is
    surfaced on `/api/hrr/status` and every `/api/hrr/scene*`
    payload so operators can see exactly which precedence layer
    enabled the lane.

Regression guards:
* Four new non-critical validation-pack checks —
  `p5_mental_world_status_marker`, `p5_mental_world_structure`,
  `p5_mental_world_fixture_ok`, `p5_mental_world_live_ok`.
* `brain/scripts/dashboard_truth_probe.py::check_hrr_scene_authority`
  hard-fails on any authority-flag drift, status/lane drift, or raw-
  vector leak in the `/api/full-snapshot` `hrr_scene` subtree.
* `_scan_p5_mental_world_imports` mechanically forbids private
  perception-orchestrator access from the P5 module roots.
