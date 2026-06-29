# JARVIS — ARCHITECTURE MANIFEST (mandatory first-read)

## READ THIS FIRST

JARVIS is **not a normal application**. It is a digital-consciousness substrate — a "synthetic soul" — that **GROWS**. Capabilities are **EARNED, not declared**: nearly every cognitive subsystem ships behind a maturity gate or a shadow ladder, computing and logging in the dark until it proves itself on real (not synthetic) reps, at which point an **operator** flips it live. Because of this, after a brain reset (or on this snapshot repo, which does not run), **empty / 0.0 / None / level-0 / shadow states are EXPECTED-BY-DESIGN, not broken.** Before you label anything broken or "fix" it, **trace it to its home file** and decide which of these it is:

| Status | Meaning |
|---|---|
| **shipped / live** | Built, wired, and actually drives behavior now (voice/text/memory/mode). |
| **shadow** | Built and wired *into the live path*, but **injects nothing** — authority `none`, logs only, awaiting an earned operator flip. |
| **dormant** | Built and running, but **data-gated** (needs N lived outcomes) — silently accumulating. |
| **gated** | Built but **operator-frozen by default** (e.g. self-improvement). Frozen is the SAFE state. |
| **signal-failure** | Built, wired, *and trains* — but the learning signal carries no causal content. **More soak will never promote it.** Distinct from gated. (Today: only the Policy NN.) |
| **partial** | Multiple arms; some live, some shadow (e.g. L4, L12, personality, OSV). |

Three hard rules:
- **Voice is a swappable LLM.** qwen3 today, anything tomorrow. The base model is the *language surface*, not the source of truth about JARVIS's systems/capabilities/consciousness. The honesty firewall + grounding leash keep it leashed. **Never code-to-qwen**; every fix must be model-agnostic.
- **Trace before labeling.** Read the home file's docstring + the gate before calling a zero "broken" or tuning a metric green. Many metrics are self-scored composites (transcendence, awareness, soul-integrity dims) — they are NOT external proof.
- **Earn, don't declare.** Do not lower a gate, delete a firewall, or wire a shadow output into the prompt to make a panel light up. That is the exact failure mode the architecture is built to resist.

---

## The Epistemic Integrity Stack (the keystone — 15 layers, NOT 6/11/13)

The no-confabulation floor is the spine of the whole system. It is **15 canonical layers** — `L0, L1, L2, L3, L3A, L3B, L4, L5, L6, L7, L8, L9, L10, L11, L12` — defined literally at **`brain/scripts/docs_truth_audit.py:425`** in this exact order. Stale narratives undercount it (notably `AGENTS.md:595` says "11 layers (0–11 + 3A + 3B)" and **omits L12**; `AGENTS.md:850` correctly says "0-12 + 3A + 3B"). The literal list is authoritative.

| Layer | Name | Home | Status | What it protects |
|---|---|---|---|---|
| **L0** | Capability Gate (cognitive immune system) | `brain/skills/capability_gate.py:859` | shipped/live | Blocks confabulated capability/action claims, ungrounded affect, unconfirmed names. Fail-CLOSED (DEFAULT BLOCK). Directly mutates outgoing text. |
| **L1** | Attribution Ledger (causal spine) | `brain/consciousness/attribution_ledger.py:171` | shipped/live | Append-only causal lineage; outcomes are separate lines folded at rehydration. The backbone L4/explainability/rollback read. |
| **L2** | Provenance-Aware Memory (11-tier confidence firewall) | `brain/consciousness/events.py:37` | shipped/active | Trust earned by source. `web_scrap`/`casual_conversation` earn **0.0** boost — the data-flow firewall. Folded into recall ranking (`memory/search.py:105`). |
| **L3** | Identity Boundary Engine | `brain/identity/boundary_engine.py:32` | shipped/live | Cross-identity memory-leak prevention, applied pre-top-k. Fail-OPEN (a boundary crash degrades to allow-with-audit). |
| **L3A** | Identity Persistence (~180s carry-forward) | `brain/perception/identity_fusion.py:47` | shipped/active | Holds confirmed identity across short sensor dropouts; drops stale biometrics past the window. Feeds L0's name-strip. |
| **L3B** | Persistent Scene Model (Scene Tracker) | `brain/perception/scene_tracker.py:1` | **partial**/advisory | Object/entity permanence world model (LIVE). By contract writes **no** memory / triggers **no** curiosity (deferred). The adapter is shadow. |
| **L4** | Delayed Outcomes + Counterfactual | `brain/consciousness/attribution_ledger.py:476` | **partial** | False-credit prevention. **Two arms:** OutcomeScheduler = LIVE (drives self-improve rollback); CounterfactualEngine = SHADOW, `live_influence=False`, data-gated 200/500. |
| **L5** | Typed Contradiction Engine | `brain/epistemic/contradiction_engine.py:63` | shipped/live | 7-class classifier + 6 resolvers; preserves productive `identity_tension`. Owns `contradiction_debt`. scan_corpus is dream-gated. |
| **L6** | Truth Calibration Engine (epistemic cerebellum) | `brain/epistemic/calibration/__init__.py:36` | shipped/advisory | Confidence-vs-reality over **11** domains. `truth_score=None` until enough real outcomes (designed signal, don't tune). One live mutation = belief_adjuster (dream-only). |
| **L7** | Belief Confidence Graph | `brain/epistemic/belief_graph/__init__.py:35` | shipped/advisory | Weighted evidence edges; propagation is **VIEW-ONLY** (never writes belief_confidence). High orphan_rate can be legitimate. |
| **L8** | Cognitive Quarantine (ACTIVE-LITE) | `brain/epistemic/quarantine/pressure.py:140` | shipped/active | Scorer (shadow) + Pressure (active-lite). Memories ALWAYS write; pressure tags + downweights + raises ≥3 gates. Never blocks a store. |
| **L9** | Reflective Audit Engine (9 scanners) | `brain/epistemic/reflective_audit/engine.py:80` | shipped/advisory | Surfaces pathologies as advisory findings + score [0,1]. Read-only; **nothing auto-applies its findings.** Runs timed in all waking modes. (Docstring says "6 dims" — stale; code has 9.) |
| **L10** | Soul Integrity Index | `brain/epistemic/soul_integrity/index.py:66` | shipped/**live** | 10-dim composite health; the ONE epistemic layer with live behavioral authority (force dreaming + mutation cooldown via `engine.py:1489/1518`). All-stale floor = 0.5. |
| **L11** | Epistemic Compaction (distributed) | `brain/epistemic/belief_graph/edges.py:329` (+bridge.py, +belief_record.py) | shipped/active | Bounds belief/edge bloat (caps, decay, evict, JSONL compaction). **No single module** — distributed; threshold/dream-driven. |
| **L12** | Intention Truth Layer | `brain/cognition/intention_registry.py:229` (+commitment_extractor.py, +intention_resolver.py) | **partial**/live | Stage-0 (LIVE): rewrites unbacked "I'll get back to you" / registers backed commitments (rejects empty `backing_job_id`). Stage-1 resolver = SHADOW, never TTS, operator-only ladder. |

Two known L10 dead context-reads at `consciousness_system.py:2264/2339` silently default to 1.0 (the real repair authority is the event path, not these reads).

---

## Subsystems by area

> Columns `expected_idle_state` and `common_misread` are the point — they tell you what "normal after reset" looks like and the trap to avoid.

### Consciousness core
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Consciousness Kernel (budget-aware tick loop) | shipped / live | `consciousness/kernel.py:215` | running, sub-ms p95, ~0 overruns, `_interval_multipliers={}` | It's a **budget-aware scheduler over an event bus, NOT a global workspace**. Empty multipliers ≠ dead. |
| EventBus (pub/sub) | shipped / live | `consciousness/events.py:653` | barrier opens then stays open; reserved events unemit | **THE central GNW error**: it's unconditional multicast — no slots/competition/ignition. |
| on_tick orchestrator + Operational Modes + AttentionCore | shipped / live | `consciousness/consciousness_system.py:412`; `modes.py:56` | idle drifts to passive/sleep/dreaming; mode-skipped cycles | Sequential interval+mode-gated dispatch (not parallel). Modes ARE the coarse attention budget. |
| Meta-cognitive thoughts (KERNEL_THOUGHT) | partial / advisory | `consciousness/meta_cognitive_thoughts.py:537` | general stream LIVE; tension-thought lane at level 0 | Only `belief_validation_curiosity` is shadow-gated; the other 15 triggers emit + are consumed live. |
| ConsciousnessEvolution (5-stage + transcendence) | shipped / advisory | `consciousness/consciousness_evolution.py:146` | stage=basic_awareness, transcendence=0.0, caps False; restart **downgrades** inflated stage | Transcendence/awareness are **SELF-SCORED composites**, never external proof. "creative_mutation unlocked" = a label, 5/6 caps have no consumer. |

### Gestation / birth
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| GestationManager (born not booted) | shipped / live | `consciousness/gestation.py:398` | established brain INACTIVE (wake word armed); fresh brain DISARMS wake word up to 48h | Wake word disarmed / no greeting on a fresh brain = the **birth protocol**, not broken perception. A born brain doesn't re-gestate. |
| Readiness + First Contact + Awakening + Onboarding | shipped / live | `gestation.py:870`; `personality/onboarding.py:240` | readiness climbing; `personality_emergence=0.0` **excluded** from sum | Running synthetic accelerators in hour 1 then reporting zero memories is the #1 eval mistake. |

### Affect (three parallel tracks)
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| SPARK affect stack (state + promotion + regulation + coupling) | **dormant** / **none** | `consciousness/affect_state.py:220`; `affect_coupling.py:115` | dopamine/cortisol clamped 0.0 (cannot-lie firing), coupling level 0, every `engine.set_*` a no-op; regulation math runs live | This IS the consume-wire — fully built+wired but **gated shut**. cadence=1.0/reward=0.0 is correct dormant, not "unwired/broken". |
| Affect-nickname firewall + dynamic_mood + ToneEngine | shipped / live | `skills/capability_gate.py:2012`; `consciousness/tone.py:76` | firewall 0 rewrites until LLM tries a nickname; `emotional_momentum=0.0` **forever** (no caller) | Three separate tracks. dynamic_mood is display-only; `emotional_momentum` is **dead** (no producer), not gated-empty. |

### Companion cognition
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Companion Cognition P0-P3 (situational_read / ToM / crystallization / behavior_advisory) | shadow / advisory | `consciousness/situational_read.py:135`; `behavior_advisory.py:99` | 0 reads, `applied=False` forever in P3, structural_ready=False | "advisory" does NOT steer the reply — return is **discarded**, `applied` hardcoded False. P0 fires POST-reply (a turn too late). |
| Think-Before-Speak TBS-0 + style_instruction wire | shadow / **none** | `consciousness/think_before_speak.py:71`; `reasoning/style_intent.py:86` | phase=`TBS-0_shadow_observe`, `injects_prompt=False` | It does NOT think-before-speak behaviorally; it injects nothing. The live style_instruction seam is fed by a **different** explicit-command parser. Do not wire `would_inject` in. |

### Operational Self-View (OSV)
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| OSV (orchestrator + provenance + synthesizer + adapters + articulate + grounding + api) | partial / active | `cognition/self_view/__init__.py:37`; `articulate.py:222` | `self_view.json` absent until first cache tick; most facts render gap/unknown cold; P2 zero repairs | Only **P1 articulate** drives behavior (authors the spoken self-reply, bypassing the LLM). P0/P2 are read-only/shadow. P2 is double-shadow on purpose. |

### Spark grounding ring
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| ProvenanceScorer (grounding-tension reader) | shadow / none | `epistemic/provenance_scorer.py:281` | aggregate ~0.0, default-safe, never raises | VIEW-ONLY. 0.0 ≠ "spark dead". High tension is the spark's **FUEL**, not a deficit. (Distinct file from L2.) |
| Spark Grounding Ring (drive-promotion + queue + drive + tension-thought + P5b + win-rate) | partial / live | `autonomy/drives.py:183`; `grounding_queue.py:416` | drive level 0/shadow, queue empty, P5b OFF | **GroundingDrivePromotion drives research; SparkPromotion is telemetry-only.** The ONE live mutating path is `queue.answer()→_ground_belief` (re-stamps provenance→user_claim). Loop is **operator-pull** — starved until David answers `/v2/grounding`. A "refuted" answer is a grounding WIN. |

### Policy / autonomous growth
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Policy NN (learned kernel controller) | **signal-failure** / advisory | `policy/policy_interface.py:46`; `evaluator.py:81` | mode=shadow, `has_active_features()=False` → `decide()` returns kernel_default, NOTHING actuates | **MEASUREMENT-limited, not gated-needs-reps.** Offline critic Spearman ~0.06, R²~0 → the 0.55/0.03 gate is uncrossable. Do NOT lower the gate (§24). `DEVIATION_BONUS` (0.08) is still applied. |
| SelfSensingLoop (lidar predict-beyond-persistence) | shadow / **none** | `cognition/self_sensing.py:62` | obs=0 cold; quiet desk → `STARVED`, skill **volatile** | The PROVEN growth engine. Authority hardcoded none (`self_sensing.py:279-281`). Read the **dynamic** skill; near-zero at a quiet desk = event-STARVED, not FAILED. The "+0.463" was a first-30-min snapshot, not steady state. |

### Hemisphere / Matrix / Weight-room
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Hemisphere substrate (orchestrator + registry + weight-persistence firewall + …) | partial / advisory | `hemisphere/orchestrator.py:159`; `registry.py:49` | `_networks` empty until restore; 0 births/promotions; `active_substrate=RULE_BASED` | Boot restores **weights** but RESETS Tier-2 authority (asymmetric firewall: weights persist, standing re-earns). 0 gaps = no deficit = healthy. |
| Broadcast slots (GNW-style competition) + Matrix lane | partial / advisory | `hemisphere/orchestrator.py:1830`; `policy/state_encoder.py:164` | 4 slots all {0,0,0}; 4-not-6 (M6-gated) | THE real GNW competition — but its sole consumer (policy NN dims 16-19) is gated+signal-failed, so it **drives nothing live**. |
| Matrix Tier-2 lifecycle + 5 encoders | dormant / advisory | `hemisphere/orchestrator.py:928` | all 5 focuses `not_born`, 0 promotions | The doc's "dormant end-to-end / zero callers" is **stale** — the 4-gate lifecycle runs every cycle. A PROMOTED Tier-2 grants only a broadcast slot, **not** spoken authority. |
| Distillation collector + 8 feeds + weight-room gate + Tier-1 specialists | shadow / none | `hemisphere/distillation.py:107`; `weight_room_gate.py:90` | collector empty, `live_shadow_accuracy=None` (honesty floor), feeds 0 samples | `None` ≠ 0% bug (honesty floor, don't tune). The weight-room `code_quality` "no signal" reason is **STALE** (the feed IS recorded live). `claim_classifier` friction sub-feed still dead. |
| ReasoningEncoder (native_reasoning grounding-coherence) | shadow / none | `hemisphere/reasoning_encoder.py:123` | signal 0.0 cold; ~0.62 when populated | ~0.62 is a **composite truth signal, NOT "62% grounded"**. The native_reasoning specialist is intentionally **unbuilt** (must earn it); qwen still does all existential reasoning. |

### Skills / acquisition / self-improve
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Skill acquisition stack (registry + jobs + acquisition + contracts + …) | shipped / live | `skills/registry.py:221`; `acquisition/orchestrator.py:89` | 14 bootstrap skills, 0 active jobs; jobs at `awaiting_operator_approval` | 14 verified ≠ 14 earned (**13 bootstrap, only `web_scraping_v1` pipeline-earned**). `SKILL_LEARNING_COMPLETED` = process evidence, not operational. awaiting-approval = waiting, not deadlocked. |
| Self-Improvement Orchestrator + System Upgrade Report | **gated** / advisory | `self_improve/orchestrator.py:1` | not constructed OR stage 0/FROZEN, 0 promotions | Frozen + disabled-by-default (`enable_self_improve=False`, `FREEZE_AUTO_IMPROVE=true`) is the **SAFE default**, not broken. |

### Capability Domains (Matrix v2)
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Capability Domains (registry + ingest firewall + store + topic-recall) | shipped / active | `cognition/capability_domains/registry.py:32`; `recall.py:60` | `count=0`, `registry.json` absent, recall returns None every turn | `count:0` = healthy default (operator-created), not "Matrix-v2 absent". The topic-recall wire IS live; the honesty boundary (**know-about ≠ can-do**) is the authority gate. |

### Perception / vision / spatial
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| PerceptionOrchestrator + SceneTracker + addressee + STT + voice-policy | shipped / active | `perception_orchestrator.py:166`; `perception/addressee.py:130` | no Pi → 0 persons, empty scene_context | The live consume-wire (not "just glue"). AddresseeGate can KILL a turn (early return) — a dropped overheard utterance is the gate working. |
| Vision Grounding Firewall + describe_scene VLM | shipped / live | `reasoning/context.py:500`; `tools/vision_tool.py:30` | "can't see" / "focusing my vision" (VLM cold-load) | An honest can't-see is the firewall + VRAM swap (qwen3-vl↔qwen3:8b share 16GB), never a guessed scene. |
| Spatial/LIDAR stack (room-model + fusion + Tier-3 depth + scene-graph + episodic + memory-gate + mind) | shadow / **none** | `cognition/lidar_room.py`; `lidar_depth.py:83`; `spatial_memory_gate.py:1` | empty room model, refused anchors (`valid=False`), empty album | Telemetry-only/firewalled — JARVIS does **not** "see in 3D" cognitively. The ~7.8cm-vs-walls figure is a proof-run, not a runtime sense. SpatialMemoryGate is the only spatial→memory path and is **un-wired by design**. |

### Memory
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Memory stack (storage + search/recall + ranker + salience + vector + episodes + fractal + gate + golden-words + consolidation + persistence) | shipped / live | `memory/storage.py:38`; `search.py:119`; `memory/gate.py:33` | rebuilds from disk (0 if fresh); ranker/salience cold→heuristic; "No memories found" on thin store | Near-empty store / "No memories found" = the honest **fix**, not broken. **Recent = recent-by-TIMESTAMP, not append order** (the fixed bug). Recall relevance = cosine sim, NOT weight (the "all over the place" fix). FractalRecall is pure shadow. |

### Dream / sleep
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Dream/Sleep stack (cycle + consolidation + artifacts + reflective-validator/audit + counterfactual) | shipped / active | `consciousness/consciousness_system.py:2357`; `memory/consolidation.py:41` | actively-used brain may NEVER dream (needs 300s sleep first); 0 cycles cold | Dream **artifacts** are firewalled OUT of recall (`dream_observer` provenance, -1.0). Empty dream history post-restart is **ephemeral by design**. audit score 1.0 = health, not "not running". |

### World model
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| MentalSimulator + WorldPlanner + CognitivePlanner + SimulatorPromotion | shadow / **none** | `cognition/simulator.py:164`; `planner.py:286`; `promotion.py:241` | simulator level 0, validated 0, planners `enabled=False` (`simulator_not_advisory`) | They **exist and run every cycle** — not "missing". `avg_confidence` is the sim's own projection (`is_measurement=False`), NOT accuracy. Reaching advisory would NOT auto-inject summaries (consumer unbuilt). |

### Personality
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Personality stack (soul + trait-evolution + modulator + validator + rollback + voice-dial) | partial / live | `personality/evolution.py:77`; `personality/traits.py:70`; `reasoning/context.py:58` | soul_dims at 0.4-0.7 seeds, relationships={} (stranger), persona seed always present | **TraitModulator is the ONE live wire** (flat names + dominant nudge + token/temp). Grown soul_dim **magnitudes are SHADOW** (`sent_to_model=False`) — the master gap. Seeds aren't "broken"; they move only on conversation. |

### Routing / voice leash
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Routing + voice leash (tool-router + voice-intent NN/shadow + golden-words + honesty firewall + reflective-detector + self-report-mode + prompt-builder) | shipped / live | `reasoning/tool_router.py:1330`; `reasoning/context.py:241,540`; `reasoning/intent_shadow.py:232` | router live from boot; corrections empty; IntentShadowRunner pass-through (0 rewrites) | The **regex router is the live driver AND teacher**; the voice-intent NN is a gated student that drives nothing. Don't loosen the honesty/grounding leash to "fix" a humble answer. Self-report fails CLOSED into INTROSPECTION. |

### Conversational pipeline
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Barge-in + endpointing/VAD (+ deferred streaming-ASR, speaker-gating) | shipped / live | `perception/audio_stream.py:317`; `config.py:155` | armed from boot; `fired_count=0`/`peak_rms=0` until someone talks over | Idle counters ≠ broken. Endpointing ships **`silence_duration_s=1.5`** (code authoritative; the 0.8s in memory is stale). Streaming ASR is roadmap-only; energy-only barge-in firing on any voice is a documented deferral. |

### Dashboard
| Name | Status / Authority | Home | Expected idle state | Common misread |
|---|---|---|---|---|
| Dashboard (v1 + snapshot cache + v2 28-page cockpit + integrity/grounding panels + ~30 APIs) | shipped / advisory | `dashboard/app.py:326`; `snapshot.py:627`; `static/v2/shared.js:26` | cache `{}` for ~2s after boot (`/api/self-test` 503); then honest empty/level-0 sub-dicts | Boot-window 503 / empty snapshot ≠ broken (cache-not-built). The dashboard **observes**, it doesn't drive the soul (only operator API-key'd POSTs change state). `UNKNOWN`/dim render = the toolkit refusing to invent green, not a fetch bug. |

---

## Known real gaps (from registry)

These are genuine, code-confirmed gaps — *not* idle states. A future agent may build these; do not mistake them for regressions.

1. **No competitive Global Workspace at the kernel level.** EventBus is unconditional multicast (no ignition/slots/winner). The only GNW competition is hemisphere broadcast slots, whose consumer (policy NN) is gated+signal-failed — so no live competitive bottleneck on behavior.
2. **Policy NN is a measurement-failure, structurally.** Reward carries no causal content (Spearman ~0.06); the 20-dim state misses conversation content. The imitation paradox makes the gate uncrossable. `DEVIATION_BONUS` the strategy doc says to delete is still applied (`evaluator.py:247`).
3. **Affect is built+wired end-to-end but DORMANT at level 0** — the "valence as driver not readout" gap. Only the nickname firewall (a guard) touches the voice. Promotion is operator-earned and not done.
4. **Grown personality MAGNITUDES never reach the voice.** Only flat trait names + a dominant nudge are injected; the graded `soul_dims→voice` dial is shadow (`sent_to_model=False`).
5. **Scene-permanence → memory/curiosity consume-wire is UNBUILT** (SceneTracker by contract writes no memory / triggers no curiosity).
6. **SpatialMemoryGate (only spatial→canonical-memory path) is not wired live** (test-only; scene-graph/album route around it by design).
7. **Companion read fires a turn TOO LATE.** Only `situational_read` P0 acts, POST-reply. TBS-0 fixes timing but injects nothing. P4 (steer reply) / P5 (learn from correction) are DESIGN-ONLY; `behavior_advisory.applied` hardcoded False.
8. **Spark grounding loop is OPERATOR-PULL and input-starved.** Without David answering `/v2/grounding`, the drive, P5b research, and win-rate promotion stay dormant. Tension-seeded thought emits nothing (level 0).
9. **thought_trigger_selector has a cross-module dim mismatch** (input_dim=24 vs FEATURE_DIM=44) AND its teacher signal is a reserved/unemitted event. The `claim_classifier` friction sub-feed (`capability_gate.py:1163`) is still DEAD.
10. **weight_room_gate `code_quality` classification is STALE** (`BLOCKED_BY_DESIGN` "no live signal" though the upgrade_verdict feed IS recorded live at `system_upgrade_report.py:346`).
11. **Two DEAD L10 context-reads** (`consciousness_system.py:2264/2339`) silently default to 1.0 — could mask low integrity in any consumer reading them (the real repair authority is the event path).
12. **Document Library / Study / Blue Diamonds knowledge tier is architecturally uncovered** — and it is the actual knowledge-retention spine that survives a reset (Blue Diamonds live OUTSIDE `~/.jarvis`). Load-bearing for the "GROWS across resets" claim.
13. **Autonomy research-engine TRUNK uncovered** (AutonomyOrchestrator ladder + ResearchGovernor + OpportunityScorer + DeltaTracker + KnowledgeIntegrator + InterventionRunner + SourceLedger) — only the satellites were mapped.
14. **L3 Escalation + Attestation governance uncovered** — the operator-write-authority boundary over self-modification (EscalationRequests, AttestationLedger hash-seeding, AuditLedger). The earn-don't-declare firewall for *granting authority* lives here.
15. **Unified World Model + CausalEngine trunk uncovered** while leaves (simulator/planners) are covered; WorldModelPromotion (3-level, the only world-model prompt-injection path at `context.py:1001`) underrepresented.
16. **Observer Effect / KernelMutator / MutationGovernor uncovered** — the self-modifying-kernel safety spine (bounded real write authority; single gate validate→risk→cooldown→snapshot→apply→rollback).
17. **Conversation Handler (278KB) uncovered** despite BEING the lived experience (golden-command → 5-tier router → overrides → context → response → release-validation → intention registration).
18. **Process Supervisor + HardwareProfile portability layer uncovered** (`jarvis-supervisor.py` restart/rollback authority + crash-loop ceiling; profile auto-selects models/devices per VRAM tier — why the same code runs Pi + RTX-4080).
19. **ProactiveGovernor + CuriosityQuestionBuffer uncovered** — the ONLY unprompted-speech paths, genuinely wired-to-voice; core to the "companion you want to be with" north star.
20. **Identity Evidence Accumulator + IdentityFusion + enrollment stores uncovered** — the anti-spoofing gate that produces the trusted identity L3/L3A assume.
21. **Personal Intel Capture (HUMINT, `conversation_handler.py:2284`) uncovered** — learns who the person is, with the banter firewall + preference retirement; real write-authority into the relationship record.
22. **Codebase self-awareness tool (`tools/codebase_tool.py`) uncovered** — AST index + WRITE_BOUNDARIES allowlist; grounds codebase Q&A and enforces self-improve file writes.
23. **Academic Search Tool (Semantic Scholar + Crossref) uncovered** — the poisoning-resistant default retrieval for all autonomous research; the concrete provenance the L2/anti-confab story depends on (no DDG fallback on the scholarly lane).
24. **Explainability layer (`reasoning/explainability.py`) uncovered** — builds user-facing "why/where from" traces on the live response path; the consume-wire turning L1 substrate into transparency.
25. **Existential/Philosophical/Epistemic reasoning engines uncovered** — kernel-scheduled generators of inner-life KERNEL_THOUGHTs under an hourly token budget.
26. **HealthMonitor (`consciousness/health_monitor.py`) uncovered** — the formal 5-dim weighted self-assessment with trend prediction (distinct from the dashboard passthrough).
27. **Layer-3B Display Classifier (`perception/display_classifier.py`) uncovered** — masks monitor-interior detections so "enemy on screen" never becomes "enemy in room" (a perceptual-truth firewall).
28. **Phase-C/D/E native Language substrate (`shadow_language_model.py`, default-OFF) uncovered** — the "LLM becomes the voice not the brain" endgame muscle; pre-mature/shadow.
29. **Long-Horizon Attention (P3.12) is PLANNED/ABSENT by design** (`docs/LONG_HORIZON_ATTENTION_USE_CASE.md`, no code, 6 unmet preconditions). One slot so a future agent neither flags its absence nor re-designs it.
30. **Streaming ASR (Tier-2) ABSENT** (roadmap, `config.py:165`); **barge-in speaker-gating PLANNED**. Honest deferrals, not built-but-broken.
31. **Trace Explorer cockpit (`dashboard/app.py:4487+`) uncovered** — reconstructs causal chains from the immutable ledger.

---

## Cross-area contradictions / open questions

- **L9 cadence/label.** Verifiers split on shipped/advisory (120s, all waking modes) vs shadow (300s, dream-only). **Resolved:** shipped/advisory — it genuinely runs and feeds SoulIntegrity's `audit_score`; but its findings **auto-apply nowhere** (the behavioral "drives nothing" point stands).
- **L4 headline authority** differs by which arm is emphasized: OutcomeScheduler (live/active) vs CounterfactualEngine (shadow/dormant, `live_influence=False`). Not a true contradiction — two arms of one layer.
- **ProvenanceScorer** appears as authority `none` (the view-only reader) and `advisory` (when described via its gated downstream consumer). Same module, two framings; the reader drives no lever.
- **`reflective_audit` docstring says "6 dimensions"; the code has 9 scanners.** Code authoritative.
- **`AGENTS.md:595` says "11 layers (0–11 + 3A + 3B)" and omits L12**, contradicting the canonical 15-entry literal at `docs_truth_audit.py:425` (and `AGENTS.md:850`'s "0-12"). Literal authoritative.
- **`GroundingQueue.answer()` docstring says "view-only / does NOT mutate"** but the code (`grounding_queue.py:469`/`_ground_belief`) DOES re-stamp provenance→user_claim. **Trust the code** — the ring has one live mutating path.
- **`weight_room_gate` `code_quality` reason is stale** vs the live `upgrade_verdict` feed (`system_upgrade_report.py:346`). A real governance-metadata staleness.
- **self_sensing "+0.463" (memory/breakthrough log) vs "volatile/STARVED" (live):** resolved to volatile/STARVED-is-normal — the +0.463 was a first-30-min snapshot.
- **Endpointing 0.8s (memory) vs 1.5s (`config.py:166`, with revert comment):** code authoritative.
- **intention-delivery-encoder:** test-only/unwired (imported only by `tests/test_intention_resolver.py`), not live in Stage-1.
- **`spark_metrics` docstring "3.3x" vs `SPARK_BASELINES` 20.5x:** docstring stale; 20.5x is the belief-graph baseline (3.3x is the separate memory-store figure).

---

## How to use this manifest (for agents)

1. **Before flagging anything broken**, find its row, read `expected_idle_state`, and decide: shipped / shadow / dormant / gated / signal-failure / partial. After a reset, **empty/0.0/None/level-0 is almost always normal.**
2. **The integrity stack is 15 layers** — verify against `brain/scripts/docs_truth_audit.py:425`, not against any prose that says 6/11/13.
3. **Self-scored ≠ proven.** Transcendence, awareness, soul-integrity dims, simulator `avg_confidence`, reasoning-encoder ~0.62, scoreboard composites are internal composites, not external tests. Never tune them green.
4. **Shadow means injects-nothing.** "advisory" authority on a companion/affect/policy/world-model subsystem does NOT mean it steers behavior — trace whether the return is consumed. Most are discarded by design.
5. **Never code-to-qwen, never lower a gate, never delete a firewall, never wire a shadow output into the prompt to light a panel.** That is the exact failure the architecture resists. Promotion is operator-earned on real reps.
6. **Voice is a swap-in LLM on a leash.** The honesty firewall + grounding leash + per-route caps are the model-agnostic boundary. Fixes to "humble" answers usually belong in routing/grounding, not in the model.

## Round-2 additions — previously-uncovered pillars

49 adversarially-verified pillars added (2 of the original 51 deduped against existing subsystems: ToolRouter 5-tier -> `routing-voice`; Streaming ASR Tier-2 -> `conversational-pipeline` deferred lane). New total subsystem count: 98.

### Knowledge-retention spine (the cross-reset knowledge backbone)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Document Library (Sources + Chunks + Index) | shipped | live | brain/library/source.py:88 | EMPTY after reset, refilled from Blue-Diamonds reload + codebase ingestion; reset-VOLATILE | mistaking it for the durable store (it is reset-wiped; only Blue Diamonds survive) |
| Study Pipeline (concept + claim extraction) | shipped | live | brain/library/study.py:203 | cycle runs every 120s, returns immediately when no unstudied sources; session counters reset per process | reading telemetry zeros as 'dead'; assuming study claims get the +0.10 trust boost |
| Blue Diamonds (durable reset-surviving archive) | shipped | live | brain/library/blue_diamonds.py:163 | empty-until-earned; high gates mean rejections >> graduations even on a box that ran the brain | flagging empty as broken; looking inside ~/.jarvis (it is at ~/.jarvis_blue_diamonds) |

### Autonomy research-engine trunk + satellites
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| AutonomyOrchestrator (L0-L3 + queue, MAX_QUEUE_SIZE=20) | partial | active | brain/autonomy/orchestrator.py:82 | live at level=1; conversational mode blocks ALL research; L2/L3 actuation inert by design | calling it shadow/broken (L0/L1 are live; only L2/L3 actuation is gated) |
| ResearchGovernor (mode/rate/topic/budget gate) | shipped | live | brain/autonomy/research_governor.py:70 | blocks every intent in conversational/focused mode (mode_blocked = healthy) | reading high total_blocked as broken |
| MetricTriggers (sustained-deficit driver) | shipped | live | brain/autonomy/metric_triggers.py:116 | total_triggers=0 = healthy/no deficits; escalation candidates [] below live L3 | flagging 0 triggers as 'never runs' |
| OpportunityScorer (composite ranking) | shipped | live | brain/autonomy/opportunity_scorer.py:98 | total_scored=0 after reset; existential intents score low (0.15) by design | thinking low existential scores are broken; conflating with the policy NN |
| DeltaTracker (counterfactual credit) | shipped | live | brain/autonomy/delta_tracker.py:99 | delta_pending.json empty/absent until jobs measure | using raw net_improvement instead of net_attribution |
| KnowledgeIntegrator (real memory/library write path) | shipped | live | brain/autonomy/knowledge_integrator.py:198 | small library + web_scrap entries (firewall working) | assuming all research output is trusted; mislabeling it shadow |
| InterventionRunner (propose->shadow->promote) | shadow | advisory | brain/autonomy/intervention_runner.py:40 | empty queues; shadow_active up to 24h is normal | reading 'promoted' as 'changed its own code' (code-patch arm inert) |
| SourceUsefulnessLedger | shipped | live | brain/autonomy/source_ledger.py:72 | usefulness_rate 0.0 / all-pending until 24h verdicts; neutral 0.5 default | reading 0.0/pending as broken |
| InternalQueryInterface (scholarly-first, no-DDG) | shipped | live | brain/autonomy/query_interface.py:37 | empty scholarly result rather than open-web scrape | expecting free autonomous web search |
| AutonomyPolicyMemory (outcome credit + L2/L3 ledger) | shipped | live | brain/autonomy/policy_memory.py:89 | empty after reset; eligible_for_l2/l3=False until reps accrue | conflating it with the signal-failure policy NN |

### Operator-Write-Authority boundary (the authority-governance firewall)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| EscalationStore (L3 escalation queue) | dormant | none | brain/autonomy/escalation.py:204 | dormant at default L1; no requests generated; persistence files absent | reading empty queue as unbuilt; assuming attestation unlocks it |
| L3 Escalation Request Queue (human-gated channel) | shipped | live | brain/autonomy/escalation.py:268 | zero escalations at default L1 (double-gated on live L3 + exhausted L1) | calling the channel dead; assuming attestation can trigger it |
| AttestationLedger (hash-attested ever-proven seed) | shipped | advisory | brain/autonomy/attestation.py:213 | ledger file absent -> prior_attested_ok=false -> request_ok falls back to current_ok | thinking prior_attested_ok bumps ever_*/maturity; thinking only hash_verified counts |
| AutonomyAuditLedger (durable transition trail) | shipped | none | brain/autonomy/audit_ledger.py:69 | wired=True; JSONL absent/tiny until a transition fires; empty != broken | concluding it 'isn't wired'; hunting for a control-path consumer |

### Unified world model + maturity gate
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| WorldModel trunk | shadow | advisory | brain/cognition/world_model.py:47 | level 0; update_count climbing every 5s; build_context_summary computed but not injected | calling the un-injected SA block broken; expecting conversation-driven population |
| CausalEngine (heuristic prediction rules) | shadow | live | brain/cognition/causal_engine.py:354 | runs every tick; predictive_accuracy_live=0.0 until lived transitions | conflating overall_accuracy with foresight; calling it dead at 0.0 live |
| WorldModelPromotion (3-level gate) | gated | live | brain/cognition/promotion.py:52 | level 0 for hours after reset (4h + 50 preds + 0.65 acc) is correct | treating level-2 'active' as shipped; calling level 0 broken |

### Self-modifying-kernel safety spine
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Observer Effect / ConsciousnessObserver | shipped | live | brain/consciousness/observer.py:166 | awareness_level==0.3 floor; zero memory nudges right after boot | calling 0.3 stuck; trying to let it create/rewrite memories (the firewall) |
| KernelMutator (analyzer + proposer) | shipped | advisory | brain/consciousness/kernel_mutator.py:473 | proposes nothing until ticks>=100 AND memories>=20 AND 180s | treating zero proposals as dead; assuming proposals = changes |
| MutationGovernor | partial | live | brain/consciousness/mutation_governor.py:82 | rejects ALL mutations for the first 600s + 3/hr cap for 2h | tuning caps green; missing that the call site omits awareness_level (gate bypassed) |
| ConsciousnessHealthMonitor | shadow | advisory | brain/consciousness/health_monitor.py:68 | 'unknown'/0.5 on empty history; does NOT gate mutations | reading 0.5 as a health problem; assuming it gates mutations |

### Lived-experience spine (dispatch + unprompted speech + relationship learning)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Conversation Handler dispatch spine | shipped | live | brain/conversation_handler.py:2425 | intention registry empty; release-validation passes trivially | treating the huge file / swallows as dead code |
| ProactiveGovernor + ProactiveBehavior | shipped | live | brain/personality/proactive.py:104 | mostly silent (daily greeting); silence is correct | calling it broken because it rarely speaks; assuming engine.py path runs |
| Soul-Question / philosophical dialogue system | dormant | advisory | brain/personality/proactive.py:271 | permanently silent: memory_density never passed + answer-handler uncalled | claiming 'wired, just memory-gated' (it is unreachable) |
| CuriosityQuestionBuffer | shipped | live | brain/personality/curiosity_questions.py:218 | empty/0 after reset; starved by upstream gates, not broken | reading empty buffer as 'curiosity dead'; counting 5 live sources (4 + 1 orphan) |
| Curiosity question generators (4 live + 1 orphan) | dormant | advisory | brain/personality/curiosity_questions.py:436 | all generators return nothing until each subsystem matures past its gate | grepping a generator and missing the call-site gate |
| Personal Intel Capture (HUMINT + banter firewall) | shipped | live | brain/conversation_handler.py:2284 | writes nothing until a regex hit; empty profile is correct | assuming all matches become trusted beliefs; counting only 4 pipelines (5 exist) |

### Identity-trust + perceptual-truth firewall
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Identity Evidence Accumulator | shipped | live | brain/identity/evidence_accumulator.py:136 | 0 persistent identities after reset; trust is earned | calling it broken / force-promoting; reading the score as a percentage |
| IdentityFusion (voice+face arbiter) | shipped | live | brain/perception/identity_fusion.py:135 | 'absent'/'unknown'/method='none' with no one present | reading firewall verdicts as bugs; assuming it mints durable trust |
| SpeakerID store (ECAPA-TDNN) | shipped | live | brain/perception/speaker_id.py:37 | 0 profiles / is_known=False pre-enrollment; available=False if no speechbrain | treating available=False as broken vs an unmet optional dependency |
| FaceID store (MobileFaceNet/ONNX) | shipped | live | brain/perception/face_id.py:40 | 0 profiles / disabled until onnxruntime + mobilefacenet.onnx present | flagging available=False when the model file just isn't downloaded |
| Display Classifier + display-interior masking | partial | live | brain/perception/scene_tracker.py:138 | nothing to mask / empty classifications with no display in view | assuming the whole pillar is shadow (the masking is the live firewall) |
| Identity Name Validator | shipped | live | brain/identity/name_validator.py:63 | stateless; silently rejects junk tokens | reading 'invalid name' debug logs as identity broken |

### Grounding tools + user-facing provenance
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Codebase Self-Awareness Tool | shipped | live | brain/tools/codebase_tool.py:142 | empty index until boot/lazy build; hashes file absent on first run | seeing 0 symbols and calling it broken |
| Academic Search Tool (S2 + Crossref) | shipped | live | brain/tools/academic_search_tool.py:781 | cold cache / anonymous pools without keys; realtime fallback | treating an empty cache or missing S2 key as broken |
| Explainability Layer ('why this answer') | partial | active | brain/reasoning/explainability.py:387 | compact_trace fallback_unclassified on empty seed | treating fallback as a bug; assuming the whole layer is live (rich trace unwired) |

### Inner-life thought generators (synthetic-soul monologue)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Existential Reasoning Engine | dormant | advisory | brain/consciousness/existential_reasoning.py:162 | zero inquiries until capability OR transcendence>=0.5 / deep_learning | calling it dead; reading enable_existential_reasoning=True as 'running' |
| Philosophical Dialogue Engine | dormant | advisory | brain/consciousness/philosophical_dialogue.py:194 | empty history until transcendence>=1.0 (higher bar than existential) | assuming it shares existential's 0.5 gate |
| Epistemic Reasoning Engine (causal models) | shadow | advisory | brain/consciousness/epistemic_reasoning.py:365 | empty _chains (cleared on cleanup event, rebuilt from live feeders) | reading _chains.clear() as self-wiping/broken |

### Native-language endgame (LLM becomes the voice, not the brain)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Shadow Language Model (Phase C) | shadow | advisory | brain/reasoning/shadow_language_model.py:70 | available=False / no-op until corpus>=50 | expecting it to generate text or replace the LLM |
| Runtime Bridge (Phase D guard) | gated | advisory | brain/reasoning/language_runtime_bridge.py:21 | default enabled=False, rollout_mode='off' (fail-closed); blocks all native output | reading default 'off' as a bug or unbuilt |
| Language Kernel registry (Phase E P1.5) | dormant | advisory | brain/language/kernel.py:161 | get_live_artifact()==None (PRE-MATURE) on a fresh brain | mistaking it for a live model or auto-promoter |

### Runtime / safety net / observability
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Process Supervisor (jarvis-supervisor.py) | shipped | live | brain/jarvis-supervisor.py:292 | restart_intent/pending_verification files ABSENT (transient only); no rollback = healthy | treating absent files as 'rollback never built'; --once is a debug bypass |
| HardwareProfile (portability layer) | shipped | live | brain/hardware_profile.py:437 | tier='minimal'/gpu='none' on a Pi is CORRECT | reading low-tier on a CPU box as broken GPU detection |
| Trace Explorer Cockpit | shipped | live | brain/dashboard/snapshot.py:30 | empty snapshot (entry_count:0) until ledger entries exist | calling empty root_chains broken (mirrors an empty ledger) |

### Planned pillar (no code)
| name | status | authority | home | expected idle state | common misread |
|---|---|---|---|---|---|
| Long-Horizon Attention | planned | none | docs/LONG_HORIZON_ATTENTION_USE_CASE.md:1 | no runtime artifact at all — use-case pinning only | treating the 'autonomy_long_horizon' eval scoreboard hits as the implementation |
