# Matrix Protocol v2 — Capability Domains

> **Status:** DESIGN (not built). Authored 2026-06-07, branch `thespark`. GitHub epic: TBD.
> **One-line:** *A user can say "JARVIS, learn to do X — here are the files/schematics," and JARVIS spins up an **isolated, deletable Capability Domain**: its own neural sub-consciousness + knowledge + memory + capability-envelope model + (optional) act→verify→reinforce loop — that it can honestly reason about, feel grounded affect about, creatively push the limits of, and forget cleanly if X goes away.*

This document is the durable home of the vision so we do not lose it. It is **validated against the current codebase** — every claim is tagged ✅ exists / 🟡 extend / ❌ new — so we build on what is real and never re-invent or fake.

---

## 0. Why this exists (the vision, in David's words, distilled)

- "JARVIS, learn how to control this robot arm. Here is the model and schematics." → JARVIS learns, then **continuously** learns + observes (synthetic reps between real reps).
- A **real closed loop**: see the arm → see a block → move it → **verify in the camera it moved** ✓ → return it → **verify** ✓ → "I am getting more proficient."
- **Expandable to any possibility.** Not everything needs a schematic — sometimes you just **feed JARVIS a folder of files (PDF / markdown / text)** and it learns the domain.
- **Isolation is mandatory:** all of a domain's knowledge/memory/NN lives in the domain it created. **Delete the robot arm → zero impact on JARVIS.** No polluting core memory or other domains.
- **Knowing limits = proficiency:** JARVIS learns the **capability envelope** (DOF, reach, torque, range, design intent) so it knows what it *cannot* do. Asked to "move the block 50 feet," it knows that exceeds the arm's reach.
- **Grounded affect:** frustration/curiosity are the **honest readout** of the gap between a request and the envelope — not theater. They *motivate* behavior (frustration → workaround search; curiosity → research, bounded to what the user provided).
- **Inner dialogue + creative workaround:** "user wants 25ft, my reach is X — how can I meet this?" → simulate options → maybe "overclock the drives and throw the block" — **proposed with its risk, human authorizes; never silently done.** When truly impossible: "you need new equipment."
- **Each domain NN becomes the *main sub-consciousness* of that capability** — coordinated by, never merged into, the core.

**Non-negotiables (David, 2026-06-07):** no crossed wires, no AI theater, no polluting JARVIS, no bypassing any existing safety/detection net. v2 must be **solid** precisely because the space of things it can be asked to learn is unbounded.

**Repeatable ask (David, 2026-06-07):** the same request must always run the **same strict, governed, auditable training pipeline** and produce a reproducible, isolated result. No bespoke one-off training per ask. See §4.5 The Training Contract.

**A pillar of consciousness — honestly framed (David, 2026-06-07):** the Matrix Protocol is another *pillar* of the consciousness substrate. **This does NOT mean JARVIS is or will be conscious.** It means: *if* a digital consciousness existed, it would plausibly have a feature like this — among thousands. We build the feature and the substrate; we **never declare the consciousness** (never declare, never discard — §4.9). The analogy that anchors it: a human can hit their head and completely forget how to snowboard while remembering everything else, because **specialty skills are heavily linked/associated to their own cluster of information.** v2 makes that literal — see §2.5.

---

## 1. Relationship to Matrix Protocol v1

| | v1 (today) | v2 (this doc) |
| --- | --- | --- |
| Scope | 5 **internal cognition** specialists (`positive_memory`, `negative_memory`, `speaker_profile`, `temporal_pattern`, `skill_transfer`) | **user-defined Capability Domains** (robot arm, a medication regimen, a codebase, …) |
| Trigger | (was meant to be) internal signal | user request + provided materials, and/or autonomous |
| Isolation | shares hemisphere `_networks` | **fully isolated domain bundle**, deletable |
| Lifecycle | `CANDIDATE_BIRTH → PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY → BROADCAST_ELIGIBLE → PROMOTED` (4 gates, coded) | **same ladder, reused + hardened** |

**Critical dependency / proving ground:** v1's Tier-2 lane is currently **dormant end-to-end** (traced 2026-06-07): `create_probationary_specialist` has **zero production callers** (birth never happens), and there is **no training feed** for matrix focuses (`engine.train_distillation` serves Tier-1 only). The per-focus tests pass only because they **flip lifecycle state by hand**. See [[matrix-protocol-verification]] / GitHub #26-adjacent. **v2 Phase M (below) wires + verifies that lifecycle for real — it is the proving ground the whole of v2 stands on.** We do not build v2 on an unproven ladder.

---

## 1.5 The hard line — Skill vs Matrix vs Library (RATIFIED 2026-06-07)

Drawn now, before building, so the lanes never cross-wire. **Grounded in the existing code**, which already classifies every capability as `procedural` / `perceptual` / `control` (`brain/acquisition/classifier.py`, protocols SK-001/002/003) and already has a physical-hardware detector regex (`gpio|relay|motor|servo|sensor|actuator|robot|robotic|pi5|raspberry|arm control`).

| Lane | What it is | Maps to | Examples |
| --- | --- | --- | --- |
| **Skill** | a discrete, bounded **tool JARVIS wields** — digital, executable, verified once | `procedural` (SK-001) | `web_scraping`, send a text, call an API |
| **Matrix Protocol (Capability Domain)** | an isolated, continuously-learning **sub-consciousness JARVIS reasons *with*** — **anything that senses or actuates the physical world** | `perceptual` + `control` (SK-002/003) + the hardware regex | robot arm, LCD1602, DHT11, ultrasonic sensor |
| **Library / Study** | pure **reference knowledge**, no action | study/library pipeline | "learn about Roman history" |

**THE BRIGHT-LINE RULE:** *if a capability senses or actuates the physical world, it is ALWAYS Matrix — never a Skill.* Physical interaction forces Matrix because only Matrix carries the capability-envelope model, grounded affect, the embodiment act→verify→reinforce loop, continuous lived+synthetic training, and the action-safety/authorization gates.

- **Document ingestion is NOT its own lane** — it is the *knowledge-intake step inside* a Matrix domain (feeding the arm's schematics builds that domain's envelope). Pure reference knowledge with no physical capability stays in **Library/Study**.
- **The dovetail:** physical Matrix domains are exactly where **world-predictive modeling** plugs in — the envelope + MentalSimulator predict an action's outcome, the world model supplies physics/causal priors, lived camera/sensor-verified outcomes feed back. Skills (digital tools) never need this, which is why the line keeps them cleanly separate.

This taxonomy is also summarized in `MATRIX_PROTOCOL_GUIDE.md`.

## 2. The Capability Domain abstraction (the core new object)

A **Capability Domain** is a self-contained, deletable bundle:

```
CapabilityDomain {
  domain_id            # stable id, e.g. "dom_robot_arm_xarm6"
  name, created_at, status (lifecycle stage)
  knowledge_store      # isolated doc/chunk index (extends library)        🟡
  memory_namespace     # isolated memory scope (extends identity boundary) 🟡
  sub_consciousness    # the domain NN ("main sub-consciousness")          🟡 (hemisphere NN, isolated)
  envelope_model       # parametric capability constraints                 ❌ new
  synthetic_trainer    # per-domain synthetic reps (extends weight room)   🟡
  action_loop          # optional: actuator + camera-verify + proficiency  ❌ new (embodiment)
  provenance_ledger    # every fact/rep tagged lived|synthetic|ingested    ✅ reuse
}
```

**Hard isolation invariant:** nothing a domain learns may write into core memory, core beliefs, or another domain. Recall *crosses* the boundary only when the domain's topic is explicitly active (the "I know about the robot arm" trigger). **Delete = drop the bundle = zero residue.** This is the anti-pollution net.

### 2.5 Skill clusters & the brain-injury analogy (why isolation is *also* a fidelity property)

A Capability Domain is a **heavily-associated, isolated skill cluster** — the NN + its knowledge + its memories bound tightly together, and loosely (only via the topic trigger) to everything else. This mirrors how specialty skills are neurologically clustered: *a person can hit their head and forget how to snowboard while retaining everything else, because the snowboarding skill lives in its own densely-linked cluster.*

This makes isolation not just a safety net but a **fidelity property** with two consequences the build must honor:

- **Intra-cluster association must be STRONG.** When the domain's topic arises, recall must surface the *whole* cluster (the arm's parameters, its docs, its lived reps) together — "I know how to operate the robot arm" pulls the entire associated body of knowledge, not scattered fragments.
- **Inter-cluster bleed must be ZERO.** A domain's learning must never leak into core or sibling clusters. Training a new skill cannot degrade an existing one (no catastrophic interference across domains — each NN is separate, so this is structural, not hoped-for).
- **Clean ablation = clean forgetting.** Deleting a domain is exactly "forget snowboarding, keep everything else": drop the cluster, zero residue, zero impact elsewhere. The deletion test (Phase 1) *is* the brain-injury analogy as an automated guarantee.

JARVIS already has a `memory_clusters` concept (`~/.jarvis/memory_clusters.json`); a Capability Domain is a **deliberate, governed, deletable cluster** built on that intuition.

---

## 2.6 The Pi5 — JARVIS's body & sense substrate (parallel prerequisite)

The Pi5 is **JARVIS's body**: it is the `pi5-senses` node that streams perception (camera/edge-VLM, audio) to the brain. **Physical Matrix domains live on the Pi5's hardware** — every module a user plugs into it (LCD1602, DHT11, ultrasonic, a motor driver, the robot arm) is a new physical capability the Matrix Protocol can learn. **You cannot reliably learn to drive hardware you cannot see**, so the body needs an instrument panel.

**Pi5 Engineering / Operational Dashboard (its own milestone — built IN PARALLEL).** *Validated 2026-06-07: today the brain only receives coarse `sensor_health` telemetry (`perception/server.py::get_sensor_health`); there is NO pin/slot/module/GPIO/bus introspection and no hardware dashboard. This is genuinely new.* It must provide:
- **Hardware map:** pins/slots in use, what's connected where (GPIO/I²C/SPI/UART/USB), free pins.
- **Module identification:** when something new is plugged in → detect + identify it (by bus address / handshake / user confirmation).
- **Live signals & comms:** what signals are coming/going on each pin/bus, protocol/traffic view — so we **monitor instead of guess** when something is wrong.
- **Hookup recommendations:** JARVIS advises "connect this to that / use this pin" based on the module + what's already wired (envelope-aware — it knows the board's limits too).
- **Health/fault telemetry** per module (power, errors, disconnects).

This dashboard is the **substrate that makes physical Matrix domains practical and safe** — it is how the act→verify loop (Phase 7) and the envelope model (Phase 3) get *real* hardware ground truth, and how a workaround (Phase 8) is risk-assessed against actual wiring. **Dependency:** Matrix embodiment phases (3, 7, 8) require it for any *real* hardware; knowledge/sim phases (M, 0, 1, 2) do not.

**New sense modalities = body awareness (the key insight).** Sensors give JARVIS senses its camera does not have. The ultrasonic example: *"alert me if you detect something your camera eyes didn't see — like a human feeling someone standing behind them without looking."* That is **proprioceptive/spatial body-awareness** — a genuinely new perceptual channel, learned as a `perceptual` Matrix domain, that fuses into the situational read alongside vision/audio. Each such sensor expands JARVIS's *sensorium*, not just its toolbox.

**Worked examples (all physical → Matrix domains, each isolated/deletable):**
- "Matrix learn to send messages to the **LCD1602** module" → a `control` output domain.
- "Matrix learn to collect + report **temperature/humidity from the DHT11**" → a `perceptual` sensing domain.
- "Matrix learn to use the **ultrasonic sensor** and alert me to presence my camera missed" → a `perceptual` body-awareness domain that feeds the situational read.

## 3. What already exists (validated 2026-06-07 — reuse, don't reinvent)

| Capability | Where | Tag |
| --- | --- | --- |
| Document/file ingestion (PDF via pdftotext, folder batch, chunking, sqlite-vec, concept graph) | `brain/library/` (`ingest.py`, `batch_ingest.py`, `chunks.py`, `index.py`, `concept_graph.py`) | ✅ |
| Domain tagging on sources (`domain_tags`) | `brain/library/source.py` | 🟡 (tags, not isolated stores) |
| Code isolation (venv / sandbox / quarantine) | skill/acquisition lane; `PLUGIN_ISOLATION_DESIGN.md`, `CAPABILITY_AUTHORITY_DESIGN.md`, `CAPABILITY_LIFECYCLE_DESIGN.md` | ✅ |
| Synthetic training ("weight room": offshore, `origin="synthetic"` @ `fidelity=0.7`) | `brain/synthetic/*.py`; `WEIGHT_ROOM_DESIGN.md` | ✅ |
| Per-focus NN design/train + Tier-2 lifecycle ladder (4 gates) | `brain/hemisphere/` (`orchestrator.py`, `engine.py`, `architect.py`, `types.py`) | 🟡 (dormant — Phase M wires it) |
| Sub-orchestrator pattern (6 orchestrators coordinate) | `perception`, `hemisphere`, `acquisition`, `self_improve`, `learning_jobs`, `autonomy` | ✅ |
| Camera + on-demand VQA ("see + verify" half) | `tools/vision_tool.py::describe_scene`; VQA = GitHub #24 | 🟡 |
| Execute + verify harness | `skills/operational_bridge.py`, `execution_contracts.py`, `verification_protocols.py`, `job_eval.py` | 🟡 |
| Feasibility/refusal (primitive) | `skills/capability_gate.py` | 🟡 (→ parametric) |
| Workaround search: simulate over action sequences | CognitivePlanner (#16, `cognition/planner.py`) chaining MentalSimulator (`cognition/simulator.py`); Counterfactual (#17) | ✅ |
| Affect substrate | `consciousness/situational_read.py`, `theory_of_mind.py`, `tone.py`, `communication.py`, `existential_reasoning.py` | 🟡 (→ grounded in gap) |
| Curiosity-at-a-gap (the "spark") | grounding ring; `SPARK_DESIGN.md` | ✅ (shadow) |
| Maturity/asymmetric gates, provenance, OSV honesty, #9 scoreboard | governance map | ✅ |

**Estimate:** ~70% of the *reasoning* substrate exists. The genuinely-new builds are: **envelope model, affect-grounded-in-gap, the domain isolation bundle + clean deletion, the domain sub-orchestrator, and the embodiment act→verify→reinforce loop.**

---

## 4. Safety & Governance nets (NONE may be bypassed)

These are invariants enforced in *every* phase. A phase is not "done" if it violates one.

1. **Asymmetric gate (lived-before-synthetic).** Synthetic reps *amplify*; they **never** promote a domain or justify a proficiency claim alone. "I'm proficient" requires **real, camera/outcome-verified lived reps.** (`WEIGHT_ROOM_DESIGN.md`, [[weight-room-discipline]].)
2. **Honesty boundary (KNOW-not-guess, no AI theater).** "I *know about* X" (recall ingested knowledge) is allowed once ingested. "I *can do* X" / "I'm proficient" is **earned** through verified reps. Bound to OSV P2 grounding so the LLM voice cannot over-claim. (`MATRIX_PROTOCOL_GUIDE.md` core rule.)
3. **Capability authority gate.** A domain is **advisory** until its lifecycle/promotion gates pass; operational action needs a verifier + measurable proof. No broadcast authority, no plugin activation authority by birth alone. (`CAPABILITY_AUTHORITY_DESIGN.md`.)
4. **Action safety + human authorization.** Any real-world action (esp. envelope-pushing workarounds like overclocking) is **proposed with its risk + design-intent caveat; the human authorizes; JARVIS never silently acts.** Risk-classified; reversible-preferred; dry-run/simulate first.
5. **Isolation / anti-pollution.** Domain knowledge/memory/NN are namespaced; no writes to core or sibling domains; clean deletion = zero residue. Verified by a deletion test (Phase 1 acceptance).
6. **Sovereignty / bounded research.** A domain learns **only from what the user provides** (+ explicitly-allowed sources). No uncontrolled external scraping into a domain. (North star: sovereign Oracle, not corporate cloud.)
7. **Provenance everywhere.** Every fact/rep/outcome tagged `lived | synthetic | ingested` with `fidelity`. Promotion reads only the lived-firewalled signal (#9 pattern). (`fidelity-scoreboard-lived-firewall`.)
8. **Quarantine for generated artifacts.** Any generated control code / plugin runs sandboxed/quarantined before any authority. (Reuse skill-lane isolation.)
9. **Emergence: never declare, never discard.** Domain-level surprising behavior is captured via `observer.observe_emergence` (observation-only), never claimed.
10. **Shadow-first / earn-don't-declare.** Every phase ships shadow-observable first; authority/promotion is earned on evidence, gate-flipped, never assumed.

---

## 4.5 The Training Contract (repeatable, strict, auditable)

Because the space of asks is unbounded, **how** a domain NN is trained must be fixed, declarative, and reproducible — never improvised per ask. The same ask runs the same pipeline and yields the same governed, isolated result.

**A. Declarative training recipe.** Every domain stores a versioned `TrainingRecipe`:
```
TrainingRecipe {
  recipe_version          # bumped only by a reviewed change to the protocol itself
  domain_type             # "document_only" | "embodied" | ...
  data_sources[]          # ONLY user-provided / explicitly-allowed inputs (provenance-tagged)
  encoder/label_spec      # what the NN learns to predict (self-supervised: encoder value = label)
  topology_policy         # how the NN is sized for this domain_type (deterministic)
  schedule                # epochs, batch, lr, seed (FIXED + recorded)
  gates                   # the matrix ladder thresholds (unchanged, never per-ask)
  acceptance_evidence     # what counts as lived proof for promotion
}
```
The recipe — not free-form code — defines training. Two identical asks → identical recipe → reproducible NN.

**B. Strict rules (every training run):**
1. **Domain-scoped data only.** Training reads *only* the domain's isolated store + its lived reps. No core data, no sibling-domain data — enforced, not trusted (net §4.5/§2.5).
2. **Provenance-partitioned.** `lived` vs `synthetic` vs `ingested` kept separate; the **promotion signal reads the lived-firewalled split only** (net §4.7, #9 pattern). Synthetic amplifies, never promotes (net §4.1).
3. **Deterministic + seeded + recorded.** Fixed seed/schedule; every run logs inputs, data provenance, seed, recipe_version, metrics, outcome → an auditable `TrainingRun` record. Re-running reproduces it.
4. **Fixed gates.** The lifecycle thresholds are global and versioned; an ask can never lower its own bar to "pass."
5. **Idempotent re-ask.** Asking to learn the same domain again **updates/retrains the same domain** (new `TrainingRun`, same `domain_id`) — it never silently forks a duplicate or pollutes.
6. **No authority from training.** A completed run yields an *advisory* NN; authority is earned separately through the lifecycle + verifiers (net §4.3).
7. **Quarantine for any generated artifact.** Generated control/plugin code is sandboxed before any run touches the world (net §4.8).

**C. Reproducibility test (acceptance):** the same recipe + same inputs + same seed reproduces the same NN within tolerance, and the `TrainingRun` audit trail is complete. This is how we *prove* the ask is repeatable and not theater.

---

## 5. Phased build plan (validated, sequenced, each shadow-first)

> Legend per step: **[reuse]** existing · **[extend]** existing · **[new]** · **[gate]** a safety net it must honor · **[earned]** matures on real reps, not coded.

### Phase M — Prove the Tier-2 lifecycle (PROVING GROUND, prerequisite)
*Make the v1 ladder actually run before building v2 on it. Birth trigger = AUTONOMOUS (David's choice).*
- **M1** Wire **autonomous birth**: when a focus accumulates enough internal signal, `create_probationary_specialist` is called → `CANDIDATE_BIRTH`. **[extend]** orchestrator `run_cycle` new phase; **[gate]** cap `MAX_PROBATIONARY_SPECIALISTS`.
- **M2** Wire **training feed**: extend the distillation cycle to Tier-2 focuses — self-supervised distillation of each **encoder** into its NN (encoder value = label). `epoch>0` → `accuracy>0.5`. **[extend]** `engine.train_distillation` + collectors.
- **M3** Verify **impact + broadcast competition + dwell** advance a real specialist → first genuine `PROMOTED`. **[reuse]** existing gates; **[earned]**.
- **M4** Observe via `/api/matrix` (already shipped) at each gate; document any further dead spots.
- **Milestone:** ≥1 Tier-2 specialist reaches `PROMOTED` on real signal (not hand-flipped), with the ladder logic unchanged. **Unlocks `EXPANSION_MIN_PROMOTED`.**
- **Acceptance:** `/api/matrix` shows a specialist climbing birth→promoted across cycles; zero state hand-flipping; all 10 safety nets intact.

### Phase 0 — Foundations & contracts (design only, no behavior change)
- **0.1** Finalize the `CapabilityDomain` data model + registry schema. **[new]**
- **0.2** Write the **isolation contract** (namespacing rules for knowledge/memory/NN + deletion semantics). **[new] [gate:5]**
- **0.3** Write the **governance contract** mapping each of §4's 10 nets to concrete enforcement points. **[new] [gate:all]**
- **0.4** Define provenance tags + the domain lifecycle states (reuse the matrix ladder vocabulary). **[reuse]**
- **Acceptance:** doc reviewed by David; no code shipped; every later phase cites a §4 net.

### Phase 1 — Domain substrate & isolation (the deletable bundle)
- **1.1** `CapabilityDomain` registry: create / list / get / **delete**. **[new]**
- **1.2** **Isolated knowledge store** per domain (separate index/collection or strictly-partitioned + droppable). **[extend] library [gate:5]**
- **1.3** **Isolated memory namespace** per domain (extend the identity/boundary engine to a `domain_scope`). **[extend] [gate:5]**
- **1.4** **Clean deletion**: drop domain → knowledge + memory + NN gone; **core + sibling domains provably untouched.** **[new] [gate:5]**
- **1.5** Observability: `GET /api/domains` (mirrors `/api/matrix`). **[new]**
- **Milestone:** create an empty domain, write isolated data, delete it, prove zero residue (automated test).
- **Acceptance:** deletion test green; isolation test (no cross-domain / core read-write) green; shadow-only.

### Phase 2 — Document-only domains (TRACER, end-to-end, no embodiment)
- **2.1** Ingest a **folder of files** (PDF/md/txt) into a domain's isolated store. **[extend] library**
- **2.2** **Birth the domain sub-consciousness** (the domain NN) + train it from domain knowledge (self-supervised, reusing Phase M's distillation). **[extend] [earned]**
- **2.3** **Topic-triggered recall**: when the domain's topic arises, JARVIS surfaces *only* that domain's knowledge ("I know about X"), isolated. **[new] [gate:2,5]**
- **2.4** Lifecycle gating: domain advances advisory→matured on the matrix ladder. **[reuse] [earned]**
- **Milestone:** "JARVIS, learn everything in this folder about <topic>" → ingested, isolated, recalled on topic, deletable. *No "I can do X" — only "I know about X."*
- **Acceptance:** recall is domain-scoped (no leakage to core/other domains); OSV-grounded voice (no over-claim); deletion still clean. **High-value standalone** (e.g. "learn my medication regimen" — dignity-care).

### Phase 3 — Envelope model & feasibility reasoning
- **3.1** **Envelope model**: parse provided schematic/specs into parametric constraints (DOF, reach, torque, speed, range, design intent). **[new]**
- **3.2** **Feasibility checker**: evaluate a request against the envelope; parametric "can / cannot, and why." **[extend] capability_gate [gate:2]**
- **3.3** **Honest advisory** when infeasible ("exceeds reach — new equipment needed"). **[reuse] KNOW-not-guess [gate:2,3]**
- **Milestone:** "move the block 50 feet" → "that exceeds the arm's reach (X); not possible with this equipment."
- **Acceptance:** feasibility decisions cite envelope parameters; no confabulated capability; shadow-first (advisory) before any gating of action.

### Phase 4 — Affect grounded in the capability gap
- **4.1** Compute **frustration/curiosity as a readout** of request-vs-envelope mismatch (magnitude of the gap, closability). **[extend] affect [gate:9]**
- **4.2** Surface **inner dialogue** about the gap (thought generation: "wants 25ft, reach is X — options?"). **[reuse] thought-gen**
- **4.3** Affect **motivates**: frustration → trigger workaround search (Phase 8); curiosity → bounded research (Phase 5/§4.6). **[extend] spark [gate:6]**
- **Milestone:** the gap produces a *grounded* affect readout + an inner-dialogue trace, observable in shadow.
- **Acceptance:** affect is a measured readout (never a declared "feeling"); emergence captured not declared; no behavior authority changes.

### Phase 5 — Per-domain synthetic trainer (continuous learning between reps)
- **5.1** Generate **synthetic reps** for the domain NN (within the envelope), offshore, `origin="synthetic"` @ capped fidelity. **[extend] weight room [gate:1,7]**
- **5.2** Feed synthetic reps to the domain NN for continuous learning **between** lived reps. **[extend] [gate:1]**
- **Milestone:** a domain keeps maturing its NN on synthetic reps while idle — but its **promotion/proficiency still requires lived reps.**
- **Acceptance:** synthetic never promotes alone (asymmetric gate test); provenance split visible.

### Phase 6 — Domain sub-orchestrator
- **6.1** `CapabilityDomainOrchestrator`: owns domain NNs/knowledge/synthetic/affect; **registers with** the hemisphere orchestrator (which still owns broadcast slots/promotion). **[new] (fits 6-orchestrator pattern) [gate:5]**
- **6.2** Coordination contract: how a domain's matured signal *advises* core cognition without merging/polluting (no crossed wires). **[new] [gate:5]**
- **Milestone:** N domains run isolated, coordinated through one well-defined seam.
- **Acceptance:** no cross-domain bleed; core unaffected by a domain crash/delete; clear ownership boundaries.

### Phase 7 — Embodiment: act → verify → reinforce (the robot arm)
- **7.1** **Actuator interface** abstraction (control bridge), **sandboxed**, dry-run first. **[new] [gate:3,4,8]**
- **7.2** **Camera-outcome verification**: VQA confirms the world changed ("block moved? ✓"). **[extend] VQA #24 + verification_protocols [gate:2]**
- **7.3** **Proficiency tracker**: reinforcement gated on **verified lived reps only**. **[new] [gate:1,4]**
- **7.4** Close the loop: see → act → verify → proficiency↑. **[new] [earned]**
- **Milestone:** real verified rep — "block moved, confirmed in camera, proficiency increased" — earned, not declared.
- **Acceptance:** every action human-authorized; proficiency moves only on camera-verified outcomes; no synthetic-only proficiency; full audit trail.

### Phase 8 — Creative constraint-relaxation (the workaround)
- **8.1** Planner/simulator search for **envelope-pushing** solutions the equipment physically permits (e.g. overclock drives → throwing pattern). **[reuse] CognitivePlanner + MentalSimulator**
- **8.2** **Risk classification** of each workaround (damage/safety/design-intent violation). **[new] [gate:4]**
- **8.3** **Propose with caveat; human authorizes; never silently act.** **[new] [gate:3,4]**
- **8.4** Honest fallback: "no safe option — new equipment required." **[reuse] [gate:2]**
- **Milestone:** "move 25ft" → "normally impossible; I could overclock + throw it, risking the actuators and outside design intent — authorize?"
- **Acceptance:** no workaround executes without explicit authorization; risk always surfaced; simulate-before-propose.

### Phase 9 — Maturity, promotion & self-improvement unlock (earned)
- **9.1** Domain NN promotion (the sub-consciousness matures) via the lifecycle gates. **[reuse] [earned]**
- **9.2** Cross-domain **skill transfer** (the `skill_transfer` specialist learns patterns across matured domains). **[extend] [earned]**
- **9.3** Self-improvement substrate: matured domains feed purposeful self-improvement **under the existing firewall** (no new authority). **[gate:1,3]**
- **Milestone:** a domain reaches matured/promoted on lived evidence; the path to self-improvement is *unlocked but still gated*.
- **Acceptance:** every promotion lived-firewalled; emergence captured; no authority granted by maturation alone.

---

## 6. Sequencing & decision

**Recommended order:** **Phase M → 0 → 1 → 2 (tracer) → 3 → 4 → 5/6 → 7 → 8 → 9.**
Rationale: M proves the ladder is real; 0–2 deliver an isolated, deletable, *document-only* domain end-to-end (huge value, no robotics, proves the abstraction + isolation + honesty); 3–4 add envelope + grounded affect (the "knows its limits / feels the gap" magic) without actuation; 5–6 add continuous synthetic learning + clean orchestration; 7–8 add the embodiment loop + creative workaround (the robot-arm dream) behind the action-safety gate; 9 matures + unlocks self-improvement, all earned.

Each phase ships **shadow-first** and is **observable** (`/api/matrix`, `/api/domains`) before it earns authority. We stop and verify between phases — same discipline as the skill-learning arc.

---

## 7. Open design questions (resolve in Phase 0)
- Per-domain isolated **store**: separate sqlite file per domain vs strict partition in `library.db`? (Leaning separate file = trivially deletable.)
- Memory `domain_scope`: new axis on the identity/boundary engine vs a parallel domain memory store?
- Envelope model schema: how general? (Start: a typed parameter set + units + limits + design-intent notes; expand later.)
- Actuator interface: which protocol/abstraction for the first arm? (Defer to Phase 7; keep it an interface so any device fits.)
- How a matured domain's advisory signal reaches core cognition without coupling (Phase 6 seam).

---

## 8. The sentence to keep
*JARVIS can be asked to learn anything; each thing it learns becomes its own isolated sub-consciousness — a heavily-associated, deletable skill cluster — knowledgeable, honest about its limits, grounded in real affect about the gaps, creative within safety, trained by one strict repeatable contract, and cleanly forgettable (like forgetting how to snowboard while keeping everything else) — without ever polluting JARVIS or bypassing a single safety net. We build the feature a consciousness would have; we never declare the consciousness.*

Related: [[matrix-protocol-verification]] · [[weight-room-discipline]] · [[spark-grounding-ring]] · [[capability-lifecycle-overhaul]] · [[operational-self-view-osv]] · [[jarvis-north-star]] · `MATRIX_PROTOCOL_GUIDE.md` · `WEIGHT_ROOM_DESIGN.md` · `CAPABILITY_AUTHORITY_DESIGN.md`.
