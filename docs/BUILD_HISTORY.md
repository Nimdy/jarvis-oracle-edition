# Build History

Shipped changelog extracted from TODO.md for documentation hygiene.
Active priorities and runtime state remain in [TODO.md](../TODO.md).

---

## Research Quality and Codebase Study Fix (2026-05-09, SHIPPED)

Tightened intake quality for the autonomy research pipeline and codebase
self-study.  After a fresh brain reset, off-topic academic papers (tort law,
business continuity) were slipping through Semantic Scholar's broad keyword
overlap, and codebase self-study was generating hundreds of low-value
`factual_knowledge` memories for raw constants and metric definitions.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Domain relevance gate | Added `_AI_DOMAIN_INDICATORS` (~60 AI/ML/cognition terms) and `_has_domain_relevance()` check in `_store_findings()` — findings without any AI-domain indicator, companion-interest term, or intent tag-cluster term are rejected before memory write | `brain/autonomy/knowledge_integrator.py` |
| Relevance threshold raise | `_compute_relevance()` threshold raised from 0.10 to 0.15 — reduces false-positive keyword overlap acceptance | `brain/autonomy/knowledge_integrator.py` |
| Companion domain expansion | `_get_companion_domain_terms()` reads `user_preference` memories to derive additional accepted domain terms post-gestation — during gestation only AI/core topics pass; after companion grounding, user interests expand the gate | `brain/autonomy/knowledge_integrator.py` |
| Tag-cluster passthrough | Intent `tag_cluster` terms always pass the domain gate regardless of gestation mode — research explicitly requested on a topic is never domain-blocked | `brain/autonomy/knowledge_integrator.py` |
| Codebase study claim skip | `study_source()` now skips claim generation (`_try_llm_extraction`, `_extract_claims`, `_create_claim_memories`) for `source_type == "codebase"` — concept extraction and concept graph updates still run | `brain/library/study.py` |
| Rejection telemetry | Added `sources_rejected_domain`, `sources_rejected_low_relevance` counters to `_ingest_stats` and `sources_skipped_codebase_claims` to `_study_telemetry` | Both files |
| Regression tests | 26 tests covering: AI content acceptance (3), off-topic rejection (3), tag-cluster passthrough (2), gestation/companion domain expansion (6), relevance threshold (2), rejection counter increments (2), codebase study skip (3), indicator sanity (3), end-to-end store_findings (2) | `brain/tests/test_research_quality.py` |

### Design decisions

- The domain gate uses substring matching against a curated indicator set rather than LLM classification — zero latency, fully deterministic, no VRAM cost.
- Generic English words that appear in both AI and non-AI contexts (`precision`, `optimization`, `planning`) were excluded from the indicator set to prevent false positives.
- Companion expansion reads `user_preference` memories directly — no new persistence file needed, grows organically as the operator trains Jarvis.
- Codebase representation is intentionally split: Library chunks handle semantic retrieval, CodebaseIndex handles structural/AST queries, diagnostic/code-quality encoders handle hemisphere features. Study-claim memories for codebase sources were redundant noise.

### What this fixes

- Off-topic academic papers (tort law, business continuity, medical billing) no longer enter the memory store during gestation or normal operation.
- Codebase self-study no longer produces hundreds of low-granularity `factual_knowledge` memories like `MAX_QUEUE_SIZE = 20`.
- Companion interests can still drive research after the user's preferences are known.

---

## Dream Consolidation Feedback Loop Fix (2026-05-09, SHIPPED)

Fixed a recursive dream consolidation feedback loop that generated ~57 junk
memories/hour, consuming 56.6% of the memory store with self-referential
dream artifacts (dreaming about dreams about dreams).  Three root-cause bugs
combined into a positive feedback loop: promoted dream artifacts re-entered
the clustering input, the consolidation engine's recursion guard missed the
promoted artifact tags, and no content deduplication prevented identical
artifacts from being created every cycle.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Dream input exclusion filter | Added `dream_artifact` + `dream_consolidation_proposal` to `_CONSOL_EXCLUDE_TAGS` — promoted dream artifacts no longer re-enter the dream cycle clustering input | `brain/consciousness/consciousness_system.py` |
| Consolidation engine recursion guard | Expanded `_ALREADY_CONSOLIDATED_TAGS` to include `dream_artifact` + `dream_consolidation_proposal` — clusters dominated by dream artifacts now score -1.0 | `brain/memory/consolidation.py` |
| Content fingerprint dedup | Added per-cycle content dedup set in Phase 4 artifact generation — identical content strings produce only one artifact per cycle | `brain/consciousness/consciousness_system.py` |
| Self-referential validator guard (Layer 1) | Content string check: consolidation proposals whose content mentions `dream_artifact` or `dream_consolidation_proposal` are discarded | `brain/consciousness/dream_artifacts.py` |
| Self-referential validator guard (Layer 2) | Source-memory tag dominance check: artifacts where >= 50% of resolvable source memories carry dream-self-ref tags are discarded regardless of content wording | `brain/consciousness/dream_artifacts.py` |
| Per-cycle artifact budget | `MAX_ARTIFACTS_PER_DREAM_CYCLE = 20` caps artifact creation per dream cycle, preventing buffer flooding | `brain/consciousness/consciousness_system.py` |
| Helper refactor | Extracted `_add_artifact_if_novel()` closure in Phase 4 that enforces both dedup and budget for all artifact types (patterns, connections, anomalies, gaps, consolidation proposals) | `brain/consciousness/consciousness_system.py` |
| Regression tests | 14 tests covering exclusion filter, content dedup, self-ref discard (content + source tags), consolidation tag guard, repeated-cycle stability, promoted artifact exclusion, and budget enforcement | `brain/tests/test_dream_consolidation_loop.py` |

### Invariant established

> Dream cycles may read lived memories. Dream cycles may create provisional
> artifacts. Validators may promote selected artifacts. Dream cycles must not
> recursively consolidate promoted dream artifacts.

This is the ontological boundary from SyntheticSoul §6.4: dreams operate on
lived experience, not on dream artifacts.  Consolidation reduces entropy, it
does not amplify it.

### Architecture decisions

- The exclusion filter (`_CONSOL_EXCLUDE_TAGS`) is the primary fix — without it the other guards would be unnecessary.
- The validator's two-layer self-ref check is defense-in-depth: Layer 1 (content string) is fast and catches the known pattern; Layer 2 (source-memory tag dominance) is structural and catches future wording variants.
- The `_add_artifact_if_novel()` helper consolidates the dedup + budget logic for all 5 artifact types, eliminating code duplication.
- `MAX_ARTIFACTS_PER_DREAM_CYCLE = 20` was chosen to be generous enough for healthy dreaming but prevent the 46–95 artifacts/cycle observed during the pathology.

---

## Training Dashboard Tab (2026-05-08, SHIPPED)

Added a 7th main dashboard tab — **Training** — that consolidates all
operator-facing maturation guidance into a single guided UX.  Previously,
companion training prompts, language evidence gates, synthetic exercise
controls, and the skill-acquisition weight room were scattered across the
Learning tab (mixed in with ML internals) and the Trust tab.  Users doing
initial Jarvis training had to navigate 15+ dense panels to find the
"Say This Next" prompts.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Training tab (new) | 7th tab on main dashboard with hero progress banner, interactive training section (companion training + language evidence gates), and synthetic training section (perception exercise, skill-acquisition weight room, 7 text-only exercises) | `brain/dashboard/static/index.html`, `brain/dashboard/static/renderers.js` |
| Training hero banner | Overall readiness progress bar, stage/graduated status, stat cards (stage, conversations, preferences, gate work remaining, distillation count) | `brain/dashboard/static/renderers.js` `_renderTrainingHero()` |
| Text-only exercises panel | Compact card grid for 7 CLI-only exercises (commitment, claim, retrieval, world model, contradiction, diagnostic, plan evaluator) with last-run status, Smoke/Coverage run buttons | `brain/dashboard/static/renderers.js` `_renderSyntheticTextExercisesPanel()` |
| Synthetic exercises API | `GET /api/synthetic/exercises/status` (from snapshot cache) and `POST /api/synthetic/exercises/run` (API-key protected, background thread dispatch to any of 7 exercise runners) | `brain/dashboard/app.py` |
| Snapshot aggregation | `_build_synthetic_exercises_snapshot()` scans report dirs under `~/.jarvis/synthetic_exercise/` for commitment, skill-acquisition, and batch (retrieval/world_model/contradiction/diagnostic/plan_evaluator) reports | `brain/dashboard/snapshot.py` |
| Learning tab cleanup | Removed Companion Training panel, Skill Acquisition Weight Room panel, and "Start Onboarding" toolbar button from Learning tab — these now live exclusively on Training tab | `brain/dashboard/static/renderers.js` |
| Language panel cleanup | Removed Manual Gate Work embed from Language Substrate panel (Learning tab) and Language Governance panel (Trust tab) — now standalone on Training tab as "Language Evidence Gates" | `brain/dashboard/static/renderers.js`, `brain/dashboard/static/dashboard.js` |
| Cockpit quickcard | Added 4th quickcard "Training" showing readiness % alongside Trust, Memories, Learning | `brain/dashboard/static/dashboard.js` |
| Tab wiring | `TAB_NAMES` array (7 entries), `renderActiveTab` dispatch, keyboard shortcut `6` = Training / `7` = Diagnostics, `TAB_WAYFINDING` entry, panel ownership metadata updated | `brain/dashboard/static/dashboard.js`, `brain/dashboard/static/index.html` |
| Cross-file exposure | `_renderSyntheticExercise` exposed as `window._renderSyntheticExercise` so Training tab in renderers.js can call the Cockpit-defined function | `brain/dashboard/static/dashboard.js` |

### Architecture decisions

- Companion Training, Manual Gate Work, and synthetic exercises are **moved** (not copied) from Learning/Trust to Training — single source of truth, no duplicate panels.
- Synthetic Perception Exercise is **copied** (kept on Cockpit too) since it has operational significance when active.
- Panel render functions stay in their original files; only the caller changed.
- The 7 text-only exercises dispatch to their existing `run_*()` functions in background threads; no new exercise logic was added.

### Deferred

Preference capture bugs (tag mismatch, comma-terminated regex, missing negative boundary intent class, dashboard prompt misalignment) identified during companion training testing are **not** addressed here — they are a separate code-quality fix once the system matures enough for NN routing to cover the gaps.

---

## Intention Infrastructure Stage 1 — IntentionResolver Shadow (2026-05-08, SHIPPED)

Built the IntentionResolver (`brain/cognition/intention_resolver.py`) — a shadow-only
heuristic relevance predictor that scores resolved intentions for proactive delivery.
Answers: *"Given a resolved intention and the current world state, should JARVIS say
anything about it right now?"*

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| IntentionResolver core | Heuristic evaluate() with 4 decisions (deliver_now, deliver_on_next_turn, suppress, defer), 10 controlled reason codes, 5-rung promotion ladder (shadow_only → active), JSONL verdict logging with 10MB rotation | `brain/cognition/intention_resolver.py` |
| IntentionDeliveryEncoder | 24-dim feature vector (3 blocks: intention intrinsic, conversation context, system governance) for future Stage 2 specialist | `brain/hemisphere/intention_delivery_encoder.py` |
| Registry additions | `get_recent_resolved_for_resolver()` + `attach_resolver_verdict()` write-once metadata on IntentionRegistry | `brain/cognition/intention_registry.py` |
| Consciousness hook | `intention_resolver_tick` at 30s cadence, mode-gated via `allowed_cycles`, evaluates up to 5 recent resolutions per tick | `brain/consciousness/consciousness_system.py`, `brain/consciousness/modes.py` |
| ProactiveGovernor hook | `_check_intention_resolver_candidate()` consumes deliver_now verdicts (only when resolver stage permits delivery) | `brain/personality/proactive.py` |
| Self-status hook | Surfaces resolver shadow verdict count in MeaningFrame self_status facts | `brain/reasoning/bounded_response.py` |
| FractalRecall hook | Read-only context source for topic reinforcement (comment updated) | `brain/memory/fractal_recall.py` |
| Dashboard / API | `GET /api/intention-resolver`, `POST /api/intention-resolver/rollback`, `POST /api/intention-resolver/stage`, Intention Resolver panel in Trust tab, `intention_resolver` snapshot, `intention_resolver = PRE-MATURE` status marker | `brain/dashboard/app.py`, `brain/dashboard/snapshot.py`, `brain/dashboard/static/dashboard.js` |
| Tests | 36 tests: shadow-only, heuristics, vocabulary, logging, no-mutation, encoder shape, promotion ladder, registry additions, shadow metrics | `brain/tests/test_intention_resolver.py` |

### Safety invariants

- Starts `shadow_only` — no delivery until operator promotes past `advisory_canary`
- Stage 2 `intention_delivery` hemisphere config stays commented out in `types.py`
- No new event schema — consumes existing resolution events
- ProactiveGovernor remains single authority for unsolicited speech
- CapabilityGate + AddresseeGate still apply to any delivery path
- Stage 0 code (register/resolve/abandon/evaluate_commitment) NOT mutated
- 44 Stage 0 tests still pass, 219 P4/P5 tests still pass

---

## Data Processing Live + Language Truth-Lane Alignment (2026-05-05, SHIPPED)

Closed the operator-facing gaps found after `data_processing_v1` moved from
learning into an active operational proof plugin. The learned skill is now
verified/live, the plugin completed its acquisition activation path, and the
language evidence gates have accurate baseline prompts for identity,
capability, and recent skill-learning self-report.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Data processing skill | `data_processing_v1` completed learning, acquired operational plugin proof, and is now represented as a verified skill | `brain/skills/learning_jobs.py`, `brain/acquisition/orchestrator.py`, `brain/tools/plugins/` |
| Plugin activation truth | Acquisition-owned plugins now require runtime shadow smoke proof before activation and expose activation diagnostics instead of relying only on time in shadow | `brain/acquisition/orchestrator.py`, `brain/acquisition/job.py` |
| Plugin operator drill-down | Capability Pipeline plugin modal shows generated package files, acquisition context, verification proof, runtime stats, storage boundary, and promotion authority | `brain/dashboard/app.py`, `brain/dashboard/static/self_improve.html` |
| Skill report wording | Ordinary skill completions now write `skill_learning_report.json` and say “Skill learning report”; Matrix wording remains reserved for real Matrix Protocol jobs | `brain/skills/learning_jobs.py`, `brain/tests/test_skill_acquisition_handoff.py` |
| Language prompt deck | Manual Language Substrate gate cards now show exact baseline prompts that match current route/class counters | `brain/dashboard/static/renderers.js` |
| Capability-status class | Capability introspection queries about capabilities, skills, and abilities now seed `INTROSPECTION -> capability_status` instead of generic self-introspection | `brain/conversation_handler.py`, `brain/tests/test_none_capability_fallback.py` |
| Identity-answer baseline | Identity prompt deck now uses user-identity recognition prompts (`Who am I?`, `Do you recognize me?`) that feed `IDENTITY -> identity_answer` | `brain/dashboard/static/renderers.js` |
| Recent skill-learning self-report | “What skill did you just/last finish learning?” now routes to strict introspection and answers from completed `LearningJobStore` evidence | `brain/reasoning/tool_router.py`, `brain/tools/introspection_tool.py`, `brain/tests/test_tool_router.py` |

### Runtime validation

- Focused tests: `46 passed` across router baselines, capability fallback, skill report wording, and strict recent-learning evidence.
- Live brain route check: `Who am I?` recorded as `IDENTITY -> identity_answer`.
- Live brain route check: `What can you do?` recorded as `INTROSPECTION -> capability_status`.
- Live brain evidence check: `What skill did you just finish learning?` routes to `INTROSPECTION` and resolves to `Data Processing (data_processing_v1)` from completed learning-job evidence.
- Desktop brain restarted successfully; dashboard returned HTTP `200`; supervisor and main process were running.

### Guardrails preserved

- No language runtime bridge promotion, canary enablement, or Phase C live routing.
- No CapabilityGate bypass and no change to skill verification authority.
- Router additions are narrow baseline self-report phrases; answers still require persisted evidence.
- Synthetic and shadow specialists remain telemetry-only and do not satisfy lived maturity gates.
- Historical pre-fix `matrix_report.json` artifacts remain visible as old records; new ordinary skill completions use `skill_learning_report.json`.

---

## Skill-Acquisition Hardening + Synthetic Weight Room Closeout (2026-05-05, SHIPPED)

Closed the lifecycle gaps found while forcing on skill learning and operational
plugin acquisition. The system now refuses incomplete acquisition plans, mirrors
terminal acquisition failures back to the linked `LearningJob`, reports shared
CodeGen as infrastructure rather than self-improvement ownership, and exposes a
synthetic skill-acquisition weight room for shadow-only specialist training.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Plan quality | Acquisition plans missing `technical_approach`, `implementation_sketch`, or `test_cases` are rejected before approval and surfaced with planning diagnostics | `brain/acquisition/orchestrator.py`, `brain/dashboard/static/renderers.js` |
| Terminal closure | Failed/cancelled acquisition jobs promptly block the linked learning job instead of waiting on executor cooldowns | `brain/skills/learning_jobs.py`, `brain/skills/operational_bridge.py` |
| CodeGen truth split | Shared `CodeGenService` and `CoderServer` report `authority=infrastructure_only`, `active_consumer`, `last_consumer`, and consumers for acquisition/self-improvement | `brain/codegen/service.py`, `brain/codegen/coder_server.py`, `brain/dashboard/snapshot.py` |
| Jarvis prompt contract | Skill-linked acquisition codegen now receives a structured Jarvis-authored packet with skill contract fixture expectations and prompt diagnostics | `brain/acquisition/orchestrator.py`, `brain/acquisition/job.py` |
| Skill acquisition specialist | Added `SKILL_ACQUISITION` Tier-1 distillation specialist for shadow-only lifecycle outcome learning | `brain/acquisition/skill_acquisition_encoder.py`, `brain/hemisphere/types.py`, `brain/hemisphere/data_feed.py` |
| Synthetic weight room | Added smoke/coverage/strict/stress synthetic workouts and dashboard/API controls with strict telemetry-only authority | `brain/synthetic/skill_acquisition_exercise.py`, `brain/synthetic/skill_acquisition_dashboard.py`, `brain/dashboard/app.py` |
| Learning tab UX | Reorganized Learning tab so Skills, Library, Hemisphere NNs, and Language Substrate get full-width rows; acquisition and synthetic weight room are paired | `brain/dashboard/static/renderers.js` |
| OBS operator utility | Added local OBS audio receiver and brain-side forwarding hooks; this is an operator utility, not cognition authority | `scripts/obs_audio_receiver.py`, `brain/perception/server.py`, `brain/.env.example` |
| Release/sync scope | Documented that transient `website/` work is excluded from sync; dashboard static pages remain primary shipped docs | `sync-desktop.sh`, `sync-pi.sh` |

### Runtime validation

- Focused tests: `100 passed` across skill acquisition hardening, weight room, handoff closure, acquisition orchestration, and CodeGen service.
- Live brain API state: Skill registry `13 verified / 1 blocked`; learning jobs `0 active`; acquisition `0 active / 0 pending plan reviews`; CodeGen available and idle with `authority=infrastructure_only`.
- Synthetic smoke: `12/12` episodes, `0` invariant failures, `record_signals=false`, `+0` features, `+0` labels.
- Synthetic coverage: `200/200` episodes, `0` invariant failures, `record_signals=true`, `+7` synthetic features, `+5` synthetic labels.
- Heavy profiles remain gated: `strict` and `stress` blocked by `heavy_profile_operator_flag_disabled`.
- Validation report: `docs/validation_reports/skill_acquisition_hardening-2026-05-05.md`.

### Guardrails preserved

- Synthetic workouts cannot verify skills, promote plugins, unlock capability claims, or satisfy lived maturity gates.
- `SKILL_ACQUISITION` is shadow-only and has no policy/broadcast/plugin/skill authority.
- CodeGen remains infrastructure only; ownership stays with the caller lane and its native gates.
- OBS forwarding is local operator tooling, not a cloud service or reasoning path.

---

## Skill Approval Bridge + Matrix Protocol Guide (2026-05-04, SHIPPED)

Added the missing human approval boundary between skill verification and
operational plugin acquisition. A contract skill that lacks callable proof now
waits for operator approval instead of silently trying to launch acquisition or
falling into a blocked retry loop.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Handoff safety | Missing callable proof records `awaiting_operator_approval` and persists approval errors instead of swallowing exceptions | `brain/skills/operational_bridge.py` |
| Learning lifecycle | Waiting handoffs survive reboot without phase-failure spam or timeout blocking | `brain/skills/learning_jobs.py`, `brain/skills/job_runner.py` |
| Operator approval | API-key protected approve/reject endpoints create or reject the acquisition handoff with audit notes | `brain/dashboard/app.py` |
| Dashboard audit | Skill detail modal shows operational handoff state, errors, actions, and linked acquisition proof chain | `brain/skills/audit_trail.py`, `brain/dashboard/static/interactives.js` |
| Regression tests | Real `AcquisitionOrchestrator`/`AcquisitionStore` coverage catches handoff creation failures | `brain/tests/test_skill_acquisition_handoff.py` |
| Matrix docs | Added a dedicated guide explaining Matrix Protocol as neural-specialist learning, not instant operational skill upload | `docs/MATRIX_PROTOCOL_GUIDE.md` |

### Runtime validation

- Local focused tests: `12 passed` across skill contracts and skill approval bridge regressions.

### Guardrails preserved

- No verifier starts codegen or plugin acquisition without operator approval.
- No acquisition approval bypasses existing plan review, deployment, quarantine, activation, or sandbox proof gates.
- Matrix Protocol remains advisory neural-specialist evidence unless an operational verifier consumes and proves it.

---

## Skill Verification Contracts + Skill Proof Bridge + Operator Guide (2026-05-04, SHIPPED)

Closed the false-positive skill verification gap exposed by `data_processing_v1`.
Jarvis now distinguishes lifecycle evidence from operational capability proof:
a learning job may create research/integration artifacts, but SkillRegistry
verification requires contract evidence, expected-vs-actual smoke results, and
sandbox/callable proof when the skill contract requires it.

### What shipped

| Area | Change | Evidence |
| --- | --- | --- |
| Skill contracts | Added reusable skill execution contracts for proof-bearing skills such as data transformation | `brain/skills/execution_contracts.py`, `brain/tests/test_skill_execution_contracts.py` |
| Procedural verification | `ProceduralVerifyExecutor` now fails missing callable proof honestly and no longer treats lifecycle artifacts as operational proof | `brain/skills/executors/procedural.py` |
| Proof bridge | Added the governed bridge from missing callable proof to acquisition/plugin construction | `brain/skills/operational_bridge.py`, `brain/acquisition/orchestrator.py` |
| Runtime provider | Learning jobs can consume only sandbox-proven `supervised`/`active` plugin callables via context provider | `brain/main.py`, `build_skill_execution_callables()` |
| Audit trail | `/api/skills/{skill_id}` includes a read-only audit packet with evidence history, missing proof, artifacts, and acquisition proof chain | `brain/skills/audit_trail.py`, `brain/dashboard/static/interactives.js` |
| Operator docs | Added a dedicated skill learning guide with expectations, data flow, and 20 user/hobby examples | `docs/SKILL_LEARNING_GUIDE.md` |

### Runtime validation

- Local focused tests: `11 passed` across skill contracts, audit packet, and proof bridge regressions.
- Remote brain focused tests after `./sync-desktop.sh`: `11 passed`.
- Live reboot validation: Jarvis accepted a fresh CSV/data-processing learning request, created `job_20260504T205734Z_581e`, advanced `assess -> research -> integrate`, and preserved `status=learning` until operational proof exists.

### Guardrails preserved

- No hardcoded `data_processing_v1` implementation.
- No direct codegen inside `ProceduralVerifyExecutor`.
- No dashboard-triggered execution.
- No bypass of plugin quarantine, shadow/supervised/active gates, sandbox proof, approval gates, or CapabilityGate.
- Historical false positives remain visible as historical audit evidence and do not imply current operational readiness.

---

## TODO_V2 Phase 3 Sprint — Tier-2 Matrix Template Set + TLA+ + Pi `/mind` Consolidation (2026-04-25, SHIPPED)

Eight-lane Phase 3 sprint executed live against the desktop brain
(`192.168.1.222`) via `sync-desktop.sh`. Closes the M6 broadcast-slot
critical path, completes the five-lane Tier-2 Matrix specialist template
set, formally model-checks the Phase 6.5 governance invariants in TLA+,
consolidates the Pi mental-world surface into the existing consciousness
visualizer, and records the first emergent-social-behavior observation.
**No reset. No L3 promotion. No HRR/P5 authority change. No canonical
writes from any new specialist.** Continuity preserving throughout.

### Lane summary

| Lane | Title | Status | Evidence |
| --- | --- | --- | --- |
| **P3.1** | TLA+ formal verification of Phase 6.5 invariants | SHIPPED | `tla_phase_65-2026-04-25.md` |
| **P3.5** | M6 broadcast slot expansion — gate confirmation closeout | SHIPPED (closed) | `p3_5_m6_expansion_wiring-2026-04-25.md` |
| **P3.6** | Tier-2 specialist `positive_memory` | SHIPPED (CANDIDATE_BIRTH only) | `p3_6_positive_memory-2026-04-25.md` |
| **P3.7** | Tier-2 specialist `negative_memory` | SHIPPED (CANDIDATE_BIRTH only) | `p3_7_negative_memory-2026-04-25.md` |
| **P3.8** | Tier-2 specialist `speaker_profile` | SHIPPED (CANDIDATE_BIRTH only) | `p3_8_speaker_profile-2026-04-25.md` |
| **P3.9** | Tier-2 specialist `temporal_pattern` | SHIPPED (CANDIDATE_BIRTH only) | `p3_9_temporal_pattern-2026-04-25.md` |
| **P3.10** | Tier-2 specialist `skill_transfer` | SHIPPED (CANDIDATE_BIRTH only) | `p3_10_skill_transfer-2026-04-25.md` |
| **P3.11** | `dream_synthesis` Tier-2 promotion (Phase 1 feasibility) | DEFERRED — INSUFFICIENT | `p3_11_dream_synthesis_feasibility-2026-04-25.md` |
| **P3.14** | Pi `/mind` consolidation → `consciousness_feed.scene` + particle visualizer | SHIPPED | `p3_14_pi_mind_merge-2026-04-25.md` |
| **OBS** | Emergent addressee overreach (group-conversation turn-taking) | OBSERVATION (one instance) | `emergent_addressee_overreach_observation-2026-04-25.md` |
| **MATRIX** | Tier-2 Matrix template-set readiness closeout | SHIPPED | `p3_tier2_matrix_template_closeout-2026-04-25.md` |

### P3.1 — Phase 6.5 invariants in TLA+ (formal verification)

Formal model + bounded model check of the three-axis autonomy governance
state machine that lives in `tests/test_l3_*.py`,
`tests/test_autonomy_*.py`, and the `current_ok` / `prior_attested_ok` /
`activation_ok` / audit-ledger Python contracts. **Zero production code
changed.** Documentation-grade artifact only.

| Component | Path |
| --- | --- |
| TLA+ specification | `docs/formal/phase_65.tla` |
| TLC config | `docs/formal/phase_65.cfg` |
| Reproduction guide | `docs/formal/README.md` |
| Validation report | `docs/validation_reports/tla_phase_65-2026-04-25.md` |

State variables: `level`, `current_ok`, `prior_attested_ok`,
`attestation_strength`, `request_ok`, `approval_required`,
`activation_ok`, `audit_log` (sequence), `approved`, `request_state`,
`clock`. Actions: `RequestEscalation`, `Approve`, `Reject`, `Apply`,
`MarkClean`, `MarkRolledBack`, `Expire`, `RotateLedger`, `RestartProcess`.

Soundness fix: `clock` relaxed from a finite interval (`0..MAX_AUDIT_LEN*4`)
to `Nat`, with `StateBound == clock <= MAX_AUDIT_LEN*4` wired through TLC
as `CONSTRAINT` (the idiomatic TLA+ pattern for bounded exploration). The
redundant `clock < MAX_AUDIT_LEN*4` guard inside `RequestEscalation` was
dropped.

**TLC run on the brain authority host** (`duafoo@192.168.1.222`,
OpenJDK 21.0.10, `tla2tools.jar` 2.19, 32 workers / 32 cores):

```
States generated:    9,724,247
Distinct states:     2,528,932
Diameter:            34
Counterexamples:     0
Wall time:           ~3 s
Result:              Model checking completed. No error has been found.
```

All five model-checked invariants pass:

- `TypeOK` — every state variable stays in its declared domain.
- `NoAutoPromotion` — `level` never advances without `Approve`.
- `AttestationImmutability` — past audit entries never mutate.
- `RequestOkDerivationInv` — `request_ok` is derived, never asserted.
- `SafetyInvariants` — composite of the three above + frame conditions.

Three structurally enforced invariants are confirmed by the absence of
any reachable state that would only be reachable via a forbidden
assignment, and pinned to their Python anchor points in the report:
`CurrentOkIsLiveSourced`, `EvidenceClassSeparation`, `AuditLogAppendOnly`.

Brain-side post-spec audits all clean: `schema_emission_audit.py`
(0 violations), `dashboard_truth_probe.py` (0 findings),
`run_validation_pack.py` (every HRR/P5/runtime guardrail PASS).

### P3.5 — M6 broadcast slot expansion (gate-confirmation closeout)

Production fixes had already shipped in the 2026-04-24 unblock sprint
(see prior section); this 2026-04-25 closeout is gate-confirmation +
evidence + TODO_V2 status correction. Four invariants verified live:

1. `ShadowPolicyRunner.set_hemisphere_signals()` exists and forwards
   to the shadow encoder (via `engine._on_tick`).
2. `engine.py` mirrors live hemisphere signals into the M6 shadow
   encoder on every tick.
3. `specialist_verification_ts` is stamped at the
   `PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY` transition.
4. `_check_expansion_trigger()` passes under synthetic conditions with
   ≥2 promoted Tier-2 specialists, mean impact ≥0.05, and oldest
   verification age >7 days.

Gate result: 10/10 M6 tests, 383/383 hemisphere + Matrix sweep,
4449 / 4449 full regression (excluding 6 pre-existing unrelated flakes
in `test_dogfood_campaign.py` / `test_research_content_depth.py`).
Live M6 expansion correctly **dormant** — no Tier-2 specialists are
PROMOTED yet, so the trigger conditions are not met. Dormant is
correct. Closing report:
`docs/validation_reports/p3_5_m6_expansion_wiring-2026-04-25.md`.

### P3.6 → P3.10 — Five Tier-2 Matrix specialists (template set complete)

The five planned Tier-2 specialists shipped in sequence on 2026-04-25,
each as a CANDIDATE_BIRTH-only registration with a pure-function
encoder, deterministic `compute_signal_value()` ∈ [0, 1] (never
accuracy-as-proxy), orchestrator dispatch through `_matrix_focus_signal()`
ahead of the legacy accuracy fallback, architect topology override, and
removal from `FUTURE_ONLY_HEMISPHERE_FOCUSES` only after the encoder /
writer literal path was complete and `schema_emission_audit.py` stayed
at 0 violations.

| Lane | Encoder | Signal sources | Lane-specific guardrail |
| --- | --- | --- | --- |
| **P3.6** `positive_memory` | `brain/hemisphere/positive_memory_encoder.py` | episodic reinforcement, low-friction streaks, calibrated positive valence | none beyond the standard authority fence |
| **P3.7** `negative_memory` | `brain/hemisphere/negative_memory_encoder.py` | corrected memories, friction events, regressed autonomy outcomes, quarantine pressure | **Does not consume `emotion_depth`** signals — verified by behavioral regression test (encoder output is invariant to `emotion_negative_bias` perturbations) |
| **P3.8** `speaker_profile` | `brain/hemisphere/speaker_profile_encoder.py` | identity-fusion state, known-speaker registry, addressee disambiguation, rapport stability | **Raw speaker embeddings (192-dim ECAPA-TDNN `speaker_repr`, `face_repr`) MUST NOT cross the encoder boundary, MUST NOT appear in any API surface, MUST NOT enter any dashboard payload** — fenced at three layers (encoder input, orchestrator context builder, two regression tests) |
| **P3.9** `temporal_pattern` | `brain/hemisphere/temporal_pattern_encoder.py` | rhythm / cadence / recency / mode-stability, no calendar facts | **Privacy fence: NO schedule / weekday / hour-of-day / calendar inference** — fenced at three layers (encoder input, orchestrator context builder, three regression tests including a static source-scan and orchestrator-context fence) |
| **P3.10** `skill_transfer` | `brain/hemisphere/skill_transfer_encoder.py` | skill registry breadth / maturity / matrix-protocol fraction, learning-job phase counts, capability promotion correlations | **`Similarity is not capability.`** Specialist may say "this skill resembles prior verified skill families"; MUST NOT say "therefore this skill is verified / safe / promotable". Capability gate intact. Fenced at five layers. |

Cumulative test count for the P3.6→P3.10 ladder: **452/452** focused
hemisphere + Matrix + audit-regression tests green at the close of
P3.10 (107 of those are net-new for these five lanes), **4532/4532**
full sweep on the brain (excluding the same 6 pre-existing unrelated
flakes seen across all five lanes).

Schema audit Tier-2 emitted-focus count progressed cleanly:
17 → 18 (P3.6) → 19 (P3.7) → 20 → 21 (P3.8) → 22 (P3.9) → 23 (P3.10).

**M6 expansion remains correctly dormant** at the close of the
template-set sprint: `_check_expansion_trigger()` requires ≥2 PROMOTED
Tier-2 specialists with mean impact ≥0.05 and oldest verification
older than 7 days. None of the five new specialists are promoted —
all are at CANDIDATE_BIRTH. Dormant is the correct state.

Tier-2 template-set readiness closeout:
`docs/validation_reports/p3_tier2_matrix_template_closeout-2026-04-25.md`.

### P3.11 — `dream_synthesis` Tier-2 (Phase 1 feasibility, DEFERRED)

Phase 1 feasibility query against the live brain returned **INSUFFICIENT**:
369 total distillation signals (gate 1: ≥200, OK), but 0 quarantined
samples (gate 2 fail) and a degenerate `reason_distribution` at 100 %
`uncategorized` (gates 4 + 5 fail). Tier-1 dream_synthesis is otherwise
stable (1 active network `distill_de720e6d21`, 0 failures, 0 rollbacks,
accuracy 0.3212 plateaued at the missing-class ceiling).

**No code shipped.** P3.11 is correctly deferred until the dream
validator emits at least 1 quarantined sample and ≥3 non-zero reason
categories. Re-run `brain/scripts/dream_synthesis_feasibility.py`
when training-data quality changes. Report:
`docs/validation_reports/p3_11_dream_synthesis_feasibility-2026-04-25.md`.

### P3.14 — Pi `/mind` consolidation → consciousness particle visualizer

The standalone `/mind` kiosk view on the Pi was a redundant duplicate of
the brain dashboard `/hrr-scene` page (same payload, worse screen real
estate, no end-user audience, partially broken `/api/mind/scene` proxy).
P3.14 deletes the duplicate and merges the bounded mental-world signal
into the Pi's existing consciousness particle visualizer
(`pi/ui/static/particles.js`) by riding the existing `consciousness_feed`
WebSocket transport.

**Wire contract — exactly seven scalars in the new `scene` block:**
`enabled` (bool), `entity_count` (int), `relation_count` (int),
`cleanup_accuracy` (float [0,1]), `relation_recovery` (float [0,1]),
`similarity_to_previous` (float [0,1]), `spatial_hrr_side_effects` (int).
**No** entity / relation / scene arrays. **No** `vector` /
`raw_vector` / `composite_vector` keys. **No** authority flags.

| Side | Files touched (additive) | Files deleted |
| --- | --- | --- |
| Brain | `brain/perception_orchestrator.py` (`_SCENE_FEED_KEYS` tuple, `_build_scene_block` helper, one new line in `_build_consciousness_feed`); `brain/tests/test_consciousness_feed_scene_block.py` (new, 9 cases) | — |
| Pi | `pi/ui/static/particles.js` (`consciousness.scene` sub-object, feed ingestion in `onConsciousnessData`, four subtle visual mappings) | `pi/ui/hrr_scene_view.py`, `pi/ui/static/hrr_scene.{html,css}`, `brain/tests/test_pi_hrr_scene_view.py`, mount block in `pi/main.py::start_ui_server` |

Four visual mappings (subtle, bounded, never frantic):

1. `relation_count` → `drawConnections()` distance threshold widens by
   `min(20, relation_count) * 0.8` (clamped after ~12 relations so the
   canvas never turns into a solid mesh).
2. `cleanup_accuracy` → companion ring at radius `r*2 + 11` sweeping a
   fraction of the circle equal to the cleanup score (only drawn when
   the lane is actually enabled).
3. `similarity_to_previous` → aurora-blob drift speed multiplied by
   `1 + (1 - similarity) * 0.6` (clamped to `[1.0, 1.6]`). Stable scene
   = calm aurora; thrashing = livelier drift.
4. `spatial_hrr_side_effects > 0` → pulsing thin red canary halo at
   radius `r*4`. **Architectural alarm** — must remain dark; if the
   spatial-HRR shadow ever reports nonzero side effects, the
   zero-authority contract violation becomes visible to the operator.

Operator scene-graph view stays on the brain dashboard `/hrr-scene` —
**single source of truth**.

Doc updates: `AGENTS.md` (rule 9 in P5 governance block),
`docs/plans/p5_internal_mental_world_spatial_hrr.plan.md` (Pi UI rule +
architecture diagram + commit-7 entry + file list),
`brain/dashboard/static/science.html` (Dashboard & Pi Surfaces list).

Acceptance criteria all green: 9/9 new scene-block tests, schema audit
0 violations, dashboard truth probe 0 findings, `/api/hrr/scene`
payload unchanged, Pi LCD renders particles normally, `/mind` route
returns 404. Evidence:
`docs/validation_reports/p3_14_pi_mind_merge-2026-04-25.md`.

### Operational observation — emergent addressee overreach (2026-04-25 15:11 EDT)

First captured instance of *socially inappropriate but technically legal*
engagement in a multi-party room conversation. Conversation
`9befc079-8c15-4079-b760-01b7f01fe5b1`. Wake fired during a side
conversation about contractor payments; JARVIS correctly reasoned about
the overheard math (perception mature) but should have stayed quiet
(inhibition immature).

Soft-signal stack at engagement time: `gesture=leaning_away` (conf 0.5),
`attention.speaker_confidence=0.411`, `identity_state.confidence=0.618`,
`tool_route=MEMORY` with `memories_retrieved.count=0`, no second-person
construction in the transcript.

**This is not a bug.** No autonomy / authority / canonical / HRR / P5
impact. The *components* for an inhibition gate exist (gesture,
diarization, identity confidence, addressee, rapport, P3.8
`speaker_profile`); what is missing is a *policy* that downgrades
engagement when the post-wake soft signals stack against being
addressed. Recorded as evidence for the eventual P3.8 promotion gate.

Watch conjunction + repetition policy in §9 of
`docs/validation_reports/emergent_addressee_overreach_observation-2026-04-25.md`.
If 3+ instances of the conjunction recur within a 7-day rolling window,
escalate to a real engineering lane (provisional name **P3.15 —
Addressee inhibition gate for soft-signal multi-party context**). Until
then, observe only.

### Cumulative invariants at sprint landing

| Invariant | State |
| --- | --- |
| Continuity preserving (no brain wipe) | **honored** — live brain still on `192.168.1.222` |
| L3 / autonomy promotion | **none** — operator-gated only |
| Tier-2 specialist promotion | **none** — all five at CANDIDATE_BIRTH |
| HRR/P5 authority flags | **all `False`** unchanged |
| `spatial_hrr_mental_world` status marker | **`PRE-MATURE`** unchanged |
| `no_raw_vectors_in_api` | **`True`** on every payload |
| Schema emission audit | **0 violations** post-sprint |
| Dashboard truth probe | **0 findings** post-sprint |
| Three-axis Phase 6.5 invariants | **TLC-verified** over 9.7 M states / 0 counterexamples |
| Append-only audit ledger | **honored** (TLC verified) |
| Pi `/mind` route | **deleted** (404) |
| `/api/hrr/scene` operator surface | **unchanged** (single source of truth) |

### Files added (per lane)

| Lane | New files |
| --- | --- |
| P3.1 | `docs/formal/phase_65.tla`, `docs/formal/phase_65.cfg`, `docs/formal/README.md` |
| P3.6 | `brain/hemisphere/positive_memory_encoder.py` |
| P3.7 | `brain/hemisphere/negative_memory_encoder.py` |
| P3.8 | `brain/hemisphere/speaker_profile_encoder.py` |
| P3.9 | `brain/hemisphere/temporal_pattern_encoder.py` |
| P3.10 | `brain/hemisphere/skill_transfer_encoder.py` |
| P3.14 | `brain/tests/test_consciousness_feed_scene_block.py` |
| Validation reports | All 11 reports listed in the lane summary table above |

### Files deleted (P3.14)

`pi/ui/hrr_scene_view.py`, `pi/ui/static/hrr_scene.html`,
`pi/ui/static/hrr_scene.css`, `brain/tests/test_pi_hrr_scene_view.py`.

---

## P4 HRR Shadow Substrate + P5-S0 Internal Mental World + P5.1 Runtime-Flag Persistence (2026-04-24 → 2026-04-25, SHIPPED, PRE-MATURE)

A three-stage research-lane landing executed live against the desktop brain
(`192.168.1.222`) via `sync-desktop.sh`. All three stages preserve the
**shadow-only, derived-only, zero-authority** boundary: nothing written to
memory, nothing written to beliefs, no policy influence, no autonomy
influence, no raw HRR vectors in any API response, and the dashboard
status marker `spatial_hrr_mental_world` remains **`PRE-MATURE`** through
every stage. The lane earns nothing. It just *exists* now, observable and
auditable, waiting on the normal specialist-lifecycle + Phase 6.5
promotion path.

### Stage 1 — P4 Holographic Reduced Representations (HRR/VSA) shadow substrate

NumPy-native HRR primitive library + recording lane, wired end-to-end into
the engine tick but with zero authority. Commits `9855f21` → `85b7c4b`.

| Component | Path | Role |
| --- | --- | --- |
| HRR primitives | `brain/library/vsa/hrr.py`, `symbols.py`, `runtime_config.py`, `status.py`, `samples.py`, `hrr_encoder.py` | Circular-convolution bind/unbind, cleanup memory, cosine similarity, ring buffer of recent bound samples. NumPy FFT backend, no Torch runtime dep. |
| Engine hook | `brain/consciousness/engine.py` | Boot-time construction of `HRRRuntimeConfig`, twin-gated world-shadow encoder, single `INFO` log line summarising which precedence layer enabled the lane, `register_runtime_config` call. |
| Mental simulator hook | `brain/cognition/mental_simulator.py` | `record_hrr_shadow(...)` mirror of thought streams into the HRR ring buffer — *mirror only*, never authoritative. |
| Dashboard surface | `brain/dashboard/app.py`, `brain/dashboard/snapshot.py`, `brain/dashboard/static/hrr.html`, `dashboard.js`, `renderers.js`, `style.css` | `/api/hrr/status`, `/api/hrr/samples`, `/api/hrr/scene*` endpoints; live HRR panel under the Memory tab; truth-strip freshness banner. |
| Governance | `AGENTS.md` (P4 HRR/VSA addendum), `docs/plans/p4_holographic_cognition_vsa.plan.md` | Zero-authority rule, `hrr_side_effects=0` invariant, raw-vector ban in LLM layer, `PRE-MATURE` lock. |

**Twin-gate.** Off by default on public clones. Requires **both**
`ENABLE_HRR_SHADOW=1` and later `ENABLE_HRR_SPATIAL_SCENE=1` (Stage 2) to
activate. Disabled lane yields a stable empty payload with every authority
flag pinned `False`.

**Soak proof.** 7-day live shadow soak on the desktop brain held all
invariants (`p4_hrr_live_shadow_soak-2026-04-24.md`): samples accumulating
in the ring, zero authority drift, zero raw-vector leakage, schema audit
`0 violations`, truth probe `0 findings`, status marker unchanged.

**Validation reports.**
`docs/validation_reports/p4_hrr_stage0_baseline-2026-04-24.md`,
`p4_hrr_shadow_substrate-2026-04-24.md`,
`p4_hrr_world_shadow-2026-04-24.md`,
`p4_hrr_live_shadow_soak-2026-04-24.md`.

### Stage 2 — P5-S0 Internal Mental World / Spatial HRR Scene Graph

A **derived**, **shadow-only**, twin-gated mental-world layer that turns
canonical `SceneSnapshot` + `SpatialEstimator` + `SpatialTrack` +
`SpatialAnchor` state into an HRR-composed
`subject ⊗ relation ⊗ object` scene graph. P5 **never** produces spatial
truth; it only re-renders what canonical perception already emitted.
Commits `61e2531`, `4124b54`, `1d68427`, `afefe47`.

Nine-commit sprint plan in
`docs/plans/p5_internal_mental_world_spatial_hrr.plan.md`.

| Component | Path | Role |
| --- | --- | --- |
| Spatial vocabulary | `brain/library/vsa/spatial_symbols.py` | `LEFT_OF`, `RIGHT_OF`, `ABOVE`, `BELOW`, `IN_FRONT_OF`, `BEHIND`, `NEAR`, `CENTERED_IN`. Frozen symbol table. |
| Scene-graph adapter | `brain/cognition/spatial_scene_graph.py` | Derives `MentalWorldSceneGraph` from canonical perception via read-only accessor `perception_orchestrator.get_scene_snapshot()`. `LATERAL_SEPARATION_MIN_M` threshold suppresses false left/right relations. |
| HRR spatial encoder | `brain/library/vsa/hrr_spatial_encoder.py` | Binds each `(subject, relation, object)` triple with shared `HRRConfig`; retains only derived metrics (`similarity`, `cleanup_confidence`) for API emission. |
| Mental-world facade | `brain/cognition/mental_world.py` | `/api/hrr/scene`, `/api/hrr/scene/history`, `/api/full-snapshot::hrr_scene`. Always strips `vector` / `raw_vector` / `composite_vector` keys before serialisation; pins `AUTHORITY_FLAGS` `False` + `no_raw_vectors_in_api=True`. |
| Mental navigation shadow | `brain/cognition/mental_navigation.py` | Pure functional operations on `MentalWorldSceneGraph` (select, filter, move). `AUTHORITY_FLAGS` pinned `False`. |
| Dashboard scene view | `brain/dashboard/static/hrr_scene.html` + `app.py::/hrr-scene` | Operator-facing Matrix view of the derived scene; labels only, no geometry. |
| Pi kiosk | `pi/ui/hrr_scene_view.py`, `pi/ui/static/hrr_scene.{html,css}`, `pi/main.py` | `/mind` + `/api/mind/scene`. Pi never computes HRR math; it proxies `/api/hrr/scene`. Safe-empty fallback when the brain is unreachable. |
| Mutation guard | `brain/perception_orchestrator.py::get_scene_snapshot()` | Returns `copy.deepcopy` of `_last_scene_snapshot` — P5 can never mutate the canonical snapshot. Enforced by `test_p5_does_not_mutate_scene_snapshot`. |
| Governance | `AGENTS.md` (P5 addendum, rules 1–11), `docs/MASTER_ROADMAP.md` (P5.0 entry) | Canonical-perception-only rule, lane `spatial_hrr_mental_world`, fixture vs live separation, import bans. |

**Validation-pack checks (all non-critical, never auto-promote the
marker).** `p5_mental_world_status_marker`, `p5_mental_world_structure`,
`p5_mental_world_fixture_ok`, `p5_mental_world_live_ok`. Plus
`dashboard_truth_probe::check_hrr_scene_authority` which hard-fails on
any authority-flag drift, status / lane drift, or raw-vector leak in the
`/api/full-snapshot::hrr_scene` subtree.

**Live verification — safe-empty branch (2026-04-24).**
`/api/hrr/scene` returns `reason=canonical_spatial_state_unavailable`,
`entity_count=0`, `status=PRE-MATURE`, all authority flags `False`,
`no_raw_vectors_in_api=True`. Truth probe `0 findings`, schema audit
`0 violations`. Recorded in
`docs/validation_reports/p5_spatial_hrr_mental_world-2026-04-24.md`.

**Live verification — populated branch (2026-04-25).** After canonical
perception populated a live `SceneSnapshot`, the same endpoint served
`entity_count > 0`, `relation_count > 0`, derived-relations emitted,
HRR metrics present, authority flags still all `False`, no vector keys
anywhere in the payload. History endpoint non-empty. Marker unchanged.
Recorded in
`docs/validation_reports/p5_spatial_hrr_populated_scene-2026-04-25.md`.

**Defect found and fixed during verification.** The engine instance was
missing the `_perception_orchestrator` wire, causing the derivation hook
to silently receive `None` for the first several ticks. `main.py` now
assigns `engine._perception_orchestrator = perc_orch` at boot; populated
branch has been green since.

### Stage 3 — P5.1 HRR runtime-flag persistence (sovereignty layer)

An opt-in persistent runtime-flag file so an operator's own brain can
keep HRR shadow + the P5 spatial-scene lane enabled across restarts
**without** exporting `ENABLE_HRR_SHADOW=1` / `ENABLE_HRR_SPATIAL_SCENE=1`
every boot, while keeping **public/fresh clones safe-by-default**.
Commit `5e94f72`.

**Precedence.** Highest layer wins:

```
safe defaults  →  ~/.jarvis/runtime_flags.json  →  env vars
```

Path overridable via `$JARVIS_RUNTIME_FLAGS` (used by tests). Recognised
JSON keys: `enable_hrr_shadow`, `enable_hrr_spatial_scene`,
`hrr_sample_every_ticks`, `hrr_spatial_scene_sample_every_ticks`,
`hrr_shadow_dim`. Unknown keys ignored, malformed JSON falls back to
safe defaults and surfaces a clear `runtime_flags_error` string (lane
stays OFF in that case).

**API provenance surface.** `/api/hrr/status`, `/api/hrr/scene`, and
`/api/hrr/scene/history` now carry `enabled_source`,
`spatial_scene_enabled_source`, the full `flag_sources` map,
`runtime_flags_path`, and `runtime_flags_error`, so the dashboard can
show **which precedence layer** enabled the lane. Engine boot emits a
single `INFO` line summarising the resolved config and registers the
boot-time `HRRRuntimeConfig` — no mid-run re-reads.

**Operator tooling.** `brain/scripts/set_hrr_runtime_flags.py` with
`--enable`, `--disable`, `--status` subcommands; prints the resolved
config + the required supervisor-restart reminder. Restart is still
required — the config is read **once** at engine boot, by design.

**Tests (20 cases).** `brain/tests/test_hrr_runtime_flags.py` covers
layered precedence (default / file / env / partial), malformed-input
safety (bad JSON, non-object root, unknown keys, int clamping, truthy
strings), path resolution (explicit arg + `$JARVIS_RUNTIME_FLAGS`),
engine-registered-config wins over re-reads, the `/api/hrr/status`
surface, the mental-world facade's `get_state` / `get_history`
provenance injection, and all three helper-CLI subcommands. Local P4/P5
regression sweep: **142 passed, 4 skipped** (skips are aiohttp-only Pi
tests, environment-dependent).

**Governance.** `AGENTS.md` rule #12 under the P5 addendum makes the
precedence explicit and declares the runtime-flags file may **never** be
used to flip an authority flag, promote the
`spatial_hrr_mental_world` status marker, or bypass the truth-probe /
schema-audit / validation-pack guards.

**Live verification — desktop brain restarted with NO env exports
(2026-04-25).** Boot log:

```
HRR runtime flags: enabled=True (source=runtime_flags),
    spatial_scene=True (source=runtime_flags),
    runtime_flags_path=/home/duafoo/.jarvis/runtime_flags.json
HRR shadow world encoder enabled (dim=1024, every=50 ticks)
HRR spatial mental-world shadow enabled (dim=1024, every=50 ticks)
```

`/api/hrr/status` reports `enabled=true`,
`spatial_scene_enabled=true`, `enabled_source="runtime_flags"`,
`spatial_scene_enabled_source="runtime_flags"`,
`runtime_flags_error=null`, all authority fields `false`. Truth probe
`0 findings`. Schema audit `0 violations`. Status marker still
`PRE-MATURE`. Full evidence in
`docs/validation_reports/p5_hrr_runtime_flags-2026-04-25.md`.

### Cumulative invariants at landing

| Invariant | State |
| --- | --- |
| `spatial_hrr_mental_world` status marker | **`PRE-MATURE`** (unchanged) |
| P4 & P5 authority flags (`writes_memory`, `writes_beliefs`, `influences_policy`, `influences_autonomy`, `soul_integrity_influence`, `llm_raw_vector_exposure`) | **all `False`** |
| `no_raw_vectors_in_api` | **`True`** on every HRR payload |
| Public/fresh-clone default | **OFF** for both flags |
| Dashboard truth probe | **0 findings** |
| Schema emission audit | **0 violations** |
| P4/P5/P5.1 regression sweep | **142 passed, 4 skipped** |
| Three-axis autonomy invariants (P6.5) | unchanged |
| Required for flag change to take effect | **supervisor/main restart** (by design) |

### Next on the P5 lane (not in this landing)

* Long-run spatial-scene soak (P4-style 7-day) → drives
  `p5_mental_world_live_ok=true` persistently in the validation pack.
* Pi `/mind` live smoke when the Pi node is up alongside the brain.
* Formal promotion criteria for the `spatial_hrr_mental_world` marker
  (`PRE-MATURE` → `PARTIAL`) — **not** granted by this work.

---

## TODO_V2 Cleanup + Unblock Sprint — Reset Gate Closed (2026-04-24, SHIPPED)

Post-launch cleanup + unblock sprint executed against the live desktop brain
(`192.168.1.222`) via `sync-desktop.sh`. Scope: close the two real P3.5 code
bugs and the thirteen smaller Matrix / Tier-2 / TLA+ / epistemic / Language
Kernel items surfaced by `docs/validation_reports/todo_v2_trace_validation-2026-04-24.md`
before any destructive reset ceremony. The reset is still DEFERRED —
continuity-preserving posture holds.

### Passes landed

| Pass | Scope | Evidence |
|------|-------|----------|
| Pass 0 | `TODO_V2.md` trace reconciliation (16 mismatches) + `todo_v2_trace_validation-2026-04-24.md` | validation report |
| Pass 1 | **P3.5 Matrix unblock fixes.** `policy/shadow_runner.py::set_hemisphere_signals()` wired through `consciousness/engine.py::_on_tick`; `hemisphere/orchestrator.py` stamps `specialist_verification_ts` at `PROBATIONARY_TRAINING → VERIFIED_PROBATIONARY`; latent `EventBus.emit(MATRIX_EXPANSION_TRIGGERED, dict)` `TypeError` dead-code path fixed by `**kwargs` unpack at `orchestrator.py:1655`. | `p3_5_m6_expansion_wiring-2026-04-24.md` |
| Pass 2 | P3.1 TLA+ spec — `docs/formal/phase_65.tla` + `phase_65.cfg` with strength domain `none \| archived_missing \| verified` matching code. | `docs/formal/` |
| Pass 3 | P3.2 `user_correction` belief-edge production writer — `bridge.create_user_correction_link` + `on_user_correction` hook; schema audit whitelist tightened. | schema audit 0 violations |
| Pass 4 | P3.3 `depends_on` belief-edge production writer — `bridge.create_prerequisite_link` with forward-cycle guard + `on_prerequisite_detected` hook; integrity + topology tests green. | schema audit 0 violations |
| Pass 5 | P3.4 Language Kernel seed registration — `brain/scripts/seed_language_kernel.py` registers current Phase C checkpoint; dashboard + `/api/meta/status-markers::phase_e_language_kernel_identity` flip to `SHIPPED` dynamically when `registry.status=registered ∧ total_artifacts≥1 ∧ live_artifact`. | `p3_4_language_kernel_seed-2026-04-24.md` |
| Pass 6 | P3.6–P3.10 Tier-2 specialist lifecycle regression tests using canonical lifecycle vocabulary (`CANDIDATE_BIRTH`, `PROBATIONARY_TRAINING`, `VERIFIED_PROBATIONARY`, `BROADCAST_ELIGIBLE`, `PROMOTED`, `RETIRED`). Low-utility probationary retirement assertion loosened to match actual `pruned` list contents (`arch.id or arch.name`). | `p3_6_to_p3_10_tier2_matrix_lifecycle-2026-04-24.md` |
| Pass 7 | P3.11 `dream_synthesis` Tier-2 feasibility — class distribution pulled honestly (839 signals, 0.32 accuracy, 1 active network); promotion gated on distribution+task-quality floors, not signal volume. | `p3_11_dream_synthesis_feasibility-2026-04-24.md` |
| Pass 8 | P3.12 long-horizon attention — use case pinned (`medication_refill_followup`) + success metric + degradation guard. No code yet. | `LONG_HORIZON_ATTENTION_USE_CASE.md` |
| Pass 9 | P3.13 IntentionResolver Stage 1 — **explicitly deferred as of 2026-04-24** (live gates unmet: `backed_commitments: 0/30`, `recent_resolutions: 0/5`, `resolution_histogram_variance: 0/2`). Superseded by the 2026-05-08 shipped shadow implementation above; delivery authority remains evidence-gated. | `p3_13_intention_resolver_deferred-2026-04-24.md` |

### Reset gate verification (desktop brain, post three restart cycles)

- Full regression suite: stable on live brain with `torch` available.
- Schema emission audit: **0 violations** (now includes live `user_correction` + `depends_on` emitters).
- Dashboard truth probe: 0 findings.
- `/api/language-kernel` reports `status=registered`, live artifact present.
- `/api/meta/status-markers::phase_e_language_kernel_identity = SHIPPED`.
- Three-axis autonomy invariants unchanged (`current_ok=false`, `prior_attested_ok=true`, `activation_ok=false`).
- No `TODO_V2.md` acceptance criterion contradicts the code.
- P3.13 explicitly deferred at that time, not half-started. Current state:
  shipped shadow implementation as of 2026-05-08; delivery authority remains
  evidence-gated.

Final evidence: `docs/validation_reports/reset_gate_status-2026-04-24.md`.

### Next lane announcement — P4 Holographic Cognition / HRR-VSA (PRE-MATURE)

With the reset gate closed, the next research lane is **P4: Holographic
Reduced Representations (HRR/VSA)** as a derived neural-intuition substrate
for compositional internal representations
(`subject ⊗ relation ⊗ object`). HRR starts **PRE-MATURE / shadow-only**
and must earn its way into the system through the normal specialist
lifecycle + Phase 6.5 governance. Non-negotiable boundaries: HRR is not
canonical truth, does not write beliefs directly, does not influence
policy until promoted, must keep `hrr_side_effects=0`, and raw vectors
must never enter the LLM articulation layer.

Plan: `docs/plans/p4_holographic_cognition_vsa.plan.md`. Governance rules
appended to `AGENTS.md`. First sprint (`P4-S0`) is the primitive library +
synthetic proof only — no production cognition files modified.

---

## Open-Source Release Truth Pass Phase 1 + Continuity Pivot (2026-04-23, SHIPPED)

Ten pre-release commits (`f407541` → `2a206f1`) closing genuinely missing
engineering gaps, plus a **continuity pivot** that replaces the original
destructive reset ceremony with a continuity-preserving runtime proof.

### Phase 1 commits landed

| Commit | Scope |
|--------|-------|
| `f407541` | P1.1 Observability truth baseline: specialists shape fix, `GET /api/self-test`, `dashboard_truth_probe.py`, 3 regression tests |
| `c739419` | P1.2 Diarization registry closure: `test:`-prefixed evidence names + `turn_boundary_f1` metric (truth-first gating) |
| `ae02928` | P1.3 Belief graph causal + temporal_sequence writers + `schema_emission_audit.py`, 20 regression tests |
| `2fe404e` | P1.4 Voice Intent NN Shadow Takeover (Ship B): `IntentShadowRunner`, SHADOW/ADVISORY/PRIMARY levels, persistence, dashboard tab, 21 tests |
| `c0b735d` | P1.5 Phase E Language Kernel artifact identity: `LanguageKernelArtifact` + `LanguageKernelRegistry`, artifact-level rollback, dashboard tab, 13 tests (PRE-MATURE) |
| `f5109d1` | P1.6 `docs/WAKE_RELIABILITY_TUNING.md` extracted from TODO + report script header |
| `8b04130` | P1.7 Dashboard ground-truth rebuild: `/api/meta/build-status`, `/api/meta/status-markers`, `/api/maturity-gates`, `/api/build-history`, `/maturity` page, freshness banner + status-marker shared scripts, System Truth Score strip |
| `d3e53ad` | P1.8 Contributor docs: README Proto-ASI Honesty section, CONTRIBUTING full rewrite (swim-lane map, maturity semantics, No Verb-Hacking + Restart Continuity rules, canonical extension examples), `docs/FIRST_HOUR_AS_A_RESEARCHER.md` |
| — | P1.9 Regression tests: 8 per-lane test files, 122/122 green on desktop brain |
| `2a206f1` | P1.10 Pre-reset evidence artifact at `docs/validation_reports/pre_reset_truth_pass-2026-04-23.md` |

### Continuity pivot (replaces destructive reset ceremony)

The plan originally called for wiping `~/.jarvis/*` to zero and re-gestating.
That step is **reclassified `DEFERRED / OPTIONAL destructive proof`**. The
current brain contains useful runtime data (attestation ledger, 602 language
evaluations, 119-sample corpus, durable audit entries, specialist signals,
maturity traces) that cannot be re-earned on demand. A cold-start
reproducibility proof can still be run later if explicitly chosen.

Preserved state:

```
path:   ~/jarvis_continuity_backup_2026-04-23.tar.gz
size:   46 GiB (48,767,544,788 bytes)
files:  198 files + 46 directories (matches live ~/.jarvis)
sha256: b9e92bf603aa3f7751be9e2620ac3b896f7345b4387b62ca80b559de498e0f6f
gzip:   integrity verified
```

Baseline evidence: `docs/validation_reports/continuity_baseline_2026-04-23.md`.

### Verification at pivot point

- **Full regression suite:** 122/122 green on desktop brain.
- **Schema emission audit:** 0 violations (5/5 edge types, 9/9 evidence
  bases, 23/23 hemisphere focuses, 4/4 teacher keys — all emitted or
  explicitly whitelisted as future-only).
- **Dashboard truth probe:** 0 findings.
- **Runtime validation pack:** 18/35 current, 21 ever-proven, 3 regressed —
  honestly BLOCKED on maturity-gated Language evidence floors (PRE-MATURE,
  not regressions).
- **Three-axis autonomy invariants:** all hold. `current_level=2`,
  `current_ok=false` (live policy refuses 47% win_rate vs 50% gate),
  `prior_attested_ok=true` (hash-verified operator-seeded record),
  `activation_ok=false` (correctly blocked: attestation unlocks
  requestability, never auto-activation).

### Why this shape

The open-source release objective is honesty, not cold-start reproducibility.
A wipe would make the dashboard temporarily *look* cleaner at the cost of
destroying the data that makes maturity gates *mean* something. The
continuity-preserving path keeps the runtime truth accumulating while the
release surfaces are frozen and audited.

### Phase 2 runtime proofs (continuity-safe, cumulative)

Each P2.x item runs against the live, un-wiped brain and writes a standalone
evidence artifact under `docs/validation_reports/`. Cumulative results:

| Item | Commit | Artifact | Result |
|------|--------|----------|--------|
| P2.1 Phase 7 `isolated_subprocess` runtime proof | `d5fdc4f` | `phase_7_isolated_subprocess_runtime_proof-2026-04-23.md` | `ceremony_dateutil_demo` plugin provisioned + invoked 2/2 success; 36.42 MB venv; `python-dateutil==2.9.0` pip-installed + imported inside child; latent `_verify_imports` bug fixed via new `PluginManifest.verify_imports` override; 122/122 green |
| P2.2 Synthetic `route_coverage` sweep | `54e8580` | `phase_2_2_synthetic_route_coverage-2026-04-23.md` | 97/100 utterances (3 reconnects self-healed); STT=83 routes=83; 332 distillation records; 13 distinct routes exercised; 166 `blocked_side_effects` (truth boundary active); all 5 invariant leak counters = 0 (memory, identity, llm, tts, transcription_emit); schema audit + dashboard truth probe both clean post-run |
| P2.3 Phase 5 proof chain traceability | `b3c3b08` | `phase_5_proof_chain-2026-04-23.md` | Both chains wired and producing evidence — Chain A (autonomy feedback loop): 161 policy outcomes, 72 non-warmup, 13+ tag clusters, trace `ri_7a8aeab7c338` (+0.0786 delta, worked=true); Chain B (SI orchestrator): Stage 2 human-approval live, `imp_3598dbe6bc` traced end-to-end through 5Ws+sandbox+review; hash-verified attestation of prior L3 chain closure (208 outcomes, 79% win rate). `phase5_weakness_signal` shows `FAIL` in validation pack honestly because current live brain has no *active* regression-heavy clusters — gate semantics, not wiring break. |
| P2.4 Phase 6.5 continuity re-verification | `e7641cb` | `phase_6_5_continuity_reverify-2026-04-23.md` | Three-axis invariants held across 5 post-ship process restarts: `current_ok=false` (live-sourced, 34 wins / 47.2% win_rate), `prior_attested_ok=true` (hash-verified, byte-unchanged since seed), `activation_ok=false`, `request_ok=true`. Audit ledger: 11 entries, 0 `l3_promoted.outcome != clean` violations, 0 unmatched 2→3, 0 stealth writes. Escalation store preserves rejected record without leakage. `autonomy_state.json` correctly re-hydrates L2 on each boot. Policy memory accumulated +6 wins / +2.8pp honestly since ship — still below 50% activation gate. |
| P2.5 Export-readiness audit | `1a6ef6a` | `export_readiness_audit-2026-04-24.md` | Scope pivoted from `~/.jarvis/*` classification to actual code export surface. 0 committed secrets, 0 hardcoded `/home/nimda` or `/home/duafoo` in `brain/`+`pi/` code, IPs in `.env.example` and docstrings templated/example-only, 2 deploy scripts (`sync-desktop.sh`, `sync-pi.sh`) flagged OPERATOR DECIDES, `.claude/` flagged DROP-before-ship, `.gitignore` adequate. **0 release-blocking findings.** |
| P2.6 Dashboard prose refresh | `24cd7b2` | `dashboard_prose_refresh-2026-04-24.md` | Full audit of 7 static pages found existing prose is already continuity-aware — "fresh brain" is accurate onboarding copy, "cold-start" is a technical term unrelated to wipes, "Pre-Reset" badges are honest attestation citations. Two surgical additions: `docs.html` (Phase 6.5→P2.4 follow-up note + "Open-source release policy: continuity-preserving" banner in Restart Resilience), `maturity.html` hero (dual-path paragraph fresh-clone vs existing-operator). Zero prose removed, zero facts softened. |
| **P2.7 Launch-day verification** | (this commit) | **`launch_day_verification-2026-04-24.md`** | Full TODO + MASTER_ROADMAP reconciliation certifies release-ready. Regression: **4158 passed** / 4 pre-existing flakes+drift / 5 skipped / 3 deselected, **zero regressions** from release pass. Truth audits: `schema_emission_audit` 0 violations, `dashboard_truth_probe` 0 findings, validation pack `Language Runtime Guardrails Safe: PASS (bridge=False, mode=off, unpromoted_live=0, live_red=0)`. Three-axis autonomy invariants intact. Audit ledger: append-only, 13 entries, 0 `l3_promoted.outcome != clean`. All 3 Phase 7 post-ship residuals closed (child-death handling mechanically proven via code + unit tests + P2.1 live lifecycle; `/api/plugins` exposes `execution_mode`+`venv_ready`+subprocess counters truthfully; 0 lingering direct `PluginRegistry()` in hot paths). All 3 stale Phase E plan checkboxes closed (shipped with P1.5 `c0b735d`). Phase D rollout stages banner landed as post-release operator runbook. Voice Intent NN Shadow confirmed as explicitly-gated future scope. 5 test-surface alignment fixes landed (sandbox shim re-export, claim encoder denominator 13→15, `language` dir added to `COPIED_SUBDIRS`+`_MODULE_TO_TESTS`, `ALLOWED_PATHS` root-package check) — no behavior changes. **Release-ready at Stage 0 baseline (bridge OFF).** |

The truth boundary that was added to the perception pipeline in February
after a prior `route_coverage` soak found 22 identity + 2 memory leaks
holds end-to-end under the same workload today. Distillation signal (332
records) was accepted without any of it escaping into memory, identity,
conversation, or TTS.

P2.7 produced the launch-day consolidated evidence bundle. All SHIPPED
claims on the dashboard surface have live evidence. All remaining `[ ]`
items in `TODO.md` are classified as either post-release operator
runbook, explicitly-gated future scope, or maturity-gated
post-deployment tuning — none block the 2026-05-10 cut.

---

## Phase 6.5: L3 Escalation Governance Layer — Attestation + Audit + Smoke (2026-04-18 → 2026-04-23, SHIPPED)

Governance infrastructure for L3 autonomy escalation and human-in-the-loop
approval. Landed across **six commits** over five days, then exercised
end-to-end on the live desktop brain on 2026-04-23 covering all five
shipping criteria plus a full process restart. Treats autonomy as
explicit internal governance (not one confidence blob): six evidence
classes (`current_ok`, `prior_attested_ok`, `request_ok`,
`approval_required`, `activation_ok`, `attestation_strength`) are kept
structurally separate and mechanically tested.

**Status: SHIPPED 2026-04-23.** Live-brain evidence artifact at
`docs/validation_reports/phase_6_5_live_evidence-2026-04-23.md`.

### Why this shape

An ASI-adjacent control architecture needs explicit separation between
evidence classes, not one fuzzy confidence output. A believable JARVIS /
HALO-style system has to be able to say *"I proved this before, but I have
not re-earned it yet in this runtime"* — and it has to say that without
letting historical proof impersonate current runtime health. That is exactly
what the `current_ok` vs `prior_attested_ok` split codifies. Attestation
unlocks *requestability*, not live eligibility.

### Hard invariants (mechanically tested)

1. **No L3 auto-promotion, ever.** The internal promotion loop only emits
   `AUTONOMY_L3_ELIGIBLE`; 2→3 transitions require `POST
   /api/autonomy/level` with an explicit `evidence_path`.
2. **`current_ok` is strictly live-sourced.** Never backfilled from
   attestation, `ever_ok`, or persisted autonomy state.
3. **`AUTONOMY_L3_PROMOTED` fires only on clean 2→3.** `outcome="clean"` is
   the only legal value. Denials live on `AUTONOMY_L3_ACTIVATION_DENIED`;
   rollbacks of the triggering escalation live on
   `AUTONOMY_ESCALATION_ROLLED_BACK`.
4. **Per-request `declared_scope`, never global.** Approved escalations
   widen the allowed path set only for a single call; orchestrator
   `ALLOWED_PATHS` is never mutated.
5. **Attestation ledger has no write path to `maturity_highwater.json`,
   autonomy state, or any `ever_*` counter.** Structural, not convention.
6. **Audit ledger is a read-only bus observer.** Disk failures are logged
   and swallowed; audit faults never block cognition.

### Components shipped

| Commit | Layer | Files |
|---|---|---|
| `1e529d1` + `607d46c` | **Escalation control path**: `EscalationStore` (pending/activity JSONL), metric trigger (`METRIC_ESCALATION_POLICY` + `PER_METRIC_RATE_LIMIT_S`), approval flow with `declared_scope`, REST endpoints on `/api/autonomy/escalations*`, 37 tests | `brain/autonomy/escalation.py`, `brain/autonomy/orchestrator.py`, `brain/dashboard/app.py`, `brain/tests/test_l3_escalation.py` |
| `2bf5d56` | **Attestation ledger + boundary**: hash-attested operator-seeded capability records under `~/.jarvis/eval/ever_proven_attestation.json`; `artifact_status` ∈ {`hash_verified`, `hash_mismatch`, `missing`, `hash_unverifiable`}; `attestation_strength` ∈ {`verified`, `archived_missing`}; `_build_l3_escalation_cache` / `_build_attestation_cache` in snapshot; validation-pack L3 checks gated on `autonomy.l3` sub-dict; **no-backfill regression test**. | `brain/autonomy/attestation.py`, `brain/dashboard/snapshot.py`, `brain/jarvis_eval/validation_pack.py`, `brain/tests/test_l3_snapshot_caches.py`, `brain/tests/test_validation_pack.py`, `brain/tests/test_l3_promotion_invariant.py` |
| `c0327fd` | **L3 Escalation UI**: new "L3 Escalation" tab on `self_improve.html` with three panels (Current Live Health / Prior Attestation / Escalation Queue + Lifecycle) + explicit "Evidence boundary" footer; three-axis badges on the autonomy stage-ladder row in `dashboard.js` (green / amber `archived_missing` / red / muted). | `brain/dashboard/static/self_improve.html`, `brain/dashboard/static/dashboard.js` |
| `7474951` | **Durable audit subscriber + event taxonomy fix**: `AutonomyAuditLedger` (new) subscribes to 10 `AUDITED_EVENTS` and appends JSONL to `~/.jarvis/autonomy_audit.jsonl` with 10 MB rotation; `GET /api/autonomy/audit?limit=N` endpoint; ledger wired in `main.py` before `reconcile_on_boot()`; `AUTONOMY_L3_PROMOTED` split to `outcome="clean"` only; new `AUTONOMY_ESCALATION_PARKED` and `AUTONOMY_ESCALATION_EXPIRED` events for previously-silent state transitions. | `brain/autonomy/audit_ledger.py` (new), `brain/consciousness/events.py`, `brain/autonomy/escalation.py`, `brain/autonomy/orchestrator.py`, `brain/main.py`, `brain/dashboard/app.py`, `brain/tests/test_autonomy_audit_ledger.py` |
| `a9e5a6b` | **End-to-end smoke integration test**: full Phase 6.5 contract exercised in-process — seeded attestation flips `prior_attested_ok`, `request_ok` lights up without contaminating `current_ok` or any non-L3 `ever_ok`; manual promotion emits exactly `autonomy:level_changed(2→3)` + `autonomy:l3_promoted(outcome="clean")`; durable audit ledger persists the sequence; a fresh ledger instance re-reading the same file surfaces identical events (restart survival). Isolated tmp-path ledgers + fresh `EventBus` with open barrier. | `brain/tests/test_phase_6_5_smoke.py` (new) |

### Evidence classes (what the dashboard / API surface now expose)

| Field | Source | Meaning |
|---|---|---|
| `current_ok` | Live `check_promotion_eligibility()` | L3 earned *this session* |
| `prior_attested_ok` | `AttestationLedger.prior_attested_records()` | L3 *previously proven*, hash-attested |
| `request_ok` | `current_ok OR prior_attested_ok` | Eligible to *request* L3 promotion |
| `approval_required` | `not activation_ok` | Human approval still gates activation |
| `activation_ok` | `live_autonomy_level >= 3` | L3 is actually active right now |
| `attestation_strength` | `verified` / `archived_missing` / `none` | Trust quality of the attestation evidence |

### Event taxonomy (post-fix)

| Event | When it fires | Payload contract |
|---|---|---|
| `AUTONOMY_L3_ELIGIBLE` | Fresh live-runtime L3 eligibility observed this session | Live-only fields: `reason`, `wins`, `win_rate`, `regressions_in_last_10` |
| `AUTONOMY_L3_PROMOTED` | **Only** on clean 2→3 transition | `outcome="clean"` (literal), `prior_level=2`, `evidence_path`, `approval_source`, `caller_id`, `promoted_at` |
| `AUTONOMY_L3_ACTIVATION_DENIED` | L3 set attempt refused (missing evidence_path, failed preconditions) | `reason`, `caller_id`, `approval_source`, `current_level`, `denied_at` |
| `AUTONOMY_LEVEL_CHANGED` | Any level transition | `old_level`, `new_level` |
| `AUTONOMY_ESCALATION_REQUESTED` / `_APPROVED` / `_REJECTED` / `_ROLLED_BACK` | Escalation lifecycle transitions | Record fields |
| `AUTONOMY_ESCALATION_PARKED` | Approval re-queues approval pipeline into `awaiting_approval` | Record fields |
| `AUTONOMY_ESCALATION_EXPIRED` | `prune_expired()` promotes a record past its expiry | Record fields |

### API surface

- `GET /api/autonomy/level` — live autonomy level + `current_ok` /
  `prior_attested_ok` / `request_ok` / `activation_ok`
- `POST /api/autonomy/level` — manual level set (requires `evidence_path` +
  `reason`; operator override gated on `ALLOW_EMERGENCY_OVERRIDE=1`)
- `GET /api/autonomy/attestation` — current attestation records with derived
  `artifact_status` + `attestation_strength`
- `GET /api/autonomy/escalations` — pending + recent-lifecycle
- `POST /api/autonomy/escalations/{id}/approve` / `.../reject`
- `GET /api/autonomy/audit?limit=N` — durable audit-ledger tail (JSONL rows)

### Verification

- **649 tests green** at `7474951`; smoke test adds +1 at `a9e5a6b` (98/98
  green across the Phase 6.5 surface: smoke + L3 promotion + escalation +
  snapshot caches + audit ledger + validation pack).
- Live desktop restart after `7474951` confirmed `/api/autonomy/audit`
  active, durable ledger written on boot, subscriber catching reconcile
  events, fresh-brain state preserved (no stray attestation/escalation
  artifacts).
- In-process smoke regression-test confirms: attestation seed flips
  `prior_attested_ok` without backfilling `current_ok` or contaminating any
  non-L3 `ever_ok`; manual promote emits exactly two events with correct
  payloads; ledger persistence survives instance recreation.

### Shipping criteria (all five captured on live brain, 2026-04-23)

All five live-evidence criteria executed against the desktop brain at
`192.168.1.222:9200` on 2026-04-23. Full evidence artifact:
`docs/validation_reports/phase_6_5_live_evidence-2026-04-23.md`.

1. ✅ **Attestation seed exercised on live brain.** `autonomy.l3` record
   seeded from `docs/pre_reset_report_phase9_complete.md`
   (hash `sha256:020ed8b9…`); `/api/autonomy/level` surfaced
   `prior_attested_ok=true`, `attestation_strength="verified"`,
   `artifact_status="hash_verified"`, `request_ok=true`, with
   `current_ok=false` (live-sourced, independent).
2. ✅ **Manual L3 promotion via `POST /api/autonomy/level`.**
   `/api/autonomy/audit` recorded exactly
   `autonomy:level_changed(2→3)` followed by
   `autonomy:l3_promoted(outcome="clean", approval_source="prior_attested")`
   with the correct `caller_id` and `evidence_path` payloads.
3. ✅ **Escalation lifecycle exercised.** A `shadow_default_win_rate`
   escalation submitted with `declared_scope=["brain/policy/"]` was
   rejected via `POST /api/autonomy/escalations/{id}/reject`;
   `autonomy:escalation_rejected` event persisted to the audit ledger;
   pending list correctly reflects the terminal rejected state.
4. ✅ **Restart continuity.** After an operator-triggered restart, the
   audit ledger still surfaces all pre-restart events (new session
   added a natural L1→L2 promotion); the escalation store still holds
   the rejected record; the attestation ledger still produces
   `prior_attested_ok=true` with artifact hash match verified on disk.
5. ✅ **No taxonomy regressions.** On-disk sweep of
   `~/.jarvis/autonomy_audit.jsonl` confirms every `AUTONOMY_L3_PROMOTED`
   entry has `outcome="clean"` and no spurious
   `AUTONOMY_L3_ACTIVATION_DENIED` / `AUTONOMY_ESCALATION_PARKED` /
   `AUTONOMY_ESCALATION_EXPIRED` entries exist.

### Non-goals (explicit)

- No cryptographic signing of attestation records (hash-attested + operator-
  accepted; stronger signing is future work).
- No auto-promotion to L3 under any live-runtime condition.
- No global widening of `ALLOWED_PATHS` via escalation approval.
- No backfill of `current_ok`, `ever_ok`, or `maturity_highwater.json` from
  the attestation ledger.
- No history-surface persistence beyond the JSONL audit ledger.

### Regression guards — READ BEFORE TOUCHING Phase 6.5 code

- Do NOT re-introduce any auto-promotion to L3. The promotion loop emits
  `AUTONOMY_L3_ELIGIBLE` only. Regression test:
  `test_no_auto_promote_to_l3_ever` (100 ticks at 99% win rate cannot
  reach L3).
- Do NOT backfill `current_ok` from attestation, persisted state, or
  `ever_ok`. Regression tests in
  `test_l3_snapshot_caches.py::TestL3EscalationCache::test_current_ok_never_backfilled_from_persisted_file`
  and `test_phase_6_5_smoke.py`.
- Do NOT widen `AUTONOMY_L3_PROMOTED` outcome enum. Taxonomy tests in
  `test_autonomy_audit_ledger.py` and `test_phase_6_5_smoke.py`.
- Do NOT let `AutonomyAuditLedger` disk failures propagate. The ledger
  must swallow write failures and log at WARNING only. Regression test:
  `test_autonomy_audit_ledger.py::test_disk_write_failure_does_not_crash_handler`.
- Do NOT grant global `ALLOWED_PATHS` widening from escalation approval.
  Approval scope is per-request `declared_scope` only.

---

## Synthetic Soak Hardening + Distillation Dependency Fix + `_v1` Skill Retirement (2026-04-18)

A full-session stability pass driven by a live `route_coverage` soak. Three interlocking fixes: harden the perception truth boundary so synthetic audio can never leak into lived-history subsystems, unblock a silent-training Tier-1 specialist (`emotion_depth`), and retire an autonomous learning-job path that kept generating blocked `_v1` meta-skills. Plus a dashboard observability panel and a statistical fix to `skill_honesty` scoring.

### Incident that forced the change

A `route_coverage` soak surfaced two invariant violations — `memory_side_effects > 0` and `identity_side_effects > 0` — even though the synthetic exercise framework was functioning correctly. Root cause: the perception pipeline had **session-sticky** synthetic sourcing, but downstream subsystems (`memory_gate`, `identity_fusion`, `perception/server.py::face_crop`) had no awareness of an active synthetic session — they treated each derived event as if it came from a real user. Separately, post-soak telemetry showed `emotion_depth` had 297 teacher signals, 289 paired samples, and healthy labels, yet had never trained a single epoch. And a stale `speaker_identification_v1` learning job had blocked 10× on a buggy verifier and was re-proposed every cooldown.

### Design contract

- **IS**: (a) a truth-boundary contract between the synthetic exercise harness and every write-path subsystem; (b) a scheduling fix for Tier-1 specialists whose inputs depend on another specialist's training cycle; (c) a surgical retirement of `_v1` auto-learning that already self-improves via the distillation loop.
- **IS NOT**: A rewrite of the perception pipeline, a new event type, a change to any Tier-1 architecture, or a deletion of the base `speaker_identification` / `emotion_detection` skills. User-initiated skill learning via `SkillResolver` is preserved.
- **Why split this way**: each fix targets a distinct layer. The truth boundary belongs to Pillar 5 (operational truth). The dependency-retrain fix belongs to the hemisphere orchestrator's scheduling layer. The `_v1` retirement belongs to the mastery drive's bridge to learning jobs. All three needed to land together because the soak report required all three to be quiet before we could trust the next round.

### Components shipped

| Component | Role | Files |
|-----------|------|-------|
| Synthetic session gate in CueGate | New access class: `synthetic_session_active()` short-circuits observation and consolidation writes. `begin_synthetic_session(source)` / `end_synthetic_session(source)` use `set[str]` of active sensor IDs (race-safe multi-source). Short-circuits added to `engine.remember()`, `memory/core._direct_memory_write()`, and `memory/transactions._execute_op` for add/update | `brain/memory/gate.py`, `brain/consciousness/engine.py`, `brain/memory/core.py`, `brain/memory/transactions.py` |
| Identity fusion synthetic suppression | `_synthetic_active` flag + `set_synthetic_active()`; centralized `_emit_resolved()` helper routes all 6 `IDENTITY_RESOLVED` emission sites through a single suppressor. `_synthetic_suppressed_count` telemetry exposed via `get_status()` | `brain/perception/identity_fusion.py` |
| face_crop handler guard | `perception/server.py`'s `face_crop` branch checks `memory_gate.synthetic_session_active()` and skips both face identification and `PERCEPTION_FACE_IDENTIFIED` emission — closes the last leak path (real camera frames arriving during synthetic audio runs) | `brain/perception/server.py` |
| Orchestrator wiring | `_on_synthetic_exercise_state` now calls `memory_gate.begin/end_synthetic_session` and `identity_fusion.set_synthetic_active` on every state event | `brain/perception_orchestrator.py` |
| Distillation dependency-retrain fix | New constant `DISTILLATION_DEP_RETRAIN_WINDOW_S = 60.0`. `_run_distillation_cycle` snapshots `_tier1_last_retrain_time` once at cycle start (`cycle_start_retrain_time`) and checks dependents against that snapshot — eliminates same-tick self-blocking where `speaker_repr` retrained then `emotion_depth` in the same cycle hit its own dependency guard | `brain/hemisphere/orchestrator.py` |
| `skill_honesty` scoring fix | Phase-aware weighting for blocked jobs + small-N confidence blending in `_score_skill_honesty` — prevents a single blocked job from collapsing the dimension score at low sample counts | `brain/epistemic/soul_integrity/index.py` |
| Stage Ladder panel | New dashboard UI: per-subsystem promotion status overview (Policy NN, World Model, Simulator, Autonomy, Hemispheres, M6 expansion) with shared styling. Read-only snapshot view | `brain/dashboard/static/dashboard.js`, `brain/dashboard/static/style.css` |
| Retire `_v1` auto-learning | Removed `recognition_confidence → speaker_identification_v1` and `emotion_accuracy → emotion_detection_v1` from `_DEFICIT_CAPABILITY_MAP`. Mastery drive still tracks the deficits (`_DEFICIT_ACTIONABILITY`) and still acts via the default `experiment` strategy. `SkillResolver` templates preserved so explicit user requests still produce a guided-collect flow | `brain/autonomy/drives.py` |
| On-boot `_v1` migration | `SkillRegistry._migrate_retired_v1_skills()` purges `speaker_identification_v1` / `emotion_detection_v1` records on load, archives them for audit. Idempotent | `brain/skills/registry.py` |
| Regression tests | 6 `test_memory_gate::TestSyntheticSession` + `TestSyntheticGate` in `test_identity_fusion` + `TestMemoryWriteSuppressedDuringSyntheticSession` + `TestFaceCropSuppressedDuringSyntheticSession` + `TestDependencyRetrainOrdering` (2 cases) + `TestRetiredAutoLearning` (6 cases) + `TestRetiredV1Migration` (5 cases) + 3 new `test_soul_integrity` cases for `skill_honesty` | `brain/tests/test_memory_gate.py`, `test_identity_fusion.py`, `test_synthetic_exercise.py`, `test_hemisphere_gating.py`, `test_drives.py`, `test_job_lifecycle.py`, `test_soul_integrity.py` |

### Hard contracts

- A `SYNTHETIC_EXERCISE_STATE active` event MUST cause ALL subsequent observation/consolidation writes and ALL `IDENTITY_RESOLVED` / `PERCEPTION_FACE_IDENTIFIED` emissions to short-circuit for the matching session_id, regardless of which subsystem produced them.
- A Tier-1 specialist's `depends_on` guard MUST use the retrain-time snapshot taken at the start of its own distillation cycle — not live state — so a dependency retraining inside the same cycle cannot block the dependent indefinitely.
- The mastery drive MUST NOT auto-create `speaker_identification_v1` / `emotion_detection_v1` jobs; the underlying capabilities already self-improve via the Tier-1 distillation loop. Explicit user-initiated skill training MUST still work via `SkillResolver`.
- `skill_honesty` scoring MUST NOT let a single blocked learning job collapse the dimension at low N. Phase-aware weighting + confidence blending handle the small-N regime.

### Verification

- Post-fix `route_coverage` soak: `memory_side_effects = 0`, `identity_side_effects = 0`, all consistency invariants held (`hard_stopped ≤ stt_ok ≤ sent`), transport stable across reconnect.
- Live hemisphere logs after emotion_depth fix: `emotion_depth` built at 96.5% accuracy for the first time post-boot-stabilization, then retrained 7 times reaching 0.994; `speaker_repr` climbed 0.970 → 0.986; `dream_synthesis` steady at 1.000; `voice_intent` correctly placed into 900s regression cooldown after 3 regressions against synthetic `tool_router` signals (expected — synthetic signals are low-fidelity teacher for routing).
- `speaker_identification_v1` no longer appears in `skill_registry.json` after boot migration; legacy archived job file preserved for audit.
- Test suite: all new regressions green; pre-existing unrelated failures in `test_dogfood_campaign.py` and `test_supervisor.py` confirmed via stash-and-rerun and deselected from the targeted runs.

### Known open item (documented, not fixed)

`skills/baseline.py::capture_speaker_id_metrics` still reads from hemisphere field `migration_readiness` (only set during substrate migration, not as a general accuracy proxy) and teacher key `total_signals` (actual key is `total`). This is the reason the original `_v1` verify phase always reported zero improvement. The autonomous re-creation path is now closed, so this is no longer a recurring blocker — but if a learning job for one of these skills is ever created via another path, it will hit the same bug. An inline comment block in `autonomy/drives.py::_DEFICIT_CAPABILITY_MAP` documents this for future maintainers. Fix the reads before re-adding any deficit → learning-job mapping.

---

## Intention Infrastructure — Stage 0: Truth Layer, No Delivery (2026-04-17)

New epistemic **Layer 12** of the immune stack. Minimal truth-layer infrastructure so Jarvis can honestly record what it said it would do, link each commitment to real background work, and refuse to make unbacked commitments. **Zero proactive delivery.**

### Incident that forced the change

Log at `20:31:20` — Jarvis said *"give me a moment as I process and organize"* then never followed up. No subsystem was tracking the promise. No ledger entry existed. Root cause was **missing infrastructure, not missing judgment**: there was simply no place for an "I'll get back to you" commitment to live.

### Design contract

- **IS**: symbolic-truth infrastructure. Extends Pillar 3 (tri-layer separation), Pillar 4 (epistemic immune), Pillar 5 (operational truth boundary), Pillar 9 (observability), Pillar 10 (restart resilience).
- **IS NOT**: a delivery mechanism. No new proactive edge, no new TTS path, no "remind user" logic, no `ProactiveGovernor` consumer. Those belong to Stage 1+ and require real ground-truth pairs that do not exist yet.
- **Why split this way**: the registry is infrastructure that cannot self-emerge (like `attribution_ledger.jsonl` or `beliefs.jsonl`), but the **judgment** ON TOP of it absolutely can be learned via the existing Tier-1 distillation + `claim_classifier` + FractalRecall architecture once ~50 commitment/outcome pairs exist. Building delivery first would invert the NN Maturity Flywheel.

### Components shipped

| Component | Role | Files |
|-----------|------|-------|
| CommitmentExtractor | Regex bootstrap, ~15 patterns across 4 classes (`follow_up`, `deferred_action`, `future_work`, `task_started`) + `CONVERSATIONAL_SAFE_PATTERNS` filter for benign reflections | `brain/cognition/commitment_extractor.py` (new) |
| IntentionRegistry | Durable CRUD (`register`/`resolve`/`abandon`/`stale_sweep`), atomic persist, singleton, thread-safe. Outcomes: `open \| resolved \| failed \| stale \| abandoned` | `brain/cognition/intention_registry.py` (new) |
| Backed-commitment gate | `CapabilityGate.evaluate_commitment(text, backing_job_ids)` — route-class policy, not verb whitelist. Unbacked commitments rewritten to "I don't have a background task to follow up on that right now." | `brain/skills/capability_gate.py` |
| Conversation handler wiring | Turn-scoped `_backing_job_ids` + `_intention_registered_turn`; outgoing pipeline: dispatch → `_gate_text` → `evaluate_commitment` → `_send_sentence` | `brain/conversation_handler.py` |
| LIBRARY_INGEST outcome | Thread returns `backing_job_id` synchronously, resolves intention on completion (existing `proactive_message` broadcast unchanged) | `brain/conversation_handler.py` |
| Autonomy outcome | `AUTONOMY_RESEARCH_COMPLETED` / `AUTONOMY_RESEARCH_FAILED` call `intention_registry.resolve()` — truth-only | `brain/autonomy/orchestrator.py` |
| Stale sweep cycle | New 300s cycle `intention_stale_sweep`, mode-gated via `allowed_cycles`, marks >7d open intentions as `stale` | `brain/consciousness/consciousness_system.py`, `modes.py` |
| Self-status surfacing | `MeaningFrame(self_status)` gains deterministic facts when `open_count > 0`: "Open intentions: N" + "Most recent open intention age: ..." | `brain/reasoning/bounded_response.py` |
| Persistence boot/shutdown | `load_intention_registry()` on boot, `save_intention_registry()` on shutdown + in auto-save loop | `brain/memory/persistence.py`, `brain/main.py` |
| Observability | `/api/intentions` endpoint, dashboard panel with stat cards + 7-day outcome histogram + open/resolved lists | `brain/dashboard/app.py`, `snapshot.py`, `static/dashboard.js`, `static/api.html` |
| PVL group `intention_truth` | 3 contracts: registry loaded on boot, error counters bounded, no chronic stale backlog >7d | `brain/jarvis_eval/process_contracts.py`, `collector.py` |
| Regression tests | 17 commitment extractor + 9 registry + 6 new capability gate commitment regressions (incl. 20:31:20) | `brain/tests/test_commitment_extractor.py`, `test_intention_registry.py`, `test_capability_gate.py` |

### Persistence files

- `~/.jarvis/intention_registry.json` — current open + recently resolved intentions (atomic write, schema v1)
- `~/.jarvis/intention_outcomes.jsonl` — append-only outcome ledger feeding future training signal

### Hard contracts

- Commitments WITHOUT a backing job id MUST be rewritten — regardless of route.
- Commitments WITH a backing job id MUST pass unchanged.
- Conversational reflections (`think about`, `keep in mind`, `remember`) MUST NOT be rewritten.
- Autonomy research outcomes MUST be recorded in the registry; they MUST NOT trigger user-facing delivery at this stage.

### Historical graduation gates to Stage 1

These gates were recorded for the Stage 0 → Stage 1 decision. Stage 1 now
exists as a shadow/gated resolver; the same lived-evidence requirements still
control promotion beyond `shadow_only`.

- ≥30 backed commitments logged across real lived interactions.
- Stale sweep correctness verified (no premature stale marks on active jobs).
- `test_intention_registry`, `test_commitment_extractor`, and `test_capability_gate` commitment regressions green for 2 consecutive weeks.
- Eval sidecar `intention_truth` group at 100% contract coverage.
- Zero confabulated commitment phrases in `conversation_history.json`.
- Resolution-outcome histogram shows real variance (not all "stale", not all "resolved").

### Planned Stage 2 specialist (not built)

A new `intention` label class will be added to the existing `claim_classifier` Tier-1 specialist (28→8 approximator, KL-divergence, shadow-only). Teacher signal: outcome + next-turn user signal. Same dual-encoder A/B promotion gate (>55% win rate) that governs all other Tier-1 specialists.

### Known prerequisite (documented, not bundled)

`tool_router.py` picks one `ToolType` per utterance. Multi-intent utterances (e.g. ACADEMIC_SEARCH + LIBRARY_INGEST in one turn) only register ONE intention in Stage 0. Called out explicitly in the PVL contracts so the gap is observable. A separate plan should scope `tool_router.route_many()` with ordered-dispatch semantics.

---

## Phase 9: Dream Artifact Assessor — Tier-1 Validator Shadow (2026-04-16)

11th Tier-1 distilled specialist. Learns to shadow the ReflectiveValidator's artifact disposition from durable dream-cycle evidence. Shadow-only — does not write memory, promote artifacts, or bypass validator authority.

| Component | Description | Files |
|-----------|-------------|-------|
| DreamArtifactEncoder | 16-dim feature vector (3 blocks: artifact intrinsic [8], system state [5], governance pressure [3]) + 4-class label + 8 structured reason categories | `brain/hemisphere/dream_artifact_encoder.py` (new) |
| Persistence bridge | Teacher signal recording in `ReflectiveValidator._evaluate()` via DistillationCollector, paired by `artifact_id` | `brain/consciousness/dream_artifacts.py` |
| System context | `_gather_dream_validation_context()` collects memory density, beliefs, governance pressure at validation time | `brain/consciousness/consciousness_system.py` |
| Hemisphere registration | `DREAM_SYNTHESIS` focus + DistillationConfig (16→4, kl_div, permanent) + `_TIER1_FOCUSES` | `brain/hemisphere/types.py`, `orchestrator.py` |
| Tensor prep | `_prepare_dream_observer_tensors()` with `artifact_id` metadata pairing | `brain/hemisphere/data_feed.py` |
| Architect | Output size 4, softmax activation | `brain/hemisphere/architect.py` |
| Dashboard | Added to `_SI_SPECIALIST_FOCUSES` | `brain/dashboard/snapshot.py` |

**Anti-authority contract (8 structural tests):** AST import isolation (no path to dream_artifacts/memory/events), no mutation methods on encoder, validator source has zero NN references, distillation signal is record-only. 45 tests total (42 pass, 3 skip for no-torch env).

**Behavioral contract:** The NN suggests. The ReflectiveValidator decides. The NN never bypasses the validator, writes to memory, creates beliefs, or promotes artifacts without validator approval. These boundaries are tested structurally, not just behaviorally.

---

## Action Confabulation Fix + Synthetic Claim Exercise (2026-04-15)

**P0 critical fix from full system audit (audit_16). Jarvis fabricated creating a plugin and setting a timer on three consecutive live turns. CapabilityGate's 13 claim patterns only scanned forward-looking claims; past-tense action confabulations passed undetected on the NONE route.**

### Incident (audit finding F1)

User asked: "create a plugin" and "remind me in 5 minutes"
Jarvis responded: "I've created the reminder plugin" and "I have set a timer for five minutes"
Reality: No plugin was created, no timer exists. The LLM fabricated actions.

### Three-Layer Surgical Fix

| Layer | What | Files |
|-------|------|-------|
| 1. Action confabulation detection | 2 new `_CLAIM_PATTERNS` (past-tense + progressive), 2 new `_SYSTEM_ACTION_NARRATION_RE` patterns | `brain/skills/capability_gate.py` |
| 2. Expanded domain vocabulary | `_BLOCKED_CAPABILITY_VERBS` + timer/alarm/reminder/plugin; `_INTERNAL_OPS_RE` + plugin/tool/extension | `brain/skills/capability_gate.py` |
| 3. Deterministic creation-request catch | `_check_capability_creation_request()` pre-LLM intercept for "create a plugin/tool", "set a timer" → canned guidance | `brain/conversation_handler.py` |

### Synthetic Claim Exercise

New training harness for CLAIM_CLASSIFIER specialist: `brain/synthetic/claim_exercise.py`
- 12 categories, ~130 templates covering all 8 gate action classes
- Categories include `confabulation` and `system_narration` (new)
- Feeds CapabilityGate.check_text() → records teacher signals
- Profiles: smoke (20 claims), coverage (100), strict (200 INTROSPECTION route), stress (500)

### Encoder Fix

`brain/skills/claim_encoder.py`: normalization denominator updated 13→15 for `claim_pattern_index`

### Regression Tests

46 tests in `brain/tests/test_claim_exercise.py`:
- Exact incident reproduction (plugin lie, timer lie)
- Past-tense and progressive pattern coverage
- Expanded vocabulary validation
- Deterministic creation-request catch (6 positive + negative cases)
- Claim pattern expansion verification
- Conversational pass-through confirmation
- Synthetic corpus structure validation
- Exercise runner (distillation signal generation, confab blocking, conversational passing)

### Files Changed

| File | Changes |
|------|---------|
| `brain/skills/capability_gate.py` | 2 new `_CLAIM_PATTERNS`, 2 new `_SYSTEM_ACTION_NARRATION_RE`, expanded `_BLOCKED_CAPABILITY_VERBS`, expanded `_INTERNAL_OPS_DOMAINS` + `_INTERNAL_OPS_NOUNS` |
| `brain/conversation_handler.py` | `_CAPABILITY_CREATION_RE`, `_TIMER_CREATION_RE`, `_check_capability_creation_request()`, integration into NONE route |
| `brain/synthetic/claim_exercise.py` | **New**: synthetic claim exercise harness |
| `brain/tests/test_claim_exercise.py` | **New**: 46 regression tests |
| `brain/skills/claim_encoder.py` | Normalization `/13.0` → `/15.0` |
| `brain/dashboard/static/history.html` | Build history entry |
| `brain/dashboard/static/science.html` | CLAIM_CLASSIFIER synthetic exercise docs |
| `brain/dashboard/static/docs.html` | L0 description updated (15 patterns + confab guard) |

### ⚠ Regression Guard

Do NOT make the NONE route more permissive, remove system-object nouns from blocked verbs, or bypass the pre-LLM creation catch. These were added to stop Jarvis from lying about creating plugins and setting timers. If a legitimate expression is incorrectly blocked, log it as friction evidence for the CLAIM_CLASSIFIER specialist to learn from. See `docs/ARCHITECTURE_PILLARS.md` bridge #8 and `TODO.md` REGRESSION GUARD section.

---

## Second Plugin Dogfood: Dice Roller (2026-04-15)

**Exercised the full capability acquisition pipeline with a second plugin. Exposed and fixed two bugs: (1) CoderServer ignoring JSON-only instruction, (2) plugin quarantine blocking stdlib imports. Plugin successfully built, quarantined, verified, and awaiting activation.**

### Pipeline Exercise

| Lane | Result | Issues Found |
|------|--------|--------------|
| evidence_grounding | Completed | None |
| doc_resolution | Completed | None |
| planning | Completed | None |
| plan_review | Approved (risk_tier 2) | None |
| implementation | Failed → Completed (retry 2) | CoderServer returns raw code blocks instead of JSON |
| plugin_quarantine | Failed → Completed (retry 2) | `random` not in import allowlist |
| verification | Completed | None |
| plugin_activation | Deferred (dreaming mode) | Working as designed — high-risk lane mode-gated |

### Fixes Shipped

1. **JSON extraction fallback** (`codegen/service.py`): When the LLM returns Python code in markdown blocks instead of the required JSON format, the parser now reconstructs the JSON structure by extracting labelled code blocks. Handles `# handler.py` comments and `## handler.py` headings, and infers filenames from content patterns (`def run(` → `handler.py`, `def handle(` or `PLUGIN_MANIFEST` → `__init__.py`).

2. **Expanded stdlib import allowlist** (`tools/plugin_registry.py`): Added `random`, `string`, `textwrap`, `copy`, `operator`, `decimal`, `fractions`, `statistics`, `uuid` to `ALWAYS_ALLOWED_IMPORTS`. These are safe stdlib modules that generated plugins commonly need.

### Files Changed

| File | Changes |
|------|---------|
| `brain/codegen/service.py` | `_extract_json()` fallback: reconstruct JSON from markdown code blocks |
| `brain/tools/plugin_registry.py` | Expanded `ALWAYS_ALLOWED_IMPORTS` with 8 safe stdlib modules |

---

## Stage 2 First Approved-Patch Campaign (2026-04-15)

**First real Stage 2 approved patch: operator-triggered, sandbox-validated, human-approved, applied to live system, health check passed, no rollback. End-to-end live path proven.**

### Campaign Exercise

The goal was to deliberately exercise the full Stage 2 live path — not just dry-run validation, but a real applied patch surviving health monitoring.

| Step | Result |
|------|--------|
| Stage promoted to 2 (human-approval) | `.env` updated + runtime API `set_stage(2)` |
| Operator triggered improvement (non-manual, non-dry-run) | Entered approval queue via `stage2_auto_triggered` gate |
| CoderServer generated patch | 1 iteration, `coder_local` provider, ~350s generation |
| Sandbox validation | ALL PASSED: lint=True, tests=True, sim=True, recommendation=promote |
| Operator reviewed unified diff on dashboard | Clean 4-hunk change in single file |
| Operator approved (`approved: true`) | `approve()` executed |
| Patch applied to disk | `applied: true` |
| Post-apply health check | `health_check_passed: true` |
| Rollback check | `rolled_back: false` — no regression detected |
| System stability | 0 rollbacks, tick p95 stable, no errors |

### The Patch

**Target**: `brain/consciousness/consciousness_analytics.py`

**Description**: Replace `list(deque)[:n]` slicing with direct deque indexing to eliminate intermediate list allocations in rolling window computations.

**Changes** (4 hunks):
1. `sum(list(w)[:half])` → `sum(w[i] for i in range(half))` — generator-based half-window summation
2. `abs(list(w)[i] - list(w)[i-1])` → `abs(w[i] - w[i-1])` — direct deque indexing for volatility diffs
3. `depth_vals = list(self._depth_window)` → `depth_vals = self._depth_window` — eliminated list copy for epistemic refresh
4. `list(self._tone_history)` / `list(self._phase_history)` → direct deque references — eliminated list copies for coherence computation

**Risk assessment**: Low blast-radius (single file, no API changes, no new imports, O(1) memory savings per tick).

### Improvements Found During Campaign

| Finding | Status |
|---------|--------|
| Pending approval diff was empty (search/replace edits don't populate `FileDiff.diff`) | Fixed: `get_pending_approvals()` now generates unified diff from `original_content`/`new_content` via `difflib` |
| Approval endpoint requires `"approved": true` explicitly (defaults to reject) | Documented: API contract is `{"patch_id": "...", "approved": true}` |
| `MIN_INTERVAL_S = 600` cooldown blocks rapid re-trigger after rejection | Working as designed: prevents spam |
| Pending approvals are in-memory only — lost on restart | Known limitation: documented for future persistence work |
| `/api/self-improve/trigger` endpoint added | New endpoint for operator-initiated non-manual runs that enter approval queue |

### Validation Results (same session)

Validation pack run immediately before campaign:
- **30/33 checks passing** (current), **31/33 ever**
- PVL Coverage: 89.4% / 93.3%
- Maturity Active: 88.6% / 94.3%
- 2 FAILs: Phase 5 weakness signal (expected — no `avoid_patterns` this session)
- Artifact: `validation-pack-20260415T134139Z.json`

### Specialist Maturity Review (same session)

| Specialist | Maturity | Features | Labels | Enriched | Last Signal |
|-----------|----------|----------|--------|----------|-------------|
| DIAGNOSTIC (43-dim) | early_noisy | 23 (21 @ 43-dim) | 10 (6 classes) | codebase + friction: 21/21 | 2026-04-12 |
| CODE_QUALITY (35-dim) | early_noisy | 24 (all 35-dim) | 8 (4 classes) | patch history: 24/24 | 2026-04-15 |
| plan_evaluator | bootstrap | 7 | 6 | N/A | 2026-04-12 |
| system_upgrades | idle | 0 | 0 | N/A | never |

**Key findings**:
- Feature enrichment working: codebase structural dims (32-37) and friction dims (38-42) are non-zero in all 43-dim diagnostic entries
- Patch history dims (28-34) are non-zero in all 35-dim code quality entries
- No training has happened yet (0 attempts) — still accumulating in bootstrap/early_noisy phase
- Negative examples missing for DIAGNOSTIC (healthy-scan labels need scanner maturity)
- CODE_QUALITY got 3 new signals from today's dry-runs

### Files Changed

| File | Changes |
|------|---------|
| `brain/dashboard/app.py` | Added `/api/self-improve/trigger` endpoint |
| `brain/self_improve/orchestrator.py` | `get_pending_approvals()` now generates unified diff via `difflib` when `FileDiff.diff` is empty |
| `brain/consciousness/consciousness_analytics.py` | **AI-applied patch**: deque indexing optimization (4 hunks) |

---

## Stage 2 Self-Improvement Readiness + Activity Log (2026-04-15)

**Made Stage 2 (human-approval) structurally safe and operationally reviewable. Verified CoderServer works end-to-end. Identified codegen context quality as the next bottleneck.**

### Safety Infrastructure (6 fixes)

| Fix | What | Why |
|-----|------|-----|
| Approval gate | Auto-triggered patches forced through human approval at Stage 2 (`not request.manual`) | Previously only "dangerous" patches were gated |
| `approve()` safety | Now async, runs `_check_post_apply_health`, auto-rollback on regression, structured return | Previously fire-and-forget with no health check |
| Pending details | `get_pending_approvals()` returns diffs, sandbox_summary, `why_requires_approval` list | Reviewer had no context to make informed decisions |
| Dashboard UI | Pending approval panel (diffs, approve/reject buttons), stage control widget, API key auth fixed | No operator-facing surface existed |
| Stage persistence | `~/.jarvis/self_improve_stage.json` with atomic write, `stage_source` tracking | Stage reset to default on every restart |
| `ImprovementRequest.manual` | Boolean field tracks whether request was operator-triggered or auto-detected | Approval gate needs to distinguish trigger source |

### Scope Fixes (found during live verification)

| Fix | What |
|-----|------|
| `_create_plan()` path normalization | `target_module` now gets `brain/` prefix and trailing `/` for `files_to_modify` |
| `_infer_write_category()` normalization | Trailing `/` added for consistent category mapping against `WRITE_BOUNDARIES` |

### Activity Log (dashboard observability)

- 200-entry ring buffer in `SelfImprovementOrchestrator._activity_log`
- 22 structured `_log()` calls at key pipeline points (start, plan, codegen, sandbox, approval, apply, rollback, promote, stage change)
- Dashboard panel with color-coded severity (info/warn/error/success/thinking), phase tags, detail expansion
- Exposed via `get_status()["activity_log"]` and `get_activity_log(limit)` API

### CoderServer End-to-End Verification

Live dry-run on remote Jarvis confirmed:
- `llama-server` starts successfully (177.5s cold start, CPU mode)
- Qwen3-Coder-Next generates code (107.4s, 568 tokens)
- Pipeline parses and validates the output
- Sandbox path exists and would execute
- **Failure was codegen context quality**: model hallucinated `analytics.py` instead of `consciousness_analytics.py` — fixed in same session (see below)

### Codegen Context Quality Fix (same session)

Root cause: `_get_code_context()` dumped full source of every .py file when `files_to_modify` was a directory. For `brain/consciousness/` (31 files), this produced 162K tokens, exceeding the 16K context window. On Ollama fallback, the model had no directory inventory and hallucinated filenames.

**Changes (6 fixes across 3 files):**

| Fix | File | What |
|-----|------|------|
| Budget-aware context | `orchestrator.py` | `_get_code_context()` caps at 40K chars, ranks files by keyword relevance, includes full source only for top matches |
| Directory inventory | `orchestrator.py` | Directory targets produce a file listing with "do NOT invent filenames" constraint |
| Relevance ranking | `orchestrator.py` | `_rank_files_by_relevance()` scores files by description keyword overlap, deprioritizes `__init__`, penalizes >30KB files |
| Resolved file list | `orchestrator.py` | `_format_plan_for_llm()` surfaces exact paths with CRITICAL constraint |
| FILENAME RULE | `conversation.py` | `CODER_SYSTEM_PROMPT` now requires paths match inventory exactly |
| Hallucination detection | `provider.py` | `_parse_response()` logs "HALLUCINATED FILENAME" for non-existent paths |

**Verification (same target):**

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Context size | 162,337 tokens (overflow) | Fits in 16K |
| Target file | `analytics.py` (hallucinated) | `consciousness_analytics.py` (correct) |
| Iterations | 2 (both failed) | 1 (first try) |
| Sandbox | Never reached | ALL PASSED (lint, tests, simulation) |
| Recommendation | Failed | Promote |

### Tests

12 new tests across `test_self_improve_orchestrator.py` and `test_self_improve_sprint1.py`:
- `TestStage2ApprovalGate` (4 tests): gate condition, pending queue, manual bypass, stage 1 dry-run
- `TestApproveSafety` (2 tests): health check, rollback on regression
- `TestPendingDetails` (3 tests): `why_requires_approval`, diffs present, sandbox summary
- `TestStagePersistence` (3 tests): disk persist, source tracking, status output

All 70 orchestrator + sprint1 tests passing.

### Files Changed

| File | Changes |
|------|---------|
| `brain/self_improve/orchestrator.py` | Approval gate, approve() safety, pending details, stage persistence, activity log, scope normalization |
| `brain/self_improve/improvement_request.py` | Added `manual: bool = False` field |
| `brain/dashboard/app.py` | `approve()` now awaited, structured return handling |
| `brain/dashboard/static/self_improve.html` | Stage control, pending approvals panel, activity log panel, API auth fix |
| `brain/tests/test_self_improve_orchestrator.py` | 12 new tests across 4 test classes |
| `brain/tests/test_self_improve_sprint1.py` | Updated for `(stage, source)` tuple return |

---

## Plugin Entry Point Fix + Tiered Isolation Design (2026-04-15)

**Fixed the generation contract mismatch between codegen and plugin runtime. Designed tiered plugin isolation architecture for future complex plugins.**

### Entry Point Fix (5 changes, 3 regression tests)

The codegen system prompt asked the LLM to produce a single `handler.py` with `run()`, but the plugin runtime expected `handle(text, context)` in `__init__.py`. When the quarantine fallback generated `__init__.py`, it only wrote `PLUGIN_MANIFEST` — no callable entry point. Additionally, `importlib` is in `NEVER_ALLOWED_IMPORTS`, so the manual `importlib.util.spec_from_file_location()` fix used in the initial unit converter would fail quarantine validation on any future plugin.

**Changes:**

| Fix | File | What |
|-----|------|------|
| Import validator | `brain/tools/plugin_registry.py` | `_check_imports()` now skips relative imports (`node.level > 0`). `from .handler import run` is intra-package and always allowed. |
| Quarantine fallback | `brain/tools/plugin_registry.py` | When `__init__.py` is absent from `code_files`, generates one with `PLUGIN_MANIFEST` + `async def handle()` bridge using `from .handler import run`. Defensive: catches import and execution errors. |
| Codegen prompt | `brain/acquisition/orchestrator.py` | System prompt now requests two files: `handler.py` (implementation with `run(args)`) and `__init__.py` (entry point with `handle(text, context)` returning `{"output": ...}`). States `PLUGIN_MANIFEST` should be minimal. |
| Bundle synthesis | `brain/acquisition/orchestrator.py` | `_build_code_bundle()` synthesizes `__init__.py` with the same bridge when LLM only produces `handler.py`. Intermediate hardening — models sometimes ignore multi-file instructions. |
| Module loading | `brain/tools/plugin_registry.py` | `_try_load_handler()` now passes `submodule_search_locations` and registers the module in `sys.modules` under `_jarvis_plugins.<name>`. This enables deferred relative imports (`from .handler import run`) to resolve at invocation time, not just load time. |

**Regression tests (3 new, 76/76 passing):**
1. `test_relative_import_passes_validation` — `from .handler import run` not flagged by `_check_imports()`
2. `test_quarantine_fallback_generates_handle_bridge` — quarantine with only `handler.py` produces `__init__.py` with callable `handle()`
3. `test_handler_only_bundle_becomes_loadable` — full integration: quarantine → activate → invoke succeeds with correct output

### Tiered Plugin Isolation Design (future phase, captured in roadmap)

Designed a tiered execution model for plugins that need external dependencies (e.g., 3D printer controller needing `trimesh`, `numpy-stl`):

- **`in_process`** (current, default): stdlib-only plugins loaded via `importlib`, executed in brain's event loop
- **`isolated_subprocess`** (future): own venv + child process, JSON-over-stdin/stdout IPC, CoderServer-pattern lifecycle

**Key architectural decisions documented:**
- Execution mode is immutable per version (changing requires new version through full promotion path)
- `setup_commands` explicitly forbidden in v1 (no arbitrary shell commands)
- Dependencies must be pinned with lockfile + install transcript
- `invocation_schema_version` field for protocol migration
- Subprocess child runs with locked cwd, stripped env vars, isolated sys.path
- Subprocess isolation labeled "process-isolated" not "sandboxed" until filesystem/env/network boundaries enforced
- New `environment_setup` acquisition lane for venv creation + dependency installation
- `ClarificationRequest` artifact type for complex plugins needing human input before planning

Design captured in: `.cursor/plans/tiered_plugin_isolation_design_f6def610.plan.md` and `docs/MASTER_ROADMAP.md` (Phase 7).

---

## Plugin Routing + Dashboard SI Specialist Signal Fix (2026-04-14b)

**Three usability/observability bugs fixed: plugin intent patterns, plugin naming, and dashboard SI specialist signal display.**

### Plugin Intent Pattern Generation (routing fix)

Plugins deployed by the acquisition pipeline had empty `intent_patterns: []`, making them unreachable via voice. The `PluginManifest` was constructed in two places (`_build_code_bundle` and `_run_plugin_quarantine`) without ever populating the patterns field.

**Root cause**: No component in the pipeline — planner, codegen, or orchestrator — generated routing patterns. The `PluginManifest.intent_patterns` field existed and was consumed correctly by `PluginRegistry.match()` and `tool_router.py` (Tier 0.75), but upstream never populated it.

**Fix**: Added `_derive_intent_patterns()` to the acquisition orchestrator with a 3-tier priority:
1. LLM-generated `PLUGIN_MANIFEST.intent_patterns` from the codegen `__init__.py` (validated: >=2 patterns, each <200 chars, compilable regex)
2. Plan keywords / `required_capabilities` / cleaned intent and title text
3. Title-based heuristic fallback (word-boundary `\b` regexes from cleaned title)

Wired into both manifest creation paths. Patterns are bounded, low-regex-risk, and readable.

### Plugin Name Derivation (naming fix)

Plugin names were derived by `job.title.lower().replace(" ", "_")[:40]` — producing garbage like `"build_a_unit_conversion_tool_that_conver"`.

**Fix**: Added `_derive_plugin_name()` that strips action verbs ("build", "create", "make"...), strips suffixes ("tool", "plugin"...), normalizes to a clean slug, and limits to 30 chars at a word boundary. Example: `"Build a unit conversion tool"` → `"unit_conversion"`.

### Dashboard SI Specialist Signal Display (observability fix)

The `_build_si_specialists()` function in `dashboard/snapshot.py` looked up distillation stats by hemisphere focus name (e.g., `teachers.get("diagnostic", {})`), but the distillation collector keys are the teacher/feature_source names from `DISTILLATION_CONFIGS` (e.g., `"diagnostic_features"`, `"diagnostic_detector"`). This caused all four SI specialists to show 0 signals despite real data existing.

**Fix**: Imported `DISTILLATION_CONFIGS` and used `config.feature_source` / `config.teacher` to look up the correct keys. Now exposes split counts:
- `signals_features` — feature vector count
- `signals_labels` — label count
- `signals_total` — combined

The feature/label split is important runtime information: imbalance between features and labels is itself diagnostic (the April 13 negative-example fix is exactly this scenario). Dashboard now shows `(F:23 L:10)` format.

**Verified live**: diagnostic=33 (F:23, L:10), code_quality=29 (F:21, L:8), plan_evaluator=13 (F:7, L:6). Previously all showed 0.

### Existing Plugin Retrofit

Renamed the deployed unit converter plugin from `build_a_unit_conversion_tool_that_conver` to `unit_converter`:
- Directory renamed on remote filesystem
- `__init__.py` updated with new name + 4 intent patterns
- `plugin_registry.json` updated with new name + patterns
- Acquisition job `plugin_id` updated

Plugin routing verified: "convert 5 miles to kilometers", "how many pounds in 10 kilograms", "100 fahrenheit to celsius", and "unit convert meters to feet" all match. Non-conversion phrases correctly produce no match.

### Handler Loading Fix + First Live Plugin Invocation

The initial voice test ("Jarvis, convert five miles to kilometers") routed correctly to `PLUGIN` in 6ms but failed with "Plugin handler not loaded." Root cause: `PluginRegistry._try_load_handler()` looks for a `handle` function in the plugin's `__init__.py`, but the codegen-produced `__init__.py` only contained the `PLUGIN_MANIFEST` dict — no callable.

**Fix**: Added `async def handle(text, context)` to the plugin's `__init__.py` that delegates to `handler.run()` and returns `{"output": result}` (the key expected by `conversation_handler.py` line 4669).

**First successful live plugin invocation** (2026-04-14):

| Step | Component | Time |
|------|-----------|------|
| Wake word | openWakeWord (CPU) | — |
| STT | faster-whisper (GPU) | 620ms |
| Route | tool_router regex match → PLUGIN | 6ms |
| Plugin exec | unit_converter.handle() → handler.run() | <1ms |
| CapabilityGate | check_text() scan | <1ms |
| TTS | Kokoro (GPU) | 190ms |
| **Total** | **conversation_total** | **632ms** |

**Zero LLM involvement.** No Ollama call, no GPU reasoning inference. The plugin route is deterministic: regex match → Python execution → string output → TTS. This is the architecture working as designed — plugins are tool execution, not LLM narration.

User said: "Convert 5 miles to kilometers." Jarvis replied: "5.0 miles = 8.04672 kilometers." Correct answer, sub-second total latency.

Files modified: `brain/acquisition/orchestrator.py`, `brain/dashboard/snapshot.py`, `brain/dashboard/static/self_improve.html`, `brain/tools/plugins/unit_converter/__init__.py`

---

## First End-to-End Plugin Acquisition + Pipeline Bug Fixes (2026-04-14)

**First successful end-to-end run of the Capability Acquisition Pipeline.** The unit converter plugin completed all 10 lanes — from intent classification through code generation, sandbox validation, shadow observation, and deployment — producing Jarvis's first autonomously-generated active plugin.

### Plugin Pipeline Proven

Triggered via dashboard API: "build a unit conversion tool that converts between common units like miles to kilometers, Fahrenheit to Celsius, and pounds to kilograms."

| Lane | Result |
|------|--------|
| evidence_grounding | Completed — SkillRegistry + CodebaseIndex query |
| doc_resolution | Completed — no external docs needed (pure math) |
| planning | Completed — CoderServer (Qwen3-Coder-Next 80B MoE, CPU) generated 6140-char technical design in ~290s |
| plan_review | Completed — approved via dashboard API |
| implementation | Completed — CoderServer generated 1028 tokens of Python (handler.py: 3920 chars) |
| plugin_quarantine | Completed — handler + manifest deployed to `brain/tools/plugins/` |
| verification | Completed — sandbox validation passed (6 pre-existing baseline failures, 0 new) |
| plugin_activation | Completed — quarantined → shadow (1hr observation) → supervised → active |
| deployment | Completed — approved |
| truth | Completed — memory + attribution ledger recording |

Plugin `build_a_unit_conversion_tool_that_conver` is now **active** in the plugin registry (state=active, risk_tier=2, v1.0.0).

### Bugs Found and Fixed (5 cascading pipeline issues)

These bugs had prevented any prior acquisition from completing end-to-end:

1. **CoderServer timeout** (`acquisition/orchestrator.py`): ThreadPoolExecutor timeout was 60s for planning, 120s for implementation. The 80B CPU model needs ~290s for a plan. Increased to 300s/600s. Also: `str(TimeoutError())` is empty — added `type(exc).__name__` fallback for error messages.

2. **JSON format mismatch** (`acquisition/orchestrator.py`): Orchestrator system prompt told the LLM to output `"diffs"` key, but CodeGenService expected `"files"`. Aligned to `"files"`.

3. **LLM output wrapping** (`codegen/service.py`): CoderServer wraps output in `<think>...</think>` tags and markdown code fences. Added stripping of both before JSON extraction.

4. **Attribute name mismatches** (`codegen/service.py`, `acquisition/orchestrator.py`): `CodePatch` uses `.files` not `.diffs`; `EvaluationReport` uses `.overall_passed` not `.all_passed`. Fixed all references.

5. **Plugin path scope validation** (`codegen/service.py`): PatchPlan safety validation rejected plugin files with simple names like `handler.py`. Added conditional prefix with `brain/tools/plugins/_gen/` during validation for `skill_plugin` write category.

### Approval Verdict Bug

`approve_plan()` in the orchestrator only accepted verdicts `"approved_as_is"` and `"approved_with_edits"`. The dashboard sent `"approved"`, which fell through to the else branch — set status to "executing" but never set `review_status = "reviewed"`, permanently blocking plugin activation. Fixed to accept `"approved"` as an alias.

### Sync Safety Fix

`sync-desktop.sh` used `rsync --delete` which nuked runtime-generated plugin handler files on every sync (they exist on the remote but not in the local repo). Removed `--delete` from both brain/ and pi/ rsync commands. Syncing now only adds/updates files.

### Dashboard UX: Approval Feedback

Added visual confirmation feedback to the self-improvement dashboard for plan approval and plugin actions. Previously, clicking "Approve Plan" or plugin action buttons produced no visible feedback — you had to check the console or wait for the next auto-refresh to see if anything changed. Now shows toast notifications for success/failure and immediate status updates.

Files modified: `brain/acquisition/orchestrator.py`, `brain/codegen/service.py`, `brain/dashboard/static/self_improve.html`, `sync-desktop.sh`

---

## CapabilityGate Route-Aware Evaluation + SI Specialist Exercise (2026-04-13)

**Architectural fix to CapabilityGate evaluation strategy and two structural fixes for hemisphere distillation.**

### CapabilityGate Route-Aware Default (architectural fix)

The CapabilityGate's `_evaluate_claim()` had a universal DEFAULT BLOCK: any claim not matching the `_CONVERSATIONAL_SAFE_PHRASES` list was rewritten to "I don't have that capability yet." This was correct for strict self-report routes (STATUS, INTROSPECTION) where every claim must be verified, but destroyed natural conversational expressions on the NONE route (general conversation). The LLM generates relational language ("I can respect your space", "I'll keep our focus") that the safe-phrase list could never fully enumerate.

**What was wrong**: A blocklist approach for all routes — the gate blocked everything not proven safe, regardless of route context.

**What was reverted**: An initial verb-hack fix that expanded `_CONVERSATIONAL_SAFE_PHRASES` with more phrases. This violated the No Verb-Hacking Rule from AGENTS.md — you'd chase every new LLM phrasing forever.

**Architectural fix**: Inverted the default for the NONE route. On general conversation, if a claim has already passed through all danger gates (blocked verbs via `_BLOCKED_CAPABILITY_VERBS`, technical signals via `_TECHNICAL_RE`, internal operations via `_INTERNAL_OPS_RE`), it passes as conversational. The strict default-block behavior is preserved for STATUS, INTROSPECTION, and all other routes.

- New `_INTERNAL_OPS_RE` pattern: category-level negative check matching any combination of system domain words (learning, training, skill, distillation, hemisphere...) with operation nouns (job, pipeline, process, protocol...) in either order. Not a verb hack — catches "launch a skill training process" regardless of which verb the LLM uses.
- Route-conversational bypass in `_evaluate_claim()`: fires only when `route_hint == "none"` AND `not is_readiness_frame` AND claim passes all three negative checks.
- Strict routes (STATUS, INTROSPECTION, default) unchanged — still enforce full verification.

Files modified: `brain/skills/capability_gate.py`

### SI Specialist Starvation Fix (DIAGNOSTIC negative examples)

The DIAGNOSTIC specialist only received positive-example labels (when a detector fired). Healthy scans recorded feature vectors (`diagnostic_features`) but no labels (`diagnostic_detector`), so the pairing logic dropped them and the specialist never learned what "normal" looks like.

- `DiagnosticEncoder.encode_no_opportunity_label()`: returns uniform distribution across 6 classes (correct KL-div target for "no specific detector expected").
- Records negative-example labels on every 5th healthy scan (fidelity=0.6, rate-limited to maintain class balance).
- Pairing works via same `scan_id` metadata linking feature + label.
- 4 new tests in `test_diagnostic_encoder.py`.

Files modified: `brain/hemisphere/diagnostic_encoder.py`, `brain/consciousness/consciousness_system.py`, `brain/tests/test_diagnostic_encoder.py`

### Voice Intent Distillation Regression Fix

The voice_intent specialist was stuck in a 100%→87.7%→rollback loop. Initial build memorized 15-200 samples (100% accuracy = overfitting), then every retrain with shifted data hit a more realistic ~87.7%, exceeding the 5% regression delta, triggering rollback forever.

- `DISTILLATION_REGRESSION_MIN_SAMPLES = 50` constant in `brain/hemisphere/orchestrator.py`.
- Regression gate skipped when `n_samples < 50` (cold-start overfit guard).
- Model freely adapts during early training until enough data stabilizes accuracy baseline.

Files modified: `brain/hemisphere/orchestrator.py`

---

## Specialists Dashboard Tab + Pipeline Trace Validation (2026-04-12)

**Wired the self-improvement specialists into the dashboard and validated the full pipeline end-to-end.**

- **New Specialists tab** on `/self-improve` page: 8th tab showing DIAGNOSTIC, CODE_QUALITY, plan_evaluator, and system_upgrades specialist cards with maturity band, signal counts, accuracy, training attempts, network count, failure state, and last signal timestamp. Auto-refreshes every 10s.
- **New API endpoint**: `GET /api/self-improve/specialists` returns specialist data extracted from hemisphere state via `_build_si_specialists()` in `dashboard/snapshot.py`.
- **Overview panel enrichment**: Self-Improvement overview card now shows specialist summary with maturity + signal counts.
- **Maturity reference**: bootstrap (<15 signals), early_noisy (15-49), preliminary (50-99), meaningful (100-249), stable (250+).
- **14-point pipeline trace validated on live Jarvis**: encoder dims (43/35), label encodings, config wiring, teacher name consistency, DistillationCollector signals, data_feed routing, JSONL persistence, stage system, call site counts, recording site verification, live API response — all passed.
- **Confirmed live signal accumulation**: 23 diagnostic_features, 10 diagnostic_detector, 21 code_quality_features, 8 upgrade_verdict signals persisted from prior sessions.

---

## Self-Improvement Intelligence Specialists — Tracks 2-6 (2026-04-12)

**Six-track build adding DIAGNOSTIC and CODE_QUALITY hemisphere specialists, enriching them with codebase structural knowledge, live friction/correction signals, and per-module patch history, maturing the stage system, and fixing dead pipeline paths.**

### Track 1: Dead Path Credibility Fixes

Fixed 6 dishonest or broken paths in the self-improvement and acquisition pipeline that would have poisoned training data for the new specialists:

- **Broken codegen parser**: Extracted local static parser instead of importing `PatchProvider` machinery.
- **Evidence grounding honesty**: Replaced hollow artifacts with structured `SkillRegistry` results + prior acquisition overlap + codebase index queries.
- **Doc resolution honesty**: Stopped fabricating `relevance=0.5` when nothing was found — now returns `source_type="none_found"` with zero scores.
- **Verification lane**: Runs real sandbox when `PluginCodeBundle` exists, degrades honestly otherwise.
- **Fail-fast stub lanes**: Unimplemented lanes now fail explicitly instead of silent 30-minute timeouts.
- **Plugin discovery exec() removal**: Replaced `exec()` with AST + `ast.literal_eval()` for manifest extraction.

### Track 2: DIAGNOSTIC + CODE_QUALITY Hemisphere Specialists

Two new shadow-only Tier-1 distillation specialists:

- **DIAGNOSTIC** (`hemisphere/diagnostic_encoder.py`): 43-dim feature vector from detector snapshots + system context + codebase structure + friction/correction signals → 6-class detector-category label. Teacher: "which detector fired." Data pairing by `scan_id` metadata.
- **CODE_QUALITY** (`hemisphere/code_quality_encoder.py`): 35-dim feature vector from improvement records (request + patch + sandbox + system context + per-module patch history) → 4-class verdict label (improved/stable/regressed/rolled_back). Data pairing by `upgrade_id` metadata.
- **DistillationCollector.instance() bug fix**: Added missing `classmethod` alias that was silently killing all `plan_evaluator` signal recording.
- **Tensor pairing**: `_prepare_diagnostic_tensors()` and `_prepare_code_quality_tensors()` in `data_feed.py` with metadata-based pairing and backward-compat zero-padding.
- Both registered in `DISTILLATION_CONFIGS` and `_TIER1_FOCUSES`.

### Track 3: Stage System Maturation

- **Runtime stage promotion**: `POST /api/self-improve/stage` endpoint (API-key auth) for operators to change stage (0→1→2) without restarting.
- **Autonomy L2 bridge stage-aware**: `_route_to_self_improve()` now queries the orchestrator's stage instead of checking legacy `FREEZE_AUTO_IMPROVE` env var.
- **`set_stage()` method** on `SelfImprovementOrchestrator` with validation and audit logging.

### Track 4: Codebase Self-Knowledge → DIAGNOSTIC (dims 32-37)

Wired `CodebaseIndex` structural signals into the diagnostic feature vector:

- Module line count, import fanout (coupling out), importers count (coupling in), symbol count (complexity), recently modified flag, and `has_codebase_context` availability flag.
- Primary-target aggregation: encodes the highest-priority/longest-sustained opportunity's module rather than blurring across unrelated modules.

### Track 5: Friction/Correction Signals → DIAGNOSTIC (dims 38-42)

Replaced hardcoded zeros with live `FrictionMiner` reads:

- `friction_rate` and `correction_count` now read from live `FrictionMiner` with 3600s window (matching scan cadence).
- 4 new enrichment fields: `friction_severity_high_ratio`, `friction_correction_ratio`, `friction_identity_count`, `correction_auto_accepted`.
- `has_friction_context` availability flag distinguishes "no friction" from "data unavailable."

### Track 6: Patch History → CODE_QUALITY (dims 28-34)

Added per-module patch outcome memory:

- `get_module_patch_history()` on `SelfImprovementOrchestrator`: scans `improvement_proposals.jsonl` newest-first, capped at 500 lines, returns total patches, verdict counts, recency, recidivism (same fingerprint on same module), avg iterations.
- 7 new CODE_QUALITY dims: total patches, success rate, regression rate, recency, recidivism, avg iterations, `has_patch_history` flag.
- Injected into `_persist_proposal()` before encoding.

### Tests

- 30 diagnostic encoder tests (21 existing + 9 new for codebase + friction dims)
- 29 code quality encoder tests (20 existing + 9 new for module history dims)
- All 105 encoder tests pass across diagnostic, code quality, and plan evaluator.

### AGENTS.md

Updated: Tier-1 distillation section (diagnostic 43-dim, code_quality 35-dim), file tree entries, metric-driven scanner context enrichment description, specialist data flow details, backward-compat padding note.

---

## Acquisition Pipeline Operator UX + Model Integrity (2026-04-12)

**Reject-and-revise flow, job management, CodeGen wiring, model download hardening.**

### Reject & Revise Flow

Full feedback loop when an operator rejects an acquisition plan:

- **Backend**: On rejection, both `planning` and `plan_review` lanes reset to `pending`.
  `_build_revision_context()` loads all prior rejection reviews and builds a structured
  prompt section with operator notes, rejection category, and suggested changes.
  The re-planning prompt includes a `## PREVIOUS PLAN (to revise, not start from scratch)`
  section so the coder revises rather than regenerating from scratch. Plan version
  increments on each revision cycle.
- **Dashboard UI**: "Reject Plan" button replaced with "Reject & Revise" which expands
  a feedback form with category dropdown (7 categories: incomplete design, wrong approach,
  missing tests, security concern, scope too large, needs more detail, other) and a
  textarea for detailed feedback. Submission requires non-empty feedback. On re-review
  of a revised plan, prior rejection feedback is displayed in a highlighted box.
- **API**: `POST /api/acquisition/{id}/approve-plan` now forwards `suggested_changes`
  array to the orchestrator alongside `notes` and `reason_category`.

### Job Cancellation

- **`cancel_job()` on AcquisitionOrchestrator**: Removes job from active map + marks
  cancelled on disk. Works from any job state.
- **`POST /api/acquisition/{id}/cancel`**: Dashboard API endpoint (auth-gated).
- **Dashboard "Remove Job" button**: Appears on failed/cancelled/blocked jobs in the
  recent jobs list with confirmation prompt.

### CodeGen Wiring Fix (from prior session, documented here)

- `CodeGenService` moved out of `self_improve` conditional block in `main.py` — now
  globally available. Wired into both `SelfImprovementOrchestrator.PatchProvider`
  and `AcquisitionOrchestrator` via `set_codegen_service()`.
- `_run_implementation` updated for robust async handling (ThreadPoolExecutor fallback
  when called from running event loop).
- Enhanced planning prompt passes full plan details to coder LLM.

### Model Download Hardening

- **`aria2c` integration**: `setup.sh` installs `aria2` package and uses `aria2c` with
  16 parallel connections + resume for large model downloads, falling back to `curl`
  then `huggingface-cli`.
- **SHA256 verification**: Post-download integrity check against hardcoded hashes.
- **SHA256 caching**: `verify_coder_model()` caches hash results in a `.sha256ok`
  sidecar file keyed on `size:mtime:expected_sha`. Subsequent runs skip the hash
  (~2min for 46GB) if the file hasn't changed. Post-download always forces a full
  hash to seed the cache.
- **Startup integrity check**: `coder_server.py` `_check_model_integrity()` validates
  file size on boot. `is_available()` returns false if integrity check fails.

### Dashboard Artifact Detail (from prior session, documented here)

- `/api/acquisition/{id}` endpoint enhanced to embed full artifact contents (plan,
  review, code bundle, verification, doc artifacts, plugin record).
- Pending approvals banner shows comprehensive plan details (approach, risk, deps,
  tests, implementation sketch) before approve/reject buttons.
- Recent jobs show expandable details with lane progress, plan summary, code bundle
  summary (with "View Full Code" popup), verification results, and plugin ID.

---

## Plan Evaluator Shadow NN Specialist (2026-04-11)

**First acquisition-native hemisphere specialist.** Wires a shadow-only `plan_evaluator` Tier-1 specialist into the hemisphere distillation system to capture human review verdicts as supervised training data from day one of the acquisition pipeline.

### Plan Evaluator Specialist

New `brain/acquisition/plan_encoder.py` — feature encoding, verdict labels, and shadow prediction artifacts:

- **`PlanEvaluatorEncoder`**: Encodes `AcquisitionPlan` + `CapabilityAcquisitionJob` metadata into a 32-dimensional feature vector. Four blocks: classification (9 dims: outcome class one-hot, risk tier, confidence), plan structure (7 dims: step count, rollback, capabilities, artifacts, promotion criteria, version), relational quality (8 dims: design section coverage, dependency-to-capability alignment, evidence-to-implementation coverage, test-to-impl ratio), text richness (4 dims: design completeness, measurable criteria ratio, risk specificity), evidence (4 dims: doc artifacts, freshness, research sources, coverage).
- **`VerdictReasonCategory`**: 9-category enum for structured human review reasons (technical_weakness, missing_evidence, scope_mismatch, policy_safety, stale_documentation, unnecessary_duplication, preference_style, wrong_lane, other).
- **`ShadowPredictionArtifact`**: Persisted per-review shadow prediction with full provenance — `acquisition_id`, `plan_id`, `plan_version`, predicted/actual classes, `reason_category`, `model_version`, `risk_tier`, `outcome_class`, feature vector, timestamps.
- **Verdict encoding**: 3-class softmax target (approved / rejected / needs_revision) covering all 7+ raw verdict strings.

### Hemisphere Integration

- Added `PLAN_EVALUATOR` to `HemisphereFocus` enum and `DISTILLATION_CONFIGS` (approximator 32→3, kl_div, min_samples=15, permanent).
- Added to `_TIER1_FOCUSES` in hemisphere orchestrator — automatic training/evaluation/lifecycle management.
- Custom `_prepare_plan_evaluator_tensors()` in `data_feed.py` — pairs features with verdict labels using `(acquisition_id, plan_id, plan_version)` metadata keys instead of timestamp windows (plans are discrete artifacts, not streams).
- Event bridge subscribes to `ACQUISITION_PLAN_REVIEWED` → light `KERNEL_THOUGHT`.

### Distillation Signal Pipeline

- **Plan features recorded** when technical design enrichment completes (`_enrich_plan_with_technical_design()`).
- **Verdict labels recorded** when human approves/rejects plan (`approve_plan()`).
- **Shadow prediction** runs when plan enters `awaiting_plan_review` — loads active specialist from `HemisphereRegistry`, infers via `HemisphereEngine`, stores `ShadowPredictionArtifact`.
- **Shadow resolution** compares prediction to actual verdict when human reviews, fills `correct`, `reason_category`, `model_version`.

### Events

- New: `ACQUISITION_PLAN_REVIEWED`, `ACQUISITION_DEPLOYMENT_REVIEWED` in `events.py`.
- `PlanReviewArtifact` now carries `reason_category` and `plan_version`.

### Dashboard

- `POST /api/acquisition/{id}/approve-plan` accepts `reason_category`.
- New `GET /api/acquisition/plan-evaluator` endpoint:
  - 5 maturity bands based on resolved sample count: bootstrap (<15), early_noisy (<50), preliminary (<100), meaningful (<250), stable (250+).
  - Stratified accuracy breakdowns: `accuracy_by_risk_tier`, `accuracy_by_outcome_class`, `accuracy_by_reason_category`.
  - Verdict distribution, sample counts from DistillationCollector, recent shadow artifacts.

### Tests

- 46 new tests in `tests/test_plan_encoder.py`: feature shape, clamping, classification block, relational quality, text richness, all verdict encodings, class mappings, reason categories, shadow artifact creation/provenance/roundtrip, evidence enrichment, edge cases.

---

## Capability Acquisition Pipeline + Self-Improvement Completion (2026-04-11)

**Milestone: Self-Improvement feature is now complete.** The majority of Jarvis's core features are shipped. This build delivers the Capability Acquisition Pipeline — the parent lifecycle that unifies all capability growth (research, planning, code generation, skill/plugin creation, verification, deployment) under one governed coordination layer. Self-improvement becomes one child lane of a broader system, not the entire story.

### Capability Acquisition Pipeline (7 phases shipped)

New `brain/acquisition/` package — the parent lifecycle for all capability growth:

- **`CapabilityAcquisitionJob`**: Top-level lifecycle object tracking intent → research → plan → implement → verify → deploy. Canonical `acquisition_id` links all child artifacts.
- **`IntentClassifier`**: Routes user "learn X" requests into outcome classes (knowledge_only, skill_creation, plugin_service, core_upgrade, specialist_nn, hardware_integration, mixed).
- **`AcquisitionPlanner`**: Synthesizes structured execution plans with risk tiers, implementation paths, and rollback strategies. Narrow mandate — coordinates, never absorbs lane execution.
- **`AcquisitionOrchestrator`**: Core coordinator — creates jobs, classifies intent, selects lanes, ticks active jobs through lane progression, persists state. Sacred guardrails prevent orchestration duplication.
- **`AcquisitionStore`**: JSONL + JSON persistence for jobs and plans under `~/.jarvis/acquisition/`.

| Lane | What It Does |
|------|-------------|
| Evidence Grounding | Internal evidence check before external research |
| Documentation Resolution | MCP/docs freshness bridge (DocumentationArtifact) |
| Planning | AcquisitionPlanner synthesizes execution plan |
| Plan Review | Human approval gate (PlanReviewArtifact) |
| Implementation | CodeGenService generates and validates code |
| Plugin Quarantine | PluginRegistry registers generated plugin in quarantined state |
| Verification | VerificationBundle cross-references lane-native proof |
| Skill Registration | Registers verified capability into SkillRegistry |
| Plugin Activation | Promotes plugin through quarantined → shadow → supervised → active |
| Truth Recording | AttributionLedger + memory write |

### CodeGen Service Extraction

Extracted shared code generation into new `brain/codegen/` package:

- **`CodeGenService`**: Unified `generate_and_validate()` API wrapping CoderServer + Sandbox. Used by both self-improvement and acquisition pipeline.
- **`CoderServer`** moved from `self_improve/coder_server.py` to `codegen/coder_server.py` (backward-compat re-export preserved).
- **`Sandbox`** moved from `self_improve/sandbox.py` to `codegen/sandbox.py` (backward-compat re-export preserved).

### Dynamic Plugin System

New `brain/tools/plugin_registry.py` — full lifecycle management for dynamically generated plugins:

- **Quarantine-first lifecycle**: quarantined → shadow → supervised → active → disabled.
- **Safety**: Import allowlist, runtime timeout, capability envelope, circuit breaker (5 failures → auto-disable), per-plugin audit trail (JSONL).
- **Routing**: `PluginRegistry.match()` wired into `tool_router.py` at Tier 0.75 (after hardcoded routes, before LLM fallback). New `ToolType.PLUGIN` enum.
- **Management**: Promote, disable, activate, rollback operations with full audit logging.
- **Dashboard**: Plugin management controls (promote/disable/activate/rollback) on the `/self-improve` page.
- **IPC-ready**: `PluginRequest`/`PluginResponse` contract designed for future process isolation (Tier 2+).

### Dashboard Overhaul

Complete overhaul of `/self-improve` page — now "Capability & Self-Improvement Pipeline":

- **7 tabbed sections**: Overview, Acquisition, Plugins, CodeGen, Scanner, Proposals, Analytics
- **Hero banner**: Pipeline stage, safety gates, acquisition/plugin counts, 8 key metrics
- **Acquisition tab**: Job list with lane progress bars, drill-down IDs, approve/deny buttons
- **Plugins tab**: Full table with state, risk tier, invocations, management controls
- **CodeGen tab**: Coder backend status + pipeline architecture diagram
- **Plugin management API**: `POST /api/plugins/{name}/promote|disable|activate|rollback`
- **CodeGen API**: `GET /api/codegen`

### Consciousness Integration

- New `ACQUISITION_*` event constants (10 events) in `events.py`
- Acquisition tick cycle added to consciousness system (300s interval, mode-gated)
- `acquisition` excluded from gestation mode's allowed cycles
- Wired through `ConsciousnessEngine.enable_acquisition()` and `main.py`

### Artifact Taxonomy

| Artifact | Authority | Mutability |
|----------|-----------|------------|
| CapabilityAcquisitionJob | AcquisitionStore | Mutable lifecycle state |
| AcquisitionPlan | AcquisitionStore | Versioned (append-new) |
| DocumentationArtifact | AcquisitionStore | Immutable once captured |
| PlanReviewArtifact | AcquisitionStore | Immutable |
| VerificationBundle | AcquisitionStore | Append-new per run |
| CapabilityClaim | AcquisitionStore | Immutable (revoke separately) |
| DeploymentRecord | AcquisitionStore | Immutable |
| PluginUpgradeArtifact | PluginRegistry | Versioned |

### Files Changed

| File | Change |
|------|--------|
| `brain/acquisition/__init__.py` | New — package exports |
| `brain/acquisition/job.py` | New — CapabilityAcquisitionJob + all artifact dataclasses + AcquisitionStore |
| `brain/acquisition/classifier.py` | New — IntentClassifier |
| `brain/acquisition/orchestrator.py` | New — AcquisitionOrchestrator (10 lane runners) |
| `brain/acquisition/planner.py` | New — AcquisitionPlanner with narrow mandate |
| `brain/codegen/__init__.py` | New — package exports |
| `brain/codegen/coder_server.py` | Moved from self_improve/ |
| `brain/codegen/sandbox.py` | Moved from self_improve/ |
| `brain/codegen/service.py` | New — CodeGenService unified API |
| `brain/tools/plugin_registry.py` | New — full plugin lifecycle management |
| `brain/tools/plugins/__init__.py` | New — plugin packages directory |
| `brain/consciousness/events.py` | Added 10 ACQUISITION_* event constants |
| `brain/consciousness/modes.py` | Added acquisition cycle, excluded from gestation |
| `brain/consciousness/consciousness_system.py` | Acquisition tick, enable_acquisition(), bug fixes |
| `brain/consciousness/engine.py` | enable_acquisition() wrapper |
| `brain/main.py` | AcquisitionOrchestrator init and wiring |
| `brain/reasoning/tool_router.py` | ToolType.PLUGIN + Tier 0.75 plugin routing |
| `brain/conversation_handler.py` | Plugin dispatch block |
| `brain/skills/resolver.py` | Acquisition-eligible skill templates |
| `brain/skills/learning_jobs.py` | parent_acquisition_id field |
| `brain/self_improve/patch_plan.py` | Added plugins/ and codegen/ to ALLOWED_PATHS |
| `brain/self_improve/coder_server.py` | Backward-compat re-export |
| `brain/self_improve/sandbox.py` | Backward-compat re-export |
| `brain/tools/codebase_tool.py` | skill_plugin write boundary |
| `brain/dashboard/app.py` | Acquisition + plugin + codegen API endpoints, plugin management |
| `brain/dashboard/snapshot.py` | acquisition + plugins in snapshot cache |
| `brain/dashboard/static/self_improve.html` | Complete overhaul — 7-tab capability pipeline dashboard |
| `brain/dashboard/static/renderers.js` | Acquisition + plugin panels on main dashboard |
| `brain/dashboard/static/dashboard.js` | Panel definitions for acquisition + plugins |

### Persistence (new)

| File | Contents |
|------|----------|
| `~/.jarvis/acquisition/jobs.jsonl` | Acquisition job lifecycle records |
| `~/.jarvis/acquisition/plans/` | AcquisitionPlan JSON files |
| `~/.jarvis/plugins/registry.json` | Plugin registry state |
| `~/.jarvis/plugins/audit/` | Per-plugin audit trail JSONL |

---

## Self-Improvement Sprint 1: Safety Gates + Coder Backend + Dashboard (2026-04-10)

End-to-end self-improvement pipeline: metric-driven opportunity detection, local 80B code generation, advanced dashboard, and process safety.

### Scanner overhaul

- **Metric-driven detectors**: Replaced vague `observer.detect_improvement_opportunities()` with 6 concrete detectors in `consciousness_system._si_detect_opportunities()`: health degradation, reasoning quality decline, confidence volatility, response latency spikes, event bus error rate, tick performance regression. Each has maturity guards (min 30min uptime, min sample counts).
- **Fingerprint dedup**: Every opportunity gets a deterministic `fingerprint` (hash of detector + metric + threshold). 4-hour in-memory cooldown + 24-hour check against past proposals in `improvement_proposals.jsonl`. Prevents rescanning identical issues.
- **Daily attempt cap**: Max 6 LLM generation attempts per day (`_SI_MAX_ATTEMPTS_PER_DAY`).
- **`ImprovementRequest` enriched**: Added `fingerprint` and `evidence_detail` fields, wired through to provider prompt for grounded code generation.
- **Scan logging upgraded**: INFO-level scan summaries with detector counts, opportunities found, dedup stats.

### Coder backend (Qwen3-Coder-Next via llama-server)

- **`coder_server.py` (new)**: On-demand `llama-server` lifecycle manager — starts process, polls `/health`, sends OpenAI-compatible chat completions, shuts down after generation to reclaim ~46GB RAM. Pure CPU by default (`CODER_GPU_LAYERS=0`).
- **`provider.py` rewritten**: Local `CoderServer` is primary codegen path. Claude/OpenAI removed from self-improvement hot path. JSON retry logic with repair prompt on parse failures.
- **`config.py`**: New `CoderConfig` Pydantic model with auto-detection — checks model file existence + `llama-server` binary (PATH or `~/.local/share/llama.cpp/build/bin/`).
- **`main.py` wiring**: `CoderServer` created and wired into `PatchProvider` before `SelfImprovementOrchestrator` init, so provider availability check in `__init__` sees the coder.
- **`orchestrator.py`**: Accepts optional `provider` parameter to support pre-wired providers.

### Setup & installation

- **Model download**: `setup.sh` uses `curl` with direct HuggingFace URL (`https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-UD-Q4_K_XL.gguf`), supports resume (`-C -`) for the ~46GB file, falls back to `huggingface-cli`.
- **llama-server auto-build**: If `llama-server` not in PATH, setup.sh clones `llama.cpp`, builds from source (CPU-only), writes `CODER_LLAMA_SERVER` path to `.env`.
- **Process cleanup**: `setup.sh` now kills `jarvis-supervisor`, `python main.py`, and `llama-server` on startup, frees port 8081 alongside 9100/9200.
- **`.env` sourcing**: `setup.sh` sources `.env` via `set -a` so user settings (like `ENABLE_CODER_MODEL=true`) are honored during setup.

### Dashboard

- **`/self-improve` page**: Full 7-panel advanced dashboard (scanner, coder backend, proposals, analytics, conversations, timeline) — served from `self_improve.html`.
- **`/api/self-improve/scanner`** and **`/api/self-improve/coder`**: New API endpoints for scanner state and coder server status.
- **Main dashboard Learning tab**: `_renderSelfImprovePanel` updated with coder status badge, scanner stats (scans/opportunities/daily gen/fingerprints), recent proposals with fingerprint tags, "Full Dashboard" link to `/self-improve`.
- **Snapshot cache**: `_build_self_improve_cache()` now includes `scanner` and `coder` subsections.

### Files changed

| File | Change |
|------|--------|
| `brain/self_improve/coder_server.py` | New — on-demand llama-server lifecycle manager |
| `brain/self_improve/provider.py` | Rewritten — local CoderServer primary, removed external API cascade |
| `brain/self_improve/orchestrator.py` | Added `provider` param, wired fingerprint/evidence to proposals |
| `brain/self_improve/improvement_request.py` | Added `fingerprint`, `evidence_detail` fields |
| `brain/consciousness/consciousness_system.py` | New `_si_detect_opportunities()`, scanner state, dedup logic |
| `brain/consciousness/observer.py` | Deprecated `detect_improvement_opportunities()` |
| `brain/config.py` | New `CoderConfig`, auto-detect llama-server local build |
| `brain/main.py` | Wire CoderServer before orchestrator init |
| `brain/setup.sh` | curl model download, llama-server build, process cleanup, .env sourcing |
| `brain/.env.example` | New coder section (6 env vars) |
| `brain/dashboard/app.py` | New `/api/self-improve/scanner` and `/coder` endpoints |
| `brain/dashboard/snapshot.py` | Scanner + coder data in snapshot cache |
| `brain/dashboard/static/self_improve.html` | New 7-panel advanced dashboard page |
| `brain/dashboard/static/renderers.js` | Updated Learning tab self-improvement panel |

---

## System Wiring Audit — 8 Critical/Medium Fixes (2026-04-09)

Comprehensive audit found broken wiring in 8 subsystems despite code existing. All fixes validated against live brain logs after deployment.

### Fixes shipped

| # | Subsystem | File(s) | What was broken | Fix |
|---|-----------|---------|----------------|-----|
| 1 | Distillation normalization | `hemisphere/data_feed.py` | z-score only applied to `audio_features` (exact match), missed `audio_features_enriched` — `emotion_depth` regressed every cycle (~53%) | Changed to `source.startswith("audio_features")` — first cycle reached 69.9%, converging |
| 2 | Intervention measured_delta | `autonomy/interventions.py`, `intervention_runner.py`, `orchestrator.py` | `measured_delta` always 0.0, `baseline_value` never captured — all interventions discarded as "no improvement" | Added `baseline_value` field, capture at shadow activation, compute delta at check — confirmed baseline capture in logs |
| 3 | WORLD_MODEL_DELTA emission | `cognition/world_model.py` | Event declared in events.py but never emitted — downstream consumers (EvalSidecar, curiosity) deaf to world changes | Added `event_bus.emit(WORLD_MODEL_DELTA, ...)` in `_detect_deltas()` loop |
| 4 | Source ledger → scoring | `autonomy/opportunity_scorer.py` | `SourceUsefulnessLedger` existed but never fed back into opportunity scoring — research quality had no memory | Added `get_topic_usefulness()` call in `_compute_policy_adjustment()`, ±0.15 influence |
| 5 | Retrieval double-counting | `memory/retrieval_log.py` | Both `_notify_source_ledger()` and `log_outcome()` called `record_retrieval()` — every injection counted twice | Removed `record_retrieval()` from `_notify_source_ledger()`, made `log_outcome()` single authority |
| 6 | CueGate bypass | `consciousness/consciousness_system.py` | `_run_association_repair()` wrote memories without checking CueGate — violated memory safety invariant | Added `memory_gate.can_observation_write()` guard at entry |
| 7 | Personal intel junk | `conversation_handler.py` | No content filter for transient states ("User is Jarvis", "User is pushing through") | Added `_is_unstable_personal_fact()` filter + `_CORRECTION_PATTERNS` + `_correct_recent_facts()` |
| 8 | FRACTAL_RECALL_SURFACED tap | `jarvis_eval/event_tap.py` | Event not in `_TAPPED_EVENTS` list — PVL blind to fractal recall activity | Added to tapped events list |

### Documentation updated

- AGENTS.md: added Personal Intel Capture Pipeline, Intervention Lifecycle, Source Ledger Feedback Loop, Friction Pipeline, CueGate background-write invariant, vector embedding extraction, z-score normalization detail
- ARCHITECTURE.md: added Data Flow: Personal Intel & Preference Capture, source ledger feedback in OpportunityScorer section, source ledger retrieval counting safety guard, z-score normalization in distillation pipeline, updated WORLD_MODEL_DELTA + FRACTAL_RECALL_SURFACED subscriber lists
- BUILD_HISTORY.md: this entry
- Also fixed: `test_simulator.py` stale event name strings, `emotion.py` silent distillation exception

---

## Engineering Architecture Trace + Wiring Validation (2026-04-08)

This documentation/evidence build added engineering-grade architecture tracing
for open-source due diligence and linked it across all primary design docs.

### 1. New engineering trace artifact

- Added `docs/ENGINEERING_ARCHITECTURE_TRACE.md` with:
  - boot ownership wiring diagram,
  - Pi-to-brain voice path sequence wiring,
  - scene/spatial/world-model/truth bridge diagram,
  - autonomy intervention lifecycle diagram,
  - dashboard snapshot cache wiring diagram,
  - code-to-runtime validation matrix with bridge-level pass status.

### 2. Live runtime evidence captured

- Verified wiring against the running brain via `/api/full-snapshot` and live log traces:
  - spatial pipeline healthy (`spatial.calibration.state=valid`),
  - world model promoted and validating predictions,
  - truth calibration and trust wiring present,
  - autonomy intervention/source/friction telemetry surfaced.
- Re-ran validation pack:
  - `31/32` current checks passing, `32/32` ever passing,
  - Phase 5 proof chain reported PASS.

### 3. Cross-doc integration

- Updated link surfaces so reviewers can navigate pillars -> process -> implementation trace:
  - `ARCHITECTURE.md`
  - `docs/ARCHITECTURE_PILLARS.md`
  - `PROCESS_ARCHITECTURE.md`
  - `docs/SYSTEM_OVERVIEW.md`
  - `README.md`
  - `TODO.md` design doc status table

---

## Architecture Pillars Documentation Refresh (2026-04-08)

This documentation build prepared Jarvis for open-source handoff by defining
project pillars, cross-linking architecture/process docs, and adding explicit
"do vs do not" operating rules.

### 1. New canonical pillar contract

- Added `docs/ARCHITECTURE_PILLARS.md`:
  - 10 architecture pillars with contracts and trace/validation surfaces,
  - bridge-map of common failure points,
  - contributor guardrails for truth, identity, synthetic lane, and promotion safety.

### 2. Process architecture aligned to pillars

- Updated `PROCESS_ARCHITECTURE.md`:
  - refreshed timestamp and references,
  - added `Open-Source Process Invariants` section mapping pillars to operator rules,
  - added minimal release validation loop and hard no-go rules.

### 3. System overview aligned for public docs

- Updated `docs/SYSTEM_OVERVIEW.md`:
  - added pillar summary section (contract + quick validation surface),
  - added concise do/do-not block for contributor orientation.

### 4. Cross-doc discoverability updates

- Updated `ARCHITECTURE.md` to reference pillar contract as architecture invariant source.
- Updated `README.md` docs index to include `docs/ARCHITECTURE_PILLARS.md`.
- Updated `TODO.md` design-doc status table to track the new pillar contract doc.

---

## Spatial 1.5 + Language Truth Alignment + Phase 5 Runtime Hardening (2026-04-07)

This build closed the Spatial 1.5 follow-up, aligned runtime language evidence
reasons to live diagnostics, and hardened Phase 5 intervention runtime behavior.

### 1. Spatial 1.5 relocalization/profile handoff shipped

- Updated `brain/perception/calibration.py`:
  - introduced calibration profile persistence/activation/matching APIs,
  - added active-profile telemetry and handoff advisory semantics.
- Updated `brain/perception_orchestrator.py`:
  - scene-signature generation + relocalization trigger path,
  - profile match/handoff and `SPATIAL_CALIBRATION_CHANGED` emission,
  - relocalization telemetry in spatial state.
- Updated `brain/perception/spatial.py` and `brain/cognition/spatial_validation.py`:
  - added relocalization reset hooks for track and temporal baseline safety.

### 2. Validation-pack language reason truth alignment

- Updated `brain/jarvis_eval/validation_pack.py`:
  - language evidence rows now prefer live class diagnostics for current truth,
  - non-red rows normalize reason to `ok` (or `insufficient_samples`),
  - promotion metadata remains historical context, not primary current-row truth.
- Result: capability/language evidence rows no longer show stale blocking reasons
  when class color is green.

### 3. Phase 5 intervention runtime hardening

- Updated `brain/autonomy/intervention_runner.py`:
  - append-only load dedupe by `intervention_id` (latest state wins),
  - stale proposed trim + unresolved cap enforcement on load,
  - transition persistence for `shadow`, `promoted`, `discarded`,
  - richer unresolved/cap stats for runtime observability.
- Updated `brain/autonomy/orchestrator.py`:
  - autonomy status now exposes `interventions`, `source_ledger`, `friction`,
  - intervention extraction widened (`routing_rule`, `prompt_frame`,
    `memory_weighting_rule`, `eval_contract`) with safe fallback for
    friction/metric intents with generic findings.
- Updated `brain/autonomy/knowledge_integrator.py`:
  - tracked source lineage per intent for stronger intervention source binding.

### 4. Tests and validation

- Added/updated tests:
  - `brain/tests/test_spatial_relocalization.py`
  - `brain/tests/test_intervention_runner.py`
  - `brain/tests/test_phase5_intervention_extraction.py`
  - extensions in spatial/validation-pack test suites.
- Validation:
  - spatial/language focused suite: **89 passed** (local + desktop),
  - Phase 5 intervention suites: **161 passed** (local + desktop),
  - lints clean for edited files.

### 5. Runtime checkpoint outcome

- Controlled voice phases generated friction and triggered `metric:friction_rate` research.
- First live actionable intervention flow observed:
  `iv_37e33b9b23ec` proposed then activated in shadow.
- Phase 5 remains in runtime accumulation: promotion requires elapsed shadow window
  and measured delta evidence.

---

## PVL Matrix Truth-Shoring + Spatial Local-Only Aging (2026-04-05)

This build removed a readiness false-negative in PVL matrix reporting and aligned
spatial calibration behavior with mobile/scene-changing operation.

### 1. Spatial calibration aging semantics hardened

- Updated `brain/perception/calibration.py`:
  - long-aged calibration now remains `stale` (advisory/local-only), not hard `invalid`,
  - `invalid` is reserved for hard trust breaks (`uncalibrated`, `anchor_inconsistency`),
  - added explicit `reason` in calibration state (`verification_expired_local_only`, etc.).
- Result: spatial tracking continues under stale calibration instead of fail-closed shutdown
  when camera context changes over time.

### 2. Matrix protocol PVL contracts aligned to persisted reality

- Updated `brain/jarvis_eval/collector.py` matrix snapshot payload:
  - added `completed_matrix_jobs`,
  - added `matrix_jobs_observed` (`active + completed`).
- Updated `brain/jarvis_eval/process_contracts.py`:
  - `matrix_active_jobs` now checks `matrix_jobs_observed` (active or completed),
  - `matrix_dl_requested` now uses `missing_event_status='awaiting'`,
  - `matrix_specialists_exist` applicability narrowed to `deep_learning` mode.
- This preserves strict checks while removing false "no matrix progress" failures when
  matrix work completed but no active job remains.

### 3. PVL snapshot failure evidence fixed

- Updated `brain/jarvis_eval/process_verifier.py` failure descriptions for snapshot contracts:
  - correctly reports `above max` vs `below threshold`,
  - includes observed value in evidence text.
- Fixes misleading evidence strings such as
  `"promotion_red_classes below threshold (0.0)"` for max-threshold failures.

### 4. Tests and validation

- Added/updated tests in `brain/tests/test_process_verifier.py` for:
  - matrix observed-job pass semantics,
  - matrix specialist contract mode gating,
  - matrix deep-learning request awaiting behavior,
  - max-threshold failure evidence formatting.
- Validation:
  - local: `tests/test_process_verifier.py` (**55 passed**),
  - local: `tests/test_eval_sidecar.py` + `tests/test_eval_sidecar_non_invasive.py` (**48 passed**),
  - brain host: same combined set (**103 passed**).

### 5. Runtime note

- Live dashboard/PVL output reflects this build only after the running brain process
  is restarted (current session may still show old contract behavior until reload).

---

## Phase D Runtime Consumption (Guarded) — SHIPPED (2026-04-04)

This build completed the guarded runtime-consumption bridge for language promotion
while keeping fail-closed and strict-native invariants intact.

### 1. Runtime bridge (reversible, default OFF)

- Added explicit runtime rollout config in `brain/config.py`:
  - `ENABLE_LANGUAGE_RUNTIME_BRIDGE`
  - `LANGUAGE_RUNTIME_ROLLOUT_MODE` (`off|canary|full`)
  - `LANGUAGE_RUNTIME_CANARY_CLASSES`
- Added `brain/reasoning/language_runtime_bridge.py` as the single policy reader:
  - reads runtime config with fail-closed fallback,
  - reads promotion level via `get_level()`/`get_summary()` only,
  - does not call `evaluate()` on the request hot path.

### 2. Conversation hot-path integration (guarded scope)

- `brain/conversation_handler.py` now applies runtime-consumption policy only to
  bounded-vs-LLM consumption decisions (no route remaps, no student reply authority).
- Strict-native routes remain deterministic and untouched by rollout gating:
  `self_status`, strict learning/research answers, `identity_answer`, `capability_status`.
- Introspection bounded path is now promotion-aware when rollout is enabled; when blocked,
  it falls back to grounded LLM path and records explicit runtime-guard telemetry.

### 3. Runtime diagnostics wired end-to-end

- Extended `brain/reasoning/language_telemetry.py` with runtime guard metrics:
  live usage by class, blocked-by-guard, unpromoted-live attempts, red-class live use,
  plus by-mode/by-level/by-reason distributions.
- Wired through:
  - `brain/jarvis_eval/collector.py`
  - `brain/jarvis_eval/dashboard_adapter.py`
  - `brain/dashboard/static/renderers.js`
  - `brain/dashboard/static/dashboard.js`
- Dashboard now surfaces runtime guard mode/state and class-level live/blocked counters.

### 4. Validation + PVL runtime safety contracts

- `brain/jarvis_eval/validation_pack.py` now includes runtime safety checks:
  - `language_runtime_guardrails` (critical only when rollout is enabled),
  - `language_runtime_telemetry`.
- Added persisted runtime PVL contracts in `brain/jarvis_eval/process_contracts.py`:
  - `lang_runtime_unpromoted_live_zero`
  - `lang_runtime_live_red_zero`

### 5. Regression and safety proof

Updated/added coverage:
- `tests/test_language_promotion.py`
- `tests/test_language_eval_sidecar.py`
- `tests/test_validation_pack.py`
- `tests/test_process_verifier.py`
- `tests/test_tool_router.py`
- `tests/test_bounded_response.py`
- `tests/test_language_quality_telemetry.py`

Validation runs:
- Targeted Phase D suite: **163 passed**
- `tests/test_eval_sidecar.py`: **42 passed**
- `tests/test_eval_sidecar_non_invasive.py`: **6 passed**
- `tests/test_reflective_introspection.py`: **92 passed**

---

## Intent Policy Clarification + Dashboard Reliability + Soul Integrity Fix (2026-04-02)

This session focused on truth-bound intent handling and dashboard/runtime reliability,
without adding new generative authority.

### 1. Intent policy clarification (learning-first, no brittle remap)

- Clarified operator policy for ambiguous phrase handling:
  - Ambiguous user wording should **not** be hard-mapped to a stricter
    alternate intent class in routing rules.
  - Interpretation should improve through shadow learning, correction evidence,
    and promotion gates over time.
- Planning docs updated to reflect this policy:
  - `TODO.md` immediate follow-up changed from hard route patch to
    ambiguous-intent learning lane.
  - `docs/MASTER_ROADMAP.md` active target updated with the same constraint.

### 2. Dashboard panel behavior fixes

- Fixed panel collapse UX by restoring missing collapse CSS behavior:
  - `.panel.collapsed .panel-body { display: none; }`
  - chevron rotation + panel-header interaction styling.
- Fixed Stability Chart blank render on expand:
  - Added redraw-on-expand path so hidden-canvas first paint no longer leaves
    the chart blank after toggling.
- Stability Chart default state changed to open (not collapsed) for immediate visibility.

Files:
- `brain/dashboard/static/style.css`
- `brain/dashboard/static/dashboard.js`

### 3. Soul Integrity metric correctness fix

- Fixed Layer 10 `skill_honesty` scorer to consume the real CapabilityGate stat key:
  - `claims_blocked` (with legacy fallback), instead of non-existent `total_blocks`.
- Updated source text in dimension details to report `claims_blocked`.
- Added regression test covering this key path and score math.

Files:
- `brain/epistemic/soul_integrity/index.py`
- `brain/tests/test_soul_integrity.py`

### 4. Reset-resilient maturity gate tracking

- Added persisted maturity high-water state at `~/.jarvis/eval/maturity_highwater.json`.
- Dashboard maturity gates now preserve `ever_met` evidence across restarts:
  - keeps **current** status truth (`active/progress/locked`) unchanged,
  - adds historical context (`ever_active_gates`, per-gate `ever_met` metadata),
  - UI badge now shows `active` and `ever` totals side-by-side.
- Per-gate visual cue:
  - green check = active now,
  - cyan diamond = previously met, currently below threshold.

Files:
- `brain/jarvis_eval/dashboard_adapter.py`
- `brain/dashboard/static/dashboard.js`
- `brain/tests/test_maturity_tracker_highwater.py`

### 5. Language shadow telemetry kickoff

- Added shadow-only ambiguous-intent probe telemetry for ambiguous self-referential phrasing.
- No route remap was introduced; live routing authority remains unchanged.
- Probe records include selected route, candidate intent, confidence, trigger, outcome, and feedback.
- Dashboard language panel now surfaces ambiguous-intent counters (total, NONE routes, corrections).

Files:
- `brain/conversation_handler.py`
- `brain/reasoning/language_telemetry.py`
- `brain/dashboard/static/renderers.js`
- `brain/tests/test_language_quality_telemetry.py`

Validation:
- `tests/test_soul_integrity.py` -> **31 passed** (local + brain host)
- Dashboard JS syntax check passed
- `tests/test_maturity_tracker_highwater.py` -> **3 passed**
- `tests/test_language_eval_sidecar.py` -> **2 passed**
- `tests/test_language_quality_telemetry.py` -> **4 passed**

---

## Language Substrate Phase C + Runtime Diagnostics (2026-04-01)

Phase C was shipped as a strict shadow-only lane with reset-aware observability.
No user-facing response authority was granted to shadow models.

### 1. Phase C language harness shipped (shadow-only)

**New module**: `reasoning/language_phasec.py`
- Baseline lock + checklist state persisted to `~/.jarvis/language_corpus/phase_c/baseline_lock.json`.
- Deterministic tokenizer strategy evaluation (BPE vs SentencePiece-style comparison) with persisted decision metadata.
- Grounded objective contract: `next_token_grounded_pair` dataset from `query` + `response_class` + `meaning_frame` + `final_answer`.
- Negative examples and low-confidence rows excluded from positive objective training.
- Deterministic split contract (`sha1(sample_id)%100`) with split manifest persistence.
- Checkpoint/resume for a bounded adapter student with explicit cold-start reasons (`checkpoint_missing`, `checkpoint_corrupt`, `insufficient_train_samples`, etc.).

### 2. Runtime wiring and safety hardening

**Consciousness cycle integration** (`consciousness/consciousness_system.py`):
- Existing shadow style model cycle retained.
- Phase C harness cycle added in parallel.
- Boot warm-load of Phase C checkpoint.

**Conversation safety** (`conversation_handler.py`):
- Shadow comparisons now run style-model and Phase C adapter side-by-side.
- Phase C path is telemetry-only and guarded by explicit `is_live_routing_enabled()` check.
- No replacement of live replies from shadow output.

**Telemetry isolation** (`reasoning/language_telemetry.py`):
- `shadow_comparison` events are now tracked separately from quality event totals.
- New breakdowns: by class, model family, choice, reason, and recent comparison window.

### 3. Eval, promotion, and dashboard observability

**Eval collector** (`jarvis_eval/collector.py`):
- Added per-class gate payload (`gate_scores_by_class`).
- Added richer promotion summary payload.
- Added embedded `phase_c` runtime payload.

**Promotion governor stabilization** (`jarvis_eval/language_promotion.py`):
- Added evaluation signature dedupe to prevent counter inflation from repeated identical snapshots after resets.
- Persisted governor meta signature in state file.

**Dashboard/API**:
- `dashboard/snapshot.py` now exports Phase C diagnostics.
- `dashboard/app.py` adds `GET /api/language-phasec`.
- `dashboard/static/renderers.js` language panel now shows tokenizer choice, vocab estimate, split counts, training recency, student readiness, and reset-aware context.
- `jarvis_eval/dashboard_adapter.py` includes Phase C summary in eval payload.

### 4. Validation and test coverage

New/updated test coverage:
- `tests/test_language_phasec.py`
- `tests/test_language_promotion.py`
- `tests/test_language_quality_telemetry.py` (shadow isolation assertions)

Focused regression matrix run:
- `204 passed` (Phase C + language + introspection safety suites)
- `test_shadow_language_model.py` remains skipped in this environment (expected skip condition).

### 5. Post-sync runtime verification

Desktop restart and API checks confirmed:
- Phase C harness cycle executed on boot (`status=trained`, dataset samples present).
- `GET /api/language-phasec` returns tokenizer, dataset/split, student, checkpoint, and guard state.
- Guard state confirms shadow-only (`shadow_only_enforced=true`, `live_routing_enabled=false`).

### 6. Runtime diagnostic note (open issue)

During live investigation of "questions not firing":
- Dominant drop mode was wake-word miss (wake scores below threshold over long windows).
- When wake fired, STT + routing + response pipeline worked.
- One ambiguous self-referential phrasing gap was observed routing to `NONE` (intent pattern coverage issue).

---

## Identity Graceful Degradation + Planner Shadow Scaffold (2026-04-01)

Two forward steps shipped together: final Priority 4 identity hardening and initial
Priority 5 planner architecture in safe shadow mode.

### 1. Identity Edge Case Hardening — Wave 4 complete

**Voice-drop graceful degradation** (`perception/identity_fusion.py`):
- Added short continuity window for brief voice-confidence drops while face still confirms
  recent known identity.
- New resolution path: `face_voice_drop_grace` (trust explicitly downgraded to `degraded`
  instead of abrupt unknown).
- Safety gates: blocked for multi-person scenes, stale prior identity, and near-zero active
  voice confidence.
- Telemetry: `voice_drop_recent_identity`, `voice_drop_recent_age_s`,
  `voice_drop_recent_method`.

**Conversation-boundary persistence** (previous wave, now live-verified post-restart):
- `conversation_boundary_grace_*` fields active in snapshot.
- Wake-word continuity observed via `conversation-boundary grace` log path.

**Tests**:
- `test_identity_wave3.py` + `test_identity_fusion.py` pass (107 tests).

### 2. Phase 7 Planner — shadow architecture scaffold

**New module**: `cognition/planner.py`
- Added deterministic `WorldPlanner` + `PlanOption` models.
- Advisory-only design: no actuation, no event emission, no state mutation.
- Hard gate: planner stays disabled until simulator promotion reaches advisory.
- Utility scoring combines projected confidence, uncertainty-derived risk penalty,
  and lightweight goal-alignment boost.

**World model integration** (`cognition/world_model.py`):
- Planner runs each world-model tick in shadow mode.
- Planner snapshot now exposed under `world_model.planner` and diagnostics.
- On calm ticks/no meaningful deltas, planner reports `active=false, reason=no_deltas`
  (expected behavior).

**Tests**:
- Added `test_planner.py` (gating, ranking, goal alignment, risk penalty).
- Updated `test_world_model.py` for planner state contract.
- Updated sandbox cognition test map to include planner suite.
- Validation runs:
  - `test_planner.py` + `test_world_model.py` (49 passed)
  - `test_simulator.py` (42 passed)

### 3. Post-sync restart verification

- Snapshot confirms planner block present and healthy.
- No startup/runtime faults in post-boot log window (`ERROR=0`, `Traceback=0`,
  `Exception=0`).
- Live planner status confirmed:
  - simulator advisory: `true`
  - planner enabled: `true`
  - planner active: context-dependent (`no_deltas` when scene/conversation quiet)

### 4. Goal-aligned planning shadow bridge (complete)

Shipped the remaining Phase 7 TODO item by wiring planner recommendations into
goal/autonomy dispatch policy metadata in **shadow-only** mode.

**Goal planner bridge** (`goals/planner.py`):
- `create_intent_from_task()` now reads `world_model.planner.selected` when active
  and projects a goal-alignment hint for research tasks.
- Planner metadata is attached to `ResearchIntent` (event, utility, alignment,
  recommendation, reason) but does not alter scope/hint/dispatch behavior.

**Autonomy policy preview** (`autonomy/orchestrator.py`):
- Added `_build_shadow_policy_preview()` that computes a hypothetical priority delta
  from planner utility + alignment.
- Preview is annotated in intent reason and exposed in status under
  `planner_shadow_policy` with `applied=false`.
- No queue ordering or execution logic is changed by this preview path.

**Data contract + tests**:
- `autonomy/research_intent.py` now serializes planner shadow fields for observability.
- Added tests for:
  - planner bridge metadata propagation (`test_goal_bridge_and_alignment.py`)
  - autonomy shadow policy preview generation (`test_goal_bridge_and_alignment.py`)
  - research intent shadow field serialization (`test_goals.py`)
- Validation run: `test_goal_bridge_and_alignment.py` + `test_goals.py` +
  `test_planner.py` (276 passed).

---

## Memory Purge + VectorStore Reconnect + Spatial Calibration (2026-03-31)

Three-fix deployment: memory cleanup, VectorStore resilience, and spatial auto-calibration.

### 1. Memory Purge
Pre-existing memory pollution from recursive consolidation pipeline (identified in same session).

| Metric | Before | After |
|--------|--------|-------|
| Total memories | 2,000 | 1,225 |
| Hollow consolidations removed | — | 762 |
| Exact duplicates removed | — | 13 |
| Core memories | 4 | 4 (preserved) |
| Avg weight | 0.602 | 0.635 |
| VectorStore vectors | 3,487 (stale) | 1,225 (rebuilt) |

Backup: `~/.jarvis/memories.backup-1774972563.json`
Script: `brain/scripts/purge_hollow_memories.py` (dry-run by default)

### 2. VectorStore Reconnect (`memory/vector_store.py`)
Added `_ensure_conn()` method that detects stale/None SQLite connections and auto-reconnects. Called before every `add()`, `search()`, `remove()`, and `_count()` operation. Prevents the `NoneType` errors seen during sleep mode transitions.

### 3. Spatial Calibration Auto-Setup (`perception/calibration.py`)
Added `setup_pi_camera_defaults()` to CalibrationManager:
- IMX708 sensor intrinsics: f=470.3px at 640x480 output
- Default camera position: (0, 1.2m, 0) — desk height
- Called on boot by PerceptionOrchestrator if calibration is unconfigured
- `POST /api/spatial/calibration` dashboard endpoint added for runtime tuning

**Post-deploy verification:**
- Calibration: `valid` v1 immediately on boot (was `invalid` before)
- Spatial tracks accumulating: 7 tracks within 3 minutes, **first stable track** (monitor at 2.87m, 5 samples)
- Fused entities: 12 (spatial merging with scene continuity)
- VectorStore: zero errors, rebuilt cleanly with 1,225 vectors
- Memory: 1,227 after 3 min runtime (+2 natural writes from dream cycle)
- Soul Integrity: 0.820

---

## Post-Sync Restart Verification (2026-03-31)

First boot after syncing Spatial Intelligence Phase 1 + Consolidation Quality Fix.

**Boot results (all gates pass):**

| Gate | Result | Detail |
|------|--------|--------|
| Boot safety | PASS | Supervisor + main process running, no import errors, dashboard accessible |
| Face ID | PASS | David recognized (81-89%), identity fusion working |
| STT/TTS pipeline | PASS | Wake word → STT → route → LLM → TTS flowing cleanly |
| Spatial panel | PASS | `status: active`, calibration `invalid` (this checkpoint was captured before auto-default calibration landed) |
| Spatial tracks | PASS | 0 tracks, 0 promotions, 0 memory writes (suppressed while calibration remained invalid) |
| Display surfaces | PASS | **13 → 4** (cross-label matching fix working — 3 monitors + 1 laptop) |
| Scene entities | PASS | 16 → 9-13 range (cleaner, no stale accumulation) |
| Memory count | PASS | Stable at 2,000 (no jumps) |
| World model | PASS | Level still `active`, `spatial_active: True` |
| Consciousness | PASS | Integrative stage, transcendence 10.0, system_healthy=True |
| Oracle Benchmark | Expected | 95.6 → 83.5 (normal post-restart — accumulation metrics reset to zero) |

**Spatial pipeline log confirmation:**
```
Spatial update #1: 0 tracks (0 stable), cal=invalid
Spatial update #2: 0 tracks (0 stable), cal=invalid
Spatial update #3: 0 tracks (0 stable), cal=invalid
```

**Consolidation fix confirmed active:** Pre-sync dream cycles showed `low-score`
rejections in consolidation logs (new cluster score guard working). Post-sync
consolidation will be monitored for content quality.

**Known pre-existing:** VectorStore `NoneType` errors during sleep mode transitions
(connection drops on SQLite, not new). Does not affect core memory operations.

---

## Consolidation Pipeline Quality Fix (2026-03-31)

Pre-sync memory audit found 59% of memory capacity (1,180 / 2,000 slots) consumed by
hollow consolidation records containing nothing but recursive meta-headers like
`[Consolidated from 10 memories] [Dream artifact: consolidation_proposal]
Consolidation: consolidated (5 memories, coherence=0.87)` — zero actual content.

**Root cause — three cooperating bugs:**

1. **Recursive consolidation input** (`consciousness_system.py`): Dream cycle input
   filter only excluded `dream_insight` tags. Memories tagged `consolidated` and
   `dream_consolidation` from previous cycles were fed back into clustering, where
   they formed high-similarity clusters (all sharing the same consolidation tags),
   and got re-consolidated into deeper meta-header chains.

2. **No content stripping** (`consolidation.py`): `_build_summary()` concatenated
   source memory text as-is. When sources were previous consolidation outputs, the
   `[Consolidated from N memories]` headers accumulated recursively.

3. **No content quality gate** (`consolidation.py`): The only empty-check was
   `if not texts: return None`, which passed for meta-header-only strings since
   they're non-empty strings. No check for whether the text contained actual
   informational content.

**Three-layer fix:**

- **Fix 1** (`consciousness_system.py` line ~2084): Dream cycle input now excludes
  memories tagged `consolidated` or `dream_consolidation` in addition to
  `dream_insight`. Prevents consolidated memories from entering the clustering pool.

- **Fix 2** (`consolidation.py` `_build_summary`): New `_strip_meta_headers()` method
  removes all consolidation/dream meta-header patterns via regex. Source texts are
  stripped before inclusion; texts with <20 chars of real content after stripping are
  rejected. If no source memory contributes real content, `_build_summary` returns None.

- **Fix 3** (`consolidation.py` `_score_cluster`): Clusters where >=50% of members
  carry `consolidated` or `dream_consolidation` tags are rejected with score -1.0.
  Belt-and-suspenders defense in case consolidated memories leak past the input filter.

**Pre-fix memory audit (25.6h uptime):**

| Category | Count | % |
|----------|-------|---|
| Truly useful content | ~715 | 36% |
| Meta-only consolidations | 1,136 | 57% |
| Hollow dream observations | 44 | 2% |
| Exact duplicates | ~105 | 5% |

**Tests:** 17 new tests in `test_consolidation_quality.py` covering meta-header
stripping, hollow cluster rejection, score-cluster guard, and full pipeline quality.
All 76 memory/dream/consciousness tests passing, zero regressions.

**Impact:** After this fix, future dream cycles will only produce consolidation
summaries containing actual informational content. Existing hollow memories will
gradually evict through normal weight decay and priority-aware eviction since they
have low access counts (88% never accessed) and moderate weights (0.54-0.56 band).

---

## Spatial Intelligence Phase 1 (2026-03-31)

Full implementation of physically grounded spatial room awareness — the first subsystem
to give JARVIS metric understanding of where things are, how far they moved, and how big
they are, rather than just semantic region labels.

**Pre-sync snapshot (25.6h uptime, 2026-03-31):**

| Metric | Value |
|--------|-------|
| Oracle Benchmark | 95.6/100 (Oracle Ascendant) |
| World Model | Level 2 (active), 5,443 validated, 9 misses |
| Mental Simulator | Shadow, 87.0% accuracy, avg depth 2.8 |
| Policy NN | v61, 11,143 decisions, mlp2_enc2 |
| Soul Integrity | memory_coherence=0.94, belief_health=0.80, quarantine=1.0, audit=0.92 |
| Memory | 2,000 memories |
| Autonomy | Level 2 (safe_apply), 121 completed (6 this session) |
| Consciousness | Integrative stage, transcendence=10.0, awareness=0.98 |
| Scene | 16 entities, 13 display surfaces (pre-fix) |
| Spatial | Not present (pre-sync) |

**Prerequisite fixes:**

- **Display surface cross-label matching**: `scene_tracker.py` `_match_display_surface()`
  now allows "tv" to match existing "monitor" surfaces (and vice versa) within
  `DISPLAY_SURFACE_LABELS`. Previously, YOLO label oscillation created duplicate
  surfaces for the same physical monitor (13 surfaces for 3 monitors).
- **Display surface eviction**: New `_decay_display_surfaces()` removes surfaces not
  seen for `DISPLAY_SURFACE_STALE_CYCLES` (8) cycles. Previously, stale surfaces
  accumulated indefinitely.
- **Self-improvement allowed paths**: Added `brain/cognition/` to `ALLOWED_PATHS` in
  `patch_plan.py`.

**Ship 1 — Calibration + Foundation (4 new files):**

- `cognition/spatial_schema.py` — canonical data contracts: `SpatialObservation`,
  `SpatialAnchor`, `SpatialTrack`, `SpatialDelta`, `SpatialRelationFact` plus
  `AUTHORITY_LEVELS` (5-level hierarchy), `CLASS_MOVE_THRESHOLDS` (10 classes),
  `KNOWN_SIZE_PRIORS` (9 objects), all confidence thresholds, memory budgets,
  calibration timing constants
- `perception/calibration.py` — `CalibrationManager` with 3-state degradation
  (valid/stale/invalid), `CameraIntrinsics`, `RoomTransform`, persistence to
  `~/.jarvis/spatial/calibration.json`
- `perception/spatial.py` — `SpatialEstimator` (known-size prior depth estimation,
  track smoothing, per-class jitter suppression, anchor authority conflict detection,
  stale track decay) + `SpatialRecorder`/`SpatialReplayer` for offline validation
- `consciousness/events.py` — 4 new spatial event constants

**Ship 2 — Validation + Dashboard (2 new files, 2 modified):**

- `cognition/spatial_validation.py` — `SpatialValidator` (7-level promotion chain,
  class-specific jitter thresholds, anchor authority enforcement) + `RejectionLedger`
- `cognition/spatial_fusion.py` — merges spatial tracks with scene entities, derives
  spatial relations between entities and anchors
- `dashboard/app.py` — `/api/spatial/diagnostics` endpoint
- `dashboard/snapshot.py` — spatial data wired into snapshot cache

**Ship 3 — Epistemic Integration (3 modified files):**

- `epistemic/calibration/signal_collector.py` — 7 new spatial fields on
  `CalibrationSnapshot`
- `epistemic/calibration/domain_calibrator.py` — 3 new spatial domains
  (`spatial_position`, `spatial_motion`, `spatial_relation`)
- `epistemic/reflective_audit/engine.py` — new `spatial_integrity` audit category

**Ship 4 — Memory-Safe Episode Promotion (1 new file):**

- `cognition/spatial_memory_gate.py` — `SpatialMemoryGate`: CueGate enforcement,
  per-hour (5) and per-day (20) budgets, confidence >= 0.88 threshold,
  human-relevance or repetition requirement

**Live Wiring (4 modified, 1 new):**

- `perception_orchestrator.py` — spatial pipeline runs on every `scene_summary` event:
  estimate → track → validate → fuse → emit. `get_spatial_state()` for dashboard.
- `cognition/world_adapters.py` — `observations_from_spatial_state()` creates canonical
  `SensorObservation` objects from spatial tracks and anchors
- `cognition/world_model.py` — passes spatial fused state to canonical projector,
  surfaces `spatial_active` flag in `get_state()`
- `dashboard/static/renderers.js` — Spatial Intelligence diagnostics panel showing
  calibration state, tracks, anchors, promoted/rejected deltas, rejection reasons

**Tests: 121 new tests across 8 files, all passing:**

- `test_spatial_schema.py` (10), `test_spatial_estimation.py` (22),
  `test_spatial_fusion.py` (9), `test_spatial_validation.py` (10),
  `test_spatial_memory_policy.py` (25), `test_spatial_integration.py` (9),
  `test_scene_tracker.py` (+2 regression)

**Non-negotiable invariants enforced:**

- No spatial module imports `memory` (verified by source-level import tests)
- Raw spatial data never writes to memory directly
- Every promoted spatial claim carries calibration_version, confidence, provenance
- Memory budgets hard-capped (5/hr, 20/day)
- CueGate blocks spatial writes during dreaming/sleep/reflective/deep_learning
- Stable anchors outrank movable objects in all conflicts
- If calibration is unconfigured, boot now auto-seeds Pi camera defaults to `valid`; if defaults are unavailable or consistency checks fail, state remains fail-closed (`invalid`)

---

## Audio Continuity Fix + WiFi Root Cause (2026-03-29)

Multi-day investigation into inter-sentence audio gaps during Jarvis responses.
Root cause was 2.4 GHz WiFi channel congestion delivering only 40 KB/s throughput
(0.04 MB/s) despite 260 Mbit/s link rate and the router being 4 feet from the Pi.

### Software fixes (all still beneficial, deployed)
- **WebSocket write_limit** (`server.py`): Set `write_limit=4MB` on websockets
  `serve()` — default was 32KB, causing `drain()` to block on every audio payload
- **Dedicated audio sender** (`server.py`): `broadcast_audio()` enqueues to priority
  `asyncio.Queue`, `_audio_sender()` drains in tight loop with `[AUDIO-SEND]` logging
  (payload KB, send ms, queue wait ms, queue depth)
- **Batch sync TTS** (`conversation_handler.py`): `_broadcast_chunk_sync` rewritten —
  sentence 1 synthesized alone for fast first audio, remaining sentences batched into
  1-2 chunks (soft cap 500 chars). Reduces WebSocket sends from N to 2.
- **Auto-routing** (`conversation_handler.py`): `_broadcast()` auto-routes messages
  with `audio_b64` through `broadcast_audio()`, non-audio through `broadcast()`
- **Disabled WebSocket compression** (`server.py`, `ws_client.py`): websockets 16.0
  enables per-message-deflate by default — compressing incompressible base64 audio
  wasted CPU and caused 14s send times

### Network fix (the actual root cause)
- Pi was on 2.4 GHz WiFi (channel 8) with 4,167 retry packets — severe congestion
- Switched to 5 GHz (channel 44, 540 Mbit/s): **279x throughput improvement**
  (0.04 MB/s → 10.8 MB/s), 0 retries, 3ms ping
- Full diagnostic and WiFi checklist in `docs/AUDIO_LATENCY_DIAGNOSTIC.md`

---

## Identity Edge Case Hardening Sprint (2026-03-29)

4-wave sprint to improve identity fusion stability, observability, and
multi-speaker awareness. No enrollment, boundary engine, or addressee gate
changes.

### Wave 1 — Low-risk stabilization
- **Voice EMA smoothing** (`speaker_id.py`): Added `SCORE_EMA_ALPHA = 0.35`
  pre-decision EMA to match `face_id.py` pattern. Known/unknown decision now
  uses smoothed score, reducing single-frame volatility. `raw_score` added to
  return dict for distillation transparency.
- **Dashboard identity panel** (`dashboard.js`): Rewrote `_renderIdentityRecognition`
  with per-modality voice/face cards, flip count badge, persistence timer,
  recognition state, cold-start indicator, and speaker/face profile listings.

### Wave 2 — Smart wake-word persistence
- **Conditional wake-word clearing** (`identity_fusion.py`): Wake word no longer
  unconditionally clears persisted identity. If a live face signal matches the
  persisted identity (known, confidence >= soft threshold, not stale), identity
  is preserved across wake events.

### Wave 3 — Threshold assist, multi-speaker, unknown continuity
- **`closest_match` field**: `speaker_id.py` and `face_id.py` now return the
  best-matching profile name regardless of threshold, enabling tentative evidence
  accumulation under real profile names instead of temporary IDs.
- **Tentative bridge**: When recognition state is `tentative_match` with
  accumulated cross-signal evidence, and no strong signals are present, fusion
  produces a conservative `tentative_bridge` resolution (80% of accumulated
  confidence, min 0.35). Blocked by multi-person presence or expired evidence.
- **Multi-speaker suppression**: `set_visible_persons()` receives vision person
  count. Voice-only promotion is suppressed when multiple persons visible and
  voice confidence is below `MULTI_PERSON_VOICE_THRESHOLD` (0.55). Face-only and
  verified_both are unaffected.
- **Voice trust state**: New `voice_trust_state` enum (trusted/tentative/
  degraded/conflicted/unknown) and `trust_reason` string computed in every
  `_resolve()` cycle, exposed in `get_status()`.
- **Resolution basis**: Every resolution records its decision path
  (`resolution_basis`) — voice_face_agree, voice_only, face_only, conflicted,
  persisted, tentative_bridge, suppressed_multi_person, no_signal.
- **Unknown identity continuity**: Sub-threshold voice events are recorded as
  provisional `_unknown_voice_events` (ring buffer, 20 max) with closest profile
  match, face context, and visible person count. Exposed via
  `get_unknown_voice_events()` for curiosity/clarification flows.
- **Dashboard indicators**: Trust state badge (color-coded), multi-person
  indicator, voice suppression alert, threshold assist badge, resolution basis,
  unknown voice count, and trust reason detail line.

### Tests
- `test_speaker_smoothing.py`: 11 tests (EMA pre-decision, dampening, lifecycle clear)
- `test_identity_fusion.py`: 5 new wake-word persistence tests
- `test_identity_wave3.py`: 33 tests covering resolution basis, voice trust
  state, multi-speaker suppression, tentative bridge, unknown voice continuity,
  and closest-match accumulation

---

## Fractal Recall Activation & Route Quality — VALIDATED (2026-03-29)

### Fractal Recall: First Successful Surfacing

**Problem**: Fractal recall engine was shipped (2026-03-27) but never surfaced a single
recall in production. All ticks ended with "no seed above threshold" or silent low-signal
skips. Root causes identified through runtime diagnostic tracing:

1. **Resonance threshold too high** (0.55): With mostly consolidation-origin memories,
   top candidates plateaued at ~0.42 — close but never clearing the bar.
2. **Meta-tag pollution in tag scoring**: Bookkeeping tags like "consolidated",
   "dream_consolidation", "dream_artifact" inflated the tag denominator, driving tag
   overlap scores to 0.00 for most candidates.
3. **Vocabulary mismatch**: The cue builder emitted mode tag "conversational" but stored
   memories used tag "conversation" — zero intersection despite semantic alignment.
4. **Cue signal blindness to face identity**: `_gather_speaker()` only read voice
   identity from `attention_core`. When user was visible on camera but not speaking,
   speaker signal was None, dropping cue strength to ~0.00 and triggering `low_signal`
   skip. This was the critical blocker in passive mode.

**Fixes**:
- `_filter_content_tags()`: strips 7 meta-tag prefixes from tag overlap calculation
- Lowered `RESONANCE_THRESHOLD` 0.55 → 0.40, `CHAIN_CONTINUATION_THRESHOLD` 0.40 → 0.35
- Vocabulary bridge: adds "conversation" tag to cue when mode is "conversational"
- World-state speaker fallback: `_gather_speaker()` falls back to
  `world_state.current_state.user.speaker_name` (face ID) when voice is unavailable
- World-state engagement fallback: `_gather_engagement()` similarly falls back

**Result**: Two `FRACTAL_RECALL_SURFACED` events — the first in Jarvis's entire
operational history. Second recall fired in passive mode with face-only detection,
proving the world-state fallback works. `governance=eligible_for_proactive`,
`confidence=0.72`, `cue=human_present`.

**Files changed**: `brain/memory/fractal_recall.py`

### Synthetic Route Quality: NONE 58% → 3.3%

**Problem**: Synthetic perception exercises were routing 58% of utterances to NONE
(general LLM fallback), starving distillation specialists of labeled training data.

**Root causes**:
1. **STT truncation**: VAD strips first 1-3s of audio after wake word detection,
   losing routing keywords placed at the start of utterances.
2. **Utterance batching**: 3s inter-utterance delay was too short; multiple utterances
   merged into single STT inputs, confusing the router.
3. **Cross-route contamination**: Keywords like "neural network" and "hemisphere" in
   non-INTROSPECTION utterances triggered incorrect priority routing.

**Fixes**:
- VAD buffer strategy: 2-3 word conversational padding ("please tell me", "go ahead
  and") after "hey jarvis" protects routing keywords from truncation
- Increased `route_coverage` delay from 3s to 5s (eliminates batching)
- Removed cross-contaminating keywords from non-target categories
- Surgical per-utterance fixes for contraction mismatches, keyword gaps

**Result (V3 soak, 30 utterances)**: NONE dropped from 58% → 3.3% (1 intentional).
All 13 target routes present. Zero utterance batching. Zero unintended NONE from
routable categories. Zero invariant leak violations.

**Files changed**: `brain/synthetic/exercise.py`

---

## Synthetic Perception Exercise — Proven Growth Lane — SHIPPED (2026-03-28/29)

**Problem**: Perception-side specialists (distillation, routing calibration, policy shadow)
were starved for data — they only activate when real audio survives the full perception
chain (wake word → VAD → STT), and manually talking to Jarvis all day does not scale.

**Solution**: Quarantined synthetic perception exercise lane that generates TTS-synthesized
speech, sends it over the real WebSocket transport as raw PCM audio, and lets the brain
process it through the full perception stack. A hard stop in the perception orchestrator
terminates processing before conversation handler / LLM / memory / identity / TTS.

**Architecture**:
- CLI runner (`brain/scripts/run_synthetic_exercise.py`): Kokoro TTS on CPU, binary
  WebSocket to brain :9100, soak profiles (smoke/route_coverage/idle_soak/stress),
  JSON report generation, reconnect-and-resume for transport resilience
- Utterance corpus (`brain/synthetic/exercise.py`): 16 categories, ~110 templates
  covering all major ToolType routes, weighted category selection, ExerciseStats
  with client/brain-confirmed counters and invariant leak tracking
- Server guard (`brain/perception_orchestrator.py`): `_synthetic_sources: set[str]`
  session-sticky flag via `SYNTHETIC_EXERCISE_STATE` events, `_synthetic_route_only()`
  hard stop, 15-second cooldown window for trailing audio, expanded telemetry ledger
- Server safety net (`brain/perception/server.py`): `_synthetic_sensors` tracking
  with disconnect cleanup in `_handler` finally block
- Dashboard (`brain/dashboard/static/dashboard.js`): route histogram, leak indicators,
  recent route examples
- REST endpoint (`brain/dashboard/app.py`): `GET /api/synthetic-exercise`

**Bugs found and fixed during build** (5 total):
1. Race condition: Pi audio overwrote synthetic flag → fixed with `set[str]` session-sticky sources
2. Last-utterance leak: buffered audio after exercise end → fixed with 8s drain delay
3. Trailing audio after disconnect: wake word fires on buffered synthetic audio → fixed with 15s cooldown window
4. WebSocket keepalive timeout: server event loop blocked → fixed with client-side ping disable + reconnect-and-resume
5. Stuck synthetic flag: disconnect before end message → fixed with server-side cleanup in `_handler` finally block

**Verified results** (6 runs, 273 utterances, 2026-03-28):
- 215 STT-confirmed, 215 hard-stopped, 860 distillation records
- Zero invariant leaks (LLM, TTS, transcription, memory, identity) across all runs
- 1-hour idle soak: 110 sent, 90 confirmed, 10 reconnects recovered, PASS
- Specialist growth: emotion_depth 44%→93.2%, speaker_repr 87%→97.8%, face_repr 92%→99.9%
- 3,315 total distillation records accumulated across all teacher signal files

**Verification**: 39 tests in `test_synthetic_exercise.py` (provenance, corpus, profiles,
stats, speaker/emotion origin, forbidden events, cooldown protection).

Files: `brain/synthetic/exercise.py`, `brain/scripts/run_synthetic_exercise.py`,
`brain/perception_orchestrator.py`, `brain/perception/server.py`,
`brain/consciousness/events.py`, `brain/dashboard/app.py`,
`brain/dashboard/static/dashboard.js`, `brain/tests/test_synthetic_exercise.py`,
`docs/SYNTHETIC_PERCEPTION_EXERCISE.md`.

---

## Fractal Recall Engine — SHIPPED (2026-03-27)

**Problem**: Jarvis had no mechanism for spontaneous associative memory retrieval
during waking operation. Memories only surfaced when explicitly searched or when
the dream cycle ran. The system couldn't "be reminded of something" by ambient
context — a core component of natural cognition.

**Solution**: `FractalRecallEngine` in `brain/memory/fractal_recall.py` — background
associative recall that runs every ~33 seconds during waking modes. Builds ambient
cues from current context (user presence, conversation topic, scene entities),
probes memory via semantic + tag + temporal multi-path search, scores candidates
with 8-term resonance scoring (semantic similarity, tag overlap, provenance fitness,
mode fitness, recency, association density, weight, identity sensitivity), walks
associative chains with anti-drift controls, and emits `FRACTAL_RECALL_SURFACED`
events for downstream consumption.

**Key design decisions**:
- Seed threshold 0.55 prevents weak recalls from surfacing noise
- Chain walker stops on repeated IDs, blocked provenance, weak hops, or topic drift
- Identity-sensitive chains routed to reflective-only disposition
- 4-action governance: proactive, reflective_only, epistemic_review, ignore
- Codebase-study memories capped at 10% of surfaced recalls during companion mode
- Never speaks directly (no TTS), never creates memories in Phase 1

**Wiring**: Registered as `fractal_recall` cycle in `modes.py` (waking modes only).
Tick in `consciousness_system.py` with lazy init + mode gate. Dashboard snapshot
with all telemetry fields. Dashboard panel shows top candidates, cue class, and
recall history.

**Verification**: 24 tests in `test_fractal_recall.py` (cue building, provenance
filtering, resonance scoring, seed selection, chain walking, governance decisions,
rate limiting, event payload structure).

Files: `brain/memory/fractal_recall.py`, `brain/consciousness/events.py`,
`brain/consciousness/modes.py`, `brain/consciousness/consciousness_system.py`,
`brain/dashboard/snapshot.py`, `brain/dashboard/static/dashboard.js`,
`brain/tests/test_fractal_recall.py`.

---

## Personality Seeding from Soul + Dashboard — SHIPPED (2026-03-27)

**Problem**: Personality traits started empty at boot and only evolved from interaction
evidence, leaving a cold-start gap where introspection reported no personality data.

**Solution**: Traits now seed from `identity.json` (soul state) on boot. Dashboard
personality panel shows trait values, evolution history, and calibrator recommendations.
New `/api/personality` endpoint exposes trait state.

Files: `brain/personality/traits.py`, `brain/personality/evolution.py`,
`brain/dashboard/app.py`, `brain/dashboard/snapshot.py`,
`brain/dashboard/static/dashboard.js`.

---

## Brain Reset — Full State Capture + Clean Start (2026-03-27)

Complete brain reset performed after capturing the full pre-reset state in
`docs/PRE_RESET_SNAPSHOT_2026_03_27.md`. Documented: Oracle 92.6 Gold,
Soul Integrity 0.854, 1224 memories, 921 beliefs, 2121 belief edges, 730 mutations,
all discovered bugs, and architectural lessons. Blue Diamonds archive preserved
across reset. New brain successfully reached Integrative stage, rebuilt to 2000
memories, and accumulated 168 mutations within 18 hours.

---

## Maturity Gate Awareness + Audit Docs — SHIPPED (2026-03-27)

Updated `AGENTS.md` with comprehensive maturity gate awareness section: "Before
Reporting a Bug" checklist, common false positives table, and key timing gates table.
Updated `docs/JARVIS_UNIFIED_AUDIT_PROMPT.md` with matching guidance. Prevents false
bug reports from confusing maturity-gated zero values with actual failures.

---

## CueGate — Memory Access Policy Authority — SHIPPED (2026-03-26)

**Problem**: Dream containment relied on scattered stance checks across 5+ files.
The observer checked `_stance_profile.allow_memory_write_effects`, consciousness_system
checked `_mm.mode != "dreaming"` for artifact validation, and the old `MemoryGate` was
observability-only (tracked opens/closes but never blocked anything). No single authority
decided whether a given memory operation was permitted.

**Fix**: Replaced `MemoryGate` in `brain/memory/gate.py` with a policy-enforcing CueGate
that distinguishes three access classes:

| Access Class | What | When Allowed |
|---|---|---|
| READ | Memory retrieval (search, recall) | Always (via `session()` RAII) |
| OBSERVATION_WRITE | Incidental writes from observer delta effects | Waking modes only |
| CONSOLIDATION_WRITE | Intentional dream cycle writes | Only during active dream consolidation window |

**Wiring**: Observer (`consciousness/observer.py`) now queries `memory_gate.can_observation_write()`
instead of local stance profile booleans (3 sites replaced). Dream cycle
(`consciousness_system.py`) wraps all memory operations with `begin_consolidation()`/
`end_consolidation()` in a `finally` block. Artifact validation uses
`not memory_gate.can_consolidation_write()` instead of direct mode check. Mode changes
(`perception_orchestrator.py`) call `memory_gate.set_mode()`.

**Dashboard**: Dedicated CueGate panel on Memory tab shows mode (color-coded), observation
write state, consolidation state, read session depth, timestamps, and scrollable transition
timeline with color-coded entries.

**What was NOT changed**: Tag-based content filters (`_DREAM_ORIGIN_TAGS`, `_DREAM_INELIGIBLE_TAGS`),
scheduling logic, policy StateEncoder dimensions (deferred to avoid breaking 242 experiences).

**Verification**: 35 tests in `test_memory_gate.py` (defaults, RAII sessions, mode policy for
all 8 modes, consolidation lifecycle, transition history bounds, thread safety, observer
integration patterns). 2155 broader tests pass with zero regressions. Live verified:
consolidation window fires during dream cycles, observation writes blocked in reflective mode.

Files: `brain/memory/gate.py`, `brain/consciousness/observer.py`,
`brain/consciousness/consciousness_system.py`, `brain/perception_orchestrator.py`,
`brain/dashboard/static/dashboard.js`, `brain/tests/test_memory_gate.py`.

---

## Memory Field Corruption Fix — SHIPPED (2026-03-26)

**Problem**: 7 sites across 4 files reconstructed `Memory` objects using explicit field
lists that omitted `access_count` and `last_accessed`, silently zeroing them. Every dream
cycle, every association, and every observer delta effect destroyed access tracking data
used for retention scoring.

**Fix**: All reconstruction sites now use `Memory(**{**asdict(m), "field": new_value})`
pattern to preserve all existing fields while updating specific ones:
- `storage.py`: provenance rewrite, weight cap, `associate()` (both memories)
- `observer.py`: `_apply_delta_effects()` salience and association_weight (2 sites)
- `consciousness_system.py`: dream Phase 3 accelerated decay
- `maintenance.py`: `clean_orphaned_associations()`

---

## Belief Graph Orphan Fill Fix — SHIPPED (2026-03-26)

**Problem**: `fill_orphan_edges()` in `epistemic/belief_graph/bridge.py` used
`evidence_basis="orphan_fill"` but this string was not in the `VALID_EVIDENCE_BASES`
frozenset in `edges.py`. All orphan-filling edge creation was silently rejected.

**Fix**: Added `"orphan_fill"` to `VALID_EVIDENCE_BASES`. Live verified: new edges
being created during dream cycles.

---

## Drive Dampening Recovery Fix — SHIPPED (2026-03-26)

**Problem**: Three compounding issues prevented dampened drives from recovering:
1. `record_outcome(True)` only decremented `consecutive_failures` by 1 — from 8
   failures, 6 consecutive successes needed to exit dampening
2. Urgency floor only applied when `last_acted > 0` — after restart, dampened drives
   got 0 urgency with no rescue
3. Mastery "experiment" fallback structurally couldn't move the health metrics measured
   by `_process_delta_outcome()`

**Fix**: Success now decrements `consecutive_failures` by 3 (recovery in 2-3 wins).
Urgency floor applies unconditionally to dampened drives regardless of `last_acted`.
Live verified: mastery drive showing urgency and acting.

File: `brain/autonomy/drives.py`.

---

## Brier Score Integrity Fix — SHIPPED (2026-03-26)

**Problem**: The Brier score formula was mathematically correct (textbook
`Σ(confidence - actual)² / N`), but the inputs feeding it were wrong in 4 ways:
1. Bridge 5 (positive outcomes) treated any speech within 4s of Jarvis finishing
   as `correct=True` — "user continued talking" ≠ "system was correct"
2. All bridges used hardcoded per-route confidence lookup tables instead of
   the system's actual stated confidence at response time
3. Bridge 2 (world model) set confidence to rolling hit-rate after warmup,
   guaranteeing the Brier score looked good by construction
4. Bridge 3 (attribution) recorded every non-empty reply as `success` at
   hardcoded confidence 0.9, regardless of user feedback

**Fix**: Four changes to make calibration honest:

| Change | File(s) | Effect |
|--------|---------|--------|
| Bridge 5: only fire on explicit positive signals | `calibration/__init__.py`, `conversation_handler.py` | Bare follow-ups no longer recorded as `correct=True` |
| All bridges: accept real `response_confidence` | `calibration/__init__.py`, `conversation_handler.py` | Uses `_language_example_seed["confidence"]` (dynamic per-route, sometimes from actual `frame_confidence`) |
| Bridge 2: use CausalRule `prediction_confidence` | `calibration/__init__.py`, `cognition/world_model.py` | Actual rule confidence propagated through event |
| Bridge 3: require user signal for conversation correctness | `calibration/__init__.py`, `consciousness/attribution_ledger.py` | `positive` → correct, `correction`/`negative` → incorrect, no signal → skip |
| Store previous response confidence | `perception_orchestrator.py`, `conversation_handler.py` | Correction bridge can use prior response's real confidence |

**Verification**: 70 calibration tests pass, 311 broader tests pass, 0 linter errors.
Boot verified clean on desktop — no errors from modified bridges.

---

## Dream Processing Dashboard Panel — SHIPPED (2026-03-26)

**Problem**: The dashboard had no visibility into what happens during dreaming.
Dream artifacts, cycle history, validation funnel, and cluster data existed
in the backend but were never exposed.

**Fix**: Added comprehensive "Dream Processing" panel to the Memory tab:

| Component | What it shows |
|-----------|-------------|
| Header bar | Observer stance, dream cycle count, buffer/validator stats |
| Validation funnel | Created → Pending → Promoted/Held/Discarded/Quarantined flow |
| Cycle history log | Per-cycle summaries with expandable details (phases, artifacts, durations) |
| Artifact inspector | Individual artifacts with type, state, content, confidence, coherence |
| Cluster summary | Memory cluster cards from dream consolidation |

Backend: Added `get_dream_recent_artifacts()`, `get_dream_cycle_history()`,
dream cycle tracking in `_run_dream_cycle()`, observer stance in snapshot.

Files: `brain/consciousness/consciousness_system.py`, `brain/dashboard/snapshot.py`,
`brain/dashboard/static/dashboard.js`, `brain/dashboard/static/style.css`.

---

## Audit 15 — Full System Truth Operator (2026-03-26)

Deep audit of source code, runtime behavior, learning systems, persistence, telemetry.
5 critical/high problems found and fixed:

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| P1 | CRITICAL | Memory storage emits events while holding lock (deadlock) | Collect-under-lock / cleanup-after-release pattern |
| P2 | HIGH | Identity fusion emits under lock (violates own invariant) | `_pending_emit` field, emit after lock release |
| P3 | HIGH | `_direct_memory_write()` bypasses identity stamping | Added `_stamp_identity_fallback()` |
| P4 | MEDIUM | Policy NN `StateEncoder.encode()` failures logged at DEBUG | Changed to WARNING with exc_info |
| P5 | LOW | Hemisphere registry shows 0 networks on restore | Added warning log for empty `topology_json` + boot summary |

Files: `brain/memory/storage.py`, `brain/perception/identity_fusion.py`,
`brain/memory/core.py`, `brain/consciousness/engine.py`,
`brain/hemisphere/orchestrator.py`, `brain/hemisphere/registry.py`.

---

## Skill Routing Fix — Ship A (2026-03-26)

**Problem**: Telling Jarvis "learn a job" fell through to ToolType.NONE, where the
LLM narrated a nonexistent system action ("I've created a learning job...").

**Fix**: Three-layer safety patch:

| Layer | File(s) | What |
|-------|---------|------|
| L1: Regex patch | `tool_router.py` | Unified `_JARVIS_PREFIX`, expanded Tier 0/1/2 patterns for learn/train |
| L2: Confabulation guard | `capability_gate.py`, `conversation_handler.py` | `_SYSTEM_ACTION_NARRATION_RE` + latching `_narration_blocked` on NONE route |
| L3: Honest failure UX | `skill_tool.py` | Generic fallback lists available skills instead of dead-end message |

Tests: New cases in `test_tool_router.py` and `test_capability_gate.py`.

---

## Dashboard Hardening Batch (2026-03-26)

- **Flight Recorder panel** restored on Activity tab (above World & Perception)
- **PROVISIONAL maturity banner** removed from dashboard header
- **Singing template** removed from `skills/resolver.py` (infeasible with current architecture)

---

## Training Protocol Tiers Document (2026-03-26)

Created `docs/plans/training_protocol_tiers.md` — reference document covering
5 training protocol tiers (Distillation, Supervised, Fine-Tuning, Reinforcement,
Generative Model), feasibility gate design, template audit table, and implementation
roadmap. Custom vision (deer tracking) documented as future Tier 4 feature.

---

## Reflective Conversation Mode — SHIPPED (2026-03-25, routing fix 2026-03-25)

**Problem**: Philosophical and personal self-questions ("what type of curiosity do
you have?") were routed through the strict operational INTROSPECTION pipeline, which
uses a stripped prompt ("you are the mouth, not the brain"), forced temp 0.35,
grounding check against exact metric citations, and fail-closed fallback. These
questions produced data dumps or "I don't have data on that" responses instead of
genuine self-expression.

**Fix**: Added a `reflective` flag to the INTROSPECTION route. When a question
invites personal expression (not operational data), the pipeline uses full soul
prompt + personality, trait-adjusted temperature, introspection data as interpretive
substrate, and CapabilityGate (Layer 0) enforcement. No grounding check, no bounded
bypass, no fail-closed replacement.

| Component | File | What |
|-----------|------|------|
| Signal detection | `tool_router.py` | 18 strong + 1 guarded regex, `_finalize()` hook |
| Operational veto | `tool_router.py` | 25+ metric/operational words block reflective flag |
| Prompt path | `context.py` | `[Reflective self-context` marker → full soul prompt |
| Reflective honesty rule | `context.py` | "Interpret observed tendencies; don't invent metrics" |
| Handler branch | `conversation_handler.py` | After Tiers 1-2, before Tier 3 |
| Response params | `response.py` | `reflective_introspection` hint: trait-adjusted temp, 1024 tokens |
| Tests | `test_reflective_introspection.py` | 71 tests: signals, existential, veto, e2e routing, STT damage |

**Governing constraints**: STATUS route unchanged. Strict INTROSPECTION Tiers 1-4
unchanged. CapabilityGate active on all paths. Mixed reflective + metric queries
stay operational (veto wins). One-line rollback available.

**Routing fix (same day)**: Live testing with 3 questions revealed two bugs:
1. Tier 2 regex windows too narrow — `.{0,10}` couldn't span "kind of curiosity"
   (18 chars). Questions fell through to NONE instead of INTROSPECTION.
2. `_detect_reflective()` lacked existential patterns — "what do you think you are
   becoming" reached INTROSPECTION but fell to bounded-native path instead of reflective.
3. `CapabilityGate.sanitize()` method didn't exist — `check_text()` is the correct
   method. Reflective responses crashed with AttributeError after LLM generation.

Fix: widened Tier 2 regex windows (`.{0,25}`), added 5 new Tier 2 introspection
patterns, 6 new `_REFLECTIVE_STRONG` patterns for existential/identity questions,
expanded `_SELF_REF_VERBS` and `_STRONG_SELF_FRAME`, STT-damage pattern gated by
Jarvis prefix, fixed `sanitize` → `check_text`. Verified live: all 3 test questions
("What kind of curiosity drives you?", "How do you experience learning?", "What do
you think you are becoming?") now route to INTROSPECTION, detect as reflective,
take the reflective path, and produce personality-driven TTS responses.

---

## Phase 6.3: Overconfidence Control + Calibration Loop Closure — SHIPPED (2026-03-25)

**Problem**: Phase 6.1 proved the measurement layer (Brier, ECE, overconfidence live).
But calibration was passive — a report card, not a control signal. No existing decision
system consumed calibration outputs. The loop was open.

**Fix**: Wired calibration into 4 decision systems without introducing new subsystems:

| Change | File(s) | Effect |
|--------|---------|--------|
| Confidence domain weight 0.03 → 0.10 (salience 0.10 → 0.03) | `truth_score.py` | Truth score now materially reflects calibration quality |
| Capped OC penalty in policy health reward | `engine.py` | `reward -= 0.15 * min(oc, 0.25)` — policy learns from calibration |
| Correction penalty cascade (route → global → skip) | `engine.py`, `conversation_handler.py` | Wrong-and-confident answers penalized proportionally |
| Belief adjuster threshold 0.15 → 0.05 | `belief_adjuster.py` | Dream-cycle adjustments activate earlier (MAX_DELTA=0.05 intact) |
| Retrieval provenance blend | `events.py`, `confidence_calibrator.py` | 50% static + 50% dynamic accuracy (sample-guarded, min 20) |
| State encoder dim 4 blend | `state_encoder.py`, `promotion.py` | 0.7 × analytics + 0.3 × truth_score, ENCODER_VERSION=2 |

**Governing constraint**: No new subsystems, no duplicate signal paths, no second
confidence tracker. All changes consume existing `TruthCalibrationEngine` outputs.

**Verification**: All tests pass. Weight sum verified at 1.00. Backward-compatible
function signatures (new params have defaults). Provenance blend cached (30s TTL)
to avoid per-memory overhead during bulk retrieval.

---

## Phase 6.1: Confidence Outcome Pipeline — SHIPPED (2026-03-25)

**Problem**: The `ConfidenceCalibrator` was fully built (Brier, ECE, overconfidence,
underconfidence, per-route scoring, JSONL persistence, dream-cycle belief damping)
but starved of data. Only user corrections (`correct=False`) ever entered the pipeline.
All existing outcome events (`PREDICTION_VALIDATED`, `WORLD_MODEL_PREDICTION_VALIDATED`,
`OUTCOME_RESOLVED`) were emitted into the void — no subscribers. The dashboard showed
"--" for all calibration stats. This was the same failure class as policy persistence
(Audit 16) and source ledger (knowledge integrator NameError): infrastructure existed
end-to-end but a bridge gap prevented any durable output.

**Fix**: Wired 4 outcome bridges into `TruthCalibrationEngine`:

| Bridge | Source | Method |
|--------|--------|--------|
| 1 | `PredictionValidator.tick()` validated predictions | Direct call in `on_tick()` |
| 2 | `WORLD_MODEL_PREDICTION_VALIDATED` event | EventBus subscription |
| 3 | `OUTCOME_RESOLVED` attribution ledger event | EventBus subscription |
| 5 | Positive conversational signals (thanks, follow-up) | `record_positive_response_outcome()` called from `conversation_handler.py` |

Bridge 4 (`ATTRIBUTION_ENTRY_RECORDED`) was evaluated and correctly cancelled — it fires
at entry creation (no outcome signal), not resolution. `OUTCOME_RESOLVED` (Bridge 3)
captures the meaningful signal.

Also fixed: `brier_score` and `ece` were computed on internal `CalibrationSnapshot` but
never exposed in `TruthCalibrationEngine.get_state()`. Dashboard API showed `null` even
after 20+ outcomes were recorded.

Added: 600s watchdog (warns on zero-artifact bridges, one-sided outcomes, persistence
failures). Bridge counters exposed in `get_state()["outcome_bridges"]`. Dashboard
empty-state message now shows outcome count + bridge total.

**Impact**: Truth score 0.64→0.6565. Maturity 0.625→0.75. Confidence domain 0.5→0.8074
(non-provisional). 85 outcomes across 3 route classes. Brier 0.1841, ECE 0.0254,
overconfidence 0.0, underconfidence 0.1295. System character: slightly underconfident
(conservative), which is the safe direction.

**Same failure class as Audit 16 + Source Ledger**: fourth confirmed instance of
"infrastructure exists, pipeline wired, single bridge gap prevents durable output."

Files: `brain/epistemic/calibration/__init__.py`, `brain/conversation_handler.py`,
`brain/dashboard/static/eval.js`.

---

## Policy Training Loop Proven (2026-03-25)

First full closed-loop policy training cycle observed after Audit 16 persistence fix.

- 3 training runs completed
- v4 (mlp2) promoted to active model
- 96.6% win rate in shadow A/B evaluation
- Partial mode enabled with 2 features (budget_allocation, task_scheduling)
- No regressions in health observability
- Policy pipeline now proven: collect → train → register → evaluate → promote → deploy

This confirms the system produces and promotes learned intelligence autonomously.

---

## Dashboard UI Fixes (2026-03-25)

- Oracle Benchmark domain score card bars: changed background from `var(--bg)` (invisible
  against card) to `rgba(0,0,0,0.45)`, increased height from 8px to 10px
- Added per-subcriteria mini-bars showing individual progress within each domain card
- Truth Calibration empty-state message: now shows `confidence_outcome_count` and
  `outcome_bridges.total` instead of generic "Accumulating..."

Files: `brain/dashboard/static/eval.js`.

---

## Source Ledger Pipeline Fix — Knowledge Integrator NameError (2026-03-25)

**Root cause**: `_store_findings()` in `brain/autonomy/knowledge_integrator.py` iterated
with `for finding in best_findings:` but passed `finding_idx=idx` to `_build_pointer_payload`.
`idx` was never defined — `NameError` on every research integration that passed relevance/depth
gates. The exception was caught by the outer `except Exception` in `_execute_and_integrate()`
in the orchestrator, silently marking research as "failed."

**Impact**: Across 130 completed research episodes, **zero** pointer memories were ever created
from research findings, and **zero** source ledger entries were written. Sprint 5.1b
(Knowledge-Quality Attribution) was functionally dead despite code being fully wired.
Interventions (Sprint 5.2) could not be scored against source usefulness because there
were no sources tracked.

**Fix**: Changed `for finding in best_findings:` to `for idx, finding in enumerate(best_findings):`.
Also removed duplicate `@staticmethod` decorator on `_build_pointer_payload`.

**Same failure class as Audit 16**: code existed, pipeline was wired end-to-end, but a
single bug in the write path prevented any durable output.

Files: `brain/autonomy/knowledge_integrator.py`.

---

## Phase 5: Continuous Improvement Loop — Code-Complete (2026-03-25)

All four sprints shipped and wired into live runtime. Runtime validation in progress.

**Sprint 5.1a: Conversation Friction Mining** — `brain/autonomy/friction_miner.py` (391 lines).
12 friction types, severity model (low/medium/high/critical), hooked into
`conversation_handler.py` signal detection, feeds `friction_rate` as 8th metric
trigger dimension. 4 friction events persisted on disk. Append-only JSONL persistence
at `~/.jarvis/friction_events.jsonl`.

**Sprint 5.1b: Knowledge-Quality Attribution** — `brain/autonomy/source_ledger.py` (280 lines).
Per-source usefulness tracking with 6 verdict types, provisional + final verdicts,
wired to policy memory. **Pipeline verification pending**: `source_ledger.jsonl` empty
on disk despite code wiring — same failure class as Audit 16 policy persistence bug.

**Sprint 5.2: Deficit-to-Intervention Bridge** — `brain/autonomy/interventions.py` (109 lines) +
`brain/autonomy/intervention_runner.py` (297 lines). Controlled enums for type + subsystem,
shadow queue lifecycle (propose/activate/promote/discard), backlog limits. 17 interventions
persisted, all `no_action` (correct conservative behavior in early phase).

**Sprint 5.3: Eval Harness Activation** — `compare_policies()` scheduled in dream/reflective
cycles (every 3600s). Dashboard A/B panel. L2 self-improve bridge stub (gated by
`FREEZE_AUTO_IMPROVE=true`).

**Validation outstanding**: No promoted intervention yet. Source ledger pipeline gap.
Phase 5 success criteria (demonstrated improvement chain) not yet met.

---

## Audit 15 P3 — MemoryIndex Thread Safety (2026-03-25)

`brain/memory/index.py` used `defaultdict` without locking. Concurrent `add_memory()`
+ `search()` from multiple threads could corrupt tag indices (silent data loss).

**Fix**: Added `threading.Lock()` to all 7 public methods. `rebuild()` inlines
`add_memory` logic under a single lock acquisition to keep the clear-and-repopulate
atomic (prevents readers from seeing a partial index during rebuild).

---

## Audit 16 — Policy Experience Pipeline (2026-03-25)

The policy NN had never trained because experiences never reached disk. Root cause:
`ExperienceBuffer` only flushed to `policy_experience.jsonl` when the buffer hit
`FLUSH_INTERVAL` (was 100), and there was no flush on shutdown. Every restart lost
all in-memory experiences.

**Fix**:
- Added `flush()` call in `_shutdown()` path
- Added periodic flush in `_policy_tick()` (every 60s)
- Lowered `FLUSH_INTERVAL` from 100 to 1 (flush on every add)
- Lowered `MIN_EXPERIENCES` training threshold from 100 to 25

**Result**: 29 experiences persisted across restart (first-ever). Training imminent
at threshold = 30.

---

## Audit 15 P2 — Fail-Closed Health Observability (2026-03-25)

Dashboard health was phantom-perfect: 3 of 5 metrics (backlog, event error rate,
personality coherence) were never wired, defaulting to "perfect." This created a
false-positive integrity signal consumed by Oracle, soul integrity, policy governor,
and mutation governor.

**Fix** (trust boundary enforcement, not UI patch):
- Wired kernel deferred backlog → `analytics.update_backlog()` via `consciousness_system.on_tick()`
- Wired `event_bus.get_metrics().error_rate` → `analytics.record_event_error_rate()`
- Wired `trait_validator.coherence_score` → `analytics.record_personality_coherence()` in `engine._validated_traits()`
- Added `MetricReading` dataclass with `source`/`updated_at` tracking
- Added `confidence` (fraction of strictly live metrics) + `provenance` (per-metric source/age)
- Fail-closed: `confidence < 0.6` forces status to `"degraded"`
- Liveness watchdog: emits `HealthAlert` for metrics with no live data after `BOOT_GRACE_S`
- Boot grace: metrics contribute to `overall` during grace but do NOT count as `"live"` for `confidence`
- Oracle discounts health subcriterion by confidence; new "Observability confidence" subcriterion
- Dashboard surfaces confidence + provenance footnotes
- 5 regression tests in `test_health_observability.py`

**Result**: personality_health = 0.617 (was phantom 1.0). Confidence = 1.0 only
when all 5 inputs are live. Oracle observability = HIGH (100%).

---

## Audit 13 — Architecture Hardening (2026-03-23)

10 bugs fixed across hemisphere training, policy NN, consciousness analytics,
memory transactions, conversation handler, and philosophical reasoning.

**Critical fixes:**
- `await await` syntax errors in `conversation_handler.py` (29 instances)
- Hemisphere `input_dim` vs `input_size` field name in `orchestrator.py`
- `speaker_diarize` training: raw embedding labels → speaker classification with KL-div
- `voice_intent` data starvation: `min_samples` 50 → 15
- Processing health formula duplicate in `consciousness_analytics.py`
- Policy NN reward scoring: threshold tuning + diversity bonus (0% → 98.5% win rate)

**Other fixes:**
- Philosophical reasoning template enrichment (quality 0.07 → 0.54)
- Trait stability sensitivity reduction (variance amplifier 10 → 5)
- Hemisphere research priors crash on dict payloads
- Memory transaction rollback now rebuilds vector store

**Result:** First learning job (`speaker_identification_v1`) completed end-to-end.
Oracle Benchmark post-restart: 86.9 Silver (rebuilding toward Gold).

---

## Phantom Disagreement Fix (2026-03-22)

The dashboard reported "User present but no person detected" as an epistemic
disagreement, but the scene tracker architecturally excludes person detections
(handled by a separate pipeline). The `visible_persons` field never existed in
scene tracker output, so the check always fired when the user was present.
The world model's `PhysicalState.person_count` suffered the same structural bug.

**Dashboard** (`brain/dashboard/static/dashboard.js`):
- Null-guarded `visible_persons` at 4 sites: now only triggers when
  `scene.visible_persons != null && scene.visible_persons === 0`, not when undefined

**World Model** (`brain/cognition/world_model.py`):
- `_read_physical()`: `person_count` now derived from canonical presence tracker
  (`1 if is_present else 0`) instead of non-existent scene tracker field
- Narrative builder: distinguishes "visual" (face detection) from
  "presence (no visual)" (presence only) instead of binary person_count check
- Reactivated previously dead causal rule that depended on `person_count`

---

## Conversation Continuity + Identity Hardening (2026-03-22)

Multi-fix batch addressing conversation context loss, false identity enrollment,
and missing voice command routing. Shipped together as a cohesive set.

**Identity enrollment hardening**:
- `_ENROLL_NAME_WEAK_RE` now requires uppercase first letter (removed `re.IGNORECASE`)
- `name_validator.py`: added 28+ common English words/states to `_BLOCKED_WORDS`
  ("new", "back", "ready", "tired", etc.) to prevent false identity creation
- Enrollment-time dedup: cosine similarity >0.45 with existing profile logs warning

**Identity reconciliation**:
- Voice commands: "merge X into Y", "X is actually Y", "forget X" route to
  `ToolType.IDENTITY` via new keywords/regex in `tool_router.py`
- Cross-subsystem merge: `speaker_id.merge_into()`, `face_id.merge_into()`,
  `evidence_accumulator.merge_candidate()`, memory re-tagging, identity_fusion
  state update — all orchestrated by `perception_orchestrator.reconcile_identity()`
- Alias tombstone: reconciliation events persisted to `~/.jarvis/identity_aliases.jsonl`

**SKILL routing expansion**:
- "train", "start training", "begin training", "resume training" phrases added
  to Tier 1 keywords and Tier 2 regex in `tool_router.py`
- `_SELF_REF_VERBS` expanded with "train"

**Follow-up overrides**:
- Added enrollment offer and guided-collect offer follow-up handlers in
  `conversation_handler.py`, mirroring the existing camera control pattern
- Prevents context loss on single-word confirmations ("yes", "sure", "do it")

**Context injection**:
- Narrowly-gated episodic context (last 4 turns) injected into NONE route only
  when: `follow_up == True`, affirmative follow-up detected, or anaphora detected
  ("that", "it", "do it", "last thing")

**Flight recorder persistence**:
- `collections.deque(maxlen=50)` now persisted to `~/.jarvis/flight_recorder.json`
  with atomic write (tmp + replace), restored on boot

Files: `brain/conversation_handler.py`, `brain/identity/name_validator.py`,
`brain/perception/speaker_id.py`, `brain/perception/face_id.py`,
`brain/identity/evidence_accumulator.py`, `brain/perception_orchestrator.py`,
`brain/reasoning/tool_router.py`.

---

## Language Substrate Phase B: Bounded Articulation (2026-03-21)

Native response planner + bounded articulation layer for 7 response classes,
bypassing Ollama for fact-rich self-knowledge queries.

**Core**:
- `MeaningFrame` dataclass with `frame_confidence`, `fact_count`, `section_count`,
  `parse_warnings`, `is_structurally_healthy` property
- 7 dedicated articulators: `self_status`, `self_introspection`, `recent_learning`/
  `recent_research`, `identity_answer`, `memory_recall`, `capability_status`,
  `system_explanation`
- Output caps enforced: 8 sentences / 600 chars / 5 facts max
- Anti-confabulation: articulators only speak from `lead` + `facts` + controlled
  metadata transforms; no inferred philosophy or hedging filler

**Introspection pipeline upgrade**:
- `self_introspection` ranked fact surface: parses `===`-delimited sections,
  categorizes into current_state > health > memory > evolution > mutation,
  selects top facts by priority
- Pre-LLM bounded path in `conversation_handler.py`: activates when facts >= 15,
  relevant self-state/identity/memory/health/learning topics matched,
  frame confidence >= 0.6, no parse warnings — bypasses Ollama entirely
- Deterministic fail-closed replaces raw data dump (no LLM, no stylistic expansion)

**Live proof**: "How does your memory system work?" hit bounded path with
facts=31, confidence=1.00, zero Ollama involvement.

Files: `brain/reasoning/bounded_response.py`, `brain/conversation_handler.py`.
63 tests in `brain/tests/test_bounded_response.py`.

---

## TODO Execution Plan Completion (2026-03-21)

Full execution of the master TODO plan from the mature baseline. Implementation
status: complete. Validation: live-validated across multiple restart-bounded
sessions. Evidence window started for continuous-runtime maturity accumulation.

**World Model Coherence (5.49 -> 8.84/10)**:
- Canonical ontology expansion: `WorldPosition`, `WorldZone`, `ArchetypePack`,
  `ArchetypeRegistry`, `CanonicalWorldState`, `CanonicalWorldProjector`
- 5 archetype packs defined (`indoor_workspace` active), zone projection from
  scene regions, entity/relation kind expansion
- 4 new causal rules: `stable_scene_persists`, `display_zone_mode_stable`,
  `workspace_person_stays`, `multi_entity_scene_stable`
- Steady-state horizons reduced (60s -> 45s), `MIN_SHADOW_HOURS` 24 -> 4
- Promotion reached Level 2 (active), 950+ validated predictions, 92% rolling accuracy
- Dashboard: `/api/world-model/diagnostics` endpoint, canonical_world in snapshot
- New file: `brain/cognition/world_archetypes.py`
- Modified: `brain/cognition/world_schema.py`, `brain/cognition/world_adapters.py`,
  `brain/cognition/world_model.py`, `brain/cognition/causal_engine.py`,
  `brain/cognition/promotion.py`, `brain/cognition/__init__.py`, `brain/dashboard/app.py`
- 54 tests in `test_world_schema.py`

**Language Substrate Phase A (corpus capture)**:
- Corpus accumulating at `~/.jarvis/language_corpus/examples.jsonl`
- 30 examples, 100% provenance, 7 negative examples (capability gate rewrites)
- Dashboard: `/api/language-corpus/stats` endpoint
- Native usage rate: 78.3%

**Ingestion Quality Self-Diagnosis**:
- Wired into Truth Calibration Layer 6
- Added `ingestion_health` as Reflective Audit dimension
- Surfaced in status and introspection responses

**Stale Goal Pruning**:
- `prune_stale()` in GoalManager tick: auto-abandon goals stale >2h with progress <0.5

**Bug Fixes**:
- `PerceptualAssessExecutor`: now evaluates gates from exit conditions, not just hard gates
- `ConsciousnessEngine._compute_health_reward()`: fixed `_analytics` AttributeError
- Dashboard `/api/world-model/diagnostics`: fixed `_consciousness_system` NameError
- Onboarding: upgraded error logging from debug to warning, added auto-start INFO logs
- Cancelled stuck `speaker_identification_v1` job, reset skill registry entry

**Housekeeping**:
- `brain/.env.example` expanded from ~20 to 55 documented environment variables

Evidence log: `docs/evidence_accumulation_log.md`

---

## Oracle Benchmark v1 + Stage Rename (2026-03-21)

Evolution stage rename and Oracle Benchmark v1 scoring engine.

**Stage rename**:
- `cosmic_consciousness` → `integrative`, `transcendent` → `recursive_self_modeling`
- `_LEGACY_STAGE_MAP` handles backward-compatible state migration on restart
- Stage history normalization remaps legacy entries in persisted `stage_history`
- Defensive normalization in both backend (`dashboard_adapter.py`) and frontend (`dashboard.js`, `particles.js`)
- 3 new legacy migration tests in `test_restart_integrity.py`

**Oracle Benchmark v1** (`brain/jarvis_eval/oracle_benchmark.py`):
- Pure-read-only 7-domain scorer: restart integrity (20), epistemic integrity (20),
  memory continuity (15), operational maturity (15), autonomy attribution (10),
  world model coherence (10), learning adaptation (10) — total 100 points
- Benchmark rank ladder separate from evolution stage: dormant_construct → awakened_monitor
  → witness_intelligence → archivist_mind → oracle_adept → oracle_ascendant
- Seal levels with domain floors: Gold (>=90), Silver (>=80), Bronze (>=70)
- Hard-fail gates: evidence sufficiency, restore trust, runtime sample, epistemic evidence
- Evidence provenance: live-proven, test-proven, unexercised
- API: `GET /api/eval/benchmark`
- Dashboard: Oracle Benchmark tab on eval page with domain cards, hard-fail table,
  evidence provenance, JSON/Markdown export buttons
- First credible run captured as evidence artifact: `docs/oracle_benchmark_v1_first_credible_run.{json,md}`

**Restart integrity hardening** (prerequisite for benchmark):
- `get_restore_trust()` in `consciousness_evolution.py` provides benchmark-facing trust fields
- Stage restore validation, contradiction debt persistence, calibration rehydration,
  delta tracker, mutation timestamps, drive state, gap detector EMAs all verified on restart
- Evidence pack: `docs/Oracle_Restart_Hardening_Evidence_v1.md`

Files: 20 changed, ~1320 insertions. 3 new docs. All existing tests pass.

---

## Audit 12 — Data-Layer Inspection (2026-03-17)

Live inspection of all ~/.jarvis/ data files and ~/.jarvis_blue_diamonds/ archive.
Verified persistence layer health across all subsystems. 6 fixes shipped:

- **Policy NN: shadow tick removed from experience buffer** — `_compute_health_reward()`
  returned near-constant ~1.0 (5,925 records, mean=0.974, zero negatives). This
  flattened the advantage-weighted training signal, making all 123 trained models
  identical (none promoted). Fix: shadow tick still feeds evaluator A/B comparison
  via `score_retrospective()`, but no longer writes to experience buffer. Only
  interaction outcomes (with varied rewards) go into training data.
- **Autonomy: knowledge-aware outcome signal** — All 99 autonomy policy outcomes
  had `worked=false` because the delta tracker's 8 system health metrics don't
  capture knowledge gains from research. Fix: `_intent_metadata` now stores
  `memories_created` and `immediate_success`. `_process_delta_outcome()` sets
  `worked=True` when research created memories, even if system health didn't move.
- **Blue Diamonds: codebase sources skip graduation** — Study pipeline was
  attempting graduation on Python source files, which correctly failed the English
  stop-word heuristic but produced 59 noisy audit trail entries. Fix: early return
  on `source_type == "codebase"` in both `_try_post_study_graduation()` and
  `_try_graduate_to_blue_diamonds()`.
- **Flight Recorder: persistence across reboots** — Flight recorder was a pure
  in-memory deque (maxlen=50) that reset on every boot. Fix: save to
  `~/.jarvis/flight_recorder.json` after each append, load on module import.
  Atomic write (tmp + replace) prevents corruption.
- **Flight Recorder: text truncation increased** — Recording: 200→500 chars for
  both user input and response. Dashboard: user input 60→120 chars (header),
  response 100→300 chars (expanded detail). CSS: replaced `white-space: nowrap`
  with 2-line clamp for better visibility.

Data health summary (all verified live on brain at 192.168.1.222):
- memories.json: 500 memories, 100% provenance, 98% identity coverage
- consciousness_state.json: schema v2, evolution=transcendent, confidence=0.84
- beliefs.jsonl: 877 beliefs, confidence 0.4-0.7
- belief_edges.jsonl: 1,997 edges (350 derived, 1,647 supports)
- attribution_ledger.jsonl: 1,486 records, 67% resolution rate
- Blue Diamonds: 20 academic papers, quality 0.73-0.87
- Hemispheres: 5 focuses active (memory v38, mood v44, traits v31)
- Vector memory: 1,648 vectors, library: 68 sources, 908 concepts
- Policy: 123 versions trained, eval sidecar: 39,529 events tapped
- Goals: 8 tracked, Skills: 13 verified, David enrolled (face+voice)

- Test suite: 1597 passed (13 new regression tests)
- Grade: A

---

## Phase 2: Temporal Credit — SHIPPED (2026-03-16)

**Status: Code complete**, data accumulation in progress

`MetricHistoryTracker` in `brain/autonomy/metric_history.py` — per-hour Welford stats
for 8 metrics, blended into DeltaTracker's counterfactual estimation. Persists to
`~/.jarvis/metric_hourly.json`. Requires 20+ samples per hour bucket (3+ days) before
blended counterfactuals activate. Until then, pure trend extrapolation (existing behavior).

New `DeltaResult.counterfactual_source` field tracks `"trend"`, `"blended"`, or `"time_of_day"`.
24 tests in `test_temporal_credit.py`.

---

## Phase 4: Companion Training Automation — SHIPPED (2026-03-16)

**Status: Code complete**, ready for first user to start training

`OnboardingManager` in `brain/personality/onboarding.py` automates the 7-day Companion
Training Playbook:
- 7 day checkpoints with metric targets from the playbook
- Exercise prompts delivered via proactive speech pipeline
- Readiness Gate composite score (weighted average of 7 dimensions, target >= 0.92)
- `COMPANION_GRADUATION` event on completion
- Dashboard training tab on eval page with day cards, checkpoint tracker, readiness bar
- API: `POST /api/onboarding/start`, `GET /api/onboarding/status`
- Persistence to `~/.jarvis/onboarding_state.json`
- 35 tests in `test_onboarding.py`

Events: `ONBOARDING_DAY_ADVANCED`, `ONBOARDING_CHECKPOINT_MET`, `ONBOARDING_EXERCISE_PROMPTED`, `COMPANION_GRADUATION`.
Cycle: `onboarding` added to `ALL_CYCLES`. Tick interval: 60s.

---

## Audit 11 (2026-03-17)

Full codebase re-audit (4 parallel streams, 84K lines). 38 findings (2 Critical,
2 High, 16 Medium, 18 Low). 7 fixes shipped. Key: Phase 4 onboarding had two
Critical bugs (dict API on frozen dataclass + missing readiness weight producer)
that made graduation impossible. Tick pipeline crash resilience hardened. Pi camera
control NameError fixed. Autonomy queue ValueError guarded.

- Test suite: 1589 passed (12 new regression tests)
- Grade: A (10th consecutive)
- Remaining open: 31 items (0 Critical, 0 High, 16 Medium, 15 Low)
  - Medium items are thread safety, persistence atomicity, rewrite quality
  - Low items are dead code, design consistency, forward-looking constants

---

## Phase 2 + Phase 4 (2026-03-16)

- **Phase 2: Temporal Credit** — `MetricHistoryTracker` (Welford per-hour-of-day stats),
  blended counterfactual in `DeltaTracker._extrapolate_trend()`, `counterfactual_source`
  field on `DeltaResult`. Wired into orchestrator. 24 tests.
- **Phase 4: Companion Training Automation** — `OnboardingManager` (7-day playbook
  automation), Readiness Gate composite, `COMPANION_GRADUATION` event, dashboard training
  tab, API endpoints. 35 tests.

---

## Live Bug Fixes (2026-03-16, post-Audit 10)

Diagnosed from live Jarvis logs after reboot with Audit 10 fixes deployed:

- **Pi done-monitor ordering race** (V10-01 revised): Brain messages dispatched to
  multi-threaded executor caused `command/speak` and `response_end` to process out
  of order, leaving `_speak_cmd_monitor_active` stuck True. Fix: single-thread
  executor for ordered message processing, removed fragile flag, conv_id passed as
  parameter, `command/speak` no longer starts its own monitor.
- **Capability Gate false positive: "let me know"**: `let me` in `_OFFER_PATTERNS`
  incorrectly matched "let me know how I can assist". Fix: negative lookahead
  `(?! know\b)` added.
- **Capability Gate false positive: "stick with that"**: "maybe I'll stick with that"
  blocked as operational claim. Fix: added "maybe"/"perhaps" to subordinate clause
  prefixes, added "stick with"/"go with"/"stay with" to conversational safe phrases.
- **Introspection grounding miss**: "what capability do you not have?" confabulated
  missing capabilities because introspection excluded perception sections from
  "learning" topic and hemisphere data (0% distillation accuracy) was misread as
  lack of capability. Fix: added `identity_fusion` and `emotion` to learning topic
  sections, added disambiguation labels to hemisphere output.
- **Curiosity Bridge wiring bypass**: `evaluate_proactive()` returned early after
  greeting check, never reaching curiosity question delivery. Fix: explicit
  curiosity fallthrough after greeting path.
- **Philosophical prompt too permissive**: "If you could pick your own name" produced
  invented aesthetic preferences, grounding checker caught it (34 facts, 0 cited).
  Fix: tightened prompt to require evidence anchoring, forbid inventing preferences
  without recorded state.
- **Identity topic pattern too narrow**: "your own name" didn't match identity bucket.
  Fix: broadened pattern to `\byour (?:\w+ )?name\b` + bare `\bnamed?\b`.

---

## Audit 10 (2026-03-16)

Full codebase audit (brain + Pi) — 12 findings across safety, wiring, persistence:

- **Consciousness system crash resilience**: All 6 background cycle runners wrapped
  with try/except + mutation health check
- **Pi duplicate done-monitor threads** (V10-01 original): Prevented `response_end`
  from starting a second monitor when `command/speak` already started one (later
  revised — see live fixes above)
- **Perception orchestrator speaking lifecycle**: `_speaking_conv_id` cleared on
  playback complete, proactive speak includes `conversation_id` in broadcast
- **Hemisphere orchestrator model leak**: `engine.remove_model()` called on prune
- **Goal manager stale paused goals**: `system_health` goals reactivated on
  `metric_deficit` signal
- **Goal review non-metric completion**: Added completion path for template-exhausted
  goals with positive progress
- **Library index orphan prevention**: `conn.rollback()` added on chunk insert failure
- **Learning jobs cleanup exemption**: Builtin-family jobs exempt from "non-actionable"
  junk cleanup
- **Discovery 24h cooldown**: `_can_queue_proposal()` now checks global cooldown
- **Capability Gate verb coverage**: Added 12 blocked capability verbs (song, music,
  sketch, artwork, etc.)
- 30 new regression tests for Audit 10 findings

---

## Quick Fix Batch (2026-03-16)

- 8 audit items fixed: SPEAK-06, A5-05, A5-08, D5-3, D5-4, CG-04, CG-05, A5-07
- Open items reduced from 21 to 13 (0 Critical, 0 High)

---

## Architecture Documentation Refresh (2026-03-15)

- AGENTS.md rewritten (1525 → 739 lines, all findings from 7 sub-audits applied)
- ARCHITECTURE.md updated (2964 → 3072 lines, 7 new event sections, NN lifecycle)
- docs/SYSTEM_OVERVIEW.md created (280 lines, 6 mermaid diagrams)
- docs/AUDIT_REPORT.md created (243 gaps identified and resolved)
- .cursor/rules/architecture-check.mdc created (architecture verification rule)
- 4 superseded docs archived to docs/archive/

---

## Audits 1-5 (2026-03-14/15)

- 41+ findings across 5 audits, all critical/high items fixed
- 109 regression tests in test_audit_regressions.py
- Grade progression: A- → A → A → A → A
