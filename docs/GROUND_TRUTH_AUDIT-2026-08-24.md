# Ground-truth audit — 2026-08-24 (pass 2, complete)

Adversarial, evidence-first. Labels and output shape follow
`docs/SYSTEM_TRUTH_AUDIT_PROMPT.md`. Maturity/pillar gates checked against
`docs/MATURITY_GATES_REFERENCE.md` §§1–24 and `docs/ARCHITECTURE_PILLARS.md`.

Live brain: `duafoo@192.168.1.222`, `main.py` pid **19147** (started 16:51:30).
Branch: `feat/osv-p1-answer-path` @ `9c15401` (this pass adds this document).
Synced runtime is `/home/duafoo/duafoo` (no `.git` on the brain copy).

**This is not omniscience.** Anything not opened is **NOT EXERCISED**, not “fine.”
Evidence order: live runtime → persisted `~/.jarvis` → probes/pack → source → docs.

Pass 1 locked the turn floor, P1, spark math, maturity math, policy honesty lock,
probe battery, v2 GET liveness, restart ≠ wipe. Pass 2 traces the remaining
release-critical loops (self-improve, plugins/acquisition, epistemic L0–L12,
connectome pulse identity, persist split, SI apply, spoken-turn holes) and
re-probes live instruments.

No gates were flipped. No merge to `main`. Operator owns the stack.

---

# 1. SYSTEM VERDICT

**PARTIALLY OPERATIONAL — TRACE / TRUTH GAPS REMAIN**

JARVIS is a real two-device cognitive workshop. Inner life, memory store, L0
CapabilityGate, world-model L2, autonomy L2 (human-gated L3), epistemic write
stack, dashboard honesty kit, and restart-honest high-water are wired and
reporting. The LLM is voice, not the brain. Shadow / `PRE-MATURE` / `not_born`
means **wired, zero authority**.

She is not yet “curious then asks you.” Spark stays shadow: 11/20 external
answers, rate 1.0, pending 0. Inner SPARK thought `belief_validation_curiosity`
cannot fire because `_run_meta_thoughts_inner` never passes `grounding_tension`.
Several memory persist holes are real (session-start `remember(str)`, native
about-X with no conversation row, domain-recall can steal MEMORY, LLM store of
ungated `full_text`). Stage 2 self-improve is architecturally queued and
**broken after Approve** (health gate vs `PerformanceMetrics` dataclass) — it
is also **frozen at stage 0**, which is the correct Pillar 7 posture.

Oracle Silver + Validation Pack BLOCKED is **honest Pillar 9 divergence**, not
a scoring bug. Do not collapse them. Do not lower gates.

---

# 2. CURRENT REALITY VS CLAIMED REALITY

| Surface | Claimed | Actual | Verdict | Evidence |
|---|---|---|---|---|
| Release-path traceability | Origin → terminal chain on spoken turns | Spoken chunks pass L0+L12 before TTS. `OutputReleaseValidator` ledgers IDs **after** speech; no mute on `blocked`. Pack: released=5, validated=5, without=0. | **PARTIAL** | `conversation_handler.py:2538-2553,5573-5582`; pack `output_release_validation` PASS |
| Retry transparency | Retries emit lifecycle events | EventBus retries exist; this session had no handler-retry fire. | **NOT EXERCISED** | no retry events in this bounce’s probes |
| Memory provenance integrity | One `engine.remember(CreateMemoryData)` path | Canonical path **VERIFIED**. Session-start + enrollment `remember(str)` swallowed. Consolidation `storage.add` skips `MEMORY_WRITE`. Observer/transactions are second emitters. | **PARTIAL** / holes **REAL BUG** | `engine.py:375`; `conversation_handler.py:2923,4812` |
| Output validation gate | Nothing released without validation | L0 is the real mute/rewrite. Release validator is schema ledger, not a speech mute. Pack 0 `released_without_validation`. | **PARTIAL** (L0 **VERIFIED**; validator **PARTIAL**) | pack + `_gate_text` |
| Replayability | Trace explorer reconstructable | `/api/reconstructability`: only `trace_explorer` is reconstructable; ops/world-model/universe synthesized. | **PARTIAL** | `snapshot.py:_build_reconstructability_metadata` |
| Dashboard reconstructability | Honesty kit, no fake-zero | `window.V2` fail-closed on shared.js pages. `dashboard_truth_probe` 0 findings. Universe is live-synthesized. | **VERIFIED** kit / **PARTIAL** reconstructability | probe PASS; `shared.js` |
| Acquisition pipeline | Job → lanes → truth | Wired. Live `learning_jobs=0`, `acquisitions=0`. Pack NEVER on job contracts. | **VERIFIED** code / **DATA-GATED** live | pack + `~/.jarvis/learning_jobs` empty |
| Plugin lifecycle | quarantined → shadow → supervised → active | Ladder **VERIFIED**. Live: 4 quarantined, 0 subprocess, never `active` this window. | **VERIFIED** / **NOT EXERCISED** promote | `/api/plugins` |
| Self-improvement Stage 2 | Human-approve → apply → verify → ledger | Stage **0 frozen**. Queue artifacts exist. `approve()` health gate calls `.get()` on a dataclass → fail-closed rollback. Restart-verify `conv=None`. | **BROKEN** if unfrozen / **NOT EXERCISED** live | `orchestrator.py:1003-1018,1171`; `/api/self-improve` stage=0 |
| Golden command lane | Deterministic, conf 1.0, skips P1, still gated | Code **VERIFIED**. No Golden issued this bounce. | **VERIFIED** code / **NOT EXERCISED** live | `golden_words.py`; router `:1339` |
| Explainability path | Ledger + reconstructability | Attribution JSONL 9621 lines. Panels partial. | **PARTIAL** | `attribution_ledger.jsonl` |
| World model validation | L2 after 50 preds / 4h / 0.65 | Live L2, 24893 validated, 93%. Simulator is a **separate** gate (L1 advisory, 2618, 86%, 48h). | **VERIFIED** | `/api/world-model/diagnostics` |
| Policy NN path | Shadow A/B → 55% decisive → features | Honesty lock: `nn_reward = kernel_reward = actual_reward` ⇒ always tie. 0/3 current. Trainer JSONL **alive**. | **VERIFIED** measurement failure | `evaluator.py:236-253`; `/api/policy` |
| Specialist training | Distillation + ladder | 8/8 sample floors. Voice-intent 248/500, agreement 0.634, **shadow**, 0 primary overrides. | **VERIFIED** samples ≠ authority | `/api/intent-shadow`; maturity hemi 8/8 |

---

# 3. TOP FINDINGS (RANKED)

### F1 — Session-start and enrollment `engine.remember(str)` never persist
- **Severity:** P1 truth / continuity sidecar
- **Surface:** memory write
- **Symptom:** “first words this session” / “first met X” rows missing
- **Root cause:** `remember(self, data: CreateMemoryData)` at `engine.py:375`; callers at `conversation_handler.py:2923` and `:4812` pass a string + kwargs. `TypeError` swallowed by `except: pass`.
- **Why dangerous:** restart-honest continuity narrative has a hole at session boundaries; not a store wipe.
- **Evidence:** signature vs call; tests **UNTESTED** for this hop. Live memories **768** still span 2026-06-23→today.
- **Fix direction:** wrap those two sites in `CreateMemoryData(...)`. Do not change `remember` to accept `str`.
- **Class:** **REAL BUG**

### F2 — Domain-recall can overwrite about-X MEMORY
- **Severity:** P1 operator-trust (wrong mouth)
- **Surface:** spoken routing
- **Symptom:** “what do you know about X” can dump an ingested domain instead of lived memory
- **Root cause:** about-X sets `ToolType.MEMORY` (`:3072-3095`); domain probe (`:3097-3118`) skips only golden + `self_view_kind`, not `about_subjects`. `recall.py` `_TOPIC_MATCH_MIN = 2`.
- **Why dangerous:** lived pet/person recall replaced by library scrap. Lived Skyler worked because no domain matched — path still live.
- **Evidence:** order in `handle_transcription`; `test_capability_domains_phase2.py` does not pin handler overwrite.
- **Fix direction:** skip domain-recall when `about_subjects` is set (or when routing.tool is already MEMORY).
- **Class:** **REAL BUG**

### F3 — Persist split: ungated LLM store vs missing native MEMORY store
- **Severity:** P1 memory provenance
- **Surface:** persist
- **Symptom:** spoken ≠ stored; native about-X has episode row only
- **Root cause:** `_persist_spoken_turn` only at P1 `:3450`, emergence `:3934`, INTROSPECTION `:4333`. `respond_stream` persists ungated `full_text` (`response.py:875-880`) while TTS is `_gate_text`. Native MEMORY (`:3572-3608`) never calls `_persist_spoken_turn`; `episodes.add_assistant_turn` still runs (`:5904`).
- **Why dangerous:** CapabilityGate rewrites can be forgotten; ungated claims can be remembered; next-turn hybrid search misses native about-X.
- **Evidence:** three call sites of `_persist_spoken_turn`; INTROSPECTION `persist_response=False` is **VERIFIED** (`test_continuity_write_path.py`).
- **Fix direction:** persist **gated spoken text** on all native mouths; do not persist ungated `full_text` when the gate rewrote.
- **Class:** **REAL BUG**

### F4 — Inner SPARK thought never receives `grounding_tension`
- **Severity:** P1 curiosity last-mile (internal)
- **Surface:** spark / meta-thoughts
- **Symptom:** she thinks, never names a high-tension belief, never asks
- **Root cause:** `belief_validation_curiosity` needs `ctx.grounding_tension ≥ 0.40` and `grounding_target_id` (`meta_cognitive_thoughts.py:499-505`). Only production caller `_run_meta_thoughts_inner` (`consciousness_system.py:728-744`) does not pass those keys. Even if it did: `TensionThoughtPromotion` shadow would log and `return None` (no `KERNEL_THOUGHT`); `_record` also drops `belief_id`.
- **Why dangerous:** designed SPARK seed is a dead wire. Empty tray is **also** drive-win + answered-never-reopen (not this bug alone).
- **Fix direction:** pass last `GroundingTensionReport` into context. Keep TensionThoughtPromotion **shadow**. Do not TTS. Do not flip `GroundingDrivePromotion`.
- **Class:** **REAL BUG** (unwired trigger). Promotion remaining shadow is **DATA-GATED / design**.

### F5 — Stage 2 Approve would write-then-rollback (dormant)
- **Severity:** P1 if Stage 2 is ever enabled; **not live**
- **Surface:** self-improve
- **Symptom:** operator Approve does not stick
- **Root cause:** `kernel.get_performance()` returns `PerformanceMetrics` dataclass (`kernel.py:121-128,285`). Health gate uses `.get("p95_tick_ms")` / `.get("last_tick_ms")` (`self_improve/orchestrator.py:1003-1018`). Exception → `pre_p95=0` → fail-closed rollback. `approve()` also passes `conv=None` into `_prepare_restart_verify`. Tests mock **dicts**.
- **Why dangerous:** unfreezing Stage 2 without this fix makes “Approve” a snapshot restore. Default `SELF_IMPROVE_STAGE=0` **correctly hides it**.
- **Fix direction:** read dataclass fields. Do **not** flip stage. Staging evidence after the wiring fix.
- **Class:** **REAL BUG** on a rollout-gated path → treat as **staging blocker**, not a production patch this bounce.

### F6 — Face last-seen / fusion.current can stamp the wrong owner
- **Severity:** P1 identity (guest/family)
- **Surface:** identity → memory stamp
- **Symptom:** about-me follows sticky last fused identity; guest can bind to David
- **Root cause:** face persist 90s/180s (`identity_fusion.py:693-712`). `_on_speaker_id` unknown does not clear name (`perception_orchestrator.py:1190`). `resolve_for_memory` prefers `fusion.current` (`resolver.py:100`). `engine._current_speaker` is never assigned in production.
- **Why dangerous:** family/guest contract (“about-me follows this-turn speaker”).
- **Evidence:** fusion tests exist; about-me × persist method **UNTESTED**.
- **Fix direction:** about-me and remember stamp must use this-turn handler `speaker`, not last-seen face, when voice is known.
- **Class:** **REAL BUG** (path); live this bounce speaker=David **VERIFIED** on PVL identity contracts.

### F7 — Skylar alias + about-me courtesy leak
- **Severity:** P2 recall quality
- **Surface:** MEMORY aboutness
- **Root cause:** first-sentence aboutness is substring (`memory_tool.py:236-252`). Courtesy “You're welcome, David” still contains David.
- **Class:** **REAL BUG** (quality). Lived Skyler spelling **VERIFIED**.

### F8 — Registry / docs say 8 P1 kinds; code has 10
- **Severity:** P2 signage
- **Surface:** OSV
- **Evidence:** `articulate.py:29-33` vs `subsystem_registry.json:755` vs `ARCHITECTURE.md:771`.
- **Class:** **SIGNAGE**

### F9 — Maturity Autonomy 5/5 and Hemisphere 8/8 over-read as authority
- **Severity:** P2 operator trust
- **Surface:** `maturity.html`
- **Actual:** `auto_level` threshold is **2**. `auto_wins_l3` is `total_wins≥25`, **not** `eligible_for_l3` (needs WR≥0.50, 0 regressions in last 10, **and** `evidence_path` to move). Live L3 reason: “Meets thresholds but **5 regressions in last 10 jobs**.” Hemisphere 8/8 = sample floors + 4 slots. Voice-intent samples ≠ live router.
- **Class:** **MISLEADING signage**. Do not tune 25 / 0.55.

### F10 — Grounding POST is not view-only
- **Severity:** P2 signage (behavior is the intended closure)
- **Surface:** `grounding.html` / API docstring
- **Actual:** `_ground_belief` re-stamps `user_claim` and nudges confidence (`grounding_queue.py:473-508`). Confirm **or** refute counts `grounded=True`. Live totals: enqueued 11, answered 11, **confirmed 8, refuted 3**, validation_history all 1.0.
- **Class:** **SIGNAGE** (docs lie; closure is correct)

### F11 — Policy 0/3 and `policy_win_rate` ever=True
- **Severity:** P2 measurement honesty
- **Actual:** honesty lock forces ties. Session A/B counters reset (this bounce **3/100**). `ever_met=True` on decisive WR is high-water from the **pre-lock circular scorer**. Features-of-8 is structurally unreachable (auto-enable 5, pipeline 4).
- **Class:** **MISLEADING** if read as “train harder.” **UNCROSSABLE** until interleaved execution. Do not tune 0.55/100.

### F12 — Connectome “87 untracked”
- **Severity:** P3 observability
- **Actual:** `_classify`: no design record → `untracked`. Live deviations: untracked 87, as-designed 17, as-designed-idle 3, **DEVIATION 0**. Pulse type **is** `bus.emit` event name. KERNEL_TICK has no subscriber → SSE without universe pulse.
- **Class:** **SIGNAGE** / registry coverage. Not 87 dead wires.

### F13 — Pack BLOCKED on skill-learning + PVL 75.9%
- **Severity:** none as a product bug
- **Actual:** `learning_jobs=0`, `acquisitions=0`. Triggers (`create_job`, 300s tick) **alive**. PVL 82/114 (75.9%), ever 93 (86.1%). Soul 0.778/0.870 recovering. HRR primitive library ever-ok, current stale age.
- **Class:** **DATA-GATED / restart debt.** Do not lower 85%.

### F14 — Auditor pytest wrote into live `~/.jarvis` (this pass)
- **Severity:** process, not product
- **What happened:** pytest on the brain host overwrote `plugin_registry.json` with `test_plugin` active. **Restored** from `/tmp/gt_api.json` pre-test snapshot to the four quarantined plugins. Live `/api/plugins` (in-process) stayed 4 quarantined the whole time. `improvements.json` test dry-run history restored to empty. Grounding drive **11** and memories **768** untouched. Contaminated copies kept at `~/.jarvis/auditor_restore_2026-08-24/`.
- **Class:** **AUDITOR ERROR**. Rule added to `AGENT_MAP.md`.

---

# 4. ACCEPTANCE GATES SCORECARD (G1–G8)

Trace acceptance gates from the audit prompt.

| Gate | Result | Blocks “trace-complete”? | Notes |
|---|---|---|---|
| **G1** Final output lineage | **PARTIAL** | Yes for a hard “complete” claim | L0 rewrite is live. Release validator is post-speech ID hygiene. Pack 5/5/0. |
| **G2** Cross-boundary context | **PARTIAL** | Yes | Pi→brain WS **VERIFIED** (Pi connected 2, 39700 events). Face is last-seen. Persist split. |
| **G3** Retry transparency | **NOT EXERCISED** | No (lane idle) | No handler-retry this bounce. |
| **G4** Memory provenance | **PARTIAL** | Yes | Canonical path + quarantine-never-reject **VERIFIED**. F1/F3/F6 holes. |
| **G5** Policy evidence | **PARTIAL** | No for release of speech (policy is shadow) | A/B identity lock **VERIFIED**. Window RAM-only. Trainer JSONL persists. |
| **G6** Output validation before release | **PARTIAL** | Borderline | L0 **is** before TTS. `validate_output` is after. |
| **G7** Replayability | **PARTIAL** | Yes for omniscience | Trace explorer reconstructable; most v2 pages synthesized. |
| **G8** Dashboard reconstructability | **PARTIAL** | Yes if “every green is a ledger” | Honesty kit **VERIFIED**. Maturity signage **MISLEADING** on L3/hemi/policy. |

**Trace-complete claim right now: no.** Spoken L0 path is the strongest chain.

---

# 5. FALSE SIGNALS / OVERSTATED SURFACES

| Surface | Overstatement | Reality |
|---|---|---|
| Autonomy Pipeline 5/5 ACTIVE | “She may self-mod / L3” | Level **2**. L3 needs evidence_path + 0 regressions (live: 5 regressions). L2 code-patch bridge still `return False`. |
| Hemisphere 8/8 | “NN routes speech” | Sample floors + 4 slots. Heuristic router live. Intent 0 primary overrides. |
| Neural Policy 0/3 + ever WR | “almost / used to win” | Honesty lock ⇒ ties. `ever_met` is pre-lock high-water. |
| Spark empty Pending | “queue broken after bounce” | 11 answered never reopen; last `Grounding drive (SHADOW)` log **2026-07-03**; batch only if grounding **wins** a 60s tick. Fuel `inferred_count=1` + scrap tension. |
| Grounding “view-only” | UI/API copy | Belief provenance **is** mutated on answer. Intended P3 closure. |
| Connectome untracked 87 | “dead bus” | No registry design field. True `DEVIATION` count **0**. |
| World Model 4/4 | “simulator mature” | Simulator is a separate 48h/100/0.70 gate, currently advisory. |
| Dream “500-buffer” | ring size | Ring **200**. 500 is the created accumulation gate. Live 713 created, 366 promoted. |
| Soul docstring “worst-case” | min of dims | Weighted mean of non-stale dims. Live 0.778, weakest `autonomy_effectiveness=0.5`. |
| SPARK_DESIGN “weighted max” | graph-level max | Per-belief blend 0.45/0.35/0.20 then aggregate `0.6 tmax + 0.4 tmean`. |
| AUTONOMOUS_DRIVE_MAP “no self-sensing field” | stale | Fields exist; **still drive no urgency**. |
| ARCHITECTURE.md “8 kinds” / “26 v2 pages” | drift | 10 kinds; 33 HTML, two retired. |
| `approve()` “same safety as normal pipeline” | docstring | Health `.get` + `conv=None`. |
| Plugin `supervised` | human-in-the-loop | User-routable with results. |
| Isolation = sandbox | design language | Process-isolated, **not** FS/net/cgroup (honest in `plugin_process.py` docstring). |
| Oracle Silver vs pack BLOCKED | contradiction | Pillar 9: two instruments. Oracle composite 88, seal Silver, `world_grounding_coherence=0` staged WARN not BLOCK. |

---

# 6. VERIFIED STRONG PATHS

Be specific. These are real.

- **Two-device split.** Pi senses (`pi-lidar` + `pi5-senses` connected, 0 disconnects, last event 0s). Brain decides. Pi has no `remember()` / beliefs.
- **Spoken turn floor.** STT + speaker fusion + heuristic router + P1 override + about-X MEMORY + L0 `_gate_text` + Kokoro TTS. Live PVL voice_pipeline **4/4 PASS**, identity_pipeline **4/4 PASS**, memory_pipeline **4/4 PASS**.
- **P1 OSV.** 10 kinds including `continuity` and `answer_path`. LLM does not author self-facts. Speakable `answer_path` lived 14:45–15:26. Teacher revoice after TTS; student **`not_born`**; live_voice `deterministic_floor`.
- **Wipe-lie skip.** `contradicts_measured_continuity` + INTROSPECTION `persist_response=False`. Tests pin it. Continuity kind measured restart ≠ wipe.
- **Restart honesty.** Memories **768**. Boot `Autonomy persisted L2` then **`Boot auto-restore: L1 → L2` … `snapshot_match=True`** at 16:51:41. High-water 31/35 ever. Matrix specialists probationary after boot by design.
- **CapabilityGate L0.** Fail-closed DEFAULT BLOCK + `_gate_text` exception strip. Pack HRR non-influence **PASS**. Language bridge **OFF**, unpromoted_live=0.
- **World model L2.** 24893 validated, 93%, synthetic excluded from promotion hist.
- **Quarantine on remember.** Soft tag/downweight, never reject the write.
- **Belief graph.** 533 beliefs.jsonl, 1715 edges.jsonl, maturity active-beliefs/edges gates green. Extraction on MEMORY_WRITE **VERIFIED**.
- **Spark math.** \(N\ge20 \land rate\ge0.40 \land t_{shadow}\ge4h\). Confirm **or** refute = grounded. Shadow default. Advisory TTS only at level 1. P5b default OFF. Queue sha1 keys; answered never reopen.
- **Policy honesty lock.** `nn_reward = kernel_reward = actual_reward`. Promotion uncrossable without interleaved execution. **Do not retune.**
- **Affect.** 19931 paired, live-tick 183h, still level 0 because the controller **never auto-promotes** (operator call). Correct.
- **TBS-0.** Injects nothing. `injects_prompt=False`.
- **Golden words.** Exact prefix, conf 1.0, skips P1 classify, **does not** skip CapabilityGate.
- **Synthetic gyms.** Cannot satisfy lived WM/sim/eval/skill-proof gates. Origin tagged. Fidelity 0.7 into distillation only.
- **Probes this bounce.** `dashboard_truth_probe` PASS 0/0/0. `schema_emission_audit` 0 violations (edges 5/5, evidence 10/10). `docs_truth_audit` PASS 0 fail / 1 warn unserved HTML / 4 info. Self-test: engine_alive + serializer_shape OK; overall `blocked` **because pack blocked** (honest).
- **Contract pytest (brain venv, 21 files):** **541 passed, 2 failed.** Failures are test drift / live-host isolation (`casual_conversation` extra provenance; `maybe_promote` reads teacher samples from disk). Not treated as product regressions without a second isolated run.

---

# 7. REMAINING ACTIONS

**Do not implement until the operator says go.** No gate flips.

### P0 release blockers
None that currently dump ungoverned speech. L0 is live. Pack `released_without_validation=0`.

### P1 truth / trace hardening (REAL BUG wiring)
1. `CreateMemoryData` at session-start + enrollment (`conversation_handler.py:2923,4812`).
2. Skip domain-recall when `about_subjects` / MEMORY already won.
3. Persist **gated spoken** on native MEMORY; stop storing ungated `full_text` when the gate rewrote.
4. Pass `grounding_tension` + `grounding_target_id` into `_run_meta_thoughts_inner`. Keep spark **shadow**. No TTS.
5. About-me / remember stamp = this-turn speaker when voice is known (F6).
6. **Do not enable Stage 2** until health gate reads `PerformanceMetrics` fields and `approve()` arms restart-verify.

### P2 operator UX / observability honesty
- Registry + ARCHITECTURE.md: 10 P1 kinds.
- Maturity labels: `auto_wins_l3` is a count; hemisphere 8/8 is samples; policy 0/3 is the honesty lock.
- Grounding UI: answering **does** close a belief.
- Shadow handler docstring still says “enqueue NOTHING” — code now pull-enqueues (`ddc6d8b`).
- SPARK_DESIGN / AUTONOMOUS_DRIVE_MAP / soul “worst-case” docstring.

### P3 deferred / data-gated
- Spark tray fuel: batch even when grounding is not top drive; filter library-scrap from inferred headline. **Keep shadow.**
- Skylar alias; about-me courtesy filter.
- Language evidence n→30 (do not lower 30).
- PVL 75.9% → 85% by lived events (do not lower 85%).
- Soul 0.778 → 0.870 recovering.
- Skill-learning job actually run (`LEARN SKILL` / gate auto-job) — staging evidence, not a code “fix.”
- SI Stage 2 e2e after F5 — staging evidence.
- Plugin dotted-import allowlist (`os.path` / `urllib.parse` split on `.`) — latent.
- Full `brain/tests/` (~265 files) in **isolated** tmp HOME.
- Browser click-through of every v2 page (DevTools was down pass 1; not repeated as UI verification this pass).

---

# 8. SURGICAL FIX PLAN

Only REAL BUG rows. No Do-Not-Tune gates. Operator must still say go.

### FIX-A — session-start / enrollment remember
- **Files:** `brain/conversation_handler.py:2923`, `:4812`
- **Change:** `canonical_remember(CreateMemoryData(payload=..., type=..., tags=..., weight=..., provenance="observed"))`
- **Artifact:** a `session_start` / `milestone` row in `memories.json` after next session/enroll
- **Validation:** new unit test that `remember` is not called with `str`; bounce + one greeting
- **Classification:** REAL BUG
- **§24:** n/a (wiring)
- **Pillars:** 3 (tri-layer), 10 (continuity). Preserve never-discard; this **adds** the missing write.
- **Fresh-brain:** safe — extra observed rows, no trust inflation.

### FIX-B — domain-recall must not steal MEMORY
- **Files:** `brain/conversation_handler.py:3097-3118`
- **Change:** also skip when `routing.tool == MEMORY` or `extracted_args.about_subjects`
- **Artifact:** about-X still ToolType.MEMORY in flight recorder when a domain also matches
- **Validation:** handler test with about-X + matching domain
- **Classification:** REAL BUG
- **§24:** n/a
- **Pillars:** 4 (truth boundary), 6 (identity/scope)
- **Fresh-brain:** safe

### FIX-C — persist gated spoken text
- **Files:** `conversation_handler.py` MEMORY native arms; `reasoning/response.py:875-880`
- **Change:** `_persist_spoken_turn` after native MEMORY; stream persist uses gated text (or skip persist when `persist_response=False` and finalize from spoken)
- **Artifact:** conversation memory equals TTS text
- **Validation:** extend `test_continuity_write_path.py`
- **Classification:** REAL BUG
- **§24:** n/a
- **Pillars:** 4, 10
- **Fresh-brain:** safe

### FIX-D — SPARK thought context
- **Files:** `consciousness_system.py:_run_meta_thoughts_inner`
- **Change:** copy `grounding_tension`, `grounding_target_id`, claim, provenance from `_last_grounding_report` (or ProvenanceScorer.compute)
- **Artifact:** inner thought logs naming a belief when tension ≥ 0.40 — **still no TTS** while TensionThoughtPromotion is shadow
- **Validation:** unit test context keys; no change to promotion JSON
- **Classification:** REAL BUG (wiring)
- **§24:** TensionThoughtPromotion / GroundingDrivePromotion are **Do Not Tune** — this fix does not flip them
- **Pillars:** 8 (goal-aligned autonomy), 7 (no authority grant)
- **Fresh-brain:** safe — without tension the trigger still stays silent

### FIX-E — this-turn speaker stamp
- **Files:** `identity/resolver.py:100-119`; handler about-me cue already uses `speaker=`
- **Change:** when this-turn voice name is known and not `unknown`, stamp that, not `fusion.current` persist/face_only
- **Classification:** REAL BUG
- **§24:** identity thresholds Do Not Tune — this is owner selection, not a threshold
- **Pillars:** 6
- **Fresh-brain:** safe

### FIX-F — SI health gate (do not enable Stage 2)
- **Files:** `self_improve/orchestrator.py:1003-1018,1171`; tests that mock dicts
- **Change:** `perf = kernel.get_performance(); pre_p95 = perf.p95_tick_ms`; pass a real conversation into `_prepare_restart_verify`
- **Artifact:** after a **staging** Stage 2 approve, snapshot + pending_verification, not immediate rollback
- **Classification:** REAL BUG (dormant)
- **§24:** SI stage system **Do Not Tune** — this does not change stage; it makes the existing gate honest
- **Pillars:** 7
- **Fresh-brain:** safe
- **Do not ship as “turn Stage 2 on.”**

---

# 9. AUDIT CONFIDENCE

**Directly evidenced (live):** pid 19147; GET probes; pack 27/44 current, 30 ever, 3 regressed; PVL 82/114 75.9%; maturity 29/35 active, 31 ever; grounding queue 11/11 answered, pending 0, 8 confirm / 3 refute; grounding drive level 0, 11 outcomes, rate 1.0, 1491.9h shadow, 36 selections_shadowed; memories 768; beliefs 533 / edges 1715; autonomy L2 restore log; WM L2 24893/0.93; sim L1 2618/0.86; policy shadow, 3 A/B this session; voice-lab `not_born`; HRR PRE-MATURE all influence false; plugins 4 quarantined; SI stage 0 frozen; Pi 2 links; Oracle Silver 88 integrative; truth cal 0.727 ACTIVE; soul 0.778 recovering; intent 248/500 agreement 0.634 shadow; affect 19931 paired still L0; last Grounding SHADOW log 2026-07-03; probes PASS.

**Code-traced then labeled:** F1–F6, SI health `.get`, domain-recall order, persist call sites, SPARK context keys, connectome `_classify`, ProvenanceScorer weights, soul 10-D, L0–L12 fire sites, plugin ladder, acquisition lanes, Golden skip-P1, TBS-0.

**Inferred:** empty tray this bounce is drive-win (no SHADOW log since July 3) + answered-never-reopen + inferred_count=1. Face-stamp guest leak not lived this bounce (speaker=David). Domain-recall steal not lived (Skyler worked).

**Not exercised:** full 265-file pytest in isolated HOME; Stage 2 apply; plugin promote-to-active; Golden command; TBS-2; OSV P2 active; P5b; L3 escalate; browser click-through; Pi process inspection beyond `/api/pi5`; every EventBus retry; live compact of belief JSONL.

**Auditor incident:** pytest on the live host wrote `plugin_registry.json` / SI dry-run files. Restored registry + empty improvements from the pre-test API snapshot. Live in-memory plugins never flipped. Grounding 11 and memories 768 untouched. **Do not repeat.**

---

# 10. FIX PLAN SELF-AUDIT

| Fix ID | Finding Classification | §24 Gate Class | Pillar(s) | Preserved? | Rollout-Gate | Fresh-brain Safe? | Outcome |
|---|---|---|---|---|---|---|---|
| FIX-A | REAL BUG | n/a wiring | 3, 10 | yes | none | yes | **Kept as P1** |
| FIX-B | REAL BUG | n/a wiring | 4, 6 | yes | none | yes | **Kept as P1** |
| FIX-C | REAL BUG | n/a wiring | 4, 10 | yes | none | yes | **Kept as P1** |
| FIX-D | REAL BUG (unwired) | Do-Not-Tune promotion **untouched** | 7, 8 | yes — still shadow | none | yes | **Kept as P1** |
| FIX-E | REAL BUG | identity thresholds untouched | 6 | yes | none | yes | **Kept as P1** |
| FIX-F | REAL BUG dormant | SI stage Do-Not-Tune (not flipped) | 7 | yes | `SELF_IMPROVE_STAGE=0` | yes | **Downgraded to Staging Acceptance Evidence** for apply; **Kept** as “do not unfreeze until fixed” |
| Spark batch-every-tick | DATA-GATED drive-win | Do-Not-Tune GroundingDrivePromotion | 7, 8 | — | shadow | — | **Deleted from P1 — not a REAL BUG** |
| Lower PVL 85% / spark 20/0.40 / policy 0.55 | PRE-MATURE / design | **Do Not Tune** | 7, 9 | would invert | — | — | **Deleted** |
| Flip native_voice / P2 / L3 / language bridge | gated mouths | Do Not Tune | 7 | would invert | OFF | — | **Deleted** |
| F8/F9/F10/F12 signage | SIGNAGE | n/a | 9 | yes | none | yes | **Downgraded to Signage** (P2 copy) |

---

# 11. SIGNAL TRACES AND MATH (locked this pass)

## 11.1 Spoken turn

```
Pi PCM (often 48 kHz) → int16 16 kHz → ws:9100
  → openWakeWord → Silero VAD → LaptopSTT
  → this-turn ECAPA speaker + emotion; face = last-seen (90s persist)
  → IdentityFusion → handle_transcription
  → personal intel + TBS-0 (injects nothing)
  → tool_router heuristic LIVE (voice-intent observe-only)
  → P1 classify_self_question? → articulate_self_view (10 kinds)
  → else about-X MEMORY?
  → else domain-recall (can steal MEMORY)   ← F2
  → native speak or respond_stream
  → _gate_text L0 + L12 → Kokoro TTS → Pi aplay
  → persist split (F3) → revoice teacher AFTER speech (not the mouth)
```

P1 kinds: `identity, capabilities, recent_changes, health, weaknesses, gated_capabilities, unknowns, consciousness_query, continuity, answer_path`. `\bremember about\b` returns None so MEMORY can win.

## 11.2 Memory write / recall

Canonical: synthetic-session block → identity stamp → create → L8 soft quarantine → salience advisory → store → tag index → vector → `MEMORY_WRITE`.

Recall: vector → identity boundary (fail-open on exception) → ranker (heuristic fallback) → aboutness / curiosity skip / wipe skip.

Salience blend: start 0.2, +0.1 per 500 validated, cap 0.6. Live blend **0.200 / 0.600** (progress). Ranker trains 42, enabled.

## 11.3 Spark / drives / autonomy

Provenance tension (view-only):

\[
t = 0.45(1-c_{\mathrm{eff}})_{\mathrm{inferred|scrap}} + 0.35\cdot\mathbf{1}_{\mathrm{orphan}} + 0.20\cdot P_{\mathrm{quarantine}}
\]

then outward (self/ops ×0.2, identity ×0.7) and anti-fix (×0.5). Aggregate \(0.6 t_{\max}+0.4 t_{\mathrm{mean}}\). Grounded provenance `{observed,user_claim,external_source}` → t=0.

Grounding urgency: tension + 0.20 gaps + 0.20 drift + 0.10 misses, floor 0.10, select if ≥0.25. Self-sensing fields **do not** enter `_URGENCY_FNS`.

Promote one step (shadow→advisory→active):

\[
N\ge 20 \;\land\; |hist|\ge 20 \;\land\; \overline{hist}\ge 0.40 \;\land\; t_{\mathrm{shadow}}\ge 4\mathrm{h}
\]

Live: 11/20, rate 1.0, 1491.9h, **not ready**. Batch enqueue only inside shadow handler = only if grounding **wins**.

Autonomy: L2 `wins≥10 ∧ WR≥0.40`; L3 `wins≥25 ∧ WR≥0.50 ∧ regressions_10=0` **and** `set_autonomy_level(3, evidence_path=...)`. Live WR 72.2%, 57 wins, **5 regressions** → L3 denied. Boot auto-restore L2 only (never L3).

## 11.4 Policy

\[
\texttt{nn\_reward}=\texttt{kernel\_reward}=\texttt{actual\_reward}
\Rightarrow \mathrm{margin}=0 \Rightarrow \mathrm{tie}
\]

Eligible for control: \(N\ge100 \land \mathrm{decisiveWR}>0.55 \land \mathrm{decisive}\ge\max(30,\lfloor0.15N\rfloor) \land \overline{\mathrm{margin}}>0.03 \land n_{\mathrm{nn}}>n_{\mathrm{kernel}}\). Uncrossable in pure shadow. `DEVIATION_BONUS` is dead.

## 11.5 Maturity `_gate`

\[
pct=\min(100,\;100\cdot c/t),\quad
status=\begin{cases}
\mathrm{active}&c\ge t\\
\mathrm{progress}&0<c<t\\
\mathrm{locked}&c=0\lor c=\mathrm{None}
\end{cases}
\]

UI: active→ACTIVE; else ever_met→RECOVERING; else PRE-MATURE. Backend `progress`/`locked` are not those words.

Live 29/35, 31 ever. Gestation 2/2. Policy 0/3. WM 4/4. Cortex 3/4 (salience blend). Autonomy 5/5 at **L2**. Hemi 8/8. Dream 3/4 (245/500 buffer). Epistemic 4/5 (soul recovering). Truth cal now **ACTIVE** 0.727/0.650 (pass 1 was 0.636 recovering).

## 11.6 Soul integrity

Weights sum to 1.0: memory 0.12, belief 0.12, identity 0.10, skill 0.10, truth 0.12, graph 0.08, quarantine 0.08, autonomy 0.10, audit 0.10, stability 0.08.

\[
index=\frac{\sum_{\mathrm{not\ stale}} s_i w_i}{\sum_{\mathrm{not\ stale}} w_i}
\]

Repair <0.50, critical <0.30. Live 0.778, weakest autonomy_effectiveness 0.5 (N warmup floor).

## 11.7 Epistemic fire map (L0–L12)

| Layer | Fires on | Persist | Verdict |
|---|---|---|---|
| L0 CapabilityGate | spoken chunks | confab jsonl (dashboard) | **VERIFIED** |
| L1 Attribution | spoken + jobs | attribution_ledger.jsonl (9621) | **VERIFIED** |
| L2 provenance boost | recall | — | **VERIFIED**; dynamic half DATA-GATED |
| ProvenanceScorer | drive tick, view-only | — | **VERIFIED** code |
| L3 identity boundary | recall | — | **VERIFIED**; fail-open |
| L3A persist | perception | RAM 180s | **VERIFIED** |
| L3B scene | detections | RAM | **PARTIAL** no memory write |
| L4 delayed | 15s tick | pending RAM | **PARTIAL** |
| L5 contradictions | MEMORY_WRITE | beliefs.jsonl | **VERIFIED** |
| L6 calibration | 120s tick | calibration_truth.jsonl (334) | **VERIFIED** |
| L7 graph | new belief, not raw MEMORY_WRITE | belief_edges.jsonl | **VERIFIED** |
| L8 quarantine | remember + 60s | candidates jsonl | **VERIFIED** never reject write |
| L9 audit | 300s | RAM ring | **VERIFIED** advisory |
| L10 soul | 120s | RAM | **VERIFIED** |
| L11 compaction | size triggers | edges jsonl rewrite | **NOT EXERCISED** live |
| L12 Stage-0 | spoken `_gate_text` | intention_registry.json | **VERIFIED** |
| L12 Stage-1 resolver | — | verdicts jsonl | **SHADOW** |

P2 `ground_self_claims(..., active=False)` shadow. OSV_P2 not flipped.

## 11.8 Self-improve / plugins

SI: `ENABLE_SELF_IMPROVE` default false, stage 0 frozen, scanner scan_count 3, coder model on disk, GPU layers 0, pending 0, total_improvements 0. Stage 2 queue is real (`pending_approvals.json`). Apply path **BROKEN** if unfrozen (F5).

Plugins: quarantined → shadow (zero result) → supervised (routable) → active only via owner deploy. Live 4 quarantined, 0 invoke. Skill-learning tick 300s; 0 jobs this window → pack NEVER. Trigger not dead.

## 11.9 Connectome

107 nodes, 259 edges, 145 subscribed types, 121 emit-map events. Pulse identity = event name **VERIFIED**. True DEVIATION **0**.

## 11.10 Pack (live)

status=blocked. 27/44 current, 30 ever, 3 regressed (`pvl_coverage`, `soul_integrity`, `hrr_primitive_library`). Critical blocked: pvl_coverage 75.9% (ever 86.1%), learning_job_started 0, job_phase_advanced 0, skill_learning_completed 0, skill_learning_lifecycle 1/4. Language evidence floors n<30 **DATA-GATED**. Language runtime guardrails PASS (bridge off). HRR truth-boundary PASS. P5 mental world PRE-MATURE PASS. `output_release_validation` PASS. `l3_escalation_requestable` NEVER (5 regressions).

Oracle: Silver, Adept, integrative, composite 88.0, `is_measurement=False`, restore verified. world_grounding floor BLOCK staged as WARN (P5 coupling not active).

---

# 12. NON-NEGOTIABLE QUESTIONS

1. **Released output without reconstructible origin→terminal?** Yes possible: release validator is post-speech; dashboard chat (when enabled) skips golden/release-validation. Default chat env OFF. L0 still gates spoken chunks.
2. **Retry without lifecycle evidence?** Not seen this bounce. **NOT EXERCISED.**
3. **Memory write without acceptable provenance?** Canonical path stamps provenance. F1 writes never land. Consolidation `storage.add` skips MEMORY_WRITE. Observer/transactions emit without full stamp/quarantine/salience.
4. **Output released without validation?** L0 runs before TTS. Pack 0 without_validation on the 5 counted releases. Validator itself is not a mute.
5. **Dashboard trust certainty on partial panels?** Yes if misread. Reconstructability metadata marks most as partial/non_reconstructable. V2 fail-closed helps. Maturity 5/5 and 8/8 still over-read.
6. **Acquisition/plugin/SI truly governed?** Yes as state machines with human gates. SI apply is not a closed loop. Plugins never left quarantine this window.
7. **Golden deterministic and auditable?** Code yes. Not fired this bounce.
8. **Learning credited on dead pipes?** Distillation sample counts are real JSONL. Policy 0/3 is **not** a dead trainer (JSONL + `.pt` persist) — it is a measurement lock. Voice-intent samples ≠ router. Spark 11/20 is real external answers.
9. **Restart distinguishes current vs ever?** Yes. High-water + pack ever_ok + Oracle restore verified. Policy A/B session counters **do** reset (Pillar 10, not a bug).
10. **End-to-end trace-auditable today?** **No.** Strong on L0 spoken path. Weak on persist split, SI apply, dashboard synthesis, SPARK thought wire.
11. **Every BROKEN opened in source, maturity ruled out?** Yes for F1–F6, F5 dormant, F14 auditor. Pack reds and spark empty tray are **not** BROKEN.
12. **Any Do-Not-Tune proposal?** Removed. Spark 20/0.40, policy 0.55/100, PVL 85%, L3 25/0.50, language n=30, SI stage, native_voice, P2, L3 stay.
13. **Oracle vs pack?** Distinct by Pillar 9. Signage, not a scoring merge.
14. **Unexercised lanes labeled as such?** Golden, SI apply, plugin active, TBS-2, P5b, full pytest isolated, v2 click-through.
15. **Fix plan vs fresh brain?** FIX-A–E add writes/routing honesty without assuming mature counts. FIX-F does not enable Stage 2.

---

# 13. COVERAGE STATEMENT

**Locked pass 2:** turn floor + P1 kinds, memory canonical path **and** named bypasses, persist split, domain-recall steal, identity last-seen, spark promotion **and** tray empty causes, inner SPARK dead context, policy honesty lock, WM vs simulator, hemisphere 8/8 meaning, L0–L12 fire map + formulas, connectome pulse identity, SI stage-0 vs broken approve, plugin ladder, acquisition vs skill-learning dual pipeline, Golden skip-P1, TBS-0, synthetic-vs-lived fences, dashboard reconstructability classes, live probe battery, pack math, Oracle vs PVL, restart auto-restore L2.

**Overnight addendum 2026-08-24 (pid 5680, branch `3541336`):** Pi=senses / brain=VLM room inventory closed. VISION kitchen-lie traced to dinner-chat memory+history, not the camera. See `docs/OVERNIGHT_WIRE_AUDIT-2026-08-24.md`.

**Still not locked:** isolated full pytest, live SI apply, live plugin promotion, live Golden, live compact, browser pixel tour, every EventBus retry, #24 VQA targeting (user question still not the VLM prompt).

That is ground truth for this workshop **as of pid 19147**, not a claim that every line executed.

---

# 14. WHAT TO FINISH (order). No gate tuning.

1. Operator: **no bounce required** for the plugin-registry restore (live API already matched). Optional bounce still picks up any unsynced docs only after `./sync-desktop.sh`.
2. Code (when you say go): FIX-A → B → C → D → E. Spark batch-every-tick is optional fuel, not a gate.
3. Signage: kinds 10, L3 row, grounding view-only copy, shadow docstring.
4. Do **not** flip revoice, native_voice, OSV P2, GroundingDrivePromotion, language bridge, L3, SI stage.
)
