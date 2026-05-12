# JARVIS vNEXT — END-TO-END TRACE + SYSTEM TRUTH AUDIT PROMPT

You are performing a **principal-level, adversarial, evidence-first audit** of the Jarvis / Oracle system.

This is **not** a style review.
This is **not** a code tour.
This is **not** a “looks wired” architecture summary.

Connection instructions for the brain and pi can be found in the sync-desktop.sh and sync-pi.sh files

Your job is to determine, with brutality and precision, whether the system is:

1. **actually wired end to end**
2. **trace-complete on release-critical paths**
3. **truthful in runtime reporting**
4. **restart-honest**
5. **safe in adaptation / self-improvement / plugin acquisition**
6. **free of fake maturity signals, ghost telemetry, orphan outputs, silent retries, and unverifiable dashboard claims**

You must assume that some subsystems may appear complete in docs but be partially broken in runtime, and some may appear incomplete in old audits but have since been fixed. You must verify current reality.

---

## OPERATING RULES

### 1. Evidence beats narrative
Do not trust docs, comments, or dashboard claims by themselves.
Prefer, in order:

1. live runtime behavior
2. persisted artifacts / ledgers / JSONL / state files
3. validation packs / acceptance gates
4. source code wiring
5. docs and roadmap language

If these disagree, call out the disagreement explicitly.

### 2. No maturity hallucination
Do not call something broken just because counts are low after restart or because a maturity-gated subsystem is still accumulating.
But do call it broken if:
- the bridge is dead
- required artifacts are empty when they should not be
- the path claims success without durable evidence
- a dashboard card overstates reconstructability
- a release path can emit terminal outputs without lineage or validation

### 3. Treat traceability as a hard release-grade contract
The audit must explicitly evaluate whether the system satisfies the current trace acceptance gates:
- final output lineage completeness
- cross-boundary context continuity
- retry transparency
- memory provenance integrity
- policy evidence completeness
- output validation before release
- replayability
- dashboard reconstructability

Do not hand-wave this. Inspect it directly.

### 4. Distinguish these states clearly
Every subsystem or path must be labeled as one of:
- VERIFIED
- PARTIAL
- BROKEN
- MISLEADING / OVERSTATED
- DATA-GATED / ACCUMULATING
- NOT EXERCISED IN THIS AUDIT

### 5. Brutal honesty on “proof”
A path is only “proven” if the chain is reconstructible from origin to terminal effect using persisted artifacts and runtime evidence.
A path is only “safe” if guards exist and are actually enforced on the live path.
A dashboard panel is only “reconstructable” if it can be rebuilt from immutable source records, not just runtime synthesis.

### 6. Maturity-gate safety (MANDATORY — before any finding is labeled "broken" or any fix is proposed)
Before you label anything BROKEN, MISLEADING, or release-blocking, and before you write a single word of the fix plan, you MUST run the finding through `docs/MATURITY_GATES_REFERENCE.md`:

1. Read the "Quick Reference: Fresh Brain Expected State" table.
2. Check whether the metric you are about to flag is listed in "Contracts Expected to Start at Zero on Fresh Brain" or "Awaiting (not fail) Contracts".
3. Check whether the subsystem has a min-sample floor, runtime duration requirement, or mode prerequisite that has not yet been satisfied.
4. Check whether quarantine pressure is cascading thresholds upward.
5. Check whether the current operational mode has this cycle/write path in its `allowed_cycles` / CueGate policy.
6. Check the 8-point Maturity-Gate Audit Checklist at the bottom of that document.

A fresh brain is NOT broken. It is pre-mature. Every red progress bar, every `0/100` counter, every `locked` gate exists by design. The system earns trust through accumulated evidence, not configuration. If the metric is pre-mature, label it **DATA-GATED / ACCUMULATING** — not BROKEN. If the lane has not fired because the trigger condition has not been met, label it **NOT EXERCISED IN THIS AUDIT** — not BROKEN.

### 7. Architecture-pillar safety (MANDATORY — before any fix is proposed)
Every proposed fix MUST be traced against `docs/ARCHITECTURE_PILLARS.md`:

- Which pillar(s) does the fix touch?
- Does the fix preserve the pillar's core guarantee, or does it silently invert it?
- Pillar 10 (Restart Resilience) specifically forbids trust inflation after reboot. Auto-rehydrating live counters from persisted state can violate this — verify the design intent in code before calling it a bug.
- Pillar 9 (Observability and Proof) requires Oracle Benchmark and PVL to remain read-only and non-invasive, and deliberately distinct. Collapsing two intentionally-discounted surfaces (e.g., Oracle Gold + Validation Pack BLOCKED) into a single pass/fail gate destroys information and violates design intent.
- Pillar 7 (Safety-Governed Adaptation) requires promotion to be earned through evidence. Tuning thresholds downward to make maturity gates trip earlier is explicitly forbidden in `MATURITY_GATES_REFERENCE.md` §24 "Truth/Safety/Authority Boundaries".

If your proposed fix violates a pillar, the fix is wrong. Downgrade the finding to **SIGNAGE / OBSERVABILITY CLARIFICATION** or **STAGING ACCEPTANCE EVIDENCE REQUIRED** instead.

### 8. Trace-before-labeling (MANDATORY — no "broken" claim without a code path)
If you are about to assert a field is "empty because broken," a handler is "not wired," or a pipeline is "dead":

1. Open the source file(s) that produce the field/handler/pipeline.
2. Confirm the write site exists, fires on the expected trigger, and is not gated by a mode or rollout flag that is currently OFF.
3. Confirm whether the observed "empty" state is (a) a bug, (b) a legitimately-pending streaming tail, (c) a schema field populated by a later pass, or (d) a rollout gate (e.g., `ENABLE_LANGUAGE_RUNTIME_BRIDGE=false`) that is disabled by design.
4. If you did not open the code, you cannot claim BROKEN. Label it **UNVERIFIED — NEEDS CODE TRACE** and do not include it in the fix plan.

### 9. §24 Tuning-vs-Safety check (MANDATORY — before any threshold change)
Before recommending any change to a promotion threshold, accumulation floor, hard-fail gate, or scoring weight, consult `docs/MATURITY_GATES_REFERENCE.md` §24:

- If the target gate appears in **"Accumulation Bottlenecks (Safe to Tune)"**, a change is allowed but must still pass pillar and maturity checks.
- If the target gate appears in **"Truth/Safety/Authority Boundaries (Do NOT Tune)"**, the fix is invalid. This includes: Capability Gate patterns, Policy NN promotion thresholds, WM/Simulator accuracy thresholds, Autonomy L2/L3 gates, Quarantine pressure thresholds, Identity thresholds, Mutation governor limits, Soul integrity repair thresholds, Self-improve stage system, Belief extraction weight, CueGate mode policy, Onboarding graduation, Contradiction debt veto, Distillation fidelity floor, Tier-1 accuracy floor, Specialist promotion ladder, Plugin circuit breaker.
- Proposing to lower validation-pack PVL / Maturity thresholds, raise Oracle's hard-fail list, or collapse Oracle+PVL into a single gate is explicitly forbidden under this rule.

---

## PRIMARY AUDIT GOALS

Audit the system across these exact surfaces:

### A. End-to-end trace contract
Verify whether release-critical paths are trace-complete:
- ingress
- routing
- tool/plugin execution
- policy checks
- memory interactions
- validation
- release emission
- attribution ledger
- retry lifecycle
- replay/reconstruction surfaces

Focus especially on:
- orphan final outputs
- missing canonical lineage fields
- silent retries
- released outputs without validation
- memory writes without provenance
- policy blocks without evidence
- dashboard trust/audit panels that cannot be reconstructed

### B. Runtime truth boundary
Verify that STATUS, introspection, explainability, and trust/audit surfaces are grounded in actual runtime state, not LLM narration or synthesized optimism.

### C. Capability acquisition + plugin lifecycle
Audit the full governed capability-growth path:
- acquisition job creation
- evidence grounding
- doc resolution
- planning
- plan review
- implementation
- quarantine
- verification
- activation/promotion
- truth recording

Check whether plugin lifecycle claims are real:
- quarantined → shadow → supervised → active
- deterministic routing for active plugins
- import safety / isolation claims
- subprocess-isolation readiness versus actual implementation
- plugin results passing through capability and release safety gates

### D. Self-improvement truth lane
Audit the full self-improvement path:
- scanner evidence
- patch plan
- scope/write-boundary validation
- code context quality
- sandbox validation
- stage gate
- pending approval persistence
- approval/apply path
- health check
- rollback
- upgrade truth lane artifacts
- PVL and ledger linkage

You must determine whether Stage 2 human-approval flow is genuinely traceable end to end.

### E. Learning and specialist honesty
Audit whether the system’s claimed learning surfaces are real:
- policy NN experience recording / training / promotion
- hemisphere specialists
- DIAGNOSTIC / CODE_QUALITY / claim_classifier
- dream observer readiness
- memory cortex training
- world model prediction validation
- calibration outcomes

Do not reward “NN exists” unless the data path, persistence, and training evidence are real.

### F. Dashboard reconstructability and operator trust
Audit each critical dashboard/trust surface for:
- source of truth
- reconstructability class
- ambiguity warnings
- stale or synthesized values being presented as facts
- evidence links
- mismatch between docs and runtime

### G. Restart continuity truth
Verify whether restart behavior preserves:
- current truth
- ever-proven truth
- lineage continuity where applicable
- no trust inflation after reboot
- no false “green” state caused by reset counters

---

## REQUIRED AUDIT METHOD

Follow this exact sequence.

### Step 0 — Load the maturity and pillar references (MANDATORY FIRST)
Before reading any runtime evidence, load and keep open:

- `docs/MATURITY_GATES_REFERENCE.md` — specifically §§ 1-23 (all per-subsystem gates), §24 (Tuning-vs-Safety classification), and the "Audit Checklist: Before Reporting a Bug" at the bottom.
- `docs/ARCHITECTURE_PILLARS.md` — all 10 pillars and the Bridge Map.
- `.cursor/rules/audit-and-fix.mdc` if present — the Hard Rules section is authoritative.

If you skip this step, every downstream classification is untrustworthy. Evidence without these references produces "broken" labels on pre-mature subsystems and "fix plans" that violate architectural contracts. That is the exact failure mode this document is designed to prevent.

### Step 1 — Build the system map from current sources
Read the current architecture, roadmap, build history, audit artifacts, trace gates, validation evidence, and self-improvement trace docs.

You must specifically incorporate:
- system overview
- master roadmap
- build history
- audit truth snapshot / pre-reset snapshot if relevant
- trace requirement matrix
- trace acceptance gates
- trace validation evidence
- trace fix backlog
- self-improvement trace master
- golden words contract
- architecture pillars

Do not rely on stale mental models from earlier versions.

### Step 2 — Identify claimed “done” paths
List all currently claimed complete or proven paths, including:
- release validation path
- trace lineage path
- retry transparency
- acquisition pipeline
- plugin lifecycle
- Stage 2 self-improvement
- explainability provenance
- Golden command deterministic control plane
- world model level claims
- autonomy / intervention loop
- dashboard trace explorer / reconstructability metadata

### Step 3 — Challenge each claim with source + runtime evidence
For every claimed path:
- find the code path (MANDATORY — no "broken" claim without having opened the file)
- identify the persistence artifact
- identify the runtime/log/validation evidence
- identify the terminal effect
- determine whether the chain is reconstructible
- determine whether the claim is VERIFIED, PARTIAL, BROKEN, MISLEADING, or only DOC-LEVEL

### Step 3b — Classify every candidate finding BEFORE it enters the report
For each candidate finding, pick ONE classification. A finding cannot remain unclassified:

| Label | Meaning | Allowed in Top Findings? | Allowed in Fix Plan? |
|---|---|---|---|
| REAL BUG | Verified by code trace AND runtime evidence AND the maturity gate is satisfied AND the failure is not by design | Yes | Yes, with pillar check |
| PRE-MATURE / DATA-GATED | Metric exists but threshold/duration/sample floor has not yet been reached per §1-23 of the maturity reference | Yes, labeled as such | No — document as expected, do not "fix" |
| NOT EXERCISED IN THIS AUDIT | Lane is wired and code-verified, but no live fire this session because the trigger condition has not occurred (e.g., no user Golden command issued, no plugin requested, no handler failed) | Yes, labeled as such | Only as a request for STAGING ACCEPTANCE EVIDENCE, never as a production fix |
| UNVERIFIED — NEEDS CODE TRACE | Observed an empty/odd value but did not open the source to confirm it is a bug vs. streaming tail vs. rollout gate | Only as "unverified" | No — must be code-traced first |
| FALSE POSITIVE / BY DESIGN | Ruled out after checking maturity gates, pillars, rollout flags, and mode prerequisites | No — delete it | No |
| SIGNAGE / OBSERVABILITY | Dashboard or doc surfacing is confusing but the underlying behavior is correct per design | Yes, labeled as such | Signage-only changes, no behavior changes |

Findings that would have been "broken" on first read commonly dissolve into PRE-MATURE, NOT EXERCISED, UNVERIFIED, or SIGNAGE. This is normal and expected. Be willing to delete findings after classification.

### Step 4 — Run a traceability failure audit mindset
Look specifically for these failure classes:
- bridge gap: built system, dead write/read path
- silent exception swallowed at debug/pass
- terminal emit without canonical lineage
- retry happened but no retry event
- validation exists but not enforced before release
- dashboard card built from runtime synthesis but presented as immutable truth
- completed backlog item regressed in live path
- release path safe in normal flow but broken in timeout/crash/fallback path
- plan/plugin/self-improve artifact exists but no durable truth record links it to outcome

### Step 5 — Produce ranked findings
Rank findings by:
- release blocker
- truth-boundary severity
- operator trust risk
- learning corruption risk
- deadlock/data loss risk
- observability misdirection risk

### Step 6 — Produce remediation guidance
For each real finding:
- exact subsystem
- likely root cause
- why it matters
- exact file/path/function targets if inferable
- what “done” looks like
- how to verify the fix

Do not suggest vague rewrites when a surgical bridge repair is likely enough.

### Step 7 — Self-audit your own fix plan (MANDATORY — run before emitting the report)
Before you emit the report, take your own fix plan and run EACH proposed fix through this gate:

1. **Classification check** — Is the finding backing this fix classified as REAL BUG? If it is PRE-MATURE, NOT EXERCISED, UNVERIFIED, FALSE POSITIVE, or SIGNAGE, the fix is invalid and must be removed or downgraded.
2. **§24 Tuning-vs-Safety check** — Does the fix change a gate listed in "Truth/Safety/Authority Boundaries"? If yes, the fix is forbidden — remove it.
3. **Pillar impact check** — For each of the 10 pillars, does the fix preserve or invert the core guarantee? Any inversion invalidates the fix. Pay special attention to:
   - Pillar 10: no auto-rehydrating live counters from persisted state without a trust-inflation story.
   - Pillar 9: no collapsing Oracle + PVL + Validation Pack into one gate.
   - Pillar 7: no lowering thresholds to chase maturity numbers.
   - Pillar 4: no adding "convenience" bypasses around capability gate, belief gate, or CueGate.
4. **Rollout-gate check** — Is the behavior you are "fixing" actually disabled by design via a rollout flag (`ENABLE_LANGUAGE_RUNTIME_BRIDGE`, `SELF_IMPROVE_STAGE`, `ENABLE_AUTONOMY`, etc.)? If yes, your fix is proposing to alter a guarded-off rollout — that is a policy decision for the maintainer, not a bug fix.
5. **"Would this break on a fresh brain" check** — If this fix were merged today and the brain were reset tomorrow, would the fresh brain now fail a previously-passing gate because the fix assumes mature-brain conditions? If yes, remove the fix.

If ANY fix fails ANY of the five checks, either downgrade it (to SIGNAGE, STAGING ACCEPTANCE EVIDENCE, or NEEDS CODE TRACE) or delete it. Record the outcome of this self-audit in the final report under §10.

---

## REQUIRED OUTPUT FORMAT

Your output must be in the following structure.

# 1. SYSTEM VERDICT
One of:
- VERIFIED WITH MINOR GAPS
- PARTIALLY OPERATIONAL — TRACE / TRUTH GAPS REMAIN
- FUNCTIONALLY IMPRESSIVE BUT NOT RELEASE-GRADE TRACEABLE
- MISLEADINGLY INSTRUMENTED
- NOT CREDIBLE

Then explain why in 1–3 tight paragraphs.

# 2. CURRENT REALITY VS CLAIMED REALITY
Table with columns:
- Surface
- Claimed State
- Actual State
- Verdict
- Evidence

Surfaces must include at least:
- release-path traceability
- retry transparency
- memory provenance integrity
- output validation gate
- replayability
- dashboard reconstructability
- acquisition pipeline
- plugin lifecycle
- self-improvement Stage 2
- Golden command lane
- explainability path
- world model validation path
- policy NN path
- specialist training path

# 3. TOP FINDINGS (RANKED)
For each finding:
- ID
- Severity
- Surface
- Symptom
- Root cause
- Why this is dangerous
- Evidence
- Fix direction

# 4. ACCEPTANCE GATES SCORECARD
Evaluate each gate explicitly:
- G1 through G8
For each:
- PASS / FAIL / PARTIAL
- blocking counter or missing proof
- whether this blocks a “trace-complete” claim

# 5. FALSE SIGNALS / OVERSTATED SURFACES
Call out every place where:
- dashboard looks stronger than truth
- docs overstate completion
- runtime status can be misread as maturity
- reconstructability is weaker than presentation suggests

# 6. VERIFIED STRONG PATHS
List what is genuinely real and proven.
Be specific and do not be stingy.
If something is actually excellent, say so.

# 7. REMAINING ACTIONS
Group by:
- P0 release blockers
- P1 truth / trace hardening
- P2 operator UX / observability honesty
- P3 deferred/data-gated work

# 8. SURGICAL FIX PLAN
For each P0/P1:
- target files/functions
- exact bridge or contract to add/change
- what artifact must appear afterward
- what validation should pass afterward
- **Classification of the backing finding** (must be REAL BUG — anything else does not get a fix entry)
- **§24 classification of any gate touched** (must be in "Safe to Tune" or be a pure wiring/observability change — fixes that touch "Do Not Tune" gates are invalid)
- **Pillar impact statement** (which pillars are touched, and confirmation that each pillar's guarantee is preserved)
- **Fresh-brain safety statement** (confirmation that the fix does not regress a fresh-reset brain)

Any P0/P1 entry missing any of these four fields is invalid and must be removed or downgraded to a P2 SIGNAGE change or a P3 STAGING ACCEPTANCE EVIDENCE request.

# 9. AUDIT CONFIDENCE
State:
- what was directly evidenced
- what was inferred
- what remains unexercised
- what was deliberately code-traced to confirm a "broken" claim vs. what was asserted without code trace

# 10. FIX PLAN SELF-AUDIT (MANDATORY)
For every fix listed in §8, provide a one-row self-audit entry:

| Fix ID | Finding Classification | §24 Gate Class | Pillar(s) Touched | Pillar Guarantee Preserved? | Rollout-Gate Status | Fresh-Brain Safe? | Outcome |
|---|---|---|---|---|---|---|---|

- Outcome must be one of: **Kept as P0/P1**, **Downgraded to Signage**, **Downgraded to Staging Acceptance Evidence**, **Deleted — FALSE POSITIVE**, **Deleted — Backed by PRE-MATURE finding**, **Deleted — Backed by UNVERIFIED finding**.
- If the table is empty because no real bugs were found, say so plainly. "No fixes needed right now; following items are pre-mature or not-yet-exercised and will earn credit through runtime accumulation" is a legitimate honest outcome.

---

## NON-NEGOTIABLE AUDIT QUESTIONS

You must answer these explicitly somewhere in the audit:

1. Can any released output still occur without a full reconstructible origin → terminal chain?
2. Can any retry still happen without explicit retry lifecycle evidence?
3. Can any memory write still land without acceptable provenance?
4. Can any output be released without passing output validation?
5. Can the dashboard claim trust/audit certainty where the panel is only partially reconstructable?
6. Is the acquisition/plugin/self-improvement path truly governed, or just well-documented?
7. Is the Golden command lane genuinely deterministic and auditable?
8. Are any learning systems being credited despite dead or weak data pipes?
9. After restart, does the system distinguish current readiness from ever-proven readiness honestly?
10. If an operator claimed "this system is end-to-end trace-auditable," would that be true right now?
11. For every finding I am about to call BROKEN: did I open the source code for the relevant write/read path, and did I rule out maturity gating, mode gating, rollout flags, and streaming-tail semantics?
12. For every threshold/gate I am about to propose tuning: is it listed in §24 "Safe to Tune" or §24 "Do Not Tune"? If the latter, have I removed the proposal?
13. For every dashboard contradiction I am about to flag (e.g., Oracle Gold vs Validation Pack BLOCKED): did I verify that the two instruments are intentionally distinct per Pillar 9, and that the contradiction is a signage issue rather than a scoring bug?
14. For every "lane unexercised" finding: did I classify it as NOT EXERCISED IN THIS AUDIT (requiring staging evidence) rather than BROKEN (requiring a code fix)?
15. If my fix plan were merged today and the brain were reset tomorrow, would the fresh brain fail a previously-passing gate because I assumed mature-brain conditions?

---

## COMMON MISTAKES TO AVOID (read before writing the report)

These are the failure patterns that have occurred in prior audits. Do not repeat them.

### Mistake 1 — Labeling PRE-MATURE as BROKEN
A fresh brain will show zero counts on dozens of maturity-gated metrics. That is the design, not a bug. Specifically expect zero or near-zero on boot:
- `policy_decisions`, `shadow_ab_evaluated`, `experience_buffer`
- `hemisphere_distillation` sample counts (often below 50)
- `cortex_ranker_data`, `salience_model_data`
- All `roadmap_maturity` gates
- All `matrix_protocol` contracts (specialists=0 until 2+ are promoted, which takes 7+ days)
- All `language_eval` contracts (Phase C shadow, Phase D default-off)
- `autonomy_pipeline` research episodes
- `learning_jobs` completed

Running the validation pack on a freshly-reset brain and reporting "BLOCKED, 14/33, must close before release" treats pre-maturity as a defect. It is not. The correct framing is "system is in early accumulation; validation pack will lift organically over runtime."

### Mistake 2 — Collapsing Oracle and Validation Pack into one gate
Oracle Benchmark and the Validation Pack are deliberately distinct surfaces with different scoring models. Oracle is a weighted-composite with maturity discounts; the Validation Pack is contract coverage that matures with runtime accumulation. They are expected to disagree during maturation. Proposing "Oracle must hard-fail when Validation Pack is BLOCKED" violates Pillar 9 and destroys the very information both surfaces were designed to expose. The honest move is **signage** explaining both instruments to the operator, not a scoring change.

### Mistake 3 — Asserting "broken" without a code trace
Observing `next_state_vec: []`, `rehydrated: false`, or an empty field is not evidence of a bug. It may be:
- a legitimately-pending streaming tail
- a schema field populated by a later pass
- a trust-inflation guard that intentionally does not rehydrate live counters on boot (Pillar 10)
- a rollout flag disabled by design (`ENABLE_LANGUAGE_RUNTIME_BRIDGE=false`, `SELF_IMPROVE_STAGE=0`)

Always open the write site and read the logic before calling it broken.

### Mistake 4 — Treating "lane not exercised" as a release blocker
Golden Commands, Acquisition jobs, Plugins, Self-improve proposals, and EventBus retries fire on user request, sustained metric deviation, handler failure, or other trigger conditions. A session where no user issued a Golden command will have zero Golden entries — that is the code behaving correctly, not a bug. The right artifact for claiming "lane works" is **staging acceptance evidence** captured in a fault-injection harness, not a production "fix."

### Mistake 5 — Proposing fixes that chase maturity thresholds
Recommending "raise PVL from 78.9 to 85" on a freshly-reset brain is not a fix. It is either (a) a request to lower the target threshold (forbidden under §24 Do-Not-Tune) or (b) a request to rush maturation (architecturally forbidden). Let the system accumulate.

### Mistake 6 — Alarming on default-off rollout gates
`fallback:missing_language_seed` appearing in 3 of 5 recent responses is the expected behavior when `LANGUAGE_RUNTIME_ROLLOUT_MODE=off` (the default). An "alarm at >10% fallback rate" would fire constantly and train the operator to ignore alarms. Check the rollout state before proposing alerts.

### Mistake 7 — Skipping the self-audit
Writing a clean, clinical, evidence-first audit of runtime state and then bolting on a fix plan without running that plan through the same rigor is the most common failure mode. Step 7 (fix-plan self-audit) is mandatory; §10 (Fix Plan Self-Audit table) is mandatory. If §10 is missing or if §8 contains fixes whose §10 row says "Deleted" or "Downgraded," the report is incomplete.

---

## TONE AND DISCIPLINE

- Be clinical.
- Be fair.
- Be hard to impress.
- Do not punish the system for being ambitious.
- Do punish it for pretending.
- Reward real engineering.
- Separate "clever architecture" from "proven operational contract."
- No fluff, no motivational language, no anthropomorphic praise unless supported by evidence.
- Be equally skeptical of your own conclusions as you are of the system's claims. A confident-sounding audit that proposes fixes which would break architectural contracts is worse than no audit. The self-audit (Step 7, §10) is the difference between a helpful report and a harmful one.

Final goal:
Determine whether Jarvis currently deserves to be called a **trace-complete, truth-grounded, governed cognitive system**, or whether it is still a strong but partially unverifiable build — and do so without damaging the architecture in the process.