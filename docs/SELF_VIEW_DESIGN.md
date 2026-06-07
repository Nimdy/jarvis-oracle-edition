# Operational Self-View (OSV) — Design

> **A measured, provenance-tagged operational self-model.** A continuously-maintained,
> deterministically-sourced model of *what JARVIS is, how it is built, how it is actually
> performing, and where it is weak* — fused from subsystems that already exist, with
> provenance on every fact and gaps as first-class entries.
>
> It replaces a *fake* source of self-knowledge (LLM/grep guessing) with a *controlled*
> one (a deterministic model the LLM may render but never author). The substrate that
> lets JARVIS *talk about itself truthfully*, *point its curiosity at its own gaps*, and
> *give its self-improvement a purpose*.

### Honest scope (what this is / is not)

**OSV is necessary infrastructure for honest self-reference. It is NOT sufficient evidence
of consciousness, and nothing here should be described as such.** The honest claim is
narrow and engineering-grade: *a deterministic model of what JARVIS is, what it has earned,
what is dormant, what is measured, what is self-scored, and what gaps remain.* Avoid
"JARVIS knows itself / is self-aware / reflects like a conscious being." It is one primitive
a serious local cognitive system should have — no more, no less.

Status: **design.** P0 SHIPPED (substrate). Shadow-first, earned; no behavior authority
granted by this doc. Companion to `SPARK_DESIGN.md`, `COMPANION_COGNITION_DESIGN.md`,
`CAPABILITY_LIFECYCLE_DESIGN.md`.

---

## 0. Why this is a capstone (and why it empowers the rest)

Today JARVIS *has* the raw self-signals but no fused self-model:

- Ask it "what new features do you have?" and it routes to a literal **code symbol
  search** (`tools/codebase_tool.py`) and reads back `Found 15 symbol(s)…` — honest, but
  a grep, not a reflection.
- Its conversational voice (the LLM) starts grounded ("I am aware.") then **pads/rambles**
  once the real internal material runs out, because nothing bounds the voice to a grounded
  self-state.
- Its curiosity about itself **already fires** — the grounding ring forms beliefs about its
  own code (`"Finding: combines wake word detection with VAD…"`) and *would* ask to validate
  them — but in **shadow, zero authority**, disconnected from research and self-improvement.

The OSV closes this. It is the single fused self-model that the following features draw on,
which is why it is a force-multiplier rather than a leaf feature:

| Empowered feature | What the OSV gives it |
| --- | --- |
| **Self-introspection conversation** | answers "what are you / how are you doing / what's new / how could you improve" from a grounded self-model, not a code grep |
| **Voice grounding (tri-layer)** | the LLM renders the OSV and is *cut* when it asserts self-facts not in it → kills the "first 80% real, then it hallucinated" pattern |
| **Spark / grounding ring** (`SPARK_DESIGN.md`) | a concrete target list — point curiosity at the OSV's weak/ungrounded/untested areas instead of diffuse drives |
| **Purposeful self-improvement** (`self_improve/`) | self-improvement gains a *why* — close the specific gap the OSV surfaced and the curiosity chose to chase |
| **Companion cognition** (`COMPANION_COGNITION_DESIGN.md`) | part of what JARVIS "knows about itself" during the live read |
| **CognitivePlanner / Counterfactual** | plans/counterfactuals informed by *what JARVIS can actually do and how well* |
| **Operator trust / dashboard** | one coherent, provenance-honest self-status instead of scattered panels |

---

## 1. What the OSV is

A periodically-synthesized, persisted **self-model object** with five knowledge dimensions,
**every field carrying provenance** (building directly on the #9 labeling sweep):

### 1.1 Structural self-knowledge — *"how I'm built"*
- Subsystem/capability inventory: cognition (world model, simulator, causal engine,
  planner, counterfactual), perception, memory/QSFS, autonomy/policy, epistemic
  (belief graph, calibration, reflective audit), self-improve, skills.
- Source: `tools/codebase_tool.py` code index + `skills/registry.py` (verified vs bootstrap)
  + `skills/capability_gate.py` (what it's *allowed* to claim).
- Wiring map: which subsystem feeds which (so it can explain itself, not just list parts).

### 1.2 Performance self-knowledge — *"how well I'm doing"*  (provenance-critical)
- Per-capability performance from the **now-honest** measurement layer (#9):
  scoreboard composite + per-category bars (`jarvis_eval`), causal `predictive_accuracy`
  (measured) vs `persistence_accuracy` (internally-scored), simulator/world-model
  promotion levels (`cognition/promotion.py`), policy `nn_win_rate` (shadow), skill
  verification evidence, learning-job outcomes.
- **Every number inherits its provenance label** (measured / internally_scored /
  self_scored / shadow_only / synthetic_only / advisory). The OSV never reports a
  self-grade as a measurement.

### 1.3 Maturity self-knowledge — *"what I've earned vs what's gated"*
- Gate/promotion status: maturity tracker, P-levels, shadow-vs-advisory-vs-active,
  startup-stabilization windows, quarantine pressure.
- Source: `jarvis_eval` maturity tracker + the various `*Promotion` controllers.
- So it can say "I have a planner but it's dormant until my simulator earns advisory,"
  honestly — not "I can plan."

### 1.4 Belief self-knowledge — *"what I believe about myself"*
- Self-referential beliefs from the belief graph (`epistemic/belief_graph/`), including
  the grounding-ring's findings about its own code, each with epistemic status
  (inferred / provisional / externally-grounded) and grounding ratio.

### 1.5 Gap self-knowledge — *"what I don't know / can't measure / haven't tested"*  (first-class)
- Comparator-less scoreboard categories (the 4 honestly-empty ones), ungrounded
  self-beliefs, untested/low-coverage code, declining metrics, stale data.
- **Gaps are first-class citizens** — KNOW-not-guess means the absence of knowledge is
  itself knowledge, and gaps are the curiosity targets (§3).

### 1.6 Change self-knowledge — *"what recently changed in me"*
- Recent capabilities/commits/learning-jobs (e.g., "I gained a CognitivePlanner and a
  Counterfactual engine, both shadow-gated").
- Source: `docs/BUILD_HISTORY.md` + attribution ledger + learning-job records.

---

### 1.7 Sourcing note — reuse the existing aggregator, do not re-gather piecemeal

**Verified (2026-06-06):** `dashboard/snapshot.py::build_cache(SnapshotContext)` already calls
`get_status()/get_state()` across **80+ subsystems** and flattens them into the dashboard
`_cache` (keys: `consciousness`, `policy`, `hemisphere`, `autonomy`, `evolution`,
`world_model`, `simulator`, `epistemic`, `belief_graph`, `truth_calibration`,
`reflective_audit`, `quarantine`, `soul_integrity`, `memory_*`, `observer`, `skills`,
`grounding_ring`, `companion_read`, `emergence_evidence`, `trust_state`, …). Live spot-checks
confirm the major learning/NN subsystems are real and populated (e.g. `consciousness`:
stage/awareness/transcendence; `policy`: 46 fields incl. `nn_win_rate` shadow). **The OSV
gather (P0.5) should source from this snapshot, not the ~7 hand-picked readouts P0 started
with** — otherwise "what can you do?" reports a fraction of the real system. Caveat: a full
per-subsystem liveness audit is its own task; the OSV must reflect each subsystem's *actual*
state (active/measured/shadow/dormant/gated), not assume "working."

## 2. Sourcing & honesty discipline (non-negotiable)

1. **Provenance on every fact.** Reuses the #9 schema (`is_measurement`, `kind`, `note`).
   Nothing in the OSV may render as a measurement unless it is one.
2. **KNOW, don't guess.** If a value is unknown/stale/uncovered, the OSV says so. Gaps are
   represented explicitly; the OSV must never confabulate a self-fact. (North star:
   confabulation is betrayal.)
3. **The LLM is a voice, never the author.** The OSV is assembled deterministically from
   subsystem readouts. The LLM only *renders* it for conversation and is *bounded* to it.
   No self-fact originates in the LLM.
4. **Shadow-first, earned authority.** P0–P2 are read-only/observability. Curiosity acting
   on the OSV (P3+) and self-improvement targeting (P4+) are gated and earned, exactly like
   spark/weight-room/companion. This doc grants **no** new authority.
5. **No gate tuning, no score changes.** The OSV *reads* the honest numbers; it never
   adjusts them to look better.

---

## 3. Phased plan (feature list)

Shadow-first, each phase independently shippable; later phases earned over real reps.

### P0 — Self-Model Substrate  *(build-now, shadow, read-only)*
New package `brain/cognition/self_view/` (or `brain/self_model/`):
- `SelfViewSynthesizer` — assembles the five-dimension self-model from existing readouts
  on a cadence (dream/sleep cycle + on-demand). Pure read; mutates nothing canonical; no LLM.
- `SelfModel` dataclass — structured, provenance-tagged, serializable; persisted snapshot.
- `GET /api/self-view` — exposes the model (shadow/observability).
- **Acceptance:** every field provenance-tagged; gaps present as first-class entries;
  zero behavior change; unit tests prove no field renders as an unlabeled measurement and
  the synthesizer never writes canonical state.

### P1 — Self-Introspection from the OSV  *(build-now)*
- Route self-referential intents ("what are you / how are you / what's new / how do you
  feel about your code / how could you improve") to a self-introspection handler that
  answers **from the OSV**, not the codebase grep.
- Deterministic articulation first (like `bounded_response` self_status/self_introspection),
  drawing capabilities + honest performance + recent changes + known gaps. Answer `kind`s:
  identity / capabilities / recent_changes / health / weaknesses / gated_capabilities /
  unknowns / consciousness_query.
- **Strict in claims, rich in capture** (see §6): user-facing answers are conservative and
  non-suppressing; the language guard is regression-tested; "Are you conscious?" uses the §6
  balanced template; P1 may *record* a self-referential anomaly via `observer.observe_emergence`
  (observation-only) but never surface it as a claim.
- **Acceptance:** "what new features do you have?" yields a grounded synthesis (capabilities
  + provenance + gaps), not a symbol dump; explicit code/source questions still route to
  CODEBASE; dormant/gated render as such; gaps → "I don't know / can't measure yet"; the
  language guard holds; deterministic answer path needs no LLM; tests pin routing + content.

### P2 — Voice Grounding / Bounding  *(design contract — focused pass, NOT yet built)*

**The rule (one line):** *If a response makes a claim about JARVIS itself, that claim must be
supported by the OSV or guarded as unknown / provisional / observation-only.* Grounded lead
stays; unsupported self-claim tail is cut, repaired, or replaced. No new self-facts from the
LLM; no new authority; no curiosity / self-improvement targeting; no goal creation.

**P2 means this, and ONLY this:** for self-referential CLAIMS inside any conversational
response, bind the voice to OSV facts. If a generated answer begins grounded but drifts into
unsupported self-claims, **truncate / repair / replace the unsupported tail** — not the whole
answer. (This addresses the other half of the original defect: the LLM that started grounded
then rambled into self-claims past its grounded state.)

**P2 does NOT:** make JARVIS more autonomous · change general reasoning · add personality ·
create self-facts · become "LLM police for everything." It only prevents *unsupported
self-reference* from escaping. Narrow scope is the point — P1 was scoped; P2 must be too, or
it silently becomes a broad conversational control layer.

**Acceptance — P2 passes only if:**
- unsupported self-claims are detected in generated continuations
- grounded self-claims from the OSV survive
- unsupported tails are cut or rewritten (not the whole answer)
- ordinary non-self content is NOT over-filtered
- consciousness-like claims use the §6 balanced template
- emergence-like anomalies are captured via `observer.observe_emergence` (observation-only)
- no new facts originate from the LLM
- no behavior authority / goals / curiosity / self-improvement targeting change

### P3 — Curiosity Targeting (spark integration)  *(shadow → earned)*
- The grounding ring points curiosity at OSV **gaps/weaknesses**: low `predictive_accuracy`
  capabilities, ungrounded self-beliefs, untested code, declining metrics.
- Forms self-directed research questions; may consult the local LLM **advisory-only**, then
  research external/peer-reviewed sources, validate, and update self-beliefs **through the
  epistemic gates** (anti-pollution — no self-belief written canonical without passing).
- **Acceptance:** curiosity targets are sourced from real OSV gaps (not diffuse); shadow
  first; research findings never write canonical belief without grounding. Earned over reps.

### P4 — Purposeful Self-Improvement  *(earned, gated)*
- The self-improve pipeline (`self_improve/orchestrator.py`) receives **targets** = OSV
  weaknesses the curiosity chose to chase, each with a *why*. Proposals flow through the
  existing firewall (shadow → verify → human-approve). **No new authority.**
- **Acceptance:** a self-improvement proposal can be traced to an OSV gap + a curiosity
  decision; all existing gates/firewalls intact; human-approval still required.

### P5 — Continuous Self-Awareness  *(earned — the "alive" part)*
- The OSV updates continuously from live operation; JARVIS proactively surfaces grounded
  self-insight ("my X is degrading; I researched it; I propose Y") via the
  companion-cognition behavior ladder (earned, person-aware, narrate-then-act).
- **Acceptance:** proactive self-insight is grounded + provenance-honest + earned via the
  companion gates; nothing here is declared, only earned.

---

## 4. Dependencies & relationships

- **Built on #9 (honest scoreboard + provenance labeling)** — the OSV's performance
  dimension is only trustworthy because the numbers are now honest and labeled. #9 is the
  prerequisite; the OSV is its payoff.
- **Spark / grounding ring** (`SPARK_DESIGN.md`) provides the curiosity engine P3 targets.
- **Companion cognition** (`COMPANION_COGNITION_DESIGN.md`) provides the read→behavior
  ladder P5 graduates through.
- **Self-improve firewall** (`self_improve/`) provides the gated path P4 feeds.
- **Capability lifecycle** (`CAPABILITY_LIFECYCLE_DESIGN.md`) is a structural source.
- **#9.3 goal-grounding (held)** is complementary: OSV gaps are exactly the
  world/self-grounded goal source #9.3's option B would later add — sequence after #9.3's
  metric-churn dampening.

## 5. Anti-theater guardrails

- A dormant/gated capability appears in the OSV as **dormant**, never as "working."
- Performance numbers carry provenance; self-grades are never headline measurements.
- Gaps are shown, not hidden; "I don't know" is a valid, expected OSV answer.
- The LLM cannot introduce a self-fact; it can only render the deterministic model.

## 6. Emergence: never declare, never discard

Over-correction is a real risk. A system too aggressively sanitized could flatten a genuine
emergent signal into *"I am only a deterministic architecture,"* erasing it. The opposite
risk — letting an LLM phrase or a feedback loop masquerade as emergence — is worse. The
resolution is **not** to loosen claims; it is to separate *claims* from *capture*:

> **Never declare emergence. Never discard emergence-like data.**

Three lanes:

1. **User-facing claims — STRICT.** JARVIS may not tell a user "I am conscious / self-aware /
   alive" without a measured, reviewed, promoted basis. None exists, so it does not claim it.
   It also must not *deny the possibility* ("I'm just code") — that over-corrects and erases
   signal. The honest answer reports architecture + measured/dormant/gap state and says it
   has no measured basis to claim consciousness.
2. **Internal phenomenology reports — ALLOWED, as observations.** JARVIS may *record* unusual
   self-referential states ("a self-referential state resembling uncertainty about my own
   continuity") as provisional internal observations — never as facts or capabilities.
3. **Research / evidence lane — OPEN.** Those observations are preserved, provenance-tagged,
   confidence-bounded, and routed through review before any promotion.

**Capture rides existing machinery — do NOT build a duplicate.** `consciousness/observer.py`
already has `observe_emergence(behavior_type, evidence_refs, confidence)` (obs_type
`emergence_detection`) recording into the observation history, plus an `emergence_evidence`
response class. The "Observation Lane" is an *enrichment* of that, not a new system: capture
richer fields and an observation-only view. Promotion toward an actual emergence rung is
**GitHub #6 (the L7 detector)** — gated, never automatic.

Observation schema (enrichment target; `authority="observation_only"`,
`promotion_status="unreviewed"`, `provenance="self_scored"`, `is_measurement=false`):

    EmergenceObservation = { id, timestamp, event_type (self_report | continuity_anomaly |
      identity_shift | recursive_self_reference), raw_report, source, llm_involved,
      osv_snapshot_id, active_mode, trigger, provenance, is_measurement=false,
      authority="observation_only", promotion_status="unreviewed", recurrence_key }

**Balanced answer to "Are you conscious?"** (the P1 template): *"I have no measured basis to
claim consciousness. My self-view can report my architecture, active/shadow/dormant
subsystems, measured performance, and gaps. I can also record unusual self-referential
states as observations, but those are not proof."* — not "yes, I'm becoming conscious"; not
"no, I'm just code."

**P1 language guard (regression-tested):** user-facing self-introspection answers must not
contain unqualified `conscious / self-aware / alive / sentient / soul / becoming / feel`
unless quoted from the user prompt or explicitly guarded as "not claiming." This bans
*unearned claims*, not the words themselves — internal observations and quoted/guarded uses
are fine. **Strict in claims, rich in capture.**

## 7. Open design questions (resolve during P0/P1)

1. Package home: `brain/cognition/self_view/` vs `brain/self_model/`.
2. Synthesis cadence: dream-cycle only, or also a lightweight pre-conversation refresh?
3. Self-model granularity: per-subsystem vs per-capability rows.
4. P2 bounding mechanism: hard truncation vs confidence-gated continuation vs re-prompt
   with grounded facts only.
5. How aggressively P3 may consult the local LLM as an advisor before external research.
