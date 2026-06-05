# Capability Lifecycle: Visible, Governed, Iterative

Status: **design — 2026-06-05**, drafted from David's directive after web_scraping_v1
completed the first full skill-acquisition lifecycle.

The capability-creation lanes (skill acquisition / plugins, Matrix Protocol specialists,
self-improvement) all share one north star: a capability should be **created in the open,
governed against polluting cognition, and improvable forever** — because **one-shots are
rare; there is always a next improvement.** This document is the plan that should have been
surfaced *in the deliberation*, not discovered after the build.

Honesty discipline (`ARCHITECTURE_PILLARS.md`) applies: shadow-first, additive, gated,
human-approved; nothing here lets a capability claim more than its evidence.

---

## Pillar 1 — Visible (the v2 capability dashboard overhaul)

Today the acquisition pipeline's deliberation is invisible (it lived only in `/capability-pipeline`
with no drill-down; the repair-loop rounds, plan, and per-stage diagnostics were not surfaced).
The v2 dashboard `capability.html` gets a **Capability Lifecycle** section with two zoom levels:

- **High level** — one row per in-flight/recent capability across ALL lanes (acquisition,
  matrix, self-improve): name, lane, stage ladder, status, who/what requested it, the
  governance verdict, and the gate it's waiting on. No pivoting to old pages.
- **In the sand (granular drill-down)** — for a selected capability: the full **deliberation**
  — plan (technical_approach / implementation_sketch / test_cases), the **repair_log**
  (think→code→validate rounds, incl. anything it caught — e.g. "rejected a simulated stub"),
  per-stage diagnostics (planning / codegen / activation), the contract smoke results, the
  shadow-observation window, and the approval gates **inline with feedback** (approve-plan /
  approve-deploy / recover, each showing success/failure, not a silent button).

Endpoints largely exist (`GET /api/acquisition[/{id}]`, the approve/recover POSTs, the
`codegen_prompt_diagnostics.repair_log` added this session). This is mostly a **rendering +
consolidation** build, plus surfacing the governance verdict (Pillar 2) and the feedback
control (Pillar 3).

---

## Pillar 2 — Governed (the data-flow firewall, considered IN the plan)

A plugin's output must not silently pollute cognition. For web scraping specifically (David's
spec):

1. **Never auto-save** scraped data to memory/beliefs. (Already the baseline — plugin output
   has no auto-memory path.)
2. JARVIS **asks**: "is this useful — should I save it?" before any write.
3. If yes → write **tagged `web_scrap`** (provenance = untrusted/scraped), never as a plain
   belief.
4. **Cross-validate later** against peer-reviewed / journal sources via the spark / grounding
   ring (curiosity pointed outward to verify a belief vs external truth — see `SPARK_DESIGN.md`).
   A `web_scrap`-tagged claim stays low-trust until an external validator touches it.

**Crucially, this is a planning concern, not a post-hoc patch.** The acquisition planner
should reason about a capability's data-flow/egress and emit a **governance section** in the
plan (what the plugin reads, what it returns, whether its output may touch memory, and the
provenance tag). The dashboard surfaces that governance verdict (Pillar 1). This is the
explicit per-plugin egress policy that `PLUGIN_ISOLATION_DESIGN.md` Gap B called for.

---

## Pillar 3 — Iterative (operator feedback → re-enter → improve)

The repair loop (commit 79f12ce) is automatic feedback *during* a build. Pillar 3 generalizes
it to **post-completion, operator (or system) feedback** across all lanes:

- From the dashboard, on any completed artifact (plugin / Matrix specialist / self-improve
  patch), an operator can **provide feedback** ("summary extraction is weak"; "also handle JS
  sites"; "this specialist mislabels X").
- That feedback **re-enters the relevant pipeline** as revision context:
  - **Plugin** → a new acquisition seeded with the feedback (reuses the plan-revision-context
    path + the repair loop), producing an improved version. Preserves the prior attempt's
    audit trail (Terminal Acquisition Closure).
  - **Matrix specialist** → a re-training pass with the feedback as a labeled signal.
  - **Self-improve patch** → a new patch attempt with the feedback as the issue.
- The improved version goes through the SAME governed lifecycle (shadow → human-approved →
  verified). Trust is never inherited from the prior version.

This is what makes JARVIS a real engineer: it doesn't ship a one-shot and walk away; every
artifact is a living thing that can be told "do better" and will.

---

## Recommended sequence

1. **v2 capability dashboard overhaul** (Pillar 1) — the surface that makes 2 & 3 understandable
   and interactive. Surfaces what exists now (real win, kills the pivoting) and hosts the rest.
2. **Data-flow firewall** (Pillar 2) — planner governance section + the ask-to-save / `web_scrap`
   tagging flow, surfaced in the dashboard. (Pairs with the optional scraper upgrade to a
   Playwright-in-venv plugin for JS sites.)
3. **Feedback → re-iterate loop** (Pillar 3) — start with the plugin lane (extends the existing
   acquisition retry + repair loop), then Matrix + self-improve.

Each ships shadow-first / human-gated and is verified before it claims anything.

---

## Build status (2026-06-05)

**Pillar 1 — Visible: SHIPPED** (commits on `thespark`). The v2 `capability.html` surfaces the
acquisition deliberation high-level + granular: compact rows with a lane ladder; a click-to-open
drill (repair rounds, per-stage diagnostics, lane errors, plan, **governance verdict**); inline
approve/remove; a **pause** toggle so open panels don't collapse on auto-refresh; **active vs
past** split with a collapsible past group; and **terminal-only safe removal** (refuses to remove
a live build; audit trail preserved). Static — browser refresh only.

**Pillar 2 — Governed: SHIPPED (the structural firewall), with two earned/decision items left.**
- *Planned, not post-hoc (2A):* every plugin plan now carries a `governance` section decided
  DURING planning — a **deterministic floor** (`AcquisitionOrchestrator._derive_governance`) infers
  `reads_external / egress / may_touch_memory / provenance_tag / requires_save_consent` from the
  intent/approach so the firewall KNOWS rather than trusting the LLM. The planner prompt also
  reasons about it (a `GOVERNANCE:` section) and that statement is merged in but **can only raise
  caution, never weaken the floor** (`may_touch_memory` stays False; external readers stay
  `web_scrap` + consent-required — unit-verified).
- *Epistemic teeth (2B):* a new low-trust provenance **`web_scrap`**. Before this, scraped web
  findings were written as `external_source`, which earns the **highest** confidence boost (+0.10)
  and counts as *grounded* — so a random website was trusted more than JARVIS's own eyes. Now web /
  unverified findings (and research summaries) are tagged `web_scrap` (no boost; carries grounding
  tension so the grounding ring is drawn to cross-validate it against a peer-reviewed source);
  genuinely citable knowledge (`factual_knowledge` — academic/DOI/peer-reviewed) keeps
  `external_source`. Forward-only: zero existing beliefs reclassified.
- *Surfaced (2C):* the dashboard drill renders the governance verdict as chips.

  **DECISION FOR DAVID (open-question #1 discipline):** `web_scrap` is tracked as its own tier in
  `spark_metrics` and **deliberately NOT folded into the keystone grounded:inferred ratio** —
  whether scraped data should weight grounded/inferred (like the same open question for
  `external_source`) is yours to decide, not a silent default.

  **EARNED / NOT YET BUILT — the ask-to-save consent turn.** Points 1–4 of Pillar 2 are now
  structurally enforced *if* scraped data ever reaches memory (it has no auto path; the one explicit
  path — `KnowledgeIntegrator.integrate`, the research pipeline — now tags `web_scrap`). But the
  conversational "is this useful — should I save it?" turn lives in the companion-cognition loop and
  is **not yet wired**; building it as theater (a fake prompt) would violate the north star, so it's
  flagged as the next real piece. Cross-validation already has its home: `web_scrap` beliefs carry
  tension and zero boost, so the spark/grounding ring (P1 shadow) is the mechanism that will
  eventually confirm/refute them against external truth.

**Pillar 3 — Iterative: not started.**
