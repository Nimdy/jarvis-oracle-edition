# First Hour as a Researcher

> **Primary reading surface:** The dashboard `/learning` page now consolidates
> the operator and researcher first-hour path for open-source release. This
> Markdown file remains source/reference material and should not be deleted until
> a later archive pass.
>
**Audience:** engineers, AI researchers, and curious operators who want to understand JARVIS Oracle Edition from the inside — not the marketing version.

**Partner doc:** [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md) is the operator-facing first-run ceremony. This doc is the **engineering-lens** counterpart: the endpoints, files, and audit surfaces that let you verify the system yourself.

If you skim only one section, skim [Section 3 — The Three-Axis Cache](#3-the-three-axis-cache).

---

## 0. Mindset before you start

JARVIS is **architecturally complete, runtime-maturing**. A fresh brain is not broken; it is pre-mature. The first hour's goal is not "make it smart". It is to **verify the truth substrate is honest**.

Three mantras carry you through:

- **Truth before capability.** Every claim is evidence-gated.
- **Live over persisted.** `current_ok` never comes from disk.
- **Proof over promotion.** Autonomy and specialists never self-promote.

---

## 1. Boot the brain and observe gestation (~0–2 hours)

A freshly-reset brain enters **gestation**: it studies itself, seeds a personality, builds a foundation of facts, and discovers its own architecture. Let it run. Do not talk to it yet.

While it gestates, open the dashboard and watch these panels move:

- `http://brain-ip:9200/` — overall cockpit
- `http://brain-ip:9200/self-improve` — specialists, distillation, scanner

You will see:

- Consciousness stage climb from `nascent` to `reflective` to `integrative`.
- Teacher queues fill; student networks register.
- The World Model record its first predictions.

**Do not promote autonomy during gestation.** The promotion gates are designed to refuse, but running the synthetic exercise or pressing "Stage 2" on self-improvement will waste the identity baseline.

---

## 2. Four endpoints to probe immediately

These are the four endpoints that define the truth surface. Memorize them.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/self-test` | Consolidated verdict. Cache ready? Engine alive? Serializer shape intact? Attestation consistent? Validation pack green? |
| `GET /api/meta/status-markers` | Single authoritative `SHIPPED` / `PARTIAL` / `PRE-MATURE` / `DEFERRED` map. The dashboard prose pages render *every* architectural claim from here. |
| `GET /api/meta/build-status` | Per-page freshness: on-disk mtime vs. process start time. The `j-freshness-banner` uses this to catch stale code. |
| `GET /api/maturity-gates` | Structured parse of [`docs/MATURITY_GATES_REFERENCE.md`](MATURITY_GATES_REFERENCE.md) with a live overlay of current gate values. |

```bash
curl -s http://brain-ip:9200/api/self-test | jq
curl -s http://brain-ip:9200/api/meta/status-markers | jq .markers
curl -s http://brain-ip:9200/api/meta/build-status | jq
curl -s http://brain-ip:9200/api/maturity-gates | jq '.sections | length'
```

Expected on a fresh brain:

- `/api/self-test`: `ok=false`, `status="not_ready"` for the first few seconds, then `ok=true` with `validation_pack.status="mature"` *or* `"blocked"`. Blocked is fine — it means the system is honest about what hasn't been proven.
- `/api/meta/status-markers`: most entries `SHIPPED` for observability and governance, `PARTIAL` or `PRE-MATURE` for specialists and world-model gates.
- `/api/meta/build-status`: `process_start_time` close to `now`; page mtimes in the past.

---

## 3. The Three-Axis Cache

The single most important invariant in the codebase: **every governed capability is described by three separate fields**, never collapsed into one "is this true?" flag.

| Field | Meaning |
|-------|---------|
| `current_ok` | Is the capability provable from live, in-memory runtime state *right now*? |
| `prior_attested_ok` | Has an operator-accepted attestation (hash-attested ledger record) ever established this capability? |
| `ever_ok` | Has the live system, at any point in this process's history, produced `current_ok=true`? |

From those three, two derived fields:

| Field | Meaning |
|-------|---------|
| `request_ok` | Can an operator request promotion? Equivalent to `current_ok or prior_attested_ok`. |
| `activation_ok` | Is the capability actually active in the current process? |

Read that table twice. Every rejected "why is this green / why is this red?" issue traces to collapsing these.

**Probe the autonomy axis:**

```bash
curl -s http://brain-ip:9200/api/autonomy/level | jq
```

On a fresh brain you should see `current_level=2`, `current_ok=false`, `prior_attested_ok=false`, `request_ok=false`, `activation_ok=false`. That is correct — promotion is gated by both live metrics and operator approval.

**Invariant you should assert yourself:**

```bash
# current_ok must NEVER be true when the live policy has not recomputed it.
# If you find a case where current_ok=true but the autonomy orchestrator has
# not evaluated check_promotion_eligibility() this session, that is a bug.
curl -s http://brain-ip:9200/api/full-snapshot | jq '.autonomy'
```

---

## 4. Three audit scripts worth running

Each is read-only, each catches a different class of silent drift.

```bash
# 1. Dashboard truth probe — walks /api/full-snapshot and flags
#    empty-where-data-exists, shape violations, and stale assumptions.
python -m brain.scripts.dashboard_truth_probe

# 2. Schema emission audit — every declared edge type, evidence basis,
#    hemisphere focus, and teacher key must either be emitted in code
#    or explicitly whitelisted as future-only. Catches "declared but
#    unwired" maturity lies.
python -m brain.scripts.schema_emission_audit

# 3. Validation pack — full runtime verification of claim consistency,
#    cache shape, attestation ledger, soul integrity, and the ten pillars.
python -m brain.scripts.run_validation_pack
```

If any of these is red on a freshly-synced brain, that is a bug worth reporting — not a maturity gate.

---

## 5. The dashboard pages, in the order to read them

1. **`/`** — Live cockpit. Read the top System Truth Score strip first. It aggregates self-test + autonomy three-axis + status-marker counts into a single line.
2. **`/maturity`** — Live mirror of `docs/MATURITY_GATES_REFERENCE.md` with current gate values inlined. Tells you exactly what has to happen for a `PRE-MATURE` gate to become `SHIPPED`.
3. **`/self-improve`** — Specialists, distillation, scanner, L3 governance tab, Voice Intent Shadow tab, Phase E Kernel tab. This is where the self-modification pipeline lives.
4. **`/science`** — The math and architecture behind every NN, gate, and scoring formula. Status markers show you which sections describe shipped code vs. pre-mature scaffolding.
5. **`/docs`** — System reference: every component, data flow, and invariant.
6. **`/history`** — Build history rendered from `docs/BUILD_HISTORY.md`.
7. **`/api-reference`** — Every REST/WebSocket endpoint.

Each page carries a `j-freshness-banner` that lights up if the page's source is newer on disk than the running process. If you see that banner, click Restart before drawing conclusions.

---

## 6. How to extend safely

- To add a new **skill**, follow the canonical example in [CONTRIBUTING.md](../CONTRIBUTING.md).
- To add a new **belief-graph edge writer**, follow the canonical example in the same file. Never add an edge type to `VALID_EDGE_TYPES` without a writer — the schema audit will fail.
- To add a new **specialist**, wire a teacher signal first, register under the distillation adapter, and start at `PRE-MATURE`. Let live evidence promote it.
- To add a new **dashboard claim**, tag it with a `data-status-marker="..."` attribute and update `/api/meta/status-markers` in the same PR. Untagged claims will render as `UNKNOWN` (a regression in the truth surface).

---

## 7. How to read a failure

Before filing an issue, answer these four questions:

1. **Which swim lane?** Perception, Skills, Reasoning, Hemisphere, Cognition, Epistemic, Autonomy, Eval, or Dashboard? (See the table in [CONTRIBUTING.md](../CONTRIBUTING.md).)
2. **Which invariant?** Which of the ten pillars is the apparent failure violating? (See [ARCHITECTURE_PILLARS.md](ARCHITECTURE_PILLARS.md).)
3. **What does `/api/self-test` say?** If self-test is green and the symptom persists, this is an observability gap — the most interesting kind of bug.
4. **Is this a maturity gate, not a bug?** Consult `/api/maturity-gates`. A fresh brain should have `PRE-MATURE` markers on most specialists and promotion gates. That is expected.

If you still believe it's a bug, open an issue with the outputs of:

```bash
curl -s http://brain-ip:9200/api/self-test
curl -s http://brain-ip:9200/api/meta/status-markers
python -m brain.scripts.dashboard_truth_probe
```

---

## 8. What to resist in the first hour

- **Don't promote autonomy manually.** The refusal is a feature.
- **Don't run the synthetic exercise before Stage 2.** It contaminates the baseline.
- **Don't treat a `PRE-MATURE` marker as a bug.** Check the maturity-gates page first.
- **Don't edit prose pages to reword claims.** If a claim is wrong, the fix is almost certainly in `/api/meta/status-markers` or in the underlying subsystem — not in the HTML.
- **Don't reach for the LLM.** Almost everything interesting happens in structured subsystems. If you find yourself routing through the LLM to "understand" a cache field, you are probably about to introduce a verb-hack.

---

## 9. Where to go next

- [AGENTS.md](../AGENTS.md) — the field-manual for agents working in this codebase.
- [docs/MASTER_ROADMAP.md](MASTER_ROADMAP.md) — what is SHIPPED, PARTIAL, PRE-MATURE, and DEFERRED, with pointers to the evidence.
- [docs/ARCHITECTURE_PILLARS.md](ARCHITECTURE_PILLARS.md) — the ten non-negotiables.
- [docs/SYSTEM_TRUTH_AUDIT_PROMPT.md](SYSTEM_TRUTH_AUDIT_PROMPT.md) — the prompt used when doing a full-system truth audit.
- [docs/WAKE_RELIABILITY_TUNING.md](WAKE_RELIABILITY_TUNING.md) — operational playbook for wake-word tuning (evidence-gated).

Welcome. Ask harder questions than you expect the system to answer comfortably — that is the only way you find the seams that still need work.

— David Eierdam
