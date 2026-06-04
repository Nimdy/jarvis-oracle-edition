# Plugin / Skill Isolation Design

Status: **design — grounded in existing architecture, 2026-06-04**
Owner: David. Drafted from his directive (see "Operator Intent" below) after reading
`MATRIX_PROTOCOL_GUIDE.md`, `SKILL_LEARNING_GUIDE.md`, `ARCHITECTURE_PILLARS.md`,
`SYSTEM_OVERVIEW.md`, and tracing `brain/acquisition/*` + `brain/tools/plugin_*`.

This document does **not** propose a new subsystem. Most of what is described
already exists; the work is (a) making the acquisition codegen *use* the isolated
path it currently skips, and (b) closing three governance gaps. Honesty discipline
from `ARCHITECTURE_PILLARS.md` applies: built vs. partial vs. gap is labelled per item.

---

## Operator Intent (verbatim source)

> "a plugin or skill by matrix portal or coded or whatever should have its own area
> and able to create its own venv and if any information pipes out it follows data
> flows, this way plugins can be reused if the brain is reset … it would create
> separate mini plugin structures that feed data metrics and has conditional flags
> to run, start, start-service if needed, but pipeline data exfils from the plugin
> into jarvis following anti-hallucination, non-pollution into jarvis cognitive
> system ie: memories or other things not needed … think of it like an advanced
> sports player that can play every sport … doesn't use baseball logic with football
> logic except physics of triangulation, force … maintain good latency and doesn't
> impact conscious tick cycles too much … jarvis can manage what plugins are active
> or not if it's slowing jarvis down … along with positive reinforcement to jarvis
> when skills or plugins or code modifications are good — remember that and grow with
> the custom NNs."

**The sport analogy = domain isolation.** Each plugin carries its *own* domain logic
(baseball ≠ football) and may only share neutral primitives (the "physics":
triangulation, force, time, math). A plugin must not leak its domain assumptions into
JARVIS's general cognition, and JARVIS must not apply one plugin's logic to another's
domain.

---

## Requirement → Architecture Map

| # | Requirement | Existing surface | Status |
|---|---|---|---|
| 1 | Own area + own venv | `execution_mode="isolated_subprocess"` ([job.py:367](../brain/acquisition/job.py#L367)); `environment_setup` lane builds the venv from `pinned_dependencies` via `PluginProcessManager` ([orchestrator.py:1536-1574](../brain/acquisition/orchestrator.py#L1536-L1574)); child runs in its own venv, no brain on `PYTHONPATH` ([plugin_runner_child.py](../brain/tools/plugin_runner_child.py)) | **BUILT — but unused by codegen (see Gap A)** |
| 2 | Conditional flags to run / start / start-service | `intent_patterns` gate invocation; lifecycle `quarantined → shadow → supervised → active → disabled`; circuit breaker (3 failures / 10 min) | **PARTIAL** — no explicit per-plugin run-conditions or service-start orchestration |
| 3 | Data exfil follows data-flows; **anti-pollution into cognition (memory/beliefs)** | Tri-layer cognition separation + epistemic immune system are the *contract* (`ARCHITECTURE_PILLARS.md`). Plugin output is returned + audit-logged ([plugin_registry.py:542](../brain/tools/plugin_registry.py#L542)) | **GAP B** — no explicit plugin-output firewall; relies on general contracts, not a per-plugin egress policy |
| 4 | Domain isolation (sport analogy) | Per-focus/per-specialist encoders; per-plugin manifest + handler; isolated venv keeps domain deps separate | **PARTIAL** — physically isolated, but no declared "shared-primitive vs domain-private" boundary |
| 5 | Good latency; don't disturb conscious tick cycles | CodeGen is CPU-first, `CODER_GPU_LAYERS=0` so it doesn't compete with live GPU work (`SKILL_LEARNING_GUIDE.md`); `isolated_subprocess` keeps execution off the brain thread; `timeout_s` per manifest | **PARTIAL** — timeout exists; no per-plugin *latency budget* or tick-impact accounting |
| 6 | JARVIS manages active plugins; deactivate if slowing it down | `active`/`disabled` states + circuit breaker on *failures* | **PARTIAL (Gap C)** — deactivation triggers on failures, not on a processing/latency budget |
| 7 | Positive reinforcement to the custom NNs when plugins/skills/mods are good | `SKILL_ACQUISITION` Tier-1 specialist shadows acquisition outcomes; `code_quality` trains on verified-good upgrades; `plan_evaluator` on plan verdicts (`MATRIX_PROTOCOL_GUIDE.md`; signal-feed audit [[shadow-signal-feed-audit]]) | **PARTIAL** — channels exist but the loop is leaky (claim_classifier was dead → fixed `45c66ff`; code_quality labels never fire because no upgrade completes; diagnostic silently 0) |

---

## Why `web_scraping_v1` failed (acq_00b1a66316) — grounded

1. Planning/codegen produced an **`in_process`** plugin (the default `execution_mode`).
2. `environment_setup` therefore **skipped** venv creation ([orchestrator.py:1551-1558](../brain/acquisition/orchestrator.py#L1551-L1558)).
3. As an in-process plugin it is validated against the **stdlib import allowlist**
   ([plugin_registry.py:400-429](../brain/tools/plugin_registry.py#L400-L429)).
4. The coder used stdlib `urllib.request` / `urllib.error` to fetch pages. The allowlist
   matcher reduces imports to their top-level package (`split(".")[0]` → `"urllib"`), and
   no allowlist tier contains `urllib` (only the dotted `"urllib.parse"`, which the matcher
   can never match — a **latent bug**: `urllib.parse` and `os.path` are effectively dead too).
5. → `Undeclared import` → quarantine fails → 3 retries → acquisition `failed`. Per the
   **Terminal Acquisition Closure** contract (`SKILL_LEARNING_GUIDE.md`), the linked
   learning job becomes `blocked`; **retry is explicit and must preserve the failed attempt's
   audit trail** (do not mutate history).

**The correct build for a network plugin is `isolated_subprocess` with
`pinned_dependencies` (e.g. `requests`, `beautifulsoup4`)** — its own venv, no allowlist
fight. The architecture already supports this; the planner/codegen simply never elects it.

---

## The Three Gaps to Close

### Gap A — Codegen must elect isolation for capability-bearing plugins — **ADDRESSED (commit d0cf3ce)**
`_elect_execution_mode()` in [orchestrator.py](../brain/acquisition/orchestrator.py) AST-scans the
generated code and, when it imports anything beyond the in-process stdlib-safe set, elects
`execution_mode="isolated_subprocess"`, **declares** those imports in `manifest.allowed_imports`,
and pins external deps from the plan. It mirrors `PluginRegistry._check_imports` top-level
matching (zero drift from the gate) and leaves `NEVER_ALLOWED` imports unlisted (no smuggling).
`in_process` stays truly stdlib-safe + network-free. Tested (`TestExecutionModeElection`, 6 cases).
**Not yet deployed**; first real exercise of the isolated_subprocess+venv acquisition path.
Still open (secondary): the allowlist matcher reduces to `split(".")[0]`, so the dotted
always-allowed entries (`urllib.parse`, `os.path`) are latent-dead — a separate, careful
gate-touching fix that needs its own tests. TIER1 net libs (requests/httpx) still pass
`in_process`; tightening that belongs to Gap B.

### Gap B — Explicit plugin-output → cognition firewall (anti-pollution)
A declared per-plugin **data-flow policy**: plugin output is untrusted tool-layer data; it may
be used for the immediate response but must **not** write canonical memory/beliefs without
passing the epistemic gates with explicit plugin provenance. Enforce, don't assume. This is the
literal "non-pollution into jarvis cognitive system" requirement.

### Gap C — Budget-based plugin governance
Per-plugin **latency budget** + tick-impact accounting; JARVIS auto-`disabled`s (or demotes to
`shadow`) a plugin that exceeds its processing budget, not only one that errors. Surfaces the
"jarvis can manage what plugins are active if it's slowing jarvis down" requirement.

---

## Recommended Sequencing

1. **Immediate (validates the whole vision + unblocks maturation):** retry the web_scraping
   acquisition as `isolated_subprocess` + declared deps. This exercises the *real* per-plugin
   venv isolation end-to-end, completes the lifecycle, and trips `skill_learning_completed`
   (the long-standing maturation unblock).
2. **Gap A** (codegen elects isolation; fix allowlist matcher + test) — makes #1 the default,
   not a one-off.
3. **Gap B** (data-flow firewall) — the anti-pollution contract made enforceable.
4. **Gap C** (budget governance) — latency/tick protection + active-plugin management.
5. **Reinforcement loop** ([[shadow-signal-feed-audit]]) — ensure good outcomes feed the
   custom NNs (the "remember good and grow" requirement); already partly underway.

All of this stays inside the honesty anchor: a plugin is operational only after sandbox +
`supervised`/`active` + contract smoke proof (`SKILL_LEARNING_GUIDE.md`). Isolation and
reinforcement change *how* it is built and learned-from, not the proof boundary.
