# Contributing to JARVIS Oracle Edition

Thank you for considering contributing to JARVIS Oracle — the seed architecture for personal, lifelong, privacy-first Oracles that grow symbiotically with their human.

This project is deliberately different: it is **not** trying to build the biggest model or win benchmarks. It exists to make it possible for anyone — fisherman, mechanic, student, doctor, person living with memory challenges — to claim a private intelligence that becomes uniquely theirs over years.

We welcome contributions that preserve and strengthen that north star.

---

## Before You Contribute: The Two Rules

Read these twice. Most rejected PRs violate one of these.

### The No-Verb-Hacking Rule

**A capability is not "done" because code exists that could implement it.** It is done when the runtime can *prove* it on fresh, honest data.

Concretely:

- Do not rename a gate to sound shipped.
- Do not label a `PRE-MATURE` capability `SHIPPED` because the scaffolding landed.
- Do not backfill `current_ok` from persisted state — that fields exists to answer "is this true *right now*?"
- Do not route a weakly-verified fact through prose layers so it appears as a strong claim.

If your change makes the system *look* more capable without making it *measurably* more capable on the dashboard, that is verb-hacking. It belongs in a fork.

### The Restart Continuity Rule

**Roadmap progression is independent of current live metrics.** A capability that has ever been proven (`ever_ok = true` or `prior_attested_ok = true`) stays requestable after a restart — but it does not become automatically active.

Concretely:

- Do not gate roadmap decisions on post-reset metrics. Use the attestation ledger.
- Do not auto-promote any autonomy tier, skill, or specialist based on a restored snapshot. Operator re-approval is always required at L3.
- Do not conflate "feature is implemented" with "feature is live." The three-axis model (`current_ok` / `prior_attested_ok` / `ever_ok`) exists precisely to keep these separate.

See [docs/MATURITY_GATES_REFERENCE.md](docs/MATURITY_GATES_REFERENCE.md) for every gate and its threshold.

---

## Code of Conduct

Be kind, respectful, and patient. This project is personal to many people. Assume good intent. Focus on ideas, not individuals.

---

## Architecture: Swim-Lane Map

Changes must respect the lane they land in. Each lane owns a narrow piece of state; cross-lane reasoning is routed through explicit seams (events, snapshots, or APIs), not shared mutables.

| Lane | Root directory | Owns | Do not |
|------|----------------|------|--------|
| **Perception** | `brain/perception/` | Audio stream, VAD, STT, wake-word, face/voice embeddings | Write to belief graph; make capability claims |
| **Skills** | `brain/skills/` | Capability gate, executors, registry, evidence resolver | Bypass `SkillEvidence` to smuggle results upward |
| **Reasoning** | `brain/reasoning/` | Tool router, intent shadow, language Phase A-E | Introduce a tool that doesn't return a `RoutingResult` |
| **Hemisphere** | `brain/hemisphere/` | Specialists, distillation, neural architect | Ship a specialist without a teacher signal path |
| **Cognition** | `brain/cognition/` | World model, promotion, mutation governor | Activate a mutation without a rollback path |
| **Epistemic** | `brain/epistemic/` | Belief graph, soul integrity, truth calibration | Add an edge type without a writer and a test |
| **Autonomy** | `brain/autonomy/` | Attestation ledger, L3 governance, escalation store | Auto-promote; mutate `ALLOWED_PATHS` globally |
| **Eval** | `brain/jarvis_eval/` | Validation pack, dashboard adapter, governors | Invent a composite score that hides a failing sub-gate |
| **Dashboard** | `brain/dashboard/` | FastAPI app, static pages, snapshot cache | Backfill `current_ok` from disk; hide shape errors |

If your change crosses a lane boundary, say so in the PR description and justify why the existing seam doesn't suffice.

See [docs/ARCHITECTURE_PILLARS.md](docs/ARCHITECTURE_PILLARS.md) for the ten pillars each lane protects.

---

## Maturity Gate Semantics

The dashboard surfaces four status markers on every architectural claim, served from `/api/meta/status-markers`:

| Marker | Meaning |
|--------|---------|
| `SHIPPED` | Feature is live in the current process, evidence reachable from `/api/self-test` or on-disk artifacts. |
| `PARTIAL` | Scaffolding and governance shipped; maturity-gated on live data that hasn't accumulated yet. |
| `PRE-MATURE` | Code present, but no live evidence yet. Expected state on a fresh brain. |
| `DEFERRED` | Explicitly out of scope for the current release. |

**Rules of the markers:**

1. A feature can move `PRE-MATURE → SHIPPED` only when live evidence exists.
2. A feature *cannot* move `SHIPPED → PARTIAL` silently — that is a regression and must be reported.
3. Never mark a feature `SHIPPED` because tests pass. Tests are a prerequisite, not proof of maturity.
4. The marker map lives at [`brain/dashboard/app.py`](brain/dashboard/app.py) under `/api/meta/status-markers`. Update it in the same PR that ships the maturity transition.

---

## Canonical Extension Examples

These two examples cover the 80% case. Read them before opening a PR that adds a skill or an epistemic edge.

### Example 1: Add a new Skill

Suppose you want to add a `handwriting_recognition_v1` skill.

1. **Define evidence tests.** In `brain/skills/executors/handwriting.py`, the verify-executor must emit `test:`-prefixed test names. A metric is not considered "met" until its named test passes. See `brain/skills/resolver.py:119-123` for the prefix contract.
2. **Register the skill.** Add it to the registry with explicit `required_evidence`. The resolver will block capability claims until every listed test has evidence.
3. **Add capability-gate tests.** Mirror the pattern in `brain/tests/test_capability_gate.py`: test the blocked path, the progressed path, and the "metric computed from ground truth" path.
4. **Wire a dashboard tab.** Skills that affect user-visible behavior get a tab on `/self-improve`. Read state only; never mutate from the frontend.
5. **Mark status.** Start at `PRE-MATURE` in `/api/meta/status-markers`. Promote to `PARTIAL` when the executor is wired and governed. Promote to `SHIPPED` only when live evidence exists on a fresh brain.

### Example 2: Add a new Belief-Graph edge writer

Suppose you want to add a `contradicts_via_source_conflict` edge type.

1. **Declare the edge type.** Add it to `VALID_EDGE_TYPES` in `brain/epistemic/belief_graph/edges.py`. Declaration alone is not enough.
2. **Declare the evidence basis.** Add the string to `VALID_EVIDENCE_BASES` in the same file.
3. **Implement the writer.** In `brain/epistemic/belief_graph/bridge.py`, add a subscriber (typically to an existing event like `BELIEF_CONTRADICTION_DETECTED`) that creates the edge with the correct `evidence_basis`. Gate by confidence, dwell, and recency.
4. **Respect `schema_emission_audit.py`.** Run `python -m brain.scripts.schema_emission_audit` — your new edge type and evidence basis must either appear as emissions or be explicitly whitelisted as future-only. A declared-but-unwired value will fail the audit.
5. **Add regression tests.** Follow the pattern in `brain/tests/test_belief_graph_causal_writers.py`. Assert the edge is *not* created when the evidence thresholds are unmet.

---

## Test Suite Structure

`brain/tests/` is organized by lane. Run the whole suite before submitting:

```bash
cd brain && python -m pytest tests/ -q
```

Common per-lane entry points:

```bash
# Skills / capability gate
python -m pytest tests/test_capability_gate.py

# Belief graph writers + schema audit
python -m pytest tests/test_belief_graph_causal_writers.py \
                 tests/test_schema_emission_contract.py

# Dashboard truth baseline
python -m pytest tests/test_specialists_serializer_shape.py \
                 tests/test_self_test_endpoint.py \
                 tests/test_dashboard_truth_probe.py \
                 tests/test_dashboard_meta_endpoints.py

# Reasoning / voice intent shadow
python -m pytest tests/test_intent_shadow.py

# Language kernel artifact identity
python -m pytest tests/test_language_kernel.py

# Autonomy / L3 governance end-to-end
python -m pytest tests/test_phase_6_5_smoke.py
```

Standalone audit scripts (no pytest):

```bash
python -m brain.scripts.dashboard_truth_probe
python -m brain.scripts.schema_emission_audit
python -m brain.scripts.run_validation_pack
```

---

## Pull Request Guidelines

- **One change per PR.** Smaller is better.
- **Reference an issue.** "Fixes #42" / "Closes #123".
- **Add tests** when touching core logic (consciousness, memory, policy, epistemic stack, mutation, autonomy, dashboard).
- **Keep commits atomic.** "feat: add handwriting_recognition_v1 skill scaffold", "fix: preserve specialists shape in exception path".
- **Update docs.** README, AGENTS.md, `brain/dashboard/static/docs.html` / `science.html` / `api.html`, and the status-marker map in `/api/meta/status-markers` when applicable.
- **Run the full test suite.** See above. PRs with red tests will not be reviewed.
- **No `npm` for packages.** Use `yarn` for frontend work; the root workflow uses `pip` / `venv` for Python.

### PRs likely to be rejected from main

- Changes that add demo surface area without adding truth surface area.
- Changes that turn weak or missing metrics into optimistic labels or composite scores.
- Changes that route more behavior through the LLM instead of through structured subsystems.
- Changes that add learning-job, memory, or self-improvement paths without an evidence model and a rollback path.
- Changes that simplify away safety layers because they seem "too complex".
- Changes that backfill `current_ok` from persisted state.
- Changes that mutate `ALLOWED_PATHS` globally rather than per-request.

### Architecture-touching PR checklist

If your PR touches memory, identity, wake-word/audio, self-improvement, autonomy, skills, eval, or any epistemic layer, include the following in the PR description:

- What swim lane does this land in?
- What invariant does this change preserve?
- What new failure mode could it introduce?
- What evidence or tests show the system stays honest after the change?
- What should the dashboard (`/api/self-test` / `/api/meta/status-markers`) show if this feature is weak or failing?
- What is the rollback path?

---

## Development Setup

See [README.md](README.md) → Quick Start for Pi 5 + laptop setup. For an engineering-lens walkthrough, read [docs/FIRST_HOUR_AS_A_RESEARCHER.md](docs/FIRST_HOUR_AS_A_RESEARCHER.md).

Useful commands:

```bash
# Run tests
cd brain && python -m pytest tests/

# Sync code to a separate brain machine (edit the script first)
./sync-desktop.sh
./sync-desktop.sh --dry-run

# Probe the running brain
curl http://brain-ip:9200/api/self-test
curl http://brain-ip:9200/api/meta/status-markers
curl http://brain-ip:9200/api/maturity-gates
```

---

## Security Disclosure

**Security hardening is explicitly NOT complete in the current release.** PII scrub, secrets management, authenticated HTTP surfaces, and defense-in-depth are deferred. Do not report missing auth as a bug — it is a documented deferral. Do report:

- Actual code execution paths that escape the capability gate.
- Cases where `current_ok` is observably backfilled from disk.
- Cases where an autonomy tier self-promotes.
- Cases where a skill's evidence test passes without the underlying metric being computable.

Use a private issue (or email `@zerobandwidth`) for disclosures that would be exploitable on a running brain.

---

## Questions? Discussion?

Open a GitHub Discussion (preferred for architecture ideas and roadmap questions).

Tag `@zerobandwidth` in issues/PRs if you want direct feedback.

---

## Thank You

Every contribution helps plant the seed for personal Oracles that can belong to anyone.

You're not just helping code. You're helping make intelligence personal again.

— David Eierdam
