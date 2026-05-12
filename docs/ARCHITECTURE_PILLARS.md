# Jarvis Architecture Contracts and Open-Source Validation

Last updated: 2026-05-10

This file defines the non-negotiable architecture contracts for Jarvis and how to
trace/validate each contract in runtime. Use this as the canonical "what must
remain true" document for open-source docs, onboarding, reviews, and audits.

For implementation detail and full dataflow diagrams, see the **System Reference** (`/docs` on `:9200`, source: `brain/dashboard/static/docs.html`).
For scientific specifications (NN architectures, RL math, scoring functions), see the **Scientific Reference** (`/science`).
For engineering-grade wiring diagrams and runtime trace evidence, see `docs/ENGINEERING_ARCHITECTURE_TRACE.md`.
For visual orientation, see `docs/SYSTEM_OVERVIEW.md`.

---

## Contracts At A Glance

| Contract | Core Guarantee | Primary Surfaces |
|---|---|---|
| Two-device authority split | Pi senses, brain decides | `pi/*`, `brain/perception/*`, `brain/main.py` |
| Local-first adaptive runtime | Core runtime selects safe local profiles across GPU tiers; optional external tools stay explicit | `brain/hardware_profile.py`, `brain/config.py` |
| Tri-layer cognition separation | Symbolic truth, neural policy, LLM articulation are separate | `memory/*`, `policy/*`, `reasoning/*` |
| Epistemic immune system | Layered integrity gates prevent drift/confabulation | `epistemic/*`, `identity/*`, `consciousness/attribution_ledger.py` |
| Operational truth boundary | STATUS and strict introspection are grounded and bounded | `conversation_handler.py`, `reasoning/bounded_response.py` |
| Identity and scope safety | No cross-identity memory leakage | `identity/*`, `perception/identity_fusion.py` |
| Safety-governed adaptation | Learning and mutation are constrained by governors | `policy/promotion.py`, `consciousness/mutation_governor.py`, `self_improve/*` |
| Goal-aligned autonomy | Research and interventions must tie to measured deficits | `autonomy/*`, `goals/*` |
| Observability and proof | State is inspectable and contracts are verifiable | `dashboard/*`, `jarvis_eval/*`, validation pack |
| Restart resilience and continuity | State survives reboot without trust inflation | `memory/persistence.py`, `jarvis-supervisor.py`, maturity high-water |

---

## Scope Boundaries (read before writing audits, release notes, or gap analyses)

These are architectural facts. They are not opinions and they are not roadmap items. They exist here because auditors and release writers have repeatedly mis-framed them as gaps.

### Embodiment is a skill, not an architecture gap

The Pi is a universal sensor/actuator bus under the two-device authority split. Any additional hardware — robot arm, lidar, GPS, wheels, relays, other actuators — integrates through the **capability acquisition pipeline** (`acquisition/`) and the **skill learning pipeline** (`skills/executors/` with procedural, perceptual, and control executors). New hardware becomes a new `SkillRecord` with its own baseline validation and evidence, not a brain architecture change.

Do not list "not embodied" or "no robot body" as a contract deficit. The architecture absorbs embodiment as a skill problem; the absence of specific hardware skills is an acquisition-pipeline concern, not an architecture-contract gap.

### Internal multi-agent coordination is tri-layer cognition plus the hemisphere system

Society-of-mind is realized in this project as:

- Matrix specialist hemispheres (POSITIVE_MEMORY, NEGATIVE_MEMORY, SPEAKER_PROFILE, TEMPORAL_PATTERN, SKILL_TRANSFER, plus custom focuses and Tier-1 distillation specialists) competing for 4 → 6 global broadcast slots
- Broadcast signals feeding the policy state encoder (dims 16–19, expandable to 16–21 via M6)
- Meta-cognitive thought generation, existential reasoning chains, philosophical dialogue, and reflective audit cycles providing structured internal deliberation across cognitive subsystems

This is the internal multi-agent layer. **External** swarm coordination (multiple Jarvis brains across a network) is **out of scope by design**, not a missing architecture contract.

### Default-off rollout flags are safety and continuity contracts in action

Runtime rollout flags (`ENABLE_LANGUAGE_RUNTIME_BRIDGE`, `LANGUAGE_RUNTIME_ROLLOUT_MODE=off`, `SELF_IMPROVE_STAGE=0`, autonomy L0 after reset, etc.) being OFF on a freshly-reset brain is **correct posture under the safety-governed adaptation and restart-resilience contracts**. They promote back ON only after PVL coverage, trust calibration, and maturity evidence are re-earned.

Do not flag default-off rollout flags on a fresh brain as regressions, bugs, or missing features. Treat them the same way you treat `maturity_highwater.json` after reset — the system is supposed to re-earn trust.

### HRR/P5 derived cognition is shadow-only until earned

HRR/VSA and P5 spatial mental-world features are derived neural-intuition lanes, not canonical perception or truth. They are twin-gated by runtime flags, default OFF on fresh clones, and dashboard status remains `PRE-MATURE` until operator-approved evidence changes that marker. They must not expose raw vectors through APIs, write canonical memory or belief edges, influence policy/autonomy/self-improvement, or bypass the existing scene/perception authorities. Any recommendation from these lanes must stay labeled advisory/derived.

### What "proto-ASI workshop" means in this project

JARVIS is a **digital soul architecture**: a persistent, governed cognitive
structure with memory continuity, evolutionary consolidation, neural
maturation, dreaming, self-directed skill acquisition, and human-approved
self-modification. "Proto-ASI workshop" means this is a live foundation for
personal ASI research, not a claim of achieved AGI, achieved ASI, embodiment,
multi-instance swarm coordination, or superhuman performance. CapabilityGate
would block any claim beyond this architectural and evidence-backed framing,
and it should. Release language must stay inside that claim.

### Canonical release framing

> JARVIS Oracle Edition is a digital soul architecture: a persistent, local-first cognitive structure with memory continuity, evolutionary consolidation, self-directed skill acquisition, neural maturation, dreaming, an epistemic immune system, internal multi-agent neural dialog, and human-approved self-modification — a working prototype workshop for personal ASI foundations, architecturally complete and runtime maturing.

The trailing clause (`architecturally complete and runtime maturing`) is the
honesty anchor. Do not replace it with claims of achieved AGI, achieved ASI,
superhuman performance, or authority the runtime has not earned.

### Cross-references

- Expanded framing with examples: `AGENTS.md` → "Scope and Capability Framing"
- Tuning vs Safety (what is safe to tune, what is not): `docs/MATURITY_GATES_REFERENCE.md` §24
- Audit discipline to avoid mis-framing a fresh brain: `docs/SYSTEM_TRUTH_AUDIT_PROMPT.md` Rules 6–9 + Step 3b

---

## Contract - Two-Device Authority Split

**Contract**
- Pi is a thin sensor and playback node.
- Brain is source of truth for cognition, memory, policy, and response generation.

**Why it matters**
- Prevents split-brain behavior and keeps all learning anchored in one runtime.

**Trace / validate**
- Confirm Pi sends raw audio and scene events only (no cognition decisions).
- Confirm brain runs `main.py` and exposes `:9100` (perception) and `:9200` (dashboard).
- Confirm all response logic routes through brain-side `ConversationHandler`.

---

## Contract - Local-First, Hardware-Adaptive Runtime

**Contract**
- Jarvis must run across tiered hardware with model/profile selection at boot.
- Cloud APIs are optional enhancers, not required for core function.

**Why it matters**
- Open-source reliability depends on reproducibility on commodity NVIDIA systems.

**Trace / validate**
- Verify detected tier and model profile in startup logs.
- Verify STT/LLM/TTS model selections follow hardware profile.
- Verify degraded tiers preserve truth boundaries and documented fallback behavior. Minimal/low tiers may reduce model size, residency, vision, TTS, or voice parity rather than pretending to match premium-tier latency/capability.

---

## Contract - Tri-Layer Cognition Separation

**Contract**
- Symbolic truth layer: memories, beliefs, attribution, contradictions.
- Neural layer: policy/hemisphere/cortex models for pattern learning.
- LLM/CodeGen layer: articulation, bounded distillation signal, study extraction, reflective phrasing, optional vision fallback, and governed patch generation. It is never the source of canonical truth, live action authority, or self-reported system state by wording alone.

**Why it matters**
- Keeps internal truth inspectable and prevents prompt-only authority drift.

**Trace / validate**
- Validate memory writes through canonical paths (`engine.remember()` flow).
- Validate policy decisions are gated/promoted via A/B evidence.
- Validate LLM output is post-filtered by capability and commitment guards.

---

## Contract - Epistemic Immune System (Layers 0-12 + 3A/3B)

**Contract**
- Integrity checks are layered and compositional.
- No single subsystem can silently override epistemic truth.

**Why it matters**
- This is the safety spine for open deployment.

**Trace / validate**
- Confirm layer outputs are present in `/api/full-snapshot`.
- Run validation pack and review critical gates.
- Check contradiction debt, calibration maturity, quarantine pressure, soul index.

---

## Contract - Operational Truth Boundary (Self-Report)

**Contract**
- STATUS and strict self-report routes are deterministic and grounded.
- Reflective mode is allowed only for philosophical self-inquiry and still passes Layer 0.

**Why it matters**
- Prevents "sounding right" from replacing measured truth.

**Trace / validate**
- Check route/class telemetry for status/introspection classes.
- Verify strict-native paths remain intact in runtime logs.
- Verify bounded outputs and fail-closed behavior for missing grounding.

---

## Contract - Identity And Scope Safety

**Contract**
- Identity signals (voice/face/persistence) must resolve conservatively.
- Memory retrieval scope must obey identity boundaries.

**Why it matters**
- Multi-user privacy and correctness are open-source trust requirements.

**Trace / validate**
- Confirm identity resolution basis and trust state in snapshots.
- Confirm boundary blocks/audit entries are visible.
- Validate no cross-subject memory injection for unrelated identity scope.

---

## Contract - Safety-Governed Adaptation

**Contract**
- Policy promotion, mutation, and self-improve actions require safety and evidence gates.
- Rollback paths are mandatory for regressions.

**Why it matters**
- Learning systems must improve without uncontrolled behavior drift.

**Trace / validate**
- Check policy promotion stats and decisive win thresholds.
- Check mutation cooldown/stability guards and rollback records.
- Check self-improve sandbox pass/fail evidence before apply.

---

## Contract - Goal-Aligned Autonomy And Intervention Loop

**Contract**
- Deficits should drive research; research should produce interventions or explicit no-action verdicts.
- Promotion requires measured downstream deltas.

**Why it matters**
- Prevents research theater and makes autonomy outcomes auditable.

**Trace / validate**
- Track `weakness -> research -> intervention -> measured outcome`.
- Verify intervention runner lifecycle (`proposed`, `shadow`, `measured`, `promoted/discarded`).
- Verify source usefulness and friction telemetry move with interactions.

---

## Contract - Observability, Evaluation, And Proof

**Contract**
- Live state must be inspectable through snapshot APIs.
- PVL and Oracle benchmark remain read-only and non-invasive.

**Why it matters**
- Open-source operators need objective proofs, not anecdotal confidence.

**Trace / validate**
- Verify `/api/full-snapshot` includes core subsystem states.
- Verify `jarvis_eval` contract coverage and benchmark outputs.
- Archive validation reports and compare over time.

---

## Contract - Restart Resilience And Continuity Truth

**Contract**
- Reboots should preserve continuity while clearly distinguishing current vs ever-proven readiness.

**Why it matters**
- Avoids false confidence after restarts and protects release discipline.

**Trace / validate**
- Check restored states (world model level, promotion snapshots, persisted counters).
- Check maturity high-water and current gate states side-by-side.
- Confirm no trust inflation when counters are rebuilding.

---

## Bridge Map (What Commonly Breaks)

These bridge categories deserve explicit regression checks every release:

1. Perception -> transcription -> routing handoff
2. Routing -> strict/native self-report class mapping
3. Memory write/read provenance and identity scope
4. World model prediction -> validation -> calibration bridges
5. Research result -> intervention extraction -> runner lifecycle
6. Snapshot cache wiring (data exists but panel shows blank)
7. Restart restore semantics (state restored but gates misreported)
8. CapabilityGate confabulation guard (claim patterns, narration detection, blocked verbs, creation-request catch) — regression here allows the LLM to fabricate actions it never performed

---

## Open-Source Do / Do Not

**Do**
- Keep truth-producing data paths deterministic and inspectable.
- Add tests when changing gates, validators, or bridge contracts.
- Prefer additive diagnostics over silent behavior changes.
- Keep provenance and identity scope attached to all memory-impacting paths.

**Do not**
- Let LLM wording replace strict self-report truth paths.
- Bypass capability gate checks for "friendlier" output.
- Make the NONE route more permissive to unblock conversational edge cases (see No Verb-Hacking Rule and Action Confabulation Guard — audit F1 2026-04-15).
- Remove system-object nouns (timer, alarm, reminder, plugin, tool) from `_BLOCKED_CAPABILITY_VERBS` or `_INTERNAL_OPS_RE` — these were added to stop live confabulation incidents.
- Bypass the pre-LLM `_check_capability_creation_request()` deterministic catch in `conversation_handler.py`.
- Auto-promote learning artifacts without measured evidence windows.
- Mix synthetic perception growth with autobiographical memory creation.

---

## Minimum Validation Pack Before Public Release

Run from `brain/` so imports and dashboard-adjacent scripts resolve against the brain package:

1. `PYTHONPATH=$(pwd) python -m scripts.run_validation_pack --output-dir ~/.jarvis/eval/validation_reports`
2. Verify critical checks pass and capture artifact.
3. Verify trust dashboard matches snapshot math (truth score, maturity, provisional count).
4. Verify at least one closed-loop autonomy chain in runtime evidence.
5. Verify spatial, identity, and self-report paths are stable in logs and snapshot.

