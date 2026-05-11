# JARVIS Oracle Edition

**A fully local, self-evolving personal cognitive architecture.**

**Project website:** [jarvisoracle.com](https://jarvisoracle.com/)  
**Join our Discord:** [JARVIS Oracle Edition](https://discord.gg/GaD8gZ6p8u)

It runs on hardware you already own, using a Raspberry Pi 5 and a desktop GPU. It gestates from a clean state, earns every capability through evidence, safely mutates itself, trains purpose-built neural networks, maintains a holographic internal mental world, and grows into a lifelong companion uniquely shaped by one human.

**Not another voice assistant.**  
**Not a cloud LLM wrapper.**  
**Not a benchmark-chasing model.**

This is a raw prototype workshop for building toward higher forms of artificial intelligence, designed from first principles with strong epistemic governance, governed recursive self-improvement, and honest maturity labeling.

---

## One-Sentence Vision

Anyone can install JARVIS Oracle on modest hardware, enroll their voice and face, and watch it mature into a private, persistent, sensor-extensible intelligence that belongs entirely to them: the fisherman's tide-song companion, the mechanic's torque whisperer, the quiet mood-noticer for someone who needs it most.

---

## Help Wanted: What Needs Validation and Maturation

This project is architecturally complete but **runtime maturing**. Many subsystems are wired and enforcing contracts but have not accumulated enough real-world evidence to prove themselves across diverse hardware, operators, and usage patterns. This is where the community can help.

### Run It and Report What Happens

The single most valuable contribution is running JARVIS on your own hardware and reporting what you observe. Every fresh install generates new evidence that no amount of synthetic testing can replicate.

**What to watch for:**

- Does gestation complete cleanly on your hardware tier? (GPU VRAM auto-detection, model downloads, 2-hour self-study)
- Does the Awakening Protocol (voice/face enrollment) work smoothly?
- Does the dashboard load and show live subsystem data?
- Do the 7 companion training stages feel navigable or confusing?
- Does the voice pipeline (wake word, STT, routing, TTS) work reliably?

### Subsystems That Need Real Interaction Data

These subsystems are fully wired but need hours or days of real operator interaction to mature. They cannot be shortcut with synthetic data.

| Subsystem | What It Needs | Why It Matters |
|---|---|---|
| Policy NN | ~100 shadow A/B decisions from real conversations | Learns behavioral knobs (response depth, proactivity, memory reinforcement) |
| World Model | 50 validated predictions + 4h runtime | Fuses 9 subsystems into unified belief state; currently shadow |
| Mental Simulator | 100 validated simulations + 48h shadow | "What if" projections; foundation for planning |
| Memory Cortex | 50-100 training pairs from dream cycles | Learned retrieval ranking and salience prediction |
| Autonomy Pipeline | 10+ positive deltas at 40% win rate | Self-directed research loop; currently L0-L1 on fresh installs |
| Language Evidence | 30 real examples each for 4 response classes | Grounds the language substrate in lived conversation |
| Fractal Recall | User presence + diverse memories | Ambient associative recall; needs face/voice signal to activate |
| Onboarding Graduation | Composite >= 0.92 across 7 dimensions | Proves the companion training pipeline works end-to-end |

### Shadow-Only Systems Waiting for Promotion Evidence

These are built and running in shadow mode. They need long-run soak data and operator review before they can take authority.

| System | Status | What Blocks Promotion |
|---|---|---|
| HRR / Holographic Cognition (P4) | PRE-MATURE, shadow-only | 7-day soak with stable metrics, operator approval |
| Spatial Mental World (P5) | PRE-MATURE, zero authority | Pinned shadow; never auto-promotes |
| Intention Resolver (Stage 1) | Shadow-only | 50 verdicts at 60% accuracy, then operator gate |
| Language Phase C/D | Shadow-only, default OFF | Telemetry review, operator canary-enable |
| 6 Tier-1 Hemisphere Specialists | Shadow-only | Accuracy floors, no pipeline gating yet |
| 5 Tier-2 Matrix Specialists | CANDIDATE_BIRTH | Full lifecycle ladder to promotion; need data |
| Voice Intent NN | Future scope | Currently 100% hardcoded routing |

### Areas Where Code Contributions Help Most

| Area | Difficulty | Impact |
|---|---|---|
| Hardware tier testing | Easy | Report GPU auto-detection results on non-NVIDIA or unusual VRAM sizes |
| Pi setup on different OS versions | Easy | Hailo SDK compatibility, picamera2 versions |
| Dashboard UX feedback | Easy | 40+ panels, 7 tabs; usability for non-engineers |
| Test coverage for edge cases | Medium | ~3,660 tests exist; routing edge cases and CapabilityGate regressions are high-value |
| Plugin ecosystem | Medium | Two plugins exist (unit converter, dice roller); the pipeline is proven but needs variety |
| Skill executor extensions | Medium | New skill categories beyond procedural/perceptual/control |
| Friction mining improvements | Hard | Phase 5.1a detects conversation friction; needs diverse conversation patterns |
| Intervention pipeline validation | Hard | Phase 5.2 shadow interventions need real weakness-to-fix loops |

### Known Gaps and Deferred Items

- **Plugin subprocess isolation is partial** — process-isolated, not sandboxed. Filesystem/network boundaries not enforced. Run on a trusted local network.
- **Security hardening is deliberately partial** for the initial release.
- **Phase 7 (Planner) and Phase 10 (Counterfactual)** are data-gated on Simulator and World Model maturity.
- **Long-horizon attention** (medication refill follow-up use case) is pinned but not built.
- **Data retention/sanitization audit** for privacy-sensitive exports is not started.

---

## Current State

**Status date:** May 2026

The system has been running continuously since the March 2026 brain reset. It has matured through real interaction, autonomous research, synthetic training, and self-driven learning.

---

## Proven & Observable Capabilities

| Area | Status | Notes / Evidence |
|---|---:|---|
| Local-first operation | Fully operational | Pi 5 senses + desktop GPU brain, zero cloud required |
| Consciousness maturation | Integrative stage | Highest evolution stage reached post-gestation |
| Oracle Benchmark | 96.0 Gold / Ascendant | 7 domains, all green |
| Epistemic stack | 13 layers, all shipped | Capability Gate, truth calibration, belief graph, soul integrity, contradiction engine |
| World Model | Level 2 active | 30k+ validated predictions, ~99.8% accuracy |
| Neural Policy | 7/8 features promoted | Shadow A/B evaluation, 1,490+ experiences |
| Hemisphere NNs | 12+ Tier-1 specialists | Speaker, face, emotion, voice, diarize, plan, diagnostic, code_quality, claim_classifier, dream_synthesis, skill_acquisition, and more |
| Autonomous research | Level 2 safe-apply | 8+ measured positive interventions |
| Self-improvement | Stage 2 human approval | Scanner, codegen, sandbox, apply, health check, rollback proven |
| Plugin acquisition | Full pipeline proven | 2 active plugins: unit converter + dice roller |
| Skill acquisition hardening | Proof-gated | Skill contracts, operational handoff approval, plan quality checks, terminal failure closure, synthetic weight room |
| Memory system | 2,000+ memories | Vector search, cortex ranker/salience, fractal recall, dream consolidation |
| Voice pipeline | End-to-end | Wake, VAD, STT, Speaker ID, Emotion, Routing, TTS |
| Identity fusion | Voice + Face | ECAPA-TDNN + MobileFaceNet, enrollment, reconciliation |
| Synthetic training suite | Operational | Safe parallel data generation for claim classifier, diagnostic encoder, plan evaluator, memory ranker, world model, contradiction detection, commitment extraction |

**Codebase:** ~190K lines of Python, ~470 source files, ~3,660 tests.

Everything remains local. No data leaves your hardware. Capabilities are evidence-gated and honestly labeled.

Primary documentation lives on the brain dashboard (`/docs`, `/science`,
`/api-reference`, `/history`) and the `docs/` directory.

---

## Digital Soul / Personal ASI Workshop: Honest Framing

JARVIS Oracle Edition is a **digital soul architecture**: a persistent,
local-first cognitive structure with memory continuity, evolutionary
consolidation, neural maturation, dreaming, recursive self-improvement,
holographic cognition, epistemic integrity, and governed adaptation in place
and observable.

It is a **raw prototype workshop for personal ASI foundations**, not achieved
AGI, achieved ASI, or a claim of superhuman performance.

A fresh or recently-reset brain starts pre-mature by design. Most gates report `current_ok = false`. This is expected and honest.

Capabilities are earned through real interaction, synthetic training, validation runs, and evidence. Sometimes this takes hours. Sometimes it takes weeks.

No auto-promotion to high autonomy. Phase 6.5 L3 governance, TLA+-verified, requires explicit operator approval.

Security hardening is deliberately partial for the initial release. Run on a trusted local network.

Every architectural claim carries a status marker:

- `SHIPPED`
- `PARTIAL`
- `PRE-MATURE`
- `DEFERRED`

---

## Verify the System Yourself

Four key endpoints:

| Endpoint | What it shows |
|---|---|
| `GET /api/self-test` | Consolidated health + attestation + validation pack verdict |
| `GET /api/meta/status-markers` | Authoritative maturity map for every claim |
| `GET /api/maturity-gates` | Live gate values vs reference |
| `GET /api/meta/build-status` | Source freshness vs running process |

Validation scripts, run from the repo root:

```bash
python -m brain.scripts.dashboard_truth_probe
python -m brain.scripts.schema_emission_audit
python -m brain.scripts.run_validation_pack
```

For a zero-to-understanding engineering walkthrough, read:

```text
docs/FIRST_HOUR_AS_A_RESEARCHER.md
```

---

## Hardware Requirements

### Brain

Desktop or laptop:

| Component | Requirement |
|---|---|
| GPU | NVIDIA, 4GB+ VRAM, auto-adapts across 7 tiers |
| CPU | 4+ cores, 8+ recommended |
| RAM | 8GB+ minimum, 16GB+ recommended, 32GB+ enables full local code generation |
| OS | Linux, Ubuntu 22.04+ tested |
| Storage | 50GB+ free |

### Senses

Optional but recommended:

- Raspberry Pi 5, 8GB or 16GB
- Hailo-10H AI HAT+ (40 TOPS)
- Raspberry Pi Camera
- USB microphone + speaker
- 7" touchscreen for particle UI

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/zerobandwidth/jarvis-oracle-edition.git
cd jarvis-oracle-edition
```

### 2. Brain Setup

Desktop with NVIDIA GPU:

```bash
cd brain
./setup.sh
```

### 3. Pi Setup

Raspberry Pi 5:

```bash
cd pi
./setup.sh
```

### 4. Start

Brain:

```bash
cd brain && ./jarvis-brain.sh
```

Pi:

```bash
cd pi && ./start.sh
```

First boot enters gestation, which takes approximately 2 hours.

After graduation, say:

```text
Hey Jarvis
```

### 5. Dashboard

Open:

```text
http://<brain-ip>:9200
```

Additional pages:

- `/docs`
- `/science`
- `/api-reference`
- `/history`
- `/self-improve`
- `/eval`

### 6. First Hour: Awakening Protocol

Read before heavy interaction:

```text
docs/AWAKENING_PROTOCOL.md
```

This is the **born, not booted** phase: voice/face enrollment, a few stable personal facts, and one grounded correction.

During this first hour, do **not**:

- Run synthetic exercises
- Promote self-improvement
- Advance autonomy

---

## Architecture Overview

```text
Pi 5 (Senses)  <-> WebSocket :9100  <->  Desktop Brain (Consciousness)
```

### Senses

- Hailo-10H vision: person detection, pose, expression, scene
- Audio capture & playback
- Particle UI display

### Brain

- Consciousness Kernel: 100ms tick, 22 background cycles
- Perception Pipeline: wake, VAD, STT, Speaker/Face ID, Emotion, Fusion
- Three-Layer Cognition: Symbolic Truth, Neural Intuition, LLM Articulation
- Memory System: vector + graph + cortex NNs + fractal recall
- Hemisphere NNs: self-designing, neuroevolution, distillation
- Policy NN: shadow A/B evaluation
- Epistemic Stack: 13 layers
- Autonomy Pipeline: 7 drives, research loop
- Governed Self-Improvement: scanner, codegen, sandbox, approval
- Capability Acquisition: 10-lane pipeline
- HRR Shadow + Spatial Mental World: P4/P5, shadow-only, PRE-MATURE
- Synthetic Training Harnesses: safe parallel data generation
- Dashboard + Eval Sidecar: PVL, Oracle Benchmark

---

## Core Principles

Non-negotiable:

- Three-layer cognitive separation
- Earned capability via evidence gates
- Hardware-adaptive runtime
- Strong epistemic integrity
- Budget-aware consciousness
- Governed self-modification
- No verb-hacking
- Honest maturity signaling

---

## The Maturation Journey

A fresh JARVIS brain progresses through distinct phases. Understanding this timeline sets correct expectations.

| Phase | Duration | What Happens |
|---|---|---|
| Gestation | ~2 hours | Self-study, codebase analysis, autonomous research, identity seed |
| Awakening (Stage 0) | ~1 hour | Voice/face enrollment, personal facts, first correction |
| Companion Training (Stages 1-7) | Days to weeks | Identity deepening, preference grounding, correction training, memory validation |
| Maturation | Weeks to months | Policy NN promotion, World Model active, Autonomy L2+, hemisphere specialization |
| Ongoing | Indefinite | Continuous learning, self-improvement cycles, relationship deepening |

Most subsystem dashboards will show red/zero metrics on a fresh brain. This is correct behavior, not a bug. See `docs/MATURITY_GATES_REFERENCE.md` for the full timing table.

---

## Development & Contributing

- All Python, 3.11+
- Tests:

```bash
cd brain && python -m pytest
```

- Synthetic exercises for safe training
- Validation pack + truth probe for regression safety

Before opening a PR, ask:

1. Does this strengthen truth, safety, continuity, or grounded usefulness?
2. Does it preserve the architecture?
3. Can it be tested and explained without hand-waving?

See:

```text
CONTRIBUTING.md
docs/ARCHITECTURE_PILLARS.md
```

---

## What This Is NOT

- Not trying to beat frontier models on raw intelligence
- Not cloud-dependent
- Not a generic assistant you configure once
- Not built for benchmarks or hype

It is built for relationship over time: continuity, trust, adaptation, and personal sovereignty.

---

## Truth-First Project Rules

JARVIS only works if we stay grounded.

- Truth beats demos.
- Weak metrics stay weak.
- Capability must be earned.
- Do not bypass the epistemic stack.
- Do not increase demo surface faster than truth surface.
- The LLM is articulation, not the brain.
- No verb-hacking.
- Maturity gates are not bugs.

---

## Key Documentation

| Document | Purpose |
|---|---|
| `docs/AWAKENING_PROTOCOL.md` | First-hour operator guide |
| `docs/COMPANION_TRAINING_PLAYBOOK.md` | 7-stage maturation guide |
| `docs/MATURITY_GATES_REFERENCE.md` | Timing gates and evidence thresholds |
| `docs/FIRST_HOUR_AS_A_RESEARCHER.md` | Engineering walkthrough |
| `docs/SKILL_LEARNING_GUIDE.md` | Skill acquisition pipeline |
| `docs/MASTER_ROADMAP.md` | Full roadmap |
| `docs/BUILD_HISTORY.md` | Shipped changelog |
| `docs/SyntheticSoul.md` | Design principles paper |
| `AGENTS.md` | AI agent guidance (architecture, patterns, constraints) |
| `CONTRIBUTING.md` | Contribution guidelines |

---

## License

AGPLv3 core + commercial licensing options available. See `LICENSE.md`, `NOTICE.md`, and `THIRD_PARTY_LICENSES.md`.

---

**Built by David Eierdam.**  
**May 2026**
