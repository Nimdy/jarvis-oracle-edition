# dashboardV2 surface truth

**What this is:** a wiring map of `brain/dashboard/static/v2/` — what each page
claims, which API it reads, whether it can write, and what live probe
2026-08-24 actually returned. Use this before treating a green badge as
authority.

**What this is not:** a click-through of every pixel (browser MCP was down).
Every listed GET was hit live at `http://192.168.1.222:9200` unless marked FAIL.

Honesty kit: `static/v2/shared.js` (`window.V2`) — missing numbers render as
UNKNOWN, never fake zero-green. Chat palette on every page **does** write
(`/api/chat`) even on read-only pages. That is a live turn.

Nav source of truth: `shared.js` `NAVPAGES`. `brain.html` and `flow.html` are
retired redirects to `universe.html`.

---

## Maturity ladder (`maturity.html`) — traced

**URL:** `/static/v2/maturity.html` (also `/v2/maturity`)
**Read:** `GET /api/eval/snapshot` every 5s. **Writes:** none (except shared chat).
**Builder:** `jarvis_eval/dashboard_adapter.py::_build_maturity_tracker`

The page is honest about restart vs wipe: **current** vs **ever-met high-water**.
`ACTIVE` = met now. `RECOVERING` = ever-met, below threshold now. `PRE-MATURE` =
never proven.

It is **not** a map of every capability. Spark / grounding / native_voice /
voice-intent *authority* / OSV P2 are **absent**. A green row here does not
mean that subsystem speaks or chooses.

Live 2026-08-24 (pid 19147): **28 / 35 active**, **31 ever-active**.

| Category | Live | Read this as |
|---|---|---|
| Gestation 2/2 | Graduated | Workshop is past Stage 0 |
| Neural Policy **0/3** | Shadow A/B 0/100, win 0%, flags 0/8 | Policy NN is a **measurement failure**, not “almost promoted.” Do not treat as the brain. |
| World Model 4/4 | Level 2, 24850 validated, 94% | Sense-of-world gate is earned. Simulator is a **separate** gate (yardsticks). |
| Memory Cortex 3/4 | Ranker on; salience blend 0.200 / 0.600 | Ranker live; salience still mixing |
| Autonomy Pipeline 5/5 | Level **2**, 77 episodes, 57 deltas, 72.2% win | L2 research/safe_apply is earned. The **L3 delta row is also “active” (57≥25) while level is still 2** — that row is a metric, not “she may self-mod.” L3 stays human-gated. |
| Hemisphere 8/8 | 16 nets, 4/4 slots, distillation samples past min | Samples ≠ spoken authority. Voice-intent samples 245; the **router is still the heuristic.** |
| Dream 3/4 | 713 artifacts, 366 promoted; buffer 242/500 | Dream cycle real; 500-buffer not full |
| Epistemic 3/5 | Truth cal 0.636/0.650 recovering; soul 0.801/0.870 recovering; graph 1409 edges | Immune system is live and still rebuilding two floors |

Oracle stage: **integrative**, restore **verified**.
Validation pack: **blocked**, 3 recovering: `pvl_coverage`, `soul_integrity`, `hrr_primitive_library`. That banner is restart debt, not a new crash.

---

## All v2 pages

Legend: **R** read-only · **W** operator POSTs (key-gated) · **live** = GET 200 this probe.

### Core

| Page | Mode | APIs | Live | Notes |
|---|---|---|---|---|
| cockpit.html | R | `/api/full-snapshot`, `/api/kernel/performance`, `/api/consciousness/thoughts`, SSE `/api/events/stream` | yes | Inner-life HUD. Thoughts are templates, not speech. |
| awakening.html | R | `/api/full-snapshot` → `gestation` | yes | Stage 0 story |
| integrity.html | R | `/api/eval/snapshot`, `/api/self-test`, `/api/full-snapshot` | yes | Self-scored vs grounded vs earned. Companion to maturity. |
| **maturity.html** | R | `/api/eval/snapshot` | yes | This file’s deep trace |
| yardsticks.html | R | `/api/world-model/diagnostics`, `/api/policy`, `/api/eval/autonomy-ab`, `/api/self-sensing` | yes | Circular-score detector. Policy + self-sensing STARVED belong here. |
| memory.html | R | `/api/eval/snapshot`, `/api/full-snapshot`, `/api/memory-integrity` | yes | Store health, not recall quality |
| cognition.html | **W** | consciousness, personality, existential, philosophical, goals, intention-resolver, language-kernel, full-snapshot | yes | **Can mutate goals / resolver stage.** Inner dialogue is here. |
| universe.html | R | `/api/connectome`, `/api/events/stream` | yes | Live wiring pulses. Replaced retired brain/flow pages. |

### Senses

| Page | Mode | APIs | Live | Notes |
|---|---|---|---|---|
| identity.html | **W** | identity, candidates, speakers, faces, sensor-health, scene | yes | Enroll / forget. Scene `entity_count` 0 is **object-tracker**, not “nobody here.” Foot now shows Hailo `person_bbox_count` + VLM `caption`. |
| voice.html | R | full-snapshot, `/api/intent-shadow` | yes | Intent-shadow: observations exist; **level is not primary.** |
| camera.html | **W** | `/api/config`, `/api/scene`, POST `/api/camera/control` | yes | Steers camera only. `/api/scene` now also carries `caption` (brain VLM of Pi JPEG) and `person_bbox_count` (Hailo persons, not tracked entities). Zero tracked objects with a person in-frame is the person/object split, not a dead feed. HRR is not this page. |
| pi5.html | R | `/api/pi5` | yes | Body / lidar / fusion |
| spatial.html | **W** | hrr + scene + spatial diagnostics / calibration | yes | Shadow spatial — HRR is **not** canonical live scene. Perception card is Hailo persons + brain VLM caption (this rig is not RealSense). HRR stage is labeled on `/api/hrr/status`. |
| spatial-core.html | **W** | `/api/pi5`, `/api/spatial/camera-calib` | yes | Calib write |

### Ops

| Page | Mode | APIs | Live | Notes |
|---|---|---|---|---|
| capability.html | **W** | skills, acquisition, plugins, codegen, self-improve, attestation | yes | Self-mod / skill jobs. Human-approve. |
| autonomy.html | **W** | full-snapshot, POST `/api/autonomy/level` | yes | Can change autonomy **level** (operator). Do not confuse with spark. |
| synthetic.html | **W** | full-snapshot, POST synthetic runs | yes | Gyms. Cannot satisfy lived gates. |
| ops.html | **W** | status, health, logs, settings, **restart/shutdown/save** | yes | Process control. Operator only. |
| training.html | **W** | onboarding, playbook, self-test, full-snapshot | yes | Companion stages |
| matrix.html | R | `/api/matrix` | yes | Tier-2 lifecycle. Specialists were probationary after restart. |
| domains.html | R* | `/api/domains` (create/ingest/delete exist, page is mostly read) | yes | count 0 live |

### Lab / insight

| Page | Mode | APIs | Live | Notes |
|---|---|---|---|---|
| lab.html | R | full-snapshot, self-improve/specialists | yes | |
| nnfleet.html | R | `/api/nn-fleet` | yes | **Source of `not_born` / shadow.** Read before adding a “new NN.” |
| voicelab.html | R | `/api/voice-lab` | yes | Teacher pairs exist; student **not_born**; live_voice = deterministic floor |
| hrr.html | R | `/api/hrr/status`, samples | yes | Shadow VSA |
| timeline.html | R | full-snapshot | yes | Flight recorder |
| immune.html | R | full-snapshot | yes | L5–L10 |
| provenance.html | R | ledger + trace explorer | yes | |
| **grounding.html** | **W** | `/api/grounding/queue` + POST answer | yes | Spark operator-pull. Pending 0 after 11 answered. **This is the Johnny 5 tray.** |
| emergence.html | R | full-snapshot | yes | Observations, not proof of consciousness |
| prove.html | R | full-snapshot | yes | “Prove it” pack |

### Retired

| Page | Notes |
|---|---|
| brain.html | Redirect → universe.html (stale topology.json) |
| flow.html | Redirect → universe.html (hand-authored tracer retired) |

---

## Test suite (yes, it exists)

- **~265** files under `brain/tests/`.
- Dashboard truth: `python -m brain.scripts.dashboard_truth_probe` and `tests/test_dashboard_truth_probe.py`, `tests/test_dashboard_meta_endpoints.py`.
- Spark/queue: `tests/test_grounding_pull_queue.py`, `tests/test_spark_invariants.py`.
- Eval: `tests/test_validation_pack.py`.
- These tests do **not** replace a soak. They pin contracts. Maturity *numbers* are live-earned.

---

## How to read a green badge

1. Open this file + [AGENT_MAP.md](AGENT_MAP.md).
2. Ask: does this page **write**, or only watch?
3. Ask: is the gate **authority** or **a sample count**? (Hemisphere samples ≠ router control.)
4. If it is missing from the maturity ladder (spark, native_voice, OSV P2, voice-intent primary), look at `nn_fleet` / `voice-lab` / `grounding.html` / AGENT_MAP — not this ladder.

---

## Verdict (2026-08-24)

The v2 dashboard is a real instrument, not a marketing skin. APIs behind the nav
are live. The danger is **category error**: maturity.html is a *subset* of
growth gates, cockpit shows *inner life that does not speak*, grounding.html
is the only spark close, voicelab shows a mouth that is not born.

Do not “fix” a recovering gate. Do not treat policy 0/3 as a routing bug.
Do not treat hemisphere 8/8 as “she chooses with the NN.”
The north star remains: inner thought (cockpit/cognition) → grounding tray
(grounding.html) → you answer → she learns. Everything else is watch or gym.
