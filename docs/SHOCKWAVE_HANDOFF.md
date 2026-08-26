# Shockwave handoff — JARVIS Oracle Edition

**Checkpoint:** 2026-08-26. Operator: David.  
**Assign this file to any bot taking JARVIS dev** (Shockwave, Megatron roster, next Grok/Claude session).  
**This is not a greenfield. Gated is not missing. Do not guess from a dashboard looking wrong.**

Canon, in this order, before the first edit:

1. This file (where we are, what not to do).
2. [AGENT_MAP.md](AGENT_MAP.md) — what the system *is*, one spoken turn, who has authority.
3. [AGENTS.md](../AGENTS.md) — field manual.
4. [MATURITY_GATES_REFERENCE.md](MATURITY_GATES_REFERENCE.md) + `brain/nn_fleet_registry.json` — before touching any NN.
5. GitHub **#83** — couple what exists, stop opening organs.

If a change is conversation, OSV, memory, routing, TTS, or preferences: match a box on AGENT_MAP first.

---

## What this project is

Two devices. One mind. The LLM is **voice**, not the brain.

Repo: `https://github.com/Nimdy/jarvis-oracle-edition.git`

| Device | Role | Must not |
|---|---|---|
| **Pi 5** (`192.168.1.248`, `pi/`) | Senses + speakers: camera, Hailo **person+pose only**, JPEG `/snapshot`, mic, playback. Cheap on-device so the body can go mobile later. | Decide, remember, route, claim, **CPU YOLO room objects** |
| **Brain** (`192.168.1.222`, `brain/`) | Perception finish, GPU VLM caption of the Pi JPEG, Layer 3B tracker, route, memory, OSV, policy, TTS | Be replaced by “just ask the LLM” |

Ollama articulates. It does not author self-facts, scene facts, or tools.

North star: a sovereign local companion that cannot lie about high-stakes truth, grows only on **earned evidence**, and stays mobile-capable (Pi goes with the person; brain keeps the world model). Lost-phone / path-block / roof-wear is the **later HRR minds-eye**. HRR is **PRE-MATURE, zero authority**. That is not a missing feature.

---

## Where we are (git)

| | |
|---|---|
| **Branch** | `fix/audit-real-bugs-2026-08-24` |
| **HEAD at this checkpoint** | `531e809` — `fix: live VQA, fresh Pi grab, retry-after-wrong, TTS markdown` |
| **Merge `main`** | **No**, unless David asks |

Operator owns start/stop of supervisor, `main.py`, and lidar. Agents **sync** (`./sync-desktop.sh`, `./sync-pi.sh`). They do **not** bounce unless asked.

A process restart is **not a wipe**. Weights, memories, and promotion JSON persist under `~/.jarvis`. Do not “fix” restart by clearing that tree. Tier-2 *authority* re-earns; data stays.

---

## One spoken turn (the thing agents break)

**P1 OSV → VISION (live JPEG + VLM) → about-X MEMORY → LLM.**

| Lane | When | What speaks | LLM authors facts? |
|---|---|---|---|
| **P1 OSV** | Self-question classified | Deterministic `articulate_self_view` | **No.** Revoice is teacher-only until `native_voice` is born |
| **VISION** | Look / what do you see / targeted visual question / retry after wrong | Live Pi JPEG + brain VLM | **No scene.** Dinner-chat must not name the room |
| **MEMORY** | About a person/pet/topic, not a P1 kind | Recalled payloads | No self-facts from an OSV dump |
| **LLM** | Everything else | Ollama as voice, under L0 | Must not invent tools, jobs, or self-metrics |

Live chooser: **heuristic keyword router**. `voice_intent` is **shadow** and does not pick the turn. Do not “fix routing” with a phrase regex for one prompt. Log friction; fix the **class**.

Golden Commands are the pre-NN exact-match floor: `Jarvis[, GOLDEN COMMAND] <EXACT BODY>`. Body exact, not fuzzy synonyms. Hey/hi/hello/ok before Jarvis is prefix-only.

P1 spec-sheet on TTS is the **floor** until `native_voice` is born. `revoice.py` writes “lead with the gist” and does **not** speak. Do not add `briefing_register` or exec/tech/ops mouths. Reverted in `69d7819`.

---

## This campaign — already closed (do not re-solve)

Lived on this branch through `531e809`:

- `/api/scene` zeros vs old dashboard: **not HRR down**. Hailo is 1-class person. Room names = brain VLM of Pi JPEG. Payload now carries `caption` + `person_bbox_count`.
- Kitchen lie on VISION: cooking-dinner **memories** overrode the desk caption. Live frame is scene authority; fail-close ungrounded kitchen/stove. Persist what was **spoken**, not the LLM draft.
- “What do you currently see?” dumped OSV; “from the camera” was solid. Visual-present class routes VISION. Self-view (“how do you see yourself”) stays INTROSPECTION.
- Golden `VISION STATUS` speaks the live caption (Hey Jarvis prefix allowed). No “I don’t have data on that yet” theater.
- Targeted VQA: user question is the `describe_scene` prompt. Finger-counts are not room inventory.
- Frozen `/snapshot` (identical JPEG md5 while Hailo still ran): Pi **copies** out of the DMA slot; VQA uses `?grab=1`. Fetch the JPEG **before** “focusing my vision.”
- “That is wrong / check again” after a look **re-fires VISION** + new grab + friction `VISION:correction` + voice-intent teacher VISION. Do not steal MEMORY “remember when that was a lie.”
- TTS strips `**` and `\(4+4=8\)` before Kokoro.
- Empty Hailo summaries (0 objects, 1 person) refresh **person occlusion** on the brain tracker. They do **not** decay VLM desks/monitors and do **not** put YOLO on the Pi.
- Face 0.55: after re-enroll, soak showed **David `known=true`** (~0.85). **Do not lower 0.55.**

---

## What Shockwave should do next

Default job: **#83 — couple what exists.** Integrity first. Burst lab + unattended soak. David is not a full-time teacher.

Before adding a classifier, route override, `kind`, register, preference key, or parallel articulator:

1. Search the repo (`rg` verbosity, briefing, gist, revoice, response_style, native_voice, theory_of_mind, think_before_speak).
2. Open `brain/nn_fleet_registry.json` and `docs/MATURITY_GATES_REFERENCE.md` for that name.
3. If the file says shadow / teacher-only / `not_born` / default-OFF → **stop**. Tell the operator the gate. Do not bypass it with keywords.
4. If user preferences already store it, **use that store**.

After code: sync. David bounces. Verify with **lived evidence**: STT line, `route=`, snapshot `sha=` / `age_ms` / `fresh=`, spoken text, fusion `method` + face/voice `known`. Claim “fixed” only with a lived turn.

Memory: one write path (`engine.remember`), one recall (`search_memory`). About-me = **this-turn speaker**, not a hardcoded companion. About-X from the query, not soul `known_names`. Curiosity asks are not autobiography.

**L3 personal security is a lock, not a miss.** A guest must not hear David’s dog, preferences, or family. Fail-closed “I don’t have that recorded” for the *wrong person* is correct. Empty recall for David is a stamp bug in `identity/resolver.py`, not a reason to weaken `_policy_guest`.

---

## What NOT to do (hard)

**Do not guess because something looked wrong.**

Lived:

- `/api/scene` zeros + old dashboard “had data” = tracker-only vs caption+person, **not** HRR down.
- Face `known=false` after bounce = 0.55 + voice dip, **not** wiped identity. Fusion can still be David via voice/persisted.
- “I don’t have that recorded” for a guest = L3 working.
- Spec-sheet on TTS = deterministic floor waiting for `native_voice`, **not** a missing mouth.

**Do not flip gates** unless David names the gate:

`OSV_P2_ACTIVE`, revoice-live, voice-intent, `native_voice`, GroundingDrivePromotion, language bridge, L3, HRR, Spark/Grounding, CapabilityGate L0, autonomy L2/L3, Weight-Room `enforces`.

**Do not:**

- Merge `main` unless asked.
- Wipe `~/.jarvis`, scars, memories, or weights “to fix restart.”
- Start/stop supervisor, `main.py`, or lidar.
- Put room-object YOLO / VLM on the Pi (WiFi / take-it-with-you).
- Tune Face **0.55**.
- Add Golden commands as fuzzy synonyms.
- Add `briefing_register` or a second articulator because P1 “sounds too technical.”
- Teach the router the **wrong** label (e.g. “check again” as NONE/chat). If you override a route, the teacher is the **final** tool.
- Delete memories to “fix” a lie. Tag / downweight / don’t recall as fact. **Never discard.**
- Open Matrix / new specialists / new dashboards / new epics. ~98 subsystems. Gaps are uncoupled, not missing.
- Fine-tune qwen3-vl because fingers were counted wrong. Frozen Ollama. Decline if not sure; look again if told wrong.
- Treat Spark `pending=0` as a bug. Operator-pull. Starved on purpose.
- Treat fractal recall chain length 1.0 / dream skip as broken if there were fewer than 5 recent lived memories.
- List “not embodied” or “no multi-agent” as architectural gaps. Embodiment is skill acquisition. Hemisphere **is** the internal multi-agent. Default-OFF flags are governance, not missing features.

---

## Known leftovers (couple; do not open an organ)

- Periodic GPU caption still **sparse** (hours, not 5–30 min). The 90s timeout stopped the 12.7h hang; cadence still does not tick as written.
- Kitchen-lie memory `mem_91hZ` (stove scene) and the later mash: tags/downweight **did not survive** a bounce (RAM persist overwrote disk). VISION must not retrieve it; MEMORY still might.
- “How are you feeling?” often routes **STATUS** and dumps integrity composite, not affect. Check existing OSV/STATUS lanes; do not invent a feelings organ.
- After bounce, voice can dip below known. IDENTITY may say “tell me your name” / “voice doesn’t match David.” Face can be unknown until match ≥ 0.55. That is not amnesia.
- Calibration watchdog: 500 outcomes 100% `correct=True` — measurement skew. Do not silence by flipping cal gates.

---

## Hardware / sync / dashboards

| | |
|---|---|
| Brain | `duafoo@192.168.1.222`, key `~/.ssh/id_jarvis_desktop`, code `~/duafoo/brain`, logs `~/.jarvis/brain.log`, data `~/.jarvis/` |
| Pi | `nimda@192.168.1.248`, key `~/.ssh/id_jarvis_pi` |
| Sync | `./sync-desktop.sh` / `./sync-pi.sh` from repo root |
| Dashboard | brain `:9200` — `/v2/camera`, `/v2/identity`, `/v2/spatial`, `/v2/grounding` |

Tracker `visible=0` with a person sitting here = Hailo person vs VLM objects, **not** a dead camera. Spatial HRR is **shadow**, not the live scene. This rig is Hailo + VLM, not RealSense.

---

## How to report

Lived evidence or it did not happen: STT, `route=`, snapshot `sha=` / `age_ms` / `fresh=`, spoken text, fusion `method` + face/voice known.

If you almost added a parallel path, name the **existing** file and the **gate** instead.

Read AGENT_MAP + AGENTS.md before the first edit. Continue coupling. Do not take over the stack.

---

*Checkpoint 2026-08-26. Branch `fix/audit-real-bugs-2026-08-24` @ `531e809`. Do not re-litigate the park list unless a parked item’s blocker has actually changed (hardware arrived, confirmatory test confirmed, person_aware_fraction moved on lived turns, operator named a gate).*
