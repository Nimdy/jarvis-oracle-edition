# NOW — load-bearing card (refresh this file, not the others)

**Date: 2026-08-31.** Operator: David.

This is the **only** file allowed to hold current branch, life-phase, and leftovers.
Roster, turn lanes, classify, and gates live elsewhere. If this file disagrees with
`SHOCKWAVE_HANDOFF.md` git/stage lines, **this file wins** and the handoff is frozen.

When branch, stage, or leftovers change: edit **this file**. Do not copy HEAD into
the handoff, the playbook, or a skill.

---

## Where we are

| | |
|---|---|
| **Repo** | `~/projects/jarvis-oracle-edition` → origin `Nimdy/jarvis-oracle-edition` |
| **Branch** | `feat/nn-fleet-consume` (cut from `feat/gestation-period` `f5f0c02`) |
| **HEAD (committed)** | `f5f0c02` — Stage 6 household MEMORY native + TAP identity law |
| **vs origin** | `feat/gestation-period` pushed. This cut is nn-fleet `live_state` + consumed. |
| **Merge `main`** | **No**, unless David asks |
| **Gestation** | **Graduated.** Do not re-run birth. Do not wipe `~/.jarvis`. |
| **Life phase** | Companion **Stage 6 spoken lived**. Gate work is nn-fleet consume, not Stage 7. Playbook “elevate to L2” is **DOC DRIFT**. |
| **Autonomy** | **L1 research.** Readiness composite live ~0.62 vs Stage 7 target 0.92. Do not grind Stage 7 for the number. |
| **Operational queue** | **1.** Stage 6 spoken lived (kids/family/interrupt MEMORY native). **2. CURRENT:** manual gate work — nn-fleet `live_state` + **consumed**. **Not** Workstream 2 unless David names it. **Not** playbook Stage 7. |

Household facts live in **memory**, not in code. Operator fact card (Grok TAP-as-David; do not invent):

| | |
|---|---|
| Name | David (call him David). Brief. |
| Job | Software engineer. Not a plumber. |
| Family | **Tonya**, Lily, Owen. Dog Skyler/Skylar (border collie) is **not** the family list. Store/STT still **Tanya** / Skylar — do not grind. |
| Food / color | Pizza. Blue. |
| Hobbies | DJing, drone flying, fishing, camping, beach, family time, tinkering with tech. Also EDM. |
| Morning | Wake 6, coffee, **walk Skylar**, desk. (Store still has after-work walk leftover.) |
| Interrupt | Do not interrupt on a call. Camera+audio “in a conversation vs talking in the office” is **sensing**, not a Stage 6 recall. |
| Privacy | Do not bring up medical proactively. Family is private. |

Do not hardcode names or jobs into the router. Playbook “Sarah” is stale.

---

## Swim lanes

| Who | Job | Must not |
|---|---|---|
| **David** | Owns the stack (start/stop/bounce/wipe). Sits. Picks the gate. | |
| **Grok 4.6** (in-chair) | Couple existing wires. Teacher/class bugs. Direction. | New organs. Flip a gated mouth. Bounce unless asked. |
| **Shockwave** | Trace **one turn** vs DNA. Docs vs log. | Code. Bounce. Stamp **REAL** on a working wire. |
| **Megatron** | Reclassify Shockwave. Keep / send back / hand a named slice to Grok 4.6. | Implement. Pair-program. |

Cursor is not on the roster.

**How we test (do not mix these). Verbal conversation testing always uses the mind path.**

Every spoken/verbal sit — Grok, Shockwave, or David — enters at `handle_transcription`. That is either **Pi STT** or **`POST /api/operator/tap`**. There is no third mouth.

| Kind | What it is | What it is not | Keep? |
|---|---|---|---|
| **Sit** | Pi voice **or** TAP → `handle_transcription` → router → lane → L0 → TTS. Score `brain.log` (`OPERATOR-PROXY TAP` or STT) + the spoken string. | pytest. `/api/chat` (retired **410**). Synthetic gym audio. | Yes. This is Stage 6 / soul proof. |
| **Contract pin** | `brain/tests/` on WSL — L0 sweep, aboutness, TAP provenance, `/api/chat` 410, busy/follow-up flags. **These tests are correct.** They catch wire breaks before a sit. | A spoken sit. Green pytest ≠ she recalled Tonya. | **Yes. Do not delete or skip them.** |
| **Dashboard** | Instruments (`/v2/nnfleet` `live_state` + consumed, prove **verdicts** not titles). Read **after** a sit. | Proof of a turn. A green badge is not a sit. | Yes, as instruments. |
| **Synthetic gym** | Telemetry / weight-room only. | `handle_transcription`. Authority. Enrollment. | Never as a sit. |

TAP default is a **new sit** (`follow_up=false`). Continue only if `GET /api/operator/tap/status` shows FOLLOW_UP **or** `expects_follow_up` **and** you mean to answer her. Mute the Pi mic when an agent TAPs so office speech does not barge in.

**Do not shotgun TAP.** One sit. Wait idle **and** store (log + memory row if a fact was taught). This is not an app that returns 200 and is done. Canon: [OPERATOR_PROXY_TAP.md](OPERATOR_PROXY_TAP.md).

---

## What this is (one paragraph)

Two devices, one mind. Pi = senses. Brain = source of truth. **The LLM is voice, not the brain.**
This is a prototype workshop for personal-ASI *foundations* — persistent cognitive structure,
earned gates, epistemic floor. It is **not** achieved consciousness, AGI, or ASI. Inner-life
HUD numbers (awareness 0.98, recursive_self_modeling, soul bars) are **not the mouth**.

Default: the subsystem already exists and is **shadow / PRE-MATURE / `not_born` / default-OFF**.
That is success. Do not build a parallel. Do not flip a gate to make a sentence warmer.

---

## One spoken turn (authority)

Live chooser: **heuristic keyword router**. `voice_intent` is **shadow** and does not pick the turn.

**P1 OSV → VISION (live JPEG + VLM) → about-X MEMORY → LLM.** Then CapabilityGate L0 → TTS.

| Lane | Speaks | LLM authors those facts? |
|---|---|---|
| P1 OSV | Deterministic articulator | **No.** `native_voice` is `not_born`. Revoice is teacher-only. |
| VISION | Pi JPEG + brain VLM | **No scene.** Dinner-chat must not name the room. |
| MEMORY | Recalled payloads; about-me = **this-turn speaker** | No OSV dump as autobiography |
| LLM | Ollama under L0 | Must not invent tools, jobs, self-metrics |

Full diagrams: [AGENT_MAP.md](AGENT_MAP.md).

---

## Classify (exactly one per finding)

**REAL ≠ live. REAL = defect.**

| Class | Meaning |
|---|---|
| **LIVE / WIRED** | Lane fired as drawn. Not a bug. Do **not** call this REAL. |
| **REAL** | Contract broken **and** the gate is already earned. Fix-class only. |
| **GATED / EXPECTED** | Shadow, PRE-MATURE, `not_born`, sample floor, leftover cadence |
| **THEATER** | LLM authored a fact no sensor/lane had |
| **DOC DRIFT** | Book ≠ log **and** the wire matches DNA |
| **PROCESS BREAK** | Required station skipped or a ledger forged |

DNA test, worked examples, process table: [shockwave/GROWTH_VALIDATION.md](shockwave/GROWTH_VALIDATION.md).

---

## Do not misread these surfaces

| Surface | Honest read | Hand-wave |
|---|---|---|
| `/v2/nnfleet` | **Authority table.** Read `live_state` + whether anything **consumes** the NN output.  Samples ≠ control. | Hemisphere “8/8” / high accuracy = she routes with the NN |
| `/v2/prove` | Read **verdicts**, not claim **titles**. Claim 1 title (“NN routes better than regex”) is **false today**; PENDING is correct. Claim 2 PROVEN is store-survival of `dream_observer`, not “retrieved in a conversation.” | Quote a title as proven science |
| `/v2/cognition` soul bars | Derived traits. **SHADOW mouth:** `soul_dims→voice` is `sent_to_model=False`. TraitModulator injects **flat names** + dominant nudge only. | curiosity 0.92 = she sounds curious |
| TBS / companion read | TBS-0 **injects nothing**. Situational read fires **after** the reply. P4 `applied=False`. | “inner thoughts drive the sentence” |
| Spark / curiosity | Operator-pull. `/v2/grounding`. Autonomy curiosity gated to L3; live level is L1. | Empty-room chatter, or pending=0 as a bug |
| Policy NN | Shadow. Measurement-failed. Flags off. | Wait for policy to become the brain |
| Cockpit / consciousness HUD | Templates, not speech | Proof of digital consciousness |
| `/api/full-snapshot` | Valid **instrument**, ~1MB. Not the weekly ritual. | Dump 80 keys and invent an organ |
| Face this-tick `< 0.55` unknown | **Expected** until crop ≥ 0.55. Fusion can still be David via voice/persisted. | Lower Face 0.55; “identity wiped” |
| Stage 6 `belief_orphan_rate` / “recall precision” | Orphan is graph health (unlinked `external_source`). High after a wipe is expected. `memory_recall_precision` was `1 − orphan` — **not** the mouth. Training page now labels spoken recall vs orphan instrument. | Graduate on the chip; treat 0.54 as failed hippocampus |
| L3 guest fail-closed | Lock working | Weaken `_policy_guest` so recall “works” |

Dashboard map: [V2_SURFACE_TRUTH.md](V2_SURFACE_TRUTH.md).

---

## Probe card (after David talks, after TAP, or after bounce)

Do **not** start from full-snapshot.

```
0. GET :9200/api/operator/tap/status   busy / ear / follow_up — wait if busy
1. brain.log: STT **or** OPERATOR-PROXY TAP → follow_up= → route= → snapshot sha/age/fresh → spoken text → fusion
2. ps: pid + start (is this the code you think it is?)
3. GET :9200/api/scene     caption + person_bbox_count
4. GET :9200/api/identity  method, face known, voice known
5. GET :9200/api/onboarding/status   current_stage, checkpoints, readiness
6. GET :9200/api/nn-fleet  live_state / consumed — not sample counts
7. GET :9200/api/eval/snapshot   current vs high-water
8. JSON under ~/.jarvis/ only for the disputed memory id / grounding level / dream_stats
```

SSH brain: `duafoo@192.168.1.222` key `~/.ssh/id_jarvis_desktop`.
Workspace is WSL. `./sync-desktop.sh` is not a bounce. Operator owns start/stop.

---

## Hard no (this house)

Do not bounce unless David asks. Do not wipe `~/.jarvis`. Do not merge `main`.
Do not start/stop supervisor, `main.py`, or lidar.
Do not run pytest on the live brain host against `~/.jarvis`.

Do not flip unless David **names** the gate:

`OSV_P2_ACTIVE`, revoice-live, voice-intent primary, `native_voice`, soul-dial
(`sent_to_model`), GroundingDrivePromotion, language bridge, L3, Spark, HRR,
SI Stage 2, Weight-Room `enforces`, CapabilityGate L0, autonomy L2/L3, Face **0.55**.

Do not: YOLO on the Pi; Golden fuzzy synonyms; `briefing_register` / extra articulators;
phrase-hack one prompt; delete memories to “fix” a lie; open Matrix / new specialists /
new dashboards / new epics; treat Spark pending=0 as a bug; treat Stage 6 orphan chip
as graduation.

**Memory / L0 — agents keep failing this. Read the next section or you will ship a bypass.**

---

## STOP — spoken memory vs silent memory (do not bypass the epistemic layer)

Lived 2026-08-31. David sat Stage 6. Agents keep treating “the mouth was wrong” as “memory is missing / L0 is too strict / wire fractal into speech.” That is how this workshop becomes an LLM wrapper.

**Only one lane speaks recall.** `search_memory` → native MEMORY formatter **or** LLM+pref inject → **CapabilityGate L0** → TTS.

These exist, are wired, and **must not be coupled into the mouth** to “fix” a Stage 6 sit:

| Lane | Authority | Speaks? | Common agent crime |
|---|---|---|---|
| sqlite-vec + **ranker** (live) | Scoring | Via search_memory | Re-sort by weight; skip ranker |
| Salience NN | Write advisory, blend 0.2 | No | Flip blend to make recall “better” |
| **L3** personal security | Lock | Fail-closed for the wrong person | Weaken `_policy_guest` so David hears his dog |
| Provenance boost | Scoring | No | Promote `dream_observer` / scrap to testimony |
| **L0 CapabilityGate** | Last filter before TTS | Strips/rewrites | Remove `music`/`dance` from blocked verbs; skip L0 on MEMORY; whitelist EDM |
| Belief graph | Edges from `MEMORY_WRITE` | No | Treat orphan ~0.47 (`external_source`) as “family missing” |
| **Fractal recall** | Background chains. Never writes canonical. Dream fitness **&lt; 0**. Cue 0.40 | **No. Never.** | “no seed above 0.40” as a bug; speak a chain |
| **Dream** | Consolidate. `dream_observer` in store. Student shadow | Not autobiography | Quote prove.html Claim 2 as “retrieved in conversation” |
| **HRR / spatial album** | PRE-MATURE, album **OFF** | No | Treat as recall; persist vectors in the album |
| Pref inject | LLM path only | Yes, on NONE | Steal native MEMORY about-me onto the LLM so it “sounds better” |
| Native MEMORY formatter | Fail-closed retrieved payloads | Yes | Let LLM narrate over retrieval; emit “I can pull more details if you want” (that is a **tool claim**; L0 is right to kill it) |

If fractal did not fire, dream did not speak, HRR is PRE-MATURE — **that is success**, not a hole.

### Stage 6 leftovers (couple the consume-wire; do not open an organ)

TAP-lived 2026-08-31 (operator-proxy, `follow_up=false` new sits). Do not “fix” these by weakening L0 or L3.

1. **About-me** — TAP `tap_d2fb3f28664c` **route=MEMORY** (card wants MEMORY). L3 1/32 blocked, 20 passed. Retrieved 8 (2 observation / 6 conversation). Mouth named **software engineer**, then coffee/desk + camera greeting mash. Did **not** name prefer-brief / pizza / EDM. Ranking leftover.
2. **EDM** — TAP mouth **named electronic dance music**. L0 same-sentence couple lived. Keep verbs blocked.
3. **Family TAP theater** — Pi/TAP on NONE let the LLM author cousins / Ethan / Emily-Mike. **Couple (2026-08-31):** household *questions* route **MEMORY** → native formatter. Lived empty-mouth: recaps filled top_k so keyword never ran. Ranker still sees the live question; keyword/fact fill still runs (L3 on); recaps filtered after. No extra verb list. Do **not** skip ranker. Do **not** couple fractal/HRR/dream into speech. Voice-intent stays shadow. Needs **bounce**.
4. **Job TAP** `tap_42d4f124d180` — **software engineer LIVE** (NONE + inject). After-work walk / Skylar padding leftover. Do not grind STT Skylar vs stored Skyler. Same class as Tanya vs Tonya.
5. **Morning TAP** card line now MEMORY (same household couple). Store may still only have after-work Skylar walk — native will speak payloads, not invent a morning walk. `memory_recall_precision` 0.54 is `1 − orphan`, **not** spoken 9/10.
6. Kitchen vision leftover: drop forever.
7. Uncommitted WSL teacher/persist (overlap-only correction, bounce-durable restatement, injection sort, episodic vs self-pref). **Lived** for family/job on Pi. That pile is **not** an L0 bypass. Uncommitted ≠ bounced onto the PID.

Park unless the blocker actually changed: Matrix 3–9, #32 domain NN, L7, Thought Maturity P3, policy live, lidar rebase, SpatialMemoryGate→remember, affect-expression, WR `enforces`, Connectome L4, Spark Stage 2, HRR, language bridge, native_voice, voice-intent live.

---

## Read next (in this order)

1. This file.
2. [OPERATOR_PROXY_TAP.md](OPERATOR_PROXY_TAP.md) — how agents sit (verbal testing). Pytest stays as pins.
3. [AGENT_MAP.md](AGENT_MAP.md) — turn contract.
4. [shockwave/GROWTH_VALIDATION.md](shockwave/GROWTH_VALIDATION.md) — classify + DNA + process immune.
5. [MATURITY_GATES_REFERENCE.md](MATURITY_GATES_REFERENCE.md) + `/api/nn-fleet` — before touching an NN.
6. Frozen roster/hard-nos snapshot: [SHOCKWAVE_HANDOFF.md](SHOCKWAVE_HANDOFF.md) (do not take its git HEAD as current).
7. Field manual: [AGENTS.md](../AGENTS.md).
8. GitHub **#83** if the question is “what to build next” after Stage 6.
