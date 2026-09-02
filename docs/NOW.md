# NOW — load-bearing card (refresh this file, not the others)

**Date: 2026-09-02.** Operator: David.

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
| **Branch** | `feat/project-2-oneshot` (cut from `feat/nn-fleet-consume` `a98f018` so a bad slice can be dropped) |
| **HEAD (committed)** | `feat/project-2-oneshot`. Household leftover: family fill + drop pronoun-name scar (`User's … is She`); job `engineer` cue; food≠color. **Needs bounce** before a family-class sit. |
| **vs origin** | Safety branch **not** `main`. Recovery branch `feat/nn-fleet-consume` untouched. Do not force L2. Do not auto L3. |
| **Board** | **Only board:** [JARVIS Next — ordered queue](https://github.com/users/Nimdy/projects/8) (sort by Sequence). [Project 2](https://github.com/users/Nimdy/projects/2) is **closed archive**. North star **#42**. **#83** is frozen notes — do not execute those checkboxes. |
| **Merge `main`** | **No**, unless David asks |
| **Gestation** | **Graduated.** Do not re-run birth. Do not wipe `~/.jarvis`. |
| **Life phase** | Stage 6 curriculum **parked** (operator: chips do not matter). Do not grind Stage 6/7. Gate work is **#83 couple**, not playbook homework. |
| **Autonomy** | **L2 safe-apply, operator-named keep 2026-09-01.** Earned on policy wins (15/10). If a bounce restores L1 and she qualifies again, **let her earn L2** — that is learned, not a defect. L2 code-patch bridge still does not apply (SI stage &lt; 2). Do not demote. Do not auto L3. |
| **Operational queue** | **[Project 8](https://github.com/users/Nimdy/projects/8) Sequence 1–5 is Now.** **#25** family-class consume coupled on WSL (privacy pref / pronoun-name scar out; type-scan fill for taught people facts — no kinship list). **Needs bounce + sit** *who is in my family* with no names. Then job / pizza / **#23** P2 ramble-cut. Do not close #42/#83. **Not flipped live:** native_voice, WR `enforces`, P4, L3, voice-intent, Face 0.55. |

Household facts live in **memory**, not in code. Operator fact card (Grok TAP-as-David; do not invent):

| | |
|---|---|
| Name | David (call him David). Brief. |
| Job | Software engineer. Not a plumber. |
| Family | **Tonya**/Tanya, Lily, Owen. 2026-08-31 David taught Skyler **is** family too. Store/STT still Tanya / Skylar — do not grind names. Do not hardcode the roster. |
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

**HARD RULE — ASK THEN WAIT.** One TAP. Then stop. Wait `busy=false` **and** `speaking=false` **and** FOLLOW_UP closed **and** the write (log + store). Score the mouth. **Then** you may ask if the next sit is allowed. A loop of six prompts is shotgun. “Bounce around classes / use the rotated line” is still **one at a time**. Idle in the TAP 200 is not stored. Lived 2026-08-31: an agent fired 6 TAPs in ~52s; that is a process break. Canon: [OPERATOR_PROXY_TAP.md](OPERATOR_PROXY_TAP.md).

---

## True vision (do not stray)

North star **#42**: a **living companion you want to be with** — playful, warm, honest, unique, growing. **Integrity is the floor, not the ceiling.** High-stakes truth (memory, identity, golden writes, grounding) stays fail-closed. Banter is free. Charming because she is honestly an AI, never a fake human. Not ASI. Not consciousness. Workshop for personal-ASI *foundations*.

**The LLM is voice, not the brain.** JARVIS thinks (router, OSV, memory, gates). The LLM speaks the assembled turn, and may author **general knowledge** (who made Halo). It must not author David’s family, job, or inner HUD numbers.

**Couple what exists. Stop opening organs (#83).** Shadow / PRE-MATURE / `not_born` / default-OFF is success, not a hole. Do not skip the **ranker**. Do not verb-hack routing so the NN never learns. Do not couple fractal / HRR / dream into the mouth to pass a quiz.

**Plastic brain:** tagging, ranker, dream consolidation, fractal chains, HRR album. A correction is a **new stored fact** the ranker can retrieve next time — not a new matcher. If Skylar dies in ten years, that is a new fact with time — not a regex. Fractal/HRR/dream still do not speak today. That is success.

**Agent failure mode (this is the theater):** mouth missed X → add X to a allowlist (wife/son/dog/cousin, extra verbs, skip L0). The sit goes green. The NN never learns. Next leftover is great-great-grandmother. Do not do that. Integrity floor = LLM must not **invent** family. Plastic = whoever you taught, the ranker ranks. Tests must pin recaps-out and ranker-used, **never who is in the roster**.

Board: [JARVIS Next — ordered queue](https://github.com/users/Nimdy/projects/8). [Project 2](https://github.com/users/Nimdy/projects/2) is **closed archive**. Tracker: close on the **mouth**, not on a wire. #25 was closed too early **twice**; reopened 2026-09-02. #83 is frozen notes.

Two devices, one mind. Pi = senses. Brain = source of truth. Inner-life HUD (awareness 0.98, soul bars) is **not the mouth**.

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
| Stage 6 `belief_orphan_rate` / “recall precision” | Orphan is graph health (unlinked `external_source`). High after a wipe is expected. `memory_recall_precision` was `1 − orphan` — **not** the mouth. Chip is **operator-scored 9/10**; tick leaves it unset on purpose. Empty circle = expected, not a broken install, and does **not** lock L3 / native_voice / autonomy. | Graduate on the chip; treat 0.54 as failed hippocampus; grind quizzes because the box is empty |
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
Workspace is WSL `~/projects/jarvis-oracle-edition`. **Code only here.**
`./sync-desktop.sh` (brain) / `./sync-pi.sh` (Pi). Sync is not a bounce. Git push is not a sync.
Operator owns start/stop/bounce.

---

## Hard no (this house)

**Never write JARVIS source on the brain (`192.168.1.222`) or the Pi.** No vim, no patch, no `python` one-liners that edit `brain/` or `pi/`. SSH is logs, APIs, `~/.jarvis` JSON, process. Pytest on WSL only.

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
as graduation; **shotgun TAP** (two sits before the first has stored).

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
| **Roster ontology** | Ranker + taught prefs | Via MEMORY search | Expand wife/son/dog/cousin/great-great lists so a sit names Skyler. That is theater. A correction writes a fact; the next family question retrieves it. |

If fractal did not fire, dream did not speak, HRR is PRE-MATURE — **that is success**, not a hole.

### Stage 6 leftovers (couple the consume-wire; do not open an organ)

TAP-lived 2026-08-31 (operator-proxy, `follow_up=false` new sits). Do not “fix” these by weakening L0 or L3.

1. **About-me ranking** — Pi sit 2026-08-31 16:57 STT **“Jarvis, what do you remember about me?”** conv=`48687d54` **route=MEMORY**. Greeting recap **gone**. Mouth: same-session closer “everything is indeed going well / feel free to reach out” (from follow-up “Yes, everything's going well”) + 6am coffee + software-engineer career mash. No pizza/brief/EDM. **Coupled (needs bounce):** skip session closers + phatic user-turns from about-me. Ranker still scores. Store keeps the rows. Re-sit after bounce. Do not stack a follow-up smalltalk before the about-me question.
2. **EDM** — TAP mouth **named electronic dance music**. L0 same-sentence couple lived. Keep verbs blocked.
3. **Family** — MEMORY native (stop LLM inventing Emily/Mike). Pi sit omitted Skyler because a **kinship-word filter** dropped pet facts. That filter is the cousin/great-great hack. **Removed:** wife/son/dog ontology on family recall. Family = taught prefs the ranker scores; conversation recaps still out. Plastic teach, not a relation list. Needs bounce.
4. **Job TAP** `tap_42d4f124d180` — **software engineer LIVE** (NONE + inject). After-work walk / Skylar padding leftover. Do not grind STT Skylar vs stored Skyler. Same class as Tanya vs Tonya.
5. **Morning** MEMORY native lived: coffee, wake 6, after-work Skylar walk (store leftover, not a missing organ). Do not invent a morning walk in code.
6. Kitchen vision leftover: drop forever.
7. WSL teacher/persist pile **committed** in `f5f0c02`. Bounce 2026-08-31 15:57 → PID **4924** loaded the overlay. Next code still syncs with `./sync-desktop.sh` before the banner can fire.

### nn-fleet consume (this branch) — 2026-08-31

`/api/nn-fleet` now splits **design wire** vs **this-host firing**.

| Field | Meaning |
|---|---|
| `wiring_confirmed` / `inference_consumed` | June-30 audit: a consumer *wire* exists |
| `live_state` + `consumed_now` | This host. Overlay wins over frozen `maturity_state` prose |

Lived 2026-09-01 bounce PID **19451**. `/api/nn-fleet`: 9 `consumed_now` (includes world-model inject). WR `enforces=False`. intent_shadow still shadow.

| NN | Design | Live truth | Mouth? |
|---|---|---|---|
| **memory_ranker** | consumed | `score_batch` reorders retrieval. `ranker_used=True` on Stage 6 sits | Via MEMORY search only |
| hemisphere_engine / distillation / registry / data_feed | consumed | Train + infer into policy **state vector** dims 16–19 | **No.** Policy NN is shadow |
| audio_emotion / display_classifier / conflict_classifier | consumed | TTS/mood; scene display line; ContradictionEngine | Not family recall |
| **world_model** | `inference_consumed=True` (inject *wire*) | Live **level 2**, overlay `consumed_now=True` **prompt inject on** (LLM context, not P1). Earned this birth. P1 health now says **L2 (prompt inject on the conversational path, not family recall)** when promotion `level_name=active`. Do not demote. Do not treat inject as family recall. | LLM context only |
| **intent_shadow** | not consumed | **182 predictions / 210 obs**, rolling ~0.69, still `shadow`, 0 rescues. June-30 “25870/0 preds” is **stale**. Dead-wire was fixed. Heuristic still routes. **Do not flip.** | No |
| weight_room_gate | not consumed | `enforces=False` (P2 would-block). Overlay keeps `consumed_now=False` | No |

HRR dormant. Salience dormant (cold-start deadlock — do not “fix” by flipping advisory). Policy NN shadow. positive_memory **NN** orphaned (heuristic scalar broadcasts). Claim-friction teacher feed is already fixed (`cc04f08`); registry prose is stale.

Board: [Project 8](https://github.com/users/Nimdy/projects/8) is the ranked queue (Sequence). [Project 2](https://github.com/users/Nimdy/projects/2) is **closed archive**. **#83** is frozen notes. North star **#42**. Do not follow #2/#4/#5/#7 as “do now.” Parked issues are closed `not_planned` (L7 / Matrix remainder / domain NN / HRR / lidar / connectome L4 / thought P3 / self-sensing P3).

Do not skip ranker. Do not treat 9 `consumed_now` as “she routes with NNs.” Ranker is the recall mouth. Heuristic still routes. World-model inject is LLM context, not family facts.

`/api-reference` FastAPI inventory is **172** (was stamped 170; missing `GET /api/operator/tap/status`). HTML/CSS live after `./sync-desktop.sh`. `.py` needs a bounce. Git push is not a sync.

### #83 WS1 / WS2 / WS3 (operator-named 2026-09-01)

| Stream | Status | Mouth / authority |
|---|---|---|
| **WS1** curiosity critic OOS | **NOT CONFIRMED.** Honest stop. Shadow-forever. | No proposer swap, no STEP 4, no PTZ, no re-run |
| **WS2** VQA #24 | Coupled (`vqa_prompt` wraps the spoken question) | Lived |
| **WS2** TTS markdown | Coupled (`BrainTTS._clean_for_speech`) | Lived |
| **WS2** OSV P2 | Coupled **pre-TTS** in `_gate_text` when `p2_active_default()`. Env still default **off**. | Mouth does not cut until `OSV_P2_ACTIVE=true` on the process. **Ask before that bounce.** |
| **WS2** TBS-0 | Lived on `How are you?` flight `pre_speech` (stance=none, `injected=false`). | Do not concat into `_style_instruction` (TBS-2 / P4) |
| **WS2** thin soul STATUS/MEMORY | Lived soul-dial log. Phatic how-are-you stays STATUS. Articulator no longer speaks this-turn `STATUS` or cortex pair HUD. Persist spoken STATUS. | `native_voice` stays `not_born`. |
| **WS3** WR P1 | SI snapshot now shows `signals_lived` / `signals_synthetic`; `live_shadow_accuracy=None` until scored inference | **`enforces=False`** |
| **WS3** claim_classifier friction | Pairing prefers `origin==friction_correction` | Shadow student |
| **WS3** thought_trigger 24→13 | Encoder matches config. Collector pair on `THOUGHT_VALIDATION_OUTCOME`. data_feed branch exists | **Not** in `_TIER1_FOCUSES`. Still `blocked_by_design`. Thought Maturity P3 parked |
| **Small** Golden UNVALIDATED LEARNING | Exact-match informational. Native queue+OSV-gap reply. Persist spoken. | No LLM. No new facts. |
| **Small** `/learning` lab strip | Genesis / pending / last correction / P2 / identity. Sit starter points at Golden + `/v2/grounding`. | Not a 29th v2 page. |
| **Small** wellness nag cap | `_last_wellness_ts` 4h on volume **and** screen | Spark queue still operator-pull |
| **Small** P1 world-model label | Follows promotion `level_name` | L2 ≠ family recall |

STATUS phatic how-are-you lived on PID **21970**. Overnight remainder loaded on PID **22894** (TAP idle, freshness not stale). Do not TAP while operator sleeps. Do not flip `OSV_P2_ACTIVE` unless named.

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
8. GitHub **[Project 8](https://github.com/users/Nimdy/projects/8)** (Sequence) if the question is “what to build next.” **#83** is frozen notes.
