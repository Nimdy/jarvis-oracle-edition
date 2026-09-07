# Agent map — read this before you edit

**This is the unskippable contract.** `AGENTS.md` is the field manual.
This file is what the system *is*, how one spoken turn actually flows, and
what you must not invent. If a change is conversation, OSV, memory, routing,
TTS, or preferences: match a box on these diagrams first.

Roster: Grok 4.6 in-chair; Shockwave debug/docs; Megatron on Shockwave’s pulse.
Current branch / stage / leftovers: [NOW.md](NOW.md) (**wins** over any other
git/stage line). Frozen roster snapshot: [SHOCKWAVE_HANDOFF.md](SHOCKWAVE_HANDOFF.md).
Do not skip those for a fresh “the dashboard looks wrong” pass. **REAL ≠ live.**

Lived miss 2026-08-24 (`69d7819`): an agent added `briefing_register` /
exec-tech-ops mouths instead of using the path already drawn here.

**How we test:** verbal conversation testing always hits `handle_transcription`.
That is Pi STT **or** `POST /api/operator/tap` (second ear). Canon:
[OPERATOR_PROXY_TAP.md](OPERATOR_PROXY_TAP.md). `POST /api/chat` is retired
(410 — it skipped the router). Do not forge face/voice.

`brain/tests/` are **contract pins and they stay.** They catch L0 / aboutness /
TAP provenance / 410 regressions. Green pytest is not a Stage 6 sit. Synthetic
gym audio must never reach `handle_transcription`. Dashboard pages are
instruments; read them after a sit.

---

## What this is

Two devices. One mind. The LLM is **voice**, not the brain.

| Device | Role | Must not |
|---|---|---|
| **Pi 5** (`pi/`) | Senses + speakers: camera, Hailo **person+pose**, JPEG `/snapshot`, mic, playback, kiosk. Cheap on-device so the body can go mobile later. | Decide, remember, route, claim, **CPU YOLO room objects** (not the object path; operator takes the Pi on WiFi) |
| **Brain** (`brain/`, desktop GPU) | Perception finish, **VLM room caption** of the Pi JPEG, Layer 3B tracker (VLM objects + Hailo person occlusion), route, memory, OSV, policy, TTS. HRR mental-world is the later minds-eye (lost-phone / path-block / house-wear) and stays **PRE-MATURE / zero authority** until earned. | Be replaced by “just ask the LLM”. Do not put room inventory on the Pi. |

A process restart is **not a wipe**. Weights, memories, and promotion JSON persist.
Tier-2 *authority* re-earns. `current_ok` is live-sourced. Do not “fix” restart
by clearing `~/.jarvis`.

Shadow / `PRE-MATURE` / `not_born` / default-OFF means **wired, zero authority**.
That is not a missing feature. Do not build a parallel.

```mermaid
flowchart LR
  subgraph pi [Pi 5 — senses]
    mic[Mic]
    cam[Camera / Hailo]
    spk[Speaker]
  end
  subgraph brain [Brain — source of truth]
    perc[handle_transcription — the spoken mind]
    route[Tool router + P1 overrides]
    mind[OSV / memory / LLM-as-voice]
    tts[Kokoro TTS]
  end
  tap[Operator TAP]
  mic -->|PCM ws:9100 STT| perc
  tap -->|text inject operator_proxy| perc
  cam -->|PerceptionEvent| perc
  perc --> route --> mind --> tts -->|audio_b64| spk
```

---

## One spoken turn (this is the thing agents break)

Validated against `conversation_handler.py` (2026-08-24). Order is load-bearing.

```mermaid
flowchart TD
  stt["STT or TAP → handle_transcription"]
  intel[Personal intel + response_style store]
  tbs["TBS-0 ToM read — SHADOW, injects nothing"]
  router[tool_router.route]
  p1{classify_self_question?}
  vis{VISION look / VQA / retry after wrong?}
  about{about-X subject and not a P1 kind?}
  osv[P1: articulate_self_view — grounded floor speaks]
  eyes["Pi JPEG snapshot + brain VLM caption"]
  visSpeak[Mouth reports caption; history/memory do not locate the room]
  mem[MEMORY: search_memory]
  llm[LLM path: prefs + length hint in prompt]
  gate[CapabilityGate + output release]
  speak[TTS to Pi]
  seed["voice_seed / revoice — AFTER speech, teacher only"]

  stt --> intel --> tbs --> router --> p1
  p1 -->|yes: identity capabilities continuity answer_path ...| osv
  p1 -->|no| vis
  vis -->|yes| eyes --> visSpeak
  vis -->|no| about
  about -->|yes| mem
  about -->|no| llm
  osv --> gate --> speak --> seed
  visSpeak --> gate --> speak
  mem --> gate --> speak
  llm --> gate --> speak
```

| Lane | When | What speaks | LLM authors facts? |
|---|---|---|---|
| **P1 OSV** | Self-question classified (`articulate.py` kinds) | Deterministic articulator | **No.** Revoice is teacher-only until `native_voice` is born |
| **VISION** | Look / what do you see, or a targeted visual question (how many / what color / holding / on-off in the current frame). **Retry after wrong:** “that is/that's wrong”, “check again”, “try again”, “look again” when the prior look is still in-window. | Live Pi JPEG + brain VLM. Generic look speaks the caption. Targeted VQA passes the user's question into `describe_scene` and speaks the VLM answer (or declines). Finger-counts are not room inventory. Retry takes a **new** grab with the original question plus the correction; it does not let the text LLM agree without looking. Teacher label for the retry is VISION (`follow_up_retry`). | **No scene.** Dinner-chat must not name the room. Fail-close kitchen/stove/… if caption lacks them. Lived miss 2026-08-24. Lived miss 2026-08-25: “check again” after a finger count routed NONE. Do not add a Golden command per question. Do not steal MEMORY “remember when … that was a lie”. |
| **MEMORY** | About a person/pet/topic, not a P1 kind | Recalled payloads, first-sentence aboutness | No self-facts from OSV dump |
| **LLM** | Everything else | Ollama as voice, under L0 | Must not invent tools, jobs, or self-metrics |

Live chooser today: **heuristic keyword router**. Voice-intent NN is **shadow**
and does not pick the turn. Do not “fix routing” by adding a phrase regex for
one prompt. Log friction; fix the class.

---

## Who is the mouth on P1

```mermaid
flowchart LR
  osv[OSV model — full facts]
  art[articulate_self_view — grounded floor]
  tts[What she says live]
  rv[revoice.py — gist, offer to go deeper]
  nn["native_voice student — not_born"]

  osv --> art --> tts
  art -.->|teacher pair after TTS| rv -.->|corpus| nn
  nn -.->|only once earned| tts
```

- Spec-sheet on TTS = the floor. **Expected** until `native_voice` is born.
- `revoice.py` already writes *“lead with the gist”*. It does **not** speak.
- `response_style` (concise/detailed), `_policy_response_length`, ToM
  `verbosity_pref`, TBS-0 `lean_concise` already exist. They hit the **LLM**
  path. P1 bypasses them **on purpose**.
- Do not add `briefing_register` or exec/tech/ops articulators.

---

## Memory: one write path, one *spoken* recall path

Agents keep fucking this up. Fractal, dream, HRR, salience, belief-graph, and
ranker are **not missing**. They are **not the mouth**. Full table: [NOW.md](NOW.md)
§ STOP — spoken memory vs silent memory.

**Write** (`engine.remember`): synthetic-block → identity-stamp → quarantine
→ salience (NN gated, heuristic fallback) → store → vector index → `MEMORY_WRITE`.
Banter/soft tastes downgrade to `casual_conversation`. Do not add a second store.
Dream promotion, when it happens, still comes through this gate (`dream_observer`).

**Spoken recall** (`search_memory`): vector → **L3 personal security** (identity
boundary) → ranker → aboutness / wipe-skip → **native MEMORY formatter or**
LLM+pref inject → **CapabilityGate L0** → TTS.

About-me = **this-turn speaker**, not a hardcoded companion. About-X comes
from the query, not soul `known_names`. First sentence must be about the
subject. Curiosity asks are not autobiography. OSV-contradicted wipe claims
stay in the store (never discard) and must not be recalled as fact.

**Silent (must not be coupled into speech to “fix” a sit):**

| Lane | Speaks? |
|---|---|
| Fractal recall (cue 0.40; dream provenance fitness &lt; 0; never writes canonical) | **Never** |
| Dream cycle / `dream_observer` / dream_synthesis student | Not autobiography |
| HRR / spatial album (PRE-MATURE, album OFF) | No |
| Belief graph | No |
| Salience NN | No (write advisory) |

Native MEMORY (`_format_personal_activity_memory_reply`) is the **epistemic
floor** for about-X: fail-closed to retrieved payloads. Do **not** steal
about-me onto the LLM so it sounds warmer. Do **not** emit
“I can pull more details if you want” — that is a tool-shaped claim; L0 is
**correct** to kill it. Do **not** remove `music`/`dance` from L0 blocked
verbs because a *user* preference sentence was stripped; the sweep must
require first-person **in the same sentence** as the verb, which is already
the gate’s contract.

### Personal security (L3) — this is a lock, not a miss

A guest must not hear David's dog, preferences, or family. The fail-closed
line ("I don't have a specific memory recorded about that") for the *wrong
person* is **correct**. Empty recall for the *right* person is a stamp bug
in `identity/resolver.py`, not a reason to weaken `_policy_guest`.

```mermaid
flowchart TD
  q[This-turn speaker]
  stamp[resolver.resolve_for_memory]
  vec[vector candidates]
  l3{L3 personal security}
  speak[Native MEMORY speaks hits]
  closed["Fail-closed: I don't have that recorded"]

  q --> stamp --> vec --> l3
  l3 -->|same person primary_user / known_human| speak
  l3 -->|guest or other person| closed
```

Lived 2026-08-24 17:40: David was mis-stamped **guest**, so the lock hid
Skyler from *him*. Feature working, querier wrong. Do not "fix Skyler" by
letting guests through.

---

## Symptom → existing path (do not invent)

| Inner thoughts, never asks you | Spark would-have → `GroundingQueue` (operator-pull, no TTS). Advisory TTS only after 20 external answers | Flip `GroundingDrivePromotion`; nag TTS in shadow |
| She sounds / does | Existing path | Do not |
|---|---|---|
| Spec sheet on “what can you do” | P1 capabilities floor; mouth is `revoice` / `native_voice` | New register, shorter fake articulator as “source of truth” |
| Walk-through is a 98-list | Kind `answer_path`, not capabilities | Steal into capabilities |
| “Starting fresh” after restart | Kind `continuity` from measured store | LLM wipe narrative; do not clear `~/.jarvis` |
| About Skyler dumps OSV | About-X MEMORY override | INTROSPECTION because the query contains “you” |
| Guest / other person asked about Skyler and she “doesn’t know” | L3 personal security (`guest_blocked_personal`) | Weaken the boundary so recall “works for everyone” |
| “Look at me / see my face” only describes the shirt | IDENTITY enroll/refresh for this-turn speaker | Lower face 0.55; new biometric stack |
| David asked about Skyler and she “doesn’t know” | Stamp bug: querier typed guest. Fix resolver. Store still has the dog. | Delete memories or skip L3 |
| About me is empty / is the dog | Cue rewrite me → this-turn speaker | Hardcode David |
| About me dumps “first words this session” + “I don't have that capability yet” | Native MEMORY spoke session bookkeeping; formatter invited “pull more details”; **L0 correctly killed the invite** | Steal to LLM; skip L0; delete the session memory |
| “You like EDM” never spoken; coding-session / playlist pad | Pref inject fired; L0 residual sweep cut `music`/`dance` because first-person was in **another** sentence | Remove those verbs from L0; EDM whitelist |
| Fractal `no seed above 0.40` / dream didn't speak / HRR empty | Silent lanes. Success. | Wire them into TTS “for Stage 6” |
| Too long / too technical | `response_style`, length hint, ToM, TBS-0, revoice | `briefing_register` |
| Wrong tool | Intent class + `nn_fleet_registry.json` | One-off verb regex |
| Flag default-OFF | Governance. See `MATURITY_GATES_REFERENCE.md` | Treat as a bug and flip it |

---

## Operator constraints (this house)

- Do not merge to `main` unless asked.
- Do not start/stop supervisor, `main.py`, or lidar. Operator owns the stack.
- Do not run pytest on the live brain host against `~/.jarvis`. Tests write registries (lived 2026-08-24: `plugin_registry.json` overwritten, restored from snapshot). Run `brain/tests/` on WSL — they are contract pins, not sits.
- Agent verbal sits: `GET /api/operator/tap/status` then `POST /api/operator/tap`. Never `POST /api/chat`.
- **Never code on the brain or the Pi.** Edit on WSL. Sync `./sync-desktop.sh` / `./sync-pi.sh`. Operator bounces. Git push is not a sync.
- Sync with `./sync-desktop.sh` when code should hit the brain; they bounce.
- Do not flip `OSV_P2_ACTIVE`, revoice-live, voice-intent, or native_voice
  unless the operator asks.
- Do not wipe scars or memories.

---

## Where the long docs live (after this file)

| Need | File |
|---|---|
| Current branch / stage / leftovers | [NOW.md](NOW.md) |
| Agent field manual (VRAM, events, env) | `AGENTS.md` |
| Full architecture essay | `ARCHITECTURE.md` |
| Two-device mermaid (overview, older) | `docs/SYSTEM_OVERVIEW.md` |
| OSV P0–P5 | `docs/SELF_VIEW_DESIGN.md` |
| Gates | `docs/MATURITY_GATES_REFERENCE.md` |
| NN maturity / `not_born` | `brain/nn_fleet_registry.json` |
| Spark / curiosity | `docs/SPARK_DESIGN.md` |
| Companion / ToM | `docs/COMPANION_COGNITION_DESIGN.md` |
| TBS shadow | `docs/THINK_BEFORE_SPEAK.md` |
| First hour | `docs/FIRST_HOUR_AS_A_RESEARCHER.md` |
| dashboardV2 wiring / maturity ladder | `docs/V2_SURFACE_TRUTH.md` |
| Ground-truth audit 2026-08-24 | `docs/GROUND_TRUTH_AUDIT-2026-08-24.md` |
