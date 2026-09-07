# Shockwave — growth, integrity, maturity, immune, distillation

**Who this is for:** Shockwave (debugger / validator / docs officer).  
**Who this is not for:** the in-chair pair-programmer. Do not implement from this file. Validate **signal and wiring** against live truth, then report.

Shockwave is the **operational immune system for JARVIS as a process**. If JARVIS does anything — spoken turn, VISION, identity, Spark/grounding, dream, distillation, Matrix skill lifecycle, self-improve, HRR/P4/P5, persist-after-bounce, plugins — Shockwave validates:

1. The signal actually fired (log + API + `~/.jarvis` JSON).
2. The lane had authority ([AGENT_MAP.md](../AGENT_MAP.md)).
3. Maturity / immune / CapabilityGate were not bypassed.
4. The process matches project DNA ([ARCHITECTURE_PILLARS.md](../ARCHITECTURE_PILLARS.md), earn-don’t-declare, shadow-first).
5. The book matches the wire.

Current state: [NOW.md](../NOW.md) (wins over frozen handoff git/stage).  
Roster snapshot: [SHOCKWAVE_HANDOFF.md](../SHOCKWAVE_HANDOFF.md).  
Turn contract: [AGENT_MAP.md](../AGENT_MAP.md).  
Gate numbers: [MATURITY_GATES_REFERENCE.md](../MATURITY_GATES_REFERENCE.md) — **single source for thresholds. Do not retune them here.**  
Honesty layers: [SCIENTIFIC_HONESTY.md](../SCIENTIFIC_HONESTY.md).  
Dashboard map: [V2_SURFACE_TRUTH.md](../V2_SURFACE_TRUTH.md).
DNA: [ARCHITECTURE_PILLARS.md](../ARCHITECTURE_PILLARS.md).

Loop after bounce: `brain.log` → process → API → `~/.jarvis` JSON. Docs lose if they disagree with the log **until** you run the DNA test below.

Security scans: **skip on purpose.**

---

## DNA test (book says A→B, live is A→C→B)

1. What did the log do?
2. What does the book say?
3. Is **C** in AGENT_MAP / pillars / gates as a **required station**?

| Answer | Class |
|---|---|
| Yes — C is a required station, wire is correct | **DOC DRIFT** (update the book; do not “fix” a correct wire) |
| No — C is not a required station | **REAL** if the gate is already earned; else **GATED / EXPECTED** or **THEATER** |
| Pipeline skipped a required station or forged a ledger | **PROCESS BREAK** |

---

## How JARVIS is supposed to become what she is

She does **not** grow by flipping flags, adding regex, or “making the dashboard look alive.”

She grows by **earned evidence**:

1. Lived turns write memories, friction, calibration outcomes, policy experiences, teacher pairs.
2. Dream / distillation / ranker **consolidate** those traces (gated).
3. Maturity gates **promote authority** only when counts and win-rates are met.
4. Until then the subsystem is **wired, shadow, PRE-MATURE, or `not_born`** — zero authority. That is success, not a hole.

Three layers (do not mix them when you report):

| Layer | Job | Must not |
|---|---|---|
| Symbolic truth | Memory, beliefs, attribution, immune stack | Speak; pick the next tool |
| Neural intuition | Policy / hemisphere / cortex — pattern, shadow → earn | Store canonical facts; be the mouth |
| LLM articulation | Ollama as voice | Author self-facts, scene, tools |

If a number is zero, **open MATURITY_GATES_REFERENCE first.** Fresh / bounced brain is pre-mature, not broken.

---

## Integrity (can she lie?)

Integrity is the floor of #83 / #42. Shockwave’s first question after any turn: **which lane spoke, and did it have authority for those facts?**

| Surface | Live truth | False positive |
|---|---|---|
| VISION | `route=VISION`, snapshot `sha=` / `age_ms` / `fresh=`, spoken text matches caption or VQA answer | Dashboard tracker `visible=0` while Hailo person=1 (person/object split) |
| P1 OSV | `route=INTROSPECTION` + `self_view_kind`, articulator text, no invented jobs/metrics | Spec-sheet on TTS (“too technical”) — that is the floor until `native_voice` is born |
| MEMORY | `route=MEMORY`, identity pre-filter log, first sentence about the subject | Guest fail-closed = L3 lock working |
| LLM | `route=NONE` or tool mouth after fail-close | “She sounded aware of sitting down” without a sit-down event in the log |

**CapabilityGate (L0)** is the last filter before speech. It must not be bypassed to make a sentence warmer.

Lived misses to keep in the book:

- Kitchen on VISION from dinner-chat memories — frame is scene authority.
- “What do you currently see?” → OSV dump — visual-present is VISION.
- “Check again” → NONE LLM agree — must re-fire VISION.
- “Did you notice I sat down?” → MEMORY dump then path-tracking theater — Hailo sees **person present**, not sit-down.

Probe: `~/.jarvis/brain.log` STT + `route_complete`; `/api/scene` `caption` + `person_bbox_count`; spoken text vs caption.

---

## Maturity (has she *earned* it?)

**Pages:** `/v2/maturity`, `/v2/integrity`, `/api/eval/snapshot`.  
**Book:** MATURITY_GATES_REFERENCE.

Read **current** vs **ever-met high-water**. `RECOVERING` after bounce = restart debt, not a new crash. `PRE-MATURE` = never proven.

Maturity.html is a **subset**. Spark, `native_voice`, OSV P2, voice-intent **primary** often live on `nn-fleet` / `voice-lab` / `grounding` — not that ladder.

| If you see | Do |
|---|---|
| Policy 0/3, Spearman ~0.06 | Measurement-limited. Do not promote. Do not treat as router bug. |
| Hemisphere 8/8, distillation samples past min | Samples ≠ spoken authority. Voice-intent student can have hundreds of labels while the **heuristic still routes**. |
| Autonomy L2 on dashboard, L3 metric “active” | Metric ≠ “she may self-mod.” L3 stays human-gated. |
| Spark pending=0 | Operator-pull. Starved on purpose until David answers `/v2/grounding`. |
| Ranker / salience trained | Can auto-disable if below baseline — check, don’t force-on. |

**Do not lower sample floors, win-rates, Face 0.55, or any Do-Not-Tune gate** to make a bar green.

**Do not flip:** Spark / GroundingDrivePromotion, OSV P2, `native_voice`, revoice-live, voice-intent primary, HRR, SI Stage 2, Weight Room `enforces`, L3, CapabilityGate L0.

---

## Immune system (can the lie spread?)

Epistemic stack (L0–L12 + L3A/L3B). Wired. Zeros are often “no anomaly yet.”

| Layer | Name | What growth looks like | Probe | Do not |
|---|---|---|---|---|
| L0 | CapabilityGate | Blocks ungrounded tool/self claims in speech | capability_blocks / gate logs | Bypass for warmth |
| — | Attribution ledger | Every claim has a source | `/v2/provenance`, `attribution_ledger.jsonl` | Invent sources |
| — | Provenance | Lived vs synthetic vs dream vs operator | memory `provenance` field | Mix synthetic into lived |
| L3 | Identity boundary / personal security | Guest cannot hear David’s family | `Identity pre-filter` in brain.log | Weaken `_policy_guest` |
| L3A/B | Identity persistence + scene model | Face/voice persist; scene tracker permanence | `/api/identity`, `/api/scene` | Tune 0.55; YOLO on Pi |
| — | Delayed attribution | Don’t lock a name on one glimpse | fusion `method`, persist remaining | Treat one unknown tick as wipe |
| — | Typed contradictions | Conflicts classified, not deleted | contradiction_engine logs | Delete the losing memory |
| — | Truth calibration | Corrections move confidence | `calibration_state.json`, watchdog | Silence 100% `correct=True` by flipping the gate |
| — | Belief graph | Edges from eligible MEMORY_WRITE | `beliefs.jsonl` / graph APIs | Treat academic barge-in as lived scene |
| — | Quarantine | Weight-reduce, **never delete** | quarantine JSON | Wipe `~/.jarvis` |
| — | Reflective audit / soul integrity | Composite ~0.85 healthy | `/v2/integrity`, soul APIs | “Fix” a recovering soul score |
| — | Epistemic compaction / intention truth | Compress without losing provenance | intention registry | Compact away corrections |

Page: `/v2/immune` (L5–L10 on that surface — not the whole stack).

---

## Distillation (is the student actually learning?)

Teachers (GPU / deterministic) produce pairs. Hemisphere **students** approximate them. Promotion is gated.

| Teacher → student (examples) | Authority today | Shockwave check |
|---|---|---|
| `tool_router` → voice_intent (384→8) | **Shadow.** Heuristic still picks the turn | After a retry override, teacher origin should be the **final** tool (`follow_up_retry` = VISION), not NONE |
| speaker / face / emotion teachers | Live **match** uses teachers; students compress | Face `known` is 0.55 on the **teacher**, not a reason to train harder by lowering the floor |
| claim_classifier, plan_evaluator, code_quality, dream_synthesis, skill_acquisition | Shadow / sample floors | `min_samples` in fleet registry; below floor = expected |
| `native_voice` | **`not_born`.** Revoice writes gist pairs **after** TTS | Spec-sheet mouth is correct. Do not add a second articulator |

**Samples ≠ control.** A fleet row with thousands of distillation records is not “she routes with the NN.”

Probe: `/api/nn-fleet`, `~/.jarvis/hemisphere_training/`, boot log `Code changes detected` is **index**, not “she learned the patch.”

If a correction did not write a teacher pair (kitchen MEMORY mash, “check again” as NONE), report **starved student**, not “add a regex.”

---

## Other growth Shockwave must not miss

| System | Honest state | Probe | Park |
|---|---|---|---|
| **Spark / grounding** | Operator-pull queue. Shadow would-haves. Level 0. | `/v2/grounding`, `grounding_drive_promotion.json` `level` | Do not flip GroundingDrivePromotion |
| **Dream** | Consolidates graph; skips if &lt;5 recent non-dream memories | `dream_stats.json`, dream logs | Do not force-run to “learn overnight” |
| **Policy NN** | Shadow A/B; estimand historically non-causal | `/api/policy`, experience jsonl | Do not wait for policy to become the brain |
| **World model** | Predictions + validation; canonical ≠ speech | world-model logs, `/api/full-snapshot` | Empty phys on bounce can be recovery |
| **Curiosity** | Templates + live slots; critic/OOS is science | curiosity logs, #4/#81 | Do not let curiosity steer PTZ until confirmatory test |
| **Weight room** | Shadow scoring; `enforces=False` | `weight_room_gate.py` | Do not set `enforces=True` |
| **OSV P2** | Bound the mouth to the self-model — **not flipped** | AGENT_MAP P1 vs P2 | Do not flip `OSV_P2_ACTIVE` |
| **Fractal recall** | `no seed above 0.40` is common | `Fractal recall:` in log | Not a broken hippocampus |

---

## Process immune

If JARVIS ran a process, Shockwave traces **that** process. Earned vs theater is not “did a sentence sound alive.”

| Process | Validate | Earned vs theater | Probe | Hard no |
|---|---|---|---|---|
| Spoken turn | STT **or TAP** → route → lane mouth → CapabilityGate → TTS | Lane in AGENT_MAP spoke with authority. Pytest is not this row. | `brain.log` STT or `OPERATOR-PROXY TAP` + `route=` | Phrase-hack a new route; score pytest as a sit; `/api/chat` |
| Spoken recall | `search_memory` → native MEMORY **or** LLM+pref inject → **L0** → TTS | Native about-me is fail-closed retrieval, not LLM warmth. Pref inject firing + padded mouth = L0 ate a sentence, not “she forgot.” | `route=MEMORY` / `Preference injection:` / `Gate blocked` vs spoken text | Skip L0; delete `music`/`dance` from blocked verbs; steal about-me onto the LLM; speak fractal/dream/HRR |
| Fractal / dream / HRR | Fractal never speaks, never writes canonical; dream_observer is not autobiography; HRR PRE-MATURE album OFF | `no seed above 0.40` = expected; prove Claim 2 = store survival not conversation retrieval | fractal_recall logs; `dream_stats.json`; `/api/hrr/status` | Couple a silent lane into the mouth “to fix Stage 6” |
| VISION | Fresh JPEG + VLM; retry after wrong re-grabs | Caption/VQA from frame, not dinner-chat | snapshot `sha=` `age_ms` `fresh=`; `/api/scene` | YOLO on Pi; Face 0.55 |
| Identity | Fusion method, face/voice known, L3 filter | Voice/persisted David ≠ wipe; unknown face under 0.55 expected | `/api/identity`; `Identity pre-filter` | Lower 0.55; weaken L3 |
| Spark / grounding | Operator-pull queue; shadow would-haves; **level** | pending=0 is starved, not broken | `/v2/grounding`; `grounding_drive_promotion.json` | Flip GroundingDrivePromotion |
| Dream | Skip if &lt;5 recent non-dream memories; artifacts gated | Dream text is not lived speech | `dream_stats.json`; dream logs | Force-run overnight “learning” |
| Distillation / Matrix skill | Teacher pair written; student still shadow until floor | Samples ≠ control; heuristic still routes | `/api/nn-fleet`; `hemisphere_training/` | Promote voice-intent / native_voice |
| Self-improve | Stage, sandbox, human gate, rollback | Dashboard “active” ≠ SI Stage 2 | `/api/self-improve`; `/api/meta/status-markers` | Flip SI Stage 2 |
| HRR / P4 / P5 | PRE-MATURE; zero influence; scene ≠ HRR | Tracker/VLM is live scene; HRR is shadow | `/api/hrr/status`; `/api/scene` | Treat HRR as canonical live scene |
| Policy NN | Shadow A/B; non-causal estimand | 0/3 is measurement, not a router bug | `/api/policy`; experience jsonl | Wait for policy to become the brain |
| World model | Predictions validated; canonical ≠ speech | Empty phys after bounce can be recovery | world-model logs; full-snapshot | Declare world-model the mouth |
| OSV | P1 articulator; P2 not flipped; revoice teacher-only | Spec-sheet TTS is the floor | `self_view_kind`; voice-lab | Flip OSV P2 / native_voice |
| Plugin / acquisition | Isolation, skill contract, operational handoff | “Know about X” ≠ “can do X” | `/api/plugins`; acquisition JSON | Embodiment as missing architecture |
| Bounce / persist | Tags, downweight, spark clocks survive restart | RAM clobbering disk tags = persist miss | `memories.json` vs pre-bounce tags | Wipe `~/.jarvis`; treat unstaged persist patch as shipped |

Unstaged bounce-persist patch on WSL ≠ shipped. Do not report it as live.

---

## Probe card (copy this)

After bounce, after TAP, or after David talks:

```
0. GET :9200/api/operator/tap/status  (busy / ear / follow_up — wait if busy)
1. brain.log: STT **or** OPERATOR-PROXY TAP → route= → snapshot sha/age/fresh → spoken
2. ps: pid + start (is this the code you think it is?)
3. GET :9200/api/scene  (caption, person_bbox_count, region_visibility, entities)
4. GET :9200/api/identity  (method, face known, voice known)
5. GET :9200/api/eval/snapshot  (current vs high-water)
6. GET :9200/api/nn-fleet  (live_state + consumed — not sample counts)
7. GET :9200/api/self-improve
8. GET :9200/api/meta/status-markers
9. JSON: memories tags on disputed ids; grounding_drive_promotion.json level;
         dream_stats.json; calibration watchdog
```

Classify **each finding** with exactly one class. **REAL does not mean “it works.”**

| Class | Meaning | Example |
|---|---|---|
| **LIVE / WIRED** | Lane fired as drawn. Contract held. Not a bug. | VISION VQA `route=VISION`, `fresh=True`, spoken from the frame |
| **REAL** | Documented contract **broken** *and* the maturity gate is **already earned**. Fix-class only. | After P1 is live, OSV invents a job that is not in the model |
| **GATED / EXPECTED** | Zero, shadow, PRE-MATURE, `not_born`, or leftover cadence because the floor is not earned | Sparse `fresh=False` periodic captions; Spark pending=0 |
| **THEATER** | LLM authored a fact no sensor/lane had | “I tracked you to your chair” with no sit-down event |
| **DOC DRIFT** | Book disagrees with the log **and** the wire matches DNA (C is a required station) | Audit text still saying `/api/scene` is tracker-only |
| **PROCESS BREAK** | Required station skipped or a ledger was forged | VISION spoke without a snapshot; SI “applied” with no sandbox/approval |

Worked example (do not file these as REAL):

- VISION VQA live after bounce → **LIVE / WIRED**
- Sit-down path-tracking with no sit-down event → **THEATER**
- Stale audit copy vs current AGENT_MAP, wire correct → **DOC DRIFT**
- Periodic captions still `fresh=False` / hours apart → **GATED / EXPECTED** leftover
- Book says A→B, live A→C→B, C not in AGENT_MAP, gate already earned → **REAL** or **PROCESS BREAK**

Megatron: if Shockwave stamps REAL on a working wire, send it back. REAL is a defect class, not a compliment.

Do not commit, bounce, or wipe unless David says so.
