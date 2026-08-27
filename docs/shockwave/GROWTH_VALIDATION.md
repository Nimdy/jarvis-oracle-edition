# Shockwave — growth, integrity, maturity, immune, distillation

**Who this is for:** Shockwave (debugger / validator / docs officer).  
**Who this is not for:** the in-chair pair-programmer. Do not implement from this file. Validate **signal and wiring** against live truth, then report.

Roster: [SHOCKWAVE_HANDOFF.md](../SHOCKWAVE_HANDOFF.md).  
Turn contract: [AGENT_MAP.md](../AGENT_MAP.md).  
Gate numbers: [MATURITY_GATES_REFERENCE.md](../MATURITY_GATES_REFERENCE.md) — **single source for thresholds. Do not retune them here.**  
Honesty layers: [SCIENTIFIC_HONESTY.md](../SCIENTIFIC_HONESTY.md).  
Dashboard map: [V2_SURFACE_TRUTH.md](../V2_SURFACE_TRUTH.md).

Loop after bounce: `brain.log` → process → API → `~/.jarvis` JSON. Docs lose if they disagree with the log.

Security scans: **skip on purpose.**

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

## Probe card (copy this)

After bounce or after David talks:

```
1. brain.log: STT → route= → snapshot sha/age/fresh → spoken
2. ps: pid + start (is this the code you think it is?)
3. GET :9200/api/scene  (caption, person_bbox_count, region_visibility, entities)
4. GET :9200/api/identity  (method, face known, voice known)
5. GET :9200/api/eval/snapshot  (current vs high-water)
6. GET :9200/api/nn-fleet  (shadow / not_born / sample counts)
7. JSON: memories tags on disputed ids; grounding_drive_promotion.json level;
         dream_stats.json; calibration watchdog
```

Classify **each finding** with exactly one class. **REAL does not mean “it works.”**

| Class | Meaning | Example |
|---|---|---|
| **LIVE / WIRED** | Lane fired as drawn. Contract held. Not a bug. | VISION VQA `route=VISION`, `fresh=True`, spoken from the frame |
| **REAL** | Documented contract **broken** *and* the maturity gate is **already earned**. Fix-class only. | After P1 is live, OSV invents a job that is not in the model |
| **GATED / EXPECTED** | Zero, shadow, PRE-MATURE, `not_born`, or leftover cadence because the floor is not earned | Sparse `fresh=False` periodic captions; Spark pending=0 |
| **THEATER** | LLM authored a fact no sensor/lane had | “I tracked you to your chair” with no sit-down event |
| **DOC DRIFT** | Book disagrees with the log | Audit text still saying `/api/scene` is tracker-only |

Worked example (do not file these as REAL):

- VISION VQA live after bounce → **LIVE / WIRED**
- Sit-down path-tracking with no sit-down event → **THEATER**
- Stale audit copy vs current AGENT_MAP → **DOC DRIFT**
- Periodic captions still `fresh=False` / hours apart → **GATED / EXPECTED** leftover

Megatron: if Shockwave stamps REAL on a working wire, send it back. REAL is a defect class, not a compliment.

Do not commit, bounce, or wipe unless David says so.
