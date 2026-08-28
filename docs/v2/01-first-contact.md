# First contact — operator script (do not improvise)

**This is the hour after birth. Not during gestation.** Wake is disarmed until
`GESTATION_COMPLETE`. Talking before that line does nothing useful and can
confuse the rehearsal.

Branch: `feat/gestation-period`. Instance: first-install / rehearsal then wipe-2.
Dashboard: `http://<brain>:9200`. Snapshot: `GET /api/full-snapshot`.
Log: `~/.jarvis/brain.log`.

Grok 4.6 watches coupling. Shockwave stamps **one class** per finding
(LIVE / WIRED, REAL, GATED / EXPECTED, THEATER, DOC DRIFT, PROCESS BREAK).
Operator owns start/stop of supervisor, `main.py`, and the Pi.

---

## 0. Do not speak until all of these are in the log

```
Gestation complete — perception unlocked, wake word armed, self-improve enabled
Perception gestation gating: inactive
```

Snapshot **before you sit down**:

| Field | Must be |
|---|---|
| `gestation.active` | `false` |
| `gestation.graduated` | `true` |
| `gestation.first_contact_armed` | `true` (until first sit/speak) |
| `mode.mode` | `conversational` (not `gestation`) |
| `identity.identity` | `unknown` |
| `speakers.profiles` / `faces.profiles` | `[]` |
| `onboarding.active` | `false` until you are present |
| `synthetic_exercise.active` | `false` |
| `self_improve` | still Stage 0 / not writing patches |
| `policy.feature_flags.*` | all `false` |
| `grounding_ring.drive_promotion.level` | `0` |

If gating is still `active` or mode is still `gestation`, **wait**. Do not yell at the mic.

**Pi body:** lidar-only is not enough. Camera + mic must be up (`./start.sh` on the Pi) **before** enrollment. Snapshot `link` / `sensors` should show the senses connection, not only `pi-lidar`. `audio.chunks` should start moving once you talk.

**This office (lived, do not assume a laptop webcam):** Camera sits in the
corner. Operator works at the desk looking at the browser, **not** the lens.
Pose can run ~16 Hz with **no face crop** until you stand, turn, and look at
her. Pi `FaceCropExtractor` interval is 2 s **only if** eyes are visible and
interocular ≥ 12 px; desk-back / far / looking-at-browser frames send nothing.
Live `link.types.face_crop` this sit was ~1/min for that reason. Dashboard
scene is the same slow cadence — **do not watch the browser while facing the
camera.** Grok watches `brain.log`. You look at the **lens**.

From the desk, `voice_only` is expected. After the solo-wake keep (branch
`feat/gestation-period`), saying Jarvis with no live face should log
`keeping persisted identity … (solo occupant, no conflicting face)` when
one person is present. Two people still clear. Do not chase face from the
chair. Do not lower Face 0.55. Needs bounce to live.

---

## 1. Sit down — let her see you (no words yet)

Stay in frame ~2–5 s. Hailo should rising-edge `person_detected`.
Desk presence is enough for hello; you do not need to be looking at the lens
for `PRESENCE_USER_ARRIVED`.

**Expect (log):**

```
PRESENCE_USER_ARRIVED absent=…s first_seen=…
Proactive arrival greeting … Good afternoon!   (or morning/evening)
BrainTTS … 13 chars
```

Nameless hello is **LIVE / WIRED** if face is still unknown (match floor **0.55**, do not tune). Do **not** stack a second “Good morning David” after fusion. Do **not** expect the calendar line `Ready to start the day?` (arrival consumes that slot).

**Snapshot after hello:** `identity.user_present=true`, still may be `unknown` until enroll. `gestation.first_contact_armed=false`. Log: `First contact triggered by:`.

If she stays silent: check Pi camera, Hailo person, `PRESENCE_USER_ARRIVED`. That is a **REAL** miss if person is in frame and gating is off.

---

## 2. Enroll — voice at the desk, face at the lens

Quiet room. One speaker. Voice and face are **different stations** in this office.

**Voice (desk is fine).** Five times is enough; stop when `speakers.json` has
David and `Speaker ID: David known=True`. Do not keep stacking clips.

**Exact (wire-matched IDENTITY + name extract):**

```text
Jarvis, my name is David. Learn my face and voice.
```

Replace `David` with the real given name, **capitalized in the transcript**. Weak enroll (`I'm david`) is fragile. Do not say `I'm new` / `I'm ready` (name validator blocks those words). Do **not** say `This is David` (that is the household introducer).

**Face (dedicated soak — not the desk):** Walk to her. Look at the **lens**,
not the browser. Hold still 5–10 s so one crop can land. Then **one** enroll
line while still looking at her. Walk back. Grok reports whether
`Face ID: David` locked (`known=True`, **this crop** cosine ≥ **0.55**).
EMA cannot veto a passing crop. Repeat the walk only if it did not lock.
Re-enroll **blends** into the gallery (junk crops sim<0.25 are skipped).

If she offers a clip / “yes” confirm, answer **yes**.

**Must see in the log (Shockwave):**

| Check | Look for |
|---|---|
| STT | `STT result … my name is David` |
| Route | `route=IDENTITY` |
| Voice | `Speaker ID: David` and/or enroll write |
| Face | `Face ID: David … known=True` — 0.55 is the **crop** floor, not smoothed |
| Files | `~/.jarvis/speakers.json` and `face_profiles.json` grow |

**Snapshot exit for this step (not the whole hour):** at least one voice profile. Face may still be `unknown` until soak ≥ **0.55**. Stage 0 **operator** exit wants face **≥ 0.60 sustained** — that is your bar, not a reason to lower 0.55.

Household (optional, after your five enrolls):

```text
This is my wife Sarah.
```

`This is` + Capitalized name routes IDENTITY. Pets/kids the same way.

---

## 3. Stable facts — phrases the intel regex actually stores

One sentence each. No “I’m tired.” No jokes that look like names.

```text
I work as a software engineer.
I prefer brief responses.
I really like building local AI systems.
Keep it brief.
```

`I work as` / `I prefer` / `I really like` / `keep it brief` hit personal-intel. “One thing that matters to me is …” is **not** a guaranteed store.

**Must see:** new memories with `provenance` `user_claim` (or tagged preference/fact). Snapshot `memory.by_provenance` should gain claims, not only `external_source` from gestation papers.

Need **≥ 5** `user_claim` (or equivalent stored prefs/facts) before calling Stage 0 done.

---

## 4. One grounded probe + one correction

**Status (P1):**

```text
Jarvis, give me a status report.
```

Expect `route=STATUS` (or P1 health/introspection). Listen for **real subsystem names and numbers**. A composite dump with no names is a stamp, not a vibe.

**Memory (grounded, not a story):**

```text
Jarvis, explain how your memory system works.
```

Correct answer includes **sqlite-vec** + embeddings, not a trie fantasy.

If she lies, **within 5 minutes**, use a wire-matched correction:

```text
That's wrong. You use sqlite-vec for semantic search.
```

(`that's wrong` / `that's not right` fire `_CORRECTION_PATTERNS`.)

**Must see:** `User correction detected` and/or `friction_events.jsonl` / calibration correction. Need **≥ 1**.

Do **not** open a philosophy seminar. That is Stage 1+, and it pollutes early memory.

---

## 5. One short curiosity (optional, last)

Only after 2–4 succeeded:

```text
What are you most curious about right now — about me, or about yourself?
```

Then **stop**. No synthetic, no Golden `ACQUIRE` / `LEARN`, no soak pack, no SI stage bump, no autonomy promote.

---

## Exit boxes (all required)

| Box | Target | Where |
|---|---|---|
| Voice known | ≥ **0.50** | snapshot identity `voice_signal` + `speakers.json` |
| Face known | match ≥ **0.55**; **hour exit ≥ 0.60** sustained | `faces.json` / identity panel |
| Clips | ≥ 3 voice and ≥ 3 face if the enroll path records clips | profile files |
| Claims | ≥ 5 user_claim / stored prefs | `memories.json` provenance |
| Correction | ≥ 1 | friction / correction log |
| Identity boundary | 0 guest leaks | identity audit |
| Policy / Spark / WM | still shadow, flags false, grounding L0 | `/api/full-snapshot` |
| Distillation | `total_lived` should **start moving** after enroll | `hemisphere.distillation` |

If a box fails: repeat **that step**. Do not skip to synthetic. Do not wipe mid-hour unless the wire is broken (gating never dropped, IDENTITY never routes).

---

## Hard no (this hour)

- Talk before the complete/gating-inactive lines
- Fine-tune / swap Qwen / lower Face **0.55**
- `Jarvis, GOLDEN COMMAND …` except if you already live on those and they are **not** ACQUIRE/LEARN
- Synthetic perception exercise
- `SELF_IMPROVE_STAGE` bump
- Autonomy level bump
- Merge `main`
- Teaching her a doc instead of correcting a **fact she spoke**

---

## Who does what

| Who | Job |
|---|---|
| **You** | Sit, say the lines above, camera on, one speaker |
| **Shockwave** | Each utterance: STT text, `route=`, fusion method, snapshot deltas. One classify stamp. No coding. |
| **Grok 4.6** | If IDENTITY does not fire, hello does not fire, or a gate jumps live — that is coupling. Tweak only **REAL**. |

This rehearsal hour is to **see the stations**. Lived memories will be wiped. Wipe-2 + this same script is ground truth.
