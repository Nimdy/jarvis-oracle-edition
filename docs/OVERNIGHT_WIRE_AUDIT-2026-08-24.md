# Overnight wire audit — 2026-08-24 (after vision kitchen lie)

Operator went to bed. This is a **coupling / honesty** pass, not a new organ.
Branch: `fix/audit-real-bugs-2026-08-24` @ `3541336` (code) + this doc.
No gates flipped. No merge to `main`.

North star still #83: couple what exists. Integrity is the floor.

---

## Tonight's lived loop (closed)

| What | Evidence | Status |
|---|---|---|
| Pi captures, brain names the room | Hailo 1-class person; `yolov8n.onnx` **never** on the Pi; GPU caption of `/snapshot` lists desk / 3 monitors / chair / keyboard / mouse | **Architecture, not a missing Pi YOLO.** Do not put object detection on the Pi (WiFi / take-it-with-you). |
| Dashboard zeros vs old dashboard | v2 camera was tracker-only; old “data” was Person Detected + GPU caption | `/api/scene` now carries `caption` + `person_bbox_count`. Bounce showed person=1, 5 VLM candidates, coding surface. |
| “You’re still in the kitchen” | Caption was the desk. Spoken text invented stove / pot / cutting board from **cooking-dinner chat memories** + conversation history | VISION now skips memory/history; fail-closes ungrounded place claims to the caption. **Needs brain bounce** to load. Retry after user correction was already honest. |
| Person present after bounce | Pi `_was_person_present` latch dropped on disconnect | `person_detected` re-fired; `user_present=true` |

HRR stayed `PRE-MATURE`, zero influence. Empty HRR scene is tracker-empty, not “HRR down.”

---

## Turn lanes (AGENT_MAP)

P1 OSV → VISION (live JPEG + VLM) → about-X MEMORY → LLM.

VISION is the same class as P1: **LLM does not author the scene.** Cooking-dinner is not visual evidence.

---

## Remaining REAL (not gated)

Do these. Do **not** open Matrix / HRR / policy / P2 / native_voice.

1. **Bounce the brain** so `3541336` is the mouth. Until then kitchen can still leak on a first VISION turn if dinner chat is in history.
2. **Kitchen-lie memory is on disk.** `memories.json` / `conversation_history.json` stored the false kitchen reply. VISION no longer retrieves it; other LLM turns still might paraphrase it. Operator: next lab session, tell her that reply was false (calibration). Optional later: tag hallucinated VISION replies so they cannot re-enter recall. Relates to #12 honesty.
3. **#24 VQA targeting still open.** VISION still calls `describe_scene` with the **generic** prompt, then the text LLM answers “how many fingers / what color.” The kitchen miss is a different leak (memory override). Targeting the VLM with the user’s actual question is still the #24 gap. Do not add Pi processing.
4. **Hallucinated VISION replies persist as conversation memories** (see 2). Same honesty class as storing a wrong self-fact.
5. **Face 0.55** still often `known=false` (~0.39–0.55). **Do not tune.** Enroll path is “look at my face” (0758857). Wake-word still clears persist without face confirmation.
6. **#83 Workstream 2 still open:** TBS situational read fires **after** the reply; OSV P2 not flipped (do not flip tonight); TTS markdown strip; VQA prompt pass-through.
7. **Dashboard copy (same honesty class as #12, not new pages):** `identity.html` and `spatial.html` still hide `caption` / `person_bbox_count` — tracker 0 looks like “sees nothing.” `spatial.html` calls the HRR graph “canonical live scene” and mentions RealSense (this rig is Hailo + VLM). v1 `/mind` says she *believes* the room while PRE-MATURE. Camera paints 0 persons as amber (idle desk). Leave cockpit/cognition without caption (inner life ≠ vision).
8. **VISION persist (fixed same night):** LLM draft used to `remember()` *before* the fail-close swapped TTS to the caption. Now `persist_response=False` and `_persist_spoken_turn` stores what was actually spoken.
9. **Later, not tonight:** empty Hailo summaries still skip `SceneTracker.update`, so region-visibility never sees the body; brain `person_detected` is a rising-edge counter (reconnect can look like two people — do **not** lower 0.55); presence memories hardcode “desk”; `sensor_status`/`camera_state` events unused.

---

## EXPECTED (do not “fix”)

| Item | Why |
|---|---|
| Scene `visible_count=0` with VLM candidates | Promotion needs ~3 caption cycles. Candidates **are** the list. |
| Hailo `0 objects, 1 person` | Person-only HEF. Objects are brain VLM. |
| CPU YOLO disabled | Model absent **and** operator does not want Pi CPU for objects. |
| HRR PRE-MATURE, all influence false | Gate. Downstream of tracker. |
| Spark shadow, pending=0 | Operator-pull. #4 / #83. |
| native_voice / revoice / OSV P2 / L3 / SI stage | Do not flip. |
| Edge VLM on Hailo idle | Optional Pi load. Brain GPU caption is the room path. |

---

## Morning checklist (20 min, #83 protocol A)

1. Bounce brain if not on `3541336`.
2. `/v2/camera.html` — persons in frame, caption, entities (candidates OK).
3. Ask **“what do you see in the room?”** once. Must match caption (desk/monitors), not kitchen.
4. `/v2/grounding` — one correction if the kitchen lie is still in memory: “that kitchen description was wrong; I was in the office.”
5. Optional: “look at my face” if you want face confirm (threshold stays 0.55).

Do **not**: install `yolov8n.onnx` on the Pi; promote HRR; flip P2.

---

## GitHub tracker hygiene (this pass)

Comments on existing issues only — #83 said stop opening organs.

- **#24** — kitchen lie + caption-as-authority + VQA prompt still generic.
- **#83** — Workstream 2 VQA/integrity note; Pi=senses brain=VLM; branch pointer.
- **#12** — dashboard zeros vs caption; kitchen-lie stored as a memory.

Code: https://github.com/Nimdy/jarvis-oracle-edition/commit/3541336 (inventory + kitchen fail-close). Persist-spoken follow-up on the same branch after this doc.
