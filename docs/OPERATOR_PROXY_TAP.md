# Operator-proxy TAP — agents sit the real mind

**Date: 2026-08-31.** This is how an agent operates JARVIS as David without
forging the ear.

The LLM is voice, not the brain. Pytest of `CapabilityGate.check_text` is a
**contract pin**, not a sit. Dashboard `POST /api/chat` was a **bypass**
(LLM + L0, no router). It is retired. Synthetic gym audio **must never**
reach `handle_transcription`.

## What this is for

David’s original agent-assisted training vision: an agent talks to JARVIS
the same way David does — she routes, recalls, thinks, gates, speaks on
the Pi — so we can catch broken wires **before** it is only the user, the
brain, and the Pi. After a wipe, TAP is how we rehearse first-contact.
Dashboard pages (cockpit, maturity, nn fleet, prove, …) stay **instruments**;
the TAP sit is the proof they are not glass-box theater.

## The seam

```
Pi mic → VAD → STT ─────────────────────────────┐
                                                ▼
Agent/operator ──keyed TAP──► inject_operator_turn ──► handle_transcription
                                                │
                                                ▼
                              router → MEMORY/OSV/VISION/LLM → L0 → TTS → Pi
```

STT is the ear. TAP is a **second ear that only accepts text**. From
`handle_transcription` onward the mind is the same.

**Do not:** speak into the Pi mic; POST `/api/chat`; begin a synthetic gym
session; forge ECAPA/face scores; bump Face 0.55.

## Identity vs evidence

| Field | TAP does | TAP must not |
|---|---|---|
| This-turn speaker | Enrolled identity she already has (David). Unknown live face crop does not steal L3. | Rewrite fusion; fake ECAPA/face scores; hardcode a name string |
| L3 / about-me | Same person as the enrolled speaker — she already knows if it's him | Weaken guest lock |
| Writes | Provenance **`operator_proxy`** | Stamp `observed` / ear `user_claim` as if the mic heard it |
| Face/voice floors | Unchanged | Mint enrollment from a TAP |
| Voice-intent teacher | **Skipped** | Pretend TAP text is STT |
| Wake / addressee | Skipped (no audio) | Fake a wake score |
| Live collision | **Refuse** if a voice turn is in flight | Barge in as David while David is talking |

David for **scope**. Proxy for **evidence**. After a wipe, TAP can rehearse;
it still does not count as ear-earned enrollment unless David sits the
real line.

## Follow-up (not every sit)

The ear opens a short **FOLLOW_UP** window (~4s) after TTS. That is how David
says “yes” without a wake. TAP **skip-wake is not follow_up**. Always sending
`follow_up=True` steals a new question into enroll/camera/research continuation.

| | |
|---|---|
| New sit | `follow_up` omitted or `false` — default when the ear is IDLE |
| Continuation | ear is FOLLOW_UP, **or** last mouth invited (`expects_follow_up`), **or** client `{ "follow_up": true }` |
| Most replies | do **not** invite. “Have a great day” is not a follow-on. |

`GET /api/operator/tap/status` → `{ speaking, busy, ear, follow_up_listening, follow_up_remaining_s, last_spoken_invites_follow_up, last_conversation_id }`.

TAP `POST` body: `{ "text", "speaker", "follow_up": true\|false\|omit }`.
Returns also `follow_up`, `follow_up_reason`, `expects_follow_up`.

Busy = TTS or LISTENING capture. FOLLOW_UP is **not** busy — that is the continue window.

## API

`POST /api/operator/tap` (API key, same as other operator writes).

```json
{ "text": "Jarvis, what do you remember about me?", "speaker": "David" }
```

Returns `{ ok, conversation_id, route, spoken, speaker, provenance, refused, follow_up, expects_follow_up, ear, ... }`.
Busy / synthetic-session-active → HTTP 409, no turn.

Ledger: `~/.jarvis/operator_tap.jsonl`.

## Chat box

`POST /api/chat` is **gone** (410). The v2 💬 control, if present, must hit
**this TAP**, not the LLM. Wiring the box to TAP is allowed; wiring it back
to `_response_gen.respond` is a process break.

## How an agent scores a sit

**Do not shotgun.** This is not a CRUD app. After a turn she still writes memory, identity stamp, ranking, teachers, L0, TTS. Stacking the next TAP in 5s races that.

1. `GET /api/operator/tap/status` — if `busy` or `speaking`, wait.
2. TAP the line (`follow_up` omit/false = new sit). One sit.
3. Wait until `busy=false` **and** `speaking=false` **and** the FOLLOW_UP window is over.
4. Then wait for the **write**: `OPERATOR-PROXY TAP` in `brain.log`, plus the expected `user_preference` / `personal_fact` row in `~/.jarvis/memories.json` if this turn taught a fact. Do not fire the next question on TTS-done alone.
5. Score the **mouth**. If the TAP log line is missing, the TAP did not happen.
6. Next sit only after that. Lived 2026-08-31: a 10-pack with 5s gaps invented Ethan and stored the lie as guest before Owen could be consumed.

Lived Pi voice remains the only proof of wake/VAD/STT/fusion/Kokoro hardware.
TAP proves the **mind** after the ear.

## How the rest of testing works (pytest stays)

Verbal conversation testing **always** goes through this TAP (or David’s
Pi mic). That is the only way an agent walks the mind.

The other tests we already have are **correct**. Do not delete them, skip
them, or treat them as sits.

| Kind | Where | Pass means | Fail means |
|---|---|---|---|
| **Sit** | TAP or Pi voice | Mouth + `OPERATOR-PROXY TAP` / STT in `brain.log` | Pytest green. Chat 410. Gym audio. |
| **Contract pin** | `brain/tests/` on WSL | The wire is still as drawn (L0 same-sentence sweep, about-me session-header skip, TAP provenance `operator_proxy`, `/api/chat` 410, follow_up not default). | A Stage 6 leftover is closed. |
| **Dashboard** | `/v2/*` after a sit | Instrument honest (`live_state`, verdicts). | She spoke the family. |
| **Synthetic gym** | Weight-room / telemetry | Student saw a pair. | Authority, enrollment, or a sit. |

Lived 2026-08-31: local pytest could not import `perception_orchestrator`
(no pydantic/fastapi in that venv) so TAP tests became **source pins** of
`app.py` / `perception_orchestrator.py`. That is still a valid pin. Run
the import-heavy suite where the venv exists. Never pytest on the live
brain host against `~/.jarvis`.

An agent that reports “tests passed” as Stage 6 closed has failed this
contract.

## Dashboard glass box

TAP does not replace `/v2/nnfleet` honesty (`live_state` + consumed). It lets
an agent **drive** a turn and then read those pages. Samples still ≠ control.
Soul-dial still `sent_to_model=False` until earned.
