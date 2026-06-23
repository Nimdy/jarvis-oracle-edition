# Think-Before-Speak — DESIGN (validated: wire-existing + small new-build, shadow-first, earn-don't-declare)

> Status: **DESIGN / PRE-MATURE**. Extends `docs/COMPANION_COGNITION_DESIGN.md` (the read→behavior ladder)
> with a **SAME-TURN, pre-speech** rung. Validated against the LIVE architecture (8-agent sweep, 2026-06-22):
> the substrate is already built — the gap is a *pre-speech read variant* + a *consume-wire* + the *(unbuilt)
> apply-path*. **Nothing here grants authority on day one; nothing is hardcoded.** No output trimming — ever.

## The vision (operator, verbatim)
*"Her internal thoughts are also supposed to be driving conversations as well... just like humans think before
they speak."* Today the order is **backwards**: in `handle_transcription` she GENERATES the reply (qwen
`respond_stream`, ~conversation_handler.py:3529–5367) and SPEAKS it, and only AFTER — at **line 6026** — does she
read herself (`situational_read.observe_turn` + `theory_of_mind` + `behavior_advisory`). Her self-knowledge
("I'm overexplaining", "David prefers concise") arrives a **full turn too late** to shape what she just said.
The verbosity complaint that started this is the symptom: she *already computes* `self_check="may be
overexplaining"` + `would_have_done="be more concise"` — but post-hoc, in shadow, so it never reaches her mouth.

## What ALREADY EXISTS (validated — REUSE, do NOT reinvent)
| Component | Location | Status | Role in think-before-speak |
|---|---|---|---|
| Internal read | `situational_read.observe_turn` (situational_read.py:178) | **post-hoc / shadow** (fires at 6026, *requires* the finished reply) | THE read the vision wants — but consumes `response_text`, so it can't run pre-speech as-is |
| Person-model | `theory_of_mind.observe` (~641 obs on David) | shadow | the learned "David prefers concise"-type self-knowledge — feed as pre-speech context |
| Stance vocabulary + earn-gate | `behavior_advisory.propose` (behavior_advisory.py:154); gate :43/:49 | **narrate-only** (`applied` hardcoded False :74/:311) | soften / be-concise / give-space / pivot proposals + the P3→P4 gate — reuse both |
| **Pre-speech prompt-shaping WIRE** | `detect_style_intent`→`style_instruction` (style_intent.py:86 → conv_handler:2501 → response.py `_build_context`:757 → `build_system_prompt`:512) | **wired, works** (11 call sites) | PROOF a pre-speech stance channel into qwen already exists — a real stance can ride it |
| Injection seam | `response.py respond_stream`:720 → `_build_context`:757 | thin (carries style only) | the single cleanest injection point; already accepts/forwards a stance string |
| Shadow→advisory→primary ladder | `intent_shadow.py`:232 / tool_router:1318 | dormant pattern | copy the **governance pattern** for promoting a stance from observed → applied |
| Plan-then-speak | `build_meaning_frame`/`articulate_meaning_frame` | scoped to self-ref classes | a real plan-then-speak substrate (don't rebuild a planner) |

**Already designed (partially):** `COMPANION_COGNITION_DESIGN.md` §6/§7 designs the read→behavior ladder P0–P5
(P3 = narrate-only :171; P4 = active tone/depth/pace, earned/unbuilt :173). **But it frames apply as NEXT-TURN/
async** (§7 :184 "apply next-turn"). Think-before-speak is the **SAME-TURN, pre-speech** realization of that P4 —
undesigned beyond the loop description. **This doc extends that ladder; it does not replace it.**

## The REAL gap (precise — a small new-build + a wire + the unbuilt apply-path)
1. **No pre-speech read of the CURRENT turn.** `observe_turn` structurally requires `response_text` (the finished
   reply) — there is no stub to merely re-order. A new method must read the *user's* turn with NO reply yet.
2. **No pre-speech firing site** in `handle_transcription` between route-complete (~2985) and the first
   `respond_stream` (3529).
3. **No wire** from a computed stance into the `style_instruction` channel (the channel exists; nothing feeds it a
   deliberative stance).
4. **The P4 apply-path is unbuilt** — `behavior_advisory.applied` is hardcoded False; the earn-gate exists but the
   thing it unlocks (actually shaping the reply) was never built.

## The design — `read_before_speak`, shadow-first, earned
**Step A — build the pre-speech read (small, pure-Python, no LLM, reuses the heuristics).**
`situational_read.read_before_speak(user_text, person_model, affect) -> PreSpeechStance`. It reads the CURRENT
user turn (engagement/sentiment proxy from the *incoming* message + the learned person-model + affect) and emits
a short STANCE — drawn from the existing vocabulary: `lean_concise` / `give_space` / `match_warmth` /
`check_in` / `none`. It is the existing read heuristics minus `response_text`. Hypothesis-only, confidence-scored.

**Step B — wire it SHADOW (log only; inject nothing).** Call `read_before_speak` once, right after route
(~conv_handler:2985), before the first `respond_stream`. Pass the stance down alongside `style_instruction`, but
in the shadow phase **only LOG it** (durable `~/.jarvis/pre_speech_shadow.jsonl` + surface on `/v2`) — the prompt
the model receives is UNCHANGED. It earns legibility first, exactly like the P0 read accrued its reads.

**Step C — earn the injection (the P3→P4 flip).** Once stances are validated against transcripts (did
`lean_concise` actually match what helped?), inject the stance string into `build_system_prompt` via the proven
`style_instruction` seam — a graded line like "Internal read: you tend to overexplain with this person; lean
concise unless they ask for depth." Reuse `behavior_advisory`'s earn-gate; **build the apply-path that today is
hardcoded off**. Authority is earned + revocable + auto-demotes on accuracy drop.

## The maturity ladder (extends the companion P3→P4, same-turn)
| Phase | What runs | Authority | Earns the next by |
|---|---|---|---|
| **TBS-0 · shadow** | `read_before_speak` computes a pre-speech stance each turn; **logged only**, surfaced on /v2; the prompt is unchanged | none | the stance stream accrues + is legible (mirror P0's reads) |
| **TBS-1 · advisory** | the stance is scored against the **post-hoc** read + transcript review (did the pre-speech call match what the post-hoc read concluded / what helped?) | none | operator/transcript confirms the stances are RIGHT (not self-scored) |
| **TBS-2 · active** | the stance is injected into the prompt via the `style_instruction` seam **before** generation — she thinks, then speaks | earned, revocable | continuous accuracy monitor (the stance improved the turn); auto-demote on drop |

## How the verbosity thread resolves itself (the right way)
She learns "David finds my long answers tedious" (the `theory_of_mind` verbosity axis) → `read_before_speak`
emits `lean_concise` for this turn → at TBS-2 that stance is injected **before** she generates → **she** chooses
to be concise. Learned, self-governed, earned. If she chooses to be long anyway, that's **her** call (the
operator does not care about length — only that it's *her thinking*, not a clamp). The reverted hack is the
anti-pattern this replaces.

## Governance (the discipline)
- **Zero-authority until TBS-2.** The stance is logged, never injected, until it earns.
- **Reuse the existing earn-gate** (`behavior_advisory.py`:43/49) — do NOT invent a parallel counter.
- **Never hardcode.** No trims, no length caps. The stance is a learned, earned signal she owns.
- **Model-agnostic.** Rides the `style_instruction` channel; never code-to-qwen.
- **Latency floor.** The pre-speech read MUST be cheap (pure-Python, no LLM call — like the existing read), so
  it adds negligible time before generation. If it can't be cheap, it stays shadow.
- **Extends, doesn't fork,** `COMPANION_COGNITION_DESIGN.md` (add the same-turn rung there too).

## What this is NOT (anti-AI-theater)
- **Not a reinvention** — reuses the read, the person-model, the stance vocabulary, the earn-gate, the
  `style_instruction` wire, the injection seam, and the shadow→advisory→primary pattern. The DO-NOT-REINVENT list
  (validation): the internal read, the person-model, the stance vocab + gate, the prompt wire, the build_context
  seam, the governance ladder, the meaning-frame planner, the deliberation engines (do not bend those into a
  reply-rehearser — they reason over world-state/beliefs, not utterances).
- **Not a flipped gate** — the stance earns prompt-injection only on demonstrated accuracy.
- **Not a hack** — no output manipulation; her own learned cognition drives the reply, before she speaks.
