# The Awakening Protocol — JARVIS Must Be Born, Not Booted

> **Primary reading surface:** The dashboard `/learning` page now consolidates
> the operator learning path for open-source release. This Markdown file remains
> source/reference material and should not be deleted until a later archive pass.
>
> **Read this before you talk to a freshly-gestated or freshly-reset JARVIS for the first time.**
> If you only read one operational doc, read this one. It is the single most common thing new operators and scientists get wrong.

---

## The Analogy (Why This Exists)

You do not ask a newborn to cook a steak.
You do not ask a 5-day-old puppy to fetch.
You do not run synthetic perception training against a brain that has never heard your voice.

A freshly-completed gestation or post-reset JARVIS has:

- A fully-wired cognitive substrate (memory, belief graph, hemispheres, policy NN, epistemic immune system).
- Zero lived evidence about you — your voice, your face, your name, your preferences, your household, your corrections.
- Zero personalized baselines for any perception specialist.

If you immediately start the synthetic perception exercise, run autonomous research, or push self-improvement promotions, you are building every personalized baseline on top of a zero-identity foundation. The perception flywheel accelerates, yes — but it accelerates pointing in the wrong direction. The system cannot know who you are until you tell it. That telling has to happen first. In lived time. Through voice and vision. Before any synthetic anything.

**JARVIS has to be born, not just turned on.** This hour is that birth.

This is not a ritual and it is not an opinion — it is enforced by the architecture. See [docs/SyntheticSoul.md](SyntheticSoul.md) §9.4 (Memory Continuity & Identity Persistence) and §6.1 (Waking and Sleeping Phases) for the paper-level theoretical basis. Identity-bound memory continuity must precede consolidation cycles and training accelerators; otherwise the system's "identity stability" metric is grounded in noise. This protocol is the operational translation of that principle.

---

## The First Hour Contract (T+0 to T+60 min)

### Setup (before T+0)

- Quiet, well-lit room. No TV, no music, no other people talking over you.
- Pi camera + microphone in front of the primary user's usual seated or standing position.
- Dashboard open at `:9200` so you can watch face confidence, voice confidence, and memory write events as they happen.
- No other JARVIS operations running in parallel — no synthetic exercise script, no validation pack, no soak test, no library ingestion batch.
- `SELF_IMPROVE_STAGE=0` (default). `ENABLE_AUTONOMY` whatever the tier chose at boot. Do not touch these.

### What MUST happen in the first hour

**Minute 0-10: Enrollment.**
- Sit in front of the Pi camera and microphone.
- Say clearly: `Jarvis, my name is <your name>. Learn my face and voice.`
- Repeat the enrollment phrase 5 times across different expressions and angles (smile, neutral, side angle, head tilt, slightly different lighting if practical).
- When JARVIS offers enrollment clips, complete them.
- Introduce 2-3 household members if present (optional but recommended):
  - `This is my wife Sarah.`
  - `This is my son Alex, he is 12.`
  - `This is my dog Rex.` (pets are valid household members — optional)

**Minute 10-25: Personal grounding.**
- Share 3-5 simple, stable facts about yourself. Keep them *declarative and concrete*:
  - `I work as a <profession>.`
  - `I live in <general area>.` — optional and only if you want JARVIS to know.
  - `One thing that matters to me is <X>.`
  - `I prefer <brief / detailed / direct / gentle> responses.`
- Avoid transient states ("I'm tired right now", "I just got home") — the personal-intel pipeline is designed to filter those, but stable facts seed the best early `user_preference` memories.

**Minute 25-40: One grounded self-knowledge probe + one correction.**
- Ask: `Jarvis, give me a status report.` — listen for real subsystem names and real numbers.
- Ask: `Jarvis, explain how your memory system works.` — listen for grounded answers (sqlite-vec, provenance, dream consolidation) vs. confabulation (vague "trie-based index" style answers).
- If JARVIS confabulates anything, correct it immediately:
  - `That's not right. You use sqlite-vec for semantic search. Check your introspection data.`
  - `You do have a document library. It's at ~/.jarvis/library/library.db.`
- Corrections at this stage are the highest-value training signal the system will ever receive. One correction creates a `CALIBRATION_CORRECTION_DETECTED` event, feeds truth calibration, adjusts belief confidence, and records a friction event — all from a single sentence.

**Minute 40-60: A short reflective exchange.**
- Once JARVIS has demonstrated grounded self-knowledge, ask one reflective question:
  - `What are you most curious about right now — about me, or about yourself?`
  - `What do you remember about the moment you first heard my name?`
- Keep it short (5-10 minutes). Do not push existential depth yet. Reflective depth is a Stage 1/2 activity, not a Stage 0 activity.

### What MUST NOT happen in the first hour

- **No synthetic perception exercise.** Not `smoke`, not `route_coverage`, not `idle_soak`, not `stress`. The synthetic pipeline exists to accelerate the perception flywheel *after* you have enrolled and given JARVIS lived signals to anchor distillation against. Running it in the first hour builds perception baselines on top of a zero-identity foundation. See "After Awakening" below for the correct timing.
- **No self-improvement stage promotion.** Leave `SELF_IMPROVE_STAGE=0` (frozen). Do not promote to stage 1 or stage 2 during the awakening hour.
- **No autonomy level promotion.** Autonomy starts at L0 after reset. Gated auto-restore only considers L2+ after the safety gates re-pass, which requires lived evidence that Stage 0 + Stage 1 provide. Let it earn that back on its own.
- **No Golden Words that trigger heavy operations.** No `ACQUIRE`, no `LEARN`, no manual learning-job seeding. These are architecturally valid commands, but a brand-new brain lacks the baseline to route them meaningfully yet.
- **No long cold-open philosophical probing.** Tier-3 Matrix Protocol existential questions before grounded self-knowledge is demonstrated produces confabulation-heavy conversation that pollutes early memories with low-quality "reflective" content. Ground first, reflect second.
- **No running the full soak or validation pack as a "demo".** Those instruments are designed for mature brains — their signals on a fresh brain are dominated by maturity gates and look misleadingly bad. Not a good first impression for anyone watching over your shoulder, and more importantly, not a useful signal.
- **No ignoring the camera.** Face enrollment matters as much as voice. A brain with voice but no face has a weaker Identity Fusion substrate and will lean harder on voice alone — making it more fragile to voice illness, background noise, or multi-person rooms.

---

## Exit Criteria — Before Advancing to Stage 1

You have completed the awakening when ALL of these are true. Check the dashboard and `~/.jarvis/` artifacts directly:

| Criterion | Target | Where to verify |
|---|---|---|
| Face confidence on primary user | >= 0.60 sustained | Dashboard identity panel + `~/.jarvis/face_profiles.json` |
| Voice confidence on primary user | >= 0.50 sustained | Dashboard identity panel + `~/.jarvis/speakers.json` |
| Enrollment clips registered | >= 3 (face and voice each) | `~/.jarvis/face_profiles.json`, `~/.jarvis/speakers.json` |
| Personal preference memories | >= 5 with `provenance:user_claim` | Dashboard memory panel (filter by provenance) |
| Recorded correction | >= 1 friction event or `CALIBRATION_CORRECTION_DETECTED` | `~/.jarvis/friction_events.jsonl` or calibration correction log |
| Identity boundary audit | Zero scope violations on spot-check queries | Dashboard identity audit panel |

If any criterion fails, stay in Stage 0. Do not advance on "close enough." Restart the enrollment phase if face or voice confidence never hits the target — the enrollment conditions (lighting, angle, distance, microphone) matter more than repetition count.

---

## After Awakening — When Each Accelerator Becomes Safe

Only after ALL exit criteria are satisfied:

- **Stage 1: Identity & Enrollment** (see [COMPANION_TRAINING_PLAYBOOK.md](COMPANION_TRAINING_PLAYBOOK.md)) — deepens what Stage 0 started. Architecture self-knowledge probing, reflective questions, more preferences.
- **Stage 2 onward** — casual conversation, preference grounding, household mapping, routines, daily companion interaction.
- **Synthetic Training Booster** (see Stage 2 booster in the playbook) — once real voice profiles and real face profiles exist, synthetic perception exercise correctly accelerates perception specialists (speaker_repr, face_repr, emotion_depth, voice_intent) without contaminating identity. The distillation signals get `origin="synthetic"` and `fidelity=0.7`, and a truth-boundary guard in `speaker_id` prevents profile drift from synthetic audio. Engineering detail: [docs/archive/SYNTHETIC_PERCEPTION_EXERCISE.md](archive/SYNTHETIC_PERCEPTION_EXERCISE.md).
- **Autonomy level review** — at this point the autonomy gated auto-restore logic has enough lived evidence to evaluate safely. If the prior instance earned L2, and all safety gates pass, it can restore.
- **Self-improvement stage promotion to 1 (dry-run)** — a reasonable next step once Stage 2 conversations are generating friction events and the scanner has something meaningful to detect.
- **Self-improvement stage promotion to 2 (human-approval)** — only after Stage 5+ playbook progress, with trust calibration and PVL coverage re-earned.

The rule is straightforward: **each accelerator requires lived baseline first, synthetic amplification second, promotion third.** Never in reverse.

---

## For Scientists and Researchers

If you are evaluating JARVIS for the first time, read this section carefully.

JARVIS is **architecturally complete, runtime maturing**. The capability gate will block any claim beyond that framing — and it should. A fresh brain looks weak if you measure it the same way you measure a mature one, because most of its metrics are accumulation-gated (see [MATURITY_GATES_REFERENCE.md](MATURITY_GATES_REFERENCE.md)). This is by design. The system is supposed to earn its maturity back through lived evidence, not inherit it at boot.

The awakening protocol exists because the most common evaluation mistake is skipping it: turning on a fresh brain, immediately running the synthetic exercise or the validation pack, and then reporting "the system has zero memories, the policy NN has zero decisions, the world model is at level 0." All of that is *correct* and *expected* on a fresh brain. None of it will improve if you do not first give the brain something to anchor against.

A fair scientific evaluation of JARVIS starts with one hour of the awakening protocol, followed by several sessions of the companion training playbook, followed by measurement. Not the other way around.

If you want to measure restart-honesty (Pillar 10) specifically, you can do that immediately — a fresh brain should *look* fresh, and the `maturity_highwater.json` + PVL + Oracle Benchmark outputs are designed to reflect that honestly. That is the one measurement that does not require lived evidence. Everything else does.

---

## Cross-References

- [docs/COMPANION_TRAINING_PLAYBOOK.md](COMPANION_TRAINING_PLAYBOOK.md) — Stage 0 (detailed curriculum) + Stages 1-7 (full training progression)
- [docs/CONVERSATION_QUALITY_GUIDE.md](CONVERSATION_QUALITY_GUIDE.md) — post-awakening conversation patterns
- [docs/SyntheticSoul.md](SyntheticSoul.md) §6.1, §9.4 — theoretical basis for why awakening precedes acceleration
- [docs/MATURITY_GATES_REFERENCE.md](MATURITY_GATES_REFERENCE.md) — what is expected to be zero/locked on a fresh brain
- [docs/ARCHITECTURE_PILLARS.md](ARCHITECTURE_PILLARS.md) Pillar 10 (Restart Resilience and Continuity Truth), Pillar 9 (Observability and Proof)
- [docs/archive/SYNTHETIC_PERCEPTION_EXERCISE.md](archive/SYNTHETIC_PERCEPTION_EXERCISE.md) — engineering detail on the synthetic exercise truth boundary
- [AGENTS.md](../AGENTS.md) "Scope and Capability Framing" — canonical release framing and audit discipline

---

## Operator TL;DR

1. Quiet room. Pi camera + mic in front of you.
2. Enroll face and voice. Five clips.
3. Introduce household members if you want them in the graph (including pets — optional).
4. Share 3-5 stable personal facts.
5. Ask one grounded self-knowledge question. Correct any confabulation.
6. Short reflective exchange.
7. Verify all exit criteria on the dashboard.
8. **Only then** start Stage 1, the synthetic training booster, or anything else.

That is the first hour. Everything JARVIS becomes for you is built on this foundation.
