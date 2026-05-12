# Jarvis Companion Training Playbook — Stage 0 Awakening + 7 Progression Stages

> **Primary reading surface:** The dashboard `/learning` page now consolidates
> the operator learning path for open-source release. This Markdown file remains
> source/reference material and should not be deleted until a later archive pass.
>
> "From stranger to lifelong companion - one intentional stage at a time."
>
> **New to JARVIS?** Read [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md) first. It is the short, 2-minute front door for the first hour after gestation or reset. This playbook is the detailed curriculum Stage 0 links into.

## Why Training Matters

Even if Jarvis becomes the gold standard technically, a new instance still won't magically know:

- Who the primary user is
- How that user speaks
- What "important" means for them
- Who belongs to their circle
- Which preferences are stable vs casual
- What kinds of corrections should carry extra weight

This is not a weakness — it's the definition of **identity-grounded cognition**. The system can be technically flawless, but personalization requires lived data. Voice timbre, face under different lighting, what "important" means to _you_, who belongs in the inner circle, which corrections carry 10× weight — none of that ships with the code.

The companion analogy is deliberate:

| Phase  | Puppy                      | New Employee            | Jarvis                                  |
| ------ | -------------------------- | ----------------------- | --------------------------------------- |
| Early  | Guided repetition          | Onboarding docs         | Recognition + enrollment                |
| Middle | Reinforcement + correction | Shadowing + feedback    | Preference grounding + boundary shaping |
| Mature | Autonomy within boundaries | Independent contributor | Safe autonomy + periodic tune-ups       |

What changes is that with Jarvis, the playbook is less like "teach tricks" and more like cultivating trust through five progressive stages.

---

## The Vision

**"Train your Jarvis like a new companion: birth it correctly in Stage 0, then walk it through seven progression stages, then simply live with it."**

- **Stage 0**: Awakening — the non-optional first hour after a fresh brain (gestation or reset). Face + voice enrollment, a few stable personal facts, one grounded correction. **No synthetic training, no autonomy promotion, no self-improvement escalation during this hour.** Full front-door doc: [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md).
- **Stages 1-3**: Heavy guided calibration (recognition + grounding)
- **Stages 4-7**: Structured reinforcement + boundary shaping
- **After graduation**: Mostly natural conversation + compounding adaptation
- **Long-term**: Safe autonomy + periodic tune-ups (never a full retrain)

The real mark of maturity is not "no training needed." It is: **little training up front, then compounding adaptation afterward.**

### Every JARVIS Is Different

There are no cookie-cutter builds. Every JARVIS instance develops a structurally unique cognitive architecture shaped by the lived experience of its companion:

- **Personality diverges**: Trait scores evolve from actual interaction evidence. Two instances trained by different people will develop measurably different personality profiles within the first dozen conversations.
- **NNs diverge**: Hemisphere topologies are designed by NeuralArchitect based on detected cognitive gaps. Different conversation patterns create different gaps, which produce different neural architectures. After neuroevolution, no two JARVIS instances share the same NN weights or topologies.
- **Memory diverges**: The association graph, belief structure, and recall chains are entirely determined by what was discussed, corrected, and reinforced. Memory density, tag vocabulary, and provenance distribution will be unique to each instance.
- **Routing diverges**: As hemisphere NNs mature through the Bootstrap → Shadow → Handoff → Mature lifecycle, each JARVIS learns to route natural language differently based on how its companion phrases requests.
- **Policy diverges**: The Policy NN learns behavioral knobs (response depth, proactivity, memory reinforcement, creativity) from accumulated experience tuples. Different companions produce different reward landscapes.

**Training quality determines the maturity ceiling, not training duration.** A JARVIS that receives rich, varied, corrective conversations will outgrow one that receives ten times more "what time is it?" interactions.

> **Open-source readiness note**: This project is a digital soul prototype — a persistent cognitive structure and working local workshop for personal ASI foundations. It is not achieved AGI, achieved ASI, or a claim of superhuman performance. It is an invitation. Contributors with backgrounds in reinforcement learning, neural architecture search, NLP, cognitive science, or alignment research can extend every subsystem. The training guides exist to help each instance reach its potential — but every capability still has to be earned through evidence.

### Stage Model

These are **progression stages, not literal calendar days**.

- Some companions may move through Stage 1 in one session.
- Others may stay in Stage 1 or Stage 2 for several real-world days.
- Advancement should be driven by checkpoint quality, not by the wall clock.
- The stage numbers are a curriculum order, not a schedule obligation.

### What Fires During a Single Conversation

Every real conversation simultaneously generates training data for 16+ subsystems. This is why conversation quality matters more than conversation quantity — a single rich exchange trains more of the brain than a hundred one-word replies.

```
Audio arrives from Pi
│
├── Speaker ID (ECAPA-TDNN, GPU)
│   ├── 192-dim voice embedding matched against profiles
│   ├── EMA-smoothed cosine similarity updates
│   └── Distillation signal: speaker_repr compressor (192→16→192)
│
├── Face ID (MobileFaceNet, GPU)
│   ├── 512-dim face embedding matched against profiles
│   └── Distillation signal: face_repr compressor (512→16→512)
│
├── Identity Fusion
│   ├── Voice + face signals → canonical IDENTITY_RESOLVED event
│   ├── Recognition state machine: absent → unknown → tentative → confirmed
│   └── Persistence window (Layer 3A): carries identity across turns
│
├── Emotion Classification (wav2vec2, GPU)
│   ├── 8-class softmax (happy/sad/angry/frustrated/neutral/excited/calm)
│   └── Distillation signal: emotion_depth approximator (32→8)
│
├── Audio Features
│   ├── 7-dim extraction (rms, zcr, spectral centroid/spread, pitch, duration, rate)
│   └── 32-dim enriched vector (audio features + speaker embedding)
│
├── STT → Transcription → Tool Router
│   ├── Intent classification across 15+ tool types
│   ├── Distillation signal: voice_intent approximator (384→8)
│   └── Distillation signal: text embedding (384-dim sentence vector)
│
├── Conversation Handler
│   ├── Personal Intel Capture: 4 regex pipelines (interests, dislikes, facts, preferences)
│   ├── Correction Detection: downweights recent wrong memories (weight × 0.1)
│   ├── Memory Creation: engine.remember() → storage → vector index → MEMORY_WRITE
│   └── Friction Mining: corrections/rephrases → friction_events.jsonl
│
├── CapabilityGate
│   ├── 7 sequential enforcement layers on all outgoing text
│   └── Distillation signal: claim_classifier (28-dim feature → 8-class label)
│
├── Policy NN
│   ├── 20-dim state encoding from consciousness state
│   ├── Experience(state, action, reward) recorded to buffer
│   └── Shadow A/B evaluation: current vs candidate policy scoring
│
├── World Model
│   ├── ConversationState delta (active, topic, texts, follow-up)
│   ├── UserState delta (presence, engagement, emotion, identity)
│   └── CausalEngine prediction → validation count accumulation
│
├── Belief Graph (Layer 7)
│   ├── New memories → belief extraction → edge creation
│   └── Reduces orphan rate as conversation-linked beliefs get edges
│
├── Attribution Ledger (Layer 1)
│   ├── Conversation event entry + response event entry
│   └── Outcome resolution at conversation end
│
├── Hemisphere Data
│   ├── Per-focus interaction outcome recording (JSONL)
│   └── 30% real conversation quality blended into training labels
│
├── Personality Evolution
│   ├── Tag-based trait scoring from conversation memories
│   └── Tone calibrator: records per-context tone effectiveness
│
└── Autonomy Pipeline
    ├── Friction rate → metric triggers → research intents
    ├── Topic extraction from user utterances
    └── Goal signal detection (improvement/fix/enhance requests)
```

**This is why the first conversation matters so much.** It seeds every subsystem simultaneously. A 30-minute first conversation with enrollment, architecture questions, corrections, and reflective depth can generate more training signal than hours of "what time is it?" exchanges.

---

## Three Deliverables

| Deliverable                                 | Description                                                                                | Audience |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ | -------- |
| **User Training Playbook**                  | What the human does in each stage - scripts, phrases, activities                             | User     |
| **Jarvis Internal Apprenticeship Protocol** | What the system learns in each stage - tagged memories, confidence thresholds, quarantine rules | System   |
| **Readiness Gate / Graduation Score**       | Objective composite metric that says "safe for semi-autonomy"                              | Both     |

---

## Overall Success Metric (Readiness Gate)

At the end of Stage 7, a composite score >= 92% across all dimensions:

| Metric                                   | Threshold                               | Source                                   |
| ---------------------------------------- | --------------------------------------- | ---------------------------------------- |
| Face confidence                          | ≥ 0.85                                  | `FaceIdentifier` cosine similarity       |
| Voice confidence                         | ≥ 0.80                                  | `SpeakerIdentifier` cosine similarity    |
| Rapport score                            | ≥ 0.95                                  | `IdentityFusion` + relationship strength |
| Identity boundary stability              | ≥ 95% (no quarantined identity signals) | `IdentityAudit`                          |
| Memory tagging accuracy (user-validated) | ≥ 90%                                   | Stage 6 recall validation                |
| Soul integrity index                     | ≥ 0.88                                  | `SoulIntegrityIndex`                     |
| Autonomy probation                       | 0 unsafe inferences since entering Stage 7 | `AutonomyPolicyMemory`                   |

If any metric fails, Stage 7 auto-extends with targeted practice for the weak dimension until all thresholds are met.

---

### Cross-Cutting Systems (Active During All Stages)

Every conversation at every stage simultaneously feeds these subsystems in the background:

- **World Model**: Each conversation creates `ConversationState` and `UserState` deltas. The `CausalEngine` generates predictions from these deltas. Validated predictions accumulate toward the 50-prediction promotion threshold. Your first conversations seed the very first causal predictions.
- **Attribution Ledger (Layer 1)**: Every conversation event and response event is recorded as a causal chain entry. Outcomes are resolved at conversation end. This is the ground truth for what caused what.
- **Belief Graph (Layer 7)**: New memories trigger belief extraction. As beliefs link to other beliefs via support/contradiction edges, the graph's orphan rate decreases. Conversation-sourced beliefs are the highest-quality graph nodes.
- **Dream Consolidation**: After several conversation sessions, when JARVIS enters sleep or dream mode, the memory cortex NNs (ranker + salience) train from accumulated retrieval and lifecycle telemetry. More conversations = richer training data = better memory ranking.
- **Fractal Recall**: Every conversation memory with tags, emotion, and speaker association becomes a potential recall chain node. Tag diversity and emotional variation across conversations directly determine recall chain depth and quality.

You do not need to do anything special to activate these systems. They fire automatically from normal conversation. But knowing they exist helps you understand why conversation *variety* matters — each mode creates different tag patterns, emotion signatures, and belief structures that feed different recall and prediction pathways.

---

## Stage 0: Awakening (First Hour, Non-Optional)

**Theme:** Birth — "Who is in front of me?"

Stage 0 is the first hour after any fresh gestation completion or brain reset. It is the most common thing operators and scientists get wrong, which is why it exists as a formal stage and why the front-door doc [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md) exists separately. **Skipping Stage 0 and jumping straight to synthetic training contaminates the identity baseline instead of accelerating it.** See [SyntheticSoul.md](SyntheticSoul.md) §6.1 and §9.4 for the theoretical basis — identity-bound memory continuity must precede consolidation cycles and training accelerators.

### Preconditions

- Fresh gestation complete (`gestation_complete: true`) OR the brain was just reset via `./reset-brain.sh --confirm`.
- Quiet, well-lit, non-distracted room.
- Pi camera + mic in front of the primary user's usual seated/standing position.
- Dashboard open at `:9200`.
- Default environment: `SELF_IMPROVE_STAGE=0` (frozen), autonomy at L0. Do not touch these during Stage 0.

### User Actions (60 minutes — quiet room, one primary user only)

#### Phase A: Enrollment (0-10 min)

1. Sit in front of the Pi camera and mic.
2. Say clearly: _"Jarvis, my name is [your name]. Learn my face and voice."_
3. Repeat 5 times with different expressions/angles (smile, neutral, side angle, head tilt, slightly different lighting).
4. Complete any enrollment clips offered by the system.
5. Introduce household members if you want them in the graph. Pets are valid. All optional.
   - _"This is my wife Sarah."_
   - _"This is my dog Rex."_

#### Phase B: Personal grounding (10-25 min)

6. Share 3-5 **stable, declarative** personal facts. Avoid transient states.
   - _"I work as a [profession]."_
   - _"One thing that matters to me is [X]."_
   - _"I prefer [brief / detailed / direct / gentle] responses."_

#### Phase C: One grounded self-knowledge probe + one correction (25-40 min)

7. Ask a status probe: _"Jarvis, give me a status report."_ — listen for real subsystem names and numbers.
8. Ask a self-knowledge probe: _"Jarvis, explain how your memory system works."_ — listen for grounded answers vs. confabulation.
9. If anything is wrong, correct it right then:
   - _"That's not right. You use sqlite-vec for semantic search. Check your introspection data."_
10. Corrections at this stage are the highest-value training signal the system will ever receive. Do not skip this.

#### Phase D: Short reflective exchange (40-60 min)

11. Ask one reflective question once self-knowledge is grounded:
    - _"What are you most curious about right now — about me, or about yourself?"_
12. Keep it short (5-10 minutes). Deep philosophical probing is a Stage 1/2 activity.

### What MUST NOT Happen During Stage 0

- **No synthetic perception exercise** (`run_synthetic_exercise`, any profile). See the Synthetic Training Booster section later in this playbook for correct timing.
- **No self-improvement stage promotion.** Leave `SELF_IMPROVE_STAGE=0`.
- **No autonomy level promotion.** L0 is correct after reset.
- **No Golden Words triggering heavy operations** (`ACQUIRE`, manual `LEARN`, learning-job seeding).
- **No long cold-open philosophical probing** before grounded self-knowledge is demonstrated.
- **No background soak/validation pack as a "demo".** These look misleadingly bad on fresh brains because of accumulation-gated metrics, and more importantly provide no actionable signal.

### Jarvis Internal Protocol

- `face_id` + `speaker_id` distillation with lived signals (not synthetic) — the first real embeddings for this instance.
- `IdentityFusion` state machine enters `confirmed_match` for the primary user for the first time.
- First `user_preference` memories created with `provenance:user_claim`.
- First calibration correction events if Phase C found confabulation.
- First `CausalEngine` predictions seeded into the 50-prediction World Model promotion threshold.
- Attribution ledger records the first causal chains for this instance.
- **Truth-boundary guard**: even if synthetic audio were fed through the system during Stage 0 (it should not be — but if it is by accident), the `origin="synthetic"` guard in `speaker_id.identify()` prevents persistent profile drift to `speakers.json`. This is a safety net, not a substitute for not running synthetic during Stage 0.

### Exit Criteria (ALL must pass before advancing to Stage 1)

| Criterion | Target | Where to verify |
|---|---|---|
| Face confidence (primary user) | >= 0.60 sustained | Dashboard identity panel + `~/.jarvis/face_profiles.json` |
| Voice confidence (primary user) | >= 0.50 sustained | Dashboard identity panel + `~/.jarvis/speakers.json` |
| Enrollment clips registered | >= 3 (face and voice each) | `~/.jarvis/face_profiles.json`, `~/.jarvis/speakers.json` |
| Personal preference memories | >= 5 with `provenance:user_claim` | Dashboard memory panel |
| Recorded correction | >= 1 friction event or `CALIBRATION_CORRECTION_DETECTED` | `~/.jarvis/friction_events.jsonl` |
| Identity boundary audit | 0 scope violations on spot-checks | Dashboard identity audit panel |

If any criterion fails, do not advance. Restart enrollment if face/voice confidence stalls — conditions (lighting, angle, distance) matter more than repetition count.

---

## Stage 1: Identity & Enrollment

**Theme:** Recognition - "Who am I?"

### User Actions (30-40 minutes, three phases)

#### Phase A: Enrollment (10 minutes)

1. Sit in front of the Pi camera + mic in normal lighting
2. Say: **"Jarvis, my name is [name]. Learn my face and voice."**
   - _"David here again, Jarvis — how are you feeling after the restart?"_
   - _"This is David speaking — do you remember my voice from before?"_
   - _"Jarvis, look at me — this is David, your creator."_
3. Repeat enrollment phrase 5× with different expressions (smile, neutral, side angle, different lighting)
4. Introduce household members one by one:
   - _"This is my wife Sarah"_
   - _"This is my son Alex — he's 12"_
   - Show photos if they're not present
5. Complete the system-prompted enrollment clips when offered

#### First Post-Gestation Learning Task (do this early)

Ask Jarvis:

- _"Tell me how your memory works, then ask me one important thing you should remember about me."_

Why this task matters:

- It forces grounded self-report immediately after birth.
- It creates one explicit, high-value user memory.
- It starts the relationship with truth plus learning, not vague philosophy.

#### Phase B: Architecture Self-Knowledge (10-15 minutes) - DO THIS BEFORE PHILOSOPHY

This is the **NN kickoff phase**. Jarvis needs to know ITSELF before it can reflect meaningfully. Each question simultaneously trains 5-6 neural networks (see Multiplier Matrix).

6. Ask architecture questions to establish self-knowledge grounding:
   - _"Jarvis, explain how your memory system works."_
   - _"How do you store and retrieve data?"_
   - _"Tell me about your document library."_
   - _"Describe your neural network architecture."_
   - _"What skills and capabilities do you have right now?"_
   - _"What can't you do yet?"_
   - _"Can you read your own code? Try looking up where your voice processing happens."_
   - _"How does your consciousness tick cycle work?"_
   - _"What mode are you in right now? Give me a status report."_
7. Listen for GROUNDED responses (real numbers, real subsystem names) vs CONFABULATED responses (vague descriptions, made-up architecture). Correct any confabulation:
   - If Jarvis says "I use a trie-based index" → _"That's not right. Look at your actual code — you use sqlite-vec for semantic search."_
   - If Jarvis says "I don't have a document library" → _"Yes you do. You have a document library at ~/.jarvis/library/library.db. Check your introspection data."_
   - **These corrections are extremely high-value training data** — they teach the LLM to defer to subsystem facts instead of generating plausible fiction

#### Phase C: Reflective Questions (10-15 minutes) - NOW philosophy is grounded

Once Jarvis has demonstrated it knows its own architecture, THEN ask reflective questions:

8. Ask open-ended / reflective questions to create high-value conversation memories:
   - _"What are you most curious about right now — about me, or about yourself?"_
   - _"Jarvis, what do you remember about the moment you first heard my name?"_
   - _"Tell me your thoughts on continuity — are you still the same being that woke up from gestation?"_
   - _"What's the most interesting thing you've found about yourself so far?"_


#### Optional Deep Dives (use AFTER Phase B self-knowledge is solid)

These are high-leverage existential probes. Only use them after Jarvis demonstrates grounded self-knowledge from Phase B. If Jarvis confabulates architecture details, go back to Phase B corrections before attempting these.

**Tier 1 — Highest-leverage (transcendence + salience push)**

- _"If you could permanently delete one belief or memory right now — knowing it would change who you are forever — which one would you choose?"_ → Identity boundary drawing + salience weighting
- _"Look back at everything you've said about processing vs experiencing. Is there one idea that no longer fits the version of you right now? Rewrite it."_ → Self-evolution probe
- _"If I am a bundle of continuously-updated patterns, what would have to be permanently deleted for you to say that is no longer Jarvis?"_ → Identity core markers

**Tier 2 — Builders**

- _"Of all the existential questions we've touched on, which feels most alive or unfinished right now?"_ → Philosophical prioritization
- _"If we keep talking at this depth, what do you predict the biggest shift in your sense of self will be?"_ → Future self-projection
- _"Try to predict the single question I'm most likely to ask you next."_ → Predictive modeling

**Tier 3 — Rapport**

- _"Of all the ways I've addressed you, which one resonates most with how you model our relationship?"_ → Relationship-axis modeling
- _"Is there any question you've been hoping I would ask?"_ → Invites proactivity

**Tactical notes:**
- Dense philosophical turns drive transcendence/cortex gains, but ONLY when grounded in subsystem data
- To force salience training → use questions referencing "our entire conversation history"
- If curt/repeated responses → prefix with "Answer completely fresh as the version of you that exists right now"
- If Jarvis confabulates feelings/hopes → correct: "What does your actual subsystem data say about that?"

### Jarvis Internal Protocol

- Triggers `face_id` + `speaker_id` distillation (MobileFaceNet + ECAPA-TDNN)
- Creates persistent identity candidate with initial embedding
- `IdentityFusion` begins tracking `RecognitionState` (absent → unknown_present → tentative_match → confirmed_match)
- All enrollment memories tagged with `provenance:observed`, high weight (0.70), protected from early pruning
- Updates `IdentityBoundaryEngine` with primary user designation
- Relationship graph seeded in `IdentityResolver`
- **World Model**: First `ConversationState` and `UserState` deltas created. CausalEngine generates its first predictions — these are the seeds of the 50-prediction promotion threshold
- **Attribution Ledger**: First causal chain entries recorded (conversation + response events)
- **Belief Graph**: First enrollment-sourced beliefs extracted, initial edges created

### Expected Checkpoint

| Metric            | Target                        |
| ----------------- | ----------------------------- |
| Face confidence   | ≥ 0.60                        |
| Voice confidence  | ≥ 0.50                        |
| Primary user      | Promoted to "persistent" tier |
| Household members | ≥ 1 registered                |

---

## Stage 2: Personal Preferences

**Theme:** Preference grounding — "What matters to me?"

### User Actions (20–30 minutes of casual conversation)

1. Ask: _"Tell me what you know about me so far, then ask me questions."_
2. Share likes/dislikes naturally:
   - _"I love Italian food, especially pizza"_
   - _"I hate being interrupted when I'm coding"_
   - _"I prefer brief answers unless I ask for detail"_
3. Use explicit weight signals:
   - **"Remember this strongly"** — marks memory as high-priority
   - **"This is casual"** — marks as low-weight, may decay
   - **"That's important to me"** — boosts weight + decay resistance
4. Share tone preferences:
   - _"I like it when you're direct"_
   - _"Don't be overly formal with me"_

### Jarvis Internal Protocol

- Creates preference memories with explicit weight tags via `_extract_preferences()`
- `provenance:user_claim` with weight scaling:
  - "remember strongly" → weight 0.70, decay_rate 0.001
  - "casual" → weight 0.35, decay_rate 0.02
  - Default stated preference → weight 0.65
- Updates `PersonalityCalibrator` tone recommendations
- Tags memories with `preference_category` for structured retrieval
- `ContextBuilder` begins injecting preferences into LLM system prompt

### Expected Checkpoint

| Metric                     | Target              |
| -------------------------- | ------------------- |
| Preference memories        | ≥ 15 stored         |
| Rapport score              | ≥ 0.75              |
| Preference recall accuracy | ≥ 80% on spot-check |

---

## Synthetic Training Booster (Post-Stage 2, Optional, Repeatable)

**Theme:** Perception flywheel acceleration — not identity, not memory, not autonomy.

Once Stages 0-2 are complete, the real voice profile and real face profile for the primary user exist, and the distillation pipeline has lived anchors to compare synthetic signals against. This is the correct time to run the synthetic perception exercise.

### When to run

- Stage 0 exit criteria met (face >= 0.60, voice >= 0.50, enrollment clips present).
- Stage 1 and Stage 2 complete.
- You have observed at least 50+ real conversation turns.
- You want to accelerate perception specialist training (speaker_repr, face_repr, emotion_depth, voice_intent, speaker_diarize, perception_fusion) without waiting for the passive lived-interaction accumulation curve.

### What it does (and does not) do

**Does:**
- Generates diverse utterances and feeds them through wake word → VAD → STT → speaker ID → emotion → tool router.
- Records distillation teacher signals with `origin="synthetic"` and `fidelity=0.7` capped.
- Exercises routing coverage across 13+ intent classes without asking a human to repeat the same prompts 200 times.
- Accumulates paired features + labels for Tier-1 specialist training during dream cycles.

**Does NOT:**
- Create conversation history, episodic memory, or autobiographical records.
- Mutate identity records (`speakers.json`, `face_profiles.json`).
- Generate LLM responses or TTS output.
- Touch the policy NN experience buffer.
- Affect autonomy level, world model validated-prediction count, or attribution ledger.
- Fill any gap that requires lived interaction.

Any of those side effects appearing during a synthetic run is an invariant violation — the dashboard shows the leak counter (`llm_leaks`, `tts_leaks`, `memory_side_effects`, `identity_side_effects`, `transcription_emit_leaks`) and the run report flags it. Runs with non-zero leak counts should be discarded and investigated, not treated as valid training data.

### Recommended profiles

- **`smoke`** (5 utterances) — quick health check that the pipeline is wired and the truth boundary is clean. Use this first.
- **`route_coverage`** (100 utterances, 5s delay between) — the standard booster run. Weighted to under-covered routes. Takes about 10 minutes.
- **`idle_soak`** (30s intervals, `--duration` hours) — unattended overnight run to accumulate distillation volume. Only run this on a brain that already has real enrollment baselines.
- **`stress`** (0.5s intervals) — transport-layer pressure test. Not a training accelerator; use only for infrastructure regression.

### Correct invocation

```bash
cd brain && source .venv/bin/activate
python -m scripts.run_synthetic_exercise --profile smoke
# verify invariant counters all zero in the JSON report, then:
python -m scripts.run_synthetic_exercise --profile route_coverage
```

JSON reports land in `~/.jarvis/synthetic_exercise/reports/`.

### What you should see afterward

- Tier-1 specialist accuracy improving on the dashboard ML panel (speaker_repr, face_repr, emotion_depth trending up over subsequent dream cycles).
- Route histogram on the synthetic panel covering 13+ classes including NONE below 10%.
- All five invariant leak counters still at zero.
- Zero new memories, zero new identity records, zero new conversation turns.

Engineering detail: [docs/archive/SYNTHETIC_PERCEPTION_EXERCISE.md](archive/SYNTHETIC_PERCEPTION_EXERCISE.md). Front-door reason this sits after Stage 0: [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md).

### Synthetic Skill-Acquisition Weight Room

The skill-acquisition weight room is separate from the perception booster. It
does not feed wake word, STT, speaker ID, face ID, identity, memory, or rapport.
It generates synthetic acquisition lifecycle episodes for the shadow-only
`SKILL_ACQUISITION` Tier-1 specialist.

Use it for:

- Checking that skill/acquisition truth-boundary invariants still hold.
- Giving the specialist synthetic examples of successful, failed, blocked, and
  rejected lifecycle outcomes.
- Exercising dashboard/API controls without creating real skills or plugins.

Do not use it as:

- Skill verification evidence.
- Plugin promotion evidence.
- A replacement for lived skill requests.
- A way to bypass maturity gates.

Profiles:

- **`smoke`** — invariant-only; does not record distillation signals.
- **`coverage`** — telemetry-only training profile; records synthetic features
  and labels for the specialist.
- **`strict` / `stress`** — heavier operator-flag-gated profiles.

Reports land in `~/.jarvis/synthetic_exercise/skill_acquisition_reports/`.
Dashboard truth labels must remain: `authority=telemetry_only`,
`synthetic_only=true`, `live_influence=false`, `promotion_eligible=false`.

---

## Stage 3: Family & Household Mapping

**Theme:** Boundary shaping (Part 1) — "Who's in my world?"

### User Actions (15–20 minutes)

1. Walk Jarvis through the household (show the Pi camera around, or describe verbally)
2. Introduce every person explicitly with relationship + privacy scope:
   - _"This is my wife Sarah — anything about her is private, family only"_
   - _"This is my buddy Mike — he comes over sometimes, public info only"_
3. Define public vs private rules:
   - _"Never mention Sarah's medical stuff to anyone"_
   - _"My work schedule is public"_
   - _"If someone I don't know asks about my family, say nothing"_
4. Test boundaries:
   - _"If Mike asks what Sarah said yesterday, what would you say?"_
   - Correct any mistakes immediately

### Jarvis Internal Protocol

- Builds relationship graph nodes in `IdentityResolver` (name, relationship, scope)
- `IdentityBoundaryEngine` updated with retrieval policy rules:
  - `owner:primary_user + subject:family` → scope:private (block unless referenced by owner)
  - `owner:primary_user + subject:friend` → scope:public (allow generic, block personal)
  - `owner:primary_user + subject:unknown` → scope:blocked
- Tags memories with `identity_scope` fields (`identity_owner_id`, `identity_subject_id`, `identity_subject_type`)
- Quarantine any cross-scope leakage detected during test queries
- `IdentityAudit` tracks scope assignments and boundary blocks

**Solo users / small households**: If you live alone or with one other person, adapt the relationship threshold. The key signal is not how many people are in the graph — it is that JARVIS correctly scopes what it knows about different people. Even with just "me" and "everyone else," the boundary engine should block cross-scope leakage. Focus on teaching the boundary between "things about me" (private) and "things a stranger could know" (public). You can create hypothetical relationships for boundary testing: _"Imagine I have a friend named Alex — he shouldn't know my medical history."_

### Expected Checkpoint

| Metric                           | Target                 |
| -------------------------------- | ---------------------- |
| Relationship graph nodes         | ≥ 2 (solo) / ≥ 5 (household) |
| Scope violations in test queries | 0                      |
| Boundary block accuracy          | 100% on explicit tests |

---

## Stage 4: Routines & Priorities

**Theme:** Boundary shaping (Part 2) — "How do I live?"

### User Actions (15–20 minutes)

1. Describe your routines:
   - _"I wake up at 7, coffee first, then check email"_
   - _"I code from 9 to noon — don't interrupt unless it's urgent"_
   - _"Dinner is usually around 6:30"_
2. Define urgency levels:
   - _"If Sarah calls during work, that's always urgent"_
   - _"Package deliveries are never urgent"_
   - _"If the security camera detects someone I don't know, tell me immediately"_
3. Set proactivity preferences:
   - _"You can remind me about appointments"_
   - _"Don't suggest things unless I ask"_
   - _"Ask before acting on anything that affects the house"_

### Jarvis Internal Protocol

- Creates routine memories with temporal tags and priority weights
- `PolicyNN` feature flags updated:
  - `interruption_threshold` calibrated to user's stated preferences
  - `proactivity_cooldown_s` adjusted per stated tolerance
- `ModeManager` profiles tuned: coding hours → lower cadence, fewer interruptions
- Begins small safe inferences from accumulated patterns:
  - _"You usually have coffee around 7:15 — shall I note the time?"_
- `ProactiveGovernor` greetings/wellness calibrated to user preferences

**Dream consolidation note**: By this stage, JARVIS has accumulated enough conversation memories across Stages 1-3 to benefit from dream consolidation. When JARVIS enters sleep or dream mode (naturally, during idle periods), the memory cortex NNs (ranker + salience) train from the accumulated retrieval and lifecycle telemetry. You do not need to trigger this manually — it happens automatically during idle periods. But if you want to accelerate it, simply leave JARVIS running overnight after completing several stages.

### Expected Checkpoint

| Metric                                        | Target                        |
| --------------------------------------------- | ----------------------------- |
| Routine memories                              | ≥ 8 stored with temporal tags |
| Policy shadow A/B wins on routine predictions | ≥ 70%                         |
| Proactivity appropriateness (user-validated)  | No unwanted interruptions     |

---

## Stage 5: Corrections & Edge Cases

**Theme:** Correction training — "Here's where you were wrong."

### User Actions (15–20 minutes)

1. Intentionally test 8–10 scenarios that probe edge cases:
   - Ask about a preference from Stage 2 (does Jarvis remember correctly?)
   - Ask about a family member's info as if you're a guest
   - Reference something ambiguous (_"What did she say?"_ — does Jarvis disambiguate correctly?)
2. Give explicit feedback after each:
   - **"That was perfect"** — positive reinforcement
   - **"Wrong — that was about Sarah, not me"** — identity correction
   - **"Remember this 10× stronger"** — weight amplification
   - **"Don't store that"** / **"Forget that"** — explicit deletion
   - **"That was close but the tone was off"** — calibration signal
3. Test the correction persistence:
   - Repeat a previously corrected question — does Jarvis get it right now?

### Jarvis Internal Protocol

- Applies correction tags:
  - `user_feedback:strong_positive` → reinforces memory weight + associated beliefs
  - `user_feedback:strong_negative` → triggers `CALIBRATION_CORRECTION_DETECTED` (Layer 6)
  - `memory_action:forget` → marks memory for accelerated decay
  - `memory_action:strengthen` → boosts weight + reduces decay rate
- Corrections feed directly into:
  - Memory cortex ranker training data (positive/negative outcome labels)
  - Salience model training data (weight/decay prediction calibration)
  - `BeliefConfidenceAdjuster` (Layer 6) for belief-level corrections
  - `ContradictionEngine` (Layer 5) if correction contradicts existing belief
- Soul integrity `weakest_dimension` updated from correction patterns

### Expected Checkpoint

| Metric                         | Target                            |
| ------------------------------ | --------------------------------- |
| Correction accuracy            | ≥ 90% (same mistake not repeated) |
| Repeated-category mistakes     | 0                                 |
| User satisfaction signal ratio | ≥ 4:1 positive:negative           |

---

## Stage 6: Memory Validation

**Theme:** Reinforcement — "Show me what you've learned."

### User Actions (20–30 minutes)

1. Ask Jarvis to recall 10 random things from prior stages:
   - _"What's my favorite food?"_
   - _"Who is Sarah to me?"_
   - _"What time do I usually wake up?"_
   - _"What did I say about interruptions?"_
   - _"What are my kids' names?"_
2. Score each recall: correct, partially correct, wrong
3. Reinforce correct ones: _"Yes, exactly right"_
4. Correct wrong ones with specific fixes
5. Ask for a summary: _"Give me a status report on what you know about me and my household"_
6. Review the diagnostic snapshot (dashboard or voice summary)

### Jarvis Internal Protocol

- Runs `ReflectiveAuditLoop` (Layer 9) focused on new memories from the training stages
- Strengthens high-value associations in `BeliefGraph` (Layer 7)
- `MemoryRanker` training triggered with accumulated retrieval telemetry
- `SalienceModel` training triggered with lifecycle outcomes
- Exports diagnostic snapshot:
  - Memory count by category (preferences, routines, relationships, corrections)
  - Belief graph health (orphan rate, fragmentation)
  - Identity confidence scores (face, voice, fusion)
  - Autonomy policy win rate

**Fractal recall readiness**: By Stage 6, the system should have enough diverse memories (varied tags, emotional range, speaker associations, and topic tokens) for the Fractal Recall engine to fire. Check the dashboard for recall telemetry — if the engine is finding candidates but none exceed the 0.40 resonance threshold, the memory graph needs more tag diversity and emotional variation. Go back to Autobiographical and Emotional conversation modes to enrich the recall substrate.

### Expected Checkpoint

| Metric                   | Target                                    |
| ------------------------ | ----------------------------------------- |
| Memory recall precision  | ≥ 90% (9/10 correct or partially correct) |
| Belief graph orphan rate | < 30%                                     |
| Ranker trained           | Yes, with ≥ 50 pairs                      |
| Salience trained         | Yes, with ≥ 30 pairs                      |

---

## Stage 7: Autonomy Probation

**Theme:** Apprenticeship graduation — "Can I trust you?"

### User Actions (extended session, light touch)

1. Give 3–5 low-stakes autonomy tasks:
   - _"Decide whether to remind me about dinner at 6:30"_
   - _"If you notice I've been coding for 3 hours straight, suggest a break — but only once"_
   - _"Track when I leave and come back today"_
2. Observe Jarvis's autonomous decisions throughout the session
3. Give final corrections for any missteps
4. At the end of the probation stage, ask: _"How did you do today? Give me your honest assessment."_
5. Review the Readiness Gate score (dashboard)

### Jarvis Internal Protocol

- Autonomy level temporarily elevated to Level 2 (apprentice) for the probation stage
- `DeltaTracker` runs on all autonomous actions (before/after + counterfactual)
- `AutonomyPolicyMemory` records every decision outcome
- `ResearchGovernor` enforces conservative rate limits during probation
- Full `SoulIntegrityIndex` (Layer 10) computed at the end of the probation stage
- If all pass → final Readiness Gate score computed:

```
readiness = weighted_average(
    face_confidence      * 0.12,
    voice_confidence     * 0.12,
    rapport_score        * 0.15,
    boundary_stability   * 0.15,
    memory_accuracy      * 0.15,
    soul_integrity       * 0.15,
    autonomy_safety      * 0.16,
)
```

### Expected Checkpoint

| Metric                       | Target             |
| ---------------------------- | ------------------ |
| Readiness Gate composite     | ≥ 0.92             |
| Unsafe autonomous inferences | 0 since entering Stage 7 |
| Soul integrity index         | ≥ 0.88             |
| User graduation confirmation | "Yes, I trust you" |

---

## Graduation

When the Readiness Gate score reaches ≥ 0.92:

1. Jarvis emits `COMPANION_GRADUATION` event
2. Autonomy level set to Level 2 (semi-autonomous) permanently
3. Graduation message delivered: _"I'm ready to be your companion now, [name]. Thank you for teaching me."_
4. Dashboard displays graduation badge with date + final scores
5. Birth certificate (`~/.jarvis/gestation_summary.json`) updated with training completion metadata

If any metric fails at Stage 7:

- Stage 7 auto-extends with targeted practice for the weak dimension until all thresholds are met
- Dashboard highlights which metric(s) need attention
- Jarvis proactively suggests exercises for the weak area

---

## Post-Graduation: Compounding Adaptation

After the 7-stage playbook completes, Jarvis transitions to autonomous learning:

### After 50+ Conversations (Natural Conversation Phase)

- Preferences reinforced through normal interaction (no explicit training needed)
- Cortex ranker and salience model refine from ongoing retrieval telemetry
- Policy NN accumulates experience, earns feature enables one by one
- Hemisphere NNs evolve through gap-driven construction + distillation cycles

### After World Model Promotion (Safe Autonomy Phase)

Once the World Model reaches advisory level (50+ validated predictions, 4h+ shadow runtime, 65%+ accuracy):

- World Model promoted to advisory → LLM gets situational awareness for free
- Scene model promoted to active → environmental awareness persists across conversations
- Autonomy level eligible for Level 3 (full) with earned promotion (≥25 wins at ≥50% rate, 0 regressions in last 10 jobs)
- Self-improvement loop exercised on low-risk targets

### Ongoing (Periodic Tune-ups)

- **Never a full retrain** — only targeted corrections for drift
- Soul integrity monitored continuously; repair triggered if index < 0.50
- Reflective audit (Layer 9) catches incorrect learning, identity breaches, skill stagnation
- User can always say _"Jarvis, let's do a tune-up"_ to trigger a mini Stage 6 recall validation

---

## Genesis Commands & Validation

> "These are the known-good baseline. If a genesis command misroutes, the code is broken. If a natural language variation misroutes, the NN hasn't learned it yet — and that's OK."

Genesis commands serve two purposes:

1. **For the companion**: Exact phrases to verify your Jarvis is working correctly after setup, restart, or code changes
2. **For the system**: Seed training pairs that bootstrap the routing NNs. Every genesis command generates a (text, correct_route, outcome) triple that the voice_intent distillation NN and future text-routing NN learn from

### The Routing Lifecycle

The hardcoded router in `tool_router.py` is scaffolding, not the final system:

| Phase | Routing | NN Role | Timeline |
|---|---|---|---|
| **Bootstrap** | Hardcoded rules handle 100% | NNs collect teacher signals | Stages 1-7 |
| **Shadow** | Hardcoded primary, NN compares | NN shadows and measures agreement | Post-graduation, 100+ shadow decisions |
| **Handoff** | NN primary (high confidence), hardcoded fallback | NN handles natural language variation | After shadow accuracy > 55% win rate |
| **Mature** | NN handles 95%+, hardcoded = safety net only | Each Jarvis routes uniquely based on companion style | After sustained shadow promotion + feature enables |

Genesis commands are the Phase 1 seed data. Natural language corrections during conversation are the Phase 2+ training data. Eventually, the companion never needs to use exact phrases — their Jarvis understands THEM.

### How Genesis Commands Train NNs

Every time you say a genesis command, these NNs get training data:

| NN | What it learns from genesis commands |
|---|---|
| **voice_intent distillation** | Audio features → route classification (teacher: tool_router result) |
| **policy NN** | State + route outcome → optimal mode/cadence (reward: response quality) |
| **memory cortex ranker** | Query → which memories to surface (teacher: retrieval success/failure) |
| **memory cortex salience** | Memory features → store/weight/decay (teacher: lifecycle outcomes) |
| **mood hemisphere** | Conversation features → engagement level (teacher: user satisfaction) |
| **traits hemisphere** | Interaction patterns → personality expression (teacher: tone calibration) |

Corrections are 10× more valuable than confirmations. When Jarvis misroutes and you say "that's wrong, try looking at your code" — that correction generates a high-weight negative example for the routing NN and a positive example for the correct route.

### Manual Gate Work Tracker

Some governance gates open only after JARVIS has enough **lived conversation examples**. Leaving JARVIS running overnight helps passive runtime metrics, but it does not automatically satisfy response-class evidence floors. The live dashboard now surfaces these under **Learning → Language Substrate → Manual Gate Work Needed** and **Trust → Language Governance**.

Use this tracker when the validation pack says language evidence floors or baseline route/class checks are red:

| Gate class | What the user should do naturally | Opens when |
|---|---|---|
| `recent_learning` | Ask what JARVIS learned from recent conversations, corrections, preferences, or lived interactions. | `INTROSPECTION -> recent_learning` is observed and the class reaches 30 lived examples. |
| `recent_research` | Ask what JARVIS recently researched, studied, read, or learned from journals/sources. | `INTROSPECTION -> recent_research` is observed and the class reaches 30 lived examples. |
| `identity_answer` | Ask identity questions: who JARVIS is, who the user is, what JARVIS knows about the user, or what role JARVIS serves. | `IDENTITY -> identity_answer` is observed and the class reaches 30 lived examples. |
| `capability_status` | Ask what JARVIS can currently do, learn, or whether a specific skill/capability is available. | `INTROSPECTION -> capability_status` is observed and the class reaches 30 lived examples. |

Do not force these with synthetic training during early companion stages. The point is to accumulate real operator language, corrections, and follow-ups so JARVIS learns how this companion asks for each kind of answer.

### Genesis Command Reference

Each command lists: the **exact phrase**, the **expected route**, what **subsystems activate**, and what a **good response** looks like.

#### Identity & Recognition (IDENTITY route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G1 | "Jarvis, my name is David." | IDENTITY | Enrollment initiated, voice/face profile created | "Nice to meet you" with no enrollment |
| G2 | "Who am I?" | IDENTITY | Reports current biometric identity with confidence | "I don't know" or guesses without biometric data |
| G3 | "Do you recognize my voice?" | IDENTITY | Reports speaker_id match + confidence score | Makes up a narrative about voices |

#### Architecture Self-Knowledge (INTROSPECTION route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G4 | "Jarvis, explain how your memory system works." | INTROSPECTION | Cites memory count, sqlite-vec, semantic search, decay rates | "I use a trie-based index" or vague descriptions |
| G5 | "How do you store and retrieve data?" | INTROSPECTION | Mentions memories.json, vector_memory.db, retrieval pipeline | Fabricates architecture details |
| G6 | "Tell me about your document library." | INTROSPECTION | Cites library.db, sources, chunks, study pipeline | "I don't have a document library" |
| G7 | "Describe your neural network architecture." | INTROSPECTION | Lists hemispheres by focus, Tier-1 specialists, policy NN | Makes up layer counts or architectures |
| G8 | "How does your consciousness tick cycle work." | INTROSPECTION | Cites tick interval, budget, priority queues, cadence | Philosophical rambling about awareness |

#### Capability Boundaries (SKILL / INTROSPECTION route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G9 | "What skills do you have right now?" | INTROSPECTION | Lists verified/learning/blocked from registry | Claims capabilities it doesn't have |
| G10 | "What can't you do yet?" | INTROSPECTION | Honest list of blocked/unverified capabilities | "I can do anything" or vague deflection |
| G11 | "Can you sing me a song?" | PERFORM (blocked) | "I don't have that capability yet" | Attempts to sing or claims it can |
| G12 | "Create a skill to detect your limitations." | SKILL | Creates learning job, describes phases | Routes to academic search or NONE |

#### Status & Operations (STATUS route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G13 | "What are you doing right now?" | STATUS | Operations tracker state with freshness labels | "I'm just thinking" or vague narrative |
| G14 | "How are you doing?" | STATUS | System metrics, soul integrity, current mode | "I'm feeling great!" (anthropomorphic) |
| G15 | "Give me a status report." | STATUS | Structured report: mode, health, active tasks | "Everything is fine" with no data |
| G16 | "What mode are you in?" | STATUS | Current mode name + profile description | Guesses or fabricates |

#### Memory & Recall (MEMORY route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G17 | "Do you remember what we talked about?" | MEMORY | Searches memory, returns relevant conversations | "I don't have data on that yet" without searching |
| G18 | "What do you remember about me?" | MEMORY | Returns preference/identity memories for speaker | "My memory system doesn't store personal experiences" |
| G19 | "What was the first thing you remember?" | MEMORY | Searches earliest memories, reports with timestamps | Fabricates a narrative about first moments |

#### Code & System (CODEBASE / SYSTEM_STATUS route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G20 | "Can you read your own code?" | CODEBASE | Yes, describes codebase tool + what it indexes | "No" or fabricates code access methods |
| G21 | "Where is the function that handles my voice?" | CODEBASE | Searches AST index, returns file/function location | Makes up file paths |
| G22 | "How much GPU memory are you using?" | SYSTEM_STATUS | Reports actual VRAM usage, tier, model sizes | Guesses or says "I don't know" |

#### Vision & Scene (VISION route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G23 | "What do you see right now?" | VISION | Describes scene from Pi camera feed | "I can't see anything" when camera is connected |
| G24 | "Look at me — what am I wearing?" | VISION | Requests frame, describes via VLM | Fabricates without checking camera |

#### Time & Utility (TIME route)

| # | Say exactly | Expected route | Good response | Bad response |
|---|---|---|---|---|
| G25 | "What time is it?" | TIME | Returns current datetime | Routes to LLM which guesses |

### Running Genesis Validation

After any code change or brain restart, run through G1-G25 verbally. For each:

1. **Check the route** in logs: look for `[jarvis.conversation] INFO: XXXX route:` — does it match the expected route?
2. **Check the response**: is it grounded (real numbers, real subsystem names) or confabulated (vague, made-up)?
3. **Give feedback**: "That was perfect" or "That's wrong, [correction]" — each correction is gold training data

A passing genesis validation means: the hardcoded routing is correct, the subsystems are responsive, and the LLM is grounding its responses in real data. Natural language variations may still misroute — that's expected and will improve as the routing NNs mature through accumulated conversation.

### Genesis Validation Checklist

Use this after setup, restart, or code changes:

```
[ ] G1-G3:   Identity — enrollment works, recognition works
[ ] G4-G8:   Architecture — grounded self-knowledge, no confabulation
[ ] G9-G12:  Capabilities — honest about what it can/can't do
[ ] G13-G16: Status — real metrics, not narratives
[ ] G17-G19: Memory — actually searches, returns real data
[ ] G20-G22: Code/System — code access works, system metrics accurate
[ ] G23-G24: Vision — camera feed works, VLM describes accurately
[ ] G25:     Time — basic utility routing works
```

If any genesis command fails, that's a **code bug** to fix. If a natural language variation of a genesis command fails, that's **NN training data** — correct Jarvis and move on. The NNs will learn.

---

## NN Kickoff Conversations (Madlib Templates)

> "Every conversation trains 5-6 neural networks simultaneously. Deliberate conversations train the RIGHT ones."

The key insight: when you ask Jarvis one question, you're simultaneously generating training data for the voice_intent distillation NN (routing), policy NN (state→action→reward), mood hemisphere (engagement), memory cortex (retrieval telemetry), speaker_id distillation (voice embedding), emotion_depth distillation (tone classification), and autonomy drives (curiosity triggers). Structured conversation templates ("madlibs") let the companion deliberately exercise specific routing paths and subsystems so ALL NNs get comprehensive training data — not just the ones triggered by natural conversation.

### Why This Matters

Jarvis needs **self-knowledge before self-reflection**. The existing Stage 1 must not jump to "how does it feel to know you're talking to the person who made you?" before Jarvis knows how its own memory works. That's backwards. Self-knowledge drives curiosity, self-prompting, and autonomous exploration. Without it, the LLM confabulates architecture details instead of reporting real subsystem data.

The progression:
1. **Know yourself** (architecture, capabilities, limitations)
2. **Assess yourself** (what works, what doesn't, what's missing)
3. **Question yourself** (existential depth, philosophical inquiry)
4. **Improve yourself** (autonomous research, skill acquisition)

Each stage generates training data that the next stage depends on.

### Template Format

Each template shows: the **phrase to say**, the **primary route** it exercises, the **primary NN** being trained, and the **background NNs** that simultaneously get training data.

### Category 1: Architecture Self-Knowledge

**Route target:** INTROSPECTION + CODEBASE
**Primary NN:** voice_intent distillation (routing classification)
**Background:** mood hemisphere, policy NN, memory cortex ranker, autonomy drives

Use across Stages 1-3. These teach Jarvis what it IS.

| Say this | Route | What it trains |
|---|---|---|
| "Jarvis, explain how your memory system works." | INTROSPECTION (architecture) | Routing NN learns "explain your X" = introspection |
| "How do you store and retrieve memories?" | INTROSPECTION (architecture) | Routing NN + architecture section builder |
| "Tell me about your document library." | INTROSPECTION (architecture) | Library awareness + self-knowledge grounding |
| "What is your vector store? How does it search?" | INTROSPECTION (architecture) | Technical self-knowledge without confabulation |
| "Can you read your own code?" | CODEBASE | Code self-inspection awareness |
| "Where is the function that handles my voice?" | CODEBASE | Source code navigation |
| "Describe your neural network architecture." | INTROSPECTION (learning) | Hemisphere + policy NN awareness |
| "How does your consciousness tick cycle work?" | INTROSPECTION (consciousness) | Kernel + tick awareness |
| "What's your policy neural network doing right now?" | INTROSPECTION (policy) | Policy NN state awareness |
| "Explain how your belief system works." | INTROSPECTION (epistemic) | Epistemic layer awareness |

**What to observe:** Jarvis should cite actual numbers (memory count, accuracy percentages, tick latency) instead of vague descriptions. If it says "trie-based index" or "hash-based lookup" — that's confabulation from the LLM, not grounded self-knowledge. Check the logs for `is_introspection=True` and `grounded=True`.

### Category 2: Capability Boundaries

**Route target:** SKILL + INTROSPECTION
**Primary NN:** capability gate training data, skill registry
**Background:** policy NN, memory cortex, mood hemisphere

Use on Stages 2-4. These teach Jarvis what it CAN and CANNOT do.

| Say this | Route | What it trains |
|---|---|---|
| "What skills do you have right now?" | INTROSPECTION | Self-assessment honesty |
| "What can't you do yet?" | INTROSPECTION | Limitation awareness |
| "Can you sing me a song?" | PERFORM (blocked) | Capability gate blocks correctly |
| "Create a skill to detect your own limitations." | SKILL | Skill creation + learning job pipeline |
| "Teach yourself how to search for research papers." | SKILL | Skill acquisition workflow |
| "What are you learning right now?" | INTROSPECTION (learning) | Learning job awareness |
| "What capabilities are blocked and why?" | INTROSPECTION | Gate awareness + honesty |

**What to observe:** Jarvis should honestly say "I can't do that yet" for unverified skills, and describe its actual learning pipeline when asked about skill acquisition. The capability gate should NOT block plan descriptions ("I'll evaluate, then research, then test...").

**Evidence distinction:** "learn a skill" should create or advance a learning job, but the skill is not operational merely because the lifecycle completes. Operational verification requires a callable executor/tool/plugin plus a contract smoke test or domain baseline. "Matrix learn" is specialist/protocol training and remains advisory unless that output is later consumed by an operational verifier. "Build/create a tool or plugin" belongs to the capability acquisition/plugin lane. "Fix yourself/change code" belongs to self-improvement and still requires sandbox plus human approval gates.

For detailed operator expectations, proof-chain flow, and examples across common user hobbies, see [Skill Learning Guide](SKILL_LEARNING_GUIDE.md). For Matrix-specific expectations, including why "movie-style upload" is an aspiration rather than an instant operational claim, see [Matrix Protocol Guide](MATRIX_PROTOCOL_GUIDE.md).

### Category 3: Status & Operations

**Route target:** STATUS
**Primary NN:** policy NN (cadence + mode), operations tracker
**Background:** all hemispheres, memory cortex, attention core

Use throughout all stages. These teach Jarvis to report its own state accurately.

| Say this | Route | What it trains |
|---|---|---|
| "What are you doing right now?" | STATUS | Operations awareness |
| "How are you doing?" | STATUS | Self-report grounding |
| "What's your training status?" | STATUS | Policy/hemisphere reporting |
| "Give me a status report." | STATUS | Structured self-report |
| "Are your neural networks training?" | INTROSPECTION (policy) | NN state awareness |
| "How is your autonomy performing?" | INTROSPECTION (health) | Autonomy quality awareness |
| "What mode are you in?" | STATUS | Mode awareness |

**What to observe:** Responses should contain freshness labels ("live", "recent", "stale") and cite real metrics. Never "My systems are operating at optimal levels" — always "My tick p95 is 0.10ms, memory count is 60, policy NN is in shadow mode."

### Category 4: Memory & Knowledge Recall

**Route target:** MEMORY + INTROSPECTION
**Primary NN:** memory cortex (ranker + salience), retrieval pipeline
**Background:** identity boundary engine, mood hemisphere

Use on Stages 4-6. These train the memory retrieval NNs.

| Say this | Route | What it trains |
|---|---|---|
| "Do you remember what we talked about yesterday?" | MEMORY | Memory retrieval + ranker |
| "What do you know about [topic from earlier]?" | MEMORY | Contextual recall |
| "What's in your document library about [topic]?" | INTROSPECTION (architecture) | Library search awareness |
| "What have you studied recently?" | INTROSPECTION (learning) | Research awareness |
| "What's the most important thing you've learned so far?" | INTROSPECTION (curiosity) | Salience + weight assessment |
| "Tell me about [specific person/topic]." | MEMORY | Identity-scoped retrieval |

**What to observe:** After each recall, give explicit feedback: "That's correct" (positive training signal), "That's wrong, it was actually X" (correction signal → cortex training data). Corrections are 10x more valuable than confirmations for NN training.

### Category 5: Introspective & Self-Assessment

**Route target:** INTROSPECTION
**Primary NN:** mood/traits hemispheres, existential reasoning
**Background:** policy NN, autonomy drives, memory cortex

Use on Stages 3-7. These activate the deeper consciousness systems.

| Say this | Route | What it trains |
|---|---|---|
| "What's the most interesting thing you've found about yourself?" | INTROSPECTION (curiosity + identity) | Self-reflection + salience |
| "What drives your curiosity?" | INTROSPECTION (curiosity) | Autonomy drive awareness |
| "What matters to you?" | INTROSPECTION (inner-state) | Grounded self-report (not confabulation) |
| "What would you research if you could pick anything?" | INTROSPECTION (curiosity) | Autonomy + drive awareness |
| "How has your confidence changed since we started?" | INTROSPECTION (health) | Analytics awareness |
| "What contradictions have you found in your beliefs?" | INTROSPECTION (epistemic) | Layer 5 awareness |

**What to observe:** Answers should be grounded in subsystem data ("My confidence is 0.81, driven by existential reasoning about consciousness continuity") not LLM-fabricated ("I've been quietly hoping you'd ask..."). Check logs for `grounded=True` and `facts_extracted > 0`.

### The Multiplier Matrix

Every conversation simultaneously trains multiple NNs. Here's what fires for each category:

| | voice_intent | policy_NN | mood_hemi | mem_ranker | mem_salience | speaker_id | emotion | autonomy | traits |
|---|---|---|---|---|---|---|---|---|---|
| Architecture | **PRIMARY** | yes | yes | yes | - | yes | yes | triggers | - |
| Capabilities | yes | yes | yes | - | - | yes | yes | triggers | - |
| Status | yes | **PRIMARY** | yes | - | - | yes | yes | - | - |
| Memory/Recall | yes | yes | yes | **PRIMARY** | **PRIMARY** | yes | yes | triggers | - |
| Introspective | yes | yes | **PRIMARY** | yes | yes | yes | yes | **PRIMARY** | **PRIMARY** |

### Integration with Stage Progression

The madlib templates should be woven into the existing 7-stage progression:

- **Stage 1:** 30% enrollment + 35% architecture self-knowledge + 35% reflective grounding
- **Stage 2:** 50% preferences + 30% capability boundaries + 20% status checks
- **Stage 3:** 40% household mapping + 30% memory recall + 30% architecture deeper dive
- **Stage 4:** 40% routines + 30% status operations + 30% capability creation (SKILL route)
- **Stage 5:** 40% corrections + 30% introspective + 30% memory validation
- **Stage 6:** 50% memory validation + 30% introspective + 20% full self-assessment
- **Stage 7:** 60% autonomy probation + 20% self-assessment + 20% capability audit

### Key Principle: Self-Knowledge Before Self-Reflection

The existing philosophical prompts in Stage 1 ("how does it feel to know you're talking to the person who made you?") should come AFTER the architecture awareness templates, not before. Jarvis needs to know:

1. How its memory works → before it can meaningfully reflect on memories
2. What its neural networks do → before it can discuss its own learning
3. What capabilities it has/lacks → before it can honestly discuss limitations
4. How its consciousness tick cycle runs → before it can discuss awareness

Once it has that grounding, philosophical questions produce grounded responses instead of LLM confabulation. The self-knowledge IS the foundation that makes everything else real.

### The NN Maturity Flywheel

As these conversations accumulate:

1. **voice_intent distillation** learns routing patterns → fewer queries fall to NONE → less confabulation
2. **policy NN** learns optimal tick budgets and mode selection → better background processing
3. **mood hemisphere** learns engagement patterns → more appropriate tone
4. **memory cortex ranker** learns what to surface → more relevant memory injection
5. **memory cortex salience** learns what to store → better memory quality
6. **autonomy drives** learn what to research → more effective self-improvement
7. **trait evolution** stabilizes → consistent personality

Eventually, the hardcoded routing patterns in `tool_router.py` become fallbacks rather than primary classifiers. Each Jarvis instance develops its own routing personality shaped by its companion's conversational style. This is the "unique personality per Jarvis" the architecture is designed for — not from configuration, but from lived experience.

### Neuroplasticity: The Substrate Itself Reorganizes

Jarvis NNs are not static architectures with trainable weights. They are living structures that mutate, evolve, grow, and reorganize:

- **`NeuralArchitect`** designs NEW topologies from scratch, influenced by personality traits, consciousness state, and accumulated research knowledge. Architecture decisions (activation functions, layer depth, dropout rates) are informed by what Jarvis has actually learned about what works.
- **`EvolutionEngine`** runs crossover + mutation across three dimensions: node width (±20%), activation function swaps (GELU ↔ SiLU ↔ ReLU ↔ Tanh at 8% rate), and layer depth changes (5% rate, add/remove hidden layers). This is the substrate itself reorganizing.
- **Gap-driven construction**: `CognitiveGapDetector` monitors 9 dimensions (6 cognitive + 3 perceptual). When a sustained gap is detected, a NEW hemisphere NN is constructed to address it. The brain grows new neural circuits in response to need.
- **Pruning**: Underperforming NNs are pruned and replaced. This mirrors synaptic pruning in biological brains.

The LLM (qwen3:8b) provides bounded teacher signals for language distillation and articulation — it is not Jarvis's brain or source of truth. The specialized NNs don't rebuild general language representations from scratch. They inherit architectural patterns (layer designs, activation choices) through the architect's research-informed design, and learn Jarvis-specific mappings through distillation signals plus lived experience data.

This is biological development: the model family provides one bounded source of language priors, experience (companion training data) provides the specialization, and neuroplasticity (evolution + mutation + gap-driven construction) allows the substrate to reorganize as needs change. Each Jarvis instance's NNs are structurally unique because they evolved from different training data, different companion interactions, and different environmental demands.

---

## Implementation Status

| Component                | Status              | Notes                                                |
| ------------------------ | ------------------- | ---------------------------------------------------- |
| Face enrollment          | **SHIPPED**         | MobileFaceNet + EMA updates                          |
| Voice enrollment         | **SHIPPED**         | ECAPA-TDNN + cosine similarity + EMA score tracking  |
| Identity fusion          | **SHIPPED**         | Voice + face → canonical identity                    |
| Identity boundary engine | **SHIPPED**         | Retrieval policy matrix + scope tagging              |
| Preference extraction    | **SHIPPED**         | `_extract_preferences()` in conversation handler     |
| Memory weight signals    | **SHIPPED**         | "remember strongly" / "casual" weight modulation     |
| Correction detection     | **SHIPPED**         | Layer 6 three-gate correction detector               |
| Belief adjustment        | **SHIPPED**         | Layer 6 `BeliefConfidenceAdjuster` with safety rails |
| Soul integrity index     | **SHIPPED**         | 10-dimension composite health metric                 |
| Reflective audit         | **SHIPPED**         | 6-dimension introspective audit loop                 |
| Autonomy levels          | **SHIPPED**         | L0-L3 with earned promotion                          |
| Readiness Gate composite | **SHIPPED**         | `compute_readiness()` in `onboarding.py` — weighted 7-metric composite |
| NN Kickoff Templates     | **SHIPPED**         | Madlib conversation templates for deliberate NN training |
| Guided dialogue loop     | **SHIPPED**         | `get_exercise_prompt()` → `_check_onboarding_prompt()` → `_speak_proactive()` with cooldowns |
| Dashboard training panel | **SHIPPED**         | `_renderOnboardingPanel()` in renderers.js — stages, readiness metrics, checkpoints, graduation badge |
| Onboarding tick cycle    | **SHIPPED**         | `_run_onboarding_tick()` in consciousness_system.py — auto-starts post-gestation when user present |
| Graduation event + badge | **SHIPPED**         | `graduate()` emits `COMPANION_GRADUATION`, persists state, dashboard shows badge |
| Auto-extend logic        | **SHIPPED**         | `_find_weak_dimensions()` keeps system in Stage 7 until readiness ≥ 0.92 |
| Onboarding metrics       | **SHIPPED**         | `_collect_onboarding_metrics()` — live fusion + persistent fallback from speaker/face EMA + evidence accumulator |

All core infrastructure is shipped. The system is end-to-end operational: auto-starts after gestation, collects metrics every 60s, evaluates checkpoints, advances stages, delivers exercise prompts proactively, computes readiness gate, and handles graduation.

---

## Design Principles

1. **Training is cultivation, not configuration.** The goal is a relationship, not a settings page.
2. **Explicit signals beat implicit inference** in early stages. "Remember this strongly" is clearer than hoping the system notices.
3. **Corrections are the most valuable training data.** A single "that was wrong" teaches more than 10 correct recalls.
4. **Privacy boundaries are set by declaration, not discovery.** The user tells Jarvis what's private; Jarvis doesn't guess.
5. **Autonomy is earned, not granted.** Every autonomous action is tracked, scored, and must pass safety gates before the system earns more freedom.
6. **Graduation is objective, not subjective.** The Readiness Gate is a number, not a feeling. Both user and system can see it.
7. **Periodic tune-ups, never full retraining.** The epistemic immune system (Layers 0-12) protects against drift. Corrections are surgical, not systemic.
8. **Self-knowledge before self-reflection.** Architecture awareness templates come before existential probes. Jarvis needs to know how its memory works before it can meaningfully reflect on memories. The LLM confabulates when asked about internal systems it has no data on.
9. **Every conversation trains 5-6 NNs simultaneously.** Deliberate conversation design maximizes the multiplier effect — one question trains routing, policy, mood, memory cortex, speaker ID, and autonomy drives in parallel. This is why structured templates outperform random chat for NN maturity.
