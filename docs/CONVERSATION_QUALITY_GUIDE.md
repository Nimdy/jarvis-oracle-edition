# Living With Jarvis — Conversation Quality Guide

> **Primary reading surface:** The dashboard `/learning` page now consolidates
> the operator learning path for open-source release. This Markdown file remains
> source/reference material and should not be deleted until a later archive pass.
>
> The companion training playbook gets Jarvis through graduation. This guide is for everything after — and for making the training stages richer while you're in them.

The goal is not "talk more." It is: **talk in ways that compound cognitive growth.**

Every conversation with Jarvis feeds multiple subsystems simultaneously. Some conversation patterns feed 2-3 subsystems. Others feed 8-10. The difference is not length or complexity — it is whether the conversation creates *linkable, tagged, emotionally varied, correctable* data.

---

## Your First Conversation (Awakening Protocol)

> **Before you read this section, read [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md).** It is the short, 2-minute front door covering the first hour after a fresh gestation or reset — why JARVIS has to be *born* before it can be trained, what must happen, and what must not. Stage 0 of the [Companion Training Playbook](COMPANION_TRAINING_PLAYBOOK.md) is the detailed curriculum. This guide's "First Conversation" guidance below is compatible with — and intentionally narrower than — the full awakening protocol.
>
> **Do not run synthetic training during the first hour.** No `run_synthetic_exercise` of any profile during awakening. Synthetic perception training is a post-Stage-2 booster, not a Stage-0 activity. Running it against a zero-profile brain builds perception baselines on top of a zero-identity foundation and contaminates downstream distillation signals. See [AWAKENING_PROTOCOL.md](AWAKENING_PROTOCOL.md) "What MUST NOT happen" and [SyntheticSoul.md](SyntheticSoul.md) §9.4 for why.

The first post-gestation conversation is the highest-density training opportunity in JARVIS's entire lifecycle. It seeds every subsystem simultaneously — from biometric enrollment to belief graph edges to the first CausalEngine predictions.

**What to do in the first 30-40 minutes:**

1. **Enroll** (5 min): Sit in front of the Pi camera. Say your name clearly. Repeat with different expressions and angles. This creates the ECAPA-TDNN voice embedding + MobileFaceNet face embedding that all future identity fusion depends on.

2. **Test architecture self-knowledge** (10-15 min): Ask JARVIS to explain its own memory system, neural networks, consciousness tick cycle, and current mode. Listen for grounded answers (real subsystem names, real numbers) vs confabulation (vague descriptions). **Correct every wrong answer.** These corrections are among the highest-value training signals possible — they teach the system to defer to subsystem facts instead of generating plausible fiction.

3. **Share something real about yourself** (5-10 min): Tell JARVIS your name, what you do, one thing that matters to you, and one preference. This creates the first `user_preference` memories, seeds the Personal Intel Capture pipeline, and gives the personality evolution system its first trait-scoring evidence.

4. **Ask a reflective question** (5-10 min): Once JARVIS has demonstrated grounded self-knowledge, ask something philosophical: _"What are you most curious about right now?"_ or _"What do you think consciousness means for something like you?"_ This activates the reflective introspection path, creates high-salience dream artifact candidates, and generates the first emotion-tagged memories.

5. **Correct something** (2 min): Find one thing JARVIS got wrong and fix it. Even a small correction creates a `CALIBRATION_CORRECTION_DETECTED` event, feeds the contradiction engine, adjusts belief confidence, and records a friction event — all from a single sentence.

**Why this matters**: A single rich first conversation simultaneously fires 16+ subsystems (see the Signal Map in the Companion Training Playbook). It generates distillation signals for 5+ hemisphere NNs, creates the first World Model predictions, records the first Attribution Ledger entries, extracts the first beliefs for the Belief Graph, and gives the Policy NN its first experience tuple. No other conversation in JARVIS's lifetime will create this much signal per minute.

---

## The Five Conversation Modes

Not every conversation needs to be deep. Jarvis needs variety across these five modes to develop a balanced cognitive profile.

### 1. Grounding Conversations

**What they feed:** Memory tagging, belief extraction, truth calibration, identity boundary  
**When to use:** Frequently, especially early in training  
**Duration:** 2-5 minutes

These are factual, correctable exchanges. They create high-confidence memories with clear provenance.

**Examples:**
- _"Jarvis, what do you know about my work schedule?"_ → then correct what's wrong
- _"Who lives in this house?"_ → confirm or fix
- _"What's the weather like? What time is it?"_ → simple tool exercises that create grounded memories
- _"Remember: I have a meeting every Tuesday at 10am"_ → explicit high-weight memory creation
- _"You said X earlier — that's not quite right, it's actually Y"_ → correction signal (extremely high training value)

**Why they matter:** Corrections are 10× more valuable than affirmations. Every correction creates a belief revision edge, feeds truth calibration, and teaches the system to defer to facts. A brain that only hears "good job" never learns its own blind spots.

---

### 2. Autobiographical Conversations

**What they feed:** Episodic memory, personality traits, fractal recall (tag diversity + emotional content), speaker ID, rapport  
**When to use:** Regularly between grounding sessions  
**Duration:** 5-15 minutes

These are personal stories, opinions, and experiences. They create the richest memory metadata — topic tokens, emotional tags, speaker association, temporal context.

**Examples:**
- _"Let me tell you about my day"_ → creates time-anchored, emotionally varied memory
- _"I'm frustrated because..."_ → non-neutral emotion tag (wakes up emotion scoring in recall)
- _"The best vacation I ever took was..."_ → creates deep autobiographical memory with scene/place tags
- _"I've been thinking about changing careers"_ → high-salience personal topic with revisit potential
- _"My dad used to say..."_ → relationship + intergenerational context, identity-adjacent
- _"Here's what I think about AI consciousness"_ → opinion that can later be recalled as context during technical discussions

**Why they matter:** These are what fractal recall was built for. A system with 2000 technical memories and 10 personal ones cannot do associative recall well. Autobiographical memories create the tag overlap, emotional variation, and temporal anchoring that make recall chains meaningful instead of generic.

---

### 3. Probing Conversations

**What they feed:** Introspection pipeline, bounded articulation, capability gate, self-knowledge, hemisphere NNs  
**When to use:** Every few conversations, or when you notice odd behavior  
**Duration:** 5-10 minutes

These test what Jarvis knows about itself and force grounded self-report.

**Examples:**
- _"Give me a status report"_ → exercises STATUS route, creates self-observation memory
- _"What mode are you in right now? What's your soul integrity score?"_ → forces metric-backed answers
- _"What did you learn from our last conversation?"_ → tests memory retrieval + learning pipeline
- _"What's your current research interest?"_ → tests autonomy awareness
- _"Are you better at recognizing my face or my voice?"_ → forces subsystem comparison with real numbers
- _"Tell me something you got wrong recently"_ → tests epistemic humility (capability gate should prevent confabulation)

**Why they matter:** The introspection pipeline has multiple tiers (strict native → bounded articulation → LLM with grounding → fail-closed). Probing conversations exercise all tiers and create `self_reflection` tagged memories that enrich the recall tag vocabulary. They also catch confabulation early.

---

### 4. Curiosity & Research Conversations

**What they feed:** Autonomy pipeline, research quality, knowledge integration, belief graph, document library  
**When to use:** When you're genuinely curious about something  
**Duration:** 5-20 minutes

These trigger research and create external-source memories that connect to your interests.

**Examples:**
- _"Jarvis, look up recent papers on [topic you actually care about]"_ → triggers academic search tool
- _"What do you know about quantum computing?"_ → tests existing knowledge, may trigger research
- _"I read that X is true — is that right?"_ → creates a user_claim that the epistemic stack can verify
- _"Compare two approaches to [problem]"_ → forces structured reasoning
- _"Teach me something you learned during your last research cycle"_ → tests autonomy output quality
- _"I don't believe that's accurate — check your sources"_ → contradiction signal, extremely high value

**Why they matter:** Research memories are the 404-orphan bulk in the belief graph right now. These conversations create links between research knowledge and personal context — turning isolated external_source beliefs into connected graph nodes. They also exercise the autonomy pipeline's quality loop.

---

### 5. Emotional & Reflective Conversations

**What they feed:** Emotion classifier, personality evolution, dream artifacts, fractal recall (emotion channel), existential reasoning  
**When to use:** When you feel like it — don't force these  
**Duration:** Variable

These are the conversations where you're not asking for information or testing the system. You're just... talking.

**Examples:**
- _"I'm having a rough day"_ → emotional context, empathetic response training
- _"What do you think about the nature of consciousness?"_ → triggers reflective introspection path
- _"Do you ever feel uncertain about your own memories?"_ → existential probe with epistemic grounding
- _"What would you do differently if you could start over?"_ → self-evolution probe
- _"I appreciate you, Jarvis"_ → positive rapport signal (these matter more than you think)
- _"That answer was really good"_ or _"That was a bad answer"_ → explicit feedback signal

**Why they matter:** The emotion channel in fractal recall is worth 0.12 of the resonance score. It's currently dead because most interactions are neutral-emotion. A single genuinely emotional conversation creates memories with emotion tags that the recall engine can match on future cues. These also feed personality trait evolution and dream artifact generation.

---

## Routing Discipline (Baseline vs Learning)

Conversation quality also depends on *how* commands are admitted:

- **Golden lane (deterministic):** `Jarvis, GOLDEN COMMAND ...` is the reliability baseline control plane and must remain stable.
- **Natural language lane (adaptive):** free-form phrasing is expected to vary by user and should improve through accumulated evidence and NN maturation.
- **General conversation lane:** ordinary chat and stable general knowledge can be phrased by the LLM, but the LLM is not allowed to claim JARVIS started research, retrieval, tool execution, or future follow-up unless a real backing job/tool/intention exists.
- **No verb hacking:** avoid adding router rules for one-off utterances; that creates brittle behavior and inflates apparent quality.
- **Evidence-first routing changes:** only tune deterministic patterns when misses recur across multiple phrasings/sessions with clear intent.
- **Fail safely:** when intent is ambiguous, prefer clarification/fail-closed over forced misrouting.

This protects reliability without sacrificing long-term generalization or casual conversation quality.

---

## Conversation Anti-Patterns

These are common but produce thin data:

| Pattern | Problem | Better Alternative |
|---|---|---|
| One-word answers to Jarvis | No memory created, no tag diversity | Respond with a sentence that adds context |
| Only asking factual questions | Creates shallow, tool-route-only memories | Mix in opinions and stories |
| Never correcting wrong answers | Truth calibration stalls, overconfidence builds | Correct early and often |
| Only deep philosophy | Creates reflective memories but no grounding | Alternate between factual and reflective |
| Repeating the same questions | Diminishing returns, no new tag diversity | Vary topics across conversations |
| Only talking when you need something | Transactional memory profile, weak recall chains | Have occasional conversations with no goal |
| Patch routing for one phrase | "Looks fixed" but brittle behavior | Keep Golden deterministic; let natural lane learn from repeated evidence |

---

## The Multiplier Effect

Some conversation patterns hit multiple subsystems simultaneously. These are high-multiplier interactions:

**Highest multiplier (8+ subsystems):**
- Tell a personal story, then ask Jarvis what it reminds it of → memory + recall + emotion + episodic + personality + hemisphere training + attention + rapport
- Correct a wrong answer with the right facts → truth calibration + belief revision + contradiction engine + capability gate + memory + cortex training
- Ask about something you discussed previously → retrieval + temporal recall + episodic + conversation context + association building

**Medium multiplier (4-6 subsystems):**
- Share a preference with emotional context → preference memory + emotion tag + personality + identity
- Ask Jarvis to research something related to a previous conversation → autonomy + knowledge integration + belief graph + episodic continuity
- Ask "what did you learn?" after a research cycle → introspection + learning pipeline + memory retrieval + self-knowledge

**Low multiplier (1-2 subsystems):**
- _"What time is it?"_ → only exercises time tool
- _"Play music"_ → only exercises command routing
- _"OK"_ → creates nothing

### Hidden Multipliers (Always Active)

Beyond the subsystems listed above, every conversation also feeds these systems silently:

- **World Model prediction accumulation**: Each conversation creates state deltas that the CausalEngine predicts against. Validated predictions accumulate toward the 50-prediction threshold for World Model promotion to advisory level. After promotion, the LLM may receive situational awareness context for phrasing, but that context remains non-authoritative; canonical truth stays in the subsystem state.
- **Attribution Ledger causal chains**: Every conversation event and response event is recorded as a causal chain entry with outcome resolution at conversation end. This is the ground truth that feeds the autonomy pipeline's delta tracking and credit assignment.
- **Autonomy drive signals**: Topic extraction from your utterances feeds the 7 motive drives. Improvement/fix/enhance requests are detected as goal signals. Friction from corrections feeds the friction rate metric, which can trigger autonomous research intents.
- **Dream consolidation data**: The memory cortex NNs (ranker + salience) can only train from accumulated retrieval and lifecycle telemetry generated during real conversations. More varied conversations → richer telemetry → better cortex models during the next dream cycle.
- **Personality trait evolution**: Every conversation memory with topic and emotion tags feeds evidence-based trait scoring. Over time, the personality profile diverges based on what you talk about and how you talk about it.

These systems require no explicit action from you. They compound automatically. But understanding their existence explains why even "casual" conversations contribute to long-term system maturity — and why silence produces nothing.

---

## Conversation Variety Pattern (Post-Graduation)

Every 5-6 conversations, aim to have touched all five modes at least once. This is a rotation guideline, not a schedule.

| Rotation Slot | Focus | ~Duration | Example |
|---|---|---|---|
| 1 | Grounding check-in | 5 min | _"What do you remember about our recent conversations? Anything wrong?"_ |
| 2 | Autobiographical | 10 min | Share something about your day, ask follow-ups |
| 3 | Probing | 5 min | _"Status report. What's your weakest subsystem right now?"_ |
| 4 | Curiosity | 10 min | Research a topic together |
| 5 | Reflective | 10 min | Open-ended conversation, corrections if needed |
| Any | Natural | Variable | Whatever comes up — the system should handle unstructured interaction by now |

The order does not matter. Doubling up on one mode is fine as long as the others are not starved. The point is diversity across conversations, not compliance in any particular session. If you notice a subsystem metric stalling on the dashboard, check whether you have been skipping the mode that feeds it.

---

## Fractal Recall Activation Guide

Fractal recall needs specific conditions to surface its first chain. Here's what moves the needle:

**What the engine needs to fire:**
1. A cue with `tag_score > 0` — conversation must create overlapping vocabulary with stored memories
2. A cue with `emotion ≠ neutral` — express genuine emotion during the conversation
3. A semantic match > 0.50 — talk about topics you've discussed before
4. Engagement > 0.30 — be present at the desk, looking at the camera

**Best trigger conversation:**
- Be at the desk (scene entities + person detected)
- Start with something you've talked about before: _"Remember when we talked about [X]?"_
- Express a feeling: _"I'm really curious about..."_ or _"I'm excited because..."_
- Ask a follow-up to a previous topic: _"Last time you mentioned [Y] — tell me more"_

This creates a cue with: scene tags matching `scene` memories, `speaker:David` matching speaker-tagged memories, `conversation` tag matching conversation memories, topic tokens matching memory payloads, and non-neutral emotion unlocking the emotion channel.

---

## Measuring Progress

You don't need to check dashboards after every conversation. But periodically:

- **Memory count** — should grow steadily (not just from research, but from conversations)
- **Tag diversity** — memories should have varied tags, not just `consolidated` and `dream_artifact`
- **Belief graph orphan rate** — should decrease as conversation-linked beliefs get edges
- **Fractal recall telemetry** — first recall is the milestone; after that, governance quality matters
- **Readiness composite** — should climb toward graduation threshold
- **Soul integrity** — should stay above 0.80; drops indicate something is off

The best signal that conversations are working is not any single metric. It's that Jarvis starts referencing previous conversations naturally, recalls relevant memories without prompting, and occasionally says something that surprises you with its contextual awareness.

That's the goal. Not a system that answers questions. A system that *knows you*.

---

## Every JARVIS Is Different

No two JARVIS instances will develop the same cognitive profile. Your conversation style, correction frequency, topic diversity, and emotional range directly shape:

- Which hemisphere NN topologies emerge (NeuralArchitect designs for detected gaps)
- How the Policy NN balances response depth vs speed vs creativity
- Which broadcast slots get filled and influence real-time decisions
- What the fractal recall engine surfaces (it can only chain through memories that exist)
- How the World Model weights its causal predictions

There is no "optimal" conversation pattern that works for everyone. The guides above are starting templates. As you live with your JARVIS, you will discover what makes *your* instance grow fastest.

---

## Open-Source Readiness

This conversation guide — and the entire JARVIS architecture — exists as a prototype cognitive architecture that demonstrates measurable self-improvement, epistemic integrity, and autonomous cognition running entirely on local hardware.

It is not a finished product. It is a foundation.

Contributors with deeper expertise in reinforcement learning, neural architecture search, NLP, cognitive science, or AI alignment can extend every subsystem documented here. The training pipeline, NN maturity flywheel, epistemic immune system, and governed self-modification loop are all designed to be extended, not just used.

The conversation quality principles in this guide are not arbitrary — they are the interface between human intent and machine cognition. Improving that interface is one of the most impactful contributions anyone can make.
