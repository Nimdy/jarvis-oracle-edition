# Companion Cognition — The Live Internal Read, Theory-of-Mind, and Read→Behavior Loop

> Status: **DESIGN — not built.** Companion to `SPARK_DESIGN.md` and `WEIGHT_ROOM_DESIGN.md`.
> Authored 2026-05-31 from a 3-agent read-only grounding (theory `a83cd…`, machinery `aa9ad…`,
> multi-user `af8bc…`), tied to file:line. No code until read + approved.

## 1. Thesis

The north star is Iron Man's JARVIS / Halo's Cortana: a companion that runs a **continuous internal
read** while it talks — *"are they engaged? bored? is the room tense? am I overexplaining? should I
pivot or give them space?"* — and lets that read **change how it behaves**, getting better at the read
**over time, learned from the companion.**

The grounding shows this is a **wiring problem, not a missing-machinery problem.** The substrate exists:
- a continuous kernel tick (`consciousness/kernel.py`, 0.1s; meta-thoughts every `META_THOUGHT_INTERVAL_S=8.0s`),
- a thought generator (`consciousness/meta_cognitive_thoughts.py`),
- an affect readout (`consciousness/affect_state.py`, shadow),
- a **real per-person relationship model** (`consciousness/soul.py` `Relationship`: familiarity, rapport, preferences, interactions per name),
- identity-boundary scoping (`identity/boundary_engine.py`), emotion classification + tone calibration (`consciousness/engine.py:~560` `calibrate_tone`).

But it is **behaviorally inert**: thoughts fire on *idle background cycles from static state* (not during
conversation, not from live context), they are *logged and influence nothing* in-exchange (only seed the
autonomy research queue via `autonomy/event_bridge.py`), there is **no theory-of-mind** (only a raw
emotion label), and **no thought→belief crystallization** (thoughts and beliefs are silos). The
machinery is "a write-only journal that influences nothing."

This design makes the internal read **(1) live during conversation, (2) a model of the other person held
as hypotheses, (3) wired to behavior through a gated ladder, (4) able to crystallize into beliefs, and
(5) learned from the companion** — all shadow-first, honest, and per-instance.

## 2. The unification (why this ties the whole project together)

Companion cognition is the **conversational loop that the spark and the affect layer were always for**:

```
   live conversation
        │
        ▼
   SITUATIONAL READ  ── theory-of-mind: what does this person feel/want/think? (hypothesis)
        │                 self-monitoring: am I clear / overexplaining / losing them?
        ├─────────────▶ AFFECT (affect_state): the read drives dopamine/serotonin/cortisol
        │                 (tense read → cortisol; resolved → dopamine) — already built, shadow
        ├─────────────▶ CURIOSITY/GROUNDING (the spark): "am I reading them right?" →
        │                 a grounding question, sometimes asked OUT LOUD ("you've gone quiet —
        │                 want me to drop this?") = external validation of the read
        ▼
   BEHAVIOR  ── tone / depth / pace / pivot / give-space / disengage / clarify
        │        (gated: shadow → advisory → active; affect-cadence coupling is part of this)
        ▼
   FEEDBACK  ── companion confirms/corrects ("I wasn't annoyed, I was thinking") →
                the read's model updates (learned social cognition) →
                recurring patterns CRYSTALLIZE into beliefs (self / relational / world)
```

The spark (`SPARK_DESIGN.md`) supplies the **validate-against-reality** half; the affect layer supplies
the **felt-state** half; this design supplies the **read + behavior + learning** that makes them a
companion instead of a journal.

## 3. Foundation: one JARVIS, one companion, household-aware (the architecture is already right)

**This is NOT a global / multi-tenant app.** It is **one JARVIS bonded to one primary companion** (the
main user), who **also recognizes, interacts with, and learns about the other people around that user**
(family, friends, visitors) — without ever becoming *their* companion or leaking the main user's private
world to them.

That is exactly what the code already is: one `SoulService` / `ConsciousnessEngine` / `IdentityFusion`,
one primary operator, **genuine per-person relationships** (`soul.py` `Relationship`: familiarity /
rapport / preferences per name) for everyone around them, and **identity-boundary scope-gating**
(`identity/boundary_engine.py`: owner:primary + subject:family/friend/unknown policies) so the main
user's private memories stay private. The theory scopes "multiple JARVIS brains" out by design; this
matches.

So there is **no multi-tenant / global re-architecture in scope** — the earlier Option-A-vs-B fork is
moot. Companion cognition simply extends the existing per-person relationship model: a **deep, privileged
bond with the primary companion** (the 7-stage apprenticeship, the calibrated self-model) plus **lighter
relationships and reads for the others around them.**

## 4. The five components

**(1) Live situational read.** A read fires **per conversation turn** from **live context** (what was
said, user emotion, presence/scene, rapport, pace, silence/latency, addressee) — not the current static
idle-tick context (memory count, uptime). Extend `meta_cognitive_thoughts` with a conversation-driven
trigger and a live context object; keep the 3s cooldown spirit but gate on conversational events, not the
8s idle timer. Output: a short-lived `SituationalRead { engagement, sentiment_shift, wants, self_check,
suggested_adjustment, confidence }`.

**(2) Theory-of-mind (user-state model).** Extend the per-person `Relationship` with an **inferred current
state** — what this person seems to feel / want / how they're responding — built from emotion + rapport +
conversation signals + history, **held as confidence-scored HYPOTHESES, never asserted as fact.** This is
the missing "model the other mind." It is grounded by the spark (ask/observe) and by the companion's
corrections.

**(3) Read→behavior ladder.** The read can do two things; they are NOT the same risk class.
- **(3a) Ask a question** — this path ALREADY EXISTS and must be *fed, not forked*. The working
  desk-object / unknown-voice questions and the spark's grounding questions all flow through one
  pipeline: `personality/curiosity_questions.py` `CuriosityQuestionBuffer` → `personality/proactive.py`
  `ProactiveGovernor` (≤~3–6/hr, engagement + user-stress gated) → TTS → outcome learning
  (engaged/dismissed/**annoyed** → cooldown ×4/×8, never-ask-twice via memory `has_existing_answer`). The
  spark already integrates here (`autonomy/orchestrator.py:2608` `_ask_grounding_curiosity_question`,
  `source="research"`, shared 3/hr cap). **The internal read asks by building a `CuriosityQuestion` and
  calling `buf.add()`** — inheriting every dedup / rate-limit / annoyance-learning / memory-suppression
  guard the desk/voice behavior depends on. No new ask-path; zero risk to what works. The "tried to
  resolve, couldn't, still want to know" gate is already the design: novelty is detected and *not
  auto-answered* — the gap becomes the question.
- **(3b) Adjust the conversation** — tone / depth / pace / pivot / give-space / disengage. *This* is the
  genuinely-new, riskiest capability (agency over the exchange). Gated exactly like
  `cognition/promotion.py`: **shadow** (narrate the would-be adjustment, change nothing) → **advisory**
  (surface/suggest; affect-cadence coupling proposes, unapplied) → **active** (actually adjust, earned by
  being right vs companion feedback, kill-switch, fastest auto-demote). Disengage/back-away is last and
  most conservative.

**(4) Crystallization — ride the existing gates, no new threshold.** Transient reads mostly **evaporate**
(they are not beliefs — events ≠ beliefs, the `interaction_review` lesson). A read that matters becomes a
belief through the machinery that ALREADY governs this — never an invented "N corroborations" number:
- **eligibility** — `contradiction_engine._is_belief_eligible` (weight/tag floors) +
  `EXTRACTION_DISCARD_THRESHOLD` (0.2): a low-confidence read ("they *might* be annoyed" @0.18) never
  becomes a belief no matter how often it recurs;
- **recurrence** — the existing `TensionRecord.revisit_count` / `maturation_score` (stable only at ≥50
  revisits AND ≥0.90 maturation, accumulating faster with corroboration) is the real "N occurrences" gate;
- **confidence** — set/validated by `epistemic/calibration/*` (≥20 outcomes, drift detection), not hand-set;
- **revisable** — beliefs version/supersede on new evidence (`contradiction_engine` version-collapse) and
  recalibrate: *"how food tastes changes"* is just supersede + recalibrate. Beliefs are knowledge and
  knowledge moves.
- **personality vs knowledge** — a read that is *self/personality-level* (who JARVIS is) does NOT enter
  the belief graph; it feeds the slow **0.7-inertia** trait layer (`consciousness/soul.py` semi-stable
  traits + `personality/evolution.py`), with the rollback safeguard. Core values stay immutable. So the
  two stability classes the operator named — fluid knowledge vs slow personality — are kept on their
  existing separate tracks.

The one genuinely-missing piece is an explicit *observation → settled-knowledge* status transition
(today recurring observations version-collapse but never graduate from "observed" to "factual"). If we
build it, it **rides the same gates above** (maturation + calibration), it is not a new counter.

**(5) Companion-learning loop.** The read is a **skill learned from the companion**, maturity-gated like
the 7-stage apprenticeship. Corrections ("I wasn't annoyed, I was thinking") update the user-state model
and, over time, the read's calibration. Early reads are crude; mature reads are nuanced. Being corrected
is **success**, not failure (same ethic as grounding).

## 5. Honesty guardrails (the no-theater contract)

- **The read is a hypothesis, never a fact.** Confidence-scored; JARVIS never states a person's inner
  state as truth. It may *ask* ("you've gone quiet — want me to drop this?") — which is a grounding act.
- **Theory-of-mind is validated externally** (ask / observe / companion feedback) — it cannot self-confirm.
- **Affect stays a readout** (`affect_state` cannot-lie clamp) and behavior-influence is shadow-first/gated.
- **Crystallization requires corroboration** — no single read becomes a belief.
- **Per-person isolation is enforced** (identity-boundary) — one person's read/beliefs never leak to another.

## 6. Rollout (smallest-safe-first; each gated on the prior)

- **P0 · Live read, logged only.** Fire situational-read thoughts during conversation from live context;
  surface them on the dashboard. Zero behavior change. *Advance when:* reads are coherent and track the
  actual conversation (operator eyeballs them), no tick-latency regression.
- **P1 · Theory-of-mind model (shadow).** Per-person inferred-state hypotheses on the `Relationship`;
  surfaced + confidence-scored. Drives nothing.
- **P2 · Crystallization valve (shadow→logged).** Recurring reads propose belief crystallizations
  ("would form: David prefers directness"); logged, not written, until the valve is trusted.
- **P3 · Read→behavior ADVISORY.** The read suggests adjustments (and affect-cadence proposes); narrated,
  operator sees "would have softened / wrapped up / asked." Still doesn't auto-act.
- **P4 · Read→behavior ACTIVE (earned).** Actual tone/depth/pace/pivot adjustments, gate-earned by being
  right, kill-switch, auto-demote. Disengage/back-away is the last and most conservative.
- **P5 · Companion-learning loop.** Corrections tune the read; maturity-gated calibration.

## 7. Risks (folded in up front)

- **Behavioral agency gone wrong** (pivots/disengages incorrectly) → hardest gating, narrate-long-before-act,
  smallest steps, kill-switch; disengage last.
- **Theory-of-mind wrong/creepy** → hypothesis-not-fact + validate; never assert; ask when unsure.
- **Re-pollution** → the crystallization valve (recurrence + corroboration), not per-turn beliefs.
- **Over-thinking / latency** → bounded read rate; the read must not slow the response (compute async,
  apply next-turn where needed).
- **Others around the main user** → keep their reads/beliefs lighter and scope-gated so JARVIS stays the
  *primary companion's* companion, never leaks the main user's private world, and never mistakes a visitor
  for the bond.

## 8. Open questions (decide before P3/P4)

1. How much **pivot-authority**, and how fast is it earned? (Tone/depth first; disengage much later?)
2. *(grounded)* Asking rides the existing `CuriosityQuestionBuffer` + `ProactiveGovernor` — "ask when it
   wants" is the existing curiosity gate (detect gap → don't auto-answer → ask), tolerance is the governor
   rate that already learns from dismissal/annoyance. Open sub-point: the shared hourly budget once the
   read is another question source.
3. *(grounded)* Crystallization rides existing eligibility + `TensionRecord` maturation + calibration
   gates — no invented number. Open: build the optional observation→settled-knowledge transition now, or
   let version-collapse + calibration suffice?
4. How different is the read for the **primary companion vs the others around them** — depth of
   theory-of-mind, and what (if anything) crystallizes into beliefs for non-primary people?
5. **Learning signal** — explicit corrections only, or also implicit (the companion's next reaction)?
6. Does the live read run **every turn** or only when a salience/affect threshold trips (cost vs coverage)?

## 9. What this is NOT

Not new consciousness machinery — it **wires and enriches** the existing tick + affect + relationship
model. Not unbounded agency — behavior is gated shadow→active and earns trust by being right. Not a
claim to feel or to know minds — reads are hypotheses, validated externally. Not multi-tenant — one
JARVIS, one primary companion, aware of the people around them. It is the **completion** of the
SyntheticSoul theory's already-stated internal-thought / social-adaptation / companion-learning pillars,
made to actually touch behavior — honestly, single-companion, and learned over time.
