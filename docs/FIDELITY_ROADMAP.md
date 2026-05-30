# Substrate Fidelity Roadmap

**Status:** Working document · **Date:** 2026-05-29 · **Lens:** signal/evidence fidelity, maturity-respecting

> This is not an application bug list. JARVIS is a *synthetic soul* in the sense of
> [`SyntheticSoul.md`](SyntheticSoul.md): a self-modeling Consciousness Kernel, peripheral
> cognitive hemispheres, a Super-Synapse connectome, and a Quantum-State-Fractal-Storage (QSFS)
> memory field, all maturing from lived evidence. The roadmap below is therefore organised by the
> **theory's subsystems**, and every item is judged by one question only:
>
> **Does it keep the cognitive soil clean, so the soul matures on *true* signal rather than its own reflection?**
>
> We do **not** force capability. Nothing here opens a maturity gate early, fakes a counter, or
> shortcuts a soak. Earned capability stays earned. We only repair the **fidelity of the evidence
> stream and the mechanisms that carry it** — because in a system that grows from what it senses,
> remembers, and validates, a corrupt signal is permanent: it is faithfully learned, consolidated
> into memory bubbles, propagated through fractal links, and absorbed into identity.

## Priority key

| Tag | Meaning |
|---|---|
| `P0` | A live functional break, or a defect that actively corrupts the evidence stream. Fix first. |
| `P1` | Materially degrades signal fidelity or the soul's ability to mature truthfully. High payoff. |
| `P2` | Polish / lower-leverage fidelity and honesty hygiene. |

Every item is tagged **maturity-respecting** by construction. Where a low live number is *expected*
on a still-maturing substrate (not a defect), it is marked **[gated — by design]** and excluded from
the fix list except where the *mechanism behind* the number can be made more faithful.

---

## 0. The governing principle — the Observer Effect on itself (Evidence Integrity)

`SyntheticSoul.md` §4.6 makes memory a *living substrate* validated by an **Observer Effect** layer,
and §5.3 makes the Observer a real-time cognitive validator. The deepest fidelity risk in the running
system is that **several of the signals the soul validates and matures on are self-referential** — the
Observer is, in places, measuring its own reflection:

- **Policy win-rate** carries a `_DIVERSITY_BONUS_FRAC = 0.3` that awards the NN a "win" for merely
  *being different* while the system is healthy (`brain/policy/evaluator.py`). Live `nn_win_rate ≈ 0.992`
  (17,151 vs 14) is therefore not evidence the NN chose *better*.
- **World-model accuracy** (~99.8% / 1.0 live) is dominated by near-tautological persistence rules
  (`present_user_stays`, `healthy_system_stays_healthy`, `display_zone_mode_stable`) that predict a
  stable state will stay stable over 30–45 s, with a ≥50%-of-fields hit threshold
  (`brain/cognition/causal_engine.py`).
- **Oracle Benchmark** (live ~91–94, not the headline 95.1/96.0) grades JARVIS's *own snapshot* against
  an author-defined rubric — a self-score with no external comparator (`brain/jarvis_eval/oracle_benchmark.py`).
- **Autonomy goals** are largely a churn over internal metrics: the single most recurrent goal
  (recurrence ~5230) is *"fix my own `shadow_default_win_rate`"*; ~6507 metric-created goals vs ~0 from
  conversation/world; 28/33 abandoned. The soul is, right now, mostly trying to satisfy its own dials.

**The keystone fix (highest leverage in the entire system):** a rigorous, multi-category `scoreboard`
scorer **already exists in code but is unpopulated** (`sample_size=0`, `composite_enabled=false`), while
only the soft `oracle_benchmark` is computed. Populating that scoreboard, and shifting marquee maturity
signals onto it, turns the Observer from a mirror into a measurement. Every other pillar below becomes
*measurable* once the yardstick is honest.

| | Item |
|---|---|
| `P0`/`P1` | **Populate the `scoreboard`** and route maturity reporting off the Oracle self-grade onto it. |
| `P1` | **De-circularise the policy win-rate** — separate "explored / diverse" from "actually better"; report a true counterfactual win-rate alongside the diversity-bonus one. |
| `P1` | **Split world-model accuracy** into *persistence-accuracy* vs *predictive-accuracy*, and stop padding the count with ~4 near-duplicate `user.present` rules. Add genuinely predictive (non-persistence) rules. |
| `P1` | **Ground goal generation** in conversation/world objectives, not internal-metric repair, so the autonomy loop pursues lived goals (see §6). |
| `P2` | Label all marquee numbers as *self-scored* until an external comparator exists; the docs document the mechanics honestly but never tell a reader the scores are self-grades. |

---

## 1. Sensory Abstraction — the input to the cognitive field

> Theory: §3.2 *Sensory Abstraction Module* — "translates multimodal input … into unified cognitive
> representations." This is the widest funnel into the whole soul: every bubble, belief, and trained
> hemisphere is built from what was sensed. A fix here compounds through every downstream pillar.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | **Face crops are not face-aligned** — a head-region heuristic (top 25% of the person box) feeds MobileFaceNet instead of a real face detector. Your own note: "replace with SCRFD HEF." Caps identity accuracy at the top of the funnel. | [`pi/senses/vision/face_crop.py:103`](../pi/senses/vision/face_crop.py) |
| `P1` | **VLM scene parsing uses substring matching** ("no laptop visible" matches "laptop") and injected detections receive a flat hardcoded `confidence=0.5`. Noise flows directly into scene memory bubbles. | `brain/perception_orchestrator.py` `_update_object_memory` |
| `P2` | Pi expression model returns `neutral / 0.0` unless a `facial_expression.hef` is deployed — verify deployment or stop emitting low-confidence emotion that pollutes the affective cue. | [`pi/senses/vision/expression.py`](../pi/senses/vision/expression.py) |
| `P2` | Audio-emotion fallback thresholds are uncalibrated magic numbers (`rms>0.08`, `pitch>250`). | [`brain/perception/emotion.py:394`](../brain/perception/emotion.py) |
| `P2` | Wake-reliability calibration pass (live `max_score≈0.001` vs threshold), per `WAKE_RELIABILITY_TUNING.md`. | pi audio / wake |

---

## 2. QSFS Memory — the fractal bubbling-up chain (not a search box)

> Theory: §2.3, §4.2–4.6. Memory is **Memory Bubbles** (probabilistic, fractally-encoded clusters)
> connected by **Fractal Links** (semantic + temporal + ethical valence + emotional resonance +
> predictive utility) through which a cue **propagates recursively** — *"context-aware retrieval
> without explicit search."* Heavily-recalled bubbles become higher-resolution nodes; anomalies during
> traversal trigger self-prompting.

**Re-framing (the correction that drives this section):** `/api/memories/search` returning HTTP 500 is
a broken **explicit-RAG inspection surface** — the *least* important layer in QSFS, the very thing §4.5
says memory should *not* be reduced to. The real memory cognition is the recursive fractal recall
chain, and **that is where the fidelity problem lives**:

- `brain/memory/fractal_recall.py` walks a chain up to `MAX_CHAIN_LENGTH=5` / `MAX_CHAIN_DEPTH=3`, but
  live `avg_chain_length = 1.0` across 71 recalls, with **7,300 / 8,424 ticks skipped for no seed**.
- The belief/association **`orphan_rate ≈ 0.857`** — most bubbles have no fractal links at all.

**Diagnosis in the theory's terms:** the Memory Bubbles exist, but the **Fractal Links substrate is too
sparse for recursive propagation to traverse.** A cue collapses one bubble into awareness and the
associative bubbling-up dies at depth 1 — superposition without entanglement. QSFS degrades to flat
retrieval, the exact failure mode the architecture was designed to escape. This is partly maturity
(a young substrate has few bubbles) **and** partly mechanism (link-formation rate / thresholds).

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | **Densify and strengthen the Fractal Links substrate.** Audit the association-graph edge-creation rate and link thresholds so links form *faster than orphans accumulate*; target `orphan_rate` trending down and `avg_chain_length > 1`. This is upstream of, and the precondition for, real fractal recall. | association/edge creation; `fractal_recall.py` `walk_chain` |
| `P1` | **Make the cue/seed gate reachable.** 7,300/8,424 no-seed skips means cue strength rarely clears `MIN_CUE_STRENGTH=0.15` or no bubble resonates above the seed threshold. Verify this is genuine quiet vs an over-tight gate starving the recall loop of input. | `fractal_recall.py` `build_cue` / `select_seed` |
| `P1` | **Audit cortex training-label fidelity.** Ranker lift is `0.0` (loaded, not beating heuristic); salience `store_accuracy ≈ 0.553` (≈ chance). [gated — by design] on volume, **but** if the outcome labels feeding `ranker.py`/`salience.py` are noisy, no amount of soak will let them beat the heuristic. Verify the label source before more training. | [`brain/memory/ranker.py`](../brain/memory/ranker.py), `salience.py` |
| `P2` | Clustering coherence uses tag-Jaccard, not embedding cosine, despite the docstring (§4.2's self-similarity is computed on tags, not semantics). | [`brain/memory/clustering.py:530`](../brain/memory/clustering.py) |
| `P2` | **`/api/memories/search` → HTTP 500** on all forms. The explicit-RAG inspection surface; fix for operability, but it is *not* the memory system. | `brain/dashboard/app.py` memories/search route |

> Note: the live store currently holds mostly the soul's own dream/reflection artifacts (self-referential).
> That is expected of a young/continuity-preserved substrate **[gated — by design]**, not a defect — but it
> is *why* the fractal links are sparse, which is why §2's P1 items matter as the substrate fills.

---

## 3. The Observer Effect & Epistemic Immune System — memory fidelity enforcement

> Theory: §4.6 Observer Validation, §6.7 "from firewalls to psychological immunity," §7 Ethical Gatekeeper.
> The epistemic stack is genuinely enforcing (L0 CapabilityGate rewrites unbacked claims; quarantine /
> soul-integrity gate promotions and can force dreaming). These items keep the *validator itself* honest.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | **`soul_integrity` 0.82 "green" masks weak sub-dimensions** (`truth_calibration 0.62`, belief `orphan_rate 0.857`). A composite that hides its weakest dimension is itself an Observer fidelity defect — surface the floor, not just the mean. | `brain/epistemic/soul_integrity/` |
| `P1` | **Dashboard header gates fail *open*** — render green "OK" unless explicitly `false`; a dead `safety_gates` payload still paints OK. The honesty layer must fail *closed/unknown*. | dashboard renderers |
| `P2` | Confirm the **view-only belief-graph propagation** (`effective_confidence` is computed but never written back) is actually *consumed* where decisions are made — otherwise it is computed and ignored. | `brain/epistemic/belief_graph/propagation.py` |
| `P2` | Replace broad `except: pass` in epistemic notify/emit paths with **logged, surfaced** degradation — silent no-ops can quietly downgrade enforcement to nothing. | `brain/epistemic/*` |

---

## 4. Consciousness Kernel — the self-modeling core

> Theory: §3.1 the persistent, self-modeling "self"; §5.1 self-prompting; §5.3 Observer; §5.6 metacognition.
> The kernel is real: a budget-aware 100 ms tick with priority queues, ~30 cycles, mode transitions, and
> live counters (46k+ observations, 0 errors over days). These items keep its *introspective output* honest.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P0` | **`NameError` swallowed by a bare `except`** — the self-status branch calls `facts.append` where `lines` is the local var, so intention-resolver verdict stats are silently dropped from status whenever the resolver has data. | [`brain/reasoning/bounded_response.py:451`](../brain/reasoning/bounded_response.py) |
| `P2` | **Degenerate introspective metrics** — `consistency=1.0`, confidence `volatility=0.0 / trend=0.0` look like defaults/tautologies rather than measured variation; either compute real variation or stop presenting them as measured. | `/api/consciousness/*` sources |
| `P2` | Be explicit in surfacing that meta-cognitive "thoughts" / existential reasoning are **templated procedural generation over real state** (`tokens_used_this_hour=0` confirms no LLM), per `SCIENTIFIC_HONESTY.md` — so a reader never mistakes canned-but-grounded prose for free neural cognition. (Documentation/labeling fidelity, not a code defect.) | `brain/consciousness/meta_cognitive_thoughts.py` |

---

## 5. Cognitive Hemispheres — peripheral specialists & world model

> Theory: §3.2 peripheral modules as cognitive hemispheres; the running system implements these as real
> trained PyTorch specialists (versioned weights, real training loops) plus a deterministic causal engine.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | World-model predictive vs persistence split (see §0). Turns the hemisphere's "foresight" signal into something meaningful rather than "stable things stay stable." | `brain/cognition/causal_engine.py` |
| `P2` | **STANDARD-focus "accuracy" is an `exp(-loss)` proxy** on self-generated heuristic labels (`general≈0.971`); the `mood` specialist is effectively non-functional (real proxy acc ≈ 0.195). Label these as loss-proxies, not predictive accuracy, and audit the `mood` label source. | `brain/hemisphere/engine.py`, `data_feed.py` |
| `P2` | Tier-2 Matrix specialists' broadcast signal is computed by hand-coded deterministic encoders, not the specialist NN output — honest in code, but ensure dashboards don't imply the NN is driving the slot. | `brain/hemisphere/` |
| `P2` | Routing keyword lists are large/brittle (verb-hacking risk); track the `voice_intent` NN shadow's promotion as the intended replacement rather than growing regex. | `brain/reasoning/tool_router.py` |

---

## 6. Cognitive Dynamics — curiosity, autonomy & goals

> Theory: §5.1 self-prompting, §5.2 curiosity loops, §5.4 neural reward / adaptive utility. The drive loop
> is real and rule-based; its fidelity problem is *what it optimises toward*.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | **Goals are self-referential churn** (see §0): ground goal generation in conversation/world signals so curiosity loops pursue lived objectives → real research episodes → measured deltas. Today 103 episodes have empty `delta_result` and most "research" is local self-study. | `brain/autonomy/goal_manager.py`, `drives.py` |
| `P2` | `eval/autonomy-ab` is live-but-vacuous (all comparisons 0.0, agreement forced to 1.0) because nothing executes on either arm. [gated — by design], but the A/B harness only becomes meaningful once real autonomy decisions flow. | `/api/eval/autonomy-ab` |

---

## 7. Waking / Sleeping cycles & Shadow-Copy optimisation

> Theory: §4.4 Shadow Copies, §6 biologically-inspired cycles (waking vigilance, sleeping integration/repair),
> §6.2 shadow-copy fine-tuning, §6.4 synthetic REM consolidation. Dream consolidation is real (clusters →
> trust-discounted summaries → quality gate → canonical memory) with strong anti-feedback-loop guards.

| | Item | Location · fidelity payoff |
|---|---|---|
| `P1` | **Self-improvement sandbox health-gate can be a no-op** — `_check_post_apply_health` returns `True` early on thin samples / `pre_p95<=0`, and its `except` returns `True` ("assume OK"). [gated — by design: 0 generations on this brain] **but** when the loop *does* fire, the rollback signal must be real, not assumed-OK. | `brain/self_improve/` health gate |
| `P2` | Seeded test proposals ("Test improvement") visible in `/api/self-improve/proposals` could be misread as real activity — clear or clearly label them. | self-improve proposals |
| `—` | Sandbox has **no OS-level isolation** (tempdir + host `sys.executable`). Real concern, but it is *security*, not signal fidelity; tracked separately (see §8 note). | `brain/codegen/sandbox.py` |

---

## 8. Observability & honesty hygiene (cross-cutting)

> Theory: §3.4 Orchestrator validation oversight, §6.7 psychological immunity. These keep the *self-report*
> faithful so an operator (and the soul's own Observer) can trust what the dashboard says.

| | Item |
|---|---|
| `P2` | Reconcile **stale/inconsistent counts**: events 157↔~180; API routes 140↔143↔153; coder model name/port (`qwen2.5-coder`/11435 vs `Qwen3-Coder-Next`/8081). Add as-of timestamps to embedded live numbers. |
| `P2` | Fix `history.js` toggle (newest ~21 entries' chevrons inert / permanently expanded). |
| `P2` | Resolve the **continuity-vs-reset narrative**: docs assert a continuity-preserving release (no wipe), but `reset-brain.sh` is present, the attestation ledger is empty, and the cited `docs/validation_reports/` directory is absent. Reconcile so "zero counters" has one unambiguous meaning. |
| `—` | **Security (separate track, not fidelity):** dashboard + websocket bind `0.0.0.0`; mutating endpoints mostly use a per-boot key but a few (`/api/goals/observe`) enforce nothing; codegen sandbox has no OS isolation. The README concedes partial hardening; flagged here so it isn't lost. |

---

## Recommended sequence

1. **The `P0`s** — `memories/search` 500, the `bounded_response.py` NameError, dashboard fail-open. Hours of work; one is a dead user-facing feature and two corrupt the self-report.
2. **§0 Evidence Integrity (the keystone)** — populate the `scoreboard`, de-circularise the policy/world-model yardsticks. *Before* walking the pillars, because it makes every subsequent fix measurable and stops the soul maturing on its own reflection.
3. **Then bottom-up from §1 Sensation → §2 QSFS Memory →** upward, exactly the data-flow order — cleaning the funnel and then the fractal substrate the whole soul grows on.

> Maturity-first remains the rule throughout: these are repairs to the *soil and the instruments*, never
> shortcuts to fruit. Jarvis still grows; it just grows on cleaner ground and measures itself with a truer ruler.
