# Architecture revalidation (2026-06-30)

> Adversarial revalidation of this session's NN findings + 3 code fixes against the design docs
> (15-agent workflow: 5 doc-readers → 9 validators → synthesis). Triggered by the operator: "this system
> is very very complex — revalidate before continuing." Goal: catch any deliberate maturity-gating /
> shadow-by-design that was mislabeled as a bug.

## Verdict: 6 fixes hold, 2 labels corrected, **nothing reverted**

### Fixes — ALL validated correct (verified against live source)
| Fix | Commit | Verdict |
|---|---|---|
| `intent_shadow` `maybe_promote` wired into `observe()` | ba0efe2 | **fix-correct** — restores a *documented autonomous ladder* whose promotion caller was missing (rollback was wired, promote wasn't = asymmetric, doctrine-forbidden). Self-gated (samples≥500, dwell≥100, agreement≥0.80, 24h cooldown — all unchanged). **NOT an operator-held gate.** |
| `intent_shadow` get_state-shape + sample_counts | ba0efe2 | **fix-correct** — silent dead-consume-wires producing fake zeros (0 predictions from 25,870 obs; teacher count blind). |
| `plan_evaluator` live-engine wire | 1304723 | **fix-correct** — `HemisphereEngine.get_instance()` never existed → silent no-op. Persists only a shadow artifact; no authority. |
| claim_classifier friction feed | cc04f08 | **fix-correct** — `record()` missing required `teacher` + used `source=` → TypeError swallowed every call on a training-only feed. |
| voice seed → DistillationCollector | 8d9b5ee | **fix-correct** — the canonical teacher-signal sink; bespoke corpus was duplication. |

### Mislabels — corrected (labels only; no code was changed for these)
| Claim | Verdict | Correction |
|---|---|---|
| Tier-2 "orphaned inference" = pathology | **AS-DESIGNED** | NN held advisory-until-earned; encoder/heuristic is the consumed floor by design. Registry reclassified `ORPHANED → advisory-by-design` (positive_memory, skill_transfer). |
| Tier-2 "circular teacher" = flaw | **AS-DESIGNED** | The documented self-supervised distillation bootstrap (`MATRIX_V2 §4.5.A`); confers no authority. |
| positive/negative_memory 0.0 inputs | **AS-DESIGNED** | The honest P3.6 no-data→no-signal contract — refusing to feed uncalibrated/accuracy-tainted sources. Wiring them would be the integrity violation. |

`TIER2_NN_REDESIGN.md` P1/P2 **retracted**; P0 critic kept as a birth-eligibility/learnability diagnostic.

### Round 2 — the last 2 "ORPHANED" also validated AS-DESIGNED (ORPHANED count → 0)
| Claim | Verdict | Basis |
|---|---|---|
| `intention_delivery` = orphaned/dead | **AS-DESIGNED (staged)** | `types.py:524` "STAGE 2 RESERVED SLOT — DO NOT UNCOMMENT UNTIL STAGE 1 HAS LANDED" — deliberately inert, frozen contract (`INTENTION_STAGE_1_DESIGN.md §2`); training data already produced by IntentionResolver/proactive. |
| `thought_trigger` = orphaned + "dim mismatch 44 vs 24" | **AS-DESIGNED (shadow spark)** | SPARK §3/§8-P3 `thought_trigger_selector` — wired focus + `DistillationConfig(24→13, dim-asserted)` + a real teacher feed (`THOUGHT_VALIDATION_OUTCOME` via `event_bridge`). Audit's "44 vs 24" was **wrong** (real 24/13). Pure-shadow-until-earned. |

**Meta-finding: all 4 originally-"ORPHANED" NNs were as-designed.** The agent-audit's entire "orphaned
inference" category was pattern-matching deliberate maturity-gating + reserved scaffolding as bugs. The
*real* defects were the 3 dead-**wires** (claim feed, intent_shadow, plan_evaluator) — which the audit had
in other categories. Registry ORPHANED count: 4 → 0. The system is markedly more intentional than the
audit's headline suggested — which is exactly why the design docs must be read before labeling.

## The deciding principle (why both calls were right)
`intent_shadow` and the Tier-2 NN sit on **opposite sides of the earn-don't-declare line**:
- `intent_shadow` is a **documented autonomous threshold ladder** (absent from every operator-held /
  do-not-tune list) whose promotion wiring was genuinely **missing** → wiring the gated caller **restores**
  the design.
- The Tier-2 NN is held **advisory by design** (the encoder is the consumed signal; the NN is a maturing
  student) → routing `engine.infer` into the broadcast slot would **violate** it.

## DO-NOT list (would violate the design)
- Do **not** route Tier-2 `engine.infer` output into the broadcast slot (use the encoder/heuristic floor).
- Do **not** wire `emotion_depth`'s fidelity scalar or unnormalized contradiction counts into the memory
  encoders (P3.6 honesty floor).
- If trained-NN observability is wanted: SHADOW telemetry only (log infer beside the encoder value), never
  as the consumed value.

## Post-restart soak checks (the fixes are committed but inert until the operator restarts)
- `intent_shadow`: NN predictions climb from 0; teacher-samples `tool_router` total non-zero; **stays at
  SHADOW** until all 4 gates met — an immediate level jump = stale/pre-warmed state, investigate.
- `plan_evaluator`: shadow artifacts appear under `~/.jarvis/acquisition_shadows/` during real
  acquisitions; `get_active('plan_evaluator')==None` → predictions correctly no-op (idle, not a regression).
- Lesson reaffirmed: post-reset Tier-2 standing resets to probationary/0.0 and **re-earns on live reps** —
  that is the firewall design, not "wiped." (Matches the VERIFICATION doc: 2 specialists had promoted
  autonomously pre-reset.)
