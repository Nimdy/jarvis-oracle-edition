# Tier-2 Matrix NN redesign — un-orphan the inference + a real teacher + an earned flip

> Design session (2026-06-30), born from the NN-fleet audit. Diagnoses one specialist (`positive_memory`)
> end-to-end and proposes the fix for all five Tier-2 Matrix specialists. **Design only — no blind flip.**
> The first concrete step is a shadow *critic test* that validates the signal BEFORE we build the teacher.

## The pathology (grounded in code)
All five Tier-2 Matrix specialists (`positive_memory`, `negative_memory`, `speaker_profile`,
`temporal_pattern`, `skill_transfer`) share one shape:

1. **Orphaned inference.** `HemisphereOrchestrator._compute_signal_value` calls `_matrix_focus_signal`
   first (orchestrator.py ~1903); for these focuses that returns the **heuristic**
   `*Encoder.compute_signal_value(ctx)` and *returns* — so `self._engine.infer(network.id, …)` is never
   reached. `architect.py:226-231` says it outright: *"the orchestrator does not consume the raw output."*
   The trained NN does **no work**; the hand-coded heuristic feeds the broadcast slot.
2. **Circular teacher.** Training labels come from `_matrix_label(encoder.compute_signal_value(ctx))`
   (orchestrator.py), i.e. the NN is trained on a bucketed version of its **own encoder's** scalar, from
   the **same** features. It learns a deterministic function of its own input — self-mimicry. "Accuracy"
   is trivially high and **means nothing** about the world; yet `downstream_reward = accuracy * 1.5`
   inflates the impact/verification score.

Net: a trained network that neither acts nor learns anything real. This is the fleet's dominant
"feed wired, inference orphaned" disease at its root.

## The design — three coupled changes (the same earn-don't-declare pattern as self-sensing + the voice NN)

### 1. A real teacher (break the circle) — *predict the next lived signal*
Replace the self-label with the **next observed real lived signal**. `positive_memory_encoder` already
reads the genuine `positive_affect` / `mood_positivity` lived values; the teacher label should be
`_matrix_label(real_affect_at_t+1)`, not `_matrix_label(encoder_scalar_at_t)`. The NN's job becomes
**predicting where the real affect regime is heading** from the current features — a genuine task, not a
mirror. (Mirrors self-sensing: predict the next real frame.)

### 2. A persistence baseline (so "skill" means something) — *the negative control*
Credit the NN only for predicting the next lived signal **better than persistence** (assume affect stays
the same). `skill = accuracy(NN) − accuracy(persistence)`. This is exactly the self-sensing control that
separated a real engine (self-sensing, skill > 0) from a dead end (policy NN, Spearman ~0.06). No skill
above persistence ⇒ the focus is a dead end and we do **not** flip it (and we say so — no §24 tuning).

### 3. Un-orphan the inference, gated on EARNED maturity — *the flip*
In `_matrix_focus_signal`, for a Tier-2 focus:
- **If** stage ≥ `BROADCAST_ELIGIBLE` **and** the NN beats persistence on lived data (weight-room gate
  passes) → return `engine.infer(network.id, encoder.encode(ctx))` mapped to a scalar.
- **Else** → return the heuristic `compute_signal_value(ctx)` — the safe floor for immature/unproven NNs.

The heuristic stays the floor forever for any focus that never earns it. The NN speaks **only** once it
has earned it, and the flip is **reversible** (auto-rollback if lived accuracy degrades, like
intent_shadow's rollback).

## Validation discipline (validate the signal BEFORE building — the "critic test")
Per the autonomous-growth lesson (validate the causal signal offline first — it saved weeks twice): do
**not** build the teacher/flip until a shadow critic test shows the signal is real.

## Phased rollout
- **P0 — shadow critic test (BUILD FIRST, safe).** In the matrix observation loop, additionally buffer
  `(features_at_t, real_lived_signal_at_t+1)` and log, per focus, `accuracy(NN-style predictor)` vs
  `accuracy(persistence)` — **no flip, no behavior change**. Surface on `/v2/nnfleet` + the NN lab. This
  answers "does this focus carry a real predictable signal?" for each of the five.
- **P1 — real-teacher distillation (earned by P0).** For focuses that pass P0, switch the training label
  to the next-lived-signal and train through the canonical `DistillationCollector` (lived-only, the
  weight-room firewall already in place).
- **P2 — earned inference-flip (gated).** Flip `_matrix_focus_signal` to consume the NN once stage ≥
  `BROADCAST_ELIGIBLE` **and** the lived skill > persistence; keep auto-rollback. Heuristic remains the
  pre-maturity fallback.
- Applies identically to all five Tier-2 specialists; `positive_memory` is the template.

## What this is NOT
- Not a blind flip — the heuristic floor stays until the NN earns the slot on lived data.
- Not keeping the circular teacher — a focus that can't beat persistence is reported as a dead end, not
  tuned green.
- Not new authority — Tier-2 broadcast influence is still gated by the existing weight-room + dwell gates.

## Honest status
`positive_memory` (and its four siblings) are correctly tracked as `ORPHANED` in `nn_fleet_registry.json`.
This redesign is the path to make them genuinely contribute to her cognition — earned, not declared.
Next concrete step: build **P0 (the shadow critic test)** so we learn which of the five carry a real
signal worth distilling.
