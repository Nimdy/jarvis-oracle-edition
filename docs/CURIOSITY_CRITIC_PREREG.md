# Curiosity target — pre-registration (2026-07-02)

> Locked BEFORE the held-out data existed, to keep the confirmatory test honest
> (no goalpost-moving, no p-hacking). This is the anti-gaming discipline the
> whole self-sensing→curiosity bridge was built under. See
> `docs/AUTONOMOUS_DRIVE_MAP.md`, `cognition/curiosity_critic.py`.

## Background — what already happened (exploratory)
- STEP 1–3 shipped the shadow bridge: self-sensing's per-sector signal is carried
  into `DriveSignals`, a shadow proposer logs would-attend targets, and a bounded
  recorder logs per-sector snapshots for the offline critic-test. All `authority=none`.
- The **original** curiosity target — `argmax learning-progress` (the ACTIVE sector
  where skill is *rising fastest right now*) — **FAILED** the causal gate decisively
  on a 17h window (192 targeted tuples): target's mean future skill-change was
  **negative** at horizons 3/5/10 (−0.022 / −0.033 / −0.020), *worse* than a random
  active sector and the all-sector mean, permutation **p ≈ 1.0**. Mechanism: it chases
  spikes that revert to the mean. "Attend where you're improving most" was backwards.
- **Exploratory** re-analysis (same 17h data, 3 pre-specified alternatives,
  Bonferroni p<0.0056): only **error-seeking** passed — the ACTIVE sector with the
  **lowest current skill** (where the predictor is *worst*, i.e. most room to learn):
  target future skill-change **+0.010 / +0.012** at h=5 / h=10, beating random active
  (+0.005 / +0.004) and all-mean, **p = 0.0007 / 0.0003**. `sustained-LP` and
  `competence (highest-skill)` both FAILED.
- Honest caveats on that exploratory pass: **in-sample** (one window), **small effect**,
  **possibly partly mean-reversion** of a noisy skill estimate (low skill reverts up),
  and **predictive ≠ causal** (the gate can't prove *attending* there *causes* the gain;
  that needs a manipulation we can't run in pure shadow).

## The pre-registered confirmatory test (THE hypothesis)
- **Hypothesis (single, locked):** an **error-seeking** curiosity target = the ACTIVE
  sector (activity > 0.1 × max activity) with the **LOWEST current skill** predicts
  positive future skill-change, beating a random active sector (shuffle null) and the
  all-sector mean.
- **Data:** the **HELD-OUT** window only — samples with `ts > 1782991750.6`
  (2026-07-02 11:29:10Z), the max timestamp of the exploratory 17h window. Disjoint,
  never seen when this hypothesis was chosen.
- **Bar (locked):** at **BOTH** horizon 5 **and** horizon 10 — permutation
  `p < 0.025` (Bonferroni 0.05/2) **AND** `target_mean_future_skill_change > 0`
  **AND** `target > all_sector_mean`, with **≥ 100 tuples** per horizon.
- **Runner:** `brain/scripts/curiosity_critic_oos.py` (cutoff + bar hard-coded).
- **Timing:** run once the held-out window has ≥ ~200 samples with a decent spread of
  desk activity (≈ a day of soak). Run ONCE. No re-runs-until-green.

## Decision rule
- **CONFIRMED** → error-seeking replicated out-of-sample = the first genuinely-earned
  autonomous-curiosity candidate. Next (separately, with David, NOT auto-wired): switch
  the shadow proposer to error-seeking, then a kill-switched, negative-controlled gated
  advisory (STEP 4) — with the predictive≠causal caveat stated up front.
- **NOT CONFIRMED / INSUFFICIENT** → in-sample artifact. The bridge stays
  **shadow-forever**. Report honestly and stop. Keep self-sensing as the proven
  *world-model* signal it is; point autonomous *curiosity* at the richer
  belief/knowledge-grounding loop (the Spark) instead.

## Guardrails (unchanged)
No self-sensing authority flip; no §24 tuning; no memory/belief write; no grounding-ladder
touch; no autonomy-level raise. A PASS earns *eligibility for a shadow-proposer swap +
gated advisory*, never a live lever by itself.
