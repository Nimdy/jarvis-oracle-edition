# Capability Authority: Shadow-First, Reversible, Self-Protecting

Status: **design + build — 2026-06-05**, from David's directive after the capability
lifecycle shipped. This is GENERAL — it governs the live-or-not authority of **every
acquired skill/plugin**, not any one skill.

## The doctrine (David's rule, made bulletproof)

> Everything is born in shadow. Promotion to active always needs the owner. Demotion
> back to shadow is a reversible circuit breaker the owner can pull **and** JARVIS can
> pull itself. The prior good version is always kept as the floor. Anything with
> irreversible real-world side effects is mocked in shadow until promoted.

Why this is the project's spirit, not a bolt-on: the whole substrate already earns trust
in shadow (the spark, companion cognition, the specialists). This completes the pattern by
adding the **backward edge** — active → shadow — turning a one-way ratchet into a control
loop with a circuit breaker. For the dignity anchor, the decisive property is: **the live
behavior a person depends on never changes without a trusted human's yes, and a misbehaving
live capability can always be pulled back to safety instantly.**

## Invariants

1. **Per `skill_id`, at most ONE version is `active` (authoritative).** The rest are
   `shadow` (executed for observation, results suppressed — already true in the resolver)
   or `quarantined`/`disabled`.
2. **Promotion raises authority → owner-gated.** Making a version authoritative
   (`make_authoritative`) is atomic per skill: it demotes the current active to shadow and
   records it as that version's `prior_authoritative` (the last-known-good floor).
3. **Demotion lowers authority → safe, so it is NOT owner-gated.** The owner can demote,
   and JARVIS can demote itself (the asymmetric gate — same rule as the weight-room: block
   authority freely, never raise it without a human). Demoting the active version
   re-promotes the skill's last-known-good floor so the skill keeps serving on the prior
   trusted version; if there is no floor, the skill goes **dormant** (no active) — safe:
   better silent than wrong.
4. **The circuit breaker auto-demotes (not auto-disables).** A live capability that trips
   the breaker (repeated crashes) is pulled to shadow and the floor is restored
   autonomously, then the owner is told — instead of just dying.
5. **Reversible + instant.** Both versions stay resident; a toggle is just "which one is
   authoritative," never a rebuild.
6. **Every authority change is audited** (actor, reason, from→to, timestamp) and emits an
   event.

## The one hard edge (named, not yet built)

Shadow works because shadow is observe-only. That holds for read/compute (even a shadow
web-scrape is a low-harm public GET). It BREAKS for irreversible side effects — you cannot
"shadow" *sending an email* or *unlocking a door* by really doing it. So for side-effecting
actions, "shadow" must mean **dry-run / mocked** until promoted. The governance/firewall
verdict (`reads_external`, and a future `writes_external`/`actuates`) is the right signal to
drive this. Flagged for when JARVIS controls anything physical in a home.

## Build increments

1. **Registry core (this commit):** `skill_id` / `prior_authoritative` / `last_authoritative_at`
   on `PluginRecord`; `versions_for_skill` / `active_for_skill`; `make_authoritative`
   (owner promote, atomic, records the floor); `demote` (owner OR auto, reverts to floor or
   dormant); circuit-breaker rewired to auto-demote-with-fallback; `PLUGIN_AUTHORITY_CHANGED`
   event + audit. Degrades gracefully when `skill_id` is empty (single-version skills).
2. **Wiring:** populate `skill_id` at deploy from the acquisition's `requested_by.skill_id`;
   one-time backfill for existing records (orchestrator owns the acquisition↔registry link).
3. **Surface:** `/api/plugins/{name}/demote` + a per-skill authority view; a dashboard
   control showing each skill's active version, its shadow candidates, the floor, and the
   shadow⇄active toggle.
4. **Self-protecting (later):** wire a health signal (contract failures, not just crashes)
   into auto-demote; mocked-shadow for side-effecting capabilities.

All shadow-first, additive, audited, and verified before it claims anything.
