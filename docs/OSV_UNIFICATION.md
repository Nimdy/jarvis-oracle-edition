# OSV Unification — feed the architecture manifest into JARVIS's self-view

> Status: **Phase A SHIPPED** (read-only/shadow). Closes the legibility loop: JARVIS describes her
> own architecture from the SAME code-grounded, drift-locked map that humans + AI agents use —
> not ~19 bespoke adapters + confabulation. Grounded-first, earn-don't-declare; no behavior authority.

## The vision
The architecture manifest (`subsystem_registry.json`, 98 subsystems + the 15-layer integrity stack,
CI-drift-locked) is the trusted self-description we built so AI stops getting JARVIS wrong. The OSV
(`cognition/self_view/`) is JARVIS's *own* self-model. They were disconnected. Unifying them means
**her self-knowledge becomes as complete + grounded as the external map** — the prerequisite for
truthful self-introspection, gap-targeted curiosity, and purposeful self-improvement.

## What OSV was (grounded)
6 provenance-honest dimensions (structural / performance / maturity / belief / **gaps (first-class)** /
change). Its structural dimension came from **~19 bespoke per-subsystem adapters** reading the dashboard
snapshot (`gather.subsystems_from_cache`) and did **not** read the registry. Real, but partial + bespoke.

## The unification map
| OSV dimension | Source |
|---|---|
| **Structural** "how I'm built" | **registry** → the full 98-subsystem map + 15-layer stack (home/status/does) |
| **Maturity** "earned vs gated" | registry status as *designed*-maturity, **cross-checked against the live gates** (caveat) |
| **Performance / Change** "how I'm doing now" | the live topology activity (snapshot deltas + events) — the viz dual-source |
| **Gaps** (first-class) | registry `gaps` → curiosity targets |

## The non-negotiable caveat
Registry `status` is a **point-in-time, code-grounded** assessment — DESIGNED-maturity, **not live**.
So registry facts get **`ADVISORY` provenance** (`is_measurement=False`) and never masquerade as a live
measurement. **Live state = the snapshot adapters; live maturity = the actual gates.** Registry =
structural truth + designed-maturity; never let static status read as live.

## Phase A — SHIPPED (read-only/shadow)
`gather.architecture_from_registry()` reads `subsystem_registry.json` → `sources["architecture"]`:
98 subsystems (each: name/area/status/authority/does/home as ADVISORY Facts), the 15-entry integrity
stack, and the registry gaps. The synthesizer adds an `architecture` section to the `SelfModel` and merges
the registry gaps into the first-class `gaps`. Verified: 98 subsystems, 15-entry stack, 31 gaps, status
`is_measurement=False`. Degrades to a first-class gap if the registry is unreadable. Read-only; zero
authority. (Needs a brain restart to load.)

## Phased plan (the rest)
- **B (P1/P2):** self-introspection + voice-grounding answer from the enriched OSV — "how are you built /
  what's subsystem X / what's not working" — grounded in the 98-map, no confabulation/ramble. Reuse
  `articulate.py` + the existing P1/P2 introspection.
- **C:** feed the live topology activity (snapshot deltas + events from the viz tap) into the
  performance/change dimensions — "what's actively working in me right now."
- **D+ (earned, gated):** curiosity targets the registry gaps; self-improvement closes them. Stays P3+/earned.

## Validation (anti-theater)
1. **Substrate exists** → consume-wire, not new machinery (OSV + registry + live tap all built).
2. **Maturity-gated correctly** → the read is P0/P1 read-only; curiosity-acting (P3) + self-improve (P4)
   stay earned. No authority touched.
3. **Anti-theater** → registry is code-grounded + CI-locked; gaps stay first-class. Grounded self-knowledge.
4. **Reuse, don't reinvent** → OSV `gather`/`articulate` + registry + viz dual-source (design §1.7).

## What this is NOT
Not new authority (read-only). Not a replacement for the live adapters (live state stays live; registry
adds the structural backbone + designed-maturity). Not a claim of completeness — gaps are first-class.
