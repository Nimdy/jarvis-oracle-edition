# Spatial Place Consolidation — persistent place memory (DESIGN, shadow / zero-authority)

> Status: **DESIGN / PRE-MATURE**. Operator-raised (David, 2026-06-22). This is a
> *shadow-lane consolidation* of the existing spatial-episodic album — **not** a new
> perception system, **not** an authority change, **not** new geometry. It must live
> entirely inside the P4 HRR + P5 Mental-World governance rules (AGENTS.md §"HRR / VSA
> Governance Rules"). Nothing here flips a status marker or grants authority.

## The observation (operator, validated by trace)
JARVIS is **stationary** — the camera/lidar pan (and the operator PTZ), but the *body
never relocates*. Yet the spatial-episodic album (`memory/spatial_episodic_store.py`) is
**session-keyed**: every session spins up a fresh "world" (`~/.jarvis/spatial/episodic/
<session>.jsonl`). Live: **163 sessions** — i.e. ~163 near-duplicate snapshots of the
*same room*. Confirmed by trace: there is **no place-recognition / merge / consolidation**
anywhere in the spatial code (`grep` for `same_room|place_recog|merge_room|consolidat`
returns nothing in `cognition/` / `spatial/` / `perception/`).

**Consequence:** spatial memory *fragments* instead of *accumulating*. JARVIS re-learns the
room every session rather than knowing "this is the same room I already know."

**Why it matters beyond tidiness — it's one root cause behind three things we've seen:**
- **Self-sensing washout** (the predictor re-learns the room each idle period → regresses
  toward persistence → dynamic skill collapsed +49% → −6%).
- **World-model is ~98% tautological persistence** (it re-confirms the static room with
  nothing to predict *against*).
- **The album fragments** into 163 same-room worlds.

All three are one missing idea: **the system never separates "the persistent place (known,
stable)" from "what is changing (the signal)."**

## The design — place-keyed consolidation
A derived consolidation over the *already-canonical* scene graphs. It changes **how the
shadow album is keyed and accumulated**, not what authority it has.

1. **Place recognition (derived, NO new geometry).** Fingerprint the current scene from
   **canonical room-frame geometry** — the lidar room model's dimensions/walls + the
   `SpatialAnchor`s, read via the **public accessors** (`perception_orchestrator.
   get_scene_snapshot()`, `get_spatial_tracks()`, `get_spatial_anchors()`) per P5 rule 2.
   On session start, match the fingerprint against known places (a deterministic geometry
   similarity over canonical coords — thresholds from `cognition.spatial_schema`, not
   redefined). Match → continue that place; novel → new place (rare for a stationary unit).
   *No independent raw-detection geometry, no alternate 2.5D pipeline (P5 rule 3).*

2. **Persistent place record (the compression).** The album becomes **place-keyed**: one
   durable record per recognized place that accumulates the consolidated, **vector-free**
   scene graph (`graph.to_dict()`) — stable geometry + objects-seen with recency/confidence,
   updated by each session's observation. **163 worlds → ~1 persistent place.** Storage stays
   append-only JSONL under the album root, still stamped with the album's pinned-false
   `AUTHORITY_FLAGS` and `loaded_from_store=True` on restore.

3. **Holographic accumulation (vectors stay transient).** The place's HRR is **re-encoded on
   demand** from the consolidated graph (P4 contract: "never stores a raw HRR vector — the
   world is re-encodable losslessly from {graph, dim, seed, vocab_version}"). The
   *accumulation* lives in the consolidated **graph** (bundle new canonical bindings into the
   place with recency/confidence); a holographic bundle may be computed *in-process* for
   similarity/recall but is **never persisted and never enters any API** (HRR rule 7, P5
   rules 6–7).

4. **Episodes become change-deltas.** Genuine changes — a new object, moved furniture, a
   visitor — are recorded as sparse **change-episodes anchored to the place**, not as new
   worlds. **Compress the room; keep the changes** (preserve the temporal record).

5. **Self-sensing tie-in (downstream, gated separately).** Once a persistent place exists,
   the self-sensing predictor can predict **deviations from the known place** (the place = the
   persistence prior) → focuses learning on motion → addresses the washout. This is a *benefit*,
   not part of this design's authority surface, and earns its own gate.

## What this is NOT (anti-AI-theater / anti-gaming — held to AGENTS.md)
- **NOT a new perception system / geometry** (P5 rules 1, 3): a derived projection over the
  existing canonical scene graph. It observes/detects/corrects nothing.
- **NOT an authority change** (HRR rule 3; P5 rule 7): stays **zero-authority, shadow,
  PRE-MATURE**. Writes no canonical memory / beliefs / policy / autonomy / identity. Still a
  "photo album, not testimony" — just a *consolidated* one. `hrr_side_effects` stays 0.
- **NOT raw vectors anywhere** (HRR rule 7; P5 rule 6): only the vector-free consolidated
  graph persists; the truth-probe that walks `hrr_scene` for `vector`/`raw_vector` keys must
  still pass.
- **NOT a metric to game.** The value is a **real, verifiable** reduction (163 worlds → ~1
  place) and a persistent, accumulating room representation — not an impressive-looking
  self-score. The compression count and the place-match are deterministic and auditable.
- **Respects the structural isolation** — no forbidden imports; `_scan_hrr_forbidden_imports`
  and `hrr_policy_non_influence` must remain green; the twin gate (`ENABLE_HRR_SHADOW` +
  `ENABLE_HRR_SPATIAL_SCENE`, default OFF) still governs activation.

## Integrity guardrails (the design's own anti-gaming)
- Place-match uses **canonical, deterministic geometry** — never a fabricated similarity, and
  it must *fail closed* to "new place" when geometry is unavailable (mirror the facade's empty
  scene `reason="canonical_spatial_state_unavailable"`).
- Consolidation must **preserve the change-record** — collapse stable structure, never erase
  the temporal truth of what changed/when (no over-compression that hides reality).
- Honest status: **PRE-MATURE / advisory / zero-authority**; never auto-flips to SHIPPED.
- The only "win" claimed is the **measurable** compression + persistence — no claim of
  understanding, no authority, no maturity-number movement (§24).

## Fit with the roadmap
This is a concrete step inside **#80 (re-base the spatial stack for 3D/4D lidar — voxel ingest
+ camera-as-color)** and complements the self-sensing engine (#7). It is the "persistent place
vs transient signal" principle applied at the album layer.

## Validation outcome (2026-06-22) — verdict: BUILD-WITH-FIXES
A 4-lens multi-agent validation (signal-trace + architecture, HRR/P5 governance, anti-gaming /
anti-AI-theater, regression) against the live code + AGENTS.md. **All four lenses: sound_with_fixes.**
The problem is **trace-confirmed REAL, not theater**: `world_id = {session_id}:{tick}`
(spatial_episodic_store.py:175), `session_id` regenerates every **boot** (so the "~163 worlds" are
~163 reboots), the only dedup is **session-local** (`fp == self._last_fingerprint`), and there is
**zero cross-session place-matching anywhere**. The design stays honestly inside the zero-authority
shadow lane and claims only a measurable, auditable win. **Build only after pinning these (ordered):**

1. **FIRST COMMIT — close the structural-scan coverage gap.** `memory/spatial_episodic_store.py` is
   in **no** validation-pack scan root (`_HRR_MODULE_ROOTS` / `_P5_MENTAL_WORLD_ROOTS`,
   validation_pack.py:210-236) — so the album's stdlib-only isolation + zero-authority contract is
   currently **honor-system, not mechanically enforced.** Add the album (and any new consolidation
   module) to `_P5_MENTAL_WORLD_ROOTS` so `_scan_hrr_forbidden_imports` + `_scan_p5_mental_world_imports`
   enforce it. Valuable regardless of consolidation.
2. **Geometry source = anchors-only (correctness fix).** The design named the lidar room model, but
   `get_lidar_room_model()` (perception/server.py:162) is a *separate* accessor, not one of the three
   public canonical accessors. Fingerprint from **`SpatialAnchors`** (`position_room_m` + `dimensions_m`,
   perception/spatial.py:210-222) + canonical scene-state room dims via the three public accessors. If
   the lidar room model is needed, add it as an explicit, justified canonical read — never smuggled.
3. **Place-record data model (explicit).** `record_kind` discriminator (`"observation"` vs
   `"place_consolidated"`), `place_id`, `place_geometry_hash`, `calibration_version`. Mutable accumulator
   but **append-only-audited** (append updated place state; never rewrite history). Every record carries
   the same all-false `AUTHORITY_FLAGS`, `status:"PRE-MATURE"`, `lane:"spatial_hrr_mental_world"`.
4. **Place-match algorithm in `spatial_schema.py`** with thresholds **derived from existing constants**
   (`CLASS_MOVE_THRESHOLDS` / `DEFAULT_MOVE_THRESHOLD`, spatial_schema.py:46-59), never invented (P5-3).
   Deterministic geometry distance over canonical room-frame coords; **fail-closed to new-place** when
   canonical geometry is unavailable.
5. **HRR re-encode contract (pinned).** Place HRR re-encoded **in-process only, on demand** from the
   consolidated graph; **no vector ever touches disk or any API.** Add a truth-probe that walks the
   **persisted album records** (not just API) for `vector`/`raw_vector`/`composite_vector` and fails on
   any hit.
6. **Aggregate honesty.** A consolidated place record is a multi-session aggregate — flag it
   (`record_kind:"place_consolidated"` / `loaded_from_store`) so it is never mistaken for a live world.
7. **Self-sensing tie-in behind its OWN gate** (named, default-OFF, operator-approved per HRR-3). Ship
   consolidation + pass the truth-probe FIRST; wire the persistence-prior into `self_sensing.py` only after.
8. **Measure on LIVE data; add tests.** Do NOT assert 163→1 in advance — report the real ratio post-build.
   Add `brain/tests/test_spatial_place_consolidation.py` (same-geometry→same place_id, different→different,
   unavailable→new, records vector-free + authority-false + PRE-MATURE), a validation-pack
   `p5_place_consolidation_structure` check, and an AGENTS.md **P5-13** rule.

**Bottom line:** sound, real, honestly inside the shadow lane, and worth building — but pin the internals
(especially #1, the *unenforced* isolation) before writing a line of consolidation code.
