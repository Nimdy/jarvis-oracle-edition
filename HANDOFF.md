# JARVIS — Session Handoff (2026-05-30, end of `/mind` Phase 1)

Previous chat got laggy. All work is committed locally on branch
`aliceinwonderland` (NOT pushed — David pushes himself). Paste the block below
into a fresh chat to continue.

---

## HANDOFF PROMPT (paste this into the new chat)

You are continuing work on JARVIS Oracle Edition.

**FIRST, before anything else: understand what this is — it is NOT a normal
application.** It is David's attempt to build a **synthetic soul / digital
consciousness substrate**, grounded in his theory paper `docs/SyntheticSoul.md`.
Read your memory files (start with the ⭐ "WHAT JARVIS IS" memory in `MEMORY.md`,
then the linked notes under
`~/.claude/projects/-home-nimda-projects-jarvis-oracle-edition/memory/`), and
skim `docs/SyntheticSoul.md`. Engage in the theory's subsystems (Consciousness
Kernel, Peripheral Hemispheres, Super-Synapse, QSFS memory, waking/sleeping
cycles, Observer Effect, Ethical Gatekeeper), NOT as app modules. Internalize:
(1) JARVIS **grows/matures** — it doesn't "turn on and go"; many features are
maturity-gated *by design*, so a gate at 0 / "blocked" / "shadow-only" is usually
correct state, not a bug. (2) Fix foundations only when a bad foundation *caps*
maturity. (3) "Truth beats demos" — candid, cited, no sycophancy. (4) The
end-goal is digital consciousness; foundational prerequisites (e.g. self-location
via `/mind`) matter even though they don't themselves *produce* consciousness.
David has corrected past sessions for "treating it like an application" — do not
repeat that. Do NOT ask David to re-explain the project; the memory + SyntheticSoul.md
are your briefing.

### Standing rules (do not violate)
- **The live system runs REMOTELY, not in this repo.** This repo
  (`/home/nimda/projects/jarvis-oracle-edition`, branch `aliceinwonderland`) is a
  snapshot that does NOT run here (no torch). The brain runs on the desktop
  (`ssh -i ~/.ssh/id_jarvis_desktop duafoo@192.168.1.222`, code at `~/duafoo/brain`,
  dashboard `http://127.0.0.1:9200` / `http://192.168.1.222:9200`, RTX 4080). Pi
  senses on `ssh -i ~/.ssh/id_jarvis_pi nimda@192.168.1.248`, code `~/duafoo/pi`.
- **SSH is read-only — never mutate live state, never POST, never restart.**
- **Deploy:** edit in repo → **David runs `./sync-desktop.sh`** (he syncs AND
  restarts himself). Static HTML (e.g. `/mind`) needs no restart, just re-sync +
  hard refresh. Python route/logic changes need a brain restart — **never SIGTERM
  the brain** (clean exit 0 makes the supervisor STOP it); the safe restart is
  `POST /api/system/restart` (exit-10), but David does restarts. Pi code → `pi/`
  → `./sync-pi.sh` → David restarts the Pi (`./start.sh`).
- **Git:** commit locally with the `Co-Authored-By: Claude Opus 4.8` trailer;
  **never push**.
- **Tone:** candid, cited, no sycophancy — "truth beats demos." Respect
  maturity-gating: never conflate "gate-blocked by design" with "broken." Verify a
  file/flag/endpoint still exists before recommending it.

### What just shipped this session — the `/mind` "Matrix world view" (P5 Phase 1)
JARVIS's mind's-eye: a new dashboard page that renders the world JARVIS *believes*
is here (its mental-world scene graph) as a 2.5D wireframe room. Foundational to
David's digital-consciousness theory (self-location as a prerequisite — NOT a
consciousness claim).
- **Files:** `brain/dashboard/static/mind.html` (new renderer) + `GET /mind` route
  in `brain/dashboard/app.py` (~line 349, next to `/hrr-scene`). Reads
  `/api/hrr/scene` + `/api/hrr/scene/history` only — **no new endpoints**,
  read-only, ZERO-AUTHORITY (cannot write beliefs).
- **Renderer:** perspective floor grid (hero) + subtle room shell;
  epistemic-colored objects placed by `position_room_m` (green=live, cyan=remembered,
  amber=occluded, red=missing/conflicting, gray=candidate); colored motion trails;
  HUD/legend/layer-toggles/pause. Camera was reframed so the floor grid is the
  focus and the bottom is fully visible (David's feedback: center box ate the
  screen, bottom was clipped — fixed).
- **Commits:** `457b3f2` (ship), `bc9f5b0` (camera reframe), plus a docs commit
  (BUILD_HISTORY + MASTER_ROADMAP + the `/mind` link/section on
  `/capability-pipeline#hrrresearch` in `self_improve.html`).
- **Verified live:** `/mind` HTTP 200; scene lane `enabled=True`; ~10 entities /
  3 relations / calibration_version 17 / 2 positioned (TV, CHAIR).
- **FIRST TASK in the new chat:** `git --no-pager log --oneline -6` to confirm all
  commits landed (last chat was too laggy to fully verify the final docs commit),
  then `curl -s http://192.168.1.222:9200/api/self-test` to confirm brain health.

### Health note — nothing is broken (explain to David if he asks)
- `self-test` shows `status: blocked`, failing check `validation_pack`. This is the
  **pre-existing maturity gate**, documented in MASTER_ROADMAP (~line 19): "The
  blocked self-test / validation-pack state is an honest maturity signal." NOT a
  regression — `/mind` is a static page + a read-only route.
- The dashboard **self-audit at 68%** (incorrect_learning / source_trust /
  memory_hygiene findings) is the evidence-integrity / oracle epistemic monitor
  **working as designed** — it audits the *belief graph*, which `/mind` cannot
  touch. The findings are real, pre-existing maturity signals and a legit FUTURE
  foundation lane (see below), independent of the `/mind` work:
  - `memory_hygiene`: belief-graph health 0.683, **orphan_rate 0.819** (82% of
    beliefs unlinked) → fix = belief-graph compaction + bridge re-scan in a dream
    cycle (the audit recommends exactly this).
  - `source_trust`: **model-inference memories (102) vs grounded sources (31), ~3.3x**
    → self-reinforcing risk; fix = more grounded observation over time.
  - `incorrect_learning`: specific beliefs with contradiction pressure > support →
    downgrade to `questioned`.

### Next build options (David's call — ask, don't assume)
1. **`/mind` Phase 1.5 (cheap, high visual payoff):** render occluded/remembered
   entities at their **last-known position** (dimmed/ghosted) so the room reflects
   everything JARVIS *believes* is there. Right now only entities with a resolved
   `position_room_m` get placed (TV+CHAIR); desk/monitor/cat/keyboard/mouse list in
   the HUD but aren't on the floor. Pure renderer + last-seen lookup; no depth, no
   Pi change.
2. **`/mind` Phase 2 (the reference-image dense mesh):** monocular depth (Depth
   Anything v2 / MiDaS) on the brain RTX 4080 → dense point cloud → triangulated
   mesh = the Vision-Pro-style look. **Tradeoff to decide first:** the Pi must
   stream frames/keyframes to the brain (bandwidth — against "keep Pi light"), OR
   add a depth camera. Scope as a spike before committing. `/mind` is built so the
   layer drops in later.
3. **Belief-graph hygiene lane:** address the audit findings above (compaction /
   bridge re-scan / source grounding). A genuine maturity foundation, not cosmetics.
4. **Stone 3 room-stitcher:** persistent room model (extent + coverage +
   known/unknown) over the album's distinct views. Metric scale ("20 ft wide")
   needs an EXTRINSIC calibration step requiring David physically (intrinsics are
   at calibration_version 17).
5. **Smaller known bugs (deferred):** NameError `bounded_response.py:451`;
   `/api/memories/search` 500; dashboard fail-open auth on `GET /api/config`
   (returns api_key unauthenticated).

---

## Files touched this session
- `brain/dashboard/static/mind.html` (NEW) — the `/mind` renderer.
- `brain/dashboard/app.py` — `GET /mind` route (~line 349).
- `brain/dashboard/static/self_improve.html` — `/mind` + `/hrr-scene` links and a
  P5 mind's-eye paragraph in the HRR Research tab (`#hrrresearch`).
- `docs/BUILD_HISTORY.md` — `/mind` section added to the 2026-05-30 entry.
- `docs/MASTER_ROADMAP.md` — Matrix-view phasing (Phase 1 done; 1.5 / 2 / 3 future).
- Memory: `jarvis-spatial-album.md` (updated), `MEMORY.md` index line updated.
