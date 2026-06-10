# Spatial Cognition — Backlog

> Companion to `docs/SPATIAL_FUSION.md` (the architecture). This is the **what's-next list** —
> what's shipped, what's deferred, and *when* each item makes sense. GitHub epic **#21**,
> growth **#19** (Spatial Cognition S1→S5).
> Status legend: ✅ shipped & live · ⬜ open · 🔒 gated (earned, not coded) · ⏭️ superseded.

## ✅ Shipped (all merged to `main`, all live)
- Room model + truth-layer hardening (0.12/8.0 range gate, drop telemetry, geometry fixes PR #71)
- Top-down (north-up, no-rotation) + 3D hologram views, instrument strip, lidar-optional guarantee
- Per-rig extrinsic (`~/.jarvis/lidar_extrinsic.json`)
- **Tier-2 object fusion** (camera label + lidar metric range) **with stable track IDs** (surfaces the
  existing scene-tracker; flickers filtered; de-duped by track_id)
- **Tier-3 dense colored 3D** (Depth-Anything on the brain GPU, lidar-anchored to metric). The flip's
  root cause — Depth-Anything outputs DEPTH not disparity — is fixed; cloud anchors at the physical yaw
- Camera↔lidar **manual calibration slider** (FOV/yaw/pitch, live anchor-fit readout) + **reset**
- **Liveness-truth** fix (camera/Hailo read alive when alive; scene activity kept separate)
- **Tier-0 reflectivity** — the S2 `quality` byte (was read + discarded) now streams per bin →
  per-bin median intensity → a live **material map** (matte/glossy/glass read differently) + **◉ Matrix
  mode** (green code-rain glow driven by real reflectivity)
- Honesty firewall throughout: `authority=spatial_telemetry_only`, `writes_beliefs=false`
- Validation trace confirmed the HRR/scene-graph cognition pipeline **already exists**
  (`engine.py:844-860` → `derive_scene_graph` → `encode_scene_graph` → `HRRSpatialShadow`) — not rebuilt

## 🎯 When the 3D/4D lidar lands — the big reframe (planned hardware swap)
A real 3D lidar streams a full point cloud, not a single horizontal plane. This **re-bases the stack**:
- ⬜ **3D ingest + voxel/octree room model** — the current `LidarRoomModel` is 2D-polar (one plane);
  a 3D lidar needs a volumetric accumulator. New module.
- ⬜ **Re-base the fusion** — once the lidar gives dense 3D directly, the camera flips role: from
  *anchoring* monocular depth → to *coloring/labeling* the lidar's real 3D geometry. The Tier-0
  reflectivity + Tier-2 labels both ride along.
- The thin-Pi `scan_2d`/`points_polar` contract + the **lidar-optional guarantee** (test #69) mean the
  swap is a Pi adapter + this brain-side 3D work — not a teardown.
- ⏭️ Tilt-sweep 3D-from-2D and DenseBoost 32 kHz — **superseded** by the hardware swap.

## 🔧 Spatial extensions (optional, anytime)
- ⬜ Confidence-weight wall extraction by reflectivity (strong return = trustworthy wall)
- ⬜ **Tier-1 people-radar** — track a moving person via lidar change vs an empty-room baseline
  (also feeds the yaw self-cal a clean signal)
- ⬜ GET_HEALTH proactive Protection-Stop detection (low value; we already recover reactively)
- ⬜ Reflectivity into the people-radar (clothing/skin reflect differently than walls)

## 🔒 Gated / earned spatial cognition (later, deliberately)
- 🔒 Feed lidar/depth into the **existing scene-graph → HRR** (`engine.py:844`) — "JARVIS reasons about
  the room," not just renders it. Gated on validation; test #69 keeps the camera-only path byte-identical.
- 🔒 **Stability-gating tiers** (±30 cm live-viz / ±15 temp / ±8 skeleton / ±5 persistent-memory) before
  any of this becomes *canonical room memory*. (Skeleton already hits ±0.6 cm when warm.)
- 🔒 **Tier-4 4D spatial memory** — persistent, labeled, time-aware room JARVIS *remembers*
  (feeds `spatial_episodic_store`, then the scene graph). Earned, not coded.
- ⬜ Yaw self-calibrator firms up passively as a person moves over time (no action; advisory).

## 🌅 Beyond spatial (if stepping off this thread)
The consciousness thesis moves more here than another lidar feature would:
- **Conversational soul** (#42) · **QSFS fractal-link memory substrate** (#10, flagged load-bearing) ·
  **fidelity keystone scoreboard** (#9) · companion cognition (#2) · spark/grounding ring (#4)

---
*Operational note: depth sidecar + Pi lidar node run in `screen` sessions (survive a brain restart; a
full Pi/OS reboot relaunches via `start.sh`). Restart cmds in `lidar-room-model` memory + `SPATIAL_FUSION.md`.*
