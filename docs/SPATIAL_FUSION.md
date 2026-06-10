# Spatial Fusion — LiDAR + Camera (the "where" meets the "what")

> Status legend: ✅ shipped & live · 🟡 designed / partial · ⬜ planned · 🔒 gated (earned, not coded)
> GitHub: epic **#21** (the 2D LiDAR sense), growth **#19** (Spatial Cognition S1→S5).

## 1. The thesis (why this is not a sensor demo)

Environmental awareness — JARVIS knowing *where it is* — is a stated prerequisite of the
synthetic-soul architecture (the Sensory Abstraction Module; `docs/SyntheticSoul.md`). It is
**not** a claim of consciousness; it is a foundation one is allowed to build.

The split is honest and load-bearing, and it is just **tri-layer cognition applied to perception**:

| Sensor | Supplies | Epistemic status |
|---|---|---|
| **LiDAR (S2)** | exact metric **geometry** — walls, range, position (±30 mm) | **truth** (it measured it) |
| **Camera (Hailo/YOLO)** | the **label** — "chair", "tv" | **hypothesis** (a guess about *what*) |
| **Fusion** | label *attached to* geometry | **telemetry-only / shadow** until verified |

A fused object is "the camera's guess at *what* + the lidar's measurement of *where*". The label is
**never asserted as fact** here — it stays a hypothesis until a verification step (camera re-confirm +
grounding) earns it. This is the same firewall as everywhere else in JARVIS: `writes_beliefs=false`,
`authority=spatial_telemetry_only`, lived-before-synthetic, earn-don't-declare.

## 2. The pipeline (dev → brain → Pi)

Built on the dev machine, deployed via `./sync-desktop.sh` (brain, `duafoo@192.168.1.222`) and
`./sync-pi.sh` (Pi, `nimda@192.168.1.248`); the running system is verified by SSH/curl, never assumed.

```
Pi (EYES, thin)                         Brain (ALL processing)
─────────────────                       ───────────────────────
S2 @ 10Hz ─ lidar_node.py ── scan_2d ──▶ perception/server.py  ─┐
  points_polar=[[bearing,range]]          → LidarRoomModel       │   /api/pi5
                                            (median+MAD+walls)    ├─▶  lidar / lidar_room
camera ─ Hailo YOLO ─ scene_summary ──▶  perception_orchestrator │     fused_objects
  bbox + label + conf                      get_scene_snapshot()   │   /v2/spatial-core
                                                                  │     (top-down radar +
dashboard/snapshot.py  _build_fused_objects(ctx):                 │      instrument strip +
  fuse(camera entities, lidar profile, extrinsic, intrinsics) ────┘      object overlay)
```

- **Pi stays thin** (David's law): capture + stream raw points; ALL geometry/fusion is brain-side.
- The S2's specifics (pyrplidar, 1 Mbaud, STOP+RESET recovery, DenseBoost) are isolated in
  `pi/senses/lidar/lidar_sensor.py` — the brain consumes a **generic** `scan_2d` / `points_polar`
  contract, so a different lidar is a drop-in Pi adapter (✅ brand-agnostic, test-pinned).

## 3. Frames & the extrinsic

- Room frame: **X = right, Y = up, Z = forward**, bearing 0 = +Z (the camera's optical axis).
  *Left-handed / Unity-style* — a camera-fusion frame must match this (not OpenGL Z-back).
- The S2 is a single horizontal slice at its **mount height**. `cognition/lidar_calibration.py`
  (`LidarExtrinsic`) maps lidar→room: a yaw about +Y plus a translation (ty = scan-plane height).
- **Per-rig config** `~/.jarvis/lidar_extrinsic.json` (file → env → identity). A user with no/uncalibrated
  lidar defaults to identity (no-op). ✅ David's rig: `ty_m=1.092` (43″ off floor), lidar 4.1 cm above the
  co-axial camera; `yaw_rad` seeded **+10°** from the live "person centred → lidar ~350°" reading.
- `room_bearing = lidar_bearing + yaw` ⇒ to look a camera bearing up in the lidar profile, use
  `lidar_bearing = room_bearing − yaw`.

## 4. The fusion contract (`cognition/lidar_fusion.py`)

`fuse(camera_entities, lidar_profile, *, yaw, focal_px, principal_x, mount_height)` → `FusedObject[]`:

```
label, label_confidence        # CAMERA — a hypothesis (label_provenance=camera_hypothesis)
bearing_rad                    # room/camera-frame bearing (from bbox: atan2(cx-principal_x, focal))
lidar_bearing_rad              # = room − yaw (where the profile was sampled)
range_m                        # LIDAR metric range (None ⇒ camera-only, honestly unplaced)
position_room_m / _lidar_m     # (x, y=mount, z) in each frame (lidar-frame sits on the return)
has_lidar_range                # False ⇒ no metric confirmation
authority=spatial_telemetry_only · writes_beliefs=false
```

Pure stdlib, no live-brain deps, registered in `_HRR_MODULE_ROOTS` (forbidden-import scan) so it
structurally cannot write beliefs/memory/policy. `YawEstimator` self-calibrates yaw from co-observed
**moving** objects and **refuses** to declare a yaw without angular spread (earns it).

## 5. The Tier ladder

| Tier | Capability | Status |
|---|---|---|
| 0 | keep the per-point **intensity/reflectivity** channel (currently discarded) | ⬜ |
| 1 | **people-radar** — track a moving person via lidar change vs an empty-room baseline | ⬜ |
| **2** | **object ranging** — camera label + lidar metric range, placed on the radar | ✅ live (#72/#73/#74/#75) |
| 3 | 🚩 **flagship** — monocular depth (Depth-Anything) **anchored by the lidar 360° ring** → dense, metrically-true, colored 3D room | 🟡 **brain-side, in build** (see §8) |
| 4 | **4D spatial memory** — persistent, labeled, time-aware room JARVIS *remembers* (feeds the scene graph) | 🔒 |

## 6. What's shipped vs gated

**✅ Shipped & live:** room model + 0.12/8.0 range gating + drop telemetry; top-down (north-up, no-rotation)
& 3D views + instrument strip; lidar-optional guarantee (test-pinned); per-rig extrinsic; geometry
hardening (seam wall/opening, range self-gate, valid_fraction, frame trap — PR #71); **Tier-2 fusion**
(core #72, wiring #73, overlay #74, readability #75). Door-closed validation **passed** (±1 cm).

**🔒 Gated / next:**
- **Yaw self-calibration** — `YawEstimator` built; wire it to capture a walk-through (moving person =
  clean signal) and *suggest* a refined yaw (advisory, never auto-overwrites the config).
- **Validation walk-through** (David's gate): door-open + person-walking. Closes the gate **and** feeds the
  yaw cal. Required before any *live meaning-attach*.
- **Mental-world integration (R4)** — the `derive_scene_graph` lidar-anchor-only path so the room
  populates JARVIS's scene graph even camera-dark. Gated on the walk-through; test #69 enforces the
  camera-only path stays byte-identical.
- **Tier-3 flagship** — the dense 3D reconstruction.

## 7. Tier-3 flagship — the brain-side plan (dense colored 3D)

Decided by a 5-agent recon (2026-06-10). The metric TRUTH comes from the lidar anchor, not
the depth net — so the depth model is the cheap, swappable part and **the anchoring solve is
the real IP**. The brain is an **RTX 4080 + Ryzen 9 7950X** with torch / onnxruntime-gpu /
opencv / transformers already present (Depth-Anything-V2-Small ONNX runs ~10–15 FPS there),
so the loop lives **brain-side**; the Pi stays thin (capture + JPEG only). The Hailo-10H
(confirmed via `hailortcli`) has a precompiled `depth_anything_v2_vits` HEF — a **deferred,
optional** on-device latency optimization, *not* the path to first light.

Pipeline: Pi samples RGB keyframes → brain runs Depth-Anything (relative disparity) →
**`lidar_depth.anchor_depth_affine` rescales it to metric using the lidar scan row** →
`depth_to_points` lifts to a dense colored cloud (holes kept as holes) → `dense_points` on
`/api/pi5` → rendered in `spatial-core.html`'s 3D view.

| Phase | What | Status |
|---|---|---|
| **0** | `cognition/lidar_depth.py` — the anchoring core (affine scale+shift solve + point lift), refuses to fabricate, holes stay holes; 8 synthetic-depth tests | ✅ shipped (PR #78) |
| 1 | Pi→brain keyframe transport — opt-in `vision_frame_sample` event (low-cadence, default OFF), brain single-slot receiver + drop telemetry | ⬜ next (the true blocker) |
| 2 | brain-side Depth-Anything ONNX inference (lazy, off the kernel tick) + anchor against the live lidar; shadow telemetry (scale/inlier stability vs the wall ground-truth) | ⬜ gated on 1 + yaw |
| 3 | `dense_points` → cache → `spatial-core` 3D render (decimated ≤10k, depth-sorted), honest "depth GUESS · scale lidar-MEASURED" banner | ⬜ gated on 1+2 |
| 4 | **optional** on-device Hailo-10H depth (precompiled HEF, time-sliced) — re-anchored by the same solve; promote only if it measurably wins | 🔒 gated, optional |

Honest status: **no real dense-3D *render* exists yet** — Phases 1–3 are the path. Phase 0
(the truth-making math) is real, tested, and shippable today with zero frames/models.

## 8. Honesty guardrails (non-negotiable)

- Labels are **hypotheses** until verified; the overlay says so (`labels are camera GUESSES · ranges
  lidar-MEASURED`). A camera misdetection ("suitcase" = the tower) is shown *as* a hypothesis, not a fact.
- Walls are extruded to a synthetic height for the 3D view — the sensor reads **one horizontal plane**;
  the UI states this. The 3D is honest *extrusion*, not raw 3D truth.
- No fabricated geometry. `stability_30s` warms up before it can red-flag. `effective_max_m` is the
  config cap, never the self-gating observed max.
