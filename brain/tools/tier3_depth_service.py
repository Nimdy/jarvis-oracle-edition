"""Tier-3 dense-3D depth SIDECAR (runs on the brain, separate process).

A safe sidecar — it does NOT touch the brain's cognition core. On a slow loop it:
grabs one RGB frame from the Pi → runs Depth-Anything-V2 on the GPU → anchors the relative
depth to METRIC using the live lidar ring (cognition.lidar_depth) → lifts to a dense colored
cloud (lidar frame) → atomically writes a slot file the dashboard reads. Telemetry-only:
writes geometry, never beliefs. Honest — only emits a cloud when the anchor is VALID; holes
stay holes. If anything fails it logs and keeps looping (the dashboard just shows no cloud).

Start:  nohup ~/duafoo/brain/.venv/bin/python ~/duafoo/brain/tools/tier3_depth_service.py \
          > /tmp/tier3_depth.log 2>&1 &
Stop:   pkill -f tier3_depth_service
"""
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, "/home/duafoo/duafoo/brain")
from cognition.lidar_depth import (anchor_depth_affine, anchor_best_yaw, depth_to_points,
                                    lidar_plane_row)
from cognition.lidar_calibration import load_extrinsic

PI_SNAPSHOT = "http://192.168.1.248:8080/snapshot"
BRAIN_PI5 = "http://localhost:9200/api/pi5"
SLOT = os.path.expanduser("~/.jarvis/dense_points.json")
CALIB = os.path.expanduser("~/.jarvis/camera_calib.json")   # live manual-calibration slider state
PX, PY = 320.0, 240.0
INTERVAL_S = 4.0
STRIDE = 5            # denser sample → fills the lattice gaps (~12k pts at 640x480)
MAX_POINTS = 14000


def load_calib(default_yaw_deg):
    """Live camera calibration (driven by the dashboard slider). manual=True ⇒ anchor at the
    EXACT focal+yaw the operator dialled (no search) so the cloud responds to the slider."""
    try:
        with open(CALIB) as f:
            c = json.load(f)
        return (float(c.get("focal_px", 640.0)), float(c.get("yaw_deg", default_yaw_deg)),
                bool(c.get("manual", False)), int(c.get("pitch_row_offset", 0)))
    except Exception:
        return 640.0, default_yaw_deg, False, 0


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def grab_frame():
    import cv2
    raw = urllib.request.urlopen(PI_SNAPSHOT, timeout=6).read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def live_lidar_profile():
    d = json.loads(urllib.request.urlopen(BRAIN_PI5, timeout=6).read())
    room = (d.get("lidar_room") or {}).get("pi-lidar") or {}
    return room.get("profile") or [], room


def write_slot(payload):
    tmp = SLOT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, SLOT)            # atomic


def main():
    os.makedirs(os.path.dirname(SLOT), exist_ok=True)
    import torch
    from transformers import pipeline
    from PIL import Image
    dev = 0 if torch.cuda.is_available() else -1
    log(f"loading Depth-Anything-V2-Small (GPU={dev == 0}) ...")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    log("model ready. entering loop.")
    import cv2

    n = 0
    while True:
        t0 = time.time()
        try:
            profile, room = live_lidar_profile()
            if sum(1 for r in profile if r) < 30:
                log("lidar sparse — keeping last good"); time.sleep(INTERVAL_S); continue
            rgb = grab_frame()
            pred = pipe(Image.fromarray(rgb))["predicted_depth"]
            pred = np.asarray(pred.detach().float().cpu().numpy() if hasattr(pred, "detach")
                              else pred, dtype=np.float32)
            if pred.ndim == 3:
                pred = pred[0]
            h, w = rgb.shape[:2]
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

            ex = load_extrinsic()
            cam_h = getattr(ex, "camera_height_m", 1.219)
            focal, cyaw_deg, manual, pitch_off = load_calib(math.degrees(ex.yaw_rad))
            y_row = max(0, min(479, lidar_plane_row(cam_h, ex.ty_m, focal, PY) + pitch_off))
            row = [float(v) for v in pred[y_row]]
            if manual:
                # MANUAL calibration: anchor at the EXACT focal+yaw the operator dialled (no
                # search) so the cloud responds live to the slider. Only the physical (scale>0)
                # gate holds — the operator tunes to maximise corr + visual wall-alignment.
                a = anchor_depth_affine(row, profile, yaw_rad=math.radians(cyaw_deg), focal_px=focal,
                                        principal_x=PX, min_inliers=15, min_corr=0.0)
                yaw_used = math.radians(cyaw_deg)
            else:
                # auto: search a PHYSICALLY-plausible yaw band; refuse spurious far-off fits.
                a, yaw_used = anchor_best_yaw(row, profile, base_yaw_rad=ex.yaw_rad, focal_px=focal,
                                              principal_x=PX, search_deg=15.0, step_deg=2.0,
                                              min_inliers=15, min_corr=0.4)
            if not a.valid:
                log(f"#{n} anchor refused ({a.reason}, corr={a.corr:+.2f}{', manual' if manual else ''}) — keeping last good")
                time.sleep(INTERVAL_S); continue

            pts = depth_to_points(pred.tolist(), rgb.tolist(), a, focal_px=focal, principal_x=PX,
                                  principal_y=PY, camera_height_m=cam_h, yaw_rad=yaw_used,
                                  stride=STRIDE, max_points=MAX_POINTS)
            write_slot({
                "ts": time.time(), "valid": True, "points": pts, "n": len(pts),
                "scale": round(a.scale, 6), "shift": round(a.shift, 6),
                "inliers": a.inlier_count, "rms": round(a.rms, 5), "corr": round(a.corr, 3),
                "yaw_used_deg": round(math.degrees(yaw_used), 1), "focal_px": round(focal, 1),
                "mode": "manual" if manual else "auto",
                "frame_wh": [w, h], "stride": STRIDE,
                "authority": "spatial_telemetry_only", "writes_beliefs": False,
                "provenance": "camera_depth_GUESS · scale_lidar_MEASURED",
            })
            n += 1
            log(f"#{n} cloud {len(pts)}pts {'MANUAL' if manual else 'auto'} focal={focal:.0f} "
                f"scale={a.scale:.4f} corr={a.corr:+.2f} yaw={math.degrees(yaw_used):+.0f}° ({time.time()-t0:.2f}s)")
        except Exception as e:        # never die — keep the last good cloud; it goes stale at 20s
            log(f"loop error: {type(e).__name__}: {e} — keeping last good")
        dt = time.time() - t0
        if dt < INTERVAL_S:
            time.sleep(INTERVAL_S - dt)


if __name__ == "__main__":
    main()
