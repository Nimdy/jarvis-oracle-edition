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
from cognition.lidar_depth import anchor_depth_affine, depth_to_points, lidar_plane_row
from cognition.lidar_calibration import load_extrinsic

PI_SNAPSHOT = "http://192.168.1.248:8080/snapshot"
BRAIN_PI5 = "http://localhost:9200/api/pi5"
SLOT = os.path.expanduser("~/.jarvis/dense_points.json")
FOCAL, PX, PY = 470.0, 320.0, 240.0
INTERVAL_S = 4.0
STRIDE = 7
MAX_POINTS = 9000


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
                write_slot({"ts": time.time(), "valid": False, "reason": "lidar_sparse", "points": []})
                time.sleep(INTERVAL_S); continue
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
            y_row = lidar_plane_row(cam_h, ex.ty_m, FOCAL, PY)
            row = [float(v) for v in pred[y_row]]
            a = anchor_depth_affine(row, profile, yaw_rad=ex.yaw_rad, focal_px=FOCAL,
                                    principal_x=PX, min_inliers=15)
            if not a.valid:
                write_slot({"ts": time.time(), "valid": False, "reason": a.reason, "points": []})
                log(f"#{n} anchor refused ({a.reason})"); time.sleep(INTERVAL_S); continue

            pts = depth_to_points(pred.tolist(), rgb.tolist(), a, focal_px=FOCAL, principal_x=PX,
                                  principal_y=PY, camera_height_m=cam_h, yaw_rad=ex.yaw_rad,
                                  stride=STRIDE, max_points=MAX_POINTS)
            write_slot({
                "ts": time.time(), "valid": True, "points": pts, "n": len(pts),
                "scale": round(a.scale, 6), "shift": round(a.shift, 6),
                "inliers": a.inlier_count, "rms": round(a.rms, 5),
                "frame_wh": [w, h], "stride": STRIDE,
                "authority": "spatial_telemetry_only", "writes_beliefs": False,
                "provenance": "camera_depth_GUESS · scale_lidar_MEASURED",
            })
            n += 1
            log(f"#{n} cloud {len(pts)}pts scale={a.scale:.4f} inliers={a.inlier_count} "
                f"rms={a.rms:.4f} ({time.time()-t0:.2f}s)")
        except Exception as e:        # never die — the dashboard just shows no cloud
            log(f"loop error: {type(e).__name__}: {e}")
            try:
                write_slot({"ts": time.time(), "valid": False, "reason": "service_error", "points": []})
            except Exception:
                pass
        dt = time.time() - t0
        if dt < INTERVAL_S:
            time.sleep(INTERVAL_S - dt)


if __name__ == "__main__":
    main()
