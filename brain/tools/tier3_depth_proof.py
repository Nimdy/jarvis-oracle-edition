"""Single-frame end-to-end PROOF for the Tier-3 dense-3D flagship (run ON THE BRAIN).

Grab one real frame from the Pi → run Depth-Anything-V2 on the GPU → anchor the relative
depth to metric using the LIVE lidar ring → lift to a colored cloud → VALIDATE the anchored
geometry against the lidar's measured walls. No streaming, no viz — just: does the chain
produce metrically-true geometry on the real room? Prints everything; writes a cloud sample.
"""
import io
import json
import math
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, "/home/duafoo/duafoo/brain")
from cognition.lidar_depth import anchor_depth_affine, depth_to_points, lidar_plane_row
from cognition.lidar_calibration import load_extrinsic

PI_SNAPSHOT = "http://192.168.1.248:8080/snapshot"
BRAIN_PI5 = "http://localhost:9200/api/pi5"
FOCAL, PX, PY = 470.0, 320.0, 240.0


def grab_frame():
    raw = urllib.request.urlopen(PI_SNAPSHOT, timeout=8).read()
    import cv2
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)   # BGR
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return rgb, len(raw)


def live_lidar_profile():
    d = json.loads(urllib.request.urlopen(BRAIN_PI5, timeout=8).read())
    room = (d.get("lidar_room") or {}).get("pi-lidar") or {}
    return room.get("profile") or [], room


def run_depth(rgb):
    """Depth-Anything-V2-Small on the GPU. Returns a relative depth/disparity map (HxW,
    larger = closer), resized to the frame resolution."""
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=dev)
    from PIL import Image
    out = pipe(Image.fromarray(rgb))
    pred = out["predicted_depth"]
    if hasattr(pred, "detach"):
        pred = pred.detach().float().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    # resize to frame size
    import cv2
    h, w = rgb.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred, dev


def main():
    t0 = time.time()
    rgb, nbytes = grab_frame()
    print(f"[1] frame: {rgb.shape} from Pi ({nbytes} bytes)")

    profile, room = live_lidar_profile()
    nlit = sum(1 for r in profile if r)
    print(f"[2] lidar: {len(profile)} bins, {nlit} with a return; "
          f"room {room.get('filter_stats',{}).get('room_width_m')}x"
          f"{room.get('filter_stats',{}).get('room_depth_m')}m")
    if nlit < 30:
        print("    !! too few lidar returns — anchor will refuse. Aborting proof."); return

    ex = load_extrinsic()
    print(f"[3] extrinsic: yaw={math.degrees(ex.yaw_rad):.1f}deg ty(lidar)={ex.ty_m}m "
          f"cam_h={getattr(ex,'camera_height_m',1.219)}m")

    td = time.time()
    pred, dev = run_depth(rgb)
    print(f"[4] depth: {pred.shape} range[{pred.min():.3f},{pred.max():.3f}] "
          f"(GPU={dev==0}, {time.time()-td:.2f}s)")

    cam_h = getattr(ex, "camera_height_m", 1.219)
    y_row = lidar_plane_row(cam_h, ex.ty_m, FOCAL, PY)
    row = [float(v) for v in pred[y_row]]                       # disparity along the scan row
    a = anchor_depth_affine(row, profile, yaw_rad=ex.yaw_rad, focal_px=FOCAL,
                            principal_x=PX, min_inliers=15)
    print(f"[5] ANCHOR (row {y_row}): valid={a.valid} scale={a.scale:.5f} shift={a.shift:.5f} "
          f"inliers={a.inlier_count} rms={a.rms:.5f} reason={a.reason}")
    if not a.valid:
        print("    anchor refused — depth/lidar overlap too thin this frame. Try again or "
              "widen the row band."); return

    pts = depth_to_points(pred.tolist(), rgb.tolist(), a, focal_px=FOCAL, principal_x=PX,
                          principal_y=PY, camera_height_m=cam_h, yaw_rad=ex.yaw_rad,
                          stride=8, max_points=8000)
    print(f"[6] cloud: {len(pts)} points (stride 8)")

    # VALIDATION: at the anchor row, the cloud's radial range must agree with the lidar
    bin_w = 2 * math.pi / len(profile)
    errs = []
    for x in range(0, len(row), 9):
        d = row[x]
        if d <= 0:
            continue
        inv = a.scale * d + a.shift
        if inv <= 1e-3:
            continue
        z = 1.0 / inv
        theta = math.atan2(x - PX, FOCAL)
        radial = z / max(0.1, math.cos(theta))
        lr = profile[int(((theta - ex.yaw_rad) % (2*math.pi)) / bin_w) % len(profile)]
        if lr:
            errs.append(abs(radial - lr))
    if errs:
        errs.sort()
        print(f"[7] VALIDATION vs lidar walls @ scan row: median |err|={errs[len(errs)//2]*100:.1f}cm "
              f"max={max(errs)*100:.1f}cm over {len(errs)} bearings")
    # geometry sanity: nearest vs farthest point depth
    zs = sorted(math.hypot(p[0], p[2]) for p in pts)
    if zs:
        print(f"    cloud radial range: nearest {zs[0]:.2f}m  median {zs[len(zs)//2]:.2f}m  "
              f"farthest {zs[-1]:.2f}m")
    # dump a small sample for eyeballing
    with open("/tmp/tier3_cloud_sample.json", "w") as f:
        json.dump({"n": len(pts), "scale": a.scale, "shift": a.shift, "inliers": a.inlier_count,
                   "sample": pts[:400]}, f)
    print(f"[8] wrote /tmp/tier3_cloud_sample.json | total {time.time()-t0:.1f}s")
    print("PROOF: " + ("PASS — metric cloud from real frame + lidar anchor"
                       if errs and errs[len(errs)//2] < 0.25 else
                       "PARTIAL — chain ran; check anchor error above"))


if __name__ == "__main__":
    main()
