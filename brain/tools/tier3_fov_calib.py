"""Measure the TRUE camera focal/FOV from data (the flip's root cause).

The dense cloud anchors at a spurious +50° yaw instead of the ~+10° physical mount. The
hypothesis: the configured focal (470px ≈ 68° HFOV, an IMX708-narrow assumption) is wrong
for the actual lens; a wrong focal compresses/expands the column→bearing map, and the depth
"fixes" it by rotating to a bogus yaw. So search (focal × yaw) for where the camera depth
TRULY agrees with the lidar — at a physically-plausible yaw. If a clearly-better focal pops
out near the +10° mount, that's the lens, and that un-flips the cloud honestly.

Averages the correlation over a few frames so a single noisy frame can't decide it.
"""
import json, math, sys, urllib.request
import numpy as np
sys.path.insert(0, "/home/duafoo/duafoo/brain")
import cv2, torch
from transformers import pipeline
from PIL import Image

PX_FRAC = 0.5      # principal_x = frame_w * 0.5
PY = 240.0
N_FRAMES = 4

pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if torch.cuda.is_available() else -1)


def grab():
    raw = urllib.request.urlopen("http://192.168.1.248:8080/snapshot", timeout=8).read()
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    d = json.loads(urllib.request.urlopen("http://localhost:9200/api/pi5", timeout=8).read())
    prof = (d.get("lidar_room") or {}).get("pi-lidar", {}).get("profile") or []
    pred = pipe(Image.fromarray(rgb))["predicted_depth"]
    pred = np.asarray(pred.detach().float().cpu().numpy() if hasattr(pred, "detach") else pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]))
    return pred, prof, rgb.shape[1]


def corr_at(row, prof, focal, yaw, px, n_bins):
    bw = 2 * math.pi / n_bins
    ds, ts = [], []
    for x in range(len(row)):
        d = float(row[x])
        if d <= 0:
            continue
        th = math.atan2(x - px, focal)
        ct = math.cos(th)
        if ct <= 0.1:
            continue
        rng = prof[int(((th - yaw) % (2 * math.pi)) / bw) % n_bins]
        if not rng or rng <= 0:
            continue
        ds.append(d); ts.append(1.0 / (rng * ct))
    if len(ds) < 30:
        return None, 0
    ds, ts = np.array(ds), np.array(ts)
    if ds.std() < 1e-6 or ts.std() < 1e-6:
        return 0.0, len(ds)
    return float(np.corrcoef(ds, ts)[0, 1]), len(ds)


def main():
    frames = []
    for _ in range(N_FRAMES):
        try:
            frames.append(grab())
        except Exception as e:
            print("grab failed:", e)
    if not frames:
        print("no frames"); return
    w = frames[0][2]; px = w * PX_FRAC; n_bins = len(frames[0][1])
    print(f"frame_w={w} principal_x={px:.0f} n_bins={n_bins} frames={len(frames)}")
    print(f"configured focal=470 (HFOV {2*math.degrees(math.atan2(px,470)):.0f}°)\n")

    focals = list(range(150, 701, 10))
    yaws = [math.radians(y) for y in range(-35, 36, 2)]
    best = (-2, None, None)
    grid = {}
    for f in focals:
        bestrow = (-2, None)
        for yw in yaws:
            cs = []
            for pred, prof, _w in frames:
                # the anchor row shifts a little with focal (mount offset), but ~centre; use PY
                c, n = corr_at([float(v) for v in pred[int(PY)]], prof, f, yw, px, n_bins)
                if c is not None:
                    cs.append(c)
            if cs:
                mc = float(np.mean(cs))
                if mc > bestrow[0]:
                    bestrow = (mc, math.degrees(yw))
                if mc > best[0]:
                    best = (mc, f, math.degrees(yw))
        grid[f] = bestrow
    # report the per-focal best correlation + its yaw (the calibration landscape)
    print("focal  HFOV  best_corr  @yaw")
    for f in focals[::2]:
        c, y = grid[f]
        bar = "#" * int(max(0, c) * 30)
        print(f"{f:4d}  {2*math.degrees(math.atan2(px,f)):4.0f}°  {c:+.2f}  {y:+5.0f}°  {bar}")
    print(f"\nBEST: focal={best[1]} (HFOV {2*math.degrees(math.atan2(px,best[1])):.0f}°) "
          f"yaw={best[2]:+.0f}° corr={best[0]:+.2f}")
    print(f"(configured focal=470 → for comparison: {grid.get(470,(0,0))})")


if __name__ == "__main__":
    main()
