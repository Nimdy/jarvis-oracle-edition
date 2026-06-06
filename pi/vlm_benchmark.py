#!/usr/bin/env python3
"""Standalone Hailo-10H VLM feasibility benchmark (edge scene-captioning).

Answers the gating questions before wiring Qwen2-VL-2B into the senses pipeline:
  1. Does the VLM HEF LOAD on this Hailo-10H?  (capacity)
  2. How FAST is a scene caption?               (latency)
  3. Is the caption actually USEFUL?            (quality)

IMPORTANT — exclusive device: there is no hailort multi-process service on this Pi
(detector.py uses a plain in-process VDevice), so the running senses (main.py) hold
the Hailo EXCLUSIVELY. This benchmark needs the device to itself:

    # stop the senses first (your normal kill), then:
    cd ~/duafoo/pi && .venv/bin/python vlm_benchmark.py
    # then ./start.sh to bring the senses back

If main.py is still running, VDevice() will fail to acquire — the script reports
that clearly and exits without disturbing anything.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Use the STACK-INSTALLED HEF (compatible with the live hailort), NOT the older
# ~/hailo-genai copy (different build -> HAILO_INVALID_OPERATION on VLM create).
DEFAULT_HEF = "/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef"
DEFAULT_PROMPT = ("Describe what you see briefly. List any people, objects, or "
                  "activities. Be factual; if something is not present, do not mention it.")


def _capture_frame() -> "object | None":
    """Grab one real frame via rpicam-still (camera is free when the senses are stopped)."""
    import numpy as np
    out = Path("/tmp/vlm_bench_frame.jpg")
    try:
        subprocess.run(
            ["rpicam-still", "-n", "-t", "400", "--width", "1280", "--height", "720",
             "-o", str(out)],
            check=True, capture_output=True, timeout=20,
        )
        import cv2
        img = cv2.imread(str(out))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        print(f"[warn] camera capture failed ({exc}); falling back to a synthetic frame")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default=DEFAULT_HEF)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    import numpy as np
    try:
        from hailo_platform.genai import VLM, VDevice
    except Exception as exc:
        print(f"[FAIL] cannot import hailo_platform.genai VLM: {exc}")
        return 2

    if not Path(args.hef).exists():
        print(f"[FAIL] HEF not found: {args.hef}")
        return 2

    print(f"[1/4] acquiring Hailo VDevice (SHARED group — required by genai) ...")
    t0 = time.time()
    try:
        # Hailo genai models REQUIRE a VDevice created with the "SHARED" group_id
        # (the proven simple_llm_chat pattern). A bare VDevice() fails VLM creation
        # with HAILO_INVALID_OPERATION. The detection path uses the same SHARED group,
        # which is what lets detection + VLM coexist in one process.
        params = VDevice.create_params()
        params.group_id = "SHARED"
        vdevice = VDevice(params)
    except Exception as exc:
        print(f"[FAIL] could not acquire the Hailo device: {exc}")
        print("       -> the senses (main.py) are probably still running and holding it.")
        print("       -> stop main.py, then re-run this benchmark.")
        return 3
    print(f"      VDevice ok ({time.time()-t0:.1f}s)")

    vlm = None
    try:
        print(f"[2/4] loading VLM HEF (this is the capacity test) ...")
        t0 = time.time()
        vlm = VLM(vdevice, args.hef)
        load_s = time.time() - t0
        print(f"      LOADED ok in {load_s:.1f}s")
        try:
            print(f"      input_frame_shape={vlm.input_frame_shape} "
                  f"format_type={vlm.input_frame_format_type} order={vlm.input_frame_format_order}")
        except Exception as exc:
            print(f"      (frame-format introspection n/a: {exc})")

        print(f"[3/4] preparing a frame (336x336 RGB, per the working recipe) ...")
        frame = _capture_frame()  # already RGB, or None
        if frame is None:
            frame = (np.random.rand(336, 336, 3) * 255).astype(np.uint8)
            print(f"      using synthetic frame {frame.shape} (caption content not meaningful)")
        else:
            try:
                import cv2
                frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_LINEAR)
            except Exception as exc:
                print(f"      [warn] resize failed ({exc}); sending raw frame")
            print(f"      real camera frame -> {frame.shape}")
        frame = np.ascontiguousarray(frame.astype(np.uint8))

        # Structured chat prompt (mirrors simple_vlm_chat.py exactly).
        prompt = [
            {"role": "system",
             "content": [{"type": "text", "text": "You are a helpful assistant that describes scenes factually."}]},
            {"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]},
        ]

        print(f"[4/4] running {args.runs} generation(s) (max_tokens={args.max_tokens}) ...")
        for i in range(args.runs):
            try:
                vlm.clear_context()
            except Exception:
                pass
            t0 = time.time()
            try:
                out = vlm.generate_all(prompt=prompt, frames=[frame], temperature=0.1,
                                       seed=42, max_generated_tokens=args.max_tokens)
                text = out if isinstance(out, str) else "".join(list(out))
                text = text.split(". [{'type'")[0].split("<|im_end|>")[0]
            except Exception as exc:
                print(f"      [FAIL] generate_all() errored: {exc!r}")
                return 4
            dt = time.time() - t0
            tag = "warmup" if i == 0 else "steady"
            print(f"      run {i+1} ({tag}): {dt:.2f}s  -> {text.strip()[:280]!r}")

        print("\n[RESULT] VLM loads + captions on the Hailo-10H. "
              "Use the steady-run latency to judge the in-process integration.")
        return 0
    finally:
        try:
            if vlm is not None:
                vlm.release()
        except Exception:
            pass
        try:
            vdevice.release()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
