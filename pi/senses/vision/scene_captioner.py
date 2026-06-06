"""Edge scene captioner — Qwen2-VL-2B on the Hailo-10H (Companion edge-VLM).

Runs the vision-language model IN-PROCESS, sharing the detector's VDevice (the same
way expression/pose do), so the Pi produces its own compact scene description instead
of shipping a camera frame to the desktop GPU. This keeps perception local (good for a
mobile Pi) and frees desktop VRAM.

Reality the design respects (measured on this hardware):
  - The HEF takes ~76s to load -> loaded LAZILY in a background thread; never blocks
    startup or the senses loop.
  - A caption is ~8.8s of Hailo time and the model time-slices ONE device with 15fps
    person detection (ROUND_ROBIN) -> captioning is kept LOW-cadence + terse, and the
    caller measures detection fps during a caption and auto-disables on degradation.
  - The genai VLM requires a VDevice created with group_id="SHARED" (a bare VDevice
    fails with HAILO_INVALID_OPERATION); the detector now creates its shared VDevice
    that way when edge captioning is enabled.

Flag-gated OFF by default. Fail-safe: any load/caption error disables the captioner
and never touches detection.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger("jarvis.senses.captioner")

# The STACK-INSTALLED HEF (compatible with the live hailort). NOT the older
# ~/hailo-genai copy, which is a different build that fails VLM creation.
DEFAULT_HEF = "/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef"

# Terse, factual, negation-aware prompt — keeps generation short (lower latency =
# smaller Hailo-contention window) and matches the brain's honest-parse expectations.
_PROMPT = ("Describe the scene in one short sentence. List only the people and objects "
           "that are actually visible. Do not mention anything that is not present.")
_SYSTEM = "You describe scenes factually and briefly."


class SceneCaptioner:
    """Lazily-loaded VLM scene captioner sharing the detector's Hailo VDevice."""

    def __init__(self, vdevice, hef_path: str = DEFAULT_HEF, max_tokens: int = 48):
        self._vdevice = vdevice
        self._hef = hef_path
        self._max_tokens = max_tokens
        self._vlm = None
        self._loading = False
        self._load_failed = False
        self._disabled = False
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start_load_async(self) -> None:
        """Kick off the (~76s) HEF load in a background daemon thread."""
        if self._vlm is not None or self._loading or self._load_failed or self._disabled:
            return
        self._loading = True
        threading.Thread(target=self._load, name="vlm-load", daemon=True).start()

    def _load(self) -> None:
        try:
            from hailo_platform.genai import VLM
            t0 = time.time()
            vlm = VLM(self._vdevice, self._hef)
            self._vlm = vlm
            logger.info("Edge VLM loaded in %.0fs (%s)", time.time() - t0, self._hef)
        except Exception:
            logger.exception("Edge VLM load FAILED — captioner disabled (detection unaffected)")
            self._load_failed = True
        finally:
            self._loading = False

    @property
    def ready(self) -> bool:
        return self._vlm is not None and not self._load_failed and not self._disabled

    @property
    def loading(self) -> bool:
        return self._loading

    def disable(self, reason: str = "") -> None:
        self._disabled = True
        logger.warning("Edge captioner DISABLED%s", f": {reason}" if reason else "")

    # -- inference ----------------------------------------------------------

    def caption(self, frame_rgb) -> "tuple[str | None, int]":
        """Blocking caption of an RGB frame. Returns (text, latency_ms); (None, 0) on any failure.

        Serialized by a lock so two captions never overlap on the device.
        """
        if not self.ready or frame_rgb is None:
            return None, 0
        if not self._lock.acquire(blocking=False):
            return None, 0  # a caption is already running — skip this cycle
        try:
            import cv2
            img = cv2.resize(frame_rgb, (336, 336), interpolation=cv2.INTER_LINEAR)
            img = np.ascontiguousarray(img.astype(np.uint8))
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}]},
            ]
            t0 = time.time()
            try:
                self._vlm.clear_context()
            except Exception:
                pass
            out = self._vlm.generate_all(
                prompt=prompt, frames=[img], temperature=0.1, seed=42,
                max_generated_tokens=self._max_tokens,
            )
            text = out if isinstance(out, str) else "".join(list(out))
            text = text.split(". [{'type'")[0].split("<|im_end|>")[0].strip()
            return (text or None), int((time.time() - t0) * 1000)
        except Exception:
            logger.exception("Edge VLM caption failed")
            return None, 0
        finally:
            self._lock.release()

    def release(self) -> None:
        try:
            if self._vlm is not None:
                self._vlm.release()
        except Exception:
            pass
        self._vlm = None
