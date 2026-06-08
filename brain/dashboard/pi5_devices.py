"""Derive per-device operational status for the Pi5 nervous-system dashboard.

Pure / dependency-free (no FastAPI) so it is unit-testable in any environment.
The cardinal rule here: every status is backed by telemetry that *actually
flows* — fps, inference-event recency, the audio device — never a fabricated
green light. The Hailo is explicitly *inferred* from inference events (direct
temp/util is a later phase); the mic has no direct telemetry yet and says so.
"""
from __future__ import annotations

from typing import Any

# Freshness window (s): a sensor/inference event seen within this counts the
# device as operational; older = stale. Generous because the Pi health-report
# cadence is ~30s.
PI5_FRESH_S = 30.0


def derive_pi5_devices(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the device list from the dashboard snapshot cache."""
    sh = cache.get("sensor_health", {}) or {}
    types = (cache.get("link", {}) or {}).get("types", {}) or {}
    lidar = cache.get("lidar", {}) or {}
    speakers = cache.get("speakers", {}) or {}
    sids = cache.get("sensors", []) or []
    prim = sh.get(sids[0], {}) if sids else {}

    def _age(*etypes: str):
        ages = [types[t]["last_seen_age_s"] for t in etypes
                if t in types and types[t].get("last_seen_age_s") is not None]
        return min(ages) if ages else None

    def _op(age):
        if age is None:
            return "unknown"
        return "operational" if age <= PI5_FRESH_S else "stale"

    cam_age = _age("scene_summary", "scene_caption", "face_crop", "person_detected")
    hailo_age = _age("scene_caption", "scene_summary", "pose_detected", "face_crop")
    return [
        {"name": "Pi node", "kind": "node", "present": bool(sids),
         "status": "up" if sids else "down",
         "detail": {"sensor_id": sids[0] if sids else None, "uptime_s": prim.get("uptime_s")}},
        {"name": "Camera (imx519)", "kind": "camera",
         "present": (prim.get("camera_fps") or 0) > 0 or cam_age is not None,
         "status": _op(cam_age),
         "detail": {"fps": prim.get("camera_fps"), "last_frame_event_age_s": cam_age}},
        {"name": "Hailo-10H NPU", "kind": "npu", "present": hailo_age is not None,
         "status": _op(hailo_age),
         "detail": {"last_inference_age_s": hailo_age},
         "note": "inferred from inference events; direct temp/util telemetry is phase 2"},
        {"name": "Speaker", "kind": "speaker", "present": bool(speakers.get("available")),
         "status": ("playing" if prim.get("audio_playing")
                    else "available" if speakers.get("available") else "down"),
         "detail": {"current": speakers.get("current")}},
        {"name": "Microphone", "kind": "mic", "present": True, "status": "telemetry_pending",
         "detail": {}, "note": "no direct mic telemetry yet (RMS/wake-score) — phase 2"},
        {"name": "RPLIDAR", "kind": "lidar", "present": bool(lidar),
         "status": "operational" if lidar else "absent",
         "detail": {"sensors": list(lidar.keys())}},
    ]
