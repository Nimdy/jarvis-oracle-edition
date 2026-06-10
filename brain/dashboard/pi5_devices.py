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
    link = cache.get("link", {}) or {}
    types = link.get("types", {}) or {}
    audio = link.get("audio", {}) or {}
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

    # --- per-stream heartbeats kept SEPARATE (don't collapse a quiet scene into a dead device) ---
    senses_age = _age("sensor_health", "sensor_status")     # the Pi senses-loop heartbeat
    senses_alive = senses_age is not None and senses_age <= PI5_FRESH_S
    cam_fps = prim.get("camera_fps") or 0
    scene_age = _age("scene_summary", "scene_caption", "face_crop", "person_detected")  # scene-pipeline ACTIVITY
    infer_age = _age("scene_caption", "scene_summary", "pose_detected", "face_crop")     # last DETECTION event
    pose_age = _age("pose_detected")
    face_age = _age("face_crop")

    # camera is ALIVE if it's capturing frames and the senses loop is heartbeating — scene
    # events are activity-dependent (quiet room ⇒ no events ≠ dead camera).
    if cam_fps > 0 and senses_alive:
        cam_status = "operational"
    elif scene_age is not None and scene_age <= PI5_FRESH_S:
        cam_status = "operational"
    elif cam_fps > 0 or scene_age is not None or senses_alive:
        cam_status = "stale"
    else:
        cam_status = "unknown"

    # the Hailo runs the detector once per camera frame, so a fresh detection event proves it;
    # if the scene is quiet but the camera+loop are alive, it's running (idle), not stale.
    if infer_age is not None and infer_age <= PI5_FRESH_S:
        hailo_status = "operational"
    elif senses_alive and cam_fps > 0:
        hailo_status = "operational"
    elif infer_age is not None or senses_alive:
        hailo_status = "stale"
    else:
        hailo_status = "unknown"

    return [
        {"name": "Pi node", "kind": "node", "present": bool(sids),
         "status": "up" if sids else "down",
         "detail": {"sensor_id": sids[0] if sids else None, "uptime_s": prim.get("uptime_s")}},
        {"name": "Camera (imx519)", "kind": "camera",
         "present": cam_fps > 0 or scene_age is not None,
         "status": cam_status,
         "detail": {"fps": cam_fps, "senses_loop_age_s": senses_age,
                    "scene_activity_age_s": scene_age, "face_age_s": face_age},
         "note": "alive = capturing fps + senses-loop heartbeat; scene_activity is quiet when nothing changes (not a fault)"},
        {"name": "Hailo-10H NPU", "kind": "npu", "present": infer_age is not None or (cam_fps > 0 and senses_alive),
         "status": hailo_status,
         "detail": {"last_inference_event_age_s": infer_age, "pose_age_s": pose_age, "camera_fps": cam_fps},
         "note": "detector runs per camera frame; last_inference_event is the last DETECTION (quiet scene ⇒ running but no events)"},
        {"name": "Speaker", "kind": "speaker", "present": bool(speakers.get("available")),
         "status": ("playing" if prim.get("audio_playing")
                    else "available" if speakers.get("available") else "down"),
         "detail": {"current": speakers.get("current")}},
        {"name": "Microphone", "kind": "mic",
         "present": (audio.get("chunks") or 0) > 0,
         "status": _op(audio.get("last_recv_age_s")) if audio.get("chunks") else "telemetry_pending",
         "detail": {"chunks": audio.get("chunks"), "last_audio_age_s": audio.get("last_recv_age_s")}},
        {"name": "RPLIDAR", "kind": "lidar", "present": bool(lidar),
         "status": "operational" if lidar else "absent",
         "detail": {"sensors": list(lidar.keys())}},
    ]
