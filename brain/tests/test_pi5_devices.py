"""Pi5 device-status liveness — the fix for 'camera STALE while it's capturing at 15fps'.

A quiet scene (no scene_summary/face events) must NOT collapse a live camera into 'stale':
the camera-alive signal is fps + the senses-loop heartbeat, kept SEPARATE from scene activity.
"""
from dashboard.pi5_devices import derive_pi5_devices


def _cache(fps, senses_age, scene_age, pose_age=None):
    types = {}
    if senses_age is not None:
        types["sensor_health"] = {"last_seen_age_s": senses_age}
    if scene_age is not None:
        types["scene_summary"] = {"last_seen_age_s": scene_age}
        types["face_crop"] = {"last_seen_age_s": scene_age}
    if pose_age is not None:
        types["pose_detected"] = {"last_seen_age_s": pose_age}
    return {"sensors": ["pi5-senses"], "sensor_health": {"pi5-senses": {"camera_fps": fps}},
            "link": {"types": types}}


def _dev(devs, kind):
    return next(d for d in devs if d["kind"] == kind)


def test_camera_operational_while_capturing_despite_quiet_scene():
    # the bug: fps=15 + senses loop fresh, but scene events stale (quiet room) → was 'stale'
    devs = derive_pi5_devices(_cache(fps=15, senses_age=2.0, scene_age=99.0))
    assert _dev(devs, "camera")["status"] == "operational"
    assert _dev(devs, "npu")["status"] == "operational"      # detector runs per frame


def test_camera_stale_only_when_truly_dead():
    # fps=0 and the senses loop heartbeat gone → genuinely stale, not a quiet scene
    devs = derive_pi5_devices(_cache(fps=0, senses_age=120.0, scene_age=120.0))
    assert _dev(devs, "camera")["status"] in ("stale", "unknown")


def test_heartbeats_kept_separate_in_detail():
    devs = derive_pi5_devices(_cache(fps=15, senses_age=2.0, scene_age=45.0, pose_age=3.0))
    cam = _dev(devs, "camera")["detail"]
    assert cam["senses_loop_age_s"] == 2.0 and cam["scene_activity_age_s"] == 45.0   # not collapsed
    assert _dev(devs, "npu")["detail"]["pose_age_s"] == 3.0


def test_scene_activity_alone_still_marks_operational():
    # camera_fps missing but scene events fresh → still operational (events prove frames)
    devs = derive_pi5_devices(_cache(fps=0, senses_age=None, scene_age=2.0))
    assert _dev(devs, "camera")["status"] == "operational"
