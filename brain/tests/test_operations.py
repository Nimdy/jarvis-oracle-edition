from __future__ import annotations

import time


def test_user_facing_stale_subsystem_does_not_win_current_state():
    from consciousness.operations import OP_ACTIVE, OP_IDLE, synthesize_v2

    now = time.time()
    raw = {
        "subsystems": {
            "stt": {
                "status": "transcribing",
                "detail": "old transcription",
                "age_s": 3600,
            },
        },
        "timeline": [],
        "interactive_path": {
            "conversation_id": "conv-stale",
            "stages": [
                {
                    "key": "stt",
                    "label": "STT",
                    "status": OP_ACTIVE,
                    "detail": "old stage",
                    "ts": now - 3600,
                },
            ],
        },
        "boot_ts": now - 7200,
    }

    view = synthesize_v2(raw, {"phase": "LISTENING", "mode": "passive"})

    assert view["subsystems"]["stt"]["status"] == OP_IDLE
    assert view["subsystems"]["stt"]["stale_cleared"] is True
    assert view["current"]["name"] != "stt"
    assert view["interactive_path"]["stages"][0]["status"] == OP_IDLE
    assert view["interactive_path"]["stages"][0]["stale_cleared"] is True
