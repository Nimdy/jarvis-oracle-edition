"""NN-fleet consume overlay — design wire vs this-host firing.

Pins the honesty split:
  inference_consumed  = a consumer wire exists (audit)
  consumed_now        = that wire is firing on this host

Does not flip gates. Does not construct the live intent_shadow runner.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard import nn_fleet


def _quiet_overlays(monkeypatch, **named):
    monkeypatch.setattr(nn_fleet, "_live_world_model", named.get("world_model", lambda: None))
    monkeypatch.setattr(nn_fleet, "_live_simulator", named.get("simulator", lambda: None))
    monkeypatch.setattr(nn_fleet, "_live_intent_shadow", named.get("intent_shadow", lambda: None))
    monkeypatch.setattr(
        nn_fleet, "_live_weight_room",
        named.get("weight_room", lambda: {"enforces": False, "phase": "P2_lived_baseline_registry"}),
    )


def _by_name(view: dict, name: str) -> dict:
    rec = next(r for r in view["records"] if r["name"] == name)
    return rec


def test_fleet_has_36_records(monkeypatch):
    _quiet_overlays(monkeypatch)
    view = nn_fleet.build_fleet_view(None)
    assert view["total"] == 36
    assert view["generated_live"] is True
    assert "consumed_now" in view["note"]
    assert "true" in view["by_consumed_now"]


def test_world_model_level0_is_gated_not_mouth(monkeypatch):
    _quiet_overlays(
        monkeypatch,
        world_model=lambda: {
            "promotion_level": 0,
            "promotion_level_name": "shadow",
            "total_validated": 13722,
            "synthetic_validated": 0,
        },
    )
    wm = _by_name(nn_fleet.build_fleet_view(None), "world_model")
    assert wm["inference_consumed"] is True  # design wire exists
    assert wm["consumed_now"] is False
    assert wm["consumed_now_of"] == "llm_prompt_inject"
    assert wm["live_state"] == "gated"
    assert wm["live"]["promotion_level"] == 0


def test_world_model_level1_inject_is_consumed_now(monkeypatch):
    _quiet_overlays(
        monkeypatch,
        world_model=lambda: {
            "promotion_level": 1,
            "promotion_level_name": "advisory",
            "total_validated": 71,
            "synthetic_validated": 0,
        },
    )
    wm = _by_name(nn_fleet.build_fleet_view(None), "world_model")
    assert wm["consumed_now"] is True
    assert wm["live_state"] == "live-earning"


def test_world_model_unread_file_fail_closed_gated(monkeypatch):
    _quiet_overlays(monkeypatch, world_model=lambda: None)
    wm = _by_name(nn_fleet.build_fleet_view(None), "world_model")
    assert wm["consumed_now"] is False
    assert wm["live_state"] == "gated"


def test_intent_shadow_predicting_still_pass_through(monkeypatch):
    _quiet_overlays(
        monkeypatch,
        intent_shadow=lambda: {
            "level": "shadow",
            "observations_total": 209,
            "nn_predictions_total": 181,
            "rolling_agreement": 0.6851,
            "rescues_applied": 0,
            "primary_overrides_applied": 0,
            "ready_for_promotion": False,
        },
    )
    rec = _by_name(nn_fleet.build_fleet_view(None), "intent_shadow")
    assert rec["inference_consumed"] is False
    assert rec["consumed_now"] is False
    assert rec["live_state"] == "shadow"
    assert rec["live"]["nn_predictions_total"] == 181
    assert "pass-through" in rec["live"]["consume_note"]


def test_intent_shadow_advisory_would_show_consumed_now(monkeypatch):
    """Overlay truth only — does not promote the runner."""
    _quiet_overlays(
        monkeypatch,
        intent_shadow=lambda: {
            "level": "advisory",
            "observations_total": 500,
            "nn_predictions_total": 500,
            "rescues_applied": 1,
            "primary_overrides_applied": 0,
        },
    )
    rec = _by_name(nn_fleet.build_fleet_view(None), "intent_shadow")
    assert rec["consumed_now"] is True
    assert rec["live_state"] == "advisory"


def test_weight_room_enforces_false_is_not_consumed(monkeypatch):
    _quiet_overlays(monkeypatch)
    rec = _by_name(nn_fleet.build_fleet_view(None), "weight_room_gate")
    assert rec["consumed_now"] is False
    assert rec["live"]["enforces"] is False
    assert rec["live_state"] == "shadow"


def test_memory_ranker_stays_consumed_now(monkeypatch):
    _quiet_overlays(monkeypatch)
    rec = _by_name(nn_fleet.build_fleet_view(None), "memory_ranker")
    assert rec["inference_consumed"] is True
    assert rec["consumed_now"] is True
    assert rec["live_state"] == "live-earning"


def test_read_promotion_json_from_jarvis_home(tmp_path, monkeypatch):
    (tmp_path / "world_model_promotion.json").write_text('{"level": 0, "total_validated": 3}')
    monkeypatch.setattr(nn_fleet, "_JARVIS_HOME", tmp_path)
    st = nn_fleet._live_world_model()
    assert st == {
        "promotion_level": 0,
        "promotion_level_name": "shadow",
        "total_validated": 3,
        "synthetic_validated": 0,
    }


def test_overlay_does_not_construct_intent_shadow_runner(monkeypatch):
    seen = {"called": False}

    def boom():
        seen["called"] = True
        raise AssertionError("must not construct the live runner")

    monkeypatch.setattr(nn_fleet, "_live_intent_shadow", lambda: None)
    monkeypatch.setattr(nn_fleet, "_live_world_model", lambda: None)
    monkeypatch.setattr(nn_fleet, "_live_simulator", lambda: None)
    monkeypatch.setattr(nn_fleet, "_live_weight_room", lambda: {"enforces": False})
    # If someone rewires overlay to call get_intent_shadow_runner, this would fire
    # only if they import it — this test asserts the helper stays None-safe.
    from reasoning import intent_shadow as _is
    monkeypatch.setattr(_is, "get_intent_shadow_runner", boom)
    nn_fleet.build_fleet_view(None)
    assert seen["called"] is False
