"""Tests for the P5 ``check_hrr_scene_authority`` truth-probe check."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.dashboard_truth_probe import (
    SEVERITY_FAIL,
    SEVERITY_INFO,
    check_hrr_scene_authority,
)


def _good_hrr_scene() -> dict:
    return {
        "status": "PRE-MATURE",
        "lane": "spatial_hrr_mental_world",
        "enabled": True,
        "entity_count": 2,
        "relation_count": 1,
        "entities": [
            {"entity_id": "a", "label": "cup", "state": "visible", "region": "desk_center"},
            {"entity_id": "b", "label": "cup", "state": "visible", "region": "desk_center"},
        ],
        "relations": [
            {
                "source_entity_id": "a", "target_entity_id": "b",
                "relation_type": "left_of", "value_m": 0.5, "confidence": 0.85,
            },
        ],
        "metrics": {"entities_encoded": 2, "relations_encoded": 1},
        "writes_memory": False,
        "writes_beliefs": False,
        "influences_policy": False,
        "influences_autonomy": False,
        "soul_integrity_influence": False,
        "llm_raw_vector_exposure": False,
        "no_raw_vectors_in_api": True,
    }


def test_happy_path_has_no_findings():
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": _good_hrr_scene()}, findings)
    assert [f.code for f in findings] == []


def test_missing_block_is_info_not_fail():
    findings: list = []
    check_hrr_scene_authority({}, findings)
    assert len(findings) == 1
    assert findings[0].severity == SEVERITY_INFO
    assert findings[0].code == "hrr_scene.missing"


def test_wrong_status_is_fail():
    bad = _good_hrr_scene()
    bad["status"] = "SHIPPED"
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": bad}, findings)
    fail_codes = [f.code for f in findings if f.severity == SEVERITY_FAIL]
    assert "hrr_scene.status.not_pre_mature" in fail_codes


def test_wrong_lane_is_fail():
    bad = _good_hrr_scene()
    bad["lane"] = "something_else"
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": bad}, findings)
    fail_codes = [f.code for f in findings if f.severity == SEVERITY_FAIL]
    assert "hrr_scene.lane.wrong" in fail_codes


def test_authority_flag_true_is_fail():
    bad = _good_hrr_scene()
    bad["writes_memory"] = True
    bad["influences_policy"] = True
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": bad}, findings)
    fail_codes = {f.code for f in findings if f.severity == SEVERITY_FAIL}
    assert "hrr_scene.authority.writes_memory" in fail_codes
    assert "hrr_scene.authority.influences_policy" in fail_codes


def test_no_raw_vectors_in_api_must_be_true():
    bad = _good_hrr_scene()
    bad["no_raw_vectors_in_api"] = False
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": bad}, findings)
    fail_codes = {f.code for f in findings if f.severity == SEVERITY_FAIL}
    assert "hrr_scene.authority.no_raw_vectors_in_api" in fail_codes


def test_raw_vector_leak_detected_in_nested_entity():
    bad = _good_hrr_scene()
    bad["entities"][0]["vector"] = [0.1, 0.2]
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": bad}, findings)
    fail_codes = {f.code for f in findings if f.severity == SEVERITY_FAIL}
    assert "hrr_scene.raw_vector_leak" in fail_codes


def test_non_dict_block_is_fail():
    findings: list = []
    check_hrr_scene_authority({"hrr_scene": ["not", "a", "dict"]}, findings)
    fail_codes = {f.code for f in findings if f.severity == SEVERITY_FAIL}
    assert "hrr_scene.shape.not_dict" in fail_codes
