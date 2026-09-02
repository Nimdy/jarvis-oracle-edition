"""thought_trigger_selector encoder matches DistillationConfig 24→13."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from hemisphere.thought_trigger_encoder import (
    FEATURE_DIM,
    NUM_TRIGGER_TYPES,
    THOUGHT_TRIGGER_NAMES,
    encode,
    encode_label,
    record_teacher_pair,
)
from hemisphere.types import DISTILLATION_CONFIGS
from consciousness.meta_cognitive_thoughts import THOUGHT_TRIGGER_NAMES as LIVE_NAMES


def test_dims_match_live_config():
    cfg = DISTILLATION_CONFIGS["thought_trigger_selector"]
    assert FEATURE_DIM == 24 == cfg.input_dim
    assert NUM_TRIGGER_TYPES == 13 == cfg.output_dim == len(LIVE_NAMES)
    assert tuple(THOUGHT_TRIGGER_NAMES) == tuple(LIVE_NAMES)


def test_encode_is_24_and_clamped():
    vec = encode([0.2, 1.5, -0.1] + [0.0] * 20)
    assert len(vec) == FEATURE_DIM
    assert all(0.0 <= x <= 1.0 for x in vec)
    assert vec[0] == 0.2
    assert vec[1] == 1.0
    assert vec[2] == 0.0


def test_encode_does_not_one_hot_the_trigger():
    """Label leak: trigger name must not occupy a 12/13-wide block in features."""
    a = encode([0.1] * 20, trigger_type="self_observation")
    b = encode([0.1] * 20, trigger_type="belief_validation_curiosity")
    assert a[:20] == b[:20]


def test_encode_label_is_trigger_one_hot_not_outcome():
    lab = encode_label("belief_validation_curiosity")
    assert len(lab) == NUM_TRIGGER_TYPES
    assert sum(lab) == 1.0
    assert lab[LIVE_NAMES.index("belief_validation_curiosity")] == 1.0
    assert encode_label("not_a_trigger") == [0.0] * NUM_TRIGGER_TYPES


def test_record_teacher_pair_writes_both_signals():
    recorded = []

    class _C:
        def record(self, **kwargs):
            recorded.append(kwargs)

    with patch("hemisphere.distillation.distillation_collector", _C()):
        record_teacher_pair(
            thought_id="intent_1",
            trigger_name="belief_validation_curiosity",
            grounded=True,
        )

    teachers = {r["teacher"] for r in recorded}
    assert teachers == {"thought_trigger_features", "thought_trigger_resolver"}
    feat = next(r for r in recorded if r["teacher"] == "thought_trigger_features")
    lab = next(r for r in recorded if r["teacher"] == "thought_trigger_resolver")
    assert len(feat["data"]) == FEATURE_DIM
    assert len(lab["data"]) == NUM_TRIGGER_TYPES
    assert feat["metadata"]["thought_id"] == "intent_1"
    assert feat["origin"] == "live"
    assert feat["fidelity"] == 1.0


def test_record_skips_empty_thought_id():
    recorded = []

    class _C:
        def record(self, **kwargs):
            recorded.append(kwargs)

    with patch("hemisphere.distillation.distillation_collector", _C()):
        record_teacher_pair(thought_id="", trigger_name="self_observation")
    assert recorded == []


def test_data_feed_has_pairing_branch():
    src = Path(__file__).resolve().parent.parent.joinpath("hemisphere/data_feed.py").read_text()
    assert "def _prepare_thought_trigger_tensors(" in src
    assert 'if focus_name == "thought_trigger_selector":' in src
    assert "thought_id" in src


def test_not_in_tier1_focuses():
    src = Path(__file__).resolve().parent.parent.joinpath("hemisphere/orchestrator.py").read_text()
    start = src.find("_TIER1_FOCUSES = frozenset({")
    end = src.find("})", start)
    block = src[start:end]
    assert "THOUGHT_TRIGGER_SELECTOR" not in block


def test_still_blocked_by_design():
    from hemisphere.weight_room_gate import classify, MODE_BLOCKED_BY_DESIGN

    assert classify("thought_trigger_selector")["mode"] == MODE_BLOCKED_BY_DESIGN


def test_research_intent_records_the_pair():
    src = Path(__file__).resolve().parent.parent.joinpath("autonomy/research_intent.py").read_text()
    assert "record_teacher_pair" in src
    assert "belief_validation_curiosity" in src
