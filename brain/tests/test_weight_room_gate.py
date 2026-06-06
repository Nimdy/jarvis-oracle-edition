"""Weight-Room P2 — lived-baseline registry + would-block evaluator (SHADOW)."""

from __future__ import annotations

from hemisphere.weight_room_gate import (
    WeightRoomGate, classify,
    MODE_HYBRID, MODE_LIVED, MODE_BLOCKED_BY_DESIGN, MODE_NOT_YET_GATABLE, MODE_EXEMPT,
    EXEMPT_SPECIALISTS, _HYBRID_MIN_SYNTHETIC, _HYBRID_MIN_LIVED,
)


class TestClassify:
    def test_tier1_exempt(self):
        for s in EXEMPT_SPECIALISTS:
            assert classify(s)["mode"] == MODE_EXEMPT

    def test_rare_event_hybrid_by_substring(self):
        # collector keys carry source suffixes — substring match must still classify
        assert classify("skill_acquisition_outcome")["mode"] == MODE_HYBRID
        assert classify("skill_acquisition_features")["mode"] == MODE_HYBRID
        assert classify("plan_features")["mode"] == MODE_HYBRID
        assert classify("claim_classifier")["mode"] == MODE_HYBRID
        assert classify("diagnostic")["mode"] == MODE_HYBRID

    def test_blocked_by_design(self):
        assert classify("thought_trigger_selector")["mode"] == MODE_BLOCKED_BY_DESIGN
        assert classify("code_quality")["mode"] == MODE_BLOCKED_BY_DESIGN

    def test_unknown_defaults_not_yet_gatable(self):
        # conservative honest default — never a silent allow
        assert classify("some_new_specialist")["mode"] == MODE_NOT_YET_GATABLE


class TestWouldBlock:
    def setup_method(self):
        self.gate = WeightRoomGate()

    def test_hybrid_blocks_without_lived_baseline(self):
        # plenty of synthetic, zero lived -> would_block (lived baseline not met)
        d = self.gate._evaluate_one("skill_acquisition_outcome", lived=0, synthetic=200)
        assert d["decision"] == "would_block"
        assert d["lived_baseline_met"] is False

    def test_hybrid_allows_when_floor_met(self):
        d = self.gate._evaluate_one(
            "skill_acquisition_outcome",
            lived=_HYBRID_MIN_LIVED, synthetic=_HYBRID_MIN_SYNTHETIC,
        )
        assert d["decision"] == "would_allow"
        assert d["lived_baseline_met"] is True

    def test_hybrid_synthetic_alone_is_not_enough(self):
        # the whole point: synthetic reps can NEVER substitute for the lived baseline
        d = self.gate._evaluate_one("plan_evaluator", lived=0, synthetic=10_000)
        assert d["decision"] == "would_block"

    def test_exempt_never_blocks(self):
        d = self.gate._evaluate_one("voice_intent", lived=0, synthetic=0)
        assert d["decision"] == "exempt"
        assert d["lived_baseline_met"] is True

    def test_blocked_by_design_is_distinct_from_would_block(self):
        # not eligible to be measured yet -> its own honest state, not "failed the floor"
        d = self.gate._evaluate_one("thought_trigger_selector", lived=0, synthetic=500)
        assert d["decision"] == MODE_BLOCKED_BY_DESIGN
        assert d["decision"] != "would_block"

    def test_lived_mode_floor(self):
        assert self.gate._evaluate_one("dream_synthesis", lived=3, synthetic=0)["decision"] == "would_block"
        assert self.gate._evaluate_one("dream_synthesis", lived=10, synthetic=0)["decision"] == "would_allow"

    def test_enforces_nothing(self):
        # get_status must declare it enforces nothing (shadow contract)
        st = self.gate.get_status()
        assert st["enforces"] is False
        assert st["authority"] == "shadow_would_block_only"
        assert st["phase"] == "P2_lived_baseline_registry"

    def test_evaluate_all_never_raises(self):
        # fail-closed-to-shadow: even with no collector it returns a dict, never raises
        assert isinstance(self.gate.evaluate_all(), dict)
