"""Tests for the ClaimClassifierEncoder and CapabilityGate teacher signal recording."""

import os
import sys
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills.claim_encoder import (
    ClaimClassifierEncoder,
    FEATURE_DIM,
    NUM_CLASSES,
    LABEL_CLASSES,
    _DECISION_TAG_TO_CLASS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_context() -> dict:
    return {
        "token_count": 8,
        "char_len": 45,
        "has_first_person": True,
        "has_we_pronoun": False,
        "claim_pattern_index": 3,
        "is_readiness_frame": False,
        "pattern_category": "ability",
        "has_blocked_verb": False,
        "has_technical_signal": False,
        "has_internal_ops": False,
        "is_purely_conversational": True,
        "is_preference_alignment": False,
        "is_grounded_observation": False,
        "has_subordinate_context": False,
        "has_reflective_exclusion": False,
        "has_verified_skill_context": False,
        "route_is_none": True,
        "route_is_strict": False,
        "status_mode": False,
        "registry_verified": False,
        "registry_learning": False,
        "registry_unknown": True,
        "perception_evidence_fresh": False,
        "identity_confirmed": False,
        "family_block_count": 5,
        "session_block_count": 2,
        "time_since_last_block": 120.0,
    }


# ===========================================================================
# ClaimClassifierEncoder.encode() tests
# ===========================================================================

class TestClaimClassifierEncoderEncode:
    def test_output_dimension(self):
        vec = ClaimClassifierEncoder.encode(_full_context())
        assert len(vec) == FEATURE_DIM

    def test_all_values_in_range(self):
        vec = ClaimClassifierEncoder.encode(_full_context())
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_empty_context_produces_defaults(self):
        vec = ClaimClassifierEncoder.encode({})
        assert len(vec) == FEATURE_DIM
        # time_since_last_block defaults to 3600.0 → 1.0 after normalization
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_text_shape_block(self):
        ctx = _full_context()
        ctx["token_count"] = 10
        ctx["char_len"] = 60
        ctx["has_first_person"] = True
        ctx["has_we_pronoun"] = True
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[0] == pytest.approx(0.5, abs=0.01)  # 10/20
        assert vec[1] == pytest.approx(0.5, abs=0.01)  # 60/120
        assert vec[2] == 1.0  # has_first_person
        assert vec[3] == 1.0  # has_we_pronoun

    def test_pattern_match_block(self):
        ctx = _full_context()
        ctx["claim_pattern_index"] = 6
        ctx["is_readiness_frame"] = True
        ctx["pattern_category"] = "intention"
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[4] == pytest.approx(6 / 15.0, abs=0.01)
        assert vec[5] == 1.0  # readiness
        assert vec[6] == 0.0  # not ability
        assert vec[7] == 1.0  # intention

    def test_signal_flags_block(self):
        ctx = _full_context()
        ctx["has_blocked_verb"] = True
        ctx["has_technical_signal"] = True
        ctx["has_internal_ops"] = True
        ctx["is_purely_conversational"] = False
        ctx["is_preference_alignment"] = True
        ctx["is_grounded_observation"] = True
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[8] == 1.0
        assert vec[9] == 1.0
        assert vec[10] == 1.0
        assert vec[11] == 0.0
        assert vec[12] == 1.0
        assert vec[13] == 1.0

    def test_route_context_block(self):
        ctx = _full_context()
        ctx["route_is_none"] = False
        ctx["route_is_strict"] = True
        ctx["status_mode"] = True
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[17] == 0.0
        assert vec[18] == 1.0
        assert vec[19] == 1.0

    def test_registry_state_block(self):
        ctx = _full_context()
        ctx["registry_verified"] = True
        ctx["registry_learning"] = False
        ctx["registry_unknown"] = False
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[20] == 1.0
        assert vec[21] == 0.0
        assert vec[22] == 0.0

    def test_history_block_normalization(self):
        ctx = _full_context()
        ctx["family_block_count"] = 100  # over 50 cap
        ctx["session_block_count"] = 20  # over 10 cap
        ctx["time_since_last_block"] = 7200  # over 3600 cap
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[25] == 1.0  # clamped
        assert vec[26] == 1.0  # clamped
        assert vec[27] == 1.0  # clamped

    def test_history_block_zero(self):
        ctx = _full_context()
        ctx["family_block_count"] = 0
        ctx["session_block_count"] = 0
        ctx["time_since_last_block"] = 0
        vec = ClaimClassifierEncoder.encode(ctx)
        assert vec[25] == 0.0
        assert vec[26] == 0.0
        assert vec[27] == 0.0


# ===========================================================================
# ClaimClassifierEncoder.encode_label() tests
# ===========================================================================

class TestClaimClassifierEncoderLabel:
    def test_output_dimension(self):
        label, meta = ClaimClassifierEncoder.encode_label("conversational")
        assert len(label) == NUM_CLASSES

    def test_one_hot_for_each_class(self):
        for tag, cls_idx in _DECISION_TAG_TO_CLASS.items():
            label, meta = ClaimClassifierEncoder.encode_label(tag)
            assert label[cls_idx] == 1.0, f"tag '{tag}' should map to class {cls_idx}"
            assert sum(label) == pytest.approx(1.0)

    def test_all_eight_classes_reachable(self):
        reached = set()
        for tag, cls_idx in _DECISION_TAG_TO_CLASS.items():
            reached.add(cls_idx)
        assert reached == set(range(NUM_CLASSES)), f"Missing classes: {set(range(NUM_CLASSES)) - reached}"

    def test_unknown_tag_defaults_to_conversational(self):
        label, meta = ClaimClassifierEncoder.encode_label("some_unknown_tag")
        assert label[0] == 1.0
        assert meta["class_index"] == 0

    def test_bracket_stripping(self):
        label, meta = ClaimClassifierEncoder.encode_label("[blocked]")
        assert label[6] == 1.0
        assert meta["class_name"] == "blocked"

    def test_metadata_has_required_fields(self):
        label, meta = ClaimClassifierEncoder.encode_label("verified")
        assert "decision_tag" in meta
        assert "class_index" in meta
        assert "class_name" in meta

    def test_correction_label(self):
        label, meta = ClaimClassifierEncoder.encode_correction_label()
        assert len(label) == NUM_CLASSES
        assert label[0] == 1.0
        assert sum(label) == pytest.approx(1.0)
        assert meta["decision_tag"] == "friction_correction"

    def test_conversational_variants_map_to_class_0(self):
        tags = ["conversational", "subordinate-conversational", "route-conversational"]
        for tag in tags:
            label, _ = ClaimClassifierEncoder.encode_label(tag)
            assert label[0] == 1.0, f"tag '{tag}' should map to class 0"

    def test_verified_variants_map_to_class_2(self):
        tags = ["verified", "verified-context", "verified-offer"]
        for tag in tags:
            label, _ = ClaimClassifierEncoder.encode_label(tag)
            assert label[2] == 1.0, f"tag '{tag}' should map to class 2"

    def test_matrix_variants_map_to_class_5(self):
        tags = ["matrix:operational", "matrix:limited"]
        for tag in tags:
            label, _ = ClaimClassifierEncoder.encode_label(tag)
            assert label[5] == 1.0, f"tag '{tag}' should map to class 5"


# ===========================================================================
# CapabilityGate signal recording integration tests
# ===========================================================================

class TestCapabilityGateSignalRecording:
    def _make_gate(self):
        from skills.capability_gate import CapabilityGate
        from skills.registry import SkillRegistry, _default_skills
        reg = SkillRegistry(path="/dev/null")
        reg._skills = {r.skill_id: r for r in _default_skills()}
        reg._loaded = True
        reg.save = lambda: None
        gate = CapabilityGate(reg)
        return gate

    def test_label_distribution_starts_empty(self):
        gate = self._make_gate()
        assert gate.get_claim_label_distribution() == {}

    def test_check_text_populates_label_counts(self):
        gate = self._make_gate()
        gate.check_text("I can help you with that.")
        dist = gate.get_claim_label_distribution()
        assert sum(dist.values()) > 0

    def test_blocked_claim_records_blocked_label(self):
        gate = self._make_gate()
        gate.check_text("I can sing you a beautiful song.")
        dist = gate.get_claim_label_distribution()
        assert dist.get("blocked", 0) > 0

    def test_conversational_claim_records_conversational_label(self):
        gate = self._make_gate()
        gate.check_text("I can help you with that.")
        dist = gate.get_claim_label_distribution()
        assert dist.get("conversational", 0) > 0

    def test_ring_buffer_tracks_blocks(self):
        gate = self._make_gate()
        gate.check_text("I can sing you a song.")
        assert len(gate._recent_claims) > 0
        last = gate._recent_claims[-1]
        assert "claim_id" in last
        assert last["corrected"] is False
        assert last["decision_tag"] in ("blocked", "rewritten")

    def test_friction_correction_marks_claim(self):
        gate = self._make_gate()
        gate.check_text("I can sing you a song.")
        assert len(gate._recent_claims) > 0
        claim_before = gate._recent_claims[-1].copy()
        gate.record_friction_correction(time.time())
        assert gate._recent_claims[-1]["corrected"] is True

    def test_friction_correction_skips_old_claims(self):
        gate = self._make_gate()
        gate._recent_claims.append({
            "claim_id": "old_claim",
            "timestamp": time.time() - 120,  # 2 minutes ago (>60s eviction)
            "decision_tag": "blocked",
            "corrected": False,
        })
        gate.record_friction_correction(time.time())
        # Old entries are evicted during friction correlation
        assert len(gate._recent_claims) == 0

    def test_friction_correction_skips_already_corrected(self):
        gate = self._make_gate()
        gate._recent_claims.append({
            "claim_id": "already_done",
            "timestamp": time.time(),
            "decision_tag": "blocked",
            "corrected": True,
        })
        gate.record_friction_correction(time.time())
        # No crash, and the only entry remains corrected

    def test_ring_buffer_evicts_stale_entries(self):
        gate = self._make_gate()
        old_ts = time.time() - 90  # 90 seconds ago (>60s eviction)
        gate._recent_claims.append({
            "claim_id": "stale",
            "timestamp": old_ts,
            "decision_tag": "blocked",
            "corrected": False,
        })
        gate.record_friction_correction(time.time())
        assert len(gate._recent_claims) == 0

    def test_get_stats_includes_label_distribution(self):
        gate = self._make_gate()
        gate.check_text("I can help you understand.")
        stats = gate.get_stats()
        assert "claim_label_distribution" in stats


# ===========================================================================
# Tensor preparation tests
# ===========================================================================

class TestClaimClassifierTensors:
    def _make_collector(self, feature_count: int, verdict_count: int):
        """Mock distillation collector with claim_features and claim_verdict signals."""
        mock = MagicMock()

        class FakeSignal:
            def __init__(self, data, fidelity, metadata):
                self.data = data
                self.fidelity = fidelity
                self.metadata = metadata
                self.timestamp = time.time()

        features = []
        verdicts = []
        for i in range(max(feature_count, verdict_count)):
            cid = f"claim_{i}"
            if i < feature_count:
                features.append(FakeSignal(
                    data=[0.5] * FEATURE_DIM,
                    fidelity=1.0,
                    metadata={"claim_id": cid},
                ))
            if i < verdict_count:
                label = [0.0] * NUM_CLASSES
                label[i % NUM_CLASSES] = 1.0
                verdicts.append(FakeSignal(
                    data=label,
                    fidelity=1.0,
                    metadata={"claim_id": cid, "class_name": LABEL_CLASSES[i % NUM_CLASSES]},
                ))

        def get_training_batch(signal_type, limit=200, min_fidelity=0.0):
            if signal_type == "claim_features":
                return features[:limit]
            elif signal_type == "claim_verdict":
                return [v for v in verdicts[:limit] if v.fidelity >= min_fidelity]
            return []

        mock.get_training_batch = get_training_batch
        mock.count = lambda sig: feature_count if sig == "claim_features" else verdict_count
        return mock

    def test_returns_none_below_min_samples(self):
        from hemisphere.data_feed import _prepare_claim_classifier_tensors
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["claim_classifier"]
        collector = self._make_collector(5, 5)
        result = _prepare_claim_classifier_tensors(collector, config)
        assert result is None

    def test_returns_tensors_above_min_samples(self):
        from hemisphere.data_feed import _prepare_claim_classifier_tensors
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["claim_classifier"]
        collector = self._make_collector(20, 20)
        result = _prepare_claim_classifier_tensors(collector, config)
        assert result is not None
        features, labels, weights = result
        assert features.shape[1] == FEATURE_DIM
        assert labels.shape[1] == NUM_CLASSES
        assert features.shape[0] == labels.shape[0] == weights.shape[0]

    def test_zero_pads_short_features(self):
        from hemisphere.data_feed import _prepare_claim_classifier_tensors
        from hemisphere.types import DISTILLATION_CONFIGS

        config = DISTILLATION_CONFIGS["claim_classifier"]
        mock = MagicMock()

        class FakeSignal:
            def __init__(self, data, fidelity, metadata):
                self.data = data
                self.fidelity = fidelity
                self.metadata = metadata
                self.timestamp = time.time()

        features = []
        verdicts = []
        for i in range(20):
            cid = f"claim_{i}"
            features.append(FakeSignal(
                data=[0.5] * 20,  # shorter than FEATURE_DIM
                fidelity=1.0,
                metadata={"claim_id": cid},
            ))
            label = [0.0] * NUM_CLASSES
            label[0] = 1.0
            verdicts.append(FakeSignal(
                data=label,
                fidelity=1.0,
                metadata={"claim_id": cid},
            ))

        def get_training_batch(signal_type, limit=200, min_fidelity=0.0):
            if signal_type == "claim_features":
                return features[:limit]
            elif signal_type == "claim_verdict":
                return [v for v in verdicts[:limit] if v.fidelity >= min_fidelity]
            return []

        mock.get_training_batch = get_training_batch

        result = _prepare_claim_classifier_tensors(mock, config)
        assert result is not None
        feat_tensor, _, _ = result
        assert feat_tensor.shape[1] == FEATURE_DIM

    def test_friction_correction_overrides_original_verdict(self):
        """When both a fidelity=1.0 and a fidelity=0.7 verdict exist for
        the same claim_id, the higher fidelity one wins."""
        from hemisphere.data_feed import _prepare_claim_classifier_tensors
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["claim_classifier"]
        mock = MagicMock()

        class FakeSignal:
            def __init__(self, data, fidelity, metadata):
                self.data = data
                self.fidelity = fidelity
                self.metadata = metadata
                self.timestamp = time.time()

        features = []
        verdicts = []
        for i in range(20):
            cid = f"claim_{i}"
            features.append(FakeSignal(
                data=[0.5] * FEATURE_DIM,
                fidelity=1.0,
                metadata={"claim_id": cid},
            ))
            blocked_label = [0.0] * NUM_CLASSES
            blocked_label[6] = 1.0  # blocked
            verdicts.append(FakeSignal(
                data=blocked_label,
                fidelity=1.0,
                metadata={"claim_id": cid},
            ))
            if i == 0:
                correction_label = [0.0] * NUM_CLASSES
                correction_label[0] = 1.0  # conversational correction
                verdicts.append(FakeSignal(
                    data=correction_label,
                    fidelity=0.7,
                    metadata={"claim_id": cid},
                ))

        def get_training_batch(signal_type, limit=200, min_fidelity=0.0):
            if signal_type == "claim_features":
                return features[:limit]
            elif signal_type == "claim_verdict":
                return [v for v in verdicts[:limit] if v.fidelity >= min_fidelity]
            return []

        mock.get_training_batch = get_training_batch

        result = _prepare_claim_classifier_tensors(mock, config)
        assert result is not None
        _, labels, _ = result
        # claim_0 should use the fidelity=1.0 label (blocked), not the 0.7 correction
        assert labels[0][6].item() == pytest.approx(1.0)

    def test_dispatched_via_prepare_distillation_tensors(self):
        from hemisphere.data_feed import prepare_distillation_tensors
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["claim_classifier"]
        collector = self._make_collector(20, 20)
        result = prepare_distillation_tensors("claim_classifier", collector, config)
        assert result is not None


# ===========================================================================
# Hemisphere config tests
# ===========================================================================

class TestHemisphereConfig:
    def test_claim_classifier_in_focus_enum(self):
        from hemisphere.types import HemisphereFocus
        assert hasattr(HemisphereFocus, "CLAIM_CLASSIFIER")
        assert HemisphereFocus.CLAIM_CLASSIFIER.value == "claim_classifier"

    def test_claim_classifier_in_distillation_configs(self):
        from hemisphere.types import DISTILLATION_CONFIGS
        assert "claim_classifier" in DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["claim_classifier"]
        assert config.input_dim == FEATURE_DIM
        assert config.output_dim == NUM_CLASSES
        assert config.loss == "kl_div"
        assert config.is_permanent is True
        assert config.feature_source == "claim_features"
        assert config.teacher == "claim_verdict"

    def test_claim_classifier_in_tier1_focuses(self):
        from hemisphere.orchestrator import _TIER1_FOCUSES
        from hemisphere.types import HemisphereFocus
        assert HemisphereFocus.CLAIM_CLASSIFIER in _TIER1_FOCUSES
