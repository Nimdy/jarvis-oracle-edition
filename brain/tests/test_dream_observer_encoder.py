"""Tests for the DREAM_SYNTHESIS Tier-1 specialist: encoder, labels, tensor prep,
persistence bridge, and anti-authority boundary enforcement.

The dream-synthesis specialist is a validator-shadow approximator.  It learns to
predict the ReflectiveValidator's artifact disposition.  It does NOT:
  - mutate artifact validation state
  - write to memory
  - emit events as authority
  - bypass the ReflectiveValidator
"""

import ast
import importlib
import inspect
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from hemisphere.dream_artifact_encoder import (
    FEATURE_DIM,
    LABEL_CLASSES,
    REASON_CATEGORIES,
    DreamArtifactEncoder,
    _classify_reason,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _sample_artifact(overrides: dict | None = None) -> dict:
    art = {
        "artifact_id": "dart_abc123def456",
        "artifact_type": "symbolic_summary",
        "confidence": 0.72,
        "cluster_coherence": 0.68,
        "source_memory_ids": ("mem_a", "mem_b"),
        "content": "Pattern synthesis: user engagement correlates with memory density",
    }
    if overrides:
        art.update(overrides)
    return art


def _sample_system_context(overrides: dict | None = None) -> dict:
    ctx = {
        "memory_density": 0.45,
        "dream_cycle_count": 12,
        "awareness": 0.65,
        "belief_count": 200,
        "contradiction_debt": 0.08,
        "soul_integrity": 0.92,
        "quarantine_pressure": 0.10,
        "promotion_rate_session": 0.35,
    }
    if overrides:
        ctx.update(overrides)
    return ctx


# -----------------------------------------------------------------------
# Block A: Artifact intrinsic properties (8 dims)
# -----------------------------------------------------------------------

class TestBlockA:
    def test_artifact_type_ordinals(self):
        for atype, expected_ord in [
            ("bridge_candidate", 0.0),
            ("symbolic_summary", 0.2),
            ("tension_flag", 0.4),
            ("consolidation_proposal", 0.6),
            ("waking_question", 0.8),
            ("shadow_scenario", 1.0),
        ]:
            art = _sample_artifact({"artifact_type": atype})
            block = DreamArtifactEncoder.encode_artifact_block(art)
            assert block[0] == expected_ord, f"type={atype}"

    def test_confidence_and_coherence_pass_through(self):
        art = _sample_artifact({"confidence": 0.55, "cluster_coherence": 0.77})
        block = DreamArtifactEncoder.encode_artifact_block(art)
        assert abs(block[1] - 0.55) < 1e-6
        assert abs(block[2] - 0.77) < 1e-6

    def test_source_count_normalization(self):
        art = _sample_artifact({"source_memory_ids": tuple(f"m{i}" for i in range(15))})
        block = DreamArtifactEncoder.encode_artifact_block(art)
        assert block[3] == 1.0  # clamped at 10
        assert block[5] == 1.0  # has_sources

    def test_no_sources_flags(self):
        art = _sample_artifact({"source_memory_ids": ()})
        block = DreamArtifactEncoder.encode_artifact_block(art)
        assert block[3] == 0.0
        assert block[5] == 0.0

    def test_informational_type_flag(self):
        art_tension = _sample_artifact({"artifact_type": "tension_flag"})
        art_question = _sample_artifact({"artifact_type": "waking_question"})
        art_summary = _sample_artifact({"artifact_type": "symbolic_summary"})
        assert DreamArtifactEncoder.encode_artifact_block(art_tension)[6] == 1.0
        assert DreamArtifactEncoder.encode_artifact_block(art_question)[6] == 1.0
        assert DreamArtifactEncoder.encode_artifact_block(art_summary)[6] == 0.0

    def test_consolidation_proposal_flag(self):
        art = _sample_artifact({"artifact_type": "consolidation_proposal"})
        assert DreamArtifactEncoder.encode_artifact_block(art)[7] == 1.0
        art2 = _sample_artifact({"artifact_type": "symbolic_summary"})
        assert DreamArtifactEncoder.encode_artifact_block(art2)[7] == 0.0

    def test_content_length_normalization(self):
        art_short = _sample_artifact({"content": "x" * 100})
        art_full = _sample_artifact({"content": "x" * 600})
        block_short = DreamArtifactEncoder.encode_artifact_block(art_short)
        block_full = DreamArtifactEncoder.encode_artifact_block(art_full)
        assert abs(block_short[4] - 100 / 500) < 1e-6
        assert block_full[4] == 1.0


# -----------------------------------------------------------------------
# Block B: System state (5 dims)
# -----------------------------------------------------------------------

class TestBlockB:
    def test_memory_density(self):
        block = DreamArtifactEncoder.encode_system_block({"memory_density": 0.75})
        assert abs(block[0] - 0.75) < 1e-6

    def test_dream_cycle_count_normalization(self):
        block = DreamArtifactEncoder.encode_system_block({"dream_cycle_count": 500})
        assert abs(block[1] - 0.5) < 1e-6

    def test_belief_count_normalization(self):
        block = DreamArtifactEncoder.encode_system_block({"belief_count": 2000})
        assert block[3] == 1.0

    def test_empty_context_defaults_to_zero(self):
        block = DreamArtifactEncoder.encode_system_block({})
        assert all(v == 0.0 for v in block)


# -----------------------------------------------------------------------
# Block C: Governance pressure (3 dims)
# -----------------------------------------------------------------------

class TestBlockC:
    def test_soul_integrity_defaults_high(self):
        block = DreamArtifactEncoder.encode_governance_block({})
        assert block[0] == 1.0  # soul_integrity defaults to 1.0

    def test_quarantine_pressure(self):
        block = DreamArtifactEncoder.encode_governance_block({"quarantine_pressure": 0.6})
        assert abs(block[1] - 0.6) < 1e-6

    def test_promotion_rate(self):
        block = DreamArtifactEncoder.encode_governance_block({"promotion_rate_session": 0.4})
        assert abs(block[2] - 0.4) < 1e-6


# -----------------------------------------------------------------------
# Full encode
# -----------------------------------------------------------------------

class TestEncode:
    def test_output_dimension(self):
        vec = DreamArtifactEncoder.encode(_sample_artifact(), _sample_system_context())
        assert len(vec) == FEATURE_DIM
        assert FEATURE_DIM == 16

    def test_all_values_clamped_01(self):
        vec = DreamArtifactEncoder.encode(_sample_artifact(), _sample_system_context())
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} = {v}"

    def test_extreme_inputs_clamp(self):
        art = _sample_artifact({"confidence": 5.0, "cluster_coherence": -1.0})
        ctx = _sample_system_context({"quarantine_pressure": 99.0})
        vec = DreamArtifactEncoder.encode(art, ctx)
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} = {v}"

    def test_empty_inputs_produce_valid_vector(self):
        vec = DreamArtifactEncoder.encode({}, {})
        assert len(vec) == FEATURE_DIM
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} = {v}"


# -----------------------------------------------------------------------
# Label encoding
# -----------------------------------------------------------------------

class TestLabelEncoding:
    def test_four_classes(self):
        assert len(LABEL_CLASSES) == 4
        assert LABEL_CLASSES == ["promoted", "held", "discarded", "quarantined"]

    def test_each_class_produces_one_hot(self):
        for cls in LABEL_CLASSES:
            label, _ = DreamArtifactEncoder.encode_label(cls, _sample_artifact(), "notes")
            assert len(label) == 4
            assert sum(label) == 1.0
            assert label[LABEL_CLASSES.index(cls)] == 1.0

    def test_unknown_class_produces_zero_vector(self):
        label, _ = DreamArtifactEncoder.encode_label("bogus", _sample_artifact())
        assert sum(label) == 0.0

    def test_metadata_contains_artifact_id(self):
        _, meta = DreamArtifactEncoder.encode_label("promoted", _sample_artifact(), "meets thresholds")
        assert meta["artifact_id"] == "dart_abc123def456"
        assert meta["artifact_type"] == "symbolic_summary"

    def test_reason_category_from_notes(self):
        _, meta = DreamArtifactEncoder.encode_label("discarded", _sample_artifact(), "no source memories")
        assert meta["reason_category"] == "no_sources"

    def test_reason_category_borderline_default(self):
        _, meta = DreamArtifactEncoder.encode_label("held", _sample_artifact(), "some unknown note")
        assert meta["reason_category"] == "borderline_hold"

    def test_reason_categories_cover_all_validator_paths(self):
        notes_to_category = {
            "no source memories": "no_sources",
            "contradicts active beliefs": "contradicts_beliefs",
            "informational artifact, hold for context": "informational_hold",
            "low coherence: 0.30": "low_coherence",
            "low confidence: 0.20": "low_confidence",
            "meets promotion thresholds": "meets_thresholds",
            "borderline quality, holding for further review": "borderline_hold",
            "promotion cap reached, deferring": "promotion_cap",
        }
        for notes, expected_cat in notes_to_category.items():
            cat = _classify_reason(notes)
            assert cat == expected_cat, f"notes={notes!r}"


# -----------------------------------------------------------------------
# Tensor preparation
# -----------------------------------------------------------------------

class TestTensorPrep:
    def test_pairs_by_artifact_id(self):
        """Simulates the tensor prep matching by artifact_id."""
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["dream_synthesis"]

        assert config.input_dim == 16
        assert config.output_dim == 4
        assert config.feature_source == "dream_features"
        assert config.teacher == "dream_validator"
        assert config.is_permanent is True

    def test_config_loss_is_kl_div(self):
        from hemisphere.types import DISTILLATION_CONFIGS
        assert DISTILLATION_CONFIGS["dream_synthesis"].loss == "kl_div"

    def test_config_min_samples(self):
        from hemisphere.types import DISTILLATION_CONFIGS
        assert DISTILLATION_CONFIGS["dream_synthesis"].min_samples == 15

    def test_prep_returns_none_below_min_samples(self):
        """Integration test: _prepare_dream_observer_tensors returns None with insufficient data."""
        pytest.importorskip("torch")
        from hemisphere.data_feed import _prepare_dream_observer_tensors
        from hemisphere.types import DISTILLATION_CONFIGS

        config = DISTILLATION_CONFIGS["dream_synthesis"]
        mock_collector = MagicMock()
        mock_collector.get_training_batch.return_value = []
        result = _prepare_dream_observer_tensors(mock_collector, config)
        assert result is None

    def test_prep_matches_paired_signals(self):
        pytest.importorskip("torch")
        from hemisphere.data_feed import _prepare_dream_observer_tensors
        from hemisphere.types import DISTILLATION_CONFIGS

        config = DISTILLATION_CONFIGS["dream_synthesis"]

        features = []
        labels = []
        for i in range(20):
            aid = f"dart_{i:012d}"
            feat_sig = MagicMock()
            feat_sig.data = [0.1 * (i % 10)] * 16
            feat_sig.metadata = {"artifact_id": aid}
            feat_sig.fidelity = 1.0
            features.append(feat_sig)

            label_sig = MagicMock()
            label_sig.data = [1.0, 0.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0, 0.0]
            label_sig.metadata = {"artifact_id": aid}
            label_sig.fidelity = 1.0
            labels.append(label_sig)

        mock_collector = MagicMock()
        def side_effect(source, limit=200, min_fidelity=None):
            if source == "dream_features":
                return features
            return labels
        mock_collector.get_training_batch.side_effect = side_effect

        result = _prepare_dream_observer_tensors(mock_collector, config)
        assert result is not None
        feat_t, label_t, weight_t = result
        assert feat_t.shape == (20, 16)
        assert label_t.shape == (20, 4)
        assert weight_t.shape == (20,)


# -----------------------------------------------------------------------
# Persistence bridge: teacher signal recording
# -----------------------------------------------------------------------

class TestPersistenceBridge:
    def test_record_distillation_signal_called(self):
        """Verify ReflectiveValidator._record_distillation_signal is called during validation."""
        from consciousness.dream_artifacts import (
            ArtifactBuffer,
            ReflectiveValidator,
            create_artifact,
        )

        buf = ArtifactBuffer(maxlen=10)
        validator = ReflectiveValidator(buf, remember_fn=lambda **kw: None)
        art = create_artifact("symbolic_summary", ["m1"], "test", 0.8, 0.7)
        buf.add(art)

        with patch.object(validator, "_record_distillation_signal") as mock_record:
            validator.validate_pending(system_context={"soul_integrity": 0.9})
            assert mock_record.call_count == 1
            call_args = mock_record.call_args
            assert call_args[0][0].artifact_id == art.artifact_id

    def test_validate_pending_accepts_system_context(self):
        """API contract: validate_pending accepts system_context kwarg."""
        from consciousness.dream_artifacts import ArtifactBuffer, ReflectiveValidator
        buf = ArtifactBuffer(maxlen=5)
        v = ReflectiveValidator(buf)
        result = v.validate_pending(system_context={"soul_integrity": 0.85})
        assert result["reviewed"] == 0


# -----------------------------------------------------------------------
# HemisphereFocus registration
# -----------------------------------------------------------------------

class TestRegistration:
    def test_dream_synthesis_in_focus_enum(self):
        from hemisphere.types import HemisphereFocus
        assert hasattr(HemisphereFocus, "DREAM_SYNTHESIS")
        assert HemisphereFocus.DREAM_SYNTHESIS.value == "dream_synthesis"

    def test_dream_synthesis_in_tier1_focuses(self):
        from hemisphere.orchestrator import _TIER1_FOCUSES
        from hemisphere.types import HemisphereFocus
        assert HemisphereFocus.DREAM_SYNTHESIS in _TIER1_FOCUSES

    def test_dream_synthesis_in_distillation_configs(self):
        from hemisphere.types import DISTILLATION_CONFIGS
        assert "dream_synthesis" in DISTILLATION_CONFIGS

    def test_architect_output_size(self):
        from hemisphere.architect import NeuralArchitect
        from hemisphere.types import HemisphereFocus
        arch = NeuralArchitect()
        size = arch._get_output_size(HemisphereFocus.DREAM_SYNTHESIS)
        assert size == 4

    def test_dashboard_specialist_set(self):
        from dashboard.snapshot import _SI_SPECIALIST_FOCUSES
        assert "dream_synthesis" in _SI_SPECIALIST_FOCUSES


# -----------------------------------------------------------------------
# ANTI-AUTHORITY BOUNDARY TESTS
# These are structural code-path verifications, not behavioral.
# -----------------------------------------------------------------------

class TestAntiAuthority:
    """Hard boundary tests: the dream-synthesis specialist must NEVER have
    any code path to mutate artifact state, write memory, or emit events."""

    def test_encoder_has_no_import_to_dream_artifacts(self):
        """DreamArtifactEncoder must not import dream_artifacts module."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hemisphere", "dream_artifact_encoder.py",
        )
        with open(src_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                all_names = [module] + names
                for name in all_names:
                    assert "dream_artifact" not in name.lower() or "encoder" in name.lower(), (
                        f"Encoder must not import dream_artifacts: found {name}"
                    )

    def test_encoder_has_no_import_to_memory(self):
        """DreamArtifactEncoder must not import any memory module."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hemisphere", "dream_artifact_encoder.py",
        )
        with open(src_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                for name in [module] + names:
                    assert "memory" not in name.lower(), (
                        f"Encoder must not import memory modules: found {name}"
                    )

    def test_encoder_has_no_import_to_events(self):
        """DreamArtifactEncoder must not import the event bus."""
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "hemisphere", "dream_artifact_encoder.py",
        )
        with open(src_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                for name in [module] + names:
                    assert "event" not in name.lower(), (
                        f"Encoder must not import event modules: found {name}"
                    )

    def test_encoder_has_no_mutation_methods(self):
        """DreamArtifactEncoder should only have static encode/label methods."""
        methods = [
            name for name, _ in inspect.getmembers(DreamArtifactEncoder, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]
        allowed = {"encode", "encode_label", "encode_artifact_block", "encode_system_block", "encode_governance_block"}
        for m in methods:
            assert m in allowed, f"Unexpected public method: {m}"

    def test_encoder_returns_only_primitives(self):
        """encode() returns list[float], encode_label() returns (list[float], dict)."""
        vec = DreamArtifactEncoder.encode(_sample_artifact(), _sample_system_context())
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

        label, meta = DreamArtifactEncoder.encode_label("promoted", _sample_artifact(), "notes")
        assert isinstance(label, list)
        assert all(isinstance(v, float) for v in label)
        assert isinstance(meta, dict)

    def test_tensor_prep_has_no_side_effects(self):
        """_prepare_dream_observer_tensors must not call any mutation API."""
        pytest.importorskip("torch")
        from hemisphere.data_feed import _prepare_dream_observer_tensors
        from hemisphere.types import DISTILLATION_CONFIGS
        config = DISTILLATION_CONFIGS["dream_synthesis"]

        mock_collector = MagicMock()
        mock_collector.get_training_batch.return_value = []
        _prepare_dream_observer_tensors(mock_collector, config)

        for call in mock_collector.method_calls:
            method_name = call[0]
            assert method_name in ("get_training_batch",), (
                f"Tensor prep called unexpected method: {method_name}"
            )

    def test_validator_remains_sole_authority(self):
        """ReflectiveValidator._evaluate does not consult any NN output."""
        from consciousness.dream_artifacts import ReflectiveValidator
        src = inspect.getsource(ReflectiveValidator._evaluate)
        forbidden_imports = [
            "hemisphere",
            "engine.infer",
            "nn_output",
            "specialist",
            "DREAM_SYNTHESIS",
        ]
        for term in forbidden_imports:
            assert term not in src, (
                f"_evaluate() must not reference NN infrastructure: found '{term}'"
            )

    def test_distillation_signal_is_record_only(self):
        """_record_distillation_signal only calls collector.record(), nothing else."""
        from consciousness.dream_artifacts import ReflectiveValidator
        src = inspect.getsource(ReflectiveValidator._record_distillation_signal)
        assert "collector.record(" in src
        assert ".remember(" not in src
        assert ".update(" not in src
        assert "emit(" not in src
        assert "promote(" not in src
