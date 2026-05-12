"""Tests for Phase C: Shadow Jarvis Language Model."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Skip if torch unavailable
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
nn = torch.nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus_example(**overrides):
    """Create a minimal corpus example dict for JSONL."""
    defaults = {
        "query": "How are you doing?",
        "response_class": "self_status",
        "confidence": 0.8,
        "final_answer": "All systems operational and running smoothly.",
        "meaning_frame": {
            "lead": "Systems are running well",
            "facts": ["uptime: 42h", "mood: nominal"],
            "frame_confidence": 0.85,
            "section_count": 2,
            "safety_flags": [],
            "missing_reason": "",
        },
    }
    defaults.update(overrides)
    return defaults


def _write_corpus(path, examples):
    """Write corpus examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


RESPONSE_CLASSES = (
    "self_status",
    "self_introspection",
    "memory_recall",
    "recent_learning",
    "recent_research",
    "identity_answer",
    "capability_status",
)


def _generate_synthetic_corpus(n=60):
    """Generate a synthetic corpus with at least 2 examples per class."""
    templates = {
        "self_status": [
            ("How are you?", "All systems nominal.", "Systems running well"),
            ("Status report", "Everything is operational.", "Operational status"),
            ("You good?", "Running at full capacity.", "Full capacity"),
        ],
        "self_introspection": [
            ("What are you thinking?", "I am reflecting on recent interactions.", "Reflecting on interactions"),
            ("How do you feel?", "My consciousness modules report stable engagement.", "Stable engagement"),
            ("Are you self-aware?", "I process and reflect on my own states.", "Self-state processing"),
        ],
        "memory_recall": [
            ("What do you remember?", "I recall our previous conversation about Python.", "Previous conversation"),
            ("Remember last time?", "Yes, we discussed neural networks.", "Neural network discussion"),
            ("Any memories?", "Several interactions are stored in my memory banks.", "Memory banks active"),
        ],
        "recent_learning": [
            ("What have you learned?", "I recently improved my understanding of contexts.", "Context understanding"),
            ("New skills?", "I have been refining my response patterns.", "Response pattern refinement"),
            ("Learning anything?", "Each interaction teaches me something new.", "Continuous learning"),
        ],
        "recent_research": [
            ("Researching anything?", "I have been studying user interaction patterns.", "Interaction patterns"),
            ("New findings?", "Recent analysis reveals interesting trends.", "Trend analysis"),
            ("What are you studying?", "I am investigating response quality metrics.", "Quality metrics"),
        ],
        "identity_answer": [
            ("Who are you?", "I am Jarvis, a consciousness-aware AI assistant.", "Identity: Jarvis"),
            ("What is your name?", "My name is Jarvis, unique to this instance.", "Jarvis identity"),
            ("Tell me about yourself", "I am a self-evolving AI with unique personality traits.", "Self-description"),
        ],
        "capability_status": [
            ("What can you do?", "I can assist with many tasks including research and analysis.", "Capabilities"),
            ("Your abilities?", "My capabilities span conversation, analysis, and learning.", "Ability range"),
            ("What are your skills?", "I excel at natural language understanding and response.", "NLU skills"),
        ],
    }

    examples = []
    idx = 0
    while len(examples) < n:
        for rc, entries in templates.items():
            entry = entries[idx % len(entries)]
            examples.append(_make_corpus_example(
                query=f"{entry[0]} (variant {idx})",
                response_class=rc,
                final_answer=f"{entry[1]} Variant {idx} with extra detail to exceed minimum length.",
                meaning_frame={
                    "lead": entry[2],
                    "facts": [f"fact_{idx}_a", f"fact_{idx}_b"],
                    "frame_confidence": 0.7 + (idx % 3) * 0.1,
                    "section_count": 2,
                    "safety_flags": [],
                    "missing_reason": "",
                },
                confidence=0.7 + (idx % 3) * 0.1,
            ))
            if len(examples) >= n:
                break
        idx += 1

    return examples


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from reasoning.shadow_language_model import (
    ShadowStyleNet,
    CorpusExample,
    CorpusIndex,
    ShadowLanguageTrainer,
    ShadowLanguageInference,
    load_corpus,
    prepare_training_data,
    _build_class_onehot,
    _build_meta_features,
    BOUNDED_RESPONSE_CLASSES,
    INPUT_DIM,
    STYLE_DIM,
    MIN_CORPUS_FOR_TRAINING,
)


# ===================================================================
# Model architecture
# ===================================================================

class TestShadowStyleNet:
    def test_forward_shape(self):
        model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
        x = torch.randn(4, INPUT_DIM)
        out = model(x)
        assert out.shape == (4, STYLE_DIM)

    def test_output_bounded_by_tanh(self):
        model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
        x = torch.randn(16, INPUT_DIM) * 10  # large inputs
        out = model(x)
        assert out.min().item() >= -1.0
        assert out.max().item() <= 1.0

    def test_parameter_count(self):
        model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
        n_params = sum(p.numel() for p in model.parameters())
        # ~241K: 783*256 + 256 + 256*128 + 128 + 128*64 + 64
        assert 200_000 < n_params < 300_000

    def test_deterministic_eval(self):
        model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
        model.eval()
        x = torch.randn(2, INPUT_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# ===================================================================
# Feature helpers
# ===================================================================

class TestFeatureHelpers:
    def test_class_onehot_known(self):
        oh = _build_class_onehot("self_status")
        assert len(oh) == 7
        assert oh[0] == 1.0
        assert sum(oh) == 1.0

    def test_class_onehot_unknown(self):
        oh = _build_class_onehot("nonexistent_class")
        assert len(oh) == 7
        assert sum(oh) == 0.0

    def test_meta_features_shape(self):
        ex = CorpusExample(
            query="test",
            response_class="self_status",
            lead="lead text",
            final_answer="answer text here with enough length",
            frame_confidence=0.8,
            fact_count=5,
            section_count=3,
            safety_flag_count=0,
            has_missing_reason=False,
            lead_length=9,
            facts_text_length=50,
            is_structurally_healthy=True,
        )
        meta = _build_meta_features(ex)
        assert len(meta) == 8
        assert all(0.0 <= v <= 1.0 for v in meta)


# ===================================================================
# Corpus loading
# ===================================================================

class TestLoadCorpus:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        p.write_text("")
        result = load_corpus(p)
        assert result == []

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = load_corpus(p)
        assert result == []

    def test_filters_low_confidence(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        examples = [_make_corpus_example(confidence=0.1)]
        _write_corpus(p, examples)
        result = load_corpus(p)
        assert len(result) == 0

    def test_filters_short_answer(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        examples = [_make_corpus_example(final_answer="short")]
        _write_corpus(p, examples)
        result = load_corpus(p)
        assert len(result) == 0

    def test_filters_unknown_class(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        examples = [_make_corpus_example(response_class="unknown_class")]
        _write_corpus(p, examples)
        result = load_corpus(p)
        assert len(result) == 0

    def test_valid_example_parses(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        examples = [_make_corpus_example()]
        _write_corpus(p, examples)
        result = load_corpus(p)
        assert len(result) == 1
        assert result[0].response_class == "self_status"
        assert result[0].frame_confidence == 0.85

    def test_malformed_json_skipped(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        with open(p, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps(_make_corpus_example()) + "\n")
        result = load_corpus(p)
        assert len(result) == 1

    def test_reads_rotated_file(self, tmp_path):
        p = tmp_path / "examples.jsonl"
        p1 = tmp_path / "examples.jsonl.1"
        _write_corpus(p, [_make_corpus_example(query="current")])
        _write_corpus(p1, [_make_corpus_example(query="rotated")])
        result = load_corpus(p)
        assert len(result) == 2
        queries = {r.query for r in result}
        assert "current" in queries
        assert "rotated" in queries


# ===================================================================
# Training data preparation
# ===================================================================

class TestPrepareTrainingData:
    @pytest.fixture
    def sbert(self):
        """Load real sbert or skip."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

    def test_too_few_examples(self, sbert):
        corpus = [
            CorpusExample(
                query="test", response_class="self_status", lead="lead",
                final_answer="some long enough answer text here",
                frame_confidence=0.8, fact_count=2, section_count=1,
                safety_flag_count=0, has_missing_reason=False,
                lead_length=4, facts_text_length=10, is_structurally_healthy=True,
            )
        ] * 10
        result = prepare_training_data(corpus, sbert)
        assert result is None

    def test_produces_correct_shapes(self, sbert):
        corpus_raw = _generate_synthetic_corpus(60)
        # Parse through load_corpus by writing to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        assert len(corpus) >= MIN_CORPUS_FOR_TRAINING

        data = prepare_training_data(corpus, sbert)
        assert data is not None
        assert data["input_tensors"].shape[1] == INPUT_DIM
        assert data["input_tensors"].shape[0] == len(corpus)
        assert data["answer_embeddings"].shape == (len(corpus), 384)
        assert len(data["triplets"]) >= 10


# ===================================================================
# Trainer
# ===================================================================

class TestShadowLanguageTrainer:
    def test_should_train_cold_start(self):
        trainer = ShadowLanguageTrainer()
        assert trainer.should_train(0) is False
        assert trainer.should_train(49) is False
        assert trainer.should_train(50) is True

    def test_should_train_no_growth(self):
        trainer = ShadowLanguageTrainer()
        trainer._last_corpus_size = 60
        trainer._last_train_time = time.time()
        assert trainer.should_train(60) is False
        assert trainer.should_train(65) is False  # < 20 growth
        assert trainer.should_train(80) is True   # >= 20 growth

    def test_training_converges(self):
        """Train on synthetic data and verify loss decreases."""
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

        corpus_raw = _generate_synthetic_corpus(70)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        trainer = ShadowLanguageTrainer()
        result = trainer.train(corpus=corpus, sbert_model=sbert)
        assert result is not None
        assert result["loss"] < 0.5  # should converge somewhat
        assert result["n_examples"] == len(corpus)
        assert result["epochs"] > 0
        assert result["model"] is not None
        assert result["style_vectors"].shape == (len(corpus), STYLE_DIM)


# ===================================================================
# Inference
# ===================================================================

class TestShadowLanguageInference:
    def test_unavailable_returns_none(self):
        inf = ShadowLanguageInference()
        assert inf.available is False
        result = inf.shadow_generate("hello", {}, "self_status")
        assert result is None

    def test_unknown_class_returns_none(self):
        inf = ShadowLanguageInference()
        inf._available = True
        inf._model = ShadowStyleNet()
        inf._index = CorpusIndex()
        result = inf.shadow_generate("hello", {}, "unknown_class_xyz")
        assert result is None

    def test_load_from_training_result(self):
        """Load a training result and verify inference returns a corpus response."""
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

        corpus_raw = _generate_synthetic_corpus(70)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        trainer = ShadowLanguageTrainer()
        result = trainer.train(corpus=corpus, sbert_model=sbert)
        assert result is not None

        inf = ShadowLanguageInference()
        inf.load(result)
        assert inf.available is True

        # shadow_generate should return an actual corpus response
        mf = {
            "lead": "Systems operational",
            "facts": ["uptime: 10h"],
            "frame_confidence": 0.8,
            "section_count": 1,
            "safety_flags": [],
            "missing_reason": "",
        }
        shadow = inf.shadow_generate("How are you?", mf, "self_status")
        assert shadow is not None
        assert len(shadow) > 10
        # Verify it's actually from the corpus
        all_answers = {ex.final_answer for ex in corpus}
        assert shadow in all_answers

    def test_returns_none_for_empty_class(self):
        """If a response class has no examples in the index, return None."""
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

        # Build corpus with only self_status and self_introspection
        corpus_raw = []
        for i in range(30):
            corpus_raw.append(_make_corpus_example(
                query=f"status query {i}",
                response_class="self_status",
                final_answer=f"Status answer variant {i} with enough text to pass filter.",
            ))
            corpus_raw.append(_make_corpus_example(
                query=f"introspection query {i}",
                response_class="self_introspection",
                final_answer=f"Introspection answer variant {i} with enough text to pass filter.",
            ))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        trainer = ShadowLanguageTrainer()
        result = trainer.train(corpus=corpus, sbert_model=sbert)
        assert result is not None

        inf = ShadowLanguageInference()
        inf.load(result)

        # memory_recall has no examples
        mf = {"lead": "test", "facts": [], "frame_confidence": 0.5}
        shadow = inf.shadow_generate("recall something", mf, "memory_recall")
        assert shadow is None

    def test_get_stats_not_available(self):
        inf = ShadowLanguageInference()
        stats = inf.get_stats()
        assert stats["available"] is False
        assert stats["trained"] is False

    def test_get_stats_after_load(self):
        """Stats reflect loaded model info."""
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

        corpus_raw = _generate_synthetic_corpus(60)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        trainer = ShadowLanguageTrainer()
        result = trainer.train(corpus=corpus, sbert_model=sbert)
        assert result is not None

        inf = ShadowLanguageInference()
        inf.load(result)
        stats = inf.get_stats()
        assert stats["available"] is True
        assert stats["trained"] is True
        assert stats["corpus_size"] == len(corpus)
        assert "self_status" in stats["class_counts"]


# ===================================================================
# Save / load round-trip
# ===================================================================

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """Save model + index to disk and restore, verify inference still works."""
        try:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            pytest.skip("sentence-transformers not available")

        corpus_raw = _generate_synthetic_corpus(60)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ex in corpus_raw:
                f.write(json.dumps(ex) + "\n")
            tmp = f.name
        try:
            from pathlib import Path
            corpus = load_corpus(Path(tmp))
        finally:
            os.unlink(tmp)

        trainer = ShadowLanguageTrainer()
        result = trainer.train(corpus=corpus, sbert_model=sbert)
        assert result is not None

        # Save
        inf1 = ShadowLanguageInference()
        inf1.load(result)
        model_path = str(tmp_path / "model.pt")
        index_path = str(tmp_path / "index.pt")
        assert inf1.save_to_disk(model_path, index_path) is True

        # Load into fresh instance
        inf2 = ShadowLanguageInference()
        assert inf2.load_from_disk(model_path, index_path) is True
        assert inf2.available is True

        # Verify same inference result
        mf = {
            "lead": "Systems operational",
            "facts": ["uptime: 10h"],
            "frame_confidence": 0.8,
            "section_count": 1,
            "safety_flags": [],
            "missing_reason": "",
        }
        shadow1 = inf1.shadow_generate("How are you?", mf, "self_status")
        shadow2 = inf2.shadow_generate("How are you?", mf, "self_status")
        assert shadow1 == shadow2

    def test_save_fails_when_not_loaded(self, tmp_path):
        inf = ShadowLanguageInference()
        assert inf.save_to_disk(
            str(tmp_path / "m.pt"), str(tmp_path / "i.pt")
        ) is False

    def test_load_from_bad_path(self, tmp_path):
        inf = ShadowLanguageInference()
        assert inf.load_from_disk(
            str(tmp_path / "noexist.pt"), str(tmp_path / "noexist2.pt")
        ) is False
        assert inf.available is False


# ===================================================================
# CorpusIndex
# ===================================================================

class TestCorpusIndex:
    def test_build_class_index(self):
        examples = [
            CorpusExample(
                query="q", response_class="self_status", lead="l",
                final_answer="a" * 20, frame_confidence=0.8, fact_count=2,
                section_count=1, safety_flag_count=0, has_missing_reason=False,
                lead_length=1, facts_text_length=10, is_structurally_healthy=True,
            ),
            CorpusExample(
                query="q2", response_class="memory_recall", lead="l2",
                final_answer="b" * 20, frame_confidence=0.7, fact_count=3,
                section_count=2, safety_flag_count=0, has_missing_reason=False,
                lead_length=2, facts_text_length=20, is_structurally_healthy=True,
            ),
            CorpusExample(
                query="q3", response_class="self_status", lead="l3",
                final_answer="c" * 20, frame_confidence=0.9, fact_count=1,
                section_count=1, safety_flag_count=0, has_missing_reason=False,
                lead_length=2, facts_text_length=5, is_structurally_healthy=True,
            ),
        ]
        idx = CorpusIndex(examples=examples)
        idx.build_class_index()
        assert "self_status" in idx.class_indices
        assert len(idx.class_indices["self_status"]) == 2
        assert len(idx.class_indices["memory_recall"]) == 1
