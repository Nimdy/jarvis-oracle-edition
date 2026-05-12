"""Phase C: Shadow Jarvis Language Model.

A style embedding network that learns how this specific JARVIS instance
phrases responses, trained on its own language corpus.  At inference time
it retrieves the most stylistically similar past response for comparison
with the live bounded articulator output.

The model never generates text — it learns a style embedding space and
retrieves real verified corpus responses.  Zero hallucination risk.

Cold-start safe: gracefully returns None when corpus is too small.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LANGUAGE_CORPUS_PATH = JARVIS_DIR / "language_corpus" / "examples.jsonl"
LANGUAGE_STYLE_DIR = JARVIS_DIR / "hemispheres" / "language_style"
CORPUS_CACHE_PATH = LANGUAGE_STYLE_DIR / "corpus_cache.pt"

MIN_CORPUS_FOR_TRAINING = 50
EMBEDDING_DIM = 384
STYLE_DIM = 64
INPUT_DIM = EMBEDDING_DIM * 2 + 7 + 8  # 783: query_emb + lead_emb + class_onehot + meta

BOUNDED_RESPONSE_CLASSES = (
    "self_status",
    "self_introspection",
    "memory_recall",
    "recent_learning",
    "recent_research",
    "identity_answer",
    "capability_status",
)
_CLASS_TO_IDX = {c: i for i, c in enumerate(BOUNDED_RESPONSE_CLASSES)}

# ---------------------------------------------------------------------------
# Torch imports (lazy — not all installs have torch)
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Model
# ===================================================================

class ShadowStyleNet(nn.Module if _TORCH_AVAILABLE else object):
    """Small feed-forward that maps (query_emb, lead_emb, class, meta) → style vector."""

    def __init__(self, input_dim: int = INPUT_DIM, style_dim: int = STYLE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, style_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
# Corpus loading
# ===================================================================

@dataclass
class CorpusExample:
    """Minimal parsed corpus example for training."""
    query: str
    response_class: str
    lead: str
    final_answer: str
    frame_confidence: float
    fact_count: int
    section_count: int
    safety_flag_count: int
    has_missing_reason: bool
    lead_length: int
    facts_text_length: int
    is_structurally_healthy: bool


def load_corpus(path: Path | None = None) -> list[CorpusExample]:
    """Load and filter corpus examples from JSONL."""
    path = path or LANGUAGE_CORPUS_PATH
    if not path.exists():
        return []

    examples: list[CorpusExample] = []
    rotated = path.with_suffix(path.suffix + ".1")
    paths = []
    if rotated.exists():
        paths.append(rotated)
    paths.append(path)

    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rc = rec.get("response_class", "")
                    if rc not in _CLASS_TO_IDX:
                        continue
                    if rc == "negative_example":
                        continue
                    conf = float(rec.get("confidence", 0) or 0)
                    if conf < 0.3:
                        continue
                    fa = str(rec.get("final_answer", "") or "")
                    if not fa or len(fa) < 10:
                        continue

                    mf = rec.get("meaning_frame") or {}
                    facts = mf.get("facts", []) or []
                    facts_text = " ".join(str(f) for f in facts)

                    examples.append(CorpusExample(
                        query=str(rec.get("query", "") or "")[:500],
                        response_class=rc,
                        lead=str(mf.get("lead", "") or "")[:300],
                        final_answer=fa[:2000],
                        frame_confidence=max(0.0, min(1.0, float(mf.get("frame_confidence", 0) or 0))),
                        fact_count=min(len(facts), 20),
                        section_count=min(int(mf.get("section_count", 0) or 0), 10),
                        safety_flag_count=min(len(mf.get("safety_flags", []) or []), 5),
                        has_missing_reason=bool(mf.get("missing_reason")),
                        lead_length=min(len(str(mf.get("lead", "") or "")), 300),
                        facts_text_length=min(len(facts_text), 2000),
                        is_structurally_healthy=bool(mf.get("frame_confidence", 0) or 0 >= 0.4),
                    ))
        except Exception:
            logger.warning("Shadow language model: failed to read %s", p, exc_info=True)

    return examples


# ===================================================================
# Embedding + feature preparation
# ===================================================================

def _get_sbert_model():
    """Get the shared SentenceTransformer, reusing VectorStore's if available."""
    if not _SBERT_AVAILABLE:
        return None
    try:
        from memory.vector_store import VectorStore
        # If VectorStore has been instantiated, reuse its model
        # to avoid loading the same 80MB model twice
        import gc
        for obj in gc.get_referrers(VectorStore):
            if isinstance(obj, VectorStore) and getattr(obj, "_model", None) is not None:
                return obj._model
    except Exception:
        pass
    # Fall back to loading our own instance
    try:
        from config import get_models_dir
        _cache = str(get_models_dir() / "huggingface")
    except Exception:
        _cache = None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu",
                                   cache_folder=_cache, local_files_only=True)
    except Exception:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2", device="cpu", cache_folder=_cache)
        except Exception:
            logger.warning("Shadow language model: failed to load SentenceTransformer")
            return None


def _encode_texts(model, texts: list[str]) -> Any:
    """Batch-encode texts to embeddings."""
    import torch
    if not texts:
        return torch.zeros(0, EMBEDDING_DIM)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    return embeddings


def _build_class_onehot(response_class: str) -> list[float]:
    """One-hot encode a response class."""
    vec = [0.0] * len(BOUNDED_RESPONSE_CLASSES)
    idx = _CLASS_TO_IDX.get(response_class)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _build_meta_features(ex: CorpusExample) -> list[float]:
    """Extract 8 normalized meta features from a corpus example."""
    return [
        ex.frame_confidence,
        min(ex.fact_count / 20.0, 1.0),
        min(ex.section_count / 10.0, 1.0),
        min(ex.safety_flag_count / 5.0, 1.0),
        1.0 if ex.has_missing_reason else 0.0,
        min(ex.lead_length / 300.0, 1.0),
        min(ex.facts_text_length / 2000.0, 1.0),
        1.0 if ex.is_structurally_healthy else 0.0,
    ]


def prepare_training_data(
    corpus: list[CorpusExample],
    sbert_model: Any,
) -> dict[str, Any] | None:
    """Prepare input tensors, answer embeddings, and triplets for training.

    Returns dict with keys: input_tensors, answer_embeddings, triplets, examples
    or None if data is insufficient.
    """
    import torch
    import torch.nn.functional as F

    if len(corpus) < MIN_CORPUS_FOR_TRAINING:
        return None

    # Batch encode queries and leads
    queries = [ex.query for ex in corpus]
    leads = [ex.lead if ex.lead else " " for ex in corpus]
    answers = [ex.final_answer for ex in corpus]

    query_embs = _encode_texts(sbert_model, queries)   # (N, 384)
    lead_embs = _encode_texts(sbert_model, leads)       # (N, 384)
    answer_embs = _encode_texts(sbert_model, answers)   # (N, 384) — for triplet building

    # Build full input tensors
    rows = []
    for i, ex in enumerate(corpus):
        class_oh = _build_class_onehot(ex.response_class)
        meta = _build_meta_features(ex)
        row = torch.cat([
            query_embs[i],
            lead_embs[i],
            torch.tensor(class_oh, dtype=torch.float32),
            torch.tensor(meta, dtype=torch.float32),
        ])
        rows.append(row)

    input_tensors = torch.stack(rows)  # (N, 783)

    # Normalize answer embeddings for cosine similarity
    answer_norm = F.normalize(answer_embs, dim=1)

    # Build triplets: (anchor_idx, positive_idx, negative_idx)
    triplets = []
    # Group by response class
    class_indices: dict[str, list[int]] = {}
    for i, ex in enumerate(corpus):
        class_indices.setdefault(ex.response_class, []).append(i)

    cos_sim = answer_norm @ answer_norm.T  # (N, N)

    for rc, indices in class_indices.items():
        if len(indices) < 2:
            continue
        other_indices = [i for i in range(len(corpus)) if corpus[i].response_class != rc]
        if not other_indices:
            continue

        for anchor in indices:
            # Positive: same class, highest answer similarity (excluding self)
            best_pos = -1
            best_sim = -2.0
            for cand in indices:
                if cand == anchor:
                    continue
                sim = float(cos_sim[anchor, cand])
                if sim > best_sim:
                    best_sim = sim
                    best_pos = cand
            if best_pos < 0:
                continue

            # Negative: different class, random selection
            neg = other_indices[anchor % len(other_indices)]
            triplets.append((anchor, best_pos, neg))

    if len(triplets) < 10:
        logger.info("Shadow language model: only %d triplets, need >= 10", len(triplets))
        return None

    return {
        "input_tensors": input_tensors,
        "answer_embeddings": answer_embs,
        "triplets": triplets,
        "examples": corpus,
    }


# ===================================================================
# Trainer
# ===================================================================

class ShadowLanguageTrainer:
    """Trains the ShadowStyleNet on language corpus data."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._lock = threading.Lock()
        self._last_corpus_size = 0
        self._last_train_time = 0.0

    def should_train(self, corpus_size: int) -> bool:
        """Check if training should run."""
        if corpus_size < MIN_CORPUS_FOR_TRAINING:
            return False
        if corpus_size <= self._last_corpus_size:
            return False
        if corpus_size - self._last_corpus_size < 20 and self._last_train_time > 0:
            return False
        return True

    def train(
        self,
        corpus: list[CorpusExample] | None = None,
        sbert_model: Any = None,
    ) -> dict[str, Any] | None:
        """Train the style network. Returns model info dict or None on failure."""
        import torch
        import torch.optim as optim

        if not self._lock.acquire(blocking=False):
            return None
        try:
            if corpus is None:
                corpus = load_corpus()
            if sbert_model is None:
                sbert_model = _get_sbert_model()
            if sbert_model is None:
                logger.warning("Shadow language model: no SentenceTransformer available")
                return None

            if len(corpus) < MIN_CORPUS_FOR_TRAINING:
                logger.info(
                    "Shadow language model deferred: %d examples < %d minimum",
                    len(corpus), MIN_CORPUS_FOR_TRAINING,
                )
                return None

            data = prepare_training_data(corpus, sbert_model)
            if data is None:
                return None

            input_tensors = data["input_tensors"].to(self._device)
            triplets = data["triplets"]

            # Build model
            model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
            model.to(self._device)
            model.train()

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            triplet_loss_fn = torch.nn.TripletMarginLoss(margin=0.3)

            # Training loop
            n_triplets = len(triplets)
            batch_size = min(16, n_triplets)
            best_loss = float("inf")
            patience = 5
            patience_counter = 0
            total_epochs = 30

            for epoch in range(total_epochs):
                # Shuffle triplets
                perm = torch.randperm(n_triplets)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n_triplets, batch_size):
                    batch_idx = perm[start:start + batch_size]
                    anchors = []
                    positives = []
                    negatives = []
                    for idx in batch_idx:
                        a, p, n = triplets[int(idx)]
                        anchors.append(a)
                        positives.append(p)
                        negatives.append(n)

                    anchor_out = model(input_tensors[anchors])
                    pos_out = model(input_tensors[positives])
                    neg_out = model(input_tensors[negatives])

                    loss = triplet_loss_fn(anchor_out, pos_out, neg_out)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            model.eval()

            # Compute style vectors for all corpus examples
            with torch.no_grad():
                style_vectors = model(input_tensors).cpu()

            self._last_corpus_size = len(corpus)
            self._last_train_time = time.time()

            logger.info(
                "Shadow language model trained: %d examples, %d triplets, "
                "loss=%.4f, %d epochs",
                len(corpus), n_triplets, best_loss, epoch + 1,
            )

            return {
                "model": model,
                "style_vectors": style_vectors,
                "answer_embeddings": data["answer_embeddings"].cpu(),
                "examples": corpus,
                "loss": best_loss,
                "n_triplets": n_triplets,
                "n_examples": len(corpus),
                "epochs": epoch + 1,
                "train_time": self._last_train_time,
            }
        except Exception:
            logger.warning("Shadow language model training failed", exc_info=True)
            return None
        finally:
            self._lock.release()


# ===================================================================
# Corpus index + inference
# ===================================================================

@dataclass
class CorpusIndex:
    """Pre-computed style vectors and metadata for retrieval."""
    style_vectors: Any = None      # (N, 64) tensor
    answer_embeddings: Any = None  # (N, 384) tensor
    examples: list[CorpusExample] = field(default_factory=list)
    class_indices: dict[str, list[int]] = field(default_factory=dict)
    built_at: float = 0.0
    corpus_size: int = 0

    def build_class_index(self) -> None:
        self.class_indices = {}
        for i, ex in enumerate(self.examples):
            self.class_indices.setdefault(ex.response_class, []).append(i)


class ShadowLanguageInference:
    """Inference engine for shadow language model."""

    def __init__(self) -> None:
        self._model: Any = None
        self._index: CorpusIndex | None = None
        self._sbert: Any = None
        self._lock = threading.Lock()
        self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._model is not None and self._index is not None

    def load(self, train_result: dict[str, Any]) -> None:
        """Load from a training result."""
        with self._lock:
            self._model = train_result["model"]
            self._model.eval()
            idx = CorpusIndex(
                style_vectors=train_result["style_vectors"],
                answer_embeddings=train_result["answer_embeddings"],
                examples=train_result["examples"],
                built_at=time.time(),
                corpus_size=train_result["n_examples"],
            )
            idx.build_class_index()
            self._index = idx
            self._available = True
            logger.info("Shadow language inference loaded: %d examples", idx.corpus_size)

    def load_from_disk(self, model_path: str, index_path: str) -> bool:
        """Load saved model and index from disk."""
        if not _TORCH_AVAILABLE:
            return False
        import torch
        try:
            saved = torch.load(index_path, map_location="cpu", weights_only=False)
            model = ShadowStyleNet(INPUT_DIM, STYLE_DIM)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()

            examples = [CorpusExample(**e) for e in saved["examples"]]
            idx = CorpusIndex(
                style_vectors=saved["style_vectors"],
                answer_embeddings=saved["answer_embeddings"],
                examples=examples,
                built_at=saved.get("built_at", 0.0),
                corpus_size=len(examples),
            )
            idx.build_class_index()

            with self._lock:
                self._model = model
                self._index = idx
                self._available = True
            logger.info("Shadow language model restored from disk: %d examples", len(examples))
            return True
        except Exception:
            logger.warning("Shadow language model restore failed", exc_info=True)
            return False

    def save_to_disk(self, model_path: str, index_path: str) -> bool:
        """Save model and index to disk."""
        if not _TORCH_AVAILABLE or not self._available:
            return False
        import torch
        from dataclasses import asdict
        try:
            LANGUAGE_STYLE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), model_path)
            torch.save({
                "style_vectors": self._index.style_vectors,
                "answer_embeddings": self._index.answer_embeddings,
                "examples": [asdict(e) for e in self._index.examples],
                "built_at": self._index.built_at,
            }, index_path)
            return True
        except Exception:
            logger.warning("Shadow language model save failed", exc_info=True)
            return False

    def shadow_generate(
        self,
        query: str,
        meaning_frame: Any,
        response_class: str,
    ) -> str | None:
        """Generate a shadow response by retrieving the closest corpus match.

        Returns the final_answer of the most stylistically similar corpus
        example for the given response_class, or None if unavailable.
        """
        if not self._available or not _TORCH_AVAILABLE:
            return None

        if response_class not in _CLASS_TO_IDX:
            return None

        idx = self._index
        if idx is None or response_class not in idx.class_indices:
            return None

        class_examples = idx.class_indices[response_class]
        if not class_examples:
            return None

        try:
            import torch
            import torch.nn.functional as F

            sbert = self._get_sbert()
            if sbert is None:
                return None

            # Build input vector
            mf = meaning_frame
            if hasattr(mf, "to_dict"):
                mf_dict = mf.to_dict()
            elif isinstance(mf, dict):
                mf_dict = mf
            else:
                mf_dict = {}

            lead = str(mf_dict.get("lead", "") or "")
            facts = mf_dict.get("facts", []) or []
            facts_text = " ".join(str(f) for f in facts)

            query_emb = torch.tensor(sbert.encode(query or " "), dtype=torch.float32)
            lead_emb = torch.tensor(sbert.encode(lead or " "), dtype=torch.float32)
            class_oh = torch.tensor(_build_class_onehot(response_class), dtype=torch.float32)
            meta = torch.tensor([
                float(mf_dict.get("frame_confidence", 0) or 0),
                min(len(facts) / 20.0, 1.0),
                min(int(mf_dict.get("section_count", 0) or 0) / 10.0, 1.0),
                min(len(mf_dict.get("safety_flags", []) or []) / 5.0, 1.0),
                1.0 if mf_dict.get("missing_reason") else 0.0,
                min(len(lead) / 300.0, 1.0),
                min(len(facts_text) / 2000.0, 1.0),
                1.0 if float(mf_dict.get("frame_confidence", 0) or 0) >= 0.4 else 0.0,
            ], dtype=torch.float32)

            input_vec = torch.cat([query_emb, lead_emb, class_oh, meta]).unsqueeze(0)

            with torch.no_grad():
                style_vec = self._model(input_vec)  # (1, 64)

            # Compare with class examples
            class_styles = idx.style_vectors[class_examples]  # (K, 64)
            style_norm = F.normalize(style_vec, dim=1)
            class_norm = F.normalize(class_styles, dim=1)
            similarities = (style_norm @ class_norm.T).squeeze(0)  # (K,)

            best_local_idx = int(torch.argmax(similarities).item())
            best_global_idx = class_examples[best_local_idx]
            return idx.examples[best_global_idx].final_answer

        except Exception:
            logger.warning("Shadow language inference failed", exc_info=True)
            return None

    def _get_sbert(self):
        if self._sbert is None:
            self._sbert = _get_sbert_model()
        return self._sbert

    def get_stats(self) -> dict[str, Any]:
        """Return status for dashboard/snapshot."""
        if not self._available or self._index is None:
            return {
                "available": False,
                "corpus_size": 0,
                "trained": False,
            }
        idx = self._index
        return {
            "available": True,
            "corpus_size": idx.corpus_size,
            "trained": True,
            "built_at": idx.built_at,
            "class_counts": {k: len(v) for k, v in idx.class_indices.items()},
        }


# ===================================================================
# Module-level singleton
# ===================================================================

shadow_language_trainer = ShadowLanguageTrainer()
shadow_language_inference = ShadowLanguageInference()
