"""Memory Ranker NN — learned reranking of memory retrieval candidates.

Replaces the hardcoded scoring formula in _hybrid_search() with a small MLP
that learns from retrieval telemetry: which memories were injected into the
prompt AND led to successful conversations.

The heuristic formula is preserved as feature dim 3 and as the fallback scorer
when the ranker has insufficient training data.

Architecture: MLP 12 -> 32 -> 16 -> 1 (sigmoid), ~700 params, CPU inference.
Trained during dream/sleep cycles from MemoryRetrievalLog pairs.
Persisted at ~/.jarvis/memory_ranker.pt
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
RANKER_PATH = JARVIS_DIR / "memory_ranker.pt"

INPUT_DIM = 12
MIN_TRAINING_PAIRS = 50
MAX_TRAINING_PAIRS = 2000
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 0.001
FALLBACK_THRESHOLD = 0.4  # auto-disable ranker if success rate drops below this
REENABLE_COOLDOWN_S = 600.0  # 10 min cooldown after auto-disable before re-enable
FLAP_WINDOW = 3  # disable permanently after this many auto-disables in a session
BASELINE_REFRESH_INTERVAL = 50  # re-sample heuristic baseline every N outcomes

_torch_available = False
try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass


def _build_ranker_model() -> Any:
    """Build the 3-layer MLP ranker."""
    if not _torch_available:
        return None
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


class MemoryRanker:
    """Learned memory retrieval ranker with heuristic fallback."""

    _instance: MemoryRanker | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._trained = False
        self._enabled = True
        self._train_count = 0
        self._last_train_time: float = 0.0
        self._last_loss: float = 0.0
        self._last_accuracy: float = 0.0
        self._baseline_success_rate: float | None = None
        self._eval_window: list[bool] = []
        self._disable_count = 0
        self._last_disable_time: float = 0.0
        self._disable_reason: str = ""
        self._permanently_disabled = False
        self._baseline_sample_counter = 0

        if _torch_available:
            self._model = _build_ranker_model()

    @classmethod
    def get_instance(cls) -> MemoryRanker:
        if cls._instance is None:
            cls._instance = MemoryRanker()
        return cls._instance

    def is_ready(self) -> bool:
        """True if the ranker is trained, enabled, and PyTorch is available."""
        return _torch_available and self._model is not None and self._trained and self._enabled

    def score(self, features: list[float]) -> float:
        """Score a single candidate. Returns float in [0,1]."""
        if not self.is_ready():
            return features[3] if len(features) > 3 else 0.0

        try:
            t = torch.tensor([features[:INPUT_DIM]], dtype=torch.float32)
            with torch.no_grad():
                return float(self._model(t).item())
        except Exception:
            return features[3] if len(features) > 3 else 0.0

    def score_batch(self, feature_batch: list[list[float]]) -> list[float]:
        """Score multiple candidates in one forward pass."""
        if not self.is_ready() or not feature_batch:
            return [f[3] if len(f) > 3 else 0.0 for f in feature_batch]

        try:
            t = torch.tensor(
                [f[:INPUT_DIM] for f in feature_batch],
                dtype=torch.float32,
            )
            with torch.no_grad():
                scores = self._model(t).squeeze(-1)
            return scores.tolist() if scores.dim() > 0 else [float(scores.item())]
        except Exception:
            return [f[3] if len(f) > 3 else 0.0 for f in feature_batch]

    def train_from_pairs(
        self,
        features: list[list[float]],
        labels: list[float],
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
    ) -> dict[str, Any]:
        """Train the ranker from retrieval telemetry pairs.

        Returns training metrics: {loss, accuracy, pairs, epochs, duration_ms}.
        """
        if not _torch_available or self._model is None:
            return {"error": "PyTorch not available"}

        n = len(features)
        if n < MIN_TRAINING_PAIRS:
            return {"error": f"Need {MIN_TRAINING_PAIRS} pairs, have {n}"}

        start = time.time()

        X = torch.tensor(
            [f[:INPUT_DIM] for f in features[-MAX_TRAINING_PAIRS:]],
            dtype=torch.float32,
        )
        Y = torch.tensor(
            labels[-MAX_TRAINING_PAIRS:],
            dtype=torch.float32,
        ).unsqueeze(1)

        n_samples = X.shape[0]
        split = max(1, int(n_samples * 0.8))
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        self._model.train()
        for epoch in range(epochs):
            perm = torch.randperm(X_train.shape[0])
            epoch_loss = 0.0
            batches = 0

            for i in range(0, X_train.shape[0], DEFAULT_BATCH_SIZE):
                idx = perm[i:i + DEFAULT_BATCH_SIZE]
                batch_x, batch_y = X_train[idx], Y_train[idx]

                optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            if X_val.shape[0] > 0:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(X_val)
                    val_loss = criterion(val_pred, Y_val).item()
                self._model.train()

                if val_loss < best_val_loss - 0.001:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self._model.eval()
        with torch.no_grad():
            all_pred = self._model(X)
            final_loss = criterion(all_pred, Y).item()
            binary_pred = (all_pred > 0.5).float()
            binary_label = (Y > 0.5).float()
            accuracy = (binary_pred == binary_label).float().mean().item()

        self._trained = True
        self._train_count += 1
        self._last_train_time = time.time()
        self._last_loss = final_loss
        self._last_accuracy = accuracy

        duration_ms = (time.time() - start) * 1000.0

        self.save()

        result = {
            "loss": round(final_loss, 4),
            "accuracy": round(accuracy, 4),
            "pairs": n_samples,
            "epochs": epoch + 1,
            "duration_ms": round(duration_ms, 1),
            "train_count": self._train_count,
        }
        logger.info("Memory ranker trained: %s", result)
        return result

    def record_outcome(self, success: bool) -> None:
        """Track ranker-scored retrieval outcomes for auto-disable + re-enable."""
        self._eval_window.append(success)
        if len(self._eval_window) > 100:
            self._eval_window = self._eval_window[-100:]

        if self._permanently_disabled:
            return

        if not self._enabled:
            elapsed = time.time() - self._last_disable_time
            if elapsed >= REENABLE_COOLDOWN_S:
                recent = self._eval_window[-20:]
                if len(recent) >= 20 and sum(recent) / len(recent) >= FALLBACK_THRESHOLD:
                    self._enabled = True
                    self._disable_reason = ""
                    logger.info("Memory ranker re-enabled after cooldown (recent rate %.2f)",
                                sum(recent) / len(recent))
            return

        if len(self._eval_window) >= 50:
            rate = sum(self._eval_window) / len(self._eval_window)
            if self._baseline_success_rate is not None and rate < self._baseline_success_rate * 0.8:
                self._enabled = False
                self._disable_count += 1
                self._last_disable_time = time.time()
                self._disable_reason = (
                    f"success rate {rate:.2f} < baseline {self._baseline_success_rate:.2f} * 0.8"
                )
                logger.warning("Memory ranker auto-disabled (#%d): %s",
                               self._disable_count, self._disable_reason)
                if self._disable_count >= FLAP_WINDOW:
                    self._permanently_disabled = True
                    self._disable_reason = f"flap guard: {self._disable_count} disables in session"
                    logger.warning("Memory ranker permanently disabled: %s", self._disable_reason)

        self._baseline_sample_counter += 1
        if self._baseline_sample_counter >= BASELINE_REFRESH_INTERVAL:
            self._baseline_sample_counter = 0
            self._refresh_baseline()

    def _refresh_baseline(self) -> None:
        """Periodically re-sample heuristic baseline from retrieval log."""
        try:
            from memory.retrieval_log import memory_retrieval_log
            metrics = memory_retrieval_log.get_eval_metrics(window=100)
            hr = metrics.get("heuristic_success_rate", 0.0)
            if hr > 0:
                self._baseline_success_rate = hr
        except Exception:
            pass

    def set_baseline_success_rate(self, rate: float) -> None:
        """Set the heuristic baseline success rate for regression detection."""
        if self._baseline_success_rate is None and rate > 0:
            self._baseline_success_rate = rate

    def enable(self) -> None:
        if not self._permanently_disabled:
            self._enabled = True
            self._disable_reason = ""

    def disable(self) -> None:
        self._enabled = False

    def save(self, path: str | Path = "") -> bool:
        if not _torch_available or self._model is None or not self._trained:
            return False
        save_path = Path(path) if path else RANKER_PATH
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": self._model.state_dict(),
                "train_count": self._train_count,
                "last_loss": self._last_loss,
                "last_accuracy": self._last_accuracy,
                "baseline_success_rate": self._baseline_success_rate,
            }, save_path)
            logger.info("Memory ranker saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save memory ranker: %s", exc)
            return False

    def load(self, path: str | Path = "") -> bool:
        if not _torch_available:
            return False
        load_path = Path(path) if path else RANKER_PATH
        if not load_path.exists():
            return False
        try:
            if self._model is None:
                self._model = _build_ranker_model()
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(checkpoint["state_dict"])
            self._model.eval()
            self._trained = True
            self._train_count = checkpoint.get("train_count", 1)
            self._last_loss = checkpoint.get("last_loss", 0.0)
            self._last_accuracy = checkpoint.get("last_accuracy", 0.0)
            self._baseline_success_rate = checkpoint.get("baseline_success_rate")
            logger.info(
                "Memory ranker loaded: train_count=%d loss=%.4f accuracy=%.4f",
                self._train_count, self._last_loss, self._last_accuracy,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load memory ranker: %s", exc)
            return False

    def get_stats(self) -> dict[str, Any]:
        eval_rate = sum(self._eval_window) / len(self._eval_window) if self._eval_window else 0.0
        cooldown_remaining = 0.0
        if not self._enabled and self._last_disable_time > 0 and not self._permanently_disabled:
            cooldown_remaining = max(0.0, REENABLE_COOLDOWN_S - (time.time() - self._last_disable_time))
        return {
            "available": _torch_available,
            "trained": self._trained,
            "enabled": self._enabled,
            "ready": self.is_ready(),
            "train_count": self._train_count,
            "last_loss": round(self._last_loss, 4),
            "last_accuracy": round(self._last_accuracy, 4),
            "last_train_time": self._last_train_time,
            "baseline_success_rate": round(self._baseline_success_rate, 4) if self._baseline_success_rate else None,
            "eval_window_size": len(self._eval_window),
            "eval_success_rate": round(eval_rate, 4),
            "model_path": str(RANKER_PATH),
            "model_exists": RANKER_PATH.exists(),
            "disable_count": self._disable_count,
            "disable_reason": self._disable_reason,
            "permanently_disabled": self._permanently_disabled,
            "cooldown_remaining_s": round(cooldown_remaining, 1),
        }


_ranker: MemoryRanker | None = None


def get_memory_ranker() -> MemoryRanker | None:
    """Get the global ranker instance (lazy init)."""
    global _ranker
    if _ranker is None:
        _ranker = MemoryRanker.get_instance()
    return _ranker


def init_memory_ranker() -> MemoryRanker:
    """Initialize and load persisted ranker model."""
    ranker = get_memory_ranker()
    if ranker:
        ranker.load()
    return ranker
