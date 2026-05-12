"""Memory Salience Model — learned gate for memory storage decisions.

Predicts whether a memory is worth storing, what initial weight it should have,
and how fast it should decay. Trained from memory lifecycle outcomes: memories
that got reinforced/retrieved are positive, those that decayed unused are negative.

Advisory in v1: blends with rule-based defaults at a configurable ratio
(starts at 20% model / 80% rules, increases as model accuracy improves).

Architecture: MLP 11 -> 24 -> 12 -> 3 (sigmoid), ~500 params, CPU inference.
Trained during dream/sleep cycles from MemoryLifecycleLog pairs.
Persisted at ~/.jarvis/memory_salience.pt
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
SALIENCE_PATH = JARVIS_DIR / "memory_salience.pt"

INPUT_DIM = 11
OUTPUT_DIM = 3
MIN_TRAINING_PAIRS = 100
MAX_TRAINING_PAIRS = 2000
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 0.001

INITIAL_MODEL_BLEND = 0.2
MAX_MODEL_BLEND = 0.6
BLEND_INCREASE_THRESHOLD = 500  # validated predictions before blend increases

_torch_available = False
try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass


@dataclass
class SaliencePrediction:
    """Output of the salience model."""
    store_confidence: float
    predicted_weight: float
    predicted_decay_rate: float
    model_used: bool = False


def _build_salience_model() -> Any:
    if not _torch_available:
        return None
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 24),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, OUTPUT_DIM),
        nn.Sigmoid(),
    )


class SalienceModel:
    """Learned salience gate with rule-based fallback blending."""

    _instance: SalienceModel | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._trained = False
        self._enabled = True
        self._train_count = 0
        self._last_train_time: float = 0.0
        self._last_loss: float = 0.0
        self._last_accuracy: float = 0.0
        self._model_blend: float = INITIAL_MODEL_BLEND
        self._validated_predictions: int = 0

        if _torch_available:
            self._model = _build_salience_model()

    @classmethod
    def get_instance(cls) -> SalienceModel:
        if cls._instance is None:
            cls._instance = SalienceModel()
        return cls._instance

    def is_ready(self) -> bool:
        return _torch_available and self._model is not None and self._trained and self._enabled

    def predict(self, features: list[float]) -> SaliencePrediction:
        """Predict salience for a candidate memory.

        Returns store_confidence, predicted_weight, predicted_decay_rate.
        """
        if not self.is_ready():
            return SaliencePrediction(
                store_confidence=1.0,
                predicted_weight=0.5,
                predicted_decay_rate=0.5,
                model_used=False,
            )

        try:
            t = torch.tensor([features[:INPUT_DIM]], dtype=torch.float32)
            with torch.no_grad():
                output = self._model(t).squeeze(0)
            return SaliencePrediction(
                store_confidence=float(output[0].item()),
                predicted_weight=float(output[1].item()),
                predicted_decay_rate=float(output[2].item()),
                model_used=True,
            )
        except Exception:
            return SaliencePrediction(
                store_confidence=1.0,
                predicted_weight=0.5,
                predicted_decay_rate=0.5,
                model_used=False,
            )

    def advise_weight(
        self,
        rule_weight: float,
        rule_decay_rate: float,
        features: list[float],
    ) -> tuple[float, float]:
        """Blend model prediction with rule-based defaults.

        Returns (adjusted_weight, adjusted_decay_rate).
        Advisory only: never zeroes out weight or blocks creation.
        """
        pred = self.predict(features)
        if not pred.model_used:
            return rule_weight, rule_decay_rate

        blend = self._model_blend

        if pred.store_confidence < 0.3:
            weight_adj = rule_weight * (1.0 - 0.3 * blend)
        elif pred.store_confidence > 0.8:
            weight_adj = rule_weight * (1.0 + 0.1 * blend)
        else:
            weight_adj = rule_weight

        blended_weight = rule_weight * (1.0 - blend) + pred.predicted_weight * blend
        final_weight = max(0.05, min(0.90, (weight_adj + blended_weight) / 2.0))

        decay_scale = pred.predicted_decay_rate * 0.1
        blended_decay = rule_decay_rate * (1.0 - blend) + decay_scale * blend
        final_decay = max(0.001, min(0.1, blended_decay))

        return final_weight, final_decay

    def record_validated_prediction(self) -> None:
        """Increment validated prediction count and possibly increase blend ratio."""
        self._validated_predictions += 1
        if (self._validated_predictions >= BLEND_INCREASE_THRESHOLD
                and self._model_blend < MAX_MODEL_BLEND):
            old_blend = self._model_blend
            self._model_blend = min(MAX_MODEL_BLEND, self._model_blend + 0.1)
            if abs(self._model_blend - old_blend) > 0.01:
                logger.info(
                    "Salience model blend increased: %.2f -> %.2f (after %d validations)",
                    old_blend, self._model_blend, self._validated_predictions,
                )
                self._validated_predictions = 0

    def train_from_pairs(
        self,
        features: list[list[float]],
        store_labels: list[float],
        weight_labels: list[float],
        decay_labels: list[float],
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
    ) -> dict[str, Any]:
        """Train the salience model from lifecycle pairs."""
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
        Y = torch.stack([
            torch.tensor(store_labels[-MAX_TRAINING_PAIRS:], dtype=torch.float32),
            torch.tensor(weight_labels[-MAX_TRAINING_PAIRS:], dtype=torch.float32),
            torch.tensor(decay_labels[-MAX_TRAINING_PAIRS:], dtype=torch.float32),
        ], dim=1)

        n_samples = X.shape[0]
        split = max(1, int(n_samples * 0.8))
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()
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
            store_acc = ((all_pred[:, 0] > 0.5) == (Y[:, 0] > 0.5)).float().mean().item()

        self._trained = True
        self._train_count += 1
        self._last_train_time = time.time()
        self._last_loss = final_loss
        self._last_accuracy = store_acc

        duration_ms = (time.time() - start) * 1000.0

        self.save()

        result = {
            "loss": round(final_loss, 4),
            "store_accuracy": round(store_acc, 4),
            "pairs": n_samples,
            "epochs": epoch + 1,
            "duration_ms": round(duration_ms, 1),
            "train_count": self._train_count,
            "model_blend": round(self._model_blend, 2),
        }
        logger.info("Salience model trained: %s", result)
        return result

    def save(self, path: str | Path = "") -> bool:
        if not _torch_available or self._model is None or not self._trained:
            return False
        save_path = Path(path) if path else SALIENCE_PATH
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": self._model.state_dict(),
                "train_count": self._train_count,
                "last_loss": self._last_loss,
                "last_accuracy": self._last_accuracy,
                "model_blend": self._model_blend,
                "validated_predictions": self._validated_predictions,
            }, save_path)
            logger.info("Salience model saved to %s", save_path)
            return True
        except Exception as exc:
            logger.error("Failed to save salience model: %s", exc)
            return False

    def load(self, path: str | Path = "") -> bool:
        if not _torch_available:
            return False
        load_path = Path(path) if path else SALIENCE_PATH
        if not load_path.exists():
            return False
        try:
            if self._model is None:
                self._model = _build_salience_model()
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(checkpoint["state_dict"])
            self._model.eval()
            self._trained = True
            self._train_count = checkpoint.get("train_count", 1)
            self._last_loss = checkpoint.get("last_loss", 0.0)
            self._last_accuracy = checkpoint.get("last_accuracy", 0.0)
            self._model_blend = checkpoint.get("model_blend", INITIAL_MODEL_BLEND)
            self._validated_predictions = checkpoint.get("validated_predictions", 0)
            logger.info(
                "Salience model loaded: train_count=%d loss=%.4f blend=%.2f",
                self._train_count, self._last_loss, self._model_blend,
            )
            return True
        except RuntimeError as exc:
            if "size mismatch" in str(exc):
                logger.warning(
                    "Salience model checkpoint incompatible (INPUT_DIM changed to %d), "
                    "discarding old weights — will retrain from scratch",
                    INPUT_DIM,
                )
            else:
                logger.error("Failed to load salience model: %s", exc)
            return False
        except Exception as exc:
            logger.error("Failed to load salience model: %s", exc)
            return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "available": _torch_available,
            "trained": self._trained,
            "enabled": self._enabled,
            "ready": self.is_ready(),
            "train_count": self._train_count,
            "last_loss": round(self._last_loss, 4),
            "last_accuracy": round(self._last_accuracy, 4),
            "last_train_time": self._last_train_time,
            "model_blend": round(self._model_blend, 2),
            "validated_predictions": self._validated_predictions,
            "model_path": str(SALIENCE_PATH),
            "model_exists": SALIENCE_PATH.exists(),
        }


_salience: SalienceModel | None = None


def get_salience_model() -> SalienceModel | None:
    global _salience
    if _salience is None:
        _salience = SalienceModel.get_instance()
    return _salience


def init_salience_model() -> SalienceModel:
    salience = get_salience_model()
    if salience:
        salience.load()
    return salience
