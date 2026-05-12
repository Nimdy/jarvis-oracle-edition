"""Policy NN — small PyTorch models for learned kernel control.

Architecture menu:
  - MLP-2layer (64/64)
  - MLP-3layer (128/64/32)
  - GRU(64) + MLP head

All models output a PolicyDecision-compatible vector.
Telemetry: arch/device/model_id set on init; encode latency tracked per forward.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from policy.state_encoder import STATE_DIM

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — policy NN disabled")

ACTION_DIM = 8


if TORCH_AVAILABLE:

    class MLP2Layer(nn.Module):
        def __init__(self, input_dim: int = STATE_DIM, hidden: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, ACTION_DIM),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class MLP3Layer(nn.Module):
        def __init__(self, input_dim: int = STATE_DIM) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, ACTION_DIM),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class GRUPolicy(nn.Module):
        def __init__(self, input_dim: int = STATE_DIM, hidden: int = 64) -> None:
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, ACTION_DIM),
                nn.Tanh(),
            )
            self._hidden: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                x = x.unsqueeze(1)

            batch_size = x.size(0)
            h0 = None
            if self._hidden is not None:
                if self._hidden.size(1) == batch_size:
                    h0 = self._hidden.detach()
            out, self._hidden = self.gru(x, h0)
            return self.head(out[:, -1, :])

        def reset_hidden(self) -> None:
            self._hidden = None


ARCHITECTURE_REGISTRY: dict[str, type] = {}
if TORCH_AVAILABLE:
    ARCHITECTURE_REGISTRY = {
        "mlp2": MLP2Layer,
        "mlp3": MLP3Layer,
        "gru": GRUPolicy,
    }


class PolicyNNController:
    """Wraps a PyTorch model and converts state dicts to PolicyDecision."""

    def __init__(self, arch: str = "mlp2", input_dim: int = STATE_DIM) -> None:
        self._arch_name = arch
        self._model: Any = None
        self._input_dim = input_dim
        self._encoder: Any = None

        # Strip _encN suffix from composite arch labels (e.g. "mlp2_enc2" -> "mlp2")
        import re
        base_arch = re.sub(r"_enc\d+$", "", arch)

        if TORCH_AVAILABLE and base_arch in ARCHITECTURE_REGISTRY:
            self._model = ARCHITECTURE_REGISTRY[base_arch](input_dim=input_dim)
            logger.info("PolicyNN initialized: %s (input_dim=%d)", arch, input_dim)
        else:
            logger.warning("PolicyNN not available (arch=%s, torch=%s)", arch, TORCH_AVAILABLE)

        self._sync_telemetry()

    def set_encoder(self, encoder: Any) -> None:
        self._encoder = encoder

    @property
    def model(self) -> Any:
        return self._model

    @property
    def arch_name(self) -> str:
        return self._arch_name

    def forward(self, state: dict[str, Any]) -> Any:
        """Run the NN and return a PolicyDecision."""
        import time as _time
        from policy.policy_interface import PolicyDecision

        if not TORCH_AVAILABLE or self._model is None:
            return PolicyDecision(confidence=0.0, source="nn_unavailable")

        if self._encoder is None:
            from policy.state_encoder import StateEncoder
            self._encoder = StateEncoder()

        t0 = _time.perf_counter()
        vec = self._encoder.encode(state)
        encode_ms = (_time.perf_counter() - t0) * 1000.0

        x = torch.tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            raw = self._model(x)

        if raw.dim() > 1:
            raw = raw.squeeze(0)

        values = raw.tolist()

        mode_val = values[5] if len(values) > 5 else 0.0
        if mode_val < -0.25:
            suggested_mode = "idle"
        elif mode_val > 0.65:
            suggested_mode = "focused"
        elif mode_val > 0.25:
            suggested_mode = "conversational"
        else:
            suggested_mode = ""

        resp_val = values[6] if len(values) > 6 else 0.0
        if resp_val < -0.25:
            response_length_hint = "brief"
        elif resp_val > 0.25:
            response_length_hint = "detailed"
        else:
            response_length_hint = ""

        return PolicyDecision(
            thought_weights_delta={
                "philosophical": values[0] * 0.1,
                "contextual": values[1] * 0.1,
                "reactive": values[2] * 0.1,
                "introspective": values[3] * 0.1,
            },
            budget_ms=int(16 + values[4] * 5) if abs(values[4]) > 0.1 else None,
            run_tasks=[],
            mutation_rank=[],
            confidence=max(0.0, min(1.0, (abs(values[0]) + abs(values[1])) / 2)),
            source=f"nn:{self._arch_name}",
            suggested_mode=suggested_mode,
            response_length_hint=response_length_hint,
        )

    def save(self, path: str) -> None:
        if TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> bool:
        if not TORCH_AVAILABLE or self._model is None:
            return False
        try:
            self._model.load_state_dict(torch.load(path, weights_only=True))
            self._model.eval()
            self._sync_telemetry()
            return True
        except Exception:
            logger.exception("Failed to load policy NN from %s", path)
            return False

    def _sync_telemetry(self) -> None:
        from policy.telemetry import policy_telemetry
        policy_telemetry.arch = self._arch_name if self._model else "none"
        device = "cpu"
        if TORCH_AVAILABLE and self._model is not None:
            try:
                p = next(self._model.parameters(), None)
                if p is not None:
                    device = str(p.device)
            except StopIteration:
                pass
        policy_telemetry.device = device


def migrate_weights(
    old_model: Any,
    new_model: Any,
    old_dim: int = STATE_DIM,
    new_dim: int = 22,
) -> bool:
    """Zero-pad first linear layer weights from old_dim to new_dim for warm-start.

    Copies all weights from old_model to new_model. For the first Linear layer
    (which has input_features == old_dim), the weight matrix is zero-padded to
    accommodate new_dim inputs. All other layers are copied verbatim.

    Returns True on success, False if migration is not possible.
    """
    if not TORCH_AVAILABLE:
        return False

    if old_model is None or new_model is None:
        return False

    try:
        old_sd = old_model.state_dict()
        new_sd = new_model.state_dict()

        for key in new_sd:
            if key not in old_sd:
                continue

            old_tensor = old_sd[key]
            new_tensor = new_sd[key]

            if old_tensor.shape == new_tensor.shape:
                new_sd[key] = old_tensor.clone()
            elif (
                len(old_tensor.shape) == 2
                and old_tensor.shape[1] == old_dim
                and new_tensor.shape[1] == new_dim
                and old_tensor.shape[0] == new_tensor.shape[0]
            ):
                padded = torch.zeros_like(new_tensor)
                padded[:, :old_dim] = old_tensor
                new_sd[key] = padded
                logger.info(
                    "Migrated layer %s: %s -> %s (zero-padded %d new dims)",
                    key, list(old_tensor.shape), list(new_tensor.shape),
                    new_dim - old_dim,
                )
            else:
                logger.warning(
                    "Shape mismatch for %s: old=%s, new=%s — skipping",
                    key, list(old_tensor.shape), list(new_tensor.shape),
                )

        new_model.load_state_dict(new_sd)
        logger.info("Weight migration complete: %d -> %d dims", old_dim, new_dim)
        return True

    except Exception:
        logger.exception("Weight migration failed")
        return False
