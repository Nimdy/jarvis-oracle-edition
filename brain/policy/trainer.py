"""Trainer — advantage-weighted imitation learning.

When the NN's own proposal (nn_action) exists and was associated with a
positive advantage (reward > baseline), train toward the nn_action.
Otherwise fall back to imitating the kernel action, weighted by reward.

This breaks the "perfect copy → ties forever" equilibrium by allowing the
NN to reinforce its own good deviations from the kernel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from policy.experience_buffer import ExperienceBuffer, Experience

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BATCH_SIZE = 32
MIN_TRAINING_SAMPLES = 20
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
TRAINING_INTERVAL_S = 600.0
MIN_REWARD_WEIGHT = 0.1
ADVANTAGE_THRESHOLD = 0.05
MAX_REPLAY_SAMPLES = 768


@dataclass
class TrainingResult:
    epochs: int = 0
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    training_time_s: float = 0.0
    samples_used: int = 0
    converged: bool = False
    nn_target_pct: float = 0.0


class PolicyTrainer:
    def __init__(self, buffer: ExperienceBuffer) -> None:
        self._buffer = buffer
        self._last_training_time: float = 0.0
        self._reward_ema: float = 0.0
        self._ema_initialized: bool = False

    def should_train(self) -> bool:
        if len(self._buffer) < MIN_TRAINING_SAMPLES:
            return False
        if time.time() - self._last_training_time < TRAINING_INTERVAL_S:
            return False
        return True

    def _update_reward_baseline(self, rewards: list[float]) -> float:
        """Maintain an EMA reward baseline for advantage computation."""
        if not rewards:
            return self._reward_ema
        batch_mean = sum(rewards) / len(rewards)
        if not self._ema_initialized:
            self._reward_ema = batch_mean
            self._ema_initialized = True
        else:
            self._reward_ema = 0.95 * self._reward_ema + 0.05 * batch_mean
        return self._reward_ema

    def train_imitation(self, model: Any, epochs: int = MAX_EPOCHS) -> TrainingResult:
        """Advantage-weighted imitation learning.

        For each sample:
        - Compute advantage = reward - baseline (EMA of recent rewards).
        - If nn_action exists AND advantage > ADVANTAGE_THRESHOLD, train
          toward nn_action (reinforcing good NN deviations).
        - Otherwise train toward kernel action (standard imitation).
        - Weight all samples by clamp(1 + alpha * norm_advantage, 0.1, 3.0).
        """
        if not TORCH_AVAILABLE or model is None:
            return TrainingResult()

        replay_cap = min(len(self._buffer), MAX_REPLAY_SAMPLES)
        replay_data = self._buffer.sample_blended(
            replay_cap,
            priority_fraction=0.5,
            recent_bias=0.7,
            temperature=0.5,
        )
        split_idx = int(len(replay_data) * 0.8)
        train_data = replay_data[:split_idx]
        val_data = replay_data[split_idx:]
        if len(train_data) < MIN_TRAINING_SAMPLES:
            logger.info("Not enough data for training (%d < %d)", len(train_data), MIN_TRAINING_SAMPLES)
            return TrainingResult(samples_used=len(train_data))

        all_rewards = [e.reward for e in train_data]
        baseline = self._update_reward_baseline(all_rewards)
        reward_std = max(0.01, (sum((r - baseline) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5)

        start = time.time()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_loss = float("inf")
        patience_counter = 0
        epoch_losses: list[float] = []
        nn_target_count = 0
        total_samples = 0

        for epoch in range(epochs):
            model.train()
            import random
            random.shuffle(train_data)

            total_loss = 0.0
            batches = 0

            for i in range(0, len(train_data), BATCH_SIZE):
                batch = train_data[i:i + BATCH_SIZE]
                if len(batch) < 2:
                    continue

                states = torch.tensor([e.state_vec for e in batch], dtype=torch.float32)

                targets_list = []
                weights_list = []
                for e in batch:
                    advantage = e.reward - baseline
                    norm_adv = advantage / reward_std

                    use_nn = (
                        e.nn_action is not None
                        and advantage > ADVANTAGE_THRESHOLD
                    )
                    if use_nn:
                        targets_list.append(self._action_to_vec(e.nn_action))
                        nn_target_count += 1
                    else:
                        targets_list.append(self._action_to_vec(e.action))

                    w = max(MIN_REWARD_WEIGHT, min(3.0, 1.0 + norm_adv))
                    weights_list.append(w)
                    total_samples += 1

                targets = torch.tensor(targets_list, dtype=torch.float32)
                weights = torch.tensor(weights_list, dtype=torch.float32).unsqueeze(1)

                pred = model(states)
                per_sample_loss = ((pred - targets) ** 2).mean(dim=1, keepdim=True)
                loss = (per_sample_loss * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(batches, 1)
            epoch_losses.append(avg_loss)

            val_loss = self._validate(model, val_data, baseline, reward_std)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info("Early stopping at epoch %d (best_val_loss=%.4f)", epoch, best_loss)
                    break

        model.eval()
        training_time = time.time() - start
        self._last_training_time = time.time()

        nn_pct = (nn_target_count / max(total_samples, 1)) * 100

        result = TrainingResult(
            epochs=epoch + 1,
            final_loss=avg_loss,
            best_loss=best_loss,
            training_time_s=training_time,
            samples_used=len(train_data),
            converged=patience_counter < EARLY_STOP_PATIENCE,
            nn_target_pct=nn_pct,
        )

        from policy.telemetry import policy_telemetry
        policy_telemetry.record_train_complete(
            epochs=result.epochs,
            loss=result.best_loss,
            duration_s=result.training_time_s,
            epoch_losses=epoch_losses,
        )

        logger.info(
            "Training complete: %d epochs, loss=%.4f, nn_target=%.1f%%, time=%.1fs, replay=%d",
            result.epochs, result.best_loss, nn_pct, result.training_time_s, len(replay_data),
        )
        return result

    def _validate(self, model: Any, val_data: list[Experience],
                  baseline: float = 0.0, reward_std: float = 1.0) -> float:
        if not val_data or not TORCH_AVAILABLE:
            return float("inf")

        model.eval()
        with torch.no_grad():
            states = torch.tensor([e.state_vec for e in val_data], dtype=torch.float32)
            targets_list = []
            weights_list = []
            for e in val_data:
                adv = e.reward - baseline
                use_nn = e.nn_action is not None and adv > ADVANTAGE_THRESHOLD
                targets_list.append(self._action_to_vec(e.nn_action if use_nn else e.action))
                weights_list.append(max(MIN_REWARD_WEIGHT, min(3.0, 1.0 + adv / reward_std)))
            targets = torch.tensor(targets_list, dtype=torch.float32)
            weights = torch.tensor(weights_list, dtype=torch.float32).unsqueeze(1)
            pred = model(states)
            per_sample_loss = ((pred - targets) ** 2).mean(dim=1, keepdim=True)
            loss = (per_sample_loss * weights).mean()
        return loss.item()

    @staticmethod
    def _action_to_vec(action: dict[str, Any]) -> list[float]:
        """Convert action dict to fixed-size target vector.

        thought_weights are absolute values (0.5-2.0), scaled so 1.0→0.0, 2.0→1.0.
        budget_ms centered on 16, scaled by 10.
        Dims 5-7 encode mode/response hints as simple floats.
        """
        tw = action.get("thought_weights_delta", {})
        budget = action.get("budget_ms", 16) or 16
        mode = action.get("suggested_mode", "") or ""
        resp = action.get("response_length_hint", "") or ""
        mode_val = {"idle": -0.5, "conversational": 0.5, "focused": 0.8}.get(mode, 0.0)
        resp_val = {"brief": -0.5, "moderate": 0.0, "detailed": 0.5}.get(resp, 0.0)
        return [
            max(-1.0, min(1.0, tw.get("philosophical", 1.0) - 1.0)),
            max(-1.0, min(1.0, tw.get("contextual", 1.0) - 1.0)),
            max(-1.0, min(1.0, tw.get("reactive", 1.0) - 1.0)),
            max(-1.0, min(1.0, tw.get("introspective", 1.0) - 1.0)),
            max(-1.0, min(1.0, (budget - 16) / 10.0)),
            mode_val,
            resp_val,
            0.0,
        ]
