"""Experience Buffer — rolling buffer of (state, action, reward) tuples for training.

Stored on disk at ~/.jarvis/policy_experience.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import random
import math

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
EXPERIENCE_PATH = JARVIS_DIR / "policy_experience.jsonl"
MAX_BUFFER_SIZE = 5000
FLUSH_INTERVAL = 1
_MAX_JSONL_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class Experience:
    state_vec: list[float]
    action: dict[str, Any]
    reward: float
    timestamp: float
    next_state_vec: list[float] | None = None
    nn_action: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceBuffer:
    def __init__(self, max_size: int = MAX_BUFFER_SIZE) -> None:
        self._buffer: deque[Experience] = deque(maxlen=max_size)
        self._unflushed: int = 0

    @staticmethod
    def _priority_weight(exp: Experience, temperature: float = 0.5) -> float:
        """Softmax-like priority based on reward magnitude.

        Higher absolute rewards replay more often without fully starving the tail.
        """
        reward_mag = abs(exp.reward)
        scale = max(0.1, temperature)
        return math.exp(min(10.0, reward_mag / scale))

    def add(self, experience: Experience) -> None:
        self._buffer.append(experience)
        self._unflushed += 1
        if self._unflushed >= FLUSH_INTERVAL:
            self.flush()

    def sample(self, n: int) -> list[Experience]:
        """Uniform random sample for training."""
        items = list(self._buffer)
        if len(items) <= n:
            return items
        return random.sample(items, n)

    def sample_recency(self, n: int, recent_bias: float = 0.7) -> list[Experience]:
        """Blend recent and uniform replay.

        `recent_bias=0.7` means ~70% of samples come from the recent half of the
        buffer and the remainder from the full history.
        """
        items = list(self._buffer)
        if len(items) <= n:
            return items
        recent_n = max(1, int(n * max(0.0, min(1.0, recent_bias))))
        uniform_n = max(0, n - recent_n)
        recent_pool = items[max(0, len(items) // 2):]
        picked: list[Experience] = []
        seen: set[int] = set()

        for exp in random.sample(recent_pool, min(recent_n, len(recent_pool))):
            idx = id(exp)
            if idx not in seen:
                picked.append(exp)
                seen.add(idx)
        if uniform_n > 0:
            for exp in random.sample(items, min(uniform_n, len(items))):
                idx = id(exp)
                if idx not in seen:
                    picked.append(exp)
                    seen.add(idx)
                if len(picked) >= n:
                    break
        if len(picked) < n:
            for exp in items:
                idx = id(exp)
                if idx not in seen:
                    picked.append(exp)
                    seen.add(idx)
                if len(picked) >= n:
                    break
        return picked[:n]

    def sample_priority(self, n: int, temperature: float = 0.5) -> list[Experience]:
        """Priority-weighted sampling by reward magnitude."""
        items = list(self._buffer)
        if len(items) <= n:
            return items
        available = items[:]
        picked: list[Experience] = []
        while available and len(picked) < n:
            weights = [self._priority_weight(exp, temperature=temperature) for exp in available]
            chosen = random.choices(available, weights=weights, k=1)[0]
            picked.append(chosen)
            available.remove(chosen)
        return picked

    def sample_blended(
        self,
        n: int,
        priority_fraction: float = 0.5,
        recent_bias: float = 0.7,
        temperature: float = 0.5,
    ) -> list[Experience]:
        """Blend priority and recency replay for policy training."""
        items = list(self._buffer)
        if len(items) <= n:
            return items

        priority_n = max(0, int(n * max(0.0, min(1.0, priority_fraction))))
        recency_n = max(0, n - priority_n)
        picked: list[Experience] = []
        seen: set[int] = set()

        for exp in self.sample_priority(priority_n, temperature=temperature):
            idx = id(exp)
            if idx not in seen:
                picked.append(exp)
                seen.add(idx)
        for exp in self.sample_recency(recency_n, recent_bias=recent_bias):
            idx = id(exp)
            if idx not in seen:
                picked.append(exp)
                seen.add(idx)
            if len(picked) >= n:
                break
        if len(picked) < n:
            for exp in random.sample(items, len(items)):
                idx = id(exp)
                if idx not in seen:
                    picked.append(exp)
                    seen.add(idx)
                if len(picked) >= n:
                    break
        return picked[:n]

    def get_recent(self, n: int) -> list[Experience]:
        return list(self._buffer)[-n:]

    def get_all(self) -> list[Experience]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def get_stats(self) -> dict[str, Any]:
        items = list(self._buffer)
        reward_mags = sorted((abs(e.reward) for e in items), reverse=True)
        return {
            "size": len(self._buffer),
            "max_size": self._buffer.maxlen or MAX_BUFFER_SIZE,
            "unflushed": self._unflushed,
            "recent_bias_default": 0.7,
            "priority_temperature_default": 0.5,
            "top_reward_magnitude": reward_mags[0] if reward_mags else 0.0,
        }

    def split(self, train_ratio: float = 0.8) -> tuple[list[Experience], list[Experience]]:
        """Split into train/validation sets."""
        items = list(self._buffer)
        split_idx = int(len(items) * train_ratio)
        return items[:split_idx], items[split_idx:]

    def flush(self) -> None:
        """Append unflushed experiences to disk."""
        if self._unflushed == 0:
            return
        try:
            EXPERIENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            count = self._unflushed
            with open(EXPERIENCE_PATH, "a") as f:
                for exp in list(self._buffer)[-count:]:
                    f.write(json.dumps(asdict(exp), default=str) + "\n")
            self._unflushed = 0
            logger.info("Policy experience flush: wrote %d, total_buf=%d, path=%s",
                        count, len(self._buffer), EXPERIENCE_PATH)
            self._maybe_rotate()
        except Exception:
            logger.exception("Failed to flush experience buffer")

    @staticmethod
    def _maybe_rotate() -> None:
        """Trim JSONL to last half when file exceeds size limit."""
        try:
            if not EXPERIENCE_PATH.exists():
                return
            if EXPERIENCE_PATH.stat().st_size <= _MAX_JSONL_BYTES:
                return
            with open(EXPERIENCE_PATH, "r") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2:]
            with open(EXPERIENCE_PATH, "w") as f:
                f.writelines(keep)
            logger.info("Rotated policy_experience.jsonl: %d→%d lines", len(lines), len(keep))
        except OSError:
            pass

    def load(self) -> int:
        """Load experiences from disk into buffer, filtering contaminated entries.

        Shadow-tick entries (source=shadow_tick) with near-constant positive
        rewards pollute training.  They are stripped on load and the clean
        file is rewritten.
        """
        if not EXPERIENCE_PATH.exists():
            return 0
        count = 0
        skipped = 0
        try:
            with open(EXPERIENCE_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    source = data.get("metadata", {}).get("source", "")
                    if source == "shadow_tick":
                        skipped += 1
                        continue
                    exp = Experience(
                        state_vec=data["state_vec"],
                        action=data["action"],
                        reward=data["reward"],
                        timestamp=data["timestamp"],
                        next_state_vec=data.get("next_state_vec"),
                        nn_action=data.get("nn_action"),
                        metadata=data.get("metadata", {}),
                    )
                    self._buffer.append(exp)
                    count += 1
            if skipped > 0:
                logger.info("Purged %d shadow_tick entries from experience buffer", skipped)
                self._rewrite_clean()
            logger.info("Loaded %d experiences from %s", count, EXPERIENCE_PATH)
        except Exception:
            logger.exception("Failed to load experience buffer")
        return count

    def _rewrite_clean(self) -> None:
        """Rewrite the JSONL file with only the in-memory (clean) buffer."""
        try:
            tmp = EXPERIENCE_PATH.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for exp in self._buffer:
                    f.write(json.dumps(asdict(exp), default=str) + "\n")
            os.replace(tmp, EXPERIENCE_PATH)
            self._unflushed = 0
        except OSError as exc:
            logger.warning("Failed to rewrite clean experience buffer: %s", exc)
