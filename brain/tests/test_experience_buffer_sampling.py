import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from policy.experience_buffer import Experience, ExperienceBuffer
from memory.gate import MemoryGate


def _exp(idx: int, reward: float) -> Experience:
    return Experience(
        state_vec=[0.0] * 20,
        action={"response_depth": idx / 10.0},
        reward=reward,
        timestamp=time.time() + idx,
    )


def test_sample_blended_returns_unique_capped_batch():
    random.seed(7)
    buf = ExperienceBuffer(max_size=20)
    for i in range(10):
        buf.add(_exp(i, reward=float(i)))

    sample = buf.sample_blended(6, priority_fraction=0.5, recent_bias=0.7, temperature=0.5)
    assert len(sample) == 6
    assert len({id(x) for x in sample}) == 6


def test_priority_sampling_prefers_high_reward_magnitudes():
    random.seed(11)
    buf = ExperienceBuffer(max_size=30)
    for i in range(12):
        reward = 0.1 if i < 10 else 4.0 + i
        buf.add(_exp(i, reward=reward))

    sample = buf.sample_priority(4, temperature=0.5)
    rewards = sorted(abs(e.reward) for e in sample)
    assert rewards[-1] >= 14.0


def test_memory_gate_session_tracks_open_close():
    gate = MemoryGate()
    assert gate.is_open() is False
    with gate.session("test_search", actor="test"):
        assert gate.is_open() is True
        stats = gate.get_stats()
        assert stats["depth"] == 1
    stats = gate.get_stats()
    assert stats["is_open"] is False
    assert stats["total_opens"] == 1
    assert len(stats["recent_transitions"]) == 2
