"""Fidelity #13: the post-apply health gate must FAIL-CLOSED.

It used to return True ("assume OK") when it couldn't verify health — a fake
rollback signal that would KEEP an unverifiable self-modification. The honest
behavior: if you can't prove the patch is healthy, roll it back.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from self_improve.orchestrator import SelfImprovementOrchestrator


def _orch():
    # _check_post_apply_health uses no instance state, only kernel_loop + module
    # constants, so a bare instance is enough to exercise it.
    return SelfImprovementOrchestrator.__new__(SelfImprovementOrchestrator)


def test_no_baseline_fails_closed():
    with patch("consciousness.kernel.kernel_loop") as kl:
        kl.get_performance.return_value = {"p95_tick_ms": 0.0}  # no baseline
        assert asyncio.run(_orch()._check_post_apply_health("/tmp/x")) is False


def test_kernel_error_fails_closed():
    with patch("consciousness.kernel.kernel_loop") as kl:
        kl.get_performance.side_effect = RuntimeError("perf unavailable")
        assert asyncio.run(_orch()._check_post_apply_health("/tmp/x")) is False


def test_thin_samples_fails_closed():
    # real baseline, but post-apply ticks never accumulate (< 10) -> cannot verify -> rollback
    calls = {"n": 0}
    def perf():
        calls["n"] += 1
        return {"p95_tick_ms": 5.0, "last_tick_ms": 0.0}  # baseline ok, no usable ticks
    with patch("consciousness.kernel.kernel_loop") as kl, \
         patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        kl.get_performance.side_effect = lambda: perf()
        assert asyncio.run(_orch()._check_post_apply_health("/tmp/x")) is False


def test_healthy_still_passes():
    # baseline + plenty of good ticks, no regression -> True (we didn't break the happy path)
    def perf():
        return {"p95_tick_ms": 5.0, "last_tick_ms": 5.0}
    with patch("consciousness.kernel.kernel_loop") as kl, \
         patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        kl.get_performance.side_effect = lambda: perf()
        assert asyncio.run(_orch()._check_post_apply_health("/tmp/x")) is True
