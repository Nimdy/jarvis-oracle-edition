"""Fidelity #13: the post-apply health gate must FAIL-CLOSED and ACTUALLY measure.

It used to (a) import a non-existent `kernel_loop` (dead import -> always ImportError
-> no baseline) and (b) return True ("assume OK") whenever it couldn't verify health.
Net: the gate never measured and always passed — pure theater. Fixed to read the real
kernel (self._engine._kernel.get_performance) and fail CLOSED (rollback) when it
genuinely can't verify; the happy path still passes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from self_improve.orchestrator import SelfImprovementOrchestrator


def _orch(get_perf=None, has_kernel=True):
    # _check_post_apply_health uses only self._engine._kernel + module constants,
    # so a bare instance with a mocked engine is enough to exercise it.
    o = SelfImprovementOrchestrator.__new__(SelfImprovementOrchestrator)
    o._engine = MagicMock()
    if not has_kernel:
        o._engine._kernel = None
    elif get_perf is not None:
        o._engine._kernel.get_performance = get_perf
    return o


def _run(o):
    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        return asyncio.run(o._check_post_apply_health("/tmp/x"))


def test_no_kernel_handle_fails_closed():
    assert _run(_orch(has_kernel=False)) is False


def test_no_baseline_fails_closed():
    assert _run(_orch(get_perf=lambda: {"p95_tick_ms": 0.0})) is False


def test_kernel_error_fails_closed():
    assert _run(_orch(get_perf=MagicMock(side_effect=RuntimeError("perf unavailable")))) is False


def test_thin_samples_fails_closed():
    # real baseline, but post-apply ticks never accumulate (last_tick_ms=0) -> cannot verify
    assert _run(_orch(get_perf=lambda: {"p95_tick_ms": 5.0, "last_tick_ms": 0.0})) is False


def test_healthy_still_passes():
    # baseline + plenty of good ticks, no regression -> True (happy path preserved)
    assert _run(_orch(get_perf=lambda: {"p95_tick_ms": 5.0, "last_tick_ms": 5.0})) is True
