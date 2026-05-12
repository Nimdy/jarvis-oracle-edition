"""Boot stabilization tests for MutationGovernor."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from consciousness.kernel_config import KernelConfig
from consciousness.mutation_governor import MutationGovernor, SystemHealth


def _healthy_system() -> SystemHealth:
    return SystemHealth(tick_p95_ms=5.0, deferred_backlog=0, avg_tick_ms=2.0)


def test_mutation_governor_rejects_during_boot_stabilization_window():
    gov = MutationGovernor()
    gov._boot_stabilization_s = 120.0
    gov._boot_time = time.time()

    decision = gov.evaluate(
        proposal_changes={"tw.contextual": 1.1},
        proposal_confidence=0.9,
        current_config=KernelConfig(),
        health=_healthy_system(),
    )

    assert decision.approved is False
    assert any("Boot stabilization active" in v for v in decision.violations)
    assert "startup stabilization window" in decision.reasoning


def test_mutation_governor_ignores_boot_gate_after_window_elapsed():
    gov = MutationGovernor()
    gov._boot_stabilization_s = 10.0
    gov._boot_time = time.time() - 60.0
    gov._last_mutation_time = time.time() - 600.0  # bypass cooldown

    decision = gov.evaluate(
        proposal_changes={"tw.contextual": 1.1},
        proposal_confidence=0.9,
        current_config=KernelConfig(),
        health=_healthy_system(),
    )

    assert all("Boot stabilization active" not in v for v in decision.violations)
