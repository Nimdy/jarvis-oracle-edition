"""Base executor interface for learning job phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PhaseResult:
    """Outcome of a phase executor run."""
    progressed: bool
    message: str = ""
    artifact: Optional[dict[str, Any]] = None
    evidence: Optional[dict[str, Any]] = None
    gate_updates: Optional[list[dict[str, Any]]] = None
    metric_updates: Optional[dict[str, float]] = None


class PhaseExecutor:
    """Base class for phase executors.

    Subclasses set ``capability_type`` and ``phase``, then implement ``run()``.
    """

    capability_type: str = "procedural"
    phase: str = "assess"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            job.capability_type == self.capability_type
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        raise NotImplementedError
