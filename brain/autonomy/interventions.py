"""Candidate Intervention types — Phase 5.2.

Defines the CandidateIntervention dataclass with controlled enums for
target_subsystem and intervention type. Every research action must produce
either a candidate intervention or a 'research_no_action' verdict.

Allowed types (Sprint 5.2):
  threshold_change, routing_rule, prompt_frame, eval_contract,
  memory_weighting_rule, calibration_adjustment, research_no_action

Deferred types (Sprint 5.3+ / L2):
  code_patch, schema_change, new_subsystem
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TargetSubsystem = Literal[
    "conversation", "routing", "memory", "calibration",
    "eval", "autonomy", "language", "world_model",
]

InterventionType = Literal[
    "threshold_change", "routing_rule", "prompt_frame",
    "eval_contract", "memory_weighting_rule",
    "calibration_adjustment", "research_no_action",
]

_ALLOWED_TYPES: frozenset[str] = frozenset({
    "threshold_change", "routing_rule", "prompt_frame",
    "eval_contract", "memory_weighting_rule",
    "calibration_adjustment", "research_no_action",
})

_DEFERRED_TYPES: frozenset[str] = frozenset({
    "code_patch", "schema_change", "new_subsystem",
})

_VALID_SUBSYSTEMS: frozenset[str] = frozenset({
    "conversation", "routing", "memory", "calibration",
    "eval", "autonomy", "language", "world_model",
})


@dataclass
class CandidateIntervention:
    intervention_id: str = ""
    change_type: str = "research_no_action"
    target_subsystem: str = ""
    target_symbol: str = ""
    trigger_deficit: str = ""
    source_ids: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    proposed_change: str = ""
    expected_metric: str = ""
    expected_direction: str = ""
    falsifier: str = ""
    risk_level: str = "low"
    shadow_only: bool = True
    status: str = "proposed"
    created_at: float = field(default_factory=time.time)
    shadow_start: float = 0.0
    shadow_end: float = 0.0
    baseline_value: float = 0.0
    measured_delta: float = 0.0
    verdict: str = ""

    def __post_init__(self) -> None:
        if not self.intervention_id:
            self.intervention_id = f"iv_{uuid.uuid4().hex[:12]}"

    @property
    def is_no_action(self) -> bool:
        return self.change_type == "research_no_action" or self.status == "no_action"

    @property
    def is_allowed_type(self) -> bool:
        return self.change_type in _ALLOWED_TYPES

    @property
    def is_deferred_type(self) -> bool:
        return self.change_type in _DEFERRED_TYPES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateIntervention:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def make_no_action(
    trigger_deficit: str,
    source_ids: list[str],
    evidence_summary: str = "",
    falsifier: str = "",
) -> CandidateIntervention:
    """Create a no-action intervention with disciplined reasoning."""
    return CandidateIntervention(
        change_type="research_no_action",
        trigger_deficit=trigger_deficit,
        source_ids=source_ids,
        evidence_summary=evidence_summary or "Research reviewed; no actionable change identified.",
        falsifier=falsifier,
        status="no_action",
    )
