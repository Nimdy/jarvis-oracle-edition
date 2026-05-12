"""Improvement Request — structured description of what needs changing and why.

Created by observer/analytics/emergence/evaluator when a plateau or regression
is detected.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ImprovementRequest:
    id: str = field(default_factory=lambda: f"imp_{uuid.uuid4().hex[:10]}")
    timestamp: float = field(default_factory=time.time)
    type: Literal[
        "performance_optimization",
        "policy_model_upgrade",
        "consciousness_enhancement",
        "bug_fix",
        "architecture_improvement",
    ] = "consciousness_enhancement"
    target_module: str = ""
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    priority: float = 0.5
    constraints: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    manual: bool = False
    golden_trace_id: str = ""
    golden_command_id: str = ""
    golden_authority_class: str = ""
    golden_status: str = "none"
    fingerprint: str = ""
    evidence_detail: dict[str, Any] = field(default_factory=dict)
    # Phase 6.5: per-request path allowlist for L3 escalation flow.
    # When empty, the request inherits the default ALLOWED_PATHS global.
    # When non-empty, PatchPlan.validate_scope uses this narrow set
    # instead of ALLOWED_PATHS, so approved escalations cannot widen
    # scope beyond what the operator approved. No global mutation of
    # ALLOWED_PATHS ever occurs. See
    # docs/plans/phase_6_5_l3_escalation.plan.md.
    declared_scope: list[str] = field(default_factory=list)

    @property
    def scope_paths(self) -> list[str]:
        """Effective allowlist of directories this improvement can touch.

        Returns ``declared_scope`` if it is non-empty, otherwise the
        default ``patch_plan.ALLOWED_PATHS``. Must stay in sync with
        :func:`self_improve.patch_plan.PatchPlan.validate_scope`.
        """
        if self.declared_scope:
            return list(self.declared_scope)
        from self_improve.patch_plan import ALLOWED_PATHS
        return list(ALLOWED_PATHS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "target_module": self.target_module,
            "description": self.description,
            "evidence": self.evidence,
            "priority": self.priority,
            "constraints": self.constraints,
            "requires_approval": self.requires_approval,
            "golden_trace_id": self.golden_trace_id,
            "golden_command_id": self.golden_command_id,
            "golden_authority_class": self.golden_authority_class,
            "golden_status": self.golden_status,
            "fingerprint": self.fingerprint,
            "evidence_detail": self.evidence_detail,
            "declared_scope": list(self.declared_scope),
        }
