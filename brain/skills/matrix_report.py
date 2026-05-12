"""Matrix Report — standard result schema for all Matrix Protocol learning outcomes.

Every Matrix Protocol learning job produces a MatrixReport at the end of
verification, regardless of protocol family.  This gives the user (and the
dashboard) a uniform, interpretable summary of what was learned, how it was
tested, and what the system is now allowed to claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatrixReport:
    target_skill: str
    protocol_selected: str          # "SK-001", "SK-002", "SK-003", "SK-004"
    capability_class: str           # "procedural", "perceptual", "control"
    data_used: dict[str, Any] = field(default_factory=dict)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    claimability: str = "unverified"  # "unverified", "verified_limited", "verified_operational"
    specialist_born: bool = False
    next_step: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_skill": self.target_skill,
            "protocol_selected": self.protocol_selected,
            "capability_class": self.capability_class,
            "data_used": self.data_used,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "failure_reasons": self.failure_reasons,
            "claimability": self.claimability,
            "specialist_born": self.specialist_born,
            "next_step": self.next_step,
            "timestamp": self.timestamp,
        }

    def summary_text(self) -> str:
        """Human-readable summary for TTS delivery."""
        if self.checks_failed:
            status = "did not fully pass"
            detail = f"Failed checks: {', '.join(self.checks_failed[:3])}."
            if self.failure_reasons:
                detail += f" Reason: {self.failure_reasons[0]}"
        elif self.claimability == "verified_operational":
            status = "passed all checks"
            detail = f"Protocol {self.protocol_selected} fully satisfied."
        elif self.claimability == "verified_limited":
            status = "passed with limitations"
            detail = f"Protocol {self.protocol_selected} passed, but with caveats."
        else:
            status = "is still unverified"
            detail = "More data or testing is needed."

        specialist_note = ""
        if self.specialist_born:
            specialist_note = " A specialist neural network was created for this capability."

        next_note = ""
        if self.next_step:
            next_note = f" Next step: {self.next_step}"

        return (
            f"Learning report for {self.target_skill}: {status}. "
            f"{detail}{specialist_note}{next_note}"
        )
