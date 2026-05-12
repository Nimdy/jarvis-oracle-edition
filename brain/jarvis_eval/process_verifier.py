"""Process Verifier — evaluates process contracts against live data.

Reads events and snapshots from the eval store and determines whether
each architectural process contract is passing, failing, or not applicable.

The verifier is read-only: it never writes to memory, events, or core state.
It produces ProcessVerdict records that feed the dashboard adapter.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from jarvis_eval.process_contracts import (
    ALL_CONTRACTS,
    PROCESS_GROUPS,
    ProcessContract,
    ContractStatus,
    get_contracts_by_group,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessVerdict:
    """Result of evaluating a single contract."""

    contract_id: str
    group: str
    label: str
    status: ContractStatus
    last_verified: float = 0.0
    first_seen: float = 0.0
    consecutive_passes: int = 0
    consecutive_fails: int = 0
    evidence: str = ""
    training_stage: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Aggregate result of one verification pass."""

    verdicts: list[ProcessVerdict] = field(default_factory=list)
    total_contracts: int = 0
    applicable_contracts: int = 0
    passing_contracts: int = 0
    failing_contracts: int = 0
    awaiting_contracts: int = 0
    coverage_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "total_contracts": self.total_contracts,
            "applicable_contracts": self.applicable_contracts,
            "passing_contracts": self.passing_contracts,
            "failing_contracts": self.failing_contracts,
            "awaiting_contracts": self.awaiting_contracts,
            "coverage_pct": round(self.coverage_pct, 1),
            "timestamp": self.timestamp,
        }


class ProcessVerifier:
    """Evaluates all process contracts against accumulated eval data."""

    def __init__(self) -> None:
        self._verdict_history: dict[str, ProcessVerdict] = {}
        self._seen_modes: set[str] = set()
        self._seen_event_types: set[str] = set()
        self._last_result: VerificationResult | None = None
        self._run_count: int = 0
        self._hydrated: bool = False

    def hydrate_from_history(self, all_events: list[dict[str, Any]]) -> None:
        """Seed seen_event_types and seen_modes from full event history.

        Call once at startup so that events which have already scrolled
        out of the recent-tail window are still counted as "seen".
        """
        for ev in all_events:
            et = ev.get("event_type", "")
            if et:
                self._seen_event_types.add(et)
            mode = ev.get("mode", "")
            if mode:
                self._seen_modes.add(mode)
        self._hydrated = True
        logger.info(
            "PVL hydrated from %d events — %d event types, %d modes",
            len(all_events), len(self._seen_event_types), len(self._seen_modes),
        )

    def verify(
        self,
        recent_events: list[dict[str, Any]],
        latest_snapshots: dict[str, dict[str, Any]],
        current_mode: str = "",
    ) -> VerificationResult:
        """Run one verification pass across all contracts."""
        now = time.time()
        self._run_count += 1

        if current_mode:
            self._seen_modes.add(current_mode)

        event_types_in_window: set[str] = set()
        for ev in recent_events:
            et = ev.get("event_type", "")
            if et:
                event_types_in_window.add(et)
                self._seen_event_types.add(et)
            mode = ev.get("mode", "")
            if mode:
                self._seen_modes.add(mode)

        verdicts: list[ProcessVerdict] = []
        passing = 0
        failing = 0
        awaiting = 0
        applicable = 0

        for contract in ALL_CONTRACTS:
            prev = self._verdict_history.get(contract.contract_id)

            status = self._evaluate_contract(
                contract, event_types_in_window, latest_snapshots,
            )

            if status == "not_applicable":
                verdict = ProcessVerdict(
                    contract_id=contract.contract_id,
                    group=contract.group,
                    label=contract.label,
                    status="not_applicable",
                    last_verified=now,
                    first_seen=prev.first_seen if prev else 0.0,
                    training_stage=contract.training_stage,
                    evidence="Mode prerequisites not met",
                )
            else:
                applicable += 1

                if status == "pass":
                    if prev and prev.status == "pass" and prev.first_seen > 0:
                        first = prev.first_seen
                    else:
                        first = now
                    cons_pass = (prev.consecutive_passes + 1) if prev and prev.status == "pass" else 1
                    verdict = ProcessVerdict(
                        contract_id=contract.contract_id,
                        group=contract.group,
                        label=contract.label,
                        status="pass",
                        last_verified=now,
                        first_seen=first,
                        consecutive_passes=cons_pass,
                        consecutive_fails=0,
                        training_stage=contract.training_stage,
                        evidence=self._describe_evidence(contract, latest_snapshots),
                    )
                    passing += 1

                elif status == "awaiting":
                    verdict = ProcessVerdict(
                        contract_id=contract.contract_id,
                        group=contract.group,
                        label=contract.label,
                        status="awaiting",
                        last_verified=now,
                        first_seen=prev.first_seen if prev else 0.0,
                        training_stage=contract.training_stage,
                        evidence="No data yet",
                    )
                    awaiting += 1

                else:
                    cons_fail = (prev.consecutive_fails + 1) if prev and prev.status == "fail" else 1
                    verdict = ProcessVerdict(
                        contract_id=contract.contract_id,
                        group=contract.group,
                        label=contract.label,
                        status="fail",
                        last_verified=now,
                        first_seen=prev.first_seen if prev and prev.first_seen > 0 else now,
                        consecutive_passes=0,
                        consecutive_fails=cons_fail,
                        training_stage=contract.training_stage,
                        evidence=self._describe_failure(contract, latest_snapshots),
                    )
                    failing += 1

            self._verdict_history[contract.contract_id] = verdict
            verdicts.append(verdict)

        coverage = (passing / applicable * 100.0) if applicable > 0 else 0.0

        result = VerificationResult(
            verdicts=verdicts,
            total_contracts=len(ALL_CONTRACTS),
            applicable_contracts=applicable,
            passing_contracts=passing,
            failing_contracts=failing,
            awaiting_contracts=awaiting,
            coverage_pct=coverage,
            timestamp=now,
        )
        self._last_result = result
        return result

    def _evaluate_contract(
        self,
        contract: ProcessContract,
        event_types: set[str],
        snapshots: dict[str, dict[str, Any]],
    ) -> ContractStatus:
        """Evaluate a single contract. Returns status."""

        if not self._is_applicable(contract):
            return "not_applicable"

        if contract.method == "event":
            return self._check_event(contract, event_types)
        elif contract.method == "snapshot":
            return self._check_snapshot(contract, snapshots)
        elif contract.method == "compound":
            ev_ok = self._check_event(contract, event_types)
            snap_ok = self._check_snapshot(contract, snapshots)
            if ev_ok == "pass" and snap_ok == "pass":
                return "pass"
            if ev_ok == "awaiting" or snap_ok == "awaiting":
                return "awaiting"
            return "fail"

        return "awaiting"

    def _is_applicable(self, contract: ProcessContract) -> bool:
        """Check if the contract's mode prerequisites are met."""
        if contract.required_modes:
            if not self._seen_modes.intersection(contract.required_modes):
                return False
        if contract.excluded_modes:
            if self._seen_modes and self._seen_modes.issubset(contract.excluded_modes):
                return False
        return True

    def _check_event(
        self, contract: ProcessContract, event_types: set[str],
    ) -> ContractStatus:
        """Check if the required event has fired.

        Checks both the current window AND the accumulated session history.
        Once an event type is seen in any window during this session, it
        stays seen — events don't "un-happen."
        """
        if not contract.event_type:
            return "awaiting"

        if contract.event_type in event_types:
            return "pass"

        if contract.event_type in self._seen_event_types:
            return "pass"

        if not self._seen_event_types:
            return "awaiting"

        return contract.missing_event_status

    def _check_snapshot(
        self, contract: ProcessContract, snapshots: dict[str, dict[str, Any]],
    ) -> ContractStatus:
        """Check if a snapshot metric meets the threshold."""
        if not contract.snapshot_source or not contract.snapshot_key:
            return "awaiting"

        source_data = snapshots.get(contract.snapshot_source)
        if source_data is None:
            return "awaiting"

        value = _resolve_dotted_key(source_data, contract.snapshot_key)
        if value is None:
            return "awaiting"

        try:
            num = float(value) if not isinstance(value, (int, float)) else value
        except (TypeError, ValueError):
            if value:
                return "pass"
            return "awaiting"

        if num < contract.snapshot_min:
            return "fail"

        if contract.snapshot_max is not None and num > contract.snapshot_max:
            return "fail"

        return "pass"

    def _describe_evidence(
        self, contract: ProcessContract, snapshots: dict[str, dict[str, Any]],
    ) -> str:
        """Build a short evidence string for a passing contract."""
        if contract.method == "event":
            return f"Event {contract.event_type} observed"
        elif contract.method == "snapshot":
            src = snapshots.get(contract.snapshot_source or "", {})
            val = _resolve_dotted_key(src, contract.snapshot_key or "")
            return f"{contract.snapshot_key}={val}"
        return "compound check passed"

    def _describe_failure(
        self,
        contract: ProcessContract,
        snapshots: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Build a short failure description."""
        if contract.method == "event":
            return f"Event {contract.event_type} not observed"
        elif contract.method == "snapshot":
            if snapshots and contract.snapshot_source and contract.snapshot_key:
                src = snapshots.get(contract.snapshot_source or "", {})
                val = _resolve_dotted_key(src, contract.snapshot_key or "")
                if val is not None:
                    try:
                        num = float(val) if not isinstance(val, (int, float)) else float(val)
                        if contract.snapshot_max is not None and num > contract.snapshot_max:
                            return (
                                f"{contract.snapshot_key} above max "
                                f"({contract.snapshot_max}), got {num}"
                            )
                        if num < contract.snapshot_min:
                            return (
                                f"{contract.snapshot_key} below threshold "
                                f"({contract.snapshot_min}), got {num}"
                            )
                    except (TypeError, ValueError):
                        pass
            return f"{contract.snapshot_key} below threshold ({contract.snapshot_min})"
        return "compound check failed"

    def get_last_result(self) -> VerificationResult | None:
        return self._last_result

    def get_stats(self) -> dict[str, Any]:
        return {
            "run_count": self._run_count,
            "seen_modes": sorted(self._seen_modes),
            "seen_event_types_count": len(self._seen_event_types),
            "verdict_count": len(self._verdict_history),
        }


def _resolve_dotted_key(data: dict[str, Any], key: str) -> Any:
    """Resolve a dotted key path like 'ranker.train_count' in nested dicts."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current
