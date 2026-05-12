"""Phase 6.5 L3 escalation request queue.

When live autonomy is at L3 and a sustained metric deficit has resisted
L1 research (veto + long duration + low win rate), the system *asks the
human* for authority to attempt a more invasive code change. This
module owns the queue of those requests, their audit trail, and the
narrow per-request path scope that approved escalations run under.

Invariants (enforced by tests in
``brain/tests/test_l3_escalation.py``)
-------------------------------------
- Auto-generated escalation requests require live ``autonomy_level >=
  3`` at submission time. Attestation alone (``prior_attested_ok``)
  does NOT generate escalation requests; it only unlocks *manual* L3
  promotion via the API. This is the "two escalation generation modes"
  rule from the plan.
- Each request carries a ``declared_scope`` — a narrow allowlist of
  path prefixes the eventual patch is permitted to touch. There is NO
  global mutation of ``ALLOWED_PATHS``. The scope is threaded into
  ``SelfImprovementOrchestrator.attempt_improvement`` per request and
  validated by ``PatchPlan.validate_scope`` on a per-call basis.
- Per-metric rate limit (1 live request per 24h). A new request for
  the same metric while a live request exists is rejected, not
  silently dropped.
- All lifecycle transitions (submit / approve / reject / expire /
  rollback) append an entry to ``~/.jarvis/escalation_activity.jsonl``
  and emit an event on the bus.

Persistence
-----------
Pending queue: ``~/.jarvis/pending_escalations.json`` — JSON list,
atomic tempfile + replace.
Activity log: ``~/.jarvis/escalation_activity.jsonl`` — one JSON line
per event, append-only.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rate limit window: one live auto-generated escalation per metric per day.
PER_METRIC_RATE_LIMIT_S = 24 * 3600.0

# Default expiry: a request that sits unapproved for 24h becomes
# ``expired`` and is no longer activatable. The operator can still
# resubmit manually after addressing whatever made the deficit stale.
DEFAULT_EXPIRY_S = 24 * 3600.0

_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))
_PENDING_PATH = _JARVIS_DIR / "pending_escalations.json"
_ACTIVITY_PATH = _JARVIS_DIR / "escalation_activity.jsonl"


# Metric -> default target_module / declared_scope. The scope is
# intentionally narrow; each entry is exactly one ALLOWED_PATHS prefix
# (see brain/self_improve/patch_plan.py). Metrics without a registered
# module are refused at submit time — the operator must add an entry
# rather than let the system guess.
METRIC_ESCALATION_POLICY: dict[str, dict[str, Any]] = {
    "confidence_volatility": {
        "target_module": "brain/consciousness/",
        "declared_scope": ["brain/consciousness/"],
    },
    "tick_p95_ms": {
        "target_module": "brain/consciousness/",
        "declared_scope": ["brain/consciousness/"],
    },
    "reasoning_coherence": {
        "target_module": "brain/reasoning/",
        "declared_scope": ["brain/reasoning/"],
    },
    "processing_health": {
        "target_module": "brain/consciousness/",
        "declared_scope": ["brain/consciousness/"],
    },
    "shadow_default_win_rate": {
        "target_module": "brain/policy/",
        "declared_scope": ["brain/policy/"],
    },
    "memory_recall_miss_rate": {
        "target_module": "brain/memory/",
        "declared_scope": ["brain/memory/"],
    },
    "friction_rate": {
        "target_module": "brain/consciousness/",
        "declared_scope": ["brain/consciousness/"],
    },
    "barge_in_rate": {
        "target_module": "brain/perception/",
        "declared_scope": ["brain/perception/"],
    },
}


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------


@dataclass
class EscalationRequest:
    """Structured description of what needs changing and why.

    Fields chosen to match the archived ``docs/archive/nextphase.md``
    specification plus the Phase 6.5 additions (``declared_scope``,
    ``expires_at``).
    """

    id: str = field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:10]}")
    metric: str = ""
    metric_context_summary: str = ""
    severity: str = "high"  # low | medium | high
    target_module: str = ""
    suggested_fix: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    declared_scope: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    source: str = "metric_trigger"  # metric_trigger | operator
    # Live autonomy level at the moment of submission. Required for
    # auditability — lets us tell apart "L3-era request" from a future
    # regression where the L3 gate is bypassed.
    submitted_autonomy_level: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EscalationRecord:
    """Request + lifecycle state as stored on disk."""

    request: EscalationRequest
    status: str = "pending"  # pending | approved | rejected | rolled_back | expired
    approved_by: str = ""
    approved_at: float = 0.0
    approval_reason: str = ""
    rejected_by: str = ""
    rejected_at: float = 0.0
    rejection_reason: str = ""
    improvement_record_id: str = ""
    outcome: str = ""  # clean | rolled_back | empty string while pending
    rollback_reason: str = ""
    rolled_back_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "status": self.status,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "approval_reason": self.approval_reason,
            "rejected_by": self.rejected_by,
            "rejected_at": self.rejected_at,
            "rejection_reason": self.rejection_reason,
            "improvement_record_id": self.improvement_record_id,
            "outcome": self.outcome,
            "rollback_reason": self.rollback_reason,
            "rolled_back_at": self.rolled_back_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EscalationRecord":
        req_d = d.get("request", {})
        req = EscalationRequest(**{
            k: v for k, v in req_d.items()
            if k in EscalationRequest.__dataclass_fields__
        })
        return cls(
            request=req,
            status=d.get("status", "pending"),
            approved_by=d.get("approved_by", ""),
            approved_at=float(d.get("approved_at", 0.0) or 0.0),
            approval_reason=d.get("approval_reason", ""),
            rejected_by=d.get("rejected_by", ""),
            rejected_at=float(d.get("rejected_at", 0.0) or 0.0),
            rejection_reason=d.get("rejection_reason", ""),
            improvement_record_id=d.get("improvement_record_id", ""),
            outcome=d.get("outcome", ""),
            rollback_reason=d.get("rollback_reason", ""),
            rolled_back_at=float(d.get("rolled_back_at", 0.0) or 0.0),
        )


class EscalationStoreError(Exception):
    pass


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class EscalationStore:
    """Persistent queue of L3 escalation requests.

    Not thread-safe for concurrent writers; the dashboard app runs on a
    single event loop and this store is called from the orchestrator
    tick / API handlers, both of which are serialized per-request.
    """

    def __init__(
        self,
        pending_path: Path | str | None = None,
        activity_path: Path | str | None = None,
        rate_limit_s: float = PER_METRIC_RATE_LIMIT_S,
        default_expiry_s: float = DEFAULT_EXPIRY_S,
    ) -> None:
        self._pending_path = Path(pending_path) if pending_path else _PENDING_PATH
        self._activity_path = Path(activity_path) if activity_path else _ACTIVITY_PATH
        self._rate_limit_s = float(rate_limit_s)
        self._default_expiry_s = float(default_expiry_s)

    @property
    def pending_path(self) -> Path:
        return self._pending_path

    @property
    def activity_path(self) -> Path:
        return self._activity_path

    # -- read ---------------------------------------------------------------

    def load_all(self) -> list[EscalationRecord]:
        if not self._pending_path.exists():
            return []
        try:
            raw = json.loads(self._pending_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Escalation store corrupt at %s: %s — treating as empty",
                self._pending_path, exc,
            )
            return []
        if not isinstance(raw, list):
            return []
        out: list[EscalationRecord] = []
        for entry in raw:
            if isinstance(entry, dict):
                try:
                    out.append(EscalationRecord.from_dict(entry))
                except (TypeError, KeyError) as exc:
                    logger.warning("Skipping malformed escalation entry: %s", exc)
        return out

    def list_pending(self) -> list[EscalationRecord]:
        self.prune_expired()
        return [r for r in self.load_all() if r.status == "pending"]

    def get(self, request_id: str) -> EscalationRecord | None:
        for r in self.load_all():
            if r.request.id == request_id:
                return r
        return None

    # -- write --------------------------------------------------------------

    def submit(self, req: EscalationRequest) -> EscalationRecord:
        """Append a new request to the queue, enforcing rate limit and policy.

        Raises :class:`EscalationStoreError` if the metric has no
        registered escalation policy, the metric is rate-limited, or
        ``submitted_autonomy_level < 3``.
        """
        if req.metric not in METRIC_ESCALATION_POLICY:
            raise EscalationStoreError(
                f"Metric {req.metric!r} has no escalation policy registered. "
                "Add it to METRIC_ESCALATION_POLICY before firing an escalation."
            )
        if req.submitted_autonomy_level < 3:
            raise EscalationStoreError(
                "Auto-generated escalation requires live autonomy_level >= 3 "
                f"(got {req.submitted_autonomy_level}). Attestation alone does "
                "not satisfy this gate; the operator must first manually "
                "promote to L3 via POST /api/autonomy/level."
            )
        if not req.declared_scope:
            raise EscalationStoreError(
                "EscalationRequest.declared_scope must be a non-empty list of "
                "path prefixes (no global ALLOWED_PATHS mutation is permitted)"
            )

        self.prune_expired()

        now = time.time()
        records = self.load_all()
        for r in records:
            if r.request.metric != req.metric:
                continue
            if r.status == "pending":
                raise EscalationStoreError(
                    f"Live escalation already exists for metric {req.metric!r} "
                    f"(id={r.request.id}); resolve it before submitting another"
                )
            if r.status in {"approved", "rolled_back", "rejected", "expired", "parked"}:
                last_event = max(
                    r.approved_at, r.rejected_at, r.rolled_back_at,
                    r.request.created_at,
                )
                if now - last_event < self._rate_limit_s:
                    raise EscalationStoreError(
                        f"Metric {req.metric!r} rate-limited: last terminal "
                        f"event {now - last_event:.0f}s ago "
                        f"(need {self._rate_limit_s:.0f}s)"
                    )

        if req.expires_at <= 0:
            req.expires_at = now + self._default_expiry_s

        record = EscalationRecord(request=req, status="pending")
        records.append(record)
        self._atomic_write(records)
        self._log_activity(
            "submit", record,
            submitted_autonomy_level=req.submitted_autonomy_level,
        )
        return record

    def approve(
        self,
        request_id: str,
        *,
        approved_by: str,
        approval_reason: str,
        improvement_record_id: str,
    ) -> EscalationRecord:
        records = self.load_all()
        target: EscalationRecord | None = None
        for r in records:
            if r.request.id == request_id:
                target = r
                break
        if target is None:
            raise EscalationStoreError(f"Unknown escalation id={request_id!r}")
        if target.status != "pending":
            raise EscalationStoreError(
                f"Cannot approve escalation id={request_id!r} in status "
                f"{target.status!r}; only 'pending' is valid"
            )
        target.status = "approved"
        target.approved_by = approved_by
        target.approved_at = time.time()
        target.approval_reason = approval_reason
        target.improvement_record_id = improvement_record_id
        self._atomic_write(records)
        self._log_activity("approve", target)
        return target

    def reject(
        self,
        request_id: str,
        *,
        rejected_by: str,
        rejection_reason: str,
    ) -> EscalationRecord:
        records = self.load_all()
        target: EscalationRecord | None = None
        for r in records:
            if r.request.id == request_id:
                target = r
                break
        if target is None:
            raise EscalationStoreError(f"Unknown escalation id={request_id!r}")
        if target.status != "pending":
            raise EscalationStoreError(
                f"Cannot reject escalation id={request_id!r} in status "
                f"{target.status!r}; only 'pending' is valid"
            )
        target.status = "rejected"
        target.rejected_by = rejected_by
        target.rejected_at = time.time()
        target.rejection_reason = rejection_reason
        self._atomic_write(records)
        self._log_activity("reject", target)
        return target

    def mark_outcome(
        self,
        request_id: str,
        outcome: str,
        *,
        rollback_reason: str = "",
    ) -> EscalationRecord:
        """Record the runtime outcome of an approved escalation.

        ``outcome`` must be ``"clean"`` (patch applied and health ok)
        or ``"rolled_back"`` (post-apply health failed and the patch
        was reverted). Called by the approval flow after the
        SelfImprovementOrchestrator runs.
        """
        if outcome not in {"clean", "rolled_back", "parked"}:
            raise EscalationStoreError(
                f"outcome must be 'clean' | 'rolled_back' | 'parked' (got {outcome!r})"
            )
        records = self.load_all()
        target: EscalationRecord | None = None
        for r in records:
            if r.request.id == request_id:
                target = r
                break
        if target is None:
            raise EscalationStoreError(f"Unknown escalation id={request_id!r}")
        if target.status != "approved":
            raise EscalationStoreError(
                f"mark_outcome requires status 'approved' (got {target.status!r})"
            )
        target.outcome = outcome
        if outcome == "rolled_back":
            target.status = "rolled_back"
            target.rollback_reason = rollback_reason
            target.rolled_back_at = time.time()
        elif outcome == "parked":
            target.status = "parked"
            target.rollback_reason = rollback_reason
        self._atomic_write(records)
        self._log_activity("outcome", target, outcome=outcome)
        return target

    def prune_expired(self, event_emit: Any = None) -> int:
        """Mark overdue pending requests as expired and emit audit events.

        ``event_emit`` is an optional callable ``(event_type, **kwargs)``
        used only for tests; in production, the lazy fallback imports
        the module-level event bus. Bus emission failures must never
        prevent the on-disk state transition.
        """
        records = self.load_all()
        now = time.time()
        changed = 0
        expired_records: list[EscalationRecord] = []
        for r in records:
            if r.status == "pending" and r.request.expires_at and now >= r.request.expires_at:
                r.status = "expired"
                changed += 1
                self._log_activity("expire", r)
                expired_records.append(r)
        if changed:
            self._atomic_write(records)
            if event_emit is None:
                try:
                    from consciousness.events import event_bus
                    event_emit = lambda et, **kw: event_bus.emit(et, **kw)  # noqa: E731
                except Exception:
                    event_emit = None
            if event_emit is not None:
                for r in expired_records:
                    try:
                        from consciousness.events import AUTONOMY_ESCALATION_EXPIRED
                        event_emit(
                            AUTONOMY_ESCALATION_EXPIRED,
                            escalation_id=r.request.id,
                            metric=r.request.metric,
                            severity=r.request.severity,
                            expires_at=r.request.expires_at,
                            submitted_autonomy_level=r.request.submitted_autonomy_level,
                        )
                    except Exception:
                        logger.exception("Failed to emit AUTONOMY_ESCALATION_EXPIRED")
        return changed

    # -- internals ----------------------------------------------------------

    def _atomic_write(self, records: list[EscalationRecord]) -> None:
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in records]
        fd, tmp_path = tempfile.mkstemp(
            prefix=".escalation_", suffix=".json",
            dir=str(self._pending_path.parent),
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._pending_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _log_activity(
        self,
        action: str,
        record: EscalationRecord,
        **extra: Any,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "action": action,
            "escalation_id": record.request.id,
            "metric": record.request.metric,
            "status": record.status,
        }
        entry.update(extra)
        try:
            self._activity_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._activity_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Failed to append escalation activity: %s", exc)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def approve_and_apply_escalation(
    store: "EscalationStore",
    *,
    request_id: str,
    approved_by: str,
    approval_reason: str,
    self_improve_orchestrator: Any,
    ollama_client: Any = None,
    event_emit: Any = None,
) -> dict[str, Any]:
    """Approve a pending escalation and route it through self-improvement.

    Builds an ``ImprovementRequest`` with ``declared_scope`` set to the
    escalation's narrow path list — NOT a global mutation of
    ``ALLOWED_PATHS``. Runs ``attempt_improvement(manual=True)``, reads
    the resulting record status, and calls :meth:`EscalationStore.mark_outcome`
    to record whether the patch landed cleanly or was rolled back by
    post-apply health.

    Parameters
    ----------
    store
        Persistence store holding the pending request.
    request_id
        Escalation request id to approve.
    approved_by, approval_reason
        Caller identity and human-supplied reason for the audit log.
    self_improve_orchestrator
        The live ``SelfImprovementOrchestrator`` instance. Passed in
        rather than imported to keep this module testable without a
        running improvement pipeline.
    ollama_client
        Forwarded to ``attempt_improvement``. May be ``None`` in tests.
    event_emit
        Optional callable ``event_emit(event_type, **kwargs)`` used to
        surface ``AUTONOMY_ESCALATION_APPROVED`` and
        ``AUTONOMY_ESCALATION_ROLLED_BACK``. When ``None``, falls back
        to the global event bus.

    Returns
    -------
    dict
        Summary suitable for an API response. Shape::

            {
                "status": "approved" | "rolled_back",
                "escalation_id": "...",
                "improvement_record_id": "...",
                "outcome": "clean" | "rolled_back",
                "rollback_reason": "...",
            }
    """
    from self_improve.improvement_request import ImprovementRequest

    if event_emit is None:
        from consciousness.events import event_bus
        event_emit = lambda et, **kw: event_bus.emit(et, **kw)  # noqa: E731

    record = store.get(request_id)
    if record is None:
        raise EscalationStoreError(f"Unknown escalation id={request_id!r}")
    if record.status != "pending":
        raise EscalationStoreError(
            f"Cannot approve escalation id={request_id!r} in status "
            f"{record.status!r}"
        )

    req = record.request
    improvement_request = ImprovementRequest(
        type="architecture_improvement",
        target_module=req.target_module,
        description=(
            f"L3 escalation approval for {req.metric}: "
            f"{req.metric_context_summary}"
        ),
        evidence=list(req.evidence_refs) + [
            f"metric:{req.metric}",
            f"deficit_summary:{req.metric_context_summary[:120]}",
        ],
        priority=0.9,
        requires_approval=False,
        manual=True,
        declared_scope=list(req.declared_scope),
        evidence_detail={
            "intent_type": f"escalation:{req.metric}",
            "escalation_id": req.id,
            "submitted_autonomy_level": req.submitted_autonomy_level,
            "approved_by": approved_by,
        },
    )

    # Mark the escalation as approved BEFORE running the pipeline so
    # concurrent readers see a consistent lifecycle. The improvement
    # record id is filled in after attempt_improvement returns.
    store.approve(
        request_id,
        approved_by=approved_by,
        approval_reason=approval_reason,
        improvement_record_id="",
    )
    try:
        from consciousness.events import AUTONOMY_ESCALATION_APPROVED
        event_emit(
            AUTONOMY_ESCALATION_APPROVED,
            escalation_id=req.id,
            metric=req.metric,
            approved_by=approved_by,
            approval_reason=approval_reason,
            declared_scope=list(req.declared_scope),
        )
    except Exception:
        logger.exception("Failed to emit AUTONOMY_ESCALATION_APPROVED")

    improvement_record = None
    try:
        improvement_record = await self_improve_orchestrator.attempt_improvement(
            improvement_request,
            ollama_client=ollama_client,
            dry_run=False,
            manual=True,
        )
    except Exception as exc:
        # attempt_improvement itself threw. Always record a rollback
        # outcome so the record cannot be left stuck in "approved"
        # with no resolution.
        rollback_reason = f"attempt_improvement_exception:{exc!r}"[:500]
        store.mark_outcome(
            request_id, "rolled_back", rollback_reason=rollback_reason,
        )
        try:
            from consciousness.events import AUTONOMY_ESCALATION_ROLLED_BACK
            event_emit(
                AUTONOMY_ESCALATION_ROLLED_BACK,
                escalation_id=req.id,
                metric=req.metric,
                rollback_reason=rollback_reason,
                improvement_record_id=improvement_request.id,
            )
        except Exception:
            logger.exception("Failed to emit AUTONOMY_ESCALATION_ROLLED_BACK")
        raise

    # Reload the record after the async call in case another thread
    # mutated it; update the improvement_record id back.
    records = store.load_all()
    for r in records:
        if r.request.id == req.id:
            r.improvement_record_id = getattr(improvement_request, "id", "")
            break
    store._atomic_write(records)  # type: ignore[attr-defined]

    status = getattr(improvement_record, "status", "")
    rollback_reason = ""
    if status in {"rolled_back", "reverted"} or getattr(
        improvement_record, "was_rolled_back", False
    ):
        outcome = "rolled_back"
        rollback_reason = getattr(
            improvement_record, "rollback_reason", "post_apply_health_failed"
        )
    elif status in {"applied", "promoted", "verified"}:
        outcome = "clean"
    elif status == "awaiting_approval":
        # Phase 6.5: the escalation approval endpoint IS the human gate.
        # Reaching awaiting_approval here means the self-improve pipeline
        # escalated a second time on its own — e.g. patch.check_dangerous()
        # fired on mutation_governor / kernel_mutator / persistence. The
        # patch is NOT rolled back; it is parked in the self-improve
        # approval queue for the operator to resolve via the self-improve
        # dashboard. We mark the escalation "parked" so the rate limiter
        # counts it as terminal, but the audit trail does not lie and
        # claim it was rolled back.
        outcome = "parked"
        rollback_reason = f"improvement_status:awaiting_approval"
    else:
        # dry_run / failed / unknown — patch never landed.
        outcome = "rolled_back"
        rollback_reason = f"improvement_status:{status or 'unknown'}"

    store.mark_outcome(
        request_id, outcome, rollback_reason=rollback_reason,
    )
    if outcome == "rolled_back":
        try:
            from consciousness.events import AUTONOMY_ESCALATION_ROLLED_BACK
            event_emit(
                AUTONOMY_ESCALATION_ROLLED_BACK,
                escalation_id=req.id,
                metric=req.metric,
                rollback_reason=rollback_reason,
                improvement_record_id=improvement_request.id,
            )
        except Exception:
            logger.exception("Failed to emit AUTONOMY_ESCALATION_ROLLED_BACK")
    elif outcome == "parked":
        try:
            from consciousness.events import AUTONOMY_ESCALATION_PARKED
            event_emit(
                AUTONOMY_ESCALATION_PARKED,
                escalation_id=req.id,
                metric=req.metric,
                park_reason=rollback_reason,
                improvement_record_id=improvement_request.id,
            )
        except Exception:
            logger.exception("Failed to emit AUTONOMY_ESCALATION_PARKED")

    result_status = {
        "clean": "approved",
        "rolled_back": "rolled_back",
        "parked": "parked",
    }[outcome]
    return {
        "status": result_status,
        "escalation_id": req.id,
        "improvement_record_id": improvement_request.id,
        "outcome": outcome,
        "rollback_reason": rollback_reason,
    }


def reject_escalation(
    store: "EscalationStore",
    *,
    request_id: str,
    rejected_by: str,
    rejection_reason: str,
    event_emit: Any = None,
) -> EscalationRecord:
    """Reject a pending escalation and emit the audit event."""
    if event_emit is None:
        from consciousness.events import event_bus
        event_emit = lambda et, **kw: event_bus.emit(et, **kw)  # noqa: E731

    rec = store.reject(
        request_id,
        rejected_by=rejected_by,
        rejection_reason=rejection_reason,
    )
    try:
        from consciousness.events import AUTONOMY_ESCALATION_REJECTED
        event_emit(
            AUTONOMY_ESCALATION_REJECTED,
            escalation_id=rec.request.id,
            metric=rec.request.metric,
            rejected_by=rejected_by,
            rejection_reason=rejection_reason,
        )
    except Exception:
        logger.exception("Failed to emit AUTONOMY_ESCALATION_REJECTED")
    return rec


def submit_and_emit(
    store: "EscalationStore",
    req: EscalationRequest,
    *,
    event_emit: Any = None,
) -> EscalationRecord:
    """Submit an escalation request and emit AUTONOMY_ESCALATION_REQUESTED.

    Convenience wrapper for metric-trigger wiring and API handlers so
    event emission stays consistent across callers.
    """
    if event_emit is None:
        from consciousness.events import event_bus
        event_emit = lambda et, **kw: event_bus.emit(et, **kw)  # noqa: E731

    rec = store.submit(req)
    try:
        from consciousness.events import AUTONOMY_ESCALATION_REQUESTED
        event_emit(
            AUTONOMY_ESCALATION_REQUESTED,
            escalation_id=rec.request.id,
            metric=rec.request.metric,
            severity=rec.request.severity,
            declared_scope=list(rec.request.declared_scope),
            target_module=rec.request.target_module,
            submitted_autonomy_level=rec.request.submitted_autonomy_level,
            expires_at=rec.request.expires_at,
        )
    except Exception:
        logger.exception("Failed to emit AUTONOMY_ESCALATION_REQUESTED")
    return rec


def build_request_from_metric_deficit(
    *,
    metric: str,
    current_value: float,
    threshold: float,
    deficit_duration_s: float,
    l1_attempts: int,
    win_rate: float,
    live_autonomy_level: int,
    evidence_refs: list[str] | None = None,
) -> EscalationRequest:
    """Construct an EscalationRequest from a deficit snapshot.

    The metric policy table supplies ``target_module`` and
    ``declared_scope``. The caller (metric trigger wire) is responsible
    for verifying the trigger conditions before calling this helper;
    this function only *shapes* the request, it does not decide whether
    the deficit deserves escalation.
    """
    if metric not in METRIC_ESCALATION_POLICY:
        raise EscalationStoreError(
            f"No escalation policy for metric {metric!r}"
        )
    pol = METRIC_ESCALATION_POLICY[metric]
    summary = (
        f"Metric {metric} has been in deficit for {deficit_duration_s:.0f}s "
        f"(current={current_value:.3f}, threshold={threshold:.3f}). "
        f"L1 research has run {l1_attempts} time(s) with {win_rate:.0%} win rate "
        "and has not resolved the deficit."
    )
    return EscalationRequest(
        metric=metric,
        metric_context_summary=summary,
        severity="high" if deficit_duration_s > 480.0 * 2 else "medium",
        target_module=pol["target_module"],
        suggested_fix=(
            "Human investigation requested: L1 research has been exhausted "
            f"for {metric}."
        ),
        evidence_refs=list(evidence_refs or []),
        declared_scope=list(pol["declared_scope"]),
        submitted_autonomy_level=live_autonomy_level,
    )
