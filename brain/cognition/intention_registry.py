"""Intention Registry — canonical truth layer for JARVIS's outgoing commitments.

Stage 0 infrastructure: records every commitment JARVIS makes in outgoing speech
(e.g. "I'll get back to you", "give me a moment") and binds it to the real
background job id that was spawned that same turn. On job completion, the
intention is resolved. On boot, open intentions are restored.

Design contract:
  - Symbolic truth layer (Pillar 3). No neural inference here.
  - No delivery mechanism. The registry does not speak; it only records.
  - Append-only outcome log (`intention_outcomes.jsonl`) for future NN training.
  - Current-state snapshot (`intention_registry.json`) for boot restore.
  - Thread-safe; all reads/writes behind a single lock.
  - Honest-failure semantics: outcomes are one of
      resolved | failed | stale | abandoned
    with a reason string. "silently dropped" is impossible by construction.

Integration points (all callers):
  - `conversation_handler.py`: register() after CommitmentExtractor matches +
    a real backing_job_id is captured from this turn's dispatch.
  - `autonomy/orchestrator.py`: resolve() when an AUTONOMY_RESEARCH_COMPLETED
    fires with a linked backing_job_id.
  - `conversation_handler.py` LIBRARY_INGEST branch: resolve() on ingest
    thread success/failure.
  - `consciousness/consciousness_system.py`: tick stale sweep at 300s cadence.
  - `reasoning/bounded_response.py`: get_status() for self_status MeaningFrame.
  - `dashboard/app.py`: /api/intentions for observability.
  - `jarvis_eval/process_contracts.py`: intention_truth PVL group.

Non-goals (do not add in Stage 0):
  - No proactive speech path.
  - No relevance prediction.
  - No reminder / follow-up scheduler.
  - No ProactiveGovernor / AddresseeGate consumer.
  - No new hemisphere focus.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

JARVIS_DIR = Path.home() / ".jarvis"
REGISTRY_PATH = JARVIS_DIR / "intention_registry.json"
OUTCOMES_PATH = JARVIS_DIR / "intention_outcomes.jsonl"

REGISTRY_SCHEMA_VERSION = 1

_MAX_OPEN = 500
_MAX_RESOLVED_IN_MEMORY = 200
_OUTCOMES_MAX_FILE_MB = 10

DEFAULT_STALE_AFTER_S = 24 * 3600.0

IntentionOutcome = Literal["open", "resolved", "failed", "stale", "abandoned"]


# ---------------------------------------------------------------------------
# Stage 1 graduation gates
#
# These are the registry-knowable gates used by `get_graduation_status()`.
# The complete Stage 1 activation criteria live in
# `docs/INTENTION_STAGE_1_DESIGN.md`. Some criteria (PVL coverage, test-green
# weeks) are provided by external systems and are NOT evaluated here — this
# module only reports what the registry itself can honestly verify.
#
# IMPORTANT: `get_graduation_status()` is observability only. It never gates
# any runtime behavior in Stage 0. Stage 1 must be enabled by a human operator
# after reviewing the full gate panel on the dashboard.
# ---------------------------------------------------------------------------

STAGE1_MIN_BACKED_COMMITMENTS = 30
STAGE1_MIN_RESOLVED_7D = 5
STAGE1_MAX_ERRORS = 5
STAGE1_MIN_OUTCOME_VARIANCE_CLASSES = 2


# ---------------------------------------------------------------------------
# Atomic-write helper (duplicated locally to avoid memory.persistence import
# cycle, since memory.persistence will call into this module at boot).
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path | str, data: Any, **json_kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **json_kwargs)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _short_id() -> str:
    return "int_" + uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# IntentionRecord
# ---------------------------------------------------------------------------


@dataclass
class IntentionRecord:
    """A single declared intention bound to a real backing job."""

    id: str
    turn_id: str
    speaker_id: str
    utterance: str
    commitment_phrase: str
    commitment_type: str
    backing_job_id: str
    backing_job_kind: str
    created_at: float
    expected_duration_s: float = 0.0
    outcome: str = "open"
    resolved_at: float = 0.0
    resolution_reason: str = ""
    provenance: str = "regex_bootstrap"
    conversation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_s(self, now: float | None = None) -> float:
        return (now or time.time()) - self.created_at

    def is_open(self) -> bool:
        return self.outcome == "open"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IntentionRecord":
        return cls(
            id=d.get("id", ""),
            turn_id=d.get("turn_id", ""),
            speaker_id=d.get("speaker_id", ""),
            utterance=d.get("utterance", ""),
            commitment_phrase=d.get("commitment_phrase", ""),
            commitment_type=d.get("commitment_type", "generic"),
            backing_job_id=d.get("backing_job_id", ""),
            backing_job_kind=d.get("backing_job_kind", "unknown"),
            created_at=float(d.get("created_at", 0.0)),
            expected_duration_s=float(d.get("expected_duration_s", 0.0)),
            outcome=d.get("outcome", "open"),
            resolved_at=float(d.get("resolved_at", 0.0)),
            resolution_reason=d.get("resolution_reason", ""),
            provenance=d.get("provenance", "regex_bootstrap"),
            conversation_id=d.get("conversation_id", ""),
            metadata=dict(d.get("metadata", {}) or {}),
        )


# ---------------------------------------------------------------------------
# IntentionRegistry singleton
# ---------------------------------------------------------------------------


class IntentionRegistry:
    """In-memory registry of open + recently-resolved intentions.

    Writes:
      - Current state snapshot: `~/.jarvis/intention_registry.json` (atomic).
      - Append-only outcome log: `~/.jarvis/intention_outcomes.jsonl`.

    Reads:
      - `get_open()`, `get_by_backing_job()`, `get_status()`, `get_open_count()`.

    Lifecycle:
      - register() adds an open record.
      - resolve() closes with outcome in {resolved, failed}.
      - abandon() closes with outcome=abandoned (explicit drop, honest).
      - stale_sweep() closes open records older than max_age with outcome=stale.
    """

    _instance: "IntentionRegistry | None" = None

    @classmethod
    def get_instance(cls) -> "IntentionRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: "OrderedDict[str, IntentionRecord]" = OrderedDict()
        self._resolved: "OrderedDict[str, IntentionRecord]" = OrderedDict()
        self._by_backing: dict[str, str] = {}
        self._total_registered = 0
        self._total_resolved = 0
        self._total_stale = 0
        self._total_abandoned = 0
        self._total_failed = 0
        self._errors = 0
        self._last_error = ""
        self._loaded = False
        JARVIS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        utterance: str,
        commitment_phrase: str,
        commitment_type: str,
        backing_job_id: str,
        backing_job_kind: str,
        turn_id: str = "",
        speaker_id: str = "",
        conversation_id: str = "",
        expected_duration_s: float = 0.0,
        provenance: str = "regex_bootstrap",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Register a new intention bound to a real backing job.

        Returns the intention id on success, or None if backing_job_id is
        empty (strict invariant: every registered intention MUST have a
        real backing job id).
        """
        if not backing_job_id:
            self._errors += 1
            self._last_error = "register called with empty backing_job_id"
            logger.warning("IntentionRegistry.register rejected: empty backing_job_id")
            return None

        rec = IntentionRecord(
            id=_short_id(),
            turn_id=turn_id or _short_id(),
            speaker_id=speaker_id or "",
            utterance=(utterance or "")[:500],
            commitment_phrase=(commitment_phrase or "")[:200],
            commitment_type=commitment_type or "generic",
            backing_job_id=backing_job_id,
            backing_job_kind=backing_job_kind or "unknown",
            created_at=time.time(),
            expected_duration_s=max(0.0, float(expected_duration_s)),
            outcome="open",
            provenance=provenance,
            conversation_id=conversation_id,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._open[rec.id] = rec
            if len(self._open) > _MAX_OPEN:
                oldest_id, oldest_rec = self._open.popitem(last=False)
                oldest_rec.outcome = "abandoned"
                oldest_rec.resolved_at = time.time()
                oldest_rec.resolution_reason = "evicted_by_max_open_cap"
                self._resolved[oldest_id] = oldest_rec
                self._total_abandoned += 1
                self._trim_resolved_locked()
                self._append_outcome_locked(oldest_rec)
            self._by_backing[backing_job_id] = rec.id
            self._total_registered += 1

        logger.debug(
            "Intention registered: id=%s kind=%s phrase=%r backing=%s",
            rec.id, rec.backing_job_kind, rec.commitment_phrase, rec.backing_job_id,
        )
        return rec.id

    def resolve(
        self,
        *,
        backing_job_id: str = "",
        intention_id: str = "",
        outcome: str = "resolved",
        reason: str = "",
        result_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve an open intention. Outcome should be 'resolved' or 'failed'.

        Looks up by backing_job_id first, falls back to intention_id.
        Returns True if a matching open intention was found and closed.
        """
        if outcome not in ("resolved", "failed"):
            logger.warning("resolve() with unsupported outcome %r", outcome)
            return False

        now = time.time()
        rec: IntentionRecord | None = None
        with self._lock:
            if backing_job_id and backing_job_id in self._by_backing:
                rid = self._by_backing.pop(backing_job_id)
                rec = self._open.pop(rid, None)
            if rec is None and intention_id and intention_id in self._open:
                rec = self._open.pop(intention_id, None)
                if rec and rec.backing_job_id in self._by_backing:
                    self._by_backing.pop(rec.backing_job_id, None)

            if rec is None:
                return False

            rec.outcome = outcome
            rec.resolved_at = now
            rec.resolution_reason = (reason or "")[:200]
            if result_summary:
                rec.metadata["result_summary"] = result_summary[:500]
            if metadata:
                for k, v in metadata.items():
                    rec.metadata.setdefault(k, v)

            self._resolved[rec.id] = rec
            self._trim_resolved_locked()
            if outcome == "resolved":
                self._total_resolved += 1
            else:
                self._total_failed += 1

            self._append_outcome_locked(rec)

        logger.debug(
            "Intention resolved: id=%s outcome=%s reason=%r", rec.id, rec.outcome, rec.resolution_reason
        )
        return True

    def abandon(
        self,
        *,
        intention_id: str = "",
        backing_job_id: str = "",
        reason: str = "explicit_abandon",
    ) -> bool:
        """Explicitly mark an open intention as abandoned. Honest failure path."""
        now = time.time()
        rec: IntentionRecord | None = None
        with self._lock:
            if backing_job_id and backing_job_id in self._by_backing:
                rid = self._by_backing.pop(backing_job_id)
                rec = self._open.pop(rid, None)
            if rec is None and intention_id and intention_id in self._open:
                rec = self._open.pop(intention_id, None)
                if rec and rec.backing_job_id in self._by_backing:
                    self._by_backing.pop(rec.backing_job_id, None)
            if rec is None:
                return False
            rec.outcome = "abandoned"
            rec.resolved_at = now
            rec.resolution_reason = (reason or "explicit_abandon")[:200]
            self._resolved[rec.id] = rec
            self._trim_resolved_locked()
            self._total_abandoned += 1
            self._append_outcome_locked(rec)
        return True

    def stale_sweep(self, max_age_s: float = DEFAULT_STALE_AFTER_S) -> int:
        """Mark open intentions older than max_age_s as stale.

        Returns the number of intentions marked stale. Called from the
        consciousness kernel's background cycle at 300s cadence. Never
        emits user-facing events.
        """
        if max_age_s <= 0:
            return 0
        now = time.time()
        staled: list[IntentionRecord] = []
        with self._lock:
            for rid, rec in list(self._open.items()):
                if (now - rec.created_at) >= max_age_s:
                    self._open.pop(rid, None)
                    if rec.backing_job_id in self._by_backing:
                        self._by_backing.pop(rec.backing_job_id, None)
                    rec.outcome = "stale"
                    rec.resolved_at = now
                    rec.resolution_reason = f"stale_after_{int(max_age_s)}s"
                    self._resolved[rec.id] = rec
                    self._total_stale += 1
                    staled.append(rec)
            self._trim_resolved_locked()
            for rec in staled:
                self._append_outcome_locked(rec)
        if staled:
            logger.info("IntentionRegistry.stale_sweep marked %d as stale", len(staled))
        return len(staled)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_open(self) -> list[IntentionRecord]:
        with self._lock:
            return list(self._open.values())

    def get_open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def get_recent_resolved(self, n: int = 50) -> list[IntentionRecord]:
        with self._lock:
            recs = list(self._resolved.values())
        return recs[-n:][::-1]

    def get_recent_resolved_for_resolver(self, n: int = 50) -> list[IntentionRecord]:
        """Stage 1 read-only hook: return the last N resolved intentions.

        Surfaces resolved/failed records for the IntentionResolver to score.
        Identical semantics to get_recent_resolved() but exists as a separate
        method so the call-graph makes the Stage 1 dependency explicit.
        """
        return self.get_recent_resolved(n=n)

    def attach_resolver_verdict(
        self, intention_id: str, verdict_payload: dict[str, Any]
    ) -> bool:
        """Stage 1 supervision hook: attach a resolver verdict to a record.

        Write-once shadow metadata — does not change the intention outcome.
        Returns True if the record was found and the verdict was attached.
        """
        with self._lock:
            rec = self._resolved.get(intention_id)
            if rec is None:
                rec = self._open.get(intention_id)
            if rec is None:
                return False
            if "resolver_verdict" in rec.metadata:
                return False
            rec.metadata["resolver_verdict"] = verdict_payload
        return True

    def get_by_backing_job(self, backing_job_id: str) -> IntentionRecord | None:
        with self._lock:
            rid = self._by_backing.get(backing_job_id)
            if rid and rid in self._open:
                return self._open[rid]
            for rec in self._resolved.values():
                if rec.backing_job_id == backing_job_id:
                    return rec
        return None

    def get_by_id(self, intention_id: str) -> IntentionRecord | None:
        with self._lock:
            if intention_id in self._open:
                return self._open[intention_id]
            return self._resolved.get(intention_id)

    def get_status(self) -> dict[str, Any]:
        """Summary view for self_status MeaningFrame + dashboard."""
        now = time.time()
        with self._lock:
            open_recs = list(self._open.values())
            most_recent_age = 0.0
            oldest_age = 0.0
            if open_recs:
                oldest_age = max((now - r.created_at) for r in open_recs)
                most_recent_age = min((now - r.created_at) for r in open_recs)
            outcome_histogram_7d = self._outcome_histogram_locked(days=7)
            return {
                "open_count": len(self._open),
                "resolved_buffer_count": len(self._resolved),
                "most_recent_open_intention_age_s": round(most_recent_age, 2),
                "oldest_open_intention_age_s": round(oldest_age, 2),
                "total_registered": self._total_registered,
                "total_resolved": self._total_resolved,
                "total_failed": self._total_failed,
                "total_stale": self._total_stale,
                "total_abandoned": self._total_abandoned,
                "outcome_histogram_7d": outcome_histogram_7d,
                "errors": self._errors,
                "last_error": self._last_error,
                "loaded": self._loaded,
            }

    def get_graduation_status(self) -> dict[str, Any]:
        """Report Stage-0 → Stage-1 graduation readiness (observability only).

        Returns a structured gate check that the dashboard, PVL, and any
        future operator workflow can read. Each gate is one of:
          - ``"pass"``        : criterion met
          - ``"pending"``     : criterion not yet met (normal for fresh brain)
          - ``"unknown"``     : criterion cannot be evaluated from the registry
                                alone (requires external verification, e.g.
                                PVL coverage or regression-suite green-weeks)

        This method NEVER flips any runtime flag. Stage 1 activation is a
        human operator decision gated on the full design-doc checklist, not
        on the ``stage1_ready`` boolean returned here.

        See ``docs/INTENTION_STAGE_1_DESIGN.md`` for the complete criteria.
        """
        with self._lock:
            total_registered = self._total_registered
            errors = self._errors
            hist_7d = self._outcome_histogram_locked(days=7)

        resolved_7d = int(hist_7d.get("resolved", 0))
        non_zero_classes = sum(1 for v in hist_7d.values() if v > 0)

        def _status(ok: bool) -> str:
            return "pass" if ok else "pending"

        gates: list[dict[str, Any]] = [
            {
                "id": "backed_commitments_logged",
                "label": "Backed commitments logged",
                "status": _status(total_registered >= STAGE1_MIN_BACKED_COMMITMENTS),
                "observed": total_registered,
                "required": STAGE1_MIN_BACKED_COMMITMENTS,
                "source": "registry",
                "description": (
                    f"register() accepted >= {STAGE1_MIN_BACKED_COMMITMENTS} "
                    "intentions bound to real backing job ids"
                ),
            },
            {
                "id": "resolution_histogram_variance",
                "label": "Outcome histogram has variance",
                "status": _status(non_zero_classes >= STAGE1_MIN_OUTCOME_VARIANCE_CLASSES),
                "observed": non_zero_classes,
                "required": STAGE1_MIN_OUTCOME_VARIANCE_CLASSES,
                "source": "registry",
                "description": (
                    ">= 2 non-zero outcome classes in the 7-day histogram "
                    "(so the Stage-1 resolver sees varied supervision)"
                ),
            },
            {
                "id": "recent_resolutions",
                "label": "Recent resolutions produced",
                "status": _status(resolved_7d >= STAGE1_MIN_RESOLVED_7D),
                "observed": resolved_7d,
                "required": STAGE1_MIN_RESOLVED_7D,
                "source": "registry",
                "description": (
                    f"At least {STAGE1_MIN_RESOLVED_7D} intentions resolved "
                    "cleanly in the last 7 days (evidence the autonomy + "
                    "library-ingest outcome hooks are firing end-to-end)"
                ),
            },
            {
                "id": "persistence_error_floor",
                "label": "Persistence error floor",
                "status": _status(errors <= STAGE1_MAX_ERRORS),
                "observed": errors,
                "required": STAGE1_MAX_ERRORS,
                "source": "registry",
                "description": (
                    "Registry write/load errors stay at or below the bounded "
                    "envelope (save, load, and outcome-append must be stable)"
                ),
            },
            {
                "id": "pvl_intention_truth_coverage",
                "label": "PVL intention_truth coverage",
                "status": "unknown",
                "observed": None,
                "required": 1.0,
                "source": "jarvis_eval",
                "description": (
                    "100% coverage of the intention_truth PVL group. "
                    "Evaluated by the eval sidecar; see /api/eval or the "
                    "Process Verification Layer panel."
                ),
            },
            {
                "id": "regression_suite_green_weeks",
                "label": "Regression suite green (2 consecutive weeks)",
                "status": "unknown",
                "observed": None,
                "required": 2,
                "source": "external",
                "description": (
                    "test_intention_registry, test_commitment_extractor, and "
                    "test_capability_gate commitment regressions must stay "
                    "green for 2 consecutive weeks before Stage 1. Operator "
                    "verification required; not evaluated at runtime."
                ),
            },
        ]

        registry_knowable = [g for g in gates if g["source"] == "registry"]
        all_registry_pass = all(g["status"] == "pass" for g in registry_knowable)

        return {
            "stage": 0,
            "next_stage": 1,
            "gates": gates,
            "registry_gates_passed": all_registry_pass,
            "stage1_ready": False,
            "stage1_readiness_note": (
                "Stage 1 activation is a human operator decision. The "
                "registry-knowable gates are observability only; "
                "pvl_intention_truth_coverage and "
                "regression_suite_green_weeks must be confirmed externally."
            ),
            "evaluated_at": time.time(),
            "design_doc": "docs/INTENTION_STAGE_1_DESIGN.md",
        }

    def _outcome_histogram_locked(self, days: int = 7) -> dict[str, int]:
        cutoff = time.time() - (days * 24 * 3600.0)
        hist: dict[str, int] = {"resolved": 0, "failed": 0, "stale": 0, "abandoned": 0}
        for rec in self._resolved.values():
            if rec.resolved_at < cutoff:
                continue
            hist[rec.outcome] = hist.get(rec.outcome, 0) + 1
        return hist

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> bool:
        with self._lock:
            payload = {
                "_provenance": {
                    "schema_version": REGISTRY_SCHEMA_VERSION,
                    "saved_at": time.time(),
                },
                "open": [rec.to_dict() for rec in self._open.values()],
                "resolved": [rec.to_dict() for rec in self._resolved.values()],
                "counters": {
                    "total_registered": self._total_registered,
                    "total_resolved": self._total_resolved,
                    "total_failed": self._total_failed,
                    "total_stale": self._total_stale,
                    "total_abandoned": self._total_abandoned,
                },
            }
        try:
            _atomic_write_json(REGISTRY_PATH, payload, indent=2, default=str)
            return True
        except Exception as exc:
            self._errors += 1
            self._last_error = f"save: {type(exc).__name__}: {exc}"
            logger.exception("IntentionRegistry.save failed")
            return False

    def load(self) -> int:
        """Load persisted open + recently-resolved intentions on boot.

        Returns the number of open intentions restored.
        """
        if self._loaded:
            return 0
        self._loaded = True
        if not REGISTRY_PATH.exists():
            logger.info("No persisted intention registry at %s", REGISTRY_PATH)
            return 0
        try:
            raw = json.loads(REGISTRY_PATH.read_text())
        except Exception as exc:
            self._errors += 1
            self._last_error = f"load: {type(exc).__name__}: {exc}"
            logger.exception("Failed to load intention registry from %s", REGISTRY_PATH)
            return 0

        open_rows = raw.get("open", []) if isinstance(raw, dict) else []
        resolved_rows = raw.get("resolved", []) if isinstance(raw, dict) else []
        counters = raw.get("counters", {}) if isinstance(raw, dict) else {}

        with self._lock:
            for row in open_rows:
                try:
                    rec = IntentionRecord.from_dict(row)
                    if rec.id and rec.outcome == "open":
                        self._open[rec.id] = rec
                        if rec.backing_job_id:
                            self._by_backing[rec.backing_job_id] = rec.id
                except Exception:
                    continue
            for row in resolved_rows[-_MAX_RESOLVED_IN_MEMORY:]:
                try:
                    rec = IntentionRecord.from_dict(row)
                    if rec.id and rec.outcome != "open":
                        self._resolved[rec.id] = rec
                except Exception:
                    continue

            self._total_registered = int(counters.get("total_registered", len(self._open) + len(self._resolved)))
            self._total_resolved = int(counters.get("total_resolved", 0))
            self._total_failed = int(counters.get("total_failed", 0))
            self._total_stale = int(counters.get("total_stale", 0))
            self._total_abandoned = int(counters.get("total_abandoned", 0))

        logger.info(
            "IntentionRegistry loaded: %d open, %d resolved buffered",
            len(self._open), len(self._resolved),
        )
        return len(self._open)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_resolved_locked(self) -> None:
        while len(self._resolved) > _MAX_RESOLVED_IN_MEMORY:
            self._resolved.popitem(last=False)

    def _append_outcome_locked(self, rec: IntentionRecord) -> None:
        """Append an outcome record to intention_outcomes.jsonl (append-only)."""
        try:
            self._maybe_rotate_outcomes_locked()
            line = {
                "type": "intention_outcome",
                "id": rec.id,
                "turn_id": rec.turn_id,
                "speaker_id": rec.speaker_id,
                "commitment_phrase": rec.commitment_phrase,
                "commitment_type": rec.commitment_type,
                "backing_job_id": rec.backing_job_id,
                "backing_job_kind": rec.backing_job_kind,
                "created_at": rec.created_at,
                "resolved_at": rec.resolved_at,
                "outcome": rec.outcome,
                "resolution_reason": rec.resolution_reason,
                "latency_s": round(max(0.0, rec.resolved_at - rec.created_at), 3),
                "provenance": rec.provenance,
                "metadata": rec.metadata,
            }
            with open(OUTCOMES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, separators=(",", ":"), default=str) + "\n")
        except Exception as exc:
            self._errors += 1
            self._last_error = f"append_outcome: {type(exc).__name__}: {exc}"
            if self._errors <= 3:
                logger.warning("IntentionRegistry outcome append failed: %s", exc)

    def _maybe_rotate_outcomes_locked(self) -> None:
        try:
            if not OUTCOMES_PATH.exists():
                return
            size = OUTCOMES_PATH.stat().st_size
            if size < _OUTCOMES_MAX_FILE_MB * 1024 * 1024:
                return
            with open(OUTCOMES_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            half = len(lines) // 2
            with open(OUTCOMES_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[half:])
            logger.info(
                "intention_outcomes.jsonl rotated: kept %d of %d lines",
                len(lines) - half, len(lines),
            )
        except Exception as exc:
            if self._errors <= 3:
                logger.warning("intention_outcomes rotation failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton for easy import
# ---------------------------------------------------------------------------

intention_registry = IntentionRegistry.get_instance()


__all__ = [
    "IntentionRecord",
    "IntentionRegistry",
    "intention_registry",
    "REGISTRY_PATH",
    "OUTCOMES_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "DEFAULT_STALE_AFTER_S",
    "STAGE1_MIN_BACKED_COMMITMENTS",
    "STAGE1_MIN_RESOLVED_7D",
    "STAGE1_MAX_ERRORS",
    "STAGE1_MIN_OUTCOME_VARIANCE_CLASSES",
]
