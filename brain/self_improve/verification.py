"""Verification -- checkpoint-restart-verify loop for self-improvement patches.

After a patch passes sandbox + in-memory health checks, the orchestrator:
  1. Snapshots the original files (already exists)
  2. Captures baseline metrics
  3. Writes a pending_verification.json marker atomically
  4. Applies the patch to disk
  5. Gracefully saves all state
  6. Writes restart_intent.json and exits with code 10
  7. The supervisor relaunches the brain process

On next boot, main.py detects the pending file and:
  - If boot_count >= max_retries: auto-rollback and restart clean
  - Otherwise: increment boot_count, run normally for verification_period,
    then compare post-restart metrics to baselines.  Promote or rollback.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

PENDING_FILE = Path("~/.jarvis/pending_verification.json").expanduser()

DEFAULT_VERIFICATION_PERIOD_S = 180.0
DEFAULT_MIN_TICKS = 500
DEFAULT_MAX_RETRIES = 2
MAX_VERIFICATION_CEILING_S = 600.0  # absolute wall-clock cap

P95_TOLERANCE = 0.20         # 20% increase is "stable"
ERROR_TOLERANCE = 1          # up to 1 new error is "stable"
MEMORY_GROWTH_FACTOR = 2.0   # 2x baseline is "blowup"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class PendingVerification:
    patch_id: str
    description: str
    files_changed: list[str]
    snapshot_path: str
    conversation_id: str
    baselines: dict[str, float]
    target_metrics: list[str] = field(default_factory=list)
    applied_at: float = 0.0
    verification_period_s: float = DEFAULT_VERIFICATION_PERIOD_S
    min_ticks: int = DEFAULT_MIN_TICKS
    boot_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    python_argv: list[str] = field(default_factory=list)
    git_sha: str = ""
    host: str = ""
    pid: int = 0
    started_at: float = 0.0
    verification_reason: str = ""
    verification_verdict: str = ""
    upgrade_id: str = ""


# ---------------------------------------------------------------------------
# Write / Read / Clear
# ---------------------------------------------------------------------------


def write_pending(
    patch_id: str,
    description: str,
    files_changed: list[str],
    snapshot_path: str,
    conversation_id: str,
    baselines: dict[str, float],
    target_metrics: list[str] | None = None,
    upgrade_id: str = "",
) -> PendingVerification:
    """Atomically write the pending verification marker before applying a patch."""
    from memory.persistence import atomic_write_json

    pending = PendingVerification(
        patch_id=patch_id,
        description=description,
        files_changed=files_changed,
        snapshot_path=snapshot_path,
        conversation_id=conversation_id,
        baselines=baselines,
        target_metrics=target_metrics or [],
        applied_at=time.time(),
        python_argv=sys.argv[:],
        git_sha=_get_git_sha(),
        host=platform.node(),
        pid=os.getpid(),
        started_at=time.time(),
        upgrade_id=upgrade_id or "",
    )
    atomic_write_json(PENDING_FILE, _to_dict(pending), indent=2)
    logger.info("Wrote pending verification for patch %s", patch_id)
    return pending


def read_pending() -> PendingVerification | None:
    """Read and validate the pending verification marker.  Returns None if absent/corrupt."""
    if not PENDING_FILE.exists():
        return None
    try:
        data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        return PendingVerification(
            patch_id=data["patch_id"],
            description=data.get("description", ""),
            files_changed=data.get("files_changed", []),
            snapshot_path=data["snapshot_path"],
            conversation_id=data.get("conversation_id", ""),
            baselines=data.get("baselines", {}),
            target_metrics=data.get("target_metrics", []),
            applied_at=data.get("applied_at", 0.0),
            verification_period_s=data.get("verification_period_s", DEFAULT_VERIFICATION_PERIOD_S),
            min_ticks=data.get("min_ticks", DEFAULT_MIN_TICKS),
            boot_count=data.get("boot_count", 0),
            max_retries=data.get("max_retries", DEFAULT_MAX_RETRIES),
            python_argv=data.get("python_argv", []),
            git_sha=data.get("git_sha", ""),
            host=data.get("host", ""),
            pid=data.get("pid", 0),
            started_at=data.get("started_at", 0.0),
            verification_reason=data.get("verification_reason", ""),
            verification_verdict=data.get("verification_verdict", ""),
            upgrade_id=data.get("upgrade_id", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt pending verification file, removing: %s", exc)
        clear_pending()
        return None


def increment_boot_count() -> int:
    """Atomically bump boot_count in the pending file.  Returns the new count."""
    from memory.persistence import atomic_write_json

    if not PENDING_FILE.exists():
        return 0
    try:
        data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        data["boot_count"] = data.get("boot_count", 0) + 1
        atomic_write_json(PENDING_FILE, data, indent=2)
        logger.info("Pending verification boot_count incremented to %d", data["boot_count"])
        return data["boot_count"]
    except Exception as exc:
        logger.error("Failed to increment boot count: %s", exc)
        return -1


def clear_pending() -> None:
    """Delete the pending verification marker file."""
    try:
        PENDING_FILE.unlink(missing_ok=True)
        logger.info("Cleared pending verification marker")
    except OSError as exc:
        logger.warning("Failed to clear pending verification: %s", exc)


def has_pending() -> bool:
    """Check if a pending verification exists (used by orchestrator to refuse new patches)."""
    return PENDING_FILE.exists()


# ---------------------------------------------------------------------------
# Baseline capture
# ---------------------------------------------------------------------------


def capture_baselines(engine: Any) -> dict[str, float]:
    """Snapshot current system metrics for pre-restart comparison.

    Only call when the system has been running long enough to have meaningful
    data (the orchestrator ensures at least 100 kernel ticks have elapsed).
    """
    baselines: dict[str, float] = {}

    try:
        if engine._kernel:
            perf = engine._kernel.get_performance()
            baselines["tick_p95_ms"] = perf.p95_tick_ms
            baselines["tick_avg_ms"] = perf.avg_tick_ms
            baselines["tick_count"] = float(perf.tick_count)
    except Exception:
        logger.debug("Failed to capture kernel baselines", exc_info=True)

    try:
        cs = engine.consciousness
        state = cs.get_state()
        baselines["confidence_avg"] = getattr(state, "confidence_avg", 0.5)
    except Exception:
        logger.debug("Failed to capture consciousness baselines", exc_info=True)

    try:
        from consciousness.event_validator import event_validator
        stats = event_validator.get_stats()
        baselines["event_violations"] = float(stats.get("total_violations", 0))
    except Exception:
        logger.debug("Failed to capture event validator baselines", exc_info=True)

    try:
        from memory.storage import memory_storage
        baselines["memory_count"] = float(len(memory_storage.get_all()))
    except Exception:
        logger.debug("Failed to capture memory baselines", exc_info=True)

    try:
        from dashboard.app import _health
        snap = _health.snapshot()
        baselines["error_count"] = float(snap.get("error_count", 0))
        baselines["response_latency_ema"] = float(snap.get("avg_response_latency_ms", 0.0))
    except Exception:
        logger.debug("Failed to capture health baselines", exc_info=True)

    return baselines


def capture_current_metrics(engine: Any) -> dict[str, float]:
    """Capture the same metric set as baselines, for post-restart comparison."""
    return capture_baselines(engine)


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------


VerificationResult = Literal["improved", "stable", "regressed"]


def compare_metrics(
    baselines: dict[str, float],
    current: dict[str, float],
    target_metrics: list[str] | None = None,
) -> tuple[VerificationResult, dict[str, Any]]:
    """Compare post-restart metrics to pre-restart baselines.

    Returns (result, details) where details explains each check.
    """
    details: dict[str, Any] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    # tick p95: must not increase by more than P95_TOLERANCE
    base_p95 = baselines.get("tick_p95_ms", 0.0)
    curr_p95 = current.get("tick_p95_ms", 0.0)
    if base_p95 > 0 and curr_p95 > 0:
        p95_ratio = curr_p95 / base_p95
        details["tick_p95"] = {"baseline": base_p95, "current": curr_p95, "ratio": round(p95_ratio, 3)}
        if p95_ratio > 1.0 + P95_TOLERANCE:
            regressions.append(f"tick_p95 regressed: {base_p95:.2f}ms -> {curr_p95:.2f}ms")
        elif p95_ratio < 1.0 - P95_TOLERANCE:
            improvements.append("tick_p95")

    # error count: must not grow beyond tolerance
    base_errors = baselines.get("error_count", 0.0)
    curr_errors = current.get("error_count", 0.0)
    error_delta = curr_errors - base_errors
    details["error_count"] = {"baseline": base_errors, "current": curr_errors, "delta": error_delta}
    if error_delta > ERROR_TOLERANCE:
        regressions.append(f"error_count grew by {error_delta:.0f}")

    # memory count: must not blow up
    base_mem = baselines.get("memory_count", 0.0)
    curr_mem = current.get("memory_count", 0.0)
    if base_mem > 0:
        mem_ratio = curr_mem / base_mem
        details["memory_count"] = {"baseline": base_mem, "current": curr_mem, "ratio": round(mem_ratio, 3)}
        if mem_ratio > MEMORY_GROWTH_FACTOR:
            regressions.append(f"memory blowup: {base_mem:.0f} -> {curr_mem:.0f}")

    # event validation violations: should not spike
    base_violations = baselines.get("event_violations", 0.0)
    curr_violations = current.get("event_violations", 0.0)
    violation_delta = curr_violations - base_violations
    details["event_violations"] = {"baseline": base_violations, "current": curr_violations, "delta": violation_delta}
    if violation_delta > 5:
        regressions.append(f"event violations spiked by {violation_delta:.0f}")

    # target-specific metrics (if the patch claims to improve specific things)
    if target_metrics:
        for metric_key in target_metrics:
            base_val = baselines.get(metric_key)
            curr_val = current.get(metric_key)
            if base_val is not None and curr_val is not None:
                details[f"target_{metric_key}"] = {"baseline": base_val, "current": curr_val}
                if curr_val < base_val * 0.95:
                    improvements.append(metric_key)

    details["regressions"] = regressions
    details["improvements"] = improvements

    if regressions:
        reason = "regressed: " + "; ".join(regressions)
        details["reason"] = reason
        return "regressed", details
    if improvements:
        reason = "improved: " + ", ".join(improvements)
        details["reason"] = reason
        return "improved", details

    parts = []
    if "tick_p95" in details:
        d = details["tick_p95"]
        parts.append(f"tick_p95 {d['baseline']:.2f}→{d['current']:.2f}ms ({d['ratio']:.2f}x)")
    if "error_count" in details:
        d = details["error_count"]
        parts.append(f"errors Δ{d['delta']:.0f}")
    reason = "stable: " + (", ".join(parts) if parts else "all metrics within tolerance")
    details["reason"] = reason
    return "stable", details


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_verification_result(
    verdict: str,
    reason: str,
) -> None:
    """Stamp the pending file with the verification outcome before clearing.

    *verdict* is the terminal classification: improved | stable | regressed |
    stable_insufficient_data.  *reason* is the human-readable explanation.
    Both are persisted so that the improvements history can record them
    even after the pending file is deleted.
    """
    from memory.persistence import atomic_write_json

    if not PENDING_FILE.exists():
        return
    try:
        data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        data["verification_verdict"] = verdict
        data["verification_reason"] = reason
        atomic_write_json(PENDING_FILE, data, indent=2)
    except Exception as exc:
        logger.warning("Failed to write verification result: %s", exc)


def _to_dict(p: PendingVerification) -> dict[str, Any]:
    return {
        "patch_id": p.patch_id,
        "description": p.description,
        "files_changed": p.files_changed,
        "snapshot_path": p.snapshot_path,
        "conversation_id": p.conversation_id,
        "baselines": p.baselines,
        "target_metrics": p.target_metrics,
        "applied_at": p.applied_at,
        "verification_period_s": p.verification_period_s,
        "min_ticks": p.min_ticks,
        "boot_count": p.boot_count,
        "max_retries": p.max_retries,
        "python_argv": p.python_argv,
        "git_sha": p.git_sha,
        "host": p.host,
        "pid": p.pid,
        "started_at": p.started_at,
        "verification_reason": p.verification_reason,
        "verification_verdict": p.verification_verdict,
        "upgrade_id": p.upgrade_id,
    }


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""
