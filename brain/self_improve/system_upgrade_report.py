"""Canonical ``system_upgrades`` truth lane for self-improvement.

Structured JSON reports under ``~/.jarvis/system_upgrades/reports/`` are the
source of truth for upgrade identity, evidence summaries, and verdicts.
Bounded ``self_improvement`` memories are allowed only when a matching report
file exists for the same ``upgrade_id``.

Training samples for the analytics-only ``SYSTEM_UPGRADES`` hemisphere are
appended only when :func:`is_complete_for_training` passes (anti-corruption).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from memory.persistence import atomic_write_json

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("~/.jarvis/system_upgrades/reports").expanduser()
TRAINING_SAMPLES_PATH = Path("~/.jarvis/system_upgrades/training_samples.jsonl").expanduser()

UpgradeVerdict = Literal[
    "none",
    "pending_verification",
    "verified_stable",
    "verified_improved",
    "verified_regressed",
    "rolled_back",
]


def mint_upgrade_id() -> str:
    return f"upg_{uuid.uuid4().hex[:12]}"


def mint_attempt_id() -> str:
    return f"att_{uuid.uuid4().hex[:10]}"


def report_path(upgrade_id: str) -> Path:
    return REPORTS_DIR / f"{upgrade_id}.json"


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class SystemUpgradeReport:
    upgrade_id: str
    request_id: str
    conversation_id: str = ""
    plan_id: str = ""
    description_short: str = ""
    target_module: str = ""
    request_type: str = ""
    orchestrator_status: str = "pending"
    verdict: UpgradeVerdict = "none"
    dry_run: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)
    snapshot_path: str = ""
    files_changed: list[str] = field(default_factory=list)
    sandbox_summary: dict[str, Any] = field(default_factory=dict)
    verification_verdict: str = ""
    verification_reason: str = ""
    related_upgrade_id: str = ""
    caused_by_upgrade_id: str = ""
    quarantine_pressure: float = 0.0
    soul_integrity: float = 0.0
    patch_provider: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SystemUpgradeReport:
        return SystemUpgradeReport(
            upgrade_id=d["upgrade_id"],
            request_id=d.get("request_id", ""),
            conversation_id=d.get("conversation_id", ""),
            plan_id=d.get("plan_id", ""),
            description_short=d.get("description_short", ""),
            target_module=d.get("target_module", ""),
            request_type=d.get("request_type", ""),
            orchestrator_status=d.get("orchestrator_status", "pending"),
            verdict=d.get("verdict", "none"),  # type: ignore[arg-type]
            dry_run=bool(d.get("dry_run", False)),
            attempts=list(d.get("attempts", [])),
            snapshot_path=d.get("snapshot_path", ""),
            files_changed=list(d.get("files_changed", [])),
            sandbox_summary=dict(d.get("sandbox_summary", {})),
            verification_verdict=d.get("verification_verdict", ""),
            verification_reason=d.get("verification_reason", ""),
            related_upgrade_id=d.get("related_upgrade_id", ""),
            caused_by_upgrade_id=d.get("caused_by_upgrade_id", ""),
            quarantine_pressure=float(d.get("quarantine_pressure", 0.0)),
            soul_integrity=float(d.get("soul_integrity", 0.0)),
            patch_provider=d.get("patch_provider", ""),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
        )


def load_report(upgrade_id: str) -> SystemUpgradeReport | None:
    path = report_path(upgrade_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SystemUpgradeReport.from_dict(data)
    except Exception as exc:
        logger.warning("Corrupt upgrade report %s: %s", upgrade_id, exc)
        return None


def save_report(report: SystemUpgradeReport) -> None:
    ensure_dirs()
    report.updated_at = time.time()
    # Round-trip through JSON with a string fallback so tests using MagicMock
    # stubs (or any non-JSON-native leaf) cannot break persistence.
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    atomic_write_json(report_path(report.upgrade_id), payload, indent=2)


def list_report_paths(limit: int = 200) -> list[Path]:
    ensure_dirs()
    paths = sorted(REPORTS_DIR.glob("upg_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[:limit]


def count_reports() -> int:
    ensure_dirs()
    return sum(1 for _ in REPORTS_DIR.glob("upg_*.json"))


def count_training_samples() -> int:
    if not TRAINING_SAMPLES_PATH.exists():
        return 0
    try:
        n = 0
        with TRAINING_SAMPLES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except OSError:
        return 0


def verdict_from_parts(
    orchestrator_status: str,
    verification_verdict: str = "",
) -> UpgradeVerdict:
    if orchestrator_status == "verifying":
        return "pending_verification"
    if orchestrator_status == "rolled_back":
        return "rolled_back"
    if orchestrator_status == "promoted":
        v = (verification_verdict or "").lower()
        if v == "improved":
            return "verified_improved"
        if v == "regressed":
            return "verified_regressed"
        return "verified_stable"
    return "none"


def can_write_bounded_memory(report: SystemUpgradeReport) -> bool:
    if not report.upgrade_id or not report.request_id:
        return False
    if not report_path(report.upgrade_id).exists():
        return False
    if not (report.description_short or report.target_module):
        return False
    return True


def is_complete_for_training(report: SystemUpgradeReport) -> bool:
    if not can_write_bounded_memory(report):
        return False
    ss = report.sandbox_summary
    if not ss.get("overall_passed"):
        return False
    if not ss.get("sim_executed"):
        return False
    if report.verdict in ("none", "pending_verification"):
        return False
    return True


def sandbox_summary_from_evaluation_report(report_obj: Any) -> dict[str, Any]:
    if report_obj is None:
        return {}
    return {
        "overall_passed": bool(getattr(report_obj, "overall_passed", False)),
        "lint_passed": bool(getattr(report_obj, "lint_passed", False)),
        "lint_executed": bool(getattr(report_obj, "lint_executed", False)),
        "tests_passed": bool(getattr(report_obj, "all_tests_passed", False)),
        "tests_executed": bool(getattr(report_obj, "tests_executed", False)),
        "sim_passed": bool(getattr(report_obj, "sim_passed", False)),
        "sim_executed": bool(getattr(report_obj, "sim_executed", False)),
        "sim_p95_before": float(getattr(report_obj, "sim_p95_before", 0.0)),
        "sim_p95_after": float(getattr(report_obj, "sim_p95_after", 0.0)),
        "recommendation": getattr(report_obj, "recommendation", ""),
    }


def sync_report_from_record(
    record: Any,
    *,
    dry_run: bool = False,
    verification_verdict: str = "",
    verification_reason: str = "",
    quarantine_pressure: float = 0.0,
    soul_integrity: float = 0.0,
) -> SystemUpgradeReport:
    """Upsert :class:`SystemUpgradeReport` from a live :class:`ImprovementRecord`."""
    uid = getattr(record, "upgrade_id", "") or mint_upgrade_id()
    setattr(record, "upgrade_id", uid)

    prev = load_report(uid)
    created = prev.created_at if prev else time.time()

    req = getattr(record, "request", None)
    request_id = getattr(req, "id", "") if req else ""
    desc = (getattr(req, "description", "") or "")[:240]
    target = getattr(req, "target_module", "") if req else ""
    rtype = getattr(req, "type", "") if req else ""
    plan = getattr(record, "plan", None)
    plan_id_raw = getattr(plan, "id", "") if plan else ""
    plan_id = plan_id_raw if isinstance(plan_id_raw, str) else ""
    patch = getattr(record, "patch", None)
    files: list[str] = []
    provider = ""
    if patch is not None:
        files = [getattr(fd, "path", str(fd)) for fd in getattr(patch, "files", [])][:24]
        provider = str(getattr(patch, "provider", "") or "")

    ev = getattr(record, "report", None)
    sandbox_summary = sandbox_summary_from_evaluation_report(ev)

    verdict = verdict_from_parts(
        getattr(record, "status", "pending"),
        verification_verdict,
    )

    rep = SystemUpgradeReport(
        upgrade_id=uid,
        request_id=request_id,
        conversation_id=getattr(record, "conversation_id", "") or "",
        plan_id=plan_id,
        description_short=desc,
        target_module=target,
        request_type=str(rtype),
        orchestrator_status=str(getattr(record, "status", "pending")),
        verdict=verdict,
        dry_run=dry_run or bool(getattr(record, "status", "") == "dry_run"),
        attempts=list(prev.attempts) if prev else [],
        snapshot_path=str(getattr(record, "snapshot_path", "") or ""),
        files_changed=files,
        sandbox_summary=sandbox_summary,
        verification_verdict=verification_verdict,
        verification_reason=(verification_reason or "")[:500],
        quarantine_pressure=quarantine_pressure,
        soul_integrity=soul_integrity,
        patch_provider=provider,
        created_at=created,
    )
    save_report(rep)
    return rep


def append_attempt_to_report(
    upgrade_id: str,
    attempt_id: str,
    iteration: int,
    sandbox_summary: dict[str, Any],
) -> None:
    rep = load_report(upgrade_id)
    if not rep:
        logger.warning("append_attempt: missing report %s", upgrade_id)
        return
    rep.attempts.append({
        "attempt_id": attempt_id,
        "iteration": iteration,
        "sandbox": sandbox_summary,
        "t": time.time(),
    })
    rep.sandbox_summary = sandbox_summary
    save_report(rep)


def maybe_append_training_sample(report: SystemUpgradeReport) -> None:
    if not is_complete_for_training(report):
        return
    ensure_dirs()
    ss = report.sandbox_summary
    features = {
        "target_module_hash": hash(report.target_module) % 1000 / 1000.0,
        "files_n": min(len(report.files_changed), 24) / 24.0,
        "lint_pass": 1.0 if ss.get("lint_passed") else 0.0,
        "tests_pass": 1.0 if ss.get("tests_passed") else 0.0,
        "sim_pass": 1.0 if ss.get("sim_passed") else 0.0,
        "sim_delta": max(
            -1.0,
            min(1.0, (ss.get("sim_p95_after", 0.0) - ss.get("sim_p95_before", 0.0)) / 50.0),
        ),
        "quarantine": max(0.0, min(1.0, report.quarantine_pressure)),
        "soul_integrity": max(0.0, min(1.0, report.soul_integrity)),
    }
    labels = {
        "verified_improved": 1.0 if report.verdict == "verified_improved" else 0.0,
        "verified_stable": 1.0 if report.verdict == "verified_stable" else 0.0,
        "verified_regressed": 1.0 if report.verdict == "verified_regressed" else 0.0,
        "rolled_back": 1.0 if report.verdict == "rolled_back" else 0.0,
    }
    row = {
        "t": time.time(),
        "upgrade_id": report.upgrade_id,
        "verdict": report.verdict,
        "features": features,
        "labels": labels,
    }
    try:
        with TRAINING_SAMPLES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        logger.warning("Failed to append upgrade training sample", exc_info=True)

    # Record CODE_QUALITY hemisphere teacher label
    try:
        from hemisphere.code_quality_encoder import CodeQualityEncoder
        from hemisphere.distillation import DistillationCollector
        label = CodeQualityEncoder.encode_verdict_label(report.verdict)
        collector = DistillationCollector.instance()
        collector.record(
            teacher="upgrade_verdict",
            signal_type="verdict",
            data=label,
            metadata={"upgrade_id": report.upgrade_id},
            origin="self_improve",
            fidelity=1.0,
        )
    except Exception:
        logger.debug("Failed to record code quality verdict label", exc_info=True)


def load_recent_training_samples(limit: int = 50) -> list[dict[str, Any]]:
    if not TRAINING_SAMPLES_PATH.exists():
        return []
    try:
        lines = TRAINING_SAMPLES_PATH.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def get_pvl_snapshot() -> dict[str, Any]:
    """Read-only metrics for PVL snapshot contracts (eval collector)."""
    return {
        "upgrade_reports_total": float(count_reports()),
        "upgrade_training_samples": float(count_training_samples()),
        "truth_lane_ready": 1.0 if REPORTS_DIR.is_dir() else 0.0,
    }


def recent_reports_for_introspection(limit: int = 8) -> list[dict[str, Any]]:
    """Lightweight summaries for introspection (no patch bodies)."""
    summaries: list[dict[str, Any]] = []
    for path in list_report_paths(limit):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({
            "upgrade_id": data.get("upgrade_id"),
            "request_id": data.get("request_id"),
            "status": data.get("orchestrator_status"),
            "verdict": data.get("verdict"),
            "description_short": (data.get("description_short") or "")[:120],
            "files_changed_n": len(data.get("files_changed") or []),
            "sandbox_passed": (data.get("sandbox_summary") or {}).get("overall_passed"),
            "updated_at": data.get("updated_at"),
        })
    return summaries
