"""Dashboard-safe runner/status surface for the skill-acquisition weight room.

This module is intentionally authority-free. It can run synthetic exercises and
summarize reports, but it cannot verify skills, promote plugins, or mutate live
capability state.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from synthetic.skill_acquisition_exercise import PROFILES
from synthetic.skill_acquisition_exercise import REPORT_DIR
from synthetic.skill_acquisition_exercise import run_skill_acquisition_exercise
from synthetic.skill_acquisition_exercise import write_report


HEAVY_PROFILE_ENV = "ENABLE_SYNTHETIC_SKILL_ACQUISITION_HEAVY"
STARTUP_GRACE_S = 60.0


def _truth_boundary() -> dict[str, Any]:
    return {
        "authority": "telemetry_only",
        "synthetic_only": True,
        "live_influence": False,
        "promotion_eligible": False,
        "can_verify_skills": False,
        "can_promote_plugins": False,
        "can_unlock_capabilities": False,
    }


def _load_reports(limit: int = 10) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    try:
        if not REPORT_DIR.exists():
            return reports
        paths = sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[: max(1, limit)]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["path"] = str(path)
                    data["mtime"] = path.stat().st_mtime
                    reports.append(data)
            except Exception:
                continue
    except Exception:
        return []
    return reports


def _report_totals(reports: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    episodes = 0
    features = 0
    labels = 0
    for report in reports:
        episodes += int(report.get("episodes", 0) or 0)
        features += int(report.get("features_recorded", 0) or 0)
        labels += int(report.get("labels_recorded", 0) or 0)
        for key, val in (report.get("scenarios") or {}).items():
            scenarios[key] = scenarios.get(key, 0) + int(val or 0)
        for key, val in (report.get("outcomes") or {}).items():
            outcomes[key] = outcomes.get(key, 0) + int(val or 0)
    return {
        "synthetic_episodes_total": episodes,
        "synthetic_features_total": features,
        "synthetic_labels_total": labels,
        "scenario_totals": scenarios,
        "outcome_totals": outcomes,
    }


def _distillation_counts() -> dict[str, int]:
    try:
        from hemisphere.distillation import DistillationCollector

        collector = DistillationCollector.instance()
        if collector is None:
            return {"features": 0, "labels": 0, "quarantined": 0}
        stats = collector.get_stats()
        teachers = stats.get("teachers", {}) if isinstance(stats, dict) else {}
        feature_stats = teachers.get("skill_acquisition_features", {}) or {}
        label_stats = teachers.get("skill_acquisition_outcome", {}) or {}
        return {
            "features": int(feature_stats.get("total", 0) or 0),
            "labels": int(label_stats.get("total", 0) or 0),
            "quarantined": int(feature_stats.get("quarantined", 0) or 0)
            + int(label_stats.get("quarantined", 0) or 0),
        }
    except Exception:
        return {"features": 0, "labels": 0, "quarantined": 0}


def _profile_defaults() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "episode_count": profile.episode_count,
            "record_signals": profile.record_signals,
            "description": profile.description,
        }
        for name, profile in PROFILES.items()
    }


def evaluate_gates(
    profile_name: str,
    *,
    engine: Any = None,
    startup_ts: float | None = None,
) -> dict[str, Any]:
    """Return whether a synthetic profile is allowed to run right now."""
    reasons: list[str] = []
    profile = PROFILES.get(profile_name)
    if profile is None:
        return {
            "allowed": False,
            "blocked_reasons": ["unknown_profile"],
            "profile": profile_name,
            **_truth_boundary(),
        }

    # Smoke is explicitly no-record by default and safe as a boundary check.
    if profile_name == "smoke":
        return {
            "allowed": True,
            "blocked_reasons": [],
            "profile": profile_name,
            "record_signals": False,
            "gate_level": "invariant_check",
            **_truth_boundary(),
        }

    if engine is None:
        reasons.append("engine_not_ready")

    if startup_ts is not None and time.time() - float(startup_ts) < STARTUP_GRACE_S:
        reasons.append("startup_grace_active")

    acq = getattr(engine, "_acquisition_orchestrator", None) if engine else None
    if acq is None:
        reasons.append("acquisition_orchestrator_unavailable")
    else:
        try:
            acq_status = acq.get_status()
            if int(acq_status.get("active_count", 0) or 0) > 0:
                reasons.append("active_acquisition_in_progress")
            scheduler = acq_status.get("scheduler", {}) or {}
            if scheduler.get("pressure_level") == "high":
                reasons.append("quarantine_pressure_high")
        except Exception:
            reasons.append("acquisition_status_unavailable")

    codegen = getattr(engine, "_codegen_service", None) if engine else None
    if codegen is not None:
        try:
            cg_status = codegen.get_status()
            if cg_status.get("active_consumer"):
                reasons.append("codegen_busy")
        except Exception:
            reasons.append("codegen_status_unavailable")

    gestation = getattr(engine, "_gestation_manager", None) if engine else None
    if gestation is not None:
        try:
            g_status = gestation.get_status()
            if g_status.get("active") and not g_status.get("graduated"):
                reasons.append("gestation_active")
        except Exception:
            pass

    try:
        from hemisphere.distillation import DistillationCollector

        if DistillationCollector.instance() is None:
            reasons.append("distillation_collector_unavailable")
    except Exception:
        reasons.append("distillation_collector_unavailable")

    if profile_name in {"strict", "stress"}:
        heavy_enabled = str(os.environ.get(HEAVY_PROFILE_ENV, "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not heavy_enabled:
            reasons.append("heavy_profile_operator_flag_disabled")

    return {
        "allowed": not reasons,
        "blocked_reasons": reasons,
        "profile": profile_name,
        "record_signals": bool(profile.record_signals),
        "gate_level": "training" if profile_name == "coverage" else "heavy_training",
        **_truth_boundary(),
    }


class SkillAcquisitionWeightRoomRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._run: dict[str, Any] = {"status": "idle"}

    def status(self, *, engine: Any = None, startup_ts: float | None = None) -> dict[str, Any]:
        reports = _load_reports(limit=10)
        totals = _report_totals(reports)
        distill = _distillation_counts()
        with self._lock:
            run = dict(self._run)
        gates = {
            name: evaluate_gates(name, engine=engine, startup_ts=startup_ts)
            for name in PROFILES
        }
        latest = reports[0] if reports else {}
        return {
            "enabled": True,
            "run": run,
            "profiles": _profile_defaults(),
            "gates": gates,
            "reports": reports,
            "latest_report": latest,
            "last_profile": latest.get("profile_name", ""),
            "last_passed": latest.get("passed", False) if latest else False,
            "distillation": distill,
            **totals,
            **_truth_boundary(),
        }

    def start(
        self,
        profile_name: str,
        *,
        engine: Any = None,
        startup_ts: float | None = None,
        count: int | None = None,
        seed: int | None = None,
        no_record: bool = False,
    ) -> dict[str, Any]:
        gate = evaluate_gates(profile_name, engine=engine, startup_ts=startup_ts)
        if not gate["allowed"]:
            return {"started": False, **gate}
        profile = PROFILES[profile_name]
        if no_record and profile.record_signals:
            from dataclasses import replace

            profile = replace(profile, record_signals=False)
        with self._lock:
            if self._run.get("status") == "running":
                return {
                    "started": False,
                    "allowed": False,
                    "blocked_reasons": ["run_already_active"],
                    **_truth_boundary(),
                }
            before = _distillation_counts()
            self._run = {
                "status": "running",
                "profile": profile.name,
                "target_episodes": int(count or profile.episode_count),
                "episodes_done": 0,
                "started_at": time.time(),
                "ended_at": 0.0,
                "report_path": "",
                "error": "",
                "record_signals": bool(profile.record_signals),
                "feature_count_before": before["features"],
                "label_count_before": before["labels"],
                "feature_gain": 0,
                "label_gain": 0,
            }

        def _progress(_stats: Any, done: int, total: int) -> None:
            with self._lock:
                self._run["episodes_done"] = done
                self._run["target_episodes"] = total

        def _run() -> None:
            try:
                stats = run_skill_acquisition_exercise(
                    profile=profile,
                    count=count,
                    seed=seed,
                    progress_callback=_progress,
                )
                path = write_report(stats)
                after = _distillation_counts()
                with self._lock:
                    self._run.update({
                        "status": "completed" if stats.passed else "failed",
                        "episodes_done": stats.episodes,
                        "target_episodes": stats.episodes,
                        "ended_at": time.time(),
                        "report_path": str(path),
                        "passed": stats.passed,
                        "invariant_failures": list(stats.invariant_failures),
                        "feature_gain": max(0, after["features"] - int(self._run.get("feature_count_before", 0))),
                        "label_gain": max(0, after["labels"] - int(self._run.get("label_count_before", 0))),
                    })
            except Exception as exc:
                with self._lock:
                    self._run.update({
                        "status": "failed",
                        "ended_at": time.time(),
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    })

        thread = threading.Thread(target=_run, name=f"skill-acq-weight-room-{profile.name}", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        return {
            "started": True,
            **gate,
            "run": self.status(engine=engine, startup_ts=startup_ts).get("run", {}),
        }


weight_room_runner = SkillAcquisitionWeightRoomRunner()


def get_skill_acquisition_weight_room_status(
    *,
    engine: Any = None,
    startup_ts: float | None = None,
) -> dict[str, Any]:
    return weight_room_runner.status(engine=engine, startup_ts=startup_ts)


def start_skill_acquisition_weight_room_run(
    profile_name: str,
    *,
    engine: Any = None,
    startup_ts: float | None = None,
    count: int | None = None,
    seed: int | None = None,
    no_record: bool = False,
) -> dict[str, Any]:
    return weight_room_runner.start(
        profile_name,
        engine=engine,
        startup_ts=startup_ts,
        count=count,
        seed=seed,
        no_record=no_record,
    )

