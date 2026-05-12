"""Synthetic diagnostic encoder exercise.

Generates synthetic health/performance scenarios covering all 6 detector
categories and feeds them through DiagnosticEncoder.encode() (a @staticmethod
with zero side effects). Records training signals via DistillationCollector
at capped fidelity (0.7) with origin="synthetic".

Truth boundary:
  - DiagnosticEncoder.encode() and encode_label() are @staticmethod — pure
  - Records through real DistillationCollector (intentional — these ARE
    training signals, just at capped fidelity)
  - 0.7 fidelity ensures synthetic signals are weighted below real ones
  - origin="synthetic" provides provenance tracking
  - Does NOT read any live subsystem state — all inputs are fabricated
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hemisphere.diagnostic_encoder import (
    FEATURE_DIM,
    DiagnosticEncoder,
)

logger = logging.getLogger(__name__)

SYNTHETIC_FIDELITY = 0.7
SYNTHETIC_ORIGIN = "synthetic"
REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise"
LABEL_DIM = 6


# ---------------------------------------------------------------------------
# Scenario corpus — covers all 6 detector types + negative examples
# ---------------------------------------------------------------------------

def _base_snapshot() -> dict[str, Any]:
    """Healthy baseline snapshot (no detectors firing)."""
    return {
        "health": {"overall": 0.9, "worst_score": 0.8},
        "reasoning": {"overall": 0.9, "coherence": 0.85},
        "confidence": {"volatility": 0.05, "current": 0.7, "trend": 0.0},
        "latency": {"total": 100, "slow_gt_5s": 1, "avg_slow_ms": 6000.0},
        "event_bus": {"emitted": 5000, "errors": 2},
        "tick": {"p95_ms": 25.0},
    }


def _base_context() -> dict[str, Any]:
    """Healthy baseline context."""
    return {
        "uptime_s": 7200.0,
        "quarantine_pressure": 0.1,
        "soul_integrity": 0.85,
        "mode": "passive",
        "evolution_stage": 2,
        "consciousness_stage": 2,
        "health_trend_slope": 0.0,
        "mutations_last_hour": 2,
        "active_learning_jobs": 1,
        "improvements_today": 1,
        "last_improvement_age_s": 3600.0,
        "sandbox_pass_rate": 0.8,
        "friction_rate": 0.05,
        "correction_count": 1,
        "autonomy_level": 1,
        "target_module_lines": 200,
        "target_import_fanout": 5,
        "target_importers": 3,
        "target_symbol_count": 20,
        "target_recently_modified": False,
        "has_codebase_context": True,
        "friction_severity_high_ratio": 0.1,
        "friction_correction_ratio": 0.1,
        "friction_identity_count": 0,
        "correction_auto_accepted": 0,
        "has_friction_context": True,
    }


def _build_scenarios() -> list[dict[str, Any]]:
    """Build scenario corpus covering all 6 detector types + negatives."""
    scenarios: list[dict[str, Any]] = []

    # --- Negative examples (healthy, no detectors firing) ---
    for i in range(5):
        ctx = _base_context()
        ctx["uptime_s"] += i * 1800
        scenarios.append({
            "name": f"healthy_baseline_{i}",
            "detector_type": None,
            "snapshot": _base_snapshot(),
            "opportunities": [],
            "context": ctx,
        })

    # --- health_degraded ---
    for i in range(5):
        snap = _base_snapshot()
        snap["health"]["overall"] = 0.3 + random.uniform(0.0, 0.2)
        snap["health"]["worst_score"] = 0.2 + random.uniform(0.0, 0.15)
        opps = [{"type": "health_degraded", "priority": 4,
                 "sustained_count": 2 + i,
                 "evidence_detail": {"worst_component": "memory"}}]
        scenarios.append({
            "name": f"health_degraded_{i}",
            "detector_type": "health_degraded",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    # --- reasoning_decline ---
    for i in range(5):
        snap = _base_snapshot()
        snap["reasoning"]["overall"] = 0.4 + random.uniform(0.0, 0.1)
        snap["reasoning"]["coherence"] = 0.3 + random.uniform(0.0, 0.15)
        opps = [{"type": "reasoning_decline", "priority": 3,
                 "sustained_count": 2 + i,
                 "evidence_detail": {"depth": 0.5, "coherence": 0.35}}]
        scenarios.append({
            "name": f"reasoning_decline_{i}",
            "detector_type": "reasoning_decline",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    # --- confidence_volatile ---
    for i in range(5):
        snap = _base_snapshot()
        snap["confidence"]["volatility"] = 0.35 + random.uniform(0.0, 0.15)
        snap["confidence"]["trend"] = -0.3
        opps = [{"type": "confidence_volatile", "priority": 2,
                 "sustained_count": 1 + i,
                 "evidence_detail": {"volatility": 0.4}}]
        scenarios.append({
            "name": f"confidence_volatile_{i}",
            "detector_type": "confidence_volatile",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    # --- slow_responses ---
    for i in range(5):
        snap = _base_snapshot()
        snap["latency"]["slow_gt_5s"] = 25 + i * 5
        snap["latency"]["total"] = 100
        snap["latency"]["avg_slow_ms"] = 8000 + i * 1000
        opps = [{"type": "slow_responses", "priority": 3,
                 "sustained_count": 2 + i,
                 "evidence_detail": {"avg_slow_ms": 9000}}]
        scenarios.append({
            "name": f"slow_responses_{i}",
            "detector_type": "slow_responses",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    # --- event_bus_errors ---
    for i in range(5):
        snap = _base_snapshot()
        snap["event_bus"]["errors"] = 60 + i * 20
        snap["event_bus"]["emitted"] = 5000
        opps = [{"type": "event_bus_errors", "priority": 4,
                 "sustained_count": 2 + i,
                 "evidence_detail": {"error_rate": 0.015}}]
        scenarios.append({
            "name": f"event_bus_errors_{i}",
            "detector_type": "event_bus_errors",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    # --- tick_performance ---
    for i in range(5):
        snap = _base_snapshot()
        snap["tick"]["p95_ms"] = 55.0 + i * 10
        opps = [{"type": "tick_performance", "priority": 3,
                 "sustained_count": 2 + i,
                 "evidence_detail": {"p95_ms": 60.0}}]
        scenarios.append({
            "name": f"tick_performance_{i}",
            "detector_type": "tick_performance",
            "snapshot": snap,
            "opportunities": opps,
            "context": _base_context(),
        })

    return scenarios


def _randomize_scenario(base: dict[str, Any]) -> dict[str, Any]:
    """Apply random perturbations for stress testing."""
    import copy
    s = copy.deepcopy(base)

    snap = s["snapshot"]
    for block in ("health", "reasoning", "confidence"):
        for k, v in snap.get(block, {}).items():
            if isinstance(v, float):
                snap[block][k] = max(0.0, min(1.0, v + random.uniform(-0.1, 0.1)))

    ctx = s["context"]
    ctx["uptime_s"] += random.uniform(-1000, 3000)
    ctx["quarantine_pressure"] = max(0.0, min(1.0, ctx["quarantine_pressure"] + random.uniform(-0.1, 0.1)))
    ctx["friction_rate"] = max(0.0, min(1.0, ctx.get("friction_rate", 0.0) + random.uniform(-0.05, 0.05)))

    return s


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticExerciseProfile:
    name: str
    scenario_count: int
    randomize: bool
    record_signals: bool
    description: str = ""


PROFILES: dict[str, DiagnosticExerciseProfile] = {
    "smoke": DiagnosticExerciseProfile(
        name="smoke", scenario_count=20, randomize=False, record_signals=False,
        description="Quick check (20 scenarios, no distillation recording)",
    ),
    "coverage": DiagnosticExerciseProfile(
        name="coverage", scenario_count=40, randomize=True, record_signals=True,
        description="All 6 types x 5 + 10 negatives (40 scenarios)",
    ),
    "stress": DiagnosticExerciseProfile(
        name="stress", scenario_count=200, randomize=True, record_signals=True,
        description="Randomized high-volume (200 scenarios)",
    ),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticExerciseStats:
    scenarios_requested: int = 0
    scenarios_encoded: int = 0
    features_recorded: int = 0
    labels_recorded: int = 0
    detectors_exercised: Counter = field(default_factory=Counter)
    negative_examples: int = 0
    dim_check_passes: int = 0
    dim_check_failures: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.dim_check_failures > 0:
            reasons.append(f"dim_check_failures={self.dim_check_failures}")
        if self.scenarios_encoded == 0 and self.scenarios_requested > 0:
            reasons.append("zero_encodings")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "scenarios_requested": self.scenarios_requested,
            "scenarios_encoded": self.scenarios_encoded,
            "features_recorded": self.features_recorded,
            "labels_recorded": self.labels_recorded,
            "detectors_exercised": dict(self.detectors_exercised),
            "negative_examples": self.negative_examples,
            "dim_check_passes": self.dim_check_passes,
            "dim_check_failures": self.dim_check_failures,
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Diagnostic Exercise — {self.scenarios_encoded} encoded, "
            f"{self.features_recorded} features + {self.labels_recorded} labels "
            f"in {self.elapsed_s:.1f}s",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.detectors_exercised:
            lines.append("  Detectors: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.detectors_exercised.items())
            ))
        lines.append(f"  Negative examples: {self.negative_examples}")
        if self.fail_reasons:
            lines.append(f"  FAIL: {', '.join(self.fail_reasons)}")
        else:
            lines.append("  PASS: all checks hold")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_diagnostic_exercise(
    profile: DiagnosticExerciseProfile | None = None,
    count: int | None = None,
    collector: Any | None = None,
) -> DiagnosticExerciseStats:
    """Run a synchronous diagnostic encoder exercise.

    If collector is provided (a DistillationCollector), records signals.
    Otherwise, encodes only (for testing).
    """
    if profile is None:
        profile = PROFILES["coverage"]

    stats = DiagnosticExerciseStats(profile_name=profile.name)
    base_scenarios = _build_scenarios()
    n = count or profile.scenario_count
    stats.scenarios_requested = n

    scenarios_to_run: list[dict[str, Any]] = []
    while len(scenarios_to_run) < n:
        for sc in base_scenarios:
            if profile.randomize and len(scenarios_to_run) >= len(base_scenarios):
                scenarios_to_run.append(_randomize_scenario(sc))
            else:
                scenarios_to_run.append(sc)
            if len(scenarios_to_run) >= n:
                break

    for sc in scenarios_to_run:
        try:
            snapshot = sc["snapshot"]
            opportunities = sc["opportunities"]
            context = sc["context"]
            detector_type = sc.get("detector_type")

            scan_id = f"synthetic_{int(time.time() * 1000)}_{stats.scenarios_encoded}"

            features = DiagnosticEncoder.encode(snapshot, opportunities, context)

            if len(features) != FEATURE_DIM:
                stats.dim_check_failures += 1
            else:
                stats.dim_check_passes += 1

            stats.scenarios_encoded += 1

            if detector_type:
                stats.detectors_exercised[detector_type] += 1
                label, metadata = DiagnosticEncoder.encode_label(opportunities[0])
            else:
                stats.negative_examples += 1
                label, metadata = DiagnosticEncoder.encode_no_opportunity_label()

            if len(label) != LABEL_DIM:
                stats.dim_check_failures += 1

            if profile.record_signals and collector is not None:
                try:
                    collector.record(
                        teacher="diagnostic_features",
                        signal_type="synthetic_diagnostic",
                        data=features,
                        fidelity=SYNTHETIC_FIDELITY,
                        origin=SYNTHETIC_ORIGIN,
                        metadata={"scan_id": scan_id},
                    )
                    stats.features_recorded += 1
                except Exception:
                    pass

                try:
                    collector.record(
                        teacher="diagnostic_detector",
                        signal_type="synthetic_diagnostic",
                        data=label,
                        fidelity=SYNTHETIC_FIDELITY,
                        origin=SYNTHETIC_ORIGIN,
                        metadata={
                            "scan_id": scan_id,
                            **metadata,
                        },
                    )
                    stats.labels_recorded += 1
                except Exception:
                    pass

        except Exception as exc:
            stats.errors.append(f"{sc.get('name', '?')}: {type(exc).__name__}: {exc}")

    stats.end_time = time.time()
    logger.info(
        "Diagnostic exercise: %d scenarios, %d features + %d labels recorded",
        stats.scenarios_encoded, stats.features_recorded, stats.labels_recorded,
    )
    return stats
