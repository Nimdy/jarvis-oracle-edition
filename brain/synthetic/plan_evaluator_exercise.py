"""Synthetic plan evaluator exercise.

Generates synthetic acquisition plan structures at varying quality levels
and feeds them through PlanEvaluatorEncoder.encode() (a @staticmethod with
zero side effects). Records training signals via DistillationCollector at
capped fidelity (0.7) with origin="synthetic".

Truth boundary:
  - PlanEvaluatorEncoder.encode() is @staticmethod — pure
  - Mock objects are plain namespace objects, never real CapabilityAcquisitionJob
  - Records through real DistillationCollector at capped fidelity
  - NEVER writes to ~/.jarvis/acquisition_shadows/ (no shadow predictions)
  - NEVER calls AcquisitionOrchestrator methods
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from acquisition.plan_encoder import (
    FEATURE_DIM,
    PlanEvaluatorEncoder,
    encode_verdict,
)

logger = logging.getLogger(__name__)

SYNTHETIC_FIDELITY = 0.7
SYNTHETIC_ORIGIN = "synthetic"
REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise"
LABEL_DIM = 3


# ---------------------------------------------------------------------------
# Mock object builders
# ---------------------------------------------------------------------------

_OUTCOME_CLASSES = [
    "knowledge_only", "skill_creation", "plugin_creation",
    "core_upgrade", "specialist_nn", "hardware_integration", "mixed",
]


def _mock_job(
    outcome_class: str = "knowledge_only",
    risk_tier: int = 0,
    classification_confidence: float = 0.8,
    required_lanes: list[str] | None = None,
) -> SimpleNamespace:
    """Create a lightweight mock job with plan-relevant fields."""
    return SimpleNamespace(
        outcome_class=outcome_class,
        risk_tier=risk_tier,
        classification_confidence=classification_confidence,
        required_lanes=required_lanes or [],
    )


def _mock_plan(
    implementation_path: list[str] | None = None,
    verification_path: list[str] | None = None,
    rollback_path: list[str] | None = None,
    required_capabilities: list[str] | None = None,
    required_artifacts: list[str] | None = None,
    promotion_criteria: list[str] | None = None,
    version: int = 1,
    user_story: str = "",
    technical_approach: str = "",
    implementation_sketch: str = "",
    dependencies: list[str] | None = None,
    test_cases: list[str] | None = None,
    risk_analysis: str = "",
    doc_artifact_ids: list[str] | None = None,
) -> SimpleNamespace:
    """Create a lightweight mock plan with encoder-relevant fields."""
    return SimpleNamespace(
        implementation_path=implementation_path or [],
        verification_path=verification_path or [],
        rollback_path=rollback_path or [],
        required_capabilities=required_capabilities or [],
        required_artifacts=required_artifacts or [],
        promotion_criteria=promotion_criteria or [],
        version=version,
        user_story=user_story,
        technical_approach=technical_approach,
        implementation_sketch=implementation_sketch,
        dependencies=dependencies or [],
        test_cases=test_cases or [],
        risk_analysis=risk_analysis,
        doc_artifact_ids=doc_artifact_ids or [],
    )


# ---------------------------------------------------------------------------
# Plan template corpus — 3 quality levels
# ---------------------------------------------------------------------------

def _build_approved_plans() -> list[dict[str, Any]]:
    """High-quality plans that should be approved."""
    plans: list[dict[str, Any]] = []

    for i in range(5):
        oc = random.choice(["skill_creation", "plugin_creation", "knowledge_only"])
        job = _mock_job(
            outcome_class=oc,
            risk_tier=random.choice([0, 1]),
            classification_confidence=random.uniform(0.7, 0.95),
        )
        plan = _mock_plan(
            implementation_path=[f"step_{j}" for j in range(random.randint(3, 10))],
            verification_path=["lint_check", "test_run", "sandbox_eval"],
            rollback_path=["snapshot_restore", "revert_changes"],
            required_capabilities=["code_generation", "testing"],
            required_artifacts=["doc_1", "doc_2"],
            promotion_criteria=[
                "accuracy >= 0.80",
                "test pass rate > 95%",
                f"latency < {random.randint(200, 500)}ms",
            ],
            version=1,
            user_story="As a user, I want improved response quality",
            technical_approach="We will implement a new scoring algorithm "
                             "that weights recency and relevance adaptively",
            implementation_sketch="1. Add scoring module\n2. Wire into pipeline\n"
                                 "3. Add tests\n4. Run benchmark",
            dependencies=["code_generation", "testing"],
            test_cases=["test_accuracy", "test_regression", "test_edge_cases"],
            risk_analysis="Low risk. Rollback via snapshot. No external dependencies.",
            doc_artifact_ids=["doc_1"],
        )
        plans.append({
            "name": f"approved_{i}",
            "verdict": "approved",
            "job": job,
            "plan": plan,
        })

    return plans


def _build_needs_revision_plans() -> list[dict[str, Any]]:
    """Medium-quality plans that need revision."""
    plans: list[dict[str, Any]] = []

    for i in range(5):
        oc = random.choice(_OUTCOME_CLASSES)
        job = _mock_job(
            outcome_class=oc,
            risk_tier=random.choice([1, 2]),
            classification_confidence=random.uniform(0.5, 0.8),
        )
        plan = _mock_plan(
            implementation_path=[f"step_{j}" for j in range(random.randint(1, 4))],
            verification_path=["basic_check"],
            rollback_path=[],
            required_capabilities=[],
            promotion_criteria=["should work well"],
            version=random.randint(1, 3),
            user_story="Improve something",
            technical_approach="Use existing patterns",
            implementation_sketch="",
            dependencies=[],
            test_cases=[],
            risk_analysis="",
        )
        plans.append({
            "name": f"needs_revision_{i}",
            "verdict": "needs_revision",
            "job": job,
            "plan": plan,
        })

    return plans


def _build_rejected_plans() -> list[dict[str, Any]]:
    """Low-quality plans that should be rejected."""
    plans: list[dict[str, Any]] = []

    for i in range(5):
        oc = random.choice(["core_upgrade", "hardware_integration"])
        job = _mock_job(
            outcome_class=oc,
            risk_tier=random.choice([2, 3]),
            classification_confidence=random.uniform(0.3, 0.6),
            required_lanes=["deployment"],
        )
        plan = _mock_plan(
            implementation_path=[f"risky_step_{j}" for j in range(random.randint(10, 20))],
            verification_path=[],
            rollback_path=[],
            required_capabilities=["admin_access", "network_control"],
            promotion_criteria=[],
            version=1,
            user_story="",
            technical_approach="",
            implementation_sketch="",
            risk_analysis="",
        )
        plans.append({
            "name": f"rejected_{i}",
            "verdict": "rejected",
            "job": job,
            "plan": plan,
        })

    return plans


def _randomize_plan_item(base: dict[str, Any]) -> dict[str, Any]:
    """Apply random perturbations for stress testing."""
    import copy
    item = copy.deepcopy(base)

    job = item["job"]
    job.risk_tier = max(0, min(3, job.risk_tier + random.choice([-1, 0, 1])))
    job.classification_confidence = max(0.0, min(1.0,
        job.classification_confidence + random.uniform(-0.15, 0.15)))

    plan = item["plan"]
    n_steps = len(plan.implementation_path)
    if random.random() < 0.3:
        plan.implementation_path.append(f"extra_step_{n_steps}")
    if random.random() < 0.3 and plan.rollback_path:
        plan.rollback_path = []

    return item


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class PlanEvaluatorExerciseProfile:
    name: str
    plan_count: int
    randomize: bool
    record_signals: bool
    description: str = ""


PROFILES: dict[str, PlanEvaluatorExerciseProfile] = {
    "smoke": PlanEvaluatorExerciseProfile(
        name="smoke", plan_count=15, randomize=False, record_signals=False,
        description="Quick check (15 plans, no distillation recording)",
    ),
    "coverage": PlanEvaluatorExerciseProfile(
        name="coverage", plan_count=30, randomize=True, record_signals=True,
        description="3 classes x 10 plans (30 total)",
    ),
    "stress": PlanEvaluatorExerciseProfile(
        name="stress", plan_count=150, randomize=True, record_signals=True,
        description="Randomized high-volume (150 plans)",
    ),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PlanEvaluatorExerciseStats:
    plans_requested: int = 0
    plans_encoded: int = 0
    features_recorded: int = 0
    labels_recorded: int = 0
    verdicts_exercised: Counter = field(default_factory=Counter)
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
        if self.plans_encoded == 0 and self.plans_requested > 0:
            reasons.append("zero_encodings")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "plans_requested": self.plans_requested,
            "plans_encoded": self.plans_encoded,
            "features_recorded": self.features_recorded,
            "labels_recorded": self.labels_recorded,
            "verdicts_exercised": dict(self.verdicts_exercised),
            "dim_check_passes": self.dim_check_passes,
            "dim_check_failures": self.dim_check_failures,
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Plan Evaluator Exercise — {self.plans_encoded} encoded, "
            f"{self.features_recorded} features + {self.labels_recorded} labels "
            f"in {self.elapsed_s:.1f}s",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.verdicts_exercised:
            lines.append("  Verdicts: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.verdicts_exercised.items())
            ))
        if self.fail_reasons:
            lines.append(f"  FAIL: {', '.join(self.fail_reasons)}")
        else:
            lines.append("  PASS: all checks hold")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_plan_evaluator_exercise(
    profile: PlanEvaluatorExerciseProfile | None = None,
    count: int | None = None,
    collector: Any | None = None,
) -> PlanEvaluatorExerciseStats:
    """Run a synchronous plan evaluator exercise.

    If collector is provided (a DistillationCollector), records signals.
    Otherwise, encodes only (for testing).
    """
    if profile is None:
        profile = PROFILES["coverage"]

    stats = PlanEvaluatorExerciseStats(profile_name=profile.name)

    all_plans = (
        _build_approved_plans()
        + _build_needs_revision_plans()
        + _build_rejected_plans()
    )

    n = count or profile.plan_count
    stats.plans_requested = n

    items_to_run: list[dict[str, Any]] = []
    while len(items_to_run) < n:
        for item in all_plans:
            if profile.randomize and len(items_to_run) >= len(all_plans):
                items_to_run.append(_randomize_plan_item(item))
            else:
                items_to_run.append(item)
            if len(items_to_run) >= n:
                break

    for item in items_to_run:
        try:
            job = item["job"]
            plan = item["plan"]
            verdict = item["verdict"]

            acq_id = f"synthetic_{int(time.time() * 1000)}_{stats.plans_encoded}"
            plan_id = f"plan_{stats.plans_encoded}"

            features = PlanEvaluatorEncoder.encode(job, plan)

            if len(features) != FEATURE_DIM:
                stats.dim_check_failures += 1
            else:
                stats.dim_check_passes += 1

            stats.plans_encoded += 1
            stats.verdicts_exercised[verdict] += 1

            label = encode_verdict(verdict)
            if len(label) != LABEL_DIM:
                stats.dim_check_failures += 1

            if profile.record_signals and collector is not None:
                pair_key = f"{acq_id}:{plan_id}:1"
                try:
                    collector.record(
                        teacher="plan_features",
                        signal_type="synthetic_plan_evaluator",
                        data=features,
                        fidelity=SYNTHETIC_FIDELITY,
                        origin=SYNTHETIC_ORIGIN,
                        metadata={
                            "acquisition_id": acq_id,
                            "plan_id": plan_id,
                            "plan_version": 1,
                        },
                    )
                    stats.features_recorded += 1
                except Exception:
                    pass

                try:
                    collector.record(
                        teacher="acquisition_planner",
                        signal_type="synthetic_plan_evaluator",
                        data=label,
                        fidelity=SYNTHETIC_FIDELITY,
                        origin=SYNTHETIC_ORIGIN,
                        metadata={
                            "acquisition_id": acq_id,
                            "plan_id": plan_id,
                            "plan_version": 1,
                            "verdict": verdict,
                        },
                    )
                    stats.labels_recorded += 1
                except Exception:
                    pass

        except Exception as exc:
            stats.errors.append(f"{item.get('name', '?')}: {type(exc).__name__}: {exc}")

    stats.end_time = time.time()
    logger.info(
        "Plan evaluator exercise: %d plans, %d features + %d labels recorded",
        stats.plans_encoded, stats.features_recorded, stats.labels_recorded,
    )
    return stats
