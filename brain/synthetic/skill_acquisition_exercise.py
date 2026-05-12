"""Synthetic skill-acquisition exercise.

Quarantined weight-room for the SKILL_ACQUISITION specialist. It generates
fabricated lifecycle episodes and records capped-fidelity distillation pairs.

Truth boundary:
  - Uses plain mock objects only, never real stores or registries.
  - Does not create LearningJob, CapabilityAcquisitionJob, PluginRegistry,
    memories, identity records, plugins, approvals, or live capabilities.
  - Optional distillation writes are provenance-tagged origin="synthetic" and
    capped below real operator/sandbox outcomes.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from acquisition.skill_acquisition_encoder import SkillAcquisitionEncoder

SYNTHETIC_FIDELITY = 0.7
SYNTHETIC_ORIGIN = "synthetic"
REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise" / "skill_acquisition_reports"


@dataclass(frozen=True)
class SkillAcquisitionExerciseProfile:
    name: str
    episode_count: int
    randomize: bool = True
    record_signals: bool = False
    description: str = ""


PROFILES = {
    "smoke": SkillAcquisitionExerciseProfile(
        name="smoke",
        episode_count=12,
        randomize=False,
        record_signals=False,
        description="Invariant check only; no distillation recording.",
    ),
    "coverage": SkillAcquisitionExerciseProfile(
        name="coverage",
        episode_count=200,
        randomize=True,
        record_signals=True,
        description="Balanced synthetic coverage for skill acquisition outcomes.",
    ),
    "strict": SkillAcquisitionExerciseProfile(
        name="strict",
        episode_count=120,
        randomize=True,
        record_signals=True,
        description="High-risk boundary cases with capped synthetic fidelity.",
    ),
    "stress": SkillAcquisitionExerciseProfile(
        name="stress",
        episode_count=500,
        randomize=True,
        record_signals=True,
        description="Large synthetic batch for training throughput.",
    ),
}

_SCENARIOS = [
    ("clean_procedural_skill", "verified"),
    ("ambiguous_skill_request", "blocked"),
    ("missing_evidence", "planning_failed"),
    ("weak_plan", "planning_failed"),
    ("bad_prompt", "implementation_failed"),
    ("sandbox_failure", "implementation_failed"),
    ("contract_mismatch", "contract_failed"),
    ("stub_plugin", "implementation_failed"),
    ("retry_success", "verified"),
    ("retry_failure", "contract_failed"),
    ("deployment_blocked", "blocked"),
    ("verified_after_proof", "verified"),
]


@dataclass
class SkillAcquisitionExerciseStats:
    profile_name: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    episodes: int = 0
    features_recorded: int = 0
    labels_recorded: int = 0
    scenarios: Counter = field(default_factory=Counter)
    outcomes: Counter = field(default_factory=Counter)
    invariant_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invariant_failures and self.episodes > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "episodes": self.episodes,
            "features_recorded": self.features_recorded,
            "labels_recorded": self.labels_recorded,
            "scenarios": dict(self.scenarios),
            "outcomes": dict(self.outcomes),
            "invariant_failures": list(self.invariant_failures),
            "passed": self.passed,
        }


def _lane(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status, error="" if status != "failed" else "synthetic_failure")


def _make_episode(scenario: str, outcome: str, idx: int) -> tuple[Any, Any, Any]:
    acq_id = f"synthetic_skill_acq_{idx}"
    failed_planning = outcome == "planning_failed"
    failed_impl = outcome == "implementation_failed"
    contract_failed = outcome == "contract_failed"
    verified = outcome == "verified"

    lanes = {
        "planning": _lane("failed" if failed_planning else "completed"),
        "plan_review": _lane("completed"),
        "implementation": _lane("failed" if failed_impl else "completed"),
        "plugin_quarantine": _lane("completed" if not (failed_planning or failed_impl) else "pending"),
        "verification": _lane("completed" if not (failed_planning or failed_impl) else "pending"),
    }
    job = SimpleNamespace(
        acquisition_id=acq_id,
        status="completed" if verified else ("failed" if outcome != "blocked" else "awaiting_approval"),
        outcome_class="plugin_creation",
        risk_tier=1 if scenario != "deployment_blocked" else 2,
        classification_confidence=1.0,
        doc_artifact_ids=["synthetic_doc"] if scenario != "missing_evidence" else [],
        artifact_refs=["synthetic_plan"],
        requested_by={
            "source": "skill_operational_handoff",
            "skill_id": "data_processing_v1",
            "contract_id": "data_transform_v1",
            "learning_job_id": "synthetic_learning_job",
        },
        lanes=lanes,
        planning_diagnostics={
            "failure_reason": "synthetic_weak_plan" if failed_planning else "",
            "missing_fields": ["test_cases"] if scenario == "weak_plan" else [],
        },
        codegen_prompt_diagnostics={
            "prompt_hash": "" if scenario == "bad_prompt" else f"hash_{idx}",
            "contract_id": "data_transform_v1" if scenario != "bad_prompt" else "",
            "prompt_preview": "synthetic prompt with CSV fixture" if scenario != "bad_prompt" else "thin prompt",
        },
        verification_id="synthetic_verification" if not (failed_planning or failed_impl) else "",
        completed_at=time.time() if verified else 0.0,
    )
    plan = SimpleNamespace(
        technical_approach="" if failed_planning else "Parse CSV with csv.DictReader and compute numeric totals.",
        implementation_sketch="" if scenario == "weak_plan" else "def run(args): ...",
        test_cases=[] if scenario == "weak_plan" else ["csv_basic_totals returns numeric_sums"],
        dependencies=["csv"],
    )
    verification = SimpleNamespace(
        overall_passed=verified,
        lane_verdicts={
            "sandbox_validation": not failed_impl,
            "skill_contract_fixture": verified,
        },
        risk_assessment={
            "sandbox_status": "failed" if failed_impl else "passed",
            "skill_contract_status": "failed" if contract_failed else ("passed" if verified else "not_run"),
            "skill_contract_results": [{"name": "csv_basic_totals", "passed": verified}],
        },
    )
    return job, plan, verification


def run_skill_acquisition_exercise(
    profile: SkillAcquisitionExerciseProfile | None = None,
    count: int | None = None,
    seed: int | None = None,
    collector: Any | None = None,
    progress_callback: Any | None = None,
) -> SkillAcquisitionExerciseStats:
    profile = profile or PROFILES["smoke"]
    rng = random.Random(seed)
    total = int(count or profile.episode_count)
    stats = SkillAcquisitionExerciseStats(profile_name=profile.name)

    if collector is None and profile.record_signals:
        from hemisphere.distillation import DistillationCollector
        collector = DistillationCollector.instance()

    scenarios = list(_SCENARIOS)
    for i in range(total):
        scenario, outcome = rng.choice(scenarios) if profile.randomize else scenarios[i % len(scenarios)]
        job, plan, verification = _make_episode(scenario, outcome, i)
        features = SkillAcquisitionEncoder.encode(job, plan, verification)
        label = SkillAcquisitionEncoder.encode_label(outcome)
        if len(features) != 40:
            stats.invariant_failures.append(f"feature_dim:{len(features)}")
        if len(label) != 5:
            stats.invariant_failures.append(f"label_dim:{len(label)}")

        episode_id = job.acquisition_id
        if collector is not None and profile.record_signals:
            meta = {
                "episode_id": episode_id,
                "acquisition_id": episode_id,
                "scenario": scenario,
                "outcome": outcome,
                "synthetic": True,
            }
            collector.record(
                teacher="skill_acquisition_features",
                signal_type="synthetic_skill_acquisition",
                data=features,
                metadata=meta,
                origin=SYNTHETIC_ORIGIN,
                fidelity=SYNTHETIC_FIDELITY,
            )
            collector.record(
                teacher="skill_acquisition_outcome",
                signal_type="synthetic_skill_acquisition",
                data=label,
                metadata=meta,
                origin=SYNTHETIC_ORIGIN,
                fidelity=SYNTHETIC_FIDELITY,
            )
            stats.features_recorded += 1
            stats.labels_recorded += 1

        stats.episodes += 1
        stats.scenarios[scenario] += 1
        stats.outcomes[outcome] += 1
        if progress_callback is not None:
            try:
                progress_callback(stats, i + 1, total)
            except Exception:
                pass

    stats.ended_at = time.time()
    return stats


def write_report(stats: SkillAcquisitionExerciseStats) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{int(time.time())}_{stats.profile_name}.json"
    path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "PROFILES",
    "SkillAcquisitionExerciseProfile",
    "SkillAcquisitionExerciseStats",
    "run_skill_acquisition_exercise",
    "write_report",
]

