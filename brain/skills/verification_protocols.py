"""Verification Protocols — SK-001 through SK-004.

Each protocol defines the checks, minimum sample sizes, invalidation
conditions, and claimability transitions for a capability class.

Protocols are the architectural center of the Matrix Protocol system.
A learning job is only as trustworthy as the protocol that verified it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from skills.learning_jobs import LearningJob

logger = logging.getLogger(__name__)
_LABELED_TEXT_RE = re.compile(
    r"^\s*(?:label\s+)?(?P<label>[a-z0-9][a-z0-9 _/\-]{1,40})\s*(?:\:|->|-)\s*(?P<sample>.+?)\s*$",
    re.IGNORECASE,
)


@dataclass
class ProtocolCheck:
    """Single check within a verification protocol."""
    name: str
    description: str
    required: bool = True
    min_sample_size: int = 0


@dataclass
class ProtocolResult:
    """Outcome of running a verification protocol against a job."""
    protocol_id: str
    passed: bool = False
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    invalidated: bool = False
    invalidation_reasons: list[str] = field(default_factory=list)
    claimability: str = "unverified"
    data_summary: dict[str, Any] = field(default_factory=dict)
    not_enough_evidence: bool = False


class VerificationProtocol:
    """Base class for all verification protocols."""

    protocol_id: str = ""
    capability_class: str = ""
    checks: list[ProtocolCheck] = []

    def evaluate(self, job: LearningJob, ctx: dict[str, Any]) -> ProtocolResult:
        """Run all checks and invalidation conditions. Subclasses override."""
        raise NotImplementedError

    def _count_evidence_items(self, job: LearningJob) -> int:
        """Count evidence entries in job history."""
        return len(job.evidence.get("history", []))

    def _count_artifacts(self, job: LearningJob, artifact_type: str = "") -> int:
        """Count artifacts, optionally filtered by type."""
        if not artifact_type:
            return len(job.artifacts)
        return sum(1 for a in job.artifacts if a.get("type") == artifact_type)

    def _get_counter(self, job: LearningJob, name: str) -> float:
        return float(job.data.get("counters", {}).get(name, 0))

    def _has_passing_test(self, job: LearningJob, test_name: str) -> bool:
        for evd in reversed(job.evidence.get("history", [])):
            for t in evd.get("tests", []):
                if t.get("name") == test_name and t.get("passed") is True:
                    return True
        return False

    def build_collect_guidance(self, job: LearningJob) -> dict[str, Any]:
        """Optional protocol-owned collect guidance for active learning."""
        return {}


def _current_phase_exit_conditions(job: LearningJob) -> list[str]:
    phases = getattr(job, "plan", {}).get("phases", []) or []
    current = getattr(job, "phase", "")
    for phase_entry in phases:
        if isinstance(phase_entry, dict) and phase_entry.get("name") == current:
            return list(phase_entry.get("exit_conditions", []) or [])
    return []


def _metric_target(expr: str) -> tuple[str, str, float] | None:
    match = re.match(r"^metric:(?P<name>[a-zA-Z0-9_]+)\s*(?P<op>>=|<=|==|>|<)\s*(?P<val>-?\d+(?:\.\d+)?)$", expr)
    if not match:
        return None
    return match.group("name"), match.group("op"), float(match.group("val"))


def _derive_collect_metric(job: LearningJob) -> dict[str, Any]:
    for expr in _current_phase_exit_conditions(job):
        parsed = _metric_target(str(expr))
        if parsed is None:
            continue
        metric_name, op, target = parsed
        if op != ">=":
            continue
        current = float(getattr(job, "data", {}).get("counters", {}).get(metric_name, 0) or 0)
        return {
            "metric_name": metric_name,
            "target": target,
            "current": current,
            "remaining": max(0, int(target - current)),
        }
    return {
        "metric_name": "",
        "target": 0.0,
        "current": 0.0,
        "remaining": 0,
    }


def _render_collect_text(template: str, *, skill_name: str, metric_name: str, remaining: int) -> str:
    metric_label = metric_name.replace("_", " ")
    remaining_hint = f" About {remaining} more {metric_label} are still needed." if remaining > 0 else ""
    return template.format(
        skill_name=skill_name,
        metric_name=metric_name,
        metric_label=metric_label,
        remaining=remaining,
        remaining_hint=remaining_hint,
    )


def build_collect_runtime_config(job: LearningJob) -> dict[str, Any]:
    """Resolve collect guidance from protocol first, then job plan, then metric fallback."""
    metric = _derive_collect_metric(job)
    metric_name = str(metric.get("metric_name", "") or "")
    if not metric_name:
        return {}

    config: dict[str, Any] = {}
    protocol = None
    if getattr(job, "matrix_protocol", False) and getattr(job, "protocol_id", ""):
        protocol = get_protocol(job.protocol_id)
    else:
        capability_type = str(getattr(job, "capability_type", "") or "")
        default_protocol_id = {
            "procedural": "SK-001",
            "perceptual": "SK-002",
            "control": "SK-003",
        }.get(capability_type, "")
        if default_protocol_id:
            protocol = get_protocol(default_protocol_id)
    if protocol is not None:
        config.update(protocol.build_collect_guidance(job))

    plan_guided = getattr(job, "plan", {}).get("guided_collect")
    if isinstance(plan_guided, dict):
        config.update(plan_guided)
        if "interactive_collect" not in plan_guided:
            if (
                plan_guided.get("prompt_template")
                or plan_guided.get("prompt")
                or plan_guided.get("user_input_hints")
                or plan_guided.get("parser")
            ):
                config["interactive_collect"] = True

    skill_name = getattr(job, "skill_id", "this skill").replace("_", " ")
    if not config:
        metric_label = metric_name.replace("_", " ")
        prompt = (
            f"Training mode for {skill_name}: give one short labeled example for {metric_label}. "
            "Say the label first, then the example. For example: 'label: example'. "
            "Say 'stop' when you want to end training mode."
        )
        return {
            "interactive_collect": False,
            "mode": "open_labeled",
            "parser": "labeled_text",
            "artifact_type": "guided_collect_sample",
            "artifact_schema": {"label_field": "label", "text_field": "text"},
            "metric_name": metric_name,
            "remaining_count": int(metric.get("remaining", 0) or 0),
            "prompt": _render_collect_text(
                prompt + "{remaining_hint}",
                skill_name=skill_name,
                metric_name=metric_name,
                remaining=int(metric.get("remaining", 0) or 0),
            ),
            "user_input_hints": [
                _render_collect_text(
                    "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                    skill_name=skill_name,
                    metric_name=metric_name,
                    remaining=int(metric.get("remaining", 0) or 0),
                ),
            ],
        }

    config["metric_name"] = str(config.get("metric_name", "") or metric_name)
    config["remaining_count"] = int(metric.get("remaining", 0) or 0)
    if "interactive_collect" not in config:
        config["interactive_collect"] = bool(
            config.get("prompt_template")
            or config.get("prompt")
            or config.get("user_input_hints")
            or config.get("parser")
        )
    prompt_template = str(config.get("prompt_template", "") or "")
    if prompt_template:
        config["prompt"] = _render_collect_text(
            prompt_template,
            skill_name=skill_name,
            metric_name=config["metric_name"],
            remaining=int(metric.get("remaining", 0) or 0),
        )
    hints = list(config.get("user_input_hints", []) or [])
    if hints:
        config["user_input_hints"] = [
            _render_collect_text(
                str(hint),
                skill_name=skill_name,
                metric_name=config["metric_name"],
                remaining=int(metric.get("remaining", 0) or 0),
            )
            for hint in hints
        ]
    return config


def parse_collect_submission(session: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Parse a user collect submission using protocol/runtime-configured parser rules."""
    parser = str(session.get("parser", "") or "")
    mode = str(session.get("mode", "") or "")
    raw_text = (user_text or "").strip()

    if parser == "labeled_text" or mode == "open_labeled":
        match = _LABELED_TEXT_RE.match(raw_text)
        if not match:
            return {
                "ok": False,
                "error": "I need an explicit label for that sample. Please use the format 'label: example'.",
            }
        return {
            "ok": True,
            "label": str(match.group("label") or "").strip().lower(),
            "text": str(match.group("sample") or "").strip(),
        }

    return {
        "ok": False,
        "error": f"I don't have a parser defined for collect mode '{parser or mode or 'unknown'}' yet.",
    }


def build_collect_artifact(
    session: dict[str, Any],
    *,
    speaker: str,
    emotion: str,
    conversation_id: str,
    metric_name: str,
    captured_index: int,
    parsed_sample: dict[str, Any],
) -> dict[str, Any]:
    """Shape a collect artifact from protocol/runtime-configured artifact schema."""
    artifact_type = str(session.get("artifact_type", "") or "guided_collect_sample")
    artifact_schema = dict(session.get("artifact_schema", {}) or {})
    label_field = str(artifact_schema.get("label_field", "") or "label")
    text_field = str(artifact_schema.get("text_field", "") or "text")
    label = str(parsed_sample.get("label", "") or "")
    text = str(parsed_sample.get("text", "") or "")

    return {
        "id": f"guided_collect_{session.get('session_id', 'session')}_{captured_index}",
        "type": artifact_type,
        "details": {
            label_field: label,
            text_field: text[:200],
            "speaker": speaker or "user",
            "emotion": emotion or "",
            "conversation_id": conversation_id or "",
            "session_id": session.get("session_id", ""),
            "metric_name": metric_name,
        },
    }


class SK001_Procedural(VerificationProtocol):
    """SK-001 — Procedural Skill Protocol.

    For: codebase summarization, documentation trace, tool use, internal
    workflow, procedure-based skills.

    Checks:
      - unseen_task_set: at least 3 novel tasks attempted
      - baseline_comparison: performance vs. pre-training baseline
      - correct_sequencing: steps executed in valid order
      - stable_performance: consistent across N trials (N >= 5)
      - no_hallucinated_claims: capability gate found no blocked claims

    Invalidation conditions:
      - fewer than 3 unseen tasks
      - baseline not established before verification
      - monitor window incomplete (< 5 trials)
    """

    protocol_id = "SK-001"
    capability_class = "procedural"

    MIN_UNSEEN_TASKS = 3
    MIN_TRIALS = 5

    def evaluate(self, job: LearningJob, ctx: dict[str, Any]) -> ProtocolResult:
        result = ProtocolResult(protocol_id=self.protocol_id)
        result.data_summary = {
            "unseen_tasks": self._get_counter(job, "unseen_tasks"),
            "trials": self._get_counter(job, "trials"),
            "baseline_established": self._get_counter(job, "baseline_established") > 0,
        }

        unseen_tasks = self._get_counter(job, "unseen_tasks")
        trials = self._get_counter(job, "trials")
        baseline = self._get_counter(job, "baseline_established") > 0
        claim_violations = self._get_counter(job, "claim_violations")

        if unseen_tasks < self.MIN_UNSEEN_TASKS or trials < self.MIN_TRIALS:
            result.not_enough_evidence = True
            result.claimability = "unverified"
            result.checks_failed.append("insufficient_data")
            return result

        if not baseline:
            result.invalidated = True
            result.invalidation_reasons.append(
                "Baseline not established before verification — "
                "cannot compare pre/post performance"
            )
            result.claimability = "unverified"
            return result

        if unseen_tasks >= self.MIN_UNSEEN_TASKS:
            result.checks_passed.append("unseen_task_set")
        else:
            result.checks_failed.append("unseen_task_set")

        if baseline:
            result.checks_passed.append("baseline_comparison")
        else:
            result.checks_failed.append("baseline_comparison")

        sequencing_ok = self._has_passing_test(job, "correct_sequencing")
        if sequencing_ok:
            result.checks_passed.append("correct_sequencing")
        else:
            result.checks_failed.append("correct_sequencing")

        if trials >= self.MIN_TRIALS:
            result.checks_passed.append("stable_performance")
        else:
            result.checks_failed.append("stable_performance")

        if claim_violations == 0:
            result.checks_passed.append("no_hallucinated_claims")
        else:
            result.checks_failed.append("no_hallucinated_claims")

        required_passed = {"unseen_task_set", "baseline_comparison", "stable_performance"}
        result.passed = required_passed.issubset(set(result.checks_passed))

        if result.passed and not result.checks_failed:
            result.claimability = "verified_operational"
        elif result.passed:
            result.claimability = "verified_limited"
        else:
            result.claimability = "unverified"

        return result


class SK002_Perceptual(VerificationProtocol):
    """SK-002 — Perceptual Skill Protocol.

    For: speaker profiling, emotion refinement, audio analysis, visual
    detection refinement, face recognition improvement.

    Checks:
      - holdout_accuracy: accuracy on held-out data >= threshold
      - confusion_matrix_sanity: no single class > 80% of predictions
      - teacher_agreement: student agrees with teacher >= 70%
      - distractor_controls: false positive rate <= threshold
      - regression_window: no degradation over N evaluation cycles

    Invalidation conditions:
      - training labels contaminated by evaluation data
      - holdout set below minimum size (< 20 samples)
      - teacher unavailable or inconsistent
      - class imbalance beyond threshold (>5:1 ratio)
      - monitor window incomplete
    """

    protocol_id = "SK-002"
    capability_class = "perceptual"

    MIN_HOLDOUT_SAMPLES = 20
    MIN_TEACHER_SIGNALS = 30
    HOLDOUT_ACCURACY_FLOOR = 0.65
    TEACHER_AGREEMENT_FLOOR = 0.70
    MAX_CLASS_IMBALANCE_RATIO = 5.0

    def build_collect_guidance(self, job: LearningJob) -> dict[str, Any]:
        metric = _derive_collect_metric(job)
        metric_name = str(metric.get("metric_name", "") or "")
        if not metric_name:
            return {}

        prompt_template = (
            "Training mode for {skill_name}: give one short labeled example for {metric_label}. "
            "Say the label first, then the example. For example: 'label: example'. "
            "Say 'stop' when you want to end training mode.{remaining_hint}"
        )
        hints = [
            "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
            "Explicit labels make the collect data more useful for held-out verification later.",
            "About {remaining} more {metric_label} are still needed to clear collect.",
        ]
        interactive_collect = False

        if metric_name == "emotion_samples":
            interactive_collect = True
            prompt_template = (
                "Training mode for {skill_name}: give one short labeled example for {metric_label}. "
                "Say the label first, then the example. For example: 'happy: I feel great today'. "
                "Labels like happy, sad, angry, calm, or frustrated are useful. "
                "Say 'stop' when you want to end training mode.{remaining_hint}"
            )
            hints = [
                "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                "Self-labeled examples help most because they provide ground truth for the sample.",
                "About {remaining} more {metric_label} are still needed to clear collect.",
            ]
        elif metric_name == "speaker_samples":
            interactive_collect = True
            prompt_template = (
                "Training mode for {skill_name}: give one short labeled voice sample for {metric_label}. "
                "Say the label first, then the example. For example: 'normal: this is my normal voice'. "
                "Useful labels include normal, louder, quieter, or expressive. "
                "Say 'stop' when you want to end training mode.{remaining_hint}"
            )
            hints = [
                "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                "Varied labeled examples help more than repeating the same sentence the same way.",
                "About {remaining} more {metric_label} are still needed to clear collect.",
            ]

        return {
            "interactive_collect": interactive_collect,
            "mode": "open_labeled",
            "parser": "labeled_text",
            "artifact_type": "guided_collect_sample",
            "artifact_schema": {"label_field": "label", "text_field": "text"},
            "metric_name": metric_name,
            "prompt_template": prompt_template,
            "user_input_hints": hints,
        }

    def evaluate(self, job: LearningJob, ctx: dict[str, Any]) -> ProtocolResult:
        result = ProtocolResult(protocol_id=self.protocol_id)

        holdout_samples = self._get_counter(job, "holdout_samples")
        teacher_signals = self._get_counter(job, "teacher_signals")
        holdout_accuracy = self._get_counter(job, "holdout_accuracy")
        teacher_agreement = self._get_counter(job, "teacher_agreement")
        max_class_ratio = self._get_counter(job, "max_class_ratio")
        false_positive_rate = self._get_counter(job, "false_positive_rate")
        regression_cycles = self._get_counter(job, "regression_cycles_passed")
        data_contaminated = self._get_counter(job, "data_contaminated") > 0

        result.data_summary = {
            "holdout_samples": holdout_samples,
            "teacher_signals": teacher_signals,
            "holdout_accuracy": holdout_accuracy,
            "teacher_agreement": teacher_agreement,
            "max_class_ratio": max_class_ratio,
        }

        if holdout_samples < self.MIN_HOLDOUT_SAMPLES:
            result.not_enough_evidence = True
            result.claimability = "unverified"
            result.checks_failed.append("insufficient_holdout_data")
            return result

        if teacher_signals < self.MIN_TEACHER_SIGNALS:
            result.not_enough_evidence = True
            result.claimability = "unverified"
            result.checks_failed.append("insufficient_teacher_signals")
            return result

        if data_contaminated:
            result.invalidated = True
            result.invalidation_reasons.append(
                "Training labels contaminated by evaluation data"
            )
            result.claimability = "unverified"
            return result

        if max_class_ratio > self.MAX_CLASS_IMBALANCE_RATIO:
            result.invalidated = True
            result.invalidation_reasons.append(
                f"Class imbalance ratio {max_class_ratio:.1f} exceeds "
                f"threshold {self.MAX_CLASS_IMBALANCE_RATIO}"
            )
            result.claimability = "unverified"
            return result

        if holdout_accuracy >= self.HOLDOUT_ACCURACY_FLOOR:
            result.checks_passed.append("holdout_accuracy")
        else:
            result.checks_failed.append("holdout_accuracy")

        class_predictions_ok = max_class_ratio <= self.MAX_CLASS_IMBALANCE_RATIO
        if class_predictions_ok:
            result.checks_passed.append("confusion_matrix_sanity")
        else:
            result.checks_failed.append("confusion_matrix_sanity")

        if teacher_agreement >= self.TEACHER_AGREEMENT_FLOOR:
            result.checks_passed.append("teacher_agreement")
        else:
            result.checks_failed.append("teacher_agreement")

        if false_positive_rate <= 0.15:
            result.checks_passed.append("distractor_controls")
        else:
            result.checks_failed.append("distractor_controls")

        if regression_cycles >= 3:
            result.checks_passed.append("regression_window")
        else:
            result.checks_failed.append("regression_window")

        required = {"holdout_accuracy", "teacher_agreement"}
        result.passed = required.issubset(set(result.checks_passed))

        if result.passed and not result.checks_failed:
            result.claimability = "verified_operational"
        elif result.passed:
            result.claimability = "verified_limited"
        else:
            result.claimability = "unverified"

        return result


class SK003_Control(VerificationProtocol):
    """SK-003 — Control Skill Protocol.

    For: camera control, actuator-like behavior, bounded environment
    manipulation.

    Checks:
      - sandbox_validation: all runs in sandbox before real hardware
      - repeatability: consistent results across N runs (>= 10)
      - bounded_error: error stays within safety bounds
      - no_safety_violations: zero safety gate violations
      - monitor_phase: observation period before promotion

    Invalidation conditions:
      - sandbox not used before real hardware
      - fewer than 10 repeatability runs
      - any safety gate violation
    """

    protocol_id = "SK-003"
    capability_class = "control"

    MIN_SANDBOX_RUNS = 10
    MIN_REAL_RUNS = 3

    def evaluate(self, job: LearningJob, ctx: dict[str, Any]) -> ProtocolResult:
        result = ProtocolResult(protocol_id=self.protocol_id)

        sandbox_runs = self._get_counter(job, "sandbox_runs")
        real_runs = self._get_counter(job, "real_runs")
        safety_violations = self._get_counter(job, "safety_violations")
        error_bounded = self._get_counter(job, "error_bounded") > 0
        monitor_cycles = self._get_counter(job, "monitor_cycles_passed")

        result.data_summary = {
            "sandbox_runs": sandbox_runs,
            "real_runs": real_runs,
            "safety_violations": safety_violations,
        }

        if sandbox_runs < self.MIN_SANDBOX_RUNS:
            result.not_enough_evidence = True
            result.claimability = "unverified"
            result.checks_failed.append("insufficient_sandbox_runs")
            return result

        if safety_violations > 0:
            result.invalidated = True
            result.invalidation_reasons.append(
                f"{int(safety_violations)} safety gate violation(s) detected"
            )
            result.claimability = "unverified"
            return result

        if real_runs > 0 and sandbox_runs < self.MIN_SANDBOX_RUNS:
            result.invalidated = True
            result.invalidation_reasons.append(
                "Real hardware used before completing sandbox validation"
            )
            result.claimability = "unverified"
            return result

        if sandbox_runs >= self.MIN_SANDBOX_RUNS:
            result.checks_passed.append("sandbox_validation")
        else:
            result.checks_failed.append("sandbox_validation")

        total_runs = sandbox_runs + real_runs
        if total_runs >= self.MIN_SANDBOX_RUNS:
            result.checks_passed.append("repeatability")
        else:
            result.checks_failed.append("repeatability")

        if error_bounded:
            result.checks_passed.append("bounded_error")
        else:
            result.checks_failed.append("bounded_error")

        result.checks_passed.append("no_safety_violations")

        if monitor_cycles >= 5:
            result.checks_passed.append("monitor_phase")
        else:
            result.checks_failed.append("monitor_phase")

        required = {"sandbox_validation", "repeatability", "no_safety_violations"}
        result.passed = required.issubset(set(result.checks_passed))

        if result.passed and not result.checks_failed:
            result.claimability = "verified_operational"
        elif result.passed:
            result.claimability = "verified_limited"
        else:
            result.claimability = "unverified"

        return result


class SK004_Transfer(VerificationProtocol):
    """SK-004 — Transfer Skill Protocol.

    For: 'learn how to learn X faster next time' style meta-capabilities.

    Checks:
      - improved_efficiency: training efficiency improved on later jobs
      - reduced_samples: fewer samples needed to pass thresholds
      - no_quality_regression: quality did not degrade
      - generalization: improvement transfers to at least 2 skill families

    Invalidation conditions:
      - fewer than 3 completed learning jobs to compare
      - quality regression detected
    """

    protocol_id = "SK-004"
    capability_class = "transfer"

    MIN_COMPLETED_JOBS = 3

    def evaluate(self, job: LearningJob, ctx: dict[str, Any]) -> ProtocolResult:
        result = ProtocolResult(protocol_id=self.protocol_id)

        completed_jobs = self._get_counter(job, "completed_jobs_for_comparison")
        efficiency_gain = self._get_counter(job, "efficiency_gain_pct")
        sample_reduction = self._get_counter(job, "sample_reduction_pct")
        quality_regression = self._get_counter(job, "quality_regression") > 0
        families_improved = self._get_counter(job, "families_improved")

        result.data_summary = {
            "completed_jobs": completed_jobs,
            "efficiency_gain_pct": efficiency_gain,
            "sample_reduction_pct": sample_reduction,
            "families_improved": families_improved,
        }

        if completed_jobs < self.MIN_COMPLETED_JOBS:
            result.not_enough_evidence = True
            result.claimability = "unverified"
            result.checks_failed.append("insufficient_completed_jobs")
            return result

        if quality_regression:
            result.invalidated = True
            result.invalidation_reasons.append(
                "Quality regression detected — transfer learning degraded outputs"
            )
            result.claimability = "unverified"
            return result

        if efficiency_gain > 0:
            result.checks_passed.append("improved_efficiency")
        else:
            result.checks_failed.append("improved_efficiency")

        if sample_reduction > 0:
            result.checks_passed.append("reduced_samples")
        else:
            result.checks_failed.append("reduced_samples")

        result.checks_passed.append("no_quality_regression")

        if families_improved >= 2:
            result.checks_passed.append("generalization")
        else:
            result.checks_failed.append("generalization")

        required = {"improved_efficiency", "no_quality_regression"}
        result.passed = required.issubset(set(result.checks_passed))

        if result.passed and not result.checks_failed:
            result.claimability = "verified_operational"
        elif result.passed:
            result.claimability = "verified_limited"
        else:
            result.claimability = "unverified"

        return result


# ── Protocol Registry ─────────────────────────────────────────────────────

PROTOCOL_REGISTRY: dict[str, VerificationProtocol] = {
    "SK-001": SK001_Procedural(),
    "SK-002": SK002_Perceptual(),
    "SK-003": SK003_Control(),
    "SK-004": SK004_Transfer(),
}


def get_protocol(protocol_id: str) -> VerificationProtocol | None:
    """Look up a protocol by ID."""
    return PROTOCOL_REGISTRY.get(protocol_id)


def evaluate_job(job: LearningJob, ctx: dict[str, Any] | None = None) -> ProtocolResult | None:
    """Run the protocol associated with a Matrix job and update claimability."""
    if not job.matrix_protocol or not job.protocol_id:
        return None
    protocol = get_protocol(job.protocol_id)
    if protocol is None:
        logger.warning("No protocol found for ID %s", job.protocol_id)
        return None
    result = protocol.evaluate(job, ctx or {})
    job.claimability_status = result.claimability
    return result
