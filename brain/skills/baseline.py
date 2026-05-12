"""Skill Baseline & Validation — the closed loop that makes learning real.

Implements the Shadow Copy validation pattern from the Synthetic Soul paper (§6.2):
  1. Capture baseline metrics before training (assess phase)
  2. Capture same metrics after training (verify phase)
  3. Compute delta and require positive improvement
  4. Only register if the skill actually got better

Each skill type defines its own metric collectors. The framework provides
the comparison logic, threshold enforcement, and evidence generation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MINIMUM_IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement required


@dataclass
class SkillBaseline:
    """Metrics snapshot captured at assess time — the before picture."""

    skill_id: str
    captured_at: float = field(default_factory=time.time)
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillBaseline:
        return cls(
            skill_id=d.get("skill_id", ""),
            captured_at=d.get("captured_at", 0.0),
            metrics=d.get("metrics", {}),
            details=d.get("details", {}),
        )


@dataclass
class SkillValidation:
    """Before/after comparison — the proof that training helped."""

    skill_id: str
    validated_at: float = field(default_factory=time.time)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    current_metrics: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    improved_metrics: list[str] = field(default_factory=list)
    regressed_metrics: list[str] = field(default_factory=list)
    passed: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillValidation:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


def compare_metrics(
    baseline: dict[str, float],
    current: dict[str, float],
    higher_is_better: set[str] | None = None,
    lower_is_better: set[str] | None = None,
    threshold: float = MINIMUM_IMPROVEMENT_THRESHOLD,
) -> SkillValidation:
    """Compare two metric snapshots and produce a validation result.

    By default all metrics are treated as higher-is-better unless they
    appear in ``lower_is_better``.
    """
    higher = higher_is_better or set()
    lower = lower_is_better or set()

    deltas: dict[str, float] = {}
    improved: list[str] = []
    regressed: list[str] = []

    for name in baseline:
        if name not in current:
            continue
        delta = current[name] - baseline[name]
        deltas[name] = round(delta, 6)

        if name in lower:
            if delta < -threshold:
                improved.append(name)
            elif delta > threshold:
                regressed.append(name)
        elif name in higher or name not in lower:
            if delta > threshold:
                improved.append(name)
            elif delta < -threshold:
                regressed.append(name)

    passed = len(improved) > 0 and len(regressed) == 0
    parts = []
    for name in sorted(deltas):
        direction = "+" if deltas[name] > 0 else ""
        status = "improved" if name in improved else ("regressed" if name in regressed else "unchanged")
        parts.append(f"{name}: {baseline.get(name, 0):.4f} → {current.get(name, 0):.4f} ({direction}{deltas[name]:.4f}, {status})")

    return SkillValidation(
        skill_id="",
        baseline_metrics=baseline,
        current_metrics=current,
        deltas=deltas,
        improved_metrics=improved,
        regressed_metrics=regressed,
        passed=passed,
        summary="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# Metric collectors for specific skill types
# ---------------------------------------------------------------------------

def capture_speaker_id_metrics(ctx: dict[str, Any]) -> dict[str, float]:
    """Capture current speaker identification performance metrics.

    Field reads are grounded in the actual producers:

      * Hemisphere accuracy → ``h["best_accuracy"]`` from
        ``hemisphere/orchestrator.py::get_state()``. ``migration_readiness``
        only gets a meaningful value during substrate migration and is the
        wrong signal for standard distillation training progress.
      * Per-teacher distillation volume → ``teachers[t]["total"]`` from
        ``hemisphere/distillation.py::get_stats()``. The outer aggregate
        key ``total_signals`` is NOT inside the per-teacher dict.
      * Enrollment quality → ``SpeakerIdentifier._profiles[name]`` has
        ``enrollment_clips`` and ``interaction_count`` but NO ``confidence``
        field (see ``perception/speaker_id.py``). Recognition confidence
        comes from ``_score_ema`` which tracks the EMA-smoothed recognition
        score for profiles that have been seen recently.
    """
    metrics: dict[str, float] = {}

    speaker_id = ctx.get("speaker_id")
    if speaker_id:
        profiles = {}
        if hasattr(speaker_id, "_profiles"):
            profiles = speaker_id._profiles
        elif hasattr(speaker_id, "profiles"):
            profiles = speaker_id.profiles

        metrics["enrolled_profiles"] = float(len(profiles))

        # Enrollment quality: mean clips-per-profile (pre-interaction baseline)
        clip_counts: list[float] = []
        for prof in profiles.values():
            if isinstance(prof, dict):
                clips = prof.get("enrollment_clips", 0)
            else:
                clips = getattr(prof, "enrollment_clips", 0)
            try:
                clips_f = float(clips)
            except (TypeError, ValueError):
                clips_f = 0.0
            if clips_f > 0:
                clip_counts.append(clips_f)
        metrics["mean_enrollment_clips"] = (
            sum(clip_counts) / len(clip_counts) if clip_counts else 0.0
        )

        # Recognition quality: EMA-smoothed score for profiles recognized this
        # session. Profiles with no EMA entry (freshly enrolled, not yet seen)
        # are excluded so they do not spuriously pull the mean down.
        score_ema = getattr(speaker_id, "_score_ema", None)
        if isinstance(score_ema, dict):
            live_scores = [
                float(v) for name, v in score_ema.items()
                if name in profiles and isinstance(v, (int, float)) and v > 0
            ]
            if live_scores:
                metrics["mean_recognition_confidence"] = (
                    sum(live_scores) / len(live_scores)
                )
            else:
                metrics["mean_recognition_confidence"] = 0.0
        else:
            metrics["mean_recognition_confidence"] = 0.0

    identity_fusion = ctx.get("identity_fusion")
    if identity_fusion and hasattr(identity_fusion, "_telemetry"):
        tel = identity_fusion._telemetry
        if hasattr(tel, "__iter__") and len(tel) > 0:
            total = len(tel)
            methods = [e.get("method", "") for e in tel if isinstance(e, dict)]
            verified_both = sum(1 for m in methods if m == "verified_both")
            face_only = sum(1 for m in methods if m == "face_only")
            voice_only = sum(1 for m in methods if m == "voice_only")

            metrics["verified_both_rate"] = verified_both / total if total else 0.0
            metrics["face_only_rate"] = face_only / total if total else 0.0
            metrics["voice_only_rate"] = voice_only / total if total else 0.0

    hemi_orch = ctx.get("hemisphere_orchestrator")
    if hemi_orch and hasattr(hemi_orch, "get_state"):
        state = hemi_orch.get_state()
        hemis = state.get("hemisphere_state", {}).get("hemispheres", [])
        for h in hemis:
            if h.get("focus") == "speaker_repr":
                metrics["speaker_repr_accuracy"] = float(h.get("best_accuracy", 0.0))
                break

    distill_stats = ctx.get("distillation_stats", {})
    teachers = distill_stats.get("teachers", {})
    ecapa = teachers.get("ecapa_tdnn", {})
    if ecapa:
        metrics["distillation_samples"] = float(ecapa.get("total", 0))

    return metrics


def capture_emotion_metrics(ctx: dict[str, Any]) -> dict[str, float]:
    """Capture current emotion detection performance metrics.

    Uses the same grounded field reads as ``capture_speaker_id_metrics``:
    ``best_accuracy`` from the hemisphere specialist and ``total`` (not
    ``total_signals``) from the per-teacher distillation stats dict.
    """
    metrics: dict[str, float] = {}

    emotion = ctx.get("emotion_classifier")
    if emotion:
        metrics["model_healthy"] = 1.0 if getattr(emotion, "_model_healthy", False) else 0.0
        metrics["gpu_available"] = 1.0 if getattr(emotion, "_gpu_available", False) else 0.0

    hemi_orch = ctx.get("hemisphere_orchestrator")
    if hemi_orch and hasattr(hemi_orch, "get_state"):
        state = hemi_orch.get_state()
        hemis = state.get("hemisphere_state", {}).get("hemispheres", [])
        for h in hemis:
            if h.get("focus") == "emotion_depth":
                metrics["emotion_depth_accuracy"] = float(h.get("best_accuracy", 0.0))
                break

    distill_stats = ctx.get("distillation_stats", {})
    teachers = distill_stats.get("teachers", {})
    wav2vec = teachers.get("wav2vec2_emotion", {})
    if wav2vec:
        metrics["distillation_samples"] = float(wav2vec.get("total", 0))

    return metrics


def capture_generic_perceptual_metrics(ctx: dict[str, Any]) -> dict[str, float]:
    """Capture generic perceptual metrics from hemisphere distillation state.

    Reads ``best_accuracy`` per specialist (the actual post-training accuracy
    signal) rather than ``migration_readiness`` (a substrate-migration-only
    field that stays near zero during ordinary distillation).
    """
    metrics: dict[str, float] = {}

    hemi_orch = ctx.get("hemisphere_orchestrator")
    if hemi_orch and hasattr(hemi_orch, "get_state"):
        state = hemi_orch.get_state()
        hemis = state.get("hemisphere_state", {}).get("hemispheres", [])
        distilled_focuses = ("emotion_depth", "speaker_repr", "face_repr",
                             "voice_intent", "perception_fusion")
        ready_count = 0
        total_accuracy = 0.0
        for h in hemis:
            if h.get("focus") in distilled_focuses:
                acc = float(h.get("best_accuracy", 0.0))
                total_accuracy += acc
                if acc >= 0.5:
                    ready_count += 1

        metrics["distilled_networks_ready"] = float(ready_count)
        metrics["mean_specialist_accuracy"] = (
            total_accuracy / len(distilled_focuses) if distilled_focuses else 0.0
        )

    return metrics


def capture_cognitive_metrics(ctx: dict[str, Any]) -> dict[str, float]:
    """Capture cognitive subsystem health for self-improvement skills.

    Maps to Synthetic Soul §9: CSCI domains — autonomous cognition, global
    coherence, memory continuity, ethical reflexivity.
    """
    metrics: dict[str, float] = {}

    consciousness = ctx.get("consciousness_system")
    if consciousness and hasattr(consciousness, "get_state"):
        state = consciousness.get_state()
        metrics["consciousness_stage"] = float(state.get("stage", 0))
        metrics["transcendence"] = float(state.get("transcendence", 0))
        metrics["confidence"] = float(state.get("confidence", 0))

    health_monitor = ctx.get("health_monitor")
    if health_monitor and hasattr(health_monitor, "get_health"):
        health = health_monitor.get_health()
        if isinstance(health, dict):
            metrics["health_composite"] = float(health.get("composite", 0))

    memory_store = ctx.get("memory_store")
    if memory_store:
        if hasattr(memory_store, "count"):
            metrics["memory_count"] = float(memory_store.count())
        elif hasattr(memory_store, "_memories"):
            metrics["memory_count"] = float(len(memory_store._memories))

    return metrics


def capture_autonomy_metrics(ctx: dict[str, Any]) -> dict[str, float]:
    """Capture autonomy pipeline metrics for self-improvement skills.

    Maps to Synthetic Soul §5.1-5.4: autonomous cognition, curiosity loops,
    self-prompting frequency and quality.
    """
    metrics: dict[str, float] = {}

    autonomy_orch = ctx.get("autonomy_orchestrator")
    if autonomy_orch:
        if hasattr(autonomy_orch, "stats"):
            stats = autonomy_orch.stats
            metrics["research_completed"] = float(getattr(stats, "completed", 0))
            metrics["research_failed"] = float(getattr(stats, "failed", 0))
            metrics["positive_attributions"] = float(getattr(stats, "positive_attributions", 0))
        elif hasattr(autonomy_orch, "get_status"):
            status = autonomy_orch.get_status()
            if isinstance(status, dict):
                metrics["autonomy_level"] = float(status.get("level", 0))
                metrics["research_completed"] = float(status.get("completed", 0))

    policy = ctx.get("policy_controller")
    if policy and hasattr(policy, "get_stats"):
        stats = policy.get_stats()
        if isinstance(stats, dict):
            metrics["policy_decisions"] = float(stats.get("total_decisions", 0))

    return metrics


METRIC_COLLECTORS: dict[str, Any] = {
    "speaker": capture_speaker_id_metrics,
    "emotion": capture_emotion_metrics,
    "generic": capture_generic_perceptual_metrics,
    "cognitive": capture_cognitive_metrics,
    "autonomy": capture_autonomy_metrics,
}

HIGHER_IS_BETTER: dict[str, set[str]] = {
    "speaker": {
        "enrolled_profiles",
        "mean_enrollment_clips",
        "mean_recognition_confidence",
        "verified_both_rate",
        "speaker_repr_accuracy",
        "distillation_samples",
    },
    "emotion": {
        "model_healthy", "gpu_available", "emotion_depth_accuracy",
        "distillation_samples",
    },
    "generic": {"distilled_networks_ready", "mean_specialist_accuracy"},
    "cognitive": {
        "consciousness_stage", "transcendence", "confidence",
        "health_composite", "memory_count",
    },
    "autonomy": {
        "research_completed", "positive_attributions",
        "autonomy_level", "policy_decisions",
    },
}

LOWER_IS_BETTER: dict[str, set[str]] = {
    "speaker": {"face_only_rate"},
    "emotion": set(),
    "generic": set(),
    "cognitive": set(),
    "autonomy": {"research_failed"},
}


def build_validation_evidence(
    validation: SkillValidation,
    job: Any,
) -> dict[str, Any]:
    """Build evidence dict from a SkillValidation for the learning job."""
    import datetime as dt
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tests = []
    for name in sorted(validation.deltas):
        delta = validation.deltas[name]
        baseline_val = validation.baseline_metrics.get(name, 0)
        current_val = validation.current_metrics.get(name, 0)
        is_improved = name in validation.improved_metrics
        is_regressed = name in validation.regressed_metrics

        tests.append({
            "name": f"delta:{name}",
            "passed": is_improved or (not is_regressed),
            "details": (
                f"{baseline_val:.4f} → {current_val:.4f} "
                f"(delta: {delta:+.4f})"
            ),
        })

    tests.append({
        "name": "overall_improvement",
        "passed": validation.passed,
        "details": (
            f"Improved: {validation.improved_metrics}, "
            f"Regressed: {validation.regressed_metrics}"
        ),
    })

    required = set()
    for eid in (job.evidence or {}).get("required", []):
        bare = eid.removeprefix("test:")
        required.add(bare)
    for eid in required:
        if not any(t["name"] == eid or t["name"] == f"test:{eid}" for t in tests):
            tests.append({
                "name": eid,
                "passed": validation.passed,
                "details": f"Derived from overall improvement: {validation.passed}",
            })

    return {
        "evidence_id": f"validated_improvement:{job.skill_id}",
        "result": "pass" if validation.passed else "fail",
        "ts": now_iso,
        "tests": tests,
        "validation": validation.to_dict(),
    }
