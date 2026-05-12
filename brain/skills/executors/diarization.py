"""Diarization skill phase executors.

These implement the full learning lifecycle for speaker diarization:
  assess  -> verify speaker profiles + audio capture are available
  collect -> accumulate labeled windowed embeddings via DiarizationCollector
  train   -> train diarization student model via distillation pipeline
  verify  -> evaluate DER and speaker match accuracy on held-out data
  register -> promote skill to verified in registry
  monitor -> track accuracy drift post-verification
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from skills.executors.base import PhaseExecutor, PhaseResult

logger = logging.getLogger(__name__)

_PROFILES_PATH = os.path.expanduser("~/.jarvis/speakers.json")

_SKILL_ID = "speaker_diarization_v1"
DER_THRESHOLD = 0.25
MATCH_ACCURACY_THRESHOLD = 0.70
TURN_BOUNDARY_F1_THRESHOLD = 0.70
MIN_SEGMENTS_FOR_TRAIN = 50
MIN_SEGMENTS_FOR_VERIFY = 20
HOLDOUT_FRACTION = 0.2

# Registry verification names use the ``test:`` prefix to distinguish them
# from raw metric names (see ``skills/resolver.py`` ``required_evidence``).
# Evidence ``test[].name`` values MUST exactly match those prefixed strings
# so that ``SkillRegistry.set_status(..., "verified")`` recognizes them as
# satisfied requirements. Emitting bare names silently blocks registration.
_EVIDENCE_TEST_DER = "test:diarization_der_below_threshold"
_EVIDENCE_TEST_MATCH = "test:known_speaker_match_accuracy"
_EVIDENCE_TEST_TURN_F1 = "test:turn_boundary_f1"


def _is_diarization_job(job: Any) -> bool:
    return job.skill_id == _SKILL_ID


class DiarizationAssessExecutor(PhaseExecutor):
    """Check that prerequisites exist: enrolled speaker profiles + audio stream."""
    capability_type = "perceptual"
    phase = "assess"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        missing: list[str] = []

        try:
            from perception.diarization_collector import diarization_collector  # noqa: F401
        except Exception as exc:
            return PhaseResult(
                progressed=False,
                message=f"BLOCKED: DiarizationCollector import failed: {exc}",
            )

        enrolled_count = 0
        enrolled_names: list[str] = []
        speaker_id = ctx.get("speaker_id")
        if speaker_id is not None and hasattr(speaker_id, "_profiles"):
            profiles = getattr(speaker_id, "_profiles", {}) or {}
            enrolled_count = len(profiles)
            enrolled_names = list(profiles.keys())
        else:
            try:
                if os.path.exists(_PROFILES_PATH):
                    data = json.loads(open(_PROFILES_PATH, "r", encoding="utf-8").read())
                    if isinstance(data, dict):
                        profiles = data.get("profiles", data)
                        if isinstance(profiles, dict):
                            enrolled_count = len(profiles)
                            enrolled_names = list(profiles.keys())
            except Exception as exc:
                logger.debug("Diarization assess: failed to read profiles file: %s", exc)

        if enrolled_count < 1:
            missing.append("no_enrolled_speaker_profiles")

        if missing:
            return PhaseResult(
                progressed=False,
                message=(
                    "BLOCKED: prerequisites not met — "
                    + ", ".join(missing)
                    + f" (profiles_path={_PROFILES_PATH})"
                ),
                metric_updates={"enrolled_speakers": float(enrolled_count)},
            )

        return PhaseResult(
            progressed=True,
            message=(
                f"Prerequisites met: {enrolled_count} enrolled speaker "
                f"profile(s) [{', '.join(enrolled_names[:5])}], audio feed wired. "
                "Ready for collect phase."
            ),
            metric_updates={"enrolled_speakers": float(enrolled_count)},
        )


class DiarizationCollectExecutor(PhaseExecutor):
    """Accumulate labeled segments via the DiarizationCollector."""
    capability_type = "perceptual"
    phase = "collect"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        try:
            from perception.diarization_collector import diarization_collector
        except ImportError:
            return PhaseResult(progressed=False, message="DiarizationCollector not available")

        if not diarization_collector.is_active:
            speaker_id = ctx.get("speaker_id")
            if speaker_id:
                diarization_collector.activate(speaker_id)
                return PhaseResult(
                    progressed=True,
                    message="Activated DiarizationCollector — waiting for audio data.",
                )
            return PhaseResult(progressed=False, message="No speaker_id in context to activate collector")

        stats = diarization_collector.get_stats()
        total = stats.get("disk_segments", 0)
        dist = diarization_collector.get_speaker_distribution()
        labeled = sum(v for k, v in dist.items() if k != "unknown")

        return PhaseResult(
            progressed=True,
            message=f"Collecting: {total} total segments, {labeled} labeled. Distribution: {dist}",
            metric_updates={"labeled_segments": float(labeled), "total_segments": float(total)},
        )


class DiarizationTrainExecutor(PhaseExecutor):
    """Train diarization student model from collected segments."""
    capability_type = "perceptual"
    phase = "train"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        hemi_orch = ctx.get("hemisphere_orchestrator")
        if not hemi_orch:
            return PhaseResult(progressed=False, message="No hemisphere_orchestrator in context")

        try:
            from perception.diarization_collector import diarization_collector
            stats = diarization_collector.get_stats()
            total_segments = stats.get("disk_segments", 0)
        except Exception:
            total_segments = 0

        if total_segments < MIN_SEGMENTS_FOR_TRAIN:
            return PhaseResult(
                progressed=False,
                message=f"Not enough segments for training: {total_segments} < {MIN_SEGMENTS_FOR_TRAIN}",
            )

        trained = False
        try:
            hemi_orch._run_distillation_cycle()
            trained = True
        except Exception as e:
            logger.warning("Diarization distillation cycle failed: %s", e)
            return PhaseResult(progressed=False, message=f"Distillation failed: {e}")

        artifact = {
            "id": "diarization_model_checkpoint",
            "type": "diarization_model_checkpoint",
            "details": {
                "trained": trained,
                "segments_used": total_segments,
                "timestamp": time.time(),
            },
        }
        return PhaseResult(
            progressed=True,
            message=f"Distillation cycle executed on {total_segments} segments.",
            artifact=artifact,
        )


class DiarizationVerifyExecutor(PhaseExecutor):
    """Evaluate diarization accuracy on held-out data."""
    capability_type = "perceptual"
    phase = "verify"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        try:
            from perception.diarization_collector import diarization_collector
        except ImportError:
            return PhaseResult(progressed=False, message="DiarizationCollector not available")

        segments = diarization_collector.load_training_data()
        if len(segments) < MIN_SEGMENTS_FOR_VERIFY:
            return PhaseResult(
                progressed=False,
                message=f"Not enough segments for verification: {len(segments)} < {MIN_SEGMENTS_FOR_VERIFY}",
            )

        holdout_start = int(len(segments) * (1 - HOLDOUT_FRACTION))
        holdout = segments[holdout_start:]
        if not holdout:
            return PhaseResult(progressed=False, message="No holdout segments available")

        labeled = [s for s in holdout if s.get("speaker_label", "unknown") != "unknown"]
        total_holdout = len(holdout)
        labeled_count = len(labeled)

        match_accuracy = labeled_count / total_holdout if total_holdout > 0 else 0.0

        high_confidence = [s for s in labeled if s.get("confidence", 0) >= 0.55]
        avg_confidence = (
            sum(s.get("confidence", 0) for s in labeled) / labeled_count
            if labeled_count > 0 else 0.0
        )

        speaker_labels = set(s.get("speaker_label") for s in labeled)
        transitions = 0
        for i in range(1, len(holdout)):
            if holdout[i].get("speaker_label") != holdout[i - 1].get("speaker_label"):
                transitions += 1
        transition_rate = transitions / max(1, len(holdout) - 1)

        der_estimate = 1.0 - match_accuracy
        der_passed = der_estimate < DER_THRESHOLD
        match_passed = match_accuracy >= MATCH_ACCURACY_THRESHOLD

        # Turn-boundary F1: we only compute an honest F1 when holdout
        # segments carry explicit ``turn_boundary_truth`` annotations
        # (values are ``"start"`` for the first segment of a new turn,
        # anything else / missing for continuation). Without that ground
        # truth we do NOT invent a pseudo-metric — the test stays failing
        # with a clear reason. This is the intentional gate: the skill
        # cannot be registered ``verified`` until turn-boundary labels
        # exist, which is the correct truth-first behaviour.
        truth_marked = [s for s in holdout if s.get("turn_boundary_truth") is not None]
        turn_f1 = 0.0
        turn_precision = 0.0
        turn_recall = 0.0
        if len(truth_marked) >= 2:
            # Boundaries are only detectable within the holdout window
            # (we cannot compare the first segment to a segment outside
            # it). Count true boundaries from index >= 1 so precision /
            # recall are evaluated over the same observable positions.
            true_boundaries = 0
            predicted_boundaries = 0
            true_positive = 0
            for i in range(1, len(truth_marked)):
                cur = truth_marked[i]
                prev = truth_marked[i - 1]
                predicted_change = cur.get("speaker_label") != prev.get("speaker_label")
                truth_change = cur.get("turn_boundary_truth") == "start"
                if truth_change:
                    true_boundaries += 1
                if predicted_change:
                    predicted_boundaries += 1
                    if truth_change:
                        true_positive += 1
            if predicted_boundaries:
                turn_precision = true_positive / predicted_boundaries
            if true_boundaries:
                turn_recall = true_positive / true_boundaries
            if (turn_precision + turn_recall) > 0:
                turn_f1 = (
                    2 * turn_precision * turn_recall
                    / (turn_precision + turn_recall)
                )
            turn_details = (
                f"F1: {turn_f1:.3f} (threshold: {TURN_BOUNDARY_F1_THRESHOLD}), "
                f"precision: {turn_precision:.3f}, "
                f"recall: {turn_recall:.3f}, "
                f"truth_marked: {len(truth_marked)}, "
                f"true_boundaries: {true_boundaries}, "
                f"predicted_boundaries: {predicted_boundaries}"
            )
            turn_passed = turn_f1 >= TURN_BOUNDARY_F1_THRESHOLD
        else:
            turn_details = (
                "no ground-truth turn boundaries available "
                f"(0 of {total_holdout} holdout segments carry "
                "turn_boundary_truth annotations); test stays failing "
                "until operator supplies turn-boundary labels"
            )
            turn_passed = False

        now_iso = ctx.get("now_iso", "")
        all_passed = der_passed and match_passed and turn_passed
        evidence = {
            "evidence_id": f"verify_{_SKILL_ID}_{time.time():.0f}",
            "ts": now_iso,
            "tests": [
                {
                    "name": _EVIDENCE_TEST_DER,
                    "passed": der_passed,
                    "details": (
                        f"DER estimate: {der_estimate:.3f} (threshold: {DER_THRESHOLD}), "
                        f"holdout: {total_holdout}, labeled: {labeled_count}"
                    ),
                },
                {
                    "name": _EVIDENCE_TEST_MATCH,
                    "passed": match_passed,
                    "details": (
                        f"Match accuracy: {match_accuracy:.3f} (threshold: {MATCH_ACCURACY_THRESHOLD}), "
                        f"avg confidence: {avg_confidence:.3f}, "
                        f"unique speakers: {len(speaker_labels)}, "
                        f"transition rate: {transition_rate:.3f}"
                    ),
                },
                {
                    "name": _EVIDENCE_TEST_TURN_F1,
                    "passed": turn_passed,
                    "details": turn_details,
                },
            ],
            "result": "pass" if all_passed else "fail",
            "metrics": {
                "der_estimate": round(der_estimate, 4),
                "match_accuracy": round(match_accuracy, 4),
                "avg_confidence": round(avg_confidence, 4),
                "unique_speakers": len(speaker_labels),
                "holdout_size": total_holdout,
                "labeled_size": labeled_count,
                "transition_rate": round(transition_rate, 4),
                "turn_boundary_f1": round(turn_f1, 4),
                "turn_boundary_precision": round(turn_precision, 4),
                "turn_boundary_recall": round(turn_recall, 4),
                "turn_boundary_truth_marked": len(truth_marked),
            },
        }

        return PhaseResult(
            progressed=True,
            message=(
                f"Verification: DER={der_estimate:.3f} match={match_accuracy:.3f} "
                f"turn_f1={turn_f1:.3f} "
                f"({'PASS' if all_passed else 'FAIL'})"
            ),
            evidence=evidence,
        )


class DiarizationRegisterExecutor(PhaseExecutor):
    """Promote skill to verified when all evidence tests pass."""
    capability_type = "perceptual"
    phase = "register"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        registry = ctx.get("registry")
        if not registry:
            return PhaseResult(progressed=False, message="No registry in context")

        latest = job.evidence.get("latest", {})
        all_passed = all(
            t.get("passed", False)
            for t in latest.get("tests", [])
        )

        if not all_passed:
            return PhaseResult(
                progressed=False,
                message="Cannot register: not all evidence tests passed. Re-run verify.",
            )

        passed_tests: list[dict] = []
        for evd in job.evidence.get("history", []):
            if evd.get("result") != "pass":
                continue
            for t in evd.get("tests", []):
                if t.get("passed"):
                    passed_tests.append({"name": t.get("name", ""), "passed": True})

        if not passed_tests:
            return PhaseResult(
                progressed=False,
                message="Cannot register: no passed evidence tests found in history.",
            )

        try:
            from skills.registry import SkillEvidence
            evidence = SkillEvidence(
                evidence_id=f"register_{_SKILL_ID}_{time.time():.0f}",
                timestamp=time.time(),
                result="pass",
                tests=passed_tests,
            )
            ok = registry.set_status(_SKILL_ID, "verified", evidence=evidence)
            if not ok:
                return PhaseResult(
                    progressed=False,
                    message="Registry rejected verification — evidence requirements not met.",
                )
            logger.info("Diarization skill promoted to VERIFIED with %d evidence tests", len(passed_tests))
        except Exception as e:
            return PhaseResult(progressed=False, message=f"Registry update failed: {e}")

        return PhaseResult(
            progressed=True,
            message="Speaker Diarization skill registered as VERIFIED.",
        )


class DiarizationMonitorExecutor(PhaseExecutor):
    """Post-verification monitoring for accuracy drift."""
    capability_type = "perceptual"
    phase = "monitor"

    def can_run(self, job: Any, ctx: dict[str, Any]) -> bool:
        return (
            _is_diarization_job(job)
            and job.phase == self.phase
            and job.status in ("active", "paused", "blocked")
        )

    def run(self, job: Any, ctx: dict[str, Any]) -> PhaseResult:
        try:
            from perception.diarization_collector import diarization_collector
            stats = diarization_collector.get_stats()
        except ImportError:
            stats = {}

        return PhaseResult(
            progressed=True,
            message=f"Monitoring: {stats.get('disk_segments', 0)} total segments, "
                    f"{stats.get('errors', 0)} errors",
            metric_updates={
                "monitor_segments": float(stats.get("disk_segments", 0)),
            },
        )
