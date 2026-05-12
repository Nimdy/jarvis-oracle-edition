"""Learning Job Store and Orchestrator.

A learning job is a persisted, multi-phase workflow that lets Jarvis genuinely
acquire a new capability.  Phases: assess -> research -> acquire -> integrate ->
collect -> train -> verify -> register -> (monitor).

Jobs produce artifacts and evidence.  Only evidence can flip a skill to
``verified`` in the SkillRegistry.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["active", "paused", "awaiting_operator_approval", "blocked", "completed", "failed"]
JobPhase = Literal[
    "assess", "research", "acquire", "integrate",
    "collect", "train", "verify", "register", "monitor",
]
CapabilityType = Literal["procedural", "perceptual", "control"]

DEFAULT_DIR = os.path.expanduser("~/.jarvis/learning_jobs")

MAX_PHASE_FAILURES = 10
TICK_COOLDOWN_S = 120.0
PHASE_TIMEOUT_S = 3600.0

_NATIVE_COMPLETION_ALLOWLIST: frozenset[str] = frozenset({
    "emotion_detection_v1",
    "speaker_identification_v1",
})

_NATIVE_PERCEPTUAL_PLANS: dict[str, dict] = {
    "speaker_identification_v1": {
        "summary": "Native perceptual: speaker identification via ECAPA-TDNN distillation",
        "phases": [
            {"name": "assess", "exit_conditions": ["gate:speaker_profiles_exist"]},
            {"name": "collect", "exit_conditions": ["metric:speaker_samples>=5"]},
            {"name": "train", "exit_conditions": ["artifact:train_tick"]},
            {"name": "verify", "exit_conditions": [
                "evidence:test:speaker_id_accuracy_min",
                "evidence:test:speaker_id_false_positive_max",
            ]},
            {"name": "register", "exit_conditions": ["skill_status:verified"]},
        ],
    },
    "emotion_detection_v1": {
        "summary": "Native perceptual: emotion detection via wav2vec2 distillation",
        "phases": [
            {"name": "assess", "exit_conditions": ["gate:emotion_model_available"]},
            {"name": "collect", "exit_conditions": ["metric:emotion_samples>=5"]},
            {"name": "train", "exit_conditions": ["artifact:train_tick"]},
            {"name": "verify", "exit_conditions": [
                "evidence:test:emotion_accuracy_min",
                "evidence:test:emotion_confusion_matrix_ok",
            ]},
            {"name": "register", "exit_conditions": ["skill_status:verified"]},
        ],
    },
}


def _utc_iso(ts: float | None = None) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts or time.time(), dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


ClaimabilityStatus = Literal[
    "unverified", "verified_limited", "verified_operational",
]
PromotionStatus = Literal[
    "none", "candidate", "probationary", "promoted", "retired",
]


@dataclass
class LearningJob:
    job_id: str
    skill_id: str
    capability_type: CapabilityType
    risk_level: str = "low"
    status: JobStatus = "active"
    phase: JobPhase = "assess"

    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)

    requested_by: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5

    # ── Matrix Protocol fields ────────────────────────────────────────
    matrix_protocol: bool = False
    protocol_id: str = ""               # "SK-001", "SK-002", etc.
    matrix_target: str = ""             # original user description
    verification_profile: str = ""      # protocol family name
    claimability_status: ClaimabilityStatus = "unverified"
    specialist_focus: str | None = None
    promotion_status: PromotionStatus = "none"

    plan: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=lambda: {"hard": [], "soft": []})
    data: dict[str, Any] = field(default_factory=lambda: {"sources": [], "counters": {}})
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=lambda: {
        "required": [], "latest": None, "history": [],
    })
    events: list[dict[str, Any]] = field(default_factory=list)
    failure: dict[str, Any] = field(default_factory=lambda: {
        "count": 0, "last_error": None, "last_failed_phase": None,
    })
    executor_state: dict[str, Any] = field(default_factory=dict)
    ledger_entry_id: str = ""
    parent_acquisition_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **self.__dict__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LearningJob:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class LearningJobStore:
    """Persists each job as a separate JSON file under ``~/.jarvis/learning_jobs/``."""

    def __init__(self, root: str = DEFAULT_DIR) -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path_for(self, job_id: str) -> str:
        return os.path.join(self.root, f"{job_id}.json")

    def save(self, job: LearningJob) -> None:
        job.updated_at = _utc_iso()
        path = self._path_for(job.job_id)
        try:
            from memory.persistence import atomic_write_json
            atomic_write_json(path, job.to_dict())
        except Exception:
            logger.exception("Failed to save learning job %s", job.job_id)

    def load(self, job_id: str) -> LearningJob | None:
        path = self._path_for(job_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return LearningJob.from_dict(raw)
        except Exception:
            logger.exception("Failed to load learning job %s", job_id)
            return None

    def load_all(self) -> list[LearningJob]:
        jobs: list[LearningJob] = []
        if not os.path.isdir(self.root):
            return jobs
        for fname in os.listdir(self.root):
            if not fname.endswith(".json"):
                continue
            job_id = fname[:-5]
            job = self.load(job_id)
            if job is not None:
                jobs.append(job)
        return jobs

    def delete(self, job_id: str) -> bool:
        path = self._path_for(job_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


class LearningJobOrchestrator:
    """Manages learning job lifecycle — create, advance, fail, tick."""

    def __init__(
        self,
        store: LearningJobStore,
        registry: Any = None,
    ) -> None:
        self.store = store
        self._registry = registry
        self._active_jobs: dict[str, LearningJob] = {}
        self._context_providers: dict[str, Any] = {}
        self._dispatcher = self._build_dispatcher()
        self._load_active()
        self._cleanup_actuator_junk_jobs()
        self._purge_completed_generic_fallback_jobs()
        self._wire_acquisition_terminal_events()

    def set_context_provider(self, key: str, provider: Any) -> None:
        """Register a runtime object that executors need (e.g., tts_engine)."""
        self._context_providers[key] = provider

    def _wire_acquisition_terminal_events(self) -> None:
        """Mirror terminal acquisition results without waiting for slow verify ticks."""
        try:
            from consciousness.events import event_bus
            event_bus.on("acquisition:failed", self._on_acquisition_terminal)
            event_bus.on("acquisition:cancelled", self._on_acquisition_terminal)
        except Exception:
            logger.debug("Learning job acquisition terminal listeners unavailable", exc_info=True)

    def _build_cycle_context(self, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime_ctx = dict(ctx) if ctx else {}
        runtime_ctx.update({k: v() if callable(v) else v for k, v in self._context_providers.items()})
        if self._registry and "registry" not in runtime_ctx:
            runtime_ctx["registry"] = self._registry
        return runtime_ctx

    def _on_acquisition_terminal(self, acquisition_id: str = "", **_: Any) -> None:
        if not acquisition_id:
            return
        ctx = self._build_cycle_context()
        for job in list(self._active_jobs.values()):
            handoff = (job.data or {}).get("operational_handoff", {})
            linked_acquisition_id = (
                getattr(job, "parent_acquisition_id", "")
                or (handoff.get("acquisition_id", "") if isinstance(handoff, dict) else "")
            )
            if linked_acquisition_id == acquisition_id:
                self._sync_terminal_acquisition_if_needed(job, ctx)
                return

    @staticmethod
    def _build_dispatcher() -> Any:
        """Wire up the executor dispatcher with all known phase executors."""
        try:
            from skills.executors.dispatcher import ExecutorDispatcher
            from skills.executors.procedural import (
                ProceduralAssessExecutor,
                ProceduralResearchExecutor,
                ProceduralAcquireExecutor,
                ProceduralIntegrateExecutor,
                ProceduralVerifyExecutor,
                ProceduralRegisterExecutor,
            )
            from skills.executors.perceptual import (
                PerceptualAssessExecutor,
                PerceptualCollectExecutor,
                PerceptualTrainExecutor,
                PerceptualVerifyExecutor,
                PerceptualRegisterExecutor,
            )
            from skills.executors.control import (
                ControlAssessExecutor,
                ControlCollectExecutor,
                ControlTrainExecutor,
                ControlVerifyExecutor,
                ControlRegisterExecutor,
            )
            from skills.executors.diarization import (
                DiarizationAssessExecutor,
                DiarizationCollectExecutor,
                DiarizationTrainExecutor,
                DiarizationVerifyExecutor,
                DiarizationRegisterExecutor,
                DiarizationMonitorExecutor,
            )
            d = ExecutorDispatcher()
            for ex_cls in (
                ProceduralAssessExecutor, ProceduralResearchExecutor, ProceduralAcquireExecutor,
                ProceduralIntegrateExecutor, ProceduralVerifyExecutor, ProceduralRegisterExecutor,
                DiarizationAssessExecutor, DiarizationCollectExecutor, DiarizationTrainExecutor,
                DiarizationVerifyExecutor, DiarizationRegisterExecutor, DiarizationMonitorExecutor,
                PerceptualAssessExecutor, PerceptualCollectExecutor, PerceptualTrainExecutor, PerceptualVerifyExecutor,
                PerceptualRegisterExecutor,
                ControlAssessExecutor, ControlCollectExecutor, ControlTrainExecutor, ControlVerifyExecutor,
                ControlRegisterExecutor,
            ):
                d.register(ex_cls())
            logger.info("ExecutorDispatcher wired with %d executors", len(d.executors))
            return d
        except Exception:
            logger.warning("Could not build ExecutorDispatcher — falling back to auto-advance only")
            return None

    _ACTUATOR_JUNK_KEYWORDS = frozenset({
        "zoom", "camera", "focus", "autofocus", "pan", "tilt",
        "exposure", "brightness", "wide_angle",
    })

    def _load_active(self) -> None:
        for job in self.store.load_all():
            if job.status in ("active", "paused", "awaiting_operator_approval", "blocked"):
                self._active_jobs[job.job_id] = job
        self._inject_native_plans()
        if self._active_jobs:
            logger.info("Loaded %d active learning jobs", len(self._active_jobs))

    def _inject_native_plans(self) -> None:
        """Inject proper phase plans for allowlisted native perceptual jobs.

        Jobs created by the capability gate or auto-discovery may arrive
        with an empty plan dict, which blocks auto-advance.  Only the
        strict allowlist gets this treatment.
        """
        for job in self._active_jobs.values():
            if job.skill_id not in _NATIVE_COMPLETION_ALLOWLIST:
                continue
            if job.plan and job.plan.get("phases"):
                continue
            native_plan = _NATIVE_PERCEPTUAL_PLANS.get(job.skill_id)
            if not native_plan:
                continue
            job.plan = dict(native_plan)
            if not job.evidence.get("required"):
                verify_phase = next(
                    (p for p in native_plan["phases"] if p["name"] == "verify"),
                    None,
                )
                if verify_phase:
                    job.evidence["required"] = [
                        cond.split(":", 1)[1]
                        for cond in verify_phase["exit_conditions"]
                        if cond.startswith("evidence:")
                    ]
            if job.capability_type != "perceptual":
                job.capability_type = "perceptual"
            if not job.matrix_protocol:
                job.matrix_protocol = True
                job.protocol_id = "SK-002"
            job.events.append({
                "ts": _utc_iso(),
                "type": "plan_injected",
                "msg": f"Native plan injected for {job.skill_id}",
            })
            self.store.save(job)
            logger.info(
                "Injected native plan for job %s (skill=%s, phase=%s)",
                job.job_id, job.skill_id, job.phase,
            )

    def _cleanup_actuator_junk_jobs(self) -> None:
        """Block-close jobs whose skill_id matches a known actuator keyword
        or fails actionability checks.

        Actuator jobs were auto-created by the capability gate before the
        actuator-domain suppression fix.  Non-actionable jobs are garbage
        from emotional/stance phrase leakage.
        """
        cleaned = 0
        is_actionable = None
        try:
            from skills.discovery import is_actionable_capability_phrase
            is_actionable = is_actionable_capability_phrase
        except ImportError:
            pass

        for job_id, job in list(self._active_jobs.items()):
            if job.status in ("completed", "failed"):
                continue
            sid = job.skill_id.lower()
            should_block = any(kw in sid for kw in self._ACTUATOR_JUNK_KEYWORDS)
            reason = "superseded_by_builtin_actuator_capability"

            if not should_block and is_actionable:
                import re as _re
                try:
                    from skills.discovery import BUILTIN_FAMILIES
                    clean_id = _re.sub(r'_v\d+$', '', sid)
                    is_builtin = any(
                        clean_id == fam_id or clean_id in fam.aliases
                        for fam_id, fam in BUILTIN_FAMILIES.items()
                    )
                except ImportError:
                    is_builtin = False
                if not is_builtin:
                    phrase = _re.sub(r'\s+v\d+$', '', sid.replace("_", " "))
                    if not is_actionable(phrase):
                        should_block = True
                        reason = "non_actionable_skill_phrase"

            if should_block:
                job.status = "blocked"
                job.updated_at = _utc_iso()
                job.events.append({
                    "ts": _utc_iso(),
                    "type": "job_blocked",
                    "msg": reason,
                })
                self.store.save(job)
                self._active_jobs.pop(job_id, None)
                cleaned += 1
        if cleaned:
            logger.info("Cleaned up %d junk learning jobs", cleaned)

        self._purge_verify_blocked_junk()
        self._purge_terminal_unverifiable_jobs()

    def _purge_verify_blocked_junk(self) -> None:
        """Remove blocked jobs that failed verification with no feasible path.

        These are auto-gate-created jobs from speech fragments that reached
        the verify phase but had no domain-specific verification method.
        Deletes job files from disk and removes orphaned SkillRecords.
        """
        _JUNK_ERRORS = ("No verifiable artifact found", "no_verification_method")
        to_purge: list[str] = []
        for job in self.store.load_all():
            if job.status != "blocked":
                continue
            fail = getattr(job, "failure", None) or {}
            last_err = fail.get("last_error") or ""
            if last_err and any(e in last_err for e in _JUNK_ERRORS):
                to_purge.append(job.job_id)

        purged = 0
        for job_id in to_purge:
            if self.delete_job(job_id, remove_skill=True):
                purged += 1

        if purged:
            logger.info("Purged %d verify-blocked junk jobs + skill records", purged)

    @staticmethod
    def _is_generic_fallback_job(job: LearningJob) -> bool:
        plan_summary = (job.plan or {}).get("summary", "") or ""
        required = ((job.evidence or {}).get("required", [])) or []
        return (
            job.capability_type == "procedural"
            and "Auto-generated from:" in plan_summary
            and required == ["test:procedure_smoke"]
        )

    @classmethod
    def _is_terminal_unverifiable_job(cls, job: LearningJob) -> bool:
        last_error = ((job.failure or {}).get("last_error") or "").lower()
        if "no_verification_method" not in last_error:
            return False
        return cls._is_generic_fallback_job(job)

    def _purge_terminal_unverifiable_jobs(self) -> None:
        """Delete generic fallback jobs once verification proves they are unreal."""
        purged = 0
        for job in self.store.load_all():
            if not self._is_terminal_unverifiable_job(job):
                continue
            if self.delete_job(job.job_id, remove_skill=True):
                purged += 1
        if purged:
            logger.info("Purged %d terminal unverifiable fallback learning jobs", purged)

    def _purge_completed_generic_fallback_jobs(self) -> None:
        """Delete completed generic fallback jobs that should never count as real skills."""
        purged = 0
        for job in self.store.load_all():
            if job.status != "completed":
                continue
            if not self._is_generic_fallback_job(job):
                continue
            if self.delete_job(job.job_id, remove_skill=True):
                purged += 1
        if purged:
            logger.info("Purged %d completed generic fallback learning jobs", purged)

    def create_job(
        self,
        skill_id: str,
        capability_type: CapabilityType,
        requested_by: dict[str, Any],
        risk_level: str = "low",
        required_evidence: list[str] | None = None,
        priority: float = 0.5,
        plan: dict[str, Any] | None = None,
        hard_gates: list[dict[str, Any]] | None = None,
    ) -> LearningJob | None:
        try:
            import re as _re
            from skills.discovery import is_actionable_capability_phrase, BUILTIN_FAMILIES
            clean_id = _re.sub(r'_v\d+$', '', skill_id)
            is_builtin = any(
                clean_id == fam_id or clean_id in fam.aliases
                for fam_id, fam in BUILTIN_FAMILIES.items()
            )
            if not is_builtin:
                phrase = clean_id.replace("_", " ").strip()
                if not is_actionable_capability_phrase(phrase):
                    logger.warning("create_job rejected non-actionable skill_id: '%s'", skill_id)
                    return None
        except ImportError:
            pass

        import re as _re2
        canonical = _re2.sub(r'_v\d+$', '', skill_id)
        for existing in self._active_jobs.values():
            existing_canonical = _re2.sub(r'_v\d+$', '', existing.skill_id)
            if existing_canonical == canonical and existing.status in (
                "active", "running", "in_progress", "awaiting_operator_approval", "blocked"
            ):
                logger.info(
                    "create_job dedup: '%s' already covered by job '%s' (skill_id='%s', status=%s)",
                    skill_id, existing.job_id, existing.skill_id, existing.status,
                )
                return None

        ts = _utc_iso()
        job_id = f"job_{ts.replace(':', '').replace('-', '')}_{uuid.uuid4().hex[:4]}"
        job = LearningJob(
            job_id=job_id,
            skill_id=skill_id,
            capability_type=capability_type,
            risk_level=risk_level,
            status="active",
            phase="assess",
            requested_by=requested_by,
            priority=priority,
            created_at=ts,
            updated_at=ts,
        )
        if plan:
            job.plan = plan
        if required_evidence:
            job.evidence["required"] = required_evidence
        if hard_gates:
            job.gates["hard"] = hard_gates

        job.events.append({"ts": ts, "type": "job_created", "msg": "Created learning job."})
        self.store.save(job)
        self._active_jobs[job.job_id] = job

        if self._registry:
            rec = self._registry.get(skill_id)
            if rec is not None and rec.status == "unknown":
                self._registry.set_status(skill_id, "learning")
                rec.learning_job_id = job.job_id
                self._registry.save()

        try:
            from consciousness.events import event_bus, SKILL_LEARNING_STARTED
            event_bus.emit(
                SKILL_LEARNING_STARTED,
                job_id=job.job_id, skill_id=skill_id,
                capability_type=capability_type,
            )
        except Exception:
            pass

        try:
            from consciousness.attribution_ledger import attribution_ledger
            job.ledger_entry_id = attribution_ledger.record(
                subsystem="learning_jobs",
                event_type="learning_job_created",
                actor=requested_by.get("source", "unknown") if isinstance(requested_by, dict) else "unknown",
                source=requested_by.get("source", "") if isinstance(requested_by, dict) else "",
                data={"skill_id": skill_id, "capability_type": capability_type, "requested_by": requested_by or {}},
                evidence_refs=[
                    {"kind": "skill", "id": skill_id},
                    {"kind": "job", "id": job.job_id},
                ],
            )
            self.store.save(job)
        except Exception:
            pass

        logger.info("Created learning job %s for skill %s", job.job_id, skill_id)
        return job

    def set_gate(self, job: LearningJob, gate_id: str, state: str, details: str = "") -> None:
        found = False
        for bucket in ("hard", "soft"):
            for g in job.gates.get(bucket, []):
                if g.get("id") == gate_id:
                    g["state"] = state
                    if details:
                        g["details"] = details
                    found = True
        if not found:
            job.gates.setdefault("hard", []).append({
                "id": gate_id, "kind": "custom", "required": True,
                "state": state, "details": details,
            })
        job.events.append({"ts": _utc_iso(), "type": "gate_update", "msg": f"{gate_id} -> {state}"})
        self.store.save(job)

    def add_artifact(self, job: LearningJob, artifact: dict[str, Any]) -> None:
        artifact_id = artifact.get("id", "")
        artifact_type = artifact.get("type", "")
        artifact_path = artifact.get("path", "")
        for idx, existing in enumerate(job.artifacts):
            if (
                existing.get("id", "") == artifact_id
                and existing.get("type", "") == artifact_type
                and existing.get("path", "") == artifact_path
            ):
                job.artifacts[idx] = artifact
                job.events.append({
                    "ts": _utc_iso(), "type": "artifact_updated",
                    "msg": artifact.get("id", artifact.get("type", "artifact")),
                })
                self.store.save(job)
                return

        job.artifacts.append(artifact)
        job.events.append({
            "ts": _utc_iso(), "type": "artifact_added",
            "msg": artifact.get("id", artifact.get("type", "artifact")),
        })
        self.store.save(job)

    def record_evidence(self, job: LearningJob, evidence_obj: dict[str, Any]) -> None:
        job.evidence["history"].append(evidence_obj)
        job.evidence["latest"] = evidence_obj
        job.events.append({
            "ts": _utc_iso(), "type": "evidence_recorded",
            "msg": evidence_obj.get("evidence_id", "evidence"),
        })
        self.store.save(job)
        try:
            from consciousness.events import event_bus, SKILL_VERIFICATION_RECORDED
            event_bus.emit(
                SKILL_VERIFICATION_RECORDED,
                job_id=job.job_id, skill_id=job.skill_id,
                evidence_id=evidence_obj.get("evidence_id", ""),
            )
        except Exception:
            pass

    def advance_phase(self, job: LearningJob, next_phase: JobPhase) -> None:
        old_phase = job.phase
        job.phase = next_phase
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        job.events.append({
            "ts": _utc_iso(), "type": "phase_changed",
            "msg": f"{old_phase} -> {next_phase}",
        })
        self.store.save(job)
        try:
            from consciousness.events import event_bus, SKILL_JOB_PHASE_CHANGED
            event_bus.emit(
                SKILL_JOB_PHASE_CHANGED,
                job_id=job.job_id, skill_id=job.skill_id,
                old_phase=old_phase, new_phase=next_phase,
            )
        except Exception:
            pass
        try:
            from consciousness.attribution_ledger import attribution_ledger
            attribution_ledger.record(
                subsystem="learning_jobs",
                event_type="learning_job_phase_changed",
                data={"skill_id": job.skill_id, "old_phase": old_phase, "new_phase": next_phase},
                evidence_refs=[{"kind": "job", "id": job.job_id}, {"kind": "skill", "id": job.skill_id}],
                parent_entry_id=job.ledger_entry_id,
            )
        except Exception:
            pass
        logger.info("Job %s phase: %s -> %s", job.job_id, old_phase, next_phase)

    def complete_job(self, job: LearningJob) -> None:
        report = self._build_completion_report(job)

        if job.matrix_protocol:
            if report.get("checks_failed"):
                job.claimability_status = "unverified"
            elif job.capability_type == "perceptual":
                job.claimability_status = "verified_operational"
            else:
                job.claimability_status = "verified_limited"

        job.status = "completed"
        summary = report.get("summary_text", "Learning job completed.")
        job.events.append({"ts": _utc_iso(), "type": "job_completed", "msg": summary})

        is_matrix = bool(job.matrix_protocol)
        report_filename = "matrix_report.json" if is_matrix else "skill_learning_report.json"
        report_id = "matrix_report" if is_matrix else "skill_learning_report"
        report_type = "matrix_report" if is_matrix else "skill_learning_report"
        report_path = os.path.join(
            os.path.expanduser("~/.jarvis/learning_jobs"), job.job_id, report_filename,
        )
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            job.artifacts.append({
                "id": report_id, "type": report_type, "path": report_path,
            })
        except Exception:
            pass

        self.store.save(job)
        self._active_jobs.pop(job.job_id, None)

        try:
            from consciousness.events import event_bus, SKILL_LEARNING_COMPLETED
            event_bus.emit(
                SKILL_LEARNING_COMPLETED,
                job_id=job.job_id, skill_id=job.skill_id,
                report=report,
            )
        except Exception:
            pass
        try:
            from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
            import time as _time
            attribution_ledger.record(
                subsystem="learning_jobs",
                event_type="learning_job_completed",
                data={"skill_id": job.skill_id},
                evidence_refs=[{"kind": "job", "id": job.job_id}, {"kind": "skill", "id": job.skill_id}],
                parent_entry_id=job.ledger_entry_id,
            )
            if job.ledger_entry_id:
                _created = job.created_at
                try:
                    from datetime import datetime, timezone
                    _ct = datetime.fromisoformat(_created.replace("Z", "+00:00")).timestamp()
                except Exception:
                    _ct = _time.time()
                attribution_ledger.record_outcome(job.ledger_entry_id, "success", build_outcome_data(
                    confidence=0.9,
                    latency_s=round(_time.time() - _ct, 2),
                    source="verification_test",
                    tier="medium",
                    scope="skill_verification",
                    blame_target="general",
                    skill_id=job.skill_id,
                ))
        except Exception:
            pass
        logger.info("Learning job %s completed (claimability=%s)", job.job_id, job.claimability_status)

        self._supersede_stale_siblings(job)

    def _supersede_stale_siblings(self, completed_job: LearningJob) -> None:
        """Auto-cancel blocked/stale jobs for the same skill after one completes."""
        import re as _re
        canonical = _re.sub(r'_v\d+$', '', completed_job.skill_id)
        to_cancel: list[str] = []

        for other_id, other in list(self._active_jobs.items()):
            if other.job_id == completed_job.job_id:
                continue
            other_canonical = _re.sub(r'_v\d+$', '', other.skill_id)
            if other_canonical != canonical:
                continue
            if other.status in ("blocked", "failed"):
                to_cancel.append(other_id)

        for other in self.store.load_all():
            if other.job_id == completed_job.job_id:
                continue
            if other.job_id in [c for c in to_cancel]:
                continue
            other_canonical = _re.sub(r'_v\d+$', '', other.skill_id)
            if other_canonical != canonical:
                continue
            if other.status in ("blocked", "failed"):
                to_cancel.append(other.job_id)

        for jid in to_cancel:
            self.delete_job(jid, remove_skill=False)
            logger.info(
                "Superseded stale job %s (skill=%s) — completed job %s takes precedence",
                jid, completed_job.skill_id, completed_job.job_id,
            )

    @staticmethod
    def _build_completion_report(job: LearningJob) -> dict:
        """Build a 5Ws completion report from a finishing learning job."""
        from datetime import datetime, timezone

        # Who requested it and when
        req = job.requested_by or {}
        who = req.get("speaker", "system")
        trigger = req.get("user_text", "")
        created = job.created_at

        # Timeline: compute duration
        try:
            t0 = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            t1 = time.time()
            duration_s = round(t1 - t0)
        except Exception:
            duration_s = 0

        phase_timeline = []
        for ev in job.events:
            if ev.get("type") == "phase_changed":
                phase_timeline.append({"ts": ev.get("ts", ""), "transition": ev.get("msg", "")})

        # What evidence was collected
        checks_passed = []
        checks_failed = []
        evidence_latest = job.evidence.get("latest") or {}
        for t in evidence_latest.get("tests", []):
            if t.get("passed"):
                checks_passed.append(t.get("name", "unknown"))
            else:
                checks_failed.append(t.get("name", "unknown"))
        verification_details = evidence_latest.get("details", "") or ""
        for t in evidence_latest.get("tests", []):
            if t.get("details"):
                verification_details = t["details"]
                break

        # What artifacts were produced
        artifact_summary = []
        research_content = None
        for a in job.artifacts:
            entry = {"type": a.get("type", "unknown"), "id": a.get("id", "")}
            if a.get("type") == "research_summary" and a.get("path"):
                try:
                    with open(a["path"], "r") as f:
                        research_content = json.load(f)
                    entry["content_preview"] = research_content.get("approach", "")[:200]
                except Exception:
                    pass
            artifact_summary.append(entry)

        # What gates were evaluated
        gate_summary = []
        for g in job.gates.get("hard", []):
            gate_summary.append({
                "id": g.get("id", ""),
                "state": g.get("state", "unknown"),
                "details": g.get("details", ""),
            })

        # Failure context
        fail = job.failure or {}
        failure_count = fail.get("count", 0)

        # Build human-readable summary
        skill_name = job.skill_id.replace("_v1", "").replace("_", " ").title()
        proto = job.protocol_id or "unknown"
        n_phases = len(phase_timeline)
        duration_str = f"{duration_s // 60}m {duration_s % 60}s" if duration_s else "unknown"
        claimability = job.claimability_status

        if checks_failed:
            status_line = f"completed with issues ({len(checks_failed)} check(s) failed)"
        elif claimability in ("verified_limited", "verified_operational"):
            status_line = f"verified as {claimability.replace('_', ' ')}"
        else:
            status_line = "completed"

        if job.matrix_protocol:
            summary_text = (
                f"Matrix Protocol learning report for {skill_name}: {status_line}. "
                f"Protocol {proto}, {job.capability_type} class. "
                f"Completed {n_phases} phase transitions in {duration_str}. "
                f"{len(checks_passed)} verification checks passed"
                + (f", {len(checks_failed)} failed" if checks_failed else "")
                + "."
            )
            report_kind = "matrix_protocol"
        else:
            summary_text = (
                f"Skill learning report for {skill_name}: {status_line}. "
                f"{job.capability_type.title()} skill class. "
                f"Completed {n_phases} phase transitions in {duration_str}. "
                f"{len(checks_passed)} verification checks passed"
                + (f", {len(checks_failed)} failed" if checks_failed else "")
                + "."
            )
            report_kind = "skill_learning"

        return {
            "job_id": job.job_id,
            "skill_id": job.skill_id,
            "skill_name": skill_name,
            "protocol_id": proto,
            "capability_type": job.capability_type,
            "matrix_protocol": job.matrix_protocol,
            "report_kind": report_kind,

            "requested_by": who,
            "trigger_text": trigger,
            "created_at": created,
            "completed_at": _utc_iso(),
            "duration_s": duration_s,
            "duration_human": duration_str,

            "phase_timeline": phase_timeline,
            "final_phase": job.phase,
            "status": job.status,
            "claimability": claimability,
            "promotion_status": job.promotion_status,
            "specialist_born": job.specialist_focus is not None,

            "gates": gate_summary,
            "artifacts": artifact_summary,
            "research_content": research_content,

            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "verification_details": verification_details,
            "failure_count": failure_count,

            "summary_text": summary_text,
        }

    def fail_job(self, job: LearningJob, error: str) -> None:
        job.failure["count"] = int(job.failure.get("count", 0)) + 1
        job.failure["last_error"] = error
        job.failure["last_failed_phase"] = job.phase
        job.status = "failed"
        job.events.append({"ts": _utc_iso(), "type": "job_failed", "msg": error})
        self.store.save(job)
        self._active_jobs.pop(job.job_id, None)
        self._propagate_blocked(job, f"failed: {error}")
        try:
            from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
            import time as _time
            attribution_ledger.record(
                subsystem="learning_jobs",
                event_type="learning_job_failed",
                data={"skill_id": job.skill_id, "phase": job.phase, "error": error[:200]},
                evidence_refs=[{"kind": "job", "id": job.job_id}, {"kind": "skill", "id": job.skill_id}],
                parent_entry_id=job.ledger_entry_id,
            )
            if job.ledger_entry_id:
                _created = job.created_at
                try:
                    from datetime import datetime, timezone
                    _ct = datetime.fromisoformat(_created.replace("Z", "+00:00")).timestamp()
                except Exception:
                    _ct = _time.time()
                attribution_ledger.record_outcome(job.ledger_entry_id, "failure", build_outcome_data(
                    confidence=0.85,
                    latency_s=round(_time.time() - _ct, 2),
                    source="verification_test",
                    tier="medium",
                    scope="skill_verification",
                    blame_target="general",
                    skill_id=job.skill_id,
                    phase=job.phase,
                    error=error[:200],
                ))
        except Exception:
            pass
        logger.warning("Learning job %s failed: %s", job.job_id, error)

    def pause_job(self, job_id: str) -> bool:
        """Pause an active job. Returns True if paused."""
        job = self._active_jobs.get(job_id)
        if not job or job.status != "active":
            return False
        job.status = "paused"
        job.events.append({"ts": _utc_iso(), "type": "job_paused", "msg": "Manually paused"})
        self.store.save(job)
        logger.info("Learning job %s paused", job_id)
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job. Returns True if resumed."""
        job = self._active_jobs.get(job_id)
        if not job:
            loaded = self.store.load(job_id)
            if loaded and loaded.status == "paused":
                loaded.status = "active"
                loaded.updated_at = _utc_iso()
                loaded.events.append({"ts": _utc_iso(), "type": "job_resumed", "msg": "Manually resumed"})
                self.store.save(loaded)
                self._active_jobs[job_id] = loaded
                logger.info("Learning job %s resumed (reloaded)", job_id)
                return True
            return False
        if job.status != "paused":
            return False
        job.status = "active"
        job.updated_at = _utc_iso()
        job.events.append({"ts": _utc_iso(), "type": "job_resumed", "msg": "Manually resumed"})
        self.store.save(job)
        logger.info("Learning job %s resumed", job_id)
        return True

    def approve_operational_handoff(
        self,
        job_id: str,
        acquisition_orchestrator: Any,
        *,
        approved_by: str = "human",
        notes: str = "",
    ) -> dict[str, Any]:
        """Approve a pending operational handoff and start acquisition."""

        job = self._active_jobs.get(job_id) or self.store.load(job_id)
        if job is None:
            return {"ok": False, "reason": "job_not_found"}

        handoff = (job.data or {}).get("operational_handoff", {})
        linked_acquisition_id = (
            getattr(job, "parent_acquisition_id", "")
            or (handoff.get("acquisition_id", "") if isinstance(handoff, dict) else "")
        )
        if linked_acquisition_id:
            return {
                "ok": False,
                "reason": "handoff_already_linked_to_acquisition",
                "acquisition_id": linked_acquisition_id,
                "message": "Reject or cancel the linked acquisition instead of rejecting the skill handoff.",
            }
        if not isinstance(handoff, dict) or handoff.get("status") != "awaiting_operator_approval":
            return {"ok": False, "reason": "handoff_not_awaiting_operator_approval"}

        try:
            from skills.execution_contracts import get_contract
            from skills.operational_bridge import start_operational_handoff
            contract = get_contract(job.skill_id)
            if contract is None:
                return {"ok": False, "reason": "contract_not_found"}
            ok, detail = start_operational_handoff(
                job,
                contract,
                acquisition_orchestrator,
                approved_by=approved_by or "human",
                notes=notes,
            )
        except Exception as exc:
            logger.warning("Operational handoff approval failed for job %s: %s", job_id, exc)
            handoff.update({
                "status": "approval_failed",
                "last_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "last_error_type": type(exc).__name__,
                "updated_at": _utc_iso(),
            })
            self.store.save(job)
            return {"ok": False, "reason": "approval_failed", "error": str(exc)}

        job.events.append({
            "ts": _utc_iso(),
            "type": "operational_handoff_approved" if ok else "operational_handoff_failed",
            "msg": detail,
        })
        self.store.save(job)
        self._active_jobs[job.job_id] = job
        if ok and self._registry:
            self._registry.set_status(job.skill_id, "learning")
        return {"ok": ok, "acquisition_id": detail if ok else "", "reason": "" if ok else detail}

    def retry_operational_handoff(
        self,
        job_id: str,
        acquisition_orchestrator: Any,
        *,
        approved_by: str = "human",
        notes: str = "",
    ) -> dict[str, Any]:
        """Retry a failed/cancelled operational handoff with a fresh acquisition."""

        job = self._active_jobs.get(job_id) or self.store.load(job_id)
        if job is None:
            return {"ok": False, "reason": "job_not_found"}

        handoff = (job.data or {}).get("operational_handoff", {})
        if not isinstance(handoff, dict) or handoff.get("status") not in ("acquisition_failed", "acquisition_cancelled"):
            return {"ok": False, "reason": "handoff_not_retryable"}

        previous_acquisition_id = (
            getattr(job, "parent_acquisition_id", "")
            or handoff.get("acquisition_id", "")
        )
        handoff.setdefault("previous_acquisition_ids", [])
        if previous_acquisition_id and previous_acquisition_id not in handoff["previous_acquisition_ids"]:
            handoff["previous_acquisition_ids"].append(previous_acquisition_id)
        handoff.update({
            "status": "awaiting_operator_approval",
            "approval_required": True,
            "acquisition_id": "",
            "retry_requested_by": approved_by or "human",
            "retry_notes": notes,
            "retry_requested_at": _utc_iso(),
            "last_error": "",
            "last_error_type": "",
            "updated_at": _utc_iso(),
        })
        job.parent_acquisition_id = ""
        job.status = "awaiting_operator_approval"
        job.failure = {"count": 0, "last_error": None, "last_failed_phase": None}
        job.updated_at = _utc_iso()
        job.events.append({
            "ts": _utc_iso(),
            "type": "operational_handoff_retry",
            "msg": previous_acquisition_id or "retry",
        })
        self.store.save(job)
        self._active_jobs[job.job_id] = job
        if self._registry:
            self._registry.set_status(job.skill_id, "learning")

        return self.approve_operational_handoff(
            job_id,
            acquisition_orchestrator,
            approved_by=approved_by,
            notes=notes or "Operator retried operational proof build.",
        )

    def reject_operational_handoff(
        self,
        job_id: str,
        *,
        rejected_by: str = "human",
        reason: str = "",
    ) -> dict[str, Any]:
        """Reject a pending operational build without treating it as technical failure."""

        job = self._active_jobs.get(job_id) or self.store.load(job_id)
        if job is None:
            return {"ok": False, "reason": "job_not_found"}

        handoff = (job.data or {}).get("operational_handoff", {})
        linked_acquisition_id = (
            getattr(job, "parent_acquisition_id", "")
            or (handoff.get("acquisition_id", "") if isinstance(handoff, dict) else "")
        )
        if linked_acquisition_id:
            return {
                "ok": False,
                "reason": "handoff_already_linked_to_acquisition",
                "acquisition_id": linked_acquisition_id,
                "message": "Reject or cancel the linked acquisition instead of rejecting the skill handoff.",
            }
        if not isinstance(handoff, dict) or handoff.get("status") != "awaiting_operator_approval":
            return {"ok": False, "reason": "handoff_not_awaiting_operator_approval"}

        msg = reason or "operator_rejected_operational_build"
        handoff.update({
            "status": "operator_rejected_operational_build",
            "rejected_by": rejected_by or "human",
            "rejection_reason": msg,
            "rejected_at": _utc_iso(),
            "updated_at": _utc_iso(),
        })
        job.status = "blocked"
        job.failure = {
            "count": 0,
            "last_error": "operator_rejected_operational_build",
            "last_failed_phase": job.phase,
        }
        job.updated_at = _utc_iso()
        job.events.append({
            "ts": _utc_iso(),
            "type": "operational_handoff_rejected",
            "msg": msg,
        })
        self.store.save(job)
        self._active_jobs.pop(job.job_id, None)
        self._propagate_blocked(job, f"operator rejected operational build: {msg[:120]}")
        return {"ok": True, "reason": "operator_rejected_operational_build"}

    def _propagate_blocked(self, job: LearningJob, reason: str) -> None:
        """Propagate blocked/failed status back to the SkillRecord."""
        try:
            if self._registry:
                self._registry.set_status(job.skill_id, "blocked")
                logger.info("Propagated blocked status to skill '%s': %s", job.skill_id, reason[:60])
        except Exception:
            logger.debug("Failed to propagate blocked status for skill '%s'", job.skill_id, exc_info=True)

    def delete_job(self, job_id: str, remove_skill: bool = False) -> bool:
        """Delete a job from disk + memory with safe cleanup.

        - If the job is ``blocked`` or ``completed`` and a *different* completed
          job already exists for the same skill, the skill record's
          ``learning_job_id`` is cleared/repointed to the surviving completed
          job so the registry stays consistent.
        - Pass ``remove_skill=True`` only if you also want to remove the
          associated SkillRecord (usually only for junk/auto-gate jobs).
        - Active (non-terminal) jobs are refused unless ``remove_skill`` is
          True, to prevent accidental mid-flight deletion.
        """
        job = self._active_jobs.get(job_id)
        if not job:
            job = self.store.load(job_id)

        if job is None:
            return False

        if job.status in ("active", "running", "in_progress", "awaiting_operator_approval") and not remove_skill:
            logger.warning(
                "Refusing to delete in-flight job %s (status=%s). "
                "Pass remove_skill=True to force.", job_id, job.status,
            )
            return False

        skill_id = job.skill_id

        self._active_jobs.pop(job_id, None)
        if not self.store.delete(job_id):
            logger.warning("store.delete failed for %s", job_id)

        if remove_skill and skill_id and self._registry:
            self._registry.remove(skill_id)
        elif skill_id and self._registry:
            self._reconcile_skill_after_delete(skill_id, job_id)

        logger.info(
            "Deleted learning job %s (skill=%s, remove_skill=%s)",
            job_id, skill_id, remove_skill,
        )
        return True

    def _reconcile_skill_after_delete(self, skill_id: str, deleted_job_id: str) -> None:
        """Fix up the skill record after deleting one of its jobs.

        If the skill's ``learning_job_id`` pointed at the deleted job, repoint
        it to a surviving completed job for the same skill (if any) or clear it.
        Also ensure the skill status is consistent: if a completed job exists
        the skill should be ``verified``, not ``blocked``/``learning``.
        """
        try:
            rec = self._registry.get(skill_id)
            if rec is None:
                return

            if rec.learning_job_id != deleted_job_id:
                return

            import re as _re
            canonical = _re.sub(r'_v\d+$', '', skill_id)
            surviving_completed: LearningJob | None = None
            for other in self.store.load_all():
                if other.job_id == deleted_job_id:
                    continue
                other_canonical = _re.sub(r'_v\d+$', '', other.skill_id)
                if other_canonical == canonical and other.status == "completed":
                    surviving_completed = other
                    break

            if surviving_completed:
                rec.learning_job_id = surviving_completed.job_id
                if rec.status in ("blocked", "learning"):
                    rec.status = "verified"
                    rec.updated_at = time.time()
            else:
                rec.learning_job_id = None

            self._registry.save()
            logger.info(
                "Reconciled skill '%s' after deleting job %s → learning_job_id=%s",
                skill_id, deleted_job_id, rec.learning_job_id,
            )
        except Exception:
            logger.debug("Skill reconciliation failed for '%s'", skill_id, exc_info=True)

    def cleanup_blocked_jobs(self, max_age_s: float = 3600.0) -> int:
        """Delete all blocked jobs older than max_age_s. Returns count deleted.

        Junk jobs (verify-blocked with no verification method) also have
        their orphaned SkillRecord removed from the registry.
        """
        _JUNK_ERRORS = ("No verifiable artifact found", "no_verification_method")
        now = time.time()
        to_delete: list[tuple[str, bool]] = []
        for job in self.store.load_all():
            if job.status != "blocked":
                continue
            try:
                import datetime as _dt
                updated_ts = _dt.datetime.fromisoformat(
                    job.updated_at.rstrip("Z")
                ).replace(tzinfo=_dt.timezone.utc).timestamp()
                age = now - updated_ts
            except Exception:
                age = float("inf")
            if age > max_age_s:
                fail = getattr(job, "failure", None) or {}
                is_junk = any(e in fail.get("last_error", "") for e in _JUNK_ERRORS)
                to_delete.append((job.job_id, is_junk))

        count = 0
        for jid, remove_skill in to_delete:
            if self.delete_job(jid, remove_skill=remove_skill):
                count += 1
        if count:
            logger.info("Cleaned up %d blocked learning jobs (max_age=%.0fs)", count, max_age_s)
        return count

    def recover_blocked_job(self, job_id: str) -> bool:
        """Reset a blocked job to active/assess phase so it can retry."""
        job = self.store.load(job_id)
        if not job:
            logger.warning("recover_blocked_job: job %s not found on disk", job_id)
            return False
        if job.status not in ("blocked", "failed"):
            logger.info("recover_blocked_job: job %s is '%s', not blocked/failed — skipping", job_id, job.status)
            return False
        job.status = "active"
        job.phase = "assess"
        job.failure = {"count": 0}
        job.updated_at = _utc_iso()
        job.events.append({"ts": _utc_iso(), "type": "job_recovered", "msg": "Reset to assess phase for retry"})
        self.store.save(job)
        self._active_jobs[job.job_id] = job
        if self._registry:
            self._registry.set_status(job.skill_id, "learning")
        logger.info("Recovered blocked job %s (skill=%s) — reset to assess", job_id, job.skill_id)
        return True

    def get_job_detail(self, job_id: str) -> dict[str, Any] | None:
        """Return full job state for detail API. Checks memory first, then disk."""
        job = self._active_jobs.get(job_id)
        if not job:
            job = self.store.load(job_id)
        if not job:
            return None
        return job.to_dict()

    def get_active_jobs(self) -> list[LearningJob]:
        return list(self._active_jobs.values())

    def run_cleanup(self) -> dict[str, int]:
        """Public entry point for on-demand junk job cleanup.

        Returns counts of cleaned/purged jobs.
        """
        before_active = len(self._active_jobs)
        self._cleanup_actuator_junk_jobs()
        after_active = len(self._active_jobs)
        return {"cleaned": before_active - after_active}

    def get_status(self) -> dict[str, Any]:
        """Dashboard-friendly snapshot."""
        try:
            from skills.registry import get_default_skill_ids
            _defaults = get_default_skill_ids()
        except Exception:
            _defaults = frozenset()
        now = time.time()
        active = []
        for job in self._active_jobs.values():
            try:
                import datetime as _dt
                updated_ts = _dt.datetime.fromisoformat(job.updated_at.rstrip("Z")).replace(
                    tzinfo=_dt.timezone.utc).timestamp()
                phase_age = now - updated_ts
            except Exception:
                phase_age = 0
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "skill_id": job.skill_id,
                "status": job.status,
                "phase": job.phase,
                "capability_type": job.capability_type,
                "priority": job.priority,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "phase_age_s": round(phase_age, 1),
                "stale": phase_age > PHASE_TIMEOUT_S,
                "artifact_count": len(job.artifacts),
                "evidence_count": len(job.evidence.get("history", [])),
                "event_count": len(job.events),
                "is_default_skill": job.skill_id in _defaults or
                    any(job.skill_id.startswith(d) for d in _defaults),
            }
            if job.matrix_protocol:
                entry["matrix_protocol"] = True
                entry["protocol_id"] = job.protocol_id
                entry["claimability_status"] = job.claimability_status
                entry["promotion_status"] = job.promotion_status
            active.append(entry)
        all_jobs = self.store.load_all()
        return {
            "active_count": len(self._active_jobs),
            "total_count": len(all_jobs),
            "active_jobs": active,
            "completed_count": sum(1 for j in all_jobs if j.status == "completed"),
            "failed_count": sum(1 for j in all_jobs if j.status == "failed"),
        }

    def run_cycle(self, ctx: dict[str, Any] | None = None) -> None:
        """Called from the consciousness tick loop. Ticks all active jobs."""
        if not self._active_jobs:
            return
        ctx = self._build_cycle_context(ctx)
        for job in list(self._active_jobs.values()):
            try:
                self._tick_job(job, ctx)
            except Exception:
                logger.exception("Learning job tick failed for %s", job.job_id)

    def _tick_job(self, job: LearningJob, ctx: dict[str, Any]) -> None:
        """Run executor for the current phase, then try auto-advance."""
        if job.status == "awaiting_operator_approval":
            return

        if self._sync_terminal_acquisition_if_needed(job, ctx):
            return

        if self._is_terminal_unverifiable_job(job):
            self.delete_job(job.job_id, remove_skill=True)
            logger.warning("Purged terminal unverifiable learning job %s", job.job_id)
            return

        fail = job.failure
        if fail.get("count", 0) >= MAX_PHASE_FAILURES and fail.get("last_failed_phase") == job.phase:
            if job.status != "blocked":
                job.status = "blocked"
                reason = f"Phase {job.phase} failed {fail['count']} times — giving up."
                job.events.append({"ts": _utc_iso(), "type": "job_blocked", "msg": reason})
                self.store.save(job)
                self._active_jobs.pop(job.job_id, None)
                self._propagate_blocked(job, reason)
                logger.warning("Job %s blocked: %d failures in %s", job.job_id, fail["count"], job.phase)
            return

        # Phase timeout: if phase hasn't progressed in PHASE_TIMEOUT_S, mark blocked
        now = time.time()
        try:
            import datetime as _dt
            updated_ts = _dt.datetime.fromisoformat(job.updated_at.rstrip("Z")).replace(
                tzinfo=_dt.timezone.utc).timestamp()
            phase_age = now - updated_ts
        except Exception:
            phase_age = 0
        if phase_age > PHASE_TIMEOUT_S and job.status == "active":
            reason = f"phase_timeout: {job.phase} stale for {phase_age:.0f}s"
            job.status = "blocked"
            job.events.append({"ts": _utc_iso(), "type": "job_blocked", "msg": reason})
            self.store.save(job)
            self._active_jobs.pop(job.job_id, None)
            self._propagate_blocked(job, reason)
            logger.warning("Job %s blocked: phase timeout in %s (%.0fs)", job.job_id, job.phase, phase_age)
            return

        last_tick = job.executor_state.get("_last_tick_ts", 0.0)
        now = time.time()
        if now - last_tick < TICK_COOLDOWN_S:
            return
        job.executor_state["_last_tick_ts"] = now

        if self._dispatcher is not None and self._registry:
            try:
                self._dispatcher.tick_one_job(job, ctx, self, self._registry)
                if job.status == "blocked":
                    reason = str((job.failure or {}).get("last_error") or "learning_job_blocked")
                    self.store.save(job)
                    self._active_jobs.pop(job.job_id, None)
                    self._propagate_blocked(job, reason)
                return
            except Exception:
                logger.exception("Executor dispatch failed for job %s", job.job_id)
        try:
            from skills.job_runner import try_auto_advance
            if self._registry:
                advanced = try_auto_advance(job, self._registry, self)
                if advanced:
                    logger.debug("Job %s auto-advanced to %s", job.job_id, job.phase)
        except Exception:
            logger.exception("Auto-advance failed for job %s", job.job_id)

    def _sync_terminal_acquisition_if_needed(self, job: LearningJob, ctx: dict[str, Any]) -> bool:
        """Close a learning job promptly when its acquisition proof lane is terminal."""
        handoff = (job.data or {}).get("operational_handoff", {})
        linked_acquisition_id = (
            getattr(job, "parent_acquisition_id", "")
            or (handoff.get("acquisition_id", "") if isinstance(handoff, dict) else "")
        )
        if not linked_acquisition_id or not ctx.get("acquisition_orchestrator"):
            return False
        try:
            from skills.operational_bridge import (
                ACQUISITION_CANCELLED,
                ACQUISITION_FAILED,
                sync_acquisition_proof,
            )
            sync_acquisition_proof(job, ctx)
            handoff_after = (job.data or {}).get("operational_handoff", {})
            if (
                job.status == "blocked"
                and isinstance(handoff_after, dict)
                and handoff_after.get("status") in {ACQUISITION_FAILED, ACQUISITION_CANCELLED}
            ):
                reason = str((job.failure or {}).get("last_error") or handoff_after.get("status"))
                self.store.save(job)
                self._active_jobs.pop(job.job_id, None)
                self._propagate_blocked(job, reason)
                return True
        except Exception:
            logger.debug("Terminal acquisition sync failed for %s", job.job_id, exc_info=True)
        return False
