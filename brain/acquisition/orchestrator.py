"""Acquisition Orchestrator — thin coordination spine for capability acquisition.

Foundational rule: the acquisition layer owns intent-to-lane coordination.
Each lane remains authoritative for its own execution semantics and local truth.

Sacred guardrails:
  1. No lane-local execution logic in acquisition.
  2. No lane-local truth storage in acquisition.
  3. No bypass of child safety gates.
  4. No child lifecycle compression into acquisition summaries.
  5. No plugin activation without lane-native proof refs.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from consciousness.events import event_bus

from acquisition.job import (
    AcquisitionStore,
    AcquisitionPlan,
    CapabilityAcquisitionJob,
    DocumentationArtifact,
    PluginCodeBundle,
    ResearchArtifact,
    _artifact_id,
)
from acquisition.classifier import IntentClassifier, ClassificationResult, _LANE_MAP, _RISK_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lane execution policy — classifies lanes by operational weight so the
# orchestrator can gate heavy/risky work based on cognitive mode.
# ---------------------------------------------------------------------------

_BACKGROUND_SAFE_LANES = frozenset({
    "evidence_grounding",
    "doc_resolution",
    "planning",
    "plan_review",
    "truth",
    "skill_registration",
})

_CONDITIONALLY_ALLOWED_LANES = frozenset({
    "implementation",
    "environment_setup",
    "plugin_quarantine",
    "verification",
})

_HIGH_RISK_LANES = frozenset({
    "plugin_activation",
    "deployment",
    "self_improve",
    "matrix_specialist",
})

_HEAVY_LANE_ALLOWED_MODES = frozenset({
    "passive", "conversational", "reflective", "focused", "deep_learning",
})

_CONDITIONAL_LANE_ALLOWED_MODES = frozenset({
    "passive", "conversational", "reflective", "focused", "deep_learning",
    "dreaming",
})

_BACKGROUND_WORKER_LANES = frozenset({
    "planning",
    "implementation",
    "environment_setup",
    "verification",
})

_SKILL_ACQUISITION_PROMPT_VERSION = "skill_acquisition_prompt_v1"


class AcquisitionOrchestrator:
    """Coordinates capability acquisition jobs across subsystems.

    Responsibilities:
      - Create acquisition jobs from user intent
      - Classify intent into outcome classes
      - Initialize required lanes
      - Tick active jobs forward (lane dispatch)
      - Persist state
      - Provide status for dashboard/API

    NOT responsible for:
      - Lane-internal execution logic
      - Lane-internal truth storage
      - Overriding child safety gates
    """

    # Quarantine pressure thresholds
    PRESSURE_NORMAL = 0.3      # < 0.3: all lanes execute normally
    PRESSURE_ELEVATED = 0.6    # 0.3-0.6: high-risk deferred, shadow thresholds doubled
    # > 0.6: only background-safe lanes, plugin promotion blocked

    def __init__(self, store: AcquisitionStore | None = None) -> None:
        self._store = store or AcquisitionStore()
        self._classifier = IntentClassifier()
        self._active_jobs: dict[str, CapabilityAcquisitionJob] = {}
        self._total_created: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0
        self._current_mode: str = "passive"
        self._quarantine_pressure: float = 0.0
        self._pressure_level: str = "normal"
        self._suppressed_lanes: set[str] = set()
        self._codegen_service: Any = None
        self._lane_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="acquisition-lane",
        )
        self._lane_futures: dict[tuple[str, str], concurrent.futures.Future] = {}

        self._load_active_jobs()

    def set_codegen_service(self, service: Any) -> None:
        """Wire the CodeGenService for plan enrichment and implementation."""
        self._codegen_service = service
        logger.info("AcquisitionOrchestrator: CodeGenService wired (available=%s)",
                     service.coder_available if service else False)

    # ── boot ───────────────────────────────────────────────────────────

    def _load_active_jobs(self) -> None:
        """Restore in-flight jobs from disk on boot."""
        try:
            for status in ("pending", "classifying", "planning", "awaiting_plan_review",
                           "executing", "verifying", "awaiting_approval", "deployed"):
                for job in self._store.list_jobs(status=status):
                    self._active_jobs[job.acquisition_id] = job
            if self._active_jobs:
                logger.info("AcquisitionOrchestrator: restored %d active jobs", len(self._active_jobs))
        except Exception as exc:
            logger.warning("AcquisitionOrchestrator: failed to restore jobs: %s", exc)

    # ── create ─────────────────────────────────────────────────────────

    def create(
        self,
        user_text: str,
        requested_by: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> CapabilityAcquisitionJob:
        """Create a new acquisition job from user intent."""
        job = CapabilityAcquisitionJob(
            title=user_text[:120],
            user_intent=user_text,
            requested_by=requested_by or {},
        )

        # Classify
        result = self._classifier.classify(user_text, context)
        job.outcome_class = result.outcome_class
        job.classification_confidence = result.confidence
        job.classified_at = result.classified_at
        job.required_lanes = result.required_lanes
        job.risk_tier = result.risk_tier
        job.set_status("classifying")

        # Set permission model based on risk tier
        self._apply_risk_tier(job)

        # Initialize lanes
        for lane_name in result.required_lanes:
            job.init_lane(lane_name)

        # Record in attribution ledger
        self._record_ledger_entry(job, "acquisition:created")

        job.set_status("planning" if "planning" in job.required_lanes else "executing")

        # Persist + track
        self._store.save_job(job)
        self._active_jobs[job.acquisition_id] = job
        self._total_created += 1

        # Emit event
        try:
            from consciousness.events import ACQUISITION_CREATED
            event_bus.emit(ACQUISITION_CREATED, {
                "acquisition_id": job.acquisition_id,
                "title": job.title,
                "outcome_class": job.outcome_class,
                "risk_tier": job.risk_tier,
            })
        except Exception:
            pass

        logger.info(
            "Acquisition created: %s [%s] tier=%d lanes=%s",
            job.acquisition_id, job.outcome_class, job.risk_tier,
            ",".join(job.required_lanes),
        )
        return job

    def create_skill_proof_handoff(
        self,
        learning_job: Any,
        contract: Any,
        handoff: dict[str, Any],
    ) -> CapabilityAcquisitionJob:
        """Create or return a governed plugin/callable proof job for a skill.

        This bypasses heuristic intent classification intentionally: the caller
        already has a SkillExecutionContract proving that an operational
        executor is required. The acquisition lane still owns planning, codegen,
        quarantine, verification, approval, activation, and truth recording.
        """

        learning_job_id = getattr(learning_job, "job_id", "")
        skill_id = getattr(learning_job, "skill_id", "")

        for existing in self._store.list_jobs():
            requested = existing.requested_by or {}
            if (
                requested.get("source") == "skill_operational_handoff"
                and requested.get("learning_job_id") == learning_job_id
                and existing.status not in ("failed", "cancelled")
            ):
                if existing.acquisition_id not in self._active_jobs and existing.status not in ("completed",):
                    self._active_jobs[existing.acquisition_id] = existing
                return existing

        title = f"Build operational proof plugin for {skill_id}"
        user_intent = self._build_skill_proof_intent(skill_id, contract, handoff)
        job = CapabilityAcquisitionJob(
            title=title,
            user_intent=user_intent,
            requested_by={
                "source": "skill_operational_handoff",
                "skill_id": skill_id,
                "learning_job_id": learning_job_id,
                "contract_id": getattr(contract, "contract_id", ""),
                "required_executor_kind": getattr(contract, "required_executor_kind", ""),
            },
        )
        job.outcome_class = "plugin_creation"
        job.classification_confidence = 1.0
        job.classified_at = time.time()
        job.required_lanes = list(_LANE_MAP["plugin_creation"])
        job.risk_tier = _RISK_MAP["plugin_creation"]
        job.learning_job_id = learning_job_id

        self._apply_risk_tier(job)
        for lane_name in job.required_lanes:
            job.init_lane(lane_name)

        self._record_ledger_entry(job, "acquisition:skill_proof_handoff_created")
        job.set_status("planning")
        self._store.save_job(job)
        self._active_jobs[job.acquisition_id] = job
        self._total_created += 1

        try:
            from consciousness.events import ACQUISITION_CREATED
            event_bus.emit(ACQUISITION_CREATED, {
                "acquisition_id": job.acquisition_id,
                "title": job.title,
                "outcome_class": job.outcome_class,
                "risk_tier": job.risk_tier,
                "skill_id": skill_id,
                "learning_job_id": learning_job_id,
            })
        except Exception:
            pass

        logger.info(
            "Skill proof handoff created: %s skill=%s learning_job=%s contract=%s",
            job.acquisition_id, skill_id, learning_job_id, getattr(contract, "contract_id", ""),
        )
        return job

    @staticmethod
    def _build_skill_proof_intent(skill_id: str, contract: Any, handoff: dict[str, Any]) -> str:
        fixtures = handoff.get("smoke_fixtures", []) or []
        fixture_text = json.dumps(fixtures, ensure_ascii=True)
        return (
            f"Build a Jarvis plugin that provides operational proof for skill '{skill_id}'. "
            f"The plugin must satisfy contract '{getattr(contract, 'contract_id', '')}' "
            f"with executor kind '{getattr(contract, 'required_executor_kind', '')}'. "
            "It must accept the fixture input as user text and return structured JSON/dict output "
            "matching the expected fields. "
            f"Smoke test name: {getattr(contract, 'smoke_test_name', '')}. "
            f"Fixtures: {fixture_text[:2000]}"
        )

    # ── tick (called from consciousness kernel) ────────────────────────

    def _sample_quarantine_pressure(self) -> None:
        """Read current quarantine pressure and compute suppression policy."""
        try:
            from epistemic.quarantine.pressure import QuarantinePressure
            pressure = QuarantinePressure.instance()
            if pressure:
                self._quarantine_pressure = pressure.composite
            else:
                self._quarantine_pressure = 0.0
        except Exception:
            self._quarantine_pressure = 0.0

        self._suppressed_lanes = set()
        if self._quarantine_pressure >= self.PRESSURE_ELEVATED:
            self._pressure_level = "high"
            self._suppressed_lanes = (
                _CONDITIONALLY_ALLOWED_LANES | _HIGH_RISK_LANES
            ) - _BACKGROUND_SAFE_LANES
        elif self._quarantine_pressure >= self.PRESSURE_NORMAL:
            self._pressure_level = "elevated"
            self._suppressed_lanes = set(_HIGH_RISK_LANES)
        else:
            self._pressure_level = "normal"

    def tick(self, *, mode: str = "passive") -> None:
        """Advance active jobs. Called periodically from consciousness tick loop."""
        now = time.time()
        self._current_mode = mode
        self._sample_quarantine_pressure()
        completed_ids: list[str] = []

        for acq_id, job in list(self._active_jobs.items()):
            try:
                self._tick_job(job, now)
                self._store.save_job(job)

                if job.status in ("completed", "failed", "cancelled"):
                    completed_ids.append(acq_id)
            except Exception as exc:
                logger.warning("Acquisition tick error for %s: %s", acq_id, exc)

        for acq_id in completed_ids:
            self._active_jobs.pop(acq_id, None)

    def _tick_job(self, job: CapabilityAcquisitionJob, now: float) -> None:
        """Advance a single job through its lanes."""
        if job.status in ("completed", "failed", "cancelled"):
            return

        # For knowledge_only, fast-path: evidence grounding + truth, then complete
        if job.outcome_class == "knowledge_only":
            self._tick_knowledge_only(job)
            return

        # Check if awaiting human input
        if job.status in ("awaiting_plan_review", "awaiting_approval"):
            return

        # Walk through lanes in order
        for lane_name in job.required_lanes:
            ls = job.lanes.get(lane_name)
            if ls is None:
                continue

            if ls.status == "pending":
                self._dispatch_lane(job, lane_name)
                return
            elif ls.status == "running":
                self._check_lane_progress(job, lane_name)
                return
            elif ls.status == "failed":
                if ls.retry_count < 3:
                    ls.status = "pending"
                    job.touch()
                    return
                else:
                    job.set_status("failed")
                    self._total_failed += 1
                    self._emit_event("ACQUISITION_FAILED", job)
                    return

        # All lanes completed
        if all(ls.status in ("completed", "skipped") for ls in job.lanes.values()):
            job.set_status("completed")
            self._total_completed += 1
            self._emit_event("ACQUISITION_COMPLETED", job)

    def _tick_knowledge_only(self, job: CapabilityAcquisitionJob) -> None:
        """Fast path for knowledge-only acquisitions."""
        eg_lane = job.lanes.get("evidence_grounding")
        truth_lane = job.lanes.get("truth")

        if eg_lane and eg_lane.status == "pending":
            eg_lane.status = "completed"
            eg_lane.completed_at = time.time()
            job.touch()

        if truth_lane and truth_lane.status == "pending":
            truth_lane.status = "completed"
            truth_lane.completed_at = time.time()
            job.touch()

        if all(ls.status in ("completed", "skipped") for ls in job.lanes.values()):
            job.set_status("completed")
            self._total_completed += 1

    def _lane_allowed_in_mode(self, lane_name: str) -> bool:
        """Check whether a lane is allowed given current mode and quarantine pressure."""
        # Quarantine pressure suppression takes priority
        if lane_name in getattr(self, "_suppressed_lanes", set()):
            return False

        mode = getattr(self, "_current_mode", "passive")
        if lane_name in _BACKGROUND_SAFE_LANES:
            return True
        if lane_name in _CONDITIONALLY_ALLOWED_LANES:
            return mode in _CONDITIONAL_LANE_ALLOWED_MODES
        if lane_name in _HIGH_RISK_LANES:
            return mode in _HEAVY_LANE_ALLOWED_MODES
        return mode in _HEAVY_LANE_ALLOWED_MODES

    def _dispatch_lane(self, job: CapabilityAcquisitionJob, lane_name: str) -> None:
        """Start a lane by delegating to the appropriate subsystem."""
        ls = job.lanes[lane_name]

        # ── Mode-aware lane gating ───────────────────────────────────
        if not self._lane_allowed_in_mode(lane_name):
            mode = getattr(self, "_current_mode", "passive")
            logger.info(
                "Acquisition %s: lane '%s' deferred (mode=%s, policy=background_only)",
                job.acquisition_id, lane_name, mode,
            )
            return

        # ── Human gate: plan review ──────────────────────────────────
        if lane_name == "plan_review" and job.risk_tier >= 1:
            needs_review = (
                job.classification_confidence < 0.7
                or job.risk_tier >= 2
                or self._has_low_freshness_docs(job)
            )
            if needs_review:
                job.set_status("awaiting_plan_review")
                ls.status = "running"
                ls.started_at = time.time()
                self._emit_event("ACQUISITION_APPROVAL_NEEDED", job, {"gate": "plan_review"})
                plan = self._store.load_plan(job.plan_id) if job.plan_id else None
                if plan:
                    self._run_shadow_prediction(job, plan)
                return
            else:
                ls.status = "skipped"
                ls.completed_at = time.time()
                job.touch()
                return

        # ── Human gate: deployment approval ──────────────────────────
        if lane_name == "deployment" and job.risk_tier >= 2:
            if job.approval_status == "not_required":
                job.approval_status = "pending"
            if job.approval_status == "pending":
                job.set_status("awaiting_approval")
                ls.status = "running"
                ls.started_at = time.time()
                self._emit_event("ACQUISITION_APPROVAL_NEEDED", job, {"gate": "deployment"})
                return

        job.start_lane(lane_name)
        self._emit_event("ACQUISITION_LANE_STARTED", job, {"lane": lane_name})

        if lane_name in _BACKGROUND_WORKER_LANES:
            self._start_background_lane(job, lane_name)
            return

        # ── Evidence grounding ───────────────────────────────────────
        if lane_name == "evidence_grounding":
            self._run_evidence_grounding(job)
            return

        # ── Documentation resolution ─────────────────────────────────
        if lane_name == "doc_resolution":
            self._run_doc_resolution(job)
            return

        # ── Planning ─────────────────────────────────────────────────
        if lane_name == "planning":
            self._run_planning(job)
            return

        # ── Truth recording ──────────────────────────────────────────
        if lane_name == "truth":
            self._run_truth_recording(job)
            return

        # ── Implementation (CodeGen) ─────────────────────────────────
        if lane_name == "implementation":
            self._run_implementation(job)
            return

        # ── Environment setup (venv for isolated_subprocess) ─────────
        if lane_name == "environment_setup":
            self._run_environment_setup(job)
            return

        # ── Plugin quarantine ────────────────────────────────────────
        if lane_name == "plugin_quarantine":
            self._run_plugin_quarantine(job)
            return

        # ── Verification ─────────────────────────────────────────────
        if lane_name == "verification":
            self._run_verification(job)
            return

        # ── Skill registration ───────────────────────────────────────
        if lane_name == "skill_registration":
            self._run_skill_registration(job)
            return

        # ── Plugin activation (promote from quarantine) ─────────────
        if lane_name == "plugin_activation":
            self._run_plugin_activation(job)
            return

        # ── Unimplemented lanes: fail-fast with clear message ─────
        if lane_name in ("self_improve", "matrix_specialist"):
            job.fail_lane(
                lane_name,
                f"Lane '{lane_name}' is not yet implemented. "
                f"This lane will be wired in a future phase."
            )
            logger.info(
                "Acquisition %s: lane '%s' is not yet implemented, failing explicitly",
                job.acquisition_id, lane_name,
            )
            return

        job.fail_lane(
            lane_name,
            f"Unknown lane '{lane_name}' — no dispatch handler registered."
        )
        logger.warning(
            "Acquisition %s: no dispatch handler for lane '%s'",
            job.acquisition_id, lane_name,
        )

    def _start_background_lane(self, job: CapabilityAcquisitionJob, lane_name: str) -> None:
        """Run heavyweight lanes off the brain/dashboard hot path."""
        key = (job.acquisition_id, lane_name)
        future = self._lane_futures.get(key)
        if future and not future.done():
            return

        def _run() -> None:
            try:
                self._run_lane_body(job, lane_name)
            except Exception as exc:
                logger.exception(
                    "Acquisition %s: background lane %s failed",
                    job.acquisition_id,
                    lane_name,
                )
                job.fail_lane(lane_name, f"{type(exc).__name__}: {str(exc)[:300]}")
            finally:
                self._store.save_job(job)

        self._lane_futures[key] = self._lane_executor.submit(_run)

    def _run_lane_body(self, job: CapabilityAcquisitionJob, lane_name: str) -> None:
        if lane_name == "planning":
            self._run_planning(job)
            return
        if lane_name == "implementation":
            self._run_implementation(job)
            return
        if lane_name == "environment_setup":
            self._run_environment_setup(job)
            return
        if lane_name == "verification":
            self._run_verification(job)
            return
        raise ValueError(f"Lane '{lane_name}' is not configured for background execution.")

    def _check_lane_progress(self, job: CapabilityAcquisitionJob, lane_name: str) -> None:
        """Check if a running lane has finished or can advance."""
        ls = job.lanes.get(lane_name)
        if ls is None:
            return
        if lane_name in _BACKGROUND_WORKER_LANES:
            future = self._lane_futures.get((job.acquisition_id, lane_name))
            if future is not None:
                if future.done():
                    self._lane_futures.pop((job.acquisition_id, lane_name), None)
                    try:
                        future.result()
                    except Exception as exc:
                        job.fail_lane(lane_name, f"{type(exc).__name__}: {str(exc)[:300]}")
                return
            if ls.started_at and (time.time() - ls.started_at) <= 1800:
                logger.warning(
                    "Acquisition %s: lane %s is running without a worker; resetting to pending",
                    job.acquisition_id,
                    lane_name,
                )
                ls.status = "pending"
                job.touch()
                return
        # Human-gated lanes are checked by approve_plan / approve_deployment
        if lane_name in ("plan_review", "deployment"):
            return
        # Plugin activation re-checks gate conditions each tick
        if lane_name == "plugin_activation":
            self._run_plugin_activation(job)
            return
        # Timeout: fail lane after 30 minutes
        if ls.started_at and (time.time() - ls.started_at) > 1800:
            job.fail_lane(lane_name, "Lane timed out after 30 minutes")

    # ── Lane implementations ───────────────────────────────────────────

    def _run_evidence_grounding(self, job: CapabilityAcquisitionJob) -> None:
        """Evidence grounding: check internal sources before external research."""
        try:
            existing = self._check_internal_evidence(job)
            research = ResearchArtifact(
                acquisition_id=job.acquisition_id,
                existing_capabilities=existing.get("capabilities", []),
                prior_attempts=existing.get("prior_attempts", []),
                sources=existing.get("sources", []),
            )
            self._store.save_research(research)
            job.add_artifact_ref(research.artifact_id)
            job.research_intent_id = research.artifact_id
            job.complete_lane("evidence_grounding", child_id=research.artifact_id)
            self._emit_event("ACQUISITION_LANE_COMPLETED", job, {"lane": "evidence_grounding"})
        except Exception as exc:
            logger.warning("Evidence grounding failed: %s", exc)
            job.fail_lane("evidence_grounding", str(exc))

    def _check_internal_evidence(self, job: CapabilityAcquisitionJob) -> dict[str, Any]:
        """Check SkillRegistry, prior acquisitions, library, codebase for existing evidence."""
        result: dict[str, Any] = {"capabilities": [], "prior_attempts": [], "sources": []}
        intent_lower = job.user_intent.lower()

        # Check SkillRegistry for matching capabilities
        try:
            from skills.registry import skill_registry
            for skill in skill_registry.get_all():
                skill_name = skill.get("name", "").lower()
                if not skill_name:
                    continue
                keywords = [w for w in skill_name.split() if len(w) > 2]
                if any(kw in intent_lower for kw in keywords):
                    result["capabilities"].append(skill.get("skill_id", ""))
                    result["sources"].append({
                        "type": "skill_registry",
                        "id": skill.get("skill_id", ""),
                        "name": skill.get("name", ""),
                        "status": skill.get("status", "unknown"),
                    })
        except Exception:
            pass

        # Check prior acquisitions for overlap
        try:
            all_jobs = self._store.list_jobs()
            for prev in all_jobs:
                if prev.acquisition_id == job.acquisition_id:
                    continue
                if _text_overlap(prev.title, job.title):
                    result["prior_attempts"].append(prev.acquisition_id)
                    result["sources"].append({
                        "type": "prior_acquisition",
                        "id": prev.acquisition_id,
                        "title": prev.title,
                        "status": prev.status,
                        "outcome_class": prev.outcome_class,
                    })
        except Exception:
            pass

        # Check codebase index for relevant symbols/modules
        try:
            from tools.codebase_tool import CodebaseIndex
            idx = CodebaseIndex.get_instance()
            if idx:
                hits = idx.search(job.user_intent, max_results=5)
                for hit in hits:
                    result["sources"].append({
                        "type": "codebase",
                        "symbol": hit.get("symbol", hit.get("name", "")),
                        "file": hit.get("file", ""),
                        "relevance": hit.get("score", 0.0),
                    })
        except Exception:
            pass

        return result

    def _run_doc_resolution(self, job: CapabilityAcquisitionJob) -> None:
        """Documentation resolution: search library and codebase, then produce honest artifact."""
        try:
            citations: list[dict[str, Any]] = []
            source_type = "none_found"
            relevance = 0.0
            freshness_score = 0.0

            # Search the document library for relevant chunks
            try:
                from library.index import LibraryIndex
                lib_idx = LibraryIndex.get_instance()
                if lib_idx:
                    results = lib_idx.search(job.user_intent, top_k=5)
                    for r in results:
                        score = r.get("score", 0.0) if isinstance(r, dict) else getattr(r, "score", 0.0)
                        text = r.get("text", "") if isinstance(r, dict) else getattr(r, "text", "")
                        source_id = r.get("source_id", "") if isinstance(r, dict) else getattr(r, "source_id", "")
                        if score > 0.3:
                            citations.append({
                                "origin": "library",
                                "source_id": source_id,
                                "text_preview": text[:200],
                                "score": round(score, 3),
                            })
                    if citations:
                        source_type = "local_doc"
                        relevance = max(c["score"] for c in citations)
                        freshness_score = 0.7
            except Exception:
                pass

            # Search codebase index for relevant code context
            if not citations:
                try:
                    from tools.codebase_tool import CodebaseIndex
                    idx = CodebaseIndex.get_instance()
                    if idx:
                        hits = idx.search(job.user_intent, max_results=3)
                        for hit in hits:
                            score = hit.get("score", 0.0)
                            if score > 0.2:
                                citations.append({
                                    "origin": "codebase",
                                    "symbol": hit.get("symbol", hit.get("name", "")),
                                    "file": hit.get("file", ""),
                                    "score": round(score, 3),
                                })
                        if citations:
                            source_type = "repo_doc"
                            relevance = max(c["score"] for c in citations)
                            freshness_score = 0.9
                except Exception:
                    pass

            doc = DocumentationArtifact(
                acquisition_id=job.acquisition_id,
                source_type=source_type,
                topic=job.title,
                version_scope="latest",
                relevance=round(relevance, 3),
                freshness_score=round(freshness_score, 3),
                citations=citations,
            )
            self._store.save_doc(doc)
            job.doc_artifact_ids.append(doc.artifact_id)
            job.add_artifact_ref(doc.artifact_id)
            job.complete_lane("doc_resolution", child_id=doc.artifact_id)
            self._emit_event("ACQUISITION_LANE_COMPLETED", job, {"lane": "doc_resolution"})
        except Exception as exc:
            logger.warning("Doc resolution failed: %s", exc)
            job.fail_lane("doc_resolution", str(exc))

    def _run_planning(self, job: CapabilityAcquisitionJob) -> None:
        """Planning lane: synthesize structural plan + LLM technical design."""
        try:
            from acquisition.planner import AcquisitionPlanner
            planner = AcquisitionPlanner()

            doc_artifacts = []
            for doc_id in job.doc_artifact_ids:
                doc = self._store.load_doc(doc_id)
                if doc:
                    doc_artifacts.append(doc)

            plan = planner.synthesize(job, doc_artifacts=doc_artifacts)

            self._enrich_plan_with_technical_design(job, plan)
            quality_error = self._plan_quality_error(job, plan)
            if quality_error:
                self._record_planning_diagnostics(job, plan, quality_error)
                raise ValueError(quality_error)

            self._store.save_plan(plan)
            job.plan_id = plan.plan_id
            job.add_artifact_ref(plan.plan_id)
            job.complete_lane("planning", child_id=plan.plan_id)

            self._emit_event("ACQUISITION_PLAN_READY", job, {
                "plan_id": plan.plan_id,
                "risk_level": plan.risk_level,
            })
        except Exception as exc:
            logger.warning("Planning failed: %s", exc)
            job.fail_lane("planning", str(exc))
            self._record_skill_acquisition_feature(job, locals().get("plan"), stage="planning_failed")
            self._record_skill_acquisition_label(job)

    def _enrich_plan_with_technical_design(
        self, job: CapabilityAcquisitionJob, plan: AcquisitionPlan
    ) -> None:
        """Ask the coder LLM to produce a technical design for the plan."""
        codegen = getattr(self, "_codegen_service", None)
        if codegen is None or not codegen.coder_available:
            plan.technical_approach = (
                "Technical design unavailable — coder model not loaded. "
                "Plan contains structural information only."
            )
            job.planning_diagnostics = {
                "failure_reason": "planning_failed_coder_unavailable",
                "raw_output_length": 0,
                "missing_fields": [],
                "retry_count": getattr(job.lanes.get("planning"), "retry_count", 0) if job.lanes else 0,
                "coder_available": False,
                "updated_at": time.time(),
            }
            logger.info("Planning %s: coder not available, skipping technical design", job.acquisition_id)
            return

        from self_improve.conversation import PLANNER_SYSTEM_PROMPT
        system_prompt = PLANNER_SYSTEM_PROMPT

        user_prompt = (
            f"Design a plugin for the following request:\n\n"
            f"Request: {job.user_intent}\n"
            f"Outcome class: {job.outcome_class}\n"
            f"Risk tier: {job.risk_tier}\n"
            f"Required capabilities: {', '.join(plan.required_capabilities)}\n"
        )

        # Include operator feedback from prior rejection(s) for revision
        revision_context = self._build_revision_context(job, plan)
        if revision_context:
            user_prompt += f"\n{revision_context}"
            plan.version = (plan.version or 1) + 1
            logger.info("Planning %s: revision v%d with operator feedback", job.acquisition_id, plan.version)

        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if hasattr(codegen, "set_consumer"):
                codegen.set_consumer("acquisition")
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    raw = pool.submit(
                        lambda: asyncio.run(codegen.generate(
                            messages=[{"role": "user", "content": user_prompt}],
                            system_prompt=system_prompt,
                        ))
                    ).result(timeout=300)
            else:
                raw = asyncio.run(codegen.generate(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=system_prompt,
                ))

            if raw:
                self._parse_technical_design(plan, raw)
                job.planning_diagnostics = {
                    "failure_reason": "",
                    "raw_output_length": len(raw),
                    "raw_output_preview": self._bounded_preview(raw),
                    "missing_fields": self._plan_missing_fields(plan),
                    "retry_count": getattr(job.lanes.get("planning"), "retry_count", 0) if job.lanes else 0,
                    "coder_available": True,
                    "updated_at": time.time(),
                }
                logger.info("Planning %s: technical design produced (%d chars)", job.acquisition_id, len(raw))
            else:
                plan.technical_approach = "Coder model returned empty response."
                job.planning_diagnostics = {
                    "failure_reason": "planning_failed_empty_coder_response",
                    "raw_output_length": 0,
                    "raw_output_preview": "",
                    "missing_fields": self._plan_missing_fields(plan),
                    "retry_count": getattr(job.lanes.get("planning"), "retry_count", 0) if job.lanes else 0,
                    "coder_available": True,
                    "updated_at": time.time(),
                }
                logger.warning("Planning %s: coder returned empty response", job.acquisition_id)
        except Exception as exc:
            err_msg = str(exc) or type(exc).__name__
            plan.technical_approach = f"Technical design generation failed: {err_msg}"
            job.planning_diagnostics = {
                "failure_reason": "planning_failed_technical_design_generation",
                "raw_output_length": 0,
                "raw_output_preview": "",
                "missing_fields": self._plan_missing_fields(plan),
                "retry_count": getattr(job.lanes.get("planning"), "retry_count", 0) if job.lanes else 0,
                "coder_available": True,
                "error": err_msg[:500],
                "updated_at": time.time(),
            }
            logger.warning("Planning %s: technical design failed: %s", job.acquisition_id, err_msg)

        self._record_plan_features_signal(job, plan, getattr(plan, "version", 1))

    @staticmethod
    def _plan_quality_error(job: CapabilityAcquisitionJob, plan: AcquisitionPlan) -> str:
        """Return a planning failure reason for incomplete operational plans."""
        if getattr(job, "outcome_class", "") != "plugin_creation":
            return ""

        technical_approach = (getattr(plan, "technical_approach", "") or "").strip()
        implementation_sketch = (getattr(plan, "implementation_sketch", "") or "").strip()
        test_cases = getattr(plan, "test_cases", None) or []

        structural_only = (
            technical_approach
            == "Technical design unavailable — coder model not loaded. Plan contains structural information only."
        )
        if structural_only:
            return "planning_failed_coder_unavailable"

        if technical_approach == "Coder model returned empty response.":
            return "planning_failed_empty_coder_response"
        if technical_approach.startswith("Technical design generation failed:"):
            return "planning_failed_technical_design_generation"

        missing: list[str] = []
        if not technical_approach:
            missing.append("technical_approach")
        if not implementation_sketch:
            missing.append("implementation_sketch")
        if not test_cases:
            missing.append("test_cases")
        if missing:
            return "planning_failed_incomplete_plan:" + ",".join(missing)
        return ""

    @staticmethod
    def _plan_missing_fields(plan: AcquisitionPlan) -> list[str]:
        missing: list[str] = []
        if not (getattr(plan, "technical_approach", "") or "").strip():
            missing.append("technical_approach")
        if not (getattr(plan, "implementation_sketch", "") or "").strip():
            missing.append("implementation_sketch")
        if not (getattr(plan, "test_cases", None) or []):
            missing.append("test_cases")
        return missing

    @staticmethod
    def _bounded_preview(text: str, limit: int = 1200) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "\n...[truncated]"

    def _record_planning_diagnostics(
        self,
        job: CapabilityAcquisitionJob,
        plan: AcquisitionPlan,
        failure_reason: str,
    ) -> None:
        diag = dict(getattr(job, "planning_diagnostics", {}) or {})
        diag.update({
            "failure_reason": failure_reason,
            "raw_output_length": int(diag.get("raw_output_length", 0) or 0),
            "raw_output_preview": str(diag.get("raw_output_preview", "") or "")[:1500],
            "missing_fields": self._plan_missing_fields(plan),
            "retry_count": getattr(job.lanes.get("planning"), "retry_count", 0) if job.lanes else 0,
            "updated_at": time.time(),
        })
        job.planning_diagnostics = diag

    def _build_revision_context(
        self, job: CapabilityAcquisitionJob, plan: AcquisitionPlan
    ) -> str:
        """Build revision context from all prior rejection reviews for this job."""
        if not job.plan_review_id:
            return ""
        # Load the most recent review; scan artifact_refs for any older rejection reviews too
        reviews: list[Any] = []
        seen: set[str] = set()
        for art_id in list(job.artifact_refs or []) + [job.plan_review_id]:
            if art_id in seen:
                continue
            seen.add(art_id)
            rev = self._store.load_review(art_id)
            if rev and getattr(rev, "verdict", "") == "rejected":
                reviews.append(rev)
        if not reviews:
            return ""

        lines = [
            "## IMPORTANT: PRIOR PLAN WAS REJECTED BY OPERATOR",
            "The previous version of this plan was reviewed and rejected.",
            "You MUST address the operator's feedback in your revised design.",
            "",
        ]
        for rev in reviews:
            notes = getattr(rev, "operator_notes", "") or ""
            category = getattr(rev, "reason_category", "unknown")
            changes = getattr(rev, "suggested_changes", []) or []
            lines.append(f"Rejection reason: {category}")
            if notes:
                lines.append(f"Operator notes: {notes}")
            if changes:
                lines.append("Suggested changes:")
                for ch in changes:
                    desc = ch.get("description", str(ch)) if isinstance(ch, dict) else str(ch)
                    lines.append(f"  - {desc}")
            lines.append("")

        prev_approach = getattr(plan, "technical_approach", "") or ""
        if prev_approach and prev_approach != "Technical design unavailable — coder model not loaded. Plan contains structural information only.":
            lines.append("## PREVIOUS PLAN (to revise, not start from scratch):")
            lines.append(prev_approach[:2000])
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _parse_technical_design(plan: AcquisitionPlan, raw: str) -> None:
        """Parse the coder LLM's structured technical design into plan fields."""
        cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
        try:
            if cleaned.startswith("{"):
                data = json.loads(cleaned)
                plan.user_story = str(data.get("user_story", "") or "")
                plan.technical_approach = str(data.get("technical_approach", "") or "")
                plan.implementation_sketch = str(data.get("implementation_sketch", "") or "")
                plan.risk_analysis = str(data.get("risk_analysis", "") or "")
                deps = data.get("dependencies", [])
                tests = data.get("test_cases", [])
                plan.dependencies = [str(d).strip() for d in (deps if isinstance(deps, list) else str(deps).split(",")) if str(d).strip()]
                plan.test_cases = [str(t).strip() for t in (tests if isinstance(tests, list) else str(tests).split("\n")) if str(t).strip()]
                return
        except Exception:
            pass

        sections = {
            "USER STORY:": "user_story",
            "TECHNICAL APPROACH:": "technical_approach",
            "IMPLEMENTATION SKETCH:": "implementation_sketch",
            "DEPENDENCIES:": "_dependencies_raw",
            "TEST CASES:": "_test_cases_raw",
            "RISK ANALYSIS:": "risk_analysis",
        }

        current_field = None
        current_lines: list[str] = []
        parsed: dict[str, str] = {}

        header_re = re.compile(
            r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:[*_]{1,3})?"
            r"(USER STORY|TECHNICAL APPROACH|IMPLEMENTATION SKETCH|DEPENDENCIES|TEST CASES|RISK ANALYSIS)"
            r"(?:[*_]{1,3})?\s*:?\s*(.*)$",
            re.IGNORECASE,
        )
        header_to_field = {h.rstrip(":"): f for h, f in sections.items()}

        for line in cleaned.split("\n"):
            stripped = line.strip()
            matched = False
            match = header_re.match(stripped)
            if match:
                if current_field:
                    parsed[current_field] = "\n".join(current_lines).strip()
                current_field = header_to_field.get(match.group(1).upper(), "")
                current_lines = []
                remainder = match.group(2).strip()
                if remainder:
                    current_lines.append(remainder)
                matched = True
            if not matched and current_field:
                current_lines.append(line.rstrip())

        if current_field:
            parsed[current_field] = "\n".join(current_lines).strip()

        plan.user_story = parsed.get("user_story", "")
        plan.technical_approach = parsed.get("technical_approach", "")
        plan.implementation_sketch = parsed.get("implementation_sketch", "")
        plan.risk_analysis = parsed.get("risk_analysis", "")

        deps_raw = parsed.get("_dependencies_raw", "")
        if deps_raw and deps_raw.lower() != "none":
            plan.dependencies = [d.strip() for d in deps_raw.replace("\n", ",").split(",") if d.strip()]

        tests_raw = parsed.get("_test_cases_raw", "")
        if tests_raw:
            plan.test_cases = [t.strip() for t in tests_raw.split("\n") if t.strip()]

    def _run_truth_recording(self, job: CapabilityAcquisitionJob) -> None:
        """Truth lane: record outcome in attribution ledger + memory."""
        try:
            self._record_ledger_entry(job, "acquisition:completed")

            # Write summary to memory
            try:
                from memory.core import create_memory
                summary = (
                    f"Capability acquisition '{job.title}' "
                    f"(outcome={job.outcome_class}, status={job.status})"
                )
                create_memory(
                    content=summary,
                    memory_type="experience",
                    tags=["acquisition", job.outcome_class, job.acquisition_id],
                    provenance="system",
                )
            except Exception:
                pass

            job.complete_lane("truth")
            self._emit_event("ACQUISITION_LANE_COMPLETED", job, {"lane": "truth"})
        except Exception as exc:
            logger.warning("Truth recording failed: %s", exc)
            job.fail_lane("truth", str(exc))

    def _load_doc_evidence(self, job: CapabilityAcquisitionJob) -> list[DocumentationArtifact]:
        docs: list[DocumentationArtifact] = []
        for doc_id in job.doc_artifact_ids:
            doc = self._store.load_doc(doc_id)
            if doc:
                docs.append(doc)
        return docs

    @staticmethod
    def _meaningful_doc_evidence(docs: list[DocumentationArtifact]) -> list[DocumentationArtifact]:
        return [
            d for d in docs
            if getattr(d, "source_type", "") != "none_found"
            and (getattr(d, "citations", None) or getattr(d, "relevance", 0.0) > 0)
        ]

    def _skill_contract_context(self, job: CapabilityAcquisitionJob) -> dict[str, Any]:
        requested = job.requested_by or {}
        skill_id = requested.get("skill_id", "")
        if not skill_id:
            return {}
        try:
            from dataclasses import asdict
            from skills.execution_contracts import get_contract
            contract = get_contract(skill_id)
            if contract is None:
                return {"skill_id": skill_id, "contract_found": False}
            return {
                "skill_id": skill_id,
                "learning_job_id": requested.get("learning_job_id", ""),
                "contract_id": contract.contract_id,
                "family": contract.family,
                "required_executor_kind": contract.required_executor_kind,
                "requires_sandbox": contract.requires_sandbox,
                "smoke_test_name": contract.smoke_test_name,
                "smoke_fixtures": [asdict(f) for f in contract.smoke_fixtures],
            }
        except Exception as exc:
            return {"skill_id": skill_id, "contract_error": str(exc)[:200]}

    def _build_acquisition_codegen_packet(
        self,
        job: CapabilityAcquisitionJob,
        plan: AcquisitionPlan,
    ) -> tuple[list[dict[str, str]], str, list[dict[str, Any]]]:
        """Build the Jarvis-authored prompt packet for acquisition codegen."""
        docs = self._load_doc_evidence(job)
        doc_rows = [
            {
                "artifact_id": d.artifact_id,
                "source_type": d.source_type,
                "topic": d.topic,
                "relevance": d.relevance,
                "freshness_score": d.freshness_score,
                "citation_count": len(d.citations or []),
            }
            for d in docs
        ]
        contract = self._skill_contract_context(job)
        revision_context = self._build_revision_context(job, plan)
        prompt = (
            "## Jarvis Skill Acquisition Engineering Mode\n"
            "Jarvis is producing an operational proof plugin, not asking the coder to invent truth.\n"
            "Use only the structured acquisition state, plan, evidence, and contract fixtures below.\n"
            "The generated plugin is a candidate artifact; skill verification remains owned by the SkillRegistry and contract smoke tests.\n\n"
            f"## Acquisition\n"
            f"Acquisition ID: {job.acquisition_id}\n"
            f"Title: {job.title}\n"
            f"User intent: {job.user_intent}\n"
            f"Outcome class: {job.outcome_class}\n"
            f"Risk tier: {job.risk_tier}\n"
            f"Requested by: {json.dumps(job.requested_by or {}, sort_keys=True)}\n\n"
            f"## Plan\n"
            f"Plan ID: {plan.plan_id}\n"
            f"Objective: {plan.objective}\n"
            f"Risk level: {plan.risk_level}\n"
            f"Technical approach:\n{plan.technical_approach}\n\n"
            f"Implementation sketch:\n{plan.implementation_sketch}\n\n"
            f"Implementation steps:\n{json.dumps(plan.implementation_path, indent=2)}\n\n"
            f"Required capabilities:\n{json.dumps(plan.required_capabilities, indent=2)}\n\n"
            f"Dependencies:\n{json.dumps(plan.dependencies, indent=2)}\n\n"
            f"Plan test cases:\n{json.dumps(plan.test_cases, indent=2)}\n\n"
            f"Risk analysis:\n{plan.risk_analysis}\n\n"
            f"## Documentation Evidence\n{json.dumps(doc_rows, indent=2)}\n\n"
            f"## Skill Contract Fixture\n{json.dumps(contract, indent=2)}\n\n"
        )
        if revision_context:
            prompt += f"{revision_context}\n"
        prompt += (
            "## Validation Requirements\n"
            "- The plugin must parse the fixture input shape and return JSON-serializable structured data.\n"
            "- For CSV totals, return numeric sums as numbers, not prose.\n"
            "- Do not write files, spawn processes, access credentials, or create network side effects.\n"
            "- The code must be deterministic and stdlib-only unless dependencies were explicitly approved.\n"
        )
        system_prompt = (
            "You are Jarvis's skill-acquisition code generation engine. "
            "You implement candidate plugins from Jarvis-authored acquisition packets. "
            "Return ONLY valid JSON with files for handler.py and __init__.py. "
            "Do not include markdown or explanations."
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        job.codegen_prompt_diagnostics = {
            "prompt_version": _SKILL_ACQUISITION_PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "prompt_preview": self._bounded_preview(prompt, limit=1500),
            "source_artifact_ids": list(job.artifact_refs or []),
            "doc_artifact_ids": list(job.doc_artifact_ids or []),
            "skill_id": contract.get("skill_id", ""),
            "contract_id": contract.get("contract_id", ""),
            "plan_id": plan.plan_id,
            "updated_at": time.time(),
        }
        self._record_skill_acquisition_feature(job, plan, stage="prompt_emitted")
        evidence_bundle = [d.to_dict() for d in docs]
        if contract.get("contract_id"):
            evidence_bundle.append({
                "source_type": "skill_contract",
                "artifact_id": contract.get("contract_id", ""),
                "topic": contract.get("family", "skill_contract"),
                "relevance": 1.0,
                "freshness_score": 1.0,
                "citations": [{"source": "skills.execution_contracts", "skill_id": contract.get("skill_id", "")}],
            })
        return [{"role": "user", "content": prompt}], system_prompt, evidence_bundle

    def _run_implementation(self, job: CapabilityAcquisitionJob) -> None:
        """Implementation lane: generate code via CodeGenService → PluginCodeBundle."""
        try:
            if not job.plan_id:
                job.fail_lane("implementation", "No plan available for implementation")
                return

            plan = self._store.load_plan(job.plan_id)
            if not plan:
                job.fail_lane("implementation", f"Plan {job.plan_id} not found in store")
                return

            codegen_service = getattr(self, "_codegen_service", None)

            if codegen_service is None or not codegen_service.coder_available:
                job.fail_lane("implementation", "codegen_unavailable")
                self._record_skill_acquisition_feature(job, plan, stage="implementation_failed")
                self._record_skill_acquisition_label(job)
                return

            messages, system_prompt, evidence_bundle = self._build_acquisition_codegen_packet(job, plan)
            if hasattr(codegen_service, "set_consumer"):
                codegen_service.set_consumer("acquisition")

            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        lambda: asyncio.run(codegen_service.generate_and_validate(
                            messages=messages,
                            system_prompt=system_prompt,
                            write_category="skill_plugin",
                            evidence_bundle=evidence_bundle,
                            risk_tier=job.risk_tier,
                        ))
                    ).result(timeout=600)
            else:
                result = asyncio.run(
                    codegen_service.generate_and_validate(
                        messages=messages,
                        system_prompt=system_prompt,
                        write_category="skill_plugin",
                        evidence_bundle=evidence_bundle,
                        risk_tier=job.risk_tier,
                    )
                )

            if result.get("success"):
                bundle = self._build_code_bundle(job, plan, result)
                if bundle:
                    self._store.save_code_bundle(bundle)
                    job.code_bundle_id = bundle.bundle_id
                    job.add_artifact_ref(bundle.bundle_id)

                job.complete_lane("implementation")
                self._emit_event("ACQUISITION_CODE_GENERATED", job, {
                    "lane": "implementation", "codegen": True,
                    "code_bundle_id": job.code_bundle_id,
                    "validation_errors": [],
                })
            else:
                errors = result.get("validation_errors", ["Unknown error"])
                job.fail_lane("implementation", "; ".join(errors))
                self._record_skill_acquisition_feature(job, plan, stage="implementation_failed")
                self._record_skill_acquisition_label(job)
                self._emit_event("ACQUISITION_FAILED", job, {
                    "lane": "implementation",
                    "validation_errors": errors,
                })
        except Exception as exc:
            job.fail_lane("implementation", str(exc))
            self._record_skill_acquisition_feature(job, plan if "plan" in locals() else None, stage="implementation_error")
            self._record_skill_acquisition_label(job)

    # ------------------------------------------------------------------
    # Plugin name + intent pattern derivation
    # ------------------------------------------------------------------

    _STRIP_VERBS = re.compile(
        r"^(build|create|make|develop|implement|add|write|design|generate)\s+"
        r"(a|an|the|my)?\s*",
        re.IGNORECASE,
    )
    _STRIP_SUFFIX = re.compile(
        r"\s+(tool|plugin|capability|feature|module|utility)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _derive_plugin_name(title: str, acquisition_id: str) -> str:
        """Produce a clean, short slug from a job title."""
        slug = AcquisitionOrchestrator._STRIP_VERBS.sub("", title)
        slug = AcquisitionOrchestrator._STRIP_SUFFIX.sub("", slug)
        slug = slug.strip().lower().replace(" ", "_").replace("-", "_")
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        slug = re.sub(r"_+", "_", slug).strip("_")
        slug = slug[:30]
        slug = slug.rstrip("_")
        return slug or f"plugin_{acquisition_id[-8:]}"

    @staticmethod
    def _derive_intent_patterns(
        job: CapabilityAcquisitionJob,
        plan: AcquisitionPlan | None = None,
        code_files: dict[str, str] | None = None,
    ) -> list[str]:
        """Derive intent-matching regex patterns for a plugin.

        Priority order:
          1. LLM-generated PLUGIN_MANIFEST.intent_patterns (from __init__.py)
          2. Plan keywords / required_capabilities
          3. Title-based heuristic fallback
        """
        # --- Source 1: LLM-generated manifest in code ---
        if code_files:
            init_src = code_files.get("__init__.py", "")
            if "intent_patterns" in init_src:
                try:
                    from tools.plugin_registry import PluginRegistry
                    manifest_dict = PluginRegistry._extract_manifest_safe(init_src) or {}
                    llm_patterns = manifest_dict.get("intent_patterns", [])
                    if isinstance(llm_patterns, list) and len(llm_patterns) >= 2:
                        valid = []
                        for p in llm_patterns:
                            if not isinstance(p, str) or len(p) > 200:
                                continue
                            try:
                                re.compile(p)
                                valid.append(p)
                            except re.error:
                                pass
                        if len(valid) >= 2:
                            return valid[:8]
                except Exception:
                    pass

        # --- Source 2: Plan capabilities / keywords ---
        keywords: list[str] = []
        if plan:
            keywords.extend(plan.required_capabilities or [])
            keywords.extend(plan.dependencies or [])
        if job.user_intent:
            cleaned = AcquisitionOrchestrator._STRIP_VERBS.sub("", job.user_intent)
            cleaned = AcquisitionOrchestrator._STRIP_SUFFIX.sub("", cleaned).strip()
            if cleaned:
                keywords.append(cleaned)
        if job.title:
            cleaned_title = AcquisitionOrchestrator._STRIP_VERBS.sub("", job.title)
            cleaned_title = AcquisitionOrchestrator._STRIP_SUFFIX.sub("", cleaned_title).strip()
            if cleaned_title and cleaned_title not in keywords:
                keywords.append(cleaned_title)

        patterns: list[str] = []

        nouns = [k.lower().strip() for k in keywords if k.strip()]
        nouns = list(dict.fromkeys(nouns))

        for noun in nouns[:4]:
            words = noun.split()
            if len(words) >= 2:
                core = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
                patterns.append(core)
            elif len(words) == 1 and len(words[0]) >= 3:
                patterns.append(r"\b" + re.escape(words[0]) + r"\b")

        # --- Source 3: Title heuristic fallback ---
        if not patterns:
            slug = AcquisitionOrchestrator._STRIP_VERBS.sub("", job.title)
            slug = AcquisitionOrchestrator._STRIP_SUFFIX.sub("", slug).strip().lower()
            words = slug.split()[:3]
            if words:
                patterns.append(r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b")

        return patterns[:8]

    def _build_code_bundle(
        self, job: CapabilityAcquisitionJob, plan: AcquisitionPlan, result: dict,
    ) -> PluginCodeBundle | None:
        """Extract generated code from codegen result into a PluginCodeBundle."""
        patch = result.get("patch")
        if patch is None:
            return None

        code_files: dict[str, str] = {}
        files = getattr(patch, "files", []) if not isinstance(patch, dict) else patch.get("files", [])
        for fd in files:
            if isinstance(fd, dict):
                filepath = fd.get("path", fd.get("filepath", fd.get("filename", "")))
                new_content = fd.get("new_content", fd.get("content", ""))
            else:
                filepath = getattr(fd, "path", getattr(fd, "filepath", ""))
                new_content = getattr(fd, "new_content", getattr(fd, "content", ""))

            if filepath and new_content:
                filename = filepath.split("/")[-1] if "/" in filepath else filepath
                code_files[filename] = new_content

        if not code_files:
            return None

        if (job.requested_by or {}).get("source") == "skill_operational_handoff":
            self._ensure_skill_plugin_runtime_contract(code_files)

        # Synthesize __init__.py bridge if LLM only produced handler.py
        if "handler.py" in code_files and "__init__.py" not in code_files:
            code_files["__init__.py"] = (
                "PLUGIN_MANIFEST = {}\n\n\n"
                "async def handle(text: str, context: dict) -> dict:\n"
                "    try:\n"
                "        from .handler import run\n"
                "    except Exception:\n"
                '        return {"output": "Plugin handler not available"}\n'
                "    try:\n"
                '        payload = {"text": text, "input": text, "request": text}\n'
                '        if isinstance(context, dict) and context.get("input_type"):\n'
                '            payload["input_type"] = context.get("input_type")\n'
                '        return {"output": run(payload)}\n'
                "    except Exception as exc:\n"
                '        return {"output": f"Plugin execution failed: {exc}"}\n'
            )

        plugin_name = self._derive_plugin_name(job.title, job.acquisition_id)
        if (job.requested_by or {}).get("source") == "skill_operational_handoff":
            # Skill proof retries often share the same title. Include the acquisition
            # suffix so a revised attempt cannot silently collide with stale packages.
            plugin_name = f"{plugin_name}_{job.acquisition_id[-6:]}"
        intent_patterns = self._derive_intent_patterns(job, plan, code_files)

        from tools.plugin_registry import PluginManifest
        manifest = PluginManifest(
            name=plugin_name,
            description=job.title,
            created_by=f"acquisition:{job.acquisition_id}",
            risk_tier=job.risk_tier,
            supervision_mode="shadow",
            intent_patterns=intent_patterns,
        )

        bundle = PluginCodeBundle(
            acquisition_id=job.acquisition_id,
            code_files=code_files,
            manifest_candidate=manifest.to_dict(),
            source_plan_id=plan.plan_id,
            doc_artifact_ids=list(job.doc_artifact_ids),
        )
        bundle.code_hash = bundle.compute_hash()
        return bundle

    @staticmethod
    def _ensure_skill_plugin_runtime_contract(code_files: dict[str, str]) -> None:
        """Normalize generated skill plugins to the runtime + proof contract.

        Skill contract verification and the plugin bridge both expect a
        synchronous ``run(args)`` callable in ``handler.py``. Coder output often
        follows the user-facing plugin shape and emits ``handle(request)``
        instead. That is a valid implementation body, but not a valid proof
        callable until it is adapted.
        """
        handler_src = code_files.get("handler.py", "")
        if not handler_src or "def run(" in handler_src:
            return
        if "def handle(" not in handler_src:
            return

        code_files["handler.py"] = handler_src.rstrip() + (
            "\n\n\n"
            "def run(args):\n"
            "    payload = dict(args) if isinstance(args, dict) else {\"request\": args, \"input\": args}\n"
            "    if \"input\" not in payload and \"request\" in payload:\n"
            "        payload[\"input\"] = payload[\"request\"]\n"
            "    return handle(payload)\n"
        )

    def _run_environment_setup(self, job: CapabilityAcquisitionJob) -> None:
        """Environment setup: create venv and install pinned deps for isolated plugins.

        For in_process plugins, this lane completes immediately (skipped).
        For isolated_subprocess plugins, creates the venv via PluginProcessManager.
        """
        from acquisition.job import EnvironmentSetupArtifact

        bundle = self._store.load_code_bundle(job.code_bundle_id) if job.code_bundle_id else None
        manifest_dict = bundle.manifest_candidate if bundle else {}
        execution_mode = manifest_dict.get("execution_mode", "in_process")

        artifact = EnvironmentSetupArtifact(
            acquisition_id=job.acquisition_id,
            plugin_name=manifest_dict.get("name", job.user_intent[:30] if job.user_intent else ""),
            execution_mode=execution_mode,
        )

        if execution_mode != "isolated_subprocess":
            artifact.skipped = True
            artifact.skip_reason = f"execution_mode is '{execution_mode}', no venv needed"
            self._store.save_environment_setup(artifact)
            job.environment_setup_id = artifact.artifact_id
            job.complete_lane("environment_setup")
            self._emit_event("ACQUISITION_LANE_COMPLETED", job, {"lane": "environment_setup", "skipped": True})
            logger.info("Acquisition %s: environment_setup skipped (in_process)", job.acquisition_id)
            return

        pinned_deps = manifest_dict.get("pinned_dependencies", [])
        artifact.pinned_dependencies = pinned_deps

        import asyncio
        try:
            from tools.plugin_process import PluginProcessManager
            plugin_dir = Path(__file__).parent.parent / "tools" / "plugins" / artifact.plugin_name
            if not plugin_dir.exists():
                plugin_dir.mkdir(parents=True, exist_ok=True)

            mgr = PluginProcessManager(
                plugin_name=artifact.plugin_name,
                plugin_dir=plugin_dir,
                pinned_dependencies=pinned_deps,
            )

            try:
                ok, log = asyncio.run(mgr.ensure_venv())
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, mgr.ensure_venv())
                    ok, log = future.result(timeout=420)

            artifact.install_log = log
            artifact.venv_path = str(mgr._venv_dir)
            artifact.import_verification_passed = ok

            if ok:
                artifact.installed_packages = pinned_deps
                self._store.save_environment_setup(artifact)
                job.environment_setup_id = artifact.artifact_id
                job.complete_lane("environment_setup")
                self._emit_event("ACQUISITION_LANE_COMPLETED", job, {"lane": "environment_setup"})
                logger.info("Acquisition %s: environment_setup completed for %s", job.acquisition_id, artifact.plugin_name)
            else:
                self._store.save_environment_setup(artifact)
                job.environment_setup_id = artifact.artifact_id
                job.fail_lane("environment_setup", f"Venv setup failed: {log[:200]}")
                logger.warning("Acquisition %s: environment_setup failed: %s", job.acquisition_id, log[:200])
        except Exception as exc:
            artifact.install_log = str(exc)
            self._store.save_environment_setup(artifact)
            job.environment_setup_id = artifact.artifact_id
            job.fail_lane("environment_setup", f"Environment setup error: {exc}")
            logger.exception("Acquisition %s: environment_setup error", job.acquisition_id)

    def _run_plugin_quarantine(self, job: CapabilityAcquisitionJob) -> None:
        """Plugin quarantine: deploy code to disk (NOT routable).

        Consumes PluginCodeBundle from the implementation lane. Stub plugins are
        not valid operational proof and must not enter the plugin registry.
        """
        try:
            from tools.plugin_registry import PluginManifest

            # Try to load the code bundle produced by the implementation lane
            bundle = None
            if job.code_bundle_id:
                bundle = self._store.load_code_bundle(job.code_bundle_id)

            if bundle and bundle.code_files:
                code_files = dict(bundle.code_files)
                manifest = PluginManifest.from_dict(bundle.manifest_candidate)
                plugin_name = manifest.name
                logger.info(
                    "Quarantine %s: using implementation bundle %s (%d files)",
                    job.acquisition_id, bundle.bundle_id, len(code_files),
                )
            else:
                job.fail_lane("plugin_quarantine", "missing_code_bundle_no_stub_allowed")
                return

            from tools.plugin_registry import get_plugin_registry
            registry = get_plugin_registry()

            ok, errors = registry.quarantine(plugin_name, code_files, manifest, job.acquisition_id)
            if ok:
                job.plugin_id = plugin_name
                job.complete_lane("plugin_quarantine", child_id=plugin_name)
                self._emit_event("ACQUISITION_PLUGIN_DEPLOYED", job, {
                    "lane": "plugin_quarantine", "plugin_name": plugin_name,
                    "state": "quarantined", "from_bundle": bundle is not None,
                })
            else:
                job.fail_lane("plugin_quarantine", "; ".join(errors))
        except Exception as exc:
            job.fail_lane("plugin_quarantine", str(exc))

    def _run_verification(self, job: CapabilityAcquisitionJob) -> None:
        """Verification lane: check lane verdicts + run real sandbox when code exists."""
        try:
            from acquisition.job import VerificationBundle
            bundle = VerificationBundle(
                acquisition_id=job.acquisition_id,
                lane_verdicts={},
                overall_passed=True,
            )

            # Check each completed lane
            for lane_name, ls in job.lanes.items():
                if lane_name in ("truth", "verification", "plan_review",
                                 "plugin_activation", "deployment"):
                    continue
                if ls.status == "completed":
                    bundle.lane_verdicts[lane_name] = True
                elif ls.status == "failed":
                    bundle.lane_verdicts[lane_name] = False
                    bundle.overall_passed = False
                elif ls.status == "skipped":
                    bundle.lane_verdicts[lane_name] = True

            # Run real sandbox validation on generated code when available
            sandbox_ran = False
            code_bundle = None
            if job.code_bundle_id:
                code_bundle = self._store.load_code_bundle(job.code_bundle_id)
                if code_bundle and code_bundle.code_files:
                    sandbox_ran = self._run_sandbox_on_bundle(bundle, code_bundle)

            if not sandbox_ran:
                bundle.risk_assessment["sandbox_status"] = "not_applicable"
                bundle.risk_assessment["reason"] = (
                    "no_code_bundle" if not job.code_bundle_id
                    else "code_bundle_empty_or_load_failed"
                )
                if (job.requested_by or {}).get("source") == "skill_operational_handoff":
                    bundle.overall_passed = False

            if code_bundle and (job.requested_by or {}).get("source") == "skill_operational_handoff":
                contract_passed = self._run_skill_contract_on_bundle(job, bundle, code_bundle)
                bundle.lane_verdicts["skill_contract_fixture"] = contract_passed
                if not contract_passed:
                    bundle.overall_passed = False

            self._record_skill_acquisition_feature(job, None, verification=bundle, stage="verification_complete")
            self._record_skill_acquisition_label(job, bundle)

            self._store.save_verification(bundle)
            job.verification_id = bundle.verification_id
            job.add_artifact_ref(bundle.verification_id)
            job.complete_lane("verification", child_id=bundle.verification_id)

            self._emit_event("ACQUISITION_VERIFIED", job, {
                "verification_id": bundle.verification_id,
                "overall_passed": bundle.overall_passed,
                "sandbox_ran": sandbox_ran,
            })
        except Exception as exc:
            job.fail_lane("verification", str(exc))

    def _run_sandbox_on_bundle(
        self, bundle: Any, code_bundle: PluginCodeBundle
    ) -> bool:
        """Run sandbox validation on a PluginCodeBundle. Returns True if sandbox ran."""
        try:
            from self_improve.code_patch import CodePatch, FileDiff

            diffs = []
            for filepath, content in code_bundle.code_files.items():
                filename = filepath.split("/")[-1]
                diffs.append(FileDiff(path=f"brain/tools/plugins/_gen/{filename}", new_content=content))

            if not diffs:
                bundle.risk_assessment["sandbox_status"] = "skipped"
                bundle.risk_assessment["reason"] = "no_code_files_in_bundle"
                return False

            patch = CodePatch(
                plan_id=code_bundle.source_plan_id,
                provider="acquisition_verification",
                files=diffs,
                description=f"Verification of bundle {code_bundle.bundle_id}",
            )

            # Run AST + lint validation (synchronous parts of sandbox)
            validation_errors = patch.validate()
            bundle.risk_assessment["patch_validation_errors"] = validation_errors
            if validation_errors:
                bundle.overall_passed = False
                bundle.lane_verdicts["sandbox_validation"] = False
                bundle.sandbox_result_ref = f"inline:validation_failed:{len(validation_errors)}_errors"
                bundle.risk_assessment["sandbox_status"] = "failed"
                return True

            # Run sandbox evaluation if available
            try:
                from codegen.sandbox import Sandbox
                import asyncio

                sandbox = Sandbox()
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        report = pool.submit(
                            lambda: asyncio.run(sandbox.evaluate(patch))
                        ).result(timeout=60)
                else:
                    report = asyncio.run(sandbox.evaluate(patch))

                bundle.sandbox_result_ref = f"sandbox:{code_bundle.bundle_id}"
                passed = getattr(report, "overall_passed", False)
                bundle.lane_verdicts["sandbox_validation"] = passed
                bundle.risk_assessment["sandbox_status"] = "passed" if passed else "failed"
                bundle.risk_assessment["sandbox_details"] = {
                    "overall_passed": passed,
                    "lint_passed": getattr(report, "lint_passed", None),
                    "test_passed": getattr(report, "test_passed", None),
                }
                if not passed:
                    bundle.overall_passed = False
                return True
            except Exception as sandbox_exc:
                bundle.risk_assessment["sandbox_status"] = "error"
                bundle.risk_assessment["sandbox_error"] = str(sandbox_exc)[:200]
                bundle.sandbox_result_ref = f"sandbox:error:{code_bundle.bundle_id}"
                return True

        except Exception as exc:
            bundle.risk_assessment["sandbox_status"] = "error"
            bundle.risk_assessment["sandbox_error"] = str(exc)[:200]
            logger.warning("Sandbox validation of bundle failed: %s", exc)
            return False

    def _run_skill_contract_on_bundle(
        self,
        job: CapabilityAcquisitionJob,
        bundle: Any,
        code_bundle: PluginCodeBundle,
    ) -> bool:
        """Run the linked skill smoke fixture against generated plugin code."""
        try:
            import importlib.util
            import tempfile
            from pathlib import Path as _Path
            from skills.execution_contracts import get_contract

            skill_id = (job.requested_by or {}).get("skill_id", "")
            contract = get_contract(skill_id) if skill_id else None
            if contract is None or not contract.smoke_fixtures:
                bundle.risk_assessment["skill_contract_status"] = "missing_contract_or_fixture"
                return False

            handler_src = code_bundle.code_files.get("handler.py", "")
            if not handler_src:
                bundle.risk_assessment["skill_contract_status"] = "missing_handler"
                return False

            with tempfile.TemporaryDirectory(prefix="jarvis_skill_contract_") as tmp:
                handler_path = _Path(tmp) / "handler.py"
                handler_path.write_text(handler_src, encoding="utf-8")
                spec = importlib.util.spec_from_file_location("_jarvis_candidate_handler", handler_path)
                if spec is None or spec.loader is None:
                    bundle.risk_assessment["skill_contract_status"] = "handler_import_unavailable"
                    return False
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                run = getattr(mod, "run", None)
                if not callable(run):
                    bundle.risk_assessment["skill_contract_status"] = "missing_run_callable"
                    return False

                results = []
                for fixture in contract.smoke_fixtures:
                    actual = run({
                        "text": fixture.input,
                        "input": fixture.input,
                        "request": fixture.input,
                        "input_type": fixture.input_type,
                    })
                    if isinstance(actual, dict) and "output" in actual and isinstance(actual["output"], dict):
                        actual = actual["output"]
                    passed = self._contract_expected_subset(fixture.expected, actual)
                    results.append({
                        "name": fixture.name,
                        "passed": passed,
                        "expected": fixture.expected,
                        "actual": actual,
                    })

            bundle.risk_assessment["skill_contract_status"] = "passed" if all(r["passed"] for r in results) else "failed"
            bundle.risk_assessment["skill_contract_results"] = results
            return all(r["passed"] for r in results)
        except Exception as exc:
            bundle.risk_assessment["skill_contract_status"] = "error"
            bundle.risk_assessment["skill_contract_error"] = str(exc)[:200]
            return False

    @staticmethod
    def _contract_expected_subset(expected: Any, actual: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for key, exp_val in expected.items():
                if key not in actual:
                    return False
                if not AcquisitionOrchestrator._contract_expected_subset(exp_val, actual[key]):
                    return False
            return True
        if isinstance(expected, list):
            return list(actual or []) == expected
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(expected) - float(actual)) < 1e-6
        return actual == expected

    # ── Hard activation gate ─────────────────────────────────────────

    # Minimum shadow observation periods per risk tier (seconds)
    _MIN_SHADOW_DURATION = {0: 300, 1: 1800, 2: 3600, 3: 3600}  # tier 0: 5m, tier 1: 30m, tier 2+: 1h

    def _can_activate(self, job: CapabilityAcquisitionJob, rec: Any) -> tuple[bool, str]:
        """Single decision gate for plugin promotion. Returns (allowed, reason)."""
        # 1. Verification must have completed and passed
        ver_lane = job.lanes.get("verification")
        if not ver_lane or ver_lane.status != "completed":
            return False, "verification_not_completed"

        if job.verification_id:
            bundle = self._store.load_verification(job.verification_id)
            if bundle and not bundle.overall_passed:
                return False, "verification_failed"
            if (
                bundle
                and (job.requested_by or {}).get("source") == "skill_operational_handoff"
                and not bundle.lane_verdicts.get("skill_contract_fixture")
            ):
                return False, "skill_contract_fixture_not_passed"

        # 2. Skill handoff plugins must prove the deployed shadow runtime can
        # execute the same fixture path that verification checked pre-deploy.
        if (
            rec
            and rec.state == "shadow"
            and (job.requested_by or {}).get("source") == "skill_operational_handoff"
        ):
            passed, reason = self._ensure_shadow_runtime_smoke(job, rec)
            if not passed:
                return False, reason

        # 3. Minimum shadow observation duration per risk tier
        if rec and rec.state == "shadow" and rec.activated_at:
            required = self._MIN_SHADOW_DURATION.get(job.risk_tier, 3600)
            elapsed = time.time() - rec.activated_at
            if elapsed < required:
                return False, f"shadow_observation_incomplete:{elapsed:.0f}/{required:.0f}s"

        # 4. Tier 2+ requires plan review to have been approved
        if job.risk_tier >= 2 and job.review_status not in ("reviewed", "not_required"):
            return False, "plan_review_required"

        # 5. Quarantine pressure must not be elevated
        try:
            from epistemic.quarantine.pressure import QuarantinePressure
            pressure = QuarantinePressure.instance()
            if pressure and pressure.composite > 0.3:
                return False, f"quarantine_pressure_elevated:{pressure.composite:.2f}"
        except Exception:
            pass

        return True, "all_gates_passed"

    def _ensure_shadow_runtime_smoke(self, job: CapabilityAcquisitionJob, rec: Any) -> tuple[bool, str]:
        """Invoke the deployed shadow plugin on the contract smoke fixture.

        Verification proves the generated handler's output. This observation proves
        the quarantined plugin was actually loaded and can run through the registry
        invocation path before activation may complete.
        """
        diagnostics = job.activation_diagnostics.setdefault("shadow_runtime_smoke", {})
        if diagnostics.get("passed"):
            return True, "shadow_runtime_smoke_passed"

        try:
            import asyncio
            import uuid
            from skills.execution_contracts import get_contract
            from tools.plugin_registry import PluginRequest, get_plugin_registry

            skill_id = (job.requested_by or {}).get("skill_id", "")
            contract = get_contract(skill_id) if skill_id else None
            if contract is None or not contract.smoke_fixtures:
                diagnostics.update({
                    "passed": False,
                    "error": "missing_contract_or_fixture",
                    "updated_at": time.time(),
                })
                return False, "shadow_runtime_smoke_failed"

            fixture = contract.smoke_fixtures[0]
            request = PluginRequest(
                request_id=f"shadow_smoke_{uuid.uuid4().hex[:10]}",
                plugin_name=rec.name,
                user_text=fixture.input,
                context={
                    "text": fixture.input,
                    "input": fixture.input,
                    "request": fixture.input,
                    "input_type": fixture.input_type,
                    "origin": "acquisition_shadow_smoke",
                    "acquisition_id": job.acquisition_id,
                    "skill_id": skill_id,
                    "fixture_name": fixture.name,
                },
                timeout_s=10.0,
            )
            registry = get_plugin_registry()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(
                        lambda: asyncio.run(registry.invoke(request))
                    ).result(timeout=20)
            else:
                response = asyncio.run(registry.invoke(request))

            diagnostics.update({
                "passed": bool(response.success),
                "request_id": request.request_id,
                "plugin_name": rec.name,
                "fixture_name": fixture.name,
                "invocation_count": getattr(rec, "invocation_count", 0),
                "success_count": getattr(rec, "success_count", 0),
                "failure_count": getattr(rec, "failure_count", 0),
                "duration_ms": round(getattr(response, "duration_ms", 0.0), 1),
                "error": response.error or "",
                "updated_at": time.time(),
            })
            if response.success:
                return True, "shadow_runtime_smoke_passed"
            return False, "shadow_runtime_smoke_failed"
        except Exception as exc:
            diagnostics.update({
                "passed": False,
                "error": str(exc)[:200],
                "updated_at": time.time(),
            })
            return False, "shadow_runtime_smoke_failed"

    def _run_plugin_activation(self, job: CapabilityAcquisitionJob) -> None:
        """Plugin activation: promote from quarantined through shadow to active.

        Uses the hard activation gate: verification passed, shadow observation met,
        approval satisfied (tier 2+), no elevated quarantine pressure.
        """
        try:
            from tools.plugin_registry import get_plugin_registry
            registry = get_plugin_registry()

            plugin_name = job.plugin_id
            if not plugin_name:
                job.fail_lane("plugin_activation", "No plugin to activate")
                return

            rec = registry.get_record(plugin_name)
            if not rec:
                job.fail_lane("plugin_activation", f"Plugin '{plugin_name}' not in registry")
                return

            terminal_denials = {
                "verification_failed",
                "skill_contract_fixture_not_passed",
                "shadow_runtime_smoke_failed",
            }

            allowed, reason = self._can_activate(job, rec)
            if not allowed and reason in terminal_denials:
                job.fail_lane("plugin_activation", reason)
                job.set_status("failed")
                self._total_failed += 1
                self._emit_event("ACQUISITION_FAILED", job, {
                    "lane": "plugin_activation",
                    "reason": reason,
                    "plugin_name": plugin_name,
                })
                return

            # Step 1: quarantined → shadow (only after non-fakeable proof passes)
            if rec.state == "quarantined":
                if not allowed:
                    logger.info(
                        "Plugin %s activation deferred: %s", plugin_name, reason,
                    )
                    return
                ok = registry.activate(plugin_name, "shadow")
                if not ok:
                    job.fail_lane("plugin_activation", "Activation to shadow failed")
                    return
                rec = registry.get_record(plugin_name)
                logger.info("Plugin %s activated to shadow, observation period started", plugin_name)
                return  # wait for shadow observation period on next tick

            # Step 2: shadow → supervised → active (gated)
            if rec and rec.state in ("shadow", "supervised"):
                if not allowed:
                    logger.info(
                        "Plugin %s activation deferred: %s", plugin_name, reason,
                    )
                    return  # re-check on next tick

                # Promote through remaining tiers
                if rec.state == "shadow":
                    registry.promote(plugin_name)  # shadow → supervised
                    rec = registry.get_record(plugin_name)
                if rec and rec.state == "supervised":
                    registry.promote(plugin_name)  # supervised → active
                    rec = registry.get_record(plugin_name)

            final_state = rec.state if rec else "unknown"
            job.complete_lane("plugin_activation", child_id=plugin_name)
            self._emit_event("ACQUISITION_PLUGIN_DEPLOYED", job, {
                "lane": "plugin_activation", "plugin_name": plugin_name,
                "state": final_state,
            })
        except Exception as exc:
            job.fail_lane("plugin_activation", str(exc))

    def _run_skill_registration(self, job: CapabilityAcquisitionJob) -> None:
        """Skill registration lane: create a LearningJob as child."""
        try:
            from skills.learning_jobs import LearningJobStore, LearningJobOrchestrator
            from skills.resolver import resolve_skill

            resolution = resolve_skill(job.user_intent)
            if resolution:
                # We don't create the job directly — we let the existing
                # skill system handle it with the parent_acquisition_id link
                job.complete_lane("skill_registration", child_id=resolution.skill_id)
                self._emit_event("ACQUISITION_LANE_COMPLETED", job, {
                    "lane": "skill_registration", "skill_id": resolution.skill_id,
                })
            else:
                job.complete_lane("skill_registration")
        except Exception as exc:
            job.fail_lane("skill_registration", str(exc))

    def _has_low_freshness_docs(self, job: CapabilityAcquisitionJob) -> bool:
        """Check if any doc artifacts have low freshness scores."""
        for doc_id in job.doc_artifact_ids:
            doc = self._store.load_doc(doc_id)
            if doc and doc.freshness_score < 0.5:
                return True
        return False

    # ── human approval API ─────────────────────────────────────────────

    def approve_plan(self, acquisition_id: str, verdict: str, notes: str = "",
                     suggested_changes: list[dict[str, Any]] | None = None,
                     reviewed_by: str = "human",
                     reason_category: str = "unknown") -> bool:
        """Handle human plan review decision."""
        from acquisition.job import PlanReviewArtifact
        job = self._active_jobs.get(acquisition_id)
        if not job or job.status != "awaiting_plan_review":
            return False

        plan = self._store.load_plan(job.plan_id) if job.plan_id else None
        plan_version = getattr(plan, "version", 0) if plan else 0
        if verdict in ("approved", "approved_as_is", "approved_with_edits") and plan is not None:
            quality_error = self._plan_quality_error(job, plan)
            if quality_error:
                self._record_planning_diagnostics(job, plan, quality_error)
                self._store.save_job(job)
                logger.warning(
                    "Refusing approval for incomplete plan %s (%s): %s",
                    job.plan_id, acquisition_id, quality_error,
                )
                return False

        review = PlanReviewArtifact(
            acquisition_id=acquisition_id,
            plan_id=job.plan_id,
            verdict=verdict,
            reason_category=reason_category,
            operator_notes=notes,
            suggested_changes=suggested_changes or [],
            reviewed_by=reviewed_by,
            plan_version=plan_version,
        )
        self._store.save_review(review)
        job.plan_review_id = review.artifact_id
        job.add_artifact_ref(review.artifact_id)

        self._record_plan_verdict_signal(job, plan, verdict, reason_category, plan_version)
        self._resolve_shadow_prediction(job, verdict, plan_version, reason_category)

        from consciousness.events import ACQUISITION_PLAN_REVIEWED
        self._safe_emit(ACQUISITION_PLAN_REVIEWED, {
            "acquisition_id": acquisition_id,
            "plan_id": job.plan_id,
            "plan_version": plan_version,
            "verdict": verdict,
            "reason_category": reason_category,
            "reviewed_by": reviewed_by,
        })

        plan_review_lane = job.lanes.get("plan_review")
        if plan_review_lane:
            plan_review_lane.status = "completed"
            plan_review_lane.completed_at = time.time()

        if verdict in ("approved", "approved_as_is", "approved_with_edits"):
            job.review_status = "reviewed"
            job.set_status("executing")
        elif verdict == "rejected":
            # Reset planning + plan_review lanes for re-planning with feedback
            planning_lane = job.lanes.get("planning")
            if planning_lane:
                planning_lane.status = "pending"
                planning_lane.started_at = 0.0
                planning_lane.completed_at = 0.0
                planning_lane.error = ""
            if plan_review_lane:
                plan_review_lane.status = "pending"
                plan_review_lane.started_at = 0.0
                plan_review_lane.completed_at = 0.0
                plan_review_lane.error = ""
            job.set_status("planning")
            logger.info(
                "Plan rejected for %s (v%d): re-planning with operator feedback: %s",
                acquisition_id, plan_version, notes[:120] if notes else "(no notes)",
            )
        elif verdict == "cancelled":
            job.set_status("cancelled")
        else:
            job.set_status("executing")

        self._store.save_job(job)
        return True

    def approve_deployment(self, acquisition_id: str, approved: bool,
                           approved_by: str = "human") -> bool:
        """Handle human deployment approval decision."""
        job = self._active_jobs.get(acquisition_id)
        if not job or job.status != "awaiting_approval":
            return False

        if approved:
            job.approval_status = "approved"
            job.approved_by = approved_by
            job.approval_timestamp = time.time()
            deployment_lane = job.lanes.get("deployment")
            if deployment_lane:
                deployment_lane.status = "completed"
                deployment_lane.completed_at = time.time()
            job.set_status("deployed")
        else:
            job.approval_status = "denied"
            job.set_status("cancelled")

        from consciousness.events import ACQUISITION_DEPLOYMENT_REVIEWED
        self._safe_emit(ACQUISITION_DEPLOYMENT_REVIEWED, {
            "acquisition_id": acquisition_id,
            "approved": approved,
            "approved_by": approved_by,
        })

        self._store.save_job(job)
        return True

    def cancel_job(self, acquisition_id: str, reason: str = "operator_cancelled") -> bool:
        """Cancel and remove a job from the active set (any state)."""
        job = self._active_jobs.get(acquisition_id)
        if not job:
            return False
        job.set_status("cancelled")
        self._store.save_job(job)
        del self._active_jobs[acquisition_id]
        logger.info("Acquisition %s cancelled and removed: %s", acquisition_id, reason)
        return True

    # ── query ──────────────────────────────────────────────────────────

    def get_job(self, acquisition_id: str) -> CapabilityAcquisitionJob | None:
        job = self._active_jobs.get(acquisition_id)
        if job:
            return job
        return self._store.load_job(acquisition_id)

    # Per-status stall thresholds (seconds)
    _STALL_THRESHOLDS: dict[str, float] = {
        "running": 1800,           # active execution lanes: 30 min
        "pending": 600,            # background-safe lanes: 10 min
        "awaiting_plan_review": 86400,  # review gates: 24h
        "awaiting_approval": 86400,
    }

    def _compute_stall_info(self, job: CapabilityAcquisitionJob) -> dict[str, Any] | None:
        """Detect and describe stalled acquisition jobs."""
        now = time.time()

        # Find the current lane (first non-completed, non-skipped)
        current_lane = None
        current_ls = None
        next_lane = None
        for i, lane_name in enumerate(job.required_lanes):
            ls = job.lanes.get(lane_name)
            if ls is None:
                continue
            if ls.status in ("completed", "skipped"):
                continue
            if current_lane is None:
                current_lane = lane_name
                current_ls = ls
                # Find next expected lane
                for j_lane in job.required_lanes[i + 1:]:
                    j_ls = job.lanes.get(j_lane)
                    if j_ls and j_ls.status not in ("completed", "skipped"):
                        next_lane = j_lane
                        break
                break

        if current_lane is None or current_ls is None:
            return None

        # Determine time in current state
        ref_time = current_ls.started_at or current_ls.completed_at or job.updated_at
        elapsed = now - ref_time if ref_time else 0

        # Determine blocked reason
        blocked_reason = ""
        observation_info: dict[str, Any] | None = None
        if job.status == "awaiting_plan_review":
            blocked_reason = "awaiting_human_plan_review"
        elif job.status == "awaiting_approval":
            blocked_reason = "awaiting_human_deployment_approval"
        elif current_ls.status == "pending" and not self._lane_allowed_in_mode(current_lane):
            blocked_reason = f"mode_restriction:{self._current_mode}"
        elif current_ls.status == "failed":
            blocked_reason = f"lane_failed:{current_ls.error[:60]}" if current_ls.error else "lane_failed"
        elif current_lane == "plugin_activation" and current_ls.status == "running" and job.plugin_id:
            try:
                from tools.plugin_registry import get_plugin_registry
                rec = get_plugin_registry().get_record(job.plugin_id)
                if rec and getattr(rec, "state", "") == "shadow" and getattr(rec, "activated_at", 0):
                    required = self._MIN_SHADOW_DURATION.get(job.risk_tier, 3600)
                    observed = max(0.0, now - rec.activated_at)
                    remaining = max(0.0, required - observed)
                    if remaining > 0:
                        blocked_reason = "shadow_observation_window"
                        observation_info = {
                            "required_s": round(required),
                            "observed_s": round(observed),
                            "remaining_s": round(remaining),
                            "runtime_smoke_passed": bool(
                                (job.activation_diagnostics.get("shadow_runtime_smoke") or {}).get("passed")
                            ),
                        }
            except Exception:
                pass

        # Determine threshold and severity
        threshold_key = job.status if job.status in self._STALL_THRESHOLDS else current_ls.status
        threshold = self._STALL_THRESHOLDS.get(threshold_key, 1800)
        if observation_info:
            threshold = observation_info["required_s"]
        is_stalled = elapsed > threshold
        if observation_info:
            is_stalled = False

        if not is_stalled and not blocked_reason:
            return None

        severity = "info"
        if blocked_reason:
            severity = "info"
        if is_stalled and threshold_key in ("running", "pending"):
            severity = "error" if elapsed > threshold * 3 else "warn"
        elif is_stalled:
            severity = "warn"

        return {
            "acquisition_id": job.acquisition_id,
            "title": job.title,
            "current_lane": current_lane,
            "lane_status": current_ls.status,
            "next_expected_lane": next_lane,
            "elapsed_s": round(elapsed),
            "threshold_s": round(threshold),
            "blocked_reason": blocked_reason,
            "severity": severity,
            "is_stalled": is_stalled,
            "observation": observation_info,
        }

    def get_status(self) -> dict[str, Any]:
        """Status for dashboard snapshot."""
        summary = self._store.get_summary()
        summary["classifier"] = self._classifier.get_status()
        summary["runtime"] = {
            "active_in_memory": len(self._active_jobs),
            "total_created": self._total_created,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
        }
        summary["scheduler"] = {
            "current_mode": self._current_mode,
            "background_safe_lanes": sorted(_BACKGROUND_SAFE_LANES),
            "deferred_lanes": sorted(
                ln for j in self._active_jobs.values()
                for ln, ls in j.lanes.items()
                if ls.status == "pending" and not self._lane_allowed_in_mode(ln)
            ),
            "quarantine_pressure": round(self._quarantine_pressure, 3),
            "pressure_level": self._pressure_level,
            "suppressed_lanes": sorted(self._suppressed_lanes),
        }

        # Pending approvals
        pending_approvals = []
        for j in self._active_jobs.values():
            if j.status not in ("awaiting_plan_review", "awaiting_approval"):
                continue
            entry: dict[str, Any] = {
                "acquisition_id": j.acquisition_id,
                "title": j.title,
                "status": j.status,
                "risk_tier": j.risk_tier,
                "outcome_class": j.outcome_class,
                "gate": "plan_review" if j.status == "awaiting_plan_review" else "deployment",
                "planning_diagnostics": j.planning_diagnostics or {},
                "codegen_prompt_diagnostics": j.codegen_prompt_diagnostics or {},
            }
            if j.plan_id:
                plan = self._store.load_plan(j.plan_id)
                if plan:
                    entry["plan_summary"] = {
                        "objective": plan.objective,
                        "technical_approach": plan.technical_approach or "",
                        "risk_analysis": plan.risk_analysis or "",
                        "risk_level": plan.risk_level or "",
                        "dependencies": plan.dependencies or [],
                        "test_cases": plan.test_cases or [],
                        "implementation_sketch": plan.implementation_sketch or "",
                        "doc_count": len(plan.doc_artifact_ids or []),
                        "version": plan.version,
                    }
                    # Include prior rejection feedback if this is a revision
                    if j.plan_review_id:
                        prev_review = self._store.load_review(j.plan_review_id)
                        if prev_review and getattr(prev_review, "verdict", "") == "rejected":
                            entry["plan_summary"]["prior_rejection"] = {
                                "notes": getattr(prev_review, "operator_notes", "") or "",
                                "category": getattr(prev_review, "reason_category", ""),
                                "changes": getattr(prev_review, "suggested_changes", []) or [],
                            }
            if j.status == "awaiting_approval" and j.verification_id:
                vb = self._store.load_verification(j.verification_id)
                if vb:
                    entry["verification_summary"] = {
                        "overall_passed": vb.overall_passed,
                        "lane_verdicts": vb.lane_verdicts,
                    }
            pending_approvals.append(entry)
        summary["pending_approvals"] = pending_approvals

        # Stall detection
        stalls = []
        for j in self._active_jobs.values():
            info = self._compute_stall_info(j)
            if info:
                stalls.append(info)
        summary["stalled_jobs"] = stalls

        return summary

    # ── internal helpers ───────────────────────────────────────────────

    def _apply_risk_tier(self, job: CapabilityAcquisitionJob) -> None:
        tier = job.risk_tier
        if tier == 0:
            job.review_status = "not_required"
            job.approval_status = "not_required"
            job.supervision_mode = "none"
        elif tier == 1:
            job.review_status = "not_required"   # conditional — checked at dispatch
            job.approval_status = "pending"
            job.supervision_mode = "shadow"
        elif tier == 2:
            job.review_status = "pending"
            job.approval_status = "pending"
            job.supervision_mode = "supervised"
        elif tier >= 3:
            job.review_status = "pending"
            job.approval_status = "pending"
            job.supervision_mode = "bounded"

    def _record_ledger_entry(self, job: CapabilityAcquisitionJob, event_type: str) -> None:
        """Record acquisition event in attribution ledger."""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            eid = attribution_ledger.record(
                subsystem="acquisition",
                event_type=event_type,
                actor="acquisition_orchestrator",
                source=job.user_intent[:200],
                confidence=job.classification_confidence,
                data={
                    "acquisition_id": job.acquisition_id,
                    "outcome_class": job.outcome_class,
                    "risk_tier": job.risk_tier,
                },
            )
            if not job.ledger_entry_id:
                job.ledger_entry_id = eid
        except Exception as exc:
            logger.debug("Attribution ledger record failed: %s", exc)

    @staticmethod
    def _safe_emit(event_name: str, data: dict[str, Any]) -> None:
        """Emit an event with proper kwarg expansion."""
        try:
            event_bus.emit(event_name, **data)
        except Exception:
            logger.debug("Failed to emit %s", event_name, exc_info=True)

    def _emit_event(self, event_suffix: str, job: CapabilityAcquisitionJob,
                    extra: dict[str, Any] | None = None) -> None:
        """Emit an acquisition event via the event bus."""
        try:
            event_name = f"acquisition:{event_suffix.lower().removeprefix('acquisition_')}"
            data = {
                "acquisition_id": job.acquisition_id,
                "title": job.title,
                "status": job.status,
                "outcome_class": job.outcome_class,
                **(extra or {}),
            }
            event_bus.emit(event_name, **data)
        except Exception:
            pass

    # ── Plan evaluator distillation signals ────────────────────────────

    def _record_plan_features_signal(
        self, job: CapabilityAcquisitionJob, plan: Any, plan_version: int,
    ) -> None:
        """Record the plan feature vector to the distillation collector (input half)."""
        try:
            from acquisition.plan_encoder import PlanEvaluatorEncoder
            vec = PlanEvaluatorEncoder.encode(job, plan)
            from hemisphere.distillation import DistillationCollector
            collector = DistillationCollector.instance()
            if collector:
                collector.record(
                    teacher="plan_features",
                    signal_type="plan_features",
                    data=vec,
                    metadata={
                        "acquisition_id": job.acquisition_id,
                        "plan_id": getattr(plan, "plan_id", ""),
                        "plan_version": plan_version,
                    },
                    origin="acquisition",
                    fidelity=1.0,
                )
        except Exception as exc:
            logger.debug("Plan features signal recording failed: %s", exc)

    def _record_plan_verdict_signal(
        self, job: CapabilityAcquisitionJob, plan: Any,
        verdict: str, reason_category: str, plan_version: int,
    ) -> None:
        """Record the verdict label to the distillation collector (label half)."""
        try:
            from acquisition.plan_encoder import encode_verdict
            label = encode_verdict(verdict)
            from hemisphere.distillation import DistillationCollector
            collector = DistillationCollector.instance()
            if collector:
                collector.record(
                    teacher="acquisition_planner",
                    signal_type="verdict",
                    data=label,
                    metadata={
                        "acquisition_id": job.acquisition_id,
                        "plan_id": getattr(plan, "plan_id", "") if plan else "",
                        "plan_version": plan_version,
                        "verdict": verdict,
                        "reason_category": reason_category,
                    },
                    origin="human_review",
                    fidelity=1.0,
                )
        except Exception as exc:
            logger.debug("Plan verdict signal recording failed: %s", exc)

    def _record_skill_acquisition_feature(
        self,
        job: CapabilityAcquisitionJob,
        plan: Any | None = None,
        verification: Any | None = None,
        *,
        stage: str = "",
        origin: str = "acquisition",
        fidelity: float = 1.0,
    ) -> None:
        """Record shadow-only skill acquisition features for NN training."""
        try:
            from acquisition.skill_acquisition_encoder import SkillAcquisitionEncoder
            from hemisphere.distillation import DistillationCollector
            if plan is None and job.plan_id:
                plan = self._store.load_plan(job.plan_id)
            vec = SkillAcquisitionEncoder.encode(job, plan, verification)
            collector = DistillationCollector.instance()
            if collector:
                collector.record(
                    teacher="skill_acquisition_features",
                    signal_type="skill_acquisition",
                    data=vec,
                    metadata={
                        "episode_id": job.acquisition_id,
                        "acquisition_id": job.acquisition_id,
                        "skill_id": (job.requested_by or {}).get("skill_id", ""),
                        "stage": stage,
                    },
                    origin=origin,
                    fidelity=fidelity,
                )
        except Exception as exc:
            logger.debug("Skill acquisition feature recording failed: %s", exc)

    def _record_skill_acquisition_label(
        self,
        job: CapabilityAcquisitionJob,
        verification: Any | None = None,
        *,
        origin: str = "acquisition",
        fidelity: float = 1.0,
    ) -> None:
        """Record final/terminal label for the skill acquisition specialist."""
        try:
            from acquisition.skill_acquisition_encoder import (
                SkillAcquisitionEncoder,
                outcome_from_state,
            )
            from hemisphere.distillation import DistillationCollector
            outcome = outcome_from_state(job, verification)
            collector = DistillationCollector.instance()
            if collector:
                collector.record(
                    teacher="skill_acquisition_outcome",
                    signal_type="skill_acquisition",
                    data=SkillAcquisitionEncoder.encode_label(outcome),
                    metadata={
                        "episode_id": job.acquisition_id,
                        "acquisition_id": job.acquisition_id,
                        "skill_id": (job.requested_by or {}).get("skill_id", ""),
                        "outcome": outcome,
                    },
                    origin=origin,
                    fidelity=fidelity,
                )
        except Exception as exc:
            logger.debug("Skill acquisition label recording failed: %s", exc)

    def _run_shadow_prediction(
        self, job: CapabilityAcquisitionJob, plan: Any,
    ) -> None:
        """Run the plan_evaluator specialist in shadow mode and persist the prediction."""
        try:
            from acquisition.plan_encoder import (
                PlanEvaluatorEncoder, ShadowPredictionArtifact, label_to_class,
            )
            vec = PlanEvaluatorEncoder.encode(job, plan)
            plan_version = getattr(plan, "version", 0)

            from hemisphere.registry import HemisphereRegistry
            from hemisphere.engine import HemisphereEngine

            registry = HemisphereRegistry()
            active_meta = registry.get_active("plan_evaluator")
            if active_meta is None:
                return

            network_id = getattr(active_meta, "network_id", None) or getattr(active_meta, "id", None)
            if not network_id:
                return

            engine = HemisphereEngine.get_instance() if hasattr(HemisphereEngine, "get_instance") else None
            if engine is None:
                return

            probs = engine.infer(network_id, vec)
            if not probs or len(probs) < 3:
                return

            pred_class = label_to_class(probs)
            model_ver = ""
            try:
                meta = registry.get_active("plan_evaluator")
                model_ver = getattr(meta, "id", "") if meta else ""
            except Exception:
                pass
            artifact = ShadowPredictionArtifact(
                acquisition_id=job.acquisition_id,
                plan_id=getattr(plan, "plan_id", ""),
                plan_version=plan_version,
                predicted_probs=probs,
                predicted_class=pred_class,
                feature_vector=vec,
                model_version=model_ver,
                risk_tier=job.risk_tier,
                outcome_class=job.outcome_class,
            )
            self._store_shadow_prediction(artifact)
            logger.info(
                "Shadow prediction for %s: %s (probs=%.2f/%.2f/%.2f)",
                job.acquisition_id, pred_class, *probs[:3],
            )
        except Exception as exc:
            logger.debug("Shadow prediction failed (expected if specialist not trained): %s", exc)

    def _store_shadow_prediction(self, artifact: Any) -> None:
        """Persist shadow prediction artifact to disk."""
        try:
            from pathlib import Path
            shadow_dir = Path.home() / ".jarvis" / "acquisition_shadows"
            shadow_dir.mkdir(parents=True, exist_ok=True)
            key = f"{artifact.acquisition_id}_{artifact.plan_id}_v{artifact.plan_version}"
            path = shadow_dir / f"{key}.json"
            from memory.persistence import atomic_write_json
            atomic_write_json(path, artifact.to_dict())
        except Exception as exc:
            logger.debug("Shadow prediction persistence failed: %s", exc)

    def _load_shadow_prediction(self, acquisition_id: str, plan_id: str, plan_version: int) -> Any:
        """Load a previously stored shadow prediction."""
        try:
            from pathlib import Path
            from acquisition.plan_encoder import ShadowPredictionArtifact
            key = f"{acquisition_id}_{plan_id}_v{plan_version}"
            path = Path.home() / ".jarvis" / "acquisition_shadows" / f"{key}.json"
            if not path.exists():
                return None
            import json as _json
            d = _json.loads(path.read_text())
            return ShadowPredictionArtifact.from_dict(d)
        except Exception:
            return None

    def _resolve_shadow_prediction(
        self, job: CapabilityAcquisitionJob, verdict: str, plan_version: int,
        reason_category: str = "unknown",
    ) -> None:
        """Compare shadow prediction with actual verdict and log accuracy."""
        try:
            from acquisition.plan_encoder import verdict_to_class
            shadow = self._load_shadow_prediction(job.acquisition_id, job.plan_id, plan_version)
            if shadow is None:
                return
            actual_class = verdict_to_class(verdict)
            shadow.actual_verdict = verdict
            shadow.actual_class = actual_class
            shadow.reason_category = reason_category
            shadow.correct = (shadow.predicted_class == actual_class)
            shadow.reviewed_at = time.time()
            self._store_shadow_prediction(shadow)
            logger.info(
                "Shadow accuracy for %s: predicted=%s actual=%s correct=%s",
                job.acquisition_id, shadow.predicted_class, actual_class, shadow.correct,
            )
        except Exception as exc:
            logger.debug("Shadow resolution failed: %s", exc)


def _text_overlap(a: str, b: str) -> bool:
    """Check if two title strings have meaningful word overlap."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    stop = {"a", "an", "the", "to", "how", "learn", "about", "do", "is", "what"}
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap >= 0.4
