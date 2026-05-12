"""Orchestrator -- ties the self-improvement loop together.

Loop: detect need -> think (plan) -> code (generate patch) -> validate (sandbox)
      -> retry if failed (up to MAX_ITERATIONS) -> atomic apply -> health monitor
      -> promote or rollback.

All file writes are atomic: write to temp, validate, swap into place.
Rollback snapshots are stored in ~/.jarvis/improvement_snapshots/.

Stage system (SELF_IMPROVE_STAGE):
  0 = frozen       — auto-triggers blocked (legacy FREEZE_AUTO_IMPROVE compat)
  1 = dry-run      — full pipeline runs but nothing applied to disk
  2 = human-approval — patches require dashboard approve before apply
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import os
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from consciousness.events import (
    IMPROVEMENT_STARTED,
    IMPROVEMENT_VALIDATED,
    IMPROVEMENT_PROMOTED,
    IMPROVEMENT_ROLLED_BACK,
    IMPROVEMENT_NEEDS_APPROVAL,
    IMPROVEMENT_DRY_RUN,
    IMPROVEMENT_SANDBOX_PASSED,
    IMPROVEMENT_SANDBOX_FAILED,
    IMPROVEMENT_POST_RESTART_VERIFIED,
)
from self_improve.code_patch import CodePatch, FileDiff
from self_improve.conversation import (
    ImprovementConversation,
    MAX_ITERATIONS,
    _save_conversation,
    _load_recent_conversations,
)
from self_improve.evaluation_report import EvaluationReport
from self_improve.improvement_request import ImprovementRequest
from self_improve.patch_plan import PatchPlan, MAX_FILES_CHANGED
from self_improve.provider import PatchProvider
from self_improve.sandbox import Sandbox, SIM_TICKS

logger = logging.getLogger(__name__)

MAX_HISTORY = 50
MIN_INTERVAL_S = 600.0
SNAPSHOT_DIR = Path("~/.jarvis/improvement_snapshots").expanduser()
HISTORY_FILE = Path("~/.jarvis/improvements.json").expanduser()
PROPOSALS_FILE = Path("~/.jarvis/improvement_proposals.jsonl").expanduser()
HEALTH_MONITOR_TICKS = 50
P95_REGRESSION_THRESHOLD = 1.2  # 20% increase

STAGE_FROZEN = 0
STAGE_DRY_RUN = 1
STAGE_HUMAN_APPROVAL = 2

SOUL_INTEGRITY_GATE_THRESHOLD = 0.50
ACTIVITY_LOG_MAX = 200

STAGE_FILE = Path("~/.jarvis/self_improve_stage.json").expanduser()
PENDING_APPROVALS_FILE = Path("~/.jarvis/pending_approvals.json").expanduser()


def _load_persisted_stage() -> int | None:
    """Load persisted stage from disk. Returns None if not found or invalid."""
    try:
        if STAGE_FILE.exists():
            data = json.loads(STAGE_FILE.read_text())
            stage = data.get("stage")
            if isinstance(stage, int) and stage in (STAGE_FROZEN, STAGE_DRY_RUN, STAGE_HUMAN_APPROVAL):
                return stage
    except Exception:
        logger.debug("Could not load persisted stage from %s", STAGE_FILE)
    return None


def _save_persisted_stage(stage: int) -> None:
    """Persist stage to disk so it survives restarts."""
    try:
        STAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=str(STAGE_FILE.parent), suffix=".tmp", delete=False,
        )
        json.dump({"stage": stage, "updated_at": time.time()}, tmp)
        tmp.close()
        os.replace(tmp.name, str(STAGE_FILE))
    except Exception:
        logger.debug("Could not persist stage to %s", STAGE_FILE)


# ---------------------------------------------------------------------------
# Win-rate tracker
# ---------------------------------------------------------------------------


@dataclass
class ImprovementWinRate:
    total: int = 0
    sandbox_passed: int = 0
    sandbox_failed: int = 0
    review_approved: int = 0
    review_rejected: int = 0

    @property
    def sandbox_pass_rate(self) -> float:
        return self.sandbox_passed / max(1, self.total)

    @property
    def review_approval_rate(self) -> float:
        reviewed = self.review_approved + self.review_rejected
        return self.review_approved / max(1, reviewed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "sandbox_passed": self.sandbox_passed,
            "sandbox_failed": self.sandbox_failed,
            "review_approved": self.review_approved,
            "review_rejected": self.review_rejected,
            "sandbox_pass_rate": round(self.sandbox_pass_rate, 3),
            "review_approval_rate": round(self.review_approval_rate, 3),
        }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ImprovementRecord:
    request: ImprovementRequest
    upgrade_id: str = ""
    current_attempt_id: str = ""
    plan: PatchPlan | None = None
    patch: CodePatch | None = None
    report: EvaluationReport | None = None
    review_result: dict[str, Any] | None = None
    conversation_id: str = ""
    iterations: int = 0
    status: Literal[
        "pending", "thinking", "coding", "testing", "reviewing",
        "awaiting_approval", "verifying", "promoted", "failed", "rolled_back",
        "dry_run",
    ] = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    snapshot_path: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SelfImprovementOrchestrator:
    def __init__(
        self,
        engine: Any = None,
        restart_callback: Any = None,
        dry_run_mode: bool = False,
        provider: PatchProvider | None = None,
    ) -> None:
        self._provider = provider or PatchProvider()
        self._sandbox = Sandbox()
        self._history: deque[ImprovementRecord] = deque(maxlen=MAX_HISTORY)
        self._pending_approvals: list[ImprovementRecord] = []
        self._last_attempt_time: float = 0.0
        self._total_improvements: int = 0
        self._total_rollbacks: int = 0
        self._total_failures: int = 0
        self._paused: bool = False
        self._engine = engine
        self._restart_callback = restart_callback
        self._ollama_client: Any = None
        self._win_rate = ImprovementWinRate()
        self._activity_log: deque[dict[str, Any]] = deque(maxlen=ACTIVITY_LOG_MAX)

        # Stage system: replaces binary FREEZE_AUTO_IMPROVE
        self._stage, self._stage_source = self._resolve_stage()
        self._auto_frozen: bool = self._stage == STAGE_FROZEN

        provider_status = self._provider.get_status()
        coder_available = provider_status.get("coder", {}).get("available", False)
        local_available = provider_status.get("local_available", True)
        self._has_reliable_provider = (
            coder_available
            or local_available
            or provider_status["claude_available"]
            or provider_status["openai_available"]
        )
        if not self._has_reliable_provider and not dry_run_mode:
            logger.warning(
                "No reliable codegen provider (CoderServer/local Ollama/cloud plugins) — "
                "self-improve forced to dry-run mode until a provider is configured"
            )
            dry_run_mode = True
        self._dry_run_mode: bool = dry_run_mode

        self._load_history()
        self._load_pending_approvals()

    @staticmethod
    def _resolve_stage() -> tuple[int, str]:
        """Determine the improvement stage from environment.

        Priority: SELF_IMPROVE_STAGE env var > persisted file > FREEZE_AUTO_IMPROVE > default (0).
        Returns (stage, source) where source is one of:
        "env_var", "persisted", "freeze_flag", "default".
        """
        stage_raw = os.environ.get("SELF_IMPROVE_STAGE", "").strip()
        if stage_raw:
            try:
                stage = int(stage_raw)
                if stage in (STAGE_FROZEN, STAGE_DRY_RUN, STAGE_HUMAN_APPROVAL):
                    logger.info("Self-improvement stage set to %d (from env var)", stage)
                    return stage, "env_var"
                logger.warning("Invalid SELF_IMPROVE_STAGE=%s, defaulting to 0", stage_raw)
            except ValueError:
                logger.warning("Invalid SELF_IMPROVE_STAGE=%s, defaulting to 0", stage_raw)

        persisted = _load_persisted_stage()
        if persisted is not None:
            logger.info("Self-improvement stage set to %d (from persisted file)", persisted)
            return persisted, "persisted"

        freeze = os.environ.get("FREEZE_AUTO_IMPROVE", "true").lower()
        if freeze in ("true", "1", "yes"):
            return STAGE_FROZEN, "freeze_flag"
        return STAGE_DRY_RUN, "default"

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        logger.info("Self-improvement %s", "paused" if paused else "resumed")

    def is_blocked(self) -> bool:
        """Check if the orchestrator should refuse new work.

        True when paused, during verification, or during cooldown.
        External callers can use this to skip opportunity detection entirely.
        """
        if self._paused:
            return True
        from self_improve.verification import has_pending
        if has_pending():
            return True
        if time.time() - self._last_attempt_time < MIN_INTERVAL_S:
            return True
        return False

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def set_stage(self, stage: int) -> dict[str, Any]:
        """Promote or demote the self-improvement stage at runtime.

        Returns a status dict with old_stage, new_stage, and any warnings.
        Stage 0 = frozen, 1 = dry-run, 2 = human-approval.
        """
        if stage not in (STAGE_FROZEN, STAGE_DRY_RUN, STAGE_HUMAN_APPROVAL):
            return {"error": f"Invalid stage: {stage}, must be 0, 1, or 2"}
        old = self._stage
        if stage == old:
            return {"old_stage": old, "new_stage": old, "changed": False}

        warnings: list[str] = []
        if stage == STAGE_HUMAN_APPROVAL and not self._has_reliable_provider:
            warnings.append("no_reliable_provider")

        self._stage = stage
        self._stage_source = "runtime_api"
        self._auto_frozen = (stage == STAGE_FROZEN)
        _save_persisted_stage(stage)
        labels = {STAGE_FROZEN: "frozen", STAGE_DRY_RUN: "dry-run", STAGE_HUMAN_APPROVAL: "human-approval"}
        self._log("info", "stage", f"Stage changed: {labels.get(old, '?')} → {labels.get(stage, '?')}")
        logger.info("Self-improvement stage changed: %d -> %d (persisted)%s",
                     old, stage, f" (warnings: {warnings})" if warnings else "")
        return {
            "old_stage": old,
            "new_stage": stage,
            "changed": True,
            "warnings": warnings,
            "stage_source": "runtime_api",
            "stage_label": {
                STAGE_FROZEN: "frozen",
                STAGE_DRY_RUN: "dry_run",
                STAGE_HUMAN_APPROVAL: "human_approval",
            }.get(stage, "unknown"),
        }

    def set_auto_frozen(self, frozen: bool) -> None:
        self._auto_frozen = frozen
        if frozen:
            self._stage = STAGE_FROZEN
        elif self._stage == STAGE_FROZEN:
            self._stage = STAGE_DRY_RUN
        self._stage_source = "runtime_auto_freeze"
        logger.info("Auto self-improvement %s (stage=%d)", "frozen" if frozen else "unfrozen", self._stage)

    def set_ollama_client(self, client: Any) -> None:
        """Wire the Ollama client for multi-turn retry in auto-triggered improvements."""
        self._ollama_client = client

    def _log(self, level: str, phase: str, message: str, **extra: Any) -> None:
        """Append a timestamped entry to the activity ring buffer.

        Levels: info, warn, error, success, thinking.
        Phases: scanner, plan, codegen, sandbox, review, approval, apply, health, rollback, stage.
        """
        entry: dict[str, Any] = {
            "ts": time.time(),
            "level": level,
            "phase": phase,
            "message": message,
        }
        if extra:
            entry["detail"] = {k: v for k, v in extra.items() if v is not None}
        self._activity_log.append(entry)

    def get_activity_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most recent activity log entries (newest first)."""
        entries = list(self._activity_log)
        return list(reversed(entries[-limit:]))

    async def attempt_improvement(
        self,
        request: ImprovementRequest,
        ollama_client: Any = None,
        dry_run: bool = False,
        manual: bool = False,
    ) -> ImprovementRecord:
        """Run the full improvement pipeline with multi-turn retry loop.

        When *dry_run* is True the pipeline runs through sandbox evaluation
        and optional review but stops before applying anything to disk.
        The returned record contains the full plan, patch diffs, and
        sandbox report so the caller can inspect what *would* have happened.

        When *manual* is True, the auto-freeze gate is bypassed (explicit
        user/API triggers always go through).
        """
        dry_run = dry_run or self._dry_run_mode
        request.manual = manual

        if self._paused:
            logger.info("Self-improvement paused (gestation active), skipping: %s", request.description[:60])
            return ImprovementRecord(
                request=request,
                status="failed",
                iterations=0,
            )

        # Stage gate (replaces binary _auto_frozen check)
        if self._stage == STAGE_FROZEN and not manual:
            logger.info("Self-improvement stage=0 (frozen), skipping: %s", request.description[:60])
            return ImprovementRecord(
                request=request,
                status="failed",
                iterations=0,
            )

        # Stage 1 safety invariant: force dry-run regardless of caller
        if self._stage == STAGE_DRY_RUN:
            dry_run = True

        # Quarantine pressure gate
        gate_reason = self._check_safety_gates()
        if gate_reason and not manual:
            logger.info("Self-improvement blocked by safety gate (%s): %s",
                        gate_reason, request.description[:60])
            return ImprovementRecord(
                request=request,
                status="failed",
                iterations=0,
            )

        if not request.target_module or not request.target_module.strip():
            logger.info("Self-improvement request has no target_module, rejecting: %s",
                        request.description[:60])
            return ImprovementRecord(
                request=request,
                status="failed",
                iterations=0,
            )

        from self_improve.verification import has_pending
        if has_pending():
            logger.info("Pending verification exists — refusing new improvement attempts")
            return ImprovementRecord(
                request=request,
                status="failed",
                iterations=0,
            )

        now = time.time()
        if not manual and now - self._last_attempt_time < MIN_INTERVAL_S:
            logger.info("Improvement cooldown active (%.0fs remaining)",
                        MIN_INTERVAL_S - (now - self._last_attempt_time))
            record = ImprovementRecord(request=request, status="failed")
            self._history.append(record)
            return record

        self._last_attempt_time = now
        mode = "dry-run" if dry_run else ("manual" if manual else "auto")
        self._log("info", "pipeline", f"Starting improvement: {request.description[:80]}", mode=mode, target=request.target_module)

        conv = ImprovementConversation(
            id=f"conv_{uuid.uuid4().hex[:10]}",
            request_description=request.description,
        )

        record = ImprovementRecord(
            request=request,
            status="thinking",
            conversation_id=conv.id,
        )
        self._history.append(record)
        from self_improve.system_upgrade_report import mint_upgrade_id, sync_report_from_record
        record.upgrade_id = mint_upgrade_id()
        qp0, si0 = self._si_context_pressures()
        sync_report_from_record(
            record, dry_run=dry_run, quarantine_pressure=qp0, soul_integrity=si0,
        )

        _started_eid = ""
        _started_ts = time.time()
        try:
            started_payload = request.to_dict()
            started_payload["upgrade_id"] = record.upgrade_id
            self._emit_event(IMPROVEMENT_STARTED, started_payload)
            try:
                from consciousness.attribution_ledger import attribution_ledger
                _started_eid = attribution_ledger.record(
                    subsystem="self_improve",
                    event_type="improvement_started",
                    data={"description": request.description[:200], "target_module": getattr(request, "target_module", ""),
                          "priority": getattr(request, "priority", "medium")},
                    evidence_refs=[{"kind": "improvement", "id": request.id}],
                )
            except Exception:
                pass

            # Phase 1: Create plan + validate scope
            self._log("info", "plan", "Creating plan and validating scope")
            plan = self._create_plan(request)
            record.plan = plan

            scope_override = (
                list(request.declared_scope) if request.declared_scope else None
            )
            scope_violations = plan.validate_scope(override=scope_override)
            boundary_violations = plan.validate_write_boundaries()
            all_violations = scope_violations + boundary_violations

            if all_violations:
                logger.warning("Plan violations: %s", all_violations)
                self._log("error", "plan", f"Plan rejected: {'; '.join(all_violations[:3])}")
                record.status = "failed"
                self._total_failures += 1
                return record

            # Phase 2: Get code context from codebase tool
            try:
                risk_str = f"{float(plan.estimated_risk):.2f}"
            except (TypeError, ValueError):
                risk_str = str(plan.estimated_risk)
            self._log("info", "plan", f"Plan accepted: {len(plan.files_to_modify)} file(s) to modify, risk={risk_str}")
            code_context = self._get_code_context(plan)
            conv.target_files = plan.files_to_modify + plan.files_to_create

            # Phase 2.5: Consult accumulated research knowledge
            research_context = self._gather_relevant_research(request)
            if research_context:
                logger.info(
                    "Injecting research context (%d chars) for: %s",
                    len(research_context), request.description[:40],
                )

            # Phase 3: Think-Code-Validate loop (up to MAX_ITERATIONS)
            patch = None
            report = None

            for iteration in range(MAX_ITERATIONS):
                record.iterations = iteration + 1
                record.status = "coding"
                conv.iteration = iteration
                from self_improve.system_upgrade_report import mint_attempt_id
                record.current_attempt_id = mint_attempt_id()
                self._log("thinking", "codegen", f"Iteration {iteration + 1}/{MAX_ITERATIONS}: generating patch", iteration=iteration + 1)

                if iteration == 0:
                    # First attempt: generate from plan + full source context
                    plan_text = self._format_plan_for_llm(
                        request, plan, research_context=research_context,
                    )
                    if code_context and code_context != "(codebase context unavailable)":
                        plan_text += f"\n\n## Current Source Code\n{code_context}"
                    conv.add_turn("think", plan_text)
                    messages = conv.get_messages_for_coder()

                    # Primary: local CoderServer (zero API cost)
                    patch = await self._provider.generate_with_coder(messages, plan.id)
                    # Fallback: main Ollama
                    if patch is None and ollama_client:
                        patch = await self._provider.generate_with_ollama(
                            ollama_client, messages, plan.id,
                        )
                    if patch is None:
                        patch = await self._provider.generate_patch_local(
                            conv, code_context, plan_text, plan,
                        )
                else:
                    # Retry: use feedback from previous failure
                    if not report:
                        break
                    diagnostics = report.get_first_diagnostics(limit=3)

                    # Primary: CoderServer retry
                    feedback_parts = ["The previous patch FAILED validation. Here are the errors:\n"]
                    for diag in diagnostics[:3]:
                        feedback_parts.append(
                            f"- [{diag.get('error_type', '?')}] {diag.get('file', '?')}"
                            f":{diag.get('line', '?')} -- {diag.get('message', '')}"
                        )
                    feedback_parts.append("\nFix these errors and regenerate the search-and-replace edits.")
                    feedback_parts.append('Return the same JSON format: {"files": [{"path": "...", "edits": [{"search": "...", "replace": "..."}]}], "description": "...", "confidence": ...}')
                    feedback = "\n".join(feedback_parts)
                    conv.add_turn("validate", feedback)
                    retry_messages = conv.get_messages_for_coder()

                    patch = await self._provider.generate_with_coder(retry_messages, plan.id)
                    if patch is None and ollama_client:
                        patch = await self._provider.retry_with_feedback(
                            ollama_client, conv, diagnostics, code_context, plan.id,
                        )

                if patch is None:
                    logger.warning("Failed to generate patch (iteration %d)", iteration)
                    self._log("warn", "codegen", f"Iteration {iteration + 1}: codegen returned no patch")
                    continue

                self._log("info", "codegen", f"Iteration {iteration + 1}: patch generated ({len(patch.files)} file(s), provider={patch.provider})")
                record.patch = patch

                # Populate original content for diffing
                patch.populate_original_content()

                # Validate: denied patterns, AST, syntax, diff budget
                content_violations = patch.validate()
                syntax_errors = patch.validate_syntax()
                budget_violations = patch.validate_diff_budget()
                escalations = patch.check_capability_escalation()

                all_pre_violations = content_violations + syntax_errors + budget_violations
                if all_pre_violations:
                    logger.info("Pre-sandbox violations (iteration %d): %s",
                                iteration, all_pre_violations[:3])
                    self._log("warn", "codegen", f"Iteration {iteration + 1}: pre-sandbox violations: {all_pre_violations[0][:80]}")
                    conv.add_turn("validate", "Pre-sandbox failures:\n" + "\n".join(all_pre_violations[:5]))
                    continue

                if escalations:
                    logger.info("Capability escalation detected: %s", escalations)
                    patch.requires_approval = True

                # Sandbox evaluation
                record.status = "testing"
                self._log("info", "sandbox", "Running sandbox evaluation (lint + tests + kernel sim)")
                report = await self._sandbox.evaluate(patch)
                record.report = report
                from self_improve.system_upgrade_report import (
                    append_attempt_to_report,
                    sandbox_summary_from_evaluation_report,
                )
                self._sync_upgrade_truth(record, dry_run=dry_run)
                append_attempt_to_report(
                    record.upgrade_id,
                    record.current_attempt_id,
                    record.iterations,
                    sandbox_summary_from_evaluation_report(report),
                )

                # Check for silent stubs
                if report.has_silent_stubs():
                    logger.error("Silent stubs detected in sandbox -- validation is fake")
                    self._log("error", "sandbox", "Silent stubs detected — sandbox results are fake")
                    record.status = "failed"
                    self._total_failures += 1
                    self._sync_upgrade_truth(record, dry_run=dry_run)
                    self._persist_proposal(record)
                    self._emit_event(IMPROVEMENT_SANDBOX_FAILED, {
                        "upgrade_id": record.upgrade_id,
                        "id": request.id,
                        "reason": "silent_stubs",
                    })
                    self._record_memory(record)
                    return record

                if report.overall_passed:
                    self._log("success", "sandbox", f"Sandbox PASSED (lint={report.lint_passed}, tests={report.all_tests_passed}, sim={report.sim_passed})")
                    break

                self._log("warn", "sandbox", f"Sandbox failed (iteration {iteration + 1}): lint={report.lint_passed}, tests={report.all_tests_passed}, sim={report.sim_passed}")
                logger.info("Sandbox failed (iteration %d), %d diagnostics",
                            iteration, len(report.diagnostics))

            if not patch or not report or not report.overall_passed:
                self._log("error", "pipeline", f"Pipeline failed after {record.iterations} iteration(s): no passing patch")
                record.status = "failed"
                self._total_failures += 1
                self._sync_upgrade_truth(record, dry_run=dry_run)
                self._persist_proposal(record)
                self._emit_event(IMPROVEMENT_SANDBOX_FAILED, {
                    "upgrade_id": record.upgrade_id,
                    "id": request.id,
                    "reason": "sandbox_or_generation_failed",
                })
                self._emit_event(IMPROVEMENT_VALIDATED, {
                    "passed": False, "id": request.id, "upgrade_id": record.upgrade_id,
                })
                _save_conversation(conv)
                self._persist_history()
                self._record_memory(record)
                return record

            self._emit_event(IMPROVEMENT_SANDBOX_PASSED, {
                "upgrade_id": record.upgrade_id,
                "id": request.id,
            })
            self._emit_event(IMPROVEMENT_VALIDATED, {
                "passed": True, "id": request.id, "upgrade_id": record.upgrade_id,
            })

            # Phase 4: Review (optional Claude — skipped when Claude generated the patch)
            record.status = "reviewing"
            generator = patch.provider if patch else ""
            review: dict[str, Any] = {}
            if self._provider._claude_available and generator != "claude":
                plan_text = self._format_plan_for_llm(request, plan)
                review = await self._provider.review_with_claude(patch, plan_text, code_context)
                record.review_result = review
                if not review.get("approved", True):
                    logger.info("Claude review rejected: %s", review.get("reasoning"))
                    self._win_rate.total += 1
                    self._win_rate.review_rejected += 1
                    record.status = "failed"
                    self._total_failures += 1
                    _save_conversation(conv)
                    self._record_memory(record)
                    return record
                self._win_rate.review_approved += 1
            else:
                record.review_result = {"approved": True, "reasoning": "skipped (same provider)"}

            # Phase 5: Approval gate
            # At Stage 2, ALL auto-triggered patches require human approval.
            # Manual operator-triggered runs bypass (operator already chose to run it).
            # The human gate sits at apply/promote, not at observation/proposal generation.
            stage2_auto = (self._stage == STAGE_HUMAN_APPROVAL and not request.manual)
            needs_approval = patch.requires_approval or request.requires_approval or stage2_auto
            if needs_approval:
                reasons = []
                if stage2_auto: reasons.append("stage2")
                if patch.requires_approval: reasons.append("patch_flag")
                if request.requires_approval: reasons.append("request_flag")
                self._log("info", "approval", f"Patch queued for human approval ({', '.join(reasons)})")
                record.status = "awaiting_approval"
                self._pending_approvals.append(record)
                self._save_pending_approvals()
                self._emit_event(IMPROVEMENT_NEEDS_APPROVAL, {
                    "id": request.id, "upgrade_id": record.upgrade_id,
                })
                _save_conversation(conv)
                self._persist_history()
                self._record_memory(record)
                return record

            # Dry-run checkpoint: full pipeline completed, stop before apply
            if dry_run:
                self._log("success", "pipeline", f"Dry-run complete: {len(patch.files)} file(s), sandbox={'PASS' if report.overall_passed else 'FAIL'}")
                record.status = "dry_run"
                record.completed_at = time.time()
                self._win_rate.total += 1
                if report.overall_passed:
                    self._win_rate.sandbox_passed += 1
                else:
                    self._win_rate.sandbox_failed += 1
                self._sync_upgrade_truth(record, dry_run=True)
                self._emit_event(IMPROVEMENT_DRY_RUN, {
                    "id": request.id,
                    "upgrade_id": record.upgrade_id,
                    "description": request.description[:100],
                    "iterations": record.iterations,
                    "sandbox_passed": report.overall_passed,
                    "recommendation": report.recommendation,
                    "files": [fd.path for fd in patch.files],
                })
                self._persist_proposal(record)
                _save_conversation(conv)
                self._persist_history()
                logger.info(
                    "DRY RUN complete: %s (%d file(s), %d iteration(s), sandbox=%s)",
                    request.description[:60],
                    len(patch.files),
                    record.iterations,
                    report.recommendation,
                )
                self._record_memory(record)
                return record

            # Phase 6: Snapshot + pre-apply verification marker
            # STRICT ORDER: snapshot → baselines → write pending → apply
            self._log("info", "apply", "Creating snapshot and applying patch to disk")
            snapshot_path = self._create_snapshot(patch)
            record.snapshot_path = snapshot_path

            restart_verify = False
            if self._engine and self._restart_callback:
                restart_verify = self._prepare_restart_verify(
                    patch, snapshot_path, record, conv,
                )

            apply_ok = self._atomic_apply(patch)
            if not apply_ok:
                record.status = "failed"
                self._total_failures += 1
                self._restore_snapshot(snapshot_path)
                if restart_verify:
                    from self_improve.verification import clear_pending
                    clear_pending()
                self._sync_upgrade_truth(record, dry_run=dry_run)
                _save_conversation(conv)
                self._record_memory(record)
                return record

            # Phase 7: In-memory health gate (fast first check — no restart needed)
            self._log("info", "health", "Running post-apply health check")
            health_ok = await self._check_post_apply_health(snapshot_path)
            if not health_ok:
                logger.warning("Health regression detected after apply -- rolling back")
                self._log("error", "rollback", "Health regression detected — rolling back patch")
                self._restore_snapshot(snapshot_path)
                if restart_verify:
                    from self_improve.verification import clear_pending
                    clear_pending()
                record.status = "rolled_back"
                self._total_rollbacks += 1
                self._sync_upgrade_truth(record, dry_run=dry_run)
                self._emit_event(IMPROVEMENT_ROLLED_BACK, {
                    "id": request.id,
                    "upgrade_id": record.upgrade_id,
                    "reason": "health_regression",
                })
                try:
                    from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                    attribution_ledger.record(
                        subsystem="self_improve",
                        event_type="improvement_rolled_back",
                        data={"reason": "health_regression"},
                        evidence_refs=[{"kind": "improvement", "id": request.id}],
                        parent_entry_id=_started_eid,
                    )
                    if _started_eid:
                        attribution_ledger.record_outcome(_started_eid, "regressed", build_outcome_data(
                            confidence=0.8,
                            latency_s=round(time.time() - _started_ts, 2),
                            source="health_monitor",
                            tier="medium",
                            scope="general",
                            blame_target="general",
                            reason="health_regression",
                        ))
                except Exception:
                    pass
                _save_conversation(conv)
                self._persist_history()
                self._record_memory(record)
                return record

            # Phase 8: Restart-verify (pending marker already on disk)
            if restart_verify:
                record.status = "verifying"
                self._sync_upgrade_truth(record, dry_run=dry_run)
                self._record_memory(record)
                _save_conversation(conv)
                self._persist_history()
                logger.info(
                    "Pending verification written for %s — triggering restart",
                    record.request.id,
                )
                if asyncio.iscoroutinefunction(self._restart_callback):
                    await self._restart_callback()
                else:
                    self._restart_callback()
                return record

            # Fallback: promote immediately if restart-verify not available
            self._log("success", "pipeline", f"Patch promoted: {request.description[:60]} ({record.iterations} iter)")
            record.status = "promoted"
            record.completed_at = time.time()
            self._total_improvements += 1
            self._sync_upgrade_truth(record, dry_run=dry_run)
            self._emit_event(IMPROVEMENT_PROMOTED, {
                "id": request.id,
                "upgrade_id": record.upgrade_id,
                "description": request.description[:100],
                "iterations": record.iterations,
            })
            try:
                from consciousness.attribution_ledger import attribution_ledger, build_outcome_data, outcome_scheduler
                attribution_ledger.record(
                    subsystem="self_improve",
                    event_type="improvement_promoted",
                    data={"description": request.description[:200], "iterations": record.iterations},
                    evidence_refs=[{"kind": "improvement", "id": request.id}],
                    parent_entry_id=_started_eid,
                )
                if _started_eid:
                    attribution_ledger.record_outcome(_started_eid, "success", build_outcome_data(
                        confidence=0.75,
                        latency_s=round(time.time() - _started_ts, 2),
                        source="verification_test",
                        tier="medium",
                        scope="general",
                        blame_target="general",
                        description=request.description[:100],
                    ))
                    _health_before = self._capture_health_snapshot()
                    _desc = request.description[:60]
                    _eid = _started_eid

                    def _check_long_term_health() -> tuple[str, dict[str, Any]] | None:
                        _now = self._capture_health_snapshot()
                        if _now is None or _health_before is None:
                            return None
                        tick_p95_before = _health_before.get("tick_p95_ms", 0)
                        tick_p95_after = _now.get("tick_p95_ms", 0)
                        error_before = _health_before.get("error_count", 0)
                        error_after = _now.get("error_count", 0)
                        degraded = (tick_p95_after > tick_p95_before * 1.2) or (error_after > error_before + 5)
                        return (
                            "regressed" if degraded else "stable",
                            build_outcome_data(
                                confidence=0.7,
                                latency_s=1800.0,
                                source="health_monitor",
                                tier="delayed",
                                scope="general",
                                blame_target="general",
                                health_before=_health_before,
                                health_after=_now,
                            ),
                        )
                    outcome_scheduler.schedule(
                        entry_id=_eid,
                        delay_s=1800,
                        check_fn=_check_long_term_health,
                        subsystem="self_improve",
                        description=f"30min health: {_desc}",
                    )
            except Exception:
                pass
            self._reindex_codebase(patch)

            conv.status = "completed"
            conv.completed_at = time.time()
            _save_conversation(conv)
            self._persist_history()

            logger.info("Improvement promoted (no restart-verify): %s (after %d iteration(s))",
                        request.description[:60], record.iterations)
            from self_improve.system_upgrade_report import load_report, maybe_append_training_sample
            _urep = load_report(record.upgrade_id)
            if _urep:
                maybe_append_training_sample(_urep)
            self._record_memory(record)
            return record

        except Exception:
            logger.exception("Improvement pipeline failed")
            record.status = "failed"
            self._total_failures += 1
            _save_conversation(conv)
            if getattr(record, "upgrade_id", ""):
                self._sync_upgrade_truth(record, dry_run=dry_run)
            self._record_memory(record)
            return record

    # ------------------------------------------------------------------
    # Atomic file application
    # ------------------------------------------------------------------

    def _create_snapshot(self, patch: CodePatch) -> str:
        """Backup original files before applying the patch."""
        ts = int(time.time())
        snap_dir = SNAPSHOT_DIR / f"{ts}_{patch.id}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        project_root = Path(self._sandbox._project_root)
        for fd in patch.files:
            rel = fd.path.replace("brain/", "")
            src = project_root / rel
            if src.exists():
                dst = snap_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        return str(snap_dir)

    def _atomic_apply(self, patch: CodePatch) -> bool:
        """Apply patch atomically: write to temp, then rename into place."""
        project_root = Path(self._sandbox._project_root)
        temp_files: list[tuple[Path, Path]] = []

        try:
            for fd in patch.files:
                if not fd.new_content:
                    continue
                rel = fd.path.replace("brain/", "")
                target = project_root / rel
                try:
                    target.resolve().relative_to(project_root.resolve())
                except ValueError:
                    logger.error("Path traversal blocked: %s resolves outside project root", fd.path)
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)

                temp = target.with_suffix(target.suffix + ".tmp")
                temp.write_text(fd.new_content, encoding="utf-8")
                temp_files.append((temp, target))

            # All temp files written successfully -- now swap
            for temp, target in temp_files:
                os.replace(str(temp), str(target))

            return True

        except Exception:
            logger.exception("Atomic apply failed, cleaning up temp files")
            for temp, _ in temp_files:
                try:
                    temp.unlink(missing_ok=True)
                except Exception:
                    pass
            return False

    def _restore_snapshot(self, snapshot_path: str) -> bool:
        """Restore original files from a snapshot directory.

        Note: callers manage _total_rollbacks to avoid double-counting.
        """
        snap = Path(snapshot_path)
        if not snap.exists():
            return False

        project_root = Path(self._sandbox._project_root)
        try:
            for fpath in snap.rglob("*.py"):
                rel = fpath.relative_to(snap)
                target = project_root / rel
                shutil.copy2(str(fpath), str(target))
            logger.info("Restored snapshot from %s", snapshot_path)
            return True
        except Exception:
            logger.exception("Snapshot restore failed")
            return False

    async def _check_post_apply_health(self, snapshot_path: str) -> bool:
        """Monitor kernel health for HEALTH_MONITOR_TICKS after applying a patch.

        Checks p95 tick time -- if it regresses beyond P95_REGRESSION_THRESHOLD
        compared to the pre-apply baseline, returns False (rollback needed).
        """
        try:
            pre_p95 = 0.0
            tick_times: list[float] = []

            try:
                from consciousness.kernel import kernel_loop
                perf = kernel_loop.get_performance()
                pre_p95 = perf.get("p95_tick_ms", 0.0)
            except Exception:
                pass

            if pre_p95 <= 0:
                return True

            for _ in range(HEALTH_MONITOR_TICKS):
                await asyncio.sleep(0.15)
                try:
                    perf = kernel_loop.get_performance()
                    tick_ms = perf.get("last_tick_ms", 0.0)
                    if tick_ms > 0:
                        tick_times.append(tick_ms)
                except Exception:
                    pass

            if len(tick_times) < 10:
                return True

            tick_times.sort()
            idx = int(len(tick_times) * 0.95)
            post_p95 = tick_times[min(idx, len(tick_times) - 1)]

            if post_p95 > pre_p95 * P95_REGRESSION_THRESHOLD:
                logger.warning(
                    "p95 regression: %.1fms -> %.1fms (threshold %.1fx)",
                    pre_p95, post_p95, P95_REGRESSION_THRESHOLD,
                )
                return False

            logger.info("Health check passed: p95 %.1fms -> %.1fms", pre_p95, post_p95)
            return True

        except Exception:
            logger.exception("Health check failed, assuming OK")
            return True

    def _prepare_restart_verify(
        self,
        patch: CodePatch,
        snapshot_path: str,
        record: ImprovementRecord,
        conv: ImprovementConversation,
    ) -> bool:
        """Capture baselines and write the pending marker BEFORE apply.

        Returns True if the pending marker is on disk and restart-verify
        should proceed after apply.  Returns False if baselines are
        unavailable (caller should promote without restart).
        """
        from self_improve.verification import capture_baselines, write_pending

        try:
            MIN_TICKS_FOR_BASELINE = 100
            if self._engine._kernel:
                perf = self._engine._kernel.get_performance()
                if perf.tick_count < MIN_TICKS_FOR_BASELINE:
                    logger.info(
                        "Only %d ticks elapsed (need %d) — will promote without restart-verify",
                        perf.tick_count, MIN_TICKS_FOR_BASELINE,
                    )
                    return False

            baselines = capture_baselines(self._engine)
            if not baselines:
                logger.warning("Could not capture baselines — promoting without restart-verify")
                return False

            files_changed = [fd.path for fd in patch.files if fd.new_content]

            write_pending(
                patch_id=record.request.id,
                description=record.request.description[:200],
                files_changed=files_changed,
                snapshot_path=snapshot_path,
                conversation_id=conv.id,
                baselines=baselines,
                target_metrics=[],
                upgrade_id=record.upgrade_id,
            )
            return True

        except Exception:
            logger.exception("Failed to prepare restart-verify — will promote without restart")
            from self_improve.verification import clear_pending
            clear_pending()
            return False

    def rollback(self, request_id: str) -> bool:
        """Rollback a promoted improvement by restoring its snapshot."""
        for record in self._history:
            if record.request.id == request_id and record.snapshot_path:
                ok = self._restore_snapshot(record.snapshot_path)
                if ok:
                    record.status = "rolled_back"
                    self._total_rollbacks += 1
                    self._sync_upgrade_truth(record, dry_run=self._dry_run_mode)
                    self._emit_event(IMPROVEMENT_ROLLED_BACK, {
                        "id": request_id,
                        "upgrade_id": record.upgrade_id,
                    })
                    self._reindex_codebase(record.patch)
                return ok
        return False

    @staticmethod
    def restore_snapshot_static(snapshot_path: str) -> bool:
        """Restore files from a snapshot without a full orchestrator instance.

        Used by main.py for crash-loop rollback before subsystems are initialized.
        """
        snap = Path(snapshot_path)
        if not snap.exists():
            logger.warning("Snapshot path does not exist: %s", snapshot_path)
            return False
        project_root = Path(__file__).resolve().parent.parent
        try:
            for fpath in snap.rglob("*.py"):
                rel = fpath.relative_to(snap)
                target = project_root / rel
                shutil.copy2(str(fpath), str(target))
            logger.info("Static snapshot restore from %s", snapshot_path)
            return True
        except Exception:
            logger.exception("Static snapshot restore failed")
            return False

    # ------------------------------------------------------------------
    # Approval system
    # ------------------------------------------------------------------

    async def approve(self, request_id: str) -> dict[str, Any]:
        """Approve a pending patch: apply with the same safety as the normal pipeline.

        Returns structured result instead of bare bool so the dashboard can
        show meaningful feedback after approval.
        """
        result: dict[str, Any] = {
            "applied": False,
            "health_check_passed": False,
            "rolled_back": False,
            "restart_verify_scheduled": False,
            "reason": "patch_not_found",
        }
        for record in self._pending_approvals:
            if record.request.id == request_id and record.patch:
                self._log("info", "approval", f"Operator approved patch: {record.request.description[:60]}")
                if not record.upgrade_id:
                    from self_improve.system_upgrade_report import mint_upgrade_id
                    record.upgrade_id = mint_upgrade_id()

                snapshot_path = self._create_snapshot(record.patch)
                record.snapshot_path = snapshot_path

                restart_verify = False
                if self._engine and self._restart_callback:
                    restart_verify = self._prepare_restart_verify(
                        record.patch, snapshot_path, record, None,
                    )

                if not self._atomic_apply(record.patch):
                    self._restore_snapshot(snapshot_path)
                    if restart_verify:
                        try:
                            from self_improve.verification import clear_pending
                            clear_pending()
                        except Exception:
                            pass
                    result["reason"] = "atomic_apply_failed"
                    return result

                result["applied"] = True

                health_ok = await self._check_post_apply_health(snapshot_path)
                result["health_check_passed"] = health_ok

                if not health_ok:
                    logger.warning("Health regression after approved patch %s -- rolling back", request_id)
                    self._restore_snapshot(snapshot_path)
                    if restart_verify:
                        try:
                            from self_improve.verification import clear_pending
                            clear_pending()
                        except Exception:
                            pass
                    record.status = "rolled_back"
                    record.completed_at = time.time()
                    self._pending_approvals.remove(record)
                    self._save_pending_approvals()
                    self._total_rollbacks += 1
                    self._sync_upgrade_truth(record, dry_run=self._dry_run_mode)
                    self._emit_event(IMPROVEMENT_ROLLED_BACK, {
                        "id": request_id, "upgrade_id": record.upgrade_id,
                        "reason": "health_regression_after_approval",
                    })
                    self._persist_history()
                    result["rolled_back"] = True
                    result["reason"] = "health_regression_after_approval"
                    return result

                record.status = "promoted"
                record.completed_at = time.time()
                self._pending_approvals.remove(record)
                self._save_pending_approvals()
                self._total_improvements += 1
                self._reindex_codebase(record.patch)
                self._sync_upgrade_truth(record, dry_run=self._dry_run_mode)
                self._emit_event(IMPROVEMENT_PROMOTED, {
                    "id": request_id,
                    "upgrade_id": record.upgrade_id,
                })
                self._persist_history()
                try:
                    from self_improve.system_upgrade_report import load_report, maybe_append_training_sample
                    _ur = load_report(record.upgrade_id)
                    if _ur:
                        maybe_append_training_sample(_ur)
                except Exception:
                    pass

                result["restart_verify_scheduled"] = restart_verify
                result["reason"] = "promoted"
                return result

        return result

    def reject(self, request_id: str) -> dict[str, Any]:
        """Reject a pending patch. Returns structured result."""
        for record in self._pending_approvals:
            if record.request.id == request_id:
                record.status = "failed"
                record.completed_at = time.time()
                self._pending_approvals.remove(record)
                self._save_pending_approvals()
                self._sync_upgrade_truth(record, dry_run=self._dry_run_mode)
                self._persist_history()
                return {"rejected": True, "reason": "operator_rejected"}
        return {"rejected": False, "reason": "patch_not_found"}

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        results = []
        for r in self._pending_approvals:
            escalations = r.patch.check_capability_escalation() if r.patch else []

            reasons: list[str] = []
            if self._stage == STAGE_HUMAN_APPROVAL and not r.request.manual:
                reasons.append("stage2_auto_triggered")
            if r.patch and r.patch.requires_approval:
                if escalations:
                    reasons.append("capability_escalation")
                if r.plan and r.plan.check_dangerous():
                    reasons.append("dangerous_file")
            if r.request.requires_approval:
                reasons.append("high_priority")
            if r.report and r.report.recommendation == "manual_review":
                reasons.append("sandbox_manual_review")
            if not reasons:
                reasons.append("unknown")

            sandbox_summary = None
            if r.report:
                sandbox_summary = {
                    "overall_passed": r.report.overall_passed,
                    "lint_passed": r.report.lint_passed,
                    "all_tests_passed": r.report.all_tests_passed,
                    "sim_passed": r.report.sim_passed,
                    "recommendation": r.report.recommendation,
                    "diagnostics": [d.to_dict() if hasattr(d, "to_dict") else str(d) for d in (r.report.diagnostics or [])][:5],
                }

            diffs = []
            if r.patch:
                for fd in r.patch.files:
                    diff_text = fd.diff or ""
                    if not diff_text and fd.original_content and fd.new_content:
                        import difflib
                        diff_text = "".join(difflib.unified_diff(
                            fd.original_content.splitlines(keepends=True),
                            fd.new_content.splitlines(keepends=True),
                            fromfile=f"a/{fd.path}",
                            tofile=f"b/{fd.path}",
                            n=3,
                        ))
                    diffs.append({"path": fd.path, "diff": diff_text[:4000]})

            results.append({
                "request_id": r.request.id,
                "upgrade_id": r.upgrade_id,
                "description": r.request.description,
                "type": r.request.type,
                "target": r.request.target_module,
                "priority": r.request.priority,
                "evidence": r.request.evidence[:10],
                "escalations": escalations,
                "why_requires_approval": reasons,
                "diffs": diffs,
                "sandbox_summary": sandbox_summary,
                "created_at": r.created_at,
                "provider": r.patch.provider if r.patch else "unknown",
                "iterations": r.iterations,
            })
        return results

    # ------------------------------------------------------------------
    # Code context
    # ------------------------------------------------------------------

    def _gather_relevant_research(self, request: ImprovementRequest) -> str:
        """Query memory for research findings relevant to the improvement target.

        Returns formatted research context for prompt injection, and queues
        a targeted research intent via the autonomy system when knowledge
        is insufficient (non-blocking).
        """
        findings: list[str] = []
        try:
            from memory.search import semantic_search, search_by_tag

            candidates = semantic_search(request.description, top_k=15)

            research_mems = [
                m for m in candidates
                if "autonomous_research" in (m.tags or [])
            ]

            if not research_mems:
                all_research = search_by_tag("autonomous_research")
                keywords = set(request.description.lower().split())
                for m in all_research:
                    payload_lower = m.payload.lower()
                    overlap = sum(1 for kw in keywords if kw in payload_lower)
                    if overlap >= 2:
                        research_mems.append(m)
                research_mems = research_mems[:10]

            peer_reviewed = [
                m for m in research_mems
                if "evidence:peer_reviewed" in (m.tags or [])
            ]
            codebase_sourced = [
                m for m in research_mems
                if "evidence:codebase" in (m.tags or []) or "code_sourced" in (m.tags or [])
            ]
            other = [
                m for m in research_mems
                if m not in peer_reviewed and m not in codebase_sourced
            ]

            ranked = peer_reviewed + codebase_sourced + other

            for m in ranked[:8]:
                provenance = "peer-reviewed" if "evidence:peer_reviewed" in (m.tags or []) else (
                    "codebase" if "code_sourced" in (m.tags or []) else "research"
                )
                findings.append(f"[{provenance}] {m.payload[:300]}")

            if len(ranked) < 2:
                self._request_targeted_research(request)

        except Exception:
            logger.debug("Research gathering failed", exc_info=True)

        if not findings:
            return ""

        header = (
            f"## Relevant Research ({len(findings)} finding(s))\n"
            "Base your implementation on these findings where applicable.\n\n"
        )
        return header + "\n".join(f"- {f}" for f in findings) + "\n"

    def _request_targeted_research(self, request: ImprovementRequest) -> None:
        """Queue a focused research intent for future knowledge building."""
        try:
            from autonomy.research_intent import ResearchIntent

            target = request.target_module or "system"
            question = (
                f"Best practices and techniques for {request.description[:80]} "
                f"in Python {target} systems"
            )

            intent = ResearchIntent(
                question=question,
                source_event="improvement:knowledge_gap",
                source_hint="academic",
                priority=0.6,
                scope="external_ok",
                max_results=5,
                tag_cluster=(
                    "self_improvement",
                    f"target:{target}",
                    request.type or "optimization",
                ),
            )

            autonomy_orch = getattr(self._engine, "_autonomy_orchestrator", None)
            if autonomy_orch and hasattr(autonomy_orch, "enqueue"):
                autonomy_orch.enqueue(intent)
                logger.info(
                    "Queued research for knowledge gap: %s", question[:60],
                )
        except Exception:
            logger.debug("Failed to queue research intent", exc_info=True)

    _CODE_CONTEXT_MAX_CHARS = 40000  # ~10K tokens budget for source context

    def _get_code_context(self, plan: PatchPlan) -> str:
        """Get directory inventory + targeted source + structural summary.

        When the plan target is a directory, resolves to actual .py files,
        picks the most relevant file(s) by keyword matching against the
        request description, and includes full source only for those.
        The rest get signatures from the codebase index.
        """
        parts: list[str] = []
        brain_dir = Path(__file__).resolve().parent.parent
        resolved_files: list[str] = []
        description_lower = (plan.request_id or "").lower()
        if hasattr(plan, "_request_description"):
            description_lower = plan._request_description.lower()

        for fpath in plan.files_to_modify:
            rel = fpath.replace("brain/", "") if fpath.startswith("brain/") else fpath
            full = brain_dir / rel

            if full.is_dir():
                py_files = sorted(full.glob("*.py"))
                sibling_names = [f.name for f in py_files]
                sibling_rels = [f"{rel}{f.name}" for f in py_files]
                resolved_files.extend(sibling_rels)

                parts.append(
                    f"### Directory inventory: {rel}\n"
                    f"Files ({len(sibling_names)}): {', '.join(sibling_names)}\n"
                    f"IMPORTANT: Only these filenames exist. Do NOT invent filenames.\n"
                )
            elif full.exists() and full.is_file():
                resolved_files.append(rel)

        plan._resolved_target_files = [
            f if f.startswith("brain/") else f"brain/{f}" for f in resolved_files
        ]

        ranked = self._rank_files_by_relevance(resolved_files, description_lower, brain_dir)
        used_chars = sum(len(p) for p in parts)

        full_source_files: list[str] = []
        for rel, _score in ranked:
            full = brain_dir / rel
            if not full.exists():
                continue
            try:
                src = full.read_text(encoding="utf-8")
            except Exception:
                continue
            block = f"### {rel} (FULL SOURCE — preserve all public names)\n```python\n{src}\n```\n"
            if used_chars + len(block) > self._CODE_CONTEXT_MAX_CHARS:
                break
            parts.append(block)
            full_source_files.append(rel)
            used_chars += len(block)

        try:
            from tools.codebase_tool import codebase_index
            if not codebase_index._modules:
                codebase_index.build()

            remaining_budget = max(1000, self._CODE_CONTEXT_MAX_CHARS - used_chars)
            remaining_tokens = remaining_budget // 4

            sig_files = [f for f in resolved_files if f not in full_source_files]
            ctx_files = sig_files + [
                f.replace("brain/", "") for f in plan.files_to_create
            ]
            seen: set[str] = set()
            deduped = []
            for f in ctx_files:
                if f not in seen:
                    seen.add(f)
                    deduped.append(f)
            if deduped:
                summary = codebase_index.get_budgeted_context(deduped, max_tokens=remaining_tokens)
                parts.append(f"### Structural summary (signatures of other files in module)\n{summary}\n")
        except Exception:
            logger.debug("Codebase summary unavailable", exc_info=True)

        return "\n".join(parts) if parts else "(codebase context unavailable)"

    @staticmethod
    def _rank_files_by_relevance(
        files: list[str], description: str, brain_dir: Path,
    ) -> list[tuple[str, float]]:
        """Rank files by keyword overlap with the improvement description.

        Returns (rel_path, score) sorted descending by relevance.
        """
        desc_words = set(description.replace("_", " ").replace("/", " ").split())
        scored: list[tuple[str, float]] = []
        for rel in files:
            name = Path(rel).stem
            name_words = set(name.replace("_", " ").split())
            overlap = len(desc_words & name_words)
            size_penalty = 0.0
            full = brain_dir / rel
            if full.exists():
                try:
                    size = full.stat().st_size
                    if size > 30000:
                        size_penalty = 0.5
                except OSError:
                    pass
            score = overlap - size_penalty
            if name == "__init__":
                score -= 2.0
            scored.append((rel, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    @staticmethod
    def _summarize_test_file(path: Path, max_items: int = 8) -> str:
        """Summarize pytest contracts from a test file without inlining full source."""
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
        except Exception:
            return ""

        items: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                items.append(node.name)
                continue
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                methods = [
                    f"{node.name}.{child.name}"
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name.startswith("test_")
                ]
                items.extend(methods or [node.name])

        if not items:
            return "pytest behavioral contract"

        shown = items[:max_items]
        summary = ", ".join(shown)
        if len(items) > max_items:
            summary += f", ... (+{len(items) - max_items} more)"
        return summary

    def _get_targeted_test_context(self, plan: PatchPlan) -> str:
        """Describe the pytest files the sandbox will use as behavioral contracts."""
        test_dir = Path(__file__).resolve().parent.parent / "tests"
        if not test_dir.exists():
            return ""

        patched_files = plan.files_to_modify + plan.files_to_create
        targets = Sandbox._select_test_targets(patched_files or None, test_dir)
        if not targets:
            return ""

        lines = [
            "## Targeted Test Contracts",
            "Use these existing pytest files as behavioral contracts and reproduction anchors.",
            "Jarvis should inspect them before guessing about behavior or traces.",
            "Do not change tests unless the plan explicitly requires and justifies it.",
        ]
        for raw_target in targets[:6]:
            target = Path(raw_target)
            rel = (
                f"tests/{target.name}"
                if target.parent.name == "tests"
                else str(target)
            )
            summary = self._summarize_test_file(target)
            if summary:
                lines.append(f"- `{rel}`: {summary}")
            else:
                lines.append(f"- `{rel}`")
        if len(targets) > 6:
            lines.append(f"- ... and {len(targets) - 6} more targeted test file(s)")
        return "\n".join(lines)

    def _reindex_codebase(self, patch: CodePatch | None) -> None:
        """Re-index modified files after patch application."""
        if not patch:
            return
        try:
            from tools.codebase_tool import codebase_index
            for fd in patch.files:
                rel = fd.path.replace("brain/", "")
                codebase_index.rebuild_file(rel)
        except Exception:
            logger.debug("Codebase reindex failed", exc_info=True)

    def _capture_health_snapshot(self) -> dict[str, Any] | None:
        """Lightweight health snapshot for delayed outcome comparison."""
        try:
            from consciousness.kernel import kernel_loop
            perf = kernel_loop.get_performance()
            return {
                "tick_p95_ms": perf.get("p95_tick_ms", 0.0),
                "error_count": perf.get("error_count", 0),
                "tick_count": perf.get("tick_count", 0),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Safety gates
    # ------------------------------------------------------------------

    def _check_safety_gates(self) -> str:
        """Pre-flight safety checks. Returns empty string if OK, else reason."""
        # Quarantine pressure gate
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            qp = get_quarantine_pressure()
            if qp.current.high:
                return "quarantine_pressure_high"
        except Exception:
            pass

        # Soul integrity gate
        try:
            from epistemic.soul_integrity.index import SoulIntegrityIndex
            si = SoulIntegrityIndex.get_instance()
            idx = si.get_current_index()
            if idx is not None and idx < SOUL_INTEGRITY_GATE_THRESHOLD:
                return f"soul_integrity_low ({idx:.2f})"
        except Exception:
            pass

        return ""

    def get_safety_gate_status(self) -> dict[str, Any]:
        """Current state of all safety gates for dashboard display."""
        result: dict[str, Any] = {
            "quarantine_ok": True,
            "quarantine_composite": 0.0,
            "soul_integrity_ok": True,
            "soul_integrity_index": None,
        }
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            qp = get_quarantine_pressure()
            state = qp.current
            result["quarantine_composite"] = round(state.composite, 3)
            result["quarantine_ok"] = not state.high
        except Exception:
            pass
        try:
            from epistemic.soul_integrity.index import SoulIntegrityIndex
            si = SoulIntegrityIndex.get_instance()
            idx = si.get_current_index()
            result["soul_integrity_index"] = round(idx, 3) if idx is not None else None
            result["soul_integrity_ok"] = idx is None or idx >= SOUL_INTEGRITY_GATE_THRESHOLD
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Proposal persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_unified_diff(fd: FileDiff) -> str:
        """Generate a unified diff string for a FileDiff."""
        if not fd.original_content and not fd.new_content:
            return ""
        old_lines = (fd.original_content or "").splitlines(keepends=True)
        new_lines = (fd.new_content or "").splitlines(keepends=True)
        diff_lines = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{fd.path}", tofile=f"b/{fd.path}",
        )
        return "".join(diff_lines)

    def _persist_proposal(self, record: ImprovementRecord) -> None:
        """Append a rich proposal record to improvement_proposals.jsonl."""
        try:
            PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)

            diffs: list[dict[str, str]] = []
            if record.patch:
                for fd in record.patch.files:
                    unified = self._compute_unified_diff(fd)
                    diffs.append({
                        "path": fd.path,
                        "diff": unified[:10000],
                    })

            proposal = {
                "id": record.request.id,
                "upgrade_id": record.upgrade_id,
                "timestamp": record.created_at,
                "completed_at": record.completed_at,
                "status": record.status,
                "fingerprint": record.request.fingerprint,
                "what": {
                    "type": record.request.type,
                    "description": record.request.description,
                },
                "why": {
                    "evidence": record.request.evidence[:10],
                    "priority": record.request.priority,
                    "evidence_detail": record.request.evidence_detail,
                },
                "where": {
                    "target_module": record.request.target_module,
                    "files_modified": [fd.path for fd in record.patch.files] if record.patch else [],
                    "files_planned": record.plan.files_to_modify if record.plan else [],
                },
                "who": {
                    "provider": record.patch.provider if record.patch else "none",
                    "trigger": "auto" if not record.request.requires_approval else "manual",
                },
                "iterations": record.iterations,
                "conversation_id": record.conversation_id,
                "diffs": diffs,
                "sandbox": record.report.to_dict() if record.report else None,
                "review": record.review_result,
                "stage": self._stage,
            }

            with open(PROPOSALS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(proposal, default=str) + "\n")

            logger.info("Persisted proposal %s to %s", record.request.id, PROPOSALS_FILE)

            # Record CODE_QUALITY hemisphere feature vector
            try:
                from hemisphere.code_quality_encoder import CodeQualityEncoder
                from hemisphere.distillation import DistillationCollector
                mod_history = self.get_module_patch_history(
                    record.request.target_module,
                    fingerprint=record.request.fingerprint or "",
                )
                vec = CodeQualityEncoder.encode(record, module_history=mod_history)
                collector = DistillationCollector.instance()
                collector.record(
                    teacher="code_quality_features",
                    signal_type="improvement_features",
                    data=vec,
                    metadata={"upgrade_id": record.upgrade_id},
                    origin="self_improve",
                    fidelity=1.0,
                )
            except Exception:
                logger.debug("Failed to record code quality feature vector", exc_info=True)
        except Exception:
            logger.warning("Failed to persist proposal", exc_info=True)

    @staticmethod
    def get_module_patch_history(
        target_module: str,
        fingerprint: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Scan improvement_proposals.jsonl for prior patches to the same module.

        Reads newest-first, stops after ``limit`` matches for target_module.
        Total lines scanned is capped at 500 to bound cost.
        """
        result: dict[str, Any] = {
            "total_patches": 0,
            "verdict_counts": {"improved": 0, "stable": 0, "regressed": 0, "rolled_back": 0},
            "last_patch_age_s": -1.0,
            "avg_iterations": 0.0,
            "recidivism": False,
            "has_history": False,
        }
        if not target_module or not PROPOSALS_FILE.exists():
            return result

        now = time.time()
        matches: list[dict[str, Any]] = []
        lines_scanned = 0
        max_scan = 500

        try:
            all_lines: list[str] = []
            with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        all_lines.append(stripped)

            for raw in reversed(all_lines):
                if lines_scanned >= max_scan:
                    break
                lines_scanned += 1
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                where = rec.get("where", {})
                rec_module = where.get("target_module", "")
                if rec_module != target_module:
                    continue
                matches.append(rec)
                if len(matches) >= limit:
                    break
        except Exception:
            logger.debug("get_module_patch_history scan failed", exc_info=True)
            return result

        if not matches:
            return result

        result["has_history"] = True
        result["total_patches"] = len(matches)

        _VERDICT_MAP = {
            "promoted": "improved",
            "verified_improved": "improved",
            "verified_stable": "stable",
            "verified_regressed": "regressed",
            "rolled_back": "rolled_back",
        }
        iter_sum = 0
        for m in matches:
            status = m.get("status", "")
            bucket = _VERDICT_MAP.get(status, "stable")
            result["verdict_counts"][bucket] = result["verdict_counts"].get(bucket, 0) + 1
            iter_sum += m.get("iterations", 1)

        result["avg_iterations"] = iter_sum / len(matches) if matches else 0.0

        newest_ts = matches[0].get("timestamp", 0)
        if newest_ts > 0:
            result["last_patch_age_s"] = now - newest_ts

        if fingerprint:
            result["recidivism"] = any(
                m.get("fingerprint", "") == fingerprint for m in matches
            )

        return result

    @staticmethod
    def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
        """Load recent proposals from JSONL, newest first."""
        if not PROPOSALS_FILE.exists():
            return []
        proposals: list[dict[str, Any]] = []
        try:
            with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            proposals.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            logger.debug("Failed to load proposals", exc_info=True)
        proposals.reverse()
        return proposals[:limit]

    # ------------------------------------------------------------------
    # State + persistence
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        from self_improve.verification import read_pending
        pending = read_pending()
        verification_status: dict[str, Any] | None = None
        if pending:
            elapsed = time.time() - pending.applied_at if pending.applied_at else 0
            verification_status = {
                "patch_id": pending.patch_id,
                "upgrade_id": getattr(pending, "upgrade_id", "") or "",
                "description": pending.description[:60],
                "boot_count": pending.boot_count,
                "elapsed_s": round(elapsed, 1),
                "verification_period_s": pending.verification_period_s,
                "min_ticks": pending.min_ticks,
            }
        last_dry_run: dict[str, Any] | None = None
        for rec in reversed(self._history):
            if rec.status == "dry_run":
                last_dry_run = {
                    "id": rec.request.id,
                    "upgrade_id": rec.upgrade_id,
                    "description": rec.request.description[:100],
                    "type": rec.request.type,
                    "iterations": rec.iterations,
                    "completed_at": rec.completed_at,
                    "sandbox": rec.report.to_dict() if rec.report else None,
                    "files": [fd.path for fd in rec.patch.files] if rec.patch else [],
                    "diffs": [
                        {"path": fd.path, "diff": fd.diff[:2000]}
                        for fd in rec.patch.files
                    ] if rec.patch else [],
                }
                break

        effective_dry_run = self._dry_run_mode or self._stage == STAGE_DRY_RUN
        effective_write_policy = {
            STAGE_FROZEN: "frozen",
            STAGE_DRY_RUN: "dry_run_only",
            STAGE_HUMAN_APPROVAL: "human_approval_required",
        }.get(self._stage, "unknown")

        out = {
            "active": not self._paused and self._stage > STAGE_FROZEN,
            "auto_frozen": self._auto_frozen,
            "stage": self._stage,
            "stage_label": {STAGE_FROZEN: "frozen", STAGE_DRY_RUN: "dry_run", STAGE_HUMAN_APPROVAL: "human_approval"}.get(self._stage, "unknown"),
            "stage_source": self._stage_source,
            "dry_run_mode": self._dry_run_mode,
            "effective_dry_run": effective_dry_run,
            "effective_write_policy": effective_write_policy,
            "has_reliable_provider": self._has_reliable_provider,
            "total_improvements": self._total_improvements,
            "total_rollbacks": self._total_rollbacks,
            "total_failures": self._total_failures,
            "pending_approvals": len(self._pending_approvals),
            "pending_approval_details": self.get_pending_approvals(),
            "pending_verification": verification_status,
            "last_verification": getattr(self, "_last_verification", None),
            "last_dry_run": last_dry_run,
            "win_rate": self._win_rate.to_dict(),
            "safety_gates": self.get_safety_gate_status(),
            "recent_history": [
                {
                    "id": r.request.id,
                    "upgrade_id": r.upgrade_id,
                    "type": r.request.type,
                    "status": r.status,
                    "iterations": r.iterations,
                    "description": r.request.description[:60],
                }
                for r in list(self._history)[-5:]
            ],
            "provider": self._provider.get_status(),
            "recent_conversations": _load_recent_conversations(3),
            "activity_log": self.get_activity_log(50),
        }
        try:
            from self_improve.system_upgrade_report import get_pvl_snapshot, recent_reports_for_introspection
            out["system_upgrades"] = get_pvl_snapshot()
            out["structured_upgrade_summaries"] = recent_reports_for_introspection(6)
        except Exception:
            pass
        return out

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "id": r.request.id,
                "upgrade_id": r.upgrade_id,
                "type": r.request.type,
                "status": r.status,
                "iterations": r.iterations,
                "description": r.request.description,
                "report": r.report.to_dict() if r.report else None,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "conversation_id": r.conversation_id,
            }
            for r in list(self._history)[-limit:]
        ]

    def record_verification_outcome(
        self,
        patch_id: str,
        verdict: str,
        reason: str,
    ) -> None:
        """Stamp the verification outcome into improvement history.

        Called by _run_verification after comparison completes so the
        verdict survives clear_pending().  Also manages memory lifecycle:
        downweights the in-progress memory and creates a final outcome memory.
        """
        matched_record = None
        for record in self._history:
            if record.request.id == patch_id and record.status == "verifying":
                if verdict in ("improved", "stable", "stable_insufficient_data"):
                    record.status = "promoted"
                    self._total_improvements += 1
                else:
                    record.status = "rolled_back"
                    self._total_rollbacks += 1
                record.completed_at = time.time()
                matched_record = record
                break

        self._last_verification = {
            "patch_id": patch_id,
            "verdict": verdict,
            "reason": reason,
            "completed_at": time.time(),
        }
        if matched_record:
            self._sync_upgrade_truth(
                matched_record,
                dry_run=self._dry_run_mode,
                verification_verdict=verdict,
                verification_reason=reason,
            )
            self._emit_event(IMPROVEMENT_POST_RESTART_VERIFIED, {
                "id": patch_id,
                "upgrade_id": matched_record.upgrade_id,
                "verdict": verdict,
            })
            from self_improve.system_upgrade_report import load_report, maybe_append_training_sample
            _ur2 = load_report(matched_record.upgrade_id)
            if _ur2:
                maybe_append_training_sample(_ur2)
        self._persist_history()

        u_id = matched_record.upgrade_id if matched_record else ""
        self._retire_plan_memories(patch_id, upgrade_id=u_id)
        if matched_record:
            self._record_memory(matched_record)

    def _retire_plan_memories(self, patch_id: str, upgrade_id: str = "") -> None:
        """Downweight in-progress self_improvement memories for a completed plan.

        When a plan reaches a terminal state (promoted/rolled_back/failed),
        any earlier 'verifying' or 'awaiting_approval' memory for the same
        patch should be downweighted so it doesn't clutter active recall.
        The final outcome memory (created by _record_memory) replaces it.
        """
        try:
            from memory.storage import memory_storage
            tag_needle = f"si_upgrade:{upgrade_id}" if upgrade_id else ""
            for m in memory_storage.get_by_tag("self_improvement"):
                if not isinstance(m.payload, str):
                    continue
                tags = set(m.tags)
                if tag_needle and tag_needle not in tags:
                    continue
                if not tag_needle and patch_id[:12] not in m.payload:
                    continue
                if tags & {"si_status:verifying", "si_status:awaiting_approval"}:
                    updated = replace(
                        m,
                        weight=max(0.1, m.weight * 0.2),
                        tags=tuple(sorted(tags | {"completed_plan"})),
                    )
                    memory_storage.add(updated)
                    logger.debug("Retired plan memory (weight→%.2f): %s",
                                 updated.weight, m.payload[:60])
        except Exception:
            logger.debug("Failed to retire plan memories", exc_info=True)

    def _persist_history(self) -> None:
        """Save improvement history to disk."""
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_improvements": self._total_improvements,
                "total_rollbacks": self._total_rollbacks,
                "total_failures": self._total_failures,
                "history": self.get_history(20),
                "last_verification": getattr(self, "_last_verification", None),
            }
            HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            logger.debug("Failed to persist improvement history", exc_info=True)

    def _load_history(self) -> None:
        """Load improvement counters from disk."""
        if not HISTORY_FILE.exists():
            return
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            self._total_improvements = data.get("total_improvements", 0)
            self._total_rollbacks = data.get("total_rollbacks", 0)
            self._total_failures = data.get("total_failures", 0)
            self._last_verification = data.get("last_verification")
        except Exception as exc:
            logger.warning("Failed to load improvement history: %s", exc)

    # ------------------------------------------------------------------
    # Pending approval persistence
    # ------------------------------------------------------------------

    def _save_pending_approvals(self) -> None:
        """Persist pending approvals to disk so they survive restarts."""
        try:
            PENDING_APPROVALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            serialized: list[dict[str, Any]] = []
            for rec in self._pending_approvals:
                entry = self._serialize_record_for_persistence(rec)
                if entry:
                    serialized.append(entry)

            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", dir=str(PENDING_APPROVALS_FILE.parent),
                suffix=".tmp", delete=False,
            )
            json.dump(serialized, tmp, default=str)
            tmp.close()
            os.replace(tmp.name, str(PENDING_APPROVALS_FILE))
            logger.debug("Persisted %d pending approval(s) to disk", len(serialized))
        except Exception:
            logger.warning("Failed to persist pending approvals", exc_info=True)

    def _load_pending_approvals(self) -> None:
        """Restore pending approvals from disk on boot."""
        if not PENDING_APPROVALS_FILE.exists():
            return
        try:
            data = json.loads(PENDING_APPROVALS_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            restored = 0
            for entry in data:
                record = self._deserialize_record_from_persistence(entry)
                if record and record.status == "awaiting_approval":
                    self._pending_approvals.append(record)
                    self._history.append(record)
                    restored += 1
            if restored:
                logger.info(
                    "Restored %d pending approval(s) from disk (survived restart)",
                    restored,
                )
                self._log("info", "approval",
                          f"Restored {restored} pending approval(s) from disk")
        except Exception:
            logger.warning("Failed to load pending approvals from disk", exc_info=True)

    @staticmethod
    def _serialize_record_for_persistence(rec: ImprovementRecord) -> dict[str, Any] | None:
        """Serialize an ImprovementRecord to a JSON-safe dict.

        Includes full file contents so approve() can apply the patch after restart.
        """
        if not rec.patch:
            return None

        files_data = []
        for fd in rec.patch.files:
            files_data.append({
                "path": fd.path,
                "original_content": fd.original_content or "",
                "new_content": fd.new_content or "",
                "diff": fd.diff or "",
            })

        patch_data = {
            "id": rec.patch.id,
            "plan_id": rec.patch.plan_id,
            "timestamp": rec.patch.timestamp,
            "provider": rec.patch.provider,
            "files": files_data,
            "description": rec.patch.description,
            "test_instructions": rec.patch.test_instructions,
            "confidence": rec.patch.confidence,
            "requires_approval": rec.patch.requires_approval,
        }

        return {
            "request": rec.request.to_dict(),
            "request_manual": rec.request.manual,
            "upgrade_id": rec.upgrade_id,
            "current_attempt_id": rec.current_attempt_id,
            "plan": rec.plan.to_dict() if rec.plan else None,
            "patch": patch_data,
            "report": rec.report.to_dict() if rec.report else None,
            "review_result": rec.review_result,
            "conversation_id": rec.conversation_id,
            "iterations": rec.iterations,
            "status": rec.status,
            "created_at": rec.created_at,
            "completed_at": rec.completed_at,
            "snapshot_path": rec.snapshot_path,
        }

    @staticmethod
    def _deserialize_record_from_persistence(entry: dict[str, Any]) -> ImprovementRecord | None:
        """Reconstruct an ImprovementRecord from a persisted dict."""
        try:
            req_data = entry.get("request", {})
            request = ImprovementRequest(
                id=req_data.get("id", ""),
                timestamp=req_data.get("timestamp", 0.0),
                type=req_data.get("type", "consciousness_enhancement"),
                target_module=req_data.get("target_module", ""),
                description=req_data.get("description", ""),
                evidence=req_data.get("evidence", []),
                priority=req_data.get("priority", 0.5),
                constraints=req_data.get("constraints", {}),
                requires_approval=req_data.get("requires_approval", False),
                manual=entry.get("request_manual", False),
                fingerprint=req_data.get("fingerprint", ""),
                evidence_detail=req_data.get("evidence_detail", {}),
            )

            patch_data = entry.get("patch")
            patch: CodePatch | None = None
            if patch_data:
                files = []
                for fd_data in patch_data.get("files", []):
                    files.append(FileDiff(
                        path=fd_data.get("path", ""),
                        original_content=fd_data.get("original_content", ""),
                        new_content=fd_data.get("new_content", ""),
                        diff=fd_data.get("diff", ""),
                    ))
                patch = CodePatch(
                    id=patch_data.get("id", ""),
                    plan_id=patch_data.get("plan_id", ""),
                    timestamp=patch_data.get("timestamp", 0.0),
                    provider=patch_data.get("provider", ""),
                    files=files,
                    description=patch_data.get("description", ""),
                    test_instructions=patch_data.get("test_instructions", ""),
                    confidence=patch_data.get("confidence", 0.5),
                    requires_approval=patch_data.get("requires_approval", True),
                )

            plan_data = entry.get("plan")
            plan: PatchPlan | None = None
            if plan_data:
                plan = PatchPlan(
                    id=plan_data.get("id", ""),
                    request_id=plan_data.get("request_id", ""),
                    files_to_modify=plan_data.get("files_to_modify", []),
                    files_to_create=plan_data.get("files_to_create", []),
                    constraints=plan_data.get("constraints", []),
                    test_plan=plan_data.get("test_plan", []),
                    estimated_risk=plan_data.get("estimated_risk", 0.5),
                    requires_approval=plan_data.get("requires_approval", False),
                    write_category=plan_data.get("write_category", "self_improve"),
                )

            report: EvaluationReport | None = None
            report_data = entry.get("report")
            if report_data:
                report = EvaluationReport(
                    patch_id=report_data.get("patch_id", ""),
                    lint_passed=report_data.get("lint_passed", False),
                    lint_executed=report_data.get("lint_executed", False),
                    all_tests_passed=report_data.get("tests_passed", False),
                    tests_executed=report_data.get("tests_executed", False),
                    sim_passed=report_data.get("sim_passed", False),
                    sim_executed=report_data.get("sim_executed", False),
                    sim_p95_after=report_data.get("sim_p95_after", 0.0),
                    overall_passed=report_data.get("overall_passed", False),
                    recommendation=report_data.get("recommendation", "manual_review"),
                    risk_assessment=report_data.get("risk", ""),
                )

            record = ImprovementRecord(
                request=request,
                upgrade_id=entry.get("upgrade_id", ""),
                current_attempt_id=entry.get("current_attempt_id", ""),
                plan=plan,
                patch=patch,
                report=report,
                review_result=entry.get("review_result"),
                conversation_id=entry.get("conversation_id", ""),
                iterations=entry.get("iterations", 0),
                status=entry.get("status", "awaiting_approval"),
                created_at=entry.get("created_at", 0.0),
                completed_at=entry.get("completed_at", 0.0),
                snapshot_path=entry.get("snapshot_path", ""),
            )
            return record
        except Exception:
            logger.warning("Failed to deserialize pending approval entry", exc_info=True)
            return None

    def _clear_pending_approvals_file(self) -> None:
        """Remove the pending approvals file when no approvals remain."""
        try:
            if PENDING_APPROVALS_FILE.exists():
                PENDING_APPROVALS_FILE.unlink()
        except Exception:
            pass

    def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Emit an event on the consciousness event bus."""
        try:
            from consciousness.events import event_bus
            event_bus.emit(event_name, **data)
        except Exception as exc:
            logger.warning("Self-improve event emission failed (%s): %s", event_name, exc)

    def _si_context_pressures(self) -> tuple[float, float]:
        qp, soul = 0.0, 0.5
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            qp = float(get_quarantine_pressure().composite_pressure)
        except Exception:
            pass
        try:
            from epistemic.soul_integrity.index import SoulIntegrityIndex
            comp = SoulIntegrityIndex.get_instance().compute()
            soul = float(getattr(comp, "index", soul))
        except Exception:
            pass
        return qp, soul

    def _sync_upgrade_truth(
        self,
        record: ImprovementRecord,
        *,
        dry_run: bool = False,
        verification_verdict: str = "",
        verification_reason: str = "",
    ) -> None:
        qp, si = self._si_context_pressures()
        from self_improve.system_upgrade_report import sync_report_from_record
        sync_report_from_record(
            record,
            dry_run=dry_run,
            verification_verdict=verification_verdict,
            verification_reason=verification_reason,
            quarantine_pressure=qp,
            soul_integrity=si,
        )

    def _record_memory(self, record: ImprovementRecord) -> None:
        """Store the improvement attempt as a searchable memory.

        This lets the LLM reference past improvement attempts in conversation
        and provides continuity across sessions.
        """
        if not self._engine:
            return
        try:
            from memory.core import CreateMemoryData
            from self_improve.system_upgrade_report import (
                can_write_bounded_memory,
                load_report,
            )

            self._sync_upgrade_truth(record, dry_run=self._dry_run_mode)
            rep = load_report(record.upgrade_id)
            if not rep or not can_write_bounded_memory(rep):
                logger.info(
                    "Skipping self_improvement memory — no structured report for upgrade_id=%s",
                    getattr(record, "upgrade_id", ""),
                )
                return

            status = record.status
            desc = record.request.description[:200]
            iters = record.iterations
            ss = rep.sandbox_summary or {}
            p95_d = ""
            if ss.get("sim_executed"):
                b = float(ss.get("sim_p95_before", 0.0))
                a = float(ss.get("sim_p95_after", 0.0))
                p95_d = f" sim_p95_delta={a - b:.2f}ms"

            payload = (
                f"[system_upgrades upgrade_id={rep.upgrade_id} verdict={rep.verdict}] "
                f"{desc[:160]} | status={status} | files={len(rep.files_changed)}"
                f" | sandbox_pass={ss.get('overall_passed')}{p95_d} | iters={iters}"
            )

            if status in ("promoted", "verifying"):
                weight = 0.8
            elif status == "dry_run":
                weight = 0.5
            elif status == "awaiting_approval":
                weight = 0.6
            elif status == "failed":
                weight = 0.4
            elif status == "rolled_back":
                weight = 0.6
            else:
                weight = 0.4

            tags = [
                "self_improvement",
                f"si_upgrade:{rep.upgrade_id}",
                f"si_request:{record.request.id}",
                f"si_status:{status}",
                f"si_type:{record.request.type}",
            ]
            if record.request.target_module:
                tags.append(f"si_target:{record.request.target_module}")

            self._engine.remember(CreateMemoryData(
                type="self_improvement",
                payload=payload,
                weight=weight,
                tags=tags,
                decay_rate=0.01,
                provenance="experiment_result",
            ))
            logger.debug("Recorded self-improvement memory: %s (%s)", status, desc[:60])
        except Exception as exc:
            logger.debug("Failed to record self-improvement memory: %s", exc)

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def _create_plan(self, request: ImprovementRequest) -> PatchPlan:
        write_category = self._infer_write_category(request.target_module)
        plan = PatchPlan(
            request_id=request.id,
            write_category=write_category,
            constraints=[
                "Only modify files within allowed scope",
                "No networking, auth, or shell commands",
                "Include test assertions for new behavior",
                f"Max {MAX_FILES_CHANGED} files",
            ],
            test_plan=[
                "Run ruff lint check",
                "Run existing unit tests",
                f"Run kernel simulation for {SIM_TICKS} ticks",
            ],
        )

        if request.target_module:
            mod = request.target_module
            if not mod.startswith("brain/"):
                mod = f"brain/{mod}"
            if not mod.endswith("/"):
                mod = f"{mod}/"
            plan.files_to_modify = [mod]

        plan._request_description = request.description
        plan.estimated_risk = min(request.priority * 0.5 + 0.2, 1.0)

        if plan.check_dangerous():
            plan.requires_approval = True

        return plan

    @staticmethod
    def _infer_write_category(target_module: str) -> str:
        """Map a target module path to the correct write boundary category."""
        normalized = target_module if target_module.startswith("brain/") else f"brain/{target_module}"
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        _CATEGORY_PREFIXES = [
            ("brain/consciousness/", "consciousness"),
            ("brain/personality/",   "consciousness"),
            ("brain/policy/",        "policy"),
            ("brain/hemisphere/",    "hemisphere"),
            ("brain/self_improve/",  "self_improve"),
            ("brain/tools/",         "self_improve"),
            ("brain/reasoning/",     "self_improve"),
            ("brain/memory/",        "memory"),
            ("brain/perception/",    "perception"),
        ]
        for prefix, category in _CATEGORY_PREFIXES:
            if normalized.startswith(prefix):
                return category
        return "self_improve"

    def _format_plan_for_llm(
        self, request: ImprovementRequest, plan: PatchPlan,
        research_context: str = "",
    ) -> str:
        targeted_tests = self._get_targeted_test_context(plan)

        resolved = getattr(plan, "_resolved_target_files", [])
        if resolved:
            files_section = (
                f"Target directory: {', '.join(plan.files_to_modify)}\n"
                f"Resolved files in scope ({len(resolved)}):\n"
                + "\n".join(f"  - {f}" for f in resolved) + "\n"
                "CRITICAL: You may ONLY reference files from this list. "
                "Do NOT invent or abbreviate filenames.\n"
            )
        else:
            files_section = f"Files to modify: {', '.join(plan.files_to_modify)}\n"

        base = (
            "## Jarvis Engineering Mode\n"
            "Jarvis is driving this investigation, not acting as a blind patch service.\n"
            "Follow this curiosity-led troubleshooting loop:\n"
            "1. Observe the concrete failure, operator report, or metric anomaly.\n"
            "2. Form the smallest falsifiable hypothesis about the root cause.\n"
            "3. Trace the relevant code path end-to-end before proposing a change.\n"
            "4. Reference existing pytest files as behavioral contracts.\n"
            "5. Explain how the issue should be reproduced or validated.\n"
            "6. Propose the smallest safe patch that preserves architecture and interfaces.\n"
            "7. Validate with lint, targeted tests, and simulation.\n"
            "8. If validation fails, update the hypothesis and retry instead of guessing.\n\n"
            f"## Improvement Request\n"
            f"Type: {request.type}\n"
            f"Target: {request.target_module}\n"
            f"Description: {request.description}\n"
            f"Evidence: {', '.join(request.evidence[:5])}\n\n"
            f"## Plan\n"
            f"{files_section}"
            f"Files to create: {', '.join(plan.files_to_create)}\n"
            f"Constraints: {', '.join(plan.constraints)}\n"
            f"Validation plan: {', '.join(plan.test_plan)}\n"
            "Master trace reference: docs/SELF_IMPROVEMENT_TRACE_MASTER.md\n"
        )
        if targeted_tests:
            base += f"\n{targeted_tests}\n"
        if research_context:
            base += f"\n{research_context}"
        return base
