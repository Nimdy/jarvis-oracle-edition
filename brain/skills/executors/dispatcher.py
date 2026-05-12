"""Executor Dispatcher — matches jobs to executors and applies results."""

from __future__ import annotations

import logging
from typing import Any

from skills.executors.base import PhaseExecutor, PhaseResult

logger = logging.getLogger(__name__)


class ExecutorDispatcher:
    """Dispatches a job tick to the matching phase executor, applies results,
    then attempts auto-advance."""

    def __init__(self, executors: list[PhaseExecutor] | None = None) -> None:
        self.executors: list[PhaseExecutor] = executors or []

    def register(self, executor: PhaseExecutor) -> None:
        self.executors.append(executor)

    def tick_one_job(
        self,
        job: Any,
        ctx: dict[str, Any],
        job_orch: Any,
        registry: Any,
    ) -> PhaseResult | None:
        """Run the matching executor for a job, apply side effects, try auto-advance."""
        for ex in self.executors:
            if not ex.can_run(job, ctx):
                continue

            try:
                res = ex.run(job, ctx)
            except Exception:
                logger.exception(
                    "Executor %s.%s failed for job %s",
                    ex.capability_type, ex.phase, job.job_id,
                )
                continue

            if res.gate_updates:
                for g in res.gate_updates:
                    job_orch.set_gate(job, g["id"], g["state"], g.get("details", ""))

            if res.metric_updates:
                job.data.setdefault("counters", {})
                for k, v in res.metric_updates.items():
                    new_value = float(v)
                    current_value = float(job.data["counters"].get(k, 0.0) or 0.0)
                    # Learning-job executor metrics are monotonic counters today. Preserve the
                    # highest observed value so guided collect or external updates are not
                    # accidentally erased by a later executor tick that sees a smaller total.
                    job.data["counters"][k] = max(current_value, new_value)
                job_orch.store.save(job)

            if res.artifact:
                job_orch.add_artifact(job, res.artifact)

            if res.evidence:
                latest = job.evidence.get("latest")
                is_dup = (
                    latest
                    and latest.get("result") == res.evidence.get("result")
                    and latest.get("tests") == res.evidence.get("tests")
                )
                if not is_dup:
                    job_orch.record_evidence(job, res.evidence)

            if not res.progressed:
                handoff = (getattr(job, "data", {}) or {}).get("operational_handoff", {})
                terminal_handoff = (
                    getattr(job, "status", "") == "blocked"
                    and isinstance(handoff, dict)
                    and handoff.get("status") in {"acquisition_failed", "acquisition_cancelled"}
                )
                if not terminal_handoff:
                    fail = job.failure
                    if fail.get("last_failed_phase") == job.phase:
                        fail["count"] = fail.get("count", 0) + 1
                    else:
                        fail["count"] = 1
                        fail["last_failed_phase"] = job.phase
                    fail["last_error"] = res.message
                job_orch.store.save(job)

            try:
                from skills.job_runner import try_auto_advance
                try_auto_advance(job, registry, job_orch)
            except Exception:
                logger.exception("Auto-advance after executor failed for %s", job.job_id)

            return res

        # No executor matched — still try auto-advance (gates may have been satisfied externally)
        try:
            from skills.job_runner import try_auto_advance
            advanced = try_auto_advance(job, registry, job_orch)
            if advanced:
                return PhaseResult(progressed=True, message="Auto-advanced (exit conditions met externally).")
        except Exception:
            pass

        return None
