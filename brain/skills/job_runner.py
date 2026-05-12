"""Job Runner — auto-advance engine that moves jobs through phases.

Reads the job's ``plan.phases`` list, finds the current phase entry, checks
exit conditions via ``job_eval``, and advances to the next phase if ready.
If a hard gate is unmet, the job is marked ``blocked``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skills.learning_jobs import LearningJob, LearningJobOrchestrator
    from skills.registry import SkillRegistry

from skills.job_eval import check_exit_conditions

logger = logging.getLogger(__name__)


def _find_phase_index(job: LearningJob) -> int:
    phases = job.plan.get("phases", [])
    for i, p in enumerate(phases):
        if p.get("name") == job.phase:
            return i
    return -1


def try_auto_advance(
    job: LearningJob,
    registry: SkillRegistry,
    orch: LearningJobOrchestrator,
) -> bool:
    """Attempt to advance the job by at most one phase.

    Returns True if the job advanced.  Never skips phases.
    """
    if getattr(job, "status", "") == "awaiting_operator_approval":
        return False

    phases = job.plan.get("phases", [])
    idx = _find_phase_index(job)
    if idx < 0 or idx >= len(phases):
        return False

    exit_conditions = phases[idx].get("exit_conditions", [])
    if not exit_conditions:
        # No conditions means the phase completes immediately
        pass
    else:
        ready, unmet = check_exit_conditions(job, registry, exit_conditions)
        if not ready:
            hard_ids = {
                g.get("id") for g in job.gates.get("hard", []) if g.get("required")
            }
            hard_ids_bare = {gid.removeprefix("gate:") for gid in hard_ids}
            has_hard_block = any(
                u.startswith("gate:") and (
                    u.split(":", 1)[1] in hard_ids
                    or u.split(":", 1)[1] in hard_ids_bare
                    or u in hard_ids
                )
                for u in unmet
            )
            if has_hard_block and job.status != "blocked":
                job.status = "blocked"
                job.events.append({
                    "ts": job.updated_at,
                    "type": "job_blocked",
                    "msg": f"Unmet hard gates: {unmet[:3]}",
                })
                orch.store.save(job)
            return False

    # Ready to advance
    if idx + 1 >= len(phases):
        if job.status != "completed":
            orch.complete_job(job)
            logger.info("Job %s completed all %d phases", job.job_id, len(phases))
        return False

    next_phase_entry = phases[idx + 1]
    next_phase = next_phase_entry.get("name")
    if not next_phase:
        return False

    if job.status == "blocked":
        job.status = "active"

    orch.advance_phase(job, next_phase)
    return True
