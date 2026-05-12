"""Job Exit Condition Evaluator.

Parses condition strings and evaluates them against a job's current state.

Condition language (v1):
    gate:<gate_id>            -> gate must have state "pass"
    artifact:<type_or_id>     -> must exist in job.artifacts
    metric:<name><op><value>  -> compare job.data.counters (>=, <=, ==, >, <)
    evidence:<test_name>      -> must exist in evidence history with passed=True
    skill_status:verified     -> skill must be verified in registry
    protocol:<id>:passed      -> Matrix Protocol check must have passed
    protocol:<id>:claimable   -> claimability_status must be verified_*
"""

from __future__ import annotations

import operator
import re
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from skills.learning_jobs import LearningJob
    from skills.registry import SkillRegistry

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}

_METRIC_RE = re.compile(
    r"^metric:(?P<name>[a-zA-Z0-9_]+)\s*(?P<op>>=|<=|==|>|<)\s*(?P<val>-?\d+(?:\.\d+)?)$"
)


def _gate_passed(job: LearningJob, gate_id: str) -> bool:
    prefixed = f"gate:{gate_id}" if not gate_id.startswith("gate:") else gate_id
    bare = gate_id.removeprefix("gate:")
    for bucket in ("hard", "soft"):
        for g in job.gates.get(bucket, []):
            gid = g.get("id", "")
            if gid == prefixed or gid == bare or gid == gate_id:
                return g.get("state") == "pass"
    return False


def _has_artifact(job: LearningJob, token: str) -> bool:
    for a in job.artifacts:
        if a.get("id") == token or a.get("type") == token:
            return True
    return False


def _metric_ok(job: LearningJob, expr: str) -> bool:
    m = _METRIC_RE.match(expr)
    if not m:
        return False
    name = m.group("name")
    op_str = m.group("op")
    val = float(m.group("val"))
    cur = float(job.data.get("counters", {}).get(name, 0))
    return _OPS[op_str](cur, val)


def _evidence_present(job: LearningJob, test_name: str) -> bool:
    for evd in reversed(job.evidence.get("history", [])):
        for t in evd.get("tests", []):
            if t.get("name") == test_name and t.get("passed") is True:
                return True
    return False


def _skill_is_verified(registry: SkillRegistry, skill_id: str) -> bool:
    rec = registry.get(skill_id)
    if rec is None:
        return False
    return rec.status == "verified"


def check_exit_conditions(
    job: LearningJob,
    registry: SkillRegistry,
    exit_conditions: list[str],
) -> tuple[bool, list[str]]:
    """Evaluate a list of exit condition strings.

    Returns ``(all_met, list_of_unmet_conditions)``.
    """
    unmet: list[str] = []
    for cond in exit_conditions:
        cond = cond.strip()
        if not cond:
            continue

        if cond.startswith("gate:"):
            gate_id = cond.split(":", 1)[1]
            if not _gate_passed(job, gate_id):
                unmet.append(cond)

        elif cond.startswith("artifact:"):
            token = cond.split(":", 1)[1]
            if not _has_artifact(job, token):
                unmet.append(cond)

        elif cond.startswith("metric:"):
            if not _metric_ok(job, cond):
                unmet.append(cond)

        elif cond.startswith("evidence:"):
            test_name = cond.split(":", 1)[1]
            found = _evidence_present(job, test_name)
            if not found and ":" in test_name:
                bare_name = test_name.split(":", 1)[1]
                found = _evidence_present(job, bare_name)
            if not found:
                unmet.append(cond)

        elif cond.startswith("skill_status:"):
            want = cond.split(":", 1)[1]
            if want == "verified":
                if not _skill_is_verified(registry, job.skill_id):
                    unmet.append(cond)
            else:
                unmet.append(cond)

        elif cond.startswith("protocol:"):
            if not _protocol_condition_met(job, cond):
                unmet.append(cond)

        else:
            unmet.append(cond)

    return (len(unmet) == 0), unmet


def _protocol_condition_met(job: LearningJob, cond: str) -> bool:
    """Evaluate protocol-based exit conditions.

    Formats:
      protocol:<id>:passed    -> run protocol, check passed==True
      protocol:<id>:claimable -> check claimability_status is verified_*
    """
    parts = cond.split(":")
    if len(parts) < 3:
        return False
    proto_id = parts[1]
    check = parts[2]

    if not getattr(job, "matrix_protocol", False):
        return False

    if check == "passed":
        try:
            from skills.verification_protocols import evaluate_job
            result = evaluate_job(job)
            return result is not None and result.passed
        except Exception:
            return False
    elif check == "claimable":
        claim = getattr(job, "claimability_status", "unverified")
        return claim in ("verified_limited", "verified_operational")
    return False
