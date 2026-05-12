"""Skill Tool — handles explicit 'learn X' user requests.

Routes through the SkillResolver to classify the request, registers the skill
in the SkillRegistry (if not already present), and creates a LearningJob.
Returns a human-readable status string for the LLM to personalize.

When the user invokes the Matrix Protocol ("matrix learn X"), the tool
sets ``matrix_protocol=True`` on the job, selects a verification protocol
(SK-001 through SK-004), and requests deep_learning mode.

The orchestrator instance is set at startup from ``main.py``.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TypedDict
from typing import Any

logger = logging.getLogger(__name__)

_learning_job_orch: Any = None
_skill_registry: Any = None

_MATRIX_STRIP_RE = re.compile(
    r"\b(?:use the )?matrix\b[\s\-]*(?:to |style |protocol )?\s*",
    re.IGNORECASE,
)
_GUIDED_COLLECT_REQUEST_RE = re.compile(
    r"\b(?:help train|start training mode|training mode|calibration round|calibrate|contribute|collect samples|let'?s train)\b",
    re.IGNORECASE,
)
_GUIDED_COLLECT_CANCEL_RE = re.compile(
    r"^(?:stop|cancel|never mind|nevermind|not now|next time|quit)$",
    re.IGNORECASE,
)

_PROTOCOL_MAP: dict[str, str] = {
    "procedural": "SK-001",
    "perceptual": "SK-002",
    "control": "SK-003",
}


class SkillToolResult(TypedDict, total=False):
    outcome: str
    message: str
    skill_id: str
    skill_name: str
    capability_type: str
    job_id: str
    phase: str
    status: str
    risk_level: str
    matrix_protocol: bool
    protocol_id: str
    prompt: str
    remaining_steps: int
    guided_session_id: str


def _build_guided_collect_config(job: Any) -> dict[str, Any]:
    try:
        from skills.verification_protocols import build_collect_runtime_config
        return build_collect_runtime_config(job)
    except Exception:
        logger.debug("Collect runtime config resolution failed", exc_info=True)
        return {}


def _find_active_job(skill_id: str) -> Any | None:
    if _learning_job_orch is None:
        return None
    for job in _learning_job_orch.get_active_jobs():
        if getattr(job, "skill_id", "") == skill_id:
            return job
    return None


def _start_guided_collect_for_job(job: Any, *, speaker: str) -> SkillToolResult:
    config = _build_guided_collect_config(job)
    skill_name = getattr(job, "skill_id", "").replace("_", " ")
    if not config:
        return {
            "outcome": "guided_collect_not_available",
            "status": getattr(job, "status", "unknown"),
            "skill_id": getattr(job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(job, "phase", ""),
            "message": f"I don't have a guided collect session defined for {skill_name} yet.",
        }
    if not bool(config.get("interactive_collect", False)):
        return {
            "outcome": "guided_collect_not_available",
            "status": getattr(job, "status", "unknown"),
            "skill_id": getattr(job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(job, "phase", ""),
            "message": (
                f"{skill_name} is collecting evidence autonomously right now. "
                "It does not currently need a user-guided training round."
            ),
        }
    if getattr(job, "phase", "") != "collect":
        return {
            "outcome": "guided_collect_not_needed",
            "status": getattr(job, "status", "unknown"),
            "skill_id": getattr(job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(job, "phase", ""),
            "message": (
                f"{skill_name} is currently in phase '{getattr(job, 'phase', 'unknown')}', "
                "so a collect-phase calibration round is not needed right now."
            ),
        }

    import uuid

    session_id = f"gcollect_{uuid.uuid4().hex[:8]}"
    job.executor_state["guided_collect"] = {
        "active": True,
        "session_id": session_id,
        "speaker": speaker or "user",
        "started_at": time.time(),
        **config,
    }
    job.events.append({
        "ts": getattr(job, "updated_at", ""),
        "type": "guided_collect_started",
        "msg": f"Guided collect session started for {speaker or 'user'}",
    })
    if _learning_job_orch is not None and getattr(_learning_job_orch, "store", None) is not None:
        _learning_job_orch.store.save(job)
    return {
        "outcome": "guided_collect_started",
        "status": getattr(job, "status", "unknown"),
        "skill_id": getattr(job, "skill_id", ""),
        "skill_name": skill_name,
        "phase": getattr(job, "phase", ""),
        "job_id": getattr(job, "job_id", ""),
        "prompt": str(config.get("prompt", "") or ""),
        "remaining_steps": int(config.get("remaining_steps", 0) or 0),
        "guided_session_id": session_id,
        "message": (
            f"I started a guided collect session for {skill_name}. "
            f"{str(config.get('prompt', '') or '')}"
        ),
    }


def consume_guided_collect_turn(
    *,
    speaker: str,
    user_text: str,
    emotion: str = "",
    conversation_id: str = "",
) -> SkillToolResult | None:
    if _learning_job_orch is None:
        return None

    active_job = None
    session: dict[str, Any] | None = None
    for job in _learning_job_orch.get_active_jobs():
        candidate = getattr(job, "executor_state", {}).get("guided_collect")
        if not isinstance(candidate, dict) or not candidate.get("active"):
            continue
        if candidate.get("speaker", "user") != (speaker or "user"):
            continue
        active_job = job
        session = candidate
        break

    if active_job is None or session is None:
        return None

    skill_name = getattr(active_job, "skill_id", "").replace("_", " ")
    if _GUIDED_COLLECT_CANCEL_RE.match((user_text or "").strip()):
        session["active"] = False
        active_job.events.append({
            "ts": getattr(active_job, "updated_at", ""),
            "type": "guided_collect_cancelled",
            "msg": "User cancelled guided collect session",
        })
        _learning_job_orch.store.save(active_job)
        return {
            "outcome": "guided_collect_cancelled",
            "status": getattr(active_job, "status", "unknown"),
            "skill_id": getattr(active_job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(active_job, "phase", ""),
            "job_id": getattr(active_job, "job_id", ""),
            "message": f"Okay, I stopped the guided collect session for {skill_name}.",
        }

    if session.get("parser") == "labeled_text" or session.get("mode") == "open_labeled":
        try:
            from skills.verification_protocols import parse_collect_submission, build_collect_artifact
            parsed_sample = parse_collect_submission(session, user_text or "")
        except Exception:
            logger.debug("Collect submission parsing failed", exc_info=True)
            parsed_sample = {"ok": False, "error": "I couldn't parse that collect sample right now."}

        if not parsed_sample.get("ok"):
            prompt = str(session.get("prompt", "") or "")
            return {
                "outcome": "guided_collect_continue",
                "status": getattr(active_job, "status", "unknown"),
                "skill_id": getattr(active_job, "skill_id", ""),
                "skill_name": skill_name,
                "phase": getattr(active_job, "phase", ""),
                "job_id": getattr(active_job, "job_id", ""),
                "prompt": prompt,
                "guided_session_id": str(session.get("session_id", "") or ""),
                "message": (
                    str(parsed_sample.get("error", "") or f"I couldn't parse that sample for {skill_name}.")
                ),
            }

        label = str(parsed_sample.get("label", "") or "").strip().lower()
        metric_name = str(session.get("metric_name", "") or "")
        counters = active_job.data.setdefault("counters", {})
        counters[metric_name] = float(counters.get(metric_name, 0) or 0) + 1.0
        captured = int(session.get("captured_count", 0) or 0) + 1
        session["captured_count"] = captured
        artifact = build_collect_artifact(
            session,
            speaker=speaker,
            emotion=emotion,
            conversation_id=conversation_id,
            metric_name=metric_name,
            captured_index=captured - 1,
            parsed_sample=parsed_sample,
        )
        _learning_job_orch.add_artifact(active_job, artifact)
        _learning_job_orch.store.save(active_job)

        remaining = 0
        try:
            from skills.verification_protocols import build_collect_runtime_config
            runtime_config = build_collect_runtime_config(active_job)
            remaining = max(0, int(runtime_config.get("remaining_count", 0) or 0))
        except Exception:
            remaining = 0
        prompt = str(session.get("prompt", "") or "")
        return {
            "outcome": "guided_collect_continue",
            "status": getattr(active_job, "status", "unknown"),
            "skill_id": getattr(active_job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(active_job, "phase", ""),
            "job_id": getattr(active_job, "job_id", ""),
            "prompt": prompt,
            "remaining_steps": remaining,
            "guided_session_id": str(session.get("session_id", "") or ""),
            "message": (
                f"Recorded the '{label}' sample for {skill_name}. "
                f"About {remaining} more {metric_name.replace('_', ' ')} are still needed."
            ),
        }

    steps = list(session.get("steps", []) or [])
    step_index = int(session.get("step_index", 0) or 0)
    if not steps or step_index >= len(steps):
        session["active"] = False
        _learning_job_orch.store.save(active_job)
        return None

    step = steps[step_index]
    label = str(step.get("label", "") or f"step_{step_index}")
    metric_name = str(session.get("metric_name", "") or _derive_collect_metric(active_job).get("metric_name", "") or "")
    if not metric_name:
        metric_name = "samples"
    counters = active_job.data.setdefault("counters", {})
    counters[metric_name] = float(counters.get(metric_name, 0) or 0) + 1.0

    artifact = {
        "id": f"guided_collect_{session.get('session_id', 'session')}_{step_index}",
        "type": "guided_collect_sample",
        "details": {
            "label": label,
            "text": (user_text or "")[:200],
            "speaker": speaker or "user",
            "emotion": emotion or "",
            "conversation_id": conversation_id or "",
            "session_id": session.get("session_id", ""),
        },
    }
    _learning_job_orch.add_artifact(active_job, artifact)

    step_index += 1
    if step_index >= len(steps):
        session["active"] = False
        session["completed_at"] = time.time()
        session["step_index"] = step_index
        active_job.events.append({
            "ts": getattr(active_job, "updated_at", ""),
            "type": "guided_collect_completed",
            "msg": f"Guided collect completed with {len(steps)} captured samples",
        })
        _learning_job_orch.store.save(active_job)
        return {
            "outcome": "guided_collect_completed",
            "status": getattr(active_job, "status", "unknown"),
            "skill_id": getattr(active_job, "skill_id", ""),
            "skill_name": skill_name,
            "phase": getattr(active_job, "phase", ""),
            "job_id": getattr(active_job, "job_id", ""),
            "message": (
                f"Recorded that sample for {skill_name}. "
                f"The guided collect session is complete, and I saved {len(steps)} labeled samples."
            ),
        }

    session["step_index"] = step_index
    _learning_job_orch.store.save(active_job)
    next_prompt = steps[step_index]["prompt"]
    return {
        "outcome": "guided_collect_continue",
        "status": getattr(active_job, "status", "unknown"),
        "skill_id": getattr(active_job, "skill_id", ""),
        "skill_name": skill_name,
        "phase": getattr(active_job, "phase", ""),
        "job_id": getattr(active_job, "job_id", ""),
        "prompt": next_prompt,
        "remaining_steps": len(steps) - step_index,
        "guided_session_id": str(session.get("session_id", "") or ""),
        "message": f"Recorded the {label} sample for {skill_name}. {next_prompt}",
    }


def set_orchestrator(orch: Any) -> None:
    global _learning_job_orch
    _learning_job_orch = orch


def set_registry(registry: Any) -> None:
    global _skill_registry
    _skill_registry = registry


def _normalize_matrix_input(user_text: str) -> str:
    """Strip Matrix trigger aliases from text so the resolver sees clean input."""
    cleaned = _MATRIX_STRIP_RE.sub("", user_text).strip()
    cleaned = re.sub(r"^learn\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else user_text


def _select_protocol(capability_type: str) -> str:
    """Pick the right verification protocol for a capability class."""
    return _PROTOCOL_MAP.get(capability_type, "SK-001")


def _serialize_capability_contract(resolution: Any) -> dict[str, Any]:
    """Copy resolver contract metadata into the learning-job plan."""
    capability = getattr(resolution, "capability", None)
    if capability is None:
        return {}
    return {
        "input_type": getattr(capability, "input_type", ""),
        "output_type": getattr(capability, "output_type", ""),
        "success_metrics": list(getattr(capability, "success_metrics", ()) or ()),
        "evidence_requirements": list(getattr(capability, "evidence_requirements", ()) or ()),
        "hardware_requirements": list(getattr(capability, "hardware_requirements", ()) or ()),
        "execution_contract_id": getattr(capability, "execution_contract_id", ""),
        "required_executor_kind": getattr(capability, "required_executor_kind", ""),
        "acquisition_eligible": bool(getattr(capability, "acquisition_eligible", False)),
    }


def _request_deep_learning_mode() -> None:
    """Ask the mode manager to transition to deep_learning for Matrix jobs."""
    try:
        from consciousness.events import event_bus, MATRIX_DEEP_LEARNING_REQUESTED
        event_bus.emit(MATRIX_DEEP_LEARNING_REQUESTED)
    except Exception:
        logger.debug("Could not request deep_learning mode for Matrix job", exc_info=True)


def _get_available_skill_descriptions() -> list[str]:
    """Return human-readable names of all skill templates."""
    try:
        from skills.resolver import SKILL_TEMPLATES
    except ImportError:
        return []
    seen: set[str] = set()
    names: list[str] = []
    for _, resolution in SKILL_TEMPLATES:
        if resolution.name not in seen:
            seen.add(resolution.name)
            names.append(resolution.name.lower())
    return names


def handle_skill_request_structured(
    user_text: str,
    speaker: str = "user",
    matrix_trigger: bool = False,
) -> SkillToolResult:
    """Process an explicit 'learn X' request.

    When *matrix_trigger* is True, the job runs under the Matrix Protocol
    with stricter verification and a request for deep_learning mode.

    Returns a structured result for bounded articulation or LLM personalization.
    """
    if _learning_job_orch is None or _skill_registry is None:
        return {
            "outcome": "system_uninitialized",
            "status": "unavailable",
            "message": "My learning system isn't initialized yet. I can't start learning jobs right now.",
        }

    try:
        from skills.resolver import is_generic_fallback_resolution, resolve_skill
    except ImportError:
        return {
            "outcome": "resolver_unavailable",
            "status": "unavailable",
            "message": "My skill resolver is unavailable right now.",
        }

    resolve_text = _normalize_matrix_input(user_text) if matrix_trigger else user_text
    resolution = resolve_skill(resolve_text)
    if resolution is None:
        return {
            "outcome": "unresolved_request",
            "status": "unknown",
            "message": "I couldn't determine what skill to learn from that request.",
        }
    if is_generic_fallback_resolution(resolution):
        available = _get_available_skill_descriptions()
        return {
            "outcome": "generic_fallback",
            "status": "blocked",
            "skill_id": resolution.skill_id,
            "skill_name": resolution.name,
            "capability_type": resolution.capability_type,
            "available_skills": available,
            "message": (
                f"I don't have a structured learning path for '{resolution.name}' yet. "
                f"I can currently learn: {', '.join(available)}. "
                f"If what you want is close to one of those, try asking for it specifically."
            ),
        }

    existing = _skill_registry.get(resolution.skill_id)
    if existing is not None and existing.status == "verified":
        return {
            "outcome": "already_verified",
            "status": existing.status,
            "skill_id": existing.skill_id,
            "skill_name": existing.name,
            "capability_type": existing.capability_type,
            "message": (
                f"I already have '{existing.name}' as a verified skill. "
                f"No learning job needed."
            ),
        }

    if existing is not None and existing.status == "learning":
        active = _learning_job_orch.get_active_jobs()
        for job in active:
            if job.skill_id == resolution.skill_id:
                if _GUIDED_COLLECT_REQUEST_RE.search(user_text):
                    return _start_guided_collect_for_job(job, speaker=speaker)
                return {
                    "outcome": "already_learning",
                    "status": job.status,
                    "skill_id": existing.skill_id,
                    "skill_name": existing.name,
                    "capability_type": existing.capability_type,
                    "job_id": job.job_id,
                    "phase": job.phase,
                    "message": (
                        f"I'm already learning '{existing.name}' — "
                        f"job {job.job_id} is in phase '{job.phase}' (status: {job.status})."
                    ),
                }
        return {
            "outcome": "restart_learning",
            "status": existing.status,
            "skill_id": existing.skill_id,
            "skill_name": existing.name,
            "capability_type": existing.capability_type,
            "message": (
                f"'{existing.name}' is marked as learning but has no active job. "
                f"Let me start a new one."
            ),
        }

    registered_this_request = False
    if existing is None:
        from skills.registry import SkillRecord
        record = SkillRecord(
            skill_id=resolution.skill_id,
            name=resolution.name,
            status="unknown",
            capability_type=resolution.capability_type,
            verification_required=[e for e in resolution.required_evidence],
            notes=resolution.notes,
        )
        _skill_registry.register(record)
        registered_this_request = True

    job = _learning_job_orch.create_job(
        skill_id=resolution.skill_id,
        capability_type=resolution.capability_type,
        requested_by={"source": "user", "user_text": user_text, "speaker": speaker,
                       "matrix_trigger": matrix_trigger},
        risk_level=resolution.risk_level,
        required_evidence=resolution.required_evidence,
        plan={
            "summary": resolution.notes,
            "phases": resolution.default_phases,
            "guided_collect": dict(getattr(resolution, "guided_collect", None) or {}),
            "capability_contract": _serialize_capability_contract(resolution),
        },
        hard_gates=resolution.hard_gates,
    )

    if job is None:
        # Roll back the speculative registration so failed requests do not leave
        # ghost records on disk. Only roll back records this request created —
        # pre-existing records (e.g. from a prior verified skill) stay intact.
        if registered_this_request:
            try:
                _skill_registry.remove(resolution.skill_id)
            except Exception:
                logger.debug(
                    "Failed to roll back speculative skill registration for %s",
                    resolution.skill_id,
                    exc_info=True,
                )
        return {
            "outcome": "job_creation_failed",
            "status": "blocked",
            "skill_id": resolution.skill_id,
            "skill_name": resolution.name,
            "capability_type": resolution.capability_type,
            "message": (
                f"I couldn't create a learning job for '{resolution.name}' — "
                f"the skill ID '{resolution.skill_id}' was rejected as non-actionable."
            ),
        }

    if matrix_trigger:
        protocol_id = _select_protocol(resolution.capability_type)
        job.matrix_protocol = True
        job.protocol_id = protocol_id
        job.matrix_target = resolve_text
        job.verification_profile = protocol_id
        job.claimability_status = "unverified"
        job.events.append({
            "ts": job.created_at,
            "type": "matrix_protocol_activated",
            "msg": f"Matrix Protocol activated — protocol {protocol_id}, "
                   f"capability class {resolution.capability_type}",
        })
        _learning_job_orch.store.save(job)
        _request_deep_learning_mode()

    risk_note = ""
    if resolution.risk_level == "high":
        risk_note = " This is a high-risk skill — it will require safety gates and user presence."
    elif resolution.risk_level == "medium":
        risk_note = " This has moderate risk and will go through careful verification."

    protocol_note = ""
    if matrix_trigger:
        protocol_note = (
            f" Matrix Protocol engaged — verification protocol {job.protocol_id} "
            f"for {resolution.capability_type} capability class."
        )

    return {
        "outcome": "job_started",
        "status": job.status,
        "skill_id": resolution.skill_id,
        "skill_name": resolution.name,
        "capability_type": resolution.capability_type,
        "job_id": job.job_id,
        "phase": job.phase,
        "risk_level": resolution.risk_level,
        "matrix_protocol": matrix_trigger,
        "protocol_id": getattr(job, "protocol_id", ""),
        "message": (
            f"I started a learning job for '{resolution.name}' "
            f"(job: {job.job_id}, type: {resolution.capability_type}, "
            f"phase: {job.phase}).{risk_note}{protocol_note} "
            f"I'll work through the phases: assess, research, acquire, integrate, "
            f"verify, and register. I won't claim this skill until evidence proves it works."
        ),
    }


def handle_skill_request(
    user_text: str,
    speaker: str = "user",
    matrix_trigger: bool = False,
) -> str:
    result = handle_skill_request_structured(
        user_text,
        speaker=speaker,
        matrix_trigger=matrix_trigger,
    )
    return result.get("message", "I had trouble processing that skill learning request.")
