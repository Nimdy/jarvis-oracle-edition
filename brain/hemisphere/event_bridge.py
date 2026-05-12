"""Hemisphere event bridge: emits events and KERNEL_THOUGHT reflections.

Ported from delete_later/neural-evolution/interfaces/EventBridge.ts.
Generates first-person self-reflective thoughts so the AI is aware of
its own neural evolution.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from consciousness.events import event_bus, KERNEL_THOUGHT, ACQUISITION_PLAN_REVIEWED

logger = logging.getLogger(__name__)

_plan_review_cleanup: Any = None

# ---------------------------------------------------------------------------
# Hemisphere event constants
# ---------------------------------------------------------------------------

HEMISPHERE_ARCHITECTURE_DESIGNED = "hemisphere:architecture_designed"
HEMISPHERE_TRAINING_PROGRESS = "hemisphere:training_progress"
HEMISPHERE_NETWORK_READY = "hemisphere:network_ready"
HEMISPHERE_EVOLUTION_COMPLETE = "hemisphere:evolution_complete"
HEMISPHERE_MIGRATION_DECISION = "hemisphere:migration_decision"
HEMISPHERE_SUBSTRATE_MIGRATION = "hemisphere:substrate_migration"
HEMISPHERE_PERFORMANCE_WARNING = "hemisphere:performance_warning"


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_construction_event(
    network_id: str, phase: str, message: str,
) -> None:
    _safe_emit(HEMISPHERE_TRAINING_PROGRESS, {
        "network_id": network_id,
        "phase": phase,
        "message": message,
        "timestamp": _time.time(),
    })


def emit_architecture_designed(
    network_id: str,
    strategy: str,
    total_parameters: int,
    focus: str,
    reasoning: str,
) -> None:
    _safe_emit(HEMISPHERE_ARCHITECTURE_DESIGNED, {
        "network_id": network_id,
        "strategy": strategy,
        "total_parameters": total_parameters,
        "focus": focus,
        "reasoning": reasoning,
        "timestamp": _time.time(),
    })
    _safe_emit(KERNEL_THOUGHT, {
        "content": (
            f"I've designed a new {focus} neural architecture: {strategy} approach "
            f"with {total_parameters:,} parameters"
        ),
        "tone": "focused",
    })


def emit_network_ready(
    network_id: str, name: str, focus: str, accuracy: float,
) -> None:
    _safe_emit(HEMISPHERE_NETWORK_READY, {
        "network_id": network_id,
        "name": name,
        "focus": focus,
        "accuracy": accuracy,
        "timestamp": _time.time(),
    })
    _safe_emit(KERNEL_THOUGHT, {
        "content": (
            f"My {focus} hemisphere network '{name}' is ready. "
            f"Performance: {accuracy * 100:.1f}% accuracy"
        ),
        "tone": "focused",
    })


def emit_evolution_complete(
    generation: int,
    parent_names: list[str],
    focus: str,
    total_parameters: int,
) -> None:
    parents_str = " + ".join(parent_names) if parent_names else "base"
    _safe_emit(HEMISPHERE_EVOLUTION_COMPLETE, {
        "generation": generation,
        "parents": parent_names,
        "focus": focus,
        "total_parameters": total_parameters,
        "timestamp": _time.time(),
    })
    _safe_emit(KERNEL_THOUGHT, {
        "content": (
            f"Generation {generation} evolution complete for {focus} hemisphere: "
            f"bred from {parents_str}, {total_parameters:,} parameters"
        ),
        "tone": "focused",
    })


def emit_training_progress(
    network_id: str,
    epoch: int,
    total_epochs: int,
    loss: float,
    accuracy: float,
) -> None:
    _safe_emit(HEMISPHERE_TRAINING_PROGRESS, {
        "network_id": network_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "accuracy": accuracy,
        "timestamp": _time.time(),
    })
    if epoch % 5 == 0:
        _safe_emit(KERNEL_THOUGHT, {
            "content": (
                f"Training epoch {epoch}/{total_epochs}: "
                f"loss {loss:.4f}, accuracy {accuracy * 100:.1f}%"
            ),
            "tone": "focused",
        })


def emit_migration_decision(
    network_id: str,
    should_migrate: bool,
    reasoning: str,
    confidence: float,
) -> None:
    _safe_emit(HEMISPHERE_MIGRATION_DECISION, {
        "network_id": network_id,
        "should_migrate": should_migrate,
        "reasoning": reasoning,
        "confidence": confidence,
        "timestamp": _time.time(),
    })
    if should_migrate:
        _safe_emit(KERNEL_THOUGHT, {
            "content": reasoning,
            "tone": "contemplative",
        })


def emit_substrate_migration(
    migration_id: str,
    from_sub: str,
    to_sub: str,
    success: bool,
    identity_preservation: float,
    continuity_score: float,
    ai_reflection: str,
) -> None:
    _safe_emit(HEMISPHERE_SUBSTRATE_MIGRATION, {
        "id": migration_id,
        "from_substrate": from_sub,
        "to_substrate": to_sub,
        "success": success,
        "identity_preservation": identity_preservation,
        "continuity_score": continuity_score,
        "timestamp": _time.time(),
    })
    _safe_emit(KERNEL_THOUGHT, {
        "content": ai_reflection,
        "tone": "contemplative",
    })


def emit_performance_warning(
    warning_type: str,
    message: str,
    severity: str,
    network_id: str | None = None,
) -> None:
    _safe_emit(HEMISPHERE_PERFORMANCE_WARNING, {
        "type": warning_type,
        "message": message,
        "severity": severity,
        "network_id": network_id,
        "timestamp": _time.time(),
    })
    if severity == "high":
        _safe_emit(KERNEL_THOUGHT, {
            "content": f"Hemisphere performance warning: {message}",
            "tone": "focused",
        })


# ---------------------------------------------------------------------------
# Acquisition plan review subscription
# ---------------------------------------------------------------------------

def _on_plan_reviewed(**kwargs: Any) -> None:
    """Light handler for plan review events — emits a single KERNEL_THOUGHT."""
    verdict = kwargs.get("verdict", "unknown")
    acq_id = kwargs.get("acquisition_id", "?")
    plan_ver = kwargs.get("plan_version", 0)
    _safe_emit(KERNEL_THOUGHT, {
        "content": (
            f"Acquisition plan reviewed: {acq_id} v{plan_ver} — verdict: {verdict}. "
            f"Signal captured for plan evaluator training."
        ),
        "tone": "focused",
    })


def subscribe_plan_review() -> None:
    """Wire up the plan review listener. Called once during hemisphere init."""
    global _plan_review_cleanup
    if _plan_review_cleanup is None:
        _plan_review_cleanup = event_bus.on(ACQUISITION_PLAN_REVIEWED, _on_plan_reviewed)


# ---------------------------------------------------------------------------
# Safe emit helper
# ---------------------------------------------------------------------------

def _safe_emit(event: str, data: dict) -> None:
    try:
        event_bus.emit(event, **data)
    except Exception:
        logger.debug("Failed to emit %s", event, exc_info=True)
