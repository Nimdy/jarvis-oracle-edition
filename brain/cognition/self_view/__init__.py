"""Operational Self-View (OSV) — P0 substrate.

A deterministic, read-only, no-LLM self-model fused from existing subsystem readouts,
with provenance on every fact and gaps as first-class entries. See
``docs/SELF_VIEW_DESIGN.md``. P0 grants no behavior authority.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from cognition.self_view.provenance import (
    ALL_PROVENANCE,
    MEASUREMENT_LEVELS,
    Fact,
    Provenance,
    gap,
    unknown,
)
from cognition.self_view.synthesizer import SCHEMA_VERSION, SelfModel, SelfViewSynthesizer
from cognition.self_view.gather import gather_live_sources

logger = logging.getLogger("jarvis.self_view")

STATE_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "self_view.json")

__all__ = [
    "Provenance", "Fact", "gap", "unknown", "ALL_PROVENANCE", "MEASUREMENT_LEVELS",
    "SelfModel", "SelfViewSynthesizer", "SCHEMA_VERSION", "gather_live_sources",
    "build_self_view", "save_self_view", "load_self_view",
]


def build_self_view(engine: Any = None, eval_snapshot: dict[str, Any] | None = None,
                    skills_summary: dict[str, Any] | None = None,
                    snapshot: dict[str, Any] | None = None,
                    now: float | None = None) -> dict[str, Any]:
    """Gather live sources, synthesize the model, and return its dict form (read-only).

    ``snapshot`` is the dashboard build_cache output (the 80+ subsystem aggregator) — the
    OSV's broad source of truth. Read-only; never mutated.
    """
    import time as _t
    ts = now if now is not None else _t.time()
    sources = gather_live_sources(engine, eval_snapshot, skills_summary, snapshot)
    model = SelfViewSynthesizer().synthesize(sources, ts)
    return model.to_dict()


def save_self_view(model_dict: dict[str, Any]) -> None:
    try:
        path = Path(STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(model_dict, indent=2))
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to persist self-view snapshot", exc_info=True)


def load_self_view() -> dict[str, Any] | None:
    try:
        path = Path(STATE_PATH)
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        logger.debug("Failed to load self-view snapshot", exc_info=True)
        return None
