"""Counterfactual Evaluation Engine (Phase 10 / #17).

Read-only, no-LLM, data-gated. Evaluates the alternatives JARVIS did NOT take
after autonomy decisions and surfaces "missed opportunities" during dream/sleep
via the Layer-9 reflective audit. See engine.py for the discipline notes.
"""
from epistemic.counterfactual.engine import (
    CounterfactualEngine,
    CounterfactualFinding,
    get_counterfactual_engine,
    MIN_OUTCOMES,
    MIN_BUFFER,
    WARNING_REGRET,
)

__all__ = [
    "CounterfactualEngine",
    "CounterfactualFinding",
    "get_counterfactual_engine",
    "MIN_OUTCOMES",
    "MIN_BUFFER",
    "WARNING_REGRET",
]
