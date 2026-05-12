"""Pure scoring functions for Phase D language eval gates.

These score bounded articulation quality across 7 dimensions.
All functions are stateless — they take corpus/telemetry stats dicts
and return a float in [0.0, 1.0]. Higher is always better.

Dimensions:
  1. sample_count       — enough data to evaluate?
  2. provenance_fidelity — bounded answers sourced from verified records
  3. exactness           — deterministic classes produce exact answers
  4. hallucination_rate  — inverse: how often did grounding miss?
  5. fail_closed_correctness — when data missing, did we fail closed?
  6. native_usage_rate   — bounded path used when eligible?
  7. style_quality       — output within length/sentence caps?
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Thresholds for gate readiness
MIN_SAMPLES_FOR_EVAL = 30
PROVENANCE_GREEN = 0.90
PROVENANCE_YELLOW = 0.70
EXACTNESS_GREEN = 0.85
EXACTNESS_YELLOW = 0.60
HALLUCINATION_CEILING = 0.10   # above this = red
HALLUCINATION_WARN = 0.05      # above this = yellow
FAIL_CLOSED_GREEN = 0.90
FAIL_CLOSED_YELLOW = 0.70
NATIVE_USAGE_GREEN = 0.70
NATIVE_USAGE_YELLOW = 0.35
STYLE_GREEN = 0.90
STYLE_YELLOW = 0.70

# Transitional compatibility: older deterministic/native routes emitted
# verdict strings that end with "_native" instead of using a grounded prefix.
# Keep these explicitly grounded so historical corpus evidence scores correctly.
_LEGACY_GROUNDED_VERDICTS: frozenset[str] = frozenset({
    "introspection_capability_status_native",
    "none_route_capability_status_native",
    "none_route_memory_recall_native",
})


def score_sample_count(
    corpus_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score whether we have enough data to evaluate a response class.

    Returns 1.0 when sample count >= MIN_SAMPLES_FOR_EVAL, linear ramp below.
    If response_class is specified, uses per-class count; otherwise total.
    """
    if response_class:
        counts = corpus_stats.get("counts_by_response_class", {})
        n = counts.get(response_class, 0)
    else:
        n = corpus_stats.get("total_examples", 0)

    if n >= MIN_SAMPLES_FOR_EVAL:
        return 1.0
    return max(0.0, n / MIN_SAMPLES_FOR_EVAL)


def score_provenance_fidelity(
    corpus_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score what fraction of non-negative examples have grounded provenance.

    Negative examples (intentional bad training data) are excluded — they
    have provenance "negative:*" by design. The score measures whether
    real responses are properly sourced from verified records.
    """
    recent = corpus_stats.get("recent_examples", [])
    if response_class:
        recent = [e for e in recent if e.get("response_class") == response_class]

    # Exclude negative examples — they are intentional, not provenance failures
    recent = [e for e in recent if e.get("response_class") != "negative_example"]

    if not recent:
        # Fall back to corpus-wide counts if recent window is empty after filtering
        by_prov = corpus_stats.get("counts_by_provenance", {})
        total = corpus_stats.get("total_examples", 0)
        neg_count = sum(v for k, v in by_prov.items() if k.startswith("negative:"))
        non_neg_total = total - neg_count
        if non_neg_total <= 0:
            return 0.0
        grounded = sum(v for k, v in by_prov.items() if _is_grounded_verdict(k))
        return grounded / non_neg_total

    valid = sum(1 for e in recent if _is_grounded_verdict(e.get("provenance_verdict", "")))
    return valid / len(recent)


def _is_grounded_verdict(verdict: str) -> bool:
    """Check if a provenance verdict indicates grounded/verified sourcing."""
    if not verdict or verdict == "unknown":
        return False
    if verdict.startswith("negative:"):
        return False
    if verdict in _LEGACY_GROUNDED_VERDICTS:
        return True
    # Known grounded prefixes from the live system
    _GROUNDED_PREFIXES = (
        "bounded_", "grounded_", "native_", "deterministic_",
        "strict_", "registry_", "reflective_",
    )
    return any(verdict.startswith(p) for p in _GROUNDED_PREFIXES)


def score_exactness(
    corpus_stats: dict[str, Any],
    telemetry_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score how often deterministic response classes produce exact answers.

    Uses the native_used_by_class from telemetry — when native is used,
    the answer came from the bounded articulator (exact). When not, it
    fell back to LLM (inexact).
    """
    if response_class:
        native_by_class = telemetry_stats.get("native_used_by_class", {})
        total_by_class = telemetry_stats.get("counts_by_response_class", {})
        native = native_by_class.get(response_class, 0)
        total = total_by_class.get(response_class, 0)
    else:
        native = sum(telemetry_stats.get("native_used_by_class", {}).values())
        total = telemetry_stats.get("total_events", 0)

    if total == 0:
        return 0.0
    return native / total


def score_hallucination_rate(
    corpus_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score inverse hallucination rate (higher = fewer hallucinations).

    Negative examples (response_class="negative_example") are EXCLUDED from
    both numerator and denominator — they are intentionally captured bad
    training data, not system hallucinations. The score measures what fraction
    of real (non-negative) examples have ungrounded provenance.
    """
    by_provenance = corpus_stats.get("counts_by_provenance", {})
    by_class = corpus_stats.get("counts_by_response_class", {})

    if response_class:
        recent = corpus_stats.get("recent_examples", [])
        class_recent = [
            e for e in recent
            if e.get("response_class") == response_class
            and e.get("response_class") != "negative_example"
        ]
        if not class_recent:
            return 1.0
        ungrounded = sum(
            1 for e in class_recent
            if not _is_grounded_verdict(e.get("provenance_verdict", ""))
        )
        return max(0.0, 1.0 - (ungrounded / len(class_recent)))
    else:
        total = corpus_stats.get("total_examples", 0)
        neg_class_count = by_class.get("negative_example", 0)
        non_neg_total = total - neg_class_count
        if non_neg_total <= 0:
            return 1.0
        # Count non-negative examples with ungrounded provenance
        grounded_count = sum(v for k, v in by_provenance.items() if _is_grounded_verdict(k))
        return max(0.0, grounded_count / non_neg_total)


def score_fail_closed_correctness(
    telemetry_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score whether the system fails closed correctly when data is missing.

    fail_closed events are good — they mean the system declined to hallucinate.
    We want: when data is missing, fail_closed fires. When data is present,
    native_used fires. The score is the complement of the "fell through" rate
    (events that are neither native_used nor fail_closed).
    """
    if response_class:
        native_by_class = telemetry_stats.get("native_used_by_class", {})
        fc_by_class = telemetry_stats.get("fail_closed_by_class", {})
        total_by_class = telemetry_stats.get("counts_by_response_class", {})
        native = native_by_class.get(response_class, 0)
        fc = fc_by_class.get(response_class, 0)
        total = total_by_class.get(response_class, 0)
    else:
        native = sum(telemetry_stats.get("native_used_by_class", {}).values())
        fc = sum(telemetry_stats.get("fail_closed_by_class", {}).values())
        total = telemetry_stats.get("total_events", 0)

    if total == 0:
        return 0.0

    # Events handled correctly = native_used + fail_closed
    handled = native + fc
    return min(1.0, handled / total)


def score_native_usage_rate(
    telemetry_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score native/bounded path usage rate.

    This is the fraction of eligible events where the bounded articulator
    was used instead of falling back to LLM.
    """
    if response_class:
        native_by_class = telemetry_stats.get("native_used_by_class", {})
        total_by_class = telemetry_stats.get("counts_by_response_class", {})
        native = native_by_class.get(response_class, 0)
        total = total_by_class.get(response_class, 0)
    else:
        rate = telemetry_stats.get("native_usage_rate", 0.0)
        if rate > 0:
            return rate
        native = sum(telemetry_stats.get("native_used_by_class", {}).values())
        total = telemetry_stats.get("total_events", 0)

    if total == 0:
        return 0.0
    return native / total


def score_style_quality(
    corpus_stats: dict[str, Any],
    response_class: str = "",
) -> float:
    """Score whether bounded outputs stay within style constraints.

    Negative examples are excluded — they have lead=False and confidence=0.0
    by design (they are intentional bad training data, not style failures).

    For non-negative examples, checks:
    - lead present and non-empty
    - confidence > 0 (frame was structurally healthy)
    - no parse_warnings in safety_flags
    """
    recent = corpus_stats.get("recent_examples", [])
    if response_class:
        recent = [e for e in recent if e.get("response_class") == response_class]

    # Exclude negative examples
    recent = [e for e in recent if e.get("response_class") != "negative_example"]

    if not recent:
        return 0.0

    good = 0
    for ex in recent:
        lead = ex.get("lead", "")
        confidence = ex.get("confidence", 0.0)
        flags = ex.get("safety_flags", [])
        has_warnings = any("parse_warning" in f or "warning" in f for f in flags)

        if lead and confidence > 0 and not has_warnings:
            good += 1

    return good / len(recent)


# ── Composite gate score ──────────────────────────────────────────────

def compute_gate_scores(
    corpus_stats: dict[str, Any],
    telemetry_stats: dict[str, Any],
    response_class: str = "",
) -> dict[str, float]:
    """Compute all 7 gate scores for a response class (or globally).

    Returns a dict mapping dimension name to score in [0.0, 1.0].
    """
    return {
        "sample_count": score_sample_count(corpus_stats, response_class),
        "provenance_fidelity": score_provenance_fidelity(corpus_stats, response_class),
        "exactness": score_exactness(corpus_stats, telemetry_stats, response_class),
        "hallucination_rate": score_hallucination_rate(corpus_stats, response_class),
        "fail_closed_correctness": score_fail_closed_correctness(telemetry_stats, response_class),
        "native_usage_rate": score_native_usage_rate(telemetry_stats, response_class),
        "style_quality": score_style_quality(corpus_stats, response_class),
    }


def classify_gate(scores: dict[str, float]) -> str:
    """Classify overall gate health as green/yellow/red.

    green:  all dimensions above green thresholds, sample_count == 1.0
    yellow: no dimension below red, sample_count >= 0.5
    red:    any critical dimension failed
    """
    reason = classify_gate_reason(scores)
    if reason != "ok":
        return "red"

    sc = scores.get("sample_count", 0.0)
    pf = scores.get("provenance_fidelity", 0.0)
    ex = scores.get("exactness", 0.0)
    hr = scores.get("hallucination_rate", 1.0)
    fc = scores.get("fail_closed_correctness", 0.0)
    nu = scores.get("native_usage_rate", 0.0)
    sq = scores.get("style_quality", 0.0)

    # Green conditions
    if (sc >= 1.0
        and pf >= PROVENANCE_GREEN
        and ex >= EXACTNESS_GREEN
        and hr >= (1.0 - HALLUCINATION_WARN)
        and fc >= FAIL_CLOSED_GREEN
        and nu >= NATIVE_USAGE_GREEN
        and sq >= STYLE_GREEN):
        return "green"

    return "yellow"


def classify_gate_reason(scores: dict[str, float]) -> str:
    """Return the primary red-condition reason for a score set.

    This keeps red-pressure diagnostics explainable and allows consumers to
    separate quality-risk reds from evidence-limited reds without changing the
    existing gate color semantics.
    """
    sc = scores.get("sample_count", 0.0)
    if sc < 0.5:
        return "insufficient_samples"

    hr = scores.get("hallucination_rate", 1.0)
    if hr < (1.0 - HALLUCINATION_CEILING):  # hallucination_rate is inverted
        return "hallucination_ceiling"

    pf = scores.get("provenance_fidelity", 0.0)
    if pf < PROVENANCE_YELLOW:
        return "provenance_low"

    return "ok"


# ── Response classes eligible for Phase D eval gates ──────────────────

BOUNDED_RESPONSE_CLASSES = (
    "self_status",
    "self_introspection",
    "recent_learning",
    "recent_research",
    "memory_recall",
    "identity_answer",
    "capability_status",
)
