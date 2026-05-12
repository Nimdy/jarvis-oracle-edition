"""Diagnostic feature encoding for the DIAGNOSTIC hemisphere specialist.

Encodes a detector snapshot + opportunities + system context into a fixed
43-dim [0,1] feature vector.  Also provides label encoding for the 6-class
detector-category teacher signal.

This is a *detector-pattern approximator* — the teacher label is "which
detector fired," not "what caused the problem."  Enriched with codebase
structural features (Track 4) and live friction/correction signals (Track 5)
to improve localization and root-cause approximation.

Dimension blocks:
  dims  0-5:  Health snapshot
  dims  6-11: Performance snapshot
  dims 12-19: Contextual state
  dims 20-27: History signals
  dims 28-31: Detector firing pattern
  dims 32-37: Codebase structural features (Track 4)
  dims 38-42: Friction/correction enrichment (Track 5)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 43

_DETECTOR_CATEGORIES = [
    "health_degraded",
    "reasoning_decline",
    "confidence_volatile",
    "slow_responses",
    "event_bus_errors",
    "tick_performance",
]
_DETECTOR_INDEX = {cat: i for i, cat in enumerate(_DETECTOR_CATEGORIES)}

_MODE_ORDINALS = {
    "gestation": 0.0,
    "passive": 0.125,
    "conversational": 0.25,
    "reflective": 0.375,
    "focused": 0.5,
    "sleep": 0.625,
    "dreaming": 0.75,
    "deep_learning": 1.0,
}


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class DiagnosticEncoder:
    """Encodes system health state into a 43-dim [0,1] feature vector.

    Block layout:
      dims  0-5:  Health snapshot
      dims  6-11: Performance snapshot
      dims 12-19: Contextual state
      dims 20-27: History signals
      dims 28-31: Detector firing pattern
      dims 32-37: Codebase structural features (Track 4)
      dims 38-42: Friction/correction enrichment (Track 5)
    """

    @staticmethod
    def encode(
        detector_snapshot: dict[str, Any],
        opportunities: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[float]:
        """Produce 43-dim [0,1] feature vector from detector state."""
        vec = [0.0] * FEATURE_DIM

        health = detector_snapshot.get("health", {})
        reasoning = detector_snapshot.get("reasoning", {})
        confidence = detector_snapshot.get("confidence", {})
        latency = detector_snapshot.get("latency", {})
        event_bus = detector_snapshot.get("event_bus", {})
        tick = detector_snapshot.get("tick", {})

        # Block 1: Health snapshot (dims 0-5)
        vec[0] = _clamp(health.get("overall", 1.0))
        vec[1] = _clamp(health.get("worst_score", 1.0) if isinstance(health.get("worst_score"), (int, float)) else 1.0)
        vec[2] = _clamp(reasoning.get("overall", 1.0))
        vec[3] = _clamp(reasoning.get("coherence", 1.0))
        vec[4] = _clamp(confidence.get("volatility", 0.0))
        vec[5] = _clamp(confidence.get("current", 0.5))

        # Block 2: Performance snapshot (dims 6-11)
        trend = confidence.get("trend", 0.0)
        vec[6] = _clamp((trend + 1.0) / 2.0)

        lat_total = latency.get("total", 0)
        lat_slow = latency.get("slow_gt_5s", 0)
        vec[7] = _clamp(lat_slow / max(lat_total, 1))
        avg_slow = latency.get("avg_slow_ms", 0.0) if lat_slow > 0 else 0.0
        vec[8] = _clamp(avg_slow / 30000.0)

        eb_emitted = event_bus.get("emitted", 0)
        eb_errors = event_bus.get("errors", 0)
        vec[9] = _clamp(eb_errors / max(eb_emitted, 1))

        vec[10] = _clamp(tick.get("p95_ms", 0.0) / 200.0)

        uptime = context.get("uptime_s", 0.0)
        vec[11] = _clamp(uptime / 86400.0)

        # Block 3: Contextual state (dims 12-19)
        vec[12] = _clamp(context.get("quarantine_pressure", 0.0))
        vec[13] = _clamp(context.get("soul_integrity", 1.0))
        vec[14] = _clamp(_MODE_ORDINALS.get(context.get("mode", "passive"), 0.125))
        vec[15] = _clamp(context.get("evolution_stage", 0) / 5.0)
        vec[16] = _clamp(context.get("consciousness_stage", 0) / 4.0)
        vec[17] = _clamp((context.get("health_trend_slope", 0.0) + 1.0) / 2.0)
        vec[18] = _clamp(context.get("mutations_last_hour", 0) / 12.0)
        vec[19] = _clamp(context.get("active_learning_jobs", 0) / 5.0)

        # Block 4: History signals (dims 20-27)
        vec[20] = _clamp(len(opportunities) / 10.0)
        max_sustained = 0
        for o in opportunities:
            s = o.get("sustained_count", 0)
            if s > max_sustained:
                max_sustained = s
        vec[21] = _clamp(max_sustained / 5.0)
        vec[22] = _clamp(context.get("improvements_today", 0) / 6.0)
        vec[23] = _clamp(context.get("last_improvement_age_s", 86400.0) / 86400.0)
        vec[24] = _clamp(context.get("sandbox_pass_rate", 0.0))
        vec[25] = _clamp(context.get("friction_rate", 0.0))
        vec[26] = _clamp(context.get("correction_count", 0) / 10.0)
        vec[27] = _clamp(context.get("autonomy_level", 0) / 3.0)

        # Block 5: Detector firing pattern (dims 28-31)
        n_firing = len(opportunities)
        vec[28] = _clamp(n_firing / 6.0)

        max_pri = 0
        has_health = False
        has_perf = False
        for o in opportunities:
            pri = o.get("priority", 0)
            if pri > max_pri:
                max_pri = pri
            otype = o.get("type", "")
            if otype == "health_degraded":
                has_health = True
            if otype in ("tick_performance", "slow_responses"):
                has_perf = True

        vec[29] = _clamp(max_pri / 5.0)
        vec[30] = 1.0 if has_health else 0.0
        vec[31] = 1.0 if has_perf else 0.0

        # Block 6: Codebase structural features (dims 32-37) — Track 4
        vec[32] = _clamp(context.get("target_module_lines", 0) / 500.0)
        vec[33] = _clamp(context.get("target_import_fanout", 0) / 15.0)
        vec[34] = _clamp(context.get("target_importers", 0) / 15.0)
        vec[35] = _clamp(context.get("target_symbol_count", 0) / 50.0)
        vec[36] = 1.0 if context.get("target_recently_modified", False) else 0.0
        vec[37] = 1.0 if context.get("has_codebase_context", False) else 0.0

        # Block 7: Friction/correction enrichment (dims 38-42) — Track 5
        vec[38] = _clamp(context.get("friction_severity_high_ratio", 0.0))
        vec[39] = _clamp(context.get("friction_correction_ratio", 0.0))
        vec[40] = _clamp(context.get("friction_identity_count", 0) / 5.0)
        vec[41] = _clamp(context.get("correction_auto_accepted", 0) / 5.0)
        vec[42] = 1.0 if context.get("has_friction_context", False) else 0.0

        return vec

    @staticmethod
    def encode_no_opportunity_label() -> tuple[list[float], dict[str, Any]]:
        """Encode a negative-example label for scans where no detector fired.

        Returns a uniform distribution across all 6 classes — the correct
        KL-div target for "no specific detector is expected to fire."  Lower
        fidelity (set by caller) ensures these don't dominate the training set.
        """
        n = len(_DETECTOR_CATEGORIES)
        label = [1.0 / n] * n
        metadata = {
            "detector_type": "no_opportunity",
            "sustained_count": 0,
            "fingerprint": "",
            "top_metric": "",
            "module_hint": "",
        }
        return label, metadata

    @staticmethod
    def encode_label(
        opportunity: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        """Encode a 6-class teacher label + rich metadata from a fired detector.

        Returns (label_vector, metadata_dict).  The metadata carries extra
        fields for later label enrichment without changing output dim.
        """
        label = [0.0] * len(_DETECTOR_CATEGORIES)
        det_type = opportunity.get("type", "")
        idx = _DETECTOR_INDEX.get(det_type)
        if idx is not None:
            label[idx] = 1.0

        evidence = opportunity.get("evidence_detail", {})
        top_metric = ""
        if det_type == "health_degraded":
            top_metric = evidence.get("worst_component", "")
        elif det_type == "reasoning_decline":
            depth = evidence.get("depth", 1.0)
            coherence = evidence.get("coherence", 1.0)
            top_metric = "depth" if depth < coherence else "coherence"
        elif det_type == "confidence_volatile":
            top_metric = "volatility"
        elif det_type == "slow_responses":
            top_metric = f"avg_slow_ms={evidence.get('avg_slow_ms', 0):.0f}"
        elif det_type == "event_bus_errors":
            top_metric = f"error_rate={evidence.get('error_rate', 0):.4f}"
        elif det_type == "tick_performance":
            top_metric = f"p95_ms={evidence.get('p95_ms', 0):.1f}"

        metadata = {
            "detector_type": det_type,
            "sustained_count": opportunity.get("sustained_count", 0),
            "fingerprint": opportunity.get("fingerprint", ""),
            "top_metric": top_metric,
            "module_hint": opportunity.get("target_module", ""),
        }

        return label, metadata
