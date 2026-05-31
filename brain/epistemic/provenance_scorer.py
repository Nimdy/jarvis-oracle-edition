"""Provenance Scorer — VIEW-ONLY grounding-tension reader (SPARK_DESIGN §2.1, §3 component 1).

Station 1 of the Grounding Ring. Reads — and ONLY reads — the live belief graph
to compute a ``grounding_tension`` scalar per active belief and a single
aggregate. Tension is high when a belief is model-inferred, structurally
unsupported (no incoming evidence edges / orphaned in the graph), and the
quarantine system is under pressure.

``grounding_tension`` is the origin signal of the outward loop: the grounding
drive fires on it (``DriveSignals``), formulating a question whose answer must
come from an *external* validator (operator, Pi senses, source-cited research).

HONESTY GUARDRAILS (SPARK_DESIGN §7) — enforced structurally here:
  * VIEW-ONLY EPISTEMICS. This module NEVER mutates the frozen ``BeliefRecord``,
    NEVER calls any ``belief_store`` mutator, and NEVER writes ``beliefs.jsonl``.
    It calls only read accessors (``get_active_beliefs``, ``get_incoming``,
    ``propagate_all`` — itself a pure function) and ``compute_integrity``.
  * The tension scalar carries a provenance dict (signal → source_field →
    raw_value) so the number is auditable, never "felt".
  * Default-safe: any missing source yields tension 0.0 and
    ``sources_available=False``; never raises into the caller.

This is the *reader*. It drives NO lever on its own — the grounding drive that
consumes it ships shadow-first behind its own promotion controller (P2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Provenance vocabulary. Mirrors spark_metrics._GROUNDED / _INFERRED and the
# ProvenanceType vocabulary in consciousness/events. "grounded" = externally or
# observationally anchored; "inferred" = model produced it on its own.
_GROUNDED_PROVENANCE = frozenset({"observed", "user_claim", "external_source"})
_INFERRED_PROVENANCE = frozenset({"model_inference"})

# Edge types that count as *incoming evidential support* for a belief. ``refines``
# is excluded (matches propagation.py — refines edges do not contribute to
# effective confidence). A belief with no such incoming edge is "support-orphan".
_SUPPORT_INCOMING_TYPES = frozenset({
    "supports", "contradicts", "depends_on", "derived_from",
})

# Weighted blend of the three tension components (SPARK §2.1: weighted max of
# {quarantine pressure, orphan-ness, inference gap}). We take a weighted blend
# rather than a raw max so a belief that is inferred AND orphaned AND under
# pressure reads hotter than any single component alone, while still being
# dominated by its strongest signal.
_W_INFERENCE = 0.45   # inferred + low effective confidence
_W_ORPHAN = 0.35      # no incoming evidential edges
_W_PRESSURE = 0.20    # global quarantine pressure (composite)

# Floor below which a belief's tension is treated as noise (not surfaced).
_TENSION_NOISE_FLOOR = 0.05

# Sample cap so a large store can't add latency on the read path. Mirrors the
# spark_metrics _BELIEF_SAMPLE_CAP discipline.
_BELIEF_SAMPLE_CAP = 1000


# ---------------------------------------------------------------------------
# Facet tagging (SPARK §6 channel-selection rule).
#
# Each belief gets a facet so the router can pick the cheapest channel that CAN
# validate it: identity/user → operator; scene/physical → Pi senses;
# factual/external → web; self → introspection/operator. Derived deterministically
# from claim_type + provenance + canonical_subject (SPARK §11 open-question 5
# names this a candidate for a classifier later; the deterministic mapping is the
# day-one baseline).
# ---------------------------------------------------------------------------

Facet = str  # "identity" | "scene" | "factual" | "self"

_FACET_IDENTITY = "identity"
_FACET_SCENE = "scene"
_FACET_FACTUAL = "factual"
_FACET_SELF = "self"

_SCENE_SUBJECT_HINTS = frozenset({
    "scene", "room", "camera", "presence", "speaker", "person",
    "object", "location", "environment", "face", "voice",
})
_SELF_SUBJECT_HINTS = frozenset({
    "i", "me", "myself", "self", "jarvis", "my", "assistant",
})


def classify_facet(belief: Any) -> Facet:
    """Deterministic facet tag for channel routing. Pure read, no mutation."""
    claim_type = getattr(belief, "claim_type", "") or ""
    provenance = getattr(belief, "provenance", "") or ""
    subject = (getattr(belief, "canonical_subject", "") or "").lower()
    subject_id = getattr(belief, "identity_subject_id", "") or ""

    # Identity / self beliefs route to the operator (highest trust, lowest BW).
    if claim_type == "identity" or subject_id:
        # A belief about the assistant itself is "self" (introspection/operator);
        # a belief about a person/user is "identity" (operator).
        if subject in _SELF_SUBJECT_HINTS:
            return _FACET_SELF
        return _FACET_IDENTITY

    if subject in _SELF_SUBJECT_HINTS:
        return _FACET_SELF

    # Observation / world-model provenance about the physical world → Pi senses.
    if provenance in ("observed", "world_model", "prediction"):
        return _FACET_SCENE
    if any(h in subject for h in _SCENE_SUBJECT_HINTS):
        return _FACET_SCENE

    # Preference / policy beliefs are about the user → operator.
    if claim_type in ("preference", "policy"):
        return _FACET_IDENTITY

    # Everything else (factual / philosophical) → web / research.
    return _FACET_FACTUAL


# Cheapest external channel that can validate each facet (SPARK §6).
_FACET_CHANNEL: dict[str, str] = {
    _FACET_IDENTITY: "operator",
    _FACET_SELF: "operator",
    _FACET_SCENE: "pi_senses",
    _FACET_FACTUAL: "web",
}

# Tool hint handed to the drive / opportunity layer per facet.
_FACET_TOOL_HINT: dict[str, str] = {
    _FACET_IDENTITY: "introspection",
    _FACET_SELF: "introspection",
    _FACET_SCENE: "memory",
    _FACET_FACTUAL: "web",
}


def facet_channel(facet: Facet) -> str:
    """Cheapest external validator channel for a facet (default 'web')."""
    return _FACET_CHANNEL.get(facet, "web")


def facet_tool_hint(facet: Facet) -> str:
    """Drive/opportunity tool hint for a facet (default 'web')."""
    return _FACET_TOOL_HINT.get(facet, "web")


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeliefTension:
    """View-only grounding-tension reading for a single belief.

    ``provenance_detail`` records signal → raw_value so the scalar is auditable
    (SPARK §7 — the number must be explainable, never asserted).
    """

    belief_id: str
    grounding_tension: float
    facet: Facet
    channel: str
    is_orphan: bool
    is_inferred: bool
    base_confidence: float
    effective_confidence: float
    subject: str = ""
    rendered_claim: str = ""
    provenance: str = ""
    provenance_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "grounding_tension": round(self.grounding_tension, 4),
            "facet": self.facet,
            "channel": self.channel,
            "is_orphan": self.is_orphan,
            "is_inferred": self.is_inferred,
            "base_confidence": round(self.base_confidence, 4),
            "effective_confidence": round(self.effective_confidence, 4),
            "subject": self.subject,
            "rendered_claim": self.rendered_claim[:160],
            "provenance": self.provenance,
            "provenance_detail": {k: round(v, 4) for k, v in self.provenance_detail.items()},
        }


@dataclass
class GroundingTensionReport:
    """Aggregate grounding-tension snapshot across all sampled beliefs."""

    aggregate_tension: float = 0.0
    max_tension: float = 0.0
    mean_tension: float = 0.0
    high_tension_count: int = 0
    sampled_beliefs: int = 0
    orphan_count: int = 0
    inferred_count: int = 0
    quarantine_pressure: float = 0.0
    orphan_rate: float = 0.0
    sources_available: bool = False
    # Top-N tensions by grounding_tension (hottest first), for the drive/queue.
    top_tensions: list[BeliefTension] = field(default_factory=list)
    # Per-facet aggregate (so the router can see which channel is loudest).
    facet_tension: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_tension": round(self.aggregate_tension, 4),
            "max_tension": round(self.max_tension, 4),
            "mean_tension": round(self.mean_tension, 4),
            "high_tension_count": self.high_tension_count,
            "sampled_beliefs": self.sampled_beliefs,
            "orphan_count": self.orphan_count,
            "inferred_count": self.inferred_count,
            "quarantine_pressure": round(self.quarantine_pressure, 4),
            "orphan_rate": round(self.orphan_rate, 4),
            "sources_available": self.sources_available,
            "facet_tension": {k: round(v, 4) for k, v in self.facet_tension.items()},
            "top_tensions": [t.to_dict() for t in self.top_tensions],
        }


class ProvenanceScorer:
    """VIEW-ONLY scorer: belief provenance + edge orphans + effective confidence
    → grounding_tension (per belief and aggregate).

    Resolves the live belief/edge stores the same way ``spark_metrics`` does
    (engine.consciousness._belief_graph, falling back to BeliefGraph /
    ContradictionEngine singletons). All access is read-only.
    """

    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine

    # -- store resolution (read-only) --------------------------------------

    def _resolve_stores(self) -> tuple[Any, Any]:
        """Return (belief_store, edge_store) or (None, None). Never raises."""
        belief_store = None
        edge_store = None
        cs = getattr(self._engine, "consciousness", None) if self._engine is not None else None
        try:
            bg = getattr(cs, "_belief_graph", None) if cs else None
            if bg is None:
                from epistemic.belief_graph import BeliefGraph
                bg = BeliefGraph.get_instance()
            if bg is not None:
                belief_store = getattr(bg, "_belief_store", None)
                edge_store = getattr(bg, "_edge_store", None)
        except Exception:
            logger.debug("ProvenanceScorer: belief_graph resolution failed", exc_info=True)

        if belief_store is None:
            try:
                from epistemic.contradiction_engine import ContradictionEngine
                ce = ContradictionEngine.get_instance()
                belief_store = getattr(ce, "_belief_store", None) if ce else None
            except Exception:
                logger.debug("ProvenanceScorer: contradiction-engine fallback failed",
                             exc_info=True)
        return belief_store, edge_store

    @staticmethod
    def _quarantine_pressure() -> float:
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            return float(get_quarantine_pressure().current.composite or 0.0)
        except Exception:
            return 0.0

    # -- per-belief tension -------------------------------------------------

    @staticmethod
    def _belief_tension(
        belief: Any,
        is_orphan: bool,
        base_conf: float,
        effective_conf: float,
        pressure: float,
    ) -> tuple[float, dict[str, float]]:
        """Compute grounding_tension for one belief + its provenance dict.

        Three components (each in [0,1]):
          * inference term — model_inference provenance AND low effective
            confidence (an inferred belief the graph already doubts is the most
            tension-worthy: it points inward with nothing holding it up).
          * orphan term — no incoming evidential edges (structurally unanchored).
          * pressure term — global quarantine composite (system-wide doubt).
        """
        provenance = getattr(belief, "provenance", "") or ""
        is_inferred = provenance in _INFERRED_PROVENANCE

        # Inference term: only inferred beliefs carry it; scaled by how little
        # effective confidence the graph assigns (1 - effective). A grounded
        # belief contributes 0 here (cannot-lie: no inferred-ness ⇒ no term).
        inference_term = (1.0 - effective_conf) if is_inferred else 0.0

        orphan_term = 1.0 if is_orphan else 0.0
        pressure_term = max(0.0, min(1.0, pressure))

        tension = (
            _W_INFERENCE * inference_term
            + _W_ORPHAN * orphan_term
            + _W_PRESSURE * pressure_term
        )
        tension = max(0.0, min(1.0, tension))

        detail = {
            "inference_term": round(inference_term, 4),
            "orphan_term": round(orphan_term, 4),
            "pressure_term": round(pressure_term, 4),
            "base_confidence": round(base_conf, 4),
            "effective_confidence": round(effective_conf, 4),
        }
        return tension, detail

    # -- public API ---------------------------------------------------------

    def compute(self, top_n: int = 10) -> GroundingTensionReport:
        """Compute the grounding-tension report. VIEW-ONLY, never raises.

        Returns a default-safe (all-zero, sources_available=False) report if the
        belief graph is unavailable.
        """
        report = GroundingTensionReport()
        belief_store, edge_store = self._resolve_stores()
        if belief_store is None or not hasattr(belief_store, "get_active_beliefs"):
            return report

        try:
            beliefs = belief_store.get_active_beliefs()
        except Exception:
            logger.debug("ProvenanceScorer: get_active_beliefs failed", exc_info=True)
            return report
        if not beliefs:
            report.sources_available = True
            return report

        if len(beliefs) > _BELIEF_SAMPLE_CAP:
            beliefs = beliefs[-_BELIEF_SAMPLE_CAP:]

        pressure = self._quarantine_pressure()
        report.quarantine_pressure = pressure

        # Effective-confidence views (pure function, no mutation). Best-effort:
        # if propagation is unavailable, fall back to base confidence.
        views: dict[str, Any] = {}
        if edge_store is not None:
            try:
                from epistemic.belief_graph.propagation import propagate_all
                views = propagate_all(edge_store, belief_store)
            except Exception:
                logger.debug("ProvenanceScorer: propagate_all failed", exc_info=True)
                views = {}

        # orphan_rate (graph integrity) — observability echo of the live 0.857.
        if edge_store is not None:
            try:
                from epistemic.belief_graph.integrity import compute_integrity
                integrity = compute_integrity(edge_store, belief_store)
                report.orphan_rate = float(integrity.get("orphan_rate", 0.0) or 0.0)
            except Exception:
                logger.debug("ProvenanceScorer: compute_integrity failed", exc_info=True)

        tensions: list[BeliefTension] = []
        sum_tension = 0.0
        facet_sum: dict[str, float] = {}
        facet_n: dict[str, int] = {}

        for belief in beliefs:
            bid = getattr(belief, "belief_id", None)
            if not bid:
                continue

            # Orphan = no incoming evidential edge (refines excluded). Read-only.
            is_orphan = True
            if edge_store is not None:
                try:
                    incoming = edge_store.get_incoming(bid)
                    is_orphan = not any(
                        e.edge_type in _SUPPORT_INCOMING_TYPES for e in incoming
                    )
                except Exception:
                    is_orphan = True

            base_conf = float(getattr(belief, "belief_confidence", 0.0) or 0.0)
            view = views.get(bid)
            effective_conf = (
                float(getattr(view, "effective_confidence", base_conf))
                if view is not None else base_conf
            )

            provenance = getattr(belief, "provenance", "") or ""
            is_inferred = provenance in _INFERRED_PROVENANCE

            tension, detail = self._belief_tension(
                belief, is_orphan, base_conf, effective_conf, pressure,
            )

            if is_orphan:
                report.orphan_count += 1
            if is_inferred:
                report.inferred_count += 1

            sum_tension += tension
            if tension > report.max_tension:
                report.max_tension = tension
            if tension >= _high_tension_threshold():
                report.high_tension_count += 1

            facet = classify_facet(belief)
            facet_sum[facet] = facet_sum.get(facet, 0.0) + tension
            facet_n[facet] = facet_n.get(facet, 0) + 1

            if tension >= _TENSION_NOISE_FLOOR:
                tensions.append(BeliefTension(
                    belief_id=bid,
                    grounding_tension=tension,
                    facet=facet,
                    channel=facet_channel(facet),
                    is_orphan=is_orphan,
                    is_inferred=is_inferred,
                    base_confidence=base_conf,
                    effective_confidence=effective_conf,
                    subject=getattr(belief, "canonical_subject", "") or "",
                    rendered_claim=getattr(belief, "rendered_claim", "") or "",
                    provenance=provenance,
                    provenance_detail=detail,
                ))

        n = len(beliefs)
        report.sampled_beliefs = n
        report.mean_tension = sum_tension / n if n else 0.0
        # Aggregate = weighted blend of the loudest belief (max) and the field
        # mean, so a single hot belief raises the drive but a broadly-grounded
        # store stays quiet (homeostasis, SPARK §2.8).
        report.aggregate_tension = max(
            0.0, min(1.0, 0.6 * report.max_tension + 0.4 * report.mean_tension),
        )
        report.facet_tension = {
            f: (facet_sum[f] / facet_n[f]) for f in facet_sum if facet_n.get(f)
        }
        tensions.sort(key=lambda t: t.grounding_tension, reverse=True)
        report.top_tensions = tensions[: max(0, top_n)]
        report.sources_available = True
        return report

    def get_status(self, top_n: int = 5) -> dict[str, Any]:
        """Dashboard snapshot. Read-only."""
        return self.compute(top_n=top_n).to_dict()


def _high_tension_threshold() -> float:
    """Threshold above which a belief is "high tension". Module-level so tests /
    later phases can reference it; kept conservative so noise doesn't inflate it."""
    return 0.40
