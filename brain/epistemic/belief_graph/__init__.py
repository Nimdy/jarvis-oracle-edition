"""Epistemic Immune System -- Layer 7: Belief Confidence Graph.

Layer 7 does not decide what Jarvis believes; it models why beliefs
reinforce, weaken, or depend on one another.

Layer 5 = belief detection and conflict handling.
Layer 6 = calibration over time.
Layer 7 = relational epistemic structure.

This module wraps Layer 5's BeliefStore with a weighted directed graph
(EdgeStore) where beliefs are nodes and evidence edges carry typed
relationships (supports, contradicts, refines, depends_on, derived_from).

Public API:
    BeliefGraph.get_instance()  -- singleton access
    .on_tick()                  -- called from consciousness tick cycle
    .get_state()                -- dashboard / snapshot data
    .on_new_belief(belief)      -- called when Layer 5 extracts a new belief
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

BELIEF_GRAPH_TICK_INTERVAL_S: float = 60.0
BELIEF_GRAPH_TICK_ACCELERATED_S: float = 30.0

_instance: BeliefGraph | None = None


class BeliefGraph:
    """Orchestrator for the Belief Confidence Graph (Layer 7)."""

    _instance: BeliefGraph | None = None

    PROPAGATION_CADENCE_S: float = 600.0

    def __init__(self) -> None:
        from epistemic.belief_graph.edges import EdgeStore
        from epistemic.belief_graph.bridge import GraphBridge

        self._contradiction_engine: Any = None
        self._belief_store: Any = None
        self._edge_store: EdgeStore | None = None
        self._bridge: GraphBridge | None = None
        self._initialized = False
        self._tick_count: int = 0
        self._last_compaction: float = 0.0
        self._last_propagation_run: float = 0.0
        self._propagation_ran_once: bool = False

    @classmethod
    def get_instance(cls) -> BeliefGraph | None:
        return cls._instance

    @classmethod
    def _set_instance(cls, instance: BeliefGraph) -> None:
        cls._instance = instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine.get_instance()
            if engine is None:
                return False

            self._contradiction_engine = engine
            self._belief_store = engine._belief_store

            from epistemic.belief_graph.edges import EdgeStore
            from epistemic.belief_graph.bridge import GraphBridge

            self._edge_store = EdgeStore()
            self._edge_store.rehydrate()

            self._bridge = GraphBridge(self._edge_store, self._belief_store)
            self._bridge.subscribe()

            self._initialized = True
            BeliefGraph._set_instance(self)
            edge_count = self._edge_store.get_stats()["total_edges"]
            logger.info(
                "BeliefGraph initialized (Layer 7): %d edges rehydrated",
                edge_count,
            )

            self._run_full_propagation()
            self._last_propagation_run = time.time()

            return True
        except Exception:
            logger.exception("BeliefGraph initialization failed")
            return False

    # -- Tick cycle ---------------------------------------------------------

    def on_tick(self) -> dict[str, Any] | None:
        """Called from consciousness system tick cycle."""
        if not self._ensure_initialized():
            return None

        assert self._edge_store is not None
        assert self._bridge is not None

        self._tick_count += 1
        now = time.time()

        if now - self._last_compaction > 3600:
            if self._edge_store.needs_compaction():
                self._edge_store.compact()
                self._last_compaction = now

        if now - self._last_propagation_run >= self.PROPAGATION_CADENCE_S:
            prop_result = self._run_full_propagation()
            self._last_propagation_run = now
            if prop_result:
                try:
                    from consciousness.events import event_bus, BELIEF_GRAPH_PROPAGATION_COMPLETE
                    event_bus.emit(BELIEF_GRAPH_PROPAGATION_COMPLETE, **prop_result)
                except Exception:
                    pass

        state = self.get_state()

        if state.get("integrity"):
            try:
                from consciousness.events import event_bus, BELIEF_GRAPH_INTEGRITY_CHECK
                event_bus.emit(BELIEF_GRAPH_INTEGRITY_CHECK, **state["integrity"])
            except Exception:
                pass

        return state

    # -- New belief hook ----------------------------------------------------

    def on_new_belief(self, belief: Any) -> None:
        """Called when Layer 5 extracts a new active belief.

        Creates extraction co-occurrence and gated shared-subject edges.
        Only processes beliefs with resolution_state == 'active'.
        """
        if not self._initialized or self._bridge is None:
            return
        if getattr(belief, "resolution_state", None) != "active":
            return

        try:
            self._bridge.create_extraction_links(belief)
            self._bridge.create_shared_subject_support(belief)
            self._bridge.create_temporal_sequence_links(belief)
        except Exception:
            logger.exception("BeliefGraph.on_new_belief error")

    # -- Version link hook --------------------------------------------------

    def on_belief_versioned(self, old_belief_id: str, new_belief_id: str) -> None:
        """Called when Layer 5 versions a belief (temporal resolution)."""
        if not self._initialized or self._bridge is None:
            return
        try:
            self._bridge.create_version_link(old_belief_id, new_belief_id)
        except Exception:
            logger.exception("BeliefGraph.on_belief_versioned error")

    # -- Prerequisite hook --------------------------------------------------

    def on_prerequisite_detected(
        self,
        dependent_belief_id: str,
        prerequisite_belief_id: str,
        strength: float = 0.6,
        evidence_basis: str = "causal",
    ) -> bool:
        """Public entry point for the prerequisite-tracking writer.

        Produces a ``depends_on`` edge (dependent -> prerequisite). The
        caller is responsible for having already identified both belief
        IDs. Cycle prevention is enforced in the bridge: a new edge that
        would close a forward-dependency cycle is rejected.

        Returns True iff an edge was created or merged.
        """
        if not self._ensure_initialized() or self._bridge is None:
            return False
        try:
            return self._bridge.create_prerequisite_link(
                dependent_belief_id=dependent_belief_id,
                prerequisite_belief_id=prerequisite_belief_id,
                strength=strength,
                evidence_basis=evidence_basis,
            )
        except Exception:
            logger.exception("BeliefGraph.on_prerequisite_detected error")
            return False

    # -- User correction hook -----------------------------------------------

    def on_user_correction(
        self,
        corrected_belief_id: str,
        correcting_belief_id: str,
        strength: float = 0.9,
    ) -> bool:
        """Public entry point for user-correction edge emission.

        Produces a ``contradicts`` edge with evidence_basis ``user_correction``
        (decay-immune). Callers must already have the two belief IDs; the
        mapping from raw correction text to a belief ID is the responsibility
        of the calibration layer.

        Returns True iff an edge was created or merged.
        """
        if not self._ensure_initialized() or self._bridge is None:
            return False
        try:
            return self._bridge.create_user_correction_link(
                corrected_belief_id=corrected_belief_id,
                correcting_belief_id=correcting_belief_id,
                strength=strength,
            )
        except Exception:
            logger.exception("BeliefGraph.on_user_correction error")
            return False

    # -- Dream cycle --------------------------------------------------------

    def on_dream_cycle(self) -> None:
        """Called during dream/deep_learning cycles for maintenance."""
        if not self._initialized or self._edge_store is None:
            return
        try:
            removed = self._edge_store.decay_strengths(days=1.0)
            if removed:
                logger.info("BeliefGraph dream: decayed edges, removed %d weak edges", removed)
            self._edge_store.compact()
            self._last_compaction = time.time()

            if self._bridge:
                filled = self._bridge.fill_orphan_edges(max_per_cycle=30)
                if filled:
                    logger.info("BeliefGraph dream: filled %d orphan edges", filled)

            self._run_full_propagation()
        except Exception:
            logger.exception("BeliefGraph.on_dream_cycle error")

    def _run_full_propagation(self) -> dict[str, Any] | None:
        """Run full graph propagation.  View-only: no writes to beliefs."""
        if not self._initialized or self._edge_store is None or self._belief_store is None:
            return None
        try:
            from epistemic.belief_graph.propagation import propagate_all
            views = propagate_all(self._edge_store, self._belief_store)
            self._last_propagation_views = views
            self._propagation_ran_once = True

            boosted = sum(1 for v in views.values() if v.structural_confidence_delta > 0.01)
            diminished = sum(1 for v in views.values() if v.structural_confidence_delta < -0.01)
            if views:
                logger.info(
                    "BeliefGraph propagation: %d beliefs, %d boosted, %d diminished",
                    len(views), boosted, diminished,
                )
            return {
                "total": len(views),
                "boosted": boosted,
                "diminished": diminished,
                "neutral": len(views) - boosted - diminished,
            }
        except Exception:
            logger.exception("Propagation error")
            return None

    def get_propagation_views(self) -> dict[str, Any]:
        """Return latest propagation views for observer / communicator."""
        return getattr(self, "_last_propagation_views", {})

    # -- State (dashboard) --------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        if not self._initialized or self._edge_store is None:
            return {}

        edge_stats = self._edge_store.get_stats()
        bridge_stats = self._bridge.get_stats() if self._bridge else {}

        topology_stats: dict[str, Any] = {}
        integrity_stats: dict[str, Any] = {}
        try:
            from epistemic.belief_graph.topology import (
                get_top_beliefs_by_centrality,
                get_roots,
                get_leaves,
            )
            topology_stats["top_centrality"] = [
                {"belief_id": bid[:12], "centrality": c}
                for bid, c in get_top_beliefs_by_centrality(self._edge_store, n=5)
            ]
            topology_stats["root_count"] = len(get_roots(self._edge_store))
            topology_stats["leaf_count"] = len(get_leaves(self._edge_store))
        except Exception:
            pass

        try:
            from epistemic.belief_graph.integrity import compute_integrity
            integrity_stats = compute_integrity(self._edge_store, self._belief_store)
        except Exception:
            pass

        propagation_stats: dict[str, Any] = {
            "ran_once": self._propagation_ran_once,
            "last_run_ts": self._last_propagation_run if self._last_propagation_run > 0 else None,
            "belief_count": 0,
        }
        views = getattr(self, "_last_propagation_views", {})
        if views:
            deltas = [v.structural_confidence_delta for v in views.values()]
            propagation_stats.update({
                "belief_count": len(views),
                "boosted": sum(1 for d in deltas if d > 0.01),
                "diminished": sum(1 for d in deltas if d < -0.01),
                "max_delta": round(max(deltas), 4) if deltas else 0.0,
                "min_delta": round(min(deltas), 4) if deltas else 0.0,
            })

        return {
            "tick_count": self._tick_count,
            "initialized": self._initialized,
            **edge_stats,
            "bridge": bridge_stats,
            "topology": topology_stats,
            "integrity": integrity_stats,
            "propagation": propagation_stats,
        }
