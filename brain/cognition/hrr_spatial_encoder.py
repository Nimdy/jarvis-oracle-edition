"""HRR encoder for the P5 mental-world spatial scene graph.

Pure function: reads a :class:`cognition.spatial_scene_graph.MentalWorldSceneGraph`,
returns a composite HRR vector plus cleanup-quality metrics. The vector itself
is returned to the caller to keep in memory; the metrics dict is the only
thing exposed via ``/api/hrr/scene`` and the spatial-shadow ring.

**Non-negotiable**: this module never imports policy / belief / memory /
autonomy / identity / soul writers. The structural scan in
``jarvis_eval.validation_pack._scan_hrr_forbidden_imports`` mechanically
enforces that guarantee across the P5 module roots.

Shape of the encoded composite per scene::

    composite = Σ ( ENTITY[id] ⊛ STATE[entity.state] )
              + Σ ( REL[type]   ⊛ ENTITY[source]      )
              + Σ ( REL[type]   ⊛ ENTITY[target]      )

This is the standard role/filler superposition Plate 1995 describes for
scene representations: probing with an entity recovers its state, probing
with a relation recovers a noisy mixture of (source, target) pairs that
the cleanup memory disambiguates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cognition.spatial_scene_graph import MentalWorldSceneGraph
from library.vsa.cleanup import CleanupMemory
from library.vsa.hrr import HRRConfig, bind, project, similarity, superpose, unbind
from library.vsa.spatial_symbols import (
    NAMESPACE_ENTITY,
    NAMESPACE_RELATION,
    NAMESPACE_STATE,
    SPATIAL_RELATION_SYMBOLS,
    SPATIAL_STATE_SYMBOLS,
    seed_all_symbols,
)
from library.vsa.symbols import SymbolDictionary


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def encode_scene_graph(
    graph: MentalWorldSceneGraph,
    cfg: HRRConfig,
    symbols: SymbolDictionary,
    prev_vector: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Encode a scene graph as a composite HRR vector + cleanup metrics.

    Returns a dict with:

    * ``vector`` — composite unit-norm HRR vector (``np.ndarray``). Kept in
      memory only by the caller. Never written to disk, never exposed
      through any API surface.
    * ``entities_encoded`` — number of entity / state bindings.
    * ``relations_encoded`` — number of relation / entity bindings.
    * ``binding_cleanliness`` — mean cosine of recovered fillers vs.
      expected fillers across all unbind probes.
    * ``cleanup_accuracy`` — fraction of entity-state probes whose
      cleanup top-1 matches the expected state label.
    * ``relation_recovery`` — fraction of relation probes whose cleanup
      top-1 matches one of the expected (source or target) entities.
    * ``cleanup_failures`` — count of entity-state probes whose top-1
      label is wrong.
    * ``similarity_to_previous`` — cosine vs. ``prev_vector`` if provided.
    * ``side_effects`` — always 0. Pure function.
    """
    if not isinstance(graph, MentalWorldSceneGraph):
        raise TypeError(f"encode_scene_graph: expected MentalWorldSceneGraph, got {type(graph).__name__}")

    # Empty / unavailable graph → empty composite, nothing encoded.
    if graph.entity_count == 0:
        empty = project(np.zeros(cfg.dim, dtype=np.float32), cfg)
        return {
            "vector": empty,
            "entities_encoded": 0,
            "relations_encoded": 0,
            "binding_cleanliness": None,
            "cleanup_accuracy": None,
            "relation_recovery": None,
            "cleanup_failures": 0,
            "similarity_to_previous": None,
            "side_effects": 0,
            "reason": graph.reason,
        }

    # Make sure the standard P5 vocabulary is seeded so cleanup memories
    # have stable labels across calls / processes.
    seed_all_symbols(symbols)

    bindings: list[np.ndarray] = []
    cleanliness_scores: list[float] = []

    # Cleanup memories scoped to the labels that appear in this scene. Two
    # memories: one for entity ids (used by relation probes), one for
    # state labels (used by entity-state probes).
    entity_cleanup = CleanupMemory(cfg)
    state_cleanup = CleanupMemory(cfg)
    seen_entities: set[str] = set()
    seen_states: set[str] = set()

    # ----- Entity * state bindings -----
    entity_state_pairs: list[tuple[str, np.ndarray, str, np.ndarray]] = []
    for ent in graph.entities:
        e_vec = symbols.get(NAMESPACE_ENTITY, ent.entity_id)
        s_vec = symbols.get(NAMESPACE_STATE, ent.state)
        bindings.append(bind(e_vec, s_vec, cfg))
        if ent.entity_id not in seen_entities:
            entity_cleanup.add(ent.entity_id, e_vec)
            seen_entities.add(ent.entity_id)
        if ent.state not in seen_states:
            state_cleanup.add(ent.state, s_vec)
            seen_states.add(ent.state)
        entity_state_pairs.append((ent.entity_id, e_vec, ent.state, s_vec))

    # ----- Relation * entity bindings (one per direction) -----
    relation_probes: list[tuple[str, np.ndarray, set[str]]] = []
    relation_target_map: dict[str, set[str]] = {}
    for rel in graph.relations:
        r_vec = symbols.get(NAMESPACE_RELATION, rel.relation_type)
        src_vec = symbols.get(NAMESPACE_ENTITY, rel.source_entity_id)
        tgt_vec = symbols.get(NAMESPACE_ENTITY, rel.target_entity_id)
        bindings.append(bind(r_vec, src_vec, cfg))
        bindings.append(bind(r_vec, tgt_vec, cfg))

        # Make sure all relation-side entities have cleanup entries, even
        # entities that weren't in the entity list (shouldn't happen with
        # the canonical adapter, but keep the encoder defensive).
        for eid, evec in (
            (rel.source_entity_id, src_vec),
            (rel.target_entity_id, tgt_vec),
        ):
            if eid not in seen_entities:
                entity_cleanup.add(eid, evec)
                seen_entities.add(eid)

        rkey = rel.relation_type
        relation_target_map.setdefault(rkey, set()).update(
            {rel.source_entity_id, rel.target_entity_id}
        )

    composite = project(superpose(bindings, cfg), cfg)

    # ----- Probe quality -----
    cleanup_failures = 0
    cleanup_hits = 0
    for entity_id, e_vec, expected_state, s_vec in entity_state_pairs:
        recovered = unbind(composite, e_vec, cfg)
        cleanliness_scores.append(similarity(recovered, s_vec))
        top = state_cleanup.topk(recovered, k=1)
        if top and top[0][0] == expected_state:
            cleanup_hits += 1
        else:
            cleanup_failures += 1

    relation_hits = 0
    relation_total = 0
    seen_relation_keys: set[str] = set()
    for rel_type, expected_set in relation_target_map.items():
        if rel_type in seen_relation_keys:
            continue
        seen_relation_keys.add(rel_type)
        r_vec = symbols.get(NAMESPACE_RELATION, rel_type)
        recovered = unbind(composite, r_vec, cfg)
        cleanliness_scores.append(
            max(
                (similarity(recovered, symbols.get(NAMESPACE_ENTITY, eid)) for eid in expected_set),
                default=0.0,
            )
        )
        # Probe top-3 against the entity cleanup; success if any hit lands in expected_set.
        top = entity_cleanup.topk(recovered, k=3)
        relation_total += 1
        if top and any(label in expected_set for label, _ in top):
            relation_hits += 1

    binding_cleanliness = (
        float(sum(cleanliness_scores) / len(cleanliness_scores))
        if cleanliness_scores else None
    )
    cleanup_accuracy = (
        cleanup_hits / len(entity_state_pairs) if entity_state_pairs else None
    )
    relation_recovery = (
        relation_hits / relation_total if relation_total else None
    )

    sim_prev = None
    if prev_vector is not None and isinstance(prev_vector, np.ndarray):
        sim_prev = float(similarity(composite, prev_vector))
        # Clamp float-precision overshoot (cosine should be in [-1, 1]).
        if sim_prev > 1.0:
            sim_prev = 1.0
        elif sim_prev < -1.0:
            sim_prev = -1.0

    return {
        "vector": composite,
        "entities_encoded": len(entity_state_pairs),
        "relations_encoded": len(graph.relations),
        "binding_cleanliness": binding_cleanliness,
        "cleanup_accuracy": cleanup_accuracy,
        "relation_recovery": relation_recovery,
        "cleanup_failures": int(cleanup_failures),
        "similarity_to_previous": sim_prev,
        "side_effects": 0,
        "reason": graph.reason,
    }


# ---------------------------------------------------------------------------
# Ring-buffered shadow owner (used by consciousness/engine in Commit 5)
# ---------------------------------------------------------------------------


class HRRSpatialShadow:
    """Holds the rolling HRR spatial-scene shadow history.

    Construct once per engine, then call :meth:`maybe_sample` on each tick
    after the canonical scene/spatial state has been refreshed. Safe to
    call when the runtime is disabled: it returns ``None`` immediately
    without touching the input.

    Twin-gate aware: requires both ``runtime_cfg.enabled`` (P4 master gate)
    and ``runtime_cfg.spatial_scene_enabled`` (P5 twin gate) to sample.
    """

    RING_CAPACITY = 500

    def __init__(
        self,
        runtime_cfg: Any,
        symbol_seed: int = 0,
    ) -> None:
        from collections import deque

        self._runtime = runtime_cfg
        self._cfg = HRRConfig(dim=int(runtime_cfg.dim), seed=int(symbol_seed))
        self._symbols = SymbolDictionary(self._cfg)
        seed_all_symbols(self._symbols)
        self._samples: "deque[Dict[str, Any]]" = deque(maxlen=self.RING_CAPACITY)
        self._scene_payloads: "deque[Dict[str, Any]]" = deque(maxlen=self.RING_CAPACITY)
        self._samples_total = 0
        self._tick_counter = 0
        self._last_vector: Optional[np.ndarray] = None
        self._last_metrics: Optional[Dict[str, Any]] = None
        self._last_scene_payload: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._runtime, "spatial_scene_active", False))

    def maybe_sample(self, graph: Optional[MentalWorldSceneGraph]) -> Optional[Dict[str, Any]]:
        """Sample the encoder iff active AND on the right tick boundary."""
        if not self.enabled or graph is None:
            return None
        self._tick_counter += 1
        every = max(1, int(getattr(self._runtime, "spatial_scene_sample_every_ticks", 50)))
        if self._tick_counter % every != 0:
            return None
        result = encode_scene_graph(
            graph, self._cfg, self._symbols, prev_vector=self._last_vector
        )
        # Strip the raw vector before storing/exposing.
        metrics = {
            "tick": int(self._tick_counter),
            "timestamp": float(graph.timestamp),
            "entities_encoded": int(result["entities_encoded"]),
            "relations_encoded": int(result["relations_encoded"]),
            "binding_cleanliness": result["binding_cleanliness"],
            "cleanup_accuracy": result["cleanup_accuracy"],
            "relation_recovery": result["relation_recovery"],
            "cleanup_failures": int(result["cleanup_failures"]),
            "similarity_to_previous": result["similarity_to_previous"],
            "spatial_hrr_side_effects": 0,
            "reason": result["reason"],
        }
        scene_payload = graph.to_dict()
        scene_payload["tick"] = int(self._tick_counter)
        scene_payload["metrics"] = {
            k: v for k, v in metrics.items() if k not in ("tick", "timestamp")
        }
        self._samples_total += 1
        self._samples.append(metrics)
        self._scene_payloads.append(scene_payload)
        self._last_vector = result["vector"]
        self._last_metrics = metrics
        self._last_scene_payload = scene_payload
        return metrics

    def status(self) -> Dict[str, Any]:
        """Return a no-vector status summary for ``/api/hrr/status``."""
        latest = self._last_metrics
        return {
            "enabled": self.enabled,
            "samples_total": int(self._samples_total),
            "samples_retained": int(len(self._samples)),
            "ring_capacity": int(self.RING_CAPACITY),
            "entities_encoded": latest["entities_encoded"] if latest else None,
            "relations_encoded": latest["relations_encoded"] if latest else None,
            "binding_cleanliness": latest["binding_cleanliness"] if latest else None,
            "cleanup_accuracy": latest["cleanup_accuracy"] if latest else None,
            "relation_recovery": latest["relation_recovery"] if latest else None,
            "cleanup_failures": latest["cleanup_failures"] if latest else 0,
            "similarity_to_previous": latest["similarity_to_previous"] if latest else None,
            "spatial_hrr_side_effects": 0,
            "reason": latest["reason"] if latest else None,
        }

    def recent(self, n: int = 20) -> list:
        """Return the last ``n`` metric dicts (no vectors)."""
        if n <= 0:
            return []
        return list(self._samples)[-n:]

    def latest_scene_payload(self) -> Optional[Dict[str, Any]]:
        """Return the most-recent serialized scene payload, or None."""
        return dict(self._last_scene_payload) if self._last_scene_payload else None

    def recent_scenes(self, n: int = 20) -> list:
        """Return the last ``n`` serialized scene payloads (no vectors)."""
        if n <= 0:
            return []
        return [dict(p) for p in list(self._scene_payloads)[-n:]]


__all__ = [
    "encode_scene_graph",
    "HRRSpatialShadow",
]
