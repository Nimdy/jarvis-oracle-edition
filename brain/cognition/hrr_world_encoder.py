"""HRR shadow encoder for :class:`cognition.world_state.WorldState`.

Pure function: reads a ``WorldState``, returns a composite HRR vector plus
cleanup-quality metrics. The vector itself is returned to the caller to keep
in memory; the metrics dict is the only thing exposed via `/api/hrr/status`.

**Non-negotiable**: this module never imports policy / belief / memory /
autonomy / identity writers. The scan in
``jarvis_eval.validation_pack._scan_hrr_forbidden_imports`` mechanically
enforces this.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from library.vsa.cleanup import CleanupMemory
from library.vsa.hrr import HRRConfig, bind, project, similarity, superpose
from library.vsa.symbols import SymbolDictionary


# ---------------------------------------------------------------------------
# Bucketing helpers — keep the HRR vocabulary finite and deterministic.
# ---------------------------------------------------------------------------

def _bucket_float(value: float, edges: Iterable[Tuple[float, str]], default: str) -> str:
    """Bucket ``value`` according to ``[(max_exclusive, label), ...]``."""
    for upper, label in edges:
        if value < upper:
            return label
    return default


_ENGAGEMENT_BUCKETS = ((0.33, "low"), (0.66, "mid"))
_HEALTH_BUCKETS = ((0.33, "bad"), (0.66, "ok"))
_UNCERTAINTY_BUCKETS = ((0.33, "low"), (0.66, "mid"))


def _facts_from_world_state(ws: Any) -> Dict[str, str]:
    """Return ``{role_name: filler_label}`` — every pair is a future binding.

    Uses ``getattr`` + dotted-path resolution so this keeps working if new
    fields are added to :class:`WorldState` later without forcing a schema
    bump here.
    """
    facts: Dict[str, str] = {}

    user = getattr(ws, "user", None)
    if user is not None:
        facts["user_present"] = "yes" if getattr(user, "present", False) else "no"
        facts["user_engagement"] = _bucket_float(
            float(getattr(user, "engagement", 0.0) or 0.0),
            _ENGAGEMENT_BUCKETS, "high",
        )
        facts["user_emotion"] = str(getattr(user, "emotion", "neutral") or "neutral")
        speaker = str(getattr(user, "speaker_name", "") or "unknown")
        facts["speaker"] = speaker if speaker else "unknown"
        facts["gesture"] = str(getattr(user, "gesture", "") or "none") or "none"

    conv = getattr(ws, "conversation", None)
    if conv is not None:
        facts["conversation_active"] = "yes" if getattr(conv, "active", False) else "no"
        topic = str(getattr(conv, "topic", "") or "none")
        facts["topic"] = topic if topic else "none"
        facts["follow_up"] = "yes" if getattr(conv, "follow_up_active", False) else "no"

    sys_ = getattr(ws, "system", None)
    if sys_ is not None:
        facts["mode"] = str(getattr(sys_, "mode", "passive") or "passive")
        facts["health_bucket"] = _bucket_float(
            float(getattr(sys_, "health_score", 1.0) or 1.0),
            _HEALTH_BUCKETS, "good",
        )
        facts["active_goal_kind"] = str(getattr(sys_, "active_goal_kind", "") or "none") or "none"

    phys = getattr(ws, "physical", None)
    if phys is not None:
        count = int(getattr(phys, "person_count", 0) or 0)
        if count == 0:
            facts["person_count"] = "zero"
        elif count == 1:
            facts["person_count"] = "one"
        else:
            facts["person_count"] = "many"

    unc = getattr(ws, "uncertainty", None) or {}
    if isinstance(unc, dict) and unc:
        mean_u = float(sum(unc.values()) / max(1, len(unc)))
        facts["uncertainty_bucket"] = _bucket_float(mean_u, _UNCERTAINTY_BUCKETS, "high")

    return facts


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

def encode_world_state(
    world_state: Any,
    cfg: HRRConfig,
    symbols: SymbolDictionary,
    prev_vector: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Encode a WorldState snapshot as a composite HRR vector.

    Returns a dict with:

    * ``vector`` — the composite unit-norm HRR vector (ndarray). Kept for the
      caller's in-memory history only; never written to disk, never exposed
      via the status API.
    * ``facts_encoded`` — number of role-filler bindings in the composite.
    * ``binding_cleanliness`` — mean cosine of the recovered filler against
      the expected filler after unbind + projection.
    * ``cleanup_accuracy`` — fraction of roles whose cleanup-top-1 matches
      the expected filler.
    * ``similarity_to_previous`` — cosine similarity to ``prev_vector`` if
      provided, else ``None``.
    * ``side_effects`` — always ``0`` (this function performs none).
    """
    facts = _facts_from_world_state(world_state)
    if not facts:
        empty = project(np.zeros(cfg.dim, dtype=np.float32), cfg)
        return {
            "vector": empty,
            "facts_encoded": 0,
            "binding_cleanliness": None,
            "cleanup_accuracy": None,
            "similarity_to_previous": None,
            "side_effects": 0,
        }

    # Build bindings and a per-call cleanup memory scoped to the fillers
    # that appear in this snapshot (world-state vocabulary is small and
    # bounded, so this stays cheap).
    bindings = []
    role_filler_pairs = []
    cleanup = CleanupMemory(cfg)
    seen_fillers: Dict[str, np.ndarray] = {}
    for role_name, filler_label in facts.items():
        r_vec = symbols.role(role_name)
        f_label = f"{role_name}:{filler_label}"
        f_vec = seen_fillers.get(f_label)
        if f_vec is None:
            f_vec = symbols.entity(f_label)
            seen_fillers[f_label] = f_vec
            cleanup.add(f_label, f_vec)
        bindings.append(bind(r_vec, f_vec, cfg))
        role_filler_pairs.append((role_name, r_vec, f_label))

    composite = project(superpose(bindings, cfg), cfg)

    # Cleanup quality: how well does each role probe the correct filler?
    cleanliness_scores = []
    cleanup_hits = 0
    for role_name, r_vec, expected_label in role_filler_pairs:
        from library.vsa.hrr import unbind as _unbind
        recovered = _unbind(composite, r_vec, cfg)
        expected_vec = seen_fillers[expected_label]
        cleanliness_scores.append(similarity(recovered, expected_vec))
        top = cleanup.topk(recovered, k=1)
        if top and top[0][0] == expected_label:
            cleanup_hits += 1

    binding_cleanliness = (
        float(sum(cleanliness_scores) / len(cleanliness_scores)) if cleanliness_scores else None
    )
    cleanup_accuracy = cleanup_hits / len(role_filler_pairs) if role_filler_pairs else None

    sim_prev = None
    if prev_vector is not None and isinstance(prev_vector, np.ndarray):
        sim_prev = float(similarity(composite, prev_vector))

    return {
        "vector": composite,
        "facts_encoded": len(role_filler_pairs),
        "binding_cleanliness": binding_cleanliness,
        "cleanup_accuracy": cleanup_accuracy,
        "similarity_to_previous": sim_prev,
        "side_effects": 0,
    }


# ---------------------------------------------------------------------------
# Ring-buffered shadow owner (used by consciousness/engine)
# ---------------------------------------------------------------------------

class HRRWorldShadow:
    """Holds the rolling HRR world-shadow history.

    Construct once per engine, then call :meth:`maybe_sample` on each tick.
    Safe to call when disabled: it returns immediately without touching the
    world-state object.
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
        self._samples = deque(maxlen=self.RING_CAPACITY)
        self._samples_total = 0
        self._tick_counter = 0
        self._last_vector: Optional[np.ndarray] = None
        self._last_metrics: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(self._runtime.enabled)

    def maybe_sample(self, world_state: Any) -> Optional[Dict[str, Any]]:
        """Sample the encoder iff enabled AND on the right tick boundary."""
        if not self.enabled or world_state is None:
            return None
        self._tick_counter += 1
        every = max(1, int(self._runtime.sample_every_ticks))
        if self._tick_counter % every != 0:
            return None
        result = encode_world_state(
            world_state, self._cfg, self._symbols, prev_vector=self._last_vector
        )
        # Strip the raw vector before storing/exposing.
        metrics = {
            "tick": int(self._tick_counter),
            "facts_encoded": int(result["facts_encoded"]),
            "binding_cleanliness": result["binding_cleanliness"],
            "cleanup_accuracy": result["cleanup_accuracy"],
            "similarity_to_previous": result["similarity_to_previous"],
            "side_effects": 0,
        }
        self._samples_total += 1
        self._samples.append(metrics)
        self._last_vector = result["vector"]
        self._last_metrics = metrics
        return metrics

    def status(self) -> Dict[str, Any]:
        latest = self._last_metrics
        return {
            "enabled": self.enabled,
            "samples_total": int(self._samples_total),
            "samples_retained": int(len(self._samples)),
            "ring_capacity": int(self.RING_CAPACITY),
            "binding_cleanliness": latest["binding_cleanliness"] if latest else None,
            "cleanup_accuracy": latest["cleanup_accuracy"] if latest else None,
            "similarity_to_previous": latest["similarity_to_previous"] if latest else None,
        }

    def recent(self, n: int = 20) -> list:
        """Return the last ``n`` metrics dicts (no vectors)."""
        if n <= 0:
            return []
        return list(self._samples)[-n:]
