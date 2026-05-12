"""HRR shadow observer for :class:`cognition.simulator.MentalSimulator`.

When enabled at boot, this observer records an HRR metrics trace *after*
each simulation completes. It never mutates the simulator's trace, never
reaches into the returned :class:`SimulationTrace`, never calls the causal
engine, never triggers the simulator's trace-log append. The simulator's
public return value is byte-identical with the shadow on or off.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from cognition.hrr_world_encoder import encode_world_state
from library.vsa.hrr import HRRConfig, similarity
from library.vsa.symbols import SymbolDictionary


class HRRSimulationShadow:
    """Records HRR metrics for each completed mental-simulation trace.

    Construct once per simulator. When disabled, :meth:`record_trace` is a
    cheap no-op. Memory is bounded by a fixed-size deque of lightweight
    metrics dicts; no raw HRR vectors enter the ring buffer.
    """

    RING_CAPACITY = 200

    def __init__(self, runtime_cfg: Any, symbol_seed: int = 0) -> None:
        self._runtime = runtime_cfg
        self._cfg = HRRConfig(dim=int(runtime_cfg.dim), seed=int(symbol_seed))
        self._symbols = SymbolDictionary(self._cfg)
        self._samples: "deque[Dict[str, Any]]" = deque(maxlen=self.RING_CAPACITY)
        self._samples_total = 0
        self._last_metrics: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(self._runtime.enabled)

    def record_trace(
        self,
        before_ws: Any,
        delta: Any,
        after_ws: Any,
        trace: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one simulation trace as HRR metrics. Returns the metrics
        dict, or ``None`` when disabled / given insufficient inputs.

        The ``trace`` argument is accepted for symmetry but intentionally
        not dereferenced: we don't want this observer coupled to the
        simulator's internal trace schema.
        """
        if not self.enabled or before_ws is None or after_ws is None:
            return None

        try:
            before = encode_world_state(before_ws, self._cfg, self._symbols)
            after = encode_world_state(after_ws, self._cfg, self._symbols)
        except Exception:
            return None  # observer must never take the simulator down

        delta_similarity = float(similarity(before["vector"], after["vector"]))
        cleanliness_before = before["binding_cleanliness"]
        cleanliness_after = after["binding_cleanliness"]
        cleanliness_delta = None
        if cleanliness_before is not None and cleanliness_after is not None:
            cleanliness_delta = float(cleanliness_after - cleanliness_before)

        delta_event = None
        if delta is not None:
            delta_event = str(getattr(delta, "event", "") or "") or None
            delta_facet = str(getattr(delta, "facet", "") or "") or None
        else:
            delta_facet = None

        metrics = {
            "delta_event": delta_event,
            "delta_facet": delta_facet,
            "facts_before": int(before["facts_encoded"]),
            "facts_after": int(after["facts_encoded"]),
            "delta_similarity": delta_similarity,
            "cleanliness_before": cleanliness_before,
            "cleanliness_after": cleanliness_after,
            "cleanliness_delta": cleanliness_delta,
            "side_effects": 0,
        }
        self._samples_total += 1
        self._samples.append(metrics)
        self._last_metrics = metrics
        return metrics

    def status(self) -> Dict[str, Any]:
        latest = self._last_metrics
        return {
            "enabled": self.enabled,
            "samples_total": int(self._samples_total),
            "samples_retained": int(len(self._samples)),
            "ring_capacity": int(self.RING_CAPACITY),
            "last_delta_similarity": latest["delta_similarity"] if latest else None,
            "last_cleanliness_after": latest["cleanliness_after"] if latest else None,
        }

    def recent(self, n: int = 20) -> list:
        if n <= 0:
            return []
        return list(self._samples)[-n:]
