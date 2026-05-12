"""In-memory HRR recall advisor — shadow observer for FractalRecallEngine.

Non-authoritative: observes each emitted recall, compares what HRR would
have ranked as the nearest memory against the cue, and records lightweight
advisory metrics. **Never** mutates the cue, seed, chain, governance
recommendation, or recall history. **Never** writes to disk. **Never**
touches canonical memory / belief graph / policy / autonomy / identity.

Design constraints for this sprint (per plan Commit 6):

* In-memory cache only (no ``~/.jarvis/cache/hrr_memory_vectors.jsonl``).
* Cache is a bounded LRU (``LRU_CAPACITY = 2000`` entries).
* Observation ring buffer is ``deque(maxlen=500)``.
* Observations contain only scalar metrics + string ids, never raw vectors.

File I/O is deliberately NOT imported at module scope. The unit tests
assert this by snapshotting ``HOME`` / ``~/.jarvis/`` before and after a
simulated observation burst and asserting zero writes.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from library.vsa.hrr import HRRConfig, project, similarity, superpose
from library.vsa.symbols import SymbolDictionary


# ---------------------------------------------------------------------------
# Content → deterministic HRR vector (no filesystem, no memory writes)
# ---------------------------------------------------------------------------

def _memory_text(memory: Any) -> str:
    """Pull out a reasonable text snippet from a memory object/dict."""
    if memory is None:
        return ""
    if isinstance(memory, dict):
        for key in ("content", "text", "summary", "title"):
            val = memory.get(key)
            if isinstance(val, str) and val:
                return val
        return ""
    for attr in ("content", "text", "summary", "title"):
        val = getattr(memory, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


def _tokens(text: str, limit: int = 16) -> List[str]:
    """Cheap lowercase tokenization; stable and deterministic."""
    if not text:
        return []
    out = []
    acc = []
    for ch in text.lower():
        if ch.isalnum():
            acc.append(ch)
        else:
            if acc:
                out.append("".join(acc))
                acc.clear()
                if len(out) >= limit:
                    return out
    if acc and len(out) < limit:
        out.append("".join(acc))
    return out


def _encode_tokens(tokens: List[str], cfg: HRRConfig, symbols: SymbolDictionary) -> np.ndarray:
    if not tokens:
        return project(np.zeros(cfg.dim, dtype=np.float32), cfg)
    vecs = [symbols.entity(f"tok:{t}") for t in tokens[:16]]
    return project(superpose(vecs, cfg), cfg)


# ---------------------------------------------------------------------------
# HRRRecallAdvisor
# ---------------------------------------------------------------------------

class HRRRecallAdvisor:
    """Shadow observer for fractal-recall outputs. Zero authority.

    Call :meth:`observe` from :class:`memory.fractal_recall.FractalRecallEngine`
    just before returning a :class:`FractalRecallResult`. The method returns
    a metrics dict (or ``None`` when disabled) for callers that want to log
    it; the advisor also retains it in its ring buffer.
    """

    LRU_CAPACITY = 2000
    RING_CAPACITY = 500

    def __init__(self, runtime_cfg: Any, symbol_seed: int = 0) -> None:
        self._runtime = runtime_cfg
        self._cfg = HRRConfig(dim=int(runtime_cfg.dim), seed=int(symbol_seed))
        self._symbols = SymbolDictionary(self._cfg)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._obs: "deque[Dict[str, Any]]" = deque(maxlen=self.RING_CAPACITY)
        self._obs_total = 0
        self._last_metrics: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(self._runtime.enabled)

    # -- cache -----------------------------------------------------------

    def _get_or_build_vector(self, memory_id: str, memory: Any) -> np.ndarray:
        if memory_id in self._cache:
            vec = self._cache.pop(memory_id)
            self._cache[memory_id] = vec
            return vec
        text = _memory_text(memory)
        vec = _encode_tokens(_tokens(text), self._cfg, self._symbols)
        self._cache[memory_id] = vec
        if len(self._cache) > self.LRU_CAPACITY:
            self._cache.popitem(last=False)
        return vec

    def _nearest_to_cue(self, cue_vec: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self._cache:
            return None
        best_id: Optional[str] = None
        best_score = -2.0
        for mid, vec in self._cache.items():
            score = float(similarity(cue_vec, vec))
            if score > best_score:
                best_score = score
                best_id = mid
        return (best_id, best_score) if best_id is not None else None

    # -- public observer -------------------------------------------------

    def observe(
        self,
        cue: Any,
        seed: Any,
        chain: Any,
        governance_action: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or cue is None or seed is None or chain is None:
            return None
        try:
            cue_text = str(getattr(cue, "text", "") or "")
            cue_class = str(getattr(cue, "cue_class", "") or "")
            cue_tokens = _tokens(cue_text)
            # Blend cue-class symbol with token superposition.
            cue_components: List[np.ndarray] = []
            if cue_class:
                cue_components.append(self._symbols.role(f"cue_class:{cue_class}"))
            cue_components.append(_encode_tokens(cue_tokens, self._cfg, self._symbols))
            cue_vec = project(superpose(cue_components, self._cfg), self._cfg)

            seed_id = str(getattr(seed, "memory_id", "") or "")
            # Populate cache for seed + chain (bounded by LRU).
            self._get_or_build_vector(seed_id, getattr(seed, "memory", None))
            chain_ids: List[str] = []
            for cand in list(chain):
                cid = str(getattr(cand, "memory_id", "") or "")
                if cid:
                    chain_ids.append(cid)
                    self._get_or_build_vector(cid, getattr(cand, "memory", None))

            nearest = self._nearest_to_cue(cue_vec)
            hrr_top_id, hrr_top_score = (None, None)
            if nearest is not None:
                hrr_top_id, hrr_top_score = nearest

            seed_overlap = bool(hrr_top_id is not None and hrr_top_id == seed_id)
            help_hint = bool(hrr_top_id in chain_ids) if hrr_top_id else False

            metrics: Dict[str, Any] = {
                "cue_class": cue_class or None,
                "cue_token_count": len(cue_tokens),
                "seed_id": seed_id or None,
                "chain_size": len(chain_ids),
                "hrr_top_id": hrr_top_id,
                "hrr_top_score": hrr_top_score,
                "seed_overlap": seed_overlap,
                "advisory_helpful": help_hint,
                "governance_action": governance_action or None,
                "cache_size": len(self._cache),
                "side_effects": 0,
            }
        except Exception:
            return None

        self._obs_total += 1
        self._obs.append(metrics)
        self._last_metrics = metrics
        return metrics

    def status(self) -> Dict[str, Any]:
        latest = self._last_metrics
        # Running help-rate: fraction of observations where hrr top-1 was
        # in the chain (informational — not a promotion gate this sprint).
        retained = list(self._obs)
        helpful = sum(1 for m in retained if m.get("advisory_helpful"))
        help_rate = float(helpful / len(retained)) if retained else None
        return {
            "enabled": self.enabled,
            "samples_total": int(self._obs_total),
            "samples_retained": int(len(self._obs)),
            "ring_capacity": int(self.RING_CAPACITY),
            "lru_capacity": int(self.LRU_CAPACITY),
            "cache_size": int(len(self._cache)),
            "help_rate": help_rate,
            "last_seed_overlap": latest.get("seed_overlap") if latest else None,
            "last_hrr_top_score": latest.get("hrr_top_score") if latest else None,
        }

    def recent(self, n: int = 20) -> list:
        if n <= 0:
            return []
        return list(self._obs)[-n:]
