"""Cleanup memory: nearest-symbol lookup for noisy HRR outputs.

After an unbind operation, the recovered vector typically has noise on the
order of 1/sqrt(dim) per component. The cleanup memory snaps it back to the
closest clean symbol in a registered vocabulary.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from library.vsa.hrr import HRRConfig, similarity


class CleanupMemory:
    """Cosine nearest-neighbor lookup over a fixed symbol table.

    Build once with a set of ``(label, vector)`` pairs; then call
    :meth:`lookup` or :meth:`topk`. Snap distance is cosine similarity.
    """

    def __init__(self, cfg: HRRConfig) -> None:
        self._cfg = cfg
        self._labels: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, dim)
        self._norms: Optional[np.ndarray] = None  # shape (N,)

    def add(self, label: str, vector: np.ndarray) -> None:
        """Add a single clean symbol. Rebuilds the internal matrix on next query."""
        if vector.shape != (self._cfg.dim,):
            raise ValueError(f"cleanup.add: expected shape ({self._cfg.dim},), got {vector.shape}")
        self._labels.append(label)
        v = vector.astype(np.float64, copy=False)[None, :]  # keep precision for dot products
        if self._matrix is None:
            self._matrix = v.copy()
        else:
            self._matrix = np.concatenate([self._matrix, v], axis=0)
        self._norms = None

    def add_many(self, pairs: Iterable[Tuple[str, np.ndarray]]) -> None:
        for label, vec in pairs:
            self.add(label, vec)

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> Sequence[str]:
        return tuple(self._labels)

    def _ensure_norms(self) -> None:
        if self._matrix is None:
            return
        if self._norms is None:
            norms = np.linalg.norm(self._matrix, axis=1)
            norms[norms == 0.0] = 1.0
            self._norms = norms

    def lookup(self, vec: np.ndarray) -> Tuple[Optional[str], float]:
        """Return ``(label, score)`` of the best match, or ``(None, 0.0)`` if empty."""
        results = self.topk(vec, k=1)
        return results[0] if results else (None, 0.0)

    def topk(self, vec: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Return the ``k`` best ``(label, cosine)`` matches, sorted descending."""
        if self._matrix is None or len(self._labels) == 0:
            return []
        if vec.shape != (self._cfg.dim,):
            raise ValueError(f"cleanup.topk: expected shape ({self._cfg.dim},), got {vec.shape}")
        self._ensure_norms()
        q = vec.astype(np.float64, copy=False)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        sims = (self._matrix @ q) / (self._norms * qn)
        k_eff = min(k, sims.shape[0])
        # argpartition for efficiency, then sort the small slice
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._labels[int(i)], float(sims[int(i)])) for i in idx]


def false_positive_decoys(
    query: np.ndarray,
    cleanup: CleanupMemory,
    expected_label: Optional[str],
    threshold: float = 0.5,
) -> int:
    """Count how many non-expected labels exceed ``threshold``.

    Used to characterize the FP rate of a cleanup memory against noisy queries.
    """
    count = 0
    for label, score in cleanup.topk(query, k=len(cleanup)):
        if label == expected_label:
            continue
        if score >= threshold:
            count += 1
    return count


_ = similarity  # re-export preserved via __init__; keep static analyzers happy
