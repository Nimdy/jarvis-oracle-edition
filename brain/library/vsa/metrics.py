"""Quality metrics for HRR substrate evaluation.

All metrics are pure functions that consume ndarray vectors and return scalars
or small dicts. No I/O, no logging.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from library.vsa.cleanup import CleanupMemory
from library.vsa.hrr import HRRConfig, bind, similarity, superpose, unbind


def cleanup_accuracy(
    queries: Sequence[Tuple[np.ndarray, str]],
    cleanup: CleanupMemory,
) -> float:
    """Fraction of queries whose nearest cleanup label equals the expected label."""
    if not queries:
        return 0.0
    hits = 0
    for vec, expected in queries:
        label, _ = cleanup.lookup(vec)
        if label == expected:
            hits += 1
    return hits / len(queries)


def cleanup_top3_accuracy(
    queries: Sequence[Tuple[np.ndarray, str]],
    cleanup: CleanupMemory,
) -> float:
    """Fraction of queries whose expected label appears in the top-3 cleanup matches."""
    if not queries:
        return 0.0
    hits = 0
    for vec, expected in queries:
        top = cleanup.topk(vec, k=3)
        if any(label == expected for label, _ in top):
            hits += 1
    return hits / len(queries)


def false_positive_rate(
    queries: Sequence[Tuple[np.ndarray, str]],
    cleanup: CleanupMemory,
    threshold: float = 0.5,
) -> float:
    """Mean count of non-expected labels above ``threshold`` divided by vocab size.

    A well-behaved substrate keeps this below ~0.05 at dim 1024.
    """
    if not queries or len(cleanup) == 0:
        return 0.0
    total = 0
    for vec, expected in queries:
        for label, score in cleanup.topk(vec, k=len(cleanup)):
            if label == expected:
                continue
            if score >= threshold:
                total += 1
    return total / (len(queries) * max(1, len(cleanup)))


def superposition_capacity(
    role_filler_pairs: Sequence[Tuple[str, np.ndarray, str, np.ndarray]],
    cleanup: CleanupMemory,
    cfg: HRRConfig,
    threshold: float = 0.5,
) -> int:
    """How many ``(role, filler)`` bindings can be superposed before recovery fails.

    ``role_filler_pairs`` is a sequence of ``(role_label, role_vec, filler_label,
    filler_vec)`` tuples. Returns the largest prefix length for which every
    role recovers its expected filler via cleanup (top-1, score >= threshold).
    """
    bundle = np.zeros(cfg.dim, dtype=role_filler_pairs[0][1].dtype if role_filler_pairs else np.float32)
    capacity = 0
    for i, (role_label, role_vec, filler_label, filler_vec) in enumerate(role_filler_pairs):
        bundle = bundle + bind(role_vec, filler_vec, cfg)
        ok = True
        for j in range(i + 1):
            rl, rv, fl, _fv = role_filler_pairs[j]
            recovered = unbind(bundle, rv, cfg)
            label, score = cleanup.lookup(recovered)
            if label != fl or score < threshold:
                ok = False
                break
        if ok:
            capacity = i + 1
        else:
            break
    return capacity


def role_recovery_accuracy(
    role_filler_pairs: Sequence[Tuple[str, np.ndarray, str, np.ndarray]],
    cleanup: CleanupMemory,
    cfg: HRRConfig,
) -> float:
    """Bind all pairs once, then probe each role. Fraction of correct recoveries."""
    if not role_filler_pairs:
        return 0.0
    bundle = superpose(
        (bind(role_vec, filler_vec, cfg) for _rl, role_vec, _fl, filler_vec in role_filler_pairs),
        cfg,
    )
    hits = 0
    for _rl, role_vec, filler_label, _fv in role_filler_pairs:
        recovered = unbind(bundle, role_vec, cfg)
        label, _ = cleanup.lookup(recovered)
        if label == filler_label:
            hits += 1
    return hits / len(role_filler_pairs)


def noise_tolerance(
    clean_vec: np.ndarray,
    cleanup: CleanupMemory,
    expected_label: str,
    cfg: HRRConfig,
    noise_levels: Iterable[float] = (0.0, 0.05, 0.10, 0.20),
    trials_per_level: int = 16,
    rng: np.random.Generator | None = None,
) -> Mapping[float, float]:
    """For each noise level, recovery accuracy under additive Gaussian noise."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for sigma in noise_levels:
        hits = 0
        for _ in range(trials_per_level):
            noise = rng.standard_normal(cfg.dim).astype(clean_vec.dtype, copy=False) * sigma
            probe = clean_vec + noise
            label, _ = cleanup.lookup(probe)
            if label == expected_label:
                hits += 1
        out[float(sigma)] = hits / trials_per_level
    return out


def similarity_drift(series: Sequence[np.ndarray]) -> float:
    """Mean cosine similarity between consecutive vectors in a time series.

    Values close to 1.0 mean the state is stable; values near 0 mean it is
    thrashing. Used to sanity-check the world-state shadow encoder.
    """
    if len(series) < 2:
        return 1.0
    sims = [similarity(series[i - 1], series[i]) for i in range(1, len(series))]
    return float(sum(sims) / len(sims))
