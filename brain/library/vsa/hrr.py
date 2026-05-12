"""HRR primitive operations.

Reference backend is numpy FFT on CPU, float32 vectors. Circular convolution
is computed via FFT; unbind uses **conjugate correlation** (not spectral
division) so noise stays bounded and the inverse is numerically well-conditioned.

Design rules (non-negotiable for this substrate):

* Pure functions only. No side effects, no I/O, no logging of raw vectors.
* Deterministic: ``make_symbol(name, seed, dim)`` produces the same vector for
  the same ``(name, seed, dim)`` triple across processes / restarts.
* Fixed dim across a dictionary; callers MUST use the dim from ``HRRConfig``.
* Vectors are unit-norm after :func:`project`. ``bind`` / ``superpose`` do not
  normalize in-place; call :func:`project` explicitly if normalization is
  required.
* No vector is ever written to disk, no vector leaves this module's numeric
  boundary for policy / belief / memory / autonomy / LLM consumption.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HRRConfig:
    """Dimensionality + determinism knobs for the HRR substrate.

    ``dim`` must be a positive integer. ``seed`` scopes the symbol namespace;
    two ``HRRConfig`` instances with different seeds produce disjoint symbol
    spaces even for identical names.
    """

    dim: int = 1024
    seed: int = 0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"HRRConfig.dim must be positive, got {self.dim}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"HRRConfig.dtype must be float32 or float64, got {self.dtype}")


def _np_dtype(cfg: HRRConfig) -> np.dtype:
    return np.dtype(np.float32) if cfg.dtype == "float32" else np.dtype(np.float64)


def _symbol_seed(name: str, cfg_seed: int) -> int:
    """Hash ``(cfg_seed, name)`` to a deterministic 32-bit seed.

    Using SHA-256 gives us a stable, platform-independent integer regardless of
    the Python ``hash()`` randomization. We take the low 32 bits so numpy's
    ``default_rng`` accepts it on every platform.
    """
    h = hashlib.sha256(f"{cfg_seed}|{name}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def make_symbol(name: str, cfg: HRRConfig) -> np.ndarray:
    """Deterministic unit-norm **unitary** random vector for ``name`` under ``cfg``.

    Uses Plate's unitary-HRR construction: the FFT of the returned vector has
    unit magnitude at every frequency bin (random phase). This makes the
    bind/unbind roundtrip noise-free up to float precision, because
    ``|FFT(b)|^2 = 1`` everywhere and conjugate correlation recovers the
    original spectrum exactly.

    Two calls with the same ``(name, cfg.seed, cfg.dim)`` return identical
    vectors. Different names produce near-orthogonal vectors.
    """
    seed = _symbol_seed(name, cfg.seed)
    rng = np.random.default_rng(seed)
    n = cfg.dim
    # Random phases in [0, 2π) for one half of the spectrum; enforce conjugate
    # symmetry so the inverse FFT yields a real-valued vector.
    phases = np.zeros(n, dtype=np.float64)
    half = n // 2
    # bins 1..half-1 get random phases; bins half+1..n-1 mirror with negation
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=max(0, half - 1))
    if half - 1 > 0:
        phases[1:half] = random_phases
        phases[half + 1 : n] = -random_phases[::-1]
    # DC bin (0) and Nyquist (half, if n even) must be real → phase 0 or π.
    # Pick 0 deterministically to keep the construction simple.
    spectrum = np.exp(1j * phases)  # unit magnitude everywhere
    vec = np.fft.ifft(spectrum).real
    # Numerical floor correction: norm is exactly 1/sqrt(n) * sqrt(n) = 1 in
    # theory, but float error can drift it by ~1e-16. Re-normalize to be safe.
    out = vec.astype(_np_dtype(cfg), copy=False)
    return project(out, cfg)


def project(vec: np.ndarray, cfg: HRRConfig) -> np.ndarray:
    """Normalize ``vec`` to unit L2 norm, casting to ``cfg.dtype``.

    Zero vectors are returned as-is (still zero); callers must not feed random
    zeros into :func:`bind` or :func:`similarity`.
    """
    arr = np.asarray(vec, dtype=_np_dtype(cfg))
    if arr.shape != (cfg.dim,):
        raise ValueError(f"project expected shape ({cfg.dim},), got {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.copy()
    return (arr / norm).astype(_np_dtype(cfg), copy=False)


def bind(a: np.ndarray, b: np.ndarray, cfg: HRRConfig) -> np.ndarray:
    """Circular convolution of ``a`` and ``b`` (the HRR bind operator).

    Implemented via FFT: ``ifft(fft(a) * fft(b))``. Result is returned
    un-normalized; call :func:`project` if you need unit norm.
    """
    _check_shape(a, cfg, "bind.a")
    _check_shape(b, cfg, "bind.b")
    Fa = np.fft.fft(a)
    Fb = np.fft.fft(b)
    conv = np.fft.ifft(Fa * Fb).real
    return conv.astype(_np_dtype(cfg), copy=False)


def unbind(composite: np.ndarray, key: np.ndarray, cfg: HRRConfig) -> np.ndarray:
    """Inverse of :func:`bind` via **conjugate correlation**, not division.

    Given ``c = bind(a, b)``, ``unbind(c, b)`` approximately recovers ``a``.
    The conjugate formulation is numerically stable because it avoids
    division by near-zero spectral components. Noise is bounded and scales
    with 1/sqrt(dim) for independent random keys.

    Formula: ``ifft(fft(c) * conj(fft(key)))``.
    """
    _check_shape(composite, cfg, "unbind.composite")
    _check_shape(key, cfg, "unbind.key")
    Fc = np.fft.fft(composite)
    Fk = np.fft.fft(key)
    correlation = np.fft.ifft(Fc * np.conj(Fk)).real
    return correlation.astype(_np_dtype(cfg), copy=False)


def superpose(vectors: Iterable[np.ndarray], cfg: HRRConfig) -> np.ndarray:
    """Elementwise sum of an iterable of same-dim vectors.

    Returned un-normalized; projecting afterwards is usually desirable when
    stacking more than a handful of fact bindings.
    """
    acc = np.zeros(cfg.dim, dtype=_np_dtype(cfg))
    count = 0
    for v in vectors:
        _check_shape(v, cfg, "superpose")
        acc = acc + v.astype(_np_dtype(cfg), copy=False)
        count += 1
    if count == 0:
        return acc
    return acc


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Zero-vector arguments return 0.0."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"similarity shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    na = float(np.linalg.norm(a_arr))
    nb = float(np.linalg.norm(b_arr))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


def _check_shape(vec: np.ndarray, cfg: HRRConfig, ctx: str) -> None:
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"{ctx}: expected numpy.ndarray, got {type(vec).__name__}")
    if vec.shape != (cfg.dim,):
        raise ValueError(f"{ctx}: expected shape ({cfg.dim},), got {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{ctx}: vector contains NaN or Inf")
