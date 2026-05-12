"""Tests for HRR primitive operations (brain/library/vsa/hrr.py).

Exercised properties:
* bind/unbind roundtrip recovers a filler from a role-filler binding
* unbind uses conjugate correlation (not spectral division) — numerical stability
* project returns unit-norm vectors
* No NaN / Inf escape any primitive
* make_symbol is deterministic across processes (same seed → same vector)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from library.vsa.hrr import (
    HRRConfig,
    bind,
    make_symbol,
    project,
    similarity,
    superpose,
    unbind,
)


def _cfg(dim: int = 1024, seed: int = 0) -> HRRConfig:
    return HRRConfig(dim=dim, seed=seed)


def test_make_symbol_deterministic():
    cfg = _cfg()
    a1 = make_symbol("person", cfg)
    a2 = make_symbol("person", cfg)
    np.testing.assert_array_equal(a1, a2)


def test_make_symbol_different_names_distinct():
    cfg = _cfg()
    a = make_symbol("person", cfg)
    b = make_symbol("room", cfg)
    # Different names should be near-orthogonal at dim 1024.
    sim = similarity(a, b)
    assert abs(sim) < 0.15, f"unexpectedly high similarity {sim} between distinct symbols"


def test_make_symbol_different_seeds_distinct():
    a = make_symbol("person", _cfg(seed=0))
    b = make_symbol("person", _cfg(seed=1))
    assert not np.allclose(a, b), "seed change must produce a different vector"


def test_project_unit_norm():
    cfg = _cfg()
    v = make_symbol("x", cfg)
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-5, f"project should produce unit norm, got {norm}"


def test_project_rejects_wrong_shape():
    cfg = _cfg(dim=128)
    with pytest.raises(ValueError):
        project(np.zeros(64, dtype=np.float32), cfg)


def test_project_preserves_zero_vector():
    cfg = _cfg(dim=128)
    z = np.zeros(128, dtype=np.float32)
    out = project(z, cfg)
    assert np.all(out == 0.0)


def test_bind_unbind_roundtrip_single():
    cfg = _cfg()
    role = make_symbol("is_in", cfg)
    filler = make_symbol("room:office", cfg)
    bound = bind(role, filler, cfg)
    recovered = unbind(bound, role, cfg)
    sim_to_filler = similarity(recovered, filler)
    assert sim_to_filler > 0.95, f"bind/unbind roundtrip lost too much signal: {sim_to_filler}"


def test_bind_is_associative_like_for_unbind():
    """unbind(bind(a, b), b) ≈ a. This is the core HRR contract."""
    cfg = _cfg()
    for name in ("alpha", "beta", "gamma"):
        a = make_symbol(f"a:{name}", cfg)
        b = make_symbol(f"b:{name}", cfg)
        recovered = unbind(bind(a, b, cfg), b, cfg)
        assert similarity(recovered, a) > 0.95


def test_superpose_allows_multiple_role_recovery():
    cfg = _cfg()
    r1 = make_symbol("role:subject", cfg)
    f1 = make_symbol("filler:user", cfg)
    r2 = make_symbol("role:object", cfg)
    f2 = make_symbol("filler:office", cfg)
    bundle = superpose((bind(r1, f1, cfg), bind(r2, f2, cfg)), cfg)
    # Each role probe should pull out its filler above noise.
    assert similarity(unbind(bundle, r1, cfg), f1) > 0.6
    assert similarity(unbind(bundle, r2, cfg), f2) > 0.6
    # Cross-probes should be near zero.
    assert abs(similarity(unbind(bundle, r1, cfg), f2)) < 0.3
    assert abs(similarity(unbind(bundle, r2, cfg), f1)) < 0.3


def test_no_nan_or_inf_escapes():
    cfg = _cfg()
    a = make_symbol("a", cfg)
    b = make_symbol("b", cfg)
    outputs = [
        bind(a, b, cfg),
        unbind(bind(a, b, cfg), b, cfg),
        superpose([a, b], cfg),
        project(a, cfg),
    ]
    for out in outputs:
        assert np.all(np.isfinite(out)), "primitive produced NaN or Inf"


def test_unbind_uses_conjugate_not_division():
    """Regression: if unbind were implemented as spectral division, a near-zero
    spectral component in the key would blow up. With conjugate correlation,
    the recovery degrades gracefully instead.
    """
    cfg = _cfg(dim=512)
    a = make_symbol("x", cfg)
    key = make_symbol("k", cfg)
    # Surgically force one spectral component of key to be tiny.
    F = np.fft.fft(key)
    F[7] = F[7] * 1e-9
    weak_key = np.fft.ifft(F).real.astype(np.float32)
    weak_key = project(weak_key, cfg)
    # Still-finite recovery under the weak key proves we're not dividing.
    recovered = unbind(bind(a, weak_key, cfg), weak_key, cfg)
    assert np.all(np.isfinite(recovered))
    # With conjugate, we still expect positive correlation with a.
    assert similarity(recovered, a) > 0.2


def test_similarity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        similarity(np.zeros(10), np.zeros(11))


def test_hrrconfig_validates_dim_and_dtype():
    with pytest.raises(ValueError):
        HRRConfig(dim=0)
    with pytest.raises(ValueError):
        HRRConfig(dtype="float16")
