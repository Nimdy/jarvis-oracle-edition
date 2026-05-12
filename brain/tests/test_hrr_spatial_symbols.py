"""Tests for the P5 spatial-symbol vocabulary (brain/library/vsa/spatial_symbols.py).

These tests enforce four invariants:

1. Full canonical coverage of :data:`cognition.spatial_schema.SpatialRelationType`
   (including ``centered_in``).
2. Determinism and near-orthogonality at ``dim=1024``.
3. The direct ``make_symbol`` path and the ``SymbolDictionary`` path yield
   identical vectors for the same config. This is the "shared HRRConfig" rule.
4. Bind/unbind round-trip cleanliness ``>= 0.95`` on the relation vocabulary.
"""

from __future__ import annotations

import os
import sys
from typing import get_args

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from cognition.spatial_schema import SpatialRelationType
from library.vsa.hrr import HRRConfig, bind, project, similarity, unbind
from library.vsa.symbols import SymbolDictionary
from library.vsa.spatial_symbols import (
    CANONICAL_RELATION_SYMBOLS,
    CENTERED_IN,
    LEFT_OF,
    MENTAL_NAV_SYMBOLS,
    MOVE_FORWARD,
    NAMESPACE_AXIS,
    NAMESPACE_NAV,
    NAMESPACE_RELATION,
    NAMESPACE_STATE,
    RIGHT_OF,
    SPATIAL_AXIS_SYMBOLS,
    SPATIAL_RELATION_SYMBOLS,
    SPATIAL_STATE_SYMBOLS,
    axis_symbol,
    entity_symbol,
    nav_symbol,
    relation_string_to_symbol,
    relation_symbol,
    seed_all_symbols,
    state_symbol,
)


def _cfg(dim: int = 1024, seed: int = 0) -> HRRConfig:
    return HRRConfig(dim=dim, seed=seed)


# ---------------------------------------------------------------------------
# Vocabulary shape
# ---------------------------------------------------------------------------


def test_full_canonical_relation_coverage():
    """Every SpatialRelationType literal must appear in CANONICAL_RELATION_SYMBOLS."""
    canonical = set(get_args(SpatialRelationType))
    covered = set(CANONICAL_RELATION_SYMBOLS)
    missing = canonical - covered
    extra = covered - canonical
    assert not missing, f"canonical relations missing from P5 vocabulary: {sorted(missing)}"
    assert not extra, f"P5 canonical set has non-canonical entries: {sorted(extra)}"


def test_centered_in_included():
    assert CENTERED_IN == "centered_in"
    assert "centered_in" in CANONICAL_RELATION_SYMBOLS
    assert "centered_in" in SPATIAL_RELATION_SYMBOLS


def test_all_vocab_tuples_are_unique_strings():
    for vocab_name, vocab in (
        ("SPATIAL_RELATION_SYMBOLS", SPATIAL_RELATION_SYMBOLS),
        ("SPATIAL_AXIS_SYMBOLS", SPATIAL_AXIS_SYMBOLS),
        ("SPATIAL_STATE_SYMBOLS", SPATIAL_STATE_SYMBOLS),
        ("MENTAL_NAV_SYMBOLS", MENTAL_NAV_SYMBOLS),
    ):
        assert all(isinstance(x, str) and x for x in vocab), f"{vocab_name} has non-strings"
        assert len(set(vocab)) == len(vocab), f"{vocab_name} has duplicates"


# ---------------------------------------------------------------------------
# Determinism + near-orthogonality
# ---------------------------------------------------------------------------


def test_relation_string_to_symbol_is_deterministic():
    cfg = _cfg()
    v1 = relation_string_to_symbol(LEFT_OF, cfg)
    v2 = relation_string_to_symbol(LEFT_OF, cfg)
    np.testing.assert_array_equal(v1, v2)
    assert v1.shape == (cfg.dim,)
    assert v1.dtype == np.float32


def test_relation_string_to_symbol_rejects_empty():
    cfg = _cfg()
    with pytest.raises(ValueError):
        relation_string_to_symbol("", cfg)
    with pytest.raises(ValueError):
        relation_string_to_symbol(None, cfg)  # type: ignore[arg-type]


def test_all_relation_symbols_are_near_orthogonal_at_dim_1024():
    cfg = _cfg(dim=1024)
    vecs = [relation_string_to_symbol(r, cfg) for r in SPATIAL_RELATION_SYMBOLS]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s = similarity(vecs[i], vecs[j])
            assert abs(s) < 0.15, (
                f"{SPATIAL_RELATION_SYMBOLS[i]} vs {SPATIAL_RELATION_SYMBOLS[j]} "
                f"too similar: {s}"
            )


def test_left_of_and_right_of_are_distinct():
    cfg = _cfg()
    left = relation_string_to_symbol(LEFT_OF, cfg)
    right = relation_string_to_symbol(RIGHT_OF, cfg)
    assert not np.allclose(left, right)
    assert abs(similarity(left, right)) < 0.15


def test_different_seeds_produce_disjoint_vectors():
    v0 = relation_string_to_symbol(LEFT_OF, _cfg(seed=0))
    v1 = relation_string_to_symbol(LEFT_OF, _cfg(seed=1))
    assert not np.allclose(v0, v1)


# ---------------------------------------------------------------------------
# Shared-config rule: make_symbol path must equal SymbolDictionary path.
# ---------------------------------------------------------------------------


def test_make_symbol_path_matches_symbol_dictionary_path():
    """Guards the "shared HRRConfig / no independent seed or dim" rule.

    If these two paths diverge, spatial symbols become incompatible with
    the P4 world / simulation / recall rings which use the dictionary path.
    """
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    direct = relation_string_to_symbol(LEFT_OF, cfg)
    via_dict = relation_symbol(LEFT_OF, symbols)
    np.testing.assert_array_equal(direct, via_dict)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def test_seed_all_symbols_populates_full_vocab():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    seed_all_symbols(symbols)
    assert set(symbols.known(NAMESPACE_RELATION)) == set(SPATIAL_RELATION_SYMBOLS)
    assert set(symbols.known(NAMESPACE_AXIS)) == set(SPATIAL_AXIS_SYMBOLS)
    assert set(symbols.known(NAMESPACE_STATE)) == set(SPATIAL_STATE_SYMBOLS)
    assert set(symbols.known(NAMESPACE_NAV)) == set(MENTAL_NAV_SYMBOLS)


def test_seed_all_symbols_idempotent():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    seed_all_symbols(symbols)
    count_once = len(symbols)
    seed_all_symbols(symbols)
    assert len(symbols) == count_once


def test_entity_symbol_creates_disjoint_namespace():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    e = entity_symbol("cup_0", symbols)
    r = relation_symbol("left_of", symbols)
    # Same name (if any collision existed) would still produce distinct
    # vectors because namespaces differ.
    assert abs(similarity(e, r)) < 0.2


def test_axis_symbol_guards_valid_labels():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    for a in SPATIAL_AXIS_SYMBOLS:
        v = axis_symbol(a, symbols)
        assert v.shape == (cfg.dim,)
    with pytest.raises(ValueError):
        axis_symbol("w", symbols)


def test_state_and_nav_symbol_helpers():
    cfg = _cfg()
    symbols = SymbolDictionary(cfg)
    for s in SPATIAL_STATE_SYMBOLS:
        assert state_symbol(s, symbols).shape == (cfg.dim,)
    for a in MENTAL_NAV_SYMBOLS:
        assert nav_symbol(a, symbols).shape == (cfg.dim,)
    with pytest.raises(ValueError):
        state_symbol("", symbols)
    with pytest.raises(ValueError):
        nav_symbol("teleport", symbols)


# ---------------------------------------------------------------------------
# Bind / unbind round-trip on the relation vocabulary
# ---------------------------------------------------------------------------


def test_bind_unbind_roundtrip_on_relation_pair():
    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    seed_all_symbols(symbols)

    entity_b = entity_symbol("monitor_0", symbols)
    rel = relation_symbol(LEFT_OF, symbols)

    # Build (rel ⊛ entity_b), then probe with rel to recover entity_b.
    composite = bind(rel, entity_b, cfg)
    recovered = unbind(composite, rel, cfg)
    cos = similarity(project(recovered, cfg), project(entity_b, cfg))
    assert cos >= 0.95, f"bind/unbind roundtrip degraded: {cos}"


def test_bind_unbind_roundtrip_across_full_canonical_set():
    cfg = _cfg(dim=1024)
    symbols = SymbolDictionary(cfg)
    seed_all_symbols(symbols)

    entity_a = entity_symbol("cup_0", symbols)
    scores = []
    for r in CANONICAL_RELATION_SYMBOLS:
        rel = relation_symbol(r, symbols)
        composite = bind(rel, entity_a, cfg)
        recovered = unbind(composite, rel, cfg)
        scores.append(similarity(project(recovered, cfg), project(entity_a, cfg)))
    mean_cos = sum(scores) / len(scores)
    assert mean_cos >= 0.95, (
        f"round-trip cleanliness across canonical set too low: mean={mean_cos:.4f}, "
        f"scores={scores}"
    )


# ---------------------------------------------------------------------------
# No raw-vector leakage surface: module never writes anywhere on import
# ---------------------------------------------------------------------------


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    """Importing spatial_symbols must not touch the filesystem."""
    import importlib
    import library.vsa.spatial_symbols as mod
    # Force a fresh import in an isolated cwd and assert no files were created.
    monkeypatch.chdir(tmp_path)
    importlib.reload(mod)
    created = list(tmp_path.iterdir())
    assert created == [], f"import created unexpected files: {created}"


def test_move_forward_symbol_constant_is_exported():
    # Sanity check that Commit 8 symbols are reachable via public API.
    assert MOVE_FORWARD == "move_forward"
    assert MOVE_FORWARD in MENTAL_NAV_SYMBOLS
