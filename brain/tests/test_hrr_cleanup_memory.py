"""Tests for CleanupMemory + metrics (brain/library/vsa/cleanup.py, metrics.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from library.vsa.cleanup import CleanupMemory
from library.vsa.hrr import HRRConfig, bind, project, similarity, unbind
from library.vsa.metrics import (
    cleanup_accuracy,
    cleanup_top3_accuracy,
    false_positive_rate,
    noise_tolerance,
    role_recovery_accuracy,
    superposition_capacity,
)
from library.vsa.symbols import SymbolDictionary


def _setup(n_fillers: int = 16, dim: int = 1024):
    cfg = HRRConfig(dim=dim, seed=0)
    d = SymbolDictionary(cfg)
    fillers = [(f"filler_{i}", d.entity(f"filler_{i}")) for i in range(n_fillers)]
    cleanup = CleanupMemory(cfg)
    for label, vec in fillers:
        cleanup.add(label, vec)
    return cfg, d, fillers, cleanup


def test_cleanup_recovers_clean_symbol_perfectly():
    cfg, _d, fillers, cleanup = _setup()
    for label, vec in fillers:
        got_label, score = cleanup.lookup(vec)
        assert got_label == label
        assert score > 0.95


def test_cleanup_topk_sorted_descending():
    _cfg, _d, fillers, cleanup = _setup(n_fillers=8)
    probe = fillers[0][1]
    top = cleanup.topk(probe, k=4)
    assert len(top) == 4
    scores = [s for _l, s in top]
    assert scores == sorted(scores, reverse=True)
    assert top[0][0] == fillers[0][0]


def test_cleanup_accuracy_on_clean_queries():
    cfg, _d, fillers, cleanup = _setup()
    queries = [(vec, label) for label, vec in fillers]
    acc = cleanup_accuracy(queries, cleanup)
    assert acc == 1.0
    top3 = cleanup_top3_accuracy(queries, cleanup)
    assert top3 == 1.0


def test_false_positive_rate_low_at_dim_1024():
    cfg, _d, fillers, cleanup = _setup(n_fillers=16, dim=1024)
    queries = [(vec, label) for label, vec in fillers]
    fp = false_positive_rate(queries, cleanup, threshold=0.5)
    assert fp < 0.05, f"FP rate unexpectedly high at dim 1024: {fp}"


def test_role_recovery_accuracy_bundle_of_8():
    cfg = HRRConfig(dim=1024, seed=0)
    d = SymbolDictionary(cfg)
    cleanup = CleanupMemory(cfg)
    pairs = []
    for i in range(8):
        r_label = f"role_{i}"
        f_label = f"filler_{i}"
        r_vec = d.role(r_label)
        f_vec = d.entity(f_label)
        pairs.append((r_label, r_vec, f_label, f_vec))
        cleanup.add(f_label, f_vec)
    acc = role_recovery_accuracy(pairs, cleanup, cfg)
    assert acc >= 0.75, f"role recovery of 8 bindings should be >= 0.75, got {acc}"


def test_superposition_capacity_curve():
    cfg = HRRConfig(dim=1024, seed=0)
    d = SymbolDictionary(cfg)
    cleanup = CleanupMemory(cfg)
    pairs = []
    for i in range(32):
        r_vec = d.role(f"role_{i}")
        f_vec = d.entity(f"filler_{i}")
        pairs.append((f"role_{i}", r_vec, f"filler_{i}", f_vec))
        cleanup.add(f"filler_{i}", f_vec)
    cap = superposition_capacity(pairs, cleanup, cfg, threshold=0.4)
    # At dim 1024 a strict cleanup at threshold 0.4 should handle several bindings.
    assert cap >= 4, f"capacity unexpectedly low: {cap}"


def test_noise_tolerance_degrades_monotonically():
    cfg = HRRConfig(dim=1024, seed=0)
    d = SymbolDictionary(cfg)
    cleanup = CleanupMemory(cfg)
    labels = [f"filler_{i}" for i in range(8)]
    vecs = [d.entity(l) for l in labels]
    for l, v in zip(labels, vecs):
        cleanup.add(l, v)
    results = noise_tolerance(
        clean_vec=vecs[0],
        cleanup=cleanup,
        expected_label=labels[0],
        cfg=cfg,
        noise_levels=(0.0, 0.1, 0.5),
        trials_per_level=32,
        rng=np.random.default_rng(42),
    )
    assert results[0.0] >= 0.9
    # Accuracy should not increase as noise grows.
    assert results[0.1] >= results[0.5] - 1e-9


def test_empty_cleanup_returns_none():
    cfg = HRRConfig(dim=64, seed=0)
    cleanup = CleanupMemory(cfg)
    label, score = cleanup.lookup(np.zeros(64, dtype=np.float32))
    assert label is None
    assert score == 0.0
