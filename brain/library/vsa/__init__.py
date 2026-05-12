"""Holographic Reduced Representations (HRR) / VSA substrate.

PRE-MATURE research lane: this module implements the dormant substrate only.
No authority over canonical memory, belief graph, policy, autonomy, Soul
Integrity, or LLM articulation. All live-loop hooks that consume these
primitives are gated behind the ``ENABLE_HRR_SHADOW=0`` environment flag
(default OFF, read once at boot by :class:`HRRRuntimeConfig`).

Backend is numpy FFT on CPU; torch remains a future GPU/autograd option.
"""

from library.vsa.hrr import (
    HRRConfig,
    bind,
    make_symbol,
    project,
    similarity,
    superpose,
    unbind,
)
from library.vsa.symbols import SymbolDictionary
from library.vsa.cleanup import CleanupMemory
from library.vsa.metrics import (
    cleanup_accuracy,
    cleanup_top3_accuracy,
    false_positive_rate,
    noise_tolerance,
    role_recovery_accuracy,
    similarity_drift,
    superposition_capacity,
)
from library.vsa.runtime_config import HRRRuntimeConfig
from library.vsa.status import (
    get_hrr_samples,
    get_hrr_status,
    register_recall_advisory_reader,
    register_recall_advisory_recent,
    register_simulation_shadow_reader,
    register_simulation_shadow_recent,
    register_spatial_scene_reader,
    register_spatial_scene_recent,
    register_world_shadow_reader,
    register_world_shadow_recent,
)

__all__ = [
    "HRRConfig",
    "HRRRuntimeConfig",
    "get_hrr_status",
    "get_hrr_samples",
    "register_world_shadow_reader",
    "register_simulation_shadow_reader",
    "register_recall_advisory_reader",
    "register_spatial_scene_reader",
    "register_world_shadow_recent",
    "register_simulation_shadow_recent",
    "register_recall_advisory_recent",
    "register_spatial_scene_recent",
    "SymbolDictionary",
    "CleanupMemory",
    "bind",
    "unbind",
    "superpose",
    "project",
    "similarity",
    "make_symbol",
    "cleanup_accuracy",
    "cleanup_top3_accuracy",
    "false_positive_rate",
    "superposition_capacity",
    "role_recovery_accuracy",
    "noise_tolerance",
    "similarity_drift",
]
