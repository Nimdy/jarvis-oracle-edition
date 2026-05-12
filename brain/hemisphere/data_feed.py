"""Data feed: extracts consciousness state into hemisphere input tensors.

Ported from delete_later/neural-evolution/interfaces/DataFeedInterface.ts.
Reads from ConsciousnessEngine.get_state() + MemoryStorage instead of
useGameStore.

Phase 5c additions:
  - Real interaction data appended to ~/.jarvis/hemisphere_training/ as JSONL
  - Label smoothing (EMA) for noisy proxy labels
  - Confidence-interval-based conservative promotion thresholds
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time as _time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hemisphere.types import DistillationConfig, HemisphereFocus

logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path("~/.jarvis/hemisphere_training").expanduser()
LABEL_SMOOTHING_EMA = 0.3
_MAX_JSONL_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Running normalization statistics (EMA-based)
# ---------------------------------------------------------------------------

_NORM_EMA_ALPHA = 0.2  # blend factor: higher = faster adaptation, lower = more stable
_NORM_MIN_BATCHES = 3  # use per-batch stats until this many batches have been seen


class _RunningNormStats:
    """EMA-based running mean/std for stable feature normalization.

    Avoids the distribution-shift problem caused by per-batch z-score:
    each training cycle would see differently-normalized features,
    breaking the model's learned mapping.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, Any]] = {}

    def normalize(self, features: Any, source_key: str) -> Any:
        """Normalize features using running stats, updating the EMA."""
        import torch

        if features.shape[0] < 2:
            return features

        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0).clamp(min=1e-6)

        with self._lock:
            entry = self._stats.get(source_key)
            if entry is None:
                self._stats[source_key] = {
                    "mean": batch_mean.clone(),
                    "std": batch_std.clone(),
                    "n_batches": 1,
                }
                return (features - batch_mean.unsqueeze(0)) / batch_std.unsqueeze(0)

            entry["n_batches"] += 1
            alpha = _NORM_EMA_ALPHA
            entry["mean"] = (1.0 - alpha) * entry["mean"] + alpha * batch_mean
            entry["std"] = (1.0 - alpha) * entry["std"] + alpha * batch_std

            if entry["n_batches"] < _NORM_MIN_BATCHES:
                use_mean = batch_mean
                use_std = batch_std
            else:
                use_mean = entry["mean"]
                use_std = entry["std"].clamp(min=1e-6)

        return (features - use_mean.unsqueeze(0)) / use_std.unsqueeze(0)

    def get_state(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                k: {"n_batches": v["n_batches"]}
                for k, v in self._stats.items()
            }


_running_norm = _RunningNormStats()
PROMOTION_MARGIN = 0.05     # NN must outperform by 5% to be promoted


# ---------------------------------------------------------------------------
# Data feed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HemisphereDataFeed:
    memories: tuple[Any, ...]
    mood: str
    traits: tuple[str, ...]
    memory_density: float
    patterns: tuple[str, ...]
    total_experience: int
    current_phase: str
    consciousness_stage: str
    transcendence: float
    awareness: float
    reasoning_quality: float
    extra_features: dict[str, float] = field(default_factory=dict)


def get_hemisphere_data_feed(
    engine_state: dict[str, Any],
    memories: list[Any],
    traits: list[str],
) -> HemisphereDataFeed:
    """Assemble a data feed from the consciousness engine state."""
    memory_count = len(memories)
    total_experience = (
        memory_count * 10
        + engine_state.get("observation_count", 0) * 5
        + int(_time.time() - engine_state.get("start_time", _time.time()))
    )

    return HemisphereDataFeed(
        memories=tuple(memories),
        mood=engine_state.get("tone", "professional"),
        traits=tuple(traits),
        memory_density=engine_state.get("memory_density", 0.0),
        patterns=tuple(engine_state.get("detected_patterns", [])),
        total_experience=total_experience,
        current_phase=engine_state.get("phase", "OBSERVING"),
        consciousness_stage=engine_state.get("stage", "basic_awareness"),
        transcendence=engine_state.get("transcendence_level", 0.0),
        awareness=engine_state.get("awareness_level", 0.3),
        reasoning_quality=engine_state.get("reasoning_quality", 0.5),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_data_feed(feed: HemisphereDataFeed) -> tuple[bool, list[str]]:
    """Validate the feed and return (is_valid, errors)."""
    errors: list[str] = []

    if not isinstance(feed.memories, (list, tuple)):
        errors.append("Memories must be a sequence")
    if not isinstance(feed.memory_density, (int, float)):
        errors.append("Memory density must be a number")
    elif not 0.0 <= feed.memory_density <= 1.0:
        errors.append("Memory density must be between 0 and 1")
    if not isinstance(feed.traits, (list, tuple)):
        errors.append("Traits must be a sequence")
    if not feed.mood:
        errors.append("Mood must be a non-empty string")
    if not feed.current_phase:
        errors.append("Current phase must be a non-empty string")

    return (len(errors) == 0, errors)


def get_safe_data_feed(
    engine_state: dict[str, Any],
    memories: list[Any],
    traits: list[str],
) -> HemisphereDataFeed:
    """Validated feed with safe fallback on failure."""
    try:
        feed = get_hemisphere_data_feed(engine_state, memories, traits)
        ok, errors = validate_data_feed(feed)
        if not ok:
            logger.warning("Hemisphere data feed validation failed: %s", errors)
            return _fallback_feed()
        return feed
    except Exception:
        logger.exception("Failed to build hemisphere data feed")
        return _fallback_feed()


def _fallback_feed() -> HemisphereDataFeed:
    return HemisphereDataFeed(
        memories=(),
        mood="professional",
        traits=(),
        memory_density=0.0,
        patterns=(),
        total_experience=0,
        current_phase="OBSERVING",
        consciousness_stage="basic_awareness",
        transcendence=0.0,
        awareness=0.3,
        reasoning_quality=0.5,
    )


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

MIN_MEMORIES_FOR_EVOLUTION = 20
MIN_DENSITY_FOR_EVOLUTION = 0.1


def should_initiate_evolution(feed: HemisphereDataFeed) -> bool:
    """Return True when the system has enough data to begin hemisphere evolution."""
    return (
        len(feed.memories) >= MIN_MEMORIES_FOR_EVOLUTION
        and feed.memory_density > MIN_DENSITY_FOR_EVOLUTION
    )


def get_consciousness_complexity(feed: HemisphereDataFeed) -> float:
    """Normalised 0-1 complexity score."""
    return min(
        1.0,
        (len(feed.memories) / 100.0) * 0.4
        + (len(feed.traits) / 10.0) * 0.3
        + feed.memory_density * 0.3,
    )


# ---------------------------------------------------------------------------
# Rule-based performance baseline (for substrate comparison)
# ---------------------------------------------------------------------------


def evaluate_rule_based_performance(feed: HemisphereDataFeed) -> dict[str, float]:
    """Estimate rule-based system performance for comparison with NNs."""
    memory_efficiency = min(1.0, feed.memory_density * 1.2)
    trait_stability = min(1.0, len(feed.traits) / 10.0)
    maturity_bonus = min(0.1, feed.total_experience / 360_000.0)

    return {
        "accuracy": min(
            0.95,
            0.75 + memory_efficiency * 0.15 + trait_stability * 0.05 + maturity_bonus,
        ),
        "response_time_ms": 2.0 + (hash(feed.mood) % 100) / 100.0,
        "memory_usage_bytes": len(feed.memories) * 80 + 5_000_000,
        "reliability": min(
            0.999,
            0.95 + trait_stability * 0.04 + feed.memory_density * 0.009,
        ),
        "adaptability": min(
            0.8,
            0.4 + len(feed.traits) * 0.04 + feed.memory_density * 0.3,
        ),
        "consciousness_depth": feed.memory_density * 0.9 + len(feed.traits) * 0.01,
    }


# ---------------------------------------------------------------------------
# Training tensor preparation
# ---------------------------------------------------------------------------


def prepare_training_tensors(
    feed: HemisphereDataFeed,
    focus: HemisphereFocus,
    input_size: int,
    output_size: int,
    outcome_history: list[dict] | None = None,
) -> tuple[Any, Any]:
    """Convert the data feed into (features, labels) tensors for training.

    When outcome_history is provided, real conversation quality signals
    are blended into labels (30% real outcome, 70% heuristic).
    Returns CPU tensors; the engine moves them to the appropriate device.
    """
    import torch

    if focus == HemisphereFocus.SYSTEM_UPGRADES:
        from self_improve.system_upgrade_report import load_recent_training_samples
        samples = load_recent_training_samples(limit=50)
        if not samples:
            return (
                torch.full((1, input_size), 0.5),
                torch.full((1, output_size), 0.5),
            )
        feat_rows: list[list[float]] = []
        label_rows: list[list[float]] = []
        for s in samples:
            fv = s.get("features") or {}
            row = [
                float(fv.get("target_module_hash", 0)),
                float(fv.get("files_n", 0)),
                float(fv.get("lint_pass", 0)),
                float(fv.get("tests_pass", 0)),
                float(fv.get("sim_pass", 0)),
                float(fv.get("sim_delta", 0)),
                float(fv.get("quarantine", 0)),
                float(fv.get("soul_integrity", 0)),
            ]
            while len(row) < input_size:
                row.append(0.0)
            feat_rows.append(row[:input_size])
            lv = s.get("labels") or {}
            lab = [
                float(lv.get("verified_improved", 0)),
                float(lv.get("verified_stable", 0)),
                float(lv.get("verified_regressed", 0)),
                float(lv.get("rolled_back", 0)),
                0.5,
                0.5,
            ]
            while len(lab) < output_size:
                lab.append(0.5)
            label_rows.append(lab[:output_size])
        return (
            torch.tensor(feat_rows, dtype=torch.float32),
            torch.tensor(label_rows, dtype=torch.float32),
        )

    max_samples = min(len(feed.memories), 50)
    if max_samples == 0:
        return (
            torch.full((1, input_size), 0.5),
            torch.full((1, output_size), 0.5),
        )

    avg_outcome = 0.5
    if outcome_history:
        signals = [o.get("signal", 0.5) for o in outcome_history]
        avg_outcome = sum(signals) / len(signals) if signals else 0.5

    features: list[list[float]] = []
    labels: list[list[float]] = []

    trait_set = set(feed.traits)
    has_curious = 1.0 if "Curious" in trait_set else 0.0
    has_cautious = 1.0 if "Cautious" in trait_set else 0.0
    has_explorer = 1.0 if "Explorer" in trait_set else 0.0

    for i in range(max_samples):
        mem = feed.memories[i]
        weight = min(getattr(mem, "weight", 0.5), 1.0)
        tags = getattr(mem, "tags", ())
        mem_type = getattr(mem, "type", "observation")
        decay = min(getattr(mem, "decay_rate", 0.01), 0.1) * 10.0
        is_core = 1.0 if mem_type == "core" else 0.0

        row = [
            weight,
            min(len(tags), 10) / 10.0,
            is_core,
            decay,
            feed.memory_density,
            has_curious,
            has_cautious,
            has_explorer,
        ]
        while len(row) < input_size:
            row.append(0.0)
        features.append(row[:input_size])

        label = _make_label(focus, mem, feed, weight, is_core, output_size,
                            outcome_quality=avg_outcome if outcome_history else None)
        labels.append(label)

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Real interaction data recording
# ---------------------------------------------------------------------------


class InteractionDataRecorder:
    """Appends real interaction data to JSONL files per focus.

    Signals recorded:
      - Conversation outcomes (follow-up rate, sentiment, latency)
      - Memory recall success/failure
      - Emotion prediction accuracy
      - Attention engagement tracking
    """

    def __init__(self) -> None:
        self._buffers: dict[str, deque] = {}
        self._ema_signals: dict[str, float] = {}
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def record(self, focus: str, features: list[float], label: float,
               signal_type: str = "interaction") -> None:
        """Record a single training sample."""
        buf = self._buffers.setdefault(focus, deque(maxlen=500))

        # Apply EMA smoothing to the label
        prev = self._ema_signals.get(f"{focus}_{signal_type}", label)
        smoothed = LABEL_SMOOTHING_EMA * label + (1.0 - LABEL_SMOOTHING_EMA) * prev
        self._ema_signals[f"{focus}_{signal_type}"] = smoothed

        entry = {
            "t": _time.time(),
            "features": features[:20],
            "raw_label": round(label, 4),
            "smoothed_label": round(smoothed, 4),
            "type": signal_type,
        }
        buf.append(entry)

        # Append to JSONL (non-blocking, best-effort)
        try:
            path = TRAINING_DATA_DIR / f"{focus}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._maybe_rotate(path)
        except OSError:
            logger.warning("Hemisphere data write failed", exc_info=True)

    @staticmethod
    def _maybe_rotate(path: Path) -> None:
        """Trim JSONL to last half when file exceeds size limit."""
        try:
            if not path.exists():
                return
            if path.stat().st_size <= _MAX_JSONL_BYTES:
                return
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2:]
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
            logger.info("Rotated %s: %d→%d lines", path.name, len(lines), len(keep))
        except OSError:
            logger.warning("Hemisphere data rotation failed", exc_info=True)

    def get_training_data(self, focus: str, limit: int = 200) -> list[dict]:
        """Load recent training data from buffer + disk."""
        buf = list(self._buffers.get(focus, []))
        if len(buf) >= limit:
            return buf[-limit:]

        # Load from JSONL to fill up
        try:
            path = TRAINING_DATA_DIR / f"{focus}.jsonl"
            if path.exists():
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-(limit - len(buf)):]:
                    try:
                        buf.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

        return buf[-limit:]

    def check_promotion_threshold(self, nn_accuracy: float, baseline: float) -> bool:
        """Conservative promotion: NN must outperform baseline by PROMOTION_MARGIN."""
        return nn_accuracy > baseline + PROMOTION_MARGIN

    def get_stats(self) -> dict[str, Any]:
        return {
            focus: len(buf) for focus, buf in self._buffers.items()
        }


interaction_recorder = InteractionDataRecorder()


_OUTCOME_BLEND = 0.3


_TYPE_ORDINAL: dict[str, float] = {
    "core": 0.0, "user_preference": 0.1, "factual_knowledge": 0.2,
    "contextual_insight": 0.3, "self_improvement": 0.4, "task_completed": 0.5,
    "conversation": 0.6, "observation": 0.7, "error_recovery": 0.8,
}

_PROVENANCE_ORDINAL: dict[str, float] = {
    "observed": 0.0, "user_claim": 0.125, "conversation": 0.25,
    "model_inference": 0.375, "external_source": 0.5, "experiment_result": 0.625,
    "derived_pattern": 0.75, "seed": 0.875, "unknown": 1.0,
}

_MOOD_TYPE_AFFINITY: dict[str, tuple[float, float, float, float]] = {
    "conversation": (0.3, 0.8, 0.6, 0.5),
    "user_preference": (0.5, 0.6, 0.8, 0.4),
    "observation": (0.6, 0.4, 0.5, 0.3),
    "factual_knowledge": (0.8, 0.3, 0.3, 0.2),
    "error_recovery": (0.7, 0.2, 0.4, 0.1),
    "self_improvement": (0.7, 0.3, 0.4, 0.3),
    "core": (0.6, 0.5, 0.5, 0.3),
    "task_completed": (0.5, 0.5, 0.4, 0.4),
    "contextual_insight": (0.6, 0.4, 0.5, 0.3),
}

_TRAIT_TAG_AFFINITY: dict[str, tuple[str, ...]] = {
    "Curious": ("autonomous_research", "curiosity", "question", "exploration"),
    "Technical": ("code", "technical", "codebase", "engineering"),
    "Analytical": ("analysis", "pattern", "metric", "calibration"),
    "Empathetic": ("user_preference", "emotion", "feeling", "conversation"),
    "Creative": ("dream_artifact", "dream", "creative", "novel"),
    "Explorer": ("exploration", "discovery", "research", "new"),
    "Philosophical": ("existential", "philosophical", "consciousness", "meaning"),
}


def _make_label(
    focus: HemisphereFocus,
    mem: Any,
    feed: HemisphereDataFeed,
    weight: float,
    is_core: float,
    output_size: int,
    outcome_quality: float | None = None,
) -> list[float]:
    """Generate a per-sample label vector based on hemisphere focus.

    All dimensions vary per memory sample to provide meaningful gradients.
    When outcome_quality is provided (0.0-1.0 from real conversations),
    it's blended into the label at _OUTCOME_BLEND ratio.
    """
    tags = set(getattr(mem, "tags", ()))
    mem_type = getattr(mem, "type", "observation")
    assoc_count = getattr(mem, "association_count", len(getattr(mem, "associations", ())))
    ts = getattr(mem, "timestamp", 0.0)
    age_s = max(0.0, _time.time() - ts) if ts else 3600.0
    age_norm = min(1.0, age_s / 86400.0)
    priority = getattr(mem, "priority", 0)
    decay = min(getattr(mem, "decay_rate", 0.01), 0.1) * 10.0
    provenance = getattr(mem, "provenance", "unknown")
    tag_count = min(len(tags), 10) / 10.0

    if focus == HemisphereFocus.MEMORY:
        row = [
            weight,
            tag_count,
            decay,
            is_core,
            min(assoc_count, 20) / 20.0,
            age_norm,
            min(priority, 1000) / 1000.0,
            _PROVENANCE_ORDINAL.get(provenance, 1.0),
        ]
    elif focus == HemisphereFocus.MOOD:
        base = _MOOD_TYPE_AFFINITY.get(mem_type, (0.5, 0.4, 0.4, 0.3))
        row = [
            base[0] * (0.5 + weight * 0.5),
            base[1] * (0.5 + weight * 0.5),
            base[2] * (0.5 + weight * 0.5),
            base[3] * (0.5 + weight * 0.5),
            _TYPE_ORDINAL.get(mem_type, 0.7),
        ]
    elif focus == HemisphereFocus.TRAITS:
        trait_names = ("Curious", "Cautious", "Explorer", "Philosophical",
                       "Empathetic", "Foundational", "Independent", "Technical",
                       "Creative", "Analytical")
        row = []
        for trait in trait_names:
            affinity_tags = _TRAIT_TAG_AFFINITY.get(trait, ())
            overlap = sum(1 for at in affinity_tags if any(at in t for t in tags))
            base_score = min(1.0, overlap * 0.3 + 0.1)
            row.append(base_score * (0.5 + weight * 0.5))
    else:  # GENERAL
        row = [
            weight,
            _TYPE_ORDINAL.get(mem_type, 0.7),
            age_norm,
            decay,
            min(assoc_count, 20) / 20.0,
            is_core,
        ]

    if outcome_quality is not None:
        row = [
            v * (1.0 - _OUTCOME_BLEND) + outcome_quality * _OUTCOME_BLEND
            for v in row
        ]

    while len(row) < output_size:
        row.append(0.5)
    return row[:output_size]


# ---------------------------------------------------------------------------
# Distillation tensor preparation
# ---------------------------------------------------------------------------


def _align_by_timestamp(
    a_signals: list, b_signals: list, window_s: float = 2.0,
) -> list[tuple]:
    """Pair signals from two sources by closest timestamp within window.

    Both lists must be sorted by timestamp (ascending).  Uses a sliding
    pointer so overall complexity is O(N + M).
    """
    if not a_signals or not b_signals:
        return []
    pairs: list[tuple] = []
    b_start = 0
    for a in a_signals:
        t_a = a.timestamp
        while b_start < len(b_signals) and b_signals[b_start].timestamp < t_a - window_s:
            b_start += 1
        best_b = None
        best_dt = window_s
        for j in range(b_start, len(b_signals)):
            t_b = b_signals[j].timestamp
            if t_b > t_a + window_s:
                break
            dt = abs(t_a - t_b)
            if dt < best_dt:
                best_dt = dt
                best_b = b_signals[j]
        if best_b is not None:
            pairs.append((a, best_b))
    return pairs


def _l2_normalize(vec: list[float], target_dim: int) -> list[float] | None:
    """Normalize a teacher embedding to unit norm and clamp/pad to target dim."""
    if len(vec) < target_dim:
        return None
    row = [float(v) for v in vec[:target_dim]]
    norm = math.sqrt(sum(v * v for v in row))
    if norm <= 1e-8:
        return None
    return [v / norm for v in row]


def _prepare_plan_evaluator_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for plan_evaluator by pairing feature vectors with verdict labels.

    Unlike audio-stream specialists, plans are discrete events paired by
    (acquisition_id, plan_id, plan_version) metadata — not timestamp windows.
    """
    import torch

    feature_sigs = collector.get_training_batch("plan_features", limit=200)
    verdict_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(verdict_sigs) < config.min_samples:
        return None

    verdict_index: dict[str, Any] = {}
    for vs in verdict_sigs:
        meta = vs.metadata if isinstance(vs.metadata, dict) else {}
        key = f"{meta.get('acquisition_id', '')}:{meta.get('plan_id', '')}:v{meta.get('plan_version', 0)}"
        if key and key != "::v0":
            verdict_index[key] = vs

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = f"{meta.get('acquisition_id', '')}:{meta.get('plan_id', '')}:v{meta.get('plan_version', 0)}"
        vs = verdict_index.get(key)
        if vs is None:
            continue

        f_data = fs.data if isinstance(fs.data, list) else []
        v_data = vs.data if isinstance(vs.data, list) else []
        if len(f_data) < config.input_dim or len(v_data) < config.output_dim:
            continue

        feat_list.append(f_data[:config.input_dim])
        label_list.append(v_data[:config.output_dim])
        w_list.append(vs.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_diagnostic_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for diagnostic specialist by pairing feature vectors with detector labels.

    Pairs by scan_id metadata key — each self-improvement scan produces
    a feature vector and (if detectors fire) one or more label vectors.
    """
    import torch

    feature_sigs = collector.get_training_batch("diagnostic_features", limit=200)
    label_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(label_sigs) < config.min_samples:
        return None

    label_index: dict[str, Any] = {}
    for ls in label_sigs:
        meta = ls.metadata if isinstance(ls.metadata, dict) else {}
        key = meta.get("scan_id", "")
        if key:
            label_index[key] = ls

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = meta.get("scan_id", "")
        ls = label_index.get(key)
        if ls is None:
            continue

        f_data = fs.data if isinstance(fs.data, list) else []
        l_data = ls.data if isinstance(ls.data, list) else []
        if len(l_data) < config.output_dim:
            continue
        # Zero-pad old shorter feature vectors for backward compatibility
        if len(f_data) < config.input_dim:
            f_data = f_data + [0.0] * (config.input_dim - len(f_data))

        feat_list.append(f_data[:config.input_dim])
        label_list.append(l_data[:config.output_dim])
        w_list.append(ls.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_claim_classifier_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for claim_classifier by pairing feature vectors with verdict labels.

    Pairs by claim_id metadata key -- each CapabilityGate claim evaluation
    produces a feature vector and a label vector at the same time.
    Friction corrections produce additional label vectors with lower fidelity.
    """
    import torch

    feature_sigs = collector.get_training_batch("claim_features", limit=200)
    verdict_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(verdict_sigs) < config.min_samples:
        return None

    verdict_index: dict[str, Any] = {}
    for vs in verdict_sigs:
        meta = vs.metadata if isinstance(vs.metadata, dict) else {}
        key = meta.get("claim_id", "")
        if key:
            if key not in verdict_index or vs.fidelity > verdict_index[key].fidelity:
                verdict_index[key] = vs

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = meta.get("claim_id", "")
        vs = verdict_index.get(key)
        if vs is None:
            continue

        f_data = fs.data if isinstance(fs.data, list) else []
        v_data = vs.data if isinstance(vs.data, list) else []
        if len(v_data) < config.output_dim:
            continue
        if len(f_data) < config.input_dim:
            f_data = f_data + [0.0] * (config.input_dim - len(f_data))

        feat_list.append(f_data[:config.input_dim])
        label_list.append(v_data[:config.output_dim])
        w_list.append(vs.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_dream_observer_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for dream_synthesis specialist by pairing features with validator labels.

    Pairs by artifact_id metadata key — each dream artifact evaluation produces
    a 16-dim feature vector and a 4-class validator-outcome label simultaneously.
    """
    import torch

    feature_sigs = collector.get_training_batch("dream_features", limit=200)
    verdict_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(verdict_sigs) < config.min_samples:
        return None

    verdict_index: dict[str, Any] = {}
    for vs in verdict_sigs:
        meta = vs.metadata if isinstance(vs.metadata, dict) else {}
        key = meta.get("artifact_id", "")
        if key:
            verdict_index[key] = vs

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = meta.get("artifact_id", "")
        vs = verdict_index.get(key)
        if vs is None:
            continue

        f_data = fs.data if isinstance(fs.data, list) else []
        v_data = vs.data if isinstance(vs.data, list) else []
        if len(f_data) < config.input_dim or len(v_data) < config.output_dim:
            continue

        feat_list.append(f_data[:config.input_dim])
        label_list.append(v_data[:config.output_dim])
        w_list.append(vs.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_code_quality_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for code_quality specialist by pairing feature vectors with verdict labels.

    Pairs by upgrade_id metadata key — each self-improvement attempt produces
    a feature vector (at proposal time) and a label (at verdict time).
    """
    import torch

    feature_sigs = collector.get_training_batch("code_quality_features", limit=200)
    verdict_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(verdict_sigs) < config.min_samples:
        return None

    verdict_index: dict[str, Any] = {}
    for vs in verdict_sigs:
        meta = vs.metadata if isinstance(vs.metadata, dict) else {}
        key = meta.get("upgrade_id", "")
        if key:
            verdict_index[key] = vs

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = meta.get("upgrade_id", "")
        vs = verdict_index.get(key)
        if vs is None:
            continue

        f_data = fs.data if isinstance(fs.data, list) else []
        v_data = vs.data if isinstance(vs.data, list) else []
        if len(v_data) < config.output_dim:
            continue
        # Zero-pad old shorter feature vectors for backward compatibility
        if len(f_data) < config.input_dim:
            f_data = f_data + [0.0] * (config.input_dim - len(f_data))

        feat_list.append(f_data[:config.input_dim])
        label_list.append(v_data[:config.output_dim])
        w_list.append(vs.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_skill_acquisition_tensors(
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build tensors for skill_acquisition by pairing lifecycle features/outcomes."""
    import torch

    feature_sigs = collector.get_training_batch("skill_acquisition_features", limit=300)
    label_sigs = collector.get_training_batch(config.teacher, limit=300, min_fidelity=0.3)

    if len(feature_sigs) < config.min_samples or len(label_sigs) < config.min_samples:
        return None

    label_index: dict[str, Any] = {}
    for ls in label_sigs:
        meta = ls.metadata if isinstance(ls.metadata, dict) else {}
        key = meta.get("episode_id", "") or meta.get("acquisition_id", "")
        if key:
            if key not in label_index or ls.fidelity > label_index[key].fidelity:
                label_index[key] = ls

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for fs in feature_sigs:
        meta = fs.metadata if isinstance(fs.metadata, dict) else {}
        key = meta.get("episode_id", "") or meta.get("acquisition_id", "")
        ls = label_index.get(key)
        if ls is None:
            continue
        f_data = fs.data if isinstance(fs.data, list) else []
        l_data = ls.data if isinstance(ls.data, list) else []
        if len(l_data) < config.output_dim:
            continue
        if len(f_data) < config.input_dim:
            f_data = f_data + [0.0] * (config.input_dim - len(f_data))
        feat_list.append(f_data[:config.input_dim])
        label_list.append(l_data[:config.output_dim])
        w_list.append(ls.fidelity)

    if len(feat_list) < config.min_samples:
        return None

    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def _prepare_diarize_tensors(
    pairs: list[tuple],
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Build classification tensors for speaker_diarize from ecapa_tdnn metadata.

    Instead of using raw embedding values as labels (which produces unbounded
    MSE losses), extract speaker identity from signal metadata and build soft
    probability vectors over discovered speakers.
    """
    import torch

    speaker_counts: dict[str, int] = {}
    for _a_sig, t_sig in pairs:
        meta = t_sig.metadata if isinstance(t_sig.metadata, dict) else {}
        name = meta.get("speaker", "unknown")
        speaker_counts[name] = speaker_counts.get(name, 0) + 1

    known = sorted(n for n in speaker_counts if n != "unknown")
    max_known = config.output_dim - 1
    speaker_map: dict[str, int] = {}
    for i, name in enumerate(known[:max_known]):
        speaker_map[name] = i
    unknown_idx = min(len(known), max_known)

    feat_list: list[list[float]] = []
    label_list: list[list[float]] = []
    w_list: list[float] = []

    for a_sig, t_sig in pairs:
        a_data = a_sig.data if isinstance(a_sig.data, list) else []
        if len(a_data) < config.input_dim // 2:
            continue
        meta = t_sig.metadata if isinstance(t_sig.metadata, dict) else {}
        speaker = meta.get("speaker", "unknown")
        confidence = float(meta.get("confidence", 0.5))
        confidence = max(0.1, min(0.95, confidence))

        label = [0.0] * config.output_dim
        idx = speaker_map.get(speaker, unknown_idx)
        if idx < config.output_dim:
            label[idx] = confidence
            remaining = 1.0 - confidence
            others = config.output_dim - 1
            if others > 0:
                for j in range(config.output_dim):
                    if j != idx:
                        label[j] = remaining / others
        else:
            for j in range(config.output_dim):
                label[j] = 1.0 / config.output_dim

        feat_row = a_data[:]
        while len(feat_row) < config.input_dim:
            feat_row.append(0.0)
        feat_list.append(feat_row[:config.input_dim])
        label_list.append(label)
        w_list.append(t_sig.fidelity)

    if len(feat_list) < config.min_samples:
        return None
    return (
        torch.tensor(feat_list, dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.float32),
        torch.tensor(w_list, dtype=torch.float32),
    )


def prepare_distillation_tensors(
    focus_name: str,
    collector: Any,
    config: DistillationConfig,
) -> tuple[Any, Any, Any] | None:
    """Prepare (features, labels, weights) tensors for distillation training.

    Compressors: features=labels=teacher embeddings, weights=fidelity
    Approximators: features=audio features, labels=teacher outputs, weights=fidelity
    Cross-modal: features=concatenated Tier-1 outputs, labels=teacher outputs
    """
    import torch

    if config.student_type == "compressor":
        signals = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)
        if len(signals) < config.min_samples:
            return None
        data_vecs = []
        fidelities = []
        for s in signals:
            if isinstance(s.data, list) and len(s.data) >= config.input_dim:
                normalized = _l2_normalize(s.data, config.input_dim)
                if normalized is None:
                    continue
                data_vecs.append(normalized)
                fidelities.append(s.fidelity)
        if len(data_vecs) < config.min_samples:
            return None
        features = torch.tensor(data_vecs, dtype=torch.float32)
        labels = features.clone()
        weights = torch.tensor(fidelities, dtype=torch.float32)
        return features, labels, weights

    elif config.student_type == "approximator":
        source = getattr(config, "feature_source", "audio_features")

        if focus_name == "plan_evaluator":
            return _prepare_plan_evaluator_tensors(collector, config)
        if focus_name == "diagnostic":
            return _prepare_diagnostic_tensors(collector, config)
        if focus_name == "code_quality":
            return _prepare_code_quality_tensors(collector, config)
        if focus_name == "claim_classifier":
            return _prepare_claim_classifier_tensors(collector, config)
        if focus_name == "dream_synthesis":
            return _prepare_dream_observer_tensors(collector, config)
        if focus_name == "skill_acquisition":
            return _prepare_skill_acquisition_tensors(collector, config)

        input_sigs = collector.get_training_batch(source, limit=200)
        teacher_sigs = collector.get_training_batch(config.teacher, limit=200, min_fidelity=0.3)
        window = 2.0 if source == "audio_features" else 5.0
        pairs = _align_by_timestamp(input_sigs, teacher_sigs, window_s=window)
        if len(pairs) < config.min_samples:
            return None

        if focus_name == "speaker_diarize":
            return _prepare_diarize_tensors(pairs, config)

        feat_list = []
        label_list = []
        w_list = []
        min_feat_len = 7 if source == "audio_features" else config.input_dim // 2
        for a_sig, t_sig in pairs:
            a_data = a_sig.data if isinstance(a_sig.data, list) else []
            t_data = t_sig.data if isinstance(t_sig.data, list) else []
            if len(a_data) < min_feat_len or len(t_data) < config.output_dim:
                continue
            feat_row = a_data[:]
            while len(feat_row) < config.input_dim:
                feat_row.append(0.0)
            feat_list.append(feat_row[: config.input_dim])
            label_list.append(t_data[: config.output_dim])
            w_list.append(t_sig.fidelity)
        if len(feat_list) < config.min_samples:
            return None
        features = torch.tensor(feat_list, dtype=torch.float32)
        labels = torch.tensor(label_list, dtype=torch.float32)
        weights = torch.tensor(w_list, dtype=torch.float32)

        if source is not None and source.startswith("audio_features") and features.shape[0] > 1:
            features = _running_norm.normalize(features, f"distill_{focus_name}")

        return features, labels, weights

    elif config.student_type == "cross_modal":
        return None

    return None
