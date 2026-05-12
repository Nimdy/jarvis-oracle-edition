"""HRR Stage 0 synthetic capacity + cleanup exercise.

Produces a capacity curve JSON suitable for the ``hrr_primitive_library`` and
``hrr_truth_boundary`` validation-pack checks. No live-brain state is touched:
no memory write, no belief edge, no identity mutation, no TTS emission, no
event bus traffic. This script imports only from :mod:`library.vsa`.

Run from the brain directory::

    PYTHONPATH=. python synthetic/hrr_exercise.py \\
        --dim 1024 --facts 1,2,4,8,16,32 --noise 0.0,0.05,0.10 \\
        --out ../docs/validation_reports/evidence/hrr_stage0.json

The output JSON schema is stable and consumed by
:mod:`jarvis_eval.validation_pack` — do not reshape without updating both.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Make ``library.vsa`` importable whether the caller runs from brain/ or the
# repo root via PYTHONPATH=brain.
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

from library.vsa.cleanup import CleanupMemory  # noqa: E402
from library.vsa.hrr import HRRConfig, bind, similarity, superpose, unbind  # noqa: E402
from library.vsa.metrics import (  # noqa: E402
    cleanup_accuracy,
    false_positive_rate,
    superposition_capacity,
)
from library.vsa.symbols import SymbolDictionary  # noqa: E402

SCHEMA_VERSION = 1

ACCURACY_THRESHOLD_AT_8 = 0.90
ACCURACY_THRESHOLD_AT_16 = 0.75
FP_THRESHOLD = 0.05
FRESHNESS_SECONDS = 30 * 24 * 60 * 60  # 30 days


def _build_dictionary(cfg: HRRConfig, n_roles: int, n_fillers: int) -> Tuple[
    SymbolDictionary, List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]
]:
    d = SymbolDictionary(cfg)
    roles = [(f"role_{i}", d.role(f"role_{i}")) for i in range(n_roles)]
    fillers = [(f"filler_{i}", d.entity(f"filler_{i}")) for i in range(n_fillers)]
    return d, roles, fillers


def _run_at_fact_count(
    cfg: HRRConfig,
    n_facts: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Encode ``n_facts`` bindings, run cleanup under Gaussian noise ``sigma``."""
    d, roles, fillers = _build_dictionary(cfg, n_roles=n_facts, n_fillers=n_facts)
    cleanup = CleanupMemory(cfg)
    for label, vec in fillers:
        cleanup.add(label, vec)

    bundle = superpose(
        (bind(roles[i][1], fillers[i][1], cfg) for i in range(n_facts)),
        cfg,
    )

    hits = 0
    similarities: List[float] = []
    queries: List[Tuple[np.ndarray, str]] = []
    for i in range(n_facts):
        role_vec = roles[i][1]
        expected_label = fillers[i][0]
        recovered = unbind(bundle, role_vec, cfg)
        if noise_sigma > 0.0:
            noise = rng.standard_normal(cfg.dim).astype(recovered.dtype) * noise_sigma
            recovered = recovered + noise
        queries.append((recovered, expected_label))
        top = cleanup.topk(recovered, k=1)
        if top and top[0][0] == expected_label:
            hits += 1
        if top:
            similarities.append(float(top[0][1]))

    accuracy = hits / n_facts if n_facts else 0.0
    fp = false_positive_rate(queries, cleanup, threshold=0.5)
    return {
        "facts": int(n_facts),
        "noise": float(noise_sigma),
        "accuracy": float(accuracy),
        "false_positive_rate": float(fp),
        "mean_top1_similarity": float(np.mean(similarities)) if similarities else 0.0,
    }


def _check_thresholds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _row(facts: int, noise: float) -> Dict[str, Any] | None:
        for r in rows:
            if r["facts"] == facts and abs(r["noise"] - noise) < 1e-9:
                return r
        return None

    row_8 = _row(8, 0.0)
    row_16 = _row(16, 0.0)
    fp_rows = [r for r in rows if r["noise"] == 0.0]
    max_fp = max((r["false_positive_rate"] for r in fp_rows), default=0.0)

    acc_at_8 = row_8["accuracy"] if row_8 else None
    acc_at_16 = row_16["accuracy"] if row_16 else None

    ok_at_8 = (acc_at_8 is not None) and (acc_at_8 >= ACCURACY_THRESHOLD_AT_8)
    ok_at_16 = (acc_at_16 is not None) and (acc_at_16 >= ACCURACY_THRESHOLD_AT_16)
    ok_fp = max_fp <= FP_THRESHOLD
    all_pass = bool(ok_at_8 and ok_at_16 and ok_fp)

    return {
        "cleanup_accuracy_at_8": acc_at_8,
        "cleanup_accuracy_at_16": acc_at_16,
        "false_positive_rate": float(max_fp),
        "thresholds": {
            "cleanup_accuracy_at_8_min": ACCURACY_THRESHOLD_AT_8,
            "cleanup_accuracy_at_16_min": ACCURACY_THRESHOLD_AT_16,
            "false_positive_rate_max": FP_THRESHOLD,
        },
        "ok_at_8": ok_at_8,
        "ok_at_16": ok_at_16,
        "ok_fp": ok_fp,
        "all_pass": all_pass,
    }


def run_exercise(
    dim: int,
    facts: List[int],
    noise_levels: List[float],
    seed: int,
    capacity_cleanup_threshold: float = 0.4,
) -> Dict[str, Any]:
    """Run the full exercise and return the JSON-serializable result dict."""
    cfg = HRRConfig(dim=dim, seed=seed)
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    for n in facts:
        for sigma in noise_levels:
            rows.append(_run_at_fact_count(cfg, n, sigma, rng))

    # Extra: superposition_capacity at threshold 0.4 for a larger pool.
    cap_cfg = cfg
    cap_dict = SymbolDictionary(cap_cfg)
    pool = max(max(facts) * 2, 32)
    pairs = []
    cleanup = CleanupMemory(cap_cfg)
    for i in range(pool):
        r = cap_dict.role(f"role_cap_{i}")
        f = cap_dict.entity(f"filler_cap_{i}")
        pairs.append((f"role_cap_{i}", r, f"filler_cap_{i}", f))
        cleanup.add(f"filler_cap_{i}", f)
    capacity = int(
        superposition_capacity(pairs, cleanup, cap_cfg, threshold=capacity_cleanup_threshold)
    )

    gates = _check_thresholds(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "stage0_synthetic",
        "status": "PRE-MATURE",
        "generated_at": int(time.time()),
        "backend": "numpy_fft_cpu",
        "config": {
            "dim": dim,
            "seed": seed,
            "facts": facts,
            "noise_levels": noise_levels,
            "capacity_cleanup_threshold": capacity_cleanup_threshold,
        },
        "rows": rows,
        "gates": gates,
        "superposition_capacity_at_t040": capacity,
        "hrr_side_effects": 0,
        "authority_flags": {
            "policy_influence": False,
            "belief_write_enabled": False,
            "canonical_memory": False,
            "autonomy_influence": False,
            "llm_raw_vector_exposure": False,
            "soul_integrity_influence": False,
        },
    }


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HRR Stage 0 capacity exercise")
    parser.add_argument("--dim", type=int, default=1024, help="vector dimension")
    parser.add_argument(
        "--facts",
        type=_parse_int_list,
        default=[1, 2, 4, 8, 16, 32],
        help="comma-separated fact counts to probe",
    )
    parser.add_argument(
        "--noise",
        type=_parse_float_list,
        default=[0.0, 0.05, 0.10],
        help="comma-separated Gaussian noise sigmas",
    )
    parser.add_argument("--seed", type=int, default=0, help="HRR dictionary seed")
    parser.add_argument("--out", type=str, required=False, default=None, help="output JSON path")
    args = parser.parse_args(argv)

    result = run_exercise(
        dim=args.dim,
        facts=list(args.facts),
        noise_levels=list(args.noise),
        seed=args.seed,
    )

    payload = json.dumps(result, indent=2, sort_keys=True)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(payload)

    # Exit non-zero if Stage 0 thresholds fail so CI can notice.
    return 0 if result["gates"]["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
