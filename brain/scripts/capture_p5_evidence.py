"""Capture P5 mental-world fixture + live evidence JSON files.

Produces two files under ``docs/validation_reports/evidence/``:

* ``hrr_spatial_scene_fixture.json`` — deterministic fixture-scene encoder
  metrics (no engine, no API; pure encoder run with seed=0).
* ``hrr_spatial_scene_live.json`` — live engine-sampled snapshot pulled
  from ``/api/hrr/scene`` on the running brain.

Both shapes match what
:mod:`brain.jarvis_eval.validation_pack._build_p5_mental_world_checks` reads
to compute ``p5_mental_world_fixture_ok`` / ``p5_mental_world_live_ok``.

Usage::

    PYTHONPATH=brain python brain/scripts/capture_p5_evidence.py \
        --host 127.0.0.1 --port 9200

Run on the host that has the running brain (the live half needs a reachable
``/api/hrr/scene`` endpoint). The fixture half is pure-Python and runs
anywhere the encoder can be imported.

Authority guarantee: this script only **reads** state. It calls the encoder
as a pure function and reads one HTTP endpoint. No persistent writes other
than the two evidence JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Repo root resolution: this file lives at brain/scripts/capture_p5_evidence.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRAIN_ROOT = _REPO_ROOT / "brain"
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

from cognition.hrr_spatial_encoder import encode_scene_graph  # noqa: E402
from cognition.spatial_scene_graph import (  # noqa: E402
    MentalWorldEntity,
    MentalWorldRelation,
    MentalWorldSceneGraph,
)
from library.vsa.hrr import HRRConfig  # noqa: E402
from library.vsa.symbols import SymbolDictionary  # noqa: E402


_AUTHORITY_KEYS = (
    "writes_memory",
    "writes_beliefs",
    "influences_policy",
    "influences_autonomy",
    "soul_integrity_influence",
    "llm_raw_vector_exposure",
)


def _build_fixture_scene() -> MentalWorldSceneGraph:
    """Deterministic 4-entity / 2-relation scene used as the fixture probe.

    Mirrors the shape exercised by ``test_relation_recovery_hits_expected_entity_set``
    and ``test_entity_state_cleanup_accuracy_is_perfect_on_small_scene`` so the
    fixture metrics here track those tests.
    """

    def _ent(eid: str, label: str, state: str, region: str) -> MentalWorldEntity:
        return MentalWorldEntity(
            entity_id=eid,
            label=label,
            state=state,
            region=region,
            position_room_m=(0.0, 0.7, 1.0),
            confidence=0.85,
            last_seen_ts=100.0,
            is_display_surface=False,
        )

    entities = (
        _ent("cup_0", "cup", "visible", "desk_center"),
        _ent("cup_1", "cup", "occluded", "desk_left"),
        _ent("monitor_0", "monitor", "visible", "monitor_zone"),
        _ent("keyboard_0", "keyboard", "visible", "desk_center"),
    )
    relations = (
        MentalWorldRelation(
            source_entity_id="cup_0",
            target_entity_id="cup_1",
            relation_type="left_of",
            value_m=0.5,
            confidence=0.9,
        ),
        MentalWorldRelation(
            source_entity_id="monitor_0",
            target_entity_id="keyboard_0",
            relation_type="near",
            value_m=0.1,
            confidence=0.8,
        ),
    )
    return MentalWorldSceneGraph(
        timestamp=100.0,
        entities=entities,
        relations=relations,
        source_scene_update_count=3,
        source_track_count=4,
        source_anchor_count=0,
        source_calibration_version=1,
        reason=None,
    )


def _capture_fixture(dim: int = 1024, seed: int = 0) -> dict[str, Any]:
    """Run the deterministic fixture scene through the encoder and return
    a JSON-safe metrics dict matching the validation-pack expected shape.
    """
    cfg = HRRConfig(dim=dim, seed=seed)
    symbols = SymbolDictionary(cfg)
    graph = _build_fixture_scene()
    out = encode_scene_graph(graph, cfg, symbols)

    return {
        "schema_version": 1,
        "captured_at": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "deterministic_encoder_fixture",
        "config": {"dim": dim, "seed": seed},
        "scene": {
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "entities": [
                {"id": e.entity_id, "label": e.label, "state": e.state, "region": e.region}
                for e in graph.entities
            ],
            "relations": [
                {
                    "source": r.source_entity_id,
                    "target": r.target_entity_id,
                    "type": r.relation_type,
                }
                for r in graph.relations
            ],
        },
        "entities_encoded": int(out.get("entities_encoded", 0)),
        "relations_encoded": int(out.get("relations_encoded", 0)),
        "binding_cleanliness": out.get("binding_cleanliness"),
        "cleanup_accuracy": out.get("cleanup_accuracy"),
        "relation_recovery": out.get("relation_recovery"),
        "cleanup_failures": int(out.get("cleanup_failures", 0)),
        "spatial_hrr_side_effects": int(out.get("side_effects", 0)),
        "authority_flags": {k: False for k in _AUTHORITY_KEYS},
        "no_raw_vectors_in_payload": True,
    }


def _http_get(url: str, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _capture_live(host: str, port: int, samples_target: int = 3) -> dict[str, Any]:
    """Pull live ``/api/hrr/scene`` snapshots and aggregate them.

    Polls the endpoint up to ``samples_target`` times, separated by ~1.5s,
    and records the per-sample metrics. Side effects must remain at zero
    across all samples for ``live_ok`` to flip true.
    """
    base = f"http://{host}:{port}"
    samples: list[dict[str, Any]] = []
    last_status_payload: dict[str, Any] = {}
    side_effects_total = 0
    seen_ts: set[float] = set()

    deadline = time.time() + max(8.0, 2.5 * samples_target)
    while len(samples) < samples_target and time.time() < deadline:
        try:
            payload = _http_get(f"{base}/api/hrr/scene")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {
                "schema_version": 1,
                "captured_at": time.time(),
                "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "live_api_unreachable",
                "error": f"{type(exc).__name__}: {exc}",
                "samples_observed": 0,
                "spatial_hrr_side_effects": 0,
                "authority_flags": {k: False for k in _AUTHORITY_KEYS},
            }
        last_status_payload = payload
        ts = float(payload.get("timestamp") or 0.0)
        if ts in seen_ts:
            time.sleep(1.5)
            continue
        seen_ts.add(ts)
        metrics = payload.get("metrics") or {}
        side_effects_total += int(metrics.get("spatial_hrr_side_effects", 0) or 0)
        samples.append(
            {
                "timestamp": ts,
                "tick": payload.get("tick"),
                "entity_count": int(payload.get("entity_count") or 0),
                "relation_count": int(payload.get("relation_count") or 0),
                "binding_cleanliness": metrics.get("binding_cleanliness"),
                "cleanup_accuracy": metrics.get("cleanup_accuracy"),
                "relation_recovery": metrics.get("relation_recovery"),
                "cleanup_failures": int(metrics.get("cleanup_failures", 0) or 0),
                "similarity_to_previous": metrics.get("similarity_to_previous"),
                "spatial_hrr_side_effects": int(
                    metrics.get("spatial_hrr_side_effects", 0) or 0
                ),
                "reason": payload.get("reason"),
            }
        )
        time.sleep(1.5)

    authority_flags = {k: bool(last_status_payload.get(k, False)) for k in _AUTHORITY_KEYS}

    return {
        "schema_version": 1,
        "captured_at": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "live_api_hrr_scene",
        "host": host,
        "port": port,
        "endpoint": "/api/hrr/scene",
        "status": last_status_payload.get("status"),
        "lane": last_status_payload.get("lane"),
        "enabled": bool(last_status_payload.get("enabled", False)),
        "enabled_source": last_status_payload.get("enabled_source"),
        "spatial_scene_enabled_source": last_status_payload.get(
            "spatial_scene_enabled_source"
        ),
        "samples_observed": len(samples),
        "samples": samples,
        "spatial_hrr_side_effects": side_effects_total,
        "authority_flags": authority_flags,
        "no_raw_vectors_in_api": bool(
            last_status_payload.get("no_raw_vectors_in_api", True)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="127.0.0.1", help="Brain dashboard host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=9200, help="Brain dashboard port (default: 9200)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of live samples to capture (default: 3)",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="Override evidence directory (default: docs/validation_reports/evidence)",
    )
    parser.add_argument(
        "--skip-fixture", action="store_true", help="Skip the fixture half"
    )
    parser.add_argument(
        "--skip-live", action="store_true", help="Skip the live half"
    )
    args = parser.parse_args()

    evidence_dir = (
        args.evidence_dir
        if args.evidence_dir is not None
        else _REPO_ROOT / "docs" / "validation_reports" / "evidence"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = evidence_dir / "hrr_spatial_scene_fixture.json"
    live_path = evidence_dir / "hrr_spatial_scene_live.json"

    rc = 0

    if not args.skip_fixture:
        fixture_payload = _capture_fixture()
        fixture_path.write_text(
            json.dumps(fixture_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[fixture] wrote {fixture_path} "
            f"(cleanup_accuracy={fixture_payload['cleanup_accuracy']}, "
            f"relation_recovery={fixture_payload['relation_recovery']}, "
            f"side_effects={fixture_payload['spatial_hrr_side_effects']})"
        )

    if not args.skip_live:
        live_payload = _capture_live(args.host, args.port, samples_target=args.samples)
        live_path.write_text(
            json.dumps(live_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        samples_observed = live_payload.get("samples_observed", 0)
        side_effects = live_payload.get("spatial_hrr_side_effects", 0)
        if samples_observed == 0 or side_effects != 0:
            rc = max(rc, 1)
        print(
            f"[live] wrote {live_path} "
            f"(samples_observed={samples_observed}, "
            f"side_effects={side_effects}, "
            f"status={live_payload.get('status')})"
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
