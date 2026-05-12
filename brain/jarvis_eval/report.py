"""Phase A report stub — basic observability without scoring.

The eval sidecar is a read-only observer.
"""

from __future__ import annotations

import time
from typing import Any

from jarvis_eval.store import EvalStore


def build_report(store: EvalStore, started_at: float) -> dict[str, Any]:
    """Build a Phase A report: event counts, freshness, file sizes."""
    recent_events = store.read_recent_events(limit=500)
    recent_snapshots = store.read_recent_snapshots(limit=100)

    event_counts: dict[str, int] = {}
    for ev in recent_events:
        t = ev.get("event_type", "unknown")
        event_counts[t] = event_counts.get(t, 0) + 1

    snapshot_sources: dict[str, int] = {}
    latest_snapshot_values: dict[str, dict[str, Any]] = {}
    for snap in recent_snapshots:
        src = snap.get("source", "unknown")
        snapshot_sources[src] = snapshot_sources.get(src, 0) + 1
        latest_snapshot_values[src] = snap.get("metrics", {})

    last_event_ts = recent_events[-1].get("timestamp", 0) if recent_events else 0
    last_snapshot_ts = recent_snapshots[-1].get("timestamp", 0) if recent_snapshots else 0

    now = time.time()
    meta = store.get_meta()

    return {
        "event_counts_24h": event_counts,
        "snapshot_source_counts": snapshot_sources,
        "latest_snapshot_values": latest_snapshot_values,
        "file_sizes": store.get_file_sizes(),
        "rotation_counts": meta.get("rotation_counts", {}),
        "data_freshness": {
            "since_last_event_s": round(now - last_event_ts, 1) if last_event_ts else None,
            "since_last_snapshot_s": round(now - last_snapshot_ts, 1) if last_snapshot_ts else None,
            "since_last_flush_s": round(now - meta["last_flush_ts"], 1) if meta.get("last_flush_ts") else None,
        },
        "dropped_event_count": meta.get("dropped_event_count", 0),
        "sidecar_uptime_s": round(now - started_at, 1),
        "total_events_written": meta.get("total_events_written", 0),
        "total_snapshots_written": meta.get("total_snapshots_written", 0),
    }
