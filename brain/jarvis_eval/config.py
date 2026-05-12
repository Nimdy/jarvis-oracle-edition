"""Eval sidecar configuration constants."""

from __future__ import annotations

from pathlib import Path

EVAL_DIR = Path.home() / ".jarvis" / "eval"

SCORING_VERSION = "0.2.0-pvl"
SCENARIO_PACK_VERSION = "0.1.0"

COLLECTOR_INTERVAL_S = 60.0
EVENT_BUFFER_MAXLEN = 5000
FLUSH_INTERVAL_S = 10.0
MAX_JSONL_SIZE_MB = 50
COMPOSITE_ENABLED = False
ORACLE_SCORECARD_INTERVAL_S = 900.0  # sparse executive rollup every 15 minutes

EVENTS_FILE = "eval_events.jsonl"
SNAPSHOTS_FILE = "eval_snapshots.jsonl"
SCORES_FILE = "eval_scores.jsonl"
SCORECARDS_FILE = "oracle_scorecards.jsonl"
RUNS_FILE = "eval_runs.jsonl"
META_FILE = "eval_meta.json"

# PVL: Process Verification Layer
PVL_VERIFY_EVERY_N_FLUSHES = 6  # verify every 6 flushes = ~60s at 10s flush interval
PVL_EVENT_WINDOW = 500  # recent events to feed into verifier
