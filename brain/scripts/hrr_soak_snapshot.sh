#!/usr/bin/env bash
# P4 HRR live-soak snapshot loop.
#
# Captures /api/hrr/status, /api/hrr/samples, /api/meta/status-markers plus
# the dashboard truth probe and schema audit at a fixed cadence. Safe to run
# repeatedly — every tick writes a fresh timestamped set of files under
# docs/validation_reports/evidence/hrr_soak/ and appends a one-line summary
# to hrr_soak.ndjson for easy time-series review.
#
# Environment overrides:
#   HRR_SOAK_INTERVAL_S   seconds between ticks (default 900 = 15 minutes)
#   HRR_SOAK_TICKS        stop after N ticks (default: unlimited, 0 means unlimited)
#   HRR_SOAK_DASHBOARD    dashboard base URL (default http://127.0.0.1:9200)
#   HRR_SOAK_REPO         repo root (default: ~/duafoo)
#
# Usage:
#   nohup bash brain/scripts/hrr_soak_snapshot.sh > ~/duafoo/logs/hrr_soak.log 2>&1 &
set -u

INTERVAL="${HRR_SOAK_INTERVAL_S:-900}"
MAX_TICKS="${HRR_SOAK_TICKS:-0}"
DASH="${HRR_SOAK_DASHBOARD:-http://127.0.0.1:9200}"
REPO="${HRR_SOAK_REPO:-$HOME/duafoo}"

EV_DIR="$REPO/docs/validation_reports/evidence/hrr_soak"
SUMMARY="$EV_DIR/hrr_soak.ndjson"
mkdir -p "$EV_DIR"

log() { echo "[hrr_soak $(date -u +%FT%TZ)] $*"; }

tick=0
while :; do
    tick=$((tick + 1))
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    log "tick=$tick stamp=$STAMP"

    STATUS="$EV_DIR/status_${STAMP}.json"
    SAMPLES="$EV_DIR/samples_${STAMP}.json"
    MARKERS="$EV_DIR/markers_${STAMP}.json"

    curl -sS --max-time 5 "$DASH/api/hrr/status"  > "$STATUS"  || log "status fetch failed"
    curl -sS --max-time 5 "$DASH/api/hrr/samples?world=200&simulation=200&recall=200" \
        > "$SAMPLES" || log "samples fetch failed"
    curl -sS --max-time 5 "$DASH/api/meta/status-markers" > "$MARKERS" || log "markers fetch failed"

    # Optional: only run the audits every 4th tick (hourly if interval=900)
    # to avoid pounding the box with subprocesses.
    if [ $((tick % 4)) -eq 1 ]; then
        TPRO="$EV_DIR/truth_probe_${STAMP}.txt"
        SAUD="$EV_DIR/schema_audit_${STAMP}.txt"
        (
            cd "$REPO/brain" \
              && source .venv/bin/activate \
              && python scripts/dashboard_truth_probe.py > "$TPRO" 2>&1
            cd "$REPO/brain" && python -m scripts.schema_emission_audit > "$SAUD" 2>&1
        )
    fi

    # One-line summary for the ndjson time-series.
    python3 - "$STATUS" "$MARKERS" "$STAMP" "$tick" >> "$SUMMARY" <<'PY' || log "summary failed"
import json, sys, pathlib, time
status_path, marker_path, stamp, tick = sys.argv[1:5]
try:
    s = json.loads(pathlib.Path(status_path).read_text())
except Exception as e:
    s = {"_status_load_error": str(e)}
try:
    m = json.loads(pathlib.Path(marker_path).read_text())
    mk = (m.get("markers") or m).get("holographic_cognition_hrr")
except Exception as e:
    mk = None
w = s.get("world_shadow") or {}
sim = s.get("simulation_shadow") or {}
rec = s.get("recall_advisory") or {}
row = {
    "ts_utc": stamp,
    "tick": int(tick),
    "stage": s.get("stage"),
    "enabled": s.get("enabled"),
    "world_samples_total": w.get("samples_total"),
    "world_samples_retained": w.get("samples_retained"),
    "world_binding_cleanliness": w.get("binding_cleanliness"),
    "world_cleanup_accuracy": w.get("cleanup_accuracy"),
    "world_similarity_to_previous": w.get("similarity_to_previous"),
    "sim_samples_total": sim.get("samples_total"),
    "sim_samples_retained": sim.get("samples_retained"),
    "recall_samples_total": rec.get("samples_total"),
    "recall_samples_retained": rec.get("samples_retained"),
    "recall_help_rate": rec.get("help_rate"),
    "policy_influence": s.get("policy_influence"),
    "belief_write_enabled": s.get("belief_write_enabled"),
    "canonical_memory": s.get("canonical_memory"),
    "autonomy_influence": s.get("autonomy_influence"),
    "llm_raw_vector_exposure": s.get("llm_raw_vector_exposure"),
    "soul_integrity_influence": s.get("soul_integrity_influence"),
    "public_marker_holographic_cognition_hrr": mk,
    "captured_at_epoch": int(time.time()),
}
print(json.dumps(row))
PY

    if [ "$MAX_TICKS" -gt 0 ] && [ "$tick" -ge "$MAX_TICKS" ]; then
        log "reached max ticks=$MAX_TICKS; exiting"
        break
    fi

    sleep "$INTERVAL"
done
