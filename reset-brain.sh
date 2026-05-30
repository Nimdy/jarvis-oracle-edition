#!/usr/bin/env bash
# Reset Jarvis brain runtime state for a clean generational cut.
# Preserves biometrics, static models, and optionally library.db.
#
# Usage:
#   ./reset-brain.sh              # dry-run (shows what would be deleted)
#   ./reset-brain.sh --confirm    # actually delete
#   ./reset-brain.sh --keep-library --confirm   # keep library.db
#
# IMPORTANT: Stop the brain service BEFORE running this script.
# This script runs on the DESKTOP brain (192.168.1.222) or locally.

set -euo pipefail

JARVIS_DIR="${HOME}/.jarvis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_CODE_DIR="$SCRIPT_DIR/brain/tools/plugins"
DRY_RUN=true
KEEP_LIBRARY=false

for arg in "$@"; do
    case "$arg" in
        --confirm) DRY_RUN=false ;;
        --keep-library) KEEP_LIBRARY=true ;;
        --help|-h)
            echo "Usage: $0 [--confirm] [--keep-library]"
            echo "  --confirm       Actually delete files (default: dry-run)"
            echo "  --keep-library  Preserve library/library.db (curated external sources)"
            exit 0
            ;;
    esac
done

if [ ! -d "$JARVIS_DIR" ]; then
    echo "ERROR: $JARVIS_DIR does not exist. Nothing to reset."
    exit 1
fi

# Files/dirs to PRESERVE (never touch these)
PRESERVE=(
    "speakers.json"
    "face_profiles.json"
    "models"                      # ALL models: mobilefacenet, ecapa-tdnn, faster-whisper,
                                  # huggingface (wav2vec2, sbert), kokoro, coder GGUF, sha256ok
    "instance_id"
    "runtime_flags.json"          # HRR operator opt-in (not brain state)
)

# Files/dirs to WIPE (all derived/learned state)
# Cross-referenced with AGENTS.md persistence table — every entry accounted for.
WIPE=(
    # --- Memory ---
    "memories.json"
    "vector_memory.db"
    "memory_ranker.pt"
    "memory_salience.pt"
    "memory_retrieval_log.jsonl"
    "memory_lifecycle_log.jsonl"
    "memory_clusters.json"
    # --- Consciousness ---
    "consciousness_state.json"
    "consciousness_reports.json"
    "gestation_summary.json"
    "kernel_config.json"
    "kernel_snapshots"
    # --- Identity & Soul ---
    "identity.json"
    "identity_aliases.jsonl"
    "personality_snapshots.json"
    # --- Conversation & Episodes ---
    "conversation_history.json"
    "episodes.json"
    "flight_recorder.json"
    # --- Policy ---
    "policy_experience.jsonl"
    "policy_models"
    # --- Autonomy ---
    "autonomy_policy.jsonl"
    "autonomy_state.json"
    "autonomy_episodes"
    "delta_counters.json"
    "delta_pending.json"
    "drive_state.json"
    "calibration_state.json"
    "metric_hourly.json"
    # --- Epistemic Stack ---
    "beliefs.jsonl"
    "tensions.jsonl"
    "belief_edges.jsonl"
    "calibration_truth.jsonl"
    "confidence_outcomes.jsonl"
    "confidence_adjustments.jsonl"
    "quarantine_candidates.jsonl"
    "causal_models.json"
    "attribution_ledger.jsonl"
    # --- Hemispheres & NNs ---
    "hemispheres"
    "hemisphere_training"
    "gap_detector_state.json"
    # --- Self-Improvement ---
    "improvements.json"
    "improvement_proposals.jsonl"
    "improvement_conversations"
    "improvement_snapshots"
    "pending_approvals.json"
    # --- Skills & Learning ---
    "skill_registry.json"
    "learning_jobs"
    "capability_blocks.json"
    # --- Acquisition Pipeline ---
    "acquisitions"
    "acquisition_claims"
    "acquisition_code_bundles"
    "acquisition_deployments"
    "acquisition_docs"
    "acquisition_environment_setups"
    "acquisition_plans"
    "acquisition_research"
    "acquisition_reviews"
    "acquisition_verifications"
    "acquisition_shadows"
    # --- Plugins ---
    "plugins"
    "plugin_venvs"
    # --- Goals ---
    "goals.json"
    # --- Intention Truth Layer ---
    "intention_registry.json"
    "intention_outcomes.jsonl"
    # --- World Model & Simulator ---
    "world_model_promotion.json"
    "simulator_promotion.json"
    "expansion_state.json"
    # --- Friction & Interventions ---
    "friction_events.jsonl"
    "source_ledger.jsonl"
    "interventions.jsonl"
    # --- Language Substrate ---
    "language_corpus"
    "language_promotion.json"
    # --- Onboarding ---
    "onboarding_state.json"
    # --- Eval Sidecar ---
    "eval"
    "eval_comparisons.jsonl"
    # --- Routing ---
    "routing_corrections.jsonl"
    # --- Caches ---
    "academic_search_cache.json"
    "web_search_cache.json"
    "code_index.json"
    "code_index_hashes.json"
    # --- Synthetic Exercise ---
    "synthetic_exercise"
)

# Conditionally wipe library
if [ "$KEEP_LIBRARY" = false ]; then
    WIPE+=("library")
else
    WIPE+=("library/retrieval_log.jsonl")
fi

echo "=== Jarvis Brain Reset ==="
echo "  Target: $JARVIS_DIR"
echo "  Mode:   $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'LIVE DELETE')"
echo "  Library: $([ "$KEEP_LIBRARY" = true ] && echo 'PRESERVED' || echo 'WIPED')"
echo ""

echo "--- PRESERVING ---"
for item in "${PRESERVE[@]}"; do
    path="$JARVIS_DIR/$item"
    if [ -e "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  KEEP  $item ($size)"
    else
        echo "  SKIP  $item (not found)"
    fi
done
echo ""

echo "--- WIPING (~/.jarvis/ state) ---"
for item in "${WIPE[@]}"; do
    path="$JARVIS_DIR/$item"
    if [ -e "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  WIPE  $item ($size)"
        if [ "$DRY_RUN" = false ]; then
            rm -rf "$path"
        fi
    else
        echo "  SKIP  $item (not found)"
    fi
done
echo ""

# Plugin source code lives in the code tree, not ~/.jarvis/
echo "--- WIPING (plugin source code) ---"
if [ -d "$PLUGIN_CODE_DIR" ]; then
    PLUGIN_COUNT=0
    for plugin_dir in "$PLUGIN_CODE_DIR"/*/; do
        [ -d "$plugin_dir" ] || continue
        plugin_name=$(basename "$plugin_dir")
        # Skip __pycache__ and the __init__.py marker
        [ "$plugin_name" = "__pycache__" ] && continue
        size=$(du -sh "$plugin_dir" 2>/dev/null | cut -f1)
        echo "  WIPE  brain/tools/plugins/$plugin_name/ ($size)"
        PLUGIN_COUNT=$((PLUGIN_COUNT + 1))
        if [ "$DRY_RUN" = false ]; then
            rm -rf "$plugin_dir"
        fi
    done
    if [ $PLUGIN_COUNT -eq 0 ]; then
        echo "  SKIP  No plugin directories found"
    fi
else
    echo "  SKIP  Plugin code directory not found ($PLUGIN_CODE_DIR)"
fi
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN COMPLETE ==="
    echo "To actually reset, run: $0 --confirm"
    echo "(Add --keep-library to preserve library.db)"
else
    echo "--- POST-RESET VERIFICATION ---"
    echo "  consciousness_state.json: $([ -f "$JARVIS_DIR/consciousness_state.json" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  memories.json:            $([ -f "$JARVIS_DIR/memories.json" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  policy_experience.jsonl:  $([ -f "$JARVIS_DIR/policy_experience.jsonl" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  beliefs.jsonl:            $([ -f "$JARVIS_DIR/beliefs.jsonl" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  plugins/:                 $([ -d "$JARVIS_DIR/plugins" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  acquisitions/:            $([ -d "$JARVIS_DIR/acquisitions" ] && echo 'EXISTS (BAD!)' || echo 'GONE (good)')"
    echo "  speakers.json:            $([ -f "$JARVIS_DIR/speakers.json" ] && echo 'PRESERVED (good)' || echo 'MISSING (ok for fresh)')"
    echo "  face_profiles.json:       $([ -f "$JARVIS_DIR/face_profiles.json" ] && echo 'PRESERVED (good)' || echo 'MISSING (ok for fresh)')"
    echo "  models/:                  $([ -d "$JARVIS_DIR/models" ] && echo "PRESERVED (good) — $(du -sh "$JARVIS_DIR/models" 2>/dev/null | cut -f1)" || echo 'MISSING (run setup.sh to download)')"
    echo ""
    echo "=== RESET COMPLETE ==="
    echo "Gestation trigger: is_fresh_brain() will detect:"
    echo "  - no memories loaded"
    echo "  - no consciousness state"
    echo "  - no policy experience"
    echo "  - no gestation_complete flag"
    echo ""
    echo "Start the brain to begin gestation: cd ~/duafoo/brain && ./jarvis-brain.sh"
fi
