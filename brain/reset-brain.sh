#!/usr/bin/env bash
#
# reset-brain.sh — Wipe epistemic state so Jarvis re-gestates from scratch.
#
# Usage:
#   ./reset-brain.sh              # Standard reset (keeps biometrics, models, Blue Diamonds)
#   ./reset-brain.sh --nuke       # Scorched earth — wipes all state + biometrics (keeps models)
#   ./reset-brain.sh --full-wipe  # --nuke + also destroys Blue Diamonds archive
#

set -euo pipefail

JARVIS_DIR="${HOME}/.jarvis"
BLUE_DIAMONDS_DIR="${BLUE_DIAMONDS_PATH:-${HOME}/.jarvis_blue_diamonds}"
NUKE=false
FULL_WIPE=false

for arg in "$@"; do
    case "$arg" in
        --nuke)      NUKE=true ;;
        --full-wipe) NUKE=true; FULL_WIPE=true ;;
    esac
done

# ── Warning ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================="
if [[ "${NUKE}" == "true" ]]; then
    echo "  WARNING: SCORCHED EARTH RESET"
    echo ""
    echo "  This will DELETE EVERYTHING in ~/.jarvis/ including:"
    echo "    - All memories, beliefs, consciousness state"
    echo "    - Identity (speakers.json, face_profiles.json, identity.json)"
    echo "    - Hemisphere NNs, policy models, kernel config"
    echo "    - All caches, library, skills, learning jobs"
    echo ""
    echo "  Downloaded ML models (models/) are PRESERVED to avoid re-downloads."
    echo ""
    if [[ "${FULL_WIPE}" == "true" ]]; then
        echo "  *** --full-wipe: Blue Diamonds archive will also be DESTROYED ***"
    else
        echo "  Blue Diamonds archive PRESERVED at: ${BLUE_DIAMONDS_DIR}"
    fi
else
    echo "  WARNING: This will reset Jarvis's epistemic state."
    echo ""
    echo "  All memories, beliefs, calibration data, consciousness state,"
    echo "  identity, hemisphere NNs, policy, autonomy experience,"
    echo "  and the birth certificate will be erased."
    echo "  Jarvis will re-gestate on next boot."
    echo ""
    echo "  PRESERVED: biometrics (speakers, faces), downloaded models, Blue Diamonds."
    echo ""
    echo "  Blue Diamonds archive at: ${BLUE_DIAMONDS_DIR}"
    echo "  (Validated knowledge will auto-reload on next gestation)"
    echo ""
    echo "  Use --nuke to additionally wipe biometrics."
    echo "  Use --full-wipe for nuke + destroy Blue Diamonds."
fi
echo "========================================================================="
echo ""

# ── Confirmation ─────────────────────────────────────────────────────────────
read -rp "Are you sure you want to proceed? (y/N) " answer
if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

if [[ "${FULL_WIPE}" == "true" && -d "${BLUE_DIAMONDS_DIR}" ]]; then
    echo ""
    echo "  !! This will permanently destroy your curated knowledge archive !!"
    echo "  !! This data cannot be recovered.                               !!"
    read -rp "  Delete Blue Diamonds at ${BLUE_DIAMONDS_DIR}? (y/N) " bd_answer
    if [[ "${bd_answer}" == "y" || "${bd_answer}" == "Y" ]]; then
        rm -rf "${BLUE_DIAMONDS_DIR}"
        echo "  Blue Diamonds archive deleted."
    else
        echo "  Blue Diamonds preserved."
    fi
fi

echo ""

if [[ ! -d "${JARVIS_DIR}" ]]; then
    echo "Nothing to reset — ${JARVIS_DIR} does not exist."
    exit 0
fi

# ── Files to remove ──────────────────────────────────────────────────────────
FILES_TO_REMOVE=(
    "memories.json"
    "consciousness_state.json"
    "beliefs.jsonl"
    "tensions.jsonl"
    "belief_edges.jsonl"
    "calibration_truth.jsonl"
    "confidence_outcomes.jsonl"
    "confidence_adjustments.jsonl"
    "quarantine_candidates.jsonl"
    "autonomy_policy.jsonl"
    "policy_experience.jsonl"
    "memory_retrieval_log.jsonl"
    "memory_lifecycle_log.jsonl"
    "memory_ranker.pt"
    "memory_salience.pt"
    "attribution_ledger.jsonl"
    "vector_memory.db"
    "episodes.json"
    "conversation_history.json"
    "goals.json"
    "world_model_promotion.json"
    "calibration_state.json"
    "gestation_summary.json"
    "skill_registry.json"
    "capability_blocks.json"
    "personality_snapshots.json"
    "memory_clusters.json"
    "consciousness_reports.json"
    "causal_models.json"
    "identity.json"
    "improvements.json"
    "improvement_proposals.jsonl"
    "pending_approvals.json"
    "autonomy_state.json"
    "delta_counters.json"
    "delta_pending.json"
    "drive_state.json"
    "metric_hourly.json"
    "gap_detector_state.json"
    "identity_aliases.jsonl"
    "expansion_state.json"
    "simulator_promotion.json"
    "flight_recorder.json"
    "friction_events.jsonl"
    "source_ledger.jsonl"
    "interventions.jsonl"
    "onboarding_state.json"
    "language_promotion.json"
    "intention_registry.json"
    "intention_outcomes.jsonl"
    "routing_corrections.jsonl"
    "code_index.json"
    "code_index_hashes.json"
    "eval_comparisons.jsonl"
)

DIRS_TO_REMOVE=(
    "autonomy_episodes"
    "learning_jobs"
    "library"
    "hemispheres"
    "hemisphere_training"
    "kernel_snapshots"
    "improvement_snapshots"
    "plugins"
    "plugin_venvs"
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
    "eval"
    "synthetic_exercise"
    "language_corpus"
    "improvement_conversations"
    "policy_models"
)

# ── Files/dirs preserved ─────────────────────────────────────────────────────
PRESERVED=(
    "speakers.json"
    "face_profiles.json"
    "models/"                 # ALL downloaded ML models (avoids re-downloads)
    "instance_id"
    # Blue Diamonds lives OUTSIDE ~/.jarvis at ~/.jarvis_blue_diamonds/
)

# ── Remove files ─────────────────────────────────────────────────────────────

if [[ "${NUKE}" == "true" ]]; then
    # Scorched earth: wipe everything EXCEPT downloaded models
    echo "── NUKE: Removing everything in ${JARVIS_DIR}/ ─────────────────────"
    echo "  (Downloaded models in models/ are PRESERVED to avoid re-downloads)"
    echo ""
    for item in "${JARVIS_DIR}"/*; do
        [[ -e "${item}" ]] || continue
        name="$(basename "${item}")"
        if [[ "${name}" == "models" ]]; then
            size=$(du -sh "${item}" 2>/dev/null | cut -f1)
            echo "  KEEP  ${name} (${size})"
        else
            echo "  WIPE  ${name}"
            rm -rf "${item}"
        fi
    done
    echo ""
    echo "  All state destroyed. Downloaded models preserved."
else
    removed=()
    skipped=()

    for f in "${FILES_TO_REMOVE[@]}"; do
        target="${JARVIS_DIR}/${f}"
        if [[ -f "${target}" ]]; then
            rm -f "${target}"
            removed+=("${f}")
        else
            skipped+=("${f}")
        fi
    done

    for d in "${DIRS_TO_REMOVE[@]}"; do
        target="${JARVIS_DIR}/${d}"
        if [[ -d "${target}" ]]; then
            rm -rf "${target}"
            removed+=("${d}/")
        else
            skipped+=("${d}/")
        fi
    done

    # ── Summary ──────────────────────────────────────────────────────────────────
    echo "── Removed ────────────────────────────────────────────────────────────"
    if [[ ${#removed[@]} -gt 0 ]]; then
        for item in "${removed[@]}"; do
            echo "  - ${item}"
        done
    else
        echo "  (nothing to remove — already clean)"
    fi

    echo ""
    echo "── Skipped (not found) ──────────────────────────────────────────────"
    if [[ ${#skipped[@]} -gt 0 ]]; then
        for item in "${skipped[@]}"; do
            echo "  - ${item}"
        done
    else
        echo "  (all targets existed)"
    fi

    echo ""
    echo "── Preserved ──────────────────────────────────────────────────────────"
    for item in "${PRESERVED[@]}"; do
        target="${JARVIS_DIR}/${item}"
        if [[ -e "${target}" ]]; then
            echo "  + ${item}"
        else
            echo "  . ${item}  (not present)"
        fi
    done
fi

echo ""
if [[ -d "${BLUE_DIAMONDS_DIR}" ]]; then
    bd_count=$(sqlite3 "${BLUE_DIAMONDS_DIR}/archive.db" "SELECT COUNT(*) FROM diamonds;" 2>/dev/null || echo "?")
    echo "── Blue Diamonds ──────────────────────────────────────────────────"
    echo "  Archive preserved at: ${BLUE_DIAMONDS_DIR}"
    echo "  Diamonds available:   ${bd_count}"
    echo "  (Will auto-reload on next gestation)"
fi

echo ""
echo "Brain reset complete. Jarvis will re-gestate on next boot."
