#!/usr/bin/env bash
# Sync code to the desktop brain (192.168.1.222)
# Usage: ./sync-desktop.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DESKTOP_HOST="duafoo@192.168.1.222"
SSH_KEY="$HOME/.ssh/id_jarvis_desktop"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

EXCLUDES=(
    ".venv/"
    "__pycache__/"
    "*.pyc"
    ".git/"
    "delete_later/"
    "old_references_delete_later/"
    "node_modules/"
    ".env"
    "*.egg-info/"
    ".pytest_cache/"
    ".ruff_cache/"
    "models/"
    "website/"
)

EXCLUDE_ARGS=""
for pat in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pat"
done

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "DRY RUN — no files will be changed"
fi

echo "=== Syncing brain/ to desktop ==="
# brain/ already contains dashboard/static/* so the HRR panel HTML/JS/CSS
# and every other static dashboard page rides along with this sync.
rsync -avz $DRY_RUN $EXCLUDE_ARGS \
    -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/brain/" "$DESKTOP_HOST:~/duafoo/brain/"

echo "=== Syncing pi/ to desktop (for reference) ==="
# pi/ui/static/* (including hrr_scene.html/css for /mind) is synced here.
rsync -avz $DRY_RUN $EXCLUDE_ARGS \
    -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/pi/" "$DESKTOP_HOST:~/duafoo/pi/"

echo "=== Syncing docs/ to desktop (build history, plans, validation reports) ==="
# BUILD_HISTORY.md, MASTER_ROADMAP.md, plans/, validation_reports/, and every
# other governance / evidence doc must be available on the running brain so
# dashboard links, status-marker docs, and post-mortem probes all resolve.
rsync -avz $DRY_RUN $EXCLUDE_ARGS \
    -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/docs/" "$DESKTOP_HOST:~/duafoo/docs/"

echo "=== Syncing root files ==="
for f in AGENTS.md ARCHITECTURE.md PROCESS_ARCHITECTURE.md TODO.md TODO_V2.md \
         README.md CONTRIBUTING.md LICENSE.md NOTICE.md THIRD_PARTY_LICENSES.md \
         About_JARVIS_ORACLE_EDITION_SCIENTIFIC_PAPER.md requirements.txt; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        rsync -avz $DRY_RUN \
            -e "ssh $SSH_OPTS" \
            "$SCRIPT_DIR/$f" "$DESKTOP_HOST:~/duafoo/$f"
    fi
done

echo "=== Done ==="
echo "NOTE: runtime HRR flag changes require a supervisor/main restart."
echo "      See brain/scripts/set_hrr_runtime_flags.py --status"
