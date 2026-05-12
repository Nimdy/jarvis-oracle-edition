#!/usr/bin/env bash
# Sync code to the Raspberry Pi 5 (senses node)
# Usage: ./sync-pi.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PI_HOST="nimda@192.168.1.248"
SSH_KEY="$HOME/.ssh/id_jarvis_pi"
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

echo "=== Syncing pi/ to pi ==="
rsync -avz --delete $DRY_RUN $EXCLUDE_ARGS \
    -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/pi/" "$PI_HOST:~/duafoo/pi/"

echo "=== Syncing brain/ to pi (for reference) ==="
rsync -avz --delete $DRY_RUN $EXCLUDE_ARGS \
    -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/brain/" "$PI_HOST:~/duafoo/brain/"

echo "=== Syncing root files ==="
for f in AGENTS.md ARCHITECTURE.md PROCESS_ARCHITECTURE.md TODO.md README.md requirements.txt; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        rsync -avz $DRY_RUN \
            -e "ssh $SSH_OPTS" \
            "$SCRIPT_DIR/$f" "$PI_HOST:~/duafoo/$f"
    fi
done

echo "=== Done ==="
