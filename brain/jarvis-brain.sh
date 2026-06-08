#!/usr/bin/env bash
# Jarvis Brain — entry point that launches the process supervisor.
#
# The supervisor manages the brain lifecycle: spawn, monitor, restart,
# rollback, crash backoff. systemd manages this script (and therefore
# the supervisor), not the brain directly.
#
# Usage:
#   ./jarvis-brain.sh          # Run via supervisor (normal mode)
#   ./jarvis-brain.sh --once   # Bypass supervisor, run main.py directly (debugging)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[jarvis-brain] ERROR: venv Python not found at $VENV_PYTHON" >&2
    echo "[jarvis-brain] Run ./setup.sh first" >&2
    exit 1
fi

if [[ "${1:-}" == "--once" ]]; then
    shift
    exec "$VENV_PYTHON" -u main.py "$@"
fi

exec "$VENV_PYTHON" -u jarvis-supervisor.py "$@"
