#!/usr/bin/env bash
# Jarvis Brain — systemd service installer
#
# Usage:
#   ./service-install.sh install    Install and start the service
#   ./service-install.sh uninstall  Stop and remove the service
#   ./service-install.sh status     Show service status
#   ./service-install.sh logs       Tail live logs (Ctrl+C to stop)
#   ./service-install.sh restart    Restart the service
#   ./service-install.sh stop       Stop the service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="jarvis-brain"
TEMPLATE="${SCRIPT_DIR}/jarvis-brain.service"
WRAPPER="${SCRIPT_DIR}/jarvis-brain.sh"
ENV_FILE="${SCRIPT_DIR}/.env"

# Detect whether to use system-level or user-level systemd
if [[ $EUID -eq 0 ]]; then
    UNIT_DIR="/etc/systemd/system"
    CTL="systemctl"
    JOURNAL_ARGS=()
else
    UNIT_DIR="${HOME}/.config/systemd/user"
    CTL="systemctl --user"
    JOURNAL_ARGS=("--user")
fi

UNIT_FILE="${UNIT_DIR}/${SERVICE_NAME}.service"

_install() {
    if [[ ! -f "$TEMPLATE" ]]; then
        echo "ERROR: Service template not found: $TEMPLATE" >&2
        exit 1
    fi
    if [[ ! -x "$WRAPPER" ]]; then
        echo "ERROR: Wrapper script not found or not executable: $WRAPPER" >&2
        echo "Run: chmod +x $WRAPPER" >&2
        exit 1
    fi

    echo "Installing ${SERVICE_NAME} service..."
    echo "  Unit dir:    $UNIT_DIR"
    echo "  Working dir: $SCRIPT_DIR"
    echo "  Exec:        $WRAPPER"
    echo "  Env file:    $ENV_FILE"

    mkdir -p "$UNIT_DIR"

    # Patch template with actual paths
    sed \
        -e "s|__WORKING_DIR__|${SCRIPT_DIR}|g" \
        -e "s|__EXEC_START__|${WRAPPER}|g" \
        -e "s|__USER__|$(whoami)|g" \
        -e "s|__ENV_FILE__|${ENV_FILE}|g" \
        "$TEMPLATE" > "$UNIT_FILE"

    $CTL daemon-reload
    $CTL enable "$SERVICE_NAME"
    $CTL start "$SERVICE_NAME"

    echo ""
    echo "Service installed and started."
    echo "  Status:  $CTL status $SERVICE_NAME"
    echo "  Logs:    journalctl ${JOURNAL_ARGS[*]:-} -u $SERVICE_NAME -f"
    echo "  Stop:    $CTL stop $SERVICE_NAME"
    echo "  Restart: $CTL restart $SERVICE_NAME"
}

_uninstall() {
    echo "Uninstalling ${SERVICE_NAME} service..."

    $CTL stop "$SERVICE_NAME" 2>/dev/null || true
    $CTL disable "$SERVICE_NAME" 2>/dev/null || true

    if [[ -f "$UNIT_FILE" ]]; then
        rm -f "$UNIT_FILE"
        echo "  Removed: $UNIT_FILE"
    fi

    $CTL daemon-reload
    echo "Service uninstalled."
}

_status() {
    $CTL status "$SERVICE_NAME" || true
}

_logs() {
    journalctl "${JOURNAL_ARGS[@]}" -u "$SERVICE_NAME" -f --no-hostname -o short-iso
}

_restart() {
    $CTL restart "$SERVICE_NAME"
    echo "Service restarted."
}

_stop() {
    $CTL stop "$SERVICE_NAME"
    echo "Service stopped."
}

case "${1:-help}" in
    install)   _install ;;
    uninstall) _uninstall ;;
    status)    _status ;;
    logs)      _logs ;;
    restart)   _restart ;;
    stop)      _stop ;;
    *)
        echo "Usage: $0 {install|uninstall|status|logs|restart|stop}"
        exit 1
        ;;
esac
