#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Jarvis Senses — clean start
#
# Aggressively kills ALL Jarvis-related processes (python, chromium, aiohttp),
# releases Hailo NPU, frees ports, verifies venv, then launches fresh.
#
# Usage:  ./start.sh          — normal start
#         ./start.sh --setup  — run full setup.sh first (deps, models, hw check)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════╗"
echo "  ║   JARVIS SENSES — Clean Start   ║"
echo "  ╚══════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Kill ALL Jarvis-related processes ─────────────────────────────────────

echo -e "${YELLOW}→${NC} Killing any existing Jarvis processes..."

KILLED=0

kill_matching() {
    local pattern="$1"
    local label="$2"
    while IFS= read -r line; do
        local pid="${line%% *}"
        local cmdline="${line#* }"
        if [ -n "$pid" ] && [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            # Skip SSH sessions whose args happen to contain our pattern
            case "$cmdline" in ssh*|*/ssh\ *) continue ;; esac
            echo -e "  ${RED}✗${NC} Killing ${label} (PID $pid)"
            kill -9 "$pid" 2>/dev/null || true
            KILLED=$((KILLED + 1))
        fi
    done < <(pgrep -af "$pattern" 2>/dev/null || true)
}

# Python processes running from this directory (main.py, setup imports, etc.)
kill_matching "python.*main\.py" "python main.py"
kill_matching "python3.*main\.py" "python3 main.py"

# Any python loading our senses modules (catches mid-startup kills)
kill_matching "python.*senses" "senses module"

# Chromium kiosk (any instance pointing at 8080 or launched with --kiosk)
kill_matching "chromium.*localhost:8080" "chromium kiosk"
kill_matching "chromium.*--kiosk" "chromium kiosk"

# Orphaned aiohttp/arecord that might hold the port or mic
kill_matching "arecord.*hw:3" "arecord"

if [ "$KILLED" -gt 0 ]; then
    echo -e "${YELLOW}→${NC} Waiting for processes to release resources..."
    sleep 2
    echo -e "${GREEN}✓${NC} Killed $KILLED process(es)"
else
    echo -e "${GREEN}✓${NC} No existing processes found"
fi

# ── 2. Free port 8080 (catch anything we missed) ────────────────────────────

for PORT in 8080; do
    if command -v fuser &>/dev/null; then
        fuser -k "${PORT}/tcp" 2>/dev/null && \
            echo -e "  ${YELLOW}⚠${NC} Freed port $PORT via fuser" || true
    else
        PID_ON_PORT=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
        if [ -n "$PID_ON_PORT" ]; then
            echo -e "  ${YELLOW}⚠${NC} Port $PORT still bound by PID $PID_ON_PORT — killing"
            kill -9 "$PID_ON_PORT" 2>/dev/null || true
            sleep 1
        fi
    fi
done

# ── 3. Release Hailo NPU if locked ──────────────────────────────────────────

HAILO_DEV="/dev/hailo0"
if [ -c "$HAILO_DEV" ]; then
    HAILO_PIDS=$(fuser "$HAILO_DEV" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)
    if [ -n "$HAILO_PIDS" ]; then
        echo -e "  ${YELLOW}⚠${NC} Hailo NPU locked by PIDs: $HAILO_PIDS"
        for pid in $HAILO_PIDS; do
            echo -e "  ${RED}✗${NC} Force-killing Hailo holder (PID $pid)"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2
        echo -e "  ${GREEN}✓${NC} Hailo NPU released"
    else
        echo -e "${GREEN}✓${NC} Hailo NPU is free"
    fi
else
    echo -e "${YELLOW}⚠${NC} Hailo device not found at $HAILO_DEV (will run in stub mode)"
fi

# ── 4. Unmute USB mics ──────────────────────────────────────────────────────

for CARD_NUM in $(arecord -l 2>/dev/null | grep "^card " | grep -oP 'card \K[0-9]+'); do
    # Unmute any capture switches (name varies by mic model)
    amixer -c "$CARD_NUM" sset 'Mic Capture' unmute 2>/dev/null || true
    amixer -c "$CARD_NUM" sset 'Mic' unmute 2>/dev/null || true
    amixer -c "$CARD_NUM" sset 'Capture' unmute 2>/dev/null || true
    # Also try the numid approach for mics that only expose numids
    for NID in $(amixer -c "$CARD_NUM" contents 2>/dev/null \
        | grep -B1 "type=BOOLEAN" | grep "name=.*[Cc]apture.*[Ss]witch\|name=.*[Mm]ute" \
        | grep -oP 'numid=\K[0-9]+'); do
        amixer -c "$CARD_NUM" cset "numid=$NID" on 2>/dev/null || true
    done
    # Max out capture volume
    amixer -c "$CARD_NUM" sset 'Mic Capture' 100% 2>/dev/null || true
    amixer -c "$CARD_NUM" sset 'Mic' 100% 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Unmuted ALSA capture on card $CARD_NUM"
done

# ── 5. Setup if requested ───────────────────────────────────────────────────

if [[ "${1:-}" == "--setup" ]]; then
    echo ""
    echo -e "${CYAN}Running full setup...${NC}"
    exec "$SCRIPT_DIR/setup.sh"
fi

# ── 6. Verify venv exists ───────────────────────────────────────────────────

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠${NC} No .venv found — running setup first"
    exec "$SCRIPT_DIR/setup.sh"
fi

source .venv/bin/activate
echo -e "${GREEN}✓${NC} Activated .venv ($(python3 --version))"

# ── 7. Quick sanity check ───────────────────────────────────────────────────

echo -e "${YELLOW}→${NC} Verifying imports..."
if python3 -c "from config import SensesConfig; SensesConfig()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Config OK"
else
    echo -e "${RED}✗${NC} Config import failed — running full setup"
    exec "$SCRIPT_DIR/setup.sh"
fi

# ── 8. WiFi optimization ─────────────────────────────────────────────────────

# Disable WiFi power management — prevents throughput collapse on Pi 5.
# Without this, 2.4 GHz can drop to 40 KB/s even with the router nearby.
if command -v iw &>/dev/null; then
    sudo iw dev wlan0 set power_save off 2>/dev/null && \
        echo -e "${GREEN}✓${NC} WiFi power save disabled" || true
fi

# Warn if connected on 2.4 GHz (channels 1-14) — throughput will be terrible
WIFI_CHAN=$(nmcli -t -f active,chan dev wifi list ifname wlan0 2>/dev/null | grep '^yes:' | cut -d: -f2 || true)
if [ -n "$WIFI_CHAN" ] && [ "$WIFI_CHAN" -le 14 ] 2>/dev/null; then
    echo -e "${RED}⚠ WARNING: WiFi on 2.4 GHz (channel $WIFI_CHAN) — audio will lag!${NC}"
    echo -e "${YELLOW}  Switch to 5 GHz SSID for 270x better throughput.${NC}"
elif [ -n "$WIFI_CHAN" ]; then
    echo -e "${GREEN}✓${NC} WiFi on 5 GHz (channel $WIFI_CHAN)"
fi

# ── 9. Launch ────────────────────────────────────────────────────────────────

# Pi 5 has DRM card1 (DSI display bridge) with no vendor file.
# ONNX Runtime probes all DRM cards for GPUs and warns on missing vendor.
# Suppress since we use Hailo NPU (PCIe), not GPU, for inference.
export ORT_LOG_LEVEL=3  # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR

echo ""
echo -e "${CYAN}=== Starting Jarvis Senses ===${NC}"
echo ""
exec python3 main.py "$@"
