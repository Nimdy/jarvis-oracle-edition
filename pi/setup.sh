#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════╗"
echo "  ║   JARVIS SENSES — Pi 5 Setup    ║"
echo "  ║       Thin Sensor Node           ║"
echo "  ╚══════════════════════════════════╝"
echo -e "${NC}"

# --- Python check ---
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" &> /dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python: $($PYTHON --version)"

# --- Virtual environment ---
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
else
    echo -e "${YELLOW}→${NC} Creating virtual environment (with system site-packages)..."
    $PYTHON -m venv .venv --system-site-packages
fi
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Activated .venv"

# --- Core Dependencies ---
echo -e "${YELLOW}→${NC} Installing Python dependencies..."
pip install --upgrade pip -q 2>&1 | tail -1
pip install -r requirements.txt -q 2>&1 | tail -1
echo -e "${GREEN}✓${NC} Dependencies installed"

# --- Vision Models ---
echo ""
echo -e "${CYAN}=== Hailo Vision Models ===${NC}"

MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

download_model() {
    local path="$1"
    local url="$2"
    local name="$3"
    local size="$4"

    if [ -f "$path" ]; then
        echo -e "${GREEN}✓${NC} $name"
        return 0
    fi
    echo -e "${YELLOW}→${NC} Downloading $name ($size)..."
    if curl -fSL --progress-bar -o "$path" "$url"; then
        echo -e "${GREEN}✓${NC} $name downloaded"
        return 0
    else
        rm -f "$path"
        echo -e "${RED}✗${NC} Failed to download $name"
        return 1
    fi
}

download_model "$MODELS_DIR/yolov8s.hef" \
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.1.0/hailo10h/yolov8s.hef" \
    "Hailo YOLOv8s detection (v5.1.0)" "~18MB"

# CPU scene detection model (YOLOv8n ONNX — runs on Pi5 CPU every ~3s)
if [ ! -f "$MODELS_DIR/yolov8n.onnx" ]; then
    echo -e "${YELLOW}→${NC} YOLOv8n ONNX model not found."
    echo -e "  Export on brain: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True)\""
    echo -e "  Then copy: scp yolov8n.onnx pi:$MODELS_DIR/yolov8n.onnx"
else
    echo -e "${GREEN}✓${NC} YOLOv8n ONNX scene detector"
fi

download_model "$MODELS_DIR/scrfd_2.5g.hef" \
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/scrfd_2.5g.hef" \
    "Hailo SCRFD face detection" "~1.5MB"

download_model "$MODELS_DIR/yolov8s_pose.hef" \
    "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8s_pose.hef" \
    "Hailo YOLOv8s-Pose" "~14MB"

# --- Clean up stale models from old architecture (TTS/STT moved to brain) ---
echo ""
echo -e "${CYAN}=== Stale Model Cleanup ===${NC}"

STALE_MODELS=(
    "$MODELS_DIR/kokoro-v1.0.onnx"
    "$MODELS_DIR/voices-v1.0.bin"
    "$MODELS_DIR/en_GB-semaine-medium.onnx"
    "$MODELS_DIR/en_GB-semaine-medium.onnx.json"
    "$MODELS_DIR/en_US-amy-medium.onnx"
    "$MODELS_DIR/en_US-amy-medium.onnx.json"
    "$MODELS_DIR/ggml-base.en-q5_1.bin"
)
STALE_DIRS=(
    "$MODELS_DIR/moonshine-base"
    "$MODELS_DIR/moonshine-tiny"
)

stale_size=0
stale_found=0
for f in "${STALE_MODELS[@]}"; do
    if [ -f "$f" ]; then
        fsize=$(stat -c%s "$f" 2>/dev/null || echo 0)
        stale_size=$((stale_size + fsize))
        stale_found=$((stale_found + 1))
    fi
done
for d in "${STALE_DIRS[@]}"; do
    if [ -d "$d" ]; then
        dsize=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
        stale_size=$((stale_size + dsize))
        stale_found=$((stale_found + 1))
    fi
done

if [ "$stale_found" -gt 0 ]; then
    stale_mb=$((stale_size / 1024 / 1024))
    echo -e "${YELLOW}⚠${NC} Found $stale_found stale model(s) (~${stale_mb}MB) from old TTS/STT architecture"
    echo -e "  TTS and STT now run on the brain GPU. These files are unused."
    read -p "  Remove stale models? [y/N] " -r
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        for f in "${STALE_MODELS[@]}"; do
            [ -f "$f" ] && rm -f "$f" && echo -e "  ${GREEN}✓${NC} Removed $(basename "$f")"
        done
        for d in "${STALE_DIRS[@]}"; do
            [ -d "$d" ] && rm -rf "$d" && echo -e "  ${GREEN}✓${NC} Removed $(basename "$d")/"
        done
        echo -e "${GREEN}✓${NC} Freed ~${stale_mb}MB"
    else
        echo -e "  Skipped — models kept"
    fi
else
    echo -e "${GREEN}✓${NC} No stale models found"
fi

# --- Hardware checks ---
echo ""
echo -e "${CYAN}=== Hardware Checks ===${NC}"

if command -v rpicam-hello &> /dev/null; then
    echo -e "${GREEN}✓${NC} Camera tools (rpicam)"
else
    echo -e "${YELLOW}⚠${NC} rpicam tools not found — camera may not work"
fi

if command -v hailortcli &> /dev/null; then
    echo -e "${GREEN}✓${NC} HailoRT CLI"
    if hailortcli fw-control identify &> /dev/null; then
        echo -e "${GREEN}✓${NC} Hailo device detected"
    else
        echo -e "${YELLOW}⚠${NC} Hailo device not responding (may need reboot)"
    fi
else
    echo -e "${YELLOW}⚠${NC} HailoRT not found — detection disabled"
fi

if command -v arecord &> /dev/null; then
    MIC_COUNT=$(arecord -l 2>/dev/null | grep -c "^card" || true)
    echo -e "${GREEN}✓${NC} Audio: $MIC_COUNT input device(s)"
else
    echo -e "${YELLOW}⚠${NC} ALSA tools not found"
fi

if [ -e /dev/fb0 ] || [ -e /dev/dri/card0 ]; then
    echo -e "${GREEN}✓${NC} Display framebuffer available"
else
    echo -e "${YELLOW}⚠${NC} No display detected (headless mode)"
fi

# --- Summary ---
echo ""
echo -e "${CYAN}=== Pi Role ===${NC}"
echo -e "  Mode: ${CYAN}Thin Sensor Node${NC}"
echo -e "  Streams raw mic audio to brain via WebSocket binary frames"
echo -e "  Vision processing via Hailo AI HAT+ (local)"
echo -e "  Audio playback of brain-synthesized speech"
echo -e "  All audio intelligence (wake word, VAD, STT, TTS) runs on brain"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${CYAN}=== Launching via start.sh (clean start) ===${NC}"
echo ""
exec "$SCRIPT_DIR/start.sh" "$@"
