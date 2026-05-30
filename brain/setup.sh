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
echo "  ║   JARVIS BRAIN — Laptop Setup    ║"
echo "  ║         Python Edition           ║"
echo "  ╚══════════════════════════════════╝"
echo -e "${NC}"

# --- Kill existing brain processes ------------------------------------------
KILLED_ANY=0

# Supervisor
if pgrep -f "python.*jarvis-supervisor" > /dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Stopping supervisor..."
    pkill -f "python.*jarvis-supervisor" 2>/dev/null || true
    KILLED_ANY=1
fi

# Brain main.py
if pgrep -f "python.*main\.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Stopping brain process..."
    pkill -f "python.*main\.py" 2>/dev/null || true
    KILLED_ANY=1
fi

# llama-server (coder model — can hold ~46GB RAM)
if pgrep -f "llama-server" > /dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Stopping llama-server..."
    pkill -f "llama-server" 2>/dev/null || true
    KILLED_ANY=1
fi

if [ $KILLED_ANY -eq 1 ]; then
    sleep 2
    # Force-kill anything still hanging
    for PAT in "python.*jarvis-supervisor" "python.*main\.py" "llama-server"; do
        if pgrep -f "$PAT" > /dev/null 2>&1; then
            pkill -9 -f "$PAT" 2>/dev/null || true
        fi
    done
    sleep 1
    echo -e "${GREEN}✓${NC} Previous processes stopped"
fi

# Free ports if something is holding them
for PORT in 9100 9200 8081; do
    PID=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}→${NC} Freeing port $PORT (pid $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 1
    fi
done

# Unload all Ollama models from VRAM to start clean
if command -v ollama &> /dev/null && pgrep -x "ollama" > /dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Unloading Ollama models from VRAM..."
    for M in $(curl -s http://localhost:11434/api/ps 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for m in d.get('models', []):
        print(m['name'])
except: pass
" 2>/dev/null); do
        curl -s http://localhost:11434/api/generate -d "{\"model\":\"$M\",\"prompt\":\"\",\"keep_alive\":0}" > /dev/null 2>&1
        echo -e "  Unloaded ${CYAN}$M${NC}"
    done
    echo -e "${GREEN}✓${NC} VRAM cleared"
fi

# --- Python ----------------------------------------------------------------
if command -v python3 &> /dev/null; then
    PYTHON=python3
    PY_VERSION=$($PYTHON --version 2>&1)
    echo -e "${GREEN}✓${NC} Python: ${PY_VERSION}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

if ! $PYTHON -c "import ensurepip" 2>/dev/null; then
    echo -e "${YELLOW}→${NC} Installing python3-venv..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip
    echo -e "${GREEN}✓${NC} python3-venv installed"
fi

if ! command -v pdftotext &>/dev/null; then
    echo -e "${YELLOW}⚠${NC} pdftotext not found (optional, for PDF text extraction)"
    echo -e "  Install with: ${CYAN}sudo apt install poppler-utils${NC}"
else
    echo -e "${GREEN}✓${NC} pdftotext available"
fi

if ! command -v aria2c &>/dev/null; then
    echo -e "${YELLOW}→${NC} Installing aria2 (fast multi-connection downloader for large models)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq aria2
    echo -e "${GREEN}✓${NC} aria2 installed"
else
    echo -e "${GREEN}✓${NC} aria2 available"
fi

# --- Virtual environment ---------------------------------------------------
VENV_DIR="$SCRIPT_DIR/.venv"

if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${YELLOW}→${NC} Removing broken virtual environment..."
    rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment exists"
else
    echo -e "${YELLOW}→${NC} Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}✗ Failed to create virtual environment.${NC}"
        echo -e "  Try: ${CYAN}sudo apt install python3-venv python3-pip${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓${NC} Activated: $(which python)"

# Make CUDA/cuDNN libs from pip-installed NVIDIA wheels visible to ONNX Runtime
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, site
paths = []
for base in site.getsitepackages():
    n = os.path.join(base, 'nvidia')
    if os.path.isdir(n):
        for root, dirs, files in os.walk(n):
            if root.endswith('/lib'):
                paths.append(root)
print(':'.join(paths))
PY
):${LD_LIBRARY_PATH}"

if [ -n "${LD_LIBRARY_PATH}" ]; then
    echo -e "${GREEN}✓${NC} LD_LIBRARY_PATH configured for NVIDIA runtime wheels"
fi

# Ensure pip exists inside the venv (some Debian/Ubuntu systems skip it)
if ! python -m pip --version &>/dev/null; then
    echo -e "${YELLOW}→${NC} Bootstrapping pip into venv..."
    python -m ensurepip --upgrade 2>/dev/null \
        || $PYTHON -m ensurepip --upgrade 2>/dev/null \
        || { echo -e "${RED}✗ Cannot install pip. Run: sudo apt install python3-pip python3-venv${NC}"; exit 1; }
fi

# --- Core Dependencies (fast, lightweight) ---------------------------------
echo ""
echo -e "${YELLOW}→${NC} Upgrading pip..."
python -m pip install --upgrade pip -q
echo -e "${YELLOW}→${NC} Installing core Python dependencies..."
python -m pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Core dependencies installed"

# --- Heavy ML Dependencies (show progress) --------------------------------
echo ""
echo -e "${CYAN}=== ML Dependencies ===${NC}"
echo -e "  These may take 5-15 minutes on first install."
echo ""

install_pkg() {
    local pkg="$1"
    local desc="$2"
    local check_import="$3"

    if [ -n "$check_import" ] && python -c "import $check_import" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $desc already installed"
        return 0
    fi
    echo -e "${YELLOW}→${NC} Installing $desc..."
    if python -m pip install "$pkg"; then
        echo -e "${GREEN}✓${NC} $desc installed"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $desc failed (feature disabled, non-fatal)"
        return 1
    fi
}

configure_nvidia_runtime_paths() {
    export LD_LIBRARY_PATH="$(python - <<'PY'
import os, site
paths = []
for base in site.getsitepackages():
    n = os.path.join(base, 'nvidia')
    if os.path.isdir(n):
        for root, dirs, files in os.walk(n):
            if root.endswith('/lib'):
                paths.append(root)
print(':'.join(paths))
PY
):${LD_LIBRARY_PATH}"
}

check_onnx_gpu() {
    python - <<'PY'
import onnxruntime as ort
provs = ort.get_available_providers()
assert 'CUDAExecutionProvider' in provs, provs
print("  ORT providers:", provs)
PY
}

# PyTorch (needed by policy layer + sentence-transformers)
install_pkg "torch>=2.0" "PyTorch (neural policy)" "torch"

# CUDA 12 cuBLAS — ctranslate2 (used by faster-whisper) links against libcublas.so.12.
# Newer PyTorch may pull nvidia-cublas 13+ which lacks the .so.12 symlink.
# Installing cu12 alongside cu13 is safe and keeps both runtimes happy.
install_pkg "nvidia-cublas-cu12" "CUDA 12 cuBLAS (for faster-whisper/ctranslate2)" ""

# Sentence Transformers (needed by semantic memory)
install_pkg "sentence-transformers>=3.0" "Sentence Transformers (semantic memory)" "sentence_transformers"

# sqlite-vec (needed by vector store)
install_pkg "sqlite-vec>=0.1.6" "sqlite-vec (vector DB)" "sqlite_vec"

# Transformers (for multimodal, optional)
install_pkg "transformers>=4.40" "Transformers (multimodal)" "transformers"

# Anthropic (optional, controlled by ENABLE_CLAUDE)
install_pkg "anthropic>=0.40" "Anthropic SDK (optional)" "anthropic"

# OpenAI (optional, for self-improvement)
install_pkg "openai>=1.0" "OpenAI SDK (optional)" "openai"

# onnxruntime-gpu MUST be installed BEFORE packages that depend on onnxruntime
# (faster-whisper, kokoro-onnx) to prevent the CPU-only onnxruntime from
# partially installing and clobbering the GPU version's files.
#
# CPU onnxruntime can sneak in as a dependency of faster-whisper/kokoro-onnx.
# If both packages coexist, Python may import the CPU one → CUDAExecutionProvider
# missing → check fails → triggers a 1.7GB re-download. Clean it pre-emptively.
if python -m pip show onnxruntime >/dev/null 2>&1 && python -m pip show onnxruntime-gpu >/dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Removing CPU onnxruntime (conflicts with GPU version)..."
    python -m pip uninstall -y onnxruntime 2>/dev/null || true
    configure_nvidia_runtime_paths
fi

if check_onnx_gpu >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} onnxruntime-gpu already available"
else
    echo -e "${YELLOW}→${NC} Installing onnxruntime-gpu (CUDA + TensorRT providers)..."
    python -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    rm -rf "$VENV_DIR"/lib/python3.*/site-packages/onnxruntime* 2>/dev/null || true
    python -m pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.20" \
        && configure_nvidia_runtime_paths \
        && check_onnx_gpu \
        && echo -e "${GREEN}✓${NC} onnxruntime-gpu + CUDA/cuDNN runtime installed" \
        || echo -e "${YELLOW}⚠${NC} onnxruntime-gpu failed — TTS will use CPU (non-fatal)"
fi

# faster-whisper (GPU-accelerated speech-to-text for Pi audio clips)
install_pkg "faster-whisper>=1.1" "faster-whisper (GPU STT)" "faster_whisper"

# openWakeWord (CPU-based wake word detection — runs on brain now)
# Install without deps to avoid tflite-runtime (no Python 3.12 wheels).
# We use inference_framework="onnx" with the already-installed onnxruntime.
if python -c "import openwakeword" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} openWakeWord (wake word) already installed"
else
    echo -e "${YELLOW}→${NC} Installing openWakeWord (wake word)..."
    python -m pip install --no-deps "openwakeword>=0.6" \
        && echo -e "${GREEN}✓${NC} openWakeWord (wake word) installed" \
        || echo -e "${YELLOW}⚠${NC} openWakeWord failed (feature disabled, non-fatal)"
fi

# Kokoro TTS (GPU-accelerated text-to-speech for brain-side synthesis)
install_pkg "kokoro-onnx>=0.4.0" "Kokoro ONNX TTS" "kokoro_onnx"

# SpeechBrain (speaker identification on GPU)
install_pkg "speechbrain>=1.0" "SpeechBrain (speaker ID)" "speechbrain"

# Final onnxruntime-gpu integrity check — other packages may have pulled in
# CPU-only onnxruntime as a dependency, partially overwriting the GPU version.
# This MUST run before OWW model download since download_models() imports
# onnxruntime internally and will fail if the GPU clobber left it broken.
# Step 1: evict CPU onnxruntime if it snuck back in
if python -m pip show onnxruntime >/dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Removing CPU onnxruntime (reinstalled by dependency)..."
    python -m pip uninstall -y onnxruntime 2>/dev/null || true
fi

if ! python -c "import onnxruntime; assert hasattr(onnxruntime, 'InferenceSession')" 2>/dev/null; then
    echo -e "${YELLOW}→${NC} Repairing onnxruntime-gpu (dependency conflict detected)..."
    python -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    rm -rf "$VENV_DIR"/lib/python3.*/site-packages/onnxruntime* 2>/dev/null || true
    python -m pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.20" \
        && configure_nvidia_runtime_paths \
        && check_onnx_gpu \
        && echo -e "${GREEN}✓${NC} onnxruntime-gpu repaired" \
        || echo -e "${YELLOW}⚠${NC} onnxruntime-gpu repair failed (non-fatal)"
elif ! check_onnx_gpu >/dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Reconfiguring NVIDIA runtime paths for onnxruntime-gpu..."
    configure_nvidia_runtime_paths
    if check_onnx_gpu >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} onnxruntime-gpu runtime paths verified"
    else
        echo -e "${YELLOW}⚠${NC} onnxruntime-gpu installed but CUDAExecutionProvider is not active"
    fi
fi

# Download OWW ONNX model files (bundled models are not included with --no-deps).
# Placed AFTER the onnxruntime-gpu integrity check because download_models()
# imports onnxruntime internally. If faster-whisper/kokoro/speechbrain clobbered
# onnxruntime-gpu with CPU-only onnxruntime, the download fails silently.
if python -c "
import os, glob, openwakeword
model_dir = os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models')
onnx_files = glob.glob(os.path.join(model_dir, '*.onnx'))
exit(0 if len(onnx_files) >= 1 else 1)
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} openWakeWord ONNX models present"
else
    echo -e "${YELLOW}→${NC} Downloading openWakeWord ONNX models..."
    python -c "from openwakeword.utils import download_models; download_models()" \
        && echo -e "${GREEN}✓${NC} openWakeWord models downloaded" \
        || echo -e "${YELLOW}⚠${NC} openWakeWord model download failed (non-fatal)"
fi

# --- Download Kokoro TTS model files --------------------------------------
echo ""
echo -e "${CYAN}=== TTS Models ===${NC}"
JARVIS_MODELS_DIR="$HOME/.jarvis/models"
mkdir -p "$JARVIS_MODELS_DIR"

KOKORO_MODEL="$JARVIS_MODELS_DIR/kokoro-v1.0.onnx"
KOKORO_VOICES="$JARVIS_MODELS_DIR/voices-v1.0.bin"
KOKORO_BASE_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

if [ -f "$KOKORO_MODEL" ] && [ -f "$KOKORO_VOICES" ]; then
    echo -e "${GREEN}✓${NC} Kokoro TTS model files present"
else
    echo -e "${YELLOW}→${NC} Downloading Kokoro TTS model files..."
    if [ ! -f "$KOKORO_MODEL" ]; then
        curl -fSL -o "$KOKORO_MODEL" "$KOKORO_BASE_URL/kokoro-v1.0.onnx" \
            && echo -e "  ${GREEN}✓${NC} kokoro-v1.0.onnx downloaded" \
            || echo -e "  ${YELLOW}⚠${NC} Failed to download kokoro model (non-fatal)"
    fi
    if [ ! -f "$KOKORO_VOICES" ]; then
        curl -fSL -o "$KOKORO_VOICES" "$KOKORO_BASE_URL/voices-v1.0.bin" \
            && echo -e "  ${GREEN}✓${NC} voices-v1.0.bin downloaded" \
            || echo -e "  ${YELLOW}⚠${NC} Failed to download voices file (non-fatal)"
    fi
fi

# --- .env ------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        echo -e "${YELLOW}→${NC} Created .env from .env.example — edit it with your settings"
    fi
fi

# Source .env so setup.sh can read user settings (model flags, ports, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# --- GPU detection + VRAM-aware tier selection (needed by STT + model pull) -
echo ""
echo -e "${CYAN}=== Hardware Profile ===${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo -e "${GREEN}✓${NC} GPU: ${GPU_NAME} (${GPU_MEM_MB} MB VRAM)"
else
    GPU_MEM_MB=0
    echo -e "${YELLOW}⚠${NC} No NVIDIA GPU detected — will use CPU-only models"
fi

TIER="${JARVIS_GPU_TIER:-}"
if [ -z "$TIER" ]; then
    if [ "$GPU_MEM_MB" -lt 4000 ] 2>/dev/null; then TIER="minimal"
    elif [ "$GPU_MEM_MB" -lt 6000 ] 2>/dev/null; then TIER="low"
    elif [ "$GPU_MEM_MB" -lt 8000 ] 2>/dev/null; then TIER="medium"
    elif [ "$GPU_MEM_MB" -lt 12000 ] 2>/dev/null; then TIER="high"
    elif [ "$GPU_MEM_MB" -lt 16500 ] 2>/dev/null; then TIER="premium"
    elif [ "$GPU_MEM_MB" -lt 24500 ] 2>/dev/null; then TIER="ultra"
    else TIER="extreme"
    fi
    echo -e "  Auto-detected tier: ${CYAN}${TIER}${NC}"
else
    echo -e "  Tier override: ${CYAN}${TIER}${NC}"
fi

# --- Pre-download faster-whisper model (tier-aware) -----------------------
if [ -z "${STT_MODEL:-}" ]; then
    case "$TIER" in
        minimal) STT_MODEL="tiny";;
        low)     STT_MODEL="small";;
        medium)  STT_MODEL="medium";;
        high)    STT_MODEL="large-v3-turbo";;
        premium) STT_MODEL="large-v3";;
        *)       STT_MODEL="large-v3";;
    esac
fi

echo ""
echo -e "${CYAN}=== STT Model ===${NC}"
echo -e "  Pre-downloading faster-whisper model: ${CYAN}${STT_MODEL}${NC} (tier: ${TIER})"
python -c "
import os
dl_root = os.path.expanduser('~/.jarvis/models/faster-whisper')
os.makedirs(dl_root, exist_ok=True)
try:
    from faster_whisper import WhisperModel
    print('  Downloading/verifying model (this may take a minute on first run)...')
    m = WhisperModel('${STT_MODEL}', device='cpu', compute_type='int8', download_root=dl_root)
    del m
    print('  ✓ Model ${STT_MODEL} ready')
except ImportError:
    print('  ⚠ faster-whisper not available — skipping model download')
except Exception as e:
    print(f'  ⚠ Model download issue: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠${NC} Could not pre-download STT model (non-fatal)"

# --- Pre-download speaker ID model (ECAPA-TDNN) ---------------------------
echo ""
echo -e "${CYAN}=== Speaker ID Model ===${NC}"
echo -e "  Pre-downloading ECAPA-TDNN speaker verification model..."
python -c "
import huggingface_hub as _hfh
_orig_dl = _hfh.hf_hub_download
def _compat_dl(*args, **kwargs):
    kwargs.pop('use_auth_token', None)
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else '')
    try:
        return _orig_dl(*args, **kwargs)
    except Exception as e:
        if '404' in str(e) and 'custom' in str(filename):
            raise ValueError(f'{filename} not found in repo (expected, non-fatal)')
        raise
_hfh.hf_hub_download = _compat_dl

import os
save_dir = os.path.expanduser('~/.jarvis/models/ecapa-tdnn')
os.makedirs(save_dir, exist_ok=True)
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['ffmpeg']
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=save_dir, run_opts={'device': 'cpu'})
    del m
    print('  ✓ ECAPA-TDNN model cached locally')
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
        m = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=save_dir, run_opts={'device': 'cpu'})
        del m
        print('  ✓ ECAPA-TDNN model cached locally (legacy API)')
    except ImportError:
        print('  ⚠ SpeechBrain not available — skipping speaker ID model')
except ValueError as e:
    if 'custom' in str(e):
        print('  ✓ ECAPA-TDNN core files cached (custom.py not in repo — normal)')
    else:
        print(f'  ⚠ Speaker ID model issue: {e}')
except Exception as e:
    print(f'  ⚠ Speaker ID model download issue: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠${NC} Could not pre-download speaker ID model (non-fatal)"

# --- Pre-download face embedding model (MobileFaceNet/ArcFace ONNX) -------
FACE_MODEL_DIR="$HOME/.jarvis/models"
FACE_MODEL="$FACE_MODEL_DIR/mobilefacenet.onnx"
mkdir -p "$FACE_MODEL_DIR"

if [ -f "$FACE_MODEL" ]; then
    FACE_SZ=$(du -h "$FACE_MODEL" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Face embedding model already cached ($FACE_SZ)"
else
    echo -e "  Downloading MobileFaceNet (w600k_mbf.onnx)..."
    FACE_OK=0

    HF_URL="https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx"
    echo -e "    Downloading from HuggingFace..."
    curl -sS -L --fail --connect-timeout 10 --max-time 120 -o "$FACE_MODEL" "$HF_URL" 2>/dev/null && FACE_OK=1
    if [ $FACE_OK -eq 1 ]; then
        echo -e "    ${GREEN}✓${NC} Downloaded w600k_mbf.onnx"
    else
        rm -f "$FACE_MODEL"
    fi

    if [ $FACE_OK -eq 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Could not download face embedding model (non-fatal)"
        echo -e "    Manual: search HuggingFace for 'insightface buffalo w600k_mbf'"
        echo -e "    Save w600k_mbf.onnx → $FACE_MODEL"
    else
        FACE_SZ=$(du -h "$FACE_MODEL" | cut -f1)
        echo -e "  ${GREEN}✓${NC} Face embedding model cached ($FACE_SZ)"
    fi
fi

# --- Pre-download audio emotion model (wav2vec2) --------------------------
echo -e "  Pre-downloading wav2vec2 emotion classification model..."
python -c "
import os
cache_dir = os.path.expanduser('~/.jarvis/models/huggingface')
os.makedirs(cache_dir, exist_ok=True)
try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    name = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
    AutoFeatureExtractor.from_pretrained(name, cache_dir=cache_dir)
    AutoModelForAudioClassification.from_pretrained(name, cache_dir=cache_dir)
    print('  ✓ wav2vec2 emotion model cached locally')
except ImportError:
    print('  ⚠ transformers not available — skipping emotion model')
except Exception as e:
    print(f'  ⚠ Emotion model download issue: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠${NC} Could not pre-download emotion model (non-fatal)"

# --- Pre-download sentence-transformers embedding model --------------------
echo -e "  Pre-downloading sentence-transformers embedding model..."
python -c "
import os
cache_dir = os.path.expanduser('~/.jarvis/models/huggingface')
os.makedirs(cache_dir, exist_ok=True)
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('all-MiniLM-L6-v2', device='cpu', cache_folder=cache_dir)
    print('  ✓ all-MiniLM-L6-v2 cached locally')
except ImportError:
    print('  ⚠ sentence-transformers not available — skipping')
except Exception as e:
    print(f'  ⚠ Embedding model download issue: {e}')
" 2>/dev/null || echo -e "${YELLOW}⚠${NC} Could not pre-download embedding model (non-fatal)"

# --- Ollama ----------------------------------------------------------------
echo ""
echo -e "${CYAN}=== Ollama Setup ===${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama installed"
else
    echo -e "${YELLOW}→${NC} Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓${NC} Ollama installed"
fi

if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo -e "${YELLOW}→${NC} Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

case "$TIER" in
    minimal) DEF_MODEL="qwen3:1.7b"; DEF_FAST="qwen3:1.7b"; DEF_VISION="";;
    low)     DEF_MODEL="qwen3:4b";   DEF_FAST="qwen3:1.7b"; DEF_VISION="";;
    medium)  DEF_MODEL="qwen3:8b";   DEF_FAST="qwen3:4b";   DEF_VISION="";;
    high)    DEF_MODEL="qwen3:8b";   DEF_FAST="qwen3:4b";   DEF_VISION="qwen2.5vl:7b";;
    premium) DEF_MODEL="qwen3:8b";   DEF_FAST="qwen3:8b";   DEF_VISION="qwen2.5vl:7b";;
    ultra)   DEF_MODEL="qwen3:14b";  DEF_FAST="qwen3:8b";   DEF_VISION="qwen2.5vl:7b";;
    extreme) DEF_MODEL="qwen3:32b";  DEF_FAST="qwen3:14b";  DEF_VISION="qwen2.5vl:7b";;
    *)       DEF_MODEL="qwen3:8b";   DEF_FAST="qwen3:4b";   DEF_VISION="qwen2.5vl:7b";;
esac

MODEL="${OLLAMA_MODEL:-$DEF_MODEL}"
FAST_MODEL="${OLLAMA_FAST_MODEL:-$DEF_FAST}"
VISION_MODEL="${OLLAMA_VISION_MODEL:-$DEF_VISION}"

echo -e "  LLM model: ${CYAN}${MODEL}${NC}"
echo -e "  Fast model: ${CYAN}${FAST_MODEL}${NC}"
echo -e "  Vision model: ${CYAN}${VISION_MODEL:-disabled}${NC}"

pull_model() {
    local m="$1"
    local label="$2"
    if [ -z "$m" ]; then return; fi
    echo -e "  Checking ${label}: ${CYAN}${m}${NC}"
    if ollama list 2>/dev/null | grep -q "$m"; then
        echo -e "  ${GREEN}✓${NC} ${m} available"
    else
        echo -e "  ${YELLOW}→${NC} Pulling ${m} (this may take a while)..."
        ollama pull "$m"
        echo -e "  ${GREEN}✓${NC} ${m} downloaded"
    fi
}

pull_model "$MODEL" "primary"
if [ "$FAST_MODEL" != "$MODEL" ]; then
    pull_model "$FAST_MODEL" "fast"
fi
pull_model "$VISION_MODEL" "vision"

# --- Memory directory ------------------------------------------------------
mkdir -p "$HOME/.jarvis"

# --- Coder Model (Qwen3-Coder-Next via llama-server) ----------------------
echo ""
echo -e "${CYAN}=== Self-Improvement Coder ===${NC}"

CODER_MODEL_DIR="$HOME/.jarvis/models"
CODER_REPO="unsloth/Qwen3-Coder-Next-GGUF"
CODER_BASE_URL="https://huggingface.co/${CODER_REPO}/resolve/main"

# Detect total system RAM (GB)
TOTAL_RAM_GB=0
if [ -f /proc/meminfo ]; then
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
elif command -v sysctl &> /dev/null; then
    TOTAL_RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    TOTAL_RAM_GB=$(( TOTAL_RAM_BYTES / 1024 / 1024 / 1024 ))
fi
echo -e "  System RAM: ${CYAN}${TOTAL_RAM_GB}GB${NC}"

# RAM-aware quant selection (must leave headroom for OS + Jarvis + Ollama)
#   56GB+ → UD-Q4_K_XL  (~46GB model, best quality)
#   48GB+ → UD-IQ4_XS   (~38GB model, good quality)
#   32GB+ → UD-IQ2_M    (~25GB model, acceptable quality)
#   <32GB → disabled (would OOM)
CODER_MIN_RAM=32
if [ -n "${CODER_MODEL_PATH:-}" ]; then
    CODER_MODEL="$CODER_MODEL_PATH"
    CODER_MODEL_NAME="$(basename "$CODER_MODEL")"
    CODER_QUANT="custom"
elif [ "$TOTAL_RAM_GB" -ge 56 ]; then
    CODER_MODEL_NAME="Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
    CODER_MODEL="$CODER_MODEL_DIR/$CODER_MODEL_NAME"
    CODER_QUANT="UD-Q4_K_XL (~46GB)"
elif [ "$TOTAL_RAM_GB" -ge 48 ]; then
    CODER_MODEL_NAME="Qwen3-Coder-Next-UD-IQ4_XS.gguf"
    CODER_MODEL="$CODER_MODEL_DIR/$CODER_MODEL_NAME"
    CODER_QUANT="UD-IQ4_XS (~38GB)"
elif [ "$TOTAL_RAM_GB" -ge "$CODER_MIN_RAM" ]; then
    CODER_MODEL_NAME="Qwen3-Coder-Next-UD-IQ2_M.gguf"
    CODER_MODEL="$CODER_MODEL_DIR/$CODER_MODEL_NAME"
    CODER_QUANT="UD-IQ2_M (~25GB)"
else
    CODER_MODEL_NAME=""
    CODER_MODEL=""
    CODER_QUANT="none"
fi

CODER_MODEL_URL="${CODER_BASE_URL}/${CODER_MODEL_NAME}"

# SHA256 hashes from HuggingFace LFS for integrity verification
declare -A CODER_SHA256=(
    ["Qwen3-Coder-Next-UD-Q4_K_XL.gguf"]="4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36"
    ["Qwen3-Coder-Next-UD-IQ4_XS.gguf"]="abf56d7fe8a0a99c15d220c13de4aa57b69cfba6ef4c2a007b56e34d7b40cd11"
    ["Qwen3-Coder-Next-UD-IQ2_M.gguf"]="f9d82aaa687a1c50734c0e679e06fe8f6fa175b05676a9069eba633342d2d876"
)
declare -A CODER_EXPECTED_SIZE=(
    ["Qwen3-Coder-Next-UD-Q4_K_XL.gguf"]="49608478720"
    ["Qwen3-Coder-Next-UD-IQ4_XS.gguf"]="38429272064"
    ["Qwen3-Coder-Next-UD-IQ2_M.gguf"]="24962293760"
)
CODER_EXPECTED_SHA="${CODER_SHA256[$CODER_MODEL_NAME]:-}"
CODER_EXPECTED_SZ="${CODER_EXPECTED_SIZE[$CODER_MODEL_NAME]:-}"

verify_coder_model() {
    local model_path="$1"
    local expected_sha="$2"
    local expected_size="$3"
    local force_hash="${4:-false}"

    if [ ! -f "$model_path" ]; then
        echo "missing"
        return 1
    fi

    local actual_size
    actual_size=$(stat -c%s "$model_path" 2>/dev/null || stat -f%z "$model_path" 2>/dev/null)
    if [ -n "$expected_size" ] && [ "$actual_size" != "$expected_size" ]; then
        echo "size_mismatch:expected=${expected_size},actual=${actual_size}"
        return 1
    fi

    if [ -n "$expected_sha" ]; then
        # Cache SHA256 results keyed on path + size + mtime to avoid
        # re-hashing a ~46GB file on every setup/restart.
        local cache_file="${model_path}.sha256ok"
        local actual_mtime
        actual_mtime=$(stat -c%Y "$model_path" 2>/dev/null || stat -f%m "$model_path" 2>/dev/null)
        local cache_key="${actual_size}:${actual_mtime}:${expected_sha}"

        if [ "$force_hash" != "true" ] && [ -f "$cache_file" ]; then
            local cached_key
            cached_key=$(cat "$cache_file" 2>/dev/null)
            if [ "$cached_key" = "$cache_key" ]; then
                echo -e "  ${GREEN}✓${NC} SHA256 cached — file unchanged since last verification" >&2
                echo "ok"
                return 0
            fi
        fi

        echo -e "  ${YELLOW}→${NC} Verifying SHA256 (this takes a moment for large files)..." >&2
        local actual_sha
        actual_sha=$(sha256sum "$model_path" | awk '{print $1}')
        if [ "$actual_sha" != "$expected_sha" ]; then
            rm -f "$cache_file"
            echo "sha256_mismatch:expected=${expected_sha:0:16}...,actual=${actual_sha:0:16}..."
            return 1
        fi

        # Write cache — next run skips the hash if file hasn't changed
        echo "$cache_key" > "$cache_file"
    fi

    echo "ok"
    return 0
}

download_coder_model() {
    local url="$1"
    local target="$2"
    local model_name="$3"

    if command -v aria2c &>/dev/null; then
        echo -e "  ${YELLOW}→${NC} Downloading with aria2c (16 connections)..."
        aria2c -x 16 -s 16 -c \
            -d "$(dirname "$target")" \
            -o "$(basename "$target")" \
            "$url" \
            && return 0
        echo -e "  ${YELLOW}⚠${NC} aria2c failed, falling back to curl..."
    fi

    echo -e "  ${YELLOW}→${NC} Downloading with curl (resume supported)..."
    curl -fL# -C - -o "$target" "$url" \
        && return 0

    echo -e "  ${YELLOW}⚠${NC} curl failed — trying huggingface-cli..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$CODER_REPO" \
            "$model_name" \
            --local-dir "$(dirname "$target")" \
            && return 0
    fi

    return 1
}

ENABLE_CODER="${ENABLE_CODER_MODEL:-false}"
CODER_STATUS="disabled"

if [ "$ENABLE_CODER" = "true" ]; then
    if [ "$TOTAL_RAM_GB" -lt "$CODER_MIN_RAM" ]; then
        echo -e "  ${RED}✗${NC} System has ${TOTAL_RAM_GB}GB RAM — minimum ${CODER_MIN_RAM}GB required"
        echo -e "    Coder model would OOM your system. Skipping."
        echo -e "    Upgrade to 32GB+ RAM or set ENABLE_CODER_MODEL=false"
        CODER_STATUS="insufficient-ram"
    else
        echo -e "  Selected quant: ${CYAN}${CODER_QUANT}${NC} (for ${TOTAL_RAM_GB}GB RAM)"
        mkdir -p "$CODER_MODEL_DIR"

        NEED_DOWNLOAD=false
        if [ -f "$CODER_MODEL" ]; then
            VERIFY_RESULT=$(verify_coder_model "$CODER_MODEL" "$CODER_EXPECTED_SHA" "$CODER_EXPECTED_SZ")
            if [ "$VERIFY_RESULT" = "ok" ]; then
                CODER_SZ=$(du -h "$CODER_MODEL" | cut -f1)
                echo -e "  ${GREEN}✓${NC} Coder model verified ($CODER_SZ, SHA256 OK)"
                CODER_STATUS="ready"
            else
                echo -e "  ${RED}✗${NC} Coder model CORRUPT or INCOMPLETE: ${VERIFY_RESULT}"
                echo -e "  ${YELLOW}→${NC} Removing corrupt file and re-downloading..."
                rm -f "$CODER_MODEL"
                NEED_DOWNLOAD=true
            fi
        else
            NEED_DOWNLOAD=true
        fi

        if [ "$NEED_DOWNLOAD" = "true" ]; then
            echo -e "  ${YELLOW}→${NC} Downloading Qwen3-Coder-Next (${CODER_QUANT})..."
            echo -e "    Source: ${CYAN}${CODER_REPO}${NC}"
            echo -e "    Target: ${CYAN}${CODER_MODEL}${NC}"
            echo ""
            if download_coder_model "$CODER_MODEL_URL" "$CODER_MODEL" "$CODER_MODEL_NAME"; then
                VERIFY_RESULT=$(verify_coder_model "$CODER_MODEL" "$CODER_EXPECTED_SHA" "$CODER_EXPECTED_SZ" "true")
                if [ "$VERIFY_RESULT" = "ok" ]; then
                    echo -e "  ${GREEN}✓${NC} Coder model downloaded and verified (SHA256 OK)"
                    CODER_STATUS="ready"
                else
                    echo -e "  ${RED}✗${NC} Downloaded file FAILED verification: ${VERIFY_RESULT}"
                    echo -e "  ${RED}✗${NC} Removing corrupt download. Please retry."
                    rm -f "$CODER_MODEL"
                    CODER_STATUS="download-corrupt"
                fi
            else
                echo -e "  ${YELLOW}⚠${NC} All download methods failed (non-fatal). Retry with:"
                echo -e "    aria2c -x 16 -s 16 -c -d '$CODER_MODEL_DIR' -o '$CODER_MODEL_NAME' '$CODER_MODEL_URL'"
            fi
        fi

        # Write selected model path to .env so config.py picks it up
        if [ -f "$SCRIPT_DIR/.env" ] && [ "$CODER_STATUS" = "ready" ]; then
            if grep -q "^CODER_MODEL_PATH=" "$SCRIPT_DIR/.env" 2>/dev/null; then
                sed -i "s|^CODER_MODEL_PATH=.*|CODER_MODEL_PATH=$CODER_MODEL|" "$SCRIPT_DIR/.env"
            else
                echo "CODER_MODEL_PATH=$CODER_MODEL" >> "$SCRIPT_DIR/.env"
            fi
        fi

        # --- llama-server binary --------------------------------------------------
        LLAMA_SERVER_BIN="${CODER_LLAMA_SERVER:-llama-server}"
        if command -v "$LLAMA_SERVER_BIN" &> /dev/null; then
            echo -e "  ${GREEN}✓${NC} llama-server found: $(which "$LLAMA_SERVER_BIN")"
        else
            echo -e "  ${YELLOW}⚠${NC} llama-server not found in PATH"
            LLAMA_CPP_DIR="$HOME/.local/share/llama.cpp"
            LLAMA_SERVER_LOCAL="$LLAMA_CPP_DIR/build/bin/llama-server"

            if [ -x "$LLAMA_SERVER_LOCAL" ]; then
                echo -e "  ${GREEN}✓${NC} Found local build: $LLAMA_SERVER_LOCAL"
                echo -e "    Add to PATH: ${CYAN}export PATH=\"$LLAMA_CPP_DIR/build/bin:\$PATH\"${NC}"
            else
                echo -e "  ${YELLOW}→${NC} Building llama.cpp from source..."
                if ! command -v cmake &> /dev/null; then
                    echo -e "  ${YELLOW}⚠${NC} cmake not found. Installing build dependencies..."
                    sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential
                fi
                if [ -d "$LLAMA_CPP_DIR/.git" ]; then
                    echo -e "    Updating existing llama.cpp repo..."
                    git -C "$LLAMA_CPP_DIR" pull --quiet 2>/dev/null || true
                else
                    echo -e "    Cloning llama.cpp..."
                    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR" 2>/dev/null
                fi

                if [ -d "$LLAMA_CPP_DIR" ]; then
                    echo -e "    Compiling (CPU-only, this may take a few minutes)..."
                    cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" \
                        -DCMAKE_BUILD_TYPE=Release \
                        -DGGML_CUDA=OFF \
                        > /dev/null 2>&1 \
                    && cmake --build "$LLAMA_CPP_DIR/build" --config Release -j"$(nproc)" \
                        --target llama-server > /dev/null 2>&1 \
                    && {
                        echo -e "  ${GREEN}✓${NC} llama-server built: $LLAMA_SERVER_LOCAL"
                        export PATH="$LLAMA_CPP_DIR/build/bin:$PATH"
                        if [ -f "$SCRIPT_DIR/.env" ] && ! grep -q "CODER_LLAMA_SERVER" "$SCRIPT_DIR/.env"; then
                            echo "CODER_LLAMA_SERVER=$LLAMA_SERVER_LOCAL" >> "$SCRIPT_DIR/.env"
                            echo -e "    Added CODER_LLAMA_SERVER to .env"
                        fi
                    } || {
                        echo -e "  ${YELLOW}⚠${NC} Build failed (non-fatal). Install manually:"
                        echo -e "    ${CYAN}https://github.com/ggerganov/llama.cpp#build${NC}"
                        CODER_STATUS="missing-binary"
                    }
                else
                    echo -e "  ${YELLOW}⚠${NC} git clone failed. Install llama-server manually:"
                    echo -e "    ${CYAN}https://github.com/ggerganov/llama.cpp#build${NC}"
                    CODER_STATUS="missing-binary"
                fi
            fi
        fi
    fi
else
    if [ "$TOTAL_RAM_GB" -lt "$CODER_MIN_RAM" ]; then
        echo -e "  Coder model: ${YELLOW}disabled${NC} (${TOTAL_RAM_GB}GB RAM — need ${CODER_MIN_RAM}GB+)"
    else
        echo -e "  Coder model: ${YELLOW}disabled${NC} (set ENABLE_CODER_MODEL=true to enable)"
        echo -e "    Recommended quant for ${TOTAL_RAM_GB}GB RAM: ${CYAN}${CODER_QUANT}${NC}"
    fi
fi

# --- Summary ---------------------------------------------------------------
echo ""
echo -e "${CYAN}=== Engine Status ===${NC}"
echo -e "  Hardware tier: ${CYAN}${TIER}${NC} (${GPU_MEM_MB:-0}MB VRAM)"
echo -e "  Models: LLM=${MODEL} Fast=${FAST_MODEL} Vision=${VISION_MODEL:-disabled} STT=${STT_MODEL}"
if [ "$CODER_STATUS" = "ready" ]; then
    echo -e "  Coder: ${GREEN}${CODER_STATUS}${NC} (${CODER_QUANT}, ${TOTAL_RAM_GB}GB RAM)"
elif [ "$CODER_STATUS" = "insufficient-ram" ]; then
    echo -e "  Coder: ${RED}${CODER_STATUS}${NC} (${TOTAL_RAM_GB}GB RAM, need ${CODER_MIN_RAM}GB+)"
else
    echo -e "  Coder: ${CYAN}${CODER_STATUS}${NC}"
fi
python -c "
engines = []
try:
    import torch; engines.append(f'PyTorch: {torch.__version__} (GPU: {torch.cuda.is_available()})')
except: engines.append('PyTorch: not installed')
try:
    import sentence_transformers; engines.append('Semantic Memory: ready')
except: engines.append('Semantic Memory: disabled')
try:
    import anthropic; engines.append('Claude API: available')
except: engines.append('Claude API: not installed')
try:
    import openai; engines.append('OpenAI API: available')
except: engines.append('OpenAI API: not installed')
try:
    import faster_whisper; engines.append('faster-whisper STT: ready')
except: engines.append('faster-whisper STT: not installed')
try:
    import openwakeword; engines.append('openWakeWord: ready')
except: engines.append('openWakeWord: not installed')
try:
    import kokoro_onnx; engines.append('Kokoro TTS: ready')
except: engines.append('Kokoro TTS: not installed')
try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    gpu = 'GPU' if 'CUDAExecutionProvider' in provs else 'CPU-only'
    engines.append(f'ONNX Runtime: {ort.__version__} ({gpu})')
except: engines.append('ONNX Runtime: not installed')
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['ffmpeg']
    import speechbrain; engines.append('SpeechBrain (speaker ID): ready')
except: engines.append('SpeechBrain (speaker ID): not installed')
for e in engines:
    print(f'  {e}')
" 2>/dev/null || echo "  Could not check engines"

# --- Launch ----------------------------------------------------------------
echo ""
echo -e "${CYAN}=== Starting Jarvis Brain ===${NC}"
LOG_FILE="/tmp/jarvis-brain.log"
echo -e "  Logging to: ${CYAN}${LOG_FILE}${NC}"
echo ""
exec python -u main.py 2>&1 | tee -a "$LOG_FILE"