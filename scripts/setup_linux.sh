#!/usr/bin/env bash
# setup_linux.sh — Reproducible Linux build for MnemeFusion with CUDA
#
# Usage:
#   # Full setup (first time on a fresh machine):
#   bash scripts/setup_linux.sh
#
#   # Rebuild only (after git pull, skips apt/rust/cmake install):
#   bash scripts/setup_linux.sh --rebuild-only
#
#   # Multi-arch build (sm_75 + sm_86 + sm_89 + detected GPU):
#   bash scripts/setup_linux.sh --multi-arch
#
# Requirements:
#   - Ubuntu 22.04+ (or similar with apt)
#   - NVIDIA GPU with driver 525+ (for CUDA 12.x)
#   - ~10GB disk for build artifacts + model
#
# What this script does:
#   1. Installs system deps (cmake, pkg-config, libclang)
#   2. Installs Rust if missing
#   3. Detects CUDA compute capability from GPU
#   4. Applies required patches to cargo registry sources
#   5. Builds mnemefusion-python with CUDA support
#   6. Creates a Python venv and installs the wheel
#   7. Runs a smoke test to verify everything works

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Use sudo if available and not already root
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &>/dev/null; then
    SUDO="sudo"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
REBUILD_ONLY=false
MULTI_ARCH=false

for arg in "$@"; do
    case "$arg" in
        --rebuild-only) REBUILD_ONLY=true ;;
        --multi-arch) MULTI_ARCH=true ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

cd "$WORKSPACE"
info "Workspace: $WORKSPACE"

# ─────────────────────────────────────────────────────────────
# Step 1: System dependencies
# ─────────────────────────────────────────────────────────────
if [ "$REBUILD_ONLY" = false ]; then
    info "Installing system dependencies..."
    if command -v apt-get &>/dev/null; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq pkg-config libclang-dev build-essential python3-venv python3-dev
    else
        warn "Not a Debian/Ubuntu system — install pkg-config, libclang-dev, python3-venv manually"
    fi

    # cmake: need 3.28+ for CUDA 12.x. Ubuntu 22.04 ships 3.22 (too old).
    CMAKE_VERSION=$(cmake --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' || echo "0.0")
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -lt 3 ] || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 28 ]; }; then
        info "cmake $CMAKE_VERSION is too old, installing 3.31.6..."
        CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz"
        wget -q "$CMAKE_URL" -O /tmp/cmake.tar.gz
        $SUDO tar -xzf /tmp/cmake.tar.gz -C /usr/local --strip-components=1
        rm /tmp/cmake.tar.gz
        info "cmake $(cmake --version | head -1)"
    else
        info "cmake $CMAKE_VERSION is OK"
    fi

    # Rust
    if ! command -v cargo &>/dev/null; then
        info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    info "Rust: $(rustc --version)"

    # Ensure cc symlink exists (cmake 3.31+ requires it)
    if ! command -v cc &>/dev/null; then
        if command -v gcc &>/dev/null; then
            $SUDO ln -sf "$(which gcc)" /usr/bin/cc
            info "Created cc -> gcc symlink"
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Detect GPU and set CUDA environment
# ─────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found — NVIDIA driver not installed"
fi

info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Detect compute capability
CUDA_ARCH=""
if command -v nvidia-smi &>/dev/null; then
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$CC_RAW" ] && [ "$CC_RAW" != "[N/A]" ]; then
        CUDA_ARCH="$CC_RAW"
    fi
fi
# Fallback: try deviceQuery or hardcode common architectures
if [ -z "$CUDA_ARCH" ]; then
    warn "Could not auto-detect compute capability. Common values:"
    warn "  GTX 1650 Ti = 75, RTX 3050/3060/A40 = 86, RTX 4090 = 89, A100 = 80, H100 = 90"
    read -p "Enter compute capability (e.g., 86): " CUDA_ARCH
fi
# Multi-arch: build for multiple GPU architectures in one binary
if [ "$MULTI_ARCH" = true ]; then
    CUDA_ARCH="${CUDA_ARCH};75;86;89"
    # Deduplicate
    CUDA_ARCH=$(echo "$CUDA_ARCH" | tr ';' '\n' | sort -u | tr '\n' ';' | sed 's/;$//')
    info "Multi-arch CUDA build: sm_${CUDA_ARCH//;/, sm_}"
else
    info "CUDA compute capability: sm_${CUDA_ARCH}"
fi

# Find CUDA toolkit
CUDA_PATH="${CUDA_PATH:-}"
if [ -z "$CUDA_PATH" ]; then
    for candidate in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-13; do
        if [ -d "$candidate" ]; then
            CUDA_PATH="$candidate"
            break
        fi
    done
fi
if [ -z "$CUDA_PATH" ] || [ ! -d "$CUDA_PATH" ]; then
    error "CUDA toolkit not found. Set CUDA_PATH or install CUDA toolkit."
fi
info "CUDA path: $CUDA_PATH"

export CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
export LLAMA_BUILD_SHARED_LIBS=1
export CUDA_PATH
export CUDACXX="${CUDA_PATH}/bin/nvcc"

# libclang path for bindgen
if [ -d /usr/lib/x86_64-linux-gnu ]; then
    export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
elif [ -d /usr/lib64 ]; then
    export LIBCLANG_PATH=/usr/lib64
fi

# ─────────────────────────────────────────────────────────────
# Step 3: Remove .cargo/config.toml if it has Windows paths
# ─────────────────────────────────────────────────────────────
if [ -f "$WORKSPACE/.cargo/config.toml" ]; then
    if grep -qi 'C:\\' "$WORKSPACE/.cargo/config.toml" 2>/dev/null; then
        info "Removing .cargo/config.toml with Windows paths"
        rm "$WORKSPACE/.cargo/config.toml"
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 4: Fetch dependencies and apply patches
# ─────────────────────────────────────────────────────────────
info "Fetching cargo dependencies..."
cargo fetch --quiet

info "Applying dependency patches..."
python3 "$SCRIPT_DIR/apply_patches.py" --clean

# ─────────────────────────────────────────────────────────────
# Step 5: Build
# ─────────────────────────────────────────────────────────────
info "Building mnemefusion-python with CUDA support (this takes ~8-10 minutes)..."
cd "$WORKSPACE/mnemefusion-python"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    info "Created Python venv"
fi
source .venv/bin/activate

# Install Python deps
pip install -q maturin sentence-transformers

# Build and install
maturin develop --release --features entity-extraction-cuda 2>&1 | tail -20

info "Build complete!"

# ─────────────────────────────────────────────────────────────
# Step 6: Copy backend shared libraries
# ─────────────────────────────────────────────────────────────
# Find the cmake build output
BUILD_BIN=""
for d in "$WORKSPACE"/target/release/build/llama-cpp-sys-2-*/out/build/bin/; do
    if [ -d "$d" ]; then
        BUILD_BIN="$d"
    fi
done

if [ -n "$BUILD_BIN" ]; then
    info "Copying backend .so files from $BUILD_BIN to workspace root..."
    for f in "$BUILD_BIN"/*.so*; do
        if [ -f "$f" ]; then
            cp -v "$f" "$WORKSPACE/" 2>/dev/null || true
        fi
    done
else
    warn "Build output bin/ not found — backend .so files may need manual copying"
fi

# ─────────────────────────────────────────────────────────────
# Step 7: Smoke test
# ─────────────────────────────────────────────────────────────
cd "$WORKSPACE"
info "Running smoke test..."
python3 -c "
import mnemefusion
import tempfile, os
td = tempfile.mkdtemp()
db = os.path.join(td, 'smoke.mfdb')
mem = mnemefusion.Memory(db)
mem.add('Smoke test memory', [0.1] * 384)
results = mem.query('smoke test', [0.1] * 384, 5)
assert len(results) == 1, f'Expected 1 result, got {len(results)}'
print('Smoke test passed: Memory add + query works')

# Test LLM extraction if model exists
model_paths = [
    'models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf',
    'models/qwen3-4b/Qwen3-4B-Instruct-2507.Q4_K_M.gguf',
]
for mp in model_paths:
    if os.path.exists(mp):
        try:
            mem2 = mnemefusion.Memory(os.path.join(td, 'smoke_llm.mfdb'), {'model_path': mp})
            result = mem2.extract_text('Bob is a software engineer at Google.')
            print(f'LLM extraction works: {len(result.get(\"entities\", []))} entities found')
        except Exception as e:
            print(f'LLM extraction failed: {e}')
        break
else:
    print('No model file found — skipping LLM extraction test')

import shutil
shutil.rmtree(td)
" 2>&1

info "Setup complete! Environment variables for future sessions:"
echo "  export CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
echo "  export LLAMA_BUILD_SHARED_LIBS=1"
echo "  export CUDA_PATH=${CUDA_PATH}"
echo "  export CUDACXX=${CUDA_PATH}/bin/nvcc"
echo ""
info "To rebuild after code changes:"
echo "  cd $WORKSPACE/mnemefusion-python"
echo "  source .venv/bin/activate"
echo "  maturin develop --release --features entity-extraction-cuda"
