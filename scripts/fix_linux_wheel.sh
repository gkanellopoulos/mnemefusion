#!/usr/bin/env bash
# fix_linux_wheel.sh — Fix patchelf SONAME hashing in maturin Linux wheels
#
# When maturin builds a wheel on Linux, patchelf renames SONAMEs with hashes:
#   libggml-base.so.0 → libggml-base-8d64534e.so.0
#
# Dynamic backends (libggml-cpu.so, libggml-cuda.so) depend on the original
# names and fail to load. This script creates symlinks to fix it.
#
# Also removes bundled libcuda.so (conflicts with system CUDA driver).
#
# Usage:
#   # After maturin develop or maturin build:
#   bash scripts/fix_linux_wheel.sh [site-packages-dir]
#
#   # Auto-detect from venv:
#   bash scripts/fix_linux_wheel.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${RED}[WARN]${NC} $*"; }

# Find the mnemefusion.libs directory
LIBS_DIR=""

if [ $# -ge 1 ]; then
    # User-provided path
    LIBS_DIR="$1"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    # From active venv
    for d in "$VIRTUAL_ENV"/lib/python*/site-packages/mnemefusion.libs; do
        if [ -d "$d" ]; then
            LIBS_DIR="$d"
            break
        fi
    done
fi

# Fallback: search common locations
if [ -z "$LIBS_DIR" ] || [ ! -d "$LIBS_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    WORKSPACE="$(dirname "$SCRIPT_DIR")"
    for d in \
        "$WORKSPACE/mnemefusion-python/.venv/lib"/python*/site-packages/mnemefusion.libs \
        /usr/local/lib/python*/dist-packages/mnemefusion.libs \
        "$HOME/.local/lib"/python*/site-packages/mnemefusion.libs; do
        if [ -d "$d" ]; then
            LIBS_DIR="$d"
            break
        fi
    done
fi

if [ -z "$LIBS_DIR" ] || [ ! -d "$LIBS_DIR" ]; then
    echo "mnemefusion.libs directory not found."
    echo "Usage: $0 [path/to/mnemefusion.libs]"
    exit 1
fi

info "Fixing SONAME symlinks in: $LIBS_DIR"
cd "$LIBS_DIR"

# Create symlinks: hashed name → original name
# Pattern: libFOO-HEXHASH.so.N → libFOO.so.N
FIXED=0
for hashed in lib*-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].so*; do
    [ -f "$hashed" ] || continue
    # Extract original name by removing the hash
    original=$(echo "$hashed" | sed 's/-[0-9a-f]\{8\}\(\.so\)/\1/')
    if [ "$original" != "$hashed" ] && [ ! -e "$original" ]; then
        ln -sf "$hashed" "$original"
        info "  $original -> $hashed"
        FIXED=$((FIXED + 1))
    fi
done

# Remove bundled libcuda.so (conflicts with system CUDA driver)
for cuda_lib in libcuda.so* libnvidia*.so*; do
    if [ -f "$cuda_lib" ] && [ ! -L "$cuda_lib" ]; then
        info "  Removing bundled $cuda_lib (use system driver instead)"
        rm "$cuda_lib"
        FIXED=$((FIXED + 1))
    fi
done

if [ "$FIXED" -eq 0 ]; then
    info "No fixes needed — all symlinks already in place"
else
    info "Fixed $FIXED items"
fi
