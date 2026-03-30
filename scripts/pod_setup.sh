#!/bin/bash
set -e

echo "=== $(date) Installing system deps ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq pkg-config libclang-dev gcc g++ cmake 2>&1
echo "=== $(date) System deps done ==="

echo "=== $(date) Installing Rust ==="
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1
fi
source $HOME/.cargo/env
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "=== $(date) cmake version ==="
cmake --version | head -1

echo "=== $(date) Fetching deps and applying patches ==="
cd /workspace/mnemefusion
export LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu
export CUDA_PATH=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_ARCHITECTURES=86
export LLAMA_BUILD_SHARED_LIBS=1

# Fetch cargo deps first (so apply_patches can find them)
cargo fetch 2>&1

# Apply patches
python3 scripts/apply_patches.py --clean 2>&1

echo "=== $(date) Creating venv and installing Python deps ==="
cd /workspace/mnemefusion/mnemefusion-python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip 2>&1
pip install maturin sentence-transformers openai numpy 2>&1

echo "=== $(date) Building with CUDA ==="
maturin develop --release --features entity-extraction-cuda 2>&1

echo "=== $(date) Copying backend .so files ==="
WORKSPACE=/workspace/mnemefusion
for d in $(find $WORKSPACE/target/release/build/llama-cpp-sys-2-*/out/build/ -name "*.so" -o -name "*.so.*" 2>/dev/null | xargs -r dirname | sort -u); do
    cp -v $d/*.so* $WORKSPACE/ 2>/dev/null || true
done

echo "=== $(date) Smoke test ==="
cd /workspace/mnemefusion
export LD_LIBRARY_PATH=/workspace/mnemefusion
export MNEMEFUSION_DLL_DIR=/workspace/mnemefusion
python3 -c "
import mnemefusion
import tempfile, os
d = tempfile.mkdtemp()
m = mnemefusion.Memory(os.path.join(d, 'test.mfdb'), {'embedding_dim': 4})
mid = m.add('test memory', [0.1, 0.2, 0.3, 0.4], {'speaker': 'user'})
print(f'Memory added: {mid}')
try:
    r = m.consolidate()
    print(f'Consolidate: {r}')
except Exception as e:
    print(f'Consolidate (expected without LLM): {e}')
print('Smoke test PASSED!')
"

echo "=== $(date) SETUP COMPLETE ==="
