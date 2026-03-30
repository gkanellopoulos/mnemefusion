#!/bin/bash
set -euo pipefail

# =============================================================================
# MnemeFusion Full Benchmark Suite
#
# Runs all three benchmarks end-to-end on a RunPod A40:
#   1. LoCoMo (1540q, 10-conv) — fresh ingestion + 3 evaluation runs
#   2. LongMemEval Oracle (500q) — per-question fresh ingestion + evaluation
#   3. LongMemEval Atomic (500q) — per-question atomic DB ingestion + 3 query runs
#
# Prerequisites:
#   - pod_setup.sh already ran (Rust, Python deps, CUDA build all done)
#   - OPENAI_API_KEY set
#   - Model GGUF at /workspace/model.gguf (or set MODEL_PATH)
#   - LongMemEval dataset at fixtures/longmemeval/longmemeval_s_cleaned.json
#   - LoCoMo dataset at evals/locomo/locomo10.json
#
# Usage:
#   # Run everything (in tmux!)
#   bash scripts/run_benchmarks.sh
#
#   # Run specific benchmark
#   bash scripts/run_benchmarks.sh locomo
#   bash scripts/run_benchmarks.sh oracle
#   bash scripts/run_benchmarks.sh atomic
#
#   # Resume (skips completed steps)
#   bash scripts/run_benchmarks.sh
# Disk usage estimate (/tmp has ~20GB on RunPod):
#   Model GGUF:      ~2.5GB
#   LoCoMo DB:       ~200MB (deleted after LoCoMo completes)
#   500 atomic DBs:  ~12.5GB (25MB each)
#   Total peak:      ~15GB (fits /tmp's 20GB limit)
# =============================================================================

WORKSPACE="${WORKSPACE:-/workspace/mnemefusion}"
MODEL_SRC="${MODEL_PATH:-/workspace/model.gguf}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/benchmark_results}"
VENV_DIR="$WORKSPACE/mnemefusion-python/.venv"

# Copy model to /tmp to avoid FUSE mmap hang
MODEL="/tmp/model.gguf"
if [ ! -f "$MODEL" ]; then
    echo "=== Copying model to /tmp (avoids FUSE mmap hang) ==="
    cp "$MODEL_SRC" "$MODEL"
fi

# Activate venv
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Set runtime env
export LD_LIBRARY_PATH="${WORKSPACE}:${LD_LIBRARY_PATH:-}"
export MNEMEFUSION_DLL_DIR="$WORKSPACE"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Verify API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

# Determine which benchmarks to run
BENCHMARK="${1:-all}"

echo "============================================================"
echo "MnemeFusion Benchmark Suite"
echo "  Workspace:   $WORKSPACE"
echo "  Model:       $MODEL"
echo "  Results:     $RESULTS_DIR"
echo "  Benchmark:   $BENCHMARK"
echo "  Date:        $(date -Iseconds)"
echo "============================================================"

# Download LongMemEval dataset if needed
LME_DIR="$WORKSPACE/evals/longmemeval/fixtures/longmemeval"
mkdir -p "$LME_DIR"

if [ ! -f "$LME_DIR/longmemeval_oracle.json" ] || [ ! -f "$LME_DIR/longmemeval_s_cleaned.json" ]; then
    echo ""
    echo "=== Downloading LongMemEval dataset ==="
    pip install datasets 2>/dev/null || true
    python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('xiaowu0162/longmemeval-cleaned')

oracle_path = '$LME_DIR/longmemeval_oracle.json'
s_path = '$LME_DIR/longmemeval_s_cleaned.json'

print('Writing oracle split...')
with open(oracle_path, 'w') as f:
    json.dump([dict(row) for row in ds['oracle']], f)
print(f'  Oracle: {len(ds[\"oracle\"])} entries')

print('Writing s_cleaned split...')
with open(s_path, 'w') as f:
    json.dump([dict(row) for row in ds['s_cleaned']], f)
print(f'  S-cleaned: {len(ds[\"s_cleaned\"])} entries')
print('Done!')
"
fi

# Quick smoke test
echo ""
echo "=== Smoke test ==="
cd "$WORKSPACE"
python3 -c "
import torch  # must be before mnemefusion on Linux
import mnemefusion
import tempfile, os
d = tempfile.mkdtemp()
m = mnemefusion.Memory(os.path.join(d, 'smoke.mfdb'), {'embedding_dim': 4})
mid = m.add('smoke test', [0.1, 0.2, 0.3, 0.4], {'speaker': 'user'})
intent, results, ctx = m.query('smoke', [0.1, 0.2, 0.3, 0.4], 5)
print(f'Smoke test PASSED (added={mid}, results={len(results)})')
del m
import shutil; shutil.rmtree(d)
"

# =============================================================================
# Benchmark 1: LoCoMo (1540q, 10-conv, 3 runs)
# =============================================================================

run_locomo() {
    echo ""
    echo "============================================================"
    echo "BENCHMARK 1: LoCoMo (10-conv, 1540q, 3 runs)"
    echo "  Started: $(date -Iseconds)"
    echo "============================================================"

    cd "$WORKSPACE"
    # DB on /tmp to avoid FUSE mmap hang; results on /workspace for persistence
    DB_PATH="/tmp/locomo_10conv.mfdb"
    OUTPUT="$RESULTS_DIR/locomo_results.json"

    # Check if already done (results JSON exists with 3 runs)
    if [ -f "$OUTPUT" ]; then
        echo "  Results file exists: $OUTPUT"
        echo "  To rerun, delete it first."
        echo "  SKIPPED"
        return 0
    fi

    # If DB exists on /workspace from a prior run, copy to /tmp
    if [ ! -f "$DB_PATH" ] && [ -f "$RESULTS_DIR/locomo_10conv.mfdb" ]; then
        echo "  Copying DB from /workspace to /tmp..."
        cp "$RESULTS_DIR/locomo_10conv.mfdb" "$DB_PATH"
    fi

    python3 evals/locomo/run_eval.py \
        --dataset evals/locomo/locomo10.json \
        --use-llm \
        --llm-model "$MODEL" \
        --llm-tier quality \
        --extraction-passes 1 \
        --db-path "$DB_PATH" \
        --output "$OUTPUT" \
        --runs 3 \
        --verbose \
        2>&1 | tee "$RESULTS_DIR/locomo_log.txt"

    echo ""
    echo "  LoCoMo COMPLETE: $(date -Iseconds)"
    echo "  Results: $OUTPUT"
    echo "  DB: $DB_PATH"

    # Copy DB to /workspace for persistence (survives pod restart)
    cp "$DB_PATH" "$RESULTS_DIR/locomo_10conv.mfdb" 2>/dev/null || true

    # Free /tmp space for atomic benchmark (~200MB)
    rm -f "$DB_PATH"
}


# =============================================================================
# Benchmark 2: LongMemEval Oracle (500q)
# =============================================================================

run_oracle() {
    echo ""
    echo "============================================================"
    echo "BENCHMARK 2: LongMemEval Oracle (500q)"
    echo "  Started: $(date -Iseconds)"
    echo "============================================================"

    cd "$WORKSPACE"
    ORACLE_OUTPUT="$RESULTS_DIR/longmemeval_oracle_results.json"

    if [ -f "$ORACLE_OUTPUT" ]; then
        echo "  Results file exists: $ORACLE_OUTPUT"
        echo "  To rerun, delete it first."
        echo "  SKIPPED"
        return 0
    fi

    # Oracle dataset must exist
    ORACLE_DATA="$WORKSPACE/evals/longmemeval/fixtures/longmemeval/longmemeval_oracle.json"
    if [ ! -f "$ORACLE_DATA" ]; then
        echo "  ERROR: Oracle dataset not found at $ORACLE_DATA"
        echo "  Download with: python3 -c \"from datasets import load_dataset; ...\""
        return 1
    fi

    python3 evals/longmemeval/run_eval.py \
        --mode oracle \
        --llm-model "$MODEL" \
        2>&1 | tee "$RESULTS_DIR/longmemeval_oracle_log.txt"

    # Copy results to results dir
    ORACLE_DEFAULT="$WORKSPACE/evals/longmemeval/fixtures/longmemeval/longmemeval_results_oracle_binary.json"
    if [ -f "$ORACLE_DEFAULT" ]; then
        cp "$ORACLE_DEFAULT" "$ORACLE_OUTPUT"
    fi

    echo ""
    echo "  LongMemEval Oracle COMPLETE: $(date -Iseconds)"
    echo "  Results: $ORACLE_OUTPUT"
}


# =============================================================================
# Benchmark 3: LongMemEval Atomic (500q, 3 query runs)
# =============================================================================

run_atomic() {
    echo ""
    echo "============================================================"
    echo "BENCHMARK 3: LongMemEval Atomic (500q, ingestion + 3 query runs)"
    echo "  Started: $(date -Iseconds)"
    echo "============================================================"

    cd "$WORKSPACE"

    # --- Phase 1: Ingestion ---
    ATOMIC_DB_DIR="/tmp/atomic_dbs"
    MANIFEST="$ATOMIC_DB_DIR/ingest_manifest.json"

    # Check if ingestion complete (manifest has 500 completed)
    INGESTION_DONE=false
    if [ -f "$MANIFEST" ]; then
        COMPLETED=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(len(m.get('completed',[])))")
        echo "  Ingestion manifest: $COMPLETED questions already done"
        if [ "$COMPLETED" -ge 500 ]; then
            INGESTION_DONE=true
            echo "  Ingestion COMPLETE — skipping to query phase"
        fi
    fi

    if [ "$INGESTION_DONE" = false ]; then
        echo ""
        echo "  --- Phase 1: Atomic DB Ingestion ---"
        echo "  DB dir:  $ATOMIC_DB_DIR"
        echo "  Model:   $MODEL"
        echo ""

        # s_cleaned.json must exist
        S_DATA="$WORKSPACE/evals/longmemeval/fixtures/longmemeval/longmemeval_s_cleaned.json"
        if [ ! -f "$S_DATA" ]; then
            echo "  ERROR: s_cleaned dataset not found at $S_DATA"
            echo "  Download with: python3 -c \"from datasets import load_dataset; ...\""
            return 1
        fi

        python3 evals/longmemeval/run_ingest_atomic.py \
            --llm-model "$MODEL" \
            --db-dir "$ATOMIC_DB_DIR" \
            2>&1 | tee "$RESULTS_DIR/atomic_ingest_log.txt"

        # Backup manifest to /workspace
        if [ -f "$MANIFEST" ]; then
            cp "$MANIFEST" "$RESULTS_DIR/atomic_ingest_manifest.json"
        fi
    fi

    # --- Phase 2: Query runs (3x) ---
    # Build slim dataset if needed (avoids loading 265MB for query-only)
    SLIM_PATH="$WORKSPACE/evals/longmemeval/fixtures/longmemeval/longmemeval_s_slim.json"
    if [ ! -f "$SLIM_PATH" ]; then
        echo ""
        echo "  Building slim dataset for query phase..."
        pip install ijson 2>/dev/null || true
        python3 "$WORKSPACE/evals/longmemeval/build_slim_dataset.py" 2>/dev/null || \
            echo "  WARNING: build_slim_dataset.py failed, query will use full dataset (265MB)"
    fi

    for RUN in 1 2 3; do
        QUERY_RESULTS="$ATOMIC_DB_DIR/query_results_run${RUN}.json"

        if [ -f "$QUERY_RESULTS" ]; then
            QCOUNT=$(python3 -c "import json; r=json.load(open('$QUERY_RESULTS')); print(len(r))")
            if [ "$QCOUNT" -ge 500 ]; then
                echo "  Run $RUN: $QCOUNT results — SKIPPED (already complete)"
                continue
            fi
        fi

        echo ""
        echo "  --- Phase 2: Query Run $RUN/3 ---"

        # Move previous results out of the way (run_query_bim.py caches to query_results.json)
        if [ -f "$ATOMIC_DB_DIR/query_results.json" ]; then
            mv "$ATOMIC_DB_DIR/query_results.json" "$ATOMIC_DB_DIR/query_results_prev.json"
        fi

        python3 evals/longmemeval/run_query_bim.py \
            --db-dir "$ATOMIC_DB_DIR" \
            --cycles 0 \
            2>&1 | tee "$RESULTS_DIR/atomic_query_run${RUN}_log.txt"

        # Rename results for this run
        if [ -f "$ATOMIC_DB_DIR/query_results.json" ]; then
            cp "$ATOMIC_DB_DIR/query_results.json" "$QUERY_RESULTS"
            cp "$QUERY_RESULTS" "$RESULTS_DIR/atomic_query_run${RUN}.json"
        fi
    done

    # Copy all results to persistent storage
    for f in "$ATOMIC_DB_DIR"/query_results_run*.json; do
        [ -f "$f" ] && cp "$f" "$RESULTS_DIR/"
    done

    echo ""
    echo "  LongMemEval Atomic COMPLETE: $(date -Iseconds)"
    echo "  Results: $RESULTS_DIR/atomic_query_run{1,2,3}.json"
}


# =============================================================================
# Summary
# =============================================================================

print_summary() {
    echo ""
    echo "============================================================"
    echo "ALL BENCHMARKS COMPLETE"
    echo "  Finished: $(date -Iseconds)"
    echo "============================================================"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    ls -lh "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no JSON results yet)"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in $RESULTS_DIR/"
    echo "  2. Copy to local: scp -r pod:$RESULTS_DIR/ ."
    echo "  3. Update evals/README.md and evals/longmemeval/README.md with new numbers"
}


# =============================================================================
# Main
# =============================================================================

case "$BENCHMARK" in
    locomo)
        run_locomo
        ;;
    oracle)
        run_oracle
        ;;
    atomic)
        run_atomic
        ;;
    all)
        run_locomo
        run_oracle
        run_atomic
        print_summary
        ;;
    *)
        echo "Usage: $0 [locomo|oracle|atomic|all]"
        echo "  Default: all"
        exit 1
        ;;
esac
