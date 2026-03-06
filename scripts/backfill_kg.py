#!/usr/bin/env python3
"""Backfill KG triples onto an existing MnemeFusion database using Triplex.

Copies the source DB, loads Triplex, and runs KG extraction on all memories.
This adds entity-to-entity relationship edges without modifying any existing
memories, embeddings, or entity profiles.

Usage:
    python scripts/backfill_kg.py \
        --source tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb \
        --output tests/benchmarks/fixtures/eval_s36_with_kg.mfdb \
        --triplex-model models/triplex/Triplex-Q4_K_M.gguf
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# Ensure stdout is unbuffered when redirected to file
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="Backfill KG triples onto existing DB")
    parser.add_argument("--source", required=True, help="Source .mfdb file path")
    parser.add_argument("--output", required=True, help="Output .mfdb file path (copy of source + KG)")
    parser.add_argument("--triplex-model", required=True, help="Path to Triplex GGUF model")
    parser.add_argument("--in-place", action="store_true", help="Modify source DB directly (skip copy)")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    triplex_model = Path(args.triplex_model)

    # Validate inputs
    if not source.exists():
        print(f"ERROR: Source DB not found: {source}")
        sys.exit(1)
    if not triplex_model.exists():
        print(f"ERROR: Triplex model not found: {triplex_model}")
        sys.exit(1)

    # Copy DB (unless --in-place)
    if args.in_place:
        db_path = source
        print(f"Operating in-place on: {db_path}")
    else:
        print(f"Copying {source} -> {output}...")
        shutil.copy2(source, output)
        db_path = output
        print(f"  [OK] Copy complete ({output.stat().st_size / 1024 / 1024:.1f} MB)")

    # Import mnemefusion
    try:
        import torch  # must import before sentence_transformers on Windows
    except ImportError:
        pass

    import mnemefusion

    # Open DB
    print(f"\nOpening database: {db_path}")
    mem = mnemefusion.Memory(str(db_path), {"embedding_dim": 768})
    print("  [OK] Database opened")

    # Enable Triplex KG extraction
    print(f"\nLoading Triplex model: {triplex_model}")
    t0 = time.time()
    mem.enable_kg_extraction(str(triplex_model))
    print(f"  [OK] Triplex loaded in {time.time() - t0:.1f}s")

    # Run backfill
    print("\n" + "=" * 60)
    print("STARTING KG BACKFILL")
    print("=" * 60)
    t0 = time.time()
    processed = mem.backfill_kg()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("KG BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Memories with triples: {processed}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 3600:.1f}h)")
    print(f"  DB size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {db_path}")


if __name__ == "__main__":
    main()
