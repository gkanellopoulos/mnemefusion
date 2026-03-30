#!/usr/bin/env python3
"""Merge all per-haystack BIM .mfdb files into a single consolidated database.

This script copies data verbatim from 500 individual .mfdb files into one DB.
No re-ingestion, no embedding computation, no LLM calls — pure data copy.

Source files are never modified (opened read-only via the API).

Usage:
    python merge_bim_dbs.py --src-dir B:/benchmark/bim_master --out B:/benchmark/bim_merged.mfdb
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge BIM per-haystack DBs into one")
    parser.add_argument("--src-dir", required=True, help="Directory with per-question .mfdb files")
    parser.add_argument("--out", required=True, help="Output merged .mfdb path")
    parser.add_argument("--dry-run", action="store_true", help="Count memories without writing")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Memories per add_batch call (default: 500)")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_path = args.out

    if os.path.exists(out_path) and not args.dry_run:
        print(f"ERROR: Output file already exists: {out_path}")
        print("  Delete it first or choose a different path.")
        sys.exit(1)

    # Load dataset to get authoritative question IDs
    dataset_path = Path(__file__).parent / "fixtures" / "longmemeval" / "longmemeval_s_slim.json"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    import json
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    question_ids = {entry["question_id"] for entry in dataset}

    # Match to .mfdb files using exact question IDs
    mfdb_files = []
    missing = []
    for qid in sorted(question_ids):
        db_path = src_dir / f"{qid}.mfdb"
        if db_path.exists():
            mfdb_files.append(db_path)
        else:
            missing.append(qid)
    print(f"Found {len(mfdb_files)} master .mfdb files for {len(question_ids)} questions",
          flush=True)
    if missing:
        print(f"  Missing: {len(missing)} (first 5: {missing[:5]})", flush=True)

    if not mfdb_files:
        print("ERROR: No master .mfdb files found")
        sys.exit(1)

    try:
        import mnemefusion
    except ImportError:
        print("ERROR: mnemefusion not installed")
        sys.exit(1)

    # First pass: count total memories and detect embedding dimension
    print("\nPhase 1: Counting memories...", flush=True)
    total_memories = 0
    embedding_dim = None
    seen_ids = set()
    duplicates = 0

    for i, mfdb in enumerate(mfdb_files):
        try:
            src = mnemefusion.Memory(str(mfdb), {"embedding_dim": 768})
            ids = src.list_ids()
            for mid in ids:
                if mid in seen_ids:
                    duplicates += 1
                else:
                    seen_ids.add(mid)
            total_memories += len(ids)

            if embedding_dim is None and ids:
                m = src.get(ids[0])
                embedding_dim = len(m["embedding"])

            if (i + 1) % 100 == 0:
                print(f"  Scanned {i + 1}/{len(mfdb_files)} files, "
                      f"{total_memories} memories so far", flush=True)
        except Exception as e:
            print(f"  WARNING: Failed to read {mfdb.name}: {e}", flush=True)
            continue

    unique_count = len(seen_ids)
    print(f"\nPhase 1 complete:", flush=True)
    print(f"  Total memories across all files: {total_memories}", flush=True)
    print(f"  Unique memory IDs: {unique_count}", flush=True)
    print(f"  Duplicates (same ID in multiple files): {duplicates}", flush=True)
    print(f"  Embedding dimension: {embedding_dim}", flush=True)

    if args.dry_run:
        print("\nDry run complete. No output written.", flush=True)
        return

    if embedding_dim is None:
        print("ERROR: Could not detect embedding dimension")
        sys.exit(1)

    # Second pass: copy all memories into merged DB using add_batch
    print(f"\nPhase 2: Copying {unique_count} memories into {out_path}...", flush=True)
    start = time.time()

    dst = mnemefusion.Memory(out_path, {"embedding_dim": embedding_dim})

    written = 0
    skipped = 0
    errors = 0
    written_ids = set()
    batch = []

    def flush_batch():
        nonlocal written, errors
        if not batch:
            return
        try:
            result = dst.add_batch(batch)
            written += result["created_count"]
            if result["errors"]:
                errors += len(result["errors"])
                for err in result["errors"][:3]:
                    print(f"    Batch error: {err}", flush=True)
        except Exception as e:
            errors += len(batch)
            print(f"    Batch write failed: {e}", flush=True)
        batch.clear()

    for i, mfdb in enumerate(mfdb_files):
        try:
            src = mnemefusion.Memory(str(mfdb), {"embedding_dim": embedding_dim})
            ids = src.list_ids()

            for mid in ids:
                if mid in written_ids:
                    skipped += 1
                    continue

                m = src.get(mid)
                if m is None:
                    skipped += 1
                    continue

                entry = {
                    "content": m["content"],
                    "embedding": m["embedding"],
                }
                if m.get("metadata"):
                    entry["metadata"] = m["metadata"]
                if m.get("created_at"):
                    entry["timestamp"] = m["created_at"]

                batch.append(entry)
                written_ids.add(mid)

                if len(batch) >= args.batch_size:
                    flush_batch()

        except Exception as e:
            print(f"  WARNING: Failed to process {mfdb.name}: {e}", flush=True)
            errors += 1
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = written / elapsed if elapsed > 0 else 0
            pct = 100 * written / unique_count if unique_count else 0
            print(f"  [{i + 1}/{len(mfdb_files)}] Written: {written}/{unique_count} "
                  f"({pct:.1f}%), Rate: {rate:.0f} mem/s, "
                  f"Elapsed: {elapsed:.0f}s", flush=True)

    # Flush remaining
    flush_batch()

    elapsed = time.time() - start
    print(f"\nPhase 2 complete in {elapsed:.1f}s:", flush=True)
    print(f"  Written: {written}", flush=True)
    print(f"  Skipped (duplicates): {skipped}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Output: {out_path}", flush=True)
    print(f"  Output size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB", flush=True)

    # Verify — must close dst first to release redb lock
    del dst
    print(f"\nPhase 3: Verification...", flush=True)
    verify = mnemefusion.Memory(out_path, {"embedding_dim": embedding_dim})
    verify_count = verify.count()
    print(f"  Merged DB count: {verify_count}", flush=True)
    if verify_count == written:
        print(f"  PASS: count matches written ({written})", flush=True)
    else:
        print(f"  WARN: count ({verify_count}) != written ({written})", flush=True)


if __name__ == "__main__":
    main()
