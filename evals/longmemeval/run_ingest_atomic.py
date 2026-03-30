#!/usr/bin/env python3
"""
Per-Question Atomic DB Ingestion for LongMemEval

Ingests all 500 LongMemEval haystacks into individual .mfdb files.
Each question gets its own database with full LLM entity extraction.
These atomic DBs are then queried by run_query_bim.py --cycles 0.

Multi-GPU support: each worker is pinned to a specific GPU via CUDA_VISIBLE_DEVICES.
Workers are assigned round-robin across available GPUs.

Usage:
    # Single GPU, single worker
    python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs

    # Single GPU, 2 workers (run in separate terminals)
    python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs \
        --num-workers 2 --worker 0
    python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs \
        --num-workers 2 --worker 1

    # 2 GPUs, 4 workers (2 per GPU)
    python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs \
        --num-workers 4 --worker 0 --gpu-ids 0,1

    # Limit to N questions (for benchmarking throughput)
    python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs \
        --max-questions 5
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# GPU pinning must happen before any CUDA library import
def pin_gpu(worker: int, gpu_ids: List[int]):
    """Pin this worker to a specific GPU via CUDA_VISIBLE_DEVICES."""
    if not gpu_ids:
        return
    gpu_idx = gpu_ids[worker % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    print(f"  Worker {worker} pinned to GPU {gpu_idx}")


def main():
    args = parse_args()

    # Pin GPU BEFORE importing torch/mnemefusion (CUDA init happens at import)
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        pin_gpu(args.worker, gpu_ids)

    # Now safe to import CUDA-dependent libraries
    try:
        import torch  # noqa: F401 — must be imported before mnemefusion on Linux
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        sys.exit(1)

    try:
        import mnemefusion
    except ImportError:
        print("ERROR: mnemefusion not installed.")
        sys.exit(1)

    run(args, SentenceTransformer, mnemefusion)


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "longmemeval"


# =============================================================================
# Dataset
# =============================================================================

def load_dataset() -> List[Dict]:
    """Load the LongMemEval s-mode dataset."""
    path = FIXTURES_DIR / "longmemeval_s_cleaned.json"
    if not path.exists():
        print(f"ERROR: Dataset not found at {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_session_date(date_str: str) -> Optional[float]:
    """Parse a haystack date string to Unix timestamp."""
    if not date_str:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%a)", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


# =============================================================================
# Ingestion
# =============================================================================

def ingest_question(
    entry: Dict,
    embedder,
    mnemefusion,
    llm_model_path: str,
    db_path: str,
    user_only: bool = False,
) -> Dict:
    """Ingest all turns for one question into a .mfdb file.

    Returns a stats dict with timing and counts.
    """
    qid = entry["question_id"]
    sessions_raw = entry["haystack_sessions"]
    dates = entry.get("haystack_dates", [])
    session_ids = entry.get("haystack_session_ids", [])

    # Prepare turns
    all_contents = []
    all_metadata = []
    all_timestamps = []

    for sess_idx, session_turns in enumerate(sessions_raw):
        session_date = dates[sess_idx] if sess_idx < len(dates) else ""
        session_id = str(session_ids[sess_idx]) if sess_idx < len(session_ids) else str(sess_idx)
        timestamp = parse_session_date(session_date)

        for turn_idx, turn in enumerate(session_turns):
            content = turn.get("content", "")
            if not content or not content.strip():
                continue
            role = turn.get("role", "user")
            if user_only and role == "assistant":
                continue

            metadata = {
                "speaker": role,
                "session_id": session_id,
                "session_idx": str(sess_idx),
                "turn_idx": str(turn_idx),
                "dialog_id": f"S{session_id}:{turn_idx}",
            }
            if session_date:
                metadata["session_date"] = session_date

            all_contents.append(content)
            all_metadata.append(metadata)
            all_timestamps.append(timestamp)

    num_turns = len(all_contents)
    num_sessions = len(sessions_raw)

    # Batch embed
    t_embed = time.time()
    all_embeddings = embedder.encode(all_contents, show_progress_bar=False, batch_size=64)
    embed_time = time.time() - t_embed

    # Create memory engine
    embedding_dim = embedder.get_sentence_embedding_dimension()
    config = {"embedding_dim": embedding_dim}
    mem = mnemefusion.Memory(db_path, config)
    mem.enable_llm_entity_extraction(llm_model_path, "quality", 1)
    embed_fn = lambda text: embedder.encode(text, show_progress_bar=False).tolist()
    mem.set_embedding_fn(embed_fn)

    # Ingest all turns
    t_ingest = time.time()
    for i in range(num_turns):
        mem.add(
            all_contents[i],
            all_embeddings[i].tolist(),
            all_metadata[i],
            timestamp=all_timestamps[i],
        )
    ingest_time = time.time() - t_ingest

    # Post-ingestion: summarize profiles and precompute fact embeddings
    try:
        mem.summarize_profiles()
        mem.precompute_fact_embeddings()
    except Exception:
        pass

    mem.set_user_entity("user")

    # Collect stats
    stats = {
        "question_id": qid,
        "question_type": entry.get("question_type", ""),
        "num_turns": num_turns,
        "num_sessions": num_sessions,
        "embed_time_s": round(embed_time, 1),
        "ingest_time_s": round(ingest_time, 1),
        "total_time_s": round(embed_time + ingest_time, 1),
        "turns_per_sec": round(num_turns / max(ingest_time, 0.01), 2),
        "db_path": db_path,
        "user_only": user_only,
        "timestamp": datetime.now().isoformat(),
    }

    # Clean up LLM resources
    del mem
    gc.collect()

    return stats


# =============================================================================
# Main
# =============================================================================

def run(args, SentenceTransformer, mnemefusion):
    data = load_dataset()
    print(f"Loaded {len(data)} questions")

    # Apply max-questions limit
    if args.max_questions:
        data = data[:args.max_questions]
        print(f"Limited to {len(data)} questions")

    # Setup output directory
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest (tracks completed ingestions)
    manifest_path = db_dir / "ingest_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"completed": [], "stats": []}

    completed_ids = {s["question_id"] for s in manifest["stats"]}

    # Worker partitioning
    my_questions = []
    for q_idx, entry in enumerate(data):
        if args.num_workers > 1 and (q_idx % args.num_workers) != args.worker:
            continue
        if entry["question_id"] in completed_ids:
            continue
        my_questions.append((q_idx, entry))

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    # Print run config
    total_turns = sum(
        sum(len(s) for s in e["haystack_sessions"]) for _, e in my_questions
    )
    print(f"\n{'=' * 70}")
    print(f"Atomic DB Ingestion (LongMemEval)")
    print(f"  LLM model:    {args.llm_model}")
    print(f"  DB directory:  {db_dir}")
    print(f"  Worker:        {args.worker}/{args.num_workers}")
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        print(f"  GPU:           {gpu_ids[args.worker % len(gpu_ids)]} (of {gpu_ids})")
    print(f"  Questions:     {len(my_questions)} to process ({len(completed_ids)} already done)")
    print(f"  Total turns:   {total_turns}")
    print(f"  User-only:     {args.user_only}")
    print(f"{'=' * 70}\n")

    if not my_questions:
        print("Nothing to do — all questions already ingested.")
        return

    # Process questions
    run_start = time.time()
    processed = 0

    for q_idx, entry in my_questions:
        qid = entry["question_id"]
        qtype = entry["question_type"]
        num_sessions = len(entry["haystack_sessions"])
        num_turns = sum(len(s) for s in entry["haystack_sessions"])

        db_path = str(db_dir / f"{qid}.mfdb")

        # Skip if DB already exists (crash recovery)
        if os.path.exists(db_path):
            print(f"[{processed+1}/{len(my_questions)}] {qid} — DB exists, skipping")
            processed += 1
            continue

        print(f"[{processed+1}/{len(my_questions)}] {qid} ({qtype}) "
              f"— {num_sessions} sessions, {num_turns} turns")

        try:
            stats = ingest_question(
                entry, embedder, mnemefusion, args.llm_model, db_path,
                user_only=args.user_only,
            )

            print(f"  Done: {stats['ingest_time_s']}s ingest + {stats['embed_time_s']}s embed "
                  f"= {stats['total_time_s']}s total ({stats['turns_per_sec']} turns/s)")

            # Update manifest (crash-safe: write after each question)
            manifest["stats"].append(stats)
            manifest["completed"].append(qid)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            # Remove partial DB if it exists
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass
            continue

        processed += 1

        # Progress summary every 10 questions
        if processed % 10 == 0:
            elapsed = time.time() - run_start
            avg_per_q = elapsed / processed
            remaining = len(my_questions) - processed
            eta_hours = (remaining * avg_per_q) / 3600
            print(f"\n  --- Progress: {processed}/{len(my_questions)} "
                  f"({elapsed/3600:.1f}h elapsed, ~{eta_hours:.1f}h remaining) ---\n")

    # Final summary
    elapsed = time.time() - run_start
    total_ingest = sum(s["ingest_time_s"] for s in manifest["stats"]
                       if s["question_id"] in {e["question_id"] for _, e in my_questions})

    print(f"\n{'=' * 70}")
    print(f"INGESTION COMPLETE")
    print(f"  Processed:     {processed} questions")
    print(f"  Wall time:     {elapsed/3600:.2f} hours")
    print(f"  Total ingest:  {total_ingest/3600:.2f} hours")
    print(f"  DB directory:  {db_dir}")
    print(f"  Manifest:      {manifest_path}")
    print(f"{'=' * 70}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-question atomic DB ingestion for LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single worker, all 500 questions
  python run_ingest_atomic.py --llm-model /path/to/model.gguf --db-dir ./atomic_dbs

  # 2 GPUs, 4 workers total
  python run_ingest_atomic.py --llm-model ... --db-dir ./atomic_dbs \\
      --num-workers 4 --worker 0 --gpu-ids 0,1

  # Benchmark: 5 questions to measure throughput
  python run_ingest_atomic.py --llm-model ... --db-dir ./atomic_dbs --max-questions 5
""",
    )
    parser.add_argument("--llm-model", type=str, required=True,
                        help="Path to GGUF model for LLM entity extraction")
    parser.add_argument("--db-dir", type=str, required=True,
                        help="Output directory for per-question .mfdb files")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--worker", type=int, default=0,
                        help="This worker's index (0-based)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1'). "
                             "Workers assigned round-robin.")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (for benchmarking)")
    parser.add_argument("--user-only", action="store_true",
                        help="Ingest only user turns (skip assistant turns)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
