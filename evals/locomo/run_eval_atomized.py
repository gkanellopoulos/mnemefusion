#!/usr/bin/env python3
"""
LoCoMo Atomized Evaluation for MnemeFusion

Tests the DB-per-entity architecture: one .mfdb file per conversation,
matching how MnemeFusion is designed to be used in production (one DB
per user/customer/project).

Compares directly with the standard LoCoMo benchmark (single shared DB)
to quantify the accuracy gain from architectural entity isolation.

Usage:
    export OPENAI_API_KEY=sk-...

    # Atomized evaluation with pre-built per-conversation DBs
    python run_eval_atomized.py --db-dir fixtures/atomized_dbs --skip-ingestion

    # Full ingestion + evaluation (creates per-conversation DBs)
    python run_eval_atomized.py --use-llm --llm-model ../../models/phi-4-mini/...gguf

    # Multi-run for publication
    python run_eval_atomized.py --db-dir fixtures/atomized_dbs --skip-ingestion --runs 3

Dataset: LoCoMo — 10 conversations, ~1,540 questions (categories 1-4)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import tempfile
import shutil

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

# IMPORTANT: Import torch BEFORE mnemefusion on Linux — mnemefusion loads libggml-cuda.so
# which can poison CUDA symbols and break torch's libc10_cuda.so loading.
try:
    import torch  # Must import before sentence_transformers on Windows (DLL search order)
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    sys.exit(1)

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed.")
    print("Install with: cd mnemefusion-python && maturin develop --release")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI package not installed.")
    sys.exit(1)

# Import shared utilities from run_eval
from run_eval import (
    load_locomo_dataset,
    prepare_documents,
    LLMClient,
    calculate_f1_score,
    calculate_bleu_score,
    QuestionResult,
    EvaluationResults,
    print_results,
)


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 25


def create_per_conversation_dbs(
    conversations: List[Dict],
    db_dir: str,
    embedder: SentenceTransformer,
    use_llm: bool = False,
    llm_model_path: str = None,
    llm_tier: str = "quality",
    extraction_passes: int = 1,
) -> Dict[str, str]:
    """
    Create one .mfdb file per conversation.

    Returns:
        Dict mapping conv_id -> db_path
    """
    os.makedirs(db_dir, exist_ok=True)
    embedding_dim = embedder.get_sentence_embedding_dimension()
    db_map = {}

    for conv in conversations:
        conv_id = conv.get('sample_id', 'unknown')
        db_path = os.path.join(db_dir, f"{conv_id}.mfdb")
        db_map[conv_id] = db_path

        if os.path.exists(db_path):
            print(f"  [SKIP] {conv_id}: DB already exists at {db_path}")
            continue

        print(f"\n  Ingesting conversation {conv_id}...")
        config = {"embedding_dim": embedding_dim, "entity_extraction_enabled": True}
        mem = mnemefusion.Memory(db_path, config)

        # Enable LLM extraction
        if use_llm and llm_model_path:
            try:
                mem.enable_llm_entity_extraction(llm_model_path, llm_tier, extraction_passes)
            except Exception as e:
                print(f"    [WARN] LLM extraction failed: {e}")

        # Set embedding function for fact embeddings
        mem.set_embedding_fn(
            lambda text: embedder.encode(text, show_progress_bar=False).tolist()
        )

        # Extract documents for this conversation only
        documents, _ = prepare_documents([conv])
        print(f"    {len(documents)} documents")

        # Ingest
        t0 = time.time()
        if use_llm:
            for i, (doc_id, content, metadata) in enumerate(documents):
                embedding = embedder.encode(content, show_progress_bar=False).tolist()
                mem.add(content, embedding, metadata)
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(documents)} docs...", flush=True)
        else:
            # Batch embedding + add
            contents = [c for _, c, _ in documents]
            all_embeddings = embedder.encode(contents, show_progress_bar=False, batch_size=64)
            for i, (doc_id, content, metadata) in enumerate(documents):
                mem.add(content, all_embeddings[i].tolist(), metadata)

        elapsed = time.time() - t0
        print(f"    Ingested in {elapsed:.1f}s")

        # Post-ingestion cleanup
        if use_llm:
            try:
                facts_removed, profiles_deleted = mem.consolidate_profiles()
                print(f"    Consolidated: -{facts_removed} facts, -{profiles_deleted} profiles")
            except Exception:
                pass
            try:
                mem.summarize_profiles()
            except Exception:
                pass

        # Precompute fact embeddings
        try:
            mem.precompute_fact_embeddings()
        except Exception:
            pass

        # Rebuild speaker embeddings (1p -> 3p pronoun substitution)
        try:
            mem.rebuild_speaker_embeddings()
        except Exception:
            pass

        mem.close()

        # Report stats
        mem2 = mnemefusion.Memory(db_path, config)
        n_memories = mem2.count()
        n_profiles = mem2.count_entity_profiles()
        print(f"    DB: {n_memories} memories, {n_profiles} profiles")
        mem2.close()

    return db_map


def run_atomized_evaluation(
    dataset_path: str,
    db_dir: str = None,
    num_conversations: int = None,
    max_questions: int = None,
    categories: List[int] = None,
    top_k: int = TOP_K,
    use_llm: bool = False,
    llm_model_path: str = None,
    llm_tier: str = "quality",
    extraction_passes: int = 1,
    skip_ingestion: bool = False,
    runs: int = 1,
    verbose: bool = False,
) -> EvaluationResults:
    """Run LoCoMo evaluation with one DB per conversation."""

    print("=" * 70)
    print("MnemeFusion LoCoMo ATOMIZED Evaluation")
    print("=" * 70)
    print(f"\n  Architecture:     One DB per conversation (production-realistic)")
    print(f"  Answer model:     GPT-4o-mini, temperature=0")
    print(f"  Judge model:      GPT-4o-mini, temperature=0")
    print(f"  Scoring:          Binary CORRECT/WRONG (Mem0-compatible judge prompt)")
    cat_list = categories if categories else [1, 2, 3, 4]
    print(f"  Categories:       {cat_list}")
    print(f"  Embedding model:  {EMBEDDING_MODEL}")
    if use_llm:
        print(f"  Extraction:       Native LLM ({llm_tier} tier, {extraction_passes} pass)")
    else:
        print(f"  Extraction:       Disabled (baseline)")
    if runs > 1:
        print(f"  Runs:             {runs} (will report mean +/- stddev)")
    print("=" * 70)

    # Load dataset
    conversations, all_questions = load_locomo_dataset(dataset_path, num_conversations)

    # Filter questions
    if categories:
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in categories]
    else:
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in [1, 2, 3, 4]]

    if max_questions:
        questions = questions[:max_questions]

    print(f"\nEvaluating {len(questions)} questions across {len(conversations)} conversations")

    # Per-conversation question counts
    conv_q_counts = defaultdict(int)
    for _, _, _, cid, _ in questions:
        conv_q_counts[cid] += 1
    for cid, cnt in sorted(conv_q_counts.items()):
        print(f"  {cid}: {cnt} questions")

    # Initialize embedder (CPU when LLM is active, to save GPU VRAM)
    embed_gpu = not use_llm
    if embed_gpu:
        embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        if torch.cuda.is_available():
            embedder.to('cuda')
    else:
        embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")

    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {embedding_dim}")

    # Create or locate per-conversation DBs
    _temp_ctx = None
    if not db_dir:
        _temp_ctx = tempfile.mkdtemp(prefix="locomo_atomized_")
        db_dir = _temp_ctx

    if skip_ingestion:
        # Build db_map from existing files
        db_map = {}
        for conv in conversations:
            conv_id = conv.get('sample_id', 'unknown')
            db_path = os.path.join(db_dir, f"{conv_id}.mfdb")
            if os.path.exists(db_path):
                db_map[conv_id] = db_path
            else:
                print(f"  [WARN] No DB for {conv_id} at {db_path}")
        print(f"\n  Found {len(db_map)}/{len(conversations)} pre-built DBs in {db_dir}")
    else:
        print(f"\nCreating per-conversation DBs in {db_dir}...")
        t0 = time.time()
        db_map = create_per_conversation_dbs(
            conversations, db_dir, embedder,
            use_llm=use_llm, llm_model_path=llm_model_path,
            llm_tier=llm_tier, extraction_passes=extraction_passes,
        )
        print(f"\nIngestion complete in {time.time() - t0:.1f}s")

    # Open all per-conversation Memory instances
    mem_instances = {}
    for conv_id, db_path in db_map.items():
        config = {"embedding_dim": embedding_dim}
        mem = mnemefusion.Memory(db_path, config)
        # Set embedding function for fact embeddings
        mem.set_embedding_fn(
            lambda text: embedder.encode(text, show_progress_bar=False).tolist()
        )
        try:
            mem.precompute_fact_embeddings()
        except Exception:
            pass
        mem_instances[conv_id] = mem

    llm = LLMClient(model="gpt-4o-mini")

    # Multi-run support
    run_accuracies = []

    for run_idx in range(runs):
        if runs > 1:
            print(f"\n{'#' * 70}")
            print(f"# RUN {run_idx + 1}/{runs}")
            print(f"{'#' * 70}")

        print(f"\nEvaluating {len(questions)} questions...")
        print("-" * 70)

        results = []
        latencies = []
        skipped = 0
        eval_start = time.time()

        for i, (question, ground_truth, category, conv_id, evidence) in enumerate(questions):
            if verbose or (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(questions)}] Cat {category} ({conv_id}): {question[:50]}...")

            # Route question to its conversation's DB
            mem = mem_instances.get(conv_id)
            if not mem:
                skipped += 1
                continue

            # Generate query embedding
            query_embedding = embedder.encode(question, show_progress_bar=False).tolist()

            # Query
            recall_k = max(top_k, 20)
            start = time.time()
            intent_info, query_results, profile_context = mem.query(
                question, query_embedding, recall_k
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

            # Extract content and IDs
            retrieved_content = []
            retrieved_ids = []
            for result_dict, scores_dict in query_results:
                content = result_dict.get("content", "")
                metadata = result_dict.get("metadata", {})
                dialog_id = metadata.get("dialog_id", result_dict.get("id", ""))
                session_date = metadata.get("session_date", "")
                speaker = metadata.get("speaker", "")
                # Format with date context for temporal reasoning (matching standard eval)
                if session_date:
                    formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
                else:
                    formatted = f"{speaker}: {content}" if speaker else content
                retrieved_content.append(formatted)
                retrieved_ids.append(dialog_id)

            # Recall@K
            evidence_set = set(evidence) if evidence else set()
            r_at_5 = r_at_10 = r_at_20 = 0.0
            if evidence_set:
                found_at_5 = len(evidence_set & set(retrieved_ids[:5]))
                found_at_10 = len(evidence_set & set(retrieved_ids[:10]))
                found_at_20 = len(evidence_set & set(retrieved_ids[:min(20, len(retrieved_ids))]))
                r_at_5 = found_at_5 / len(evidence_set)
                r_at_10 = found_at_10 / len(evidence_set)
                r_at_20 = found_at_20 / len(evidence_set)

            # Build context
            n_profile = min(len(profile_context), 5)
            content_budget = top_k - n_profile
            context_with_facts = profile_context[:n_profile] + retrieved_content[:content_budget]

            # Generate answer + judge
            gen_start = time.time()
            answer, tokens = llm.generate_answer(question, context_with_facts)
            judge_score = llm.judge_answer(question, ground_truth, answer)
            gen_latency = (time.time() - gen_start) * 1000
            f1 = calculate_f1_score(answer, ground_truth)
            bleu = calculate_bleu_score(answer, ground_truth)

            result = QuestionResult(
                question_id=f"{conv_id}_{i}",
                question=question,
                ground_truth=ground_truth,
                generated_answer=answer,
                category=category,
                llm_judge_score=judge_score,
                f1_score=f1,
                bleu_score=bleu,
                retrieval_latency_ms=latency_ms,
                generation_latency_ms=gen_latency,
                tokens_used=tokens,
                memories_retrieved=len(retrieved_content[:top_k]),
                recall_at_5=r_at_5,
                recall_at_10=r_at_10,
                recall_at_20=r_at_20,
            )
            results.append(result)

        eval_time = time.time() - eval_start

        if skipped:
            print(f"\n  [WARN] Skipped {skipped} questions (no DB for conversation)")

        # Compute metrics
        if results:
            accuracy = sum(1 for r in results if r.llm_judge_score == 1) / len(results)
            run_accuracies.append(accuracy)
            avg_f1 = sum(r.f1_score for r in results) / len(results)
            avg_recall_5 = sum(r.recall_at_5 for r in results) / len(results)
            avg_recall_10 = sum(r.recall_at_10 for r in results) / len(results)
            avg_recall_20 = sum(r.recall_at_20 for r in results) / len(results)

            print(f"\n{'=' * 70}")
            print(f"ATOMIZED RESULTS (Run {run_idx + 1})")
            print(f"{'=' * 70}")
            print(f"  Overall accuracy:  {accuracy * 100:.1f}% ({sum(1 for r in results if r.llm_judge_score == 1)}/{len(results)})")
            print(f"  Avg F1:            {avg_f1:.3f}")
            print(f"  Recall@5:          {avg_recall_5 * 100:.1f}%")
            print(f"  Recall@10:         {avg_recall_10 * 100:.1f}%")
            print(f"  Recall@20:         {avg_recall_20 * 100:.1f}%")

            if latencies:
                latencies_sorted = sorted(latencies)
                p50 = latencies_sorted[len(latencies_sorted) // 2]
                p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
                print(f"  Latency:           P50={p50:.0f}ms  P95={p95:.0f}ms")

            # Per-category
            category_names = {1: "Single-hop", 2: "Multi-hop", 3: "Temporal", 4: "Open-domain"}
            for cat in sorted(set(r.category for r in results)):
                cat_results = [r for r in results if r.category == cat]
                cat_acc = sum(1 for r in cat_results if r.llm_judge_score == 1) / len(cat_results)
                cat_name = category_names.get(cat, f"Category {cat}")
                print(f"  {cat_name:15s}     {cat_acc * 100:.1f}% (n={len(cat_results)})")

            print(f"  Eval time:         {eval_time:.1f}s")

    # Multi-run summary
    if runs > 1 and run_accuracies:
        import statistics
        mean_acc = statistics.mean(run_accuracies) * 100
        stddev = statistics.stdev(run_accuracies) * 100 if len(run_accuracies) > 1 else 0
        print(f"\n{'=' * 70}")
        print(f"MULTI-RUN SUMMARY ({runs} runs)")
        print(f"{'=' * 70}")
        print(f"  Accuracy: {mean_acc:.1f}% +/- {stddev:.1f}%")
        print(f"  Per-run:  {['%.1f%%' % (a*100) for a in run_accuracies]}")

    # Save results
    results_path = os.path.join(
        Path(__file__).parent, "fixtures", "locomo_results_atomized.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    output = {
        "mode": "atomized",
        "description": "One DB per conversation (production-realistic architecture)",
        "num_questions": len(results),
        "num_conversations": len(db_map),
        "accuracy": run_accuracies[-1] if run_accuracies else 0,
        "run_accuracies": run_accuracies,
        "per_question": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "category": r.category,
                "correct": r.llm_judge_score == 1,
                "answer": r.generated_answer,
                "ground_truth": r.ground_truth,
                "recall_at_5": r.recall_at_5,
                "recall_at_10": r.recall_at_10,
                "recall_at_20": r.recall_at_20,
            }
            for r in results
        ],
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    for mem in mem_instances.values():
        mem.close()
    if _temp_ctx and not db_dir.startswith(str(Path(__file__).parent)):
        shutil.rmtree(_temp_ctx, ignore_errors=True)

    return None  # Full EvaluationResults integration if needed later


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo Atomized Evaluation (one DB per conversation)"
    )
    parser.add_argument("--dataset", type=str,
                        default=str(Path(__file__).parent.parent / "tests" / "benchmarks" / "fixtures" / "locomo10.json"),
                        help="Path to locomo10.json")
    parser.add_argument("--db-dir", type=str, default=None,
                        help="Directory with per-conversation .mfdb files")
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Use existing DBs (requires --db-dir)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable LLM entity extraction during ingestion")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Path to GGUF model file")
    parser.add_argument("--llm-tier", type=str, default="quality",
                        help="Model tier: balanced or quality")
    parser.add_argument("--extraction-passes", type=int, default=1)
    parser.add_argument("--num-conversations", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs (for mean +/- stddev)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    if args.skip_ingestion and not args.db_dir:
        print("ERROR: --skip-ingestion requires --db-dir")
        sys.exit(1)

    run_atomized_evaluation(
        dataset_path=args.dataset,
        db_dir=args.db_dir,
        num_conversations=args.num_conversations,
        max_questions=args.max_questions,
        categories=args.categories,
        use_llm=args.use_llm,
        llm_model_path=args.llm_model,
        llm_tier=args.llm_tier,
        extraction_passes=args.extraction_passes,
        skip_ingestion=args.skip_ingestion,
        runs=args.runs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
