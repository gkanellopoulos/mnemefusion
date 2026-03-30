#!/usr/bin/env python3
"""
BIM Phase 2: Escalation Query — Salience-Only -> Consolidation Cycles

Takes master DBs from Phase 1 (run_ingest_bim.py) and tests escalating
conditions on each question:

  Condition 0:  salience-only (query master DB, no consolidation)
  Condition 1:  1 consolidation cycle, then query
  Condition 3:  3 cycles, then query
  Condition 5:  5 cycles, then query
  Condition 10: 10 cycles, then query
  ...           (configurable via --cycles)

For each question × condition, the script:
  1. Copies the master .mfdb to a temp working copy
  2. Runs N consolidation cycles (with embedding fn for narratives)
  3. Queries and generates an answer via RAG (GPT-5-mini)
  4. Judges the answer (GPT-4o, official LongMemEval protocol)
  5. Records score, recall@K, consolidation stats, latency

Early-stop mode (--early-stop): stops escalating a question once it scores 1.
This answers "at which condition does retrieval first succeed?"

Usage:
    # Test all 500 questions, default cycle counts [0, 1, 3, 5, 10]
    python run_query_bim.py --db-dir ./bim_master_dbs

    # Custom cycle counts
    python run_query_bim.py --db-dir ./bim_master_dbs --cycles 0,1,2,3,5,10,20

    # Early-stop: stop escalating once a question succeeds
    python run_query_bim.py --db-dir ./bim_master_dbs --early-stop

    # Only failed s-mode questions (314)
    python run_query_bim.py --db-dir ./bim_master_dbs --failed-only

    # Parallel workers (no GPU needed — consolidation uses LLM but query doesn't)
    python run_query_bim.py --db-dir ./bim_master_dbs --num-workers 4 --worker 0 \\
        --llm-model /path/to/model.gguf

    # Multi-GPU for consolidation LLM calls
    python run_query_bim.py --db-dir ./bim_master_dbs --num-workers 4 --worker 0 \\
        --llm-model /path/to/model.gguf --gpu-ids 0,1
"""

import argparse
import gc
import json
import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict


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

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        pin_gpu(args.worker, gpu_ids)

    try:
        import torch  # noqa: F401
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

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. pip install openai")
        sys.exit(1)

    run(args, SentenceTransformer, mnemefusion, OpenAI)


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RAG_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-4o-2024-08-06"
TOP_K = 20
DEFAULT_CYCLES = [0, 1, 3, 5, 10]

MODEL_PRICING = {
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-5-mini":        {"input": 0.25, "output": 2.00},
}

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "longmemeval"

RAG_PROMPT = """You are a helpful assistant answering questions based on conversation history.

Retrieved memories (dates in brackets show when each conversation occurred):
{context}
{date_line}
Question: {question}

Answer the question based on the information in the retrieved memories. Look carefully through ALL the memories for relevant details.
For temporal questions, use the dates in brackets to calculate the answer.
If you find ANY relevant information, provide an answer. Only say "I don't have enough information" if the memories truly contain nothing related to the question.
Keep your answer very concise — ideally a few words or one sentence."""

# Official LongMemEval judge prompts (binary yes/no)
JUDGE_PROMPTS = {
    "general": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}
Correct Answer: {gold_answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",

    "temporal-reasoning": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. The answer may involve specific time information. If the model response gives a time that is off by one (e.g., has the wrong day of the week but the correct date), answer yes.

Question: {question}
Correct Answer: {gold_answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",

    "knowledge-update": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. The correct answer may contain the most up-to-date information, but the model response may give both the outdated and updated information, or just the most updated information. If the model response gives the updated answer, answer yes.

Question: {question}
Correct Answer: {gold_answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",

    "single-session-preference": """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. The answer may involve preference of the user. Please be lenient and answer yes if the model response conveys the same preference as the correct answer even if the specific entities mentioned are not exactly the same.

Question: {question}
Correct Answer: {gold_answer}
Model Response: {hypothesis}

Is the model response correct? Answer yes or no only.""",

    "abstention": """I will give you a question, and a response from a model. The question DOES NOT have an answer (there is no relevant information in the conversation). Please answer yes if the model response correctly identifies that the question cannot be answered with the available information. Answer no if the model fabricates an answer.

Question: {question}
Model Response: {hypothesis}

Does the model correctly identify that the question cannot be answered? Answer yes or no only.""",
}


# =============================================================================
# Dataset
# =============================================================================

def load_dataset(dataset_override: Optional[str] = None) -> List[Dict]:
    """Load the LongMemEval s-mode dataset.

    Uses slim pre-processed file (0.5MB) instead of full dataset (265MB)
    to avoid MemoryError. Build slim file with: python build_slim_dataset.py
    """
    if dataset_override:
        p = Path(dataset_override)
        if not p.exists():
            print(f"ERROR: Dataset not found at {p}")
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    slim_path = FIXTURES_DIR / "longmemeval_s_slim.json"
    if slim_path.exists():
        with open(slim_path, encoding="utf-8") as f:
            return json.load(f)

    # Fallback to full dataset (may OOM on low-memory machines)
    path = FIXTURES_DIR / "longmemeval_s_cleaned.json"
    if not path.exists():
        print(f"ERROR: Dataset not found. Run build_slim_dataset.py first, or place {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    for entry in raw:
        entry["_gold_turn_contents"] = get_gold_turn_contents(entry)
        entry.pop("haystack_sessions", None)
    return raw


def load_failed_question_ids() -> set:
    """Load question IDs that failed in the s-mode evaluation (score=0)."""
    path = FIXTURES_DIR / "longmemeval_results_s_combined.json"
    if not path.exists():
        print(f"ERROR: Combined s-mode results not found at {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return {r["question_id"] for r in results if r.get("score", 1) == 0}


def get_gold_turn_contents(entry: Dict) -> List[str]:
    """Extract content of turns marked with has_answer=True."""
    gold = []
    for session in entry["haystack_sessions"]:
        for turn in session:
            if turn.get("has_answer"):
                gold.append(turn["content"].strip())
    return gold


def compute_recall(retrieved: List[str], gold: List[str], k: int) -> float:
    """Recall@K: fraction of gold turns found in top-K retrieved."""
    if not gold:
        return 1.0
    top_k = retrieved[:k]
    found = 0
    for g in gold:
        for r in top_k:
            if g in r or r in g:
                found += 1
                break
    return found / len(gold)


# =============================================================================
# Consolidation
# =============================================================================

def run_consolidation_cycles(mem, embed_fn, num_cycles: int) -> Dict:
    """Run N consolidation cycles on a memory instance.

    Returns aggregate stats across all cycles.
    """
    if num_cycles == 0:
        return {"cycles": 0, "total_ms": 0}

    total_ms = 0
    agg = defaultdict(int)

    for cycle in range(num_cycles):
        try:
            report = mem.consolidate(embedding_fn=embed_fn)
            total_ms += report.get("duration_ms", 0)
            for key in ("re_evaluated", "reinforced", "merged", "expanded",
                        "promoted", "decayed"):
                agg[key] += report.get(key, 0)
        except Exception as e:
            print(f"    WARNING: consolidation cycle {cycle+1} failed: {e}")
            break

    return {
        "cycles": num_cycles,
        "total_ms": total_ms,
        "re_evaluated": agg["re_evaluated"],
        "reinforced": agg["reinforced"],
        "merged": agg["merged"],
        "expanded": agg["expanded"],
        "promoted": agg["promoted"],
        "decayed": agg["decayed"],
    }


# =============================================================================
# Query & Judge
# =============================================================================

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def query_and_answer(
    mem, question: str, embedder, client, question_date: str = "",
) -> Tuple[str, List[str], list, float, float]:
    """Query MnemeFusion and generate answer via RAG.

    Returns: (hypothesis, raw_contents, results, latency_ms, cost)
    """
    query_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()

    t0 = time.time()
    intent_info, results, profile_context = mem.query(question, query_embedding, TOP_K)
    latency_ms = (time.time() - t0) * 1000

    context_parts = []
    raw_contents = []
    for result_dict, scores_dict in results:
        content = result_dict.get("content", "")
        raw_contents.append(content)
        metadata = result_dict.get("metadata", {})
        session_date = metadata.get("session_date", "")
        speaker = metadata.get("speaker", "")

        if session_date:
            formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
        else:
            formatted = f"{speaker}: {content}" if speaker else content
        context_parts.append(formatted)

    if profile_context:
        context_parts.append(f"[Profile summary] {profile_context}")

    context_str = "\n".join([f"- {c}" for c in context_parts])
    date_line = f"\nCurrent date: {question_date}" if question_date else ""

    prompt = RAG_PROMPT.format(context=context_str, date_line=date_line, question=question)
    cost = 0.0
    try:
        rag_kwargs = {"model": RAG_MODEL, "messages": [{"role": "user", "content": prompt}]}
        if not RAG_MODEL.startswith("gpt-5"):
            rag_kwargs["max_tokens"] = 200
            rag_kwargs["temperature"] = 0
        response = client.chat.completions.create(**rag_kwargs)
        answer = response.choices[0].message.content.strip()
        if response.usage:
            cost = estimate_cost(RAG_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
    except Exception as e:
        print(f"    [ERROR] Answer generation failed: {e}")
        answer = "Error generating answer"

    return answer, raw_contents, results, latency_ms, cost


def judge_answer(
    question: str, gold_answer: str, hypothesis: str, client,
    question_type: str = "", question_id: str = "",
) -> Tuple[int, str, float]:
    """Official LongMemEval binary yes/no judge."""
    if question_id.endswith("_abs"):
        prompt_key = "abstention"
    elif question_type in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
        prompt_key = question_type
    else:
        prompt_key = "general"

    prompt_template = JUDGE_PROMPTS[prompt_key]

    if prompt_key == "abstention":
        prompt = prompt_template.format(question=question, hypothesis=hypothesis)
    else:
        prompt = prompt_template.format(
            question=question, gold_answer=gold_answer, hypothesis=hypothesis,
        )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        cost = 0.0
        if response.usage:
            cost = estimate_cost(JUDGE_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
        score = 1 if "yes" in raw.lower() else 0
        return score, raw, cost
    except Exception as e:
        print(f"    [ERROR] Judge failed: {e}")
        return -1, f"Judge error: {e}", 0.0


# =============================================================================
# Single Question × Condition
# =============================================================================

def evaluate_condition(
    entry: Dict,
    num_cycles: int,
    master_db_path: str,
    embedder,
    mnemefusion,
    client,
    llm_model_path: Optional[str],
    work_dir: str,
    enable_trace: bool = False,
) -> Dict:
    """Evaluate one question under one consolidation condition.

    1. Copy master DB to work_dir
    2. Run consolidation cycles
    3. Query + RAG + Judge
    4. Return result dict
    """
    qid = entry["question_id"]
    qtype = entry["question_type"]
    question = entry["question"]
    gold_answer = str(entry["answer"])
    question_date = entry.get("question_date", "")
    condition_name = f"cycles-{num_cycles}"

    # For cycles=0 (read-only), open master DB directly to avoid copy + MemoryError.
    # For cycles>0 (consolidation mutates DB), copy to work_dir first.
    # Always copy to temp — prevents stale lock files on master DBs after crash
    work_db = os.path.join(work_dir, f"{qid}_{condition_name}.mfdb")
    for _attempt in range(5):
        try:
            shutil.copy2(master_db_path, work_db)
            break
        except PermissionError:
            time.sleep(2)
    else:
        shutil.copy2(master_db_path, work_db)  # final attempt, let it raise

    embedding_dim = embedder.get_sentence_embedding_dimension()
    config = {"embedding_dim": embedding_dim}
    if enable_trace:
        config["enable_trace"] = True
    mem = mnemefusion.Memory(work_db, config)

    if llm_model_path:
        mem.enable_llm_entity_extraction(llm_model_path, "quality", 1)

    embed_fn = lambda text: embedder.encode(text, show_progress_bar=False).tolist()
    mem.set_embedding_fn(embed_fn)
    mem.set_user_entity("user")

    # Consolidation
    consol_stats = {}
    if num_cycles > 0:
        print(f"    Consolidating ({num_cycles} cycles)...", end=" ", flush=True)
        t0 = time.time()
        consol_stats = run_consolidation_cycles(mem, embed_fn, num_cycles)
        consol_time = time.time() - t0
        print(f"({consol_time:.1f}s, merged={consol_stats.get('merged', 0)}, "
              f"expanded={consol_stats.get('expanded', 0)}, "
              f"decayed={consol_stats.get('decayed', 0)})")

        # Refresh profiles after consolidation
        try:
            mem.summarize_profiles()
            mem.precompute_fact_embeddings()
        except Exception:
            pass

    # Query
    hypothesis, raw_contents, results, latency_ms, rag_cost = query_and_answer(
        mem, question, embedder, client, question_date=question_date,
    )

    # Capture trace (before closing mem)
    query_trace = None
    if enable_trace:
        query_trace = mem.last_query_trace()

    # Recall
    gold_contents = entry.get("_gold_turn_contents")
    if gold_contents is None:
        gold_contents = get_gold_turn_contents(entry)
    r5 = compute_recall(raw_contents, gold_contents, 5)
    r10 = compute_recall(raw_contents, gold_contents, 10)
    r20 = compute_recall(raw_contents, gold_contents, 20)

    # Judge
    score, reasoning, judge_cost = judge_answer(
        question, gold_answer, hypothesis, client,
        question_type=qtype, question_id=qid,
    )
    total_cost = rag_cost + judge_cost

    # Top-K diagnostics
    top_k_scores = [
        {
            "rank": i + 1,
            "fused_score": round(sd.get("fused_score", 0), 6),
            "salience_score": sd.get("salience_score"),
            "semantic_score": round(sd.get("semantic_score", 0), 4),
            "entity_score": round(sd.get("entity_score", 0), 4),
            "is_narrative": bool(rd.get("metadata", {}).get("__mf_merged__")),
            "content_preview": rd.get("content", "")[:80],
        }
        for i, (rd, sd) in enumerate(results[:20])
    ]

    label = "YES" if score == 1 else "NO" if score == 0 else "ERR"
    print(f"    -> {label}  R@5={r5:.0%} R@10={r10:.0%} R@20={r20:.0%}  "
          f"latency={latency_ms:.0f}ms  cost=${total_cost:.4f}")
    print(f"      Answer: {hypothesis[:100].encode('ascii', 'replace').decode()}")

    result = {
        "question_id": qid,
        "question_type": qtype,
        "condition": condition_name,
        "num_cycles": num_cycles,
        "question": question,
        "gold_answer": gold_answer,
        "hypothesis": hypothesis,
        "score": score,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "api_cost": round(total_cost, 6),
        "recall_at_5": round(r5, 4),
        "recall_at_10": round(r10, 4),
        "recall_at_20": round(r20, 4),
        "num_gold_turns": len(gold_contents),
        "top_k_scores": top_k_scores,
    }
    if question_date:
        result["question_date"] = question_date
    if consol_stats:
        result["consolidation_summary"] = consol_stats
    if query_trace is not None:
        result["_trace"] = query_trace

    # Cleanup
    del mem
    gc.collect()
    if work_db != master_db_path:
        try:
            os.remove(work_db)
        except OSError:
            pass

    return result


# =============================================================================
# Main
# =============================================================================

def run(args, SentenceTransformer, mnemefusion, OpenAI):
    data = load_dataset(getattr(args, 'dataset', None))
    print(f"Loaded {len(data)} questions")

    # Build question lookup
    question_map = {e["question_id"]: e for e in data}

    # Filter to failed questions only
    if args.failed_only:
        failed_ids = load_failed_question_ids()
        data = [e for e in data if e["question_id"] in failed_ids]
        print(f"Filtered to {len(data)} failed s-mode questions")

    if args.max_questions:
        data = data[:args.max_questions]
        print(f"Limited to {len(data)} questions")

    # Parse cycle counts
    cycles = [int(x) for x in args.cycles.split(",")]
    cycles.sort()

    # Filter to questions with available master DBs
    db_dir = Path(args.db_dir)
    available = []
    missing = []
    for entry in data:
        db_path = db_dir / f"{entry['question_id']}.mfdb"
        if db_path.exists():
            available.append(entry)
        else:
            missing.append(entry["question_id"])
    if missing:
        print(f"Note: {len(missing)} master DBs not yet available (skipping)")
    if not available:
        print(f"ERROR: No master DBs found in {db_dir}")
        print(f"  Run run_ingest_bim.py first to create master DBs.")
        sys.exit(1)
    data = available

    # Load results (crash-safe resume)
    worker_suffix = f"_w{args.worker}" if args.num_workers > 1 else ""
    results_path = db_dir / f"query_results{worker_suffix}.json"
    trace_path = db_dir / f"query_traces{worker_suffix}.jsonl"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results")
    else:
        all_results = []

    done_keys = {(r["question_id"], r["num_cycles"]) for r in all_results}

    # Worker partitioning
    my_questions = []
    for q_idx, entry in enumerate(data):
        if args.num_workers > 1 and (q_idx % args.num_workers) != args.worker:
            continue
        my_questions.append(entry)

    # Initialize
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    client = OpenAI()

    # Work directory for temporary DB copies
    work_dir = tempfile.mkdtemp(prefix="bim_query_")

    print(f"\n{'=' * 70}")
    print(f"BIM Phase 2: Escalation Query")
    print(f"  Master DBs:    {db_dir}")
    print(f"  Cycles:        {cycles}")
    print(f"  Questions:     {len(my_questions)}")
    print(f"  Worker:        {args.worker}/{args.num_workers}")
    print(f"  Early stop:    {args.early_stop}")
    print(f"  LLM model:     {args.llm_model or 'NONE (no consolidation LLM)'}")
    print(f"  RAG model:     {RAG_MODEL}")
    print(f"  Judge model:   {JUDGE_MODEL}")
    print(f"  Results:       {results_path}")
    if args.enable_trace:
        print(f"  Traces:        {trace_path}")
    print(f"{'=' * 70}\n")

    # Warn if consolidation requested without LLM
    if max(cycles) > 0 and not args.llm_model:
        print("WARNING: Consolidation cycles > 0 requested but --llm-model not provided.")
        print("  Consolidation requires LLM for narrative generation.")
        print("  Cycles > 0 will likely produce no merges/expansions.\n")

    # Process questions
    run_start = time.time()
    processed = 0
    category_scores = defaultdict(lambda: defaultdict(list))  # cat -> cycle -> [scores]

    for entry in my_questions:
        qid = entry["question_id"]
        qtype = entry["question_type"]
        master_db = str(db_dir / f"{qid}.mfdb")

        print(f"\n[{processed+1}/{len(my_questions)}] {qid} ({qtype})")
        print(f"  Q: {entry['question'][:100]}")

        solved_at = None  # Track first successful condition

        for num_cycles in cycles:
            condition_name = f"cycles-{num_cycles}"

            # Skip if already evaluated
            if (qid, num_cycles) in done_keys:
                # Check if it was a success (for early-stop logic)
                prev = next((r for r in all_results
                            if r["question_id"] == qid and r["num_cycles"] == num_cycles), None)
                if prev and prev.get("score") == 1:
                    solved_at = num_cycles
                print(f"  [{condition_name}] Already done — skipping")
                if args.early_stop and solved_at is not None:
                    print(f"  -> Early stop: solved at cycles-{solved_at}")
                    break
                continue

            # Early stop: skip remaining conditions if already solved
            if args.early_stop and solved_at is not None:
                print(f"  [{condition_name}] Skipped (solved at cycles-{solved_at})")
                continue

            print(f"  [{condition_name}]")
            result = evaluate_condition(
                entry, num_cycles, master_db, embedder, mnemefusion,
                client, args.llm_model, work_dir,
                enable_trace=args.enable_trace,
            )

            # Extract trace before appending to results (traces go to separate file)
            trace_data = result.pop("_trace", None)

            all_results.append(result)
            done_keys.add((qid, num_cycles))
            category_scores[qtype][num_cycles].append(result["score"])

            # Save after each condition (crash-safe)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            # Append trace to JSONL file (one line per question, crash-safe)
            if trace_data is not None:
                trace_entry = {
                    "question_id": qid,
                    "question_type": qtype,
                    "num_cycles": num_cycles,
                    "question": entry["question"],
                    "score": result["score"],
                    "trace": trace_data,
                }
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

            if result["score"] == 1:
                solved_at = num_cycles

            if args.early_stop and solved_at is not None:
                print(f"  -> Early stop: solved at cycles-{solved_at}")
                break

        processed += 1

        # Progress summary every 25 questions
        if processed % 25 == 0:
            elapsed = time.time() - run_start
            print(f"\n  --- Progress: {processed}/{len(my_questions)} "
                  f"({elapsed/3600:.1f}h elapsed) ---")
            print_running_summary(all_results, cycles)
            print()

    # Cleanup work directory
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    # Final summary
    elapsed = time.time() - run_start
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY ({processed} questions, {elapsed/3600:.2f}h)")
    print(f"{'=' * 70}")
    print_running_summary(all_results, cycles)
    print_escalation_analysis(all_results, cycles)
    print(f"\n  Results saved to: {results_path}")

    # Merge hint
    if args.num_workers > 1:
        print(f"\n  To merge all worker results:")
        print(f"    python run_query_bim.py --db-dir {args.db_dir} --merge")


def print_running_summary(results: List[Dict], cycles: List[int]):
    """Print accuracy breakdown by condition and category."""
    for num_cycles in cycles:
        cond_results = [r for r in results if r.get("num_cycles") == num_cycles and r["score"] >= 0]
        if not cond_results:
            continue
        acc = sum(r["score"] for r in cond_results) / len(cond_results) * 100
        avg_r5 = sum(r.get("recall_at_5", 0) for r in cond_results) / len(cond_results)
        avg_r10 = sum(r.get("recall_at_10", 0) for r in cond_results) / len(cond_results)
        avg_r20 = sum(r.get("recall_at_20", 0) for r in cond_results) / len(cond_results)
        print(f"  cycles-{num_cycles:<4}  n={len(cond_results):>4}  acc={acc:>5.1f}%  "
              f"R@5={avg_r5:.1%} R@10={avg_r10:.1%} R@20={avg_r20:.1%}")


def print_escalation_analysis(results: List[Dict], cycles: List[int]):
    """Analyze at which condition each question first succeeds."""
    # Group by question
    by_question = defaultdict(dict)
    for r in results:
        if r["score"] >= 0:
            by_question[r["question_id"]][r.get("num_cycles", 0)] = r["score"]

    # Count where each question first succeeds
    first_success = defaultdict(int)  # cycle_count -> num_questions
    never_solved = 0

    for qid, cycle_scores in by_question.items():
        solved = False
        for nc in sorted(cycle_scores.keys()):
            if cycle_scores[nc] == 1:
                first_success[nc] += 1
                solved = True
                break
        if not solved:
            never_solved += 1

    total = len(by_question)
    print(f"\n  ESCALATION ANALYSIS ({total} questions)")
    print(f"  {'Condition':<15} {'First solved':>12} {'Cumulative':>10} {'% of total':>10}")
    print(f"  {'-' * 50}")

    cumulative = 0
    for nc in sorted(first_success.keys()):
        count = first_success[nc]
        cumulative += count
        pct = cumulative / total * 100
        print(f"  cycles-{nc:<7} {count:>12} {cumulative:>10} {pct:>9.1f}%")

    if never_solved:
        print(f"  {'never':<15} {never_solved:>12} {'':>10} "
              f"{never_solved/total*100:>9.1f}%")

    # Key insight
    if first_success:
        sal_only = first_success.get(0, 0)
        if sal_only > 0:
            print(f"\n  -> Salience-only recovers {sal_only}/{total} ({sal_only/total*100:.1f}%) "
                  f"of previously-failed questions")
        consolidation_total = sum(v for k, v in first_success.items() if k > 0)
        if consolidation_total > 0:
            print(f"  -> Consolidation adds {consolidation_total} more "
                  f"({consolidation_total/total*100:.1f}%)")


def merge_worker_results(args):
    """Merge per-worker result files into a single combined file."""
    import glob as glob_mod

    db_dir = Path(args.db_dir)
    pattern = str(db_dir / "query_results_w*.json")
    worker_files = sorted(glob_mod.glob(pattern))

    if not worker_files:
        print(f"ERROR: No worker files found matching {pattern}")
        sys.exit(1)

    all_results = []
    seen = set()
    for wf in worker_files:
        with open(wf, encoding="utf-8") as f:
            results = json.load(f)
        for r in results:
            key = (r["question_id"], r.get("num_cycles", 0))
            if key not in seen:
                all_results.append(r)
                seen.add(key)
        print(f"  Loaded {len(results)} results from {Path(wf).name}")

    combined_path = db_dir / "query_results_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nMerged {len(all_results)} results into {combined_path}")

    # Print summary
    cycles = sorted(set(r.get("num_cycles", 0) for r in all_results))
    print_running_summary(all_results, cycles)
    print_escalation_analysis(all_results, cycles)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BIM Phase 2: Escalation Query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default escalation: 0, 1, 3, 5, 10 cycles
  python run_query_bim.py --db-dir ./bim_master_dbs

  # Custom cycles
  python run_query_bim.py --db-dir ./bim_master_dbs --cycles 0,1,2,5,10,20

  # Early stop (stop escalating once question succeeds)
  python run_query_bim.py --db-dir ./bim_master_dbs --early-stop

  # Parallel workers with GPU for consolidation
  python run_query_bim.py --db-dir ./bim_master_dbs --num-workers 4 --worker 0 \\
      --llm-model /path/to/model.gguf --gpu-ids 0,1

  # Merge worker results
  python run_query_bim.py --db-dir ./bim_master_dbs --merge
""",
    )
    parser.add_argument("--db-dir", type=str, required=True,
                        help="Directory containing master .mfdb files from Phase 1")
    parser.add_argument("--cycles", type=str, default="0,1,3,5,10",
                        help="Comma-separated consolidation cycle counts (default: 0,1,3,5,10)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Path to LLM model (required for cycles > 0)")
    parser.add_argument("--early-stop", action="store_true",
                        help="Stop escalating a question once it scores 1")
    parser.add_argument("--failed-only", action="store_true",
                        help="Only query the 314 questions that failed s-mode eval")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--worker", type=int, default=0,
                        help="This worker's index (0-based)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for consolidation LLM")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-worker result files and exit")
    parser.add_argument("--enable-trace", action="store_true",
                        help="Enable pipeline tracing and write per-question traces to a JSONL file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to custom dataset JSON (default: longmemeval_s_slim.json)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.merge:
        merge_worker_results(args)
        sys.exit(0)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    main()
