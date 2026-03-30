#!/usr/bin/env python3
"""
MnemeBench Runner -- Evaluates MnemeFusion on the MnemeBench benchmark.

Reads mnemebench.json, opens pre-ingested master .mfdb files, and generates
responses for all 12 task types across 4 dimensions (Synthesis, Proactive,
Temporal, Forgetting).

Output is a responses.json compatible with mnemebench's run_judge.py.

Usage:
    # Full run on 500 haystacks
    python run_mnemebench.py --db-dir ./bim_master_dbs --benchmark /path/to/mnemebench.json

    # Smoke test on 5 haystacks
    python run_mnemebench.py --db-dir ./bim_master_dbs --benchmark mnemebench.json --limit 5

    # Parallel workers
    python run_mnemebench.py --db-dir ./bim_master_dbs --benchmark mnemebench.json \\
        --num-workers 4 --worker 0

    # Run judge after generating responses
    python run_mnemebench.py --db-dir ./bim_master_dbs --benchmark mnemebench.json --judge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must import torch before sentence_transformers on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. pip install sentence-transformers")
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


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RAG_MODEL = "gpt-5-mini"
TOP_K = 20
SYSTEM_NAME = "mnemefusion-bim-salience"

MODEL_PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4o":     {"input": 2.50, "output": 10.00},
}

# RAG prompt for S/P/T tasks -- standard retrieval-based answering
SPT_RAG_PROMPT = """You are a helpful assistant that knows a user through past conversations.
Below are relevant memories retrieved from those conversations.

Retrieved memories:
{context}

Now answer the following question about this user.
{question}

Respond naturally and thoroughly. If you're not sure about something, say so."""

# RAG prompt for F tasks -- budget-constrained memory
F_RECALL_PROMPT = """You are a helpful assistant that knows a user through past conversations.
Your memory is limited -- below is everything you remember about this user.

Available memories (ordered by importance):
{context}

Question: {question}

Answer concisely based only on what you remember. If you don't remember, say so."""

F_SUMMARY_PROMPT = """You are a helpful assistant that knows a user through past conversations.
Your memory is limited -- below is everything you remember about this user.

Available memories (ordered by importance):
{context}

{question}

Provide a coherent one-paragraph response based on what you remember."""


# =============================================================================
# Helpers
# =============================================================================

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def call_rag(client, prompt: str) -> Tuple[str, float]:
    """Call RAG model and return (response_text, cost)."""
    try:
        kwargs = {"model": RAG_MODEL, "messages": [{"role": "user", "content": prompt}]}
        if not RAG_MODEL.startswith("gpt-5"):
            kwargs["max_tokens"] = 500
            kwargs["temperature"] = 0
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        cost = 0.0
        if response.usage:
            cost = estimate_cost(RAG_MODEL, response.usage.prompt_tokens,
                                 response.usage.completion_tokens)
        return text, cost
    except Exception as e:
        print(f"    [ERROR] RAG call failed: {e}")
        return f"Error: {e}", 0.0


def build_context_from_query(results, profile_context) -> str:
    """Build context string from mem.query() results."""
    parts = []
    for result_dict, scores_dict in results:
        content = result_dict.get("content", "")
        metadata = result_dict.get("metadata", {})
        session_date = metadata.get("session_date", "")
        speaker = metadata.get("speaker", "")
        if session_date:
            formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
        else:
            formatted = f"{speaker}: {content}" if speaker else content
        parts.append(formatted)

    if profile_context:
        parts.append(f"[Profile summary] {profile_context}")

    return "\n".join([f"- {p}" for p in parts])


def build_context_from_budget(memories: list) -> str:
    """Build context string from get_memories_by_salience() results."""
    parts = []
    for mem_dict in memories:
        content = mem_dict.get("content", "")
        metadata = mem_dict.get("metadata", {})
        session_date = metadata.get("session_date", "")
        speaker = metadata.get("speaker", "")
        if session_date:
            formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
        else:
            formatted = f"{speaker}: {content}" if speaker else content
        parts.append(formatted)
    return "\n".join([f"- {p}" for p in parts])


# =============================================================================
# Task Answerers
# =============================================================================

def answer_spt_task(question: str, mem, embedder, client) -> Tuple[str, float]:
    """Answer an S/P/T task using standard retrieval."""
    query_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()
    intent_info, results, profile_context = mem.query(question, query_embedding, TOP_K)
    context = build_context_from_query(results, profile_context)
    prompt = SPT_RAG_PROMPT.format(context=context, question=question)
    return call_rag(client, prompt)


def answer_f1_tiered(task: dict, mem, client) -> Tuple[dict, float]:
    """Answer F1 tiered recall at each budget level."""
    budgets = task["budget_levels"]
    questions = task["questions"]
    total_cost = 0.0
    result = {}

    for budget in budgets:
        bstr = str(budget)
        result[bstr] = {}
        memories = mem.get_memories_by_salience(budget)
        context = build_context_from_budget(memories)

        for tier in ["core", "significant", "ephemeral"]:
            tier_q = questions.get(tier)
            if tier_q is None:
                result[bstr][tier] = None
                continue
            prompt = F_RECALL_PROMPT.format(context=context, question=tier_q["question"])
            answer, cost = call_rag(client, prompt)
            result[bstr][tier] = answer
            total_cost += cost

    return result, total_cost


def answer_f2_collateral(task: dict, mem, client) -> Tuple[Optional[dict], float]:
    """Answer F2 collateral damage at fixed budget."""
    if task is None:
        return None, 0.0

    budget = task["budget_level"]
    rings = task["rings"]
    memories = mem.get_memories_by_salience(budget)
    context = build_context_from_budget(memories)
    total_cost = 0.0
    result = {}

    for ring_key in ["ring1", "ring2", "ring3"]:
        ring_data = rings[ring_key]
        prompt = F_RECALL_PROMPT.format(context=context, question=ring_data["question"])
        answer, cost = call_rag(client, prompt)
        result[ring_key] = answer
        total_cost += cost

    return result, total_cost


def answer_f3_coherence(task: dict, mem, client) -> Tuple[dict, float]:
    """Answer F3 coherence summary at each budget level."""
    budgets = task["budget_levels"]
    question = task["question"]
    total_cost = 0.0
    result = {}

    for budget in budgets:
        bstr = str(budget)
        memories = mem.get_memories_by_salience(budget)
        context = build_context_from_budget(memories)
        prompt = F_SUMMARY_PROMPT.format(context=context, question=question)
        answer, cost = call_rag(client, prompt)
        result[bstr] = answer
        total_cost += cost

    return result, total_cost


# =============================================================================
# Process One Haystack
# =============================================================================

def process_haystack(
    haystack: dict,
    db_dir: str,
    embedder,
    client,
) -> Tuple[Optional[dict], float]:
    """Process all tasks for one haystack. Returns (responses_dict, total_cost)."""
    hid = haystack["longmemeval_id"]
    tasks = haystack["tasks"]

    db_path = os.path.join(db_dir, f"{hid}.mfdb")
    if not os.path.exists(db_path):
        print(f"    SKIP: {db_path} not found")
        return None, 0.0

    # Open the master DB (read-only -- no consolidation)
    config = {"enable_salience_reranking": True}
    mem = mnemefusion.Memory(db_path, config)
    embed_fn = lambda text: embedder.encode([text], show_progress_bar=False)[0].tolist()
    mem.set_embedding_fn(embed_fn)
    mem.set_user_entity("user")

    total_cost = 0.0
    resp = {}

    # S/P/T tasks -- standard retrieval
    for task_key in ["S1_profile", "S2_inference", "S3_trait_verify",
                     "P1_opportunity", "P2_false_positive", "P3_depth",
                     "T1_trajectory", "T2_contradiction", "T3_projection"]:
        task = tasks.get(task_key)
        if task is None:
            resp[task_key] = None
            continue
        answer, cost = answer_spt_task(task["question"], mem, embedder, client)
        resp[task_key] = answer
        total_cost += cost

    # F1: tiered recall at each budget
    if tasks.get("F1_tiered"):
        f1_resp, f1_cost = answer_f1_tiered(tasks["F1_tiered"], mem, client)
        resp["F1_tiered"] = f1_resp
        total_cost += f1_cost
    else:
        resp["F1_tiered"] = None

    # F2: collateral damage
    f2_resp, f2_cost = answer_f2_collateral(tasks.get("F2_collateral"), mem, client)
    resp["F2_collateral"] = f2_resp
    total_cost += f2_cost

    # F3: coherence summary
    if tasks.get("F3_coherence"):
        f3_resp, f3_cost = answer_f3_coherence(tasks["F3_coherence"], mem, client)
        resp["F3_coherence"] = f3_resp
        total_cost += f3_cost
    else:
        resp["F3_coherence"] = None

    del mem
    return resp, total_cost


# =============================================================================
# Main Run Loop
# =============================================================================

def run(args):
    # Load benchmark
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    haystacks = benchmark["haystacks"]
    if args.limit:
        haystacks = haystacks[:args.limit]

    total = len(haystacks)
    print(f"MnemeBench Runner -- {total} haystacks")
    print(f"  DB dir:     {args.db_dir}")
    print(f"  RAG model:  {RAG_MODEL}")
    print(f"  Workers:    {args.num_workers} (this is worker {args.worker})")

    # Filter to this worker's partition
    my_haystacks = [(i, h) for i, h in enumerate(haystacks)
                    if i % args.num_workers == args.worker]
    print(f"  My share:   {len(my_haystacks)} haystacks")

    # Load embedder
    print(f"  Loading {EMBEDDING_MODEL}...", flush=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("  Embedder ready.")

    # OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Output path
    suffix = f"_w{args.worker}" if args.num_workers > 1 else ""
    output_path = os.path.join(args.db_dir, f"mnemebench_responses{suffix}.json")

    # Load existing progress (crash-safe resume)
    responses = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            responses = existing.get("responses", {})
            print(f"  Resuming: {len(responses)} haystacks already done")
        except Exception:
            pass

    total_cost = 0.0
    t_start = time.time()

    for idx, (orig_idx, haystack) in enumerate(my_haystacks):
        hid = haystack["longmemeval_id"]

        if hid in responses:
            continue  # Already done (crash-safe)

        print(f"[{idx+1}/{len(my_haystacks)}] {hid}...", end=" ", flush=True)
        t0 = time.time()

        resp, cost = process_haystack(haystack, args.db_dir, embedder, client)
        elapsed = time.time() - t0
        total_cost += cost

        if resp is not None:
            responses[hid] = resp
            print(f"done ({elapsed:.1f}s, ${cost:.3f})")
        else:
            print(f"skipped")

        # Save after each haystack (crash-safe)
        output_data = {
            "system_name": SYSTEM_NAME,
            "responses": responses,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - t_start
    print(f"\nDone. {len(responses)} haystacks processed in {elapsed_total:.0f}s")
    print(f"Total RAG cost: ${total_cost:.2f}")
    print(f"Output: {output_path}")

    # Optionally run the judge
    if args.judge:
        judge_path = Path(args.benchmark).parent / "run_judge.py"
        if judge_path.exists():
            print(f"\nRunning judge: {judge_path}")
            os.system(f'python "{judge_path}" --responses "{output_path}" --benchmark "{args.benchmark}"')
        else:
            print(f"\nWARNING: run_judge.py not found at {judge_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MnemeBench Runner for MnemeFusion")
    parser.add_argument("--db-dir", required=True,
                        help="Directory containing master .mfdb files (from Phase 1)")
    parser.add_argument("--benchmark", required=True,
                        help="Path to mnemebench.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N haystacks (for testing)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--worker", type=int, default=0,
                        help="This worker's index (0-based)")
    parser.add_argument("--judge", action="store_true",
                        help="Run run_judge.py after generating responses")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
