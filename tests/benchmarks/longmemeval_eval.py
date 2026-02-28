#!/usr/bin/env python3
"""
LongMemEval Stepped Evaluation for MnemeFusion

Iterative development benchmark: processes questions one at a time with full
LLM entity extraction, GPT-4o percentage-based judging.

NOT for publication — for validating our library works on a second benchmark
before investing in GPU rental for a full run.

Modes:
  oracle  — Evidence-only sessions (~36 turns/q, ~5 min/q). Tests extraction + RAG.
             If we fail here, the problem is us, not retrieval.
  s       — Full haystack (~490 turns/q, ~75 min/q). Tests full retrieval pipeline.

Usage:
    # Oracle mode (recommended first)
    python longmemeval_eval.py --mode oracle \\
        --llm-model ../../models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

    # Full haystack mode
    python longmemeval_eval.py --mode s \\
        --llm-model ../../models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

    # Resume from question N
    python longmemeval_eval.py --mode oracle --start-at 5 \\
        --llm-model ../../models/phi-4-mini/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

Dataset: LongMemEval (ICLR 2025), 500 questions, 6 categories.
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import argparse
import gc
import json
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

try:
    from mnemefusion_cuda_wrapper import mnemefusion
except ImportError:
    try:
        import mnemefusion
    except ImportError:
        print("ERROR: mnemefusion not installed.")
        sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("ERROR: sentence-transformers not installed.")
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
RAG_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-5-mini"  # 10x cheaper than gpt-4o, comparable reasoning quality
TOP_K = 20
EXTRACTION_PASSES = 1

# Pricing per 1M tokens (for cost tracking)
MODEL_PRICING = {
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-5-mini":   {"input": 0.25, "output": 2.00},
    "gpt-5-nano":   {"input": 0.05, "output": 0.40},
}

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "longmemeval"

RAG_PROMPT = """You are a helpful assistant answering questions based on conversation history.

Retrieved memories (dates in brackets show when each conversation occurred):
{context}

Question: {question}

Answer the question based on the information in the retrieved memories. Look carefully through ALL the memories for relevant details.
For temporal questions, use the dates in brackets to calculate the answer.
If you find ANY relevant information, provide an answer. Only say "I don't have enough information" if the memories truly contain nothing related to the question.
Keep your answer concise and factual."""

JUDGE_PROMPT = """You are evaluating a conversational memory system's ability to recall information.

Question: {question}
Ground truth answer: {gold_answer}
System's answer: {hypothesis}

Rate the system's answer on a scale of 0-100:
- 95-100: Correct. Contains the key information from the ground truth.
- 80-94: Mostly correct. Right direction, minor details differ or extra info included.
- 50-79: Partially correct. Some relevant information but missing key parts.
- 20-49: Mostly wrong. Tangentially related but misses the core answer.
- 0-19: Completely wrong, irrelevant, or "I don't know" when the answer exists.

For abstention questions (where the correct answer is that no information exists):
- 95-100 if the system correctly says it doesn't have that information.
- 0-20 if the system fabricates an answer.

Respond with ONLY a JSON object, no other text:
{{"score": <integer 0-100>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# Core Functions
# =============================================================================

def load_dataset(mode: str) -> List[Dict]:
    """Load the LongMemEval dataset."""
    if mode == "oracle":
        path = FIXTURES_DIR / "longmemeval_oracle.json"
    elif mode == "s":
        path = FIXTURES_DIR / "longmemeval_s_cleaned.json"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not path.exists():
        print(f"ERROR: Dataset not found at {path}")
        print(f"Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned")
        sys.exit(1)

    print(f"Loading {path.name}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} questions")
    return data


def ingest_question(
    entry: Dict,
    embedder: SentenceTransformer,
    llm_model_path: Optional[str],
    extraction_passes: int,
    tmp_dir: str,
) -> "mnemefusion.Memory":
    """Create a MnemeFusion instance and ingest all turns for one question."""
    db_path = os.path.join(tmp_dir, "eval.mfdb")
    embedding_dim = embedder.get_sentence_embedding_dimension()
    config = {"embedding_dim": embedding_dim}
    mem = mnemefusion.Memory(db_path, config)

    # Enable LLM extraction if model provided
    if llm_model_path:
        mem.enable_llm_entity_extraction(llm_model_path, "quality", extraction_passes)

    # Set embedding function for fact embeddings
    mem.set_embedding_fn(lambda text: embedder.encode(text, show_progress_bar=False).tolist())

    # Collect all turns with metadata
    sessions = entry["haystack_sessions"]
    dates = entry.get("haystack_dates", [])
    session_ids = entry.get("haystack_session_ids", [])

    turns = []
    for sess_idx, session in enumerate(sessions):
        session_date = dates[sess_idx] if sess_idx < len(dates) else ""
        session_id = str(session_ids[sess_idx]) if sess_idx < len(session_ids) else str(sess_idx)

        for turn_idx, turn in enumerate(session):
            content = turn["content"]
            if not content or not content.strip():
                continue
            metadata = {
                "speaker": turn["role"],
                "session_id": session_id,
                "session_idx": str(sess_idx),
                "turn_idx": str(turn_idx),
                "dialog_id": f"S{session_id}:{turn_idx}",
            }
            if session_date:
                metadata["session_date"] = session_date
            turns.append((content, metadata, session_date))

    # Batch-embed all turns
    contents = [t[0] for t in turns]
    print(f"    Embedding {len(contents)} turns...", end=" ", flush=True)
    t0 = time.time()
    all_embeddings = embedder.encode(contents, show_progress_bar=False, batch_size=64)
    print(f"({time.time() - t0:.1f}s)")

    # Ingest one by one (each triggers LLM extraction if enabled)
    print(f"    Ingesting {len(turns)} turns", end="", flush=True)
    t0 = time.time()
    for i, (content, metadata, session_date) in enumerate(turns):
        embedding = all_embeddings[i].tolist()

        # Parse session_date to Unix timestamp
        timestamp = None
        if session_date:
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%a)", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    dt = datetime.strptime(session_date.strip(), fmt)
                    timestamp = dt.timestamp()
                    break
                except ValueError:
                    continue

        mem.add(content, embedding, metadata, timestamp=timestamp)

        # Progress dots
        if (i + 1) % 50 == 0:
            print(".", end="", flush=True)

    elapsed = time.time() - t0
    print(f" ({elapsed:.1f}s, {len(turns) / max(elapsed, 0.01):.1f} turns/s)")

    # Post-ingestion: summarize profiles and precompute fact embeddings
    try:
        mem.summarize_profiles()
        n = mem.precompute_fact_embeddings()
    except Exception:
        pass

    return mem


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in USD."""
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def query_and_answer(
    mem: "mnemefusion.Memory",
    question: str,
    embedder: SentenceTransformer,
    client: OpenAI,
) -> Tuple[str, List[str], float, float]:
    """Query MnemeFusion and generate answer via RAG. Returns (answer, context, latency_ms, cost)."""
    query_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()

    t0 = time.time()
    intent_info, results, profile_context = mem.query(question, query_embedding, TOP_K)
    latency_ms = (time.time() - t0) * 1000

    # Format retrieved memories as context
    context_parts = []
    for result_dict, scores_dict in results:
        content = result_dict.get("content", "")
        metadata = result_dict.get("metadata", {})
        session_date = metadata.get("session_date", "")
        speaker = metadata.get("speaker", "")

        if session_date:
            formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
        else:
            formatted = f"{speaker}: {content}" if speaker else content
        context_parts.append(formatted)

    # Add profile context if available
    if profile_context:
        context_parts.append(f"[Profile summary] {profile_context}")

    context_str = "\n".join([f"- {c}" for c in context_parts])

    # Generate answer
    prompt = RAG_PROMPT.format(context=context_str, question=question)
    cost = 0.0
    try:
        response = client.chat.completions.create(
            model=RAG_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        if response.usage:
            cost = estimate_cost(RAG_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)
    except Exception as e:
        print(f"    [ERROR] Answer generation failed: {e}")
        answer = "Error generating answer"

    return answer, context_parts, latency_ms, cost


def judge_answer(
    question: str,
    gold_answer: str,
    hypothesis: str,
    client: OpenAI,
) -> Tuple[int, str, float]:
    """Judge correctness with GPT-5 mini, returning percentage score + reasoning + cost."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        hypothesis=hypothesis,
    )

    try:
        # GPT-5 models use reasoning mode which doesn't support temperature=0
        # or max_tokens. Omit both — defaults work fine for short JSON output.
        judge_kwargs = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}]}
        if not JUDGE_MODEL.startswith("gpt-5"):
            judge_kwargs["max_tokens"] = 200
            judge_kwargs["temperature"] = 0
        response = client.chat.completions.create(**judge_kwargs)
        raw = response.choices[0].message.content.strip()
        cost = 0.0
        if response.usage:
            cost = estimate_cost(JUDGE_MODEL, response.usage.prompt_tokens, response.usage.completion_tokens)

        # Handle markdown-wrapped JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        score = int(result.get("score", 0))
        reasoning = result.get("reasoning", "")
        return score, reasoning, cost

    except Exception as e:
        print(f"    [ERROR] Judge failed: {e}")
        return -1, f"Judge error: {e}", 0.0


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def run_evaluation(args):
    """Run the stepped LongMemEval evaluation."""
    # Load dataset
    data = load_dataset(args.mode)

    # Initialize embedder (loaded once, reused across all questions)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize OpenAI client
    client = OpenAI()

    # Results file for persistence
    results_path = FIXTURES_DIR / f"longmemeval_results_{args.mode}.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} previous results from {results_path.name}")
    else:
        all_results = []

    # Category tracking
    category_scores = defaultdict(list)
    for r in all_results:
        category_scores[r["question_type"]].append(r["score"])

    # Print header
    print(f"\n{'=' * 70}")
    print(f"LongMemEval Stepped Evaluation")
    print(f"  Mode:       {args.mode} ({'evidence-only' if args.mode == 'oracle' else 'full haystack'})")
    print(f"  LLM:        {args.llm_model or 'DISABLED (no entity extraction)'}")
    print(f"  RAG model:  {RAG_MODEL}")
    print(f"  Judge:      {JUDGE_MODEL} (percentage-based, ${MODEL_PRICING.get(JUDGE_MODEL, {}).get('input', '?')}/{MODEL_PRICING.get(JUDGE_MODEL, {}).get('output', '?')} per 1M tok)")
    print(f"  Questions:  {len(data)} total, starting at #{args.start_at}")
    print(f"  Results:    {results_path.name}")
    print(f"{'=' * 70}\n")

    completed_ids = {r["question_id"] for r in all_results}
    cumulative_cost = sum(r.get("api_cost", 0.0) for r in all_results)

    for q_idx in range(args.start_at, len(data)):
        entry = data[q_idx]
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        gold_answer = str(entry["answer"])
        is_abstention = qid.endswith("_abs")

        # Skip already-evaluated questions
        if qid in completed_ids:
            continue

        num_sessions = len(entry["haystack_sessions"])
        num_turns = sum(len(s) for s in entry["haystack_sessions"])

        print(f"--- Question {q_idx + 1}/{len(data)} ---")
        print(f"  ID:       {qid}")
        print(f"  Type:     {qtype}{'  [ABSTENTION]' if is_abstention else ''}")
        print(f"  Haystack: {num_sessions} sessions, {num_turns} turns")
        print(f"  Q:        {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  Gold:     {gold_answer[:100]}{'...' if len(gold_answer) > 100 else ''}")

        # Create temp directory for this question's DB
        tmp_dir = tempfile.mkdtemp(prefix=f"lme_{q_idx}_")
        try:
            # Step 1: Ingest
            print(f"  [INGEST]")
            t_start = time.time()
            mem = ingest_question(entry, embedder, args.llm_model, EXTRACTION_PASSES, tmp_dir)
            ingest_time = time.time() - t_start

            # Step 2: Query + Answer
            print(f"  [QUERY]")
            hypothesis, context_parts, latency_ms, rag_cost = query_and_answer(mem, question, embedder, client)
            print(f"    Latency: {latency_ms:.0f}ms, Retrieved: {len(context_parts)} memories")
            print(f"    Answer:  {hypothesis[:120]}{'...' if len(hypothesis) > 120 else ''}")

            # Step 3: Judge
            print(f"  [JUDGE]")
            score, reasoning, judge_cost = judge_answer(question, gold_answer, hypothesis, client)
            q_cost = rag_cost + judge_cost
            cumulative_cost += q_cost
            print(f"    Score:   {score}/100")
            print(f"    Reason:  {reasoning[:120]}{'...' if len(reasoning) > 120 else ''}")
            print(f"    Cost:    ${q_cost:.4f} (cumulative: ${cumulative_cost:.4f})")

            # Record result
            result = {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "gold_answer": gold_answer,
                "hypothesis": hypothesis,
                "score": score,
                "reasoning": reasoning,
                "latency_ms": latency_ms,
                "ingest_time_s": round(ingest_time, 1),
                "num_turns": num_turns,
                "is_abstention": is_abstention,
                "api_cost": round(q_cost, 6),
            }
            all_results.append(result)
            completed_ids.add(qid)
            category_scores[qtype].append(score)

            # Save after each question (crash-safe)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            # Running summary
            valid = [r for r in all_results if r["score"] >= 0]
            total_avg = sum(r["score"] for r in valid) / max(1, len(valid))
            print(f"\n  Running: {len(all_results)} done, avg score {total_avg:.1f}/100")
            for cat, scores in sorted(category_scores.items()):
                vs = [s for s in scores if s >= 0]
                if vs:
                    avg = sum(vs) / len(vs)
                    pass_rate = sum(1 for s in vs if s >= 80) / len(vs) * 100
                    print(f"    {cat:<30} n={len(vs):>3}  avg={avg:>5.1f}  pass(>=80)={pass_rate:>5.1f}%")

            # Stop on failure if requested
            if args.stop_on_fail and score < 50:
                print(f"\n  *** LOW SCORE ({score}/100) --- stopping for diagnosis ***")
                print(f"  Top retrieved context:")
                for i, ctx in enumerate(context_parts[:5]):
                    print(f"    [{i+1}] {ctx[:150]}{'...' if len(ctx) > 150 else ''}")
                break

            print()  # Blank line between questions

        finally:
            # Explicitly release Memory to free model resources before next question
            try:
                del mem
            except NameError:
                pass
            gc.collect()
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        # Max questions limit
        if args.max_questions and len(all_results) >= args.max_questions:
            print(f"\n  Reached --max-questions {args.max_questions}, stopping.")
            break

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY ({len(all_results)} questions)")
    print(f"{'=' * 70}")

    valid_results = [r for r in all_results if r["score"] >= 0]
    total_cost = sum(r.get("api_cost", 0.0) for r in all_results)
    if valid_results:
        overall_avg = sum(r["score"] for r in valid_results) / len(valid_results)
        pass_80 = sum(1 for r in valid_results if r["score"] >= 80) / len(valid_results) * 100
        pass_50 = sum(1 for r in valid_results if r["score"] >= 50) / len(valid_results) * 100

        print(f"  Overall avg score:     {overall_avg:.1f}/100")
        print(f"  Pass rate (>=80):      {pass_80:.1f}%")
        print(f"  Pass rate (>=50):      {pass_50:.1f}%")
        print(f"  Total API cost:        ${total_cost:.4f}")
        print(f"  RAG model:             {RAG_MODEL}")
        print(f"  Judge model:           {JUDGE_MODEL}")
        print()

        print(f"  {'Category':<30} {'Count':>5} {'Avg':>6} {'>=80':>6} {'>=50':>6}")
        print(f"  {'-' * 53}")
        for cat in sorted(category_scores.keys()):
            scores = [s for s in category_scores[cat] if s >= 0]
            if scores:
                avg = sum(scores) / len(scores)
                p80 = sum(1 for s in scores if s >= 80) / len(scores) * 100
                p50 = sum(1 for s in scores if s >= 50) / len(scores) * 100
                print(f"  {cat:<30} {len(scores):>5} {avg:>5.1f}% {p80:>5.1f}% {p50:>5.1f}%")

        # Abstention breakdown
        abs_results = [r for r in valid_results if r["is_abstention"]]
        non_abs = [r for r in valid_results if not r["is_abstention"]]
        if abs_results:
            abs_avg = sum(r["score"] for r in abs_results) / len(abs_results)
            print(f"\n  Abstention questions:   n={len(abs_results)}, avg={abs_avg:.1f}")
            non_avg = sum(r["score"] for r in non_abs) / len(non_abs) if non_abs else 0
            print(f"  Non-abstention:         n={len(non_abs)}, avg={non_avg:.1f}")

    print(f"\n  Results saved to: {results_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemEval Stepped Evaluation")
    parser.add_argument("--mode", choices=["oracle", "s"], default="oracle",
                        help="Dataset mode: oracle (evidence-only) or s (full haystack)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Path to LLM model for entity extraction (e.g., models/phi-4-mini/...gguf)")
    parser.add_argument("--start-at", type=int, default=0,
                        help="Start from question N (0-indexed)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Stop after N questions")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop when a question scores <50 for diagnosis")
    parser.add_argument("--extraction-passes", type=int, default=1,
                        help="Number of LLM extraction passes (default: 1)")
    args = parser.parse_args()

    EXTRACTION_PASSES = args.extraction_passes

    # Verify OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    run_evaluation(args)
