#!/usr/bin/env python3
"""
LongMemEval Evaluation for MnemeFusion

Evaluates MnemeFusion on LongMemEval (ICLR 2025) — 500 questions across
6 categories testing long-term conversational memory.

Modes:
  oracle  — Evidence-only sessions (~36 turns/q). Tests extraction + RAG quality.
  s       — Full haystack (~490 turns/q). Tests end-to-end retrieval pipeline.

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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed.")
    print("Install with: cd mnemefusion-python && maturin develop --release")
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
RAG_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5-mini"  # Used for --detailed-scoring mode (0-100 scale)
JUDGE_MODEL_STANDARD = "gpt-4o-2024-08-06"  # Official LongMemEval protocol
TOP_K = 20
EXTRACTION_PASSES = 1

# Pricing per 1M tokens (for cost tracking)
MODEL_PRICING = {
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-5-mini":        {"input": 0.25, "output": 2.00},
    "gpt-5-nano":        {"input": 0.05, "output": 0.40},
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

# Official LongMemEval category-specific judge prompts (binary yes/no)
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

# Detailed 0-100 scoring prompt (used with --detailed-scoring flag)
JUDGE_PROMPT_DETAILED = """You are evaluating a conversational memory system's ability to recall information.

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


def get_gold_turn_contents(entry: Dict) -> List[str]:
    """Extract content of turns marked with has_answer=True."""
    gold_contents = []
    for session in entry["haystack_sessions"]:
        for turn in session:
            if turn.get("has_answer"):
                gold_contents.append(turn["content"].strip())
    return gold_contents


def compute_recall(retrieved_contents: List[str], gold_contents: List[str], k: int) -> float:
    """Compute Recall@K: fraction of gold turns found in top-K retrieved memories."""
    if not gold_contents:
        return 1.0  # No gold turns = vacuously correct
    top_k = retrieved_contents[:k]
    found = 0
    for gold in gold_contents:
        for ret in top_k:
            # Substring match: gold turn content appears in retrieved memory
            if gold in ret or ret in gold:
                found += 1
                break
    return found / len(gold_contents)


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

    # Enable first-person pronoun → "user" entity resolution.
    # Queries with "I", "me", "my" will resolve to the "user" profile,
    # ensuring user's own memories get the entity score boost.
    mem.set_user_entity("user")

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
    question_date: str = "",
) -> Tuple[str, List[str], List[str], float, float]:
    """Query MnemeFusion and generate answer via RAG. Returns (answer, context, raw_contents, latency_ms, cost)."""
    query_embedding = embedder.encode([question], show_progress_bar=False)[0].tolist()

    t0 = time.time()
    intent_info, results, profile_context = mem.query(question, query_embedding, TOP_K)
    latency_ms = (time.time() - t0) * 1000

    # Format retrieved memories as context
    context_parts = []
    raw_contents = []  # Raw content for recall computation
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

    # Add profile context if available
    if profile_context:
        context_parts.append(f"[Profile summary] {profile_context}")

    context_str = "\n".join([f"- {c}" for c in context_parts])

    # Build date line for temporal context
    date_line = f"\nCurrent date: {question_date}" if question_date else ""

    # Generate answer
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

    return answer, context_parts, raw_contents, latency_ms, cost


def judge_answer(
    question: str,
    gold_answer: str,
    hypothesis: str,
    client: OpenAI,
    question_type: str = "",
    question_id: str = "",
    detailed_scoring: bool = False,
) -> Tuple[int, str, float]:
    """Judge correctness, returning score + reasoning + cost.

    If detailed_scoring=True: uses JUDGE_PROMPT_DETAILED with gpt-5-mini, returns 0-100 score.
    If detailed_scoring=False (default): uses official LongMemEval binary prompts with
    gpt-4o-2024-08-06, returns 1 (yes) or 0 (no).
    """
    if detailed_scoring:
        # Detailed 0-100 scoring mode (legacy behavior)
        prompt = JUDGE_PROMPT_DETAILED.format(
            question=question,
            gold_answer=gold_answer,
            hypothesis=hypothesis,
        )
        model = JUDGE_MODEL
        try:
            judge_kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            if not model.startswith("gpt-5"):
                judge_kwargs["max_tokens"] = 200
                judge_kwargs["temperature"] = 0
            response = client.chat.completions.create(**judge_kwargs)
            raw = response.choices[0].message.content.strip()
            cost = 0.0
            if response.usage:
                cost = estimate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)

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
    else:
        # Official LongMemEval binary yes/no scoring
        # Select the appropriate prompt
        if question_id.endswith("_abs"):
            prompt_key = "abstention"
        elif question_type in ("temporal-reasoning", "knowledge-update", "single-session-preference"):
            prompt_key = question_type
        else:
            prompt_key = "general"

        prompt_template = JUDGE_PROMPTS[prompt_key]

        # Abstention prompt doesn't include gold_answer
        if prompt_key == "abstention":
            prompt = prompt_template.format(question=question, hypothesis=hypothesis)
        else:
            prompt = prompt_template.format(
                question=question,
                gold_answer=gold_answer,
                hypothesis=hypothesis,
            )

        model = JUDGE_MODEL_STANDARD
        try:
            judge_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 10,
            }
            response = client.chat.completions.create(**judge_kwargs)
            raw = response.choices[0].message.content.strip()
            cost = 0.0
            if response.usage:
                cost = estimate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)

            label = "yes" in raw.lower()
            score = 1 if label else 0
            return score, raw, cost

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

    # Determine scoring mode
    detailed_scoring = args.detailed_scoring
    scoring_mode = "detailed" if detailed_scoring else "binary"
    judge_model = JUDGE_MODEL if detailed_scoring else JUDGE_MODEL_STANDARD

    # Initialize embedder (loaded once, reused across all questions)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    # Initialize OpenAI client
    client = OpenAI()

    # Results file for persistence (separate files for binary vs detailed)
    scoring_suffix = "detailed" if detailed_scoring else "binary"
    worker_suffix = f"_w{args.worker}" if args.num_workers > 1 else ""
    results_path = FIXTURES_DIR / f"longmemeval_results_{args.mode}_{scoring_suffix}{worker_suffix}.json"
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

    # Print methodology header
    print(f"\n{'=' * 70}")
    print(f"LongMemEval Stepped Evaluation")
    print(f"  Mode:       {args.mode} ({'evidence-only' if args.mode == 'oracle' else 'full haystack'})")
    if detailed_scoring:
        print(f"  Scoring:    Continuous 0-100 (internal)")
    else:
        print(f"  Scoring:    Binary yes/no (official protocol)")
    print(f"  LLM:        {args.llm_model or 'DISABLED (no entity extraction)'}")
    print(f"  RAG model:  {RAG_MODEL}")
    judge_pricing = MODEL_PRICING.get(judge_model, {})
    print(f"  Judge:      {judge_model} (${judge_pricing.get('input', '?')}/{judge_pricing.get('output', '?')} per 1M tok)")
    print(f"  Questions:  {len(data)} total, starting at #{args.start_at}")
    print(f"  Results:    {results_path.name}")
    print(f"{'=' * 70}\n")

    completed_ids = {r["question_id"] for r in all_results}
    cumulative_cost = sum(r.get("api_cost", 0.0) for r in all_results)

    new_count = 0  # Track NEW questions processed (not previously loaded results)
    for q_idx in range(args.start_at, len(data)):
        # Worker partitioning: each worker handles every Nth question
        if args.num_workers > 1 and (q_idx % args.num_workers) != args.worker:
            continue

        entry = data[q_idx]
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        gold_answer = str(entry["answer"])
        is_abstention = qid.endswith("_abs")
        question_date = entry.get("question_date", "")

        # Skip already-evaluated questions
        if qid in completed_ids:
            continue

        # Category filter
        if args.category and qtype != args.category:
            continue

        num_sessions = len(entry["haystack_sessions"])
        num_turns = sum(len(s) for s in entry["haystack_sessions"])

        print(f"--- Question {q_idx + 1}/{len(data)} ---")
        print(f"  ID:       {qid}")
        print(f"  Type:     {qtype}{'  [ABSTENTION]' if is_abstention else ''}")
        print(f"  Haystack: {num_sessions} sessions, {num_turns} turns")
        print(f"  Q:        {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"  Gold:     {gold_answer[:100]}{'...' if len(gold_answer) > 100 else ''}")
        if question_date:
            print(f"  Date:     {question_date}")

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
            hypothesis, context_parts, raw_contents, latency_ms, rag_cost = query_and_answer(
                mem, question, embedder, client, question_date=question_date
            )
            print(f"    Latency: {latency_ms:.0f}ms, Retrieved: {len(context_parts)} memories")
            print(f"    Answer:  {hypothesis[:120]}{'...' if len(hypothesis) > 120 else ''}")

            # Step 2.5: Recall@K
            gold_contents = get_gold_turn_contents(entry)
            r5 = compute_recall(raw_contents, gold_contents, 5)
            r10 = compute_recall(raw_contents, gold_contents, 10)
            r20 = compute_recall(raw_contents, gold_contents, 20)
            print(f"    Recall:  R@5={r5:.0%}  R@10={r10:.0%}  R@20={r20:.0%}  ({len(gold_contents)} gold turns)")

            # Step 3: Judge
            print(f"  [JUDGE]")
            score, reasoning, judge_cost = judge_answer(
                question, gold_answer, hypothesis, client,
                question_type=qtype, question_id=qid,
                detailed_scoring=detailed_scoring,
            )
            q_cost = rag_cost + judge_cost
            cumulative_cost += q_cost
            if detailed_scoring:
                print(f"    Score:   {score}/100")
            else:
                print(f"    Score:   {'YES (1)' if score == 1 else 'NO (0)' if score == 0 else f'ERROR ({score})'}")
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
                "scoring_mode": scoring_mode,
                "latency_ms": latency_ms,
                "ingest_time_s": round(ingest_time, 1),
                "num_turns": num_turns,
                "is_abstention": is_abstention,
                "api_cost": round(q_cost, 6),
                "recall_at_5": round(r5, 4),
                "recall_at_10": round(r10, 4),
                "recall_at_20": round(r20, 4),
                "num_gold_turns": len(gold_contents),
            }
            if question_date:
                result["question_date"] = question_date
            all_results.append(result)
            completed_ids.add(qid)
            category_scores[qtype].append(score)
            new_count += 1

            # Save after each question (crash-safe)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            # Running summary
            valid = [r for r in all_results if r["score"] >= 0]
            avg_r5 = sum(r.get("recall_at_5", 0) for r in valid) / max(1, len(valid))
            avg_r10 = sum(r.get("recall_at_10", 0) for r in valid) / max(1, len(valid))
            avg_r20 = sum(r.get("recall_at_20", 0) for r in valid) / max(1, len(valid))

            if detailed_scoring:
                total_avg = sum(r["score"] for r in valid) / max(1, len(valid))
                print(f"\n  Running: {len(all_results)} done, avg score {total_avg:.1f}/100, R@5={avg_r5:.0%} R@10={avg_r10:.0%} R@20={avg_r20:.0%}")
                for cat, scores in sorted(category_scores.items()):
                    vs = [s for s in scores if s >= 0]
                    if vs:
                        avg = sum(vs) / len(vs)
                        pass_rate = sum(1 for s in vs if s >= 80) / len(vs) * 100
                        print(f"    {cat:<30} n={len(vs):>3}  avg={avg:>5.1f}  pass(>=80)={pass_rate:>5.1f}%")
            else:
                overall_acc = sum(r["score"] for r in valid) / max(1, len(valid)) * 100
                print(f"\n  Running: {len(all_results)} done, accuracy {overall_acc:.1f}%, R@5={avg_r5:.0%} R@10={avg_r10:.0%} R@20={avg_r20:.0%}")
                for cat, scores in sorted(category_scores.items()):
                    vs = [s for s in scores if s >= 0]
                    if vs:
                        acc = sum(vs) / len(vs) * 100
                        print(f"    {cat:<30} n={len(vs):>3}  acc={acc:>5.1f}%")

            # Stop on failure if requested
            fail_threshold = 50 if detailed_scoring else 1
            if args.stop_on_fail and score < fail_threshold:
                label = f"{score}/100" if detailed_scoring else f"{'NO' if score == 0 else score}"
                print(f"\n  *** LOW SCORE ({label}) --- stopping for diagnosis ***")
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
        if args.max_questions and new_count >= args.max_questions:
            print(f"\n  Reached --max-questions {args.max_questions}, stopping.")
            break

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY ({len(all_results)} questions, {scoring_mode} scoring)")
    print(f"{'=' * 70}")

    valid_results = [r for r in all_results if r["score"] >= 0]
    total_cost = sum(r.get("api_cost", 0.0) for r in all_results)

    if valid_results:
        avg_r5 = sum(r.get("recall_at_5", 0) for r in valid_results) / max(1, len(valid_results))
        avg_r10 = sum(r.get("recall_at_10", 0) for r in valid_results) / max(1, len(valid_results))
        avg_r20 = sum(r.get("recall_at_20", 0) for r in valid_results) / max(1, len(valid_results))

        if detailed_scoring:
            # Detailed 0-100 mode: show average scores and pass rates
            overall_avg = sum(r["score"] for r in valid_results) / len(valid_results)
            pass_80 = sum(1 for r in valid_results if r["score"] >= 80) / len(valid_results) * 100
            pass_50 = sum(1 for r in valid_results if r["score"] >= 50) / len(valid_results) * 100

            print(f"  Overall avg score:     {overall_avg:.1f}/100")
            print(f"  Recall:                R@5={avg_r5:.1%}  R@10={avg_r10:.1%}  R@20={avg_r20:.1%}")
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

        else:
            # Binary yes/no mode: show accuracy percentages
            overall_acc = sum(r["score"] for r in valid_results) / len(valid_results) * 100

            # Task-averaged accuracy: mean of per-category accuracies (each category weighted equally)
            cat_accuracies = {}
            for cat in sorted(category_scores.keys()):
                scores = [s for s in category_scores[cat] if s >= 0]
                if scores:
                    cat_accuracies[cat] = sum(scores) / len(scores) * 100
            task_avg_acc = sum(cat_accuracies.values()) / max(1, len(cat_accuracies))

            # Abstention accuracy
            abs_results = [r for r in valid_results if r["is_abstention"]]
            non_abs = [r for r in valid_results if not r["is_abstention"]]

            print(f"  Overall accuracy:      {overall_acc:.1f}%  (mean across all {len(valid_results)} labels)")
            print(f"  Task-averaged acc:     {task_avg_acc:.1f}%  (mean of {len(cat_accuracies)} category accuracies)")
            print(f"  Recall:                R@5={avg_r5:.1%}  R@10={avg_r10:.1%}  R@20={avg_r20:.1%}")
            print(f"  Total API cost:        ${total_cost:.4f}")
            print(f"  RAG model:             {RAG_MODEL}")
            print(f"  Judge model:           {judge_model}")
            print()

            print(f"  {'Category':<30} {'Count':>5} {'Acc':>7}")
            print(f"  {'-' * 44}")
            for cat in sorted(cat_accuracies.keys()):
                scores = [s for s in category_scores[cat] if s >= 0]
                print(f"  {cat:<30} {len(scores):>5} {cat_accuracies[cat]:>6.1f}%")

            # Abstention breakdown
            if abs_results:
                abs_acc = sum(r["score"] for r in abs_results) / len(abs_results) * 100
                non_abs_acc = sum(r["score"] for r in non_abs) / len(non_abs) * 100 if non_abs else 0
                print(f"\n  Abstention accuracy:   {abs_acc:.1f}%  (n={len(abs_results)})")
                print(f"  Non-abstention acc:    {non_abs_acc:.1f}%  (n={len(non_abs)})")

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
    parser.add_argument("--category", type=str, default=None,
                        help="Only process questions of this type (e.g., knowledge-update)")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop when a question scores <50 for diagnosis")
    parser.add_argument("--extraction-passes", type=int, default=1,
                        help="Number of LLM extraction passes (default: 1)")
    parser.add_argument("--detailed-scoring", action="store_true",
                        help="Use 0-100 continuous scoring with gpt-5-mini instead of official binary yes/no")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers (each loads its own LLM instance)")
    parser.add_argument("--worker", type=int, default=0,
                        help="This worker's index (0-based, must be < num-workers)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge per-worker result files into a single results file, then print summary")
    args = parser.parse_args()

    EXTRACTION_PASSES = args.extraction_passes

    # Merge mode: combine worker result files and print final summary
    if args.merge:
        scoring_suffix = "detailed" if args.detailed_scoring else "binary"
        import glob as glob_mod
        pattern = str(FIXTURES_DIR / f"longmemeval_results_{args.mode}_{scoring_suffix}_w*.json")
        worker_files = sorted(glob_mod.glob(pattern))
        if not worker_files:
            print(f"ERROR: No worker files found matching {pattern}")
            sys.exit(1)

        all_results = []
        seen_ids = set()
        for wf in worker_files:
            with open(wf, encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if r["question_id"] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r["question_id"])
            print(f"  Loaded {len(data)} results from {Path(wf).name}")

        # Save merged file
        merged_path = FIXTURES_DIR / f"longmemeval_results_{args.mode}_{scoring_suffix}.json"
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Merged {len(all_results)} results -> {merged_path.name}")

        # Print summary
        valid = [r for r in all_results if r["score"] >= 0]
        if valid:
            category_scores = defaultdict(list)
            for r in valid:
                category_scores[r["question_type"]].append(r["score"])

            overall_acc = sum(r["score"] for r in valid) / len(valid) * 100
            cat_accs = {}
            for cat, scores in category_scores.items():
                cat_accs[cat] = sum(scores) / len(scores) * 100
            task_avg = sum(cat_accs.values()) / max(1, len(cat_accs))

            avg_r5 = sum(r.get("recall_at_5", 0) for r in valid) / len(valid)
            avg_r10 = sum(r.get("recall_at_10", 0) for r in valid) / len(valid)
            avg_r20 = sum(r.get("recall_at_20", 0) for r in valid) / len(valid)
            total_cost = sum(r.get("api_cost", 0) for r in all_results)

            print(f"\n{'=' * 70}")
            print(f"MERGED SUMMARY ({len(all_results)} questions, {scoring_suffix} scoring)")
            print(f"{'=' * 70}")
            print(f"  Overall accuracy:      {overall_acc:.1f}%")
            print(f"  Task-averaged acc:     {task_avg:.1f}%")
            print(f"  Recall:                R@5={avg_r5:.1%}  R@10={avg_r10:.1%}  R@20={avg_r20:.1%}")
            print(f"  Total API cost:        ${total_cost:.4f}")
            for cat in sorted(cat_accs.keys()):
                n = len(category_scores[cat])
                print(f"  {cat:<30} n={n:>3}  acc={cat_accs[cat]:>5.1f}%")

            abs_results = [r for r in valid if r.get("is_abstention")]
            non_abs = [r for r in valid if not r.get("is_abstention")]
            if abs_results:
                abs_acc = sum(r["score"] for r in abs_results) / len(abs_results) * 100
                non_abs_acc = sum(r["score"] for r in non_abs) / len(non_abs) * 100 if non_abs else 0
                print(f"\n  Abstention accuracy:   {abs_acc:.1f}%  (n={len(abs_results)})")
                print(f"  Non-abstention acc:    {non_abs_acc:.1f}%  (n={len(non_abs)})")

        sys.exit(0)

    # Validate worker args
    if args.worker >= args.num_workers:
        print(f"ERROR: --worker {args.worker} must be < --num-workers {args.num_workers}")
        sys.exit(1)

    # Verify OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    if args.num_workers > 1:
        print(f"  Worker {args.worker}/{args.num_workers} — processing questions {args.worker}, {args.worker + args.num_workers}, {args.worker + 2*args.num_workers}, ...")

    run_evaluation(args)
