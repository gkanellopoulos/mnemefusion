#!/usr/bin/env python3
"""
Industry-Standard Evaluation for MnemeFusion

This script evaluates MnemeFusion using the same methodology as Mem0 and other
industry memory systems, enabling direct comparison.

Metrics (matching Mem0's evaluation):
- LLM-as-Judge: Binary correctness score from GPT-4o-mini
- F1 Score: Token-level precision/recall
- BLEU-1: Unigram overlap with ground truth
- Token Consumption: Tokens used for answer generation
- Latency: Query time (p50, p95, p99)

Dataset: LoCoMo (Long-term Conversation Memory)
- 10 conversations, ~2000 questions total
- Categories: Single-hop, Multi-hop, Temporal, Open-domain

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY=sk-...

    # Run full evaluation
    python industry_eval.py

    # Run on subset for testing
    python industry_eval.py --max-questions 50

    # Run specific categories only
    python industry_eval.py --categories 1 2 3

Reference:
    Mem0 evaluation: https://github.com/mem0ai/mem0/tree/main/evaluation
    LoCoMo benchmark: https://github.com/snap-research/locomo
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import contextlib
import tempfile

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

# CUDA Support: Pre-load CUDA DLLs if available (enables GPU acceleration)
try:
    from mnemefusion_cuda_wrapper import mnemefusion
    print("[CUDA] GPU acceleration enabled via wrapper")
except ImportError:
    # Fallback to standard import if wrapper not available
    try:
        import mnemefusion
        print("[INFO] Using standard CPU-only import (CUDA wrapper not available)")
    except ImportError:
        print("ERROR: mnemefusion not installed.")
        print("Install with: cd mnemefusion-python && pip install -e .")
        sys.exit(1)

# Check dependencies
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install sentence-transformers numpy")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QuestionResult:
    """Result for a single question evaluation"""
    question_id: str
    question: str
    ground_truth: str
    generated_answer: str
    category: int

    # Metrics
    llm_judge_score: int  # 0 or 1
    f1_score: float
    bleu_score: float

    # Performance
    retrieval_latency_ms: float
    generation_latency_ms: float
    tokens_used: int
    memories_retrieved: int

    # Recall@K metrics (Solution 6)
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    retrieved_dialog_ids: List[str] = field(default_factory=list)

    # Debug info
    retrieved_content: List[str] = field(default_factory=list)


@dataclass
class EvaluationResults:
    """Aggregated evaluation results"""
    total_questions: int = 0

    # Overall metrics
    llm_judge_accuracy: float = 0.0
    avg_f1_score: float = 0.0
    avg_bleu_score: float = 0.0

    # Per-category metrics
    category_results: Dict[int, Dict] = field(default_factory=dict)

    # Performance
    avg_retrieval_latency_ms: float = 0.0
    p50_retrieval_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0
    p99_retrieval_latency_ms: float = 0.0
    avg_tokens_per_question: float = 0.0
    total_tokens_used: int = 0

    # Recall@K metrics (Solution 6)
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0

    # Timing
    total_ingestion_time_s: float = 0.0
    total_evaluation_time_s: float = 0.0

    # Dataset info
    num_documents: int = 0
    num_conversations: int = 0


# =============================================================================
# Metrics Implementation
# =============================================================================

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score between prediction and ground truth.

    This matches the standard implementation used by Mem0 and SQuAD evaluations.
    """
    def normalize_text(text) -> List[str]:
        """Lowercase, remove punctuation, split into tokens"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    common_tokens = set(pred_tokens) & set(truth_tokens)

    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_bleu_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate BLEU-1 (unigram) score.

    Simplified BLEU-1 implementation matching industry standard.
    """
    def normalize_text(text) -> List[str]:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    # Count matching unigrams
    truth_counts = defaultdict(int)
    for token in truth_tokens:
        truth_counts[token] += 1

    matches = 0
    for token in pred_tokens:
        if truth_counts[token] > 0:
            matches += 1
            truth_counts[token] -= 1

    # BLEU-1 precision
    precision = matches / len(pred_tokens) if pred_tokens else 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(truth_tokens) / len(pred_tokens))) if pred_tokens else 0.0

    return bp * precision


# =============================================================================
# LLM Integration
# =============================================================================

class LLMClient:
    """OpenAI client for answer generation and judging"""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_tokens = 0

    def generate_answer(self, question: str, context: List[str], max_tokens: int = 150) -> Tuple[str, int]:
        """
        Generate an answer to the question using retrieved context.

        Returns:
            Tuple of (answer, tokens_used)
        """
        # Format context
        context_str = "\n".join([f"- {c}" for c in context[:25]])  # 20 memories + up to 5 profile facts

        prompt = f"""You are a helpful assistant answering questions based on conversation history.

Retrieved memories (dates in brackets show when each conversation occurred):
{context_str}

Question: {question}

Answer the question based on the information in the retrieved memories. Look carefully through ALL the memories for relevant details — the answer may be mentioned briefly or indirectly.
For temporal questions (when did X happen), use the dates in brackets to calculate the answer.
For example, if a memory from [8 May 2023] mentions "yesterday", the event happened on 7 May 2023.
If you find ANY relevant information, provide an answer. Only say "I don't have enough information" if the memories truly contain nothing related to the question.
Keep your answer concise and factual."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )

            answer = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            self.total_tokens += tokens

            return answer, tokens

        except Exception as e:
            print(f"  [ERROR] Answer generation failed: {e}")
            return "Error generating answer", 0

    def judge_answer(self, question: str, ground_truth: str, prediction: str) -> int:
        """
        Use LLM to judge if the prediction is correct.

        Returns:
            1 if correct, 0 if incorrect
        """
        # Check for obvious "don't know" answers first (fast path)
        prediction_lower = prediction.lower()
        dont_know_phrases = [
            "i don't have",
            "i do not have",
            "cannot find",
            "no information",
            "unable to",
            "don't have enough",
            "not mentioned",
            "not specified",
            "no specific",
            "cannot determine",
            "not available"
        ]
        for phrase in dont_know_phrases:
            if phrase in prediction_lower:
                return 0

        prompt = f"""Task: Determine if the AI's answer contains the correct information.

Question: {question}
Correct Answer: {ground_truth}
AI's Answer: {prediction}

Does the AI's answer contain the key information from the correct answer?
Answer only "YES" or "NO"."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )

            verdict = response.choices[0].message.content.strip().upper()
            self.total_tokens += response.usage.total_tokens

            return 1 if verdict.startswith("YES") else 0

        except Exception as e:
            print(f"  [ERROR] Judge failed: {e}")
            return 0


# =============================================================================
# MnemeFusion Integration
# =============================================================================

class MnemeFusionEvaluator:
    """Evaluator for MnemeFusion memory system"""

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5", use_gpu: bool = True):
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.embedder.to('cuda')
                    print(f"  [OK] Using GPU: {torch.cuda.get_device_name(0)}")
            except:
                print("  [INFO] GPU not available, using CPU")

        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.embedding_dim}")

        self.memory = None
        self.db_path = None

    def create_memory_store(self, db_path: str, use_slm: bool = False, slm_model_path: str = None,
                            use_llm: bool = False, llm_model_path: str = None, llm_tier: str = "balanced"):
        """Create a new MnemeFusion memory store"""
        self.db_path = db_path

        config = {
            "embedding_dim": self.embedding_dim,
            "entity_extraction_enabled": True,
        }

        if use_slm and slm_model_path:
            config["use_slm"] = True
            config["slm_model_path"] = slm_model_path
            config["slm_metadata_extraction_enabled"] = True
            print(f"  [SLM] Enabled with model: {slm_model_path}")

        print(f"  [DEBUG] Config: {config}")
        self.memory = mnemefusion.Memory(db_path, config)

        # Enable native LLM entity extraction if requested
        if use_llm and llm_model_path:
            try:
                print(f"  [LLM] Loading Qwen3 model: {llm_model_path}")
                self.memory.enable_llm_entity_extraction(llm_model_path, llm_tier)
                print(f"  [LLM] Enabled {llm_tier} tier extraction")
            except Exception as e:
                print(f"  [LLM] Warning: Failed to enable LLM extraction: {e}")
                print(f"  [LLM] Make sure the wheel was built with --features entity-extraction")

        print(f"  [OK] Created memory store at {db_path}")

    def ingest_documents(self, documents: List[Tuple[str, str, Dict]], use_llm: bool = False) -> float:
        """
        Ingest documents into memory store.

        Args:
            documents: List of (doc_id, content, metadata) tuples
            use_llm: If True, use individual add() for LLM extraction (slower but enables extraction)

        Returns:
            Total ingestion time in seconds
        """
        print(f"\nIngesting {len(documents)} documents...")
        if use_llm:
            print("  [LLM] Using individual add() for LLM extraction (slower)")
        start_time = time.time()

        # Reserve capacity for vector index
        print(f"  Reserving capacity for {len(documents)} vectors...")
        self.memory.reserve_capacity(len(documents))

        # Prepare batch
        contents = [doc[1] for doc in documents]
        metadatas = [doc[2] for doc in documents]

        # Generate embeddings in batches
        print("  Generating embeddings...")
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            embeddings = self.embedder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())

            if (i + batch_size) % 500 == 0:
                print(f"    Embedded {min(i + batch_size, len(contents))}/{len(contents)}")

        # Ingest into MnemeFusion
        print("  Adding to memory store...")

        if use_llm:
            # Use individual add() to enable LLM extraction
            # This is much slower but extracts entity profiles
            created_count = 0
            error_count = 0
            for i, (doc_id, content, metadata) in enumerate(documents):
                try:
                    self.memory.add(content, all_embeddings[i], metadata)
                    created_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"    [WARN] Failed to add doc {i}: {e}")

                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed_so_far = time.time() - start_time
                    rate = (i + 1) / elapsed_so_far
                    remaining = (len(documents) - i - 1) / rate if rate > 0 else 0
                    print(f"    Progress: {i + 1}/{len(documents)} ({rate:.1f} docs/s, ~{remaining:.0f}s remaining)")

            elapsed = time.time() - start_time
            print(f"  [OK] Ingested {created_count} documents in {elapsed:.1f}s")
            if error_count > 0:
                print(f"  [WARN] {error_count} errors during ingestion")
        else:
            # Use batch add for fast ingestion (no LLM extraction)
            memories_to_add = []
            for i, (doc_id, content, metadata) in enumerate(documents):
                memories_to_add.append({
                    "content": content,
                    "embedding": all_embeddings[i],
                    "metadata": metadata
                })

            result = self.memory.add_batch(memories_to_add)
            elapsed = time.time() - start_time
            print(f"  [OK] Ingested {result['created_count']} documents in {elapsed:.1f}s")

            if result.get('errors'):
                print(f"  [WARN] {len(result['errors'])} errors during ingestion")

        return elapsed

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[str], float]:
        """
        Retrieve relevant memories for a query using multi-dimensional fusion.

        Uses MnemeFusion's query() method which combines:
        - Semantic similarity (vector search)
        - BM25 keyword matching
        - Entity relevance
        - Temporal relevance
        - Causal relevance

        Returns:
            Tuple of (list of formatted context strings with dates, latency in ms)
        """
        contents, dialog_ids, latency_ms = self.retrieve_with_ids(query, top_k)
        return contents, latency_ms

    def retrieve_with_ids(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str], float]:
        """
        Retrieve relevant memories with their dialog IDs for Recall@K measurement.

        Returns:
            Tuple of (formatted contents, dialog_ids, latency_ms)
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query], show_progress_bar=False)[0].tolist()

        # Use multi-dimensional query (not just vector search)
        start = time.time()
        intent_info, results = self.memory.query(query, query_embedding, top_k)
        latency_ms = (time.time() - start) * 1000

        # Extract content WITH metadata for temporal reasoning
        contents = []
        dialog_ids = []
        for result_dict, scores_dict in results:
            content = result_dict.get("content", "")
            metadata = result_dict.get("metadata", {})
            session_date = metadata.get("session_date", "")
            speaker = metadata.get("speaker", "")
            dialog_id = metadata.get("dialog_id", "")

            # Format with date context for temporal reasoning
            if session_date:
                formatted = f"[{session_date}] {speaker}: {content}" if speaker else f"[{session_date}] {content}"
            else:
                formatted = f"{speaker}: {content}" if speaker else content
            contents.append(formatted)
            dialog_ids.append(dialog_id)

        return contents, dialog_ids, latency_ms



# =============================================================================
# Dataset Loading
# =============================================================================

def load_locomo_dataset(dataset_path: str, num_conversations: int = None) -> Tuple[List[Dict], List[Tuple]]:
    """
    Load LoCoMo dataset.

    Returns:
        Tuple of (conversations, questions)
        - conversations: Raw conversation data
        - questions: List of (question, answer, category, conv_id, evidence_ids)
    """
    print(f"\nLoading LoCoMo dataset from {dataset_path}...")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("\nTo download:")
        print("  1. git clone https://github.com/snap-research/locomo.git")
        print("  2. cp locomo/data/locomo10.json tests/benchmarks/fixtures/")
        sys.exit(1)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = data if isinstance(data, list) else [data]

    if num_conversations:
        conversations = conversations[:num_conversations]

    # Extract questions
    questions = []
    for conv in conversations:
        conv_id = conv.get('sample_id', 'unknown')
        qa_data = conv.get('qa', [])

        for qa in qa_data:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            category = qa.get('category', 0)
            evidence = qa.get('evidence', [])

            if question and answer:
                questions.append((question, answer, category, conv_id, evidence))

    print(f"  [OK] Loaded {len(conversations)} conversations, {len(questions)} questions")

    # Category breakdown
    cat_counts = defaultdict(int)
    for _, _, cat, _, _ in questions:
        cat_counts[cat] += 1
    print(f"  Categories: {dict(cat_counts)}")

    return conversations, questions


def prepare_documents(conversations: List[Dict]) -> List[Tuple[str, str, Dict]]:
    """
    Prepare conversation turns as documents for ingestion.

    Returns:
        List of (doc_id, content, metadata) tuples
    """
    documents = []

    for conv in conversations:
        conv_id = conv.get('sample_id', 'unknown')
        conv_data = conv.get('conversation', {})

        session_idx = 1
        while f'session_{session_idx}' in conv_data:
            session_key = f'session_{session_idx}'
            session_date = conv_data.get(f'session_{session_idx}_date_time', f'session_{session_idx}')
            turns = conv_data.get(session_key, [])

            for turn_idx, turn in enumerate(turns):
                speaker = turn.get('speaker', 'unknown')
                text = turn.get('text', '')
                dialog_id = turn.get('dia_id', f'{conv_id}_s{session_idx}_t{turn_idx}')

                if not text:
                    continue

                metadata = {
                    'conversation_id': conv_id,
                    'session_idx': str(session_idx),
                    'session_date': session_date,
                    'turn_idx': str(turn_idx),
                    'speaker': speaker,
                    'dialog_id': dialog_id
                }

                documents.append((dialog_id, text, metadata))

            session_idx += 1

    return documents


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation(
    dataset_path: str,
    num_conversations: int = None,
    max_questions: int = None,
    categories: List[int] = None,
    top_k: int = 25,
    use_slm: bool = False,
    slm_model_path: str = None,
    use_llm: bool = False,
    llm_model_path: str = None,
    llm_tier: str = "balanced",
    output_path: str = None,
    verbose: bool = False,
    db_path: str = None,
    skip_ingestion: bool = False
) -> EvaluationResults:
    """
    Run full industry-standard evaluation.

    Args:
        dataset_path: Path to locomo10.json
        num_conversations: Limit number of conversations (None = all)
        max_questions: Limit number of questions (None = all)
        categories: List of categories to evaluate (None = all, 1-4 are standard)
        top_k: Number of memories to retrieve
        use_slm: Whether to use SLM for metadata extraction (Python subprocess)
        slm_model_path: Path to SLM model (required if use_slm=True)
        use_llm: Whether to use native LLM extraction (Qwen3 via llama-cpp-2)
        llm_model_path: Path to GGUF model file (required if use_llm=True)
        llm_tier: Model tier - "balanced" (4B) or "quality" (8B)
        output_path: Path to save detailed results
        verbose: Print detailed progress

    Returns:
        EvaluationResults with all metrics
    """
    print("=" * 70)
    print("MnemeFusion Industry-Standard Evaluation")
    print("=" * 70)
    print(f"Methodology: Mem0-compatible (LLM-Judge, F1, BLEU)")
    print(f"Dataset: LoCoMo")
    print(f"Judge LLM: GPT-4o-mini")
    if use_llm:
        print(f"Entity Extraction: Qwen3 Native LLM ({llm_tier} tier)")
    elif use_slm:
        print(f"Entity Extraction: Python SLM (0.6B)")
    else:
        print(f"Entity Extraction: Disabled (baseline)")
    print("=" * 70)

    # Load dataset
    conversations, all_questions = load_locomo_dataset(dataset_path, num_conversations)

    # Filter questions by category if specified
    # Categories 1-4 are the standard evaluation categories
    if categories:
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in categories]
    else:
        # Default: use categories 1-4 (standard LoCoMo evaluation)
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in [1, 2, 3, 4]]

    if max_questions:
        questions = questions[:max_questions]

    print(f"\nEvaluating {len(questions)} questions")

    # Prepare documents
    documents = prepare_documents(conversations)
    print(f"Prepared {len(documents)} documents")

    # Initialize components
    evaluator = MnemeFusionEvaluator()
    llm = LLMClient(model="gpt-4o-mini")

    # Database: --db-path for persistent storage, else temporary (auto-cleanup)
    _temp_ctx = tempfile.TemporaryDirectory() if not db_path else contextlib.nullcontext()
    with _temp_ctx as _temp_dir:
        if not db_path:
            db_path = os.path.join(_temp_dir, "eval.mfdb")

        # Skip ingestion when reusing an existing persistent DB
        _ingest_now = not (skip_ingestion and os.path.exists(db_path))

        evaluator.create_memory_store(
            db_path,
            use_slm=use_slm and _ingest_now,
            slm_model_path=slm_model_path,
            use_llm=use_llm and _ingest_now,
            llm_model_path=llm_model_path,
            llm_tier=llm_tier
        )

        if _ingest_now:
            # Ingest documents (use individual add if LLM extraction is enabled)
            ingestion_time = evaluator.ingest_documents(documents, use_llm=use_llm)
        else:
            print(f"  [SKIP] Using existing DB at {db_path} — skipping ingestion")
            ingestion_time = 0.0

        # Evaluate each question
        print(f"\nEvaluating {len(questions)} questions...")
        print("-" * 70)

        results = []
        latencies = []
        eval_start = time.time()

        for i, (question, ground_truth, category, conv_id, evidence) in enumerate(questions):
            if verbose or (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(questions)}] Category {category}: {question[:50]}...")

            # Retrieve with dialog IDs for Recall@K (Solution 6)
            # Use top_k=20 for recall measurement, but only pass top_k to answer generation
            recall_k = max(top_k, 20)
            retrieved_content, retrieved_ids, retrieval_latency = evaluator.retrieve_with_ids(question, top_k=recall_k)
            latencies.append(retrieval_latency)

            # Calculate Recall@K (Solution 6)
            # Check if evidence dialog IDs appear in retrieved results at various K
            evidence_set = set(evidence) if evidence else set()
            r_at_5 = 0.0
            r_at_10 = 0.0
            r_at_20 = 0.0
            if evidence_set:
                # Filter out synthetic profile fact IDs (empty dialog_id) for recall
                real_ids = [did for did in retrieved_ids if did]
                found_at_5 = len(evidence_set & set(real_ids[:5]))
                found_at_10 = len(evidence_set & set(real_ids[:10]))
                found_at_20 = len(evidence_set & set(real_ids[:20]))
                r_at_5 = found_at_5 / len(evidence_set)
                r_at_10 = found_at_10 / len(evidence_set)
                r_at_20 = found_at_20 / len(evidence_set)

            # Rust core now prepends synthetic profile facts automatically
            context_with_facts = retrieved_content[:top_k]

            # Generate answer
            gen_start = time.time()
            answer, tokens = llm.generate_answer(question, context_with_facts)
            gen_latency = (time.time() - gen_start) * 1000

            # Judge
            judge_score = llm.judge_answer(question, ground_truth, answer)

            # Calculate metrics
            f1 = calculate_f1_score(answer, ground_truth)
            bleu = calculate_bleu_score(answer, ground_truth)

            # Store result
            result = QuestionResult(
                question_id=f"{conv_id}_{i}",
                question=question,
                ground_truth=ground_truth,
                generated_answer=answer,
                category=category,
                llm_judge_score=judge_score,
                f1_score=f1,
                bleu_score=bleu,
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=gen_latency,
                tokens_used=tokens,
                memories_retrieved=len(retrieved_content[:top_k]),
                recall_at_5=r_at_5,
                recall_at_10=r_at_10,
                recall_at_20=r_at_20,
                evidence_ids=evidence if evidence else [],
                retrieved_dialog_ids=retrieved_ids[:20],
                retrieved_content=retrieved_content[:3] if verbose else []
            )
            results.append(result)

            if verbose:
                recall_str = f"R@10: {r_at_10:.0%}" if evidence_set else "R@10: N/A"
                print(f"    Judge: {'Y' if judge_score else 'N'} | F1: {f1:.2f} | BLEU: {bleu:.2f} | {recall_str}")

        eval_time = time.time() - eval_start
        print("-" * 70)

        # Aggregate results
        final_results = EvaluationResults(
            total_questions=len(results),
            num_documents=len(documents),
            num_conversations=len(conversations),
            total_ingestion_time_s=ingestion_time,
            total_evaluation_time_s=eval_time,
            total_tokens_used=llm.total_tokens
        )

        # Overall metrics
        if results:
            final_results.llm_judge_accuracy = sum(r.llm_judge_score for r in results) / len(results) * 100
            final_results.avg_f1_score = sum(r.f1_score for r in results) / len(results) * 100
            final_results.avg_bleu_score = sum(r.bleu_score for r in results) / len(results) * 100
            final_results.avg_tokens_per_question = sum(r.tokens_used for r in results) / len(results)

            # Recall@K (Solution 6) — only for questions with evidence
            results_with_evidence = [r for r in results if r.evidence_ids]
            if results_with_evidence:
                final_results.recall_at_5 = sum(r.recall_at_5 for r in results_with_evidence) / len(results_with_evidence) * 100
                final_results.recall_at_10 = sum(r.recall_at_10 for r in results_with_evidence) / len(results_with_evidence) * 100
                final_results.recall_at_20 = sum(r.recall_at_20 for r in results_with_evidence) / len(results_with_evidence) * 100

            # Latency percentiles
            sorted_latencies = sorted(latencies)
            final_results.avg_retrieval_latency_ms = np.mean(latencies)
            final_results.p50_retrieval_latency_ms = np.percentile(latencies, 50)
            final_results.p95_retrieval_latency_ms = np.percentile(latencies, 95)
            final_results.p99_retrieval_latency_ms = np.percentile(latencies, 99)

        # Per-category metrics
        for cat in set(r.category for r in results):
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                cat_with_evidence = [r for r in cat_results if r.evidence_ids]
                cat_recall_5 = sum(r.recall_at_5 for r in cat_with_evidence) / len(cat_with_evidence) * 100 if cat_with_evidence else 0.0
                cat_recall_10 = sum(r.recall_at_10 for r in cat_with_evidence) / len(cat_with_evidence) * 100 if cat_with_evidence else 0.0
                cat_recall_20 = sum(r.recall_at_20 for r in cat_with_evidence) / len(cat_with_evidence) * 100 if cat_with_evidence else 0.0
                final_results.category_results[cat] = {
                    "count": len(cat_results),
                    "llm_judge_accuracy": sum(r.llm_judge_score for r in cat_results) / len(cat_results) * 100,
                    "avg_f1_score": sum(r.f1_score for r in cat_results) / len(cat_results) * 100,
                    "avg_bleu_score": sum(r.bleu_score for r in cat_results) / len(cat_results) * 100,
                    "recall_at_5": cat_recall_5,
                    "recall_at_10": cat_recall_10,
                    "recall_at_20": cat_recall_20,
                    "questions_with_evidence": len(cat_with_evidence),
                }

        # Print results
        print_results(final_results)

        # Save detailed results if requested
        if output_path:
            save_results(final_results, results, output_path)

        return final_results


def print_results(results: EvaluationResults):
    """Print formatted evaluation results"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Conversations:     {results.num_conversations}")
    print(f"  Documents:         {results.num_documents}")
    print(f"  Questions:         {results.total_questions}")

    print(f"\n{'='*70}")
    print("OVERALL METRICS (Mem0-compatible)")
    print("=" * 70)
    print(f"  LLM-Judge Accuracy:  {results.llm_judge_accuracy:.1f}%")
    print(f"  F1 Score:            {results.avg_f1_score:.1f}%")
    print(f"  BLEU-1 Score:        {results.avg_bleu_score:.1f}%")

    print(f"\n{'='*70}")
    print("RETRIEVAL RECALL (Solution 6)")
    print("=" * 70)
    print(f"  Recall@5:            {results.recall_at_5:.1f}%")
    print(f"  Recall@10:           {results.recall_at_10:.1f}%")
    print(f"  Recall@20:           {results.recall_at_20:.1f}%")
    if results.recall_at_10 > 0 and results.recall_at_20 > 0:
        if results.recall_at_10 < 50 and results.recall_at_20 > 70:
            print(f"  Diagnosis:           Ranking problem (right memories found but buried)")
        elif results.recall_at_20 < 50:
            print(f"  Diagnosis:           Recall problem (memories not found at all)")
        elif results.recall_at_10 > 70 and results.llm_judge_accuracy < 60:
            print(f"  Diagnosis:           Reasoning problem (outside MnemeFusion scope)")

    print(f"\n{'='*70}")
    print("PERFORMANCE")
    print("=" * 70)
    print(f"  Retrieval Latency:")
    print(f"    Average:           {results.avg_retrieval_latency_ms:.1f}ms")
    print(f"    P50:               {results.p50_retrieval_latency_ms:.1f}ms")
    print(f"    P95:               {results.p95_retrieval_latency_ms:.1f}ms")
    print(f"    P99:               {results.p99_retrieval_latency_ms:.1f}ms")
    print(f"  Token Consumption:")
    print(f"    Avg per question:  {results.avg_tokens_per_question:.0f}")
    print(f"    Total:             {results.total_tokens_used:,}")
    print(f"  Timing:")
    print(f"    Ingestion:         {results.total_ingestion_time_s:.1f}s")
    print(f"    Evaluation:        {results.total_evaluation_time_s:.1f}s")

    print(f"\n{'='*70}")
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 70)

    category_names = {
        1: "Single-hop (factual)",
        2: "Multi-hop (reasoning)",
        3: "Temporal (time-based)",
        4: "Open-domain (knowledge)",
        5: "Adversarial"
    }

    print(f"  {'Category':<25} {'Count':>6} {'Judge':>8} {'R@5':>7} {'R@10':>7} {'R@20':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for cat in sorted(results.category_results.keys()):
        cat_data = results.category_results[cat]
        cat_name = category_names.get(cat, f"Category {cat}")
        r5 = f"{cat_data.get('recall_at_5', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        r10 = f"{cat_data.get('recall_at_10', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        r20 = f"{cat_data.get('recall_at_20', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        print(f"  {cat_name:<25} {cat_data['count']:>6} {cat_data['llm_judge_accuracy']:>7.1f}% {r5:>7} {r10:>7} {r20:>7}")

    print(f"\n{'='*70}")
    print("COMPARISON WITH COMPETITORS")
    print("=" * 70)
    print(f"  {'System':<20} {'LLM-Judge':>12} {'Latency (p95)':>15}")
    print(f"  {'-'*20} {'-'*12} {'-'*15}")
    print(f"  {'MnemeFusion':<20} {results.llm_judge_accuracy:>11.1f}% {results.p95_retrieval_latency_ms:>12.1f}ms")
    print(f"  {'Mem0 (reported)':<20} {'66.9%':>12} {'1,440ms':>15}")
    print(f"  {'OpenAI Memory':<20} {'52.9%':>12} {'N/A':>15}")
    print(f"  {'Hindsight':<20} {'89.6%':>12} {'N/A':>15}")
    print("=" * 70)


def save_results(final_results: EvaluationResults, question_results: List[QuestionResult], output_path: str):
    """Save detailed results to JSON"""
    output = {
        "summary": asdict(final_results),
        "questions": [asdict(r) for r in question_results]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Detailed results saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Industry-standard evaluation for MnemeFusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation
    python industry_eval.py

    # Quick test (50 questions)
    python industry_eval.py --max-questions 50

    # With SLM metadata extraction
    python industry_eval.py --use-slm --slm-model opt/models/qwen3-0.6b.gguf

    # Specific categories only
    python industry_eval.py --categories 1 2
        """
    )

    parser.add_argument(
        "--dataset",
        default="tests/benchmarks/fixtures/locomo10.json",
        help="Path to LoCoMo dataset (default: tests/benchmarks/fixtures/locomo10.json)"
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=None,
        help="Number of conversations to use (default: all)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions to evaluate (default: all)"
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=None,
        help="Categories to evaluate (default: 1 2 3 4)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of memories to retrieve for answer generation (default: 25, includes synthetic profile facts)"
    )
    parser.add_argument(
        "--use-slm",
        action="store_true",
        help="Enable SLM metadata extraction (Python subprocess)"
    )
    parser.add_argument(
        "--slm-model",
        default=None,
        help="Path to SLM model file (.gguf)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable native LLM entity extraction (Qwen3 via llama-cpp-2)"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Path to GGUF model file (e.g., models/qwen3-4b/Qwen3-4B-Instruct-2507.Q4_K_M.gguf)"
    )
    parser.add_argument(
        "--llm-tier",
        default="balanced",
        choices=["balanced", "quality"],
        help="LLM model tier: balanced (4B) or quality (8B)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save detailed results JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Persistent DB path. DB survives after run, enabling --skip-ingestion next time. "
             "Example: tests/benchmarks/fixtures/eval_baseline.mfdb"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion if --db-path DB already exists. For fast re-evaluation with same data."
    )

    args = parser.parse_args()
    print(f"[DEBUG] Parsed args: use_slm={args.use_slm}, slm_model={args.slm_model}, use_llm={args.use_llm}, llm_model={args.llm_model}")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Run evaluation
    run_evaluation(
        dataset_path=args.dataset,
        num_conversations=args.num_conversations,
        max_questions=args.max_questions,
        categories=args.categories,
        top_k=args.top_k,
        use_slm=args.use_slm,
        slm_model_path=args.slm_model,
        use_llm=args.use_llm,
        llm_model_path=args.llm_model,
        llm_tier=args.llm_tier,
        output_path=args.output,
        verbose=args.verbose,
        db_path=args.db_path,
        skip_ingestion=args.skip_ingestion
    )


if __name__ == "__main__":
    main()
